# Toolchain architecture

Python 3.10+, standard library only. No third-party packages, so the toolchain
runs anywhere Python does and cannot rot when a dependency changes.

## Why Python for tools, C++ for sources

**Tools in Python.** The analysis passes are iterative and exploratory - you
rerun them constantly while refining heuristics. Fast iteration matters more
than runtime speed, and the whole pipeline over a 4 MB image finishes in about
five seconds.

**Sources in C++17.** The game is C++, and the RenderWare 3.6 layer beneath it
is C. C++ is the only choice that reproduces the original structure honestly,
and it compiles on a modern host for the eventual port. C would force the class
hierarchies to be faked; anything higher-level could not match.

## Module layout

The engine knows nothing about any one game. A game is a directory under `games/`
with a `game.json` (see the repository README), and every tool takes `--game`
(defaulting to the only game present, or `$PS2_GAME`).

```
engine/ps2decomp/
  game.py      a game's game.json: its files, its layout, $gp detection
  layout.py    the memory layout the interpreter and tests use (per game)
  iso9660.py   ISO9660 reader: directory tree, random access, extraction
  elf.py       32-bit little-endian ELF reader
  r5900.py     Emotion Engine disassembler
  vu.py        Vector Unit microcode disassembler
  analysis.py  function discovery and code/data classification
  symbols.py   strings, address cross-references, pointer tables
  naming.py    names from evidence: kernel stubs, diagnostic strings
  signatures.py relocation-masked function signatures, cross-game library
  emu.py       R5900 interpreter, for verification
  symexec.py   symbolic execution of branch-free functions
  paths.py     control flow, frames, memory, inlining, loops
  calls.py     calling convention: live parameters, preserved registers
  testmem.py   memory profiles for test inputs
  structs.py   struct type recovery (type unification over the whole binary)
  typed.py     renders derived code against recovered types
  cppcheck.py  compiles emitted C++ and runs it against the interpreter
  vu0.py       VU0 macro mode (COP2): the arithmetic model, shared with lifted code
  lift.py      lifts machine code to C++, one statement per instruction
  structure.py the lifted statements with structured control flow and named locals
  liftcheck.py compiles lifted C++ and runs it against the interpreter
  difftest.py  differential-test harness

engine/tools/
  isotool.py    disc inspection and extraction
  analyze.py    full static-analysis pipeline
  disasm.py     annotated assembly emitter
  vudis.py      VU microcode disassembler driver
  vuhazards.py  where VU microcode reads Q, P or the flags early, or stalls
  imagemap.py   byte-level accounting of the image
  candidates.py ranks functions by reconstructability
  derive.py     derives, verifies and emits accessor functions
  lift.py       lifts every function, verifies each, emits the verified ones
  refusals.py   why the functions that were not derived were refused
  blockers.py   which instructions keep functions out of the interpreter
  loops.py      loop shapes, for planning loop support
  inspect_function.py  one function through derive, emit, compile and test
  structtypes.py reports on the recovered struct types
  fields.py     measures field-offset consistency (the groundwork for structs.py)
  matchdiff.py  byte-comparison, kept for if the compiler is ever obtained
  signatures.py names functions, exports them to the library, compares binaries
  progress.py   progress accounting

engine/data/ee_syscalls.csv   EE kernel service numbers -> names
engine/signatures/library.json  named signatures from every game so far

engine/runtime/
  include/ps2rt/  types.h and generated.h, which generated code compiles against,
                  and lifted.h, the register context lifted code runs on
  src/eefpu.cpp   reference EE FPU arithmetic (no IEEE infinities or NaNs)
  src/lifted_support.cpp  multiply/divide, MMI lanes, unaligned access for lifted code
  src/vu0.cpp     VU0 macro mode for lifted code, mirroring vu0.py

engine/host/      the PC port's platform layer (engine/docs/PORTING.md): the
                  kernel, threads and interrupts, DMA, VIF, the VUs in micro
                  mode, the software GS, the IOP over SIF, SPU2, input, the
                  window, timers and the clock (--virtual-time), and
                  --gpu-shadow, which checks the GPU renderer against the
                  software GS frame by frame
engine/gpu/       ps2gpu: SDL3 and Vulkan - the window's presentation, and the
                  GS on the GPU (engine/docs/GPU.md), with its own tests
```

Two more tools watch and read what the port runs: `engine/tools/watchdog.py`
alerts when a long job's log goes silent or its memory grows, and
`engine/tools/vuhazards.py` reads a dump of VU instruction memory (written
beside `--frames` pictures) for the pipeline timing the microcode depends on.

Game-specific code lives with its game: for Burnout 3,
`games/burnout3/tools/vdb.py` and `vdbhash.py` read the ValueDB tuning database.

Each layer depends only on the ones above it, and none of them depend on the
CLI drivers.

## `emu.py` - the R5900 interpreter

This is what makes verification possible without the original compiler. It
executes the integer core, common loads and stores, 128-bit `lq`/`sq`, a COP1
single-precision subset, and the MMI operations the string routines need - enough
for 84.8% of discovered functions. `$gp` is seeded with the game's global
pointer (`0x004E8670` in Burnout 3), detected
from the startup code, because a great many functions reach globals through it.

Anything outside the subset raises `UnsupportedInstruction`. That refusal is the
design: an interpreter that quietly approximated an instruction would hand back
confident wrong answers, which is worse than a gap.

Two correctness details that cost real debugging:

* **Branch-likely nullification.** `beql`/`bnel` skip their delay slot when *not*
  taken. Executing it unconditionally corrupted a mask register inside the SIMD
  `strlen` and made it return wrong lengths for every input.
* **Quadword halves.** `lq`/`sq` must preserve the upper 64 bits of a register,
  because Metrowerks spills callee-saved registers with them. Dropping those bits
  corrupts every restore.

## Deriving functions automatically

Reading and reconstructing one function at a time does not scale. Three modules
turn the mechanical cases into a batch process:

```
symexec.py   expression model + straight-line symbolic execution
paths.py     acyclic control flow, the stack frame, memory, call inlining
derive.py    differential verification and C++ emission
```

### `symexec.py` - the expression model

Registers hold *expressions* instead of values. The tree does two jobs: it renders
as C for a human, and it evaluates against concrete inputs so the derivation can be
tested. Both come from the same tree deliberately - a derivation that cannot be
tested is worth very little.

Floats are modelled as 32-bit **patterns**, not as a separate value domain. That
mirrors the hardware: a float register holds 32 bits, and `lwc1`/`swc1`/`mtc1`/
`mfc1` move them without interpreting anything. Interpretation is confined to the
arithmetic nodes, which is what makes bit moves exact.

Two traversal helpers, `children` and `rebuild`, are the basis of every walk over a
tree, and they **raise** on an unrecognised node type. That is not pedantry: the
earlier hand-written versions silently returned nothing for node types added later,
so arguments nested inside `Not`, `Select` and `SignExtend32` became invisible and
the emitter produced functions referencing undeclared parameters.

### `paths.py` - control flow, frames, memory

Most small functions have a conditional. Those are derived exactly by enumerating
every route from entry to a return and recording the branch conditions along each,
so the function becomes an ordered list of guarded results.

* **Loops and underived calls are refused.** A loop's result depends on a trip count
  that is not a function of the arguments, so no finite expression describes it.
* **The stack frame is modelled separately** from real memory, so a prologue's
  register spills and the matching epilogue reloads cancel instead of looking like
  writes a caller could observe.
* **Stores feed an overlay that later loads consult.** Every derived expression
  refers to memory as it stood on entry, so a setter that returns the field it just
  wrote would otherwise be derived against stale contents. Where aliasing cannot be
  ruled out, the function is refused rather than guessed at.
* **Constant conditions are folded.** A flag stored and immediately loaded back
  yields a constant-true branch; exploring both sides anyway manufactures a path
  that can never run, and `select` picks the first path whose conditions hold, so an
  infeasible path could be chosen.
* **Derived callees are inlined**, which is why derivation runs to a fixed point -
  each pass unlocks the callers of what the previous pass proved. Only *verified*
  callees are offered, so one bad derivation cannot propagate upward.

### `derive.py` - verification and emission

Each derivation is differential-tested against the original machine code before any
source is written, and only the ones that agreed are emitted. The emitted C++ is then
compiled (`cppcheck.py`) and run on the same cases, because what is verified is a
tree and what ships is its rendering; a function whose compiled form disagrees is
dropped.

Memory effects are compared, not just return values. Writes are **journalled and
rolled back** rather than restored from a snapshot of a chosen region: these
functions write to globals far outside any such region, and leaving those changes in
place both corrupted later cases and made correct derivations look wrong. Writes
into the callee's own frame are excluded from the comparison, since the frame is
dead on return and the original makes many such writes while a derived expression
makes none.

A function only counts as `verified` when the test actually varied something. A
constant or a fixed-global read ignores its arguments, so eight argument sets
exercise one input eight times; those are recorded a tier lower rather than
inflating the figure.

### Traps worth knowing about

Every one of these produced plausible-looking wrong output, and none were caught by
reading the code:

* **Operator precedence.** A tree evaluates from its structure, so a rendering bug
  leaves the arithmetic right while the printed C means something else:
  `arg1 << 2 + arg0` is `arg1 << (2 + arg0)` in C.
* **A duplicated renderer.** The emitter kept its own copy of the precedence logic
  so it could print `Read*` helpers, and the two drifted. There is now one renderer
  with the Load form as its only hook.
* **Store ordering.** Expressions refer to entry-state memory, so emitting stores
  before the return inverts a read-modify-write.
* **Shift widths.** Word shifts mask the amount to 5 bits and doubleword shifts to
  6. One mask for both turns a shift of 32 into a shift of zero.
* **`lui` sign extension.** The result is sign-extended into the 64-bit register, so
  `lui 0xFFFF` then `dsrl32 32` is `0xFFFFFFFF`, not 0.
* **Float bit moves.** Holding float registers as host doubles canonicalises NaN
  payloads, altering values a copy must preserve.

## Lifting: the complete tier

Derivation enumerates paths and produces readable expressions, so it cannot
finish large functions: half the executable's code is in functions of 96
instructions or more, where it verified under 8%. Lifting is the mechanical
complement. Each instruction becomes one C++ statement over `lifted::Context`
(all GPRs with their upper 64 bits, HI/LO for both pipelines, the FPU registers,
accumulator and condition). Branches become gotos, and a delay slot is emitted
inside its branch's block, or twice when a jump also lands on it. Calls pass the
whole context. The interpreter is the specification, and each handler has a
lifted counterpart that `test_lift.py` checks against it, family by family.

Verification uses the derivation's test plans and memory, but compares more.
After each case the *entire* register state and every byte written must match
the original. Callees run in the interpreter with the context handed over and
back (`Machine.run_to_return`), so each function is checked on its own. Because
the whole state is compared, the lifter also finds interpreter bugs that
`$v0`-plus-memory comparison let through. `pcpyld $v1, $v1, $v0` was one: the
interpreter wrote the low half before reading the source for the high half.

## Calling functions that are not derived

A derived function no longer has to wait for its callees. A call into a function
that has not been derived is kept as a call (`paths.Call`), passing the arguments
the callee reads (`calls.parameters`, integer and float). Its `$v0` becomes a
`CallResult`, and memory after it is a new *epoch*: `Load` carries the epoch it
reads, so a load after the call is a different value from the same load before.

Such a derivation is no longer a set of guarded expressions over entry memory -
a branch after a call depends on what the call did - so its paths are merged into
a decision tree (`PathDerivation.tree`). Verification runs that tree
(`paths.Execution`): stores within an epoch are held until the next call, the
*original* callee runs in the interpreter, and a load from an earlier epoch reads
memory with the journalled writes since then undone. The emitter
(`derive.StatementEmitter`) writes statements in order, capturing into a local any
value a later statement needs that a store or a call could change. Callees that
were not derived are declared; the compiled check links them back to the
interpreter, so the compiled code is tested against original callees too.

Only call-free derivations are inlined. A derived function that calls an
underived one is called by its own callers, by its emitted signature.

A *barrier* (`paths.Barrier`) is an epoch boundary with no call: where a load may
or may not overlap an earlier store, the held stores are applied and the load
reads the new epoch. The walk's epoch lives on its store overlay (`paths.Overlay`)
because a barrier arises deep inside an instruction step. Emitted code needs
nothing for it - its statements already run in order against live memory.

## `structs.py` and `typed.py` - struct types

Types are recovered by unification in the style of Steensgaard's points-to
analysis, with field offsets. A *type node* stands for every value that is
dereferenced: function arguments, return values, and the contents of pointer
fields. Nodes collect the offsets and widths they are accessed at. Evidence that
two values are the same object joins their nodes: an argument passed into a call,
a returned value, a pointer stored into a field. A displaced pointer passed into a
call (`base + 0x1E4`) records an *embedded* type instead.

What keeps this from collapsing into one type:

* **A conflict guard.** Two nodes join only if every offset both use has the same
  set of widths, checked recursively through pointer fields and embedded types.
  Generic routines - memcpy and the like - are refused at the first type they
  disagree with. A subset rule was tried first and let a node that already mixed
  widths absorb everything.
* **Specific links first.** Links are applied in order of how many callers the
  callee has, so generic routines meet established types and are refused rather
  than fusing types before their own evidence arrives.
* **Only arguments the callee reads.** A register that merely still holds a value
  at a call is not an argument. Linking those was half of all links, and joined
  the arguments of unrelated functions.
* **A flow-aware scan.** Register meanings are carried along paths and intersected
  where paths join; loop heads drop whatever the loop overwrites; code after a
  return is unreachable until a branch reaches it.

The guard compares widths, so width agreement is no evidence of quality. The
independent checks (`engine/tools/structtypes.py`) are float-versus-integer use and
signedness at offsets shared by several functions; without the guard both drop
and one type absorbs 1,422 functions.

Global memory is one more node, `globals`, standing for the whole address space. A
register holding a constant address - from `lui`, or the fixed `$gp` - holds
`(globals, address)`, so global variables are fields of that node at absolute
addresses, and a global whose address is passed to a function becomes an embedded
*global object*. For globals the emitter prefers the object form, so an object's
first field does not read as a separate scalar.

The emitter (`typed.py`) renders a typed value as a *view*: a thin object over a
guest address with an accessor per field, a pointer field returning the view of
its target. A load or store becomes a field access only when it matches the field
exactly - offset, width and signedness - so a wrong type can make the source less
tidy but cannot change its meaning, and the compiled check runs it regardless.
Global variables become `g_004876E8()` / `set_g_004876E8(v)` and global objects
constant views such as `g_0049A278_obj.f_8()`.

Arrays are fields with a stride. An address in the linear form base + offset +
stride * index (`typed.linear`, exact through shifts, multiplies and sums) renders
as `arg0.f_10(i)`; a constant offset on a detected run of fields renders as the
element it is, `arg0.set_f_40(3, 0)`. The derived accesses supply the stride
evidence, since the scan cannot follow a variable index.

`derive.reroll` then folds unrolled loops back up: a block of element writes that
repeats with every index advanced by one common step, and nothing else different,
becomes a `for` loop. It works on the emitted statements, which are generated in a
fixed shape, and only ever on writes of constants or already-captured values, so
the order and content of memory writes are unchanged.
Before emitting, the derived functions' own accesses are added to the graph as
evidence: the scan cannot follow a variable index, but an unrolled loop shows
every concrete access it makes, each already verified.

## `r5900.py` - the disassembler

A stock MIPS disassembler mis-decodes a real PS2 binary, because the R5900 is
MIPS III plus Sony/Toshiba extensions. The ones that matter:

* **MMI** (opcode `0x1C`) - 128-bit integer SIMD, with four nested sub-tables
  (MMI0-MMI3) selected by the shift-amount field, plus `PMFHL`/`PMTHL` and the
  parallel shifts. The entry point alone is 31 `padduw` instructions, so getting
  this wrong derails decoding immediately.
* **`LQ`/`SQ`** (opcodes `0x1E`/`0x1F`) - 128-bit load/store.
* **COP2** - on the EE this is the VU0 macro-mode interface, not a conventional
  coprocessor.
* **MIPS III 64-bit arithmetic**, fully implemented on the EE.

`decode(word, addr)` is a pure function, so analysis passes can decode the same
address repeatedly without bookkeeping. Each `Instruction` carries control-flow
flags (branch, call, return, delay slot, likely-branch) that the later passes
run on.

Correctness is covered by 40 unit tests in `engine/tests/test_r5900.py`, hand-assembled
from the instruction-set documentation so the expected values are independent of
the decoder. Writing them caught a real bug: COP1 packs its register fields
differently from the integer unit (`fd` in the shift-amount field, `fs` in `rd`),
which a shared operand formatter had got wrong.

## `analysis.py` - function discovery

The binary has no symbols and no section boundaries, so everything is recovered.
The passes run in order of decreasing confidence, which keeps false positives
down:

1. **Call-target harvesting.** Every `jal` names a function entry directly. The
   compiler only emits `jal` for real call sites, so a `jal` landing inside the
   image is near-proof. 40,430 call sites resolve to 5,677 accepted targets.
2. **Recursive traversal.** From the entry point and every harvested target, walk
   the control-flow graph, following branches and claiming delay slots, stopping
   at returns. This marks bytes *provably* reachable as code, rather than bytes
   that merely happen to decode.
3. **Pointer-table targets.** Virtual calls and callbacks emit no `jal` at all, so
   their targets are invisible to step 1. Every aligned word pointing inside the
   image is considered, then filtered hard: the target must land on a stack
   prologue or just past another function's terminator, decode as a run of real
   instructions, and **end in `jr $ra`**. That last test is what separates code
   from data which merely decodes - a virtual method has to return. 2,597 accepted,
   37 rejected for never returning.
4. **Prologue scanning.** A final `addiu $sp, $sp, -N` sweep catches the remainder:
   888 more.
5. **Extent from contiguous code.** A function's end is the unbroken run of its own
   instructions from its entry, bridging only short all-nop alignment padding.
   Taking the furthest address *reached* instead lets one tail jump swallow
   everything in between - which it did, until it was fixed.

Each function records *why* it was classified, so results stay auditable rather
than being an opaque guess.

Current result: **9,163 functions**, 86.2% of the image proven to be MIPS code,
99.67% of functions ending in a return, and 0 of 814,598 instructions failing to
decode. The coverage figure is a **lower bound** - it counts only bytes proven
reachable, and a further 2.65% of the image is VU microcode rather than MIPS.

Two false-positive traps were worth closing:

* A zero word decodes as a perfectly valid `nop`, so any test that only asks "does
  this decode?" accepts a block of padding as a function. `_looks_like_code` also
  requires real work - at least two non-`nop` instructions and a majority of
  non-padding.
* Seventeen "functions" were sitting inside the read-only data block, decoding
  tables of small integers as instructions on `$zero`, and one pointer array that
  pointed at itself. The return requirement removed all of them.

## `symbols.py` - naming leverage

Three reliable kinds of leaked information:

* **String cross-references.** MIPS materialises a 32-bit address with a
  `lui`/`addiu` pair, or `lui` plus a load offset. Reassembling those recovers
  the constant; when it lands in a string, the call site gains a strong
  human-readable hint. Register tracking resets at any control transfer, so a
  `lui` whose partner sits past a branch is dropped rather than mispaired.
* **RenderWare provenance.** RCS `$Id:` tags name the exact middleware sources
  linked in, identifying which regions are documented-API middleware rather than
  game code.
* **Pointer tables.** Runs of consecutive words pointing at known function
  entries are vtables or jump tables, which reveal C++ class layout. Requiring
  several consecutive valid entries keeps random data from registering.

Everything here generates *hints*. Nothing renames a function without a recorded
reason.

## Data flow

Paths are relative to the game's directory, as `game.json` names them.

```
disc image
   |  isotool.py
extracted/<executable>
   |  analyze.py   (elf -> analysis -> symbols)
build/<database>.json      functions, regions, strings, xrefs, tables
   |                     \
   |  derive.py           \  progress.py  + config/functions.csv
src/generated/              docs/PROGRESS.md
```

The database is generated once and everything downstream reads it, so the
expensive passes are not repeated. It is not committed - it is reproducible from
a disc image in about five seconds.
