# Methodology

## Progress tiers

Every discovered function sits in exactly one tier, recorded in the game's
`config/functions.csv`.

### `undecompiled`

The analysis found the function and knows its address range. Nothing else is
known. Every function starts here.

### `analyzed`

Someone has read the disassembly and written down what the function does - its
argument count, the structure offsets it touches, its role. No source yet. This
captures understanding, which is often the hard part, but it is not code.

### `reconstructed`

C++ source exists and is believed to behave correctly. It has **not** been tested
against the original. Source at this tier is a hypothesis.

### `lifted`

The function has been translated instruction by instruction into C++ over an
explicit register context (`engine/ps2decomp/lift.py`), compiled, and run on the
same test cases against the original. After every case the **whole register
file** and every byte of memory written agreed. That is a stricter comparison than
for derived functions, which compare `$v0` and memory.

Lifted code is emitted in the most readable form that passes. That is structured
(loops and ifs, named locals) where possible, and in the best case with
single-use temporaries folded into their uses. The folded form drops writes to
registers no caller may read, so it is compared at the calling-convention level:
`v0`/`v1`, `$sp`, `$gp`, `$ra`, the callee-saved registers whole, the float
results and callee-saved float registers, and memory. That is the level derived
functions are verified at. Every other form is compared on the whole register
file.

A structured form must pass a second check as well: it is compared with the
function's plain lifted form - verified against the interpreter, and a statement
per instruction - natively, on 2,000 random inputs (`engine/ps2decomp/
equivalence.py`). Both run on the same registers and the same hashed memory; a
call reaches a stub whose effect depends on its target, its place in the
sequence and its arguments; what is compared is what a caller can see. Two
structuring bugs passed the interpreter's random tests because the paths they
broke were never taken, and were found only when a game ran natively; this
check runs a hundred times more cases per function. A form that fails it falls
back to the next, as for a failed verification.

Verdicts are remembered between runs (`engine/ps2decomp/verdicts.py`). A
function's verdict depends only on its own C++ text and the environment it is
tested in - its callees run in the interpreter from the original code - so the
key is the text, the form of comparison, and a fingerprint of the interpreter,
test memory, runtime, verification code, image, analysis and seeds. A verdict
is reused only when all of these are unchanged. A `--no-cache` run recomputes
every verdict and is the check on the two things the key cannot see: the
compiler's version, and the order functions run in within a process.

Lifted source is correct and complete, and enough to run the game on a host, but
it is not readable: it says `c.r[2] = c.r[4] + 5`, not what the function means.
So it is reported as its own tier and never counted as `verified`. A function
that is later derived moves up to `verified`, and its lifted copy is only the
fallback.

### `verified`

The reconstruction has been differential-tested against the original machine code
running in the R5900 interpreter, on a corpus that includes edge cases, and they
agreed. This is the project's bar for done.

**Only `verified` counts as decompiled.** The headline figure in
[PROGRESS.md](../../games/burnout3/docs/PROGRESS.md) reports verified bytes and
nothing else. A second figure, "machine-checked", adds the `lifted` tier: how
much of the program exists as C++ proven to behave like the original, readable or
not.

## Why functional reconstruction, and what it does and does not prove

The original compiler is known - Metrowerks CodeWarrior MIPS 2.4.1.01, named in
the executable's `.comment` section - but it is abandoned proprietary software and
could not be obtained. Without it, byte-matching is impossible: Metrowerks and GCC
differ systematically in register allocation, instruction scheduling and
delay-slot filling, so a GCC attempt would plateau short of a match and never
close.

So the project targets behaviour instead of bytes, and replaces byte-matching with
differential testing:

1. Run the **original machine code** in `engine/ps2decomp/emu.py`, an R5900 interpreter.
2. Run the reconstruction on the same inputs.
3. Compare.

Be precise about the strength of this. A passing differential test shows the two
agree **on the inputs tested**. That is evidence, not proof - a function whose
behaviour depends on an input the corpus never supplies could still differ. The
harness therefore records how many cases ran, puts edge cases ahead of random
ones, and always reports the case count rather than just "passed".

It is a weaker guarantee than byte-matching, and the right response is to say so
rather than to blur the two. What it does give is an objective, automated check
that fails loudly on a regression - which untested source does not.

## Verifying a reconstruction

`engine/ps2decomp/difftest.py` provides the harness for hand-written reconstructions, and
`games/burnout3/tests/test_reconstruction.py` holds those comparisons, so
`python -m unittest discover -s games/burnout3/tests` re-checks every verified function.

`engine/tools/derive.py` does the same job for derived functions, in bulk.

The interpreter covers the integer core, the common loads and stores, 128-bit
`lq`/`sq`, COP1 single precision, and the MMI operations the string routines need -
enough to execute **84.8%** of the 9,163 discovered functions. `$gp` is seeded with
`0x004E8670`, recovered from the startup code, since many functions reach globals
through it.

Anything outside the subset raises `UnsupportedInstruction` rather than being
approximated: an interpreter that quietly guessed would manufacture exactly the
false confidence this exists to remove. The remaining gap is almost entirely VU0
macro mode.

### What is actually compared

Return values are not sufficient. The harness compares **memory effects** too,
because a derivation with the right return value and the wrong store address would
otherwise pass.

Writes are journalled and rolled back rather than restored from a snapshot of a
chosen region. These functions write to globals far outside any such region, and an
earlier version that snapshotted only a scratch area left those changes in place -
which corrupted later cases and made correct derivations look wrong. It could
equally have produced false passes.

Writes into the callee's own stack frame are excluded. The frame is dead the moment
the function returns, and the original spills registers into it constantly while a
derived expression writes none, so counting them would fail every function with a
prologue.

### Automated derivation

Roughly a third of verified functions were not written by hand at all. Small
branch-free and acyclic functions are derived symbolically, differential-tested, and
only then emitted as C++ - see [ARCHITECTURE.md](ARCHITECTURE.md). A derivation that
fails or cannot be executed is dropped rather than emitted with a caveat.

Two honesty rules apply to that output:

* A function counts as `verified` only if the test varied something. A constant or a
  fixed-global read ignores its arguments, so eight argument sets exercise one input
  eight times; those land a tier lower.
* Generated functions keep placeholder `func_XXXXXXXX` names, and recovered types
  placeholder `S_XXXXXXXX_aN` names. The behaviour and the layouts are evidence;
  what a function or type is *for* has not been worked out, and a descriptive name
  invented now would read as a finding later.
* Recovered types are presentation, never trusted for meaning. A memory access is
  shown as a field only when it matches the field exactly, and the compiled form of
  every typed function is run against the original.

### The emitted source is compiled and run too

Verifying a derivation means evaluating its expression tree. What ships is the C++
that tree is rendered to, and for a long time nothing checked that the two agree.
They did not. The renderer printed signed and unsigned comparisons identically, and
C++ promotion rules quietly narrowed 64-bit register arithmetic: `ReadU8(p) << 24`
was an `int` shift, and 419 shifts in the file were undefined behaviour. It all
compiled without a single error.

So `derive.py --emit` now compiles the generated file with MSVC (`/O2`, so the
optimiser gets its chance to exploit undefined behaviour) into a library with a
dispatch table, and runs every function on the same cases its derivation was
verified on, with memory routed back to the interpreter
(`engine/ps2decomp/cppcheck.py`). A function whose compiled form disagrees is dropped
and the file re-emitted. Where no compiler is installed the check says it was
skipped; it never passes by default.

The rules the rendering follows to make this hold:

* Every value is a `u64` register image. Arguments are `u64`, and the `Read*`
  helpers return the loaded value already sign- or zero-extended.
* Operators whose meaning depends on width or signedness carry it explicitly:
  `(s64)` on signed comparisons, `(u64)((s32)x >> n)` for `sra`,
  `(x & 0xFFFFFFFF) >> n` for `srl`, masked variable shift counts, `1ull` for a
  shifted literal.
* Float constants print as exact literals: `FloatBits(-1.0f)`, not `0xBF800000`.
* The reference EE FPU helpers live in `engine/runtime/src/eefpu.cpp`.

### Bugs this process caught

Worth listing, because each argues for the approach. None were visible to
inspection, and several were in the *verifier* rather than the reconstruction:

* `beql`/`bnel` nullify their delay slot when not taken; executing it unconditionally
  clobbered a mask register in the SIMD `strlen`.
* Word and doubleword shifts share one mask in the expression model, and the 5-bit
  one turns a shift of 32 into a shift of zero.
* `lui` was not sign-extended into the 64-bit register.
* The verification harness restored only a scratch region, so writes to globals
  persisted across cases.
* Journalling then exposed that stack-frame writes must be excluded.
* Float registers held as host doubles canonicalised NaN payloads, so a `lwc1`/`swc1`
  copy altered the value. Here the derivation was right and the interpreter wrong.
* An overflowing `mul.s` crashed the interpreter outright, because the EE's FPU is
  not IEEE-754: it has no infinities and saturates to the largest representable
  magnitude. A port must reproduce that rather than "fix" it.
* `simplify` turned an unsigned `sltu` tested against zero into a signed comparison.
  The tests passed only small positive pointers, so both readings agreed.
* Branching void functions were emitted without a `return;` at the end of each
  early path, so it fell through into the later paths' stores. Their multi-store
  paths were also not captured before writing. Compiling and running the output
  found 12 such functions, all of which had passed as trees.
* Word operations (`addu`, `addiu`, `sll`, `sllv`, ...) sign-extend their 32-bit
  result into the 64-bit register; the model kept the full 64-bit value. That is
  invisible unless upper bits are observed, and with only small positive pointers
  as arguments they never were. Passing $t0..$t3 arguments exposed it: an `sllv`
  result reached an `slt`. Word results are now wrapped in a sign extension that
  the simplifier removes wherever only the low 32 bits are used or the value
  provably fits. The first version of that simplification was exponential on the
  shared subtrees of unrolled loops; it is memoised on the (immutable) nodes.
* The first statement emitter for calls captured a value *after* the store that
  changed it, and let a local declared in one branch be used in the other. The
  compiler caught the second; the compiled run would have caught the first.
* Interpreting the original callees made verification slow - decoding every
  instruction from scratch dominated. Decodes are now cached per address, and a
  verification case may run at most 100,000 instructions.
* Declaring float arguments `u32` instead of `u64` made `farg0 & 0x7FFFFFFF` plus
  -127 wrap at 32 bits. The compiled check caught both affected functions.

## The original toolchain, for the record

The executable's `.comment` section names its compiler outright:

```
MW MIPS C Compiler (2.4.1.01)   PlayStation2
```

Metrowerks CodeWarrior MIPS 2.4.1.01 for PlayStation 2. Normally identifying this
is the opening problem of a decompilation; here the build stamped its own identity
into the binary.

It is nonetheless abandoned proprietary software - Metrowerks went to Freescale and
then NXP, and the PS2 product has not been sold in about two decades - which is
why this project verifies behaviour instead of bytes.

If `mwccmips` 2.4.1.01 is ever obtained, byte-matching becomes available as a
*stronger* check on top of the existing one, and `engine/tools/matchdiff.py` is kept for
exactly that: it compares a compiled object against the original image and scores
it both exactly and semantically.

```sh
python engine/tools/matchdiff.py 0x0010BE40 --object games/burnout3/build/syscalls.o --symbol SignalSema
```

The kernel stubs in `games/burnout3/src/kernel/syscalls.cpp` are the natural calibration targets,
being short enough that their source is unambiguous.

## Conventions for reconstructed source

* C++17, since the original is C++ (with RenderWare 3.6 underneath in C).
* Fixed-width types from `engine/runtime/include/ps2rt/types.h`.
* Every reconstructed function carries a comment naming the address it came
  from and its current tier.
* Names that are recovered from evidence (a string, an RTTI entry, a RenderWare
  API signature) are used as-is. Names that are inferred get a comment saying so.
  Never silently invent a name that looks authoritative.
* Structure layouts live in headers, with offsets documented, because they are
  the reusable product of the work.
