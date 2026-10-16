---
name: ps2-decomp
description: Decompile a PlayStation 2 game into verified, readable C++ with this repository's engine - setting up a new game, running the derive/verify pipeline, triaging why functions are refused, extending the engine, and reporting progress honestly. Use when working on any game under games/, adding a game, or changing engine/.
---

# PS2 decompilation by functional reconstruction

This repository turns a PS2 executable into C++ whose behaviour has been checked
against the original machine code. There is no original compiler, so nothing is
byte-matched. Instead every function goes through the same check:

1. The original code runs in an R5900 interpreter (`engine/ps2decomp/emu.py`).
2. A symbolic derivation of the same function runs on the same inputs.
3. The C++ emitted from that derivation is compiled and run on them too.

A function counts as **verified** only when all three agree on the return value
and on every byte of memory written.

There is a second, mechanical tier. `engine/tools/lift.py` translates every
function instruction by instruction into C++ over a register context and verifies
it the same way, comparing the whole register file. With `--structure` it emits
the verified form as structured C (`structure.py`: loops and ifs, named locals,
callee-saved saves elided), itself verified the same way. Lifted code is complete
and correct but not as readable as derived code, so it is the `lifted` tier,
reported apart; functions that lift but cannot be tested are emitted marked
NOT VERIFIED and recorded `reconstructed`. Derivation turns lifted functions into
readable ones over time.

The progress report gives two numbers. Report both, and never blur them:
`VERIFIED` (readable, derived source) and `MACHINE-CHECKED` (verified or lifted). Read `engine/docs/METHODOLOGY.md` before
changing what "verified" means, and `engine/docs/ARCHITECTURE.md` before changing
the engine.

Running the result as a PC port - bring-up, accuracy, performance, PC
features, working autonomously - is skill `ps2-port`; what may be published
and how a release is made is skill `ps2-release`.

## Non-negotiable rules

- **Only verified code counts.** Never report a derived-but-untested or
  reconstructed-but-untested function as progress. The headline number is
  verified code bytes from `engine/tools/progress.py`, never a function count.
- **Refuse rather than approximate.** When the interpreter or the deriver meets
  something it does not model, it raises (`UnsupportedInstruction`,
  `Unsupported`). Keep it that way. A quiet approximation hands back confident
  wrong answers, and the tests cannot catch what they never exercise.
- **Every engine change is a regression risk for every function.** After
  changing `engine/`, run both test suites and the full pipeline. Compare the
  three numbers (derived, compiled agree, verified %) with the last commit. A
  drop is a bug even if the new feature works.
- **Never commit disc material.** The ISO, `extracted/`, `build/` and `asm/` are
  gitignored. Check `git diff --cached --name-only` before every commit.
- **The engine stays game-independent.** Facts about one game go in its
  `game.json` or its own directory, never as constants in `engine/`.

## Repository map

The engine is its own repository (`ps2-port-engine`); each game is a
repository of its own (`<game>-pc`, e.g. `burnout3-pc`) with the engine as a
git submodule at `engine/`, pinned to an engine tag. Paths here are from a
game repository's root. An engine change is made in the submodule (its own
commits, pushed to `ps2-port-engine`), checked against every game's scripted
runs, tagged, and then each game moves its pin.

```
engine/ps2decomp/   the library (interpreter, derivation, struct recovery, ...)
engine/tools/       CLI drivers; every one takes --game (default: $PS2_GAME, or
                    the only directory under games/)
engine/runtime/     ps2rt: PS2 instruction semantics the generated C++ compiles
                    against, shared by verification and the port
engine/host/        the native host a port runs on (memory, dispatch, kernel,
                    devices); see engine/docs/PORTING.md
engine/tests/       unit tests: python -m unittest discover -s engine/tests
games/<name>/       game.json, src/, include/, config/functions.csv, docs/,
                    tests/, port/ (CMakeLists.txt, overrides.cpp), and
                    (gitignored) extracted/ build/ asm/
```

## Adding a new game

1. Ask the user for the disc image. Never look for one online.
2. In a new game repository with this engine as its `engine/` submodule
   (`git submodule add https://github.com/danielgomesvieira2000/ps2-port-engine engine`).
3. `python engine/tools/port.py init <name> --iso "<iso>" --title "<Title>"`
   writes `games/<name>/` from `engine/templates/game` - `game.json` (`name`,
   `title`, `iso` relative to the game directory, and `executable`, read from
   SYSTEM.CNF's `BOOT2` line), the port's build, an empty overrides.cpp, the
   player's README, a findings stub - and the repository's `.gitignore`.
   Leave `layout` out at first. The image bounds
   come from the ELF, and `$gp` is detected from the entry code. If
   `engine/ps2decomp/game.py`'s `detect_gp` returns None, find `$gp` by hand in
   the entry code and set `"layout": {"gp": "0x..."}`.
4. Extract and analyse:
   ```sh
   python engine/tools/isotool.py extract "<iso>" <EXECUTABLE> -o games/<name>/extracted/
   python engine/tools/analyze.py --game games/<name>
   python engine/tools/imagemap.py --game games/<name>
   ```
   If imagemap shows a large `data` region in the middle of code, look for VU
   microcode (`python engine/tools/vudis.py ... --stats`) and add it to
   `vu_regions` in `game.json`.
5. Check the test memory layout. The defaults put scratch memory at 0x01800000
   and the test stack at 0x01F00000. If the executable or its BSS reaches into
   those, override `scratch`/`stack_top` in `layout` to free addresses below 32 MB.
6. Run the pipeline, then record progress:
   ```sh
   python -u engine/tools/derive.py --game games/<name> --emit --csv > games/<name>/build/derive.log
   python engine/tools/progress.py --game games/<name> --markdown > games/<name>/docs/PROGRESS.md
   ```
   Derivation runs on all but two cores (`--jobs`), bottom-up through the call
   graph so every function sees its derived callees; on a 4 MB executable a
   full run takes about 11 minutes (`--jobs 1`: serial, in address order, over
   half an hour). Run it in the background, and never start two runs at once:
   they write the same outputs.
   Then lift, after derivation, because lifted rows yield to derived ones:
   ```sh
   python -u engine/tools/lift.py --game games/<name> --structure --emit --csv > games/<name>/build/lift.log
   ```
   lift.py remembers each verdict (`build/verdicts.json`, see
   `engine/ps2decomp/verdicts.py`): a function whose C++ text is unchanged, in an
   unchanged environment, is not verified again. A change to structuring only
   re-verifies the functions whose structured text changed - minutes, not the
   forty a cold run takes. Any change to the interpreter, test memory, runtime,
   verification code or the game's analysis invalidates every verdict. Run with
   `--no-cache` before reporting a milestone: it recomputes everything, which
   checks what the cache key cannot see (compiler version, run order).
7. Name what the evidence supports, and share it:
   ```sh
   python engine/tools/signatures.py --game games/<name> name     # config/names.csv
   python engine/tools/signatures.py --game games/<name> export   # into the library
   ```
   `name` names kernel stubs from `engine/data/ee_syscalls.csv`, functions that
   print their own name in diagnostics, and functions whose relocation-masked
   signature is already named in `engine/signatures/library.json`. Every name
   records its source. Run `export` after names improve, so the next game
   benefits. `compare --elf <other.elf>` measures how much code two executables
   share, such as the SDK.
8. Start `games/<name>/docs/FINDINGS.md` with what the binary revealed: the
   compiler (`.comment` section), middleware (RCS `$Id:` strings), the SDK
   version, and the VU regions.
9. Build and run the native port (engine/docs/PORTING.md):
   ```sh
   python engine/tools/port.py --game games/<name>
   ```
   `port.py init` (step 3) wrote its `port/` (CMakeLists.txt naming the
   target, an empty overrides.cpp). The run's report - why it stopped, the guest
   call stack, kernel services, device registers - names the next thing the
   host must provide; implement it in engine/host if every game needs it, as
   an override in the game's port if only this one does.

## The improvement loop

Progress comes from extending the engine so whole classes of refused functions
start to verify. Work the biggest class first:

1. `python engine/tools/refusals.py --game <dir>` groups every function that was
   not derived by the deriver's own reason. Use `--detail "<category>"` for the
   exact messages.
2. `python engine/tools/blockers.py --game <dir> --single` classifies each
   function by the full set of obstacles it contains. It counts the functions
   blocked by exactly one obstacle, since those are the only ones a single fix
   can reach. Removing the most common *first* obstacle has repeatedly unlocked
   nothing.
3. Pick the largest category with a sound fix. Write a failing unit test in
   `engine/tests/` from hand-assembled words first, then implement the fix.
4. To debug a single function, use
   `python engine/tools/inspect_function.py --game <dir> 0xADDRESS`. It shows the
   emitted C++ and, for each disagreeing case, what the original, the model and
   the compiled code produced.
5. Run both test suites and the full pipeline. Compare the numbers, then commit
   with the before and after numbers in the message, and add a section to the
   game's FINDINGS.md when the change taught something about the binary.

Hand-written reconstructions of functions the deriver cannot do (SIMD string
routines, hashes) go in `games/<name>/src/`. Verify them with
`engine/ps2decomp/difftest.py` in `games/<name>/tests/test_reconstruction.py`.

## R5900 facts that cause silent bugs

Each of these has already produced plausible wrong output once:

- Word operations (`addu`, `addiu`, `sll`, `lw`, ...) sign-extend their 32-bit
  result into the 64-bit register, and `lui` is sign-extended too.
- Word shifts mask the amount to 5 bits; doubleword shifts mask it to 6.
- `beql`/`bnel` (branch-likely) execute their delay slot only when taken.
- The EE FPU is not IEEE-754: there are no infinities or NaNs, and overflow
  saturates to the largest magnitude. Keep float bits as u32, never as host
  doubles.
- `lq`/`sq` move 128 bits. Callee-saved registers are spilled with them, so their
  upper halves must survive.
- EABI passes 8 integer arguments (`$a0`-`$a3`, `$t0`-`$t3`) and 8 float
  arguments (`$f12`-`$f19`). Liveness decides which ones a function reads.
- In emitted C++, operator precedence and store order are where bugs hide. The
  tree can be right while the printed C means something else. This is why the
  compiled check exists; never skip it when MSVC is available.

## Working practices that paid off

- A verified function is only verified on the inputs its tests reached. When
  a port's picture looks wrong, build it once from the plain lifted form
  (`lift.py --emit` without `--structure`) and compare: the plain form is
  mechanical and the better reference. Where they differ,
  `engine/tools/bisect_lifted.py` finds the function in a dozen builds, and
  an override running both forms from the same state shows the register that
  differs. Burnout 3's garage lost its cars this way - a structured form
  that never handed back $f20/$f21, whose tests never took a path with a call
  - and the recorded regression pictures had the bug in them too.
- Measure the frame rate where the game is heavy (`--frame-times N`), not the
  run's total: menus at hundreds of frames a second hide a race at twenty.
  `--sample` gives functions and source lines; for compiled VU code, the
  assembly (`cl /FAs` of a generated file) showed what a pair costs.

- Never just wait on a long run. Start a watchdog beside it -
  `python engine/tools/watchdog.py --log <its log> --done <its last line>` under
  a monitor - which reports a log silent too long, memory past a limit, or a
  traceback. Every long stage prints a heartbeat, so silence means stuck. Bound
  any new per-item work (a time and memory cap per function) before running it
  at full scale: a check without one once hung for two hours at 10 GB.
- Make tests deterministic: seed every random input from the function address,
  and reset all registers on every interpreter call.
- When derivation and the interpreter share a bug, they agree and "verify"
  wrong code (`sqrt.s` read fs instead of ft in both). Check the encoding of every
  new instruction against the EE manual, not against the disassembler alone. The
  lifter's whole-register comparison is a second line of defence.
- Profile before optimising. Most slowness came from repeated work: uncached
  catalog lookups, re-deriving unchanged functions, re-running the original for
  the compiled check.
- Patch Python files with the Edit tool or a script written to a file. Shell
  heredocs have turned `\b`, `\x00` and `\\n` into control bytes. Some files are
  CRLF: match patterns against the raw bytes.
- Document as you go. FINDINGS.md records what the binary taught; METHODOLOGY.md
  records what verification means; PROGRESS.md is generated, never hand-edited.
