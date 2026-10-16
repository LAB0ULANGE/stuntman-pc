# ps2-port-engine

A toolchain and runtime for turning PlayStation 2 games into **native PC
ports**: the game's program decompiled into C++ whose behaviour is checked
against the original machine code, then compiled and run on a host that
provides what the PS2 did - the EE kernel, the IOP, sound, the GS drawn with
Vulkan, the VUs, pads, memory cards, the disc - plus what a PC port wants:
higher resolutions, widescreen, frame pacing, settings, mouse, rumble.

It was built on *Burnout 3: Takedown*
([burnout3-pc](https://github.com/danielgomesvieira2000/burnout3-pc), its
first port). The engine itself is game-independent: a game is a directory
with a `game.json`, its own port code and its test runs, in a repository of
its own.

> **No game material here.** This repository holds no game code, assets,
> disc images or BIOS files. A port is made from a disc image its player
> owns; everything taken from it stays on that player's machine.

## How the pieces fit

```
ps2-port-engine/            this repository (MIT)
  ps2decomp/                the library: ELF/ISO readers, R5900 and VU
                            disassemblers, analysis, the interpreter, symbolic
                            derivation, lifting and structuring, verification
  tools/                    the command-line drivers: isotool, analyze, derive,
                            lift, progress, signatures, port, regress,
                            timeline (the frame-pacing profiler), ...
  runtime/                  ps2rt: PS2 instruction semantics the C++ compiles
                            against (verification and the port alike)
  host/                     the native host: memory, dispatch, threads, kernel,
                            IOP, SPU2, GS + GPU bridge, VIF/VU, devices, window
  gpu/                      the Vulkan display and GS renderer
  signatures/               function names shared between games
  docs/                     METHODOLOGY (what "verified" means), ARCHITECTURE,
                            PORTING (every option of a port)
  tests/                    the toolchain's unit tests
  .claude/skills/           ps2-decomp, ps2-port, ps2-release: the workflow
                            and its rules, for Claude Code

<game>-pc/                  a game's repository (e.g. burnout3-pc)
  engine/                   this repository, a git submodule pinned to a tag
  games/<game>/             game.json, port/ (overrides, PC features, test
                            runs), config/, docs/, tests/
```

A game repository is cloned with `git clone --recursive`. The tools are run
from its root (`python engine/tools/port.py --game games/<game>`), and Claude
Code finds the skills in `engine/.claude/skills`.

## Requirements

* Windows 10/11 x64, Python 3.10+ (standard library; Pillow for the art
  scripts), Visual Studio 2022 with C++ (MSVC, its CMake and Ninja)
* For a port: a Vulkan 1.3 GPU, a processor with AVX2
* A disc image you own, for each game: an .iso, or a CD's raw .bin with its
  .cue (Mode 1 or Mode 2 sectors)

## Starting a port of a new game

1. A repository for it, with this one as its engine:
   `git submodule add https://github.com/danielgomesvieira2000/ps2-port-engine engine`
2. The game's directory, from `templates/`:
   `python engine/tools/port.py init <game> --iso "<your disc image>" --title "<Its Title>"`
   writes `games/<game>/` (game.json with the executable read from the
   disc's SYSTEM.CNF, the port's build, an empty overrides.cpp, the player's
   README, a findings stub) and the repository's `.gitignore`, which keeps
   disc material and the game's code out.
3. The pipeline, from the game repository's root:

   ```sh
   python engine/tools/isotool.py extract "<iso>" <EXECUTABLE> -o games/<game>/extracted/
   python engine/tools/analyze.py   --game games/<game>
   python engine/tools/signatures.py --game games/<game> name
   python engine/tools/derive.py    --game games/<game> --emit --csv
   python engine/tools/lift.py      --game games/<game> --structure --emit --csv
   python engine/tools/port.py      --game games/<game>      # build and run natively
   ```

4. The bring-up loop: each run reports why it stopped; what every game needs
   is fixed here, what only this game needs in its own `port/`. See
   [docs/PORTING.md](docs/PORTING.md) and the skills.

What belongs here: what two games have proved they both need, and the
plumbing every port uses (the host's hooks, settings storage, the profiler).
A game's own features - its menus' mouse support, its widescreen anchors,
its packaging - stay in its repository until a second game needs the same;
then the common part moves here, shaped by both.

An engine version is tagged only after every game's recorded runs pass
against it: `python engine/tools/regress_all.py [<other game repositories>]`.
A game repository then moves its `engine/` submodule to the new tag.

## Tests

```sh
python -m unittest discover -s engine/tests       # from a game repository
python -m unittest discover -s tests              # from this repository
```

The host's own tests build with every port (`ps2host_tests`). A change to the
engine is checked against every game's scripted runs
(`engine/tools/regress.py`) before it is tagged: those need the games' discs,
so they run on a developer's machine, not in CI.

## Legal

This repository contains original tooling only - no game code, assets or
binaries, nothing extracted from a disc. Reverse engineering for
interoperability and preservation is protected in many jurisdictions; the
ports made with it redistribute no game assets and need the player's own disc.

MIT licence: [LICENSE](LICENSE).
