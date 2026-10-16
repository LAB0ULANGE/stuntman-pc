---
name: ps2-port
description: Bring up, debug, verify and polish a native PC port of a PS2 game with this repository's host (engine/host, engine/gpu) - the bring-up loop, overrides, scripted test runs and regressions, PCSX2 as ground truth, the frame-pacing profiler and benchmarks, PC features (settings, widescreen, mouse, rumble), and working autonomously on it. Use when running or changing a port under games/<name>/port, or engine/host, engine/gpu, engine/runtime.
---

# PS2 ports: bring-up, accuracy, performance, polish

The decompilation (skill `ps2-decomp`) gives every function of a game in C++.
The **host** (`engine/host`, `engine/gpu`) runs that code natively: the EE
kernel, the IOP (its modules run in an interpreter or replaced by C++), the
SPU2, the GS drawn on Vulkan, VU1 microprograms compiled ahead of time, pads,
memory cards, the disc. A game's port is `games/<name>/port/`: a CMake file,
`overrides.cpp` (host code in place of the game's functions), its scripted test
runs (`runs/`) and its PC features. Read `engine/docs/PORTING.md` first: the
port's options and most environment variables are there; the PCSX2 replay and
VU tools document theirs (`engine/tools/pcsx2dump.py`, `PS2HOST_REPLAY_*`,
`PS2HOST_VU_*` in `engine/host/src/host.cpp` and `vu.cpp`).

## The rules that keep a port honest

- **The engine stays game-independent.** A fix every game needs goes in
  `engine/`; what only this game needs goes in its `port/` (an override, a
  hook). Never a game's address or quirk as a constant in `engine/`.
- **The engine takes what two games proved, plus plumbing.** Host hooks,
  settings storage, the profiler: engine. A game's own PC features (its menus'
  mouse, its widescreen anchors, its packaging) stay in its repository, even
  when they look general - until a second game needs the same; then move the
  common part, shaped by both. Don't generalise from one game.
- **An engine version is tagged only when every game passes.** The engine is
  a submodule (`engine/`) of each game repository, pinned to a tag. An engine
  change: commit in `engine/`, `python engine/tools/regress_all.py
  [<other game repositories>]` (each one's `engine/` checked out at the
  candidate commit - it warns otherwise), then - with the user's go-ahead, the
  engine is public - push, tag, and move each game's pin.
- **A new game starts from the templates**:
  `python engine/tools/port.py init <name> --iso "<disc>" --title "..."`.
- **A scripted run is the test.** `--virtual-time` runs are deterministic:
  the same presses on the same frames, byte-identical pictures on any machine.
  Every change to the GPU bridge, the renderer, the host's memory, DMA or VU
  paths runs **every** recorded run of **every** game
  (`python engine/tools/regress.py --game games/<name> <run> --gpu`). A picture
  that changes is a fix or a bug: find which before `--update`. The heaviest run
  (Burnout 3: `race`) is the real GPU test - lighter runs have passed while the
  race broke.
- **Measure before and after, the same way.** A speed claim needs the same
  benchmark before and after (below), on a quiet machine, and the frame-time
  percentiles - not an impression, not one number.
- **Real time is not the virtual clock.** Frame numbers hold only under
  `--virtual-time`; in real time presses drift with the machine. Use
  `--real-time-from N` to get exactly somewhere, then measure in real time.
- **Never commit what came from the disc** - see skill `ps2-release`.

## Bring-up: the loop

```sh
python engine/tools/port.py --game games/<name>          # build, run, report; again until no new code is found
python engine/tools/port.py --game games/<name> build    # build only
games/<name>/build/port/cmake/<name>.exe <elf> --disc <iso> --window --gpu --audio --timeout 600
```

Each run ends with a report: why it stopped, the guest call stack, the kernel
services and device registers it used. That names the next thing to provide:

1. **Stopped on something unimplemented** (a syscall, an IOP library call, a
   device register, a GS/VIF feature): implement it in `engine/host` from the
   hardware's documented behaviour (and PCSX2's), with a test in
   `engine/host/tests` - not a stub that returns success.
2. **Code copied at run time** (`--discover`): port.py translates it and
   builds again.
3. **Hangs**: `--trace`, `--watch-write ADDR` (who writes a value), `--peek`,
   `--sample` (where the EE spends its time), `PS2HOST_HITCH`.
4. **Wrong pictures**: compare the GPU renderer with the software GS
   (`--frames` under both, `engine/tools/framediff.py`), then with PCSX2
   (below). Bisect a picture with `--gs-dump-frame N` + `engine/tools/gsmem.py`.
5. **A function translated wrong**: build from the plain lifted form and
   `engine/tools/bisect_lifted.py` (skill `ps2-decomp`).

Then record test runs that reach the game's main places (boot, menus,
gameplay) as `port/runs/<run>.args` (+ `regress.py --update`), documented in
`port/runs/README.md`. VU microprograms: record the manifest with
`--vu-interpret --vu-record games/<name>/build/vu_programs.txt` on the runs
that reach new microcode and rebuild; the report's `VU: ... (N compiled)` line
must say they are all compiled before anything is timed (the interpreter is
1.5x slower). The manifest is game-derived: `build/`, never committed.

## Overrides: the game's own functions, the host's way

`port/overrides.cpp` replaces or wraps guest functions by address
(`ps2host::Override(0xADDR, Fn, "why")`; the original is `F_<ADDR>(c)`,
`GuestCall` calls a guest function from host code). Good uses: a movie that
can be skipped, a loading wait made fast, a menu entry the port adds (PC
SETTINGS in Burnout 3's main menu, drawn by the game's own menu code), a hook
for widescreen or the mouse. Find the function with the tools, not guesses:
`--watch-write` on the value that matters, the probes in `port/mouse.cpp`
(a selector's moves and their callers), `--trace`, the frame's `--peek` dumps.
Write what was learned in the game's `docs/FINDINGS.md`. Override as little
as possible: the game's own code, sounds and behaviour are the point.

## A player's capture, replayed

F12 in a playing port (or `PS2HOST_CAPTURE_FRAME=n`) writes a capture folder
beside the settings, and with it `replay.bin`: the frames' GS stream (memory,
registers, every GIF packet). Replay it through the software GS at 1x and
the GPU at any scale (`PS2HOST_REPLAY_GPU=1`, `PS2HOST_REPLAY_SCALE=n`): if
the software GS draws it right and the GPU not, bisect with
`PS2HOST_REPLAY_STOP_PRIM=n`, comparing a buffer's rectangle in the two
`gsmem.bin` (engine/tools/gsmem.py's `pixel`) - mind which buffer the bad
pixels are drawn into before being copied to the one shown.
`PS2HOST_GPU_TEXDUMP=TBP` writes each GPU decode of a texture there. Found
so: alpha-tested batches drawn in two passes (AFAIL FB_ONLY) reordered
overlapping primitives (Burnout 3's crash titles, dark streaks).

## Ground truth: PCSX2

When the port and the software GS agree but the picture still looks wrong,
compare with PCSX2 (installed as a reference; the player's own BIOS in
`bios/`, never committed). A GS dump from PCSX2 (`.gs.zst`) replays through
the port's GS: `engine/tools/pcsx2dump.py dump.gs.zst out/` then
`<exe> --gs-replay out/replay.bin out/` (`PS2HOST_REPLAY_GPU=1` through the
GPU renderer, `PS2HOST_REPLAY_STOP_PRIM=n` to stop at a primitive and diff
memory mid-frame). VU differences: `--vu-snapshot`, `PS2HOST_VU_FIND`,
`PS2HOST_VU_STEPS`. What PCSX2 taught so far: VU and EE floats round toward
zero; Ad blends at full strength; 16-bit framebuffer masks are in RGBA32
positions; volume sweeps fade voices (sound).

## Performance and frame pacing

The goal is every picture on time: **no dropped frame in gameplay**.

- **The profiler:** `PS2HOST_PROFILE=file` (`PS2HOST_PROFILE_FROM=N`) records
  every thread (EE, VIF1 worker, GS thread, present) and every picture's way
  to the screen; open the file in ui.perfetto.dev. The summary at the end
  gives the present intervals' percentiles, every late or dropped frame and
  what was busy before it. `engine/tools/timeline.py file at T` (every lane
  around a moment), `longest`, `frames` (each lane's busy time a frame).
- **The benchmark:** a scripted run with `--real-time-from N` (virtual time
  exactly to the gameplay, then real time), windowed, at a fixed internal
  resolution, sound on (`PS2HOST_AUDIO_MUTE=1` to measure in silence), from a
  copy of the executable with its own `pc_settings.ini` (never the player's).
  Run it at least twice: rare late frames vary from run to run.
- **Where time goes:** `--sample`, `--sample-gs`, `--sample-vif1` (functions
  by thread), `--frame-times N`, `PS2HOST_GPU_PROFILE`, `PS2HOST_GPU_LOG`
  (draws, write-backs, reloads, decodes), `PS2HOST_GPU_WRITE_LOG` (memory
  brought back to the CPU), `PS2HOST_AUDIO_LOG` / `PS2HOST_SPU_LOG` (sound).
- **What fixed late frames before:** the CPU waiting for the GPU (a download
  for a partial write - patched on the GPU instead), threads that must wake on
  time at low priority (the blank's timer, the present thread: highest
  priority, high-resolution waitable timers), Windows dropping the timer
  resolution for a hidden window, per-texture driver allocations (images from
  pooled blocks), a full texture cache. Fix the cause the profiler names; then
  the benchmark again.

## PC features

The host gives every game: the window and fullscreen (Alt+Enter; fullscreen
picks a refresh that is a multiple of 60), vsync and frame pacing, the
internal resolution, settings kept in `pc_settings.ini` (`ps2host/settings.h`),
the disc picker on first start, a memory card folder, rumble, the mouse
(`TakeHostMouse`, `SetHostCursor`, `SetPadHook`), a log beside the executable,
PCSX2 texture packs (`ps2host/texture_pack.h`: the "texture pack" folder
beside the settings, PCSX2's XXH3 names, DDS kept block-compressed on the GPU,
Tab toggles; scripted runs never load one - test with `PS2HOST_TEXTURE_PACK`
and `--real-time-from`).
Per game: a settings menu inside the game's own menus where one can be hooked,
widescreen (the game's camera and its 2D, with per-element anchors - Burnout
3's `port/widescreen.cpp`: RenderWare's RwCameraSetViewWindow), the mouse's
targets (the game's menu objects), the window title and version
(`game.json`), an original icon (`port/icon.ico`). Art the port ships is
original (a script draws it: `port/tools/make_cursor.py`, `make_icon_art.py`),
never the game's.

## Working autonomously

The user often hands over long stretches ("continue autonomously and
continuously", nights away). Then:

- Plan first: write down the goal, how it will be measured, the steps in the
  order that fixes the most; keep the plan in the game's docs and update it.
- Pick the recommended next step yourself and do it; don't stop to ask about
  routine steps. Ask only about what is the user's to decide (a licence,
  publishing, deleting their data, what the game should do).
- Every step: change, build, test (the engine's tests, the scripted runs),
  measure, commit with before/after numbers in the message. Revert what does
  not help - and say so in the report.
- Watch long runs: a watchdog on their log (silence, memory) alongside, not
  just the completion notice (`engine/tools/watchdog.py` under a monitor).
- Never drive the user's own mouse or keyboard to test, never change their
  system (volume, settings, files outside the repo) - scripted input
  (`--press`, `--mouse`), copies of the executable and settings in a scratch
  folder. A muted measurement writes silence; it never sets the system mixer.
- Patch code with the Edit tool or a patch script written to a file (raw
  strings) - never in a shell heredoc: they have turned `\n` and `\x00` into
  real bytes inside C++ and Python strings, again and again.
- At the end: what was done (with the numbers), what was tried and dropped,
  what is left and the next step - honestly, including what did not work.
