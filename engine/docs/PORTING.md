# From decompilation to a native port

Decompilation produces C++ for each function and proves it equivalent to the
original. A port runs that C++ on a PC. This document describes the layer in
between, the **native host** (`engine/host/`), and how a game becomes an
executable.

## The pieces

```
engine/runtime/   PS2 instruction semantics: FPU, VU0 macro mode, integer edge
                  cases, 128-bit MMI. Shared by verification and the port, so the
                  port runs exactly the semantics every function was verified on.
engine/host/      The PC platform: guest memory, dispatch, the EE kernel, devices.
games/<g>/src/lifted/   The game's translated functions (lift.py --emit).
games/<g>/port/   The game's port: CMakeLists.txt and overrides.cpp
                  (a new game's from engine/templates: port.py init).
```

Translated code reaches its environment only through a few hooks
(`ps2rt/lifted.h`, `ps2rt/generated.h`):

| hook | in verification | in the port |
|---|---|---|
| `ReadU32`, `Write32`, ... | the Python interpreter's memory | `ps2host::Memory`: flat RDRAM, scratchpad, register windows |
| `Call`, `TailCall` | the interpreter runs the original callee | the dispatch table runs the translated callee, or an override |
| `Syscall` | a fixed stub | `kernel.cpp`, the EE kernel's services |
| `Tick` | aborts a runaway loop | the same, with a budget large enough to play |

Because every translated function was verified against the interpreter through
these same hooks, a port is correct exactly as far as its host implements them
faithfully.

## Building and running

```sh
python engine/tools/lift.py --game games/<g> --structure --emit   # translate
python engine/tools/port.py --game games/<g>                      # build and run
python engine/tools/port.py --game games/<g> run -- --trace       # log every call
```

Options for finding out why boot stops, passed after `run --`:

| option | what it shows |
|---|---|
| `--trace` | every call, not just the first of each function |
| `--notes` | every host event (DMA, interrupts, SIF, kernel calls), repeats included |
| `--watch ADDR` | each call of the function at ADDR, with its argument registers and `$ra` |
| `--peek ADDR` | 64 bytes of guest memory at ADDR, in the report |
| `--window` | the picture, live, in a window (close it to end the run): SDL3 + Vulkan (`engine/gpu`), or GDI when built without `PS2HOST_VULKAN` or without a Vulkan device |
| `--gs-trace` | every GS image transfer and primitive, with its TEX0 fields |
| `--until-frame N` | stop when frame N is done - with `--virtual-time`, the same point of the game every time |
| `--virtual-time` | the scheduler keeps the clock (each loop slice is 1 ms, waiting skips to the next event): the EE timers, alarms, the IOP and the vblanks follow it, so a scripted run (`--press`) is the same run on any machine under any load; frame-identical across runs |
| `--gpu` | the GPU renderer draws instead of the software GS (`gpu_bridge.cpp`); local memory is kept in step with the GPU's images where the game reads it (transfers, CLUT loads, textures, the picture). Compare its `--frames` with a software run's using `engine/tools/framediff.py` |
| `--gpu-shadow` | the GPU renderer draws every primitive too; each frame its targets are compared with the software GS's (and written beside `--frames` pictures as gpu_/soft_ BMPs) |
| `--gpu-shadow-rtt` | the same, a texture that is a GPU target sampled from the target (render to texture), as the GPU renderer will on its own |
| `--gpu-show` | `--gpu-shadow-rtt`, and the window (`--window`) shows the GPU renderer's picture instead of the software GS's |
| `--gs-trace-frame N` | the same for six frames from frame N (a picture is drawn across several) |
| `--gs-dump-frame N` | GS local memory, all 4 MB, written to `gsmem_N.bin` (in the `--frames` directory) when frame N is done - under `--gpu`, as the GPU has it. `engine/tools/gsmem.py` shows a buffer from a dump, or compares two runs' dumps buffer by buffer |
| `--sample` | a sampling profiler on the EE thread (a thousand samples a second, eight frames deep): at the end, the functions most often running and most often on the stack, by name, and where it was running by source file and line (inlined code by the line inlined: a VU step in vu_core.h, not the block) - the port builds with symbols for it |
| `--gs-inline` | the GS model (and the GPU renderer behind it) on the EE's thread rather than its own (`gs_thread.cpp`): for debugging, and for `--sample` to see the GS's share |
| `--frames-from N` | with `--frames`: only from frame N on (the dump keeps 200 pictures) |
| `--frame-times N` | every N frames, how long they took in wall-clock time, the frame (vertical blank) rate and how many DMA transfers to VIF1 - a game's pictures of 3D, usually one each - went a second (`regress.py --frame-times N` prints them). In real time a game that cannot keep up draws every other blank: Burnout 3's race runs at 58 blanks and 25-30 pictures a second. In real time the line also counts the vertical blanks lost - due while the last was still undelivered, the EE busy through a whole frame - which is what puts a movie's pictures behind its sound |
| `--audio` | the SPU2's mixed output on the sound device, reverb included |
| `--sample-gs` | `--sample` on the GS thread instead (the GS model, the GPU bridge, their submissions) |
| `--sample-vif1` | `--sample` on VIF1's worker (real time: VIF1, VU1 and what they feed the GS) |
| `--vif1-sync` | in real time, VIF1 and VU1 on the EE's thread as under `--virtual-time` (vif1_worker.cpp) |
| `--press BUTTON@FRAME` | hold a pad button for 6 frames from that frame (cross, circle, start, up, ...); `BUTTON@FROM-TO` holds it from one frame to another (`cross@17650-19400`: accelerate through a race); repeatable |
| `--real-time-from N` | a `--virtual-time` run goes on in real time from frame N - the clock carrying on, the blanks' timer and VIF1's worker started - so a scripted run gets exactly to what is to be measured, then is measured as played (with the player's settings beside the executable) |
| `--fast-until N[:F]` | in real time, the vertical blanks F times as fast (8 unless said) until frame N, then at their rate: a scripted run's way to the part measured (`PS2HOST_PROFILE_FROM`). Faster than the EE keeps up with, the game falls behind its frame count and the presses miss: Burnout 3's race script holds at 2 |
| `PS2HOST_PROFILE=FILE`, `PS2HOST_PROFILE_FROM=N` | the frame-pacing profiler (`ps2host/timeline.h`): every thread's work and every picture's way to the screen, as a timeline (Chrome's trace format: ui.perfetto.dev), from frame N; at the end a summary - the present intervals' percentiles, the frames dropped or late, and what was busy before each |
| `--mouse FRAME:X,Y[,L\|R\|U\|D]` | the mouse at (X, Y) of the picture (0 to 1 across it) from that frame, its left or right button clicked or the wheel turned up or down then; repeatable. With any, the window's mouse is not read - a scripted run's menus driven by the mouse |
| `--memcard DIR` | a memory card in slot 1, kept as a directory: the game's saves are files there (`ps2host/mc2.h` answers libmc2's handlers; the port binds them). Without it the slot is empty |
| `--vu-record FILE`, `--vu-interpret` | record the VU microprograms a run starts, for compiling ahead of time; run them interpreted (see below) |
| `--vu-fast` | compiled VU1 programs leave out the MAC and STATUS flags no instruction of theirs can read ("Fast mode" below): exact but for those flags, ~15% less VU1 time; exact is the default |
| `--audio-wav FILE` | the same output recorded to a WAV file, for checking sound without listening |
| `PS2HOST_AUDIO_LOG=1`, `PS2HOST_AUDIO_MUTE=1`, `PS2HOST_SPU_LOG=1`, `PS2HOST_SPU_NO_REVERB=1` | sound, looked into: every two seconds how the SPU2's samples came (the longest wait between batches) and what the device made of them (ran out, dropped, fewest buffers queued); the device played unheard; once a second the voices sounding, their volumes fixed or sweeping, the main volumes; the reverb left out |
| `--watch-write ADDR` | every store to a guest address, with the function making it and four of its callers |
| `--iop-trace` | every IOP library call the game's own modules make |
| `--timeout S` | the time limit (30 s); `--budget N` makes a run deterministic |

The guest's own debug output (`kputs`) appears as `[guest]` lines.

With `--disc` and no executable, the port boots the one the disc names
(SYSTEM.CNF's `BOOT2`), read from the image. Started with no arguments at
all - from Explorer, or a shortcut - a port built with `ps2_add_port(... GAME
<dir>)` plays: it asks for the player's own disc image the first time (a file
dialog, checked to be the disc game.json's executable comes from), remembers
it in `pc_settings.ini` beside the executable, and runs on the GPU with a
memory card in `memcard/` there, in a window with sound and no time limit.
It is a windowed program, so no console opens beside it: started from a
terminal it writes to that one, and started with no arguments to a log named
after it (`burnout3.log`), which an error message points to.

The window's title is game.json's `title`. The executable's icon (the
window's and the taskbar's too) is the port's own, `port/icon.ico` - original
art, safe to publish with the port (Burnout 3's from
`port/tools/make_icon_art.py`). A player can give their own build the game's
logo instead: `port/tools/make_icon.py` makes `build/port/icon.ico` from their
disc (the port run to the title screen, the logo's texture taken from GS
memory with `gsmem.py texture`), which `ps2_add_port` takes over the port's
own. Made from the disc, that one is never committed or shared. A `--virtual-time` run
without `--memcard` leaves the player's settings alone and uses the defaults,
so its pictures stay as recorded.

`port.py` generates `build/port/generated/dispatch.cpp` (every translated
function by address, with names from `config/names.csv`), builds `port/` with
the CMake and Ninja that ship with Visual Studio, and runs the result on the
executable in the game's `extracted/` directory. Nothing from the disc is built
into the port; the user supplies it.

The run ends with a report: why it stopped, the guest call stack at that point,
each thread's state, the kernel services used, and every device register
touched. That report is the port's progress measure, as `progress.py` is the
decompilation's. The first thing in it that the host lacks is the next thing to
implement. A run that goes on longer than `--timeout` seconds (30 by default)
stops at its next loop head and reports where it was. If the host process
itself outlives that by 90 seconds, port.py kills it and says so: that is a
host bug, not the game's.

### Under `--gpu`, for finding what the GPU renderer gets wrong

* `PS2HOST_GPU_LOG=1`: every 100 frames, each GPU image (target, depth buffer,
  view) with how often it was reloaded and why, and what came back to the CPU;
* `PS2HOST_GPU_DUMP_FRAME=N` (and `PS2HOST_GPU_DUMP_DIR`): during frame N, each
  target as drawing leaves it, as a BMP, and a line per pass on stderr with
  FRAME, PRIM, TEX0, ALPHA, TEST and ZBUF - to find the pass where a picture
  goes wrong.

The way in, as for the race (GPU.md): run the same scripted input under both
renderers and compare the pictures (`framediff.py`); find the frame; dump its
passes to find the pass; dump GS memory from both renderers after that frame
to see which buffer differs, and from it which rule of the GS the GPU path
does not keep.

### When a picture is wrong because a translation is

Every lifted form was checked against the original, but only on the inputs
the tests reached. Build the port from the plain lifted form as well (lift.py
without --structure) and compare the scripted run's pictures with the
structured build's: the plain form is mechanical and the better reference.
Where they differ, `engine/tools/bisect_lifted.py` finds the function: it
builds the port from one emission with chosen functions' bodies from the
other, renders the frame, and halves the candidates (the functions the run
calls whose two forms differ) until one is left - twelve builds for the car
select's missing cars (0x0042DAE0, FINDINGS.md). To see how the culprit's
two forms differ, override it in the port with a function that runs both
from the same registers and memory and prints the first difference.

## Speed

Where the time goes: the report's `time:` line (sections, timed with the
timestamp counter) and `--sample` (by function). What it showed, and what was
done about it:

* guest memory: the derived code reaches main RAM inline (`generated.h`,
  `ram_base`); registers, the scratchpad and watched writes go to the port;
* calls: `Dispatch` keeps a direct-mapped cache of address -> function and
  call counter;
* wait loops: under `--virtual-time` a loop spinning until an interrupt spends
  slice after slice; a port names such loops (`ps2host::AddIdleLoop`, the
  loop head) and time goes straight to the next event there. Burnout 3's is
  RenderWare's wait for its DMA buffers to flip (0x001F24E0);
* VU1: each instruction pair is decoded once for its stalls; the integer unit
  runs directly;
* DMA reads plain memory through a pointer.

Further: GS memory addresses come from precomputed pages; the GPU bridge
hands images off by the blocks drawn since their last write-back and writes
back only those rows; batches into the same target share a rendering pass;
COP2's FMAC operations are decoded once (VU and macro mode alike), and
their arithmetic is chosen once an instruction rather than once a lane;
image transfers into GS memory write a row's run of pixels at a time
(the page and its offsets found once a page, not once a pixel).

Further still: DMA hands VIF0, VIF1 and PATH3 a run of words at a time;
the scratchpad channels copy through pointers when no store is watched;
`gs::TheGs()` is inline.

Burnout 3's scripted runs under `--gpu`: the car select from 3m55s to about
43s; the race script from 8m28s to about 3m30s (the software GS: 46
minutes). Each frame's GPU work now runs while the next frame is made
(GPU.md, *submission*). VU1 (an interpreter, with its timing model) and
DMA/VIF are now the largest parts; then the GPU's hand-offs between views of
the same memory. The sampler (`--sample`) shows where: run with it before
choosing what to speed up next. Wall-clock times on a laptop vary by half
again over a day (throttling): compare builds back to back.

Since: an IMAGE packet's quadwords go to the transfer together (Gs::Words);
the FMAC computes four lanes at once in single precision (SSE; the
equivalence with the double-precision model is argued in vu_math.h and
checked by VuLanesAreTheModel); and VU microprograms are compiled ahead of
time (below). Under `--gpu` the car select now takes about 24 s and the
race script about 2 minutes (the software GS: 46 minutes).

Each of these was checked by running the scripted runs and comparing the
pictures with the recorded ones (`engine/tools/regress.py`, runs/README.md).

`--frame-times` showed where the race script really stands: the menus run
at hundreds of frames a second, the race itself (frames 9,300-9,900, the
crash and glow scenes) at 12-22. What `--sample`'s line view found there,
and what was done:

* CLUTs drawn by the GPU loaded on the GPU (GPU.md, step 7): 29 s of the
  script's waiting gone;
* every write to GS memory scanned all 2,048 cached textures for the ones
  it invalidates (3%): textures are now indexed by page;
* each texture checked every GPU image for drawing over it on every
  textured primitive: it keeps the few images that can overlap it;
* timers: three nested timed sections a primitive (73 million in the
  script) cost two clock reads each; the bridge times itself once;
* VU flags landing went through a call nearly every pair (5%): inline;
* UNPACK's data went into a byte buffer a word at a time: whole elements
  are read from the DMA words, and an unmasked element is copied as is
  (DMA and VIF from 23% of the run to 11%);
* compiled VU code: a stall check that can never stall is left out (VF0;
  a field written four or more pairs before in the block; a field the
  block has not written, from its fourth pair: see `needless_wait`), and
  the MAC flags are found for four lanes at once;
* the GS on a thread of its own (`gs_thread.cpp`). To the EE the GS is a
  sink: the host answers CSR reads itself, and nothing it does comes back
  before the frame is shown. So the EE side writes each call that feeds it
  (a quadword from XGKICK, a run of PATH2 or PATH3 words, a privileged
  register) into a ring, and the GS thread makes the same calls in the same
  order - the GS model, the GPU bridge, the renderer's recording - while the
  EE, VIF1 and VU1 go on. At the end of a frame, before anything reads the
  GS (the picture, the GPU's frame, dumps), `SyncGs` waits for the ring to
  empty; a scripted run draws the same pictures. Timed sections are kept
  per thread, so the report's shares can add up to more than 100%.
  The car select from 23 s to 15 s; the race script from about 90 s to 56 s.

## VU microprograms compiled ahead of time

The vector units run microprograms the game uploads at run time (MPG), so
the analysis never sees them as code. A run records them instead:

```sh
burnout3.exe <elf> <run args> --vu-interpret --vu-record games/burnout3/build/vu_programs.txt
python engine/tools/port.py --game games/burnout3 build    # compiles what was recorded
```

`--vu-record` writes each program a run starts - the unit, the address and
the instruction pairs reachable from there - and the targets of jumps to a
register's address (JR, JALR), as programs of their own; runs add to the
file, and `--vu-interpret` makes sure every jump is seen by the interpreter.
The file holds the game's microcode: it stays in build/.

`port.py` hands it to `engine/tools/vurecomp.py`, which writes C++ into the
port's generated sources: each program as blocks of straight-line code (from
a branch target to the next delay slot), each instruction pair's registers,
fields and operation constants. What a pair does is the interpreter's own
steps (`ps2host::VuCore`, vu_core.h) - the stalls, the FMAC's and the
divider's latencies, the flags reaching the lower instructions four cycles
late, all of it - so a compiled program ends in the state the interpreter
would, to the cycle. `Vu::Run` takes a program compiled from the code now in
instruction memory (checked word for word: an MPG may have replaced it) and
the interpreter otherwise, and goes back to compiled code wherever a branch
lands on a block.

The interpreter stays the reference. `VuCompiledIsTheInterpreter` (host
tests) compiles 120 random programs as the build goes and runs each from six
random states both ways, comparing every register, flag, pipeline, cycle and
byte of data memory; delaying one result by a cycle fails it in 397 of 720
runs. The report's VU line says how many pairs ran compiled, how many
programs were entered and how many were found stale.

Since: what the compiled code does is still the interpreter's model to the
cycle (VuCompiledIsTheInterpreter compares every register, flag, pipeline
and cycle), but much less of it is done at run time. Each step was argued
sound and checked by that test and the scripted runs' pictures:

* a stall check that can never stall is left out: VF0; a field written four
  or more pairs before in the block (a pair takes a cycle at least, a result
  lands four after its issue); a field the block has not written, from its
  fourth pair (one written before the block lands by the entry cycle + 3);
* the MAC flags are found four lanes at once, and not cleaned again (every
  operation that sets them leaves clean fields);
* flags land lazily - when a lower instruction reads or sets them, the queue
  is full, or the run ends - which is what eager landing would have left;
* operands an earlier FMAC of the block left clean are not cleaned again
  (vurecomp tracks it field by field);
* a block keeps the cycle and the run's state in locals (MSVC re-read them
  from memory after every store), and runs whole or not at all - so its
  pairs check no limit, and an FMAC's flags that a later FMAC of the block
  (four or more pairs from its end) pushes over, with no flag instruction
  between, are never pushed: only STATUS's sticky bits are gathered;
* a program is compared with instruction memory only when it has been
  written since (an MPG or a store into it), not on every entry.

Building for AVX2 was tried: compiled programs then differ from the
interpreter in 4 of 720 random runs, so the port stays on SSE2.

The pairs macro mode shares (MOVE, MR32, MFIR, MTIR, DIV, SQRT, RSQRT;
OPMULA, OPMSUB, ITOF, FTOI, ABS) are VuCore steps of their own rather than
a trip through macro mode's COP2; the R register's few go that way still.

Burnout 3, the car select and the race recorded: 67 programs, 11,364 pairs;
every pair of either run compiled (428 million and 2.46 billion). The car
select 32 s to 24 s on the same build (`--vu-interpret` for the
comparison); the race script about 2 minutes.

### Fast mode (`--vu-fast`)

Most of a compiled pair's time went to the MAC and STATUS flags: the four
lanes' zero and sign bits, the sticky bits, and the queue they land through
four cycles later - about 30% of every pair. Few programs read them: of
Burnout 3's 70 VU1 programs, 40 never do, and the other 30 read MAC alone
(FMAND, FMOR), never STATUS. So vurecomp writes a second set of blocks for
each VU1 program that never reads or sets STATUS (`fast_drops`), in which an
FMAC computes no flags at all unless a MAC reader can see them: a walk of the
program from the pair after it - branches and delay slots as they run, the
FMAC's own pair taken as a delay slot too where a branch is before it - up to
where a later FMAC's flags have landed for certain (four pairs on), or the
program's end. A reader reached first, a jump to a register's address or
code not recorded keep the flags. The newest flags landed at any reader are
then always an FMAC's that kept them, and the cycles are the exact model's,
so every reader reads what it would have.

What it gives up: the flags a run leaves at its end, which a later run could
read before an FMAC of its own lands (the EE never reads VU1's flags; VU0
programs are always exact). `VuFastIsTheExactButFlags` (host tests) runs the
corpus's programs without STATUS readers both ways from random states -
every register, CLIP, Q, P, pipeline, cycle and byte of memory the same, the
flags different - and `regress.py --vu-fast` checks a scripted run's
pictures against the exact ones (it refuses `--update`).

Burnout 3: 5,841 of its 8,970 VU1 FMACs leave their flags out; `--vu-bench`
on 6,000 race runs 5.82 -> 4.93 ns a pair; VIF1's worker in the real-time
race ~71% -> ~60% busy; the car select, the training race and the race draw
every picture as the exact VU does.

## Code that exists only at run time

Some code is never where the analysis looks for it:

* **copied code** - start-up code keeps kernel extensions as data and copies
  them into kernel RAM (Burnout 3 installs three: the alarm services, and two
  more), then runs them there;
* **unreached code** - functions in the executable that no call reaches and no
  pointer table names: virtual methods, stored callbacks, tail-call targets.

The analysis recovers what it can of the second kind from the gaps between the
functions it found (`engine/ps2decomp/gaps.py`). For the rest, the host records
every call to an address with no translation whose bytes it finds in the
executable (`--discover`), and `port.py` translates that code at the address it
runs from and builds again, until a run finds nothing new. A call to a copy of a
function already translated runs that translation directly.

These translations are not verified - the interpreter never runs them - and the
generated source says so.

## Overrides

`ps2host::Override(address, function, name)` replaces the guest function at an
address with a host implementation. It is the mechanism for:

* **hardware redirection** - library routines that drive the GS, VU1 or the IOP
  are replaced by host code that does the same job with the PC's facilities;
* **enhancements** - resolution, frame rate, input, widescreen: hook the
  function that decides it.

Prefer overriding a whole library routine (`sceGsSyncV`, `sceSifCallRpc`) to
emulating the hardware it touches: the routine's contract is smaller than the
hardware's, and `names.csv` already identifies many of them. Routines every
game links - the SDK's - are overridden by name in the engine
(`engine/host/src/sdk.cpp`), wherever the game's names table gives that name,
so a new game gets them as soon as its routines are named.

## What the host provides, by stage

1. **Boot** (done): memory map with its mirrors (kuseg, kseg0, kseg1) and IOP
   RAM; ELF loading; dispatch; kernel services for start-up (SetupThread,
   SetupHeap, FlushCache); the syscall table in guest kernel RAM, so SetSyscall
   and programs that read or patch the table behave as on the console; DMA
   completing at once; GS vertical blank; the EE timers in host time, with
   their interrupts (`timers.cpp`).
2. **Threads** (done): EE threads as host fibers under the kernel's
   strict-priority scheduler; semaphores, sleep and wake-up block and resume.
3. **Interrupts** (vertical blank and DMA done): handlers run at a safe point,
   which is the running thread's next loop head (`lifted::loop_head_hook`) or
   between threads. Vertical blank comes 60 times a second in real time, or at
   once when every thread waits. INTC_STAT (0x1000F000) holds the causes
   raised since the game cleared them (writing 1 clears): libgraph's
   sceGsSyncV clears VBLANK_START and polls it; a port names that poll an
   idle loop (`AddIdleLoop`) so it is waited for asleep. DMA completion is posted by the transfer and
   delivered after the store that started it. Alarms (SetAlarm) run their
   callbacks at the first frame at or past their time.
4. **IOP services** (begun): the host answers the SDK's SIF commands as the
   IOP would (`iop.cpp`): RPC bind and call answered with RPC-end packets in
   the EE's receive buffer and the SIF0 DMA interrupt, run by the scheduler.
   Servers are host functions registered by id (`RegisterRpcServer`); a server
   with none still binds and its calls are noted with their callers, which
   names the library function behind each. CD/DVD (`cdvd.cpp`) reads the
   user's disc image (`disc.cpp`, ISO 9660, `--disc`). Also done: file I/O
   (`fileio.cpp`), module loading (`loadfile.cpp`: loads succeed and run
   nothing) and the controller manager's handshake (`dbc.cpp`). Servers for
   modules a game ships on its own disc belong to its port
   (`games/<g>/port/iop_servers.cpp`). Pads, sound and memory cards follow.
   DMA channels VIF0, VIF1 and GIF are walked (`dma.cpp`), chains included,
   and VIF1's stream is decoded (`vif.cpp`).
   **A game's own IOP modules run** in an R3000 interpreter (`iop_vm.cpp`).
   Each is loaded from the disc and relocated, and its imports are bound to
   the host's IOP libraries (`iop_libs.cpp`, by the numbers in
   `engine/data/iop_exports.csv`). Its threads run under the IOP kernel's
   scheduler, and its SIF commands go to and from the EE. The port names the
   modules to run: `ps2host::iop::RunModule("RWA.IRX")`. Prefer this to
   rewriting a module's protocol; sound drivers are the usual case. The iopheap
   server (`iopheap.cpp`) shares the interpreter's heap. `--iop-trace` logs
   every library call a module makes.
5. **Graphics** (begun): a software GS (`gs.cpp`, `gs_draw.cpp`,
   `gs_memory.cpp`) takes GIF packets from all three paths. It keeps local
   memory in the GS's own swizzled layouts and rasterises sprites, triangles
   and lines with texturing, fog, alpha, depth and destination alpha tests,
   blending and FBMSK, primitives queued and drawn in parallel bands of rows
   (each pixel still sees them in order). PATH3 waits while VIF1 masks it and
   goes one EOP-ended group per unmask, as the GIF lets it. The displayed
   picture goes to the window (`--window`) or BMP files (`--frames`). VU1 runs microprograms in an interpreter
   (`vu.cpp`). Its upper half goes through the runtime's verified VU0
   arithmetic; the pipeline timing microcode can see (stalls, Q/P latency,
   flags 4 cycles late, LOI's I read by the next bundle) is modelled, and
   `engine/tools/vuhazards.py` lists where a program depends on it. VIF1
   (`vif.cpp`) loads it with MPG and UNPACK and starts it with
   MSCAL, and its XGKICKs are PATH1; VU0 micro mode runs `vcallms`. The
   GPU renderer is under way: engine/docs/GPU.md.
6. **Audio** (begun): the SPU2 (`spu2.cpp`) decodes, envelopes and mixes its
   voices at 48 kHz, with the reverb libsd sets; `--audio` plays them and `--audio-wav` records them.
   **Input**: a DualShock 2 on the window's game controller (SDL3) and the keyboard (`dbc.cpp`). Then saves and
   enhancements.

7. **The IPU** (`ipu.cpp`): the MPEG decoder libmpeg drives, so a game's
   movies play. Registers IPU_CMD (BUSY in bit 63, VDEC's and FDEC's
   result), IPU_CTRL, IPU_BP (BP, IFC, FP) and IPU_TOP; the FIFOs at
   0x10007000/0x10007010; DMA channels 3 (FROM_IPU) and 4 (TO_IPU), which the
   generic "completed at once" handling leaves to it. The commands BCLR,
   BDEC, VDEC, FDEC, SETIQ, SETVQ, CSC, PACK and SETTH (IDEC is reported, not
   modelled), over the MPEG-1/2 VLC tables (`ipu_tables.inc`, generated
   from annex B) and an IEEE 1180 reference IDCT.
   * A command runs in steps (a header field, a macroblock), each all or
     nothing: it reads ahead through the FIFO and on into what channel 4 has
     yet to transfer - its source chain walked on a copy - and only a step
     that finishes takes the bits, MADR, QWC and TADR moving by the
     quadwords taken. A step that runs out of data waits, BUSY; what it
     changed is put back (the DC predictors), as it will run again.
   * Channel 4 keeps the FIFO full; restarted in a block, the tag it came
     from (CHCR's TAG field, kept as the hardware keeps it) says whether the
     chain ends after it - a `refe` fence does. libmpeg stops and restarts
     the channel around its other transfers and rewinds MADR by IFC + FP.
   * After a macroblock, BDEC goes over a slice's zero stuffing to the next
     start code and says so in SCD, as libmpeg expects.
   Checked against FFmpeg's decode of the same streams (games/burnout3
   FINDINGS, "Movies"): every picture of the boot movies within IDCT
   rounding.

Each stage is measured the same way: run, read the report, implement what
stopped it.
