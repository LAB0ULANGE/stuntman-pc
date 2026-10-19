/*
 * The host's run state, shared by its parts: dispatch, the kernel, devices and
 * the report. Not part of the interface a game port uses (host.h).
 */

#ifndef PS2HOST_RUN_H
#define PS2HOST_RUN_H

#include <atomic>
#include <chrono>
#include <map>
#include <string>
#include <functional>
#include <vector>

#include "ps2host/host.h"
#include "ps2rt/lifted.h"

namespace ps2gpu {
class Display;
}

namespace ps2host {

/* Thrown when the guest asks for something the host does not provide yet: the
   run stops there and the report says what it was. */
struct Unimplemented {
    std::string what;
};

/* Thrown when the guest program exits through the kernel. */
struct GuestExit {
    s32 code;
};

/* Run a guest function: an override if there is one, else the translated code. */
void Dispatch(lifted::Context& c, u32 address);

/* A fresh EE register context. VU0's vf0 reads (0, 0, 0, 1) in hardware: macro
   code multiplies by vf0w for a 1.0 (Burnout 3's menu fades do), so it starts
   so, and nothing writes it (lifted::Cop2 drops writes to vf0). */
inline lifted::Context NewContext() {
    lifted::Context c{};
    c.vf[0][3] = 0x3F800000u;
    return c;
}

/* What the run touched, for the report at the end. */
struct Stats {
    std::map<u32, u64> calls;                  /* function -> times called */
    std::map<s32, u64> syscalls;               /* service -> times called */
    std::map<u32, u64> register_reads, register_writes;   /* device register -> count */
    std::vector<u32> stopped_in;               /* the stack when the run stopped */
    std::vector<u32> interrupted_in;           /* ...and, if that was a handler, the thread's */
    std::vector<std::string> notes;            /* one line per first-time event */
    /* Code run where it was copied to: (run address, the executable's copy). */
    std::vector<std::pair<u32, u32>> discovered;
};
Stats& RunStats();

/* Whether --trace was given: log every call and kernel service, not only the first. */
bool Tracing();

/* A one-time note: logged the first time `key` is seen, then only counted. */
void Note(const std::string& key, const std::string& text);
/* Whether a Note under `key` would be logged (so its text is worth making). */
bool Noting(const std::string& key);

/* The program's code (code.cpp), which a game may patch while the translated
   code stays the executable's. KeepCode keeps an executable section as loaded
   (and watches stores into the code until the first frame, to name who makes
   them); CheckCode lists the words changed since, each once a value;
   CodeFrame checks at the first frame and stops watching; CodeReport lists
   every word changed now, for the report; ForgetCode drops it all. */
void KeepCode(u32 address, const u8* bytes, u32 size);
std::vector<std::string> CheckCode();
void CodeFrame(u64 frame);
std::string CodeReport();
void ForgetCode();

/* Threads (threads.cpp): run the program's first thread from `entry` under the
   kernel's scheduler; returns only by throwing why the run ended. */
void RunThreads(const lifted::Context& initial, u32 entry, s64 loop_budget);
/* The running thread's guest call stack (or the interrupt handlers'). */
std::vector<u32>& CurrentCalls();
/* The sampler's (profile.cpp): the guest call stack now, innermost first, up
   to max entries, read while the EE's host thread is suspended. */
int GuestCallsNow(u32* out, int max);
/* Guest coroutines: code that switches guest stacks itself - a game's own task
   system, saving a task's registers and jumping into another's - cannot run
   as nested native calls. Each guest stack runs on a host fiber of the EE
   thread running it instead, and the port's overrides of the game's switch
   routines switch fibers where the game switched stacks. The thread keeps its
   own fiber (null below); blocking in the kernel inside a coroutine resumes it
   there. Never in an interrupt handler. */
struct Coroutine;
/* A coroutine of the running thread that runs `body` once first switched to;
   when body returns, it is finished and the thread's own fiber resumes. */
Coroutine* CreateCoroutine(std::function<void()> body);
/* Run `to` (null: the thread's own fiber) until something switches back. */
void SwitchCoroutine(Coroutine* to);
/* The coroutine running now; null on the thread's own fiber. */
Coroutine* CurrentCoroutine();
bool CoroutineFinished(const Coroutine* co);
/* Free a coroutine that is not running; native frames left on it are dropped. */
void DeleteCoroutine(Coroutine* co);
/* In an interrupt handler: the call stack of the thread it interrupted. */
std::vector<u32> InterruptedCalls();
/* Every guest register context: the threads' and the interrupt handlers'. */
void ForEachContext(void (*visit)(lifted::Context&));
/* From another thread, stop translated code at its next loop head: `spend`
   (setting every budget to -1), then again every `every`, until `stopped`
   (waiting up to `every` for the run to end) says it has, or `limit` has
   passed. True when it stopped. Written again because one write can be lost:
   see the Watchdog in host.cpp. */
bool SpendUntilStopped(const std::function<void()>& spend,
                       const std::function<bool(std::chrono::milliseconds)>& stopped,
                       std::chrono::milliseconds every, std::chrono::milliseconds limit);
/* Each thread's state, for the report. */
std::string ThreadReport();
/* The thread, semaphore and interrupt services; false for any other service. */
bool ThreadSyscall(lifted::Context& c, s32 number, u64& result);

/* The IOP over SIF (iop.cpp): the transfers SifSetDma queues, and delivering
   one queued answer to the EE (false when none waits). */
void SifDma(u32 transfers, u32 count);
bool DeliverIopReply();
/* Whether an answer from the IOP waits to be delivered. */
bool IopReplyPending();
/* A packet from the IOP to the EE (a module's sceSifSendCmd): queued, and
   delivered as the SIF0 interrupt at the next safe point. */
void QueueIopPacket(std::vector<u8> packet);
/* sceSifSendCmd for the host's own IOP servers (iop.cpp): `data` written to the
   EE at `dest`, then a command packet - header (sizes, `dest`, `command`) and
   `fields` - for the EE's handler of `command`. Sent while a server answers an
   RPC call, it follows that call's RPC end. */
void SendCmdToEe(u32 command, const std::vector<u8>& fields, u32 dest, const std::vector<u8>& data);
/* The SIF software registers (sceSifGetSreg, sceSifSetSreg): set by the EE
   with command 0x80000001, read and set by IOP modules. 0 until set. */
u32 SoftwareRegister(u32 index);
void SetSoftwareRegister(u32 index, u32 value);
/* A host RPC server's: its answer (the RPC end) reaches the EE `ns` later, on
   the host clock (virtual under --virtual-time) - a device's time, such as a
   CD read's. */
void DelayRpcReply(u64 ns);
/* Ask for interrupts to be delivered at the running thread's next loop head. */
void RequestDelivery();
/* The RPC-end packet answering an RPC request packet, for a server at
   `server` with receive buffers `buffer`/`cbuffer` (IOP addresses). */
void QueueRpcEnd(const std::vector<u8>& request, u32 server, u32 buffer, u32 cbuffer);
/* The IOP services the engine implements: CD/DVD (cdvd.cpp), ... */
void InstallIopServers();
void InstallFileIo();
void InstallLoadFile();
/* The IOP rebooted with a replacement image from the disc (IOPRPnnn.IMG):
   the releases of its modules - loadfile's and fileio's answer their version
   checks. */
void UseIopImage(const std::string& path);
/* A module's release in an IOP image: its library tag, eight characters of
   name (space-padded) then four digits - "loadfile2430", "fileio  2430".
   False when the image has no such tag. */
bool ModuleRelease(const std::vector<u8>& image, const std::string& name, char release[4]);
/* The release file I/O's version check (procedure 0xFF) answers, and its answer:
   the four characters, then four zero bytes. */
void SetFileIoRelease(const char release[4]);
std::vector<u8> FileIoVersionAnswer();
void InstallDbc();
void InstallIopHeap();
/* libmc's memory card server (mcserv.cpp) over --memcard's directory. */
void InstallMcserv();
/* Signal an EE semaphore from the host (the IOP finishing a request); the
   running thread is not preempted. */
void SignalSemaphore(u32 id);
/* Run the DMA-completion handlers registered for a channel, as its interrupt. */
void RaiseDmac(u32 channel);
/* A channel's transfer has completed: its interrupt is delivered at the running
   thread's next loop head, or between threads - never inside the store that
   started the transfer, since code goes on to set up for the interrupt. */
void PostDmac(u32 channel);
/* A device's INTC cause (VIF1's interrupt bit): raised as RaiseIntc does, at
   the running thread's next loop head, as PostDmac's - never inside the store
   that caused it, nor inside a handler. */
void PostIntc(u32 cause);
/* VIF1's stalls at a VIFcode's interrupt bit, for the report (devices.cpp). */
u64 Vif1InterruptStalls();
/* A channel a device moves at its own pace (the IPU's) has finished: its
   D_STAT bit, and its interrupt as PostDmac (devices.cpp). */
void DmaChannelEnded(u32 channel);
/* Set (before budgets are spent) when the run must end, so the loop-head hook
   does not take a spent budget for its own request. */
std::atomic<bool>& StopRequested();

/* SDK library routines the host implements, applied by name (sdk.cpp). */
void ApplySdkOverrides(const FunctionEntry* table, std::size_t count);

/* The EE kernel: lifted::Syscall's implementation (kernel.cpp). */
void KernelSyscall(lifted::Context& c);
const char* SyscallName(s32 service);

/* Device registers with hardware meaning (devices.cpp). */
void InstallDevices();
/* Each vertical blank: write the displayed picture to `dir` every `every`
   frames, at most `limit` of them (--frames). */
void SetFrameDumps(const std::string& dir, int every, int limit, u64 from = 0);
/* The GPU renderer (gpu_bridge.cpp). Shadow: it draws beside the software GS;
   each frame its targets are compared with local memory (and written as BMP
   files into `dump_dir` when that is not empty); ShadowRtt also samples GPU
   targets as textures. Primary (--gpu): it draws instead of the software GS. */
enum class GpuMode { Shadow, ShadowRtt, Primary };
bool StartGpuRenderer(std::string* error, GpuMode mode, bool show);
void GpuShadowFrame(u64 frame, const std::string& dump_dir);
/* A capture (F12): the GPU's passes from `frame` on dumped into
   `passes_dir`, until GpuCaptureEnd; then the picture shown, at the
   internal resolution and the PS2's, into `dir` (screen.bmp, screen_1x.bmp)
   - what it was, as text. GpuPrimary: the GPU draws (--gpu), so there is
   something to capture. */
void GpuCaptureBegin(u64 frame, const std::string& passes_dir);
void GpuCaptureEnd();
std::string GpuCaptureScreen(const std::string& dir);
bool GpuPrimary();
/* The GPU's texture cache (gpu_bridge.cpp, Bridge::texture_cache): "on", "off"
   (every CLUT the GPU loads, loaded again), or "check" (loaded all the same,
   and any CLUT the cache would have kept that came out different said). */
void SetTextureCache(const std::string& mode);
/* How a batch in two passes (or blended in the shader) decides a primitive
   overlaps what it drew: "exact" (triangles sharing a pixel), "exact-direct"
   (the same, each pair's axes worked out anew, as before) or "bbox" (the
   bounding rectangles). */
void SetBatchOverlap(const std::string& mode);
/* The CPU's writes to GPU local memory (texture and CLUT uploads) copied
   together before the GPU next uses that memory ("batch", the default), or
   each at once ("each", as before): the "memory_writes" setting. */
void SetMemoryWrites(const std::string& mode);
/* The GPU images newer than local memory kept in a list (the "newer_list"
   setting, on by default; --newer-list on|off), not looked for among all. */
void SetNewerList(bool on);
/* The GPU images listed by the pages of memory they lie over (the
   "owned_index" setting, on by default; --owned-index on|off), not looked
   for among all. */
void SetOwnedIndex(bool on);
/* COLCLAMP off honoured on the GPU - a blend's low eight bits, in the
   shader (the "colclamp_wrap" setting, "on" by default); "off", such blends
   clamp as the blender does: faster, and wrong where a game counts by
   wrapping (Stuntman's shadows); "fast", a flat colour added that way drawn
   by the blender as an addition or a subtraction (the same while the count
   does not wrap), the rest in the shader. */
void SetColclampWrap(const std::string& mode);
/* The destination alpha test's setup: over the batch's rectangle ("batch",
   the "date_area" setting's default) or the whole target ("full";
   --date-area). */
void SetDateArea(const std::string& mode);
/* The destination alpha test's setup within the draws' pass ("on": the
   target read where it is drawn, VK_KHR_dynamic_rendering_local_read - no
   pass ended for it); "check": the same, then compared with the target as a
   pass of its own reads it (the pixels that differ on stderr); "off": a
   pass of its own. "on" is the "date_in_pass" setting's default
   (--date-in-pass).
   Returns the mode taken ("off" where the GPU cannot). */
std::string SetDateInPass(const std::string& mode);
/* The GPU renderer's Vulkan commands made on a thread of their own, not the
   GS thread (the "vulkan_thread" setting, on by default; --vulkan-thread). */
void SetVulkanThread(bool on);
/* The GPU renderer on a thread of its own, beside the GS model and the
   bridge ("on"); "check": the answers it works out for the bridge's
   write-backs compared with the renderer's (waiting for it each time); "off":
   on the GS thread. "on" is the "render_thread" setting's default
   (--render-thread).
   Returns the mode taken. */
std::string SetRenderThread(const std::string& mode);
/* The window's picture taken from the renderer and presented on its thread,
   in turn, the GS thread not waiting for it (the "present_on_render_thread"
   setting, on by default; --present-on-render-thread; with render_thread). */
void SetPresentOnRenderThread(bool on);
/* For --frame-times: since the last call, the pictures presented on the
   renderer's thread, how many frames the EE was ahead of each, and the frames
   queued for the renderer now. */
std::string PresentLag();
/* The frame (vertical blank) the EE is at. */
u64 EeFrame();
/* The renderer's queue (render_thread): "fast" - a draw's state, target and
   depth buffer only when they change, the "render_queue" setting's default
   - or "plain" (--render-queue). */
void SetRenderQueue(const std::string& mode);
/* The renderer on its thread at most `frames` frames behind the GS thread,
   which waits past that (the "render_queue_frames" setting, 1 by default, 0
   for no limit; --render-queue-frames). */
void SetRenderQueueFrames(u32 frames);
/* "check"'s findings: the answers compared, those that differed. */
bool RenderThreadCheck(std::uint64_t* answers, std::uint64_t* differ);
/* "check"'s findings: the DATE batches checked, the pixels that differed. */
bool DateInPassCheck(std::uint64_t* batches, std::uint64_t* differ);
/* The GPU renderer's internal resolution, now (a GS replay's; a run takes
   the "resolution" setting between frames). */
void GpuSetScale(u32 scale);
/* F12 or L3+R3 (the window's): capture the next frame. */
void RequestCapture();
std::string GpuShadowSummary();
/* --until-frame: stop the run when frame N is done (with --virtual-time, the
   same point every time). */
void SetUntilFrame(u64 frame);
/* --frame-times N: every N frames, a line with the frame rate over them. */
void SetFrameTimes(unsigned every);
/* The GS on a thread of its own (gs_thread.cpp; --gs-inline keeps it on the
   EE's): what feeds the GS goes through these, in order; SyncGs waits until
   the GS has taken all of it, before anything reads the GS's state. */
void StartGsThread();
/* On the EE: wait until VIF1's worker and the GS thread have taken all that
   was given them (a backlog run up while fast-forwarding, drained before
   frames are paced again). */
void GsDrain();
void StopGsThread();
void SyncGs();
void GsQword(int path, u64 low, u64 high);
void GsWord(int path, u32 word);
void GsWords(int path, const u32* words, u32 count, u32 from = ~0u);
void GsPrivileged(u32 address, u64 value);
/* The end of frame N, in the GS's stream: the handler (the GPU bridge's, when
   the GPU draws: hand the frame to the GPU, show its picture) runs on the GS
   thread once the frame's packets are taken, so the EE need not wait for
   them. False when there is no handler: the EE syncs and does it itself. */
void SetGsFrameHandler(std::function<void(u64)> handler);
/* VIF1, VU1 and what feeds the GS on a thread of their own, in real time
   (vif1_worker.cpp): Vif1Run queues a command the worker runs in order (at
   once without the worker, or on it); JoinVif1 waits until all have run. */
void StartVif1Worker();
/* Whether real time runs VIF1 on its worker (default yes; --vif1-sync: no). */
void SetVif1Worker(bool on);
/* VIF1_FBRST written with STC alone (a stall at an interrupt bit ended): the
   EE goes on without waiting for VIF1's worker (the "vif1_stc_no_wait"
   setting, on by default; --vif1-stc-no-wait). */
void SetVif1StcNoWait(bool on);
/* VIF1's worker at most this many frames behind the EE (1; the
   "vif1_frames_behind" setting, --vif1-frames-behind). */
void SetVif1FramesBehind(int frames);
bool Vif1StcNoWait();
/* ...the writes that went on so with the worker busy (would have waited). */
void CountStcUnwaited();
u64 StcUnwaited();
/* Real time: vertical blanks `factor` times as often as the display's (1:
   as they come) - for stretches the game only waits through, a loading
   screen counting frames while it loads (a game's hook turns it on and
   off). */
void SetFastForward(unsigned factor);
/* --fast-until N[:F]: in real time, the blanks F times as fast (8 unless
   said) until frame N - a scripted run's way to what is to be measured -
   then at their rate. Faster than the EE keeps up with, the game falls
   behind its frame count and the run's presses miss. */
void SetFastUntil(u64 frame, unsigned factor);
/* --real-time-from N: a --virtual-time run goes on in real time from frame
   N (the clock carrying on from where it was, the blanks' timer and VIF1's
   worker started): a scripted run's exact way to what is to be measured,
   measured as it is played. */
void SetRealTimeFrom(u64 frame);
/* The clock out of virtual time, carrying on from where it was. */
void LeaveVirtualClock();
/* The factor SetFastForward set (1: none). */
unsigned FastForward();
/* Real time: when the last vertical blank was due (ClockNs), on the timer's
   own grid - not when the EE took it, which it does at its next loop head. */
u64 LastVblankNs();
/* The calling thread runs the game in real time (the EE's, the GS thread,
   VIF1's worker): Windows is asked not to slow it for power (on a hybrid
   processor, not to park it on an efficiency core) and to run it above
   normal priority. */
void FastThread();
bool Vif1Async();
bool OnVif1Worker();
bool Vif1Busy();
void Vif1Run(std::function<void()> command);
/* Vif1Run for a frame's end: the EE waits while the worker is two frames
   behind. */
void Vif1RunFrame(std::function<void()> command);
/* `why` and `address` name the wait for PS2HOST_JOIN_LOG. */
void JoinVif1(const char* why = nullptr, u32 address = 0);
/* GIF_STAT read while the worker has work (memory.cpp): what it will hold
   once the worker has run what is queued, as the EE can tell (devices.cpp),
   checked on the worker in turn; false when it cannot tell, and the read
   waits. Vif1Settled: a read with the queue run (the EE's). */
bool Vif1Guessable(u32 address, u32 size);
bool Vif1Guess(u32 address, u32* value);
void Vif1Settled(u32 address, u32 value);
/* Guesses made, and those found wrong. */
void Vif1Guesses(u64* guesses, u64* wrong);
/* --check-guesses (virtual time): every status read compared with the guess
   real time would have made, as if the worker were always behind. */
void SetCheckGuesses(bool on);
/* How long the worker has worked, in all (microseconds). */
u64 Vif1BusyMicroseconds();
/* How long the EE has waited for it, in all (microseconds): for its queue to
   empty, for it to catch up a frame, for room in the queue. */
void Vif1WaitMicroseconds(u64* join, u64* frame, u64* full);
bool GsFrame(u64 frame);
/* When the EE ended frame `frame` (GsFrame; ClockNs), 0 if not known. */
u64 FrameEndedNs(u64 frame);
/* On the GS thread, as frame `frame` is finished (real time): when it should
   be shown (ClockNs) so frames reach the screen evenly spaced - each the
   same time after the EE ended it, the longest delay of the last few -
   rather than as each happened to come through VIF1's worker and the GS
   thread. The display's own thread waits for it (Display::PresentImageAt). */
u64 FrameDue(u64 frame);
/* How long the GS thread has worked, in all (microseconds). */
u64 GsBusyMicroseconds();
/* How long the GPU has worked on what --gpu handed it, in all (microseconds;
   0 without it, or where the GPU keeps no time). */
u64 GpuBusyMicroseconds();
bool UntilFrameReached();
/* --gs-trace-frame: trace the GS during one frame. */
void SetGsTraceFrame(long long frame);
/* --gs-dump-frame N: GS local memory, all 4 MB, written to
   <frames dir or .>/gsmem_<N>.bin when frame N is done (with --gpu, as the
   GPU has it) - to set beside another run's (engine/tools/gsmem.py). */
void SetGsDumpFrame(long long frame);
void FrameDone(u64 frame);
/* The interlaced field being shown, 0 or 1 (CSR.FIELD): the frame's parity,
   changing at each vertical blank. */
u32 ShownField();
/* The window (window.cpp): opened with --window, presented each frame. */
bool OpenWindow(const wchar_t* title, int width, int height);
/* The game's name, for the window's title (instead of the host's own, "PS2
   port"): set before the window opens. */
void SetGameTitle(const std::string& title);
const std::string& GameTitle();
bool WindowOpen();
bool WindowClosed();
void PresentFrame();
/* The window's events alone (PresentFrame without the picture, which the GS
   thread presents): false once it has been closed. */
bool PumpWindow();
/* Fullscreen on or off (saved as the "fullscreen" setting, settings.h, which
   the window opens with); and whether it is. */
void SetWindowFullscreen(bool on);
bool WindowFullscreen();
/* Vertical sync on or off (the "vsync" setting): no tearing, or the least
   delay. */
void SetWindowVsync(bool on);
bool WindowVsync();
/* The picture's shape (the "aspect" setting: "4:3", "16:9" or "auto", the
   window's own), shown so; and as width over height, for a game to draw it
   so - auto's the window's now. */
void SetWindowAspect(const std::string& aspect);
float PictureAspect();
/* A line after the window's title (a game's tool says what it is doing);
   empty for the title alone. */
void SetWindowStatus(const std::string& status);
/* A panel of text over the picture (a game's tool, its menu): the lines,
   the first a heading, one starting with '>' picked out; none when empty. */
void SetWindowOverlay(const std::vector<std::string>& lines);
/* The game's own picture rate, beside the vertical blanks': a game that
   cannot keep up draws a picture over several blanks (Stuntman's stunt: 6
   pictures a second at 38 blanks), and a blank rate alone says nothing of
   what the player sees. Tick once a blank, `fresh` when the picture shown
   was drawn since the last one; once a second (by `now_ns`, steady) it
   gives the line to show and starts again. */
struct PictureRate {
    u64 from_ns = 0, blanks = 0, pictures = 0;
    bool Tick(bool fresh, u64 now_ns, std::string* line);
};
/* A picture presented (gpu_bridge.cpp): new, or the one shown before. */
void CountPresentedPicture(bool fresh);
/* show_fps=on in the PC settings: the rate over the picture, in the window's
   overlay while nothing else is in it (PumpWindow, once a blank). */

/* The window's display, null when there is none (or no Vulkan one). */
ps2gpu::Display* WindowDisplay();
/* What the window shows instead of the software GS's picture (BGRA, as
   Gs::Picture gives it); false falls back to the software picture. */
void SetPictureSource(std::function<bool(std::vector<u32>*, u32*, u32*)> source);
/* Whether the window has the keyboard (input is read only then). */
bool WindowFocused();
/* Whether a key (a Win32 virtual key) is down, on the window's keyboard. */
bool HostKeyDown(int key);
/* The window's game controller, as a DualShock 2 reads (window.cpp): false
   when none is plugged in. `buttons` bit n is input n held; the sticks are
   right X, Y, left X, Y (0x80 centred); l2, r2 the triggers' travel. */
struct HostPadState {
    u16 buttons = 0;
    u8 sticks[4] = {0x80, 0x80, 0x80, 0x80};
    u8 l2 = 0, r2 = 0;
};
bool HostPad(HostPadState* state);
/* The mouse over the window (window.cpp), in the picture's terms: x and y
   from 0 to 1 across the game's picture as shown (past them outside it);
   `held` the buttons down (bit 0 left, 1 middle, 2 right), and since the last
   call `pressed` those that went down, `wheel` the notches turned (up
   positive), `moved` whether it moved. False without a window that has one. */
struct HostMouseState {
    bool over = false;
    float x = 0.0f, y = 0.0f;
    u32 held = 0, pressed = 0;
    float wheel = 0.0f;
    bool moved = false;
};
bool TakeHostMouse(HostMouseState* state);
/* The game's own pointer (RGBA pixels, rows top-down, clicking at hot_x,
   hot_y), scaled with the window; shown while ShowHostCursor(true), the
   system's otherwise (hidden in fullscreen). */
void SetHostCursor(const u32* rgba, u32 width, u32 height, u32 hot_x, u32 hot_y);
void ShowHostCursor(bool on);
/* The window's controller's motors, as a DualShock 2's: the large one 0-255,
   the small one 0-255 (a DualShock 2's is on or off: 0 or 255). */
void HostRumble(u8 large, u8 small);
/* Controllers (dbc.cpp): each frame, the devices write their state. */
void DbcFrame();
/* The DualShock 2's buttons as the game last read them (bit n: input n held,
   libpad's order - 3 is Start): the keyboard, the game controller and
   --press together. For hooks that act on a button as the game would. */
u16 PadButtonsHeld();
/* A game's own input, as DualShock 2 buttons held (bit n: input n), asked for
   each time the pad is read (once a frame, on the EE's thread, at the
   vertical blank): what the mouse means, say. */
void SetPadHook(u16 (*hook)());
/* --press: hold `button` (cross, start, up, ...) for 6 frames from `frame`. */
bool ScriptPress(const std::string& button, u64 from, u64 to);
/* --mouse FRAME:X,Y[,L|R|U|D]: the mouse at (X, Y) of the picture (0 to 1)
   from that frame - and its left or right button clicked, or the wheel
   turned up or down, then. With any, the window's own mouse is not read. */
bool ScriptMouse(const std::string& spec);
bool ScriptedMouse(HostMouseState* state);
/* Real time: vertical blanks that came while the last was still undelivered (threads.cpp). */
u64 LostVblanks();
/* Real time: how long the EE has slept with every thread waiting - its
   headroom (threads.cpp). */
u64 IdleNanoseconds();
/* The EE timers (timers.cpp): their registers, the host time to their next
   interrupt (< 0: none), and raising the interrupts that are due. */
void InstallTimers();
s64 NanosToNextTimerEvent();
void DeliverTimers();
/* The host's clock, in nanoseconds (timers.cpp), which the EE timers, alarms
   and the IOP's time follow. Real time by default. With --virtual-time it is
   the scheduler's: each spent loop budget advances it by a fixed slice and
   waiting skips it to the next event, so a scripted run is the same run on any
   machine, under any load. Chosen before anything reads it. */
u64 ClockNs();
void SetVirtualClock(bool on);
bool VirtualClock();
void AdvanceClock(u64 ns);
/* COP0 for translated code (mfc0, lifted::read_cop0_hook): Count is the EE's
   cycles (294,912,000 a second) on ClockNs since StartCop0, which installs the
   hook; every other register reads as 0 and is noted (timers.cpp). */
void StartCop0();
u64 ReadCop0(u32 reg);
/* Run the INTC handlers for `cause`, if it is enabled (threads.cpp). */
void RaiseIntc(u32 cause);
/* INTC_STAT: the causes raised (a frame's vertical blanks, RaiseIntc's), and
   clearing those written as 1 (threads.cpp). */
u32 IntcStat();
void ClearIntcStat(u32 bits);
/* Sound (audio.cpp): opened with --audio; the SPU2 hands it 16-bit stereo
   frames at 48 kHz. */
bool OpenAudio();
/* The sound played a touch faster or slower (1% at most) so that what is
   queued for the device stays near kSteerTarget buffers ("audio_steer"
   setting, on by default; --audio-steer). */
void SetAudioSteer(bool on);
/* Real time's vertical blanks at 50 Hz when the game sets up PAL (SetGsCrt
   mode 3; VideoModeAsked), not NTSC's 59.94 always (the "pal_timing" setting,
   on by default; --pal-timing). A virtual-time run keeps 59.94. */
void SetPalTiming(bool on);
/* In real time, the IOP runs while the EE waits for the GS thread or the VIF1
   worker (IopWhileEeWaits, called in those waits; the "iop_while_waiting"
   setting, on by default; --iop-while-waiting on|off): a sound stream's
   thread fills its buffer in time. Only the EE's own thread runs it. */
void SetIopWhileWaiting(bool on);
void IopWhileEeWaits();
void VideoModeAsked(u32 mode);
u32 VblankPeriodUs();
/* The game's frame rate (50 PAL, 60 NTSC) to the window: fullscreen takes a
   refresh it divides evenly, not above the desktop's (the "fullscreen_refresh"
   setting: match, the default, or desktop). */
void SetWindowGameRate(int hz);
constexpr double kSteerTarget = 5.0;
/* The rate to play at for `level` frames queued, aiming at `target`, of `capacity`. */
double SteerRatio(double level, double target, double capacity);
/* Batches of stereo frames resampled at a ratio (input frames per output
   frame), linearly, seamless from one batch to the next. */
class AudioSteer {
public:
    void Stretch(const short* frames, unsigned count, double ratio, std::vector<short>* out);
private:
    double position_ = 0.0;     // the next output's position, from the last batch's last frame (-1)
    short last_[2] = {0, 0};
};
/* Stops the sound and finishes the WAV recording (--audio-wav), before the
   process ends. */
void CloseAudio();
/* ...and/or recorded to a WAV file (--audio-wav). */
bool OpenWavDump(const std::string& path);
void AudioFrames(const short* frames, unsigned count);

}  // namespace ps2host

#endif  /* PS2HOST_RUN_H */
