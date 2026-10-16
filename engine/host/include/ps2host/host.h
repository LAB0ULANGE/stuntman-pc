/*
 * The native host: runs a game's translated code on a PC.
 *
 * The lifted functions (engine/runtime/include/ps2rt/lifted.h) talk to their
 * world through a handful of hooks - memory reads and writes, calls, syscalls.
 * During verification those hooks lead back to the Python interpreter; here they
 * are served natively: guest memory is a set of flat arrays, a call is a lookup
 * in a table of translated functions, a syscall is a host implementation of the
 * EE kernel service.
 *
 * A game port links three things: the engine (runtime + this host), the game's
 * translated functions with the dispatch table engine/tools/port.py generates
 * for them, and the game's own overrides - host implementations that replace a
 * guest function at its address, which is how hardware access is redirected and
 * how a port's enhancements hook in.
 */

#ifndef PS2HOST_HOST_H
#define PS2HOST_HOST_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ps2rt/lifted.h"

namespace ps2host {

using GuestFunction = void (*)(lifted::Context&);

/* One translated function: its guest address, its code, when known its name, and
   its size in bytes - which identifies a copy of it made at run time. */
struct FunctionEntry {
    u32 address;
    GuestFunction function;
    const char* name;
    u32 size;
};

/* The game's translated functions, sorted by address (port.py generates this). */
void RegisterFunctions(const FunctionEntry* table, std::size_t count);

/* A host implementation that runs instead of the guest function at `address`.
   Overrides take precedence over translated functions; registering twice
   replaces the earlier one. */
void Override(u32 address, GuestFunction function, const char* name);
/* A loop the game waits in until an interrupt ends it (a port's, from what
   it found the time going to): under --virtual-time, when its slice runs out
   the clock goes straight to the next event - vertical blank, timer, alarm -
   as when every thread waits, rather than a slice's worth at a time; and it
   gets short slices, so a wait that is over ends soon after. `address` is the
   loop's head, the target of its backward branch. In real time the loop
   waits asleep instead of spinning, while an InterruptWait is held (the
   port's around the function the loop is in). */
void AddIdleLoop(u32 address);
/* Held around a guest function that may wait in an idle loop (an override's):
   meanwhile the vertical blank timer looks at the EE's loop heads every
   millisecond, so the loop's wait starts asleep soon after it starts - not
   at the next interrupt, a core spun until then. */
struct InterruptWait {
    InterruptWait();
    ~InterruptWait();
    InterruptWait(const InterruptWait&) = delete;
    InterruptWait& operator=(const InterruptWait&) = delete;
};

/* The name of the function at `address`, or null. */
const char* FunctionName(u32 address);

/* A host implementation of an IOP RPC server: given the procedure number and the
   data the EE sent, returns what the EE receives. Servers every game uses (file
   I/O, CD/DVD, pads, memory cards) belong to the engine; a game whose IOP
   modules are its own registers its servers from its port. */
using RpcServer = std::function<std::vector<u8>(u32 procedure, const std::vector<u8>& data)>;
void RegisterRpcServer(u32 sid, RpcServer server);

struct Options {
    std::string elf_path;          /* the game's executable, from the user's own disc */
    std::string disc_path;         /* the user's image of the game disc (.iso) */
    /* Loop iterations a run may take in all; unlimited by default, since a game
       runs indefinitely - the timeout is what stops a run that is stuck. */
    s64 loop_budget = INT64_MAX;
    bool trace_calls = false;      /* log every call, not just the first of each */
    bool all_notes = false;        /* log every host note, not just the first of each */
    bool iop_trace = false;        /* log every IOP library call a module makes */
    int report_limit = 40;         /* how many entries each report section lists */
    int timeout_seconds = 30;      /* stop and report a run that goes on longer */
    std::string discover_path;     /* where to record code found copied at run time */
    std::vector<u32> watch;        /* functions whose every call logs its argument registers */
    std::vector<u32> peek;         /* guest addresses whose 64 bytes the report shows */
    std::string dump_ram;          /* --dump-ram FILE: the EE's 32 MB of RAM, written as the run ends */
    std::vector<u32> watch_writes; /* guest addresses whose every store is logged */
    std::string frames_dir;        /* write the displayed picture here as BMP files */
    int frames_every = 30;         /* ...every this many frames */
    long long frames_from = 0;     /* ...from this frame on (--frames-from) */
    bool window = false;           /* show the picture in a window as it runs */
    bool gs_trace = false;         /* log GS image transfers and textured primitives */
    long long gs_trace_frame = -1; /* ... during this frame only */
    long long gs_dump_frame = -1;  /* write GS local memory when this frame is done */
    bool sample = false;           /* --sample: where the time goes, by function (profile.h) */
    long long sample_from = 0;     /* ...from this frame on (--sample-from) */
    bool sample_gs = false;        /* ...on the GS thread (--sample-gs) */
    bool sample_vif1 = false;      /* ...on VIF1's worker (--sample-vif1) */
    std::string vu_snapshot;       /* --vu-snapshot FILE FROM COUNT: VU1 runs for --vu-bench */
    long long vu_snapshot_from = 0;
    unsigned vu_snapshot_count = 0;
    std::string vu_bench;          /* --vu-bench FILE: those runs, timed (and nothing else) */
    bool check_guesses = false;    /* --check-guesses: GIF_STAT guesses checked (virtual time) */
    bool vif1_sync = false;        /* --vif1-sync: VIF1 on the EE's thread in real time too */
    std::string vu_record;         /* --vu-record: the microprograms run, for vurecomp.py */
    bool vu_interpret = false;     /* --vu-interpret: no compiled microprograms */
    bool vu_fast = false;          /* --vu-fast: VU1 programs without MAC/STATUS readers skip those flags */
    bool gpu = false;              /* draw on the GPU instead of the software GS (gpu_bridge.cpp) */
    bool gpu_shadow = false;       /* draw on the GPU too, and compare (gpu_bridge.cpp) */
    bool gpu_shadow_rtt = false;   /* ...sampling GPU targets as textures */
    bool gpu_show = false;         /* ...and the window shows the GPU's picture */
    bool virtual_time = false;     /* the scheduler keeps time: runs repeat exactly (run.h) */
    unsigned long long real_time_from = 0;  /* --real-time-from N: real time from frame N */
    unsigned long long until_frame = 0;    /* stop when this frame is done (0: never) */
    unsigned frame_times = 0;              /* every N frames, the wall-clock frame rate (0: never) */
    bool gs_inline = false;                /* the GS on the EE's thread rather than its own */
    bool audio = false;            /* play the SPU2's output on the sound device */
    std::string audio_wav;         /* record the SPU2's output to this WAV file */
    std::string memcard;           /* --memcard DIR: a memory card in slot 1, kept as this directory (mc2.h) */
    struct Press { std::string button; u64 from, to; };
    std::vector<Press> presses;    /* --press BUTTON@FRAME (six frames) or BUTTON@FROM-TO (held) */
};

/* Load the executable, run it from its entry point, and report how far it got.
   Returns the process exit code: 0 when the guest exited, 2 when it stopped on
   something the host does not provide yet. */
int Run(const Options& options);

/* Command-line front end shared by every game port: parses the options and runs. */
int Main(int argc, char** argv);

}  // namespace ps2host

#endif  /* PS2HOST_HOST_H */
