/*
 * The IOP interpreter's state, shared by the interpreter (iop_vm.cpp) and the
 * host's implementations of the IOP libraries (iop_libs.cpp).
 */

#ifndef PS2HOST_IOP_VM_INTERNAL_H
#define PS2HOST_IOP_VM_INTERNAL_H

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ps2host/host.h"

namespace ps2host {
namespace iop {

constexpr u32 kRamSize = 2u << 20;
// Where host code returns to from a call into the IOP, and where a thread's
// entry returns to (ExitThread): addresses no code occupies.
constexpr u32 kReturnFromCall = 0x1FFFFFF0u;
constexpr u32 kThreadExit = 0x1FFFFFE0u;
// Where an RPC server's function returns to, in its server thread.
constexpr u32 kRpcReturn = 0x1FFFFFD0u;

struct Context {
    u32 r[32] = {};
    u32 hi = 0, lo = 0;
    u32 pc = 0, npc = 4;
};

enum class Wait { None, Sema, Event, Sleep, Delay, Dormant, Rpc };

struct RpcRequest {
    u32 server = 0, procedure = 0, size = 0, receive = 0, receive_size = 0;
    std::vector<u8> packet;
};

struct RpcServerEntry {
    u32 sid = 0, server = 0, function = 0, buffer = 0, cfunction = 0, cbuffer = 0, queue = 0;
};

struct Thread {
    u32 id = 0;
    Context c;
    s32 priority = 0;
    u64 queued = 0;                 // when it became ready, for FIFO among equals
    bool ready = false;
    Wait wait = Wait::Dormant;
    u32 object = 0;                 // the semaphore or event flag waited on
    u32 event_bits = 0, event_mode = 0, event_result = 0;
    s32 wakeups = 0;
    u64 until_us = 0;               // DelayThread
    u32 entry = 0, stack = 0, stack_size = 0;
    RpcRequest serving;             // the RPC request its server function is running
    u32 rpc_queue = 0;              // the queue it serves (sceSifRpcLoop)
};

struct Semaphore {
    s32 count = 0, max = 0;
    u32 attr = 0;
    std::deque<u32> waiting;        // thread ids
};

struct EventFlag {
    u32 bits = 0, attr = 0;
};

struct Alarm {
    u64 due_us = 0;
    u32 handler = 0, arg = 0;
};

// What a library function did to the calling thread.
enum class Outcome { Return, Block, Yield };

struct Vm;
using Library = std::function<Outcome(Vm& vm, Context& c)>;

struct Vm {
    u8* ram = nullptr;                          // the 2 MB, shared with the EE's view
    std::map<u32, std::pair<std::string, u32>> stubs;  // stub address -> (library, export)
    std::vector<Library> libraries;              // by id, patched into the stubs
    std::map<u32, Thread> threads;
    u32 current = 0;                             // the running thread, 0 in a handler
    bool in_interrupt = false;
    bool reschedule = false;
    u64 counter = 0;
    u32 next_id = 1;
    std::map<u32, Semaphore> semaphores;
    std::map<u32, EventFlag> events;
    std::map<u32, Alarm> alarms;
    std::vector<std::pair<u32, u32>> completions;  // (function, data): DMA-done callbacks due
    std::map<u32, std::pair<u32, u32>> command_handlers;   // command -> (function, data)
    std::map<u32, RpcServerEntry> rpc_servers;            // server address -> entry
    std::map<u32, u32> rpc_queues;                        // queue address -> thread
    std::map<u32, std::deque<RpcRequest>> rpc_pending;    // queue address -> requests
    u32 command_buffer = 0;                      // where arriving packets are put (the host's)
    u32 handler_table = 0;                       // sceSifSetCmdBuffer's, for its answer
    u64 instructions = 0;
    bool trace = false;
    std::map<u32, std::pair<u64, u32>> calls;    // with trace: target -> (count, a caller)
    u32 executing = 0;                           // the instruction being run, for messages

    // Memory, by IOP address (any segment; the RAM mirrors every 2 MB).
    u8* At(u32 address, u32 size);
    u32 Read(u32 address, u32 size);
    void Write(u32 address, u32 value, u32 size);
    std::string String(u32 address, u32 limit = 1024);
    std::string Where();

    // The IOP's heap (sysmem): first fit over free RAM.
    u32 Allocate(u32 size);
    void Free(u32 address);
    u32 LargestFree() const;
    std::map<u32, u32> free_blocks{{0x00010000u, 0x00170000u}};   // address -> size
    std::map<u32, u32> used_blocks;

    // Calls an IOP function to completion in interrupt context (handlers,
    // module entry points); `args` go in $a0-$a3.
    u32 Call(u32 function, std::vector<u32> args);
    /* Runs `count` instructions of `c` (for tests). */
    void Step(Context& c, u64 count);

    Thread* Current() { return current ? &threads[current] : nullptr; }
    void MakeReady(Thread& t);
    // The calling thread waits; false (and the call fails) in interrupt context.
    bool Block(Context& c, Wait why, u32 object);
    u64 NowUs() const;
};

Vm& TheVm();

/* The i-th argument of a call: $a0-$a3, then the stack. */
u32 Arg(Vm& vm, const Context& c, int i);
inline void Return(Context& c, u32 value) { c.r[2] = value; }

/* The library function for (library, export), or null. */
Library FindLibrary(const std::string& library, u32 index);
/* Its documented name, for messages. */
std::string ExportName(const std::string& library, u32 index);

/* Starts the next waiting request of a server thread in sceSifRpcLoop. */
void ServeNext(Vm& vm, Thread& t);

/* Due sound-hardware interrupts (iop_libs.cpp), run by the scheduler. */
void SoundInterrupts(Vm& vm);

/* printf-style formatting from IOP memory: `next` yields each argument word. */
std::string Format(Vm& vm, u32 format, const std::function<u32()>& next);

}  // namespace iop
}  // namespace ps2host

#endif  /* PS2HOST_IOP_VM_INTERNAL_H */
