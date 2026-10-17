/*
 * The IOP libraries a module imports, implemented by the host (iop_vm.h).
 *
 * Each function is bound by library and export number; the numbers and names
 * are the SDK's (engine/data/iop_exports.csv, from ps2sdk's export tables), and
 * the error codes are the kernel's (ps2sdk kerr.h). Arguments arrive as the
 * IOP's calling convention passes them: $a0-$a3, then the stack.
 *
 *   sysmem, sysclib, stdio   memory, C library, console output (logged as [iop])
 *   intrman                  interrupts: the host has no interrupt masking
 *   thbase, thsemap, thevent the IOP kernel's threads, semaphores, event flags,
 *   thfpool                  fixed-size memory pools, clock and alarms, on the
 *                            interpreter's scheduler
 *   sifman, sifcmd           SIF DMA and commands, to and from the EE (iop.cpp)
 *   libsd                    the sound processor (SPU2): state is kept, sound
 *                            data is copied into SPU RAM, and transfer interrupts
 *                            arrive at the rate the hardware's would - nothing
 *                            is played yet.
 *
 * A function the host lacks stops the run with its name (iop_vm.cpp).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cctype>
#include <map>
#include <memory>
#include <sstream>

#include "iop_vm_internal.h"
#include "spu2.h"
#include "ps2host/disc.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"
#include "ps2host/profile.h"

namespace ps2host {
namespace iop {

namespace {

constexpr s32 KE_OK = 0, KE_ILLEGAL_CONTEXT = -100, KE_UNKNOWN_THID = -407;
constexpr s32 KE_NOT_DORMANT = -414, KE_NOT_WAIT = -416, KE_CAN_NOT_WAIT = -417;
constexpr s32 KE_SEMA_ZERO = -419, KE_EVF_COND = -421, KE_UNKNOWN_SEMID = -408;
constexpr u32 WEF_OR = 1, WEF_CLEAR = 0x10;
constexpr u64 kClockHz = 36864000;              // the IOP's system clock

using Map = std::map<std::pair<std::string, u32>, Library>;

u32 A(Vm& vm, Context& c, int i) { return Arg(vm, c, i); }

Outcome Ret(Context& c, u32 value) {
    Return(c, value);
    return Outcome::Return;
}

u64 UsToClock(u64 us) { return us * kClockHz / 1000000; }
u64 ClockToUs(u64 clock) { return clock * 1000000 / kClockHz; }

void Log(const std::string& text) {
    std::string line = text;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (!line.empty()) std::fprintf(stderr, "[iop] %s\n", line.c_str());
}

// ---------------------------------------------------------------- threads

Outcome CreateThread(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 0);           // iop_thread_t: attr, option, thread, stacksize, priority
    Thread t;
    t.id = vm.next_id++;
    t.entry = vm.Read(p + 8, 4);
    t.stack_size = (vm.Read(p + 12, 4) + 15) & ~15u;
    t.priority = s32(vm.Read(p + 16, 4));
    t.stack = vm.Allocate(t.stack_size);
    if (!t.stack) return Ret(c, u32(-400));          // KE_NO_MEMORY
    t.wait = Wait::Dormant;
    vm.threads[t.id] = t;
    return Ret(c, t.id);
}

Outcome Start(Vm& vm, Context& c, u32 id, u32 arg) {
    auto found = vm.threads.find(id);
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    Thread& t = found->second;
    if (t.wait != Wait::Dormant) return Ret(c, u32(KE_NOT_DORMANT));
    t.c = Context{};
    t.c.r[4] = arg;
    t.c.r[28] = c.r[28];
    t.c.r[29] = t.stack + t.stack_size - 16;
    t.c.r[31] = kThreadExit;
    t.c.pc = t.entry;
    t.c.npc = t.entry + 4;
    vm.MakeReady(t);
    Return(c, KE_OK);
    return vm.reschedule && !vm.in_interrupt ? Outcome::Yield : Outcome::Return;
}

Outcome StartThread(Vm& vm, Context& c) { return Start(vm, c, A(vm, c, 0), A(vm, c, 1)); }
Outcome StartThreadArgs(Vm& vm, Context& c) { return Start(vm, c, A(vm, c, 0), A(vm, c, 2)); }

Outcome ExitThread(Vm& vm, Context& c) {
    Thread* t = vm.Current();
    if (!t) return Ret(c, u32(KE_ILLEGAL_CONTEXT));
    t->ready = false;
    t->wait = Wait::Dormant;
    return Outcome::Block;
}

Outcome DeleteThread(Vm& vm, Context& c) {
    auto found = vm.threads.find(A(vm, c, 0));
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    if (found->second.wait != Wait::Dormant || found->first == vm.current) return Ret(c, u32(KE_NOT_DORMANT));
    vm.Free(found->second.stack);
    vm.threads.erase(found);
    return Ret(c, KE_OK);
}

Outcome TerminateThread(Vm& vm, Context& c) {
    auto found = vm.threads.find(A(vm, c, 0));
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    // Ended while it waited, it leaves the queue it waited in: the next
    // SignalSema or FreeFpl goes to the next waiter - not to it, which may be
    // deleted by then.
    const u32 id = found->first;
    auto leave = [id](std::deque<u32>& waiting) {
        waiting.erase(std::remove(waiting.begin(), waiting.end(), id), waiting.end());
    };
    for (auto& entry : vm.semaphores) leave(entry.second.waiting);
    for (auto& entry : vm.pools) leave(entry.second.waiting);
    found->second.ready = false;
    found->second.wait = Wait::Dormant;
    return Ret(c, KE_OK);
}

Outcome GetThreadId(Vm& vm, Context& c) {
    return Ret(c, vm.in_interrupt ? u32(KE_ILLEGAL_CONTEXT) : vm.current);
}

// CheckThreadStack(): the bytes left on the calling thread's stack, from $sp
// down to the stack's base.
Outcome CheckThreadStack(Vm& vm, Context& c) {
    const Thread* t = vm.in_interrupt ? nullptr : vm.Current();
    if (!t) return Ret(c, u32(KE_ILLEGAL_CONTEXT));
    const u32 sp = c.r[29];
    return Ret(c, sp > t->stack ? sp - t->stack : 0);
}

Outcome ChangeThreadPriority(Vm& vm, Context& c) {
    u32 id = A(vm, c, 0);
    if (id == 0) id = vm.current;
    auto found = vm.threads.find(id);
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    found->second.priority = s32(A(vm, c, 1));
    Return(c, KE_OK);
    return vm.in_interrupt ? Outcome::Return : Outcome::Yield;
}

Outcome RotateThreadReadyQueue(Vm& vm, Context& c) {
    Return(c, KE_OK);
    return vm.in_interrupt ? Outcome::Return : Outcome::Yield;
}

Outcome GetThreadCurrentPriority(Vm& vm, Context& c) {
    Thread* t = vm.Current();
    return Ret(c, t ? u32(t->priority) : u32(KE_ILLEGAL_CONTEXT));
}

Outcome SleepThread(Vm& vm, Context& c) {
    Thread* t = vm.Current();
    if (!t) return Ret(c, u32(KE_CAN_NOT_WAIT));
    Return(c, KE_OK);
    if (t->wakeups > 0) {
        --t->wakeups;
        return Outcome::Return;
    }
    vm.Block(c, Wait::Sleep, 0);
    return Outcome::Block;
}

Outcome WakeupThread(Vm& vm, Context& c) {
    auto found = vm.threads.find(A(vm, c, 0));
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    Thread& t = found->second;
    if (!t.ready && t.wait == Wait::Sleep) {
        t.c.r[2] = KE_OK;
        vm.MakeReady(t);
    } else {
        ++t.wakeups;
    }
    Return(c, KE_OK);
    return vm.reschedule && !vm.in_interrupt ? Outcome::Yield : Outcome::Return;
}

Outcome CancelWakeupThread(Vm& vm, Context& c) {
    u32 id = A(vm, c, 0);
    if (id == 0) id = vm.current;
    auto found = vm.threads.find(id);
    if (found == vm.threads.end()) return Ret(c, u32(KE_UNKNOWN_THID));
    const s32 count = found->second.wakeups;
    found->second.wakeups = 0;
    return Ret(c, u32(count));
}

Outcome DelayThread(Vm& vm, Context& c) {
    Thread* t = vm.Current();
    if (!t) return Ret(c, u32(KE_CAN_NOT_WAIT));
    t->until_us = vm.NowUs() + A(vm, c, 0);
    Return(c, KE_OK);
    vm.Block(c, Wait::Delay, 0);
    return Outcome::Block;
}

void PutClock(Vm& vm, u32 p, u64 clock) {
    vm.Write(p, u32(clock), 4);
    vm.Write(p + 4, u32(clock >> 32), 4);
}

u64 GetClock(Vm& vm, u32 p) { return u64(vm.Read(p, 4)) | (u64(vm.Read(p + 4, 4)) << 32); }

Outcome GetSystemTime(Vm& vm, Context& c) {
    PutClock(vm, A(vm, c, 0), UsToClock(vm.NowUs()));
    return Ret(c, KE_OK);
}

Outcome GetSystemTimeLow(Vm& vm, Context& c) { return Ret(c, u32(UsToClock(vm.NowUs()))); }

Outcome USec2SysClock(Vm& vm, Context& c) {
    PutClock(vm, A(vm, c, 1), UsToClock(A(vm, c, 0)));
    return Ret(c, KE_OK);
}

Outcome SysClock2USec(Vm& vm, Context& c) {
    const u64 us = ClockToUs(GetClock(vm, A(vm, c, 0)));
    vm.Write(A(vm, c, 1), u32(us / 1000000), 4);
    vm.Write(A(vm, c, 2), u32(us % 1000000), 4);
    return Ret(c, KE_OK);
}

Outcome SetAlarm(Vm& vm, Context& c) {
    const u64 clock = GetClock(vm, A(vm, c, 0));
    const u32 handler = A(vm, c, 1), arg = A(vm, c, 2);
    vm.alarms[handler] = {vm.NowUs() + ClockToUs(clock), handler, arg};
    return Ret(c, KE_OK);
}

Outcome CancelAlarm(Vm& vm, Context& c) {
    vm.alarms.erase(A(vm, c, 0));
    return Ret(c, KE_OK);
}

// ---------------------------------------------------------------- semaphores

Outcome CreateSema(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 0);           // iop_sema_t: attr, option, initial, max
    Semaphore s;
    s.attr = vm.Read(p, 4);
    s.count = s32(vm.Read(p + 8, 4));
    s.max = s32(vm.Read(p + 12, 4));
    const u32 id = vm.next_id++;
    vm.semaphores[id] = s;
    return Ret(c, id);
}

Outcome DeleteSema(Vm& vm, Context& c) {
    auto found = vm.semaphores.find(A(vm, c, 0));
    if (found == vm.semaphores.end()) return Ret(c, u32(KE_UNKNOWN_SEMID));
    for (u32 id : found->second.waiting) {
        Thread& t = vm.threads[id];
        t.c.r[2] = u32(-425);                        // KE_WAIT_DELETE
        vm.MakeReady(t);
    }
    vm.semaphores.erase(found);
    return Ret(c, KE_OK);
}

Outcome SignalSema(Vm& vm, Context& c) {
    auto found = vm.semaphores.find(A(vm, c, 0));
    if (found == vm.semaphores.end()) return Ret(c, u32(KE_UNKNOWN_SEMID));
    Semaphore& s = found->second;
    if (!s.waiting.empty()) {
        Thread& t = vm.threads[s.waiting.front()];
        s.waiting.pop_front();
        t.c.r[2] = KE_OK;
        vm.MakeReady(t);
    } else if (s.count < s.max || s.max <= 0) {
        ++s.count;
    }
    Return(c, KE_OK);
    return vm.reschedule && !vm.in_interrupt ? Outcome::Yield : Outcome::Return;
}

Outcome WaitSema(Vm& vm, Context& c) {
    auto found = vm.semaphores.find(A(vm, c, 0));
    if (found == vm.semaphores.end()) return Ret(c, u32(KE_UNKNOWN_SEMID));
    Semaphore& s = found->second;
    if (s.count > 0) {
        --s.count;
        return Ret(c, KE_OK);
    }
    if (!vm.Block(c, Wait::Sema, found->first)) return Ret(c, u32(KE_CAN_NOT_WAIT));
    s.waiting.push_back(vm.current);
    return Outcome::Block;
}

Outcome PollSema(Vm& vm, Context& c) {
    auto found = vm.semaphores.find(A(vm, c, 0));
    if (found == vm.semaphores.end()) return Ret(c, u32(KE_UNKNOWN_SEMID));
    if (found->second.count <= 0) return Ret(c, u32(KE_SEMA_ZERO));
    --found->second.count;
    return Ret(c, KE_OK);
}

Outcome ReferSemaStatus(Vm& vm, Context& c) {
    auto found = vm.semaphores.find(A(vm, c, 0));
    if (found == vm.semaphores.end()) return Ret(c, u32(KE_UNKNOWN_SEMID));
    const u32 p = A(vm, c, 1);           // attr, option, initial, max, current, waiting, ...
    const Semaphore& s = found->second;
    vm.Write(p, s.attr, 4);
    vm.Write(p + 4, 0, 4);
    vm.Write(p + 8, 0, 4);
    vm.Write(p + 12, u32(s.max), 4);
    vm.Write(p + 16, u32(s.count), 4);
    vm.Write(p + 20, u32(s.waiting.size()), 4);
    return Ret(c, KE_OK);
}

// ---------------------------------------------------------------- event flags

bool Satisfied(u32 bits, u32 want, u32 mode) {
    return (mode & WEF_OR) ? (bits & want) != 0 : (bits & want) == want;
}

// Wakes the threads whose wait an event flag's new bits satisfy.
void CheckWaiters(Vm& vm, u32 id) {
    EventFlag& e = vm.events[id];
    for (auto& [tid, t] : vm.threads) {
        if (t.ready || t.wait != Wait::Event || t.object != id) continue;
        if (!Satisfied(e.bits, t.event_bits, t.event_mode)) continue;
        if (t.event_result) vm.Write(t.event_result, e.bits, 4);
        if (t.event_mode & WEF_CLEAR) e.bits &= ~t.event_bits;
        t.c.r[2] = KE_OK;
        vm.MakeReady(t);
    }
}

Outcome CreateEventFlag(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 0);           // iop_event_t: attr, option, bits
    EventFlag e;
    e.attr = vm.Read(p, 4);
    e.bits = vm.Read(p + 8, 4);
    const u32 id = vm.next_id++;
    vm.events[id] = e;
    return Ret(c, id);
}

Outcome DeleteEventFlag(Vm& vm, Context& c) {
    vm.events.erase(A(vm, c, 0));
    return Ret(c, KE_OK);
}

Outcome SetEventFlag(Vm& vm, Context& c) {
    const u32 id = A(vm, c, 0);
    if (!vm.events.count(id)) return Ret(c, u32(-409));     // KE_UNKNOWN_EVFID
    vm.events[id].bits |= A(vm, c, 1);
    CheckWaiters(vm, id);
    Return(c, KE_OK);
    return vm.reschedule && !vm.in_interrupt ? Outcome::Yield : Outcome::Return;
}

Outcome ClearEventFlag(Vm& vm, Context& c) {
    const u32 id = A(vm, c, 0);
    if (!vm.events.count(id)) return Ret(c, u32(-409));
    vm.events[id].bits &= A(vm, c, 1);
    return Ret(c, KE_OK);
}

Outcome WaitEventFlag(Vm& vm, Context& c) {
    const u32 id = A(vm, c, 0), want = A(vm, c, 1), mode = A(vm, c, 2), result = A(vm, c, 3);
    if (!vm.events.count(id)) return Ret(c, u32(-409));
    EventFlag& e = vm.events[id];
    if (Satisfied(e.bits, want, mode)) {
        if (result) vm.Write(result, e.bits, 4);
        if (mode & WEF_CLEAR) e.bits &= ~want;
        return Ret(c, KE_OK);
    }
    if (!vm.Block(c, Wait::Event, id)) return Ret(c, u32(KE_CAN_NOT_WAIT));
    Thread* t = vm.Current();
    t->event_bits = want;
    t->event_mode = mode;
    t->event_result = result;
    return Outcome::Block;
}

Outcome PollEventFlag(Vm& vm, Context& c) {
    const u32 id = A(vm, c, 0), want = A(vm, c, 1), mode = A(vm, c, 2), result = A(vm, c, 3);
    if (!vm.events.count(id)) return Ret(c, u32(-409));
    EventFlag& e = vm.events[id];
    if (!Satisfied(e.bits, want, mode)) return Ret(c, u32(KE_EVF_COND));
    if (result) vm.Write(result, e.bits, 4);
    if (mode & WEF_CLEAR) e.bits &= ~want;
    return Ret(c, KE_OK);
}

Outcome ReferEventFlagStatus(Vm& vm, Context& c) {
    const u32 id = A(vm, c, 0), p = A(vm, c, 1);    // attr, option, initBits, currBits, numThreads
    if (!vm.events.count(id)) return Ret(c, u32(-409));
    const EventFlag& e = vm.events[id];
    vm.Write(p, e.attr, 4);
    vm.Write(p + 4, 0, 4);
    vm.Write(p + 8, 0, 4);
    vm.Write(p + 12, e.bits, 4);
    vm.Write(p + 16, 0, 4);
    return Ret(c, KE_OK);
}

// ---------------------------------------------------------------- fixed-size pools

// thfpool: CreateFpl(iop_fpl_param_t*: attr, option, block_size, blocks) takes
// blocks * block_size from the heap; AllocateFpl gives a block or, with none
// free, waits as WaitSema does - FreeFpl then hands the freed block straight
// to the first waiter; pAllocateFpl/ipAllocateFpl answer KE_NO_MEMORY instead.
constexpr s32 KE_NO_MEMORY = -400, KE_ILLEGAL_MEMSIZE = -427, KE_UNKNOWN_FPLID = -412;
constexpr s32 KE_ILLEGAL_MEMBLOCK = -426, KE_WAIT_DELETE = -425;

Outcome CreateFpl(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 0);
    FixedPool pool;
    pool.attr = vm.Read(p, 4);
    const s32 size = s32(vm.Read(p + 8, 4)), blocks = s32(vm.Read(p + 12, 4));
    if (size <= 0 || blocks <= 0) return Ret(c, u32(KE_ILLEGAL_MEMSIZE));
    pool.block_size = (u32(size) + 3) & ~3u;         // blocks stay word-aligned
    pool.blocks = u32(blocks);
    pool.base = vm.Allocate(pool.block_size * pool.blocks);
    if (!pool.base) return Ret(c, u32(KE_NO_MEMORY));
    for (u32 i = 0; i < pool.blocks; ++i) pool.free.push_back(pool.base + i * pool.block_size);
    const u32 id = vm.next_id++;
    vm.pools[id] = std::move(pool);
    return Ret(c, id);
}

Outcome DeleteFpl(Vm& vm, Context& c) {
    auto found = vm.pools.find(A(vm, c, 0));
    if (found == vm.pools.end()) return Ret(c, u32(KE_UNKNOWN_FPLID));
    for (u32 id : found->second.waiting) {
        Thread& t = vm.threads[id];
        t.c.r[2] = u32(KE_WAIT_DELETE);
        vm.MakeReady(t);
    }
    vm.Free(found->second.base);
    vm.pools.erase(found);
    return Ret(c, KE_OK);
}

Outcome AllocateFpl(Vm& vm, Context& c) {
    auto found = vm.pools.find(A(vm, c, 0));
    if (found == vm.pools.end()) return Ret(c, u32(KE_UNKNOWN_FPLID));
    FixedPool& pool = found->second;
    if (!pool.free.empty()) {
        const u32 block = pool.free.front();
        pool.free.pop_front();
        return Ret(c, block);
    }
    if (!vm.Block(c, Wait::Fpl, found->first)) return Ret(c, u32(KE_CAN_NOT_WAIT));
    pool.waiting.push_back(vm.current);
    return Outcome::Block;
}

Outcome PollAllocateFpl(Vm& vm, Context& c) {
    auto found = vm.pools.find(A(vm, c, 0));
    if (found == vm.pools.end()) return Ret(c, u32(KE_UNKNOWN_FPLID));
    FixedPool& pool = found->second;
    if (pool.free.empty()) return Ret(c, u32(KE_NO_MEMORY));
    const u32 block = pool.free.front();
    pool.free.pop_front();
    return Ret(c, block);
}

Outcome FreeFpl(Vm& vm, Context& c) {
    auto found = vm.pools.find(A(vm, c, 0));
    if (found == vm.pools.end()) return Ret(c, u32(KE_UNKNOWN_FPLID));
    FixedPool& pool = found->second;
    const u32 block = A(vm, c, 1);
    if (block < pool.base || block >= pool.base + pool.block_size * pool.blocks ||
        (block - pool.base) % pool.block_size != 0)
        return Ret(c, u32(KE_ILLEGAL_MEMBLOCK));
    if (!pool.waiting.empty()) {
        Thread& t = vm.threads[pool.waiting.front()];
        pool.waiting.pop_front();
        t.c.r[2] = block;                            // its AllocateFpl's result
        vm.MakeReady(t);
    } else {
        pool.free.push_back(block);
    }
    Return(c, KE_OK);
    return vm.reschedule && !vm.in_interrupt ? Outcome::Yield : Outcome::Return;
}

// ---------------------------------------------------------------- memory, C library

Outcome AllocSysMemory(Vm& vm, Context& c) { return Ret(c, vm.Allocate(A(vm, c, 1))); }
Outcome FreeSysMemory(Vm& vm, Context& c) { vm.Free(A(vm, c, 0)); return Ret(c, KE_OK); }
Outcome QueryMemSize(Vm&, Context& c) { return Ret(c, kRamSize); }
Outcome QueryMaxFreeMemSize(Vm& vm, Context& c) { return Ret(c, vm.LargestFree()); }

Outcome Memcpy(Vm& vm, Context& c) {
    const u32 d = A(vm, c, 0), s = A(vm, c, 1), n = A(vm, c, 2);
    if (n) std::memmove(vm.At(d, n), vm.At(s, n), n);
    return Ret(c, d);
}

Outcome Bcopy(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0), d = A(vm, c, 1), n = A(vm, c, 2);
    if (n) std::memmove(vm.At(d, n), vm.At(s, n), n);
    return Ret(c, 0);
}

Outcome Memset(Vm& vm, Context& c) {
    const u32 d = A(vm, c, 0), n = A(vm, c, 2);
    if (n) std::memset(vm.At(d, n), int(A(vm, c, 1) & 0xFF), n);
    return Ret(c, d);
}

Outcome Bzero(Vm& vm, Context& c) {
    const u32 d = A(vm, c, 0), n = A(vm, c, 1);
    if (n) std::memset(vm.At(d, n), 0, n);
    return Ret(c, 0);
}

Outcome Memcmp(Vm& vm, Context& c) {
    const u32 n = A(vm, c, 2);
    return Ret(c, n ? u32(std::memcmp(vm.At(A(vm, c, 0), n), vm.At(A(vm, c, 1), n), n)) : 0);
}

Outcome Strlen(Vm& vm, Context& c) { return Ret(c, u32(vm.String(A(vm, c, 0), 1u << 20).size())); }

Outcome Strcpy(Vm& vm, Context& c) {
    const std::string s = vm.String(A(vm, c, 1), 1u << 20);
    std::memcpy(vm.At(A(vm, c, 0), u32(s.size()) + 1), s.c_str(), s.size() + 1);
    return Ret(c, A(vm, c, 0));
}

Outcome Strncpy(Vm& vm, Context& c) {
    const u32 d = A(vm, c, 0), n = A(vm, c, 2);
    const std::string s = vm.String(A(vm, c, 1), n);
    for (u32 i = 0; i < n; ++i) vm.Write(d + i, i < s.size() ? u8(s[i]) : 0, 1);
    return Ret(c, d);
}

Outcome Strcmp(Vm& vm, Context& c) {
    return Ret(c, u32(std::strcmp(vm.String(A(vm, c, 0), 1u << 20).c_str(),
                                  vm.String(A(vm, c, 1), 1u << 20).c_str())));
}

Outcome Strncmp(Vm& vm, Context& c) {
    const u32 n = A(vm, c, 2);
    return Ret(c, u32(std::strncmp(vm.String(A(vm, c, 0), n).c_str(),
                                   vm.String(A(vm, c, 1), n).c_str(), n)));
}

Outcome Strcat(Vm& vm, Context& c) {
    const u32 d = A(vm, c, 0);
    const std::string joined = vm.String(d, 1u << 20) + vm.String(A(vm, c, 1), 1u << 20);
    std::memcpy(vm.At(d, u32(joined.size()) + 1), joined.c_str(), joined.size() + 1);
    return Ret(c, d);
}

Outcome Strchr(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0);
    const char want = char(A(vm, c, 1));
    const std::string text = vm.String(s, 1u << 20);
    if (!want) return Ret(c, s + u32(text.size()));
    const std::size_t at = text.find(want);
    return Ret(c, at == std::string::npos ? 0 : s + u32(at));
}

Outcome Strrchr(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0);
    const std::string text = vm.String(s, 1u << 20);
    const std::size_t at = text.rfind(char(A(vm, c, 1)));
    return Ret(c, at == std::string::npos ? 0 : s + u32(at));
}

Outcome Tolower(Vm& vm, Context& c) { (void)vm; return Ret(c, u32(std::tolower(int(c.r[4] & 0xFF)))); }
Outcome Toupper(Vm& vm, Context& c) { (void)vm; return Ret(c, u32(std::toupper(int(c.r[4] & 0xFF)))); }

Outcome Strtol(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0), end = A(vm, c, 1);
    const std::string text = vm.String(s, 256);
    char* stop = nullptr;
    const long value = std::strtol(text.c_str(), &stop, int(A(vm, c, 2)));
    if (end) vm.Write(end, s + u32(stop - text.c_str()), 4);
    return Ret(c, u32(value));
}

Outcome Strtoul(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0), end = A(vm, c, 1);
    const std::string text = vm.String(s, 256);
    char* stop = nullptr;
    // 32-bit unsigned long on the IOP: out of range saturates there, as strtoul's does.
    const unsigned long long value = std::strtoull(text.c_str(), &stop, int(A(vm, c, 2)));
    if (end) vm.Write(end, s + u32(stop - text.c_str()), 4);
    const bool negative = text.find_first_not_of(" \t\n\v\f\r") != std::string::npos &&
                          text[text.find_first_not_of(" \t\n\v\f\r")] == '-';
    u32 result;
    if (negative) result = u32(value);               // strtoul negates in the unsigned type
    else result = value > 0xFFFFFFFFull ? 0xFFFFFFFFu : u32(value);
    return Ret(c, result);
}

Outcome Strstr(Vm& vm, Context& c) {
    const u32 s = A(vm, c, 0);
    const std::string text = vm.String(s, 1u << 20), want = vm.String(A(vm, c, 1), 1u << 20);
    const std::size_t at = text.find(want);
    return Ret(c, at == std::string::npos ? 0 : s + u32(at));
}

// Arguments of a variadic call: from argument `first` on.
std::function<u32()> Varargs(Vm& vm, Context& c, int first) {
    auto index = std::make_shared<int>(first);
    return [&vm, &c, index]() { return Arg(vm, c, (*index)++); };
}

// Arguments from a va_list: on the IOP, a pointer to consecutive words.
std::function<u32()> VaList(Vm& vm, u32 list) {
    auto at = std::make_shared<u32>(list);
    return [&vm, at]() { const u32 v = vm.Read(*at, 4); *at += 4; return v; };
}

Outcome Sprintf(Vm& vm, Context& c) {
    const std::string text = Format(vm, A(vm, c, 1), Varargs(vm, c, 2));
    std::memcpy(vm.At(A(vm, c, 0), u32(text.size()) + 1), text.c_str(), text.size() + 1);
    return Ret(c, u32(text.size()));
}

Outcome Vsprintf(Vm& vm, Context& c) {
    const std::string text = Format(vm, A(vm, c, 1), VaList(vm, A(vm, c, 2)));
    std::memcpy(vm.At(A(vm, c, 0), u32(text.size()) + 1), text.c_str(), text.size() + 1);
    return Ret(c, u32(text.size()));
}

Outcome Printf(Vm& vm, Context& c) {
    const std::string text = Format(vm, A(vm, c, 0), Varargs(vm, c, 1));
    Log(text);
    return Ret(c, u32(text.size()));
}

Outcome Puts(Vm& vm, Context& c) {
    Log(vm.String(A(vm, c, 0)));
    return Ret(c, 0);
}

// ---------------------------------------------------------------- interrupts

Outcome CpuSuspendIntr(Vm& vm, Context& c) {
    if (const u32 state = A(vm, c, 0)) vm.Write(state, 0, 4);
    return Ret(c, KE_OK);
}

Outcome Succeed(Vm&, Context& c) { return Ret(c, KE_OK); }

Outcome QueryIntrContext(Vm& vm, Context& c) { return Ret(c, vm.in_interrupt ? 1 : 0); }

// ---------------------------------------------------------------- SIF

// sceSifSetCmdBuffer(table, entries): where a module's own command handlers are
// listed. The host keeps handlers itself (command_handlers), so only the
// previous table is answered.
Outcome SetCmdBuffer(Vm& vm, Context& c) {
    const u32 previous = vm.handler_table;
    vm.handler_table = A(vm, c, 0);
    return Ret(c, previous);
}

Outcome AddCmdHandler(Vm& vm, Context& c) {
    vm.command_handlers[A(vm, c, 0)] = {A(vm, c, 1), A(vm, c, 2)};
    return Ret(c, KE_OK);
}

Outcome RemoveCmdHandler(Vm& vm, Context& c) {
    vm.command_handlers.erase(A(vm, c, 0));
    return Ret(c, KE_OK);
}

// sceSifSendCmd(command, packet, packet_size, src, dest, size): the packet goes
// to the EE's receive buffer and `size` bytes from IOP `src` to EE `dest`.
Outcome SendCmd(Vm& vm, Context& c) {
    const u32 command = A(vm, c, 0), packet = A(vm, c, 1), psize = A(vm, c, 2);
    const u32 src = A(vm, c, 3), dest = A(vm, c, 4), size = A(vm, c, 5);
    Memory& ee = GuestMemory();
    for (u32 i = 0; i < size; ++i) ee.Write(dest + i, vm.Read(src + i, 1), 1);
    vm.Write(packet, (psize & 0xFF) | (size << 8), 4);
    vm.Write(packet + 4, dest, 4);
    vm.Write(packet + 8, command, 4);
    std::vector<u8> bytes(psize);
    std::memcpy(bytes.data(), vm.At(packet, psize), psize);
    if (vm.trace) {
        std::string words;
        for (u32 i = 16; i + 4 <= psize && i < 48; i += 4) {
            char w[12];
            std::snprintf(w, sizeof w, " %08X", unsigned(vm.Read(packet + i, 4)));
            words += w;
        }
        std::fprintf(stderr, "[iop] to the EE: command %u,%s\n", unsigned(command), words.c_str());
    }
    QueueIopPacket(std::move(bytes));
    return Ret(c, 1);
}

// sceSifSetDma(transfers, count): IOP memory to EE memory, each
// { src, dest, size, attr }; done at once.
Outcome SetDma(Vm& vm, Context& c) {
    const u32 list = A(vm, c, 0), count = A(vm, c, 1);
    Memory& ee = GuestMemory();
    for (u32 k = 0; k < count; ++k) {
        const u32 src = vm.Read(list + 16 * k, 4), dest = vm.Read(list + 16 * k + 4, 4);
        const u32 size = vm.Read(list + 16 * k + 8, 4);
        for (u32 i = 0; i < size; ++i) ee.Write(dest + i, vm.Read(src + i, 1), 1);
    }
    return Ret(c, 1);
}

// sceSifSetRpcQueue(queue, thread): the thread that serves the queue.
Outcome SetRpcQueue(Vm& vm, Context& c) {
    vm.rpc_queues[A(vm, c, 0)] = A(vm, c, 1);
    return Ret(c, KE_OK);
}

// sceSifRegisterRpc(server, sid, function, buffer, cfunction, cbuffer, queue)
Outcome RegisterRpc(Vm& vm, Context& c) {
    RpcServerEntry e;
    e.server = A(vm, c, 0);
    e.sid = A(vm, c, 1);
    e.function = A(vm, c, 2);
    e.buffer = A(vm, c, 3);
    e.cfunction = A(vm, c, 4);
    e.cbuffer = A(vm, c, 5);
    e.queue = A(vm, c, 6);
    vm.rpc_servers[e.server] = e;
    char text[128];
    std::snprintf(text, sizeof text, "IOP: a module serves RPC 0x%08X (function 0x%08X)",
                  unsigned(e.sid), unsigned(e.function));
    Note("iop rpc " + std::to_string(e.sid), text);
    return Ret(c, KE_OK);
}

// sceSifRpcLoop(queue): the thread serves requests for ever. It waits here;
// each request runs the server's function in it (ServeNext), and returning
// from that sends the answer and waits for the next (iop_vm.cpp).
Outcome RpcLoop(Vm& vm, Context& c) {
    Thread* t = vm.Current();
    if (!t) return Ret(c, u32(KE_CAN_NOT_WAIT));
    t->rpc_queue = A(vm, c, 0);
    vm.rpc_queues[t->rpc_queue] = t->id;
    vm.Block(c, Wait::Rpc, t->rpc_queue);
    return Outcome::Block;          // the scheduler starts serving (ServeNext)
}

// sceSifSetDmaIntr(transfers, count, callback, data): sceSifSetDma, then the
// callback once the transfer is done - as an interrupt, after this returns.
Outcome SetDmaIntr(Vm& vm, Context& c) {
    const u32 callback = A(vm, c, 2), data = A(vm, c, 3);
    SetDma(vm, c);
    if (callback) vm.completions.push_back({callback, data});
    return Ret(c, 1);
}

Outcome DmaStat(Vm&, Context& c) { return Ret(c, u32(-1)); }    // < 0: finished

// ---------------------------------------------------------------- cdvdman
// The disc, read from the user's image. Reads finish at once.

// sceCdCallback's function, and the commands done since it last ran: it runs
// as the CD/DVD interrupt's, with the command (SCECdFuncRead = 1), when the
// scheduler next looks (CdInterrupts).
struct CdState {
    u32 callback = 0;
    std::vector<u32> done;
};
CdState& Cd() {
    static CdState state;
    return state;
}
constexpr u32 kCdFuncRead = 1;

// sceCdRead(lsn, sectors, buffer, mode): mode points at sceCdRMode
// { u8 trycount, spindlctrl, datapattern, pad }; the pattern picks 2048, 2328
// or 2340 bytes a sector.
Outcome CdRead(Vm& vm, Context& c) {
    const u32 lsn = A(vm, c, 0), sectors = A(vm, c, 1), buffer = A(vm, c, 2), mode = A(vm, c, 3);
    const u32 pattern = mode ? *vm.At(mode + 2, 1) : 0;
    std::vector<u8> data(std::size_t(sectors) * Disc::PatternSize(pattern));
    if (!GuestDisc().IsOpen() || !GuestDisc().ReadPattern(lsn, sectors, pattern, data.data()))
        return Ret(c, 0);
    std::memcpy(vm.At(buffer, u32(data.size())), data.data(), data.size());
    if (Cd().callback) Cd().done.push_back(kCdFuncRead);
    return Ret(c, 1);
}

// sceCdTrayReq(mode, u32* traycnt): 2 (SCECdTrayCheck) asks whether the tray
// moved since last asked - never, the image stays in: 0 in *traycnt. 0 and 1
// (open, close) are noted and done as if (the image stays in all the same).
// 1: done.
Outcome CdTrayReq(Vm& vm, Context& c) {
    const u32 mode = A(vm, c, 0), traycnt = A(vm, c, 1);
    if (mode < 2) Note("cdvdman tray " + std::to_string(mode), std::string("IOP: sceCdTrayReq asks the tray to ") +
                       (mode ? "close" : "open") + "; the disc image stays in");
    if (traycnt) vm.Write(traycnt, 0, 4);
    return Ret(c, 1);
}

// sceCdCallback(function): the function called as each command ends; gives
// the one set before.
Outcome CdCallback(Vm& vm, Context& c) {
    const u32 before = Cd().callback;
    Cd().callback = A(vm, c, 0);
    if (!Cd().callback) Cd().done.clear();
    return Ret(c, before);
}

Outcome CdSync(Vm&, Context& c) { return Ret(c, 0); }              // done
Outcome CdGetError(Vm&, Context& c) { return Ret(c, 0); }          // SCECdErNO
Outcome CdGetDiskType(Vm&, Context& c) {                          // SCECdPS2CD, SCECdPS2DVD
    return Ret(c, GuestDisc().IsCd() ? 0x12 : 0x14);
}
Outcome CdDiskReady(Vm&, Context& c) { return Ret(c, 2); }         // SCECdComplete
// sceCdSearchFile(sceCdlFILE* fp, const char* name): the file's place on the
// disc, in fp = { u32 lsn; u32 size; char name[16]; u8 date[8]; }. The name as
// a module writes it, "\MOVIES\INTRO.XAV;1" - here a device ("cdrom0:")
// before it is skipped, case does not matter and ";1" may be left out
// (Disc::Find). fp's name is the directory record's identifier, its date the
// record's as cdvdman lays it out: [1] second, [2] minute, [3] hour, [4] day,
// [5] month, [6..7] the year (TO RECHECK against a console's: XAVSTRM reads
// the lsn and size). 1: found, 0: not.
Outcome CdSearchFile(Vm& vm, Context& c) {
    const u32 fp = A(vm, c, 0);
    std::string name = vm.String(A(vm, c, 1));
    const auto colon = name.find(':');
    if (colon != std::string::npos) name.erase(0, colon + 1);
    Disc::File file;
    if (!GuestDisc().IsOpen() || !GuestDisc().Find(name, &file) || file.name.empty()) return Ret(c, 0);
    u8* out = vm.At(fp, 32);
    std::memset(out, 0, 32);
    std::memcpy(out, &file.lsn, 4);
    std::memcpy(out + 4, &file.size, 4);
    std::memcpy(out + 8, file.name.data(), std::min<std::size_t>(file.name.size(), 15));
    const u32 year = 1900u + file.date[0];
    const u8 date[8] = {0, file.date[5], file.date[4], file.date[3], file.date[2], file.date[1],
                        u8(year), u8(year >> 8)};
    std::memcpy(out + 24, date, 8);
    return Ret(c, 1);
}

// sceCdMmode(media): the media the drive is told to expect (CD, DVD); the host
// reads the disc image as it is. 1: done.
Outcome CdMmode(Vm&, Context& c) { return Ret(c, 1); }

Outcome FlushDcache(Vm&, Context& c) { return Ret(c, 0); }
Outcome CheckInit(Vm&, Context& c) { return Ret(c, 1); }
// sceSifGetSreg(index), sceSifSetSreg(index, value): the SIF software
// registers the EE sets with command 0x80000001 - a module waiting on one
// polls it (Stuntman's XAVSTRM.IRX on 31, as its movie starts).
Outcome GetSreg(Vm& vm, Context& c) { return Ret(c, SoftwareRegister(A(vm, c, 0))); }
Outcome SetSreg(Vm& vm, Context& c) {
    const u32 index = A(vm, c, 0), before = SoftwareRegister(index);
    SetSoftwareRegister(index, A(vm, c, 1));
    return Ret(c, before);
}

// ---------------------------------------------------------------- libsd

struct Sound {
    Spu2 spu;
    struct Handler { u32 function = 0, arg = 0; };
    Handler transfer[2], spu2;
    // The older API (libsd 2.x): a callback per transfer channel and one for
    // the SPU2's interrupt, each called with 0 - when no handler of the newer
    // kind (sceSdSetTransIntrHandler, sceSdSetSpu2IntrHandler) is set.
    u32 transfer_callback[2] = {0, 0}, irq_callback = 0;
    struct Block {
        bool active = false;
        u32 address = 0, size = 0, half = 0;
        u64 period_us = 0, due_us = 0;
    } block[2];
    std::vector<u32> pending;       // channels whose transfer interrupt is due now
};

Sound& S() {
    static Sound sound;
    return sound;
}

Outcome SdInit(Vm&, Context& c) { return Ret(c, KE_OK); }

Outcome SdSetParam(Vm&, Context& c) { S().spu.SetParam(c.r[4] & 0xFFFF, c.r[5] & 0xFFFF); return Ret(c, 0); }
Outcome SdGetParam(Vm&, Context& c) { return Ret(c, S().spu.GetParam(c.r[4] & 0xFFFF)); }
Outcome SdSetSwitch(Vm&, Context& c) { S().spu.SetSwitch(c.r[4] & 0xFFFF, c.r[5]); return Ret(c, 0); }
Outcome SdGetSwitch(Vm&, Context& c) { return Ret(c, S().spu.GetSwitch(c.r[4] & 0xFFFF)); }
Outcome SdSetAddr(Vm&, Context& c) { S().spu.SetAddr(c.r[4] & 0xFFFF, c.r[5]); return Ret(c, 0); }
Outcome SdGetAddr(Vm&, Context& c) { return Ret(c, S().spu.GetAddr(c.r[4] & 0xFFFF)); }
Outcome SdSetCoreAttr(Vm&, Context& c) { S().spu.SetCoreAttr(c.r[4] & 0xFFFF, c.r[5] & 0xFFFF); return Ret(c, 0); }
Outcome SdGetCoreAttr(Vm&, Context& c) { return Ret(c, S().spu.GetCoreAttr(c.r[4] & 0xFFFF)); }

// Pitch: 0x1000 plays a sample at its recorded rate; a semitone is 2^(1/12).
Outcome SdNote2Pitch(Vm&, Context& c) {
    const double semitones = (double(s32(c.r[6] & 0xFFFF)) - double(c.r[4] & 0xFFFF)) +
                             (double(s16(c.r[7] & 0xFFFF)) - double(c.r[5] & 0xFFFF)) / 128.0;
    const double pitch = 4096.0 * std::pow(2.0, semitones / 12.0);
    return Ret(c, u32(pitch > 0x3FFF ? 0x3FFF : pitch));
}

Outcome SdPitch2Note(Vm&, Context& c) {
    const double pitch = double(c.r[6] & 0xFFFF);
    if (pitch <= 0) return Ret(c, 0);
    const double semitones = 12.0 * std::log2(pitch / 4096.0) + double(c.r[4] & 0xFFFF);
    const s32 note = s32(std::floor(semitones));
    const s32 fine = s32((semitones - note) * 128.0);
    return Ret(c, u32((note << 8) | (fine & 0xFF)));
}

u32 Batch(u32 func, u32 entry, u32 value) {
    Spu2& s = S().spu;
    switch (func) {
        case 0x01: s.SetParam(entry, value & 0xFFFF); return 0;
        case 0x02: s.SetSwitch(entry, value); return 0;
        case 0x03: s.SetAddr(entry, value); return 0;
        case 0x04: s.SetCoreAttr(entry, value & 0xFFFF); return 0;
        case 0x11: return s.GetParam(entry);
        case 0x12: return s.GetSwitch(entry);
        case 0x13: return s.GetAddr(entry);
        case 0x14: return s.GetCoreAttr(entry);
        default: throw Unimplemented{"IOP: sceSdProcBatch function " + std::to_string(func)};
    }
}

Outcome SdProcBatch(Vm& vm, Context& c) {
    const u32 batch = A(vm, c, 0), rets = A(vm, c, 1), count = A(vm, c, 2);
    for (u32 i = 0; i < count; ++i) {
        const u32 head = vm.Read(batch + 8 * i, 4), value = vm.Read(batch + 8 * i + 4, 4);
        const u32 result = Batch(head & 0xFFFF, head >> 16, value);
        if (rets) vm.Write(rets + 4 * i, result, 4);
    }
    return Ret(c, count);
}

Outcome SdProcBatchEx(Vm& vm, Context& c) {
    const u32 batch = A(vm, c, 0), rets = A(vm, c, 1), count = A(vm, c, 2), voices = A(vm, c, 3);
    u32 done = 0;
    for (u32 i = 0; i < count; ++i) {
        const u32 head = vm.Read(batch + 8 * i, 4), value = vm.Read(batch + 8 * i + 4, 4);
        for (u32 v = 0; v < 24; ++v) {
            if (!(voices & (1u << v))) continue;
            // The entry's voice field is the low bits of the parameter number.
            const u32 result = Batch(head & 0xFFFF, (head >> 16) | (v << 1), value);
            if (rets) vm.Write(rets + 4 * done, result, 4);
            ++done;
        }
    }
    return Ret(c, done);
}

// sceSdVoiceTrans(channel, mode, iop_address, spu_address, size): into SPU RAM
// at once; the channel's transfer interrupt follows, as the DMA's would.
Outcome SdVoiceTrans(Vm& vm, Context& c) {
    const u32 channel = c.r[4] & 1, mode = c.r[5] & 0xFFFF, iop = A(vm, c, 2), spu = A(vm, c, 3);
    const u32 size = A(vm, c, 4);
    Sound& s = S();
    if ((mode & 3) == 0) {
        if (size) s.spu.Write(spu, vm.At(iop, size), size);
    } else if ((mode & 3) == 1) {
        if (size) s.spu.Read(spu, vm.At(iop, size), size);
    }
    // Only a DMA transfer interrupts when done; SD_TRANS_MODE_IO (8) is the CPU
    // writing through the SPU2's registers, finished when the call returns.
    if (!(mode & 8)) s.pending.push_back(channel);
    // As libsd answers (ps2sdk voice.c): DMA the size, IO-mode writes 0.
    return Ret(c, (mode & 8) ? 0 : size);
}

Outcome SdVoiceTransStatus(Vm&, Context& c) { return Ret(c, 1); }     // complete

// sceSdBlockTrans(channel, mode, iop_address, size, ...): with SD_TRANS_LOOP,
// the hardware streams the buffer's two halves in turn and interrupts after
// each, at the output rate (48 kHz, 16-bit stereo: 192,000 bytes a second).
Outcome SdBlockTrans(Vm& vm, Context& c) {
    const u32 channel = c.r[4] & 1, mode = c.r[5] & 0xFFFF, address = A(vm, c, 2), size = A(vm, c, 3);
    Sound::Block& b = S().block[channel];
    if ((mode & 3) == 2) {                   // SD_TRANS_STOP
        b.active = false;
        return Ret(c, 0);
    }
    b.active = true;
    b.address = address;
    b.size = size;
    b.half = 0;
    const u32 bytes = (mode & 0x10) ? size / 2 : size;
    b.period_us = u64(bytes) * 1000000 / 192000;
    b.due_us = vm.NowUs() + b.period_us;
    return Ret(c, 0);
}

Outcome SdBlockTransStatus(Vm&, Context& c) {
    const Sound::Block& b = S().block[c.r[4] & 1];
    return Ret(c, (b.half << 24) | ((b.address + b.half * (b.size / 2)) & 0xFFFFFF));
}

Outcome SdStopTrans(Vm&, Context& c) {
    S().block[c.r[4] & 1].active = false;
    return Ret(c, 0);
}

Outcome SdSetTransIntrHandler(Vm&, Context& c) {
    Sound::Handler& h = S().transfer[c.r[4] & 1];
    const u32 previous = h.function;
    h = {c.r[5], c.r[6]};
    return Ret(c, previous);
}

// sceSdSetTransCallback(channel, function), sceSdSetIRQCallback(function):
// the previous one back.
Outcome SdSetTransCallback(Vm&, Context& c) {
    u32& f = S().transfer_callback[c.r[4] & 1];
    const u32 previous = f;
    f = u32(c.r[5]);
    return Ret(c, previous);
}

Outcome SdSetIrqCallback(Vm&, Context& c) {
    u32& f = S().irq_callback;
    const u32 previous = f;
    f = u32(c.r[4]);
    return Ret(c, previous);
}

Outcome SdSetSpu2IntrHandler(Vm&, Context& c) {
    const u32 previous = S().spu2.function;
    S().spu2 = {c.r[4], c.r[5]};
    return Ret(c, previous);
}

Outcome SdGetTransIntrHandlerArgument(Vm&, Context& c) { return Ret(c, S().transfer[c.r[4] & 1].arg); }
Outcome SdGetSpu2IntrHandlerArgument(Vm&, Context& c) { return Ret(c, S().spu2.arg); }
Outcome SdReturnsPrevious(Vm&, Context& c) { return Ret(c, 0); }

// sceSdEffectAttr: int core, int mode, short depth_L, short depth_R, int delay, int feedback.
Outcome SdSetEffectAttr(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 1);
    const s32 depth_left = s32(std::int16_t(vm.Read(p + 8, 2))), depth_right = s32(std::int16_t(vm.Read(p + 10, 2)));
    return Ret(c, u32(S().spu.SetEffectAttr(c.r[4] & 1, vm.Read(p + 4, 4), depth_left, depth_right,
                                            vm.Read(p + 12, 4), vm.Read(p + 16, 4))));
}

Outcome SdGetEffectAttr(Vm& vm, Context& c) {
    const u32 p = A(vm, c, 1);
    u32 mode = 0, delay = 0, feedback = 0;
    s32 depth_left = 0, depth_right = 0;
    S().spu.GetEffectAttr(c.r[4] & 1, &mode, &depth_left, &depth_right, &delay, &feedback);
    vm.Write(p, c.r[4] & 1, 4);
    vm.Write(p + 4, mode, 4);
    vm.Write(p + 8, u32(depth_left) & 0xFFFF, 2);
    vm.Write(p + 10, u32(depth_right) & 0xFFFF, 2);
    vm.Write(p + 12, delay, 4);
    vm.Write(p + 16, feedback, 4);
    return Ret(c, 0);
}

const Map& Table() {
    static const Map table = {
        {{"sysmem", 4}, AllocSysMemory}, {{"sysmem", 5}, FreeSysMemory},
        {{"sysmem", 6}, QueryMemSize}, {{"sysmem", 7}, QueryMaxFreeMemSize},
        {{"sysmem", 8}, QueryMaxFreeMemSize},
        {{"sysmem", 14}, Printf},                                    // Kprintf
        {{"stdio", 4}, Printf}, {{"stdio", 7}, Puts},
        {{"sysclib", 7}, Tolower}, {{"sysclib", 6}, Toupper},
        {{"sysclib", 11}, Memcmp}, {{"sysclib", 12}, Memcpy}, {{"sysclib", 13}, Memcpy},
        {{"sysclib", 14}, Memset}, {{"sysclib", 15}, Memcmp}, {{"sysclib", 16}, Bcopy},
        {{"sysclib", 17}, Bzero}, {{"sysclib", 19}, Sprintf}, {{"sysclib", 20}, Strcat},
        {{"sysclib", 21}, Strchr}, {{"sysclib", 22}, Strcmp}, {{"sysclib", 23}, Strcpy},
        {{"sysclib", 27}, Strlen}, {{"sysclib", 29}, Strncmp}, {{"sysclib", 30}, Strncpy},
        {{"sysclib", 32}, Strrchr}, {{"sysclib", 34}, Strstr}, {{"sysclib", 36}, Strtol},
        {{"sysclib", 38}, Strtoul}, {{"sysclib", 42}, Vsprintf},
        {{"intrman", 4}, Succeed}, {{"intrman", 5}, Succeed}, {{"intrman", 6}, Succeed},
        {{"intrman", 7}, Succeed}, {{"intrman", 8}, Succeed}, {{"intrman", 9}, Succeed},
        {{"intrman", 17}, CpuSuspendIntr}, {{"intrman", 18}, Succeed},
        {{"intrman", 23}, QueryIntrContext},
        {{"thbase", 4}, CreateThread}, {{"thbase", 5}, DeleteThread},
        {{"thbase", 6}, StartThread}, {{"thbase", 7}, StartThreadArgs},
        {{"thbase", 8}, ExitThread}, {{"thbase", 10}, TerminateThread},
        {{"thbase", 11}, TerminateThread}, {{"thbase", 14}, ChangeThreadPriority},
        {{"thbase", 15}, ChangeThreadPriority}, {{"thbase", 16}, RotateThreadReadyQueue},
        {{"thbase", 20}, GetThreadId}, {{"thbase", 21}, CheckThreadStack},
        {{"thbase", 24}, SleepThread},
        {{"thbase", 25}, WakeupThread}, {{"thbase", 26}, WakeupThread},
        {{"thbase", 27}, CancelWakeupThread}, {{"thbase", 28}, CancelWakeupThread},
        {{"thbase", 33}, DelayThread}, {{"thbase", 34}, GetSystemTime},
        {{"thbase", 35}, SetAlarm}, {{"thbase", 36}, SetAlarm},
        {{"thbase", 37}, CancelAlarm}, {{"thbase", 38}, CancelAlarm},
        {{"thbase", 39}, USec2SysClock}, {{"thbase", 40}, SysClock2USec},
        {{"thbase", 42}, GetThreadCurrentPriority}, {{"thbase", 43}, GetSystemTimeLow},
        {{"thsemap", 4}, CreateSema}, {{"thsemap", 5}, DeleteSema},
        {{"thsemap", 6}, SignalSema}, {{"thsemap", 7}, SignalSema},
        {{"thsemap", 8}, WaitSema}, {{"thsemap", 9}, PollSema},
        {{"thsemap", 11}, ReferSemaStatus}, {{"thsemap", 12}, ReferSemaStatus},
        {{"thevent", 4}, CreateEventFlag}, {{"thevent", 5}, DeleteEventFlag},
        {{"thevent", 6}, SetEventFlag}, {{"thevent", 7}, SetEventFlag},
        {{"thevent", 8}, ClearEventFlag}, {{"thevent", 9}, ClearEventFlag},
        {{"thevent", 10}, WaitEventFlag}, {{"thevent", 11}, PollEventFlag},
        {{"thevent", 13}, ReferEventFlagStatus}, {{"thevent", 14}, ReferEventFlagStatus},
        {{"sifman", 5}, Succeed}, {{"sifman", 7}, SetDma}, {{"sifman", 8}, DmaStat},
        {{"sifman", 29}, CheckInit},
        {{"sifcmd", 4}, Succeed}, {{"sifcmd", 5}, Succeed}, {{"sifcmd", 6}, GetSreg},
        {{"sifcmd", 7}, SetSreg}, {{"sifcmd", 8}, SetCmdBuffer}, {{"sifcmd", 9}, Succeed},
        {{"sifcmd", 10}, AddCmdHandler}, {{"sifcmd", 11}, RemoveCmdHandler},
        {{"sifcmd", 12}, SendCmd}, {{"sifcmd", 13}, SendCmd}, {{"sifcmd", 14}, Succeed},
        {{"sifcmd", 17}, RegisterRpc}, {{"sifcmd", 19}, SetRpcQueue}, {{"sifcmd", 22}, RpcLoop},
        {{"sifman", 32}, SetDmaIntr},
        {{"cdvdman", 4}, Succeed}, {{"cdvdman", 6}, CdRead}, {{"cdvdman", 8}, CdGetError},
        {{"cdvdman", 10}, CdSearchFile}, {{"cdvdman", 11}, CdSync}, {{"cdvdman", 12}, CdGetDiskType}, {{"cdvdman", 13}, CdDiskReady},
        {{"cdvdman", 75}, CdMmode}, {{"cdvdman", 37}, CdCallback}, {{"cdvdman", 14}, CdTrayReq},
        {{"thfpool", 4}, CreateFpl}, {{"thfpool", 5}, DeleteFpl}, {{"thfpool", 6}, AllocateFpl},
        {{"thfpool", 7}, PollAllocateFpl}, {{"thfpool", 8}, PollAllocateFpl}, {{"thfpool", 9}, FreeFpl},
        {{"loadcore", 5}, FlushDcache},
        {{"libsd", 2}, SdInit}, {{"libsd", 4}, SdInit},
        {{"libsd", 5}, SdSetParam}, {{"libsd", 6}, SdGetParam},
        {{"libsd", 7}, SdSetSwitch}, {{"libsd", 8}, SdGetSwitch},
        {{"libsd", 9}, SdSetAddr}, {{"libsd", 10}, SdGetAddr},
        {{"libsd", 11}, SdSetCoreAttr}, {{"libsd", 12}, SdGetCoreAttr},
        {{"libsd", 13}, SdNote2Pitch}, {{"libsd", 14}, SdPitch2Note},
        {{"libsd", 15}, SdProcBatch}, {{"libsd", 16}, SdProcBatchEx},
        {{"libsd", 17}, SdVoiceTrans}, {{"libsd", 18}, SdBlockTrans},
        {{"libsd", 19}, SdVoiceTransStatus}, {{"libsd", 20}, SdBlockTransStatus},
        {{"libsd", 21}, SdSetTransCallback}, {{"libsd", 22}, SdSetIrqCallback},
        {{"libsd", 23}, SdSetEffectAttr}, {{"libsd", 24}, SdGetEffectAttr},
        {{"libsd", 25}, Succeed}, {{"libsd", 26}, SdSetTransIntrHandler},
        {{"libsd", 27}, SdSetSpu2IntrHandler}, {{"libsd", 28}, SdGetTransIntrHandlerArgument},
        {{"libsd", 29}, SdGetSpu2IntrHandlerArgument}, {{"libsd", 30}, SdStopTrans},
        {{"libsd", 31}, Succeed}, {{"libsd", 32}, Succeed}, {{"libsd", 33}, Succeed},
    };
    return table;
}

std::map<std::pair<std::string, u32>, std::string> LoadNames() {
    std::map<std::pair<std::string, u32>, std::string> names;
    // engine/data/iop_exports.csv, built in (engine/host/CMakeLists.txt).
    static const char kExports[] =
#include "iop_exports.inc"
        ;
    std::istringstream in(kExports);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("library,", 0) == 0) continue;
        std::stringstream fields(line);
        std::string library, index, name;
        std::getline(fields, library, ',');
        std::getline(fields, index, ',');
        std::getline(fields, name, ',');
        names[{library, u32(std::stoul(index))}] = name;
    }
    return names;
}

}  // namespace

Library FindLibrary(const std::string& library, u32 index) {
    const auto found = Table().find({library, index});
    return found == Table().end() ? Library() : found->second;
}

std::string ExportName(const std::string& library, u32 index) {
    static const auto names = LoadNames();
    const auto found = names.find({library, index});
    return found == names.end() ? "?" : found->second;
}

void EnableSound() {
    S().spu.SetOutput([](const Spu2::s16* frames, u32 count) { AudioFrames(frames, count); });
}

// Due sound interrupts: a voice transfer's completion, a block transfer's
// half-buffer. Called by the interpreter's scheduler (iop_vm.cpp).
void SoundInterrupts(Vm& vm) {
    Sound& s = S();
    std::vector<u32> due;
    due.swap(s.pending);
    const u64 now = vm.NowUs();
    for (u32 ch = 0; ch < 2; ++ch) {
        Sound::Block& b = s.block[ch];
        if (b.active && b.period_us && now >= b.due_us) {
            b.half ^= 1;
            b.due_us += b.period_us;
            if (b.due_us < now) b.due_us = now + b.period_us;   // fell behind: don't burst
            due.push_back(ch);
        }
    }
    for (u32 ch : due) {
        if (s.transfer[ch].function) vm.Call(s.transfer[ch].function, {ch, s.transfer[ch].arg});
        else if (s.transfer_callback[ch]) vm.Call(s.transfer_callback[ch], {0});
    }
    // The SPU2's own interrupt: playback (or a transfer) reached IRQA.
    Timed timed(Section::Spu2);
    s.spu.Advance(now, [&](u32 core) {
        if (s.spu2.function) vm.Call(s.spu2.function, {1u << core, s.spu2.arg});
        else if (s.irq_callback) vm.Call(s.irq_callback, {0});
    });
}

// Due CD/DVD interrupts: sceCdCallback's function, once a command it
// finished. Called by the interpreter's scheduler (iop_vm.cpp).
void CdInterrupts(Vm& vm) {
    std::vector<u32> done;
    done.swap(Cd().done);
    for (u32 command : done)
        if (Cd().callback) vm.Call(Cd().callback, {command});
}

std::string Format(Vm& vm, u32 format, const std::function<u32()>& next) {
    const std::string f = vm.String(format, 4096);
    std::string out;
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (f[i] != '%') {
            out += f[i];
            continue;
        }
        std::string spec = "%";
        ++i;
        while (i < f.size() && std::strchr("-+ #0", f[i])) spec += f[i++];
        if (i < f.size() && f[i] == '*') { spec += std::to_string(s32(next())); ++i; }
        while (i < f.size() && std::isdigit(static_cast<unsigned char>(f[i]))) spec += f[i++];
        if (i < f.size() && f[i] == '.') {
            spec += f[i++];
            if (i < f.size() && f[i] == '*') { spec += std::to_string(s32(next())); ++i; }
            while (i < f.size() && std::isdigit(static_cast<unsigned char>(f[i]))) spec += f[i++];
        }
        while (i < f.size() && std::strchr("hlLqjzt", f[i])) ++i;  // all words on the IOP
        if (i >= f.size()) break;
        const char conversion = f[i];
        char buffer[512];
        switch (conversion) {
            case 'd': case 'i':
                std::snprintf(buffer, sizeof buffer, (spec + "d").c_str(), s32(next())); break;
            case 'u': case 'x': case 'X': case 'o':
                std::snprintf(buffer, sizeof buffer, (spec + conversion).c_str(), unsigned(next())); break;
            case 'p':
                std::snprintf(buffer, sizeof buffer, "0x%08X", unsigned(next())); break;
            case 'c':
                std::snprintf(buffer, sizeof buffer, (spec + "c").c_str(), int(next() & 0xFF)); break;
            case 's':
                std::snprintf(buffer, sizeof buffer, (spec + "s").c_str(), vm.String(next(), 400).c_str()); break;
            case '%': std::snprintf(buffer, sizeof buffer, "%%"); break;
            default: std::snprintf(buffer, sizeof buffer, "%%%c", conversion); break;
        }
        out += buffer;
    }
    return out;
}

}  // namespace iop
}  // namespace ps2host
