/*
 * EE threads, semaphores and interrupts, as the EE kernel schedules them.
 *
 * Each guest thread is a host fiber with its own register context and native
 * stack; translated code takes its context as an argument, so it runs unchanged
 * on any of them. Scheduling follows the kernel: the ready thread with the best
 * (lowest) priority runs, first-come among equals, and keeps running until it
 * blocks (WaitSema on zero, SleepThread), yields (RotateThreadReadyQueue), exits,
 * or makes a better thread ready - there is no time slicing.
 *
 * Interrupts run the handlers the game registered, in interrupt context, at a
 * safe point: the running thread's next loop head (lifted::loop_head_hook, which
 * a spent budget reaches - the host spends it to get control there), or between
 * threads. Two sources:
 *   - DMA completion (PostDmac), posted by the transfer and delivered after the
 *     store that started it, since code goes on to set up for its interrupt;
 *   - the vertical blank, start and end, 60 times a second in real time, as the
 *     display raises it whatever the processor is doing - code busy-waiting on
 *     work a vblank handler does depends on it. When every thread waits, frames
 *     are delivered at once instead, so idle time costs nothing. Runs with an
 *     explicit --budget have no real-time vblank, so they are deterministic.
 * If no thread becomes ready even after many frames, the program is deadlocked
 * here, and the run stops with each thread's state in the report.
 *
 * A fault inside a fiber cannot unwind across fibers, so it is caught there and
 * rethrown by the scheduler, on the host's own stack.
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "synchronization.lib")

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <cstdio>
#include <exception>
#include <thread>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ps2host/host.h"
#include "ps2host/timeline.h"
#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"
#include "ps2host/profile.h"

namespace ps2host {

namespace {

enum class State { Dormant, Ready, Running, Waiting, Sleeping };

const char* StateName(State s) {
    switch (s) {
        case State::Dormant: return "dormant";
        case State::Ready: return "ready";
        case State::Running: return "running";
        case State::Waiting: return "waiting on a semaphore";
        case State::Sleeping: return "sleeping";
    }
    return "?";
}

struct Thread {
    u32 id = 0;
    u32 entry = 0, stack = 0, stack_size = 0, gp = 0;
    s32 priority = 0;
    State state = State::Dormant;
    u64 queued = 0;                 // order among ready threads of equal priority
    u32 waiting_on = 0;             // the semaphore, while Waiting
    s32 wakeups = 0;                // WakeupThread calls not yet consumed by a sleep
    u64 runs = 0;                   // times the scheduler switched to it, for the report
    lifted::Context context = NewContext();
    std::vector<u32> calls;         // the guest call stack, for the report
    LPVOID fiber = nullptr;         // the fiber it runs on now (a coroutine's, while one runs)
    // Guest coroutines (run.h): the thread's own fiber while one of them runs,
    // which runs, and the thread's own call stack meanwhile.
    LPVOID home = nullptr;
    Coroutine* coroutine = nullptr;
    std::vector<u32> home_calls;
};

}  // namespace

struct Coroutine {
    LPVOID fiber = nullptr;
    std::function<void()> body;
    std::vector<u32> calls;         // its guest call stack, while it is not running
    bool finished = false;
};

namespace {

struct Semaphore {
    s32 count = 0, max = 0;
};

struct Handler {
    u32 cause, function, argument;      // cause: the INTC cause, or the DMA channel
};

// INTC causes the frame delivers: vertical blank start and end.
constexpr u32 kVblankStart = 2, kVblankEnd = 3;
// Frames that may pass with every thread waiting before the run is called stuck.
constexpr u64 kIdleFrames = 600;

struct Scheduler {
    std::map<u32, std::unique_ptr<Thread>> threads;
    u32 next_thread = 1;
    Thread* current = nullptr;
    LPVOID host_fiber = nullptr;
    u64 counter = 0;
    std::exception_ptr stop;
    std::map<u32, Semaphore> semaphores;
    u32 next_semaphore = 1;
    std::map<u32, Handler> handlers;        // INTC handlers, by id
    std::map<u32, Handler> dmac_handlers;   // DMA-completion handlers, by id
    u32 next_handler = 1;
    u32 intc_mask = 0;
    u32 dmac_mask = 0;
    s64 loop_budget = 0;
    u64 frames = 0;
    bool in_interrupt = false;
    lifted::Context interrupt_context = NewContext();
    u32 pending_dmac = 0;                   // channels whose interrupt waits
    u32 pending_intc = 0;                   // INTC causes a device posted (PostIntc)
    u32 dmac_missed = 0;                    // channels that ended while their interrupt was off
    bool realtime = false;                  // a vblank timer runs (unlimited budget)
    // --virtual-time: the scheduler keeps the clock. A thread runs a slice of
    // loop iterations at a time; each slice is a fixed span of time.
    bool virtual_time = false;
    u64 next_vblank_ns = 0;
    std::atomic<bool> vblank_due{false};    // set by the timer, taken by LoopHead
    std::atomic<u32> vblank_owed{0};        // real time: more that came while one was due (TakeVblank)
    std::atomic<bool> timer_due{false};     // an EE timer's interrupt time has come
    std::atomic<lifted::Context*> running{nullptr};     // the thread on the processor
    bool delivery_requested = false;        // the current thread's budget was zeroed
    s64 saved_budget = 0;                   // ...and this is what it was
    std::vector<u32> interrupt_calls;
    // SetAlarm: a callback after a number of horizontal blanks (15,734 a
    // second, NTSC), run in interrupt context when the host clock passes it.
    // The timer thread wakes for the next one, as for a timer interrupt.
    struct Alarm { u32 callback, common, time; u64 due_ns; };
    std::map<s32, Alarm> alarms;
    s32 next_alarm = 1;
    std::mutex alarms_lock;                 // the timer thread reads the next due time
};

Scheduler& S() {
    static Scheduler scheduler;
    return scheduler;
}

void MakeReady(Thread& t) {
    t.state = State::Ready;
    t.queued = ++S().counter;
}

Thread* Find(u32 id) {
    auto found = S().threads.find(id);
    return found == S().threads.end() ? nullptr : found->second.get();
}

// Give the processor back to the scheduler; returns when this thread runs again.
void GiveWay() {
    Scheduler& s = S();
    if (s.in_interrupt || !s.current) return;     // handlers never block
    SwitchToFiber(s.host_fiber);
}

// Called after a thread became ready: run it now if it is better than this one.
void Preempt(Thread& woken) {
    Scheduler& s = S();
    if (s.in_interrupt || !s.current || woken.priority >= s.current->priority) return;
    MakeReady(*s.current);
    GiveWay();
}

void CALLBACK ThreadMain(void* parameter) {
    Thread& t = *static_cast<Thread*>(parameter);
    try {
        Dispatch(t.context, t.entry);
        t.state = State::Dormant;               // returning from its entry ends a thread
    } catch (...) {
        S().stop = std::current_exception();
    }
    SwitchToFiber(S().host_fiber);              // never resumed
}

Thread* PickReady() {
    Thread* best = nullptr;
    for (auto& [id, t] : S().threads) {
        if (t->state != State::Ready) continue;
        if (!best || t->priority < best->priority ||
            (t->priority == best->priority && t->queued < best->queued))
            best = t.get();
    }
    return best;
}

// Run the handlers registered for `cause` among `handlers`, in interrupt context.
bool RunHandlers(const std::map<u32, Handler>& handlers, u32 cause) {
    Scheduler& s = S();
    bool ran = false;
    for (const auto& [id, h] : handlers) {
        if (h.cause != cause) continue;
        lifted::Context& c = s.interrupt_context;
        c.r[4] = cause;
        c.r[5] = h.argument;
        c.r[29] = 0x01F80000u;                  // an interrupt stack clear of the threads'
        c.r[31] = 0;
        // The kernel leaves $gp as the interrupted code had it: the program's.
        if (s.current) c.r[28] = s.current->context.r[28];
        else if (!s.threads.empty()) c.r[28] = s.threads.begin()->second->context.r[28];
        c.budget = StopRequested() ? -1 : s.loop_budget;     // a handler stops too
        const bool nested = s.in_interrupt;
        s.in_interrupt = true;
        Dispatch(c, h.function);
        s.in_interrupt = nested;
        ran = true;
    }
    // What the handler posted (PostDmac, PostIntc) waits for no other event:
    // the interrupted thread's next loop head.
    if (ran && !s.in_interrupt && (s.pending_dmac || s.pending_intc)) RequestDelivery();
    return ran;
}

// Alarms whose time has come: each callback(id, time, common), in interrupt
// context, once.
u64 NowNs() { return ClockNs(); }

bool DeliverAlarms() {
    Scheduler& s = S();
    bool ran = false;
    for (;;) {
        s32 id = 0;
        Scheduler::Alarm alarm{};
        {
            std::lock_guard<std::mutex> hold(s.alarms_lock);
            const u64 now = NowNs();
            auto it = s.alarms.begin();
            while (it != s.alarms.end() && it->second.due_ns > now) ++it;
            if (it == s.alarms.end()) break;
            id = it->first;
            alarm = it->second;
            s.alarms.erase(it);
        }
        lifted::Context& c = s.interrupt_context;
        c.r[4] = u64(s64(id));
        c.r[5] = alarm.time;
        c.r[6] = alarm.common;
        c.r[29] = 0x01F80000u;
        c.r[31] = 0;
        if (s.current) c.r[28] = s.current->context.r[28];
        else if (!s.threads.empty()) c.r[28] = s.threads.begin()->second->context.r[28];
        c.budget = StopRequested() ? -1 : s.loop_budget;
        const bool nested = s.in_interrupt;
        s.in_interrupt = true;
        Dispatch(c, alarm.callback);
        s.in_interrupt = nested;
        ran = true;
    }
    return ran;
}

// Host time to the next alarm, < 0 when none is set.
s64 NanosToNextAlarm() {
    Scheduler& s = S();
    std::lock_guard<std::mutex> hold(s.alarms_lock);
    if (s.alarms.empty()) return -1;
    u64 next = ~u64(0);
    for (const auto& [id, a] : s.alarms) next = std::min(next, a.due_ns);
    const u64 now = NowNs();
    return next > now ? s64(next - now) : 0;
}

void StartRealTime();
u64 g_real_time_from = 0;               // SetRealTimeFrom

// INTC_STAT (0x1000F000): the causes raised since the game last cleared them
// (writing 1 clears). libgraph's sceGsSyncV clears VBLANK_START and polls for
// it - read as plain memory, the 1 it wrote, it never waited: Mister Mosquito's
// main loop ran five times a frame, its clocks with it.
std::atomic<u32> g_intc_stat{0};

// One frame's interrupts, delivered to the handlers registered for them.
bool DeliverFrame() {
    Scheduler& s = S();
    ++s.frames;
    g_intc_stat.fetch_or((1u << kVblankStart) | (1u << kVblankEnd), std::memory_order_relaxed);
    if (s.virtual_time && g_real_time_from && s.frames >= g_real_time_from) StartRealTime();
    {
        Timed timed(Section::Iop);
        iop::Run();                         // the IOP's timers and sound interrupts
    }
    FrameDone(s.frames);
    bool ran = false;
    for (u32 cause : {kVblankStart, kVblankEnd})
        if (s.intc_mask & (1u << cause)) ran |= RunHandlers(s.handlers, cause);
    ran |= DeliverAlarms();
    return ran;
}

void DeliverPendingDmac() {
    Scheduler& s = S();
    while (s.pending_dmac || s.pending_intc) {
        if (s.pending_dmac) {
            u32 channel = 0;
            while (!((s.pending_dmac >> channel) & 1)) ++channel;
            s.pending_dmac &= s.pending_dmac - 1;
            RaiseDmac(channel);
            continue;
        }
        // A device's INTC cause (PostIntc): one after another, never nested -
        // a handler that posts the next (VIF1 resumed, and stalled again)
        // has it run here once it has returned.
        u32 cause = 0;
        while (!((s.pending_intc >> cause) & 1)) ++cause;
        s.pending_intc &= s.pending_intc - 1;
        RaiseIntc(cause);
    }
}

void SpendBudget(lifted::Context& c) {
    reinterpret_cast<std::atomic<s64>*>(&c.budget)->store(0);
}

std::atomic<u64> g_lost_vblanks{0};
std::atomic<u64> g_idle_ns{0};      // real time: the EE asleep, every thread waiting (--frame-times)

// The EE's wait, on the profiler's timeline (timeline.h).
void IdleSpan(std::chrono::steady_clock::time_point from) {
    if (!timeline::On()) return;
    const u64 from_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(from.time_since_epoch()).count());
    timeline::Span("idle", from_ns, timeline::Now());
}
bool g_vif1_worker_wanted = true;   // --vif1-sync turns it off

// A vertical blank that comes while the last is still undelivered is owed,
// not dropped: the game counts blanks for its time (a movie's pictures
// among it) while the sound keeps the real clock, so each one lost put the
// picture further behind. Up to this many wait; past them the EE is that far
// behind, and they are lost (--frame-times counts them).
constexpr u32 kOwedVblanks = 3;

// The timer thread, stopped and joined as the process exits: left running,
// it read the scheduler while the statics were being destroyed (a crash at
// exit, after the run's report).
std::atomic<bool> g_timer_stop{false};
std::thread& TimerThread() {
    static std::thread thread;
    return thread;
}
void StopTimer() {
    g_timer_stop = true;
    if (TimerThread().joinable()) TimerThread().join();
}

// Takes a due vertical blank, leaving the next one due at once if one is owed.
// PS2HOST_HITCH: when the last blank came due (the timer's), to tell how
// late the EE took it - and what it did meanwhile, for the sampler.
std::atomic<u64> g_blank_due_ns{0};
const bool g_hitch_blanks = std::getenv("PS2HOST_HITCH") != nullptr;

bool TakeVblank(Scheduler& s) {
    if (!s.vblank_due.exchange(false)) return false;
    if (g_hitch_blanks) {
        const u64 due = g_blank_due_ns.load(std::memory_order_relaxed), now = ClockNs();
        if (due && now > due + 4'000'000) {
            SampleOnly(due + 500'000, now);
            std::fprintf(stderr, "[hitch] a blank taken %.1f ms late\n", (now - due) / 1e6);
        }
        g_blank_due_ns.store(0, std::memory_order_relaxed);
    }
    u32 owed = s.vblank_owed.load();
    while (owed && !s.vblank_owed.compare_exchange_weak(owed, owed - 1)) {
    }
    if (owed) s.vblank_due = true;
    return true;
}

// Raises the vertical blank in real time: marks a frame due and spends the
// running thread's budget, so its next loop head delivers it. Windows sleeps
// to its timer's resolution (15.6 ms unless raised; the scheduler raises it
// to 1 ms in real time), so the thread sleeps to just short of the moment
// and yields to it: a blank two milliseconds late was a blank the EE took
// with the next, and lost.
//
// A spend can itself be lost: the EE's own decrement of its budget (a loop
// head's, not atomic) or a loop head's new budget written over it. The
// blank then waited for the next one's spend - a whole period late, a
// picture held a frame longer (Burnout 3's race: one in ten frames, the EE
// spinning in its packet wait all the while). So while a blank or a timer
// interrupt waits to be taken, the budget is spent again every half
// millisecond.
std::atomic<unsigned> g_fast_forward{1};    // SetFastForward
std::atomic<u64> g_fast_until{0};           // SetFastUntil: blanks faster before this frame...
std::atomic<unsigned> g_fast_until_factor{8};   // ...this many times
std::atomic<int> g_interrupt_waits{0};      // InterruptWait
// Counted up as the timer marks a blank or a timer due (or looks at the EE's
// loop heads): what an idle loop's wait sleeps on (WaitOnAddress), woken at
// once - a sleep of 200 us was a millisecond or more, the frame late with it.
std::atomic<u32> g_timer_seq{0};
void TimerTick() {
    g_timer_seq.fetch_add(1, std::memory_order_release);
    WakeByAddressAll(&g_timer_seq);
}
std::atomic<u64> g_last_vblank_ns{0};       // LastVblankNs

/* Asleep until `when`, to a tenth of a millisecond or so: a high-resolution
   waitable timer (Windows 10 1803 on), not Sleep's millisecond steps that can
   overshoot by more; sleep_until where there is none. */
void SleepUntil(std::chrono::steady_clock::time_point when) {
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                                       TIMER_ALL_ACCESS);
    const auto left = when - std::chrono::steady_clock::now();
    if (left <= std::chrono::steady_clock::duration::zero()) return;
    if (!timer) {
        std::this_thread::sleep_until(when);
        return;
    }
    LARGE_INTEGER due;
    due.QuadPart = -std::max<long long>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() / 100);
    if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, INFINITE);
    else std::this_thread::sleep_until(when);
}

// The real-time blanks' period (VideoModeAsked): NTSC's 59.94 Hz, or PAL's
// 50 when the game sets PAL up and pal_timing is on.
std::atomic<u32> g_vblank_us{16683};
std::atomic<bool> g_pal_timing{false};

void VblankTimer() {
    using clock = std::chrono::steady_clock;
    const auto margin = std::chrono::microseconds(700);    // the precise sleep's (SleepUntil), then a spin
    auto next = clock::now() + std::chrono::microseconds(g_vblank_us.load());
    while (!StopRequested() && !g_timer_stop) {
        const auto period = std::chrono::microseconds(g_vblank_us.load(std::memory_order_relaxed));
        // Sleep to the next vertical blank, or to an EE timer's interrupt if that
        // comes first.
        Scheduler& s = S();
        auto wake = next;
        s64 timer = NanosToNextTimerEvent();
        const s64 alarm = NanosToNextAlarm();
        if (alarm >= 0 && (timer < 0 || alarm < timer)) timer = alarm;
        const auto timer_at = timer >= 0 ? clock::now() + std::chrono::nanoseconds(timer) : clock::time_point::max();
        wake = std::min(wake, timer_at);
        const bool untaken = s.vblank_due.load() || s.timer_due.load();
        const auto retry_at = clock::now() + std::chrono::microseconds(500);
        const bool retrying = untaken && retry_at < wake;
        if (retrying) wake = retry_at;
        // Far from the next event, while the EE is in a function that waits
        // for an interrupt (InterruptWait): a look at its loop heads every
        // millisecond meanwhile (a plain sleep, not the precise wait below) -
        // its idle loop (AddIdleLoop) then waits there asleep, not spinning a
        // core through the rest of its frame (LoopHead). Not otherwise: a
        // look at the loop heads of busy code cost the race a fifth of a core.
        const auto poke_at = clock::now() + std::chrono::milliseconds(1);
        if (g_interrupt_waits.load(std::memory_order_relaxed) > 0 && poke_at + margin < wake) {
            std::this_thread::sleep_until(poke_at);
            if (lifted::Context* c = s.running.load()) SpendBudget(*c);
            TimerTick();
            continue;
        }
        if (wake - clock::now() > margin) SleepUntil(wake - margin);
        while (clock::now() < wake && !StopRequested() && !g_timer_stop) std::this_thread::yield();
        if (clock::now() >= next) {
            const auto late = clock::now() - next;
            if (late > std::chrono::milliseconds(1) && timeline::On())
                timeline::Instant("blank late (us)", timeline::Now(),
                                  u64(std::chrono::duration_cast<std::chrono::microseconds>(late).count()));
            g_last_vblank_ns.store(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(next.time_since_epoch()).count()),
                                   std::memory_order_relaxed);
            unsigned factor = g_fast_forward.load(std::memory_order_relaxed);
            if (S().frames < g_fast_until.load(std::memory_order_relaxed)) factor = std::max(factor, g_fast_until_factor.load(std::memory_order_relaxed));
            next += factor > 1 ? period / factor : period;
            if (clock::now() >= next + period) next = clock::now() + period;   // a stall: start again from now
            if (g_hitch_blanks && !s.vblank_due.load())
                g_blank_due_ns.store(ClockNs(), std::memory_order_relaxed);
            if (s.vblank_due.exchange(true)) {
                if (s.vblank_owed.load() < kOwedVblanks) s.vblank_owed.fetch_add(1);
                else {
                    g_lost_vblanks.fetch_add(1, std::memory_order_relaxed);
                    timeline::BlankLost();
                }
            }
        }
        if (timer >= 0 && (!retrying || clock::now() >= timer_at)) s.timer_due = true;
        if (lifted::Context* c = s.running.load())
            if (!retrying || s.vblank_due.load() || s.timer_due.load()) SpendBudget(*c);
        TimerTick();
    }
}

// --virtual-time: loop iterations a thread runs before the scheduler looks
// again, and the time they stand for (an iteration is some tens of
// instructions; the EE runs about 300 million a second).
constexpr s64 kSliceIterations = 20000;
constexpr u64 kNsPerIteration = 50;
constexpr u64 kFrameNs = 16683333;          // NTSC, 59.94 Hz

// Virtual time: whether the clock has reached the next vertical blank.
bool VblankReached() {
    Scheduler& s = S();
    const u64 now = ClockNs();
    if (now < s.next_vblank_ns) return false;
    s.next_vblank_ns += kFrameNs;
    if (s.next_vblank_ns <= now) s.next_vblank_ns = now + kFrameNs;
    return true;
}

// AddIdleLoop: the loop heads a port names, and their short slice.
std::unordered_set<u32>& IdleLoops() {
    static std::unordered_set<u32> loops;
    return loops;
}
constexpr s64 kIdleSliceIterations = 256;

// lifted::loop_head_hook: a spent budget is either this host's request to
// deliver interrupts, or the end of the run.
void LoopHead(lifted::Context& c, u32 address) {
    Scheduler& s = S();
    if (StopRequested() || s.in_interrupt || !s.current || &c != &s.current->context)
        throw lifted::Abort{address, 2};
    const bool idle = s.virtual_time && IdleLoops().count(address) != 0;
    // Real time in an idle loop (a game's wait for an interrupt - Burnout 3's
    // for its next picture's vertical blank, 60% of the EE's time in its
    // menus): wait for one asleep, and come back soon (a small budget). The
    // IOP runs as it did while the loop spun: at the interrupts.
    const bool realtime_idle = s.realtime && IdleLoops().count(address) != 0;
    if (realtime_idle) {
        const auto idle_from = std::chrono::steady_clock::now();
        for (;;) {
            u32 seen = g_timer_seq.load(std::memory_order_acquire);
            if (s.vblank_due || s.timer_due || s.delivery_requested || StopRequested() || IopReplyPending()) break;
            WaitOnAddress(&g_timer_seq, &seen, sizeof seen, 2);
        }
        g_idle_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - idle_from).count()),
                            std::memory_order_relaxed);
        IdleSpan(idle_from);
    }
    if (s.virtual_time && !s.delivery_requested) {
        // A slice ran: its time passes; what that time brings is delivered. In
        // an idle loop, the time to the next event passes at once.
        u64 step = u64(kSliceIterations) * kNsPerIteration;
        if (idle) {
            u64 wait = s.next_vblank_ns > ClockNs() ? s.next_vblank_ns - ClockNs() : 0;
            const s64 timer = NanosToNextTimerEvent(), alarm = NanosToNextAlarm();
            if (timer >= 0) wait = std::min(wait, u64(timer));
            if (alarm >= 0) wait = std::min(wait, u64(alarm));
            step = std::max<u64>(wait, u64(kIdleSliceIterations) * kNsPerIteration);
        }
        AdvanceClock(step);
        if (VblankReached()) s.vblank_due = true;
        s.timer_due = true;                 // DeliverTimers raises only what is due
    }
    const bool frame = TakeVblank(s);
    const bool timers = s.timer_due.exchange(false);
    const bool requested = s.delivery_requested;
    // With a real-time timer a budget can be spent for a frame another loop head
    // already took; without one, a spent budget that nobody asked for is the end.
    if (!frame && !timers && !requested && !s.realtime && !s.virtual_time) throw lifted::Abort{address, 2};
    s.delivery_requested = false;
    c.budget = requested ? s.saved_budget
             : s.virtual_time ? (idle ? kIdleSliceIterations : kSliceIterations)
             : realtime_idle ? kIdleSliceIterations : s.loop_budget;
    if (StopRequested()) c.budget = -1;         // the watchdog fired meanwhile
    // In real time the timer may have spent the budget since the blank and
    // timers were looked at: its spend kept (VblankTimer).
    if (s.realtime && (s.vblank_due.load() || s.timer_due.load())) c.budget = 0;
    if (frame) DeliverFrame();
    if (timers) {
        DeliverTimers();
        DeliverAlarms();
    }
    while (DeliverIopReply()) {
    }
    DeliverPendingDmac();
    // Returning from the interrupt, a thread its handler woke may run first.
    Thread* best = PickReady();
    if (best && best->priority < s.current->priority) {
        MakeReady(*s.current);
        GiveWay();
    }
}

}  // namespace

void AddIdleLoop(u32 address) { IdleLoops().insert(address); }

InterruptWait::InterruptWait() { g_interrupt_waits.fetch_add(1, std::memory_order_relaxed); }
InterruptWait::~InterruptWait() { g_interrupt_waits.fetch_sub(1, std::memory_order_relaxed); }

std::atomic<bool>& StopRequested() {
    static std::atomic<bool> stop{false};
    return stop;
}

void RaiseDmac(u32 channel) {
    Scheduler& s = S();
    // The notes' keys and texts, made once a channel (a movie raises
    // channels 3 and 4 thousands of times a second).
    struct Texts {
        std::string key, masked_key, masked, delivered, none_key, none;
    };
    static const std::vector<Texts> texts = [] {
        std::vector<Texts> all;
        for (u32 c = 0; c < 32; ++c) {
            const std::string n = std::to_string(c), key = "dmac " + n;
            all.push_back({key, key + " masked", "DMAC: channel " + n + " completed while its interrupt is disabled",
                           "DMAC: channel " + n + " interrupt delivered", key + " none",
                           "DMAC: channel " + n + " interrupt has no handler"});
        }
        return all;
    }();
    const Texts& t = texts[channel & 31];
    if (!(s.dmac_mask & (1u << channel))) {
        // The interrupt is status AND mask: the status stays set (D_STAT),
        // and enabling the channel raises it then (_EnableDmac).
        s.dmac_missed |= 1u << channel;
        Note(t.masked_key, t.masked);
        return;
    }
    if (RunHandlers(s.dmac_handlers, channel))
        Note(t.key, t.delivered);
    else
        Note(t.none_key, t.none);
}

void RaiseIntc(u32 cause) {
    Scheduler& s = S();
    g_intc_stat.fetch_or(1u << cause, std::memory_order_relaxed);
    if (s.intc_mask & (1u << cause)) RunHandlers(s.handlers, cause);
}

u32 IntcStat() { return g_intc_stat.load(std::memory_order_relaxed); }
void ClearIntcStat(u32 bits) { g_intc_stat.fetch_and(~bits, std::memory_order_relaxed); }

void PostDmac(u32 channel) {
    S().pending_dmac |= 1u << channel;
    RequestDelivery();
}

void PostIntc(u32 cause) {
    S().pending_intc |= 1u << cause;
    RequestDelivery();
}

void RequestDelivery() {
    Scheduler& s = S();
    if (s.in_interrupt || !s.current || s.delivery_requested) return;
    lifted::Context& c = s.current->context;
    s.saved_budget = c.budget;
    s.delivery_requested = true;
    c.budget = 0;
}

namespace {
// GuestCallsNow's read, apart: no C++ objects, so it can be guarded.
int ReadCalls(const u32* data, std::size_t size, u32* out, int max) {
    int n = 0;
    __try {
        for (std::size_t i = size; i > 0 && n < max; --i) out[n++] = data[i - 1];
    } __except (1) {
        n = 0;
    }
    return n;
}
}  // namespace

int GuestCallsNow(u32* out, int max) {
    // The EE's thread is suspended (the sampler's): its vector may be mid
    // change, so the read is a guarded one into the caller's array.
    Scheduler& s = S();
    Thread* current = s.current;
    if (!current) return 0;
    const std::vector<u32>& calls = s.in_interrupt ? s.interrupt_calls : current->calls;
    return ReadCalls(calls.data(), calls.size(), out, max);
}

std::vector<u32>& CurrentCalls() {
    Scheduler& s = S();
    if (s.in_interrupt || !s.current) return s.interrupt_calls;
    return s.current->calls;
}

namespace {

void CALLBACK CoroutineMain(void* parameter) {
    Coroutine& co = *static_cast<Coroutine*>(parameter);
    try {
        co.body();
    } catch (...) {
        S().stop = std::current_exception();
        SwitchToFiber(S().host_fiber);          // never resumed
    }
    co.finished = true;
    SwitchCoroutine(nullptr);                   // never resumed
}

}  // namespace

Coroutine* CreateCoroutine(std::function<void()> body) {
    if (!S().current || S().in_interrupt) throw Unimplemented{"a guest coroutine outside a thread"};
    auto* co = new Coroutine;
    co->body = std::move(body);
    co->fiber = CreateFiber(1 << 20, CoroutineMain, co);
    return co;
}

void SwitchCoroutine(Coroutine* to) {
    Scheduler& s = S();
    Thread* t = s.current;
    if (!t || s.in_interrupt) throw Unimplemented{"a guest coroutine switch outside a thread"};
    Coroutine* from = t->coroutine;
    if (from == to) return;
    if (!from) t->home = t->fiber;
    // The call stack the report and the sampler show goes with the fiber.
    (from ? from->calls : t->home_calls) = std::move(t->calls);
    t->calls = std::move(to ? to->calls : t->home_calls);
    t->coroutine = to;
    t->fiber = to ? to->fiber : t->home;
    SwitchToFiber(t->fiber);
}

Coroutine* CurrentCoroutine() {
    const Scheduler& s = S();
    return s.current && !s.in_interrupt ? s.current->coroutine : nullptr;
}

bool CoroutineFinished(const Coroutine* co) { return co->finished; }

void DeleteCoroutine(Coroutine* co) {
    if (!co) return;
    if (co == CurrentCoroutine()) throw Unimplemented{"a guest coroutine deleted while it runs"};
    DeleteFiber(co->fiber);
    delete co;
}

std::vector<u32> InterruptedCalls() {
    Scheduler& s = S();
    return s.in_interrupt && s.current ? s.current->calls : std::vector<u32>();
}

void ForEachContext(void (*visit)(lifted::Context&)) {
    for (auto& [id, t] : S().threads) visit(t->context);
    visit(S().interrupt_context);
}

bool SpendUntilStopped(const std::function<void()>& spend,
                       const std::function<bool(std::chrono::milliseconds)>& stopped,
                       std::chrono::milliseconds every, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    for (;;) {
        spend();
        if (stopped(every)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
}

namespace {

/* The run's real time (RunThreads, or StartRealTime): the blanks' timer,
   VIF1's worker. */
void StartRealTimeThreads() {
    timeBeginPeriod(1);             // sleeps to the millisecond, not Windows' 15.6 ms
    // ...even with the window hidden behind others: Windows 11 drops a
    // process's timer resolution then (a 200 us sleep took 15 ms, a blank
    // seen 14 ms late), unless told not to; and the process not throttled.
    PROCESS_POWER_THROTTLING_STATE process{};
    process.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    process.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    process.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &process, sizeof process);
    FastThread();
    if (!TimerThread().joinable()) {
        TimerThread() = std::thread([] {
            timeline::NameThread("vertical blank timer");
            // Asleep almost always, it must wake on time with every core busy
            // (a blank 13 ms late was a frame shown late): the highest priority.
            FastThread();
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
            VblankTimer();
        });
        std::atexit(StopTimer);
    }
    if (g_vif1_worker_wanted) StartVif1Worker();
}

/* --real-time-from: a virtual-time run in real time from here (a frame's
   delivery: VIF1 and the GS between frames), the clock going on from where
   it was. */
void StartRealTime() {
    Scheduler& s = S();
    LeaveVirtualClock();
    s.virtual_time = false;
    s.realtime = s.loop_budget == INT64_MAX;
    s.vblank_due = false;
    s.timer_due = false;
    for (auto& [id, t] : s.threads) t->context.budget = s.loop_budget;
    if (s.realtime) StartRealTimeThreads();
    std::fprintf(stderr, "[host] real time from frame %llu\n", (unsigned long long)s.frames);
}

}  // namespace

void RunThreads(const lifted::Context& initial, u32 entry, s64 loop_budget) {
    Scheduler& s = S();
    s.loop_budget = loop_budget;
    lifted::loop_head_hook = LoopHead;
    StartCop0();
    s.virtual_time = VirtualClock();
    s.realtime = loop_budget == INT64_MAX && !s.virtual_time;
    if (s.realtime) StartRealTimeThreads();
    s.next_vblank_ns = ClockNs() + kFrameNs;
    s.host_fiber = ConvertThreadToFiber(nullptr);
    auto main = std::make_unique<Thread>();
    main->id = s.next_thread++;
    main->entry = entry;
    main->context = initial;
    main->context.budget = s.virtual_time ? kSliceIterations : loop_budget;
    main->fiber = CreateFiber(1 << 20, ThreadMain, main.get());
    Thread& first = *main;
    s.threads[first.id] = std::move(main);
    MakeReady(first);
    u64 idle = 0;
    for (;;) {
        if (s.stop) std::rethrow_exception(s.stop);
        if (StopRequested()) throw lifted::Abort{0, 2};     // the time limit, between threads
        // Answers from the IOP arrive between threads, as interrupts.
        while (DeliverIopReply()) {
        }
        DeliverPendingDmac();
        Thread* next = PickReady();
        if (!next) {
            if (first.state == State::Dormant)
                throw GuestExit{0};                 // the program's first thread returned
            if (s.virtual_time) {
                // Every thread waits: time skips to the next vertical blank or
                // timer or alarm interrupt, whichever comes first.
                u64 wait = s.next_vblank_ns > ClockNs() ? s.next_vblank_ns - ClockNs() : 0;
                const s64 timer = NanosToNextTimerEvent(), alarm = NanosToNextAlarm();
                if (timer >= 0) wait = std::min(wait, u64(timer));
                if (alarm >= 0) wait = std::min(wait, u64(alarm));
                AdvanceClock(wait);
                DeliverTimers();
                DeliverAlarms();
                if (!VblankReached()) continue;
            } else if (s.realtime) {
                // Every thread waits: so does the host, for the next vertical
                // blank or timer interrupt - the game runs at its own pace.
                // The IOP runs meanwhile, as the console's does beside the
                // EE: run only at each frame (DeliverFrame), a module's
                // thread that waited (a delay, a semaphore its own reader
                // signals) went on once a frame, and a loading screen that
                // asks it for a file piece by piece sat idle between pieces.
                // An answer it sends wakes the EE at once.
                const auto idle_from = std::chrono::steady_clock::now();
                while (!s.vblank_due && !s.timer_due && !StopRequested()) {
                    {
                        Timed timed(Section::Iop);
                        iop::Run(50'000);           // a short slice: the blank may come meanwhile
                    }
                    if (IopReplyPending()) break;
                    SleepUntil(std::chrono::steady_clock::now() + std::chrono::microseconds(200));    // (precise: SleepUntil)
                }
                g_idle_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - idle_from).count()),
                                    std::memory_order_relaxed);
                IdleSpan(idle_from);
                if (s.timer_due.exchange(false)) {
                    DeliverTimers();
                    DeliverAlarms();
                }
                if (!TakeVblank(s)) continue;
            }
            DeliverFrame();
            if (++idle > kIdleFrames) {
                std::string states;
                for (auto& [id, t] : s.threads)
                    states += "\n    thread " + std::to_string(id) + ": " + StateName(t->state);
                throw Unimplemented{"every thread is waiting, and " + std::to_string(kIdleFrames) +
                                    " frames of interrupts woke none:" + states};
            }
            continue;
        }
        idle = 0;
        s.current = next;
        ++next->runs;
        next->state = State::Running;
        s.running = &next->context;
        SwitchToFiber(next->fiber);
        s.running = nullptr;
        s.current = nullptr;
        if (TakeVblank(s)) DeliverFrame();
        if (s.timer_due.exchange(false)) {
            DeliverTimers();
            DeliverAlarms();
        }
        if (s.stop) std::rethrow_exception(s.stop);
    }
}

std::string ThreadReport() {
    std::string out;
    for (auto& [id, t] : S().threads) {
        char line[160];
        std::snprintf(line, sizeof line, "  thread %u: %s, priority %d, entry 0x%08X, ran %llu times\n", id,
                      StateName(t->state), t->priority, t->entry, static_cast<unsigned long long>(t->runs));
        out += line;
        // Where it is: the innermost calls of its guest stack.
        const std::size_t from = t->calls.size() > 10 ? t->calls.size() - 10 : 0;
        for (std::size_t i = from; i < t->calls.size(); ++i) {
            const char* name = FunctionName(t->calls[i]);
            std::snprintf(line, sizeof line, "      0x%08X %s\n", t->calls[i], name ? name : "");
            out += line;
        }
    }
    return out;
}

void SignalSemaphore(u32 id) {
    Scheduler& s = S();
    auto found = s.semaphores.find(id);
    if (found == s.semaphores.end()) return;
    Thread* taker = nullptr;
    for (auto& [tid, t] : s.threads) {
        if (t->state == State::Waiting && t->waiting_on == id &&
            (!taker || t->priority < taker->priority ||
             (t->priority == taker->priority && t->queued < taker->queued)))
            taker = t.get();
    }
    if (taker) MakeReady(*taker);
    else found->second.count++;
}

// The thread, semaphore and interrupt services; false for any other.
bool ThreadSyscall(lifted::Context& c, s32 number, u64& result) {
    Scheduler& s = S();
    Memory& m = GuestMemory();
    const u32 a0 = u32(c.r[4]), a1 = u32(c.r[5]);
    switch (number) {
        case 0xFC: case -0xFD: {                     // SetAlarm / iSetAlarm(time, callback, common)
            // time is in horizontal blanks: 262.5 a field, NTSC.
            const u32 time = a0 & 0xFFFF;
            std::lock_guard<std::mutex> hold(s.alarms_lock);
            const s32 id = s.next_alarm++;
            s.alarms[id] = {a1, u32(c.r[6]), time, NowNs() + u64(double(time) * 1e9 / 15734.25)};
            result = u64(s64(id));
            if (!s.in_interrupt) RequestDelivery();   // one already due runs at once
            return true;
        }
        case 0xFE: case -0xFF: {                     // ReleaseAlarm / iReleaseAlarm(id)
            std::lock_guard<std::mutex> hold(s.alarms_lock);
            result = s.alarms.erase(s32(a0)) ? 0 : u64(s64(-1));
            return true;
        }
        case 0x20: {                                 // CreateThread(param*)
            auto t = std::make_unique<Thread>();
            t->id = s.next_thread++;
            // struct ThreadParam { status, func, stack, stackSize, gpReg, initPriority }
            t->entry = u32(m.Read(a0 + 4, 4));
            t->stack = u32(m.Read(a0 + 8, 4));
            t->stack_size = u32(m.Read(a0 + 12, 4));
            t->gp = u32(m.Read(a0 + 16, 4));
            t->priority = s32(m.Read(a0 + 20, 4));
            result = t->id;
            s.threads[t->id] = std::move(t);
            return true;
        }
        case 0x22: {                                 // StartThread(id, arg)
            Thread* t = Find(a0);
            if (!t || t->state != State::Dormant) { result = u64(-1); return true; }
            t->context = NewContext();
            t->context.r[4] = a1;
            t->context.r[28] = t->gp;
            t->context.r[29] = t->stack + t->stack_size - 0x2A0;
            t->context.budget = s.virtual_time ? kSliceIterations : s.loop_budget;
            t->calls.clear();
            if (t->home) t->fiber = t->home;    // a coroutine's fiber stays its owner's
            t->home = nullptr;
            t->coroutine = nullptr;
            if (t->fiber) DeleteFiber(t->fiber);
            t->fiber = CreateFiber(1 << 20, ThreadMain, t);
            MakeReady(*t);
            result = a0;
            Preempt(*t);
            return true;
        }
        case 0x23: case 0x24:                        // ExitThread / ExitDeleteThread
            if (s.current) {
                s.current->state = State::Dormant;
                GiveWay();
            }
            result = 0;
            return true;
        case 0x2F:                                   // GetThreadId
            result = s.current ? s.current->id : 1;
            return true;
        case 0x29: case -0x29: {                     // ChangeThreadPriority(id, priority)
            Thread* t = a0 == 0 ? s.current : Find(a0);
            if (!t) { result = u64(-1); return true; }
            result = u32(t->priority);
            t->priority = s32(a1);
            // A thread that is no longer the best gives way.
            if (t == s.current) {
                if (Thread* better = PickReady(); better && better->priority < t->priority) {
                    MakeReady(*t);
                    GiveWay();
                }
            }
            return true;
        }
        case 0x2B: case -0x2B: {                     // RotateThreadReadyQueue(priority)
            if (s.current && s.current->priority == s32(a0) && !s.in_interrupt) {
                MakeReady(*s.current);
                GiveWay();
            }
            result = 0;
            return true;
        }
        case 0x32:                                   // SleepThread
            if (s.current) {
                if (s.current->wakeups > 0) {
                    s.current->wakeups--;
                } else {
                    s.current->state = State::Sleeping;
                    GiveWay();
                }
            }
            result = 0;
            return true;
        case 0x33: case 0x34: case -0x34: {          // WakeupThread / iWakeupThread
            Thread* t = Find(a0);
            if (!t) { result = u64(-1); return true; }
            if (t->state == State::Sleeping) {
                MakeReady(*t);
                Preempt(*t);
            } else {
                t->wakeups++;
            }
            result = 0;
            return true;
        }
        case 0x40: {                                 // CreateSema(param*): count, max, init
            Semaphore sema;
            sema.count = s32(m.Read(a0 + 8, 4));     // struct SemaParam { count, max_count, init_count, ... }
            sema.max = s32(m.Read(a0 + 4, 4));
            const u32 id = s.next_semaphore++;
            s.semaphores[id] = sema;
            result = id;
            return true;
        }
        case 0x41:                                   // DeleteSema
            s.semaphores.erase(a0);
            result = a0;
            return true;
        case 0x42: case 0x43: case -0x42: case -0x43: {   // SignalSema / iSignalSema
            auto found = s.semaphores.find(a0);
            if (found == s.semaphores.end()) { result = u64(-1); return true; }
            // The best thread waiting on it takes the signal; else the count rises.
            Thread* taker = nullptr;
            for (auto& [id, t] : s.threads) {
                if (t->state == State::Waiting && t->waiting_on == a0 &&
                    (!taker || t->priority < taker->priority ||
                     (t->priority == taker->priority && t->queued < taker->queued)))
                    taker = t.get();
            }
            if (taker) {
                MakeReady(*taker);
                Preempt(*taker);
            } else {
                found->second.count++;
            }
            result = a0;
            return true;
        }
        case 0x44: {                                 // WaitSema(id)
            auto found = s.semaphores.find(a0);
            if (found == s.semaphores.end()) { result = u64(-1); return true; }
            if (found->second.count > 0) {
                found->second.count--;
            } else if (s.current && !s.in_interrupt) {
                s.current->state = State::Waiting;
                s.current->waiting_on = a0;
                s.current->queued = ++s.counter;
                GiveWay();
            }
            result = a0;
            return true;
        }
        case 0x45: case -0x45: {                     // PollSema / iPollSema
            auto found = s.semaphores.find(a0);
            if (found == s.semaphores.end() || found->second.count <= 0) {
                result = u64(-1);
                return true;
            }
            found->second.count--;
            result = a0;
            return true;
        }
        case 0x10: case -0x10: {                     // AddIntcHandler(cause, handler, next, arg)
            const u32 id = s.next_handler++;
            s.handlers[id] = Handler{a0, a1, u32(c.r[7])};
            result = id;
            return true;
        }
        case 0x11: case -0x11:                       // RemoveIntcHandler(cause, id)
            s.handlers.erase(a1);
            result = 0;
            return true;
        case 0x12: case -0x12: {                     // AddDmacHandler(channel, handler, next, arg)
            const u32 id = s.next_handler++;
            s.dmac_handlers[id] = Handler{a0, a1, u32(c.r[7])};
            result = id;
            return true;
        }
        case 0x13: case -0x13:                       // RemoveDmacHandler(channel, id)
            s.dmac_handlers.erase(a1);
            result = 0;
            return true;
        case 0x16: case -0x16: {                     // _EnableDmac(channel)
            const u32 bit = 1u << (a0 & 31);
            s.dmac_mask |= bit;
            // A channel that ended while its interrupt was off interrupts
            // now, as the DMAC's line is status AND mask - if its status
            // (D_STAT) is still set. RenderWare turns channels 1 and 2 off
            // and on around its packets; dropped, an end waited for the next
            // vertical blank, and the race lost a frame every second or so.
            const u32* stat = reinterpret_cast<const u32*>(GuestMemory().Translate(0x1000E010u, 4));
            if ((s.dmac_missed & bit) && stat && (*stat & bit)) PostDmac(a0 & 31);
            s.dmac_missed &= ~bit;
            result = 1;
            return true;
        }
        case 0x17: case -0x17:                       // _DisableDmac(channel)
            s.dmac_mask &= ~(1u << (a0 & 31));
            result = 1;
            return true;
        case 0x14: case -0x14:                       // _EnableIntc(cause)
            s.intc_mask |= 1u << (a0 & 31);
            result = 1;
            return true;
        case 0x15: case -0x15:                       // _DisableIntc(cause)
            s.intc_mask &= ~(1u << (a0 & 31));
            result = 1;
            return true;
        default:
            return false;
    }
}

u64 LostVblanks() { return g_lost_vblanks.load(std::memory_order_relaxed); }
u64 IdleNanoseconds() { return g_idle_ns.load(std::memory_order_relaxed); }
void SetVif1Worker(bool on) { g_vif1_worker_wanted = on; }
void SetRealTimeFrom(u64 frame) { g_real_time_from = frame; }

void SetFastUntil(u64 frame, unsigned factor) {
    g_fast_until = frame;
    g_fast_until_factor = factor ? factor : 1;
}

void SetFastForward(unsigned factor) {
    factor = factor ? factor : 1;
    const unsigned before = g_fast_forward.exchange(factor);
    // Back to real time from fast-forwarding (a loading screen): VIF1's
    // worker and the GS thread fell behind the EE (a race's start: 0.3 s,
    // drained over its first 30 frames - a stutter at the green light);
    // they catch up here, on the loading screen's last picture.
    if (before > 1 && factor == 1 && !VirtualClock()) GsDrain();
}
unsigned FastForward() { return g_fast_forward.load(std::memory_order_relaxed); }
u64 LastVblankNs() { return g_last_vblank_ns.load(std::memory_order_relaxed); }
void SetPalTiming(bool on) { g_pal_timing = on; }
void VideoModeAsked(u32 mode) {
    // SetGsCrt's mode: 2 NTSC, 3 PAL (the vertical blanks of real time follow; a
    // virtual-time run keeps its own, the same in every run).
    const u32 us = g_pal_timing.load() && mode == 3 ? 20000 : 16683;
    SetWindowGameRate(us == 20000 ? 50 : 60);
    if (g_vblank_us.exchange(us) != us)
        std::fprintf(stderr, "[host] vertical blanks every %u us (%s)\n", unsigned(us), us == 20000 ? "PAL" : "NTSC");
}
u32 VblankPeriodUs() { return g_vblank_us.load(); }

void FastThread() {
    THREAD_POWER_THROTTLING_STATE throttling{};
    throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0;               // never throttled
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &throttling, sizeof throttling);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
}

}  // namespace ps2host
