/*
 * The EE's four timers (EE User's Manual, chapter 4), in host time.
 *
 * Each timer counts at its MODE's clock - BUSCLK (147.456 MHz), BUSCLK/16,
 * BUSCLK/256 or the horizontal blank (15,734 Hz, NTSC) - from the value last
 * written to COUNT, as the host clock passes. MODE bit 6 (ZRET) wraps the count
 * after it reaches COMP; otherwise it wraps at 0x10000. Reaching COMP with
 * CMPE (bit 8) set raises EQUF (bit 10), and wrapping with OVFE (bit 9) raises
 * OVFF (bit 11); either interrupts the EE on INTC cause 9 + the timer. The
 * flags clear when 1 is written to them.
 *
 * The SDK's alarms (SetAlarm, DelayThread's kernel patch) run on timer 2's
 * compare and overflow interrupts, so a game that sleeps on an alarm waits on
 * this. Interrupts are raised when the host looks (DeliverTimers): the timer
 * thread wakes for the next event (NanosToNextTimerEvent) as for a vertical
 * blank.
 *
 * Registers: Tn_COUNT +0x00, Tn_MODE +0x10, Tn_COMP +0x20, Tn_HOLD +0x30 at
 * 0x10000000 (T0), 0x10000800 (T1), 0x10001000 (T2), 0x10001800 (T3).
 */

#include <atomic>
#include <chrono>
#include <mutex>

#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {
std::atomic<bool> g_virtual_clock{false};
std::atomic<u64> g_clock_ns{0};
std::atomic<u64> g_real_offset_ns{0};   // real time after virtual (LeaveVirtualClock): the clock goes on from there

u64 SteadyNs() {
    using namespace std::chrono;
    return u64(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
}  // namespace

u64 ClockNs() {
    if (g_virtual_clock) return g_clock_ns;
    return SteadyNs() - g_real_offset_ns.load(std::memory_order_relaxed);
}

void LeaveVirtualClock() {
    if (!g_virtual_clock) return;
    g_real_offset_ns = SteadyNs() - g_clock_ns.load();
    g_virtual_clock = false;
}

void SetVirtualClock(bool on) {
    g_clock_ns = 1000000000ull;             // a second in: no event is due at zero
    g_virtual_clock = on;
}

bool VirtualClock() { return g_virtual_clock; }

void AdvanceClock(u64 ns) { g_clock_ns += ns; }

namespace {

constexpr u32 kBases[4] = {0x10000000u, 0x10000800u, 0x10001000u, 0x10001800u};
constexpr u32 kZret = 1u << 6, kCue = 1u << 7, kCmpe = 1u << 8, kOvfe = 1u << 9;
constexpr u32 kEqUf = 1u << 10, kOvFf = 1u << 11;
constexpr u32 kFirstCause = 9;              // INTC: TIM0 is 9, TIM3 12

struct Timer {
    u32 mode = 0, comp = 0, hold = 0;
    u32 origin_count = 0;                   // the count at origin_ns
    u64 origin_ns = 0;
    u64 checked = 0;                        // ticks since the origin, last delivery
};

struct Timers {
    std::mutex lock;
    Timer t[4];
};

Timers& T() {
    static Timers timers;
    return timers;
}

u64 NowNs() { return ClockNs(); }

double Rate(u32 mode) {
    switch (mode & 3) {
        case 0: return 147456000.0;
        case 1: return 147456000.0 / 16;
        case 2: return 147456000.0 / 256;
        default: return 15734.25;           // NTSC horizontal blanks
    }
}

u64 Ticks(const Timer& t, u64 now) {
    if (!(t.mode & kCue) || now <= t.origin_ns) return 0;
    return u64(double(now - t.origin_ns) * Rate(t.mode) / 1e9);
}

// The period the count wraps at, and the count `ticks` after the origin.
u64 Period(const Timer& t) { return (t.mode & kZret) ? u64(t.comp & 0xFFFF) + 1 : 0x10000; }

u32 CountAt(const Timer& t, u64 ticks) {
    return u32((u64(t.origin_count) + ticks) % Period(t));
}

// How many times the count is `target` within ticks (from, to] after the origin.
u64 Passes(const Timer& t, u64 from, u64 to, u64 target) {
    const u64 period = Period(t), start = t.origin_count;
    auto upto = [&](u64 ticks) -> u64 {       // passes in (0, ticks]
        const u64 value = start + ticks;      // the unwrapped count
        if (value < target) return 0;
        return (value - target) / period + 1 - (start >= target ? (start - target) / period + 1 : 0);
    };
    return upto(to) - upto(from);
}

void Rebase(Timer& t, u64 now) {
    t.origin_count = CountAt(t, Ticks(t, now));
    t.origin_ns = now;
    t.checked = 0;
}

}  // namespace

void InstallTimers() {
    Memory& m = GuestMemory();
    for (u32 n = 0; n < 4; ++n) {
        const u32 base = kBases[n];
        m.OnRead(base, [n](u32, u64) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            const Timer& t = s.t[n];
            return u64(CountAt(t, Ticks(t, NowNs())));
        });
        m.OnWrite(base, [n](u32, u64 value) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            Timer& t = s.t[n];
            t.origin_count = u32(value) & 0xFFFF;
            t.origin_ns = NowNs();
            t.checked = 0;
        });
        m.OnRead(base + 0x10, [n](u32, u64) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            return u64(s.t[n].mode);
        });
        m.OnWrite(base + 0x10, [n](u32, u64 value) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            Timer& t = s.t[n];
            const u64 now = NowNs();
            Rebase(t, now);                    // the count so far, at the old clock
            const u32 v = u32(value);
            const u32 flags = (t.mode & (kEqUf | kOvFf)) & ~(v & (kEqUf | kOvFf));
            t.mode = (v & 0x3FF) | flags;
        });
        m.OnRead(base + 0x20, [n](u32, u64) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            return u64(s.t[n].comp);
        });
        m.OnWrite(base + 0x20, [n](u32, u64 value) {
            Timers& s = T();
            std::lock_guard<std::mutex> hold(s.lock);
            Timer& t = s.t[n];
            Rebase(t, NowNs());
            t.comp = u32(value) & 0xFFFF;
        });
    }
}

s64 NanosToNextTimerEvent() {
    Timers& s = T();
    std::lock_guard<std::mutex> hold(s.lock);
    const u64 now = NowNs();
    s64 best = -1;
    for (const Timer& t : s.t) {
        if (!(t.mode & kCue) || !(t.mode & (kCmpe | kOvfe))) continue;
        const u64 ticks = Ticks(t, now), count = CountAt(t, ticks), period = Period(t);
        u64 wait = ~u64(0);
        if (t.mode & kCmpe) {
            const u64 target = t.comp & 0xFFFF;
            wait = target > count ? target - count : target + period - count;
        }
        if ((t.mode & kOvfe) && !(t.mode & kZret)) wait = std::min(wait, period - count);
        const s64 ns = s64(double(wait) * 1e9 / Rate(t.mode)) + 1;
        if (best < 0 || ns < best) best = ns;
    }
    return best;
}

void DeliverTimers() {
    u32 due = 0;
    {
        Timers& s = T();
        std::lock_guard<std::mutex> hold(s.lock);
        const u64 now = NowNs();
        for (u32 n = 0; n < 4; ++n) {
            Timer& t = s.t[n];
            if (!(t.mode & kCue)) continue;
            const u64 ticks = Ticks(t, now);
            if (ticks == t.checked) continue;
            bool raise = false;
            if (Passes(t, t.checked, ticks, t.comp & 0xFFFF)) {
                if (t.mode & kCmpe) { t.mode |= kEqUf; raise = true; }
            }
            if (!(t.mode & kZret) && Passes(t, t.checked, ticks, 0)) {
                if (t.mode & kOvfe) { t.mode |= kOvFf; raise = true; }
            }
            t.checked = ticks;
            if (raise) due |= 1u << n;
        }
    }
    // Handlers run without the lock: they read and write the timers.
    for (u32 n = 0; n < 4; ++n)
        if (due & (1u << n)) RaiseIntc(kFirstCause + n);
}

}  // namespace ps2host
