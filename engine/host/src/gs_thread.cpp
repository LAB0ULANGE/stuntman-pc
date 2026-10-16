/*
 * The GS on a thread of its own (run.h: StartGsThread).
 *
 * On the console the GS draws while the EE and VU1 go on; to the EE it is a
 * sink - the host answers CSR reads itself, and nothing the GS does comes back
 * before the frame is shown. So the host's GS model (and the GPU bridge behind
 * it) can take its packets on another thread: the EE side writes each call -
 * a quadword, a run of words, a privileged register - into a ring, in order,
 * and the GS thread makes the same calls in the same order. Nothing reads the
 * GS's state but at the end of a frame (the picture, the GPU's frame, the
 * dumps) and of the run, and there SyncGs waits for the ring to empty: the GS
 * sees the same calls as without the thread, and a scripted run draws the
 * same pictures.
 *
 * One producer (the EE thread, which runs the DMA, VIF and VU1 that feed the
 * GS) and one consumer. An exception on the GS thread is handed to the EE
 * thread at its next call here.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "ps2host/gs.h"
#include "ps2host/timeline.h"
#include "ps2host/profile.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kRing = 1u << 22;                 // words (16 MB)
constexpr u32 kMaxRun = kRing / 8;              // the longest run of words one entry carries
enum : u32 { kQword = 1, kWord = 2, kWords = 3, kPrivileged = 4, kWrap = 5, kFrame = 6 };

std::atomic<u64> g_gs_busy_us{0};       // the GS thread at work, not waiting for more (--frame-times)

// GsFrame's work, on the GS thread at the frame's place in the stream.
std::function<void(u64)> g_frame_handler;

// When the EE ended each of the last frames (GsFrame), by frame number.
std::atomic<u64> g_frame_ended[256];

// PS2HOST_HITCH: each frame's way to the screen - when the EE ended it (and
// its idle time and VIF1's worker's busy time then), when the worker passed
// it on, when the GS thread took and finished it - and a frame that came a
// period late, or took long on its way, told.
struct FrameWay { std::atomic<u64> ee{0}, worker{0}, idle{0}, worker_us{0}; };
FrameWay g_way[256];
const bool g_hitch = std::getenv("PS2HOST_HITCH") != nullptr;

void HitchFrame(u64 frame, u64 took, u64 done) {
    static u64 last_blank = 0, last_busy = 0;
    const u64 blank = g_frame_ended[frame & 255].load(std::memory_order_relaxed);
    const u64 busy = g_gs_busy_us.load(std::memory_order_relaxed);
    const u64 gap = blank && last_blank && blank > last_blank ? blank - last_blank : 0;
    const auto ms = [&](u64 t) { return t && blank ? (double(t) - double(blank)) / 1e6 : -1.0; };
    const FrameWay& w = g_way[frame & 255];
    const FrameWay& before = g_way[(frame - 1) & 255];
    const u64 idle = w.idle.load(std::memory_order_relaxed), idle_before = before.idle.load(std::memory_order_relaxed);
    const u64 vu = w.worker_us.load(std::memory_order_relaxed), vu_before = before.worker_us.load(std::memory_order_relaxed);
    const u64 ee = w.ee.load(std::memory_order_relaxed), ee_before = before.ee.load(std::memory_order_relaxed);
    if (ms(done) > 22.0 || gap > 20'000'000)
        std::fprintf(stderr, "[hitch] frame %llu: blank %+.1f ms after the last; EE %.1f, VIF1 %.1f, GS took %.1f done %.1f ms after it; GS busy %.1f ms; EE frame %.1f ms, idle %.1f, VIF1 worker busy %.1f\n",
                     (unsigned long long)frame, last_blank ? gap / 1e6 : 0.0, ms(ee),
                     ms(w.worker.load(std::memory_order_relaxed)), ms(took), ms(done),
                     (busy - last_busy) / 1e3, ee > ee_before ? (ee - ee_before) / 1e6 : -1.0,
                     idle >= idle_before ? (idle - idle_before) / 1e6 : -1.0,
                     vu >= vu_before ? (vu - vu_before) / 1e3 : -1.0);
    last_blank = blank;
    last_busy = busy;
}

// With VIF1 on its worker (vif1_worker.cpp) the worker feeds the GS: what
// the EE would feed it goes through the worker's queue, in turn (privileged
// registers, frame ends), or waits for the queue to empty first.
struct Producing {
    Producing() {
        if (Vif1Async() && !OnVif1Worker()) JoinVif1("producing");
    }
};

u32 Header(u32 type, int path, u32 count) { return type << 28 | u32(path & 0xF) << 24 | (count & 0xFFFFFF); }

void Pause() {
#if defined(_MSC_VER)
    _mm_pause();
#endif
}

struct Feed {
    std::vector<u32> ring = std::vector<u32>(kRing);
    // Positions in words, counted from the start (the ring index is mod kRing).
    std::atomic<u64> head{0};                   // written up to, by the EE thread
    std::atomic<u64> tail{0};                   // done up to, by the GS thread
    std::atomic<bool> stop{false};
    std::atomic<bool> sleeping{false};
    std::mutex mutex;
    std::condition_variable wake;
    std::exception_ptr failed;                  // set by the GS thread, taken by the EE's
    std::atomic<bool> has_failed{false};
    std::thread thread;

    ~Feed() { Stop(); }

    void Stop() {
        if (!thread.joinable()) return;
        stop = true;
        {
            std::lock_guard<std::mutex> lock(mutex);
            wake.notify_one();
        }
        thread.join();
    }

    void Rethrow() {
        if (has_failed.load()) {
            has_failed = false;
            std::rethrow_exception(failed);
        }
    }

    /* Room for `words` contiguous words: their index in the ring. */
    u32* Reserve(u32 words) {
        Rethrow();
        u64 at = head.load(std::memory_order_relaxed);
        if (at % kRing + words > kRing) {       // no room before the end: wrap
            WaitFor(kRing - at % kRing);
            ring[at % kRing] = Header(kWrap, 0, 0);
            at += kRing - at % kRing;
            head.store(at, std::memory_order_release);
        }
        WaitFor(words);
        return &ring[at % kRing];
    }

    void WaitFor(u32 words) {
        const u64 at = head.load(std::memory_order_relaxed);
        if (at + words - tail.load(std::memory_order_acquire) <= kRing) return;
        Timed timed(Section::GsWait);
        while (at + words - tail.load(std::memory_order_acquire) > kRing) {
            Rethrow();
            std::this_thread::yield();
        }
    }

    void Publish(u32 words) {
        head.store(head.load(std::memory_order_relaxed) + words, std::memory_order_release);
        if (sleeping.load()) {
            std::lock_guard<std::mutex> lock(mutex);
            wake.notify_one();
        }
    }

    void Sync() {
        const u64 target = head.load(std::memory_order_relaxed);
        if (tail.load(std::memory_order_acquire) >= target) return Rethrow();
        Timed timed(Section::GsWait);
        for (int spins = 0; tail.load(std::memory_order_acquire) < target; ++spins) {
            Rethrow();
            if (spins < 4096) Pause();
            else std::this_thread::yield();
        }
        Rethrow();
    }

    void Run() {
        gs::Gs& g = gs::TheGs();
        while (true) {
            u64 t = tail.load(std::memory_order_relaxed);
            u64 h = head.load(std::memory_order_acquire);
            if (t == h) {
                // Briefly busy, then asleep until there is more.
                for (int spins = 0; spins < 200 && t == h; ++spins) {
                    Pause();
                    h = head.load(std::memory_order_acquire);
                }
                if (t == h) {
                    std::unique_lock<std::mutex> lock(mutex);
                    sleeping = true;
                    wake.wait(lock, [&] { return stop.load() || head.load() != tail.load(); });
                    sleeping = false;
                    if (stop.load() && head.load() == tail.load()) return;
                    continue;
                }
            }
            const auto began = std::chrono::steady_clock::now();
            struct Busy {
                std::chrono::steady_clock::time_point began;
                ~Busy() {
                    const auto now = std::chrono::steady_clock::now();
                    g_gs_busy_us.fetch_add(u64(std::chrono::duration_cast<std::chrono::microseconds>(now - began).count()),
                                           std::memory_order_relaxed);
                    if (timeline::On())
                        timeline::Span("busy", u64(std::chrono::duration_cast<std::chrono::nanoseconds>(began.time_since_epoch()).count()),
                                       u64(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()));
                }
            } busy{began};
            try {
                while (t < h) {
                    const u32 at = u32(t % kRing);
                    const u32* e = &ring[at];
                    const u32 type = e[0] >> 28, count = e[0] & 0xFFFFFF;
                    const int path = int((e[0] >> 24) & 0xF);
                    u32 size = 1;
                    switch (type) {
                        case kQword:
                            g.Qword(path, u64(e[1]) | u64(e[2]) << 32, u64(e[3]) | u64(e[4]) << 32);
                            size = 5;
                            break;
                        case kWord:
                            g.Word(path, e[1]);
                            size = 2;
                            break;
                        case kWords:
                            g.Words(path, e + 2, count, e[1]);
                            size = 2 + count;
                            break;
                        case kPrivileged:
                            g.SetPrivileged(e[1], u64(e[2]) | u64(e[3]) << 32);
                            size = 4;
                            break;
                        case kFrame:
                            if (g_frame_handler) {
                                const u64 frame = u64(e[1]) | u64(e[2]) << 32;
                                const u64 took = g_hitch ? ClockNs() : 0;
                                g_frame_handler(frame);
                                if (g_hitch) HitchFrame(frame, took, ClockNs());
                            }
                            size = 3;
                            break;
                        case kWrap:
                            size = kRing - at;
                            break;
                        default:
                            break;
                    }
                    t += size;
                    tail.store(t, std::memory_order_release);
                }
            } catch (...) {
                failed = std::current_exception();
                has_failed = true;
                // Skip what is left: the EE thread stops at its next call.
                tail.store(head.load(std::memory_order_acquire), std::memory_order_release);
            }
        }
    }
};

Feed* g_feed = nullptr;

Feed& TheFeed() {
    static Feed feed;
    return feed;
}

}  // namespace

void StartGsThread() {
    if (g_feed) return;
    g_feed = &TheFeed();
    g_feed->thread = std::thread([] {
        timeline::NameThread("GS");
        FastThread();
        if (SampleGsThread()) StartSampler();
        g_feed->Run();
    });
}

void GsDrain() {
    if (Vif1Async() && !OnVif1Worker()) JoinVif1("drain");
    if (g_feed) g_feed->Sync();
}

void StopGsThread() {
    if (!g_feed) return;
    g_feed->Sync();
    g_feed->Stop();
    g_feed = nullptr;
}

void SyncGs() {
    JoinVif1("syncgs");
    if (g_feed) g_feed->Sync();
}

void GsQword(int path, u64 low, u64 high) {
    if (!g_feed) return gs::TheGs().Qword(path, low, high);
    Producing producing;
    u32* e = g_feed->Reserve(5);
    e[0] = Header(kQword, path, 0);
    e[1] = u32(low);
    e[2] = u32(low >> 32);
    e[3] = u32(high);
    e[4] = u32(high >> 32);
    g_feed->Publish(5);
}

void GsWord(int path, u32 word) {
    if (!g_feed) return gs::TheGs().Word(path, word);
    Producing producing;
    u32* e = g_feed->Reserve(2);
    e[0] = Header(kWord, path, 0);
    e[1] = word;
    g_feed->Publish(2);
}

void GsWords(int path, const u32* words, u32 count, u32 from) {
    if (!g_feed) return gs::TheGs().Words(path, words, count, from);
    Producing producing;
    while (count) {
        const u32 run = count < kMaxRun ? count : kMaxRun;
        u32* e = g_feed->Reserve(2 + run);
        e[0] = Header(kWords, path, run);
        e[1] = from;
        std::memcpy(e + 2, words, std::size_t(run) * 4);
        g_feed->Publish(2 + run);
        words += run;
        count -= run;
        if (from != gs::Gs::kNoSource) from += 4 * run;
    }
}

void SetGsFrameHandler(std::function<void(u64)> handler) { g_frame_handler = std::move(handler); }


u64 FrameEndedNs(u64 frame) { return g_frame_ended[frame & 255].load(std::memory_order_relaxed); }

/* The steady clock, as the vertical blank's timer and the display keep time
   (ClockNs goes on from a virtual clock's time after --real-time-from). */
u64 SteadyNowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
}

u64 FrameDue(u64 frame) {
    // The delay each frame came with, the longest of the last 120 (two
    // seconds: a race's heavy moments come together) the one
    // every frame keeps to (and half a millisecond more), so frames go out
    // a frame's period apart however much each took on its way; never more
    // than a period and a half behind - a frame later than that goes at once.
    static u64 delays[120] = {};
    static int next = 0;
    const u64 ended = FrameEndedNs(frame);
    const u64 now = SteadyNowNs();
    if (!ended || now < ended) return now;
    delays[next] = now - ended;
    next = (next + 1) % 120;
    if (timeline::On()) timeline::Instant("frame's delay (us)", timeline::Now(), (now - ended) / 1000);
    u64 keep = 0;
    for (u64 d : delays) keep = std::max(keep, d);
    // At least 6 ms: a frame that alone takes longer on its way (a heavy
    // frame's VU1 work, 12 ms past its blank) still goes out on time, for
    // under a third of a frame's delay where all is quick.
    keep = std::min<u64>(std::max<u64>(keep + 500'000, 6'000'000), 25'000'000);
    return std::max(now, ended + keep);
}

u64 GsBusyMicroseconds() { return g_gs_busy_us.load(std::memory_order_relaxed); }

bool GsFrame(u64 frame) {
    if (!g_frame_handler) return false;
    if (!OnVif1Worker()) {
        // The frame's vertical blank as the timer had it: frames a period
        // apart however late the EE took each (a long loop runs on to its
        // next loop head first).
        const u64 blank = VirtualClock() ? 0 : LastVblankNs();
        g_frame_ended[frame & 255].store(blank ? blank : SteadyNowNs(), std::memory_order_relaxed);
        if (g_hitch) {
            g_way[frame & 255].ee.store(ClockNs(), std::memory_order_relaxed);
            g_way[frame & 255].idle.store(IdleNanoseconds(), std::memory_order_relaxed);
            g_way[frame & 255].worker_us.store(Vif1BusyMicroseconds(), std::memory_order_relaxed);
        }
    } else if (g_hitch) {
        g_way[frame & 255].worker.store(ClockNs(), std::memory_order_relaxed);
    }
    if (Vif1Async() && !OnVif1Worker()) {
        Vif1RunFrame([frame] { GsFrame(frame); });
        return true;
    }
    if (!g_feed) {
        g_frame_handler(frame);
        return true;
    }
    u32* e = g_feed->Reserve(3);
    e[0] = Header(kFrame, 0, 0);
    e[1] = u32(frame);
    e[2] = u32(frame >> 32);
    g_feed->Publish(3);
    return true;
}

void GsPrivileged(u32 address, u64 value) {
    if (Vif1Async() && !OnVif1Worker()) {
        Vif1Run([address, value] { GsPrivileged(address, value); });
        return;
    }
    if (!g_feed) return gs::TheGs().SetPrivileged(address, value);
    u32* e = g_feed->Reserve(4);
    e[0] = Header(kPrivileged, 0, 0);
    e[1] = address;
    e[2] = u32(value);
    e[3] = u32(value >> 32);
    g_feed->Publish(4);
}

}  // namespace ps2host
