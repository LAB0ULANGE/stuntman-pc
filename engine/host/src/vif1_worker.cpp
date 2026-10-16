/*
 * VIF1, VU1 and what they feed the GS, on a thread of their own (real time;
 * run.h: StartVif1Worker) - as PCSX2's MTVU does it.
 *
 * On the console VIF1, VU1 and the GS work through a frame's display list
 * while the EE goes on to the next frame's logic. Walked on the EE's thread,
 * the list - VIF1's unpacking, every VU1 program it starts, their XGKICKs -
 * was most of the thread's time, and the EE's own work waited behind it: the
 * race drew a picture every other vertical blank where the console draws one
 * each, its VU1 work alone 15 ms of the 16.7.
 *
 * Here what goes to VIF1 and the GS is a queue of commands the worker runs in
 * order (Vif1Run): a VIF1 transfer's words, copied out of memory as its DMA
 * chain is walked on the EE's thread (the transfer ends there and then, its
 * interrupt as before); a PATH3 transfer's, likewise, so the GIF takes it in
 * turn with VIF1's; the GS's privileged registers the EE writes (the display
 * flip); the end of each frame (the picture shown). So a frame's picture
 * comes after all its drawing, while the EE makes the next.
 *
 * The EE waits for the queue to empty (JoinVif1) before anything that could
 * see the worker's state or race with it: VIF1's and the GIF's registers and
 * FIFOs, VU1's memories (memory.cpp), and anything else it would feed the GS
 * directly. Virtual time runs each command at once, on the EE's thread: its
 * runs stay the same run.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "ps2host/profile.h"
#include "ps2host/timeline.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

thread_local bool t_on_worker = false;

// Queued and not yet run, at most: past these the EE waits (the worker runs
// behind the EE by this many commands, no more).
constexpr std::size_t kMaxQueued = 4096;

struct Worker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable wake, drained;
    std::deque<std::function<void()>> queue;
    bool stop = false;
    std::atomic<std::size_t> pending{0};    // queued or running
    std::atomic<int> frames{0};             // frame ends queued or running (Vif1RunFrame)
    std::exception_ptr failed;
    std::atomic<u64> busy_us{0};            // --frame-times: the worker's working time

    void Run() {
        t_on_worker = true;
        FastThread();
        if (SampleVif1Thread()) StartSampler();
        for (;;) {
            std::function<void()> command;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) return;          // stopping, all run
                command = std::move(queue.front());
                queue.pop_front();
            }
            const auto began = std::chrono::steady_clock::now();
            try {
                if (!failed) command();
            } catch (...) {
                failed = std::current_exception();
            }
            const auto ended = std::chrono::steady_clock::now();
            busy_us += u64(std::chrono::duration_cast<std::chrono::microseconds>(ended - began).count());
            if (timeline::On())
                timeline::Span("busy", u64(std::chrono::duration_cast<std::chrono::nanoseconds>(began.time_since_epoch()).count()),
                               u64(std::chrono::duration_cast<std::chrono::nanoseconds>(ended.time_since_epoch()).count()));
            {
                std::lock_guard<std::mutex> lock(mutex);
                --pending;
            }
            drained.notify_all();
        }
    }
};

Worker* g_worker = nullptr;

// --frame-times: the EE's time waiting for the worker - for the queue to
// empty (JoinVif1), for it to come within kFramesBehind frames, for room.
std::atomic<u64> g_join_us{0}, g_frame_wait_us{0}, g_full_us{0};

struct Waited {
    std::atomic<u64>& total;
    std::chrono::steady_clock::time_point began = std::chrono::steady_clock::now();
    ~Waited() {
        total += u64(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began)
                         .count());
    }
};

Worker& TheWorker() {
    static Worker worker;
    return worker;
}

void StopWorker() {
    if (!g_worker) return;
    {
        std::lock_guard<std::mutex> lock(g_worker->mutex);
        g_worker->stop = true;
    }
    g_worker->wake.notify_all();
    if (g_worker->thread.joinable()) g_worker->thread.join();
    g_worker = nullptr;
}

void Rethrow(Worker& w) {
    if (!w.failed) return;
    std::exception_ptr failed = w.failed;
    w.failed = nullptr;
    std::rethrow_exception(failed);
}

}  // namespace

void StartVif1Worker() {
    if (g_worker) return;
    g_worker = &TheWorker();
    g_worker->thread = std::thread([] {
        timeline::NameThread("VIF1 worker");
        g_worker->Run();
    });
    std::atexit(StopWorker);
}

bool Vif1Async() { return g_worker != nullptr; }

bool OnVif1Worker() { return t_on_worker; }

bool Vif1Busy() { return g_worker && g_worker->pending.load(std::memory_order_acquire) != 0; }

u64 Vif1BusyMicroseconds() { return g_worker ? g_worker->busy_us.load() : 0; }

void Vif1WaitMicroseconds(u64* join, u64* frame, u64* full) {
    *join = g_join_us.load();
    *frame = g_frame_wait_us.load();
    *full = g_full_us.load();
}

void Vif1Run(std::function<void()> command) {
    if (!g_worker || t_on_worker) {
        command();
        return;
    }
    Worker& w = *g_worker;
    std::unique_lock<std::mutex> lock(w.mutex);
    if (w.pending.load() >= kMaxQueued) {
        Timed timed(Section::GsWait);
        Waited waited{g_full_us};
        w.drained.wait(lock, [&] { return w.pending.load() < kMaxQueued / 2; });
    }
    ++w.pending;
    w.queue.push_back(std::move(command));
    lock.unlock();
    w.wake.notify_one();
}

void Vif1RunFrame(std::function<void()> command) {
    if (!g_worker || t_on_worker) {
        command();
        return;
    }
    Worker& w = *g_worker;
    // The worker at most kFramesBehind frames behind the EE: what is shown
    // lags what the game has done by no more (input to picture), and the EE
    // waits for the worker, not the other way round.
    constexpr int kFramesBehind = 2;
    if (w.frames.load() >= kFramesBehind) {
        Timed timed(Section::GsWait);
        Waited waited{g_frame_wait_us};
        std::unique_lock<std::mutex> lock(w.mutex);
        w.drained.wait(lock, [&] { return w.frames.load() < kFramesBehind; });
    }
    ++w.frames;
    Vif1Run([&w, command = std::move(command)] {
        command();
        --w.frames;
    });
}

namespace {

// PS2HOST_JOIN_LOG: the EE's waits for the worker by what waited (a memory
// read or write by its address), at exit - where to look when --frame-times
// shows the EE waiting.
std::mutex g_join_log_mutex;
std::map<std::pair<std::string, u32>, std::pair<u64, u64>> g_join_log;     // joins, microseconds

void LogJoin(const char* why, u32 address, u64 us) {
    std::lock_guard<std::mutex> lock(g_join_log_mutex);
    if (g_join_log.empty())
        std::atexit([] {
            for (const auto& [key, joins] : g_join_log)
                std::fprintf(stderr, "[join] %s %08X: %llu joins, %llu us\n", key.first.c_str(), unsigned(key.second),
                             (unsigned long long)joins.first, (unsigned long long)joins.second);
        });
    auto& joins = g_join_log[{why ? why : "?", address}];
    ++joins.first;
    joins.second += us;
}

}  // namespace

void JoinVif1(const char* why, u32 address) {
    if (!g_worker || t_on_worker) return;
    Worker& w = *g_worker;
    if (w.pending.load(std::memory_order_acquire) != 0) {
        Timed timed(Section::GsWait);
        const auto began = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            w.drained.wait(lock, [&] { return w.pending.load() == 0; });
        }
        const u64 us = u64(std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - began).count());
        g_join_us += us;
        static const bool log = std::getenv("PS2HOST_JOIN_LOG") != nullptr;
        if (log) LogJoin(why, address, us);
    }
    Rethrow(w);
}

}  // namespace ps2host
