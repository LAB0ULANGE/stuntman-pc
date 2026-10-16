/*
 * The frame-pacing profiler (timeline.h): per-thread buffers of spans and
 * moments, written at the end as a Chrome trace, with a summary of the
 * frames - their pacing, the dropped ones and what was busy before them.
 */

#include "ps2host/timeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ps2host::timeline {

namespace {

using u64 = std::uint64_t;

struct Event {
    u64 from, to;               // to == from: a moment
    const char* name;
    u64 value;
};

struct Lane {
    std::string name;
    std::vector<Event> events;
};

std::mutex g_lanes_mutex;
std::vector<Lane*>& Lanes() {
    static std::vector<Lane*> lanes;            // kept to the end (a thread's lane outlives it)
    return lanes;
}

const char* Path() {
    static const char* path = std::getenv("PS2HOST_PROFILE");
    return path && *path ? path : nullptr;
}

u64 FromFrame() {
    static const u64 from = std::getenv("PS2HOST_PROFILE_FROM") ? std::strtoull(std::getenv("PS2HOST_PROFILE_FROM"), nullptr, 10) : 0;
    return from;
}

std::atomic<bool> g_on{false};
std::atomic<u64> g_frame{0};

// The frames and pictures, from every thread: rare, so under a lock.
std::mutex g_frames_mutex;
struct Picture { u64 due, shown; bool dropped; };
std::vector<u64> g_blanks;                      // when each frame began (Frame)
std::vector<u64> g_lost;                        // when a blank was lost
std::vector<Picture> g_pictures;

Lane& ThisLane() {
    thread_local Lane* lane = nullptr;
    if (!lane) {
        lane = new Lane;
        lane->events.reserve(1 << 16);
        std::lock_guard<std::mutex> hold(g_lanes_mutex);
        lane->name = "thread " + std::to_string(Lanes().size());
        Lanes().push_back(lane);
    }
    return *lane;
}

}  // namespace

bool On() { return g_on.load(std::memory_order_relaxed); }

u64 Now() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
}

void NameThread(const char* name) {
    if (!Path()) return;
    Lane& lane = ThisLane();
    std::lock_guard<std::mutex> hold(g_lanes_mutex);
    lane.name = name;
}

void Span(const char* name, u64 from_ns, u64 to_ns) {
    if (!On() || to_ns <= from_ns) return;
    Lane& lane = ThisLane();
    if (!lane.events.empty()) {
        Event& last = lane.events.back();
        if (last.name == name && last.to <= from_ns && from_ns - last.to < 20'000 && last.to != last.from) {
            last.to = to_ns;
            return;
        }
    }
    lane.events.push_back({from_ns, to_ns, name, 0});
}

void Instant(const char* name, u64 at_ns, u64 value) {
    if (!On()) return;
    ThisLane().events.push_back({at_ns, at_ns, name, value});
}

void Frame(u64 frame) {
    if (!Path()) return;
    g_frame.store(frame, std::memory_order_relaxed);
    if (!On() && frame >= FromFrame()) g_on.store(true);
    if (!On()) return;
    const u64 now = Now();
    Instant("frame", now, frame);
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_blanks.push_back(now);
}

void BlankLost() {
    if (!On()) return;
    const u64 now = Now();
    Instant("blank lost", now, g_frame.load(std::memory_order_relaxed));
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_lost.push_back(now);
}

void Presented(u64 due_ns, u64 shown_ns) {
    if (!On()) return;
    Span("present", shown_ns > 2'000'000 ? shown_ns - 2'000'000 : shown_ns, shown_ns);   // marked as the 2 ms before it
    if (shown_ns > due_ns + 1'000'000) Instant("shown after its time (us)", shown_ns, (shown_ns - due_ns) / 1000);
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_pictures.push_back({due_ns, shown_ns, false});
}

void Dropped(u64 due_ns) {
    if (!On()) return;
    Instant("picture dropped", Now());
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_pictures.push_back({due_ns, 0, true});
}

namespace {

std::string Escape(const std::string& s) {
    std::string out;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        out += ch;
    }
    return out;
}

/* How long each lane was busy in [from, to): its spans' overlap, idle
   waits not counted. */
std::vector<std::pair<std::string, double>> BusyIn(u64 from, u64 to) {
    std::map<std::string, double> busy;
    for (const Lane* lane : Lanes())
        for (const Event& e : lane->events) {
            if (e.to == e.from || e.to <= from || e.from >= to || std::strcmp(e.name, "idle") == 0) continue;
            const u64 a = std::max(e.from, from), b = std::min(e.to, to);
            busy[lane->name + ": " + e.name] += double(b - a) / 1e6;
        }
    std::vector<std::pair<std::string, double>> sorted(busy.begin(), busy.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return sorted;
}

}  // namespace

void Finish() {
    if (!Path()) return;
    g_on.store(false);
    std::lock_guard<std::mutex> hold_lanes(g_lanes_mutex);
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    u64 origin = ~u64(0);
    for (const Lane* lane : Lanes())
        if (!lane->events.empty()) origin = std::min(origin, lane->events.front().from);
    if (origin == ~u64(0)) {
        std::fprintf(stderr, "[profile] nothing recorded (PS2HOST_PROFILE_FROM past the run's end?)\n");
        return;
    }
    if (std::FILE* f = std::fopen(Path(), "w")) {
        std::fprintf(f, "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
        bool first = true;
        int tid = 0;
        for (const Lane* lane : Lanes()) {
            ++tid;
            std::fprintf(f, "%s{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s\"}}",
                         first ? "" : ",\n", tid, Escape(lane->name).c_str());
            first = false;
            for (const Event& e : lane->events) {
                const double ts = double(e.from - origin) / 1e3;
                if (e.to == e.from)
                    std::fprintf(f, ",\n{\"ph\":\"i\",\"s\":\"t\",\"name\":\"%s\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"args\":{\"value\":%llu}}",
                                 e.name, tid, ts, (unsigned long long)e.value);
                else
                    std::fprintf(f, ",\n{\"ph\":\"X\",\"name\":\"%s\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f}",
                                 e.name, tid, ts, double(e.to - e.from) / 1e3);
            }
        }
        std::fprintf(f, "\n]}\n");
        std::fclose(f);
    }
    // The summary: the pictures as shown.
    std::vector<u64> shown;
    unsigned dropped = 0;
    for (const Picture& p : g_pictures) {
        if (p.dropped) ++dropped;
        else shown.push_back(p.shown);
    }
    std::sort(shown.begin(), shown.end());
    std::vector<double> intervals;
    for (std::size_t i = 1; i < shown.size(); ++i) intervals.push_back(double(shown[i] - shown[i - 1]) / 1e6);
    std::fprintf(stderr, "\n=== frame pacing (PS2HOST_PROFILE) ===\n");
    std::fprintf(stderr, "%zu blanks, %zu pictures shown, %u dropped before they were shown, %zu blanks lost by the EE\n",
                 g_blanks.size(), shown.size(), dropped, g_lost.size());
    if (intervals.empty()) {
        std::fprintf(stderr, "no pictures shown (no window, or the software GS's): the blanks alone\n");
        std::vector<double> gaps;
        for (std::size_t i = 1; i < g_blanks.size(); ++i) gaps.push_back(double(g_blanks[i] - g_blanks[i - 1]) / 1e6);
        intervals = gaps;
    }
    if (intervals.empty()) return;
    std::vector<double> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    const auto at = [&](double q) { return sorted[std::min(sorted.size() - 1, std::size_t(q * double(sorted.size())))]; };
    const double nominal = at(0.5);
    std::fprintf(stderr, "interval: median %.2f ms, 95%% %.2f, 99%% %.2f, 99.9%% %.2f, longest %.2f\n", nominal, at(0.95),
                 at(0.99), at(0.999), sorted.back());
    unsigned late = 0, repeats = 0;
    for (double ms : intervals)
        if (ms > nominal * 1.5) {
            ++late;
            repeats += unsigned(ms / nominal + 0.5) - 1;
        }
    std::fprintf(stderr, "frames shown late: %u (%u frames' time the picture repeated)\n", late, repeats);
    // The worst: what was busy in the frame before each.
    const std::vector<u64>& times = shown.size() > 1 ? shown : g_blanks;
    std::vector<std::size_t> worst;
    for (std::size_t i = 0; i < intervals.size(); ++i)
        if (intervals[i] > nominal * 1.5) worst.push_back(i);
    std::sort(worst.begin(), worst.end(), [&](std::size_t a, std::size_t b) { return intervals[a] > intervals[b]; });
    if (worst.size() > 12) worst.resize(12);
    for (std::size_t i : worst) {
        const u64 to = times[i + 1], from = to - u64(intervals[i] * 1e6) - u64(nominal * 1e6);
        std::fprintf(stderr, "  at %.3f s: %.1f ms; busy then:", double(to - origin) / 1e9, intervals[i]);
        int n = 0;
        for (const auto& [what, ms] : BusyIn(from, to)) {
            if (n++ == 5) break;
            std::fprintf(stderr, " %s %.1f ms;", what.c_str(), ms);
        }
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "the timeline: %s (ui.perfetto.dev)\n", Path());
}

}  // namespace ps2host::timeline
