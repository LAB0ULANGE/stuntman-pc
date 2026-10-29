#include "ps2host/host.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

#include "ps2host/disc.h"
#include "ps2host/gs.h"
#include "ps2host/iop_vm.h"
#include "ps2host/vif.h"
#include "ps2host/vu.h"
#include "ps2host/mc2.h"
#include "ps2host/memory.h"
#include "ps2host/profile.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/timeline.h"
#include "ps2rt/generated.h"

namespace ps2host {

/* The renderer's own choices (PC settings, or the options that override
   them): a run's, and a replay's (GsReplay) - the GPU time a replay
   measures is then the game's. */
void ApplyRendererSettings(const Options& options) {
    // The texture cache: asked, else the setting's, on by default.
    const char* check = std::getenv("PS2HOST_TEXTURE_CACHE_CHECK");
    SetTextureCache(!options.texture_cache.empty() ? options.texture_cache
                    : check && *check == '1'       ? std::string("check")
                                                   : Setting("texture_cache", "on"));
    SetBatchOverlap(!options.batch_overlap.empty() ? options.batch_overlap : Setting("batch_overlap", "exact"));
    SetMemoryWrites(Setting("memory_writes", "batch"));
    SetTextureDecode(Setting("texture_decode", "direct"));
    SetTextureReuse(SettingOn("texture_reuse", std::getenv("PS2HOST_TEXTURE_REUSE") != nullptr));
    SetBarriers(Setting("barriers", std::getenv("PS2HOST_BARRIERS") ? std::getenv("PS2HOST_BARRIERS") : "all"));
    SetNewerList((!options.newer_list.empty() ? options.newer_list : Setting("newer_list", "on")) != "off");
    SetOwnedIndex((!options.owned_index.empty() ? options.owned_index : Setting("owned_index", "on")) != "off");
    SetColclampWrap(Setting("colclamp_wrap", "fast"));
    SetDateArea(!options.date_area.empty() ? options.date_area : Setting("date_area", "batch"));
    SetDateInPass(!options.date_in_pass.empty() ? options.date_in_pass : Setting("date_in_pass", "on"));
}

/* --gs-replay <file> <dir>: a GS stream (engine/tools/pcsx2dump.py's, from a
   PCSX2 GS dump) through the GS model alone: local memory and registers set,
   then GIF packets, privileged registers and vertical blanks in turn - each
   blank's picture to <dir>/replay_<n>.bmp, the GS trace to <dir>/trace.txt.
   What the model draws from the console's own commands, beside what PCSX2
   drew (the dump's picture): a GS fault or one before it, told apart.
   PS2HOST_REPLAY_GPU=1: the GPU renderer draws instead (as --gpu), each
   picture to <dir>/gpu_<n>/ (screen.bmp at the internal resolution,
   PS2HOST_REPLAY_SCALE, default 1; screen_1x.bmp). */
int GsReplay(const std::string& file, const std::string& out) {
    std::FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "gs-replay: cannot open %s\n", file.c_str());
        return 66;
    }
    std::vector<unsigned char> data;
    {
        unsigned char buffer[1 << 16];
        std::size_t n;
        while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) data.insert(data.end(), buffer, buffer + n);
        std::fclose(f);
    }
    const std::size_t kMemory = 4u << 20;
    if (data.size() < 5 + kMemory || std::memcmp(data.data(), "GSRP1", 5) != 0) {
        std::fprintf(stderr, "gs-replay: not a replay stream\n");
        return 65;
    }
    std::filesystem::create_directories(out);
    gs::Gs& g = gs::TheGs();
    std::memcpy(g.memory.data(), data.data() + 5, kMemory);
    const bool gpu = std::getenv("PS2HOST_REPLAY_GPU") != nullptr;
    if (gpu) {
        std::string error;
        if (!StartGpuRenderer(&error, GpuMode::Primary, false)) {
            std::fprintf(stderr, "gs-replay: no GPU (%s)\n", error.c_str());
            return 69;
        }
        const char* scale = std::getenv("PS2HOST_REPLAY_SCALE");
        if (scale) GpuSetScale(u32(std::max(1, std::atoi(scale))));
        ApplyRendererSettings(Options{});
        if (g.on_memory) g.on_memory(0, 16384, true);       // local memory as the dump has it, on the GPU too
    }
    // PS2HOST_REPLAY_BENCH=n (with PS2HOST_REPLAY_GPU): the stream drawn n
    // times over, no trace or pictures kept - the GPU's time a picture and
    // the wall clock's, to weigh a change to the renderer or the bridge.
    const int bench = std::getenv("PS2HOST_REPLAY_BENCH") ? std::max(1, std::atoi(std::getenv("PS2HOST_REPLAY_BENCH"))) : 0;
    std::FILE* trace = bench ? nullptr : std::fopen((out + "/trace.txt").c_str(), "w");
    if (trace) {
        g.trace_out = trace;
        g.trace = true;
    }
    std::size_t at = 5 + kMemory;
    auto u32_at = [&](std::size_t o) { u32 v; std::memcpy(&v, data.data() + o, 4); return v; };
    auto u64_at = [&](std::size_t o) { u64 v; std::memcpy(&v, data.data() + o, 8); return v; };
    int pictures = 0;
    std::vector<u32> words;
    const std::size_t start = at;
    const u64 gpu_from = GpuBusyMicroseconds();
    const auto wall_from = std::chrono::steady_clock::now();
    for (int round = 0; round < std::max(1, bench); ++round)
    for (at = start; at < data.size();) {
        const char kind = char(data[at++]);
        if (kind == 'R') {
            g.WriteRegister(u32_at(at), u64_at(at + 4));
            at += 12;
        } else if (kind == 'P') {
            g.SetPrivileged(u32_at(at), u64_at(at + 4));
            at += 12;
        } else if (kind == 'W') {
            const int path = data[at];
            const u32 bytes = u32_at(at + 1);
            at += 5;
            words.resize(bytes / 4);
            std::memcpy(words.data(), data.data() + at, words.size() * 4);
            at += bytes;
            // PS2HOST_REPLAY_STOP_PRIM=n: the replay ends as primitive n is drawn
            // (memory then, to compare two renderers mid-frame).
            static const long long stop = std::getenv("PS2HOST_REPLAY_STOP_PRIM")
                                              ? std::atoll(std::getenv("PS2HOST_REPLAY_STOP_PRIM")) : -1;
            if (stop < 0) {
                g.Words(path, words.data(), u32(words.size()));
            } else {
                bool stopped = false;
                for (std::size_t w = 0; w < words.size() && !stopped; w += 4) {
                    g.Words(path, words.data() + w, u32(std::min<std::size_t>(4, words.size() - w)));
                    stopped = (long long)g.primitives() >= stop;
                }
                if (stopped) {
                    at = data.size();
                    break;
                }
            }
        } else if (kind == 'V') {
            g.Finish();
            if (trace) std::fprintf(trace, "[gs] ---- vertical blank %d\n", pictures);
            if (gpu && bench) {
                GpuShadowFrame(u64(pictures++), std::string());
            } else if (gpu) {
                GpuShadowFrame(u64(pictures), std::string());
                const std::string dir = out + "/gpu_" + std::to_string(pictures++);
                std::filesystem::create_directories(dir);
                GpuCaptureScreen(dir);
            } else {
                g.WriteBmp(out + "/replay_" + std::to_string(pictures++) + ".bmp");
            }
        } else {
            std::fprintf(stderr, "gs-replay: bad record at %zu\n", at - 1);
            return 65;
        }
    }
    g.Finish();
    if (gpu && g.on_memory) g.on_memory(0, 16384, false);     // what the GPU drew, into local memory first
    if (bench && pictures) {
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_from).count();
        std::fprintf(stderr, "gs-replay bench: %d pictures, GPU %.2f ms a picture, wall %.2f ms a picture\n", pictures,
                     double(GpuBusyMicroseconds() - gpu_from) / 1000.0 / pictures, wall * 1000.0 / pictures);
    }
    if (std::FILE* m = std::fopen((out + "/gsmem.bin").c_str(), "wb")) {      // local memory at the end
        std::fwrite(g.memory.data(), 1, kMemory, m);
        std::fclose(m);
    }
    if (trace) {
        g.trace = false;
        g.trace_out = stderr;
        std::fclose(trace);
    }
    std::fprintf(stderr, "gs-replay: %d pictures to %s\n", pictures, out.c_str());
    return 0;
}

namespace {

const FunctionEntry* g_table = nullptr;
std::size_t g_count = 0;
u32 g_image_start = 0, g_image_end = 0;       // the executable's bytes, as loaded
std::unordered_map<u32, FunctionEntry> g_overrides;
bool g_trace = false;
bool g_all_notes = false;
std::vector<u32> g_watch;
std::set<std::string> g_noted;

// Calls already made, by address: the function and its call count, so a call
// costs a lookup in a small table rather than a map's (a frame makes hundreds
// of thousands). Cleared when an override replaces a function.
struct CachedCall {
    u32 address = 0;
    const FunctionEntry* entry = nullptr;
    u64* count = nullptr;
};
constexpr std::size_t kCallCache = 1u << 14;
CachedCall g_call_cache[kCallCache];

const FunctionEntry* Find(u32 address) {
    auto found = g_overrides.find(address);
    if (found != g_overrides.end()) return &found->second;
    const FunctionEntry* end = g_table + g_count;
    const FunctionEntry* at = std::lower_bound(
        g_table, end, address, [](const FunctionEntry& e, u32 a) { return e.address < a; });
    return (at != end && at->address == address) ? at : nullptr;
}

std::string Hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%08X", value);
    return text;
}

std::string Describe(u32 address) {
    const char* name = FunctionName(address);
    return name ? Hex(address) + " " + name : Hex(address);
}

// The running thread's guest call stack, kept as the host unwinds through
// translated code.
struct Frame {
    explicit Frame(u32 address) : calls(CurrentCalls()) { calls.push_back(address); }
    ~Frame() { calls.pop_back(); }
    std::vector<u32>& calls;
};

}  // namespace

namespace {

// Code the program copied elsewhere and runs there - start-up code installs
// routines in kernel RAM as syscall handlers. A call to an address with no
// translation is matched against the program's functions by content: a copy of
// one, byte for byte, runs that function's translation. The copies are
// position-independent (their branches are relative and their calls absolute, so
// they reach the same targets), so the translation behaves the same there.
const FunctionEntry* FindCopy(u32 address) {
    static std::unordered_map<u32, const FunctionEntry*> copies;
    auto known = copies.find(address);
    if (known != copies.end()) return known->second;
    Memory& memory = GuestMemory();
    const FunctionEntry* found = nullptr;
    const u8* here = memory.Translate(address, 8);
    for (std::size_t i = 0; here && i < g_count && !found; ++i) {
        const FunctionEntry& e = g_table[i];
        if (!e.function || e.size < 8) continue;
        const u8* there = memory.Translate(address, e.size);
        const u8* original = memory.Translate(e.address, e.size);
        if (there && original && std::memcmp(there, original, e.size) == 0) found = &e;
    }
    copies[address] = found;
    if (found) {
        Note("copy " + Hex(address), "code at " + Hex(address) + " is a copy of " +
             Describe(found->address) + "; running its translation");
    }
    return found;
}

// Where in the loaded executable the code at `address` came from, or 0: the
// executable is searched for the bytes there. Code kept as data and copied out at
// run time (a kernel extension, an overlay) has no translation yet; finding its
// source lets engine/tools/port.py translate it for the next build.
u32 FindSource(u32 address) {
    // Code in the executable itself that no function starts at - a tail call's
    // target the analysis missed - is its own source.
    if (address >= g_image_start && address < g_image_end) return address;
    constexpr u32 kPattern = 32;
    Memory& memory = GuestMemory();
    const u8* here = memory.Translate(address, kPattern);
    const u8* image = memory.Translate(g_image_start, g_image_end - g_image_start);
    if (!here || !image || g_image_end - g_image_start < kPattern) return 0;
    const u8* end = image + (g_image_end - g_image_start) - kPattern;
    for (const u8* at = image; at <= end; at += 4) {
        const u32 source = g_image_start + u32(at - image);
        if (source != address && std::memcmp(at, here, kPattern) == 0) return source;
    }
    return 0;
}

}  // namespace

void RegisterFunctions(const FunctionEntry* table, std::size_t count) {
    g_table = table;
    g_count = count;
    // Before the game's own overrides, which may replace these in turn.
    ApplySdkOverrides(table, count);
}

void Override(u32 address, GuestFunction function, const char* name) {
    g_overrides[address] = FunctionEntry{address, function, name, 0};
    for (CachedCall& line : g_call_cache) line = CachedCall{};
}

const char* FunctionName(u32 address) {
    const FunctionEntry* entry = Find(address);
    return entry ? entry->name : nullptr;
}

bool Tracing() { return g_trace; }

Stats& RunStats() {
    static Stats stats;
    return stats;
}

std::mutex g_notes_mutex;               // VIF1's worker notes too

bool Noting(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_notes_mutex);
    return g_all_notes || !g_noted.count(key);
}

void Note(const std::string& key, const std::string& text) {
    std::lock_guard<std::mutex> lock(g_notes_mutex);
    if (g_noted.insert(key).second) {
        RunStats().notes.push_back(text);
        std::fprintf(stderr, "[host] %s\n", text.c_str());
    } else if (g_all_notes) {
        std::fprintf(stderr, "[host] %s (again)\n", text.c_str());
    }
}

void Dispatch(lifted::Context& c, u32 address) {
    CachedCall& line = g_call_cache[(address >> 2) & (kCallCache - 1)];
    if (line.address == address && line.entry && g_watch.empty()) {
        ++*line.count;
        if (g_trace) std::fprintf(stderr, "[call] %s\n", Describe(address).c_str());
        Frame frame(address);
        try {
            line.entry->function(c);
        } catch (...) {
            if (RunStats().stopped_in.empty()) {
                RunStats().stopped_in = CurrentCalls();
                RunStats().interrupted_in = InterruptedCalls();
            }
            throw;
        }
        return;
    }
    const FunctionEntry* entry = Find(address);
    const bool cacheable = entry && entry->function;     // the program's own, or an override
    if (!entry || !entry->function) {
        if (const FunctionEntry* copy = FindCopy(address)) entry = copy;
    }
    if (!entry || !entry->function) {
        if (const u32 source = FindSource(address)) {
            RunStats().discovered.emplace_back(address, source);
            throw Unimplemented{"a call to " + Hex(address) + ", code " +
                                (source == address ? std::string("no function covers")
                                                   : "copied from " + Hex(source) + " at run time") +
                                ", not translated yet (port.py translates it for the next build)"};
        }
        throw Unimplemented{"a call to " + Hex(address) + ", which has no translated function"};
    }
    u64& count = RunStats().calls[address];
    if (++count == 1 || g_trace)
        std::fprintf(stderr, "[call] %s\n", Describe(address).c_str());
    if (cacheable) line = CachedCall{address, entry, &count};
    if (std::find(g_watch.begin(), g_watch.end(), address) != g_watch.end()) {
        std::fprintf(stderr, "[watch] %s a0=%016llX a1=%016llX a2=%016llX a3=%016llX "
                     "f12=%08X f13=%08X ra=%08X s0=%08X s1=%08X sp=%08X gp=%08X t0=%08X\n",
                     Describe(address).c_str(),
                     (unsigned long long)c.r[4], (unsigned long long)c.r[5],
                     (unsigned long long)c.r[6], (unsigned long long)c.r[7], c.f[12], c.f[13],
                     unsigned(c.r[31]), unsigned(c.r[16]), unsigned(c.r[17]), unsigned(c.r[29]),
                     unsigned(c.r[28]), unsigned(c.r[8]));
    }
    Frame frame(address);
    try {
        entry->function(c);
    } catch (...) {
        // The innermost frame keeps the stack as it was where the run stopped;
        // unwinding empties it on the way out.
        if (RunStats().stopped_in.empty()) {
            RunStats().stopped_in = CurrentCalls();
            RunStats().interrupted_in = InterruptedCalls();
        }
        throw;
    }
    if (std::find(g_watch.begin(), g_watch.end(), address) != g_watch.end()) {
        // PS2HOST_BLANK_LOG: when, on the clock its blanks are logged on.
        static const bool timed = std::getenv("PS2HOST_BLANK_LOG") != nullptr;
        char at[40] = "";
        if (timed)
            std::snprintf(at, sizeof at, " at %.3f",
                          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count());
        std::fprintf(stderr, "[watch] %s returned v0=%016llX f0=%08X s0=%08X s1=%08X sp=%08X%s\n",
                     Describe(address).c_str(), (unsigned long long)c.r[2], c.f[0],
                     unsigned(c.r[16]), unsigned(c.r[17]), unsigned(c.r[29]), at);
    }
}

namespace {

// ELF32 little-endian, the PS2's executable format: load every PT_LOAD segment.
u32 LoadElf(const std::string& path, std::vector<u8> data) {
    auto u16at = [&](std::size_t at) { u16 v; std::memcpy(&v, &data.at(at + 1) - 1, 2); return v; };
    auto u32at = [&](std::size_t at) { u32 v; std::memcpy(&v, &data.at(at + 3) - 3, 4); return v; };
    if (data.size() < 52 || std::memcmp(data.data(), "\x7F" "ELF", 4) != 0)
        throw Unimplemented{path + " is not an ELF file"};
    const u32 entry = u32at(24), phoff = u32at(28);
    const u16 phentsize = u16at(42), phnum = u16at(44);
    Memory& memory = GuestMemory();
    for (u16 i = 0; i < phnum; ++i) {
        const std::size_t ph = phoff + std::size_t(i) * phentsize;
        if (u32at(ph) != 1) continue;                       // PT_LOAD
        const u32 offset = u32at(ph + 4), vaddr = u32at(ph + 8);
        const u32 filesz = u32at(ph + 16), memsz = u32at(ph + 20);
        if (!memsz) continue;
        u8* target = memory.Translate(vaddr, memsz);
        if (!target || std::size_t(offset) + filesz > data.size())
            throw Unimplemented{"a segment of " + path + " lies outside guest memory"};
        std::memcpy(target, data.data() + offset, filesz);
        std::memset(target + filesz, 0, memsz - filesz);
        if (filesz && !g_image_start) {
            g_image_start = vaddr;
            g_image_end = vaddr + filesz;
        }
    }
    // The sections holding instructions (SHF_ALLOC | SHF_EXECINSTR), as loaded:
    // a game that patches them is found out (code.cpp).
    const u32 shoff = u32at(32);
    const u16 shentsize = u16at(46), shnum = u16at(48);
    for (u16 i = 0; shoff && i < shnum; ++i) {
        const std::size_t sh = shoff + std::size_t(i) * shentsize;
        const u32 type = u32at(sh + 4), flags = u32at(sh + 8), addr = u32at(sh + 12);
        const u32 offset = u32at(sh + 16), size = u32at(sh + 20);
        if (type != 1 || (flags & 6u) != 6u || !size || std::size_t(offset) + size > data.size()) continue;
        if (memory.Translate(addr, size)) KeepCode(addr, data.data() + offset, size);
    }
    return entry;
}

u32 LoadElf(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw Unimplemented{"cannot open the executable " + path};
    return LoadElf(path, std::vector<u8>((std::istreambuf_iterator<char>(file)),
                                         std::istreambuf_iterator<char>()));
}

// No executable named: the one the disc boots (SYSTEM.CNF), read from it.
u32 LoadBootElf() {
    const std::string boot = GuestDisc().BootPath();
    std::vector<u8> data;
    if (boot.empty() || !GuestDisc().ReadFile(boot, &data))
        throw Unimplemented{"the disc image has no executable to boot (SYSTEM.CNF)"};
    return LoadElf(boot, std::move(data));
}

void Report(const Options& options, const std::string& outcome) {
    const Stats& stats = RunStats();
    std::printf("\n=== host report ===\n%s\n", outcome.c_str());
    std::printf("functions executed: %zu distinct\n", stats.calls.size());
    const std::vector<u32>& stack = stats.stopped_in;
    if (!stack.empty()) {
        std::printf("guest call stack where it stopped (innermost last):\n");
        const std::size_t from = stack.size() > std::size_t(options.report_limit)
                                     ? stack.size() - options.report_limit : 0;
        for (std::size_t i = from; i < stack.size(); ++i)
            std::printf("  %s\n", Describe(stack[i]).c_str());
    }
    if (!stats.interrupted_in.empty()) {
        std::printf("in an interrupt handler; the thread it interrupted was in:\n");
        const std::vector<u32>& thread = stats.interrupted_in;
        const std::size_t from = thread.size() > std::size_t(options.report_limit)
                                     ? thread.size() - options.report_limit : 0;
        for (std::size_t i = from; i < thread.size(); ++i)
            std::printf("  %s\n", Describe(thread[i]).c_str());
    }
    std::printf("threads:\n%s%s", ThreadReport().c_str(), iop::Report().c_str());
    if (!options.dump_ram.empty()) {
        if (std::FILE* f = std::fopen(options.dump_ram.c_str(), "wb")) {
            std::fwrite(GuestMemory().rdram(), 1, 32u << 20, f);
            std::fclose(f);
            std::printf("EE RAM written to %s\n", options.dump_ram.c_str());
        }
    }
    for (u32 address : options.peek) {
        std::printf("memory at 0x%08X:", unsigned(address));
        for (u32 i = 0; i < 64; i += 4) {
            if (i % 16 == 0) std::printf("\n  +%02X", unsigned(i));
            const u8* bytes = GuestMemory().Translate(address + i, 4);
            u32 word = 0;
            if (bytes) std::memcpy(&word, bytes, 4);
            std::printf(bytes ? "  %08X" : "  --------", unsigned(word));
        }
        std::printf("\n");
    }
    const std::string code = CodeReport();
    if (!code.empty()) std::printf("code changed by the program (not followed by the translation):\n%s", code.c_str());
    std::printf("VIF1 commands:%s\n", Vif1().Summary().c_str());
    // Whether the game goes through VIF1's interrupt-bit stalls at all.
    std::printf("VIF1: %llu interrupt stalls\n", static_cast<unsigned long long>(Vif1InterruptStalls()));
    std::printf("GS: %llu primitives, %llu pixels written\n",
                static_cast<unsigned long long>(gs::TheGs().primitives()),
                static_cast<unsigned long long>(gs::TheGs().pixels_drawn()));
    std::printf("%s\n", gs::TheGs().Summary().c_str());
    std::printf("%s\n", ProfileSummary().c_str());
    const std::string samples = SampleReport(std::getenv("PS2HOST_SAMPLE_TOP") ? std::atoi(std::getenv("PS2HOST_SAMPLE_TOP")) : 30);
    if (!samples.empty()) std::printf("%s", samples.c_str());
    if (!GpuShadowSummary().empty()) std::printf("%s\n", GpuShadowSummary().c_str());
    std::printf("kernel services used:\n");
    for (const auto& [service, count] : stats.syscalls)
        std::printf("  %4d  %-24s x%llu\n", service, SyscallName(service),
                    (unsigned long long)count);
    auto registers = [&](const char* title, const std::map<u32, u64>& seen) {
        std::printf("%s: %zu\n", title, seen.size());
        int shown = 0;
        for (const auto& [address, count] : seen) {
            if (shown++ == options.report_limit) break;
            std::printf("  0x%08X x%llu\n", address, (unsigned long long)count);
        }
    };
    {
        // Where the time goes: the functions called most - a loop that never
        // ends shows here as the functions it keeps calling.
        std::vector<std::pair<u64, u32>> hot;
        for (const auto& [address, count] : stats.calls) hot.push_back({count, address});
        std::sort(hot.rbegin(), hot.rend());
        std::printf("most called:\n");
        for (std::size_t i = 0; i < hot.size() && i < std::size_t(options.report_limit); ++i)
            std::printf("  %-40s x%llu\n", Describe(hot[i].second).c_str(), (unsigned long long)hot[i].first);
    }
    registers("device registers read", stats.register_reads);
    registers("device registers written", stats.register_writes);
    for (const auto& [run, source] : stats.discovered)
        std::printf("discovered: code at %s is a copy of %s\n", Hex(run).c_str(), Hex(source).c_str());
    if (!options.discover_path.empty()) {
        // Merged with what earlier runs found: each run stops at the first one.
        std::FILE* out = std::fopen(options.discover_path.c_str(), "a");
        if (out) {
            for (const auto& [run, source] : stats.discovered)
                std::fprintf(out, "0x%08X 0x%08X\n", run, source);
            std::fclose(out);
        }
    }
}

}  // namespace

namespace {

// Stops a run that goes on too long, so it ends with a report rather than being
// killed without one: the running context's loop budget is exhausted, and the
// translated code stops at its next loop head (lifted::Tick) with the guest call
// stack intact.
class Watchdog {
public:
    explicit Watchdog(int seconds) : thread_([this, seconds] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!done_.wait_for(lock, std::chrono::seconds(seconds), [this] { return finished_; })) {
            fired = true;
            StopRequested() = true;
            // Spent again every 10 ms, not once: Tick's `--c.budget` is a plain
            // read-modify-write (`sub qword ptr [budget],1`), and a loop with
            // nothing else in it - Stuntman's wait on COP0 Count at 0x001A3208,
            // two instructions once compiled - is inside one nearly all the
            // time. A single -1 stored between its read and its write is
            // overwritten with the old budget less one, and the loop never sees
            // it. The budget stays a plain s64: an atomic one would cost every
            // Tick a locked instruction.
            // A run that does not stop is stuck in host code (no guest loop
            // head comes): say where the guest was, rather than be killed
            // without a word.
            const bool stopped = SpendUntilStopped(
                [] {
                    ForEachContext([](lifted::Context& each) {
                        reinterpret_cast<std::atomic<s64>*>(&each.budget)->store(-1);
                    });
                },
                [&](std::chrono::milliseconds every) {
                    return done_.wait_for(lock, every, [this] { return finished_; });
                },
                std::chrono::milliseconds(10), std::chrono::seconds(20));
            if (!stopped) {
                u32 calls[32];
                const int n = GuestCallsNow(calls, 32);
                std::fprintf(stderr, "\n=== host report ===\nstuck: the run did not stop 20 s after its time "
                                     "limit - the host is in a loop of its own\nguest call stack (innermost first):");
                for (int i = 0; i < n; ++i) std::fprintf(stderr, " 0x%08X", unsigned(calls[i]));
                std::fprintf(stderr, "\n");
                std::fflush(nullptr);
                // PS2HOST_STUCK_RAM=file: EE RAM as it stands, to look into.
                if (const char* path = std::getenv("PS2HOST_STUCK_RAM")) {
                    if (std::FILE* f = std::fopen(path, "wb")) {
                        std::fwrite(GuestMemory().Translate(0, 32u << 20), 1, 32u << 20, f);
                        std::fclose(f);
                        std::fprintf(stderr, "EE RAM written to %s\n", path);
                    }
                }
                // The IOP's threads too (read while it runs: a diagnostic, last),
                // and where each of the host's own is.
                std::fprintf(stderr, "%s", iop::Report().c_str());
                std::fflush(nullptr);
                std::fprintf(stderr, "%s", HostThreadStacks().c_str());
                std::fflush(nullptr);
                std::_Exit(3);
            }
        }
    }) {}
    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        done_.notify_all();
        thread_.join();
    }
    std::atomic<bool> fired{false};

private:
    std::mutex mutex_;
    std::condition_variable done_;
    bool finished_ = false;
    std::thread thread_;
};

}  // namespace

int Run(const Options& options) {
    SetVirtualClock(options.virtual_time);  // before anything reads the clock
    g_trace = options.trace_calls;
    g_all_notes = options.all_notes;
    GuestMemory().WatchWrites(options.watch_writes);
    GuestMemory().WatchReads(options.watch_reads);
    InstallCrashReport();
    SetVif1Worker(!options.vif1_sync);
    SetPalTiming((!options.pal_timing.empty() ? options.pal_timing : Setting("pal_timing", "on")) == "on");
    SetLateBlanksTogether((!options.late_blanks.empty() ? options.late_blanks : Setting("late_blanks", "apart")) == "together");
    SetIopWhileWaiting((!options.iop_while_waiting.empty() ? options.iop_while_waiting
                                                           : Setting("iop_while_waiting", "on")) == "on");
    SetVif1FramesBehind(std::stoi(!options.vif1_frames_behind.empty() ? options.vif1_frames_behind
                                                                      : Setting("vif1_frames_behind", "1")));
    SetVif1StcNoWait((!options.vif1_stc_no_wait.empty() ? options.vif1_stc_no_wait
                                                          : Setting("vif1_stc_no_wait", "on")) == "on");
    // A check: the worker in virtual time too, the run otherwise the same.
    if (options.vif1_worker_virtual && !options.vif1_sync) StartVif1Worker();
    if (std::getenv("PS2HOST_GS_DIGEST")) gs::TheGs().digesting = true;
    SetCheckGuesses(options.check_guesses);
    if (!options.vu_snapshot.empty())
        Vu::Snapshots(options.vu_snapshot, u64(options.vu_snapshot_from), options.vu_snapshot_count);
    // A scripted run (--virtual-time) draws the same pictures every time: not
    // with the settings a player keeps beside the executable - the defaults,
    // unless it brings a memory card (and so settings) of its own, or goes on
    // in real time (--real-time-from: measured as played).
    if (!options.virtual_time || !options.memcard.empty() || options.real_time_from) {
        // The PC settings: beside the memory card (its directory's parent),
        // or the executable, so they go where the player's saves go.
        std::filesystem::path where;
        if (!options.memcard.empty()) {
            where = std::filesystem::absolute(options.memcard).parent_path();
        } else {
            wchar_t exe[MAX_PATH] = {};
            GetModuleFileNameW(nullptr, exe, MAX_PATH);
            where = std::filesystem::path(exe).parent_path();
        }
        LoadSettings((where / "pc_settings.ini").string());
    }
    if (options.window) OpenWindow(L"PS2 port", 1280, 896);      // or the game's title (SetGameTitle)
    iop::SetTrace(options.iop_trace);
    gs::TheGs().trace = options.gs_trace;
    SetGsTraceFrame(options.gs_trace_frame);
    SetGsDumpFrame(options.gs_dump_frame);
    if (options.sample) {
        SetSampleFrom(u64(options.sample_from));
        if (options.sample_gs) SetSampleGsThread(true);
        else if (options.sample_vif1) SetSampleVif1Thread(true);
        else StartSampler();
    }
    if (!options.vu_record.empty()) Vu::RecordPrograms(options.vu_record);
    {
        // Played (a window, real time), the setting, weave by default; a
        // scripted run's pictures stay each field alone unless asked.
        const bool played = options.window && !options.virtual_time;
        const std::string mode = !options.deinterlace.empty() ? options.deinterlace
                                 : played                      ? Setting("deinterlace", "weave")
                                                               : "off";
        gs::TheGs().weave = mode == "weave" || mode == "auto";
        gs::TheGs().weave_auto = mode == "auto";
    }
    if (!options.memcard.empty()) mc2::SetCardDirectory(options.memcard);
    if (options.vu_interpret) Vu1().use_compiled = Vu0().use_compiled = false;
    // VU1 without the flags no instruction reads (PORTING.md's "Fast mode"):
    // --vu-fast, or the "vu_fast" setting (off unless set).
    Vu1().fast = options.vu_fast || SettingOn("vu_fast", false);
    // The EE FPU's model (eefpu.cpp): float_model=ps2 the hardware's datapaths
    // bit for bit, else each result exact and rounded toward zero (the default;
    // PS2RT_EE_FLOAT=ps2 sets it too, for a scripted run).
    if (Setting("float_model", "") == "ps2") generated::SetFloatModel(true);
    SetUntilFrame(options.until_frame);
    SetFrameTimes(options.frame_times);
    gs::TheGs().whole_qwords =
        (!options.gs_whole_qwords.empty() ? options.gs_whole_qwords : Setting("gs_whole_qwords", "on")) != "off";
    // A measure of all but the drawing (the EE's ceiling): the GS takes the
    // primitives in and draws none, nothing drawing in its stead.
    if (options.gs_no_draw) gs::TheGs().rasterize = false;
    if (options.gpu || options.gpu_shadow) {
        std::string error;
        const GpuMode mode = options.gpu ? GpuMode::Primary : options.gpu_shadow_rtt ? GpuMode::ShadowRtt : GpuMode::Shadow;
        SetGpuRing(std::atoi(Setting("gpu_ring", std::getenv("PS2HOST_GPU_RING") ? std::getenv("PS2HOST_GPU_RING") : "3").c_str()));
        if (!StartGpuRenderer(&error, mode, options.gpu_show))
            std::fprintf(stderr, "%s: %s\n", options.gpu ? "--gpu" : "--gpu-shadow", error.c_str());
        ApplyRendererSettings(options);
        SetVulkanThread((!options.vulkan_thread.empty() ? options.vulkan_thread : Setting("vulkan_thread", "on")) == "on");
        SetRenderQueue(!options.render_queue.empty() ? options.render_queue : Setting("render_queue", "batch"));
        SetRenderQueueFrames(u32(std::stoul(!options.render_queue_frames.empty() ? options.render_queue_frames
                                                                              : Setting("render_queue_frames", "1"))));
        SetPresentOnRenderThread((!options.present_on_render_thread.empty()
                                      ? options.present_on_render_thread
                                      : Setting("present_on_render_thread", "on")) == "on");
        SetRenderThread(!options.render_thread.empty() ? options.render_thread : Setting("render_thread", "on"));
        if (options.gs_no_draw) gs::TheGs().on_draw = nullptr;     // (--gs-no-draw: nor the GPU)
    }
    bool sound = false;
    if (options.audio) {
        sound = OpenAudio();
        SetAudioSteer((!options.audio_steer.empty() ? options.audio_steer : Setting("audio_steer", "on")) == "on");
        if (!sound) std::fprintf(stderr, "no sound device: running silent\n");
    }
    if (!options.audio_wav.empty()) {
        if (OpenWavDump(options.audio_wav)) sound = true;
        else std::fprintf(stderr, "cannot write %s\n", options.audio_wav.c_str());
    }
    if (sound) iop::EnableSound();
    for (const auto& press : options.presses)
        if (!ScriptPress(press.button, press.from, press.to))
            std::fprintf(stderr, "--press: no button %s\n", press.button.c_str());
    if (!options.frames_dir.empty())
        SetFrameDumps(options.frames_dir, options.frames_every, 200, u64(options.frames_from));
    g_watch = options.watch;
    InstallDevices();
    InstallIopServers();
    if (!options.gs_inline) StartGsThread();
    std::fprintf(stderr, "%s\n", SettingsReport().c_str());
    lifted::Context c = NewContext();
    timeline::NameThread("EE");
    Watchdog watchdog(options.timeout_seconds);
    std::string outcome;
    int code = 2;
    try {
        if (!options.disc_path.empty() && !GuestDisc().Open(options.disc_path))
            throw Unimplemented{"cannot open the disc image " + options.disc_path};
        const u32 entry = options.elf_path.empty() ? LoadBootElf() : LoadElf(options.elf_path);
        // The kernel enters a program with a stack at the top of RAM; the program's
        // own start-up code then asks for its real one (SetupThread).
        c.r[29] = kRdramSize - 0x10;
        c.r[31] = 0;
        std::fprintf(stderr, "[host] entry %s\n", Describe(entry).c_str());
        RunThreads(c, entry, options.loop_budget);
    } catch (const GuestExit& exit) {
        outcome = "the guest exited with code " + std::to_string(exit.code);
        code = 0;
    } catch (const Unimplemented& missing) {
        outcome = "stopped: " + missing.what;
    } catch (const MemoryFault& fault) {
        outcome = std::string("stopped: a ") + (fault.write ? "write" : "read") +
                  " at " + Hex(fault.address) + ", outside guest memory";
    } catch (const lifted::Abort& abort) {
        static const char* reasons[] = {"", "a trap", "a loop that did not finish within the budget",
                                        "a callee that failed", "a memory fault", "a COP2 operation"};
        const char* reason = abort.reason < 6 ? reasons[abort.reason] : "an abort";
        if (abort.reason == 2 && watchdog.fired)
            reason = "the time limit, in the loop";
        outcome = std::string("stopped: ") + reason + " at " + Hex(abort.address);
        if (abort.reason == 2 && watchdog.fired && abort.address == 0)
            outcome = "stopped: the time limit, between threads (no thread was in a loop)";
        if (abort.reason == 2 && UntilFrameReached())
            outcome = "stopped: frame " + std::to_string(options.until_frame) + " reached (--until-frame)";
    }
    // The GS takes what it was sent before the report reads it.
    try {
        StopGsThread();
    } catch (const Unimplemented& missing) {
        if (outcome.empty()) outcome = "stopped: " + missing.what;
    }
    timeline::Finish();
    Report(options, outcome);
    return code;
}

int Main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace") options.trace_calls = true;
        else if (arg == "--notes") options.all_notes = true;
        else if (arg == "--iop-trace") options.iop_trace = true;
        else if (arg == "--window") options.window = true;
        else if (arg == "--gs-trace") options.gs_trace = true;
        else if (arg == "--gpu") options.gpu = true;
        else if (arg == "--gpu-shadow") options.gpu_shadow = true;
        else if (arg == "--gpu-shadow-rtt") options.gpu_shadow = options.gpu_shadow_rtt = true;
        else if (arg == "--gpu-show") options.gpu_shadow = options.gpu_shadow_rtt = options.gpu_show = true;
        else if (arg == "--virtual-time") options.virtual_time = true;
        else if (arg == "--until-frame" && i + 1 < argc) options.until_frame = std::stoull(argv[++i]);
        else if (arg == "--frame-times" && i + 1 < argc) options.frame_times = unsigned(std::stoul(argv[++i]));
        else if (arg == "--gs-inline") options.gs_inline = true;
        else if (arg == "--gs-no-draw") options.gs_no_draw = true;
        else if (arg == "--gs-trace-frame" && i + 1 < argc) options.gs_trace_frame = std::stoll(argv[++i]);
        else if (arg == "--gs-dump-frame" && i + 1 < argc) options.gs_dump_frame = std::stoll(argv[++i]);
        else if (arg == "--sample") options.sample = true;
        else if (arg == "--sample-gs") options.sample = options.sample_gs = true;
        else if (arg == "--sample-vif1") options.sample = options.sample_vif1 = true;
        else if (arg == "--vif1-sync") options.vif1_sync = true;
        else if (arg == "--vif1-worker-virtual") options.vif1_worker_virtual = true;
        else if (arg == "--vif1-frames-behind" && i + 1 < argc) options.vif1_frames_behind = argv[++i];
        else if (arg == "--vif1-stc-no-wait" && i + 1 < argc) options.vif1_stc_no_wait = argv[++i];
        else if (arg == "--check-guesses") options.check_guesses = true;
        else if (arg == "--vu-snapshot" && i + 3 < argc) {
            options.vu_snapshot = argv[++i];
            options.vu_snapshot_from = std::stoll(argv[++i]);
            options.vu_snapshot_count = unsigned(std::stoul(argv[++i]));
        }
        else if (arg == "--vu-bench" && i + 1 < argc) options.vu_bench = argv[++i];
        else if (arg == "--vu-check" && i + 1 < argc) options.vu_check = argv[++i];
        else if (arg == "--gs-replay" && i + 2 < argc) {
            const std::string file = argv[++i], out = argv[++i];
            return GsReplay(file, out);
        }
        else if (arg == "--sample-from" && i + 1 < argc) {
            options.sample = true;
            options.sample_from = std::stoll(argv[++i]);
        }
        else if (arg == "--vu-record" && i + 1 < argc) options.vu_record = argv[++i];
        else if (arg == "--vu-interpret") options.vu_interpret = true;
        else if (arg == "--deinterlace" && i + 1 < argc) options.deinterlace = argv[++i];
        else if (arg == "--texture-cache" && i + 1 < argc) options.texture_cache = argv[++i];
        else if (arg == "--batch-overlap" && i + 1 < argc) options.batch_overlap = argv[++i];
        else if (arg == "--newer-list" && i + 1 < argc) options.newer_list = argv[++i];
        else if (arg == "--owned-index" && i + 1 < argc) options.owned_index = argv[++i];
        else if (arg == "--gs-whole-qwords" && i + 1 < argc) options.gs_whole_qwords = argv[++i];
        else if (arg == "--date-area" && i + 1 < argc) options.date_area = argv[++i];
        else if (arg == "--date-in-pass" && i + 1 < argc) options.date_in_pass = argv[++i];
        else if (arg == "--vulkan-thread" && i + 1 < argc) options.vulkan_thread = argv[++i];
        else if (arg == "--render-thread" && i + 1 < argc) options.render_thread = argv[++i];
        else if (arg == "--present-on-render-thread" && i + 1 < argc) options.present_on_render_thread = argv[++i];
        else if (arg == "--render-queue" && i + 1 < argc) options.render_queue = argv[++i];
        else if (arg == "--render-queue-frames" && i + 1 < argc) options.render_queue_frames = argv[++i];
        else if (arg == "--audio-steer" && i + 1 < argc) options.audio_steer = argv[++i];
        else if (arg == "--pal-timing" && i + 1 < argc) options.pal_timing = argv[++i];
        else if (arg == "--late-blanks" && i + 1 < argc) options.late_blanks = argv[++i];
        else if (arg == "--iop-while-waiting" && i + 1 < argc) options.iop_while_waiting = argv[++i];
        else if (arg == "--vu-fast") options.vu_fast = true;
        else if (arg == "--memcard" && i + 1 < argc) options.memcard = argv[++i];
        else if (arg == "--audio") options.audio = true;
        else if (arg == "--audio-wav" && i + 1 < argc) options.audio_wav = argv[++i];
        else if (arg == "--press" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const std::size_t at = spec.find('@');
            if (at != std::string::npos) {
                const std::string when = spec.substr(at + 1);
                const std::size_t dash = when.find('-');
                const u64 from = std::stoull(when.substr(0, dash));
                const u64 to = dash == std::string::npos ? from + 5 : std::stoull(when.substr(dash + 1));
                options.presses.push_back({spec.substr(0, at), from, to});
            }
        }
        else if (arg == "--real-time-from" && i + 1 < argc) {
            options.real_time_from = std::stoull(argv[++i]);
            SetRealTimeFrom(options.real_time_from);
        }
        else if (arg == "--fast-until" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const std::size_t colon = spec.find(':');
            SetFastUntil(std::stoull(spec.substr(0, colon)), colon == std::string::npos ? 8u : unsigned(std::stoul(spec.substr(colon + 1))));
        }
        else if (arg == "--mouse" && i + 1 < argc) {
            if (!ScriptMouse(argv[++i])) std::fprintf(stderr, "--mouse %s: FRAME:X,Y[,L|R|U|D]\n", argv[i]);
        }
        else if (arg == "--frames" && i + 1 < argc) options.frames_dir = argv[++i];
        else if (arg == "--frames-every" && i + 1 < argc) options.frames_every = std::stoi(argv[++i]);
        else if (arg == "--frames-from" && i + 1 < argc) options.frames_from = std::stoll(argv[++i]);
        else if (arg == "--budget" && i + 1 < argc) options.loop_budget = std::stoll(argv[++i]);
        else if (arg == "--limit" && i + 1 < argc) options.report_limit = std::stoi(argv[++i]);
        else if (arg == "--timeout" && i + 1 < argc) options.timeout_seconds = std::stoi(argv[++i]);
        else if (arg == "--discover" && i + 1 < argc) options.discover_path = argv[++i];
        else if (arg == "--disc" && i + 1 < argc) options.disc_path = argv[++i];
        else if (arg == "--watch" && i + 1 < argc)
            options.watch.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg == "--watch-write" && i + 1 < argc)
            options.watch_writes.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg == "--watch-read" && i + 1 < argc)
            options.watch_reads.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg == "--dump-ram" && i + 1 < argc) options.dump_ram = argv[++i];
        else if (arg == "--peek" && i + 1 < argc)
            options.peek.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg.rfind("--", 0) != 0) options.elf_path = arg;
        else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 64;
        }
    }
    if (!options.vu_check.empty()) return Vu::Check(options.vu_check);
    if (!options.vu_bench.empty()) {
        // With --sample, a hundred passes under the sampler and its report.
        if (options.sample) {
            StartSampler();
            SetSampleFrom(0);
            SampleFrame(0);
        }
        const int status = Vu::Bench(options.vu_bench, options.sample ? 100 : 5, options.vu_fast);
        if (options.sample) std::printf("%s", SampleReport().c_str());
        return status;
    }
    if (options.elf_path.empty() && options.disc_path.empty()) {
        std::fprintf(stderr, "usage: %s <executable from the game disc> | --disc <image> [--trace] [--notes] [--watch addr] [--peek addr] "
                             "[--budget N] [--limit N] [--timeout seconds]\n", argv[0]);
        return 64;
    }
    return Run(options);
}

}  // namespace ps2host

// The hooks lifted code calls through.
namespace lifted {

void Call(Context& c, u32 target) { ps2host::Dispatch(c, target); }

void TailCall(Context& c, u32 target, u32 /*return_address*/, u32 /*entry_sp*/) {
    // The guest jumps and never comes back here; natively, the callee runs and
    // the translated function returns straight after (it emits `return;`).
    ps2host::Dispatch(c, target);
}

void Syscall(Context& c) { ps2host::KernelSyscall(c); }

// vcallms: VU0's registers are the ones macro mode uses, so the microprogram
// runs on the calling thread's and leaves its results there.
void Vcallms(Context& c, u32 address) {
    ps2host::Vu& vu = ps2host::Vu0();
    std::memcpy(vu.regs.vf, c.vf, sizeof c.vf);
    std::memcpy(vu.regs.vc, c.vc, sizeof c.vc);
    std::memcpy(vu.regs.vacc, c.vacc, sizeof c.vacc);
    vu.Run(address);
    std::memcpy(c.vf, vu.regs.vf, sizeof c.vf);
    std::memcpy(c.vc, vu.regs.vc, sizeof c.vc);
    std::memcpy(c.vacc, vu.regs.vacc, sizeof c.vacc);
}

}  // namespace lifted

// Derived code's hooks: a guest stack for out-parameters, and pointer calls.
namespace generated {

namespace {
u32 g_stack = 0x01F00000u;      // below the guest stack the kernel hands out
}

u64 StackEnter(u32 size) { g_stack -= size; return g_stack; }
void StackLeave(u32 size) { g_stack += size; }

u64 CallPointer(u32 address, u64, u64, u64, u64, u64, u64, u64, u64,
                u64, u64, u64, u64, u64, u64, u64, u64) {
    throw ps2host::Unimplemented{"a derived pointer call to " + std::to_string(address)};
}

}  // namespace generated
