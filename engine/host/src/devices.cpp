/*
 * Device registers with hardware meaning, stage one: just enough for code that
 * starts a transfer and waits for it, or waits for a vertical blank, to go on.
 *
 * DMA: setting a channel's STR bit starts a transfer; here it completes at once
 *      - the bit clears and the channel's interrupt status is raised. VIF0, VIF1
 *      and GIF transfers are walked (dma.cpp), chains included: VIF1's data
 *      goes to its VIFcode decoder, the rest is counted. Nothing reaches a GS
 *      yet.
 * GIF: while PATH3 is masked, a PATH3 transfer's data waits in the GIF FIFO
 *      (16 quadwords deep), and GIF_STAT's FQC counts it - code handing PATH3
 *      over to VIF1 waits for the GIF to have taken its data before letting VIF1
 *      go. Unmasking drains it.
 * GS:  CSR reports a vertical blank on every read, its field bit alternating, so
 *      a wait for the next frame ends.
 * Timers: in host time, with their interrupts (timers.cpp).
 * VIF1: quadwords stored to its FIFO are decoded as VIFcodes (vif.cpp);
 *      MSKPATH3 shows in GIF_STAT's M3P bit, which code that hands PATH3 over
 *      waits on.
 *
 * Every register the game touches is counted for the report, so the next device
 * to implement is the one the game leans on.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <cstring>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "ps2host/dma.h"
#include "ps2host/timeline.h"
#ifdef PS2HOST_VULKAN
#include "ps2gpu/display.h"
#endif
#include "ps2host/gs.h"
#include "ps2host/iop_vm.h"
#include "ps2host/ipu.h"
#include "ps2host/memory.h"
#include "ps2host/path3.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/vif.h"
#include "ps2host/vu.h"
#include "ps2host/profile.h"

namespace ps2host {

namespace {

struct Channel {
    u32 base;
    const char* name;
    u32 status_bit;                 // its bit in D_STAT
    bool walked;                    // its transfers are followed (dma.cpp)
};

const Channel kChannels[] = {
    {0x10008000u, "VIF0", 0, true}, {0x10009000u, "VIF1", 1, true},
    {0x1000A000u, "GIF", 2, true}, {0x1000B000u, "IPU_FROM", 3, false},
    {0x1000B400u, "IPU_TO", 4, false}, {0x1000C000u, "SIF0", 5, false},
    {0x1000C400u, "SIF1", 6, false}, {0x1000C800u, "SIF2", 7, false},
    {0x1000D000u, "SPR_FROM", 8, true}, {0x1000D400u, "SPR_TO", 9, true},
};

// Register offsets within a channel.
constexpr u32 kMadr = 0x10, kQwc = 0x20, kTadr = 0x30, kAsr0 = 0x40, kAsr1 = 0x50;
constexpr u32 kGifFifoDepth = 16;

constexpr u32 kDStat = 0x1000E010u;

// D_STAT: writing 1 clears a status bit (0-15) and flips a mask bit (16-31), so
// what a store leaves is computed from what was there, kept here.
u32 d_stat = 0;

void SetDStat(u32 value) {
    d_stat = value;
    *reinterpret_cast<u32*>(GuestMemory().Translate(kDStat, 4)) = value;
}
// VIF1's interrupt bit (EE User's Manual 6.3.4): a VIFcode with bit 31 set
// stalls VIF1 once it is done, its data included - VIF1_STAT's VIS and INT
// set, INTC cause 5 raised - unless VIF1_ERR.MII masks it; the EE's handler
// does what the stream waited for (Stuntman: a texture's upload on PATH3) and
// writes FBRST.STC, and the stream goes on. The DMA transfer feeding it stops
// there too (dma.h, SendDmaBlocksUntil) and goes on from there. Nothing of it
// happens unless a VIFcode in the stream carries the bit.
constexpr u32 kVif1Stat = 0x10003C00u, kVif1Fbrst = 0x10003C10u, kVif1Err = 0x10003C20u;
constexpr u32 kVifStatVis = 1u << 10, kVifStatInt = 1u << 11;
constexpr u32 kIntcVif1 = 5;
VifInterruptScanner g_vif1_scan;            // follows every word VIF1 takes
bool g_vif1_stalled = false;
std::vector<u32> g_vif1_leftover;           // read past the stall, not taken yet
std::atomic<u64> g_vif1_stalls{0};
bool g_vif1_end_pending = false;            // the stalled chain ends with its owed data

constexpr u32 kGsCsr = 0x12001000u;
constexpr u32 kGifStat = 0x10003020u;
constexpr u32 kGifStatM3P = 1u << 1;
constexpr u32 kGifStatFqcShift = 24, kGifStatFqc = 0x1Fu << kGifStatFqcShift;

// PATH3: GIF DMA data to the GS, held while VIF1 has PATH3 masked - it goes
// after the PATH2 data VIF1 sends meanwhile, as the GIF arbitrates.
//
// A mask takes effect only at the end of the GIF packet PATH3 is in - the one
// whose tag has EOP set - so unmasking and masking again at once lets exactly
// one EOP-ended group through. Burnout 3 queues a picture's texture uploads
// (161,280 words at the car select) on PATH3 while it is masked and opens a
// window like that before each group of draws: five groups, five windows,
// each texture slot refilled between the draws that use its contents.
// Releasing everything at the first window refilled every slot early, and the
// garage was drawn with whatever was uploaded last into its slots (the font).
Path3Queue g_path3_held;
bool g_path3_masked = false;

void Path3(const u32* words, u32 count, u32 from) {
    // Behind what is still held, even unmasked: the GIF keeps PATH3 in order.
    if (g_path3_masked || !g_path3_held.empty()) g_path3_held.Push(words, count);
    else GsWords(3, words, count, from);
}

void ReleasePath3(bool all) {
    g_path3_held.ReleaseBlock(all, [](const u32* words, std::size_t count) {
        GsWords(3, words, u32(count));
    });
}

u32* Register(u32 address) { return reinterpret_cast<u32*>(GuestMemory().Translate(address, 4)); }

// The scratchpad channels: SADR (base + 0x80) is the scratchpad address.
// SPR_FROM copies QWC quadwords from it to MADR; SPR_TO copies from memory
// (normal or a source chain) into it.
constexpr u32 kSadr = 0x80;

// Main RAM or the scratchpad, with nothing watching stores: no hooks to run,
// so a copy may go through pointers.
bool Plain(const Memory& m, u32 address) {
    return !m.watching() && ((address & 0x1FFFFFFFu) < 0x10000000u || (address >> 28) == 0x7);
}

std::string Scratchpad(const Channel& ch, u32 chcr, bool describe) {
    Memory& m = GuestMemory();
    u32 sadr = u32(m.Read(ch.base + kSadr, 4)) & 0x3FF0u;
    const u32 madr = u32(m.Read(ch.base + kMadr, 4)), qwc = u32(m.Read(ch.base + kQwc, 4));
    char text[160];
    if (ch.base == 0x1000D000u) {                       // SPR_FROM
        if ((chcr >> 2) & 3) return "chain mode is not modelled (nothing copied)";
        for (u32 i = 0; i < 16 * qwc; i += 16) {           // a quadword at a time (SADR is aligned)
            const u32 from = kScratchpadBase + ((sadr + i) & 0x3FF0u);
            u8* to = Plain(m, madr + i) ? m.Translate(madr + i, 16) : nullptr;
            if (const u8* source = to ? m.Translate(from, 16) : nullptr) {
                std::memcpy(to, source, 16);
                continue;
            }
            for (u32 k = 0; k < 16; k += 4) m.Write(madr + i + k, m.Read(from + k, 4), 4);
        }
        *Register(ch.base + kMadr) = madr + 16 * qwc;
        *Register(ch.base + kQwc) = 0;
        *Register(ch.base + kSadr) = (sadr + 16 * qwc) & 0x3FF0u;
        if (!describe) return {};
        std::snprintf(text, sizeof text, "%u quadwords from scratchpad 0x%04X", unsigned(qwc), unsigned(sadr));
        return text;
    }
    DmaChannelState state{chcr, madr, qwc, u32(m.Read(ch.base + kTadr, 4)),
                          u32(m.Read(ch.base + kAsr0, 4)), u32(m.Read(ch.base + kAsr1, 4))};
    u8* scratchpad = Plain(m, kScratchpadBase) ? m.Translate(kScratchpadBase, kScratchpadSize) : nullptr;
    const DmaOutcome outcome = SendDmaBlocks(state, [&sadr, scratchpad](const u32* words, u32 count, u32 from) {
        for (u32 i = 0; i < count; ++i) {
            g_dma_source = from + 4 * i;
            if (scratchpad) std::memcpy(scratchpad + (sadr & 0x3FFCu), &words[i], 4);
            else GuestMemory().Write(kScratchpadBase + (sadr & 0x3FFCu), words[i], 4);
            sadr = (sadr + 4) & 0x3FFFu;
        }
    });
    *Register(ch.base + kMadr) = state.madr;
    *Register(ch.base + kQwc) = state.qwc;
    *Register(ch.base + kTadr) = state.tadr;
    *Register(ch.base + kSadr) = sadr & 0x3FF0u;
    *Register(ch.base) = state.chcr;
    if (!describe) return {};
    std::snprintf(text, sizeof text, "%llu quadwords to scratchpad", static_cast<unsigned long long>(outcome.quadwords));
    return text;
}

// GIF_STAT as VIF1's worker will leave it, worked out on the EE's thread, so
// a read need not wait for the worker (Vif1Guess). Burnout 3 starts each
// display-list packet by waiting for VIF1, VU1 and the GIF to be idle; with
// its transfers ending as they start, every wait was the EE waiting for all
// the worker had - ~15% of the EE's time at a race's start, the pictures it
// missed. Only the worker changes GIF_STAT: VIF1's MSKPATH3 (from the FIFO or
// a transfer) and a GIF transfer held behind the mask filling the FIFO. The
// EE follows the FIFO's commands and the GIF transfers itself, and takes a
// VIF1 transfer to leave GIF_STAT as the one before it left it. Each guess
// is checked on the worker, in turn; a wrong one ends guessing.
struct GifShadow {
    bool known = false;
    u32 value = 0;
};
GifShadow g_gif_shadow;                         // the EE's
std::atomic<u32> g_gif_after_vif1{0};           // the worker's: GIF_STAT as the last VIF1 transfer left it
std::atomic<bool> g_gif_after_known{false};
// The EE's: the register last answered by a guess, nothing it follows
// queued since - read again, the game is waiting for it to change.
u32 g_guessed_last = 0;
std::atomic<u64> g_guesses{0}, g_wrong{0};
std::atomic<bool> g_guessing_off{false};
bool g_check_guesses = false;

bool Shadowing() { return g_check_guesses || Vif1Async(); }

// With nothing queued, GIF_STAT is what memory holds (a check takes the
// worker to be always behind: only its guesses count).
void GifShadowFromMemory() {
    if (!g_check_guesses && !Vif1Busy()) g_gif_shadow = {true, *Register(kGifStat)};
}

void ShadowMask(bool masked) {
    u32& v = g_gif_shadow.value;
    v = masked ? (v | kGifStatM3P) : (v & ~(kGifStatM3P | kGifStatFqc));
}

u32 HeldInFifo(u64 quadwords) { return quadwords < kGifFifoDepth ? u32(quadwords) : kGifFifoDepth; }

// A VIF1 or GIF chain's words as the EE walked it, for VIF1's worker (Walk).
// Kept in a pool: a race walks a hundred chains a frame, and a vector made
// for each grew by reallocating for each of its runs - a sixth of the EE's
// time went to copying them about.
struct ChainCopy {
    std::vector<u32> words;
    std::vector<std::pair<u32, u32>> runs;      // (count, from) each, in order
};
std::mutex g_chain_pool_lock;
std::vector<ChainCopy*> g_chain_pool;

std::shared_ptr<ChainCopy> NewChainCopy() {
    ChainCopy* copy = nullptr;
    {
        std::lock_guard<std::mutex> hold(g_chain_pool_lock);
        if (!g_chain_pool.empty()) {
            copy = g_chain_pool.back();
            g_chain_pool.pop_back();
        }
    }
    if (!copy) {
        copy = new ChainCopy;
        copy->words.reserve(1u << 16);
        copy->runs.reserve(1024);
    }
    copy->words.clear();            // (the capacity stays)
    copy->runs.clear();
    return std::shared_ptr<ChainCopy>(copy, [](ChainCopy* done) {
        std::lock_guard<std::mutex> hold(g_chain_pool_lock);
        if (g_chain_pool.size() < 16) g_chain_pool.push_back(done);
        else delete done;
    });
}

// Walks a started transfer and leaves the registers as it would; returns a
// note (when `describe`: a transfer is noted only the first time).
std::string Walk(const Channel& ch, u32 chcr, bool describe) {
    Memory& m = GuestMemory();
    if (ch.base == 0x1000D000u || ch.base == 0x1000D400u) return Scratchpad(ch, chcr, describe);
    DmaChannelState state{chcr, u32(m.Read(ch.base + kMadr, 4)), u32(m.Read(ch.base + kQwc, 4)),
                          u32(m.Read(ch.base + kTadr, 4)), u32(m.Read(ch.base + kAsr0, 4)),
                          u32(m.Read(ch.base + kAsr1, 4))};
    // A run of words at a time, each handed on directly.
    auto to_vif = [](VifStream& vif) {
        return [&vif](const u32* words, u32 count, u32 from) { vif.Words(words, count, from); };
    };
    DmaBlockSink sink = [](const u32*, u32, u32) {};
    if (ch.base == 0x10008000u) sink = to_vif(Vif0());
    if (ch.base == 0x10009000u) sink = to_vif(Vif1());
    if (ch.base == 0x1000A000u) sink = Path3;
    // With VIF1's worker, VIF1's and the GIF's words are copied as the chain
    // is walked and handed to it in a command (the transfer ends here, as on
    // the console its data is taken).
    const bool queued = Vif1Async() && (ch.base == 0x10009000u || ch.base == 0x1000A000u);
    auto copy = queued ? NewChainCopy() : nullptr;
    if (queued)
        sink = [copy](const u32* words, u32 count, u32 from) {
            copy->words.insert(copy->words.end(), words, words + count);
            copy->runs.emplace_back(count, from);
        };
    // VIF1 stalls after a VIFcode with the interrupt bit (g_vif1_scan): its
    // words up to there go, and the transfer stops.
    const bool vif1_channel = ch.base == 0x10009000u;
    const bool interrupts = vif1_channel && !(*Register(kVif1Err) & 1);     // VIF1_ERR.MII
    DmaStallingSink stalling = [&sink, vif1_channel, interrupts](const u32* words, u32 count, u32 from) {
        if (!vif1_channel) {
            sink(words, count, from);
            return DmaTaken{count, false};
        }
        u32 taken = 0;
        bool stall = false;
        while (taken < count) {
            const VifInterruptScanner::Stop stop = g_vif1_scan.Scan(words + taken, count - taken);
            taken += stop.words;
            if (!stop.stall) break;
            if (interrupts) {
                stall = true;
                break;
            }
        }
        if (taken) sink(words, taken, from);
        return DmaTaken{taken, stall};
    };
    if (vif1_channel) state.end_pending = g_vif1_end_pending;
    char text[160] = "";
    try {
        Timed timed(Section::Dma);      // (VU1 and the GS, which it drives, timed apart)
        DmaOutcome outcome;
        // Resumed: first what was read past the stall.
        if (vif1_channel && !g_vif1_leftover.empty()) {
            std::vector<u32> before;
            before.swap(g_vif1_leftover);
            const DmaTaken taken = stalling(before.data(), u32(before.size()), state.madr);
            if (taken.stall) {
                outcome.stalled = true;
                g_vif1_leftover.assign(before.begin() + taken.words, before.end());
            }
        }
        if (!outcome.stalled) {
            outcome = SendDmaBlocksUntil(state, stalling);
            if (outcome.stalled && vif1_channel)
                g_vif1_leftover.assign(outcome.leftover, outcome.leftover + outcome.leftover_count);
        }
        if (vif1_channel) {
            g_vif1_end_pending = state.end_pending;
            if (outcome.stalled) {
                g_vif1_stalled = true;
                ++g_vif1_stalls;
                *Register(kVif1Stat) |= kVifStatVis | kVifStatInt;
                Note("vif1 interrupt", "VIF1: a VIFcode's interrupt bit stalls the stream (INTC 5, "
                                       "until FBRST.STC)");
                PostIntc(kIntcVif1);
            }
        }
        if (describe)
            std::snprintf(text, sizeof text, "%llu quadwords, %u tags%s",
                      static_cast<unsigned long long>(outcome.quadwords), outcome.tags,
                      outcome.stalled ? ", stalled at a VIFcode's interrupt bit"
                      : outcome.ended ? "" : ", stopped: the chain does not end");
        const bool gif = ch.base == 0x1000A000u, vif1 = ch.base == 0x10009000u;
        const u64 quadwords = outcome.quadwords;
        auto fifo = [gif, quadwords] {
            if (gif && (*Register(kGifStat) & kGifStatM3P))
                *Register(kGifStat) = (*Register(kGifStat) & ~kGifStatFqc) | (HeldInFifo(quadwords) << kGifStatFqcShift);
        };
        auto learn = [vif1] {
            if (!vif1) return;
            g_gif_after_vif1 = *Register(kGifStat);
            g_gif_after_known = true;
        };
        if (Shadowing() && (gif || vif1)) {
            GifShadowFromMemory();
            g_guessed_last = 0;             // a read from here on asks again
            if (vif1) {
                g_gif_shadow = {g_gif_after_known.load(), g_gif_after_vif1.load()};
            } else if (g_gif_shadow.known && (g_gif_shadow.value & kGifStatM3P)) {
                g_gif_shadow.value = (g_gif_shadow.value & ~kGifStatFqc) | (HeldInFifo(quadwords) << kGifStatFqcShift);
            }
        }
        if (queued) {
            Vif1Run([copy, gif, fifo, learn] {
                const u32* at = copy->words.data();
                for (const auto& [count, from] : copy->runs) {
                    if (gif) Path3(at, count, from);
                    else Vif1().Words(at, count, from);
                    at += count;
                }
                fifo();
                learn();
            });
        } else {
            fifo();
            learn();
        }
    } catch (const MemoryFault& fault) {
        std::snprintf(text, sizeof text, "stopped: read outside memory at 0x%08X", unsigned(fault.address));
    }
    *Register(ch.base + kMadr) = state.madr;
    *Register(ch.base + kQwc) = state.qwc;
    *Register(ch.base + kTadr) = state.tadr;
    *Register(ch.base + kAsr0) = state.asr0;
    *Register(ch.base + kAsr1) = state.asr1;
    *Register(ch.base) = state.chcr;
    return text;
}
constexpr u32 kVif1Fifo = 0x10005000u;

}  // namespace

namespace {
std::string g_frames_dir;
int g_frames_every = 30, g_frames_limit = 100, g_frames_written = 0;
u64 g_frames_from = 0;
}  // namespace

void SetFrameDumps(const std::string& dir, int every, int limit, u64 from) {
    g_frames_dir = dir;
    g_frames_every = every > 0 ? every : 1;
    g_frames_limit = limit;
    g_frames_from = from;
}

namespace {
long long g_trace_frame = -1;
}  // namespace

void SetGsTraceFrame(long long frame) { g_trace_frame = frame; }

namespace {
long long g_dump_frame = -1;
}  // namespace

void SetGsDumpFrame(long long frame) { g_dump_frame = frame; }

namespace {
u64 g_until_frame = 0;
bool g_until_reached = false;
}  // namespace

void SetUntilFrame(u64 frame) { g_until_frame = frame; }

namespace {
unsigned g_frame_times = 0;
u64 g_vif1_transfers = 0;       // DMA to VIF1 started: a game's frame of geometry, usually one each
}  // namespace

void SetFrameTimes(unsigned every) { g_frame_times = every; }

/* --frame-times: the frames since the last line, and how fast they went. */
void FrameTimes(u64 frame) {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point last = Clock::now();
    static u64 last_frame = 0, last_transfers = 0, last_lost = 0, last_idle = 0, last_worker_us = 0, last_gs_us = 0;
    if (!g_frame_times || frame < last_frame + g_frame_times) return;
    const u64 lost = LostVblanks(), idle = IdleNanoseconds(), worker_us = Vif1BusyMicroseconds();
    const u64 gs_us = GsBusyMicroseconds();
    const Clock::time_point now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last).count();
    char job_text[1024] = "";
    if (seconds > 0)
        std::snprintf(job_text, sizeof job_text, "%s, GS thread busy %d%%",
                      worker_us != last_worker_us
                          ? (", VIF1's worker busy " +
                             std::to_string(int(100.0 * double(worker_us - last_worker_us) / 1e6 / seconds + 0.5)) + "%")
                                .c_str()
                          : "",
                      int(100.0 * double(gs_us - last_gs_us) / 1e6 / seconds + 0.5));
    last_worker_us = worker_us;
    last_gs_us = gs_us;
    // The EE's waits for the worker, when it waited at all.
    static u64 last_join = 0, last_behind = 0, last_full = 0;
    u64 join = 0, behind = 0, full = 0;
    Vif1WaitMicroseconds(&join, &behind, &full);
    if (seconds > 0 && (join != last_join || behind != last_behind || full != last_full)) {
        const auto share = [&](u64 us) { return int(100.0 * double(us) / 1e6 / seconds + 0.5); };
        const std::size_t used = std::strlen(job_text);
        std::snprintf(job_text + used, sizeof job_text - used, ", EE waiting: joins %d%%, frames %d%%, room %d%%",
                      share(join - last_join), share(behind - last_behind), share(full - last_full));
    }
    last_join = join;
    last_behind = behind;
#ifdef PS2HOST_VULKAN
    {
        const std::string lag = PresentLag();
        const std::size_t used = std::strlen(job_text);
        std::snprintf(job_text + used, sizeof job_text - used, "%s", lag.c_str());
    }
#endif
    {
        const std::string gaps = iop::RunGaps();
        const std::size_t used = std::strlen(job_text);
        std::snprintf(job_text + used, sizeof job_text - used, "%s", gaps.c_str());
    }
    last_full = full;
    static u64 last_gpu_us = 0;
    const u64 gpu_us = GpuBusyMicroseconds();
    if (gpu_us != last_gpu_us && seconds > 0) {
        const std::size_t used = std::strlen(job_text);
        std::snprintf(job_text + used, sizeof job_text - used, ", GPU busy %d%%",
                      int(100.0 * double(gpu_us - last_gpu_us) / 1e6 / seconds + 0.5));
    }
    last_gpu_us = gpu_us;
#ifdef PS2HOST_VULKAN
    if (ps2gpu::Display* display = WindowDisplay()) {
        const ps2gpu::Display::Pacing p = display->TakePacing();
        if (p.presents && seconds > 0) {
            const std::size_t used = std::strlen(job_text);
            std::snprintf(job_text + used, sizeof job_text - used,
                          ", presents %.1f/s every %.2f ms (jitter %.2f, longest %.1f)", double(p.presents) / seconds,
                          p.mean_ms, p.jitter_ms, p.longest_ms);
            if (p.skipped) {
                const std::size_t more = std::strlen(job_text);
                std::snprintf(job_text + more, sizeof job_text - more, ", %llu skipped (the GPU behind)",
                              (unsigned long long)p.skipped);
            }
        }
    }
#endif
    static u64 last_guesses = 0;
    u64 guesses = 0, wrong = 0;
    Vif1Guesses(&guesses, &wrong);
    if (guesses != last_guesses) {
        const std::size_t used = std::strlen(job_text);
        std::snprintf(job_text + used, sizeof job_text - used, ", %llu status reads guessed (%llu wrong in all)",
                      (unsigned long long)(guesses - last_guesses), (unsigned long long)wrong);
    }
    last_guesses = guesses;
    // Frames here are vertical blanks; a game that cannot keep up draws a
    // picture every other one (or less), which the VIF1 transfers show.
    std::fprintf(stderr, "[frames] %llu-%llu: %.2f s, %.1f frames a second, %.1f VIF1 transfers a second%s\n",
                 static_cast<unsigned long long>(last_frame), static_cast<unsigned long long>(frame), seconds,
                 seconds > 0 ? double(frame - last_frame) / seconds : 0.0,
                 seconds > 0 ? double(g_vif1_transfers - last_transfers) / seconds : 0.0,
                 ((lost != last_lost ? ", " + std::to_string(lost - last_lost) + " vertical blanks lost (the EE busy through them)"
                                    : std::string()) +
                 (idle != last_idle && seconds > 0
                      ? ", EE idle " + std::to_string(int(100.0 * double(idle - last_idle) / 1e9 / seconds + 0.5)) + "%"
                      : std::string()) + job_text).c_str());
    last_lost = lost;
    last_idle = idle;
    last = now;
    last_frame = frame;
    last_transfers = g_vif1_transfers;
}
bool UntilFrameReached() { return g_until_reached; }

// F12 (or L3+R3): the next frame captured whole, for looking into what it
// drew - captures/<time>-frame<N>/ beside the PC settings: the picture at
// the internal resolution and at the PS2's, every drawing pass, GS local
// memory and the GS trace of the frame.
std::atomic<bool> g_capture_request{false};
long long g_capture_frame = -1;     // the frame the capture began at, -1: none
u64 g_capture_drawn = 0;            // Gs::drawn at the last blank
int g_capture_drawing = 0;          // blanks since the start the GS drew in
std::string g_capture_dir;

// The capture ends after three blanks the GS drew in: a picture drawn over
// two blanks is whole in it, wherever it began. Counted in blanks alone, it
// ended at once in real time, where a late EE takes its blanks one after
// another with nothing drawn between (an empty trace, no passes). A game
// drawing nothing ends after a second.
bool CaptureDone(u64 frame) {
    const u64 drawn = gs::TheGs().drawn.load(std::memory_order_relaxed);
    if (drawn != g_capture_drawn) ++g_capture_drawing;
    g_capture_drawn = drawn;
    const long long blanks = (long long)frame - g_capture_frame;
    return blanks >= 2 && (g_capture_drawing >= 3 || blanks >= 50);
}

void StartCapture(u64 frame) {
    namespace fs = std::filesystem;
    fs::path base = SettingsPath().empty() ? fs::path("captures") : fs::path(SettingsPath()).parent_path() / "captures";
    char name[64];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    std::snprintf(name, sizeof name, "%04d%02d%02d-%02d%02d%02d-frame%llu", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<unsigned long long>(frame + 1));
    const fs::path dir = base / name;
    std::error_code error;
    fs::create_directories(dir / "passes", error);
    if (error) {
        std::fprintf(stderr, "[capture] no directory %s: %s\n", dir.string().c_str(), error.message().c_str());
        return;
    }
    g_capture_dir = dir.string();
    gs::Gs& g = gs::TheGs();
    // The frames' GS stream (replay.bin), for --gs-replay: memory as it is
    // now, the GPU's drawing in it, the registers, then what comes.
    SyncGs();
    g_capture_frame = (long long)frame;
    g_capture_drawn = g.drawn.load(std::memory_order_relaxed);
    g_capture_drawing = 0;
    GpuCaptureBegin(frame + 1, (dir / "passes").string());
    g.Finish();
    if (g.on_memory) g.on_memory(0, 16384, false);
    g.BeginRecord(std::fopen((dir / "replay.bin").string().c_str(), "wb"));
    if (std::FILE* f = std::fopen((dir / "trace.txt").string().c_str(), "w")) {
        g.trace_out = f;
        g.trace = true;
        std::fprintf(f, "[gs] ---- frame %llu (a capture)\n", static_cast<unsigned long long>(frame + 1));
    }
    std::fprintf(stderr, "[capture] frame %llu to %s\n", static_cast<unsigned long long>(frame + 1),
                 g_capture_dir.c_str());
}

void FinishCapture(u64 frame) {
    gs::Gs& g = gs::TheGs();
    SyncGs();
    GpuCaptureEnd();
    g.EndRecord();
    if (g.trace_out != stderr) {
        g.trace = false;
        std::fclose(g.trace_out);
        g.trace_out = stderr;
    }
    const std::string screen = GpuCaptureScreen(g_capture_dir);
    g.Finish();
    if (g.on_memory) g.on_memory(0, 16384, false);      // the GPU's drawing, back in memory
    if (std::FILE* f = std::fopen((g_capture_dir + "/gsmem.bin").c_str(), "wb")) {
        std::fwrite(g.memory.data(), 1, 4u << 20, f);
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen((g_capture_dir + "/info.txt").c_str(), "w")) {
        std::fprintf(f, "frame %llu (from %lld, %d blanks drawn in)\n%s\nresolution setting: %s\n",
                     static_cast<unsigned long long>(frame), g_capture_frame + 1, g_capture_drawing, screen.c_str(),
                     Setting("resolution", "original").c_str());
        std::fprintf(f, "files: screen.bmp (as drawn, at the internal resolution), screen_1x.bmp (the PS2's "
                        "640x448), passes/ (each target as a pass left it, at 1x), gsmem.bin (GS local memory "
                        "after the frame: engine/tools/gsmem.py), trace.txt (the frame's GS packets and "
                        "primitives), replay.bin (the frames' GS stream: --gs-replay, PS2HOST_REPLAY_GPU, "
                        "PS2HOST_REPLAY_SCALE, PS2HOST_REPLAY_STOP_PRIM)\n");
        std::fclose(f);
    }
    std::fprintf(stderr, "[capture] saved %s\n", g_capture_dir.c_str());
    g_capture_frame = -1;
}

void RequestCapture() { g_capture_request = true; }

// The field being shown (CSR.FIELD): it changes at each vertical blank, an
// interlaced picture's two fields in turn (FrameDone).
std::atomic<u32> g_field{0};

u32 ShownField() { return g_field.load(std::memory_order_relaxed); }

std::atomic<u64> g_ee_frame{0};
u64 EeFrame() { return g_ee_frame.load(std::memory_order_relaxed); }

void FrameDone(u64 frame) {
    g_ee_frame.store(frame, std::memory_order_relaxed);
    g_field.store(u32(frame & 1), std::memory_order_relaxed);
    CodeFrame(frame);
    FrameTimes(frame);
    timeline::Frame(frame);
    SampleFrame(frame);
    Vu::SnapshotFrame(frame);
    // Time passes: unmasked, PATH3 sends what it still holds (in turn with
    // VIF1's, whose state PATH3's is).
    Vif1Run([] {
        if (!g_path3_masked && !g_path3_held.empty()) ReleasePath3(true);
    });
    if (g_until_frame && frame >= g_until_frame && !g_until_reached) {
        g_until_reached = true;
        StopRequested() = true;
        ForEachContext([](lifted::Context& each) { each.budget = -1; });
    }
    const bool dump = !g_frames_dir.empty() && g_frames_written < g_frames_limit && frame >= g_frames_from &&
                      frame % u64(g_frames_every) == 0;
    // Woven, a picture is this field and the one before: that one read too,
    // a frame ahead of the dump (Gs::Picture keeps it). A field is on the
    // lines of the parity the game drew it for - CSR.FIELD as it read it,
    // the frame before's.
    gs::TheGs().weave_parity = u32(frame + 1) & 1;
    const bool stash = gs::TheGs().weave && !dump && !g_frames_dir.empty() && g_frames_written < g_frames_limit &&
                       frame + 1 >= g_frames_from && (frame + 1) % u64(g_frames_every) == 0;
    // Nothing here reads the GS this frame (no picture to the CPU, no dump,
    // no trace): the GS thread ends the frame itself where the frame's
    // packets end - the GPU's frame, the window's picture - and the EE goes
    // straight on (it waited for the GS and the GPU at every frame before).
    // PS2HOST_CAPTURE_FRAME=N: as F12 at frame N (a scripted run's).
    static const long long capture_at = std::getenv("PS2HOST_CAPTURE_FRAME") ? std::atoll(std::getenv("PS2HOST_CAPTURE_FRAME")) : -1;
    if ((long long)frame == capture_at) g_capture_request = true;
    if (g_capture_request.exchange(false) && g_capture_frame < 0 && GpuPrimary()) StartCapture(frame);
    const bool capturing = g_capture_frame >= 0 && CaptureDone(frame);
    const bool digest = gs::TheGs().digesting;      // PS2HOST_GS_DIGEST: what the frame gave the GS, all of it
    const bool cpu_reads_gs = dump || stash || (g_dump_frame >= 0 && (long long)frame == g_dump_frame) ||
                              g_trace_frame >= 0 || capturing || digest;
    if (!cpu_reads_gs && GsFrame(frame)) {
        DbcFrame();
        if (!PumpWindow()) throw GuestExit{0};          // the user closed the window
        return;
    }
    // What reads the GS from here on - the picture, the GPU's frame, the
    // dumps - sees it with all of the frame's packets taken.
    JoinVif1();
    SyncGs();
    if (digest)
        std::fprintf(stderr, "[gs digest] frame %llu %016llX (%llu FBRST.STC not waited for)\n",
                     (unsigned long long)frame, (unsigned long long)gs::TheGs().digest,
                     (unsigned long long)StcUnwaited());
    if (capturing) FinishCapture(frame);
    if (g_dump_frame >= 0 && (long long)frame == g_dump_frame) {
        gs::Gs& g = gs::TheGs();
        g.Finish();
        if (g.on_memory) g.on_memory(0, 16384, false);      // the GPU's drawing, back in memory
        const std::string path = (g_frames_dir.empty() ? std::string(".") : g_frames_dir) + "/gsmem_" +
                                 std::to_string(frame) + ".bin";
        if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(g.memory.data(), 1, 4u << 20, f);
            std::fclose(f);
            std::fprintf(stderr, "[gs] local memory after frame %llu written to %s\n",
                         static_cast<unsigned long long>(frame), path.c_str());
        }
    }
    if (g_trace_frame >= 0) {
        // From the end of the frame before to the end of six frames on: the
        // game draws a picture across several.
        if ((long long)frame + 1 == g_trace_frame) gs::TheGs().trace = true;
        else if ((long long)frame == g_trace_frame + 5) gs::TheGs().trace = false;
        if (gs::TheGs().trace) std::fprintf(gs::TheGs().trace_out, "[gs] ---- frame %llu\n", static_cast<unsigned long long>(frame + 1));
    }
    DbcFrame();
    PresentFrame();
    if (WindowClosed()) throw GuestExit{0};             // the user closed the window
    GpuShadowFrame(frame, dump ? g_frames_dir : std::string());
    if (stash) {
        std::vector<u32> field;
        u32 w = 0, h = 0;
        gs::TheGs().Picture(&field, &w, &h);
    }
    if (!dump) return;
    char name[64];
    std::snprintf(name, sizeof name, "/frame_%06llu.bmp", static_cast<unsigned long long>(frame));
    if (gs::TheGs().WriteBmp(g_frames_dir + name)) ++g_frames_written;
    // VU1's memories beside the picture, for engine/tools/vudis.py.
    for (const auto& [file, span] : {std::make_pair("/vu1_micro.bin", Vu1().micro), std::make_pair("/vu1_data.bin", Vu1().data)})
        if (FILE* out = std::fopen((g_frames_dir + file).c_str(), "wb")) {
            std::fwrite(span.data(), 1, span.size(), out);
            std::fclose(out);
        }
}

bool Vif1Guessable(u32 address, u32 size) { return size == 4 && address == kGifStat; }

bool Vif1Guess(u32 address, u32* value) {
    if (!Vif1Async() || address != kGifStat || g_guessing_off || !g_gif_shadow.known) return false;
    // Read again straight away, the game waiting for it to change: it waits
    // for the worker.
    if (g_guessed_last == address) return false;
    g_guessed_last = address;
    const u32 guess = g_gif_shadow.value;
    ++g_guesses;
    Vif1Run([guess] {
        const u32 actual = *Register(kGifStat);
        if (actual == guess) return;
        g_guessing_off = true;
        if (g_wrong.fetch_add(1) < 8)
            std::fprintf(stderr, "[vif1] GIF_STAT guessed 0x%08X, held 0x%08X: no more guesses\n", unsigned(guess),
                         unsigned(actual));
    });
    *value = guess;
    return true;
}

void Vif1Settled(u32 address, u32 value) {
    if (address != kGifStat) return;
    if (g_check_guesses) {
        // What real time would have answered: a guess, or a wait for the worker.
        if (g_gif_shadow.known && g_guessed_last != address) {
            g_guessed_last = address;
            ++g_guesses;
            if (value != g_gif_shadow.value && g_wrong.fetch_add(1) < 32)
                std::fprintf(stderr, "[check] GIF_STAT would have been guessed 0x%08X, holds 0x%08X\n",
                             unsigned(g_gif_shadow.value), unsigned(value));
        } else {
            g_guessed_last = 0;
        }
        g_gif_shadow = {true, value};
        return;
    }
    if (g_guessed_last == address) g_guessed_last = 0;
    GifShadowFromMemory();
}

void Vif1Guesses(u64* guesses, u64* wrong) {
    *guesses = g_guesses.load();
    *wrong = g_wrong.load();
}

void SetCheckGuesses(bool on) { g_check_guesses = on; }

u64 Vif1InterruptStalls() { return g_vif1_stalls.load(); }

void DmaChannelEnded(u32 channel) {
    SetDStat(d_stat | (1u << channel));
    PostDmac(channel);
}

// A channel's STR set (or VIF1's stalled transfer going on, FBRST.STC): its
// transfer walked, and - unless VIF1 stalled in it - ended.
void StartTransfer(const Channel& ch, u32 value) {
    Memory& memory = GuestMemory();
    const u32 chcr = ch.base, madr = ch.base + kMadr, qwc = ch.base + kQwc;
    // The text only when it is logged: a DMA start is thousands a second.
    const std::string key = std::string("dma ") + ch.name;
    const bool traced = gs::TheGs().trace && (ch.base == 0x10009000u || ch.base == 0x1000A000u);
    const bool describe = traced || Noting(key);
    char text[256] = "";
    if (describe)
        std::snprintf(text, sizeof text,
                      "DMA %s started: CHCR=0x%08X MADR=0x%08X QWC=%u TADR=0x%08X (completed at once)",
                      ch.name, unsigned(value), unsigned(memory.Read(madr, 4)),
                      unsigned(memory.Read(qwc, 4)), unsigned(memory.Read(ch.base + kTadr, 4)));
    if (traced) std::fprintf(gs::TheGs().trace_out, "[dma] %s\n", text);
    if (ch.base == 0x10009000u) ++g_vif1_transfers;
    const std::string walked = ch.walked ? Walk(ch, value, describe) : std::string();
    if (describe) Note(key, ch.walked ? std::string(text) + ": " + walked : std::string(text));
    if (ch.base == 0x10009000u && g_vif1_stalled) return;  // STR stays set: not done
    *Register(chcr) &= ~0x100u;                         // done
    SetDStat(d_stat | (1u << ch.status_bit));
    PostDmac(ch.status_bit);
}

void InstallDevices() {
    Memory& m = GuestMemory();
    for (const Channel& ch : kChannels) {
        if (ch.base == 0x1000B000u || ch.base == 0x1000B400u) continue;     // the IPU's (ipu.cpp)
        m.OnWrite(ch.base, [ch](u32, u64 value) {
            if (!(value & 0x100)) return;                       // STR not set
            StartTransfer(ch, u32(value));
        });
    }
    // VIF1_FBRST: STC (bit 3) ends a stall at an interrupt bit, and the
    // transfer goes on; RST (bit 0) resets VIF1 - its stall with it.
    m.OnWrite(kVif1Fbrst, [](u32, u64 value) {
        if (value & 1) {
            g_vif1_scan.Reset();
            g_vif1_leftover.clear();
            g_vif1_end_pending = false;
            g_vif1_stalled = false;
            *Register(kVif1Stat) &= ~(kVifStatVis | kVifStatInt);
        }
        if ((value & 8) && g_vif1_stalled) {
            g_vif1_stalled = false;
            *Register(kVif1Stat) &= ~(kVifStatVis | kVifStatInt);
            StartTransfer(kChannels[1], *Register(kChannels[1].base));
        }
    });
    m.OnWrite(kDStat, [](u32, u64 value) {
        const u32 written = u32(value);
        SetDStat((d_stat & ~(written & 0xFFFFu)) ^ (written & 0xFFFF0000u));
    });
    Vif1().on_mask_path3 = [](bool masked) {
        if (gs::TheGs().trace)
            std::fprintf(stderr, "[vif1] MSKPATH3 %d, %zu PATH3 words held\n", int(masked), g_path3_held.size());
        u32* stat = reinterpret_cast<u32*>(GuestMemory().Translate(kGifStat, 4));
        // Unmasking lets PATH3 take its turn: the packet group it is in goes
        // through before a mask can stop it again.
        *stat = masked ? (*stat | kGifStatM3P) : (*stat & ~(kGifStatM3P | kGifStatFqc));
        g_path3_masked = masked;
        if (!masked) ReleasePath3(false);
    };
    Vif1().on_direct = [](u32 word) { GsWord(2, word); };
    Vif1().on_direct_block = [](const u32* words, u32 count, u32 from) { GsWords(2, words, count, from); };
    Vif1().on_trace = [](const char* what) {
        if (gs::TheGs().trace) std::fprintf(gs::TheGs().trace_out, "[vif1] %s, %zu PATH3 words held\n", what, g_path3_held.size());
        // FLUSHA waits for PATH3 to finish: unmasked, all it holds goes.
        if (!g_path3_masked) ReleasePath3(true);
    };
    // VU1 behind VIF1; its XGKICKs are GIF PATH1.
    Vif1().vu = &Vu1();
    Vif1().double_buffered = true;
    Vif0().vu = &Vu0();
    Vu1().on_kick = [](u64 low, u64 high) { GsQword(1, low, high); };
    Vu1().on_kick_block = [](const u32* words, u32 count) { GsWords(1, words, count); };
    // The GS's privileged registers: what is displayed. A 64-bit store reaches
    // both words; each passes the whole register on.
    for (u32 reg : {0x12000000u, 0x12000020u, 0x12000070u, 0x12000080u, 0x12000090u, 0x120000A0u, 0x120000E0u}) {
        for (u32 half : {0u, 4u}) {
            GuestMemory().OnWrite(reg + half, [reg](u32, u64) {
                GsPrivileged(reg, GuestMemory().Read(reg, 8));
            });
        }
    }
    // The FIFO takes whole quadwords; the store's last word hands all four over.
    // With VIF1's worker, in turn after what it has (memory.cpp lets the
    // store through without waiting).
    m.OnWrite(kVif1Fifo + 12, [](u32, u64) {
        const u32* fifo = Register(kVif1Fifo);
        const std::array<u32, 4> words{fifo[0], fifo[1], fifo[2], fifo[3]};
        if (Shadowing()) {
            // Commands of a word, as the FIFO's quadwords are here: MSKPATH3's
            // effect followed, the others leaving GIF_STAT be; anything else,
            // and what it leaves is not known.
            GifShadowFromMemory();
            g_guessed_last = 0;
            for (u32 word : words) {
                const u32 cmd = (word >> 24) & 0x7F;
                if (cmd == 0x06) ShadowMask((word >> 15) & 1);
                else if (!(cmd <= 0x05 || cmd == 0x07 || cmd == 0x10 || cmd == 0x11 || cmd == 0x13))
                    g_gif_shadow.known = false;
            }
        }
        // The interrupt scanner follows these words too. Stored by the EE,
        // they are not held back: a bit among them raises the interrupt only.
        for (u32 word : words) {
            if (g_vif1_scan.Scan(&word, 1).stall && !(*Register(kVif1Err) & 1)) {
                ++g_vif1_stalls;
                *Register(kVif1Stat) |= kVifStatInt;
                PostIntc(kIntcVif1);
            }
        }
        Vif1Run([words] {
            for (u32 word : words) Vif1().Word(word);
        });
    });
    m.OnRead(kGsCsr, [](u32, u64 stored) {
        // FIELD: the field shown since the last vertical blank - the same
        // however often it is read (it changed at every read: a game reading
        // it twice in a field saw both).
        const u64 field = u64(ShownField()) << 13;
        return (stored & ~u64(1u << 13)) | field | (1u << 3);   // VSINT pending
    });
    m.OnRead(0x1000F000u, [](u32, u64) { return u64(IntcStat()); });     // INTC_STAT
    m.OnWrite(0x1000F000u, [](u32, u64 value) { ClearIntcStat(u32(value) & 0x7FFFu); });
    InstallTimers();
    InstallIpu();
}

}  // namespace ps2host
