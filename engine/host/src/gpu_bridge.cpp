/*
 * The GPU GS renderer (engine/gpu, engine/docs/GPU.md) driven by the host's GS
 * model, in one of two modes.
 *
 * --gpu-shadow: beside the software GS, and checked against it. Every
 * primitive the software GS draws is also handed to ps2gpu with the state it was
 * drawn with. Textures are decoded from the software GS's local memory (with
 * --gpu-shadow-rtt, a texture that is a GPU target is sampled from the target
 * instead); depth buffers start each frame as local memory has them. At the end
 * of each frame every target drawn into is read back and compared with local
 * memory, and the agreement is reported.
 *
 * --gpu: instead of the software GS. The GS model still takes every GIF packet,
 * register and image transfer, but draws nothing (Gs::rasterize false); the GPU
 * draws, and its images stand for regions of local memory.
 *
 * Local memory is kept twice: the GS model's, which the CPU reads and writes,
 * and a copy on the GPU (GsRenderer::Memory*), which images are swizzled into
 * and out of by a compute shader; per 256-byte block the bridge knows which
 * copy is current. Each GPU image (a target or depth buffer, with the blocks
 * it covers) knows whether the GPU has drawn what memory lacks (gpu_newer) or
 * memory has changed under it (stale). Before the CPU reads or writes local
 * memory - a transfer, a local copy, a CLUT load, a texture decoded, the
 * picture shown - the GPU images over it are swizzled into the GPU's copy and
 * the blocks the CPU lacks come back to it; after it is written, the images
 * there are stale and reload, from the GPU's copy once the CPU's writes have
 * gone over. A target drawn under another width or format than one it
 * overlaps (Burnout 3's reflection buffers, the race's blur passes in its
 * depth buffer's memory), or a depth buffer drawn into as colour, is
 * reconciled the same way - all of it on the GPU, waiting for nothing. An image stands for the
 * rows drawn into so far, not all the scissor allows: Burnout 3's depth buffer
 * starts where its frame buffer's 448 rows end, and a 512-row reach would have
 * each depth write make the frame stale. A texture that is a target is sampled
 * from the target itself. Other textures are decoded into a cache keyed by
 * their registers, TEXA and their CLUT, dropped when memory under them is
 * written or the GPU draws over them.
 *
 * Known difference: the GPU's blender rounds where the GS truncates, so a
 * blended pixel can be 1 brighter (invisible; in shadow runs, Burnout 3's frame
 * copy doubles it into the 16-bit display buffer).
 *
 * Without PS2HOST_VULKAN the options report that they are unavailable.
 */

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ps2host/gs.h"
#include "ps2host/profile.h"
#include "ps2host/run.h"
#include "ps2host/timeline.h"
#include "ps2host/settings.h"
#include "ps2host/texture_pack.h"

#if PS2HOST_VULKAN
#include "ps2gpu/display.h"
#include "ps2gpu/gs_renderer.h"
#endif

namespace ps2host {

#if PS2HOST_VULKAN

ps2gpu::Gpu* WindowGpu();       // window.cpp: the window's device, when there is one

namespace {

using gs::u32;
using gs::u64;

u32 Bits(u64 value, int low, int count) { return u32((value >> low) & ((1ull << count) - 1)); }

/* Bits a pixel of a format takes in local memory (the 8H/4HL/4HH formats live
   in 32-bit pixels). */
u32 PixelBits(u32 psm) {
    switch (psm) {
        case gs::PSMCT16: case gs::PSMCT16S: case gs::PSMZ16: case gs::PSMZ16S: return 16;
        case gs::PSMT8: return 8;
        case gs::PSMT4: return 4;
        default: return 32;
    }
}

bool Is16(u32 psm) { return psm == gs::PSMCT16 || psm == gs::PSMCT16S || psm == gs::PSMZ16 || psm == gs::PSMZ16S; }

/* The blocks a buffer's first `rows` rows take: whole rows of pages (32 blocks
   each), FBW pages wide. */
std::pair<u32, u32> Blocks(u32 fbp, u32 fbw, u32 psm, u32 rows) {
    const u32 page_rows = Is16(psm) ? 64 : 32;
    const u32 begin = fbp * 32;
    return {begin, begin + std::max(1u, (rows + page_rows - 1) / page_rows) * std::max(1u, fbw) * 32};
}

u32 To16(u32 c) {
    return ((c >> 3) & 31) | (((c >> 11) & 31) << 5) | (((c >> 19) & 31) << 10) | ((c >> 31) << 15);
}

u32 From16(u32 c) {
    return ((c & 31) << 3) | (((c >> 5) & 31) << 11) | (((c >> 10) & 31) << 19) | ((c & 0x8000) ? 0x80000000u : 0);
}

struct Extent {
    u32 x1 = 0, y1 = 0;             // the furthest pixel drawn, exclusive
};

/* Local memory's 512 pages (8 KB, 32 blocks each), as a set. */
using Pages = std::bitset<512>;

/* A GPU image standing for a region of local memory (--gpu). */
struct Owned {
    u32 fbp = 0, fbw = 1, psm = 0;  // FBP (ZBP) in 2048-word units, FBW, PSM
    bool depth = false;
    u32 width = 0, height = 0;      // the image
    // The blocks it stands for: the rows used so far (drawn into, or sampled),
    // of all it could hold; of those, the ones the GPU has drawn into.
    u32 begin = 0, end = 0, full_end = 0;
    u32 drawn_end = 0;
    // The blocks drawn into since it last went to GPU memory: what another
    // image over them must wait for (not all it was ever drawn over - the
    // race's depth buffer is cleared to 512 rows once a frame, over the glow
    // buffers after its 448, and drawn into below 448 ten thousand times).
    u32 dirty_begin = ~0u, dirty_end = 0;
    u32 x1 = 0, y1 = 0;             // how much of it the GPU has drawn
    bool gpu_newer = false;         // drawn on the GPU, not yet in local memory
    bool stale = true;              // local memory changed since the image was loaded
    bool stale_beyond = false;      // ...in rows past `end`: stale when it grows into them
    int why = 0;                    // what made it stale: 0 new, 1 memory written, 2 a draw over it, 3 another's write-back
    bool by16 = false;              // ...only ever a 16-bit view's drawing (Burnout 3's depth into the frame's alpha)
    // The rows a reload brings back (stale ones), [stale_row0, stale_row1):
    // a reload was the whole image, and the race's blur passes, drawn into
    // the first rows of its depth buffer's memory, had its 640x448 reloaded
    // a thousand times a second for a few rows each.
    u32 stale_row0 = 0, stale_row1 = ~0u;
    // The same by page, as finely as known: what changed under the image
    // since it was loaded, and what it drew since its last write-back. A
    // texture sampled from it needs neither unless the pages it reads are
    // in them (the race's road reads 16 texels of a 1024x256 view over its
    // depth buffer, while drawing Z into other pages of it - 200 reloads,
    // each with the depth buffer's write-back, a picture).
    Pages stale_pages, dirty_pages;
    // The page rectangle last drawn (x0, x1, y0, y1, inclusive), all of it
    // in dirty_pages while `rect_dirty`: most primitives land inside it.
    u32 rect[4] = {};
    bool rect_dirty = false;
    u64 reloads[4] = {};            // reloads, by why
    u64 drawn_serial = 0;           // when the GPU last drew into it (Bridge::serial)
    u64 checked = 0;                // Bridge::newer_epoch when Use last looked at the others
    // Bridge::owned_pages: the blocks it is listed over, [begin, indexed_end);
    // the last ForOverlapping that went over it.
    u32 indexed_end = 0;
    u64 visited = 0;
};

/* A texture decoded into the renderer (--gpu). */
struct Cached {
    u64 renderer_key = 0;
    // A texture pack's replacement (texture_pack.h): drawn instead, or its
    // file still being read (looked for again each time it is drawn); as of
    // the pack's generation (Tab turns it on and off).
    bool replaced = false, pack_pending = false;
    texture_pack::Name pack_name;
    u64 replacement = 0;            // the renderer's image for it (texture_pack::Image::id)
    u64 pack_generation = 0;
    u32 begin = 0, end = 0;
    bool valid = false;
    u64 decoded_serial = 0;         // Bridge::serial when decoded
    u64 content = 0;                // a hash of its blocks when decoded
    bool decoded = false;
    // The images whose blocks it can share (all they could ever cover), as of
    // `owned_seen` images: images are only ever added, and live as long as
    // the bridge.
    std::vector<const Owned*> over;
    std::size_t owned_seen = 0;
    std::list<u64>::iterator lru;
};

struct Bridge {
    std::unique_ptr<ps2gpu::Gpu> gpu;
    std::unique_ptr<ps2gpu::GsRenderer> renderer;
    bool primary = false;
    // --gpu-shadow-rtt: a texture that is a GPU target is sampled from it, as
    // the GPU will have to on its own.
    bool targets_as_textures = false;

    // Shadow: the texture last decoded, by what decides its contents, and the
    // memory drawn into since.
    std::tuple<u64, u64, u64, u64> texture_state{~0ull, 0, 0, 0};
    u64 texture_key = 0;
    std::map<u32, u32> since_decode;  // start -> blocks
    bool drawn_into(u32 bp, u32 blocks) const {
        for (const auto& [start, count] : since_decode)
            if (start < bp + blocks && bp < start + count) return true;
        return false;
    }
    std::map<std::tuple<u32, u32, u32>, Extent> drawn;  // this frame's targets
    u64 frames = 0;
    double agreement_sum = 0;
    std::string dump_dir;           // a dump asked for, made at the next frame with drawing
    std::set<u32> depth_seeded;     // depth buffers loaded from local memory this frame

    // Primary.
    std::map<std::tuple<u32, u32, u32, bool>, Owned> owned;
    // The images drawn on the GPU and not yet in local memory (gpu_newer) -
    // and some no longer, dropped when next looked at (ForNewer): what a
    // write-back is looked for in, without going over all of `owned` (the
    // stunt has hundreds, a few newer at a time). newer_list setting.
    // The images over each page (32 blocks) of local memory, as far as they
    // reach (Owned::indexed_end): what a write or a draw over blocks makes
    // stale, without going over all of `owned` (ForOverlapping). The
    // owned_index setting.
    std::vector<std::vector<Owned*>> owned_pages = std::vector<std::vector<Owned*>>(512);
    u64 visit = 0;
    bool owned_index = true;
    mutable std::vector<Owned*> newer;
    mutable bool newer_sorted = true;
    bool newer_list = true;
    std::unordered_map<u64, Cached> textures;
    std::list<u64> texture_lru;     // most recent first
    // The textures over each page (32 blocks) of local memory, for what a
    // write there invalidates: scanning all 2,048 on every write was 3% of a
    // race. Elements of `textures` stay put until erased.
    std::vector<std::vector<Cached*>> texture_pages = std::vector<std::vector<Cached*>>(512);
    void IndexTexture(Cached* t, bool add) {
        for (u32 page = t->begin >> 5; page <= ((std::max(t->end, t->begin + 1) - 1) >> 5) && page < 512; ++page) {
            std::vector<Cached*>& list = texture_pages[page];
            if (add) {
                list.push_back(t);
            } else {
                auto found = std::find(list.begin(), list.end(), t);
                if (found != list.end()) {
                    *found = list.back();
                    list.pop_back();
                }
            }
        }
    }
    u64 next_texture = 1;
    u64 serial = 0;                 // counts GPU draws, so a texture knows if one went over it
    // Bumped when an image starts holding drawing memory lacks, or reaches
    // further: until then, an image Use has checked against the others needs
    // no second look (tens of thousands of draws a frame go to one target).
    u64 newer_epoch = 1;
    // Local memory is kept twice: the GS model's (the CPU's) and the
    // renderer's (the GPU's), which images are swizzled into and out of. Per
    // 256-byte block, which copy is current.
    enum : u8 { kSame = 0, kCpuNewer = 1, kGpuNewer = 2 };
    std::vector<u8> blocks = std::vector<u8>(16384, kSame);
    u64 downloads = 0, uploads = 0;  // blocks copied GPU -> CPU, CPU -> GPU
    // Why blocks came back to the CPU (PS2HOST_GPU_LOG): 0 a texture decoded,
    // 1 the GS model reading, 2 the GS model writing.
    int cause = 1;
    u64 downloads_by[3] = {};
    // A transfer's blocks it writes in part where the GPU holds them newer:
    // its pixels there patched into the GPU's memory once it is done
    // (ApplyPatch), not the blocks brought back first (the CPU would wait for
    // every draw before them).
    struct Patch {
        gs::Gs::TouchSeen transfer{};
        std::vector<u32> blocks;            // sorted
    } patch;
    u64 patched_words = 0, patches = 0;
    std::map<u32, u64> texture_downloads;   // by the texture's TEX0 PSM << 16 | TBW
    u64 write_backs = 0, reloads = 0, decodes = 0;
    u64 unchanged = 0;              // textures invalidated whose texels proved the same
    u64 gpu_decodes = 0;            // textures decoded on the GPU from its copy of local memory
    u64 replacements = 0;           // a texture pack's in their place (texture_pack.h)
    // The CLUT, in rows of 16 entries (bit r for row r): which the GS model
    // holds as loaded (a load from blocks the CPU has), which the GPU's copy
    // holds (loaded from GPU memory, where the GPU drew them, or sent over).
    // Every row is at least one of the two. A GPU load bumps its rows'
    // generations, which key the textures read through them.
    u32 clut_cpu = 0xFFFF, clut_gpu = 0;
    u64 clut_generation[16] = {};
    u64 gpu_clut_loads = 0;
    // The texture cache (texture_cache, --texture-cache): a CLUT the GPU
    // loads again from the same blocks, set the same way, with nothing
    // written or drawn there since, is the same CLUT - not loaded again, its
    // rows' generation kept, so the textures read through it stay decoded
    // (Stuntman's stunt loads one ~90 times a frame, each time decoding its
    // 512x512 texture again). A block's last CPU write (stamp), the last draw
    // over it (Owned::drawn_serial) and the load's settings make a source;
    // each GPU row knows the source it was loaded from (0: none, or rows sent
    // from the CPU since). "check": loaded all the same, the GPU's CLUT read
    // before and after and any difference said - the cache's soundness, run
    // before it is trusted.
    enum class TextureCache { kOff, kOn, kCheck } texture_cache = TextureCache::kOn;
    std::vector<u64> block_stamp = std::vector<u64>(16384, 0);
    u64 stamp = 0;
    u64 gpu_clut_source[16] = {};
    u64 clut_loads_kept = 0, clut_checks = 0, clut_check_differences = 0;
    // PS2HOST_GPU_DUMP_FRAME=N (and PS2HOST_GPU_DUMP_DIR): during frame N, each
    // target as it stood when drawing moved on to another, as a BMP, with a
    // line per pass on stderr - to see which pass of a frame goes wrong.
    u64 frame = 0;
    long long dump_frame = -1, dump_last = -1;    // the frames whose passes are dumped
    bool Dumping() const { return (long long)frame >= dump_frame && (long long)frame <= std::max(dump_frame, dump_last); }
    std::string dump_passes = ".";
    std::tuple<u32, u32, u32> last_target{~0u, 0, 0};
    u32 pass = 0;
};

Bridge* g_bridge = nullptr;

// Bumped by whatever could change what OnDraw works out for a primitive from
// its registers alone - an image's staleness, memory synced, a write-back, a
// CLUT load, a texture decoded or released, an image made or reloaded, a
// draw outside the blocks its target had dirty: while it stands still, a
// primitive with the registers of the one before draws as that one did
// (OnDraw's fast path).
u64 g_mutations = 0;

/* `f` for each image the GPU holds newer than local memory, in the order
   of `owned` (its keys'): the order write-backs land in, as before. `f` may
   write images back, not draw. */
template <typename F>
void ForNewer(const Bridge& b, F&& f) {
    if (!b.newer_list) {
        for (const auto& [key, o] : b.owned)
            if (o.gpu_newer) f(const_cast<Owned&>(o));
        return;
    }
    auto& list = b.newer;
    list.erase(std::remove_if(list.begin(), list.end(), [](const Owned* o) { return !o->gpu_newer; }), list.end());
    if (!b.newer_sorted) {
        std::sort(list.begin(), list.end(), [](const Owned* x, const Owned* y) {
            return std::make_tuple(x->fbp, x->fbw, x->psm, x->depth) < std::make_tuple(y->fbp, y->fbw, y->psm, y->depth);
        });
        b.newer_sorted = true;
    }
    for (std::size_t i = 0; i < list.size(); ++i)
        if (list[i]->gpu_newer) f(*list[i]);
}

/* `o` listed over the blocks it reaches: [begin, the furthest of its ends). */
void IndexOwned(Bridge& b, Owned& o) {
    const u32 reach = std::min(16384u, std::max({o.full_end, o.end, o.drawn_end, o.dirty_end == 0 ? 0u : o.dirty_end}));
    if (reach <= o.indexed_end) return;
    const u32 from = o.indexed_end ? ((o.indexed_end - 1) >> 5) + 1 : (o.begin >> 5);
    for (u32 page = from; page <= ((reach - 1) >> 5) && page < 512; ++page) b.owned_pages[page].push_back(&o);
    o.indexed_end = reach;
}

/* `f` for each image that may lie over blocks [begin, end) - each once, in
   no particular order (`f` makes its own test of them, and changes only the
   image it is given). */
template <typename F>
void ForOverlapping(Bridge& b, u32 begin, u32 end, F&& f) {
    if (!b.owned_index) {
        for (auto& [key, o] : b.owned) f(o);
        return;
    }
    end = std::min(end, 16384u);
    if (begin >= end) return;
    const u64 visit = ++b.visit;
    for (u32 page = begin >> 5; page <= ((end - 1) >> 5); ++page)
        for (Owned* o : b.owned_pages[page])
            if (o->visited != visit) {
                o->visited = visit;
                f(*o);
            }
}

/* Whether the CLUT has rows the GPU's copy lacks. */
bool b_clut_rows_cpu_only(const Bridge& b) { return (b.clut_cpu & ~b.clut_gpu) != 0; }
std::map<std::string, u64> g_decode_kinds;      // PS2HOST_GPU_COUNT
// PS2HOST_GPU_COUNT: which write-backs anything read again - a reload, a
// texture or CLUT from GPU memory, the CPU - before their blocks were
// written over: the block's last write-back, and whether each was read.
bool CountingWriteBacks() {
    static const bool on = std::getenv("PS2HOST_GPU_COUNT") != nullptr;
    return on;
}
std::vector<u32> g_wb_owner(16384, 0);
std::vector<u8> g_wb_read(1, 1);
std::vector<u32> g_wb_blocks(1, 0);             // blocks each still owns
u64 g_wb_done = 0, g_wb_never_read = 0, g_wb_never_read_blocks = 0;
void SetWriteBackOwner(u32 i, u32 id) {
    const u32 old = g_wb_owner[i];
    if (old == id) return;
    if (old && !g_wb_read[old]) ++g_wb_never_read_blocks;
    if (old && --g_wb_blocks[old] == 0 && !g_wb_read[old]) ++g_wb_never_read;
    g_wb_owner[i] = id;
    if (id) ++g_wb_blocks[id];
}
void WriteBackRead(u32 begin, u32 end) {
    if (!CountingWriteBacks()) return;
    for (u32 i = begin; i < std::min(end, 16384u); ++i)
        if (g_wb_owner[i]) g_wb_read[g_wb_owner[i]] = 1;
}
void WriteBackLost(u32 begin, u32 end) {        // written over: the write-back there done with
    if (!CountingWriteBacks()) return;
    for (u32 i = begin; i < std::min(end, 16384u); ++i) SetWriteBackOwner(i, 0);
}

// ------------------------------------------------------------------ --gpu

}  // namespace

u64 GpuBusyMicroseconds() { return ps2gpu::GpuBusyNanoseconds() / 1000; }

namespace {

/* Blocks [begin, end) whose copy on one side is behind the other's brought up
   to date, in runs: `to_gpu` the GPU's from the CPU's, else the CPU's from the
   GPU's (which waits for the GPU). */
void Coherent(Bridge& b, u32 begin, u32 end, bool to_gpu) {
    end = std::min(end, 16384u);
    const u8 behind = to_gpu ? Bridge::kCpuNewer : Bridge::kGpuNewer;
    u8* bytes = gs::TheGs().memory.data();
    const u8* blocks = b.blocks.data();
    for (u32 i = begin; i < end;) {
        // To the next block behind (memchr, sixteen at a time and more: a
        // reload looks over all of an image's blocks, ~300 times a picture,
        // and finds none behind, or few).
        const void* found = std::memchr(blocks + i, behind, end - i);
        if (!found) break;
        i = u32(static_cast<const u8*>(found) - blocks);
        u32 j = i;
        while (j < end && b.blocks[j] == behind) b.blocks[j++] = Bridge::kSame;
        if (to_gpu) {
            b.renderer->MemoryWrite(i, j, bytes + std::size_t(i) * 256);
            b.uploads += j - i;
        } else {
            // The CPU waits for the GPU here: on the profiler's timeline, by why.
            static const char* const kWhy[3] = {"GPU to CPU: a texture", "GPU to CPU: a read", "GPU to CPU: a write"};
            const u64 from = timeline::On() ? timeline::Now() : 0;
            b.renderer->MemoryRead(i, j, bytes + std::size_t(i) * 256);
            if (from) timeline::Span(kWhy[b.cause], from, timeline::Now());
            static const bool log_writes = std::getenv("PS2HOST_GPU_WRITE_LOG") != nullptr;
            if (log_writes && b.cause == 2) {
                const auto& t = gs::TheGs().last_transfer;
                std::fprintf(stderr, "[gpu] blocks %X-%X back to the CPU for a write: bp %X bw %u psm %02X at %u,%u %ux%u\n",
                             unsigned(i), unsigned(j), unsigned(t.bp), unsigned(t.bw), unsigned(t.psm), unsigned(t.x),
                             unsigned(t.y), unsigned(t.w), unsigned(t.h));
                for (const auto& [key, o] : b.owned)
                    if (o.begin <= i && i < o.full_end)
                        std::fprintf(stderr, "[gpu]   over image %s %X/%u/%02X %ux%u blocks %X-%X (to %X), dirty %X-%X\n",
                                     o.depth ? "Z" : "C", unsigned(o.fbp), unsigned(o.fbw), unsigned(o.psm),
                                     unsigned(o.width), unsigned(o.height), unsigned(o.begin), unsigned(o.end),
                                     unsigned(o.full_end), unsigned(o.dirty_begin), unsigned(o.dirty_end));
            }
            b.downloads += j - i;
            b.downloads_by[b.cause] += j - i;
        }
        i = j;
    }
}

/* `o` stale for `why`, in the rows over blocks [b0, b1) (all of it, past its
   blocks or by default). */
void MarkStale(Owned& o, int why, u32 b0 = 0, u32 b1 = ~0u, const Pages* pages = nullptr, bool by16 = false) {
    ++g_mutations;
    if (pages) {
        o.stale_pages |= *pages;
    } else if (b1 != ~0u && b0 < b1) {
        for (u32 p = b0 / 32; p <= (b1 - 1) / 32 && p < 512; ++p) o.stale_pages.set(p);
    } else {
        o.stale_pages.set();
    }
    u32 r0 = 0, r1 = ~0u;
    if (b1 != ~0u && b0 < b1 && b1 > o.begin) {
        const u32 page_rows = Is16(o.psm) ? 64 : 32, row_blocks = std::max(1u, o.fbw) * 32;
        r0 = (std::max(b0, o.begin) - o.begin) / row_blocks * page_rows;
        r1 = (b1 - o.begin + row_blocks - 1) / row_blocks * page_rows;
    }
    o.by16 = (o.stale ? o.by16 : true) && by16;
    if (!o.stale) {
        o.stale = true;
        o.why = why;
        o.stale_row0 = r0;
        o.stale_row1 = r1;
    } else {
        o.stale_row0 = std::min(o.stale_row0, r0);
        o.stale_row1 = std::max(o.stale_row1, r1);
    }
}

/* A step of the bridge's on the profiler's timeline (PS2HOST_PROFILE), by name. */
struct TimelineStep {
    const char* name;
    u64 from;
    explicit TimelineStep(const char* what) : name(what), from(timeline::On() ? timeline::Now() : 0) {}
    ~TimelineStep() {
        if (from) timeline::Span(name, from, timeline::Now());
    }
};

/* What the GPU drew into an image, swizzled into the GPU's local memory - on
   the GPU, nothing waiting. */
void WriteBack(Bridge& b, Owned& o) {
    ++g_mutations;
    Timed timed(Section::GpuSync);
    const TimelineStep step("sync: a write-back");
    o.gpu_newer = false;
    o.dirty_pages.reset();
    o.rect_dirty = false;
    if (o.dirty_end <= o.dirty_begin) return;
    // Only the rows drawn into since the last write-back: the image's other
    // rows may be stale - another image drew over them since, and memory has
    // that, not this.
    const u32 page_rows = Is16(o.psm) ? 64 : 32, row_blocks = std::max(1u, o.fbw) * 32;
    const u32 first = (o.dirty_begin - o.begin) / row_blocks * page_rows;
    const u32 last = std::min(o.y1, (o.dirty_end - o.begin) / row_blocks * page_rows);
    // Blocks the CPU wrote go over first: the image covers them only in part.
    Coherent(b, o.dirty_begin, o.dirty_end, true);
    if (gs::TheGs().trace)   // in a capture's trace: an image written back to memory, which rows
        std::fprintf(gs::TheGs().trace_out, "[gpu] write-back %s %X/%u/%02X rows %u-%u cols %u\n", o.depth ? "Z" : "C",
                     unsigned(o.fbp), unsigned(o.fbw), unsigned(o.psm), unsigned(first), unsigned(last),
                     unsigned(o.x1));
    const bool done = o.depth ? b.renderer->DepthToMemory(o.fbp, o.width, o.height, o.fbw, o.psm, o.x1, last, first)
                              : b.renderer->TargetToMemory(o.fbp, o.fbw, o.psm, o.x1, last, first);
    if (!done && gs::TheGs().trace) std::fprintf(gs::TheGs().trace_out, "[gpu]   (nothing to write back)\n");
    if (done && CountingWriteBacks()) {
        // The one before over these blocks, if never read, is now lost.
        const u32 id = u32(g_wb_read.size());
        g_wb_read.push_back(0);
        g_wb_blocks.push_back(0);
        ++g_wb_done;
        for (u32 i = std::min(o.dirty_begin, 16384u); i < std::min(o.dirty_end, 16384u); ++i) SetWriteBackOwner(i, id);
    }
    if (done) {
        ++b.write_backs;
        std::fill(b.blocks.begin() + std::min(o.dirty_begin, 16384u),
                  b.blocks.begin() + std::min(o.dirty_end, 16384u), Bridge::kGpuNewer);
    }
    o.dirty_begin = ~0u;
    o.dirty_end = 0;
}

/* Before the CPU reads or writes local memory [begin, end): what the GPU drew
   there comes back to it; written, the GPU images there must reload. */
void ApplyPatch(Bridge& b);

void Sync(Bridge& b, u32 begin, u32 end, bool write, const Owned* except = nullptr) {
    ++g_mutations;
    ApplyPatch(b);
    ForNewer(b, [&](Owned& o) {
        if (&o != except && o.dirty_begin < end && begin < o.dirty_end) WriteBack(b, o);
    });
    {
        Timed timed(Section::GpuSync);
        const TimelineStep step("sync: memory for the CPU");
        Coherent(b, begin, end, false);
    }
    WriteBackRead(begin, end);
    if (write) WriteBackLost(begin, end);
    if (write) {
        std::fill(b.blocks.begin() + begin, b.blocks.begin() + std::min(end, 16384u), Bridge::kCpuNewer);
        std::fill(b.block_stamp.begin() + begin, b.block_stamp.begin() + std::min(end, 16384u), ++b.stamp);
    }
    if (write)
        ForOverlapping(b, begin, end, [&](Owned& o) {
            if (&o == except) return;
            if (end <= o.begin || o.full_end <= begin) return;
            if (begin < o.end) {
                MarkStale(o, 1, begin, end);
            } else {
                o.stale_beyond = true;
            }
        });
    if (write && begin < end)
        for (u32 page = begin >> 5; page <= ((end - 1) >> 5) && page < 512; ++page)
            for (Cached* t : b.texture_pages[page])
                if (t->begin < end && begin < t->end) t->valid = false;
}

/* An image transfer about to write blocks [begin, end) (Gs::on_write): the
   blocks it writes in part come back from the GPU first, those it writes
   whole need nothing of what was there - a read-back waits for all the GPU
   has queued, and a race's texture uploads into blocks the GPU drew over
   cost a wait each (1,400 of ~6 ms in a minute of racing, 4 KB each). */
void SyncWrite(Bridge& b, u32 begin, u32 end, const u8* cover) {
    ++g_mutations;
    ApplyPatch(b);                          // one before it, if it never finished
    u32 first = end, last = begin;          // the blocks written
    for (u32 i = begin; i < end; ++i)
        if (cover[i - begin]) {
            first = std::min(first, i);
            last = i + 1;
        }
    if (first >= last) return;
    WriteBackRead(first, last);             // (written in part: read; a measure's upper bound)
    // What the GPU images drew over them into the GPU's memory first (on the
    // GPU, nothing waiting): written back later, it would land over these.
    ForNewer(b, [&](Owned& o) {
        if (o.dirty_begin < last && first < o.dirty_end) WriteBack(b, o);
    });
    // Blocks written in part where the GPU's are newer: the transfer's pixels
    // patched into the GPU's when it is done, the blocks still the GPU's.
    b.patch.transfer = gs::TheGs().last_transfer;
    for (u32 i = first; i < last; ++i)
        if (cover[i - begin] == 1 && b.blocks[i] == Bridge::kGpuNewer) b.patch.blocks.push_back(i);
    for (u32 i = first; i < last; ++i)
        if (cover[i - begin] && !(cover[i - begin] == 1 && b.blocks[i] == Bridge::kGpuNewer)) b.blocks[i] = Bridge::kCpuNewer;
    ++b.stamp;
    for (u32 i = first; i < last && i < 16384; ++i)
        if (cover[i - begin]) b.block_stamp[i] = b.stamp;
    ForOverlapping(b, first, last, [&](Owned& o) {
        if (last <= o.begin || o.full_end <= first) return;
        if (first < o.end) MarkStale(o, 1, first, last);
        else o.stale_beyond = true;
    });
    for (u32 page = first >> 5; page <= ((last - 1) >> 5) && page < 512; ++page)
        for (Cached* t : b.texture_pages[page])
            if (t->begin < last && first < t->end) t->valid = false;
}

/* The pending patch (Bridge::Patch): each pixel of the transfer's rectangle
   that lies in a block it waits for, its bytes in its word (the CPU's copy
   has them, the transfer written) under the pixel's mask - 32, 24 or 16
   bits, a byte, a nibble - into the GPU's memory. */
void ApplyPatch(Bridge& b) {
    if (b.patch.blocks.empty()) return;
    const gs::Gs::TouchSeen t = b.patch.transfer;
    std::vector<u32> blocks;
    blocks.swap(b.patch.blocks);
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    u32 bytes = 0, base_mask = 0;
    switch (t.psm) {
        case 0x00: case 0x30: bytes = 4; base_mask = 0xFFFFFFFFu; break;          // CT32, Z32
        case 0x01: case 0x31: bytes = 3; base_mask = 0x00FFFFFFu; break;          // CT24, Z24
        case 0x02: case 0x0A: case 0x32: case 0x3A: bytes = 2; base_mask = 0xFFFFu; break;
        case 0x13: bytes = 1; base_mask = 0xFFu; break;                           // T8
        case 0x14: bytes = 0; base_mask = 0xFu; break;                            // T4
        default: return;                    // (Touch sends no others here)
    }
    const u8* memory = gs::TheGs().memory.data();
    std::map<u32, u32> masks;               // word -> its mask
    for (u32 y = t.y; y < t.y + t.h; ++y)
        for (u32 x = t.x; x < t.x + t.w; ++x) {
            u32 nibble = 0;
            const u32 address = gs::Memory::Address(t.bp, t.bw, x % 2048, y % 2048, t.psm, &nibble);
            if (!std::binary_search(blocks.begin(), blocks.end(), address / 256)) continue;
            const u32 shift = bytes ? (address % 4) * 8 : (address % 4) * 8 + 4 * nibble;
            masks[address / 4] |= base_mask << shift;
        }
    std::vector<u32> words;
    words.reserve(masks.size() * 3);
    for (const auto& [word, mask] : masks) {
        u32 value;
        std::memcpy(&value, memory + std::size_t(word) * 4, 4);
        words.insert(words.end(), {word, value, mask});
    }
    b.renderer->MemoryPatch(words.data(), masks.size());
    b.patched_words += masks.size();
    ++b.patches;
}

/* A target or depth buffer about to be drawn into or sampled: up to date with
   memory and whatever else the GPU drew over its blocks. */
Owned& Use(Bridge& b, u32 fbp, u32 fbw, u32 psm, bool depth, u32 width, u32 height, u32 rows) {
    // Tens of thousands of draws a frame go to the same target as the one
    // before: that one, without the map (its elements stay put).
    static Owned* last = nullptr;
    static std::tuple<u32, u32, u32, bool> last_key;
    const auto key = std::make_tuple(fbp, fbw, psm, depth);
    if (!last || key != last_key) {
        last = &b.owned[key];
        last_key = key;
    }
    Owned& o = *last;
    if (!o.width) {
        o.fbp = fbp;
        o.fbw = fbw;
        o.psm = psm;
        o.depth = depth;
        o.width = width;
        o.height = height;
        std::tie(o.begin, o.end) = Blocks(fbp, fbw, psm, rows);
        o.full_end = Blocks(fbp, fbw, psm, height).second;
        o.drawn_end = o.begin;
        o.stale_pages.set();
        IndexOwned(b, o);
    } else {
        const u32 end = Blocks(fbp, fbw, psm, rows).second;
        if (end > o.end) {
            o.end = end;
            IndexOwned(b, o);
            o.checked = 0;
            if (o.stale_beyond) MarkStale(o, 1);     // rows drawn over before it reached them (stale or not)
        }
    }
    if (!o.stale && o.checked == b.newer_epoch) return o;
    ++g_mutations;
    // Another image's drawing over these blocks reaches this one through memory.
    ForNewer(b, [&](Owned& other) {
        if (&other != &o && other.dirty_begin < o.end && o.begin < other.dirty_end) {
            const u32 b0 = other.dirty_begin, b1 = other.dirty_end;
            WriteBack(b, other);
            MarkStale(o, 3, b0, b1, nullptr, Is16(other.psm));
        }
    });
    o.checked = b.newer_epoch;
    if (o.stale) {
        // Loaded from the GPU's local memory, the CPU's writes there first -
        // and its own drawing not yet there: an image goes stale for rows
        // another drew over while newer than memory in rows of its own.
        if (o.gpu_newer) WriteBack(b, o);
        Timed timed(Section::GpuSync);
        const TimelineStep step("sync: an image reloaded");
        ++b.reloads;
        Coherent(b, o.begin, o.full_end, true);
        const u32 r0 = std::min(o.stale_row0, o.height), r1 = std::min(o.stale_row1, o.height);
        {
            const u32 page_rows = Is16(o.psm) ? 64 : 32, row_blocks = std::max(1u, o.fbw) * 32;
            WriteBackRead(o.begin + r0 / page_rows * row_blocks, o.begin + (r1 + page_rows - 1) / page_rows * row_blocks);
        }
        if (gs::TheGs().trace)   // in a capture's trace: an image brought back from memory, and why (Owned::why)
            std::fprintf(gs::TheGs().trace_out, "[gpu] reload %s %X/%u/%02X why %d rows %u-%u\n", depth ? "Z" : "C",
                         unsigned(o.fbp), unsigned(o.fbw), unsigned(o.psm), o.why, unsigned(r0), unsigned(r1));
        // A 32-bit image stale only from a 16-bit view's drawing over it (the
        // race's depth into the frame's alpha): its own colour kept where only
        // alpha changed (ReloadKeepsColour) - not a picture as memory holds it.
        b.renderer->ReloadKeepsColour(o.by16 && !Is16(o.psm));
        // Depth buffers come back whole: the renderer keeps one a target
        // size over the same memory, and a part of one reloaded left the
        // race's garage a few pixels off the software GS.
        if ((r0 == 0 && r1 >= o.height) || depth) {
            if (depth) b.renderer->DepthFromMemory(o.fbp, o.width, o.height, o.fbw, o.psm);
            else b.renderer->TargetFromMemory(o.fbp, o.fbw, o.psm, o.width, o.height);
        } else if (r0 < r1) {
            if (depth) b.renderer->DepthRowsFromMemory(o.fbp, o.width, o.height, o.fbw, o.psm, r0, r1 - r0);
            else b.renderer->TargetRowsFromMemory(o.fbp, o.fbw, o.psm, o.width, o.height, r0, r1 - r0);
        }
        b.renderer->ReloadKeepsColour(false);
        o.stale_row0 = 0;
        o.stale_row1 = ~0u;
        o.stale_pages.reset();
        ++o.reloads[o.why];
        o.stale = o.stale_beyond = false;
    }
    return o;
}

/* The pages of `o` under its pixels [x0, x1) x [y0, y1) (64x32 a page, 64x64
   16-bit). */
Pages PagesOf(const Owned& o, u32 x0, u32 x1, u32 y0, u32 y1) {
    Pages pages;
    const u32 page_rows = Is16(o.psm) ? 64 : 32, fbw = std::max(1u, o.fbw);
    x1 = std::min(x1, fbw * 64);
    x0 = std::min(x0, x1 ? x1 - 1 : 0);
    x1 = std::max(x1, x0 + 1);
    y1 = std::max(y1, y0 + 1);
    for (u32 py = y0 / page_rows; py <= (y1 - 1) / page_rows; ++py)
        for (u32 px = x0 / 64; px <= (x1 - 1) / 64; ++px) {
            const u32 page = o.begin / 32 + py * fbw + px;
            if (page < 512) pages.set(page);
        }
    return pages;
}

/* After the GPU drew into `o`, pixels [x0, x1) x [y0, y1): memory and the
   other images over those blocks are behind. */
void Drew(Bridge& b, Owned& o, u32 x0, u32 x1, u32 y1, u32 y0) {
    const u32 page_rows = Is16(o.psm) ? 64 : 32, row_blocks = std::max(1u, o.fbw) * 32;
    y1 = std::min(std::max(y1, y0 + 1), o.height);
    y0 = std::min(y0, y1 - 1);
    const u32 db = o.begin + (y0 / page_rows) * row_blocks;
    const u32 de = o.begin + ((y1 + page_rows - 1) / page_rows) * row_blocks;
    // Already newer than memory over these pages: the others over them were
    // made stale then (any reloaded since wrote this one back first).
    const u32 fbw = std::max(1u, o.fbw), cx1 = std::min(x1, fbw * 64);
    const u32 cx0 = std::min(x0, cx1 ? cx1 - 1 : 0);
    const u32 r[4] = {cx0 / 64, (std::max(cx1, cx0 + 1) - 1) / 64, y0 / page_rows, (y1 - 1) / page_rows};
    const bool in_rect = o.rect_dirty && r[0] >= o.rect[0] && r[1] <= o.rect[1] && r[2] >= o.rect[2] && r[3] <= o.rect[3];
    Pages pages;
    if (!in_rect) pages = PagesOf(o, x0, x1, y0, y1);
    const bool inside = o.gpu_newer && db >= o.dirty_begin && de <= o.dirty_end &&
                        (in_rect || (pages & ~o.dirty_pages).none());
    if (!o.gpu_newer) {
        b.newer.push_back(&o);
        b.newer_sorted = b.newer.size() < 2;
    }
    o.gpu_newer = true;
    o.x1 = std::max(o.x1, std::min(x1, o.width));
    o.y1 = std::max(o.y1, y1);
    o.drawn_end = std::max(o.drawn_end, Blocks(o.fbp, o.fbw, o.psm, o.y1).second);
    o.end = std::max(o.end, o.drawn_end);
    o.dirty_begin = std::min(o.dirty_begin, db);
    o.dirty_end = std::max(o.dirty_end, de);
    if (o.dirty_end > o.indexed_end || o.end > o.indexed_end || o.drawn_end > o.indexed_end) IndexOwned(b, o);
    if (!in_rect) {
        o.dirty_pages |= pages;
        std::memcpy(o.rect, r, sizeof r);
        o.rect_dirty = true;
    }
    o.drawn_serial = ++b.serial;    // textures over it decoded before this are out of date
    if (inside) return;
    if (in_rect) pages = PagesOf(o, x0, x1, y0, y1);   // (not expected: inside whenever in the rectangle)
    ++g_mutations;
    ++b.newer_epoch;
    ForOverlapping(b, db, de, [&](Owned& other) {
        if (&other == &o || de <= other.begin || other.full_end <= db) return;
        if (db < other.end) {
            MarkStale(other, 2, db, de, &pages, Is16(o.psm));
        } else {
            other.stale_beyond = true;
        }
    });
}

/* The pages of target `o` a primitive sampling it as a TW x TH texture can
   read, by its texture coordinates (a texel more each side, for bilinear
   filtering); false where that is not known - the region modes, a texture
   wider than the buffer, a vertex with Q not above 0. Perspective-correct
   coordinates inside a triangle are weighted means of its vertices' S/Q and
   T/Q (weights the barycentrics times Q, all positive), so they lie between
   the vertices' least and greatest. */
bool SampledPages(const Owned& o, u64 clamp, u32 tw, u32 th, const ps2gpu::GsVertex* v, std::size_t n, Pages* out) {
    const u32 wms = u32(clamp & 3), wmt = u32(clamp >> 2 & 3);
    if (wms > 1 || wmt > 1 || tw > std::max(1u, o.fbw) * 64 || !n) return false;
    float s0 = 0, s1 = 0, t0 = 0, t1 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(v[i].q > 0)) return false;
        const float s = v[i].s / v[i].q, t = v[i].t / v[i].q;
        if (!std::isfinite(s) || !std::isfinite(t)) return false;
        if (i == 0 || s < s0) s0 = s;
        if (i == 0 || s > s1) s1 = s;
        if (i == 0 || t < t0) t0 = t;
        if (i == 0 || t > t1) t1 = t;
    }
    // Texels [lo, hi) of `size`: the whole span where REPEAT would wrap.
    const auto span = [](float c0, float c1, u32 size, u32 mode, u32* lo, u32* hi) {
        const float a = std::floor(c0 * float(size)) - 1, b = std::ceil(c1 * float(size)) + 1;
        if (mode == 0 && (a < 0 || b > float(size))) {
            *lo = 0;
            *hi = size;
            return;
        }
        *lo = u32(std::clamp(a, 0.0f, float(size - 1)));
        *hi = u32(std::clamp(b, float(*lo + 1), float(size)));
    };
    u32 x0, x1, y0, y1;
    span(s0, s1, tw, wms, &x0, &x1);
    span(t0, t1, th, wmt, &y0, &y1);
    *out = PagesOf(o, x0, x1, y0, y1);
    return true;
}

/* Target `o` about to be sampled as a texture, over `pages`: brought up to
   date (Use) only if what it lacks lies there - changed under it since it
   was loaded, or drawn by another image the GPU has not written back. */
void Sample(Bridge& b, Owned& o, const Pages& pages) {
    bool needed = o.stale && (o.stale_pages & pages).any();
    if (!needed)
        ForNewer(b, [&](Owned& other) {
            if (!needed && &other != &o && (other.dirty_pages & pages).any()) needed = true;
        });
    if (needed) Use(b, o.fbp, o.fbw, o.psm, false, o.width, o.height, 1);   // sampling grows nothing
}

/* A 64-bit hash of `size` bytes (a multiple of 8). Four independent lanes,
   folded at the end: one chain of multiplies waits on each before it, and
   textures are hashed hundreds of thousands of times a run (only compared
   within a run, so the values themselves are free to change). */
u64 Hash(const unsigned char* data, std::size_t size) {
    constexpr u64 kMul = 0xFF51AFD7ED558CCDull;
    u64 h[4] = {0x9E3779B97F4A7C15ull ^ size, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull};
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32)
        for (int lane = 0; lane < 4; ++lane) {
            u64 word;
            std::memcpy(&word, data + i + lane * 8, 8);
            h[lane] = (h[lane] ^ word) * kMul;
            h[lane] ^= h[lane] >> 29;
        }
    for (int lane = 0; i + 8 <= size; i += 8, ++lane) {
        u64 word;
        std::memcpy(&word, data + i, 8);
        h[lane] = (h[lane] ^ word) * kMul;
        h[lane] ^= h[lane] >> 29;
    }
    u64 out = h[0];
    for (int lane = 1; lane < 4; ++lane) {
        out = (out ^ h[lane]) * kMul;
        out ^= out >> 29;
    }
    return out;
}

bool GpuHolds(const Bridge& b, u32 begin, u32 end);

bool Palettized(u32 psm) {
    return psm == gs::PSMT8 || psm == gs::PSMT4 || psm == gs::PSMT8H || psm == gs::PSMT4HL || psm == gs::PSMT4HH;
}

/* The CLUT rows a palettized texture reads: all 16 for 8-bit, CSA's for 4-bit. */
u32 ClutRowsOf(u32 psm, u32 csa) {
    if (!Palettized(psm)) return 0;
    return (psm == gs::PSMT8 || psm == gs::PSMT8H) ? 0xFFFFu : 1u << (csa % 16);
}

/* The CLUT rows a palettized texture reads, as a hash: their entries where
   the GS model holds them, their generation where only the GPU does. */
u64 ClutHash(const Bridge& b, u32 psm, u32 csa) {
    const u32 rows = ClutRowsOf(psm, csa);
    if (!rows) return 0;
    // Hashed once a load: hundreds of thousands of palettized draws a frame.
    static u64 loads = ~0ull, row_hash[16];
    if (gs::TheGs().clut_loads() != loads) {
        loads = gs::TheGs().clut_loads();
        const auto& clut = gs::TheGs().Clut();
        for (u32 r = 0; r < 16; ++r) {
            u64 hash = 1469598103934665603ull;
            for (u32 i = 0; i < 16; ++i) hash = (hash ^ clut[r * 16 + i]) * 1099511628211ull;
            row_hash[r] = hash;
        }
    }
    u64 hash = 0x9E3779B97F4A7C15ull;
    for (u32 r = 0; r < 16; ++r) {
        if (!(rows >> r & 1)) continue;
        const u64 row = (b.clut_cpu >> r & 1) ? row_hash[r] : ~(b.clut_generation[r] * 0xC2B2AE3D27D4EB4Full);
        hash = (hash ^ row ^ r) * 0xFF51AFD7ED558CCDull;
    }
    return hash;
}

/* A CLUT load (on_clut_load): from the GPU's copy of local memory when the
   GPU holds its blocks newer - the race's crash and glow scenes load CLUTs
   from images just drawn, and bringing those back waited ~8 ms each - else
   left to the GS model. */
bool LoadClut(Bridge& b, u64 tex0, u64 texa) {
    ++g_mutations;
    const u32 psm = Bits(tex0, 20, 6), cbp = Bits(tex0, 37, 14), cpsm = Bits(tex0, 51, 4);
    // CSM2 (bit 55) loads from TEXCLUT's row, at entry 0 whatever CSA: the GS
    // model's load does it (the GPU's knows CSM1 only).
    const bool csm2 = Bits(tex0, 55, 1);
    const u32 csa = csm2 ? 0 : Bits(tex0, 56, 5);
    const u32 rows = ClutRowsOf(psm, csa);
    if (csm2 || !GpuHolds(b, cbp, cbp + 8)) {
        b.clut_cpu |= rows;
        b.clut_gpu &= ~rows;
        return false;
    }
    // Its source (Bridge::texture_cache): the settings, the blocks' last
    // CPU write, the last draw over them.
    u64 source = 0;
    bool same = false;
    if (b.texture_cache != Bridge::TextureCache::kOff) {
        u64 written = 0, drawn = 0;
        for (u32 i = cbp; i < cbp + 8 && i < 16384; ++i) written = std::max(written, b.block_stamp[i]);
        ForOverlapping(b, cbp, cbp + 8, [&](Owned& o) {
            if (o.begin < cbp + 8 && cbp < std::max(o.drawn_end, o.dirty_end)) drawn = std::max(drawn, o.drawn_serial);
        });
        const u64 parts[] = {cbp, cpsm, csa, u64(psm == gs::PSMT8 || psm == gs::PSMT8H), texa & 0x80FF,
                             Bits(texa, 32, 8), written, drawn};
        source = 1469598103934665603ull;
        for (u64 part : parts) source = (source ^ part) * 1099511628211ull;
        source |= 1;
        same = true;
        for (u32 r = 0; r < 16; ++r)
            if ((rows >> r & 1) && b.gpu_clut_source[r] != source) same = false;
        if (same && b.texture_cache == Bridge::TextureCache::kOn) {
            b.clut_gpu |= rows;             // the GPU's rows hold it still
            b.clut_cpu &= ~rows;
            ++b.clut_loads_kept;
            return true;
        }
    }
    Timed timed(Section::GpuSync);
    const TimelineStep step("sync: a CLUT load");
    // The check: the GPU's CLUT before a load the cache would have kept.
    u32 before[256] = {};
    const bool check = same && b.renderer->ReadClut(before);
    ForNewer(b, [&](Owned& o) {
        if (o.dirty_begin < cbp + 8 && cbp < o.dirty_end) WriteBack(b, o);
    });
    Coherent(b, cbp, cbp + 8, true);
    WriteBackRead(cbp, cbp + 8);
    ps2gpu::GsClutLoad load;
    load.cbp = cbp;
    load.cpsm = cpsm;
    load.csa = csa;
    load.eight = psm == gs::PSMT8 || psm == gs::PSMT8H;
    load.texa_lo = u32(texa & 0x80FF);
    load.ta1 = Bits(texa, 32, 8);
    b.renderer->ClutFromMemory(load);
    if (check) {
        u32 after[256] = {};
        if (b.renderer->ReadClut(after)) {
            ++b.clut_checks;
            bool differs = false;
            for (u32 r = 0; r < 16; ++r)
                if ((rows >> r & 1) && std::memcmp(before + r * 16, after + r * 16, 64) != 0) differs = true;
            if (differs && b.clut_check_differences++ < 20)
                std::fprintf(stderr, "[texture cache] CHECK: the CLUT from CBP %X (CPSM %u, CSA %u) changed though "
                             "its source did not: the cache would have shown it stale\n",
                             unsigned(cbp), unsigned(cpsm), unsigned(csa));
        }
    }
    b.clut_gpu |= rows;
    b.clut_cpu &= ~rows;
    for (u32 r = 0; r < 16; ++r)
        if (rows >> r & 1) {
            ++b.clut_generation[r];
            b.gpu_clut_source[r] = source;
        }
    ++b.gpu_clut_loads;
    return true;
}

/* The renderer's key for the texture the current primitive samples, decoded if
   need be (--gpu). */
/* Whether the GPU holds blocks [begin, end) newer than the CPU: an image drawn
   over them and not yet swizzled into GPU memory, or GPU memory newer. */
bool GpuHolds(const Bridge& b, u32 begin, u32 end) {
    bool newer = false;
    ForNewer(b, [&](Owned& o) {
        if (o.dirty_begin < end && begin < o.dirty_end) newer = true;
    });
    if (newer) return true;
    end = std::min(end, 16384u);
    for (u32 i = begin; i < end; ++i)
        if (b.blocks[i] == Bridge::kGpuNewer) return true;
    return false;
}

/* The renderer's key for the texture the current primitive samples, decoded if
   need be: by the CPU from its copy of local memory, or - when the GPU holds
   the texture's blocks newer (the race reads its own targets' alpha as
   PSMT8H textures, its depth buffer as PSMZ16) - by the GPU from its own,
   rather than wait for them to come back. */
void WriteBmp(const std::string& path, const std::vector<u32>& rgba, u32 stride, u32 w, u32 h);

// A texture pack's replacement for the texture, uploaded in its place if its
// file has been read (texture_pack::Find); false if not (yet).
bool Replace(Bridge& b, Cached& t) {
    bool pending = false;
    const std::shared_ptr<const texture_pack::Image> image = texture_pack::Find(t.pack_name, &pending);
    t.pack_pending = pending;
    if (!image) return false;
    // One image on the GPU for every texture it replaces.
    if (!b.renderer->HasReplacement(image->id) &&
        !b.renderer->UploadReplacement(image->id, image->format, image->width, image->height, image->levels,
                                       image->data.data(), image->data.size()))
        return false;
    ++g_mutations;
    b.renderer->UseReplacement(t.renderer_key, image->id);
    t.replaced = true;
    t.replacement = image->id;
    t.decoded_serial = b.serial;
    ++b.replacements;
    return true;
}

u64 CachedTexture(Bridge& b, const gs::Context& c, u64 texa, u32 tw, u32 th) {
    const u32 tbp = Bits(c.tex0, 0, 14), tbw = Bits(c.tex0, 14, 6), psm = Bits(c.tex0, 20, 6);
    // What decides the decoded texels: where and how it is stored (TBP0, TBW,
    // PSM, TW, TH), how a region mode wraps it (CLAMP), TEXA, the CLUT's
    // contents (wherever it was loaded from).
    u64 key = c.tex0 & 0x3FFFFFFFFull;
    if (Bits(c.clamp, 0, 2) >= 2 || Bits(c.clamp, 2, 2) >= 2) key ^= c.clamp * 0xC2B2AE3D27D4EB4Full;
    key ^= texa * 0x165667B19E3779F9ull;
    const u32 clut_rows = ClutRowsOf(psm, Bits(c.tex0, 56, 5));
    key ^= ClutHash(b, psm, Bits(c.tex0, 56, 5)) * 0x27D4EB2F165667C5ull;
    gs::MipLevels mips = gs::LevelsOf(c);
    if (!gs::MipMode(c)) mips.count = 1;        // levels never read: not decoded
    for (u32 n = 1; n < mips.count; ++n) key ^= (u64(mips.tbp[n]) << 6 | mips.tbw[n]) * (0x9E3779B97F4A7C15ull + n);
    Cached& t = b.textures[key];
    if (!t.renderer_key) {
        t.renderer_key = b.next_texture++;
        b.texture_lru.push_front(key);
        t.lru = b.texture_lru.begin();
        ++g_mutations;
        // Every level's blocks (they follow level 0, as a rule).
        t.begin = 16384;
        t.end = 0;
        for (u32 n = 0; n < mips.count; ++n) {
            const u32 h = std::max(1u, th >> n);
            t.begin = std::min(t.begin, mips.tbp[n]);
            t.end = std::max(t.end, std::min(16384u, mips.tbp[n] + std::max(1u, std::max(1u, mips.tbw[n]) * 64 * h *
                                                                                    PixelBits(psm) / 8 / 256 + 32)));
        }
        b.IndexTexture(&t, true);
    } else {
        b.texture_lru.splice(b.texture_lru.begin(), b.texture_lru, t.lru);
    }
    // The GPU drawing over its blocks since it was decoded makes it out of date
    // as surely as memory written there.
    if (t.owned_seen != b.owned.size()) {
        t.over.clear();
        for (const auto& [okey, o] : b.owned)
            if (o.begin < t.end && t.begin < o.full_end) t.over.push_back(&o);
        t.owned_seen = b.owned.size();
    }
    for (const Owned* o : t.over)
        if (o->drawn_serial > t.decoded_serial && o->begin < t.end && t.begin < o->drawn_end) t.valid = false;
    // A texture pack (texture_pack.h), for textures from memory - not ones the
    // GPU drew, as PCSX2 replaces none of those. Turned on or off (Tab):
    // every texture comes back once, replaced or as the game has it.
    const bool pack = texture_pack::Count() != 0 && texture_pack::Enabled();
    if (t.pack_generation != texture_pack::Generation()) {
        t.pack_generation = texture_pack::Generation();
        if (pack || t.replaced) t.valid = t.decoded = false;
    }
    if (t.valid && t.pack_pending && pack) Replace(b, t);       // its file read since
    // Its replacement gone from the GPU (not drawn for a while): again.
    if (t.replaced && !b.renderer->HasReplacement(t.replacement)) t.valid = t.decoded = false;
    // Decoded on the GPU, from its copy of local memory and its CLUT, the
    // CPU's newer blocks and CLUT rows sent over first.
    bool gpu_holds_now = false;
    auto gpu_decode = [&] {
        ForNewer(b, [&](Owned& o) {
            if (o.dirty_begin < t.end && t.begin < o.dirty_end) WriteBack(b, o);
        });
        Coherent(b, t.begin, t.end, true);
        WriteBackRead(t.begin, t.end);
        ps2gpu::GsTextureDecode d;
        d.tbp = tbp;
        d.tbw = tbw;
        d.psm = psm;
        d.tw = tw;
        d.th = th;
        d.wms = Bits(c.clamp, 0, 2);
        d.wmt = Bits(c.clamp, 2, 2);
        d.minu = Bits(c.clamp, 4, 10);
        d.maxu = Bits(c.clamp, 14, 10);
        d.minv = Bits(c.clamp, 24, 10);
        d.maxv = Bits(c.clamp, 34, 10);
        d.texa_lo = u32(texa & 0x80FF);
        d.ta1 = Bits(texa, 32, 8);
        d.csa = Bits(c.tex0, 56, 5);
        d.levels = mips.count;
        for (u32 n = 0; n < mips.count; ++n) {
            d.level_tbp[n] = mips.tbp[n];
            d.level_tbw[n] = mips.tbw[n];
        }
        // Through the GPU's CLUT, its rows the GPU lacks sent over first.
        d.clut = nullptr;
        if (clut_rows) {
            const u32 sent = b.clut_cpu & ~b.clut_gpu;
            b.renderer->ClutRows(gs::TheGs().Clut().data(), sent);
            for (u32 r = 0; r < 16; ++r)
                if (sent >> r & 1) b.gpu_clut_source[r] = 0;
            b.clut_gpu |= b.clut_cpu;
        }
        ++g_mutations;
        b.renderer->TextureFromMemory(t.renderer_key, d);
        // PS2HOST_GPU_TEXDUMP=TBP (hex): each decode of a texture there, as the
        // GPU has it, to gpu_tex_<n>_<primitive>.bmp (the CPU's beside it as
        // cpu_tex_<n>.bmp) - what a draw sampled, against the GS model.
        // (getenv once: a texture decoded is common, and getenv takes a lock.)
        static const char* const texdump = std::getenv("PS2HOST_GPU_TEXDUMP");
        if (const char* want = texdump) {
            static int dumped = 0;
            if (tbp == u32(std::strtoul(want, nullptr, 16)) && dumped < 64) {
                std::vector<u32> pixels;
                u32 w = 0, h = 0;
                if (b.renderer->ReadTexture(t.renderer_key, &pixels, &w, &h)) {
                    const std::string n = std::to_string(dumped) + "_" + std::to_string(gs::TheGs().primitives());
                    if (std::FILE* raw = std::fopen(("gpu_tex_" + n + ".rgba").c_str(), "wb")) { std::fwrite(pixels.data(), 4, pixels.size(), raw); std::fclose(raw); }
                    std::fprintf(stderr, "[texdump] %s: %ux%u psm %02X, %s\n", n.c_str(), unsigned(w), unsigned(h),
                                 unsigned(psm), gpu_holds_now ? "from the GPU's memory" : "from the CPU's");
                }
                ++dumped;
            }
        }
        static const bool counting = std::getenv("PS2HOST_GPU_COUNT") != nullptr;
        if (counting) {
            char key[96];
            std::snprintf(key, sizeof key, "psm %02X tbp %04X tbw %u %ux%u clutrows %d", unsigned(psm), unsigned(tbp),
                          unsigned(tbw), unsigned(tw), unsigned(th), int((clut_rows & ~b.clut_cpu) != 0));
            ++g_decode_kinds[key];
        }
        t.decoded_serial = b.serial;
        ++b.gpu_decodes;
    };
    // The GPU has newer blocks, or the CLUT rows it reads are only there:
    // decoded there without a look at the CPU's copy.
    if (!t.valid && (GpuHolds(b, t.begin, t.end) || (clut_rows & ~b.clut_cpu))) {
        Timed timed(Section::GpuSync);
        const TimelineStep step("sync: a texture decoded on the GPU");
        gpu_holds_now = true;
        gpu_decode();
        t.replaced = t.pack_pending = false;
        t.valid = true;
        t.decoded = false;              // the CPU's copy was not read: no hash to compare
    }
    if (!t.valid) {
        Timed timed(Section::GpuSync);
        const TimelineStep step("sync: a texture from the CPU");
        b.cause = 0;
        const u64 before = b.downloads;
        Sync(b, t.begin, t.end, false);
        if (b.downloads != before) b.texture_downloads[psm << 16 | tbw] += b.downloads - before;
        // Games send the same texels again and again (Burnout 3 its textures
        // every frame, through PATH3): decoded again only if its blocks hold
        // something else than when it was last decoded.
        const u64 content = Hash(gs::TheGs().memory.data() + std::size_t(t.begin) * 256,
                                 std::size_t(t.end - t.begin) * 256);
        // Something else: decoded on the GPU all the same (texel by texel on
        // the CPU, the race's streamed textures were a twelfth of the GS
        // thread), the hash kept to compare the next time.
        if (!t.decoded || content != t.content) {
            t.replaced = t.pack_pending = false;
            if (pack)
                t.pack_name = texture_pack::NameOf(gs::TheGs().memory, c.tex0, c.clamp, texa,
                                                   gs::TheGs().Clut().data());
            if (!pack || !Replace(b, t)) gpu_decode();
            t.content = content;
            ++b.decodes;
        } else {
            ++b.unchanged;
            t.decoded_serial = b.serial;
        }
        t.valid = true;
        t.decoded = true;
    }
    const u64 renderer_key = t.renderer_key;
    // Keep the cache bounded: the least recently used go (4,096: a race keeps
    // over 2,048 about, and each one evicted came back decoded again).
    while (b.textures.size() > 4096) {
        const u64 old = b.texture_lru.back();
        b.texture_lru.pop_back();
        auto found = b.textures.find(old);
        if (found != b.textures.end()) {
            ++g_mutations;
            b.renderer->ReleaseTexture(found->second.renderer_key);
            b.IndexTexture(&found->second, false);
            b.textures.erase(found);
        }
    }
    (void)tw;
    return renderer_key;
}

// ------------------------------------------------------------------ both

ps2gpu::GsVertex Convert(const gs::Vertex& v, bool fst, u32 tw, u32 th) {
    ps2gpu::GsVertex out{};
    out.x = v.x;
    out.y = v.y;
    out.z = v.z;                    // clamped to the depth format by the caller
    out.r = v.r;
    out.g = v.g;
    out.b = v.b;
    out.a = v.a;
    out.fog = v.fog;
    if (fst) {
        // TW and TH are powers of two: dividing by 16 and by them is scaling
        // by a power of two, which the reciprocal does exactly.
        out.s = float(v.u) * (1.0f / float(16 * tw));
        out.t = float(v.v) * (1.0f / float(16 * th));
        out.q = 1.0f;
    } else {
        out.s = v.s;
        out.t = v.t;
        out.q = v.q;
    }
    return out;
}

/* A primitive as triangles: sprites as two (the second vertex's colour), lines
   and points as one pixel wide quads. */
std::size_t Triangles(const gs::Gs::DrawCall& call, bool fst, u32 tw, u32 th, ps2gpu::GsVertex* v, bool* gouraud) {
    const u32 type = u32(call.prim & 7);
    auto quad = [&](ps2gpu::GsVertex a, ps2gpu::GsVertex b) {
        ps2gpu::GsVertex ab = b, ba = b;
        ab.x = b.x; ab.y = a.y; ab.t = a.t;
        ba.x = a.x; ba.y = b.y; ba.s = a.s;
        a.r = b.r; a.g = b.g; a.b = b.b; a.a = b.a; a.fog = b.fog; a.z = b.z;   // flat: the second vertex's, Z too
        v[0] = a; v[1] = ab; v[2] = ba; v[3] = ab; v[4] = b; v[5] = ba;
        *gouraud = true;
        return std::size_t(6);
    };
    if (type == 6 && call.count >= 2)
        return quad(Convert(call.vertices[0], fst, tw, th), Convert(call.vertices[1], fst, tw, th));
    // Lines and points as the software GS draws them: a pixel per step along
    // the major axis, at the line's position rounded, both ends included. A
    // pixel n is sampled at n, so the band [p - 0.5, p + 0.5) across the line
    // holds exactly the rounded pixel, and the ends reach half a pixel past.
    if (type == 0 && call.count >= 1) {             // a point: its pixel
        ps2gpu::GsVertex a = Convert(call.vertices[call.count - 1], fst, tw, th), b = a;
        a.x -= 0.5f;
        a.y -= 0.5f;
        b.x += 0.5f;
        b.y += 0.5f;
        return quad(a, b);
    }
    if ((type == 1 || type == 2) && call.count >= 2) {   // a line: a band one pixel across
        ps2gpu::GsVertex a = Convert(call.vertices[0], fst, tw, th), b = Convert(call.vertices[1], fst, tw, th);
        const bool flat = std::fabs(b.x - a.x) >= std::fabs(b.y - a.y);
        float& a_major = flat ? a.x : a.y;
        float& b_major = flat ? b.x : b.y;
        const float reach = b_major >= a_major ? 0.5f : -0.5f;
        a_major -= reach;
        b_major += reach;
        ps2gpu::GsVertex a0 = a, a1 = a, b0 = b, b1 = b;
        if (flat) { a0.y -= 0.5f; b0.y -= 0.5f; a1.y += 0.5f; b1.y += 0.5f; }
        else { a0.x -= 0.5f; b0.x -= 0.5f; a1.x += 0.5f; b1.x += 0.5f; }
        v[0] = a0; v[1] = b0; v[2] = a1; v[3] = b0; v[4] = b1; v[5] = a1;
        return 6;
    }
    if (call.count >= 3) {
        for (int i = 0; i < 3; ++i) v[i] = Convert(call.vertices[i], fst, tw, th);
        return 3;
    }
    return 0;
}

void WriteBmp(const std::string& path, const std::vector<u32>& rgba, u32 stride, u32 w, u32 h);

/* PS2HOST_GPU_DUMP_FRAME: the target drawing just left, as it stands. */
void DumpPass(Bridge& s, const std::tuple<u32, u32, u32>& target) {
    const auto [fbp, fbw, psm] = target;
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    if (!s.renderer->ReadTarget(fbp, fbw, psm, &pixels, &w, &h)) return;
    char name[96];
    std::snprintf(name, sizeof name, "/pass_%03u_%X_%u_%02X.bmp", unsigned(s.pass), unsigned(fbp), unsigned(fbw),
                  unsigned(psm));
    WriteBmp(s.dump_passes + name, pixels, w, std::min(w, 640u), std::min(h, 512u));
}

void OnDraw(const gs::Gs::DrawCall& call) {
    // Timed only past the fast path below: two clock reads a primitive were
    // a twentieth of the GS thread.
    std::optional<Timed> timed;
    Bridge& s = *g_bridge;
    const u32 type = u32(call.prim & 7);
    if (type == 7) return;
    if (!s.primary && type < 3) return;                 // the shadow compares what it has always drawn
    const gs::Context& c = *call.context;
    ps2gpu::GsDrawState state;
    state.fbp = Bits(c.frame, 0, 9);
    state.fbw = std::max(1u, Bits(c.frame, 16, 6));
    state.psm = Bits(c.frame, 24, 6);
    state.fbmsk = u32(c.frame >> 32);
    state.width = std::min(2048u, state.fbw * 64);
    // The rows a page holds twice over in a 16-bit format: a 16-bit view of a
    // 448-row picture is 896 rows (Burnout 3 writes its depth into the
    // picture's alpha through one - at 512 rows, the lower half went unwritten,
    // its haze and the sun's halo with it).
    state.height = Is16(state.psm) ? 1024 : 512;
    state.scissor_x0 = int(Bits(c.scissor, 0, 11));
    state.scissor_x1 = int(Bits(c.scissor, 16, 11));
    state.scissor_y0 = int(Bits(c.scissor, 32, 11));
    state.scissor_y1 = int(Bits(c.scissor, 48, 11));
    state.gouraud = (call.prim >> 3) & 1;
    state.textured = (call.prim >> 4) & 1;
    state.sprite = type == 6 && s.renderer->Scale() > 1;     // (at 1x it would only split batches)
    state.sprite_prim = type == 6;
    const bool fst = (call.prim >> 8) & 1;
    const u32 tw = 1u << std::min(Bits(c.tex0, 26, 4), 10u), th = 1u << std::min(Bits(c.tex0, 30, 4), 10u);
    // The primitive as triangles, and the pixels it can reach.
    ps2gpu::GsVertex v[6];
    bool gouraud = state.gouraud;
    const std::size_t n = Triangles(call, fst, tw, th, v, &gouraud);
    state.gouraud = gouraud;
    if (!n) return;
    u32 x0 = state.width, x1 = 0, y1 = 0, y0 = state.height;
    for (std::size_t i = 0; i < n; ++i) {
        x0 = std::min(x0, u32(std::clamp(std::floor(v[i].x), 0.0f, float(state.width))));
        // Exclusive: the GS draws the pixels whose centres lie before the far edge.
        x1 = std::max(x1, u32(std::clamp(std::ceil(v[i].x), 0.0f, float(state.width))));
        y1 = std::max(y1, u32(std::clamp(std::ceil(v[i].y), 0.0f, float(state.height))));
        y0 = std::min(y0, u32(std::clamp(std::floor(v[i].y), 0.0f, float(state.height))));
    }
    y0 = std::max(y0, u32(std::max(state.scissor_y0, 0)));
    x0 = std::max(x0, u32(std::max(state.scissor_x0, 0)));
    x1 = std::min(x1, u32(state.scissor_x1 + 1));
    y1 = std::min(y1, u32(state.scissor_y1 + 1));
    if (s.primary) {
        // Wholly outside the scissor (or the buffer): nothing to draw, and no
        // image to bring up to date for it. Burnout 3's garage sends thousands
        // a frame to FRAME 0, FBW 1, whose colour and Z alias the same page.
        float lo_x = v[0].x, lo_y = v[0].y, hi_x = v[0].x, hi_y = v[0].y;
        for (std::size_t i = 1; i < n; ++i) {
            lo_x = std::min(lo_x, v[i].x);
            lo_y = std::min(lo_y, v[i].y);
            hi_x = std::max(hi_x, v[i].x);
            hi_y = std::max(hi_y, v[i].y);
        }
        const float left = float(std::max(state.scissor_x0, 0)), top = float(std::max(state.scissor_y0, 0));
        const float right = float(std::min(state.scissor_x1, int(state.width) - 1));
        const float bottom = float(std::min(state.scissor_y1, int(state.height) - 1));
        if (std::floor(hi_x) < left || std::ceil(lo_x) > right || std::floor(hi_y) < top || std::ceil(lo_y) > bottom)
            return;
        if (Bits(c.test, 16, 1) && Bits(c.test, 17, 2) == 0) return;    // ZTST NEVER: every pixel fails
    }
    const u32 rows = std::max(1u, y1);
    // The registers of the primitive before, and nothing changed since: its
    // state, target and depth buffer again (most of a frame's tens of
    // thousands of primitives; the lookups and checks were half the GS
    // thread's time).
    struct Fast {
        bool valid = false;
        u64 mutations = 0;
        const gs::Context* context = nullptr;
        u64 regs[12] = {};
        u32 rows = 0, zmax = 0;
        ps2gpu::GsDrawState state;
        Owned* target = nullptr;
        Owned* depth = nullptr;
        Owned* source = nullptr;    // a target sampled as the texture: fast only while it is not stale
        bool draws_colour = false;
    };
    static Fast fast;
    const u64 regs[12] = {call.prim, c.frame, c.scissor, c.tex0, c.tex1, c.clamp, c.alpha, c.test, c.zbuf,
                          call.texa, call.fogcol | (u64(call.pabe) << 63), c.miptbp1 ^ (c.miptbp2 * 31)};
    if (s.primary && fast.valid && fast.mutations == g_mutations && fast.context == call.context &&
        rows <= fast.rows && std::memcmp(regs, fast.regs, sizeof regs) == 0 && !s.Dumping() &&
        (!fast.source || !fast.source->stale)) {
        ps2gpu::GsDrawState st = fast.state;
        st.gouraud = gouraud;
        for (std::size_t i = 0; i < n; ++i) v[i].z = std::min(v[i].z, fast.zmax);
        s.renderer->DrawTriangles(st, v, n);
        if (fast.draws_colour) Drew(s, *fast.target, x0, x1, y1, y0);
        if (fast.depth) Drew(s, *fast.depth, x0, x1, y1, y0);
        return;
    }
    fast.valid = false;
    timed.emplace(Section::GpuBridge);
    if (s.primary && s.Dumping()) {
        const auto key = std::make_tuple(state.fbp, state.fbw, state.psm);
        if (key != s.last_target) {
            if (std::get<0>(s.last_target) != ~0u) DumpPass(s, s.last_target);
            ++s.pass;
            std::fprintf(stderr, "[gpu-pass] %03u: frame %X/%u/%02X fbmsk %08X; prim %llX tex0 %llX clamp %llX; "
                                 "alpha %llX test %llX zbuf %llX\n",
                         unsigned(s.pass), unsigned(state.fbp), unsigned(state.fbw), unsigned(state.psm),
                         unsigned(state.fbmsk), static_cast<unsigned long long>(call.prim),
                         static_cast<unsigned long long>(c.tex0), static_cast<unsigned long long>(c.clamp),
                         static_cast<unsigned long long>(c.alpha), static_cast<unsigned long long>(c.test),
                         static_cast<unsigned long long>(c.zbuf));
            s.last_target = key;
        }
    }
    Owned* target = nullptr;
    Owned* tex_source = nullptr;
    if (s.primary) target = &Use(s, state.fbp, state.fbw, state.psm, false, state.width, state.height, rows);
    if (state.textured) {
        const u32 tex_psm = Bits(c.tex0, 20, 6), tbp_blocks = Bits(c.tex0, 0, 14);
        // A target read as PSMT8H (Burnout 3's glow and blur passes, ~900 a
        // picture in a race) indexes the GPU's CLUT by its alpha byte in the
        // shader (4): it was swizzled into GPU memory and decoded again each
        // time.
        const u32 tex_format = tex_psm == gs::PSMCT32 ? 0 : tex_psm == gs::PSMCT24 ? 1
                             : (tex_psm == gs::PSMCT16 || tex_psm == gs::PSMCT16S) ? 2
                             : (tex_psm == gs::PSMT8H && s.primary) ? 4 : 3;
        state.tw = tw;
        state.th = th;
        // A texture that is a target the GPU has drawn is sampled from it
        // (render to texture).
        if ((s.primary || s.targets_as_textures) && tex_format != 3 && tbp_blocks % 32 == 0) {
            const u32 candidates[2] = {tex_format == 2 ? tex_psm : gs::PSMCT32,
                                       tex_format == 2 ? tex_psm : tex_format == 4 ? gs::PSMCT32 : gs::PSMCT24};
            const u32 tex_fbp = tbp_blocks / 32, tex_fbw = Bits(c.tex0, 14, 6);
            for (u32 psm : candidates) {
                const bool known = s.primary ? s.owned.count(std::make_tuple(tex_fbp, tex_fbw, psm, false)) != 0
                                             : s.renderer->HasTarget(tex_fbp, tex_fbw, psm);
                if (!known) continue;
                if (s.primary) {
                    Owned& source = s.owned[std::make_tuple(tex_fbp, tex_fbw, psm, false)];
                    Pages sampled;
                    if (SampledPages(source, c.clamp, tw, th, v, n, &sampled)) Sample(s, source, sampled);
                    else Use(s, tex_fbp, tex_fbw, psm, false, source.width, source.height, 1);   // sampling grows nothing
                    tex_source = &source;
                }
                if (tex_format == 4 && b_clut_rows_cpu_only(s)) {
                    // The CLUT rows only the CPU has, over to the GPU's first.
                    const u32 sent = s.clut_cpu & ~s.clut_gpu;
                    s.renderer->ClutRows(gs::TheGs().Clut().data(), sent);
                    for (u32 r = 0; r < 16; ++r)
                        if (sent >> r & 1) s.gpu_clut_source[r] = 0;
                    s.clut_gpu |= s.clut_cpu;
                }
                state.texture_target = true;
                state.tex_fbp = tex_fbp;
                state.tex_fbw = tex_fbw;
                state.tex_target_psm = psm;
                break;
            }
            state.tex_format = tex_format;
            state.ta0 = Bits(call.texa, 0, 8);
            state.ta1 = Bits(call.texa, 32, 8);
            state.aem = Bits(call.texa, 15, 1) != 0;
        }
        if (!state.texture_target) {
            if (s.primary) {
                state.texture = CachedTexture(s, c, call.texa, tw, th);
            } else {
                // Decoded again when the texture's registers change, local memory
                // is uploaded into, or a draw since went into the texture's blocks.
                const auto key = std::make_tuple(c.tex0, c.clamp, call.texa, gs::TheGs().uploads());
                const u32 tbp = Bits(c.tex0, 0, 14);
                const u32 tex_blocks = std::max(1u, Bits(c.tex0, 14, 6) * 64 * th * PixelBits(tex_psm) / 8 / 256);
                if (key != s.texture_state || s.drawn_into(tbp, tex_blocks)) {
                    std::vector<u32> pixels;
                    u32 w = 0, h = 0;
                    gs::TheGs().DecodeTexture(&pixels, &w, &h);
                    // Named by the registers that place it: decoding it again
                    // replaces it rather than adding another.
                    s.texture_key = c.tex0 & 0x1FFFFFFFFFFFFFFFull;
                    s.renderer->UploadTexture(s.texture_key, pixels.data(), w, h);
                    s.texture_state = key;
                    s.since_decode.clear();
                }
                state.texture = s.texture_key;
            }
        }
        state.tfx = ps2gpu::TextureFunction(Bits(c.tex0, 35, 2));
        state.tcc = Bits(c.tex0, 34, 1) != 0;
        const int k = int(std::int32_t(Bits(c.tex1, 32, 12) << 20) >> 20);
        const u32 mmin = Bits(c.tex1, 6, 3);
        state.bilinear = k <= 0 ? Bits(c.tex1, 5, 1) != 0 : (mmin == 1 || mmin == 4 || mmin == 5);
        // Mipmapped, per pixel (gs.frag's Sample): the levels decoded with it
        // (CachedTexture). A texture of K alone that never minifies, as before.
        state.mip = s.primary && !state.texture_target ? gs::MipMode(c) : 0;
        state.wms = Bits(c.clamp, 0, 2);
        state.wmt = Bits(c.clamp, 2, 2);
        state.minu = Bits(c.clamp, 4, 10);
        state.maxu = Bits(c.clamp, 14, 10);
        state.minv = Bits(c.clamp, 24, 10);
        state.maxv = Bits(c.clamp, 34, 10);
    }
    state.alpha_test = Bits(c.test, 0, 1) != 0;
    state.atst = Bits(c.test, 1, 3);
    state.aref = Bits(c.test, 4, 8);
    state.afail = Bits(c.test, 12, 2);
    state.date = Bits(c.test, 14, 1) != 0 && state.psm != gs::PSMCT24;
    state.datm = Bits(c.test, 15, 1);
    state.blend = (call.prim >> 6) & 1;
    state.a = ps2gpu::BlendColour(std::min(2u, Bits(c.alpha, 0, 2)));
    state.b = ps2gpu::BlendColour(std::min(2u, Bits(c.alpha, 2, 2)));
    state.c = ps2gpu::BlendAlpha(std::min(2u, Bits(c.alpha, 4, 2)));
    state.d = ps2gpu::BlendColour(std::min(2u, Bits(c.alpha, 6, 2)));
    state.fix = Bits(c.alpha, 32, 8);
    state.pabe = call.pabe;
    state.fog = (call.prim >> 5) & 1;
    state.fogcol = u32(call.fogcol & 0xFFFFFF);
    // Depth: ZTE with ZBUF.
    const u32 zpsm = 0x30 | Bits(c.zbuf, 24, 4);
    const u32 zmax = zpsm == gs::PSMZ32 ? 0xFFFFFFFFu : zpsm == gs::PSMZ24 ? 0xFFFFFFu : 0xFFFFu;
    state.depth = Bits(c.test, 16, 1) != 0;
    state.ztst = Bits(c.test, 17, 2);
    state.zbp = Bits(c.zbuf, 0, 9);
    state.depth_write = Bits(c.zbuf, 32, 1) == 0;
    // An alpha test failing every pixel (ATST NEVER) writes Z only for AFAIL
    // ZB_ONLY: otherwise none, whatever ZMSK says. Burnout 3's glow passes
    // (ATST NEVER, FB_ONLY, ZTST ALWAYS, ZMSK 0) were taken as writing their
    // depth buffer - over the glow images' own memory: each pass made those
    // stale, and reloaded the depth buffer (at 4x, a third of the GPU's time
    // went to moving pictures to and from its memory).
    if (state.alpha_test && state.atst == 0 && state.afail != 2) state.depth_write = false;
    // ZTST ALWAYS with ZMSK: the depth buffer plays no part. Burnout 3's race
    // blurs into scratch targets at its depth buffer's address with ZBUF left
    // there; treating that as a depth buffer would bring it up to date - a
    // write-back and a reload - for every one of their draws.
    if (state.depth && state.ztst == 1 && !state.depth_write) state.depth = false;
    // ZTST ALWAYS into a depth buffer over the colour target's own memory:
    // the GS writes each pixel's colour and its Z into the same memory, the Z
    // in its own layout - other words of it. As a depth buffer that is no
    // use; as memory it is: the race's glow passes lay their threshold grey
    // down this way, as colour and again as Z (the Z = the grey). So the Z
    // is drawn as colour into a view of that memory in the Z format, which
    // GPU memory reconciles with the colour like any two views.
    bool z_as_colour = false;
    if (s.primary && state.depth && state.ztst == 1) {
        const auto colour = Blocks(state.fbp, state.fbw, state.psm, rows);
        const auto z = Blocks(state.zbp, state.fbw, zpsm, rows);
        if (colour.first < z.second && z.first < colour.second) {
            state.depth = false;
            // An alpha test that fails every pixel writes Z only for AFAIL
            // ZB_ONLY (the race's glow passes: ATST NEVER, FB_ONLY - no Z).
            const bool none_pass = state.alpha_test && state.atst == 0;
            z_as_colour = state.depth_write && (!none_pass || state.afail == 2);
        }
    }
    Owned* depth = nullptr;
    if (state.depth) {
        if (s.primary) {
            depth = &Use(s, state.zbp, state.fbw, zpsm, true, state.width, 512, rows);   // (the renderer's kDepthRows)
        } else if (s.depth_seeded.insert(state.zbp).second) {
            // Each depth buffer starts the frame as local memory has it, so the
            // GPU tests against what the software GS drew.
            std::vector<u32> z(std::size_t(state.width) * state.height);
            const gs::Gs& g = gs::TheGs();
            for (u32 y = 0; y < state.height; ++y)
                for (u32 x = 0; x < state.width; ++x)
                    z[std::size_t(y) * state.width + x] = g.memory.Read(state.zbp * 32, state.fbw, x, y, zpsm);
            s.renderer->UploadDepth(state.zbp, state.width, state.height, z.data());
        }
    }

    for (std::size_t i = 0; i < n; ++i) v[i].z = std::min(v[i].z, zmax);
    // Not timed apart: tens of millions of primitives a race, and a timer's
    // two clock reads each cost as much as recording one.
    s.renderer->DrawTriangles(state, v, n);
    if (s.primary) {
        // FBMSK over every bit the format stores: the colour is not written.
        const u32 stored = state.psm == gs::PSMCT24 ? 0xFFFFFFu : Is16(state.psm) ? 0x80F8F8F8u : 0xFFFFFFFFu;
        const bool draws_colour = (state.fbmsk & stored) != stored;
        if (draws_colour) Drew(s, *target, x0, x1, y1, y0);
        if (depth && state.depth_write) Drew(s, *depth, x0, x1, y1, y0);
        // The next primitive may take the fast path - unless this one's
        // texture is a target, lies over what it draws (each primitive
        // makes it stale), or its Z goes in as colour.
        bool may = !z_as_colour && !s.Dumping();
        // A target that samples itself (feedback) sees each primitive's
        // drawing in the next: those go the whole way.
        if (state.texture_target && tex_source == target) may = false;
        if (may && state.textured && !state.texture_target) {
            const u32 tbp = Bits(c.tex0, 0, 14), tbw = Bits(c.tex0, 14, 6), tpsm = Bits(c.tex0, 20, 6);
            const u32 tend = std::min(16384u, tbp + std::max(1u, tbw * 64 * th * PixelBits(tpsm) / 8 / 256 + 32));
            // Over the rows a fast primitive can draw (it reaches no further
            // than this one's rows), not all the target could hold: the
            // race keeps textures past its depth buffer's 448 rows.
            const u32 colour_end = Blocks(state.fbp, state.fbw, state.psm, rows).second;
            if (target->begin < tend && tbp < colour_end) may = false;
            if (depth && depth->begin < tend && tbp < Blocks(depth->fbp, depth->fbw, depth->psm, rows).second)
                may = false;
        }
        if (may) {
            fast.valid = true;
            fast.mutations = g_mutations;
            fast.context = call.context;
            std::memcpy(fast.regs, regs, sizeof regs);
            fast.rows = rows;
            fast.zmax = zmax;
            fast.state = state;
            fast.target = target;
            fast.depth = depth && state.depth_write ? depth : nullptr;
            fast.source = state.texture_target ? tex_source : nullptr;
            fast.draws_colour = draws_colour;
        }
        if (z_as_colour) {
            // The Z, as the bytes the Z format stores, into the Z-format view.
            ps2gpu::GsDrawState zs;
            zs.fbp = state.zbp;
            zs.fbw = state.fbw;
            zs.psm = zpsm;
            zs.width = state.width;
            zs.height = state.height;
            zs.scissor_x0 = state.scissor_x0;
            zs.scissor_y0 = state.scissor_y0;
            zs.scissor_x1 = state.scissor_x1;
            zs.scissor_y1 = state.scissor_y1;
            zs.gouraud = state.gouraud;
            if (zpsm == gs::PSMZ24) zs.fbmsk = 0xFF000000u;
            ps2gpu::GsVertex zv[6];
            for (std::size_t i = 0; i < n; ++i) {
                zv[i] = v[i];
                const u32 z = v[i].z;
                const u32 c = Is16(zpsm) ? From16(z & 0xFFFF) : z;
                zv[i].r = u8(c);
                zv[i].g = u8(c >> 8);
                zv[i].b = u8(c >> 16);
                zv[i].a = u8(c >> 24);
            }
            Owned& view = Use(s, zs.fbp, zs.fbw, zs.psm, false, zs.width, zs.height, rows);
            s.renderer->DrawTriangles(zs, zv, n);
            Drew(s, view, x0, x1, y1, y0);
        }
        return;
    }
    Extent& e = s.drawn[std::make_tuple(state.fbp, state.fbw, state.psm)];
    e.x1 = std::max(e.x1, x1);
    e.y1 = std::max(e.y1, y1);
    u32& blocks = s.since_decode[state.fbp * 32];
    blocks = std::max(blocks, std::max(1u, state.fbw * 64 * e.y1 * PixelBits(state.psm) / 8 / 256));
}

void WriteBmp(const std::string& path, const std::vector<u32>& rgba, u32 stride, u32 w, u32 h) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const u32 data = w * h * 4, file = 54 + data;
    unsigned char header[54] = {'B', 'M'};
    auto put32 = [&](int at, u32 v) { for (int i = 0; i < 4; ++i) header[at + i] = (v >> (8 * i)) & 0xFF; };
    put32(2, file);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, u32(-std::int32_t(h)));                   // top-down
    header[26] = 1;
    header[28] = 32;
    put32(34, data);
    std::fwrite(header, 1, 54, f);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x) {
            const u32 c = rgba[std::size_t(y) * stride + x];
            const unsigned char px[4] = {(unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c, 0xFF};
            std::fwrite(px, 1, 4, f);
        }
    std::fclose(f);
}

}  // namespace

/* The displayed buffer to the window from the GPU's target (the software
   picture where the GPU has none), on the thread the GS runs on. */
/* The internal resolution the "resolution" setting asks for (PC SETTINGS):
   original 1, 2x 2, display as many times the PS2's 448 lines as the window
   has (at most 4). */
u32 WantedScale(const ps2gpu::Display* display) {
    const std::string wanted = Setting("resolution", "original");
    // "2x" (the menu's), or 3x and 4x written into the file by hand.
    if (wanted.size() == 2 && wanted[1] == 'x' && wanted[0] >= '2' && wanted[0] <= '4') return u32(wanted[0] - '0');
    if (wanted == "display") {
        const u32 lines = display ? display->PixelHeight() : 0;
        return std::min(4u, std::max(1u, lines / 448u));
    }
    return 1;
}

/* A new internal resolution: what the GPU drew goes to its local memory
   first, the renderer drops its targets and depth buffers, and every image
   comes back from memory at the new size as it is next used. */
void Rescale(Bridge& b, u32 scale) {
    for (auto& [key, o] : b.owned)
        if (o.gpu_newer) WriteBack(b, o);
    b.renderer->SetScale(scale);
    for (auto& [key, o] : b.owned) {
        o.stale = true;
        o.stale_beyond = false;
        o.why = 0;
        o.by16 = false;
        o.stale_row0 = 0;
        o.stale_row1 = ~0u;
        o.stale_pages.set();
        o.gpu_newer = false;
        o.dirty_begin = ~0u;
        o.dirty_end = 0;
        o.dirty_pages.reset();
        o.rect_dirty = false;
        o.checked = 0;
    }
    ++b.newer_epoch;
    ++g_mutations;
    std::fprintf(stderr, "[gpu] internal resolution: %ux\n", unsigned(scale));
}

bool g_present_on_render_thread = false;     // SetPresentOnRenderThread

void PresentGpuPicture(ps2gpu::Display& display, u64 due_ns, u64 frame) {
    u32 fbp = 0, fbw = 0, psm = 0;
    gs::Gs& gs = gs::TheGs();
    // present_on_render_thread: what the GS model and the bridge know of the
    // picture taken here, the rest - its image from the renderer, the
    // present - in turn on the renderer's thread, the GS thread going on (it
    // waited each frame for all the renderer had: a tenth of its time in
    // Stuntman's stunt). Field mode woven as below; nothing shown when the
    // renderer has no image after all (the picture before stays).
    if (g_bridge && g_present_on_render_thread && gs.Displayed(&fbp, &fbw, &psm)) {
        auto found = g_bridge->owned.find(std::make_tuple(fbp, fbw, psm, false));
        if (found != g_bridge->owned.end() && !found->second.stale) {
            const u32 shown_lines = gs.ShownLines();
            const bool weave = gs.weave && gs.FieldMode() && shown_lines <= 512;
            const u32 dby = weave ? gs.DisplayedDby() : 0, parity = u32(frame + 1) & 1;
            const u32 lines = std::min(shown_lines, 448u);
            static u64 shown_serial = 0;
            const bool fresh = found->second.drawn_serial != shown_serial;
            shown_serial = found->second.drawn_serial;
            ps2gpu::Display* shown = &display;
            g_bridge->renderer->Then([=](ps2gpu::GsRenderer& r) {
                std::uint64_t image = 0;
                int layout = 0;
                u32 w = 0, h = 0;
                const u32 k = r.Scale();
                const auto present = [&](u32 width, u32 height) {
                    CountPresentedPicture(fresh);
                    if (due_ns) shown->PresentImageAt(image, layout, width, height, due_ns);
                    else shown->PresentImage(image, layout, width, height);
                };
                if (weave && r.WeaveField(fbp, fbw, psm, dby, shown_lines, parity, &image, &layout, &w, &h))
                    present(std::min(w, 640u * k), h);
                else if (r.DisplayTarget(fbp, fbw, psm, &image, &layout, &w, &h))
                    present(std::min(w, 640u * k), std::min(h, lines * k));
            });
            return;
        }
    }
    // Field mode, woven (gs.weave: --deinterlace, the deinterlace setting):
    // this field on the lines of the parity the game drew it for, the one
    // before on the others, the whole picture shown (Gs::Picture's weave).
    if (g_bridge && gs.weave && gs.Displayed(&fbp, &fbw, &psm)) {
        const u32 lines = gs.ShownLines();
        auto found = g_bridge->owned.find(std::make_tuple(fbp, fbw, psm, false));
        std::uint64_t image = 0;
        int layout = 0;
        u32 w = 0, h = 0;
        const u32 k = g_bridge->renderer->Scale();
        if (gs.FieldMode() && lines <= 512 && found != g_bridge->owned.end() && !found->second.stale &&
            g_bridge->renderer->WeaveField(fbp, fbw, psm, gs.DisplayedDby(), lines, u32(frame + 1) & 1, &image,
                                           &layout, &w, &h)) {
            static u64 shown_serial = 0;
            CountPresentedPicture(found->second.drawn_serial != shown_serial);
            shown_serial = found->second.drawn_serial;
            if (due_ns) display.PresentImageAt(image, layout, std::min(w, 640u * k), h, due_ns);
            else display.PresentImage(image, layout, std::min(w, 640u * k), h);
            return;
        }
    }
    if (g_bridge && gs.Displayed(&fbp, &fbw, &psm)) {
        auto found = g_bridge->owned.find(std::make_tuple(fbp, fbw, psm, false));
        std::uint64_t image = 0;
        int layout = 0;
        u32 w = 0, h = 0;
        if (found != g_bridge->owned.end() && !found->second.stale &&
            g_bridge->renderer->DisplayTarget(fbp, fbw, psm, &image, &layout, &w, &h)) {
            // The target's own pixels, as many as the picture's 640x448 at the
            // internal resolution.
            const u32 k = g_bridge->renderer->Scale();
            const u32 lines = std::min(gs::TheGs().ShownLines(), 448u);    // 224 in field mode
            // A new picture when the GPU drew into the one shown since it was
            // last presented (show_fps).
            static u64 shown_serial = 0;
            CountPresentedPicture(found->second.drawn_serial != shown_serial);
            shown_serial = found->second.drawn_serial;
            if (due_ns) display.PresentImageAt(image, layout, std::min(w, 640u * k), std::min(h, lines * k), due_ns);
            else display.PresentImage(image, layout, std::min(w, 640u * k), std::min(h, lines * k));
            return;
        }
    }
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    if (gs::TheGs().Picture(&pixels, &w, &h)) display.Present(pixels.data(), w, h);
}

/* The texture pack (texture_pack.h): the "texture pack" folder beside the PC
   settings - the release's own folder - or the "texture_pack_dir" setting's,
   or PS2HOST_TEXTURE_PACK's; on unless "texture_pack" is off (Tab). Not for a
   scripted run (no PC settings): it draws the game's own textures. */
void OpenTexturePack() {
    namespace fs = std::filesystem;
    std::string dir = std::getenv("PS2HOST_TEXTURE_PACK") ? std::getenv("PS2HOST_TEXTURE_PACK") : "";
    if (dir.empty() && !SettingsPath().empty()) {
        dir = Setting("texture_pack_dir");
        if (dir.empty()) dir = (fs::path(SettingsPath()).parent_path() / "texture pack").u8string();
    }
    if (dir.empty()) return;
    texture_pack::SetEnabled(SettingOn("texture_pack", true));
    texture_pack::SetBlockCompression(g_bridge->renderer->BlockCompression());
    const std::size_t files = texture_pack::Open(dir);
    if (files)
        std::fprintf(stderr, "[texture pack] %zu textures in %s (%s; Tab turns it on and off)\n", files, dir.c_str(),
                     texture_pack::Enabled() ? "on" : "off");
}

bool StartGpuRenderer(std::string* error, GpuMode mode, bool show) {
    static Bridge bridge;
    // On the window's device when there is a window: one Vulkan device for both.
    ps2gpu::Gpu* device = WindowGpu();
    if (!device) {
        bridge.gpu = ps2gpu::CreateGpu(error);
        if (!bridge.gpu) return false;
        device = bridge.gpu.get();
    }
    bridge.renderer = ps2gpu::CreateGsRenderer(*device, error);
    if (!bridge.renderer) return false;
    g_bridge = &bridge;
    OpenTexturePack();
    bridge.primary = mode == GpuMode::Primary;
    // The CPU's waits for the GPU, on the profiler's timeline, by why.
    ps2gpu::SetGpuWaitObserver([](const char* why, std::uint64_t from_ns, std::uint64_t to_ns) {
        timeline::Span(why, from_ns, to_ns);
    });
    if (const char* dump = std::getenv("PS2HOST_GPU_DUMP_FRAME")) bridge.dump_frame = std::atoll(dump);
    if (const char* dir = std::getenv("PS2HOST_GPU_DUMP_DIR")) bridge.dump_passes = dir;
    bridge.targets_as_textures = mode == GpuMode::ShadowRtt;
    gs::Gs& g = gs::TheGs();
    g.on_draw = OnDraw;
    if (bridge.primary) {
        g.rasterize = false;
        g.on_memory = [](u32 begin, u32 end, bool write) {
            g_bridge->cause = write ? 2 : 1;
            Sync(*g_bridge, begin, end, write);
        };
        g.on_write = [](u32 begin, u32 end, const u8* cover) {
            g_bridge->cause = 2;
            SyncWrite(*g_bridge, begin, end, cover);
        };
        g.on_clut_load = [](u64 tex0, u64 texa) { return LoadClut(*g_bridge, tex0, texa); };
        g.on_transfer_done = [] { ApplyPatch(*g_bridge); };
    }
    if (show || bridge.primary) {
        // The window shows the GPU's picture: the displayed buffer read back
        // from its target (the software picture where the GPU has none).
        SetPictureSource([](std::vector<u32>* pixels, u32* width, u32* height) {
            u32 fbp = 0, fbw = 0, psm = 0;
            if (!g_bridge || !gs::TheGs().Displayed(&fbp, &fbw, &psm)) return false;
            if (g_bridge->primary) {
                auto found = g_bridge->owned.find(std::make_tuple(fbp, fbw, psm, false));
                if (found == g_bridge->owned.end() || found->second.stale) return false;
            }
            std::vector<u32> rgba;
            u32 w = 0, h = 0;
            if (!g_bridge->renderer->ReadTarget(fbp, fbw, psm, &rgba, &w, &h)) return false;
            const u32 shown_w = std::min(w, 640u), shown_h = std::min(h, 448u);
            pixels->resize(std::size_t(shown_w) * shown_h);
            for (u32 y = 0; y < shown_h; ++y)
                for (u32 x = 0; x < shown_w; ++x) {
                    const u32 c = rgba[std::size_t(y) * w + x];
                    (*pixels)[std::size_t(y) * shown_w + x] = (c & 0xFF00FF00u) | ((c & 0xFF) << 16) | ((c >> 16) & 0xFF);
                }
            *width = shown_w;
            *height = shown_h;
            return true;
        });
    }
    if (bridge.primary) {
        // The end of each frame on the GS thread, at its place in the stream:
        // the frame to the GPU, and the window's picture straight from the
        // displayed target (no read-back); the EE goes on meanwhile.
        SetGsFrameHandler([](u64 frame) {
            GpuShadowFrame(frame, std::string());
            // The internal resolution, between frames (the GS thread's).
            const u32 scale = WantedScale(WindowDisplay());
            if (scale != g_bridge->renderer->Scale()) Rescale(*g_bridge, scale);
            if (ps2gpu::Display* display = WindowDisplay()) {
                const bool pace = !VirtualClock() && SettingOn("frame_pacing", true);    // (PC SETTINGS)
                PresentGpuPicture(*display, pace ? FrameDue(frame) : 0, frame);
            } else if (gs::TheGs().weave && gs::TheGs().FieldMode()) {
                // No window, woven all the same (--deinterlace weave): what a
                // capture (F12, PS2HOST_CAPTURE_FRAME) shows is both fields.
                gs::Gs& gs = gs::TheGs();
                u32 fbp = 0, fbw = 0, psm = 0, w = 0, h = 0;
                std::uint64_t image = 0;
                int layout = 0;
                if (gs.Displayed(&fbp, &fbw, &psm))
                    g_bridge->renderer->WeaveField(fbp, fbw, psm, gs.DisplayedDby(), gs.ShownLines(),
                                                   u32(frame + 1) & 1, &image, &layout, &w, &h);
            }
        });
    }
    std::fprintf(stderr, "gpu: drawing on %s %s\n", device->Describe().c_str(),
                 bridge.primary ? "instead of the software GS" : "beside the software GS");
    return true;
}

bool GpuPrimary() { return g_bridge && g_bridge->primary; }

void SetDateArea(const std::string& mode) {
    if (!g_bridge) return;
    g_bridge->renderer->SetDateArea(mode != "full");
    if (mode == "full") std::fprintf(stderr, "[gpu] DATE set up over the whole target\n");
}

bool g_render_queue_fast = false;           // SetRenderQueue

void SetRenderQueue(const std::string& mode) {
    g_render_queue_fast = mode != "plain";
    if (g_bridge) g_bridge->renderer->SetQueueFast(g_render_queue_fast);
}

void SetPresentOnRenderThread(bool on) {
    g_present_on_render_thread = on;
    if (g_bridge) g_bridge->renderer->SetThenQueued(on);
}

std::string SetRenderThread(const std::string& mode) {
    if (!g_bridge) return "off";
    // The renderer itself again (its queue run), then on a thread if asked.
    if (std::unique_ptr<ps2gpu::GsRenderer> inner = g_bridge->renderer->TakeInner())
        g_bridge->renderer = std::move(inner);
    if (mode != "on" && mode != "check") return "off";
    g_bridge->renderer = ps2gpu::CreateThreadedRenderer(std::move(g_bridge->renderer), mode == "check");
    g_bridge->renderer->SetThenQueued(g_present_on_render_thread);
    g_bridge->renderer->SetQueueFast(g_render_queue_fast);
    if (mode == "check")
        std::fprintf(stderr, "[gpu] the renderer on a thread of its own, its answers checked\n");
    return mode;
}

bool RenderThreadCheck(std::uint64_t* answers, std::uint64_t* differ) {
    return g_bridge && g_bridge->renderer->RenderThreadCheck(answers, differ);
}

void SetVulkanThread(bool on) {
    if (!g_bridge) return;
    if (on && !g_bridge->renderer->SetRecordThread(true))
        std::fprintf(stderr, "[gpu] vulkan_thread=on: not with this renderer\n");
    if (!on) g_bridge->renderer->SetRecordThread(false);
}

std::string SetDateInPass(const std::string& mode) {
    if (!g_bridge) return "off";
    const int want = mode == "check" ? 2 : mode == "on" ? 1 : 0;
    const int got = g_bridge->renderer->SetDateInPass(want);
    if (want && !got)
        std::fprintf(stderr, "[gpu] date_in_pass=%s: this GPU cannot (DATE set up in a pass of its own)\n", mode.c_str());
    if (got == 2) std::fprintf(stderr, "[gpu] DATE set up within the draws' pass, checked against the targets\n");
    return got == 2 ? "check" : got == 1 ? "on" : "off";
}

bool DateInPassCheck(std::uint64_t* batches, std::uint64_t* differ) {
    return g_bridge && g_bridge->renderer->DateInPassCheck(batches, differ);
}

void SetOwnedIndex(bool on) {
    if (!g_bridge) return;
    g_bridge->owned_index = on;
    if (!on) std::fprintf(stderr, "[gpu] images over blocks: looked for among all\n");
}

void SetNewerList(bool on) {
    if (!g_bridge) return;
    g_bridge->newer_list = on;
    if (!on) std::fprintf(stderr, "[gpu] images newer than memory: looked for among all\n");
}

void SetBatchOverlap(const std::string& mode) {
    if (!g_bridge) return;
    g_bridge->renderer->SetExactOverlap(mode != "bbox");
    g_bridge->renderer->SetPreparedAxes(mode != "exact-direct");
    if (mode == "bbox") std::fprintf(stderr, "[gpu] batch overlap: bounding rectangles\n");
}

void SetTextureCache(const std::string& mode) {
    if (!g_bridge) return;
    g_bridge->texture_cache = mode == "off"     ? Bridge::TextureCache::kOff
                              : mode == "check" ? Bridge::TextureCache::kCheck
                                                : Bridge::TextureCache::kOn;
    if (mode != "on") std::fprintf(stderr, "[texture cache] %s\n", mode.c_str());
}

void GpuSetScale(u32 scale) {
    if (g_bridge && scale != g_bridge->renderer->Scale()) Rescale(*g_bridge, scale);
}

void GpuCaptureBegin(u64 frame, const std::string& passes_dir) {
    if (!g_bridge) return;
    g_bridge->dump_passes = passes_dir;
    g_bridge->dump_frame = (long long)frame;
    g_bridge->dump_last = (long long)frame + 1;     // a game drawing a picture every other blank
}

std::string GpuCaptureScreen(const std::string& dir) {
    if (!g_bridge) return "no GPU";
    Bridge& s = *g_bridge;
    u32 fbp = 0, fbw = 0, psm = 0;
    gs::TheGs().Displayed(&fbp, &fbw, &psm);
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    char text[160];
    const u32 k = s.renderer->Scale();
    if (s.renderer->ReadTargetFull(fbp, fbw, psm, &pixels, &w, &h))
        WriteBmp(dir + "/screen.bmp", pixels, w, std::min(w, 640u * k), std::min(h, 448u * k));
    if (s.renderer->ReadTarget(fbp, fbw, psm, &pixels, &w, &h))
        WriteBmp(dir + "/screen_1x.bmp", pixels, w, std::min(w, 640u), std::min(h, 448u));
    // Field mode, woven: the picture the window shows (both fields).
    gs::Gs& gs = gs::TheGs();
    std::uint64_t image = 0;
    int layout = 0;
    if (gs.weave && gs.FieldMode() &&
        s.renderer->WeaveField(fbp, fbw, psm, gs.DisplayedDby(), gs.ShownLines(), 1 - ShownField(), &image, &layout,
                               &w, &h) &&
        s.renderer->ReadWoven(&pixels, &w, &h))
        WriteBmp(dir + "/screen_woven.bmp", pixels, w, std::min(w, 640u * k), h);
    std::snprintf(text, sizeof text, "shown: FBP %X FBW %u PSM %02X; internal resolution %ux", unsigned(fbp),
                  unsigned(fbw), unsigned(psm), unsigned(k));
    return text;
}

void GpuShadowFrame(u64 frame, const std::string& dump_dir) {
    if (!g_bridge) return;
    Bridge& s = *g_bridge;
    if (s.primary) {
        if (s.Dumping() && std::get<0>(s.last_target) != ~0u) DumpPass(s, s.last_target);
        s.frame = frame + 1;
        // The frame's drawing to the GPU now, to run while the next is made,
        // rather than when something next waits on the GPU.
        {
            Timed timed(Section::GpuRender);
            s.renderer->Kick();
        }
        if (frame % 100 == 0)
            std::fprintf(stderr, "[gpu] frame %llu: %llu draws, %llu write-backs, %llu reloads, %llu decodes, %zu images, %zu textures\n",
                         static_cast<unsigned long long>(frame), static_cast<unsigned long long>(s.serial),
                         static_cast<unsigned long long>(s.write_backs), static_cast<unsigned long long>(s.reloads),
                         static_cast<unsigned long long>(s.decodes), s.owned.size(), s.textures.size());
        if (frame % 300 == 0 && std::getenv("PS2HOST_GPU_PROFILE"))
            std::fprintf(stderr, "[gpu] frame %llu: %s\n", static_cast<unsigned long long>(frame),
                         ps2gpu::GpuProfileReport().c_str());
        if (frame % 100 == 0 && std::getenv("PS2HOST_GPU_COUNT")) {
            std::fprintf(stderr, "[gpu]   last 100 frames: %s\n", ps2gpu::RendererCounters().c_str());
            std::fprintf(stderr, "[gpu]   write-backs so far %llu; written over unread %llu (blocks %llu)\n",
                         (unsigned long long)g_wb_done, (unsigned long long)g_wb_never_read,
                         (unsigned long long)g_wb_never_read_blocks);
            std::vector<std::pair<u64, std::string>> kinds;
            for (const auto& [k, n] : g_decode_kinds) kinds.emplace_back(n, k);
            std::sort(kinds.rbegin(), kinds.rend());
            for (std::size_t i = 0; i < kinds.size() && i < 12; ++i)
                std::fprintf(stderr, "[gpu]     GPU decodes %llu: %s\n", (unsigned long long)kinds[i].first, kinds[i].second.c_str());
            g_decode_kinds.clear();
        }
        if (std::getenv("PS2HOST_GPU_LOG")) {
            std::fprintf(stderr, "[gpu]   KB to the CPU: %llu for textures, %llu for reads, %llu for writes;",
                         static_cast<unsigned long long>(s.downloads_by[0] / 4),
                         static_cast<unsigned long long>(s.downloads_by[1] / 4),
                         static_cast<unsigned long long>(s.downloads_by[2] / 4));
            for (const auto& [kind, blocks] : s.texture_downloads)
                std::fprintf(stderr, " psm %02X tbw %u: %llu KB", unsigned(kind >> 16), unsigned(kind & 0xFFFF),
                             static_cast<unsigned long long>(blocks / 4));
            std::fprintf(stderr, "\n");
        }
        if (std::getenv("PS2HOST_GPU_LOG"))
            for (const auto& [key, o] : s.owned)
                std::fprintf(stderr, "[gpu]   %s %X/%u/%02X blocks %X-%X drawn to %X, %ux%u; reloads new %llu memory %llu draw %llu write-back %llu\n",
                             o.depth ? "Z" : "C", unsigned(o.fbp), unsigned(o.fbw), unsigned(o.psm), unsigned(o.begin),
                             unsigned(o.end), unsigned(o.drawn_end), unsigned(o.x1), unsigned(o.y1),
                             static_cast<unsigned long long>(o.reloads[0]), static_cast<unsigned long long>(o.reloads[1]),
                             static_cast<unsigned long long>(o.reloads[2]), static_cast<unsigned long long>(o.reloads[3]));
        return;
    }
    if (!dump_dir.empty()) s.dump_dir = dump_dir;
    s.renderer->Flush();
    const gs::Gs& g = gs::TheGs();
    u32 shown_fbp = 0, shown_fbw = 0, shown_psm = 0;
    g.Displayed(&shown_fbp, &shown_fbw, &shown_psm);
    std::string line;
    u64 total = 0, close = 0;
    for (const auto& [key, extent] : s.drawn) {
        const auto [fbp, fbw, psm] = key;
        std::vector<u32> pixels;
        u32 w = 0, h = 0;
        if (!s.renderer->ReadTarget(fbp, fbw, psm, &pixels, &w, &h)) continue;
        const u32 x1 = std::min(extent.x1, w), y1 = std::min(extent.y1, h);
        const bool sixteen = psm == gs::PSMCT16 || psm == gs::PSMCT16S;
        const int slack = sixteen ? 8 : 2;
        u64 n = 0, ok = 0;
        for (u32 y = 0; y < y1; ++y)
            for (u32 x = 0; x < x1; ++x) {
                const u32 want = g.FramePixel(fbp, fbw, psm, x, y), got = pixels[std::size_t(y) * w + x];
                bool same = true;
                for (int shift = 0; shift < (psm == gs::PSMCT32 ? 32 : 24); shift += 8) {   // alpha too where it is stored
                    const int a = int((want >> shift) & 0xFF), b = int((got >> shift) & 0xFF);
                    if (std::abs(a - b) > slack) same = false;
                }
                ++n;
                ok += same;
            }
        total += n;
        close += ok;
        char text[96];
        std::snprintf(text, sizeof text, " %X/%u/%02X %ux%u %.1f%%", unsigned(fbp), unsigned(fbw), unsigned(psm),
                      unsigned(x1), unsigned(y1), n ? 100.0 * double(ok) / double(n) : 100.0);
        line += text;
        if (!s.dump_dir.empty() && x1 && y1) {
            char name[96];
            std::snprintf(name, sizeof name, "/gpu_%06llu_%X.bmp", static_cast<unsigned long long>(frame),
                          unsigned(fbp));
            WriteBmp(s.dump_dir + name, pixels, w, x1, y1);
            // The software GS's, beside it.
            std::vector<u32> reference(std::size_t(x1) * y1);
            for (u32 y = 0; y < y1; ++y)
                for (u32 x = 0; x < x1; ++x) reference[std::size_t(y) * x1 + x] = g.FramePixel(fbp, fbw, psm, x, y);
            std::snprintf(name, sizeof name, "/soft_%06llu_%X.bmp", static_cast<unsigned long long>(frame),
                          unsigned(fbp));
            WriteBmp(s.dump_dir + name, reference, x1, x1, y1);
        }
    }
    if (total) {
        ++s.frames;
        s.agreement_sum += double(close) / double(total);
        std::fprintf(stderr, "[gpu-shadow] frame %llu:%s (shown %X)\n", static_cast<unsigned long long>(frame),
                     line.c_str(), unsigned(shown_fbp));
    }
    if (!s.drawn.empty()) s.dump_dir.clear();
    s.drawn.clear();
    s.depth_seeded.clear();
}

std::string GpuShadowSummary() {
    if (!g_bridge) return "";
    char text[320];
    if (g_bridge->primary) {
        std::snprintf(text, sizeof text,
                      "gpu: %llu images into GPU local memory, %llu out of it; %llu KB to the CPU, %llu KB"
                      " from it; %llu textures decoded (%llu more found unchanged), %llu on the GPU;"
                      " %llu CLUTs loaded on the GPU",
                      static_cast<unsigned long long>(g_bridge->write_backs),
                      static_cast<unsigned long long>(g_bridge->reloads),
                      static_cast<unsigned long long>(g_bridge->downloads / 4),
                      static_cast<unsigned long long>(g_bridge->uploads / 4),
                      static_cast<unsigned long long>(g_bridge->decodes),
                      static_cast<unsigned long long>(g_bridge->unchanged),
                      static_cast<unsigned long long>(g_bridge->gpu_decodes),
                      static_cast<unsigned long long>(g_bridge->gpu_clut_loads));
        std::string summary = std::string(text) + "\ngpu: " + ps2gpu::SubmitStats();
        char cache[200];
        std::snprintf(cache, sizeof cache,
                      "\ngpu: texture cache %s: %llu CLUT loads kept; checked %llu, %llu different",
                      g_bridge->texture_cache == Bridge::TextureCache::kOff     ? "off"
                      : g_bridge->texture_cache == Bridge::TextureCache::kCheck ? "check"
                                                                                : "on",
                      static_cast<unsigned long long>(g_bridge->clut_loads_kept),
                      static_cast<unsigned long long>(g_bridge->clut_checks),
                      static_cast<unsigned long long>(g_bridge->clut_check_differences));
        summary += cache;
        if (texture_pack::Count())
            summary += "\ngpu: texture pack of " + std::to_string(texture_pack::Count()) + " files, " +
                       std::to_string(g_bridge->replacements) + " textures replaced";
        return summary;
    }
    if (!g_bridge->frames) return "";
    std::snprintf(text, sizeof text, "gpu-shadow: %llu frames compared, %.1f%% of pixels agree on average",
                  static_cast<unsigned long long>(g_bridge->frames), 100.0 * g_bridge->agreement_sum / double(g_bridge->frames));
    return text;
}

#else

bool StartGpuRenderer(std::string* error, GpuMode, bool) {
    if (error) *error = "built without the GPU renderer (PS2HOST_VULKAN)";
    return false;
}
void GpuSetScale(u32) {}
void GpuShadowFrame(gs::u64, const std::string&) {}
std::string GpuShadowSummary() { return ""; }

#endif

}  // namespace ps2host
