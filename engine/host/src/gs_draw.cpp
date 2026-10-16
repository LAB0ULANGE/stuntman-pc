/*
 * GS drawing (gs.h): primitives from the vertex queue, rasterized in software,
 * each pixel through the GS's pipeline (GS User's Manual, chapter 5):
 *
 *   scissor -> texture (TEX0/CLAMP/TEXA/CLUT; TFX modulate/decal/highlight)
 *   -> fog -> alpha test (TEST) -> depth test (ZBUF) -> blending (ALPHA,
 *   PABE, COLCLAMP) -> frame mask (FBMSK) -> FRAME and ZBUF writes.
 *
 * Primitives are queued with their decoded state (Setup, the CLUT they use)
 * and drawn in batches: the rows are dealt out in groups of 8 to worker
 * threads, each of which draws every queued primitive over its own rows, so
 * every pixel still sees the primitives in order and the picture is the same
 * as drawn one by one. A primitive that samples memory queued ones write waits
 * for them; one that samples what it writes itself is drawn alone, in order.
 * Whatever reads local memory - the picture, transfers, CLUT loads - finishes
 * the queue first (Gs::Finish).
 *
 * Stage one samples textures at the nearest texel or bilinearly and does not
 * dither; anti-aliasing is not modelled.
 */

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#include "ps2host/gs.h"
#include "ps2host/profile.h"

namespace ps2host {
namespace gs {

namespace {

u32 Bits(u64 value, int low, int count) { return u32((value >> low) & ((1ull << count) - 1)); }

u32 Rgba(u32 r, u32 g, u32 b, u32 a) { return r | g << 8 | b << 16 | a << 24; }

u32 From16(u32 c, u64 texa) {
    const u32 r = (c & 31) << 3, g = ((c >> 5) & 31) << 3, b = ((c >> 10) & 31) << 3;
    const u32 a = (c & 0x8000) ? Bits(texa, 32, 8) : ((Bits(texa, 15, 1) && !(c & 0x7FFF)) ? 0 : Bits(texa, 0, 8));
    return Rgba(r, g, b, a);
}

u32 To16(u32 c) {
    return ((c >> 3) & 31) | (((c >> 11) & 31) << 5) | (((c >> 19) & 31) << 10) | ((c >> 31) << 15);
}

// Pixels drawn by this thread since it last reported them.
thread_local u64 t_pixels = 0;

int Wrap(int i, u32 size, u32 mode, u32 min, u32 max) {
    switch (mode) {
        case 0: return i & int(size - 1);                                   // REPEAT
        case 1: return std::clamp(i, 0, int(size) - 1);                     // CLAMP
        case 2: return std::clamp(i, int(min), int(max));                   // REGION_CLAMP
        default: return (i & int(min)) | int(max);                         // REGION_REPEAT
    }
}

u32 BitsPerPixel(u32 psm) {
    switch (psm) {
        case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: return 16;
        case PSMT8: return 8;
        case PSMT4: return 4;
        default: return 32;
    }
}

// Whether row y belongs to band `band` of `bands`: rows go round in groups of 8.
bool Owned(int y, int band, int bands) { return bands <= 1 || (y >= 0 && ((y >> 3) % bands) == band); }

/* The worker threads a batch's bands are drawn on. Workers live for the
   process. */
class RowPool {
public:
    static RowPool& Get() {
        static RowPool* pool = new RowPool;     // never destroyed: its threads wait to the end
        return *pool;
    }
    int threads() const { return int(workers_) + 1; }
    /* body(i0, i1) over [begin, end), in chunks, on the workers and the caller. */
    void Run(int begin, int end, const std::function<void(int, int)>& body) {
        {
            std::lock_guard<std::mutex> hold(lock_);
            body_ = &body;
            next_ = begin;
            end_ = end;
            chunk_ = 1;
            pending_ = workers_;
            ++generation_;
        }
        wake_.notify_all();
        int a, b;
        while (Take(&a, &b)) body(a, b);
        std::unique_lock<std::mutex> hold(lock_);
        done_.wait(hold, [&] { return pending_ == 0; });
        body_ = nullptr;
    }

private:
    RowPool() {
        const unsigned cores = std::thread::hardware_concurrency();
        workers_ = cores > 2 ? std::min(cores - 1, 7u) : 1u;
        for (unsigned i = 0; i < workers_; ++i) std::thread([this] { Work(); }).detach();
    }
    bool Take(int* a, int* b) {
        std::lock_guard<std::mutex> hold(lock_);
        if (next_ >= end_) return false;
        *a = next_;
        *b = std::min(end_, next_ + chunk_);
        next_ = *b;
        return true;
    }
    void Work() {
        u64 seen = 0;
        for (;;) {
            const std::function<void(int, int)>* body;
            {
                std::unique_lock<std::mutex> hold(lock_);
                wake_.wait(hold, [&] { return generation_ != seen; });
                seen = generation_;
                body = body_;
            }
            int a, b;
            while (body && Take(&a, &b)) (*body)(a, b);
            std::lock_guard<std::mutex> hold(lock_);
            if (--pending_ == 0) done_.notify_one();
        }
    }
    unsigned workers_ = 0;
    std::mutex lock_;
    std::condition_variable wake_, done_;
    const std::function<void(int, int)>* body_ = nullptr;
    int next_ = 0, end_ = 0, chunk_ = 1;
    unsigned pending_ = 0;
    u64 generation_ = 0;
};

// A batch is drawn when it holds this many primitives, or before anything
// reads local memory; one of fewer pixels than this is drawn on one thread.
constexpr std::size_t kBatchPrimitives = 1024;
constexpr long long kParallelPixels = 8192;

// Texture coordinates of a vertex in texels: UV directly (1/16 texel), or
// S/Q, T/Q scaled by the texture's size.
void Coords(const Vertex& v, bool fst, u32 tw, u32 th, float* u, float* t) {
    if (fst) { *u = float(v.u) / 16.0f; *t = float(v.v) / 16.0f; }
    else {
        const float q = v.q != 0.0f ? v.q : 1.0f;
        *u = v.s / q * float(tw);
        *t = v.t / q * float(th);
    }
}

}  // namespace

MipLevels LevelsOf(const Context& c) {
    MipLevels m;
    m.tbp[0] = Bits(c.tex0, 0, 14);
    m.tbw[0] = Bits(c.tex0, 14, 6);
    const u32 mxl = std::min(Bits(c.tex1, 2, 3), 6u), mmin = Bits(c.tex1, 6, 3);
    if (mxl == 0 || mmin < 2 || mmin > 5) return m;
    m.count = mxl + 1;
    const u32 tw = 1u << std::min(Bits(c.tex0, 26, 4), 10u), th = 1u << std::min(Bits(c.tex0, 30, 4), 10u);
    const u32 bits = BitsPerPixel(Bits(c.tex0, 20, 6));
    for (u32 n = 1; n <= mxl; ++n) {
        if (n <= 3 && Bits(c.tex1, 9, 1)) {
            const u32 w = std::max(1u, tw >> (n - 1)), h = std::max(1u, th >> (n - 1));
            m.tbp[n] = m.tbp[n - 1] + w * h * bits / 8 / 256;
            m.tbw[n] = std::max(1u, m.tbw[n - 1] / 2);
        } else {
            const u64 r = n <= 3 ? c.miptbp1 : c.miptbp2;
            const int shift = int((n - 1) % 3) * 20;
            m.tbp[n] = Bits(r, shift, 14);
            m.tbw[n] = Bits(r, shift + 14, 6);
        }
    }
    return m;
}

u32 MipMode(const Context& c) {
    static const bool off = std::getenv("PS2HOST_NO_MIPMAPS") != nullptr;     // to compare: level 0 alone, as before
    if (off) return 0;
    const int k = int(s32(Bits(c.tex1, 32, 12) << 20) >> 20);
    if (LevelsOf(c).count == 1 || (Bits(c.tex1, 0, 1) && k <= 0)) return 0;
    return 2u | std::min(Bits(c.tex1, 2, 3), 6u) << 2 | Bits(c.tex1, 6, 3) << 5 | Bits(c.tex1, 5, 1) << 8 |
           Bits(c.tex1, 0, 1) << 9 | Bits(c.tex1, 19, 2) << 10 | Bits(c.tex1, 32, 12) << 16;
}

MipChoice ChooseLevel(u32 m, float q) {
    MipChoice out;
    const int mxl = int((m >> 2) & 7), mmin = int((m >> 5) & 7);
    float lod = float(s32(m << 4) >> 20) / 16.0f;      // K, 4 fractional bits
    if (!((m >> 9) & 1)) lod += -std::log2(std::fabs(q)) * float(1u << ((m >> 10) & 3));
    if (!(lod > 0.0f)) {
        out.linear = (m >> 8) & 1;
        return out;
    }
    out.linear = mmin == 4 || mmin == 5;
    if (mmin == 2 || mmin == 4) {
        out.n0 = out.n1 = int(std::min(std::floor(lod + 0.5f), float(mxl)));
        return out;
    }
    lod = std::min(lod, float(mxl));
    out.n0 = int(std::floor(lod));
    out.f = lod - float(out.n0);
    out.n1 = out.n0 < mxl && out.f > 0.0f ? out.n0 + 1 : out.n0;
    return out;
}

// ---------------------------------------------------------------- primitives

void Gs::Kick(bool draw) {
    const u32 type = Prim() & 7;
    auto shift = [&](int keep_first) {
        if (keep_first) { queue_[1] = queue_[2]; queued_ = 2; }
        else { queue_[0] = queue_[1]; queue_[1] = queue_[2]; queued_ = 2; }
    };
    switch (type) {
        case 0:                                         // point
            if (draw) Draw();
            queued_ = 0;
            break;
        case 1: case 6:                                 // line, sprite
            if (queued_ < 2) return;
            if (draw) Draw();
            queued_ = 0;
            break;
        case 2:                                         // line strip
            if (queued_ < 2) return;
            if (draw) Draw();
            queue_[0] = queue_[1];
            queued_ = 1;
            break;
        case 3:                                         // triangle
            if (queued_ < 3) return;
            if (draw) Draw();
            queued_ = 0;
            break;
        case 4:                                         // triangle strip
            if (queued_ < 3) return;
            if (draw) Draw();
            shift(0);
            break;
        case 5:                                         // triangle fan
            if (queued_ < 3) return;
            if (draw) Draw();
            shift(1);
            break;
        default:
            queued_ = 0;
            break;
    }
}

void Gs::DecodeTexture(std::vector<u32>* pixels, u32* width, u32* height) const {
    Finish();
    if (setup_stale_) const_cast<Gs*>(this)->Prepare();
    const Setup& s = setup_;
    pixels->resize(std::size_t(s.tw) * s.th);
    for (u32 v = 0; v < s.th; ++v)
        for (u32 u = 0; u < s.tw; ++u) (*pixels)[std::size_t(v) * s.tw + u] = TexelAt(s, int(u), int(v));
    *width = s.tw;
    *height = s.th;
}

bool Gs::Displayed(u32* fbp, u32* fbw, u32* psm) const {
    int circuit = Bits(pmode_, 1, 1) ? 1 : Bits(pmode_, 0, 1) ? 0 : -1;
    if (circuit < 0) {
        if (!dispfb_[1] && !dispfb_[0]) return false;
        circuit = dispfb_[1] ? 1 : 0;
    }
    const u64 fb = dispfb_[circuit];
    *fbp = Bits(fb, 0, 9);
    *fbw = Bits(fb, 9, 6);
    *psm = Bits(fb, 15, 5);
    return *fbw != 0;
}

u32 Gs::FramePixel(u32 fbp, u32 fbw, u32 psm, u32 x, u32 y) const {
    Finish();
    const u32 c = memory.Read(fbp * 32, fbw, x, y, psm);
    if (psm == PSMCT16 || psm == PSMCT16S) return From16(c, 0x80ull << 32);
    if (psm == PSMCT24) return (c & 0xFFFFFF) | 0x80000000u;
    return c;
}

void Gs::Finish() const { const_cast<Gs*>(this)->FinishJobs(); }

void Gs::Draw() {
    // Timed only when it rasterises: under --gpu the bridge times itself, and
    // a timer per primitive is a measurable share of a race.
    std::optional<Timed> timed;
    if (rasterize) timed.emplace(Section::GsDraw);
    ++primitives_;
    // The software rasteriser's state; without it (--gpu), made only if the
    // texture is decoded from it - a few thousand of 73 million primitives.
    if (rasterize) Prepare();
    else setup_stale_ = true;
    if (on_draw) {
        Finish();                               // the second renderer reads local memory as of now
        on_draw(DrawCall{Prim(), &Ctx(), texa_, queue_, queued_, fogcol_, (pabe_ & 1) != 0});
    }
    if (trace) {
        const Context& c = Ctx();
        const Vertex& v = queue_[0];
        const Vertex& w = queue_[queued_ - 1];
        std::fprintf(trace_out, "[gs] prim %llX tex0 tbp %X tbw %u psm %02X %ux%u cbp %X cpsm %u csm %u csa %u cld %u"
                     " | (%.1f,%.1f)-(%.1f,%.1f) uv (%u,%u)-(%u,%u) st (%.3f,%.3f)-(%.3f,%.3f) q %.3f-%.3f"
                     " rgba %02X%02X%02X%02X z %X-%X from %08X xyz %08X\n",
                     (unsigned long long)Prim(), Bits(c.tex0, 0, 14), Bits(c.tex0, 14, 6), Bits(c.tex0, 20, 6),
                     1u << Bits(c.tex0, 26, 4), 1u << Bits(c.tex0, 30, 4), Bits(c.tex0, 37, 14),
                     Bits(c.tex0, 51, 4), Bits(c.tex0, 55, 1), Bits(c.tex0, 56, 5), Bits(c.tex0, 61, 3),
                     v.x, v.y, w.x, w.y, v.u, v.v, w.u, w.v, v.s, v.t, w.s, w.t, v.q, w.q, w.r, w.g, w.b, w.a, v.z, w.z,
                     rgba_source_, xyz_source_);
        if (queued_ == 3)
            std::fprintf(trace_out, "[gs]   middle vertex (%.1f,%.1f) st (%.4f,%.4f) q %.4f\n", queue_[1].x,
                         queue_[1].y, queue_[1].s, queue_[1].t, queue_[1].q);
        std::fprintf(trace_out, "[gs]   tex1 %llX clamp %llX texa %llX tfx %u tcc %u\n", (unsigned long long)c.tex1,
                     (unsigned long long)c.clamp, (unsigned long long)texa_, Bits(c.tex0, 35, 2), Bits(c.tex0, 34, 1));
        std::fprintf(trace_out, "[gs]   frame fbp %X fbw %u psm %02X fbmsk %08X; zbuf zbp %X psm %02X zmsk %u; test %llX alpha %llX\n",
                     Bits(c.frame, 0, 9), Bits(c.frame, 16, 6), Bits(c.frame, 24, 6), Bits(c.frame, 32, 32),
                     Bits(c.zbuf, 0, 9), Bits(c.zbuf, 24, 4), Bits(c.zbuf, 32, 1),
                     (unsigned long long)c.test, (unsigned long long)c.alpha);
    }
    if (trace) {
        // PS2HOST_GS_DUMP_BEFORE=TBP,PSM (hex): local memory to gsmem_before.bin
        // as the traced stream's first primitive with that texture is drawn.
        static const char* want = std::getenv("PS2HOST_GS_DUMP_BEFORE");
        static bool dumped = false;
        if (want && !dumped) {
            unsigned tbp = 0, psm = 0;
            const Context& c = Ctx();
            if (std::sscanf(want, "%x,%x", &tbp, &psm) == 2 && Bits(c.tex0, 0, 14) == tbp &&
                Bits(c.tex0, 20, 6) == psm && (Prim() >> 4 & 1)) {
                dumped = true;
                if (std::FILE* f = std::fopen("gsmem_before.bin", "wb")) {
                    std::fwrite(memory.data(), 1, 4u << 20, f);
                    std::fclose(f);
                }
            }
        }
    }
    const u32 type = u32(Prim() & 7);
    if (type == 7 || !rasterize) return;
    Job job{setup_, {}, queued_, 0};
    for (u32 i = 0; i < queued_ && i < 3; ++i) job.v[i] = queue_[i];
    // About how many pixels: the box around it (half for a triangle).
    float x0 = job.v[0].x, x1 = x0, y0 = job.v[0].y, y1 = y0;
    for (u32 i = 1; i < queued_ && i < 3; ++i) {
        x0 = std::min(x0, job.v[i].x); x1 = std::max(x1, job.v[i].x);
        y0 = std::min(y0, job.v[i].y); y1 = std::max(y1, job.v[i].y);
    }
    const double box = double(x1 - x0 + 1) * double(y1 - y0 + 1);
    job.pixels = (long long)std::min(1e9, type >= 3 && type <= 5 ? box / 2 : box);
    const Setup& s = job.s;
    const bool textured = Bits(s.prim, 4, 1) != 0;
    if (textured) {
        // Sampling what queued primitives write: they go first.
        for (const Job& queued : jobs_) {
            const Setup& q = queued.s;
            if ((s.tbp < q.frame_end && q.fbp < s.tex_end) || (s.tbp < q.z_end && q.zbp < s.tex_end)) {
                FinishJobs();
                break;
            }
        }
        // Sampling what it writes itself: drawn alone, in order.
        if ((s.tbp < s.frame_end && s.fbp < s.tex_end) || ((s.zte || !s.zmsk) && s.tbp < s.z_end && s.zbp < s.tex_end)) {
            FinishJobs();
            DrawJob(job, 0, 1);
            pixels_ += t_pixels;
            t_pixels = 0;
            return;
        }
    }
    jobs_.push_back(job);
    if (jobs_.size() >= kBatchPrimitives) FinishJobs();
}

void Gs::FinishJobs() {
    if (jobs_.empty()) return;
    Timed timed(Section::GsDraw);
    long long pixels = 0;
    for (const Job& job : jobs_) pixels += job.pixels;
    if (pixels < kParallelPixels) {
        for (const Job& job : jobs_) DrawJob(job, 0, 1);
        pixels_ += t_pixels;
        t_pixels = 0;
    } else {
        RowPool& pool = RowPool::Get();
        const int bands = pool.threads();
        pool.Run(0, bands, [&](int b0, int b1) {
            for (int band = b0; band < b1; ++band)
                for (const Job& job : jobs_) DrawJob(job, band, bands);
            pixels_ += t_pixels;
            t_pixels = 0;
        });
    }
    jobs_.clear();
}

void Gs::DrawJob(const Job& job, int band, int bands) {
    switch (job.s.prim & 7) {
        case 0: DrawLine(job, band, bands); break;      // a point: the line from its vertex to itself
        case 1: case 2: DrawLine(job, band, bands); break;
        case 3: case 4: case 5: DrawTriangle(job, band, bands); break;
        case 6: DrawSprite(job, band, bands); break;
        default: break;
    }
}

void Gs::DrawSprite(const Job& job, int band, int bands) {
    const Setup& st = job.s;
    const Vertex& a = job.v[0];
    const Vertex& b = job.v[1];
    const bool tme = Bits(st.prim, 4, 1), fst = Bits(st.prim, 8, 1);
    float ua, va, ub, vb;
    Coords(a, fst, st.tw, st.th, &ua, &va);
    Coords(b, fst, st.tw, st.th, &ub, &vb);
    const float x0 = std::min(a.x, b.x), x1 = std::max(a.x, b.x);
    const float y0 = std::min(a.y, b.y), y1 = std::max(a.y, b.y);
    const int ix0 = int(std::ceil(x0)), ix1 = int(std::ceil(x1));
    const int iy0 = int(std::ceil(y0)), iy1 = int(std::ceil(y1));
    const float du = x1 > x0 ? (ub - ua) / (b.x - a.x) : 0, dv = y1 > y0 ? (vb - va) / (b.y - a.y) : 0;
    // Only what the scissor lets through (Plot would drop the rest).
    const int cx0 = std::max({ix0, st.sx0, 0}), cx1 = std::min(ix1, st.sx1 + 1);
    const int cy0 = std::max({iy0, st.sy0, 0}), cy1 = std::min(iy1, st.sy1 + 1);
    for (int y = cy0; y < cy1; ++y) {
        if (!Owned(y, band, bands)) continue;
        for (int x = cx0; x < cx1; ++x) {
            const float u = ua + (float(x) - a.x) * du, v = va + (float(y) - a.y) * dv;
            Plot(st, x, y, b.z, b.r, b.g, b.b, b.a, u, v, tme, b.fog);
        }
    }
}

void Gs::DrawTriangle(const Job& job, int band, int bands) {
    const Setup& st = job.s;
    const Vertex& a = job.v[0];
    const Vertex& b = job.v[1];
    const Vertex& c = job.v[2];
    const bool iip = Bits(st.prim, 3, 1), tme = Bits(st.prim, 4, 1), fst = Bits(st.prim, 8, 1);
    const u32 tw = st.tw, th = st.th;
    const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (area == 0.0f) return;
    const int minx = std::max(int(std::floor(std::min({a.x, b.x, c.x}))), st.sx0);
    const int maxx = std::min(int(std::ceil(std::max({a.x, b.x, c.x}))), st.sx1);
    const int miny = std::max(int(std::floor(std::min({a.y, b.y, c.y}))), st.sy0);
    const int maxy = std::min(int(std::ceil(std::max({a.y, b.y, c.y}))), st.sy1);
    float ua = 0, va = 0, ub = 0, vb = 0, uc = 0, vc = 0;
    if (fst) { Coords(a, true, tw, th, &ua, &va); Coords(b, true, tw, th, &ub, &vb); Coords(c, true, tw, th, &uc, &vc); }
    // Edge functions in double: positions are 12.4 fixed point, so their
    // products are exact there and a pixel on an edge gives exactly 0. Such a
    // pixel is drawn only on a top or left edge (the GS's fill convention, as
    // its sprites' [ceil x0, ceil x1)): one where stepping right - or down, on
    // a horizontal edge - goes inside.
    const double ax = a.x, ay = a.y, bx = b.x, by = b.y, cx = c.x, cy = c.y;
    const double sign = area > 0 ? 1.0 : -1.0;
    const auto top_left = [&](double x0, double y0, double x1, double y1) {
        const double gx = (y0 - y1) * sign, gy = (x1 - x0) * sign;    // the inside's direction
        return gx > 0 || (gx == 0 && gy > 0);
    };
    const bool tl0 = top_left(bx, by, cx, cy), tl1 = top_left(cx, cy, ax, ay), tl2 = top_left(ax, ay, bx, by);
    for (int y = miny; y <= maxy; ++y) {
        if (!Owned(y, band, bands)) continue;
        for (int x = minx; x <= maxx; ++x) {
            const double px = x, py = y;
            const double e0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * sign;
            const double e1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * sign;
            const double e2 = ((ax - px) * (by - py) - (ay - py) * (bx - px)) * sign;
            if (e0 < 0 || e1 < 0 || e2 < 0) continue;
            if ((e0 == 0 && !tl0) || (e1 == 0 && !tl1) || (e2 == 0 && !tl2)) continue;
            const float w0 = float(e0 * sign / area), w1 = float(e1 * sign / area);
            const float w2 = 1.0f - w0 - w1;
            const Vertex& last = c;
            u8 r = last.r, g = last.g, bl = last.b, al = last.a;
            if (iip) {
                r = u8(w0 * a.r + w1 * b.r + w2 * c.r);
                g = u8(w0 * a.g + w1 * b.g + w2 * c.g);
                bl = u8(w0 * a.b + w1 * b.b + w2 * c.b);
                al = u8(w0 * a.a + w1 * b.a + w2 * c.a);
            }
            const u8 fog = u8(w0 * a.fog + w1 * b.fog + w2 * c.fog);
            // Depth in double, from the first vertex: 24 and 32-bit depths do not
            // survive float, and a flat 0xFFFFFF came out 0xFFFFFE, failing GEQUAL.
            const u32 z = u32(std::clamp(double(a.z) + double(w1) * (double(b.z) - double(a.z)) +
                                         double(w2) * (double(c.z) - double(a.z)), 0.0, 4294967295.0));
            if (fst) {
                Plot(st, x, y, z, r, g, bl, al, w0 * ua + w1 * ub + w2 * uc, w0 * va + w1 * vb + w2 * vc, tme, fog);
            } else {
                const float s = w0 * a.s + w1 * b.s + w2 * c.s, t = w0 * a.t + w1 * b.t + w2 * c.t;
                const float q = w0 * a.q + w1 * b.q + w2 * c.q;
                const float qq = q != 0.0f ? q : 1.0f;
                Plot(st, x, y, z, r, g, bl, al, s / qq * float(tw), t / qq * float(th), tme, fog, qq);
            }
        }
    }
}

void Gs::DrawLine(const Job& job, int band, int bands) {
    const Setup& st = job.s;
    // A point is the line from its (last) vertex to itself.
    const bool point = (st.prim & 7) == 0;
    const Vertex& a = point ? job.v[job.count ? job.count - 1 : 0] : job.v[0];
    const Vertex& b = point ? a : job.v[1];
    const bool iip = Bits(st.prim, 3, 1), tme = Bits(st.prim, 4, 1), fst = Bits(st.prim, 8, 1);
    const int steps = std::max(1, int(std::max(std::fabs(b.x - a.x), std::fabs(b.y - a.y))));
    float ua, va, ub, vb;
    Coords(a, fst, st.tw, st.th, &ua, &va);
    Coords(b, fst, st.tw, st.th, &ub, &vb);
    for (int i = 0; i <= steps; ++i) {
        const float f = float(i) / float(steps);
        const auto mix = [&](float p, float q) { return p + (q - p) * f; };
        const int y = int(std::lround(mix(a.y, b.y)));
        if (!Owned(y, band, bands)) continue;
        const u8 r = iip ? u8(mix(a.r, b.r)) : b.r, g = iip ? u8(mix(a.g, b.g)) : b.g;
        const u8 bl = iip ? u8(mix(a.b, b.b)) : b.b, al = iip ? u8(mix(a.a, b.a)) : b.a;
        Plot(st, int(std::lround(mix(a.x, b.x))), y, u32(double(a.z) + (double(b.z) - double(a.z)) * double(f)),
             r, g, bl, al, mix(ua, ub), mix(va, vb), tme, u8(mix(a.fog, b.fog)));
    }
}

// ---------------------------------------------------------------- pixels

// The registers a primitive's pixels use, decoded once.
void Gs::Prepare() {
    const Context& c = Ctx();
    const u64 prim = Prim(), test = c.test;
    Setup& s = setup_;
    setup_stale_ = false;
    s.prim = prim;
    s.texa = texa_;
    s.clut = clut_;
    s.sx0 = int(Bits(c.scissor, 0, 11));
    s.sx1 = int(Bits(c.scissor, 16, 11));
    s.sy0 = int(Bits(c.scissor, 32, 11));
    s.sy1 = int(Bits(c.scissor, 48, 11));
    s.tfx = Bits(c.tex0, 35, 2);
    s.tcc = Bits(c.tex0, 34, 1) != 0;
    const int k = int(s32(Bits(c.tex1, 32, 12) << 20) >> 20);
    const u32 mmin = Bits(c.tex1, 6, 3);
    s.linear = k <= 0 ? Bits(c.tex1, 5, 1) != 0 : (mmin == 1 || mmin == 4 || mmin == 5);
    s.mip = MipMode(c);
    s.levels = LevelsOf(c);
    s.tbp = Bits(c.tex0, 0, 14);
    s.tbw = Bits(c.tex0, 14, 6);
    s.tpsm = Bits(c.tex0, 20, 6);
    s.tw = 1u << std::min(Bits(c.tex0, 26, 4), 10u);
    s.th = 1u << std::min(Bits(c.tex0, 30, 4), 10u);
    s.csa = Bits(c.tex0, 56, 5);
    s.wms = Bits(c.clamp, 0, 2);
    s.wmt = Bits(c.clamp, 2, 2);
    s.minu = Bits(c.clamp, 4, 10);
    s.maxu = Bits(c.clamp, 14, 10);
    s.minv = Bits(c.clamp, 24, 10);
    s.maxv = Bits(c.clamp, 34, 10);
    s.ate = Bits(test, 0, 1) != 0;
    s.atst = Bits(test, 1, 3);
    s.aref = Bits(test, 4, 8);
    s.afail = Bits(test, 12, 2);
    s.date = Bits(test, 14, 1) != 0;
    s.datm = Bits(test, 15, 1);
    s.zte = Bits(test, 16, 1) != 0;
    s.ztst = Bits(test, 17, 2);
    s.zbp = Bits(c.zbuf, 0, 9) * 32;
    s.zpsm = Bits(c.zbuf, 24, 4) | 0x30;
    s.zmsk = Bits(c.zbuf, 32, 1) != 0;
    s.fbp = Bits(c.frame, 0, 9) * 32;
    s.fbw = Bits(c.frame, 16, 6);
    s.fpsm = Bits(c.frame, 24, 6);
    s.fbmsk = u32(c.frame >> 32);
    s.abe = Bits(prim, 6, 1) != 0;
    s.pabe = Bits(pabe_, 0, 1) != 0;
    s.colclamp = Bits(colclamp_, 0, 1) != 0;
    s.fba = Bits(c.fba, 0, 1) != 0;
    s.a = Bits(c.alpha, 0, 2);
    s.b = Bits(c.alpha, 2, 2);
    s.c = Bits(c.alpha, 4, 2);
    s.d = Bits(c.alpha, 6, 2);
    s.fix = Bits(c.alpha, 32, 8);
    s.fge = Bits(prim, 5, 1) != 0;
    s.fcr = Bits(fogcol_, 0, 8);
    s.fcg = Bits(fogcol_, 8, 8);
    s.fcb = Bits(fogcol_, 16, 8);
    // The blocks it reads (its texture) and writes (frame and depth, down to
    // the scissor's last row).
    const u32 rows = u32(std::max(0, s.sy1 + 1));
    s.tex_end = s.tbp + std::max(1u, s.tbw * 64 * s.th * BitsPerPixel(s.tpsm) / 8 / 256);
    for (u32 n = 1; n < s.levels.count; ++n)        // and its mipmap levels'
        s.tex_end = std::max(s.tex_end, s.levels.tbp[n] + std::max(1u, std::max(1u, s.levels.tbw[n]) * 64 *
                                                                            std::max(1u, s.th >> n) *
                                                                            BitsPerPixel(s.tpsm) / 8 / 256));
    s.frame_end = s.fbp + std::max(1u, s.fbw * 64 * rows * BitsPerPixel(s.fpsm) / 8 / 256);
    s.z_end = s.zbp + std::max(1u, s.fbw * 64 * rows * BitsPerPixel(s.zpsm) / 8 / 256);
}

// TEX1: MMAG (bit 5) filters magnification, MMIN (6-8) minification, chosen by
// the level of detail: K (bits 32-43, 4 fractional bits) alone for the whole
// primitive, or per pixel by its Q when mipmapped (MipMode, ChooseLevel).
u32 Gs::Texel(const Setup& st, float fu, float fv, float q) const {
    if (!st.mip) return Texel(st, fu, fv, 0, st.linear);
    const MipChoice m = ChooseLevel(st.mip, q);
    const u32 c0 = Texel(st, fu, fv, m.n0, m.linear);
    if (m.n1 == m.n0) return c0;
    const u32 c1 = Texel(st, fu, fv, m.n1, m.linear);
    u32 out = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        const float a = float((c0 >> shift) & 0xFF), b = float((c1 >> shift) & 0xFF);
        out |= u32(std::floor(a + (b - a) * m.f + 0.5f)) << shift;
    }
    return out;
}

// Level n's texel at (u, v) in level 0's texels: the nearest, or (linear) the
// four around (u - 0.5, v - 0.5), as the GS does.
u32 Gs::Texel(const Setup& st, float fu, float fv, int n, bool linear) const {
    if (n) {
        fu /= float(1 << n);
        fv /= float(1 << n);
    }
    if (!linear) return TexelAt(st, int(std::floor(fu)), int(std::floor(fv)), n);
    // The GS's coordinates have 4 fractional bits: the weights are sixteenths,
    // and each blend truncates. A game that filters its own last frame back
    // in, over and over, relies on that: Mister Mosquito's cutscene blur lets
    // an old picture fade only because each pass loses a little (rounded to
    // the nearest, a blurred picture stood for good).
    const int x = int(std::floor((fu - 0.5f) * 16.0f)), y = int(std::floor((fv - 0.5f) * 16.0f));
    const int u0 = x >> 4, v0 = y >> 4, a = x & 15, b = y & 15;
    const u32 t00 = TexelAt(st, u0, v0, n), t10 = TexelAt(st, u0 + 1, v0, n);
    const u32 t01 = TexelAt(st, u0, v0 + 1, n), t11 = TexelAt(st, u0 + 1, v0 + 1, n);
    u32 out = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        const int c00 = int((t00 >> shift) & 0xFF), c10 = int((t10 >> shift) & 0xFF);
        const int c01 = int((t01 >> shift) & 0xFF), c11 = int((t11 >> shift) & 0xFF);
        const int top = c00 + (((c10 - c00) * a) >> 4), bottom = c01 + (((c11 - c01) * a) >> 4);
        out |= u32(top + (((bottom - top) * b) >> 4)) << shift;
    }
    return out;
}

u32 Gs::TexelAt(const Setup& s, int iu, int iv, int n) const {
    u32 raw;
    if (n == 0) {
        const int u = Wrap(iu, s.tw, s.wms, s.minu, s.maxu);
        const int v = Wrap(iv, s.th, s.wmt, s.minv, s.maxv);
        raw = memory.Read(s.tbp, s.tbw, u32(u) & 2047, u32(v) & 2047, s.tpsm);
    } else {
        // Level n: TW-n by TH-n, a region's bounds shifted as its coordinates.
        const u32 w = std::max(1u, s.tw >> n), h = std::max(1u, s.th >> n);
        const int u = std::clamp(Wrap(iu, w, s.wms, s.minu >> n, s.maxu >> n), 0, int(w) - 1);
        const int v = std::clamp(Wrap(iv, h, s.wmt, s.minv >> n, s.maxv >> n), 0, int(h) - 1);
        raw = memory.Read(s.levels.tbp[n], std::max(1u, s.levels.tbw[n]), u32(u), u32(v), s.tpsm);
    }
    // A Z-format texture reads as its CT counterpart (Burnout 3's race reads
    // its depth buffer as a PSMZ16 texture, into the frame's alpha).
    switch (s.tpsm) {
        case PSMCT32: case PSMZ32: return raw;
        case PSMCT24: case PSMZ24: {
            const u32 rgb = raw & 0xFFFFFF;
            const u32 a = (Bits(s.texa, 15, 1) && !rgb) ? 0 : Bits(s.texa, 0, 8);
            return rgb | a << 24;
        }
        case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: return From16(raw, s.texa);
        case PSMT8: case PSMT8H: return (*s.clut)[raw & 0xFF];
        case PSMT4: case PSMT4HL: case PSMT4HH: return (*s.clut)[(s.csa * 16 + (raw & 15)) & 0xFF];
        default: return raw;
    }
}

void Gs::Plot(const Setup& st, int x, int y, u32 z, u8 r, u8 g, u8 b, u8 a, float u, float v, bool textured,
              u8 fog, float q) {
    // Scissor.
    if (x < st.sx0 || x > st.sx1 || y < st.sy0 || y > st.sy1 || x < 0 || y < 0) return;
    u32 cr = r, cg = g, cb = b, ca = a;
    if (textured) {
        const u32 texel = Texel(st, u, v, q);
        const u32 tr = texel & 0xFF, tg = (texel >> 8) & 0xFF, tb = (texel >> 16) & 0xFF, ta = texel >> 24;
        const bool tcc = st.tcc;
        switch (st.tfx) {
            case 0:                                    // MODULATE
                cr = std::min(255u, tr * r >> 7); cg = std::min(255u, tg * g >> 7); cb = std::min(255u, tb * b >> 7);
                ca = tcc ? std::min(255u, ta * a >> 7) : a;
                break;
            case 1:                                    // DECAL
                cr = tr; cg = tg; cb = tb; ca = tcc ? ta : a;
                break;
            case 2:                                    // HIGHLIGHT
                cr = std::min(255u, (tr * r >> 7) + a); cg = std::min(255u, (tg * g >> 7) + a);
                cb = std::min(255u, (tb * b >> 7) + a); ca = tcc ? std::min(255u, ta + a) : a;
                break;
            default:                                   // HIGHLIGHT2
                cr = std::min(255u, (tr * r >> 7) + a); cg = std::min(255u, (tg * g >> 7) + a);
                cb = std::min(255u, (tb * b >> 7) + a); ca = tcc ? ta : a;
                break;
        }
    }
    // Fog (PRIM.FGE): towards FOGCOL as F falls, colour only.
    if (st.fge) {
        cr = (fog * cr + (255 - fog) * st.fcr) >> 8;
        cg = (fog * cg + (255 - fog) * st.fcg) >> 8;
        cb = (fog * cb + (255 - fog) * st.fcb) >> 8;
    }
    // Alpha test.
    bool write_rgb = true, write_alpha = true, write_z = true;
    if (st.ate) {
        const u32 ref = st.aref;
        bool pass = true;
        switch (st.atst) {
            case 0: pass = false; break;
            case 1: pass = true; break;
            case 2: pass = ca < ref; break;
            case 3: pass = ca <= ref; break;
            case 4: pass = ca == ref; break;
            case 5: pass = ca >= ref; break;
            case 6: pass = ca > ref; break;
            default: pass = ca != ref; break;
        }
        if (!pass) {
            switch (st.afail) {
                case 0: return;                                         // KEEP
                case 1: write_z = false; break;                         // FB_ONLY
                case 2: write_rgb = write_alpha = false; break;         // ZB_ONLY
                default: write_alpha = false; write_z = false; break;   // RGB_ONLY
            }
        }
    }
    const u32 fbp = st.fbp, fbw = st.fbw, fpsm = st.fpsm;
    // Destination alpha test: the frame's alpha bit (31 for 32-bit, 15 for 16-bit
    // formats; 24-bit frames have none) must equal DATM - a stencil games draw
    // into alpha first.
    if (st.date && fpsm != PSMCT24) {
        const u32 here = memory.Read(fbp, fbw, u32(x), u32(y), fpsm);
        const u32 bit = (fpsm == PSMCT16 || fpsm == PSMCT16S) ? (here >> 15) & 1 : here >> 31;
        if (bit != st.datm) return;
    }
    // Depth test.
    const u32 zbp = st.zbp, zpsm = st.zpsm;
    if (st.zte) {
        const u32 old = memory.Read(zbp, fbw, u32(x), u32(y), zpsm);
        bool pass = true;
        switch (st.ztst) {
            case 0: pass = false; break;
            case 1: pass = true; break;
            case 2: pass = z >= old; break;
            default: pass = z > old; break;
        }
        if (!pass) return;
    } else {
        write_z = false;
    }
    if (st.zmsk) write_z = false;
    // Blending.
    u32 dest = memory.Read(fbp, fbw, u32(x), u32(y), fpsm);
    if (fpsm == PSMCT16 || fpsm == PSMCT16S) dest = From16(dest, 0x80ull << 32);
    if (fpsm == PSMCT24) dest |= 0x80u << 24;
    const bool blend = st.abe && !(st.pabe && !(ca & 0x80));
    if (blend) {
        const u32 dr = dest & 0xFF, dg = (dest >> 8) & 0xFF, db = (dest >> 16) & 0xFF, da = dest >> 24;
        const int factor = st.c == 0 ? int(ca) : st.c == 1 ? int(da) : int(st.fix);
        auto pick = [&](u32 sel, u32 src, u32 dst) { return sel == 0 ? int(src) : sel == 1 ? int(dst) : 0; };
        auto mix = [&](u32 src, u32 dst) {
            int value = ((pick(st.a, src, dst) - pick(st.b, src, dst)) * factor >> 7) + pick(st.d, src, dst);
            return st.colclamp ? u32(std::clamp(value, 0, 255)) : u32(value) & 0xFF;
        };
        cr = mix(cr, dr);
        cg = mix(cg, dg);
        cb = mix(cb, db);
    }
    if (st.fba) ca |= 0x80;
    u32 out = Rgba(cr, cg, cb, ca);
    u32 mask = st.fbmsk;
    if (!write_rgb) mask |= 0x00FFFFFF;
    if (!write_alpha) mask |= 0xFF000000;
    out = (out & ~mask) | (dest & mask);
    if (mask != 0xFFFFFFFF) {
        if (fpsm == PSMCT16 || fpsm == PSMCT16S) memory.Write(fbp, fbw, u32(x), u32(y), fpsm, To16(out));
        else memory.Write(fbp, fbw, u32(x), u32(y), fpsm, out);
        ++t_pixels;
    }
    if (write_z) memory.Write(zbp, fbw, u32(x), u32(y), zpsm, z);
}

}  // namespace gs
}  // namespace ps2host
