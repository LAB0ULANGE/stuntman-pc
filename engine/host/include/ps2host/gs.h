/*
 * The Graphics Synthesizer, stage one: a software GS.
 *
 * GIF packets from the three paths (PATH1 from VU1, not yet; PATH2 from VIF1's
 * DIRECT; PATH3 from the GIF's own DMA) set GS registers, upload images into
 * the GS's 4 MB of local memory, and draw primitives into it; the display
 * registers say which part of local memory is the picture. Frames are written
 * out as BMP files on request (--frames), which is how the port's progress is
 * seen until there is a window.
 *
 * Local memory is laid out as the GS lays it out - pages, blocks and columns
 * in the order its manual gives for each pixel format (gs_memory.cpp) - since
 * games address it directly: a texture uploaded as one format and read as
 * another must come out as on the console.
 */

#ifndef PS2HOST_GS_H
#define PS2HOST_GS_H

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ps2host {
namespace gs {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s32 = std::int32_t;

// Pixel storage formats (PSM).
enum Psm : u32 {
    PSMCT32 = 0x00, PSMCT24 = 0x01, PSMCT16 = 0x02, PSMCT16S = 0x0A,
    PSMT8 = 0x13, PSMT4 = 0x14, PSMT8H = 0x1B, PSMT4HL = 0x24, PSMT4HH = 0x2C,
    PSMZ32 = 0x30, PSMZ24 = 0x31, PSMZ16 = 0x32, PSMZ16S = 0x3A,
};

/* Local memory: 4 MB, addressed by (base pointer in 256-byte blocks, buffer
   width in 64-pixel units, x, y, format). */
class Memory {
public:
    Memory() : bytes_(4u << 20) {}
    u32 Read(u32 bp, u32 bw, u32 x, u32 y, u32 psm) const;
    void Write(u32 bp, u32 bw, u32 x, u32 y, u32 psm, u32 value);
    /* `count` pixels of row y from x on (no wrap past 2047), as Write each. */
    void WriteRun(u32 bp, u32 bw, u32 x, u32 y, u32 psm, const u32* values, u32 count);
    u8* data() { return bytes_.data(); }
    const u8* data() const { return bytes_.data(); }

    /* The byte address of a pixel (for 4-bit formats, of its byte; *nibble
       says which half). Exposed for tests. */
    static u32 Address(u32 bp, u32 bw, u32 x, u32 y, u32 psm, u32* nibble);
    /* The same from the manual's block and column tables, step by step:
       Address is it precomputed a page at a time (tests compare the two). */
    static u32 AddressReference(u32 bp, u32 bw, u32 x, u32 y, u32 psm, u32* nibble);

private:
    std::vector<u8> bytes_;
};

struct Vertex {
    float x = 0, y = 0;         // pixels, after XYOFFSET
    u32 z = 0;
    u8 r = 0, g = 0, b = 0, a = 0;
    float s = 0, t = 0, q = 1;  // ST/Q
    u32 u = 0, v = 0;           // UV, 1/16 texel
    u8 fog = 0;
};

struct Context {                // one of the two drawing contexts
    u64 tex0 = 0, clamp = 0, tex1 = 0, xyoffset = 0, scissor = 0, alpha = 0;
    u64 test = 0, fba = 0, frame = 0, zbuf = 0;
    u64 miptbp1 = 0, miptbp2 = 0;   // mipmap levels 1-3 and 4-6: TBP and TBW of each
};

/* TEX1's mipmapping. The levels a primitive samples: MXL + 1 when MMIN picks
   among them (2-5), else level 0 alone - and where each lies: MIPTBP1/2's TBP
   and TBW, or for levels 1-3 with MTBA each straight after the one before
   (TBW halved, at least 1). Level n is TW-n by TH-n. */
struct MipLevels {
    u32 count = 1;
    u32 tbp[7] = {}, tbw[7] = {};
};
MipLevels LevelsOf(const Context& c);
/* How a pixel picks among them, packed (the GPU's GsDrawState::mip): 0 none -
   MMAG or MMIN by K alone, as the whole primitive - else bit 1, MXL in bits
   2-4, MMIN 5-7, MMAG 8, LCM 9, L 10-11, K 16-27. A primitive of K alone that
   never minifies has none. */
u32 MipMode(const Context& c);
/* A pixel's choice by its Q: level n0 (and n1, mixed by f, or n1 == n0), each
   filtered bilinear or not - the GS's LOD, (log2(1/|Q|) << L) + K or K alone
   (LCM), magnifying (MMAG, level 0) up to 0, else MMIN: 2 and 4 the nearest
   level, 3 and 5 the two around; 4 and 5 bilinear within one. gs.frag's
   Sample does the same. */
struct MipChoice {
    int n0 = 0, n1 = 0;
    float f = 0.0f;
    bool linear = false;
};
MipChoice ChooseLevel(u32 mode, float q);

class Gs {
public:
    Memory memory;
    /* --gs-trace: log image transfers and primitives (stderr). */
    bool trace = false;
    std::FILE* trace_out = stderr;      // where the trace goes (a capture's trace.txt)

    /* A GS register write by address (A+D, REGLIST, or the GIF's own). */
    void WriteRegister(u32 address, u64 value);

    /* A quadword of GIF data on a path (1, 2 or 3). */
    void Qword(int path, u64 low, u64 high);
    /* 32-bit words, gathered into quadwords. */
    void Word(int path, u32 word);
    /* `count` words, as Word each - g_dma_source following them from guest
       address `from` (kNoSource: left alone) - except that an IMAGE packet's
       whole quadwords go to the transfer together. */
    static constexpr u32 kNoSource = ~0u;
    void Words(int path, const u32* words, u32 count, u32 from = kNoSource);
    /* Recording a GS stream as --gs-replay reads it (GSRP1): local memory and
       the registers now, then every GIF word and privileged register write,
       and Blank's vertical blanks, until EndRecord. Memory must be current
       (Finish, the GPU's drawing brought back). */
    void BeginRecord(std::FILE* out);
    void RecordBlank();
    void EndRecord();

    /* Privileged registers (the 0x12000000 window): what is displayed. */
    void SetPrivileged(u32 address, u64 value);

    /* The displayed picture, 32-bit BGRA, and its size. False if nothing is.
       In field mode (SMODE2 INT and FFMD) each field shows every line of the
       frame buffer, half the display's height: those lines, each twice - or,
       with `weave`, on the lines of `weave_parity` (0 the even ones) of the
       picture, the field before on the others, as a television interlaces
       them: a game drawing each field half a line apart (Stuntman) shows its
       full height, still. */
    bool Picture(std::vector<u32>* pixels, u32* width, u32* height) const;
    bool weave = false;
    u32 weave_parity = 0;
    /* How many lines of the frame buffer the display shows. */
    u32 ShownLines() const;
    /* Interlaced, field mode (SMODE2 INT and FFMD): a field a frame. */
    bool FieldMode() const;
    bool WriteBmp(const std::string& path) const;

    /* Every primitive as it is drawn, for a second renderer (the GPU's, which
       --gpu-shadow checks against this one): PRIM, the context it draws
       with, TEXA, and its vertices (3 for a triangle, 2 for a sprite). */
    struct DrawCall {
        u64 prim;
        const Context* context;
        u64 texa;
        const Vertex* vertices;
        u32 count;
        u64 fogcol;
        bool pabe;
    };
    std::function<void(const DrawCall&)> on_draw;
    /* --gpu: the GS model draws nothing itself (rasterize false) and a second
       renderer draws instead; to keep its images and local memory in step it
       hears of every other access to local memory first - blocks
       [begin, end), read or written: image transfers, local copies, CLUT
       loads, the displayed picture. */
    bool rasterize = true;
    /* PS2HOST_GS_DIGEST: what the GS is given - every register write, every
       transferred quadword, every privileged register - hashed in order
       (FNV-1a), so two runs can be shown to have fed it the same. */
    bool digesting = false;
    u64 digest = 0xCBF29CE484222325ull;
    void Mix(u64 v) { digest = (digest ^ v) * 0x100000001B3ull; }
    /* Words: whole quadwords straight to Qword (true, the gs_whole_qwords
       setting's default), or each word through Word as before. */
    bool whole_qwords = true;
    std::function<void(u32 begin, u32 end, bool write)> on_memory;
    /* An image transfer about to write blocks [begin, end), when set (else
       on_memory): cover[i] for block begin + i - 0 not written, 1 written in
       part, 2 written whole. A block written whole needs nothing of what was
       there before. */
    std::function<void(u32 begin, u32 end, const u8* cover)> on_write;
    /* --gpu: a CLUT load (TEX0 with its CLD, and TEXA) about to happen; true
       if the second renderer made it from its own copy of local memory - the
       model's CLUT then keeps its old entries for those (on_memory is not
       called, and nothing reads them but that renderer). */
    std::function<bool(u64 tex0, u64 texa)> on_clut_load;
    /* A transfer into local memory that Touch announced is done: its last
       pixel written (the GPU bridge patches what it holds newer then). */
    std::function<void()> on_transfer_done;
    /* The texture the primitive being drawn samples, decoded as the software
       rasteriser reads it (CLUT, TEXA): RGBA in memory, rows top-down. */
    void DecodeTexture(std::vector<u32>* pixels, u32* width, u32* height) const;
    /* The CLUT as the last TEX0 loaded it. */
    const std::array<u32, 256>& Clut() const { return *clut_; }
    /* Changes whenever a CLUT is loaded. */
    u64 clut_loads() const { return clut_loads_; }
    /* Which frame buffer is displayed: FBP (as FRAME has it), FBW, PSM. */
    bool Displayed(u32* fbp, u32* fbw, u32* psm) const;
    /* The displayed buffer's first line (DISPFB's DBY). */
    u32 DisplayedDby() const;
    /* A pixel of a frame buffer as RGBA, whatever its format (alpha 0x80 for
       24-bit formats). */
    u32 FramePixel(u32 fbp, u32 fbw, u32 psm, u32 x, u32 y) const;

    /* Changes whenever an image transfer or a local copy writes local memory. */
    u64 uploads() const { return uploads_; }

    u64 primitives() const { return primitives_; }
    u64 pixels_drawn() const { Finish(); return pixels_; }

    /* Draws what is queued. Primitives are queued and drawn in batches, rows
       spread over threads (each pixel still sees them in order); anything
       that reads local memory - the picture, transfers, CLUT loads - finishes
       them first, and so must a caller reading `memory` directly. */
    void Finish() const;
    /* For the report: GIF tags per path, register writes, image data. */
    std::string Summary() const;

private:
    struct Path {
        u64 tag_low = 0, tag_high = 0;
        u32 loops = 0, reg = 0, nregs = 0, format = 0;
        bool in_packet = false;
        u32 pending[4] = {};
        u32 pending_count = 0;
    };
    Path paths_[4];
    /* The drawing state a primitive's pixels all use, decoded from the
       registers once per primitive (Prepare) rather than once per pixel, and
       kept with the primitive while it waits to be drawn. */
    struct Setup {
        u64 prim, texa;                             // PRIM (with PRMODE), TEXA
        std::shared_ptr<const std::array<u32, 256>> clut;
        u32 tex_end, frame_end, z_end;              // the blocks it reads and writes, exclusive
        int sx0, sx1, sy0, sy1;                     // SCISSOR, inclusive
        u32 tfx;
        bool tcc, linear;
        u32 tbp, tbw, tpsm, tw, th, csa;            // TEX0
        u32 mip;                                    // MipMode: levels 1 and on in `levels`
        MipLevels levels;
        u32 wms, wmt, minu, maxu, minv, maxv;       // CLAMP
        bool ate, date, zte, zmsk, abe, colclamp, fba, pabe, fge;
        u32 fcr, fcg, fcb;                          // FOGCOL
        u32 atst, aref, afail, datm;                // TEST
        u32 ztst, zbp, zpsm;                        // TEST, ZBUF
        u32 fbp, fbw, fpsm, fbmsk;                  // FRAME
        u32 a, b, c, d, fix;                        // ALPHA
    };
    Setup setup_{};
    bool setup_stale_ = false;
    u32 dma_source_ = 0;                // where the word being taken came from (--gs-trace)          // !rasterize: Prepare left to DecodeTexture, which alone reads it
    void Prepare();
    /* A primitive waiting to be drawn. */
    struct Job {
        Setup s;
        Vertex v[3];
        u32 count;
        long long pixels;                       // about how many, for spreading the work
    };
    std::vector<Job> jobs_;
    void FinishJobs();
    /* One primitive, the rows of band `band` of `bands` (rows in groups of 8
       dealt round). */
    void DrawJob(const Job& job, int band, int bands);
    void Packed(u32 descriptor, u64 low, u64 high);
    void Kick(bool draw);
    void Draw();
    void DrawSprite(const Job& job, int band, int bands);
    void DrawTriangle(const Job& job, int band, int bands);
    void DrawLine(const Job& job, int band, int bands);
    void Plot(const Setup& st, int x, int y, u32 z, u8 r, u8 g, u8 b, u8 a, float u, float v, bool textured, u8 fog,
              float q = 1.0f);
    u32 Texel(const Setup& st, float u, float v, float q) const;     /* filtered as TEX1 says, by Q */
    u32 Texel(const Setup& st, float u, float v, int n, bool linear) const;    /* in level n */
    u32 TexelAt(const Setup& st, int u, int v, int n = 0) const;      /* one texel, wrapped or clamped */
    void Transfer(u64 low, u64 high);           // HWREG data: host to local
    void TransferBlock(const u8* data, u32 qwords);     // IMAGE data, quadwords of it
    /* Tells on_memory of a rectangle of a buffer about to be read or written. */
    void Touch(u32 bp, u32 bw, u32 psm, u32 x, u32 y, u32 w, u32 h, bool write) const;
public:
    /* The transfer Touch last looked at (PS2HOST_GPU_WRITE_LOG: what made the
       GPU's memory come back for a write). */
    struct TouchSeen { u32 bp, bw, psm, x, y, w, h; bool write; };
    mutable TouchSeen last_transfer{};
private:
    void LocalToLocal();

    std::FILE* record_ = nullptr;               // BeginRecord's stream
    Context ctx_[2];
    u64 prim_ = 0, prmode_ = 0, prmodecont_ = 1, rgbaq_ = 0, st_ = 0, uv_ = 0;
    u8 fog_ = 0;                                // FOG: F for XYZ2/XYZ3 vertices
    u64 texclut_ = 0, texa_ = 0, fogcol_ = 0, dthe_ = 0, colclamp_ = 1, pabe_ = 0;
    u64 bitbltbuf_ = 0, trxpos_ = 0, trxreg_ = 0, trxdir_ = 3;
    float q_ = 1;
    u32 rgba_source_ = 0;                       // where the last RGBAQ came from (trace)
    u32 xyz_source_ = 0;                        // ...and the last vertex (trace)
    Vertex queue_[3];
    u32 queued_ = 0;
    u32 strip_ = 0;                             // vertices since the primitive started
    // Image transfer progress.
    u32 trx_x_ = 0, trx_y_ = 0;
    bool trx_active_ = false;
    u32 trx_partial_ = 0, trx_partial_bytes_ = 0;    // 24-bit pixels across quadwords
    u32 cbp0_ = 0, cbp1_ = 0;                       // CLUT bases CLD compares with
    // Privileged.
    u64 pmode_ = 0, smode2_ = 0, dispfb_[2] = {}, display_[2] = {}, bgcolor_ = 0;
    mutable std::vector<u32> woven_;            // Picture's, with `weave`: both fields
    mutable u32 woven_width_ = 0, woven_height_ = 0;
    u64 primitives_ = 0, uploads_ = 0;
    std::atomic<u64> pixels_{0};               // written by the row workers too
    u64 tags_[4] = {}, image_qwords_ = 0;
    std::array<u64, 256> register_writes_{};
    // CLUT, loaded per TEX0's CLD; copied when it changes, so that queued
    // primitives keep the one they were drawn with.
    std::shared_ptr<std::array<u32, 256>> clut_ = std::make_shared<std::array<u32, 256>>();
    u64 clut_loads_ = 0;
    void LoadClut(u64 tex0);
    u64 Prim() const { return (prmodecont_ & 1) ? prim_ : ((prim_ & 7) | (prmode_ & ~7ull)); }
    const Context& Ctx() const { return ctx_[(Prim() >> 9) & 1]; }
};

/* The one GS. Inline, through a pointer set when it is made: it is asked
   for once a word of every GIF transfer. */
namespace detail {
extern Gs* g_the_gs;
Gs& MakeGs();
}  // namespace detail
inline Gs& TheGs() {
    Gs* gs = detail::g_the_gs;
    return gs ? *gs : detail::MakeGs();
}

}  // namespace gs
}  // namespace ps2host

#endif  /* PS2HOST_GS_H */
