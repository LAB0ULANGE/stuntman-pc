/*
 * GsRenderer on Vulkan (gs_renderer.h): triangles batched by state and drawn
 * with dynamic rendering into R8G8B8A8 targets, one per (FBP, FBW, PSM).
 *
 * One vertex and one fragment shader (shaders/gs.vert, gs.frag) do the GS's
 * per-pixel work: the texture function, the alpha test. Blending is the
 * fixed-function blender's, with the GS's factor - source alpha or FIX, where
 * 0x80 is 1.0 - as the fragment's second output (dual-source blending): the
 * GS's Cv = (A - B) * C + D becomes a source and a destination factor and an
 * operation, one pipeline per ALPHA setting, made when first used.
 *
 * Everything is synchronous for now - a flush records, submits and waits - which
 * keeps the first steps simple to check; batching across frames and keeping the
 * GPU busy come once the output matches the software GS.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

#include "gpu_internal.h"
#include "gs_frag.h"
#include "gs_frag_single.h"
#include "gs_frag_blend.h"
#include "gs_vert.h"
#include "date_frag.h"
#include "date_local_frag.h"
#include "date_check_frag.h"
#include "date_vert.h"
#include "swizzle_comp.h"
#include "swizzle_image_comp.h"
#include "swizzle_depth_comp.h"
#include "depth_load_frag.h"
#include "decode_comp.h"
#include "patch_comp.h"

namespace ps2gpu {

namespace {

struct PackedVertex {
    float x, y;
    std::uint8_t r, g, b, a;
    float s, t, q;
    std::uint32_t z;
    std::uint8_t fog, pad[3];
    float range[4];             // a sprite's S, T at its first and last pixels (gs.vert)
    float pad2[4];
};
// The vertex ring hands out 256-byte aligned runs, a batch's first vertex
// found by its offset: whole vertices to 256 bytes.
static_assert(256 % sizeof(PackedVertex) == 0, "a vertex must divide the ring's alignment");

/* What the fragment shader is told, after the vertex shader's 16 bytes. */
struct FragmentParams {
    std::uint32_t textured, tfx, tcc, atst, aref, fix_factor, fix, afail, fge, fogcol, pabe;
    std::uint32_t bilinear, wms, wmt, minu, maxu, minv, maxv;
    std::uint32_t tw, th, tex_format, ta0, ta1, aem;
    std::uint32_t logic, keep;
    std::uint32_t scale;        // the texture's pixels to a GS texel (a target drawn larger: the internal resolution)
    std::uint32_t blend_ops;    // bits 0-7: blending in the shader, ALPHA's A, B, C, D, two bits each;
                                // 8-15: a sprite on a target drawn larger, the target's scale (0: none)
};
constexpr std::uint32_t kFragmentOffset = 16;

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // Where in `memory` (ImageMemory: a piece of a block, of a size class -
    // class 0 its own allocation), and of which memory type.
    VkDeviceSize offset = 0;
    std::uint32_t piece_class = 0, memory_type = 0;
    VkImageView view = VK_NULL_HANDLE;
    std::uint32_t width = 0, height = 0;    // in GS pixels
    std::uint32_t scale = 1;                // the image's pixels to a GS pixel, each way (targets, depth)
    std::uint32_t levels = 1;               // mipmap levels (a texture pack's replacements)
    std::uint32_t pw() const { return width * scale; }
    std::uint32_t ph() const { return height * scale; }
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkDescriptorSet sampled = VK_NULL_HANDLE;   /* a target read by the DATE setup pass */
    VkDescriptorSet storage = VK_NULL_HANDLE;   /* a target as the swizzle's storage image */
    VkImageView depth_view = VK_NULL_HANDLE;    /* a depth buffer's depth alone, to sample */
    VkDescriptorSet depth_sampled = VK_NULL_HANDLE;     /* ...for the swizzle */
    VkDescriptorSet input = VK_NULL_HANDLE;     /* a target read in its own pass (date_local.frag) */
};

// Depth with stencil: the stencil carries the destination alpha test.
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

struct Texture {
    Image image;
    VkDescriptorSet sets[8] = {};       /* by sampler */
    /* A texture drawn as a texture pack's replacement (UseReplacement): its
       id, 0 for none. A replacement itself (replacements_): gs.frag's
       tex_format for it - bit 8, levels - 1 in bits 9-11 - when it was last
       drawn (frames, Kick's), its size in memory. */
    std::uint64_t replacement = 0;
    std::uint32_t replaced = 0;
    std::uint64_t last_used = 0;
    VkDeviceSize bytes = 0;
};

// What the frames' command buffers hold (RendererCounters, PS2HOST_GPU_LOG).
std::uint64_t g_barriers = 0, g_transitions = 0, g_dispatches = 0, g_draws = 0, g_passes = 0, g_feedback = 0;

void Transition(VkCommandBuffer cmd, Image& t, VkImageLayout to, VkAccessFlags dst_access,
                VkPipelineStageFlags dst_stage) {
    if (t.layout == to) return;
    ++g_transitions;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = t.layout;
    b.newLayout = to;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = dst_access;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {t.aspect, 0, t.levels, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    t.layout = to;
}

/* Everything written so far, visible to everything after. */
void MemoryBarrier(VkCommandBuffer cmd) {
    ++g_barriers;
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

/* decode.comp's push constants: GsTextureDecode without the CLUT. */
struct DecodeParams {
    std::uint32_t tbp, tbw, psm, tw, th, wms, wmt, minu, maxu, minv, maxv, texa_lo, ta1, csa;
    std::uint32_t mode = 0;     // 0 a texture, 1 a CLUT load
};

/* swizzle.comp's push constants. */
struct SwizzleParams {
    std::uint32_t bp, bw, psm, cols, rows, stride, to_memory, depth, row0;
    std::uint32_t scale = 1;    // the image's pixels to a GS pixel: one taken to memory, a square filled from it
    std::uint32_t keep_colour = 0;  // from memory: the square's colour kept where memory differs in alpha alone
};

constexpr std::uint32_t kDepthRows = 512;   // a depth buffer's rows (DrawTriangles; the bridge's)
constexpr VkDeviceSize kLocalMemory = 4u << 20;
constexpr VkDeviceSize kLinear = VkDeviceSize(2048) * 1024 * 4;    // the largest image, a value a pixel (at scale 1)
constexpr std::uint32_t kMaxScale = 4;

/* FBMSK as the bits of the target's RGBA bytes it keeps. FBMSK is in RGBA32
   bit positions whatever the format (the GS masks the colour before it is
   made 16-bit, as the software GS and PCSX2 do): a 16-bit format stores the
   top five bits of each colour byte and alpha's top bit (0x80F8F8F8), the
   rest stay 0 either way, so count as kept; a 24-bit format keeps the alpha
   it does not store. (Read as a 16-bit pixel's bits, Burnout 3's FBMSK
   0x3FFF wrote alpha's top two bits alone, not its byte: its depth into the
   picture's alpha - the haze, the sun's halo - came out wrong.) */
std::uint32_t KeptBits(std::uint32_t fbmsk, std::uint32_t psm) {
    if (psm == 0x02 || psm == 0x0A || psm == 0x32 || psm == 0x3A) return fbmsk | ~0x80F8F8F8u;
    if (psm == 0x01 || psm == 0x31) return fbmsk | 0xFF000000u;
    return fbmsk;
}

/* The channels written whole or in part (Vulkan's write mask: R, G, B, A in
   bits 0-3). */
std::uint32_t WriteMask(std::uint32_t kept) {
    std::uint32_t mask = 0;
    for (int channel = 0; channel < 4; ++channel)
        if (((kept >> (8 * channel)) & 0xFF) != 0xFF) mask |= 1u << channel;
    return mask;
}

/* Whether some channel is written only in part. */
bool PartlyMasked(std::uint32_t kept) {
    for (int channel = 0; channel < 4; ++channel) {
        const std::uint32_t byte = (kept >> (8 * channel)) & 0xFF;
        if (byte != 0 && byte != 0xFF) return true;
    }
    return false;
}

/* A blend the blender cannot do: FIX over 0x80 is a factor over 1.0, which
   a fixed-point target clamps to 1 (Burnout 3's glow: (Cs - Cd) * 0xFF/128,
   then Cs * 0xFF/128 - at half its strength). The shader blends instead,
   from a copy of the target. */
bool Is16Bit(std::uint32_t psm) { return psm == 0x02 || psm == 0x0A || psm == 0x32 || psm == 0x3A; }

bool ShaderBlend(const GsDrawState& s) {
    if (!s.blend) return false;
    // Into a 16-bit target: the blend's result cut to five bits a channel,
    // as the GS stores it (the blender would keep eight: the next blend over
    // it would read more than the GS has - Burnout 3's picture copied into
    // its 16-bit display buffer, blended over the last, came out 9 brighter
    // in the dark).
    if (Is16Bit(s.psm)) return true;
    if (s.a == s.b) return false;
    // The destination's alpha as the factor (C = Ad): over 128 on the GS,
    // over 255 as the blender reads a byte (DST_ALPHA) - at half its strength.
    // In the shader for a small target (Burnout 3's sky: its 64x32 gradient
    // blended by the alpha laid down first) and for sprites (its full-screen
    // passes - Cs * Ad + Cd over the picture, after a crash: strips side by
    // side, one copy of the picture for all), not for the race's scene: it
    // blends by Ad too, thousands of overlapping triangles, each needing a
    // fresh copy of the picture - 60 frames a second at 3x became 47. There
    // the frame's alpha is small (under 16) while the scene is drawn: half
    // strength of almost nothing.
    if (s.c == BlendAlpha::Dest) return s.fbw <= 2 || s.sprite_prim;
    return s.c == BlendAlpha::Fix && s.fix > 0x80;
}
constexpr std::uint32_t kShaderBlend = 1u << 15;

/* The pipeline a state needs: blending and its operands, and the write mask
   (bits 9-12, inverted so that 0 is the plain pipeline); bit 15 the blend
   in the shader. */
std::uint32_t PipelineKey(const GsDrawState& s) {
    const std::uint32_t masked = (~WriteMask(KeptBits(s.fbmsk, s.psm)) & 0xF) << 9;
    if (!s.blend) return masked;
    return masked | 1u | std::uint32_t(s.a) << 1 | std::uint32_t(s.b) << 3 | std::uint32_t(s.c) << 5 |
           std::uint32_t(s.d) << 7 | (ShaderBlend(s) ? kShaderBlend : 0u);
}

/* Cv = (A - B) * C + D as the blender's src * sf (op) dst * df. Each of Cs and
   Cd ends up with a coefficient (A is it) - (B is it), times C, plus (D is it):
   0, 1, C, -C, 1 - C, or 1 + C (more than the blender's factors reach; taken
   as 1). */
VkPipelineColorBlendAttachmentState Blending(std::uint32_t key, bool dual_source) {
    VkPipelineColorBlendAttachmentState b{};
    b.colorWriteMask = ~(key >> 9) & 0xF;           // R, G, B, A are bits 0-3, as Vulkan has them
    if (!(key & 1) || (key >> 13) & 3) return b;    // (a logic op replaces blending)
    if (key & kShaderBlend) return b;               // (the shader blends)
    const auto a = BlendColour((key >> 1) & 3), bb = BlendColour((key >> 3) & 3), d = BlendColour((key >> 7) & 3);
    const auto c = BlendAlpha((key >> 5) & 3);
    const bool dest_alpha = c == BlendAlpha::Dest;
    const VkBlendFactor kC = dest_alpha ? VK_BLEND_FACTOR_DST_ALPHA
                             : dual_source ? VK_BLEND_FACTOR_SRC1_ALPHA : VK_BLEND_FACTOR_SRC_ALPHA;
    const VkBlendFactor kOneMinusC = dest_alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA
                                     : dual_source ? VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
                                                   : VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    auto coefficient = [&](BlendColour who, bool* negative) {
        const int m = int(a == who) - int(bb == who);
        const bool plus_one = d == who;
        *negative = false;
        if (m == 0) return plus_one ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
        if (m == 1) return plus_one ? VK_BLEND_FACTOR_ONE : kC;
        if (plus_one) return kOneMinusC;
        *negative = true;
        return kC;
    };
    bool src_negative, dst_negative;
    b.srcColorBlendFactor = coefficient(BlendColour::Source, &src_negative);
    b.dstColorBlendFactor = coefficient(BlendColour::Dest, &dst_negative);
    b.colorBlendOp = VK_BLEND_OP_ADD;
    if (!src_negative && dst_negative) b.colorBlendOp = VK_BLEND_OP_SUBTRACT;
    else if (src_negative && !dst_negative) b.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
    else if (src_negative && dst_negative) b.srcColorBlendFactor = b.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    // The GS writes the source alpha as it is; blending is for colour only.
    b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    b.alphaBlendOp = VK_BLEND_OP_ADD;
    b.blendEnable = VK_TRUE;
    return b;
}

bool SameBatch(const GsDrawState& x, const GsDrawState& y) {
    return x.fbp == y.fbp && x.fbw == y.fbw && x.psm == y.psm && x.fbmsk == y.fbmsk && x.scissor_x0 == y.scissor_x0 &&
           x.scissor_y0 == y.scissor_y0 && x.scissor_x1 == y.scissor_x1 && x.scissor_y1 == y.scissor_y1 &&
           x.textured == y.textured && (!x.textured || (x.texture == y.texture && x.tfx == y.tfx &&
           x.tcc == y.tcc && x.bilinear == y.bilinear && x.mip == y.mip && x.wms == y.wms && x.wmt == y.wmt && x.minu == y.minu &&
           x.maxu == y.maxu && x.minv == y.minv && x.maxv == y.maxv && x.tw == y.tw && x.th == y.th &&
           x.texture_target == y.texture_target && x.tex_fbp == y.tex_fbp && x.tex_fbw == y.tex_fbw &&
           x.tex_target_psm == y.tex_target_psm && x.tex_format == y.tex_format && x.ta0 == y.ta0 &&
           x.ta1 == y.ta1 && x.aem == y.aem)) &&
           x.alpha_test == y.alpha_test && (!x.alpha_test || (x.atst == y.atst && x.aref == y.aref && x.afail == y.afail)) &&
           PipelineKey(x) == PipelineKey(y) && (!x.blend || (x.fix == y.fix && x.pabe == y.pabe)) && x.depth == y.depth &&
           (!x.depth || (x.zbp == y.zbp && x.ztst == y.ztst && x.depth_write == y.depth_write)) &&
           x.fog == y.fog && (!x.fog || x.fogcol == y.fogcol) && x.date == y.date && (!x.date || x.datm == y.datm) &&
           x.sprite == y.sprite && x.sprite_prim == y.sprite_prim;
}

/* PS2HOST_GPU_COUNT: why batches end - the first state SameBatch finds
   changed, or a flush from elsewhere (a sync, a CLUT, a decode, a read) -
   for RendererCounters. */
const char* BreakReason(const GsDrawState& x, const GsDrawState& y) {
    if (x.fbp != y.fbp || x.fbw != y.fbw || x.psm != y.psm) return "target";
    if (x.fbmsk != y.fbmsk) return "fbmsk";
    if (x.scissor_x0 != y.scissor_x0 || x.scissor_y0 != y.scissor_y0 || x.scissor_x1 != y.scissor_x1 ||
        x.scissor_y1 != y.scissor_y1)
        return "scissor";
    if (x.textured != y.textured) return "textured on/off";
    if (x.textured) {
        if (x.texture_target != y.texture_target || x.tex_fbp != y.tex_fbp || x.tex_fbw != y.tex_fbw ||
            x.tex_target_psm != y.tex_target_psm)
            return "texture from a target";
        if (x.texture != y.texture) return "texture";
        if (x.tfx != y.tfx || x.tcc != y.tcc) return "tfx/tcc";
        if (x.bilinear != y.bilinear || x.mip != y.mip) return "filter";
        if (x.wms != y.wms || x.wmt != y.wmt || x.minu != y.minu || x.maxu != y.maxu || x.minv != y.minv ||
            x.maxv != y.maxv)
            return "clamp";
        if (x.tw != y.tw || x.th != y.th || x.tex_format != y.tex_format) return "texture size/format";
        if (x.ta0 != y.ta0 || x.ta1 != y.ta1 || x.aem != y.aem) return "texa";
    }
    if (x.alpha_test != y.alpha_test || (x.alpha_test && (x.atst != y.atst || x.aref != y.aref || x.afail != y.afail)))
        return "alpha test";
    if (PipelineKey(x) != PipelineKey(y)) return "pipeline (blend, depth test, ...)";
    if (x.blend && (x.fix != y.fix || x.pabe != y.pabe)) return "blend FIX/PABE";
    if (x.depth != y.depth || (x.depth && (x.zbp != y.zbp || x.ztst != y.ztst || x.depth_write != y.depth_write)))
        return "depth";
    if (x.fog != y.fog || (x.fog && x.fogcol != y.fogcol)) return "fog";
    if (x.date != y.date || (x.date && x.datm != y.datm)) return "date";
    if (x.sprite != y.sprite || x.sprite_prim != y.sprite_prim) return "sprite/triangle";
    return "?";
}
/* A batch's triangle, for the exact overlap test (DrawTriangles). */
struct BatchTriangle {
    double x[3], y[3];
    int x0, y0, x1, y1;                 // its bounding pixels, inclusive
    // Worked out once (Prepare): its edges' normals, its own extent along
    // each, and whether it has no area.
    double nx[3], ny[3], lo[3], hi[3];
    bool flat;
};

void Prepare(BatchTriangle& t) {
    t.flat = (t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (t.y[1] - t.y[0]) == 0.0;
    for (int e = 0; e < 3; ++e) {
        t.nx[e] = -(t.y[(e + 1) % 3] - t.y[e]);
        t.ny[e] = t.x[(e + 1) % 3] - t.x[e];
        double lo = 1e300, hi = -1e300;
        for (int i = 0; i < 3; ++i) {
            const double p = t.nx[e] * t.x[i] + t.ny[e] * t.y[i];
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        t.lo[e] = lo;
        t.hi[e] = hi;
    }
}

/* Whether two triangles' interiors meet: no axis among their six edges'
   normals separates them (a shared edge, or a corner, is not meeting - the
   rasterizer's top-left rule gives a pixel centre on it to one only). A
   triangle with no area draws nothing: it meets nothing. */
bool InteriorsMeet(const BatchTriangle& a, const BatchTriangle& b) {
    auto area = [](const BatchTriangle& t) {
        return (t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (t.y[1] - t.y[0]);
    };
    if (area(a) == 0.0 || area(b) == 0.0) return false;
    for (const BatchTriangle* t : {&a, &b})
        for (int e = 0; e < 3; ++e) {
            const double nx = -(t->y[(e + 1) % 3] - t->y[e]), ny = t->x[(e + 1) % 3] - t->x[e];
            double a0 = 1e300, a1 = -1e300, b0 = 1e300, b1 = -1e300;
            for (int i = 0; i < 3; ++i) {
                const double pa = nx * a.x[i] + ny * a.y[i], pb = nx * b.x[i] + ny * b.y[i];
                a0 = std::min(a0, pa);
                a1 = std::max(a1, pa);
                b0 = std::min(b0, pb);
                b1 = std::max(b1, pb);
            }
            if (a1 <= b0 || b1 <= a0) return false;
        }
    return true;
}

/* The same, by the triangles' prepared axes and extents (Prepare): the same
   products, so the same answer - each extent once, not once a pair. */
bool InteriorsMeetPrepared(const BatchTriangle& a, const BatchTriangle& b) {
    if (a.flat || b.flat) return false;
    const auto separated = [](const BatchTriangle& t, const BatchTriangle& other) {
        for (int e = 0; e < 3; ++e) {
            double lo = 1e300, hi = -1e300;
            for (int i = 0; i < 3; ++i) {
                const double p = t.nx[e] * other.x[i] + t.ny[e] * other.y[i];
                lo = std::min(lo, p);
                hi = std::max(hi, p);
            }
            if (t.hi[e] <= lo || hi <= t.lo[e]) return true;
        }
        return false;
    };
    return !separated(a, b) && !separated(b, a);
}

std::map<std::string, std::uint64_t> g_breaks;
const char* g_flush_why = nullptr;              // set by the batching's own flushes
bool CountingBreaks() {
    static const bool on = std::getenv("PS2HOST_GPU_COUNT") != nullptr;
    return on;
}

class VulkanGsRenderer final : public GsRenderer {
public:
    explicit VulkanGsRenderer(VulkanGpu& gpu) : g_(gpu) {}
    ~VulkanGsRenderer() override;
    bool Init(std::string* error);
    void UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                       std::uint32_t height) override;
    bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                           std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) override;
    bool HasReplacement(std::uint64_t id) const override { return replacements_.count(id) != 0; }
    void UseReplacement(std::uint64_t key, std::uint64_t id) override { textures_[key].replacement = id; }
    bool BlockCompression() const override { return g_.bc; }
    void UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, const std::uint32_t* z) override;
    void UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, const std::uint32_t* pixels,
                      std::uint32_t width, std::uint32_t height) override;
    void ReleaseTexture(std::uint64_t key) override;
    bool HasTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) const override {
        return targets_.count(Key{fbp, fbw, psm}) != 0;
    }
    void DrawTriangles(const GsDrawState& state, const GsVertex* vertices, std::size_t count) override;
    void Flush() override;
    void Kick() override {
        Flush();
        g_.Flush();
        ++frame_;
        if (date_in_pass_ == 2) ReportDateCheck(frame_ % 600 == 0);
        EvictReplacements();
    }
    void TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                              std::uint32_t height, std::uint32_t first_row, std::uint32_t rows) override;
    void ReloadKeepsColour(bool keep) override { keep_colour_ = keep; }
    void DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                             std::uint32_t zpsm, std::uint32_t first_row, std::uint32_t rows) override;
    bool DisplayTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint64_t* image, int* layout,
                       std::uint32_t* width, std::uint32_t* height) override {
        auto found = targets_.find(Key{fbp, fbw, psm});
        if (found == targets_.end() || !found->second.image || found->second.layout == VK_IMAGE_LAYOUT_UNDEFINED)
            return false;
        Kick();                         // the drawing, on the queue before the picture's copy
        const Image& t = found->second;
        *image = reinterpret_cast<std::uint64_t>(t.image);
        *layout = int(t.layout);
        *width = t.pw();
        *height = t.ph();
        return true;
    }
    bool WeaveField(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t first_line,
                    std::uint32_t lines, std::uint32_t parity, std::uint64_t* image, int* layout,
                    std::uint32_t* width, std::uint32_t* height) override;
    bool ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t>* z) override;
    std::uint32_t Scale() const override { return scale_; }
    void SetScale(std::uint32_t scale) override;
    void MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) override;
    void MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) override;
    void MemoryPatch(const std::uint32_t* words, std::size_t count) override;
    bool TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                        std::uint32_t rows, std::uint32_t first_row) override;
    void TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                          std::uint32_t height) override;
    bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                       std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows, std::uint32_t first_row) override;
    void DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                         std::uint32_t zpsm) override;
    void TextureFromMemory(std::uint64_t key, const GsTextureDecode& decode) override;
    void ClutFromMemory(const GsClutLoad& load) override;
    void ClutRows(const std::uint32_t* clut, std::uint32_t rows) override;
    bool ReadClut(std::uint32_t* clut) override;
    void SetExactOverlap(bool exact) override { exact_overlap_ = exact; }
    void SetPreparedAxes(bool on) override { prepared_axes_ = on; }
    void SetDateArea(bool batch) override { date_batch_area_ = batch; }
    bool SetRecordThread(bool on) override {
        Flush();
        return g_.SetRecordThread(on);
    }
    int SetDateInPass(int mode) override;
    bool DateInPassCheck(std::uint64_t* batches, std::uint64_t* differ) override;
    bool ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                    std::uint32_t* width, std::uint32_t* height) override;
    bool ReadWoven(std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height) override {
        Flush();
        return woven_.image && ReadImage(woven_, pixels, width, height);
    }
    /* An image's every pixel, back to the CPU. */
    bool ReadImage(Image& t, std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height);
    bool ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                        std::uint32_t* width, std::uint32_t* height) override;
    bool ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                     std::uint32_t* height) override;
    bool ReadImage(Image& image, std::vector<std::uint32_t>* pixels);

private:
    using Key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    Image* TargetFor(const GsDrawState& state);
    /* The depth buffer a draw uses: ZBP's, or the target's own when it tests
       nothing (dynamic rendering wants one attachment layout throughout). */
    Image* DepthFor(std::uint32_t key, std::uint32_t width, std::uint32_t height);
    bool MakeImage(Image& image, std::uint32_t width, std::uint32_t height, VkImageUsageFlags usage,
                   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, std::uint32_t scale = 1, std::uint32_t levels = 1);
    void DestroyImage(Image& image);
    /* Device-local memory for images: pieces of 64 MB blocks, a power of two
       (64 KB to 16 MB) each, kept for reuse once freed - the textures that
       come and go a frame (a full cache, a new area of a track) made no
       allocation of the driver's each (vkAllocateMemory goes to the system's
       kernel). Larger, an allocation of its own. */
    bool ImageMemory(const VkMemoryRequirements& req, std::uint32_t type, Image& image);
    void FreeImageMemory(VkDeviceMemory memory, VkDeviceSize offset, std::uint32_t piece_class, std::uint32_t type);
    std::map<std::uint64_t, std::vector<std::pair<VkDeviceMemory, VkDeviceSize>>> image_pieces_;   // (type, class) -> free
    std::vector<VkDeviceMemory> image_blocks_;
    void Fill(Texture& texture, const std::uint32_t* pixels);
    VkPipeline PipelineFor(std::uint32_t key);
    /* The destination alpha test's setup pass: stencil 1 where the target's
       alpha bit equals DATM. */
    bool MakeDatePipeline();
    /* Over `area` (the image's pixels): the stencil is read nowhere else. */
    void DateSetup(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm, const VkRect2D& area);
    bool date_batch_area_ = true;   /* SetDateArea */
    /* SetDateInPass: 0 the setup a pass of its own, as above; 1 within the
       draws' pass (the target read where it is drawn, after a barrier: no
       pass ended, no image changing layout), the stencil the same; 2 the
       same, then checked against the target before the batch draws. */
    int date_in_pass_ = 0;
    VkDescriptorSetLayout input_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool input_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout date_local_layout_ = VK_NULL_HANDLE;
    VkPipeline date_local_pipeline_ = VK_NULL_HANDLE;
    VkPipeline check_pipeline_ = VK_NULL_HANDLE;
    bool MakeDateInPass();
    VkDescriptorSet InputSet(Image& target);
    /* Within the open pass, over `area`: the stencil 0, then 1 where the alpha bit is DATM. */
    void DateSetupInPass(VkCommandBuffer cmd, Image& target, std::uint32_t datm, const VkRect2D& area);
    /* Mode 2, outside a pass: the pixels of `area` whose stencil is not what their alpha bit says. */
    void DateCheck(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm, const VkRect2D& area);
    VkBuffer check_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory check_memory_ = VK_NULL_HANDLE;
    void* check_mapped_ = nullptr;
    VkDescriptorSet check_set_ = VK_NULL_HANDLE;
    std::uint64_t checked_batches_ = 0;
    std::uint32_t reported_differ_ = 0;
    void ReportDateCheck(bool always);
    /* The colour target's layout in the draws' pass: one the pass can read it in. */
    VkImageLayout TargetLayout() const {
        return date_in_pass_ ? VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    /* A target's descriptor for sampling it. */
    VkDescriptorSet SampledSet(Image& image);
    /* Copies of the target being drawn, when it is also the texture: one a
       size, kept (one image resized in turn - the race alternates targets of
       several sizes - waited for the GPU each time: 32 s of a race script). */
    std::map<std::pair<std::uint32_t, std::uint32_t>, Image> feedbacks_;
    Image* feedback_ = nullptr;     /* the one the batch being recorded copies into */
    /* A batch that samples its own target reads one copy, made as it begins;
       the GS reads each primitive's texels as the ones before it left them.
       What the batch has drawn so far (GS pixels), so a primitive that reads
       there starts a batch of its own, with a new copy (Burnout 3's speed
       blur: layers each drawn over the last, as one batch read all from the
       picture before them - a fainter blur). */
    bool drawn_any_ = false;
    int drawn_x0_ = 0, drawn_y0_ = 0, drawn_x1_ = 0, drawn_y1_ = 0;     /* pixels, inclusive */
    /* The exact overlap test (SetExactOverlap): the batch's triangles, and a
       grid of 32-pixel cells listing those over each. */
    static constexpr int kCellShift = 5, kCells = 64;
    bool exact_overlap_ = true;
    bool prepared_axes_ = true;         // SetExactOverlap: "exact" (prepared) or "exact-direct"
    std::vector<BatchTriangle> batch_tris_;
    std::vector<std::vector<std::uint32_t>> batch_cells_ = std::vector<std::vector<std::uint32_t>>(kCells * kCells);
    std::vector<std::uint32_t> used_cells_;
    template <typename F>
    void ForCells(const BatchTriangle& t, F&& f) {
        const int cx0 = std::clamp(t.x0 >> kCellShift, 0, kCells - 1), cx1 = std::clamp(t.x1 >> kCellShift, 0, kCells - 1);
        const int cy0 = std::clamp(t.y0 >> kCellShift, 0, kCells - 1), cy1 = std::clamp(t.y1 >> kCellShift, 0, kCells - 1);
        for (int cy = cy0; cy <= cy1; ++cy)
            for (int cx = cx0; cx <= cx1; ++cx) f(std::uint32_t(cy * kCells + cx));
    }
    static BatchTriangle TriangleOf(const GsVertex* v) {
        BatchTriangle t{};
        double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
        for (int i = 0; i < 3; ++i) {
            t.x[i] = v[i].x;
            t.y[i] = v[i].y;
            x0 = std::min(x0, t.x[i]);
            y0 = std::min(y0, t.y[i]);
            x1 = std::max(x1, t.x[i]);
            y1 = std::max(y1, t.y[i]);
        }
        t.x0 = int(std::floor(x0));
        t.y0 = int(std::floor(y0));
        t.x1 = int(std::ceil(x1));
        t.y1 = int(std::ceil(y1));
        Prepare(t);
        return t;
    }
    /* Whether a triangle of these meets one the batch drew. */
    bool MeetsBatch(const GsVertex* v, std::size_t count) {
        for (std::size_t i = 0; i + 3 <= count; i += 3) {
            const BatchTriangle t = TriangleOf(v + i);
            bool meets = false;
            ForCells(t, [&](std::uint32_t cell) {
                if (meets) return;
                for (std::uint32_t other : batch_cells_[cell]) {
                    const BatchTriangle& o = batch_tris_[other];
                    if (o.x1 < t.x0 || t.x1 < o.x0 || o.y1 < t.y0 || t.y1 < o.y0) continue;
                    if (prepared_axes_ ? InteriorsMeetPrepared(t, o) : InteriorsMeet(t, o)) {
                        meets = true;
                        return;
                    }
                }
            });
            if (meets) return true;
        }
        return false;
    }
    void AddToBatch(const GsVertex* v, std::size_t count) {
        for (std::size_t i = 0; i + 3 <= count; i += 3) {
            const BatchTriangle t = TriangleOf(v + i);
            const std::uint32_t index = std::uint32_t(batch_tris_.size());
            batch_tris_.push_back(t);
            ForCells(t, [&](std::uint32_t cell) {
                if (batch_cells_[cell].empty()) used_cells_.push_back(cell);
                batch_cells_[cell].push_back(index);
            });
        }
    }
    void ClearBatchTriangles() {
        for (std::uint32_t cell : used_cells_) batch_cells_[cell].clear();
        used_cells_.clear();
        batch_tris_.clear();
    }
    /* The batch's sprites (DrawTriangles: those meeting another edge to edge
       reach that edge): where on the screen and in the texture, and their
       first vertex in pending_. */
    struct SpriteSpan {
        float x0, x1, y0, y1, s0, s1, t0, t1;
        std::size_t vertex;
    };
    std::vector<SpriteSpan> sprites_;
    /* A flat triangle (one Z, one Q: text, a HUD) on a target drawn larger,
       the last one (a quad is two, sharing a diagonal), and the batch's
       quads: each kept half a texel inside its texels (a letter of a font
       page off its neighbour's: dark lines along italic letters' edges),
       but for an edge it shares with another (a mesh of one picture). */
    struct FlatTriangle {
        GsVertex v[3];
        std::size_t vertex;
    };
    FlatTriangle last_flat_{};
    bool last_flat_valid_ = false;
    struct FlatQuad {
        GsVertex v[4];
        std::size_t vertex[2];          // its triangles' first vertices in pending_
        float lo[2], hi[2];             // S, T: its texels' extent
        float range[4];                 // ...and what it keeps to
    };
    std::vector<FlatQuad> quads_;
    void FlatQuadRange(FlatQuad& q, const GsDrawState& state);
    /* ...and what it reads of its target (texels; for blending in the shader
       the pixels it draws), so the copy is of that alone - a whole target a
       copy was 60 a race frame, ten megabytes each at 3x. read_all_: what it
       reads is not a rectangle (REPEAT past the texture, REGION_REPEAT). */
    bool read_any_ = false, read_all_ = false;
    int read_x0_ = 0, read_y0_ = 0, read_x1_ = 0, read_y1_ = 0;
    void Reads(int x0, int y0, int x1, int y1) {
        if (!read_any_) {
            read_x0_ = x0;
            read_y0_ = y0;
            read_x1_ = x1;
            read_y1_ = y1;
            read_any_ = true;
            return;
        }
        read_x0_ = std::min(read_x0_, x0);
        read_y0_ = std::min(read_y0_, y0);
        read_x1_ = std::max(read_x1_, x1);
        read_y1_ = std::max(read_y1_, y1);
    }
    /* The rendering pass the last batch left open, so the next one into the
       same target and depth buffer goes on in it (thousands of batches a
       frame, each beginning and ending its own, cost the GPU most of its
       time); anything else recorded ends it (VulkanGpu::end_pass). */
    std::uint32_t scale_ = 1;        // the internal resolution: targets' pixels to a GS pixel
    /* The scale an image of this width is drawn at (targets and depth
       buffers alike: one width, one scale, so the images a key names agree):
       the internal resolution, but a small one's - 320 wide or less: a
       game's scratch targets for its glow, blur and reflections, which it
       only reads blurred - at the PS2's own. Burnout 3's glow bounces
       through 64-320-wide views of one memory some fifty times a frame,
       each brought back from memory: at 4x those were a third of the GPU's
       time, and the upscaled glow ghosted (a second, offset outline around
       its words). */
    std::uint32_t ScaleOf(std::uint32_t width) const { return width <= 320 ? 1u : scale_; }
    Image* open_target_ = nullptr;
    Image* open_depth_ = nullptr;
    void EndPass(VkCommandBuffer cmd) {
        if (!open_target_) return;
        vkCmdEndRendering(cmd);
        open_target_ = open_depth_ = nullptr;
    }
    VkPipeline date_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet SetFor(Texture& texture, int sampler);
    bool Buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer, VkDeviceMemory* memory, void** mapped);

    VulkanGpu& g_;
    VkShaderModule vert_ = VK_NULL_HANDLE, frag_ = VK_NULL_HANDLE, frag_blend_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    bool keep_colour_ = false;      /* ReloadKeepsColour */
    VkSampler samplers_[8] = {};    /* bilinear << 2 | clamp_s << 1 | clamp_t */
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::map<std::uint32_t, VkPipeline> pipelines_;
    std::uint32_t last_pipeline_key_ = ~0u;
    VkPipeline last_pipeline_ = VK_NULL_HANDLE;
    std::map<Key, Image> targets_;
    Image woven_;                       /* WeaveField's picture: both fields */
    std::map<Key, Image> depths_;   /* by (ZBP or a target's own, width, height) */
    std::map<std::uint64_t, Texture> textures_;
    /* A texture pack's replacements, by id (UploadReplacement); those not
       drawn for the longest go when they come to over kReplacementBudget. */
    std::unordered_map<std::uint64_t, Texture> replacements_;
    VkDeviceSize replacement_bytes_ = 0;
    std::uint64_t frame_ = 0;
    static constexpr VkDeviceSize kReplacementBudget = VkDeviceSize(1536) << 20;
    void EvictReplacements();
    void DestroyTexture(Texture texture);
    Texture blank_;                 /* bound when nothing is textured */
    GsDrawState state_;
    std::vector<PackedVertex> pending_;
    /* Vertex data for what is recorded: appended to chunks of host-visible
       memory, all reusable once the submission that reads them has run. */
    struct VertexChunk {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped;
        VkDeviceSize size;
    };
    // One ring a command buffer slot: a slot's vertices are free again once
    // its submission has run, whatever the others are doing.
    struct VertexRing {
        std::vector<VertexChunk> chunks;
        std::size_t chunk = 0;
        VkDeviceSize used = 0;
        bool reset_pending = false;
    };
    VertexRing vertex_rings_[VulkanGpu::kSlots];
    void* Vertices(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset);
    /* Upload data, the same way: a slot's chunks, free again once its
       submission has run. A staging buffer an upload, freed (or kept, 64 at
       most, of one size) after the submission, was the driver allocating and
       freeing memory hundreds of times a frame in a race. */
    VertexRing staging_rings_[VulkanGpu::kSlots];
    /* Released textures by size (width << 32 | height), image, memory, view and
       descriptor sets whole, for the next texture of that size: a race releases
       and makes textures by the hundred a frame, and each made one was a driver
       allocation and each released one a free. At most kPooledTextures kept. */
    std::unordered_map<std::uint64_t, std::vector<Texture>> texture_pool_;
    std::size_t pooled_textures_ = 0;
    static constexpr std::size_t kPooledTextures = 1024;
    void* Staging(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset);
    void* RingAlloc(VertexRing* rings, VkBufferUsageFlags usage, VkDeviceSize bytes, VkBuffer* buffer,
                    VkDeviceSize* offset);
    /* A staging buffer, freed once the submission that reads it has run. */
    void FreeAfterSubmit(VkBuffer buffer, VkDeviceMemory memory);
    /* Staging buffers for uploads, kept mapped and used again once the GPU
       is done with them: allocating and freeing one an upload (the race
       uploads to GPU memory tens of thousands of times) was 3% of the run
       in the driver. By size class (a power of two, 64 KB at least). */
    struct Staged {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };
    std::vector<Staged> staging_free_;
    std::unordered_map<VkBuffer, Staged> staging_live_;

    /* GS local memory on the GPU (swizzle.comp): the memory itself, and the
       linear buffer texture decodes go through (images go to and from
       memory straight: SwizzleImage, DepthOut, DepthIn). */
    bool InitMemory(std::string* error);
    bool DeviceBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer, VkDeviceMemory* memory);
    /* A texture image of this size under `key`, made (or made again) if need be. */
    Texture* TextureImage(std::uint64_t key, std::uint32_t width, std::uint32_t height);
    /* The state a command buffer's draws last set: set again only when it
       changes (a race records ~2,000 batches a picture, and most set what
       the one before did - a dozen calls into the driver each). */
    struct DrawState {
        std::uint64_t recording = ~0ull;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet sets[2] = {};
        int depth_test = -1, stencil_test = -1, compare = -1, depth_write = -1;
        VkViewport viewport{};
        VkRect2D scissor{};
        float size[2] = {-1, -1};
        VkBuffer vertices = VK_NULL_HANDLE;
    } drawn_;
    // The CLUT for the draws (gs.frag's set 1): a target read as PSMT8H.
    VkDescriptorSetLayout clut_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet clut_set_ = VK_NULL_HANDLE;
    VkBuffer gsmem_ = VK_NULL_HANDLE, linear_ = VK_NULL_HANDLE, clut_ = VK_NULL_HANDLE;
    VkDeviceMemory gsmem_memory_ = VK_NULL_HANDLE, linear_memory_ = VK_NULL_HANDLE, clut_memory_ = VK_NULL_HANDLE;
    VkShaderModule decode_ = VK_NULL_HANDLE;
    VkPipeline decode_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule swizzle_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout swizzle_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool swizzle_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet swizzle_set_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_layout_ = VK_NULL_HANDLE;
    VkPipeline swizzle_pipeline_ = VK_NULL_HANDLE;
    /* The swizzle straight to and from a colour target (StorageSet). */
    VkShaderModule swizzle_image_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout image_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool storage_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_image_layout_ = VK_NULL_HANDLE;
    // MemoryPatch's: memory's set (the swizzle's), eight words in the push constants.
    VkShaderModule patch_ = VK_NULL_HANDLE;
    VkPipelineLayout patch_layout_ = VK_NULL_HANDLE;
    VkPipeline patch_pipeline_ = VK_NULL_HANDLE;
    VkPipeline swizzle_image_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet StorageSet(Image& image);
    void SwizzleImage(VkCommandBuffer cmd, const SwizzleParams& params, Image& image);
    /* Depth buffers: to memory sampled (the swizzle's DEPTH_IMAGE), back drawn
       (depth_load.frag) - no copy through a linear buffer either way. */
    VkShaderModule swizzle_depth_ = VK_NULL_HANDLE, depth_load_vert_ = VK_NULL_HANDLE,
                   depth_load_frag_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout depth_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_depth_layout_ = VK_NULL_HANDLE, depth_load_layout_ = VK_NULL_HANDLE;
    VkPipeline swizzle_depth_pipeline_ = VK_NULL_HANDLE, depth_load_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet DepthSampledSet(Image& image);
    void DepthOut(VkCommandBuffer cmd, const SwizzleParams& params, Image& depth);
    void DepthIn(VkCommandBuffer cmd, const SwizzleParams& params, Image& depth);
    bool MakeDepthPipelines();
};

void* VulkanGsRenderer::Vertices(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset) {
    return RingAlloc(vertex_rings_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bytes, buffer, offset);
}

void* VulkanGsRenderer::Staging(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset) {
    // (Not Buffer's pooled staging path: that is for TRANSFER_SRC alone.)
    return RingAlloc(staging_rings_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, bytes,
                     buffer, offset);
}

void* VulkanGsRenderer::RingAlloc(VertexRing* rings, VkBufferUsageFlags usage, VkDeviceSize bytes,
                                  VkBuffer* buffer, VkDeviceSize* offset) {
    // The slot the commands using these are recorded into.
    VertexRing& ring = rings[g_.slot()];
    if (!ring.reset_pending) {
        ring.reset_pending = true;
        g_.AfterSubmit([&ring] {
            ring.chunk = 0;
            ring.used = 0;
            ring.reset_pending = false;
        });
    }
    const VkDeviceSize aligned = (bytes + 255) & ~VkDeviceSize(255);
    while (ring.chunk < ring.chunks.size() && ring.used + aligned > ring.chunks[ring.chunk].size) {
        ++ring.chunk;
        ring.used = 0;
    }
    if (ring.chunk == ring.chunks.size()) {
        VertexChunk chunk{};
        chunk.size = std::max<VkDeviceSize>(VkDeviceSize(8) << 20, aligned);
        if (!Buffer(chunk.size, usage, &chunk.buffer, &chunk.memory, &chunk.mapped))
            return nullptr;
        ring.chunks.push_back(chunk);
        ring.used = 0;
    }
    VertexChunk& chunk = ring.chunks[ring.chunk];
    *buffer = chunk.buffer;
    *offset = ring.used;
    void* at = static_cast<std::uint8_t*>(chunk.mapped) + ring.used;
    ring.used += aligned;
    return at;
}

void VulkanGsRenderer::FreeAfterSubmit(VkBuffer buffer, VkDeviceMemory memory) {
    auto live = staging_live_.find(buffer);
    if (live != staging_live_.end()) {
        const Staged staged = live->second;
        staging_live_.erase(live);
        g_.AfterSubmit([this, staged] {
            if (staging_free_.size() < 64) {
                staging_free_.push_back(staged);
                return;
            }
            vkUnmapMemory(g_.device, staged.memory);
            vkDestroyBuffer(g_.device, staged.buffer, nullptr);
            vkFreeMemory(g_.device, staged.memory, nullptr);
        });
        return;
    }
    VkDevice device = g_.device;
    g_.AfterSubmit([device, buffer, memory] {
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
    });
}

bool VulkanGsRenderer::Buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer,
                              VkDeviceMemory* memory, void** mapped) {
    if (usage == VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
        VkDeviceSize size = 64 * 1024;
        while (size < bytes) size *= 2;
        for (std::size_t i = 0; i < staging_free_.size(); ++i)
            if (staging_free_[i].size == size) {
                const Staged staged = staging_free_[i];
                staging_free_[i] = staging_free_.back();
                staging_free_.pop_back();
                staging_live_[staged.buffer] = staged;
                *buffer = staged.buffer;
                *memory = staged.memory;
                *mapped = staged.mapped;
                return true;
            }
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
        vkBindBufferMemory(g_.device, *buffer, *memory, 0);
        if (vkMapMemory(g_.device, *memory, 0, size, 0, mapped) != VK_SUCCESS) return false;
        staging_live_[*buffer] = Staged{*buffer, *memory, *mapped, size};
        return true;
    }
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(g_.device, *buffer, *memory, 0);
    return vkMapMemory(g_.device, *memory, 0, bytes, 0, mapped) == VK_SUCCESS;
}

bool VulkanGsRenderer::MakeImage(Image& t, std::uint32_t width, std::uint32_t height, VkImageUsageFlags usage,
                                 VkFormat format, std::uint32_t scale, std::uint32_t levels) {
    t.width = width;
    t.height = height;
    t.scale = std::max(1u, scale);
    t.levels = std::max(1u, levels);
    t.aspect = format == kDepthFormat ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {t.pw(), t.ph(), 1};
    ici.mipLevels = t.levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    if (vkCreateImage(g_.device, &ici, nullptr, &t.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_.device, t.image, &req);
    if (!ImageMemory(req, g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), t)) return false;
    vkBindImageMemory(g_.device, t.image, t.memory, t.offset);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {t.aspect, 0, t.levels, 0, 1};
    if (vkCreateImageView(g_.device, &vci, nullptr, &t.view) != VK_SUCCESS) return false;
    if (format == kDepthFormat && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(g_.device, &vci, nullptr, &t.depth_view) != VK_SUCCESS) return false;
    }
    return true;
}

void VulkanGsRenderer::DestroyImage(Image& t) {
    if (t.view) vkDestroyImageView(g_.device, t.view, nullptr);
    if (t.depth_view) vkDestroyImageView(g_.device, t.depth_view, nullptr);
    if (t.image) vkDestroyImage(g_.device, t.image, nullptr);
    if (t.memory) FreeImageMemory(t.memory, t.offset, t.piece_class, t.memory_type);
    t = Image{};
}

bool VulkanGsRenderer::ImageMemory(const VkMemoryRequirements& req, std::uint32_t type, Image& t) {
    constexpr VkDeviceSize kBlock = VkDeviceSize(64) << 20, kLargest = VkDeviceSize(16) << 20;
    std::uint32_t piece_class = 16;                 // 64 KB
    while ((VkDeviceSize(1) << piece_class) < std::max(req.size, req.alignment)) ++piece_class;
    t.memory_type = type;
    if ((VkDeviceSize(1) << piece_class) > kLargest) {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        t.piece_class = 0;
        t.offset = 0;
        return vkAllocateMemory(g_.device, &mai, nullptr, &t.memory) == VK_SUCCESS;
    }
    auto& free = image_pieces_[std::uint64_t(type) << 32 | piece_class];
    if (free.empty()) {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = kBlock;
        mai.memoryTypeIndex = type;
        VkDeviceMemory block = VK_NULL_HANDLE;
        if (vkAllocateMemory(g_.device, &mai, nullptr, &block) != VK_SUCCESS) return false;
        image_blocks_.push_back(block);
        const VkDeviceSize piece = VkDeviceSize(1) << piece_class;
        for (VkDeviceSize at = kBlock; at >= piece; at -= piece) free.emplace_back(block, at - piece);
    }
    t.memory = free.back().first;
    t.offset = free.back().second;
    t.piece_class = piece_class;
    free.pop_back();
    return true;
}

void VulkanGsRenderer::FreeImageMemory(VkDeviceMemory memory, VkDeviceSize offset, std::uint32_t piece_class,
                                       std::uint32_t type) {
    if (!piece_class) {
        vkFreeMemory(g_.device, memory, nullptr);
        return;
    }
    image_pieces_[std::uint64_t(type) << 32 | piece_class].emplace_back(memory, offset);
}

bool VulkanGsRenderer::Init(std::string* error) {
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    vert_ = module(spirv_gs_vert, sizeof spirv_gs_vert);
    frag_ = g_.dual_source ? module(spirv_gs_frag, sizeof spirv_gs_frag)
                           : module(spirv_gs_frag_single, sizeof spirv_gs_frag_single);
    frag_blend_ = module(spirv_gs_frag_blend, sizeof spirv_gs_frag_blend);
    if (!vert_ || !frag_ || !frag_blend_) {
        if (error) *error = "the GS shaders did not load";
        return false;
    }
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &binding;
    vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &set_layout_);
    VkDescriptorSetLayoutBinding clut_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                              nullptr};
    VkDescriptorSetLayoutCreateInfo cli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    cli.bindingCount = 1;
    cli.pBindings = &clut_binding;
    vkCreateDescriptorSetLayout(g_.device, &cli, nullptr, &clut_set_layout_);
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 8192 + 4;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    vkCreateDescriptorPool(g_.device, &dpi, nullptr, &pool_);
    for (int i = 0; i < 8; ++i) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = (i & 4) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = (i & 2) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = (i & 1) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vkCreateSampler(g_.device, &sci, nullptr, &samplers_[i]);
    }
    VkPushConstantRange push[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, 8},
                                   {VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, sizeof(FragmentParams)}};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    // Set 2: the target's copy that blending in the shader reads (a texture's
    // layout, so a target's sampled set binds there too).
    const VkDescriptorSetLayout set_layouts[3] = {set_layout_, clut_set_layout_, set_layout_};
    lci.setLayoutCount = 3;
    lci.pSetLayouts = set_layouts;
    lci.pushConstantRangeCount = 2;
    lci.pPushConstantRanges = push;
    vkCreatePipelineLayout(g_.device, &lci, nullptr, &layout_);
    if (!PipelineFor(0) || !MakeDatePipeline()) {
        if (error) *error = "vkCreateGraphicsPipelines failed";
        return false;
    }
    const std::uint32_t white = 0xFFFFFFFFu;
    if (!MakeImage(blank_.image, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        if (error) *error = "no image for the blank texture";
        return false;
    }
    Fill(blank_, &white);
    g_.end_pass = [this](VkCommandBuffer cmd) { EndPass(cmd); };
    return InitMemory(error);
}

bool VulkanGsRenderer::DeviceBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer,
                                    VkDeviceMemory* memory) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
    return vkBindBufferMemory(g_.device, *buffer, *memory, 0) == VK_SUCCESS;
}

bool VulkanGsRenderer::InitMemory(std::string* error) {
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!DeviceBuffer(kLocalMemory, usage, &gsmem_, &gsmem_memory_) ||
        !DeviceBuffer(kLinear, usage, &linear_, &linear_memory_) ||
        !DeviceBuffer(1024, usage, &clut_, &clut_memory_)) {
        if (error) *error = "no GPU memory for GS local memory";
        return false;
    }
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &clut_set_layout_;
        if (vkAllocateDescriptorSets(g_.device, &ai, &clut_set_) != VK_SUCCESS) {
            if (error) *error = "no descriptor set for the CLUT";
            return false;
        }
        const VkDescriptorBufferInfo info{clut_, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = clut_set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sizeof spirv_swizzle_comp;
    ci.pCode = spirv_swizzle_comp;
    vkCreateShaderModule(g_.device, &ci, nullptr, &swizzle_);
    const VkShaderStageFlags memory_stages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3;
    dli.pBindings = bindings;
    vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &swizzle_set_layout_);
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    vkCreateDescriptorPool(g_.device, &dpi, nullptr, &swizzle_pool_);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = swizzle_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &swizzle_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &dai, &swizzle_set_) != VK_SUCCESS) {
        if (error) *error = "no descriptor set for GS local memory";
        return false;
    }
    VkDescriptorBufferInfo infos[3] = {{gsmem_, 0, VK_WHOLE_SIZE}, {linear_, 0, VK_WHOLE_SIZE},
                                       {clut_, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = swizzle_set_;
        writes[i].dstBinding = std::uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_.device, 3, writes, 0, nullptr);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             std::uint32_t(std::max(sizeof(SwizzleParams), sizeof(DecodeParams)))};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &swizzle_set_layout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    vkCreatePipelineLayout(g_.device, &lci, nullptr, &swizzle_layout_);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = swizzle_;
    cpi.stage.pName = "main";
    cpi.layout = swizzle_layout_;
    if (!swizzle_ ||
        vkCreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &swizzle_pipeline_) != VK_SUCCESS) {
        if (error) *error = "the swizzle shader did not load";
        return false;
    }
    {
        // The swizzle on a target's image: set 1 the image.
        VkShaderModuleCreateInfo ii{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ii.codeSize = sizeof spirv_swizzle_image_comp;
        ii.pCode = spirv_swizzle_image_comp;
        vkCreateShaderModule(g_.device, &ii, nullptr, &swizzle_image_);
        VkDescriptorSetLayoutBinding image_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo il{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        il.bindingCount = 1;
        il.pBindings = &image_binding;
        vkCreateDescriptorSetLayout(g_.device, &il, nullptr, &image_set_layout_);
        VkDescriptorPoolSize image_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024};
        VkDescriptorPoolCreateInfo ip{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ip.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ip.maxSets = 1024;
        ip.poolSizeCount = 1;
        ip.pPoolSizes = &image_size;
        vkCreateDescriptorPool(g_.device, &ip, nullptr, &storage_pool_);
        const VkDescriptorSetLayout layouts[2] = {swizzle_set_layout_, image_set_layout_};
        VkPipelineLayoutCreateInfo pl = lci;
        pl.setLayoutCount = 2;
        pl.pSetLayouts = layouts;
        vkCreatePipelineLayout(g_.device, &pl, nullptr, &swizzle_image_layout_);
        VkComputePipelineCreateInfo pi = cpi;
        pi.stage.module = swizzle_image_;
        pi.layout = swizzle_image_layout_;
        if (!swizzle_image_ ||
            vkCreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &pi, nullptr, &swizzle_image_pipeline_) !=
                VK_SUCCESS) {
            if (error) *error = "the swizzle shader (images) did not load";
            return false;
        }
    }
    if (!MakeDepthPipelines()) {
        if (error) *error = "the depth buffer shaders did not load";
        return false;
    }
    {
        VkShaderModuleCreateInfo pi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        pi.codeSize = sizeof spirv_patch_comp;
        pi.pCode = spirv_patch_comp;
        vkCreateShaderModule(g_.device, &pi, nullptr, &patch_);
        VkPushConstantRange words{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &swizzle_set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &words;
        vkCreatePipelineLayout(g_.device, &pl, nullptr, &patch_layout_);
        VkComputePipelineCreateInfo pp = cpi;
        pp.stage.module = patch_;
        pp.layout = patch_layout_;
        if (!patch_ || vkCreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &pp, nullptr, &patch_pipeline_) != VK_SUCCESS) {
            if (error) *error = "the memory patch shader did not load";
            return false;
        }
    }
    ci.codeSize = sizeof spirv_decode_comp;
    ci.pCode = spirv_decode_comp;
    vkCreateShaderModule(g_.device, &ci, nullptr, &decode_);
    cpi.stage.module = decode_;
    if (!decode_ ||
        vkCreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &decode_pipeline_) != VK_SUCCESS) {
        if (error) *error = "the texture decode shader did not load";
        return false;
    }
    // Local memory starts as zeros, the CPU's copy too.
    g_.Record([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, gsmem_, 0, kLocalMemory, 0);
        MemoryBarrier(cmd);
    });
    return true;
}

bool VulkanGsRenderer::MakeDepthPipelines() {
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    swizzle_depth_ = module(spirv_swizzle_depth_comp, sizeof spirv_swizzle_depth_comp);
    depth_load_vert_ = module(spirv_date_vert, sizeof spirv_date_vert);
    depth_load_frag_ = module(spirv_depth_load_frag, sizeof spirv_depth_load_frag);
    if (!swizzle_depth_ || !depth_load_vert_ || !depth_load_frag_) return false;
    // To memory: the swizzle with the depth buffer sampled as set 1.
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &depth_set_layout_) != VK_SUCCESS) return false;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   std::uint32_t(std::max(sizeof(SwizzleParams), sizeof(DecodeParams)))};
    const VkDescriptorSetLayout layouts[2] = {swizzle_set_layout_, depth_set_layout_};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 2;
    lci.pSetLayouts = layouts;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(g_.device, &lci, nullptr, &swizzle_depth_layout_) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = swizzle_depth_;
    cpi.stage.pName = "main";
    cpi.layout = swizzle_depth_layout_;
    if (vkCreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &swizzle_depth_pipeline_) !=
        VK_SUCCESS)
        return false;
    // Back: drawn, Z from memory as the depth, every pixel (ALWAYS) - the
    // stencil as it was.
    const VkPushConstantRange fragment_push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SwizzleParams)};
    VkPipelineLayoutCreateInfo fli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    fli.setLayoutCount = 1;
    fli.pSetLayouts = &swizzle_set_layout_;
    fli.pushConstantRangeCount = 1;
    fli.pPushConstantRanges = &fragment_push;
    if (vkCreatePipelineLayout(g_.device, &fli, nullptr, &depth_load_layout_) != VK_SUCCESS) return false;
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = depth_load_vert_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = depth_load_frag_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pColorBlendState = &blending;
    pci.pDepthStencilState = &depth;
    pci.pDynamicState = &dynamic;
    pci.layout = depth_load_layout_;
    return vkCreateGraphicsPipelines(g_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &depth_load_pipeline_) ==
           VK_SUCCESS;
}

VkDescriptorSet VulkanGsRenderer::DepthSampledSet(Image& image) {
    if (image.depth_sampled) return image.depth_sampled;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &depth_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.depth_sampled) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{samplers_[3], image.depth_view,
                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.depth_sampled;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.depth_sampled;
}

void VulkanGsRenderer::DepthOut(VkCommandBuffer cmd, const SwizzleParams& params, Image& d) {
    drawn_ = DrawState{};
    Transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorSet sets[2] = {swizzle_set_, DepthSampledSet(d)};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_depth_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_depth_layout_, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, swizzle_depth_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    ++g_dispatches;
    vkCmdDispatch(cmd, (params.cols + 7) / 8, (params.rows + 7) / 8, 1);
    g_.Mark(cmd, kProfSwizzle);
}

void VulkanGsRenderer::DepthIn(VkCommandBuffer cmd, const SwizzleParams& params, Image& d) {
    drawn_ = DrawState{};
    Transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    z.imageView = d.view;
    z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {d.pw(), d.ph()}};
    ri.layerCount = 1;
    ri.pDepthAttachment = &z;
    ri.pStencilAttachment = &z;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_load_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_load_layout_, 0, 1, &swizzle_set_, 0,
                            nullptr);
    vkCmdPushConstants(cmd, depth_load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof params, &params);
    const VkViewport vp{0, 0, float(d.pw()), float(d.ph()), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    const std::uint32_t k = d.scale;
    const std::uint32_t y0 = std::min(params.row0, d.height) * k;
    const std::uint32_t y1 = std::min(params.row0 + params.rows, d.height) * k;
    const VkRect2D sc{{0, std::int32_t(y0)}, {std::min(params.cols, d.width) * k, y1 - std::min(y0, y1)}};
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    g_.Mark(cmd, kProfSwizzle);
}

VkDescriptorSet VulkanGsRenderer::StorageSet(Image& image) {
    if (image.storage) return image.storage;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = storage_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &image_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.storage) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.storage;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.storage;
}

/* The swizzle on a colour target's image itself (GENERAL, as storage). */
void VulkanGsRenderer::SwizzleImage(VkCommandBuffer cmd, const SwizzleParams& params, Image& image) {
    drawn_ = DrawState{};
    Transition(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorSet sets[2] = {swizzle_set_, StorageSet(image)};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_image_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_image_layout_, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, swizzle_image_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    ++g_dispatches;
    vkCmdDispatch(cmd, (params.cols + 7) / 8, (params.rows + 7) / 8, 1);
    g_.Mark(cmd, kProfSwizzle);
}

void VulkanGsRenderer::MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) {
    if (end <= begin) return;
    Flush();
    const VkDeviceSize size = VkDeviceSize(end - begin) * 256;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(size, &buffer, &offset);
    if (!mapped) return;
    std::memcpy(mapped, bytes, std::size_t(size));
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        const VkBufferCopy copy{offset, VkDeviceSize(begin) * 256, size};
        vkCmdCopyBuffer(cmd, buffer, gsmem_, 1, &copy);
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::MemoryPatch(const std::uint32_t* words, std::size_t count) {
    if (!count || !patch_pipeline_) return;
    Flush();
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, patch_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, patch_layout_, 0, 1, &swizzle_set_, 0, nullptr);
        for (std::size_t i = 0; i < count; i += 8) {
            std::uint32_t push[32] = {};
            for (std::size_t k = 0; k < 8 && i + k < count; ++k) {
                push[4 * k + 0] = words[3 * (i + k) + 0];
                push[4 * k + 1] = words[3 * (i + k) + 1];
                push[4 * k + 2] = words[3 * (i + k) + 2];
            }
            vkCmdPushConstants(cmd, patch_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
            ++g_dispatches;
            vkCmdDispatch(cmd, 1, 1, 1);
        }
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) {
    if (end <= begin) return;
    Flush();
    const VkDeviceSize size = VkDeviceSize(end - begin) * 256;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return;
    g_.Run([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        const VkBufferCopy copy{VkDeviceSize(begin) * 256, 0, size};
        vkCmdCopyBuffer(cmd, gsmem_, buffer, 1, &copy);
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    }, "a memory read-back");
    std::memcpy(bytes, mapped, std::size_t(size));
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
}

bool VulkanGsRenderer::TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                                      std::uint32_t rows, std::uint32_t first_row) {
    Flush();   // a target the batch being drawn makes is found too
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    Image& t = found->second;
    cols = std::min(cols, t.width);
    rows = std::min(rows, t.height);
    if (first_row >= rows) return true;
    rows -= first_row;
    if (!cols) return true;
    g_.Record([&](VkCommandBuffer cmd) {
        SwizzleImage(cmd, SwizzleParams{fbp * 32, fbw, psm, cols, rows, t.pw(), 1, 0, first_row, t.scale}, t);
        MemoryBarrier(cmd);
    });
    return true;
}

void VulkanGsRenderer::TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                        std::uint32_t width, std::uint32_t height) {
    Flush();
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    const bool existed = targets_.count(Key{fbp, fbw, psm}) != 0;
    Image* t = TargetFor(state);
    SwizzleParams params{fbp * 32, fbw, psm, t->width, t->height, t->pw(), 0, 0, 0, t->scale};
    params.keep_colour = existed && keep_colour_ && t->scale > 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        SwizzleImage(cmd, params, *t);
        MemoryBarrier(cmd);
    });
}

bool VulkanGsRenderer::DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                                     std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows,
                                     std::uint32_t first_row) {
    Flush();   // a target the batch being drawn makes is found too
    auto found = depths_.find(Key{zbp, width, height});
    if (found == depths_.end()) return false;
    Image& d = found->second;
    cols = std::min(cols, d.width);
    rows = std::min(rows, d.height);
    if (first_row >= rows) return true;
    rows -= first_row;
    if (!cols) return true;
    g_.Record([&](VkCommandBuffer cmd) {
        DepthOut(cmd, SwizzleParams{zbp * 32, fbw, zpsm, cols, rows, d.pw(), 1, 1, first_row, d.scale}, d);
        MemoryBarrier(cmd);
    });
    return true;
}

void VulkanGsRenderer::TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                            std::uint32_t width, std::uint32_t height, std::uint32_t first_row,
                                            std::uint32_t rows) {
    // An image not made yet holds nothing: all of it, then.
    if (!targets_.count(Key{fbp, fbw, psm})) return TargetFromMemory(fbp, fbw, psm, width, height);
    Flush();
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    Image* t = TargetFor(state);
    if (first_row >= t->height) return;
    rows = std::min(rows, t->height - first_row);
    SwizzleParams params{fbp * 32, fbw, psm, t->width, rows, t->pw(), 0, 0, first_row, t->scale};
    params.keep_colour = keep_colour_ && t->scale > 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        SwizzleImage(cmd, params, *t);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                           std::uint32_t fbw, std::uint32_t zpsm, std::uint32_t first_row,
                                           std::uint32_t rows) {
    // The depth images are kept by size as well: one not made yet for this
    // size holds nothing, and comes whole.
    if (!depths_.count(Key{zbp, width, height})) return DepthFromMemory(zbp, width, height, fbw, zpsm);
    Flush();
    Image* d = DepthFor(zbp, width, height);
    if (first_row >= d->height) return;
    rows = std::min(rows, d->height - first_row);
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        DepthIn(cmd, SwizzleParams{zbp * 32, fbw, zpsm, d->width, rows, d->pw(), 0, 1, first_row, d->scale}, *d);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                       std::uint32_t fbw, std::uint32_t zpsm) {
    Flush();
    Image* d = DepthFor(zbp, width, height);
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        DepthIn(cmd, SwizzleParams{zbp * 32, fbw, zpsm, d->width, d->height, d->pw(), 0, 1, 0, d->scale}, *d);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::SetScale(std::uint32_t scale) {
    scale = std::min(std::max(1u, scale), kMaxScale);
    if (scale == scale_) return;
    // Every target, depth buffer and feedback copy goes (the caller has had
    // what the GPU drew written to its local memory first); they come back
    // at the new size, from memory, as they are next asked for. The linear
    // buffer holds an image's own pixels: it grows with the scale.
    Flush();
    g_.Submit();
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(g_.device);
    }
    drawn_ = DrawState{};
    open_target_ = open_depth_ = nullptr;
    feedback_ = nullptr;
    auto drop = [&](Image& image) {
        if (image.sampled) vkFreeDescriptorSets(g_.device, pool_, 1, &image.sampled);
        if (image.storage) vkFreeDescriptorSets(g_.device, storage_pool_, 1, &image.storage);
        if (image.depth_sampled) vkFreeDescriptorSets(g_.device, pool_, 1, &image.depth_sampled);
        if (image.input) vkFreeDescriptorSets(g_.device, input_pool_, 1, &image.input);
        DestroyImage(image);
    };
    for (auto& [key, t] : targets_) drop(t);
    for (auto& [key, d] : depths_) drop(d);
    for (auto& [size, copy] : feedbacks_) drop(copy);
    targets_.clear();
    depths_.clear();
    feedbacks_.clear();
    scale_ = scale;
}

VkPipeline VulkanGsRenderer::PipelineFor(std::uint32_t key) {
    // Batch after batch asks for the same one.
    if (key == last_pipeline_key_ && last_pipeline_) return last_pipeline_;
    auto found = pipelines_.find(key);
    if (found != pipelines_.end()) {
        last_pipeline_key_ = key;
        last_pipeline_ = found->second;
        return found->second;
    }
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = key & kShaderBlend ? frag_blend_ : frag_;
    stages[1].pName = "main";
    VkVertexInputBindingDescription binding{0, sizeof(PackedVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[6] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PackedVertex, x)},
        {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(PackedVertex, r)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PackedVertex, s)},
        {3, 0, VK_FORMAT_R32_UINT, offsetof(PackedVertex, z)},
        {4, 0, VK_FORMAT_R8_UNORM, offsetof(PackedVertex, fog)},
        {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PackedVertex, range)}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    input.vertexBindingDescriptionCount = 1;
    input.pVertexBindingDescriptions = &binding;
    input.vertexAttributeDescriptionCount = 6;
    input.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    const VkPipelineColorBlendAttachmentState blend = Blending(key, g_.dual_source);
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    if ((key >> 13) & 3) {                          // bits 13-14: the logic op of a partly masked draw
        blending.logicOpEnable = VK_TRUE;
        blending.logicOp = ((key >> 13) & 3) == 1 ? VK_LOGIC_OP_AND : VK_LOGIC_OP_OR;
    }
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    // Depth's test, write and comparison are set per batch (Vulkan 1.3).
    // The destination alpha test: drawn only where the setup pass set stencil 1.
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 1, 0, 1};
    depth.back = depth.front;
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                       VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 6;
    dynamic.pDynamicStates = dynamic_states;
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pColorBlendState = &blending;
    pci.pDepthStencilState = &depth;
    pci.pDynamicState = &dynamic;
    pci.layout = layout_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    pipelines_[key] = pipeline;
    return pipeline;
}

bool VulkanGsRenderer::MakeDatePipeline() {
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    VkShaderModule vert = module(spirv_date_vert, sizeof spirv_date_vert);
    VkShaderModule frag = module(spirv_date_frag, sizeof spirv_date_frag);
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.stencilTestEnable = VK_TRUE;
    depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 1, 1, 1};
    depth.back = depth.front;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blending;
    pci.pDynamicState = &dynamic;
    pci.layout = layout_;
    const VkResult r = vkCreateGraphicsPipelines(g_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &date_pipeline_);
    vkDestroyShaderModule(g_.device, vert, nullptr);
    vkDestroyShaderModule(g_.device, frag, nullptr);
    return r == VK_SUCCESS;
}

VkDescriptorSet VulkanGsRenderer::SampledSet(Image& image) {
    if (image.sampled) return image.sampled;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.sampled) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{samplers_[3], image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.sampled;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.sampled;
}

void VulkanGsRenderer::DateSetup(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm,
                                 const VkRect2D& area) {
    if (!SampledSet(target)) return;
    Transition(cmd, target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    Transition(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    z.imageView = depth.view;
    z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo stencil = z;
    stencil.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;   // 0 everywhere first
    stencil.clearValue.depthStencil = {0.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = area;
    ri.layerCount = 1;
    ri.pDepthAttachment = &z;
    ri.pStencilAttachment = &stencil;
    vkCmdBeginRendering(cmd, &ri);
    drawn_ = DrawState{};           // its own pipeline, sets, viewport and scissor
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, date_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &target.sampled, 0, nullptr);
    VkViewport vp{0, 0, float(target.pw()), float(target.ph()), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &area);
    FragmentParams params{};
    params.textured = datm;                         // the setup shader's first word is DATM
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, sizeof params, &params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    g_.Mark(cmd, kProfDate);
    vkCmdEndRendering(cmd);
}

bool VulkanGsRenderer::MakeDateInPass() {
    if (date_local_pipeline_ && check_pipeline_ && check_set_) return true;
    if (!g_.local_read) return false;
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    if (!input_set_layout_) {
        VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                             nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 1;
        dli.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &input_set_layout_) != VK_SUCCESS) return false;
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1024};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpi.maxSets = 1024;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &size;
        if (vkCreateDescriptorPool(g_.device, &dpi, nullptr, &input_pool_) != VK_SUCCESS) return false;
        // The draws' push constant ranges: theirs stay as they were set.
        VkPushConstantRange push[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, 8},
                                       {VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, sizeof(FragmentParams)}};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &input_set_layout_;
        lci.pushConstantRangeCount = 2;
        lci.pPushConstantRanges = push;
        if (vkCreatePipelineLayout(g_.device, &lci, nullptr, &date_local_layout_) != VK_SUCCESS) return false;
    }
    VkShaderModule vert = module(spirv_date_vert, sizeof spirv_date_vert);
    VkShaderModule local = module(spirv_date_local_frag, sizeof spirv_date_local_frag);
    VkShaderModule check = module(spirv_date_check_frag, sizeof spirv_date_check_frag);
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const VkPipelineColorBlendAttachmentState none{};       // the target there, nothing written to it
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blending;
    pci.pDynamicState = &dynamic;
    VkResult r = VK_SUCCESS;
    if (!date_local_pipeline_) {
        // In the draws' pass: its target and depth buffer, their depth state
        // set as theirs is (DrawState); the stencil replaced by the reference.
        stages[1].module = local;
        depth.stencilTestEnable = VK_TRUE;
        depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0};
        depth.back = depth.front;
        blending.attachmentCount = 1;
        blending.pAttachments = &none;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &format;
        const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                         VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                         VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                                         VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        dynamic.dynamicStateCount = 7;
        dynamic.pDynamicStates = states;
        pci.layout = date_local_layout_;
        r = vkCreateGraphicsPipelines(g_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &date_local_pipeline_);
        if (r != VK_SUCCESS) date_local_pipeline_ = VK_NULL_HANDLE;
    }
    if (!check_pipeline_) {
        // A pass of its own: drawn where the stencil is the reference.
        stages[1].module = check;
        depth.stencilTestEnable = VK_TRUE;
        depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 0xFF, 0, 0};
        depth.back = depth.front;
        blending.attachmentCount = 0;
        rendering.colorAttachmentCount = 0;
        const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                         VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        dynamic.dynamicStateCount = 3;
        dynamic.pDynamicStates = states;
        pci.layout = layout_;
        r = vkCreateGraphicsPipelines(g_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &check_pipeline_);
        if (r != VK_SUCCESS) check_pipeline_ = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(g_.device, vert, nullptr);
    vkDestroyShaderModule(g_.device, local, nullptr);
    vkDestroyShaderModule(g_.device, check, nullptr);
    if (!check_buffer_ && Buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &check_buffer_, &check_memory_, &check_mapped_)) {
        std::memset(check_mapped_, 0, 16);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &clut_set_layout_;
        if (vkAllocateDescriptorSets(g_.device, &ai, &check_set_) == VK_SUCCESS) {
            const VkDescriptorBufferInfo info{check_buffer_, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = check_set_;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &info;
            vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
        } else {
            check_set_ = VK_NULL_HANDLE;
        }
    }
    return date_local_pipeline_ && check_pipeline_ && check_set_;
}

int VulkanGsRenderer::SetDateInPass(int mode) {
    if (mode != 0 && !MakeDateInPass()) mode = 0;
    if (mode == date_in_pass_) return mode;
    // The targets' layout in the draws' pass changes: none open across.
    Flush();
    g_.Submit();
    date_in_pass_ = mode;
    return mode;
}

VkDescriptorSet VulkanGsRenderer::InputSet(Image& target) {
    if (target.input) return target.input;
    if (!input_pool_) return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = input_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &input_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &target.input) != VK_SUCCESS) return target.input = VK_NULL_HANDLE;
    VkDescriptorImageInfo info{VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = target.input;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return target.input;
}

void VulkanGsRenderer::DateSetupInPass(VkCommandBuffer cmd, Image& target, std::uint32_t datm, const VkRect2D& area) {
    // What the pass has drawn, visible to the reads of the target where it is drawn.
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    ++g_barriers;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT, 1, &b, 0, nullptr, 0, nullptr);
    DrawState& d = drawn_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, date_local_pipeline_);
    d.pipeline = date_local_pipeline_;
    // Its layout's set 0 is not the draws': theirs bound again after.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, date_local_layout_, 0, 1, &target.input, 0, nullptr);
    d.sets[0] = d.sets[1] = VK_NULL_HANDLE;
    d.size[0] = d.size[1] = -1;
    if (d.depth_test != 0) vkCmdSetDepthTestEnable(cmd, VkBool32(d.depth_test = 0));
    if (d.depth_write != 0) vkCmdSetDepthWriteEnable(cmd, VkBool32(d.depth_write = 0));
    if (d.compare < 0) vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS), d.compare = 1;
    if (d.stencil_test != 1) vkCmdSetStencilTestEnable(cmd, VkBool32(d.stencil_test = 1));
    const VkViewport vp{0, 0, float(target.pw()), float(target.ph()), 0, 1};
    if (std::memcmp(&vp, &d.viewport, sizeof vp) != 0) {
        d.viewport = vp;
        vkCmdSetViewport(cmd, 0, 1, &vp);
    }
    if (std::memcmp(&area, &d.scissor, sizeof area) != 0) {
        d.scissor = area;
        vkCmdSetScissor(cmd, 0, 1, &area);
    }
    // 0 over the area (as the pass of its own clears it), then 1 where the bit is DATM.
    for (const std::uint32_t value : {0u, 1u}) {
        const std::uint32_t which = value ? datm : 2u;
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, value);
        vkCmdPushConstants(cmd, date_local_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, 4, &which);
        ++g_draws;
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    g_.Mark(cmd, kProfDate);
}

void VulkanGsRenderer::DateCheck(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm,
                                 const VkRect2D& area) {
    if (!SampledSet(target) || !check_set_) return;
    ++checked_batches_;
    Transition(cmd, target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    Transition(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    // (the stencil written in the pass, visible to this one's tests)
    MemoryBarrier(cmd);
    VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    z.imageView = depth.view;
    z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = area;
    ri.layerCount = 1;
    ri.pDepthAttachment = &z;
    ri.pStencilAttachment = &z;
    vkCmdBeginRendering(cmd, &ri);
    drawn_ = DrawState{};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, check_pipeline_);
    const VkDescriptorSet sets[2] = {target.sampled, check_set_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 2, sets, 0, nullptr);
    const VkViewport vp{0, 0, float(target.pw()), float(target.ph()), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &area);
    // Stencil 1: the bit is DATM; 0: it is not; any other value: wrong anywhere.
    const std::uint32_t values[4][2] = {{1, datm}, {0, datm ^ 1u}, {2, 2}, {3, 2}};
    for (const auto& value : values) {
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, value[0]);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, 4, &value[1]);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    g_.Mark(cmd, kProfDate);
    vkCmdEndRendering(cmd);
}

bool VulkanGsRenderer::DateInPassCheck(std::uint64_t* batches, std::uint64_t* differ) {
    if (!check_mapped_) return false;
    Flush();
    g_.Record([&](VkCommandBuffer cmd) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0,
                             nullptr, 0, nullptr);
    });
    g_.Submit("DATE check");
    if (batches) *batches = checked_batches_;
    if (differ) *differ = static_cast<const volatile std::uint32_t*>(check_mapped_)[0];
    return true;
}

void VulkanGsRenderer::ReportDateCheck(bool always) {
    if (!check_mapped_) return;
    const std::uint32_t differ = static_cast<const volatile std::uint32_t*>(check_mapped_)[0];
    if (!always && differ == reported_differ_) return;
    reported_differ_ = differ;
    std::fprintf(stderr, "[ps2gpu] DATE within the pass checked: %llu batches, %u pixels differ (frame %llu)\n",
                 (unsigned long long)checked_batches_, differ, (unsigned long long)frame_);
}

VkDescriptorSet VulkanGsRenderer::SetFor(Texture& texture, int sampler) {
    VkDescriptorSet& set = texture.sets[sampler];
    if (set) return set;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{samplers_[sampler], texture.image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return set;
}

void VulkanGsRenderer::Fill(Texture& texture, const std::uint32_t* pixels) {
    Image& t = texture.image;
    const VkDeviceSize bytes = VkDeviceSize(t.width) * t.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    std::memcpy(mapped, pixels, bytes);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.width, t.height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
}

void VulkanGsRenderer::UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                                     std::uint32_t height) {
    Flush();                        // what is pending may sample the old contents
    Texture* texture = TextureImage(key, width, height);
    if (!texture) return;
    texture->replacement = 0;
    Fill(*texture, pixels);
}

bool VulkanGsRenderer::UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width,
                                         std::uint32_t height, std::uint32_t levels, const std::uint8_t* data,
                                         std::size_t bytes) {
    static const VkFormat kFormats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                        VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK,
                                        VK_FORMAT_BC7_UNORM_BLOCK};
    if (format > 4 || (format != 0 && !g_.bc) || width == 0 || height == 0) return false;
    levels = std::min(8u, std::max(1u, levels));
    // Each level's place in `data`.
    std::vector<VkBufferImageCopy> copies;
    VkDeviceSize at = 0;
    for (std::uint32_t n = 0; n < levels; ++n) {
        const std::uint32_t w = std::max(1u, width >> n), h = std::max(1u, height >> n);
        const VkDeviceSize size = format == 0 ? VkDeviceSize(w) * h * 4
                                              : VkDeviceSize((w + 3) / 4) * ((h + 3) / 4) * (format == 1 ? 8 : 16);
        if (at + size > bytes) break;
        VkBufferImageCopy copy{};
        copy.bufferOffset = at;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, n, 0, 1};
        copy.imageExtent = {w, h, 1};
        copies.push_back(copy);
        at += size;
    }
    if (copies.empty()) return false;
    Flush();
    auto old = replacements_.find(id);
    if (old != replacements_.end()) {
        replacement_bytes_ -= old->second.bytes;
        DestroyTexture(old->second);
        replacements_.erase(old);
    }
    Texture texture;
    if (!MakeImage(texture.image, width, height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   kFormats[format], 1, std::uint32_t(copies.size()))) {
        DestroyTexture(texture);
        return false;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(at, &buffer, &offset);
    if (!mapped) {
        DestroyTexture(texture);
        return false;
    }
    std::memcpy(mapped, data, std::size_t(at));
    for (VkBufferImageCopy& copy : copies) copy.bufferOffset += offset;
    Image& t = texture.image;
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdCopyBufferToImage(cmd, buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               std::uint32_t(copies.size()), copies.data());
        g_.Mark(cmd, kProfUpload);
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
    texture.replaced = 0x100u | (std::uint32_t(copies.size()) - 1) << 9;
    texture.last_used = frame_;
    texture.bytes = at;
    replacement_bytes_ += at;
    replacements_[id] = texture;
    return true;
}

void VulkanGsRenderer::EvictReplacements() {
    if (replacement_bytes_ <= kReplacementBudget) return;
    // The longest undrawn first, down to three quarters of the budget; none
    // drawn this frame or the one before.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> by_age;     // last_used, id
    for (const auto& [id, texture] : replacements_) by_age.emplace_back(texture.last_used, id);
    std::sort(by_age.begin(), by_age.end());
    for (const auto& [used, id] : by_age) {
        if (replacement_bytes_ <= kReplacementBudget / 4 * 3 || used + 2 > frame_) break;
        auto found = replacements_.find(id);
        replacement_bytes_ -= found->second.bytes;
        DestroyTexture(found->second);
        replacements_.erase(found);
    }
}

void VulkanGsRenderer::DestroyTexture(Texture texture) {
    // Recorded draws may still sample it: it goes after they have run.
    VkDevice device = g_.device;
    VkDescriptorPool pool = pool_;
    g_.AfterSubmit([this, device, pool, texture]() mutable {
        for (VkDescriptorSet& set : texture.sets)
            if (set) vkFreeDescriptorSets(device, pool, 1, &set);
        if (texture.image.view) vkDestroyImageView(device, texture.image.view, nullptr);
        if (texture.image.image) vkDestroyImage(device, texture.image.image, nullptr);
        if (texture.image.memory)
            FreeImageMemory(texture.image.memory, texture.image.offset, texture.image.piece_class,
                            texture.image.memory_type);
    });
}

Texture* VulkanGsRenderer::TextureImage(std::uint64_t key, std::uint32_t width, std::uint32_t height) {
    Texture& texture = textures_[key];
    if (texture.image.width != width || texture.image.height != height) {
        if (texture.image.image) g_.Submit("a texture resized");   // recorded draws may still sample it
        for (VkDescriptorSet& set : texture.sets)
            if (set) {
                vkFreeDescriptorSets(g_.device, pool_, 1, &set);
                set = VK_NULL_HANDLE;
            }
        DestroyImage(texture.image);
        auto pooled = texture_pool_.find(std::uint64_t(width) << 32 | height);
        if (pooled != texture_pool_.end() && !pooled->second.empty()) {
            texture = pooled->second.back();
            pooled->second.pop_back();
            --pooled_textures_;
            return &texture;
        }
        if (!MakeImage(texture.image, width, height,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
            return nullptr;
    }
    return &texture;
}

void VulkanGsRenderer::TextureFromMemory(std::uint64_t key, const GsTextureDecode& d) {
    Flush();                        // what is pending may sample the old contents
    // Mipmap levels below level 0, each at the left edge (gs.frag's Texel).
    const std::uint32_t levels = std::min(7u, std::max(1u, d.levels));
    std::uint32_t height = 0;
    for (std::uint32_t n = 0; n < levels; ++n) height += std::max(1u, d.th >> n);
    Texture* texture = TextureImage(key, d.tw, height);
    if (!texture) return;
    texture->replacement = 0;
    Image& t = texture->image;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        if (d.clut) vkCmdUpdateBuffer(cmd, clut_, 0, 1024, d.clut);
        MemoryBarrier(cmd);
        drawn_ = DrawState{};
        std::uint32_t y = 0;
        for (std::uint32_t n = 0; n < levels; ++n) {
            const std::uint32_t w = std::max(1u, d.tw >> n), h = std::max(1u, d.th >> n);
            const DecodeParams params{n ? d.level_tbp[n] : d.tbp, n ? std::max(1u, d.level_tbw[n]) : d.tbw, d.psm,
                                      w, h, d.wms, d.wmt, d.minu >> n, d.maxu >> n, d.minv >> n, d.maxv >> n,
                                      d.texa_lo, d.ta1, d.csa};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_layout_, 0, 1, &swizzle_set_, 0,
                                    nullptr);
            vkCmdPushConstants(cmd, swizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
            ++g_dispatches;
            vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
            g_.Mark(cmd, kProfDecode);
            MemoryBarrier(cmd);
            Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy copy{};
            copy.bufferRowLength = w;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageOffset = {0, std::int32_t(y), 0};
            copy.imageExtent = {std::min(w, t.width), std::min(h, t.height - y), 1};
            vkCmdCopyBufferToImage(cmd, linear_, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            g_.Mark(cmd, kProfDecode);
            y += h;
            // The next level's decode writes where this copy reads.
            if (n + 1 < levels) MemoryBarrier(cmd);
        }
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::ClutFromMemory(const GsClutLoad& load) {
    Flush();                        // what is pending may draw the CLUT, or sample through the old one
    DecodeParams params{load.cbp, 1, load.cpsm, load.eight ? 256u : 16u, 1, 0, 0, 0, 0, 0, 0,
                        load.texa_lo, load.ta1, load.csa};
    params.mode = 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        drawn_ = DrawState{};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_layout_, 0, 1, &swizzle_set_, 0,
                                nullptr);
        vkCmdPushConstants(cmd, swizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
        vkCmdDispatch(cmd, (params.tw + 7) / 8, 1, 1);
        g_.Mark(cmd, kProfDecode);
        MemoryBarrier(cmd);
    });
}

bool VulkanGsRenderer::ReadClut(std::uint32_t* clut) {
    Flush();
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(1024, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        VkBufferCopy copy{0, 0, 1024};
        vkCmdCopyBuffer(cmd, clut_, buffer, 1, &copy);
        MemoryBarrier(cmd);
    }, "a CLUT read-back");
    std::memcpy(clut, mapped, 1024);
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

void VulkanGsRenderer::ClutRows(const std::uint32_t* clut, std::uint32_t rows) {
    rows &= 0xFFFF;
    if (!rows) return;
    Flush();                        // what is pending samples through the old rows
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        for (std::uint32_t r = 0; r < 16;) {
            if (!(rows >> r & 1)) {
                ++r;
                continue;
            }
            std::uint32_t e = r;
            while (e < 16 && (rows >> e & 1)) ++e;
            vkCmdUpdateBuffer(cmd, clut_, VkDeviceSize(r) * 64, VkDeviceSize(e - r) * 64, clut + r * 16);
            r = e;
        }
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                    const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) {
    Flush();                        // what is pending draws into the old contents first
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    Image* t = TargetFor(state);
    const std::uint32_t w = std::min(width, t->width), h = std::min(height, t->height);
    const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    for (std::uint32_t y = 0; y < h; ++y)
        std::memcpy(static_cast<std::uint8_t*>(mapped) + std::size_t(y) * w * 4, pixels + std::size_t(y) * width,
                    std::size_t(w) * 4);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, *t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
    });
}

void VulkanGsRenderer::ReleaseTexture(std::uint64_t key) {
    auto found = textures_.find(key);
    if (found == textures_.end()) return;
    Flush();
    Texture texture = found->second;
    textures_.erase(found);
    texture.replacement = 0;
    // Recorded draws may still sample it: it goes after they have run - into
    // the pool, or away.
    if (texture.image.image && pooled_textures_ < kPooledTextures) {
        ++pooled_textures_;
        g_.AfterSubmit([this, texture] {
            texture_pool_[std::uint64_t(texture.image.width) << 32 | texture.image.height].push_back(texture);
        });
        return;
    }
    VkDevice device = g_.device;
    VkDescriptorPool pool = pool_;
    g_.AfterSubmit([this, device, pool, texture]() mutable {
        for (VkDescriptorSet& set : texture.sets)
            if (set) vkFreeDescriptorSets(device, pool, 1, &set);
        if (texture.image.view) vkDestroyImageView(device, texture.image.view, nullptr);
        if (texture.image.image) vkDestroyImage(device, texture.image.image, nullptr);
        if (texture.image.memory)
            FreeImageMemory(texture.image.memory, texture.image.offset, texture.image.piece_class,
                            texture.image.memory_type);
    });
}

bool VulkanGsRenderer::WeaveField(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t first_line,
                                  std::uint32_t lines, std::uint32_t parity, std::uint64_t* image, int* layout,
                                  std::uint32_t* width, std::uint32_t* height) {
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end() || !found->second.image || found->second.layout == VK_IMAGE_LAYOUT_UNDEFINED)
        return false;
    Image& t = found->second;
    if (first_line >= t.height) return false;
    lines = std::min(lines, t.height - first_line);
    if (!lines) return false;
    const bool fresh = !woven_.image || woven_.width != t.width || woven_.height != 2 * lines || woven_.scale != t.scale;
    if (fresh) {
        DestroyImage(woven_);
        if (!MakeImage(woven_, t.width, 2 * lines, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_FORMAT_R8G8B8A8_UNORM, t.scale))
            return false;
    }
    const std::uint32_t k = t.scale;
    std::vector<VkImageCopy> rows;
    rows.reserve(fresh ? 2 * lines : lines);
    for (std::uint32_t y = 0; y < lines; ++y)
        for (std::uint32_t side = 0; side < (fresh ? 2u : 1u); ++side) {
            VkImageCopy r{};
            r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.srcOffset = {0, std::int32_t((first_line + y) * k), 0};
            r.dstOffset = {0, std::int32_t((2 * y + ((parity + side) & 1)) * k), 0};
            r.extent = {t.pw(), k, 1};
            rows.push_back(r);
        }
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        Transition(cmd, woven_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdCopyImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, woven_.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, std::uint32_t(rows.size()), rows.data());
        Transition(cmd, woven_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
    });
    Kick();                             // the copy, on the queue before the picture's
    *image = reinterpret_cast<std::uint64_t>(woven_.image);
    *layout = int(woven_.layout);
    *width = woven_.pw();
    *height = woven_.ph();
    return true;
}

Image* VulkanGsRenderer::TargetFor(const GsDrawState& state) {
    Image& t = targets_[Key{state.fbp, state.fbw, state.psm}];
    if (t.image) return &t;
    MakeImage(t, state.width, state.height,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                  (g_.local_read ? VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT : 0u),
              VK_FORMAT_R8G8B8A8_UNORM, ScaleOf(state.width));
    // Local memory starts as zeros.
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearColorValue zero{};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        g_.Mark(cmd, kProfClear);
    });
    return &t;
}

Image* VulkanGsRenderer::DepthFor(std::uint32_t key, std::uint32_t width, std::uint32_t height) {
    Image& d = depths_[Key{key, width, height}];
    if (d.image) return &d;
    MakeImage(d, width, height,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              kDepthFormat, ScaleOf(width));
    // Local memory starts as zeros: the farthest Z.
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearDepthStencilValue zero{0.0f, 0};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, d.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        g_.Mark(cmd, kProfClear);
    });
    return &d;
}

void VulkanGsRenderer::UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                   const std::uint32_t* z) {
    Flush();
    Image* d = DepthFor(zbp, width, height);
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    float* out = static_cast<float*>(mapped);
    for (std::size_t i = 0; i < std::size_t(width) * height; ++i) out[i] = float(double(z[i]) / 4294967296.0);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, *d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, d->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
    });
}

void VulkanGsRenderer::DrawTriangles(const GsDrawState& state, const GsVertex* v, std::size_t count) {
    if (!SameBatch(state, state_)) {
        if (CountingBreaks() && !pending_.empty()) {
            g_flush_why = BreakReason(state, state_);
            if (state.textured != state_.textured) {
                // What else differs, the texture's state set aside (Stuntman:
                // the alpha test, its untextured primitives ATST NEVER with
                // AFAIL FB_ONLY between its textured GREATER ones).
                GsDrawState t = state;
                t.textured = state_.textured;
                t.texture = state_.texture;
                t.tfx = state_.tfx; t.tcc = state_.tcc; t.bilinear = state_.bilinear; t.mip = state_.mip;
                t.wms = state_.wms; t.wmt = state_.wmt; t.minu = state_.minu; t.maxu = state_.maxu;
                t.minv = state_.minv; t.maxv = state_.maxv; t.tw = state_.tw; t.th = state_.th;
                t.texture_target = state_.texture_target; t.tex_fbp = state_.tex_fbp; t.tex_fbw = state_.tex_fbw;
                t.tex_target_psm = state_.tex_target_psm; t.tex_format = state_.tex_format;
                t.ta0 = state_.ta0; t.ta1 = state_.ta1; t.aem = state_.aem;
                static std::set<std::string> names;
                g_flush_why = names.insert(std::string("textured on/off; then ") + BreakReason(t, state_)).first->c_str();
            }
        }
        Flush();
    }
    // A primitive sampling the target it draws into, where the batch has
    // drawn already: the batch so far first (the next takes a new copy).
    const bool feedback = state.textured && state.texture_target && state.tex_fbp == state.fbp &&
                          state.tex_fbw == state.fbw && count >= 3;
    const bool shader_blend = ShaderBlend(state) && count >= 3;
    // An alpha test whose failing pixels still write (AFAIL FB_ONLY or
    // ZB_ONLY with depth written, RGB_ONLY) is drawn in two passes over the
    // batch - the passing pixels of all its primitives, then the failing
    // ones: a primitive over one drawn before it in the batch begins a batch
    // of its own, or the earlier one's failing pixels land on top of it
    // (Burnout 3's crash titles: the black outline drawn four times, the
    // letters' fill over them - the outline's soft edges came out on top, as
    // dark streaks through the letters).
    const bool depth_written = state.depth && state.depth_write;
    const bool two_pass = state.alpha_test && state.atst != 1 && count >= 3 &&
                          (((state.afail == 1 || state.afail == 2) && depth_written) || state.afail == 3);
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (feedback || shader_blend || two_pass) {
        float fx0 = 1e30f, fy0 = 1e30f, fx1 = -1e30f, fy1 = -1e30f;
        float u0 = 1e30f, v0 = 1e30f, u1 = -1e30f, v1 = -1e30f;
        const float tw = float(state.tw ? state.tw : 2048), th = float(state.th ? state.th : 2048);
        for (std::size_t i = 0; i < count; ++i) {
            fx0 = std::min(fx0, v[i].x);
            fy0 = std::min(fy0, v[i].y);
            fx1 = std::max(fx1, v[i].x);
            fy1 = std::max(fy1, v[i].y);
            const float q = v[i].q != 0.0f ? v[i].q : 1.0f;
            const float u = v[i].s / q * tw, w = v[i].t / q * th;
            u0 = std::min(u0, u);
            v0 = std::min(v0, w);
            u1 = std::max(u1, u);
            v1 = std::max(v1, w);
        }
        // The pixels it covers, and the texels it reads (two by two around
        // the coordinate when bilinear).
        x0 = int(std::floor(fx0));
        y0 = int(std::floor(fy0));
        x1 = int(std::ceil(fx1)) - 1;
        y1 = int(std::ceil(fy1)) - 1;
        const float b = state.bilinear ? 0.5f : 0.0f;
        const int tu0 = int(std::floor(u0 - b)), tv0 = int(std::floor(v0 - b));
        const int tu1 = int(std::ceil(u1 + b)) - 1, tv1 = int(std::ceil(v1 + b)) - 1;
        const auto overlaps = [&](int l, int t, int r, int bottom) {
            return drawn_any_ && r >= drawn_x0_ && l <= drawn_x1_ && bottom >= drawn_y0_ && t <= drawn_y1_;
        };
        // Over what the batch drew: by the rectangles, and (exact) where a
        // triangle of it meets one the batch drew - triangles of a mesh side
        // by side share no pixel, and their order then changes nothing.
        const bool over = overlaps(x0, y0, x1, y1) && (!exact_overlap_ || count % 3 != 0 || MeetsBatch(v, count));
        if ((feedback && overlaps(tu0, tv0, tu1, tv1)) || ((shader_blend || two_pass) && over)) {
            if (CountingBreaks()) {
                static std::set<std::string> names;
                char two[160];
                std::snprintf(two, sizeof two,
                              "two-pass alpha test over itself (ATST %u AFAIL %u AREF %u ZTST %u zwrite %d FBMSK %08X PSM %02X blend %d)",
                              unsigned(state.atst), unsigned(state.afail), unsigned(state.aref), unsigned(state.ztst),
                              int(state.depth && state.depth_write), unsigned(state.fbmsk), unsigned(state.psm),
                              int(state.blend));
                g_flush_why = feedback ? "reads what the batch drew (feedback)"
                              : shader_blend ? "blends in the shader over itself"
                                             : names.insert(two).first->c_str();
            }
            Flush();
        }
        // What the copy must hold: the texels as CLAMP leaves them.
        if (feedback) {
            const int w = int(tw), h = int(th);
            auto axis = [&](int lo, int hi, std::uint32_t mode, std::uint32_t min, std::uint32_t max, int size,
                            int* a, int* b) {
                if (mode == 1) lo = std::clamp(lo, 0, size - 1), hi = std::clamp(hi, 0, size - 1);
                else if (mode == 2) lo = std::clamp(lo, int(min), int(max)), hi = std::clamp(hi, int(min), int(max));
                else if (mode == 3 || lo < 0 || hi >= size) return false;
                *a = lo;
                *b = hi;
                return true;
            };
            int ra = 0, rb = 0, rc = 0, rd = 0;
            if (axis(tu0, tu1, state.wms, state.minu, state.maxu, w, &ra, &rb) &&
                axis(tv0, tv1, state.wmt, state.minv, state.maxv, h, &rc, &rd))
                Reads(ra, rc, rb, rd);
            else
                read_all_ = true;
        }
        if (shader_blend) Reads(x0, y0, x1, y1);
    }
    state_ = state;
    if ((shader_blend || two_pass) && exact_overlap_) AddToBatch(v, count);
    if (feedback || shader_blend || two_pass) {
        if (!drawn_any_) {
            drawn_x0_ = x0;
            drawn_y0_ = y0;
            drawn_x1_ = x1;
            drawn_y1_ = y1;
        } else {
            drawn_x0_ = std::min(drawn_x0_, x0);
            drawn_y0_ = std::min(drawn_y0_, y0);
            drawn_x1_ = std::max(drawn_x1_, x1);
            drawn_y1_ = std::max(drawn_y1_, y1);
        }
        drawn_any_ = true;
    }
    // A sprite's S and T at its first and last pixels (the GS samples a
    // pixel at its whole coordinates: x from ceil(x0) to ceil(x1) - 1): no
    // further than the GS reads - a glyph of a font's page stays off its
    // neighbour's texels (a dark line between letters). Where a sprite meets
    // another of the batch edge to edge, on the screen and in the texture
    // (strips of a movie, a picture), both reach the outer edge of their
    // pixels there, half a pixel on, and meet as one: no seam either.
    // (Not kept: a range no coordinate reaches.)
    float range[4] = {-1e9f, -1e9f, 1e9f, 1e9f};
    if (state.sprite && count >= 3) {
        SpriteSpan sp{};
        sp.x0 = sp.x1 = v[0].x;
        sp.y0 = sp.y1 = v[0].y;
        for (std::size_t i = 0; i < count; ++i) {
            const float q = v[i].q != 0.0f ? v[i].q : 1.0f;
            if (v[i].x <= sp.x0) sp.x0 = v[i].x, sp.s0 = v[i].s / q;
            if (v[i].x >= sp.x1) sp.x1 = v[i].x, sp.s1 = v[i].s / q;
            if (v[i].y <= sp.y0) sp.y0 = v[i].y, sp.t0 = v[i].t / q;
            if (v[i].y >= sp.y1) sp.y1 = v[i].y, sp.t1 = v[i].t / q;
        }
        // The coordinate at a's position along an axis: its first or last
        // pixel's, `out` pixels further out.
        auto at = [](float a0, float a1, float c0, float c1, bool last_end, float out) {
            const float first = std::ceil(a0), last = std::max(first, std::ceil(a1) - 1.0f);
            const float per = a1 > a0 ? (c1 - c0) / (a1 - a0) : 0.0f;
            return last_end ? c0 + (last + out - a0) * per : c0 + (first - out - a0) * per;
        };
        range[0] = at(sp.x0, sp.x1, sp.s0, sp.s1, false, 0.0f);
        range[2] = at(sp.x0, sp.x1, sp.s0, sp.s1, true, 0.0f);
        range[1] = at(sp.y0, sp.y1, sp.t0, sp.t1, false, 0.0f);
        range[3] = at(sp.y0, sp.y1, sp.t0, sp.t1, true, 0.0f);
        {
            // (A target's strips too: the frame's copy to the display buffer in
            // 32-pixel strips - a faint line through text at each. The race's
            // glow, over strips of a buffer cut up in memory, comes a strip a
            // batch: those keep to the GS's reach.)
            auto same = [](float a, float b) { return std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(a)); };
            const std::size_t first_vertex = pending_.size();
            for (std::size_t n = sprites_.size(); n-- > 0 && sprites_.size() - n <= 256;) {
                SpriteSpan& o = sprites_[n];
                auto widen = [&](int mine, int theirs, float value_mine, float value_theirs) {
                    range[mine] = value_mine;
                    for (int k = 0; k < 6 && o.vertex + k < pending_.size(); ++k)
                        pending_[o.vertex + k].range[theirs] = value_theirs;
                };
                if (same(o.y0, sp.y0) && same(o.y1, sp.y1) && same(o.t0, sp.t0) && same(o.t1, sp.t1)) {
                    if (same(o.x1, sp.x0) && same(o.s1, sp.s0))         // it, then this one to its right
                        widen(0, 2, at(sp.x0, sp.x1, sp.s0, sp.s1, false, 0.5f),
                              at(o.x0, o.x1, o.s0, o.s1, true, 0.5f));
                    else if (same(o.x0, sp.x1) && same(o.s0, sp.s1))    // this one to its left
                        widen(2, 0, at(sp.x0, sp.x1, sp.s0, sp.s1, true, 0.5f),
                              at(o.x0, o.x1, o.s0, o.s1, false, 0.5f));
                }
                if (same(o.x0, sp.x0) && same(o.x1, sp.x1) && same(o.s0, sp.s0) && same(o.s1, sp.s1)) {
                    if (same(o.y1, sp.y0) && same(o.t1, sp.t0))         // it, then this one below
                        widen(1, 3, at(sp.y0, sp.y1, sp.t0, sp.t1, false, 0.5f),
                              at(o.y0, o.y1, o.t0, o.t1, true, 0.5f));
                    else if (same(o.y0, sp.y1) && same(o.t0, sp.t1))    // this one above
                        widen(3, 1, at(sp.y0, sp.y1, sp.t0, sp.t1, true, 0.5f),
                              at(o.y0, o.y1, o.t0, o.t1, false, 0.5f));
                }
            }
            sp.vertex = first_vertex;
            sprites_.push_back(sp);
        }
    }
    const bool flat = scale_ > 1 && state.textured && !state.sprite && !state.texture_target && count == 3 &&
                      v[0].z == v[1].z && v[1].z == v[2].z && v[0].q == v[1].q && v[1].q == v[2].q;
    if (!flat) last_flat_valid_ = false;
    if (flat) {
        auto same = [](const GsVertex& a, const GsVertex& b) {
            return a.x == b.x && a.y == b.y && a.s == b.s && a.t == b.t;
        };
        int shared = 0, lone = -1;
        if (last_flat_valid_) {
            for (int i = 0; i < 3; ++i) {
                bool found = false;
                for (int j = 0; j < 3; ++j) found = found || same(v[i], last_flat_.v[j]);
                if (found) ++shared;
                else lone = i;
            }
        }
        if (shared == 2 && lone >= 0) {
            // The second of a quad: the two kept to the quad's texels.
            FlatQuad q{};
            q.v[0] = last_flat_.v[0];
            q.v[1] = last_flat_.v[1];
            q.v[2] = last_flat_.v[2];
            q.v[3] = v[lone];
            q.vertex[0] = last_flat_.vertex;
            q.vertex[1] = pending_.size();
            FlatQuadRange(q, state);
            for (int n = 0; n < 4; ++n) range[n] = q.range[n];
            for (int k = 0; k < 3; ++k)
                for (int n = 0; n < 4; ++n) pending_[q.vertex[0] + k].range[n] = q.range[n];
            quads_.push_back(q);
            last_flat_valid_ = false;
        } else {
            last_flat_.v[0] = v[0];
            last_flat_.v[1] = v[1];
            last_flat_.v[2] = v[2];
            last_flat_.vertex = pending_.size();
            last_flat_valid_ = true;
        }
    }
    for (std::size_t i = 0; i + 2 < count; i += 3) {
        for (int k = 0; k < 3; ++k) {
            // Flat shading takes the colour of the triangle's last vertex.
            const GsVertex& c = state.gouraud ? v[i + k] : v[i + 2];
            const GsVertex& p = v[i + k];
            pending_.push_back({p.x, p.y, c.r, c.g, c.b, c.a, p.s, p.t, p.q, p.z, p.fog, {0, 0, 0},
                                {range[0], range[1], range[2], range[3]}, {0, 0, 0, 0}});
        }
    }
}

void VulkanGsRenderer::FlatQuadRange(FlatQuad& q, const GsDrawState& state) {
    const float tw = float(state.tw ? state.tw : 2048), th = float(state.th ? state.th : 2048);
    for (int a = 0; a < 2; ++a) {
        q.lo[a] = 1e30f;
        q.hi[a] = -1e30f;
    }
    for (const GsVertex& p : q.v) {
        const float w = p.q != 0.0f ? p.q : 1.0f;
        q.lo[0] = std::min(q.lo[0], p.s / w);
        q.hi[0] = std::max(q.hi[0], p.s / w);
        q.lo[1] = std::min(q.lo[1], p.t / w);
        q.hi[1] = std::max(q.hi[1], p.t / w);
    }
    // Half a texel in (bilinear reads a texel either side of the coordinate).
    const float in[2] = {(state.bilinear ? 0.5f : 0.01f) / tw, (state.bilinear ? 0.5f : 0.01f) / th};
    bool open_lo[2] = {false, false}, open_hi[2] = {false, false};
    // An edge shared with another quad of the batch (both its corners, in
    // place and texture): that side not kept, on either.
    auto corner_on = [](const FlatQuad& f, const GsVertex& p, int axis, bool high) {
        const float w = p.q != 0.0f ? p.q : 1.0f;
        const float c = axis == 0 ? p.s / w : p.t / w;
        return std::fabs(c - (high ? f.hi[axis] : f.lo[axis])) <= 1e-6f;
    };
    for (std::size_t n = quads_.size(); n-- > 0 && quads_.size() - n <= 256;) {
        FlatQuad& o = quads_[n];
        int shared[2], found = 0;
        for (int i = 0; i < 4 && found < 2; ++i)
            for (int j = 0; j < 4; ++j)
                if (q.v[i].x == o.v[j].x && q.v[i].y == o.v[j].y && q.v[i].s == o.v[j].s && q.v[i].t == o.v[j].t) {
                    shared[found++] = i;
                    break;
                }
        if (found < 2) continue;
        bool changed = false;
        for (int axis = 0; axis < 2; ++axis)
            for (int high = 0; high < 2; ++high) {
                if (!corner_on(q, q.v[shared[0]], axis, high) || !corner_on(q, q.v[shared[1]], axis, high)) continue;
                (high ? open_hi : open_lo)[axis] = true;
                // ...and the other quad's side through the same corners.
                for (int other = 0; other < 2; ++other)
                    if (corner_on(o, q.v[shared[0]], axis, other) && corner_on(o, q.v[shared[1]], axis, other)) {
                        o.range[axis + (other ? 2 : 0)] = other ? o.hi[axis] : o.lo[axis];
                        changed = true;
                    }
            }
        if (changed)
            for (int t = 0; t < 2; ++t)
                for (int k = 0; k < 3; ++k)
                    for (int c = 0; c < 4; ++c) pending_[o.vertex[t] + k].range[c] = o.range[c];
    }
    for (int axis = 0; axis < 2; ++axis) {
        q.range[axis] = open_lo[axis] ? q.lo[axis] : std::min(q.lo[axis] + in[axis], 0.5f * (q.lo[axis] + q.hi[axis]));
        q.range[axis + 2] = open_hi[axis] ? q.hi[axis] : std::max(q.hi[axis] - in[axis], 0.5f * (q.lo[axis] + q.hi[axis]));
    }
}

void VulkanGsRenderer::Flush() {
    if (CountingBreaks() && !pending_.empty()) ++g_breaks[g_flush_why ? g_flush_why : "a flush from elsewhere"];
    g_flush_why = nullptr;
    drawn_any_ = false;
    ClearBatchTriangles();
    sprites_.clear();
    quads_.clear();
    last_flat_valid_ = false;
    // What the batch reads of its own target (DrawTriangles), for its copy.
    const bool read_all = read_all_ || !read_any_;
    const int read_x0 = read_x0_, read_y0 = read_y0_, read_x1 = read_x1_, read_y1 = read_y1_;
    read_any_ = read_all_ = false;
    if (pending_.empty()) return;
    const VkDeviceSize bytes = pending_.size() * sizeof(PackedVertex);
    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VkDeviceSize vertex_offset = 0;
    void* vertex_data = Vertices(bytes, &vertex_buffer, &vertex_offset);
    if (!vertex_data) {
        pending_.clear();
        return;
    }
    std::memcpy(vertex_data, pending_.data(), bytes);
    Image* t = TargetFor(state_);
    // A target's own depth buffer is keyed past any ZBP (which has 9 bits).
    // A depth buffer is kDepthRows tall whatever draws with it (a 16-bit
    // target is twice as tall: the same memory), so it is one image.
    Image* depth = DepthFor(state_.depth ? state_.zbp : 0x10000u + state_.fbp, t->width,
                            state_.depth ? std::min(t->height, kDepthRows) : t->height);
    const VkPipeline pipeline = PipelineFor(PipelineKey(state_));
    Texture* texture = &blank_;
    if (state_.textured) {
        auto found = textures_.find(state_.texture);
        if (found != textures_.end()) texture = &found->second;
        // Drawn as a texture pack's replacement.
        if (texture->replacement) {
            auto replacement = replacements_.find(texture->replacement);
            if (replacement != replacements_.end()) {
                texture = &replacement->second;
                texture->last_used = frame_;
            }
        }
    }
    // The shader addresses and filters texels itself (texelFetch), as the GS does.
    const int sampler = 3;
    VkDescriptorSet set = SetFor(*texture, sampler);
    // Render to texture: the target itself, or a copy when it is also drawn into.
    Image* source = nullptr;
    bool copy_first = false;
    const bool shader_blend = ShaderBlend(state_);
    if (state_.textured && state_.texture_target) {
        auto found = targets_.find(Key{state_.tex_fbp, state_.tex_fbw, state_.tex_target_psm});
        if (found != targets_.end()) {
            source = &found->second;
            if (source == t) {
                Image& copy = feedbacks_[{t->width, t->height}];
                if (!copy.image)
                    MakeImage(copy, t->width, t->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_FORMAT_R8G8B8A8_UNORM, t->scale);
                feedback_ = &copy;
                copy_first = true;
                set = SampledSet(copy);
            } else {
                set = SampledSet(*source);
            }
        }
    }
    // Blending in the shader reads the target as the batch began: the copy.
    VkDescriptorSet destination_set = VK_NULL_HANDLE;
    if (shader_blend) {
        if (!copy_first) {
            Image& copy = feedbacks_[{t->width, t->height}];
            if (!copy.image)
                MakeImage(copy, t->width, t->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                          VK_FORMAT_R8G8B8A8_UNORM, t->scale);
            feedback_ = &copy;
            copy_first = true;
        }
        destination_set = SampledSet(*feedback_);
    }
    FragmentParams params{};
    params.textured = state_.textured && (texture != &blank_ || source);
    params.tfx = std::uint32_t(state_.tfx);
    params.tcc = state_.tcc;
    params.atst = state_.alpha_test ? state_.atst : 1;
    params.aref = state_.aref;
    // AFAIL: what a pixel failing the alpha test still writes. Drawn as up to
    // two passes - the passing pixels with everything, then the failing ones
    // with what AFAIL keeps (the shader's mode 0 drops failing pixels, 1
    // passing ones, 2 none).
    struct Pass { std::uint32_t mode; bool depth_write; std::uint32_t write_mask; };
    const bool depth_write = state_.depth && state_.depth_write;
    const std::uint32_t kept = KeptBits(state_.fbmsk, state_.psm);
    const std::uint32_t mask = WriteMask(kept);
    // A channel FBMSK masks only in part is drawn exactly with two logic ops -
    // which replace blending, so only where the blend leaves the source as it
    // is (Burnout 3's race writes two bits of each half of its frame's pixels
    // through a 16-bit view). Otherwise the channel is written whole.
    const bool source_only = !state_.blend || (state_.a == state_.b && state_.d == BlendColour::Source) ||
                             (state_.a == BlendColour::Source && state_.b == BlendColour::Zero &&
                              state_.c == BlendAlpha::Fix && state_.fix == 0x80 && state_.d == BlendColour::Zero);
    const bool logic = g_.logic_op && PartlyMasked(kept) && source_only;
    std::vector<Pass> passes;
    if (!state_.alpha_test || state_.atst == 1) {
        passes = {{2, depth_write, mask}};
    } else {
        switch (state_.afail) {
            case 0: passes = {{0, depth_write, mask}}; break;                                   // KEEP
            case 1:                                                                              // FB_ONLY
                if (depth_write) passes = {{0, true, mask}, {1, false, mask}};
                else passes = {{2, false, mask}};
                break;
            case 2:                                                                              // ZB_ONLY
                passes = {{0, depth_write, mask}};
                if (depth_write) passes.push_back({1, true, 0});
                break;
            default: passes = {{0, depth_write, mask}, {1, false, mask & 0x7u}}; break;          // RGB_ONLY
        }
    }
    params.fge = state_.fog;
    params.fogcol = state_.fogcol;
    params.fix_factor = state_.c == BlendAlpha::Fix;
    params.fix = state_.fix;
    params.bilinear = std::uint32_t(state_.bilinear) | (source ? 0u : state_.mip);   // gs.frag's Mip
    params.wms = state_.wms;
    params.wmt = state_.wmt;
    params.minu = state_.minu;
    params.maxu = state_.maxu;
    params.minv = state_.minv;
    params.maxv = state_.maxv;
    params.tw = state_.tw;
    params.th = state_.th;
    params.tex_format = source ? state_.tex_format : texture->replaced;     // gs.frag's Replaced
    params.ta0 = state_.ta0;
    params.ta1 = state_.ta1;
    params.aem = state_.aem;
    params.scale = source ? source->scale : 1;
    params.blend_ops = std::uint32_t(state_.a) | std::uint32_t(state_.b) << 2 | std::uint32_t(state_.c) << 4 |
                       std::uint32_t(state_.d) << 6;
    if (state_.textured && t->scale > 1) params.blend_ops |= (t->scale & 0xFFu) << 8;
    if (Is16Bit(state_.psm)) params.blend_ops |= 1u << 16;    // gs.frag: colour cut to five bits a channel
    const std::uint32_t count = std::uint32_t(pending_.size());
    static int complaints = 0;
    if ((!pipeline || !set) && complaints < 5) {
        ++complaints;
        std::fprintf(stderr, "[ps2gpu] a batch was dropped: %s\n", pipeline ? "no descriptor set" : "no pipeline");
    }
    // DATE: the setup pass over the pixels the batch can draw (its vertices'
    // rectangle, a pixel more each way, within the scissor), not the target:
    // 1,400 a VBlank in Stuntman's stunt, each over the whole frame.
    VkRect2D date_area{{0, 0}, {t->pw(), t->ph()}};
    if (state_.date && date_batch_area_ && !pending_.empty()) {
        float x0 = pending_[0].x, x1 = x0, y0 = pending_[0].y, y1 = y0;
        for (const PackedVertex& p : pending_) {
            x0 = std::min(x0, p.x);
            x1 = std::max(x1, p.x);
            y0 = std::min(y0, p.y);
            y1 = std::max(y1, p.y);
        }
        const int w = int(t->width), h = int(std::min(t->height, depth->height));
        const int ax0 = std::clamp(int(std::floor(x0)) - 1, std::max(0, state_.scissor_x0), w);
        const int ay0 = std::clamp(int(std::floor(y0)) - 1, std::max(0, state_.scissor_y0), h);
        const int ax1 = std::clamp(int(std::ceil(x1)) + 2, ax0, std::min(w, state_.scissor_x1 + 1));
        const int ay1 = std::clamp(int(std::ceil(y1)) + 2, ay0, std::min(h, state_.scissor_y1 + 1));
        const int k = int(t->scale);
        date_area = {{ax0 * k, ay0 * k}, {std::uint32_t((ax1 - ax0) * k), std::uint32_t((ay1 - ay0) * k)}};
    }
    if (pipeline && set) {
        g_.RecordInPass([&](VkCommandBuffer cmd) {
            // Barriers and copies cannot go inside a rendering pass: a batch
            // needing one, or drawing elsewhere, ends the open pass first.
            // DATE set up within the pass (date_in_pass_), where nothing else ends it.
            const bool in_pass = state_.date && date_in_pass_ != 0 && InputSet(*t);
            const bool outside = (state_.date && !in_pass) || copy_first ||
                                 (source && source->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (outside || open_target_ != t || open_depth_ != depth) EndPass(cmd);
            if (CountingBreaks() && in_pass) ++g_breaks["(DATE set up within the pass)"];
            if (CountingBreaks()) {     // (measure: DATE setups whose stencil the last left as it is)
                static Image* last_t = nullptr;
                static std::uint32_t last_datm = 9;
                const bool alpha = (mask & 8u) != 0;
                if (state_.date) {
                    ++g_breaks["(DATE setups)"];
                    if (last_t == t && last_datm == state_.datm) ++g_breaks["(DATE setups, stencil unchanged)"];
                }
                if (state_.date && !alpha) { last_t = t; last_datm = state_.datm; }
                else if (t == last_t && alpha) last_t = nullptr;
            }
            if (state_.date && !in_pass) DateSetup(cmd, *t, *depth, state_.datm, date_area);
            if (copy_first) {
                ++g_feedback;
                Transition(cmd, *t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
                Transition(cmd, *feedback_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = region.srcSubresource;
                region.extent = {t->pw(), t->ph(), 1};
                if (!read_all) {
                    // The texels it reads and one around (rounding), in the image's pixels.
                    const int k = int(t->scale);
                    const int x0 = std::clamp(read_x0 - 1, 0, int(t->width) - 1) * k;
                    const int y0 = std::clamp(read_y0 - 1, 0, int(t->height) - 1) * k;
                    const int x1 = (std::clamp(read_x1 + 1, 0, int(t->width) - 1) + 1) * k;
                    const int y1 = (std::clamp(read_y1 + 1, 0, int(t->height) - 1) + 1) * k;
                    region.srcOffset = {x0, y0, 0};
                    region.dstOffset = region.srcOffset;
                    region.extent = {std::uint32_t(x1 - x0), std::uint32_t(y1 - y0), 1};
                }
                vkCmdCopyImage(cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, feedback_->image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                g_.Mark(cmd, kProfFeedback);
                Transition(cmd, *feedback_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            } else if (source) {
                Transition(cmd, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            }
            const auto open_pass = [&] {
            if (open_target_) return;
            Transition(cmd, *t, TargetLayout(),
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           (date_in_pass_ ? VK_ACCESS_INPUT_ATTACHMENT_READ_BIT : 0u),
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           (date_in_pass_ ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : 0u));
            VkRenderingAttachmentInfo colour{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colour.imageView = t->view;
            colour.imageLayout = TargetLayout();
            colour.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            Transition(cmd, *depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
            VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            z.imageView = depth->view;
            z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {t->pw(), std::min(t->ph(), depth->ph())}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &colour;
            ri.pDepthAttachment = &z;
            ri.pStencilAttachment = &z;
            ++g_passes;
            vkCmdBeginRendering(cmd, &ri);
            open_target_ = t;
            open_depth_ = depth;
            };
            open_pass();
            if (in_pass) {
                if (drawn_.recording != g_.recording_id()) drawn_ = DrawState{g_.recording_id()};
                // (within the pass's area: a target taller than its depth buffer)
                VkRect2D area = date_area;
                const int bottom = int(std::min(t->ph(), depth->ph()));
                area.extent.height = std::uint32_t(std::max(0, std::min(int(area.offset.y + area.extent.height), bottom) - area.offset.y));
                DateSetupInPass(cmd, *t, state_.datm, area);
                if (date_in_pass_ == 2) {
                    EndPass(cmd);
                    DateCheck(cmd, *t, *depth, state_.datm, area);
                    open_pass();
                }
            }
            // ZTST: NEVER, ALWAYS, GEQUAL, GREATER - the larger Z is nearer.
            static const VkCompareOp kCompare[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS,
                                                    VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
            if (drawn_.recording != g_.recording_id()) drawn_ = DrawState{g_.recording_id()};
            DrawState& d = drawn_;
            const int depth_test = state_.depth ? 1 : 0, stencil_test = state_.date ? 1 : 0;
            const int compare = int(state_.ztst & 3);
            if (d.depth_test != depth_test) vkCmdSetDepthTestEnable(cmd, VkBool32(d.depth_test = depth_test));
            if (d.stencil_test != stencil_test) vkCmdSetStencilTestEnable(cmd, VkBool32(d.stencil_test = stencil_test));
            if (d.compare != compare) vkCmdSetDepthCompareOp(cmd, kCompare[d.compare = compare]);
            if (d.sets[0] != set || d.sets[1] != clut_set_) {
                d.sets[0] = set;
                d.sets[1] = clut_set_;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 2, d.sets, 0, nullptr);
            }
            if (destination_set)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 2, 1, &destination_set, 0,
                                        nullptr);
            const VkViewport vp{0, 0, float(t->pw()), float(t->ph()), 0, 1};
            if (std::memcmp(&vp, &d.viewport, sizeof vp) != 0) {
                d.viewport = vp;
                vkCmdSetViewport(cmd, 0, 1, &vp);
            }
            const std::int32_t x0 = std::max(0, state_.scissor_x0), y0 = std::max(0, state_.scissor_y0);
            const std::int32_t x1 = std::min(std::int32_t(t->width) - 1, state_.scissor_x1);
            const std::int32_t y1 = std::min(std::int32_t(std::min(t->height, depth->height)) - 1, state_.scissor_y1);
            const std::int32_t k = std::int32_t(t->scale);
            const VkRect2D sc{{x0 * k, y0 * k},
                              {std::uint32_t(std::max(0, x1 - x0 + 1) * k), std::uint32_t(std::max(0, y1 - y0 + 1) * k)}};
            if (std::memcmp(&sc, &d.scissor, sizeof sc) != 0) {
                d.scissor = sc;
                vkCmdSetScissor(cmd, 0, 1, &sc);
            }
            const float size[2] = {float(t->width), float(t->height)};
            if (size[0] != d.size[0] || size[1] != d.size[1]) {
                d.size[0] = size[0];
                d.size[1] = size[1];
                vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, size);
            }
            // The ring's chunk bound once, at its start; the batch by its first vertex.
            if (d.vertices != vertex_buffer) {
                d.vertices = vertex_buffer;
                const VkDeviceSize zero = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &zero);
            }
            const std::uint32_t first = std::uint32_t(vertex_offset / sizeof(PackedVertex));
            // PABE splits each pass again: blended where As >= 0x80 (the shader's
            // pabe 1), not blended below (pabe 2, the pipeline without blending).
            const bool split = state_.blend && state_.pabe;
            for (const Pass& pass : passes) {
                for (std::uint32_t half = split ? 1 : 0; half <= (split ? 2u : 0u); ++half) {
                    std::uint32_t key = (PipelineKey(state_) & ~(0xFu << 9)) | ((~pass.write_mask & 0xFu) << 9);
                    if (half == 2) key &= 0xFu << 9;                // blending off
                    // Partly masked: AND with the kept bits, then OR the rest
                    // in (a channel this pass leaves alone keeps every bit).
                    std::uint32_t keep = kept;
                    for (int channel = 0; channel < 4; ++channel)
                        if (!(pass.write_mask >> channel & 1)) keep |= 0xFFu << (8 * channel);
                    for (std::uint32_t op = logic ? 1 : 0; op <= (logic ? 2u : 0u); ++op) {
                        const std::uint32_t op_key = op ? ((key & (0xFu << 9)) | op << 13) : key;
                        const VkPipeline chosen = op_key == PipelineKey(state_) ? pipeline : PipelineFor(op_key);
                        if (!chosen) continue;
                        if (d.pipeline != chosen) {
                            d.pipeline = chosen;
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, chosen);
                        }
                        const int depth_write = pass.depth_write && op != 1 ? 1 : 0;
                        if (d.depth_write != depth_write)
                            vkCmdSetDepthWriteEnable(cmd, VkBool32(d.depth_write = depth_write));
                        params.afail = pass.mode;
                        params.pabe = half;
                        params.logic = op;
                        params.keep = keep;
                        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset,
                                           sizeof params, &params);
                        ++g_draws;
                        vkCmdDraw(cmd, count, 1, first, 0);
                        g_.Mark(cmd, kProfDraw);
                    }
                }
            }
        });
    }
    pending_.clear();
}

bool VulkanGsRenderer::ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                 std::vector<std::uint32_t>* z) {
    Flush();
    auto found = depths_.find(Key{zbp, width, height});
    if (found == depths_.end()) return false;
    Image& d = found->second;
    const VkDeviceSize bytes = VkDeviceSize(d.pw()) * d.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, d, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {d.pw(), d.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, d.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a depth read-back");
    // Z / 2^32 back to the GS's Z, a GS pixel each (the middle of each square).
    const float* in = static_cast<const float*>(mapped);
    z->resize(std::size_t(width) * height);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x) {
            const double v = double(in[std::size_t(y * d.scale + d.scale / 2) * d.pw() + x * d.scale + d.scale / 2]) * 4294967296.0 + 0.5;
            (*z)[std::size_t(y) * width + x] = v <= 0.0 ? 0u : v >= 4294967295.0 ? 0xFFFFFFFFu : std::uint32_t(v);
        }
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadImage(Image& t, std::vector<std::uint32_t>* pixels) {
    Flush();
    const VkDeviceSize bytes = VkDeviceSize(t.width) * t.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.width, t.height, 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a texture read-back");
    pixels->assign(std::size_t(t.width) * t.height, 0);
    std::memcpy(pixels->data(), mapped, bytes);
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                                   std::uint32_t* height) {
    auto found = textures_.find(key);
    if (found == textures_.end() || !found->second.image.image) return false;
    *width = found->second.image.width;
    *height = found->second.image.height;
    return ReadImage(found->second.image, pixels);
}

bool VulkanGsRenderer::ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                      std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                                      std::uint32_t* height) {
    Flush();
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    return ReadImage(found->second, pixels, width, height);
}

bool VulkanGsRenderer::ReadImage(Image& t, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                                 std::uint32_t* height) {
    const VkDeviceSize bytes = VkDeviceSize(t.pw()) * t.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.pw(), t.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a target read-back");
    const std::uint32_t* from = static_cast<const std::uint32_t*>(mapped);
    pixels->assign(from, from + std::size_t(t.pw()) * t.ph());
    *width = t.pw();
    *height = t.ph();
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                  std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height) {
    Flush();
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    Image& t = found->second;
    const VkDeviceSize bytes = VkDeviceSize(t.pw()) * t.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.pw(), t.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a target read-back");
    pixels->assign(std::size_t(t.width) * t.height, 0);
    {
        const std::uint32_t* from = static_cast<const std::uint32_t*>(mapped);
        for (std::uint32_t y = 0; y < t.height; ++y)
            for (std::uint32_t x = 0; x < t.width; ++x)
                (*pixels)[std::size_t(y) * t.width + x] = from[std::size_t(y * t.scale + t.scale / 2) * t.pw() + x * t.scale + t.scale / 2];
    }
    *width = t.width;
    *height = t.height;
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

VulkanGsRenderer::~VulkanGsRenderer() {
    g_.Submit();
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(g_.device);
    }
    for (auto& [key, t] : targets_) DestroyImage(t);
    for (auto& [key, d] : depths_) DestroyImage(d);
    for (auto& [key, texture] : textures_) DestroyImage(texture.image);
    for (auto& [size, list] : texture_pool_)
        for (Texture& texture : list) DestroyImage(texture.image);
    DestroyImage(blank_.image);
    for (auto& [size, copy] : feedbacks_) DestroyImage(copy);
    for (VkDeviceMemory block : image_blocks_) vkFreeMemory(g_.device, block, nullptr);     // (ImageMemory's)
    image_blocks_.clear();
    image_pieces_.clear();
    for (const Staged& staged : staging_free_) {
        vkUnmapMemory(g_.device, staged.memory);
        vkDestroyBuffer(g_.device, staged.buffer, nullptr);
        vkFreeMemory(g_.device, staged.memory, nullptr);
    }
    for (VertexRing* rings : {vertex_rings_, staging_rings_})
        for (int slot = 0; slot < VulkanGpu::kSlots; ++slot)
        for (VertexChunk& chunk : rings[slot].chunks) {
            vkDestroyBuffer(g_.device, chunk.buffer, nullptr);
            vkFreeMemory(g_.device, chunk.memory, nullptr);
        }
    for (auto& [key, pipeline] : pipelines_) vkDestroyPipeline(g_.device, pipeline, nullptr);
    if (date_pipeline_) vkDestroyPipeline(g_.device, date_pipeline_, nullptr);
    if (date_in_pass_ == 2) ReportDateCheck(true);
    for (VkPipeline p : {date_local_pipeline_, check_pipeline_})
        if (p) vkDestroyPipeline(g_.device, p, nullptr);
    if (date_local_layout_) vkDestroyPipelineLayout(g_.device, date_local_layout_, nullptr);
    if (input_pool_) vkDestroyDescriptorPool(g_.device, input_pool_, nullptr);
    if (input_set_layout_) vkDestroyDescriptorSetLayout(g_.device, input_set_layout_, nullptr);
    if (check_buffer_) vkDestroyBuffer(g_.device, check_buffer_, nullptr);
    if (check_memory_) vkFreeMemory(g_.device, check_memory_, nullptr);
    for (VkSampler s : samplers_)
        if (s) vkDestroySampler(g_.device, s, nullptr);
    if (pool_) vkDestroyDescriptorPool(g_.device, pool_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(g_.device, set_layout_, nullptr);
    if (clut_set_layout_) vkDestroyDescriptorSetLayout(g_.device, clut_set_layout_, nullptr);
    if (layout_) vkDestroyPipelineLayout(g_.device, layout_, nullptr);
    if (vert_) vkDestroyShaderModule(g_.device, vert_, nullptr);
    if (frag_) vkDestroyShaderModule(g_.device, frag_, nullptr);
    if (frag_blend_) vkDestroyShaderModule(g_.device, frag_blend_, nullptr);
    if (swizzle_pipeline_) vkDestroyPipeline(g_.device, swizzle_pipeline_, nullptr);
    if (swizzle_layout_) vkDestroyPipelineLayout(g_.device, swizzle_layout_, nullptr);
    if (swizzle_pool_) vkDestroyDescriptorPool(g_.device, swizzle_pool_, nullptr);
    if (swizzle_set_layout_) vkDestroyDescriptorSetLayout(g_.device, swizzle_set_layout_, nullptr);
    if (swizzle_) vkDestroyShaderModule(g_.device, swizzle_, nullptr);
    if (decode_pipeline_) vkDestroyPipeline(g_.device, decode_pipeline_, nullptr);
    if (decode_) vkDestroyShaderModule(g_.device, decode_, nullptr);
    for (auto [buffer, memory] : {std::pair{gsmem_, gsmem_memory_}, std::pair{linear_, linear_memory_},
                                  std::pair{clut_, clut_memory_}}) {
        if (buffer) vkDestroyBuffer(g_.device, buffer, nullptr);
        if (memory) vkFreeMemory(g_.device, memory, nullptr);
    }
}

}  // namespace

std::uint64_t GpuBusyNanoseconds() { return g_gpu_ns.load(); }

std::string GpuProfileReport() {
    static const char* const kNames[kProfKinds] = {"other", "draws", "swizzle", "image<->linear", "memory copies",
                                                   "feedback copies", "decodes", "uploads", "clears", "DATE",
                                                   "read-backs"};
    std::uint64_t total = 0, ns[kProfKinds];
    for (int i = 0; i < kProfKinds; ++i) total += ns[i] = g_profile_ns[i].exchange(0);
    if (!total) return "";
    std::string out = "GPU " + std::to_string(total / 1000000) + " ms:";
    for (int i = 0; i < kProfKinds; ++i) {
        if (!ns[i]) continue;
        char text[64];
        std::snprintf(text, sizeof text, " %s %.1f%%", kNames[i], 100.0 * double(ns[i]) / double(total));
        out += text;
    }
    return out;
}

std::string RendererCounters() {
    static std::uint64_t last[6] = {};
    const std::uint64_t now[6] = {g_barriers, g_transitions, g_dispatches, g_draws, g_passes, g_feedback};
    char text[200];
    std::snprintf(text, sizeof text, "%llu barriers, %llu transitions, %llu dispatches, %llu draws, %llu passes, %llu feedback copies",
                  (unsigned long long)(now[0] - last[0]), (unsigned long long)(now[1] - last[1]),
                  (unsigned long long)(now[2] - last[2]), (unsigned long long)(now[3] - last[3]),
                  (unsigned long long)(now[4] - last[4]), (unsigned long long)(now[5] - last[5]));
    for (int i = 0; i < 6; ++i) last[i] = now[i];
    std::string out = text;
    if (!g_breaks.empty()) {
        std::vector<std::pair<std::uint64_t, std::string>> by;
        for (const auto& [why, n] : g_breaks) by.push_back({n, why});
        std::sort(by.rbegin(), by.rend());
        out += "; batches ended by";
        for (const auto& [n, why] : by) out += " " + why + " " + std::to_string(n) + ",";
        out.pop_back();
        g_breaks.clear();
    }
    return out;
}

std::string SubmitStats() {
    char text[160];
    std::snprintf(text, sizeof text, "%llu submissions, the GPU busy %.1f s, %.1f s waiting for them",
                  static_cast<unsigned long long>(g_submits), double(g_gpu_ns.load()) / 1e9, g_wait_seconds);
    std::string out = text;
    for (const WaitReason& w : g_waits) {
        if (!w.why || !w.count) continue;
        std::snprintf(text, sizeof text, "; %s %llu, %.1f s", w.why, static_cast<unsigned long long>(w.count),
                      w.seconds);
        out += text;
    }
    return out;
}

std::unique_ptr<GsRenderer> CreateGsRenderer(Gpu& gpu, std::string* error) {
    auto renderer = std::make_unique<VulkanGsRenderer>(static_cast<VulkanGpu&>(gpu));
    if (!renderer->Init(error)) return nullptr;
    return renderer;
}

bool TrianglesShareAPixel(const float a[6], const float b[6]) {
    BatchTriangle x{}, y{};
    for (int i = 0; i < 3; ++i) {
        x.x[i] = a[2 * i];
        x.y[i] = a[2 * i + 1];
        y.x[i] = b[2 * i];
        y.y[i] = b[2 * i + 1];
    }
    Prepare(x);
    Prepare(y);
    const bool direct = InteriorsMeet(x, y), prepared = InteriorsMeetPrepared(x, y);
    return direct == prepared ? direct : !direct;   // (a disagreement fails its test either way)
}

bool TrianglesShareAPixelBothWays(const float a[6], const float b[6], bool* agree) {
    BatchTriangle x{}, y{};
    for (int i = 0; i < 3; ++i) {
        x.x[i] = a[2 * i];
        x.y[i] = a[2 * i + 1];
        y.x[i] = b[2 * i];
        y.y[i] = b[2 * i + 1];
    }
    Prepare(x);
    Prepare(y);
    const bool direct = InteriorsMeet(x, y);
    *agree = direct == InteriorsMeetPrepared(x, y) && direct == InteriorsMeetPrepared(y, x);
    return direct;
}

}  // namespace ps2gpu
