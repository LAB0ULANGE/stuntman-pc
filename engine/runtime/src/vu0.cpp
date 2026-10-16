/*
 * VU0 macro mode for lifted code: a line-for-line mirror of engine/ps2decomp/vu0.py,
 * which documents the arithmetic model (and where it departs from the hardware).
 * The two are tested against each other through the lifted-code check.
 */

#include "ps2rt/lifted.h"
#include "ps2rt/vu_math.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <xmmintrin.h>

namespace lifted {

namespace {

/* The special1 FMAC table: (operation, operand: 0-3 a lane of ft, 4 Q, 5 I, 6 ft). */
struct Fmac { int op; int operand; bool to_acc; };

bool Special1(u32 funct, Fmac* out) {
    static const int kBroadcast[7] = {kAdd, kSub, kMadd, kMsub, kMax_, kMini, kMul};
    if (funct <= 0x1B) { *out = {kBroadcast[funct >> 2], (int)(funct & 3), false}; return true; }
    switch (funct) {
        case 0x1C: *out = {kMul, 4, false}; return true;
        case 0x1D: *out = {kMax_, 5, false}; return true;
        case 0x1E: *out = {kMul, 5, false}; return true;
        case 0x1F: *out = {kMini, 5, false}; return true;
        case 0x20: *out = {kAdd, 4, false}; return true;
        case 0x21: *out = {kMadd, 4, false}; return true;
        case 0x22: *out = {kAdd, 5, false}; return true;
        case 0x23: *out = {kMadd, 5, false}; return true;
        case 0x24: *out = {kSub, 4, false}; return true;
        case 0x25: *out = {kMsub, 4, false}; return true;
        case 0x26: *out = {kSub, 5, false}; return true;
        case 0x27: *out = {kMsub, 5, false}; return true;
        case 0x28: *out = {kAdd, 6, false}; return true;
        case 0x29: *out = {kMadd, 6, false}; return true;
        case 0x2A: *out = {kMul, 6, false}; return true;
        case 0x2B: *out = {kMax_, 6, false}; return true;
        case 0x2C: *out = {kSub, 6, false}; return true;
        case 0x2D: *out = {kMsub, 6, false}; return true;
        case 0x2F: *out = {kMini, 6, false}; return true;
    }
    return false;
}

/* The special2 accumulator forms, by their index. */
bool AccumulatorForm(u32 index, Fmac* out) {
    static const int kBroadcast[4] = {kAdd, kSub, kMadd, kMsub};
    if (index <= 0x0F) { *out = {kBroadcast[index >> 2], (int)(index & 3), true}; return true; }
    if (index >= 0x18 && index <= 0x1B) { *out = {kMul, (int)(index & 3), true}; return true; }
    switch (index) {
        case 0x1C: *out = {kMul, 4, true}; return true;
        case 0x1E: *out = {kMul, 5, true}; return true;
        case 0x20: *out = {kAdd, 4, true}; return true;
        case 0x21: *out = {kMadd, 4, true}; return true;
        case 0x22: *out = {kAdd, 5, true}; return true;
        case 0x23: *out = {kMadd, 5, true}; return true;
        case 0x24: *out = {kSub, 4, true}; return true;
        case 0x25: *out = {kMsub, 4, true}; return true;
        case 0x26: *out = {kSub, 5, true}; return true;
        case 0x27: *out = {kMsub, 5, true}; return true;
        case 0x28: *out = {kAdd, 6, true}; return true;
        case 0x29: *out = {kMadd, 6, true}; return true;
        case 0x2A: *out = {kMul, 6, true}; return true;
        case 0x2C: *out = {kSub, 6, true}; return true;
        case 0x2D: *out = {kMsub, 6, true}; return true;
    }
    return false;
}

}  // namespace

}  // namespace lifted

namespace generated {

u32 VuAdd(u32 a, u32 b) { return lifted::Binary(lifted::kAdd, a, b, 0); }
u32 VuSub(u32 a, u32 b) { return lifted::Binary(lifted::kSub, a, b, 0); }
u32 VuMul(u32 a, u32 b) { return lifted::Binary(lifted::kMul, a, b, 0); }
u32 VuMax(u32 a, u32 b) { return lifted::Binary(lifted::kMax_, a, b, 0); }
u32 VuMin(u32 a, u32 b) { return lifted::Binary(lifted::kMini, a, b, 0); }
u32 VuMadd(u32 acc, u32 a, u32 b) { return lifted::Binary(lifted::kMadd, a, b, acc); }
u32 VuMsub(u32 acc, u32 a, u32 b) { return lifted::Binary(lifted::kMsub, a, b, acc); }
u32 VuAbs(u32 a) { return lifted::Clean(a) & ~lifted::kSign; }
u32 VuItof(u32 a, u32 shift) { return lifted::Result((double)(s32)a / (double)(1u << shift)); }
u32 VuFtoi(u32 a, u32 shift) {
    return lifted::FloatToInt(lifted::Value(a) * (double)(1u << shift));
}
u32 VuDiv(u32 num, u32 den) {
    return lifted::Value(den) == 0.0 ? ((num ^ den) & lifted::kSign) | lifted::kMax
                                     : lifted::Result(lifted::Value(num) / lifted::Value(den));
}
u32 VuSqrt(u32 a) { return lifted::Result(std::sqrt(std::fabs(lifted::Value(a)))); }
u32 VuRsqrt(u32 num, u32 den) {
    const double root = std::sqrt(std::fabs(lifted::Value(den)));
    return root == 0.0 ? (num & lifted::kSign) | lifted::kMax
                       : lifted::Result(lifted::Value(num) / root);
}

}  // namespace generated

namespace lifted {

bool DecodeFmac(u32 word, FmacOp* out) {
    const u32 funct = word & 0x3F;
    Fmac fmac;
    const bool is_fmac = funct < 0x3C ? Special1(funct, &fmac)
                                      : AccumulatorForm((word & 3) | ((word >> 4) & 0x7C), &fmac);
    if (!is_fmac) return false;
    out->op = fmac.op;
    out->operand = fmac.operand;
    out->to_acc = fmac.to_acc;
    out->fd = (word >> 6) & 0x1F;
    out->fs = (word >> 11) & 0x1F;
    out->ft = (word >> 16) & 0x1F;
    out->dest = (word >> 21) & 0xF;
    return true;
}

namespace {

#if PS2RT_VU_SSE
template <int Op>
void Lanes(const Context& c, const FmacOp& f, u32 out[4]) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vf[f.fs]));
    __m128i b;
    if (f.operand == 6) b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vf[f.ft]));
    else b = _mm_set1_epi32(int(f.operand < 4 ? c.vf[f.ft][f.operand]
                              : f.operand == 4 ? c.vc[kRegQ] : c.vc[kRegI]));
    const __m128i acc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vacc));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), LanesSse<Op>(a, b, acc));
}
#else
template <int Op>
void Lanes(const Context& c, const FmacOp& f, u32 out[4]) {
    const u32* fs = c.vf[f.fs];
    const u32 broadcast = f.operand < 4 ? c.vf[f.ft][f.operand]
                        : f.operand == 4 ? c.vc[kRegQ] : c.vf[f.ft][0];
    const u32 each = f.operand == 5 ? c.vc[kRegI] : broadcast;
    for (int lane = 0; lane < 4; ++lane) {
        if (!(f.dest & (8u >> lane))) continue;
        const u32 b = f.operand == 6 ? c.vf[f.ft][lane] : each;
        out[lane] = BinaryOf<Op>(fs[lane], b, c.vacc[lane]);
    }
}
#endif

}  // namespace

void RunFmacScalar(Context& c, const FmacOp& f) {
    u32 out[4];
    const u32 broadcast = f.operand < 4 ? c.vf[f.ft][f.operand]
                        : f.operand == 4 ? c.vc[kRegQ] : f.operand == 5 ? c.vc[kRegI] : 0;
    for (int lane = 0; lane < 4; ++lane) {
        if (!(f.dest & (8u >> lane))) continue;
        out[lane] = Binary(f.op, c.vf[f.fs][lane], f.operand == 6 ? c.vf[f.ft][lane] : broadcast, c.vacc[lane]);
    }
    u32* to = f.to_acc ? c.vacc : f.fd ? c.vf[f.fd] : nullptr;
    if (!to) return;
    for (int lane = 0; lane < 4; ++lane)
        if (f.dest & (8u >> lane)) to[lane] = out[lane];
}

void FmacResult(const Context& c, const FmacOp& f, u32 out[4]) {
    u32 lanes[4] = {};
    switch (f.op) {
        case kAdd: Lanes<kAdd>(c, f, lanes); break;
        case kSub: Lanes<kSub>(c, f, lanes); break;
        case kMul: Lanes<kMul>(c, f, lanes); break;
        case kMadd: Lanes<kMadd>(c, f, lanes); break;
        case kMsub: Lanes<kMsub>(c, f, lanes); break;
        case kMax_: Lanes<kMax_>(c, f, lanes); break;
        default: Lanes<kMini>(c, f, lanes); break;
    }
    for (int lane = 0; lane < 4; ++lane) out[lane] = (f.dest & (8u >> lane)) ? lanes[lane] : 0;
}

void RunFmac(Context& c, const FmacOp& f) {
    // Every lane is read before any is written (fd may be fs or ft); a lane
    // dest leaves alone is not computed at all - the arithmetic has no other
    // effect.
    u32 out[4];
    switch (f.op) {
        case kAdd: Lanes<kAdd>(c, f, out); break;
        case kSub: Lanes<kSub>(c, f, out); break;
        case kMul: Lanes<kMul>(c, f, out); break;
        case kMadd: Lanes<kMadd>(c, f, out); break;
        case kMsub: Lanes<kMsub>(c, f, out); break;
        case kMax_: Lanes<kMax_>(c, f, out); break;
        default: Lanes<kMini>(c, f, out); break;
    }
    u32* to = f.to_acc ? c.vacc : f.fd ? c.vf[f.fd] : nullptr;
    if (!to) return;
    for (int lane = 0; lane < 4; ++lane)
        if (f.dest & (8u >> lane)) to[lane] = out[lane];
}

/* The VU's float arithmetic rounds toward zero and flushes denormals: the
   host's SSE control word set so for an instruction, as ps2host's Vu::Run sets
   it for a microprogram (VuRounding) - macro mode is the same VU0, called from
   EE code, which runs with the host's default (to nearest).
   PS2RT_EE_ROUND_NEAREST=1 leaves it to nearest, as emu.py's interpreter
   (eefpu.cpp: what verification compares against). */
const bool kMacroNearest = std::getenv("PS2RT_EE_ROUND_NEAREST") != nullptr;

struct MacroRounding {
    unsigned saved = _mm_getcsr();
    MacroRounding() { if (!kMacroNearest) _mm_setcsr((saved & ~0x6000u) | 0x6000u | 0x8000u | 0x0040u); }
    ~MacroRounding() { if (!kMacroNearest) _mm_setcsr(saved); }
};

void Cop2(Context& c, u32 word) {
    MacroRounding rounding;
    const u32 fd = (word >> 6) & 0x1F, fs = (word >> 11) & 0x1F, ft = (word >> 16) & 0x1F;
    const u32 dest = (word >> 21) & 0xF;
    const bool lane_on[4] = {(dest & 8) != 0, (dest & 4) != 0, (dest & 2) != 0, (dest & 1) != 0};
    auto write = [&](u32 index, const u32 values[4]) {
        if (index == 0) return;
        for (int lane = 0; lane < 4; ++lane)
            if (lane_on[lane]) c.vf[index][lane] = values[lane];
    };

    const u32 funct = word & 0x3F;
    const u32 index = funct < 0x3C ? 0 : (word & 3) | ((word >> 4) & 0x7C);
    FmacOp fmac;
    if (DecodeFmac(word, &fmac)) {
        RunFmac(c, fmac);
        return;
    }

    const u32* a = c.vf[fs];
    if (funct == 0x2E) {                                   /* vopmsub */
        const u32* bb = c.vf[ft];
        const u32 out[3] = {Binary(kMsub, a[1], bb[2], c.vacc[0]),
                            Binary(kMsub, a[2], bb[0], c.vacc[1]),
                            Binary(kMsub, a[0], bb[1], c.vacc[2])};
        if (fd != 0)
            for (int lane = 0; lane < 3; ++lane)
                if (lane_on[lane]) c.vf[fd][lane] = out[lane];
        return;
    }
    if (funct >= 0x30 && funct <= 0x35) {                  /* integer unit */
        const u32 x = c.vc[fs], y = c.vc[ft];
        if (funct == 0x32) {
            const s32 imm = (fd & 0x10) ? (s32)fd - 32 : (s32)fd;
            if (ft > 0 && ft < 16) c.vc[ft] = (u32)(c.vc[fs] + imm) & 0xFFFF;
            return;
        }
        const u32 out = funct == 0x30 ? x + y : funct == 0x31 ? x - y
                      : funct == 0x34 ? (x & y) : (x | y);
        if (fd > 0 && fd < 16) c.vc[fd] = out & 0xFFFF;
        return;
    }
    const u32 fsf = (word >> 21) & 3, ftf = (word >> 23) & 3;
    switch (index) {
        case 0x2E: {                                       /* vopmula */
            const u32* bb = c.vf[ft];
            const u32 out[3] = {Result(Value(a[1]) * Value(bb[2])),
                                Result(Value(a[2]) * Value(bb[0])),
                                Result(Value(a[0]) * Value(bb[1]))};
            for (int lane = 0; lane < 3; ++lane)
                if (lane_on[lane]) c.vacc[lane] = out[lane];
            return;
        }
        case 0x30: { const u32 v[4] = {a[0], a[1], a[2], a[3]}; write(ft, v); return; }
        case 0x31: { const u32 v[4] = {a[1], a[2], a[3], a[0]}; write(ft, v); return; }
        case 0x1D: {
            u32 v[4];
            for (int lane = 0; lane < 4; ++lane) v[lane] = Clean(a[lane]) & ~kSign;
            write(ft, v);
            return;
        }
        case 0x10: case 0x11: case 0x12: case 0x13: {
            static const double kScale[4] = {1.0, 16.0, 4096.0, 32768.0};
            u32 v[4];
            for (int lane = 0; lane < 4; ++lane)
                v[lane] = Result((double)(s32)a[lane] / kScale[index & 3]);
            write(ft, v);
            return;
        }
        case 0x14: case 0x15: case 0x16: case 0x17: {
            static const double kScale[4] = {1.0, 16.0, 4096.0, 32768.0};
            u32 v[4];
            for (int lane = 0; lane < 4; ++lane)
                v[lane] = FloatToInt(Value(a[lane]) * kScale[index & 3]);
            write(ft, v);
            return;
        }
        case 0x2F: case 0x3B: case 0x1F:                   /* vnop, vwaitq, vclipw */
            return;
        case 0x38: {                                       /* vdiv */
            const u32 num = c.vf[fs][fsf], den = c.vf[ft][ftf];
            c.vc[kRegQ] = Value(den) == 0.0 ? ((num ^ den) & kSign) | kMax
                                            : Result(Value(num) / Value(den));
            return;
        }
        case 0x39:                                         /* vsqrt */
            c.vc[kRegQ] = Result(std::sqrt(std::fabs(Value(c.vf[ft][ftf]))));
            return;
        case 0x3A: {                                       /* vrsqrt */
            const u32 num = c.vf[fs][fsf];
            const double root = std::sqrt(std::fabs(Value(c.vf[ft][ftf])));
            c.vc[kRegQ] = root == 0.0 ? (num & kSign) | kMax : Result(Value(num) / root);
            return;
        }
        case 0x3C:                                         /* vmtir */
            if (ft != 0 && ft < 16) c.vc[ft] = c.vf[fs][fsf] & 0xFFFF;
            return;
        case 0x3D: {                                       /* vmfir */
            u32 x = c.vc[fs] & 0xFFFF;
            if (x & 0x8000) x |= 0xFFFF0000u;
            const u32 v[4] = {x, x, x, x};
            write(ft, v);
            return;
        }
        case 0x42:                                         /* vrinit */
            c.vc[kRegR] = kOne | (c.vf[fs][fsf] & 0x7FFFFF);
            return;
        case 0x43:                                         /* vrxor */
            c.vc[kRegR] = kOne | ((c.vc[kRegR] ^ c.vf[fs][fsf]) & 0x7FFFFF);
            return;
        case 0x41: { const u32 v[4] = {c.vc[kRegR], c.vc[kRegR], c.vc[kRegR], c.vc[kRegR]};
                     write(ft, v); return; }
        case 0x40: {                                       /* vrnext */
            if (ft == 0) return;
            const u32 r = c.vc[kRegR];
            const u32 bit = ((r >> 4) & 1) ^ ((r >> 22) & 1);
            c.vc[kRegR] = kOne | (((r << 1) ^ bit) & 0x7FFFFF);
            const u32 v[4] = {c.vc[kRegR], c.vc[kRegR], c.vc[kRegR], c.vc[kRegR]};
            write(ft, v);
            return;
        }
    }
    throw Abort{word, 5};
}

void Cop2Move(Context& c, u32 word, u32 rt) {
    const u32 kind = (word >> 21) & 0x1F, index = (word >> 11) & 0x1F;
    switch (kind) {
        case 0x05:                                         /* qmtc2 */
            if (index) {
                c.vf[index][0] = (u32)c.r[rt];
                c.vf[index][1] = (u32)(c.r[rt] >> 32);
                c.vf[index][2] = (u32)c.rh[rt];
                c.vf[index][3] = (u32)(c.rh[rt] >> 32);
            }
            return;
        case 0x06:                                         /* ctc2 */
            if (index < 16) {
                if (index) c.vc[index] = (u32)c.r[rt] & 0xFFFF;
            } else if (index == (u32)kRegR) {
                c.vc[kRegR] = kOne | ((u32)c.r[rt] & 0x7FFFFF);
            } else {
                c.vc[index] = (u32)c.r[rt];
            }
            return;
        case 0x01:                                         /* qmfc2 */
            if (rt) {
                const u32* v = c.vf[index];
                c.r[rt] = (u64)v[0] | ((u64)v[1] << 32);
                c.rh[rt] = (u64)v[2] | ((u64)v[3] << 32);
            }
            return;
        case 0x02:                                         /* cfc2 */
            /* VPU-STAT (29): no VU program runs here, so both units are idle. */
            if (rt) c.r[rt] = index == 29 ? 0 : Sx32(c.vc[index]);
            return;
    }
    throw Abort{word, 5};
}

}  // namespace lifted
