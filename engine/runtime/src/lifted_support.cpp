/*
 * The lifted code's arithmetic helpers, written to match engine/ps2decomp/emu.py.
 */

#include "ps2rt/lifted.h"

#include <cmath>
#include <cstring>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace lifted {

void (*loop_head_hook)(Context& c, u32 address) = nullptr;

void BudgetSpent(Context& c, u32 address) {
    if (loop_head_hook) loop_head_hook(c, address);
    else throw Abort{address, 2};
}

void Multiply(u64& hi, u64& lo, u64 a, u64 b, bool is_signed, bool accumulate) {
    const u64 product = is_signed
        ? (u64)((s64)(s32)(u32)a * (s64)(s32)(u32)b)
        : (u64)(u32)a * (u64)(u32)b;
    u64 total = product;
    if (accumulate) {
        total += ((hi & 0xFFFFFFFFull) << 32) | (lo & 0xFFFFFFFFull);
    }
    lo = Sx32(total);
    hi = Sx32(total >> 32);
}

/* As the EE divides (emu.divide): a zero divisor gives LO = 1 for a negative
   signed dividend, else -1, and the dividend in HI. The signed case is computed in
   64 bits, where INT_MIN / -1 is defined. */
void Divide(u64& hi, u64& lo, u64 a, u64 b, bool is_signed) {
    if (is_signed) {
        const s64 x = (s32)(u32)a, y = (s32)(u32)b;
        if (y == 0) {
            lo = x < 0 ? 1 : ~0ull;
            hi = (u64)x;
            return;
        }
        const s64 q = x / y;
        lo = Sx32((u64)q);
        hi = Sx32((u64)(x - y * q));
    } else {
        const u32 x = (u32)a, y = (u32)b;
        if (y == 0) {
            lo = ~0ull;
            hi = Sx32(x);
            return;
        }
        lo = Sx32(x / y);
        hi = Sx32(x % y);
    }
}

u32 FloatMulAdd(u32 acc, u32 a, u32 b, bool subtract) {
    // Operands as the FPU reads them (a denormal is zero), the result as every
    // FPU result is narrowed (eefpu.cpp).
    const auto operand = [](u32 bits) { return double(generated::FloatValue(generated::FloatMove(bits))); };
    const double x = operand(acc);
    const double product = operand(a) * operand(b);
    return generated::FloatNarrow(subtract ? x - product : x + product);
}

void AddBytes(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi, bool subtract) {
    u64 out[2] = {0, 0};
    const u64 a[2] = {alo, ahi}, b[2] = {blo, bhi};
    for (int half = 0; half < 2; ++half) {
        for (int lane = 0; lane < 8; ++lane) {
            const u32 x = (u32)(a[half] >> (lane * 8)) & 0xFF;
            const u32 y = (u32)(b[half] >> (lane * 8)) & 0xFF;
            const u32 z = (subtract ? x - y : x + y) & 0xFF;
            out[half] |= (u64)z << (lane * 8);
        }
    }
    lo = out[0];
    hi = out[1];
}

namespace {

u32 Lane32(int op, u32 a, u32 b) {
    switch (op) {
        case kPmaxw: return (s32)a >= (s32)b ? a : b;
        case kPminw: return (s32)a <= (s32)b ? a : b;
        case kPsubw: return a - b;
        case kPaddw: return a + b;
        case kPadduw: { const u64 sum = (u64)a + b; return sum > 0xFFFFFFFFull ? 0xFFFFFFFFu : (u32)sum; }
        case kPabsw: { const s64 x = (s32)b; const s64 m = x < 0 ? -x : x;
                       return m > 0x7FFFFFFF ? 0x7FFFFFFFu : (u32)m; }
    }
    return 0;
}

/* Comparisons and halfword max/min, lane by lane at `width` bits. */
u64 NarrowLanes(int op, u64 a, u64 b, u32 width) {
    const u64 mask = width == 64 ? ~0ull : (1ull << width) - 1, sign = 1ull << (width - 1);
    u64 out = 0;
    for (u32 shift = 0; shift < 64; shift += width) {
        const u64 x = (a >> shift) & mask, y = (b >> shift) & mask;
        const s64 sx = (s64)(x ^ sign) - (s64)sign, sy = (s64)(y ^ sign) - (s64)sign;
        u64 lane = 0;
        switch (op) {
            case kPcgtb: case kPcgth: case kPcgtw: lane = sx > sy ? mask : 0; break;
            case kPceqb: case kPceqh: case kPceqw: lane = x == y ? mask : 0; break;
            case kPmaxh: lane = sx >= sy ? x : y; break;
            case kPminh: lane = sx <= sy ? x : y; break;
        }
        out |= lane << shift;
    }
    return out;
}

u64 LanesOf(int op, u64 a, u64 b) {
    u64 out = 0;
    switch (op) {
        case kPcgtb: case kPceqb: return NarrowLanes(op, a, b, 8);
        case kPcgth: case kPceqh: case kPmaxh: case kPminh: return NarrowLanes(op, a, b, 16);
        case kPcgtw: case kPceqw: return NarrowLanes(op, a, b, 32);
        default: break;
    }
    if (op == kPaddh) {
        for (int shift = 0; shift < 64; shift += 16)
            out |= ((((a >> shift) & 0xFFFF) + ((b >> shift) & 0xFFFF)) & 0xFFFF) << shift;
        return out;
    }
    if (op == kPaddub) {
        for (int shift = 0; shift < 64; shift += 8) {
            const u32 sum = (u32)((a >> shift) & 0xFF) + (u32)((b >> shift) & 0xFF);
            out |= (u64)(sum > 0xFF ? 0xFF : sum) << shift;
        }
        return out;
    }
    for (int shift = 0; shift < 64; shift += 32) {
        out |= (u64)Lane32(op, (u32)(a >> shift), (u32)(b >> shift)) << shift;
    }
    return out;
}

}  // namespace

#if defined(__SSE2__) || defined(_M_X64)
/* The same lanes in SSE2 for the operations it has exactly (signed compares,
   halfword max and min, wrapping and unsigned-saturating adds); `done` false
   for the rest. LanesAreScalar (host tests) compares the two. */
bool LanesSse2(int op, u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi) {
    const __m128i a = _mm_set_epi64x(s64(ahi), s64(alo)), b = _mm_set_epi64x(s64(bhi), s64(blo));
    __m128i r;
    switch (op) {
        case kPcgtb: r = _mm_cmpgt_epi8(a, b); break;
        case kPcgth: r = _mm_cmpgt_epi16(a, b); break;
        case kPcgtw: r = _mm_cmpgt_epi32(a, b); break;
        case kPceqb: r = _mm_cmpeq_epi8(a, b); break;
        case kPceqh: r = _mm_cmpeq_epi16(a, b); break;
        case kPceqw: r = _mm_cmpeq_epi32(a, b); break;
        case kPmaxh: r = _mm_max_epi16(a, b); break;
        case kPminh: r = _mm_min_epi16(a, b); break;
        case kPaddh: r = _mm_add_epi16(a, b); break;
        case kPaddub: r = _mm_adds_epu8(a, b); break;
        case kPaddw: r = _mm_add_epi32(a, b); break;
        case kPsubw: r = _mm_sub_epi32(a, b); break;
        default: return false;
    }
    lo = u64(_mm_cvtsi128_si64(r));
    hi = u64(_mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)));
    return true;
}
#endif

void LanesScalar(int op, u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi) {
    lo = LanesOf(op, alo, blo);
    hi = LanesOf(op, ahi, bhi);
}

void Lanes(int op, u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi) {
#if defined(__SSE2__) || defined(_M_X64)
    if (LanesSse2(op, lo, hi, alo, ahi, blo, bhi)) return;
#endif
    LanesScalar(op, lo, hi, alo, ahi, blo, bhi);
}

void ShiftHalfwordsRight(u64& lo, u64& hi, u64 blo, u64 bhi, u32 amount) {
    const u64 in[2] = {blo, bhi};
    u64 out[2] = {0, 0};
    for (int half = 0; half < 2; ++half)
        for (int shift = 0; shift < 64; shift += 16)
            out[half] |= (((in[half] >> shift) & 0xFFFF) >> (amount & 0xF)) << shift;
    lo = out[0];
    hi = out[1];
}

void ShiftLanes(u64& lo, u64& hi, u64 blo, u64 bhi, u32 width, u32 kind, u32 amount) {
    const u64 mask = (1ull << width) - 1;
    amount &= width - 1;
    const u64 in[2] = {blo, bhi};
    u64 out[2] = {0, 0};
    for (int half = 0; half < 2; ++half) {
        for (u32 shift = 0; shift < 64; shift += width) {
            u64 lane = (in[half] >> shift) & mask;
            if (kind == 0) {
                lane = (lane << amount) & mask;
            } else if (kind == 1) {
                lane >>= amount;
            } else {
                const u64 sign = 1ull << (width - 1);
                const s64 value = (s64)(lane ^ sign) - (s64)sign;   /* sign-extend the lane */
                lane = (u64)(value >> amount) & mask;
            }
            out[half] |= lane << shift;
        }
    }
    lo = out[0];
    hi = out[1];
}

void InterleaveScalar(u64& lo, u64& hi, u64 a, u64 b, u32 width) {
    const u64 mask = (1ull << width) - 1;
    u64 out[2] = {0, 0};
    for (u32 i = 0; i < 64 / width; ++i) {
        const u32 even = 2 * i * width, odd = (2 * i + 1) * width;
        out[even / 64] |= ((b >> (i * width)) & mask) << (even % 64);
        out[odd / 64] |= ((a >> (i * width)) & mask) << (odd % 64);
    }
    lo = out[0];
    hi = out[1];
}

/* PEXTLB, PEXTLH, PEXTLW: SSE2's unpack does the same (b's lanes even). */
void Interleave(u64& lo, u64& hi, u64 a, u64 b, u32 width) {
#if defined(__SSE2__) || defined(_M_X64)
    if (width == 8 || width == 16 || width == 32) {
        const __m128i x = _mm_cvtsi64_si128(s64(b)), y = _mm_cvtsi64_si128(s64(a));
        const __m128i r = width == 8 ? _mm_unpacklo_epi8(x, y) : width == 16 ? _mm_unpacklo_epi16(x, y)
                                                                             : _mm_unpacklo_epi32(x, y);
        lo = u64(_mm_cvtsi128_si64(r));
        hi = u64(_mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)));
        return;
    }
#endif
    InterleaveScalar(lo, hi, a, b, width);
}

void PackHalfwords(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi) {
    u64 low = 0, high = 0;
    for (int i = 0; i < 2; ++i) {
        low |= ((blo >> (32 * i)) & 0xFFFF) << (16 * i);
        low |= ((bhi >> (32 * i)) & 0xFFFF) << (16 * (i + 2));
        high |= ((alo >> (32 * i)) & 0xFFFF) << (16 * i);
        high |= ((ahi >> (32 * i)) & 0xFFFF) << (16 * (i + 2));
    }
    lo = low;
    hi = high;
}

void PackBytes(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi) {
    u64 low = 0, high = 0;
    for (int i = 0; i < 4; ++i) {
        low |= ((blo >> (16 * i)) & 0xFF) << (8 * i);
        low |= ((bhi >> (16 * i)) & 0xFF) << (8 * (i + 4));
        high |= ((alo >> (16 * i)) & 0xFF) << (8 * i);
        high |= ((ahi >> (16 * i)) & 0xFF) << (8 * (i + 4));
    }
    lo = low;
    hi = high;
}

void FunnelShift(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi, u32 sa) {
    /* The 256-bit value ahi:alo:bhi:blo, shifted right by sa bits; the low 128. */
    const u64 words[4] = {blo, bhi, alo, ahi};
    sa &= 0xFF;
    const u32 skip = sa / 64, bits = sa % 64;
    u64 out[2];
    for (u32 i = 0; i < 2; ++i) {
        const u64 low = skip + i < 4 ? words[skip + i] : 0;
        const u64 high = skip + i + 1 < 4 ? words[skip + i + 1] : 0;
        out[i] = bits ? (low >> bits) | (high << (64 - bits)) : low;
    }
    lo = out[0];
    hi = out[1];
}

u32 FloatRsqrt(u32 fs, u32 ft) {
    const double root = std::sqrt(std::fabs((double)generated::FloatValue(ft)));
    if (root == 0.0) return (fs & 0x80000000u) | 0x7F7FFFFFu;
    double value = (double)generated::FloatValue(fs) / root;
    const double kFloatMax = 3.4028234663852886e38;
    if (value != value) value = 0.0;
    else if (value > kFloatMax) value = kFloatMax;
    else if (value < -kFloatMax) value = -kFloatMax;
    const float narrowed = static_cast<float>(value);
    u32 bits;
    std::memcpy(&bits, &narrowed, sizeof bits);
    return bits;
}

u64 LoadWordLeft(u32 address, u64 old) {
    const u32 keep = (3 - (address & 3)) * 8;
    const u32 word = (u32)generated::ReadU32(address & ~3u);
    const u32 low = keep ? (u32)old & ((1u << keep) - 1) : 0;
    return Sx32((keep ? word << keep : word) | low);
}

u64 LoadWordRight(u32 address, u64 old) {
    const u32 drop = (address & 3) * 8;
    const u32 word = (u32)generated::ReadU32(address & ~3u);
    const u32 merged = (word >> drop) | ((u32)old & ~(0xFFFFFFFFu >> drop));
    return drop == 0 ? Sx32(merged) : (old & 0xFFFFFFFF00000000ull) | merged;
}

u64 LoadDoubleLeft(u32 address, u64 old) {
    const u32 keep = (7 - (address & 7)) * 8;
    const u64 value = generated::ReadU64(address & ~7u);
    return keep ? (value << keep) | (old & ((1ull << keep) - 1)) : value;
}

u64 LoadDoubleRight(u32 address, u64 old) {
    const u32 drop = (address & 7) * 8;
    const u64 value = generated::ReadU64(address & ~7u);
    return (value >> drop) | (old & ~(~0ull >> drop));
}

void StoreLeft(u32 address, u64 value, u32 size) {
    const u32 offset = address & (size - 1), count = offset + 1;
    const u64 data = (size == 8 ? value : value & 0xFFFFFFFFull) >> (8 * (size - count));
    for (u32 i = 0; i < count; ++i)
        generated::Write8(address - offset + i, (u32)(data >> (8 * i)) & 0xFF);
}

void StoreRight(u32 address, u64 value, u32 size) {
    const u32 count = size - (address & (size - 1));
    for (u32 i = 0; i < count; ++i)
        generated::Write8(address + i, (u32)(value >> (8 * i)) & 0xFF);
}

}  // namespace lifted
