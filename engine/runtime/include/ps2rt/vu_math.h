/*
 * The VU's arithmetic model (engine/ps2decomp/vu0.py documents it, and where it
 * departs from the hardware): clean operands, results narrowed to single
 * precision and saturated. Inline, so that the macro-mode COP2 (vu0.cpp) and
 * the micro-mode interpreter (engine/host/src/vu.cpp) compute each lane
 * without a call per step.
 */

#ifndef PS2RT_VU_MATH_H
#define PS2RT_VU_MATH_H

#include <cstring>

#include "ps2rt/types.h"

/* The model's steps, inlined into the interpreter and compiled microprograms. */
#if defined(_MSC_VER)
#define PS2RT_VU_INLINE __forceinline
#else
#define PS2RT_VU_INLINE inline __attribute__((always_inline))
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define PS2RT_VU_SSE 1
#else
#define PS2RT_VU_SSE 0
#endif
/* SSE4.1 (pminud): every x86-64 processor a Vulkan port runs on has it; MSVC
   takes its intrinsics without /arch, and other compilers say so with
   __SSE4_1__. */
#if PS2RT_VU_SSE && (defined(__SSE4_1__) || defined(_M_X64))
#include <smmintrin.h>
#define PS2RT_VU_SSE41 1
#else
#define PS2RT_VU_SSE41 0
#endif



namespace lifted {

inline constexpr u32 kSign = 0x80000000u;
inline constexpr u32 kMax = 0x7F7FFFFFu;
inline constexpr u32 kOne = 0x3F800000u;
inline constexpr int kRegR = 20, kRegI = 21, kRegQ = 22;

PS2RT_VU_INLINE u32 Clean(u32 bits) {
    const u32 exponent = bits & 0x7F800000u;
    if (exponent == 0) return bits & kSign;
    if (exponent == 0x7F800000u) return (bits & kSign) | kMax;
    return bits;
}

PS2RT_VU_INLINE double Value(u32 bits) {
    const u32 cleaned = Clean(bits);
    float value;
    std::memcpy(&value, &cleaned, sizeof value);
    return value;
}

PS2RT_VU_INLINE u32 Result(double number) {
    if (number != number) return kMax;
    const float narrowed = static_cast<float>(number);
    u32 bits;
    std::memcpy(&bits, &narrowed, sizeof bits);
    if ((bits & 0x7F800000u) == 0x7F800000u) return (bits & kSign) | kMax;
    if ((bits & 0x7F800000u) == 0) return bits & kSign;
    return bits;
}

enum Op { kAdd, kSub, kMul, kMadd, kMsub, kMax_, kMini };

/* One operation, chosen at compile time: RunFmac picks it once an
   instruction rather than once a lane. */
/* MAX's and MINI's key: the bits as a sign-and-magnitude integer. They
   compare so and hand back an operand as it is - no Clean: a denormal stays
   (programs build integer constants with LOI and MAXI). */
PS2RT_VU_INLINE s32 Order(u32 bits) {
    const s32 magnitude = s32(bits & 0x7FFFFFFFu);
    return (bits & kSign) ? -magnitude : magnitude;
}

template <int Op>
inline u32 BinaryOf(u32 a, u32 b, u32 acc) {
    if constexpr (Op == kMax_) return Order(a) < Order(b) ? b : a;
    else if constexpr (Op == kMini) return Order(b) < Order(a) ? b : a;
    const double x = Value(a), y = Value(b);
    if constexpr (Op == kAdd) return Result(x + y);
    else if constexpr (Op == kSub) return Result(x - y);
    else if constexpr (Op == kMul) return Result(x * y);
    else if constexpr (Op == kMadd) return Result(Value(acc) + Value(Result(x * y)));
    else return Result(Value(acc) - Value(Result(x * y)));
}

inline u32 Binary(int op, u32 a, u32 b, u32 acc) {
    switch (op) {
        case kAdd: return BinaryOf<kAdd>(a, b, acc);
        case kSub: return BinaryOf<kSub>(a, b, acc);
        case kMul: return BinaryOf<kMul>(a, b, acc);
        case kMadd: return BinaryOf<kMadd>(a, b, acc);
        case kMsub: return BinaryOf<kMsub>(a, b, acc);
        case kMax_: return BinaryOf<kMax_>(a, b, acc);
        case kMini: return BinaryOf<kMini>(a, b, acc);
    }
    return 0;
}

#if PS2RT_VU_SSE
/*
 * All four lanes at once in single precision. The model computes in double
 * and narrows (BinaryOf); that is the same number: the operands are clean -
 * no infinities, NaNs or denormals - and double's 53 bits are at least
 * 2 x 24 + 2, so rounding an add, a subtract or a multiply first to double
 * and then to single gives the single-precision result rounded once (and an
 * overflow the infinity Result saturates). Result is Clean's transform of
 * the bits. RunFmacScalar is the model; VuLanesAreTheModel (host tests)
 * compares the two.
 */
PS2RT_VU_INLINE __m128i CleanLanes(__m128i v) {
#if PS2RT_VU_SSE41
    // The magnitude clamped to the largest finite (an exponent of 255 is
    // above it, whatever the mantissa), zero below the smallest normal, and
    // the sign put back: Clean's three cases in six operations.
    const __m128i magnitude = _mm_and_si128(v, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i sign = _mm_xor_si128(v, magnitude);
    const __m128i clamped = _mm_min_epu32(magnitude, _mm_set1_epi32(int(kMax)));
    const __m128i denormal = _mm_cmpgt_epi32(_mm_set1_epi32(0x00800000), magnitude);
    return _mm_or_si128(sign, _mm_andnot_si128(denormal, clamped));
#else
    const __m128i exponent = _mm_and_si128(v, _mm_set1_epi32(0x7F800000));
    const __m128i sign = _mm_and_si128(v, _mm_set1_epi32(int(kSign)));
    const __m128i zero = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
    const __m128i huge = _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7F800000));
    const __m128i saturated = _mm_or_si128(sign, _mm_set1_epi32(int(kMax)));
    v = _mm_or_si128(_mm_andnot_si128(zero, v), _mm_and_si128(zero, sign));
    return _mm_or_si128(_mm_andnot_si128(huge, v), _mm_and_si128(huge, saturated));
#endif
}

inline __m128 AsFloats(__m128i v) { return _mm_castsi128_ps(v); }
inline __m128i AsBits(__m128 v) { return _mm_castps_si128(v); }

/* `Known` says which operands the caller knows to be clean already (bit 0
   a, bit 1 b, bit 2 acc) in the lanes it keeps: Clean would leave them as
   they are. Compiled VU code knows it of fields an FMAC wrote earlier in
   the same block (vurecomp.py, `clean`). */
PS2RT_VU_INLINE __m128i OrderLanes(__m128i v) {
    const __m128i negative = _mm_srai_epi32(v, 31);
    const __m128i magnitude = _mm_and_si128(v, _mm_set1_epi32(0x7FFFFFFF));
    return _mm_sub_epi32(_mm_xor_si128(magnitude, negative), negative);
}

template <int Op, int Known = 0>
PS2RT_VU_INLINE __m128i LanesSse(__m128i a, __m128i b, __m128i acc) {
    if constexpr (Op == kMax_ || Op == kMini) {
        // MAX takes b where a < b, MINI where b < a; otherwise a - by Order,
        // the operands as they are.
        const __m128i take_b = Op == kMax_ ? _mm_cmplt_epi32(OrderLanes(a), OrderLanes(b))
                                           : _mm_cmplt_epi32(OrderLanes(b), OrderLanes(a));
        return _mm_or_si128(_mm_andnot_si128(take_b, a), _mm_and_si128(take_b, b));
    }
    if constexpr ((Known & 1) == 0) a = CleanLanes(a);
    if constexpr ((Known & 2) == 0) b = CleanLanes(b);
    if constexpr (Op == kAdd) return CleanLanes(AsBits(_mm_add_ps(AsFloats(a), AsFloats(b))));
    else if constexpr (Op == kSub) return CleanLanes(AsBits(_mm_sub_ps(AsFloats(a), AsFloats(b))));
    else if constexpr (Op == kMul) return CleanLanes(AsBits(_mm_mul_ps(AsFloats(a), AsFloats(b))));
    else if constexpr (Op == kMadd || Op == kMsub) {
        const __m128i product = CleanLanes(AsBits(_mm_mul_ps(AsFloats(a), AsFloats(b))));
        const __m128 total = AsFloats((Known & 4) != 0 ? acc : CleanLanes(acc));
        return CleanLanes(AsBits(Op == kMadd ? _mm_add_ps(total, AsFloats(product))
                                             : _mm_sub_ps(total, AsFloats(product))));
    }
}
#endif

inline u32 FloatToInt(double number) {
    if (number >= 2147483647.0) return 0x7FFFFFFFu;
    if (number <= -2147483648.0) return 0x80000000u;
    return (u32)(s32)number;
}

}  // namespace lifted

#endif  /* PS2RT_VU_MATH_H */
