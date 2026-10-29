// ps2fpu.cpp - bit-exact model of the PS2 EE FPU arithmetic.
// Written from a behavioural specification (engine/docs/PS2FPU.md) only, by
// someone who had not seen any emulator's code, and checked bit for bit
// against test vectors.
#include "ps2rt/ps2fpu.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ps2fpu {
namespace {

constexpr uint32_t kSignBit = 0x80000000u;
constexpr uint32_t kMaxMag = 0x7FFFFFFFu;
constexpr uint32_t kHidden = 0x00800000u;
constexpr uint32_t kFracMask = 0x007FFFFFu;

inline uint32_t SignOf(uint32_t x) { return x & kSignBit; }
inline int ExpOf(uint32_t x) { return int((x >> 23) & 0xFF); }
// 24-bit significand with the hidden bit set (only meaningful if exp != 0).
inline uint32_t SigOf(uint32_t x) { return (x & kFracMask) | kHidden; }

// Index of the most significant set bit of a nonzero 64-bit value.
inline int TopBit(uint64_t v) {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx;
    _BitScanReverse64(&idx, v);
    return int(idx);
#else
    int n = 0;
    while (v >>= 1) ++n;
    return n;
#endif
}

// Assemble a result: saturate above exponent 255, flush to signed zero below 1.
// `frac` is only the 23 explicit fraction bits.
inline uint32_t Pack(uint32_t sign, int exp, uint32_t frac) {
    if (exp > 255) return sign | kMaxMag;
    if (exp < 1) return sign;
    return sign | (uint32_t(exp) << 23) | (frac & kFracMask);
}

// 3:2 carry-save compressor on 32-bit words (top carry is lost).
struct CarrySave {
    uint32_t sum;
    uint32_t carry;
};
inline CarrySave Csa(uint32_t x, uint32_t y, uint32_t z) {
    CarrySave r;
    r.sum = x ^ y ^ z;
    r.carry = ((x & y) | ((x ^ y) & z)) << 1;
    return r;
}

// ---------------------------------------------------------------------------
// Multiplier: low 32 bits of the hardware's (incomplete) Booth partial-product
// sum. Only bit 15 of the returned word is used by the caller.
uint32_t BoothLowWord(uint32_t a, uint32_t b) {
    uint32_t pp[8];
    uint32_t neg[8];
    for (int d = 0; d < 8; ++d) {
        // Window of three multiplier bits (2d+1, 2d, 2d-1); bit -1 is 0.
        uint32_t window = (d == 0) ? ((b << 1) & 7u) : ((b >> (2 * d - 1)) & 7u);
        uint32_t v = a << (2 * d);
        uint32_t n = 0;
        if (window == 3 || window == 4) v <<= 1;               // 2x multiplicand
        if (window == 4 || window == 5 || window == 6) {        // negative digit
            v ^= ~0u << (2 * d);                                // ones' complement from 2^(2d) up
            n = 1u << (2 * d);                                  // the pending +1
        }
        if (window == 0 || window == 7) v = 0;                  // zero digit
        pp[d] = v;
        neg[d] = n;
    }

    CarrySave s0 = Csa(pp[1], pp[2], pp[3]);
    CarrySave s1 = Csa(pp[4] & ~0x7FFu, pp[5] & ~0xFFFu, pp[6]);
    s1.carry |= neg[6] | (pp[5] & 0x800u);
    uint32_t pp7 = pp[7] | ((pp[5] & 0x400u) + neg[5]);
    CarrySave s2 = Csa(pp[0], s0.sum, s0.carry);
    CarrySave s3 = Csa(pp7, s1.sum, s1.carry);
    CarrySave s4 = Csa(s2.carry, s3.sum, s3.carry);
    CarrySave s5 = Csa(s2.sum, s4.sum, s4.carry);
    s5.carry += neg[7];
    return (s5.sum & ~0x7FFFu) + (s5.carry & ~0x7FFFu);
}

// ---------------------------------------------------------------------------
// SRT (radix 2, digits -1/0/+1) machinery shared by divide and square root.

// Digit chosen from a carry-save remainder: the low 24 bits of the sum word
// are OR'd (not added) into the upper-part sum.
inline int SelectDigit(uint32_t s, uint32_t c) {
    int32_t t = int32_t(((s & ~0xFFFFFFu) + c) | (s & 0xFFFFFFu));
    if (t >= (1 << 23)) return 1;
    if (t < -(1 << 24)) return -1;
    return 0;
}

// One remainder step: subtract (digit +1), add (digit -1) or keep (digit 0)
// `operand`, pick the next digit, then shift the remainder left by one.
inline void SrtStep(uint32_t& s, uint32_t& c, int& digit, uint32_t operand) {
    uint32_t addend = 0;
    if (digit > 0) {
        addend = ~operand;
        c += 1;               // completes the two's complement of the operand
    } else if (digit < 0) {
        addend = operand;
    }
    CarrySave next = Csa(s, c, addend);
    int nextDigit = (digit != 0) ? SelectDigit(next.sum, next.carry)
                                 : SelectDigit(s, c);
    s = next.sum << 1;
    c = next.carry << 1;
    digit = nextDigit;
}

} // namespace

// ---------------------------------------------------------------------------
uint32_t Add(uint32_t a, uint32_t b) {
    int ea = ExpOf(a), eb = ExpOf(b);
    if (ea == 0 || eb == 0) {
        if (ea != 0) return a;
        if (eb != 0) return b;
        return (SignOf(a) & SignOf(b));   // -0 only if both are negative zeros
    }

    // x: operand with the larger exponent (the first one on a tie).
    uint32_t x = a, y = b;
    if (eb > ea) { x = b; y = a; }
    int ex = ExpOf(x), ey = ExpOf(y);
    int d = ex - ey;
    if (d >= 25) return x;

    uint32_t fy = y & kFracMask;
    if (d >= 1) fy &= ~((1u << (d - 1)) - 1u);   // drop all but one guard bit's worth
    int64_t mx = int64_t(SigOf(x)) << 1;
    int64_t my = int64_t((fy | kHidden) << 1) >> d;   // exact after the clearing
    if (SignOf(x)) mx = -mx;
    if (SignOf(y)) my = -my;

    int64_t sum = mx + my;
    if (sum == 0) return 0;
    uint32_t sign = sum < 0 ? kSignBit : 0;
    uint64_t mag = uint64_t(sum < 0 ? -sum : sum);

    int top = TopBit(mag);
    uint64_t sig = (top >= 23) ? (mag >> (top - 23)) : (mag << (23 - top));
    return Pack(sign, ex + top - 24, uint32_t(sig));
}

// ---------------------------------------------------------------------------
uint32_t Mul(uint32_t a, uint32_t b) {
    uint32_t sign = SignOf(a) ^ SignOf(b);
    int ea = ExpOf(a), eb = ExpOf(b);
    if (ea == 0 || eb == 0) return sign;

    uint32_t ma = SigOf(a), mb = SigOf(b);
    uint64_t p = uint64_t(ma) * mb;
    // Bit 15 of the product comes from the hardware's truncated partial sum.
    // It changes the bits kept (23 up) only through a borrow, when bits 15-22
    // are all zero: otherwise the true product will do.
    if ((p & 0x7F8000u) == 0) {
        uint32_t low = BoothLowWord(ma, mb);
        p -= uint64_t((low ^ uint32_t(p)) & 0x8000u);
    }

    uint64_t r = p >> 23;
    int exp = ea + eb - 127;
    if (r >= (1ull << 24)) { r >>= 1; ++exp; }
    return Pack(sign, exp, uint32_t(r));
}

// ---------------------------------------------------------------------------
uint32_t Div(uint32_t a, uint32_t b) {
    uint32_t sign = SignOf(a) ^ SignOf(b);
    int ea = ExpOf(a), eb = ExpOf(b);
    if (eb == 0) return sign | kMaxMag;
    if (ea == 0) return sign;

    uint32_t s = SigOf(a) << 2, c = 0;
    const uint32_t divisor = SigOf(b) << 2;
    uint32_t q = 0;
    int digit = 1;
    for (int i = 0; i < 25; ++i) {
        q = q * 2u + uint32_t(int32_t(digit));
        SrtStep(s, c, digit, divisor);
    }

    int exp = ea - eb + 126;
    if (q >= (1u << 24)) { q >>= 1; ++exp; }
    return Pack(sign, exp, q);
}

// ---------------------------------------------------------------------------
uint32_t Sqrt(uint32_t a) {
    int ea = ExpOf(a);
    if (ea == 0) return 0;

    uint32_t s = SigOf(a) << 1, c = 0;
    if ((a & 0x00800000u) == 0) s <<= 1;   // odd unbiased exponent
    uint32_t r = 0;
    int digit = 1;
    for (int i = 0; i < 25; ++i) {
        uint32_t sd = uint32_t(int32_t(digit));
        uint32_t subtrahend = r + sd * (1u << (24 - i));
        r += sd * (1u << (25 - i));
        SrtStep(s, c, digit, subtrahend);
    }

    return (uint32_t((ea + 127) >> 1) << 23) | ((r >> 2) & kFracMask);
}

} // namespace ps2fpu
