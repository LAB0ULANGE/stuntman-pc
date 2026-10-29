/*
 * The Emotion Engine's single-precision FPU, as the derived code relies on it.
 *
 * The derived functions in src/generated/ do float arithmetic through these
 * helpers rather than with `float` operators, because the EE's FPU is not
 * IEEE-754 and a port that uses the host's arithmetic directly will drift:
 *
 *  - There are no infinities or NaNs. A result too large to represent saturates
 *    to the largest finite magnitude (bit pattern 0x7F7FFFFF), and a NaN - which
 *    can only enter as a bit pattern loaded from memory - becomes zero.
 *  - Division by zero does not trap and yields zero.
 *
 * These definitions are the reference: they reproduce the R5900 interpreter in
 * tools/ps2decomp/emu.py exactly, and the compiled-code check (tools/ps2decomp/cppcheck.py)
 * runs every derived function against that interpreter through them. Each
 * operation is computed in double precision and then rounded once to single,
 * which gives the correctly rounded single-precision result for +, -, *, / and
 * square root (double has more than twice the precision of float, so the double
 * rounding is innocuous).
 *
 *  - Results are rounded toward zero, not to nearest, and denormal operands and
 *    results are flushed to zero - the EE's FPU, as PCSX2 runs it by default
 *    ("Chop/Zero"). The VUs round so too (ps2host's Vu::Run): Burnout 3 leans on
 *    it there. PS2RT_EE_ROUND_NEAREST=1 rounds to nearest and keeps denormals,
 *    as tools/ps2decomp/emu.py's interpreter does (what cppcheck compares
 *    against).
 */

#include "ps2rt/generated.h"
#include "ps2rt/ps2fpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace generated {

namespace {

const double kFloatMax = 3.4028234663852886e38;   // bit pattern 0x7F7FFFFF

const bool kNearest = std::getenv("PS2RT_EE_ROUND_NEAREST") != nullptr;

float FromBits(u32 bits) {
    if (!kNearest && (bits & 0x7F800000u) == 0) bits &= 0x80000000u;     // a denormal operand is zero
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

u32 ToBits(double value) { return FloatNarrow(value); }

/*
 * float_model=ps2 (SetFloatModel): the FPU's arithmetic bit for bit, as the
 * hardware's datapaths compute it - not the exact result rounded toward zero
 * (ps2fpu.cpp, from the description in docs/PS2FPU.md: a one-guard-bit adder,
 * a Booth multiplier whose low partial-product bits are dropped, SRT divider
 * and square root with no correction step). Stuntman's chase cars replay
 * recorded controller inputs through the game's physics and drift off the
 * road without it. Every exponent is a number here (255 too: no infinity, no
 * NaN); the largest magnitude is 0x7FFFFFFF; zero and denormals read as zero.
 */
bool g_ps2_float = std::getenv("PS2RT_EE_FLOAT") && std::strcmp(std::getenv("PS2RT_EE_FLOAT"), "ps2") == 0;

constexpr u32 kSign = 0x80000000u;

u32 Ps2Add(u32 a, u32 b) { return ps2fpu::Add(a, b); }
u32 Ps2Mul(u32 a, u32 b) { return ps2fpu::Mul(a, b); }
u32 Ps2Div(u32 a, u32 b) { return ps2fpu::Div(a, b); }
u32 Ps2Sqrt(u32 a) { return ps2fpu::Sqrt(a); }

/* For comparisons: the value as the FPU orders it (exponent 255 is the largest). */
float Ps2Value(u32 bits) {
    if ((bits & 0x7F800000u) == 0) bits &= kSign;
    else if ((bits & 0x7F800000u) == 0x7F800000u) bits = (bits & kSign) | 0x7F7FFFFFu;
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

}  // namespace

void SetFloatModel(bool ps2) { g_ps2_float = ps2; }
bool Ps2FloatModel() { return g_ps2_float; }
u32 Ps2FloatAdd(u32 a, u32 b) { return Ps2Add(a, b); }
u32 Ps2FloatMul(u32 a, u32 b) { return Ps2Mul(a, b); }
u32 Ps2FloatDiv(u32 a, u32 b) { return Ps2Div(a, b); }
u32 Ps2FloatSqrt(u32 a) { return Ps2Sqrt(a); }

u32 FloatNarrow(double value) {
    if (value != value) {
        value = 0.0;
    } else if (value > kFloatMax) {
        value = kFloatMax;
    } else if (value < -kFloatMax) {
        value = -kFloatMax;
    }
    float narrowed = static_cast<float>(value);
    if (!kNearest) {
        // Toward zero: a value rounded away from it steps back one.
        if (std::fabs(double(narrowed)) > std::fabs(value)) narrowed = std::nextafter(narrowed, 0.0f);
        if (std::fabs(narrowed) < FLT_MIN) narrowed = std::copysign(0.0f, narrowed);
    }
    u32 bits;
    std::memcpy(&bits, &narrowed, sizeof bits);
    return bits;
}

u32 FloatAdd(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Add(a, b);
    return ToBits(double(FromBits(a)) + double(FromBits(b)));
}
u32 FloatSub(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Add(a, b ^ kSign);
    return ToBits(double(FromBits(a)) - double(FromBits(b)));
}
u32 FloatMul(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Mul(a, b);
    return ToBits(double(FromBits(a)) * double(FromBits(b)));
}

u32 FloatMulAdd(u32 acc, u32 a, u32 b) {
    if (g_ps2_float) return Ps2Add(acc, Ps2Mul(a, b));
    return ToBits(double(FromBits(acc)) + double(FromBits(a)) * double(FromBits(b)));
}

u32 FloatMulSub(u32 acc, u32 a, u32 b) {
    if (g_ps2_float) return Ps2Add(acc, Ps2Mul(a, b) ^ kSign);
    return ToBits(double(FromBits(acc)) - double(FromBits(a)) * double(FromBits(b)));
}

u32 FloatDiv(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Div(a, b);
    const double divisor = FromBits(b);
    if (divisor == 0.0) {
        return ToBits(0.0);
    }
    return ToBits(double(FromBits(a)) / divisor);
}

/* Written to match how the interpreter picks between equal or unordered operands:
   the first operand wins unless the second is strictly smaller (larger). */
u32 FloatMin(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Value(b) < Ps2Value(a) ? b : a;
    const double x = FromBits(a), y = FromBits(b);
    return ToBits(y < x ? y : x);
}

u32 FloatMax(u32 a, u32 b) {
    if (g_ps2_float) return Ps2Value(b) > Ps2Value(a) ? b : a;
    const double x = FromBits(a), y = FromBits(b);
    return ToBits(y > x ? y : x);
}

u32 FloatNeg(u32 a) { return g_ps2_float ? a ^ kSign : ToBits(-double(FromBits(a))); }
u32 FloatAbs(u32 a) { return g_ps2_float ? a & ~kSign : ToBits(std::fabs(double(FromBits(a)))); }

/* The square root of a negative number - and of -0.0 - is zero here. */
u32 FloatSqrt(u32 a) {
    if (g_ps2_float) return Ps2Sqrt(a);
    const double x = FromBits(a);
    return ToBits(x > 0.0 ? std::sqrt(x) : 0.0);
}

/* `mov.s` as the interpreter runs it: the value passes through the FPU, so a NaN
   pattern comes out as zero rather than being copied bit for bit. */
u32 FloatMove(u32 a) { return g_ps2_float ? a : ToBits(FromBits(a)); }

u32 FloatFromInt(u32 bits_as_int) {
    return ToBits(static_cast<double>(static_cast<s32>(bits_as_int)));
}

/* Truncates toward zero below 2**31 in magnitude (an exponent up to 2**30's)
   and saturates beyond, by the sign alone: the EE's cvt.w.s (emu.py's
   float_to_word). */
u32 IntFromFloat(u32 float_bits) {
    if ((float_bits & 0x7F800000u) <= 0x4E800000u)
        return static_cast<u32>(static_cast<s32>(FromBits(float_bits)));
    return (float_bits & 0x80000000u) ? 0x80000000u : 0x7FFFFFFFu;
}

float FloatValue(u32 bits) { return g_ps2_float ? Ps2Value(bits) : FromBits(bits); }

u32 FloatBits(float value) {
    u32 bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

}  // namespace generated
