// ps2fpu.h - bit-exact model of the PS2 Emotion Engine FPU arithmetic
// (add, multiply, divide, square root). Integer-only; no host floating point.
//
// Values are passed as raw 32-bit words (sign | 8-bit exponent | 23-bit
// fraction). The format is not IEEE-754: exponent 0 is always zero, exponent
// 255 is an ordinary binade, results are truncated and saturate to
// sign | 0x7FFFFFFF on overflow.
#pragma once
#include <cstdint>

namespace ps2fpu {

uint32_t Add(uint32_t a, uint32_t b);   // a + b   (a - b is Add(a, b ^ 0x80000000))
uint32_t Mul(uint32_t a, uint32_t b);   // a * b
uint32_t Div(uint32_t a, uint32_t b);   // a / b
uint32_t Sqrt(uint32_t a);              // sqrt(|a|)

} // namespace ps2fpu
