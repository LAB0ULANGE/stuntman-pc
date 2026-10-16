/*
 * Integer operations whose EE result C++ leaves undefined, for derived code.
 * They follow engine/ps2decomp/emu.py, divide(), which documents the rules.
 */

#include "ps2rt/generated.h"

namespace generated {

namespace {

u64 Sx32(s64 value) { return (u64)(s64)(s32)(u32)(u64)value; }

}  // namespace

u64 DivQuot(u64 a, u64 b) {
    const s64 x = (s32)(u32)a, y = (s32)(u32)b;
    if (y == 0) return x < 0 ? 1 : ~0ull;
    return Sx32(x / y);
}

u64 DivRem(u64 a, u64 b) {
    const s64 x = (s32)(u32)a, y = (s32)(u32)b;
    if (y == 0) return (u64)x;
    return Sx32(x - y * (x / y));
}

u64 DivQuotU(u64 a, u64 b) {
    const u32 x = (u32)a, y = (u32)b;
    if (y == 0) return ~0ull;
    return Sx32((s64)(x / y));
}

u64 DivRemU(u64 a, u64 b) {
    const u32 x = (u32)a, y = (u32)b;
    if (y == 0) return Sx32((s64)x);
    return Sx32((s64)(x % y));
}

}  // namespace generated
