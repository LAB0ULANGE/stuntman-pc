/*
 * Fixed-width types for the decompiled sources.
 *
 * The PS2 toolchain of the era spelled these as bare typedefs, and the
 * decompiled code keeps that spelling so the source reads the way the original
 * would have.  They are defined against <cstdint> so the same sources also
 * build on a modern host compiler for the eventual port.
 */

#ifndef PS2RT_TYPES_H
#define PS2RT_TYPES_H

#include <cstddef>
#include <cstdint>

using s8  = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

/*
 * The Emotion Engine's 128-bit quadword. Several of the game's hot paths move
 * data a quadword at a time via lq/sq, so the type is named explicitly rather
 * than hidden behind a pair of u64s.
 */
struct u128 {
    u64 lo;
    u64 hi;
};

static_assert(sizeof(u128) == 16, "u128 must be exactly one quadword");

#endif  /* PS2RT_TYPES_H */
