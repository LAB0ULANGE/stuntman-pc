/*
 * Support declarations for the derived accessor functions.
 *
 * The derived functions in src/generated/ read memory through these helpers
 * rather than casting integers to pointers directly. Two reasons: the emitted
 * source stays readable, and a port can define them over whatever memory model it
 * ends up using instead of assuming a flat PS2 address space.
 *
 * Recovered struct types are views over these same helpers; see the generated
 * src/generated/types.h.
 *
 * Every value in the derived code is a u64, because that is the width of an EE
 * general register and the width the derivations were verified at. Arguments are
 * u64 for the same reason: a 32-bit value arrives in its register sign-extended,
 * which is what passing an s32 to a u64 parameter does.
 */

#ifndef PS2RT_GENERATED_H
#define PS2RT_GENERATED_H

#include <cstring>

#include "ps2rt/types.h"

namespace generated {

/*
 * Reads from the game's address space (ReadS8 ... ReadU64 below); on the console
 * they are plain loads.
 *
 * The signed variants matter: the original uses `lb`/`lh`/`lw` (sign-extending)
 * in some places and `lbu`/`lhu` (zero-extending) in others, and the derived code
 * preserves that distinction because callers can depend on it.
 *
 * Each returns the value as it lands in a 64-bit register - already sign- or
 * zero-extended to a u64 - because that is what the derived code computes on.
 * Returning the narrow type instead lets C++'s promotion rules decide the width
 * of the arithmetic, and those make `ReadU8(p) << 24` a signed `int` shift.
 */
/*
 * Main RAM, which almost every access reaches, can be given to the derived code
 * directly: a port that sets `ram_base` to its 32 MB lets the reads and writes
 * below load and store there inline - at 0x00000000, its uncached mirrors
 * 0x20000000 and 0x30000000, and kseg0/kseg1 - and sends everything else (the
 * device registers, the scratchpad, an access running off the end) through the
 * port's functions above. `ram_watched` sends writes through them too, for a
 * port watching addresses.
 */
extern u8* ram_base;
extern bool ram_watched;
/* The scratchpad's 16 KB at 0x70000000, given the same way (or null). */
extern u8* scratchpad_base;

inline u8* RamPointer(u32 address, u32 size) {
    if (address - 0x80000000u < 0x40000000u) address &= 0x1FFFFFFFu;
    const u32 window = address >> 28;
    if (window == 7) {
        const u32 offset = address - 0x70000000u;
        return scratchpad_base && offset + size <= 0x4000u ? scratchpad_base + offset : nullptr;
    }
    if (!ram_base || (window != 0 && window != 2 && window != 3)) return nullptr;
    const u32 offset = address & 0x0FFFFFFFu;
    return offset + size <= (32u << 20) ? ram_base + offset : nullptr;
}

template <typename T>
inline bool RamLoad(u32 address, T* value) {
    const u8* p = RamPointer(address, sizeof(T));
    if (!p) return false;
    std::memcpy(value, p, sizeof(T));
    return true;
}

template <typename T>
inline bool RamStore(u32 address, T value) {
    if (ram_watched) return false;
    u8* p = RamPointer(address, sizeof(T));
    if (!p) return false;
    std::memcpy(p, &value, sizeof(T));
    return true;
}

/* The port's own, for what is not main RAM. */
u64 PortRead(u32 address, u32 size, bool is_signed);
void PortWrite(u32 address, u64 value, u32 size);

inline u64 ReadS8(u32 a) { s8 v; return RamLoad(a, &v) ? u64(s64(v)) : PortRead(a, 1, true); }
inline u64 ReadU8(u32 a) { u8 v; return RamLoad(a, &v) ? u64(v) : PortRead(a, 1, false); }
inline u64 ReadS16(u32 a) { s16 v; return RamLoad(a, &v) ? u64(s64(v)) : PortRead(a, 2, true); }
inline u64 ReadU16(u32 a) { u16 v; return RamLoad(a, &v) ? u64(v) : PortRead(a, 2, false); }
inline u64 ReadS32(u32 a) { s32 v; return RamLoad(a, &v) ? u64(s64(v)) : PortRead(a, 4, true); }
inline u64 ReadU32(u32 a) { u32 v; return RamLoad(a, &v) ? u64(v) : PortRead(a, 4, false); }
inline u64 ReadU64(u32 a) { u64 v; return RamLoad(a, &v) ? v : PortRead(a, 8, false); }

/*
 * Guest stack, for a derived function whose frame a callee can see - one that
 * passes the address of a local, an out-parameter. StackEnter reserves `size`
 * bytes below the current guest stack pointer and returns the new one;
 * StackLeave releases them. Derived code that only spills registers needs no
 * guest stack at all.
 */
u64 StackEnter(u32 size);
void StackLeave(u32 size);

/*
 * A `break` the original executes: a trap (an assertion's failure, usually).
 * The derivation ends that path here; whatever hosts derived code stops the
 * run there, as the console raises the exception. It does not return. (Not
 * `Trap`: lifted::Trap is the lifted code's, and structured code sees both.)
 */
void TrapAt(u32 address);

/*
 * A call through a function pointer or a virtual method: `address` is the guest
 * address the original code jumps to. A port maps guest addresses to its own
 * functions; the arguments are the full register argument vector - eight integer,
 * then eight float - since which function is reached, and so what it reads, is
 * only known at run time. Returns $v0.
 */
u64 CallPointer(u32 address, u64 arg0 = 0, u64 arg1 = 0, u64 arg2 = 0, u64 arg3 = 0,
                u64 arg4 = 0, u64 arg5 = 0, u64 arg6 = 0, u64 arg7 = 0,
                u64 farg0 = 0, u64 farg1 = 0, u64 farg2 = 0, u64 farg3 = 0,
                u64 farg4 = 0, u64 farg5 = 0, u64 farg6 = 0, u64 farg7 = 0);

/* Writes, for the derived functions that have memory effects. */
inline void Write8(u32 a, u32 v) { if (!RamStore(a, u8(v))) PortWrite(a, v, 1); }
inline void Write16(u32 a, u32 v) { if (!RamStore(a, u16(v))) PortWrite(a, v, 2); }
inline void Write32(u32 a, u32 v) { if (!RamStore(a, u32(v))) PortWrite(a, v, 4); }
inline void Write64(u32 a, u64 v) { if (!RamStore(a, v)) PortWrite(a, v, 8); }

/*
 * Single-precision arithmetic over raw bit patterns.
 *
 * Operands and results are 32-bit patterns rather than `float`, because that is
 * what the hardware registers hold: `lwc1`, `swc1`, `mtc1` and `mfc1` move all 32
 * bits without interpreting them, so a copy must preserve a payload exactly.
 * Interpretation belongs to the arithmetic, which is why it lives here.
 *
 * Two Emotion Engine behaviours a port must reproduce rather than "fix":
 *
 *  - The FPU is not IEEE-754. There are no infinities and no NaNs; an operation
 *    that overflows **saturates** to the largest representable magnitude
 *    (0x7F7FFFFF). Letting it produce an infinity instead will make physics
 *    diverge.
 *  - Division by zero does not trap. It yields zero.
 */
/* A double result as the EE's FPU leaves it in single precision: saturated, NaN
   to zero, rounded toward zero, denormals flushed (eefpu.cpp). */
u32 FloatNarrow(double value);
u32 FloatAdd(u32 a, u32 b);
u32 FloatSub(u32 a, u32 b);
u32 FloatMul(u32 a, u32 b);
u32 FloatDiv(u32 a, u32 b);
u32 FloatMin(u32 a, u32 b);
u32 FloatMax(u32 a, u32 b);
u32 FloatNeg(u32 a);
u32 FloatAbs(u32 a);
u32 FloatSqrt(u32 a);
u32 FloatMove(u32 a);

/* 32-bit division as the EE does it (div/divu), results sign-extended into a
   register: quotient and remainder. A zero divisor gives a defined result - see
   engine/ps2decomp/emu.py, divide(). */
u64 DivQuot(u64 a, u64 b);
u64 DivRem(u64 a, u64 b);
u64 DivQuotU(u64 a, u64 b);
u64 DivRemU(u64 a, u64 b);

/* acc + a * b and acc - a * b, rounded once: the FPU accumulator forms. */
u32 FloatMulAdd(u32 acc, u32 a, u32 b);
u32 FloatMulSub(u32 acc, u32 a, u32 b);

/* VU0 arithmetic on one lane, as engine/ps2decomp/vu0.py models it (operands
   flushed and clamped, results rounded and saturated). Shared with lifted code. */
u32 VuAdd(u32 a, u32 b);
u32 VuSub(u32 a, u32 b);
u32 VuMul(u32 a, u32 b);
u32 VuMax(u32 a, u32 b);
u32 VuMin(u32 a, u32 b);
u32 VuMadd(u32 acc, u32 a, u32 b);
u32 VuMsub(u32 acc, u32 a, u32 b);
u32 VuAbs(u32 a);
u32 VuItof(u32 a, u32 shift);
u32 VuFtoi(u32 a, u32 shift);
u32 VuDiv(u32 num, u32 den);
u32 VuSqrt(u32 a);
u32 VuRsqrt(u32 num, u32 den);

/* Conversions between float bit patterns and integers. */
u32 FloatFromInt(u32 bits_as_int);
u32 IntFromFloat(u32 float_bits);

/* Interprets a bit pattern as a float, for use in comparisons. */
float FloatValue(u32 bits);

/* The bit pattern of a float constant, so the source can say `FloatBits(-1.0f)`
   rather than `0xBF800000`. Every literal emitted converts back exactly. */
u32 FloatBits(float value);

}  // namespace generated

#endif  /* PS2RT_GENERATED_H */
