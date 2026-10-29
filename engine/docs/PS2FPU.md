# The PS2 Emotion Engine FPU's arithmetic, as the hardware computes it

A behavioural description of the datapaths of the R5900's single-precision
FPU (also the VU0 macro mode's): add, subtract, multiply, divide and square
root, bit for bit. It describes what the circuits compute; it contains no
program code.

## Number format

32 bits: sign (bit 31), biased exponent E (bits 30-23), fraction F (bits
22-0). It is *not* IEEE-754:

- Every E from 1 to 255 is an ordinary number, value (1.F) * 2^(E-127).
  There is no infinity and no NaN; E = 255 is just the largest binade.
- E = 0 is zero, whatever F holds (denormals do not exist; they read as zero
  of that sign).
- The largest magnitude is 0x7FFFFFFF (E 255, F all ones). A result whose
  exponent would exceed 255 saturates to sign | 0x7FFFFFFF.
- A result whose exponent would fall below 1 becomes a zero carrying the
  result's sign (sign bit only, everything else 0).
- The significand M of a nonzero operand is the 24-bit integer 1F (the hidden
  bit, value 0x800000, set).

All results are truncated (never rounded up) in the ways described below;
"truncated" means extra low bits are simply dropped.

## Addition and subtraction

Subtraction is addition with the second operand's sign bit flipped.

- If either operand is zero (E = 0): the result is the other operand,
  unchanged, if that one is nonzero. If both are zero, the result is +0
  (0x00000000) unless both signs are negative, in which case -0 (0x80000000).
- Otherwise call A the operand with the larger exponent (if the exponents are
  equal, A is the first operand) and B the other; d = E_A - E_B.
- If d >= 25, the result is A unchanged.
- The aligner keeps exactly one bit of B below A's last significand bit (a
  single guard bit) and discards the rest: when d >= 1, the d-1 lowest bits
  of B's fraction field are cleared *before* alignment (they never enter the
  adder). Then B's significand is shifted right by d with that one guard bit
  kept, i.e. the adder works on (M_A * 2) and (M_B * 2) / 2^d, which is exact
  once those bits are cleared.
- The two signed values are summed exactly (sign-magnitude to signed). If the
  exact sum is 0, the result is +0 (0x00000000).
- Normalise: find the highest set bit of |sum|; the result significand is the
  24 bits starting at that bit (lower bits truncated; if fewer than 24 bits
  exist they are padded with zeros below). The exponent is E_A + (position of
  the highest set bit) - 24, counting bit positions from 0 in the scaled sum
  above (so a sum whose top bit is at position 24, i.e. the same scale as
  2 * M_A, gives E_A).
- Pack with saturation/flush as in "Number format" and the sign of the sum.

## Multiplication

- Sign = sign_a XOR sign_b. If either operand is zero, the result is a zero
  of that sign.
- The significands (24-bit) are multiplied into a 48-bit product P. The
  result keeps P >> 23 (24 or 25 significant bits); if that is >= 2^24 it is
  shifted right one more (truncated) and the exponent incremented. Exponent =
  E_a + E_b - 127 (+1 for that shift). Pack with saturation / flush.
- **The one deviation from the exact product is bit 15 of P.** The hardware
  is a radix-4 Booth multiplier whose low partial-product columns are not all
  summed, so bit 15 of the product can differ from the true product's bit 15;
  everything else is the true product's, *except* for the borrow/carry that
  the corrected bit 15 propagates. Concretely: the hardware's product equals
  the true product P with bit 15 replaced by the hardware's own bit 15 value
  L15, computed as below; formally P_hw = P - ((L15 xor P) & 0x8000) where L15
  is a 32-bit word whose bit 15 is the hardware's bit. (Only bit 15 of L
  matters.) Since only P >> 23 is kept, this can only change the result when
  bits 15..22 of P are all zero (the borrow then propagates upward); in every
  other case P itself may be used.

### How the hardware forms L (the low 32 bits of its partial-product sum)

The multiplier digit recoding uses only the multiplier's low 16 bits, which
yield 8 radix-4 Booth digits, d = 0..7. All quantities below are 32-bit
unsigned words with wrap-around; "a" is the multiplicand significand and "b"
the multiplier significand (both 24-bit, hidden bit set).

Partial product of digit d:
- The 3-bit window for digit d is bits (2d+1, 2d, 2d-1) of b, where bit -1 is
  taken as 0 (for d = 0 the window is bits 1, 0 and an implicit 0).
- Base value V = a shifted left by 2d.
- Window 3 (011) or 4 (100): V is doubled (the "2x multiplicand" case).
- Windows 4, 5, 6 are negative digits: V is ones'-complemented in all bit
  positions from 2^(2d) upward (bits below 2^(2d) stay as they are, i.e. V
  XOR (all ones masked to bits >= 2d)), and a separate "negation bit" N_d =
  2^(2d) is produced (the +1 that completes the two's complement). For
  positive digits N_d = 0.
- Windows 0 (000) and 7 (111) are zero digits: V = 0 (N_d is still as given
  by the negative rule above, i.e. 0, since 0 and 7 are not negative).
- Call the resulting value PP_d and its negation bit N_d.

Carry-save adders: a 3:2 compressor of (x, y, z) gives sum = x ^ y ^ z and
carry = ((x & y) | ((x ^ y) & z)) << 1 (32-bit, the top carry lost).

The tree, with the hardware's truncations:
1. S0 = CSA(PP1, PP2, PP3).
2. S1 = CSA(PP4 with bits 0..10 cleared, PP5 with bits 0..11 cleared, PP6).
   Then S1.carry gets OR'd with N_6 and with (PP5 & 0x800) (PP5's bit 11
   moves into the carry word instead).
3. PP7 is modified: PP7 |= (PP5 & 0x400) + N_5 (that sum, OR'd in).
4. S2 = CSA(PP0, S0.sum, S0.carry).
5. S3 = CSA(PP7 (modified), S1.sum, S1.carry).
6. S4 = CSA(S2.carry, S3.sum, S3.carry).
7. S5 = CSA(S2.sum, S4.sum, S4.carry); then S5.carry += N_7.
8. L = (S5.sum with bits 0..14 cleared) + (S5.carry with bits 0..14 cleared).

(N_0 .. N_4 never enter: they fall in columns the hardware does not sum.)

## Division (a / b)

- Sign = sign_a XOR sign_b.
- Divisor zero (E_b = 0): result sign | 0x7FFFFFFF (also for 0 / 0).
- Dividend zero: result is a zero of that sign.
- Otherwise a radix-2 SRT divider, digits in {-1, 0, +1}, 25 iterations, no
  final correction or remainder-sign fix-up, and the quotient digits are just
  accumulated (the final quotient may therefore be one unit low).

The partial remainder is kept in carry-save form (two 32-bit words, sum and
carry, wrap-around) with the binary point between bits 24 and 25. Initially:
sum = M_a << 2, carry = 0; the divisor word D = M_b << 2. The quotient Q
starts at 0, and the "current digit" starts at +1.

Digit selection from a carry-save remainder (s, c): form the estimate
T = ((s with bits 0..23 cleared) + c) OR (s's bits 0..23), interpreted as a
signed 32-bit integer (the low 24 bits of s are OR'd in, not added). The
digit is +1 if T >= 2^23 (i.e. >= 0.25), -1 if T < -(2^24) (i.e. < -0.5),
else 0.

Each of the 25 iterations:
1. Q = Q * 2 + current digit (as a signed +1/0/-1, in 32-bit wrap-around).
2. Remainder update for the current digit: if +1, add the ones' complement
   of D and add 1 to the carry word (carry += 1) before compressing; if -1,
   add D; if 0, add 0. The addition is a carry-save compression:
   next = CSA(sum, carry, addend) (carry += 1 is applied to the carry word
   before the CSA when the digit is +1).
3. The next digit is selected from `next` if the current digit was nonzero;
   if the current digit was 0, it is selected from the remainder as it was
   before this step (sum, carry after the +1 adjustment, which for digit 0 is
   none).
4. The remainder becomes next shifted left by one (both words).

After 25 iterations: exponent = E_a - E_b + 126; if Q >= 2^24, Q is shifted
right by one (truncated) and the exponent incremented. Pack (sign, exponent,
low 23 bits of Q) with saturation / flush.

## Square root

- The sign of the operand is ignored (the root of |a|); the result is always
  positive.
- Zero operand: +0.
- Otherwise a radix-2 SRT square root using the same remainder,
  digit-selection and step rules as division, 25 iterations, no correction.
- Initial remainder: sum = M_a << 1, further shifted left by one more if the
  exponent field's lowest bit is 0 (equivalently, if bit 23 of the operand
  word is clear: an odd unbiased exponent); carry = 0. Root R starts at 0,
  current digit +1.
- Iteration i = 0..24:
  1. The "subtrahend" for this step is R + digit * 2^(24-i) (taken before R
     changes; digit as +1/0/-1, 32-bit wrap-around).
  2. R = R + digit * 2^(25-i).
  3. Remainder update and next-digit selection exactly as division's step,
     with that subtrahend in the place of D.
- Result: fraction = (R >> 2) low 23 bits; exponent = (E_a + 127) >> 1
  (integer division). No saturation is possible.

## Interface to implement

```
// all in namespace ps2fpu (or as the integrator asks)
u32 Add(u32 a, u32 b);      // a + b
u32 Mul(u32 a, u32 b);
u32 Div(u32 a, u32 b);
u32 Sqrt(u32 a);
```

Subtraction is Add(a, b ^ 0x80000000).
