"""Division and unsigned multiply: defined for every input, in every model."""

import os
import random
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine, divide
from ps2decomp.paths import derive_paths

BASE = 0x00100000
JR_RA = 0x03E00008
MASK64 = (1 << 64) - 1


def r_type(funct, rs=0, rt=0, rd=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | funct


MFLO, MFHI = 0x12, 0x10
A0, A1, V0, V1 = 4, 5, 2, 3


class TestDivideRules(unittest.TestCase):
    def test_zero_divisor(self):
        self.assertEqual(divide(7, 0, True), (MASK64, 7))
        self.assertEqual(divide((-7) & 0xFFFFFFFF, 0, True), (1, (-7) & MASK64))
        self.assertEqual(divide(7, 0, False), (MASK64, 7))
        self.assertEqual(divide(0x80000001, 0, False), (MASK64, 0xFFFFFFFF80000001))

    def test_int_min_by_minus_one(self):
        self.assertEqual(divide(0x80000000, 0xFFFFFFFF, True), (0xFFFFFFFF80000000, 0))

    def test_truncates_toward_zero(self):
        self.assertEqual(divide((-7) & 0xFFFFFFFF, 2, True), ((-3) & MASK64, (-1) & MASK64))


class TestDerivedDivision(unittest.TestCase):
    def check(self, words):
        image = struct.pack("<%dI" % len(words), *words)
        derivation = derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={})
        machine = Machine(BASE, image, 0x02000000 - BASE)
        rng = random.Random(len(words))
        for _ in range(200):
            a = rng.choice([0, 1, 7, 0x80000000, 0xFFFFFFFF, rng.getrandbits(32), rng.getrandbits(64)])
            b = rng.choice([0, 0, 1, 2, 0xFFFFFFFF, rng.getrandbits(32), rng.getrandbits(64)])
            expected = machine.call(BASE, (a, b))
            actual = derivation.evaluate((a, b), lambda *x: 0)
            self.assertEqual(actual & 0xFFFFFFFF, expected & 0xFFFFFFFF, (hex(a), hex(b)))

    def test_signed_quotient_and_remainder(self):
        # div $a0, $a1 ; mflo $v1 ; mfhi $v0 ; jr $ra ; addu $v0, $v0, $v1
        self.check([r_type(0x1A, A0, A1), r_type(MFLO, rd=V1), r_type(MFHI, rd=V0), JR_RA,
                    r_type(0x21, V0, V1, V0)])

    def test_unsigned_quotient_and_remainder(self):
        self.check([r_type(0x1B, A0, A1), r_type(MFLO, rd=V1), r_type(MFHI, rd=V0), JR_RA,
                    r_type(0x21, V0, V1, V0)])

    def test_unsigned_multiply(self):
        # multu $a0, $a1 ; mfhi $v0 ; mflo $v1 ; jr $ra ; xor $v0, $v0, $v1
        self.check([r_type(0x19, A0, A1), r_type(MFHI, rd=V0), r_type(MFLO, rd=V1), JR_RA,
                    r_type(0x26, V0, V1, V0)])


if __name__ == "__main__":
    unittest.main()
