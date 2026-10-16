"""The FPU accumulator in the symbolic model, against the interpreter."""

import os
import random
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine
from ps2decomp.paths import derive_paths

BASE = 0x00100000
JR_RA = 0x03E00008


def cop1(funct, fd, fs, ft):
    return (0x11 << 26) | (0x10 << 21) | (ft << 16) | (fs << 11) | (fd << 6) | funct


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


# adda.s $f12, $f13 ; madd.s $f0, $f12, $f13 ; msuba.s $f13, $f13 ; msub.s $f1, $f0, $f12
# then return mfc1 of $f0 + $f1 bits combined: $v0 = f0, $v1 = f1 (we compare $v0).
PROGRAM = [
    cop1(0x18, 0, 12, 13),        # adda.s  ACC = f12 + f13
    cop1(0x1C, 0, 12, 13),        # madd.s  f0 = ACC + f12 * f13
    cop1(0x1F, 0, 13, 13),        # msuba.s ACC = ACC - f13 * f13
    cop1(0x1D, 1, 0, 12),         # msub.s  f1 = ACC - f0 * f12
    cop1(0x00, 2, 0, 1),          # add.s   f2 = f0 + f1
    JR_RA,
    (0x11 << 26) | (0 << 21) | (2 << 16) | (2 << 11),   # mfc1 $v0, $f2 (delay slot)
]


class TestAccumulator(unittest.TestCase):
    def test_derivation_agrees_with_the_interpreter(self):
        image = struct.pack("<%dI" % len(PROGRAM), *PROGRAM)
        derivation = derive_paths(image, BASE, BASE, BASE + 4 * len(PROGRAM), known={})
        machine = Machine(BASE, image, 0x02000000 - BASE)
        rng = random.Random(7)
        for _case in range(50):
            floats = [bits(rng.uniform(-1000, 1000)) for _ in range(8)]
            args = (0,) * 8 + tuple(floats)
            expected = machine.call(BASE, args) & 0xFFFFFFFF
            actual = derivation.evaluate(args, lambda a, s, g: 0) & 0xFFFFFFFF
            self.assertEqual(actual, expected)
        rendered = derivation.paths[0].expression.render()
        self.assertIn("FloatMulAdd(", rendered)
        self.assertIn("FloatMulSub(", rendered)


if __name__ == "__main__":
    unittest.main()
