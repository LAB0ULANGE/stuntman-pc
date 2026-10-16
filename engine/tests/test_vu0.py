"""VU0 macro mode in the interpreter: the model vu0.py defines, pinned by programs."""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import vu0
from ps2decomp.emu import Machine, UnsupportedInstruction
from ps2decomp.r5900 import DISASM

BASE = 0x00100000
JR_RA = 0x03E00008
COP2 = 0x12 << 26


def f(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def pcpyld(rd, rs, rt): return (0x1C << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (0x0E << 6) | 0x09
def qmtc2(rt, vf): return COP2 | (5 << 21) | (rt << 16) | (vf << 11)
def qmfc2(rt, vf): return COP2 | (1 << 21) | (rt << 16) | (vf << 11)
def cfc2(rt, reg): return COP2 | (2 << 21) | (rt << 16) | (reg << 11)
def special1(funct, fd, fs, ft, dest=0xF):
    return COP2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11) | (fd << 6) | funct
def special2(index, fs, ft, dest=0xF):
    return COP2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11) | ((index >> 2) << 6) | 0x3C | (index & 3)


def run(words, args=()):
    m = Machine(BASE, struct.pack("<%dI" % len(words), *words), 0x02000000 - BASE)
    m.call(BASE, args)
    return m


def lanes(m, reg):
    return [m.regs[reg] & 0xFFFFFFFF, m.regs[reg] >> 32,
            m.regs_hi[reg] & 0xFFFFFFFF, m.regs_hi[reg] >> 32]


class TestDecoding(unittest.TestCase):
    def test_special2_is_indexed_by_bits_10_to_6_and_1_to_0(self):
        self.assertEqual(DISASM.decode(special2(0x38, 1, 2), 0).mnemonic, "vdiv")
        self.assertEqual(DISASM.decode(special2(0x30, 1, 2), 0).mnemonic, "vmove.xyzw")
        self.assertEqual(DISASM.decode(special1(0x03, 3, 1, 2, dest=8), 0).mnemonic, "vaddw.x")


class TestArithmetic(unittest.TestCase):
    # $a0..$a3 hold vf1 = (1, 2, 3, 4) and vf2 = (10, 20, 30, 40) as 64-bit pairs.
    ARGS = (f(1.0) | f(2.0) << 32, f(3.0) | f(4.0) << 32,
            f(10.0) | f(20.0) << 32, f(30.0) | f(40.0) << 32)
    LOAD = [pcpyld(8, 5, 4), qmtc2(8, 1), pcpyld(9, 7, 6), qmtc2(9, 2)]

    def test_vector_add(self):
        m = run(self.LOAD + [special1(0x28, 3, 1, 2), qmfc2(2, 3), JR_RA, 0], self.ARGS)
        self.assertEqual(lanes(m, 2), [f(11.0), f(22.0), f(33.0), f(44.0)])

    def test_destination_mask_and_broadcast(self):
        # vaddw.xz vf2, vf1, vf1w: x and z get +4; y and w keep vf2's values.
        m = run(self.LOAD + [special1(0x03, 2, 1, 1, dest=0b1010), qmfc2(2, 2), JR_RA, 0],
                self.ARGS)
        self.assertEqual(lanes(m, 2), [f(5.0), f(20.0), f(7.0), f(40.0)])

    def test_vf0_is_the_constant_0001_and_cannot_be_written(self):
        m = run([special1(0x28, 0, 0, 0), qmfc2(2, 0), JR_RA, 0])
        self.assertEqual(lanes(m, 2), [0, 0, 0, f(1.0)])

    def test_divide_into_q_then_multiply_by_q(self):
        # vdiv Q, vf2x, vf1y (10 / 2) ; vmulq vf3, vf1, Q
        divide = special2(0x38, 2, 1) & ~(0xF << 21) | (0 << 21) | (1 << 23)
        m = run(self.LOAD + [divide, special1(0x1C, 3, 1, 0), qmfc2(2, 3), JR_RA, 0], self.ARGS)
        self.assertEqual(lanes(m, 2), [f(5.0), f(10.0), f(15.0), f(20.0)])

    def test_overflow_saturates_to_the_largest_finite_value(self):
        big = 0x7F7FFFFF
        args = (big | big << 32, big | big << 32)
        m = run([pcpyld(8, 5, 4), qmtc2(8, 1), special1(0x28, 3, 1, 1), qmfc2(2, 3), JR_RA, 0],
                args)
        self.assertEqual(lanes(m, 2), [big] * 4)

    def test_max_and_mini_hand_back_an_operand_uncleaned(self):
        # MAXI with I = 0x412, a denormal: it stays 0x412 (a GIF tag's REGS).
        self.assertEqual(vu0.vu_binary("max", 0, 0x412), 0x412)
        self.assertEqual(vu0.vu_binary("mini", 0, 0x412), 0)
        # Sign and magnitude: -2 < -1 < 0 < 1, the exponent of 255 the largest.
        self.assertEqual(vu0.vu_binary("max", f(-2.0), f(-1.0)), f(-1.0))
        self.assertEqual(vu0.vu_binary("mini", f(-2.0), f(-1.0)), f(-2.0))
        self.assertEqual(vu0.vu_binary("max", 0x7F800001, f(1.0)), 0x7F800001)
        self.assertEqual(vu0.vu_binary("mini", 0x80000412, 0), 0x80000412)


class TestRefusals(unittest.TestCase):
    def test_reading_the_mac_flags_is_refused(self):
        with self.assertRaises(UnsupportedInstruction):
            run([cfc2(2, vu0.REG_MAC), JR_RA, 0])

    def test_reading_vi_registers_is_fine(self):
        run([cfc2(2, 5), JR_RA, 0])

    def test_vpu_stat_reads_idle(self):
        # No VU program runs in the model, so a wait for VU1 ends at once.
        m = run([cfc2(2, vu0.REG_VPU_STAT), JR_RA, 0])
        self.assertEqual(m.get(2), 0)


if __name__ == "__main__":
    unittest.main()
