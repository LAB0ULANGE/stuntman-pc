"""PROT3W and PPACW: the interpreter and the lifted C++ against the EE manual's definition."""

import os
import random
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT]

from ps2decomp import r5900  # noqa: E402
from ps2decomp.emu import Machine, is_supported  # noqa: E402
from ps2decomp import emu  # noqa: E402

BASE = 0x00100000


def prot3w(rd, rt):
    return (0x1C << 26) | (rt << 16) | (rd << 11) | (0x1F << 6) | 0x09


def reference(value):
    w = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
    return w[1] | (w[2] << 32) | (w[0] << 64) | (w[3] << 96)


class Prot3w(unittest.TestCase):
    def test_decodes(self):
        ins = r5900.decode(prot3w(6, 5))
        self.assertEqual((ins.mnemonic, ins.rd, ins.rt), ("prot3w", 6, 5))
        self.assertTrue(is_supported(ins))

    def test_interpreter(self):
        machine = Machine(BASE, b"\0" * 16, 0x01000000)
        rng = random.Random(3)
        for rd, rt in ((6, 5), (5, 5), (0, 5)):        # distinct, in place, to $zero
            for _ in range(200):
                value = rng.getrandbits(128)
                machine.regs[rt], machine.regs_hi[rt] = value & (2**64 - 1), value >> 64
                emu._HANDLERS["prot3w"](machine, r5900.decode(prot3w(rd, rt)))
                got = machine.regs[rd] | (machine.regs_hi[rd] << 64)
                self.assertEqual(got, 0 if rd == 0 else reference(value))


def ppacw(rd, rs, rt):
    return (0x1C << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (0x13 << 6) | 0x08


def ppacw_reference(a, b):
    word = lambda v, i: (v >> (32 * i)) & 0xFFFFFFFF
    return word(b, 0) | (word(b, 2) << 32) | (word(a, 0) << 64) | (word(a, 2) << 96)


class Ppacw(unittest.TestCase):
    def test_interpreter(self):
        machine = Machine(BASE, b"\0" * 16, 0x01000000)
        rng = random.Random(5)
        for rd, rs, rt in ((6, 4, 5), (5, 4, 5), (4, 4, 5), (6, 5, 5)):
            for _ in range(200):
                a, b = rng.getrandbits(128), rng.getrandbits(128)
                machine.regs[4], machine.regs_hi[4] = a & (2**64 - 1), a >> 64
                machine.regs[5], machine.regs_hi[5] = b & (2**64 - 1), b >> 64
                x = a if rs == 4 else b
                ins = r5900.decode(ppacw(rd, rs, rt))
                self.assertEqual(ins.mnemonic, "ppacw")
                emu._HANDLERS["ppacw"](machine, ins)
                got = machine.regs[rd] | (machine.regs_hi[rd] << 64)
                self.assertEqual(got, ppacw_reference(x, b))


if __name__ == "__main__":
    unittest.main()
