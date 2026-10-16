"""A structured function hands back the callee-saved registers it shares with
its callees.

The readable form keeps registers in locals and writes them to the context
before each call (out()). A callee-saved float register the function uses as
scratch therefore reaches the context with the function's value; on return
(leave()) the restored local has to go back, or the caller sees that value.
0x0042DAE0 in Burnout 3 did this with $f20 and $f21, and the car select drew
no cars. Its tests never took a path with a call, so verification passed.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import structure  # noqa: E402

BASE = 0x00100000
SP, RA = 29, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def addiu(rt, rs, imm): return i_type(0x09, rs, rt, imm)
def sd(rt, offset, rs): return i_type(0x3F, rs, rt, offset)
def ld(rt, offset, rs): return i_type(0x37, rs, rt, offset)
def swc1(ft, offset, rs): return i_type(0x39, rs, ft, offset)
def lwc1(ft, offset, rs): return i_type(0x31, rs, ft, offset)
def jal(target): return (3 << 26) | ((target >> 2) & 0x03FFFFFF)


JR_RA = (RA << 21) | 0x08
NOP = 0
# mov.s $f20, $f12
MOV_F20_F12 = (0x11 << 26) | (0x10 << 21) | (0 << 16) | (12 << 11) | (20 << 6) | 0x06


class TestCalleeSavedThroughCalls(unittest.TestCase):
    def setUp(self):
        callee = BASE
        self.start = BASE + 8
        words = [JR_RA, NOP,                    # callee: returns
                 addiu(SP, SP, -32),
                 sd(RA, 16, SP),
                 swc1(20, 0, SP),               # save $f20
                 MOV_F20_F12,                   # use it
                 jal(callee),
                 NOP,
                 lwc1(20, 0, SP),               # restore it
                 ld(RA, 16, SP),
                 JR_RA,
                 addiu(SP, SP, 32)]
        self.image = struct.pack("<%dI" % len(words), *words)
        self.end = BASE + 4 * len(words)

    def test_leave_hands_back_the_restored_float_register(self):
        text = structure.structure_function(self.image, BASE, self.start, self.end, readable=True)
        leave = next(line for line in text.splitlines() if "auto leave" in line)
        self.assertIn("c.f[20] = f20;", leave)
        # ...and the restore before it is kept, not folded away as dead.
        self.assertIn("f20 = ", text.split("auto in")[1].split("leave(); return;")[0])


if __name__ == "__main__":
    unittest.main()
