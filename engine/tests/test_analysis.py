"""Tests for function discovery's "does this look like code" judgement.

The check has to reject data that merely decodes - runs of zeroes are valid
`nop`s - while accepting real functions, including the very short ones: a getter
is `jr $ra; lw $v0, 8($a0)`, and an empty function is just `jr $ra; nop`. Getting
the short ones wrong lost 424 call targets, one called from 866 sites.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.analysis import ImageAnalyzer

BASE = 0x00100000
JR_RA = 0x03E00008
NOP = 0
LW_V0_8_A0 = (0x23 << 26) | (4 << 21) | (2 << 16) | 8
J_ELSEWHERE = (2 << 26) | ((BASE + 0x100) >> 2)
JAL = (3 << 26) | ((BASE + 0x100) >> 2)
B_FORWARD = 0x10000004         # beq $zero, $zero, +4


def analyzer(words):
    words = list(words) + [NOP] * 64
    return ImageAnalyzer(struct.pack("<%dI" % len(words), *words), BASE, BASE)


class TestLooksLikeCode(unittest.TestCase):
    def test_getter_with_its_body_in_the_delay_slot(self):
        self.assertTrue(analyzer([JR_RA, LW_V0_8_A0])._looks_like_code(BASE))

    def test_empty_function_is_accepted_only_as_a_call_target(self):
        a = analyzer([JR_RA, NOP])
        self.assertTrue(a._looks_like_code(BASE, called=True))
        self.assertFalse(a._looks_like_code(BASE))

    def test_thunk_is_accepted_as_a_call_target(self):
        self.assertTrue(analyzer([J_ELSEWHERE, NOP])._looks_like_code(BASE, called=True))

    def test_zeroes_are_not_code(self):
        self.assertFalse(analyzer([NOP] * 8)._looks_like_code(BASE, called=True))

    def test_empty_delay_slot_does_not_count_as_padding(self):
        """`jal; nop; b; ...` is real code; counting its nop delay slot as padding
        tipped 37 pointer-table entries below the threshold."""
        self.assertTrue(analyzer([JAL, NOP, B_FORWARD, NOP])._looks_like_code(BASE))


class TestSwitches(unittest.TestCase):
    """A switch's cases belong to its function, not to functions of their own."""

    # switch (a0) { case 0: ...; case 1: ...; }, its table at BASE + 0x100.
    CODE = [
        0x2C810002,     # 00 sltiu $at, $a0, 2
        0x1020000A,     # 04 beq $at, $zero, 0x30
        NOP,            # 08
        0x3C030010,     # 0C lui $v1, 0x0010
        0x00042080,     # 10 sll $a0, $a0, 2
        0x24630100,     # 14 addiu $v1, $v1, 0x0100
        0x00832021,     # 18 addu $a0, $a0, $v1
        0x8C820000,     # 1C lw $v0, 0($a0)
        0x00400008,     # 20 jr $v0
        NOP,            # 24
        0x24420001,     # 28 case 0: addiu $v0, $v0, 1
        JR_RA, NOP,     # 2C
        0x24420002,     # 34 case 1: addiu $v0, $v0, 2
        JR_RA, NOP,     # 38
    ]

    def image(self):
        words = self.CODE + [NOP] * (0x40 - len(self.CODE))
        words += [BASE + 0x28, BASE + 0x34] + [NOP] * 32
        return ImageAnalyzer(struct.pack("<%dI" % len(words), *words), BASE, BASE)

    def test_the_table_is_read_from_the_code_before_the_jump(self):
        self.assertEqual(self.image().jump_table(BASE + 0x20, 2), [BASE + 0x28, BASE + 0x34])

    def test_the_low_half_may_be_in_the_load(self):
        # lui $v1, 0x0010; addu $v1, $v1, $a0; lw $v1, 0x0100($v1); jr $v1
        code = list(self.CODE)
        code[3:9] = [0x00042080, 0x3C030010, 0x00641821, NOP, 0x8C630100, 0x00600008]
        words = code + [NOP] * (0x40 - len(code)) + [BASE + 0x28, BASE + 0x34] + [NOP] * 32
        a = ImageAnalyzer(struct.pack("<%dI" % len(words), *words), BASE, BASE)
        self.assertEqual(a.jump_table(BASE + 0x20, 3), [BASE + 0x28, BASE + 0x34])

    def test_the_table_may_be_summed_into_another_register(self):
        # CodeWarrior: lui $v0, 0x0010; addiu $v1, $v0, 0x0100; sll $v0, $a0, 2;
        # addu $v0, $v0, $v1; lw $v0, 0($v0); jr $v0
        code = list(self.CODE)
        code[3:9] = [0x3C020010, 0x24430100, 0x00041080, 0x00431021, 0x8C420000, 0x00400008]
        words = code + [NOP] * (0x40 - len(code)) + [BASE + 0x28, BASE + 0x34] + [NOP] * 32
        a = ImageAnalyzer(struct.pack("<%dI" % len(words), *words), BASE, BASE)
        self.assertEqual(a.jump_table(BASE + 0x20, 2), [BASE + 0x28, BASE + 0x34])

    def test_cases_are_labels_of_the_switch(self):
        a = self.image()
        a.run()
        self.assertEqual(sorted(a.functions), [BASE])
        self.assertEqual(a.functions[BASE].end, BASE + 0x40)


if __name__ == "__main__":
    unittest.main()
