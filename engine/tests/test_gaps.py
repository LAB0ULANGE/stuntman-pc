"""Recovering functions the call-graph analysis cannot reach."""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import gaps  # noqa: E402

JR_RA, NOP = 0x03E00008, 0


def code(*words):
    return struct.pack("<%dI" % len(words), *words)


def j(target):
    return (2 << 26) | ((target >> 2) & 0x3FFFFFF)


class TestExtent(unittest.TestCase):
    def test_a_function_ends_at_its_return(self):
        self.assertEqual(gaps.function_extent(code(NOP, JR_RA, NOP, 0xDEADBEEF), 0x1000, 0x1000), 12)

    def test_a_branch_past_a_return_extends_the_function(self):
        b_forward_3 = (4 << 26) | 3          # beq $zero, $zero: to the second return
        words = code(b_forward_3, NOP, JR_RA, NOP, JR_RA, NOP)
        self.assertEqual(gaps.function_extent(words, 0x1000, 0x1000), 24)

    def test_a_tail_jump_ends_a_function(self):
        self.assertEqual(gaps.function_extent(code(NOP, j(0x8000), NOP, 0xDEADBEEF), 0x1000, 0x1000), 12)

    def test_a_jump_inside_the_function_does_not(self):
        words = code(j(0x100C), NOP, NOP, JR_RA, NOP)
        self.assertEqual(gaps.function_extent(words, 0x1000, 0x1000), 20)

    def test_no_exit_before_the_limit_is_an_error(self):
        with self.assertRaises(ValueError):
            gaps.function_extent(code(NOP, NOP, JR_RA, NOP), 0x1000, 0x1000, limit=0x1008)


class TestRecover(unittest.TestCase):
    def test_a_pointed_to_function_in_a_gap_is_recovered(self):
        # known: [0x1000, 0x1008); gap function at 0x1010; a pointer to it in data.
        image = code(JR_RA, NOP, NOP, NOP, NOP, JR_RA, NOP, 0x1010)
        found = gaps.recover(image, 0x1000, [(0x1000, 0x1008)])
        self.assertIn((0x1010, 0x101C, "pointer"), found)

    def test_nothing_is_recovered_inside_a_known_function(self):
        image = code(NOP, NOP, JR_RA, NOP, 0x1004)
        self.assertEqual(gaps.recover(image, 0x1000, [(0x1000, 0x1010)]), [])


if __name__ == "__main__":
    unittest.main()
