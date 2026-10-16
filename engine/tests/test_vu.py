"""Tests for the Vector Unit microcode disassembler.

The padding encodings and the bundle layout are pinned down here because the
microcode-location work in docs/FINDINGS.md depends on them, and a regression in
either would silently invalidate that result.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.vu import (
    FLAG_E,
    FLAG_I,
    LOWER_NOP,
    LOWER_SPECIAL_GROUP,
    UPPER_NOP_BASE,
    coverage,
    decode_blob,
    decode_bundle,
    decode_lower,
    decode_upper,
)


class TestBundleLayout(unittest.TestCase):
    def test_lower_comes_first_in_memory(self):
        """A bundle is the lower word then the upper word.

        Confirmed against the real .vutext, where the padding bundle appears as
        8000033C 400002FF - lower nop, then upper nop with the E flag.
        """
        data = struct.pack("<II", LOWER_NOP, UPPER_NOP_BASE | FLAG_E)
        bundle = decode_blob(data, 0)[0]
        self.assertEqual(bundle.lower.mnemonic, "nop")
        self.assertEqual(bundle.upper.mnemonic, "nop")
        self.assertTrue(bundle.is_end)

    def test_flags_are_decoded(self):
        bundle = decode_bundle(0, FLAG_E | FLAG_I | UPPER_NOP_BASE, 0)
        self.assertIn("E", bundle.flags)
        self.assertIn("I", bundle.flags)

    def test_i_flag_makes_the_lower_word_a_float(self):
        """With I set the lower slot is a constant, not an instruction."""
        one = struct.unpack("<I", struct.pack("<f", 1.5))[0]
        bundle = decode_bundle(one, FLAG_I | UPPER_NOP_BASE, 0)
        self.assertAlmostEqual(bundle.immediate, 1.5)
        self.assertEqual(bundle.lower.mnemonic, "loi")

    def test_no_i_flag_decodes_the_lower_word_normally(self):
        bundle = decode_bundle(LOWER_NOP, UPPER_NOP_BASE, 0)
        self.assertIsNone(bundle.immediate)
        self.assertEqual(bundle.lower.mnemonic, "nop")


class TestUpper(unittest.TestCase):
    def test_nop(self):
        self.assertEqual(decode_upper(UPPER_NOP_BASE).mnemonic, "nop")

    def test_broadcast_opcodes_pick_a_lane(self):
        # Opcodes 0x00..0x03 are addx/addy/addz/addw.
        for lane, opcode in enumerate(range(0x00, 0x04)):
            word = (0xF << 21) | opcode
            self.assertTrue(decode_upper(word).mnemonic.startswith("add" + "xyzw"[lane]))

    def test_dest_mask_renders_as_a_suffix(self):
        # All four lanes set -> .xyzw
        self.assertTrue(decode_upper((0xF << 21) | 0x28).mnemonic.endswith(".xyzw"))
        # Only bit 24 -> .x
        self.assertTrue(decode_upper((0x8 << 21) | 0x28).mnemonic.endswith(".x"))

    def test_plain_register_form(self):
        ins = decode_upper((0xF << 21) | 0x2A)  # mul
        self.assertTrue(ins.mnemonic.startswith("mul."))
        self.assertEqual(len(ins.operands), 3)

    def test_unknown_special_is_reported_not_guessed(self):
        # An extended form that is not in the table must be flagged, never
        # invented - a wrong mnemonic is worse than an admitted gap.
        ins = decode_upper(0x3F | (0x1F << 6))
        self.assertFalse(ins.known)
        self.assertIn(".word", ins.format())


class TestLower(unittest.TestCase):
    def test_nop(self):
        self.assertEqual(decode_lower(LOWER_NOP).mnemonic, "nop")

    def test_opcode_is_seven_bits_wide(self):
        """Regression test for a real bug.

        The lower opcode occupies bits 31..25 - seven bits. Masking it to six
        collapsed the special group (0x40) onto opcode 0x00, which made thousands
        of real instructions in the game's microcode undecodable.
        """
        word = (LOWER_SPECIAL_GROUP << 25) | 0x30  # iadd
        self.assertEqual((word >> 25) & 0x7F, LOWER_SPECIAL_GROUP)
        self.assertNotEqual((word >> 25) & 0x3F, LOWER_SPECIAL_GROUP)
        self.assertEqual(decode_lower(word).mnemonic, "iadd")

    def test_special_group_integer_ops(self):
        for sub, name in ((0x30, "iadd"), (0x31, "isub"), (0x34, "iand"), (0x35, "ior")):
            word = (LOWER_SPECIAL_GROUP << 25) | sub
            self.assertEqual(decode_lower(word).mnemonic, name)

    def test_branch_offset_is_signed(self):
        forward = decode_lower((0x20 << 25) | 4)
        self.assertEqual(forward.operands, ("+4",))
        backward = decode_lower((0x20 << 25) | (0x800 - 4))
        self.assertEqual(backward.operands, ("-4",))

    def test_unknown_opcode_is_reported(self):
        ins = decode_lower(0x7F << 25)
        self.assertFalse(ins.known)


class TestCoverage(unittest.TestCase):
    def test_padding_is_not_counted_as_decoded(self):
        """Zero bundles must not inflate the coverage figure.

        A zero upper word decodes as `addx`, so counting it as a success would
        make a blob of padding look fully understood.
        """
        data = struct.pack("<II", 0, 0) * 10
        stats = coverage(decode_blob(data, 0))
        self.assertEqual(stats["bundles"], 10)
        self.assertEqual(stats["padding"], 10)
        self.assertEqual(stats["real"], 0)
        self.assertEqual(stats["upper_pct"], 0.0)

    def test_real_bundles_are_counted(self):
        data = struct.pack("<II", LOWER_NOP, UPPER_NOP_BASE)
        stats = coverage(decode_blob(data, 0))
        self.assertEqual(stats["real"], 1)
        self.assertEqual(stats["upper_pct"], 100.0)

    def test_programs_counted_from_end_bits(self):
        data = (
            struct.pack("<II", LOWER_NOP, UPPER_NOP_BASE)
            + struct.pack("<II", LOWER_NOP, UPPER_NOP_BASE | FLAG_E)
            + struct.pack("<II", LOWER_NOP, UPPER_NOP_BASE | FLAG_E)
        )
        self.assertEqual(coverage(decode_blob(data, 0))["programs"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
