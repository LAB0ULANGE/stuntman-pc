"""Tests for the per-game configuration: what is detected, what is configured.

The engine is generic; a game supplies where its files are and the facts that
differ between games. The global pointer is the one fact that always differs, and
it is detected from the startup code rather than written down.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import game, layout

BASE = 0x00100000


class FakeSection:
    def __init__(self, addr):
        self.addr = addr


class FakeElf:
    """Just enough of an ELF for detect_gp: an entry point and one code section."""

    def __init__(self, words, entry=BASE):
        self.data = struct.pack("<%dI" % len(words), *words)
        self.e_entry = entry
        self.exec_sections = [FakeSection(BASE)]

    def section_data(self, _section):
        return self.data


def lui(rt, imm): return (0x0F << 26) | (rt << 16) | (imm & 0xFFFF)
def addiu(rt, rs, imm): return (0x09 << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
def move(rd, rs): return (rs << 21) | (rd << 11) | 0x25          # or rd, rs, $zero
def j(target): return (2 << 26) | ((target >> 2) & 0x03FFFFFF)


class TestGlobalPointerDetection(unittest.TestCase):
    def test_built_in_another_register_then_moved(self):
        """Burnout 3's startup: lui $a0, 0x4F; addiu $a0, $a0, -0x7990; or $gp, $a0."""
        elf = FakeElf([lui(4, 0x4F), addiu(4, 4, -0x7990), move(28, 4), 0x0000000C])
        self.assertEqual(game.detect_gp(elf), 0x004E8670)

    def test_found_past_a_loop_that_jumps_backwards(self):
        """The BSS is cleared by loops before $gp is set; the scan must not follow
        them round forever."""
        elf = FakeElf([0, j(BASE), 0, lui(28, 0x50), addiu(28, 28, 0x10), 0x0000000C])
        self.assertEqual(game.detect_gp(elf), 0x00500010)

    def test_none_when_the_entry_never_sets_it(self):
        self.assertIsNone(game.detect_gp(FakeElf([0, 0, 0x03E00008, 0])))


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.saved = vars(layout.CURRENT).copy()

    def tearDown(self):
        layout.configure(**self.saved)

    def test_layout_overrides_come_from_game_json(self):
        directory = tempfile.mkdtemp()
        with open(os.path.join(directory, "game.json"), "w") as handle:
            json.dump({"executable": "NONE.00", "layout": {"scratch": "0x01400000",
                                                          "gp": 0x1234}}, handle)
        loaded = game.load(directory)
        self.assertEqual(loaded.executable, "NONE.00")
        self.assertEqual(layout.CURRENT.scratch, 0x01400000)
        self.assertEqual(layout.CURRENT.gp, 0x1234)
        self.assertEqual(loaded.database_path,
                         os.path.normpath(os.path.join(directory, "build/functions.json")))

    def test_unknown_layout_field_is_refused(self):
        with self.assertRaises(KeyError):
            layout.configure(not_a_field=1)


if __name__ == "__main__":
    unittest.main()
