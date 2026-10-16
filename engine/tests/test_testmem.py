"""Tests for the differential-test memory profiles.

These matter because a profile decides which functions can be tested at all, and a
non-deterministic one would make a failing verification impossible to reproduce.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import layout, testmem


def words(buffer):
    return struct.unpack("<%dI" % (len(buffer) // 4), buffer)


class TestProfiles(unittest.TestCase):
    def test_every_profile_fills_the_whole_region(self):
        for kind in testmem.PROFILES:
            self.assertEqual(len(testmem.build(kind, 0)), layout.CURRENT.scratch_size, kind)

    def test_build_is_deterministic(self):
        """The same profile and seed must always give the same bytes.

        An early version seeded from hash(kind), which Python randomises per process,
        so "deterministic" memory differed on every run.
        """
        for kind in testmem.PROFILES:
            self.assertEqual(testmem.build(kind, 3), testmem._build(kind, 3), kind)

    def test_consecutive_seeds_give_different_memory(self):
        """Each test case needs memory that differs from the previous case's."""
        for kind in ("mixed", "noise", "small", "pointers"):
            self.assertNotEqual(testmem.build(kind, 0), testmem.build(kind, 1), kind)

    def test_pointer_words_stay_inside_the_region(self):
        low = layout.CURRENT.scratch
        high = layout.CURRENT.scratch + layout.CURRENT.scratch_size
        for value in words(testmem.build("pointers", 0)):
            self.assertTrue(low <= value < high)
            self.assertEqual(value % 16, 0)

    def test_small_words_are_small(self):
        """Counts read from this profile must give short loops."""
        self.assertTrue(
            all(v < testmem.SMALL_LIMIT for v in words(testmem.build("small", 0)))
        )

    def test_mixed_alternates_pointers_and_small_values(self):
        values = words(testmem.build("mixed", 0))
        self.assertTrue(all(v >= layout.CURRENT.scratch for v in values[0::2]))
        self.assertTrue(all(v < testmem.SMALL_LIMIT for v in values[1::2]))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            testmem._build("nonsense", 0)

    def test_integer_arguments_point_into_the_first_half(self):
        import random

        for value in testmem.arguments(random.Random(1))[:8]:
            self.assertTrue(
                layout.CURRENT.scratch <= value < layout.CURRENT.scratch + layout.CURRENT.scratch_size // 2
            )

    def test_float_arguments_are_ordinary_floats(self):
        """Random bits would be mostly NaNs and denormals; these must be real values."""
        import math
        import random

        for bits in testmem.arguments(random.Random(1))[8:]:
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            self.assertTrue(math.isfinite(value) and abs(value) <= 1000.0)

    def test_first_four_arguments_are_unchanged(self):
        """Earlier functions keep the exact cases they were verified on."""
        import random

        rng = random.Random(7)
        before = tuple(layout.CURRENT.scratch + rng.randrange(0, layout.CURRENT.scratch_size // 2, 16)
                       for _ in range(4))
        self.assertEqual(testmem.arguments(random.Random(7))[:4], before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
