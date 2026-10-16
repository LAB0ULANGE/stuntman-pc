"""The native equivalence harness renders: its C++ is a %-format template, and a
stray `%` in it once failed a lift forty minutes in, after every verdict."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import equivalence  # noqa: E402


class TestHarness(unittest.TestCase):
    def test_the_driver_and_bodies_render(self):
        body = "void F_00100000(lifted::Context& c) { c.r[2] = 1; }"
        driver, bodies = equivalence.sources({0x00100000: (body, body, 0)})
        self.assertIn("P_00100000", driver)
        self.assertIn("S_00100000", driver)
        self.assertIn("% 9", driver)               # the small-value cases survive formatting
        self.assertEqual(len(bodies), 1)
        self.assertIn("void P_00100000", bodies[0])


if __name__ == "__main__":
    unittest.main()
