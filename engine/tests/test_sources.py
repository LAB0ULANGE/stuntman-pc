"""Hygiene checks on the repository's own sources.

Editing through shell heredocs has more than once turned an escape sequence into
the control byte it names - `\\b` into a backspace, `\\x00` into a NUL - leaving a
file that still parses but means something else. A regex with a backspace in it
silently matches different text. This catches that before it is committed.
"""

import glob
import os
import unittest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CONTROL = set(range(0x20)) - {0x09, 0x0A, 0x0D}


class TestSources(unittest.TestCase):
    def test_no_control_bytes_in_sources(self):
        patterns = ("engine/**/*.py", "engine/**/*.cpp", "engine/**/*.h",
                    "games/*/tools/*.py", "games/*/tests/*.py", "games/*/include/**/*.h",
                    "games/*/src/*/*.cpp")
        offenders = []
        for pattern in patterns:
            for path in glob.glob(os.path.join(REPO, pattern), recursive=True):
                if os.sep + "lifted" + os.sep in path or os.sep + "generated" + os.sep in path:
                    continue
                with open(path, "rb") as handle:
                    data = handle.read()
                bad = sorted({byte for byte in data if byte in CONTROL})
                if bad:
                    offenders.append("%s: %s" % (os.path.relpath(path, REPO),
                                                 ", ".join("0x%02X" % b for b in bad)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
