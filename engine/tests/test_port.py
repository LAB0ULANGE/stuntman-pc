"""The port's generated dispatch table."""
import os
import sys
import tempfile
import unittest

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "tools"),
                os.path.join(os.path.dirname(__file__), "..")]

import port  # noqa: E402


class TestDispatch(unittest.TestCase):
    def test_discoveries_merge_the_build_and_the_port(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            build, committed = os.path.join(d, "build.txt"), os.path.join(d, "port.txt")
            with open(build, "w") as f:
                f.write("0x00100010 0x00100010\n0x80076000 0x00300000\n")
            with open(committed, "w") as f:
                f.write("# found by earlier runs\n0x00100010 0x00100010\n0x00200000 0x00200000\n")
            pairs = port.read_discoveries([committed, build, os.path.join(d, "missing.txt")])
            self.assertEqual(pairs, {(0x00100010, 0x00100010), (0x80076000, 0x00300000),
                                     (0x00200000, 0x00200000)})

    def test_translated_functions_are_found(self):
        folder = tempfile.mkdtemp()
        with open(os.path.join(folder, "lifted_000.cpp"), "w") as handle:
            handle.write("void F_00100010(lifted::Context& c) {\n}\n"
                         "void F_00100008(lifted::Context& c) {\n}\n"
                         "/* void F_00100020(lifted::Context& c) { in a comment */\n")
        self.assertEqual(port.translated_functions(folder), [0x100008, 0x100010])

    def test_table_is_sorted_and_names_what_has_no_translation(self):
        text = port.dispatch_source([0x100010, 0x100008], {0x100008: "_start"}, [0x100004],
                                    {0x100008: 528})
        rows = [line.strip() for line in text.splitlines() if line.strip().startswith("{0x")]
        self.assertEqual(rows, ['{0x00100004u, nullptr, nullptr, 0u},',
                                '{0x00100008u, F_00100008, "_start", 528u},',
                                '{0x00100010u, F_00100010, nullptr, 0u},'])
        self.assertIn("kGuestFunctionCount = 3;", text)
        self.assertIn("void F_00100008(lifted::Context& c);", text)


if __name__ == "__main__":
    unittest.main()
