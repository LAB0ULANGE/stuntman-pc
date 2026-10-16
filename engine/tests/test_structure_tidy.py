"""The text rewrites applied to structured code. Each is checked again on every
function by verification; these pin down the shapes they accept and refuse."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import structure  # noqa: E402


class TestTidy(unittest.TestCase):
    def test_comparison_tested_against_zero_is_the_comparison(self):
        self.assertEqual(structure.tidy("if (((u64)((s64)s1 < (s64)0x6ull)) != 0ull) {"),
                         "if ((s64)s1 < 6) {")
        self.assertEqual(structure.tidy("if (((u64)(a0 < v1)) == 0ull) {"),
                         "if (!(a0 < v1)) {")

    def test_small_sign_extended_literal(self):
        self.assertEqual(structure.tidy("v1 = Sx32(0x9u);"), "v1 = 0x9ull;")
        # 0x80000000 and above change under sign extension.
        self.assertEqual(structure.tidy("v1 = Sx32(0x80000000u);"), "v1 = Sx32(0x80000000u);")

    def test_inner_sign_extension_of_an_addend(self):
        self.assertEqual(structure.tidy("v0 = Sx32(s0 + (Sx32((u32)s1 << 2)));"),
                         "v0 = Sx32(s0 + ((u32)s1 << 2));")
        self.assertEqual(
            structure.tidy("v0 = Sx32(s0 + (Sx32((u32)(Sx32((Sx32((u32)s1 << 3)) - s1)) << 4)));"),
            "v0 = Sx32(s0 + ((u32)(((u32)s1 << 3) - s1) << 4));")

    def test_inner_sign_extension_kept_under_other_operators(self):
        # Under a shift right the high bits reach the low 32; only sums and
        # differences are rewritten.
        line = "v0 = Sx32(s0 >> (Sx32(a0 + a1)));"
        self.assertEqual(structure.tidy(line), line)


class TestFoldConstants(unittest.TestCase):
    def test_two_steps_are_one_constant_named_by_its_string(self):
        self.assertEqual(structure.fold_constants(["    v0 = 0x4B0000ull;", "    v0 = Sx32(v0 + 0x1518u);"],
                                                  {0x4B1518: "exception"}),
                         ["    v0 = 0x4B0000ull;", '    v0 = 0x004B1518ull;  /* "exception" */'])

    def test_lui_alone_is_not_a_string(self):
        # A string beginning where lui's half points is no reason to name it.
        self.assertEqual(structure.fold_constants(["    v0 = 0x4C0000ull;"], {0x4C0000: "Slip Speed"}),
                         ["    v0 = 0x4C0000ull;"])


class TestFoldAddresses(unittest.TestCase):
    def test_memory_address_is_folded(self):
        self.assertEqual(structure.fold_addresses(["    t1 = ReadS32((u32)0x4A0000ull + 0xFFFFB0A0u);"]),
                         ["    t1 = ReadS32(0x0049B0A0u);"])
        self.assertEqual(structure.fold_addresses(["    Write32((u32)0x520000ull + 0xFFFFBAA8u, (u32)v1);"]),
                         ["    Write32(0x0051BAA8u, (u32)v1);"])

    def test_sign_extended_address_is_folded(self):
        self.assertEqual(structure.fold_addresses(["    a0 = Sx32((0x4C0000ull) + 0xFFFF9AD0u);"]),
                         ["    a0 = 0x004B9AD0ull;"])
        # Past 0x7FFFFFFF it is sign-extended, as Sx32 would.
        self.assertEqual(structure.fold_addresses(["    a0 = Sx32((0x80000000ull) + 0x10u);"]),
                         ["    a0 = 0xFFFFFFFF80000010ull;"])

    def test_known_string_is_named(self):
        self.assertEqual(structure.fold_addresses(["    a0 = Sx32((0x4C0000ull) + 0xFFFF9AD0u);"],
                                                  {0x4B9AD0: "dbcman : SendData2"}),
                         ['    a0 = 0x004B9AD0ull;  /* "dbcman : SendData2" */'])
        # The upper half alone (lui's, an offset still to come) is no string's
        # address, even where one begins.
        self.assertEqual(structure.fold_addresses(["    v0 = 0x4C0000ull;"], {0x4C0000: "Slip Speed"}),
                         ["    v0 = 0x4C0000ull;"])

    def test_a_sum_inside_a_larger_expression_is_kept(self):
        # Only a whole argument folds: here the sum is one operand of more.
        line = "    v0 = ReadS32((u32)0x4A0000ull + 0xFFFFB0A0u + a1);"
        self.assertEqual(structure.fold_addresses([line]), [line])
        line = "    v0 = a1 * (u32)0x4A0000ull + 0x10u;"
        self.assertEqual(structure.fold_addresses([line]), [line])


class TestDoWhile(unittest.TestCase):
    LOOP = ["    for (;;) {",
            "        Tick(c, 0x00100000u);",
            "        s1 = Sx32(s1 + 0x1u);",
            "        if ((s64)s1 < 6) {",
            "                continue;",
            "        } else {",
            "                break;",
            "        }",
            "    }"]

    def test_loop_ending_in_its_test(self):
        self.assertEqual(structure.loops_to_do_while(self.LOOP),
                         ["    do {",
                          "        Tick(c, 0x00100000u);",
                          "        s1 = Sx32(s1 + 0x1u);",
                          "    } while ((s64)s1 < 6);"])

    def test_reversed_test_is_negated(self):
        loop = list(self.LOOP)
        loop[4], loop[6] = "                break;", "                continue;"
        self.assertEqual(structure.loops_to_do_while(loop)[-1], "    } while (!((s64)s1 < 6));")

    def test_other_continue_keeps_the_for(self):
        loop = self.LOOP[:2] + ["        if (a0 != 0ull) { continue; }"] + self.LOOP[2:]
        self.assertEqual(structure.loops_to_do_while(loop), loop)


class TestLoopShapes(unittest.TestCase):
    def test_negate(self):
        self.assertEqual(structure.negate("v1 != 0ull"), "v1 == 0ull")
        self.assertEqual(structure.negate("(s64)s1 < 6"), "(s64)s1 >= 6")
        self.assertEqual(structure.negate("!(a && b)"), "a && b")
        self.assertEqual(structure.negate("ReadS32((u32)a0) == 0ull"), "!(ReadS32((u32)a0) == 0ull)")

    def test_independent_delay_slot_is_hoisted(self):
        lines = ["    { const bool t = v1 != 0ull;",
                 "      s2 = Sx32(s2 + s4);",
                 "      if (t) {",
                 "            a0 = s2;",
                 "      } else {",
                 "            a0 = s3;",
                 "      } }"]
        self.assertEqual(structure.hoist_delay_slots(lines),
                         ["    s2 = Sx32(s2 + s4);",
                          "    if (v1 != 0ull) {",
                          "            a0 = s2;",
                          "    } else {",
                          "            a0 = s3;",
                          "    }"])

    def test_dependent_delay_slot_stays(self):
        lines = ["    { const bool t = v1 != 0ull;",
                 "      v1 = Sx32(s2 + s4);",
                 "      if (t) {",
                 "      } }"]
        self.assertEqual(structure.hoist_delay_slots(lines), lines)
        stored = ["    { const bool t = v1 != 0ull;",
                  "      Write32((u32)a0, (u32)v1);",
                  "      if (t) {",
                  "      } }"]
        self.assertEqual(structure.hoist_delay_slots(stored), stored)

    def test_empty_then_is_inverted(self):
        lines = ["    if (at == 0ull) {", "    } else {", "            v1 = 0x1ull;", "    }"]
        self.assertEqual(structure.invert_empty_then(lines),
                         ["    if (at != 0ull) {", "            v1 = 0x1ull;", "    }"])

    def test_guarded_loop_becomes_while(self):
        lines = ["    for (;;) {",
                 "        Tick(c, 0x00101D78u);",
                 "        if (v0 == v1) {",
                 "                a0 = Sx32(a0 + 0x1u);",
                 "                continue;",
                 "        } else {",
                 "                break;",
                 "        }",
                 "    }"]
        self.assertEqual(structure.loop_guards(lines),
                         ["    while (v0 == v1) {",
                          "        Tick(c, 0x00101D78u);",
                          "        a0 = Sx32(a0 + 0x1u);",
                          "    }"])

    def test_guard_after_other_statements_breaks(self):
        lines = ["    for (;;) {",
                 "        Tick(c, 0x00101D78u);",
                 "        v0 = ReadU8((u32)a0);",
                 "        if (v0 == v1) {",
                 "                a0 = Sx32(a0 + 0x1u);",
                 "                continue;",
                 "        } else {",
                 "                break;",
                 "        }",
                 "    }"]
        self.assertEqual(structure.loop_guards(lines),
                         ["    for (;;) {",
                          "        Tick(c, 0x00101D78u);",
                          "        v0 = ReadU8((u32)a0);",
                          "        if (v0 != v1) break;",
                          "        a0 = Sx32(a0 + 0x1u);",
                          "    }"])


class TestLinks(unittest.TestCase):
    def test_link_joins_its_call(self):
        self.assertEqual(structure.fold_links(["    ra = 0x0014ECF4ull;",
                                               "    out(); Call(c, 0x0015E8A0u); in();"]),
                         ["    out(); Call(c, 0x0015E8A0u, 0x0014ECF4u); in();"])

    def test_link_without_a_call_stays(self):
        lines = ["    ra = 0x0014ECF4ull;", "    a0 = s0;", "    out(); Call(c, 0x0015E8A0u); in();"]
        self.assertEqual(structure.fold_links(lines), lines)


class TestSaveSlots(unittest.TestCase):
    PROLOGUE = [0x27BDFFE0, 0x7FB00000]          # addiu $sp, -0x20; sq $s0, 0($sp)
    EPILOGUE = [0x7BB00000, 0x03E00008, 0x27BD0020]  # lq $s0, 0($sp); jr $ra; addiu $sp, 0x20

    def slots(self, body):
        words = self.PROLOGUE + body + self.EPILOGUE
        code = [structure.DISASM.decode(w, 0x100000 + 4 * i) for i, w in enumerate(words)]
        return structure._save_slots(code, 0x100000)

    def test_a_saved_register_written_as_usual_elides(self):
        self.assertEqual(self.slots([0x24100005]), {("s0", 0)})     # addiu $s0, $zero, 5

    def test_a_cop2_move_of_a_saved_register_keeps_the_saves(self):
        # qmfc2 $s0, $vf1 writes $s0 through the context, past the local.
        self.assertIsNone(self.slots([0x48300800]))

    def test_a_128_bit_load_into_a_saved_register_keeps_the_saves(self):
        self.assertIsNone(self.slots([0x78900010]))                 # lq $s0, 0x10($a0)


if __name__ == "__main__":
    unittest.main()
