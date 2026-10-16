"""Tests for symbolic derivation of straight-line functions.

The precedence tests matter most.  An expression tree evaluates correctly
regardless of how it prints, so a rendering bug produces C that silently disagrees
with the tree it came from - which is exactly the failure this module exists to
avoid.  `arg << 2 + base` is not `(arg << 2) + base` in C.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.symexec import (
    Arg,
    Binary,
    Compare,
    Const,
    FloatPredicate,
    Load,
    Predicate,
    SignExtend32,
    Unsupported,
    derive,
    low32,
    simplify,
)

BASE = 0x00100000


def assemble(words) -> bytes:
    return struct.pack("<%dI" % len(words), *words)


def enc_i(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def enc_r(op, rs, rt, rd, sa, funct):
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def enc_j(op, target):
    """J-type: the target is a word address, not a byte address."""
    return (op << 26) | (target & 0x03FFFFFF)


JR_RA = enc_r(0, 31, 0, 0, 0, 0x08)


class TestRenderingPrecedence(unittest.TestCase):
    def test_shift_inside_addition_is_parenthesised(self):
        """`(a << 2) + b` must not print as `a << 2 + b`.

        In C, `+` binds tighter than `<<`, so the unparenthesised form means
        `a << (2 + b)` - a different expression entirely.
        """
        expr = Binary("add", Binary("shl", Arg(1), Const(2)), Arg(0))
        self.assertEqual(expr.render(), "(arg1 << 2) + arg0")

    def test_arithmetic_inside_shift_is_parenthesised_for_clarity(self):
        """`a + 4 << 2` is valid C meaning `(a + 4) << 2`, but reads wrong.

        Precedence does not require the parentheses here - `+` binds tighter than
        `<<` - so these are deliberately redundant. Redundant parentheses cannot
        change meaning, whereas a misreading can, and this form appears constantly
        in derived index arithmetic.
        """
        expr = Binary("shl", Binary("add", Arg(0), Const(4)), Const(2))
        self.assertEqual(expr.render(), "(arg0 + 4) << 2")

    def test_nested_shift_inside_subtraction_inside_shift(self):
        """The case that exposed a bad guard.

        An earlier version skipped wrapping when the rendered left operand already
        started with "(" - but `(a << 3) - b` starts with a parenthesis without
        being parenthesised as a whole, so the outer shift was left ambiguous.
        """
        inner = Binary("sub", Binary("shl", Arg(0), Const(3)), Arg(1))
        expr = Binary("shl", inner, Const(4))
        self.assertEqual(expr.render(), "((arg0 << 3) - arg1) << 4")

    def test_right_operand_of_subtraction_is_parenthesised(self):
        """`a - (b - c)` must keep its parentheses, or it becomes `a - b - c`."""
        expr = Binary("sub", Arg(0), Binary("sub", Arg(1), Arg(2)))
        self.assertEqual(expr.render(), "arg0 - (arg1 - arg2)")

    def test_or_of_ands_needs_no_parentheses(self):
        # `&` binds tighter than `|`.
        expr = Binary(
            "or",
            Binary("and", Arg(0), Const(0xFF)),
            Binary("and", Arg(1), Const(0xFF00)),
        )
        self.assertEqual(expr.render(), "arg0 & 255 | arg1 & 0xFF00")

    def test_and_of_ors_is_parenthesised(self):
        expr = Binary(
            "and", Binary("or", Arg(0), Arg(1)), Binary("or", Arg(2), Arg(3))
        )
        self.assertEqual(expr.render(), "(arg0 | arg1) & (arg2 | arg3)")


class TestCppValueSemantics(unittest.TestCase):
    """The rendering must mean, in C++, what the tree means.

    Every value is a u64 register image. These cover the operators whose C++
    meaning depends on operand types, each of which the emitter once got wrong in
    a way that compiled cleanly. The end-to-end check is tools/ps2decomp/cppcheck.py,
    which compiles the output and runs it; these pin the individual rules.
    """

    def test_signed_comparison_casts_operands(self):
        expr = Predicate("lt", Arg(0), Arg(1))
        self.assertEqual(expr.render(), "(s64)arg0 < (s64)arg1")

    def test_unsigned_comparison_has_no_casts(self):
        self.assertEqual(Predicate("ltu", Arg(0), Arg(1)).render(), "arg0 < arg1")

    def test_comparison_of_nonnegative_values_needs_no_cast(self):
        expr = Predicate("lt", Load(Arg(0), 1, False), Const(10))
        self.assertEqual(expr.render(), "*(u8*)(arg0) < 10")

    def test_negative_constant_in_signed_comparison_is_decimal(self):
        """`-0x80000000` is unsigned in C++; the decimal spelling is not."""
        expr = Predicate("ge", Arg(0), Const((-0x80000000) & ((1 << 64) - 1)))
        self.assertEqual(expr.render(), "(s64)arg0 >= -2147483648")

    def test_unsigned_compare_stays_unsigned_through_simplify(self):
        """`sltu` tested against zero must not become a signed `<`."""
        expr = simplify(Predicate("ne", Compare("ltu", Arg(0), Arg(1)), Const(0)))
        self.assertEqual(expr.op, "ltu")
        big = 0xFFFFFFFFFFFFFFFF
        self.assertTrue(expr.evaluate((1, big, 0, 0), None))
        self.assertEqual(expr.negate().op, "geu")

    def test_word_arithmetic_shift_narrows_to_s32(self):
        expr = Binary("sar", Arg(0), Const(3))
        self.assertEqual(expr.render(), "(u64)((s32)arg0 >> 3)")

    def test_logical_word_shift_masks_upper_half(self):
        self.assertEqual(Binary("shr", Arg(0), Const(4)).render(),
                         "(arg0 & 0xFFFFFFFF) >> 4")
        # An operand already below 2**32 needs no mask.
        self.assertEqual(Binary("shr", Load(Arg(0), 2, False), Const(4)).render(),
                         "*(u16*)(arg0) >> 4")

    def test_variable_shift_count_is_masked(self):
        self.assertEqual(Binary("shl", Arg(0), Arg(1)).render(), "arg0 << (arg1 & 31)")
        self.assertEqual(Binary("dshl", Arg(0), Arg(1)).render(), "arg0 << (arg1 & 63)")

    def test_shifted_literal_is_64_bit(self):
        """`1 << 31` on an int overflows; the literal must be a u64."""
        self.assertEqual(Binary("shl", Const(1), Arg(1)).render(), "1ull << (arg1 & 31)")

    def test_sign_extension_widens_back_to_u64(self):
        self.assertEqual(SignExtend32(Arg(0)).render(), "(u64)(s32)arg0")

    def test_set_less_than_is_a_u64(self):
        self.assertEqual(Compare("ltu", Arg(0), Arg(1)).render(), "(u64)(arg0 < arg1)")

    def test_float_constant_renders_as_exact_literal(self):
        minus_one = FloatPredicate("eq", Const(0xBF800000), Arg(0))
        self.assertEqual(minus_one.render(), "FloatValue(FloatBits(-1.0f)) == FloatValue(arg0)")
        # A value with no short exact decimal still round-trips.
        odd = FloatPredicate("eq", Const(0x3DCCCCCD), Arg(0))
        literal = odd.render().split("FloatBits(")[1].split("f)")[0]
        self.assertEqual(struct.unpack("<I", struct.pack("<f", float(literal)))[0], 0x3DCCCCCD)


class TestReroll(unittest.TestCase):
    """Unrolled element writes fold back into loops, and nothing else does."""

    def setUp(self):
        from derive import reroll

        self.reroll = reroll

    def test_repeated_block_becomes_a_loop(self):
        statements = []
        for i in range(4):
            statements += ["a.set_f_0(%d, 0);" % i, "a.set_f_40(%d, 0);" % i]
        self.assertEqual(self.reroll(statements), [
            "for (u64 i = 0; i < 4; i++) {",
            "    a.set_f_0(i, 0);",
            "    a.set_f_40(i, 0);",
            "}",
        ])

    def test_statements_keep_their_own_starting_index(self):
        statements = []
        for i in range(3):
            statements += ["a.set_f_10(%d, 7);" % i, "a.set_f_10(%d, 7);" % (i + 20)]
        self.assertEqual(self.reroll(statements)[1:3],
                         ["    a.set_f_10(i, 7);", "    a.set_f_10(i + 20, 7);"])

    def test_step_other_than_one(self):
        statements = ["set_g_00670BD8(%d, 0);" % i for i in (0, 2, 4, 6)]
        self.assertEqual(self.reroll(statements), [
            "for (u64 i = 0; i < 4; i++) {",
            "    set_g_00670BD8(2 * i, 0);",
            "}",
        ])

    def test_differing_values_are_left_alone(self):
        statements = ["a.set_f_0(%d, %d);" % (i, i) for i in range(5)]
        self.assertEqual(self.reroll(statements), statements)

    def test_two_iterations_are_not_a_loop(self):
        statements = ["a.set_f_0(0, 0);", "a.set_f_0(1, 0);", "x = 1;"]
        self.assertEqual(self.reroll(statements), statements)


class TestWordSignExtension(unittest.TestCase):
    """Word operations sign-extend their 32-bit result into the 64-bit register.

    The model used to keep the full 64-bit sum, which only shows where upper bits
    are observed - a comparison - and was found when an `sllv` result reached an
    `slt` with arguments other than small pointers.
    """

    def test_word_add_overflow_is_sign_extended(self):
        from ps2decomp.symexec import SignExtend32

        expr = simplify(SignExtend32(Binary("add", Arg(0), Const(1))))
        self.assertEqual(expr.evaluate((0x7FFFFFFF,) + (0,) * 15, None),
                         0xFFFFFFFF80000000)

    def test_extension_is_dropped_where_only_low_bits_are_used(self):
        from ps2decomp.symexec import SignExtend32, address_of

        inner = SignExtend32(Binary("add", Arg(0), Const(0x1E4)))
        address = address_of(Binary("add", inner, Const(0x20)))
        self.assertEqual(address.render(), "arg0 + 0x204")

    def test_extension_is_dropped_when_the_value_fits(self):
        from ps2decomp.symexec import SignExtend32

        byte_plus_one = Binary("add", Load(Arg(0), 1, False), Const(1))
        self.assertEqual(simplify(SignExtend32(byte_plus_one)), byte_plus_one)

    def test_extension_is_kept_in_a_comparison(self):
        from ps2decomp.symexec import Predicate, SignExtend32

        compare = simplify(Predicate("lt", SignExtend32(Binary("add", Arg(0), Arg(1))),
                                     Const(0)))
        self.assertEqual(compare.render(), "(s32)(arg0 + arg1) < 0")

    def test_unrolled_counter_folds(self):
        """sext(sext(i + 1) + 1) must become i + 2, or unrolled loops grow."""
        from ps2decomp.symexec import SignExtend32

        i = Load(Arg(0), 1, False)
        twice = SignExtend32(Binary("add", SignExtend32(Binary("add", i, Const(1))), Const(1)))
        self.assertEqual(simplify(twice).render(), "*(u8*)(arg0) + 2")


class TestEvaluation(unittest.TestCase):
    def _read(self, addr, size, signed):
        # A predictable stand-in: each address reads back as itself.
        value = addr & ((1 << (size * 8)) - 1)
        if signed and value & (1 << (size * 8 - 1)):
            value -= 1 << (size * 8)
        return value

    def test_evaluation_follows_the_tree_not_the_text(self):
        expr = Binary("add", Binary("shl", Arg(1), Const(2)), Arg(0))
        # (3 << 2) + 100 = 112
        self.assertEqual(expr.evaluate((100, 3, 0, 0), self._read), 112)

    def test_load_uses_the_reader(self):
        expr = Load(Binary("add", Arg(0), Const(8)), 4, False)
        self.assertEqual(expr.evaluate((0x1000, 0, 0, 0), self._read), 0x1008)

    def test_compare_is_unsigned_for_ltu(self):
        expr = Compare("ltu", Arg(0), Arg(1))
        self.assertEqual(expr.evaluate((1, 2, 0, 0), self._read), 1)
        self.assertEqual(expr.evaluate((2, 1, 0, 0), self._read), 0)


class TestSimplify(unittest.TestCase):
    def test_folds_constants(self):
        self.assertEqual(simplify(Binary("add", Const(2), Const(3))).render(), "5")

    def test_drops_additive_identity(self):
        self.assertEqual(simplify(Binary("add", Arg(0), Const(0))).render(), "arg0")

    def test_keeps_real_work(self):
        expr = simplify(Binary("add", Arg(0), Const(8)))
        self.assertEqual(expr.render(), "arg0 + 8")


class TestDerive(unittest.TestCase):
    def test_simple_field_load(self):
        # lw $v0, 8($a0) ; jr $ra ; nop
        words = [enc_i(0x23, 4, 2, 8), JR_RA, 0]
        image = assemble(words)
        result = derive(image, BASE, BASE, BASE + len(image))
        self.assertEqual(simplify(result.expression).render(), "*(s32*)(arg0 + 8)")

    def test_indexed_load_keeps_precedence(self):
        # sll $v0, $a1, 2 ; addu $v0, $v0, $a0 ; jr $ra ; lw $v0, 8($v0)
        words = [
            enc_r(0, 0, 5, 2, 2, 0x00),
            enc_r(0, 2, 4, 2, 0, 0x21),
            JR_RA,
            enc_i(0x23, 2, 2, 8),
        ]
        image = assemble(words)
        result = derive(image, BASE, BASE, BASE + len(image))
        rendered = simplify(result.expression).render()
        # The shift must stay grouped, or the C means something else.
        self.assertEqual(rendered, "*(s32*)((arg1 << 2) + arg0 + 8)")

    def test_delay_slot_after_jr_is_included(self):
        """The instruction in the return's delay slot still executes."""
        words = [JR_RA, enc_i(0x09, 4, 2, 5)]  # jr $ra ; addiu $v0, $a0, 5
        image = assemble(words)
        result = derive(image, BASE, BASE, BASE + len(image))
        # addiu sign-extends its result; only the low 32 bits are returned.
        self.assertEqual(simplify(low32(simplify(result.expression))).render(), "arg0 + 5")

    def test_refuses_a_branch(self):
        words = [enc_i(0x04, 4, 5, 2), 0, JR_RA, 0]  # beq
        with self.assertRaises(Unsupported):
            derive(assemble(words), BASE, BASE, BASE + 16)

    def test_refuses_a_store(self):
        words = [enc_i(0x2B, 4, 5, 0), JR_RA, 0]  # sw
        with self.assertRaises(Unsupported):
            derive(assemble(words), BASE, BASE, BASE + 12)

    def test_refuses_an_uninitialised_register(self):
        """Reading $s0 without setting it means the frame matters; refuse it."""
        words = [enc_r(0, 16, 0, 2, 0, 0x21), JR_RA, 0]  # addu $v0, $s0, $zero
        with self.assertRaises(Unsupported):
            derive(assemble(words), BASE, BASE, BASE + 12)




class TestEmitterSelfCheck(unittest.TestCase):
    """The generated source must not reference undeclared parameters.

    An earlier version of the emitter built each signature from the return
    expression only, so arguments used solely inside a store went undeclared and
    the output could not compile. Nothing caught it until a human read the file,
    which is not a check.
    """

    def setUp(self):
        from derive import check_signatures

        self.check = check_signatures

    def test_rejects_an_undeclared_argument(self):
        bad = "s32 func_00ABCDEF(u32 arg0) {\n    return arg1 + arg0;\n}\n"
        with self.assertRaises(SystemExit):
            self.check(bad)

    def test_accepts_a_complete_signature(self):
        good = "s32 func_00ABCDEF(u32 arg0, u32 arg1) {\n    return arg1 + arg0;\n}\n"
        self.check(good)

    def test_accepts_a_function_with_no_arguments(self):
        self.check("s32 func_00ABCDEF(void) {\n    return 1;\n}\n")

    def test_checks_store_bodies_too(self):
        bad = (
            "s32 func_00ABCDEF(u32 arg0) {\n"
            "    const u32 addr0 = arg0 + 8;\n"
            "    const u64 val0 = arg2;\n"
            "    Write32(addr0, val0);\n"
            "    return 0;\n"
            "}\n"
        )
        with self.assertRaises(SystemExit):
            self.check(bad)

class TestRendererIsNotDuplicated(unittest.TestCase):
    """The emitter must reuse the one renderer, not keep a copy of it.

    derive.py originally reimplemented the precedence logic so it could print
    Read* helper calls instead of pointer casts. The two copies drifted: a clarity
    rule added to symexec never reached the generated source, so the emitted C
    differed from what these tests were checking. The Load form is now the only
    thing the emitter overrides, and this pins that down.
    """

    def setUp(self):
        from derive import render_cpp

        self.render_cpp = render_cpp

    def _sample(self):
        # Deliberately exercises the clarity rule and a nested load.
        inner = Binary("sub", Binary("shl", Arg(0), Const(3)), Arg(1))
        return Load(Binary("shl", inner, Const(2)), 4, True)

    def test_only_the_load_form_differs(self):
        expr = self._sample()
        default = expr.render()
        emitted = self.render_cpp(expr)
        self.assertIn("*(s32*)", default)
        self.assertIn("ReadS32", emitted)
        # Strip the differing load syntax; the remaining structure must be identical.
        self.assertEqual(
            default.replace("*(s32*)(", "<L>").replace(")", "", 1),
            emitted.replace("ReadS32(", "<L>").replace(")", "", 1),
        )

    def test_emitter_keeps_the_clarity_parentheses(self):
        """The rule added in symexec has to show up in the generated source."""
        emitted = self.render_cpp(self._sample())
        self.assertIn("((arg0 << 3) - arg1) << 2", emitted)

    def test_load_renderer_is_restored_afterwards(self):
        """The hook must not leak, or later default rendering changes silently."""
        expr = self._sample()
        self.render_cpp(expr)
        self.assertIn("*(s32*)", expr.render())

class TestFloatModelling(unittest.TestCase):
    """Floats are modelled as bit patterns, matching the hardware registers.

    The interpreter and the deriver must agree exactly on float semantics, since
    they are compared against each other on every derivation. Two Emotion Engine
    behaviours are deliberate and are pinned here.
    """

    def setUp(self):
        from ps2decomp.symexec import FloatOp, FloatPredicate, IntToFloat, FloatToInt
        from ps2decomp.symexec import _bits_to_float, _float_to_bits

        self.FloatOp = FloatOp
        self.FloatPredicate = FloatPredicate
        self.IntToFloat = IntToFloat
        self.FloatToInt = FloatToInt
        self.to_float = _bits_to_float
        self.to_bits = _float_to_bits

    def _read(self, addr, size, signed):
        return 0

    def test_arithmetic_round_trips_through_bits(self):
        two = Const(self.to_bits(2.0))
        three = Const(self.to_bits(3.0))
        result = self.FloatOp("add", two, three).evaluate((0, 0, 0, 0), self._read)
        self.assertEqual(self.to_float(result), 5.0)

    def test_overflow_saturates_rather_than_becoming_infinite(self):
        """The EE's FPU has no infinities: it saturates to the largest magnitude.

        Python computes in double precision, so an overflowing multiply would
        otherwise raise when packed back to 32 bits - which crashed the interpreter
        outright on real game code.
        """
        from ps2decomp.symexec import FLOAT_MAX

        big = Const(self.to_bits(3.0e38))
        result = self.FloatOp("mul", big, Const(self.to_bits(10.0)))
        self.assertEqual(self.to_float(result.evaluate((0, 0, 0, 0), self._read)),
                         FLOAT_MAX)

    def test_negative_overflow_saturates(self):
        from ps2decomp.symexec import FLOAT_MAX

        big = Const(self.to_bits(-3.0e38))
        result = self.FloatOp("mul", big, Const(self.to_bits(10.0)))
        self.assertEqual(self.to_float(result.evaluate((0, 0, 0, 0), self._read)),
                         -FLOAT_MAX)

    def test_division_by_zero_yields_zero(self):
        """The FPU flags this rather than trapping."""
        result = self.FloatOp("div", Const(self.to_bits(1.0)), Const(self.to_bits(0.0)))
        self.assertEqual(self.to_float(result.evaluate((0, 0, 0, 0), self._read)), 0.0)

    def test_conversions(self):
        to_float = self.IntToFloat(Const(7)).evaluate((0, 0, 0, 0), self._read)
        self.assertEqual(self.to_float(to_float), 7.0)
        back = self.FloatToInt(Const(self.to_bits(-3.75))).evaluate(
            (0, 0, 0, 0), self._read
        )
        # Truncation toward zero, not rounding.
        self.assertEqual(back & 0xFFFFFFFF, (-3) & 0xFFFFFFFF)

    def test_comparison_and_negation(self):
        one = Const(self.to_bits(1.0))
        two = Const(self.to_bits(2.0))
        predicate = self.FloatPredicate("lt", one, two)
        self.assertTrue(predicate.evaluate((0, 0, 0, 0), self._read))
        self.assertFalse(predicate.negate().evaluate((0, 0, 0, 0), self._read))

    def test_float_nodes_are_traversable(self):
        """A new node type must be visible to the shared traversal.

        Adding nodes without updating it is what previously made arguments nested
        inside them invisible to the emitter.
        """
        from ps2decomp.symexec import arguments_used

        expr = self.FloatOp("mul", Load(Arg(1), 4, False), Const(0))
        self.assertEqual(arguments_used(expr), {1})

class TestVoidFunctions(unittest.TestCase):
    """A function that never writes $v0 returns nothing.

    Insisting on a return value refused 514 functions outright - every setter and
    initialiser in the binary - because the deriver treated a missing $v0 as a
    missing register rather than as a void signature.
    """

    def setUp(self):
        from ps2decomp.paths import derive_paths

        self.derive_paths = derive_paths

    def _derive(self, words):
        image = assemble(words)
        return self.derive_paths(image, BASE, BASE, BASE + len(image))

    def test_store_only_function_is_void(self):
        # sw $a1, 0($a0) ; jr $ra ; nop
        result = self._derive([enc_i(0x2B, 4, 5, 0), JR_RA, 0])
        self.assertTrue(result.is_void)
        self.assertEqual(len(result.paths[0].stores), 1)

    def test_void_function_renders_without_a_return(self):
        result = self._derive([enc_i(0x2B, 4, 5, 0), JR_RA, 0])
        self.assertNotIn("return", result.render())

    def test_void_evaluation_yields_none(self):
        """The verifier relies on None to know not to compare $v0."""
        result = self._derive([enc_i(0x2B, 4, 5, 0), JR_RA, 0])
        self.assertIsNone(result.evaluate((0, 0, 0, 0), lambda a, s, g: 0))

    def test_value_returning_function_is_not_void(self):
        # lw $v0, 8($a0) ; jr $ra ; nop
        result = self._derive([enc_i(0x23, 4, 2, 8), JR_RA, 0])
        self.assertFalse(result.is_void)
        self.assertIn("return", result.render())

    def test_mixed_void_and_value_paths_are_refused(self):
        """One route returning a value and another not is ambiguous, so refuse it.

        Choosing either reading would be a guess about the signature.
        """
        # beq $a0,$zero,+2 ; nop ; jr $ra ; nop   (taken path leaves $v0 unset)
        # followed by a path that sets $v0, reached by falling through.
        words = [
            enc_i(0x04, 4, 0, 3),        # beq $a0, $zero, -> skip the addiu
            0,                            # delay slot
            enc_i(0x09, 0, 2, 7),        # addiu $v0, $zero, 7
            JR_RA,
            0,
            JR_RA,                        # the branch target: returns with $v0 unset
            0,
        ]
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            self._derive(words)

    def test_branching_void_paths_return_and_capture(self):
        """Each non-final void path must leave, and capture before it writes.

        The paths are emitted as a run of `if` blocks. Without a `return;` an
        early path fell through into the later paths' stores, and without the
        capture a second store re-read memory the first had just overwritten.
        Both compiled cleanly; running the compiled code caught them.
        """
        from derive import render_body
        from ps2decomp.paths import Path, PathDerivation
        from ps2decomp.symexec import Predicate, Store

        first = Path(
            (Predicate("eq", Arg(1), Const(0)),), None,
            (Store(Arg(0), Const(1), 4), Store(Arg(2), Load(Arg(0), 4, False), 4)),
        )
        rest = Path((Predicate("ne", Arg(1), Const(0)),), None, (Store(Arg(0), Const(2), 4),))
        lines = render_body(PathDerivation(0, (first, rest)))
        block = lines[: lines.index("}")]
        self.assertIn("    return;", block)
        captured = [i for i, line in enumerate(block) if "const u64" in line]
        written = [i for i, line in enumerate(block) if "Write32(" in line]
        self.assertLess(max(captured), min(written))


class TestInterproceduralInlining(unittest.TestCase):
    """Calls are inlined by substituting the caller's arguments into the callee.

    Restricting this to single-path callees is why an earlier version gained almost
    nothing: most callees have a conditional in them.
    """

    def setUp(self):
        from ps2decomp.paths import derive_paths

        self.derive_paths = derive_paths

    def _derive(self, words, known=None, start_offset=0):
        image = assemble(words)
        return self.derive_paths(
            image, BASE, BASE + start_offset, BASE + len(image), known=known
        )

    def _callee_returning(self, value):
        """A callee that is `return argN + value;` for arg0."""
        words = [enc_i(0x09, 4, 2, value), JR_RA, 0]   # addiu $v0, $a0, value
        return self.derive_paths(assemble(words), BASE, BASE, BASE + 12)

    def test_call_to_underived_function_is_refused(self):
        from ps2decomp.symexec import Unsupported

        # jal 0x00100100 ; nop ; jr $ra ; nop
        words = [enc_j(0x03, (BASE + 0x100) >> 2), 0, JR_RA, 0]
        with self.assertRaises(Unsupported):
            self._derive(words, known={})

    def test_single_path_callee_is_substituted(self):
        callee = self._callee_returning(5)
        # addiu $a0, $zero, 10 ; jal callee ; nop ; jr $ra ; nop
        target = BASE + 0x40
        words = [
            enc_i(0x09, 0, 4, 10),
            enc_j(0x03, target >> 2),
            0,
            JR_RA,
            0,
        ]
        result = self._derive(words, known={target: callee})
        # 10 + 5, folded by simplification.
        self.assertEqual(simplify(result.paths[0].expression).render(), "15")

    def test_multi_path_callee_multiplies_caller_paths(self):
        """A conditional callee yields one caller path per callee route."""
        # Callee: if (arg0 == 0) return 1; return 2;
        #
        # The branch offset is counted from the delay slot, so +4 lands on index 5.
        # An earlier version used +2, which landed on the `jr $ra` at index 3 with
        # $v0 never written - and the "inconsistent void" guard correctly refused it.
        callee_words = [
            enc_i(0x04, 4, 0, 4),       # beq $a0, $zero, -> the "1" arm (index 5)
            0,                           # delay slot
            enc_i(0x09, 0, 2, 2),       # addiu $v0, $zero, 2
            JR_RA,
            0,
            enc_i(0x09, 0, 2, 1),       # addiu $v0, $zero, 1
            JR_RA,
            0,
        ]
        callee = self.derive_paths(
            assemble(callee_words), BASE, BASE, BASE + len(callee_words) * 4
        )
        self.assertGreater(len(callee.paths), 1)

        target = BASE + 0x80
        words = [enc_j(0x03, target >> 2), 0, JR_RA, 0]
        result = self._derive(words, known={target: callee})
        self.assertEqual(len(result.paths), len(callee.paths))

    def test_caller_saved_registers_are_dropped_after_a_call(self):
        """Reading a temp after a call must be refused, not served a stale value."""
        from ps2decomp.symexec import Unsupported

        callee = self._callee_returning(1)
        target = BASE + 0x40
        words = [
            enc_i(0x09, 0, 8, 7),        # addiu $t0, $zero, 7
            enc_j(0x03, target >> 2),    # jal callee   (clobbers $t0)
            0,
            enc_r(0, 8, 0, 2, 0, 0x21),  # addu $v0, $t0, $zero  <- stale read
            JR_RA,
            0,
        ]
        with self.assertRaises(Unsupported):
            self._derive(words, known={target: callee})

class TestVisitCounting(unittest.TestCase):
    """The interpreter can count how often each address executes.

    Loop analysis depends on it: the number of times a loop header is reached *is*
    the trip count, so counting visits answers "is this trip count constant?"
    directly rather than by static reasoning.

    Counting is off by default because the derivation pipeline runs millions of
    instructions and would pay for something it never reads.
    """

    def _machine(self, words):
        from ps2decomp.emu import Machine

        image = assemble(words)
        # Room for the stack the interpreter sets up above the image.
        return Machine(BASE, image, 0x02000000 - BASE)

    def test_counting_is_off_by_default(self):
        machine = self._machine([JR_RA, 0])
        machine.call(BASE, (0, 0, 0, 0))
        self.assertFalse(machine.count_visits)
        self.assertEqual(machine.visit_counts, {})

    def test_counts_each_instruction_once_in_straight_line_code(self):
        # addiu $v0, $zero, 1 ; jr $ra ; nop
        words = [enc_i(0x09, 0, 2, 1), JR_RA, 0]
        machine = self._machine(words)
        machine.count_visits = True
        machine.call(BASE, (0, 0, 0, 0))
        self.assertEqual(machine.visit_counts.get(BASE), 1)

    def test_loop_header_count_is_the_trip_count(self):
        """A counted loop's header should be reached once per iteration."""
        # addiu $t0, $zero, 3        ; counter = 3
        # loop: addiu $t0, $t0, -1   ; counter -= 1
        #       bne $t0, $zero, loop
        #       nop
        #       jr $ra
        #       nop
        loop = BASE + 4
        words = [
            enc_i(0x09, 0, 8, 3),
            enc_i(0x09, 8, 8, -1),
            enc_i(0x05, 8, 0, -2),     # bne $t0, $zero, loop
            0,
            JR_RA,
            0,
        ]
        machine = self._machine(words)
        machine.count_visits = True
        machine.call(BASE, (0, 0, 0, 0))
        self.assertEqual(machine.visit_counts.get(loop), 3)

class TestLoopUnrolling(unittest.TestCase):
    """Loops with a constant trip count unroll themselves.

    No trip count is supplied: the exit condition folds to a constant on every
    iteration, so the walk follows the one feasible side round the loop and out. A
    loop whose count depends on input forks on every iteration and is refused.
    """

    def setUp(self):
        from ps2decomp.paths import derive_paths
        from ps2decomp.symexec import Unsupported

        self.derive_paths = derive_paths
        self.Unsupported = Unsupported

    def _derive(self, words):
        image = assemble(words)
        return self.derive_paths(image, BASE, BASE, BASE + len(image))

    def test_constant_count_loop_unrolls(self):
        # v0 = 0; t0 = 3; do { v0 += a0; t0 -= 1; } while (t0 != 0); return v0;
        words = [
            enc_i(0x09, 0, 2, 0),            # addiu $v0, $zero, 0
            enc_i(0x09, 0, 8, 3),            # addiu $t0, $zero, 3
            enc_r(0, 2, 4, 2, 0, 0x21),      # loop: addu $v0, $v0, $a0
            enc_i(0x09, 8, 8, -1),           #       addiu $t0, $t0, -1
            enc_i(0x05, 8, 0, -3),           #       bne $t0, $zero, loop
            0,
            JR_RA,
            0,
        ]
        result = self._derive(words)
        self.assertEqual(len(result.paths), 1, "a constant loop must not fork")
        # Three iterations of `v0 += a0`.
        self.assertEqual(result.evaluate((5, 0, 0, 0), lambda a, s, g: 0), 15)

    def test_input_dependent_loop_is_refused(self):
        """The count comes from arg0, so no finite expression describes it."""
        # do { a0 -= 1; } while (a0 != 0); return 0;
        words = [
            enc_i(0x09, 4, 4, -1),           # loop: addiu $a0, $a0, -1
            enc_i(0x05, 4, 0, -2),           #       bne $a0, $zero, loop
            0,
            JR_RA,
            enc_i(0x09, 0, 2, 0),            # delay: addiu $v0, $zero, 0
        ]
        with self.assertRaises(self.Unsupported):
            self._derive(words)

    def test_over_long_constant_loop_is_refused(self):
        """A trip count beyond MAX_UNROLL is refused rather than unrolled."""
        from ps2decomp.paths import MAX_UNROLL

        words = [
            enc_i(0x09, 0, 8, MAX_UNROLL + 5),
            enc_i(0x09, 8, 8, -1),
            enc_i(0x05, 8, 0, -2),
            0,
            JR_RA,
            0,
        ]
        with self.assertRaises(self.Unsupported):
            self._derive(words)

    def test_doubling_expression_is_refused_quickly(self):
        """A value used twice per iteration doubles the tree every time round.

        This is the shape that hung derivation outright: a ring-buffer walk at
        0x0020DD08 built a tree growing as 2^iterations. It must now be refused on
        size, and promptly.
        """
        import time

        # v0 = a0; t0 = 40; do { v0 = v0 + v0; t0 -= 1; } while (t0 != 0);
        words = [
            enc_r(0, 4, 0, 2, 0, 0x21),      # addu $v0, $a0, $zero
            enc_i(0x09, 0, 8, 40),           # addiu $t0, $zero, 40
            enc_r(0, 2, 2, 2, 0, 0x21),      # loop: addu $v0, $v0, $v0
            enc_i(0x09, 8, 8, -1),
            enc_i(0x05, 8, 0, -3),
            0,
            JR_RA,
            0,
        ]
        started = time.time()
        with self.assertRaises(self.Unsupported):
            self._derive(words)
        self.assertLess(time.time() - started, 2.0)


class TestTreeSize(unittest.TestCase):
    def test_sizes_are_recorded_at_construction(self):
        self.assertEqual(Arg(0).tree_size, 1)
        self.assertEqual(Binary("add", Arg(0), Const(4)).tree_size, 3)

    def test_shared_subtrees_count_as_a_tree(self):
        """Sharing keeps the graph small but the printed form is a full tree."""
        x = Arg(0)
        for _ in range(10):
            x = Binary("add", x, x)
        self.assertEqual(x.tree_size, 2 ** 11 - 1)

    def test_load_width_field_is_unaffected(self):
        """Load's own `size` field (the access width) must not be shadowed."""
        self.assertEqual(Load(Arg(0), 4, False).size, 4)

class TestAliasing(unittest.TestCase):
    """Whether a load can be answered after earlier writes.

    Addresses are compared as base + offset, with widths taken into account and
    writes examined newest first.
    """

    def setUp(self):
        from ps2decomp import paths

        self.paths = paths

    def rel(self, a, sa, b, sb):
        return self.paths._relation(a, sa, b, sb)

    def test_same_base_different_offsets_are_disjoint(self):
        self.assertEqual(
            self.rel(Binary("add", Arg(0), Const(4)), 4,
                     Binary("add", Arg(0), Const(8)), 4),
            "disjoint",
        )

    def test_same_base_same_offset_same_width_is_the_same_location(self):
        a = Binary("add", Arg(0), Const(4))
        self.assertEqual(self.rel(a, 4, a, 4), "same")

    def test_widths_matter_with_constant_addresses(self):
        """A 4-byte write at 0x100 covers 0x102.

        The earlier check compared start addresses only and called these
        independent.
        """
        self.assertEqual(self.rel(Const(0x100), 4, Const(0x102), 1), "overlap")
        self.assertEqual(self.rel(Const(0x100), 4, Const(0x104), 4), "disjoint")

    def test_widths_matter_with_a_shared_base(self):
        self.assertEqual(
            self.rel(Arg(0), 4, Binary("add", Arg(0), Const(2)), 2), "overlap"
        )

    def test_different_bases_are_unknown(self):
        """Two pointers may point at the same place; nothing here can rule it out."""
        self.assertEqual(self.rel(Arg(0), 4, Arg(1), 4), "unknown")
        self.assertEqual(self.rel(Const(0x1000), 4, Arg(0), 4), "unknown")

    def test_offsets_wrap_at_32_bits(self):
        # base - 4 and base + 0: four bytes apart once wrapped, so disjoint.
        below = Binary("add", Arg(0), Const(0xFFFFFFFC))
        self.assertEqual(self.rel(below, 4, Arg(0), 4), "disjoint")
        self.assertEqual(self.rel(below, 8, Arg(0), 4), "overlap")

    def test_newest_write_wins(self):
        """A later partial write must not be skipped by an older exact match.

        Matching on exact address alone forwarded the first write here and missed
        the byte written over part of it.
        """
        from ps2decomp.symexec import Unsupported

        memory = {}
        self.paths._record_store(memory, Arg(0), 4, Const(0x11223344))
        self.paths._record_store(memory, Binary("add", Arg(0), Const(1)), 1, Const(0xAA))
        with self.assertRaises(Unsupported):
            self.paths._read_through(memory, Arg(0), 4, False)

    def test_rewriting_a_location_makes_it_newest(self):
        memory = {}
        self.paths._record_store(memory, Arg(0), 4, Const(1))
        self.paths._record_store(memory, Arg(1), 4, Const(2))
        self.paths._record_store(memory, Arg(0), 4, Const(3))
        self.assertEqual(list(memory)[-1], (Arg(0), 4))

    def test_disjoint_writes_fall_through_to_entry_memory(self):
        memory = {}
        self.paths._record_store(memory, Binary("add", Arg(0), Const(8)), 4, Const(9))
        result = self.paths._read_through(memory, Arg(0), 4, True)
        self.assertIsInstance(result, Load)

    def _read(self, addr, size, signed):
        return 0

    def test_forwarding_truncates_to_the_stored_width(self):
        """`sb` then `lbu` yields the stored byte, not the whole register."""
        memory = {}
        self.paths._record_store(memory, Arg(0), 1, Const(0x1234))
        value = self.paths._read_through(memory, Arg(0), 1, False)
        self.assertEqual(value.evaluate((0, 0, 0, 0), self._read), 0x34)

    def test_forwarding_sign_extends_signed_loads(self):
        memory = {}
        self.paths._record_store(memory, Arg(0), 1, Const(0x80))
        value = self.paths._read_through(memory, Arg(0), 1, True)
        self.assertEqual(value.evaluate((0, 0, 0, 0), self._read) & 0xFFFFFFFF,
                         0xFFFFFF80)

    def test_store_then_load_of_a_neighbouring_field_derives(self):
        """The array-writing loops that motivated this: a write at arg0+4 no longer
        blocks a read at arg0+8."""
        # sw $a1, 4($a0) ; lw $v0, 8($a0) ; jr $ra ; nop
        words = [enc_i(0x2B, 4, 5, 4), enc_i(0x23, 4, 2, 8), JR_RA, 0]
        image = assemble(words)
        result = self.paths.derive_paths(image, BASE, BASE, BASE + len(image))
        self.assertIsInstance(simplify(result.paths[0].expression), Load)


class FloatToWordTest(unittest.TestCase):
    """cvt.w.s truncates toward zero below 2**31 and saturates beyond, as the
    EE does (PCSX2's CVT_W the same): the interpreter's float_to_word and the
    deriver's FloatToInt agree on it."""

    CASES = [
        (0x3FC00000, 1),                # 1.5
        (0xBFC00000, 0xFFFFFFFF),       # -1.5 -> -1
        (0x4EFFFFFF, 0x7FFFFF80),       # the largest float below 2**31
        (0xCEFFFFFF, 0x80000080),
        (0x4F000000, 0x7FFFFFFF),       # 2**31: saturates
        (0xCF000000, 0x80000000),       # -2**31
        (0x4F32D05E, 0x7FFFFFFF),       # 3e9 (wrapped, it was 0xB2D05E00)
        (0xCF32D05E, 0x80000000),
        (0x7F800000, 0x7FFFFFFF),       # an infinity's pattern
        (0xFFFFFFFF, 0x80000000),       # a NaN's, negative
        (0x00000001, 0),                # a denormal
    ]

    def test_interpreter(self):
        from ps2decomp.emu import float_to_word
        for bits, word in self.CASES:
            self.assertEqual(float_to_word(bits), word, hex(bits))

    def test_deriver(self):
        from ps2decomp.symexec import FloatToInt
        for bits, word in self.CASES:
            self.assertEqual(FloatToInt(Const(bits)).evaluate([], None), word, hex(bits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
