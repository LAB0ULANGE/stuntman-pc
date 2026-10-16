"""Tests for struct type recovery.

Each test assembles a few tiny functions and checks one property of the recovery:
what joins types, what must not, and the scan bugs that were found on real code.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import structs, typed
from ps2decomp.symexec import Arg, Binary, Const, Load

BASE = 0x00100000

A0, A1, A2, V0, T0, AT, GP, SP, RA = 4, 5, 6, 2, 8, 1, 28, 29, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def r_type(rs, rt, rd, funct, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def lw(rt, offset, rs):
    return i_type(0x23, rs, rt, offset)


def lb(rt, offset, rs):
    return i_type(0x20, rs, rt, offset)


def sw(rt, offset, rs):
    return i_type(0x2B, rs, rt, offset)


def lui(rt, imm):
    return i_type(0x0F, 0, rt, imm)


def addiu(rt, rs, imm):
    return i_type(0x09, rs, rt, imm)


def move(rd, rs):
    return r_type(rs, 0, rd, 0x2D)          # daddu rd, rs, $zero


JR_RA = r_type(RA, 0, 0, 0x08)
NOP = 0


class Program:
    """Lays functions out back to back and builds the database the recovery reads."""

    def __init__(self):
        self.words = []
        self.functions = []

    def address(self) -> int:
        return BASE + 4 * len(self.words)

    def add(self, build) -> int:
        """`build(here)` returns the function's words; returns its address."""
        here = self.address()
        words = build(here)
        self.words += words
        self.functions.append({"addr": here, "end": here + 4 * len(words),
                               "size": 4 * len(words)})
        return here

    def recover(self):
        image = struct.pack("<%dI" % len(self.words), *self.words)
        return structs.recover({"functions": self.functions}, image, BASE)


def jal(target):
    return (3 << 26) | ((target >> 2) & 0x03FFFFFF)


def branch_ne(rs, rt, here, target):
    return i_type(0x05, rs, rt, (target - (here + 4)) // 4)


class TestJoining(unittest.TestCase):
    def test_argument_passed_to_callee_is_the_same_type(self):
        program = Program()
        callee = program.add(lambda here: [lw(V0, 8, A0), JR_RA, NOP])
        caller = program.add(lambda here: [lw(T0, 4, A0), jal(callee), NOP, JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        root = catalog.slot(caller, 0)
        self.assertEqual(root, catalog.slot(callee, 0))
        self.assertEqual(sorted(catalog.roots[root]), [4, 8])

    def test_unread_argument_register_is_not_linked(self):
        """A wrapper's untouched $a1 must not join the callee's unused arg 1.

        Linking every register that still held a value joined the arguments of
        unrelated functions into one type: half of all links were this.
        """
        program = Program()
        callee = program.add(lambda here: [lw(V0, 0, A0), JR_RA, NOP])
        other = program.add(lambda here: [lw(V0, 12, A1), JR_RA, NOP])
        caller = program.add(lambda here: [lw(T0, 4, A1), jal(callee), NOP, JR_RA, NOP])
        reader = program.add(lambda here: [lw(T0, 4, A1), jal(other), NOP, JR_RA, NOP])
        graph = program.recover()
        self.assertNotIn(("arg", callee, 1), graph.ids)
        catalog = structs.Catalog(graph)
        self.assertNotEqual(catalog.slot(caller, 1), catalog.slot(reader, 1))
        # ...whereas a callee that does read $a1 is linked.
        self.assertEqual(catalog.slot(reader, 1), catalog.slot(other, 1))

    def test_conflicting_widths_refuse_the_join(self):
        """A routine handed two incompatible types joins at most one of them."""
        program = Program()
        generic = program.add(lambda here: [lw(V0, 8, A0), JR_RA, NOP])
        word = program.add(lambda here: [lw(T0, 0, A0), jal(generic), NOP, JR_RA, NOP])
        byte = program.add(lambda here: [lb(T0, 0, A0), jal(generic), NOP, JR_RA, NOP])
        graph = program.recover()
        catalog = structs.Catalog(graph)
        self.assertNotEqual(catalog.slot(word, 0), catalog.slot(byte, 0))
        self.assertEqual(graph.refused, 1)


class TestPointersAndEmbedding(unittest.TestCase):
    def test_loaded_pointer_gets_a_type_of_its_own(self):
        program = Program()
        func = program.add(lambda here: [lw(T0, 4, A0), lw(V0, 8, T0), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        target = catalog.pointee(catalog.slot(func, 0), 4)
        self.assertIsNotNone(target)
        self.assertIn(8, catalog.roots[target])

    def test_displaced_pointer_passed_on_is_an_embedded_type(self):
        program = Program()
        inner = program.add(lambda here: [sw(0, 4, A0), JR_RA, NOP])
        outer = program.add(lambda here: [
            lw(T0, 0, A0), jal(inner), addiu(A0, A0, 16), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        root = catalog.slot(outer, 0)
        self.assertEqual(catalog.embedded(root), {16: catalog.slot(inner, 0)})
        path, owner, local, _field = catalog.locate(root, 20)
        self.assertEqual((path, owner, local), ([(16, catalog.slot(inner, 0))],
                                                catalog.slot(inner, 0), 4))

    def test_interior_pointer_accesses_land_on_the_base_type(self):
        program = Program()
        func = program.add(lambda here: [
            lw(T0, 0, A0), addiu(A1, A0, 0x100), sw(0, 4, A1), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        self.assertIn(0x104, catalog.roots[catalog.slot(func, 0)])


class TestScanFlow(unittest.TestCase):
    def test_branch_target_after_a_return_keeps_the_branch_state(self):
        """Padding after `jr $ra` is unreachable and must not erase the target.

        Treating it as reachable with an empty state intersected that emptiness
        into the branch target, and the stores there were never recorded.
        """
        program = Program()

        def build(here):
            target = here + 4 * 6
            return [
                move(A2, A0),
                branch_ne(A1, 0, here + 4, target),
                NOP,
                JR_RA,
                NOP,
                NOP,                     # padding, unreachable
                sw(0, 8, A2),            # the branch target
                JR_RA,
                NOP,
            ]

        func = program.add(build)
        catalog = structs.Catalog(program.recover())
        self.assertIn(8, catalog.roots[catalog.slot(func, 0)])

    def test_register_redefined_on_one_path_is_dropped_at_the_join(self):
        program = Program()

        def build(here):
            join = here + 4 * 4
            return [
                branch_ne(A1, 0, here, join),
                NOP,
                lw(A0, 0, A2),           # only on the fall-through path
                NOP,
                sw(0, 12, A0),           # the join: $a0 means two things here
                JR_RA,
                NOP,
            ]

        func = program.add(build)
        catalog = structs.Catalog(program.recover())
        root = catalog.slot(func, 0)
        self.assertNotIn(12, catalog.roots.get(root, {}))


class TestGlobals(unittest.TestCase):
    """Constant addresses are fields of one node, the global address space."""

    def test_lui_addressed_global_is_a_field_of_globals(self):
        program = Program()
        program.add(lambda here: [lui(AT, 0x49), lw(V0, -0x1234, AT), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        self.assertIn(0x490000 - 0x1234, catalog.roots[catalog.globals])

    def test_gp_relative_access_is_a_global(self):
        """$gp comes from the game's layout; an access off it is a global."""
        from ps2decomp import layout

        saved = layout.CURRENT.gp
        layout.configure(gp=0x004E8670)
        try:
            program = Program()
            program.add(lambda here: [lw(V0, -0x7FF0, GP), JR_RA, NOP])
            catalog = structs.Catalog(program.recover())
            self.assertIn(0x004E8670 - 0x7FF0, catalog.roots[catalog.globals])
        finally:
            layout.configure(gp=saved)

    def test_pointer_held_in_a_global_gets_a_type(self):
        program = Program()
        program.add(lambda here: [lui(AT, 0x49), lw(T0, 0x10, AT), lw(V0, 8, T0), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        target = catalog.pointee(catalog.globals, 0x490010)
        self.assertIn(8, catalog.roots[target])
        self.assertEqual(catalog.names[target], "S_g00490010")

    def test_global_address_passed_on_is_a_global_object(self):
        program = Program()
        inner = program.add(lambda here: [sw(0, 4, A0), JR_RA, NOP])
        program.add(lambda here: [
            lui(A0, 0x49), jal(inner), addiu(A0, A0, 0x100), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        self.assertEqual(catalog.embedded(catalog.globals), {0x490100: catalog.slot(inner, 0)})

    def test_constant_loads_render_as_globals(self):
        program = Program()
        inner = program.add(lambda here: [lw(V0, 4, A0), JR_RA, NOP])
        func = program.add(lambda here: [
            lui(AT, 0x49), lw(V0, 0x20, AT), lui(A0, 0x49), jal(inner),
            addiu(A0, A0, 0x100), JR_RA, NOP])
        typing = typed.Typing(structs.Catalog(program.recover()), func)
        scalar = Load(Const(0x490020), 4, True)
        self.assertEqual(typing.render_load(scalar, lambda n: n.render()), "g_00490020()")
        member = Load(Const(0x490104), 4, True)
        self.assertEqual(typing.render_load(member, lambda n: n.render()),
                         "g_00490100_obj.f_4()")


class TestArrays(unittest.TestCase):
    def test_linear_form_sees_through_shifts_and_sums(self):
        """(i << 2) + i << 4 is 0x50 * i, exactly."""
        i = Load(Arg(1), 4, True)
        index = Binary("shl", Binary("add", Binary("shl", i, Const(2)), i), Const(4))
        self.assertEqual(typed.linear(index), (0x50, i, 0))
        self.assertIsNone(typed.linear(Binary("add", Arg(1), Arg(2))))

    def _typing_for(self, words_for):
        program = Program()
        func = program.add(words_for)
        graph = program.recover()
        return graph, func

    def test_indexed_access_renders_as_array_element(self):
        graph, func = self._typing_for(lambda here: [lw(V0, 0x10, A0), JR_RA, NOP])
        catalog = structs.Catalog(graph)
        root = catalog.slot(func, 0)
        graph.field(root, 0x10).strides[4] += 1      # as derived evidence would
        catalog = structs.Catalog(graph)
        typing = typed.Typing(catalog, func)
        address = Binary("add", Binary("add", Arg(0), Binary("shl", Arg(1), Const(2))),
                         Const(0x10))
        load = Load(address, 4, True)
        self.assertEqual(typing.render_load(load, lambda n: n.render()), "arg0.f_10(arg1)")

    def test_run_of_records_is_an_array(self):
        """An unrolled loop's fields at a fixed spacing become one array."""
        def build(here):
            words = [lw(T0, 0x100, A0)]
            for i in range(8):
                words.append(sw(0, 0x40 + i * 0x50, A0))
            return words + [JR_RA, NOP]

        graph, func = self._typing_for(build)
        catalog = structs.Catalog(graph)
        root = catalog.slot(func, 0)
        self.assertEqual(catalog.arrays(root), {0x40: (0x50, 8)})
        self.assertEqual(catalog.element(root, 0x40 + 3 * 0x50), (0x40, 3))

    def test_a_few_consecutive_ints_stay_a_struct(self):
        """Eight consecutive words look like an array but are usually members."""
        graph, func = self._typing_for(
            lambda here: [lw(T0, i * 4, A0) for i in range(8)] + [JR_RA, NOP])
        catalog = structs.Catalog(graph)
        self.assertEqual(catalog.arrays(catalog.slot(func, 0)), {})


class TestTypedRendering(unittest.TestCase):
    def test_pointer_chain_renders_as_field_accesses(self):
        program = Program()
        func = program.add(lambda here: [lw(T0, 4, A0), lw(V0, 8, T0), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        typing = typed.Typing(catalog, func)
        inner = Load(Binary("add", Arg(0), Const(4)), 4, True)
        outer = Load(Binary("add", inner, Const(8)), 4, True)
        self.assertEqual(typing.render_load(outer, lambda node: node.render()),
                         "arg0.f_4().f_8()")

    def test_load_of_a_different_width_stays_a_plain_read(self):
        """Only an exact match becomes a field access, so a wrong type cannot
        change what the source means."""
        program = Program()
        func = program.add(lambda here: [lw(V0, 4, A0), JR_RA, NOP])
        catalog = structs.Catalog(program.recover())
        typing = typed.Typing(catalog, func)
        load = Load(Binary("add", Arg(0), Const(4)), 2, False)
        self.assertIsNone(typing.access(load))


if __name__ == "__main__":
    unittest.main()
