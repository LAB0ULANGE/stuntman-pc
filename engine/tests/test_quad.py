"""The 128-bit register model in derivation: upper halves, lq/sq, the MMI moves."""

import os
import random
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine
from ps2decomp.paths import Execution, derive_paths
from ps2decomp.symexec import Unsupported

BASE = 0x00100000
JR_RA = 0x03E00008
SCRATCH = 0x01800000
A0, A1, A2, V0, T0, T1 = 4, 5, 6, 2, 8, 9


def mmi2(sub, rd, rs, rt): return (0x1C << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sub << 6) | 0x09
def mmi3(sub, rd, rs, rt): return (0x1C << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sub << 6) | 0x29
def lq(rt, base, imm): return (0x1E << 26) | (base << 21) | (rt << 16) | (imm & 0xFFFF)
def sq(rt, base, imm): return (0x1F << 26) | (base << 21) | (rt << 16) | (imm & 0xFFFF)
def daddu(rd, rs, rt): return (rs << 21) | (rt << 16) | (rd << 11) | 0x2D


def _outside_frame(memory):
    """Bytes written, as production verification compares them: not the stack
    frame, which belongs to the function."""
    from ps2decomp import layout

    low = layout.CURRENT.stack_low - memory.base
    high = layout.CURRENT.stack_top + 0x10 - memory.base
    return {offset: memory.bytes[offset]
            for offset, previous in memory._journal
            for offset in range(offset, offset + len(previous)) if not low <= offset <= high}


def check(test, words, args_for, cases=30):
    """Derive `words`; compare its result and memory writes with the interpreter's."""
    image = struct.pack("<%dI" % len(words), *words)
    try:
        derivation = derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={})
    except Unsupported:
        # As the pipeline does: loops are only attempted when straight paths fail.
        derivation = derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={},
                                  loops=True)
    machine = Machine(BASE, image, 0x02000000 - BASE)
    rng = random.Random(len(words))
    for _case in range(cases):
        args = args_for(rng)
        machine.memory.write(SCRATCH, bytes(rng.getrandbits(8) for _ in range(256)))
        machine.memory.begin_journal()
        expected = machine.call(BASE, args)
        written = _outside_frame(machine.memory)
        machine.memory.rollback()
        read = lambda a, s, g: machine.memory.read_int(a, s, g)
        if derivation.sequential:
            # A loop or a call: the derivation runs, writing memory as it goes.
            actual = derivation.run(tuple(args) + (0,) * (16 - len(args)), Execution(machine))
        else:
            actual = derivation.evaluate(args, read)
            for address, value, size in derivation.evaluate_stores(args, read):
                machine.memory.write_int(address, value, size)
        replayed = _outside_frame(machine.memory)
        machine.memory.rollback()
        machine.memory.end_journal()
        if actual is not None:
            test.assertEqual(actual & 0xFFFFFFFF, expected & 0xFFFFFFFF)
        test.assertEqual(replayed, written)
    return derivation


def pointers(rng):
    return (SCRATCH + 16 * rng.randrange(4), SCRATCH + 128 + 16 * rng.randrange(4),
            rng.getrandbits(64))


class TestQuad(unittest.TestCase):
    def test_a_quadword_built_with_pcpyld_is_stored_whole(self):
        # pcpyld $t0, $a2, $a1 ; jr $ra ; sq $t0, 0($a0)
        check(self, [mmi2(0x0E, T0, A2, A1), JR_RA, sq(T0, A0, 0)], pointers)

    def test_a_struct_copy_through_a_register(self):
        # lq $t0, 0($a1) ; jr $ra ; sq $t0, 16($a0)
        check(self, [lq(T0, A1, 0), JR_RA, sq(T0, A0, 16)], pointers)

    def test_the_upper_half_of_a_loaded_quadword(self):
        # lq $t0, 0($a0) ; pcpyud $t1, $t0, $t0 ; jr $ra ; daddu $v0, $t1, $zero
        check(self, [lq(T0, A0, 0), mmi3(0x0E, T1, T0, T0), JR_RA, daddu(V0, T1, 0)], pointers)

    def test_a_128_bit_or_of_known_halves(self):
        # lq $t0, 0($a0) ; lq $t1, 0($a1) ; por $t0, $t0, $t1 ; jr $ra ; sq $t0, 32($a0)
        check(self, [lq(T0, A0, 0), lq(T1, A1, 0), mmi3(0x12, T0, T0, T1), JR_RA,
                     sq(T0, A0, 32)], pointers)

    def test_an_upper_half_never_written_is_refused(self):
        # pcpyud $v0, $a0, $a0 ; jr $ra ; nop - the caller's upper half of $a0
        image = struct.pack("<3I", mmi3(0x0E, V0, A0, A0), JR_RA, 0)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 12, known={})



COP2 = 0x12 << 26


def lqc2(vf, base, imm): return (0x36 << 26) | (base << 21) | (vf << 16) | (imm & 0xFFFF)
def sqc2(vf, base, imm): return (0x3E << 26) | (base << 21) | (vf << 16) | (imm & 0xFFFF)
def s1(funct, fd, fs, ft, dest=0xF):
    return COP2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11) | (fd << 6) | funct
def s2(index, fs, ft, dest=0xF):
    return COP2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11) | ((index >> 2) << 6) | 0x3C | (index & 3)
def qmfc2(rt, vf): return COP2 | (1 << 21) | (rt << 16) | (vf << 11)
def qmtc2(rt, vf): return COP2 | (5 << 21) | (rt << 16) | (vf << 11)


class TestVu0Derivation(unittest.TestCase):
    def test_vector_maths_through_memory(self):
        words = [
            lqc2(1, A0, 0), lqc2(2, A1, 0),
            s1(0x28, 3, 1, 2),                          # vadd vf3, vf1, vf2
            s1(0x18, 4, 3, 1),                          # vmulx vf4, vf3, vf1x
            s2(0x30, 1, 5), s2(0x2E, 1, 2), s1(0x2E, 5, 2, 1, dest=0b1110),  # vmove, vopmula, vopmsub
            s2(0x38, 1, 2) | (1 << 23), s1(0x1C, 6, 3, 0),     # vdiv Q, vmulq
            s2(0x1D, 4, 7), s2(0x31, 6, 8),             # vabs vf7, vmr32 vf8
            sqc2(4, A0, 32), sqc2(5, A0, 48), sqc2(7, A1, 32), sqc2(8, A1, 48),
            JR_RA, qmfc2(V0, 6),
        ]
        derivation = check(self, words, pointers)
        text = derivation.paths[0].expression.render() + " ".join(
            s.value.render() for s in derivation.paths[0].stores)
        for name in ("VuAdd(", "VuMul(", "VuMsub(", "VuDiv(", "VuAbs("):
            self.assertIn(name, text)

    def test_vectors_through_registers(self):
        # pcpyld $t0, $a2, $a1 ; qmtc2 $t0, vf1 ; vaddw.xz vf1, vf1, vf0w ; jr $ra ; sqc2 vf1, 0($a0)
        check(self, [mmi2(0x0E, T0, A2, A1), qmtc2(T0, 1), s1(0x03, 1, 1, 0, dest=0b1010),
                     JR_RA, sqc2(1, A0, 0)], pointers)

    def test_a_vector_sum_in_a_loop(self):
        # lqc2 vf1, 0($a0) ; loop: lqc2 vf2, 0($a1) ; vadd vf1, vf1, vf2 ;
        # addiu $a1, $a1, 16 ; addiu $a2, $a2, -1 ; bne $a2, $zero, loop ; nop ;
        # jr $ra ; sqc2 vf1, 0($a0)
        loop = BASE + 4
        words = [lqc2(1, A0, 0), lqc2(2, A1, 0), s1(0x28, 1, 1, 2),
                 (0x09 << 26) | (A1 << 21) | (A1 << 16) | 16,
                 (0x09 << 26) | (A2 << 21) | (A2 << 16) | 0xFFFF,
                 (0x05 << 26) | (A2 << 21) | (((loop - (BASE + 20) - 4) >> 2) & 0xFFFF), 0,
                 JR_RA, sqc2(1, A0, 0)]
        check(self, words, lambda rng: (SCRATCH, SCRATCH + 64, rng.randrange(1, 5)))

    def test_a_vector_spilled_to_the_stack_and_reloaded(self):
        SP = 29
        addiu = lambda rt, rs, imm: (0x09 << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
        words = [addiu(SP, SP, -32), lqc2(1, A0, 0), sqc2(1, SP, 0), s1(0x28, 1, 1, 1),
                 lqc2(1, SP, 0), addiu(SP, SP, 32), JR_RA, sqc2(1, A0, 16)]
        check(self, words, pointers)

    def test_x_and_y_of_a_64_bit_value_moved_to_vu0(self):
        ld = (0x37 << 26) | (A0 << 21) | (T0 << 16)                 # ld $t0, 0($a0)
        words = [ld, qmtc2(T0, 1), lqc2(2, A1, 0), s1(0x28, 2, 1, 1, dest=0b1100),
                 JR_RA, sqc2(2, A1, 16)]
        check(self, words, pointers)

    def test_a_vector_register_never_written_is_refused(self):
        image = struct.pack("<3I", JR_RA, qmfc2(V0, 5), 0)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 12, known={})


if __name__ == "__main__":
    unittest.main()
