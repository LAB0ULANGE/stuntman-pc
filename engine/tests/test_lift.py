"""Lifted C++ against the interpreter, instruction family by instruction family.

Each program is hand-assembled, lifted, compiled once into a library, and run from
the same random states as the interpreter. The whole register file and the memory
written must match. Skipped when no C++ compiler is installed.
"""

import os
import random
import struct
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import cppcheck, layout, lift, liftcheck
from ps2decomp.emu import Machine

BASE = 0x00100000
JR_RA = 0x03E00008
NOP = 0
ZERO, AT, V0, V1, A0, A1, A2, A3 = 0, 1, 2, 3, 4, 5, 6, 7
T0, T1, T2, T3, T4, T5, T6, T7, S0, T8, SP, RA = 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 29, 31
T9 = 25


def r_type(funct, rs=0, rt=0, rd=0, sa=0, op=0):
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def branch(op, rs, rt, at, target):
    return i_type(op, rs, rt, (target - at - 4) >> 2)


def cop1(funct, fd, fs, ft):
    return (0x11 << 26) | (0x10 << 21) | (ft << 16) | (fs << 11) | (fd << 6) | funct


def addiu(rt, rs, imm): return i_type(0x09, rs, rt, imm)
def lui(rt, imm): return i_type(0x0F, 0, rt, imm)
def jal(target): return (3 << 26) | ((target >> 2) & 0x03FFFFFF)
def mtc1(rt, fs): return (0x11 << 26) | (4 << 21) | (rt << 16) | (fs << 11)
def mfc1(rt, fs): return (0x11 << 26) | (0 << 21) | (rt << 16) | (fs << 11)
def mfc0(rt, rd): return (0x10 << 26) | (0 << 21) | (rt << 16) | (rd << 11)


def mmi_word(name, rd, rs, rt, sa=None):
    """The MMI encoding of `name`, found by asking the decoder, so the test does not
    depend on a hand-copied table."""
    from ps2decomp.r5900 import DISASM
    for funct in (0x08, 0x28, 0x09, 0x29) + tuple(range(0x30, 0x40)):
        for sub in range(32):
            word = r_type(funct, rs, rt, rd, sub if sa is None or funct < 0x30 else sa, op=0x1C)
            if DISASM.decode(word, 0).mnemonic == name:
                return word
    raise KeyError(name)


def effects(memory):
    """Bytes written, as production verification sees them: the stack frame below
    the test stack top is the function's own and not compared."""
    low = layout.CURRENT.stack_low - memory.base
    high = layout.CURRENT.stack_top + 0x10 - memory.base
    return {o: memory.bytes[o] for o, p in (memory._journal or ())
            for o in range(o, o + len(p)) if not low <= o <= high}


def programs():
    """{name: (offset, words)}; laid out one after another from BASE."""
    out = {}
    at = BASE

    def add(name, words):
        nonlocal at
        out[name] = (at, words)
        at += 4 * len(words)

    add("arithmetic", [
        addiu(V0, A0, 5), r_type(0x21, A0, A1, V1), r_type(0x00, 0, A0, T0, 3),
        r_type(0x03, 0, A1, T1, 2), r_type(0x2A, A0, A1, T2), i_type(0x0B, A0, T3, 100),
        r_type(0x3C, 0, A0, T4, 4), r_type(0x26, A0, A1, T5), lui(T6, 0x8000),
        i_type(0x0D, A2, T7, 0xFFFF), r_type(0x2D, A0, A1, T9), r_type(0x2F, A0, A1, A3),
        r_type(0x04, A2, A0, AT), r_type(0x17, A2, A1, T8),        # sllv, dsrav
        r_type(0x0A, A1, A0, A2), r_type(0x0B, A2, A1, A3),        # movz, movn
        JR_RA, r_type(0x27, A0, A1, T8),                            # nor (delay slot)
    ])
    add("cop0", [
        mfc0(V0, 9), addiu(V0, V0, 1),                              # Count + 1
        mfc0(V1, 12),                                               # Status
        JR_RA, NOP,
    ])
    loop = at
    add("count_bits", [
        r_type(0x25, 0, 0, V0),                                     # or $v0, $zero, $zero
        i_type(0x0C, A0, T0, 1),                                    # andi $t0, $a0, 1
        r_type(0x21, V0, T0, V0),                                   # addu $v0, $v0, $t0
        r_type(0x02, 0, A0, A0, 1),                                 # srl $a0, $a0, 1
        branch(0x05, A0, 0, loop + 16, loop + 4),                   # bne $a0, $zero, loop
        NOP, JR_RA, NOP,
    ])
    add("memory", [
        i_type(0x23, A0, T0, 0), i_type(0x2B, A0, T0, 4), i_type(0x24, A0, T1, 1),
        i_type(0x29, A0, T1, 8), i_type(0x37, A0, T2, 0), i_type(0x3F, A0, T2, 16),
        i_type(0x1E, A0, T3, 0), i_type(0x1F, A0, T3, 32), i_type(0x20, A0, T4, 3),
        JR_RA, i_type(0x27, A0, V0, 0),                             # lwu in the delay slot
    ])
    add("multiply", [
        r_type(0x18, A0, A1), r_type(0x12, rd=V0), r_type(0x10, rd=V1),
        r_type(0x1A, A1, A0), r_type(0x12, rd=T0), r_type(0x10, rd=T1),     # div $a1, $a0
        r_type(0x00, A0, A1, T2, op=0x1C),                          # madd $t2, $a0, $a1
        r_type(0x18, A2, A3, T3, op=0x1C),                          # mult1 $t3, $a2, $a3
        r_type(0x12, rd=T4, op=0x1C),                               # mflo1 $t4
        r_type(0x1B, A1, A2, op=0x1C),                              # divu1 $a1, $a2
        JR_RA, r_type(0x10, rd=T5, op=0x1C),                        # mfhi1 $t5
    ])
    likely = at
    add("likely", [
        branch(0x14, A0, 0, likely, likely + 12),                   # beql $a0, $zero, +12
        addiu(V0, 0, 1),                                            # nullified unless taken
        addiu(V0, V0, 2),
        JR_RA, NOP,
    ])
    callee = at
    add("callee", [JR_RA, r_type(0x21, A0, A0, V0)])               # return a0 + a0
    caller = at
    add("caller", [
        addiu(SP, SP, -16), i_type(0x3F, SP, RA, 0),                # sd $ra, 0($sp)
        jal(callee), addiu(A0, A0, 1),
        i_type(0x37, SP, RA, 0), JR_RA, addiu(SP, SP, 16),
    ])
    fpu = at
    add("float", [
        mtc1(A0, 0), mtc1(A1, 1),
        cop1(0x00, 2, 0, 1), cop1(0x02, 3, 2, 1), cop1(0x03, 4, 3, 0),  # add, mul, div
        cop1(0x34, 0, 0, 1),                                        # c.lt.s $f0, $f1
        cop1(0x04, 5, 0, 12),                                       # sqrt.s $f5, $f12
        cop1(0x24, 6, 3, 0),                                        # cvt.w.s $f6, $f3
        cop1(0x18, 0, 0, 1), cop1(0x1C, 7, 2, 1),                   # adda.s, madd.s
        (0x11 << 26) | (8 << 21) | (1 << 16) | 3,                   # bc1t +3
        NOP, mfc1(V0, 2),
        mfc1(V1, 7), JR_RA, mfc1(T0, 6),
    ])
    add("fpu_control", [
        mtc1(A0, 0), mtc1(A1, 1),
        cop1(0x34, 0, 0, 1),                                        # c.lt.s $f0, $f1
        (0x11 << 26) | (2 << 21) | (V0 << 16) | (31 << 11),         # cfc1 $v0, $31
        (0x11 << 26) | (6 << 21) | (0 << 16) | (31 << 11),          # ctc1 $zero, $31
        (0x11 << 26) | (2 << 21) | (V1 << 16) | (31 << 11),         # cfc1 $v1, $31
        JR_RA, (0x11 << 26) | (2 << 21) | (T0 << 16) | (0 << 11),   # cfc1 $t0, $0
    ])
    add("mmi_pack", [
        r_type(0x09, A0, A1, T4, 0x0E, op=0x1C),                    # pcpyld $t4, $a0, $a1
        r_type(0x09, A2, A3, T5, 0x0E, op=0x1C),                    # pcpyld $t5, $a2, $a3
        r_type(0x08, T4, T5, T0, 0x00, op=0x1C),                    # paddw $t0, $t4, $t5
        r_type(0x08, T4, T5, T1, 0x04, op=0x1C),                    # paddh $t1, $t4, $t5
        r_type(0x08, T4, T5, T2, 0x1B, op=0x1C),                    # ppacb $t2, $t4, $t5
        JR_RA, r_type(0x28, T4, T5, T3, 0x1A, op=0x1C),             # pextub $t3, $t4, $t5
    ])
    mmi = at
    add("mmi", [
        r_type(0x09, A0, A1, T0, 0x0E, op=0x1C),                    # pcpyld $t0, $a0, $a1
        r_type(0x09, T0, A2, T1, 0x12, op=0x1C),                    # pand $t1, $t0, $a2
        r_type(0x29, A2, T0, T2, 0x13, op=0x1C),                    # pnor $t2, $a2, $t0
        r_type(0x29, T0, T1, T3, 0x0E, op=0x1C),                    # pcpyud $t3, $t0, $t1
        r_type(0x08, A0, A1, T4, 0x09, op=0x1C),                    # psubb $t4, $a0, $a1
        JR_RA, r_type(0x29, 0, T0, T5, 0x1B, op=0x1C),              # pcpyh $t5, $t0
    ])
    add("unaligned", [
        i_type(0x22, A0, T0, 3), i_type(0x26, A0, T0, 0),           # lwl/lwr pair
        i_type(0x1A, A0, T1, 7), i_type(0x1B, A0, T1, 0),           # ldl/ldr pair
        i_type(0x22, A0, T2, 5), i_type(0x1B, A0, T3, 3),           # lone halves
        i_type(0x2A, A0, A1, 19), i_type(0x2E, A0, A1, 16),         # swl/swr pair
        i_type(0x2C, A0, A2, 47), i_type(0x2D, A0, A2, 40),         # sdl/sdr pair
        i_type(0x2A, A0, A3, 61), i_type(0x2D, A0, A3, 70),         # lone halves
        JR_RA, NOP,
    ])
    add("mmi_lanes", [
        r_type(0x09, A2, A3, T8, 0x0E, op=0x1C),                    # pcpyld $t8, $a2, $a3
        r_type(0x08, T8, A1, T0, 0x03, op=0x1C),                    # pmaxw $t0, $t8, $a1
        r_type(0x28, T8, A0, T1, 0x03, op=0x1C),                    # pminw $t1, $t8, $a0
        r_type(0x08, T8, A0, T2, 0x01, op=0x1C),                    # psubw $t2, $t8, $a0
        r_type(0x28, T8, A1, T3, 0x10, op=0x1C),                    # padduw $t3, $t8, $a1
        r_type(0x28, A0, T8, T4, 0x18, op=0x1C),                    # paddub $t4, $a0, $t8
        r_type(0x28, 0, T8, T5, 0x01, op=0x1C),                     # pabsw $t5, $t8
        r_type(0x36, 0, T8, T6, 5, op=0x1C),                        # psrlh $t6, $t8, 5
        JR_RA, r_type(0x08, A0, T8, T7, 0x12, op=0x1C),             # pextlw $t7, $a0, $t8
    ])
    add("mmi_compare", [
        r_type(0x09, A0, A1, T8, 0x0E, op=0x1C),                    # pcpyld $t8, $a0, $a1
        r_type(0x09, A2, A3, T9, 0x0E, op=0x1C),                    # pcpyld $t9, $a2, $a3
        mmi_word("pcgtb", T0, T8, T9), mmi_word("pcgth", T1, T8, T9), mmi_word("pcgtw", T2, T9, T8),
        mmi_word("pceqb", T3, T8, T9), mmi_word("pceqh", T4, T8, T8), mmi_word("pceqw", T5, T9, T8),
        mmi_word("pmaxh", T6, T8, T9), mmi_word("pminh", T7, T8, T9),
        mmi_word("pextuh", V0, T8, T9),
        JR_RA, mmi_word("pextuw", V1, T9, T8),
    ])
    add("trap", [r_type(0x0D), JR_RA, NOP])                          # break
    add("mmi_more", [
        r_type(0x09, A0, A1, T8, 0x0E, op=0x1C),                    # pcpyld $t8, $a0, $a1
        mmi_word("pextlh", T0, A2, T8), mmi_word("pextlb", T1, A3, T8), mmi_word("ppach", T2, T8, A2),
        mmi_word("psllh", T3, 0, T8, 5), mmi_word("psrah", T4, 0, T8, 3), mmi_word("psraw", T5, 0, T8, 31),
        mmi_word("psrlw", T6, 0, T8, 7), mmi_word("psllw", T7, 0, T8, 9),
        i_type(0x01, A1, 0x18, 0x5),                                # mtsab $a1, 5
        mmi_word("qfsrv", T9, T8, T0),
        i_type(0x01, A2, 0x19, 0x3),                                # mtsah $a2, 3
        mmi_word("qfsrv", V1, T1, T8),
        mtc1(A0, 1), mtc1(A1, 2), cop1(0x16, 3, 1, 2),              # rsqrt.s $f3, $f1, $f2
        JR_RA, NOP,
    ])
    cop2 = 0x12 << 26

    def s1(funct, fd, fs, ft, dest=0xF):
        return cop2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11) | (fd << 6) | funct

    def s2(index, fs, ft, dest=0xF):
        return (cop2 | (1 << 25) | (dest << 21) | (ft << 16) | (fs << 11)
                | ((index >> 2) << 6) | 0x3C | (index & 3))

    add("vu0", [
        r_type(0x09, A1, A0, T0, 0x0E, op=0x1C), cop2 | (5 << 21) | (T0 << 16) | (1 << 11),
        r_type(0x09, A3, A2, T1, 0x0E, op=0x1C), cop2 | (5 << 21) | (T1 << 16) | (2 << 11),
        s1(0x28, 3, 1, 2), s1(0x2A, 4, 3, 1, dest=0b1101),          # vadd, vmul.xyw
        s2(0x2A, 1, 2), s1(0x29, 5, 1, 2),                           # vmula, vmadd
        s2(0x2E, 1, 2), s1(0x2E, 6, 2, 1, dest=0b1110),              # vopmula, vopmsub
        s2(0x38, 1, 2) | (1 << 23), s1(0x1C, 7, 2, 0),               # vdiv Q, vmulq
        s2(0x39, 0, 3) | (2 << 23),                                  # vsqrt Q
        s1(0x03, 8, 4, 5, dest=0b1010), s1(0x2F, 9, 1, 2),           # vaddw.xz, vmini
        s2(0x14, 3, 10), s2(0x10, 10, 11), s2(0x31, 2, 12),          # vftoi0, vitof0, vmr32
        s2(0x1D, 6, 13), s2(0x3C, 3, 1) | (2 << 21),                 # vabs, vmtir vi1
        s1(0x30, 2, 1, 1), s2(0x3D, 2, 14),                          # viadd vi2, vmfir
        cop2 | (6 << 21) | (A0 << 16) | (21 << 11), s1(0x1E, 15, 1, 0),  # ctc2 I, vmuli
        cop2 | (1 << 21) | (V0 << 16) | (5 << 11),                   # qmfc2 $v0, vf5
        cop2 | (2 << 21) | (V1 << 16) | (2 << 11),                   # cfc2 $v1, vi2
        JR_RA, cop2 | (1 << 21) | (T2 << 16) | (9 << 11),            # qmfc2 $t2, vf9
    ])
    return out


class TestLift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cppcheck.find_vcvars()
        except cppcheck.CompilerUnavailable as exc:
            raise unittest.SkipTest(str(exc))
        cls.programs = programs()
        words = []
        for _name, (start, body) in sorted(cls.programs.items(), key=lambda kv: kv[1][0]):
            words += body
        cls.image = struct.pack("<%dI" % len(words), *words)
        work = tempfile.mkdtemp()
        source = os.path.join(work, "lifted.cpp")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write('#include "ps2rt/lifted.h"\n\n')
            for _name, (start, body) in cls.programs.items():
                handle.write(lift.lift_function(cls.image, BASE, start, start + 4 * len(body)))
        cls.machine = Machine(BASE, cls.image, 0x02000000 - BASE)
        cls.lifted = liftcheck.LiftedFunctions(liftcheck.build([source], work), cls.machine)

    def check(self, name, cases=40):
        start, _body = self.programs[name]
        rng = random.Random(start)
        scratch = layout.CURRENT.scratch
        for case in range(cases):
            args = [rng.choice([rng.getrandbits(64), rng.getrandbits(8), 0,
                                (1 << 64) - rng.getrandbits(4) - 1])
                    for _ in range(8)]
            if name == "memory":
                args[0] = scratch + 16 * rng.randrange(64)
            if name == "unaligned":
                args[0] = scratch + rng.randrange(1024)
            if name == "vu0":
                args[:4] = [struct.unpack("<Q", struct.pack("<2f", rng.uniform(-50, 50),
                                                            rng.uniform(-50, 50)))[0]
                            for _ in range(4)]
            floats = [struct.unpack("<I", struct.pack("<f", rng.uniform(-100, 100)))[0]
                      for _ in range(8)]
            m = self.machine
            m.memory.end_journal()
            m.memory.write(scratch, bytes(rng.getrandbits(8) for _ in range(2048)))
            m.memory.begin_journal()
            m.call(start, args + floats)
            expected = liftcheck.machine_state(m)
            written = effects(m.memory)
            _unused = dict(m.memory._journal and {o: m.memory.bytes[o] for o, p in m.memory._journal
                                                  for o in range(o, o + len(p))} or {})
            m.memory.rollback()
            context = liftcheck.entry_context(args + floats)
            self.assertIsNone(self.lifted.run(start, context))
            actual = liftcheck.context_state(context)
            replayed = effects(m.memory)
            _unused = dict(m.memory._journal and {o: m.memory.bytes[o] for o, p in m.memory._journal
                                                   for o in range(o, o + len(p))} or {})
            m.memory.rollback()
            m.memory.end_journal()
            self.assertEqual(actual, expected, "%s, case %d" % (name, case))
            self.assertEqual(replayed, written, "%s, case %d: memory" % (name, case))

    def test_arithmetic_and_shifts(self):
        self.check("arithmetic")

    def test_loop(self):
        self.check("count_bits")

    def test_cop0_reads_as_the_interpreter(self):
        # No host hook under verification: Count and Status read as 0, as in emu.py.
        self.check("cop0")

    def test_loads_and_stores(self):
        self.check("memory")

    def test_multiply_and_divide_on_both_pipelines(self):
        self.check("multiply")

    def test_likely_branch_nullifies_its_delay_slot(self):
        self.check("likely")

    def test_call_with_the_whole_context(self):
        self.check("caller")

    def test_float(self):
        self.check("float")

    def test_fpu_control_registers(self):
        self.check("fpu_control")

    def test_more_lane_operations(self):
        self.check("mmi_pack")

    def test_128_bit_mmi(self):
        self.check("mmi")

    def test_lane_wise_mmi(self):
        self.check("mmi_lanes")

    def test_unaligned_loads_and_stores(self):
        self.check("unaligned")

    def test_more_mmi_and_rsqrt(self):
        self.check("mmi_more")

    def test_mmi_comparisons_and_upper_interleaves(self):
        self.check("mmi_compare")

    def test_vu0_macro_mode(self):
        self.check("vu0")

    def test_break_abandons_the_run(self):
        start, _body = self.programs["trap"]
        self.assertEqual(self.lifted.run(start, liftcheck.entry_context(())), "trap")


class TestStructured(TestLift):
    """Every program again, in structured form (structure.py), against the interpreter."""

    @classmethod
    def setUpClass(cls):
        try:
            cppcheck.find_vcvars()
        except cppcheck.CompilerUnavailable as exc:
            raise unittest.SkipTest(str(exc))
        from ps2decomp import structure

        cls.programs = programs()
        words = []
        for _name, (start, body) in sorted(cls.programs.items(), key=lambda kv: kv[1][0]):
            words += body
        cls.image = struct.pack("<%dI" % len(words), *words)
        cls.structured = set()
        work = tempfile.mkdtemp()
        source = os.path.join(work, "structured.cpp")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write('#include "ps2rt/lifted.h"\n\n')
            for name, (start, body) in cls.programs.items():
                end = start + 4 * len(body)
                try:
                    handle.write(structure.structure_function(cls.image, BASE, start, end))
                    cls.structured.add(name)
                except lift.Refused:
                    handle.write(lift.lift_function(cls.image, BASE, start, end))
        cls.machine = Machine(BASE, cls.image, 0x02000000 - BASE)
        cls.lifted = liftcheck.LiftedFunctions(liftcheck.build([source], work), cls.machine)

    def test_the_programs_are_structured(self):
        self.assertTrue({"count_bits", "likely", "caller", "arithmetic", "float"} <= self.structured)


class TestLiftRefusals(unittest.TestCase):
    def test_mfc0_reads_through_the_runtime(self):
        # A port's host answers COP0 (Count from its clock): mfc0 is not a constant.
        image = struct.pack("<3I", mfc0(V0, 9), JR_RA, NOP)
        text = lift.lift_function(image, BASE, BASE, BASE + 12)
        self.assertIn("lifted::ReadCop0(9u)", text)
        self.assertNotIn("c.r[2] = 0ull;", text)

    def test_unsupported_instruction_is_refused(self):
        image = struct.pack("<3I", 0x42000002, JR_RA, NOP)      # tlbwi: no TLB on the host
        with self.assertRaises(lift.Refused):
            lift.lift_function(image, BASE, BASE, BASE + 12)

    def test_an_out_of_line_chunk_is_part_of_the_function(self):
        # 00: beq $a0, $zero, 0x40 / 08: addiu $v0, 1 / 0C: jr $ra - and, out of
        # line at 0x40: addiu $v0, 2 / b 0x0C.
        words = [0x1080000F, NOP, 0x24020001, JR_RA, NOP] + [NOP] * 11 + [0x24020002, 0x1000FFF1, NOP]
        image = struct.pack("<%dI" % len(words), *words)
        text = lift.lift_function(image, BASE, BASE, BASE + 0x14, chunks=[(BASE + 0x40, BASE + 0x4C)])
        self.assertIn("goto L_%08X;" % (BASE + 0x40), text)      # the branch out of line
        self.assertIn("L_%08X:" % (BASE + 0x40), text)
        self.assertIn("goto L_%08X;" % (BASE + 0x0C), text)      # and back
        self.assertNotIn("TailCall(c, 0x%08Xu" % (BASE + 0x40), text)

    def test_a_register_jump_dispatches_to_what_can_be_its_targets(self):
        # 00: lui $t1, 0x0010 / 04: addiu $t1, $t1, 0x18 / 08: jr $a0 / 0C: nop /
        # 10, 14, 18: addiu $v0, $zero, 1, 2, 3 / 1C: jr $ra / 20: nop; and a jump
        # table at 0x40 naming 0x10. 0x18 is an address constant; 0x14 nothing names.
        words = [lui(T1, 0x0010), addiu(T1, T1, 0x18), r_type(0x08, rs=A0), NOP,
                 addiu(V0, 0, 1), addiu(V0, 0, 2), addiu(V0, 0, 3), JR_RA, NOP] + [NOP] * 7 + [BASE + 0x10]
        image = struct.pack("<%dI" % len(words), *words)
        # A function of ordinary size: every instruction is a target, through
        # one switch (a table may be built at run time).
        text = lift.lift_function(image, BASE, BASE, BASE + 0x24)
        self.assertEqual(text.count("switch (jump)"), 1)
        self.assertIn("case 0x%08Xu:" % (BASE + 0x14), text)
        # A very large one: only what can be known.
        saved = lift.DISPATCH_ALL
        lift.DISPATCH_ALL = 4
        try:
            text = lift.lift_function(image, BASE, BASE, BASE + 0x24)
        finally:
            lift.DISPATCH_ALL = saved
        self.assertEqual(text.count("switch (jump)"), 1)
        for target in (BASE, BASE + 0x10, BASE + 0x18):
            self.assertIn("case 0x%08Xu: goto L_%08X;" % (target, target), text)
        self.assertNotIn("case 0x%08Xu:" % (BASE + 0x14), text)

    def test_vcallms_runs_vu0_through_the_host(self):
        # vcallms 0x30: the microprogram at instruction 6 (byte 0x30).
        image = struct.pack("<3I", 0x4A000038 | (6 << 6), JR_RA, NOP)
        text = lift.lift_function(image, BASE, BASE, BASE + 12)
        self.assertIn("lifted::Vcallms(c, 0x0030u);", text)

    def test_delay_slot_reached_by_a_jump_is_emitted_again(self):
        # 0: beq $a0, $zero, +8 ; 4: addiu $v0, $zero, 1 ; 8: j 4 ; 12: nop
        image = struct.pack("<6I", branch(0x04, A0, 0, BASE, BASE + 16), addiu(V0, 0, 1),
                            (2 << 26) | ((BASE + 4) >> 2), NOP, JR_RA, NOP)
        text = lift.lift_function(image, BASE, BASE, BASE + 24)
        self.assertEqual(text.count("c.r[2] = lifted::Sx32"), 2)


if __name__ == "__main__":
    unittest.main()
