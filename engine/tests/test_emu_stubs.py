"""Tests for the interpreter's testing stubs and the FPU accumulator.

The stubs let original *callees* run during a test. Their behaviour is a stated
assumption, not a reconstruction, so these tests pin exactly what was assumed.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp import layout, testmem
from ps2decomp.emu import EmulationError, Machine

BASE = 0x00100000
JR_RA = 0x03E00008


def machine(words):
    image = struct.pack("<%dI" % len(words), *words)
    return Machine(BASE, image, 0x02000000 - BASE)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class TestStubs(unittest.TestCase):
    def test_syscall_returns_zero(self):
        # addiu $v0, $zero, 7 ; syscall ; jr $ra ; nop
        m = machine([0x24020007, 0x0000000C, JR_RA, 0])
        self.assertEqual(m.call(BASE, ()), 0)

    def test_sif_register_reads_reply_every_flag_set(self):
        # addiu $v1, $zero, 0x7A (SifGetReg) ; syscall ; jr $ra ; nop
        m = machine([0x2403007A, 0x0000000C, JR_RA, 0])
        self.assertEqual(m.call(BASE, ()), -1)

    def test_mfc0_reads_zero(self):
        # addiu $v0, $zero, 7 ; mfc0 $v0, Count ; jr $ra ; nop
        m = machine([0x24020007, 0x40024800, JR_RA, 0])
        self.assertEqual(m.call(BASE, ()), 0)

    def test_calling_outside_the_executable_returns_zero(self):
        """A function pointer read from random test memory points nowhere real."""
        m = machine([0x24020007, JR_RA, 0])
        self.assertEqual(m.call(0x01800000, ()), 0)
        self.assertEqual(m.stubbed_calls, 1)

    def test_tail_call_through_a_pointer_outside_the_executable(self):
        # addiu $v0, $zero, 7 ; jr $a0 ; nop
        m = machine([0x24020007, 0x00800008, 0])
        self.assertEqual(m.call(BASE, (0x01800000,)), 0)
        self.assertEqual(m.stubbed_calls, 1)

    def test_indirect_call_inside_the_executable_runs(self):
        # 0: addiu $v0, $zero, 7 ; jr $a0 ; nop    3: addiu $v0, $zero, 9 ; jr $ra ; nop
        m = machine([0x24020007, 0x00800008, 0, 0x24020009, JR_RA, 0])
        self.assertEqual(m.call(BASE, (BASE + 12,)), 9)
        self.assertEqual(m.stubbed_calls, 0)

    def test_trap_fires_only_when_its_condition_holds(self):
        # tge $a1, $a0 ; jr $ra ; nop
        m = machine([0x00A40030, JR_RA, 0])
        m.call(BASE, (5, 4))             # 4 >= 5 is false: no trap
        with self.assertRaises(EmulationError):
            m.call(BASE, (5, 6))         # 6 >= 5: trap


class TestInstructions(unittest.TestCase):
    """Instructions added because callees of untestable functions used them."""

    MASK64 = (1 << 64) - 1

    def test_ei_and_di_do_nothing(self):
        m = machine([0x42000038, 0x42000039, JR_RA, 0x24020007])   # ei ; di ; addiu $v0, 7
        self.assertEqual(m.call(BASE, ()), 7)

    def test_sqrt_takes_its_operand_from_ft(self):
        # sqrt.s $f2, $f12 ; jr $ra ; mfc1 $v0, $f2     ($f0 holds 0, $f12 holds 4)
        m = machine([0x460C0084, JR_RA, 0x44021000])
        self.assertEqual(m.call(BASE, (0,) * 8 + (bits(4.0),)) & 0xFFFFFFFF, bits(2.0))

    UNALIGNED = 0x01800001          # one past an aligned address holding 00 11 22 ...

    def _unaligned_machine(self, words):
        m = machine(words)
        m.memory.write(self.UNALIGNED - 1, bytes(range(0, 0x100, 0x11)))
        return m

    def test_word_load_pair_assembles_an_unaligned_word(self):
        # lwl $v0, 3($a0) ; lwr $v0, 0($a0) ; jr $ra ; nop
        m = self._unaligned_machine([0x88820003, 0x98820000, JR_RA, 0])
        self.assertEqual(m.call(BASE, (self.UNALIGNED,)), 0x44332211)

    def test_doubleword_load_pair_assembles_an_unaligned_doubleword(self):
        # ldl $v0, 7($a0) ; ldr $v0, 0($a0) ; jr $ra ; nop
        m = self._unaligned_machine([0x68820007, 0x6C820000, JR_RA, 0])
        m.call(BASE, (self.UNALIGNED,))
        self.assertEqual(m.regs[2], 0x8877665544332211)

    def test_word_store_pair_writes_only_its_four_bytes(self):
        # swl $a1, 3($a0) ; swr $a1, 0($a0) ; jr $ra ; nop
        m = self._unaligned_machine([0xA8850003, 0xB8850000, JR_RA, 0])
        m.call(BASE, (self.UNALIGNED, 0xDDCCBBAA))
        self.assertEqual(m.memory.read(self.UNALIGNED - 1, 6),
                         bytes([0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0x55]))

    def test_pcpyld_reads_its_operands_before_writing(self):
        # pcpyld $a0, $a0, $a1 ; jr $ra ; nop
        m = machine([0x70852389, JR_RA, 0])
        m.call(BASE, (0x1111, 0x2222))
        self.assertEqual((m.regs[4], m.regs_hi[4]), (0x2222, 0x1111))

    def test_break_traps(self):
        m = machine([0x0000000D, JR_RA, 0])
        with self.assertRaises(EmulationError):
            m.call(BASE, ())

    def test_lwu_zero_extends(self):
        m = machine([0x9C820000, JR_RA, 0])             # lwu $v0, 0($a0)
        m.memory.write_int(0x01800000, 0xFFFFFFF0, 4)
        m.call(BASE, (0x01800000,))
        self.assertEqual(m.regs[2], 0xFFFFFFF0)

    def test_a_dma_poll_on_cop0_condition_ends(self):
        # 0: bc0f 0 (to itself) ; nop ; jr $ra ; addiu $v0, $zero, 7
        m = machine([0x4100FFFF, 0, JR_RA, 0x24020007])
        self.assertEqual(m.call(BASE, (), max_instructions=100), 7)

    def test_doubleword_variable_shift_takes_the_amount_mod_64(self):
        dsllv = 0x00A41014                       # dsllv $v0, $a0, $a1
        m = machine([dsllv, JR_RA, 0])
        m.call(BASE, (1, 40))
        self.assertEqual(m.regs[2], 1 << 40)
        m.call(BASE, (1, 104))
        self.assertEqual(m.regs[2], 1 << 40)

    def test_dsrav_keeps_the_sign(self):
        m = machine([0x00A41017, JR_RA, 0])      # dsrav $v0, $a0, $a1
        m.call(BASE, (-256 & self.MASK64, 4))
        self.assertEqual(m.regs[2], -16 & self.MASK64)

    def test_second_pipeline_leaves_hi_lo_alone(self):
        mult = 0x00850018                        # mult $a0, $a1
        mult1 = 0x70000000 | (6 << 21) | (7 << 16) | (2 << 11) | 0x18   # mult1 $v0, $a2, $a3
        mflo = 0x00001812                        # mflo $v1
        mflo1 = 0x70004012                       # mflo1 $t0
        m = machine([mult, mult1, mflo, mflo1, JR_RA, 0])
        m.call(BASE, (2, 3, 4, 5))
        self.assertEqual((m.regs[2], m.regs[3], m.regs[8]), (20, 6, 20))

    def test_madd_accumulates_into_hi_lo(self):
        mult = 0x00850018                        # mult $a0, $a1
        madd = 0x70000000 | (6 << 21) | (7 << 16) | (2 << 11)          # madd $v0, $a2, $a3
        m = machine([mult, madd, JR_RA, 0])
        self.assertEqual(m.call(BASE, (2, 3, 4, 5)), 26)


class TestAccumulator(unittest.TestCase):
    def test_madd_adds_the_product_to_the_accumulator(self):
        # adda.s $f12, $f14 ; madd.s $f4, $f12, $f14 ; jr $ra ; mfc1 $v0, $f4
        adda = 0x460E6018
        madd = 0x460E611C
        mfc1 = 0x44022000                 # mfc1 $v0, $f4
        m = machine([adda, madd, JR_RA, mfc1])
        # The operands arrive as float arguments: $f12 = 3, $f14 = 2.
        args = (0,) * 8 + (bits(3.0), 0, bits(2.0))
        # ACC = 3 + 2 = 5; f4 = ACC + 3 * 2 = 11
        self.assertEqual(m.call(BASE, args) & 0xFFFFFFFF, bits(11.0))

    def test_every_call_starts_from_clean_registers(self):
        """A register left over from an earlier run must not leak into the next."""
        # jr $ra ; daddu $v0, $t4, $zero   (returns whatever $t4 holds)
        m = machine([JR_RA, 0x0180102D])
        m.regs[12] = 0x1234
        self.assertEqual(m.call(BASE, ()), 0)


class TestWindows(unittest.TestCase):
    def test_hardware_registers_and_scratchpad_are_plain_memory(self):
        m = machine([JR_RA, 0])
        for address in (0x10008000, 0x7000_3FFC):
            self.assertEqual(m.memory.read_int(address, 4, False), 0)
            m.memory.write_int(address, 0xCAFE, 4)
            self.assertEqual(m.memory.read_int(address, 4, False), 0xCAFE)

    def test_outside_every_window_still_faults(self):
        m = machine([JR_RA, 0])
        for address in (0x10010000, 0x70004000, 0x0FFFFFFC):
            with self.assertRaises(EmulationError):
                m.memory.read_int(address, 4, False)

    def test_window_writes_are_rolled_back(self):
        m = machine([JR_RA, 0])
        m.memory.begin_journal()
        m.memory.write_int(0x1000A000, 7, 4)
        m.memory.rollback()
        m.memory.end_journal()
        self.assertEqual(m.memory.read_int(0x1000A000, 4, False), 0)


class TestFaultPages(unittest.TestCase):
    # lw $t0, 0($a2) ; addiu $t0, $t0, 1 ; sw $t0, 0($a2) ; lw $v0, 0($a0) ; jr $ra ; nop
    INCREMENT_THEN_LOAD = [0x8CC80000, 0x25080001, 0xACC80000, 0x8C820000, JR_RA, 0]
    WILD = 0x03000010

    def _machine(self):
        m = machine(self.INCREMENT_THEN_LOAD)
        m.memory.page_source = testmem.page_contents
        m.map_faults = True
        return m

    def test_a_fault_maps_the_page_and_the_call_runs_again(self):
        m = self._machine()
        counter = layout.CURRENT.scratch
        m.memory.begin_journal()
        result = m.call(BASE, (self.WILD, 0, counter))
        expected = struct.unpack_from("<i", testmem.page_contents(self.WILD & ~0xFFF), 0x10)[0]
        self.assertEqual(result, expected)
        # The increment before the fault was undone before the rerun: it happened once.
        self.assertEqual(m.memory.read_int(counter, 4, False), 1)

    def test_without_a_journal_the_fault_stands(self):
        m = self._machine()
        with self.assertRaises(EmulationError):
            m.call(BASE, (self.WILD, 0, layout.CURRENT.scratch))


class TestGlobalPointerSeeding(unittest.TestCase):
    def test_only_null_globals_are_seeded_and_into_scratch(self):
        m = machine([0] * 64)
        null_global, set_global = BASE + 0x40, BASE + 0x44
        m.memory.write_int(set_global, 0x12345678, 4)
        seeded = testmem.seed_global_pointers(m.memory, {null_global, set_global})
        self.assertEqual(seeded, 1)
        pointer = m.memory.read_int(null_global, 4, False)
        self.assertTrue(layout.CURRENT.scratch <= pointer < layout.CURRENT.scratch + layout.CURRENT.scratch_size)
        self.assertEqual(m.memory.read_int(set_global, 4, False), 0x12345678)

    # lui $t0, 0x10 ; lw $t1, 0x100($t0) ; lw $v0, 8($t1) ; jr $ra ; nop
    FOLLOWS_GLOBAL = [0x3C080010, 0x8D090100, 0x8D220008, JR_RA, 0]

    def test_a_null_pointer_found_by_running_is_seeded(self):
        """The global at BASE+0x100 holds null; the function reads a field through
        it. Running finds it, seeds it, and the function then runs."""
        m = machine(self.FOLLOWS_GLOBAL + [0] * 64)
        func = {"addr": BASE, "end": BASE + 20}
        with self.assertRaises(EmulationError):
            m.call(BASE, ())
        seeded = testmem.discover_null_pointers(m, [func], lambda a: a >= BASE + 0x40)
        self.assertEqual(seeded, [BASE + 0x100])
        pointer = m.memory.read_int(BASE + 0x100, 4, False)
        self.assertTrue(layout.CURRENT.scratch <= pointer < layout.CURRENT.scratch + layout.CURRENT.scratch_size)
        m.call(BASE, ())

    def test_nothing_outside_the_program_data_is_seeded(self):
        m = machine(self.FOLLOWS_GLOBAL + [0] * 64)
        func = {"addr": BASE, "end": BASE + 20}
        self.assertEqual(testmem.discover_null_pointers(m, [func], lambda a: False), [])
        self.assertEqual(m.memory.read_int(BASE + 0x100, 4, False), 0)


if __name__ == "__main__":
    unittest.main()
