"""Unit tests for the R5900 disassembler.

Encodings are hand-assembled from the MIPS III specification and the EE Core
User's Manual, so the expected strings here are independent of the decoder.
"""

import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.r5900 import decode, Flags


def enc_r(op, rs, rt, rd, sa, funct):
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def enc_i(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def enc_j(op, target):
    return (op << 26) | (target & 0x03FFFFFF)


class TestCoreMips(unittest.TestCase):
    def test_nop(self):
        ins = decode(0x00000000, 0x100000)
        self.assertEqual(ins.mnemonic, "nop")
        self.assertTrue(ins.is_nop)

    def test_addiu(self):
        # addiu $sp, $sp, -32
        ins = decode(enc_i(0x09, 29, 29, -32), 0x100000)
        self.assertEqual(ins.mnemonic, "addiu")
        self.assertEqual(ins.operands, ("$sp", "$sp", "-0x20"))
        self.assertEqual(ins.imm, -32)

    def test_lui_ori_pair(self):
        lui = decode(enc_i(0x0F, 0, 2, 0x1234), 0)
        self.assertEqual(lui.mnemonic, "lui")
        self.assertEqual(lui.operands, ("$v0", "0x1234"))
        ori = decode(enc_i(0x0D, 2, 2, 0x5678), 4)
        self.assertEqual(ori.mnemonic, "ori")
        self.assertEqual(ori.operands, ("$v0", "$v0", "0x5678"))

    def test_load_store(self):
        lw = decode(enc_i(0x23, 29, 31, 0x1C), 0)
        self.assertEqual(lw.mnemonic, "lw")
        self.assertEqual(lw.operands, ("$ra", "0x1c($sp)"))
        self.assertTrue(lw.flags & Flags.LOAD)

        sw = decode(enc_i(0x2B, 29, 31, 0x1C), 0)
        self.assertEqual(sw.mnemonic, "sw")
        self.assertTrue(sw.flags & Flags.STORE)

    def test_negative_load_offset(self):
        ins = decode(enc_i(0x23, 28, 4, -0x10), 0)
        self.assertEqual(ins.operands, ("$a0", "-0x10($gp)"))

    def test_jr_ra_is_return(self):
        ins = decode(enc_r(0, 31, 0, 0, 0, 0x08), 0)
        self.assertEqual(ins.mnemonic, "jr")
        self.assertTrue(ins.is_return)
        self.assertTrue(ins.is_jump)
        self.assertTrue(ins.has_delay_slot)
        self.assertTrue(ins.ends_block)

    def test_jr_other_reg_is_not_return(self):
        ins = decode(enc_r(0, 8, 0, 0, 0, 0x08), 0)
        self.assertTrue(ins.is_jump)
        self.assertFalse(ins.is_return)

    def test_jalr_default_ra(self):
        ins = decode(enc_r(0, 9, 0, 31, 0, 0x09), 0)
        self.assertEqual(ins.mnemonic, "jalr")
        self.assertEqual(ins.operands, ("$t1",))
        self.assertTrue(ins.is_call)

    def test_jal_target(self):
        ins = decode(enc_j(0x03, 0x00100200 >> 2), 0x00100000)
        self.assertEqual(ins.mnemonic, "jal")
        self.assertTrue(ins.is_call)
        self.assertEqual(ins.target, 0x00100200)

    def test_j_target_uses_delay_slot_region(self):
        ins = decode(enc_j(0x02, 0x00100040 >> 2), 0x00100000)
        self.assertEqual(ins.target, 0x00100040)
        self.assertTrue(ins.ends_block)

    def test_beq_backward(self):
        # beq $a0, $a1, -4 instructions => target = addr + 4 - 16
        ins = decode(enc_i(0x04, 4, 5, -4), 0x100040)
        self.assertEqual(ins.mnemonic, "beq")
        self.assertEqual(ins.target, 0x100040 + 4 - 16)
        self.assertTrue(ins.is_branch)
        self.assertFalse(ins.ends_block)  # conditional: still falls through

    def test_beq_zero_zero_is_b(self):
        ins = decode(enc_i(0x04, 0, 0, 2), 0x100000)
        self.assertEqual(ins.mnemonic, "b")
        self.assertTrue(ins.is_jump)
        self.assertTrue(ins.ends_block)

    def test_branch_likely_flag(self):
        ins = decode(enc_i(0x15, 4, 5, 2), 0x100000)  # bnel
        self.assertEqual(ins.mnemonic, "bnel")
        self.assertTrue(ins.is_likely)

    def test_regimm_bgezal_is_call(self):
        ins = decode(enc_i(0x01, 4, 0x11, 2), 0x100000)
        self.assertEqual(ins.mnemonic, "bgezal")
        self.assertTrue(ins.is_call)

    def test_special_arithmetic(self):
        ins = decode(enc_r(0, 4, 5, 2, 0, 0x21), 0)
        self.assertEqual(ins.mnemonic, "addu")
        self.assertEqual(ins.operands, ("$v0", "$a0", "$a1"))

    def test_shift_uses_sa(self):
        ins = decode(enc_r(0, 0, 4, 2, 3, 0x00), 0)
        self.assertEqual(ins.mnemonic, "sll")
        self.assertEqual(ins.operands, ("$v0", "$a0", 3))

    def test_mult_two_and_three_operand(self):
        two = decode(enc_r(0, 4, 5, 0, 0, 0x18), 0)
        self.assertEqual(two.operands, ("$a0", "$a1"))
        three = decode(enc_r(0, 4, 5, 2, 0, 0x18), 0)
        self.assertEqual(three.operands, ("$v0", "$a0", "$a1"))

    def test_movz_movn(self):
        self.assertEqual(decode(enc_r(0, 4, 5, 2, 0, 0x0A), 0).mnemonic, "movz")
        self.assertEqual(decode(enc_r(0, 4, 5, 2, 0, 0x0B), 0).mnemonic, "movn")


class TestMips3SixtyFourBit(unittest.TestCase):
    def test_daddu(self):
        self.assertEqual(decode(enc_r(0, 4, 5, 2, 0, 0x2D), 0).mnemonic, "daddu")

    def test_dsll32(self):
        ins = decode(enc_r(0, 0, 4, 2, 1, 0x3C), 0)
        self.assertEqual(ins.mnemonic, "dsll32")

    def test_ld_sd(self):
        self.assertEqual(decode(enc_i(0x37, 29, 4, 0), 0).mnemonic, "ld")
        self.assertEqual(decode(enc_i(0x3F, 29, 4, 0), 0).mnemonic, "sd")


class TestEmotionEngineExtensions(unittest.TestCase):
    """The instructions a generic MIPS disassembler gets wrong."""

    def test_lq_sq_are_128bit(self):
        lq = decode(enc_i(0x1E, 29, 4, 0x10), 0)
        self.assertEqual(lq.mnemonic, "lq")
        self.assertEqual(lq.operands, ("$a0", "0x10($sp)"))
        sq = decode(enc_i(0x1F, 29, 4, 0x10), 0)
        self.assertEqual(sq.mnemonic, "sq")

    def test_mmi0_paddw(self):
        # MMI funct=0x08 (MMI0), sa=0x00 => paddw
        ins = decode(enc_r(0x1C, 4, 5, 2, 0x00, 0x08), 0)
        self.assertEqual(ins.mnemonic, "paddw")
        self.assertEqual(ins.operands, ("$v0", "$a0", "$a1"))

    def test_mmi1_pextuw(self):
        ins = decode(enc_r(0x1C, 4, 5, 2, 0x12, 0x28), 0)
        self.assertEqual(ins.mnemonic, "pextuw")

    def test_mmi2_pcpyld(self):
        ins = decode(enc_r(0x1C, 4, 5, 2, 0x0E, 0x09), 0)
        self.assertEqual(ins.mnemonic, "pcpyld")

    def test_mmi3_pcpyud(self):
        ins = decode(enc_r(0x1C, 4, 5, 2, 0x0E, 0x29), 0)
        self.assertEqual(ins.mnemonic, "pcpyud")

    def test_mmi3_por(self):
        ins = decode(enc_r(0x1C, 4, 5, 2, 0x12, 0x29), 0)
        self.assertEqual(ins.mnemonic, "por")

    def test_mmi_mult1_and_mflo1(self):
        self.assertEqual(decode(enc_r(0x1C, 4, 5, 2, 0, 0x18), 0).mnemonic, "mult1")
        self.assertEqual(decode(enc_r(0x1C, 0, 0, 2, 0, 0x12), 0).mnemonic, "mflo1")

    def test_pmfhl_variants(self):
        self.assertEqual(decode(enc_r(0x1C, 0, 0, 2, 0x00, 0x30), 0).mnemonic, "pmfhl.lw")
        self.assertEqual(decode(enc_r(0x1C, 0, 0, 2, 0x03, 0x30), 0).mnemonic, "pmfhl.lh")

    def test_madd_is_three_operand_on_ee(self):
        ins = decode(enc_r(0x1C, 4, 5, 2, 0, 0x00), 0)
        self.assertEqual(ins.mnemonic, "madd")

    def test_psllw_uses_sa(self):
        ins = decode(enc_r(0x1C, 0, 4, 2, 5, 0x3C), 0)
        self.assertEqual(ins.mnemonic, "psllw")
        self.assertEqual(ins.operands, ("$v0", "$a0", 5))

    def test_cache_instruction(self):
        """Opcode 0x2F is the EE cache-maintenance instruction.

        A generic MIPS decoder leaves this undecoded; it appears in the real
        binary inside sync-wrapped cache loops.
        """
        ins = decode(0xBD180000, 0x0010E808)
        self.assertTrue(ins.is_valid)
        self.assertEqual(ins.mnemonic, "cache")
        # The rt field selects the cache operation rather than naming a register.
        self.assertEqual(ins.operands, ("0x18", "0x0($t0)"))

    def test_mfsa_mtsa(self):
        self.assertEqual(decode(enc_r(0, 0, 0, 2, 0, 0x28), 0).mnemonic, "mfsa")
        self.assertEqual(decode(enc_r(0, 4, 0, 0, 0, 0x29), 0).mnemonic, "mtsa")


class TestCoprocessors(unittest.TestCase):
    def test_mfc1_mtc1(self):
        self.assertEqual(decode(enc_r(0x11, 0x00, 4, 2, 0, 0), 0).mnemonic, "mfc1")
        self.assertEqual(decode(enc_r(0x11, 0x04, 4, 2, 0, 0), 0).mnemonic, "mtc1")

    def test_add_s(self):
        ins = decode(enc_r(0x11, 0x10, 5, 4, 2, 0x00), 0)
        self.assertEqual(ins.mnemonic, "add.s")
        self.assertEqual(ins.operands, ("$f2", "$f4", "$f5"))

    def test_lwc1_swc1_use_fpr(self):
        ins = decode(enc_i(0x31, 29, 4, 0), 0)
        self.assertEqual(ins.mnemonic, "lwc1")
        self.assertEqual(ins.operands, ("$f4", "0x0($sp)"))

    def test_bc1t_is_branch(self):
        ins = decode(enc_i(0x11, 0x08, 1, 2), 0x100000)
        self.assertEqual(ins.mnemonic, "bc1t")
        self.assertTrue(ins.is_branch)

    def test_cop0_mfc0(self):
        ins = decode(enc_r(0x10, 0x00, 4, 12, 0, 0), 0)
        self.assertEqual(ins.mnemonic, "mfc0")
        self.assertEqual(ins.operands, ("$a0", "Status"))

    def test_cop2_qmfc2(self):
        ins = decode(enc_r(0x12, 0x01, 4, 5, 0, 0), 0)
        self.assertEqual(ins.mnemonic, "qmfc2")
        self.assertEqual(ins.operands, ("$a0", "$vf5"))

    def test_lqc2_sqc2_use_vf(self):
        self.assertEqual(decode(enc_i(0x36, 29, 4, 0), 0).operands, ("$vf4", "0x0($sp)"))
        self.assertEqual(decode(enc_i(0x3E, 29, 4, 0), 0).operands, ("$vf4", "0x0($sp)"))


class TestRoundTripStream(unittest.TestCase):
    def test_typical_prologue_epilogue(self):
        """A realistic leaf-function frame decodes to the expected sequence."""
        import struct

        words = [
            enc_i(0x09, 29, 29, -0x10),      # addiu $sp, $sp, -0x10
            enc_i(0x2B, 29, 31, 0x00),       # sw    $ra, 0x0($sp)
            enc_j(0x03, 0x00101000 >> 2),    # jal   0x00101000
            0x00000000,                      # nop (delay slot)
            enc_i(0x23, 29, 31, 0x00),       # lw    $ra, 0x0($sp)
            enc_r(0, 31, 0, 0, 0, 0x08),     # jr    $ra
            enc_i(0x09, 29, 29, 0x10),       # addiu $sp, $sp, 0x10
        ]
        data = struct.pack("<%dI" % len(words), *words)
        from ps2decomp.r5900 import decode_bytes

        ins = decode_bytes(data, 0x00100000)
        got = [i.mnemonic for i in ins]
        self.assertEqual(
            got, ["addiu", "sw", "jal", "nop", "lw", "jr", "addiu"]
        )
        self.assertEqual(ins[2].target, 0x00101000)
        self.assertTrue(ins[5].is_return)
        self.assertTrue(all(i.is_valid for i in ins))


if __name__ == "__main__":
    unittest.main(verbosity=2)
