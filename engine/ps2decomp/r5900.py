"""Disassembler for the Emotion Engine CPU (MIPS R5900), as used by the PS2.

The R5900 is a MIPS III core with Sony/Toshiba extensions, so a stock MIPS
disassembler mis-decodes a meaningful fraction of a real PS2 binary.  The pieces
that matter here and that generic decoders get wrong are:

* the MMI opcode space (opcode 0x1C) - 128-bit integer SIMD, with four nested
  sub-tables MMI0..MMI3 plus PMFHL/PMTHL and the shift group;
* 128-bit LQ/SQ load/store (opcodes 0x1E/0x1F);
* COP2, which on the EE is the VU0 macro-mode interface rather than a normal
  MIPS coprocessor 2;
* MIPS III 64-bit arithmetic, which the EE fully implements.

Everything decodes into a small `Instruction` value that records enough control
flow information (branch targets, call/return classification, delay slots) for
the function-discovery and splitting passes to work off.
"""

from __future__ import annotations

import dataclasses
import struct

# -- register names ------------------------------------------------------

GPR = [
    "$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3",
    "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7",
    "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
    "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
]

FPR = ["$f%d" % i for i in range(32)]
VF = ["$vf%d" % i for i in range(32)]
VI = ["$vi%d" % i for i in range(32)]

COP0_REG = {
    0: "Index", 1: "Random", 2: "EntryLo0", 3: "EntryLo1", 4: "Context",
    5: "PageMask", 6: "Wired", 8: "BadVAddr", 9: "Count", 10: "EntryHi",
    11: "Compare", 12: "Status", 13: "Cause", 14: "EPC", 15: "PRId",
    16: "Config", 23: "BadPAddr", 24: "Debug", 25: "Perf", 28: "TagLo",
    29: "TagHi", 30: "ErrorEPC",
}

# Register indices that carry meaning for control flow.
REG_ZERO = 0
REG_RA = 31
REG_SP = 29


# -- instruction classification -----------------------------------------

class Flags:
    """Bit flags describing what an instruction does, for the analysis passes."""

    NONE = 0
    BRANCH = 1 << 0        # PC-relative conditional branch
    JUMP = 1 << 1          # unconditional transfer (j, jr, b)
    CALL = 1 << 2          # jal / jalr / bgezal etc - writes $ra
    RETURN = 1 << 3        # jr $ra
    LIKELY = 1 << 4        # ...L branch: delay slot nullified when not taken
    LOAD = 1 << 5
    STORE = 1 << 6
    FLOAT = 1 << 7
    HAS_DELAY_SLOT = 1 << 8
    TRAP = 1 << 9
    INVALID = 1 << 10


@dataclasses.dataclass(frozen=True)
class Instruction:
    """One decoded machine instruction."""

    addr: int
    raw: int
    mnemonic: str
    operands: tuple = ()
    flags: int = Flags.NONE
    target: int = -1        # resolved branch/jump target, or -1
    # Decoded field values, kept for analysis passes that need raw registers.
    rs: int = 0
    rt: int = 0
    rd: int = 0
    imm: int = 0

    # -- predicates ---------------------------------------------------

    @property
    def is_branch(self) -> bool:
        return bool(self.flags & Flags.BRANCH)

    @property
    def is_jump(self) -> bool:
        return bool(self.flags & Flags.JUMP)

    @property
    def is_call(self) -> bool:
        return bool(self.flags & Flags.CALL)

    @property
    def is_return(self) -> bool:
        return bool(self.flags & Flags.RETURN)

    @property
    def is_likely(self) -> bool:
        return bool(self.flags & Flags.LIKELY)

    @property
    def has_delay_slot(self) -> bool:
        return bool(self.flags & Flags.HAS_DELAY_SLOT)

    @property
    def is_valid(self) -> bool:
        return not self.flags & Flags.INVALID

    @property
    def is_nop(self) -> bool:
        return self.raw == 0

    @property
    def ends_block(self) -> bool:
        """True when control does not fall through past the delay slot."""
        if self.is_return:
            return True
        if self.is_jump and not self.is_call:
            return True
        return False

    def format(self, symbolizer=None) -> str:
        """Render as assembly text; `symbolizer(addr)` may supply a label."""
        if not self.operands:
            return self.mnemonic
        ops = list(self.operands)
        if self.target >= 0 and symbolizer is not None:
            label = symbolizer(self.target)
            if label:
                ops[-1] = label
        return "%-10s %s" % (self.mnemonic, ", ".join(str(o) for o in ops))

    def __str__(self) -> str:
        return self.format()


# -- field helpers -------------------------------------------------------

def _op(w):     return (w >> 26) & 0x3F
def _rs(w):     return (w >> 21) & 0x1F
def _rt(w):     return (w >> 16) & 0x1F
def _rd(w):     return (w >> 11) & 0x1F
def _sa(w):     return (w >> 6) & 0x1F
def _funct(w):  return w & 0x3F
def _imm(w):    return w & 0xFFFF
def _simm(w):
    v = w & 0xFFFF
    return v - 0x10000 if v & 0x8000 else v
def _target(w): return w & 0x03FFFFFF


def _hex(value: int) -> str:
    """Format an immediate the way the EE assembler conventionally shows it."""
    if value < 0:
        return "-0x%x" % (-value)
    return "0x%x" % value


# -- opcode tables -------------------------------------------------------

# SPECIAL (opcode 0) funct -> (mnemonic, form)
# Forms: 'd,s,t' etc describe operand order using rd/rs/rt/sa.
_SPECIAL = {
    0x00: ("sll", "d,t,a"),
    0x02: ("srl", "d,t,a"),
    0x03: ("sra", "d,t,a"),
    0x04: ("sllv", "d,t,s"),
    0x06: ("srlv", "d,t,s"),
    0x07: ("srav", "d,t,s"),
    0x08: ("jr", "s"),
    0x09: ("jalr", "jalr"),
    0x0A: ("movz", "d,s,t"),
    0x0B: ("movn", "d,s,t"),
    0x0C: ("syscall", ""),
    0x0D: ("break", ""),
    0x0F: ("sync", ""),
    0x10: ("mfhi", "d"),
    0x11: ("mthi", "s"),
    0x12: ("mflo", "d"),
    0x13: ("mtlo", "s"),
    0x14: ("dsllv", "d,t,s"),
    0x16: ("dsrlv", "d,t,s"),
    0x17: ("dsrav", "d,t,s"),
    0x18: ("mult", "mult"),
    0x19: ("multu", "mult"),
    0x1A: ("div", "s,t"),
    0x1B: ("divu", "s,t"),
    0x1C: ("dmult", "s,t"),
    0x1D: ("dmultu", "s,t"),
    0x1E: ("ddiv", "s,t"),
    0x1F: ("ddivu", "s,t"),
    0x20: ("add", "d,s,t"),
    0x21: ("addu", "d,s,t"),
    0x22: ("sub", "d,s,t"),
    0x23: ("subu", "d,s,t"),
    0x24: ("and", "d,s,t"),
    0x25: ("or", "d,s,t"),
    0x26: ("xor", "d,s,t"),
    0x27: ("nor", "d,s,t"),
    0x28: ("mfsa", "d"),
    0x29: ("mtsa", "s"),
    0x2A: ("slt", "d,s,t"),
    0x2B: ("sltu", "d,s,t"),
    0x2C: ("dadd", "d,s,t"),
    0x2D: ("daddu", "d,s,t"),
    0x2E: ("dsub", "d,s,t"),
    0x2F: ("dsubu", "d,s,t"),
    0x30: ("tge", "s,t"),
    0x31: ("tgeu", "s,t"),
    0x32: ("tlt", "s,t"),
    0x33: ("tltu", "s,t"),
    0x34: ("teq", "s,t"),
    0x36: ("tne", "s,t"),
    0x38: ("dsll", "d,t,a"),
    0x3A: ("dsrl", "d,t,a"),
    0x3B: ("dsra", "d,t,a"),
    0x3C: ("dsll32", "d,t,a"),
    0x3E: ("dsrl32", "d,t,a"),
    0x3F: ("dsra32", "d,t,a"),
}

# REGIMM (opcode 1) rt -> mnemonic
_REGIMM = {
    0x00: "bltz", 0x01: "bgez", 0x02: "bltzl", 0x03: "bgezl",
    0x08: "tgei", 0x09: "tgeiu", 0x0A: "tlti", 0x0B: "tltiu",
    0x0C: "teqi", 0x0E: "tnei",
    0x10: "bltzal", 0x11: "bgezal", 0x12: "bltzall", 0x13: "bgezall",
    0x18: "mtsab", 0x19: "mtsah",
}

# Main opcode table for the simple I-type forms.
_ITYPE = {
    0x08: ("addi", "t,s,i"),
    0x09: ("addiu", "t,s,i"),
    0x0A: ("slti", "t,s,i"),
    0x0B: ("sltiu", "t,s,i"),
    0x0C: ("andi", "t,s,u"),
    0x0D: ("ori", "t,s,u"),
    0x0E: ("xori", "t,s,u"),
    0x18: ("daddi", "t,s,i"),
    0x19: ("daddiu", "t,s,i"),
}

# Loads and stores: opcode -> (mnemonic, is_store, is_float)
_MEM = {
    0x20: ("lb", False, False),
    0x21: ("lh", False, False),
    0x22: ("lwl", False, False),
    0x23: ("lw", False, False),
    0x24: ("lbu", False, False),
    0x25: ("lhu", False, False),
    0x26: ("lwr", False, False),
    0x27: ("lwu", False, False),
    0x28: ("sb", True, False),
    0x29: ("sh", True, False),
    0x2A: ("swl", True, False),
    0x2B: ("sw", True, False),
    0x2C: ("sdl", True, False),
    0x2D: ("sdr", True, False),
    0x2E: ("swr", True, False),
    0x30: ("lwc0", False, False),
    0x31: ("lwc1", False, True),
    0x33: ("pref", False, False),
    # The EE's cache-maintenance instruction. The `rt` field is an operation
    # selector rather than a register, so it is rendered numerically.
    0x2F: ("cache", False, False),
    0x36: ("lqc2", False, False),
    0x37: ("ld", False, False),
    0x39: ("swc1", True, True),
    0x3E: ("sqc2", True, False),
    0x3F: ("sd", True, False),
    0x1A: ("ldl", False, False),
    0x1B: ("ldr", False, False),
    0x1E: ("lq", False, False),
    0x1F: ("sq", True, False),
}

# MMI (opcode 0x1C) funct -> mnemonic, for the entries that are not sub-tables.
_MMI = {
    0x00: ("madd", "d,s,t"),
    0x01: ("maddu", "d,s,t"),
    0x04: ("plzcw", "d,s"),
    0x10: ("mfhi1", "d"),
    0x11: ("mthi1", "s"),
    0x12: ("mflo1", "d"),
    0x13: ("mtlo1", "s"),
    0x18: ("mult1", "d,s,t"),
    0x19: ("multu1", "d,s,t"),
    0x1A: ("div1", "s,t"),
    0x1B: ("divu1", "s,t"),
    0x20: ("madd1", "d,s,t"),
    0x21: ("maddu1", "d,s,t"),
    0x30: ("pmfhl", "pmfhl"),
    0x31: ("pmthl", "pmthl"),
    0x34: ("psllh", "d,t,a"),
    0x36: ("psrlh", "d,t,a"),
    0x37: ("psrah", "d,t,a"),
    0x3C: ("psllw", "d,t,a"),
    0x3E: ("psrlw", "d,t,a"),
    0x3F: ("psraw", "d,t,a"),
}

# MMI0 (funct 0x08), selected by bits 10:6
_MMI0 = {
    0x00: "paddw",  0x01: "psubw",  0x02: "pcgtw",  0x03: "pmaxw",
    0x04: "paddh",  0x05: "psubh",  0x06: "pcgth",  0x07: "pmaxh",
    0x08: "paddb",  0x09: "psubb",  0x0A: "pcgtb",
    0x10: "paddsw", 0x11: "psubsw", 0x12: "pextlw", 0x13: "ppacw",
    0x14: "paddsh", 0x15: "psubsh", 0x16: "pextlh", 0x17: "ppach",
    0x18: "paddsb", 0x19: "psubsb", 0x1A: "pextlb", 0x1B: "ppacb",
    0x1E: "pext5",  0x1F: "ppac5",
}

# MMI1 (funct 0x28)
_MMI1 = {
    0x01: "pabsw",  0x02: "pceqw",  0x03: "pminw",
    0x04: "padsbh", 0x05: "pabsh",  0x06: "pceqh",  0x07: "pminh",
    0x0A: "pceqb",
    0x10: "padduw", 0x11: "psubuw", 0x12: "pextuw",
    0x14: "padduh", 0x15: "psubuh", 0x16: "pextuh",
    0x18: "paddub", 0x19: "psubub", 0x1A: "pextub", 0x1B: "qfsrv",
}

# MMI2 (funct 0x09)
_MMI2 = {
    0x00: "pmaddw", 0x02: "psllvw", 0x03: "psrlvw",
    0x04: "pmsubw", 0x08: "pmfhi",  0x09: "pmflo",  0x0A: "pinth",
    0x0C: "pmultw", 0x0D: "pdivw",  0x0E: "pcpyld",
    0x10: "pmaddh", 0x11: "phmadh", 0x12: "pand",   0x13: "pxor",
    0x14: "pmsubh", 0x15: "phmsbh",
    0x1A: "pexeh",  0x1B: "prevh",  0x1C: "pmulth", 0x1D: "pdivbw",
    0x1E: "pexew",  0x1F: "prot3w",
}

# MMI3 (funct 0x29)
_MMI3 = {
    0x00: "pmadduw", 0x03: "psravw",
    0x08: "pmthi",   0x09: "pmtlo",  0x0A: "pinteh",
    0x0C: "pmultuw", 0x0D: "pdivuw", 0x0E: "pcpyud",
    0x12: "por",     0x13: "pnor",
    0x1A: "pexch",   0x1B: "pcpyh",
    0x1E: "pexcw",
}

_PMFHL = {0x00: "pmfhl.lw", 0x01: "pmfhl.uw", 0x02: "pmfhl.slw",
          0x03: "pmfhl.lh", 0x04: "pmfhl.sh"}

# COP1 (FPU) funct for the S format.
_COP1_S = {
    0x00: ("add.s", "d,s,t"), 0x01: ("sub.s", "d,s,t"),
    0x02: ("mul.s", "d,s,t"), 0x03: ("div.s", "d,s,t"),
    0x04: ("sqrt.s", "d,t"),  0x05: ("abs.s", "d,s"),
    0x06: ("mov.s", "d,s"),   0x07: ("neg.s", "d,s"),
    0x16: ("rsqrt.s", "d,s,t"),
    0x18: ("adda.s", "s,t"),  0x19: ("suba.s", "s,t"),
    0x1A: ("mula.s", "s,t"),
    0x1C: ("madd.s", "d,s,t"), 0x1D: ("msub.s", "d,s,t"),
    0x1E: ("madda.s", "s,t"),  0x1F: ("msuba.s", "s,t"),
    0x24: ("cvt.w.s", "d,s"),
    0x28: ("max.s", "d,s,t"),  0x29: ("min.s", "d,s,t"),
    0x30: ("c.f.s", "s,t"),    0x32: ("c.eq.s", "s,t"),
    0x34: ("c.lt.s", "s,t"),   0x36: ("c.le.s", "s,t"),
}

# VU0 macro-mode (COP2) operations. Special1 is selected by funct (bits 5..0)
# below 0x3C; special2 by bits 1..0 and 10..6 together, as (bits 10..6) << 2 |
# (bits 1..0). Unlisted slots are reserved and decode as plain "cop2".
_COP2_SPECIAL1 = {
    0x00: "vaddx", 0x01: "vaddy", 0x02: "vaddz", 0x03: "vaddw",
    0x04: "vsubx", 0x05: "vsuby", 0x06: "vsubz", 0x07: "vsubw",
    0x08: "vmaddx", 0x09: "vmaddy", 0x0A: "vmaddz", 0x0B: "vmaddw",
    0x0C: "vmsubx", 0x0D: "vmsuby", 0x0E: "vmsubz", 0x0F: "vmsubw",
    0x10: "vmaxx", 0x11: "vmaxy", 0x12: "vmaxz", 0x13: "vmaxw",
    0x14: "vminix", 0x15: "vminiy", 0x16: "vminiz", 0x17: "vminiw",
    0x18: "vmulx", 0x19: "vmuly", 0x1A: "vmulz", 0x1B: "vmulw",
    0x1C: "vmulq", 0x1D: "vmaxi", 0x1E: "vmuli", 0x1F: "vminii",
    0x20: "vaddq", 0x21: "vmaddq", 0x22: "vaddi", 0x23: "vmaddi",
    0x24: "vsubq", 0x25: "vmsubq", 0x26: "vsubi", 0x27: "vmsubi",
    0x28: "vadd", 0x29: "vmadd", 0x2A: "vmul", 0x2B: "vmax",
    0x2C: "vsub", 0x2D: "vmsub", 0x2E: "vopmsub", 0x2F: "vmini",
    0x30: "viadd", 0x31: "visub", 0x32: "viaddi",
    0x34: "viand", 0x35: "vior",
    0x38: "vcallms", 0x39: "vcallmsr",
}

_COP2_SPECIAL2 = {
    0x00: "vaddax", 0x01: "vadday", 0x02: "vaddaz", 0x03: "vaddaw",
    0x04: "vsubax", 0x05: "vsubay", 0x06: "vsubaz", 0x07: "vsubaw",
    0x08: "vmaddax", 0x09: "vmadday", 0x0A: "vmaddaz", 0x0B: "vmaddaw",
    0x0C: "vmsubax", 0x0D: "vmsubay", 0x0E: "vmsubaz", 0x0F: "vmsubaw",
    0x10: "vitof0", 0x11: "vitof4", 0x12: "vitof12", 0x13: "vitof15",
    0x14: "vftoi0", 0x15: "vftoi4", 0x16: "vftoi12", 0x17: "vftoi15",
    0x18: "vmulax", 0x19: "vmulay", 0x1A: "vmulaz", 0x1B: "vmulaw",
    0x1C: "vmulaq", 0x1D: "vabs", 0x1E: "vmulai", 0x1F: "vclipw",
    0x20: "vaddaq", 0x21: "vmaddaq", 0x22: "vaddai", 0x23: "vmaddai",
    0x24: "vsubaq", 0x25: "vmsubaq", 0x26: "vsubai", 0x27: "vmsubai",
    0x28: "vadda", 0x29: "vmadda", 0x2A: "vmula",
    0x2C: "vsuba", 0x2D: "vmsuba", 0x2E: "vopmula", 0x2F: "vnop",
    0x30: "vmove", 0x31: "vmr32",
    0x34: "vlqi", 0x35: "vsqi", 0x36: "vlqd", 0x37: "vsqd",
    0x38: "vdiv", 0x39: "vsqrt", 0x3A: "vrsqrt", 0x3B: "vwaitq",
    0x3C: "vmtir", 0x3D: "vmfir", 0x3E: "vilwr", 0x3F: "viswr",
    0x40: "vrnext", 0x41: "vrget", 0x42: "vrinit", 0x43: "vrxor",
}

# Special2 operations without a destination mask.
_COP2_UNMASKED = {"vclipw", "vnop", "vdiv", "vsqrt", "vrsqrt", "vwaitq", "vmtir",
                  "vrinit", "vrxor"}


class Disassembler:
    """Decodes R5900 instructions.

    Stateless by design: `decode` is a pure function of (word, address) so the
    analysis passes can decode the same address repeatedly without bookkeeping.
    """

    # -- public API ---------------------------------------------------

    def decode(self, word: int, addr: int = 0) -> Instruction:
        word &= 0xFFFFFFFF
        if word == 0:
            return Instruction(addr, word, "nop")
        op = _op(word)
        handler = getattr(self, "_op_%02x" % op, None)
        if handler is not None:
            result = handler(word, addr)
            if result is not None:
                return result
        if op in _ITYPE:
            return self._itype(word, addr, *_ITYPE[op])
        if op in _MEM:
            return self._mem(word, addr, *_MEM[op])
        return self._invalid(word, addr)

    def decode_bytes(self, data: bytes, base_addr: int = 0):
        """Decode a little-endian instruction stream into a list."""
        count = len(data) // 4
        words = struct.unpack("<%dI" % count, data[: count * 4])
        return [self.decode(w, base_addr + i * 4) for i, w in enumerate(words)]

    # -- helpers ------------------------------------------------------

    def _invalid(self, word: int, addr: int) -> Instruction:
        return Instruction(addr, word, ".word", ("0x%08x" % word,), Flags.INVALID)

    def _branch_target(self, word: int, addr: int) -> int:
        return addr + 4 + (_simm(word) << 2)

    def _jump_target(self, word: int, addr: int) -> int:
        return ((addr + 4) & 0xF0000000) | (_target(word) << 2)

    def _format_operands(self, word: int, form: str, regs=GPR, cop1: bool = False):
        """Build operand tuple from a compact form string.

        COP1 R-format packs its register fields differently from the integer
        unit: fd sits in the shift-amount field and fs in the rd field, so the
        `cop1` flag switches to that mapping.
        """
        pick_d = _sa if cop1 else _rd
        pick_s = _rd if cop1 else _rs
        out = []
        for part in form.split(","):
            if part == "d":
                out.append(regs[pick_d(word)])
            elif part == "s":
                out.append(regs[pick_s(word)])
            elif part == "t":
                out.append(regs[_rt(word)])
            elif part == "a":
                out.append(_sa(word))
            elif part == "i":
                out.append(_hex(_simm(word)))
            elif part == "u":
                out.append(_hex(_imm(word)))
        return tuple(out)

    # -- opcode 0x00: SPECIAL -----------------------------------------

    def _op_00(self, word: int, addr: int):
        funct = _funct(word)
        entry = _SPECIAL.get(funct)
        if entry is None:
            return self._invalid(word, addr)
        mnem, form = entry
        rs, rt, rd = _rs(word), _rt(word), _rd(word)

        if funct == 0x08:  # jr
            flags = Flags.JUMP | Flags.HAS_DELAY_SLOT
            if rs == REG_RA:
                flags |= Flags.RETURN
            return Instruction(addr, word, "jr", (GPR[rs],), flags, rs=rs)

        if funct == 0x09:  # jalr
            # The two-operand form is only printed when rd is not the default $ra.
            ops = (GPR[rd], GPR[rs]) if rd != REG_RA else (GPR[rs],)
            return Instruction(
                addr, word, "jalr", ops,
                Flags.JUMP | Flags.CALL | Flags.HAS_DELAY_SLOT, rs=rs, rd=rd,
            )

        if funct in (0x18, 0x19):  # mult/multu: rd is optional on the EE
            if rd:
                ops = (GPR[rd], GPR[rs], GPR[rt])
            else:
                ops = (GPR[rs], GPR[rt])
            return Instruction(addr, word, mnem, ops, rs=rs, rt=rt, rd=rd)

        if funct in (0x0C, 0x0D, 0x0F):  # syscall/break/sync
            return Instruction(addr, word, mnem)

        # `sll $zero, $zero, 0` is canonical nop and was handled earlier; treat
        # the remaining shift-by-zero encodings normally.
        flags = Flags.TRAP if 0x30 <= funct <= 0x36 else Flags.NONE
        return Instruction(
            addr, word, mnem, self._format_operands(word, form),
            flags, rs=rs, rt=rt, rd=rd, imm=_sa(word),
        )

    # -- opcode 0x01: REGIMM ------------------------------------------

    def _op_01(self, word: int, addr: int):
        rt = _rt(word)
        mnem = _REGIMM.get(rt)
        if mnem is None:
            return self._invalid(word, addr)
        rs = _rs(word)
        if mnem in ("mtsab", "mtsah"):
            return Instruction(addr, word, mnem, (GPR[rs], _hex(_imm(word))), rs=rs)
        if mnem.startswith("t"):  # trap immediate
            return Instruction(
                addr, word, mnem, (GPR[rs], _hex(_simm(word))), Flags.TRAP, rs=rs
            )
        target = self._branch_target(word, addr)
        flags = Flags.BRANCH | Flags.HAS_DELAY_SLOT
        if mnem.endswith("l"):
            flags |= Flags.LIKELY
        if "al" in mnem:
            flags |= Flags.CALL
        return Instruction(
            addr, word, mnem, (GPR[rs], "0x%08x" % target), flags, target, rs=rs
        )

    # -- opcodes 0x02/0x03: J / JAL -----------------------------------

    def _op_02(self, word: int, addr: int):
        target = self._jump_target(word, addr)
        return Instruction(
            addr, word, "j", ("0x%08x" % target,),
            Flags.JUMP | Flags.HAS_DELAY_SLOT, target,
        )

    def _op_03(self, word: int, addr: int):
        target = self._jump_target(word, addr)
        return Instruction(
            addr, word, "jal", ("0x%08x" % target,),
            Flags.JUMP | Flags.CALL | Flags.HAS_DELAY_SLOT, target,
        )

    # -- opcodes 0x04..0x07, 0x14..0x17: branches ---------------------

    def _branch(self, word: int, addr: int, mnem: str, two_reg: bool):
        target = self._branch_target(word, addr)
        rs, rt = _rs(word), _rt(word)
        flags = Flags.BRANCH | Flags.HAS_DELAY_SLOT
        if mnem.endswith("l"):
            flags |= Flags.LIKELY
        # `beq $zero, $zero, x` is an unconditional branch; the assembler spells
        # it `b`, and treating it as a jump keeps block splitting accurate.
        if mnem == "beq" and rs == REG_ZERO and rt == REG_ZERO:
            return Instruction(
                addr, word, "b", ("0x%08x" % target,),
                Flags.JUMP | Flags.HAS_DELAY_SLOT, target,
            )
        ops = (GPR[rs], GPR[rt], "0x%08x" % target) if two_reg else (GPR[rs], "0x%08x" % target)
        return Instruction(addr, word, mnem, ops, flags, target, rs=rs, rt=rt)

    def _op_04(self, word, addr): return self._branch(word, addr, "beq", True)
    def _op_05(self, word, addr): return self._branch(word, addr, "bne", True)
    def _op_06(self, word, addr): return self._branch(word, addr, "blez", False)
    def _op_07(self, word, addr): return self._branch(word, addr, "bgtz", False)
    def _op_14(self, word, addr): return self._branch(word, addr, "beql", True)
    def _op_15(self, word, addr): return self._branch(word, addr, "bnel", True)
    def _op_16(self, word, addr): return self._branch(word, addr, "blezl", False)
    def _op_17(self, word, addr): return self._branch(word, addr, "bgtzl", False)

    # -- opcode 0x0F: LUI ---------------------------------------------

    def _op_0f(self, word: int, addr: int):
        rt = _rt(word)
        return Instruction(
            addr, word, "lui", (GPR[rt], _hex(_imm(word))), rt=rt, imm=_imm(word)
        )

    # -- opcode 0x10: COP0 --------------------------------------------

    def _op_10(self, word: int, addr: int):
        rs, rt, rd = _rs(word), _rt(word), _rd(word)
        if rs == 0x00:
            return Instruction(addr, word, "mfc0", (GPR[rt], COP0_REG.get(rd, "$%d" % rd)), rt=rt, rd=rd)
        if rs == 0x04:
            return Instruction(addr, word, "mtc0", (GPR[rt], COP0_REG.get(rd, "$%d" % rd)), rt=rt, rd=rd)
        if rs == 0x08:  # BC0
            target = self._branch_target(word, addr)
            mnem = {0: "bc0f", 1: "bc0t", 2: "bc0fl", 3: "bc0tl"}.get(rt, "bc0?")
            return Instruction(
                addr, word, mnem, ("0x%08x" % target,),
                Flags.BRANCH | Flags.HAS_DELAY_SLOT, target,
            )
        if rs == 0x10:
            mnem = {0x01: "tlbr", 0x02: "tlbwi", 0x06: "tlbwr", 0x08: "tlbp",
                    0x18: "eret", 0x38: "ei", 0x39: "di"}.get(_funct(word))
            if mnem:
                flags = Flags.JUMP if mnem == "eret" else Flags.NONE
                return Instruction(addr, word, mnem, (), flags)
        return self._invalid(word, addr)

    # -- opcode 0x11: COP1 (FPU) --------------------------------------

    def _op_11(self, word: int, addr: int):
        rs, rt, rd = _rs(word), _rt(word), _rd(word)
        if rs == 0x00:
            return Instruction(addr, word, "mfc1", (GPR[rt], FPR[rd]), Flags.FLOAT, rt=rt, rd=rd)
        if rs == 0x02:
            return Instruction(addr, word, "cfc1", (GPR[rt], "$%d" % rd), Flags.FLOAT, rt=rt, rd=rd)
        if rs == 0x04:
            return Instruction(addr, word, "mtc1", (GPR[rt], FPR[rd]), Flags.FLOAT, rt=rt, rd=rd)
        if rs == 0x06:
            return Instruction(addr, word, "ctc1", (GPR[rt], "$%d" % rd), Flags.FLOAT, rt=rt, rd=rd)
        if rs == 0x08:  # BC1
            target = self._branch_target(word, addr)
            mnem = {0: "bc1f", 1: "bc1t", 2: "bc1fl", 3: "bc1tl"}.get(rt, "bc1?")
            flags = Flags.BRANCH | Flags.HAS_DELAY_SLOT | Flags.FLOAT
            if rt >= 2:
                flags |= Flags.LIKELY
            return Instruction(addr, word, mnem, ("0x%08x" % target,), flags, target)
        if rs == 0x10:  # single precision
            entry = _COP1_S.get(_funct(word))
            if entry is None:
                return self._invalid(word, addr)
            mnem, form = entry
            return Instruction(
                addr, word, mnem,
                self._format_operands(word, form, FPR, cop1=True), Flags.FLOAT,
            )
        if rs == 0x14:  # fixed point -> single
            if _funct(word) == 0x20:
                return Instruction(
                    addr, word, "cvt.s.w", (FPR[_sa(word)], FPR[_rd(word)]), Flags.FLOAT
                )
        return self._invalid(word, addr)

    # -- opcode 0x12: COP2 (VU0 macro mode) ---------------------------

    def _op_12(self, word: int, addr: int):
        rs, rt, rd = _rs(word), _rt(word), _rd(word)
        # Data movement between the EE core and VU0.
        simple = {0x01: "qmfc2", 0x02: "cfc2", 0x05: "qmtc2", 0x06: "ctc2"}
        if rs in simple:
            reg = VF[rd] if rs in (0x01, 0x05) else VI[rd]
            return Instruction(addr, word, simple[rs], (GPR[rt], reg), rt=rt, rd=rd)
        if rs == 0x08:  # BC2
            target = self._branch_target(word, addr)
            mnem = {0: "bc2f", 1: "bc2t", 2: "bc2fl", 3: "bc2tl"}.get(rt, "bc2?")
            return Instruction(
                addr, word, mnem, ("0x%08x" % target,),
                Flags.BRANCH | Flags.HAS_DELAY_SLOT, target,
            )
        if rs & 0x10:
            return self._cop2_special(word, addr)
        return self._invalid(word, addr)

    def _cop2_special(self, word: int, addr: int):
        """Decode VU0 macro-mode operations.

        Special1 (funct below 0x3C) is the FMAC group, named with their
        destination mask: `vaddx.xyz`. Special2 (funct 0x3C..0x3F) indexes a
        second table by bits 10..6 and 1..0, where the accumulator forms,
        conversions, moves and the divide unit live. Every field is read from the
        raw word by whatever executes the instruction; the operands here are for
        reading.
        """
        funct = _funct(word)
        dest = (word >> 21) & 0xF
        mask = "".join(c for c, bit in zip("xyzw", (8, 4, 2, 1)) if dest & bit) or "xyzw"
        fd, fs, ft = _sa(word), _rd(word), _rt(word)
        if funct >= 0x3C:
            index = (word & 0x3) | ((word >> 4) & 0x7C)
            base = _COP2_SPECIAL2.get(index)
            if base is None:
                return Instruction(addr, word, "cop2", ("0x%07x" % (word & 0x1FFFFFF),))
            if base in _COP2_UNMASKED:
                return Instruction(addr, word, base, (VF[fs], VF[ft]))
            return Instruction(addr, word, "%s.%s" % (base, mask), (VF[fs], VF[ft]))
        base = _COP2_SPECIAL1.get(funct)
        if base is None:
            return Instruction(addr, word, "cop2", ("0x%07x" % (word & 0x1FFFFFF),))
        if base in ("viadd", "visub", "viaddi", "viand", "vior", "vcallms", "vcallmsr"):
            return Instruction(addr, word, base, (VI[fd], VI[fs], VI[ft]))
        return Instruction(
            addr, word, "%s.%s" % (base, mask), (VF[fd], VF[fs], VF[ft])
        )

    # -- opcode 0x1C: MMI ---------------------------------------------

    def _op_1c(self, word: int, addr: int):
        funct = _funct(word)
        sub = _sa(word)
        rs, rt, rd = _rs(word), _rt(word), _rd(word)

        if funct == 0x08:  # MMI0
            mnem = _MMI0.get(sub)
            return (
                Instruction(addr, word, mnem, (GPR[rd], GPR[rs], GPR[rt]), rs=rs, rt=rt, rd=rd)
                if mnem else self._invalid(word, addr)
            )
        if funct == 0x28:  # MMI1
            mnem = _MMI1.get(sub)
            return (
                Instruction(addr, word, mnem, (GPR[rd], GPR[rs], GPR[rt]), rs=rs, rt=rt, rd=rd)
                if mnem else self._invalid(word, addr)
            )
        if funct == 0x09:  # MMI2
            mnem = _MMI2.get(sub)
            if mnem is None:
                return self._invalid(word, addr)
            if mnem in ("pmfhi", "pmflo"):
                return Instruction(addr, word, mnem, (GPR[rd],), rd=rd)
            if mnem == "prot3w":
                return Instruction(addr, word, mnem, (GPR[rd], GPR[rt]), rt=rt, rd=rd)
            return Instruction(addr, word, mnem, (GPR[rd], GPR[rs], GPR[rt]), rs=rs, rt=rt, rd=rd)
        if funct == 0x29:  # MMI3
            mnem = _MMI3.get(sub)
            if mnem is None:
                return self._invalid(word, addr)
            if mnem in ("pmthi", "pmtlo"):
                return Instruction(addr, word, mnem, (GPR[rs],), rs=rs)
            if mnem == "pcpyh":
                return Instruction(addr, word, mnem, (GPR[rd], GPR[rt]), rt=rt, rd=rd)
            return Instruction(addr, word, mnem, (GPR[rd], GPR[rs], GPR[rt]), rs=rs, rt=rt, rd=rd)

        entry = _MMI.get(funct)
        if entry is None:
            return self._invalid(word, addr)
        mnem, form = entry
        if form == "pmfhl":
            name = _PMFHL.get(sub)
            return (
                Instruction(addr, word, name, (GPR[rd],), rd=rd)
                if name else self._invalid(word, addr)
            )
        if form == "pmthl":
            return Instruction(addr, word, "pmthl.lw", (GPR[rs],), rs=rs)
        return Instruction(
            addr, word, mnem, self._format_operands(word, form),
            rs=rs, rt=rt, rd=rd, imm=_sa(word),
        )

    # -- generic I-type and memory forms -------------------------------

    def _itype(self, word: int, addr: int, mnem: str, form: str):
        rs, rt = _rs(word), _rt(word)
        # `addiu $rt, $zero, imm` is the canonical li; keeping the real mnemonic
        # avoids surprises when the output is fed back to an assembler.
        return Instruction(
            addr, word, mnem, self._format_operands(word, form),
            rs=rs, rt=rt, imm=_simm(word),
        )

    def _mem(self, word: int, addr: int, mnem: str, is_store: bool, is_float: bool):
        rs, rt = _rs(word), _rt(word)
        if is_float:
            reg = FPR[rt]
        elif mnem in ("lqc2", "sqc2"):
            reg = VF[rt]
        elif mnem in ("pref", "cache"):
            # These encode an operation selector in `rt`, not a register.
            reg = str(rt) if mnem == "pref" else _hex(rt)
        else:
            reg = GPR[rt]
        flags = Flags.STORE if is_store else Flags.LOAD
        if is_float:
            flags |= Flags.FLOAT
        offset = _simm(word)
        return Instruction(
            addr, word, mnem, (reg, "%s(%s)" % (_hex(offset), GPR[rs])),
            flags, rs=rs, rt=rt, imm=offset,
        )


# A module-level instance is convenient and safe because the class is stateless.
DISASM = Disassembler()


def decode(word: int, addr: int = 0) -> Instruction:
    return DISASM.decode(word, addr)


def decode_bytes(data: bytes, base_addr: int = 0):
    return DISASM.decode_bytes(data, base_addr)
