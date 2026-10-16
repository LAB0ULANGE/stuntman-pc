"""Disassembler for PlayStation 2 Vector Unit microcode.

The VUs are not MIPS cores and share nothing with the EE instruction set, so the
R5900 decoder cannot read this code at all - and worse, VU words decode as
*plausible* MIPS about 96% of the time, so nothing short of a dedicated decoder
will do.

Microcode is stored as 8-byte bundles holding two instructions issued together:

    bytes 0-3   lower instruction - loads, stores, integer ALU, branches
    bytes 4-7   upper instruction - the floating-point FMAC pipeline

The upper word also carries the bundle's control flags in its top bits:

    I  bit 31   the lower word is a 32-bit float immediate, not an instruction
    E  bit 30   end of microprogram (one more bundle executes after it)
    M  bit 29   (VU0) interlock with the EE's COP2 transfers
    D  bit 28   debug break
    T  bit 27   debug halt

Bits 24..21 are the destination field mask, naming which of the x/y/z/w lanes a
result is written to; it is rendered as a suffix, so `addx.xy` writes only x and y.

The opcode tables are the hardware's, as the host's micro-mode interpreter
(engine/host/src/vu.cpp) runs them and PCSX2 lays them out:

  * upper: bits 5..0. 0x00-0x2F are the arithmetic forms; 0x3C-0x3F extend into
    bits 10..6, the "special" index being (bits 1..0) | (bits 10..6) << 2.
  * lower: bits 31..25. 0x40 is the special group, named by bits 5..0, which
    extends at 0x3C-0x3F into bits 10..6 the same way as the upper does.

Each decoded instruction also says which registers it reads and writes, as
strings: "vf5x" (one lane of a vector register), "vi3", "ACCx", "Q", "P", "I",
"R", "MAC", "STATUS", "CLIP". The hazard scanner (engine/tools/vuhazards.py)
works from those.

Coverage is reported rather than assumed: `decode_bundle` marks anything it cannot
identify, and `engine/tools/vudis.py --stats` measures how much of a real microcode blob
decodes.  Guessing at an encoding would be worse than admitting the gap.
"""

from __future__ import annotations

import dataclasses
import struct

VF = ["vf%02d" % i for i in range(32)]
VI = ["vi%02d" % i for i in range(32)]

# Distinctive padding encodings, which is what makes microcode findable in a
# stripped image at all.
LOWER_NOP = 0x8000033C
UPPER_NOP_BASE = 0x000002FF

# Bundle control flags, in the upper word.
FLAG_I = 1 << 31
FLAG_E = 1 << 30
FLAG_M = 1 << 29
FLAG_D = 1 << 28
FLAG_T = 1 << 27

_BC = "xyzw"
_LANES = (("x", 8), ("y", 4), ("z", 2), ("w", 1))


def _dest(word: int) -> str:
    """Render the destination field mask as a lane suffix."""
    mask = (word >> 21) & 0xF
    # Bit 24 is x, descending to bit 21 for w.
    lanes = "".join(c for c, bit in _LANES if mask & bit)
    return lanes or "none"


def _vf(reg: int, lanes: str) -> set:
    """A vector register's lanes; vf0 is a constant and never a dependency."""
    if reg == 0:
        return set()
    return {"vf%d%s" % (reg, lane) for lane in lanes}


def _vi(reg: int) -> set:
    return {"vi%d" % reg} if reg else set()


def _acc(lanes: str) -> set:
    return {"ACC" + lane for lane in lanes}


# -- upper (FMAC) opcodes ------------------------------------------------
#
# Bits 5..0. 0x00..0x1B come in broadcast groups of four: the low two bits select
# which lane of ft is broadcast across the operation.
_UPPER_BC = {
    0x00: "add", 0x04: "sub", 0x08: "madd", 0x0C: "msub",
    0x10: "max", 0x14: "mini", 0x18: "mul",
}

# Forms whose second operand is the Q or I register.
_UPPER_QI = {
    0x1C: ("mulq", "Q"), 0x1D: ("maxi", "I"), 0x1E: ("muli", "I"), 0x1F: ("minii", "I"),
    0x20: ("addq", "Q"), 0x21: ("maddq", "Q"), 0x22: ("addi", "I"), 0x23: ("maddi", "I"),
    0x24: ("subq", "Q"), 0x25: ("msubq", "Q"), 0x26: ("subi", "I"), 0x27: ("msubi", "I"),
}

# Plain three-register forms.
_UPPER_REG = {
    0x28: "add", 0x29: "madd", 0x2A: "mul", 0x2B: "max",
    0x2C: "sub", 0x2D: "msub", 0x2E: "opmsub", 0x2F: "mini",
}

# The special index (bits 1..0 | bits 10..6 << 2) for opcodes 0x3C..0x3F.
_UPPER_SPECIAL_BC = {0x00: "adda", 0x04: "suba", 0x08: "madda", 0x0C: "msuba", 0x18: "mula"}
_UPPER_SPECIAL_CONVERT = {
    0x10: "itof0", 0x11: "itof4", 0x12: "itof12", 0x13: "itof15",
    0x14: "ftoi0", 0x15: "ftoi4", 0x16: "ftoi12", 0x17: "ftoi15",
    0x1D: "abs",
}
_UPPER_SPECIAL_QI = {
    0x1C: ("mulaq", "Q"), 0x1E: ("mulai", "I"),
    0x20: ("addaq", "Q"), 0x21: ("maddaq", "Q"), 0x22: ("addai", "I"), 0x23: ("maddai", "I"),
    0x24: ("subaq", "Q"), 0x25: ("msubaq", "Q"), 0x26: ("subai", "I"), 0x27: ("msubai", "I"),
}
_UPPER_SPECIAL_REG = {
    0x28: "adda", 0x29: "madda", 0x2A: "mula", 0x2C: "suba", 0x2D: "msuba", 0x2E: "opmula",
}

# Which forms read the accumulator (the multiply-adds) or leave the flags alone.
_READS_ACC = ("madd", "msub", "opmsub")
_NO_FLAGS = ("max", "mini", "abs", "itof", "ftoi")


# -- lower opcodes -------------------------------------------------------
#
# Bits 31..25 - seven bits, not six: 0x40 (bit 31 set, the rest clear) selects
# the special group, named by bits 5..0, where the integer ALU, the transfers
# between register files and the elementary function unit live.
LOWER_SPECIAL_GROUP = 0x40

_LOWER_MAIN = {
    0x00: "lq", 0x01: "sq", 0x04: "ilw", 0x05: "isw",
    0x08: "iaddiu", 0x09: "isubiu",
    0x10: "fceq", 0x11: "fcset", 0x12: "fcand", 0x13: "fcor",
    0x14: "fseq", 0x15: "fsset", 0x16: "fsand", 0x17: "fsor",
    0x18: "fmeq", 0x1A: "fmand", 0x1B: "fmor", 0x1C: "fcget",
    0x20: "b", 0x21: "bal", 0x24: "jr", 0x25: "jalr",
    0x28: "ibeq", 0x29: "ibne",
    0x2C: "ibltz", 0x2D: "ibgtz", 0x2E: "iblez", 0x2F: "ibgez",
}

# Special group, by bits 5..0.
_LOWER_SPECIAL = {
    0x30: "iadd", 0x31: "isub", 0x32: "iaddi", 0x34: "iand", 0x35: "ior",
}

# Within the special group, 0x3C..0x3F extend into bits 10..6; the index is
# (bits 1..0) | (bits 10..6) << 2.
_LOWER_SPECIAL_EXT = {
    0x30: "move", 0x31: "mr32",
    0x34: "lqi", 0x35: "sqi", 0x36: "lqd", 0x37: "sqd",
    0x38: "div", 0x39: "sqrt", 0x3A: "rsqrt", 0x3B: "waitq",
    0x3C: "mtir", 0x3D: "mfir", 0x3E: "ilwr", 0x3F: "iswr",
    0x40: "rnext", 0x41: "rget", 0x42: "rinit", 0x43: "rxor",
    0x64: "mfp", 0x68: "xtop", 0x69: "xitop", 0x6C: "xgkick",
    0x70: "esadd", 0x71: "ersadd", 0x72: "eleng", 0x73: "erleng",
    0x74: "eatanxy", 0x75: "eatanxz", 0x76: "esum",
    0x78: "esqrt", 0x79: "ersqrt", 0x7A: "ercpr", 0x7B: "waitp",
    0x7C: "esin", 0x7D: "eatan", 0x7E: "eexp",
}

# Latencies in cycles, for the hazard scanner: when a result can be read.
LATENCY_Q = {"div": 7, "sqrt": 7, "rsqrt": 13}
LATENCY_P = {
    "esadd": 11, "ersadd": 18, "eleng": 18, "erleng": 24, "eatanxy": 54, "eatanxz": 54,
    "esum": 12, "esqrt": 12, "ersqrt": 18, "ercpr": 12, "esin": 29, "eatan": 54, "eexp": 44,
}
LATENCY_FMAC = 4


@dataclasses.dataclass(frozen=True)
class VuInstruction:
    """One half of a bundle."""

    raw: int
    mnemonic: str
    operands: tuple = ()
    known: bool = True
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()

    def format(self) -> str:
        if not self.known:
            return ".word 0x%08x" % self.raw
        if not self.operands:
            return self.mnemonic
        return "%-12s %s" % (self.mnemonic, ", ".join(str(o) for o in self.operands))

    def __str__(self) -> str:
        return self.format()


def _ins(word, mnemonic, operands=(), reads=(), writes=()):
    return VuInstruction(word, mnemonic, tuple(operands), True, frozenset(reads), frozenset(writes))


def _unknown(word, half):
    return VuInstruction(word, half + "?", (), known=False)


@dataclasses.dataclass(frozen=True)
class Bundle:
    """An 8-byte microcode bundle: one lower and one upper instruction."""

    addr: int
    lower_raw: int
    upper_raw: int
    lower: VuInstruction
    upper: VuInstruction
    immediate: float = None

    @property
    def flags(self) -> str:
        out = []
        for bit, name in ((FLAG_I, "I"), (FLAG_E, "E"), (FLAG_M, "M"),
                          (FLAG_D, "D"), (FLAG_T, "T")):
            if self.upper_raw & bit:
                out.append(name)
        return "".join(out)

    @property
    def is_end(self) -> bool:
        return bool(self.upper_raw & FLAG_E)

    @property
    def fully_known(self) -> bool:
        return self.upper.known and self.lower.known

    def format(self) -> str:
        flags = self.flags
        marker = "[%-5s]" % flags if flags else " " * 7
        if self.immediate is not None:
            lower_text = "loi          %g" % self.immediate
        else:
            lower_text = self.lower.format()
        return "%08x %s %-34s %s" % (self.addr, marker, self.upper.format(), lower_text)


def _arith(word, name, fd, fs, second, second_reads, dest, to_acc):
    """An FMAC operation: `second` is ft (with its lanes) or Q or I."""
    lanes = "".join(c for c, bit in _LANES if (word >> 21) & bit)
    reads = set(second_reads)
    if name.startswith("opm"):
        # The outer product crosses lanes: x from y,z; y from z,x; z from x,y.
        reads = _vf(fs, "xyz") | _vf((word >> 16) & 0x1F, "xyz")
    else:
        reads |= _vf(fs, lanes)
    if name.startswith(_READS_ACC):
        reads |= _acc("xyz" if name.startswith("opm") else lanes)
    writes = _acc(lanes) if to_acc else _vf(fd, lanes)
    if not any(name.startswith(n) for n in _NO_FLAGS):
        writes = writes | {"MAC", "STATUS"}
    target = "ACC" if to_acc else VF[fd]
    return _ins(word, "%s.%s" % (name, dest), (target, VF[fs], second), reads, writes)


def decode_upper(word: int) -> VuInstruction:
    """Decode the FMAC half of a bundle."""
    opcode = word & 0x3F
    fd, fs, ft = (word >> 6) & 0x1F, (word >> 11) & 0x1F, (word >> 16) & 0x1F
    dest = _dest(word)
    lanes = "".join(c for c, bit in _LANES if (word >> 21) & bit)

    if opcode < 0x1C:
        name = _UPPER_BC[opcode & 0x1C] + _BC[opcode & 3]
        lane = _BC[opcode & 3]
        return _arith(word, name, fd, fs, "%s%s" % (VF[ft], lane), _vf(ft, lane), dest, False)
    if opcode in _UPPER_QI:
        name, special = _UPPER_QI[opcode]
        return _arith(word, name, fd, fs, special, {special}, dest, False)
    if opcode in _UPPER_REG:
        name = _UPPER_REG[opcode]
        return _arith(word, name, fd, fs, VF[ft], _vf(ft, lanes), dest, False)
    if opcode < 0x3C:
        return _unknown(word, "upper")

    index = (opcode & 3) | (((word >> 6) & 0x1F) << 2)
    if index == 0x2F:
        return _ins(word, "nop")
    if index == 0x1F:
        return _ins(word, "clip", (VF[fs] + "xyz", VF[ft] + "w"),
                    _vf(fs, "xyz") | _vf(ft, "w"), {"CLIP"})
    if index in _UPPER_SPECIAL_CONVERT:
        return _ins(word, "%s.%s" % (_UPPER_SPECIAL_CONVERT[index], dest), (VF[ft], VF[fs]),
                    _vf(fs, lanes), _vf(ft, lanes))
    if index < 0x1C and (index & 0x1C) in _UPPER_SPECIAL_BC:
        name = _UPPER_SPECIAL_BC[index & 0x1C] + _BC[index & 3]
        lane = _BC[index & 3]
        return _arith(word, name, 0, fs, "%s%s" % (VF[ft], lane), _vf(ft, lane), dest, True)
    if index in _UPPER_SPECIAL_QI:
        name, special = _UPPER_SPECIAL_QI[index]
        return _arith(word, name, 0, fs, special, {special}, dest, True)
    if index in _UPPER_SPECIAL_REG:
        name = _UPPER_SPECIAL_REG[index]
        return _arith(word, name, 0, fs, VF[ft], _vf(ft, lanes), dest, True)
    return _unknown(word, "upper")


def _imm11(word: int) -> int:
    value = word & 0x7FF
    return value - 0x800 if value & 0x400 else value


def decode_lower(word: int) -> VuInstruction:
    """Decode the load/store/integer/branch half of a bundle."""
    if word == LOWER_NOP:
        return _ins(word, "nop")

    opcode = (word >> 25) & 0x7F
    it, is_, id_ = (word >> 16) & 0x1F, (word >> 11) & 0x1F, (word >> 6) & 0x1F
    dest = _dest(word)
    lanes = "".join(c for c, bit in _LANES if (word >> 21) & bit)
    imm11 = _imm11(word)

    if opcode == LOWER_SPECIAL_GROUP:
        sub = word & 0x3F
        if sub in _LOWER_SPECIAL:
            name = _LOWER_SPECIAL[sub]
            if name == "iaddi":
                imm5 = id_ - 0x20 if id_ & 0x10 else id_
                return _ins(word, name, (VI[it], VI[is_], "%+d" % imm5), _vi(is_), _vi(it))
            return _ins(word, name, (VI[id_], VI[is_], VI[it]), _vi(is_) | _vi(it), _vi(id_))
        if sub < 0x3C:
            return _unknown(word, "lower")
        index = (sub & 3) | (id_ << 2)
        name = _LOWER_SPECIAL_EXT.get(index)
        if name is None:
            return _unknown(word, "lower")
        fsf, ftf = _BC[(word >> 21) & 3], _BC[(word >> 23) & 3]
        if name in ("move", "mr32"):
            source = "yzwx" if name == "mr32" else "xyzw"
            reads = set().union(*(_vf(is_, source[i]) for i, c in enumerate("xyzw") if c in lanes))
            return _ins(word, "%s.%s" % (name, dest), (VF[it], VF[is_]), reads, _vf(it, lanes))
        if name in ("lqi", "lqd"):
            text = "(%s++)" % VI[is_] if name == "lqi" else "(--%s)" % VI[is_]
            return _ins(word, "%s.%s" % (name, dest), (VF[it], text), _vi(is_), _vf(it, lanes) | _vi(is_))
        if name in ("sqi", "sqd"):
            text = "(%s++)" % VI[it] if name == "sqi" else "(--%s)" % VI[it]
            return _ins(word, "%s.%s" % (name, dest), (VF[is_], text), _vf(is_, lanes) | _vi(it), _vi(it))
        if name == "div" or name == "rsqrt":
            return _ins(word, name, ("Q", VF[is_] + fsf, VF[it] + ftf),
                        _vf(is_, fsf) | _vf(it, ftf), {"Q"})
        if name == "sqrt":
            return _ins(word, name, ("Q", VF[it] + ftf), _vf(it, ftf), {"Q"})
        if name in ("waitq", "waitp"):
            return _ins(word, name, (), {name[-1].upper()})
        if name == "mtir":
            return _ins(word, name, (VI[it], VF[is_] + fsf), _vf(is_, fsf), _vi(it))
        if name == "mfir":
            return _ins(word, "%s.%s" % (name, dest), (VF[it], VI[is_]), _vi(is_), _vf(it, lanes))
        if name in ("ilwr", "iswr"):
            if name == "ilwr":
                return _ins(word, "%s.%s" % (name, dest), (VI[it], "(%s)" % VI[is_]), _vi(is_), _vi(it))
            return _ins(word, "%s.%s" % (name, dest), (VI[it], "(%s)" % VI[is_]), _vi(is_) | _vi(it))
        if name in ("rnext", "rget"):
            return _ins(word, "%s.%s" % (name, dest), (VF[it], "R"), {"R"},
                        _vf(it, lanes) | ({"R"} if name == "rnext" else set()))
        if name in ("rinit", "rxor"):
            return _ins(word, name, ("R", VF[is_] + fsf), _vf(is_, fsf) | ({"R"} if name == "rxor" else set()), {"R"})
        if name == "mfp":
            return _ins(word, "%s.%s" % (name, dest), (VF[it], "P"), {"P"}, _vf(it, lanes))
        if name in ("xtop", "xitop"):
            return _ins(word, name, (VI[it],), (), _vi(it))
        if name == "xgkick":
            return _ins(word, name, (VI[is_],), _vi(is_))
        if name in ("esadd", "ersadd", "eleng", "erleng"):
            return _ins(word, name, ("P", VF[is_]), _vf(is_, "xyz"), {"P"})
        if name == "esum":
            return _ins(word, name, ("P", VF[is_]), _vf(is_, "xyzw"), {"P"})
        if name in ("eatanxy", "eatanxz"):
            return _ins(word, name, ("P", VF[is_]), _vf(is_, "xy" if name == "eatanxy" else "xz"), {"P"})
        # esqrt ersqrt ercpr esin eatan eexp: one lane.
        return _ins(word, name, ("P", VF[is_] + fsf), _vf(is_, fsf), {"P"})

    name = _LOWER_MAIN.get(opcode)
    if name is None:
        return _unknown(word, "lower")

    if name in ("b", "bal"):
        if name == "b":
            return _ins(word, name, ("%+d" % imm11,))
        return _ins(word, name, (VI[it], "%+d" % imm11), (), _vi(it))
    if name == "jr":
        return _ins(word, name, (VI[is_],), _vi(is_))
    if name == "jalr":
        return _ins(word, name, (VI[it], VI[is_]), _vi(is_), _vi(it))
    if name in ("ibeq", "ibne"):
        return _ins(word, name, (VI[it], VI[is_], "%+d" % imm11), _vi(is_) | _vi(it))
    if name in ("ibltz", "ibgtz", "iblez", "ibgez"):
        return _ins(word, name, (VI[is_], "%+d" % imm11), _vi(is_))
    if name in ("iaddiu", "isubiu"):
        # The immediate is split, with its high bits above the register fields.
        imm15 = (word & 0x7FF) | (((word >> 21) & 0xF) << 11)
        return _ins(word, name, (VI[it], VI[is_], "0x%x" % imm15), _vi(is_), _vi(it))
    if name == "lq":
        return _ins(word, "lq.%s" % dest, (VF[it], "%d(%s)" % (imm11, VI[is_])), _vi(is_), _vf(it, lanes))
    if name == "sq":
        return _ins(word, "sq.%s" % dest, (VF[is_], "%d(%s)" % (imm11, VI[it])), _vf(is_, lanes) | _vi(it))
    if name == "ilw":
        return _ins(word, "ilw.%s" % dest, (VI[it], "%d(%s)" % (imm11, VI[is_])), _vi(is_), _vi(it))
    if name == "isw":
        return _ins(word, "isw.%s" % dest, (VI[it], "%d(%s)" % (imm11, VI[is_])), _vi(is_) | _vi(it))
    imm24 = word & 0xFFFFFF
    if name in ("fceq", "fcand", "fcor"):
        return _ins(word, name, (VI[1], "0x%06x" % imm24), {"CLIP"}, _vi(1))
    if name == "fcset":
        return _ins(word, name, ("0x%06x" % imm24,), (), {"CLIP"})
    if name == "fcget":
        return _ins(word, name, (VI[it],), {"CLIP"}, _vi(it))
    imm12 = (((word >> 21) & 1) << 11) | (word & 0x7FF)
    if name in ("fseq", "fsand", "fsor"):
        return _ins(word, name, (VI[it], "0x%03x" % imm12), {"STATUS"}, _vi(it))
    if name == "fsset":
        return _ins(word, name, ("0x%03x" % imm12,), (), {"STATUS"})
    # fmeq fmand fmor
    return _ins(word, name, (VI[it], VI[is_]), {"MAC"} | _vi(is_), _vi(it))


def decode_bundle(lower_word: int, upper_word: int, addr: int = 0) -> Bundle:
    """Decode one 8-byte bundle.

    When the I flag is set the lower word is not an instruction at all but a
    32-bit float constant, so it is reported as an immediate instead.
    """
    upper = decode_upper(upper_word)
    if upper_word & FLAG_I:
        immediate = struct.unpack("<f", struct.pack("<I", lower_word))[0]
        lower = _ins(lower_word, "loi", ("%g" % immediate,), (), {"I"})
        return Bundle(addr, lower_word, upper_word, lower, upper, immediate)
    return Bundle(addr, lower_word, upper_word, decode_lower(lower_word), upper)


def decode_blob(data: bytes, base_addr: int = 0):
    """Decode a whole microcode blob into bundles."""
    count = len(data) // 8
    out = []
    for i in range(count):
        lower, upper = struct.unpack_from("<II", data, i * 8)
        out.append(decode_bundle(lower, upper, base_addr + i * 8))
    return out


def branch_target(bundle: Bundle):
    """Where a branch in the bundle's lower goes, in bytes, or None."""
    if bundle.immediate is not None or not bundle.lower.known:
        return None
    if bundle.lower.mnemonic in ("b", "bal", "ibeq", "ibne", "ibltz", "ibgtz", "iblez", "ibgez"):
        return bundle.addr + 8 + _imm11(bundle.lower_raw) * 8
    return None


def coverage(bundles) -> dict:
    """Summarise how much of a blob decoded.

    All-zero bundles are counted separately rather than as successes.  A zero
    upper word technically decodes - opcode 0 is `addx` - but a run of zeroes is
    padding, and letting it inflate the figure would misrepresent how much real
    microcode the decoder actually understands.
    """
    total = len(bundles)
    real = [b for b in bundles if b.lower_raw or b.upper_raw]
    padding = total - len(real)
    denom = len(real)
    upper_ok = sum(1 for b in real if b.upper.known)
    lower_ok = sum(1 for b in real if b.lower.known)
    return {
        "bundles": total,
        "padding": padding,
        "real": denom,
        "upper_known": upper_ok,
        "lower_known": lower_ok,
        "both_known": sum(1 for b in real if b.fully_known),
        "programs": sum(1 for b in bundles if b.is_end),
        "upper_pct": upper_ok / denom * 100.0 if denom else 0.0,
        "lower_pct": lower_ok / denom * 100.0 if denom else 0.0,
    }
