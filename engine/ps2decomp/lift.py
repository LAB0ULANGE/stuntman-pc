"""Lift machine code to C++, one statement per instruction.

Symbolic derivation (paths.py) produces readable functions, but it enumerates
paths, and large functions explode: half the executable's code is in functions
it cannot finish. Lifting is the mechanical complement. Every instruction becomes
a C++ statement over an explicit register context (ps2rt/lifted.h), branches
become gotos, and calls pass the whole context. Any function the interpreter can
run can be lifted, whatever its size or control flow.

The translation follows the interpreter (emu.py) instruction for instruction, since
that is what it is verified against. The verification is stricter than for
derived functions: after the run, the *entire* register file must match the
original's, not just $v0, along with every byte of memory written.

Control flow, precisely:

* A branch's condition and a jump's target are read before the delay slot runs,
  as on the hardware; a "likely" branch skips its delay slot when not taken.
* A delay slot that is also a branch target is emitted a second time at its own
  label, because reaching it by a jump runs it and then falls through.
* A call (`jal`, `jalr`, `bgezal`...) sets $ra after the delay slot and goes
  through `Call`, with the whole context.
* `jr $ra` returns. `jr` through any other register is a jump table if the target
  is inside the function and a tail call if not. A branch or jump that leaves the
  function, and running off its end, are tail calls too. A tail call runs until
  control reaches *this* function's return address (`$ra` at entry): the code it
  jumps to - a shared epilogue, say - restores that from the stack, while `$ra`
  itself may have been overwritten by an earlier call.
* Every loop head counts down a budget, so a loop that never ends cannot hang the
  host.
"""

from __future__ import annotations

import struct

from . import vu0
from .emu import is_supported
from .r5900 import DISASM

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


class Refused(Exception):
    """This function cannot be lifted; the message says why."""


def name_of(address: int) -> str:
    return "F_%08X" % address


def _u32(value: int) -> str:
    return "0x%Xu" % (value & MASK32)


def _u64(value: int) -> str:
    return "0x%Xull" % (value & MASK64)


def _sx32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & 0x80000000 else value


def _r(n: int) -> str:
    return "0ull" if n == 0 else "c.r[%d]" % n


def _set(rd: int, expr: str) -> str:
    """Write a general register; $zero discards the value."""
    if rd == 0:
        return ""
    return "c.r[%d] = %s;" % (rd, expr)


def _fd(ins): return (ins.raw >> 6) & 0x1F
def _fs(ins): return (ins.raw >> 11) & 0x1F
def _ft(ins): return (ins.raw >> 16) & 0x1F


_LOADS = {"lb": "ReadS8", "lbu": "ReadU8", "lh": "ReadS16", "lhu": "ReadU16",
          "lw": "ReadS32", "lwu": "ReadU32", "ld": "ReadU64"}
_STORES = {"sb": ("Write8", "(u32)"), "sh": ("Write16", "(u32)"),
           "sw": ("Write32", "(u32)"), "sd": ("Write64", "")}
_FLOAT_BINARY = {"add.s": "FloatAdd", "sub.s": "FloatSub", "mul.s": "FloatMul",
                 "div.s": "FloatDiv", "max.s": "FloatMax", "min.s": "FloatMin"}
_FLOAT_UNARY = {"mov.s": "FloatMove", "neg.s": "FloatNeg", "abs.s": "FloatAbs",
                "cvt.w.s": "IntFromFloat", "cvt.s.w": "FloatFromInt"}
_FLOAT_COMPARE = {"c.eq.s": "==", "c.lt.s": "<", "c.le.s": "<="}
_TRAPS = {"teq": ("==", "u64"), "tne": ("!=", "u64"), "tge": (">=", "s64"),
          "tgeu": (">=", "u64"), "tlt": ("<", "s64"), "tltu": ("<", "u64")}
_NOTHING = {"nop", "sync", "cache", "pref", "ei", "di"}
_LANES = {"pmaxw": "kPmaxw", "pminw": "kPminw", "psubw": "kPsubw", "padduw": "kPadduw",
          "paddub": "kPaddub", "pabsw": "kPabsw", "paddw": "kPaddw", "paddh": "kPaddh",
          "pcgtb": "kPcgtb", "pcgth": "kPcgth", "pcgtw": "kPcgtw", "pceqb": "kPceqb",
          "pceqh": "kPceqh", "pceqw": "kPceqw", "pmaxh": "kPmaxh", "pminh": "kPminh"}
_LANE_SHIFTS = {"psllh": (16, 0), "psrah": (16, 2), "psllw": (32, 0), "psrlw": (32, 1),
                "psraw": (32, 2)}
_UNALIGNED_LOADS = {"lwl": "LoadWordLeft", "lwr": "LoadWordRight",
                    "ldl": "LoadDoubleLeft", "ldr": "LoadDoubleRight"}
_UNALIGNED_STORES = {"swl": ("StoreLeft", 4), "swr": ("StoreRight", 4),
                     "sdl": ("StoreLeft", 8), "sdr": ("StoreRight", 8)}


def statement(ins) -> str:
    """The C++ for one non-control instruction."""
    m, rs, rt, rd, imm = ins.mnemonic, ins.rs, ins.rt, ins.rd, ins.imm
    a, b = _r(rs), _r(rt)
    address = "(u32)%s + %s" % (a, _u32(imm))

    if m in _NOTHING:
        return ""
    if ins.raw >> 26 == 0x12:
        return _cop2(ins)
    if m == "lqc2":
        loads = " ".join("u32 l%d = (u32)generated::ReadU32(q + %d);" % (i, 4 * i) for i in range(4))
        assign = " ".join("c.vf[%d][%d] = l%d;" % (rt, i, i) for i in range(4)) if rt else ""
        return "{ u32 q = (%s) & ~0xFu; %s %s }" % (address, loads, assign)
    if m == "sqc2":
        stores = " ".join("generated::Write32(q + %d, c.vf[%d][%d]);" % (4 * i, rt, i) for i in range(4))
        return "{ u32 q = (%s) & ~0xFu; %s }" % (address, stores)
    if m in ("addu", "add"):
        return _set(rd, "lifted::Sx32((u32)%s + (u32)%s)" % (a, b))
    if m in ("subu", "sub"):
        return _set(rd, "lifted::Sx32((u32)%s - (u32)%s)" % (a, b))
    if m in ("and", "or", "xor"):
        return _set(rd, "%s %s %s" % (a, {"and": "&", "or": "|", "xor": "^"}[m], b))
    if m == "nor":
        return _set(rd, "~(%s | %s)" % (a, b))
    if m in ("daddu", "dadd"):
        return _set(rd, "%s + %s" % (a, b))
    if m in ("dsubu", "dsub"):
        return _set(rd, "%s - %s" % (a, b))
    if m == "slt":
        return _set(rd, "(u64)((s64)%s < (s64)%s)" % (a, b))
    if m == "sltu":
        return _set(rd, "(u64)(%s < %s)" % (a, b))
    if m in ("movz", "movn"):
        if rd == 0:
            return ""
        return "if (%s %s 0) c.r[%d] = %s;" % (b, "==" if m == "movz" else "!=", rd, a)
    if m in ("sll", "srl", "sra"):
        return _set(rd, _word_shift(m, b, str(imm)))
    if m in ("sllv", "srlv", "srav"):
        return _set(rd, _word_shift(m[:3], b, "(%s & 31)" % a))
    if m in ("dsll", "dsrl", "dsra", "dsll32", "dsrl32", "dsra32"):
        amount = imm + (32 if m.endswith("32") else 0)
        return _set(rd, _double_shift(m[:4], b, str(amount)))
    if m in ("dsllv", "dsrlv", "dsrav"):
        return _set(rd, _double_shift(m[:4], b, "(%s & 63)" % a))

    pipe = "1" if m.endswith("1") and m[:-1] in (
        "mult", "multu", "div", "divu", "mfhi", "mflo", "mthi", "mtlo", "madd", "maddu") else ""
    base = m[:-1] if pipe else m
    hi, lo = "c.hi" + pipe, "c.lo" + pipe
    if base in ("mult", "multu", "madd", "maddu"):
        signed = "true" if base in ("mult", "madd") else "false"
        accumulate = "true" if base.startswith("madd") else "false"
        text = "lifted::Multiply(%s, %s, %s, %s, %s, %s);" % (hi, lo, a, b, signed, accumulate)
        return text + (" " + _set(rd, lo) if rd else "")
    if base in ("div", "divu"):
        return "lifted::Divide(%s, %s, %s, %s, %s);" % (
            hi, lo, a, b, "true" if base == "div" else "false")
    if base == "mfhi":
        return _set(rd, hi)
    if base == "mflo":
        return _set(rd, lo)
    if base == "mthi":
        return "%s = %s;" % (hi, a)
    if base == "mtlo":
        return "%s = %s;" % (lo, a)

    if m in ("addiu", "addi"):
        return _set(rt, "lifted::Sx32((u32)%s + %s)" % (a, _u32(imm)))
    if m in ("daddiu", "daddi"):
        return _set(rt, "%s + %s" % (a, _u64(imm)))
    if m == "lui":
        return _set(rt, _u64(_sx32((imm & 0xFFFF) << 16)))
    if m in ("andi", "ori", "xori"):
        op = {"andi": "&", "ori": "|", "xori": "^"}[m]
        return _set(rt, "%s %s %s" % (a, op, _u64(imm & 0xFFFF)))
    if m == "slti":
        return _set(rt, "(u64)((s64)%s < (s64)%s)" % (a, _u64(imm)))
    if m == "sltiu":
        return _set(rt, "(u64)(%s < %s)" % (a, _u64(imm)))

    if m in _LOADS:
        read = "generated::%s(%s)" % (_LOADS[m], address)
        return _set(rt, read) if rt else "(void)%s;" % read
    if m in _STORES:
        write, cast = _STORES[m]
        return "generated::%s(%s, %s%s);" % (write, address, cast, b)
    if m == "lq":
        # As the interpreter does: the upper half is written even for $zero.
        low = ("c.r[%d] = lo; " % rt) if rt else ""
        return ("{ u32 q = (%s) & ~0xFu; u64 lo = lifted::Read64(q); "
                "u64 hi = lifted::Read64(q + 8); %sc.rh[%d] = hi; }" % (address, low, rt))
    if m == "sq":
        return ("{ u32 q = (%s) & ~0xFu; generated::Write64(q, %s); "
                "generated::Write64(q + 8, c.rh[%d]); }" % (address, b, rt))

    if m == "lwc1":
        return "c.f[%d] = (u32)generated::ReadU32(%s);" % (rt, address)
    if m == "swc1":
        return "generated::Write32(%s, c.f[%d]);" % (address, rt)
    if m == "mtc1":
        return "c.f[%d] = (u32)%s;" % (_fs(ins), b)
    if m == "mfc1":
        return _set(rt, "lifted::Sx32(c.f[%d])" % _fs(ins))
    if m == "cfc1":
        # As emu._op_cfc1: FCR0 is the implementation number, FCR31 the condition.
        value = {0: "0x2E00ull", 31: "(c.fcc ? 0x800000ull : 0ull)"}.get(ins.rd, "0ull")
        return _set(rt, value)
    if m == "ctc1":
        return ("c.fcc = (u32)((%s >> 23) & 1);" % b) if ins.rd == 31 else ";"
    if m in _FLOAT_BINARY:
        return "c.f[%d] = generated::%s(c.f[%d], c.f[%d]);" % (
            _fd(ins), _FLOAT_BINARY[m], _fs(ins), _ft(ins))
    if m in _FLOAT_UNARY:
        return "c.f[%d] = generated::%s(c.f[%d]);" % (_fd(ins), _FLOAT_UNARY[m], _fs(ins))
    if m == "sqrt.s":
        return "c.f[%d] = generated::FloatSqrt(c.f[%d]);" % (_fd(ins), _ft(ins))
    if m in _FLOAT_COMPARE:
        return "c.fcc = generated::FloatValue(c.f[%d]) %s generated::FloatValue(c.f[%d]);" % (
            _fs(ins), _FLOAT_COMPARE[m], _ft(ins))
    if m == "c.f.s":
        return "c.fcc = 0;"
    if m in ("adda.s", "suba.s", "mula.s"):
        op = {"adda.s": "FloatAdd", "suba.s": "FloatSub", "mula.s": "FloatMul"}[m]
        return "c.acc = generated::%s(c.f[%d], c.f[%d]);" % (op, _fs(ins), _ft(ins))
    if m in ("madda.s", "msuba.s", "madd.s", "msub.s"):
        target = "c.acc" if m.endswith("a.s") else "c.f[%d]" % _fd(ins)
        return "%s = lifted::FloatMulAdd(c.acc, c.f[%d], c.f[%d], %s);" % (
            target, _fs(ins), _ft(ins), "true" if m.startswith("msub") else "false")

    if m in ("pand", "por", "pxor", "pnor"):
        if rd == 0:
            return ""
        if m == "pnor":
            return "{ u64 lo = ~(%s | c.r[%d]), hi = ~(c.rh[%d] | c.rh[%d]); c.r[%d] = lo; c.rh[%d] = hi; }" % (
                a, rt, rs, rt, rd, rd)
        op = {"pand": "&", "por": "|", "pxor": "^"}[m]
        return "{ u64 lo = %s %s %s, hi = c.rh[%d] %s c.rh[%d]; c.r[%d] = lo; c.rh[%d] = hi; }" % (
            a, op, b, rs, op, rt, rd, rd)
    if m in ("paddb", "psubb"):
        if rd == 0:
            return ""
        return "lifted::AddBytes(c.r[%d], c.rh[%d], %s, c.rh[%d], %s, c.rh[%d], %s);" % (
            rd, rd, a, rs, b, rt, "true" if m == "psubb" else "false")
    if m == "pcpyld":
        if rd == 0:
            return ""
        return "{ u64 lo = %s, hi = %s; c.r[%d] = lo; c.rh[%d] = hi; }" % (b, a, rd, rd)
    if m == "pcpyud":
        if rd == 0:
            return ""
        return "{ u64 lo = c.rh[%d], hi = c.rh[%d]; c.r[%d] = lo; c.rh[%d] = hi; }" % (rs, rt, rd, rd)
    if m == "prot3w":
        if rd == 0:
            return ""
        return ("{ u64 lo = c.r[%d], hi = c.rh[%d]; "
                "c.r[%d] = (lo >> 32) | (hi << 32); "
                "c.rh[%d] = (hi & 0xFFFFFFFF00000000ull) | (lo & 0xFFFFFFFFull); }" % (rt, rt, rd, rd))
    if m == "pcpyh":
        if rd == 0:
            return ""
        return ("{ u64 lo = (%s & 0xFFFF) * 0x0001000100010001ull, "
                "hi = (c.rh[%d] & 0xFFFF) * 0x0001000100010001ull; "
                "c.r[%d] = lo; c.rh[%d] = hi; }" % (b, rt, rd, rd))

    if m in _LANES:
        if rd == 0:
            return ""
        return "lifted::Lanes(lifted::%s, c.r[%d], c.rh[%d], %s, c.rh[%d], %s, c.rh[%d]);" % (
            _LANES[m], rd, rd, a, rs, b, rt)
    if m == "psrlh":
        if rd == 0:
            return ""
        return "lifted::ShiftHalfwordsRight(c.r[%d], c.rh[%d], %s, c.rh[%d], %du);" % (
            rd, rd, b, rt, imm & 0xF)
    if m in _LANE_SHIFTS:
        if rd == 0:
            return ""
        width, kind = _LANE_SHIFTS[m]
        return "lifted::ShiftLanes(c.r[%d], c.rh[%d], %s, c.rh[%d], %d, %d, %du);" % (
            rd, rd, b, rt, width, kind, imm & (width - 1))
    if m in ("pextlb", "pextlh"):
        if rd == 0:
            return ""
        return "lifted::Interleave(c.r[%d], c.rh[%d], %s, %s, %d);" % (
            rd, rd, a, b, 8 if m == "pextlb" else 16)
    if m == "ppacb":
        if rd == 0:
            return ""
        return "lifted::PackBytes(c.r[%d], c.rh[%d], %s, c.rh[%d], %s, c.rh[%d]);" % (
            rd, rd, a, rs, b, rt)
    if m in ("pextub", "pextuh", "pextuw"):
        if rd == 0:
            return ""
        return "lifted::Interleave(c.r[%d], c.rh[%d], c.rh[%d], c.rh[%d], %d);" % (
            rd, rd, rs, rt, {"pextub": 8, "pextuh": 16, "pextuw": 32}[m])
    if m == "ppach":
        if rd == 0:
            return ""
        return "lifted::PackHalfwords(c.r[%d], c.rh[%d], %s, c.rh[%d], %s, c.rh[%d]);" % (
            rd, rd, a, rs, b, rt)
    if m == "ppacw":
        if rd == 0:
            return ""
        return ("{ u64 lo = (%s & 0xFFFFFFFFull) | (c.rh[%d] << 32), "
                "hi = (%s & 0xFFFFFFFFull) | (c.rh[%d] << 32); c.r[%d] = lo; c.rh[%d] = hi; }" % (
                    b, rt, a, rs, rd, rd))
    if m == "qfsrv":
        if rd == 0:
            return ""
        return "lifted::FunnelShift(c.r[%d], c.rh[%d], %s, c.rh[%d], %s, c.rh[%d], c.sa);" % (
            rd, rd, a, rs, b, rt)
    if m in ("mtsab", "mtsah"):
        mask, scale = (0xF, 8) if m == "mtsab" else (0x7, 16)
        return "c.sa = (u32)((%s ^ %s) & 0x%X) * %d;" % (a, _u64(imm), mask, scale)
    if m == "rsqrt.s":
        return "c.f[%d] = lifted::FloatRsqrt(c.f[%d], c.f[%d]);" % (_fd(ins), _fs(ins), _ft(ins))
    if m == "pextlw":
        if rd == 0:
            return ""
        return ("{ u64 a = %s, b = %s; c.r[%d] = (b & 0xFFFFFFFFull) | (a << 32); "
                "c.rh[%d] = (b >> 32) | (a & 0xFFFFFFFF00000000ull); }" % (a, b, rd, rd))
    if m in _UNALIGNED_LOADS:
        return _set(rt, "lifted::%s(%s, %s)" % (_UNALIGNED_LOADS[m], address, b)) if rt else             "(void)lifted::%s(%s, 0);" % (_UNALIGNED_LOADS[m], address)
    if m in _UNALIGNED_STORES:
        helper, size = _UNALIGNED_STORES[m]
        return "lifted::%s(%s, %s, %d);" % (helper, address, b, size)
    if m == "break":
        return "lifted::Trap(0x%08Xu);" % ins.addr
    if m == "mtc0":
        return ""
    if m in _TRAPS:
        op, kind = _TRAPS[m]
        return "if ((%s)%s %s (%s)%s) lifted::Trap(0x%08Xu);" % (kind, a, op, kind, b, ins.addr)
    if m == "syscall":
        return "lifted::Syscall(c);"
    if m == "mfc0":
        # Through the runtime: 0 under verification, as the interpreter reads
        # it; a port's host answers (Count from its clock). "Read" in the name
        # keeps the folding (liveness.fold) from moving it across a call.
        return _set(rt, "lifted::ReadCop0(%du)" % ins.rd)
    raise Refused("no translation for %s at 0x%08X" % (m, ins.addr))


def _cop2(ins) -> str:
    """VU0 macro mode: the operation runs in runtime/src/vu0.cpp, named in a comment."""
    m = ins.mnemonic
    if m == "vcallms":
        # The microprogram's address in instructions (8 bytes), bits 20-6.
        return "lifted::Vcallms(c, 0x%04Xu);  /* vcallms */" % (((ins.raw >> 6) & 0x7FFF) * 8)
    if m == "vcallmsr":
        # From CMSAR0, VU0 control register 27.
        return "lifted::Vcallms(c, (c.vc[27] & 0xFFFFu) * 8);  /* vcallmsr */"
    if m in ("qmtc2", "ctc2", "qmfc2", "cfc2"):
        if m == "cfc2" and ((ins.raw >> 11) & 0x1F) in vu0.UNREADABLE:
            raise Refused("cfc2 of an unmodelled VU0 flag register")
        return "lifted::Cop2Move(c, 0x%08Xu, %d);  /* %s */" % (ins.raw, ins.rt, m)
    if m.split(".")[0] not in vu0.SUPPORTED:
        raise Refused("unsupported instruction %s" % m.split(".")[0])
    return "lifted::Cop2(c, 0x%08Xu);  /* %s %s */" % (ins.raw, m, ", ".join(ins.operands))


def _word_shift(kind: str, value: str, amount: str) -> str:
    if kind == "sll":
        return "lifted::Sx32((u32)%s << %s)" % (value, amount)
    if kind == "srl":
        return "lifted::Sx32((u32)%s >> %s)" % (value, amount)
    return "lifted::Sx32((u64)((s32)(u32)%s >> %s))" % (value, amount)


def _double_shift(kind: str, value: str, amount: str) -> str:
    if kind == "dsll":
        return "(%s << %s)" % (value, amount)
    if kind == "dsrl":
        return "(%s >> %s)" % (value, amount)
    return "(u64)((s64)%s >> %s)" % (value, amount)


_CONDITIONS = {
    "beq": "%(a)s == %(b)s", "beql": "%(a)s == %(b)s",
    "bne": "%(a)s != %(b)s", "bnel": "%(a)s != %(b)s",
    "blez": "(s64)%(a)s <= 0", "blezl": "(s64)%(a)s <= 0",
    "bgtz": "(s64)%(a)s > 0", "bgtzl": "(s64)%(a)s > 0",
    "bltz": "(s64)%(a)s < 0", "bltzl": "(s64)%(a)s < 0", "bltzal": "(s64)%(a)s < 0",
    "bgez": "(s64)%(a)s >= 0", "bgezl": "(s64)%(a)s >= 0", "bgezal": "(s64)%(a)s >= 0",
    "bc1t": "c.fcc != 0", "bc1tl": "c.fcc != 0",
    "bc1f": "c.fcc == 0", "bc1fl": "c.fcc == 0",
    # COP0's condition reads as "done" (see emu.py).
    "bc0t": "true", "bc0tl": "true", "bc0f": "false", "bc0fl": "false",
}


_POINTER_WORDS = {}
_SECTIONS = []          # (bytes, base): the program's other loaded sections
_FUNCTION_STARTS = set()
# Functions up to this many instructions dispatch a jr to any of them.
DISPATCH_ALL = 3000


def use_function_starts(addresses) -> None:
    """Where the analysis found functions: a large function's jr dispatch keeps
    those inside it as targets (see lift_function)."""
    _FUNCTION_STARTS.clear()
    _FUNCTION_STARTS.update(addresses)


def use_sections(sections) -> None:
    """The program's loaded sections besides the code being lifted, as
    (bytes, address): its jump tables are read there too. Burnout 3's sit in
    .data, apart from .text; a jr to a case no table named stopped the run."""
    _SECTIONS[:] = [(bytes(data), base) for data, base in sections]
    _POINTER_WORDS.clear()


def _pointer_words(image: bytes, base: int) -> list:
    """Every aligned word of the image and the program's other sections that
    holds an address inside the image, sorted."""
    key = (id(image), len(image), base, len(_SECTIONS))
    if key not in _POINTER_WORDS:
        found = set()
        for data in [image] + [d for d, _ in _SECTIONS]:
            words = struct.unpack_from("<%dI" % (len(data) // 4), data)
            found.update(w for w in words if w & 3 == 0 and base <= w < base + len(image))
        _POINTER_WORDS.clear()
        _POINTER_WORDS[key] = sorted(found)
    return _POINTER_WORDS[key]


def jump_targets(image: bytes, base: int, code) -> set:
    """Addresses a jr through a register may reach in `code`: words anywhere in
    the image that point into it (jump tables), address constants it builds
    (lui with addiu/daddiu/ori on the same register) and its calls' return
    points. A superset of the tables' entries; the start is always included."""
    import bisect
    addresses = {ins.addr for ins in code}
    low, high = min(addresses), max(addresses) + 4
    words = _pointer_words(image, base)
    targets = {code[0].addr}
    targets.update(w for w in words[bisect.bisect_left(words, low):bisect.bisect_left(words, high)]
                   if w in addresses)
    upper = {}
    for ins in code:
        m, raw = ins.mnemonic, ins.raw
        if m == "lui":
            upper.setdefault(ins.rt, set()).add((raw & 0xFFFF) << 16)
        elif m in ("addiu", "daddiu", "ori") and ins.rs in upper:
            low16 = raw & 0xFFFF
            if m != "ori" and low16 & 0x8000:
                low16 -= 0x10000
            for hi in upper[ins.rs]:
                value = (hi | low16) if m == "ori" else (hi + low16) & MASK32
                if value in addresses:
                    targets.add(value)
        if ins.is_call and ins.addr + 8 in addresses:
            targets.add(ins.addr + 8)
    return targets


def lift_function(image: bytes, base: int, start: int, end: int, chunks=()) -> str:
    """The C++ definition of one function; raises Refused if it cannot be lifted.

    `chunks` are the function's out-of-line pieces (analysis.py `chunks`): blocks
    its branches reach outside [start, end), which the compiler placed among
    other functions' code. They are translated after the body, as part of it, so
    a branch to one is a goto rather than a tail transfer, and control never
    falls from one piece into the next: each ends with a transfer to whatever
    follows it in memory.
    """
    if end <= start:
        raise Refused("empty function")
    ranges = [(start, end)] + sorted((s, e) for s, e in chunks if e > s)
    code = []
    for first, last in ranges:
        words = struct.unpack_from("<%dI" % ((last - first) // 4), image, first - base)
        code += [DISASM.decode(word, first + 4 * i) for i, word in enumerate(words)]
    by_address = {ins.addr: ins for ins in code}
    starts = {first: last for first, last in ranges}

    for ins in code:
        if not ins.is_valid:
            raise Refused("invalid instruction at 0x%08X" % ins.addr)
        # The interpreter that verifies lifted code has no VU0 micro mode, so a
        # function that calls it lifts unverified rather than not at all.
        if not is_supported(ins) and ins.mnemonic not in ("vcallms", "vcallmsr"):
            raise Refused("unsupported instruction %s" % ins.mnemonic)
        if ins.has_delay_slot:
            slot = by_address.get(ins.addr + 4)
            if slot is None:
                raise Refused("delay slot outside the function at 0x%08X" % ins.addr)
            if slot.has_delay_slot:
                raise Refused("branch in a delay slot at 0x%08X" % ins.addr)

    inside = lambda address: any(first <= address < last for first, last in ranges)
    dispatch = any(ins.mnemonic == "jr" and ins.rs != 31 for ins in code)
    labels, loop_heads = set(), set()
    for ins in code:
        if ins.has_delay_slot and ins.mnemonic not in ("jr", "jalr") and not ins.is_call:
            if inside(ins.target):
                labels.add(ins.target)
                if ins.target <= ins.addr:
                    loop_heads.add(ins.target)
    if dispatch:
        # Where a jr through a register can land in this function: any of its
        # instructions - a table can be built at run time (Burnout 3's at
        # 0x001BD7FC is, in BSS) - and any of them can begin a loop. Only past
        # DISPATCH_ALL instructions, where every one a target made one compiler
        # take twelve minutes over a function of thirteen thousand, the targets
        # are what can be known: what a table or an address constant names, its
        # calls' return points, and the function starts the analysis found
        # inside it (where a tail transfer would run another function's way).
        # A target outside that set leaves through the default, which the host
        # reports when no function starts there - loud, never a wrong turn.
        if len(code) <= DISPATCH_ALL:
            dispatched = sorted(by_address)
        else:
            known = jump_targets(image, base, code) | {a for a in _FUNCTION_STARTS if a in by_address}
            dispatched = sorted(a for a in known if a in by_address)
        labels.update(dispatched)
        loop_heads.update(dispatched)
    for _first, last in ranges[:-1]:
        if inside(last):
            labels.add(last)                # a piece that runs on into another

    lines = ["void %s(lifted::Context& c) {" % name_of(start),
             # Where this function returns to: a tail transfer to code it does not
             # contain hands the rest of the run over, and that code returns here.
             "    const u32 entry_ra = (u32)c.r[31], entry_sp = (u32)c.r[29];",
             "    (void)entry_ra; (void)entry_sp;"]
    if dispatch:
        lines.append("    u32 jump;")
    skip = set()
    running = None                          # the end of the range being emitted
    for ins in code:
        address = ins.addr
        if address in starts:
            if running is not None:
                lines.append("    " + _transfer(running, inside))
            running = starts[address]
        if address in labels:
            lines.append("L_%08X:" % address)
            if address in loop_heads:
                lines.append("    lifted::Tick(c, 0x%08Xu);" % address)
        if address in skip:
            # A delay slot already emitted with its branch. It runs here again only
            # when a jump lands on it (its label, above); otherwise nothing reaches
            # this point - the branch block ended with a goto past it.
            if address in labels:
                lines.append("    " + statement(ins))
            continue
        if not ins.has_delay_slot:
            text = statement(ins)
            if text:
                lines.append("    " + text)
            continue

        slot = by_address[address + 4]
        slot_text = statement(slot)
        m = ins.mnemonic
        after = address + 8
        link = "c.r[31] = 0x%08Xull;" % after
        body = []
        if m in ("j", "b"):
            body = [slot_text, _transfer(ins.target, inside)]
        elif m == "jal":
            body = [slot_text, link, "lifted::Call(c, 0x%08Xu);" % ins.target]
        elif m == "jr" and ins.rs == 31:
            body = [slot_text, "return;"]
        elif m == "jr":
            body = ["jump = (u32)%s;" % _r(ins.rs), slot_text, "goto dispatch;"]
        elif m == "jalr":
            body = ["{ u32 target = (u32)%s;" % _r(ins.rs), slot_text, link,
                    "lifted::Call(c, target); }"]
        else:
            condition = _CONDITIONS[m] % {"a": _r(ins.rs), "b": _r(ins.rt)}
            taken = _transfer(ins.target, inside)
            linking = link if m in ("bltzal", "bgezal") else ""
            if m.endswith("l") and m not in ("bltzal", "bgezal"):
                body = ["if (%s) { %s %s }" % (condition, slot_text, taken)]
            elif m in ("bltzal", "bgezal"):
                body = ["{ bool t = %s;" % condition, slot_text, linking,
                        "if (t) lifted::Call(c, 0x%08Xu); }" % ins.target]
            else:
                body = ["{ bool t = %s;" % condition, slot_text, "if (t) %s }" % taken]
        lines.append("    " + " ".join(part for part in body if part))
        skip.add(address + 4)
        if address + 4 in labels:
            # Execution that branched past the delay slot continues after it.
            lines.append("    goto L_%08X;" % after if inside(after) else
                         "    lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); return;" % after)
            labels.add(after) if inside(after) else None
    lines.append("    lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp);  /* runs off the end */" % running)
    if dispatch:
        # One switch for every jr through a register: a copy at each would be
        # tens of thousands of cases apiece in a big function - beyond what a
        # compiler takes in reasonable time and memory.
        lines.append("    return;")
        lines.append("dispatch:")
        lines.append("    " + _dispatch(dispatched))
    lines.append("}")
    text = "\n".join(lines)
    # A label is only emitted at an instruction; make sure every goto has one.
    for target in sorted(labels):
        if "L_%08X:" % target not in text:
            raise Refused("label 0x%08X has no instruction" % target)
    return text + "\n"


def _transfer(target: int, inside) -> str:
    if inside(target):
        return "goto L_%08X;" % target
    return "{ lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); return; }" % target


def _dispatch(targets) -> str:
    cases = " ".join("case 0x%08Xu: goto L_%08X;" % (a, a) for a in targets)
    return ("switch (jump) { %s default: lifted::TailCall(c, jump, entry_ra, entry_sp); "
            "return; }" % cases)
