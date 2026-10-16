"""Symbolic execution of straight-line functions, to derive their semantics.

A large number of the discovered functions are tiny read-only accessors - load a
pointer from a structure, index it, return a field.  Reading each one by hand works
but does not scale, and the result is mechanical enough that it can be derived
instead.

This walks a branch-free function with *expressions* in the registers rather than
values, and reports whatever ends up in `$v0` as a tree.  That tree does two jobs:
it renders as C for a human to read, and it evaluates against concrete arguments so
the derivation can be differential-tested against the real machine code.  A
derivation that cannot be tested is worth little, so both come from the same tree.

Scope is deliberately narrow.  Anything with a branch, a store, or an instruction
outside the supported set is refused with `Unsupported` rather than approximated -
a plausible but wrong formula presented as a derivation would be worse than no
derivation at all.
"""

from __future__ import annotations

import dataclasses
import struct

from .r5900 import DISASM

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


class Unsupported(Exception):
    """Raised when a function is outside what can be derived symbolically."""


def _cached(name):
    """Memoise a function of one expression on the (immutable) node itself.

    Unrolled loops build expressions that share subtrees, so a plain recursive
    walk is exponential in the tree's depth - the first version of these turned a
    15-second derivation run into one that did not finish in ten minutes.
    """
    def wrap(function):
        def cached(expr):
            try:
                return object.__getattribute__(expr, name)
            except AttributeError:
                result = function(expr)
                object.__setattr__(expr, name, result)
                return result
        cached.__doc__ = function.__doc__
        cached.__name__ = function.__name__
        return cached
    return wrap


# -- expression tree -----------------------------------------------------

class Expr:
    """Base class for symbolic values."""

    def __post_init__(self) -> None:
        # Record the node's size *as a tree* - the size its rendering would have -
        # at construction, from the children's already-recorded sizes. That makes the
        # size free to query, which matters because unrolled loops build expressions
        # that share subtrees: a value used twice per iteration doubles the tree each
        # time round while the underlying graph grows only linearly. Anything that
        # walks such a value as a tree is exponential, and the deriver has to be able
        # to notice that before it walks it. Defined here so every dataclass subclass
        # inherits it; dataclasses call an inherited __post_init__.
        #
        # Named `tree_size`, not `size`: Load already has a field called `size` (the
        # access width), and a base-class `size` attribute becomes that field's
        # default value, which breaks every dataclass declared after it.
        object.__setattr__(
            self,
            "_tree_size",
            min(1 + sum(child.tree_size for child in children(self)), 1 << 30),
        )

    @property
    def tree_size(self) -> int:
        """Node count as a tree, i.e. the size of this expression when printed."""
        return getattr(self, "_tree_size", 1)

    def render(self) -> str:
        raise NotImplementedError

    def evaluate(self, args, read) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.render()


# The EE calling convention passes eight integer arguments in registers - $a0..$a3
# and then $t0..$t3 - and eight float arguments in $f12..$f19. The model numbers
# them as one vector: 0..7 are the integer arguments, 8..15 the float ones.
INT_ARG_REGS = (4, 5, 6, 7, 8, 9, 10, 11)
FLOAT_ARG_BASE = 8
FLOAT_ARG_REGS = tuple(range(12, 20))
ARG_COUNT = 16


def arg_name(index: int) -> str:
    """`arg0`..`arg7` for integer arguments, `farg0`..`farg7` for float ones."""
    if index >= FLOAT_ARG_BASE:
        return "farg%d" % (index - FLOAT_ARG_BASE)
    return "arg%d" % index


@dataclasses.dataclass(frozen=True)
class Arg(Expr):
    index: int  # see INT_ARG_REGS and FLOAT_ARG_REGS

    def render(self) -> str:
        if _ARG_RENDERER is not None:
            return _ARG_RENDERER(self)
        return arg_name(self.index)

    def evaluate(self, args, read) -> int:
        return args[self.index] & MASK64


@dataclasses.dataclass(frozen=True)
class Const(Expr):
    value: int

    def render(self) -> str:
        v = self.value
        if -256 < _signed(v) < 256:
            return str(_signed(v))
        return "0x%X" % (v & MASK64)

    def evaluate(self, args, read) -> int:
        return self.value & MASK64


@dataclasses.dataclass(frozen=True)
class Load(Expr):
    address: Expr
    size: int
    signed: bool
    # Which memory the load reads: 0 is memory as the function was entered, and
    # epoch k is memory as it stood when the k-th call to a function that has not
    # been derived returned. Such a callee can change anything, so a load after it
    # is a different value from the same load before it, and must compare unequal.
    epoch: int = 0

    def render(self) -> str:
        if _LOAD_RENDERER is not None:
            return _LOAD_RENDERER(self, lambda node: node.render())
        kind = {1: "s8" if self.signed else "u8",
                2: "s16" if self.signed else "u16",
                4: "s32" if self.signed else "u32",
                8: "u64", 16: "u128"}[self.size]
        return "*(%s*)(%s)" % (kind, self.address.render())

    def evaluate(self, args, read) -> int:
        address = self.address.evaluate(args, read) & MASK32
        if self.epoch:
            return read(address, self.size, self.signed, self.epoch)
        return read(address, self.size, self.signed)


@dataclasses.dataclass(frozen=True)
class CallResult(Expr):
    """What `$v0` - or, for a float result, `$f0` - held when the `index`-th
    opaque call on a path returned.

    An opaque call is one into a function that has not been derived: rather than
    inlining it, the derived code calls it. Evaluating this needs the call to have
    been made, so it is only meaningful inside a sequential evaluation that makes
    the calls (see paths.Execution).
    """

    index: int
    target: int
    # 2 for $v0; FLOAT_RESULT for $f0, as its 32 bits.
    register: int = 2

    def render(self) -> str:
        return ("r%d" if self.register == 2 else "rf%d") % self.index

    def evaluate(self, args, read) -> int:
        if self.register == 2:
            return read.result(self.index) & MASK64
        return read.float_result(self.index) & 0xFFFFFFFF


_REGISTER_NAMES = ["zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
                   "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                   "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
                   "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]


def register_name(index: int) -> str:
    """A register's name for a C++ identifier: `s0`, `f20`, `hi`, `fcc`."""
    if index < 32:
        return _REGISTER_NAMES[index]
    if index == REG_HI:
        return "hi"
    if index == REG_LO:
        return "lo"
    if index == FP_COND:
        return "fcc"
    if index == FP_ACC:
        return "facc"
    if UPPER_BASE <= index < UPPER_BASE + 32:
        return "%s_hi" % _REGISTER_NAMES[index - UPPER_BASE]
    if VF_BASE <= index < VC_BASE:
        return "vf%d%s" % ((index - VF_BASE) // 4, "xyzw"[(index - VF_BASE) % 4])
    if VC_BASE <= index < VACC_BASE:
        return "vc%d" % (index - VC_BASE)
    if VACC_BASE <= index < VACC_BASE + 4:
        return "vacc%s" % "xyzw"[index - VACC_BASE]
    if index >= FPR_BASE:
        return "f%d" % (index - FPR_BASE)
    return "r%d" % index


@dataclasses.dataclass(frozen=True)
class Var(Expr):
    """A register's value at the start of the current iteration of a loop.

    A loop whose trip count depends on its input cannot be unrolled into one
    expression, so its body is derived once over variables: every register the
    body writes is a Var, given its value before the loop on entry and the body's
    new value on each `continue`.
    """

    loop: int
    register: int

    def render(self) -> str:
        return "l%d_%s" % (self.loop, register_name(self.register))

    def evaluate(self, args, read) -> int:
        return read.var(self.loop, self.register) & MASK64


@dataclasses.dataclass(frozen=True)
class LoopExit(Expr):
    """True when a loop was left by its `index`-th exit."""

    loop: int
    index: int

    def render(self) -> str:
        return "l%d_exit == %d" % (self.loop, self.index)

    def evaluate(self, args, read) -> int:
        return read.exit_taken(self.loop) == self.index

    def negate(self) -> Expr:
        return Not1(self)


@dataclasses.dataclass(frozen=True)
class Temp(Expr):
    """A named value the emitter has already captured into a local.

    Exists only during rendering: the emitter replaces a subexpression with one of
    these once it has been stored in a `const` local, so later statements refer to
    the local instead of re-reading memory that may have changed since.
    """

    name: str

    def render(self) -> str:
        return self.name

    def evaluate(self, args, read) -> int:
        raise Unsupported("a rendering temporary has no value of its own")


# C operator precedence, tighter binding first. Rendering *must* respect this:
# `arg1 << 2 + arg0` is not `(arg1 << 2) + arg0` in C, it is `arg1 << (2 + arg0)`.
# Emitting unparenthesised text would produce source that disagrees with the tree
# it was derived from, which is the one failure mode this module exists to avoid.
PRECEDENCE = {
    "mul": 10,
    "add": 9, "sub": 9,
    "shl": 8, "shr": 8, "sar": 8,
    "dshl": 8, "dshr": 8, "dsar": 8,
    "ltu": 6, "lt": 6, "geu": 6,
    # Relational and equality operators, for branch conditions.
    "eq": 5, "ne": 5, "le": 6, "gt": 6, "ge": 6,
    "and": 4,
    "xor": 3,
    "or": 2,
}
ATOM_PRECEDENCE = 12


def _precedence(expr: "Expr") -> int:
    # Arithmetic shifts render as a parenthesised cast expression, which is an atom.
    if isinstance(expr, Binary) and expr.op in ("sar", "dsar"):
        return ATOM_PRECEDENCE
    if isinstance(expr, (Binary, Compare, Predicate)):
        return PRECEDENCE[expr.op]
    if isinstance(expr, Select):
        return 1
    return ATOM_PRECEDENCE


# -- C++ value semantics -------------------------------------------------
#
# The rendered text is C++, and it has to mean what the tree means, not merely
# look like it. The tree computes on 64-bit register values, so the rendering
# makes every value a `u64`: arguments are declared `u64`, the Read* helpers
# return the register image of the loaded value (sign- or zero-extended), and
# the operators whose meaning depends on width or signedness carry explicit
# casts. Left to C++'s own promotion rules, `ReadU8(p) << 24` is an `int` shift
# and a signed compare of two `u64`s is unsigned - an earlier version emitted
# exactly that, and only compiling and running the output exposed it.

def fits_u32(expr: "Expr") -> bool:
    """True when the value is provably below 2**32."""
    if isinstance(expr, Const):
        return (expr.value & MASK64) <= MASK32
    if isinstance(expr, Load):
        return not expr.signed and expr.size <= 4
    if isinstance(expr, (Compare, Predicate, FloatPredicate, Not1, FloatOp, VuOp,
                         IntToFloat, FloatToInt)):
        return True
    if isinstance(expr, Binary):
        if expr.op == "shr":
            return True
        if expr.op == "and":
            return fits_u32(expr.left) or fits_u32(expr.right)
        if expr.op in ("or", "xor"):
            return fits_u32(expr.left) and fits_u32(expr.right)
        if expr.op == "dshr" and isinstance(expr.right, Const):
            return (expr.right.value & 0x3F) >= 32
    if isinstance(expr, Select):
        return fits_u32(expr.when_true) and fits_u32(expr.when_false)
    return False


def nonnegative(expr: "Expr") -> bool:
    """True when the value is below 2**63, so signed and unsigned order agree."""
    if fits_u32(expr):
        return True
    if isinstance(expr, Const):
        return _signed(expr.value) >= 0
    if isinstance(expr, Binary):
        if expr.op == "dshr" and isinstance(expr.right, Const):
            return (expr.right.value & 0x3F) >= 1
        if expr.op == "and":
            return nonnegative(expr.left) or nonnegative(expr.right)
        if expr.op in ("or", "xor"):
            return nonnegative(expr.left) and nonnegative(expr.right)
    return False


def _signed_literal(value: int) -> str:
    """A constant spelled so C++ gives it its signed 64-bit value.

    Negative values are written in decimal on purpose: `-0x80000000` is unsigned
    in C++ (the hex literal is an `unsigned int` and negating it wraps), whereas
    the decimal literal 2147483648 is a `long long`.
    """
    s = _signed(value)
    if s == -(1 << 63):
        return "INT64_MIN"
    if s < 0:
        return "-%d" % -s if s <= -256 else str(s)
    return Const(value).render()


def _signed_operand(expr: "Expr", limit: int) -> str:
    """Render an operand of a signed 64-bit comparison."""
    if isinstance(expr, Const):
        return _signed_literal(expr.value)
    if isinstance(expr, SignExtend32):
        # (s32) already yields the right signed value; widening it to u64 and back
        # would only add noise.
        return "(s32)%s" % _wrap(expr.value, ATOM_PRECEDENCE)
    return "(s64)%s" % _wrap(expr, ATOM_PRECEDENCE)


def _render_order(op: str, left: "Expr", right: "Expr", symbol: str, level: int) -> str:
    """Render a relational comparison with the signedness the tree specifies."""
    if op in ("eq", "ne", "ltu", "geu") or (nonnegative(left) and nonnegative(right)):
        return "%s %s %s" % (_wrap(left, level), symbol, _wrap(right, level + 1))
    return "%s %s %s" % (
        _signed_operand(left, level), symbol, _signed_operand(right, level + 1)
    )


def _shift_amount(amount: "Expr", bits: int) -> str:
    """The shift count, masked as the hardware masks it.

    A C++ shift by the operand width or more is undefined, while the R5900 simply
    uses the low 5 (word) or 6 (doubleword) bits, so a variable count is masked
    explicitly and a constant one is reduced before printing.
    """
    mask = bits - 1
    if isinstance(amount, Const):
        return str(amount.value & mask)
    return "(%s & %d)" % (amount.render(), mask)


# How a Load is rendered varies by consumer: the default prints a pointer cast,
# while the C++ emitter prints a Read* helper call. Everything *else* about
# rendering - precedence, parenthesisation - must not vary, so it lives here once
# and the Load form is the only hook. An earlier version duplicated the whole
# renderer in the emitter, and the two drifted: a clarity rule added here never
# reached the generated source.
_LOAD_RENDERER = None


def set_load_renderer(renderer):
    """Install a custom Load renderer; returns the previous one.

    `renderer(node, render_child)` returns the text for a Load node, where
    `render_child(expr)` renders a subexpression at lowest precedence.
    """
    global _LOAD_RENDERER
    previous = _LOAD_RENDERER
    _LOAD_RENDERER = renderer
    return previous


# The second and last hook: how an argument prints. A typed argument is a view
# object in the emitted C++, so in a value context it has to read `arg0.address`.
_ARG_RENDERER = None


class arg_renderer:
    """Install a renderer for `Arg` nodes for the duration of a block."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._previous = None

    def __enter__(self):
        global _ARG_RENDERER
        self._previous, _ARG_RENDERER = _ARG_RENDERER, self._renderer
        return self

    def __exit__(self, *exc):
        global _ARG_RENDERER
        _ARG_RENDERER = self._previous
        return False


class load_renderer:
    """Context manager form of `set_load_renderer`."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._previous = None

    def __enter__(self):
        self._previous = set_load_renderer(self._renderer)
        return self

    def __exit__(self, *exc):
        set_load_renderer(self._previous)
        return False


def _wrap(child: "Expr", limit: int) -> str:
    """Render `child`, parenthesising it when its binding is looser than `limit`."""
    text = child.render()
    return "(%s)" % text if _precedence(child) < limit else text


def _narrow_in_cpp(expr) -> bool:
    """Whether an expression's rendered C++ may have a 32-bit type: the float and
    VU helpers return u32, and a bitwise combination of such values and literals
    stays u32. Everything else - loads, arguments, arithmetic - is u64."""
    if isinstance(expr, Const):
        return True
    if type(expr).__name__ in ("FloatOp", "VuOp", "IntToFloat", "FloatToInt"):
        return True
    if isinstance(expr, Binary) and expr.op in ("and", "or", "xor"):
        return _narrow_in_cpp(expr.left) and _narrow_in_cpp(expr.right)
    return False


@dataclasses.dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr

    _SYMBOL = {
        "add": "+", "sub": "-", "mul": "*", "shl": "<<",
        "shr": ">>", "sar": ">>", "and": "&", "or": "|", "xor": "^",
        "dshl": "<<", "dshr": ">>", "dsar": ">>",
    }

    def render(self) -> str:
        level = PRECEDENCE[self.op]
        if self.op in ("sar", "dsar"):
            # An arithmetic shift needs a signed operand of the right width: `sra`
            # shifts the low word as an s32, `dsra` the whole register as an s64.
            # The result is widened back to the u64 every value is.
            cast, bits = ("s32", 32) if self.op == "sar" else ("s64", 64)
            return "(u64)((%s)%s >> %s)" % (
                cast, _wrap(self.left, ATOM_PRECEDENCE), _shift_amount(self.right, bits)
            )
        if self.op in ("shl", "shr", "dshl", "dshr"):
            bits = 32 if self.op in ("shl", "shr") else 64
            left = _wrap(self.left, level)
            if isinstance(self.left, Const):
                # A bare literal is an `int`, and `1 << 31` on an int overflows.
                left = "%sull" % Const(self.left.value & MASK64).render().lstrip("-")
                if _signed(self.left.value) < 0:
                    left = "0x%Xull" % (self.left.value & MASK64)
            if isinstance(self.left, Binary) and self.left.op in ("add", "sub"):
                left = "(%s)" % left
            if self.op == "shr" and not fits_u32(self.left):
                # `srl` shifts the low word only; the upper half must not shift in.
                left = "(%s & 0xFFFFFFFF)" % self.left.render()
            if self.op in ("dshl", "dshr") and _narrow_in_cpp(self.left):
                # A float or VU helper returns u32, and so does a mask of one: a
                # 64-bit shift of it in C++ would shift at 32 bits and lose it.
                left = "(u64)%s" % _wrap(self.left, ATOM_PRECEDENCE)
            return "%s %s %s" % (left, self._SYMBOL[self.op], _shift_amount(self.right, bits))
        left = _wrap(self.left, level)
        # `a - b << 4` is valid C and means `(a - b) << 4`, but almost nobody reads
        # it that way, so an arithmetic operand of a shift gets parenthesised even
        # though precedence does not require it. Redundant parentheses cannot change
        # meaning; a misread can.
        # `_wrap` leaves an add/sub unparenthesised here because precedence allows
        # it, so wrapping unconditionally cannot double up. Testing the rendered
        # text for a leading "(" would be wrong: `(a << 3) - b` starts with one
        # without being parenthesised as a whole.
        if self.op in ("shl", "shr", "sar", "dshl", "dshr", "dsar") and isinstance(self.left, Binary):
            if self.left.op in ("add", "sub"):
                left = "(%s)" % left
        # The right operand of a left-associative operator needs a parenthesis at
        # equal precedence too, or `a - (b - c)` would print as `a - b - c`.
        return "%s %s %s" % (left, self._SYMBOL[self.op], _wrap(self.right, level + 1))

    def evaluate(self, args, read) -> int:
        a = self.left.evaluate(args, read)
        b = self.right.evaluate(args, read)
        if self.op == "add":
            return (a + b) & MASK64
        if self.op == "sub":
            return (a - b) & MASK64
        if self.op == "mul":
            return (a * b) & MASK64
        # Word shifts mask the amount to 5 bits, doubleword shifts to 6. Using the
        # 5-bit mask for both silently turns a shift of 32 into a shift of zero,
        # which is how `dsll32`/`dsra32` and the HI half of a multiply came out
        # wrong - caught by the differential tests rather than by reading.
        if self.op == "shl":
            return (a << (b & 0x1F)) & MASK64
        if self.op == "shr":
            return (a & MASK32) >> (b & 0x1F)
        if self.op == "sar":
            return (_sign32(a) >> (b & 0x1F)) & MASK64
        if self.op == "dshl":
            return (a << (b & 0x3F)) & MASK64
        if self.op == "dshr":
            return (a & MASK64) >> (b & 0x3F)
        if self.op == "dsar":
            return (_signed(a) >> (b & 0x3F)) & MASK64
        if self.op == "and":
            return a & b
        if self.op == "or":
            return a | b
        if self.op == "xor":
            return a ^ b
        raise Unsupported("cannot evaluate %r" % self.op)


@dataclasses.dataclass(frozen=True)
class Compare(Expr):
    op: str
    left: Expr
    right: Expr

    def render(self) -> str:
        # `slt`/`sltu` produce 0 or 1 as a register value; the cast keeps it a u64
        # so a later `<< 31` cannot become a signed int overflow.
        return "(u64)(%s)" % _render_order(
            self.op, self.left, self.right, "<", PRECEDENCE[self.op]
        )

    def evaluate(self, args, read) -> int:
        a = self.left.evaluate(args, read)
        b = self.right.evaluate(args, read)
        if self.op == "ltu":
            return 1 if (a & MASK64) < (b & MASK64) else 0
        return 1 if _signed(a) < _signed(b) else 0


# -- floating point ------------------------------------------------------
#
# Every expression in this model evaluates to an integer *bit pattern*, floats
# included. That mirrors the hardware - a float register holds 32 bits, and
# `mfc1`/`swc1` move those bits around without interpreting them - and it keeps
# evaluation uniform, with reinterpretation confined to the float operations
# themselves rather than smeared across the whole tree.

def _bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


# The largest finite single-precision magnitude, bit pattern 0x7F7FFFFF.
FLOAT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def _float_to_bits(value: float) -> int:
    """Encode a result the way the EE's FPU would.

    That FPU is deliberately not IEEE-754: no infinities, no NaNs, and an overflow
    saturates to the largest representable magnitude. This must match `_f32` in
    emu.py exactly - the two are compared against each other on every derivation,
    so any divergence registers as a disagreement.
    """
    if value != value:
        return _float_to_bits(0.0)
    if value > FLOAT_MAX:
        value = FLOAT_MAX
    elif value < -FLOAT_MAX:
        value = -FLOAT_MAX
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float_operand(expr: "Expr") -> str:
    """Render a float operand; a constant becomes a float literal.

    `FloatBits(-1.0f)` says what `0xBF800000` means. The literal is the shortest
    decimal that converts back to exactly the same bits, so nothing is lost; a
    NaN or infinity pattern has no literal and stays hexadecimal.
    """
    if not isinstance(expr, Const):
        return expr.render()
    bits = expr.value & MASK32
    value = _bits_to_float(bits)
    if value != value or value in (float("inf"), float("-inf")):
        return "0x%08X" % bits
    for digits in range(1, 10):
        text = "%.*g" % (digits, value)
        if struct.unpack("<I", struct.pack("<f", float(text)))[0] == bits:
            break
    if not any(c in text for c in ".en"):
        text += ".0"
    return "FloatBits(%sf)" % text


@dataclasses.dataclass(frozen=True)
class FloatOp(Expr):
    """A single-precision arithmetic operation over bit-pattern operands.

    `madd` and `msub` take three: the accumulator (`left`) plus or minus the
    product of `right` and `third`, rounded once, as the FPU does.
    """

    op: str
    left: Expr
    right: Expr = None   # None for the unary forms
    third: Expr = None   # only for madd and msub

    _NAMES = {
        "add": "FloatAdd", "sub": "FloatSub", "mul": "FloatMul", "div": "FloatDiv",
        "min": "FloatMin", "max": "FloatMax",
        "neg": "FloatNeg", "abs": "FloatAbs", "sqrt": "FloatSqrt", "mov": "FloatMove",
        "madd": "FloatMulAdd", "msub": "FloatMulSub",
    }

    def render(self) -> str:
        name = self._NAMES[self.op]
        if self.third is not None:
            return "%s(%s, %s, %s)" % (name, _float_operand(self.left),
                                       _float_operand(self.right), _float_operand(self.third))
        if self.right is None:
            return "%s(%s)" % (name, _float_operand(self.left))
        return "%s(%s, %s)" % (name, _float_operand(self.left), _float_operand(self.right))

    def evaluate(self, args, read) -> int:
        a = _bits_to_float(self.left.evaluate(args, read))
        if self.third is not None:
            product = (_bits_to_float(self.right.evaluate(args, read))
                       * _bits_to_float(self.third.evaluate(args, read)))
            return _float_to_bits(a + product if self.op == "madd" else a - product)
        if self.right is None:
            result = {
                "neg": lambda x: -x,
                "abs": abs,
                "mov": lambda x: x,
                "sqrt": lambda x: x ** 0.5 if x >= 0 else 0.0,
            }[self.op](a)
            return _float_to_bits(result)
        b = _bits_to_float(self.right.evaluate(args, read))
        if self.op == "div":
            # The FPU flags division by zero rather than trapping; the interpreter
            # yields zero, and the two must agree or every such function would
            # register as a disagreement.
            result = 0.0 if b == 0.0 else a / b
        else:
            result = {
                "add": lambda x, y: x + y,
                "sub": lambda x, y: x - y,
                "mul": lambda x, y: x * y,
                "min": min,
                "max": max,
            }[self.op](a, b)
        return _float_to_bits(result)


@dataclasses.dataclass(frozen=True)
class IntDiv(Expr):
    """A 32-bit quotient or remainder as the EE computes it (emu.divide), defined
    for every input, a zero divisor included. op: quot, rem, quotu, remu."""

    op: str
    left: Expr
    right: Expr

    _NAMES = {"quot": "DivQuot", "rem": "DivRem", "quotu": "DivQuotU", "remu": "DivRemU"}

    def render(self) -> str:
        return "%s(%s, %s)" % (self._NAMES[self.op], self.left.render(), self.right.render())

    def evaluate(self, args, read) -> int:
        from .emu import divide

        lo, hi = divide(self.left.evaluate(args, read), self.right.evaluate(args, read),
                        not self.op.endswith("u"))
        return lo if self.op.startswith("quot") else hi


@dataclasses.dataclass(frozen=True)
class VuOp(Expr):
    """A VU0 arithmetic operation on one lane, over bit patterns (see vu0.py).

    `madd`/`msub`: `left` is the accumulator lane, `right` times `third` the
    product. `itof`/`ftoi` carry their fixed-point shift in `shift`. `div` and
    `rsqrt` take numerator and denominator, `abs` and `sqrt` one operand.
    """

    op: str
    left: Expr
    right: Expr = None
    third: Expr = None
    shift: int = 0

    _NAMES = {"add": "VuAdd", "sub": "VuSub", "mul": "VuMul", "max": "VuMax",
              "mini": "VuMin", "madd": "VuMadd", "msub": "VuMsub", "abs": "VuAbs",
              "itof": "VuItof", "ftoi": "VuFtoi", "div": "VuDiv", "sqrt": "VuSqrt",
              "rsqrt": "VuRsqrt"}

    def operands(self):
        return tuple(x for x in (self.left, self.right, self.third) if x is not None)

    def render(self) -> str:
        parts = [_float_operand(x) for x in self.operands()]
        if self.op in ("itof", "ftoi"):
            parts.append(str(self.shift))
        return "%s(%s)" % (self._NAMES[self.op], ", ".join(parts))

    def evaluate(self, args, read) -> int:
        from . import vu0

        values = [x.evaluate(args, read) & MASK32 for x in self.operands()]
        if self.op in ("madd", "msub"):
            return vu0.vu_binary(self.op, values[1], values[2], values[0])
        if self.op in ("add", "sub", "mul", "max", "mini"):
            return vu0.vu_binary(self.op, values[0], values[1])
        if self.op == "abs":
            return vu0.vu_abs(values[0])
        if self.op == "itof":
            return vu0.vu_itof(values[0], self.shift)
        if self.op == "ftoi":
            return vu0.vu_ftoi(values[0], self.shift)
        if self.op == "div":
            return vu0.vu_div(values[0], values[1])
        if self.op == "sqrt":
            return vu0.vu_sqrt(values[0])
        return vu0.vu_rsqrt(values[0], values[1])


@dataclasses.dataclass(frozen=True)
class IntToFloat(Expr):
    """`cvt.s.w`: reinterpret the operand's bits as an integer, convert to float."""

    value: Expr

    def render(self) -> str:
        if isinstance(self.value, Const):
            return "FloatFromInt(%d)" % _sign32(self.value.value)
        return "FloatFromInt(%s)" % self.value.render()

    def evaluate(self, args, read) -> int:
        return _float_to_bits(float(_sign32(self.value.evaluate(args, read))))


@dataclasses.dataclass(frozen=True)
class FloatToInt(Expr):
    """`cvt.w.s`: convert a float to an integer, truncating toward zero and
    saturating past 32 bits (emu.float_to_word)."""

    value: Expr

    def render(self) -> str:
        return "IntFromFloat(%s)" % _float_operand(self.value)

    def evaluate(self, args, read) -> int:
        bits = self.value.evaluate(args, read) & MASK32
        if (bits & 0x7F800000) <= 0x4E800000:
            return int(_bits_to_float(bits)) & MASK32
        return 0x80000000 if bits & 0x80000000 else 0x7FFFFFFF


@dataclasses.dataclass(frozen=True)
class FloatPredicate(Expr):
    """A `c.<cond>.s` comparison, as tested later by `bc1t`/`bc1f`."""

    op: str  # eq lt le
    left: Expr
    right: Expr

    _SYMBOL = {"eq": "==", "lt": "<", "le": "<=", "false": "false"}

    def render(self) -> str:
        if self.op == "false":
            return "false"
        return "FloatValue(%s) %s FloatValue(%s)" % (
            _float_operand(self.left), self._SYMBOL[self.op], _float_operand(self.right)
        )

    def evaluate(self, args, read) -> int:
        if self.op == "false":
            return False
        a = _bits_to_float(self.left.evaluate(args, read))
        b = _bits_to_float(self.right.evaluate(args, read))
        return {"eq": a == b, "lt": a < b, "le": a <= b}[self.op]

    def negate(self) -> "Expr":
        return Not1(self)


@dataclasses.dataclass(frozen=True)
class Not1(Expr):
    """Logical negation of a condition, as distinct from a bitwise complement."""

    value: Expr

    def render(self) -> str:
        return "!(%s)" % self.value.render()

    def evaluate(self, args, read) -> int:
        return not self.value.evaluate(args, read)

    def negate(self) -> Expr:
        return self.value


@dataclasses.dataclass(frozen=True)
class SignExtend32(Expr):
    """Narrow to 32 bits and sign-extend, as every 32-bit MIPS result is.

    Needed to model `mult` faithfully: it multiplies the low words of its operands
    as signed values, so the raw 64-bit register contents must not be used.
    """

    value: Expr

    def render(self) -> str:
        return "(u64)(s32)%s" % _wrap(self.value, ATOM_PRECEDENCE)

    def evaluate(self, args, read) -> int:
        return _sign32(self.value.evaluate(args, read)) & MASK64


@dataclasses.dataclass(frozen=True)
class Not(Expr):
    """Bitwise complement, used to express `nor` as `~(a | b)`."""

    value: Expr

    def render(self) -> str:
        return "~%s" % _wrap(self.value, ATOM_PRECEDENCE)

    def evaluate(self, args, read) -> int:
        return ~self.value.evaluate(args, read) & MASK64


@dataclasses.dataclass(frozen=True)
class Select(Expr):
    """A ternary: the shape of a conditional move.

    `movz`/`movn` write their destination only when a condition holds, so the
    result depends on what the destination already contained - which is why the
    previous value is part of the expression rather than discarded.
    """

    condition: Expr
    when_true: Expr
    when_false: Expr

    def render(self) -> str:
        return "%s ? %s : %s" % (
            _wrap(self.condition, 7),
            _wrap(self.when_true, 7),
            _wrap(self.when_false, 7),
        )

    def evaluate(self, args, read) -> int:
        taken = self.condition.evaluate(args, read)
        branch = self.when_true if taken else self.when_false
        return branch.evaluate(args, read) & MASK64


@dataclasses.dataclass(frozen=True)
class Predicate(Expr):
    """A branch condition, evaluated with signed comparison as MIPS does."""

    op: str  # eq ne lt le gt ge, plus ltu geu for an unsigned `sltu` result
    left: Expr
    right: Expr

    _SYMBOL = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
               "ltu": "<", "geu": ">="}

    def render(self) -> str:
        return _render_order(self.op, self.left, self.right, self._SYMBOL[self.op], 7)

    def evaluate(self, args, read) -> int:
        a = self.left.evaluate(args, read)
        b = self.right.evaluate(args, read)
        if self.op == "ltu":
            return (a & MASK64) < (b & MASK64)
        if self.op == "geu":
            return (a & MASK64) >= (b & MASK64)
        a, b = _signed(a), _signed(b)
        return {
            "eq": a == b, "ne": a != b, "lt": a < b,
            "le": a <= b, "gt": a > b, "ge": a >= b,
        }[self.op]

    def negate(self) -> "Predicate":
        opposite = {"eq": "ne", "ne": "eq", "lt": "ge", "ge": "lt", "le": "gt", "gt": "le",
                    "ltu": "geu", "geu": "ltu"}
        return Predicate(opposite[self.op], self.left, self.right)


def _signed(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _sign32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & 0x80000000 else value


def children(expr: Expr) -> tuple:
    """Every subexpression of a node.

    Deliberately fail-closed: an unrecognised node type raises rather than
    returning nothing. Every traversal in the project is built on this, and the
    silent-empty version of it caused a real bug - `Not`, `Select` and
    `SignExtend32` were added without updating the argument scan, so arguments
    nested inside them became invisible and the emitter produced functions
    referencing undeclared parameters. A traversal that cannot see a new node type
    must break loudly, not quietly skip it.
    """
    if isinstance(expr, (Arg, Const, CallResult, Temp, Var, LoopExit)):
        return ()
    if isinstance(expr, Load):
        return (expr.address,)
    if isinstance(expr, (Not, Not1, SignExtend32, IntToFloat, FloatToInt)):
        return (expr.value,)
    if isinstance(expr, Select):
        return (expr.condition, expr.when_true, expr.when_false)
    if isinstance(expr, FloatOp):
        if expr.third is not None:
            return (expr.left, expr.right, expr.third)
        return (expr.left,) if expr.right is None else (expr.left, expr.right)
    if isinstance(expr, VuOp):
        return expr.operands()
    if isinstance(expr, IntDiv):
        return (expr.left, expr.right)
    if isinstance(expr, (Binary, Compare, Predicate, FloatPredicate)):
        return (expr.left, expr.right)
    raise Unsupported("no traversal rule for %s" % type(expr).__name__)


def rebuild(expr: Expr, new_children) -> Expr:
    """Reconstruct a node with replaced subexpressions, preserving its type."""
    if isinstance(expr, (Arg, Const, CallResult, Temp, Var, LoopExit)):
        return expr
    if isinstance(expr, Load):
        return Load(new_children[0], expr.size, expr.signed, expr.epoch)
    if isinstance(expr, Not):
        return Not(new_children[0])
    if isinstance(expr, Not1):
        return Not1(new_children[0])
    if isinstance(expr, SignExtend32):
        return SignExtend32(new_children[0])
    if isinstance(expr, IntToFloat):
        return IntToFloat(new_children[0])
    if isinstance(expr, FloatToInt):
        return FloatToInt(new_children[0])
    if isinstance(expr, Select):
        return Select(*new_children)
    if isinstance(expr, FloatOp):
        if expr.right is None:
            return FloatOp(expr.op, new_children[0])
        return FloatOp(expr.op, *new_children)
    if isinstance(expr, VuOp):
        kids = list(new_children) + [None] * (3 - len(new_children))
        return VuOp(expr.op, kids[0], kids[1], kids[2], expr.shift)
    if isinstance(expr, IntDiv):
        return IntDiv(expr.op, *new_children)
    if isinstance(expr, Binary):
        return Binary(expr.op, *new_children)
    if isinstance(expr, Compare):
        return Compare(expr.op, *new_children)
    if isinstance(expr, Predicate):
        return Predicate(expr.op, *new_children)
    if isinstance(expr, FloatPredicate):
        return FloatPredicate(expr.op, *new_children)
    raise Unsupported("no rebuild rule for %s" % type(expr).__name__)


def arguments_used(expr: Expr) -> set:
    """Indices of every `Arg` anywhere in an expression."""
    if isinstance(expr, Arg):
        return {expr.index}
    found = set()
    for child in children(expr):
        found |= arguments_used(child)
    return found


def substitute(expr: Expr, arguments) -> Expr:
    """Rewrite a derivation's `Arg` nodes with the caller's actual expressions.

    This is what makes a derived callee usable at a call site: the callee's result
    is expressed in terms of `arg0..arg3`, and inlining it means replacing those
    with whatever the caller had in `$a0..$a3` at the moment of the call.

    `arguments` maps an argument index to an expression. An index with no entry is
    left as-is, which keeps the result honest: if a callee reads an argument the
    caller never set, that shows up as a stray `argN` rather than silently becoming
    zero.

    Built on `children`/`rebuild` so that a new node type cannot be silently
    skipped - the earlier hand-written version missed three and substituted nothing
    inside them.
    """
    if isinstance(expr, Arg):
        return arguments.get(expr.index, expr)
    kids = children(expr)
    if not kids:
        return expr
    return rebuild(expr, [substitute(child, arguments) for child in kids])


def simplify(expr: Expr) -> Expr:
    """Fold constants and drop identities; see `_simplify`.

    Memoised on the node, and the result is marked as its own simplification.
    Without that mark every simplified node was new and uncached, so simplifying
    it again - which `address_of` does - recomputed everything beneath it, doubling
    the work at each level of a pointer chain.
    """
    try:
        return object.__getattribute__(expr, "_simplified")
    except AttributeError:
        pass
    result = _simplify(expr)
    object.__setattr__(expr, "_simplified", result)
    try:
        object.__getattribute__(result, "_simplified")
    except AttributeError:
        object.__setattr__(result, "_simplified", result)
    return result


def _simplify(expr: Expr) -> Expr:
    """Fold constants and drop identities, so the output reads like source.

    Every rule here preserves meaning exactly.  The derived code is
    differential-tested against the original, so an over-eager rule shows up as a
    disagreement rather than as quietly wrong source - which is why it is safe to
    simplify for readability at all.
    """
    if isinstance(expr, Binary):
        left, right = simplify(expr.left), simplify(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            folded = Binary(expr.op, left, right).evaluate((0, 0, 0, 0), None)
            return Const(folded)
        # x + 0, x | 0, x << 0 and friends add nothing.
        if isinstance(right, Const) and right.value == 0 and expr.op in (
            "add", "sub", "or", "xor", "shl", "shr", "sar"
        ):
            return left
        if isinstance(left, Const) and left.value == 0 and expr.op in ("add", "or", "xor"):
            return right
        # Collapse a chain of constant offsets: (x + c1) + c2 becomes x + (c1 + c2).
        # These arise constantly from `lui`/`addiu` pairs feeding a further offset,
        # and leaving them unfolded prints addresses as unreadable sums.
        if (
            expr.op == "add"
            and isinstance(right, Const)
            and isinstance(left, Binary)
            and left.op == "add"
            and isinstance(left.right, Const)
        ):
            return simplify(
                Binary("add", left.left, Const((left.right.value + right.value) & MASK64))
            )
        return Binary(expr.op, left, right)
    if isinstance(expr, Load):
        return Load(address_of(expr.address), expr.size, expr.signed, expr.epoch)
    if isinstance(expr, Compare):
        return Compare(expr.op, simplify(expr.left), simplify(expr.right))
    if isinstance(expr, SignExtend32):
        simplified = simplify(expr.value)
        inner = low32(simplified)
        if inner is not simplified:
            # Removing an inner extension can expose folds it was blocking: an
            # unrolled counter is sext(sext(i + 1) + 1), which must become i + 2 or
            # it grows with every iteration.
            inner = simplify(inner)
        if isinstance(inner, Const):
            return Const(_sign32(inner.value) & MASK64)
        if isinstance(inner, SignExtend32) or fits_s32(inner):
            return inner
        return SignExtend32(inner)
    if isinstance(expr, (FloatOp, VuOp, IntDiv, IntToFloat, FloatToInt, FloatPredicate,
                         Not1, Not, Select)):
        kids = children(expr)
        return rebuild(expr, [simplify(child) for child in kids]) if kids else expr
    if isinstance(expr, Predicate):
        left, right = simplify(expr.left), simplify(expr.right)
        # `slt` followed by `bne ..., $zero` is how a compiler spells a plain
        # comparison, and printing it as `(a < b) ? 1 : 0 != 0` hides that. The
        # 0/1 result of a Compare is never anything else, so testing it against
        # zero is exactly the comparison itself, or its negation.
        if isinstance(left, Compare) and isinstance(right, Const) and right.value == 0:
            # The Compare's signedness carries over. An earlier version mapped `ltu`
            # to a signed `lt` here, which the tests missed because they only ever
            # passed small positive pointers.
            inner = Predicate(left.op, left.left, left.right)
            if expr.op == "ne":
                return inner
            if expr.op == "eq":
                return inner.negate()
        return Predicate(expr.op, left, right)
    return expr


# -- sign extension ----------------------------------------------------------
#
# Word operations produce a 32-bit result sign-extended to 64 bits, modelled as a
# SignExtend32 around them. Most consumers only ever look at the low 32 bits - an
# address, a store of four bytes or fewer, a returned value, the operands of
# another word operation - and for those the extension is noise, so `low32`
# removes it. Where all 64 bits matter (a comparison, a doubleword shift) it
# stays, unless `fits_s32` shows the value is in range anyway.

# Operations whose low 32 result bits depend only on the low 32 bits of their
# operands.
_LOW_BITS_ONLY = ("add", "sub", "mul", "shl", "and", "or", "xor", "shr", "sar")
S32_MIN, S32_MAX = -(1 << 31), (1 << 31) - 1


@_cached("_low32")
def low32(expr: Expr) -> Expr:
    """`expr` with sign extensions removed where only its low 32 bits are used."""
    if isinstance(expr, SignExtend32):
        return low32(expr.value)
    if isinstance(expr, Binary) and expr.op in _LOW_BITS_ONLY:
        left = low32(expr.left)
        # A shift amount is masked by the hardware, so its low bits suffice too.
        right = low32(expr.right)
        if left is expr.left and right is expr.right:
            return expr
        return Binary(expr.op, left, right)
    if isinstance(expr, Not):
        inner = low32(expr.value)
        return expr if inner is expr.value else Not(inner)
    return expr


def address_of(expr: Expr) -> Expr:
    """An address expression in canonical form: only its low 32 bits are used."""
    return simplify(low32(simplify(expr)))


@_cached("_range")
def value_range(expr: Expr):
    """(low, high) bounds on the value as a signed 64-bit number, or None."""
    if isinstance(expr, Const):
        value = _signed(expr.value)
        return value, value
    if isinstance(expr, Load):
        bits = 8 * expr.size
        if expr.size > 4:
            return None
        if expr.signed:
            return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        return 0, (1 << bits) - 1
    if isinstance(expr, SignExtend32):
        return S32_MIN, S32_MAX
    if isinstance(expr, (Compare, Predicate, FloatPredicate, Not1)):
        return 0, 1
    if isinstance(expr, (FloatOp, VuOp, IntToFloat, FloatToInt)):
        return 0, MASK32
    if isinstance(expr, Arg) and expr.index >= FLOAT_ARG_BASE:
        return 0, MASK32
    if isinstance(expr, Select):
        a, b = value_range(expr.when_true), value_range(expr.when_false)
        if a is None or b is None:
            return None
        return min(a[0], b[0]), max(a[1], b[1])
    if isinstance(expr, Not):
        inner = value_range(expr.value)
        return None if inner is None else (-inner[1] - 1, -inner[0] - 1)
    if not isinstance(expr, Binary):
        return None
    if expr.op == "shr":
        if isinstance(expr.right, Const):
            return 0, MASK32 >> (expr.right.value & 0x1F)
        return 0, MASK32
    if expr.op == "sar":
        return S32_MIN, S32_MAX
    if expr.op == "dshr" and isinstance(expr.right, Const) and expr.right.value & 0x3F:
        return 0, ((1 << 64) - 1) >> (expr.right.value & 0x3F)
    a, b = value_range(expr.left), value_range(expr.right)
    if expr.op == "and":
        bounds = [r for r in (a, b) if r is not None and r[0] >= 0]
        if bounds:
            return 0, min(r[1] for r in bounds)
        if a is not None and b is not None and _within_s32(a) and _within_s32(b):
            return S32_MIN, S32_MAX
        return None
    if a is None or b is None:
        return None
    if expr.op in ("or", "xor"):
        if a[0] >= 0 and b[0] >= 0:
            return 0, (1 << max(a[1], b[1]).bit_length()) - 1
        if _within_s32(a) and _within_s32(b):
            return S32_MIN, S32_MAX
        return None
    if expr.op == "add":
        result = a[0] + b[0], a[1] + b[1]
    elif expr.op == "sub":
        result = a[0] - b[1], a[1] - b[0]
    elif expr.op == "mul":
        products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
        result = min(products), max(products)
    elif expr.op in ("shl", "dshl") and isinstance(expr.right, Const):
        factor = 1 << (expr.right.value & (0x1F if expr.op == "shl" else 0x3F))
        result = a[0] * factor, a[1] * factor
    else:
        return None
    # Outside the signed 64-bit range the value would have wrapped.
    if result[0] < -(1 << 63) or result[1] >= (1 << 63):
        return None
    return result


def _within_s32(bounds) -> bool:
    return bounds is not None and S32_MIN <= bounds[0] and bounds[1] <= S32_MAX


def fits_s32(expr: Expr) -> bool:
    """True when sign-extending the value's low 32 bits would not change it."""
    return _within_s32(value_range(expr))


# -- the symbolic machine ------------------------------------------------

_LOAD_SIZES = {
    "lb": (1, True), "lbu": (1, False),
    "lh": (2, True), "lhu": (2, False),
    "lw": (4, True), "lwu": (4, False),
    "ld": (8, False), "lq": (16, False),
}


_STORE_SIZES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8, "sq": 16}

_FLOAT_BINARY = {
    "add.s": "add", "sub.s": "sub", "mul.s": "mul", "div.s": "div",
    "max.s": "max", "min.s": "min",
}
_FLOAT_UNARY = {"mov.s": "mov", "neg.s": "neg", "abs.s": "abs", "sqrt.s": "sqrt"}
# Accumulator forms: (operation, where the result goes).
_FLOAT_ACCUMULATE = {
    "adda.s": ("add", "acc"), "suba.s": ("sub", "acc"), "mula.s": ("mul", "acc"),
    "madda.s": ("madd", "acc"), "msuba.s": ("msub", "acc"),
    "madd.s": ("madd", "fd"), "msub.s": ("msub", "fd"),
}
_FLOAT_COMPARE = {"c.eq.s": "eq", "c.lt.s": "lt", "c.le.s": "le", "c.f.s": "false"}

# The multiplier result registers. They are kept in the same dict as the general
# registers under indices the hardware cannot name, which avoids threading a second
# piece of state through every caller.
REG_HI = 64
REG_LO = 65

# Float registers and the FPU condition flag share the register dict under indices
# the integer unit cannot name, so no extra state has to be threaded through every
# caller. The condition flag holds a predicate expression, which is what `bc1t` and
# `bc1f` then branch on.
FPR_BASE = 128
FP_COND = 127
# The FPU accumulator, which adda/suba/mula/madda/msuba write and madd/msub read.
FP_ACC = 126

# The upper 64 bits of each general register, which only 128-bit operations - lq,
# sq, the MMI moves and logic - read or write. Everything else leaves them alone,
# as the hardware does. An upper half the function has not written is unknown (at
# entry it is the caller's), and reading it is refused rather than taken as zero.
UPPER_BASE = 192


def upper(index: int) -> int:
    return UPPER_BASE + index


def _upper_or_zero(get, index: int):
    return Const(0) if index == 0 else get(upper(index))


# VU0 state in the register dict: VF lanes, the control registers (VI0-VI15, then
# R, I, Q), and the accumulator lanes. None is known at entry except VF0, the
# constant (0, 0, 0, 1); a call clobbers all of it, since the callee may use VU0.
VF_BASE = 256
VC_BASE = 384
VACC_BASE = 416
VU_KEYS = tuple(range(VF_BASE + 4, VACC_BASE + 4))


def vf_lane(index: int, lane: int) -> int:
    return VF_BASE + 4 * index + lane


def vu_written(ins) -> set:
    """The VU0 state an instruction writes, for the loop analysis."""
    from . import vu0

    op = ins.raw >> 26
    if op == 0x36:                                    # lqc2
        return {vf_lane(ins.rt, lane) for lane in range(4)} if ins.rt else set()
    if op != 0x12:
        return set()
    word, m = ins.raw, ins.mnemonic
    fd, ft = (word >> 6) & 0x1F, (word >> 16) & 0x1F
    index = (word >> 11) & 0x1F
    dest = (word >> 21) & 0xF
    lanes = [lane for lane, bit in enumerate((8, 4, 2, 1)) if dest & bit]
    base = m.split(".")[0]
    if m == "qmtc2":
        return {vf_lane(index, lane) for lane in range(4)} if index else set()
    if m == "ctc2":
        return {VC_BASE + index}
    if m in ("qmfc2", "cfc2"):
        return set()
    if base == "vopmula":
        return {VACC_BASE + lane for lane in lanes if lane < 3}
    if base == "vopmsub":
        return {vf_lane(fd, lane) for lane in lanes if lane < 3} if fd else set()
    operation, to_acc, _operand = vu0._split(base)
    if operation is not None:
        if to_acc:
            return {VACC_BASE + lane for lane in lanes}
        return {vf_lane(fd, lane) for lane in lanes} if fd else set()
    if base in ("vmove", "vmr32", "vabs") or base[:5] in ("vitof", "vftoi"):
        return {vf_lane(ft, lane) for lane in lanes} if ft else set()
    if base in ("vdiv", "vsqrt", "vrsqrt"):
        return {VC_BASE + vu0.REG_Q}
    return set()


# Instructions that write an upper half, for the loop analysis: (mnemonic -> field).
QUAD_WRITERS = {"lq": "rt", "pcpyld": "rd", "pcpyud": "rd", "por": "rd", "pand": "rd",
                "pxor": "rd", "pnor": "rd"}


# Where a call's float result is read from: $f0, as CallResult.register.
FLOAT_RESULT = FPR_BASE


def fpr(index: int) -> int:
    return FPR_BASE + index


@dataclasses.dataclass(frozen=True)
class Store:
    """One memory write a function performs, as a pair of expressions."""

    address: Expr
    value: Expr
    size: int

    def render(self) -> str:
        kind = {1: "u8", 2: "u16", 4: "u32", 8: "u64", 16: "u128"}[self.size]
        return "*(%s*)(%s) = %s;" % (
            kind, simplify(self.address).render(), simplify(self.value).render()
        )


@dataclasses.dataclass
class Derivation:
    address: int
    expression: Expr
    instructions: int
    stores: tuple = ()

    def render(self) -> str:
        lines = [store.render() for store in self.stores]
        lines.append("return %s;" % simplify(self.expression).render())
        return "\n".join(lines)

    def evaluate(self, args, read) -> int:
        return _sign32(simplify(self.expression).evaluate(args, read) & MASK32)

    def evaluate_stores(self, args, read) -> list:
        """Resolve each store to a concrete (address, value, size) triple.

        Addresses and values are evaluated against the state *before* any store is
        applied, which is correct only because a derivation is refused if it ever
        loads from an address it has already written - see `derive`.
        """
        out = []
        for store in self.stores:
            address = simplify(store.address).evaluate(args, read) & MASK32
            value = simplify(store.value).evaluate(args, read)
            out.append((address, value & ((1 << (store.size * 8)) - 1), store.size))
        return out


def derive(image: bytes, base: int, start: int, end: int,
           gp_value: int = None) -> Derivation:
    """Derive the return expression and memory effects of a branch-free function.

    `$gp` is seeded with its startup value, since globals are commonly reached as
    an offset from it and refusing those would exclude a large set of otherwise
    simple functions.
    """
    if gp_value is None:
        from .layout import CURRENT

        gp_value = CURRENT.gp
    regs = {0: Const(0), 28: Const(gp_value)}
    for i, reg in enumerate(INT_ARG_REGS):
        regs[reg] = Arg(i)
    stores = []

    def get(index: int) -> Expr:
        if index == 0:
            return Const(0)
        if index not in regs:
            raise Unsupported("reads uninitialised register $%d" % index)
        return regs[index]

    addr = start
    count = 0
    while addr < end:
        word = struct.unpack_from("<I", image, addr - base)[0]
        ins = DISASM.decode(word, addr)
        count += 1
        m = ins.mnemonic

        if m == "nop" or m == "sync":
            addr += 4
            continue

        if m == "jr":
            if not ins.is_return:
                raise Unsupported("indirect jump")
            # The delay slot still executes, so fold it in before finishing.
            slot_addr = addr + 4
            if slot_addr < end:
                slot_word = struct.unpack_from("<I", image, slot_addr - base)[0]
                slot = DISASM.decode(slot_word, slot_addr)
                if slot.mnemonic in _STORE_SIZES:
                    size = _STORE_SIZES[slot.mnemonic]
                    if size == 16:
                        raise Unsupported("128-bit store")
                    target = simplify(
                        Binary("add", get(slot.rs), Const(slot.imm & MASK64))
                    )
                    stores.append(Store(target, get(slot.rt), size))
                elif slot.mnemonic not in ("nop", "sync"):
                    _apply(regs, slot, get)
            return Derivation(start, get(2), count + 1, tuple(stores))

        if ins.is_branch or ins.is_jump:
            raise Unsupported("control flow: %s" % m)

        if m in _STORE_SIZES:
            size = _STORE_SIZES[m]
            if size == 16:
                # A quadword store needs the upper half of the register, which the
                # expression model does not carry.
                raise Unsupported("128-bit store")
            target = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
            stores.append(Store(target, get(ins.rt), size))
            addr += 4
            continue

        if ins.mnemonic in _LOAD_SIZES and stores:
            # Loading after storing would make the result depend on the order the
            # stores were applied, which `evaluate_stores` does not model. Rare
            # enough to refuse rather than complicate.
            raise Unsupported("loads after a store")

        _apply(regs, ins, get)
        addr += 4

    raise Unsupported("no return reached")


def _apply_vu0(regs: dict, ins, get) -> None:
    """VU0 macro mode over lane expressions, following vu0.execute."""
    from . import vu0

    word, m = ins.raw, ins.mnemonic
    base = m.split(".")[0]
    fd, fs, ft = (word >> 6) & 0x1F, (word >> 11) & 0x1F, (word >> 16) & 0x1F
    dest = (word >> 21) & 0xF
    lanes = [lane for lane, bit in enumerate((8, 4, 2, 1)) if dest & bit]

    def lane(index, which):
        if index == 0:
            return Const((0, 0, 0, vu0.ONE)[which])
        return get(vf_lane(index, which))

    def write(index, values):
        if index:
            for which in lanes:
                regs[vf_lane(index, which)] = values[which]

    if m == "qmtc2":
        index = (word >> 11) & 0x1F
        if index:
            low = get(ins.rt) if ins.rt else Const(0)
            try:
                high = _upper_or_zero(get, ins.rt)
            except Unsupported:
                # The upper half is the caller's: z and w are unknown, and refused
                # only if something reads them - code using x and y still derives.
                high = None
            for which, (half, shift) in enumerate(((low, 0), (low, 32), (high, 0), (high, 32))):
                if half is None:
                    regs.pop(vf_lane(index, which), None)
                    continue
                part = Binary("dshr", half, Const(shift)) if shift else half
                regs[vf_lane(index, which)] = Binary("and", part, Const(MASK32))
        return
    if m == "qmfc2":
        index = (word >> 11) & 0x1F
        if ins.rt:
            x, y, z, w = (lane(index, which) for which in range(4))
            regs[ins.rt] = Binary("or", x, Binary("dshl", y, Const(32)))
            regs[upper(ins.rt)] = Binary("or", z, Binary("dshl", w, Const(32)))
        return
    if m in ("ctc2", "cfc2"):
        index = (word >> 11) & 0x1F
        if m == "cfc2":
            if index in vu0.UNREADABLE:
                raise Unsupported("cfc2 of an unmodelled VU0 flag register")
            if ins.rt:
                regs[ins.rt] = SignExtend32(get(VC_BASE + index)) if index else Const(0)
            return
        value = get(ins.rt) if ins.rt else Const(0)
        if index < 16:
            if index:
                regs[VC_BASE + index] = Binary("and", value, Const(0xFFFF))
        elif index == vu0.REG_R:
            regs[VC_BASE + index] = Binary("or", Binary("and", value, Const(0x7FFFFF)),
                                           Const(vu0.ONE))
        else:
            regs[VC_BASE + index] = Binary("and", value, Const(MASK32))
        return

    op, to_acc, operand = vu0._split(base)
    if op is not None and base not in ("vopmula", "vopmsub"):
        # Only the lanes the mask selects are read: x/y-only maths on a register
        # whose z and w are unknown is fine.
        if operand in ("x", "y", "z", "w"):
            broadcast = lane(ft, "xyzw".index(operand))
            second = lambda which: broadcast
        elif operand in ("q", "i"):
            special = get(VC_BASE + (vu0.REG_Q if operand == "q" else vu0.REG_I))
            second = lambda which: special
        elif operand == "":
            second = lambda which: lane(ft, which)
        else:
            raise Unsupported("VU0 %s" % base)
        out = [None] * 4
        for which in lanes:
            a = lane(fs, which)
            if op in ("madd", "msub"):
                out[which] = VuOp(op, get(VACC_BASE + which), a, second(which))
            else:
                out[which] = VuOp(op, a, second(which))
        if to_acc:
            for which in lanes:
                regs[VACC_BASE + which] = out[which]
        else:
            write(fd, out)
        return
    if base == "vopmula":
        pairs = ((1, 2), (2, 0), (0, 1))
        for which in lanes:
            if which < 3:
                i, j = pairs[which]
                regs[VACC_BASE + which] = VuOp("mul", lane(fs, i), lane(ft, j))
        return
    if base == "vopmsub":
        pairs = ((1, 2), (2, 0), (0, 1))
        out = [None] * 4
        for which in lanes:
            if which < 3:
                i, j = pairs[which]
                out[which] = VuOp("msub", get(VACC_BASE + which), lane(fs, i), lane(ft, j))
        if fd:
            for which in lanes:
                if which < 3:
                    regs[vf_lane(fd, which)] = out[which]
        return
    if base in ("vmove", "vmr32"):
        rotate = 1 if base == "vmr32" else 0
        write(ft, [lane(fs, (which + rotate) % 4) for which in range(4)])
        return
    if base == "vabs":
        write(ft, [VuOp("abs", lane(fs, which)) for which in range(4)])
        return
    if base[:5] in ("vitof", "vftoi"):
        kind = "itof" if base.startswith("vitof") else "ftoi"
        write(ft, [VuOp(kind, lane(fs, which), shift=int(base[5:])) for which in range(4)])
        return
    if base in ("vnop", "vwaitq"):
        return
    fsf, ftf = (word >> 21) & 3, (word >> 23) & 3
    if base == "vdiv":
        regs[VC_BASE + vu0.REG_Q] = VuOp("div", lane(fs, fsf), lane(ft, ftf))
        return
    if base == "vsqrt":
        regs[VC_BASE + vu0.REG_Q] = VuOp("sqrt", lane(ft, ftf))
        return
    if base == "vrsqrt":
        regs[VC_BASE + vu0.REG_Q] = VuOp("rsqrt", lane(fs, fsf), lane(ft, ftf))
        return
    raise Unsupported("no symbolic rule for %s" % base)


def _apply_quad(regs: dict, ins, get) -> None:
    """The 128-bit moves and logic, over (low, upper) pairs."""
    m, rd, rs, rt = ins.mnemonic, ins.rd, ins.rs, ins.rt
    if rd == 0:
        return
    if m == "pcpyld":
        low, high = get(rt), get(rs)
    elif m == "pcpyud":
        low, high = _upper_or_zero(get, rs), _upper_or_zero(get, rt)
    else:
        op = {"por": "or", "pand": "and", "pxor": "xor", "pnor": "or"}[m]
        low = Binary(op, get(rs), get(rt))
        try:
            high = Binary(op, _upper_or_zero(get, rs), _upper_or_zero(get, rt))
        except Unsupported:
            high = None          # an upper half nobody wrote: unknown, refused if read
        if m == "pnor":
            low = Not(low)
            high = Not(high) if high is not None else None
    regs[rd] = low
    if high is None:
        regs.pop(upper(rd), None)
    else:
        regs[upper(rd)] = high


def _apply(regs: dict, ins, get, epoch: int = 0) -> None:
    """Apply one instruction to the symbolic register state."""
    m = ins.mnemonic

    if m == "lq":
        # Both halves, from the quadword the hardware aligns the address down to.
        address = address_of(Binary("and", Binary("add", get(ins.rs), Const(ins.imm & MASK64)),
                                    Const(~0xF & MASK64)))
        if ins.rt:
            regs[ins.rt] = Load(address, 8, False, epoch)
            regs[upper(ins.rt)] = Load(address_of(Binary("add", address, Const(8))), 8, False,
                                       epoch)
        return
    if m in ("pcpyld", "pcpyud", "por", "pand", "pxor", "pnor"):
        _apply_quad(regs, ins, get)
        return
    if ins.raw >> 26 == 0x12:
        _apply_vu0(regs, ins, get)
        return
    if m in _LOAD_SIZES:
        size, signed = _LOAD_SIZES[m]
        address = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        regs[ins.rt] = Load(address, size, signed, epoch)
        return

    if m == "lui":
        # The result is a 32-bit value sign-extended into the 64-bit register, which
        # matters as soon as anything manipulates it 64 bits wide: `lui 0xFFFF`
        # followed by `dsrl32 ..., 32` yields 0xFFFFFFFF, not 0. Dropping the
        # sign-extension produced exactly that wrong mask.
        regs[ins.rt] = Const(_sign32((ins.imm & 0xFFFF) << 16) & MASK64)
        return
    if m in ("addiu", "addi"):
        regs[ins.rt] = SignExtend32(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        return
    if m in ("daddiu", "daddi"):
        regs[ins.rt] = Binary("add", get(ins.rs), Const(ins.imm & MASK64))
        return
    if m == "andi":
        regs[ins.rt] = Binary("and", get(ins.rs), Const(ins.imm & 0xFFFF))
        return
    if m == "ori":
        regs[ins.rt] = Binary("or", get(ins.rs), Const(ins.imm & 0xFFFF))
        return
    if m == "xori":
        regs[ins.rt] = Binary("xor", get(ins.rs), Const(ins.imm & 0xFFFF))
        return
    if m == "sltiu":
        regs[ins.rt] = Compare("ltu", get(ins.rs), Const(ins.imm & MASK64))
        return
    if m == "slti":
        regs[ins.rt] = Compare("lt", get(ins.rs), Const(ins.imm & MASK64))
        return

    # Word operations sign-extend their 32-bit result into the 64-bit register;
    # the doubleword forms keep all 64 bits. The model used to give both the
    # 64-bit result, which only shows where upper bits are observed - a `slt` on
    # the result of an `sllv` - and so went unnoticed until arguments other than
    # small pointers reached the tests. `simplify` removes the extension again
    # wherever upper bits cannot matter.
    if m in ("addu", "add"):
        regs[ins.rd] = SignExtend32(Binary("add", get(ins.rs), get(ins.rt)))
        return
    if m in ("daddu", "dadd"):
        regs[ins.rd] = Binary("add", get(ins.rs), get(ins.rt))
        return
    if m in ("subu", "sub"):
        regs[ins.rd] = SignExtend32(Binary("sub", get(ins.rs), get(ins.rt)))
        return
    if m in ("dsubu", "dsub"):
        regs[ins.rd] = Binary("sub", get(ins.rs), get(ins.rt))
        return
    if m == "and":
        regs[ins.rd] = Binary("and", get(ins.rs), get(ins.rt))
        return
    if m == "or":
        regs[ins.rd] = Binary("or", get(ins.rs), get(ins.rt))
        return
    if m == "xor":
        regs[ins.rd] = Binary("xor", get(ins.rs), get(ins.rt))
        return
    if m == "sltu":
        regs[ins.rd] = Compare("ltu", get(ins.rs), get(ins.rt))
        return
    if m == "slt":
        regs[ins.rd] = Compare("lt", get(ins.rs), get(ins.rt))
        return

    if m == "sll":
        regs[ins.rd] = SignExtend32(Binary("shl", get(ins.rt), Const(ins.imm)))
        return
    if m == "srl":
        regs[ins.rd] = SignExtend32(Binary("shr", get(ins.rt), Const(ins.imm)))
        return
    if m == "sra":
        regs[ins.rd] = Binary("sar", get(ins.rt), Const(ins.imm))
        return
    if m == "dsll":
        regs[ins.rd] = Binary("dshl", get(ins.rt), Const(ins.imm))
        return
    if m == "dsrl":
        regs[ins.rd] = Binary("dshr", get(ins.rt), Const(ins.imm))
        return
    if m == "dsra":
        regs[ins.rd] = Binary("dsar", get(ins.rt), Const(ins.imm))
        return
    # The 32-prefixed forms add 32 to the shift amount.
    if m == "dsll32":
        regs[ins.rd] = Binary("dshl", get(ins.rt), Const(ins.imm + 32))
        return
    if m == "dsrl32":
        regs[ins.rd] = Binary("dshr", get(ins.rt), Const(ins.imm + 32))
        return
    if m == "dsra32":
        regs[ins.rd] = Binary("dsar", get(ins.rt), Const(ins.imm + 32))
        return
    # Variable shifts take their amount from a register.
    if m == "sllv":
        regs[ins.rd] = SignExtend32(Binary("shl", get(ins.rt), get(ins.rs)))
        return
    if m == "srlv":
        regs[ins.rd] = SignExtend32(Binary("shr", get(ins.rt), get(ins.rs)))
        return
    if m == "srav":
        regs[ins.rd] = Binary("sar", get(ins.rt), get(ins.rs))
        return
    if m == "dsllv":
        regs[ins.rd] = Binary("dshl", get(ins.rt), get(ins.rs))
        return
    if m == "dsrlv":
        regs[ins.rd] = Binary("dshr", get(ins.rt), get(ins.rs))
        return
    if m == "dsrav":
        regs[ins.rd] = Binary("dsar", get(ins.rt), get(ins.rs))
        return

    if m == "nor":
        regs[ins.rd] = Not(Binary("or", get(ins.rs), get(ins.rt)))
        return

    # Conditional moves leave the destination untouched when the condition fails,
    # so the previous value is part of the result. Without it there is nothing to
    # express, hence the refusal rather than a guess.
    if m in ("movz", "movn"):
        if ins.rd not in regs:
            raise Unsupported("conditional move over an uninitialised destination")
        op = "eq" if m == "movz" else "ne"
        regs[ins.rd] = Select(
            Predicate(op, get(ins.rt), Const(0)), get(ins.rs), regs[ins.rd]
        )
        return

    # Multiply and divide deposit their results in HI/LO, which `mfhi`/`mflo` then
    # read. Modelling the pair keeps the common "multiply then mflo" sequence
    # derivable.
    if m == "mult":
        # A signed 32x32 -> 64 multiply. Multiplying the raw 64-bit register values
        # would be wrong whenever an operand has its high bits set, so both are
        # narrowed to 32 bits and sign-extended first. LO holds the low word
        # sign-extended; HI the high word.
        product = Binary("mul", SignExtend32(get(ins.rs)), SignExtend32(get(ins.rt)))
        regs[REG_LO] = SignExtend32(product)
        regs[REG_HI] = SignExtend32(Binary("dsar", product, Const(32)))
        if ins.rd:
            regs[ins.rd] = regs[REG_LO]
        return
    if m == "multu":
        # Zero-extended 32-bit operands: their 64-bit product cannot overflow.
        product = Binary("mul", Binary("and", get(ins.rs), Const(MASK32)),
                         Binary("and", get(ins.rt), Const(MASK32)))
        regs[REG_LO] = SignExtend32(product)
        regs[REG_HI] = SignExtend32(Binary("dshr", product, Const(32)))
        if ins.rd:
            regs[ins.rd] = regs[REG_LO]
        return
    if m in ("div", "divu"):
        # Defined for every input, a zero divisor included (emu.divide).
        suffix = "u" if m == "divu" else ""
        regs[REG_LO] = IntDiv("quot" + suffix, get(ins.rs), get(ins.rt))
        regs[REG_HI] = IntDiv("rem" + suffix, get(ins.rs), get(ins.rt))
        return
    # -- coprocessor 1 (single precision) ---------------------------------
    #
    # COP1 R-format packs its registers differently from the integer unit, and the
    # disassembler does not populate rs/rt/rd for those forms, so the fields come
    # straight from the encoding.
    if m in _FLOAT_BINARY:
        fd, fs, ft = (ins.raw >> 6) & 0x1F, (ins.raw >> 11) & 0x1F, (ins.raw >> 16) & 0x1F
        regs[fpr(fd)] = FloatOp(_FLOAT_BINARY[m], get(fpr(fs)), get(fpr(ft)))
        return
    if m in _FLOAT_ACCUMULATE:
        fd, fs, ft = (ins.raw >> 6) & 0x1F, (ins.raw >> 11) & 0x1F, (ins.raw >> 16) & 0x1F
        op, target = _FLOAT_ACCUMULATE[m]
        if op in ("madd", "msub"):
            value = FloatOp(op, get(FP_ACC), get(fpr(fs)), get(fpr(ft)))
        else:
            value = FloatOp(op, get(fpr(fs)), get(fpr(ft)))
        regs[FP_ACC if target == "acc" else fpr(fd)] = value
        return
    if m in _FLOAT_UNARY:
        fd, fs = (ins.raw >> 6) & 0x1F, (ins.raw >> 11) & 0x1F
        if m == "sqrt.s":
            fs = (ins.raw >> 16) & 0x1F      # sqrt.s fd, ft: the operand is in ft
        regs[fpr(fd)] = FloatOp(_FLOAT_UNARY[m], get(fpr(fs)))
        return
    if m == "cvt.s.w":
        fd, fs = (ins.raw >> 6) & 0x1F, (ins.raw >> 11) & 0x1F
        regs[fpr(fd)] = IntToFloat(get(fpr(fs)))
        return
    if m == "cvt.w.s":
        fd, fs = (ins.raw >> 6) & 0x1F, (ins.raw >> 11) & 0x1F
        regs[fpr(fd)] = FloatToInt(get(fpr(fs)))
        return
    if m in _FLOAT_COMPARE:
        fs, ft = (ins.raw >> 11) & 0x1F, (ins.raw >> 16) & 0x1F
        regs[FP_COND] = FloatPredicate(_FLOAT_COMPARE[m], get(fpr(fs)), get(fpr(ft)))
        return
    if m == "mtc1":
        regs[fpr(ins.rd)] = get(ins.rt)
        return
    if m == "mfc1":
        regs[ins.rt] = SignExtend32(get(fpr(ins.rd)))
        return

    if m == "mfhi":
        regs[ins.rd] = get(REG_HI)
        return
    if m == "mflo":
        regs[ins.rd] = get(REG_LO)
        return
    if m == "mthi":
        regs[REG_HI] = get(ins.rs)
        return
    if m == "mtlo":
        regs[REG_LO] = get(ins.rs)
        return

    raise Unsupported("no symbolic rule for %s" % m)
