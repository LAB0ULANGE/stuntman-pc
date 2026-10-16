"""Derive branching functions as guarded paths.

`symexec.py` handles straight-line functions, which covers accessors but little
else: most small functions have a conditional in them.  Those can still be derived
*exactly*, provided the control flow is acyclic - enumerate every route from the
entry to a return, record the branch conditions holding along each, and the
function becomes an ordered list of guarded results.

Loops are refused outright.  A loop's result depends on a trip count that is not a
function of the arguments alone, so no finite expression describes it; emitting
something anyway would be the plausible-but-wrong output this whole approach exists
to avoid.  Calls are refused for the same reason - the callee would have to be
derived first.

Everything here still ends up differential-tested against the original machine
code, so a mistake in the path logic surfaces as a disagreement rather than as
quietly wrong source.
"""

from __future__ import annotations

import dataclasses
import struct

from .r5900 import DISASM
from .symexec import (
    ARG_COUNT,
    vu_written,
    VU_KEYS,
    vf_lane,
    QUAD_WRITERS,
    UPPER_BASE,
    upper,
    address_of,
    low32,
    FLOAT_ARG_BASE,
    FLOAT_ARG_REGS,
    INT_ARG_REGS,
    MASK32,
    MASK64,
    Arg,
    Binary,
    CallResult,
    LoopExit,
    Var,
    Const,
    Expr,
    Load,
    REG_HI,
    REG_LO,
    rebuild,
    Predicate,
    Select,
    SignExtend32,
    Store,
    Unsupported,
    FP_COND,
    Not1,
    children,
    _apply,
    fpr,
    _LOAD_SIZES,
    substitute,
    _sign32,
    _STORE_SIZES,
    simplify,
)

# Caps that keep a pathological function from exploding the enumeration. A small
# accessor or predicate has a handful of paths; anything past these is not the kind
# of function this module is meant for.
MAX_PATHS = 24

# Loop unrolling bounds. MAX_UNROLL caps how often any one instruction may execute on
# a single path, i.e. the largest trip count that will be unrolled. MAX_PATH_STEPS
# caps the instructions on a path overall, so a modest loop nested in another cannot
# multiply into something enormous. Beyond these the function is refused as looping.
MAX_UNROLL = 64
MAX_PATH_STEPS = 2048

# Total instructions walked across *every* path of one function. MAX_PATHS alone is
# not a brake: it counts completed paths, and a loop whose exit depends on memory can
# fork on every iteration for a very long time before any path completes. One such
# function - a ring-buffer walk at 0x0020DD08, forking four ways per iteration - grew
# roughly as 4^64 and never returned. Capping the work, not the results, bounds it.
MAX_TOTAL_STEPS = 20_000

# Largest expression, counted as a tree, that a derivation may hold. Step counts do
# not bound this: an unrolled loop that uses a value twice per iteration doubles the
# tree each time round while taking only a handful of instructions to do it, so the
# derivation stalls in building and simplifying nodes long before any step limit
# trips. A result this large would also be unreadable as source, so refusing it loses
# nothing a reconstruction could have used.
MAX_EXPR_SIZE = 4000

# Key under which a path's total step count is kept alongside its per-address
# counts. No instruction address can collide with it.
STEPS = -1

# `$sp` is seeded with the same value the interpreter uses, so stack addresses fold
# to constants and the frame can be modelled separately from real memory.
#
# This matters more than it looks. Over a thousand functions were refused purely for
# reading `$sp`, because they have an ordinary prologue: reserve a frame, spill the
# callee-saved registers, do the work, reload them, return. Modelling the frame makes
# those spill/reload pairs cancel out, leaving the function's actual behaviour.
from . import layout
from .layout import MAX_FRAME_SIZE, callee_stack_top  # noqa: F401  (part of this API)


def _is_frame_address(expr) -> bool:
    """True when an address is a constant inside the current stack frame."""
    folded = simplify(expr)
    if not isinstance(folded, Const):
        return False
    value = folded.value & MASK64
    top = layout.CURRENT.stack_top
    return top - MAX_FRAME_SIZE <= value <= top + 0x10


@dataclasses.dataclass(frozen=True)
class Call:
    """A call the derived code makes to a function that has not been derived.

    `index` numbers the opaque calls along a path; memory after call k is epoch k.
    `arguments` pairs each argument-vector index the callee reads with the
    expression the caller passes.
    """

    index: int
    target: int
    arguments: tuple
    # For a call through a register - a function pointer or a virtual method - the
    # expression the target address is taken from; `target` is then -1.
    pointer: Expr = None


@dataclasses.dataclass(eq=False)
class Loop:
    """A loop whose trip count depends on its input, derived once over variables.

    `carried` are the registers the body writes; each is a Var inside the body,
    starting from `init` (the value before the loop, where there was one). The
    body is a statement tree whose leaves continue - giving every variable its
    next value - break to one of `exits`, or return from the function. Each
    iteration begins a new memory epoch (`head_epoch`), and so does leaving the
    loop (`after_epoch`).
    """

    id: int
    head: int
    init: tuple
    carried: tuple
    body: list
    head_epoch: int
    after_epoch: int
    exits: int


@dataclasses.dataclass
class LoopContext:
    id: int
    head: int
    body: frozenset
    carried: tuple
    exits: list


MAX_ITERATIONS = 20_000


@dataclasses.dataclass
class Path:
    """One route through a function: the conditions holding, and the outcome."""

    conditions: tuple      # Predicates, all true along this path
    expression: Expr       # what $v0 holds on return
    stores: tuple = ()
    # Conditions and opaque calls in program order, each as (number of stores
    # before it, "cond" or "call", the predicate or Call). With the stores this
    # gives the path's full event order, which matters once a call sits between
    # them: a callee sees the stores before it and not those after.
    marks: tuple = ()
    # How a path inside a loop body ends, instead of returning: ("continue", loop,
    # assignments), ("break", loop, exit) - or ("unreachable",) after a loop that
    # can only be left by returning; and on any path, ("trap", address) at a
    # `break` instruction, which the original raises as an exception there.
    leaf: tuple = None

    def end(self) -> tuple:
        return self.leaf if self.leaf is not None else ("return", self.expression)

    @property
    def calls(self) -> tuple:
        return tuple(obj for _count, kind, obj in self.marks if kind == "call")

    def events(self) -> list:
        """Conditions, stores and calls, in the order the function performs them."""
        out = []
        marks = list(self.marks)
        position = 0
        for count, store in enumerate(self.stores):
            while position < len(marks) and marks[position][0] <= count:
                out.append((marks[position][1], marks[position][2]))
                position += 1
            out.append(("store", store))
        out.extend((kind, obj) for _count, kind, obj in marks[position:])
        return out

    def render_condition(self) -> str:
        if not self.conditions:
            return "true"
        return " && ".join("(%s)" % p.render() for p in self.conditions)

    def holds(self, args, read) -> bool:
        return all(p.evaluate(args, read) for p in self.conditions)


@dataclasses.dataclass
class PathDerivation:
    """A function described as an ordered list of guarded results."""

    address: int
    paths: tuple
    # What every path returns is $f0's 32 bits - a float - rather than $v0.
    returns_float: bool = False

    @property
    def is_straight_line(self) -> bool:
        return len(self.paths) == 1 and not self.paths[0].conditions

    @property
    def is_void(self) -> bool:
        """True when no path produces a value."""
        return all(path.expression is None for path in self.paths)

    @property
    def sequential(self) -> bool:
        """True when the derivation must be run as ordered statements: it calls an
        underived function, or has a barrier where a load follows stores it may
        overlap."""
        return any(kind in ("call", "barrier", "loop") for path in self.paths
                   for _count, kind, _obj in path.marks)

    @property
    def has_calls(self) -> bool:
        """True when the derived code calls a function that was not derived.

        Such a derivation cannot be described as guarded paths over entry memory -
        a branch after the call depends on what the call did - so it is evaluated
        and emitted as a decision tree instead (see `tree`).
        """
        return any(path.calls for path in self.paths)

    def tree(self) -> list:
        """The paths merged into one decision tree of statements.

        Statements are ("store", Store), ("call", Call), ("if", predicate, then,
        else) and ("return", expression or None). Paths share their events up to
        the condition where they part, and the walk always parts them on a
        predicate and its negation, so the merge is exact; anything else is
        refused rather than guessed at.
        """
        return _build_tree([(path.events(), path.end()) for path in self.paths])

    def run(self, args, execution) -> object:
        """Evaluate the tree against real memory, making the calls; see Execution.

        Returns the low 32 bits of $v0 as a signed value, or None when void.
        """
        result = _run_tree(self.tree(), args, execution)
        execution.flush()
        value = result.value if isinstance(result, _Returned) else None
        if value is None:
            return None
        return _sign32(value & MASK32)

    @property
    def stores(self) -> tuple:
        """Every store the function can perform, across all paths."""
        out = []
        for path in self.paths:
            out.extend(path.stores)
        return tuple(out)

    def _render_return(self, path) -> list:
        """The return statement for a path, or nothing when the function is void."""
        if path.leaf is not None and path.leaf[0] == "trap":
            return ["TrapAt(0x%08Xu);" % path.leaf[1]]
        if path.expression is None:
            return []
        return ["return %s;" % simplify(path.expression).render()]

    def render(self) -> str:
        if self.is_straight_line:
            path = self.paths[0]
            lines = [store.render() for store in path.stores]
            lines.extend(self._render_return(path))
            return "\n".join(lines) if lines else "/* no observable effect */"

        lines = []
        for path in self.paths[:-1]:
            lines.append("if (%s) {" % path.render_condition())
            for store in path.stores:
                lines.append("    " + store.render())
            returned = self._render_return(path)
            lines.extend("    " + line for line in returned)
            if not returned:
                # A void path still has to leave the function here, or control would
                # fall through into the next guard.
                lines.append("    return;")
            lines.append("}")
        last = self.paths[-1]
        for store in last.stores:
            lines.append(store.render())
        lines.extend(self._render_return(last))
        return "\n".join(lines)

    def select(self, args, read):
        """The first path whose conditions all hold, or None."""
        for path in self.paths:
            if path.holds(args, read):
                return path
        return None

    def evaluate(self, args, read):
        """The returned value, or None for a void function.

        A void function's `$v0` holds whatever the caller left there, so there is
        nothing to compare and the verifier must not try.
        """
        path = self.select(args, read)
        if path is None:
            raise Unsupported("no path matched: the condition set is incomplete")
        if path.leaf is not None and path.leaf[0] == "trap":
            raise Unsupported("reached the trap at 0x%08X" % path.leaf[1])
        if path.expression is None:
            return None
        return _sign32(simplify(path.expression).evaluate(args, read) & MASK32)

    def evaluate_stores(self, args, read) -> list:
        path = self.select(args, read)
        if path is None:
            raise Unsupported("no path matched: the condition set is incomplete")
        if path.leaf is not None and path.leaf[0] == "trap":
            raise Unsupported("reached the trap at 0x%08X" % path.leaf[1])
        out = []
        for store in path.stores:
            address = simplify(store.address).evaluate(args, read) & MASK32
            value = simplify(store.value).evaluate(args, read)
            out.append((address, value & ((1 << (store.size * 8)) - 1), store.size))
        return out


def _same_event(a, b) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "cond":
        return a[1] == b[1]
    return a[1] is b[1] or a[1] == b[1]


def _build_tree(group) -> list:
    """Merge (events, result) paths that share a prefix into nested statements."""
    statements = []
    position = 0
    while True:
        heads = [events[position] if position < len(events) else None
                 for events, _result in group]
        if all(head is None for head in heads):
            statements.append(group[0][1])
            return statements
        first = heads[0]
        if first is not None and all(head is not None and _same_event(head, first)
                                     for head in heads):
            statements.append(first)
            position += 1
            continue
        if any(head is None or head[0] != "cond" for head in heads):
            raise Unsupported("paths diverge without a condition")
        sides = []
        for head, member in zip(heads, group):
            for side in sides:
                if side[0] == head[1]:
                    side[1].append(member)
                    break
            else:
                sides.append((head[1], [member]))
        if len(sides) != 2 or not (_negates(sides[0][0], sides[1][0])):
            raise Unsupported("paths part on conditions that are not complements")
        (condition, taken), (_other, untaken) = sides
        statements.append((
            "if", condition,
            _build_tree([(events[position + 1:], result) for events, result in taken]),
            _build_tree([(events[position + 1:], result) for events, result in untaken]),
        ))
        return statements


def _negates(a, b) -> bool:
    try:
        return a.negate() == b or b.negate() == a
    except Exception:
        return False


class _Returned:
    """The function returned (from anywhere, a loop body included) with `value`."""

    def __init__(self, value):
        self.value = value


_CONTINUE = object()
_BREAK = object()


def _run_tree(statements, args, execution):
    for statement in statements:
        kind = statement[0]
        if kind == "store":
            store = statement[1]
            execution.store(
                simplify(store.address).evaluate(args, execution) & MASK32,
                simplify(store.value).evaluate(args, execution),
                store.size,
            )
        elif kind == "call":
            call = statement[1]
            vector = [0] * ARG_COUNT
            for index, expr in call.arguments:
                vector[index] = simplify(expr).evaluate(args, execution)
            target = call.target
            if call.pointer is not None:
                target = simplify(call.pointer).evaluate(args, execution) & MASK32
            execution.call(call.index, target, vector)
        elif kind == "barrier":
            execution.barrier(statement[1].index)
        elif kind == "loop":
            outcome = _run_loop(statement[1], args, execution)
            if outcome is not None:
                return outcome
        elif kind == "if":
            taken = statement[1].evaluate(args, execution)
            return _run_tree(statement[2] if taken else statement[3], args, execution)
        elif kind == "continue":
            _, loop, assignments = statement
            values = {register: simplify(expr).evaluate(args, execution)
                      for register, expr in assignments}
            for register, value in values.items():
                execution.vars[(loop, register)] = value & MASK64
            return _CONTINUE
        elif kind == "break":
            execution.exits[statement[1]] = statement[2]
            return _BREAK
        elif kind == "unreachable":
            return None
        elif kind == "trap":
            raise Unsupported("reached the trap at 0x%08X" % statement[1])
        else:
            expression = statement[1]
            if expression is None:
                return _Returned(None)
            return _Returned(simplify(expression).evaluate(args, execution))
    return None


def _run_loop(loop, args, execution):
    """Run a loop to completion; returns a _Returned if its body returned."""
    initial = dict(loop.init)
    values = {register: (simplify(initial[register]).evaluate(args, execution)
                         if register in initial else 0)
              for register in loop.carried}
    for register, value in values.items():
        execution.vars[(loop.id, register)] = value & MASK64
    for _iteration in range(MAX_ITERATIONS):
        execution.barrier(loop.head_epoch)
        outcome = _run_tree(loop.body, args, execution)
        if outcome is _CONTINUE:
            continue
        if outcome is _BREAK:
            execution.barrier(loop.after_epoch)
            return None
        if isinstance(outcome, _Returned):
            return outcome
        raise Unsupported("a loop body ended without continuing or leaving")
    raise Unsupported("a loop ran more than %d iterations" % MAX_ITERATIONS)


class Execution:
    """Evaluates a derivation with opaque calls against an interpreter's memory.

    The derivation's expressions refer to memory by epoch. Within one epoch the
    walk guaranteed that every load is either forwarded from, or provably clear of,
    the stores before it, so evaluating them against memory as the epoch began is
    exact - and that is what holding the stores back until the next call achieves.
    At a call the held stores are written, the *original* callee is run in the
    interpreter, and a new epoch begins. A load from an earlier epoch reads the
    current memory with every journalled write since that epoch began undone.

    Memory must be journalling, which the verifier's replay already arranges.
    """

    def __init__(self, machine, budget: int = 2_000_000) -> None:
        self.machine = machine
        self.budget = budget
        self.memory = machine.memory
        self.markers = {0: len(self.memory._journal)}
        self.results = {}
        self.float_results = {}
        self.pending = []
        self.vars = {}
        self.exits = {}

    def var(self, loop, register):
        return self.vars[(loop, register)]

    def exit_taken(self, loop):
        return self.exits.get(loop)

    def __call__(self, address, size, signed, epoch=0):
        raw = bytearray(self.memory.read(address, size))
        offset = self.memory.offset_of(address)
        for written, previous in reversed(self.memory._journal[self.markers[epoch]:]):
            for i, byte in enumerate(previous):
                position = written + i - offset
                if 0 <= position < size:
                    raw[position] = byte
        return int.from_bytes(bytes(raw), "little", signed=signed)

    def result(self, index):
        return self.results[index]

    def float_result(self, index):
        return self.float_results[index]

    def store(self, address, value, size):
        self.pending.append((address, value, size))

    def flush(self):
        for address, value, size in self.pending:
            self.memory.write_int(address, value, size)
        self.pending = []

    def barrier(self, index):
        self.flush()
        self.markers[index] = len(self.memory._journal)

    def call(self, index, target, vector):
        self.flush()
        # Below this function's frame, which may hold an out-parameter the callee
        # writes - running the callee at the usual stack top would trample it.
        self.machine.call(target, vector, max_instructions=self.budget,
                          stack_top=callee_stack_top())
        self.results[index] = self.machine.get(2) & MASK64
        self.float_results[index] = self.machine.fpr[0] & 0xFFFFFFFF
        self.markers[index] = len(self.memory._journal)


def _retag(expr, epoch: int, memo: dict):
    """A callee's expression, with its entry-memory loads moved to `epoch`."""
    key = id(expr)
    if key in memo:
        return memo[key]
    kids = children(expr)
    if isinstance(expr, Load):
        new = Load(_retag(expr.address, epoch, memo), expr.size, expr.signed, epoch)
    elif kids:
        new = rebuild(expr, [_retag(child, epoch, memo) for child in kids])
    else:
        new = expr
    memo[key] = new
    return new


# Where the last call's float result waits: not $f0 itself, which loops carry and
# merges compare like any register, but a slot no instruction writes. Reading $f0
# before this function writes it gives the call's result; nothing else sees it.
CALL_FLOAT = 1000


def _getter(regs: dict):
    def get(index: int) -> Expr:
        if index == 0:
            return Const(0)
        if index == fpr(0) and index not in regs and CALL_FLOAT in regs:
            return regs[CALL_FLOAT]
        if index not in regs:
            if UPPER_BASE <= index < UPPER_BASE + 32:
                raise Unsupported("reads the upper half of $%d, which it did not write"
                                  % (index - UPPER_BASE))
            raise Unsupported("reads uninitialised register $%d" % index)
        return regs[index]
    return get


def _constant_truth(predicate):
    """Decide a condition that depends on nothing, or return None.

    A condition is only decidable here when every leaf under it is a constant - no
    argument, and no memory load, since a load's value is not known until the test
    supplies memory. `Unsupported` from the evaluation is treated as undecidable
    rather than allowed to escape.
    """
    def all_constant(expr) -> bool:
        kids = children(expr)
        if not kids:
            return isinstance(expr, Const)
        if isinstance(expr, Load):
            return False
        return all(all_constant(child) for child in kids)

    try:
        if not all_constant(predicate):
            return None
        return bool(predicate.evaluate((0, 0, 0, 0), None))
    except Exception:
        return None


def _condition_for(ins, get):
    """The condition that is true exactly when `ins` takes its branch."""
    zero = Const(0)
    # The FPU branches test the flag a preceding `c.<cond>.s` set, so the condition
    # is whatever comparison was recorded rather than anything in this instruction.
    if ins.mnemonic in ("bc1t", "bc1tl"):
        return get(FP_COND)
    if ins.mnemonic in ("bc1f", "bc1fl"):
        return Not1(get(FP_COND))
    table = {
        "beq": ("eq", True), "beql": ("eq", True),
        "bne": ("ne", True), "bnel": ("ne", True),
        "blez": ("le", False), "blezl": ("le", False),
        "bgtz": ("gt", False), "bgtzl": ("gt", False),
        "bltz": ("lt", False), "bltzl": ("lt", False),
        "bgez": ("ge", False), "bgezl": ("ge", False),
    }
    entry = table.get(ins.mnemonic)
    if entry is None:
        raise Unsupported("no condition rule for %s" % ins.mnemonic)
    op, two_register = entry
    right = get(ins.rt) if two_register else zero
    return Predicate(op, get(ins.rs), right)


# -- memory aliasing ---------------------------------------------------------
#
# Every derived expression denotes a value in terms of memory as it stood on entry.
# Once a function has written memory, a later load has to be answered by asking
# which earlier write, if any, could have touched the bytes it reads. That needs a
# notion of when two accesses provably do or do not overlap.
#
# Addresses are compared as *base + constant offset*. When two accesses share a base
# expression, their relative position is known exactly even though the base's value
# is not, so arg0+4 and arg0+8 can be proven apart. This is sound because a base
# expression denotes an entry-state value: two structurally equal bases evaluate to
# the same address at any point in the function. Accesses on different bases, such
# as arg0+4 and arg1+4, may be the same location and are treated as unknown.

ADDRESS_SPACE = 1 << 32


def _split_address(address):
    """Split an address into (base expression or None, offset modulo 2^32)."""
    folded = address_of(address)
    if isinstance(folded, Const):
        return None, folded.value % ADDRESS_SPACE
    if isinstance(folded, Binary) and folded.op == "add":
        if isinstance(folded.right, Const):
            return folded.left, folded.right.value % ADDRESS_SPACE
        if isinstance(folded.left, Const):
            return folded.right, folded.left.value % ADDRESS_SPACE
    return folded, 0


def _relation(first, first_size, second, second_size) -> str:
    """How two accesses relate: 'same', 'disjoint', 'overlap' or 'unknown'.

    Widths are part of the answer. Comparing start addresses alone - which an earlier
    version did - would call a 4-byte write at 0x100 and a read at 0x102 independent.
    Offsets are compared modulo 2^32, because that is how addresses wrap.
    """
    base_a, offset_a = _split_address(first)
    base_b, offset_b = _split_address(second)
    if (base_a is None) != (base_b is None):
        return "unknown"
    if base_a is not None and not (base_a is base_b or base_a == base_b):
        return "unknown"
    distance = (offset_b - offset_a) % ADDRESS_SPACE
    if distance == 0 and first_size == second_size:
        return "same"
    if distance >= first_size and ADDRESS_SPACE - distance >= second_size:
        return "disjoint"
    return "overlap"


def _record_store(memory: dict, target, size: int, value) -> None:
    """Remember a write, keeping the overlay ordered from oldest to newest.

    Re-inserting rather than assigning matters: a plain assignment to an existing
    key keeps its original position, so a repeated write would look older than a
    write that actually came after it.
    """
    key = (target, size)
    memory.pop(key, None)
    memory[key] = value


def _forward(value, size: int, signed: bool):
    """The value a load of `size` bytes sees after a store of exactly those bytes.

    A store keeps only its low `size` bytes and the load then extends them, so
    `sb` followed by `lbu` yields the stored byte - not the whole register, which is
    what forwarding the raw value used to produce.
    """
    if size == 8:
        return value
    if size == 4:
        return SignExtend32(value) if signed else Binary("and", value, Const(0xFFFFFFFF))
    if size in (1, 2):
        if signed:
            shift = Const(64 - 8 * size)
            return Binary("dsar", Binary("dshl", value, shift), shift)
        return Binary("and", value, Const((1 << (8 * size)) - 1))
    raise Unsupported("cannot forward a %d-byte store" % size)


@dataclasses.dataclass(frozen=True)
class Barrier:
    """The point where held stores must reach memory before a load can be answered.

    A load that may or may not overlap an earlier store cannot be expressed against
    the memory the epoch began with. Rather than refuse the function, the walk
    starts a new epoch here: the stores so far are applied, and the load reads
    memory as it then is - exactly what the original does. It is an opaque call
    with no callee.
    """

    index: int


class Overlay(dict):
    """The function's own writes in the current epoch, plus the epoch itself.

    The epoch travels with the overlay because a barrier can arise deep inside an
    instruction step; barriers raised are collected in `raised` for the walk to
    place among its ordered events right after the step.
    """

    def __init__(self, *args, epoch: int = 0, counter=None):
        super().__init__(*args)
        self.epoch = epoch
        self.raised = []
        # Shared by a whole derivation, so every epoch - from a barrier, a call or
        # a loop - has its own number. Numbers local to a path were enough while
        # paths never met again; a loop's body and what follows it do.
        self.counter = counter

    def fork(self) -> "Overlay":
        return Overlay(self, epoch=self.epoch, counter=self.counter)

    def next_epoch(self) -> int:
        if self.counter is None:
            return self.epoch + 1
        self.counter[0] += 1
        return self.counter[0]

    def barrier(self, store_count: int) -> int:
        self.clear()
        self.epoch = self.next_epoch()
        self.raised.append((store_count, "barrier", Barrier(self.epoch)))
        return self.epoch


class Frame(dict):
    """The function's stack frame: symbolic slots, until its address escapes.

    While no callee can see it, the frame is a map of slots to the values written
    there, and a spill and its reload simply cancel. Once a frame address is passed
    to a function that is not derived - an out-parameter, `GetPosition(&local)` - the
    callee can read and write it, so the frame has to become real memory: the
    slots are flushed as stores and later accesses go to memory. Sixteen-byte
    register spills stay symbolic; no callee touches a caller's saved registers.
    """

    def __init__(self, *args, escaped: bool = False, seeds=frozenset(), frozen=False):
        super().__init__(*args)
        self.escaped = escaped
        # Inside a loop body the frame must not change: a slot written on one
        # iteration and read on the next would need a loop variable of its own.
        self.frozen = frozen
        # The values the walk invents for callee-saved registers and $ra on entry.
        # Spilling one is bookkeeping, not data: flushing it would write a made-up
        # value - the interpreter's return sentinel - into the guest stack.
        self.seeds = seeds

    def fork(self) -> "Frame":
        return Frame(self, escaped=self.escaped, seeds=self.seeds, frozen=self.frozen)

    def escape(self, stores: list, memory) -> None:
        if self.escaped:
            return
        if self.frozen:
            raise Unsupported("a loop passes the address of a local")
        for (address, size), value in list(self.items()):
            if id(value) in self.seeds:
                continue
            if value is UNKNOWN_HALF:
                if id(self.get((address - 8, 8))) in self.seeds:
                    continue            # the upper half of a saved register's spill
                raise Unsupported("128-bit store to real memory")
            for store in _as_stores(Const(address), value, size):
                stores.append(store)
                _record_store(memory, store.address, store.size, store.value)
            del self[(address, size)]
        self.escaped = True

    def symbolic(self, address: int, size: int) -> bool:
        """Whether an access is answered by the symbolic slots rather than memory."""
        if not self.escaped:
            return True
        return any(not (address + size <= written or written + width <= address)
                   for (written, width) in self)


def _as_stores(address, value, size: int) -> list:
    """A write as stores the model can express.

    The expression model is 64 bits wide, so a 16-byte value cannot be stored as
    one. The common 16-byte write, though, is the EE's struct copy - `lq` then
    `sq` - and a 16-byte store of exactly a 16-byte load is two 8-byte copies.
    Anything else that wide is refused.
    """
    if size <= 8:
        return [Store(address, value, size)]
    if size == 16 and isinstance(value, Load) and value.size == 16:
        high = address_of(Binary("add", address, Const(8)))
        return [
            Store(address, Load(value.address, 8, False, value.epoch), 8),
            Store(high, Load(address_of(Binary("add", value.address, Const(8))), 8,
                             False, value.epoch), 8),
        ]
    raise Unsupported("%d-bit store to real memory" % (8 * size))


class _UnknownHalf:
    """The upper half of a 128-bit spill whose register's upper half is unknown."""

    def __repr__(self):
        return "UNKNOWN_HALF"


UNKNOWN_HALF = _UnknownHalf()


def _frame_spill_quad(frame: dict, address: int, low, high) -> None:
    """A 16-byte spill, as its two 8-byte halves - so that a read of either half,
    or of bytes spanning them, finds what was written there."""
    _frame_write(frame, address, 8, low)
    _frame_write(frame, address + 8, 8, UNKNOWN_HALF if high is None else high)


def _frame_write(frame: dict, address: int, size: int, value) -> None:
    """Record a write to the stack frame, newest last."""
    if getattr(frame, "frozen", False):
        raise Unsupported("a loop writes the stack frame")
    frame.pop((address, size), None)
    frame[(address, size)] = value


def _frame_read(frame: dict, address: int, size: int, signed: bool):
    """A stack-frame read, answered byte by byte from the newest writes covering it.

    A same-width reload of an 8-byte spill returns exactly what was saved - a
    callee-saved register round trip must cancel - and a narrower one takes the
    right bytes and extends them as the load does. A read spanning several
    writes - a word written in two halves, a field stored over part of a wider
    one - is assembled from the bytes each contributes. Slots used to be looked up
    by exact (address, width), and then by the single newest overlapping write,
    which refused every such read.
    """
    entries = list(frame.items())
    pieces = []
    for offset in range(size):
        byte = address + offset
        for (written, width), value in reversed(entries):
            if written <= byte < written + width:
                pieces.append((written, width, value))
                break
        else:
            pieces.append(None)
    if all(piece is None for piece in pieces):
        raise Unsupported("reads an unwritten stack slot")
    if any(piece is None for piece in pieces):
        raise Unsupported("reads a partly overwritten stack slot")
    if any(piece[2] is UNKNOWN_HALF for piece in pieces):
        raise Unsupported("reads beyond the modelled 64 bits of a stack slot")
    written, width, value = pieces[0]
    if all(piece[:2] == (written, width) for piece in pieces):
        if size == width and size >= 8:
            return value
        delta = address - written
        if delta:
            value = simplify(Binary("dshr", value, Const(8 * delta)))
        return _forward(value, size, signed)
    result, offset = None, 0
    while offset < size:
        written, width, value = pieces[offset]
        run = 1
        while offset + run < size and pieces[offset + run][:2] == (written, width):
            run += 1
        part = value
        delta = address + offset - written
        if delta:
            part = Binary("dshr", part, Const(8 * delta))
        if run < 8:
            part = Binary("and", part, Const((1 << (8 * run)) - 1))
        if offset:
            part = Binary("dshl", part, Const(8 * offset))
        result = part if result is None else Binary("or", result, part)
        offset += run
    return _forward(simplify(result), size, signed)


def _symbolic_frame(frame, target, size: int) -> bool:
    if not isinstance(frame, Frame):
        return True
    return frame.symbolic(simplify(target).value & MASK64, size)


def _arg_register(index: int) -> int:
    """The register an argument-vector index lives in."""
    if index < FLOAT_ARG_BASE:
        return INT_ARG_REGS[index]
    return fpr(FLOAT_ARG_REGS[index - FLOAT_ARG_BASE])


def _drain(memory, marks: tuple) -> tuple:
    """Marks with any barriers the last step raised appended, in order."""
    if isinstance(memory, Overlay) and memory.raised:
        marks = marks + tuple(memory.raised)
        memory.raised = []
    return marks


def _read_through(memory: dict, target, size: int, signed: bool, epoch: int = 0,
                  store_count: int = 0):
    """Answer a load given everything written so far.

    Writes are examined newest first. The first one that could touch the loaded
    bytes decides: an exact match is forwarded, a partial or unprovable overlap is
    refused, and only if every write is provably elsewhere does the load read
    entry-state memory. Matching on exact address alone - the earlier approach -
    ignored a newer write that overlapped only part of an older one.
    """
    for (address, written_size), value in reversed(list(memory.items())):
        relation = _relation(address, written_size, target, size)
        if relation == "disjoint":
            continue
        if relation == "same":
            try:
                return _forward(value, size, signed)
            except Unsupported:
                if not isinstance(memory, Overlay):
                    raise
        elif not isinstance(memory, Overlay):
            raise Unsupported("load may alias an earlier store")
        # Undecidable from addresses alone: apply the held stores and read memory.
        return Load(target, size, signed, memory.barrier(store_count))
    return Load(target, size, signed, epoch)


def _step(regs: dict, stores: list, frame: dict, memory: dict, ins, get,
          epoch: int = 0) -> None:
    """Apply one non-control instruction.

    Accesses that land in the stack frame are routed to `frame` rather than being
    recorded as real memory effects, so a prologue's register spills and the
    matching epilogue reloads cancel instead of appearing as writes the caller
    could observe.

    `memory` tracks what the function has already written to real memory. Loads
    consult it first, because the derived expressions all refer to memory as it
    stood on entry: a function that stores a value and then reads the same location
    back - a setter returning the field it just set is the common shape - would
    otherwise be derived against stale contents.
    """
    if isinstance(memory, Overlay):
        epoch = memory.epoch
    m = ins.mnemonic
    if m in ("nop", "sync"):
        return

    if m == "swc1":
        target = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        value = get(fpr(ins.rt))
        if _is_frame_address(target) and _symbolic_frame(frame, target, 4):
            _frame_write(frame, simplify(target).value & MASK64, 4, value)
            return
        stores.append(Store(target, value, 4))
        _record_store(memory, target, 4, value)
        return

    if m in ("lqc2", "sqc2"):
        # A VF register to or from a quadword, lane by lane.
        base_address = address_of(Binary("and", Binary("add", get(ins.rs),
                                                       Const(ins.imm & MASK64)),
                                         Const(~0xF & MASK64)))
        for which in range(4):
            target = address_of(Binary("add", base_address, Const(4 * which)))
            if _is_frame_address(target):
                # A spill: lane by lane through the frame model, like any other.
                if not _symbolic_frame(frame, target, 4):
                    raise Unsupported("VU0 register spilled to the stack frame")
                slot = simplify(target).value & MASK64
                if m == "lqc2":
                    if ins.rt:
                        regs[vf_lane(ins.rt, which)] = _frame_read(frame, slot, 4, False)
                else:
                    value = (Const((0, 0, 0, 0x3F800000)[which]) if ins.rt == 0
                             else get(vf_lane(ins.rt, which)))
                    _frame_write(frame, slot, 4, value)
                continue
            if m == "lqc2":
                if ins.rt:
                    regs[vf_lane(ins.rt, which)] = _read_through(memory, target, 4, False,
                                                                 epoch, len(stores))
            else:
                value = (Const((0, 0, 0, 0x3F800000)[which]) if ins.rt == 0
                         else get(vf_lane(ins.rt, which)))
                stores.append(Store(target, value, 4))
                _record_store(memory, target, 4, value)
        return

    if m == "lwc1":
        target = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        if _is_frame_address(target) and _symbolic_frame(frame, target, 4):
            regs[fpr(ins.rt)] = _frame_read(frame, simplify(target).value & MASK64, 4, False)
            return
        # Float registers hold bit patterns, so this is an unsigned 32-bit read.
        regs[fpr(ins.rt)] = _read_through(memory, target, 4, False, epoch, len(stores))
        return

    if m in _STORE_SIZES:
        size = _STORE_SIZES[m]
        target = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        if _is_frame_address(target) and _symbolic_frame(frame, target, size):
            # A spill. A 128-bit `sq` of a callee-saved register is reloaded by a
            # matching `lq`; the slot keeps both halves as written, so the pair
            # cancels, and a read of either half finds its value.
            slot = simplify(target).value & MASK64
            if size == 16:
                high = Const(0) if ins.rt == 0 else regs.get(upper(ins.rt))
                _frame_spill_quad(frame, slot, get(ins.rt), high)
            else:
                _frame_write(frame, slot, size, get(ins.rt))
            return
        value = get(ins.rt)
        if size == 16 and (ins.rt == 0 or upper(ins.rt) in regs) and not (
                _is_frame_address(target) and isinstance(frame, Frame)
                and id(value) in frame.seeds):
            # A quadword the function built: its two halves, at the address the
            # hardware aligns down to.
            aligned = address_of(Binary("and", target, Const(~0xF & MASK64)))
            high = Const(0) if ins.rt == 0 else regs[upper(ins.rt)]
            for address, half in ((aligned, value),
                                  (address_of(Binary("add", aligned, Const(8))), high)):
                stores.append(Store(address, half, 8))
                _record_store(memory, address, 8, half)
            return
        if size == 16:
            if (_is_frame_address(target) and isinstance(frame, Frame)
                    and id(value) in frame.seeds):
                # A saved register spilled after the frame escaped: still bookkeeping.
                _frame_spill_quad(frame, simplify(target).value & MASK64, value,
                                  regs.get(upper(ins.rt)))
                return
            for store in _as_stores(target, value, size):
                stores.append(store)
                _record_store(memory, store.address, store.size, store.value)
            return
        stores.append(Store(target, value, size))
        _record_store(memory, target, size, value)
        return

    if m in _LOAD_SIZES:
        size, signed_load = _LOAD_SIZES[m]
        target = address_of(Binary("add", get(ins.rs), Const(ins.imm & MASK64)))
        if _is_frame_address(target) and _symbolic_frame(frame, target, size):
            slot = simplify(target).value & MASK64
            if m == "lq":
                # Both halves, as the slot holds them; an upper half not known
                # when it was spilled is unknown again.
                regs[ins.rt] = _frame_read(frame, slot, 8, False)
                try:
                    regs[upper(ins.rt)] = _frame_read(frame, slot + 8, 8, False)
                except Unsupported:
                    regs.pop(upper(ins.rt), None)
                return
            regs[ins.rt] = _frame_read(frame, slot, size, signed_load)
            return
        if m == "lq" and memory:
            aligned = address_of(Binary("and", target, Const(~0xF & MASK64)))
            if ins.rt:
                regs[ins.rt] = _read_through(memory, aligned, 8, False, epoch, len(stores))
                regs[upper(ins.rt)] = _read_through(
                    memory, address_of(Binary("add", aligned, Const(8))), 8, False, epoch,
                    len(stores))
            return
        if memory:
            regs[ins.rt] = _read_through(memory, target, size, signed_load, epoch,
                                         len(stores))
            return

    if ins.is_branch or ins.is_jump or ins.is_call:
        raise Unsupported("branch in a delay slot")
    _apply(regs, ins, get, epoch)


# Registers a call may clobber. After inlining a callee these are dropped rather
# than left holding stale values: if the caller then reads one, the derivation is
# refused instead of quietly using a value the real code would not have had.
CALLER_SAVED = tuple(range(2, 4)) + tuple(range(4, 8)) + tuple(range(8, 16)) + (24, 25)


def _call_arguments(regs: dict) -> dict:
    """The expressions in the argument registers, as a substitution map for a callee:
    $a0..$a3 and $t0..$t3 for the integer arguments, $f12..$f19 for the float ones."""
    arguments = {index: regs[reg] for index, reg in enumerate(INT_ARG_REGS) if reg in regs}
    for k, reg in enumerate(FLOAT_ARG_REGS):
        if fpr(reg) in regs:
            arguments[FLOAT_ARG_BASE + k] = regs[fpr(reg)]
    return arguments


def _contains_load(expr) -> bool:
    if isinstance(expr, Load):
        return True
    return any(_contains_load(child) for child in children(expr))


def _check_callee_loads_are_safe(callee, arguments, memory: dict, epoch: int = 0) -> None:
    """Refuse an inline whose callee might read what the caller has already written.

    A callee's expressions describe memory as it stood when the callee was entered,
    which is *after* the caller's earlier stores. Substituting them into the caller
    evaluates those loads against entry-state memory instead, so a caller that writes
    a field and then calls something reading that field would be derived wrongly.

    Proving independence needs every address on both sides to be a constant and all
    of them distinct. Anything less is refused - the same conservatism the plain
    load-after-store path uses. The single-path version of inlining omitted this
    check entirely.
    """
    if not memory:
        return
    for path in callee.paths:
        parts = [path.expression] if path.expression is not None else []
        parts.extend(path.conditions)
        for store in path.stores:
            parts.extend((store.address, store.value))
        memo = {}
        for part in parts:
            moved = _in_caller(part, arguments, epoch, memo)
            if not _contains_load(moved):
                continue
            if not _loads_are_disjoint(moved, memory):
                raise Unsupported("callee may read a location the caller wrote")


def _loads_are_disjoint(expr, memory: dict) -> bool:
    """True when every load in `expr` is provably clear of every write in `memory`.

    Uses the same base-plus-offset reasoning as an ordinary load. Forwarding is not
    attempted across a call - a callee load that exactly matches a caller write is
    still refused here - since that would need the callee's loads rewritten rather
    than merely checked.
    """
    if isinstance(expr, Load):
        for (written, written_size) in memory:
            if _relation(written, written_size, expr.address, expr.size) != "disjoint":
                return False
    return all(_loads_are_disjoint(child, memory) for child in children(expr))


def _in_caller(expr, arguments, epoch: int, memo: dict):
    """A callee expression as the caller sees it: arguments substituted, and its
    loads reading the caller's current epoch rather than entry memory."""
    if epoch:
        expr = _retag(expr, epoch, memo)
    return substitute(expr, arguments)


def _settle_for_callee(callee, arguments, memory, stores, marks, frame=None) -> tuple:
    """Make a known callee safe to inline; returns (epoch, marks).

    A callee's loads describe memory as it stood when it was entered, after the
    caller's stores. If one may read something the caller wrote, a barrier first
    applies those stores, and the callee's loads then read the new epoch.

    A callee handed the address of one of the caller's locals reads and writes it
    through memory, so the caller's frame escapes first, exactly as for a call
    that is not inlined. Without this the inlined callee read real stack memory
    while the caller's write sat in the symbolic frame - stale data that a
    no-argument function's near-vacuous test let through.
    """
    if isinstance(frame, Frame) and any(_is_frame_address(value)
                                        for value in arguments.values()):
        frame.escape(stores, memory)
        memory.barrier(len(stores))
        marks = _drain(memory, marks)
    try:
        _check_callee_loads_are_safe(callee, arguments, memory, memory.epoch)
    except Unsupported:
        if not isinstance(memory, Overlay):
            raise
        memory.barrier(len(stores))
        marks = _drain(memory, marks)
    return memory.epoch, marks


def _apply_callee_path(callee_path, arguments, stores: list, memory: dict,
                       epoch: int = 0) -> tuple:
    """Fold one callee path into the caller, returning its conditions."""
    memo = {}
    for store in callee_path.stores:
        moved = Store(
            # The canonical address form, as for the caller's own stores: the
            # argument substituted in may be a sign-extended word result, and left
            # in place that extension hides the base from everything downstream.
            address_of(_in_caller(store.address, arguments, epoch, memo)),
            simplify(_in_caller(store.value, arguments, epoch, memo)),
            store.size,
        )
        stores.append(moved)
        _record_store(memory, moved.address, moved.size, moved.value)
    return tuple(
        simplify(_in_caller(condition, arguments, epoch, memo))
        for condition in callee_path.conditions
    )


class FloatResult(Unsupported):
    """The function leaves $v0 as a callee left it and computes $f0, and no
    caller says which it returns: derive_paths tries it as returning $f0."""


def derive_paths(image: bytes, base: int, start: int, end: int, *args, **kwargs) -> "PathDerivation":
    """_derive_paths, and again as returning a float when that is the one
    question left (FloatResult): a function whose $v0 is only a callee's
    leftover while its own code computes $f0, which no caller reads $v0 after
    (calls.int_returning) - typically a virtual method, which has no direct
    callers to say. Its test then compares $f0, as for any float-returning
    function, and a caller that did read $v0 would fail its own."""
    try:
        return _derive_paths(image, base, start, end, *args, **kwargs)
    except FloatResult:
        return _derive_paths(image, base, start, end, *args, force_float=True, **kwargs)


def _derive_paths(image: bytes, base: int, start: int, end: int,
                 gp_value: int = None, known: dict = None,
                 parameters: dict = None, loops: bool = False,
                 merge: bool = False, force_float: bool = False) -> PathDerivation:
    """Derive a function as guarded paths, handling acyclic control flow.

    `known` maps an address to an already-derived `PathDerivation`. A call to one of
    those is inlined - its result substituted with the caller's arguments - which is
    what lets derivation progress up the call graph instead of stopping at the first
    `jal`. Only single-path callees are inlined, so a conditional callee cannot
    multiply the caller's path count.
    """
    known = known or {}
    if gp_value is None:
        gp_value = layout.CURRENT.gp
    # Whether this function returns a float: its callers read $f0 after calling it
    # (calls.float_returning).
    returns_float = force_float or start in getattr(parameters, "returns_float", ())

    def word_at(addr: int) -> int:
        if not (base <= addr < base + len(image) - 3):
            raise Unsupported("address outside the image")
        return struct.unpack_from("<I", image, addr - base)[0]

    paths = []
    # Shared by every fork of this function's walk, so the cap is on total work.
    budget = [MAX_TOTAL_STEPS]
    epochs = [0]
    loop_ids = [0]
    loop_info = _loop_info(image, base, start, end) if loops else {}
    joins = _join_points(image, base, start, end) if merge else {}

    def shared_code(address: int) -> bool:
        """True for code outside this function that the walk may simply continue into.

        Compilers share tails between functions: a short function often ends by
        jumping into the middle of another one, to a sequence they both finish
        with. That is not a call - nothing returns to the jump - so the walk carries
        on there as though the code were this function's own. A jump to a known
        function *start* is different, a tail call, and stays one. Without the list
        of function starts (no `parameters`), nothing outside counts as shared.
        """
        return (parameters is not None and address not in parameters
                and address not in known and base <= address < base + len(image) - 3)

    def opaque_call(target: int, regs: dict, stores: list, marks: tuple, epoch: int,
                    frame=None, memory=None, pointer=None):
        """Record a call to an underived function; returns (marks, new epoch).

        Only the arguments the callee reads are passed, and each must be known. A
        stack address cannot be passed: the callee could write through it into this
        function's frame, which the frame model would not see.
        """
        if pointer is not None:
            # Which function a pointer reaches is not known, so neither is what it
            # reads. Every argument register still holds *something* - this
            # function's own incoming arguments, if nothing else - so passing all
            # of them would give every function with a virtual call sixteen
            # parameters. Instead pass the integer arguments up to the highest one
            # this function set (and always $a0, the object of a method call), and
            # the float arguments it set. A callee that reads more disagrees in
            # testing and the function is dropped.
            def changed(i):
                reg = _arg_register(i)
                return reg in regs and regs[reg] != Arg(i)
            highest = max([i for i in range(FLOAT_ARG_BASE) if changed(i)] + [0])
            indices = [i for i in range(highest + 1) if _arg_register(i) in regs]
            indices += [i for i in range(FLOAT_ARG_BASE, ARG_COUNT) if changed(i)]
        elif target not in parameters:
            raise Unsupported("calls an address that is not a known function start")
        else:
            indices = parameters[target]
        arguments = []
        for index in indices:
            reg = _arg_register(index)
            if reg not in regs:
                # The register holds whatever an earlier callee left there. A callee
                # that really reads it gets 0 here and garbage in the original, so
                # the test disagrees and the function is dropped; one that only
                # might read it (the analysis is conservative) is unaffected.
                arguments.append((index, Const(0)))
                continue
            if _is_frame_address(regs[reg]):
                # An out-parameter: the callee may read or write the frame, so from
                # here on it is real memory.
                if not isinstance(frame, Frame):
                    raise Unsupported("passes a stack address to an underived function")
                frame.escape(stores, memory)
            arguments.append((index, regs[reg]))
        index = memory.next_epoch() if isinstance(memory, Overlay) else epoch + 1
        call = Call(index, target, tuple(arguments), pointer)
        before_v0 = regs.get(2)
        for index in CALLER_SAVED + (REG_HI, REG_LO, FP_COND):
            regs.pop(index, None)
        for index in CALLER_SAVED:
            regs.pop(upper(index), None)
        for index in VU_KEYS:
            regs.pop(index, None)
        for index in range(20):                 # $f0..$f19 are caller-saved
            regs.pop(fpr(index), None)
        if pointer is None and target in getattr(parameters, "preserves_v0", ()):
            # The callee never touches $v0: it still holds the caller's value.
            if before_v0 is not None:
                regs[2] = before_v0
        else:
            regs[2] = CallResult(call.index, target)
        # A float result comes back in $f0. Whatever the callee left there is
        # what the original reads, so it is a result like $v0 - one the emitted
        # code can only name for a callee called by address (derive._call).
        regs[CALL_FLOAT] = CallResult(call.index, target, fpr(0))
        return marks + ((len(stores), "call", call),), call.index

    def enter_loop(head, regs, stores, frame, memory, conditions, visited, marks,
                   outer, sink):
        """Derive the loop at `head` once over variables, then continue after it."""
        body, carried, moves_sp = loop_info[head]
        if moves_sp:
            raise Unsupported("a loop moves the stack pointer")
        loop_ids[0] += 1
        identity = loop_ids[0]
        head_epoch = memory.next_epoch()
        init = tuple((register, regs[register]) for register in carried
                     if register in regs)
        body_regs = dict(regs)
        for register in carried:
            body_regs[register] = Var(identity, register)
        context = LoopContext(identity, head, body, carried, [])
        body_paths = []
        body_frame = frame.fork()
        body_frame.frozen = True
        walk(head, body_regs, [], body_frame, Overlay(epoch=head_epoch, counter=epochs),
             (), {}, (), loop=context, sink=body_paths)
        tree = _build_tree([(path.events(), path.end()) for path in body_paths])
        carried, tree = _prune_loop_variables(identity, carried, tree, context.exits)
        init = tuple((register, value) for register, value in init if register in carried)
        after_epoch = memory.next_epoch()
        node = Loop(identity, head, init, carried, tree, head_epoch, after_epoch,
                    len(context.exits))
        loop_mark = (len(stores), "loop", node)
        if not context.exits:
            sink.append(Path(conditions, None, tuple(stores), marks + (loop_mark,),
                             leaf=("unreachable",)))
            return
        count = len(context.exits)
        for index, (exit_address, exit_regs) in enumerate(context.exits):
            guards = [LoopExit(identity, j).negate() for j in range(index)]
            if index < count - 1:
                guards.append(LoopExit(identity, index))
            walk(exit_address, dict(exit_regs), list(stores), frame.fork(),
                 Overlay(epoch=after_epoch, counter=epochs),
                 conditions + tuple(guards), visited,
                 marks + (loop_mark,) + tuple((len(stores), "cond", g) for g in guards),
                 loop=outer, sink=sink)

    def merge_branch(addr, ins, predicate, slot, regs, stores, frame, memory,
                     conditions, visited, marks, loop):
        """Walk both sides of a branch to where they meet; merged registers or None.

        When neither side stores, calls or touches the frame - the branch only
        decides what some registers hold - the two routes need not stay apart:
        each differing register becomes `predicate ? taken : untaken` and the walk
        carries on once. Enumerating them instead doubles the paths at every such
        branch, which is what refused functions for having too many.
        """
        join = joins.get(addr)
        if join is None:
            return None
        region = []
        sides = []
        for taken in (True, False):
            side_regs = dict(regs)
            side_stores = list(stores)
            side_frame, side_memory = frame.fork(), memory.fork()
            if taken or not ins.is_likely:
                _step(side_regs, side_stores, side_frame, side_memory, slot,
                      _getter(side_regs), memory.epoch)
            sides.append((side_regs, side_stores, side_frame, side_memory))
            try:
                walk(ins.target if taken else addr + 8, side_regs, side_stores,
                     side_frame, side_memory, conditions, visited, marks,
                     loop=loop, sink=region, stop=join)
            except Unsupported:
                return None
        if len(region) != 2 or not all(isinstance(r, _JoinState) for r in region):
            return None
        for state in region:
            if (len(state.stores) != len(stores) or state.marks != marks
                    or state.memory.epoch != memory.epoch or len(state.memory) != len(memory)
                    or dict(state.frame) != dict(frame)):
                return None
        taken_regs, untaken_regs = region[0].regs, region[1].regs
        merged = {}
        for register in set(taken_regs) & set(untaken_regs):
            a, b = taken_regs[register], untaken_regs[register]
            merged[register] = a if (a is b or a == b) else Select(predicate, a, b)
        return join, merged

    def walk(addr: int, regs: dict, stores: list, frame: dict, memory: dict,
             conditions: tuple, visited: dict, marks: tuple = (), epoch: int = 0,
             loop=None, sink=None, stop=None):
        sink = paths if sink is None else sink
        if len(sink) > MAX_PATHS:
            raise Unsupported("too many paths")
        # Each walk owns its counts; a fork must not see its sibling's iterations.
        visited = dict(visited)

        while True:
            if stop is not None and addr == stop:
                # The end of a branch's region: hand the state back for merging.
                sink.append(_JoinState(regs, stores, frame, memory, marks))
                return
            # Loops are unrolled rather than refused. A loop whose trip count is
            # constant needs no special handling: its exit condition folds to a
            # constant on every iteration (see _constant_truth), so the walk follows
            # the single feasible side round the loop and out. A loop whose count
            # depends on input instead forks on every iteration and runs into
            # MAX_PATHS or MAX_UNROLL - refused, which is the honest outcome, since
            # no finite expression describes it.
            count = visited.get(addr, 0) + 1
            if count > MAX_UNROLL:
                raise Unsupported("loop")
            steps = visited.get(STEPS, 0) + 1
            if steps > MAX_PATH_STEPS:
                raise Unsupported("path too long")
            budget[0] -= 1
            if budget[0] < 0:
                raise Unsupported("exploration budget exhausted")
            # Sizes are recorded at construction, so this costs one attribute read
            # per live register.
            for value in regs.values():
                if value.tree_size > MAX_EXPR_SIZE:
                    raise Unsupported("expression too large")
            epoch = memory.epoch
            if loop is not None:
                if addr == loop.head and visited.get(addr):
                    # Back at the head: the next iteration starts from these values.
                    sink.append(Path(conditions, None, tuple(stores), marks, leaf=(
                        "continue", loop.id,
                        tuple((r, regs.get(r, Const(0))) for r in loop.carried))))
                    return
                if addr not in loop.body:
                    loop.exits.append((addr, dict(regs)))
                    sink.append(Path(conditions, None, tuple(stores), marks,
                                     leaf=("break", loop.id, len(loop.exits) - 1)))
                    return
            if addr in loop_info and not (loop is not None and loop.head == addr):
                enter_loop(addr, regs, stores, frame, memory, conditions, visited,
                           marks, loop, sink)
                return
            if not (start <= addr < end) and not shared_code(addr):
                raise Unsupported("control flow leaves the function")
            visited[addr] = count
            visited[STEPS] = steps
            get = _getter(regs)
            ins = DISASM.decode(word_at(addr), addr)
            m = ins.mnemonic

            if m in ("nop", "sync"):
                addr += 4
                continue

            if m == "break":
                # A trap: the original raises an exception here and never
                # comes back. The path ends with it.
                marks = _drain(memory, marks)
                sink.append(Path(conditions, None, tuple(stores), marks, leaf=("trap", addr)))
                return

            if m == "jr":
                if not ins.is_return:
                    raise Unsupported("indirect jump")
                _step(regs, stores, frame, memory,
                      DISASM.decode(word_at(addr + 4), addr + 4), get, epoch)
                marks = _drain(memory, marks)
                # A function that never writes $v0 returns nothing. Insisting on a
                # return value refused 514 functions outright - every setter and
                # initialiser in the binary - because they are simply void.
                if returns_float:
                    # Its callers read $f0: that is the result, the call's own
                    # float result if the function did not compute one.
                    result = _getter(regs)(fpr(0))
                    sink.append(Path(conditions, result, tuple(stores), marks))
                    return
                try:
                    result = _getter(regs)(2)
                except Unsupported:
                    result = None
                # $v0 only as a callee left it, while this function's own code
                # computed $f0: a function returning a float. Derived as returning
                # $v0 it would pass a test that compares $v0 - and say nothing
                # about what it really returns.
                if (isinstance(result, CallResult) and result.register == 2
                        and regs.get(fpr(0)) is not None):
                    if start not in getattr(parameters, "returns_int", ()):
                        raise FloatResult("returns a float in $f0")
                    raise Unsupported("returns a float in $f0")
                sink.append(Path(conditions, result, tuple(stores), marks))
                return

            if ins.is_branch:
                predicate = _condition_for(ins, get)
                slot = DISASM.decode(word_at(addr + 4), addr + 4)

                # When the condition folds to a constant, only one side can ever
                # run. Exploring both anyway manufactures an infeasible path, and
                # `select` picks the first path whose conditions hold - so a path
                # that can never be taken can still be chosen, which is how a
                # function that stores a flag and immediately branches on it came
                # out wrong. Folding also keeps `if (1 != 0)` out of the output.
                # A `break` in the delay slot: the compiler's division-by-zero
                # guard (`beql $divisor, $zero, next; break`). Where the slot
                # runs, the path ends in the trap.
                slot_traps = slot.mnemonic == "break"
                decided = _constant_truth(predicate)
                if decided is True:
                    if slot_traps:
                        sink.append(Path(conditions, None, tuple(stores), _drain(memory, marks),
                                         leaf=("trap", addr + 4)))
                        return
                    _step(regs, stores, frame, memory, slot, get, epoch)
                    marks = _drain(memory, marks)
                    addr = ins.target
                    continue
                if decided is False:
                    if not ins.is_likely:
                        if slot_traps:
                            sink.append(Path(conditions, None, tuple(stores), _drain(memory, marks),
                                             leaf=("trap", addr + 4)))
                            return
                        _step(regs, stores, frame, memory, slot, get, epoch)
                    marks = _drain(memory, marks)
                    addr += 8
                    continue

                if joins and not slot_traps:
                    merged = merge_branch(addr, ins, predicate, slot, regs, stores, frame,
                                          memory, conditions, visited, marks, loop)
                    if merged is not None:
                        addr, regs = merged
                        continue

                # Taken: the delay slot runs, then control moves to the target.
                taken_regs, taken_stores = dict(regs), list(stores)
                taken_frame, taken_memory = frame.fork(), memory.fork()
                if slot_traps:
                    sink.append(Path(conditions + (predicate,), None, tuple(stores),
                                     _drain(taken_memory, marks + ((len(stores), "cond", predicate),)),
                                     leaf=("trap", addr + 4)))
                else:
                    _step(taken_regs, taken_stores, taken_frame, taken_memory,
                          slot, _getter(taken_regs), epoch)
                    walk(ins.target, taken_regs, taken_stores, taken_frame, taken_memory,
                         conditions + (predicate,), visited,
                         _drain(taken_memory, marks + ((len(stores), "cond", predicate),)),
                         loop=loop, sink=sink, stop=stop)

                # Not taken: a "likely" branch nullifies its delay slot.
                fall_regs, fall_stores = dict(regs), list(stores)
                fall_frame, fall_memory = frame.fork(), memory.fork()
                negated = predicate.negate()
                if not ins.is_likely:
                    if slot_traps:
                        sink.append(Path(conditions + (negated,), None, tuple(stores),
                                         _drain(fall_memory, marks + ((len(stores), "cond", negated),)),
                                         leaf=("trap", addr + 4)))
                        return
                    _step(fall_regs, fall_stores, fall_frame, fall_memory,
                          slot, _getter(fall_regs), epoch)
                walk(addr + 8, fall_regs, fall_stores, fall_frame, fall_memory,
                     conditions + (negated,), visited,
                     _drain(fall_memory, marks + ((len(stores), "cond", negated),)),
                     loop=loop, sink=sink, stop=stop)
                return

            if m in ("b", "j"):
                _step(regs, stores, frame, memory,
                      DISASM.decode(word_at(addr + 4), addr + 4), get, epoch)
                marks = _drain(memory, marks)
                epoch = memory.epoch
                if start <= ins.target < end or shared_code(ins.target):
                    addr = ins.target
                    continue
                # A jump out of the function is a tail call: the callee's result
                # becomes this function's result, with no return of our own. These
                # are common in thunks and were previously all refused as "control
                # flow leaves the function".
                if ins.target not in known:
                    if parameters is None:
                        raise Unsupported(
                            "tail-calls an underived function: 0x%08X" % ins.target)
                    # A tail call to an underived function: call it, and its $v0
                    # is this function's result.
                    marks, _epoch = opaque_call(ins.target, regs, stores, marks, epoch,
                                                frame, memory)
                    result = regs[CALL_FLOAT] if returns_float else regs[2]
                    sink.append(Path(conditions, result, tuple(stores), marks))
                    return
                callee = known[ins.target]
                if callee.returns_float != returns_float and not callee.is_void:
                    raise Unsupported("tail-calls a function returning the other kind")
                arguments = _call_arguments(regs)
                epoch, marks = _settle_for_callee(callee, arguments, memory, stores, marks,
                                                  frame)

                # Each route through the callee becomes a route through this
                # function, since a tail call has no return of its own: the callee's
                # result and conditions are simply adopted.
                for callee_path in callee.paths:
                    branch_stores = list(stores)
                    branch_memory = memory.fork()
                    before = len(stores)
                    extra = _apply_callee_path(
                        callee_path, arguments, branch_stores, branch_memory, epoch
                    )
                    result = (
                        None if callee_path.expression is None
                        else simplify(_in_caller(callee_path.expression, arguments,
                                                 epoch, {}))
                    )
                    sink.append(Path(
                        conditions + extra, result, tuple(branch_stores),
                        marks + tuple((before, "cond", c) for c in extra),
                    ))
                return

            if ins.is_call:
                if ins.target < 0:
                    if parameters is None or ins.mnemonic != "jalr":
                        raise Unsupported("calls through a register")
                    # The target is read before the delay slot runs.
                    pointer = get(ins.rs)
                    _step(regs, stores, frame, memory,
                          DISASM.decode(word_at(addr + 4), addr + 4), get, epoch)
                    marks = _drain(memory, marks)
                    marks, epoch = opaque_call(-1, regs, stores, marks, memory.epoch,
                                               frame, memory, pointer)
                    memory = Overlay(epoch=epoch, counter=epochs)
                    addr += 8
                    continue
                if ins.target not in known:
                    if parameters is None:
                        raise Unsupported(
                            "calls an underived function: 0x%08X" % ins.target)
                    _step(regs, stores, frame, memory,
                          DISASM.decode(word_at(addr + 4), addr + 4), get, epoch)
                    marks = _drain(memory, marks)
                    marks, epoch = opaque_call(ins.target, regs, stores, marks,
                                               memory.epoch, frame, memory)
                    # The callee may have written anything, so nothing this function
                    # wrote earlier can be forwarded to a later load any more.
                    memory = Overlay(epoch=epoch, counter=epochs)
                    addr += 8
                    continue
                callee = known[ins.target]

                # The delay slot runs before the call, and often sets the last
                # argument, so apply it first.
                _step(regs, stores, frame, memory,
                      DISASM.decode(word_at(addr + 4), addr + 4), get, epoch)
                marks = _drain(memory, marks)
                arguments = _call_arguments(regs)
                epoch, marks = _settle_for_callee(callee, arguments, memory, stores, marks,
                                                  frame)

                # A callee with several paths multiplies the caller's: one caller
                # path continues for each route through the callee, carrying that
                # route's conditions. Restricting this to single-path callees is why
                # an earlier version of inlining gained almost nothing.
                for callee_path in callee.paths:
                    branch_regs = dict(regs)
                    branch_stores = list(stores)
                    branch_frame = frame.fork()
                    branch_memory = memory.fork()
                    before = len(stores)
                    extra = _apply_callee_path(
                        callee_path, arguments, branch_stores, branch_memory, epoch
                    )
                    for index in CALLER_SAVED:
                        branch_regs.pop(index, None)
                        branch_regs.pop(upper(index), None)
                    for index in VU_KEYS:
                        branch_regs.pop(index, None)
                    if callee_path.expression is not None:
                        # A float result comes back in $f0, anything else in $v0.
                        branch_regs[fpr(0) if callee.returns_float else 2] = simplify(
                            _in_caller(callee_path.expression, arguments, epoch, {}))
                    walk(addr + 8, branch_regs, branch_stores, branch_frame,
                         branch_memory, conditions + extra, visited,
                         marks + tuple((before, "cond", c) for c in extra),
                         loop=loop, sink=sink, stop=stop)
                return

            if ins.is_jump:
                raise Unsupported("unhandled jump")

            _step(regs, stores, frame, memory, ins, get, epoch)
            marks = _drain(memory, marks)
            addr += 4

    # Seed the registers to exactly the state the interpreter starts a call in, so
    # a derivation and the differential test it is checked against begin from the
    # same place.
    #
    # The callee-saved registers need values for a mundane reason: the prologue
    # *reads* them in order to spill them. Without a value, every function with a
    # frame was refused for "reads uninitialised register $16" or "$31". Seeding
    # them lets the spill and its matching reload cancel out.
    #
    # The limitation to be honest about: a function that genuinely took a value in a
    # callee-saved register would be derived as though that register were zero, and
    # the differential test would still pass, because the interpreter also starts it
    # at zero. The ABI says those registers are not parameters, so this does not
    # arise in compiled code - but it is an assumption, not something proven here.
    from .emu import RETURN_SENTINEL

    regs = {0: Const(0), 28: Const(gp_value), 29: Const(layout.CURRENT.stack_top)}
    for index in list(range(16, 24)) + [30]:   # $s0..$s7 and $fp
        regs[index] = Const(0)
    regs[31] = Const(RETURN_SENTINEL)          # $ra
    # $f20..$f31 are callee-saved, so they are spilled and reloaded for exactly the
    # same reason the integer saved registers are, and were blocking 123 functions
    # for exactly the same reason: the prologue reads them in order to save them.
    for index in range(20, 32):
        regs[fpr(index)] = Const(0)
    # Eight integer and eight float arguments. $t0..$t3 and $f12..$f19 used to be
    # left unseeded, and reading one refused 876 functions - most of them simply
    # taking more than four arguments, or a float.
    for i, reg in enumerate(INT_ARG_REGS):
        regs[reg] = Arg(i)
    for k, reg in enumerate(FLOAT_ARG_REGS):
        regs[fpr(reg)] = Arg(FLOAT_ARG_BASE + k)
    seeds = frozenset(id(regs[index]) for index in
                      list(range(16, 24)) + [30, 31] + [fpr(i) for i in range(20, 32)])
    walk(start, regs, [], Frame(seeds=seeds), Overlay(counter=epochs), (), {})

    if not paths:
        raise Unsupported("no return reached")
    if all(path.leaf is not None and path.leaf[0] == "trap" for path in paths):
        raise Unsupported("every path traps")
    # A tail call to an underived function hands on whatever the callee left in
    # $v0. In a function whose other paths return nothing, that is no value either.
    if any(path.expression is None for path in paths):
        for index, path in enumerate(paths):
            if isinstance(path.expression, CallResult) and path.calls and \
                    path.expression.index == path.calls[-1].index:
                paths[index] = Path(path.conditions, None, path.stores, path.marks)
    returned = list(_returned_values(paths))
    void_paths = sum(1 for value in returned if value is None)
    if void_paths and void_paths != len(returned):
        # Some paths produce a value and others do not. That is either a function
        # whose result is only meaningful on some routes, or a misread; either way
        # the signature is ambiguous, so refuse rather than choose.
        raise Unsupported("inconsistent void and value-returning paths")
    derivation = PathDerivation(start, tuple(paths), returns_float)
    if not derivation.sequential and _touches_frame_memory(derivation):
        # The frame is symbolic unless it escaped, and escaping always makes the
        # derivation sequential. A frame address reaching memory any other way is
        # a route this model does not account for, so refuse it rather than emit
        # code that reads the stack at a fixed address.
        raise Unsupported("accesses the stack frame through memory")
    return derivation


def _prune_loop_variables(identity, carried, tree, exits) -> tuple:
    """Drop loop variables that nothing reads; returns (carried, tree).

    Every register the body writes starts as a variable, but one that is always
    written before it is read - a temporary - needs no value carried between
    iterations. Keeping it would also give it an initial value from before the
    loop: `$t0` started as argument 4, which made the function appear to take a
    fifth argument it never reads. A variable is kept if any expression of the
    body other than its own update reads it, or any exit's registers do.
    """
    def mentions(expr, wanted):
        found, seen, stack = set(), set(), [expr]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, Var) and node.loop == identity and node.register in wanted:
                found.add(node.register)
            stack.extend(children(node))
        return found

    def readers(statements, wanted):
        found = set()
        for statement in statements:
            kind = statement[0]
            if kind == "continue" and statement[1] == identity:
                for register, expr in statement[2]:
                    found |= mentions(expr, wanted - {register})
            elif kind == "if":
                found |= mentions(statement[1], wanted)
                found |= readers(statement[2], wanted) | readers(statement[3], wanted)
            elif kind != "continue":
                for expr in tree_expressions([statement]):
                    found |= mentions(expr, wanted)
        return found

    def rewrite(statements, keep):
        out = []
        for statement in statements:
            if statement[0] == "continue" and statement[1] == identity:
                out.append(("continue", identity,
                            tuple((r, e) for r, e in statement[2] if r in keep)))
            elif statement[0] == "if":
                out.append(("if", statement[1], rewrite(statement[2], keep),
                            rewrite(statement[3], keep)))
            else:
                out.append(statement)
        return out

    keep = set(carried)
    while True:
        used = readers(tree, keep)
        for _address, regs in exits:
            for register in keep:
                if register in regs:
                    used |= mentions(regs[register], keep)
            for value in regs.values():
                used |= mentions(value, keep)
        if used >= keep:
            break
        keep &= used
    kept = tuple(register for register in carried if register in keep)
    return kept, rewrite(tree, set(kept))


@dataclasses.dataclass
class _JoinState:
    """A walk's state on reaching the join point of the branch being merged."""

    regs: dict
    stores: list
    frame: object
    memory: object
    marks: tuple


def _join_points(image, base, start, end) -> dict:
    """{branch address: the address where its two routes meet again}.

    That is the branch's immediate post-dominator in the function's control-flow
    graph - the first instruction every route from the branch must pass through.
    A delay slot belongs to both routes, so for an ordinary branch the routes part
    after the slot; a "likely" branch parts at the branch itself.
    """
    EXIT = -1

    def decode(address):
        return DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)

    successors, slots, branches = {}, set(), {}
    for address in range(start, end, 4):
        if address in slots:
            continue
        ins = decode(address)
        if not ins.has_delay_slot or address + 4 >= end:
            successors[address] = [address + 4 if address + 4 < end else EXIT]
            continue
        slot, after, target = address + 4, address + 8, ins.target
        slots.add(slot)
        inside = start <= target < end
        if ins.is_call:
            successors[address], successors[slot] = [slot], [after]
        elif ins.is_return or ins.mnemonic in ("jr", "j", "b") or (ins.is_jump and not ins.is_branch):
            successors[address] = [slot]
            successors[slot] = [target] if (ins.mnemonic in ("j", "b") and inside) else [EXIT]
        elif ins.is_branch:
            to_target = [target] if inside else [EXIT]
            if ins.is_likely:
                successors[address] = [slot, after]
                successors[slot] = to_target
                branches[address] = address
            else:
                successors[address] = [slot]
                successors[slot] = to_target + [after]
                branches[address] = slot
        else:
            successors[address], successors[slot] = [slot], [after]
    for node, targets in successors.items():
        successors[node] = [t if (t == EXIT or start <= t < end) else EXIT for t in targets]
    nodes = sorted(successors) + [EXIT]
    index = {node: i for i, node in enumerate(nodes)}
    full = (1 << len(nodes)) - 1
    pdom = {node: full for node in nodes}
    pdom[EXIT] = 1 << index[EXIT]
    changed = True
    while changed:
        changed = False
        for node in reversed(nodes[:-1]):
            meet = full
            for successor in successors[node]:
                meet &= pdom.get(successor, 1 << index[EXIT])
            value = meet | (1 << index[node])
            if value != pdom[node]:
                pdom[node] = value
                changed = True
    joins = {}
    for branch, parting in branches.items():
        strict = pdom[parting] & ~(1 << index[parting])
        for candidate in nodes:
            if strict >> index[candidate] & 1 and pdom[candidate] == strict:
                if candidate != EXIT:
                    joins[branch] = candidate
                break
    return joins


def _returned_values(paths):
    """Every value the function can return, from inside loop bodies too."""
    for path in paths:
        if path.leaf is None:
            yield path.expression
        for _count, kind, obj in path.marks:
            if kind == "loop":
                yield from _tree_returns(obj.body)


def _tree_returns(statements):
    for statement in statements:
        if statement[0] == "return":
            yield statement[1]
        elif statement[0] == "if":
            yield from _tree_returns(statement[2])
            yield from _tree_returns(statement[3])
        elif statement[0] == "loop":
            yield from _tree_returns(statement[1].body)


def tree_expressions(statements):
    """Every expression in a statement tree, loop bodies included."""
    for statement in statements:
        kind = statement[0]
        if kind == "store":
            yield statement[1].address
            yield statement[1].value
        elif kind == "call":
            for _index, expr in statement[1].arguments:
                yield expr
            if statement[1].pointer is not None:
                yield statement[1].pointer
        elif kind == "if":
            yield statement[1]
            yield from tree_expressions(statement[2])
            yield from tree_expressions(statement[3])
        elif kind == "loop":
            for _register, expr in statement[1].init:
                yield expr
            yield from tree_expressions(statement[1].body)
        elif kind == "continue":
            for _register, expr in statement[2]:
                yield expr
        elif kind == "return" and statement[1] is not None:
            yield statement[1]


def _loop_info(image, base, start, end) -> dict:
    """{head: (body addresses, carried registers, moves sp)} for each loop.

    A loop is found by its back edge - a branch or jump to an address at or
    before itself - and taken to span from the head to the last back edge's
    delay slot, which is how compiled loops are laid out. The carried registers
    are every register the body can write; a body containing a call also
    clobbers everything a call does.
    """
    from .calls import _float_operands
    from .structs import _written_register

    def decode(address):
        return DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)

    last = {}
    for address in range(start, end, 4):
        ins = decode(address)
        if (ins.is_branch or (ins.is_jump and not ins.is_call)) \
                and start <= ins.target <= address:
            last[ins.target] = max(last.get(ins.target, 0), address)
    info = {}
    for head, back in last.items():
        body = frozenset(range(head, min(back + 8, end), 4))
        carried, moves_sp, calls = set(), False, False
        uses_vu = False
        for address in body:
            ins = decode(address)
            written = _written_register(ins)
            if written:
                carried.add(written)
                moves_sp |= written == 29
            _read, wrote = _float_operands(ins)
            carried |= {fpr(r) for r in wrote}
            if ins.raw >> 26 in (0x12, 0x36, 0x3E):
                uses_vu = True
            carried |= vu_written(ins)
            field = QUAD_WRITERS.get(ins.mnemonic)
            if field is not None and getattr(ins, field):
                carried.add(upper(getattr(ins, field)))
            if ins.mnemonic.startswith(("mult", "div", "madd", "mthi", "mtlo")):
                carried |= {REG_HI, REG_LO}
            if ins.mnemonic.startswith("c."):
                carried.add(FP_COND)
            calls |= ins.is_call
        if calls and uses_vu:
            # A call clobbers all VU0 state, and carrying every lane through the
            # loop as a variable is not worth it for what is rare.
            raise Unsupported("a loop calls and uses VU0")
        if calls:
            carried |= set(CALLER_SAVED) | {fpr(i) for i in range(20)}
            carried |= {upper(i) for i in CALLER_SAVED}
            carried |= {REG_HI, REG_LO, FP_COND, 31}
        carried.discard(0)
        info[head] = (body, tuple(sorted(carried)), moves_sp)
    return info


def _touches_frame_memory(derivation) -> bool:
    roots = []
    for path in derivation.paths:
        roots += list(path.conditions)
        if path.expression is not None:
            roots.append(path.expression)
        for store in path.stores:
            if _is_frame_address(store.address):
                return True
            roots += [store.address, store.value]
    seen, stack = set(), roots
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Load) and _is_frame_address(node.address):
            return True
        stack.extend(children(node))
    return False
