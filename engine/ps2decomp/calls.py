"""Which arguments each function reads, for calling it without deriving it.

A derived function that calls one that has not been derived passes it the values
in the argument registers. Only the ones the callee actually reads are part of
the call: a register that merely still holds something is not an argument, and
passing it would make every call site's signature different.

Integer arguments come from structs.argument_use. Float arguments ($f12..$f19)
get the same treatment here: a function reads one if an instruction reads the
register before anything writes it, or passes it on to a callee that reads it.
"""

from __future__ import annotations

import collections
import re
import struct

from .r5900 import DISASM
from .structs import argument_use
from .symexec import FLOAT_ARG_BASE, FLOAT_ARG_REGS

_FLOAT_REGISTER = re.compile(r"\$f(\d+)")


def _float_operands(ins):
    """(read, written) float registers of one instruction."""
    m = ins.mnemonic
    names = [int(n) for op in ins.operands for n in _FLOAT_REGISTER.findall(str(op))]
    if not names:
        return set(), set()
    if m in ("swc1", "mfc1") or m.startswith("c."):
        return set(names), set()
    if m in ("lwc1", "mtc1", "ctc1"):
        return set(), set(names)
    if m.startswith(("adda.", "suba.", "mula.", "madda.", "msuba.")):
        return set(names), set()          # these write the accumulator, not an FPR
    # Arithmetic, moves and conversions: the first operand is the destination.
    return set(names[1:]), {names[0]}


def float_argument_use(database, image, base) -> set:
    """(function, vector index) for every float argument a function reads."""
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    used = set()
    passes = collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        original = {reg: FLOAT_ARG_BASE + k for k, reg in enumerate(FLOAT_ARG_REGS)}
        address = start
        while address < end:
            ins = DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)
            sequence = [ins]
            if ins.has_delay_slot and address + 4 < end:
                sequence.insert(0, DISASM.decode(
                    struct.unpack_from("<I", image, address + 4 - base)[0], address + 4))
            for step in sequence:
                if step.is_call or (step.is_jump and step.mnemonic == "j"):
                    target = step.target
                    if target in functions and not (start <= target < end):
                        for reg, index in original.items():
                            passes[(target, index)].add((start, index))
                    if step.is_call:
                        original = {}          # $f0..$f19 are caller-saved
                    continue
                read, written = _float_operands(step)
                for reg in read:
                    if reg in original:
                        used.add((start, original[reg]))
                for reg in written:
                    original.pop(reg, None)
            address += 8 if len(sequence) == 2 else 4
    work = list(used)
    while work:
        slot = work.pop()
        for caller in passes.get(slot, ()):
            if caller not in used:
                used.add(caller)
                work.append(caller)
    return used


class Parameters(dict):
    """{function: argument-vector indices it reads}, plus `preserves_v0`: the
    functions after a call to which $v0 still holds the caller's value, and
    `returns_float`: the functions whose callers read their result from $f0;
    `returns_int`: those whose callers read $v0 after calling them."""

    preserves_v0 = frozenset()
    returns_float = frozenset()
    returns_int = frozenset()


def parameters(database, image, base) -> dict:
    """{function: argument-vector indices it reads}, for every known function.

    Live-in analysis over each function's control flow, to a fixed point over the
    call graph (see live_parameters).
    """
    found = Parameters(live_parameters(database, image, base))
    found.preserves_v0 = v0_preserving(database, image, base)
    found.returns_float = float_returning(database, image, base)
    found.returns_int = int_returning(database, image, base)
    return found


# How far after a call a read of $f0 still counts as reading its result.
_RESULT_WINDOW = 24


def float_returning(database, image, base) -> frozenset:
    """Functions that return a float: some caller reads $f0 after calling them,
    before anything writes it.

    What a function computes into $f0 says nothing on its own - $f0 is also
    every float computation's scratch register - so the evidence is taken where
    the result is used. The scan after a call is straight-line: it stops at a
    branch, a jump, another call or the window's end, which can only miss
    evidence, never invent it. A function that ends by jumping to another hands
    that one's result to its own callers, so the evidence passes down tail calls.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}

    def decode(pc):
        return DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)

    found, tails = set(), collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        for pc in range(start, end, 4):
            ins = decode(pc)
            if ins.is_jump and ins.mnemonic == "j" and ins.target in functions                     and not (start <= ins.target < end):
                tails[start].add(ins.target)
            if not ins.is_call or ins.target not in functions:
                continue
            for after in range(pc + 8, min(end, pc + 8 + 4 * _RESULT_WINDOW), 4):
                step = decode(after)
                if step.is_branch or step.is_jump or step.is_call:
                    break
                read, written = _float_operands(step)
                if 0 in read:
                    found.add(ins.target)
                    break
                if 0 in written:
                    break
    work = list(found)
    while work:
        for target in tails.get(work.pop(), ()):
            if target not in found:
                found.add(target)
                work.append(target)
    return frozenset(found)


def int_returning(database, image, base) -> frozenset:
    """Functions some caller reads $v0 after calling, before anything writes
    it: evidence of a result in $v0, found as float_returning finds a result
    in $f0 (straight-line after the call; it can miss evidence, never invent
    it). Passed down tail calls the same way."""
    from .structs import _read_registers, _written_register

    functions = {f["addr"]: f for f in database["functions"] if f["size"]}

    def decode(pc):
        return DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)

    found, tails = set(), collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        for pc in range(start, end, 4):
            ins = decode(pc)
            if ins.is_jump and ins.mnemonic == "j" and ins.target in functions \
                    and not (start <= ins.target < end):
                tails[start].add(ins.target)
            if not ins.is_call or ins.target not in functions:
                continue
            for after in range(pc + 8, min(end, pc + 8 + 4 * _RESULT_WINDOW), 4):
                step = decode(after)
                if 2 in _read_registers(step):
                    found.add(ins.target)
                    break
                if step.is_branch or step.is_jump or step.is_call or _written_register(step) == 2:
                    break
    work = list(found)
    while work:
        for target in tails.get(work.pop(), ()):
            if target not in found:
                found.add(target)
                work.append(target)
    return frozenset(found)


def v0_preserving(database, image, base) -> frozenset:
    """Functions that leave $v0 exactly as the caller had it.

    A function that never writes $v0, and calls only functions that never write it
    either, returns with the caller's $v0 still in place - and callers do rely on
    that, returning a value they set before the call. Treating such a call's $v0 as
    the callee's result made the model return whatever the callee happened to start
    with. A call through a register or a kernel call (which returns in $v0) rules
    a function out; the rest is a fixed point over the call graph.
    """
    from .structs import _written_register

    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    candidates, needs = set(), {}
    for address, func in functions.items():
        writes, callees = False, set()
        for pc in range(func["addr"], func["end"], 4):
            ins = DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)
            if _written_register(ins) == 2 or ins.mnemonic in ("syscall", "jalr"):
                writes = True
                break
            if ins.is_call or (ins.is_jump and not ins.is_return and ins.mnemonic != "jr"):
                if not (func["addr"] <= ins.target < func["end"]):
                    callees.add(ins.target)
            elif ins.mnemonic == "jr" and not ins.is_return:
                writes = True                     # a jump table: unknown targets
                break
        if not writes:
            candidates.add(address)
            needs[address] = callees
    preserving = set(candidates)
    changed = True
    while changed:
        changed = False
        for address in list(preserving):
            if not needs[address] <= preserving:
                preserving.discard(address)
                changed = True
    return frozenset(preserving)


def scanned_parameters(database, image, base) -> dict:
    """{function: sorted argument-vector indices it reads}, for every known function.

    A function that reads no argument maps to (), so that "takes nothing" and "not
    a function anyone found" stay distinguishable - a jump into the middle of
    some other code is not a call that can be made with no arguments.
    """
    found = {f["addr"]: set() for f in database["functions"] if f["size"]}
    for function, index in argument_use(database, image, base):
        found.setdefault(function, set()).add(index)
    for function, index in float_argument_use(database, image, base):
        found.setdefault(function, set()).add(index)
    return {function: tuple(sorted(indices)) for function, indices in found.items()}


# -- liveness ------------------------------------------------------------------
#
# The scans above follow each function in address order, so a register written on
# one branch and read after the paths rejoin looks written-before-read, although
# on the other path it is read first. That missed real arguments - a caller set
# $t1 and $t2 for a callee whose analysis said it took five - and every miss
# surfaced as a disagreement, since the call then passed nothing in those
# registers. Live-in analysis over the control-flow graph has no such blind spot.

ALL_ARGUMENTS = (1 << 16) - 1
_INT_INDEX = {reg: index for index, reg in enumerate((4, 5, 6, 7, 8, 9, 10, 11))}
_FLOAT_INDEX = {reg: FLOAT_ARG_BASE + k for k, reg in enumerate(FLOAT_ARG_REGS)}


def _effects(ins) -> tuple:
    """(uses, defs) of one instruction, as bitmasks over argument-vector indices."""
    from .structs import _read_registers, _written_register

    uses = defs = 0
    for reg in _read_registers(ins):
        if reg in _INT_INDEX:
            uses |= 1 << _INT_INDEX[reg]
    written = _written_register(ins)
    if written in _INT_INDEX:
        defs |= 1 << _INT_INDEX[written]
    read, wrote = _float_operands(ins)
    for reg in read:
        if reg in _FLOAT_INDEX:
            uses |= 1 << _FLOAT_INDEX[reg]
    for reg in wrote:
        if reg in _FLOAT_INDEX:
            defs |= 1 << _FLOAT_INDEX[reg]
    return uses, defs


def _graph(func, image, base, functions) -> list:
    """The function as nodes (uses, defs, successors, call target) in address order.

    A call is its own node after its delay slot: it uses whatever the callee reads
    and defines every argument register, all of which are caller-saved. A call
    target of None means "reads everything" - a call through a register, a jump
    table, a jump to somewhere that is not a known function.
    """
    start, end = func["addr"], func["end"]
    nodes = {}
    slots = set()

    def decode(address):
        return DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)

    address = start
    while address < end:
        if address in slots:
            address += 4
            continue
        ins = decode(address)
        uses, defs = _effects(ins)
        if not ins.has_delay_slot or address + 4 >= end:
            successors = [address + 4] if address + 4 < end else []
            if ins.has_delay_slot or (ins.is_jump and not ins.is_call):
                successors = []
            nodes[address] = (uses, defs, successors, False)
            address += 4
            continue

        slot = address + 4
        slots.add(slot)
        slot_uses, slot_defs = _effects(decode(slot))
        after = address + 8
        target = ins.target
        inside = start <= target < end

        if ins.is_call:
            callee = target if ins.mnemonic == "jal" and target in functions else None
            nodes[address] = (uses, defs, [slot], False)
            nodes[slot] = (slot_uses, slot_defs, [("call", address)], False)
            nodes[("call", address)] = (callee, ALL_ARGUMENTS, [after], True)
        elif ins.is_return:
            nodes[address] = (uses, defs, [slot], False)
            nodes[slot] = (slot_uses, slot_defs, [], False)
        elif ins.is_branch:
            branch_targets = [target] if inside else [("tail", address)]
            nodes[address] = (uses, defs, [slot] + ([after] if ins.is_likely else []), False)
            nodes[slot] = (slot_uses, slot_defs,
                           branch_targets + ([] if ins.is_likely else [after]), False)
            if not inside:
                nodes[("tail", address)] = (None, 0, [], True)
        else:                                     # j, b, jr to somewhere else
            nodes[address] = (uses, defs, [slot], False)
            if inside:
                nodes[slot] = (slot_uses, slot_defs, [target], False)
            else:
                callee = target if target in functions else None
                nodes[slot] = (slot_uses, slot_defs, [("tail", address)], False)
                nodes[("tail", address)] = (callee, 0, [], True)
        address += 8
    return nodes


def _live_in(nodes, start, parameters) -> int:
    live = {key: 0 for key in nodes}

    def uses_of(node):
        first, _defs, _successors, is_call = node
        if not is_call:
            return first
        if first is None:
            return ALL_ARGUMENTS
        mask = 0
        for index in parameters.get(first, ()):
            mask |= 1 << index
        return mask

    order = sorted((key for key in nodes if isinstance(key, int)), reverse=True)
    extra = [key for key in nodes if not isinstance(key, int)]
    changed = True
    while changed:
        changed = False
        for key in extra + order:
            node = nodes[key]
            out = 0
            for successor in node[2]:
                out |= live.get(successor, 0)
            value = uses_of(node) | (out & ~node[1])
            if value != live[key]:
                live[key] = value
                changed = True
    return live.get(start, 0)


def live_parameters(database, image, base, rounds: int = 40) -> dict:
    """{function: argument-vector indices live on entry}, for every known function.

    Solved to a fixed point over the call graph: what a function reads includes
    what it passes on to callees that read it.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    graphs = {address: _graph(func, image, base, functions)
              for address, func in functions.items()}
    parameters = {address: () for address in functions}
    for _round in range(rounds):
        changed = False
        for address, nodes in graphs.items():
            mask = _live_in(nodes, address, parameters)
            found = tuple(i for i in range(16) if mask >> i & 1)
            if found != parameters[address]:
                parameters[address] = found
                changed = True
        if not changed:
            break
    return parameters
