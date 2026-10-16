"""Recover structure types from how the code uses pointers.

The method is type unification in the style of Steensgaard's points-to analysis,
extended with field offsets:

* Every value that is dereferenced gets a *type node*: the arguments of each
  function, each function's return value, and the contents of pointer fields -
  if `lw $t0, 4($a0)` is followed by a dereference of `$t0`, then field 4 of
  whatever `$a0` points to is itself a pointer, to a type of its own.
* A node collects the offsets it is accessed at, with their widths.
* Evidence that two nodes are the same type joins them: an argument passed
  unchanged into a call, a value returned, a pointer stored into a field. Joining
  two types joins whatever their common pointer fields point to as well.

Unification is only as good as its links, and one generic routine - a memcpy, a
list helper - links every type it is ever handed. The measurement in
engine/tools/fields.py showed that cutting links into widely-called functions trades one
error for another: too strict a cutoff discards real links to central objects,
too loose a one fuses unrelated types into one 700-function blob. So instead of a
cutoff, each join is *checked*: two types are joined only if they agree on the
width of every offset both use, recursively through their pointer fields. A
generic routine is refused at the first type it disagrees with. Links are applied
in order of how specific the callee is - functions with fewer callers first - so
that generic routines meet already-established types and are refused, rather
than establishing a fused type first.

Because the guard uses widths, width agreement is no longer evidence of quality.
Two properties the guard never looks at are: whether an offset is used as float
or integer, and whether it is loaded signed or unsigned. `quality()` reports
agreement on those as the independent check.

The scan is flow-aware rather than linear. A register's meaning is carried along
each path, and where paths join only the meanings all of them agree on survive;
at the head of a loop, anything the loop body overwrites is dropped. Imprecision
therefore loses links rather than inventing them.
"""

from __future__ import annotations

import bisect
import collections
import struct

import re

from . import layout
from .r5900 import DISASM, GPR

# mnemonic -> (width, kind, signed); signed is None for stores
ACCESSES = {
    "lb": (1, "int", True), "lbu": (1, "int", False),
    "lh": (2, "int", True), "lhu": (2, "int", False),
    "lw": (4, "int", True), "lwu": (4, "int", False),
    "ld": (8, "int", False), "lq": (16, "int", False),
    "lwc1": (4, "float", None),
    "sb": (1, "int", None), "sh": (2, "int", None), "sw": (4, "int", None),
    "sd": (8, "int", None), "sq": (16, "int", None), "swc1": (4, "float", None),
}
STORES = {"sb", "sh", "sw", "sd", "sq", "swc1"}
MOVES = {"daddu", "addu", "or", "dadd"}
# All eight integer argument registers: $a0..$a3, then $t0..$t3.
ARG_REGS = (4, 5, 6, 7, 8, 9, 10, 11)
CALLER_SAVED = set(range(1, 16)) | {24, 25, 31}
# 16-byte accesses are block copies (`lq`/`sq` move a struct 16 bytes at a time),
# not fields; they say nothing about the layout inside the block.
BLOCK_WIDTH = 16
MAX_TERM_DEPTH = 4
# Interior pointers further than this from their base are not followed; real
# structures here span up to ~0x7000 bytes.
MAX_DISPLACEMENT = 0x8000

# The whole global address space, as one type node. A register holding a
# constant address holds (GLOBALS, address), so the same machinery that records
# fields of a struct records global variables: an access through it is a field of
# GLOBALS at an absolute address, a pointer loaded from one gets a type of its
# own, and a global's address passed into a function is an embedded object.
GLOBALS = ("globals",)
GP = 28


def _written_register(ins):
    """The general register an instruction writes, or None."""
    m = ins.mnemonic
    if ins.is_call:
        return 31
    if m in STORES or ins.is_branch or (ins.is_jump and not ins.is_call):
        return None
    if m in ("lwc1", "mtc1", "ctc1") or m.startswith(("add.", "sub.", "mul.", "div.", "c.")):
        return None
    if m in ACCESSES or m in ("lui", "addiu", "addi", "daddiu", "daddi", "andi", "ori",
                              "xori", "slti", "sltiu", "mfc1", "qmfc2", "cfc1"):
        return ins.rt
    if ins.rd:
        return ins.rd
    return None


_REGISTER = re.compile(r"\$[a-z0-9]+")
_NUMBER = {name: index for index, name in enumerate(GPR)}


def _read_registers(ins) -> set:
    """General registers an instruction reads.

    Taken from the operand text: every register named, less the destination when
    it is named only once as the first operand. `addu $a0, $a0, $t0` names $a0
    twice and so reads it.
    """
    names = [_NUMBER[n] for op in ins.operands for n in _REGISTER.findall(str(op))
             if n in _NUMBER]
    written = _written_register(ins)
    if written is not None and names and names[0] == written and names.count(written) == 1:
        names = names[1:]
    return set(names)


def argument_use(database, image, base) -> set:
    """(function, index) for every argument a function actually reads.

    An argument register that still holds the caller's value at a call is only
    evidence if the callee looks at it. Without this, a wrapper's untouched
    `$a1..$a3` were linked into every callee it made, which joined the arguments
    of unrelated functions into one type.

    A function reads an argument if an instruction reads the register before it
    is overwritten, or passes it to a callee that reads it - so this is solved to
    a fixed point over the call graph.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    used = set()
    passes = collections.defaultdict(set)    # (callee, i) -> {(caller, i)}
    for func in functions.values():
        start, end = func["addr"], func["end"]
        original = {reg: index for index, reg in enumerate(ARG_REGS)}
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
                            passes[(target, ARG_REGS.index(reg))].add((start, index))
                    if step.is_call:
                        for reg in CALLER_SAVED:
                            original.pop(reg, None)
                    continue
                for reg in _read_registers(step):
                    if reg in original:
                        used.add((start, original[reg]))
                written = _written_register(step)
                if written is not None:
                    original.pop(written, None)
            address += 8 if len(sequence) == 2 else 4
    # Propagate: passing an argument to a callee that reads it is a read.
    work = list(used)
    while work:
        slot = work.pop()
        for caller in passes.get(slot, ()):
            if caller not in used:
                used.add(caller)
                work.append(caller)
    return used


def _sign32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _depth(term) -> int:
    depth = 0
    while term[0] == "field":
        term = term[1]
        depth += 1
    return depth


def _intersect(states):
    """Register meanings every incoming path agrees on."""
    first = states[0]
    return {reg: term for reg, term in first.items()
            if all(state.get(reg) == term for state in states[1:])}


def scan(func, image, base, functions, used):
    """Return the type events one function produces.

    Events are ("access", term, offset, width, kind, signed, is_store) and
    ("link", term_a, term_b, callee) where callee is the function whose slot the
    link leads into (None for links within this function). A term names a value:
    ("arg", f, i), ("ret", f) or ("field", term, offset).
    """
    start, end = func["addr"], func["end"]
    address = start

    def decode(addr):
        return DISASM.decode(struct.unpack_from("<I", image, addr - base)[0], addr)

    # Registers written inside each loop, dropped at the loop head: the state there
    # was computed from the entry path alone and the back edge may disagree.
    loop_kill = collections.defaultdict(set)
    addr = start
    while addr < end:
        ins = decode(addr)
        if (ins.is_branch or ins.is_jump) and start <= ins.target <= addr:
            for inner in range(ins.target, min(addr + 8, end), 4):
                reg = _written_register(decode(inner))
                if reg is not None:
                    loop_kill[ins.target].add(reg)
        addr += 4

    events = []
    pending = collections.defaultdict(list)
    has_jump_table = any(
        ins.mnemonic == "jr" and ins.rs != 31
        for ins in (decode(a) for a in range(start, end, 4))
    )
    state = {reg: (("arg", start, index), 0) for index, reg in enumerate(ARG_REGS)}
    # $gp is fixed for the whole program, so small-data accesses are globals too.
    state[GP] = (GLOBALS, layout.CURRENT.gp)
    reachable = True

    # A register holds (term, displacement): the value `term` plus a constant. The
    # displacement is what makes interior pointers work - code that does
    # `addiu $v1, $a0, 0x1E4` and then stores at 0($v1), 4($v1)... is writing fields
    # 0x1E4 and 0x1E8 of $a0's type, and dropping $v1 lost them.
    def apply(ins, state):
        m = ins.mnemonic
        if m in ACCESSES:
            width, kind, signed = ACCESSES[m]
            is_store = m in STORES
            held = state.get(ins.rs)
            offset = None
            if held is not None:
                base_term, displacement = held
                offset = displacement + ins.imm
                # Storing $zero clears a field of any kind - an int, a float, a
                # pointer - so it is evidence of width but not of kind.
                stored_kind = "zero" if is_store and ins.rt == 0 else kind
                events.append(
                    ("access", base_term, offset, width, stored_kind, signed, is_store)
                )
            if is_store:
                value = state.get(ins.rt)
                if (held is not None and value is not None and value[1] == 0
                        and width == 4 and kind == "int" and ins.rt != 0
                        and _depth(held[0]) < MAX_TERM_DEPTH):
                    events.append(("link", ("field", held[0], offset), value[0], None))
                return
            if kind == "float":
                return
            if held is not None and width == 4 and _depth(held[0]) < MAX_TERM_DEPTH:
                state[ins.rt] = (("field", held[0], offset), 0)
            else:
                state.pop(ins.rt, None)
            return
        if m in MOVES and ins.rd:
            if ins.rt == 0 and ins.rs in state:
                state[ins.rd] = state[ins.rs]
                return
            if ins.rs == 0 and ins.rt in state:
                state[ins.rd] = state[ins.rt]
                return
        if m in ("addiu", "daddiu") and ins.rs in state:
            term, displacement = state[ins.rs]
            if term == GLOBALS or abs(displacement + ins.imm) <= MAX_DISPLACEMENT:
                state[ins.rt] = (term, _sign32(displacement + ins.imm))
                return
        if m == "lui":
            state[ins.rt] = (GLOBALS, _sign32(ins.imm << 16))
            return
        if m == "ori" and state.get(ins.rs, (None,))[0] == GLOBALS:
            state[ins.rt] = (GLOBALS, _sign32(state[ins.rs][1] | ins.imm))
            return
        reg = _written_register(ins)
        if reg is not None:
            state.pop(reg, None)

    def link_call(target, state):
        for index, reg in enumerate(ARG_REGS):
            held = state.get(reg)
            # Only an undisplaced pointer is the object itself; base+8 passed to a
            # callee is a pointer to something embedded at offset 8.
            if held is None or (target, index) not in used:
                continue
            if held[1] == 0:
                events.append(("link", held[0], ("arg", target, index), target))
            elif held[1] > 0:
                # base+d handed to a callee: the callee's argument type is
                # embedded in this one at offset d.
                events.append(("embed", held[0], held[1], ("arg", target, index), target))

    while address < end:
        if address in pending or address in loop_kill or not reachable:
            incoming = pending.pop(address, [])
            if reachable:
                incoming.append(state)
            state = _intersect(incoming) if incoming else {}
            for reg in loop_kill.get(address, ()):
                state.pop(reg, None)
            # Code after a return or a jump is reached only by a recorded branch.
            # Treating it as reachable with an empty state - padding after `jr $ra`,
            # say - then intersected that emptiness into the real branch target
            # and lost everything the branch carried. A jump table's targets are
            # unknown, though, so in a function with one everything stays
            # reachable, conservatively empty.
            reachable = bool(incoming) or has_jump_table

        ins = decode(address)
        if not ins.has_delay_slot or address + 4 >= end:
            apply(ins, state)
            address += 4
            continue
        delay = decode(address + 4)

        if ins.is_call:
            apply(delay, state)
            target = ins.target if ins.mnemonic == "jal" else -1
            if target in functions:
                link_call(target, state)
            for reg in CALLER_SAVED:
                state.pop(reg, None)
            if target in functions:
                state[2] = (("ret", target), 0)
        elif ins.is_return:
            apply(delay, state)
            if 2 in state and state[2][1] == 0:
                events.append(("link", state[2][0], ("ret", start), None))
            reachable = False
        elif ins.is_branch:
            taken = dict(state)
            apply(delay, taken)
            if address < ins.target < end:
                pending[ins.target].append(taken)
            # A likely branch nullifies its delay slot when not taken, so the
            # fall-through path may or may not have run it.
            state = _intersect([state, taken]) if ins.is_likely else taken
        else:   # unconditional jump
            apply(delay, state)
            target = ins.target
            if address < target < end:
                pending[target].append(state)
            elif ins.mnemonic == "j" and target in functions:
                # A tail call: the callee's arguments and result are this
                # function's own.
                link_call(target, state)
                events.append(("link", ("ret", target), ("ret", start), target))
            reachable = False
        address += 8
    return events


class Field:
    __slots__ = ("uses", "functions", "target", "strides")

    def __init__(self):
        self.uses = collections.Counter()    # (width, kind, signed, is_store) -> count
        self.functions = set()
        self.target = None
        # Strides this field was seen indexed with: the field is element 0 of an
        # array whose elements are that many bytes apart.
        self.strides = collections.Counter()

    def widths(self) -> set:
        return {use[0] for use in self.uses if use[0] != BLOCK_WIDTH}


class TypeGraph:
    """Union-find over type nodes, each with a map of fields."""

    def __init__(self):
        self.parent = []
        self.fields = []
        self.slots = []          # the argument/return terms each node stands for
        self.embeds = []         # node -> {offset: node of the embedded type}
        self.ids = {}            # term -> node, for arg and ret terms
        self.refused = 0
        self.joined = 0

    def new(self) -> int:
        self.parent.append(len(self.parent))
        self.fields.append({})
        self.slots.append(set())
        self.embeds.append({})
        return len(self.parent) - 1

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def node(self, term) -> int:
        if term[0] == "field":
            return self.target(self.node(term[1]), term[2])
        if term not in self.ids:
            self.ids[term] = self.new()
            self.slots[self.ids[term]].add(term)
        return self.ids[term]

    def field(self, node: int, offset: int) -> Field:
        fields = self.fields[self.find(node)]
        if offset not in fields:
            fields[offset] = Field()
        return fields[offset]

    def target(self, node: int, offset: int) -> int:
        field = self.field(node, offset)
        if field.target is None:
            field.target = self.new()
        return field.target

    def compatible(self, a: int, b: int) -> bool:
        """Whether joining a and b keeps every shared offset at one width set."""
        stack, seen = [(a, b)], set()
        while stack:
            x, y = stack.pop()
            x, y = self.find(x), self.find(y)
            if x == y or (x, y) in seen:
                continue
            seen.add((x, y))
            fx, fy = self.fields[x], self.fields[y]
            ex, ey = self.embeds[x], self.embeds[y]
            for offset in ex.keys() & ey.keys():
                stack.append((ex[offset], ey[offset]))
            for offset in fx.keys() & fy.keys():
                wx, wy = fx[offset].widths(), fy[offset].widths()
                # Identical, not merely overlapping: a node that already mixes
                # widths at an offset is a generic pointer (memcpy's argument),
                # and a subset rule let it absorb every type it touched.
                if wx and wy and wx != wy:
                    return False
                if fx[offset].target is not None and fy[offset].target is not None:
                    stack.append((fx[offset].target, fy[offset].target))
        return True

    def embed(self, outer: int, offset: int, inner: int) -> bool:
        """Record that `outer` contains an `inner` at `offset`."""
        embeds = self.embeds[self.find(outer)]
        if offset in embeds:
            return self.join(embeds[offset], inner)
        embeds[offset] = inner
        return True

    def join(self, a: int, b: int) -> bool:
        if self.find(a) == self.find(b):
            return True
        if not self.compatible(a, b):
            self.refused += 1
            return False
        self.joined += 1
        work = [(a, b)]
        while work:
            x, y = work.pop()
            x, y = self.find(x), self.find(y)
            if x == y:
                continue
            # Keep the node with more fields as the root, so merging stays cheap.
            if len(self.fields[x]) > len(self.fields[y]):
                x, y = y, x
            self.parent[x] = y
            self.slots[y] |= self.slots[x]
            self.slots[x] = set()
            into = self.fields[y]
            for offset, field in self.fields[x].items():
                if offset not in into:
                    into[offset] = field
                    continue
                other = into[offset]
                other.uses.update(field.uses)
                other.functions |= field.functions
                other.strides.update(field.strides)
                if field.target is not None:
                    if other.target is None:
                        other.target = field.target
                    else:
                        work.append((field.target, other.target))
            self.fields[x] = {}
            for offset, inner in self.embeds[x].items():
                if offset in self.embeds[y]:
                    work.append((inner, self.embeds[y][offset]))
                else:
                    self.embeds[y][offset] = inner
            self.embeds[x] = {}
        return True


def recover(database, image, base):
    """Scan every function and unify; returns the TypeGraph."""
    functions = {f["addr"] for f in database["functions"] if f["size"]}
    used = argument_use(database, image, base)
    callers = collections.defaultdict(set)
    events = []
    for func in database["functions"]:
        if not func["size"]:
            continue
        for event in scan(func, image, base, functions, used):
            events.append((func["addr"], event))
            if event[0] in ("link", "embed") and event[-1] is not None:
                callers[event[-1]].add(func["addr"])

    graph = TypeGraph()
    for owner, event in events:
        if event[0] == "access":
            _kind, term, offset, width, kind, signed, is_store = event
            field = graph.field(graph.node(term), offset)
            field.uses[(width, kind, signed, is_store)] += 1
            field.functions.add(owner)
    # Specific links first, so a generic routine meets established types and is
    # refused instead of fusing them before their own evidence arrives.
    links = [event for _owner, event in events if event[0] in ("link", "embed")]
    links.sort(key=lambda event: len(callers.get(event[-1], ())) if event[-1] else 0)
    for event in links:
        if event[0] == "link":
            graph.join(graph.node(event[1]), graph.node(event[2]))
        else:
            graph.embed(graph.node(event[1]), event[2], graph.node(event[3]))
    return graph


def types(graph):
    """Roots that have fields, as {root: fields}."""
    found = {}
    for node in range(len(graph.parent)):
        if graph.find(node) == node and graph.fields[node]:
            found[node] = graph.fields[node]
    return found


def quality(graph) -> dict:
    """Agreement on what the join guard never checks, at shared offsets."""
    shared = kind_agree = kind_total = sign_agree = sign_total = 0
    for fields in types(graph).values():
        for field in fields.values():
            if len(field.functions) < 2:
                continue
            shared += 1
            kinds = {use[1] for use in field.uses
                     if use[0] == 4 and use[1] not in ("zero", "derived")}
            if kinds:
                kind_total += 1
                kind_agree += len(kinds) == 1
            signs = {use[2] for use in field.uses if use[2] is not None and use[0] < 4}
            if signs:
                sign_total += 1
                sign_agree += len(signs) == 1
    return {
        "shared offsets": shared,
        "float/int agreement": kind_agree / max(1, kind_total),
        "signedness agreement": sign_agree / max(1, sign_total),
    }


class Catalog:
    """Recovered types, named and queryable by the code emitter.

    Names are placeholders, like the function names, and are derived from the
    evidence so they stay stable between runs: a type is named after the
    lowest-addressed function that takes it as an argument (`S_0035F0A0_a0` is
    the type of argument 0 of func_0035F0A0), or that returns it; a type known
    only as the target of a pointer field is named after that field
    (`S_0035F0A0_a0_p1C`).
    """

    def __init__(self, graph: TypeGraph) -> None:
        self.graph = graph
        self.roots = types(graph)
        self.names = {}
        self.globals = None
        self._arrays = {}
        if GLOBALS in graph.ids:
            # Kept even with no direct fields: a program whose globals are only
            # ever reached as objects passed to functions still has them.
            root = graph.find(graph.ids[GLOBALS])
            self.roots.setdefault(root, graph.fields[root])
            self.globals = root
            self.names[root] = "globals"
        named = []
        for root in self.roots:
            terms = sorted(t for t in graph.slots[root] if t[0] in ("arg", "ret"))
            if terms:
                term = terms[0]
                suffix = "a%d" % term[2] if term[0] == "arg" else "r"
                named.append(("S_%08X_%s" % (term[1], suffix), root))
        for name, root in sorted(named):
            self.names[root] = name
        # Name pointer-only types after the field that leads to them, breadth
        # first from the named ones so the shortest description wins.
        frontier = [root for root, _name in sorted(self.names.items(), key=lambda i: i[1])]
        while frontier:
            following = []
            for root in frontier:
                for offset in sorted(self.roots[root]):
                    target = self.pointee(root, offset)
                    if target is not None and target not in self.names:
                        if root == self.globals:
                            # What global variable 0x4876E8 points to.
                            self.names[target] = "S_g%08X" % (offset & 0xFFFFFFFF)
                        else:
                            self.names[target] = "%s_p%s" % (
                                self.names[root], _offset_name(offset))
                        following.append(target)
                for offset, inner in sorted(self.embedded(root).items()):
                    if inner not in self.names:
                        if root == self.globals:
                            self.names[inner] = "S_g%08X_obj" % (offset & 0xFFFFFFFF)
                        else:
                            self.names[inner] = "%s_e%s" % (
                                self.names[root], _offset_name(offset))
                        following.append(inner)
            frontier = following
        for root in sorted(self.roots):
            self.names.setdefault(root, "S_node%d" % root)

    def slot(self, function: int, index: int):
        node = self.graph.ids.get(("arg", function, index))
        if node is None:
            return None
        root = self.graph.find(node)
        return root if root in self.roots else None

    def field(self, root: int, offset: int):
        return self.roots.get(root, {}).get(offset)

    def pointee(self, root: int, offset: int):
        field = self.field(root, offset)
        if field is None or field.target is None:
            return None
        target = self.graph.find(field.target)
        return target if target in self.roots else None

    # The catalog is read-only once built, so these are computed once per type.
    # Recomputing them on every lookup was most of emission's time - eleven minutes
    # of a run - because the global address space embeds thousands of objects and
    # each typed access scanned all of them.

    def embedded(self, root: int):
        """{offset: root} for the types embedded in this one that have fields."""
        cache = self.__dict__.setdefault("_embedded_cache", {})
        if root not in cache:
            found = {}
            for offset, node in self.graph.embeds[root].items():
                inner = self.graph.find(node)
                if inner in self.roots and inner != root:
                    found[offset] = inner
            cache[root] = found
        return cache[root]

    def extent(self, root: int) -> int:
        """One past the last byte any field of the type is known to occupy."""
        cache = self.__dict__.setdefault("_extent_cache", {})
        if root not in cache:
            cache[root] = max(offset + max(field.widths() or {4}) for offset, field
                              in self.roots[root].items())
        return cache[root]

    def covering(self, root: int, offset: int) -> list:
        """(at, inner) for each embedded type whose extent covers `offset`,
        nearest first."""
        cache = self.__dict__.setdefault("_covering_cache", {})
        if root not in cache:
            items = sorted(self.embedded(root).items())
            starts = [at for at, _inner in items]
            spans = [self.extent(inner) for _at, inner in items]
            cache[root] = (starts, items, spans, max(spans, default=0))
        starts, items, spans, widest = cache[root]
        found = []
        index = bisect.bisect_right(starts, offset) - 1
        while index >= 0 and offset - starts[index] < widest:
            if offset < starts[index] + spans[index]:
                found.append(items[index])
            index -= 1
        return found

    def locate(self, root: int, offset: int):
        """Where a field at `offset` of `root` is: [(embed offset, inner root)...], field.

        A direct field wins. Otherwise the field is looked for in an embedded type
        whose extent covers the offset, recursively. Returns None if neither has it.
        """
        field = self.field(root, offset)
        if field is not None:
            return [], root, offset, field
        for at, inner in self.covering(root, offset):
            found = self.locate(inner, offset - at)
            if found is not None:
                path, owner, local, field = found
                return [(at, inner)] + path, owner, local, field
        return None

    def arrays(self, root: int) -> dict:
        """{start: (stride, count)} for the arrays in a type; count None if unbounded.

        Two kinds of evidence. A field the code indexes with a variable is element 0
        of an array of that stride - its length is not known. And a run of fields
        with the same access flavour at a fixed spacing is an array the code walked
        with an unrolled loop; there the run gives the length.

        Runs need to be long to count. A struct of eight consecutive ints looks
        exactly like an int array of eight, and calling a real struct an array
        would misdescribe it, so contiguous runs need 16 elements; a run with gaps
        between elements - records, each with several fields - needs 6. Global
        memory gets no run detection at all: neighbouring globals are usually
        separate variables.
        """
        if root in self._arrays:
            return self._arrays[root]
        fields = self.roots.get(root, {})
        found = {}
        for offset, field in fields.items():
            if field.strides:
                stride = max(sorted(field.strides), key=lambda s: field.strides[s])
                found[offset] = (stride, None)
        if root != self.globals:
            found.update(self._runs(fields, found))
        self._arrays[root] = found
        return found

    def _runs(self, fields, taken) -> dict:
        by_shape = collections.defaultdict(list)
        for offset, field in fields.items():
            shape = (self.read_flavour(field), self.write_width(field))
            by_shape[shape].append(offset)
        runs = {}
        for shape, offsets in by_shape.items():
            width = (shape[0] or (shape[1], None))[0]
            if not width:
                continue
            offsets.sort()
            members = set(offsets)
            claimed = set()
            for position, start in enumerate(offsets):
                if start in claimed or start in taken:
                    continue
                best = (0, 0)
                for following in offsets[position + 1:position + 9]:
                    stride = following - start
                    if stride < width or stride > 0x10000:
                        continue
                    count = 1
                    while start + count * stride in members:
                        count += 1
                    best = max(best, (count, -stride))
                count, stride = best[0], -best[1]
                needed = 16 if stride == width else 6
                if count >= needed:
                    runs[start] = (stride, count)
                    claimed.update(start + i * stride for i in range(count))
        return runs

    def element(self, root: int, offset: int):
        """(start, index) if a constant offset falls on an element of a known run."""
        for start, (stride, count) in self.arrays(root).items():
            if count is None or offset < start:
                continue
            index, remainder = divmod(offset - start, stride)
            if remainder == 0 and index < count:
                return start, index
        return None

    def candidates(self, root: int, offset: int):
        """Every way to reach a field at `offset`, best first.

        For an ordinary type the direct field comes first, as in `locate`. For the
        global address space it comes last: a global object's fields read as
        `g_0049A278_obj.f_4()`, and the same object's first field should not read
        as an unrelated-looking `g_0049A278()` just because the scan also saw it
        accessed directly.
        """
        direct = self.field(root, offset)
        nested = []
        for at, inner in self.covering(root, offset):
            for path, owner, local, field in self.candidates(inner, offset - at):
                nested.append(([(at, inner)] + path, owner, local, field))
        found = [([], root, offset, direct)] if direct is not None else []
        return nested + found if root == self.globals else found + nested

    @staticmethod
    def read_flavour(field):
        """The (width, signed) the field is most often loaded with, or None."""
        loads = collections.Counter()
        for (width, kind, signed, is_store), count in field.uses.items():
            if is_store or width == BLOCK_WIDTH:
                continue
            # lwc1 reads a raw 32-bit pattern; the expression model treats it as
            # an unsigned word load, so it shares that flavour.
            loads[(width, bool(signed))] += count
        if not loads:
            return None
        return max(sorted(loads), key=lambda key: loads[key])

    @staticmethod
    def write_width(field):
        widths = collections.Counter()
        for (width, _kind, _signed, is_store), count in field.uses.items():
            if is_store and width != BLOCK_WIDTH:
                widths[width] += count
        if not widths:
            return None
        return max(sorted(widths), key=lambda key: widths[key])

    @staticmethod
    def kind(field) -> str:
        kinds = {kind for (width, kind, _s, _st) in field.uses
                 if kind not in ("zero", "derived") and width != BLOCK_WIDTH}
        if kinds == {"float"}:
            return "float"
        if kinds == {"int"}:
            return "int"
        return "mixed" if kinds else "unknown"

    def functions(self, root: int) -> set:
        found = set()
        for field in self.roots[root].values():
            found |= field.functions
        return found


def _offset_name(offset: int) -> str:
    return "%X" % offset if offset >= 0 else "m%X" % -offset
