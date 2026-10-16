"""Structured C++ from machine code: loops and ifs instead of gotos, locals for registers.

Lifted code (lift.py) is one statement per instruction over the register context,
with gotos for control flow: correct and complete, and hard to read. This renders
the same statements in a form a person can follow:

* Control flow is structured. Natural loops become `for (;;)` with `continue` and
  `break`; a two-way branch whose arms meet again at its immediate post-dominator
  becomes `if`/`else`. Whatever does not fit - irreducible flow, jumps into the
  middle of another construct - stays a `goto`, so the result is always faithful.
* Registers are named locals (`a0`, `v0`, `f12`, `hi`), copied from the context on
  entry and back to it around anything that takes the context: calls, syscalls,
  the moves to and from VU0, returns. `out()` and `in()` do the copying.

The output keeps lift.py's interface - `void F_xxxxxxxx(lifted::Context& c)` - so
it is verified exactly as lifted code is, comparing the whole register file and
memory against the original. Functions with jump tables or linking branches are
left to the plain lifted form.
"""

from __future__ import annotations

import re
import struct

from . import lift
from .r5900 import DISASM

GPR_NAMES = ("zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
             "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
             "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
             "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra")
SCALARS = ("hi", "lo", "hi1", "lo1", "acc", "fcc", "sa")

# Callee-saved registers. A function saves them on entry and restores them before
# it returns, so to its caller they never change; here they are plain locals that
# are never written back, and the save and restore - stack traffic the caller
# cannot observe - are left out. A function that breaks the convention fails its
# verification and keeps the plain lifted form.
CALLEE_SAVED = (16, 17, 18, 19, 20, 21, 22, 23, 30)
_SAVED = r"(?P<reg>s[0-7]|fp|ra)"
_OFFSET = r"(?: \+ (?P<off>0x[0-9A-F]+)u)?"
_SLOT = r"\(\(u32\)sp%s\)" % _OFFSET
_SPILLS = [
    re.compile(r"^\s*\{ u32 q = %s & ~0xFu; Write64\(q, %s\); Write64\(q \+ 8, c\.rh\[\d+\]\); \}$"
               % (_SLOT, _SAVED)),
    re.compile(r"^\s*Write64\(\(u32\)sp%s, %s\);$" % (_OFFSET, _SAVED)),
    re.compile(r"^\s*\{ u32 q = %s & ~0xFu; u64 lo = Read64\(q\); u64 hi = Read64\(q \+ 8\); "
               r"%s = lo; c\.rh\[\d+\] = hi; \}$" % (_SLOT, _SAVED)),
    re.compile(r"^\s*%s = ReadU64\(\(u32\)sp%s\);$" % (_SAVED, _OFFSET)),
]


def _save_slots(code, start: int):
    """{(register name, stack offset)} the prologue saves callee-saved registers
    in: the stores of them to the stack before the first branch, jump or call.
    None when the function writes a callee-saved register it never saves - its
    callers would see the change, so nothing about its saves can be elided."""
    slots, saved, written = set(), set(), set()
    in_prologue = True
    for ins in code:
        if in_prologue and ins.mnemonic in ("sd", "sq", "sw") and ins.rs == 29 \
                and (ins.rt in CALLEE_SAVED or ins.rt == 31):
            slots.add((GPR_NAMES[ins.rt], ins.imm & 0xFFFF))
            saved.add(ins.rt)
        if ins.is_branch or ins.is_jump or ins.is_call:
            in_prologue = False
        # Eliding keeps a saved register in a local, and only its low 64 bits:
        # a COP2 move reaches it through the context, and a 128-bit write sets
        # its upper half there - both of which the caller would then see. In
        # 0x002284B0 `qmfc2 $s0` left a VU0 value in its caller's $s0.
        if ins.mnemonic in ("qmfc2", "qmtc2", "cfc2", "ctc2") and ins.rt in CALLEE_SAVED:
            return None
        rd = _written_gpr(ins)
        if rd in CALLEE_SAVED:
            written.add(rd)
            wide = ins.mnemonic == "lq" or (ins.mnemonic.startswith("p") and ins.mnemonic != "pref")
            if wide and not (ins.mnemonic == "lq" and ins.rs == 29
                             and (GPR_NAMES[rd], ins.imm & 0xFFFF) in slots):
                return None
    return slots if written <= saved else None


def _written_gpr(ins):
    """The general register an instruction writes, or None (from the analysis)."""
    from .structs import _written_register
    return _written_register(ins)


def _is_spill(text: str, slots) -> bool:
    """Whether `text` saves or restores a callee-saved register in its own save
    slot - bookkeeping no caller can see. A load of one from any other stack slot
    is data (a local the code keeps in a saved register) and is kept: dropping
    such a load once made software double multiplication loop forever."""
    if not slots:
        return False
    for pattern in _SPILLS:
        match = pattern.match(text)
        if match and (match.group("reg"), int(match.group("off") or "0", 16)) in slots:
            return True
    return False


class Unstructurable(lift.Refused):
    """This function is left in the plain lifted form."""


class Block:
    def __init__(self, start):
        self.start = start
        self.code = []           # statements before the terminator
        self.kind = "fall"       # fall, jump, cond, return, tail
        self.condition = None
        self.slot = ""           # the delay slot's statement
        self.likely = False
        self.taken = None        # address
        self.next = None         # fallthrough or jump target address
        self.tail = None         # address of a tail transfer
        self.succs = []


def _blocks(code, start, end, ranges=None):
    """Basic blocks keyed by start address, with terminators decoded. `ranges`
    are the function's pieces, the body first (see lift.lift_function)."""
    ranges = ranges or [(start, end)]
    inside = lambda address: any(first <= address < last for first, last in ranges)
    by_address = {ins.addr: ins for ins in code}
    leaders = {start} | {first for first, _last in ranges}
    for ins in code:
        if ins.has_delay_slot:
            if ins.mnemonic == "jr" and ins.rs != 31:
                raise Unstructurable("jump table")
            if ins.mnemonic in ("bltzal", "bgezal", "bltzall", "bgezall"):
                raise Unstructurable("linking branch")
            if not ins.is_call and ins.mnemonic not in ("jr", "jalr") and inside(ins.target):
                leaders.add(ins.target)
            if not ins.is_call:
                leaders.add(ins.addr + 8)
    slots = {ins.addr + 4 for ins in code if ins.has_delay_slot}
    if leaders & slots:
        raise Unstructurable("a jump lands in a delay slot")
    leaders = sorted(a for a in leaders if inside(a))
    blocks = {}
    for i, first in enumerate(leaders):
        block = Block(first)
        # A block ends at the next leader or its piece's end, whichever is first:
        # pieces are not contiguous, and one does not fall into the next.
        piece_end = next(last for f, last in ranges if f <= first < last)
        limit = min(leaders[i + 1], piece_end) if i + 1 < len(leaders) else piece_end
        address = first
        while address < limit:
            ins = by_address[address]
            if not ins.has_delay_slot:
                block.code.append(lift.statement(ins))
                address += 4
                continue
            slot = lift.statement(by_address[address + 4])
            m = ins.mnemonic
            after = address + 8
            link = "c.r[31] = 0x%08Xull;" % after
            if m == "jal":
                block.code += [slot, link, "lifted::Call(c, 0x%08Xu);" % ins.target]
                address = after
                continue
            if m == "jalr":
                block.code += ["{ u32 target = (u32)%s;" % lift._r(ins.rs), slot, link,
                               "lifted::Call(c, target); }"]
                address = after
                continue
            block.slot = slot
            if m == "jr":
                block.kind = "return"
            elif m in ("j", "b"):
                if inside(ins.target):
                    block.kind, block.next = "jump", ins.target
                else:
                    block.kind, block.tail = "tail", ins.target
            else:
                block.kind = "cond"
                block.condition = lift._CONDITIONS[m] % {"a": lift._r(ins.rs), "b": lift._r(ins.rt)}
                block.likely = m.endswith("l")
                block.taken = ins.target
                block.next = after
            address = after
            break
        else:
            block.next = limit           # falls into the next block, or off the end
        blocks[first] = block
    return blocks


def _successors(block, blocks):
    out = []
    for target in (block.taken, block.next):
        if target is not None and target in blocks and target not in out:
            out.append(target)
    return out


def _dominators(order, preds, entry):
    """Immediate dominators (Cooper, Harvey and Kennedy)."""
    index = {node: i for i, node in enumerate(order)}
    idom = {entry: entry}

    def intersect(a, b):
        while a != b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for node in order[1:]:
            known = [p for p in preds[node] if p in idom]
            if not known:
                continue
            new = known[0]
            for p in known[1:]:
                new = intersect(p, new)
            if idom.get(node) != new:
                idom[node] = new
                changed = True
    return idom


def _reverse_postorder(entry, succs):
    seen, order = set(), []

    def visit(node):
        stack = [(node, iter(succs[node]))]
        seen.add(node)
        while stack:
            current, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(current)
                stack.pop()
            elif child not in seen:
                seen.add(child)
                stack.append((child, iter(succs[child])))

    visit(entry)
    return list(reversed(order))


class _Emitter:
    def __init__(self, blocks, start, end):
        self.blocks = blocks
        self.end = end
        self.succs = {a: _successors(b, blocks) for a, b in blocks.items()}
        self.order = _reverse_postorder(start, self.succs)
        reachable = set(self.order)
        preds = {a: [] for a in reachable}
        for a in reachable:
            for s in self.succs[a]:
                preds[s].append(a)
        self.idom = _dominators(self.order, preds, start)
        # Post-dominators: dominators of the reversed graph from a virtual exit.
        exit_node = "exit"
        rsuccs = {exit_node: []}
        rpreds = {a: [] for a in reachable}
        rpreds[exit_node] = []
        for a in reachable:
            rsuccs.setdefault(a, [])
            outs = [s for s in self.succs[a]]
            if not outs or blocks[a].kind in ("return", "tail") or (
                    blocks[a].kind in ("fall", "cond") and blocks[a].next not in blocks):
                outs = outs + [exit_node]
            for s in outs:
                rsuccs.setdefault(s, []).append(a)
        rorder = _reverse_postorder(exit_node, rsuccs)
        rpred = {n: [] for n in rorder}
        for n in rorder:
            for s in rsuccs.get(n, []):
                if s in rpred:
                    rpred[s].append(n)
        self.ipdom = _dominators(rorder, rpred, exit_node)
        # Loops: a back edge goes to a block that dominates its source.
        self.headers = set()
        self.loops = {}
        for a in reachable:
            for s in self.succs[a]:
                if self._dominates(s, a):
                    self.headers.add(s)
                    self.loops.setdefault(s, set()).add(a)
        self.body = {h: self._natural_loop(h, sources, preds) for h, sources in self.loops.items()}
        self.emitted = set()
        self.labels = set()
        # Small blocks reached a second time are emitted again rather than jumped
        # to - a shared error path, the tail of an `a || b` - up to this many.
        self.copies_left = 64

    def _duplicable(self, node):
        """Whether a jump to this already-emitted block may be a copy of it instead."""
        block = self.blocks.get(node)
        if block is None or node in self.headers or self.copies_left <= 0:
            return False
        if len([s for s in block.code if s]) > 6 or block.kind == "cond":
            return False
        self.copies_left -= 1
        return True

    def _returns_soon(self, node, budget=6):
        """A short run of blocks that ends the function: emitting it at a loop exit
        costs less than a goto past the loop to it."""
        block = self.blocks.get(node)
        if block is None or node in self.headers:
            return False
        size = len([x for x in block.code if x]) + (1 if block.slot else 0)
        if size > budget:
            return False
        if block.kind in ("return", "tail"):
            return True
        if block.kind in ("fall", "jump") and block.next in self.blocks:
            return self._returns_soon(block.next, budget - size)
        return False

    def _dominates(self, a, b):
        while True:
            if a == b:
                return True
            parent = self.idom.get(b)
            if parent is None or parent == b:
                return False
            b = parent

    def _natural_loop(self, header, sources, preds):
        body, work = {header}, list(sources)
        while work:
            node = work.pop()
            if node not in body:
                body.add(node)
                work.extend(preds[node])
        return body

    def _follow(self, header):
        """The single block a loop exits to, or None."""
        exits = {s for n in self.body[header] for s in self.succs[n] if s not in self.body[header]}
        return next(iter(exits)) if len(exits) == 1 else None

    def emit(self, node, stop, loops, depth):
        lines = []
        pad = "    " * depth
        while node is not None and node != stop:
            if node not in self.blocks:
                lines.append(pad + "out(); lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); return;" % node)
                return lines
            if loops and node not in self.body[loops[-1][0]]:
                # Leaving the loop: break, a goto to wherever it goes, or - for a
                # small block that returns - that block, right here.
                jump = self._transfer(node, loops, depth)
                if jump is not None:
                    lines.append(jump)
                    return lines
            if node in self.emitted:
                if self._duplicable(node):
                    body, node = self._chain(node, stop, loops, depth, copy=True)
                    lines += body
                    continue
                lines.append(pad + "goto L_%08X;" % node)
                self.labels.add(node)
                return lines
            if node in self.headers and (not loops or loops[-1][0] != node):
                follow = self._follow(node)
                lines.append(pad + "for (;;) {")
                lines.append(pad + "    lifted::Tick(c, 0x%08Xu);" % node)
                inner = self._block(node, None, loops + [(node, follow)], depth + 1)
                lines += inner
                lines.append(pad + "}")
                node = follow
                continue
            body, node = self._chain(node, stop, loops, depth)
            lines += body
        return lines

    def _block(self, node, stop, loops, depth):
        """The header's code then the rest of the loop body (`node` already marked)."""
        lines, nxt = self._chain(node, stop, loops, depth)
        if nxt is not None:
            lines += self.emit(nxt, stop, loops, depth)
        return lines

    def _transfer(self, target, loops, depth):
        """The statement for an edge to `target`, or None to fall into it."""
        pad = "    " * depth
        if target is not None and target not in self.blocks:
            return pad + "out(); lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); return;" % target
        if loops:
            header, follow = loops[-1]
            if target == header:
                return pad + "continue;"
            if target == follow:
                return pad + "break;"
            if target not in self.body[header]:
                if self._returns_soon(target):
                    return None
                self.labels.add(target)
                return pad + "goto L_%08X;" % target
        if target in self.emitted and not self._duplicable(target):
            self.labels.add(target)
            return pad + "goto L_%08X;" % target
        return None

    def _chain(self, node, stop, loops, depth, copy=False):
        """Emit one block; returns (lines, the block to continue with or None).

        A copy - a second emission of a small block - carries no label: the label
        stays with the first.
        """
        pad = "    " * depth
        block = self.blocks[node]
        self.emitted.add(node)
        lines = [] if copy else ["@L_%08X" % node]   # a label placeholder, kept if targeted
        lines += [pad + s for s in block.code if s]
        if block.kind == "return":
            lines += [pad + s for s in (block.slot,) if s]
            lines.append(pad + "out(); return;")
            return lines, None
        if block.kind == "tail":
            lines += [pad + s for s in (block.slot,) if s]
            lines.append(pad + "out(); lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); return;" % block.tail)
            return lines, None
        if block.kind in ("fall", "jump"):
            if block.slot:
                lines.append(pad + block.slot)
            target = block.next
            if target is not None and target not in self.blocks:
                lines.append(pad + "out(); lifted::TailCall(c, 0x%08Xu, entry_ra, entry_sp); "
                             "return;  /* runs off the end */" % target)
                return lines, None
            jump = self._transfer(target, loops, depth)
            if jump is not None:
                lines.append(jump)
                return lines, None
            return lines, target
        # A two-way branch.
        taken, fall = block.taken, block.next
        plain = not block.likely and not block.slot
        if block.likely:
            lines.append(pad + "if (%s) {" % block.condition)
            if block.slot:
                lines.append(pad + "    " + block.slot)
        elif plain:
            # Nothing runs between the test and the branch: no temporary needed.
            lines.append(pad + "if (%s) {" % block.condition)
        else:
            lines.append(pad + "{ const bool t = %s;" % block.condition)
            if block.slot:
                lines.append(pad + "  " + block.slot)
            lines.append(pad + "  if (t) {")
        join = self.ipdom.get(node)
        if join == "exit" or (loops and join not in self.body[loops[-1][0]]):
            join = None
        inner = depth + (1 if block.likely else 1)
        lines += self._arm(taken, join, loops, inner + (0 if block.likely else 0))
        lines.append(pad + ("} else {" if block.likely or plain else "  } else {"))
        lines += self._arm(fall, join, loops, inner)
        lines.append(pad + ("}" if block.likely or plain else "  } }"))
        if join is None:
            return lines, None
        jump = self._transfer(join, loops, depth)
        if jump is not None:
            lines.append(jump)
            return lines, None
        return lines, join

    def _arm(self, target, join, loops, depth):
        if target == join:
            return []
        jump = self._transfer(target, loops, depth + 1)
        if jump is not None:
            return [jump]
        return self.emit(target, join, loops, depth + 1)


_ATOM = r"(?:[A-Za-z_]\w*|0x[0-9A-F]+u(?:ll)?|\d+u(?:ll)?)"


def tidy(line: str) -> str:
    """Rewrites that cannot change a value, for reading. The structured function is
    verified after them, so a wrong one would show as a disagreement."""
    line = line.replace("lifted::", "").replace("generated::", "")
    # Sx32 keeps the low 32 bits, which a (u32) on an addend or a left shift
    # cannot change.
    line = re.sub(r"Sx32\(\(u32\)(%s) ([+-]|<<) \(u32\)(%s)\)" % (_ATOM, _ATOM),
                  r"Sx32(\1 \2 \3)", line)
    line = re.sub(r"Sx32\(\(u32\)(%s) ([+-]|<<) (%s)\)" % (_ATOM, _ATOM), r"Sx32(\1 \2 \3)", line)
    # Adding or or-ing zero: how MIPS spells a move.
    line = re.sub(r"= (%s) [+|] 0ull;" % _ATOM, r"= \1;", line)
    line = re.sub(r"= 0ull [+|] (%s);" % _ATOM, r"= \1;", line)
    line = re.sub(r"Sx32\((%s) \+ 0x0u\)" % _ATOM, r"Sx32(\1)", line)
    line = re.sub(r"Sx32\(0ull \+ (%s)\)" % _ATOM, r"Sx32(\1)", line)
    line = re.sub(r"\(u32\)(%s) \+ 0x0u\b" % _ATOM, r"(u32)\1", line)
    # A sign-extended literal below 2^31 is that literal; small ones read better
    # in decimal where they are compared.
    line = re.sub(r"Sx32\(0x([0-7]?[0-9A-F]{1,7})u\)", r"0x\1ull", line)
    line = re.sub(r"\(s64\)0x([0-9A-F]{1,4})ull\b", lambda m: str(int(m.group(1), 16)), line)
    line = re.sub(r"\(u(32|64)\)\((0x[0-9A-F]+ull)\)", r"(u\1)\2", line)
    # A comparison's 0 or 1, widened and tested against zero, is the comparison.
    operand = r"(?:\(s64\))?(?:%s|\d+)" % _ATOM
    line = re.sub(r"\(\(u64\)\((%s (?:<|==|!=) %s)\)\) != 0ull" % (operand, operand),
                  r"\1", line)
    line = re.sub(r"\(\(u64\)\((%s (?:<|==|!=) %s)\)\) == 0ull" % (operand, operand),
                  r"!(\1)", line)
    return _inner_sign_extensions(line)


def _closing(text, open_at):
    """The index of the parenthesis closing the one at `open_at`, or None."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _inner_sign_extensions(line):
    """`Sx32(a + (Sx32(b)))` is `Sx32(a + (b))`: the outer Sx32 keeps only the low
    32 bits of the sum, which depend only on the low 32 bits of each addend - and
    those Sx32 does not change. Likewise for subtraction."""
    changed = True
    while changed:
        changed = False
        for outer in re.finditer(r"Sx32\(", line):
            open_at = outer.end() - 1
            close = _closing(line, open_at)
            if close is None:
                continue
            argument = line[open_at + 1:close]
            for inner in re.finditer(r"(?<= [+-] )\(Sx32\(|^\(Sx32\(", argument):
                start = inner.start()
                end = _closing(argument, start)
                if end is None:
                    continue
                # Only a direct addend: `(Sx32(...))` with nothing else inside.
                if _closing(argument, start + 5) != end - 1:
                    continue
                # And the outer argument is a plain sum or difference at top level.
                depth, top = 0, []
                for i, ch in enumerate(argument):
                    depth += ch == "("
                    depth -= ch == ")"
                    if depth == 0 and ch in "*/&|^<>?%":
                        top.append(ch)
                if top:
                    continue
                argument = argument[:start] + "(" + argument[start + 6:end - 1] + ")" + argument[end + 1:]
                line = line[:open_at + 1] + argument + line[close:]
                changed = True
                break
            if changed:
                break
    # (u32)(Sx32(x)) is (u32)(x): the low 32 bits are what Sx32 keeps.
    while True:
        match = re.search(r"\(u32\)\(Sx32\(", line)
        if not match:
            return line
        open_at = match.start() + 5
        end = _closing(line, open_at)
        if end is None or _closing(line, open_at + 5) != end - 1:
            return line
        line = line[:open_at] + "(" + line[open_at + 6:end - 1] + ")" + line[end + 1:]


def _loop_test(lines, k):
    """At lines[k], `if (C) { continue; } else { break; }` (or the reverse):
    (the condition to repeat on, the number of lines it takes), or None."""
    shapes = ((("continue;", "} else {", "break;", "}"), False),
              (("break;", "} else {", "continue;", "}"), True),
              (("continue;", "}", "break;"), False),
              (("break;", "}", "continue;"), True))
    head = lines[k].strip()
    if not (head.startswith("if (") and head.endswith(") {")):
        return None
    condition = head[4:-3]
    for shape, negate in shapes:
        rest = [line.strip() for line in lines[k + 1:k + 1 + len(shape)]]
        if tuple(rest) == shape:
            return ("!(%s)" % condition if negate else condition), 1 + len(shape)
    return None


_FLIP = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


def negate(condition: str) -> str:
    """The opposite condition, spelled as plainly as it safely can be."""
    match = re.match(r"^([^ ()]*(?:\([us]64\)[^ ()]+)?) (==|!=|<|>=|>|<=) ([^ ()]+)$", condition)
    if match:
        return "%s %s %s" % (match.group(1), _FLIP[match.group(2)], match.group(3))
    if condition.startswith("!(") and _closing(condition, 1) == len(condition) - 1:
        return condition[2:-1]
    return "!(%s)" % condition


def _close_of(lines, k, texts):
    """The first line after k that is exactly one of `texts`, or None."""
    return next((j for j in range(k + 1, len(lines)) if lines[j] in texts), None)


_PLAIN_SLOT = re.compile(r"^([a-z][a-z0-9]*) = [^;]*;$")


def hoist_delay_slots(lines):
    """`{ const bool t = C; S; if (t) {...} else {...} }` is `S; if (C) {...}
    else {...}` when S cannot change C: S is a plain assignment to a name C does
    not mention, writing no memory and calling nothing. With the slot elided, it
    is just `if (C)`."""
    lines = list(lines)
    k = 0
    while k + 1 < len(lines):
        head = lines[k].strip()
        if not (head.startswith("{ const bool t = ") and head.endswith(";")):
            k += 1
            continue
        pad = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        condition = head[len("{ const bool t = "):-1]
        if lines[k + 1] == pad + "  if (t) {":
            slot, test = None, k + 1
        else:
            slot, test = lines[k + 1].strip(), k + 2
            match = _PLAIN_SLOT.match(slot)
            if (test >= len(lines) or lines[test] != pad + "  if (t) {" or not match
                    or re.search(r"\b%s\b" % match.group(1), condition)
                    or any(word in slot for word in ("Write", "Call", "(c"))
                    or "(c" in condition):
                k += 1
                continue
        middle = _close_of(lines, test, (pad + "  } else {", pad + "  } }"))
        if middle is None:
            k += 1
            continue
        end = middle if lines[middle] == pad + "  } }" else _close_of(lines, middle, (pad + "  } }",))
        if end is None:
            k += 1
            continue
        lines[end] = pad + "}"
        if middle != end:
            lines[middle] = pad + "} else {"
        lines[k:test + 1] = ([pad + slot] if slot else []) + [pad + "if (%s) {" % condition]
        k += 1
    return lines


def invert_empty_then(lines):
    """`if (C) { } else { A }` is `if (!C) { A }`."""
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (out and stripped == "} else {" and out[-1].strip().startswith("if (")
                and out[-1].strip().endswith(") {")
                and out[-1][:len(out[-1]) - len(out[-1].lstrip())]
                == line[:len(line) - len(line.lstrip())]):
            pad = line[:len(line) - len(line.lstrip())]
            out[-1] = pad + "if (%s) {" % negate(out[-1].strip()[4:-3])
            continue
        out.append(line)
    return out


def loop_guards(lines):
    """A loop ending in `if (C) { A continue; } else { break; }` is `if (!C)
    break; A`; and a loop whose first test that is - `for (;;) { Tick; if (!C)
    break; ...}` - is `while (C) { Tick; ... }`."""
    lines = list(lines)
    k = 0
    while k < len(lines):
        if lines[k].strip() != "for (;;) {":
            k += 1
            continue
        pad = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        close = _close_of(lines, k, (pad + "}",))
        if close is None:
            k += 1
            continue
        inner = pad + "    "
        arm = inner + "        "
        if (lines[close - 1] == inner + "}" and lines[close - 2] == arm + "break;"
                and lines[close - 3] == inner + "} else {" and lines[close - 4] == arm + "continue;"):
            start = next((j for j in range(close - 5, k, -1)
                          if lines[j].startswith(inner + "if (") and lines[j].endswith(") {")
                          and not lines[j].startswith(inner + " ")), None)
            body = lines[start + 1:close - 4] if start is not None else None
            if body is not None and all(line.startswith(arm) or line.startswith("L_")
                                        or not line.strip() for line in body):
                condition = lines[start].strip()[4:-3]
                lines[start:close] = ([inner + "if (%s) break;" % negate(condition)]
                                      + [line[8:] if line.startswith(arm) else line for line in body])
                close = _close_of(lines, k, (pad + "}",))
        # The guard straight after the tick: a while loop.
        if (k + 2 < close and lines[k + 1].strip().startswith("Tick(c, ")
                and re.match(r"^if \((.*)\) break;$", lines[k + 2].strip())
                and lines[k + 2].startswith(inner + "if (")):
            condition = lines[k + 2].strip()[4:-len(") break;")]
            lines[k:k + 3] = [pad + "while (%s) {" % negate(condition), lines[k + 1]]
        k += 1
    return lines


_LINK = re.compile(r"^(\s*)ra = 0x(00[0-7][0-9A-F]{5})ull;$")
_PLAIN_CALL = re.compile(r"^(\s*)out\(\); Call\(c, (0x[0-9A-F]{8}u)\); in\(\);$")


def fold_links(lines):
    """`ra = R; out(); Call(c, T); in();` is `out(); Call(c, T, R); in();` - the
    link a jal sets, given to the call that uses it."""
    out = []
    for line in lines:
        call = _PLAIN_CALL.match(line)
        link = _LINK.match(out[-1]) if out else None
        if call and link and link.group(1) == call.group(1):
            out[-1] = "%sout(); Call(c, %s, 0x%su); in();" % (call.group(1), call.group(2),
                                                         link.group(2))
            continue
        out.append(line)
    return out


def loops_to_do_while(lines):
    """`for (;;) { ...; if (C) continue; else break; }` is `do { ... } while (C);`
    when nothing else in the body continues - a `continue` in a do-while goes to
    the test, not the top."""
    lines = list(lines)
    k = 0
    while k < len(lines):
        if lines[k].strip() != "for (;;) {":
            k += 1
            continue
        pad = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        close = next((j for j in range(k + 1, len(lines)) if lines[j] == pad + "}"), None)
        if close is None:
            k += 1
            continue
        found = None
        for start in (close - 5, close - 4):
            test = _loop_test(lines, start) if start > k else None
            if test is not None and start + test[1] == close:
                found = (start, test[0])
        if found is None:
            k += 1
            continue
        start, condition = found
        body = lines[k + 1:start]
        if any("continue;" in line for line in body) or any(
                line.lstrip().startswith("L_") for line in lines[start:close]):
            k += 1
            continue
        lines[k:close + 1] = [pad + "do {"] + body + [pad + "} while (%s);" % condition]
        k += 1
    return lines


_SET_CONSTANT = re.compile(r"^(\s*)([a-z][a-z0-9]*) = (0x[0-9A-F]+)ull;$")
_ADD_CONSTANT = re.compile(r"^(\s*)([a-z][a-z0-9]*) = Sx32\(([a-z][a-z0-9]*) \+ (0x[0-9A-F]+)u\);$")


def drop_empty_else(lines):
    """`} else {` straight before the arm's closing brace: an empty else arm."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        following = lines[i + 1].strip() if i + 1 < len(lines) else None
        if line.strip() == "} else {" and following == "}":
            out.append(line.replace("} else {", "}"))
            i += 2
            continue
        if line.strip() == "} else {" and following == "} }":
            out.append(line.replace("} else {", "} }"))
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def fold_constants(lines, strings=None):
    """Join `x = 0x520000; x = Sx32(x + 0xFFFFC220u)` into one constant: how MIPS
    builds a 32-bit address in two steps. Only within straight-line code - any
    line that is not a plain assignment forgets what is known. A constant that
    is the address of a known string says so."""
    strings = strings or {}
    known, out = {}, []
    for line in lines:
        match = _ADD_CONSTANT.match(line)
        if match and match.group(3) in known:
            pad, target, source, addend = match.groups()
            value = (known[source] + int(addend, 16)) & 0xFFFFFFFF
            value = (value - (1 << 32) if value & 0x80000000 else value) & ((1 << 64) - 1)
            line = "%s%s = 0x%08Xull;" % (pad, target, value)
        match = _SET_CONSTANT.match(line)
        if match:
            pad, target, literal = match.groups()
            known[target] = int(literal, 16)
            # Not lui's value alone (its low half zero): the upper half of an
            # address still to be added to, wherever a string begins.
            if known[target] & 0xFFFF:
                line += _string_note(known[target], strings)
            out.append(line)
            continue
        assigned = re.match(r"^\s*([a-z][a-z0-9]*) = ", line)
        if assigned and ";" in line and "{" not in line and "}" not in line:
            known.pop(assigned.group(1), None)
        else:
            known = {}
        out.append(line)
    return out


# A split address once its base register is folded away (liveness.fold puts
# `lui`'s constant in place of the register): the whole address of a memory
# access, or the whole argument of a sign extension. Only there - as a whole
# argument, nothing around it binds tighter than the sum.
_SPLIT_ADDRESS = re.compile(r"((?:Read|Write)(?:U|S)?(?:8|16|32|64)\()\(u32\)(0x[0-9A-F]+)ull \+ (0x[0-9A-F]+)u(?=[),])")
_SPLIT_SIGN_EXTENDED = re.compile(r"\bSx32\(\((0x[0-9A-F]+)ull\) \+ (0x[0-9A-F]+)u\)")
_ASSIGNED_CONSTANT = re.compile(r"^\s*[a-z][a-z0-9]* = (0x[0-9A-F]+)ull;$")


def _string_note(value, strings):
    text = strings.get(value & 0xFFFFFFFF)
    if text is None:
        return ""
    shown = text if len(text) <= 48 else text[:45] + "..."
    return '  /* "%s" */' % shown.encode("unicode_escape").decode("ascii").replace("*/", "*\\/")


def fold_addresses(lines, strings=None):
    """`ReadS32((u32)0x4A0000ull + 0xFFFFB0A0u)` is `ReadS32(0x0049B0A0u)`, and
    `Sx32((0x4C0000ull) + 0xFFFF9AD0u)` the constant it sign-extends: a 32-bit
    address MIPS builds in two steps, whose first step folding put in place of
    its register. A plain assignment of an address of a known string says so."""
    strings = strings or {}
    out = []
    for line in lines:
        line = _SPLIT_ADDRESS.sub(
            lambda m: "%s0x%08Xu" % (m.group(1), (int(m.group(2), 16) + int(m.group(3), 16)) & 0xFFFFFFFF), line)

        def extended(m):
            value = (int(m.group(1), 16) + int(m.group(2), 16)) & 0xFFFFFFFF
            if value & 0x80000000:
                value |= 0xFFFFFFFF00000000
            return "0x%08Xull" % value
        line = _SPLIT_SIGN_EXTENDED.sub(extended, line)
        # Not a value lui made alone (its low half zero): that is the upper
        # half of an address still to be added to, wherever a string begins.
        assigned = _ASSIGNED_CONSTANT.match(line)
        if assigned and "/*" not in line and int(assigned.group(1), 16) & 0xFFFF:
            line += _string_note(int(assigned.group(1), 16), strings)
        out.append(line)
    return out


def _localize(text, gprs, fprs, scalars):
    text = re.sub(r"c\.r\[(\d+)\]", lambda m: _gpr(int(m.group(1)), gprs), text)
    text = re.sub(r"c\.f\[(\d+)\]", lambda m: _fpr(int(m.group(1)), fprs), text)
    for name in SCALARS:
        if re.search(r"\bc\.%s\b" % name, text):
            scalars.add(name)
            text = re.sub(r"\bc\.%s\b" % name, name, text)
    return text


def _gpr(index, used):
    used.add(index)
    return GPR_NAMES[index] if index else "0ull"


def _fpr(index, used):
    used.add(index)
    return "f%d" % index


# Statements that hand the context to something else; registers are synchronised
# around them.
_TAKES_CONTEXT = re.compile(r"\b(?:lifted::)?(Call|TailCall|Syscall|Cop2Move|Vcallms)\(c")


# What a caller can see of a function's registers after it returns, under the
# calling convention: the results, the stack and global pointers, and the
# callee-saved registers (kept as locals that are never written back).
ABI_RETURNED = ("v0", "v1", "sp", "gp", "f0", "f1")
# What a caller relies on finding as it left it: the callee-saved registers,
# integer and float. A structured function keeps them in locals; any it hands
# to its callees through the context (out(), before each call) it must hand
# back as it returns - its epilogue having restored the locals - or its caller
# sees the values it worked with. 0x0042DAE0 used $f20 and $f21 that way,
# restored them into its locals and returned without them: the car select's
# cars were not drawn. Its tests never took a path with a call.
ABI_PRESERVED = ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "fp") + tuple("f%d" % i for i in range(20, 32))


def _prepare_readable(blocks, elide, slots=None):
    """Localise every block's statements, then fold single-use temporaries.

    Returns the names to synchronise around calls. Folding needs the statements
    in their final form - locals, tidied, saves elided, calls wrapped - so that is
    done here, per block, and the later per-line pass finds nothing left to do.
    """
    from . import liveness

    gprs, fprs, scalars = set(), set(), set()

    def finish(text):
        text = tidy(_localize(text, gprs, fprs, scalars))
        if _TAKES_CONTEXT.search(text) and "out();" not in text:
            text = "out(); %s in();" % text.strip()
        return text

    for block in blocks.values():
        code = [finish(s) for s in block.code if s]
        if elide:
            code = [s for s in code if not _is_spill(s, slots)]
        block.code = code
        block.slot = finish(block.slot) if block.slot else block.slot
        if block.slot and elide and _is_spill(block.slot, slots):
            block.slot = ""
        if block.condition:
            block.condition = finish(block.condition)
    gprs.discard(0)
    synced = {GPR_NAMES[i] for i in gprs if not (elide and i in CALLEE_SAVED)}
    synced |= {"f%d" % i for i in fprs} | set(scalars)
    returned = (set(ABI_RETURNED) | set(ABI_PRESERVED)) & synced
    succs = {a: _successors(b, blocks) for a, b in blocks.items()}
    live = liveness.live_out(blocks, succs, synced, returned,
                             lambda b: b.condition, lambda b: b.slot)
    for address, block in blocks.items():
        later = [block.slot] if block.slot else []
        block.code, condition = liveness.fold(block.code, block.condition, live[address],
                                              synced, returned, later)
        block.condition = condition
    gprs.add(0)
    return gprs, fprs, scalars


def structure_function(image: bytes, base: int, start: int, end: int,
                       elide: bool = True, strings=None, readable: bool = False,
                       chunks=()) -> str:
    """Structured C++ for one function; raises lift.Refused if it cannot be.

    `elide` drops the saves and restores of callee-saved registers (see
    CALLEE_SAVED); a function that does not keep the convention needs it off.

    `readable` also folds single-use temporaries into their uses (liveness.py)
    and gives back on return only what the calling convention returns. That
    changes registers no caller may rely on, so it is verified at that level:
    results, stack and global pointers, callee-saved registers and memory.
    """
    lift.lift_function(image, base, start, end, chunks)    # the same refusals as lifting
    ranges = [(start, end)] + sorted((s, e) for s, e in chunks if e > s)
    code = []
    for first, last in ranges:
        words = struct.unpack_from("<%dI" % ((last - first) // 4), image, first - base)
        code += [DISASM.decode(word, first + 4 * i) for i, word in enumerate(words)]
    blocks = _blocks(code, start, end, ranges)
    slots = _save_slots(code, start)
    elide = elide and slots is not None
    if readable:
        has_tail = any(b.kind == "tail"
                       or (b.taken is not None and b.taken not in blocks)
                       or (b.next is not None and b.next not in blocks and b.kind != "return")
                       for b in blocks.values())
        elide = elide and not has_tail
        seen = _prepare_readable(blocks, elide, slots)
    emitter = _Emitter(blocks, start, end)
    body = emitter.emit(start, None, [], 1)
    # Blocks only reachable by a goto - from inside a loop, say - follow the main
    # body, each under its label.
    for node in emitter.order:
        if node not in emitter.emitted:
            body += emitter.emit(node, None, [], 1)
    # The registers the function names, gathered as statements become locals -
    # already, in the readable form, which localises before structuring.
    gprs, fprs, scalars = (set(x) for x in seen) if readable else (set(), set(), set())
    out_lines = []
    # A tail transfer can reach a shared epilogue that restores the saved registers
    # from the stack, so those functions keep their saves.
    elide = elide and not any("TailCall(c" in line for line in body)
    for line in body:
        if line.startswith("@L_"):
            address = int(line[3:], 16)
            if address in emitter.labels:
                # A loop header's label goes before its Tick: a goto back to the
                # header is another iteration, and has to spend budget like one.
                # After it, 0x0029CCC0 looped forever on the equivalence check.
                tick = "Tick(c, 0x%08Xu);" % address
                if out_lines and out_lines[-1].strip().endswith(tick):
                    out_lines.insert(len(out_lines) - 1, "L_%08X:;" % address)
                else:
                    out_lines.append("L_%08X:;" % address)
            continue
        localized = tidy(_localize(line, gprs, fprs, scalars))
        if elide:
            if _is_spill(localized, slots):
                continue
            localized = localized.replace("out(); return;", "leave(); return;")
        if _TAKES_CONTEXT.search(localized) and "out();" not in localized:
            indent = localized[:len(localized) - len(localized.lstrip())]
            localized = "%sout(); %s in();" % (indent, localized.strip())
        out_lines.append(localized)
    gprs.discard(0)
    names = [(GPR_NAMES[i], "c.r[%d]" % i) for i in sorted(gprs)]
    locals_only = [(GPR_NAMES[i], "c.r[%d]" % i) for i in sorted(gprs)
                   if i in CALLEE_SAVED and elide]
    names += [("f%d" % i, "c.f[%d]" % i) for i in sorted(fprs)]
    names += [(n, "c.%s" % n) for n in SCALARS if n in scalars]
    types = {"fcc": "u32", "acc": "u32", "sa": "u32"}
    lines = ["void %s(lifted::Context& c) {" % lift.name_of(start),
             "    using namespace generated;",
             "    using namespace lifted;",
             "    const u32 entry_ra = (u32)c.r[31], entry_sp = (u32)c.r[29];",
             "    (void)entry_ra; (void)entry_sp;"]
    for name, source in names:
        kind = "u32" if name.startswith("f") and name[1:].isdigit() else types.get(name, "u64")
        lines.append("    %s %s = %s;" % (kind, name, source))
    synced = [(n, src) for n, src in names if (n, src) not in locals_only]
    returned = [(n, src) for n, src in synced if n != "ra"]
    if readable:
        returned = [(n, src) for n, src in returned if n in ABI_RETURNED or n in ABI_PRESERVED]
    lines.append("    auto out = [&] { %s };" % " ".join("%s = %s;" % (s, n) for n, s in synced))
    # On return the caller sees its own $ra again, whatever links calls left in it.
    lines.append("    auto leave = [&] { %s c.r[31] = entry_ra; };"
                 % " ".join("%s = %s;" % (s, n) for n, s in returned))
    # The context is lifted::Context; everything else reads without the prefixes.
    lines.append("    auto in = [&] { %s };" % " ".join("%s = %s;" % (n, s) for n, s in synced))
    lines.append("    (void)out; (void)in; (void)leave;")
    body = drop_empty_else(hoist_delay_slots(fold_links(fold_addresses(fold_constants(out_lines, strings),
                                                                       strings))))
    lines += loop_guards(loops_to_do_while(invert_empty_then(body)))
    lines.append("}")
    return "\n".join(lines) + "\n"
