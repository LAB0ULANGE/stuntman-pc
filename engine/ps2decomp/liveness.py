"""Register liveness over structured code, and folding of single-use temporaries.

Structured code (structure.py) moves values through registers one instruction at
a time: `v0 = ReadS32(a0 + 0x20); v0 = (a1 < v0); if (v0 != 0)`. A temporary that
is defined, used once, and then dead can be folded into its use - `if ((a1 <
ReadS32(a0 + 0x20)) != 0)` - which is most of what makes code read as source.

This works on blocks whose statements are already in local-variable form. It is
deliberately conservative, and every function it changes is verified again:

* A statement it does not understand - anything compound, any call, any write -
  counts as using every register it names, and as a barrier nothing moves across.
* A definition folds only into the one place it is used, in the same block, with
  none of its operands redefined in between, and - if it reads memory - with no
  write, call or compound statement in between.
* What a block leaves live comes from a backward dataflow over the CFG, where a
  return uses only the registers the calling convention gives back.
"""

from __future__ import annotations

import re

REGISTER = r"(?:at|v[01]|a[0-3]|t[0-9]|s[0-7]|k[01]|gp|sp|fp|ra|f\d{1,2}|hi1?|lo1?|acc|fcc|sa)"
_NAME = re.compile(r"\b%s\b" % REGISTER)
_SIMPLE = re.compile(r"^\s*(%s) = (.+);$" % REGISTER)
# Things that touch memory or the context, or do anything else a temporary cannot
# be moved across.
_BARRIER = re.compile(r"\bWrite\d+\(|\bCall\(|\bTailCall\(|\bout\(\)|\bin\(\)|\bleave\(\)|"
                      r"\bSyscall\(|\bCop2|\bTick\(|\bTrap\(|\bc\.|\{|\}|\breturn\b|\bgoto\b|"
                      r"\bStore(Left|Right)\(")


def names(text: str) -> set:
    return set(_NAME.findall(text))


def simple(statement: str):
    """(target, expression) for `x = expr;` with a side-effect-free expression."""
    match = _SIMPLE.match(statement)
    if not match or _BARRIER.search(match.group(2)):
        return None
    return match.group(1), match.group(2)


def uses_defs(statement: str, synced: set, returned: set):
    """The registers a statement reads and writes, over-approximated when unsure."""
    if "leave();" in statement:
        return set(returned), set()
    if "TailCall(" in statement:
        return set(synced), set()
    if "out();" in statement or "in();" in statement:
        # A call: it may read every synchronised register and change them all.
        return set(synced) | names(statement), set(synced)
    if re.match(r"^\s*if \(", statement):
        # A conditional assignment (movn, movz): when the condition fails the
        # target keeps its old value, so it is read, and not certainly written.
        # Counting it as a definition once folded `s1 = s1 << 1` into the
        # conditional branch only, and software double multiplication looped.
        return names(statement), set()
    target = simple(statement)
    if target is not None:
        return names(target[1]), {target[0]}
    assigned = set(re.findall(r"\b(%s) = " % REGISTER, statement))
    return names(statement), assigned


def live_out(blocks, succs, synced, returned, condition_of, slot_of):
    """{block: registers live after it}, by backward dataflow to a fixed point."""
    use, define, leaving = {}, {}, {}
    for address, block in blocks.items():
        # A conditional branch out of the function is a tail transfer on its
        # taken path - often into the middle of this same function, cut short by
        # dead code - so whatever is synchronised is live after it. Missing this
        # folded `s3 = 5` into `if (v1 == s3)` at 0x002A5D40 and lost the 5 the
        # branch target returns, which held Burnout 3's music loading at state 4.
        taken = getattr(block, "taken", None)
        leaving[address] = set(synced) if taken is not None and taken not in blocks else set()
        u, d = set(), set()
        sequence = list(block.code)
        if condition_of(block):
            sequence.append("if (%s)" % condition_of(block))
        if slot_of(block):
            sequence.append(slot_of(block))
        if block.kind == "return":
            sequence.append("leave(); return;")
        elif block.kind == "tail" or (block.next is not None and block.next not in blocks
                                      and block.kind in ("fall", "jump", "cond")):
            sequence.append("out(); TailCall(c);")
        for statement in sequence:
            s_use, s_def = uses_defs(statement, synced, returned)
            u |= s_use - d
            d |= s_def
        use[address], define[address] = u, d
    live_in = {a: set(use[a]) for a in blocks}
    out = {a: set() for a in blocks}
    changed = True
    while changed:
        changed = False
        for address in blocks:
            new_out = set(leaving[address])
            for s in succs.get(address, ()):
                new_out |= live_in.get(s, set())
            if new_out != out[address]:
                out[address] = new_out
                changed = True
            new_in = use[address] | (new_out - define[address])
            if new_in != live_in[address]:
                live_in[address] = new_in
                changed = True
    return out


def fold(code: list, condition: str, live_after: set, synced: set, returned: set,
         later=()):
    """Fold single-use definitions within one block; returns (code, condition).

    A definition `x = e;` folds into its one use when:
    * the use names `x` in its text - a call reads registers through `out()`
      without naming them, which counts as a use that cannot take a fold;
    * the use is not a compound statement, whose own `lo`/`hi` temporaries would
      capture names in `e`;
    * nothing in between redefines an operand of `e`, or - when `e` reads memory -
      writes memory, calls, or does anything else it cannot be moved across;
    * `x` is dead after the use: redefined later in the block, or not live out.
    """
    code = list(code)
    changed = True
    while changed:
        changed = False
        texts = code + ([condition] if condition else [])
        for i in range(len(code)):
            definition = simple(code[i])
            if definition is None:
                continue
            target, expression = definition
            pattern = re.compile(r"\b%s\b(?!\s*=[^=])" % re.escape(target))
            operands = names(expression)
            reads_memory = "Read" in expression
            use_at, dead = None, False
            ok = True
            for j in range(i + 1, len(texts)):
                text = texts[j]
                textual = len(pattern.findall(text))
                s_use, s_def = uses_defs(text, synced, returned)
                if target in s_use or textual:
                    if use_at is not None or textual != 1 or "{" in text:
                        ok = False         # a second use, a hidden one, or a compound
                        break
                    use_at = j
                    if target in s_def:
                        dead = True        # the use redefines it: nothing after sees it
                        break
                    continue
                if target in s_def:
                    dead = True
                    break
            if ok and any(re.search(r"\b%s\b" % re.escape(target), text) for text in later):
                ok = False                 # the delay slot reads it after the condition
            if not ok or use_at is None:
                continue
            if not dead and target in live_after:
                continue
            for k in range(i + 1, use_at):
                _u, s_def = uses_defs(texts[k], synced, returned)
                if s_def & operands or (reads_memory and _BARRIER.search(texts[k])):
                    ok = False
                    break
            if not ok:
                continue
            replacement = "(%s)" % expression
            updated = pattern.sub(lambda _m: replacement, texts[use_at], count=1)
            if condition and use_at == len(code):
                condition = updated
            else:
                code[use_at] = updated
            del code[i]
            changed = True
            break
    return code, condition
