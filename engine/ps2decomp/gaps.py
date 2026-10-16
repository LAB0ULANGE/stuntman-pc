"""Functions the call-graph analysis cannot reach, recovered from the gaps.

The analysis (ImageAnalyzer) finds functions by following code from the entry
point and through pointer tables it recognises. A function reached only some
other way - a C++ virtual method, a callback whose address is built in code and
stored, a tail-call target - is never found, and its bytes are left in a gap
between known functions. The native port meets them at run time; decompilation
never sees them at all.

A gap function is recovered only where there is evidence one starts:

* `call`     - a jal or j targets the address;
* `built`    - code builds the address in a register (the analysis's xrefs);
* `pointer`  - a word somewhere in the image holds the address;
* `adjacent` - it is the first instruction after a known function and its
               padding (the weakest: compilers lay functions out end to end).

and only where the code there reaches a return before the next known function
and lifts - which data misread as code almost never does. Each recovered
function is then verified like any other; one that is not really a function
fails there, not silently.
"""

from __future__ import annotations

import bisect
import struct

from .r5900 import DISASM

# How far a recovered function may extend, at most.
MAX_FUNCTION = 0x4000


def function_extent(code: bytes, base: int, start: int, limit: int = None) -> int:
    """Where a function starting at `start` ends: after the first return - or tail
    jump, a `j` out of it, or a switch's `jr` - that no branch inside it jumps past. Code compiled from
    C keeps its body between its entry and that exit. `limit` is where the next
    known function starts; nothing reaches past it. Raises ValueError when there
    is no such exit before it."""
    limit = min(limit if limit is not None else base + len(code), base + len(code),
                start + MAX_FUNCTION)
    furthest = start
    for at in range(start, limit - 7, 4):
        ins = DISASM.decode(struct.unpack_from("<I", code, at - base)[0], at)
        within = ins.target is not None and start <= ins.target < limit
        # Branches, `b` and `j` within the function; a call goes elsewhere.
        if (ins.is_branch or (ins.is_jump and not ins.is_call)) and within:
            furthest = max(furthest, ins.target)
        tail_jump = ins.mnemonic == "j" and ins.target is not None and not within
        # A switch's `jr` ends the body too when its cases lie past it (Burnout
        # 3's 0x002A6B80, a leaf that is all switch, was refused for want of a
        # `jr $ra`); they are its own (analysis.closure).
        switch = ins.mnemonic == "jr" and ins.rs != 31
        if (ins.is_return or tail_jump or switch) and at >= furthest:
            return at + 8 - start
    raise ValueError("no return found after 0x%08X" % start)


def recover(image: bytes, base: int, functions, xref_targets=(), accept=None) -> list:
    """[(start, end, evidence)] for the functions in the gaps between `functions`
    ((start, end) pairs), found from the evidence above. `accept(start, end)`, if
    given, decides whether the code there can be a function (lifting it, say)."""
    known = sorted((s, e) for s, e in functions if e > s)
    end_of_image = base + len(image)
    words = struct.unpack_from("<%dI" % (len(image) // 4), image, 0)

    def gap_at(address):
        """(gap start, gap end) holding `address`, or None if a function covers it."""
        starts = [s for s, _e in known]
        i = bisect.bisect_right(starts, address) - 1
        if i >= 0 and known[i][0] <= address < known[i][1]:
            return None
        low = known[i][1] if i >= 0 else base
        high = known[i + 1][0] if i + 1 < len(known) else end_of_image
        return (low, high) if low <= address < high else None

    def calls_from(start, end):
        for at in range(start, end, 4):
            ins = DISASM.decode(words[(at - base) // 4], at)
            if (ins.is_call or (ins.is_jump and ins.mnemonic == "j")) and ins.target is not None:
                yield ins.target

    candidates = {}
    for start, end in known:
        for target in calls_from(start, end):
            candidates.setdefault(target, "call")
    for target in xref_targets:
        candidates.setdefault(target, "built")
    for value in words:
        if base <= value < end_of_image and not value & 3:
            candidates.setdefault(value, "pointer")
    for _start, end in known:
        at = end
        while at < end_of_image and words[(at - base) // 4] == 0:
            at += 4
        candidates.setdefault(at, "adjacent")

    recovered = []
    pending = sorted(candidates.items())
    while pending:
        found = []
        for start, evidence in pending:
            if start & 3 or not base <= start < end_of_image:
                continue
            gap = gap_at(start)
            if gap is None:
                continue
            try:
                length = function_extent(image, base, start, gap[1])
            except ValueError:
                continue
            if accept is not None and not accept(start, start + length):
                continue
            found.append((start, start + length, evidence))
            known.append((start, start + length))
            known.sort()
        recovered += found
        # Recovered functions call others the analysis did not find either.
        pending = sorted({(t, "call") for s, e, _ev in found for t in calls_from(s, e)})
    return sorted(recovered)
