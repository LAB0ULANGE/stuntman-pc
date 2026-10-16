#!/usr/bin/env python3
"""Find the timing a VU microprogram depends on.

A VU's pipelines are exposed: Q and P arrive some cycles after DIV or an EFU
operation, the MAC and clip flags a few cycles after the FMAC operation that set
them, and a conditional branch reads an integer register before the bundle just
before it has written it. Code that waits (WAITQ, the stalls) does not care;
software-pipelined code does, and an emulator that makes results available at
once runs it wrong. This lists, for a dump of VU instruction memory, every place
a program reads one of those results early - so the emulator's model can be
checked against what the code actually needs.

    python engine/tools/vuhazards.py vu1_micro.bin --start 0x0 --start 0x3A0
    python engine/tools/vuhazards.py vu1_micro.bin            # every program in it

Programs are walked along their straight-line paths from each start and each
branch target, to the E bit; distances are in bundles (one cycle each, without
stalls).
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.vu import (LATENCY_FMAC, LATENCY_P, LATENCY_Q, branch_target,
                          decode_blob)

BRANCHES = ("ibeq", "ibne", "ibltz", "ibgtz", "iblez", "ibgez", "jr", "jalr")


def paths(bundles, starts):
    """Straight-line runs: from each start and branch target to the next
    unconditional transfer or the bundle after an E bit."""
    by_addr = {b.addr: i for i, b in enumerate(bundles)}
    seen, queue = set(), list(starts)
    while queue:
        start = queue.pop()
        if start in seen or start not in by_addr:
            continue
        seen.add(start)
        run, i, stop_after = [], by_addr[start], None
        while i < len(bundles):
            b = bundles[i]
            run.append(b)
            target = branch_target(b)
            if target is not None:
                queue.append(target)
            if stop_after is not None:
                break
            if b.is_end or b.lower.mnemonic in ("b", "jr"):
                stop_after = i          # its delay slot (or the pair after E) runs
            i += 1
        yield run


def scan(run, report):
    """One path: every early read of a delayed result."""
    last = {}                                   # resource -> (index, mnemonic)
    for i, b in enumerate(run):
        for half in (b.upper, b.lower):
            for r in half.reads:
                if r in ("Q", "P") and r in last:
                    j, op = last[r]
                    latency = (LATENCY_Q if r == "Q" else LATENCY_P).get(op, 0)
                    if op in ("waitq", "waitp"):
                        continue
                    if i - j < latency:
                        report["early %s (%s, %d after)" % (r, op, i - j)].append(b.addr)
                if r in ("MAC", "CLIP", "STATUS") and r in last:
                    j, op = last[r]
                    if i - j < LATENCY_FMAC:
                        report["early %s flag (%d after %s)" % (r, i - j, op.split(".")[0])].append(b.addr)
                if r.startswith("vf") and r in last:
                    j, op = last[r]
                    if 0 < i - j < LATENCY_FMAC and not op.startswith(("lq", "move", "mr32", "mfir", "mfp", "rget", "rnext")):
                        report["stall: vf read %d after an FMAC write" % (i - j)].append(b.addr)
        if b.lower.mnemonic in BRANCHES and i > 0:
            prev = run[i - 1].lower
            hit = b.lower.reads & prev.writes
            if hit and run[i - 1].immediate is None:
                report["branch reads %s written by the bundle before (%s)"
                       % (sorted(hit)[0][:2], prev.mnemonic.split(".")[0])].append(b.addr)
        for half in (b.lower, b.upper):
            for w in half.writes:
                last[w] = (i, half.mnemonic)
        for r in ("Q", "P"):
            if b.lower.mnemonic == "wait" + r.lower():
                last[r] = (i, b.lower.mnemonic)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", help="VU instruction memory, as the host writes vu1_micro.bin")
    parser.add_argument("--start", action="append", default=[], help="a program's start (bytes)")
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args(argv)
    data = open(args.dump, "rb").read()
    bundles = decode_blob(data, 0)
    starts = [int(s, 0) for s in args.start]
    if not starts:
        starts = [0] + [b.addr + 16 for b in bundles if b.is_end and b.addr + 16 < len(data)]
    report = collections.defaultdict(list)
    for run in paths(bundles, starts):
        scan(run, report)
    for kind, where in sorted(report.items(), key=lambda kv: -len(kv[1])):
        places = sorted(set(where))
        print("%5d  %s: %s" % (len(places), kind, " ".join("0x%X" % a for a in places[:args.examples])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
