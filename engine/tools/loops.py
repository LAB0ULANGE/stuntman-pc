#!/usr/bin/env python3
"""Measure how many looping functions have a constant trip count.

Loops are the largest remaining obstacle to automatic derivation, and the right
response depends entirely on a number nobody has measured: how many of them run a
fixed number of times regardless of input. A loop with a constant trip count can be
unrolled and derived exactly. One whose count depends on its arguments cannot be
described by any finite expression, so it needs hand reconstruction.

The measurement is empirical rather than static, because the interpreter already
exists and executing the loop answers the question directly. Each function is run
several times with different arguments *and* different memory contents, counting how
often each loop header is reached. A trip count that never varies across those runs
is evidence of a constant one.

Two honest limits on that evidence:

* It is evidence, not proof. A loop whose count depends on an input the trials never
  happen to vary would look constant.
* Memory is varied as well as arguments, which matters: a count driven by a global
  rather than a parameter would otherwise be misread as constant.

    python engine/tools/loops.py
    python engine/tools/loops.py --list --limit 20
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp import layout
from ps2decomp.elf import ElfFile
from ps2decomp.emu import EmulationError, Machine
from ps2decomp.r5900 import DISASM
from ps2decomp import testmem

TRIALS = 6
INSTRUCTION_BUDGET = 20_000


def loop_headers(func, image, base) -> list:
    """Addresses branched *backwards* to - the entry of each loop in the function."""
    headers = set()
    start, end = func["addr"], func["end"]
    for index in range(func["size"] // 4):
        addr = start + index * 4
        word = struct.unpack_from("<I", image, addr - base)[0]
        ins = DISASM.decode(word, addr)
        if (ins.is_branch or ins.is_jump) and start <= ins.target <= addr:
            headers.add(ins.target)
    return sorted(headers)


def trip_counts(machine, func, headers, trial, profile="noise") -> dict:
    """Run one trial and report how often each loop header was reached."""
    rng = random.Random((func["addr"] << 8) ^ trial)
    # Fresh memory contents each trial, so a count driven by a global shows up as
    # varying rather than looking constant. The *shape* of that memory comes from a
    # profile chosen per function - see ps2decomp/testmem.py for why random bytes alone
    # left most looping functions untestable.
    machine.memory.begin_journal()
    machine.memory.write(layout.CURRENT.scratch,
                         testmem.build(profile, (func["addr"] << 4) + trial))
    args = testmem.arguments(rng)

    machine.visit_counts = {}
    machine.count_visits = True
    try:
        machine.call(func["addr"], args, max_instructions=INSTRUCTION_BUDGET)
    finally:
        machine.count_visits = False
        machine.memory.rollback()
        machine.memory.end_journal()
    return {header: machine.visit_counts.get(header, 0) for header in headers}


def classify(machine, func, headers) -> tuple:
    """Return (verdict, counts) for one function."""
    # Pick the first memory shape under which the function runs at all, then hold it
    # fixed across trials. Mixing shapes between trials would make a count that
    # depends on memory *shape* look like a count that depends on memory *contents*.
    profile = None
    for candidate in testmem.PROFILES:
        try:
            first_trial = trip_counts(machine, func, headers, 0, candidate)
        except Exception:
            continue
        profile = candidate
        break
    if profile is None:
        return "untestable", None

    observed = [first_trial]
    for trial in range(1, TRIALS):
        try:
            observed.append(trip_counts(machine, func, headers, trial, profile))
        except EmulationError:
            return "untestable", None
        except Exception:
            return "untestable", None
    if not observed:
        return "untestable", None
    first = observed[0]
    if any(sample != first for sample in observed[1:]):
        return "variable", observed
    if all(count <= 1 for count in first.values()):
        # The loop never actually went round, so this trial says nothing about it.
        return "not-entered", first
    return "constant", first


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--list", action="store_true",
                        help="list the constant-trip-count functions")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=400,
                        help="skip functions larger than this many bytes")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    machine = Machine.from_elf(elf)

    looping = []
    for func in database["functions"]:
        if not func["has_return"] or not func["size"]:
            continue
        if func["size"] > args.max_size:
            continue
        headers = loop_headers(func, image, section.addr)
        if headers:
            looping.append((func, headers))

    verdicts = collections.Counter()
    sizes = collections.Counter()
    constant = []
    for func, headers in looping:
        verdict, counts = classify(machine, func, headers)
        verdicts[verdict] += 1
        sizes[verdict] += func["size"]
        if verdict == "constant":
            constant.append((func, headers, counts))

    print("looping functions examined : %s  (size <= %d bytes)"
          % (format(len(looping), ","), args.max_size))
    print()
    print("%-14s %8s %12s" % ("verdict", "count", "bytes"))
    print("-" * 36)
    for verdict in ("constant", "variable", "not-entered", "untestable"):
        print("%-14s %8s %12s"
              % (verdict, format(verdicts[verdict], ","), format(sizes[verdict], ",")))
    print("-" * 36)
    print()
    print("`constant` means the trip count was identical across %d trials with both"
          % TRIALS)
    print("arguments and memory varied. That is evidence, not proof: a count driven by")
    print("an input the trials never varied would look constant.")
    print("`not-entered` means the loop body never ran, so the trials say nothing.")

    if args.list and constant:
        print("\nconstant-trip-count functions, most-called first:")
        constant.sort(key=lambda row: -len(row[0]["callers"]))
        for func, headers, counts in constant[: args.limit]:
            detail = ", ".join("0x%08X x%d" % (h, counts[h]) for h in headers)
            print("  0x%08X %5d bytes %4d callers  %s"
                  % (func["addr"], func["size"], len(func["callers"]), detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
