#!/usr/bin/env python3
"""Rank functions by how easy they are to reconstruct and verify.

With differential testing in place, the bottleneck is no longer verification but
choosing what to work on.  A good first target is small, is a leaf (so its
behaviour does not depend on reconstructing something else first), executes inside
the interpreter's instruction subset, and has many callers so the effort pays off
more than once.

This scores every discovered function on those axes and prints the best
candidates.  It is a prioritisation aid, not a claim that any of them is
understood.

    python engine/tools/candidates.py
    python engine/tools/candidates.py --leaf-only --limit 40
    python engine/tools/candidates.py --pure
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile
from ps2decomp.emu import is_supported
from ps2decomp.r5900 import DISASM, Flags


def analyse(database: dict, image: bytes, base: int) -> list:
    rows = []
    for func in database["functions"]:
        count = func["size"] // 4
        if count == 0:
            continue
        unsupported = set()
        loads = stores = branches = 0
        reads_global = False
        tail_calls = 0
        for i in range(count):
            addr = func["addr"] + i * 4
            word = struct.unpack_from("<I", image, addr - base)[0]
            ins = DISASM.decode(word, addr)
            if not is_supported(ins):
                unsupported.add(ins.mnemonic)
            if ins.flags & Flags.LOAD:
                loads += 1
            if ins.flags & Flags.STORE:
                stores += 1
            if ins.is_branch:
                branches += 1
            # A materialised constant address means the function touches a global,
            # so its behaviour is not a pure function of its arguments.
            if ins.mnemonic == "lui":
                reads_global = True
            # A tail call via `j` into another function is still a dependency, but
            # the database only records `jal` targets as calls - so counting only
            # those makes a tail-call thunk look like a leaf.
            if ins.mnemonic == "j" and not (func["addr"] <= ins.target < func["end"]):
                tail_calls += 1

        rows.append(
            {
                "addr": func["addr"],
                "size": func["size"],
                "instructions": count,
                "callers": len(func["callers"]),
                "calls": len(func["calls"]),
                "tail_calls": tail_calls,
                "is_leaf": not func["calls"] and tail_calls == 0,
                "executable": not unsupported,
                "unsupported": sorted(unsupported),
                "loads": loads,
                "stores": stores,
                "branches": branches,
                "writes_memory": stores > 0,
                "touches_global": reads_global,
                "has_return": func["has_return"],
                "hint": func["string_hint"],
            }
        )
    return rows


def score(row: dict) -> float:
    """Higher is a better next target.

    Executability is a hard requirement - an unexecutable function cannot be
    differential-tested at all, so it scores zero however attractive it looks.
    """
    if not row["executable"] or not row["has_return"]:
        return 0.0
    value = 1.0
    value *= 2.0 if row["is_leaf"] else 1.0
    # Small functions are quicker to understand and leave less room for a wrong
    # reading to hide.
    value *= 64.0 / max(16.0, row["size"])
    # Callers are a proxy for how much the rest of the binary depends on it.
    value *= 1.0 + min(row["callers"], 200) / 25.0
    # A function that only reads its arguments is far easier to test than one
    # whose result depends on global state.
    if not row["touches_global"]:
        value *= 1.5
    if not row["writes_memory"]:
        value *= 1.3
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--leaf-only", action="store_true")
    parser.add_argument("--pure", action="store_true",
                        help="only functions that touch no global and write no memory")
    parser.add_argument("--status", default="config/functions.csv")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)

    done = set()
    if os.path.exists(args.status):
        import csv

        with open(args.status, newline="", encoding="utf-8") as handle:
            for entry in csv.DictReader(handle):
                if entry.get("status") in ("verified", "reconstructed"):
                    done.add(int(entry["address"], 16))

    rows = analyse(database, image, section.addr)
    executable = sum(1 for r in rows if r["executable"])
    print("functions                      : %s" % format(len(rows), ","))
    print("executable in the interpreter   : %s (%.1f%%)"
          % (format(executable, ","), executable / len(rows) * 100))
    print("already reconstructed or verified: %s" % format(len(done), ","))

    candidates = [r for r in rows if r["addr"] not in done and score(r) > 0]
    if args.leaf_only:
        candidates = [r for r in candidates if r["is_leaf"]]
    if args.pure:
        candidates = [
            r for r in candidates if not r["touches_global"] and not r["writes_memory"]
        ]
    candidates.sort(key=score, reverse=True)

    print("\ntop %d candidates:" % args.limit)
    print("  %-12s %6s %5s %7s %5s %5s %6s %6s  %s"
          % ("addr", "bytes", "ins", "callers", "leaf", "glob", "writes", "score", "hint"))
    for row in candidates[: args.limit]:
        print("  0x%08X %6d %5d %7d %5s %5s %6s %6.1f  %s"
              % (row["addr"], row["size"], row["instructions"], row["callers"],
                 "yes" if row["is_leaf"] else "no",
                 "yes" if row["touches_global"] else "no",
                 "yes" if row["writes_memory"] else "no",
                 score(row), row["hint"][:28]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
