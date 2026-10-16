#!/usr/bin/env python3
"""Report which obstacles stand between each function and being derivable.

The refusal counts printed by `derive.py` record only the *first* obstacle hit in
each function, which makes them a poor guide to what is worth fixing: removing the
most common first obstacle repeatedly turned out to unlock nothing, because the
functions behind it were held up by two or three others as well.

This instead classifies every function by the *set* of obstacles it contains, found
by scanning its instructions rather than by running the deriver. The useful output
is the count of functions blocked by exactly one thing - those are the only ones a
single fix can actually reach.

    python engine/tools/blockers.py
    python engine/tools/blockers.py --single
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile
from ps2decomp.emu import is_supported
from ps2decomp.r5900 import DISASM

# Float argument registers under the o32 convention. Reading one before writing it
# means the function takes a float parameter, which the harness cannot yet supply.
FLOAT_ARG_REGS = {12, 13, 14}

# Instructions the symbolic model has no rule for, grouped so the report names a
# capability rather than an opcode.
VU_PREFIXES = ("v", "qmfc2", "qmtc2", "lqc2", "sqc2", "cop2")
MMI_NAMES = {
    "pcpyld", "pcpyud", "pcpyh", "pand", "por", "pxor", "pnor", "paddb", "psubb",
    "pmaxw", "pminw", "pextlw", "pextuw", "paddw", "psubw", "pmaddw", "pmulth",
}


def classify(func, image, base) -> set:
    """Return the set of obstacle labels present in one function."""
    count = func["size"] // 4
    obstacles = set()
    seen_float_write = set()
    start, end = func["addr"], func["end"]

    for index in range(count):
        addr = start + index * 4
        word = struct.unpack_from("<I", image, addr - base)[0]
        ins = DISASM.decode(word, addr)
        m = ins.mnemonic

        if ins.is_call:
            obstacles.add("calls")
        elif ins.is_jump and not ins.is_return and not (start <= ins.target < end):
            obstacles.add("tail-call")
        # A backward branch inside the function is a loop.
        if (ins.is_branch or ins.is_jump) and 0 <= ins.target < addr and ins.target >= start:
            obstacles.add("loop")
        if m == "jr" and not ins.is_return:
            obstacles.add("indirect-jump")

        if m.startswith(VU_PREFIXES):
            obstacles.add("vu0")
        elif m in MMI_NAMES:
            obstacles.add("mmi")
        elif m == "syscall":
            obstacles.add("syscall")
        elif m in ("multu", "div", "divu"):
            obstacles.add("unsigned-or-div")
        elif not is_supported(ins):
            obstacles.add("other:" + m)

        # Float argument use: a read of $f12..$f14 before anything wrote it.
        if m in ("swc1",) and ins.rt in FLOAT_ARG_REGS and ins.rt not in seen_float_write:
            obstacles.add("float-arg")
        if m in ("mov.s", "add.s", "sub.s", "mul.s", "div.s", "cvt.w.s"):
            fs = (ins.raw >> 11) & 0x1F
            ft = (ins.raw >> 16) & 0x1F
            if (fs in FLOAT_ARG_REGS and fs not in seen_float_write) or (
                ft in FLOAT_ARG_REGS and ft not in seen_float_write
            ):
                obstacles.add("float-arg")
        if m in ("lwc1", "mtc1"):
            seen_float_write.add(ins.rt if m == "lwc1" else ins.rd)

    if count > 96:
        obstacles.add("too-long")
    return obstacles


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--single", action="store_true",
                        help="list only functions blocked by exactly one obstacle")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)

    done = set()
    status_path = os.path.join("config", "functions.csv")
    if os.path.exists(status_path):
        import csv

        with open(status_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") in ("verified", "reconstructed"):
                    done.add(int(row["address"], 16))

    functions = [
        f for f in database["functions"]
        if f["has_return"] and f["size"] and f["addr"] not in done
    ]
    classified = [(f, classify(f, image, section.addr)) for f in functions]

    clear = [f for f, obstacles in classified if not obstacles]
    singles = collections.Counter()
    single_bytes = collections.Counter()
    appears = collections.Counter()
    for func, obstacles in classified:
        for label in obstacles:
            appears[label] += 1
        if len(obstacles) == 1:
            label = next(iter(obstacles))
            singles[label] += 1
            single_bytes[label] += func["size"]

    print("functions remaining        : %s" % format(len(functions), ","))
    print("no obstacle detected       : %s" % format(len(clear), ","))
    print()
    print("%-20s %10s %10s %12s" % ("obstacle", "appears in", "ONLY one", "bytes if only"))
    print("-" * 56)
    for label, total in appears.most_common(args.limit):
        print("%-20s %10s %10s %12s"
              % (label, format(total, ","), format(singles.get(label, 0), ","),
                 format(single_bytes.get(label, 0), ",")))
    print("-" * 56)
    print("%-20s %10s %10s" % ("", "", format(sum(singles.values()), ",")))
    print()
    print("The 'ONLY one' column is the one that matters: it is how many functions a")
    print("single capability would actually unlock. 'appears in' counts overlap and")
    print("is the number that has been misleading.")

    if args.single:
        print("\nfunctions blocked by exactly one obstacle, largest first:")
        rows = [
            (f, next(iter(o))) for f, o in classified if len(o) == 1
        ]
        rows.sort(key=lambda r: -len(r[0]["callers"]))
        for func, label in rows[: args.limit]:
            print("  0x%08X %6d bytes %4d callers  %s"
                  % (func["addr"], func["size"], len(func["callers"]), label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
