#!/usr/bin/env python3
"""Why the deriver refuses each remaining function, by its own account.

engine/tools/blockers.py classifies functions by scanning for instructions, which made it
a good guide while calls and loops were outright obstacles. They no longer are -
the deriver inlines derived callees and unrolls loops with constant trip counts -
so a static scan now overstates both. This asks the deriver itself: after deriving
to a fixed point, every remaining function is attempted once more and the reason
it is refused is recorded.

A refusal records only the first obstacle the deriver meets, so the counts are an
ordering of what to look at, not a promise of what a fix unlocks. The call
section compensates in one respect: it shows which underived callees block the
most functions, and why each of those callees is itself refused - a callee that
would unlock fifty callers is worth more than any category.

    python engine/tools/refusals.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import derive
from ps2decomp import game as game_config
from ps2decomp import calls
from ps2decomp.elf import ElfFile
from ps2decomp.emu import Machine
from ps2decomp.paths import derive_paths
from ps2decomp.symexec import Unsupported

CALLEE = re.compile(r"0x([0-9A-F]{8})$")


def category(message: str) -> str:
    """A refusal message with its specifics removed, so like counts with like."""
    message = CALLEE.sub("", message).rstrip(": ")
    return re.sub(r"\$\d+", "$n", re.sub(r"\b\d+\b", "N", message))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--detail", help="also list exact messages for this category")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="processes deriving in parallel, as derive.py")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    machine = Machine.from_elf(elf)

    started = time.time()
    if args.jobs > 1:
        verified, _rejected, _machine = derive.collect_parallel(
            database, image, section.addr, machine, game.directory, args.jobs, verbose=False)
    else:
        verified, _rejected = derive.collect(database, image, section.addr, machine,
                                             verbose=False)
    known = {func["addr"]: derivation for func, derivation, _t, _s in verified
             if not derivation.sequential}
    done = {func["addr"] for func, _d, _t, _s in verified}
    parameters = calls.parameters(database, image, section.addr)
    by_address = {f["addr"]: f for f in database["functions"]}

    reasons = {}
    for func in database["functions"]:
        if not func["size"] or not func["has_return"] or func["addr"] in done:
            continue
        long = func["size"] // 4 > derive.MAX_INSTRUCTIONS
        try:
            derive.derive_function(image, section.addr, func, known, parameters)
            # Derivable, yet not verified: too long for the deriver's size limit, or
            # its test could not run or did not agree.
            reasons[func["addr"]] = ("derives, but over the size limit" if long
                                     else "derives, but untestable or disagreed")
        except Unsupported as exc:
            reasons[func["addr"]] = str(exc)
        except Exception as exc:     # a crash is a finding too
            reasons[func["addr"]] = "deriver error: %s" % type(exc).__name__

    counts = collections.Counter(category(r) for r in reasons.values())
    sizes = collections.Counter()
    for address, reason in reasons.items():
        sizes[category(reason)] += by_address[address]["size"]
    print("derived and verified : %d   remaining with a return: %s   (%.0fs)"
          % (len(done), format(len(reasons), ","), time.time() - started))
    print()
    print("%-58s %7s %10s" % ("first obstacle the deriver meets", "funcs", "bytes"))
    print("-" * 77)
    for label, count in counts.most_common(args.limit):
        print("%-58s %7s %10s" % (label[:58], format(count, ","), format(sizes[label], ",")))

    if args.detail:
        exact = collections.Counter(r for r in reasons.values() if category(r) == args.detail)
        print()
        print("exact messages for %r:" % args.detail)
        for message, count in exact.most_common(args.limit):
            print("%7d  %s" % (count, message))

    # Calls: which underived callees block the most callers, and why.
    blocking = collections.Counter()
    for reason in reasons.values():
        match = CALLEE.search(reason)
        if match:
            blocking[int(match.group(1), 16)] += 1
    print()
    print("underived callees blocking the most callers (first obstacle only):")
    print("%-12s %8s  %s" % ("callee", "blocks", "why the callee itself is refused"))
    for callee, count in blocking.most_common(args.limit):
        own = reasons.get(callee)
        if own is None:
            own = ("no return / not a function start" if callee not in by_address
                   else "not attempted (no return or empty)")
        print("0x%08X   %8d  %s" % (callee, count, own[:80]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
