#!/usr/bin/env python3
"""Run the full static-analysis pipeline over a PS2 executable.

Produces a JSON database describing every function, region, string and
cross-reference the analysis could recover.  Downstream tools (the disassembler,
the progress report) read that database rather than re-deriving anything, so the
expensive passes run once.

    python engine/tools/analyze.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile
from ps2decomp.analysis import ImageAnalyzer
from ps2decomp import gaps, lift
from ps2decomp.r5900 import DISASM
from ps2decomp.symbols import SymbolRecovery


def build_database(elf_path: str, verbose: bool = True) -> dict:
    started = time.time()
    elf = ElfFile.load(elf_path)
    exec_sections = elf.exec_sections
    if not exec_sections:
        raise SystemExit("%s has no executable section" % elf_path)
    section = exec_sections[0]
    data = elf.section_data(section)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("image      : %s" % elf_path)
    log("load addr  : 0x%08x  size %s bytes" % (section.addr, format(len(data), ",")))
    log("entry      : 0x%08x" % elf.e_entry)

    log("discovering functions...")
    analyzer = ImageAnalyzer(data, section.addr, elf.e_entry)
    analyzer.run()
    log("  %s functions, %.1f%% of the image proven to be code"
        % (format(len(analyzer.functions), ","), analyzer.coverage()))

    log("recovering symbols...")
    symbols = SymbolRecovery(data, section.addr)
    string_count = symbols.find_strings()
    rw_tags = symbols.find_renderware_tags()
    code_ranges = [(f.start, f.end) for f in analyzer.functions.values()]
    xref_count = symbols.find_xrefs(code_ranges)

    # Functions the call graph cannot reach - virtual methods, stored callbacks,
    # tail-call targets - recovered from the gaps between those it found, where
    # there is evidence one starts and the code lifts (ps2decomp/gaps.py).
    def liftable(start, end):
        try:
            lift.lift_function(data, section.addr, start, end)
            return True
        except lift.Refused:
            return False

    recovered = gaps.recover(data, section.addr, code_ranges,
                             [x.target for x in symbols.xrefs], liftable)
    log("  %d more functions recovered from the gaps (%s bytes)"
        % (len(recovered), format(sum(e - s for s, e, _ev in recovered), ",")))
    tables = symbols.find_pointer_tables(
        list(analyzer.functions.keys()) + [s for s, _e, _ev in recovered])
    log("  %s strings, %s xrefs, %d pointer tables, %d RenderWare tags"
        % (format(string_count, ","), format(xref_count, ","), len(tables), len(rw_tags)))

    # Attach the most useful referenced string to each function as a naming hint.
    # Both sides are sorted and walked once, so this stays linear rather than
    # comparing every xref against every function.
    hints = {}
    ordered_funcs = sorted(analyzer.functions.values(), key=lambda f: f.start)
    ordered_xrefs = sorted(symbols.xrefs, key=lambda x: x.from_addr)
    cursor = 0
    for func in ordered_funcs:
        while cursor < len(ordered_xrefs) and ordered_xrefs[cursor].from_addr < func.start:
            cursor += 1
        best = None
        probe = cursor
        while probe < len(ordered_xrefs) and ordered_xrefs[probe].from_addr < func.end:
            entry = symbols.strings.get(ordered_xrefs[probe].target)
            if entry and len(entry.text) >= 4:
                if best is None or len(entry.text) > len(best):
                    best = entry.text
            probe += 1
        if best:
            hints[func.start] = best[:80]

    functions = []
    for addr in sorted(analyzer.functions):
        f = analyzer.functions[addr]
        functions.append(
            {
                "addr": f.start,
                "end": f.end,
                "size": f.size,
                "label": f.label,
                "confidence": f.confidence,
                "instructions": f.instruction_count,
                "has_return": f.has_return,
                "uses_vu": f.uses_vu,
                "callers": sorted(f.callers),
                "calls": sorted(f.calls),
                "string_hint": hints.get(f.start, ""),
                "status": "undecompiled",
            }
        )

    # The recovered functions, described as the analysis describes its own.
    by_address = {f["addr"]: f for f in functions}
    for start, end, evidence in recovered:
        words = [DISASM.decode(int.from_bytes(data[a - section.addr:a - section.addr + 4],
                                              "little"), a) for a in range(start, end, 4)]
        calls = sorted({w.target for w in words if w.is_call and w.target is not None})
        by_address[start] = {
            "addr": start,
            "end": end,
            "size": end - start,
            "label": "func_%08X" % start,
            "confidence": "recovered: %s" % evidence,
            "instructions": len(words),
            "has_return": any(w.is_return for w in words),
            "uses_vu": any(w.raw >> 26 in (0x12, 0x36, 0x3E) for w in words),
            "callers": [],
            "calls": calls,
            "string_hint": hints.get(start, ""),
            "status": "undecompiled",
        }
    for start, _end, _evidence in recovered:
        for target in by_address[start]["calls"]:
            if target in by_address:
                by_address[target]["callers"] = sorted(set(by_address[target]["callers"]) | {start})
    functions = [by_address[a] for a in sorted(by_address)]
    # Each function's out-of-line pieces (analysis.py `chunks`): translated with
    # it, so a branch to one stays inside the function.
    entries = {f["addr"] for f in functions if f["confidence"] in ("called", "entry")}
    # A "function" found by a pointer-shaped word or in a gap, that nothing
    # calls, and that a called function reaches by its own branches or by
    # falling through, is a label of that function: Burnout 3's 0x00186850, a
    # 54 KB state machine, had three of its cases taken for functions and was
    # cut at the first, its cases left out of every translation.
    labels = set()
    candidates = {f["addr"] for f in functions
                  if f["confidence"] not in ("called", "entry") and not f["callers"]}
    sources = ("called", "entry", "recovered: built", "recovered: call")
    for f in functions:
        if f["confidence"] in sources:
            reached = analyzer.closure(f["addr"], entries) or set()
            labels |= (reached & candidates) - {f["addr"]}
    if labels:
        functions = [f for f in functions if f["addr"] not in labels]
        log("  %d labels of other functions were taken for functions; dropped" % len(labels))
    for f in functions:
        f["chunks"] = [list(c) for c in analyzer.chunks(f["addr"], f["end"], entries)]
    log("  %d functions have out-of-line pieces (%s bytes)"
        % (sum(1 for f in functions if f["chunks"]),
           format(sum(e - s for f in functions for s, e in f["chunks"]), ",")))

    regions = [
        {"start": r.start, "end": r.end, "size": r.size, "kind": r.kind}
        for r in analyzer.regions()
    ]

    database = {
        "meta": {
            "source": os.path.basename(elf_path),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "analysis_seconds": round(time.time() - started, 2),
            "load_addr": section.addr,
            "image_size": len(data),
            "entry_point": elf.e_entry,
        },
        "stats": dict(analyzer.stats),
        "functions": functions,
        "regions": regions,
        "strings": [
            {"addr": s.addr, "text": s.text}
            for s in sorted(symbols.strings.values(), key=lambda s: s.addr)
        ],
        "xrefs": [
            {"from": x.from_addr, "to": x.target, "kind": x.kind} for x in symbols.xrefs
        ],
        "pointer_tables": [
            {"addr": addr, "count": count, "entries": entries}
            for addr, count, entries in tables
        ],
        "renderware": [{"addr": addr, "path": path} for addr, path in rw_tags],
    }
    log("done in %.1fs" % (time.time() - started))
    return database


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("-o", "--output", default=None,
                        help="the database to write (default: the game's)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    output = args.output or game.database_path
    os.makedirs(os.path.dirname(output), exist_ok=True)
    database = build_database(game.elf_path, verbose=not args.quiet)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(database, handle, indent=1)
    size = os.path.getsize(output)
    if not args.quiet:
        print("wrote %s (%s bytes)" % (output, format(size, ",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
