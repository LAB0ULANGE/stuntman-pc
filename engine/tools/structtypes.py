#!/usr/bin/env python3
"""Recover struct types across the whole binary and report on them.

    python engine/tools/structtypes.py
    python engine/tools/structtypes.py --show S_0011FE00_a0

The recovery itself lives in tools/ps2decomp/structs.py; derive.py uses it to type
the code it emits. This reports how much it found and how trustworthy it looks.
The join guard compares field widths, so width agreement says nothing; the two
independent checks are whether an offset shared by several functions is used as
float *or* integer, and loaded signed *or* unsigned, consistently.
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
from ps2decomp import structs
from ps2decomp.elf import ElfFile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--show", help="print the layout of the named type")
    parser.add_argument("--largest", type=int, default=8)
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]

    started = time.time()
    graph = structs.recover(database, elf.section_data(section), section.addr)
    catalog = structs.Catalog(graph)
    elapsed = time.time() - started

    if args.show:
        by_name = {name: root for root, name in catalog.names.items()}
        if args.show.startswith("g_") and catalog.globals is not None:
            address = int(args.show[2:10], 16)
            field = catalog.roots[catalog.globals].get(address)
            target = catalog.pointee(catalog.globals, address)
            if field is None:
                print("no global at 0x%08X" % address)
                return 1
            print("%s: read %s, write %s, %s, %d function(s)%s"
                  % (args.show, catalog.read_flavour(field), catalog.write_width(field),
                     catalog.kind(field), len(field.functions),
                     ("; -> " + catalog.names[target]) if target is not None else ""))
            return 0
        root = by_name.get(args.show)
        if root is None:
            print("no type named %s" % args.show)
            return 1
        print("%s: %d function(s)" % (args.show, len(catalog.functions(root))))
        for offset in sorted(catalog.roots[root]):
            field = catalog.roots[root][offset]
            target = catalog.pointee(root, offset)
            print("  %+6X  read %-10s write %-4s %-6s %3d fn  %s"
                  % (offset, catalog.read_flavour(field), catalog.write_width(field),
                     catalog.kind(field), len(field.functions),
                     ("-> " + catalog.names[target]) if target is not None else ""))
        for offset, inner in sorted(catalog.embedded(root).items()):
            print("  %+6X  embedded %s" % (offset, catalog.names[inner]))
        return 0

    ordinary = [root for root in catalog.roots if root != catalog.globals]
    sizes = sorted((len(catalog.functions(root)) for root in ordinary), reverse=True)
    pointers = sum(1 for root in catalog.roots for offset in catalog.roots[root]
                   if catalog.pointee(root, offset) is not None)
    embeds = sum(len(catalog.embedded(root)) for root in catalog.roots)
    quality = structs.quality(graph)
    print("recovered in %.1fs" % elapsed)
    print("types with at least one field         : %s" % format(len(ordinary), ","))
    print("  ...used by two or more functions    : %s"
          % format(sum(1 for size in sizes if size >= 2), ","))
    print("pointer fields (to a recovered type)  : %s" % format(pointers, ","))
    print("embedded types                        : %s" % format(embeds, ","))
    print("joins made / refused by the guard     : %s / %s"
          % (format(graph.joined, ","), format(graph.refused, ",")))
    print("functions per type, largest           : %s" % sizes[:args.largest])
    if catalog.globals is not None:
        cells = catalog.roots[catalog.globals]
        print("global variables (distinct addresses) : %s" % format(len(cells), ","))
        print("  ...holding a pointer to a type      : %s" % format(
            sum(1 for a in cells if catalog.pointee(catalog.globals, a) is not None), ","))
        print("  ...global objects of a known type   : %s"
              % format(len(catalog.embedded(catalog.globals)), ","))
    print("offsets used by 2+ functions          : %s" % format(quality["shared offsets"], ","))
    print("  ...used consistently as float or int: %.1f%%"
          % (100 * quality["float/int agreement"]))
    print("  ...loaded with consistent signedness: %.1f%%"
          % (100 * quality["signedness agreement"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
