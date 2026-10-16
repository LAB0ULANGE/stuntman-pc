#!/usr/bin/env python3
"""Disassemble PlayStation 2 Vector Unit microcode.

Reads either a `.vutext` section from an ELF or a raw address range out of a
stripped image, and prints the paired upper/lower bundles.

    python engine/tools/vudis.py extracted/NFSUNDER/NFSUNDER.ELF --section .vutext --stats
    python engine/tools/vudis.py games/burnout3/extracted/SLUS_210.50 --range 0x00469D90 0x004843D0 --stats
    python engine/tools/vudis.py games/burnout3/extracted/SLUS_210.50 --range 0x00469E00 0x00469F00
    python engine/tools/vudis.py games/burnout3/extracted/SLUS_210.50 --range 0x00469D90 0x004843D0 --programs
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.elf import ElfFile
from ps2decomp.vu import coverage, decode_blob


def load_blob(args):
    elf = ElfFile.load(args.image)
    if args.section:
        section = elf.section(args.section)
        if section is None:
            names = sorted(s.name for s in elf.sections if s.name)
            raise SystemExit(
                "no section %r; image has: %s" % (args.section, ", ".join(names))
            )
        return elf.section_data(section), section.addr
    if args.range:
        low, high = (int(v, 0) for v in args.range)
        section = elf.section_at(low) or elf.exec_sections[0]
        start = low - section.addr
        return elf.section_data(section)[start : start + (high - low)], low
    raise SystemExit("supply either --section or --range")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("--section", help="named section to read, e.g. .vutext")
    parser.add_argument("--range", nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--stats", action="store_true", help="report decode coverage only")
    parser.add_argument("--programs", action="store_true",
                        help="list microprogram boundaries from the E flag")
    parser.add_argument("--skip-padding", action="store_true",
                        help="omit all-zero bundles from the listing")
    args = parser.parse_args(argv)

    blob, base = load_blob(args)
    bundles = decode_blob(blob, base)
    stats = coverage(bundles)

    if args.stats:
        print("bundles          : %s" % format(stats["bundles"], ","))
        print("all-zero padding : %s" % format(stats["padding"], ","))
        print("real bundles     : %s" % format(stats["real"], ","))
        print("upper decoded    : %s (%.1f%%)" % (format(stats["upper_known"], ","), stats["upper_pct"]))
        print("lower decoded    : %s (%.1f%%)" % (format(stats["lower_known"], ","), stats["lower_pct"]))
        print("both decoded     : %s" % format(stats["both_known"], ","))
        print("microprograms    : %s (counted from E flags)" % format(stats["programs"], ","))
        return 0

    if args.programs:
        start = base
        index = 0
        for bundle in bundles:
            if bundle.is_end:
                index += 1
                print("program %3d: 0x%08X..0x%08X  (%s bundles)"
                      % (index, start, bundle.addr + 8,
                         format((bundle.addr + 8 - start) // 8, ",")))
                start = bundle.addr + 8
        print("\n%d microprograms" % index)
        return 0

    for bundle in bundles:
        if args.skip_padding and not (bundle.lower_raw or bundle.upper_raw):
            continue
        print(bundle.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
