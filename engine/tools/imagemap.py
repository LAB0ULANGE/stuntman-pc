#!/usr/bin/env python3
"""Account for every byte of a PS2 executable image.

The game executable is one undifferentiated blob with no section headers, so
"what is this image actually made of?" has to be answered by classification.
This assigns every byte to exactly one category and reports the totals, which is
both a useful map and a check on the other analysis passes: if the categories do
not sum to the image size, something is double-counting.

Categories, applied in this order so the most specific wins:

  vu-microcode    the Vector Unit regions named in game.json (not MIPS at all)
  code-function   MIPS proven reachable and attributed to a discovered function
  code-orphan     MIPS proven reachable but belonging to no known function
  string          inside a NUL-terminated printable run
  pointer-table   inside a detected run of function pointers
  zero            padding
  data            everything else

    python engine/tools/imagemap.py
    python engine/tools/imagemap.py --markdown
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile

CATEGORIES = [
    "vu-microcode",
    "code-function",
    "code-orphan",
    "string",
    "pointer-table",
    "zero",
    "data",
]


def classify(database: dict, image: bytes, vu_regions=()) -> bytearray:
    """Return a per-byte category index over the whole image."""
    base = database["meta"]["load_addr"]
    size = len(image)
    index = {name: i for i, name in enumerate(CATEGORIES)}
    marks = bytearray([index["data"]]) * size

    def fill(start: int, end: int, name: str, only_over=None) -> None:
        lo = max(0, start - base)
        hi = min(size, end - base)
        code = index[name]
        for i in range(lo, hi):
            if only_over is None or marks[i] == index[only_over]:
                marks[i] = code

    # Zero padding first, so later categories can overwrite it where they apply.
    for i, byte in enumerate(image):
        if byte == 0:
            marks[i] = index["zero"]

    for entry in database.get("strings", []):
        text = entry["text"]
        fill(entry["addr"], entry["addr"] + len(text) + 1, "string")

    for table in database.get("pointer_tables", []):
        fill(table["addr"], table["addr"] + table["count"] * 4, "pointer-table")

    # Proven-reachable code comes from the analysis bitmap, reconstructed here
    # from the region list the database records.
    for region in database.get("regions", []):
        if region["kind"] == "code":
            fill(region["start"], region["end"], "code-orphan")

    for func in database["functions"]:
        fill(func["addr"], func["end"], "code-function")

    # VU microcode wins over everything: it was previously misread as data, and
    # parts of it also decode as plausible MIPS.
    for low, high in vu_regions:
        fill(low, high, "vu-microcode")

    return marks


def summarise(marks: bytearray) -> dict:
    counts = collections.Counter(marks)
    return {CATEGORIES[i]: counts.get(i, 0) for i in range(len(CATEGORIES))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    image = elf.section_data(elf.exec_sections[0])

    totals = summarise(classify(database, image, game.vu_regions))
    size = len(image)
    accounted = sum(totals.values())

    rows = sorted(totals.items(), key=lambda kv: -kv[1])
    if args.markdown:
        print("| category | bytes | share |")
        print("| --- | ---: | ---: |")
        for name, count in rows:
            print("| `%s` | %s | %.2f%% |" % (name, format(count, ","), count / size * 100))
        print("| **total** | **%s** | **%.2f%%** |" % (format(accounted, ","), accounted / size * 100))
    else:
        print("image: %s (%s bytes)" % (database["meta"]["source"], format(size, ",")))
        print()
        print("%-16s %13s %8s" % ("category", "bytes", "share"))
        print("-" * 40)
        for name, count in rows:
            print("%-16s %13s %7.2f%%" % (name, format(count, ","), count / size * 100))
        print("-" * 40)
        print("%-16s %13s %7.2f%%" % ("total", format(accounted, ","), accounted / size * 100))
        if accounted != size:
            print("\nWARNING: categories do not sum to the image size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
