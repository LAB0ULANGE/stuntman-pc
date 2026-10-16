#!/usr/bin/env python3
"""Inspect and extract files from the game disc image.

The disc image itself is never committed to the repository; every command here
takes the path to a disc the user already owns.

Examples:
    python engine/tools/isotool.py info  "Burnout 3 - Takedown.iso"
    python engine/tools/isotool.py list  "Burnout 3 - Takedown.iso"
    python engine/tools/isotool.py extract "Burnout 3 - Takedown.iso" SLUS_210.50 -o games/burnout3/extracted/
    python engine/tools/isotool.py extract-all "Burnout 3 - Takedown.iso" -o games/burnout3/extracted/ --max-size 4M
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.iso9660 import Iso9660, DirectoryRecord


def _human(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size} B"


def _parse_size(text: str) -> int:
    """Parse a size like '4M', '512K', '2048' into a byte count."""
    text = text.strip().upper()
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if text and text[-1] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(text)


def cmd_info(args: argparse.Namespace) -> int:
    with Iso9660.open(args.image) as iso:
        image_size = os.path.getsize(args.image)
        print(f"image            : {args.image}")
        print(f"image size       : {_human(image_size)} ({image_size:,} bytes)")
        print(f"system id        : {iso.system_identifier}")
        print(f"volume id        : {iso.volume_identifier}")
        print(f"publisher        : {iso.publisher_identifier}")
        print(f"volume sectors   : {iso.volume_space_size:,}")
        print(f"volume bytes     : {_human(iso.volume_space_size * 2048)}")
        files = dirs = 0
        total = 0
        for record in iso.walk():
            if record.is_dir:
                dirs += 1
            else:
                files += 1
                total += record.size
        print(f"directories      : {dirs:,}")
        print(f"files            : {files:,}")
        print(f"total file bytes : {_human(total)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with Iso9660.open(args.image) as iso:
        rows: list[tuple[str, int, int, str]] = []
        for record in iso.walk():
            if record.is_dir and not args.include_dirs:
                continue
            stamp = record.recorded.isoformat(sep=" ") if record.recorded else ""
            rows.append((record.path, record.lba, record.size, stamp))
        rows.sort(key=lambda r: (-r[2], r[0]) if args.by_size else (r[0], 0))
        print(f"{'lba':>9}  {'size':>13}  {'recorded':<19}  path")
        for path, lba, size, stamp in rows:
            print(f"{lba:>9}  {size:>13,}  {stamp:<19}  {path}")
        print(f"\n{len(rows):,} entries")
    return 0


def _extract_one(iso: Iso9660, record: DirectoryRecord, out_dir: str, quiet: bool) -> int:
    dest = os.path.join(out_dir, record.path.replace("/", os.sep))
    written = iso.extract_file(record, dest)
    if not quiet:
        print(f"{_human(written):>12}  {record.path}")
    return written


def cmd_extract(args: argparse.Namespace) -> int:
    with Iso9660.open(args.image) as iso:
        total = 0
        for path in args.paths:
            record = iso.find(path)
            if record.is_dir:
                for child in iso.walk(record):
                    if not child.is_dir:
                        total += _extract_one(iso, child, args.output, args.quiet)
            else:
                total += _extract_one(iso, record, args.output, args.quiet)
        print(f"\nextracted {_human(total)} to {args.output}")
    return 0


def cmd_extract_all(args: argparse.Namespace) -> int:
    limit = _parse_size(args.max_size) if args.max_size else None
    with Iso9660.open(args.image) as iso:
        total = skipped = 0
        for record in iso.walk():
            if record.is_dir:
                continue
            if limit is not None and record.size > limit:
                skipped += 1
                continue
            total += _extract_one(iso, record, args.output, args.quiet)
        print(f"\nextracted {_human(total)} to {args.output}")
        if skipped:
            print(f"skipped {skipped} file(s) larger than {args.max_size}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print volume metadata and totals")
    p_info.add_argument("image")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list", help="list the files on the disc")
    p_list.add_argument("image")
    p_list.add_argument("--include-dirs", action="store_true")
    p_list.add_argument("--by-size", action="store_true", help="sort largest first")
    p_list.set_defaults(func=cmd_list)

    p_ext = sub.add_parser("extract", help="extract named files or directories")
    p_ext.add_argument("image")
    p_ext.add_argument("paths", nargs="+")
    p_ext.add_argument("-o", "--output", default="extracted")
    p_ext.add_argument("-q", "--quiet", action="store_true")
    p_ext.set_defaults(func=cmd_extract)

    p_all = sub.add_parser("extract-all", help="extract every file on the disc")
    p_all.add_argument("image")
    p_all.add_argument("-o", "--output", default="extracted")
    p_all.add_argument("--max-size", help="skip files larger than this, e.g. 4M")
    p_all.add_argument("-q", "--quiet", action="store_true")
    p_all.set_defaults(func=cmd_extract_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
