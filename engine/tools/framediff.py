#!/usr/bin/env python3
"""Compare two directories of frame dumps (--frames) picture by picture.

    python engine/tools/framediff.py out/soft out/gpu [--slack 8] [--diff-dir out/diff]

Two runs of the same scripted input under --virtual-time draw the same frames,
so their dumps can be laid side by side: the software GS against the GPU
renderer (--gpu), or one build against the next. For each frame both have, it
prints the share of pixels whose channels all lie within --slack of each other
and the largest difference; with --diff-dir, a PNG per frame marking the pixels
that differ (red) over a dimmed copy of the first picture.

Exits 1 when any frame falls below --min (default 0: report only).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib


def read_bmp(path: str):
    """(width, height, rows of (b, g, r) bytes per pixel, top-down)."""
    data = open(path, "rb").read()
    if data[:2] != b"BM":
        raise ValueError("%s: not a BMP" % path)
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]
    if bpp not in (24, 32):
        raise ValueError("%s: %d bits per pixel" % (path, bpp))
    step = bpp // 8
    stride = (width * step + 3) & ~3
    rows = []
    for row in range(abs(height)):
        start = offset + row * stride
        rows.append(data[start:start + width * step])
    if height > 0:
        rows.reverse()
    return width, abs(height), step, rows


def write_png(path: str, width: int, height: int, pixels: bytearray) -> None:
    """An RGB PNG from (b, g, r) bytes, rows top-down."""
    raw = bytearray()
    for row in range(height):
        line = pixels[row * width * 3:(row + 1) * width * 3]
        raw.append(0)
        for x in range(0, len(line), 3):
            raw += bytes((line[x + 2], line[x + 1], line[x]))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        handle.write(chunk(b"IEND", b""))


def compare(a: str, b: str, slack: int, diff_path: str | None):
    """(share of pixels within slack, largest channel difference)."""
    wa, ha, sa, ra = read_bmp(a)
    wb, hb, sb, rb = read_bmp(b)
    width, height = min(wa, wb), min(ha, hb)
    close = 0
    worst = 0
    marked = bytearray(width * height * 3) if diff_path else None
    for y in range(height):
        row_a, row_b = ra[y], rb[y]
        for x in range(width):
            pa, pb = x * sa, x * sb
            d = max(abs(row_a[pa] - row_b[pb]), abs(row_a[pa + 1] - row_b[pb + 1]),
                    abs(row_a[pa + 2] - row_b[pb + 2]))
            if d > worst:
                worst = d
            if d <= slack:
                close += 1
                if marked is not None:
                    at = (y * width + x) * 3
                    marked[at:at + 3] = bytes((row_a[pa] // 3, row_a[pa + 1] // 3, row_a[pa + 2] // 3))
            elif marked is not None:
                at = (y * width + x) * 3
                marked[at:at + 3] = b"\0\0\xff"
    if marked is not None:
        write_png(diff_path, width, height, marked)
    if (wa, ha) != (wb, hb):
        print("  %s: sizes differ, %dx%d against %dx%d" % (os.path.basename(a), wa, ha, wb, hb))
    return close / float(width * height or 1), worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--slack", type=int, default=8, help="largest channel difference that still agrees")
    parser.add_argument("--min", type=float, default=0.0, help="fail below this share of agreeing pixels (0-100)")
    parser.add_argument("--diff-dir", help="write a PNG per frame marking the pixels that differ")
    args = parser.parse_args(argv)
    names = sorted(n for n in os.listdir(args.first)
                   if n.endswith(".bmp") and os.path.exists(os.path.join(args.second, n)))
    if not names:
        print("no frames in common")
        return 1
    if args.diff_dir:
        os.makedirs(args.diff_dir, exist_ok=True)
    failed = 0
    total = 0.0
    for name in names:
        diff = os.path.join(args.diff_dir, name[:-4] + ".png") if args.diff_dir else None
        share, worst = compare(os.path.join(args.first, name), os.path.join(args.second, name), args.slack, diff)
        total += share
        flag = ""
        if share * 100 < args.min:
            failed += 1
            flag = "  BELOW %.1f%%" % args.min
        print("%s  %6.2f%% agree  largest difference %3d%s" % (name, share * 100, worst, flag))
    print("%d frames, %.2f%% of pixels agree on average" % (len(names), 100 * total / len(names)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
