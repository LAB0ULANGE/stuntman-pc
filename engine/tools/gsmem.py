#!/usr/bin/env python3
"""Look into GS local memory dumps (--gs-dump-frame): a buffer as a picture, or
two runs' dumps compared buffer by buffer, or an 8-bit texture with its CLUT.

    python engine/tools/gsmem.py show soft/gsmem_9400.bin D2 2 CT32 128 448 out.png
    python engine/tools/gsmem.py diff soft/gsmem_9400.bin gpu/gsmem_9400.bin D2 2 CT32 128 448 [out.png]
    python engine/tools/gsmem.py texture gsmem_3900.bin 3B60 8 512 256 3D60 out.png

A texture is as TEX0 gives it (a --gs-trace line: tbp, tbw, the size, cbp):
PSMT8, its CLUT 256 CT32 colours (CSM1), written with its alpha (the GS's 0
to 128 made 0 to 255) as an RGBA PNG.

FBP is in 2048-word units (as FRAME has it), FBW in 64-pixel units; the
format is CT32, CT24, CT16, CT16S, Z32, Z24, Z16 or Z16S. The layout is
engine/host/src/gs_memory.cpp's. A picture shows the colour (alpha dropped);
a Z buffer's low 24 bits as RGB. diff counts the pixels that differ and, with
an output, marks them red over a dimmed copy of the first.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import framediff  # noqa: E402

FORMATS = {"CT32": 0x00, "CT24": 0x01, "CT16": 0x02, "CT16S": 0x0A,
           "Z32": 0x30, "Z24": 0x31, "Z16": 0x32, "Z16S": 0x3A}

BLOCK32 = [[0, 1, 4, 5, 16, 17, 20, 21], [2, 3, 6, 7, 18, 19, 22, 23],
           [8, 9, 12, 13, 24, 25, 28, 29], [10, 11, 14, 15, 26, 27, 30, 31]]
BLOCKZ32 = [[24, 25, 28, 29, 8, 9, 12, 13], [26, 27, 30, 31, 10, 11, 14, 15],
            [16, 17, 20, 21, 0, 1, 4, 5], [18, 19, 22, 23, 2, 3, 6, 7]]
BLOCK16 = [[0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15],
           [16, 18, 24, 26], [17, 19, 25, 27], [20, 22, 28, 30], [21, 23, 29, 31]]
BLOCK16S = [[0, 2, 16, 18], [1, 3, 17, 19], [8, 10, 24, 26], [9, 11, 25, 27],
            [4, 6, 20, 22], [5, 7, 21, 23], [12, 14, 28, 30], [13, 15, 29, 31]]
BLOCKZ16 = [[24, 26, 16, 18], [25, 27, 17, 19], [28, 30, 20, 22], [29, 31, 21, 23],
            [8, 10, 0, 2], [9, 11, 1, 3], [12, 14, 4, 6], [13, 15, 5, 7]]
BLOCKZ16S = [[24, 26, 8, 10], [25, 27, 9, 11], [16, 18, 0, 2], [17, 19, 1, 3],
             [28, 30, 12, 14], [29, 31, 13, 15], [20, 22, 4, 6], [21, 23, 5, 7]]
COLUMN32 = [[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15],
            [16, 17, 20, 21, 24, 25, 28, 29], [18, 19, 22, 23, 26, 27, 30, 31],
            [32, 33, 36, 37, 40, 41, 44, 45], [34, 35, 38, 39, 42, 43, 46, 47],
            [48, 49, 52, 53, 56, 57, 60, 61], [50, 51, 54, 55, 58, 59, 62, 63]]
COLUMN16 = [[0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27],
            [4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31],
            [32, 34, 40, 42, 48, 50, 56, 58, 33, 35, 41, 43, 49, 51, 57, 59],
            [36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63],
            [64, 66, 72, 74, 80, 82, 88, 90, 65, 67, 73, 75, 81, 83, 89, 91],
            [68, 70, 76, 78, 84, 86, 92, 94, 69, 71, 77, 79, 85, 87, 93, 95],
            [96, 98, 104, 106, 112, 114, 120, 122, 97, 99, 105, 107, 113, 115, 121, 123],
            [100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127]]
MASK = (4 << 20) - 1
# PSMT8's bytes within a block's column pair, rows 0-7 (rows 8-15: these
# plus 128), as gs_memory.cpp's kColumn8.
COLUMN8 = [[0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54],
           [8, 12, 24, 28, 40, 44, 56, 60, 10, 14, 26, 30, 42, 46, 58, 62],
           [33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23],
           [41, 45, 57, 61, 9, 13, 25, 29, 43, 47, 59, 63, 11, 15, 27, 31],
           [96, 100, 112, 116, 64, 68, 80, 84, 98, 102, 114, 118, 66, 70, 82, 86],
           [104, 108, 120, 124, 72, 76, 88, 92, 106, 110, 122, 126, 74, 78, 90, 94],
           [65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119],
           [73, 77, 89, 93, 105, 109, 121, 125, 75, 79, 91, 95, 107, 111, 123, 127]]


def address(bp: int, bw: int, x: int, y: int, psm: int) -> int:
    """The byte address of pixel (x, y), as gs::Memory::Address."""
    def page(px, py):
        return bp + 32 * (py * max(bw, 1) + px)
    if psm in (0x02, 0x0A, 0x32, 0x3A):
        table = {0x02: BLOCK16, 0x0A: BLOCK16S, 0x32: BLOCKZ16, 0x3A: BLOCKZ16S}[psm]
        block = page(x // 64, y // 64) + table[(y % 64) // 8][(x % 64) // 16]
        return (block * 256 + COLUMN16[y % 8][x % 16] * 2) & MASK
    table = BLOCKZ32 if psm in (0x30, 0x31) else BLOCK32
    block = page(x // 64, y // 32) + table[(y % 32) // 8][(x % 64) // 8]
    return (block * 256 + COLUMN32[y % 8][x % 8] * 4) & MASK


def address8(bp: int, bw: int, x: int, y: int) -> int:
    """The byte of PSMT8 pixel (x, y): 128 x 64 pages, BW / 2 to a row."""
    block = bp + 32 * ((y // 64) * max(bw // 2, 1) + x // 128) + BLOCK32[(y % 64) // 16][(x % 128) // 16]
    row = y % 16
    return (block * 256 + COLUMN8[row % 8][x % 16] + (128 if row >= 8 else 0)) & MASK


def texture8(mem: bytes, tbp: int, tbw: int, w: int, h: int, cbp: int) -> list:
    """An 8-bit texture's (r, g, b, a) rows, its CLUT read as the GS loads it
    (CSM1: a 16 x 16 CT32 image, index bits 3 and 4 swapped)."""
    clut = []
    for i in range(256):
        p = (i & 0xE7) | ((i & 0x08) << 1) | ((i & 0x10) >> 1)
        a = address(cbp, 1, p % 16, p // 16, 0x00)
        clut.append((mem[a], mem[a + 1], mem[a + 2], min(255, mem[a + 3] * 2)))
    return [[clut[mem[address8(tbp, tbw, x, y)]] for x in range(w)] for y in range(h)]


def save_rgba(rows: list, path: str) -> None:
    import struct
    import zlib
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\0" + bytes(c for px in row for c in px) for row in rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
                  chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def pixel(mem: bytes, bp: int, bw: int, x: int, y: int, psm: int) -> tuple:
    """(r, g, b) of a pixel; a Z buffer's bytes as they lie."""
    a = address(bp, bw, x, y, psm)
    if psm in (0x02, 0x0A, 0x32, 0x3A):
        c = mem[a] | mem[a + 1] << 8
        if psm in (0x32, 0x3A):
            return (c & 0xFF, c >> 8, 0)
        return ((c & 31) << 3, ((c >> 5) & 31) << 3, ((c >> 10) & 31) << 3)
    return (mem[a], mem[a + 1], mem[a + 2])


def picture(mem: bytes, fbp: int, fbw: int, psm: int, w: int, h: int) -> list:
    return [[pixel(mem, fbp * 32, fbw, x, y, psm) for x in range(w)] for y in range(h)]


def save(rows: list, path: str, marks=None) -> None:
    h, w = len(rows), len(rows[0])
    out = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b = rows[y][x]
            if marks is not None:
                out += b"\0\0\xff" if marks[y][x] else bytes((b // 3, g // 3, r // 3))
            else:
                out += bytes((b, g, r))
    framediff.write_png(path, w, h, out)


def main(argv) -> int:
    if len(argv) < 8:
        print(__doc__)
        return 2
    mode = argv[1]
    if mode == "texture":
        _, _, dump, tbp, tbw, w, h, cbp, out = argv[:9]
        mem = open(dump, "rb").read()
        save_rgba(texture8(mem, int(tbp, 16), int(tbw), int(w), int(h), int(cbp, 16)), out)
        return 0
    if mode == "show":
        _, _, dump, fbp, fbw, fmt, w, h, out = argv[:9]
        mem = open(dump, "rb").read()
        save(picture(mem, int(fbp, 16), int(fbw), FORMATS[fmt], int(w), int(h)), out)
        return 0
    _, _, first, second, fbp, fbw, fmt, w, h = argv[:9]
    out = argv[9] if len(argv) > 9 else None
    a = picture(open(first, "rb").read(), int(fbp, 16), int(fbw), FORMATS[fmt], int(w), int(h))
    b = picture(open(second, "rb").read(), int(fbp, 16), int(fbw), FORMATS[fmt], int(w), int(h))
    marks = [[a[y][x] != b[y][x] for x in range(len(a[0]))] for y in range(len(a))]
    differing = sum(map(sum, marks))
    total = len(a) * len(a[0])
    print("%s %s/%s %sx%s: %d of %d pixels differ (%.2f%%)" % (fmt, fbp, fbw, w, h, differing, total,
                                                               100.0 * differing / total))
    if out:
        save(a, out, marks)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
