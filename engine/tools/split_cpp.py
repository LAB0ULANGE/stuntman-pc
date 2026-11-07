#!/usr/bin/env python3
"""A translated file as GCC and Clang take it (engine/host/CMakeLists.txt).

    split_cpp.py <file.cpp> <out dir> <count>

Writes <out dir>/<stem>_0.cpp ... <stem>_<count-1>.cpp, about the same size
each. A piece starts at a top-level function (a line "void F_...(" at column
0, with the comment line before it); every piece repeats the file's head -
what comes before its first function (the includes, the using lines).

The translated code jumps into blocks past a branch's flag, "{ const bool t =
x;", that is no longer read there: MSVC takes the jump, GCC and Clang refuse
to skip an initialisation. Each is written "{ bool t; t = x;", the same code
with nothing to skip.

A piece is rewritten only when it changes, so a build redoes only what did.
"""

from __future__ import annotations

import os
import re
import sys

FUNCTION = re.compile(r"^void F_[0-9A-Fa-f]+\(")
FLAG = re.compile(r"\{ const bool (\w+) = ")


def split(text: str, count: int) -> list[str]:
    if count <= 1:
        return [text]
    lines = text.splitlines(keepends=True)
    starts = []
    for i, line in enumerate(lines):
        if FUNCTION.match(line):
            # The comment line just above goes with its function.
            starts.append(i - 1 if i and lines[i - 1].startswith("/*") else i)
    if not starts:
        return [text] + [""] * (count - 1)
    head = "".join(lines[:starts[0]])
    total = sum(len(line) for line in lines[starts[0]:])
    pieces, current, size, target = [], [], 0, total / count
    bounds = starts[1:] + [len(lines)]
    for begin, end in zip(starts, bounds):
        chunk = "".join(lines[begin:end])
        if current and size + len(chunk) / 2 > target * (len(pieces) + 1) and len(pieces) < count - 1:
            pieces.append(current)
            current = []
        current.append(chunk)
        size += len(chunk)
    pieces.append(current)
    out = [head + "".join(p) for p in pieces]
    return out + [head] * (count - len(out))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    source, out_dir, count = argv[1], argv[2], int(argv[3])
    with open(source, encoding="utf-8", newline="") as f:
        text = FLAG.sub(r"{ bool \1; \1 = ", f.read())
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    for i, piece in enumerate(split(text, count)):
        path = os.path.join(out_dir, "%s_%d.cpp" % (stem, i))
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as f:
                if f.read() == piece:
                    os.utime(path)          # newer than the source: not split again
                    continue
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(piece)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
