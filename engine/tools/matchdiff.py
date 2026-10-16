#!/usr/bin/env python3
"""Compare candidate machine code against the original image, function by function.

This is the verification step that decides whether a reconstructed function is
actually correct.  A function is only `matched` when this reports a perfect
score.

Two comparison modes:

*   ``--object``  compare against a compiled ELF object produced by the real
    toolchain.  This is the authoritative check.
*   ``--asm``     compare against a flat file of hex words (one 32-bit word per
    line, ``#`` comments allowed).  Useful for checking hand-written assembly,
    and for testing this tool without a compiler installed.

Scoring deliberately reports two numbers, because they fail differently:

*   **exact**   bytes identical.  This is the bar for `matched`.
*   **semantic** identical after ignoring register *allocation* differences -
    same opcodes in the same order.  A high semantic score with a low exact
    score means the source logic is right and only register choice differs,
    which is the normal state of a function that is nearly done.

    python engine/tools/matchdiff.py 0x0010BE40 --asm cand.txt
    python engine/tools/matchdiff.py 0x0010BE40 --object games/burnout3/build/syscalls.o --symbol SignalSema
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile
from ps2decomp.r5900 import DISASM


def load_original(database: dict, elf_path: str, addr: int, size: int) -> list:
    elf = ElfFile.load(elf_path)
    section = elf.exec_sections[0]
    data = elf.section_data(section)
    off = addr - section.addr
    if off < 0 or off + size > len(data):
        raise SystemExit("0x%08x..0x%08x is outside the image" % (addr, addr + size))
    return list(struct.unpack_from("<%dI" % (size // 4), data, off))


def load_candidate_asm(path: str) -> list:
    """Read a flat list of 32-bit words written as hex, one per line."""
    words = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            words.append(int(line, 16) & 0xFFFFFFFF)
    return words


def load_candidate_object(path: str, symbol: str | None) -> list:
    """Pull one function's words out of a compiled object file.

    Without a symbol table the whole executable section is returned, which is
    the right behaviour for a single-function object.
    """
    elf = ElfFile.load(path)
    exec_sections = elf.exec_sections
    if not exec_sections:
        raise SystemExit("%s has no executable section" % path)

    if symbol:
        match = next(
            (s for s in elf.symbols if s.name == symbol and s.is_function), None
        )
        if match is None:
            names = sorted({s.name for s in elf.symbols if s.is_function})
            raise SystemExit(
                "symbol %r not found; object defines: %s"
                % (symbol, ", ".join(names) or "(none)")
            )
        section = elf.sections[match.shndx]
        data = elf.section_data(section)
        off = match.value - section.addr
        size = match.size or (len(data) - off)
        return list(struct.unpack_from("<%dI" % (size // 4), data, off))

    data = elf.section_data(exec_sections[0])
    return list(struct.unpack_from("<%dI" % (len(data) // 4), data))


def compare(original: list, candidate: list, base_addr: int) -> dict:
    """Score a candidate against the original and build a per-line diff."""
    rows = []
    exact = semantic = 0
    total = max(len(original), len(candidate))
    for i in range(total):
        want = original[i] if i < len(original) else None
        got = candidate[i] if i < len(candidate) else None
        addr = base_addr + i * 4

        if want is None or got is None:
            rows.append((addr, want, got, "length"))
            continue

        want_ins = DISASM.decode(want, addr)
        got_ins = DISASM.decode(got, addr)
        if want == got:
            exact += 1
            semantic += 1
            rows.append((addr, want, got, "ok"))
        elif want_ins.mnemonic == got_ins.mnemonic:
            # Same operation, different operands: almost always register
            # allocation, which is worth distinguishing from wrong logic.
            semantic += 1
            rows.append((addr, want, got, "operands"))
        else:
            rows.append((addr, want, got, "opcode"))

    return {
        "total": total,
        "exact": exact,
        "semantic": semantic,
        "exact_pct": exact / total * 100.0 if total else 0.0,
        "semantic_pct": semantic / total * 100.0 if total else 0.0,
        "rows": rows,
        "length_match": len(original) == len(candidate),
    }


def render(result: dict, show_all: bool) -> str:
    lines = []
    lines.append("instructions : %d" % result["total"])
    lines.append("exact        : %d (%.2f%%)" % (result["exact"], result["exact_pct"]))
    lines.append("semantic     : %d (%.2f%%)" % (result["semantic"], result["semantic_pct"]))
    lines.append("length match : %s" % ("yes" if result["length_match"] else "NO"))
    lines.append("")
    verdict = (
        "MATCHED - byte identical"
        if result["exact"] == result["total"] and result["length_match"]
        else "not matched"
    )
    lines.append("verdict      : %s" % verdict)
    lines.append("")

    shown = 0
    for addr, want, got, status in result["rows"]:
        if status == "ok" and not show_all:
            continue
        shown += 1
        if shown > 60 and not show_all:
            lines.append("  ... (use --all to see every line)")
            break
        want_text = DISASM.decode(want, addr).format() if want is not None else "(absent)"
        got_text = DISASM.decode(got, addr).format() if got is not None else "(absent)"
        marker = {"ok": "  ", "operands": " ~", "opcode": " !", "length": " +"}[status]
        lines.append("%s %08x  want %-34s got %s" % (marker, addr, want_text, got_text))
    if shown == 0:
        lines.append("  (no differences)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("addr", help="address of the function in the original image")
    parser.add_argument("--asm", help="candidate as a flat hex word list")
    parser.add_argument("--object", help="candidate as a compiled ELF object")
    parser.add_argument("--symbol", help="symbol to extract from --object")
    parser.add_argument("--all", action="store_true", help="show matching lines too")
    args = parser.parse_args(argv)

    if not args.asm and not args.object:
        parser.error("supply either --asm or --object")

    game = game_config.from_args(args)
    database = game.load_database()
    addr = int(args.addr, 0)
    func = next((f for f in database["functions"] if f["addr"] == addr), None)
    if func is None:
        print("no function at 0x%08x in the database" % addr, file=sys.stderr)
        return 1

    elf_path = game.elf_path
    original = load_original(database, elf_path, addr, func["size"])
    candidate = (
        load_candidate_asm(args.asm)
        if args.asm
        else load_candidate_object(args.object, args.symbol)
    )

    print("function     : %s  (0x%08x, %d bytes)" % (func["label"], addr, func["size"]))
    print(render(compare(original, candidate, addr), args.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
