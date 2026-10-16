#!/usr/bin/env python3
"""Emit annotated R5900 assembly for functions in the analysis database.

The output is meant to be *read by a human doing the decompilation*, so it
resolves what a raw disassembly leaves implicit: branch targets become local
labels, calls name the target function, materialised addresses are annotated
with the string or data they point at, and delay slots are marked so they are
not mistaken for ordinary sequential code.

    python engine/tools/disasm.py --addr 0x00100008
    python engine/tools/disasm.py --all -o asm/
    python engine/tools/disasm.py --top 20
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
from ps2decomp.r5900 import DISASM, Flags


class AsmWriter:
    """Renders one function as annotated assembly text."""

    def __init__(self, database: dict, image: bytes) -> None:
        self.db = database
        self.image = image
        self.base = database["meta"]["load_addr"]
        self.functions = {f["addr"]: f for f in database["functions"]}
        self.strings = {s["addr"]: s["text"] for s in database["strings"]}
        self.xrefs = {}
        for x in database["xrefs"]:
            self.xrefs.setdefault(x["from"], x["to"])
        self.tables = {t["addr"]: t for t in database["pointer_tables"]}

    # -- helpers ------------------------------------------------------

    def word_at(self, addr: int) -> int:
        off = addr - self.base
        if off < 0 or off + 4 > len(self.image):
            return 0
        return struct.unpack_from("<I", self.image, off)[0]

    def label_for(self, addr: int) -> str:
        func = self.functions.get(addr)
        if func:
            return func["label"]
        return "0x%08x" % addr

    def _escape(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')

    def annotate_target(self, addr: int):
        """Describe what a materialised address points at."""
        text = self.strings.get(addr)
        if text is not None:
            shown = text if len(text) <= 56 else text[:53] + "..."
            return '"%s"' % self._escape(shown)
        func = self.functions.get(addr)
        if func is not None:
            return "-> %s" % func["label"]
        table = self.tables.get(addr)
        if table is not None:
            return "pointer table, %d entries" % table["count"]
        return None

    # -- rendering ----------------------------------------------------

    def render(self, func: dict, show_bytes: bool = False) -> str:
        start, end = func["addr"], func["end"]
        instructions = []
        addr = start
        while addr < end:
            instructions.append(DISASM.decode(self.word_at(addr), addr))
            addr += 4

        # Collect branch targets that land inside this function so they can be
        # rendered as short local labels instead of bare addresses.
        local_targets = sorted(
            {
                ins.target
                for ins in instructions
                if ins.target >= 0 and start <= ins.target < end and not ins.is_call
            }
        )
        labels = {t: ".L%03d" % i for i, t in enumerate(local_targets)}

        lines = []
        lines.append("/" + "*" * 74)
        lines.append(" * %s" % func["label"])
        lines.append(" *")
        lines.append(" * address      : 0x%08x .. 0x%08x  (%d bytes, %d instructions)"
                     % (start, end, func["size"], func["instructions"]))
        lines.append(" * discovered by: %s" % func["confidence"])
        if func["callers"]:
            shown = ", ".join("0x%08x" % c for c in func["callers"][:6])
            more = "" if len(func["callers"]) <= 6 else " (+%d more)" % (len(func["callers"]) - 6)
            lines.append(" * called from  : %s%s" % (shown, more))
        else:
            lines.append(" * called from  : no direct calls (indirect or unreferenced)")
        if func["string_hint"]:
            lines.append(' * string hint  : "%s"' % self._escape(func["string_hint"]))
        if func["uses_vu"]:
            lines.append(" * NOTE         : uses VU0 macro mode; needs hand review")
        lines.append(" */")
        lines.append("glabel %s" % func["label"])

        prev_had_delay = False
        for ins in instructions:
            if ins.addr in labels:
                lines.append("%s:" % labels[ins.addr])

            text = ins.format(symbolizer=lambda a: labels.get(a) or (
                self.functions[a]["label"] if a in self.functions else None
            ))

            comment = []
            if prev_had_delay:
                comment.append("delay slot")
            target = self.xrefs.get(ins.addr)
            if target is not None:
                note = self.annotate_target(target)
                comment.append("0x%08x%s" % (target, " " + note if note else ""))
            if ins.is_call and ins.target >= 0 and ins.target in self.functions:
                pass  # already rendered as the operand label
            if not ins.is_valid:
                comment.append("does not decode as an instruction")

            prefix = "  %08x %s" % (ins.addr, ("%08x " % ins.raw) if show_bytes else "")
            line = "%s %s" % (prefix, text)
            if comment:
                line = "%-62s # %s" % (line, "; ".join(comment))
            lines.append(line)
            prev_had_delay = ins.has_delay_slot

        lines.append("/* end of %s */" % func["label"])
        return "\n".join(lines) + "\n"


def load_image(database: dict, elf_path: str) -> bytes:
    elf = ElfFile.load(elf_path)
    section = elf.exec_sections[0]
    return elf.section_data(section)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--addr", help="render a single function at this address")
    parser.add_argument("--all", action="store_true", help="render every function")
    parser.add_argument("--range", nargs=2, metavar=("START", "END"),
                        help="render the code from START to END, whatever the "
                             "analysis says its functions are")
    parser.add_argument("--top", type=int, help="render the N largest functions")
    parser.add_argument("--with-strings", action="store_true",
                        help="only functions that reference a string")
    parser.add_argument("-o", "--output", help="write .s files into this directory")
    parser.add_argument("--bytes", action="store_true", help="show raw encodings")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf_path = game.elf_path
    image = load_image(database, elf_path)
    writer = AsmWriter(database, image)

    selected = database["functions"]
    if args.range:
        start, end = (int(value, 0) for value in args.range)
        print(writer.render({"addr": start, "end": end, "size": end - start,
                             "instructions": (end - start) // 4, "label": "range_%08X" % start,
                             "confidence": "--range", "callers": [], "string_hint": "",
                             "uses_vu": False}, args.bytes))
        return 0
    if args.addr:
        wanted = int(args.addr, 0)
        selected = [f for f in selected if f["addr"] == wanted]
        if not selected:
            print("no function at 0x%08x" % wanted, file=sys.stderr)
            return 1
    elif args.with_strings:
        selected = [f for f in selected if f["string_hint"]]
    if args.top:
        selected = sorted(selected, key=lambda f: -f["size"])[: args.top]
    elif not args.all and not args.addr and not args.with_strings:
        selected = selected[:1]

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        for func in selected:
            path = os.path.join(args.output, "%s.s" % func["label"])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(writer.render(func, args.bytes))
        print("wrote %s function(s) to %s" % (format(len(selected), ","), args.output))
    else:
        for func in selected:
            print(writer.render(func, args.bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
