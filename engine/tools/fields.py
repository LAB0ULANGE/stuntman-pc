#!/usr/bin/env python3
"""Measure how consistently pointer arguments are used, as groundwork for struct types.

Before trying to recover structure layouts, this answers whether the evidence for
them is coherent. Offsets alone identify nothing - `+8` in two unrelated functions
says nothing about either - so the measurement is built on linkage instead:

1. Every function is scanned for the (offset, width) pairs at which each of its
   pointer arguments is dereferenced.
2. When a function passes one of its own arguments unchanged into a call, the
   caller's slot and the callee's slot hold the same object, so they are linked.
   Linked slots are merged (union-find) into candidate types.
3. Each candidate type is checked for conflicts: one offset accessed with two
   different widths means either the grouping is wrong, or that offset is a union or
   a field accessed piecewise.

Generic routines - strcpy, memcpy and the like - take any pointer, so linking through
them would fuse unrelated types into one. Slots linked to an unusually large number
of distinct callers are treated as such hubs, and results are reported both with and
without links through them.

The scan is linear within each function rather than a full data-flow analysis: an
argument register is followed through plain register moves until something
overwrites it. That under-reports rather than invents - a value overwritten on one
path is simply dropped - which is the right direction for a measurement.

    python engine/tools/fields.py
    python engine/tools/fields.py --hub-limit 20 --show 10
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp.elf import ElfFile
from ps2decomp.r5900 import DISASM

# mnemonic -> (width in bytes, kind)
ACCESSES = {
    "lb": (1, "int"), "lbu": (1, "int"), "sb": (1, "int"),
    "lh": (2, "int"), "lhu": (2, "int"), "sh": (2, "int"),
    "lw": (4, "int"), "lwu": (4, "int"), "sw": (4, "int"),
    "lwc1": (4, "float"), "swc1": (4, "float"),
    "ld": (8, "int"), "sd": (8, "int"),
    "lq": (16, "int"), "sq": (16, "int"),
}
STORES = {"sb", "sh", "sw", "swc1", "sd", "sq"}
MOVES = {"daddu", "addu", "or", "dadd"}           # rd = rs + $zero forms
ARG_REGS = (4, 5, 6, 7)
CALLER_SAVED = set(range(2, 16)) | {24, 25}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def written_register(ins):
    """The general register an instruction writes, or None."""
    m = ins.mnemonic
    if ins.is_call:
        return 31
    if m in STORES or ins.is_branch or (ins.is_jump and not ins.is_call):
        return None
    if m in ("lwc1", "swc1", "mtc1") or m.startswith(("add.", "sub.", "mul.", "div.", "c.")):
        return None
    if m in ACCESSES or m in ("lui", "addiu", "addi", "daddiu", "andi", "ori", "xori",
                              "slti", "sltiu", "mfc1", "qmfc2", "cfc1"):
        return ins.rt
    if ins.rd:
        return ins.rd
    return None


def scan(func, image, base):
    """Return (accesses, calls) for one function.

    accesses: {arg_index: {offset: set((width, kind, is_store))}}
    calls:    [(target, {callee_arg_index: caller_arg_index})]
    """
    held = {reg: index for index, reg in enumerate(ARG_REGS)}   # register -> arg
    accesses = collections.defaultdict(lambda: collections.defaultdict(set))
    calls = []

    def word(addr):
        return struct.unpack_from("<I", image, addr - base)[0]

    def apply(ins):
        m = ins.mnemonic
        if m in ACCESSES and ins.rs in held:
            width, kind = ACCESSES[m]
            accesses[held[ins.rs]][ins.imm].add((width, kind, m in STORES))
        # Track plain copies of an argument: `daddu rd, rs, $zero` and friends.
        if m in MOVES and ins.rd and (ins.rt == 0 and ins.rs in held):
            held[ins.rd] = held[ins.rs]
            return
        if m in MOVES and ins.rd and (ins.rs == 0 and ins.rt in held):
            held[ins.rd] = held[ins.rt]
            return
        target = written_register(ins)
        if target is not None:
            held.pop(target, None)

    addr = func["addr"]
    end = func["end"]
    while addr < end:
        ins = DISASM.decode(word(addr), addr)
        if ins.is_call:
            # The delay slot runs first and often sets the last argument.
            if addr + 4 < end:
                apply(DISASM.decode(word(addr + 4), addr + 4))
            if ins.target >= 0:
                passed = {
                    callee_index: held[reg]
                    for callee_index, reg in enumerate(ARG_REGS) if reg in held
                }
                if passed:
                    calls.append((ins.target, passed))
            for reg in CALLER_SAVED:
                held.pop(reg, None)
            addr += 8
            continue
        apply(ins)
        addr += 4
    return accesses, calls


def analyse(database, image, base, hub_limit):
    known = {f["addr"] for f in database["functions"]}
    slot_accesses = {}
    edges = []
    for func in database["functions"]:
        if not func["size"]:
            continue
        accesses, calls = scan(func, image, base)
        for arg, offsets in accesses.items():
            slot_accesses[(func["addr"], arg)] = offsets
        for target, passed in calls:
            if target not in known:
                continue
            for callee_arg, caller_arg in passed.items():
                edges.append(((func["addr"], caller_arg), (target, callee_arg)))

    incoming = collections.Counter(callee for _caller, callee in edges)
    hubs = {slot for slot, count in incoming.items() if count > hub_limit}

    def build(skip_hubs):
        uf = UnionFind()
        for slot in slot_accesses:
            uf.find(slot)
        for caller, callee in edges:
            if skip_hubs and callee in hubs:
                continue
            uf.union(caller, callee)
        groups = collections.defaultdict(list)
        for slot in list(uf.parent):
            groups[uf.find(slot)].append(slot)
        return groups

    return slot_accesses, edges, hubs, build


def describe(groups, slot_accesses, show):
    rows = []
    for members in groups.values():
        merged = collections.defaultdict(set)
        touched_by = collections.defaultdict(set)
        for slot in members:
            for offset, uses in slot_accesses.get(slot, {}).items():
                merged[offset] |= uses
                touched_by[offset].add(slot[0])
        if not merged:
            continue
        widths = {o: {w for w, _k, _s in uses} for o, uses in merged.items()}
        kinds = {o: {k for _w, k, _s in uses} for o, uses in merged.items()}
        conflicts = sum(1 for ws in widths.values() if len(ws) > 1)
        float_int = sum(1 for ks in kinds.values() if len(ks) > 1)
        functions = len({slot[0] for slot in members})
        # An offset is corroborated when two or more functions use it. A merge with
        # no conflicts proves little on its own - unrelated types using disjoint
        # offsets merge with no conflict at all - so agreement on *shared* offsets is
        # the evidence that the functions really handle the same layout.
        corroborated = [o for o, fs in touched_by.items() if len(fs) >= 2]
        agreeing = sum(1 for o in corroborated if len(widths[o]) == 1)
        rows.append({
            "members": len(members), "functions": functions, "offsets": len(merged),
            "conflicts": conflicts, "float_int": float_int, "merged": merged,
            "span": max(merged) - min(merged) if merged else 0,
            "corroborated": len(corroborated), "agreeing": agreeing,
        })
    return rows


def report(title, rows, show):
    multi = [r for r in rows if r["functions"] >= 2]
    clean = [r for r in multi if r["conflicts"] == 0]
    offsets_total = sum(r["offsets"] for r in multi)
    conflicts_total = sum(r["conflicts"] for r in multi)
    print("== %s ==" % title)
    print("  candidate types seen in 2+ functions : %s" % format(len(multi), ","))
    print("  ...with no width conflict at all      : %s (%.1f%%)"
          % (format(len(clean), ","), 100.0 * len(clean) / max(1, len(multi))))
    print("  offsets across those types            : %s" % format(offsets_total, ","))
    print("  offsets with conflicting widths       : %s (%.1f%%)"
          % (format(conflicts_total, ","), 100.0 * conflicts_total / max(1, offsets_total)))
    corroborated = sum(r["corroborated"] for r in multi)
    agreeing = sum(r["agreeing"] for r in multi)
    print("  offsets used by 2+ functions          : %s (%.0f%% of offsets)"
          % (format(corroborated, ","), 100.0 * corroborated / max(1, offsets_total)))
    print("  ...where those functions agree on width: %.1f%%"
          % (100.0 * agreeing / max(1, corroborated)))
    sizes = sorted((r["functions"] for r in multi), reverse=True)
    if sizes:
        print("  functions per type, largest five      : %s" % sizes[:5])
        print("  types spanning 5+ functions           : %d" % sum(1 for s in sizes if s >= 5))
    if show:
        print("  largest types:")
        for r in sorted(multi, key=lambda r: -r["functions"])[:show]:
            print("    %4d functions  %3d offsets  span 0x%-5X  %d conflicting  %d float/int"
                  % (r["functions"], r["offsets"], r["span"], r["conflicts"], r["float_int"]))
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--hub-limit", type=int, default=20,
                        help="callee slots linked to more callers than this are hubs")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)

    slot_accesses, edges, hubs, build = analyse(
        database, image, section.addr, args.hub_limit
    )
    used = [s for s, offs in slot_accesses.items() if offs]
    print("argument slots dereferenced at least once : %s" % format(len(used), ","))
    print("  ...at two or more distinct offsets       : %s"
          % format(sum(1 for s in used if len(slot_accesses[s]) >= 2), ","))
    print("pass-through links between slots          : %s" % format(len(edges), ","))
    print("hub slots (more than %d callers)           : %d" % (args.hub_limit, len(hubs)))
    print()

    report("all links", describe(build(False), slot_accesses, args.show), args.show)
    report("excluding links into hubs", describe(build(True), slot_accesses, args.show),
           args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
