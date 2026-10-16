#!/usr/bin/env python3
"""Name functions from evidence, and share those names between games.

    python engine/tools/signatures.py name               # write config/names.csv
    python engine/tools/signatures.py export             # add its names to the library
    python engine/tools/signatures.py compare --elf other.elf

`name` collects every name the evidence supports (kernel stubs, the status table,
self-identifying diagnostic strings, the cross-game signature library) into the game's `config/names.csv`. `export`
adds the game's named functions to `engine/signatures/library.json`, so the next
game is named from them. `compare` measures how much code the game shares with
another executable, with no names involved.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config
from ps2decomp import naming, signatures
from ps2decomp.elf import ElfFile

# Below this many instructions a match is too likely to be coincidence to name a
# function by: a leaf that returns a constant looks the same everywhere.
MIN_INSTRUCTIONS = 4


def _image(elf):
    section = elf.exec_sections[0]
    return elf.section_data(section), section.addr


def _names_path(game):
    return game.path(os.path.join(os.path.dirname(game.status), "names.csv"))


def name(game) -> int:
    database = game.load_database()
    image, base = _image(game.load_elf())
    sigs = signatures.compute(database, image, base)
    sizes = {f["addr"]: f["instructions"] for f in database["functions"]}
    library = signatures.Library.load()
    names = {}
    table = naming.syscall_table()
    for address, service in naming.syscall_stubs(database, image, base).items():
        if service in table:
            names[address] = (table[service], "kernel service %s0x%02X"
                              % ("-" if service < 0 else "", abs(service)))
    for address, text in naming.status_names(game.path(game.status)).items():
        names.setdefault(address, (text, "status table"))
    for address, (text, message) in naming.diagnostic_names(database).items():
        names.setdefault(address, (text, "diagnostic string"))
    for address, (text, evidence) in naming.rpc_names(database, image, base).items():
        names.setdefault(address, (text, "SIF RPC call sites - " + evidence))
    from_library = 0
    for address, key in sigs.items():
        if address in names or sizes.get(address, 0) < MIN_INSTRUCTIONS:
            continue
        found = library.name_for(key)
        if found:
            seen = ";".join(library.sources(key))
            names[address] = (found, "signature library (%s)" % seen)
            from_library += 1
    path = _names_path(game)
    naming.save(path, names)
    sources = collections.Counter(source.split(" (")[0].split(" - ")[0].split(" 0x")[0].split(" -0x")[0]
                                  for _n, source in names.values())
    print("named %d functions -> %s" % (len(names), path))
    for source, count in sources.most_common():
        print("  %-24s %d" % (source, count))
    return 0


def export(game) -> int:
    database = game.load_database()
    image, base = _image(game.load_elf())
    sigs = signatures.compute(database, image, base)
    sizes = {f["addr"]: f["instructions"] for f in database["functions"]}
    library = signatures.Library.load()
    before = len(library.entries)
    library.withdraw(game.executable)
    added = 0
    for address, (text, _source) in naming.load(_names_path(game)).items():
        if address in sigs and sizes.get(address, 0) >= MIN_INSTRUCTIONS:
            library.add(sigs[address], text, game.executable)
            added += 1
    library.save()
    ambiguous = sum(1 for entry in library.entries.values() if len(entry) > 1)
    print("exported %d names; library has %d signatures (%+d, %d ambiguous)"
          % (added, len(library.entries), len(library.entries) - before, ambiguous))
    return 0


def compare(game, other_path: str, show: int) -> int:
    import analyze

    database = game.load_database()
    image, base = _image(game.load_elf())
    other_db = analyze.build_database(other_path, verbose=False)
    other_image, other_base = _image(ElfFile.load(other_path))
    ours = signatures.compute(database, image, base)
    theirs = collections.Counter(signatures.compute(other_db, other_image, other_base).values())
    functions = {f["addr"]: f for f in database["functions"]}
    names = naming.load(_names_path(game))

    eligible = [a for a in ours if functions[a]["instructions"] >= MIN_INSTRUCTIONS]
    shared = [a for a in eligible if theirs[ours[a]]]
    total_bytes = sum(functions[a]["size"] for a in eligible)
    shared_bytes = sum(functions[a]["size"] for a in shared)
    print("%s against %s" % (game.executable, os.path.basename(other_path)))
    print("  functions of %d+ instructions : %d" % (MIN_INSTRUCTIONS, len(eligible)))
    print("  ...also present in the other : %d (%.1f%%), %s bytes (%.1f%%)"
          % (len(shared), 100.0 * len(shared) / max(1, len(eligible)),
             format(shared_bytes, ","), 100.0 * shared_bytes / max(1, total_bytes)))
    named = [a for a in shared if a in names]
    print("  ...of which already named    : %d" % len(named))
    if show:
        print("  largest shared functions:")
        for address in sorted(shared, key=lambda a: -functions[a]["size"])[:show]:
            label = names.get(address, ("", ""))[0] or functions[address].get("string_hint", "")
            print("    0x%08X %6d bytes  %s" % (address, functions[address]["size"], label[:60]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("command", choices=("name", "export", "compare"))
    parser.add_argument("--elf", help="the other executable, for compare")
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    if args.command == "name":
        return name(game)
    if args.command == "export":
        return export(game)
    if not args.elf:
        parser.error("compare needs --elf")
    return compare(game, args.elf, args.show)


if __name__ == "__main__":
    raise SystemExit(main())
