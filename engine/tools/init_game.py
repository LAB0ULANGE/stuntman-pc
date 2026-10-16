#!/usr/bin/env python3
"""Start a new game's port: games/<name>/ from engine/templates.

    python engine/tools/init_game.py <name> --iso "<your disc image>" [--title "Its Title"]
    python engine/tools/port.py init <name> --iso "<your disc image>" [--title ...]

For a game repository (the one whose engine/ submodule this is, or --root). The
executable comes from the disc's SYSTEM.CNF (its BOOT2 line); the files
written are game.json, port/CMakeLists.txt (the port's build, the target
named <name>), port/overrides.cpp (empty), port/runs/README.md,
port/release/README.txt (the player's README, to fill in) and
docs/FINDINGS.md - and the repository's .gitignore if it has none (one that
never lets disc material or the game's code in); an existing one is checked
for those rules. Nothing is taken from the disc but the executable's name.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
ROOT = os.path.dirname(ENGINE)
TEMPLATES = os.path.join(ENGINE, "templates")
sys.path.insert(0, ENGINE)

from ps2decomp.iso9660 import Iso9660  # noqa: E402

# What a repository's .gitignore must keep out (templates/repo/gitignore).
REQUIRED_IGNORES = ["*.iso", "*.bin", "*.cue", "extracted/", "asm/", "build/", "bios/", "games/*/src/"]


def boot_executable(iso_path: str) -> str:
    """SYSTEM.CNF's BOOT2 executable: cdrom0:\\SLUS_210.50;1 is SLUS_210.50."""
    iso = Iso9660.open(iso_path)
    text = iso.read_file(iso.find("SYSTEM.CNF")).decode("ascii", "replace")
    found = re.search(r"BOOT2\s*=\s*cdrom0?:\\?([^;\s]+)", text, re.I)
    if not found:
        raise SystemExit("init: no BOOT2 line in the disc's SYSTEM.CNF - is it a PS2 game disc?")
    return found.group(1).lstrip("\\/")


def fill(text: str, values: dict) -> str:
    for key, value in values.items():
        text = text.replace("@%s@" % key, value)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the game's directory name under games/ and the port's target (e.g. burnout3)")
    parser.add_argument("--iso", required=True, help="your disc image of the game")
    parser.add_argument("--title", help="its title, as the window shows it (default: the name)")
    parser.add_argument("--root", default=ROOT, help="the game repository (default: the engine submodule's parent)")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.name):
        raise SystemExit("init: the name is a C++ target and a folder: lower case letters, digits, _")
    game = os.path.join(root, "games", args.name)
    if os.path.exists(game):
        raise SystemExit("init: %s exists already" % game)
    iso = os.path.abspath(args.iso)
    if not os.path.exists(iso):
        raise SystemExit("init: no disc image at %s" % iso)
    executable = boot_executable(iso)
    title = args.title or args.name
    try:
        iso_entry = os.path.relpath(iso, game)
    except ValueError:  # another drive: no relative path to it
        iso_entry = iso
    if not os.path.abspath(iso).startswith(root + os.sep):
        print("init: note - the disc image is outside the repository; game.json names its "
              "folder (the repository's root is the usual place: *.iso is gitignored)")
    values = {
        "NAME": "%s (%s)" % (title, executable.replace("_", "-").replace(".", "")),
        "TITLE": title,
        "TARGET": args.name,
        "GAME": args.name,
        "EXECUTABLE": executable,
        "ISO": iso_entry.replace("\\", "/"),
    }
    source = os.path.join(TEMPLATES, "game")
    written = []
    for folder, _dirs, files in os.walk(source):
        for name in files:
            relative = os.path.relpath(os.path.join(folder, name), source)
            target = os.path.join(game, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                text = fill(f.read(), values)
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            written.append(os.path.relpath(target, root))
    ignore = os.path.join(root, ".gitignore")
    if not os.path.exists(ignore):
        with open(os.path.join(TEMPLATES, "repo", "gitignore"), encoding="utf-8") as f:
            text = f.read()
        with open(ignore, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(".gitignore")
    else:
        with open(ignore, encoding="utf-8") as f:
            lines = {line.strip() for line in f}
        missing = [rule for rule in REQUIRED_IGNORES if rule not in lines]
        if missing:
            print("init: WARNING - the repository's .gitignore does not keep out: %s "
                  "(engine/templates/repo/gitignore has them)" % ", ".join(missing))
    print("init: %s from the disc's SYSTEM.CNF: %s" % (title, executable))
    for path in written:
        print("  wrote %s" % path)
    print("""
next, from the repository's root:
  python engine/tools/isotool.py extract "%s" %s -o games/%s/extracted/
  python engine/tools/analyze.py --game games/%s
  python engine/tools/derive.py  --game games/%s --emit --csv
  python engine/tools/lift.py    --game games/%s --structure --emit --csv
  python engine/tools/port.py    --game games/%s      # build and run: the bring-up loop
(the ps2-decomp and ps2-port skills, engine/.claude/skills, have the rest)""" % (
        args.iso, executable, args.name, args.name, args.name, args.name, args.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
