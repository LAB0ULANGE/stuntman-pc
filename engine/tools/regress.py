#!/usr/bin/env python3
"""Run a scripted run and check its pictures against the ones recorded for it.

    python engine/tools/regress.py --game games/burnout3 car_select --gpu
    python engine/tools/regress.py --game games/burnout3 car_select --gpu --update

A scripted run (port/runs/<name>.args, under --virtual-time) draws the same
pictures every time with the same build, so a picture that changes is either
a fix or a bug - this says which frames did. The run's pictures (every
--every frames) are reduced to CRCs and compared with
port/runs/<name>.<renderer>.crc; only the CRCs are kept in the repository,
never the pictures. --update records the run's CRCs as the new expectation,
for a change meant to alter them (say what changed in the commit).

The run's own report (its time line) is printed too, so a run is also a
timing check.

A run's arguments may name `{memcard}`: the run then gets an empty memory
card (--memcard, a fresh directory), and what the game left on it is checked
with the pictures, file by file (memcard/<path> in the CRC list).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps2decomp import game as game_config  # noqa: E402


def crcs(directory: str) -> dict:
    out = {}
    for name in sorted(os.listdir(directory)):
        if name.startswith("frame_") and name.endswith(".bmp"):
            with open(os.path.join(directory, name), "rb") as handle:
                out[name] = "%08X" % (zlib.crc32(handle.read()) & 0xFFFFFFFF)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    game_config.add_argument(parser)
    parser.add_argument("run", help="the scripted run (port/runs/<run>.args)")
    parser.add_argument("--gpu", action="store_true", help="the GPU renderer (--gpu), else the software GS")
    parser.add_argument("--every", type=int, default=200, help="a picture every N frames")
    parser.add_argument("--vu-fast", action="store_true",
                        help="run with --vu-fast (compared with the same CRCs: what the exact VU drew)")
    parser.add_argument("--update", action="store_true", help="record this run's pictures as expected")
    parser.add_argument("--keep", help="keep the pictures in this directory")
    parser.add_argument("--frame-times", type=int, default=0, metavar="N",
                        help="print the wall-clock frame rate every N frames (the run's --frame-times)")
    args = parser.parse_args(argv)
    if args.vu_fast and args.update:
        parser.error("--update records the exact VU's pictures: not with --vu-fast")
    game = game_config.from_args(args)

    runs = game.path(os.path.join("port", "runs"))
    with open(os.path.join(runs, args.run + ".args"), encoding="utf-8") as handle:
        run_args = handle.read().split()
    executable = game.path(os.path.join(game.build, "port", "cmake", os.path.basename(
        os.path.normpath(game.directory)) + ".exe"))
    elf = game.path(os.path.join(game.extracted, game.executable))
    disc = game.path(game.iso) if game.iso else ""
    renderer = "gpu" if args.gpu else "soft"
    pictures = args.keep or tempfile.mkdtemp(prefix="regress_")
    os.makedirs(pictures, exist_ok=True)
    card = None
    if any("{memcard}" in arg for arg in run_args):
        card = tempfile.mkdtemp(prefix="regress_card_")
        run_args = [arg.replace("{memcard}", card) for arg in run_args]
    command = [executable, elf] + (["--disc", disc] if disc and os.path.exists(disc) else []) + run_args + [
        "--timeout", "14400", "--frames", pictures, "--frames-every", str(args.every)]
    if args.gpu:
        command.append("--gpu")
    if args.vu_fast:
        command.append("--vu-fast")
    if args.frame_times:
        command += ["--frame-times", str(args.frame_times)]
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - started
    for line in result.stdout.splitlines():
        if line.startswith("time:") or line.startswith("gpu: "):
            print(line)
    for line in result.stderr.splitlines():
        if line.startswith("[frames]"):
            print(line)
    print("run: %.0f s" % elapsed)

    found = crcs(pictures)
    if card:
        for root, _, files in os.walk(card):
            for name in sorted(files):
                path = os.path.join(root, name)
                with open(path, "rb") as handle:
                    key = "memcard/" + os.path.relpath(path, card).replace(os.sep, "/")
                    found[key] = "%08X" % (zlib.crc32(handle.read()) & 0xFFFFFFFF)
        shutil.rmtree(card, ignore_errors=True)
    expected_path = os.path.join(runs, "%s.%s.crc" % (args.run, renderer))
    if args.update or not os.path.exists(expected_path):
        with open(expected_path, "w", encoding="utf-8", newline="\n") as handle:
            for name, crc in found.items():
                handle.write("%s %s\n" % (name, crc))
        print("recorded %d pictures in %s" % (len(found), os.path.relpath(expected_path)))
        status = 0
    else:
        expected = {}
        with open(expected_path, encoding="utf-8") as handle:
            for line in handle:
                name, crc = line.split()
                expected[name] = crc
        changed = [name for name in expected if found.get(name) != expected[name]]
        extra = [name for name in found if name not in expected]
        for name in changed:
            print("  %s: %s" % (name, "missing" if name not in found else "changed"))
        for name in extra:
            print("  %s: new" % name)
        print("%d of %d pictures as recorded" % (len(expected) - len(changed), len(expected)))
        status = 1 if changed or extra else 0
    if not args.keep:
        shutil.rmtree(pictures, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
