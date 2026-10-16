#!/usr/bin/env python3
"""Find the function whose translation changes a picture of a scripted run.

    # two emissions of the game's src/lifted, e.g. structured and plain:
    python engine/tools/lift.py --game games/<g> --structure --emit;  cp -r games/<g>/src/lifted /tmp/a
    python engine/tools/lift.py --game games/<g> --emit;              cp -r games/<g>/src/lifted /tmp/b
    python engine/tools/bisect_lifted.py --game games/<g> --run car_select --frame 7750 \\
        --suspect /tmp/a --reference /tmp/b

Every function's lifted forms are checked against the original, but only on
the inputs the tests reach. When a scripted run draws something different
from one emission to another - a structured form against the plain,
instruction-by-instruction one - one function's form is wrong on the game's
own inputs. This builds the port from the suspect emission with chosen
functions' bodies taken from the reference, renders one frame of the run
(--virtual-time: the same frame every time) and halves the candidates until
one function is left: the functions the run calls (from the reference
build's first-call log) whose two forms differ.

It rewrites the game's src/lifted and rebuilds the port, a few minutes a
step; at the end src/lifted holds the suspect emission with the culprit's
body from the reference. Burnout 3's car select found 0x0042DAE0 in twelve
steps (games/burnout3/docs/FINDINGS.md, "The garage without its cars").
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ps2decomp import game as game_config  # noqa: E402

FUNC = re.compile(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{$")


def parse(directory):
    """({file: [chunk]}, {address: text}); a chunk is ('text', s) or ('func', address, s)."""
    files, funcs = {}, {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".cpp"):
            continue
        lines = open(os.path.join(directory, name), encoding="utf-8").read().split("\n")
        chunks, i, pending = [], 0, []
        while i < len(lines):
            match = FUNC.match(lines[i])
            if match:
                j = i
                while lines[j] != "}":
                    j += 1
                text = "\n".join(pending + lines[i:j + 1])
                address = int(match.group(1), 16)
                chunks.append(("func", address, text))
                funcs[address] = text
                pending = []
                i = j + 1
                continue
            if lines[i].startswith("/* NOT VERIFIED") and lines[i].endswith("*/"):
                pending.append(lines[i])
            else:
                if pending:
                    chunks.append(("text", "\n".join(pending)))
                    pending = []
                chunks.append(("text", lines[i]))
            i += 1
        files[name] = chunks
    return files, funcs


def write(out_dir, files, reference, use_reference):
    """The suspect emission's files, with `use_reference`'s bodies from the reference."""
    for name, chunks in files.items():
        text = "\n".join(reference[c[1]] if c[0] == "func" and c[1] in use_reference else
                         (c[2] if c[0] == "func" else c[1]) for c in chunks)
        path = os.path.join(out_dir, name)
        if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)


class Renderer:
    """Builds the port and renders one frame of a scripted run."""

    def __init__(self, game, run, frame, gpu):
        self.game, self.frame, self.gpu = game, frame, gpu
        with open(game.path(os.path.join("port", "runs", run + ".args")), encoding="utf-8") as handle:
            args = handle.read().split()
        # Stop just after the frame.
        for i, arg in enumerate(args):
            if arg == "--until-frame":
                args[i + 1] = str(frame + 1)
        if "--until-frame" not in args:
            args += ["--until-frame", str(frame + 1)]
        self.args = args
        self.root = os.path.dirname(os.path.dirname(HERE))

    def render(self, log_calls=None):
        built = subprocess.run([sys.executable, os.path.join(HERE, "port.py"), "--game", self.game.directory,
                                "build"], cwd=self.root, capture_output=True, text=True)
        if "built" not in built.stdout:
            raise SystemExit("the port did not build:\n" + built.stdout[-3000:] + built.stderr[-3000:])
        frames = tempfile.mkdtemp(prefix="bisect_")
        exe = self.game.path(os.path.join(self.game.build, "port", "cmake",
                                          os.path.basename(os.path.normpath(self.game.directory)) + ".exe"))
        command = [exe, self.game.path(os.path.join(self.game.extracted, self.game.executable))]
        if self.game.iso and os.path.exists(self.game.path(self.game.iso)):
            command += ["--disc", self.game.path(self.game.iso)]
        command += self.args + ["--timeout", "14400", "--frames", frames, "--frames-every", str(self.frame)]
        if self.gpu:
            command.append("--gpu")
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, errors="replace")
        if log_calls is not None:
            log_calls.update(int(m.group(1), 16) for m in re.finditer(r"^\[call\] 0x([0-9A-F]{8})",
                                                                      result.stderr, re.M))
        picture = os.path.join(frames, "frame_%06d.bmp" % self.frame)
        if not os.path.exists(picture):
            raise SystemExit("the run did not reach frame %d" % self.frame)
        with open(picture, "rb") as handle:
            crc = "%08X" % (zlib.crc32(handle.read()) & 0xFFFFFFFF)
        shutil.rmtree(frames, ignore_errors=True)
        return crc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--run", required=True, help="the scripted run (port/runs/<run>.args)")
    parser.add_argument("--frame", type=int, required=True, help="the frame whose picture differs")
    parser.add_argument("--suspect", required=True, help="a copy of src/lifted drawing it wrong")
    parser.add_argument("--reference", required=True, help="a copy of src/lifted drawing it right")
    parser.add_argument("--gpu", action="store_true", help="render with --gpu")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    out_dir = game.path(os.path.join("src", "lifted"))
    files, suspect = parse(args.suspect)
    _unused, reference = parse(args.reference)
    if set(suspect) != set(reference):
        raise SystemExit("the two emissions translate different functions")
    renderer = Renderer(game, args.run, args.frame, args.gpu)

    started = time.time()
    called = set()
    write(out_dir, files, reference, set(reference))
    good = renderer.render(log_calls=called)
    write(out_dir, files, reference, set())
    bad = renderer.render()
    print("reference %s, suspect %s (%.0fs)" % (good, bad, time.time() - started), flush=True)
    if good == bad:
        print("the two emissions draw frame %d the same" % args.frame)
        return 1
    candidates = sorted(a for a in suspect if a in called and suspect[a] != reference[a])
    print("%d functions differ, %d of them called" % (sum(suspect[a] != reference[a] for a in suspect),
                                                      len(candidates)), flush=True)
    # Invariant: the reference's bodies for `candidates` draw the frame right.
    while len(candidates) > 1:
        half = candidates[:len(candidates) // 2]
        write(out_dir, files, reference, set(half))
        crc = renderer.render()
        print("  reference bodies for %d of %d: %s" % (len(half), len(candidates),
                                                       "right" if crc == good else "wrong"), flush=True)
        candidates = half if crc == good else candidates[len(candidates) // 2:]
    culprit = candidates[0]
    write(out_dir, files, reference, {culprit})
    alone = renderer.render() == good
    print("culprit: F_%08X (its reference body alone draws the frame %s) - %.0fs"
          % (culprit, "right" if alone else "STILL WRONG: more than one function", time.time() - started))
    return 0 if alone else 2


if __name__ == "__main__":
    sys.exit(main())
