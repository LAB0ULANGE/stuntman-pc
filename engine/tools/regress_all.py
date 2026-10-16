#!/usr/bin/env python3
"""Every game's recorded runs, before an engine version is tagged.

    python engine/tools/regress_all.py                         # this repository's games
    python engine/tools/regress_all.py --build ../other-pc     # and another's, built first
    python engine/tools/regress_all.py --only race             # just the runs named so

An engine change reaches every port, so a version is tagged only when every
game still draws what it recorded. For each game repository named (default:
the one this engine is the submodule of) and each game in it, this runs each
recorded run: port/runs/<run>.gpu.crc with --gpu, <run>.soft.crc with the
software GS - each game's own engine/tools/regress.py, against the build it
has (--build: built first, with port.py build).

A game repository builds against its own engine/ submodule, so it tests the
engine commit checked out there: each one's is printed, and one that is not
this engine's commit is a warning - check the candidate out there first
(git -C <repo>/engine fetch <this engine> && git -C <repo>/engine checkout
<commit>). Exits non-zero if any run differs from its recording, or failed
to run.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)


def engine_commit(engine: str) -> str:
    result = subprocess.run(["git", "-C", engine, "rev-parse", "--short=12", "HEAD"],
                            capture_output=True, text=True)
    commit = result.stdout.strip() or "?"
    dirty = subprocess.run(["git", "-C", engine, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return commit + ("+changes" if dirty else "")


def recorded_runs(game: str) -> list:
    """(run, gpu) for each recording in the game's port/runs."""
    found = []
    for path in sorted(glob.glob(os.path.join(game, "port", "runs", "*.crc"))):
        stem, renderer = os.path.basename(path)[:-len(".crc")].rsplit(".", 1)
        if renderer in ("gpu", "soft") and os.path.exists(os.path.join(os.path.dirname(path), stem + ".args")):
            found.append((stem, renderer == "gpu"))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="*", default=[os.path.dirname(ENGINE)],
                        help="game repositories (default: the one this engine is in)")
    parser.add_argument("--build", action="store_true", help="build each game first (port.py build)")
    parser.add_argument("--only", action="append", default=[], metavar="RUN",
                        help="only the runs named so (repeatable)")
    parser.add_argument("--gpu-only", action="store_true", help="only the GPU renderer's recordings")
    args = parser.parse_args(argv)

    candidate = engine_commit(ENGINE)
    print("engine under test: %s (%s)" % (candidate, ENGINE))
    results = []
    mismatched = []
    for repo in args.repos:
        repo = os.path.abspath(repo)
        engine = os.path.join(repo, "engine")
        tools = os.path.join(engine, "tools")
        if not os.path.exists(os.path.join(tools, "regress.py")):
            print("%s: no engine/ submodule (git submodule update --init?)" % repo)
            results.append((repo, "-", False, "no engine"))
            continue
        commit = engine_commit(engine)
        if commit != candidate:
            mismatched.append(repo)
        for game in sorted(glob.glob(os.path.join(repo, "games", "*", "game.json"))):
            game = os.path.dirname(game)
            name = os.path.basename(game)
            print("\n=== %s (%s), engine %s%s" % (name, repo, commit,
                                                  "" if commit == candidate else "  WARNING: not the engine under test"))
            if args.build:
                built = subprocess.run([sys.executable, os.path.join(tools, "port.py"), "--game", game, "build"],
                                       cwd=repo)
                if built.returncode:
                    results.append((name, "build", False, "build failed"))
                    continue
            for run, gpu in recorded_runs(game):
                if args.only and run not in args.only:
                    continue
                if args.gpu_only and not gpu:
                    continue
                label = "%s %s" % (run, "gpu" if gpu else "soft")
                print("--- %s %s" % (name, label), flush=True)
                started = time.time()
                command = [sys.executable, os.path.join(tools, "regress.py"), "--game", game, run]
                done = subprocess.run(command + (["--gpu"] if gpu else []), cwd=repo,
                                      capture_output=True, text=True, errors="replace")
                tail = [line for line in done.stdout.splitlines() if line.strip()]
                for line in tail[-12:]:
                    print("    " + line)
                if done.returncode and done.stderr.strip():
                    print("    " + done.stderr.strip().splitlines()[-1])
                verdict = tail[-1] if tail else "no output"
                results.append((name, label, done.returncode == 0, "%s (%.0f s)" % (verdict, time.time() - started)))

    print("\n=== summary (engine %s)" % candidate)
    for name, label, ok, detail in results:
        print("  %-4s %-14s %-24s %s" % ("ok" if ok else "FAIL", name, label, detail))
    for repo in mismatched:
        print("  WARNING: %s tested its own engine commit, not %s" % (repo, candidate))
    failed = [r for r in results if not r[2]]
    if not results:
        print("  no recorded runs found")
        return 1
    print("%d of %d runs as recorded" % (len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
