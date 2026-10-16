#!/usr/bin/env python3
"""Look into a frame-pacing profile (PS2HOST_PROFILE, ps2host/timeline.h):
each thread's work around a moment, or the longest spans of each kind.

    python engine/tools/timeline.py profile.json at 24.111 [--before 60] [--after 10]
    python engine/tools/timeline.py profile.json longest [--top 15]

`at` lists, lane by lane, what happened from --before ms to --after ms
around the moment (seconds from the profile's start, as its summary gives
them); `longest` the longest spans of every name on every lane, with when.
"""

import argparse
import json
from collections import defaultdict


def load(path):
    with open(path) as f:
        data = json.load(f)
    lanes, events = {}, defaultdict(list)
    for e in data["traceEvents"]:
        if e.get("ph") == "M":
            lanes[e["tid"]] = e["args"]["name"]
        else:
            events[e["tid"]].append(e)
    return lanes, events


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile")
    parser.add_argument("what", choices=("at", "longest", "frames"))
    parser.add_argument("moment", nargs="?", type=float, help="seconds (for at)")
    parser.add_argument("--before", type=float, default=60.0, help="ms before the moment")
    parser.add_argument("--after", type=float, default=10.0, help="ms after it")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min", type=float, default=0.3, help="ms: shorter spans left out (at)")
    args = parser.parse_args()
    lanes, events = load(args.profile)
    if args.what == "at":
        at_us = args.moment * 1e6
        lo, hi = at_us - args.before * 1e3, at_us + args.after * 1e3
        for tid, name in lanes.items():
            shown = []
            for e in events[tid]:
                start, dur = e["ts"], e.get("dur", 0.0)
                if start + dur < lo or start > hi:
                    continue
                if e["ph"] == "X" and dur / 1e3 < args.min:
                    continue
                when = (start - at_us) / 1e3
                if e["ph"] == "X":
                    shown.append("%+8.1f ms  %6.1f ms  %s" % (when, dur / 1e3, e["name"]))
                else:
                    shown.append("%+8.1f ms     -       %s %s" % (when, e["name"], e.get("args", {}).get("value", "")))
            if shown:
                print("== %s" % name)
                print("\n".join(shown))
    elif args.what == "frames":
        # Each lane's "busy" time in each frame (from one blank to the next):
        # its percentiles, and the frames past the period.
        blanks = sorted(e["ts"] for tid in lanes for e in events[tid] if e["ph"] == "i" and e["name"] == "frame")
        if len(blanks) < 3:
            print("no frames")
            return
        period = sorted(b - a for a, b in zip(blanks, blanks[1:]))[len(blanks) // 2]
        for tid, name in lanes.items():
            spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in events[tid] if e["ph"] == "X" and e["name"] == "busy")
            if not spans:
                continue
            per, j = [], 0
            for a, b in zip(blanks, blanks[1:]):
                total = 0.0
                while j < len(spans) and spans[j][1] < a:
                    j += 1
                k = j
                while k < len(spans) and spans[k][0] < b:
                    total += min(spans[k][1], b) - max(spans[k][0], a)
                    k += 1
                per.append(total / 1e3)
            s = sorted(per)
            q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
            over = sum(1 for v in per if v > period / 1e3)
            print("%-14s busy a frame: median %5.1f ms, 95%% %5.1f, 99%% %5.1f, most %5.1f; past the period (%.1f ms): %d of %d"
                  % (name, q(0.5), q(0.95), q(0.99), s[-1], period / 1e3, over, len(per)))
    else:
        for tid, name in lanes.items():
            by_name = defaultdict(list)
            for e in events[tid]:
                if e["ph"] == "X":
                    by_name[e["name"]].append(e)
            for what, spans in sorted(by_name.items()):
                spans.sort(key=lambda e: -e["dur"])
                top = ", ".join("%.1f ms at %.3f s" % (e["dur"] / 1e3, e["ts"] / 1e6) for e in spans[:args.top])
                print("%s / %s (%d): %s" % (name, what, len(spans), top))


if __name__ == "__main__":
    main()
