#!/usr/bin/env python3
"""Watch a long-running job: alert when its log goes silent or its memory grows.

    python engine/tools/watchdog.py --log games/<g>/build/lift.log --done "elapsed"
    python engine/tools/watchdog.py --log ... --stall 20 --memory 8 --process python

A lift or derive run takes most of an hour, and a stage that hangs looks, from
outside, exactly like one that is working. This polls once a minute and prints a
line when:

* the log has not changed for `--stall` minutes - every long stage of the
  pipeline prints a heartbeat, so silence that long means stuck;
* the processes named `--process` together use more than `--memory` GB;
* the log shows a traceback.

It exits when the log contains `--done` (the job finished), when no matching
process is left, or after `--hours`. Each line it prints is one event, which is
what a monitor running it expects; an agent or a person running a job should
start this beside it rather than wait for the job to end.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time


def memory_gb(process: str) -> tuple:
    """(processes running, their total memory in GB), from the task list."""
    if os.name == "nt":
        output = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                                text=True).stdout
        count, total = 0, 0.0
        for row in csv.reader(io.StringIO(output)):
            if len(row) >= 5 and row[0].lower().startswith(process.lower()):
                count += 1
                total += float(row[4].replace("K", "").replace(",", "").replace(".", "")
                               .strip() or 0) / (1024 * 1024)
        return count, total
    output = subprocess.run(["ps", "-eo", "comm,rss"], capture_output=True, text=True).stdout
    rows = [line.split() for line in output.splitlines()[1:]]
    matching = [int(r[1]) for r in rows if len(r) == 2 and r[0].startswith(process)]
    return len(matching), sum(matching) / (1024 * 1024)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True, help="the job's log file")
    parser.add_argument("--done", default="", help="text in the log that means the job finished")
    parser.add_argument("--stall", type=float, default=20, help="minutes of log silence to alert on")
    parser.add_argument("--memory", type=float, default=8, help="GB of memory to alert on")
    parser.add_argument("--process", default="python", help="process name prefix to measure")
    parser.add_argument("--interval", type=float, default=60, help="seconds between checks")
    parser.add_argument("--hours", type=float, default=4, help="give up watching after this long")
    args = parser.parse_args(argv)

    started = time.time()
    alerted = set()

    def say(key: str, text: str) -> None:
        if key not in alerted:
            alerted.add(key)
            print(time.strftime("%H:%M ") + text, flush=True)

    while time.time() - started < args.hours * 3600:
        text = ""
        if os.path.exists(args.log):
            with open(args.log, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            silent = (time.time() - os.path.getmtime(args.log)) / 60
            if silent > args.stall:
                say("stall %d" % int(silent // args.stall),
                    "STALLED: %s silent for %.0f minutes" % (args.log, silent))
        if args.done and args.done in text:
            print(time.strftime("%H:%M ") + "finished", flush=True)
            return 0
        if "Traceback" in text:
            say("traceback", "FAILED: a traceback in " + args.log)
        count, gb = memory_gb(args.process)
        if count == 0 and time.time() - started > 120:
            print(time.strftime("%H:%M ") + "no %s process left" % args.process, flush=True)
            return 0
        if gb > args.memory:
            say("memory %d" % int(gb), "MEMORY: %d %s processes use %.1f GB" % (count, args.process, gb))
        time.sleep(args.interval)
    print(time.strftime("%H:%M ") + "stopped watching after %.0f hours" % args.hours, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
