#!/usr/bin/env python3
"""Lift every function to C++, verify each against the original, keep what agrees.

    python engine/tools/lift.py                  # lift, build, verify, report
    python engine/tools/lift.py --emit           # also write the verified ones out
    python engine/tools/lift.py --limit 500      # the first 500 functions only
    python engine/tools/lift.py --no-cache       # verify everything afresh

Lifting (engine/ps2decomp/lift.py) is the mechanical tier under derivation:
every instruction becomes a statement over an explicit register context. Each
lifted function is compiled and run on the same test cases a derivation would
get, against the original machine code, and counts only if the *whole* register
file and every byte written agree. The verified ones are written to the game's
`src/lifted/`. Lifted code is complete rather than readable, so it is reported
apart from the derived functions and never counted as them.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import cppcheck, lift, liftcheck, testplan, verdicts
from ps2decomp.liftverify import abi_view, verify  # noqa: F401  (also used by tests)
from ps2decomp import game as game_config
from ps2decomp.emu import Machine

# A file holds at most CHUNK functions and, past its first, CHUNK_BYTES of
# source: a few huge functions (CodeWarrior's inlining makes some of 50 KB of
# code) would otherwise put tens of megabytes in one file - one compiler at
# /O2 for most of an hour, the others idle.
CHUNK = 300
CHUNK_BYTES = 1500000


def _groups(addresses, texts):
    """The addresses split into files' worth, in order."""
    group, size = [], 0
    for address in addresses:
        length = len(texts[address])
        if group and (len(group) >= CHUNK or size + length > CHUNK_BYTES):
            yield group
            group, size = [], 0
        group.append(address)
        size += length
    if group:
        yield group

def lift_all(functions, image, base):
    """{address: C++}, plus a Counter of why the rest were refused."""
    lifted, refused = {}, collections.Counter()
    for func in functions:
        try:
            try:
                # With its out-of-line pieces, where the analysis found them;
                # pieces that cannot be joined leave the function as it was.
                lifted[func["addr"]] = lift.lift_function(image, base, func["addr"], func["end"],
                                                          func.get("chunks", ()))
            except lift.Refused:
                if not func.get("chunks"):
                    raise
                func["chunks"] = []
                lifted[func["addr"]] = lift.lift_function(image, base, func["addr"], func["end"])
        except lift.Refused as exc:
            refused[re.sub(r" at 0x[0-9A-F]+", "", str(exc))] += 1
    return lifted, refused


_CALL = re.compile(r"\b(Call|TailCall)\(c, 0x([0-9A-F]{8})u")


def _annotated(text: str, names: dict) -> str:
    """Calls to a named function say its name."""
    def name(match):
        known = names.get(int(match.group(2), 16))
        return match.group(0) + (" /* %s */" % known[0] if known else "")
    return _CALL.sub(name, text)


def _remove(path: str) -> None:
    """os.remove, waiting out a moment's lock: on Windows a scanner or a compiler
    just finishing can hold a file the run is about to replace (a 40-minute
    lift once died on that)."""
    for attempt in range(20):
        try:
            os.remove(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def write_chunks(lifted: dict, directory: str, prefix: str, unverified=None,
                 names=None) -> list:
    """Write functions in files of CHUNK. A function in `unverified` is preceded by
    why it could not be checked; a named one by its name."""
    unverified, names = unverified or {}, names or {}
    os.makedirs(directory, exist_ok=True)
    for stale in os.listdir(directory):
        if stale.startswith(prefix) and stale.endswith(".cpp"):
            _remove(os.path.join(directory, stale))
    paths = []
    for index, group in enumerate(_groups(sorted(lifted), lifted)):
        path = os.path.join(directory, "%s_%03d.cpp" % (prefix, index))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("/* Lifted from the original machine code by engine/tools/lift.py:\n"
                         " * one statement per instruction. Every function was checked\n"
                         " * against the original - registers and memory - unless marked\n"
                         " * NOT VERIFIED. Generated - do not edit. */\n\n")
            handle.write('#include "ps2rt/lifted.h"\n\n')
            handle.write("using namespace generated;\nusing namespace lifted;\n\n")
            for address in group:
                if address in names:
                    handle.write("/* %s (%s) */\n" % names[address])
                if address in unverified:
                    handle.write("/* NOT VERIFIED: %s */\n" % unverified[address])
                handle.write(_annotated(lifted[address], names))
                handle.write("\n")
        paths.append(path)
    return paths


def build(lifted: dict, work: str):
    """Compile, dropping any function the compiler rejects; returns (dll, lifted)."""
    for _attempt in range(8):
        paths = write_chunks(lifted, work, "lift")
        try:
            return liftcheck.build(paths, work), lifted
        except cppcheck.CompileError as exc:
            broken = set()
            for line in str(exc).splitlines():
                match = re.match(r"(.+?)\((\d+)\)", line)
                # Only errors in the lifted files name a function; others - the
                # compiler's own headers, its internals - are not the code's fault.
                if match and os.path.isfile(match.group(1)) and \
                        os.path.basename(match.group(1)).startswith("lift"):
                    broken |= _functions_at(match.group(1), int(match.group(2)))
            if not broken:
                raise
            print("  dropping %d function(s) that do not compile" % len(broken))
            lifted = {a: t for a, t in lifted.items() if a not in broken}
    raise RuntimeError("could not build the lifted library")


def _functions_at(path: str, line_number: int) -> set:
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()[:line_number]
    for line in reversed(lines):
        match = re.match(r"void F_([0-9A-F]{8})\(", line)
        if match:
            return {int(match.group(1), 16)}
    return set()


# The outcome of a text the compiler rejects - MSVC, for one, refuses blocks
# nested more than 128 deep. Remembered like any other, so it is not rebuilt.
NO_COMPILE = "does not compile"


def verify_texts(texts: dict, work: str, game, seeds, jobs, cache, abi=False,
                 machine=None, functions=None) -> dict:
    """{address: outcome} for every C++ text; remembered verdicts (verdicts.py)
    are reused, and only the rest are built and run. A text that does not
    compile has the outcome NO_COMPILE."""
    outcomes, missing = {}, {}
    for address, text in texts.items():
        outcome = cache.get(text, abi)
        if outcome is None:
            missing[address] = text
        else:
            outcomes[address] = outcome
    if missing:
        dll, built = build(missing, work)
        if jobs > 1 and len(built) > 50:
            fresh = verify_parallel(sorted(built), game, seeds, dll, jobs, abi)
        else:
            library = liftcheck.LiftedFunctions(dll, machine, testplan.VERIFY_BUDGET)
            fresh = {address: verify(functions[address], machine, library, abi)
                     for address in sorted(built)}
        for address in missing:
            if address not in built:
                fresh[address] = NO_COMPILE
        for address, outcome in fresh.items():
            cache.put(missing[address], abi, outcome)
            outcomes[address] = outcome
        cache.save()
    print("  %d verdicts remembered, %d verified afresh"
          % (len(texts) - len(missing), len(missing)))
    return outcomes


def equivalent_to_plain(addresses, structured, plain, abi, cache, work) -> list:
    """The addresses whose structured form behaves as their plain lifted form on
    equivalence.CASES native cases (ps2decomp/equivalence.py) - a check far wider
    than the interpreter's, which two structuring bugs once slipped past.
    Remembered like verdicts, keyed by both texts and the checker's own source."""
    import hashlib

    from ps2decomp import equivalence

    with open(equivalence.__file__, "rb") as handle:
        checker = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()[:16]
    results, missing = {}, {}
    for address in addresses:
        key = "equivalence %s %d\n%s\n//--\n%s" % (checker, equivalence.CASES, plain[address],
                                                    structured[address])
        outcome = cache.get(key, abi)
        if outcome is None:
            missing[address] = key
        else:
            results[address] = outcome
    if missing:
        try:
            checked = equivalence.check({a: (plain[a], structured[a], abi) for a in missing}, work)
        except cppcheck.CompileError as exc:
            # A checker that cannot run has found nothing: say so loudly, remember
            # nothing, and let the forms stand on their verification this time.
            print("  WARNING: the equivalence check did not build, so it was skipped:\n    %s"
                  % "\n    ".join(str(exc).splitlines()[:5]))
            return list(addresses)
        for address, difference in checked.items():
            outcome = ("equivalent" if difference is None
                       else "differs from the plain form: case %d, %d" % difference)
            cache.put(missing[address], abi, outcome)
            results[address] = outcome
        cache.save()
    differing = sorted(a for a in addresses if results[a] != "equivalent")
    if differing:
        print("  %d differ from their plain form natively: %s"
              % (len(differing), ", ".join("0x%08X" % a for a in differing[:12])))
    return [a for a in addresses if results[a] == "equivalent"]


def structure_verified(chosen, functions, image, base, game, seeds, jobs, strings=None,
                       cache=None, machine=None) -> dict:
    """Swap in the structured form of every function whose structured form verifies too.

    Structuring (structure.py) is a second translation of the same code, so it is
    checked like the first: built into its own library and run on the same cases
    against the original. Where it fails or cannot be structured, the plain lifted
    form stays.
    """
    from ps2decomp import structure

    stage = time.time()
    result = dict(chosen)
    remaining = list(chosen)
    # Most readable first - temporaries folded, checked at the calling-convention
    # level - then structured with the callee-saved saves elided, then with them
    # kept; what no form passes keeps the lifted form.
    forms = (("folded", dict(elide=True, readable=True), True),
             ("with saves elided", dict(elide=True), False),
             ("with saves kept", dict(elide=False), False))
    for label, options, abi in forms:
        structured = {}
        for address in remaining:
            func = functions[address]
            try:
                structured[address] = structure.structure_function(
                    image, base, func["addr"], func["end"], strings=strings,
                    chunks=func.get("chunks", ()), **options)
            except lift.Refused:
                pass
        if not structured:
            break
        work = game.path(os.path.join(game.build, "structure_" + label.split()[-1]))
        outcomes = verify_texts(structured, work, game, seeds, jobs, cache, abi,
                                machine, functions)
        passed = [a for a, outcome in outcomes.items() if outcome == "verified"]
        passed = equivalent_to_plain(passed, structured, chosen, abi, cache, work + "_equivalence")
        for address in passed:
            result[address] = structured[address]
        remaining = [a for a in structured if a not in passed]
        print("structured %s: %d passed of %d tried (%.0fs)"
              % (label, len(passed), len(structured), time.time() - stage))
    done = sum(1 for a in chosen if result[a] is not chosen[a])
    print("structured: %d of %d verified functions" % (done, len(chosen)))
    return result


def structure_unverified(plain, functions, image, base, game, strings, cache) -> dict:
    """Structured forms of the functions the interpreter could not check.

    Their plain lifted form is a statement per instruction, the translation
    itself; a structured form rewrites it, so it is emitted only where it
    behaves as the plain form natively (equivalent_to_plain). An unchecked
    rewrite once elided the saves of a function whose `qmfc2 $s0` then leaked
    into its caller's $s0 (0x002284B0).
    """
    from ps2decomp import structure

    stage = time.time()
    result = dict(plain)
    remaining = list(plain)
    forms = (("folded", dict(elide=True, readable=True), True),
             ("with saves elided", dict(elide=True), False),
             ("with saves kept", dict(elide=False), False))
    for label, options, abi in forms:
        structured = {}
        for address in remaining:
            func = functions[address]
            try:
                structured[address] = structure.structure_function(
                    image, base, func["addr"], func["end"], strings=strings,
                    chunks=func.get("chunks", ()), **options)
            except lift.Refused:
                pass
        if not structured:
            break
        work = game.path(os.path.join(game.build, "unverified_" + label.split()[-1]))
        passed = equivalent_to_plain(list(structured), structured, plain, abi, cache, work)
        for address in passed:
            result[address] = structured[address]
        remaining = [a for a in structured if a not in passed]
        print("structured %s (not verified, equivalent to plain): %d of %d (%.0fs)"
              % (label, len(passed), len(structured), time.time() - stage))
    return result


def update_status(path: str, verified, outcomes, names=None, unverified=None) -> None:
    """Rebuild the `lifted` rows; any other row - derived, hand-written - stays.

    A function verified both ways keeps its derived row: readable source that is
    verified is the better tier, and the lifted copy is only the fallback. A
    derivation that is only `reconstructed` - its test near-vacuous - gives way
    to a verified lifted copy, which was tested on its whole state. A
    lifted row takes its name from the game's names.csv, where every name keeps
    its evidence, so rebuilding the rows never loses one.
    """
    import csv

    names = names or {}
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["source"] != "src/lifted/":
                    rows[int(row["address"], 16)] = row
    added = 0
    for address in verified:
        existing = rows.get(address)
        if existing and existing["status"] == "verified":
            continue
        rows[address] = {
            "address": "0x%08X" % address,
            "name": (names[address][0] if address in names else
                     existing["name"] if existing else "lifted::F_%08X" % address),
            "status": "lifted",
            "source": "src/lifted/",
            "notes": "lifted instruction by instruction; registers and memory "
                     "differential-tested against the original",
        }
        added += 1
    # Lifted but not checked: source exists and is believed right, which is what
    # `reconstructed` means - never `lifted`, which says it was verified.
    for address, reason in sorted((unverified or {}).items()):
        existing = rows.get(address)
        if existing and existing["status"] in ("verified", "reconstructed"):
            continue
        rows[address] = {
            "address": "0x%08X" % address,
            "name": (names[address][0] if address in names else
                     existing["name"] if existing else "lifted::F_%08X" % address),
            "status": "reconstructed",
            "source": "src/lifted/",
            "notes": "lifted; not verified: %s" % reason,
        }
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["address", "name", "status", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print("updated %s (%d lifted rows)" % (path, added))


# Each verifying process: its own interpreter, test memory and copy of the library.
_WORKER = None


def _other_sections(elf, code) -> list:
    """The ELF's other loaded sections with contents, for lift.use_sections."""
    return [(elf.section_data(s), s.addr) for s in elf.sections
            if s.is_alloc and s.size and s is not code and s.type != 8]       # 8: NOBITS


def _start_worker(game_directory, seeds, dll):
    global _WORKER
    game = game_config.load(game_directory)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    lift.use_sections(_other_sections(elf, section))
    lift.use_function_starts(f["addr"] for f in database["functions"])
    machine = Machine.from_elf(elf)
    testplan.prepare_test_memory(machine, database, elf.section_data(section), section.addr,
                               [], seeds=seeds)
    library = liftcheck.LiftedFunctions(dll, machine, testplan.VERIFY_BUDGET)
    _WORKER = (machine, library, {f["addr"]: f for f in database["functions"]})


def _verify_share(task):
    """One process verifies one share and exits: its mapped pages never leak into
    another share, so the result does not depend on scheduling."""
    game_directory, seeds, dll, addresses, abi = task
    _start_worker(game_directory, seeds, dll)
    machine, library, functions = _WORKER
    results = []
    for number, address in enumerate(addresses, 1):
        results.append((address, verify(functions[address], machine, library, abi)))
        # A heartbeat: a watchdog takes long silence for a hang (tools/watchdog.py).
        if number % 500 == 0:
            print("    a worker has verified %d of %d" % (number, len(addresses)), flush=True)
    return results


def verify_parallel(addresses, game, seeds, dll, jobs, abi=False) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    shares = [addresses[k::jobs] for k in range(jobs)]
    tasks = [(game.directory, seeds, dll, share, abi) for share in shares if share]
    outcomes = {}
    with ProcessPoolExecutor(max_workers=len(tasks), max_tasks_per_child=1) as pool:
        for results in pool.map(_verify_share, tasks):
            outcomes.update(results)
    return outcomes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", type=lambda text: int(text, 16), action="append",
                        help="lift just this function (hex address); repeatable")
    parser.add_argument("--emit", action="store_true",
                        help="write the verified functions to the game's src/lifted/")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="processes verifying in parallel (default: all but two cores)")
    parser.add_argument("--structure", action="store_true",
                        help="also emit verified functions in structured form "
                             "(loops, ifs, named locals) where that verifies too")
    parser.add_argument("--csv", action="store_true",
                        help="mark the verified functions `lifted` in the status table")
    parser.add_argument("--no-cache", action="store_true",
                        help="verify every function afresh instead of reusing the verdicts "
                             "of unchanged ones (the remembered verdicts are refreshed)")
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    lift.use_sections(_other_sections(elf, section))
    lift.use_function_starts(f["addr"] for f in database["functions"])
    machine = Machine.from_elf(elf)
    # Every function, including those with no return of their own - thunks that
    # end in a jump: a port reaches them like any other, and the original runs to
    # its caller's return through them, so they verify like any other.
    functions = [f for f in database["functions"] if f["end"] > f["addr"]]
    if args.only:
        functions = [f for f in functions if f["addr"] in set(args.only)]
    if args.limit:
        functions = functions[:args.limit]
    started = time.time()
    seeds = testplan.prepare_test_memory(machine, database, image, section.addr, functions,
                                       verbose=True)

    lifted, refused = lift_all(functions, image, section.addr)
    print("lifted %d of %d functions; refused %d" % (len(lifted), len(functions),
                                                     sum(refused.values())))
    for reason, count in refused.most_common(10):
        print("  %5d  %s" % (count, reason))

    # A verdict holds while the function's text and everything it is tested
    # against stay the same: the image, its analysis and the test memory's seeds
    # are part of that, with the code (verdicts.fingerprint).
    with open(game.database_path, "rb") as handle:
        analysis = handle.read()
    environment = verdicts.fingerprint(os.path.dirname(os.path.dirname(HERE)),
                                       (bytes(image), analysis, sorted(seeds)))
    cache = verdicts.Verdicts(game.path(os.path.join(game.build, "verdicts.json")),
                              environment, use=not args.no_cache)

    stage = time.time()
    by_address = {f["addr"]: f for f in functions}
    work = game.path(os.path.join(game.build, "lift"))
    outcomes = verify_texts(lifted, work, game, seeds, args.jobs, cache, False,
                            machine, by_address)
    # What does not compile is not emitted: it would break the game's build.
    lifted = {a: t for a, t in lifted.items() if outcomes[a] != NO_COMPILE}
    outcomes = {a: o for a, o in outcomes.items() if o != NO_COMPILE}
    print("  building and verifying took %.0fs (%d processes)"
          % (time.time() - stage, max(1, args.jobs)))
    import json
    with open(game.path(os.path.join(game.build, "lift_outcomes.json")), "w",
              encoding="utf-8") as handle:
        json.dump({"0x%08X" % a: outcome for a, outcome in sorted(outcomes.items())}, handle,
                  indent=0)

    verified = sorted(a for a, outcome in outcomes.items() if outcome == "verified")
    tally = collections.Counter(re.sub(r"0x[0-9A-Fa-f]+|\b\d+\b", "N", outcome)
                                for outcome in outcomes.values() if outcome != "verified")
    code_bytes = sum(f["size"] for f in database["functions"])
    verified_bytes = sum(by_address[a]["size"] for a in verified)
    print("lifted and verified: %d functions, %s bytes (%.2f%% of code)"
          % (len(verified), format(verified_bytes, ","), 100.0 * verified_bytes / code_bytes))
    for reason, count in tally.most_common(12):
        print("  %5d  %s" % (count, reason))
    disagreeing = [a for a, outcome in outcomes.items() if outcome.startswith("disagrees")]
    if disagreeing:
        print("  disagreeing: %s" % ", ".join("0x%08X" % a for a in disagreeing[:20]))
    chosen = {a: lifted[a] for a in verified}
    if args.structure:
        strings = {entry["addr"]: entry["text"] for entry in database["strings"]}
        chosen = structure_verified(chosen, by_address, image, section.addr, game, seeds,
                                    args.jobs, strings, cache, machine)
    # Everything else that lifted is emitted too, marked: a complete translation,
    # with each function saying whether it was checked.
    unverified = {a: outcomes[a] for a in lifted if a not in chosen}
    for address in unverified:
        chosen[address] = lifted[address]
    if args.structure and unverified:
        chosen.update(structure_unverified({a: lifted[a] for a in unverified}, by_address, image,
                                           section.addr, game, strings, cache))
    from ps2decomp import naming
    names = naming.load(game.path(os.path.join(os.path.dirname(game.status), "names.csv")))
    if args.emit:
        out = game.path(os.path.join(os.path.dirname(game.generated), "lifted"))
        paths = write_chunks(chosen, out, "lifted", unverified, names)
        print("wrote %d files to %s (%d functions, %d not verified)"
              % (len(paths), out, len(chosen), len(unverified)))
    if args.csv:
        update_status(game.path(game.status), verified, outcomes, names, unverified)
    print("  elapsed %.0fs" % (time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
