"""Test cases for checking a function against the original machine code.

How a function is tested - the test memory and its seeding, the arguments and
memory profile chosen, running the original and a candidate on each case, and
comparing what they did to memory - shared by derivation (engine/tools/derive.py)
and lifting (liftverify.py). It decides verdicts, so remembered lift verdicts are
keyed by it (verdicts.py); derivation itself is not part of that key.
"""

from __future__ import annotations

import bisect
import hashlib
import random
import re
import time

from . import layout, structs, testmem

# Instructions one verification case may execute, callees included. Interpreting
# is slow, and a function that calls into a large original routine would otherwise
# spend minutes per case; one that needs more than this is reported untestable.
VERIFY_BUDGET = 100_000
# The callee's own frame. Writes here are not observable effects, so they are
# excluded when comparing what the original and the derivation did to memory.
# The stack, scratch memory and the rest of the layout come from the game; see
# ps2decomp/layout.py.
ARG_SETS = 8

# Why the last test that could not be run failed: (stage, exception). Set wherever
# a test run is abandoned, and read by _attempt to explain an untestable verdict.
_LAST_FAILURE = None
_HEX = re.compile(r"0x[0-9A-Fa-f]+|\b\d+\b")


def _failed(stage, exc) -> None:
    global _LAST_FAILURE
    _LAST_FAILURE = (stage, exc)


def failure_reason() -> str:
    """The last failure as a category: numbers and addresses blanked so that
    failures of one kind count together."""
    if _LAST_FAILURE is None:
        return "unknown"
    stage, exc = _LAST_FAILURE
    message = _HEX.sub("N", str(exc).splitlines()[0] if str(exc) else "")
    return "%s: %s: %s" % (stage, type(exc).__name__, message[:90])


_GRAPHS = {}


def recovered_graph(database, image, base):
    """The struct-type graph for this binary, recovered once per run.

    It is needed to seed global pointers before testing and again to type the
    emitted source; recovering it twice was a fifth of a whole run.
    """
    key = (id(database), base)
    if key not in _GRAPHS:
        _GRAPHS[key] = structs.recover(database, image, base)
    return _GRAPHS[key]


def prepare_test_memory(machine, database, image, base, functions, verbose=False,
                        seeds=None) -> list:
    """Fill scratch and seed null global pointers, identically for every tool.

    Global pointers that are null until the game runs would make every function
    using them untestable, so they are pointed at scratch: first the ones the
    recovered types mark as pointers, then the ones found by running the
    functions (see testmem.discover_null_pointers).

    Returns the addresses seeded. Passing them back as `seeds` reproduces the same
    memory without the discovery - how a worker process matches its parent.
    """
    rng = random.Random(0xB3)
    machine.memory.write(layout.CURRENT.scratch,
                         bytes(rng.randrange(256) for _ in range(layout.CURRENT.scratch_size)))
    testmem.fill_caller_frame(machine.memory)
    if seeds is not None:
        testmem.seed_global_pointers(machine.memory, seeds)
        machine.memory.page_source = testmem.page_contents
        machine.map_faults = True
        return seeds
    static = global_pointers(database, image, base)
    seeded = testmem.seed_global_pointers(machine.memory, static)
    spans = sorted((f["addr"], f["end"]) for f in database["functions"])
    starts = [start for start, _end in spans]
    scratch_low = layout.CURRENT.scratch
    scratch_high = scratch_low + layout.CURRENT.scratch_size

    def is_global(address):
        if not base <= address < machine.memory.base + machine.memory.size:
            return False
        if scratch_low <= address < scratch_high:
            return False
        index = bisect.bisect_right(starts, address) - 1
        return index < 0 or address >= spans[index][1]

    started = time.time()
    found = testmem.discover_null_pointers(machine, functions, is_global)
    if verbose:
        print("  seeded %d null global pointers for testing, and %d more found by "
              "running (%.0fs)" % (seeded, len(found), time.time() - started))
    # From here on a test that faults on unmapped memory maps the page and runs
    # again. Only after the discovery above, which finds null pointers by their
    # faults.
    machine.memory.page_source = testmem.page_contents
    machine.map_faults = True
    return sorted(set(static) | set(found))


def global_pointers(database, image, base) -> set:
    """Addresses of global variables that some code dereferences as a pointer."""
    graph = recovered_graph(database, image, base)
    if structs.GLOBALS not in graph.ids:
        return set()
    fields = graph.fields[graph.find(graph.ids[structs.GLOBALS])]
    return {address & 0xFFFFFFFF for address, field in fields.items()
            if field.target is not None and 4 in field.widths()}


def _effects_digest(memory) -> bytes:
    """A function's net memory effects, reduced to a digest for comparison.

    Only equality is ever asked of the effects in a test, and the original's are
    kept for every verified function so the compiled check can reuse them. As
    dictionaries, a function filling a large buffer cost about a hundred bytes per
    byte written, eight cases over, for every such function - enough to exhaust
    memory once seeded globals let those functions run to completion.
    """
    effects = _journal_effects(memory)
    digest = hashlib.sha1()
    for position in sorted(effects):
        digest.update((position & 0xFFFFFFFF).to_bytes(4, "little"))
        digest.update(bytes((effects[position],)))
    return digest.digest()


def _journal_effects(memory) -> dict:
    """Collapse a write journal into a net address -> byte map.

    Keyed by address, not by the journal's buffer positions: a page mapped on
    demand sits at a different position in every process that mapped pages in a
    different order, and effects recorded in one process are compared in another
    (the parallel derivation's compiled check).

    Comparing journals directly would be wrong: the original and the derivation may
    write the same bytes in a different order, or write one location twice. What has
    to match is the resulting memory, so the journal is reduced to final values.

    Writes into the callee's own stack frame are excluded. They are not observable
    effects - the frame is dead the moment the function returns - and the original
    makes plenty of them spilling registers while a derived expression makes none,
    so counting them would fail every function that has a prologue.
    """
    frame_low = layout.CURRENT.stack_low - memory.base
    frame_high = layout.CURRENT.stack_top + 0x10 - memory.base
    effects = {}
    for offset, previous in getattr(memory, "_journal", None) or ():
        for index in range(len(previous)):
            position = offset + index
            if frame_low <= position <= frame_high:
                continue
            effects[position] = memory.bytes[position]
    return {memory.address_at(position): value for position, value in effects.items()}


def _choose_profile(machine, func, args):
    """Return the first memory profile under which the original executes, or None.

    The profile only decides whether the test can *reach* the function; the original
    and the derivation always read the same memory, so it never changes a verdict.
    """
    for profile in testmem.PROFILES:
        # Pages a failing profile mapped are given back: they would otherwise use
        # up the mapping budget (emu.MAX_PAGES), and the functions tested later
        # would fault for want of a page - a loss that depends on the order.
        mark = machine.memory.page_mark()
        machine.memory.end_journal()
        machine.memory.write(layout.CURRENT.scratch, testmem.build(profile, func["addr"]))
        machine.memory.begin_journal()
        try:
            machine.call(func["addr"], args, max_instructions=VERIFY_BUDGET)
            ran = True
        except Exception as exc:
            _failed("original, every memory profile", exc)
            ran = False
        finally:
            machine.memory.rollback()
            machine.memory.end_journal()
        if ran:
            return profile
        machine.memory.unmap_to(mark)
    return None


def test_plan(func, machine):
    """The argument sets and memory profile a function is tested with, or None.

    Separate from the test itself so that anything else that must face exactly the
    same cases - the compiled C++ check - can ask for them.
    """
    # Seeded from the address so argument sets are stable across passes: a
    # function retried in a later pass must be tested on the same inputs, or a
    # pass-dependent result would be impossible to reproduce.
    local_rng = random.Random(func["addr"])
    arg_sets = [testmem.arguments(local_rng) for _ in range(ARG_SETS - 1)]
    arg_sets.append(tuple(layout.CURRENT.scratch + 16 * i for i in range(8))
                    + tuple(testmem.float_bits(local_rng) for _ in range(8)))
    profile = _choose_profile(machine, func, arg_sets[0])
    if profile is None:
        return None
    return arg_sets, profile


def run_cases(func, machine, arg_sets, profile, evaluate, record=None, recorded=None,
              float_result=False):
    """Run the original and a candidate on every case; return (outcome, tested).

    `evaluate(args)` returns the candidate's (result, writes) for one case, with
    memory as the original saw it on entry; result is None for a void function and
    writes are (address, value, size) triples. The outcome is "ok", "mismatch" or
    "failed" (the case could not be run at all).

    `record`, a list, receives the original's (result, effects) for each case;
    `recorded` supplies them instead of running the original again. The compiled
    check uses the results recorded while the derivation was verified: they are
    the same runs on the same memory, and repeating them doubled its cost.
    """
    tested = 0
    for case, args in enumerate(arg_sets):
        # Fresh memory of the chosen shape for every case, so the test varies
        # memory as well as arguments. Written outside the journal on purpose:
        # the journal does not nest, and this region is rewritten before every
        # case anyway, so nothing leaks between them.
        machine.memory.end_journal()
        machine.memory.write(layout.CURRENT.scratch,
                             testmem.build(profile, func["addr"] + case))
        # A function with stores is only verified if its *memory effects* match
        # too, so let the original run under a journal, and then check that
        # applying the candidate's stores to the same starting state reproduces
        # exactly what the original produced.
        # Journalling rather than snapshotting a region: these functions write
        # to globals well outside the scratch area, and restoring only scratch
        # left those changes in place - which both corrupted later cases and
        # made correct derivations look wrong.
        if recorded is not None:
            expected, effects = recorded[case]
        else:
            machine.memory.begin_journal()
            try:
                expected = machine.call(func["addr"], args, max_instructions=VERIFY_BUDGET)
                if float_result:
                    # A function returning a float returns it in $f0.
                    expected = machine.fpr[0] & 0xFFFFFFFF
            except Exception as exc:
                _failed("original", exc)
                machine.memory.rollback()
                machine.memory.end_journal()
                return "failed", tested
            effects = _effects_digest(machine.memory)
            machine.memory.rollback()
            if record is not None:
                record.append((expected, effects))

        # The candidate runs inside the replay journal: one that calls an underived
        # function writes memory as it goes - the callee must see the stores made
        # before it - rather than handing back a list of writes at the end.
        machine.memory.begin_journal()
        try:
            actual, writes = evaluate(args)
        except Exception as exc:
            _failed("derivation", exc)
            machine.memory.rollback()
            machine.memory.end_journal()
            return "failed", tested
        try:
            for address, value, size in writes:
                machine.memory.write_int(address, value, size)
        except Exception as exc:
            _failed("derivation's stores", exc)
            machine.memory.rollback()
            machine.memory.end_journal()
            return "failed", tested
        replayed = _effects_digest(machine.memory)
        machine.memory.rollback()
        machine.memory.end_journal()

        tested += 1
        # A void function leaves whatever the caller had in $v0, so there is no
        # return value to compare - only its memory effects carry meaning.
        # Comparing $v0 anyway registered 175 correct derivations as failures.
        if actual is not None and float_result:
            if expected != actual & 0xFFFFFFFF:
                return "mismatch", tested
        elif actual is not None and expected != actual:
            return "mismatch", tested
        if replayed != effects:
            return "mismatch", tested
    return "ok", tested


def reset_failure() -> None:
    global _LAST_FAILURE
    _LAST_FAILURE = None
