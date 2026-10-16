#!/usr/bin/env python3
"""Derive, emit, compile and test one function, showing where its forms disagree.

The full pipeline reports a function whose compiled form disagrees with the
original and drops it. This takes one such function on its own: it prints the
emitted C++ and, for each test case, what the original, the model and the
compiled code produced, so a rendering bug can be found directly.

    python engine/tools/inspect_function.py 0x001309C0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import derive
from ps2decomp import game as game_config
from ps2decomp import calls, cppcheck, layout, testmem, testplan
from ps2decomp.elf import ElfFile
from ps2decomp.emu import Machine
from ps2decomp.paths import Execution, derive_paths
from ps2decomp.symexec import Unsupported


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("address", type=lambda text: int(text, 16))
    args = parser.parse_args(argv)

    game = game_config.from_args(args)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    machine = Machine.from_elf(elf)
    parameters = calls.parameters(database, image, section.addr)
    derive.prepare_test_memory(machine, database, image, section.addr, [
        f for f in database["functions"]
        if f["size"] // 4 <= derive.MAX_INSTRUCTIONS and f["has_return"]])
    func = next(f for f in database["functions"] if f["addr"] == args.address)

    try:
        derivation = derive_paths(image, section.addr, func["addr"], func["end"],
                                  known={}, parameters=parameters)
    except Unsupported:
        derivation = derive_paths(image, section.addr, func["addr"], func["end"],
                                  known={}, parameters=parameters, loops=True)

    work = os.path.join(tempfile.gettempdir(), "b3_inspect")
    derive.emit([(func, derivation, 8, True)], work, "inspect")
    source = os.path.join(work, "accessors.cpp")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    print(text[text.index("/* 0x%08X" % func["addr"]):].split("\n}  // namespace")[0])

    dll, _functions = cppcheck.build(source, cppcheck.RUNTIME_SOURCES,
                                     os.path.join(cppcheck.RUNTIME, "include"), work)
    compiled = cppcheck.CompiledFunctions(dll, machine)
    arg_sets, profile = derive.test_plan(func, machine)
    for case, case_args in enumerate(arg_sets):
        outcomes = []
        for name in ("original", "model", "compiled"):
            machine.memory.end_journal()
            machine.memory.write(layout.CURRENT.scratch,
                                 testmem.build(profile, func["addr"] + case))
            machine.memory.begin_journal()
            try:
                if name == "original":
                    value = machine.call(func["addr"], case_args,
                                         max_instructions=derive.VERIFY_BUDGET)
                    if derivation.returns_float:
                        value = machine.fpr[0] & 0xFFFFFFFF       # returned in $f0
                elif name == "model":
                    value = (derivation.run(case_args, Execution(machine, derive.VERIFY_BUDGET))
                             if derivation.sequential
                             else derivation.evaluate(case_args, lambda a, s, g:
                                                      machine.memory.read_int(a, s, g)))
                    if not derivation.sequential:
                        for address, data, size in derivation.evaluate_stores(
                                case_args, lambda a, s, g: machine.memory.read_int(a, s, g)):
                            machine.memory.write_int(address, data, size)
                else:
                    value = compiled.evaluator(func["addr"])(case_args)[0]
                if derivation.returns_float and isinstance(value, int):
                    value &= 0xFFFFFFFF
                effects = testplan._journal_effects(machine.memory)
            except Exception as exc:
                value, effects = "error: %s" % exc, {}
            machine.memory.rollback()
            outcomes.append((name, value, effects))
        (_n, v0, e0), (_m, v1, e1), (_c, v2, e2) = outcomes
        if v0 == v1 == v2 and e0 == e1 == e2:
            continue
        print("case %d:" % case)
        for name, value, effects in outcomes:
            print("  %-8s returned %s" % (name, value))
        for name, effects in (("model", e1), ("compiled", e2)):
            diff = sorted((hex(k + machine.memory.base), e0.get(k), effects.get(k))
                          for k in set(e0) | set(effects) if e0.get(k) != effects.get(k))
            if diff:
                print("  %-8s memory differs at %d byte(s), first: %s" % (name, len(diff), diff[:4]))
    machine.memory.end_journal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
