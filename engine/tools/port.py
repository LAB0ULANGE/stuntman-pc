#!/usr/bin/env python3
"""Build a game's native port and run it.

    python engine/tools/port.py --game games/<name>            # generate, build, run
    python engine/tools/port.py --game games/<name> build      # generate and build only
    python engine/tools/port.py --game games/<name> run -- --trace
    python engine/tools/port.py init <name> --iso "<disc image>" --title "..."  # a new game

The port is the game's translated functions (src/lifted, from lift.py --emit) on
the native host (engine/host): guest memory, a dispatch table from guest address
to translated function, the EE kernel's services. This generates the dispatch
table, builds games/<name>/port/ with CMake and Ninja under MSVC, and runs the
result on the executable from the user's own disc, printing how far it got.

Some code only exists at run time: kept as data and copied out - a kernel
extension, an overlay - and run where it was copied to. The host records each
such call it meets with the code's source in the executable (--discover); this
translates those at their run addresses and builds again, until a run finds
nothing new. They are translated, not verified: nothing tests them against the
interpreter, which never sees them run.

Nothing from the disc is built into the port: the executable is loaded at run
time, from the game's extracted/ directory.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# The host stops itself at its --timeout (30 s by default); this long past it,
# it is stuck.
HOST_TIMEOUT_SECONDS = 30
HOST_GRACE_SECONDS = 90
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import cppcheck, gaps, lift, naming  # noqa: E402
from ps2decomp import game as game_config  # noqa: E402
import vurecomp  # noqa: E402

_FUNCTION = re.compile(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{$", re.M)


def translated_functions(directory: str) -> list:
    """Addresses of every translated function in the game's src/lifted."""
    found = set()
    for name in sorted(os.listdir(directory)):
        if name.endswith(".cpp"):
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                found.update(int(m.group(1), 16) for m in _FUNCTION.finditer(handle.read()))
    return sorted(found)


def dispatch_source(addresses, names: dict, untranslated=(), sizes=None) -> str:
    """The dispatch table: every translated function, by address, with its name
    where one is known; functions with no translation are listed with no code, so
    a call to one is reported by name."""
    entries = sorted(set(addresses) | set(untranslated))
    lines = ['#include "ps2host/host.h"', ""]
    lines += ["void F_%08X(lifted::Context& c);" % a for a in addresses]
    lines += ["", "extern const ps2host::FunctionEntry kGuestFunctions[] = {"]
    present = set(addresses)
    for address in entries:
        name = names.get(address)
        label = '"%s"' % name.replace('"', "") if name else "nullptr"
        code = "F_%08X" % address if address in present else "nullptr"
        lines.append("    {0x%08Xu, %s, %s, %du}," % (address, code, label,
                                                   (sizes or {}).get(address, 0)))
    lines += ["};", "extern const std::size_t kGuestFunctionCount = %d;" % len(entries), ""]
    return "\n".join(lines)


def discovered_functions(game, path: str) -> dict:
    """{run address: C++} for the code runs found copied at run time, translated
    at their run addresses from the executable's bytes."""
    if not os.path.exists(path):
        return {}
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    # A run is lifted from a slice of the image; its jump tables lie elsewhere
    # in it (Burnout 3's 0x00242AA0 reaches 0x00242BD0 through one).
    lift.use_sections([(image, section.addr)] + [(elf.section_data(s), s.addr) for s in elf.sections
                                                  if s.is_alloc and s.size and s is not section and s.type != 8])
    found = {}
    with open(path, encoding="utf-8") as handle:
        pairs = {(int(run, 16), int(source, 16)) for run, source in
                 (line.split() for line in handle if line.strip())}
    for run, source in sorted(pairs):
        offset = source - section.addr
        code = image[offset:offset + gaps.MAX_FUNCTION]
        try:
            length = gaps.function_extent(code, run, run)
            found[run] = (source, lift.lift_function(code, run, run, run + length))
        except (ValueError, lift.Refused) as exc:
            print("  cannot translate the code at 0x%08X (from 0x%08X): %s" % (run, source, exc))
    return found


def generate(game) -> str:
    """Write the dispatch table; returns its directory."""
    lifted = game.path(os.path.join(os.path.dirname(game.generated), "lifted"))
    if not os.path.isdir(lifted):
        raise SystemExit("no translated sources in %s: run lift.py --emit first" % lifted)
    addresses = translated_functions(lifted)
    database = game.load_database()
    status_dir = os.path.dirname(game.status)
    names = {a: v[0] for a, v in naming.load(game.path(os.path.join(status_dir, "names.csv"))).items()}
    untranslated = [f["addr"] for f in database["functions"]
                    if f["size"] and f["addr"] not in set(addresses)]
    out = game.path(os.path.join(game.build, "port", "generated"))
    os.makedirs(out, exist_ok=True)
    discovered = discovered_functions(game, game.path(os.path.join(game.build, "port",
                                                                   "discovered.txt")))
    # What the analysis has since found is translated - and verified - with the rest.
    discovered = {a: v for a, v in discovered.items() if a not in set(addresses)}
    copies = ['#include "ps2rt/lifted.h"', "", "using namespace generated;",
              "using namespace lifted;", "",
              "/* Code found copied at run time, translated at its run address. NOT VERIFIED. */",
              ""]
    for run, (source, text) in sorted(discovered.items()):
        if source == run:
            copies += ["/* 0x%08X: reached at run time; no function in the analysis starts "
                       "here */" % run, text]
            names.setdefault(run, "found at run time")
        else:
            copies += ["/* 0x%08X: a copy of the executable's code at 0x%08X */" % (run, source),
                       text]
            names.setdefault(run, "copy of 0x%08X" % source)
    _write_if_changed(os.path.join(out, "discovered.cpp"), "\n".join(copies) + "\n")
    path = os.path.join(out, "dispatch.cpp")
    sizes = {f["addr"]: f["size"] for f in database["functions"]}
    text = dispatch_source(sorted(set(addresses) | set(discovered)), names, untranslated, sizes)
    _write_if_changed(path, text)
    print("dispatch table: %d translated functions, %d found at run time, %d without a "
          "translation" % (len(addresses), len(discovered), len(untranslated)))
    # The microprograms runs recorded (--vu-record), compiled ahead of time.
    vurecomp.main(["--manifest", vu_manifest_path(game), "--out-dir", out])
    return out


def vu_manifest_path(game) -> str:
    """Where runs record the microprograms they start (--vu-record): game-derived, in build/."""
    return game.path(os.path.join(game.build, "vu_programs.txt"))


def _write_if_changed(path: str, text: str) -> None:
    """Leave an unchanged file untouched, so the build need not recompile it."""
    old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
    if text != old:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def _visual_studio_tool(*parts) -> str:
    vcvars = cppcheck.find_vcvars()
    root = os.path.abspath(os.path.join(os.path.dirname(vcvars), "..", "..", ".."))
    return os.path.join(root, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", *parts)


def build(game, generated: str) -> str:
    """Configure and build the port; returns the executable's path."""
    source = game.path("port")
    binary = game.path(os.path.join(game.build, "port", "cmake"))
    os.makedirs(binary, exist_ok=True)
    cmake = _visual_studio_tool("CMake", "bin", "cmake.exe")
    ninja = _visual_studio_tool("Ninja", "ninja.exe")
    script = os.path.join(binary, "build.bat")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write('@echo off\ncall "%s" >nul\n' % cppcheck.find_vcvars())
        handle.write('"%s" -S "%s" -B "%s" -G Ninja -DCMAKE_BUILD_TYPE=Release '
                     '-DCMAKE_MAKE_PROGRAM="%s" -DPS2_GENERATED_DIR="%s" >nul || exit /b 1\n'
                     % (cmake, source, binary, ninja, generated.replace("\\", "/")))
        handle.write('"%s" --build "%s" || exit /b 1\n' % (cmake, binary))
    result = subprocess.run(["cmd", "/c", script], capture_output=True, text=True)
    if result.returncode:
        errors = [line for line in (result.stdout + result.stderr).splitlines()
                  if "error" in line.lower()]
        raise SystemExit("port build failed:\n" + "\n".join(errors[:40] or
                                                           (result.stdout + result.stderr)[-4000:].splitlines()))
    tests = os.path.join(binary, "ps2host", "ps2host_tests.exe")
    result = subprocess.run([tests], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode:
        raise SystemExit("the host's own tests failed")
    target = os.path.basename(os.path.normpath(game.directory))
    executable = os.path.join(binary, target + ".exe")
    print("built %s" % executable)
    return executable


def discoveries_path(game) -> str:
    return game.path(os.path.join(game.build, "port", "discovered.txt"))


def run(game, executable: str, extra) -> int:
    elf = game.path(os.path.join(game.extracted, game.executable))
    if not os.path.exists(elf):
        raise SystemExit("no executable at %s: extract it from your disc first (isotool.py)" % elf)
    report = game.path(os.path.join(game.build, "port", "report.txt"))
    with open(report, "w", encoding="utf-8") as out:
        disc = game.path(game.iso) if game.iso else ""
        options = ["--disc", disc] if disc and os.path.exists(disc) else []
        timeout = HOST_TIMEOUT_SECONDS
        extra = list(extra)
        if "--timeout" in extra and extra.index("--timeout") + 1 < len(extra):
            timeout = int(extra[extra.index("--timeout") + 1])
        try:
            result = subprocess.run([executable, elf, "--discover", discoveries_path(game)]
                                    + options + list(extra), stdout=out, stderr=subprocess.STDOUT,
                                    timeout=timeout + HOST_GRACE_SECONDS)
            code = result.returncode
        except subprocess.TimeoutExpired:
            # The host stops itself at its own time limit; outliving it by this
            # much means the host itself is stuck, which is a host bug to report.
            out.write("\n=== host report ===\nkilled: the host outlived its own time limit by "
                      "%ds - a bug in the host, not the game\n" % HOST_GRACE_SECONDS)
            code = 3
    with open(report, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    print(text[text.find("=== host report ==="):] if "=== host report ===" in text else text[-3000:])
    print("full log: %s" % report)
    return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["init"]:
        # A new game: there is no game directory yet for --game to name.
        import init_game
        return init_game.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="all", choices=("all", "build", "run"))
    parser.add_argument("--rounds", type=int, default=60,
                        help="build-and-run rounds to translate code found at run time")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="options for the port itself, after --")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    extra = [a for a in args.extra if a != "--"]
    if args.action == "run":
        target = os.path.basename(os.path.normpath(game.directory))
        executable = game.path(os.path.join(game.build, "port", "cmake", target + ".exe"))
        return run(game, executable, extra)
    if args.action == "build":
        build(game, generate(game))
        return 0
    # Build and run until a run finds no more code copied at run time.
    for _round in range(args.rounds):
        executable = build(game, generate(game))
        before = _count_lines(discoveries_path(game))
        code = run(game, executable, extra)
        if _count_lines(discoveries_path(game)) == before:
            return code
        print("found code copied at run time: translating it and building again\n")
    return code


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    # Distinct entries: code that cannot be translated is met - and recorded -
    # again on every run, and must not count as something new.
    with open(path, encoding="utf-8") as handle:
        return len({line.strip() for line in handle if line.strip()})


if __name__ == "__main__":
    raise SystemExit(main())
