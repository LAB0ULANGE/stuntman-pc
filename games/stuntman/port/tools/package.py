#!/usr/bin/env python3
"""Package a release: games/stuntman/build/release/Stuntman-PC-<version>-windows-x64.zip

    python engine/tools/port.py --game games/stuntman build
    python games/stuntman/port/tools/package.py

The zip holds one folder, Stuntman-PC: the executable, the player's README
(port/release/README.txt, its {version} filled in from game.json), the
starting settings (port/release/pc_settings.ini) and the libraries' licences
(THIRD-PARTY-NOTICES.txt, from the build's own copies of them). No game
content: the player brings their own disc image.

It refuses a build that must not be published:
  - one with a player's icon made from their disc (build/port/icon.ico),
  - one that needs the Visual C++ runtime installed (it is linked in, /MT),
  - one that names a folder of the machine it was built on.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
GAME = os.path.dirname(PORT)
ROOT = os.path.dirname(os.path.dirname(GAME))
BUILD = os.path.join(GAME, "build")
EXE = os.path.join(BUILD, "port", "cmake", "stuntman.exe")
DEPS = os.path.join(BUILD, "port", "cmake", "_deps")

# (name in the notices, the licence file in the build's copy of it)
LIBRARIES = [
    ("SDL 3", "sdl3-src/LICENSE.txt"),
    ("volk", "volk-src/LICENSE.md"),
    ("Vulkan-Headers", "vulkan_headers-src/LICENSE.md"),
    ("glslang", "glslang-src/LICENSE.txt"),
]


def refusals(exe: bytes) -> list[str]:
    found = []
    if os.path.exists(os.path.join(BUILD, "port", "icon.ico")):
        found.append("build/port/icon.ico is there: the executable carries an icon made from a disc "
                     "(take it out and build again: port/icon.ico is the port's own)")
    for runtime in (b"VCRUNTIME140", b"MSVCP140", b"VCRUNTIME140_1"):
        if runtime.lower() + b".dll" in exe.lower():
            found.append("it needs %s.dll: the runtime must be linked in (/MT)" % runtime.decode())
    folders = {ROOT, os.path.expanduser("~")}
    for folder in folders:
        for text in (folder, folder.replace("\\", "/")):
            for encoded in (text.encode("utf-8"), text.encode("utf-16-le")):
                if encoded.lower() in exe.lower():
                    found.append("it names a folder of this machine: %s" % text)
    return sorted(set(found))


def notices() -> str:
    parts = ["Third-party software built into the Stuntman PC port\n"
             "=====================================================\n"]
    for name, path in LIBRARIES:
        full = os.path.join(DEPS, path)
        if not os.path.exists(full):
            sys.exit("package: %s's licence is missing (%s): build first" % (name, full))
        with open(full, encoding="utf-8", errors="replace") as f:
            parts.append("\n\n%s\n%s\n\n%s" % (name, "-" * len(name), f.read().strip()))
    return "\n".join(parts) + "\n"


def main() -> int:
    with open(os.path.join(GAME, "game.json"), encoding="utf-8") as f:
        version = json.load(f)["version"]
    if not os.path.exists(EXE):
        sys.exit("package: no %s: build first (engine/tools/port.py --game games/stuntman build)" % EXE)
    with open(EXE, "rb") as f:
        exe = f.read()
    refused = refusals(exe)
    if refused:
        for reason in refused:
            print("package: refused: " + reason, file=sys.stderr)
        return 1

    with open(os.path.join(PORT, "release", "README.txt"), encoding="utf-8") as f:
        readme = f.read().replace("{version}", version)
    with open(os.path.join(PORT, "release", "pc_settings.ini"), encoding="utf-8") as f:
        settings = f.read()

    out_dir = os.path.join(BUILD, "release")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Stuntman-PC-%s-windows-x64.zip" % version)
    crlf = lambda text: text.replace("\r\n", "\n").replace("\n", "\r\n")   # Notepad's
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(EXE, "Stuntman-PC/stuntman.exe")
        z.writestr("Stuntman-PC/README.txt", crlf(readme))
        z.writestr("Stuntman-PC/pc_settings.ini", crlf(settings))
        z.writestr("Stuntman-PC/THIRD-PARTY-NOTICES.txt", crlf(notices()))
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
