#!/usr/bin/env python3
"""Package a release: games/stuntman/build/release/Stuntman-PC-<version>-windows-x64.zip

    python engine/tools/port.py --game games/stuntman build
    python games/stuntman/port/tools/package.py

or, on Linux (the port built there, games/stuntman/build/port/cmake/stuntman),
Stuntman-PC-<version>-linux-x64.tar.gz:

    python3 games/stuntman/port/tools/package.py --linux

The zip holds one folder, Stuntman-PC: the executable, the player's README
(port/release/README.txt, its {version} filled in from game.json), the
starting settings (port/release/pc_settings.ini) and the libraries' licences
(THIRD-PARTY-NOTICES.txt, from the build's own copies of them). No game
content: the player brings their own disc image.

It refuses a build that must not be published:
  - one with a player's icon made from their disc (build/port/icon.ico),
  - one that needs the Visual C++ runtime installed (it is linked in, /MT),
  - one that names a folder of the machine it was built on.
The Linux one holds the same (the README port/release/README-linux.txt) and
refuses the same, the runtime there the C++ one (libstdc++, libgcc_s: linked
in) or SDL's; it says the oldest C library (glibc) the executable runs on.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
GAME = os.path.dirname(PORT)
ROOT = os.path.dirname(os.path.dirname(GAME))
BUILD = os.path.join(GAME, "build")
EXE = os.path.join(BUILD, "port", "cmake", "stuntman.exe")
LINUX_EXE = os.path.join(BUILD, "port", "cmake", "stuntman")
DEPS = os.path.join(BUILD, "port", "cmake", "_deps")

# (name in the notices, the licence file in the build's copy of it)
LIBRARIES = [
    ("SDL 3", "sdl3-src/LICENSE.txt"),
    ("volk", "volk-src/LICENSE.md"),
    ("Vulkan-Headers", "vulkan_headers-src/LICENSE.md"),
    ("glslang", "glslang-src/LICENSE.txt"),
]


def refusals(exe: bytes, linux: bool = False) -> list[str]:
    found = []
    if linux:
        for runtime in (b"libstdc++.so", b"libgcc_s.so", b"libSDL3.so"):
            if runtime in exe:
                found.append("it needs %s: it must be linked in" % runtime.decode())
    elif os.path.exists(os.path.join(BUILD, "port", "icon.ico")):
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


def glibc_needed(exe: bytes) -> str:
    """The newest GLIBC_x.y version the executable asks for: the oldest C
    library it runs on."""
    versions = {tuple(int(n) for n in m.groups() if n is not None)
                for m in re.finditer(rb"GLIBC_(\d+)\.(\d+)(?:\.(\d+))?", exe)}
    return ".".join(str(n) for n in max(versions)) if versions else "?"


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


def package_linux(version: str) -> int:
    if not os.path.exists(LINUX_EXE):
        sys.exit("package: no %s: build the port on Linux first" % LINUX_EXE)
    with open(LINUX_EXE, "rb") as f:
        exe = f.read()
    if exe[:4] != b"\x7fELF":
        sys.exit("package: %s is not a Linux executable" % LINUX_EXE)
    refused = refusals(exe, linux=True)
    if refused:
        for reason in refused:
            print("package: refused: " + reason, file=sys.stderr)
        return 1
    glibc = glibc_needed(exe)
    with open(os.path.join(PORT, "release", "README-linux.txt"), encoding="utf-8") as f:
        readme = f.read().replace("{version}", version).replace("{glibc}", glibc)
    with open(os.path.join(PORT, "release", "pc_settings.ini"), encoding="utf-8") as f:
        settings = f.read()
    third_party = notices()                 # (before the archive: none left half written)

    out_dir = os.path.join(BUILD, "release")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Stuntman-PC-%s-linux-x64.tar.gz" % version)
    now = int(time.time())

    def add(tar: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
        info = tarfile.TarInfo("Stuntman-PC/" + name)
        info.size, info.mode, info.mtime = len(data), mode, now
        info.uname = info.gname = ""
        tar.addfile(info, io.BytesIO(data))

    lf = lambda text: text.replace("\r\n", "\n").encode("utf-8")
    with tarfile.open(out, "w:gz", compresslevel=9, format=tarfile.USTAR_FORMAT) as tar:
        folder = tarfile.TarInfo("Stuntman-PC")
        folder.type, folder.mode, folder.mtime = tarfile.DIRTYPE, 0o755, now
        tar.addfile(folder)
        add(tar, "stuntman", exe, 0o755)
        add(tar, "README.txt", lf(readme), 0o644)
        add(tar, "pc_settings.ini", lf(settings), 0o644)
        add(tar, "THIRD-PARTY-NOTICES.txt", lf(third_party), 0o644)
    print("wrote %s (%.1f MB); the executable needs glibc %s or newer" % (out, os.path.getsize(out) / 1e6, glibc))
    return 0


def main() -> int:
    with open(os.path.join(GAME, "game.json"), encoding="utf-8") as f:
        version = json.load(f)["version"]
    if "--linux" in sys.argv[1:]:
        return package_linux(version)
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
