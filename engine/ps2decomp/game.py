"""A game: where its files are, and the facts about it the engine needs.

The engine is generic; a game is described by a `game.json` in its own directory:

    {
      "name": "Burnout 3: Takedown",
      "iso": "../../Burnout 3 - Takedown.iso",
      "executable": "SLUS_210.50",
      "database": "build/slus_210_50.json",
      "generated": "src/generated",
      "status": "config/functions.csv",
      "layout": {"scratch": "0x01800000"}
    }

Paths are relative to the game's directory. Everything has a default except the
executable's name, and the global pointer is detected from the program's entry
code rather than written down: it differs between games, and the startup code
always establishes it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct

from . import layout as layout_module
from .elf import ElfFile
from .r5900 import DISASM

DEFAULTS = {
    "iso": None,
    "extracted": "extracted",
    "build": "build",
    "database": "build/functions.json",
    "generated": "src/generated",
    "status": "config/functions.csv",
    "layout": {},
    "vu_regions": [],
}


@dataclasses.dataclass
class Game:
    directory: str
    name: str
    executable: str
    iso: str
    extracted: str
    build: str
    database: str
    generated: str
    status: str
    layout: dict
    vu_regions: list

    def path(self, relative: str) -> str:
        return os.path.normpath(os.path.join(self.directory, relative))

    @property
    def elf_path(self) -> str:
        return os.path.join(self.path(self.extracted), self.executable)

    @property
    def database_path(self) -> str:
        return self.path(self.database)

    def load_elf(self) -> ElfFile:
        return ElfFile.load(self.elf_path)

    def load_database(self) -> dict:
        with open(self.database_path, encoding="utf-8") as handle:
            return json.load(handle)


def load(directory: str) -> Game:
    """Read a game's configuration and apply its memory layout."""
    if os.path.isfile(directory):
        directory = os.path.dirname(directory)
    with open(os.path.join(directory, "game.json"), encoding="utf-8") as handle:
        values = dict(DEFAULTS, **json.load(handle))
    game = Game(directory=directory, name=values.get("name", values["executable"]),
                executable=values["executable"], iso=values["iso"],
                extracted=values["extracted"], build=values["build"],
                database=values["database"],
                generated=values["generated"], status=values["status"],
                layout=values["layout"],
                vu_regions=[tuple(_number(v) for v in pair) for pair in values["vu_regions"]])
    configure_layout(game)
    return game


def configure_layout(game: Game) -> layout_module.Layout:
    """Set the engine's memory layout from the game: detected, then overridden."""
    settings = {}
    if os.path.exists(game.elf_path):
        elf = game.load_elf()
        section = elf.exec_sections[0]
        settings["image_start"] = section.addr
        settings["image_end"] = max(
            [section.addr + len(elf.section_data(section))]
            + [p.vaddr + p.memsz for p in elf.segments if p.memsz])
        gp = detect_gp(elf)
        if gp is not None:
            settings["gp"] = gp
    for name, value in game.layout.items():
        settings[name] = _number(value)
    return layout_module.configure(**settings)


def _number(value):
    """A number from game.json, where addresses are written as hex strings."""
    return int(value, 0) if isinstance(value, str) else value


def detect_gp(elf: ElfFile, limit: int = 512):
    """The value the entry code puts in $gp, by scanning it with constant registers.

    The startup code builds the small-data pointer and moves it into $gp before
    calling anything that uses it. Scanning forward from the entry point with every
    register built from `lui`, `addiu`, `ori` and moves tracked finds it. Returns
    None if the entry code does not set it that way.
    """
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    base = section.addr
    known = {0: 0}
    address = elf.e_entry

    def constant(reg):
        return known.get(reg)

    for _step in range(limit):
        if not base <= address < base + len(image) - 3:
            return None
        ins = DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)
        m = ins.mnemonic
        written, value = None, None
        if m == "lui":
            written, value = ins.rt, _sign32(ins.imm << 16)
        elif m in ("addiu", "daddiu") and constant(ins.rs) is not None:
            written, value = ins.rt, _sign32(constant(ins.rs) + ins.imm)
        elif m == "ori" and constant(ins.rs) is not None:
            written, value = ins.rt, constant(ins.rs) | (ins.imm & 0xFFFF)
        elif m in ("addu", "daddu", "or") and ins.rd:
            a, b = constant(ins.rs), constant(ins.rt)
            if a is not None and b is not None:
                written, value = ins.rd, _sign32(a + b) if m != "or" else a | b
            else:
                known.pop(ins.rd, None)
        elif ins.rt and not ins.is_branch and not ins.is_jump and m not in (
                "sw", "sd", "sq", "sb", "sh", "swc1", "nop"):
            known.pop(ins.rt, None)
        if written is not None:
            known[written] = value
        if ins.is_call or m == "syscall" or ins.is_return:
            # $gp is established before the startup code calls anything, and is
            # often built in two steps (lui, then addiu), so its value is read
            # here rather than at its first write.
            return known[28] & 0xFFFFFFFF if 28 in known else None
        # Straight on, through branches and jumps alike: the startup code clears
        # the BSS with small loops that jump backwards, and following those would
        # never reach the $gp setup after them. They do not touch the registers
        # that build it.
        address += 4
    return None


def _sign32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


GAMES_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "games")


def add_argument(parser) -> None:
    """The `--game` option every tool takes."""
    parser.add_argument(
        "--game", default=None,
        help="the game's directory (with game.json); defaults to the only one in games/")


def resolve(directory=None) -> str:
    """The game directory to use: the one given, $PS2_GAME, or the only one there is."""
    directory = directory or os.environ.get("PS2_GAME")
    if directory:
        return directory
    found = sorted(entry for entry in os.listdir(GAMES_DIRECTORY)
                   if os.path.exists(os.path.join(GAMES_DIRECTORY, entry, "game.json")))
    if len(found) != 1:
        raise SystemExit("several games in %s; choose one with --game: %s"
                         % (GAMES_DIRECTORY, ", ".join(found)))
    return os.path.join(GAMES_DIRECTORY, found[0])


def from_args(args) -> Game:
    """Load the game a tool was pointed at, and configure the engine for it."""
    return load(resolve(getattr(args, "game", None)))
