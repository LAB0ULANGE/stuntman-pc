"""Function signatures that survive relinking, for recognising code across games.

Games built against the same SDK or middleware contain the same library code: the
EE kernel stubs, Sony's libraries, the C runtime, RenderWare. Linked into a
different executable, that code keeps its instructions but not its addresses. Call
targets, the halves of absolute addresses and $gp-relative offsets all change,
while the rest stays the same.

A signature is a hash of the function's words with exactly those fields masked:

* `j`/`jal` targets (26 bits);
* `lui` immediates, the high half of an absolute address;
* the immediate of an instruction whose base register still holds a `lui` result,
  the low half of the same address;
* the immediate of an instruction based on $gp (small data).

Branch offsets are relative and stay in. Masking depends only on the instruction
stream and never on the values masked, so the same code always masks the same way.
The register tracking is a linear scan with no control flow, which is enough for
determinism.

Short functions collide: every `jr $ra; nop` looks alike. So the instruction count
is part of the signature, and a library entry carrying more than one name is never
used to name anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

# Opcodes whose 16-bit immediate is added to the rs register: arithmetic and
# logical immediates, then every load and store form.
_BASED = ({0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x18, 0x19, 0x1A, 0x1B, 0x1E, 0x1F}
          | set(range(0x20, 0x30)) | {0x31, 0x33, 0x36, 0x37, 0x39, 0x3E, 0x3F})
_STORES = set(range(0x28, 0x30)) | {0x1F, 0x39, 0x3E, 0x3F}
_COP_STORES_AND_LOADS = {0x31, 0x33, 0x36, 0x39, 0x3E}   # write no general register
_GP = 28
_IMM = 0xFFFF
_TARGET = 0x03FFFFFF


def written_register(word: int):
    """The general register an instruction writes, as far as tracking needs it."""
    op = word >> 26
    rt = (word >> 16) & 31
    rd = (word >> 11) & 31
    if op == 3:                                   # jal
        return 31
    if op in (0x00, 0x1C):                        # SPECIAL, MMI
        return rd or None
    if op == 0x11 and ((word >> 21) & 31) in (0, 1, 2):   # mfc1, dmfc1, cfc1
        return rt
    if op == 0x12 and ((word >> 21) & 31) in (1, 2):      # qmfc2, cfc2
        return rt
    if op in _BASED and op not in _STORES and op not in _COP_STORES_AND_LOADS:
        return rt
    if op == 0x0F:
        return rt
    return None


def masked_words(words) -> list:
    """The words with every relocation-dependent field cleared."""
    high = set()                 # registers holding a lui result
    out = []
    for word in words:
        op = word >> 26
        rs = (word >> 21) & 31
        if op in (2, 3):
            word &= ~_TARGET
        elif op == 0x0F:
            word &= ~_IMM
        elif op in _BASED and (rs in high or rs == _GP):
            word &= ~_IMM
        out.append(word)
        written = written_register(word)
        if op == 0x0F:
            high.add(written)
        elif written is not None:
            high.discard(written)
    return out


def function_words(image: bytes, base: int, start: int, end: int) -> list:
    count = (end - start) // 4
    return list(struct.unpack_from("<%dI" % count, image, start - base))


def signature(image: bytes, base: int, start: int, end: int) -> str:
    words = masked_words(function_words(image, base, start, end))
    digest = hashlib.sha1(struct.pack("<%dI" % len(words), *words)).hexdigest()
    return "%d:%s" % (len(words), digest[:16])


def compute(database: dict, image: bytes, base: int) -> dict:
    """{function address: signature} for every function with a body."""
    return {func["addr"]: signature(image, base, func["addr"], func["end"])
            for func in database["functions"] if func["end"] > func["addr"]}


LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "signatures", "library.json")


class Library:
    """Named signatures gathered from every game: signature -> {name: [executables]}.

    Each name records the executables that gave it. Two games naming the same code
    differently leaves the entry ambiguous, and an ambiguous entry names nothing.
    Re-exporting a game first withdraws everything it contributed before, so a
    corrected name replaces the old one instead of joining it.
    """

    def __init__(self, entries=None):
        self.entries = entries or {}

    @classmethod
    def load(cls, path: str = LIBRARY_PATH) -> "Library":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle)["functions"])

    def save(self, path: str = LIBRARY_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ordered = {key: {name: sorted(sources) for name, sources in sorted(entry.items())}
                   for key, entry in sorted(self.entries.items())}
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"format": 2, "functions": ordered}, handle, indent=1)
            handle.write("\n")

    def withdraw(self, source: str) -> None:
        """Remove every name this executable contributed."""
        for key in list(self.entries):
            entry = self.entries[key]
            for name in list(entry):
                if source in entry[name]:
                    entry[name].remove(source)
                if not entry[name]:
                    del entry[name]
            if not entry:
                del self.entries[key]

    def add(self, key: str, name: str, source: str) -> None:
        sources = self.entries.setdefault(key, {}).setdefault(name, [])
        if source not in sources:
            sources.append(source)

    def sources(self, key: str) -> list:
        return sorted({s for sources in self.entries.get(key, {}).values() for s in sources})

    def name_for(self, key: str):
        """The one name this signature carries, or None if it has none or several."""
        entry = self.entries.get(key)
        if entry and len(entry) == 1:
            return next(iter(entry))
        return None
