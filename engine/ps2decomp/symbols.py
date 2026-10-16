"""Symbol and cross-reference recovery for a stripped image.

A stripped binary still leaks a surprising amount of naming information, and this
module extracts the three kinds that are actually reliable:

* **String cross-references.**  MIPS materialises a 32-bit address with a
  `lui` / `addiu` pair (or `lui` plus a load offset).  Reassembling those pairs
  recovers the constant, and when the constant lands in a string the call site
  gains a very strong human-readable hint.
* **RenderWare provenance.**  The binary keeps RCS `$Id:` tags naming the exact
  middleware source files linked in.  Those tags sit in the read-only data next
  to the code they belong to, so they identify which regions are RenderWare
  rather than game code.
* **Indirect call tables.**  Runs of aligned words pointing at known function
  starts are vtables or jump tables, which both name-adjacent functions and
  reveal C++ class layout.

Everything here is a *hint generator*.  Nothing renames a function without a
recorded reason, so a wrong guess stays auditable rather than silently
poisoning the database.
"""

from __future__ import annotations

import collections
import dataclasses
import re
import struct

from .r5900 import DISASM
from .signatures import written_register

# A printable run this long is almost certainly a real string rather than noise.
MIN_STRING_LEN = 4

_RW_ID = re.compile(rb"\$Id:\s*//([^\s$]+)")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclasses.dataclass(frozen=True)
class StringEntry:
    addr: int
    text: str

    @property
    def is_identifier_like(self) -> bool:
        return bool(_IDENT.match(self.text))


@dataclasses.dataclass
class Xref:
    """A recovered constant address and the instruction pair that built it."""

    from_addr: int
    target: int
    kind: str  # "lui+addiu" | "lui+load" | "lui+store"


_ADDRESSED = {"lw", "lh", "lb", "lbu", "lhu", "lwu", "ld", "lq", "sw", "sh", "sb", "sd",
              "sq", "lwc1", "swc1"}
_CALLEE_SAVED = set(range(16, 24)) | {30}     # $s0-$s7, $fp


class SymbolRecovery:
    def __init__(self, data: bytes, base_addr: int) -> None:
        self.data = data
        self.base = base_addr
        self.end = base_addr + len(data)
        self.strings: dict = {}
        self.xrefs: list = []
        self.xrefs_by_target: dict = collections.defaultdict(list)
        self.renderware_files: list = []

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.end

    # -- strings --------------------------------------------------------

    def find_strings(self, min_len: int = MIN_STRING_LEN) -> int:
        """Index every NUL-terminated printable run in the image."""
        data = self.data
        start = None
        for i, byte in enumerate(data):
            printable = 0x20 <= byte <= 0x7E or byte in (9, 10, 13)
            if printable:
                if start is None:
                    start = i
                continue
            if start is not None:
                # Only accept runs that actually terminate with NUL, which is
                # what distinguishes a C string from incidental printable bytes.
                if byte == 0 and i - start >= min_len:
                    text = data[start:i].decode("ascii", "replace")
                    self.strings[self.base + start] = StringEntry(self.base + start, text)
                start = None
        return len(self.strings)

    def find_renderware_tags(self) -> list:
        """Recover the RenderWare SDK source paths embedded as RCS tags."""
        out = []
        for match in _RW_ID.finditer(self.data):
            path = match.group(1).decode("ascii", "replace")
            out.append((self.base + match.start(), path))
        self.renderware_files = out
        return out

    # -- address materialisation ----------------------------------------

    def find_xrefs(self, code_ranges) -> int:
        """Reassemble `lui`+offset pairs into absolute addresses.

        `code_ranges` is an iterable of (start, end) spans to scan. Tracking is per
        register and kept conservative, so a pair is never matched across a write
        that replaced the upper half:

        * any other write to a register drops its pending upper half;
        * a branch or jump's delay slot is processed before the transfer, because
          it executes first - `lui $a0` before a `jal` and `addiu $a0` in its delay
          slot is how a string argument is usually passed;
        * a call keeps only the callee-saved registers, which the callee preserves;
        * any other control transfer drops everything.
        """
        count = 0

        def record(addr, upper, ins, kind):
            nonlocal count
            target = (upper + ins.imm) & 0xFFFFFFFF
            if self.contains(target):
                xref = Xref(addr, target, kind)
                self.xrefs.append(xref)
                self.xrefs_by_target[target].append(xref)
                count += 1

        def step(addr, pending):
            word = struct.unpack_from("<I", self.data, addr - self.base)[0]
            ins = DISASM.decode(word, addr)
            if ins.mnemonic == "lui":
                pending[ins.rt] = (ins.imm & 0xFFFF) << 16
                return ins
            if ins.mnemonic in ("addiu", "addi", "daddiu") and ins.rs in pending:
                record(addr, pending[ins.rs], ins, "lui+addiu")
            elif ins.mnemonic in _ADDRESSED and ins.rs in pending:
                kind = "lui+store" if ins.mnemonic.startswith("s") else "lui+load"
                record(addr, pending[ins.rs], ins, kind)
            written = written_register(word)
            if written is not None:
                pending.pop(written, None)
            return ins

        for start, end in code_ranges:
            pending = {}  # register -> upper immediate of the lui that set it
            addr = start
            while addr < end and self.contains(addr):
                ins = step(addr, pending)
                if ins.is_branch or ins.is_jump:
                    if addr + 4 < end and self.contains(addr + 4):
                        step(addr + 4, pending)
                    if ins.is_call:
                        for reg in [r for r in pending if r not in _CALLEE_SAVED]:
                            del pending[reg]
                    else:
                        pending.clear()
                    addr += 8
                    continue
                addr += 4
        return count

    # -- pointer tables --------------------------------------------------

    def find_pointer_tables(self, function_starts, min_entries: int = 3) -> list:
        """Locate runs of words pointing at known function entries.

        These are vtables and switch jump tables.  Requiring several consecutive
        valid entries keeps random data from registering as a table.
        """
        starts = set(function_starts)
        tables = []
        words = struct.unpack("<%dI" % (len(self.data) // 4), self.data[: (len(self.data) // 4) * 4])
        run_start = None
        run_len = 0
        for i, word in enumerate(words):
            if word in starts:
                if run_start is None:
                    run_start = i
                run_len += 1
                continue
            if run_start is not None and run_len >= min_entries:
                tables.append(
                    (self.base + run_start * 4, run_len,
                     [words[run_start + k] for k in range(run_len)])
                )
            run_start = None
            run_len = 0
        if run_start is not None and run_len >= min_entries:
            tables.append(
                (self.base + run_start * 4, run_len,
                 [words[run_start + k] for k in range(run_len)])
            )
        return tables

    # -- naming ----------------------------------------------------------

    def string_for(self, addr: int):
        return self.strings.get(addr)

    def annotate(self, addr: int):
        """Return a short human-readable note for a referenced address."""
        entry = self.strings.get(addr)
        if entry is not None:
            text = entry.text
            if len(text) > 48:
                text = text[:45] + "..."
            return '"%s"' % text.replace("\n", "\\n").replace("\r", "\\r")
        return None
