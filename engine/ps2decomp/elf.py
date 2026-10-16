"""Minimal 32-bit little-endian ELF reader, aimed at PlayStation 2 executables.

The PS2 "Emotion Engine" runs 32-bit little-endian MIPS, so this only implements
ELFCLASS32/ELFDATA2LSB.  Retail PS2 executables are almost always fully linked and
stripped, so the symbol-table support here is best-effort: it works when a
development build leaves `.symtab` in place, and simply reports nothing when the
binary is stripped.
"""

from __future__ import annotations

import dataclasses
import struct

# e_type
ET_NONE, ET_REL, ET_EXEC, ET_DYN, ET_CORE = range(5)

# e_machine
EM_MIPS = 8

# p_type
PT_NULL, PT_LOAD, PT_DYNAMIC, PT_INTERP, PT_NOTE, PT_SHLIB, PT_PHDR = range(7)

# sh_type
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_MIPS_REGINFO = 0x70000006

# sh_flags
SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

# st_info >> 4
STB_LOCAL, STB_GLOBAL, STB_WEAK = 0, 1, 2

# st_info & 0xf
STT_NOTYPE, STT_OBJECT, STT_FUNC, STT_SECTION, STT_FILE = range(5)

SHT_NAMES = {
    SHT_NULL: "NULL",
    SHT_PROGBITS: "PROGBITS",
    SHT_SYMTAB: "SYMTAB",
    SHT_STRTAB: "STRTAB",
    SHT_RELA: "RELA",
    SHT_HASH: "HASH",
    SHT_DYNAMIC: "DYNAMIC",
    SHT_NOTE: "NOTE",
    SHT_NOBITS: "NOBITS",
    SHT_REL: "REL",
    SHT_MIPS_REGINFO: "MIPS_REGINFO",
}

STT_NAMES = {
    STT_NOTYPE: "NOTYPE",
    STT_OBJECT: "OBJECT",
    STT_FUNC: "FUNC",
    STT_SECTION: "SECTION",
    STT_FILE: "FILE",
}


class ElfError(Exception):
    """Raised when a file is not a 32-bit little-endian ELF we can read."""


@dataclasses.dataclass(frozen=True)
class Section:
    index: int
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int

    @property
    def is_alloc(self) -> bool:
        return bool(self.flags & SHF_ALLOC)

    @property
    def is_exec(self) -> bool:
        return bool(self.flags & SHF_EXECINSTR)

    @property
    def is_write(self) -> bool:
        return bool(self.flags & SHF_WRITE)

    @property
    def occupies_file(self) -> bool:
        return self.type != SHT_NOBITS

    @property
    def end_addr(self) -> int:
        return self.addr + self.size

    @property
    def type_name(self) -> str:
        return SHT_NAMES.get(self.type, hex(self.type))

    @property
    def flag_str(self) -> str:
        return "".join(
            [
                "W" if self.is_write else "",
                "A" if self.is_alloc else "",
                "X" if self.is_exec else "",
            ]
        )


@dataclasses.dataclass(frozen=True)
class Segment:
    index: int
    type: int
    offset: int
    vaddr: int
    paddr: int
    filesz: int
    memsz: int
    flags: int
    align: int


@dataclasses.dataclass(frozen=True)
class Symbol:
    name: str
    value: int
    size: int
    info: int
    other: int
    shndx: int

    @property
    def bind(self) -> int:
        return self.info >> 4

    @property
    def type(self) -> int:
        return self.info & 0xF

    @property
    def type_name(self) -> str:
        return STT_NAMES.get(self.type, hex(self.type))

    @property
    def is_function(self) -> bool:
        return self.type == STT_FUNC


class ElfFile:
    """A parsed ELF image held entirely in memory (PS2 executables are small)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self._parse_header()
        self.sections = self._parse_sections()
        self.segments = self._parse_segments()
        self._section_by_name = {s.name: s for s in self.sections if s.name}
        self.symbols = self._parse_symbols()

    @classmethod
    def load(cls, path) -> "ElfFile":
        with open(path, "rb") as handle:
            return cls(handle.read())

    # -- header ----------------------------------------------------------

    def _parse_header(self) -> None:
        d = self.data
        if len(d) < 52 or d[:4] != b"\x7fELF":
            raise ElfError("not an ELF file")
        if d[4] != 1:
            raise ElfError("only ELFCLASS32 is supported")
        if d[5] != 1:
            raise ElfError("only little-endian ELF is supported")
        (
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx,
        ) = struct.unpack_from("<HHIIIIIHHHHHH", d, 16)

    # -- sections --------------------------------------------------------

    def _cstr(self, table_offset: int, index: int) -> str:
        start = table_offset + index
        end = self.data.find(b"\x00", start)
        if end < 0:
            return ""
        return self.data[start:end].decode("utf-8", "replace")

    def _parse_sections(self) -> list:
        if not self.e_shoff or not self.e_shnum:
            return []
        raw = []
        for i in range(self.e_shnum):
            off = self.e_shoff + i * self.e_shentsize
            raw.append(struct.unpack_from("<IIIIIIIIII", self.data, off))
        # The section-header string table is itself a section, so resolve names
        # only after every header is read.
        shstr_off = raw[self.e_shstrndx][4] if self.e_shstrndx < len(raw) else 0
        sections = []
        for i, fields in enumerate(raw):
            name_idx, s_type, flags, addr, offset, size, link, info, align, entsize = fields
            sections.append(
                Section(
                    index=i,
                    name=self._cstr(shstr_off, name_idx) if shstr_off else "",
                    type=s_type,
                    flags=flags,
                    addr=addr,
                    offset=offset,
                    size=size,
                    link=link,
                    info=info,
                    addralign=align,
                    entsize=entsize,
                )
            )
        return sections

    def _parse_segments(self) -> list:
        segments = []
        for i in range(self.e_phnum):
            off = self.e_phoff + i * self.e_phentsize
            p_type, p_off, vaddr, paddr, filesz, memsz, flags, align = struct.unpack_from(
                "<IIIIIIII", self.data, off
            )
            segments.append(
                Segment(i, p_type, p_off, vaddr, paddr, filesz, memsz, flags, align)
            )
        return segments

    def _parse_symbols(self) -> list:
        symtab = next((s for s in self.sections if s.type == SHT_SYMTAB), None)
        if symtab is None or symtab.entsize == 0:
            return []
        strtab = self.sections[symtab.link]
        symbols = []
        count = symtab.size // symtab.entsize
        for i in range(count):
            off = symtab.offset + i * symtab.entsize
            name_idx, value, size, info, other, shndx = struct.unpack_from(
                "<IIIBBH", self.data, off
            )
            symbols.append(
                Symbol(
                    name=self._cstr(strtab.offset, name_idx),
                    value=value,
                    size=size,
                    info=info,
                    other=other,
                    shndx=shndx,
                )
            )
        return symbols

    # -- lookup ----------------------------------------------------------

    def section(self, name: str):
        return self._section_by_name.get(name)

    def section_at(self, addr: int):
        """Return the allocated section containing a virtual address."""
        for s in self.sections:
            if s.is_alloc and s.size and s.addr <= addr < s.end_addr:
                return s
        return None

    def section_data(self, section: Section) -> bytes:
        if not section.occupies_file:
            return b"\x00" * section.size
        return self.data[section.offset : section.offset + section.size]

    def read_vaddr(self, addr: int, length: int) -> bytes:
        """Read `length` bytes from a virtual address, via the section table."""
        section = self.section_at(addr)
        if section is None:
            raise ElfError("address 0x%08x is not mapped" % addr)
        if not section.occupies_file:
            return b"\x00" * length
        start = section.offset + (addr - section.addr)
        return self.data[start : start + length]

    def read_u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read_vaddr(addr, 4))[0]

    # -- convenience -----------------------------------------------------

    @property
    def is_stripped(self) -> bool:
        return not any(s.is_function for s in self.symbols)

    @property
    def exec_sections(self) -> list:
        return [s for s in self.sections if s.is_exec and s.size]

    def describe(self) -> str:
        lines = [
            "entry point      : 0x%08x" % self.e_entry,
            "type             : %d" % self.e_type,
            "machine          : %d%s" % (self.e_machine, " (MIPS)" if self.e_machine == EM_MIPS else ""),
            "flags            : 0x%08x" % self.e_flags,
            "program headers  : %d" % self.e_phnum,
            "section headers  : %d" % self.e_shnum,
            "symbols          : %d" % len(self.symbols),
            "stripped         : %s" % ("yes" if self.is_stripped else "no"),
        ]
        return "\n".join(lines)
