"""ISO9660 reader for PlayStation 2 discs.

PS2 discs are plain ISO9660 (usually without Joliet or Rock Ridge extensions),
using 2048-byte logical sectors laid out as CD-ROM Mode 1 user data.  This module
provides a read-only directory tree plus file extraction with no third-party
dependencies.

Reference: ECMA-119, the freely published ISO9660 equivalent.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import struct
from typing import BinaryIO, Iterator

from .discimage import open_image

SECTOR_SIZE = 2048
SYSTEM_AREA_SECTORS = 16  # volume descriptors start here

VD_BOOT = 0
VD_PRIMARY = 1
VD_SUPPLEMENTARY = 2
VD_PARTITION = 3
VD_TERMINATOR = 255

FLAG_HIDDEN = 1 << 0
FLAG_DIRECTORY = 1 << 1
FLAG_ASSOCIATED = 1 << 2
FLAG_MULTI_EXTENT = 1 << 7


class IsoError(Exception):
    """Raised when the image does not look like a well-formed ISO9660 volume."""


def _both_endian32(buf: bytes, off: int) -> int:
    """Read an ECMA-119 "both byte order" 32-bit field, checking the halves agree."""
    little = struct.unpack_from("<I", buf, off)[0]
    big = struct.unpack_from(">I", buf, off + 4)[0]
    if little != big:
        raise IsoError(f"both-endian mismatch at {off}: {little} != {big}")
    return little


def _both_endian16(buf: bytes, off: int) -> int:
    little = struct.unpack_from("<H", buf, off)[0]
    big = struct.unpack_from(">H", buf, off + 2)[0]
    if little != big:
        raise IsoError(f"both-endian mismatch at {off}: {little} != {big}")
    return little


def _decode_dir_datetime(buf: bytes, off: int):
    """Decode the 7-byte directory-record timestamp (year is an offset from 1900)."""
    year, month, day, hour, minute, second, tz = struct.unpack_from("<BBBBBBb", buf, off)
    if month == 0 or day == 0:
        return None
    try:
        stamp = datetime.datetime(1900 + year, month, day, hour, minute, second)
    except ValueError:
        return None
    # tz counts 15-minute intervals east of GMT.
    return stamp - datetime.timedelta(minutes=15 * tz)


@dataclasses.dataclass(frozen=True)
class DirectoryRecord:
    """One entry in an ISO9660 directory."""

    name: str
    lba: int
    size: int
    flags: int
    recorded: object
    parent_path: str = ""

    @property
    def is_dir(self) -> bool:
        return bool(self.flags & FLAG_DIRECTORY)

    @property
    def path(self) -> str:
        if not self.parent_path:
            return self.name
        return self.parent_path + "/" + self.name

    @property
    def offset(self) -> int:
        """Byte offset of the file's first sector within the image."""
        return self.lba * SECTOR_SIZE


def _parse_identifier(raw: bytes, flags: int) -> str:
    """Turn a raw directory identifier into a usable name.

    The two reserved single-byte names are "." and ".."; regular file names carry a
    ";1" version suffix that nobody wants to see on disk.
    """
    if len(raw) == 1 and raw[0] in (0, 1):
        return "." if raw[0] == 0 else ".."
    name = raw.decode("ascii", errors="replace")
    if not flags & FLAG_DIRECTORY:
        head, sep, _version = name.partition(";")
        if sep:
            name = head
        # ECMA-119 requires a dot even for extension-less names; drop a trailing one.
        if name.endswith("."):
            name = name[:-1]
    return name


class Iso9660:
    """Random-access reader over an ISO9660 image."""

    def __init__(self, stream: BinaryIO, owns_stream: bool = False) -> None:
        self._stream = stream
        self._owns_stream = owns_stream
        self.volume_identifier = ""
        self.system_identifier = ""
        self.publisher_identifier = ""
        self.volume_space_size = 0
        self._root = None
        self._read_descriptors()

    # -- construction ----------------------------------------------------

    @classmethod
    def open(cls, path) -> "Iso9660":
        """An .iso, a raw CD image (.bin) or the .cue sheet naming one."""
        return cls(open_image(path), owns_stream=True)

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> "Iso9660":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- raw access ------------------------------------------------------

    def read_sectors(self, lba: int, count: int = 1) -> bytes:
        self._stream.seek(lba * SECTOR_SIZE)
        want = count * SECTOR_SIZE
        data = self._stream.read(want)
        if len(data) != want:
            raise IsoError("short read of %d sector(s) at LBA %d" % (count, lba))
        return data

    # -- volume descriptors ----------------------------------------------

    def _read_descriptors(self) -> None:
        lba = SYSTEM_AREA_SECTORS
        seen_primary = False
        while True:
            sector = self.read_sectors(lba)
            vd_type = sector[0]
            if sector[1:6] != b"CD001":
                raise IsoError("missing CD001 signature in descriptor at LBA %d" % lba)
            if vd_type == VD_TERMINATOR:
                break
            if vd_type == VD_PRIMARY and not seen_primary:
                self._parse_primary(sector)
                seen_primary = True
            lba += 1
            if lba > SYSTEM_AREA_SECTORS + 64:
                raise IsoError("no volume descriptor set terminator found")
        if not seen_primary:
            raise IsoError("image has no primary volume descriptor")

    def _parse_primary(self, sector: bytes) -> None:
        self.system_identifier = sector[8:40].decode("ascii", "replace").rstrip()
        self.volume_identifier = sector[40:72].decode("ascii", "replace").rstrip()
        self.volume_space_size = _both_endian32(sector, 80)
        logical_block = _both_endian16(sector, 128)
        if logical_block != SECTOR_SIZE:
            raise IsoError("unsupported logical block size %d" % logical_block)
        self.publisher_identifier = sector[318:446].decode("ascii", "replace").rstrip()
        self._root = self._parse_record(sector[156:190], 0, "")[0]

    # -- directory walking -----------------------------------------------

    def _parse_record(self, buf: bytes, off: int, parent_path: str):
        """Parse one directory record; returns (record, bytes consumed)."""
        length = buf[off]
        if length == 0:
            return None, 0
        lba = _both_endian32(buf, off + 2)
        size = _both_endian32(buf, off + 10)
        recorded = _decode_dir_datetime(buf, off + 18)
        flags = buf[off + 25]
        name_len = buf[off + 32]
        raw_name = buf[off + 33 : off + 33 + name_len]
        record = DirectoryRecord(
            name=_parse_identifier(raw_name, flags),
            lba=lba,
            size=size,
            flags=flags,
            recorded=recorded,
            parent_path=parent_path,
        )
        return record, length

    @property
    def root(self) -> DirectoryRecord:
        if self._root is None:
            raise IsoError("volume not parsed")
        return self._root

    def list_dir(self, directory: DirectoryRecord):
        """Return the real children of a directory, skipping the dot entries."""
        if not directory.is_dir:
            raise IsoError("%r is not a directory" % directory.path)
        sector_count = (directory.size + SECTOR_SIZE - 1) // SECTOR_SIZE
        data = self.read_sectors(directory.lba, sector_count)
        parent_path = directory.path if directory.path not in (".", "") else ""
        entries = []
        off = 0
        while off < directory.size:
            # A record never straddles a sector boundary; a zero length byte means
            # "skip ahead to the next sector".
            if data[off] == 0:
                off = (off // SECTOR_SIZE + 1) * SECTOR_SIZE
                continue
            record, consumed = self._parse_record(data, off, parent_path)
            off += consumed
            if record is None or record.name in (".", ".."):
                continue
            entries.append(record)
        return entries

    def walk(self, directory=None) -> Iterator[DirectoryRecord]:
        """Depth-first iteration over every record below `directory`."""
        start = directory if directory is not None else self.root
        stack = [start]
        while stack:
            current = stack.pop()
            for entry in sorted(self.list_dir(current), key=lambda e: e.name):
                yield entry
                if entry.is_dir:
                    stack.append(entry)

    def find(self, path: str) -> DirectoryRecord:
        """Look up a record by "/"-separated path, case-insensitively."""
        current = self.root
        for part in path.strip("/").split("/"):
            if not part:
                continue
            for entry in self.list_dir(current):
                if entry.name.lower() == part.lower():
                    current = entry
                    break
            else:
                raise FileNotFoundError(path)
        return current

    # -- extraction ------------------------------------------------------

    def read_file(self, record: DirectoryRecord) -> bytes:
        if record.is_dir:
            raise IsoError("%r is a directory" % record.path)
        self._stream.seek(record.offset)
        return self._stream.read(record.size)

    def extract_file(self, record: DirectoryRecord, dest) -> int:
        """Stream one file out to `dest`, returning the number of bytes written."""
        os.makedirs(os.path.dirname(os.fspath(dest)) or ".", exist_ok=True)
        remaining = record.size
        self._stream.seek(record.offset)
        written = 0
        with open(dest, "wb") as out:
            while remaining > 0:
                chunk = self._stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
        return written
