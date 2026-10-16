"""Disc images in the forms players keep them: a plain .iso, or a raw CD.

DVD games come as .iso files of 2048-byte sectors. CD games are often kept raw,
a .bin of 2352-byte sectors with a .cue sheet naming it. A raw sector starts
with a 12-byte sync pattern and a 4-byte header whose last byte is the mode:

  Mode 1:         sync, header, 2048 bytes of data at offset 16, EDC/ECC
  Mode 2 Form 1:  sync, header, 8-byte subheader, 2048 bytes at offset 24, ...

PS2 CDs are Mode 2 (XA). `open_image` hands back a stream of the 2048-byte data
of every sector, so ISO 9660 code reads either form the same way.

Reference: ECMA-130 (CD-ROM), the CD-ROM XA specification.
"""

from __future__ import annotations

import io
import os
import re
import shlex
from typing import BinaryIO

DATA_SIZE = 2048
RAW_SIZE = 2352
SYNC = b"\x00" + b"\xff" * 10 + b"\x00"


class DiscImageError(Exception):
    """Raised when a file is not a disc image this module understands."""


def cue_data_file(cue_path) -> str:
    """The image file a .cue sheet's first data track lives in."""
    cue_path = os.fspath(cue_path)
    with open(cue_path, "r", encoding="latin-1") as cue:
        for line in cue:
            line = line.strip()
            if line.upper().startswith("FILE "):
                m = re.match(r'FILE\s+"([^"]+)"', line, re.IGNORECASE)
                name = m.group(1) if m else shlex.split(line[5:])[0]
                return os.path.join(os.path.dirname(cue_path), name)
    raise DiscImageError("%s names no FILE" % cue_path)


def raw_data_offset(stream: BinaryIO) -> int | None:
    """Where a raw image's data sits in each sector (16 or 24), or None when the
    stream is a plain 2048-byte image. Sector 16 holds the volume descriptor on
    every PS2 disc, so its header is the one read."""
    stream.seek(16 * RAW_SIZE)
    head = stream.read(16)
    if len(head) != 16 or head[:12] != SYNC:
        stream.seek(0)
        if stream.read(12) != SYNC:
            return None
        stream.seek(0)
        head = stream.read(16)
    mode = head[15]
    if mode == 1:
        return 16
    if mode == 2:
        return 24
    raise DiscImageError("raw sector of mode %d" % mode)


class RawSectorStream(io.RawIOBase):
    """A read-only stream of the 2048 data bytes of each sector of a raw image."""

    def __init__(self, raw: BinaryIO, data_offset: int) -> None:
        super().__init__()
        self._raw = raw
        self._data_offset = data_offset
        self._raw.seek(0, os.SEEK_END)
        self._sectors = self._raw.tell() // RAW_SIZE
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self._sectors * DATA_SIZE + offset
        else:
            raise ValueError("whence %r" % whence)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        end = self._sectors * DATA_SIZE
        if size is None or size < 0:
            size = end - self._pos
        size = max(0, min(size, end - self._pos))
        out = bytearray()
        while size > 0:
            lsn, within = divmod(self._pos, DATA_SIZE)
            take = min(size, DATA_SIZE - within)
            self._raw.seek(lsn * RAW_SIZE + self._data_offset + within)
            chunk = self._raw.read(take)
            if len(chunk) != take:
                break
            out += chunk
            self._pos += take
            size -= take
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._raw.close()
        super().close()


def open_image(path) -> BinaryIO:
    """Open a disc image (.iso, raw .bin, or the .cue naming one) as a stream of
    2048-byte sectors."""
    path = os.fspath(path)
    if path.lower().endswith(".cue"):
        path = cue_data_file(path)
    raw = open(path, "rb")
    try:
        offset = raw_data_offset(raw)
    except Exception:
        raw.close()
        raise
    raw.seek(0)
    if offset is None:
        return raw
    return RawSectorStream(raw, offset)
