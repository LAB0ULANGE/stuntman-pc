#!/usr/bin/env python3
"""A capture's GS stream (replay.bin, GSRP1) as a PCSX2 GS dump (.gs), for
PCSX2's own renderers to draw what the port's GS model was given: the other
way from pcsx2dump.py.

    python engine/tools/gs2pcsx2.py capture/replay.bin out.gs [--serial SLES-50288 --crc 790A4ACB]
        [--blanks N]

Open out.gs in PCSX2 (it replays dumps; the software renderer, with
EmuCore/GS SaveFrame and SaveDrawStart/SaveDrawCount, writes the target as
each draw left it).

The dump (PCSX2's GSDumpBase, state version 9): 0xFFFFFFFF, the header's
size, the header (state version, state size, serial's offset and size, CRC,
screenshot size and offset), the serial; the GS state (GSState::Freeze:
version, 15 environment registers, 2 x 12 context ones, RGBAQ ST UV FOG XYZ, an
unused register, the transfer's x and y and, from version 9, its w, h,
BITBLTBUF, TRXPOS, TRXREG, rectangle, total, start, end and direction (61
bytes), local memory, the four GIF paths' tag and register, Q); the privileged registers (8 KB); then packets: 0 a
transfer (path: 0 PATH1 old, 1 PATH2, 2 PATH3, 3 PATH1), 1 a vertical blank
(field), 3 the privileged registers again.

The registers the capture began with go first as an A+D packet on PATH3, in
the capture's order (TEX0 with CLD 1 loads the CLUT again), rather than in
the frozen state: the state is left zero but for local memory. The capture
has no SMODE1/SMODE2: PAL, interlaced field mode (Stuntman's) is assumed
unless --ntsc / --frame-mode.
"""

from __future__ import annotations

import argparse
import struct

MEMORY = 4 << 20
PRIVILEGED = 0x2000


def records(data: bytes):
    if data[:5] != b"GSRP1":
        raise SystemExit("not a GSRP1 stream")
    yield "M", data[5:5 + MEMORY]
    at = 5 + MEMORY
    while at < len(data):
        kind = data[at:at + 1]
        at += 1
        if kind in (b"R", b"P"):
            address, value = struct.unpack_from("<IQ", data, at)
            at += 12
            yield kind.decode(), (address, value)
        elif kind == b"W":
            path, size = struct.unpack_from("<BI", data, at)
            at += 5
            yield "W", (path, data[at:at + size])
            at += size
        elif kind == b"V":
            yield "V", None
        else:
            raise SystemExit("bad record at %d" % (at - 1))


def ad_packet(regs) -> bytes:
    """A GIFtag (PACKED, NREG 1, A+D) and a quadword per register, EOP."""
    tag = len(regs) | 1 << 15 | 1 << 60
    out = struct.pack("<QQ", tag, 0xE)
    for address, value in regs:
        out += struct.pack("<QQ", value, address)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay")
    parser.add_argument("out")
    parser.add_argument("--serial", default="SLES-50288")
    parser.add_argument("--crc", default="790A4ACB")
    parser.add_argument("--ntsc", action="store_true", help="NTSC rather than PAL (SMODE1)")
    parser.add_argument("--frame-mode", action="store_true", help="SMODE2 FFMD 0 (frame mode)")
    parser.add_argument("--blanks", type=int, default=-1, help="stop after this many vertical blanks")
    args = parser.parse_args(argv)

    data = open(args.replay, "rb").read()
    privileged = bytearray(PRIVILEGED)
    # SMODE1 as libgraph sets it (CMOD 2 NTSC, 3 PAL); SMODE2 INT, FFMD.
    smode1 = 0x740834504 if args.ntsc else 0x740836504
    struct.pack_into("<Q", privileged, 0x10, smode1)
    struct.pack_into("<Q", privileged, 0x20, 0x1 if args.frame_mode else 0x3)
    memory = b""
    first_privileged = None             # as they were when the stream began
    initial = []
    packets = []
    started = False
    blanks = 0
    field = 0
    for kind, value in records(data):
        if kind == "M":
            memory = value
        elif kind == "R":
            initial.append(value)
        elif kind == "P":
            address, v = value
            struct.pack_into("<Q", privileged, address & 0x1FFF, v)
            if started:
                packets.append(b"\x03" + bytes(privileged))
        elif kind == "W":
            if not started:
                first_privileged = bytes(privileged)
                started = True
            path, words = value
            index = {1: 3, 2: 1, 3: 2}[path]
            packets.append(b"\x00" + struct.pack("<BI", index, len(words)) + words)
        elif kind == "V":
            packets.append(b"\x01" + bytes([field]))
            field ^= 1
            blanks += 1
            if args.blanks >= 0 and blanks >= args.blanks:
                break
    # The capture's registers first.
    setup = ad_packet(initial)
    packets.insert(0, b"\x00" + struct.pack("<BI", 2, len(setup)) + setup)
    state = struct.pack("<I", 9)
    state += bytes(15 * 8 + 24 * 8 + 5 * 8 + 8)   # environment, contexts, vertex, unused
    state += struct.pack("<II", 0, 0)               # the transfer's x, y
    state += bytes(4 + 4 + 8 + 8 + 8 + 16 + 4 + 4 + 4 + 1)    # the rest of it (version 9), none under way
    state += memory
    state += bytes(4 * (16 + 4))                    # the paths' tags and registers
    state += struct.pack("<f", 1.0)                 # Q
    serial = args.serial.encode()
    header = struct.pack("<9I", 9, len(state), 36, len(serial), int(args.crc, 16), 0, 0, 36 + len(serial), 0)
    with open(args.out, "wb") as out:
        out.write(struct.pack("<II", 0xFFFFFFFF, len(header) + len(serial)))
        out.write(header + serial)
        out.write(state)
        out.write(first_privileged or bytes(privileged))
        for p in packets:
            out.write(p)
    print("%s: %d packets, %d vertical blanks, %d registers to begin with" % (args.out, len(packets), blanks,
                                                                                 len(initial)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
