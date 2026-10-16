"""A PCSX2 GS dump (.gs, .gs.zst; Shift+F8 in PCSX2) as a stream the host's
GS model replays (--gs-replay), and PCSX2's own picture of it.

    python engine/tools/pcsx2dump.py dump.gs.zst out_dir
    <port exe> --gs-replay out_dir/replay.bin out_dir

out_dir gets replay.bin, pcsx2.png (the dump's screenshot: what PCSX2 drew)
and summary.txt. The GS model, given the console's own commands, either
draws what PCSX2 does or not: a fault in the GS model, or one before it (the
EE, VU1, DMA) told apart.

Take the dump with PCSX2's software renderer (F9): the hardware renderers keep
frame buffers on the GPU, and the dump's GS memory then has stale ones -
a game that draws its last frame back (a motion blur, Mister Mosquito's
cutscenes) replays black.

The dump (PCSX2's GSDump.cpp, GSState::Freeze, state version 9): 0xFFFFFFFF,
the header's size, the header (state version, state size, serial and
screenshot offsets), serial, screenshot (RGBA); the GS state - registers,
local memory, the GIF paths' tags; the privileged registers (8 KB); then
packets: 0 a transfer (path, size, data), 1 a vertical blank, 2 a FIFO read,
3 the privileged registers again.
"""

import os
import struct
import sys

MEMORY = 4 << 20
PRIVILEGED = 0x2000

# GSState::Freeze's registers, in order, by GS register address.
ENV = [0x00, 0x1A, 0x1C, 0x22, 0x3B, 0x3D, 0x44, 0x45, 0x46, 0x49, 0x50, None, 0x51, 0x52, None]
CONTEXT = [0x18, 0x06, 0x14, 0x08, 0x34, 0x36, 0x40, 0x42, 0x47, 0x4A, 0x4C, 0x4E]   # + context
VERTEX = [0x01, 0x02, 0x03, 0x0A]   # RGBAQ, ST, UV, FOG (XYZ would draw)
# The privileged registers a picture needs, by offset (GSPrivRegSet).
PRIVATE = range(0x00, 0xF0, 0x10)


def read_dump(path):
    raw = open(path, 'rb').read()
    if path.endswith('.zst'):
        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(raw).read()
    return raw


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    dump, out = argv[1], argv[2]
    d = read_dump(dump)
    os.makedirs(out, exist_ok=True)
    magic, header_size = struct.unpack_from('<II', d, 0)
    if magic != 0xFFFFFFFF:
        raise SystemExit('an old-format dump (no header): not handled')
    (version, state_size, serial_offset, serial_size, crc, shot_w, shot_h, shot_offset,
     shot_size) = struct.unpack_from('<9I', d, 8)
    if version != 9:
        raise SystemExit('GS state version %d: only 9 is handled' % version)
    if shot_size:
        from PIL import Image
        base = 8 + shot_offset
        Image.frombytes('RGBA', (shot_w, shot_h), d[base:base + shot_size]).convert('RGB').save(
            os.path.join(out, 'pcsx2.png'))
    state = 8 + header_size
    # Before memory: the version, 15 + 24 + 6 registers, the transfer's state.
    regs_at = state + 4
    memory_at = state + state_size - MEMORY - (4 * (16 + 4) + 4)
    records = [b'GSRP1', d[memory_at:memory_at + MEMORY]]

    def reg(address, value):
        records.append(b'R' + struct.pack('<IQ', address, value))

    env = struct.unpack_from('<15Q', d, regs_at)
    contexts = [struct.unpack_from('<12Q', d, regs_at + 120 + 96 * c) for c in range(2)]
    vertex = struct.unpack_from('<4Q', d, regs_at + 120 + 192)
    for c, values in enumerate(contexts):
        for address, value in zip(CONTEXT, values):
            reg(address + c, value)
    for address, value in zip(ENV, env):
        if address is not None and address != 0x00:
            reg(address, value)
    reg(0x00, env[0])                                   # PRIM last: it starts no primitive
    for address, value in zip(VERTEX, vertex):
        reg(address, value)

    def privileged(block):
        for off in PRIVATE:
            records.append(b'P' + struct.pack('<IQ', 0x12000000 + off, struct.unpack_from('<Q', block, off)[0]))

    at = state + state_size
    privileged(d[at:at + PRIVILEGED])
    at += PRIVILEGED
    counts = {'transfers': 0, 'bytes': 0, 'blanks': 0}
    while at < len(d):
        kind = d[at]
        at += 1
        if kind == 0:
            path, size = struct.unpack_from('<BI', d, at)
            at += 5
            data = d[at:at + size]
            at += size
            if path == 0:                               # Path1Old: the packet at the end of a 16 KB buffer
                data = data[16384 - size:] if len(data) == 16384 else data
            gif_path = {0: 1, 1: 2, 2: 3, 3: 1}.get(path)
            if gif_path is None:
                continue
            data = data[:len(data) & ~3]
            records.append(b'W' + struct.pack('<BI', gif_path, len(data)) + data)
            counts['transfers'] += 1
            counts['bytes'] += len(data)
        elif kind == 1:
            at += 1
            records.append(b'V')
            counts['blanks'] += 1
        elif kind == 2:
            at += 4
        elif kind == 3:
            privileged(d[at:at + PRIVILEGED])
            at += PRIVILEGED
        else:
            raise SystemExit('unknown packet %d at %d' % (kind, at - 1))
    with open(os.path.join(out, 'replay.bin'), 'wb') as f:
        for r in records:
            f.write(r)
    summary = 'serial %s, crc %08X; %d transfers (%d KB), %d vertical blanks' % (
        d[8 + serial_offset:8 + serial_offset + serial_size].decode(errors='replace'), crc, counts['transfers'],
        counts['bytes'] >> 10, counts['blanks'])
    open(os.path.join(out, 'summary.txt'), 'w').write(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main(sys.argv)
