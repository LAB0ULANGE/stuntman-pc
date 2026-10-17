"""Check the host's IPU IDEC against an independent MPEG-2 decoder.

A verification script, outside the build: it needs PyAV (FFmpeg) and numpy,
which the engine does not. It encodes a test picture as MPEG-2 intra frames
with FFmpeg's encoder, cuts the elementary stream into slices, has the
host's IDEC decode them (ps2host_tests.exe with PS2HOST_IDEC_CASE, which
runs each slice through the IPU's input FIFO), and compares the macroblocks
with FFmpeg's own decoding of the same stream, put through the same colour
conversion (the IPU's CSC, EE User's Manual 8.3.6). The IDCTs differ (the
host's is IEEE 1180's double-precision reference, FFmpeg's an integer one)
by a step at most: every RGB value must be one that Y, Cb and Cr within a
step of FFmpeg's give; anything else is a decoding error, reported where.

    python engine/tools/idec_check.py <ps2host_tests.exe> [--keep DIR]

Variants: frame DCT and field DCT (DTD), B.14 and B.15 (IVF), linear and
non-linear quantiser scale (QST), DC precision 8 to 10 bits (IDP), a
quantiser changing from macroblock to macroblock, and alternate scan (AS).
"""

import argparse
import io
import os
import struct
import subprocess
import sys
import tempfile

try:
    import av
    import numpy as np
except ImportError:
    raise SystemExit("idec_check.py needs PyAV and numpy (pip install av numpy), outside the engine's build")

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
          12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

# ISO/IEC 13818-2 6.3.11: the default intra quantiser matrix, natural order.
DEFAULT_INTRA = [8, 16, 19, 22, 26, 27, 29, 34,
                 16, 16, 22, 24, 27, 29, 34, 37,
                 19, 22, 26, 27, 29, 34, 34, 38,
                 22, 22, 26, 27, 29, 34, 37, 40,
                 22, 26, 27, 29, 32, 35, 40, 48,
                 26, 27, 29, 32, 35, 40, 48, 58,
                 26, 27, 29, 34, 38, 46, 56, 69,
                 27, 29, 35, 38, 46, 56, 69, 83]


class Bits:
    def __init__(self, data, at=0):
        self.data, self.bit = data, at * 8

    def get(self, n):
        value = 0
        for _ in range(n):
            byte = self.data[self.bit >> 3]
            value = (value << 1) | ((byte >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        return value

    def peek(self, n):
        saved = self.bit
        value = self.get(n)
        self.bit = saved
        return value


def start_codes(data):
    at = 0
    while True:
        at = data.find(b"\x00\x00\x01", at)
        if at < 0 or at + 3 >= len(data):
            return
        yield at, data[at + 3]
        at += 3


def picture(width, height, seed):
    """A test picture, yuv420p: gradients, edges and noise - every coefficient busy."""
    rng = np.random.default_rng(seed)
    y = np.add.outer(np.arange(height) * 3, np.arange(width) * 2).astype(np.float64) % 220 + 16
    y[height // 3: height // 2, :] = 235 - y[height // 3: height // 2, :] / 2
    y[:, width // 4: width // 4 + 3] = 16
    y += rng.normal(0, 18, y.shape)
    u = 128 + 80 * np.sin(np.add.outer(np.arange(height // 2) / 5.0, np.arange(width // 2) / 7.0))
    v = 128 + 70 * np.cos(np.add.outer(np.arange(height // 2) / 3.0, -np.arange(width // 2) / 9.0))
    u += rng.normal(0, 12, u.shape)
    v += rng.normal(0, 12, v.shape)
    clip = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.concatenate([clip(y), clip(u).reshape(-1, width), clip(v).reshape(-1, width)])


def encode(yuv, width, height, options, interlaced):
    out_buffer = io.BytesIO()
    container = av.open(out_buffer, "w", format="mpeg2video")
    stream = container.add_stream("mpeg2video", rate=25)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    context = stream.codec_context
    context.gop_size = 1
    context.max_b_frames = 0
    context.bit_rate = 40_000_000
    if interlaced:
        context.flags |= av.codec.context.Flags.interlaced_dct
    context.options = dict(options)
    frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return out_buffer.getvalue()


def decode(es):
    container = av.open(io.BytesIO(es), "r", format="mpeg2video")
    frame = next(container.decode(video=0))
    return frame.to_ndarray(format="yuv420p")


def parse(es):
    """IPU_CTRL, the intra matrix as transmitted, and the slices: (command, bytes)."""
    matrix = [DEFAULT_INTRA[z] for z in ZIGZAG]
    idp = alternate = ivf = qst = 0
    frame_dct = 1
    codes = list(start_codes(es))
    slices = []
    for index, (at, code) in enumerate(codes):
        b = Bits(es, at + 4)
        if code == 0xB3:                                     # sequence header
            b.get(12 + 12 + 4 + 4 + 18 + 1 + 10 + 1)
            if b.get(1):
                matrix = [b.get(8) for _ in range(64)]
        elif code == 0xB5:
            ext = b.get(4)
            if ext == 8:                                     # picture coding extension
                b.get(16)
                idp = b.get(2)
                structure = b.get(2)
                b.get(1)
                frame_dct = b.get(1)
                b.get(1)
                qst, ivf, alternate = b.get(1), b.get(1), b.get(1)
                assert structure == 3, "frame pictures only"
            elif ext == 3 and b.get(1):                      # quant matrix extension
                matrix = [b.get(8) for _ in range(64)]
        elif 0x01 <= code <= 0xAF:                           # a slice
            qsc = b.get(5)
            if b.peek(1):
                b.get(1 + 1 + 7)
                while b.get(1):
                    b.get(8)
            else:
                b.get(1)
            header = b.bit - (at + 4) * 8
            end = codes[index + 1][0] + 4 if index + 1 < len(codes) else len(es)
            body = es[at + 4:end]
            if index + 1 >= len(codes):
                body += b"\x00\x00\x01\xB7"
            command = 0x10000000 | header | (qsc << 16) | ((0 if frame_dct else 1) << 24)
            slices.append((command, body))
    ctrl = (idp << 16) | (alternate << 20) | (ivf << 21) | (qst << 22) | (1 << 24)
    return ctrl, matrix, slices


def csc(y, cb, cr):
    """The IPU's CSC, as the host's: RGB (16x16x3) of 4:2:0 macroblock planes (16x16, 8x8, 8x8)."""
    y = y.astype(np.int64)
    cb = np.repeat(np.repeat(cb.astype(np.int64) - 128, 2, 0), 2, 1)
    cr = np.repeat(np.repeat(cr.astype(np.int64) - 128, 2, 0), 2, 1)
    luma = (0x95 * np.maximum(0, y - 16)) >> 6
    red = ((0xCC * cr) >> 6) + 1
    green = ((-0x68 * cr) >> 6) + ((-0x32 * cb) >> 6) + 1
    blue = ((0x102 * cb) >> 6) + 1
    return np.stack([np.clip((luma + c) >> 1, 0, 255) for c in (red, green, blue)], axis=-1)


def csc_range(y, cb, cr):
    """The RGB CSC gives anywhere within one step of Y, Cb and Cr: each value's least and most."""
    low = high = None
    for dy in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dr in (-1, 0, 1):
                rgb = csc(np.clip(y.astype(np.int64) + dy, 0, 255), np.clip(cb.astype(np.int64) + db, 0, 255),
                          np.clip(cr.astype(np.int64) + dr, 0, 255))
                low = rgb if low is None else np.minimum(low, rgb)
                high = rgb if high is None else np.maximum(high, rgb)
    return low, high


def check(tests_exe, name, options, interlaced, keep):
    width, height = 128, 96
    yuv = picture(width, height, seed=len(name))
    es = encode(yuv, width, height, options, interlaced)
    reference = decode(es)
    ctrl, matrix, slices = parse(es)
    with tempfile.TemporaryDirectory() as work:
        directory = keep or work
        case = os.path.join(directory, name + ".idec")
        with open(case, "wb") as f:
            f.write(struct.pack("<I", ctrl) + bytes(matrix) + struct.pack("<I", len(slices)))
            for command, body in slices:
                f.write(struct.pack("<II", command, len(body)) + body)
        env = dict(os.environ, PS2HOST_IDEC_CASE=case)
        subprocess.run([tests_exe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        with open(case + ".out", "rb") as f:
            got = np.frombuffer(f.read(), dtype=np.uint8)
    per_row = width // 16
    count = (width // 16) * (height // 16)
    if got.size != count * 1024:
        print("%-28s FAIL: %d macroblocks out of IDEC, %d expected" % (name, got.size // 1024, count))
        return False
    got = got.reshape(count, 16, 16, 4)
    y_plane = reference[:height]
    u_plane = reference[height:height + height // 4].reshape(height // 2, width // 2)
    v_plane = reference[height + height // 4:].reshape(height // 2, width // 2)
    worst, worst_at, outside = 0, None, 0
    for mb in range(count):
        mx, my = mb % per_row, mb // per_row
        planes = (y_plane[my * 16:my * 16 + 16, mx * 16:mx * 16 + 16],
                  u_plane[my * 8:my * 8 + 8, mx * 8:mx * 8 + 8],
                  v_plane[my * 8:my * 8 + 8, mx * 8:mx * 8 + 8])
        low, high = csc_range(*planes)
        value = got[mb, :, :, :3].astype(np.int64)
        off = np.maximum(low - value, 0) + np.maximum(value - high, 0)
        outside += int((off > 0).sum())
        if off.max() > worst:
            worst, worst_at = int(off.max()), (mx, my)
    exact = int((got[:, :, :, :3].astype(np.int64) == np.stack([csc(
        y_plane[(m // per_row) * 16:(m // per_row) * 16 + 16, (m % per_row) * 16:(m % per_row) * 16 + 16],
        u_plane[(m // per_row) * 8:(m // per_row) * 8 + 8, (m % per_row) * 8:(m % per_row) * 8 + 8],
        v_plane[(m // per_row) * 8:(m // per_row) * 8 + 8, (m % per_row) * 8:(m % per_row) * 8 + 8])
        for m in range(count)])).mean() * 100)
    ok = outside == 0
    print("%-28s %s: %d slices, %d macroblocks; %d%% of RGB values exactly FFmpeg's, all within a YUV step%s" %
          (name, "ok" if ok else "FAIL", len(slices), count, exact,
           "" if ok else " - NOT: %d values, up to %d past it (macroblock %d,%d)" % (outside, worst, *worst_at)))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tests_exe", help="ps2host_tests.exe from the engine's build")
    parser.add_argument("--keep", help="keep the case files in this directory")
    args = parser.parse_args(argv)
    base = {"intra_vlc": "0", "non_linear_quant": "0", "alternate_scan": "0", "dc": "8", "qmin": "2", "qmax": "12"}
    variants = [
        ("frame DCT, B.14, linear", base, False),
        ("field DCT (DTD)", base, True),
        ("B.15 (IVF)", dict(base, intra_vlc="1"), False),
        ("non-linear scale (QST)", dict(base, non_linear_quant="1"), False),
        ("DC precision 10 bits", dict(base, dc="10"), False),
        # Adaptive quantisation (luminance and texture masking): a quantiser per macroblock.
        ("quantiser per macroblock", dict(base, qmin="1", qmax="31", lumi_mask="0.6", tcplx_mask="0.6",
                                          dark_mask="0.6"), False),
        ("alternate scan (AS)", dict(base, alternate_scan="1"), False),
    ]
    results = [check(args.tests_exe, name, options, interlaced, args.keep) for name, options, interlaced in variants]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
