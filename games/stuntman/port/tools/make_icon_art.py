#!/usr/bin/env python3
"""Draw the port's own icon (port/icon.ico): a film clapperboard with a star,
a stunt's rating. Original art, drawn here - nothing taken from the game.

    python games/stuntman/port/tools/make_icon_art.py

Needs Pillow. Writes games/stuntman/port/icon.ico (16 to 256 pixels).
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "icon.ico")


def draw(size: int = 1024) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 256.0

    def box(x0, y0, x1, y1):
        return [round(x0 * s), round(y0 * s), round(x1 * s), round(y1 * s)]

    dark, light, accent = (28, 28, 32, 255), (240, 240, 236, 255), (250, 186, 40, 255)
    # The board.
    d.rounded_rectangle(box(24, 100, 232, 228), radius=round(14 * s), fill=dark)
    # The clapper, opened a little: a stick of stripes over the board.
    stick = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stick)
    sd.rounded_rectangle(box(24, 56, 232, 92), radius=round(8 * s), fill=light)
    for i in range(5):
        x = 24 + 12 + i * 44
        sd.polygon([(round(x * s), round(56 * s)), (round((x + 22) * s), round(56 * s)),
                    (round((x + 4) * s), round(92 * s)), (round((x - 18) * s), round(92 * s))], fill=dark)
    stick = stick.rotate(8, center=(round(28 * s), round(92 * s)), resample=Image.BICUBIC)
    im.alpha_composite(stick)
    # The board's own stripes and a red take mark.
    d.rectangle(box(24, 104, 232, 124), fill=light)
    for i in range(5):
        x = 24 + 12 + i * 44
        d.polygon([(round(x * s), round(104 * s)), (round((x + 22) * s), round(104 * s)),
                   (round((x + 4) * s), round(124 * s)), (round((x - 18) * s), round(124 * s))], fill=dark)
    # A star on the board: the stunt's rating.
    cx, cy, r_out, r_in = 128, 176, 40, 16
    points = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / 5
        points.append((round((cx + r * math.cos(a)) * s), round((cy + r * math.sin(a)) * s)))
    d.polygon(points, fill=accent)
    return im


def main() -> None:
    art = draw()
    sizes = [16, 24, 32, 48, 64, 128, 256]
    art.resize((256, 256), Image.LANCZOS).save(OUT, sizes=[(n, n) for n in sizes])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
