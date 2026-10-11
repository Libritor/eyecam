"""A police officer in the eight pure colours, one scan position per pixel,
for `--mode planes --color-target targets_img/officer_8x10.png
--color-grid-w 8 --color-grid-h 10`: a blue cap with a yellow badge, a
yellow face with black eyes and mouth, a blue uniform with a badge, on
white. Four colours (white, blue, yellow, black), every cell one of the
eight pure colours, so every cell counts in the colour score. 80 positions:
about 33 min with `--plane-spc 5,5,8 --plane-passes 1`.

Also writes officer_8x10_preview.png (the picture at 60 px per cell with
the grid) and a 12 x 14 version for a longer sitting.

    python targets_img/make_officer.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

COLOURS = {"W": (255, 255, 255), "B": (0, 0, 255), "Y": (255, 255, 0),
           "K": (0, 0, 0), "R": (255, 0, 0), "C": (0, 255, 255)}

# 8 wide, 10 tall. Rows: cap (blue, yellow badge), brim, face (yellow) with
# eyes and mouth (black), chin, uniform (blue) with a yellow badge.
OFFICER_8x10 = [
    "WBBBBBBW",
    "WBBYBBBW",
    "BBBBBBBB",
    "WYYYYYYW",
    "WYKYYKYW",
    "WYYYYYYW",
    "WYYKKYYW",
    "WWYYYYWW",
    "BBBYBBBB",
    "BBBBBBBB",
]

# 12 wide, 14 tall: room for a nose, a collar and a tie. 168 positions.
OFFICER_12x14 = [
    "WWBBBBBBBBWW",
    "WBBBBBBBBBBW",
    "WBBBBYYBBBBW",
    "BBBBBBBBBBBB",
    "WWYYYYYYYYWW",
    "WWYKKYYKKYWW",
    "WWYYYYYYYYWW",
    "WWYYYKKYYYWW",
    "WWYYYYYYYYWW",
    "WWYYKKKKYYWW",
    "WWWYYYYYYWWW",
    "BBBBBWWBBBBB",
    "BBBYBBKBBBBB",
    "BBBBBBKBBBBB",
]


def write(rows, name, cell=60):
    h, w = len(rows), len(rows[0])
    assert all(len(r) == w for r in rows), name
    img = Image.new("RGB", (w, h))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            img.putpixel((x, y), COLOURS[ch])
    img.save(os.path.join(HERE, name + ".png"))
    big = img.resize((w * cell, h * cell), Image.NEAREST)
    d = ImageDraw.Draw(big)
    for x in range(1, w):
        d.line([(x * cell, 0), (x * cell, h * cell)], fill=(128, 128, 128), width=1)
    for y in range(1, h):
        d.line([(0, y * cell), (w * cell, y * cell)], fill=(128, 128, 128), width=1)
    big.save(os.path.join(HERE, name + "_preview.png"))
    counts = {c: sum(r.count(c) for r in rows) for c in COLOURS if any(c in r for r in rows)}
    print(f"{name}: {w} x {h} = {w * h} positions, cells per colour {counts}")
    return np.array([[COLOURS[ch] for ch in row] for row in rows], float) / 255


if __name__ == "__main__":
    for rows, name in ((OFFICER_8x10, "officer_8x10"), (OFFICER_12x14, "officer_12x14")):
        g = write(rows, name)
        # what the planes see: lit / dark cells per plane (Otsu needs both)
        for k, pn in enumerate("RGB"):
            lit = int((g[:, :, k] > 0.5).sum())
            print(f"   {pn} plane: {lit} lit, {g.shape[0] * g.shape[1] - lit} dark")
