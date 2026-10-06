"""Pixel versions of the EyeTap portrait, one scan position per pixel.

Hand-drawn (X = white, G = half grey, . = black):
  mann_eyetap_12.png        12 x 12, black and white, uses every row
  mann_eyetap_12_grey.png   12 x 12, three levels (hair in grey)
  mann_eyetap_16.png        16 x 16, black and white
From the photograph itself (cropped to the head, averaged per position):
  mann_photo_12_grey4.png   12 x 12, four grey levels
  mann_photo_12_grey3.png   12 x 12, three grey levels
  mann_photo_16_grey4.png   16 x 16, four grey levels

    python targets_img/make_mann.py [photo.png]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))

# 12 x 12, every row used: forehead from the top row to the chin in the
# bottom row. The EyeTap is a dark bar across the brow (row 4, ears showing
# either side) with the camera block hanging below it over the viewer's-left
# eye (rows 5-6); the other eye is uncovered, a single dark pixel with skin
# around it (row 6); then the mouth (row 9).
MANN_12 = [
    "....XXXX....",
    "...XXXXXX...",
    "...XXXXXX...",
    "...XXXXXX...",
    "..X......X..",
    "...X..XXX...",
    "...X..X.X...",
    "...XXXXXX...",
    "...XXXXXX...",
    "...XX..XX...",
    "...XXXXXX...",
    "....XXXX....",
]

# The same face with the hair in half grey: over the top of the head and
# down both sides behind the ears.
MANN_12_GREY = [
    "..GGXXXXGG..",
    "..GXXXXXXG..",
    "..GXXXXXXG..",
    "..GXXXXXXG..",
    "..X......X..",
    "..GX..XXXG..",
    "..GX..X.XG..",
    "..GXXXXXXG..",
    "..GXXXXXXG..",
    "...XX..XX...",
    "...XXXXXX...",
    "....XXXX....",
]

# 16 x 16: room for the nose.
MANN_16 = [
    "......XXXX......",
    ".....XXXXXX.....",
    "....XXXXXXXX....",
    "....XXXXXXXX....",
    "....XXXXXXXX....",
    "....XXXXXXXX....",
    "...X........X...",
    "...X...XXXXXX...",
    ".......XXX.X....",
    "....XXXXXXXX....",
    "....XXX..XXX....",
    "....XXXXXXXX....",
    "....XX....XX....",
    ".....XXXXXX.....",
    ".....XXXXXX.....",
    "......XXXX......",
]

LEVEL = {"X": 255, "G": 128, ".": 0}


def to_array(rows):
    assert len({len(r) for r in rows}) == 1, "ragged drawing"
    return np.array([[LEVEL[c] for c in r] for r in rows], dtype=np.uint8)


def from_photo(path, n, levels):
    """Crop the photo to the head (hair top to chin), average each scan
    position, stretch the contrast and round to `levels` grey levels."""
    im = Image.open(path).convert("L")
    w, h = im.size
    # head: about 2 % to 93 % of the height, centred slightly right of middle
    top, bottom = int(0.02 * h), int(0.93 * h)
    side = bottom - top
    cx = int(0.515 * w)
    box = (cx - side // 2, top, cx - side // 2 + side, bottom)
    small = np.asarray(ImageOps.autocontrast(im.crop(box), cutoff=1)
                       .resize((n, n), Image.BOX), float) / 255.0
    lo, hi = np.percentile(small, [8, 96])
    small = np.clip((small - lo) / (hi - lo), 0, 1)
    q = np.round(small * (levels - 1)) / (levels - 1)
    return (q * 255).astype(np.uint8)


def tile(a, label, size=300):
    im = Image.fromarray(a, mode="L").resize((size, size), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    n = a.shape[0]
    for k in range(n + 1):
        p = round(k * size / n)
        d.line([(p, 0), (p, size)], fill=(60, 60, 60))
        d.line([(0, p), (size, p)], fill=(60, 60, 60))
    out = Image.new("RGB", (size, size + 22), (30, 30, 30))
    out.paste(im, (0, 0))
    ImageDraw.Draw(out).text((4, size + 4), label, fill=(230, 230, 230))
    return out


def main():
    photo = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else None
    made = []
    for name, rows, label in (
            ("mann_eyetap_12", MANN_12, "A  drawn, black/white, 12x12"),
            ("mann_eyetap_12_grey", MANN_12_GREY, "B  drawn, 3 levels (hair grey), 12x12"),
            ("mann_eyetap_16", MANN_16, "C  drawn, black/white, 16x16")):
        a = to_array(rows)
        Image.fromarray(a, mode="L").save(os.path.join(HERE, name + ".png"))
        made.append((a, label))
        print(f"{name}: {a.shape[1]}x{a.shape[0]}, levels {sorted(set(a.ravel().tolist()))}, "
              f"{int((a > 0).sum())} lit of {a.size}")
    if photo:
        for name, n, lv, label in (
                ("mann_photo_12_grey4", 12, 4, "D  from the photo, 4 levels, 12x12"),
                ("mann_photo_12_grey3", 12, 3, "E  from the photo, 3 levels, 12x12"),
                ("mann_photo_16_grey4", 16, 4, "F  from the photo, 4 levels, 16x16")):
            a = from_photo(photo, n, lv)
            Image.fromarray(a, mode="L").save(os.path.join(HERE, name + ".png"))
            made.append((a, label))
            print(f"{name}: {n}x{n}, {lv} levels, {int((a > 0).sum())} lit of {a.size}")
    tiles = [tile(a, label) for a, label in made]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 310 + 10, rows * 332 + 10), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % cols) * 310, 10 + (i // cols) * 332))
    sheet.save(os.path.join(HERE, "mann_preview.png"))
    print("preview ->", os.path.join(HERE, "mann_preview.png"))


if __name__ == "__main__":
    main()
