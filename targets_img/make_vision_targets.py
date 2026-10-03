"""Vision-test targets, one scan position per pixel.

  tumbling_e_chart_12.png   four tumbling E optotypes (right, down, left, up)
                            on a 12 x 12 grid. The tumbling E is the standard
                            chart for people who cannot read letters; each E
                            is the standard 5 x 5 optotype.
  tumbling_e_big_12.png     one E at double size (strokes two positions wide)
  landolt_c_chart_12.png    four Landolt rings with the gap in four directions

The read-out is scoreable without looking at a correlation: for each
optotype, which way does it point?

    python targets_img/make_vision_targets.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

E_RIGHT = np.array([[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]], float)

C_RIGHT = np.array([[0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 0]], float)


def rotations(glyph):
    """right, down, left, up (the open side)"""
    return [glyph, np.rot90(glyph, -1), np.rot90(glyph, 2), np.rot90(glyph, 1)]


def chart(glyph):
    g = np.zeros((12, 12))
    r = rotations(glyph)
    for (y, x), k in zip(((0, 0), (0, 7), (7, 0), (7, 7)), (0, 1, 3, 2)):
        g[y:y + 5, x:x + 5] = r[k]
    return g


def big(glyph):
    g = np.zeros((12, 12))
    g[1:11, 1:11] = np.kron(glyph, np.ones((2, 2)))
    return g


def tile(a, label, size=300):
    h, w = a.shape
    s = size // max(h, w)
    im = Image.fromarray((a * 255).astype(np.uint8), mode="L").resize((w * s, h * s), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    for k in range(w + 1):
        d.line([(k * s, 0), (k * s, h * s)], fill=(60, 60, 60))
    for k in range(h + 1):
        d.line([(0, k * s), (w * s, k * s)], fill=(60, 60, 60))
    out = Image.new("RGB", (size, size + 22), (30, 30, 30))
    out.paste(im, ((size - w * s) // 2, (size - h * s) // 2))
    ImageDraw.Draw(out).text((4, size + 4), label, fill=(230, 230, 230))
    return out


def main():
    made = [("tumbling_e_chart_12", chart(E_RIGHT), "tumbling E chart: right, down, up, left"),
            ("tumbling_e_big_12", big(E_RIGHT), "one large E (strokes 2 positions wide)"),
            ("landolt_c_chart_12", chart(C_RIGHT), "Landolt rings: gap right, down, up, left")]
    tiles = []
    for name, a, label in made:
        Image.fromarray((a * 255).astype(np.uint8), mode="L").save(os.path.join(HERE, name + ".png"))
        print(f"{name}: {a.shape[1]}x{a.shape[0]}, {int(a.sum())} lit of {a.size}")
        tiles.append(tile(a, label))
    sheet = Image.new("RGB", (len(tiles) * 310 + 10, 342), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + i * 310, 10))
    sheet.save(os.path.join(HERE, "vision_preview.png"))
    print("preview ->", os.path.join(HERE, "vision_preview.png"))


if __name__ == "__main__":
    main()
