"""Test images in which the grey level is the content, one scan position per
pixel. Writes PNGs next to this file and a preview sheet.

  wedge_12x4.png      a step wedge: 12 drive levels left to right, 4 rows.
                      Measures the response curve (how the 12 Hz response
                      grows with flicker brightness), like the density
                      wedge used to characterise a film or a camera.
  nocameras_12.png    the paper's "NO CAMERAS" as a sign: NO in white over a
                      grey camera with a white slash.
  sphere_12.png       a ball lit from the upper left, four levels.
  sunset_12.png       white sun, grey sky bands, black ground.

    python targets_img/make_grey_targets.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def wedge(w=12, h=4):
    row = np.linspace(0.0, 1.0, w)
    return np.tile(row, (h, 1))


# X = white, G = 2/3 grey, g = 1/3 grey, . = black
NOCAMERAS = [
    ".X..X..XX...",
    ".XX.X.X..X..",
    ".X.XX.X..X..",
    ".X..X.X..X..",
    ".X..X..XX...",
    "............",
    "...GG.....X.",
    ".GGGGGGGGX..",
    ".GGG..GGXGG.",
    ".GGG..GXGGG.",
    ".GGGGGXGGGG.",
    ".....X......",
]
LV = {"X": 1.0, "G": 0.5, "g": 0.25, ".": 0.0}


def drawing(rows):
    assert len({len(r) for r in rows}) == 1
    return np.array([[LV[c] for c in r] for r in rows])


def sphere(n=12, levels=4):
    y, x = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2.0
    r = n * 0.46
    dx, dy = (x - cx) / r, (y - cy) / r
    inside = dx * dx + dy * dy <= 1.0
    dz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
    light = np.array([-0.55, -0.6, 0.58])
    lam = np.clip(dx * light[0] + dy * light[1] + dz * light[2], 0, 1)
    shade = 0.25 + 0.75 * lam          # the dark side stays visible
    q = np.round(shade * (levels - 1)) / (levels - 1)
    return np.where(inside, q, 0.0)


def sunset(n=12):
    g = np.zeros((n, n))
    g[0:2, :] = 0.25                   # upper sky
    g[2:5, :] = 0.5                    # lower sky
    g[5:7, :] = 0.5
    y, x = np.mgrid[0:n, 0:n]
    sun = (x - 5.5) ** 2 + (y - 4.5) ** 2 <= 2.6 ** 2
    g[sun & (y < 7)] = 1.0             # the sun, cut by the horizon
    g[7:, :] = 0.0                     # ground
    g[7, 4:8] = 0.5                    # reflection on the water
    g[8, 5:7] = 0.25
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
    made = [("wedge_12x4", wedge(), "step wedge, 12 levels x 4 rows (48 positions)"),
            ("nocameras_12", drawing(NOCAMERAS), "NO CAMERAS sign, 3 levels, 12x12"),
            ("sphere_12", sphere(), "lit ball, 4 levels, 12x12"),
            ("sunset_12", sunset(), "sunset, 4 levels, 12x12")]
    tiles = []
    for name, a, label in made:
        Image.fromarray((a * 255).round().astype(np.uint8), mode="L").save(os.path.join(HERE, name + ".png"))
        lv = sorted(set(np.round(a.ravel(), 2).tolist()))
        print(f"{name}: {a.shape[1]}x{a.shape[0]}, {len(lv)} levels, {int((a > 0).sum())} lit of {a.size}")
        tiles.append(tile(a, label))
    sheet = Image.new("RGB", (len(tiles) * 310 + 10, 342), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + i * 310, 10))
    sheet.save(os.path.join(HERE, "grey_preview.png"))
    print("preview ->", os.path.join(HERE, "grey_preview.png"))


if __name__ == "__main__":
    main()
