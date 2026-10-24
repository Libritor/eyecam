"""What a real colour photograph becomes through the three colour planes.

For each photograph: the crop, its cell averages at the grid (the best a graded
read-out could give), the same snapped to the eight corner colours at each
plane's Otsu threshold (what the planes scoring does now), and one draw of the
snapped picture with each plane read at the fidelity of the best eight-colour
sitting (vr_planes9: r 0.79 per plane), so the expected brain picture is seen
before a 30 to 45 minute sitting is spent on it. Cells with any plane within
--near of its threshold (fraction of the plane's range) are counted: those are
the cells that flip.

    python analysis/colour_photo_sheet.py --out sheet.png photo1.jpg[:x0,y0,x1,y1] photo2.jpg ...
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from smoothcolor import otsu_log  # noqa: E402

CORNERS = np.array([[r, g, b] for r in (0, 1) for g in (0, 1) for b in (0, 1)], float)
NAMES = ["black", "blue", "green", "cyan", "red", "magenta", "yellow", "white"]
PLANE_S = 18.0


def cells(img, n):
    """n x n cell averages of an RGB image in [0, 1]."""
    a = np.asarray(img.convert("RGB"), float) / 255.0
    h, w, _ = a.shape
    out = np.zeros((n, n, 3))
    for gy in range(n):
        for gx in range(n):
            out[gy, gx] = a[gy * h // n:(gy + 1) * h // n, gx * w // n:(gx + 1) * w // n].mean(axis=(0, 1))
    return out


def snap(grid, near, rng=None, r=None):
    """Snap each plane at its Otsu threshold (on the log response, as the planes scoring).
    With r, each plane is first read through noise so that its correlation with the
    truth is about r, one draw. Returns (snapped rgb, n_near, n_wrong vs clean snap)."""
    on = np.zeros(grid.shape, bool)
    n_near = np.zeros(grid.shape[:2], bool)
    for k in range(3):
        v = grid[:, :, k].copy()
        if v.std() < 1e-6:
            on[:, :, k] = v.mean() > 0.5
            continue
        thr = otsu_log(v)
        span = max(np.percentile(v, 98) - np.percentile(v, 2), 1e-9)
        n_near |= np.abs(v - thr) / span < near
        if r is not None:
            sd = v.std() * np.sqrt(1 / r ** 2 - 1)
            v = v + rng.normal(0, sd, v.shape)
            thr = otsu_log(np.clip(v, 1e-6, None)) if v.min() > 0 else otsu_log(v - v.min() + 1e-3) + v.min() - 1e-3
        on[:, :, k] = v > thr
    return on.astype(float), int(n_near.sum())


def tile(rgb, size, grid_lines=True):
    n = rgb.shape[0]
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").resize((size, size), Image.NEAREST)
    if grid_lines:
        d = ImageDraw.Draw(im)
        for i in range(1, n):
            p = i * size // n
            d.line([(p, 0), (p, size)], fill=(128, 128, 128), width=1)
            d.line([(0, p), (size, p)], fill=(128, 128, 128), width=1)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="path[:x0,y0,x1,y1] crop box in pixels")
    ap.add_argument("--grids", default="12,16")
    ap.add_argument("--near", type=float, default=0.1)
    ap.add_argument("--r", type=float, default=0.79, help="per-plane fidelity of the simulated read-out")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grids = [int(g) for g in a.grids.split(",")]
    rng = np.random.default_rng(a.seed)
    S, pad, cap = 300, 10, 22
    cols = 1 + 3 * len(grids)
    rows = len(a.photos)
    sheet = Image.new("RGB", (cols * (S + pad) + pad, rows * (S + cap + pad) + pad), (70, 70, 70))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for row, spec in enumerate(a.photos):
        path, box = (spec.rsplit(":", 1) + [None])[:2] if spec.count(":") > 1 else (spec, None)
        img = Image.open(path).convert("RGB")
        if box:
            x0, y0, x1, y1 = [int(v) for v in box.split(",")]
            img = img.crop((x0, y0, x1, y1))
        name = os.path.splitext(os.path.basename(path))[0]
        y = pad + row * (S + cap + pad)
        panels = [(img.resize((S, S)), f"{name}: the photograph, cropped")]
        for n in grids:
            g = cells(img, n)
            clean, n_near = snap(g, a.near)
            noisy, _ = snap(g, a.near, rng, a.r)
            used = sorted({NAMES[int(c[0] * 4 + c[1] * 2 + c[2])] for c in clean.reshape(-1, 3)})
            wrong = int((clean != noisy).any(axis=2).sum())
            mins = n * n * PLANE_S / 60
            panels.append((tile(g, S), f"{n}x{n} cell averages (graded read-out), {mins:.0f} min a pass"))
            panels.append((tile(clean, S), f"{n}x{n} snapped to the corners: {len(used)} colours, {n_near} cells near a threshold"))
            panels.append((tile(noisy, S), f"{n}x{n} one read-out at r {a.r:.2f} a plane: {wrong} cells differ"))
        for c, (im, text) in enumerate(panels):
            x = pad + c * (S + pad)
            sheet.paste(im, (x, y))
            d.rectangle([x, y + S, x + S, y + S + cap], fill=(40, 40, 40))
            d.text((x + 4, y + S + 4), text, fill=(230, 230, 230), font=font)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
