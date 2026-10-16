"""The colour photograph as a graded read-out, interpolated for display the way
the README shows a grey scan (one value per position, then bicubic).

Per photograph and grid: the cell averages interpolated (the ceiling, every
plane read exactly) and one read-out with each plane at the fidelity of the
best eight-colour sitting (r 0.79 a plane, Gaussian noise on the graded
value, one draw), interpolated the same way. No threshold anywhere: this is
what the planes decoder's own continuous picture would look like if the
response grades with brightness, which the wedge run has to show.

    python analysis/colour_photo_smooth.py --grids 12,16,18,20 --out sheet.png photo.jpg[:x0,y0,x1,y1] ...
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colour_photo_sheet import cells, PLANE_S  # noqa: E402


def noisy_read(g, r, rng):
    out = g.copy()
    for k in range(3):
        v = g[:, :, k]
        sd = v.std() * np.sqrt(1 / r ** 2 - 1) if v.std() > 1e-6 else 0.0
        out[:, :, k] = v + rng.normal(0, sd, v.shape)
    return np.clip(out, 0, 1)


def smooth(rgb, size):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").resize((size, size), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="path[:x0,y0,x1,y1]")
    ap.add_argument("--grids", default="12,16,18,20")
    ap.add_argument("--r", type=float, default=0.79)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grids = [int(g) for g in a.grids.split(",")]
    rng = np.random.default_rng(a.seed)
    S, pad, cap = a.size, 10, 22
    cols = 1 + 2 * len(grids)
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
            mins = n * n * PLANE_S / 60
            rr = noisy_read(g, a.r, rng)
            corr = np.mean([np.corrcoef(g[:, :, k].ravel(), rr[:, :, k].ravel())[0, 1] for k in range(3) if g[:, :, k].std() > 1e-6])
            panels.append((smooth(g, S), f"{n}x{n} every plane read exactly, interpolated ({mins:.0f} min a pass)"))
            panels.append((smooth(rr, S), f"{n}x{n} one read-out at r {a.r:.2f} a plane, interpolated (r {corr:.2f})"))
        for c, (im, text) in enumerate(panels):
            x = pad + c * (S + pad)
            sheet.paste(im, (x, y))
            d.rectangle([x, y + S, x + S, y + S + cap], fill=(40, 40, 40))
            d.text((x + 4, y + S + 4), text, fill=(230, 230, 230), font=font)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
