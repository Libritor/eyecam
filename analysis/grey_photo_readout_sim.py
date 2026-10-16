"""What a grey portrait would come back as from a black/white scan, before the
sitting: the target as the run shows it (two or four grey levels at the grid),
one read-out with every cell at the fidelity of the NO run (r 0.91, Gaussian
noise on the cell value, one draw), the same interpolated for display as the
README figure does, and the read-out thresholded back to the target's levels
with the cells-right count. Simulation from the photograph; no EEG.

    python analysis/grey_photo_readout_sim.py targets_img/photos_more/bose.png bose --out bose_sim.png
    python analysis/grey_photo_readout_sim.py targets_img/photos/mann.png mann --grids 12:2,16:4 --r 0.91
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "targets_img"))
import make_photo_gallery as g  # noqa: E402


def tile(a, S, interp=False, grid=True):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
    out = im.resize((S, S), Image.BICUBIC if interp else Image.NEAREST)
    if grid and not interp:
        d = ImageDraw.Draw(out)
        n = a.shape[0]
        for i in range(1, n):
            p = i * S // n
            d.line([(p, 0), (p, S)], fill=(110, 110, 110))
            d.line([(0, p), (S, p)], fill=(110, 110, 110))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("name", help="crop name in make_photo_gallery.CROPS")
    ap.add_argument("--grids", default="12:2,16:4", help="grid:levels, comma separated")
    ap.add_argument("--r", type=float, default=0.91, help="cell fidelity of the black/white scan (the NO run)")
    ap.add_argument("--passes", type=int, default=1, help="passes averaged (noise falls as 1/sqrt)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    S, pad, cap = 260, 10, 22
    rows = []
    for spec in a.grids.split(","):
        n, levels = (int(v) for v in spec.split(":"))
        target, head = g.target_from_photo(a.photo, a.name, n, levels)
        sd = target.std() * np.sqrt(1 / a.r ** 2 - 1) / np.sqrt(a.passes)
        read = target + rng.normal(0, sd, target.shape)
        corr = float(np.corrcoef(target.ravel(), read.ravel())[0, 1])
        back = g.quantise(np.clip(read, 0, 1), levels) if levels > 2 else (read > g.otsu(read)).astype(float)
        right = int((back == target).sum())
        mins = n * n * 8 / 60 * a.passes
        rows.append([(head.convert("RGB").resize((S, S)), f"{a.name}: the crop"),
                     (tile(target, S), f"shown: {n} x {n}, {levels} levels ({mins:.0f} min, {a.passes} pass{'es' if a.passes > 1 else ''})"),
                     (tile(np.clip(read, 0, 1), S), f"from the EEG, one value a cell (r {corr:.2f})"),
                     (tile(np.clip(read, 0, 1), S, interp=True), "same values, interpolated for display"),
                     (tile(back, S), f"back to {levels} levels: {right} of {n * n} cells right")])
    cols = len(rows[0])
    sheet = Image.new("RGB", (cols * (S + pad) + pad, len(rows) * (S + cap + pad) + pad), (70, 70, 70))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for r, row in enumerate(rows):
        y = pad + r * (S + cap + pad)
        for c, (im, text) in enumerate(row):
            x = pad + c * (S + pad)
            sheet.paste(im, (x, y))
            d.rectangle([x, y + S, x + S, y + S + cap], fill=(40, 40, 40))
            d.text((x + 4, y + S + 4), text, fill=(230, 230, 230), font=font)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
