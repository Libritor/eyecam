"""Put a colour picture back together from two sessions: brightness from a
grey scan (runs/vr_greyN, reconstruction_grid.npy at n x n) and colour from a
planes scan of the coarse colour target (runs/vr_cphotoN,
reconstruction_planes_grid.npy at m x m). The colour grid's own brightness is
removed so only its chroma is kept, the chroma is interpolated up to n x n
and added to the brightness. Scores: r of the brightness read-out against
its target, r of each plane against the colour target, and r of the combined
picture against the pair's ceiling (the two targets put together the same way).
Each read-out is put on its target's scale by matching mean and sd (the ear
decoder's rule), nothing else of the target is used.

    python analysis/colour_combine.py runs/vr_grey1 runs/vr_cphoto2 --name lincoln_c

Writes <grey session>/colour_combined.png (one value a cell), _interp.png
(bicubic, for display), colour_combine_sheet.png and colour_combine.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "targets_img"))
from make_colour_pair import combine, tile, LUMA  # noqa: E402


def scale(v, like):
    """Read-out mapped onto the target's scale: its mean and sd matched to the
    target's (the ear decoder's rule for loudness), then clipped to [0, 1]."""
    v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
    z = (v - v.mean()) / (v.std() + 1e-9)
    return np.clip(like.mean() + like.std() * z, 0, 1)


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grey_session")
    ap.add_argument("planes_session")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    luma_read = np.load(os.path.join(a.grey_session, "reconstruction_grid.npy"))
    luma_tgt = np.load(os.path.join(a.grey_session, "target.npy"))
    col_read = np.load(os.path.join(a.planes_session, "reconstruction_planes_grid.npy"))
    col_tgt = np.load(os.path.join(a.planes_session, "target_color.npy"))
    n = luma_read.shape[0]
    assert luma_read.shape == luma_tgt.shape and luma_read.shape[0] == luma_read.shape[1], "grey grid must be square"
    assert col_read.shape == col_tgt.shape, "planes grid and its target differ"
    L = scale(luma_read, luma_tgt)
    C = np.stack([scale(col_read[:, :, k], col_tgt[:, :, k]) for k in range(3)], axis=2)
    combined = combine(L, C, n)
    ceiling = combine(luma_tgt, col_tgt, n)
    r_luma = corr(luma_read, luma_tgt)
    r_planes = [corr(col_read[:, :, k], col_tgt[:, :, k]) for k in range(3)]
    r_comb = corr(combined, ceiling)
    r_chroma = corr(combined - (combined @ LUMA)[:, :, None], ceiling - (ceiling @ LUMA)[:, :, None])
    S = 24 * n
    Image.fromarray((combined * 255).astype(np.uint8), "RGB").resize((S, S), Image.NEAREST).save(os.path.join(a.grey_session, "colour_combined.png"))
    Image.fromarray((combined * 255).astype(np.uint8), "RGB").resize((S, S), Image.BICUBIC).save(os.path.join(a.grey_session, "colour_combined_interp.png"))
    panels = [(tile(ceiling, 300), f"shown: brightness {n}x{n} + colour {col_tgt.shape[1]}x{col_tgt.shape[0]}"),
              (tile(combined, 300), f"from the EEG, one value a cell (r {r_comb:.2f} to the ceiling)"),
              (Image.fromarray((combined * 255).astype(np.uint8), "RGB").resize((300, 300), Image.BICUBIC), "same values, interpolated for display"),
              (tile(L, 300), f"brightness alone (r {r_luma:.2f})"),
              (tile(C, 300), "colour alone, planes r " + ", ".join(f"{v:.2f}" for v in r_planes))]
    pad, cap = 10, 22
    sheet = Image.new("RGB", (len(panels) * (300 + pad) + pad, 300 + cap + 2 * pad), (70, 70, 70))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, (p, text) in enumerate(panels):
        x = pad + i * (300 + pad)
        sheet.paste(p, (x, pad))
        d.rectangle([x, pad + 300, x + 300, pad + 300 + cap], fill=(40, 40, 40))
        d.text((x + 4, pad + 300 + 4), text, fill=(230, 230, 230), font=font)
    sheet.save(os.path.join(a.grey_session, "colour_combine_sheet.png"))
    res = dict(name=a.name, grey_session=a.grey_session, planes_session=a.planes_session, n=n,
               colour_grid=[int(col_tgt.shape[1]), int(col_tgt.shape[0])], r_luma=r_luma, r_planes=r_planes,
               r_combined_vs_ceiling=r_comb, r_chroma_vs_ceiling=r_chroma,
               note="each read-out's mean and sd matched to its target's; chroma = colour grid minus its own brightness, bilinear to n")
    json.dump(res, open(os.path.join(a.grey_session, "colour_combine.json"), "w"), indent=1)
    print(f"brightness r {r_luma:.2f}; planes r {', '.join(f'{v:.2f}' for v in r_planes)}; "
          f"combined picture r {r_comb:.2f} to the ceiling (chroma alone r {r_chroma:.2f})")
    print(f"wrote {a.grey_session}/colour_combined.png, colour_combined_interp.png, colour_combine_sheet.png, colour_combine.json")


if __name__ == "__main__":
    main()
