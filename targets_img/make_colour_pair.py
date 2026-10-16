"""A colour photograph as two targets: brightness at the full grid for a grey
scan, colour at a coarser grid for the planes scan. The colour picture is
put back together afterwards by analysis/colour_combine.py (brightness from
the grey session, colour from the planes session, as chroma subsampling in
television). Two sessions of 19 + 11 min at 12 x 12 with colour at 6 x 6,
against 43 min for the three planes at the full grid, and the colour noise
is blurred instead of landing on every cell.

    python targets_img/make_colour_pair.py photo.jpg lincoln_c --n 12 --levels 4 --sub 2
    python targets_img/make_colour_pair.py targets_img/photos_more/bose.png bose --n 16 --levels 4 --grey-only

Writes targets_img/<name>_luma_<n>x<n>.png (grey, `levels` grey levels),
<name>_chroma_<m>x<m>.png (RGB cell averages, saturation x `--sat`) and
<name>_pair_preview.png (crop | brightness target | colour target | the two
put together, which is the most the pair can give back). Crops come from
make_photo_gallery.CROPS by name, or --crop x0,y0,x1,y1 as fractions.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_photo_gallery as g  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114])


def combine(luma, chroma_rgb, n):
    """luma (n, n) in [0, 1] + chroma of an (m, m, 3) colour grid, bilinear to n."""
    ch = chroma_rgb - (chroma_rgb @ LUMA)[:, :, None]
    up = Image.fromarray(((ch + 1) / 2 * 255).astype(np.uint8), "RGB").resize((n, n), Image.BILINEAR)
    ch_up = np.asarray(up, float) / 255 * 2 - 1
    return np.clip(luma[:, :, None] + ch_up, 0, 1)


def tile(a, S, grid=True):
    a = np.asarray(a, float)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    n = a.shape[0]
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB").resize((S, S), Image.NEAREST)
    if grid:
        d = ImageDraw.Draw(im)
        for i in range(1, n):
            p = i * S // n
            d.line([(p, 0), (p, S)], fill=(110, 110, 110))
            d.line([(0, p), (S, p)], fill=(110, 110, 110))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--sub", type=int, default=2, help="colour grid = n / sub")
    ap.add_argument("--sat", type=float, default=1.4)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 as fractions; else make_photo_gallery.CROPS[name]")
    ap.add_argument("--grey-only", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    if a.crop:
        g.CROPS[a.name] = tuple(float(v) for v in a.crop.split(","))
    im = ImageOps.exif_transpose(Image.open(a.photo)).convert("RGB")
    box = g.crop_box(im, a.name)
    head = im.crop(box)
    grey = ImageOps.autocontrast(head.convert("L"), cutoff=1)
    small = np.asarray(grey.resize((a.n, a.n), Image.BOX), float) / 255.0
    luma = g.quantise(small, a.levels)
    luma_path = os.path.join(a.outdir, f"{a.name}_luma_{a.n}x{a.n}.png")
    Image.fromarray((luma * 255).round().astype(np.uint8), "L").save(luma_path)
    panels = [(head.resize((300, 300), Image.LANCZOS), f"{a.name}: the crop"),
              (tile(luma, 300), f"brightness {a.n}x{a.n}, {a.levels} levels: grey scan, {a.n * a.n * 8 / 60:.0f} min a pass")]
    msg = [f"brightness target {luma_path}", f"  .\\quest_run.ps1 -Run grey -Target {os.path.relpath(luma_path, os.path.dirname(HERE))}"]
    if not a.grey_only:
        m = max(a.n // a.sub, 2)
        col = ImageEnhance.Color(head).enhance(a.sat)
        chroma = np.asarray(col.resize((m, m), Image.BOX), float) / 255.0
        chroma_path = os.path.join(a.outdir, f"{a.name}_chroma_{m}x{m}.png")
        Image.fromarray((chroma * 255).round().astype(np.uint8), "RGB").save(chroma_path)
        both = combine(luma, chroma, a.n)
        panels += [(tile(chroma, 300), f"colour {m}x{m}: planes scan, {m * m * 18 / 60:.0f} min"),
                   (tile(both, 300), "the two together: the most the pair gives back")]
        msg += [f"colour target {chroma_path}", f"  .\\quest_run.ps1 -Run cphoto -Target {os.path.relpath(chroma_path, os.path.dirname(HERE))}",
                f"then: python analysis/colour_combine.py runs/vr_greyN runs/vr_cphotoN --name {a.name}",
                f"total about {a.n * a.n * 8 / 60 + m * m * 18 / 60:.0f} min of scanning plus two calibrations"]
    S, pad, cap = 300, 10, 22
    sheet = Image.new("RGB", (len(panels) * (S + pad) + pad, S + cap + 2 * pad), (70, 70, 70))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, (p, text) in enumerate(panels):
        x = pad + i * (S + pad)
        sheet.paste(p, (x, pad))
        d.rectangle([x, pad + S, x + S, pad + S + cap], fill=(40, 40, 40))
        d.text((x + 4, pad + S + 4), text, fill=(230, 230, 230), font=font)
    prev = os.path.join(a.outdir, f"{a.name}_pair_preview.png")
    sheet.save(prev)
    print("\n".join(msg))
    print(f"preview {prev}")


if __name__ == "__main__":
    main()
