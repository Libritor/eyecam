"""A colour photograph as a planes target: what the three binary planes can
carry is eight colours per cell (each of red, green and blue lit or dark), so
a photo is cut to the grid, averaged per scan position, and each channel is
split at its own threshold. The result is a poster in the eight pure colours,
pixel for pixel what `--mode planes --color-target <png>` will show. The
sheet says, before any EEG, whether the thing is recognisable at that size
and how many cells sit near a threshold (the ones most likely to come out
wrong).

    python targets_img/make_colour_photo.py photo.png [--name officer_photo]
        [--n 12 | --w 12 --h 14] [--crop x0,y0,x1,y1] [--sat 1.5] [--thr otsu|0.5]

Writes targets_img/<name>_<w>x<h>.png (the target) and <name>_preview.png.
Scan time at 5 / 5 / 8 s per position, one pass each: 18 s per cell, so
8 x 10 is 24 min, 12 x 12 is 43 min, 12 x 14 is 50 min, plus rest breaks.

Rules of thumb from the grey and colour runs: a region has to be several
cells wide to survive one cell in four being wrong; colours should sit at
the corners of the RGB cube (blue uniform, yellow badge, red tie, white
shirt, black hair, green or cyan background), because a mid colour such as
skin lands near a threshold in one or two planes and comes out as any of
its neighbours. A flat-coloured illustration survives; a photograph with
shading mostly does not.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_photo_gallery import otsu  # noqa: E402

NAMES = {(0, 0, 0): "black", (1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue",
         (1, 1, 0): "yellow", (1, 0, 1): "magenta", (0, 1, 1): "cyan", (1, 1, 1): "white"}


def load(path, crop=None, sat=1.5):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop:
        w, h = im.size
        x0, y0, x1, y1 = crop
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    im = ImageOps.autocontrast(im, cutoff=1)
    if sat != 1.0:
        im = ImageEnhance.Color(im).enhance(sat)
    return im


def poster(im, w, h, thr="otsu"):
    """(planes w x h x 3 in {0,1}, margin w x h: distance of the nearest
    channel to its threshold, in units of the channel's range)."""
    small = np.asarray(im.resize((w, h), Image.BOX), float) / 255.0
    planes = np.zeros_like(small)
    margin = np.ones(small.shape[:2])
    for c in range(3):
        x = small[:, :, c]
        t = otsu(x) if thr == "otsu" else float(thr)
        planes[:, :, c] = x > t
        margin = np.minimum(margin, np.abs(x - t))
    return planes, margin, small


def counts(planes):
    out = {}
    for row in planes.reshape(-1, 3):
        k = NAMES[tuple(int(v) for v in row)]
        out[k] = out.get(k, 0) + 1
    return out


def tile(rgb, label, size=300):
    im = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    h, w = rgb.shape[:2]
    s = size / max(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for k in range(w + 1):
        p = round(k * im.width / w)
        d.line([(p, 0), (p, im.height)], fill=(70, 70, 70))
    for k in range(h + 1):
        p = round(k * im.height / h)
        d.line([(0, p), (im.width, p)], fill=(70, 70, 70))
    out = Image.new("RGB", (size, size + 22), (30, 30, 30))
    out.paste(im, ((size - im.width) // 2, 0))
    ImageDraw.Draw(out).text((4, size + 4), label, fill=(230, 230, 230))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("photo")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=None, help="square grid side")
    ap.add_argument("--w", type=int, default=12)
    ap.add_argument("--h", type=int, default=12)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 as fractions")
    ap.add_argument("--sat", type=float, default=1.5, help="saturation boost before splitting")
    ap.add_argument("--thr", default="otsu", help="per-channel threshold: otsu or a number")
    a = ap.parse_args()
    w, h = (a.n, a.n) if a.n else (a.w, a.h)
    name = a.name or os.path.splitext(os.path.basename(a.photo))[0]
    crop = tuple(float(v) for v in a.crop.split(",")) if a.crop else None
    im = load(a.photo, crop, a.sat)
    planes, margin, small = poster(im, w, h, a.thr)
    out = os.path.join(HERE, f"{name}_{w}x{h}.png")
    Image.fromarray((planes * 255).astype(np.uint8), mode="RGB").save(out)
    near = margin < 0.08
    print(f"{out}: {w}x{h}, {w * h} positions, about {w * h * 18 / 60:.0f} min at 5/5/8 s")
    print("  colours: " + ", ".join(f"{k} {v}" for k, v in sorted(counts(planes).items(), key=lambda kv: -kv[1])))
    for c, nm in enumerate("RGB"):
        print(f"  {nm} plane: {int(planes[:, :, c].sum())} lit of {w * h}")
    print(f"  cells within 0.08 of a threshold in some plane (likely to come out wrong): {int(near.sum())} of {w * h}")
    # the sheet: the photo, the cell averages, the poster at this size and at two others
    tiles = []
    ph = im.copy()
    ph.thumbnail((300, 300))
    canvas = Image.new("RGB", (300, 322), (30, 30, 30))
    canvas.paste(ph, ((300 - ph.width) // 2, 0))
    ImageDraw.Draw(canvas).text((4, 304), "the photo (cropped, saturated)", fill=(230, 230, 230))
    tiles.append(canvas)
    tiles.append(tile(small, f"cell averages {w}x{h}"))
    tiles.append(tile(planes, f"eight colours {w}x{h}, {int(near.sum())} near a threshold"))
    for (w2, h2) in ((8, 10), (16, 16)):
        if (w2, h2) != (w, h):
            p2, m2, _ = poster(im, w2, h2, a.thr)
            tiles.append(tile(p2, f"eight colours {w2}x{h2}, {int((m2 < 0.08).sum())} near a threshold"))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 310 + 10, rows * 332 + 10), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % cols) * 310, 10 + (i // cols) * 332))
    sp = os.path.join(HERE, f"{name}_preview.png")
    sheet.save(sp)
    print(f"  sheet: {sp}")


if __name__ == "__main__":
    main()
