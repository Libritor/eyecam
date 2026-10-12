"""Real photographs as scan targets, and the forced-choice gallery that
makes "recognised" a number. Each photo in targets_img/photos/ is cropped to
the head, averaged per scan position and rounded to a few grey levels, the
way the 2019 paper thresholded its portrait before scanning it. One of them
is drawn at random and sealed; analysis/photo_readout.py then names the
person from the EEG picture alone and opens the seal.

  photo_<name>_<n>.png        one target per photo (n x n cells, `levels` greys)
  photo_gallery_preview.png   the photos, the crops and the targets side by side
  photo_today.png + photo_today_seal.json   the drawn target and its answer (--draw)

    python targets_img/make_photo_gallery.py                 # all targets + the sheet
    python targets_img/make_photo_gallery.py --draw          # today's sealed photo
    python targets_img/make_photo_gallery.py --n 16 --levels 3

The photos: Mann (the paper's own high-contrast portrait, already in the
repo), Lincoln (Gardner, 1863), Einstein (Schmutzer, 1921) and Tesla
(Sarony, about 1890), the last three public-domain prints from Wikimedia
Commons, stored here at 720 px in grey. Add your own as photos/<name>.png or
.jpg and a crop box for it in CROPS (fractions of width and height, the head
from the top of the hair to the chin); a photo without a box is cropped to
its centre square.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "photos")
sys.path.insert(0, HERE)
from make_mann import tile  # noqa: E402

# (x0, y0, x1, y1) as fractions: a square around the head, hair top to chin
CROPS = {
    "mann": (0.06, 0.02, 0.97, 0.93),
    "lincoln": (0.05, 0.03, 0.89, 0.68),
    "einstein": (0.10, 0.02, 0.90, 0.62),
    "tesla": (0.11, 0.03, 0.81, 0.55),
    # the police officer (a public-domain official portrait, Rockville MD):
    # head and shoulders, so the cap-less head, the dark uniform and the
    # badge are all in the frame
    "officer": (0.16, 0.05, 0.86, 0.75),
}


def crop_box(im, name):
    w, h = im.size
    if name in CROPS:
        x0, y0, x1, y1 = CROPS[name]
        box = [x0 * w, y0 * h, x1 * w, y1 * h]
        # make it square about its centre, the larger side wins
        side = max(box[2] - box[0], box[3] - box[1])
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    else:
        side = min(w, h)
        box = [(w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2]
    return tuple(int(round(v)) for v in box)


def otsu(x):
    """Threshold between two groups of cell means (Otsu on the values)."""
    v = np.sort(np.asarray(x, float).ravel())
    best, thr = -1.0, float(np.median(v))
    for k in range(1, len(v)):
        a, b = v[:k], v[k:]
        between = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if between > best:
            best, thr = between, (v[k - 1] + v[k]) / 2
    return thr


def quantise(small, levels):
    """Cell means in [0, 1] to `levels` grey levels in [0, 1]; two levels
    split at Otsu, more are spread between the 5th and 95th percentiles."""
    if levels == 2:
        return (small > otsu(small)).astype(float)
    lo, hi = np.percentile(small, [5, 95])
    q = np.clip((small - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.round(q * (levels - 1)) / (levels - 1)


def target_from_photo(path, name, n, levels):
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    box = crop_box(im, name)
    head = ImageOps.autocontrast(im.crop(box), cutoff=1)
    small = np.asarray(head.resize((n, n), Image.BOX), float) / 255.0
    return quantise(small, levels), head


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def robustness(targets, flip=0.25, trials=400, seed=0):
    """How often the forced choice names the right photo when a fraction of
    the cells come out wrong (flipped to another level) and the rest carry
    the shown level plus noise: the margin the gallery leaves."""
    rng = np.random.default_rng(seed)
    names = list(targets)
    right = 0
    for t in range(trials):
        k = names[t % len(names)]
        g = targets[k].copy()
        m = rng.random(g.shape) < flip
        g[m] = 1 - g[m]
        g = g + rng.normal(0, 0.3, g.shape)
        best = max(names, key=lambda nm: np.nan_to_num(corr(g, targets[nm]), nan=-9))
        right += best == k
    return right / trials


def photos():
    out = []
    for p in sorted(glob.glob(os.path.join(PHOTOS, "*"))):
        if p.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append((os.path.splitext(os.path.basename(p))[0].lower(), p))
    return out


def gallery(n, levels):
    targets, heads = {}, {}
    for name, p in photos():
        targets[name], heads[name] = target_from_photo(p, name, n, levels)
    return targets, heads


def save_target(a, name):
    Image.fromarray((a * 255).round().astype(np.uint8), mode="L").save(os.path.join(HERE, name + ".png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12, help="cells per side (12: 144 positions, about 25 min)")
    ap.add_argument("--levels", type=int, default=2, help="grey levels (2 = black and white, as the paper)")
    ap.add_argument("--draw", action="store_true", help="draw one photo at random and seal the answer")
    a = ap.parse_args()
    targets, heads = gallery(a.n, a.levels)
    if len(targets) < 2:
        raise SystemExit(f"need at least two photos in {PHOTOS}")
    names = list(targets)
    files = {}
    for name in names:
        fname = f"photo_{name}_{a.n}"
        save_target(targets[name], fname)
        files[name] = f"targets_img/{fname}.png"
    if a.draw:
        name = random.choice(names)
        save_target(targets[name], "photo_today")
        seal = dict(name=name, file="photo_today.png", n=a.n, levels=a.levels, gallery=files,
                    drawn=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(seal, open(os.path.join(HERE, "photo_today_seal.json"), "w"), indent=1)
        print(f"drawn: targets_img/photo_today.png ({a.n} x {a.n}, {a.levels} levels, one of "
              f"{', '.join(names)}); the answer is sealed in targets_img/photo_today_seal.json, "
              "do not open it before the read-out")
        return
    for name in names:
        t = targets[name]
        print(f"photo_{name}_{a.n}: {a.n}x{a.n}, {a.levels} levels, {int((t > 0).sum())} lit of {t.size}")
    print("correlation between the targets (what the forced choice has to separate):")
    for i, x in enumerate(names):
        row = "  ".join(f"{corr(targets[x], targets[y]):+.2f}" for y in names)
        print(f"  {x:10s} {row}")
    print(f"forced choice right with a quarter of the cells wrong: "
          f"{100 * robustness(targets):.0f}% (chance {100 / len(names):.0f}%)")
    # the sheet: photo, crop, target for each
    tiles = []
    for name in names:
        im = ImageOps.exif_transpose(Image.open(dict(photos())[name])).convert("L")
        box = crop_box(im, name)
        w, h = im.size
        s = 300 / max(w, h)
        shown = im.resize((round(w * s), round(h * s)), Image.LANCZOS).convert("RGB")
        d = ImageDraw.Draw(shown)
        d.rectangle([v * s for v in box], outline=(255, 80, 0), width=3)
        canvas = Image.new("RGB", (300, 322), (30, 30, 30))
        canvas.paste(shown, ((300 - shown.width) // 2, 0))
        ImageDraw.Draw(canvas).text((4, 304), f"{name}: the photo and the crop", fill=(230, 230, 230))
        tiles.append(canvas)
        head = heads[name].resize((300, 300), Image.LANCZOS).convert("RGB")
        canvas = Image.new("RGB", (300, 322), (30, 30, 30))
        canvas.paste(head, (0, 0))
        ImageDraw.Draw(canvas).text((4, 304), "the head, as the eye would see it", fill=(230, 230, 230))
        tiles.append(canvas)
        tiles.append(tile((targets[name] * 255).round().astype(np.uint8),
                          f"target {a.n}x{a.n}, {a.levels} levels, {int((targets[name] > 0).sum())} lit"))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 310 + 10, rows * 332 + 10), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % cols) * 310, 10 + (i // cols) * 332))
    sheet.save(os.path.join(HERE, "photo_gallery_preview.png"))
    print("sheet: targets_img/photo_gallery_preview.png")


if __name__ == "__main__":
    main()
