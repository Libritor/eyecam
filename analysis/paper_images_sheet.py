"""One sheet: the officer, Prof Mann, J. C. Bose and the colour plates, each as the
picture and as the targets the eye would be scanned with."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = r"C:\Users\alexa\eyecam"
sys.path.insert(0, os.path.join(ROOT, "targets_img"))
import make_photo_gallery as g  # noqa: E402

OUT = sys.argv[1]
BOSE = r"C:\Users\alexa\Downloads\bose\bose_1926_meurisse.jpg"
g.CROPS["bose"] = (0.40, 0.11, 0.69, 0.37)

S, pad, cap = 300, 10, 22


def big(arr_or_img, grid=True):
    if isinstance(arr_or_img, np.ndarray):
        a = arr_or_img
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=2)
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    else:
        im = arr_or_img.convert("RGB")
    n = im.size[0]
    out = im.resize((S, S), Image.NEAREST if n <= 40 else Image.LANCZOS)
    if grid and n <= 40:
        d = ImageDraw.Draw(out)
        for i in range(1, n):
            p = i * S // n
            d.line([(p, 0), (p, S)], fill=(110, 110, 110))
            d.line([(0, p), (S, p)], fill=(110, 110, 110))
    return out


def fit(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    side = min(w, h)
    return im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)).resize((S, S), Image.LANCZOS)


rows = []
# officer
rows.append([(fit(Image.open(os.path.join(ROOT, "targets_img", "officer_illustration.png"))), "officer: the flat illustration (drawn by code, no real person)"),
             (big(Image.open(os.path.join(ROOT, "targets_img", "officer_8x10.png"))), "officer 8 x 10, four corner colours, 24 min through the planes"),
             (big(Image.open(os.path.join(ROOT, "targets_img", "officer_12x14.png"))), "officer 12 x 14, 50 min")])
# mann
head = Image.open(r"C:\Users\alexa\Downloads\SteveMannheadshot.jpg").convert("RGB").crop((180, 60, 1020, 900)).resize((S, S), Image.LANCZOS)
t12, h12 = g.target_from_photo(os.path.join(ROOT, "targets_img", "photos", "mann.png"), "mann", 12, 2)
t16, _ = g.target_from_photo(os.path.join(ROOT, "targets_img", "photos", "mann.png"), "mann", 16, 4)
rows.append([(head, "Mann: colour studio headshot (for a colour run after the wedge)"),
             (fit(h12), "Mann: the 2019 portrait, as in the gallery"),
             (big(t12), "Mann 12 x 12, two levels (the sealed photo run, 21 min)"),
             (big(t16), "Mann 16 x 16, four levels (36 min)")])
# bose
im_b = ImageOps.exif_transpose(Image.open(BOSE))
b12, hb = g.target_from_photo(BOSE, "bose", 12, 2)
b16, _ = g.target_from_photo(BOSE, "bose", 16, 4)
rows.append([(fit(im_b), "J. C. Bose, 1926 (Agence Meurisse, public domain, Wikimedia Commons)"),
             (fit(hb), "Bose: the gallery crop"),
             (big(b12), "Bose 12 x 12, two levels"),
             (big(b16), "Bose 16 x 16, four levels")])
# plates
pl = [("house", "red", "green"), ("circle", "green", "red"), ("square", "blue", "yellow"), ("apple", "yellow", "blue")]
rows.append([(big(Image.open(os.path.join(ROOT, "targets_img", "plates", f"plate_{s}_{f}_on_{gr}_10x10.png"))), f"plate: {s}, {f} on {gr} (one of 16, sealed, 30 min)") for s, f, gr in pl])

cols = max(len(r) for r in rows)
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
sheet.save(OUT)
print("wrote", OUT)
