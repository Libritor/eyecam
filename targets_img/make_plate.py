"""Colour plates for a report-free colour read-out: a symbol a child or a
non-reader knows (the LEA set: house, circle, square, apple) in one of the
eight corner colours on a ground of another, scanned by the three planes, and
named from the EEG alone, symbol and colours both. The idea is the Ishihara
plate's job done without the subject saying anything; the colours are the
display's corners, so the plate reads colour but is not yet a deficiency test
(that needs isoluminant stimuli, see the paper's Section V).

Sixteen candidates: four symbols by four colour pairs (red on green, green on
red, blue on yellow, yellow on blue), chance one in sixteen.

    python targets_img/make_plate.py --sheet            # every plate + targets_img/plates_preview.png
    python targets_img/make_plate.py --draw             # one at random, sealed: plate_today.png + plate_today_seal.json
    .\\quest_run.ps1 -Run cphoto -Target targets_img\\plate_today.png     (10 x 10, about 30 min)
    python targets_img/make_plate.py readout runs/vr_cphoto1 --seal targets_img/plate_today_seal.json
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

COLOURS = dict(red=(255, 0, 0), green=(0, 255, 0), blue=(0, 0, 255), yellow=(255, 255, 0))
PAIRS = [("red", "green"), ("green", "red"), ("blue", "yellow"), ("yellow", "blue")]   # (figure, ground)

S = {
    "house": ["...11...",
              "..1111..",
              ".111111.",
              "11111111",
              ".111111.",
              ".111111.",
              ".11..11.",
              ".11..11."],
    "circle": ["..1111..",
               ".111111.",
               "11111111",
               "11111111",
               "11111111",
               "11111111",
               ".111111.",
               "..1111.."],
    "square": ["........",
               ".111111.",
               ".111111.",
               ".111111.",
               ".111111.",
               ".111111.",
               ".111111.",
               "........"],
    "apple": ["....1...",
              "...1....",
              ".11..11.",
              "11111111",
              "11111111",
              "11111111",
              ".111111.",
              "..1111.."],
}
SYMBOLS = list(S)


def mask(symbol, n=10):
    m = np.array([[c == "1" for c in row] for row in S[symbol]], bool)
    pad = (n - 8) // 2
    out = np.zeros((n, n), bool)
    out[pad:pad + 8, pad:pad + 8] = m
    return out


def plate(symbol, fig, ground, n=10):
    m = mask(symbol, n)
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = COLOURS[ground]
    img[m] = COLOURS[fig]
    return img


def candidates(n=10):
    return [((s, f, g), plate(s, f, g, n)) for s in SYMBOLS for f, g in PAIRS]


def big(img, size=240, grid=True):
    n = img.shape[0]
    im = Image.fromarray(img, "RGB").resize((size, size), Image.NEAREST)
    if grid:
        d = ImageDraw.Draw(im)
        for i in range(1, n):
            p = i * size // n
            d.line([(p, 0), (p, size)], fill=(110, 110, 110))
            d.line([(0, p), (size, p)], fill=(110, 110, 110))
    return im


def cmd_sheet(a):
    os.makedirs(os.path.join(HERE, "plates"), exist_ok=True)
    cands = candidates(a.n)
    size, cap, pad = 200, 20, 8
    cols = len(PAIRS)
    sheet = Image.new("RGB", (cols * (size + pad) + pad, len(SYMBOLS) * (size + cap + pad) + pad), (70, 70, 70))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for (s, f, g), img in cands:
        Image.fromarray(img, "RGB").save(os.path.join(HERE, "plates", f"plate_{s}_{f}_on_{g}_{a.n}x{a.n}.png"))
        r, c = SYMBOLS.index(s), PAIRS.index((f, g))
        x, y = pad + c * (size + pad), pad + r * (size + cap + pad)
        sheet.paste(big(img, size), (x, y))
        d.rectangle([x, y + size, x + size, y + size + cap], fill=(40, 40, 40))
        d.text((x + 4, y + size + 3), f"{s}, {f} on {g}", fill=(230, 230, 230), font=font)
    out = os.path.join(HERE, "plates_preview.png")
    sheet.save(out)
    print(f"{len(cands)} plates in targets_img/plates/, preview {out}; "
          f"one plate at {a.n} x {a.n} is {a.n * a.n} cells, about {a.n * a.n * 18 / 60:.0f} min through the planes")


def cmd_draw(a):
    rng = random.SystemRandom()
    s, (f, g) = rng.choice(SYMBOLS), rng.choice(PAIRS)
    img = plate(s, f, g, a.n)
    Image.fromarray(img, "RGB").save(os.path.join(HERE, "plate_today.png"))
    big(img).save(os.path.join(HERE, "plate_today_preview.png"))
    salt = "%016x" % rng.getrandbits(64)
    answer = f"{s}|{f}|{g}"
    seal = dict(n=a.n, salt=salt, sha256=hashlib.sha256((answer + salt).encode()).hexdigest(),
                sealed_answer=answer, note="the read-out decides from the EEG before this file is read; "
                                            "do not open it by hand")
    with open(os.path.join(HERE, "plate_today_seal.json"), "w") as fh:
        json.dump(seal, fh, indent=1)
    print(f"drawn and sealed: targets_img/plate_today.png ({a.n} x {a.n}); "
          f"run .\\quest_run.ps1 -Run cphoto -Target targets_img\\plate_today.png")


def corr(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def cmd_readout(a):
    import smoothcolor
    grid = np.load(os.path.join(a.session, "reconstruction_planes_grid.npy"))
    gh, gw, _ = grid.shape
    assert gh == gw, "plates are square"
    n = gh
    scaled = np.zeros_like(grid)
    snapped = np.zeros_like(grid)
    for k in range(3):
        v = grid[:, :, k]
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        scaled[:, :, k] = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        snapped[:, :, k] = (v > smoothcolor.otsu_log(v)) if np.nanstd(v) > 0 else 0
    cands = candidates(n)
    rows = []
    for key, img in cands:
        t = img.astype(float) / 255
        rows.append((corr(scaled, t), corr(snapped, t), float((snapped == t).all(axis=2).mean()), key))
    rows.sort(key=lambda r: -r[0])
    best = rows[0][3]
    print(f"EEG picture {n} x {n}: ranking by correlation with each plate (continuous | snapped | cells agreeing)")
    for r_c, r_s, agree, (s, f, g) in rows[:5]:
        print(f"  {r_c:+.3f}  {r_s:+.3f}  {agree:5.1%}  {s}, {f} on {g}")
    print(f"decision: {best[0]}, {best[1]} on {best[2]}  (chance 1 in {len(cands)})")
    res = dict(session=a.session, n=n, decision=dict(symbol=best[0], figure=best[1], ground=best[2]),
               ranking=[dict(symbol=s, figure=f, ground=g, r_continuous=r_c, r_snapped=r_s, cells_agree=agree)
                        for r_c, r_s, agree, (s, f, g) in rows], chance=1 / len(cands))
    if a.seal:
        seal = json.load(open(a.seal))
        ans = seal["sealed_answer"]
        ok = hashlib.sha256((ans + seal["salt"]).encode()).hexdigest() == seal["sha256"]
        s, f, g = ans.split("|")
        right = (s, f, g) == best
        print(f"seal opened: {s}, {f} on {g}  ({'hash ok' if ok else 'HASH MISMATCH'}) -> "
              f"{'RIGHT' if right else 'wrong'}; symbol {'right' if s == best[0] else 'wrong'}, "
              f"colours {'right' if (f, g) == best[1:] else 'wrong'}")
        res["seal"] = dict(symbol=s, figure=f, ground=g, hash_ok=ok, right=right)
    # figure: EEG picture, snapped, decided plate, (shown)
    panels = [(big((scaled * 255).astype(np.uint8), grid=False), "from the EEG, each plane scaled"),
              (big((snapped * 255).astype(np.uint8)), "snapped at each plane's threshold"),
              (big(plate(*best, n)), f"decided: {best[0]}, {best[1]} on {best[2]}")]
    if a.seal:
        panels.append((big(plate(s, f, g, n)), f"shown: {s}, {f} on {g}"))
    size, cap, pad = 240, 20, 8
    fig = Image.new("RGB", (len(panels) * (size + pad) + pad, size + cap + 2 * pad), (70, 70, 70))
    d = ImageDraw.Draw(fig)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, text) in enumerate(panels):
        x = pad + i * (size + pad)
        fig.paste(im, (x, pad))
        d.rectangle([x, pad + size, x + size, pad + size + cap], fill=(40, 40, 40))
        d.text((x + 4, pad + size + 3), text, fill=(230, 230, 230), font=font)
    fig.save(os.path.join(a.session, "plate_readout.png"))
    with open(os.path.join(a.session, "plate_readout.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {a.session}/plate_readout.png, plate_readout.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="grid size, 8 or 10 (the symbol is 8 wide)")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--draw", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("readout")
    p.add_argument("session")
    p.add_argument("--seal", default=None)
    a = ap.parse_args()
    if a.cmd == "readout":
        cmd_readout(a)
    elif a.draw:
        cmd_draw(a)
    else:
        cmd_sheet(a)


if __name__ == "__main__":
    main()
