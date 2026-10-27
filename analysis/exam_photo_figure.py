"""Photo figure for the eye-exam paper: (a) the rig on the subject (the Quest
over the Muse headband), (b) the headset's own view of the exam's intro screen
(adb screencap, left eye), (c)-(e) the stimulus page as rendered in the
browser during the acuity, field and colour tests (Tools/exam_screens.py
captures). Panel labels are text in the PDF; the photos are raster.

    python analysis/exam_photo_figure.py docs/paper/exam2027/figures/exam_photos.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6, "font.family": "DejaVu Sans"})


def page_frames(folder):
    """Pick one ON frame of the grating, the field and the colour test from the
    page captures by what the patch region holds."""
    picks = {}
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".png"):
            continue
        im = np.asarray(Image.open(os.path.join(folder, f)).convert("RGB"), float)
        h, w, _ = im.shape
        side = int(min(h, w) * 0.6)
        y0, x0 = (h - side) // 2, (w - side) // 2
        patch = im[y0:y0 + side, x0:x0 + side]
        row = patch[side // 4]                      # a row away from the dot
        grey = row.mean(1)
        sat = (patch.max(2) - patch.min(2)).mean()
        if "grating" not in picks and grey.std() > 60 and (grey > 200).mean() > 0.3 and (grey < 60).mean() > 0.3:
            picks["grating"] = f
        if "colour" not in picks and sat > 150:
            picks["colour"] = f
        if "field" not in picks:
            quads = [patch[: side // 2, : side // 2], patch[: side // 2, side // 2:],
                     patch[side // 2:, : side // 2], patch[side // 2:, side // 2:]]
            bright = [(q.mean(2) > 240).mean() for q in quads]
            if max(bright) > 0.15 and sorted(bright)[-2] < 0.02 and sat < 20:
                picks["field"] = f
    return picks


def main():
    out = sys.argv[1]
    fig = plt.figure(figsize=(3.5, 2.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], left=0.01, right=0.99, top=0.95, bottom=0.03,
                          hspace=0.3, wspace=0.08)
    # (a) the rig on the subject
    ax = fig.add_subplot(gs[0, :3])
    rig = Image.open(os.path.join(ROOT, "docs", "paper", "icce2027", "figures", "setup.jpg"))
    w, h = rig.size
    rig = rig.crop((int(w * 0.08), int(h * 0.33), int(w * 0.92), int(h * 0.80)))
    ax.imshow(rig); ax.axis("off")
    ax.set_title("(a) the rig: Quest 3S over the Muse headband", loc="left", fontsize=5.8, pad=3)
    # (b) the headset's view of the intro screen
    ax = fig.add_subplot(gs[0, 3:])
    view = Image.open(os.path.join(ROOT, "docs", "paper", "exam2027", "figures", "headset_view_raw.png"))
    w, h = view.size
    view = view.crop((int(w * 0.03), int(h * 0.155), int(w * 0.9), int(h * 0.31)))
    ax.imshow(view); ax.axis("off")
    ax.set_title("(b) the headset's view: intro screen", loc="left", fontsize=5.8, pad=3)
    # (c)-(e) the page during three tests
    picks = page_frames(os.path.join(ROOT, "Tools", "preview", "exam"))
    for k, (key, lab) in enumerate([("grating", "(c) acuity test"),
                                    ("field", "(d) field test"),
                                    ("colour", "(e) colour test")]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        ax.axis("off")
        if key not in picks:
            ax.text(0.5, 0.5, "no capture", ha="center", va="center", transform=ax.transAxes)
            continue
        im = Image.open(os.path.join(ROOT, "Tools", "preview", "exam", picks[key])).convert("RGB")
        w, h = im.size
        side = int(min(w, h) * 0.6)
        m = int(side * 0.18)
        im = im.crop(((w - side) // 2 - m, (h - side) // 2 - m, (w + side) // 2 + m, (h + side) // 2 + m))
        ax.imshow(im)
        ax.set_title(lab, loc="left", fontsize=5.8, pad=2)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    print("wrote", out, picks)


if __name__ == "__main__":
    main()
