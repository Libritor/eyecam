"""The die figure for the ICCE paper: the cube as shown and as decoded from the
EEG, three views each plus the flat atlas, cut from mesh_comparison.png
(mesh_planes.py paint) and laid out with real text, so the PDF's labels are
selectable while the renders stay the rasters they are.

    python analysis/die_paper_figure.py runs/vr_die1/mesh_comparison.png docs/paper/icce2027/figures/die.pdf
"""
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "font.family": "DejaVu Sans"})

# panel boxes in the 1112 x 938 composite: three views and the atlas, for the shown and decoded rows
VIEWS = [(18, 268), (286, 536), (554, 804)]
ATLAS = (822, 1092)
ROWS = {"shown": (70, 320), "decoded from the EEG": (354, 604)}


def main():
    src, out = sys.argv[1], sys.argv[2]
    im = Image.open(src).convert("RGB")
    W, H = im.size
    sx, sy = W / 1112, H / 938
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05], wspace=0.04, hspace=0.12))
    for r, (label, (y0, y1)) in enumerate(ROWS.items()):
        boxes = VIEWS + [ATLAS]
        for c, (x0, x1) in enumerate(boxes):
            crop = im.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
            ax = axs[r, c]
            ax.imshow(np.asarray(crop))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=4)
            if r == 0:
                ax.set_title(["view 1", "view 2", "view 3", "the 9 by 6 atlas scanned"][c], fontsize=7.5, color="#52514e")
    fig.text(0.5, 0.01, "red ring: a cell decoded wrong (51 of 54 right)", ha="center", fontsize=7, color="#52514e")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.07)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
