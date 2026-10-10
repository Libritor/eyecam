"""Figures for a `--mode planes` colour session: the picture shown, the
picture decoded from EEG, and the decoded picture snapped to the eight
colours (wrong cells marked), plus one row per colour plane (shown, from
EEG with r, thresholded with the count right). Scored exactly as the driver
scores it (smoothcolor.metrics with the response grid: each plane's own
log-domain Otsu threshold).

    python analysis/planes_figures.py runs/vr_planes9 --label planes9 \
        --title "vr_planes9 (2026-10-10): blue at 9 Hz, patch 2, 10 revisits"

Writes <session>/<label>_colour_picture.png and <label>_planes_figure.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smoothcolor  # noqa: E402

NAMES = ["red", "green", "blue"]
CMAPS = ["Reds", "Greens", "Blues"]


def frame(ax, h, w):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def marks(ax, wrong, color="white"):
    ys, xs = np.where(wrong)
    ax.scatter(xs, ys, marker="x", s=90, c=color, linewidths=2.2, zorder=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session")
    ap.add_argument("--label", default=None, help="file prefix (default: session folder name)")
    ap.add_argument("--title", default="", help="headline over the colour picture")
    a = ap.parse_args()
    label = a.label or os.path.basename(os.path.normpath(a.session))
    target = np.load(os.path.join(a.session, "target_color.npy"))
    grid = np.load(os.path.join(a.session, "reconstruction_planes_grid.npy"))
    meta = {}
    mp = os.path.join(a.session, "planes_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    h, w, _ = target.shape
    rgb = smoothcolor.to_rgb(grid, floor=0.0)
    res = smoothcolor.metrics(rgb, target, grid)
    print(smoothcolor.summary(res))
    on = np.stack([grid[:, :, k] > smoothcolor.otsu_log(grid[:, :, k]) for k in range(3)], axis=2)
    pure = ((target <= 0.25) | (target >= 0.75)).all(axis=2)
    snapped = on.astype(float)
    wrong = pure & ~(on == (target > 0.5)).all(axis=2)
    # the decoded picture, each plane scaled to its own range, for display
    shown = np.zeros_like(rgb)
    for k in range(3):
        g = grid[:, :, k]
        shown[:, :, k] = (g - np.nanmin(g)) / max(np.nanmax(g) - np.nanmin(g), 1e-9)

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4))
    axs[0].imshow(target, interpolation="nearest")
    axs[0].set_title("shown: the eight colours")
    axs[1].imshow(np.clip(shown, 0, 1), interpolation="nearest")
    axs[1].set_title("decoded from EEG (each plane scaled to its range)")
    axs[2].imshow(snapped, interpolation="nearest")
    axs[2].set_title(f"snapped to the eight colours: {res['pure_right']} of "
                     f"{res['pure_cells']} cells right")
    marks(axs[2], wrong)
    for ax in axs:
        frame(ax, h, w)
    if a.title:
        fig.suptitle(a.title + "  (x = wrong cell)", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(a.session, f"{label}_colour_picture.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(10.8, 8))
    fd = meta.get("freq_delivered", {})
    for k in range(3):
        t = target[:, :, k] > 0.5
        g = grid[:, :, k]
        axs[k, 0].imshow(t, cmap=CMAPS[k], vmin=0, vmax=1, interpolation="nearest")
        axs[k, 0].set_title(f"shown: {NAMES[k]} plane")
        axs[k, 1].imshow(g, cmap=CMAPS[k], vmin=0, interpolation="nearest")
        hz = fd.get(NAMES[k])
        axs[k, 1].set_title(f"from EEG  (r = {res['r_planes'][k]:.2f}"
                            + (f", {hz:g} Hz)" if hz else ")"))
        axs[k, 2].imshow(on[:, :, k], cmap=CMAPS[k], vmin=0, vmax=1, interpolation="nearest")
        axs[k, 2].set_title(f"threshold: {res['plane_right'][k]} of {res['pure_cells']} right")
        marks(axs[k, 2], pure & (on[:, :, k] != t), color="black")
        for ax in axs[k]:
            frame(ax, h, w)
    if a.title:
        fig.suptitle(a.title + "\n(x = wrong after the threshold)", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(a.session, f"{label}_planes_figure.png")
    fig.savefig(p2, dpi=100)
    plt.close(fig)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
