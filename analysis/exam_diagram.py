"""Diagram for the eye-exam paper: the rig as a pipeline (top) and the four
stimuli with the block timeline (bottom), drawn as the page draws them. Vector
PDF with the text embedded, plus a PNG preview.

    python analysis/exam_diagram.py docs/paper/exam2027/figures/exam_diagram.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6,
                     "font.family": "DejaVu Sans"})
GAMMA = 2.2


def byte(L):
    return round(255 * max(0.0, min(1.0, L)) ** (1 / GAMMA)) / 255


def box(ax, x, y, w, h, text, fc="#f4f6f8", ec="#1f4e79", fs=5.6, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.7))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 0.04, lines[0], ha="center", va="top", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h - 0.04 - 0.09, "\n".join(lines[1:]), ha="center", va="top",
                fontsize=fs - 0.4, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text="", both=False, dy=0.03, fs=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=6,
                                 lw=0.7, color="#333", shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color="#333")


def pipeline(ax):
    """Four boxes across the column (each about 0.8 in wide): every line is
    kept under 15 characters at 5 pt so nothing runs past its box."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, h, y = 0.225, 0.5, 0.44
    xs = [0.005, 0.26, 0.515, 0.77]
    box(ax, xs[0], y, w, h, "Muse headband\nTP9 AF7 AF8 TP10\n+ Oz electrode\n256 Hz Bluetooth", fs=5.2)
    box(ax, xs[1], y, w, h, "phone\nMuseLog app\nraw EEG as OSC\nWi-Fi, UDP 5000", fs=5.2)
    box(ax, xs[2], y, w, h, "PC driver\nxr_session.py\n--mode exam\nlogs frames,\nscores tests", fs=5.2)
    box(ax, xs[3], y, w, h, "Quest browser\nstimulus page\nframe-exact\n12 Hz at 72 fps", fs=5.2)
    ym = y + h * 0.55
    arrow(ax, (xs[0] + w, ym), (xs[1], ym))
    arrow(ax, (xs[1] + w, ym), (xs[2], ym))
    arrow(ax, (xs[2] + w, ym), (xs[3], ym), both=True)
    ax.text((xs[2] + w + xs[3]) / 2, y - 0.02, "USB", ha="center", va="top", fontsize=4.6, color="#333")
    box(ax, xs[2], 0.03, w, 0.28, "read-out\nthresholds,\nfield map, PDF", fc="#eef5ee", ec="#2e7d32", fs=5.2)
    arrow(ax, (xs[2] + w / 2, y), (xs[2] + w / 2, 0.31))
    box(ax, xs[3], 0.03, w, 0.28, "subject\none eye covered,\neyes on the dot", fc="#fff8e6", ec="#b26a00", fs=5.2)
    arrow(ax, (xs[3] + w / 2, y), (xs[3] + w / 2, 0.31), both=True)
    ax.text(0.005, 0.17, "phantom: the same path with a\nsynthetic EEG that follows the\nlogged stimulus (the gate)",
            ha="left", va="center", fontsize=4.8, color="#555", style="italic")


def stimulus_panel(ax, kind):
    """What the headset shows: every stimulus fills the window, the fixation
    dot at its centre."""
    W, H = 1.3, 1.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="black", ec="#666", lw=0.4))
    if kind == "flash":
        ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=2))
    elif kind in ("acuity", "contrast"):
        c = 1.0 if kind == "acuity" else 0.25
        bars = 12 if kind == "acuity" else 7
        hi, lo = byte(0.5 * (1 + c)), byte(0.5 * (1 - c))
        img = np.zeros((2, 130, 3))
        stripe = 130 // (2 * bars)
        for i in range(130):
            v = hi if (i // stripe) % 2 == 0 else lo
            img[:, i, :] = v
        ax.imshow(img, extent=(0, W, 0, H), interpolation="nearest", zorder=2)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
    elif kind == "field":
        ax.plot([W / 2] * 2, [0, H], color="#333", lw=0.3, zorder=3)
        ax.plot([0, W], [H / 2] * 2, color="#333", lw=0.3, zorder=3)
        ax.add_patch(Rectangle((0, H / 2), W / 2, H / 2, fc="white", ec="none", zorder=4))
    elif kind == "colour":
        ax.add_patch(Rectangle((0, 0), W, H, fc="#00c000", ec="none", zorder=2))
    elif kind == "grey":
        g = byte(0.5)
        ax.add_patch(Rectangle((0, 0), W, H, fc=(g, g, g), ec="none", zorder=2))
    cx, cy = W / 2, H / 2
    ax.add_patch(Circle((cx, cy), 0.05, fc="black", ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy), 0.03, fc="white", ec="none", zorder=6))
    ax.add_patch(Rectangle((0, 0), W, H, fc="none", ec="#666", lw=0.5, zorder=7))   # the window's edge


def timeline(ax):
    ax.set_xlim(0, 16.6); ax.set_ylim(-0.4, 1.9); ax.axis("off")
    t = 0.0
    for k, lab in enumerate(["step k, pair 1", "step k, pair 2"]):
        for on in (True, False):
            ax.add_patch(Rectangle((t, 0), 4, 1, fc="#dbe7f3" if on else "#eeeeee", ec="#1f4e79" if on else "#999", lw=0.6))
            if on:
                tt = np.linspace(t + 0.1, t + 3.9, 300)
                sq = 0.5 + 0.35 * np.sign(np.sin(2 * np.pi * 3 * (tt - t)))
                ax.plot(tt, sq, color="#1f4e79", lw=0.5)
            ax.text(t + 2, 1.12, "ON" if on else "OFF", ha="center", va="bottom", fontsize=5)
            t += 4
        ax.text(8 * k + 4, -0.3, lab, ha="center", va="top", fontsize=5, color="#333")
    ax.text(16.3, 0.5, "...", ha="left", va="center", fontsize=7)
    ax.text(1.1, 1.5, "4 s ON (the window flickering, or the bars swapping, at 12 Hz) / 4 s OFF (grey or black); five pairs\n"
            "of flash, two per step of the rest; each step's ON blocks scored against all the test's OFF blocks",
            ha="left", va="bottom", fontsize=4.6, color="#333", linespacing=1.3)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/paper/exam2027/figures/exam_diagram.pdf"
    fig = plt.figure(figsize=(3.5, 3.1))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 0.95, 0.5], left=0.01, right=0.99, top=0.985,
                          bottom=0.02, hspace=0.35, wspace=0.08)
    ax = fig.add_subplot(gs[0, :]); pipeline(ax)
    ax.text(0.0, 1.0, "(a)", fontsize=7, weight="bold", va="top", ha="left")
    kinds = [("flash", "flash: the window\nwhite/black, the\neye's reference"),
             ("acuity", "acuity: bars swap\nplaces (reversal),\n40 to 3.5'; OFF grey"),
             ("contrast", "contrast: one\nbar width,\ncontrast 1 to 0.03"),
             ("field", "field: one\nquadrant,\nwhite on black"),
             ("colour", "colour: the window\nred, green or\nblue on black")]
    for j, (kind, lab) in enumerate(kinds):
        ax = fig.add_subplot(gs[1, j]); stimulus_panel(ax, kind)
        ax.text(0.5, -0.08, lab, transform=ax.transAxes, ha="center", va="top", fontsize=4.8, linespacing=1.2)
        if j == 0:
            ax.text(-0.02, 1.08, "(b)", transform=ax.transAxes, fontsize=7, weight="bold", va="bottom", ha="left")
    ax = fig.add_subplot(gs[2, :]); timeline(ax)
    ax.text(0.0, 1.0, "(c)", transform=ax.transAxes, fontsize=7, weight="bold", va="bottom", ha="left")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    main()
