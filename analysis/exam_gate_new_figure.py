"""Column-width figure of the gate's read-out of the four tests added on
29 October 2026: (a) vernier, amplitude against the strips' offset with the
fitted limbs, (b) the contrast sensitivity function from the three bar widths,
(c) the meridians of both eyes, each orientation over the best, (d) flicker
photometry, amplitude against red's share with the fitted null model. Right
eye circles, left eye squares; open markers are not significant. Vector PDF
with the text embedded.

    python analysis/exam_gate_new_figure.py runs/exam_gate docs/paper/exam2027/figures/exam_gate_new.pdf
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exam_paper_figure import EYE_STYLE, curve  # noqa: E402
from exam_readout import photometry_modulation  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5,
                     "font.family": "DejaVu Sans", "axes.titlesize": 6.5,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.5, "axes.linewidth": 0.5})


def csf(ax, ro):
    for eye, r in ro["eyes"].items():
        d = r.get("csf", {})
        pts = d.get("points") or []
        if not pts:
            continue
        st = EYE_STYLE.get(eye, dict(color="#555", marker="o"))
        f = np.array([p["cpd"] for p in pts]); y = np.array([p["log_cs"] for p in pts])
        ax.plot(f, y, "-", color=st["color"], lw=0.7)
        for p in pts:
            ax.plot(p["cpd"], p["log_cs"], st["marker"], ms=2.8, color=st["color"],
                    mfc="white" if p["beyond_range"] else st["color"])
        ax.plot([], [], st["marker"], color=st["color"], ms=2.8, label=f"{eye} eye")
    ax.set_xscale("log")
    ticks = sorted({round(p["cpd"], 2) for r in ro["eyes"].values() for p in (r.get("csf", {}).get("points") or [])})
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:.2g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("cycles/deg"); ax.set_ylabel("log contrast sensitivity")
    ax.set_ylim(0, 2.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.2)


def meridian(ax, ro):
    for th in (0.5, 1.0):
        ax.add_patch(plt.Circle((0, 0), th, fill=False, color="#ddd", lw=0.5))
    for eye, r in ro["eyes"].items():
        d = r.get("meridian", {})
        if not d.get("ok"):
            continue
        st = EYE_STYLE.get(eye, dict(color="#555", marker="o"))
        ang = np.array(d["angles"], float); p = np.array(d["p"], float)
        rel = np.array([np.nan if v is None else v for v in d["rel"]], float)
        off = 0.0 if eye == "right" else 0.06          # the two eyes' lines side by side
        for a_, r_, p_ in zip(ang, rel, p):
            dx, dy = math.sin(math.radians(a_)), math.cos(math.radians(a_))
            r_ = float(np.clip(np.nan_to_num(r_), 0, 1.1))
            ox, oy = -dy * off, dx * off
            ax.plot([-dx * r_ + ox, dx * r_ + ox], [-dy * r_ + oy, dy * r_ + oy], color=st["color"],
                    lw=1.8 if p_ < 0.05 else 0.8, ls="-" if p_ < 0.05 else "--", solid_capstyle="round")
        ax.plot([], [], color=st["color"], lw=1.8, label=f"{eye} eye")
        for a_, r_ in zip(ang, rel):
            if eye == "right":
                dx, dy = math.sin(math.radians(a_)), math.cos(math.radians(a_))
                ax.text(dx * 1.32, dy * 1.32, f"{a_:g}°", ha="center", va="center", fontsize=5.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("amplitude / best meridian")      # the eyes' colours as in the other panels


def photometry(ax, ro):
    for eye, r in ro["eyes"].items():
        d = (r.get("photometry", {}).get("pairs") or {}).get("rg")
        if not d:
            continue
        st = EYE_STYLE.get(eye, dict(color="#555", marker="o"))
        x = np.array(d["mixes"], float); a = np.array(d["amps"], float); p = np.array(d["p"], float)
        sig = p < 0.05
        ax.plot(x[sig], a[sig], st["marker"], ms=2.8, color=st["color"], label=f"{eye} eye")
        ax.plot(x[~sig], a[~sig], st["marker"], ms=2.8, mfc="white", color=st["color"])
        if d.get("fit_ok"):
            xx = np.linspace(0.05, 0.95, 100)
            ax.plot(xx, [d["a"] * photometry_modulation(v, d["null"]) + d["b"] for v in xx],
                    color=st["color"], lw=0.8)
            ax.axvline(d["null"], color=st["color"], lw=0.6, ls="--")
    ax.axvline(0.5, color="#999", lw=0.4, ls=":")
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("red's share of luminance"); ax.set_ylabel("amplitude (a.u.)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", handlelength=1.2)


def main():
    session, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(session, "exam_readout.json")) as f:
        ro = json.load(f)
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.7))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.12, hspace=0.7, wspace=0.6)
    curve(axs[0, 0], ro, "vernier", "offsets", "vernier offset (arcmin)", math.e)
    axs[0, 0].set_xticks([8, 4, 2, 1, 0.5]); axs[0, 0].set_xticklabels(["8", "4", "2", "1", "0.5"])
    axs[0, 0].minorticks_off()
    csf(axs[0, 1], ro)
    meridian(axs[1, 0], ro)
    photometry(axs[1, 1], ro)
    for ax, lab in zip(axs.flat, "abcd"):
        ax.text(-0.22 if lab != "c" else -0.08, 1.06, f"({lab})", transform=ax.transAxes,
                fontsize=7, weight="bold", va="bottom")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
