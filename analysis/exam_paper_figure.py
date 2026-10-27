"""Column-width figure of an eye-exam read-out for the paper: (a) acuity and
(b) contrast curves of both eyes with their fitted limbs and thresholds,
(c) the right eye's field map, (d) the colour planes of both eyes. Vector PDF
with the text embedded (pdf.fonttype 42), plus a PNG preview.

    python analysis/exam_paper_figure.py runs/exam_gate2 docs/paper/exam2027/figures/exam_gate.pdf
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5,
                     "font.family": "DejaVu Sans", "axes.titlesize": 6.5,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.5, "axes.linewidth": 0.5})
EYE_STYLE = {"right": dict(color="#1f4e79", marker="o"), "left": dict(color="#c0392b", marker="s"),
             "both": dict(color="#2e7d32", marker="^")}


def curve(ax, ro, test, xkey, xlabel, logbase):
    for eye, r in ro["eyes"].items():
        d = r.get(test, {})
        if not d.get(xkey):
            continue
        st = EYE_STYLE.get(eye, dict(color="#555", marker="o"))
        x = np.array(d[xkey], float); a = np.array(d["amps"], float); p = np.array(d["p"], float)
        sd = np.array(d.get("amp_sd", [0] * len(x)), float)
        sig = p < 0.05
        ax.errorbar(x[sig], a[sig], yerr=sd[sig], fmt=st["marker"], ms=2.6, color=st["color"],
                    ecolor=st["color"], elinewidth=0.5, capsize=1.2, lw=0.6, label=f"{eye} eye")
        ax.errorbar(x[~sig], a[~sig], yerr=sd[~sig], fmt=st["marker"], ms=2.6, mfc="white",
                    color=st["color"], ecolor=st["color"], elinewidth=0.5, capsize=1.2, lw=0.6)
        fit = d.get("fit") or {}
        if fit.get("ok") and d.get("ok"):
            xs = np.array(fit["x_used"])
            lo = min(fit["x_thr_log"], xs.min()); hi = xs.max()
            xx = np.linspace(lo, hi, 40)
            ax.plot(logbase ** xx, fit["a"] + fit["b"] * xx, color=st["color"], lw=0.9)
            ax.axvline(d["threshold"], color=st["color"], lw=0.6, ls="--")
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("response amplitude (a.u.)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.2)


def field(ax, fig, ro, eye):
    d = ro["eyes"][eye]["field"]
    rows, cols = d["rows"], d["cols"]
    rel = np.array(d["rel"], float).reshape(rows, cols)
    p = np.array(d["p"], float)
    im = ax.imshow(np.nan_to_num(rel), cmap="magma", vmin=0, vmax=1)
    for k in range(d["n"]):
        y, x = divmod(k, cols)
        if p[k] < 0.05 and np.isfinite(rel.flat[k]):
            ax.text(x, y, f"{rel.flat[k]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if rel.flat[k] < 0.55 else "black")
        else:
            ax.text(x, y, "×", ha="center", va="center", fontsize=8, color="#e74c3c")
    ax.plot(cols / 2 - 0.5, rows / 2 - 0.5, "o", ms=3, mfc="white", mec="black", mew=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5)


def colour(ax, ro):
    from matplotlib.patches import Patch
    eyes = [e for e in ro["eyes"] if ro["eyes"][e].get("colour", {}).get("ok")]
    w = 0.8 / max(len(eyes), 1)
    handles = []
    for i, eye in enumerate(eyes):
        rel = np.nan_to_num(np.array(ro["eyes"][eye]["colour"]["relative"], float))
        p = np.array(ro["eyes"][eye]["colour"]["p"], float)
        xs = np.arange(3) + (i - (len(eyes) - 1) / 2) * w
        edge = EYE_STYLE.get(eye, {}).get("color", "#555")
        bars = ax.bar(xs, rel, width=w * 0.95, color=["#d62728", "#2ca02c", "#1f77b4"],
                      edgecolor=edge, linewidth=0.8)
        handles.append(Patch(facecolor="white", edgecolor=edge, linewidth=0.8, label=f"{eye} eye"))
        for b, pv in zip(bars, p):
            if pv >= 0.05:
                b.set_hatch("////"); b.set_alpha(0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["red", "green", "blue"])
    ax.set_ylim(0, 1.3); ax.set_ylabel("rel. amplitude")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.0, ncol=2,
              columnspacing=0.8)


def main():
    session, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(session, "exam_readout.json")) as f:
        ro = json.load(f)
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.11, hspace=0.62, wspace=0.62)
    curve(axs[0, 0], ro, "acuity", "sizes", "bar width (arcmin)", math.e)
    curve(axs[0, 1], ro, "contrast", "contrasts", "contrast", 10)
    field(axs[1, 0], fig, ro, "right")
    colour(axs[1, 1], ro)
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
