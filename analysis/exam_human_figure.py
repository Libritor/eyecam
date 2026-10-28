"""Column-width figure of a human eye-exam sitting for the paper, from the
session's exam_readout.json: (a) the full-window flash per channel and eye,
(b) the flicker sweep, (c) acuity, amplitude against bar width with the fitted
limb where one fits, (d) the quadrants per eye. Right eye blue, left eye
orange; open markers and hatched bars are not significant (p >= 0.05).

    python analysis/exam_human_figure.py runs/vr_exam6 docs/paper/exam2027/figures/exam_human.pdf
"""
import json
import math
import os
import sys

import numpy as np

RIGHT, LEFT, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
CH = ["TP9", "AF7", "AF8", "TP10", "AUX"]
ALPHA = 0.05


def main(session, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5,
                         "font.family": "DejaVu Sans", "axes.titlesize": 7, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID})
    with open(os.path.join(session, "exam_readout.json")) as f:
        ro = json.load(f)
    eyes = {"right": ro["eyes"]["right"], "left": ro["eyes"]["left"]}
    col = {"right": RIGHT, "left": LEFT}
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
    # (a) flash per channel
    ax = axes[0, 0]
    x = np.arange(len(CH))
    for k, (eye, r) in enumerate(eyes.items()):
        fl = r["flash"]
        amps = [fl["amps_ch"].get(c, 0.0) for c in CH]
        ps = [fl["channel_p"].get(c, 1.0) for c in CH]
        bars = ax.bar(x + (k - 0.5) * 0.38, amps, 0.36, color=col[eye], label=f"{eye} eye")
        for b, p in zip(bars, ps):
            if p >= ALPHA:
                b.set_hatch("////"); b.set_alpha(0.45); b.set_edgecolor(col[eye])
    ax.set_xticks(x)
    ax.set_xticklabels(["TP9", "AF7", "AF8", "TP10", "Oz"])
    ax.set_ylabel("flash amplitude (a.u.)")
    ax.set_title("(a) full-window flash, 12 Hz", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RIGHT, label="right eye"), Patch(color=LEFT, label="left eye")],
              frameon=False, fontsize=6, loc="upper left")
    # (b) sweep
    ax = axes[0, 1]
    for eye, r in eyes.items():
        te = r["temporal"]
        rates, amps, ps = np.array(te["rates"]), np.array(te["amps_corr"]), np.array(te["p"])
        ax.plot(rates, amps, color=col[eye], lw=1.0, zorder=1)
        sig = ps < ALPHA
        ax.plot(rates[sig], amps[sig], "o", color=col[eye], ms=3.5)
        ax.plot(rates[~sig], amps[~sig], "o", mfc="white", color=col[eye], ms=3.5)
    ax.set_xscale("log")
    ax.set_xticks([6, 7.2, 9, 12, 18, 36])
    ax.set_xticklabels(["6", "7.2", "9", "12", "18", "36"])
    ax.minorticks_off()
    ax.set_xlabel("flicker rate (Hz)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.set_title("(b) flicker sweep, Oz", loc="left")
    # (c) acuity
    ax = axes[1, 0]
    for eye, r in eyes.items():
        ac = r["acuity"]
        sizes, amps, ps = np.array(ac["sizes"]), np.array(ac["amps"]), np.array(ac["p"])
        sig = ps < ALPHA
        ax.plot(sizes[sig], amps[sig], "o", color=col[eye], ms=3.5)
        ax.plot(sizes[~sig], amps[~sig], "o", mfc="white", color=col[eye], ms=3.5)
        fit = ac.get("fit") or {}
        if ac.get("ok") and fit.get("ok") and fit.get("x_used"):
            xs = np.exp(np.array(fit["x_used"]))
            xl = np.linspace(math.log(ac["threshold"]), max(fit["x_used"]), 20)
            ax.plot(np.exp(xl), fit["a"] + fit["b"] * xl, color=col[eye], lw=1.2)
            ax.axvline(ac["threshold"], color=col[eye], lw=0.8, ls=(0, (3, 2)))
            ax.text(ac["threshold"] * 1.08, 0.3, f"{ac['threshold']:.1f}'", color=col[eye], fontsize=6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([40, 20, 10, 5, 3])
    ax.set_xticklabels(["40", "20", "10", "5", "3"])
    ax.minorticks_off()
    ax.set_xlabel("bar width (arcmin, placeholder scale)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.set_title("(c) reversing bars, coarse to fine", loc="left")
    # (d) quadrants per eye, two small maps
    ax = axes[1, 1]
    ax.set_xlim(0, 4.6); ax.set_ylim(0, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    cmap = matplotlib.colormaps["magma"]
    for k, (eye, r) in enumerate(eyes.items()):
        fi = r["field"]
        x0 = 0.2 + k * 2.3
        for s in range(fi["n"]):
            cx, cy = s % fi["cols"], s // fi["cols"]
            v = fi["rel"][s] if fi["rel"][s] is not None else float("nan")
            resp = s in fi["responding"]
            face = cmap(0.15 + 0.8 * min(max(v, 0), 1)) if resp and np.isfinite(v) else "#f2f2f2"
            ax.add_patch(plt.Rectangle((x0 + cx, 1.9 - cy - 0.95), 0.95, 0.95, facecolor=face,
                                       edgecolor="#333", lw=0.5))
            ax.text(x0 + cx + 0.475, 1.9 - cy - 0.475, f"{v:.2f}" if resp else "x",
                    ha="center", va="center", fontsize=6,
                    color="white" if resp and v > 0.5 else INK)
        ax.text(x0 + 1.0, 2.15, f"{eye} eye", ha="center", fontsize=6.5, color=col[eye])
    ax.set_title("(d) quadrants, over the best", loc="left")
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
