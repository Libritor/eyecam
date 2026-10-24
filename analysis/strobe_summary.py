"""State of Strobe, the five sittings at a glance: one panel per marker family,
one mark per sitting, from the json files the other State of Flash scripts
write (flash_eog_all.json, state_of_flash.json, ssvep_fatigue.json).

    python analysis/strobe_summary.py  -> runs/state_of_flash/fig_strobe_summary.png + .json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "state_of_flash")
SITTINGS = [("vr_planes8", "10-07\nnight\nreference"), ("vr_planes9", "10-10\n17:57"),
            ("vr_planes10", "10-10\n19:50"), ("vr_vision1", "10-11\n12:05\ngrey"), ("vr_planes11", "10-12\n10:40"),
            ("vr_grey1", "10-17\n00:35\ngrey"), ("vr_die1", "10-17\n10:45\ndie")]
C_SIT = ["#8a8a85", "#1baf7a", "#4a3aa7", "#d9a400", "#c8102e", "#2a4d8f", "#e07b00"]   # the same colours as flash_eog's figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe5"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    eog = json.load(open(os.path.join(OUT, "flash_eog_all.json")))["sessions"]
    sof = {s["session"]: s for s in json.load(open(os.path.join(OUT, "state_of_flash.json")))["sessions"]}
    fat = json.load(open(os.path.join(OUT, "ssvep_fatigue.json")))
    names = [n for n, _ in SITTINGS]
    labels = [l for _, l in SITTINGS]
    x = np.arange(len(names))
    rows = {}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))

    # 1 blink rate on the 12 Hz stages, with its bootstrap interval
    ax = axes[0]
    for i, n in enumerate(names):
        s = eog[n]["lit_dark_12hz_stages"]
        r, ci = s["blink_per_min"], s["blink_ci"]
        ax.errorbar(i, r, yerr=[[r - ci[0]], [ci[1] - r]], fmt="o", color=C_SIT[i], ms=7, capsize=3, lw=1.5)
        ax.text(i, ci[1] + 0.4, f"{r:.1f}", ha="center", fontsize=8, color=INK)
        rows.setdefault(n, {})["blink_per_min"] = r
    ax.set_ylim(0, 16)
    ax.set_ylabel("blinks per minute", fontsize=9, color=MUTED)
    style(ax, "a. blink rate on task")

    # 2 blink release: the first second after a cell goes dark against seconds 1-4
    ax = axes[1]
    for i, n in enumerate(names):
        r = eog[n]["release_after_dark_onset"]
        a, b = r["first_second_per_min"], r["seconds_1_to_4_per_min"]
        ax.plot([i - 0.18, i + 0.18], [a, b], "-", color=C_SIT[i], lw=1.5)
        ax.plot(i - 0.18, a, "o", color=C_SIT[i], ms=7)
        ax.plot(i + 0.18, b, "o", color=C_SIT[i], ms=7, mfc="white", mew=1.5)
        ax.text(i - 0.18, a + 1.2, f"{a:.0f}", ha="center", fontsize=8, color=INK)
        rows[n]["release_first_s"], rows[n]["release_1_4_s"] = a, b
    ax.set_ylim(0, 42)
    ax.set_ylabel("blinks/min: first second (filled), seconds 1-4 (open)", fontsize=8, color=MUTED)
    style(ax, "b. release after a cell goes dark")

    # 3 Oz theta/alpha drift over the sitting, flicker epochs and dark epochs
    ax = axes[2]
    for i, n in enumerate(names):
        s = sof[n]
        # a short sitting can lack one of the two drifts (vr_grey1: 48 dark epochs)
        f = (s.get("drift_flicker") or {}).get("ta_oz", np.nan)
        d = (s.get("drift_dark") or {}).get("ta_oz", np.nan)
        ax.plot(i - 0.18, f, "o", color=C_SIT[i], ms=7)
        ax.plot(i + 0.18, d, "o", color=C_SIT[i], ms=7, mfc="white", mew=1.5)
        rows[n]["drift_ta_oz_flicker"], rows[n]["drift_ta_oz_dark"] = f, d
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-0.6, 0.3)
    ax.set_ylabel("log10 per 10 min: flicker (filled), dark (open)", fontsize=8, color=MUTED)
    style(ax, "c. theta/alpha drift on Oz")

    # 4 SSVEP of the lit cells: first quarter against last quarter of each plane's scan
    ax = axes[3]
    for i, n in enumerate(names):
        if n not in fat:
            ax.text(i, 0.05, "no colour\nplanes" if "vision" in n else "", ha="center", fontsize=7, color=MUTED)
            continue
        visits = fat[n].get("visits", [])
        for k, (plane, col) in enumerate((("red", "#e34948"), ("green", "#008300"), ("blue", "#2a78d6"))):
            lit = [v["score"] for v in sorted(visits, key=lambda v: v["t"]) if v["plane"] == plane and v["lit"]]
            if len(lit) < 8:
                continue
            q = max(len(lit) // 4, 2)
            vals = [float(np.median(lit[:q])), float(np.median(lit[-q:]))]
            xo = i + (k - 1) * 0.22
            ax.plot([xo - 0.07, xo + 0.07], [vals[0], vals[-1]], "-", color=col, lw=1.5)
            ax.plot(xo - 0.07, vals[0], "o", color=col, ms=5)
            ax.plot(xo + 0.07, vals[-1], "o", color=col, ms=5, mfc="white", mew=1.4)
            rows[n][f"ssvep_{plane}_first_last"] = [vals[0], vals[-1]]
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("lit-cell response: first (filled), last quarter (open)", fontsize=8, color=MUTED)
    style(ax, "d. SSVEP, first to last quarter")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout(w_pad=1.5)
    p = os.path.join(OUT, "fig_strobe_summary.png")
    fig.savefig(p, dpi=150)
    fig.savefig(os.path.join(OUT, "fig_strobe_summary.pdf"))      # the paper takes the vector copy
    json.dump(rows, open(os.path.join(OUT, "strobe_summary.json"), "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
