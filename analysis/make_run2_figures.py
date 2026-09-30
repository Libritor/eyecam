"""Figures for VR run 2 (2026-09-29): runs/vr_full2 (calib + grey) and runs/vr_full2b (colour + music)."""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
OUT = "runs/figures"; os.makedirs(OUT, exist_ok=True)
def norm(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98); return np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
A, B = "runs/vr_full2", "runs/vr_full2b"
dc = json.load(open(f"{A}/decoders_compare.json"))["results"]
tg = np.load(f"{A}/target.npy"); tc = np.load(f"{B}/target_color.npy"); gc = np.load(f"{B}/reconstruction_color_grid.npy")
fig, ax = plt.subplots(2, 4, figsize=(13, 6.2))
ax[0, 0].imshow(tg, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax[0, 0].set_title("target 'NO'\n8 x 6 cells, 6 s each, 12 Hz", fontsize=9.5)
for k, (m, lab) in enumerate((("line", "line detector"), ("welch", "Welch relative power (paper)"), ("cca", "CCA"))):
    im = np.asarray(Image.open(f"{A}/reconstruction_{m}.png").convert("L"), float) / 255
    ax[0, k + 1].imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[0, k + 1].set_title(f"{lab}\nr = {dc[m]['r']:.2f}   (shifted-EEG null max {max(dc[m]['null']):.2f})", fontsize=9.5)
ax[1, 0].imshow(tc, interpolation="nearest"); ax[1, 0].set_title("colour target\nR 7.2 Hz, G 9 Hz, B 12 Hz", fontsize=9.5)
rgb = np.stack([norm(gc[:, :, k]) for k in range(3)], 2)
ax[1, 1].imshow(rgb, interpolation="nearest"); ax[1, 1].set_title("colour reconstruction (6 x 4, 6 s/cell)\nr: R 0.75  G 0.86  B 0.90   hue accuracy 24/24", fontsize=9.5)
for k, (lab, cm) in enumerate((("G plane (9 Hz line)", "Greens"), ("B plane (12 Hz line)", "Blues"))):
    ax[1, k + 2].imshow(norm(gc[:, :, k + 1]), cmap=cm, interpolation="nearest"); ax[1, k + 2].set_title(lab, fontsize=9.5)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle("VR run 2, 2026-09-29 (Quest 3S browser 72 fps, Muse 2 + Oz aux): image stages", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_run2_images.png", dpi=150); plt.close(fig)
# calibration + colour calibration blocks
cal = json.load(open(f"{A}/calibration.json")); pb = cal["line"]["per_block"]; cc = json.load(open(f"{B}/ccal.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
xs = np.arange(len(pb)); ax[0].bar(xs, [b["snr"]["AUX"] for b in pb], color=[BLUE if b["kind"] == "on" else ORANGE for b in pb], width=0.7)
ax[0].axhline(1, color=MUTED, lw=1, ls="--"); ax[0].set_xticks(xs); ax[0].set_xticklabels([("ON" if b["kind"] == "on" else "off") for b in pb], fontsize=8)
ax[0].set_ylabel("Oz line ratio at 12 Hz (peak / flanks)")
ax[0].set_title(f"12 Hz calibration in VR: Oz per block  (gate PASSED, p = {cal['line']['p']:.4f})", fontsize=10)
ax[0].text(0.02, 0.95, "ON = full-panel white/black flicker", color=BLUE, transform=ax[0].transAxes, va="top", fontsize=9)
ax[0].text(0.02, 0.88, "off = black", color=ORANGE, transform=ax[0].transAxes, va="top", fontsize=9)
pos = 0; ticks, tl = [], []
for f, lab in zip([7.2, 9.0, 12.0], ["red 7.2 Hz", "green 9 Hz", "blue 12 Hz"]):
    bl = [b for b in cc["per_block"] if abs(b["hz"] - f) < 0.01]
    for b in bl: ax[1].bar(pos, b["snr"]["AUX"], color=BLUE if b["kind"] == "on" else ORANGE, width=0.8); pos += 1
    ticks.append(pos - len(bl) / 2 - 0.5); tl.append(lab); pos += 1.2
ax[1].axhline(1, color=MUTED, lw=1, ls="--"); ax[1].set_xticks(ticks); ax[1].set_xticklabels(tl); ax[1].set_ylabel("Oz line ratio at the block's frequency")
ps = {f"{r['freq']:g}": r["p"] for r in cc["results"].values()}
ax[1].set_title("colour calibration, full-panel single-colour flicker, Oz per block\n"
                f"p: red {ps.get('7.2', float('nan')):.3f}, green {ps.get('9', float('nan')):.3f}, blue {ps.get('12', float('nan')):.2f}", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_run2_calibration_blocks.png", dpi=150); plt.close(fig)
print("ok")
