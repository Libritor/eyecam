"""Result figures for the 2026-09-13/14 sessions (VR full run, sweep, alpha check).

usage: python analysis/make_result_figures.py  -> runs/figures/*.png
"""
import io, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import periodogram, welch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reconstruct  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = "runs/figures"
os.makedirs(OUT, exist_ok=True)


def load_eeg(sess):
    hdr = io.open(f"{sess}/eeg.csv", encoding="utf-8", errors="replace").readline().strip().split(",")
    E = np.genfromtxt(f"{sess}/eeg.csv", delimiter=",", skip_header=1)
    return hdr[1:], E[:, 0], E[:, 1:], reconstruct.infer_fs(E[:, 0])


def norm(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)


# ---------------- figure 1: images ----------------
sess = "runs/vr_full1"
dc = json.load(open(f"{sess}/decoders_compare.json"))["results"]
tg = np.load(f"{sess}/target.npy")
tc = np.load(f"{sess}/target_color.npy")
gc = np.load(f"{sess}/reconstruction_color_grid.npy")
fig, ax = plt.subplots(2, 4, figsize=(13, 6.2))
ax[0, 0].imshow(tg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0, 0].set_title("target 'NO'\n12 x 8 cells, 3.5 s each, 12 Hz", fontsize=9.5)
for k, (m, lab) in enumerate((("line", "line detector"), ("welch", "Welch relative power (paper)"), ("cca", "CCA"))):
    p = f"{sess}/reconstruction_{m}.png"
    im = np.asarray(Image.open(p).convert("L"), float) / 255
    ax[0, k + 1].imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    r = dc[m]["r"]; nmax = max(dc[m]["null"])
    ax[0, k + 1].set_title(f"{lab}\nr = {r:.2f}   (shifted-EEG null max {nmax:.2f})", fontsize=9.5)
ax[1, 0].imshow(tc, interpolation="nearest")
ax[1, 0].set_title("colour target\nR 7.2 Hz, G 9 Hz, B 12 Hz", fontsize=9.5)
rgb = np.stack([norm(gc[:, :, k]) for k in range(3)], 2)
ax[1, 1].imshow(rgb, interpolation="nearest")
ax[1, 1].set_title("colour reconstruction\nr: R 0.35  G 0.29  B 0.72 (null max 0.52)", fontsize=9.5)
for k, (lab, cm) in enumerate((("R plane (7.2 Hz line)", "Reds"), ("G plane (9 Hz line)", "Greens"))):
    ax[1, k + 2].imshow(norm(gc[:, :, k]), cmap=cm, interpolation="nearest")
    ax[1, k + 2].set_title(lab, fontsize=9.5)
# blue plane replaces the empty slot by drawing it over the G-plane axis? no: add as inset-free 5th? keep 4 columns: show B plane instead of G if space
ax[1, 3].imshow(norm(gc[:, :, 2]), cmap="Blues", interpolation="nearest")
ax[1, 3].set_title("B plane (12 Hz line), r = 0.72 vs null max 0.52", fontsize=9.5)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle("VR run 2026-09-14 (Quest 3S browser, Muse 2 + Oz aux): image stages", color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_vr_images.png", dpi=150)
plt.close(fig)

# ---------------- figure 2: calibration blocks ----------------
cal = json.load(open(f"{sess}/calibration.json"))
pb = cal["line"]["per_block"]
cc = json.load(open(f"{sess}/ccal.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
xs = np.arange(len(pb))
vals = [b["snr"]["AUX"] for b in pb]
cols = [BLUE if b["kind"] == "on" else ORANGE for b in pb]
ax[0].bar(xs, vals, color=cols, width=0.7)
ax[0].axhline(1, color=MUTED, lw=1, ls="--")
ax[0].set_xticks(xs); ax[0].set_xticklabels([("ON" if b["kind"] == "on" else "off") for b in pb], fontsize=8)
ax[0].set_ylabel("Oz line ratio at 12 Hz (peak / flanks)")
ax[0].set_title(f"12 Hz calibration in VR: Oz per block  (p = {cal['line']['p']:.3f})", fontsize=10)
ax[0].text(0.02, 0.95, "ON = full-panel white/black flicker", color=BLUE, transform=ax[0].transAxes, va="top", fontsize=9)
ax[0].text(0.02, 0.88, "off = black", color=ORANGE, transform=ax[0].transAxes, va="top", fontsize=9)
# colour calibration grouped by frequency
freqs = [7.2, 9.0, 12.0]
labels = ["red 7.2 Hz", "green 9 Hz", "blue 12 Hz"]
pos = 0; ticks, tlabels = [], []
for f, lab in zip(freqs, labels):
    blocks = [b for b in cc["per_block"] if abs(b["hz"] - f) < 0.01]
    for b in blocks:
        ax[1].bar(pos, b["snr"]["AUX"], color=BLUE if b["kind"] == "on" else ORANGE, width=0.8)
        pos += 1
    ticks.append(pos - len(blocks) / 2 - 0.5); tlabels.append(lab)
    pos += 1.2
ax[1].axhline(1, color=MUTED, lw=1, ls="--")
ax[1].set_xticks(ticks); ax[1].set_xticklabels(tlabels)
ax[1].set_ylabel("Oz line ratio at the block's frequency")
ps = {f"{r['freq']:g}": r["p"] for r in cc["results"].values()}
ax[1].set_title("colour calibration, full-panel single-colour flicker, Oz per block\n"
                f"p: red {ps.get('7.2', float('nan')):.2f}, green {ps.get('9', float('nan')):.3f}, blue {ps.get('12', float('nan')):.3f}", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_vr_calibration_blocks.png", dpi=150)
plt.close(fig)

# ---------------- figure 3: response curve (laptop sweep) ----------------
sw = json.load(open("runs/sweep2/sweep.json"))
names, et, X, fs = load_eeg("runs/sweep2")
ia = names.index("AUX")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
res = sorted(sw["results"].values(), key=lambda r: r["freq"])
fx = [r["delivered"] for r in res]
on = [r["channels"]["AUX"]["on_med"] for r in res]
off = [r["channels"]["AUX"]["off_med"] for r in res]
w = 0.32
ax[0].bar(np.arange(len(fx)) - w / 2, on, width=w, color=BLUE, label="flicker ON")
ax[0].bar(np.arange(len(fx)) + w / 2, off, width=w, color=ORANGE, label="OFF (black)")
ax[0].set_xticks(np.arange(len(fx))); ax[0].set_xticklabels([f"{v:.1f} Hz\np={r['p']:.3f}" for v, r in zip(fx, res)], fontsize=9)
ax[0].axhline(1, color=MUTED, lw=1, ls="--")
ax[0].set_ylabel("Oz line ratio (median over blocks)")
ax[0].set_title("laptop sweep, full-field white/black:\nthe response lives at 7.5-12 Hz, not 15-20", fontsize=10)
ax[0].legend(frameon=False)
# spectra at 10.4 Hz
f_on = [r for r in res if abs(r["freq"] - 10.4) < 0.05][0]["delivered"]
n = int(6.8 * fs); P_on, P_off = [], []
for kind, t0, t1, hz in sw["blocks"]:
    seg = X[(et >= t0 + 1) & (et <= t1), ia]
    if len(seg) < n: continue
    seg = seg[:n] - seg[:n].mean(); f, p = periodogram(seg, fs=fs, window="hann")
    if kind == "on" and abs(hz - 10.4) < 0.05: P_on.append(p)
    elif kind == "off": P_off.append(p)
P_on, P_off = np.median(P_on, 0), np.median(P_off, 0)
sel = (f >= 6) & (f <= 16)
ax[1].plot(f[sel], P_on[sel], color=BLUE, lw=2, label="flicker ON at 10.4 Hz (median of 4 blocks)")
ax[1].plot(f[sel], P_off[sel], color=ORANGE, lw=2, label="OFF (median of 20 blocks)")
ax[1].axvline(f_on, color=MUTED, lw=1, ls="--")
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("Oz power (µV²/Hz)")
ax[1].set_title("Oz spectrum at 10.4 Hz flicker: a single-bin line at the\nflicker frequency, alpha hump (8-13 Hz) unchanged", fontsize=10)
ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_response_curve.png", dpi=150)
plt.close(fig)

# ---------------- figure 4: alpha electrode check ----------------
al = json.load(open("runs/alpha1/alpha.json"))
names, et, X, fs = load_eeg("runs/alpha1")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for k, ch in enumerate(("AUX", "TP10")):
    c = names.index(ch); Pc, Po = [], []
    for kind, t0, t1 in al["blocks"]:
        seg = X[(et >= t0 + 2) & (et <= t1), c]
        if len(seg) < fs * 4: continue
        seg = seg - np.median(seg); mad = np.median(np.abs(seg)) * 1.4826 + 1e-9
        seg = np.clip(seg, -6 * mad, 6 * mad)
        f2, p2 = welch(seg, fs=fs, nperseg=int(fs * 2), noverlap=int(fs))
        (Pc if kind == "on" else Po).append(p2)
    Pc, Po = np.median(Pc, 0), np.median(Po, 0)
    sel = (f2 >= 4) & (f2 <= 20)
    ax[k].plot(f2[sel], Pc[sel], color=BLUE, lw=2, label="eyes closed")
    ax[k].plot(f2[sel], Po[sel], color=ORANGE, lw=2, label="eyes open")
    v = al["channels"][ch]
    ax[k].set_title(f"{'Oz aux' if ch == 'AUX' else ch}: eyes-closed alpha peak {v['iaf_hz']:.1f} Hz, x{v['peak_ratio']:.1f} power", fontsize=10)
    ax[k].set_xlabel("frequency (Hz)"); ax[k].set_ylabel("power (µV²/Hz)")
    ax[k].legend(frameon=False)
fig.suptitle("electrode check: the aux cup sees occipital alpha (largest on Oz, smaller behind the ear)", color=INK)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_alpha_check.png", dpi=150)
plt.close(fig)
print("figures ->", OUT, sorted(os.listdir(OUT)))
