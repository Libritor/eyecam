"""The figures of the Eye as a Camera in XR paper, each as a vector PDF with
real text (TrueType embedded, so the text stays selectable and searchable),
drawn from the session files and the json the other analyses write:

  response_curve.pdf    the laptop sweep (runs/sweep2): Oz line ratio by frequency, and the spectrum at 10.4 Hz
  no_word.pdf           "NO" per pass on the two pixel-font runs (runs/vr_pix_big, runs/vr_pix_small)
  acuity_vision1.pdf    the sealed E and ring read from the EEG (runs/vr_vision1), four panels
  gallery_16.pdf        the photograph gallery: crops over 16 x 16 four-level targets, and the colour officer
  blink_release.pdf     blinks around the moment a cell goes dark (runs/state_of_flash/flash_eog_all.json)
  ssvep_fatigue_left.pdf  every lit cell's SSVEP through the two sittings of 2026-10-10 (runs/state_of_flash/ssvep_fatigue.json)
  (colour_planes9.pdf and planes9_per_plane.pdf come from analysis/planes_figures.py, which saves a PDF beside its PNG)

    python analysis/icce_figures.py docs/paper/icce2027/figures
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import periodogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "targets_img"))
import reconstruct  # noqa: E402
import nofigure  # noqa: E402
import pix_runs_figure  # noqa: E402
from make_photo_gallery import crop_box, photos  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "font.size": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out_dir, name):
    p = os.path.join(out_dir, name + ".pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def load_eeg(sess):
    hdr = io.open(f"{sess}/eeg.csv", encoding="utf-8", errors="replace").readline().strip().split(",")
    E = np.genfromtxt(f"{sess}/eeg.csv", delimiter=",", skip_header=1)
    return hdr[1:], E[:, 0], E[:, 1:], reconstruct.infer_fs(E[:, 0])


def response_curve(out_dir):
    sess = os.path.join(ROOT, "runs", "sweep2")
    sw = json.load(open(f"{sess}/sweep.json"))
    names, et, X, fs = load_eeg(sess)
    ia = names.index("AUX")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    res = sorted(sw["results"].values(), key=lambda r: r["freq"])
    fx = [r["delivered"] for r in res]
    on = [r["channels"]["AUX"]["on_med"] for r in res]
    off = [r["channels"]["AUX"]["off_med"] for r in res]
    w = 0.34
    ax[0].bar(np.arange(len(fx)) - w / 2, on, width=w, color=BLUE, label="flicker on")
    ax[0].bar(np.arange(len(fx)) + w / 2, off, width=w, color=ORANGE, label="off (black)")
    ax[0].set_xticks(np.arange(len(fx)))
    ax[0].set_xticklabels([f"{v:.1f}" for v in fx], fontsize=8)
    ax[0].set_xlabel("flicker frequency, Hz")
    ax[0].axhline(1, color=MUTED, lw=0.8, ls="--")
    ax[0].set_ylabel("Oz line ratio, median over blocks")
    ax[0].set_title("a. response by frequency", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    f_on = [r for r in res if abs(r["freq"] - 10.4) < 0.05][0]["delivered"]
    n = int(6.8 * fs)
    P_on, P_off = [], []
    for kind, t0, t1, hz in sw["blocks"]:
        seg = X[(et >= t0 + 1) & (et <= t1), ia]
        if len(seg) < n:
            continue
        seg = seg[:n] - seg[:n].mean()
        f, p = periodogram(seg, fs=fs, window="hann")
        if kind == "on" and abs(hz - 10.4) < 0.05:
            P_on.append(p)
        elif kind == "off":
            P_off.append(p)
    P_on, P_off = np.median(P_on, 0), np.median(P_off, 0)
    sel = (f >= 6) & (f <= 16)
    ax[1].plot(f[sel], P_on[sel], color=BLUE, lw=1.6, label="flicker on at 10.4 Hz")
    ax[1].plot(f[sel], P_off[sel], color=ORANGE, lw=1.6, label="off")
    ax[1].axvline(f_on, color=MUTED, lw=0.8, ls="--")
    ax[1].set_xlabel("frequency, Hz")
    ax[1].set_ylabel("Oz power, µV²/Hz")
    ax[1].set_title("b. spectrum at 10.4 Hz flicker", loc="left", fontsize=9)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "response_curve")


def no_word(out_dir):
    rows = []
    for s, lab in (("runs/vr_pix_big", "large square"), ("runs/vr_pix_small", "paper-sized square")):
        tgt, passes = pix_runs_figure.analyse(os.path.join(ROOT, s))
        cum = []
        for k in range(len(passes)):
            c = np.nanmean(np.stack(passes[:k + 1]), axis=0)
            cum.append((c, pix_runs_figure.stats(c, tgt)))
        rows.append((lab, tgt, cum))
    ncol = 2 + max(len(r[2]) for r in rows) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.75 * ncol, 1.45 * len(rows)), squeeze=False)

    def norm(g):
        lo, hi = np.nanpercentile(g, [5, 95])
        return np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    for i, (lab, tgt, cum) in enumerate(rows):
        panels = [("shown", tgt, "nearest")]
        for k, (c, sc) in enumerate(cum):
            ttl = "pass 1" if k == 0 else f"passes 1 to {k + 1}"
            panels.append((f"{ttl}, r {sc['r']:.2f}", norm(c), "nearest"))
        c, sc = cum[-1]
        panels.append(("smoothed for display", norm(c), "bicubic"))
        panels.append((f"threshold: {sc['n'] - sc['wrong']} of {sc['n']}", (c > sc["thr"]).astype(float), "nearest"))
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j >= len(panels):
                ax.axis("off")
                continue
            ax.imshow(panels[j][1], cmap="gray", vmin=0, vmax=1, interpolation=panels[j][2])
            ax.set_title(panels[j][0], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, out_dir, "no_word")


def acuity(out_dir):
    s = os.path.join(ROOT, "runs", "vr_vision1")
    g = np.load(os.path.join(s, "reconstruction_grid.npy")).astype(float)
    t = np.load(os.path.join(s, "target.npy")).astype(float)
    m = np.isfinite(g)
    r = float(np.corrcoef(g[m], t[m])[0, 1])
    thr = nofigure.otsu(g[m])
    bw = (g > thr).astype(float)
    right = int(np.sum((bw[m] > 0.5) == (t[m] > 0.5)))
    lo, hi = np.nanpercentile(g, [5, 95])
    norm = np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    panels = [("shown to the eye", t, "nearest"), (f"read from the EEG, r {r:.2f}", norm, "nearest"),
              ("smoothed for display", norm, "bicubic"), (f"threshold: {right} of {int(m.sum())} positions right", bw, "nearest")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.6))
    for ax, (ttl, img, interp) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation=interp)
        ax.set_title(ttl, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(w_pad=0.5)
    save(fig, out_dir, "acuity_vision1")


def gallery(out_dir):
    P = dict(photos())
    cols = [("mann", "the 2019 portrait"), ("officer", "police officer"), ("tesla", "Tesla")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for j, (name, lab) in enumerate(cols):
        im = ImageOps.exif_transpose(Image.open(P[name])).convert("L")
        head = ImageOps.autocontrast(im.crop(crop_box(im, name)), cutoff=1).resize((300, 300), Image.LANCZOS)
        tgt = np.asarray(Image.open(os.path.join(ROOT, "targets_img", f"photo_{name}_16.png")).convert("L"), float) / 255
        axes[0, j].imshow(np.asarray(head), cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(lab, fontsize=8)
        axes[1, j].imshow(tgt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].set_title("16 x 16, four levels", fontsize=8)
    ill = np.asarray(Image.open(os.path.join(ROOT, "targets_img", "officer_illustration.png")).convert("RGB"))
    post = np.asarray(Image.open(os.path.join(ROOT, "targets_img", "officer_ai_12x14.png")).convert("RGB"))
    axes[0, 3].imshow(ill); axes[0, 3].set_title("officer, drawn flat", fontsize=8)
    axes[1, 3].imshow(post, interpolation="nearest"); axes[1, 3].set_title("12 x 14, eight colours", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, out_dir, "gallery_16")


def blink_release(out_dir):
    e = json.load(open(os.path.join(ROOT, "runs", "state_of_flash", "flash_eog_all.json")))["sessions"]
    sess = [("vr_planes9", "2026-10-10, 17:57", "#1baf7a"), ("vr_planes10", "2026-10-10, 19:50", "#4a3aa7"),
            ("vr_planes8", "reference night, 10-07", "#8a8a85")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for name, lab, col in sess:
        p = e[name]["psth"]
        edges = np.array(p["bin_edges_s"]); mids = (edges[:-1] + edges[1:]) / 2
        r = np.array(p["to_dark_per_min"], float); ci = np.array(p["to_dark_ci"], float)
        ax.plot(mids, r, "-", color=col, lw=1.5, label=f"{lab} (n {p['n_to_dark']})")
        ax.fill_between(mids, ci[0], ci[1], color=col, alpha=0.14, lw=0)
    ax.axvspan(edges[0], 0, color="#cfe3fb", alpha=0.45, lw=0)
    ax.axvspan(0, edges[-1], color="#fbe0d3", alpha=0.45, lw=0)
    ax.text(-1.5, ax.get_ylim()[1] * 0.93, "cell lit", ha="center", fontsize=8, color=BLUE)
    ax.text(3.0, ax.get_ylim()[1] * 0.93, "cell black", ha="center", fontsize=8, color=ORANGE)
    ax.set_xlabel("seconds from the cell going dark")
    ax.set_ylabel("blinks per minute")
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    save(fig, out_dir, "blink_release")


def fatigue(out_dir):
    f = json.load(open(os.path.join(ROOT, "runs", "state_of_flash", "ssvep_fatigue.json")))
    sess = [("vr_planes9", "first sitting, 17:57"), ("vr_planes10", "second sitting, 19:50")]
    pcol = {"red": "#e34948", "green": "#008300", "blue": BLUE}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6))
    for ax, (name, lab) in zip(axes, sess):
        v = sorted(f[name]["visits"], key=lambda q: q["t"])
        dark = [q for q in v if not q["lit"]]
        ax.scatter([q["min"] for q in dark], [q["score"] for q in dark], s=9, color="#b5b4ad", label="dark cell", zorder=2)
        if dark:
            ax.axhline(np.median([q["score"] for q in dark]), color="#b5b4ad", lw=0.8, ls=":")
        for plane in ("blue", "red", "green"):
            lit = [q for q in v if q["lit"] and q["plane"] == plane]
            if not lit:
                continue
            x = np.array([q["min"] for q in lit]); y = np.array([q["score"] for q in lit])
            hz = lit[0]["freq"]
            ax.scatter(x, y, s=12, color=pcol[plane], label=f"{plane} lit ({hz:g} Hz)", zorder=3)
            k = 5
            if len(y) >= k:
                med = np.array([np.median(y[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(y))])
                ax.plot(x, med, color=pcol[plane], lw=1.4, zorder=4)
        c = f[name]["plane_calib"]
        ax.set_title(f"{lab}: Oz d' {f[name]['calib_oz_dprime']:.1f} on white; colour calibrations red +{c['red']['stat']:.1f}, "
                     f"green +{c['green']['stat']:.1f}, blue +{c['blue']['stat']:.1f}", loc="left", fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylabel("visit response", fontsize=8)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    axes[1].set_xlabel("minutes since the first flicker")
    fig.tight_layout(h_pad=1.0)
    save(fig, out_dir, "ssvep_fatigue_left")


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for fn in (response_curve, no_word, acuity, gallery, blink_release, fatigue):
        fn(out_dir)


if __name__ == "__main__":
    main()
