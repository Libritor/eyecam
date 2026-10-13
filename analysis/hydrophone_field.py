"""Ear as a Hydrophone, the field recording: a Muse S Athena on the listener
while a hydraulophone is played and its sound reaches the ear through the
water. One MuseLog CSV, no stimulus log, no markers. What is measured: the
contact and artifact budget per channel, the whole-file spectra with every
steady line found and sized, a spectrogram search for note-like ridges
(a narrow peak that moves in frequency from one window to the next), the
ear-channel alpha and the broadband level through the session, what the
optical channels carried, and from the measured 40 Hz floor the clean
recording time an auditory steady-state response would need to be read.

    python analysis/hydrophone_field.py <csv> <out_dir>

Writes fig_spectra.png, fig_spectrogram.png, fig_timecourse.png,
fig_budget.png and hydrophone_field.json into <out_dir>.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

FS = 256.0
CH = ["TP9", "AF7", "AF8", "TP10"]
C_CH = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe5"
EPOCH_S, ART_UV = 2.0, 200.0
OPTICS = ["730nm_LEFT_OUTER", "730nm_RIGHT_OUTER", "850nm_LEFT_OUTER", "850nm_RIGHT_OUTER", "730nm_LEFT_INNER",
          "730nm_RIGHT_INNER", "850nm_LEFT_INNER", "850nm_RIGHT_INNER", "RED_LEFT_OUTER", "RED_RIGHT_OUTER",
          "AMBIENT_LEFT_OUTER", "AMBIENT_RIGHT_OUTER", "RED_LEFT_INNER", "RED_RIGHT_INNER", "AMBIENT_LEFT_INNER", "AMBIENT_RIGHT_INNER"]


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def lines_in(f, P, lo=5.0, hi=125.0, ratio=5.0):
    """Narrow peaks: a bin that is a local maximum and `ratio` times the
    median of its 0.6 to 3.5 Hz neighbourhood. Returns (Hz, ratio, uV rms)."""
    df = f[1] - f[0]
    out = []
    for k in np.where((f >= lo) & (f <= hi))[0]:
        nb = P[(np.abs(f - f[k]) > 0.6) & (np.abs(f - f[k]) < 3.5)]
        r = P[k] / np.median(nb)
        if r > ratio and P[k] == P[max(k - 2, 0):k + 3].max():
            amp = np.sqrt(max(P[k] - np.median(nb), 0) * df * 2)   # line power above the floor, as a sinusoid's rms
            out.append((float(f[k]), float(r), float(amp)))
    return out


def budget_seconds(floor, amp_uv, snr=3.0):
    """Seconds of clean recording for a sinusoid of rms amplitude amp_uv to
    reach `snr` over a floor of `floor` uV^2/Hz in a 1/T Hz bin."""
    return snr * floor / (amp_uv ** 2)


def main():
    path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    d = pd.read_csv(path, low_memory=False)
    t = pd.to_numeric(d["ms_ELAPSED"], errors="coerce").to_numpy() / 1000.0
    minutes = (t[-1] - t[0]) / 60.0
    res = dict(file=os.path.basename(path), minutes=minutes, rows=len(d),
               clock=(str(d["CLOCK_TIME"].iloc[0]), str(d["CLOCK_TIME"].iloc[-1])), channels={}, optics={})
    sos = butter(4, [1, 125], btype="band", fs=FS, output="sos")
    X, spectra, specgrams = {}, {}, {}
    for c in CH:
        x = pd.to_numeric(d[c + "_RAW"], errors="coerce")
        X[c] = sosfiltfilt(sos, (x.interpolate().bfill().ffill() - x.median()).to_numpy())
    for c in CH:
        y = X[c]
        L = int(EPOCH_S * FS); m = len(y) // L
        ep = y[:m * L].reshape(m, L); good = np.abs(ep).max(axis=1) < ART_UV
        f, P = welch(ep[good].ravel() if good.sum() >= 5 else y, fs=FS, nperseg=int(8 * FS))
        spectra[c] = (f, P)
        band = lambda a, b: float(P[(f >= a) & (f < b)].mean())
        ln = lines_in(f, P)
        res["channels"][c] = dict(good_share=float(good.mean()), rms_uv=float(np.sqrt(np.mean(ep[good] ** 2))) if good.sum() else None,
                                  delta=band(1, 4), theta=band(4, 8), alpha=band(8, 13), beta=band(13, 30), gamma=band(30, 45),
                                  floor40=float(np.median(P[(f >= 37) & (f <= 43)])), floor80=float(np.median(P[(f >= 77) & (f <= 83)])),
                                  mains60=float(P[np.argmin(np.abs(f - 60))]), lines=ln)
        fs_, ts_, S = spectrogram(y, fs=FS, nperseg=int(4 * FS), noverlap=int(3 * FS), scaling="density")
        specgrams[c] = (fs_, ts_, S)
        # moving-ridge search: a narrow peak whose frequency differs from the previous window's by 1 to 20 Hz
        prev, moving, steady = None, 0, {}
        for j in range(S.shape[1]):
            col = S[:, j]
            pk = [fr for fr, r, a in lines_in(fs_, col, 5, 120, 8.0)]
            for fr in pk:
                steady[int(round(fr))] = steady.get(int(round(fr)), 0) + 1
            if prev is not None:
                for fr in pk:
                    if all(abs(fr - q) > 1.0 for q in prev) and any(1.0 < abs(fr - q) < 20.0 for q in prev):
                        moving += 1
            prev = pk
        top = sorted(steady.items(), key=lambda kv: -kv[1])[:6]
        res["channels"][c]["steady_line_windows"] = {str(k): v for k, v in top}
        res["channels"][c]["moving_ridge_points"] = moving
        res["channels"][c]["windows"] = int(S.shape[1])
        print(f"{c}: good {good.mean():.2f} rms {res['channels'][c]['rms_uv']:.0f} uV; alpha {band(8,13):.1f}; floor40 {res['channels'][c]['floor40']:.2f}; "
              f"lines {[(round(a, 1), round(b, 1), round(u, 2)) for a, b, u in ln]}; steady {top}; moving {moving}/{S.shape[1]}")
    # optics
    for c in OPTICS:
        if c in d:
            v = pd.to_numeric(d[c], errors="coerce").to_numpy()
            res["optics"][c] = dict(n=int(np.isfinite(v).sum()), mean=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                                    lo=float(np.nanmin(v)), hi=float(np.nanmax(v)), distinct=int(len(np.unique(v[np.isfinite(v)]))))
    # time course per 10 s: ear alpha, broadband rms, 60 Hz
    W = int(10 * FS); m = len(t) // W
    alpha, rms, mains = [], [], []
    for k in range(m):
        a, r, mn = [], [], []
        for c in ("TP9", "TP10"):
            seg = X[c][k * W:(k + 1) * W]
            if np.abs(seg).max() < ART_UV:
                f, P = welch(seg, fs=FS, nperseg=int(2 * FS))
                a.append(P[(f >= 8) & (f < 13)].mean()); r.append(np.sqrt(np.mean(seg ** 2))); mn.append(P[np.argmin(np.abs(f - 60))])
        alpha.append(float(np.mean(a)) if a else np.nan); rms.append(float(np.mean(r)) if r else np.nan); mains.append(float(np.mean(mn)) if mn else np.nan)
    res["alpha_10s"], res["rms_10s"], res["mains_10s"] = alpha, rms, mains
    acc = d[["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().to_numpy()
    mag = np.linalg.norm(acc, axis=1)
    res["posture"] = dict(accel_mean_g=[float(v) for v in acc.mean(axis=0)], accel_sd_g=float(mag.std()))
    floor = float(np.mean([res["channels"][c]["floor40"] for c in ("TP9", "TP10")]))
    res["budget"] = dict(floor40_ears=floor, minutes_for_0p2uv=budget_seconds(floor, 0.2) / 60, minutes_for_0p3uv=budget_seconds(floor, 0.3) / 60,
                         minutes_for_0p5uv=budget_seconds(floor, 0.5) / 60)
    print("posture accel mean", np.round(acc.mean(axis=0), 2), "| budget at floor %.2f: %.0f min for 0.2 uV, %.0f for 0.3, %.0f for 0.5" % (
        floor, res["budget"]["minutes_for_0p2uv"], res["budget"]["minutes_for_0p3uv"], res["budget"]["minutes_for_0p5uv"]))
    json.dump(res, open(os.path.join(out_dir, "hydrophone_field.json"), "w"), indent=1, default=float)

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for j, c in enumerate(("TP9", "AF7")):
        ax = axes[j]; f, P = spectra[c]; sel = (f >= 1) & (f <= 128)
        ax.semilogy(f[sel], P[sel], color=C_CH[CH.index(c)], lw=1.2)
        for fr, r, a in res["channels"][c]["lines"]:
            ax.annotate(f"{fr:.0f} Hz", (fr, P[np.argmin(np.abs(f - fr))]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=INK)
        ax.axvspan(37, 43, color=GRID, alpha=0.8, lw=0)
        ax.set_xlabel("frequency, Hz", fontsize=9, color=MUTED)
        style(ax, f"{'a' if j == 0 else 'b'}. {c} ({'behind the left ear' if c == 'TP9' else 'left forehead'})")
    axes[0].set_ylabel("power, uV$^2$/Hz", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_spectra.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for j, c in enumerate(("TP9", "AF7")):
        fs_, ts_, S = specgrams[c]; sel = fs_ <= 128
        im = axes[j].pcolormesh(ts_ / 60.0, fs_[sel], 10 * np.log10(S[sel] + 1e-6), cmap="Greys", vmin=-15, vmax=20, shading="auto", rasterized=True)   # the mesh is a pixel grid; text and axes stay vector
        axes[j].set_ylabel("frequency, Hz", fontsize=9, color=MUTED)
        style(axes[j], f"{'a' if j == 0 else 'b'}. {c}")
        axes[j].grid(False)
    axes[1].set_xlabel("minutes", fontsize=9, color=MUTED)
    cb = fig.colorbar(im, ax=axes, shrink=0.8); cb.set_label("dB re 1 uV$^2$/Hz", fontsize=8, color=MUTED)
    fig.savefig(os.path.join(out_dir, "fig_spectrogram.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    tt = np.arange(m) * 10 / 60.0
    axes[0].semilogy(tt, alpha, "o-", color=C_CH[0], ms=3, lw=1); axes[0].set_ylabel("ear alpha, uV$^2$/Hz", fontsize=9, color=MUTED)
    axes[1].plot(tt, rms, "o-", color=C_CH[0], ms=3, lw=1); axes[1].set_ylabel("ear rms 1 to 125 Hz, uV", fontsize=9, color=MUTED)
    for ax, ttl in ((axes[0], "a. alpha behind the ears"), (axes[1], "b. broadband level behind the ears")):
        ax.set_xlabel("minutes", fontsize=9, color=MUTED); style(ax, ttl)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_timecourse.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    amps = np.linspace(0.1, 1.0, 50)
    ax.semilogy(amps, budget_seconds(floor, amps) / 60.0, color=C_CH[0], lw=1.6)
    ax.axvspan(0.2, 0.5, color=GRID, alpha=0.8, lw=0)
    ax.set_xlabel("steady-state response, uV rms", fontsize=9, color=MUTED); ax.set_ylabel("clean minutes for SNR 3", fontsize=9, color=MUTED)
    style(ax, f"a 40 Hz response over this floor ({floor:.1f} uV$^2$/Hz)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_budget.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
