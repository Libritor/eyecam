"""ABC (Aerial Boat Car) Vehicraft, the field EEG: what a Muse S Athena
gives while the participant rides each mode of the craft on Toronto's
harbour. MuseLog CSVs, no markers. Per recording: contact and artifact
budget, band powers on every channel, the theta/beta ratio of the
State-of-Float paper, the headband's own arousal index, the head motion
rhythm from the accelerometer and gyro, and whether the optical (fNIRS)
channels carried anything.

    python analysis/abc_field.py <out_dir> name=path [name=path ...]

Writes fig_quality.png, fig_spectra.png, fig_bands.png, fig_motion.png,
fig_timecourse.png and abc_field.json into <out_dir>.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

FS = 256.0
CH = ["TP9", "AF7", "AF8", "TP10"]
C_ACT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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


def load(path):
    d = pd.read_csv(path, low_memory=False)
    t = pd.to_numeric(d["ms_ELAPSED"], errors="coerce").to_numpy() / 1000.0
    X = {}
    for c in CH:
        x = pd.to_numeric(d[c + "_RAW"], errors="coerce")
        X[c] = (x.interpolate().bfill().ffill() - x.median()).to_numpy()
    acc = d[["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().to_numpy()
    gyr = d[["GYRO_X", "GYRO_Y", "GYRO_Z"]].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().to_numpy()
    ar = pd.to_numeric(d["AROUSAL_INDEX"], errors="coerce").to_numpy()
    labels = d["AROUSAL_LABEL"].dropna().value_counts(normalize=True).round(3).to_dict()
    optics = {c: int(pd.to_numeric(d[c], errors="coerce").notna().sum()) for c in OPTICS if c in d}
    hsi = {c: pd.to_numeric(d[c + "_CONNECTION_STRENGTH(HSI)"], errors="coerce").value_counts().to_dict() for c in CH}
    return dict(t=t, X=X, acc=acc, gyr=gyr, arousal=ar, labels=labels, optics=optics, hsi=hsi,
                minutes=(t[-1] - t[0]) / 60.0, n=len(d), clock=(str(d["CLOCK_TIME"].iloc[0]), str(d["CLOCK_TIME"].iloc[-1])))


def epochs(y):
    L = int(EPOCH_S * FS)
    m = len(y) // L
    ep = y[:m * L].reshape(m, L)
    return ep, np.abs(ep).max(axis=1) < ART_UV


def analyse(name, path):
    r = load(path)
    sos = butter(4, [1, 100], btype="band", fs=FS, output="sos")
    out = dict(name=name, minutes=r["minutes"], rows=r["n"], clock=r["clock"], fs_est=r["n"] / (r["minutes"] * 60),
               arousal_mean=float(np.nanmean(r["arousal"])), arousal_labels=r["labels"],
               optics_samples=r["optics"], hsi=r["hsi"], channels={})
    spectra = {}
    for c in CH:
        y = sosfiltfilt(sos, r["X"][c])
        ep, good = epochs(y)
        f, P = welch(ep[good].ravel() if good.sum() >= 5 else y, fs=FS, nperseg=int(4 * FS))
        spectra[c] = (f, P)
        band = lambda a, b: float(P[(f >= a) & (f < b)].mean())
        th, be = band(4, 8), band(13, 30)
        out["channels"][c] = dict(good_share=float(good.mean()), rms_uv=float(np.sqrt(np.mean(ep[good] ** 2))) if good.sum() else None,
                                  delta=band(1, 4), theta=th, alpha=band(8, 13), beta=be, gamma=band(30, 45),
                                  tbr=th / be if be > 0 else None, mains60=float(P[np.argmin(np.abs(f - 60))]))
    # per 10 s: ear alpha, theta/beta ratio (median over channels, as State-of-Float), arousal
    W = int(10 * FS)
    m = r["n"] // W
    alpha, tbr, ar = [], [], []
    for k in range(m):
        a_vals, t_vals = [], []
        for c in CH:
            seg = sosfiltfilt(sos, r["X"][c][k * W:(k + 1) * W])
            if np.abs(seg).max() < ART_UV:
                f, P = welch(seg, fs=FS, nperseg=int(2 * FS))
                th, be = P[(f >= 4) & (f < 8)].mean(), P[(f >= 13) & (f < 30)].mean()
                t_vals.append(th / be)
                if c in ("TP9", "TP10"):
                    a_vals.append(P[(f >= 8) & (f < 13)].mean())
        alpha.append(float(np.mean(a_vals)) if a_vals else np.nan)
        tbr.append(float(np.median(t_vals)) if t_vals else np.nan)
        a = r["arousal"][k * W:(k + 1) * W]
        ar.append(float(np.nanmean(a)) if np.isfinite(a).any() else np.nan)
    out["alpha_10s"], out["tbr_10s"], out["arousal_10s"] = alpha, tbr, ar
    out["tbr_median"] = float(np.nanmedian(tbr))
    mag = np.linalg.norm(r["acc"], axis=1)
    fa, Pa = welch(mag - mag.mean(), fs=FS, nperseg=int(20 * FS))
    sel = (fa > 0.2) & (fa < 5)
    gm = np.linalg.norm(r["gyr"], axis=1)
    out["motion"] = dict(acc_sd_g=float(mag.std()), peak_hz=float(fa[sel][np.argmax(Pa[sel])]),
                         gyro_median_dps=float(np.median(gm)), gyro_p90_dps=float(np.percentile(gm, 90)))
    return out, spectra, (fa, Pa)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    acts = [a.split("=", 1) for a in sys.argv[2:]]
    results, spectra, motion = [], {}, {}
    for name, path in acts:
        o, s, mo = analyse(name, path)
        results.append(o); spectra[name] = s; motion[name] = mo
        print(f"{name}: {o['minutes']:.1f} min, fs {o['fs_est']:.0f}, arousal {o['arousal_mean']:.2f} {o['arousal_labels']}, TBR {o['tbr_median']:.2f}, "
              f"motion {o['motion']['peak_hz']:.2f} Hz, gyro median {o['motion']['gyro_median_dps']:.0f} dps, optics samples {sum(o['optics_samples'].values())}")
        for c, v in o["channels"].items():
            print(f"   {c}: good {v['good_share']:.2f} rms {v['rms_uv'] or float('nan'):.0f} alpha {v['alpha']:.1f} theta {v['theta']:.1f} beta {v['beta']:.1f} tbr {v['tbr']:.2f}")
    names = [o["name"] for o in results]
    x = np.arange(len(CH))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, o in enumerate(results):
        axes[0].plot(x + (i - 1) * 0.22, [o["channels"][c]["good_share"] * 100 for c in CH], "o", color=C_ACT[i], ms=7, label=o["name"])
        axes[1].plot(x + (i - 1) * 0.22, [o["channels"][c]["rms_uv"] or 0 for c in CH], "o", color=C_ACT[i], ms=7)
    for ax, lab, ttl in ((axes[0], "epochs under 200 uV, %", "a. usable epochs"), (axes[1], "rms 1 to 100 Hz, uV", "b. signal level in usable epochs")):
        ax.set_xticks(x); ax.set_xticklabels(CH); ax.set_ylabel(lab, fontsize=9, color=MUTED); style(ax, ttl)
    axes[0].set_ylim(0, 105); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_quality.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for j, c in enumerate(("TP9", "AF7")):
        ax = axes[j]
        for i, name in enumerate(names):
            f, P = spectra[name][c]
            sel = (f >= 1) & (f <= 60)
            ax.semilogy(f[sel], P[sel], color=C_ACT[i], lw=1.4, label=name)
        ax.set_xlabel("frequency, Hz", fontsize=9, color=MUTED)
        style(ax, f"{'a' if j == 0 else 'b'}. {c} ({'behind the left ear' if c == 'TP9' else 'left forehead'})")
    axes[0].set_ylabel("power, uV$^2$/Hz", fontsize=9, color=MUTED); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_spectra.png"), dpi=150); plt.close(fig)

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    xb = np.arange(len(bands))
    for i, o in enumerate(results):
        vals = [np.mean([o["channels"][c][b] for c in ("TP9", "TP10")]) for b in bands]
        axes[0].semilogy(xb + (i - 1) * 0.22, vals, "o", color=C_ACT[i], ms=7, label=o["name"])
        axes[1].plot(i, o["tbr_median"], "o", color=C_ACT[i], ms=9)
        axes[1].text(i, o["tbr_median"] + 0.15, f"{o['tbr_median']:.1f}", ha="center", fontsize=8, color=INK)
    axes[0].set_xticks(xb); axes[0].set_xticklabels(bands); axes[0].set_ylabel("ear channels, uV$^2$/Hz", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False); style(axes[0], "a. band power, TP9 and TP10")
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, fontsize=8); axes[1].set_ylim(0, max(o["tbr_median"] for o in results) * 1.4)
    axes[1].set_ylabel("theta/beta ratio, median of 4 channels", fontsize=9, color=MUTED); style(axes[1], "b. theta/beta ratio")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_bands.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, name in enumerate(names):
        fa, Pa = motion[name]
        sel = (fa >= 0.2) & (fa <= 5)
        ax.semilogy(fa[sel], Pa[sel], color=C_ACT[i], lw=1.4, label=name)
    ax.set_xlabel("frequency, Hz", fontsize=9, color=MUTED); ax.set_ylabel("|acceleration| power, g$^2$/Hz", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False); style(ax, "head motion rhythm")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_motion.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, len(results), figsize=(8, 4.2), sharex="col")
    for i, o in enumerate(results):
        tt = np.arange(len(o["tbr_10s"])) * 10 / 60.0
        axes[0, i].plot(tt, o["tbr_10s"], "o-", color=C_ACT[i], ms=3, lw=1)
        axes[1, i].plot(tt, o["arousal_10s"], "o-", color=C_ACT[i], ms=3, lw=1)
        style(axes[0, i], o["name"]); style(axes[1, i])
        axes[1, i].set_xlabel("minutes", fontsize=9, color=MUTED); axes[1, i].set_ylim(0, 1)
    axes[0, 0].set_ylabel("theta/beta ratio", fontsize=9, color=MUTED)
    axes[1, 0].set_ylabel("headband arousal index", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_timecourse.png"), dpi=150); plt.close(fig)

    json.dump(dict(results=results, method=dict(epoch_s=EPOCH_S, artifact_uv=ART_UV, fs=FS,
                   tbr="theta 4-8 Hz over beta 13-30 Hz per 10 s window, median over the four channels, usable windows only")),
              open(os.path.join(out_dir, "abc_field.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
