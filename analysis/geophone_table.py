"""Ear as a geophone: what a Muse headband records while the wearer makes sound in a solid,
the table under them (MuseLog phone file, 256 Hz EEG, accelerometer ~39 Hz, gyroscope).

Segments as the wearer did them: 0-120 s tapping the table with the fingernails, 120-180 s
scratching it, 180-244 s tapping it with the fingers.

  * head vibration: accelerometer magnitude, 3 Hz high-pass, its spread and dominant rhythm;
  * the EEG at that rhythm: magnitude-squared coherence between the head's vibration and each
    electrode (EEG band-passed 1-30 Hz and resampled onto the accelerometer's clock), at the
    segment's rhythm against the median coherence 1-15 Hz;
  * tap-locked EEG: taps found as peaks of the vibration (over the 98th percentile, 120 ms
    apart), EEG 1-30 Hz averaged -0.2..0.5 s on the ear sites and the forehead, against the
    same average at random times in the same segment (the shuffle band);
  * band power per segment, and the high-frequency (40-120 Hz) power where scratching would
    leave any broadband trace.

    python analysis/geophone_table.py runs/geophone1/muse_session_20261104_053232_Muse1.csv docs/paper/hydrophone2027/figures
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, sosfiltfilt, welch

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 6.5, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5})
FS = 256.0
CH = ["TP9", "AF7", "AF8", "TP10"]
SEGS = [("fingernail taps", 0, 120, "#1f4e79"), ("scratching", 120, 180, "#c0392b"), ("finger taps", 180, 244, "#2e8540")]


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(path, low_memory=False)
    t = d["ms_ELAPSED"].to_numpy(float) / 1000
    X = d[[c + "_RAW" for c in CH]].to_numpy(float)
    A = d[["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]].ffill().bfill().to_numpy(float)
    upd = np.r_[0, np.flatnonzero(np.any(np.diff(A, axis=0) != 0, axis=1)) + 1]
    ta, a = t[upd], A[upd]
    fa = len(ta) / (ta[-1] - ta[0])
    am = np.linalg.norm(a, axis=1)
    vib = sosfiltfilt(butter(2, 3, "high", fs=fa, output="sos"), am - am.mean())
    eeg = sosfiltfilt(butter(4, [1, 30], "band", fs=FS, output="sos"), X - X.mean(0), axis=0)
    eeg_on_acc = np.column_stack([np.interp(ta, t, eeg[:, i]) for i in range(len(CH))])
    res = dict(file=os.path.basename(path), minutes=round((t[-1] - t[0]) / 60, 2), accel_rate_hz=round(fa, 1),
               hsi_median={c: float(d[f"{c}_CONNECTION_STRENGTH(HSI)"].median()) for c in CH}, segments=[])
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.75))
    rng = np.random.default_rng(0)
    for name, s0, s1, col in SEGS:
        m = (ta >= s0) & (ta < s1)
        f, P = welch(vib[m], fs=fa, nperseg=256)
        band = (f > 1) & (f < 15)
        rhythm = float(f[band][np.argmax(P[band])])
        ax[0].semilogy(f[band], P[band] * 1e6, color=col, lw=0.9, label=f"{name} (peak {rhythm:.1f} Hz)")
        coh = {}
        for i, c in enumerate(CH):
            fc, C = coherence(vib[m], eeg_on_acc[m, i], fs=fa, nperseg=128)
            k = int(np.argmin(np.abs(fc - rhythm)))
            ref = np.median(C[(fc > 1) & (fc < 15)])
            coh[c] = dict(at_rhythm=round(float(C[k]), 3), median_1_15=round(float(ref), 3))
        # taps: vibration peaks
        thr = np.percentile(np.abs(vib[m]), 98)
        idx = np.flatnonzero(m & (np.abs(vib) > thr))
        taps = []
        for i in idx:
            if not taps or ta[i] - taps[-1] > 0.12:
                taps.append(ta[i])
        n0, n1 = int(-0.2 * FS), int(0.5 * FS)
        tt = np.arange(n0, n1) / FS

        def avg(times):
            ep = []
            for tc in times:
                j = int(np.searchsorted(t, tc))
                if j + n0 >= 0 and j + n1 <= len(t):
                    e = eeg[j + n0:j + n1] - eeg[j + n0:j].mean(0)
                    if np.ptp(e) < 150:
                        ep.append(e)
            return np.array(ep)
        E = avg(taps)
        shuf = [avg(rng.uniform(s0 + 0.5, s1 - 0.5, len(taps))).mean(0) for _ in range(100)]
        ears = E[:, :, [0, 3]].mean(2).mean(0) if len(E) else np.full(len(tt), np.nan)
        sh = np.array([s[:, [0, 3]].mean(1) for s in shuf])
        lo, hi = np.percentile(sh, 2.5, 0), np.percentile(sh, 97.5, 0)
        out_frac = float(np.mean((ears < lo) | (ears > hi))) if len(E) else None
        fe, Pe = welch(X[(t >= s0) & (t < s1)] - X[(t >= s0) & (t < s1)].mean(0), fs=FS, nperseg=512, axis=0)
        bp = {b: [round(float(v), 2) for v in Pe[(fe >= lo_) & (fe < hi_)].mean(0)] for b, (lo_, hi_) in
              dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), hf_40_120=(40, 120)).items()}
        res["segments"].append(dict(name=name, start_s=s0, end_s=s1, vib_sd_mg=round(float(vib[m].std() * 1000), 2),
                                    vib_p99_mg=round(float(np.percentile(np.abs(vib[m]), 99) * 1000), 2),
                                    rhythm_hz=round(rhythm, 2), coherence=coh, taps=len(taps), tap_epochs=int(len(E)),
                                    tap_locked_outside_shuffle_frac=None if out_frac is None else round(out_frac, 3),
                                    tap_locked_ear_peak_uv=None if not len(E) else round(float(np.max(np.abs(ears))), 2),
                                    band_power_uv2hz=bp))
        if name != "scratching":
            ax[2].fill_between(tt * 1000, lo, hi, color=col, alpha=0.15, lw=0)
            ax[2].plot(tt * 1000, ears, color=col, lw=0.9, label=f"{name} (n={len(E)})")
        xs = [CH.index(c) + 0.27 * (SEGS.index((name, s0, s1, col)) - 1) for c in CH]
        ax[1].bar(xs, [coh[c]["at_rhythm"] for c in CH], width=0.25, color=col)
        ax[1].scatter(xs, [coh[c]["median_1_15"] for c in CH], marker="_", s=40, color="k", lw=1.0, zorder=3,
                      label="median 1-15 Hz" if name == "fingernail taps" else None)
    ax[0].set_xlabel("frequency, Hz"); ax[0].set_ylabel("head vibration, mg$^2$/Hz"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) the head's vibration", fontsize=6)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(CH); ax[1].set_ylabel("coherence at the rhythm")
    ax[1].set_title("(b) EEG coherence with the vibration", fontsize=6); ax[1].legend(frameon=False, fontsize=5)
    ax[2].axvline(0, color="#888", lw=0.5); ax[2].set_xlabel("ms from tap"); ax[2].set_ylabel("ear sites, uV")
    ax[2].set_title("(c) tap-locked EEG vs shuffle band", fontsize=6); ax[2].legend(frameon=False, loc="upper right")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(os.path.join(out, "fig_geophone.pdf"))
    fig.savefig(os.path.join(out, "fig_geophone_preview.png"), dpi=200)
    json.dump(res, open(os.path.join(out, "geophone_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
