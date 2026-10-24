"""State of Strobe, the flicker-free control the paper said it lacked: the same
drowsiness markers through a session with no flicker at all, the Ear as a
Microphone sitting (runs/ear_mic1: 14 min of speech from the headset, eyes open
on a dot, still). If theta/alpha on Oz falls through it at the rate it falls
through the flicker sessions, the drift is time and electrode, not strobe.

Markers per 2 s epoch after a 1-40 Hz band-pass, the same bands and the same
12 Hz line cuts as state_of_flash.py so the numbers are like for like:
log10 theta/alpha, log10 (theta+alpha)/beta and log10 alpha power on Oz (AUX)
and on the ears (TP9, TP10). Epochs over 150 uV peak are left out. Drift is the
Theil-Sen slope per 10 min over the minutes of sound, with its 95% interval.

    python analysis/flicker_free_drift.py runs/ear_mic1
Writes runs/state_of_flash/flicker_free_drift.json, fig_flicker_free.pdf/.png.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import theilslopes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from state_of_flash import band_masks  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "runs", "state_of_flash")
FS = 256.0
EPOCH_S = 2.0
LINE_HZ = 12.0        # cut the same lines as the flicker sessions, so the bands match


def markers_of(seg, fs):
    f, P = welch(seg, fs=fs, nperseg=len(seg), axis=0)
    m_theta, m_alpha, m_beta = band_masks(f, LINE_HZ)
    theta, alpha, beta = P[m_theta].mean(axis=0), P[m_alpha].mean(axis=0), P[m_beta].mean(axis=0)
    return dict(ta=np.log10(theta / alpha), tab=np.log10((theta + alpha) / beta), alpha=np.log10(alpha))


def main():
    S = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "ear_mic1")
    name = os.path.basename(os.path.normpath(S))
    eeg = pd.read_csv(os.path.join(S, "eeg.csv"))
    ev = pd.read_csv(os.path.join(S, "events.csv"))
    t = eeg["time"].to_numpy()
    names = [c for c in ["TP9", "AF7", "AF8", "TP10", "AUX"] if c in eeg.columns]
    X = eeg[names].to_numpy(float)
    sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
    Xb = sosfiltfilt(sos, X - np.median(X, axis=0), axis=0)
    t0, t1 = float(ev.t_on.min()), float(ev.t_off.max())
    oz, ears = names.index("AUX"), [names.index("TP9"), names.index("TP10")]
    rows = []
    for s in np.arange(t0, t1 - EPOCH_S, EPOCH_S):
        m = (t >= s) & (t < s + EPOCH_S)
        if m.sum() < 0.9 * EPOCH_S * FS:
            continue
        seg = Xb[m]
        peak = np.abs(seg).max(axis=0)
        ok_oz, ok_ear = peak[oz] < 150, peak[ears].max() < 150
        mk = markers_of(seg, FS)
        rows.append(dict(min=(s - t0) / 60, ok_oz=bool(ok_oz), ok_ear=bool(ok_ear),
                         ta_oz=float(mk["ta"][oz]), tab_oz=float(mk["tab"][oz]), alpha_oz=float(mk["alpha"][oz]),
                         ta_ear=float(mk["ta"][ears].mean()), tab_ear=float(mk["tab"][ears].mean()),
                         alpha_ear=float(mk["alpha"][ears].mean())))
    df = pd.DataFrame(rows)
    res = dict(session=name, minutes=(t1 - t0) / 60, epochs=len(df),
               ok_oz=int(df.ok_oz.sum()), ok_ear=int(df.ok_ear.sum()), drift={})
    print(f"{name}: {res['minutes']:.1f} min of sound with no flicker, {len(df)} epochs, "
          f"Oz usable {res['ok_oz']}, ears usable {res['ok_ear']}")
    for k in ["ta_oz", "tab_oz", "alpha_oz", "ta_ear", "tab_ear", "alpha_ear"]:
        ok = df.ok_oz if k.endswith("oz") else df.ok_ear
        d = df[ok]
        if len(d) < 20:
            res["drift"][k] = None
            continue
        slope, inter, lo, hi = theilslopes(d[k].to_numpy(), d["min"].to_numpy() / 10.0)
        res["drift"][k] = dict(per_10min=float(slope), lo=float(lo), hi=float(hi), n=int(len(d)),
                               first_3min=float(d[d["min"] < 3][k].median()), last_3min=float(d[d["min"] > d["min"].max() - 3][k].median()))
        print(f"  {k:10s} drift {slope:+.3f} log10 per 10 min  [{lo:+.3f}, {hi:+.3f}]  "
              f"first 3 min {res['drift'][k]['first_3min']:+.2f}, last 3 min {res['drift'][k]['last_3min']:+.2f}  (n {len(d)})")
    # the flicker sessions' pooled drift, for the comparison line
    pooled = {}
    sof_path = os.path.join(OUT, "state_of_flash.json")
    if os.path.exists(sof_path):
        sof = json.load(open(sof_path))
        pooled = sof.get("pooled", {})
    res["flicker_sessions_pooled"] = pooled.get("drift_flicker") or pooled.get("drift") or {}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "flicker_free_drift.json"), "w"), indent=1)
    # figure: theta/alpha and (theta+alpha)/beta on Oz and ears through the sitting
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, k, lab in [(axs[0], "ta", "log10 theta/alpha"), (axs[1], "tab", "log10 (theta+alpha)/beta")]:
        for ch, col in [("oz", "#1f4e9c"), ("ear", "#c8102e")]:
            ok = df.ok_oz if ch == "oz" else df.ok_ear
            d = df[ok]
            med = d[f"{k}_{ch}"].rolling(15, center=True, min_periods=5).median()
            ax.plot(d["min"], d[f"{k}_{ch}"], ".", ms=2, color=col, alpha=0.35)
            ax.plot(d["min"], med, "-", color=col, lw=1.8, label=f"{'Oz' if ch == 'oz' else 'ears'}, 30 s median")
            dr = res["drift"].get(f"{k}_{ch}")
            if dr:
                ax.text(0.02, 0.95 if ch == "oz" else 0.86, f"{'Oz' if ch == 'oz' else 'ears'} drift {dr['per_10min']:+.2f} per 10 min",
                        transform=ax.transAxes, fontsize=8, color=col, va="top")
        ax.set_xlabel("minutes of speech, no flicker"); ax.set_ylabel(lab); ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{name}: the drowsiness markers through a session with no flicker", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_flicker_free.{ext}"), dpi=130)
    print("wrote", os.path.join(OUT, "flicker_free_drift.json"), "and fig_flicker_free.pdf/.png")


if __name__ == "__main__":
    main()
