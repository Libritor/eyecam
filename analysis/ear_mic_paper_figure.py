"""The Ear as a Microphone figure for the ICCE paper, vector with real text:
a. the clip's loudness envelope and the brain's (the mean of the held-out
   reconstructions) through the scored part of the clip;
b. every play decoded by a model that never saw it, with the circular-shift
   null band and the shifted-EEG control arm.
Reads <session>/ear_mic_results.json, brain_envelope_64hz.npy and clip.wav.

    python analysis/ear_mic_paper_figure.py runs/ear_mic2 docs/paper/icce2027/figures/ear_mic.pdf
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ear_mic_decode as d  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6, "font.family": "DejaVu Sans"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe5"


def main():
    S, out = sys.argv[1], sys.argv[2]
    R = json.load(open(os.path.join(S, "ear_mic_results.json")))
    A, C = R["aligned"], R.get("control")
    trim = float(R.get("trim_s", 3.0)) if "trim_s" in R else 3.0
    x, fs = sf.read(os.path.join(S, "clip.wav"), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    k = int(round(trim * fs))
    x = x[k:len(x) - k] if trim > 0 else x
    y = d.zscore(d.band_env(d.song_envelope(x, fs)))
    b = d.zscore(np.load(os.path.join(S, "brain_envelope_64hz.npy")))
    n = min(len(y), len(b))
    y, b = y[:n], b[:n]
    t = np.arange(n) / d.FS_D + trim
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.18), gridspec_kw=dict(width_ratios=[1.55, 1]))
    a = ax[0]
    a.plot(t, y, color=MUTED, lw=0.5, label="loudness of the clip")
    a.plot(t, b, color="#c0392b", lw=0.7, label=f"read from the EEG, r = {A['r_avg']:.2f}")
    a.set_xlabel("time in the clip (s)")
    a.set_ylabel("envelope (z)")
    a.set_xlim(t[0], t[-1])
    a.set_title(f"a. clip loudness (grey), from the EEG (red), r = {A['r_avg']:.2f}", fontsize=6, loc="left")
    a = ax[1]
    reps = A["reps_used"]
    a.axhspan(-A["null_single_95"], A["null_single_95"], color="0.9")
    a.plot(reps, A["r_single"], "o", color="#c0392b", ms=2)
    if C:
        a.plot(C["reps_used"], C["r_single"], "x", color="0.45", ms=2)
    a.axhline(0, color="k", lw=0.5)
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.5 * (hi - lo))
    a.set_xlabel("play")
    a.set_ylabel("r, one play")
    a.text(0.02, 0.97, f"aligned, mean {A['r_single_mean']:.3f} (dots)" + (f"\nshifted EEG, mean {C['r_single_mean']:.3f} (crosses)" if C else ""),
           transform=a.transAxes, fontsize=4.8, va="top", ha="left", color=INK,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    a.set_title("b. each play, held out", fontsize=6, loc="left")
    for ax_ in ax:
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.tick_params(labelsize=5)
    fig.tight_layout(w_pad=0.6)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
