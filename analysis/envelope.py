"""SSVEP envelope over time - a diagnostic, not part of the scoring.

  1. remove the DC offset and drift (0.5 Hz high-pass), notch 60 Hz
  2. zero-phase Butterworth band-pass (order 4, sosfiltfilt) at the delivered
     flicker frequency f +- --bw Hz (and at 2f with --harmonic, summed)
  3. full-wave rectify: --rectify abs | square
  4. smooth: --smoother peak   y[n] = x[n] if x[n] > y[n-1] else y[n-1]*exp(-1/(tau fs))
                        lowpass  zero-phase low-pass at --lp-hz
                        hilbert  |analytic signal| of the band-passed EEG

The plot shows the band-passed signal with its envelope on top, shades the
calibration ON blocks and the scan positions (bright target cells darker), and
marks in red the stretches blinkmask.py would leave out (blinks, closed eyes),
so the blink detector can be checked by eye.

    python analysis/envelope.py runs/<session>                  # Oz (AUX)
    python analysis/envelope.py runs/<session> --channels TP9,TP10 --smoother hilbert
    python analysis/envelope.py runs/<session> --t0 120 --t1 180 --harmonic
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import butter, hilbert, iirnotch, sosfiltfilt, filtfilt, tf2sos  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blinkmask          # noqa: E402
import config             # noqa: E402
import reconstruct as R   # noqa: E402


def delivered_freq(session, fallback):
    for name, path in (("calibration.json", ("line", "delivered")),
                       ("xr_session.json", ("stimFreqActual",))):
        p = os.path.join(session, name)
        if os.path.exists(p):
            v = json.load(open(p))
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if v:
                return float(v)
    return fallback


def envelope(x, fs, f0, bw=1.0, harmonic=False, rectify="abs", smoother="peak",
             tau=0.5, lp_hz=1.0, mains=60.0):
    """(band-passed signal, envelope)."""
    x = np.asarray(x, float)
    x = sosfiltfilt(butter(2, 0.5, "highpass", fs=fs, output="sos"), x - np.median(x))
    if mains and mains < fs / 2 - 1:
        b, a = iirnotch(mains, 30.0, fs)
        x = filtfilt(b, a, x)
    bp = np.zeros_like(x)
    for f in ([f0, 2 * f0] if harmonic else [f0]):
        if f + bw < fs / 2:
            bp += sosfiltfilt(butter(4, [f - bw, f + bw], "bandpass", fs=fs, output="sos"), x)
    if smoother == "hilbert":
        return bp, np.abs(hilbert(bp))
    r = np.abs(bp) if rectify == "abs" else bp ** 2
    if smoother == "lowpass":
        return bp, sosfiltfilt(butter(2, lp_hz, "lowpass", fs=fs, output="sos"), r)
    decay = np.exp(-1.0 / (tau * fs))
    y = np.empty_like(r)
    prev = 0.0
    for i, v in enumerate(r):
        prev = v if v > prev else prev * decay
        y[i] = prev
    return bp, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--channels", default="AUX", help="comma list (default Oz = AUX)")
    ap.add_argument("--freq", type=float, default=None,
                    help="flicker frequency (default: the delivered one from the session)")
    ap.add_argument("--bw", type=float, default=1.0, help="band-pass half width, Hz")
    ap.add_argument("--harmonic", action="store_true", help="also band-pass 2f and sum")
    ap.add_argument("--rectify", choices=["abs", "square"], default="abs")
    ap.add_argument("--smoother", choices=["peak", "lowpass", "hilbert"], default="peak")
    ap.add_argument("--tau", type=float, default=0.5, help="peak-detector decay, s")
    ap.add_argument("--lp-hz", type=float, default=1.0, help="lowpass smoother cutoff")
    ap.add_argument("--t0", type=float, default=None, help="window start, s from the first sample")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--no-mask", action="store_true", help="do not mark masked stretches")
    ap.add_argument("--blink-uv", type=float, default=None)
    ap.add_argument("--alpha-ratio", type=float, default=None)
    ap.add_argument("--closure-min-s", type=float, default=None)
    ap.add_argument("--margin-s", type=float, default=None)
    ap.add_argument("--cursor-log", default="cursor_log.csv")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    header, rows = R.read_csv(os.path.join(a.session, "eeg.csv"))
    names = header[1:]
    t = np.array([float(r[0]) for r in rows])
    data = np.array([[float(v) for v in r[1:]] for r in rows])
    fs = R.infer_fs(t)
    f0 = a.freq or delivered_freq(a.session, config.STIM_FREQ_HZ)
    chans = [c.strip() for c in a.channels.split(",") if c.strip()]
    missing = [c for c in chans if c not in names]
    if missing:
        print(f"no {', '.join(missing)} in this session (has {', '.join(names)}); "
              f"using {'TP9' if 'TP9' in names else names[0]}")
        chans = [c for c in chans if c in names] or ["TP9" if "TP9" in names else names[0]]

    valid = None
    if not a.no_mask:
        valid, info = blinkmask.compute(t, data, names, fs, f0, blinkmask.options(a),
                                        blinkmask.calibration_off_spans(a.session))
        print(f"mask: {info['masked_frac'] * 100:.1f}% (blinks {info['blink_frac'] * 100:.1f}%, "
              f"closed eyes {info['closure_frac'] * 100:.1f}%)"
              + (f"; {info['note']}" if info["note"] else ""))

    rel = t - t[0]
    sel = np.ones(len(t), bool)
    if a.t0 is not None:
        sel &= rel >= a.t0
    if a.t1 is not None:
        sel &= rel <= a.t1
    fig, axes = plt.subplots(len(chans), 1, figsize=(15, 3.2 * len(chans)), sharex=True, squeeze=False)
    for ax, ch in zip(axes[:, 0], chans):
        bp, env = envelope(data[:, names.index(ch)], fs, f0, a.bw, a.harmonic, a.rectify,
                           a.smoother, a.tau, a.lp_hz)
        ax.plot(rel[sel], bp[sel], lw=0.4, color="#9aa4b1", label=f"{ch} band-passed {f0:.2f} Hz"
                + (" + 2f" if a.harmonic else ""))
        ax.plot(rel[sel], env[sel], lw=1.4, color="#1f5fbf", label=f"envelope ({a.smoother})")
        lim = np.percentile(np.abs(bp[sel]), 99.5) * 1.3 if sel.any() else 1
        ax.set_ylim(-lim, lim * 1.6)
        ax.set_ylabel(ch)
        # calibration ON blocks
        cal = os.path.join(a.session, "calibration.json")
        if os.path.exists(cal):
            for kind, b0, b1 in json.load(open(cal)).get("blocks", []):
                if kind == "on":
                    ax.axvspan(b0 - t[0], b1 - t[0], color="#2e9e4f", alpha=0.12, lw=0)
        # scan positions: bright target cells shaded
        cl = os.path.join(a.session, a.cursor_log)
        if os.path.exists(cl):
            _, crow = R.read_csv(cl)
            ct = np.array([float(r[0]) for r in crow])
            cx = np.array([int(r[1]) for r in crow])
            cy = np.array([int(r[2]) for r in crow])
            lum = np.array([float(r[3]) for r in crow])
            cut = np.flatnonzero((np.diff(cx) != 0) | (np.diff(cy) != 0))
            for s, e in zip(np.r_[0, cut + 1], np.r_[cut + 1, len(ct)]):
                if cx[s] >= 0:
                    ax.axvspan(ct[s] - t[0], ct[e - 1] - t[0],
                               color="#e0a020" if lum[s] > 0.5 else "#000000",
                               alpha=0.10 if lum[s] > 0.5 else 0.03, lw=0)
        if valid is not None:
            bad = ~valid & sel
            d = np.diff(np.r_[0, bad.astype(int), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                ax.axvspan(rel[s], rel[e - 1], color="#d62728", alpha=0.35, lw=0)
        if sel.any():
            ax.set_xlim(rel[sel][0], rel[sel][-1])
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time, s  (green = calibration ON, amber = bright scan cell, "
                           "red = masked: blink / eyes closed)")
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.session))}: SSVEP envelope at {f0:.2f} Hz",
                 fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(a.session, f"envelope_{'_'.join(chans)}.png")
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    main()
