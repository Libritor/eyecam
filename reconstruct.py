"""Reconstruct the viewed image from EEG + cursor log.

Method (Mann et al., HealthCom 2019, Sec. III):
  per grid cell, estimate SSVEP power = P(f0) + P(2*f0) from a Welch PSD over
  the samples recorded while the cursor sat in that cell, divided by broadband
  power in the 14-50 Hz band; that relative power is the pixel value. Rows are
  blended vertically with Eq. 1, then the grid is normalized and upscaled.

Usage:
    python reconstruct.py --session runs/sim --out runs/sim/reconstruction.png
    python reconstruct.py --session runs/live1 --channels TP9,TP10
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy.signal import welch
from scipy.ndimage import convolve1d

import config


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def load_session(session_dir, channels):
    header, rows = read_csv(os.path.join(session_dir, "eeg.csv"))
    eeg_t = np.array([float(r[0]) for r in rows])
    names = header[1:]
    data = np.array([[float(v) for v in r[1:]] for r in rows])
    if channels:
        idx = []
        for c in channels.split(","):
            c = c.strip()
            idx.append(names.index(c) if c in names else int(c))
        sig = data[:, idx].mean(axis=1)
    else:
        sig = data.mean(axis=1)

    _, crows = read_csv(os.path.join(session_dir, "cursor_log.csv"))
    cur_t = np.array([float(r[0]) for r in crows])
    gx = np.array([int(r[1]) for r in crows])
    gy = np.array([int(r[2]) for r in crows])
    return eeg_t, sig, cur_t, gx, gy


def ssvep_score(segment, fs):
    """Relative SSVEP power of one EEG segment."""
    nperseg = int(round(fs))  # 1-second segments -> 1 Hz bins at any rate
    if len(segment) < nperseg:
        return np.nan
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return psd[m].sum() * df

    nyquist = fs / 2
    f0, f1 = config.STIM_FREQ_HZ, config.HARMONIC_HZ
    sig_p = band_power(f0 - 0.6, f0 + 0.6)
    if f1 < nyquist - 1:
        sig_p += band_power(f1 - 0.6, f1 + 0.6)
    lo, hi = config.NOISE_BAND
    total = band_power(lo, min(hi, nyquist - 1))
    denom = total - sig_p  # broadband minus the stimulus bins
    if denom <= 0:
        return np.nan
    return sig_p / denom


def reconstruct(eeg_t, sig, cur_t, gx, gy, fs=None):
    if fs is None:
        # Infer the actual rate: MuseLog OSC arrives at ~64 Hz (decimated),
        # LSL/simulated sessions at 256 Hz.
        fs = 1.0 / np.median(np.diff(eeg_t))
        print(f"inferred EEG rate: {fs:.1f} Hz")
    nperseg = int(round(fs))
    grid_w, grid_h = gx.max() + 1, gy.max() + 1
    acc = np.zeros((grid_h, grid_w))
    cnt = np.zeros((grid_h, grid_w))

    # Group contiguous cursor samples by cell visit.
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))

    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(eeg_t, t0)
        i1 = np.searchsorted(eeg_t, t1)
        # Pad the window so short dwells still give >= one PSD segment.
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(sig), i0 + nperseg)
        score = ssvep_score(sig[i0:i1], fs)
        if not np.isnan(score):
            acc[gy[s], gx[s]] += score
            cnt[gy[s], gx[s]] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).any():
        grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)

    # Eq. 1 vertical blend of overlapping scan lines.
    k = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, k / k.sum(), axis=0, mode="nearest")
    return grid


def save_image(grid, out_path):
    lo, hi = np.percentile(grid, 2), np.percentile(grid, 98)
    norm = np.clip((grid - lo) / max(hi - lo, 1e-12), 0, 1)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    big = img.resize((img.width * config.UPSCALE, img.height * config.UPSCALE),
                     Image.BICUBIC)
    big.save(out_path)
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="directory containing eeg.csv and cursor_log.csv")
    ap.add_argument("--channels", default="",
                    help="comma-separated channel names or indices (default: mean of all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or os.path.join(args.session, "reconstruction.png")
    eeg_t, sig, cur_t, gx, gy = load_session(args.session, args.channels)
    grid = reconstruct(eeg_t, sig, cur_t, gx, gy)
    np.save(os.path.join(args.session, "reconstruction_grid.npy"), grid)
    save_image(grid, out)
    print(f"reconstruction -> {out}")

    tpath = os.path.join(args.session, "target.npy")
    if os.path.exists(tpath):
        target = np.load(tpath)
        if target.shape == grid.shape:
            r = np.corrcoef(target.ravel(), grid.ravel())[0, 1]
            print(f"correlation with ground-truth target: r = {r:.3f}")


if __name__ == "__main__":
    main()
