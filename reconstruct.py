"""Reconstruct the viewed image from EEG + cursor log.

Method (Mann et al., HealthCom 2019, Sec. III):
  per grid cell, estimate SSVEP power = P(f0) + P(2*f0) from a Welch PSD over
  the samples recorded while the cursor sat in that cell, divided by broadband
  power in the 14-50 Hz band; that relative power is the pixel value. Rows are
  blended vertically with Eq. 1, then the grid is normalized and upscaled.

Extensions over the paper's description:
  - channels are scored independently and combined in the power domain
    (time-domain averaging can phase-cancel SSVEP across electrode sites),
    with weights from a calibration.json produced by run_session.py;
  - samples beyond ARTIFACT_Z robust sigmas are clipped, and a cell-channel
    with too many clipped samples (blinks, motion) is dropped.

Usage:
    python reconstruct.py --session runs/sim
    python reconstruct.py --session runs/live1 --calibration runs/live1/calibration.json
    python reconstruct.py --session runs/live1 --channels TP9,TP10
"""

import argparse
import csv
import json
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


def load_session(session_dir, channels=""):
    header, rows = read_csv(os.path.join(session_dir, "eeg.csv"))
    eeg_t = np.array([float(r[0]) for r in rows])
    names = header[1:]
    data = np.array([[float(v) for v in r[1:]] for r in rows])
    if channels:
        idx = []
        for c in channels.split(","):
            c = c.strip()
            idx.append(names.index(c) if c in names else int(c))
        data = data[:, idx]
        names = [names[i] for i in idx]

    _, crows = read_csv(os.path.join(session_dir, "cursor_log.csv"))
    cur_t = np.array([float(r[0]) for r in crows])
    gx = np.array([int(r[1]) for r in crows])
    gy = np.array([int(r[2]) for r in crows])
    return eeg_t, data, names, cur_t, gx, gy


def infer_fs(eeg_t):
    """Mean rate over streaming time, excluding dead-air gaps (dropouts,
    recorder respawns) which would deflate the estimate and shift the whole
    Welch frequency axis off the 15 Hz bin. Median-dt is still avoided:
    burst-clumped timestamps (Wi-Fi aggregation) make the median dt ~0."""
    dt = np.diff(eeg_t)
    good = dt < 0.5  # >0.5 s between samples = stream gap, not sampling
    span = float(dt[good].sum())
    if span <= 0:
        raise ValueError("EEG timestamps do not advance")
    fs = float(good.sum()) / span
    if abs(fs - config.FS) > 0.02 * config.FS and abs(fs - 64) > 2:
        print(f"NOTE: inferred rate {fs:.1f} Hz is neither ~{config.FS} nor "
              "~64 Hz — check the stream")
    return fs


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


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def hf_sigma(x):
    """High-frequency noise scale, immune to slow electrode drift: robust
    sigma of the first difference, rescaled (white-noise diff has 2x var)."""
    return robust_sigma(np.diff(x)) / np.sqrt(2)


def cell_score(seg, fs, sigma_floor):
    """Artifact-aware SSVEP score for one channel's segment.

    Deviations are referenced to the segment's own median and scaled by the
    segment's own robust sigma (floored by the channel's high-frequency
    sigma): a session-global baseline mass-drops entire drift epochs, and a
    session-global sigma is either inflated by that same drift or (diff-
    based) too tight for colored noise. MAD's 50% breakdown keeps the
    threshold honest even when a blink occupies a third of the segment."""
    if len(seg) < 8:
        return np.nan
    baseline = np.median(seg)
    dev = seg - baseline
    sigma = max(robust_sigma(dev), sigma_floor)
    bad = np.abs(dev) > config.ARTIFACT_Z * sigma
    if bad.mean() > config.ARTIFACT_DROP_FRAC:
        return np.nan  # blink/motion-dominated: drop this cell-channel
    clipped = baseline + np.clip(dev, -config.ARTIFACT_Z * sigma,
                                 config.ARTIFACT_Z * sigma)
    return ssvep_score(clipped, fs)


def reconstruct(eeg_t, data, cur_t, gx, gy, fs=None, weights=None):
    """data: (n_samples, n_channels). weights: per-channel array or None."""
    if data.ndim == 1:
        data = data[:, None]
    n_ch = data.shape[1]
    if weights is None:
        weights = np.ones(n_ch)
    weights = np.asarray(weights, dtype=float)

    if fs is None:
        fs = infer_fs(eeg_t)
        print(f"inferred EEG rate: {fs:.1f} Hz")
    nperseg = int(round(fs))

    ch_sigma = [hf_sigma(data[:, c]) for c in range(n_ch)]

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
            i1 = min(len(data), i0 + nperseg)

        num = den = 0.0
        for c in range(n_ch):
            if weights[c] <= 0:
                continue
            sc = cell_score(data[i0:i1, c], fs, ch_sigma[c])
            if not np.isnan(sc):
                num += weights[c] * sc
                den += weights[c]
        if den > 0:
            acc[gy[s], gx[s]] += num / den
            cnt[gy[s], gx[s]] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).all():
        raise ValueError("no cell produced a score — EEG/cursor overlap empty?")
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


def weights_from_calibration(calib_path, names):
    with open(calib_path) as f:
        calib = json.load(f)
    wmap = calib["weights"]
    w = np.array([float(wmap.get(n, 0.0)) for n in names])
    if w.sum() <= 0:
        raise ValueError(f"calibration weights cover none of {names}")
    return w


def run(session_dir, channels="", calibration="", out=""):
    out = out or os.path.join(session_dir, "reconstruction.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels)
    weights = None
    if calibration:
        weights = weights_from_calibration(calibration, names)
        print("channel weights:",
              {n: round(float(w), 3) for n, w in zip(names, weights)})
    grid = reconstruct(eeg_t, data, cur_t, gx, gy, weights=weights)
    np.save(os.path.join(session_dir, "reconstruction_grid.npy"), grid)
    save_image(grid, out)
    print(f"reconstruction -> {out}")

    r = None
    tpath = os.path.join(session_dir, "target.npy")
    if os.path.exists(tpath):
        target = np.load(tpath)
        if target.shape == grid.shape:
            r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
            print(f"correlation with ground-truth target: r = {r:.3f}")
    return grid, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="directory containing eeg.csv and cursor_log.csv")
    ap.add_argument("--channels", default="",
                    help="comma-separated channel names or indices "
                         "(default: all, equally weighted)")
    ap.add_argument("--calibration", default="",
                    help="calibration.json with per-channel weights")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run(args.session, args.channels, args.calibration, args.out)


if __name__ == "__main__":
    main()
