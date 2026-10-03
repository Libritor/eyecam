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
from scipy.signal import welch, periodogram
from scipy.ndimage import convolve1d

import config


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def load_session(session_dir, channels="", cursor_log="cursor_log.csv"):
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

    _, crows = read_csv(os.path.join(session_dir, cursor_log))
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


def ssvep_score(segment, fs, stim_freq=None):
    """Relative steady-state power of one EEG segment at stim_freq
    (default 15 Hz visual SSVEP; 40 Hz for the auditory ASSR path)."""
    nperseg = int(round(fs))  # 1-second segments -> 1 Hz bins at any rate
    if len(segment) < nperseg:
        return np.nan
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return psd[m].sum() * df

    nyquist = fs / 2
    f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
    f1 = 2.0 * f0
    lo, hi = config.NOISE_BAND
    hi_eff = min(hi, nyquist - 1)
    fund = band_power(f0 - 0.6, f0 + 0.6)
    sig_p = fund
    sig_in_band = fund if lo <= f0 <= hi_eff else 0.0
    if f1 < nyquist - 1:
        harm = band_power(f1 - 0.6, f1 + 0.6)
        sig_p += harm
        if lo <= f1 <= hi_eff:
            sig_in_band += harm
    total = band_power(lo, hi_eff)
    denom = total - sig_in_band  # broadband minus in-band stimulus bins
    if denom <= 0:
        return np.nan
    return sig_p / denom


def line_snr(seg, fs, f0, half=0.15, flank=(0.5, 2.0)):
    """Exact-frequency line detector: mean periodogram power within +-half Hz
    of f0 (at least one bin) over the mean power 0.5-2 Hz either side.
    One Hann periodogram over the whole segment, so a 4 s cell gives 0.25 Hz
    bins: ~8x less noise per bin than the 1 Hz Welch bins of ssvep_score."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < fs * 1.5:
        return np.nan
    seg = seg - seg.mean()
    f, p = periodogram(seg, fs=fs, window="hann", detrend="constant")
    df = f[1] - f[0]
    h = max(half, df)
    pk = p[np.abs(f - f0) <= h].mean()
    fk = p[(np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])].mean()
    return pk / max(fk, 1e-12)


def paper_score(seg, fs, f0):
    """The paper's estimate, computed the way the paper computes it: ONE
    spectrum over the whole dwell (Mann et al. 2019, Sec. III: "power
    spectral density over a 1700-sample window ... the time it takes the
    cursor to pass over a point"), power at f0 plus its first harmonic,
    divided by the power over 14-50 Hz. Signal bins and the mains line are
    left out of the denominator."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < fs * 0.9:
        return np.nan
    f, p = periodogram(seg - np.median(seg), fs=fs, window="hann")
    h = max(0.3, f[1] - f[0])
    sig = p[np.abs(f - f0) <= h].sum()
    if 2 * f0 < fs / 2 - 1:
        sig += p[np.abs(f - 2 * f0) <= h].sum()
    lo, hi = config.NOISE_BAND
    band = ((f >= lo) & (f <= min(hi, fs / 2 - 1)) & (np.abs(f - f0) > h)
            & (np.abs(f - 2 * f0) > h) & (np.abs(f - 60.0) > 2.0))
    den = p[band].sum()
    return sig / den if den > 0 else np.nan


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def hf_sigma(x):
    """High-frequency noise scale, immune to slow electrode drift: robust
    sigma of the first difference, rescaled (white-noise diff has 2x var)."""
    return robust_sigma(np.diff(x)) / np.sqrt(2)


def cell_score(seg, fs, sigma_floor, stim_freq=None, method="welch"):
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
    if method == "line":
        f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
        return line_snr(clipped, fs, f0)
    if method == "paper":
        f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
        return paper_score(clipped, fs, f0)
    return ssvep_score(clipped, fs, stim_freq)


def reconstruct(eeg_t, data, cur_t, gx, gy, fs=None, weights=None,
                stim_freq=None, method="welch"):
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
        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
        # Pad the window so short dwells still give >= one PSD segment.
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)

        num = den = 0.0
        for c in range(n_ch):
            if weights[c] <= 0:
                continue
            sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], stim_freq,
                            method)
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


def run(session_dir, channels="", calibration="", out="", stim_freq=None,
        method="welch", shift_s=0.0, cursor_log="cursor_log.csv",
        weights=None, save=True):
    """shift_s != 0: circularly shift the EEG by that many seconds relative
    to the cursor log = a negative control (same data, wrong alignment)."""
    out = out or os.path.join(session_dir, "reconstruction.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    if weights is None and calibration:
        weights = weights_from_calibration(calibration, names)
        if not shift_s:
            print("channel weights:",
                  {n: round(float(w), 3) for n, w in zip(names, weights)})
    fs = infer_fs(eeg_t)
    if shift_s:
        data = np.roll(data, int(round(shift_s * fs)), axis=0)
    grid = reconstruct(eeg_t, data, cur_t, gx, gy, fs=fs, weights=weights,
                       stim_freq=stim_freq, method=method)
    if save:
        np.save(os.path.join(session_dir, "reconstruction_grid.npy"), grid)
        save_image(grid, out)
        print(f"reconstruction -> {out}")

    r = None
    tpath = os.path.join(session_dir, "target.npy")
    if os.path.exists(tpath):
        target = np.load(tpath)
        if target.shape == grid.shape:
            r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
            if save:
                print(f"correlation with ground-truth target: r = {r:.3f}")
    return grid, r


def null_r(session_dir, n_shifts=12, **kw):
    """r of the same reconstruction with the EEG circularly shifted by
    n_shifts different offsets (31 s .. ): the chance level for this session."""
    eeg_t = np.array([float(r[0]) for r in read_csv(
        os.path.join(session_dir, "eeg.csv"))[1]])
    span = eeg_t[-1] - eeg_t[0]
    rs = []
    for k in range(n_shifts):
        s = 31.0 + k * max(7.0, (span - 62.0) / max(n_shifts, 1))
        if s > span - 10:
            break
        _, r = run(session_dir, shift_s=s, save=False, **kw)
        if r is not None:
            rs.append(r)
    return rs


def reconstruct_mux(session_dir, freqs, weights=None, cursor_log="mux_log.csv",
                    out="", target=None, channels="", stride=None):
    """Multiplexed grey scan: each dwell shows n = len(freqs) neighbouring
    cells (gx..gx+n-1 of row gy), cell k tagged at freqs[k]. The line SNR at
    tag k (channel-weighted) is cell k's value; each tag's values are divided
    by their median over the scan so no tag's gain dominates."""
    out = out or os.path.join(session_dir, "reconstruction_mux.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    ch_sigma = [hf_sigma(data[:, c]) for c in range(n_ch)]
    n = len(freqs)
    if stride is None:  # legacy adjacent layout
        stride = 1
    gw, gh = (gx.max() + 1 + (n - 1) * stride) if stride > 1 else gx.max() + n, gy.max() + 1
    if target is not None:
        gh, gw = np.asarray(target).shape
    vals = {k: [] for k in range(n)}
    cells = []
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    nperseg = int(round(fs))
    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)
        row = []
        for k, f0 in enumerate(freqs):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")
                if not np.isnan(sc):
                    num += w[c] * sc
                    den += w[c]
            row.append(num / den if den > 0 else np.nan)
        cells.append((gy[s], gx[s], row))
    grid = np.full((gh, gw), np.nan)
    med = [np.nanmedian([c[2][k] for c in cells]) or 1.0 for k in range(n)]
    for y, x, row in cells:
        for k in range(n):
            xx = x + k * stride
            if xx < gw and np.isfinite(row[k]):
                grid[y, xx] = row[k] / max(med[k], 1e-9)
    if np.isnan(grid).all():
        raise ValueError("no cell produced a score")
    grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    kern = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, kern / kern.sum(), axis=0, mode="nearest")
    np.save(os.path.join(session_dir, "reconstruction_mux_grid.npy"), grid)
    save_image(grid, out)
    rr = None
    if target is not None and np.asarray(target).shape == grid.shape:
        rr = float(np.corrcoef(np.asarray(target).ravel(), grid.ravel())[0, 1])
    return grid, rr


def reconstruct_color(session_dir, freqs, weights=None, gains=None,
                      cursor_log="color_log.csv", out="", target=None,
                      channels=""):
    """Frequency-tagged RGB: each cell flickers its R, G, B components at
    freqs[0..2]; the line SNR at each frequency (channel-weighted, divided by
    the per-frequency gain from the colour calibration) is that colour plane."""
    out = out or os.path.join(session_dir, "reconstruction_color.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)
    # Per-pass colour-to-frequency assignment (tag rotation). Without the
    # file, colour k used freqs[k] throughout.
    passes = []
    pfile = os.path.join(session_dir, "color_passes.json")
    if os.path.exists(pfile):
        with open(pfile) as f:
            passes = sorted(json.load(f), key=lambda e: e["t"])

    def hz_for(t_visit):
        hz = list(freqs)
        for e in passes:
            if e["t"] <= t_visit + 0.05 and len(e.get("hz", [])) == 3:
                hz = [float(v) for v in e["hz"]]
        return hz
    ch_sigma = [hf_sigma(data[:, c]) for c in range(n_ch)]
    gw, gh = gx.max() + 1, gy.max() + 1
    acc = np.zeros((gh, gw, 3))
    cnt = np.zeros((gh, gw))
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    nperseg = int(round(fs))
    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)
        for k, f0 in enumerate(hz_for(t0)):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")
                if not np.isnan(sc):
                    num += w[c] * sc
                    den += w[c]
            if den > 0:
                acc[gy[s], gx[s], k] += num / den / max(gains[k], 1e-6)
        cnt[gy[s], gx[s]] += 1
    with np.errstate(invalid="ignore"):
        grid = acc / np.maximum(cnt, 1)[:, :, None]
    grid[cnt == 0] = np.nan
    for k in range(3):
        plane = grid[:, :, k]
        if np.isnan(plane).any():
            plane[np.isnan(plane)] = np.nanmedian(plane)
        kern = np.array(config.VERTICAL_KERNEL)
        grid[:, :, k] = convolve1d(plane, kern / kern.sum(), axis=0,
                                   mode="nearest")
    np.save(os.path.join(session_dir, "reconstruction_color_grid.npy"), grid)
    rgb = np.zeros_like(grid)
    for k in range(3):
        lo, hi = np.percentile(grid[:, :, k], 2), np.percentile(grid[:, :, k], 98)
        rgb[:, :, k] = np.clip((grid[:, :, k] - lo) / max(hi - lo, 1e-12), 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    img.resize((gw * config.UPSCALE, gh * config.UPSCALE),
               Image.BICUBIC).save(out)
    res = dict(out=out, r_planes=None, r_all=None)
    if target is not None and np.asarray(target).shape == grid.shape:
        t = np.asarray(target, float)
        res["r_planes"] = [float(np.corrcoef(t[:, :, k].ravel(),
                                             grid[:, :, k].ravel())[0, 1])
                           if t[:, :, k].std() > 0 else float("nan")
                           for k in range(3)]
        res["r_all"] = float(np.corrcoef(t.ravel(), grid.ravel())[0, 1])
        # colour identity per cell: does the dominant plane match the target's?
        dom_t = np.argmax(t, axis=2)
        dom_g = np.argmax(rgb, axis=2)
        m = t.max(axis=2) > 0.5
        res["hue_accuracy"] = float((dom_t[m] == dom_g[m]).mean()) if m.any() else float("nan")
    return grid, rgb, res


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
    ap.add_argument("--method", choices=["welch", "line", "paper"],
                    default="welch")
    ap.add_argument("--freq", type=float, default=None)
    args = ap.parse_args()
    run(args.session, args.channels, args.calibration, args.out,
        stim_freq=args.freq, method=args.method)


if __name__ == "__main__":
    main()
