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
    return _upstream_ratio(f, p, fs, f0)


def _upstream_ratio(f, p, fs, f0):
    h = max(0.3, f[1] - f[0])
    sig = p[np.abs(f - f0) <= h].sum()
    if 2 * f0 < fs / 2 - 1:
        sig += p[np.abs(f - 2 * f0) <= h].sum()
    lo, hi = config.NOISE_BAND
    band = ((f >= lo) & (f <= min(hi, fs / 2 - 1)) & (np.abs(f - f0) > h)
            & (np.abs(f - 2 * f0) > h) & (np.abs(f - 60.0) > 2.0))
    den = p[band].sum()
    return sig / den if den > 0 else np.nan


def paper_strict_score(seg, fs, f0):
    """Mann et al. 2019, Sec. III, steps 1-2, nothing added: one power
    spectrum of the window (1700 samples at 256 Hz), power at f0 plus its
    first harmonic 2*f0, divided by ALL the power from 14 to 50 Hz. No
    artifact clipping and no bins left out of the denominator (the repo's
    own variant, paper_score, does both)."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < fs * 0.9:
        return np.nan
    f, p = periodogram(seg - seg.mean(), fs=fs, window="hann")
    return _strict_ratio(f, p, fs, f0)


def _strict_ratio(f, p, fs, f0):
    h = max(0.3, f[1] - f[0])
    sig = p[np.abs(f - f0) <= h].sum()
    if 2 * f0 < fs / 2 - 1:
        sig += p[np.abs(f - 2 * f0) <= h].sum()
    lo, hi = config.NOISE_BAND
    den = p[(f >= lo) & (f <= min(hi, fs / 2 - 1))].sum()
    return sig / den if den > 0 else np.nan


def masked_paper_score(seg, valid, fs, f0, kind, sigma_floor, min_len):
    """Paper ratio from the valid samples only (blinks / closed eyes are
    missing data, not zeros): one Hann periodogram per valid piece, every
    piece zero-padded to the window length so the bins line up, averaged
    weighted by piece length (Welch-style), then the unchanged ratio. (Counting
    each short piece's line over its own wider main lobe was tried and scored
    worse on the blink gate: the wider lobe adds noise to dark positions.)
    kind 'paper' = paper_strict_score's ratio; 'upstream' = paper_score's,
    with cell_score's artifact clipping applied to the valid samples."""
    import blinkmask
    seg = np.asarray(seg, dtype=float)
    valid = np.asarray(valid, bool)
    if valid.all():                       # nothing masked: the plain score
        if kind == "paper":
            return paper_strict_score(seg, fs, f0)
        return cell_score(seg, fs, sigma_floor, f0, "paper")
    pieces = blinkmask.segments(valid, min_len)
    if not pieces:
        return np.nan
    if kind == "upstream":
        v = seg[valid]
        baseline = np.median(v)
        sigma = max(robust_sigma(v - baseline), sigma_floor)
        if (np.abs(v - baseline) > config.ARTIFACT_Z * sigma).mean() > config.ARTIFACT_DROP_FRAC:
            return np.nan
        lim = config.ARTIFACT_Z * sigma
        seg = baseline + np.clip(seg - baseline, -lim, lim)
    n = len(seg)
    acc, wsum, f = None, 0.0, None
    for a, b in pieces:
        x = seg[a:b]
        x = x - (np.median(x) if kind == "upstream" else x.mean())
        f, p = periodogram(x, fs=fs, window="hann", nfft=n)
        acc = p * (b - a) if acc is None else acc + p * (b - a)
        wsum += b - a
    if wsum < fs * 0.9:
        return np.nan
    return (_strict_ratio if kind == "paper" else _upstream_ratio)(f, acc / wsum, fs, f0)


def eq1(grid):
    """Step 3, Eq. 1: f(x) = 2x + x1 + x-1 + (x2 + x-2)/2 over the rows above
    and below. Rows missing at the image edges are left out and the weights
    renormalised (no invented rows)."""
    grid = np.asarray(grid, float)
    w = {-2: 0.5, -1: 1.0, 0: 2.0, 1: 1.0, 2: 0.5}
    out = np.zeros_like(grid)
    norm = np.zeros(grid.shape[0])
    for d, wd in w.items():
        for r in range(grid.shape[0]):
            if 0 <= r + d < grid.shape[0]:
                out[r] += wd * grid[r + d]
                norm[r] += wd
    return out / norm[:, None]


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
                stim_freq=None, method="welch", vblend=None, valid=None,
                mask_limit=0.35, min_segment_s=1.0, info=None):
    """data: (n_samples, n_channels). weights: per-channel array or None.

    method 'paper'    = the paper exactly: 1700-sample window centred on each
                        visit, paper_strict_score, then Eq. 1 (vblend default on)
    method 'upstream' = this repo's own version as published: the whole dwell,
                        artifact clipping, signal bins out of the denominator,
                        no Eq. 1 unless config.VERTICAL_KERNEL says so
    'line' / 'welch'  = the line detector / 1 Hz Welch ratio (unchanged)

    valid: optional bool per EEG sample (blinkmask.compute). Masked samples are
    left out of each position's spectrum; a visit whose dwell is more than
    mask_limit masked is dropped (it is redone live). info (dict) receives the
    masked fraction of every visit."""
    if valid is not None and method not in ("paper", "upstream"):
        raise ValueError("blink masking applies to the paper ratio "
                         "(--recon-method paper or upstream)")
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
        if valid is not None:
            frac = float(1.0 - valid[i0:i1].mean()) if i1 > i0 else 1.0
            if info is not None:
                info.setdefault("visits", []).append(
                    dict(gx=int(gx[s]), gy=int(gy[s]), t0=float(t0),
                         masked=round(frac, 4), kept=bool(frac <= mask_limit)))
            if frac > mask_limit:
                continue  # mostly blinks / closed eyes: not this position's data
        if method == "paper":
            # step 1: 1700 samples (at 256 Hz) centred on the visit - the paper's
            # "time it takes the cursor to pass over a point" - or the visit
            # itself when it is shorter (a longer window would mix neighbours)
            n = min(int(round(config.PAPER_WINDOW * fs / config.FS)), max(i1 - i0, nperseg))
            mid = (i0 + i1) // 2
            i0 = max(0, mid - n // 2)
            i1 = min(len(data), i0 + n)
        else:
            # Pad the window so short dwells still give >= one PSD segment.
            need = nperseg - (i1 - i0)
            if need > 0:
                i0 = max(0, i0 - need // 2)
                i1 = min(len(data), i0 + nperseg)

        num = den = 0.0
        for c in range(n_ch):
            if weights[c] <= 0:
                continue
            if valid is not None:
                f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
                sc = masked_paper_score(data[i0:i1, c], valid[i0:i1], fs, f0,
                                        method, ch_sigma[c],
                                        int(round(min_segment_s * fs)))
            elif method == "paper":
                f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
                sc = paper_strict_score(data[i0:i1, c], fs, f0)
            else:
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], stim_freq,
                                "paper" if method == "upstream" else method)
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

    if method == "paper":
        # step 3: Eq. 1 (the paper's own weights), unless switched off
        return eq1(grid) if vblend is not False else grid
    # Eq. 1 vertical blend of overlapping scan lines (repo behaviour).
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
        weights=None, save=True, vblend=None, mask_blinks=False,
        mask_opts=None, info=None):
    """shift_s != 0: circularly shift the EEG by that many seconds relative
    to the cursor log = a negative control (same data, wrong alignment).
    mask_blinks: leave blinks / closed eyes out (blinkmask.py); the mask is
    rolled together with the EEG for the shifted nulls."""
    out = out or os.path.join(session_dir, "reconstruction.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    if weights is None and calibration:
        weights = weights_from_calibration(calibration, names)
        if not shift_s:
            print("channel weights:",
                  {n: round(float(w), 3) for n, w in zip(names, weights)})
    fs = infer_fs(eeg_t)
    valid, minfo = None, None
    vinfo = info if info is not None else {}
    if mask_blinks:
        valid, minfo = session_mask(session_dir, fs, stim_freq, mask_opts)
    if shift_s:
        data = np.roll(data, int(round(shift_s * fs)), axis=0)
        if valid is not None:
            valid = np.roll(valid, int(round(shift_s * fs)))
    o = mask_opts or {}
    grid = reconstruct(eeg_t, data, cur_t, gx, gy, fs=fs, weights=weights,
                       stim_freq=stim_freq, method=method, vblend=vblend,
                       valid=valid, mask_limit=o.get("limit", 0.35),
                       min_segment_s=o.get("min_segment_s", 1.0), info=vinfo)
    if valid is not None:
        vinfo["mask"] = minfo
    if save and valid is not None:
        visits = vinfo.get("visits", [])
        with open(os.path.join(session_dir, "masking.json"), "w") as f:
            json.dump(dict(summary=minfo, options=o, visits=visits,
                           dropped=sum(not v["kept"] for v in visits)), f, indent=1)
        print(f"blink mask: {minfo['masked_frac'] * 100:.1f}% of the recording masked "
              f"(blinks {minfo['blink_frac'] * 100:.1f}%, closed eyes "
              f"{minfo['closure_frac'] * 100:.1f}%); "
              f"{sum(not v['kept'] for v in visits)} of {len(visits)} visits dropped"
              + (f"; {minfo['note']}" if minfo.get("note") else ""))
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


_MASKS = {}


def session_mask(session_dir, fs, stim_freq=None, opts=None):
    """(valid per sample, summary) for the session's eeg.csv, from ALL its
    channels (the frontal and Oz channels may not be in the scored set).
    Cached per session and options: the shifted nulls reuse it."""
    import blinkmask
    o = blinkmask.options(**(opts or {}))
    f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
    path = os.path.join(session_dir, "eeg.csv")
    key = (os.path.abspath(path), os.path.getmtime(path), round(f0, 3),
           tuple(sorted(o.items())))
    if key not in _MASKS:
        header, rows = read_csv(path)
        t = np.array([float(r[0]) for r in rows])
        data = np.array([[float(v) for v in r[1:]] for r in rows])
        _MASKS.clear()
        _MASKS[key] = blinkmask.compute(t, data, header[1:], fs, f0, o,
                                        blinkmask.calibration_off_spans(session_dir))
    return _MASKS[key]


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
        for k, f0 in enumerate(freqs):
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
    ap.add_argument("--method", choices=["welch", "line", "paper", "upstream"],
                    default="welch",
                    help="paper = Mann 2019 exactly (1700-sample window, "
                         "(f + 2f) / 14-50 Hz, Eq. 1); upstream = this repo's "
                         "published variant")
    ap.add_argument("--freq", type=float, default=None)
    args = ap.parse_args()
    run(args.session, args.channels, args.calibration, args.out,
        stim_freq=args.freq, method=args.method)


if __name__ == "__main__":
    main()
