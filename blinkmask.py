"""Blink and eye-closure masking for SSVEP scoring.

With the eyes shut there is no visual input, so the SSVEP is genuinely absent:
those stretches are MISSING data. They are marked invalid and left out of the
spectrum, never filled with held or interpolated values.

  blinks    AF7 / AF8 (frontal, right above the eyes) low-passed to 0.5-10 Hz;
            a sample is a blink when either channel deviates more than
            --blink-uv from zero.
  closure   8-12 Hz alpha power on Oz (AUX) in 1 s windows rising above
            --alpha-ratio times the eyes-open baseline for at least
            --closure-min-s. The flicker line f0 (and 2 f0) is cut out of the
            alpha band, because a 10-12 Hz stimulus would otherwise look like
            closed eyes. Without an Oz channel closure detection is off.
  margin    every masked stretch grows by --mask-margin-s on each side.

Used by reconstruct.py (scoring and the shifted-EEG nulls), xr_session.py
(live redo of a position) and analysis/envelope.py (display).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULTS = dict(blink_uv=100.0, alpha_ratio=3.0, closure_min_s=1.0,
                margin_s=0.2, limit=0.35, min_segment_s=1.0)

FRONTAL = ("AF7", "AF8")
OZ = ("AUX", "AUXR", "AUX1", "AUXL", "OZ")


def options(args=None, **kw):
    """Mask settings from argparse attributes (--blink-uv ...) or keywords."""
    o = dict(DEFAULTS)
    for k in o:
        v = getattr(args, k, None) if args is not None else None
        if v is not None:
            o[k] = float(v)
    o.update({k: float(v) for k, v in kw.items() if v is not None})
    return o


def oz_index(names):
    for i, n in enumerate(names):
        if n.upper() in OZ:
            return i
    return None


def _dilate(bad, n):
    if n <= 0 or not bad.any():
        return bad
    k = np.ones(2 * n + 1)
    return np.convolve(bad.astype(float), k, mode="same") > 0


def blink_mask(data, names, fs, blink_uv):
    """True where a frontal channel shows a blink-sized slow deflection."""
    bad = np.zeros(len(data), bool)
    hi = min(10.0, 0.45 * fs)
    sos = butter(2, [0.5, hi], "bandpass", fs=fs, output="sos")
    for n in FRONTAL:
        if n in names and len(data) > 30:
            x = data[:, names.index(n)].astype(float)
            y = sosfiltfilt(sos, x - np.median(x))
            bad |= np.abs(y) > blink_uv
    return bad


ALPHA_WIN_S = 2.0   # 2 s Hann windows: a line spreads +-1 Hz, so cutting the
                    # flicker out +-1 Hz really removes it (1 s would need +-2 Hz)


def alpha_bins(fs, f0, win_s=ALPHA_WIN_S):
    """Boolean bins of an rfft of win_s seconds: 8-12 Hz minus f0, 2 f0 +-1 Hz."""
    f = np.fft.rfftfreq(int(round(win_s * fs)), 1 / fs)
    band = (f >= 8.0) & (f <= 12.0)
    if f0:
        for h in (1, 2):
            band &= np.abs(f - h * f0) > 2.0 / win_s
    return f, band


def alpha_power(x, fs, f0=None, win_s=ALPHA_WIN_S, hop_s=0.25):
    """(centre times in samples, 8-12 Hz power per window) with f0 and 2 f0
    cut out of the band (see alpha_bins)."""
    n = int(round(win_s * fs))
    hop = max(1, int(round(hop_s * fs)))
    if len(x) < n:
        return np.array([], int), np.array([])
    f, band = alpha_bins(fs, f0, win_s)
    w = np.hanning(n)
    centres, pw = [], []
    for i in range(0, len(x) - n + 1, hop):
        seg = x[i:i + n] - x[i:i + n].mean()
        p = np.abs(np.fft.rfft(seg * w)) ** 2
        centres.append(i + n // 2)
        pw.append(p[band].sum())
    return np.array(centres), np.array(pw)


def closure_mask(data, names, fs, f0, alpha_ratio, closure_min_s, baseline=None):
    """True where Oz alpha stays above alpha_ratio x baseline for closure_min_s.
    Returns (mask, baseline_used, note)."""
    bad = np.zeros(len(data), bool)
    oz = oz_index(names)
    if oz is None:
        return bad, None, "no Oz (AUX) channel: eye-closure detection off"
    note = ""
    f, band = alpha_bins(fs, f0)
    left = band.sum() * (f[1] - f[0])
    if left < 2.0:
        return bad, None, (f"flicker {f0:.2f} Hz covers the 8-12 Hz alpha band (only "
                           f"{left:.1f} Hz left after cutting it out): eye-closure "
                           "detection off - use a flicker of 13 Hz or more")
    if f0 and 7.0 <= f0 <= 13.0:
        note = (f"flicker {f0:.2f} Hz is near the alpha band: it is cut out (+-1 Hz), "
                f"closure detection uses the remaining {left:.0f} Hz of 8-12 Hz")
    x = data[:, oz].astype(float)
    centres, pw = alpha_power(x - np.median(x), fs, f0)
    if not len(pw):
        return bad, None, note
    base = baseline if baseline else float(np.median(pw))
    hot = pw > alpha_ratio * base
    # sustained: runs of hot windows covering >= closure_min_s
    hop = (centres[1] - centres[0]) if len(centres) > 1 else int(fs)
    need = max(1, int(np.ceil(closure_min_s * fs / hop)))
    i = 0
    half = int(fs * ALPHA_WIN_S / 2)
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if j - i >= need:
                bad[max(0, centres[i] - half):min(len(bad), centres[j - 1] + half)] = True
            i = j
        else:
            i += 1
    return bad, base, note


def alpha_baseline(t, data, names, fs, f0, spans):
    """Median eyes-open alpha power on Oz over the given (t0, t1) spans
    (calibration OFF blocks: eyes open on the dot), or None."""
    oz = oz_index(names)
    if oz is None or not spans:
        return None
    vals = []
    for a, b in spans:
        i0, i1 = np.searchsorted(t, [a, b])
        if i1 - i0 > fs:
            x = data[i0:i1, oz].astype(float)
            vals.extend(alpha_power(x - np.median(x), fs, f0)[1])
    return float(np.median(vals)) if vals else None


def compute(t, data, names, fs, f0=None, opts=None, baseline_spans=None,
            alpha_base=None):
    """valid (bool per sample) and a summary dict. alpha_base: a baseline
    measured earlier (live use, where the recent window is too short)."""
    o = opts or options()
    blink = blink_mask(data, names, fs, o["blink_uv"])
    base = alpha_base or alpha_baseline(t, data, names, fs, f0, baseline_spans)
    clos, base_used, note = closure_mask(data, names, fs, f0, o["alpha_ratio"],
                                         o["closure_min_s"], base)
    margin = int(round(o["margin_s"] * fs))
    bad = _dilate(blink, margin) | _dilate(clos, margin)
    info = dict(blink_frac=float(blink.mean()), closure_frac=float(clos.mean()),
                masked_frac=float(bad.mean()), alpha_baseline=base_used,
                baseline_from=("calibration OFF blocks" if base else "session median"),
                note=note, frontal=[n for n in FRONTAL if n in names],
                oz=(names[oz_index(names)] if oz_index(names) is not None else None))
    return ~bad, info


def segments(valid, min_len):
    """[(i0, i1)] runs of valid samples at least min_len long."""
    v = np.concatenate(([False], np.asarray(valid, bool), [False]))
    d = np.diff(v.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def calibration_off_spans(session_dir):
    """Eyes-open baseline spans: the OFF blocks of the grey calibration."""
    import json
    import os
    p = os.path.join(session_dir, "calibration.json")
    if not os.path.exists(p):
        return None
    try:
        return [(a, b) for kind, a, b in json.load(open(p)).get("blocks", []) if kind == "off"]
    except (ValueError, TypeError):
        return None
