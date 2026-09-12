"""Coherent (software lock-in) SSVEP detection from the LOGGED stimulus timeline.

Reproducible analysis for runs/xr_live6 (live human), runs/xr_live2 (floating
ear electrodes, negative control) and runs/webgate (phantom, positive control).

Method
------
1. EEG clock.  eeg.csv `time` is the UDP ARRIVAL stamp (burst-clumped Wi-Fi,
   PC stalls).  A lock-in needs sample times good to << 1/15 s, so a
   sample-index clock is rebuilt per contiguous stream segment:
       t_hat[k] = alpha + k / fs_hat + env[k]
   with (alpha, fs_hat) a lower-envelope (min-latency) line fit of arrival time
   vs sample index and env[k] a +-2 s rolling minimum of the residual (tracks
   packet loss / rate wander).  The arrival jitter t - t_hat is reported so the
   "jitter is small vs 1/15 s" assumption is evaluated rather than assumed.
2. Page clock.  calib_log/cursor_log rows carry page_t (browser
   performance.now) and the PC arrival time; a lower-envelope line
   t_pc = a + b * page_t maps page time to PC time (WS jitter is a few ms).
3. Reference.  During flicker rows the reference is the zero-order hold of the
   logged flicker_on column (+1 = yellow frame, -1 = blue frame): the exact
   frame-quantized pattern that was on screen.  Outside flicker rows (OFF
   blocks, dark cells) a "virtual" reference floor((page_t - delta) * 2 f0) % 2
   is used, with delta chosen to match the logged column, so OFF windows measure
   the 15 Hz noise floor with the same detector.  A 90 deg (quarter period)
   shifted copy gives the quadrature channel.  The reference is evaluated at
   page_t = (t_hat - lag - a) / b; lag is scanned over +-100 ms and the lag
   maximizing mean ON-block SNR is used (same scan for the control arm).
4. Lock-in.  Per window (calibration block or cursor cell visit) and channel:
   median removal, robust clipping at 4 sigma (window with > 20 % clipped
   samples dropped -- same rule as reconstruct.cell_score), 5-45 Hz band-pass +
   60 Hz notch (zero phase), then I = LP(x * ref), Q = LP(x * ref_90),
   R = sqrt(I^2 + Q^2).  LP = moving average of tau seconds; score = mean R over
   the window.  "snr" scores divide R(15 Hz) by the median of the same
   square-wave lock-in at 12, 13, 17, 18 Hz (local noise floor).
5. Scores.  Calibration d' / ratio / rank exactly as run_session.score_calibration
   (robust d' on 6 ON vs 6 OFF blocks); reconstruction per cell visit, channels
   combined with weights derived from the lock-in calibration by the same rule
   run_session uses, Eq. 1 vertical kernel, Pearson r vs target.npy.
6. Control arm.  Every number is recomputed with the EEG circularly shifted by
   1/3 of the recording (breaks stimulus/EEG alignment, keeps the statistics).
   The reference Welch pipeline (reconstruct.py) is re-run in-process on the
   same windows as the baseline (nothing in runs/ is written).

Usage:  python analysis/lock_in.py            (all three runs)
        python analysis/lock_in.py --runs webgate --fast
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config          # noqa: E402
import reconstruct     # noqa: E402

F0 = config.STIM_FREQ_HZ           # 15 Hz
NOISE_FREQS = (12.0, 13.0, 17.0, 18.0)
HARM_NOISE_FREQS = (27.0, 28.0, 32.0, 33.0)
PAD_S = 0.5                        # filter padding around each window
ENV_HALF_S = 3.0                   # EEG clock: rolling-min half-window (s)
ENV_SMOOTH_S = 5.0                 # EEG clock: smoothing of the envelope (s)
LAGS_MS = np.arange(-100, 101, 2)  # lag scan
TAUS = {"tau1s": 1.0, "tau0.25s": 0.25, "full": None}
HEADLINE = ("sq15", "tau1s", "snr")   # pre-declared headline configuration

RUNS = {
    "xr_live6": "live human (dry electrodes, ears high-Z)",
    "xr_live2": "NEGATIVE control (floating ear electrodes)",
    "webgate": "POSITIVE control (phantom subject over real UDP)",
}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_run(run_dir):
    eeg = pd.read_csv(os.path.join(run_dir, "eeg.csv"))
    names = list(eeg.columns[1:])
    t_arr = eeg.iloc[:, 0].to_numpy(dtype=float)
    data = eeg.iloc[:, 1:].to_numpy(dtype=float)
    order = np.argsort(t_arr, kind="stable")
    if not np.all(np.diff(t_arr) >= 0):
        t_arr, data = t_arr[order], data[order]
    logs = {}
    for name in ("calib_log.csv", "cursor_log.csv"):
        df = pd.read_csv(os.path.join(run_dir, name))
        logs[name] = dict(
            ta=df["time"].to_numpy(float), gx=df["grid_x"].to_numpy(int),
            gy=df["grid_y"].to_numpy(int), lum=df["luminance"].to_numpy(float),
            fl=df["flicker_on"].to_numpy(int), pt=df["page_t"].to_numpy(float))
    with open(os.path.join(run_dir, "calibration.json")) as f:
        calib = json.load(f)
    target = np.load(os.path.join(run_dir, "target.npy"))
    return dict(t_arr=t_arr, data=data, names=names, logs=logs, calib=calib,
                blocks=[(k, float(a), float(b)) for k, a, b in calib["blocks"]],
                target=target)


# ----------------------------------------------------------------------------
# clocks
# ----------------------------------------------------------------------------

def lower_envelope_line(x, y, keep=0.10, iters=4):
    """Line y = a + b x through the LOWER envelope of a cloud whose points are
    all >= the true line (arrival = true time + non-negative latency)."""
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(iters):
        res = y - (a + b * x)
        thr = np.percentile(res, keep * 100)
        sel = res <= thr
        if sel.sum() < 10:
            break
        a, b = np.linalg.lstsq(A[sel], y[sel], rcond=None)[0]
    res = y - (a + b * x)
    a += np.percentile(res, 1)  # anchor on the (robust) minimum latency
    return a, b


def eeg_clock(t_arr, fs_nominal=config.FS):
    """Sample-index clock with per-segment lower-envelope fit + rolling-min
    residual tracking.  Returns t_hat, fs_hat, diagnostics."""
    n = len(t_arr)
    dt = np.diff(t_arr)
    cuts = np.flatnonzero(dt > 0.5) + 1          # stream gaps (>0.5 s)
    seg_bounds = np.concatenate(([0], cuts, [n]))
    t_hat = np.empty(n)
    fs_segs = []
    seg_rates = []
    # global rate from the longest segment
    lengths = np.diff(seg_bounds)
    for s, e in zip(seg_bounds[:-1], seg_bounds[1:]):
        k = np.arange(s, e, dtype=float)
        if e - s >= 20 * fs_nominal:
            a, b = lower_envelope_line(k, t_arr[s:e])
            fs_segs.append((e - s, 1.0 / b))
    if fs_segs:
        fs_hat = fs_segs[int(np.argmax([L for L, _ in fs_segs]))][1]
    else:
        fs_hat = float(fs_nominal)
    for s, e in zip(seg_bounds[:-1], seg_bounds[1:]):
        k = np.arange(s, e, dtype=float)
        base = k / fs_hat
        res = t_arr[s:e] - base
        if e - s >= 20 * fs_nominal:
            a, b = lower_envelope_line(k, t_arr[s:e])
            line = a + b * k
        else:
            line = base + np.percentile(res, 1)
        r2 = t_arr[s:e] - line
        # +-3 s rolling minimum of the residual, 5 s mean: tracks packet loss /
        # rate wander while keeping the 60 Hz mains line 93 % coherent at the
        # 6 s block scale (analysis/clock_check.py; +-2 s / 1 s gave 85 %,
        # raw arrival stamps 56 %).
        w = int(2 * ENV_HALF_S * fs_hat) + 1
        env = minimum_filter1d(r2, size=min(w, e - s), mode="nearest")
        env = uniform_filter1d(env, size=min(int(ENV_SMOOTH_S * fs_hat) + 1, e - s),
                               mode="nearest")
        t_hat[s:e] = line + env
        if e - s >= 20 * fs_nominal:
            seg_rates.append(dict(len_s=round((e - s) / fs_hat, 1),
                                  fs_lower_envelope=round(1.0 / b, 3),
                                  fs_count=round((e - s - 1) / (t_arr[e - 1] - t_arr[s]), 3)))
    jitter = t_arr - t_hat
    seg_len_s = lengths / fs_hat
    diag = dict(fs_hat=fs_hat, n_segments=len(lengths),
                gap_count=int(len(cuts)), segment_rates=seg_rates,
                seg_lengths_s=[round(float(v), 1) for v in seg_len_s],
                jitter_ms=dict(p50=float(np.percentile(jitter, 50) * 1e3),
                               p90=float(np.percentile(jitter, 90) * 1e3),
                               p99=float(np.percentile(jitter, 99) * 1e3),
                               max=float(jitter.max() * 1e3),
                               frac_gt_quarter_period=float(
                                   np.mean(np.abs(jitter) > 1 / (4 * F0)))),
                fs_infer_reference=float(reconstruct.infer_fs(t_arr)))
    return t_hat, fs_hat, diag


def page_clock(logs):
    pts = np.concatenate([lg["pt"] for lg in logs.values()])
    tas = np.concatenate([lg["ta"] for lg in logs.values()])
    o = np.argsort(pts)
    pts, tas = pts[o], tas[o]
    a, b = lower_envelope_line(pts, tas, keep=0.2)
    res = tas - (a + b * pts)
    diag = dict(a=a, b=b, ws_jitter_ms=dict(p50=float(np.percentile(res, 50) * 1e3),
                                            p95=float(np.percentile(res, 95) * 1e3),
                                            max=float(res.max() * 1e3)))
    return a, b, diag


# ----------------------------------------------------------------------------
# reference from the logged stimulus timeline
# ----------------------------------------------------------------------------

class Reference:
    """+-1 square wave in PAGE time: zero-order hold of the logged flicker_on
    during flicker rows, analytic (delta-aligned) elsewhere."""

    def __init__(self, logs):
        pt = np.concatenate([lg["pt"] for lg in logs.values()])
        fl = np.concatenate([lg["fl"] for lg in logs.values()])
        lum = np.concatenate([lg["lum"] for lg in logs.values()])
        o = np.argsort(pt, kind="stable")
        self.pt, self.fl, self.flick = pt[o], fl[o].astype(float), lum[o] > 0.05
        # delta: align the analytic wave to the logged column (frame lag of the
        # renderer, e.g. one 60 Hz frame in headless Chrome)
        best = (-1.0, 0.0)
        for d in np.arange(-0.04, 0.0401, 0.0005):
            an = (np.floor((self.pt[self.flick] - d) * 2 * F0) % 2 == 0)
            agree = float(np.mean(an == (self.fl[self.flick] > 0.5)))
            if agree > best[0]:
                best = (agree, float(d))
        self.agree, self.delta = best

    def square(self, p, f=F0, quarter=0):
        """Analytic square wave at frequency f (page time), quarter shifts by
        quarter/4 of a period."""
        q = p - self.delta - quarter / (4.0 * f)
        return 2.0 * (np.floor(q * 2 * f) % 2 == 0) - 1.0

    def __call__(self, p, quarter=0):
        q = p - quarter / (4.0 * F0)
        i = np.searchsorted(self.pt, q, side="right") - 1
        i = np.clip(i, 0, len(self.pt) - 1)
        near = (q >= self.pt[0]) & (q - self.pt[i] < 0.1) & self.flick[i]
        return np.where(near, 2.0 * self.fl[i] - 1.0, self.square(q + 0.0))


# ----------------------------------------------------------------------------
# lock-in scoring
# ----------------------------------------------------------------------------

def make_filters(fs):
    sos_bp = butter(4, [5.0, 45.0], btype="band", fs=fs, output="sos")
    b, a = iirnotch(60.0, 30.0, fs=fs)
    return np.vstack([sos_bp, tf2sos(b, a)])


def preprocess(x, sos, hf_sig, n_pad):
    """Median removal, 4 sigma robust clip (drop if >20 % clipped), band-pass,
    crop padding.  Returns (signal, clipped_fraction) or (None, frac)."""
    base = np.median(x)
    dev = x - base
    sigma = max(reconstruct.robust_sigma(dev), hf_sig)
    bad = np.abs(dev) > config.ARTIFACT_Z * sigma
    core = bad[n_pad:len(x) - n_pad] if len(x) > 2 * n_pad else bad
    frac = float(core.mean())
    if frac > config.ARTIFACT_DROP_FRAC:
        return None, frac
    dev = np.clip(dev, -config.ARTIFACT_Z * sigma, config.ARTIFACT_Z * sigma)
    if len(dev) < 3 * 8 * 2 + 1:
        return None, frac
    y = sosfiltfilt(sos, dev)
    return y[n_pad:len(y) - n_pad], frac


def lockin_mag(x, ri, rq, n_tau):
    """Mean lock-in magnitude with a tau-sample moving-average low-pass
    (n_tau=None -> single coherent average over the window).  Also returns the
    full-window complex phasor."""
    pi, pq = x * ri, x * rq
    I, Q = pi.mean(), pq.mean()
    phasor = complex(I, Q)
    if n_tau is None or n_tau >= len(x):
        return float(abs(phasor)), phasor
    cs_i = np.concatenate(([0.0], np.cumsum(pi)))
    cs_q = np.concatenate(([0.0], np.cumsum(pq)))
    mi = (cs_i[n_tau:] - cs_i[:-n_tau]) / n_tau
    mq = (cs_q[n_tau:] - cs_q[:-n_tau]) / n_tau
    return float(np.mean(np.sqrt(mi ** 2 + mq ** 2))), phasor


def score_window(x, page_t, ref, fs):
    """All score variants for one preprocessed window (x on page-time axis)."""
    out = {}
    r15 = (ref(page_t, 0), ref(page_t, 1))
    r30 = (ref.square(page_t, 2 * F0, 0), ref.square(page_t, 2 * F0, 1))
    noise15 = [(ref.square(page_t, f, 0), ref.square(page_t, f, 1))
               for f in NOISE_FREQS]
    noise30 = [(ref.square(page_t, f, 0), ref.square(page_t, f, 1))
               for f in HARM_NOISE_FREQS]
    for tname, tau in TAUS.items():
        n_tau = None if tau is None else int(round(tau * fs))
        m15, ph = lockin_mag(x, *r15, n_tau)
        m30, _ = lockin_mag(x, *r30, n_tau)
        n15 = np.median([lockin_mag(x, a, b, n_tau)[0] for a, b in noise15])
        n30 = np.median([lockin_mag(x, a, b, n_tau)[0] for a, b in noise30])
        out[("sq15", tname, "raw")] = m15
        out[("sq15", tname, "snr")] = m15 / max(n15, 1e-12)
        out[("sq15+30", tname, "raw")] = m15 + m30
        out[("sq15+30", tname, "snr")] = m15 / max(n15, 1e-12) + m30 / max(n30, 1e-12)
        if tname == "full":
            out["phasor"] = ph
    return out


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def dprime_stats(on, off):
    """Identical to run_session.score_calibration."""
    on, off = np.asarray(on, float), np.asarray(off, float)
    if len(on) < 3 or len(off) < 3:
        return dict(dprime=0.0, ratio=1.0, rank=0, n_on=len(on), n_off=len(off))
    spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
    spread = max(spread, 0.05 * abs(float(np.median(on))))
    return dict(dprime=float((np.median(on) - np.median(off)) / max(spread, 1e-12)),
                ratio=float(np.median(on) / max(np.median(off), 1e-12)),
                rank=int((on > off.max()).sum()), n_on=len(on), n_off=len(off))


def weights_rule(stats_by_ch, names):
    """run_session's weighting: (ratio-1) for channels with d' >= 0.5, else
    uniform fallback."""
    w = {n: (max(0.0, stats_by_ch[n]["ratio"] - 1.0)
             if stats_by_ch[n]["dprime"] >= config.CALIB_WEIGHT_DPRIME_MIN else 0.0)
         for n in names}
    tot = sum(w.values())
    if tot > 0:
        return {k: v / tot for k, v in w.items()}, False
    return {k: 1.0 / len(names) for k in names}, True


# ----------------------------------------------------------------------------
# analysis of one (run, EEG matrix) pair
# ----------------------------------------------------------------------------

class Analyzer:
    def __init__(self, run, data, t_hat, fs, page_ab, ref, fast=False):
        self.run, self.data, self.t_hat, self.fs = run, data, t_hat, fs
        self.a, self.b = page_ab
        self.ref = ref
        self.names = run["names"]
        self.sos = make_filters(fs)
        self.hf = [reconstruct.hf_sigma(data[:, c]) for c in range(data.shape[1])]
        self.n_pad = int(PAD_S * fs)
        self.fast = fast
        self._win_cache = {}

    def window(self, t0, t1, c):
        """Preprocessed channel-c signal for PC-time window [t0, t1] with its
        page-time axis (before lag)."""
        key = (round(t0, 6), round(t1, 6), c)
        if key in self._win_cache:
            return self._win_cache[key]
        i0 = np.searchsorted(self.t_hat, t0)
        i1 = np.searchsorted(self.t_hat, t1)
        if i1 - i0 < int(0.5 * self.fs):
            self._win_cache[key] = (None, None, 1.0)
            return self._win_cache[key]
        p0, p1 = max(0, i0 - self.n_pad), min(len(self.t_hat), i1 + self.n_pad)
        n_pad_l, n_pad_r = i0 - p0, p1 - i1
        x = self.data[p0:p1, c]
        base = np.median(x)
        dev = x - base
        sigma = max(reconstruct.robust_sigma(dev), self.hf[c])
        bad = np.abs(dev) > config.ARTIFACT_Z * sigma
        frac = float(bad[n_pad_l:len(x) - n_pad_r].mean())
        if frac > config.ARTIFACT_DROP_FRAC or len(x) < 60:
            self._win_cache[key] = (None, None, frac)
            return self._win_cache[key]
        dev = np.clip(dev, -config.ARTIFACT_Z * sigma, config.ARTIFACT_Z * sigma)
        y = sosfiltfilt(self.sos, dev)[n_pad_l:len(x) - n_pad_r]
        t = self.t_hat[i0:i1]
        self._win_cache[key] = (y, t, frac)
        return self._win_cache[key]

    def page_time(self, t, lag):
        return (t - lag - self.a) / self.b

    def calib_windows(self):
        return [(k, t0 + config.CALIB_DISCARD_S, t1) for k, t0, t1 in self.run["blocks"]]

    def calib_scores(self, lag, variants=None):
        """{variant: {ch: {'on': [...], 'off': [...]}}} plus phasors."""
        res = {}
        phasors = {n: {"on": [], "off": []} for n in self.names}
        clipped = {n: [] for n in self.names}
        for kind, t0, t1 in self.calib_windows():
            for c, n in enumerate(self.names):
                y, t, frac = self.window(t0, t1, c)
                clipped[n].append(frac)
                if y is None:
                    continue
                sc = score_window(y, self.page_time(t, lag), self.ref, self.fs)
                phasors[n][kind].append(sc.pop("phasor"))
                for key, v in sc.items():
                    if variants is not None and key not in variants:
                        continue
                    res.setdefault(key, {}).setdefault(n, {"on": [], "off": []})[kind].append(v)
        return res, phasors, clipped

    def lag_scan(self):
        key = HEADLINE
        best = None
        curve = []
        lags = LAGS_MS[::5] if self.fast else LAGS_MS
        for lag_ms in lags:
            res, _, _ = self.calib_scores(lag_ms / 1e3, variants={key})
            vals = [np.mean(res[key][n]["on"]) for n in self.names
                    if key in res and n in res[key] and res[key][n]["on"]]
            obj = float(np.mean(vals)) if vals else 0.0
            curve.append(obj)
            if best is None or obj > best[1]:
                best = (float(lag_ms) / 1e3, obj)
        curve = np.array(curve)
        return best[0], dict(best_lag_ms=best[0] * 1e3, obj_max=float(curve.max()),
                             obj_min=float(curve.min()),
                             obj_ratio_max_min=float(curve.max() / max(curve.min(), 1e-12)))

    def calibration(self, lag):
        res, phasors, clipped = self.calib_scores(lag)
        out = {}
        for key, per_ch in res.items():
            out[key] = {n: dprime_stats(per_ch[n]["on"], per_ch[n]["off"])
                        for n in self.names if n in per_ch}
            for n in self.names:
                if n not in out[key]:
                    out[key][n] = dict(dprime=0.0, ratio=1.0, rank=0, n_on=0, n_off=0)
        plv = {}
        for n in self.names:
            for kind in ("on", "off"):
                ph = np.array(phasors[n][kind])
                plv[(n, kind)] = float(abs(np.mean(ph / np.abs(ph)))) if len(ph) >= 3 else float("nan")
        clip_mean = {n: float(np.mean(v)) for n, v in clipped.items()}
        return out, plv, clip_mean

    def cell_visits(self):
        lg = self.run["logs"]["cursor_log.csv"]
        gx, gy, ta = lg["gx"], lg["gy"], lg["ta"]
        change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
        starts = np.concatenate(([0], change + 1))
        ends = np.concatenate((change + 1, [len(gx)]))
        return [(gx[s], gy[s], ta[s], ta[e - 1]) for s, e in zip(starts, ends)]

    def reconstruct(self, lag, weights_by_variant):
        """weights_by_variant: {variant: {ch: w}} -> {variant: (grid, r)}.
        Also returns per-channel grids for the headline variant."""
        lg = self.run["logs"]["cursor_log.csv"]
        gw, gh = lg["gx"].max() + 1, lg["gy"].max() + 1
        variants = list(weights_by_variant)
        acc = {v: np.zeros((gh, gw)) for v in variants}
        cnt = {v: np.zeros((gh, gw)) for v in variants}
        per_ch = {n: (np.zeros((gh, gw)), np.zeros((gh, gw))) for n in self.names}
        for gx, gy, t0, t1 in self.cell_visits():
            scores = {}
            for c, n in enumerate(self.names):
                y, t, frac = self.window(t0, t1, c)
                if y is None:
                    continue
                sc = score_window(y, self.page_time(t, lag), self.ref, self.fs)
                sc.pop("phasor")
                scores[n] = sc
                per_ch[n][0][gy, gx] += sc[HEADLINE]
                per_ch[n][1][gy, gx] += 1
            for v in variants:
                num = den = 0.0
                for n, sc in scores.items():
                    w = weights_by_variant[v].get(n, 0.0)
                    if w > 0:
                        num += w * sc[v]
                        den += w
                if den > 0:
                    acc[v][gy, gx] += num / den
                    cnt[v][gy, gx] += 1
        out = {}
        for v in variants:
            out[v] = finish_grid(acc[v], cnt[v], self.run["target"])
        ch_out = {n: finish_grid(a, c, self.run["target"]) for n, (a, c) in per_ch.items()}
        return out, ch_out


def finish_grid(acc, cnt, target):
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).all():
        return grid, float("nan")
    grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    k = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, k / k.sum(), axis=0, mode="nearest")
    r = float("nan")
    if target.shape == grid.shape and np.std(grid) > 0:
        r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
    return grid, r


# ----------------------------------------------------------------------------
# reference (Welch) pipeline, re-run in-process for a like-for-like baseline
# ----------------------------------------------------------------------------

def baseline_calibration(run, data):
    t = run["t_arr"]
    fs = reconstruct.infer_fs(t)
    out = {}
    for c, n in enumerate(run["names"]):
        sigma = reconstruct.hf_sigma(data[:, c])
        sc = {"on": [], "off": []}
        for kind, t0, t1 in run["blocks"]:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            v = reconstruct.cell_score(data[i0:i1, c], fs, sigma)
            if not np.isnan(v):
                sc[kind].append(float(v))
        out[n] = dprime_stats(sc["on"], sc["off"])
    return out


def baseline_reconstruction(run, data, weights):
    lg = run["logs"]["cursor_log.csv"]
    w = np.array([weights.get(n, 0.0) for n in run["names"]])
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        grid = reconstruct.reconstruct(run["t_arr"], data, lg["ta"], lg["gx"], lg["gy"],
                                       weights=w)
    r = float(np.corrcoef(run["target"].ravel(), grid.ravel())[0, 1]) \
        if run["target"].shape == grid.shape else float("nan")
    return grid, r


# ----------------------------------------------------------------------------
# clock validation with the 60 Hz mains line (a built-in reference oscillator)
# ----------------------------------------------------------------------------

def mains_clock_check(run, data, t_hat, fs_hat, c):
    """Sharpness of the 60 Hz mains line over the calibration span under (a) the
    sample-index clock and (b) the raw arrival stamps (Lomb-Scargle).  A clock
    good to << 1/60 s keeps the line coherent; jitter smears it."""
    from scipy.signal import lombscargle
    lg = run["logs"]["calib_log.csv"]
    s0, s1 = lg["ta"][0], lg["ta"][-1]
    i0, i1 = np.searchsorted(t_hat, s0), np.searchsorted(t_hat, s1)
    x = data[i0:i1, c].astype(float)
    x = x - np.median(x)
    sig = reconstruct.robust_sigma(x)
    x = np.clip(x, -config.ARTIFACT_Z * sig, config.ARTIFACT_Z * sig)
    x = x - x.mean()
    n = len(x)
    freqs = np.arange(58.0, 62.0001, 0.005)
    out = {}
    for label, t in (("index_clock", t_hat[i0:i1]),
                     ("index_clock_fs_infer", (np.arange(n) / run["fs_infer"]) + t_hat[i0]),
                     ("arrival_stamps", run["t_arr"][i0:i1])):
        t = t - t[0]
        P = lombscargle(t, x, 2 * np.pi * freqs, normalize=False)
        near = np.abs(freqs - 60.0) <= 0.5
        k = int(np.argmax(np.where(near, P, -1)))
        f_peak = float(freqs[k])
        floor = float(np.median(P[np.abs(freqs - f_peak) > 0.2]))
        # fraction of 59.5-60.5 band power concentrated within +-0.03 Hz of the peak
        band = np.abs(freqs - 60.0) <= 0.5
        core = np.abs(freqs - f_peak) <= 0.03
        conc = float(P[core].sum() / max(P[band].sum(), 1e-12))
        out[label] = dict(f_peak=f_peak, peak_over_floor=float(P[k] / max(floor, 1e-12)),
                          concentration=conc)
    return out


# ----------------------------------------------------------------------------
# null distribution over circular shifts and synthetic-SSVEP sensitivity
# ----------------------------------------------------------------------------

def lockin_quick(run, data, clocks, lag, variants):
    """Calibration stats + reconstruction r for given variants (no lag scan)."""
    t_hat, fs, page_ab, ref = clocks
    an = Analyzer(run, data, t_hat, fs, page_ab, ref)
    calib, _, _ = an.calibration(lag)
    weights = {v: weights_rule(calib[v], run["names"])[0] for v in variants}
    recon, _ = an.reconstruct(lag, weights)
    uni = {n: 1.0 / len(run["names"]) for n in run["names"]}
    recon_uni, _ = an.reconstruct(lag, {v: uni for v in variants})
    return calib, {v: recon[v][1] for v in variants}, {v: recon_uni[v][1] for v in variants}


def null_shifts(run, data, clocks, lag, K, variants):
    n = len(data)
    rows = []
    for frac in np.linspace(0.08, 0.92, K):
        shift = int(frac * n)
        d = np.roll(data, shift, axis=0)
        b_cal = baseline_calibration(run, d)
        _, b_r = baseline_reconstruction(run, d, run["calib"]["weights"])
        _, b_r_uni = baseline_reconstruction(run, d, {k: 1.0 for k in run["names"]})
        calib, r_w, r_u = lockin_quick(run, d, clocks, lag, variants)
        row = dict(shift=shift, base_dmax=max(s["dprime"] for s in b_cal.values()),
                   base_r=b_r, base_r_uni=b_r_uni)
        for v in variants:
            row[f"lock_dmax_{'/'.join(v)}"] = max(s["dprime"] for s in calib[v].values())
            row[f"lock_r_{'/'.join(v)}"] = r_w[v]
            row[f"lock_r_uni_{'/'.join(v)}"] = r_u[v]
        rows.append(row)
    return pd.DataFrame(rows)


def inject_ssvep(run, data, clocks, amp_uv, latency=0.08, harm=0.4):
    """Add a stimulus-locked synthetic SSVEP (sin + 0.4 sin 2nd harmonic, 0.3 s
    entrainment lag, amplitude follows the logged luminance) to every channel,
    on the sample-index clock."""
    from scipy.signal import lfilter
    t_hat, fs, (a, b), ref = clocks
    pt = (t_hat - a) / b
    lum = np.concatenate([lg["lum"] for lg in run["logs"].values()])
    lpt = np.concatenate([lg["pt"] for lg in run["logs"].values()])
    o = np.argsort(lpt, kind="stable")
    lpt, lum = lpt[o], lum[o]
    i = np.clip(np.searchsorted(lpt, pt, side="right") - 1, 0, len(lpt) - 1)
    drive = np.where((pt >= lpt[0]) & (pt - lpt[i] < 0.1), lum[i], 0.0)
    a_env = np.exp(-1.0 / (0.3 * fs))
    env = lfilter([1 - a_env], [1, -a_env], drive)
    phi = 2 * np.pi * F0 * (pt - latency)
    s = amp_uv * env * (np.sin(phi) + harm * np.sin(2 * phi))
    return data + s[:, None]


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def fmt_ch(d, key="dprime", w=7):
    return "  ".join(f"{n}={d[n][key]:+{w}.2f}" if key == "dprime" else f"{n}={d[n][key]:{w}.2f}"
                     for n in d)


def analyze_arm(run, data, clocks, label, fast):
    t_hat, fs, page_ab, ref = clocks
    an = Analyzer(run, data, t_hat, fs, page_ab, ref, fast=fast)
    t0 = time.time()
    lag, lag_diag = an.lag_scan()
    calib, plv, clip_mean = an.calibration(lag)
    # weights per variant from that variant's own calibration
    weights = {}
    fallback = {}
    for v, stats in calib.items():
        weights[v], fallback[v] = weights_rule(stats, run["names"])
    recon, recon_ch = an.reconstruct(lag, weights)
    # uniform-weight reconstruction for the headline variant
    uni = {n: 1.0 / len(run["names"]) for n in run["names"]}
    recon_uni, _ = an.reconstruct(lag, {HEADLINE: uni})
    return dict(label=label, lag=lag, lag_diag=lag_diag, calib=calib, plv=plv,
                clip_mean=clip_mean, weights=weights, fallback=fallback,
                recon=recon, recon_ch=recon_ch, recon_uni=recon_uni[HEADLINE],
                seconds=time.time() - t0)


def passed(st):
    """run_session gate; rank threshold clamped to n_on-1 as xr_session does
    for short calibrations (webgate uses 3 blocks)."""
    rank_min = min(config.CALIB_RANK_MIN, max(st.get("n_on", config.CALIB_BLOCKS) - 1, 1))
    return bool(st["dprime"] >= config.CALIB_DPRIME_MIN
                and st["ratio"] >= config.CALIB_RATIO_MIN
                and st["rank"] >= rank_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--fast", action="store_true", help="coarser lag scan")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "lock_in_results.json"))
    ap.add_argument("--null-shifts", type=int, default=20,
                    help="circular shifts for the null distribution (0 = skip)")
    ap.add_argument("--inject-amps", default="0,0.25,0.5,1,2,4,8",
                    help="synthetic SSVEP amplitudes (uV) to inject into xr_live6 ('' = skip)")
    args = ap.parse_args()
    results = {}
    for name in args.runs.split(","):
        run_dir = os.path.join(ROOT, "runs", name)
        print("=" * 100)
        print(f"RUN {name}: {RUNS.get(name, '')}")
        print("=" * 100)
        run = load_run(run_dir)
        data = run["data"]
        n = len(data)
        t_hat, fs, cdiag = eeg_clock(run["t_arr"])
        run["fs_infer"] = cdiag["fs_infer_reference"]
        a, b, pdiag = page_clock(run["logs"])
        ref = Reference(run["logs"])
        clocks = (t_hat, fs, (a, b), ref)
        print(f"EEG clock: fs_hat={fs:.4f} Hz (reference infer_fs={cdiag['fs_infer_reference']:.3f}); "
              f"segments={cdiag['n_segments']} gaps>0.5s={cdiag['gap_count']} seg_len_s={cdiag['seg_lengths_s']}")
        print(f"  per-segment rate estimates (lower-envelope slope vs sample count/span): {cdiag['segment_rates']}")
        j = cdiag["jitter_ms"]
        print(f"  arrival-stamp jitter vs index clock (ms): p50={j['p50']:.1f} p90={j['p90']:.1f} "
              f"p99={j['p99']:.1f} max={j['max']:.0f}; frac |jitter|>1/4 period(16.7ms)={j['frac_gt_quarter_period']:.3f}")
        w = pdiag["ws_jitter_ms"]
        print(f"Page clock: t_pc = {a:.3f} + {b:.7f}*page_t; WS jitter ms p50={w['p50']:.1f} p95={w['p95']:.1f} max={w['max']:.0f}")
        print(f"Reference: logged flicker_on vs analytic floor((page_t-delta)*30)%2 agreement={ref.agree:.4f} "
              f"at delta={ref.delta*1e3:+.1f} ms (ZOH of logged column used inside flicker rows)")
        # rail fractions inside calibration + scan spans
        lg = run["logs"]
        spans = [(lg["calib_log.csv"]["ta"][0], lg["calib_log.csv"]["ta"][-1]),
                 (lg["cursor_log.csv"]["ta"][0], lg["cursor_log.csv"]["ta"][-1])]
        for tag, (s0, s1) in zip(("calib", "scan"), spans):
            i0, i1 = np.searchsorted(run["t_arr"], s0), np.searchsorted(run["t_arr"], s1)
            seg = data[i0:i1]
            rail = [(float(np.mean((seg[:, c] < 5) | (seg[:, c] > 1670)))) for c in range(seg.shape[1])]
            print(f"  rail fraction during {tag}: " + "  ".join(f"{nm}={r:.3f}" for nm, r in zip(run["names"], rail)))

        # baseline (reference Welch pipeline) real + control
        shift = n // 3
        data_ctl = np.roll(data, shift, axis=0)
        b_cal = baseline_calibration(run, data)
        b_cal_c = baseline_calibration(run, data_ctl)
        wb = run["calib"]["weights"]
        _, b_r = baseline_reconstruction(run, data, wb)
        _, b_r_c = baseline_reconstruction(run, data_ctl, wb)
        n_on = sum(1 for k, _, _ in run["blocks"] if k == "on")
        print(f"  calibration blocks: {n_on} ON / {len(run['blocks']) - n_on} OFF; EEG rows={n}; control shift={shift} samples")
        print("\nBASELINE (reference Welch relative power, reconstruct.py/run_session.py re-run in-process)")
        print(f"  calib d'  real   : {fmt_ch(b_cal)}   passed={passed(b_cal[max(b_cal, key=lambda k: b_cal[k]['dprime'])])}")
        print(f"  calib d'  control: {fmt_ch(b_cal_c)}")
        print(f"  recon r (calibration.json weights {json.dumps({k: round(v, 2) for k, v in wb.items()})}): real={b_r:.3f}  control={b_r_c:.3f}")

        real = analyze_arm(run, data, clocks, "real", args.fast)
        ctl = analyze_arm(run, data_ctl, clocks, "control(shift N/3)", args.fast)

        H = HEADLINE
        print(f"\nLOCK-IN (headline: {H[0]} reference, LP tau={H[1]}, {H[2]} score)  [{real['seconds']:.0f}s + {ctl['seconds']:.0f}s]")
        for arm in (real, ctl):
            ld = arm["lag_diag"]
            print(f"  {arm['label']:>18}: lag scan best={ld['best_lag_ms']:+.0f} ms  objective max/min={ld['obj_ratio_max_min']:.3f} "
                  f"(max {ld['obj_max']:.3f}, min {ld['obj_min']:.3f})")
        for arm in (real, ctl):
            st = arm["calib"][H]
            best = max(st, key=lambda k: st[k]["dprime"])
            print(f"  {arm['label']:>18} calib d': {fmt_ch(st)}  | ratio: {fmt_ch(st, 'ratio', 5)} | rank: "
                  + " ".join(f"{k}={st[k]['rank']}/{st[k]['n_on']}" for k in st)
                  + f"  best={best} passed={passed(st[best])}")
        for arm in (real, ctl):
            st = arm["calib"][H]
            best = max(st, key=lambda k: st[k]["dprime"])
            print(f"  {arm['label']:>18} phase-locking across blocks (sq15, full-window phasor), best ch {best}: "
                  f"PLV_on={arm['plv'][(best, 'on')]:.2f} PLV_off={arm['plv'][(best, 'off')]:.2f}; "
                  f"clipped-frac per ch: " + " ".join(f"{k}={v:.2f}" for k, v in arm["clip_mean"].items()))
        for arm in (real, ctl):
            wgt = arm["weights"][H]
            print(f"  {arm['label']:>18} recon r: calib-weights={arm['recon'][H][1]:.3f} "
                  f"(w={json.dumps({k: round(v, 2) for k, v in wgt.items()})}{' UNIFORM-FALLBACK' if arm['fallback'][H] else ''})"
                  f"  uniform={arm['recon_uni'][1]:.3f}  per-ch: "
                  + " ".join(f"{k}={g[1]:.3f}" for k, g in arm["recon_ch"].items()))
        # variant table
        print("\n  VARIANT TABLE (real | control): best-channel d' (channel), passed, recon r with that variant's calib weights")
        print(f"  {'variant':<24} {'d_real':>8} {'ch':>5} {'pass':>5} {'r_real':>7} | {'d_ctl':>8} {'ch':>5} {'r_ctl':>7}")
        for v in sorted(real["calib"], key=lambda k: (k[0], k[1], k[2])):
            sr, sc = real["calib"][v], ctl["calib"][v]
            br = max(sr, key=lambda k: sr[k]["dprime"])
            bc = max(sc, key=lambda k: sc[k]["dprime"])
            print(f"  {'/'.join(v):<24} {sr[br]['dprime']:+8.2f} {br:>5} {str(passed(sr[br])):>5} {real['recon'][v][1]:7.3f} | "
                  f"{sc[bc]['dprime']:+8.2f} {bc:>5} {ctl['recon'][v][1]:7.3f}")

        # --- clock validation with the mains line (strongest-mains channel) ---
        mains = {}
        c_m = int(np.argmax([reconstruct.robust_sigma(data[:, c]) for c in range(data.shape[1])]))
        mains = mains_clock_check(run, data, t_hat, fs, c_m)
        print(f"\n  CLOCK CHECK via 60 Hz mains line on {run['names'][c_m]} over the calibration span (Lomb-Scargle 58-62 Hz):")
        for k, v in mains.items():
            print(f"    {k:<22} peak at {v['f_peak']:.3f} Hz  peak/floor={v['peak_over_floor']:.0f}  "
                  f"fraction of 59.5-60.5 Hz power within +-0.03 Hz of peak={v['concentration']:.2f}")

        # --- null distribution over circular shifts ---
        null_df = None
        if args.null_shifts > 0:
            variants = [HEADLINE, ("sq15", "full", "raw")]
            t0 = time.time()
            null_df = null_shifts(run, data, clocks, real["lag"], args.null_shifts, variants)
            print(f"\n  NULL DISTRIBUTION over {args.null_shifts} circular shifts (8-92 % of the recording) [{time.time() - t0:.0f}s]")
            print(f"  {'statistic':<34} {'real':>7} {'null mean':>9} {'null sd':>7} {'null max':>8} {'frac null >= real':>17}")

            def null_row(label, real_v, col):
                v = null_df[col].to_numpy(float)
                print(f"  {label:<34} {real_v:7.3f} {v.mean():9.3f} {v.std():7.3f} {v.max():8.3f} {np.mean(v >= real_v):17.2f}")
                return dict(real=real_v, null_mean=float(v.mean()), null_sd=float(v.std()),
                            null_max=float(v.max()), frac_null_ge_real=float(np.mean(v >= real_v)))
            null_summary = {}
            null_summary["base_dmax"] = null_row("baseline best-channel d'", max(s["dprime"] for s in b_cal.values()), "base_dmax")
            null_summary["base_r"] = null_row("baseline recon r (calib weights)", b_r, "base_r")
            for v in variants:
                tag = "/".join(v)
                sr = real["calib"][v]
                null_summary[f"lock_dmax_{tag}"] = null_row(f"lock-in {tag} best d'", max(s["dprime"] for s in sr.values()), f"lock_dmax_{tag}")
                null_summary[f"lock_r_{tag}"] = null_row(f"lock-in {tag} r (calib w)", real["recon"][v][1], f"lock_r_{tag}")
            null_summary["lock_r_uni"] = null_row(f"lock-in {'/'.join(HEADLINE)} r (uniform)", real["recon_uni"][1], f"lock_r_uni_{'/'.join(HEADLINE)}")
            null_df.to_csv(os.path.join(ROOT, "analysis", f"lock_in_null_{name}.csv"), index=False)

        # --- synthetic-SSVEP sensitivity on the real background (xr_live6) ---
        inject = {}
        if name == "xr_live6" and args.inject_amps:
            amps = [float(v) for v in args.inject_amps.split(",") if v != ""]
            variants = [HEADLINE, ("sq15", "full", "raw"), ("sq15", "tau1s", "raw")]
            print("\n  SENSITIVITY: synthetic stimulus-locked SSVEP (sin 15 Hz + 0.4 sin 30 Hz, latency 80 ms, 0.3 s entrainment,"
                  " amplitude x logged luminance) added to ALL channels of the real xr_live6 EEG on the index clock")
            hdr = [("baseline Welch dprime per ch", 44), ("lock-in sq15/tau1s/snr dprime per ch", 44),
                   ("lock-in sq15/full/raw dprime per ch", 44)]
            print(f"  {'A uV':>5} | " + " | ".join(f"{h:<{w}} {'pass':>5} {'r':>6}" for h, w in hdr))
            for A in amps:
                d = inject_ssvep(run, data, clocks, A)
                b_cal_i = baseline_calibration(run, d)
                bw, _ = weights_rule(b_cal_i, run["names"])
                _, b_r_i = baseline_reconstruction(run, d, bw)
                calib_i, r_w_i, _ = lockin_quick(run, d, clocks, real["lag"], variants)
                cells = []
                for stats, r_i in ((b_cal_i, b_r_i), (calib_i[HEADLINE], r_w_i[HEADLINE]),
                                   (calib_i[variants[1]], r_w_i[variants[1]])):
                    best = max(stats, key=lambda k: stats[k]["dprime"])
                    cells.append(f"{fmt_ch(stats, w=6):<44} {str(passed(stats[best])):>5} {r_i:6.3f}")
                print(f"  {A:5.1f} | " + " | ".join(cells))
                inject[str(A)] = dict(baseline=dict(calib=b_cal_i, r=b_r_i),
                                      lockin={"/".join(v): dict(calib=calib_i[v], r=r_w_i[v]) for v in variants})

        results[name] = dict(
            eeg_clock=cdiag, page_clock=pdiag, ref_agree=ref.agree, ref_delta_ms=ref.delta * 1e3,
            mains_clock_check=mains,
            null=(null_summary if null_df is not None else None),
            inject=inject,
            baseline=dict(calib=b_cal, calib_control=b_cal_c, r=b_r, r_control=b_r_c),
            lockin=dict(
                lag_ms=real["lag"] * 1e3, lag_diag=real["lag_diag"],
                calib={"/".join(k): v for k, v in real["calib"].items()},
                recon_r={"/".join(k): v[1] for k, v in real["recon"].items()},
                recon_r_uniform=real["recon_uni"][1],
                recon_r_per_channel={k: v[1] for k, v in real["recon_ch"].items()},
                weights={"/".join(k): v for k, v in real["weights"].items()},
                plv={f"{k[0]}_{k[1]}": v for k, v in real["plv"].items()}),
            control=dict(
                lag_ms=ctl["lag"] * 1e3, lag_diag=ctl["lag_diag"],
                calib={"/".join(k): v for k, v in ctl["calib"].items()},
                recon_r={"/".join(k): v[1] for k, v in ctl["recon"].items()},
                recon_r_uniform=ctl["recon_uni"][1],
                recon_r_per_channel={k: v[1] for k, v in ctl["recon_ch"].items()},
                plv={f"{k[0]}_{k[1]}": v for k, v in ctl["plv"].items()}),
        )
        np.save(os.path.join(ROOT, "analysis", f"lock_in_grid_{name}.npy"), real["recon"][H][0])
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
