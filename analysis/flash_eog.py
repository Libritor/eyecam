"""State of Flash, the eyes: blinks, horizontal eye movements and slow EOG
drift through the two flicker sessions of 2026-10-10 (vr_planes9 at 17:57,
vr_planes10 at 19:50), with vr_planes8 (2026-10-07) as a level reference.
The subject reported irritated eyes and a slight forehead ache afterwards,
not drowsiness.  Eye irritation on a display usually shows as a suppressed
blink rate while fixating, blinks released in the dark / rest moments, and a
blink rate, long-blink share or burst share that climbs with minutes on task.

Where the blink lives on a Muse.  Every Muse channel is measured against the
Fpz reference, which sits right above the nasion: the most blink-sensitive
spot on the head.  A blink therefore appears as the SAME negative pulse on
all the channels far from the eyes (TP9, TP10 and the Oz cup), and only as
the small forehead gradient on AF7 / AF8 (3-10 uV here, at the noise floor).
Two detectors are run on a 0.5-8 Hz band-pass:
  frontal   (as specified) a positive deflection on both AF7 and AF8 within
            100 ms, each above 3x the stage's robust sd, 50-500 ms wide at
            half height
  reference -median(TP9, TP10, AUX), positive up: a pulse above 3x the robust
            sd and at least 25 uV, 50-500 ms wide at half height, present on
            all three far channels (each >= half the median pulse); the
            40 uV subset is also reported
Each detector is also run on the sign-flipped signal (the mirror): symmetric
noise triggers both equally, so the mirror rate is the false-alarm floor and
rate - mirror the net; mirror events within 0.5 s of a blink (its band-pass
undershoot) are dropped and the floor is per minute of blink-free time.
Blink duration is the width at 20 % of the peak height (base) and at half
height (FWHM); long blink = base > 400 ms; burst = another blink within 1 s.  Horizontal eye movements: |AF7 - AF8| (0.5-15 Hz) peaks
above max(3x robust sd, 20 uV); the share that follows a cell / block change
within 0.6 s is the saccade to the new cell, the rest is the free rate.  Slow
EOG drift: 0.1-0.5 Hz power of AF7 / AF8 in 20 s segments (dry forehead
electrodes drift with sweat too).  Windows touching a movement artifact (two
or more channels above 150 uV after a 1-40 Hz band-pass, or a railed sample,
+-1 s) are left out; the minutes before the first flicker are mostly
movement and are reported with their quiet time.

Uncertainty: block bootstrap over 30 s blocks for every rate (95 % interval),
Theil-Sen slopes with 95 % intervals for drift, a paired sign-flip
permutation over matched 1-min bins for run 1 vs run 2.

    python analysis/flash_eog.py
writes runs/state_of_flash/flash_eog.json and flash_eog.png
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blinkmask  # noqa: E402
from analysis import state_of_flash as sof  # noqa: E402

OUT = os.path.join(ROOT, "runs", "state_of_flash")
SESSIONS = [("vr_planes9", "run 1 (17:57)"), ("vr_planes10", "run 2 (19:50)"),
            ("vr_planes8", "reference day (10-07)")]
AMP_FLOOR, AMP_HI, K_SD = 25.0, 40.0, 3.0
WIN_S, BLOCK_S, NBOOT, SEED = 300.0, 30.0, 2000, 7
ART_UV, MOVE_PAD_S = 150.0, 1.0
LONG_MS, BURST_S = 400.0, 1.0
H_FLOOR, SACCADE_S = 20.0, 0.6
C_LIT, C_DARK, C_INK, C_GRID = "#2a78d6", "#eb6834", "#3a3a37", "#ecebe5"
C_SESS = {"vr_planes9": "#1baf7a", "vr_planes10": "#4a3aa7", "vr_planes8": "#8a8a85",
          "vr_vision1": "#d9a400", "vr_planes11": "#c8102e"}
TAG = ""        # suffix for the output files when the sessions are given on the command line
PALETTE = ["#1baf7a", "#4a3aa7", "#d9a400", "#c8102e", "#0e7c86", "#8a8a85"]
STAGE_LABEL = {"calib": "white calib 12 Hz", "calib_blue_sweep": "blue sweep", "calib_red": "red calib",
               "calib_green": "green calib", "calib_blue": "blue calib", "plane_blue": "blue plane",
               "plane_red": "red plane", "plane_green": "green plane"}
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- helpers
def robust_sd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def clean(o):
    """JSON-safe: numpy -> python, NaN -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def stages_of(d, first):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        L = pd.read_csv(p, usecols=[0])
        lt = L.iloc[:, 0].to_numpy(float)
        if len(lt) < 10:
            continue
        out.append(dict(kind=kind, label=STAGE_LABEL.get(kind, kind), t0=float(lt[0]), t1=float(lt[-1]),
                        start_min=(lt[0] - first) / 60.0, end_min=(lt[-1] - first) / 60.0))
    out.sort(key=lambda s: s["t0"])
    return out


def transitions(state, t, fs):
    """Times of lit->dark and dark->lit edges on screen (runs >= 0.5 s on both sides)."""
    lit = np.isin(state, [sof.FLICKER, sof.TAGGED]).astype(np.int8)
    dark = (state == sof.DARK).astype(np.int8)
    code = lit - dark                                   # +1 lit, -1 dark, 0 other
    edges = np.flatnonzero(np.diff(code) != 0) + 1
    runs = np.diff(np.concatenate(([0], edges, [len(code)])))
    min_run = int(0.5 * fs)
    to_dark, to_lit = [], []
    for j, e in enumerate(edges):
        before, after = code[e - 1], code[e]
        if runs[j] < min_run or runs[j + 1] < min_run:
            continue
        if before == 1 and after == -1:
            to_dark.append(t[e])
        elif before == -1 and after == 1:
            to_lit.append(t[e])
    return np.array(to_dark), np.array(to_lit)


# ----------------------------------------------------------------- signals
def prepare(d):
    t, X, names, fs = sof.load_eeg(d)
    f0, _ = sof.stim_freq(d)
    state, _, first, last = sof.timeline(d, t, f0)
    i7, i8 = names.index("AF7"), names.index("AF8")
    far_idx = [names.index(n) for n in ("TP9", "TP10", "AUX")]
    raw = X - np.median(X, axis=0)
    railed = ((X <= 1.0) | (X >= 1649.0)).any(axis=1)
    sos_b = butter(2, [0.5, 8.0], "bandpass", fs=fs, output="sos")
    sos_h = butter(2, [0.5, 15.0], "bandpass", fs=fs, output="sos")
    sos_w = butter(2, [1.0, 40.0], "bandpass", fs=fs, output="sos")
    Bb = sosfiltfilt(sos_b, raw, axis=0)
    B40 = sosfiltfilt(sos_w, raw, axis=0)
    far = Bb[:, far_idx]
    rsig = -np.median(far, axis=1)                      # blink through the reference, positive up
    af7, af8 = Bb[:, i7], Bb[:, i8]
    afm = (af7 + af8) / 2.0
    dd = sosfiltfilt(sos_h, raw[:, i7] - raw[:, i8])
    k = int(2 * MOVE_PAD_S * fs) | 1
    big = np.stack([maximum_filter1d(np.abs(B40[:, c]), k) > ART_UV for c in range(X.shape[1])], 1).sum(1) >= 2
    move = big | maximum_filter1d(railed.astype(np.uint8), k).astype(bool)
    bm = blinkmask.blink_mask(X, names, fs, 100.0)
    bm_edges = t[np.flatnonzero(np.diff(bm.astype(int)) == 1) + 1]
    return dict(t=t, fs=fs, names=names, f0=f0, state=state, first=first, last=last, raw=raw,
                i7=i7, i8=i8, far=far, rsig=rsig, af7=af7, af8=af8, afm=afm, dd=dd, move=move,
                bm_edges=bm_edges, stages=stages_of(d, first), minutes=(t[-1] - t[0]) / 60.0)


def width_at(x, pk, frac, fs, max_s=0.75):
    """Width (ms) of each pulse at frac x its height above zero, walking out from the peak;
    a side that is still above the level after max_s is cut there."""
    n_max = int(max_s * fs)
    out = np.zeros(len(pk))
    for k, i in enumerate(pk):
        lvl = frac * x[i]
        l = i
        while l > 0 and i - l < n_max and x[l - 1] > lvl:
            l -= 1
        r = i
        while r < len(x) - 1 and r - i < n_max and x[r + 1] > lvl:
            r += 1
        out[k] = (r - l) / fs * 1000.0
    return out


def pulses(sig, fs, valid, floor_uv, k_sd=K_SD, fwhm_ms=(50.0, 500.0)):
    rsd = robust_sd(sig[valid])
    thr = max(k_sd * rsd, floor_uv)
    x = np.where(valid, sig, 0.0)
    pk, pr = find_peaks(x, height=thr, distance=int(0.25 * fs))
    pad = int(0.3 * fs)
    keep = np.array([valid[max(0, i - pad):i + pad].all() for i in pk], bool) if len(pk) else np.zeros(0, bool)
    fwhm = width_at(x, pk, 0.5, fs) if len(pk) else np.zeros(0)
    keep &= (fwhm >= fwhm_ms[0]) & (fwhm <= fwhm_ms[1])
    pk, fwhm, amp = pk[keep], fwhm[keep], pr["peak_heights"][keep]
    base = width_at(x, pk, 0.2, fs) if len(pk) else np.zeros(0)       # the pulse's own base, 20 % of its height
    return pk, amp, fwhm, base, thr, rsd


def detect_all(S):
    t, fs = S["t"], S["fs"]
    valid = ~S["move"]
    on = (t >= S["first"]) & (t <= S["last"])
    out = {}
    # reference-projected blink and its mirror
    for name, sign in (("ref", 1.0), ("ref_mirror", -1.0)):
        pk, amp, fwhm, base, thr, rsd = pulses(sign * S["rsig"], fs, valid, AMP_FLOOR)
        agree = np.array([(-sign * S["far"][i] >= 0.5 * sign * S["rsig"][i]).all() for i in pk], bool) if len(pk) else np.zeros(0, bool)
        pad = int(0.1 * fs)
        a7 = np.array([S["af7"][max(0, i - pad):i + pad].max() for i in pk]) if len(pk) else np.zeros(0)
        a8 = np.array([S["af8"][max(0, i - pad):i + pad].max() for i in pk]) if len(pk) else np.zeros(0)
        df = pd.DataFrame(dict(t=t[pk], idx=pk, amp=amp, fwhm_ms=fwhm, base_ms=base, agree=agree, af7=a7, af8=a8))
        df = df[df["agree"]].reset_index(drop=True)
        df["min"] = (df["t"] - S["first"]) / 60.0
        df["state"] = S["state"][df["idx"].to_numpy()] if len(df) else np.zeros(0, np.int8)
        ibi_prev = np.diff(df["t"].to_numpy(), prepend=-np.inf)
        ibi_next = np.diff(df["t"].to_numpy(), append=np.inf)
        df["burst"] = (ibi_prev < BURST_S) | (ibi_next < BURST_S)
        df["long"] = df["base_ms"] > LONG_MS
        df["long_fwhm"] = df["fwhm_ms"] > 150.0          # the half-height convention for a long closure
        out[name] = df
        out[name + "_thr"] = dict(threshold_uv=thr, robust_sd_uv=rsd)
    # the mirror floor is noise, not the band-pass undershoot of a real blink: mirror events within
    # 0.5 s of a blink are dropped and mirror rates are taken per minute of blink-free time
    bt, mt = out["ref"]["t"].to_numpy(), out["ref_mirror"]["t"].to_numpy()
    if len(bt) and len(mt):
        j = np.searchsorted(bt, mt)
        dl = np.abs(mt - bt[np.clip(j - 1, 0, len(bt) - 1)])
        dr = np.abs(bt[np.clip(j, 0, len(bt) - 1)] - mt)
        out["ref_mirror"] = out["ref_mirror"][np.minimum(dl, dr) >= 0.5].reset_index(drop=True)
    near = np.zeros(len(t), bool)
    w = int(0.5 * fs)
    for i in out["ref"]["idx"].to_numpy():
        near[max(0, i - w):i + w] = True
    out["near_blink"] = near
    # frontal detector (as specified) and its mirror
    r7, r8 = robust_sd(S["af7"][valid & on]), robust_sd(S["af8"][valid & on])
    for name, sign in (("frontal", 1.0), ("frontal_mirror", -1.0)):
        pk, amp, fwhm, base, thr, rsd = pulses(sign * S["afm"], fs, valid, 0.0)
        pad = int(0.1 * fs)
        both = np.array([(sign * S["af7"][max(0, i - pad):i + pad]).max() >= K_SD * r7 and
                         (sign * S["af8"][max(0, i - pad):i + pad]).max() >= K_SD * r8 for i in pk], bool) if len(pk) else np.zeros(0, bool)
        df = pd.DataFrame(dict(t=t[pk][both], idx=pk[both], amp=amp[both], fwhm_ms=fwhm[both], base_ms=base[both]))
        df["min"] = (df["t"] - S["first"]) / 60.0
        out[name] = df
        out[name + "_thr"] = dict(threshold_uv=thr, robust_sd_af7_uv=r7, robust_sd_af8_uv=r8)
    # horizontal eye movements on AF7 - AF8
    hthr = max(K_SD * robust_sd(S["dd"][valid & on]), H_FLOOR)
    hk, hp = find_peaks(np.abs(np.where(valid, S["dd"], 0.0)), height=hthr, distance=int(0.15 * fs))
    bt = out["ref"]["t"].to_numpy()
    near_blink = np.zeros(len(hk), bool)
    if len(bt) and len(hk):
        j = np.searchsorted(bt, t[hk])
        dl = np.abs(t[hk] - bt[np.clip(j - 1, 0, len(bt) - 1)])
        dr = np.abs(bt[np.clip(j, 0, len(bt) - 1)] - t[hk])
        near_blink = np.minimum(dl, dr) < 0.15
    hk, hamp = hk[~near_blink], hp["peak_heights"][~near_blink]
    to_dark, to_lit = transitions(S["state"], t, fs)
    trans = np.sort(np.concatenate([to_dark, to_lit]))
    after = np.zeros(len(hk), bool)
    if len(trans) and len(hk):
        j = np.searchsorted(trans, t[hk], side="right") - 1
        after = (j >= 0) & (t[hk] - trans[np.clip(j, 0, len(trans) - 1)] < SACCADE_S)
    out["horiz"] = pd.DataFrame(dict(t=t[hk], amp=hamp, after_change=after))
    out["horiz_thr"] = dict(threshold_uv=hthr)
    out["to_dark"], out["to_lit"] = to_dark, to_lit
    return out


# ----------------------------------------------------------------- window statistics
def block_table(S, mask, extra_times, far=None):
    """Per 30 s block inside mask: valid seconds, counts of each event list, lit/dark seconds,
    and the valid seconds outside `far` (the blink-free time for the mirror floor)."""
    t, fs = S["t"], S["fs"]
    ok = mask & ~S["move"]
    if not ok.any():
        return None
    b = np.floor((t - S["first"]) / BLOCK_S).astype(int)
    ids = np.unique(b[ok])
    sec = np.array([(ok & (b == i)).sum() / fs for i in ids])
    okf = ok & ~far if far is not None else ok
    sec_far = np.array([(okf & (b == i)).sum() / fs for i in ids])
    lit = np.isin(S["state"], [sof.FLICKER, sof.TAGGED])
    dark = S["state"] == sof.DARK
    lit_sec = np.array([(ok & lit & (b == i)).sum() / fs for i in ids])
    dark_sec = np.array([(ok & dark & (b == i)).sum() / fs for i in ids])
    counts = {}
    for key, ev_t in extra_times.items():
        if len(ev_t):
            ei = np.searchsorted(t, ev_t)
            ei = np.clip(ei, 0, len(t) - 1)
            inside = ok[ei]
            eb = b[ei]
            counts[key] = np.array([((eb == i) & inside).sum() for i in ids])
        else:
            counts[key] = np.zeros(len(ids), int)
    return dict(ids=ids, sec=sec, sec_far=sec_far, lit_sec=lit_sec, dark_sec=dark_sec, counts=counts)


def boot_rate(sec, n, nboot=NBOOT):
    """Rate per minute with a 95 % block-bootstrap interval."""
    tot = sec.sum()
    if tot <= 0:
        return float("nan"), [float("nan")] * 2
    rate = n.sum() / tot * 60.0
    nb = len(sec)
    if nb < 2:
        return float(rate), [float("nan")] * 2
    idx = rng.integers(0, nb, (nboot, nb))
    r = n[idx].sum(1) / np.maximum(sec[idx].sum(1), 1e-9) * 60.0
    return float(rate), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_rate_samples(sec, n, nboot=NBOOT):
    nb = len(sec)
    if nb < 2 or sec.sum() <= 0:
        return None
    idx = rng.integers(0, nb, (nboot, nb))
    return n[idx].sum(1) / np.maximum(sec[idx].sum(1), 1e-9) * 60.0


def slow_series(S, mask):
    """0.1-0.5 Hz power (uV^2) per 20 s Hann segment (10 s hop) inside mask: (minutes, AF7/AF8 mean, AF7-AF8)."""
    t, fs = S["t"], S["fs"]
    ok = mask & ~S["move"]
    n = int(20 * fs)
    hop = int(10 * fs)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= 0.1) & (f <= 0.5)
    w = np.hanning(n)
    norm = (w ** 2).sum() * fs
    mins, vals_af, vals_dd = [], [], []
    idx = np.flatnonzero(ok)
    if len(idx) < n:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    ramp = np.arange(n)
    for s0 in range(idx[0], idx[-1] - n, hop):
        if not ok[s0:s0 + n].all():
            continue
        segs = []
        for x in (S["raw"][s0:s0 + n, S["i7"]], S["raw"][s0:s0 + n, S["i8"]], S["raw"][s0:s0 + n, S["i7"]] - S["raw"][s0:s0 + n, S["i8"]]):
            xx = x - np.polyval(np.polyfit(ramp, x, 1), ramp)
            P = np.abs(np.fft.rfft(xx * w)) ** 2 / norm
            segs.append(P[band].sum() * (f[1] - f[0]) * 2)
        mins.append((t[s0 + n // 2] - S["first"]) / 60.0)
        vals_af.append((segs[0] + segs[1]) / 2)
        vals_dd.append(segs[2])
    return np.array(mins), np.array(vals_af), np.array(vals_dd)


def slow_power(S, mask):
    """Median log10 0.1-0.5 Hz power over the segments inside mask, AF7/AF8 mean and AF7-AF8."""
    mins, af, dd = slow_series(S, mask)
    if not len(af):
        return float("nan"), float("nan"), 0
    return float(np.log10(np.median(af))), float(np.log10(np.median(dd))), int(len(af))


def window_stats(S, D, mask, label):
    """Everything for one time window (a boolean mask over samples)."""
    bt = D["ref"]["t"].to_numpy()
    ev = dict(blink=bt, mirror=D["ref_mirror"]["t"].to_numpy(),
              hi=D["ref"]["t"][D["ref"]["amp"] >= AMP_HI].to_numpy(),
              hi_mirror=D["ref_mirror"]["t"][D["ref_mirror"]["amp"] >= AMP_HI].to_numpy(),
              frontal=D["frontal"]["t"].to_numpy(), frontal_mirror=D["frontal_mirror"]["t"].to_numpy(),
              horiz=D["horiz"]["t"].to_numpy(), horiz_free=D["horiz"]["t"][~D["horiz"]["after_change"]].to_numpy(),
              bm100=S["bm_edges"])
    B = block_table(S, mask, ev, far=D["near_blink"])
    res = dict(label=label)
    if B is None:
        res["minutes_valid"] = 0.0
        return res
    res["minutes_valid"] = float(B["sec"].sum() / 60.0)
    res["minutes_total"] = float(mask.sum() / S["fs"] / 60.0)
    res["minutes_blink_free"] = float(B["sec_far"].sum() / 60.0)
    for key in ("blink", "mirror", "hi", "hi_mirror", "frontal", "frontal_mirror", "horiz", "horiz_free", "bm100"):
        sec = B["sec_far"] if key in ("mirror", "hi_mirror") else B["sec"]
        r, ci = boot_rate(sec, B["counts"][key])
        res[key + "_per_min"] = r
        res[key + "_ci"] = ci
        res["n_" + key] = int(B["counts"][key].sum())
    # net = blink - mirror floor, the same block draws for both
    nb = len(B["ids"])
    res["net_per_min"] = res["blink_per_min"] - res["mirror_per_min"]
    if nb >= 2 and B["sec_far"].sum() > 0:
        idx = rng.integers(0, nb, (NBOOT, nb))
        rb = B["counts"]["blink"][idx].sum(1) / np.maximum(B["sec"][idx].sum(1), 1e-9) * 60.0
        rm = B["counts"]["mirror"][idx].sum(1) / np.maximum(B["sec_far"][idx].sum(1), 1e-9) * 60.0
        res["net_ci"] = [float(np.percentile(rb - rm, 2.5)), float(np.percentile(rb - rm, 97.5))]
    else:
        res["net_ci"] = [float("nan")] * 2
    # blink shape inside the window
    t = S["t"]
    ok = mask & ~S["move"]
    ei = np.clip(np.searchsorted(t, bt), 0, len(t) - 1)
    inw = ok[ei]
    sub = D["ref"][inw]
    res["n_blinks"] = int(len(sub))
    if len(sub) >= 5:
        for col, nm in (("amp", "amp_uv"), ("fwhm_ms", "fwhm_ms"), ("base_ms", "base_ms")):
            v = sub[col].to_numpy()
            bs = rng.choice(v, (NBOOT, len(v))).__array__()
            med = np.median(bs, axis=1)
            res[nm + "_median"] = float(np.median(v))
            res[nm + "_median_ci"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
            res[nm + "_iqr"] = [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
        for col, nm in (("long", "long_share"), ("long_fwhm", "fwhm_over150_share"), ("burst", "burst_share")):
            v = sub[col].to_numpy().astype(float)
            bs = rng.choice(v, (NBOOT, len(v))).mean(1)
            res[nm] = float(v.mean())
            res[nm + "_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["af7_at_blink_uv"] = float(np.median(sub["af7"]))
        res["af8_at_blink_uv"] = float(np.median(sub["af8"]))
    # lit vs dark inside the window
    lit_n = np.array([((D["ref"]["state"].isin([sof.FLICKER, sof.TAGGED])) & inw & (np.floor((bt - S["first"]) / BLOCK_S).astype(int) == i)).sum() for i in B["ids"]]) if len(bt) else np.zeros(len(B["ids"]), int)
    dark_n = np.array([((D["ref"]["state"] == sof.DARK) & inw & (np.floor((bt - S["first"]) / BLOCK_S).astype(int) == i)).sum() for i in B["ids"]]) if len(bt) else np.zeros(len(B["ids"]), int)
    if B["lit_sec"].sum() > 60 and B["dark_sec"].sum() > 60:
        rl, cil = boot_rate(B["lit_sec"], lit_n)
        rd, cid = boot_rate(B["dark_sec"], dark_n)
        nb = len(B["ids"])
        idx = rng.integers(0, nb, (NBOOT, nb))
        dl = lit_n[idx].sum(1) / np.maximum(B["lit_sec"][idx].sum(1), 1e-9) * 60.0
        dk = dark_n[idx].sum(1) / np.maximum(B["dark_sec"][idx].sum(1), 1e-9) * 60.0
        res["lit_per_min"], res["lit_ci"], res["dark_per_min"], res["dark_ci"] = rl, cil, rd, cid
        res["dark_minus_lit"] = rd - rl
        res["dark_minus_lit_ci"] = [float(np.percentile(dk - dl, 2.5)), float(np.percentile(dk - dl, 97.5))]
        res["lit_minutes"], res["dark_minutes"] = float(B["lit_sec"].sum() / 60), float(B["dark_sec"].sum() / 60)
    sp_af, sp_dd, nseg = slow_power(S, mask)
    res["slow_af_log10_uv2"], res["slow_dd_log10_uv2"], res["slow_segments"] = sp_af, sp_dd, nseg
    return res


def minute_bins(S, D, key="blink"):
    """Per 1-min bin on task: valid seconds and counts -> rate; bins with < 30 s valid dropped."""
    t, fs = S["t"], S["fs"]
    on = (t >= S["first"]) & (t <= S["last"]) & ~S["move"]
    if key == "mirror":
        on = on & ~D["near_blink"]
    m = np.floor((t - S["first"]) / 60.0).astype(int)
    ids = np.unique(m[on])
    sec = np.array([(on & (m == i)).sum() / fs for i in ids])
    if key == "horiz_free":
        ev = D["horiz"]["t"][~D["horiz"]["after_change"]].to_numpy()
    else:
        ev = (D["ref"] if key == "blink" else D["ref_mirror"])["t"].to_numpy()
    ei = np.clip(np.searchsorted(t, ev), 0, len(t) - 1)
    cnt = np.array([((m[ei] == i) & on[ei]).sum() for i in ids])
    keep = sec >= 30
    return ids[keep] + 0.5, sec[keep], cnt[keep], cnt[keep] / sec[keep] * 60.0


def drift(S, D):
    """Theil-Sen slopes (per 10 min) with 95 % intervals over minutes on task."""
    res = {}
    mid, sec, cnt, rate = minute_bins(S, D)
    if len(mid) >= 6:
        sl, ic, lo, hi = stats.theilslopes(rate, mid, alpha=0.95)
        tau = stats.kendalltau(mid, rate)
        res["blink_rate_per_10min"] = dict(slope=float(sl * 10), ci=[float(lo * 10), float(hi * 10)],
                                           kendall_tau=float(tau.statistic), p=float(tau.pvalue), n_minutes=int(len(mid)))
        mid2, sec2, cnt2, rate2 = minute_bins(S, D, "mirror")
        net = rate - np.interp(mid, mid2, rate2) if len(mid2) else rate
        sl, ic, lo, hi = stats.theilslopes(net, mid, alpha=0.95)
        res["net_rate_per_10min"] = dict(slope=float(sl * 10), ci=[float(lo * 10), float(hi * 10)])
        half = mid < mid.max() / 2
        res["first_half_per_min"] = float(cnt[half].sum() / sec[half].sum() * 60) if sec[half].sum() else float("nan")
        res["second_half_per_min"] = float(cnt[~half].sum() / sec[~half].sum() * 60) if sec[~half].sum() else float("nan")
    hm, hs, hc, hr = minute_bins(S, D, "horiz_free")
    if len(hm) >= 6:
        sl, ic, lo, hi = stats.theilslopes(hr, hm, alpha=0.95)
        tau = stats.kendalltau(hm, hr)
        res["horizontal_free_per_10min"] = dict(slope=float(sl * 10), ci=[float(lo * 10), float(hi * 10)], p=float(tau.pvalue), n_minutes=int(len(hm)))
    t = S["t"]
    sm, saf, sdd = slow_series(S, (t >= S["first"]) & (t <= S["last"]))
    if len(sm) >= 10:
        for v, nm in ((saf, "slow_af_log10_per_10min"), (sdd, "slow_dd_log10_per_10min")):
            sl, ic, lo, hi = stats.theilslopes(np.log10(v), sm, alpha=0.95)
            tau = stats.kendalltau(sm, np.log10(v))
            res[nm] = dict(slope=float(sl * 10), ci=[float(lo * 10), float(hi * 10)], p=float(tau.pvalue), n_segments=int(len(sm)))
    df = D["ref"][(D["ref"]["min"] >= 0) & (D["ref"]["t"] <= S["last"])]
    if len(df) >= 20:
        for col, nm in (("amp", "amp_uv_per_10min"), ("base_ms", "base_ms_per_10min"), ("fwhm_ms", "fwhm_ms_per_10min")):
            sl, ic, lo, hi = stats.theilslopes(df[col].to_numpy(), df["min"].to_numpy(), alpha=0.95)
            res[nm] = dict(slope=float(sl * 10), ci=[float(lo * 10), float(hi * 10)], n=int(len(df)))
        half = df["min"] < df["min"].max() / 2
        for col, nm in (("long", "long_share"), ("long_fwhm", "fwhm_over150_share"), ("burst", "burst_share")):
            a, b = df[col][half].to_numpy().astype(float), df[col][~half].to_numpy().astype(float)
            if len(a) >= 5 and len(b) >= 5:
                bs = rng.choice(b, (NBOOT, len(b))).mean(1) - rng.choice(a, (NBOOT, len(a))).mean(1)
                res[nm + "_second_minus_first_half"] = dict(first=float(a.mean()), second=float(b.mean()), diff=float(b.mean() - a.mean()),
                                                            ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=[int(len(a)), int(len(b))])
    return res


def psth(S, D, onsets, lo=-3.0, hi=6.0, step=0.5):
    """Blink rate (per min) in bins of time relative to screen transitions."""
    bt = D["ref"]["t"].to_numpy()
    edges = np.arange(lo, hi + step / 2, step)
    if len(onsets) == 0 or len(bt) == 0:
        return edges, np.full(len(edges) - 1, np.nan), np.full((2, len(edges) - 1), np.nan)
    per_onset = []
    for o in onsets:
        r = bt - o
        per_onset.append(np.histogram(r[(r >= lo) & (r < hi)], edges)[0])
    per_onset = np.array(per_onset)
    rate = per_onset.sum(0) / (len(onsets) * step) * 60.0
    idx = rng.integers(0, len(onsets), (NBOOT, len(onsets)))
    bs = per_onset[idx].sum(1) / (len(onsets) * step) * 60.0
    return edges, rate, np.percentile(bs, [2.5, 97.5], axis=0)


def release(S, D, onsets, early=(0.0, 1.0), late=(1.0, 4.0)):
    bt = D["ref"]["t"].to_numpy()
    if len(onsets) == 0:
        return None
    ne = np.array([((bt >= o + early[0]) & (bt < o + early[1])).sum() for o in onsets])
    nl = np.array([((bt >= o + late[0]) & (bt < o + late[1])).sum() for o in onsets])
    re_ = ne.sum() / (len(onsets) * (early[1] - early[0])) * 60
    rl = nl.sum() / (len(onsets) * (late[1] - late[0])) * 60
    idx = rng.integers(0, len(onsets), (NBOOT, len(onsets)))
    d = ne[idx].sum(1) / (len(onsets) * (early[1] - early[0])) * 60 - nl[idx].sum(1) / (len(onsets) * (late[1] - late[0])) * 60
    return dict(n_transitions=int(len(onsets)), first_second_per_min=float(re_), seconds_1_to_4_per_min=float(rl),
                diff=float(re_ - rl), diff_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


# ----------------------------------------------------------------- per session
def analyse(name):
    d = os.path.join(ROOT, "runs", name)
    S = prepare(d)
    D = detect_all(S)
    t = S["t"]
    on = (t >= S["first"]) & (t <= S["last"])
    res = dict(session=name, minutes=round(S["minutes"], 1), fs=round(S["fs"], 2), f0=S["f0"],
               stimulation_min=round((S["last"] - S["first"]) / 60, 2),
               movement_share_on_task=float(S["move"][on].mean()),
               thresholds=dict(reference=D["ref_thr"], frontal=D["frontal_thr"], horizontal=D["horiz_thr"]),
               stages=S["stages"])
    # whole task, pre, idle, post
    pre = (t >= S["first"] - 180) & (t < S["first"])
    res["windows"] = dict(on_task=window_stats(S, D, on, "on task"),
                          pre_last3min=window_stats(S, D, pre, "last 3 min before the first flicker"),
                          idle=window_stats(S, D, S["state"] == sof.IDLE, "between stages"),
                          post=window_stats(S, D, t > S["last"], "after the last flicker"))
    # 5-min windows on task
    res["five_min"] = []
    k = 0
    while k * WIN_S < S["last"] - S["first"] - 150:
        m = on & (t >= S["first"] + k * WIN_S) & (t < S["first"] + (k + 1) * WIN_S)
        w = window_stats(S, D, m, f"{k*5}-{(k+1)*5} min")
        w["start_min"], w["end_min"] = k * 5, min((k + 1) * 5, (S["last"] - S["first"]) / 60)
        res["five_min"].append(w)
        k += 1
    # stages
    res["stage_stats"] = []
    for st in S["stages"]:
        m = (t >= st["t0"]) & (t <= st["t1"])
        w = window_stats(S, D, m, st["label"])
        w.update(kind=st["kind"], start_min=st["start_min"], end_min=st["end_min"])
        res["stage_stats"].append(w)
    # lit vs dark, by frequency family
    tagged_kinds = {s["kind"] for s in S["stages"] if s["kind"] in ("calib_blue_sweep",)} | ({"plane_blue"} if name != "vr_planes8" else set())
    m12 = on.copy()
    for st in S["stages"]:
        if st["kind"] in tagged_kinds:
            m12 &= ~((t >= st["t0"]) & (t <= st["t1"]))
    res["lit_dark_12hz_stages"] = window_stats(S, D, m12, "12 Hz stages")
    res["lit_dark_tagged_stages"] = window_stats(S, D, on & ~m12, "9 Hz / sweep stages") if (on & ~m12).any() else None
    # transitions
    to_dark, to_lit = D["to_dark"], D["to_lit"]
    edges, r_d, ci_d = psth(S, D, to_dark)
    edges, r_l, ci_l = psth(S, D, to_lit)
    res["psth"] = dict(bin_edges_s=edges.tolist(), to_dark_per_min=r_d.tolist(), to_dark_ci=ci_d.tolist(),
                       to_lit_per_min=r_l.tolist(), to_lit_ci=ci_l.tolist(), n_to_dark=int(len(to_dark)), n_to_lit=int(len(to_lit)))
    res["release_after_dark_onset"] = release(S, D, to_dark)
    res["release_after_lit_onset"] = release(S, D, to_lit)
    # saccade to the new cell: horizontal events in the 0.6 s after a change vs the free rate
    H = D["horiz"]
    hon = H[(H["t"] >= S["first"]) & (H["t"] <= S["last"])]
    trans = np.sort(np.concatenate([to_dark, to_lit]))
    valid_min = res["windows"]["on_task"].get("minutes_valid", 0.0)
    res["horizontal"] = dict(n_transitions=int(len(trans)),
                             within_0_6s_after_change_per_change=float(hon["after_change"].sum() / max(len(trans), 1)),
                             free_per_min=float((~hon["after_change"]).sum() / max(valid_min - len(trans) * SACCADE_S / 60, 1e-9)),
                             amp_uv_median=float(np.median(hon["amp"])) if len(hon) else float("nan"))
    res["drift"] = drift(S, D)
    return res, S, D


def matched(res1, res2, S1, D1, S2, D2):
    """Run 2 - run 1 at matched minutes on task (5-min windows and paired 1-min bins) and matched stages."""
    out = dict(five_min=[])
    for w1, w2 in zip(res1["five_min"], res2["five_min"]):
        d = dict(window=w1["label"], run1=w1.get("blink_per_min"), run2=w2.get("blink_per_min"),
                 run1_net=w1.get("net_per_min"), run2_net=w2.get("net_per_min"))
        # independent block bootstraps for the difference
        B1 = _blocks(S1, D1, w1["start_min"], w1["end_min"])
        B2 = _blocks(S2, D2, w2["start_min"], w2["end_min"])
        if B1 is not None and B2 is not None:
            r1 = boot_rate_samples(B1[0], B1[1]); r2 = boot_rate_samples(B2[0], B2[1])
            if r1 is not None and r2 is not None:
                d["diff"] = d["run2"] - d["run1"]
                d["diff_ci"] = [float(np.percentile(r2 - r1, 2.5)), float(np.percentile(r2 - r1, 97.5))]
        out["five_min"].append(d)
    # paired 1-min bins, sign-flip permutation
    m1, s1, c1, r1 = minute_bins(S1, D1)
    m2, s2, c2, r2 = minute_bins(S2, D2)
    common = np.intersect1d(m1, m2)
    a = np.array([r1[np.flatnonzero(m1 == c)[0]] for c in common])
    b = np.array([r2[np.flatnonzero(m2 == c)[0]] for c in common])
    dlt = b - a
    obs = dlt.mean()
    flips = rng.choice([-1.0, 1.0], (20000, len(dlt)))
    null = (flips * dlt).mean(1)
    out["paired_minutes"] = dict(n_minutes=int(len(dlt)), run1_mean=float(a.mean()), run2_mean=float(b.mean()),
                                 diff=float(obs), p_permutation=float((np.abs(null) >= abs(obs)).mean()),
                                 diff_ci=[float(np.percentile(rng.choice(dlt, (NBOOT, len(dlt))).mean(1), q)) for q in (2.5, 97.5)])
    # matched stages: same kind, equal duration from the stage start
    out["stages"] = []
    for st1 in res1["stage_stats"]:
        st2 = next((s for s in res2["stage_stats"] if s["kind"] == st1["kind"]), None)
        if st2 is None:
            continue
        dur = min(st1["end_min"] - st1["start_min"], st2["end_min"] - st2["start_min"])
        B1 = _blocks(S1, D1, st1["start_min"], st1["start_min"] + dur)
        B2 = _blocks(S2, D2, st2["start_min"], st2["start_min"] + dur)
        if B1 is None or B2 is None:
            continue
        ra, cia = boot_rate(B1[0], B1[1]); rb, cib = boot_rate(B2[0], B2[1])
        q1 = boot_rate_samples(B1[0], B1[1]); q2 = boot_rate_samples(B2[0], B2[1])
        out["stages"].append(dict(stage=st1["label"], matched_minutes=float(dur), run1=ra, run1_ci=cia, run2=rb, run2_ci=cib,
                                  diff=rb - ra, diff_ci=[float(np.percentile(q2 - q1, 2.5)), float(np.percentile(q2 - q1, 97.5))] if q1 is not None and q2 is not None else None,
                                  run1_start_min=st1["start_min"], run2_start_min=st2["start_min"]))
    return out


def _blocks(S, D, start_min, end_min):
    t = S["t"]
    m = (t >= S["first"] + start_min * 60) & (t < S["first"] + end_min * 60) & (t >= S["first"]) & (t <= S["last"])
    B = block_table(S, m, dict(blink=D["ref"]["t"].to_numpy()))
    return None if B is None else (B["sec"], B["counts"]["blink"])


# ----------------------------------------------------------------- figure
def figure(results, store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK, "xtick.color": C_INK,
                         "ytick.color": C_INK, "text.color": C_INK, "axes.titlesize": 8.5, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[0.16, 1.15, 1, 1, 0.9], hspace=0.55, wspace=0.28)

    def style(ax):
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # stage strip (run 2 timing; run 1 shares the first 14.5 min, then its red and green planes)
    axs = fig.add_subplot(gs[0, :2])
    for st in results["vr_planes10"]["stages"]:
        c = C_LIT if st["kind"] in ("calib_blue_sweep", "plane_blue") else "#d4d3cc"
        axs.barh(0, st["end_min"] - st["start_min"], left=st["start_min"], height=0.8, color=c, alpha=0.55, lw=0)
        short = st["end_min"] - st["start_min"] < 1.5
        axs.text((st["start_min"] + st["end_min"]) / 2, 0, st["label"].replace(" 12 Hz", "").replace(" calib", ""),
                 ha="center", va="center", fontsize=5.5 if short else 6.5, rotation=90 if short else 0)
    axs.set_xlim(-0.5, 31)
    axs.set_ylim(-0.6, 0.6)
    axs.set_yticks([])
    axs.set_xticks([])
    for sp in ("top", "right", "left", "bottom"):
        axs.spines[sp].set_visible(False)
    axs.set_title("Run 2 stages (blue = 9 Hz / sweep stages; run 1 has the same order, with 7 min of blue plane and a 21 min task)", loc="left", fontsize=7.5)

    # (a) blink rate per minute over minutes on task
    ax = fig.add_subplot(gs[1, :2])
    for name, lab in SESSIONS:
        S, D = store[name]
        mid, sec, cnt, rate = minute_bins(S, D)
        mid2, sec2, cnt2, rate2 = minute_bins(S, D, "mirror")
        c = C_SESS[name]
        ax.plot(mid, rate, "-", color=c, lw=1.0, alpha=0.45)
        ax.plot(mid, rate, "o", color=c, ms=3.5, label=f"{lab}: blinks per minute")
        ax.plot(mid2, rate2, ":", color=c, lw=1.0, alpha=0.8)
        dr = results[name]["drift"].get("blink_rate_per_10min")
        if dr and len(mid) >= 6:
            sl, ic = stats.theilslopes(rate, mid)[:2]
            xx = np.array([mid.min(), mid.max()])
            ax.plot(xx, ic + sl * xx, "-", color=c, lw=2.2)
    ax.plot([], [], ":", color=C_INK, label="dotted: mirror-polarity false-alarm floor")
    ax.plot([], [], "-", color=C_INK, lw=2.2, label="thick: Theil-Sen trend")
    for st in results["vr_planes10"]["stages"]:
        ax.axvline(st["start_min"], color="#d4d3cc", lw=0.6)
    ymax = max(max(minute_bins(store[n][0], store[n][1])[3].max() for n, _ in SESSIONS) * 1.3, 10)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, 31)
    ax.set_xlabel("minutes since the first flicker (stage lines: run 2)")
    ax.set_ylabel("blinks per minute (reference-projected, >= 25 uV)")
    ax.set_title("Blink rate through each session, 1-min bins", loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    style(ax)

    # (b) lit vs dark vs rest per session
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(len(SESSIONS))
    for j, (name, lab) in enumerate(SESSIONS):
        w = results[name]["windows"]["on_task"]
        for off, key, c, lbl in ((-0.22, "lit", C_LIT, "lit / flickering cell"), (0.0, "dark", C_DARK, "black cell / OFF block")):
            r, ci = w.get(key + "_per_min", np.nan), w.get(key + "_ci", [np.nan, np.nan])
            ax.bar(j + off, r, 0.2, color=c, label=lbl if j == 0 else None)
            ax.errorbar(j + off, r, yerr=[[r - ci[0]], [ci[1] - r]], color=C_INK, capsize=2, lw=0.8)
        wi = results[name]["windows"]["idle"]
        r, ci = wi.get("blink_per_min", np.nan), wi.get("blink_ci", [np.nan, np.nan])
        ax.bar(j + 0.22, r, 0.2, color="#b5b4ad", label="between stages (rest)" if j == 0 else None)
        if np.isfinite(ci[0]):
            ax.errorbar(j + 0.22, r, yerr=[[max(r - ci[0], 0)], [max(ci[1] - r, 0)]], color=C_INK, capsize=2, lw=0.8)
        ax.text(j + 0.22, r + 0.5, f"{wi.get('minutes_valid', 0):.1f} min", ha="center", fontsize=6, color="#6b6a66")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in SESSIONS], rotation=15 if len(SESSIONS) > 3 else 0, fontsize=7)
    ax.set_ylabel("blinks per minute")
    ax.set_title("Lit vs dark vs rest (95 % block bootstrap)", loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    style(ax)

    # (c) PSTH around the lit -> dark transition
    ax = fig.add_subplot(gs[2, 0])
    for name, lab in SESSIONS:
        p = results[name]["psth"]
        e = np.array(p["bin_edges_s"]); mids = (e[:-1] + e[1:]) / 2
        r = np.array(p["to_dark_per_min"], float); ci = np.array(p["to_dark_ci"], float)
        ax.plot(mids, r, "-", color=C_SESS[name], lw=1.6, label=f"{lab} (n {p['n_to_dark']})")
        ax.fill_between(mids, ci[0], ci[1], color=C_SESS[name], alpha=0.12, lw=0)
    yl = ax.get_ylim()
    ax.axvspan(-3, 0, color=C_LIT, alpha=0.08, lw=0)
    ax.axvspan(0, 6, color=C_DARK, alpha=0.08, lw=0)
    ax.text(-2.9, yl[1] * 0.95, "lit cell", color=C_LIT, fontsize=7, va="top")
    ax.text(0.2, yl[1] * 0.95, "black", color=C_DARK, fontsize=7, va="top")
    ax.set_xlabel("seconds from the cell going dark")
    ax.set_ylabel("blinks per minute")
    ax.set_title("Blink release when the cell goes dark", loc="left")
    ax.legend(fontsize=6.5)
    style(ax)

    # (d) amplitude per 5-min window, (e) base duration per window
    for col, key, ylab, ttl in ((1, "amp_uv", "blink amplitude, uV (median, IQR)", "Blink amplitude per 5-min window"),
                                (2, "base_ms", "blink base duration, ms (median, IQR)", "Blink duration per 5-min window")):
        ax = fig.add_subplot(gs[2, col])
        for name, lab in SESSIONS:
            ws = [w for w in results[name]["five_min"] if key + "_median" in w]
            x = [(w["start_min"] + w["end_min"]) / 2 for w in ws]
            y = [w[key + "_median"] for w in ws]
            lo = [w[key + "_median"] - w[key + "_iqr"][0] for w in ws]
            hi = [w[key + "_iqr"][1] - w[key + "_median"] for w in ws]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=C_SESS[name], ms=4, lw=1.2, capsize=2, label=lab)
        if key == "base_ms":
            ax.axhline(LONG_MS, color=C_DARK, lw=0.8, ls="--")
            ax.text(0.2, LONG_MS + 8, "long blink > 400 ms", fontsize=6.5, color=C_DARK)
        ax.set_xlabel("minutes since the first flicker")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left")
        ax.legend(fontsize=6.5)
        style(ax)

    # (f) long-blink share and burst share
    ax = fig.add_subplot(gs[3, 0])
    for name, lab in SESSIONS:
        ws = [w for w in results[name]["five_min"] if "long_share" in w]
        x = [(w["start_min"] + w["end_min"]) / 2 for w in ws]
        ax.plot(x, [100 * w["fwhm_over150_share"] for w in ws], "o-", color=C_SESS[name], ms=4, lw=1.4, label=f"{lab}: long blinks")
        ax.plot(x, [100 * w["burst_share"] for w in ws], "s--", color=C_SESS[name], ms=4, lw=1.0, mfc="none", label=f"{lab}: in a burst")
    ax.set_ylim(0, 100)
    ax.set_xlabel("minutes since the first flicker")
    ax.set_ylabel("share of blinks, %")
    ax.set_title("Long blinks (FWHM > 150 ms, solid) and bursts (< 1 s apart, dashed)", loc="left")
    ax.text(0.98, 0.04, "blinks > 400 ms at the base: 0-1 % in every window", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#6b6a66")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    style(ax)

    # (g) horizontal free rate
    ax = fig.add_subplot(gs[3, 1])
    for name, lab in SESSIONS:
        ws = results[name]["five_min"]
        x = [(w["start_min"] + w["end_min"]) / 2 for w in ws]
        y = [w.get("horiz_free_per_min", np.nan) for w in ws]
        ci = np.array([w.get("horiz_free_ci", [np.nan, np.nan]) for w in ws], float)
        ax.errorbar(x, y, yerr=[np.array(y) - ci[:, 0], ci[:, 1] - np.array(y)], fmt="o-", color=C_SESS[name], ms=4, lw=1.2, capsize=2, label=lab)
    ax.set_xlabel("minutes since the first flicker")
    ax.set_ylabel("|AF7 - AF8| events per minute, not at a cell change")
    ax.set_title("Horizontal eye movements away from cell changes", loc="left")
    ax.legend(fontsize=6.5)
    style(ax)

    # (h) slow drift
    ax = fig.add_subplot(gs[3, 2])
    for name, lab in SESSIONS:
        ws = results[name]["five_min"]
        x = [(w["start_min"] + w["end_min"]) / 2 for w in ws]
        ax.plot(x, [w.get("slow_af_log10_uv2", np.nan) for w in ws], "o-", color=C_SESS[name], ms=4, lw=1.4, label=f"{lab}: AF7/AF8")
        ax.plot(x, [w.get("slow_dd_log10_uv2", np.nan) for w in ws], "s--", color=C_SESS[name], ms=4, lw=1.0, mfc="none", label=f"{lab}: AF7-AF8")
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] + 0.45 * (yl[1] - yl[0]))
    ax.set_xlabel("minutes since the first flicker")
    ax.set_ylabel("log10 power 0.1-0.5 Hz (uV^2)")
    ax.set_title("Slow EOG / electrode drift per 5-min window", loc="left")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    style(ax)

    # (i) example trace from run 2 with detections
    ax = fig.add_subplot(gs[4, :])
    S, D = store["vr_planes10"]
    t = S["t"]
    t0 = S["first"] + 1400
    a, b = np.searchsorted(t, t0), np.searchsorted(t, t0 + 40)
    st = S["state"][a:b]
    ax.fill_between(t[a:b] - S["first"], -40, 90, where=np.isin(st, [sof.FLICKER, sof.TAGGED]), color=C_LIT, alpha=0.08, lw=0)
    ax.fill_between(t[a:b] - S["first"], -40, 90, where=st == sof.DARK, color=C_DARK, alpha=0.08, lw=0)
    ax.plot(t[a:b] - S["first"], S["rsig"][a:b], color="#1baf7a", lw=0.8, label="-median(TP9, TP10, AUX), 0.5-8 Hz: the blink through the Fpz reference")
    ax.plot(t[a:b] - S["first"], S["afm"][a:b] - 30, color="#4a3aa7", lw=0.7, label="mean(AF7, AF8), 0.5-8 Hz, offset -30 uV")
    ax.plot(t[a:b] - S["first"], S["dd"][a:b] - 60 + 25, color="#8a8a85", lw=0.6, label="AF7 - AF8, 0.5-15 Hz, offset -35 uV")
    ev = D["ref"][(D["ref"]["t"] >= t0) & (D["ref"]["t"] < t0 + 40)]
    ax.plot(ev["t"] - S["first"], ev["amp"] + 4, "v", color="#e34948", ms=6, label="detected blink")
    ax.set_ylim(-60, 90)
    ax.set_xlabel("seconds since the first flicker, run 2 (green plane, 12 Hz; blue shading = lit cell, orange = black cell)")
    ax.set_ylabel("uV")
    ax.set_title("Run 2 example: 40 s of the green plane with the detections", loc="left")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    style(ax)

    fig.suptitle("State of Flash, the eyes: blinks through the flicker sessions of 2026-10-10 (run 1 at 17:57, run 2 at 19:50) "
                 "against the reference day; 95 % block-bootstrap intervals", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(OUT, f"flash_eog{TAG}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    results, store = {}, {}
    for name, lab in SESSIONS:
        res, S, D = analyse(name)
        res["label"] = lab
        results[name], store[name] = res, (S, D)
        w = res["windows"]["on_task"]
        print(f"{name:12s} {lab:22s} stim {res['stimulation_min']:5.1f} min, valid {w['minutes_valid']:5.1f} min, movement {res['movement_share_on_task']*100:4.1f} %")
        print(f"   blinks (ref-projected >= 25 uV) {w['blink_per_min']:5.2f}/min [{w['blink_ci'][0]:.2f}, {w['blink_ci'][1]:.2f}]  mirror {w['mirror_per_min']:.2f}  net {w['net_per_min']:.2f} [{w['net_ci'][0]:.2f}, {w['net_ci'][1]:.2f}]"
              f"  >= 40 uV {w['hi_per_min']:.2f} (mirror {w['hi_mirror_per_min']:.2f})  frontal spec {w['frontal_per_min']:.2f} (mirror {w['frontal_mirror_per_min']:.2f})  blinkmask100 {w['bm100_per_min']:.2f}")
        if "amp_uv_median" in w:
            print(f"   amp {w['amp_uv_median']:.1f} uV, fwhm {w['fwhm_ms_median']:.0f} ms, base {w['base_ms_median']:.0f} ms, long {w['long_share']*100:.0f} %, burst {w['burst_share']*100:.0f} %, AF7/AF8 at blink {w['af7_at_blink_uv']:.1f}/{w['af8_at_blink_uv']:.1f} uV")
        if "lit_per_min" in w:
            print(f"   lit {w['lit_per_min']:.2f} vs dark {w['dark_per_min']:.2f}, dark - lit {w['dark_minus_lit']:+.2f} [{w['dark_minus_lit_ci'][0]:+.2f}, {w['dark_minus_lit_ci'][1]:+.2f}]")
        for key, lbl in (("lit_dark_12hz_stages", "12 Hz stages"), ("lit_dark_tagged_stages", "9 Hz / sweep stages")):
            g = res.get(key)
            if g and "lit_per_min" in g:
                print(f"   {lbl:20s} lit {g['lit_per_min']:.2f} vs dark {g['dark_per_min']:.2f}, dark - lit {g['dark_minus_lit']:+.2f} [{g['dark_minus_lit_ci'][0]:+.2f}, {g['dark_minus_lit_ci'][1]:+.2f}] ({g['lit_minutes']:.1f} / {g['dark_minutes']:.1f} min)")
        for key, lbl in (("release_after_dark_onset", "after cell goes dark"), ("release_after_lit_onset", "after cell lights up")):
            r = res[key]
            if r:
                print(f"   {lbl}: {r['first_second_per_min']:.1f}/min in the first second vs {r['seconds_1_to_4_per_min']:.1f} at 1-4 s, diff {r['diff']:+.1f} [{r['diff_ci'][0]:+.1f}, {r['diff_ci'][1]:+.1f}] (n {r['n_transitions']})")
        dr = res["drift"].get("blink_rate_per_10min")
        if dr:
            print(f"   drift {dr['slope']:+.2f}/min per 10 min [{dr['ci'][0]:+.2f}, {dr['ci'][1]:+.2f}], Kendall p {dr['p']:.3f}; halves {res['drift'].get('first_half_per_min', float('nan')):.2f} -> {res['drift'].get('second_half_per_min', float('nan')):.2f}")
        for key, lbl in (("amp_uv_per_10min", "amp uV"), ("base_ms_per_10min", "base ms"), ("fwhm_ms_per_10min", "fwhm ms"),
                         ("horizontal_free_per_10min", "horiz free/min"), ("slow_af_log10_per_10min", "slow AF log10"), ("slow_dd_log10_per_10min", "slow AF7-AF8 log10")):
            v = res["drift"].get(key)
            if v:
                print(f"      slope {lbl:20s} {v['slope']:+.3f} per 10 min [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" + (f" p {v['p']:.3f}" if "p" in v else ""))
        for key in ("long_share_second_minus_first_half", "fwhm_over150_share_second_minus_first_half", "burst_share_second_minus_first_half"):
            v = res["drift"].get(key)
            if v:
                print(f"      {key:45s} {v['first']*100:4.0f} % -> {v['second']*100:4.0f} %, diff {v['diff']*100:+.0f} pts [{v['ci'][0]*100:+.0f}, {v['ci'][1]*100:+.0f}]")
        print(f"   idle {res['windows']['idle'].get('blink_per_min', float('nan')):.1f}/min ({res['windows']['idle'].get('minutes_valid', 0):.1f} min), post {res['windows']['post'].get('blink_per_min', float('nan')):.1f}/min ({res['windows']['post'].get('minutes_valid', 0):.1f} min), pre quiet {res['windows']['pre_last3min'].get('minutes_valid', 0):.2f} min {res['windows']['pre_last3min'].get('blink_per_min', float('nan')):.1f}/min")
        h = res["horizontal"]
        print(f"   horizontal: {h['within_0_6s_after_change_per_change']:.2f} events per cell change within 0.6 s, free rate {h['free_per_min']:.1f}/min, amp {h['amp_uv_median']:.0f} uV")
        for fw in res["five_min"]:
            print(f"      {fw['label']:10s} blinks {fw.get('blink_per_min', float('nan')):5.2f} [{fw.get('blink_ci', [0, 0])[0]:.1f}, {fw.get('blink_ci', [0, 0])[1]:.1f}] net {fw.get('net_per_min', float('nan')):5.2f}  amp {fw.get('amp_uv_median', float('nan')):5.1f}  base {fw.get('base_ms_median', float('nan')):4.0f} ms  long {fw.get('long_share', float('nan'))*100:3.0f} %  burst {fw.get('burst_share', float('nan'))*100:3.0f} %  horiz free {fw.get('horiz_free_per_min', float('nan')):4.1f}  slow {fw.get('slow_af_log10_uv2', float('nan')):.2f}")
    cmp_ = matched(results["vr_planes9"], results["vr_planes10"], *store["vr_planes9"], *store["vr_planes10"])
    pm = cmp_["paired_minutes"]
    print(f"\nrun 2 - run 1 at matched minutes ({pm['n_minutes']} min): {pm['run1_mean']:.2f} -> {pm['run2_mean']:.2f}, diff {pm['diff']:+.2f} [{pm['diff_ci'][0]:+.2f}, {pm['diff_ci'][1]:+.2f}], permutation p {pm['p_permutation']:.4f}")
    for d in cmp_["five_min"]:
        print(f"   {d['window']:10s} run1 {d['run1']:.2f} run2 {d['run2']:.2f} diff {d.get('diff', float('nan')):+.2f} [{d.get('diff_ci', [float('nan')]*2)[0]:+.2f}, {d.get('diff_ci', [float('nan')]*2)[1]:+.2f}]")
    for d in cmp_["stages"]:
        print(f"   {d['stage']:18s} ({d['matched_minutes']:.1f} min) run1 {d['run1']:.2f} run2 {d['run2']:.2f} diff {d['diff']:+.2f} {d['diff_ci']}")
    out = dict(sessions=results, run2_minus_run1=cmp_,
               method=dict(blink="-median(TP9,TP10,AUX) 0.5-8 Hz, positive pulse >= max(3 robust sd, 25 uV), FWHM 50-500 ms, all three far channels >= half the pulse; mirror = same on the sign-flipped signal, outside 0.5 s of a blink, per minute of blink-free time",
                           frontal="mean(AF7,AF8) 0.5-8 Hz, positive pulse >= 3 robust sd on both channels within 100 ms, FWHM 50-500 ms (as specified; at the noise floor on this montage)",
                           horizontal="|AF7-AF8| 0.5-15 Hz peaks >= max(3 robust sd, 20 uV), 150 ms apart, not within 150 ms of a blink; after_change = within 0.6 s of a lit/dark edge",
                           slow="0.1-0.5 Hz power in 20 s Hann segments, median per window, log10 uV^2",
                           artifacts="two or more channels > 150 uV (1-40 Hz) or a railed sample, +-1 s, excluded",
                           uncertainty="block bootstrap over 30 s blocks (2000 draws), Theil-Sen 95 % slope intervals, sign-flip permutation over paired 1-min bins"))
    json.dump(clean(out), open(os.path.join(OUT, f"flash_eog{TAG}.json"), "w"), indent=1)
    figure(results, store)
    print(f"\nwrote {os.path.join(OUT, f'flash_eog{TAG}.json')} and flash_eog{TAG}.png")


if __name__ == "__main__":
    # python analysis/flash_eog.py tag=_all vr_planes8="reference day" vr_planes9="run 1" ...
    # (the first session named is the level reference, as vr_planes8 was)
    args = sys.argv[1:]
    if args:
        sess = []
        for a in args:
            k, _, v = a.partition("=")
            if k == "tag":
                TAG = v
            else:
                sess.append((k, v or k))
        if sess:
            SESSIONS = sess
            for j, (k, _) in enumerate(sess):
                C_SESS.setdefault(k, PALETTE[j % len(PALETTE)])
    main()
