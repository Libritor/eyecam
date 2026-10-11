"""Refutation check, lens B (artifact / confound): the claimed frontal 1-4 Hz
lit-minus-dark gap (AF7 / AF8, about -0.16 log10, "31 % less power while a cell
is lit") in runs/vr_planes9 and runs/vr_planes10 of 2026-10-10, with
runs/vr_planes8 as the level reference.

What could make a lit / dark gap in frontal 1-4 Hz power that is not EEG:
  ocular    blinks and saccades are 0.3-3 Hz events; the subject fixates a lit
            cell and lets the eyes go on a black screen, so dark epochs carry
            more EOG.  On a Muse every channel is measured against Fpz, so a
            blink appears on all five channels, strongest on the far ones.
  transient every cell change (every 6 / 10 s, lit or dark) is a visual onset
            or offset and a saccade to the new cell: a 1-4 Hz deflection.
  movement  headset / head movement (> 150 uV 1-40 Hz on any channel).
  drift     sub-1 Hz electrode drift leaks into the 1 Hz bin of a 2 s Hann PSD
            taken on the raw signal.
  statistics 1-4 Hz power is autocorrelated and lit / dark come in blocks of
            6-30 s, so an epoch-level permutation overstates the evidence; the
            right unit is the block (cell), paired with its neighbour.

Subsets (each adds to the previous one):
  S0  the colleague's selection: pure epoch (>= 90 % one state), AF7 / AF8 not
      railed / flat / > 150 uV, minutes >= 0           (reproduction)
  S1  + no channel > 150 uV (movement)
  S2  + no blink by the driver's detector (blinkmask, 100 uV on AF7 / AF8)
  S3  + no ocular event: reference blink (-median(TP9, TP10, AUX), 0.5-8 Hz,
      >= max(25 uV, 3 robust sd), 50-500 ms wide), frontal deflection
      (blinkmask 30 uV) or saccade (|AF7 - AF8| 0.5-15 Hz >= max(20 uV, 3 sd)),
      +-0.3 s
  S4  + steady state: the epoch starts >= 1.0 s after the last cell / block
      change (no onset / offset transient, no saccade to the new cell)

Measures per 2 s epoch (non-overlapping, the colleague's grid):
  lf_raw   log10 mean PSD 1-4 Hz of the raw signal, mean of AF7 and AF8
           (the colleague's measure)
  lf_bp    the same after a 1-40 Hz band-pass (no sub-1 Hz drift leakage)
  lf_15    log10 mean PSD 1.5-4 Hz (1 Hz bin dropped)
  lf_td    log10 variance of a 1-4 Hz band-pass (time domain)
  theta    log10 mean PSD 4-7.5 Hz, stimulus lines cut (blue plane claim)
  per channel lf_bp for TP9, AF7, AF8, TP10, AUX, and for AF7 - AF8 and
  mean(AF) - mean(TP) (reference-free)

Statistics: median(lit) - median(dark) with a moving-block bootstrap (5
epochs) and an epoch permutation p (the colleague's test), then the same
contrast at block level: per block (one cell or one ON / OFF block) the median
over its clean epochs; block permutation p stratified by stage; and paired
adjacent blocks (each lit block against the nearest dark block in the same
stage) with a sign-flip permutation and a bootstrap interval of the median
paired difference.  Theil-Sen slope of the paired difference over minutes.

    python refute_confounds_5_b.py          (cwd = repo root)
writes runs/state_of_flash/refute_confounds_5_b.json and .png
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

ROOT = r"C:\Users\alexa\eyecam"
sys.path.insert(0, ROOT)
import blinkmask  # noqa: E402
from analysis import state_of_flash as sof  # noqa: E402

OUT = os.path.join(ROOT, "runs", "state_of_flash")
NAME = "refute_confounds_5_b"
SESSIONS = [("vr_planes9", "run 1 (17:57)"), ("vr_planes10", "run 2 (19:50)"), ("vr_planes8", "reference 10-07")]
EPOCH_S = 2.0
ART_UV = 150.0
RAIL_LO, RAIL_HI = 0.5, 1649.5
FLAT_S = 0.25
MARGIN_S = 0.3
STEADY_S = 1.0
B_BOOT, B_PERM = 2000, 4000
C_LIT, C_DARK, C_INK, C_NONE = "#2a78d6", "#eb6834", "#3a3a37", "#b5b4ad"
CH = ["TP9", "AF7", "AF8", "TP10", "AUX"]


# ----------------------------------------------------------------- statistics
def finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _boot_median(x, block, B, seed):
    n = len(x)
    bl = int(max(1, min(block, n // 3)))
    nb = int(np.ceil(n / bl))
    st = np.random.default_rng(seed).integers(0, n - bl + 1, size=(B, nb))
    idx = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n]
    return np.median(x[idx], axis=1)


def contrast(a, b, block=5, seed=0):
    """median(a) - median(b), moving-block bootstrap CI, permutation p over elements."""
    a, b = finite(a), finite(b)
    if len(a) < 4 or len(b) < 4:
        return dict(n_a=int(len(a)), n_b=int(len(b)), diff=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    rng = np.random.default_rng(seed)
    obs = float(np.median(a) - np.median(b))
    d = _boot_median(a, block, B_BOOT, seed + 1) - _boot_median(b, block, B_BOOT, seed + 2)
    pooled = np.concatenate([a, b])
    perm = rng.permuted(np.tile(pooled, (B_PERM, 1)), axis=1)
    dp = np.median(perm[:, :len(a)], axis=1) - np.median(perm[:, len(a):], axis=1)
    p = float((np.sum(np.abs(dp) >= abs(obs)) + 1) / (B_PERM + 1))
    return dict(n_a=int(len(a)), n_b=int(len(b)), diff=obs, ci=[float(v) for v in np.percentile(d, [2.5, 97.5])], p=p)


def block_contrast(vals, lit, stage, seed=0):
    """Block-level: median(lit blocks) - median(dark blocks); bootstrap over
    blocks; permutation of the lit / dark labels within each stage."""
    vals, lit, stage = np.asarray(vals, float), np.asarray(lit, bool), np.asarray(stage)
    ok = np.isfinite(vals)
    vals, lit, stage = vals[ok], lit[ok], stage[ok]
    if lit.sum() < 3 or (~lit).sum() < 3:
        return dict(n_lit=int(lit.sum()), n_dark=int((~lit).sum()), diff=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    obs = float(np.median(vals[lit]) - np.median(vals[~lit]))
    rng = np.random.default_rng(seed)
    a, b = vals[lit], vals[~lit]
    d = np.median(a[rng.integers(0, len(a), (B_BOOT, len(a)))], axis=1) - np.median(b[rng.integers(0, len(b), (B_BOOT, len(b)))], axis=1)
    cnt = 0
    for _ in range(B_PERM):
        lab = lit.copy()
        for s in np.unique(stage):
            m = stage == s
            lab[m] = rng.permutation(lit[m])
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        if abs(np.median(vals[lab]) - np.median(vals[~lab])) >= abs(obs):
            cnt += 1
    return dict(n_lit=int(lit.sum()), n_dark=int((~lit).sum()), diff=obs,
                ci=[float(v) for v in np.percentile(d, [2.5, 97.5])], p=float((cnt + 1) / (B_PERM + 1)))


def paired(diffs, seed=0):
    """Median of paired (lit - nearest dark block) differences: bootstrap CI,
    sign-flip permutation p, share negative."""
    d = finite(diffs)
    if len(d) < 5:
        return dict(n=int(len(d)), median=float("nan"), ci=[float("nan")] * 2, p=float("nan"), share_negative=float("nan"))
    rng = np.random.default_rng(seed)
    obs = float(np.median(d))
    bs = np.median(d[rng.integers(0, len(d), (B_BOOT, len(d)))], axis=1)
    flips = rng.choice([-1.0, 1.0], size=(B_PERM, len(d)))
    dp = np.median(flips * d[None, :], axis=1)
    p = float((np.sum(np.abs(dp) >= abs(obs)) + 1) / (B_PERM + 1))
    return dict(n=int(len(d)), median=obs, ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], p=p,
                share_negative=float((d < 0).mean()), mean=float(d.mean()))


def theil(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or x[ok].max() - x[ok].min() < 3:
        return dict(n=int(ok.sum()), slope_per_10min=float("nan"), ci=[float("nan")] * 2)
    r = stats.theilslopes(y[ok], x[ok], 0.95)
    return dict(n=int(ok.sum()), slope_per_10min=float(r[0] * 10), ci=[float(r[2] * 10), float(r[3] * 10)])


# ----------------------------------------------------------------- labelling
def labels(d, t, f0):
    """Per sample: lit, stage id, stimulus code, block id (a cell or ON/OFF
    block), seconds since the block began."""
    n = len(t)
    lit = np.zeros(n, bool)
    stage = np.full(n, -1, np.int16)
    fstim = np.zeros(n, np.float32)
    block = np.full(n, -1, np.int32)
    since = np.full(n, np.nan, np.float32)
    names, spans = [], {}
    plane_hz = {}
    try:
        pm = json.load(open(os.path.join(d, "planes_meta.json")))
        plane_hz = {str(k): float(v) for k, v in (pm.get("freq_delivered") or {}).items()}
    except Exception:
        pass
    first, last, nblk = np.inf, -np.inf, 0
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        L = pd.read_csv(p)
        if len(L) < 10 or "luminance" not in L.columns:
            continue
        L = L.sort_values("time")
        lt = L["time"].to_numpy(float)
        lum = L["luminance"].to_numpy(float)
        gx, gy = L["grid_x"].to_numpy(float), L["grid_y"].to_numpy(float)
        key = np.stack([lum > 0.05, gx, gy], axis=1)
        change = np.concatenate([[True], np.any(key[1:] != key[:-1], axis=1)])
        bid = np.cumsum(change) - 1 + nblk
        bstart = lt[change][bid - nblk]
        nblk = int(bid.max()) + 1
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5)
        sid = len(names)
        names.append(kind)
        stage[ok] = sid
        lit[ok] = lum[idx[ok]] > 0.05
        block[ok] = bid[idx[ok]]
        since[ok] = t[ok] - bstart[idx[ok]]
        if kind.endswith("_sweep"):
            f = -1.0
        elif kind.startswith("plane_"):
            f = plane_hz.get(kind.split("_")[1], f0)
        else:
            f = f0
        fstim[ok] = f
        spans[kind] = [float(lt[0]), float(lt[-1]), float(f)]
        first, last = min(first, lt[0]), max(last, lt[-1])
    return lit, stage, fstim, block, since, names, spans, first, last


def line_cut(f, fcode, f0):
    cut = np.abs(f - 60.0) <= 2.0
    sets = []
    if fcode > 0:
        sets.append(f0 if abs(fcode - 12.0) < 0.5 else fcode)
    elif fcode < 0:
        sets.extend([6.0, 7.2, 9.0, f0])
    for base in sets:
        for k in range(1, int(128 // base) + 1):
            cut |= np.abs(f - k * base) <= 0.75
        cut |= np.abs(f - base / 2) <= 0.6
    return cut


def flat_mask(x, fs):
    same = np.concatenate([[False], np.diff(x) == 0])
    w = int(round(FLAT_S * fs))
    run = np.convolve(same.astype(float), np.ones(w), mode="same") >= w - 0.5
    return np.convolve(run.astype(float), np.ones(w), mode="same") > 0


def dilate(mask, n):
    if n <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(float), np.ones(2 * n + 1), mode="same") > 0


def robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ocular_events(X, names, fs, ref_uv=None, sac_uv=None, frontal_uv=30.0):
    """Sample masks: reference blink (far channels against Fpz), frontal blink
    (blinkmask frontal_uv), saccade (|AF7 - AF8|).  Thresholds default to
    max(25 uV, 3 robust sd) and max(20 uV, 3 robust sd); fixed values can be
    given.  Returns dict of bool masks and event counts."""
    n = len(X)
    i = {nm: names.index(nm) for nm in names}
    sos8 = butter(2, [0.5, 8.0], "bandpass", fs=fs, output="sos")
    sos15 = butter(2, [0.5, 15.0], "bandpass", fs=fs, output="sos")
    far = [i[c] for c in ("TP9", "TP10") if c in i] + ([blinkmask.oz_index(names)] if blinkmask.oz_index(names) is not None else [])
    ref = -np.median(sosfiltfilt(sos8, X[:, far], axis=0), axis=1)
    thr_ref = max(25.0, 3.0 * robust_sd(ref)) if ref_uv is None else float(ref_uv)
    pk_ref, _ = find_peaks(ref, height=thr_ref, width=(0.05 * fs, 0.5 * fs), rel_height=0.5)
    ref_mask = np.zeros(n, bool)
    ref_mask[pk_ref] = True
    ref_mask = dilate(ref_mask, int(round(0.25 * fs)))
    fr_mask = blinkmask.blink_mask(X, names, fs, frontal_uv)
    h = sosfiltfilt(sos15, X[:, i["AF7"]] - X[:, i["AF8"]])
    thr_h = max(20.0, 3.0 * robust_sd(h)) if sac_uv is None else float(sac_uv)
    pk_h, _ = find_peaks(np.abs(h), height=thr_h, distance=int(0.2 * fs))
    sac_mask = np.zeros(n, bool)
    sac_mask[pk_h] = True
    sac_mask = dilate(sac_mask, int(round(0.15 * fs)))
    m = int(round(MARGIN_S * fs))
    return dict(ref=dilate(ref_mask, m), frontal=dilate(fr_mask, m), saccade=dilate(sac_mask, m),
                ref_events=pk_ref, sac_events=pk_h, thr_ref=float(thr_ref), thr_sac=float(thr_h), ref_signal=ref)


# ----------------------------------------------------------------- epochs
def epochs(d):
    t, X, names, fs = sof.load_eeg(d)
    f0, _ = sof.stim_freq(d)
    lit, stage, fstim, block, since, stage_names, spans, first, last = labels(d, t, f0)
    n = int(round(EPOCH_S * fs))
    starts = np.arange(0, len(t) - n, n)
    E = len(starts)
    sos = butter(2, [1.0, 40.0], "bandpass", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    sos14 = butter(2, [1.0, 4.0], "bandpass", fs=fs, output="sos")
    X14 = sosfiltfilt(sos14, X, axis=0)
    oc = ocular_events(X, names, fs)
    oc_low = ocular_events(X, names, fs, ref_uv=15.0, sac_uv=12.0, frontal_uv=20.0)
    blink100 = blinkmask.blink_mask(X, names, fs, 100.0)
    rail = (X <= RAIL_LO) | (X >= RAIL_HI)
    flat = np.stack([flat_mask(X[:, k], fs) for k in range(X.shape[1])], axis=1)

    seg = np.stack([X[s:s + n] for s in starts])
    seg = seg - seg.mean(axis=1, keepdims=True)
    segf = np.stack([Xf[s:s + n] for s in starts])
    segf = segf - segf.mean(axis=1, keepdims=True)
    seg14 = np.stack([X14[s:s + n] for s in starts])
    f, P = welch(seg, fs=fs, nperseg=n, axis=1)
    _, Pf = welch(segf, fs=fs, nperseg=n, axis=1)
    iAF7, iAF8 = names.index("AF7"), names.index("AF8")
    iTP9, iTP10 = names.index("TP9"), names.index("TP10")
    # reference-free derivations
    bip = segf[:, :, iAF7] - segf[:, :, iAF8]
    fe = 0.5 * (segf[:, :, iAF7] + segf[:, :, iAF8]) - 0.5 * (segf[:, :, iTP9] + segf[:, :, iTP10])
    _, Pbip = welch(bip, fs=fs, nperseg=n, axis=1)
    _, Pfe = welch(fe, fs=fs, nperseg=n, axis=1)

    out = dict(t=t[starts] + EPOCH_S / 2)
    out["minutes"] = (out["t"] - first) / 60.0
    sl = np.stack([stage[s:s + n] for s in starts])
    ll = np.stack([lit[s:s + n] for s in starts])
    fl = np.stack([fstim[s:s + n] for s in starts])
    bl = np.stack([block[s:s + n] for s in starts])
    sc = np.stack([since[s:s + n] for s in starts])
    maj = np.array([np.bincount(r + 1, minlength=len(stage_names) + 1).argmax() - 1 for r in sl])
    pure_stage = np.array([(r == m).mean() for r, m in zip(sl, maj)])
    lit_share = ll.mean(axis=1)
    out["stage"] = maj
    out["stage_name"] = [stage_names[m] if m >= 0 else "" for m in maj]
    out["fstim"] = np.array([np.median(r[r != 0]) if (r != 0).any() else 0.0 for r in fl])
    out["block"] = np.array([np.bincount(r[r >= 0]).argmax() if (r >= 0).any() else -1 for r in bl])
    out["since"] = np.nanmin(np.where(np.isfinite(sc), sc, np.inf), axis=1)
    state = np.where(lit_share >= 0.9, "lit", np.where(lit_share <= 0.1, "dark", "mixed"))
    state = np.where((maj < 0) | (pure_stage < 0.9), "idle", state)
    state = np.where(out["t"] < first, "pre", state)
    state = np.where(out["t"] > last + 1.0, "post", state)
    out["state"] = state
    peak = np.abs(segf).max(axis=1)
    out["move_any"] = (peak > ART_UV).any(axis=1)
    for k, ch in enumerate(names):
        out[f"bad_{ch}"] = (np.array([rail[s:s + n, k].mean() for s in starts]) > 0) | \
                           (np.array([flat[s:s + n, k].mean() for s in starts]) > 0) | (peak[:, k] > ART_UV)
    out["af_bad"] = out["bad_AF7"] | out["bad_AF8"]
    out["blink100"] = np.array([blink100[s:s + n].any() for s in starts])
    out["oc_ref"] = np.array([oc["ref"][s:s + n].any() for s in starts])
    out["oc_frontal"] = np.array([oc["frontal"][s:s + n].any() for s in starts])
    out["oc_sac"] = np.array([oc["saccade"][s:s + n].any() for s in starts])
    out["ocular"] = out["oc_ref"] | out["oc_frontal"] | out["oc_sac"]
    out["ocular_low"] = np.array([(oc_low["ref"] | oc_low["frontal"] | oc_low["saccade"])[s:s + n].any() for s in starts])
    out["ocular_refsac"] = out["oc_ref"] | out["oc_sac"]            # without the frontal 30 uV detector
    out["n_ref_blinks"] = np.array([((oc["ref_events"] >= s) & (oc["ref_events"] < s + n)).sum() for s in starts])
    out["n_saccades"] = np.array([((oc["sac_events"] >= s) & (oc["sac_events"] < s + n)).sum() for s in starts])
    out["n_events"] = out["n_ref_blinks"] + out["n_saccades"]
    lf = (f >= 1.0) & (f <= 4.0)
    lf15 = (f >= 1.5) & (f <= 4.0)
    slow = np.abs(f - 0.5) < 0.1
    th = (f >= 4.0) & (f <= 7.5)
    for k, ch in enumerate(names):
        out[f"lf_raw_{ch}"] = np.log10(P[:, lf, k].mean(axis=1))
        out[f"lf_bp_{ch}"] = np.log10(Pf[:, lf, k].mean(axis=1))
        out[f"lf_15_{ch}"] = np.log10(Pf[:, lf15, k].mean(axis=1))
        out[f"lf_td_{ch}"] = np.log10(seg14[:, :, k].var(axis=1))
        out[f"slow_{ch}"] = np.log10(P[:, slow, k].mean(axis=1))      # the 0.5 Hz bin of the raw PSD: sub-1 Hz drift
        thc = np.full(E, np.nan)
        for code in np.unique(out["fstim"]):
            sel = out["fstim"] == code
            keep = th & ~line_cut(f, code, f0)
            if keep.sum() == 0:
                continue
            thc[sel] = np.log10(Pf[sel][:, keep, k].mean(axis=1))
        out[f"theta_{ch}"] = thc
    out["lf_bp_AF7-AF8"] = np.log10(Pbip[:, lf].mean(axis=1))
    out["lf_bp_AF-TP"] = np.log10(Pfe[:, lf].mean(axis=1))
    dfe = pd.DataFrame(out)
    for m in ("lf_raw", "lf_bp", "lf_15", "lf_td", "theta", "slow"):
        dfe[f"{m}_AF"] = dfe[[f"{m}_AF7", f"{m}_AF8"]].mean(axis=1)
        dfe[f"{m}_TP"] = dfe[[f"{m}_TP9", f"{m}_TP10"]].mean(axis=1)
    meta = dict(names=names, fs=float(fs), f0=float(f0), first=float(first), last=float(last),
                stages=spans, n_epochs=int(E), thr_ref_blink_uv=oc["thr_ref"], thr_saccade_uv=oc["thr_sac"])
    spec = dict(f=f, Pf=Pf)
    return dfe, meta, spec


# ----------------------------------------------------------------- subsets and contrasts
def subsets(on):
    S = {}
    S["S0_colleague"] = ~on["af_bad"]
    S["S1_no_move"] = S["S0_colleague"] & ~on["move_any"]
    S["S2_no_blink100"] = S["S1_no_move"] & ~on["blink100"]
    S["S3_no_ocular"] = S["S1_no_move"] & ~on["ocular"]
    S["S4_steady"] = S["S3_no_ocular"] & (on["since"] >= STEADY_S)
    S["S3b_refsac_only"] = S["S1_no_move"] & ~on["ocular_refsac"]    # detector choice: no frontal 30 uV mask
    S["S3c_low_thresholds"] = S["S1_no_move"] & ~on["ocular_low"]     # detector choice: 15 / 12 / 20 uV fixed
    return S


def matched_events(on, S, code, col):
    """Lit vs dark epochs with the SAME number of ocular events (0, 1, 2+):
    if the gap is the event rate, it vanishes within each count."""
    fam = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"]) & S["S1_no_move"]]
    out = {}
    for k, lab in ((0, "0"), (1, "1"), (2, "2+")):
        sub = fam[(fam["n_events"] == k) if k < 2 else (fam["n_events"] >= 2)]
        out[lab] = contrast(sub.loc[sub["state"] == "lit", col], sub.loc[sub["state"] == "dark", col], 5, seed=31 + k)
        out[lab]["share_lit"] = float(((fam["n_events"] == k) if k < 2 else (fam["n_events"] >= 2))[fam["state"] == "lit"].mean())
        out[lab]["share_dark"] = float(((fam["n_events"] == k) if k < 2 else (fam["n_events"] >= 2))[fam["state"] == "dark"].mean())
    # Spearman of 1-4 Hz power with the event count, within each state
    for st in ("lit", "dark"):
        x = fam[fam["state"] == st]
        if len(x) > 10:
            r, p = stats.spearmanr(x["n_events"], x[col])
            out[f"spearman_{st}"] = dict(rho=float(r), p=float(p), n=int(len(x)))
    return out


def block_table(sub, col):
    """Per block: median of col over the clean epochs, lit flag, stage, time."""
    g = sub.groupby("block")
    tb = pd.DataFrame(dict(val=g[col].median(), lit=g["state"].first() == "lit", stage=g["stage"].first(),
                           t=g["minutes"].median(), n=g[col].size()))
    return tb[tb.index >= 0]


def paired_diffs(tb):
    """Each lit block against the nearest dark block in the same stage (by time)."""
    diffs, times = [], []
    for s in tb["stage"].unique():
        x = tb[tb["stage"] == s]
        L, D = x[x["lit"]], x[~x["lit"]]
        if len(L) == 0 or len(D) == 0:
            continue
        dt_ = D["t"].to_numpy()
        for _, r in L.iterrows():
            j = int(np.argmin(np.abs(dt_ - r["t"])))
            if abs(dt_[j] - r["t"]) <= 1.0:            # within a minute
                diffs.append(r["val"] - D["val"].to_numpy()[j])
                times.append(r["t"])
    return np.array(diffs), np.array(times)


def analyse_family(on, S, code, col):
    """One stimulus family (12 Hz stages or 9 Hz blue plane), one measure."""
    res = {}
    fam = on[np.abs(on["fstim"] - code) < 0.5] if code > 0 else on
    fam = fam[fam["state"].isin(["lit", "dark"])]
    for sname, mask in S.items():
        sub = fam[mask.loc[fam.index]]
        lit_, dark_ = sub[sub["state"] == "lit"], sub[sub["state"] == "dark"]
        r = dict(n_lit=int(len(lit_)), n_dark=int(len(dark_)))
        r["epoch"] = contrast(lit_[col], dark_[col], 5, seed=7)
        tb = block_table(sub, col)
        r["block"] = block_contrast(tb["val"], tb["lit"], tb["stage"], seed=11)
        dd, tt = paired_diffs(tb)
        r["paired"] = paired(dd, seed=13)
        r["paired_drift"] = theil(tt, dd)
        r["lit_median"], r["dark_median"] = float(np.nanmedian(lit_[col])), float(np.nanmedian(dark_[col]))
        res[sname] = r
    return res


def per_stage(on, S, col, sname="S3_no_ocular"):
    out = {}
    mask = S[sname]
    for st in on["stage_name"].unique():
        if not st:
            continue
        sub = on[(on["stage_name"] == st) & on["state"].isin(["lit", "dark"]) & mask]
        if len(sub) < 8:
            continue
        lit_, dark_ = sub[sub["state"] == "lit"], sub[sub["state"] == "dark"]
        tb = block_table(sub, col)
        dd, _ = paired_diffs(tb)
        out[st] = dict(n_lit=int(len(lit_)), n_dark=int(len(dark_)), epoch=contrast(lit_[col], dark_[col], 5, seed=3),
                       paired=paired(dd, seed=4), fstim=float(sub["fstim"].iloc[0]))
    return out


def per_channel(on, S, code, sname="S1_no_move"):
    """The lit-dark gap on every channel: a common (Fpz) signal gives the same
    gap everywhere; a frontal one gives it on AF7 / AF8 only."""
    fam = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"]) & S[sname]]
    out = {}
    for ch in CH + ["AF7-AF8", "AF-TP"]:
        col = f"lf_bp_{ch}"
        if col not in fam:
            continue
        okc = ~fam[f"bad_{ch}"] if ch in CH else ~fam["af_bad"]
        sub = fam[okc]
        out[ch] = contrast(sub.loc[sub["state"] == "lit", col], sub.loc[sub["state"] == "dark", col], 5, seed=9)
    return out


def ocular_rates(on, S, code):
    fam = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"]) & S["S1_no_move"]]
    out = {}
    for st in ("lit", "dark"):
        x = fam[fam["state"] == st]
        mins = len(x) * EPOCH_S / 60.0
        out[st] = dict(epochs=int(len(x)), ref_blinks_per_min=float(x["n_ref_blinks"].sum() / mins) if mins else float("nan"),
                       saccades_per_min=float(x["n_saccades"].sum() / mins) if mins else float("nan"),
                       share_epochs_ocular=float(x["ocular"].mean()) if len(x) else float("nan"),
                       share_epochs_blink100=float(x["blink100"].mean()) if len(x) else float("nan"))
    # rate contrast by bootstrap over epochs (block 5)
    L, D = fam[fam["state"] == "lit"], fam[fam["state"] == "dark"]
    if len(L) > 10 and len(D) > 10:
        rng = np.random.default_rng(1)
        def rate(x, s):
            v = x["n_ref_blinks"].to_numpy(float)
            n = len(v)
            bl = max(1, min(5, n // 3))
            nb = int(np.ceil(n / bl))
            st = np.random.default_rng(s).integers(0, n - bl + 1, size=(B_BOOT, nb))
            idx = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(B_BOOT, -1)[:, :n]
            return v[idx].mean(axis=1) / EPOCH_S * 60.0
        d = rate(D, 2) - rate(L, 3)
        out["dark_minus_lit_blinks_per_min"] = dict(diff=float(out["dark"]["ref_blinks_per_min"] - out["lit"]["ref_blinks_per_min"]),
                                                   ci=[float(v) for v in np.percentile(d, [2.5, 97.5])])
    return out


def ocular_share_of_gap(on, S, code, col):
    """How much of the S1 gap sits in the ocular epochs: the gap recomputed on
    ocular epochs only and on clean epochs only, and the share of each state."""
    fam = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"]) & S["S1_no_move"]]
    oc = fam[fam["ocular"]]
    cl = fam[~fam["ocular"]]
    return dict(ocular_epochs=contrast(oc.loc[oc["state"] == "lit", col], oc.loc[oc["state"] == "dark", col], 5, seed=21),
                clean_epochs=contrast(cl.loc[cl["state"] == "lit", col], cl.loc[cl["state"] == "dark", col], 5, seed=22),
                ocular_vs_clean_level=float(np.nanmedian(oc[col]) - np.nanmedian(cl[col])) if len(oc) and len(cl) else float("nan"))


# ----------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    results, data, specs = {}, {}, {}
    for name, label in SESSIONS:
        d = os.path.join(ROOT, "runs", name)
        dfe, meta, spec = epochs(d)
        data[name], specs[name] = dfe, spec
        on = dfe[(dfe["minutes"] >= 0) & dfe["state"].isin(["lit", "dark", "mixed", "idle"])].copy()
        S = subsets(on)
        res = dict(label=label, fs=meta["fs"], f0=meta["f0"], stages=meta["stages"], n_epochs=meta["n_epochs"],
                   thresholds=dict(ref_blink_uv=meta["thr_ref_blink_uv"], saccade_uv=meta["thr_saccade_uv"], frontal_blink_uv=30.0,
                                   margin_s=MARGIN_S, steady_s=STEADY_S),
                   subset_sizes={k: dict(lit=int((on.loc[v, "state"] == "lit").sum()), dark=int((on.loc[v, "state"] == "dark").sum())) for k, v in S.items()})
        fams = {"12Hz_stages": 12.0, "9Hz_blue_plane": 9.0}
        res["lf_AF"] = {}
        for tag, code in fams.items():
            if (np.abs(on["fstim"] - code) < 0.5).sum() < 20:
                continue
            res["lf_AF"][tag] = {m: analyse_family(on, S, code, f"{m}_AF") for m in ("lf_raw", "lf_bp", "lf_15", "lf_td", "slow")}
            res["lf_AF"][tag]["theta_cut"] = analyse_family(on, S, code, "theta_AF")
            res["lf_AF"][tag]["matched_event_count_lf_bp"] = matched_events(on, S, code, "lf_bp_AF")
            res["lf_AF"][tag]["per_channel_S1"] = per_channel(on, S, code, "S1_no_move")
            res["lf_AF"][tag]["per_channel_S3"] = per_channel(on, S, code, "S3_no_ocular")
            res["lf_AF"][tag]["ocular_rates_S1"] = ocular_rates(on, S, code)
            res["lf_AF"][tag]["ocular_share_lf_bp"] = ocular_share_of_gap(on, S, code, "lf_bp_AF")
        res["per_stage_S3_lf_bp"] = per_stage(on, S, "lf_bp_AF", "S3_no_ocular")
        res["per_stage_S0_lf_raw"] = per_stage(on, S, "lf_raw_AF", "S0_colleague")
        results[name] = res
        print(f"\n==== {name} {label}  epochs {meta['n_epochs']}  ref-blink thr {meta['thr_ref_blink_uv']:.1f} uV  saccade thr {meta['thr_saccade_uv']:.1f} uV")
        for tag in res["lf_AF"]:
            print(f"  -- {tag}")
            for m in ("lf_raw", "lf_bp", "lf_15", "lf_td", "slow", "theta_cut"):
                for sname, r in res["lf_AF"][tag][m].items():
                    if m in ("lf_15", "slow", "theta_cut") and sname in ("S0_colleague", "S2_no_blink100"):
                        continue
                    e, b, p = r["epoch"], r["block"], r["paired"]
                    print(f"     {m:9s} {sname:19s} n {r['n_lit']:3d}/{r['n_dark']:3d}  epoch {e['diff']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}] p {e['p']:.3f}"
                          f"  block {b['diff']:+.3f} (n {b['n_lit']}/{b['n_dark']}) p {b['p']:.3f}"
                          f"  paired {p['median']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] p {p['p']:.3f} neg {p['share_negative']:.2f} (n {p['n']})")
            me = res["lf_AF"][tag]["matched_event_count_lf_bp"]
            print("     matched event count, lf_bp lit-dark: " + "  ".join(f"{k} events {v['diff']:+.3f} p {v['p']:.2f} (n {v['n_a']}/{v['n_b']}; share lit {v['share_lit']:.2f} dark {v['share_dark']:.2f})"
                                                                         for k, v in me.items() if not k.startswith("spearman"))
                  + "  | spearman power~events: " + "  ".join(f"{st} rho {me[f'spearman_{st}']['rho']:+.2f} p {me[f'spearman_{st}']['p']:.1e}" for st in ("lit", "dark") if f"spearman_{st}" in me))
            pc = res["lf_AF"][tag]["per_channel_S1"]
            print("     per channel S1 lf_bp: " + "  ".join(f"{ch} {v['diff']:+.3f} (p {v['p']:.2f})" for ch, v in pc.items()))
            pc = res["lf_AF"][tag]["per_channel_S3"]
            print("     per channel S3 lf_bp: " + "  ".join(f"{ch} {v['diff']:+.3f} (p {v['p']:.2f})" for ch, v in pc.items()))
            oc = res["lf_AF"][tag]["ocular_rates_S1"]
            print(f"     ref blinks/min lit {oc['lit']['ref_blinks_per_min']:.1f} dark {oc['dark']['ref_blinks_per_min']:.1f}; "
                  f"saccades/min lit {oc['lit']['saccades_per_min']:.1f} dark {oc['dark']['saccades_per_min']:.1f}; "
                  f"share ocular epochs lit {oc['lit']['share_epochs_ocular']:.2f} dark {oc['dark']['share_epochs_ocular']:.2f}")
            osg = res["lf_AF"][tag]["ocular_share_lf_bp"]
            print(f"     gap within ocular epochs {osg['ocular_epochs']['diff']:+.3f} (n {osg['ocular_epochs']['n_a']}/{osg['ocular_epochs']['n_b']}), "
                  f"within clean epochs {osg['clean_epochs']['diff']:+.3f} (n {osg['clean_epochs']['n_a']}/{osg['clean_epochs']['n_b']}); ocular epochs sit {osg['ocular_vs_clean_level']:+.3f} above clean")
        print("  -- per stage, S3 lf_bp (epoch gap / paired block gap):")
        for st, r in res["per_stage_S3_lf_bp"].items():
            print(f"     {st:17s} n {r['n_lit']:3d}/{r['n_dark']:3d} epoch {r['epoch']['diff']:+.3f} p {r['epoch']['p']:.3f}   paired {r['paired']['median']:+.3f} p {r['paired']['p']:.3f} (n {r['paired']['n']})")

    # pooled tonight (epoch level and paired blocks), 12 Hz and 9 Hz, per subset
    pooled = {}
    for tag, code in (("12Hz_stages", 12.0), ("9Hz_blue_plane", 9.0)):
        pooled[tag] = {}
        for m in ("lf_raw", "lf_bp", "lf_td"):
            pooled[tag][m] = {}
            for sname in ("S0_colleague", "S1_no_move", "S2_no_blink100", "S3_no_ocular", "S4_steady", "S3b_refsac_only", "S3c_low_thresholds"):
                lits, darks, pd_ = [], [], []
                for name, _ in SESSIONS[:2]:
                    on = data[name][(data[name]["minutes"] >= 0) & data[name]["state"].isin(["lit", "dark", "mixed", "idle"])].copy()
                    S = subsets(on)
                    fam = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"]) & S[sname]]
                    lits.append(fam.loc[fam["state"] == "lit", f"{m}_AF"])
                    darks.append(fam.loc[fam["state"] == "dark", f"{m}_AF"])
                    dd, _ = paired_diffs(block_table(fam, f"{m}_AF"))
                    pd_.append(dd)
                pooled[tag][m][sname] = dict(epoch=contrast(pd.concat(lits), pd.concat(darks), 5, seed=5),
                                            paired=paired(np.concatenate(pd_), seed=6))
    results["_pooled_tonight"] = pooled
    print("\n==== pooled tonight (run 1 + run 2)")
    for tag in pooled:
        for m in pooled[tag]:
            for sname, r in pooled[tag][m].items():
                e, p = r["epoch"], r["paired"]
                print(f"  {tag:15s} {m:6s} {sname:15s} epoch {e['diff']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}] p {e['p']:.4f} (n {e['n_a']}/{e['n_b']})"
                      f"   paired blocks {p['median']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] p {p['p']:.3f} neg {p['share_negative']:.2f} (n {p['n']})")
    results["_methods"] = dict(epoch_s=EPOCH_S, artifact_uv=ART_UV, margin_s=MARGIN_S, steady_s=STEADY_S, bootstrap_B=B_BOOT, permutation_B=B_PERM,
                               subsets=dict(S0_colleague="pure epoch, AF7/AF8 not railed/flat/>150 uV, minutes >= 0",
                                            S1_no_move="S0 and no channel > 150 uV (1-40 Hz)",
                                            S2_no_blink100="S1 and no blinkmask (100 uV) sample in the epoch",
                                            S3_no_ocular="S1 and no reference blink / frontal 30 uV deflection / saccade (+-0.3 s)",
                                            S4_steady="S3 and epoch starts >= 1 s after the last cell or block change",
                                            S3b_refsac_only="S1 and no reference blink / saccade (frontal 30 uV detector not used)",
                                            S3c_low_thresholds="S1 and no ocular event at fixed low thresholds: reference 15 uV, saccade 12 uV, frontal 20 uV"),
                               measures=dict(lf_raw="log10 mean PSD 1-4 Hz, raw signal (colleague)", lf_bp="same after 1-40 Hz band-pass",
                                             lf_15="1.5-4 Hz after band-pass", lf_td="log10 variance of a 1-4 Hz band-pass", theta_cut="4-7.5 Hz, stimulus lines cut"),
                               tests=dict(epoch="median difference, moving-block bootstrap (5 epochs), permutation over epochs",
                                          block="per-block medians, bootstrap over blocks, label permutation within stage",
                                          paired="lit block minus nearest dark block in the same stage (<= 1 min), bootstrap of the median, sign-flip permutation"))
    json.dump(results, open(os.path.join(OUT, NAME + ".json"), "w"), indent=1, default=float)
    figure(data, results, specs)
    return 0


# ----------------------------------------------------------------- figure
def figure(data, results, specs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK, "xtick.color": C_INK,
                         "ytick.color": C_INK, "text.color": C_INK, "axes.titlesize": 8.5, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    subs = ["S0_colleague", "S1_no_move", "S2_no_blink100", "S3_no_ocular", "S4_steady"]
    short = ["S0 colleague", "S1 no move", "S2 no blink100", "S3 no ocular", "S4 steady"]
    # row 1: lf_bp gap per subset, epoch-level and paired-block, per session and family
    for j, (tag, ttl) in enumerate((("12Hz_stages", "12 Hz stages"), ("9Hz_blue_plane", "9 Hz blue plane"))):
        ax = axes[0, j]
        for i, (name, label) in enumerate(SESSIONS):
            r = results[name]["lf_AF"].get(tag)
            if r is None:
                continue
            y = np.arange(len(subs)) + (i - 1) * 0.22
            e = [r["lf_bp"][s]["epoch"] for s in subs]
            p = [r["lf_bp"][s]["paired"] for s in subs]
            col = {"vr_planes9": "#4a3aa7", "vr_planes10": "#1baf7a", "vr_planes8": "#6f6e69"}[name]
            ax.errorbar([v["diff"] for v in e], y, xerr=[[v["diff"] - v["ci"][0] for v in e], [v["ci"][1] - v["diff"] for v in e]],
                        fmt="o", ms=4, color=col, capsize=2, lw=1, label=f"{label}: epochs")
            ax.errorbar([v["median"] for v in p], y + 0.08, xerr=[[v["median"] - v["ci"][0] for v in p], [v["ci"][1] - v["median"] for v in p]],
                        fmt="s", ms=4, mfc="none", color=col, capsize=2, lw=1, ls=":", label=f"{label}: paired blocks")
        ax.axvline(0, color=C_NONE, lw=1)
        ax.set_yticks(range(len(subs)))
        ax.set_yticklabels(short)
        ax.set_xlabel("AF7/AF8 1-4 Hz, log10 power, lit - dark")
        ax.set_title(f"{ttl}: the gap as the subset gets cleaner", loc="left")
        ax.grid(True, axis="x", color="#ecebe5", lw=0.6)
        ax.set_axisbelow(True)
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    # per channel S1 vs S3 (run 2, 12 Hz)
    ax = axes[0, 2]
    chs = CH + ["AF7-AF8", "AF-TP"]
    for i, (name, label) in enumerate(SESSIONS[:2]):
        r = results[name]["lf_AF"]["12Hz_stages"]
        col = {"vr_planes9": "#4a3aa7", "vr_planes10": "#1baf7a"}[name]
        for k, (key, mk, lab) in enumerate((("per_channel_S1", "o", "S1 no move"), ("per_channel_S3", "s", "S3 no ocular"))):
            pc = r[key]
            x = np.arange(len(chs)) + (i - 0.5) * 0.3 + (k - 0.5) * 0.12
            vals = [pc.get(c, {}).get("diff", np.nan) for c in chs]
            lo = [pc.get(c, {}).get("ci", [np.nan, np.nan])[0] for c in chs]
            hi = [pc.get(c, {}).get("ci", [np.nan, np.nan])[1] for c in chs]
            ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt=mk, ms=4,
                        mfc=("none" if k else col), color=col, capsize=2, lw=1, label=f"{label} {lab}")
    ax.axhline(0, color=C_NONE, lw=1)
    ax.set_xticks(range(len(chs)))
    ax.set_xticklabels(chs, rotation=20)
    ax.set_ylabel("1-4 Hz log10 power, lit - dark")
    ax.set_title("12 Hz stages: the gap on every channel, and reference-free", loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, axis="y", color="#ecebe5", lw=0.6)
    ax.set_axisbelow(True)
    # row 2: paired block differences over minutes (S1 and S3), per tonight's session
    for j, (name, label) in enumerate(SESSIONS[:2]):
        ax = axes[1, j]
        dfe = data[name]
        on = dfe[(dfe["minutes"] >= 0) & dfe["state"].isin(["lit", "dark", "mixed", "idle"])].copy()
        S = subsets(on)
        for sname, mk, col, lab in (("S1_no_move", "o", C_NONE, "S1 no move"), ("S3_no_ocular", "o", C_INK, "S3 no ocular")):
            fam = on[on["state"].isin(["lit", "dark"]) & S[sname] & (on["fstim"] > 0)]
            tb = block_table(fam, "lf_bp_AF")
            dd, tt = paired_diffs(tb)
            ax.plot(tt, dd, mk, ms=3.5, color=col, alpha=0.8, label=f"{lab}: lit block - nearest dark block (n {len(dd)})")
        ax.axhline(0, color=C_NONE, lw=1)
        for st, (a, b, f) in results[name]["stages"].items():
            m0, m1 = (a - results[name]["stages"]["calib"][0]) / 60, (b - results[name]["stages"]["calib"][0]) / 60
            ax.axvspan(m0, m1, color=("#dfe9f7" if f == 12 else "#f7e6df" if f == 9 else "#eeeeea"), alpha=0.5, lw=0)
            ax.text((m0 + m1) / 2, ax.get_ylim()[1] if False else 0.95, st.replace("calib_blue_sweep", "sweep").replace("calib_", "c.").replace("plane_", "p."),
                    ha="center", va="top", fontsize=6.5, transform=ax.get_xaxis_transform())
        ax.set_xlabel("minutes since the first flicker")
        ax.set_ylabel("paired difference, log10 power")
        ax.set_title(f"{label}: paired blocks over the session (12 Hz blue, 9 Hz orange shading)", loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax.grid(True, color="#ecebe5", lw=0.6)
        ax.set_axisbelow(True)
    # ocular rates lit vs dark per session
    ax = axes[1, 2]
    xs = np.arange(len(SESSIONS))
    for k, st in enumerate(("lit", "dark")):
        vals = [results[n]["lf_AF"]["12Hz_stages"]["ocular_rates_S1"][st]["ref_blinks_per_min"] for n, _ in SESSIONS]
        ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=(C_LIT if st == "lit" else C_DARK), label=f"{st}: reference blinks / min")
        vals2 = [results[n]["lf_AF"]["12Hz_stages"]["ocular_rates_S1"][st]["saccades_per_min"] for n, _ in SESSIONS]
        ax.plot(xs + (k - 0.5) * 0.36, vals2, "v", color=C_INK, ms=5, label=("saccades / min" if k == 0 else None))
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in SESSIONS])
    ax.set_ylabel("events per minute (12 Hz stages, S1)")
    ax.set_title("Ocular events: lit cell (blue) vs black (orange)", loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, axis="y", color="#ecebe5", lw=0.6)
    ax.set_axisbelow(True)
    # row 3: mean spectra 0.5-12 Hz lit vs dark, S1 and S3, tonight pooled 12 Hz stages; plus per-stage paired gaps
    for j, sname in enumerate(("S1_no_move", "S3_no_ocular")):
        ax = axes[2, j]
        for st, col in (("lit", C_LIT), ("dark", C_DARK)):
            acc, nn = [], 0
            for name, _ in SESSIONS[:2]:
                dfe = data[name]
                on_idx = dfe.index[(dfe["minutes"] >= 0) & dfe["state"].isin(["lit", "dark", "mixed", "idle"])]
                on = dfe.loc[on_idx].copy()
                S = subsets(on)
                sel = on.index[(np.abs(on["fstim"] - 12.0) < 0.5) & (on["state"] == st) & S[sname]]
                f, Pf = specs[name]["f"], specs[name]["Pf"]
                iaf = [data[name].columns.get_loc("lf_bp_AF7"), 0]
                names = ["TP9", "AF7", "AF8", "TP10", "AUX"]
                Paf = Pf[sel][:, :, [names.index("AF7"), names.index("AF8")]].mean(axis=2)
                acc.append(Paf)
                nn += len(sel)
            Pm = np.median(np.concatenate(acc), axis=0)
            band = (f >= 0.5) & (f <= 12.5)
            ax.plot(f[band], np.log10(Pm[band]), "-", lw=1.6, color=col, label=f"{st} (n {nn} epochs)")
        ax.axvspan(1, 4, color="#eeeeea", alpha=0.6, lw=0)
        ax.set_xlabel("Hz")
        ax.set_ylabel("log10 PSD, AF7/AF8 (median over epochs)")
        ax.set_title(f"Tonight, 12 Hz stages, {sname.replace('_', ' ')}: lit vs dark spectra", loc="left")
        ax.legend(fontsize=6.5, frameon=False)
        ax.grid(True, color="#ecebe5", lw=0.6)
        ax.set_axisbelow(True)
    ax = axes[2, 2]
    y = 0
    labs = []
    for name, label in SESSIONS:
        for st, r in results[name]["per_stage_S3_lf_bp"].items():
            p = r["paired"]
            if not np.isfinite(p["median"]):
                continue
            col = {"vr_planes9": "#4a3aa7", "vr_planes10": "#1baf7a", "vr_planes8": "#6f6e69"}[name]
            ax.errorbar(p["median"], y, xerr=[[p["median"] - p["ci"][0]], [p["ci"][1] - p["median"]]], fmt="o", ms=4, color=col, capsize=2, lw=1)
            labs.append(f"{label.split(' (')[0]} {st.replace('calib_blue_sweep', 'sweep').replace('calib', 'cal').replace('plane_', 'p.')}")
            y += 1
    ax.axvline(0, color=C_NONE, lw=1)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=6.5)
    ax.set_xlabel("paired block gap, lit - dark, log10 1-4 Hz power (S3)")
    ax.set_title("Per stage, ocular-clean paired blocks", loc="left")
    ax.grid(True, axis="x", color="#ecebe5", lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("Refutation check: is the frontal 1-4 Hz lit-minus-dark gap an ocular / statistical artifact?  "
                 "circles = epoch contrast, squares = paired adjacent blocks; 95 % bootstrap intervals", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, NAME + ".png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
