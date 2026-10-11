"""Flash confounds: what could mimic a frontal EEG sign of eye irritation or a
forehead ache, measured before anyone reads one into tonight's recordings.

Sessions (one subject): runs/vr_planes9 (run 1, 17:57) and runs/vr_planes10
(run 2, 19:50) of 2026-10-10, after which the subject reported irritated eyes
and a slight forehead ache but no drowsiness; runs/vr_planes8 (2026-10-07) is a
reference day with no symptom report, used as a level reference only.

Five families of confound, each per 2 s epoch (non-overlapping), per channel:

  1 contact      AF7 / AF8 mains pick-up (59-61 Hz amplitude and its ratio to
                 55-58 + 62-65 Hz), raw offset from mid-scale (the Muse ADC
                 centre, ~850) and its drift, broadband 1-40 Hz RMS, share of
                 flat (stuck value >= 0.25 s) or railed (0 / 1650) samples;
                 per minute, with Theil-Sen drift over the stimulation
  2 movement     epochs where every channel exceeds 150 uV peak after a 1-40 Hz
                 band-pass (head movement, headset adjustment) per 5 min, and
                 the share of blink-mask edges that fall inside such epochs
  3 order        the same markers in the last 3 min before the first flicker of
                 each run (eyes open, no flicker): did run 2 start different?
  4 leakage      AF7 / AF8 (and Oz) line-to-neighbour ratio at the stimulus
                 frequency and harmonics, lit cells against black cells, and
                 the frontal band powers with the lines left in versus cut out
  5 jaw EMG      TP9 / TP10 20-100 Hz power (mains and stimulus lines cut) as
                 the temporalis alternative to 'forehead tension', with its
                 drift, lit-dark contrast and correlation with AF7 / AF8 EMG

The lit / dark labelling here comes from the luminance column of each stage
log, not from state_of_flash.timeline, which marks the whole blue sweep
(grid_x = -1 throughout) as DARK and every blue-plane cell, lit or black, as
TAGGED.

Uncertainty: moving-block bootstrap over epochs (30 s blocks; 10 s in the 3 min
pre-windows) for levels, permutation over epochs for lit-dark and run-run
contrasts, Theil-Sen slopes with the Sen 95 % interval for drift, Wilcoxon
over matched minutes for run 1 versus run 2.

    python analysis/flash_confounds.py
      -> runs/state_of_flash/flash_confounds.json, runs/state_of_flash/flash_confounds.png
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blinkmask  # noqa: E402
from analysis import state_of_flash as sof  # noqa: E402

OUT = os.path.join(ROOT, "runs", "state_of_flash")
SESSIONS = [("vr_planes9", "run 1 (17:57)"), ("vr_planes10", "run 2 (19:50)"),
            ("vr_planes8", "reference 10-07")]
EPOCH_S = 2.0
ART_UV = 150.0
RAIL_LO, RAIL_HI = 0.5, 1649.5
MID_SCALE = 850.0
FLAT_S = 0.25
PRE_S = 180.0
MATCH_MIN = 21            # minutes on task shared by both runs
BLOCK_EPOCHS = 15         # 30 s blocks for the moving-block bootstrap
B_BOOT, B_PERM = 2000, 4000
FRONTAL = ["AF7", "AF8"]
EARS = ["TP9", "TP10"]
BANDS = {"lf": (1.0, 4.0), "theta": (4.0, 7.5), "alpha": (8.0, 12.5), "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}
C_LIT, C_DARK = "#2a78d6", "#eb6834"
C_RUN = {"vr_planes9": "#4a3aa7", "vr_planes10": "#1baf7a", "vr_planes8": "#6f6e69"}
C_INK = "#3a3a37"


# ----------------------------------------------------------------- statistics
def finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def block_boot(x, block=BLOCK_EPOCHS, B=B_BOOT, seed=0, stat=np.median):
    """95 % interval of stat(x) by a moving-block bootstrap over epochs."""
    x = finite(x)
    n = len(x)
    if n < 4:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    block = int(max(1, min(block, n // 3)))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    vals = stat(x[idx], axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def level(x, block=BLOCK_EPOCHS, seed=0):
    x = finite(x)
    if len(x) == 0:
        return dict(n=0, median=float("nan"), ci=[float("nan")] * 2)
    return dict(n=int(len(x)), median=float(np.median(x)), ci=block_boot(x, block, seed=seed))


def contrast(a, b, block=BLOCK_EPOCHS, seed=0):
    """median(a) - median(b) with a bootstrap interval and a permutation p (epochs exchangeable)."""
    a, b = finite(a), finite(b)
    if len(a) < 4 or len(b) < 4:
        return dict(n_a=int(len(a)), n_b=int(len(b)), diff=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    rng = np.random.default_rng(seed)
    obs = float(np.median(a) - np.median(b))

    def boot(x, s):
        n = len(x)
        bl = int(max(1, min(block, n // 3)))
        nb = int(np.ceil(n / bl))
        st = np.random.default_rng(s).integers(0, n - bl + 1, size=(B_BOOT, nb))
        idx = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(B_BOOT, -1)[:, :n]
        return np.median(x[idx], axis=1)
    d = boot(a, seed + 1) - boot(b, seed + 2)
    pooled = np.concatenate([a, b])
    perm = rng.permuted(np.tile(pooled, (B_PERM, 1)), axis=1)
    dp = np.median(perm[:, :len(a)], axis=1) - np.median(perm[:, len(a):], axis=1)
    p = float((np.sum(np.abs(dp) >= abs(obs)) + 1) / (B_PERM + 1))
    return dict(n_a=int(len(a)), n_b=int(len(b)), diff=obs,
                ci=[float(v) for v in np.percentile(d, [2.5, 97.5])], p=p)


def theil(minutes, y):
    """Theil-Sen slope per 10 min with the Sen 95 % interval."""
    m, y = np.asarray(minutes, float), np.asarray(y, float)
    ok = np.isfinite(m) & np.isfinite(y)
    if ok.sum() < 30 or (m[ok].max() - m[ok].min()) < 4:
        return dict(n=int(ok.sum()), slope_per_10min=float("nan"), ci=[float("nan")] * 2)
    s, _, lo, hi = stats.theilslopes(y[ok], m[ok], alpha=0.95)
    return dict(n=int(ok.sum()), slope_per_10min=float(s * 10), ci=[float(lo * 10), float(hi * 10)],
                span_min=float(m[ok].max() - m[ok].min()))


def matched(pm1, pm2, col, minutes=range(MATCH_MIN), seed=0):
    """run 2 - run 1 over per-minute medians at the same minute on task."""
    a = pm1.reindex(minutes)[col].to_numpy(float)
    b = pm2.reindex(minutes)[col].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[ok]
    if len(d) < 6:
        return dict(n_minutes=int(len(d)), diff=float("nan"), ci=[float("nan")] * 2, p_wilcoxon=float("nan"))
    rng = np.random.default_rng(seed)
    bs = np.median(d[rng.integers(0, len(d), size=(B_BOOT, len(d)))], axis=1)
    if np.all(d == 0):
        p = float("nan")            # identical every minute: nothing to test
    else:
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = float("nan")
    return dict(n_minutes=int(len(d)), diff=float(np.median(d)),
                ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], p_wilcoxon=p,
                run1_median=float(np.nanmedian(a[ok])), run2_median=float(np.nanmedian(b[ok])))


# ----------------------------------------------------------------- labelling
def labels(d, t, f0):
    """Per EEG sample: lit (luminance > 0.05), stage index, stimulus frequency
    (12, 9, -1 = the blue sweep with 6 / 7.2 / 9 / 12 Hz blocks, 0 = none)."""
    n = len(t)
    lit = np.zeros(n, bool)
    stage = np.full(n, -1, np.int16)
    fstim = np.zeros(n, np.float32)
    names, spans = [], {}
    plane_hz = {}
    try:
        pm = json.load(open(os.path.join(d, "planes_meta.json")))
        plane_hz = {str(k): float(v) for k, v in (pm.get("freq_delivered") or {}).items()}
    except Exception:
        pass
    first, last = np.inf, -np.inf
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        L = pd.read_csv(p)
        if len(L) < 10 or "luminance" not in L.columns:
            continue
        lt = L["time"].to_numpy(float)
        lum = L["luminance"].to_numpy(float)
        order = np.argsort(lt)
        lt, lum = lt[order], lum[order]
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5)
        sid = len(names)
        names.append(kind)
        stage[ok] = sid
        lit[ok] = lum[idx[ok]] > 0.05
        if kind.endswith("_sweep"):
            f = -1.0
        elif kind.startswith("plane_"):
            f = plane_hz.get(kind.split("_")[1], f0)
        else:
            f = f0
        fstim[ok] = f
        spans[kind] = [float(lt[0]), float(lt[-1]), float(f)]
        first, last = min(first, lt[0]), max(last, lt[-1])
    return lit, stage, fstim, names, spans, first, last


def line_cut(f, fcode, f0):
    """Bins to leave out of a band: stimulus line, harmonics, subharmonic, mains."""
    cut = np.abs(f - 60.0) <= 2.0
    sets = []
    if fcode > 0:
        base = f0 if abs(fcode - 12.0) < 0.5 else fcode
        sets.append(base)
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


# ----------------------------------------------------------------- epochs
def epochs(d):
    t, X, names, fs = sof.load_eeg(d)
    f0, _ = sof.stim_freq(d)
    lit, stage, fstim, stage_names, spans, first, last = labels(d, t, f0)
    n = int(round(EPOCH_S * fs))
    starts = np.arange(0, len(t) - n, n)
    sos = butter(2, [1.0, 40.0], "bandpass", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    blink = blinkmask.blink_mask(X, names, fs, 100.0)
    edges = np.zeros(len(t), bool)
    edges[1:] = np.diff(blink.astype(int)) == 1
    # the same detector at lower thresholds (the Muse reference sits at Fpz, so a
    # blink at AF7 / AF8 is a difference of two nearby frontal sites and small)
    edges_thr = {}
    for thr in (30.0, 50.0):
        b = blinkmask.blink_mask(X, names, fs, thr)
        e = np.zeros(len(t), bool)
        e[1:] = np.diff(b.astype(int)) == 1
        edges_thr[thr] = e
    # how state_of_flash.timeline would have labelled the same samples
    sof_state, _, _, _ = sof.timeline(d, t, f0)
    sweep = np.array([i for i, nm in enumerate(stage_names) if nm.endswith("_sweep")])
    blue = np.array([i for i, nm in enumerate(stage_names) if nm == "plane_blue"])
    in_sweep = np.isin(stage, sweep) if len(sweep) else np.zeros(len(t), bool)
    in_blue = np.isin(stage, blue) if len(blue) else np.zeros(len(t), bool)
    timeline_check = dict(
        sweep_lit_labelled_dark_min=float((in_sweep & lit & (sof_state == sof.DARK)).sum() / fs / 60),
        sweep_total_min=float(in_sweep.sum() / fs / 60),
        blue_plane_black_labelled_tagged_min=float((in_blue & ~lit & (sof_state == sof.TAGGED)).sum() / fs / 60),
        blue_plane_total_min=float(in_blue.sum() / fs / 60),
    )
    rail = (X <= RAIL_LO) | (X >= RAIL_HI)
    flat = np.stack([flat_mask(X[:, i], fs) for i in range(X.shape[1])], axis=1)

    seg = np.stack([X[s:s + n] for s in starts])
    segf = np.stack([Xf[s:s + n] for s in starts])
    segf = segf - segf.mean(axis=1, keepdims=True)
    f, P = welch(seg - seg.mean(axis=1, keepdims=True), fs=fs, nperseg=n, axis=1)   # (E, F, ch) uV^2/Hz
    df_hz = f[1] - f[0]
    E = len(starts)
    out = dict(t=t[starts] + EPOCH_S / 2)
    out["minutes"] = (out["t"] - first) / 60.0
    # what was on screen
    sl = np.stack([stage[s:s + n] for s in starts])
    ll = np.stack([lit[s:s + n] for s in starts])
    fl = np.stack([fstim[s:s + n] for s in starts])
    maj = np.array([np.bincount(r + 1, minlength=len(stage_names) + 1).argmax() - 1 for r in sl])
    pure_stage = np.array([(r == m).mean() for r, m in zip(sl, maj)])
    lit_share = ll.mean(axis=1)
    out["stage"] = maj
    out["stage_name"] = [stage_names[m] if m >= 0 else "" for m in maj]
    out["fstim"] = np.array([np.median(r[r != 0]) if (r != 0).any() else 0.0 for r in fl])
    state = np.where(lit_share >= 0.9, "lit", np.where(lit_share <= 0.1, "dark", "mixed"))
    state = np.where((maj < 0) | (pure_stage < 0.9), "idle", state)
    tm = out["t"]
    state = np.where(tm < first - PRE_S, "wait", np.where(tm < first, "pre", state))
    state = np.where(tm > last + 1.0, "post", state)
    out["state"] = state
    # movement: 1-40 Hz peak per channel
    peak = np.abs(segf).max(axis=1)
    out["move_all"] = (peak > ART_UV).all(axis=1)
    out["move_any"] = (peak > ART_UV).any(axis=1)
    out["blink_edges"] = np.array([edges[s:s + n].sum() for s in starts])
    for thr, e in edges_thr.items():
        out[f"blink_edges_{int(thr)}"] = np.array([e[s:s + n].sum() for s in starts])
    mains = np.abs(f - 60.0) <= 0.5
    # neighbours of the mains line, without the 63 Hz harmonic of the 9 Hz plane
    nb60 = (((f >= 55) & (f <= 58)) | ((f >= 62) & (f <= 65))) & ~(np.abs(f - 63.0) <= 0.75)
    for i, ch in enumerate(names):
        out[f"dc_{ch}"] = np.median(seg[:, :, i], axis=1) - MID_SCALE
        out[f"rail_{ch}"] = np.array([rail[s:s + n, i].mean() for s in starts])
        out[f"flat_{ch}"] = np.array([flat[s:s + n, i].mean() for s in starts])
        out[f"bad_{ch}"] = (out[f"rail_{ch}"] > 0) | (out[f"flat_{ch}"] > 0) | (peak[:, i] > ART_UV)
        out[f"rms_{ch}"] = segf[:, :, i].std(axis=1)
        out[f"peak_{ch}"] = peak[:, i]
        out[f"l60_uv_{ch}"] = np.sqrt(P[:, mains, i].sum(axis=1) * df_hz)
        out[f"l60_ratio_{ch}"] = np.log10(P[:, mains, i].mean(axis=1) / P[:, nb60, i].mean(axis=1))
        # EMG 20-100 Hz with mains and all stimulus lines cut (per epoch's own stimulus)
        emg = np.full(E, np.nan)
        for code in np.unique(out["fstim"]):
            sel = out["fstim"] == code
            keep = (f >= 20) & (f <= 100) & ~line_cut(f, code, f0)
            emg[sel] = np.log10(P[sel][:, keep, i].mean(axis=1))
        out[f"emg_{ch}"] = emg
        # the same with a wider mains cut (60 +- 5 Hz): if the two agree, the
        # mains line is not leaking through the Hann window into the EMG band
        emgw = np.full(E, np.nan)
        for code in np.unique(out["fstim"]):
            sel = out["fstim"] == code
            keep = (f >= 20) & (f <= 100) & ~line_cut(f, code, f0) & ~(np.abs(f - 60.0) <= 5.0)
            emgw[sel] = np.log10(P[sel][:, keep, i].mean(axis=1))
        out[f"emgw_{ch}"] = emgw
        if ch in FRONTAL or blinkmask.oz_index([ch]) is not None:
            for bname, (lo, hi) in BANDS.items():
                inb = (f >= lo) & (f <= hi)
                raw_in = np.log10(P[:, inb, i].mean(axis=1))
                cutv = np.full(E, np.nan)
                for code in np.unique(out["fstim"]):
                    sel = out["fstim"] == code
                    keep = inb & ~line_cut(f, code, f0)
                    if keep.sum() == 0:
                        keep = inb
                    cutv[sel] = np.log10(P[sel][:, keep, i].mean(axis=1))
                out[f"{bname}_in_{ch}"] = raw_in
                out[f"{bname}_cut_{ch}"] = cutv
            # line-to-neighbour ratio at k x stimulus, k = 1..4
            for k in range(1, 5):
                snr = np.full(E, np.nan)
                for code in (9.0, 12.0):
                    sel = np.abs(out["fstim"] - code) < 0.5
                    if not sel.any():
                        continue
                    base = f0 if code == 12.0 else code
                    fl_k = k * base
                    if fl_k > 100:
                        continue
                    line = np.abs(f - fl_k) <= 0.5
                    nb = (np.abs(f - fl_k) > 1.0) & (np.abs(f - fl_k) <= 3.0) & ~line_cut(f, code, f0)
                    snr[sel] = np.log10(P[sel][:, line, i].mean(axis=1) / np.median(P[sel][:, nb, i], axis=1))
                out[f"snr{k}_{ch}"] = snr
    dfe = pd.DataFrame(out)
    dfe["minute"] = np.floor(dfe["minutes"]).astype(int)
    dfe["af_bad"] = dfe["bad_AF7"] | dfe["bad_AF8"]
    dfe["ear_bad"] = dfe["bad_TP9"] | dfe["bad_TP10"]
    for col in ("l60_uv", "dc", "rms", "flat", "rail", "emg"):
        dfe[f"{col}_AF"] = dfe[[f"{col}_AF7", f"{col}_AF8"]].mean(axis=1)
    dfe["flatrail_AF"] = dfe[["flat_AF7", "flat_AF8", "rail_AF7", "rail_AF8"]].max(axis=1)
    dfe["emg_TP"] = dfe[["emg_TP9", "emg_TP10"]].mean(axis=1)
    dfe["emg_uv_TP"] = np.sqrt(10 ** dfe["emg_TP"] * 80.0)
    dfe["emg_uv_AF"] = np.sqrt(10 ** dfe["emg_AF"] * 80.0)
    for b in BANDS:
        dfe[f"{b}_in_AF"] = dfe[[f"{b}_in_AF7", f"{b}_in_AF8"]].mean(axis=1)
        dfe[f"{b}_cut_AF"] = dfe[[f"{b}_cut_AF7", f"{b}_cut_AF8"]].mean(axis=1)
    for k in range(1, 5):
        dfe[f"snr{k}_AF"] = dfe[[f"snr{k}_AF7", f"snr{k}_AF8"]].mean(axis=1)
    meta = dict(names=names, fs=float(fs), f0=float(f0), first=float(first), last=float(last),
                t0=float(t[0]), t1=float(t[-1]), stages=spans, n_epochs=int(E),
                timeline_check=timeline_check,
                sof_timeline_note=("state_of_flash.timeline labels the blue sweep (grid_x = -1) DARK "
                                   "throughout and every blue-plane cell TAGGED; lit/dark here comes from "
                                   "the luminance column"))
    spec = dict(f=f, P=P)
    return dfe, meta, spec


# ----------------------------------------------------------------- per-session analysis
def per_minute(dfe):
    on = dfe[dfe["minutes"] >= 0].copy()
    on = on[on["state"].isin(["lit", "dark", "mixed", "idle"])]
    on["l60_ratio_AF"] = on[["l60_ratio_AF7", "l60_ratio_AF8"]].mean(axis=1)
    on["emg_uv_TP_ok"] = on["emg_uv_TP"].where(~on["ear_bad"])
    on["emg_uv_AF_ok"] = on["emg_uv_AF"].where(~on["af_bad"])
    g = on.groupby("minute")
    pm = pd.DataFrame({
        "l60_uv_AF": g["l60_uv_AF"].median(),
        "l60_ratio_AF": g["l60_ratio_AF"].median(),
        "dc_AF7": g["dc_AF7"].median(), "dc_AF8": g["dc_AF8"].median(),
        "rms_AF": g["rms_AF"].median(),
        "flatrail_AF": g["flatrail_AF"].mean(),
        "move_all": g["move_all"].sum(), "move_any": g["move_any"].sum(),
        "emg_uv_TP": g["emg_uv_TP_ok"].median().where(g["emg_uv_TP_ok"].count() >= 5),
        "emg_uv_AF": g["emg_uv_AF_ok"].median().where(g["emg_uv_AF_ok"].count() >= 5),
        "blinks_per_min": g["blink_edges"].sum() / (g.size() * EPOCH_S) * 60.0,
        "n": g.size(),
    })
    return pm


def analyse(name, dfe, meta, spec):
    res = dict(label=dict(SESSIONS)[name], first=meta["first"], last=meta["last"],
               stim_minutes=round((meta["last"] - meta["first"]) / 60, 1), fs=round(meta["fs"], 2),
               stages=meta["stages"], n_epochs=meta["n_epochs"])
    on = dfe[(dfe["minutes"] >= 0) & dfe["state"].isin(["lit", "dark", "mixed", "idle"])]
    pre = dfe[dfe["state"] == "pre"]
    pm = per_minute(dfe)
    res["per_minute"] = {k: [None if not np.isfinite(v) else float(v) for v in pm[k].to_numpy(float)] for k in pm.columns}
    res["per_minute"]["minute"] = [int(m) for m in pm.index]

    # 1 contact
    c = {}
    for ch in FRONTAL + EARS + ["AUX"]:
        c[ch] = dict(
            l60_uv=level(on[f"l60_uv_{ch}"]), l60_ratio_log10=level(on[f"l60_ratio_{ch}"]),
            dc_offset_uv=level(on[f"dc_{ch}"]), rms_1_40_uv=level(on[f"rms_{ch}"]),
            flat_share=float(on[f"flat_{ch}"].mean()), rail_share=float(on[f"rail_{ch}"].mean()),
            epochs_over_150uV=int((on[f"peak_{ch}"] > ART_UV).sum()),
            drift_l60_uv=theil(on["minutes"], on[f"l60_uv_{ch}"]),
            drift_dc_uv=theil(on["minutes"], on[f"dc_{ch}"]),
            drift_rms_uv=theil(on["minutes"], on[f"rms_{ch}"]),
            dc_range_uv=[float(pm[f"dc_{ch}"].min()), float(pm[f"dc_{ch}"].max())] if f"dc_{ch}" in pm else None,
            pre_window=dict(l60_uv=level(pre[f"l60_uv_{ch}"], 5), dc_offset_uv=level(pre[f"dc_{ch}"], 5),
                            rms_1_40_uv=level(pre[f"rms_{ch}"], 5), flat_share=float(pre[f"flat_{ch}"].mean()) if len(pre) else float("nan"),
                            rail_share=float(pre[f"rail_{ch}"].mean()) if len(pre) else float("nan")),
        )
    res["contact"] = c

    # 2 movement
    on5 = on.assign(bin5=(on["minutes"] // 5).astype(int))
    g5 = on5.groupby("bin5")
    res["movement"] = dict(
        per_5min_all=[int(v) for v in g5["move_all"].sum()], per_5min_any=[int(v) for v in g5["move_any"].sum()],
        bins_5min=[int(b) for b in g5.size().index],
        all_per_min_on_task=float(on["move_all"].sum() / max(1e-9, len(on) * EPOCH_S / 60)),
        any_per_min_on_task=float(on["move_any"].sum() / max(1e-9, len(on) * EPOCH_S / 60)),
        share_epochs_any=float(on["move_any"].mean()), share_epochs_all=float(on["move_all"].mean()),
        pre_window_all=int(pre["move_all"].sum()), pre_window_any=int(pre["move_any"].sum()),
        blink_edges_inside_move_any=int(on.loc[on["move_any"], "blink_edges"].sum()),
        blink_edges_total=int(on["blink_edges"].sum()),
        by_state={s: dict(epochs=int((on["state"] == s).sum()), move_any=int(on.loc[on["state"] == s, "move_any"].sum()))
                  for s in ("lit", "dark", "mixed", "idle")},
    )

    # 3 pre-window levels (eyes open, no flicker, last 3 min)
    pre_ok = pre[~pre["af_bad"]]
    pre_ear = pre[~pre["ear_bad"]]
    res["pre_window"] = dict(
        epochs=int(len(pre)), af_ok=int(len(pre_ok)), ear_ok=int(len(pre_ear)),
        l60_uv_AF=level(pre["l60_uv_AF"], 5), dc_AF7=level(pre["dc_AF7"], 5), dc_AF8=level(pre["dc_AF8"], 5),
        rms_AF=level(pre["rms_AF"], 5), flatrail_AF=float(pre["flatrail_AF"].mean()) if len(pre) else float("nan"),
        blinks_per_min=float(pre["blink_edges"].sum() / max(1e-9, len(pre) * EPOCH_S) * 60) if len(pre) else float("nan"),
        emg_uv_AF=level(pre_ok["emg_uv_AF"], 5), emg_uv_TP=level(pre_ear["emg_uv_TP"], 5),
        lf_AF=level(pre_ok["lf_cut_AF"], 5), theta_AF=level(pre_ok["theta_cut_AF"], 5),
        alpha_AF=level(pre_ok["alpha_cut_AF"], 5), beta_AF=level(pre_ok["beta_cut_AF"], 5),
        move_any=int(pre["move_any"].sum()), move_all=int(pre["move_all"].sum()),
    )

    # 4 leakage: lit vs dark at the stimulus frequency, frontal and Oz
    leak = {}
    for code, tag in ((12.0, "12Hz_stages"), (9.0, "9Hz_blue_plane")):
        sel = on[(np.abs(on["fstim"] - code) < 0.5) & on["state"].isin(["lit", "dark"])]
        if len(sel) < 20 or (sel["state"] == "lit").sum() < 10 or (sel["state"] == "dark").sum() < 10:
            continue
        L = {}
        for ch in ("AF", "AF7", "AF8", "AUX"):
            okc = ~sel["af_bad"] if ch.startswith("AF") else ~sel["bad_AUX"]
            s = sel[okc]
            lit_, dark_ = s[s["state"] == "lit"], s[s["state"] == "dark"]
            d = dict(n_lit=int(len(lit_)), n_dark=int(len(dark_)))
            for k in range(1, 5):
                col = f"snr{k}_{ch}"
                if col in s:
                    d[f"snr{k}_lit_minus_dark"] = contrast(lit_[col], dark_[col], 5, seed=k)
                    d[f"snr{k}_lit"] = level(lit_[col], 5)
                    d[f"snr{k}_dark"] = level(dark_[col], 5)
            if ch in ("AF", "AUX"):
                for b in BANDS:
                    d[f"{b}_lit_minus_dark_lines_in"] = contrast(lit_[f"{b}_in_{ch}"], dark_[f"{b}_in_{ch}"], 5, seed=7)
                    d[f"{b}_lit_minus_dark_lines_cut"] = contrast(lit_[f"{b}_cut_{ch}"], dark_[f"{b}_cut_{ch}"], 5, seed=7)
            L[ch] = d
        leak[tag] = L
    res["leakage"] = leak

    # 5 jaw / temporalis EMG on TP9 / TP10
    ear_ok = on[~on["ear_bad"]]
    af_ok = on[~on["af_bad"]]
    both = on[~on["ear_bad"] & ~on["af_bad"]]
    lit_e, dark_e = ear_ok[ear_ok["state"] == "lit"], ear_ok[ear_ok["state"] == "dark"]
    emg = dict(
        epochs_kept=int(len(ear_ok)), epochs_rejected_ear=int(on["ear_bad"].sum()),
        level_uv_TP=level(ear_ok["emg_uv_TP"]), level_uv_AF=level(af_ok["emg_uv_AF"]),
        drift_TP_log10_per_10min=theil(ear_ok["minutes"], ear_ok["emg_TP"]),
        drift_AF_log10_per_10min=theil(af_ok["minutes"], af_ok["emg_AF"]),
        lit_minus_dark_TP_log10=contrast(lit_e["emg_TP"], dark_e["emg_TP"], 5, seed=11),
        lit_minus_dark_AF_log10=contrast(af_ok.loc[af_ok["state"] == "lit", "emg_AF"], af_ok.loc[af_ok["state"] == "dark", "emg_AF"], 5, seed=12),
        pre_level_uv_TP=level(pre_ear["emg_uv_TP"], 5), pre_level_uv_AF=level(pre_ok["emg_uv_AF"], 5),
    )
    emg["by_channel"] = {}
    for ch in EARS + FRONTAL:
        okc = on[~on[f"bad_{ch}"]]
        emg["by_channel"][ch] = dict(
            level_uv=level(np.sqrt(10 ** okc[f"emg_{ch}"] * 80.0)),
            drift_log10_per_10min=theil(okc["minutes"], okc[f"emg_{ch}"]),
            epochs_rejected=int(on[f"bad_{ch}"].sum()),
            mains_cut_check_log10=float(np.nanmedian(okc[f"emgw_{ch}"] - okc[f"emg_{ch}"])),
            lit_minus_dark_log10=contrast(okc.loc[okc["state"] == "lit", f"emg_{ch}"],
                                          okc.loc[okc["state"] == "dark", f"emg_{ch}"], 5, seed=13),
        )
    # the 3 min before the first flicker in 10 s bins: is any of it a quiet eyes-open baseline?
    pre10 = pre.assign(bin10=np.floor((pre["t"] - res["first"]) / 10.0).astype(int)).groupby("bin10")
    res["pre_window_10s"] = dict(
        seconds_before=[int(-b * 10) for b in pre10.size().index],
        rms_AF_uv=[float(v) for v in pre10["rms_AF"].median()],
        l60_uv_AF=[float(v) for v in pre10["l60_uv_AF"].median()],
        move_any_share=[float(v) for v in pre10["move_any"].mean()],
        move_all_share=[float(v) for v in pre10["move_all"].mean()],
        quiet_bins=int(((pre10["move_any"].mean() == 0) & (pre10["rms_AF"].median() < 2 * np.median(on["rms_AF"]))).sum()),
        n_bins=int(pre10.ngroups),
        on_task_rms_AF_median=float(np.median(on["rms_AF"])),
    )
    # blink-edge rates at three thresholds, per state on task and in the pre-window
    bt = {}
    for thr, col in ((100, "blink_edges"), (50, "blink_edges_50"), (30, "blink_edges_30")):
        d = {}
        for st in ("lit", "dark", "mixed"):
            sub = on[on["state"] == st]
            d[st] = float(sub[col].sum() / max(1e-9, len(sub) * EPOCH_S) * 60) if len(sub) else float("nan")
        d["on_task"] = float(on[col].sum() / max(1e-9, len(on) * EPOCH_S) * 60)
        d["pre_window"] = float(pre[col].sum() / max(1e-9, len(pre) * EPOCH_S) * 60) if len(pre) else float("nan")
        d["on_task_inside_move_any"] = int(on.loc[on["move_any"], col].sum())
        d["on_task_total"] = int(on[col].sum())
        bt[f"{thr}uV"] = d
    res["blink_edges_per_min"] = bt
    if len(both) >= 30:
        rho, p = stats.spearmanr(both["emg_TP"], both["emg_AF"])
        emg["spearman_TP_vs_AF_epochs"] = dict(rho=float(rho), p=float(p), n=int(len(both)))
        rho_b, p_b = stats.spearmanr(both["emg_TP"], both["beta_cut_AF"])
        emg["spearman_TP_emg_vs_AF_beta_epochs"] = dict(rho=float(rho_b), p=float(p_b), n=int(len(both)))
        pmm = pm.dropna(subset=["emg_uv_TP", "emg_uv_AF"])
        if len(pmm) >= 8:
            rho_m, p_m = stats.spearmanr(pmm["emg_uv_TP"], pmm["emg_uv_AF"])
            emg["spearman_TP_vs_AF_minutes"] = dict(rho=float(rho_m), p=float(p_m), n=int(len(pmm)))
    res["emg"] = emg
    return res, pm


# ----------------------------------------------------------------- figure
def figure(data, results, pms, specs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK,
                         "xtick.color": C_INK, "ytick.color": C_INK, "text.color": C_INK,
                         "axes.titlesize": 8.5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig = plt.figure(figsize=(12.5, 16))
    gs = GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.22, top=0.955, bottom=0.035, left=0.07, right=0.98)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(5)]

    def style(ax):
        ax.grid(True, color="#ecebe5", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def minute_line(ax, col, scale=1.0, log=False):
        for name, lab in SESSIONS:
            dfe = data[name]
            pre = dfe[dfe["state"] == "pre"].groupby("minute")[col].median()
            pm = pms[name]
            x = np.concatenate([pre.index.to_numpy(float) + 0.5, pm.index.to_numpy(float) + 0.5])
            y = np.concatenate([pre.to_numpy(float), pm[col].to_numpy(float)]) * scale
            ax.plot(x, y, "-", lw=1.6, color=C_RUN[name], label=lab, marker="o", ms=2.5)
        if log:
            ax.set_yscale("log")
        ax.axvspan(-3, 0, color="#ecebe5", alpha=0.7, lw=0)
        ax.text(-1.5, ax.get_ylim()[1], "pre", ha="center", va="top", fontsize=7, color="#6f6e69")
        ax.set_xlabel("minutes since the first flicker")
        style(ax)

    # row 0: mains pick-up and offset
    ax = axes[0][0]
    minute_line(ax, "l60_uv_AF", log=True)
    ax.set_ylabel("60 Hz amplitude, AF7/AF8 (uV rms, log)")
    ax.set_title("1  Mains pick-up on the forehead electrodes, per minute (median of 2 s epochs)", loc="left")
    ax = axes[0][1]
    for name, lab in SESSIONS:
        pm = pms[name]
        dfe = data[name]
        pre = dfe[dfe["state"] == "pre"].groupby("minute")[["dc_AF7", "dc_AF8"]].median()
        x = np.concatenate([pre.index.to_numpy(float) + 0.5, pm.index.to_numpy(float) + 0.5])
        ax.plot(x, np.concatenate([pre["dc_AF7"], pm["dc_AF7"]]), "-", lw=1.6, color=C_RUN[name], label=f"{lab} AF7")
        ax.plot(x, np.concatenate([pre["dc_AF8"], pm["dc_AF8"]]), "--", lw=1.2, color=C_RUN[name], label=f"{lab} AF8")
    ax.axvspan(-3, 0, color="#ecebe5", alpha=0.7, lw=0)
    ax.axhline(0, color="#b5b4ad", lw=0.8)
    ax.set_ylabel("raw offset from mid-scale (uV)")
    ax.set_xlabel("minutes since the first flicker")
    ax.set_title("1  Electrode offset: AF7 solid, AF8 dashed (drift = contact change)", loc="left")
    style(ax)
    # row 1: broadband RMS and flat/rail share
    ax = axes[1][0]
    minute_line(ax, "rms_AF", log=True)
    ax.set_ylabel("1-40 Hz RMS, AF7/AF8 (uV, log)")
    ax.set_title("1  Broadband 1-40 Hz level on AF7/AF8 (movement epochs included)", loc="left")
    ax = axes[1][1]
    # which channel is bad when: % of 2 s epochs over 150 uV (1-40 Hz) per 5 min, per channel and session
    rows, labels_r, nbins = [], [], 0
    for name, lab in SESSIONS:
        dfe = data[name]
        on = dfe[(dfe["minutes"] >= 0) & dfe["state"].isin(["lit", "dark", "mixed", "idle"])]
        b5 = (on["minutes"] // 5).astype(int)
        for ch in results[name]["channels"]:
            r = (on[f"peak_{ch}"] > ART_UV).groupby(b5).mean() * 100
            rows.append(r)
            labels_r.append(f"{lab.split(' (')[0]}  {ch}")
            nbins = max(nbins, int(r.index.max()) + 1)
    M = np.full((len(rows), nbins), np.nan)
    for i, r in enumerate(rows):
        M[i, r.index.to_numpy()] = r.to_numpy()
    im = ax.imshow(M, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#3987e5", "#0d366b"]),
                   vmin=0, vmax=max(1.0, np.nanmax(M)))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.55 * np.nanmax(M) else C_INK)
    ax.set_yticks(range(len(labels_r)))
    ax.set_yticklabels(labels_r, fontsize=6.5)
    ax.set_xticks(range(nbins))
    ax.set_xticklabels([f"{5 * j}-{5 * j + 5}" for j in range(nbins)])
    ax.set_xlabel("minutes since the first flicker")
    ax.set_title("1  Epochs over 150 uV peak (1-40 Hz), % per 5 min: which channel was loose", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # row 2: movement per 5 min; pre-window contrast forest
    ax = axes[2][0]
    width = 0.27
    for j, (name, lab) in enumerate(SESSIONS):
        mv = results[name]["movement"]
        b = np.array(mv["bins_5min"], float)
        ax.bar(b * 5 + 2.5 + (j - 1) * width * 5 * 0.6, mv["per_5min_all"], width=width * 5 * 0.6,
               color=C_RUN[name], label=lab, lw=0)
        ax.plot(b * 5 + 2.5 + (j - 1) * width * 5 * 0.6, mv["per_5min_any"], "_", ms=9, mew=1.4, color=C_RUN[name])
    ax.set_xlabel("minutes since the first flicker (5 min bins)")
    ax.set_ylabel("2 s epochs")
    ax.set_title("2  Movement: bars = all five channels > 150 uV at once, ticks = any channel", loc="left")
    style(ax)
    ax = axes[2][1]
    # the 3 min before the first flicker, 10 s bins: AF7/AF8 1-40 Hz RMS against the on-task level
    for name, lab in SESSIONS:
        p10 = results[name]["pre_window_10s"]
        x = -np.array(p10["seconds_before"], float) / 60.0
        ax.plot(x, p10["rms_AF_uv"], "-o", ms=3, lw=1.5, color=C_RUN[name], label=lab)
        ax.axhline(p10["on_task_rms_AF_median"], color=C_RUN[name], lw=0.9, ls="--")
        mv = np.array(p10["move_any_share"])
        ax.plot(x[mv > 0], np.array(p10["rms_AF_uv"])[mv > 0], "x", ms=5, mew=1.2, color=C_RUN[name])
    ax.set_yscale("log")
    ax.set_xlabel("minutes before the first flicker")
    ax.set_ylabel("1-40 Hz RMS, AF7/AF8 (uV, log)")
    ax.set_title("3  Pre-flicker 3 min in 10 s bins: x = a channel over 150 uV, dashed = on-task level", loc="left")
    style(ax)
    # row 3: leakage spectra lit vs dark
    for j, (code, title, lines) in enumerate(((12.0, "12 Hz stages", [6, 12, 24, 36, 48]),
                                              (9.0, "9 Hz blue plane", [4.5, 9, 18, 27, 36, 45]))):
        ax = axes[3][j]
        f = None
        acc = {"lit": [], "dark": []}
        for name, _ in SESSIONS[:2]:
            dfe, sp = data[name], specs[name]
            f = sp["f"]
            sel = (np.abs(dfe["fstim"] - code) < 0.5) & ~dfe["af_bad"] & (dfe["minutes"] >= 0)
            for st in ("lit", "dark"):
                idx = np.flatnonzero(sel & (dfe["state"] == st))
                if len(idx):
                    ch = [i for i, nm in enumerate(results[name]["channels"]) if nm in FRONTAL]
                    acc[st].append(sp["P"][idx][:, :, ch].mean(axis=2))
        if f is None or not acc["lit"] or not acc["dark"]:
            ax.axis("off")
            continue
        band = (f >= 1) & (f <= 50)
        for st, col, lab in (("lit", C_LIT, "lit cell / ON block"), ("dark", C_DARK, "black cell / OFF block")):
            Pm = np.concatenate(acc[st]).mean(axis=0)
            n = sum(len(a) for a in acc[st])
            ax.plot(f[band], Pm[band], "-", lw=1.6, color=col, label=f"{lab} (n = {n} epochs)")
        for fl in lines:
            ax.axvline(fl, color="#b5b4ad", lw=0.7, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("AF7/AF8 power (uV^2/Hz)")
        ax.set_title(f"4  Leakage: AF7/AF8 spectrum, {title}, both runs (dotted = stimulus lines)", loc="left")
        ax.legend(frameon=False, fontsize=7)
        style(ax)
    # row 4: EMG per minute, TP vs AF
    ax = axes[4][0]
    minute_line(ax, "emg_uv_TP", log=True)
    ax.set_ylabel("20-100 Hz amplitude, TP9/TP10 (uV rms, log)")
    ax.set_title("5  Jaw / temporalis EMG behind the ears (mains and stimulus lines cut)", loc="left")
    ax = axes[4][1]
    for name, lab in SESSIONS:
        pm = pms[name].dropna(subset=["emg_uv_TP", "emg_uv_AF"])
        ax.plot(pm["emg_uv_TP"], pm["emg_uv_AF"], "o", ms=4.5, mfc="none", mec=C_RUN[name], mew=1.1, label=lab)
    for k, (name, _) in enumerate(SESSIONS):
        sp = results[name]["emg"].get("spearman_TP_vs_AF_minutes")
        if sp:
            ax.text(0.98, 0.04 + 0.07 * k, f"{dict(SESSIONS)[name]}: Spearman rho {sp['rho']:+.2f} (p {sp['p']:.3f}, {sp['n']} min)",
                    transform=ax.transAxes, fontsize=7, va="bottom", ha="right", color=C_RUN[name],
                    bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5))
    ax.set_xlabel("TP9/TP10 20-100 Hz (uV rms), per minute")
    ax.set_ylabel("AF7/AF8 20-100 Hz (uV rms), per minute")
    ax.set_title("5  Forehead high-frequency power against the ear EMG, minute by minute", loc="left")
    style(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.06, 0.985))
    fig.suptitle("Flash confounds: contact, movement, order, stimulus leakage and jaw EMG that could mimic a frontal sign "
                 "of eye irritation or forehead ache (2026-10-10, one subject)", x=0.01, y=0.997, ha="left", fontsize=9.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    data, results, pms, specs = {}, {}, {}, {}
    for name, lab in SESSIONS:
        d = os.path.join(ROOT, "runs", name)
        dfe, meta, spec = epochs(d)
        res, pm = analyse(name, dfe, meta, spec)
        res["channels"] = meta["names"]
        res["note_timeline"] = meta["sof_timeline_note"]
        res["timeline_check"] = meta["timeline_check"]
        data[name], results[name], pms[name], specs[name] = dfe, res, pm, spec
        c = res["contact"]
        print(f"{name:12s} {lab:16s} stim {res['stim_minutes']:5.1f} min  AF 60 Hz {c['AF7']['l60_uv']['median']:.2f}/{c['AF8']['l60_uv']['median']:.2f} uV  "
              f"AF rms {c['AF7']['rms_1_40_uv']['median']:.1f}/{c['AF8']['rms_1_40_uv']['median']:.1f} uV  "
              f"move all/any per min {res['movement']['all_per_min_on_task']:.2f}/{res['movement']['any_per_min_on_task']:.2f}  "
              f"TP emg {res['emg']['level_uv_TP']['median']:.1f} uV  AF emg {res['emg']['level_uv_AF']['median']:.1f} uV")

    # run 2 vs run 1 at matched minutes on task
    pm1, pm2 = pms["vr_planes9"], pms["vr_planes10"]
    results["_matched_minutes_run2_minus_run1"] = {
        col: matched(pm1, pm2, col) for col in ("l60_uv_AF", "l60_ratio_AF", "dc_AF7", "dc_AF8", "rms_AF", "flatrail_AF",
                                                 "move_all", "move_any", "emg_uv_TP", "emg_uv_AF", "blinks_per_min")}
    # pre-window contrasts (log10 ratios of medians, epochs)
    pre = {name: data[name][data[name]["state"] == "pre"] for name, _ in SESSIONS}
    markers = {
        "l60_uv_AF": ("60 Hz pick-up AF7/AF8", None), "rms_AF": ("1-40 Hz RMS AF7/AF8", None),
        "emg_uv_AF": ("20-100 Hz EMG AF7/AF8", "af_bad"), "emg_uv_TP": ("20-100 Hz EMG TP9/TP10", "ear_bad"),
        "lf_pow_AF": ("1-4 Hz power AF7/AF8", "af_bad"), "alpha_pow_AF": ("8-12.5 Hz power AF7/AF8", "af_bad"),
        "beta_pow_AF": ("13-30 Hz power AF7/AF8", "af_bad"), "blinks": ("blink rate (AF > 100 uV)", None),
    }
    fc = dict(labels={k: v[0] for k, v in markers.items()}, run2_over_run1_log10={}, ref_over_run1_log10={}, levels={})
    for key, (lab, badcol) in markers.items():
        vals = {}
        for name, _ in SESSIONS:
            p = pre[name]
            if badcol:
                p = p[~p[badcol]]
            if key == "blinks":
                per_epoch = p["blink_edges"].to_numpy(float) / EPOCH_S * 60.0
                vals[name] = per_epoch
            elif key.endswith("_pow_AF"):
                vals[name] = 10 ** p[key.replace("_pow_AF", "_cut_AF")].to_numpy(float)
            else:
                vals[name] = p[key].to_numpy(float)
        fc["levels"][key] = {name: level(vals[name], 5) for name, _ in SESSIONS}
        if key == "blinks":
            # rate contrast: means, bootstrap over epochs (block 5)
            def rate_contrast(a, b):
                a, b = finite(a), finite(b)
                ma, mb = a.mean(), b.mean()
                rng = np.random.default_rng(3)
                bs = []
                for _ in range(B_BOOT):
                    aa = a[rng.integers(0, len(a), len(a))].mean()
                    bb = b[rng.integers(0, len(b), len(b))].mean()
                    bs.append(np.log10((aa + 0.5) / (bb + 0.5)))
                return dict(diff=float(np.log10((ma + 0.5) / (mb + 0.5))), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])],
                            p=float("nan"), rate_a=float(ma), rate_b=float(mb))
            fc["run2_over_run1_log10"][key] = rate_contrast(vals["vr_planes10"], vals["vr_planes9"])
            fc["ref_over_run1_log10"][key] = rate_contrast(vals["vr_planes8"], vals["vr_planes9"])
        else:
            la = {n: np.log10(np.clip(v, 1e-6, None)) for n, v in vals.items()}
            fc["run2_over_run1_log10"][key] = contrast(la["vr_planes10"], la["vr_planes9"], 5, seed=21)
            fc["ref_over_run1_log10"][key] = contrast(la["vr_planes8"], la["vr_planes9"], 5, seed=22)
    results["_pre_contrast"] = fc

    # pooled leakage over both runs tonight (epoch-level, AF)
    pooled = {}
    for code, tag in ((12.0, "12Hz_stages"), (9.0, "9Hz_blue_plane")):
        frames = []
        for name, _ in SESSIONS[:2]:
            dfe = data[name]
            frames.append(dfe[(np.abs(dfe["fstim"] - code) < 0.5) & dfe["state"].isin(["lit", "dark"]) & ~dfe["af_bad"] & (dfe["minutes"] >= 0)])
        s = pd.concat(frames)
        lit_, dark_ = s[s["state"] == "lit"], s[s["state"] == "dark"]
        d = dict(n_lit=int(len(lit_)), n_dark=int(len(dark_)))
        for k in range(1, 5):
            d[f"snr{k}_AF_lit_minus_dark"] = contrast(lit_[f"snr{k}_AF"], dark_[f"snr{k}_AF"], 5, seed=k)
        for b in BANDS:
            d[f"{b}_AF_lit_minus_dark_lines_in"] = contrast(lit_[f"{b}_in_AF"], dark_[f"{b}_in_AF"], 5, seed=5)
            d[f"{b}_AF_lit_minus_dark_lines_cut"] = contrast(lit_[f"{b}_cut_AF"], dark_[f"{b}_cut_AF"], 5, seed=5)
        pooled[tag] = d
    results["_leakage_pooled_tonight"] = pooled

    # reference-level summary for each marker (level during stimulation)
    results["_methods"] = dict(
        epoch_s=EPOCH_S, artifact_uv=ART_UV, rail=[RAIL_LO, RAIL_HI], mid_scale=MID_SCALE, flat_s=FLAT_S, pre_window_s=PRE_S,
        matched_minutes=MATCH_MIN, bootstrap=dict(B=B_BOOT, block_epochs=BLOCK_EPOCHS, pre_block_epochs=5),
        permutation_B=B_PERM, drift="Theil-Sen slope per 10 min, Sen 95 % interval",
        emg="log10 mean PSD 20-100 Hz with 60 +- 2 Hz and the epoch's stimulus lines (+-0.75 Hz, all harmonics, subharmonic) cut; uV rms = sqrt(mean PSD x 80 Hz)",
        snr="log10 of mean PSD within +-0.5 Hz of k x stimulus over the median PSD 1-3 Hz away (other lines cut)",
        bands=BANDS, lit_dark="luminance > 0.05 in the stage log; epoch pure when >= 90 % one state",
        timeline_warning=results["vr_planes9"]["note_timeline"],
    )
    json.dump(results, open(os.path.join(OUT, "flash_confounds.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    figure(data, results, pms, specs, os.path.join(OUT, "flash_confounds.png"))

    mm = results["_matched_minutes_run2_minus_run1"]
    print("\nrun 2 - run 1 at matched minutes 0-20 (per-minute medians): " + ", ".join(
        f"{k} {v['diff']:+.3g} [{v['ci'][0]:+.3g}, {v['ci'][1]:+.3g}] p {v['p_wilcoxon']:.3f}" for k, v in mm.items()))
    print("pre-window run 2 / run 1 (log10): " + ", ".join(
        f"{k} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for k, v in fc["run2_over_run1_log10"].items()))
    for tag, d in pooled.items():
        print(f"leakage {tag}: " + ", ".join(f"{k} {v['diff']:+.3f} p {v['p']:.3f}" for k, v in d.items() if isinstance(v, dict)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
