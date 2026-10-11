"""State of Flash, forehead-muscle family: is there frontalis tension in the
AF7 / AF8 channels that would go with the subject's forehead ache?

Frontalis EMG is broadband and sits above the EEG: 20-45 Hz and (through the
Muse's anti-alias roll-off) 65-120 Hz.  Per 2 s Hann window, with every
stimulus line (12 Hz and its harmonics, the 9 Hz blue plane and its harmonics,
the 6 / 7.2 / 9 / 12 Hz sweep blocks and their harmonics, the subharmonics)
and the mains (60 +- 6 Hz, 120 +- 3 Hz) cut out of every window alike, this
script measures on AF7 and AF8:

  power      mean PSD over the kept bins x nominal band width, in uV^2,
             for 20-45 Hz and 65-120 Hz
  ratios     the same band on AF7/AF8 over TP9 / TP10 (temporalis, jaw) and
             over AUX (the Oz cup, neck): frontal tension should raise the
             frontal channels relative to the others
  bursts     windows whose 20-45 Hz power exceeds 3x the session median,
             counted per minute (the marker IS the artifact: no exclusion)
  slope      least-squares slope of log10 PSD against log10 f over the kept
             15-45 Hz bins (an EMG-flattened spectrum has a slope nearer 0)
  contact    60 Hz pick-up and DC offset of AF7 / AF8 per minute, because the
             facial interface presses on the forehead and a changing contact
             changes high-frequency power without any muscle

Per minute: Welch mean over the good 2 s windows with a bootstrap 95 % CI.
Drift: Theil-Sen slope over the per-minute values (95 % interval).  Run 1 vs
run 2 at matched minutes on task (paired bootstrap and sign-flip test).
Lit vs dark: each lit window paired with the nearest dark window of the same
stage within 45 s.  Reference day: vr_planes8, level only.  Good windows: no
channel above 150 uV peak after a 1-40 Hz band-pass, no blink, forehead
electrodes in contact (AF 60 Hz pick-up under 200 uV^2/Hz).

    python analysis/flash_frontalis.py

Writes runs/state_of_flash/flash_frontalis.json and flash_frontalis.png.
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
SESSIONS = [("vr_planes9", "run 1 (17:57)"), ("vr_planes10", "run 2 (19:50)"), ("vr_planes8", "reference 10-07")]
TONIGHT = ("vr_planes9", "vr_planes10")
REF = "vr_planes8"
WIN_S = 2.0
ARTIFACT_UV = 150.0
BLINK_UV = 100.0
CONTACT_60HZ = 200.0          # uV^2/Hz on AF7/AF8: above this the forehead electrodes are off the skin
PRE_S = 180.0                 # windows start this long before the first flicker
PAIR_S = 45.0
B = 2000
RNG = np.random.default_rng(20261010)

# stimulus frequencies that occur anywhere in these sessions; their harmonics and
# subharmonics are cut out of every window alike, so every comparison is on the same bins
STIM_F = (6.0, 7.2, 9.0, 12.0048)
LINES = sorted({k * f for f in STIM_F for k in range(1, int(128 / f) + 1)} | {f / 2 for f in STIM_F})
LINE_HALF = 0.75
STAGE_FREQ = {"calib": "12 Hz", "calib_red": "12 Hz", "calib_green": "12 Hz", "calib_blue": "12 Hz",
              "plane_red": "12 Hz", "plane_green": "12 Hz", "calib_blue_sweep": "sweep", "plane_blue": "9 Hz"}
STAGE_ORDER = ["calib", "calib_blue_sweep", "calib_red", "calib_green", "calib_blue", "plane_blue", "plane_red", "plane_green"]

# colours: lit / dark distinction (validated pair), run identity (validated pair), neutral reference
C_LIT, C_DARK = "#2a78d6", "#eb6834"
C_RUN = {"vr_planes9": "#4a3aa7", "vr_planes10": "#1baf7a", "vr_planes8": "#8f8e88"}
C_INK, C_GRID, C_SURF = "#3a3a37", "#ecebe5", "#fcfcfb"


# ----------------------------------------------------------------- stats helpers
def boot_mean(a, B=B):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float(np.mean(a)) if len(a) else float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(a), (B, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(a, b, B=B):
    """Ratio of means with independent resampling of both groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan"), float("nan")
    ma = a[RNG.integers(0, len(a), (B, len(a)))].mean(1)
    mb = b[RNG.integers(0, len(b), (B, len(b)))].mean(1)
    r = ma / mb
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_log_ratio(a, b, B=B):
    """log10(mean a / mean b) with CI: a and b are independent window sets."""
    m, lo, hi = boot_ratio(a, b, B)
    return float(np.log10(m)), float(np.log10(lo)), float(np.log10(hi))


def paired(d, B=B):
    """Mean of paired differences, bootstrap CI, sign-flip permutation p."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return dict(n=int(len(d)), mean=float(d.mean()) if len(d) else float("nan"))
    idx = RNG.integers(0, len(d), (B, len(d)))
    bm = d[idx].mean(1)
    signs = RNG.choice([-1.0, 1.0], (B, len(d)))
    perm = (signs * d).mean(1)
    p = float((np.abs(perm) >= abs(d.mean()) - 1e-15).mean())
    return dict(n=int(len(d)), mean=float(d.mean()), ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                p_signflip=p, positive=int((d > 0).sum()))


def theil(x, y, per=10.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        return None
    r = stats.theilslopes(y[ok], x[ok], alpha=0.95)
    return dict(n=int(ok.sum()), slope=float(r[0] * per), ci=[float(r[2] * per), float(r[3] * per)],
                span_min=float(x[ok].max() - x[ok].min()))


def spearman_perm(x, y, B=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        return None
    x, y = x[ok], y[ok]
    rho = float(stats.spearmanr(x, y)[0])
    perm = np.array([stats.spearmanr(x, RNG.permutation(y))[0] for _ in range(B)])
    return dict(n=int(len(x)), rho=rho, p_perm=float((np.abs(perm) >= abs(rho) - 1e-12).mean()))


# ----------------------------------------------------------------- loading and annotation
def annotate(d, t):
    """Per EEG sample: stage code (index into names, -1 none) and lit (luminance > 0.05)."""
    stage = np.full(len(t), -1, np.int8)
    lit = np.zeros(len(t), bool)
    names = []
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        L = pd.read_csv(p, usecols=[0, 1, 2, 3])
        if len(L) < 10:
            continue
        lt, lum = L.iloc[:, 0].to_numpy(float), L.iloc[:, 3].to_numpy(float)
        order = np.argsort(lt)
        lt, lum = lt[order], lum[order]
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5)
        names.append(kind)
        stage[ok] = len(names) - 1
        lit[ok] = lum[idx[ok]] > 0.05
    return stage, lit, names


def kept_bins(f):
    cut = np.zeros(len(f), bool)
    for L in LINES:
        cut |= np.abs(f - L) <= LINE_HALF
    cut |= np.abs(f - 60.0) <= 6.0      # mains: the ear electrodes carry a line 1e4 above the band, Hann leakage reaches +-5 Hz
    cut |= np.abs(f - 120.0) <= 3.0
    b2045 = (f >= 20) & (f <= 45) & ~cut
    b65 = (f >= 65) & (f <= 120) & ~cut
    bslope = (f >= 15) & (f <= 45) & ~cut
    m60 = np.abs(f - 60.0) <= 0.75
    return b2045, b65, bslope, m60


def windows(d):
    """One row per 2 s window from 3 min before the first flicker to the end."""
    t, X, names, fs = sof.load_eeg(d)
    f0, _ = sof.stim_freq(d)
    _, _, first, last = sof.timeline(d, t, f0)
    stage, lit, stage_names = annotate(d, t)
    ch = {nm: i for i, nm in enumerate(names)}
    n = int(round(WIN_S * fs))
    i0 = int(np.searchsorted(t, first - PRE_S))
    starts = np.arange(i0, len(t) - n, n)
    sos = butter(2, [1.0, 40.0], "bandpass", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    blink = blinkmask.blink_mask(X, names, fs, BLINK_UV)
    seg = X[starts[:, None] + np.arange(n)[None, :]]                 # (E, n, ch) raw
    dc = seg.mean(1)
    segf = Xf[starts[:, None] + np.arange(n)[None, :]]
    segf = segf - segf.mean(1, keepdims=True)
    p2p = np.abs(segf).max(1)
    f, P = welch(seg, fs=fs, nperseg=n, noverlap=0, axis=1)          # one Hann periodogram per window
    b2045, b65, bslope, m60 = kept_bins(f)
    df = pd.DataFrame(dict(t=t[starts] + WIN_S / 2))
    df["minutes"] = (df["t"] - first) / 60.0
    df["minute"] = np.floor(df["minutes"]).astype(int)
    df["on_task"] = (df["t"] >= first) & (df["t"] <= last)
    st = np.array([np.bincount(stage[s:s + n].astype(int) + 1, minlength=len(stage_names) + 1).argmax() - 1 for s in starts])
    lf = np.array([lit[s:s + n].mean() for s in starts])
    df["stage"] = [stage_names[k] if k >= 0 else "" for k in st]
    df["freq"] = [STAGE_FREQ.get(s, "") for s in df["stage"]]
    df["state"] = np.where(st < 0, "none", np.where(lf >= 0.9, "lit", np.where(lf <= 0.1, "dark", "mixed")))
    df["blink"] = np.array([blink[s:s + n].mean() for s in starts])
    df["artifact"] = p2p.max(1) > ARTIFACT_UV
    df["artifact_af"] = p2p[:, [ch["AF7"], ch["AF8"]]].max(1) > ARTIFACT_UV
    df["artifact_tp"] = p2p[:, [ch["TP9"], ch["TP10"]]].max(1) > ARTIFACT_UV
    for nm in names:
        c = ch[nm]
        df[f"{nm}_2045"] = P[:, b2045, c].mean(1) * 25.0
        df[f"{nm}_65"] = P[:, b65, c].mean(1) * 55.0
        df[f"{nm}_60hz"] = P[:, m60, c].mean(1)
        df[f"{nm}_dc"] = dc[:, c]
    for nm in ("AF7", "AF8"):
        c = ch[nm]
        lx = np.log10(f[bslope])[None, :]
        ly = np.log10(P[:, bslope, c] + 1e-12)
        lx0 = lx - lx.mean()
        df[f"{nm}_slope"] = ((lx0 * (ly - ly.mean(1, keepdims=True))).sum(1) / (lx0 ** 2).sum())
    df["AF_2045"] = (df["AF7_2045"] + df["AF8_2045"]) / 2
    df["AF_65"] = (df["AF7_65"] + df["AF8_65"]) / 2
    df["TP_2045"] = (df["TP9_2045"] + df["TP10_2045"]) / 2
    df["TP_65"] = (df["TP9_65"] + df["TP10_65"]) / 2
    df["AF_60hz"] = (df["AF7_60hz"] + df["AF8_60hz"]) / 2
    df["AF_slope"] = (df["AF7_slope"] + df["AF8_slope"]) / 2
    df["contact"] = df["AF_60hz"] < CONTACT_60HZ
    df["good"] = ~df["artifact"] & (df["blink"] == 0) & df["contact"]
    # bursts: 20-45 Hz power above 3x the session median over task windows, per channel
    for nm in ("AF7", "AF8"):
        med = df.loc[df["on_task"], f"{nm}_2045"].median()
        df[f"{nm}_burst"] = df[f"{nm}_2045"] > 3 * med
        df.attrs[f"{nm}_burst_threshold"] = float(3 * med)
    df["burst"] = df["AF7_burst"] | df["AF8_burst"]
    df.attrs.update(first=float(first), last=float(last), fs=float(fs), n_bins_2045=int(b2045.sum()),
                    n_bins_65=int(b65.sum()), n_bins_slope=int(bslope.sum()), stages=stage_names)
    return df


# ----------------------------------------------------------------- per-minute table
MINUTE_MARKERS = ["AF_2045", "AF7_2045", "AF8_2045", "AF_65", "AF7_65", "AF8_65", "AF_slope", "AF_60hz", "AF7_60hz", "AF8_60hz",
                  "TP9_2045", "TP10_2045", "AUX_2045", "TP9_65", "TP10_65", "AUX_65"]
RATIOS = [("ratio_tp_2045", "AF_2045", "TP_2045"), ("ratio_tp9_2045", "AF_2045", "TP9_2045"),
          ("ratio_tp10_2045", "AF_2045", "TP10_2045"), ("ratio_aux_2045", "AF_2045", "AUX_2045"),
          ("ratio_tp_65", "AF_65", "TP_65"), ("ratio_tp9_65", "AF_65", "TP9_65"), ("ratio_aux_65", "AF_65", "AUX_65")]


def per_minute(df):
    rows = []
    last_min = int(np.floor((df.attrs["last"] - df.attrs["first"]) / 60.0))
    prev_dc = None
    for m in range(int(df["minute"].min()), last_min + 1):
        sub = df[df["minute"] == m]
        g = sub[sub["good"]]
        row = dict(minute=m, n_windows=int(len(sub)), n_good=int(len(g)),
                   stage=sub["stage"].mode().iloc[0] if len(sub) else "",
                   lit_share=float((sub["state"] == "lit").mean()) if len(sub) else float("nan"),
                   bursts=int(sub["burst"].sum()), bursts_af7=int(sub["AF7_burst"].sum()), bursts_af8=int(sub["AF8_burst"].sum()),
                   bursts_in_good=int((sub["burst"] & sub["good"]).sum()),
                   dc_af7=float(sub["AF7_dc"].mean()) if len(sub) else float("nan"),
                   dc_af8=float(sub["AF8_dc"].mean()) if len(sub) else float("nan"),
                   dc_sd=float(np.mean([sub["AF7_dc"].std(), sub["AF8_dc"].std()])) if len(sub) > 2 else float("nan"))
        row["dc_drift"] = float(np.mean([abs(row["dc_af7"] - prev_dc[0]), abs(row["dc_af8"] - prev_dc[1])])) if prev_dc else float("nan")
        prev_dc = (row["dc_af7"], row["dc_af8"])
        enough = len(g) >= 10
        for mk in MINUTE_MARKERS:
            mean, lo, hi = boot_mean(g[mk]) if enough else (float("nan"),) * 3
            row[mk], row[mk + "_lo"], row[mk + "_hi"] = mean, lo, hi
        for name, a, b in RATIOS:
            mean, lo, hi = boot_ratio(g[a], g[b]) if enough else (float("nan"),) * 3
            row[name], row[name + "_lo"], row[name + "_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- per-session results
def lit_vs_dark(df, band):
    """Paired lit - dark (nearest dark window of the same stage within 45 s), log10 power."""
    out = {}
    g = df[df["good"] & df["on_task"]]
    groups = {"12 Hz calibration": g["stage"].isin(["calib", "calib_red", "calib_green", "calib_blue"]),
              "12 Hz planes": g["stage"].isin(["plane_red", "plane_green"]) & (g["freq"] == "12 Hz"),
              "9 Hz blue plane": (g["stage"] == "plane_blue") & (g["freq"] == "9 Hz"),
              "sweep": g["stage"] == "calib_blue_sweep"}
    for label, sel in groups.items():
        sub = g[sel]
        if len(sub) < 20:
            continue
        diffs, lit_vals, dark_vals = [], [], []
        for stg in sub["stage"].unique():
            s = sub[sub["stage"] == stg]
            L, D = s[s["state"] == "lit"], s[s["state"] == "dark"]
            if len(L) < 3 or len(D) < 3:
                continue
            dt = D["t"].to_numpy()
            for _, r in L.iterrows():
                j = np.argmin(np.abs(dt - r["t"]))
                if abs(dt[j] - r["t"]) <= PAIR_S:
                    diffs.append(np.log10(r[band]) - np.log10(D[band].iloc[j]))
                    lit_vals.append(r[band])
                    dark_vals.append(D[band].iloc[j])
        if len(diffs) >= 5:
            res = paired(diffs)
            res["lit_mean_uv2"] = boot_mean(lit_vals)
            res["dark_mean_uv2"] = boot_mean(dark_vals)
            res["percent"] = float((10 ** res["mean"] - 1) * 100)
            out[label] = res
    return out


def session_results(name, df, pm):
    first, last = df.attrs["first"], df.attrs["last"]
    task = df[df["on_task"]]
    good = task[task["good"]]
    onm = pm[(pm["minute"] >= 0) & (pm["n_good"] >= 10)]
    res = dict(session=name, first_flicker_s=first, stimulation_min=float((last - first) / 60),
               fs=df.attrs["fs"], kept_bins=dict(b2045=df.attrs["n_bins_2045"], b65=df.attrs["n_bins_65"], slope=df.attrs["n_bins_slope"]),
               windows=dict(task=int(len(task)), good=int(len(good)), artifact_any=int(task["artifact"].sum()),
                            artifact_af=int(task["artifact_af"].sum()), artifact_tp=int(task["artifact_tp"].sum()),
                            blink=int((task["blink"] > 0).sum()), no_contact=int((~task["contact"]).sum())),
               stages=df.attrs["stages"])
    # level over the whole task
    lvl = {}
    for mk in ("AF_2045", "AF7_2045", "AF8_2045", "AF_65", "AF_slope", "AF_60hz", "TP9_2045", "TP10_2045", "AUX_2045"):
        lvl[mk] = boot_mean(good[mk])
    for nm, a, b in RATIOS:
        lvl[nm] = boot_ratio(good[a], good[b])
    lvl["bursts_per_min"] = float(task["burst"].sum() / (len(task) * WIN_S / 60.0))
    lvl["bursts_per_min_af7"] = float(task["AF7_burst"].sum() / (len(task) * WIN_S / 60.0))
    lvl["bursts_per_min_af8"] = float(task["AF8_burst"].sum() / (len(task) * WIN_S / 60.0))
    lvl["burst_windows"] = int(task["burst"].sum())
    lvl["burst_windows_with_artifact_or_blink"] = int((task["burst"] & (task["artifact"] | (task["blink"] > 0))).sum())
    lvl["burst_windows_af_only_artifact"] = int((task["burst"] & task["artifact_af"]).sum())
    lvl["burst_threshold_uv2"] = dict(AF7=df.attrs["AF7_burst_threshold"], AF8=df.attrs["AF8_burst_threshold"])
    res["task_level"] = lvl
    # pre: the last 60 s before the first flicker (headset on, settling)
    pre = df[(df["t"] >= first - 60) & (df["t"] < first)]
    res["pre_last_60s"] = dict(n_windows=int(len(pre)), n_good=int(pre["good"].sum()), n_contact=int(pre["contact"].sum()),
                               AF_2045=boot_mean(pre.loc[pre["good"], "AF_2045"]), AF_65=boot_mean(pre.loc[pre["good"], "AF_65"]),
                               AF_60hz=boot_mean(pre.loc[pre["contact"], "AF_60hz"]), bursts=int(pre["burst"].sum()))
    post = df[(df["t"] > last) & (df["t"] <= last + 60)]
    res["post_60s"] = dict(n_windows=int(len(post)), n_good=int(post["good"].sum()),
                           AF_2045=boot_mean(post.loc[post["good"], "AF_2045"]), bursts=int(post["burst"].sum()))
    # drift: Theil-Sen over per-minute values (log10 power per 10 min)
    drift = {}
    x = onm["minute"].to_numpy(float) + 0.5
    for mk in ("AF_2045", "AF7_2045", "AF8_2045", "AF_65", "ratio_tp9_2045", "ratio_tp_2045", "ratio_aux_2045", "AF_60hz", "TP9_2045", "AUX_2045"):
        r = theil(x, np.log10(onm[mk].to_numpy(float)))
        if r:
            r["percent_per_10min"] = float((10 ** r["slope"] - 1) * 100)
            r["percent_ci"] = [float((10 ** r["ci"][0] - 1) * 100), float((10 ** r["ci"][1] - 1) * 100)]
            r["unit"] = "log10 per 10 min"
        drift[mk] = r
    drift["AF_slope"] = theil(x, onm["AF_slope"].to_numpy(float))
    drift["bursts"] = theil(x, onm["bursts"].to_numpy(float))
    drift["dc_sd"] = theil(x, onm["dc_sd"].to_numpy(float))
    # window-level Theil-Sen too (many more points, autocorrelated)
    gw = good[good["minutes"] >= 0]
    drift["AF_2045_windows"] = theil(gw["minutes"], np.log10(gw["AF_2045"]))
    res["drift"] = drift
    # lit vs dark
    res["lit_vs_dark"] = {"AF_2045": lit_vs_dark(df, "AF_2045"), "AF_65": lit_vs_dark(df, "AF_65")}
    # contact tracking: per-minute HF power against 60 Hz pick-up and DC drift
    res["contact"] = dict(
        rho_hf_vs_60hz=spearman_perm(onm["AF_2045"], onm["AF_60hz"]),
        rho_hf65_vs_60hz=spearman_perm(onm["AF_65"], onm["AF_60hz"]),
        rho_hf_vs_dc_drift=spearman_perm(onm["AF_2045"], onm["dc_drift"]),
        rho_hf_vs_dc_sd=spearman_perm(onm["AF_2045"], onm["dc_sd"]),
        rho_hf_vs_minute=spearman_perm(onm["minute"], onm["AF_2045"]),
        rho_60hz_vs_minute=spearman_perm(onm["minute"], onm["AF_60hz"]),
        af_60hz_first_5min=boot_mean(good[good["minutes"] < 5]["AF_60hz"]),
        af_60hz_last_5min=boot_mean(good[good["minutes"] > (last - first) / 60 - 5]["AF_60hz"]),
        af_2045_first_5min=boot_mean(good[good["minutes"] < 5]["AF_2045"]),
        af_2045_last_5min=boot_mean(good[good["minutes"] > (last - first) / 60 - 5]["AF_2045"]),
        dc_af7_range_uv=float(onm["dc_af7"].max() - onm["dc_af7"].min()) if len(onm) else float("nan"),
        dc_af8_range_uv=float(onm["dc_af8"].max() - onm["dc_af8"].min()) if len(onm) else float("nan"),
        tp_60hz_task=boot_mean(good[["TP9_60hz", "TP10_60hz"]].mean(1)))
    # last 4 task minutes against minutes 2-8 (contact partly settled), per channel: a late rise on one
    # forehead channel with its own 60 Hz pick-up still falling is muscle (or the pad), not contact
    early = good[(good["minutes"] >= 2) & (good["minutes"] < 8)]
    late = good[good["minutes"] >= (last - first) / 60 - 4]
    res["late_vs_early"] = dict(early_minutes="2-8", late_minutes="last 4", n=[int(len(early)), int(len(late))])
    for mk in ("AF7_2045", "AF8_2045", "AF7_65", "AF8_65", "TP9_2045", "TP10_2045", "AUX_2045", "AF7_60hz", "AF8_60hz"):
        res["late_vs_early"][mk] = dict(zip(("ratio", "lo", "hi"), boot_ratio(late[mk], early[mk])))
    for mk in ("AF7_slope", "AF8_slope"):
        res["late_vs_early"][mk] = float(late[mk].mean() - early[mk].mean()) if len(late) and len(early) else float("nan")
    lt = task[task["minutes"] >= (last - first) / 60 - 4]
    res["late_vs_early"]["burst_share_late"] = dict(AF7=float(lt["AF7_burst"].mean()), AF8=float(lt["AF8_burst"].mean()))
    et = task[(task["minutes"] >= 2) & (task["minutes"] < 8)]
    res["late_vs_early"]["burst_share_early"] = dict(AF7=float(et["AF7_burst"].mean()), AF8=float(et["AF8_burst"].mean()))
    # lit vs dark per channel and per calibration colour (is the effect forehead-specific?)
    pc = {}
    for ch in ("AF7", "AF8", "TP9", "TP10", "AUX"):
        pc[ch] = {band: {g: dict(percent=r["percent"], p=r["p_signflip"], n=r["n"]) for g, r in lit_vs_dark(df, f"{ch}_{band}").items()}
                  for band in ("2045", "65")}
    res["lit_vs_dark_per_channel"] = pc
    col = {}
    for stg in ("calib", "calib_red", "calib_green", "calib_blue"):
        s = good[good["stage"] == stg]
        L, D = s[s["state"] == "lit"], s[s["state"] == "dark"]
        if len(L) >= 5 and len(D) >= 5:
            col[stg] = dict(n=[int(len(L)), int(len(D))],
                            AF_2045=dict(zip(("lit_over_dark", "lo", "hi"), boot_ratio(L["AF_2045"], D["AF_2045"]))),
                            AF_65=dict(zip(("lit_over_dark", "lo", "hi"), boot_ratio(L["AF_65"], D["AF_65"]))),
                            AUX_2045=dict(zip(("lit_over_dark", "lo", "hi"), boot_ratio(L["AUX_2045"], D["AUX_2045"]))),
                            TP9_2045=dict(zip(("lit_over_dark", "lo", "hi"), boot_ratio(L["TP9_2045"], D["TP9_2045"]))))
    res["lit_vs_dark_by_calibration_colour"] = col
    # per stage level
    st = {}
    for stg in STAGE_ORDER:
        s = good[good["stage"] == stg]
        if len(s) >= 10:
            st[stg] = dict(n=int(len(s)), freq=STAGE_FREQ.get(stg), t0_min=float((s["t"].min() - first) / 60), t1_min=float((s["t"].max() - first) / 60),
                           AF_2045=boot_mean(s["AF_2045"]), AF_65=boot_mean(s["AF_65"]), AF_slope=boot_mean(s["AF_slope"]),
                           ratio_tp9_2045=boot_ratio(s["AF_2045"], s["TP9_2045"]), ratio_aux_2045=boot_ratio(s["AF_2045"], s["AUX_2045"]),
                           bursts_per_min=float(task[task["stage"] == stg]["burst"].sum() / max(1, (task["stage"] == stg).sum()) / WIN_S * 60))
    res["by_stage"] = st
    res["minutes"] = pm.replace({np.nan: None}).to_dict(orient="records")
    return res


# ----------------------------------------------------------------- run 1 vs run 2
def compare_runs(frames, mins):
    a, b = frames["vr_planes9"], frames["vr_planes10"]
    pa, pb = mins["vr_planes9"], mins["vr_planes10"]
    out = {}
    shared = sorted(set(pa.loc[(pa["minute"] >= 0) & (pa["n_good"] >= 10), "minute"]) &
                    set(pb.loc[(pb["minute"] >= 0) & (pb["n_good"] >= 10), "minute"]))
    pa_i, pb_i = pa.set_index("minute"), pb.set_index("minute")
    mm = {}
    for mk in ("AF_2045", "AF7_2045", "AF8_2045", "AF_65", "ratio_tp9_2045", "ratio_tp_2045", "ratio_aux_2045", "AF_60hz", "TP9_2045", "AUX_2045"):
        d = np.log10(pb_i.loc[shared, mk].to_numpy(float)) - np.log10(pa_i.loc[shared, mk].to_numpy(float))
        r = paired(d)
        r["ratio_run2_over_run1"] = float(10 ** r["mean"])
        r["ratio_ci"] = [float(10 ** r["ci"][0]), float(10 ** r["ci"][1])] if "ci" in r else None
        r["unit"] = "log10(run 2 / run 1), per shared minute"
        mm[mk] = r
    for mk in ("AF_slope", "bursts", "dc_sd"):
        d = pb_i.loc[shared, mk].to_numpy(float) - pa_i.loc[shared, mk].to_numpy(float)
        r = paired(d)
        r["unit"] = "run 2 - run 1, per shared minute"
        mm[mk] = r
    out["matched_minutes"] = dict(minutes=[int(m) for m in shared], markers=mm)
    # by stage (same stage in both runs; the blue and green planes have extra passes in run 2)
    bs = {}
    ga, gb = a[a["good"] & a["on_task"]], b[b["good"] & b["on_task"]]
    for stg in STAGE_ORDER:
        sa, sb = ga[ga["stage"] == stg], gb[gb["stage"] == stg]
        if len(sa) >= 10 and len(sb) >= 10:
            bs[stg] = {mk: dict(zip(("log10_ratio", "lo", "hi"), boot_log_ratio(sb[mk], sa[mk]))) for mk in ("AF_2045", "AF_65", "AF_60hz")}
            bs[stg]["n"] = [int(len(sa)), int(len(sb))]
            bs[stg]["AF_slope_diff"] = float(sb["AF_slope"].mean() - sa["AF_slope"].mean())
    out["by_stage"] = bs
    # run 2's extra minutes against its own shared minutes
    extra = pb[(pb["minute"] > max(shared)) & (pb["n_good"] >= 10)]
    if len(extra) >= 3:
        out["run2_extra_minutes"] = dict(minutes=[int(m) for m in extra["minute"]],
                                         AF_2045_log10_ratio_vs_shared=dict(zip(("log10_ratio", "lo", "hi"),
                                                                                boot_log_ratio(gb[gb["minute"] > max(shared)]["AF_2045"], gb[gb["minute"].isin(shared)]["AF_2045"]))))
    return out


def reference_level(frames):
    out = {}
    ref = frames[REF]
    gr = ref[ref["good"] & ref["on_task"]]
    for s in TONIGHT:
        g = frames[s][frames[s]["good"] & frames[s]["on_task"]]
        out[s] = {mk: dict(zip(("ratio", "lo", "hi"), boot_ratio(g[mk], gr[mk]))) for mk in ("AF_2045", "AF_65", "AF_60hz", "TP9_2045", "AUX_2045")}
        out[s]["AF_slope_diff"] = float(g["AF_slope"].mean() - gr["AF_slope"].mean())
        out[s]["bursts_per_min"] = [float(frames[s][frames[s]["on_task"]]["burst"].mean() / WIN_S * 60), float(ref[ref["on_task"]]["burst"].mean() / WIN_S * 60)]
    return out


# ----------------------------------------------------------------- figure
def figure(frames, mins, results, cmp_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK, "xtick.color": C_INK,
                         "ytick.color": C_INK, "text.color": C_INK, "axes.titlesize": 8.5, "figure.facecolor": C_SURF,
                         "axes.facecolor": C_SURF, "legend.frameon": False})
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    labels = dict(SESSIONS)

    def style(ax):
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def minute_panel(ax, mk, ylabel, title, log=True, band=True):
        for s, lab in SESSIONS:
            pm = mins[s]
            ok = pm["n_good"] >= 10 if (mk + "_lo") in pm else pm["n_windows"] >= 10
            x = pm.loc[ok, "minute"].to_numpy(float) + 0.5
            y = pm.loc[ok, mk].to_numpy(float)
            lw = 1.2 if s != REF else 1.0
            ax.plot(x, y, "-o", ms=3, lw=lw, color=C_RUN[s], label=lab, alpha=0.95 if s != REF else 0.8)
            if band and (mk + "_lo") in pm:
                ax.fill_between(x, pm.loc[ok, mk + "_lo"], pm.loc[ok, mk + "_hi"], color=C_RUN[s], alpha=0.16, lw=0)
        ax.axvspan(-3.5, 0, color="#eeeee9", zorder=0)
        ax.set_xlim(-3.2, 31)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("minutes since the first flicker")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", fontsize=7)
        style(ax)

    r1 = cmp_["matched_minutes"]["markers"]["AF_2045"]
    minute_panel(axes[0, 0], "AF_2045", "uV$^2$ (20-45 Hz, lines cut)", "a. forehead (AF7+AF8)/2, 20-45 Hz power per minute")
    le = results["vr_planes9"]["late_vs_early"]
    axes[0, 0].text(0.02, 0.03, f"run 2 / run 1 at {len(cmp_['matched_minutes']['minutes'])} matched minutes: x{r1['ratio_run2_over_run1']:.2f} "
                    f"[{r1['ratio_ci'][0]:.2f}, {r1['ratio_ci'][1]:.2f}]\nrun 1 last 4 min / min 2-8: AF7 x{le['AF7_2045']['ratio']:.1f}, AF8 x{le['AF8_2045']['ratio']:.1f}",
                    transform=axes[0, 0].transAxes, fontsize=7, va="bottom")
    minute_panel(axes[0, 1], "AF_65", "uV$^2$ (65-120 Hz, lines cut)", "b. forehead 65-120 Hz power per minute")
    minute_panel(axes[0, 2], "AF_slope", "slope of log10 PSD vs log10 f, 15-45 Hz", "c. forehead spectral slope (EMG flattens it toward 0)", log=False)
    minute_panel(axes[1, 0], "ratio_tp9_2045", "AF / TP9 (20-45 Hz)", "d. forehead over jaw (TP9) ratio, 20-45 Hz")
    minute_panel(axes[1, 1], "ratio_aux_2045", "AF / AUX (20-45 Hz)", "e. forehead over neck (AUX, Oz cup) ratio, 20-45 Hz")
    # bursts per minute on task (the settling minutes before the first flicker are all burst)
    ax = axes[1, 2]
    for s, lab in SESSIONS:
        pm = mins[s]
        ok = (pm["n_windows"] >= 10) & (pm["minute"] >= 0)
        ax.step(pm.loc[ok, "minute"].to_numpy(float), pm.loc[ok, "bursts"].to_numpy(float), where="post", color=C_RUN[s], lw=1.2, label=lab)
    ax.axvspan(-3.5, 0, color="#eeeee9", zorder=0)
    ax.set_xlim(-3.2, 31)
    ax.set_xlabel("minutes since the first flicker")
    ax.set_ylabel("2 s windows above 3x the session median, per minute")
    ax.set_title("f. frontal bursts per minute (AF7 or AF8; run 1 late: AF7)", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    style(ax)
    minute_panel(axes[2, 0], "AF_60hz", "60 Hz PSD, uV$^2$/Hz", "g. forehead contact: 60 Hz pick-up per minute (falls as the pads settle)")
    # scatter HF vs 60 Hz per minute
    ax = axes[2, 1]
    for s, lab in SESSIONS:
        pm = mins[s]
        ok = (pm["n_good"] >= 10) & (pm["minute"] >= 0)
        rho = results[s]["contact"]["rho_hf_vs_60hz"]
        txt = f"{lab}: rho {rho['rho']:+.2f}, p {rho['p_perm']:.3f}" if rho else lab
        ax.plot(pm.loc[ok, "AF_60hz"], pm.loc[ok, "AF_2045"], "o", ms=4, mfc=C_RUN[s], mec=C_SURF, mew=0.8, label=txt, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 3.0)
    ax.set_xlabel("AF 60 Hz pick-up, uV$^2$/Hz (one dot per task minute)")
    ax.set_ylabel("AF 20-45 Hz power, uV$^2$")
    ax.set_title("h. does forehead power track the contact? (Spearman over minutes)", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    style(ax)
    # lit vs dark per stage group
    ax = axes[2, 2]
    ypos, ylab = [], []
    k = 0
    for s in TONIGHT:
        lvd = results[s]["lit_vs_dark"]["AF_2045"]
        for grp, r in lvd.items():
            lm, llo, lhi = r["lit_mean_uv2"]
            dm, dlo, dhi = r["dark_mean_uv2"]
            ax.errorbar(lm, k + 0.15, xerr=[[lm - llo], [lhi - lm]], fmt="o", ms=5, color=C_LIT, capsize=2, lw=1.2,
                        label="lit / ON" if k == 0 else None)
            ax.errorbar(dm, k - 0.15, xerr=[[dm - dlo], [dhi - dm]], fmt="o", ms=5, color=C_DARK, capsize=2, lw=1.2,
                        label="dark / OFF" if k == 0 else None)
            ax.text(max(lhi, dhi) * 1.08, k, f"{r['percent']:+.0f} %, p {r['p_signflip']:.2f}, n {r['n']}", va="center", fontsize=7)
            ypos.append(k)
            ylab.append(f"{labels[s].split(' (')[0]}: {grp}")
            k += 1
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_ylim(-0.7, k + 0.6)
    ax.set_xscale("log")
    ax.set_xlabel("AF 20-45 Hz power, uV$^2$ (mean, 95 % CI); text: lit - dark, paired")
    ax.set_title("i. flicker on screen vs black on screen, forehead 20-45 Hz", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 3.0)
    style(ax)
    fig.suptitle("State of Flash, forehead muscle (frontalis) family: AF7 / AF8 high-frequency power with every stimulus line and the mains cut out, 2 s Hann windows.\n"
                 "Good windows only (no channel > 150 uV, no blink, forehead in contact); shading = 95 % bootstrap CI over windows; "
                 "grey band = before the first flicker (headset being settled).", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(OUT, "flash_frontalis.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    frames, mins, results = {}, {}, {}
    for s, lab in SESSIONS:
        d = os.path.join(ROOT, "runs", s)
        df = windows(d)
        pm = per_minute(df)
        frames[s], mins[s] = df, pm
        results[s] = session_results(s, df, pm)
        lv = results[s]["task_level"]
        print(f"{s:12s} {lab:18s} task windows {results[s]['windows']['task']:5d} good {results[s]['windows']['good']:5d} "
              f"AF 20-45 {lv['AF_2045'][0]:.2f} uV^2 [{lv['AF_2045'][1]:.2f}, {lv['AF_2045'][2]:.2f}]  "
              f"AF 65-120 {lv['AF_65'][0]:.3f}  slope {lv['AF_slope'][0]:+.2f}  AF/TP9 {lv['ratio_tp9_2045'][0]:.2f}  AF/AUX {lv['ratio_aux_2045'][0]:.2f}  "
              f"bursts/min {lv['bursts_per_min']:.2f}  60 Hz {lv['AF_60hz'][0]:.2f}")
        for mk in ("AF_2045", "AF_65", "AF_60hz"):
            r = results[s]["drift"][mk]
            if r:
                print(f"    drift {mk:9s} {r['percent_per_10min']:+6.1f} % per 10 min  95% [{r['percent_ci'][0]:+.1f}, {r['percent_ci'][1]:+.1f}]  n {r['n']} minutes")
        for grp, r in results[s]["lit_vs_dark"]["AF_2045"].items():
            print(f"    lit - dark {grp:18s} {r['percent']:+6.1f} %  CI log10 [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p {r['p_signflip']:.3f}  n {r['n']}")
        c = results[s]["contact"]
        print(f"    HF vs 60 Hz over minutes rho {c['rho_hf_vs_60hz']['rho']:+.2f} (p {c['rho_hf_vs_60hz']['p_perm']:.3f}); "
              f"HF vs DC drift rho {c['rho_hf_vs_dc_drift']['rho']:+.2f} (p {c['rho_hf_vs_dc_drift']['p_perm']:.3f}); "
              f"60 Hz first 5 min {c['af_60hz_first_5min'][0]:.2f} -> last 5 min {c['af_60hz_last_5min'][0]:.2f}; "
              f"AF 20-45 first {c['af_2045_first_5min'][0]:.2f} -> last {c['af_2045_last_5min'][0]:.2f}")
    cmp_ = compare_runs(frames, mins)
    ref = reference_level(frames)
    mm = cmp_["matched_minutes"]["markers"]
    print(f"\nrun 2 / run 1 at {len(cmp_['matched_minutes']['minutes'])} matched minutes: AF 20-45 x{mm['AF_2045']['ratio_run2_over_run1']:.2f} "
          f"[{mm['AF_2045']['ratio_ci'][0]:.2f}, {mm['AF_2045']['ratio_ci'][1]:.2f}] p {mm['AF_2045']['p_signflip']:.3f}; "
          f"AF 65-120 x{mm['AF_65']['ratio_run2_over_run1']:.2f}; 60 Hz x{mm['AF_60hz']['ratio_run2_over_run1']:.2f}; "
          f"slope {mm['AF_slope']['mean']:+.3f}; bursts {mm['bursts']['mean']:+.2f}/min")
    for s in TONIGHT:
        print(f"{s} / reference day: AF 20-45 x{ref[s]['AF_2045']['ratio']:.2f} [{ref[s]['AF_2045']['lo']:.2f}, {ref[s]['AF_2045']['hi']:.2f}], "
              f"AF 65-120 x{ref[s]['AF_65']['ratio']:.2f}, 60 Hz x{ref[s]['AF_60hz']['ratio']:.2f}, bursts/min {ref[s]['bursts_per_min'][0]:.2f} vs {ref[s]['bursts_per_min'][1]:.2f}")
    method = dict(
        window_s=WIN_S, taper="hann", lines_cut=[round(L, 3) for L in LINES], line_half_width_hz=LINE_HALF, mains_cut="60 +- 6 Hz, 120 +- 3 Hz",
        bands={"2045": "20-45 Hz, mean PSD over kept bins x 25 Hz -> uV^2", "65": "65-120 Hz, mean PSD over kept bins x 55 Hz -> uV^2"},
        slope="least-squares slope of log10 PSD on log10 f over kept 15-45 Hz bins",
        bursts="2 s windows whose 20-45 Hz power exceeds 3x the session median over task windows, per channel; either channel counts; no exclusion",
        good_window=f"no channel > {ARTIFACT_UV:.0f} uV peak after 1-40 Hz band-pass, no blink sample (blinkmask {BLINK_UV:.0f} uV), AF 60 Hz pick-up < {CONTACT_60HZ:.0f} uV^2/Hz",
        per_minute="mean over good windows, bootstrap 95 % CI (2000 resamples)", drift="Theil-Sen over per-minute values, 95 % interval, per 10 min",
        matched_minutes="paired log10 difference per shared minute, bootstrap CI over minutes, sign-flip permutation p",
        lit_vs_dark=f"each lit window paired with the nearest dark window of the same stage within {PAIR_S:.0f} s; sign-flip p",
        same_bins="the same kept bins are used in every window of every session, so stage, run and day comparisons are on identical bins")
    notes = [
        "Muse AF7/AF8 LSB is 0.40 uV; during the task their 20-45 Hz power is a few uV^2 (1.5-3 uV rms), near the device floor: sustained frontalis contraction would be 10-100x this and is not seen, but tension below ~1 uV rms is unmeasurable here.",
        "The 65-120 Hz band sits on the Muse anti-alias roll-off (PSD falls ~100x from 65 to 120 Hz); it is reported relative to itself only.",
        "vr_planes9: before the first flicker the forehead electrodes were off the skin (AF 60 Hz pick-up > 6000 uV^2/Hz) until about 2 minutes before; the last minute is the headset being settled, not a relaxed baseline. The ear electrodes carried a 60 Hz line of ~2-3e4 uV^2/Hz through run 1 (poor ear contact), hence the wide mains cut.",
        "vr_planes10: TP10 was loose (22 % of task windows above 150 uV, broadband noise 10x TP9), so ratios to TP10 and to the TP pair are not comparable between runs; the TP9 ratio is the one to read.",
        "Per-minute Theil-Sen intervals treat minutes as independent; window-level slopes are also given but their intervals are too narrow (autocorrelation).",
    ]
    json.dump(dict(script="analysis/flash_frontalis.py", date="2026-10-10", method=method, notes=notes, sessions=results,
                   run1_vs_run2=cmp_, tonight_over_reference=ref),
              open(os.path.join(OUT, "flash_frontalis.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    figure(frames, mins, results, cmp_)
    print("wrote", os.path.join(OUT, "flash_frontalis.json"), "and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
