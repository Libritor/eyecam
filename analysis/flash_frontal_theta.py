"""State of Flash, frontal theta and strain: does the forehead EEG carry the
subject's report (eyes irritated, a slight forehead ache, not drowsy) after
tonight's two VR flicker sessions?

Markers per 2 s epoch (non-overlapping), after a 1-40 Hz band-pass, with the
flicker line, its harmonics and its subharmonic cut out of every band
(analysis.state_of_flash.band_masks at the frequency on screen in that epoch;
the sweep's 6 and 7.2 Hz blocks sit inside theta, so theta is unmeasurable
there and is left out):
  th_f      log10 frontal theta power, 4-7.5 Hz, mean of AF7 and AF8 (uV^2/Hz)
  rel_th_f  frontal theta share of theta + alpha + beta power
  ta_f      log10 frontal theta / alpha (8-10.5 Hz)
  tb_f      log10 frontal theta / beta (15-30 Hz)
  fo_th     log10 frontal theta / occipital (AUX = Oz) theta
  be_f      log10 frontal beta power (arousal / tension index)
  th_oz, be_oz, th_ear, be_ear   the same bands behind the ears and on Oz
Mental effort and visual strain raise frontal midline theta; drowsiness raises
theta everywhere with falling beta.  The Muse's AF7 / AF8 are lateral
frontal, not midline, and a headache has no direct EEG sign.

Epochs are left out when any channel exceeds 150 uV peak (movement, a loose
electrode), when a channel is flat, or when a frontal channel shows a slow
0.5-10 Hz deflection above 50 uV (blink or eye movement; the Muse's FPz
reference cancels most of a blink, so tonight's blinks are 20-70 uV).

Contrasts (one subject, so uncertainty is over windows, not people):
  acute     lit segment minus the nearest black segment of the same stage
            (within 45 s); mean over pairs, bootstrap 95 % CI, sign-flip test
  drift     Theil-Sen slope per 10 min of task over per-minute medians
            (95 % interval), black epochs with one common 12 Hz mask so the
            9 Hz blue plane does not add a step, and lit 12 Hz epochs
  runs      run 2 minus run 1 at matched minutes on task (per-minute medians,
            paired over minutes), and each run against the reference day
  closure   share of time with Oz alpha >= 3x the eyes-open baseline for
            >= 1 s (blinkmask.closure_mask), per 5 min of task

    python analysis/flash_frontal_theta.py

Writes runs/state_of_flash/flash_frontal_theta.json and .png.
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
from analysis import state_of_flash as sof  # noqa: E402
import blinkmask  # noqa: E402

OUT = os.path.join(ROOT, "runs", "state_of_flash")
NAME = "flash_frontal_theta"
SESSIONS = [("run 1", "vr_planes9"), ("run 2", "vr_planes10"), ("reference", "vr_planes8")]
EPOCH_S = 2.0
ART_UV, FLAT_UV, SLOW_UV = 150.0, 1.0, 50.0
PAIR_S = 45.0
NBOOT = 4000
MIN_LEVEL_N = 15
RNG = np.random.default_rng(7)
MIXED = -1
STAGES = ["calib", "calib_blue_sweep", "calib_red", "calib_green", "calib_blue",
          "plane_blue", "plane_red", "plane_green"]
STAGE_LABEL = {"calib": "white", "calib_blue_sweep": "sweep", "calib_red": "red",
               "calib_green": "green", "calib_blue": "blue", "plane_blue": "blue plane",
               "plane_red": "red plane", "plane_green": "green plane"}
MARKERS = ["th_f", "rel_th_f", "ta_f", "tb_f", "fo_th", "be_f", "th_oz", "be_oz", "th_ear", "be_ear",
           "th_af7", "th_af8"]
FIG_MARKERS = ["th_f", "ta_f", "tb_f", "fo_th", "be_f"]
LABEL = {"th_f": "frontal theta (log10 uV^2/Hz)", "rel_th_f": "frontal theta share",
         "ta_f": "frontal theta/alpha (log10)", "tb_f": "frontal theta/beta (log10)",
         "fo_th": "frontal/occipital theta (log10)", "be_f": "frontal beta (log10 uV^2/Hz)",
         "th_oz": "Oz theta (log10)", "be_oz": "Oz beta (log10)", "th_ear": "ear theta (log10)",
         "be_ear": "ear beta (log10)", "th_af7": "AF7 theta (log10)", "th_af8": "AF8 theta (log10)"}
# validated categorical pair (dataviz palette slots 1 and 2) + neutral inks
C_FLICKER, C_DARK = "#2a78d6", "#eb6834"
C_INK, C_INK2, C_REF, C_GRID, C_BAND, C_SURF = "#3a3a37", "#8a8983", "#b5b4ad", "#ecebe5", "#f3f2ee", "#fcfcfb"


# ----------------------------------------------------------------- stimulus timeline
def read_meta(d):
    f0, _ = sof.stim_freq(d)
    plane_hz = {}
    try:
        pm = json.load(open(os.path.join(d, "planes_meta.json")))
        plane_hz = {str(k): float(v) for k, v in (pm.get("freq_delivered") or {}).items()}
    except Exception:
        pass
    sweep = []
    try:
        pc = json.load(open(os.path.join(d, "planes_calib.json")))
        blue = pc.get("blue") or {}
        order = [float(v) for v in blue.get("sweep", [])]
        per = blue.get("per_freq") or {}
        for hz in order:                       # the frequency actually delivered by the display
            key = next((k for k in per if abs(float(k) - hz) < 0.05), None)
            sweep.append(float((per.get(key) or {}).get("delivered") or hz))
    except Exception:
        pass
    return f0, plane_hz, sweep


def timeline(d, t):
    """Per EEG sample: lit (a flickering cell / ON block on screen), onscreen (a stage
    log covers it), stage code, flicker frequency of that stage (sweep: of that block)."""
    f0, plane_hz, sweep = read_meta(d)
    lit = np.zeros(len(t), bool)
    onscreen = np.zeros(len(t), bool)
    stage = np.full(len(t), MIXED, np.int8)
    fstim = np.full(len(t), np.nan)
    first, last = np.inf, -np.inf
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        if kind not in STAGES:
            print(f"  {os.path.basename(d)}: stage {kind} not used")
            continue
        L = pd.read_csv(p)
        lt = L["time"].to_numpy(float)
        lum = L["luminance"].to_numpy(float)
        order = np.argsort(lt)
        lt, lum = lt[order], lum[order]
        row_lit = lum > 0.05
        if kind == "calib_blue_sweep":
            if not sweep:
                print(f"  {os.path.basename(d)}: sweep without planes_calib.json, skipped")
                continue
            rising = np.concatenate(([False], row_lit[1:] & ~row_lit[:-1]))
            block = np.cumsum(rising)          # ON block k and the OFF block after it
            hz_row = np.array(sweep)[block % len(sweep)]
        elif kind.startswith("plane_"):
            hz_row = np.full(len(lt), plane_hz.get(kind.split("_")[1], f0))
        else:
            hz_row = np.full(len(lt), f0)
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5)
        lit[ok] = row_lit[idx[ok]]
        onscreen[ok] = True
        stage[ok] = STAGES.index(kind)
        fstim[ok] = hz_row[idx[ok]]
        first, last = min(first, lt[0]), max(last, lt[-1])
    return lit, onscreen, stage, fstim, first, last, f0


# ----------------------------------------------------------------- epochs
def masks_for(f, hz):
    """sof.band_masks at the frequency on screen; the sweep's low frequencies also
    put their 4th and 5th harmonics inside beta, cut here too (no change at 9 / 12 Hz)."""
    th, al, be = sof.band_masks(f, hz)
    cut = np.zeros(len(f), bool)
    for k in (4, 5):
        cut |= np.abs(f - k * hz) <= 0.75
    return th & ~cut, al & ~cut, be & ~cut


def epochs(t, X, names, fs, lit, onscreen, stage, fstim, first, last, f0):
    sos = butter(2, [1.0, min(40.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    fr = [names.index(n) for n in ("AF7", "AF8") if n in names]
    oz = blinkmask.oz_index(names)
    ears = [i for i, nm in enumerate(names) if nm in ("TP9", "TP10")]
    sos_slow = butter(2, [0.5, 10.0], "bandpass", fs=fs, output="sos")
    slow = np.abs(sosfiltfilt(sos_slow, X[:, fr] - np.median(X[:, fr], axis=0), axis=0)).max(axis=1)
    n = int(round(EPOCH_S * fs))
    starts = np.arange(0, len(t) - n, n)
    starts = starts[(t[starts + n - 1] - t[starts]) < EPOCH_S * 1.25]
    seg = np.stack([Xf[s:s + n] for s in starts])
    seg = seg - seg.mean(axis=1, keepdims=True)
    p2p = np.abs(seg).max(axis=1)
    art = (p2p > ART_UV).any(axis=1) | (p2p < FLAT_UV).any(axis=1)
    slowpk = np.array([slow[s:s + n].max() for s in starts])
    litfrac = np.array([lit[s:s + n].mean() for s in starts])
    onfrac = np.array([onscreen[s:s + n].mean() for s in starts])
    tc = t[starts] + EPOCH_S / 2
    state = np.full(len(starts), MIXED, int)
    state[(litfrac >= 0.9) & (onfrac >= 0.9)] = sof.FLICKER
    state[(litfrac <= 0.1) & (onfrac >= 0.9)] = sof.DARK
    state[(onfrac == 0) & (tc < first)] = sof.PRE
    state[(onfrac == 0) & (tc > last)] = sof.POST
    stg = np.array([np.bincount(stage[s:s + n][stage[s:s + n] >= 0], minlength=len(STAGES)).argmax()
                    if (stage[s:s + n] >= 0).any() else MIXED for s in starts])
    hz = np.array([np.nanmedian(fstim[s:s + n]) if np.isfinite(fstim[s:s + n]).any() else np.nan for s in starts])
    hz = np.round(hz, 1)
    f, P = welch(seg, fs=fs, nperseg=n, axis=1)               # (E, F, ch)

    def bands(mask_hz):
        th, al, be = masks_for(f, mask_hz)
        return th, al, be

    def group(P_e, mask, chans):
        return P_e[mask][:, chans].mean() if chans else np.nan

    def markers(P_e, th, al, be, theta_ok):
        thf, alf, bef = group(P_e, th, fr), group(P_e, al, fr), group(P_e, be, fr)
        ozc = [oz] if oz is not None else []
        tho, beo = group(P_e, th, ozc), group(P_e, be, ozc)
        the, bee = group(P_e, th, ears), group(P_e, be, ears)
        th7 = group(P_e, th, [fr[0]]) if fr else np.nan
        th8 = group(P_e, th, [fr[1]]) if len(fr) > 1 else np.nan
        bw_t, bw_a, bw_b = 3.5, 2.5, 13.0                       # nominal band widths, Hz
        out = dict(th_f=np.log10(thf), rel_th_f=thf * bw_t / (thf * bw_t + alf * bw_a + bef * bw_b),
                   ta_f=np.log10(thf / alf), tb_f=np.log10(thf / bef), fo_th=np.log10(thf / tho),
                   be_f=np.log10(bef), th_oz=np.log10(tho), be_oz=np.log10(beo),
                   th_ear=np.log10(the), be_ear=np.log10(bee), th_af7=np.log10(th7), th_af8=np.log10(th8))
        if not theta_ok:
            for k in ("th_f", "rel_th_f", "ta_f", "tb_f", "fo_th", "th_oz", "th_ear", "th_af7", "th_af8"):
                out[k] = np.nan
        return out
    cache = {}
    rows_s, rows_u = [], []
    th_u, al_u, be_u = bands(f0)
    for i in range(len(starts)):
        mh = hz[i] if np.isfinite(hz[i]) else f0
        if mh not in cache:
            cache[mh] = bands(mh)
        th, al, be = cache[mh]
        theta_ok = not (np.isfinite(hz[i]) and hz[i] < 8.0 and hz[i] != 0)   # 6 / 7.2 Hz block: line inside theta
        rows_s.append(markers(P[i], th, al, be, theta_ok))
        rows_u.append(markers(P[i], th_u, al_u, be_u, True))
    df = pd.DataFrame(rows_s)
    for k, v in pd.DataFrame(rows_u).items():
        df[k + "_u"] = v
    df.insert(0, "t", tc)
    df.insert(1, "minutes", (tc - first) / 60.0)
    df["state"], df["stage"], df["hz"] = state, stg, hz
    df["art"], df["slowpk"], df["litfrac"] = art, slowpk, litfrac
    df["ok"] = ~art & (slowpk <= SLOW_UV)
    return df, Xf


# ----------------------------------------------------------------- statistics
def boot_ci(v, stat=np.mean):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return [float("nan"), float("nan")]
    idx = RNG.integers(0, len(v), (NBOOT, len(v)))
    s = stat(v[idx], axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def signflip_p(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 6:
        return float("nan")
    obs = v.mean()
    null = (RNG.choice([-1.0, 1.0], (NBOOT, len(v))) * v).mean(axis=1)
    return float(((np.abs(null) >= abs(obs)).sum() + 1) / (NBOOT + 1))


def pairs_summary(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=int(len(v)), mean=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), ci=boot_ci(v),
                p=signflip_p(v), positive=int((v > 0).sum()))


def level(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < MIN_LEVEL_N:
        return dict(n=int(len(v)), median=float("nan"), ci=[float("nan")] * 2)
    return dict(n=int(len(v)), median=float(np.median(v)), ci=boot_ci(v, np.median),
                iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))])


def diff_medians(a, b):
    """median(b) - median(a) for independent epoch sets, bootstrap CI, permutation p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_LEVEL_N or len(b) < MIN_LEVEL_N:
        return dict(n=[int(len(a)), int(len(b))], diff=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    obs = float(np.median(b) - np.median(a))
    ia = RNG.integers(0, len(a), (NBOOT, len(a)))
    ib = RNG.integers(0, len(b), (NBOOT, len(b)))
    bs = np.median(b[ib], axis=1) - np.median(a[ia], axis=1)
    pooled = np.concatenate([a, b])
    null = np.empty(NBOOT)
    for k in range(NBOOT):
        perm = RNG.permutation(pooled)
        null[k] = np.median(perm[len(a):]) - np.median(perm[:len(a)])
    p = float(((np.abs(null) >= abs(obs)).sum() + 1) / (NBOOT + 1))
    return dict(n=[int(len(a)), int(len(b))], diff=obs,
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=p)


def theil(x, y, per=10.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    okm = np.isfinite(x) & np.isfinite(y)
    x, y = x[okm], y[okm]
    if len(x) < 8 or x.max() - x.min() < 4:
        return dict(n=int(len(x)), slope=float("nan"), ci=[float("nan")] * 2)
    s, _, lo, hi = stats.theilslopes(y, x, 0.95)
    return dict(n=int(len(x)), slope=float(s * per), ci=[float(lo * per), float(hi * per)],
                span_min=float(x.max() - x.min()))


# ----------------------------------------------------------------- contrasts
def segments(df):
    """Contiguous runs of epochs with one state inside one stage."""
    st = df["state"].to_numpy()
    sg = df["stage"].to_numpy()
    tt = df["t"].to_numpy()
    seg_id = np.zeros(len(df), int)
    k = 0
    for i in range(1, len(df)):
        if st[i] != st[i - 1] or sg[i] != sg[i - 1] or tt[i] - tt[i - 1] > EPOCH_S * 1.5:
            k += 1
        seg_id[i] = k
    return seg_id


def acute(df, marker, stage_sel=None):
    """Lit segment mean minus the nearest black segment mean of the same stage."""
    sub = df[(df["state"].isin([sof.FLICKER, sof.DARK])) & (df["stage"] >= 0)].copy()
    if stage_sel is not None:
        sub = sub[sub["stage"].isin(stage_sel)]
    if sub.empty:
        return pairs_summary([])
    g = sub.groupby("seg")
    segs = pd.DataFrame(dict(state=g["state"].first(), stage=g["stage"].first(),
                             t0=g["t"].min() - EPOCH_S / 2, t1=g["t"].max() + EPOCH_S / 2,
                             val=sub[sub["ok"]].groupby("seg")[marker].mean(),
                             n=sub[sub["ok"]].groupby("seg")[marker].apply(lambda v: int(np.isfinite(v).sum()))))
    segs = segs[np.isfinite(segs["val"]) & (segs["n"] >= 1)]
    litS = segs[segs["state"] == sof.FLICKER]
    dkS = segs[segs["state"] == sof.DARK]
    diffs = []
    for _, L in litS.iterrows():
        cand = dkS[dkS["stage"] == L["stage"]]
        if cand.empty:
            continue
        gap = np.maximum(0, np.maximum(cand["t0"] - L["t1"], L["t0"] - cand["t1"]))
        j = int(np.argmin(gap.to_numpy()))
        if gap.iloc[j] <= PAIR_S:
            diffs.append(L["val"] - cand["val"].iloc[j])
    return pairs_summary(diffs)


def minute_medians(df, sel, marker, min_n=5):
    sub = df[sel & df["ok"] & np.isfinite(df[marker]) & (df["minutes"] >= 0)]
    if sub.empty:
        return pd.Series(dtype=float)
    m = np.floor(sub["minutes"]).astype(int)
    g = sub.groupby(m)[marker]
    out = g.median()[g.count() >= min_n]
    return out


def closure_share(d, t, Xf, names, fs, lit, onscreen, fstim, first, last, df):
    """blinkmask.closure_mask per flicker frequency on screen, spliced by stage.  The
    detector switches itself off when the flicker leaves under 2 Hz of the 8-12 Hz
    band (it does at 9 Hz), so the blue plane and the sweep's 9 Hz blocks are
    unmeasurable: shares are over the measurable task time only."""
    spans = blinkmask.calibration_off_spans(d)
    hzs = sorted(set(np.round(fstim[np.isfinite(fstim)], 1)))
    mask = np.zeros(len(t), bool)
    measurable = np.zeros(len(t), bool)
    bases, notes = {}, {}
    for hz in hzs:
        base = blinkmask.alpha_baseline(t, Xf, names, fs, hz, spans)
        m, used, note = blinkmask.closure_mask(Xf, names, fs, hz, 3.0, 1.0, baseline=base)
        sel = np.isfinite(fstim) & (np.abs(fstim - hz) < 0.05)
        if used is not None:
            mask[sel] = m[sel]
            measurable[sel] = True
        bases[str(hz)], notes[str(hz)] = used, note
    rel = (t - first) / 60.0
    task = (rel >= 0) & (t <= last) & onscreen
    meas = task & measurable
    per5 = []
    for m0 in np.arange(0, np.ceil((last - first) / 60.0 / 5.0) * 5.0, 5.0):
        sel = task & (rel >= m0) & (rel < m0 + 5)
        ok = sel & measurable
        per5.append(dict(minute=float(m0), share=float(mask[ok].mean()) if ok.sum() >= fs * 60 else None,
                         minutes_measurable=float(ok.sum() / fs / 60), minutes_on_task=float(sel.sum() / fs / 60)))
    # episodes
    edges = np.diff(np.concatenate(([0], (mask & meas).astype(int), [0])))
    s0, s1 = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    durs = (s1 - s0) / fs
    # uncertainty: bootstrap over 30 s chunks of measurable task time
    chunk = int(30 * fs)
    idx = np.flatnonzero(meas)
    chunks = [mask[idx[i:i + chunk]].mean() for i in range(0, len(idx) - chunk, chunk)]
    return dict(per_5min=per5, overall=float(mask[meas].mean()), overall_ci=boot_ci(chunks),
                minutes_measurable=float(meas.sum() / fs / 60), minutes_on_task=float(task.sum() / fs / 60),
                lit=float(mask[meas & lit].mean()), dark=float(mask[meas & ~lit].mean()),
                episodes=int(len(durs)), median_episode_s=float(np.median(durs)) if len(durs) else float("nan"),
                baseline_uv2=bases, notes=notes)


# ----------------------------------------------------------------- per session
def analyse(label, d):
    t, X, names, fs = sof.load_eeg(d)
    lit, onscreen, stage, fstim, first, last, f0 = timeline(d, t)
    df, Xf = epochs(t, X, names, fs, lit, onscreen, stage, fstim, first, last, f0)
    df["seg"] = segments(df)
    task = (df["minutes"] >= 0) & (df["t"] <= last)
    is12 = np.isfinite(df["hz"]) & (np.abs(df["hz"] - 12.0) < 0.3)
    is9 = np.isfinite(df["hz"]) & (np.abs(df["hz"] - 9.0) < 0.3)
    lit_e, dark_e = df["state"] == sof.FLICKER, df["state"] == sof.DARK
    res = dict(dir=os.path.relpath(d, ROOT), fs=round(fs, 2), f0=round(f0, 3), channels=names,
               minutes_task=round((last - first) / 60.0, 1), first_flicker_s=round(first - t[0], 1),
               epochs_task=int(task.sum()),
               excluded=dict(any_channel_over_150uV=float((df["art"] & task).mean() / max(task.mean(), 1e-9)),
                             frontal_slow_over_50uV=float(((df["slowpk"] > SLOW_UV) & ~df["art"] & task).sum() / max(task.sum(), 1)),
                             usable_share=float((df["ok"] & task).sum() / max(task.sum(), 1))),
               epochs_by_state=dict(lit=int((lit_e & task & df["ok"]).sum()), dark=int((dark_e & task & df["ok"]).sum()),
                                    lit_12hz=int((lit_e & is12 & df["ok"]).sum()), lit_9hz=int((lit_e & is9 & df["ok"]).sum()),
                                    theta_unmeasurable_lit=int((lit_e & df["ok"] & ~np.isfinite(df["th_f"])).sum())),
               stages={STAGE_LABEL[STAGES[s]]: [round(float(df.loc[df["stage"] == s, "minutes"].min()), 1),
                                                round(float(df.loc[df["stage"] == s, "minutes"].max()), 1)]
                       for s in sorted(df.loc[df["stage"] >= 0, "stage"].unique())})
    # levels
    lev = {}
    for m in MARKERS:
        lev[m] = dict(dark_uniform=level(df.loc[dark_e & task & df["ok"], m + "_u"]),
                      lit_12hz=level(df.loc[lit_e & is12 & df["ok"], m]),
                      lit_9hz=level(df.loc[lit_e & is9 & df["ok"], m]),
                      dark_9hz_stage=level(df.loc[dark_e & is9 & df["ok"], m]))
    res["levels"] = lev
    # acute lit - dark
    stg12 = [i for i, s in enumerate(STAGES)]
    blue = [STAGES.index("plane_blue")] if is9.any() else []
    res["acute"] = {m: dict(all=acute(df, m), twelve_hz=acute(df, m, [s for s in stg12 if s not in blue and s != STAGES.index("calib_blue_sweep")]),
                            sweep=acute(df, m, [STAGES.index("calib_blue_sweep")]),
                            blue_9hz=acute(df, m, blue) if blue else None) for m in MARKERS}
    # drift
    dr = {}
    for m in MARKERS:
        mm_d = minute_medians(df, dark_e & task, m + "_u")
        mm_l = minute_medians(df, lit_e & is12 & task, m)
        sub_d = df[dark_e & task & df["ok"]]
        sub_l = df[lit_e & is12 & task & df["ok"]]
        entry = dict(dark_uniform_minutes=theil(mm_d.index.to_numpy() + 0.5, mm_d.to_numpy()),
                     lit_12hz_minutes=theil(mm_l.index.to_numpy() + 0.5, mm_l.to_numpy()),
                     dark_uniform_epochs=theil(sub_d["minutes"], sub_d[m + "_u"]),
                     lit_12hz_epochs=theil(sub_l["minutes"], sub_l[m]))
        if is9.any():
            sub_b = df[(lit_e | dark_e) & is9 & df["ok"]]
            mm_b = minute_medians(df, (lit_e | dark_e) & is9, m)
            entry["blue_plane_minutes"] = theil(mm_b.index.to_numpy() + 0.5, mm_b.to_numpy())
            entry["blue_plane_epochs"] = theil(sub_b["minutes"], sub_b[m])
        dr[m] = entry
    res["drift"] = dr
    # before / after
    pre = df[(df["t"] >= first - 180) & (df["t"] < first) & df["ok"]]
    post = df[(df["t"] > last + 1) & df["ok"]]
    res["pre_last3min"] = dict(n_ok=int(len(pre)), n_all=int(((df["t"] >= first - 180) & (df["t"] < first)).sum()),
                               **{m: level(pre[m + "_u"]) for m in ("th_f", "tb_f", "be_f")})
    res["post"] = dict(n_ok=int(len(post)), n_all=int((df["t"] > last + 1).sum()),
                       **{m: level(post[m + "_u"]) for m in ("th_f", "tb_f", "be_f")})
    res["closure"] = closure_share(d, t, Xf, names, fs, lit, onscreen, fstim, first, last, df)
    return res, df


def compare_runs(dfs):
    out = {}
    d1, d2 = dfs["run 1"], dfs["run 2"]
    for key, sel_fn, suffix, maxmin in (("dark_uniform_matched_minutes", lambda df: df["state"] == sof.DARK, "_u", 21),
                                        ("dark_uniform_first_7_min", lambda df: df["state"] == sof.DARK, "_u", 7),
                                        ("lit_12hz_matched_minutes", lambda df: (df["state"] == sof.FLICKER) & (np.abs(df["hz"] - 12.0) < 0.3), "", 21)):
        out[key] = {}
        for m in MARKERS:
            a = minute_medians(d1, sel_fn(d1), m + suffix)
            b = minute_medians(d2, sel_fn(d2), m + suffix)
            common = [k for k in a.index if k in b.index and k < maxmin]
            diffs = np.array([b[k] - a[k] for k in common])
            s = pairs_summary(diffs)
            s["minutes"] = [int(k) for k in common]
            out[key][m] = s
    return out


def vs_reference(dfs):
    ref = dfs["reference"]
    out = {}
    for run in ("run 1", "run 2"):
        df = dfs[run]
        out[run] = {}
        for m in MARKERS:
            out[run][m] = dict(
                dark_uniform=diff_medians(ref.loc[(ref["state"] == sof.DARK) & ref["ok"] & (ref["minutes"] >= 0), m + "_u"],
                                          df.loc[(df["state"] == sof.DARK) & df["ok"] & (df["minutes"] >= 0), m + "_u"]),
                lit_12hz=diff_medians(ref.loc[(ref["state"] == sof.FLICKER) & ref["ok"], m],
                                      df.loc[(df["state"] == sof.FLICKER) & df["ok"] & (np.abs(df["hz"] - 12.0) < 0.3), m]))
    return out


# ----------------------------------------------------------------- figure
def running_median(x, y, win_s=60.0):
    x, y = np.asarray(x), np.asarray(y)
    okm = np.isfinite(y)
    x, y = x[okm], y[okm]
    if len(x) < 5:
        return x, y
    half = win_s / 60.0 / 2
    out = np.array([np.median(y[(x >= xi - half) & (x <= xi + half)]) for xi in x])
    return x, out


def figure(results, dfs, runs_cmp, ref_cmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK,
                         "xtick.color": C_INK, "ytick.color": C_INK, "text.color": C_INK,
                         "axes.titlesize": 8.5, "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
                         "legend.frameon": False})
    fig = plt.figure(figsize=(12.5, 13.2))
    gs = fig.add_gridspec(4, 6, height_ratios=[2.3, 2.3, 2.5, 2.3], hspace=0.62, wspace=0.55)

    def tidy(ax):
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def course(ax, run, marker, show_ref=True):
        df = dfs[run]
        r = results[run]
        for st, c, lab in ((sof.FLICKER, C_FLICKER, "lit cell / ON block"), (sof.DARK, C_DARK, "black cell / OFF block")):
            sub = df[(df["state"] == st) & df["ok"] & (df["minutes"] >= 0) & (df["t"] <= df["t"].max())]
            sub = sub[sub["minutes"] <= r["minutes_task"]]
            ax.plot(sub["minutes"], sub[marker], ".", ms=2.2, color=c, alpha=0.45, label=lab, rasterized=True)
            xs, ys = running_median(sub["minutes"], sub[marker])
            ax.plot(xs, ys, "-", lw=1.6, color=c)
        for k, (nm, (a, b)) in enumerate(sorted(r["stages"].items(), key=lambda kv: kv[1][0])):
            if k % 2 == 0:
                ax.axvspan(a, b, color=C_BAND, lw=0, zorder=0)
            ax.text((a + b) / 2, 1.02 + 0.09 * (k % 2), nm, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color=C_INK2)
        if show_ref:
            ref = results["reference"]["levels"][marker]["dark_uniform"]["median"]
            if np.isfinite(ref):
                ax.axhline(ref, color=C_REF, lw=1.0, zorder=1)
                ax.text(ax.get_xlim()[1] if False else r["minutes_task"], ref, " reference day\n (black cells)",
                        fontsize=6.5, color=C_INK2, va="center", ha="left")
        ax.set_xlim(-0.3, r["minutes_task"] + 0.3)
        tidy(ax)

    axes_top = []
    for row, marker in enumerate(("th_f", "be_f")):
        for col, run in enumerate(("run 1", "run 2")):
            ax = fig.add_subplot(gs[row, col * 3:(col + 1) * 3])
            course(ax, run, marker)
            ax.set_title(f"{run}: {results[run]['dir'].split(os.sep)[-1]}, {LABEL[marker]}", loc="left", pad=22)
            ax.set_xlabel("minutes since the first flicker")
            if col == 0:
                ax.set_ylabel(LABEL[marker].split(" (")[0])
            axes_top.append(ax)
    for pair in (axes_top[0:2], axes_top[2:4]):
        lo = min(a.get_ylim()[0] for a in pair)
        hi = max(a.get_ylim()[1] for a in pair)
        for a in pair:
            a.set_ylim(lo, hi)
    h, l = axes_top[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, markerscale=5, bbox_to_anchor=(0.99, 0.965))

    # row 3a: acute lit - dark
    ax = fig.add_subplot(gs[2, 0:2])
    ypos = np.arange(len(FIG_MARKERS))
    for off, (run, mk, mfc) in enumerate(((("run 1"), "o", C_INK), ("run 2", "s", C_INK2), ("reference", "D", "none"))):
        for i, m in enumerate(FIG_MARKERS):
            s = results[run]["acute"][m]["all"]
            if not np.isfinite(s["mean"]):
                continue
            ax.errorbar(s["mean"], ypos[i] + (off - 1) * 0.22, xerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
                        fmt=mk, ms=4.5, color=C_INK if run != "run 2" else C_INK2, mfc=mfc if run == "reference" else (C_INK if run == "run 1" else C_INK2),
                        capsize=2, lw=1, label=run if i == 0 else None)
    ax.axvline(0, color=C_REF, lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL[m].split(" (")[0] for m in FIG_MARKERS])
    ax.invert_yaxis()
    ax.set_xlabel("lit minus nearest black segment (log10), mean, 95 % CI")
    ax.set_title("acute: flicker on screen vs black, all stages", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=7)
    tidy(ax)

    # row 3b: run 2 - run 1 and runs vs reference
    ax = fig.add_subplot(gs[2, 2:4])
    for off, (key, mk, lab) in enumerate((("r21", "o", "run 2 - run 1 (matched min)"),
                                          ("r1ref", "D", "run 1 - reference"),
                                          ("r2ref", "s", "run 2 - reference"))):
        for i, m in enumerate(FIG_MARKERS):
            if m == "fo_th":
                if off == 0:
                    ax.text(0, ypos[i], "not comparable across sessions (Oz contact)", ha="center", va="center",
                            fontsize=6.5, color=C_INK2)
                continue
            if key == "r21":
                s = runs_cmp["dark_uniform_matched_minutes"][m]
                val, ci = s["mean"], s["ci"]
            else:
                s = ref_cmp["run 1" if key == "r1ref" else "run 2"][m]["dark_uniform"]
                val, ci = s["diff"], s["ci"]
            if not np.isfinite(val):
                continue
            col = C_INK if key == "r21" else C_INK2
            ax.errorbar(val, ypos[i] + (off - 1) * 0.22, xerr=[[val - ci[0]], [ci[1] - val]], fmt=mk, ms=4.5,
                        color=col, mfc=col if key != "r1ref" else "none", capsize=2, lw=1, label=lab if i == 0 else None)
    ax.axvline(0, color=C_REF, lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("difference (log10), 95 % CI; black cells")
    ax.set_title("levels: run 2 vs run 1, and runs vs the reference day", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=7, columnspacing=0.8, handletextpad=0.4)
    tidy(ax)

    # row 3c: eye closure share per 5 min
    ax = fig.add_subplot(gs[2, 4:6])
    for run, mk, col, mfc in (("run 1", "o", C_INK, C_INK), ("run 2", "s", C_INK2, C_INK2), ("reference", "D", C_INK2, "none")):
        p5 = results[run]["closure"]["per_5min"]
        if p5:
            ax.plot([e["minute"] + 2.5 for e in p5], [100 * e["share"] if e["share"] is not None else np.nan for e in p5],
                    "-", marker=mk, ms=4.5, color=col, mfc=mfc, lw=1.2, label=run)
    ax.set_xlabel("minutes since the first flicker (5 min bins)")
    ax.set_ylabel("eye-closure share, % of time")
    ax.set_title("eye closure: Oz alpha >= 3x eyes-open baseline, >= 1 s", loc="left")
    ax.set_ylim(bottom=0)
    ax.text(0.98, 0.97, "gaps: 9 Hz blue plane,\nunmeasurable", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=C_INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=7)
    tidy(ax)

    # row 4: drift slopes per run
    for col, run in enumerate(("run 1", "run 2")):
        ax = fig.add_subplot(gs[3, col * 3:(col + 1) * 3])
        for off, (key, c, lab) in enumerate((("lit_12hz_minutes", C_FLICKER, "lit epochs, 12 Hz stages"),
                                             ("dark_uniform_minutes", C_DARK, "black epochs, all stages"))):
            for i, m in enumerate(FIG_MARKERS):
                s = results[run]["drift"][m][key]
                if not np.isfinite(s["slope"]):
                    continue
                ax.errorbar(s["slope"], ypos[i] + (off - 0.5) * 0.26, xerr=[[s["slope"] - s["ci"][0]], [s["ci"][1] - s["slope"]]],
                            fmt="o", ms=4.5, color=c, capsize=2, lw=1, label=lab if i == 0 else None)
        ax.axvline(0, color=C_REF, lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABEL[m].split(" (")[0] for m in FIG_MARKERS] if col == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel("Theil-Sen slope per 10 min of task (log10 units), 95 % interval, per-minute medians")
        ax.set_title(f"drift within {run}", loc="left")
        if col == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=7)
        tidy(ax)
    fig.suptitle("State of Flash, frontal theta and strain: AF7/AF8 theta and beta through tonight's two VR sessions "
                 "(flicker lines, harmonics and subharmonics cut out; 60 s running medians)", x=0.01, y=0.985,
                 ha="left", fontsize=9.5)
    fig.savefig(os.path.join(OUT, NAME + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main
def fmt(s, key="mean"):
    if s is None or not np.isfinite(s.get(key, np.nan)):
        return "n/a"
    return f"{s[key]:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" + (f" p {s['p']:.3f}" if np.isfinite(s.get("p", np.nan)) else "") + f" n {s['n']}"


def main():
    os.makedirs(OUT, exist_ok=True)
    results, dfs = {}, {}
    for label, name in SESSIONS:
        d = os.path.join(ROOT, "runs", name)
        print(f"== {label}: {name}")
        res, df = analyse(label, d)
        results[label], dfs[label] = res, df
        print(f"   task {res['minutes_task']} min, usable epochs {res['excluded']['usable_share']:.2f}, "
              f"lit {res['epochs_by_state']['lit']} dark {res['epochs_by_state']['dark']}; stages {res['stages']}")
        for m in FIG_MARKERS:
            print(f"   {LABEL[m]:36s} dark level {results[label]['levels'][m]['dark_uniform']['median']:+.3f}  "
                  f"acute all {fmt(res['acute'][m]['all'])}  drift dark {fmt(res['drift'][m]['dark_uniform_minutes'], 'slope')}")
        c = res["closure"]
        print(f"   closure share overall {100 * c['overall']:.2f} % [{100 * c['overall_ci'][0]:.2f}, {100 * c['overall_ci'][1]:.2f}]"
              f"  lit {100 * c['lit']:.2f} dark {100 * c['dark']:.2f}  measurable {c['minutes_measurable']:.1f} of {c['minutes_on_task']:.1f} min"
              f"  per 5 min {[round(100 * e['share'], 2) if e['share'] is not None else None for e in c['per_5min']]}")
    runs_cmp = compare_runs(dfs)
    ref_cmp = vs_reference(dfs)
    print("== run 2 minus run 1 at matched minutes (black epochs, common mask)")
    for m in MARKERS:
        print(f"   {LABEL[m]:36s} {fmt(runs_cmp['dark_uniform_matched_minutes'][m])}   first 7 min {fmt(runs_cmp['dark_uniform_first_7_min'][m])}")
    print("== vs reference day (black epochs, common mask): run 1 | run 2")
    for m in MARKERS:
        print(f"   {LABEL[m]:36s} {fmt(ref_cmp['run 1'][m]['dark_uniform'], 'diff')} | {fmt(ref_cmp['run 2'][m]['dark_uniform'], 'diff')}")
    notes = [
        "One subject; every interval is over windows (epochs, segments or minutes) of that session, not over people.",
        "Frontal = mean of AF7 and AF8, lateral frontal sites referenced to FPz on the Muse; this is not Fz midline theta.",
        "Absolute power levels between sessions depend on electrode contact (the headset was re-seated); ratios are more robust, "
        "and the frontal/occipital theta ratio across sessions mostly follows the Oz cup's contact.",
        "Theta is unmeasurable while a 6 or 7.2 Hz block is on screen (the line is inside the band); those lit epochs are left out of theta.",
        "The 9 Hz blue plane uses the 9 Hz mask (theta 5.5-7.5 Hz only, alpha 10-10.5 Hz), so its levels are not pooled with 12 Hz stages; "
        "black epochs use one common 12 Hz mask for drift and run comparisons.",
        "A headache has no direct EEG signature; forehead ache can only be approached indirectly (frontal beta as tension, theta as effort).",
        "The pre-stimulus minutes are too artifacted to serve as a baseline (few usable epochs); see pre_last3min.n_ok.",
        "Eye closure (blinkmask.closure_mask) is unmeasurable while 9 Hz is on screen (the line leaves under 2 Hz of the "
        "8-12 Hz band and the detector switches itself off): the blue plane and the sweep's 9 Hz blocks are left out, and "
        "the detector cannot tell a closed eye from a spontaneous alpha burst.",
        "Run 2's TP10 exceeded 150 uV in about a fifth of the epochs (minutes 10-20 most), so ear-channel levels and drift "
        "in run 2 follow that contact, not the brain; frontal markers lose those epochs by the any-channel rule.",
    ]
    out = dict(marker_family="frontal theta and strain", date="2026-10-10",
               method=dict(epoch_s=EPOCH_S, artifact_uv=ART_UV, frontal_slow_uv=SLOW_UV, pair_window_s=PAIR_S,
                           bootstrap=NBOOT, bands=dict(theta=[4, 7.5], alpha=[8, 10.5], beta=[[15, 23], [25, 30]]),
                           masks="analysis.state_of_flash.band_masks at the frequency on screen, plus the 4th and 5th harmonic for the sweep's 6 and 7.2 Hz blocks"),
               markers=LABEL, sessions=results, run2_minus_run1=runs_cmp, vs_reference=ref_cmp, notes=notes)
    json.dump(out, open(os.path.join(OUT, NAME + ".json"), "w"), indent=1, default=float)
    figure(results, dfs, runs_cmp, ref_cmp)
    print(f"wrote {os.path.join(OUT, NAME + '.json')} and .png")
    return 0


if __name__ == "__main__":
    # python analysis/flash_frontal_theta.py tag=_all "run 1=vr_planes9" "run 2=vr_planes10" reference=vr_planes8
    # (the sessions labelled "run 1", "run 2" and "reference" fill the comparison tables)
    args = sys.argv[1:]
    if args:
        sess = []
        for a in args:
            k, _, v = a.partition("=")
            if k == "tag":
                NAME = NAME + v
            else:
                sess.append((k, v))
        if sess:
            SESSIONS = sess
    sys.exit(main())
