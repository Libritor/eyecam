"""Independent check of one claim about the 2026-10-10 flicker sessions:

    frontal (AF7/AF8) 1-4 Hz log10 power is lower while a lit, flickering
    cell is on screen than while a black cell / OFF block is shown
    (about -0.16 log10 pooled over tonight's 12 Hz stages), and that gap is
    not photic leakage.

Everything here is recomputed from eeg.csv and the stage logs with its own
code.  The lit / dark label is the luminance column of each stage log.  The
things the original analysis did not do, and which could make the gap look
better than it is, are tested one by one:

  block permutation   epochs inside one 8 s ON block or one run of lit cells
                      share the same slow eye state, so lit / dark labels are
                      shuffled across BLOCKS within each stage, not across
                      2 s epochs
  adjacent pairs      each lit block against the dark block next to it in
                      time (block means), sign-flip test: no drift confound
  blink-free epochs   the gap recomputed on epochs with no sample above 50 uV
                      in the 0.5-10 Hz frontal signal: is it blinks?
  interior epochs     the first 2 s after every lit<->dark transition dropped:
                      is it an onset / offset transient?
  drift residuals     Theil-Sen trend over time removed within each stage
  channels            AF7, AF8 alone, Oz (AUX) and the ears (TP9 / TP10)
  estimator           Hann periodogram band power and, independently, the
                      variance of a 1-4 Hz band-passed signal per epoch

    python <this file>          (working directory: the eyecam repo root)

Writes runs/state_of_flash/refute_confounds_5_a.json and .png.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get("EYECAM_ROOT", r"C:\Users\alexa\eyecam")
sys.path.insert(0, ROOT)
import blinkmask  # noqa: E402

NAME = "refute_confounds_5_a"
OUT = os.path.join(ROOT, "runs", "state_of_flash")
SESSIONS = [("vr_planes9", "run 1 (17:57)"), ("vr_planes10", "run 2 (19:50)"),
            ("vr_planes8", "reference 10-07")]
EPOCH_S = 2.0
ART_UV = 150.0
PURE = 0.9
BLINK_STRICT_UV = 50.0
B_BOOT, B_PERM = 2000, 4000
C_LIT, C_DARK, C_INK, C_GREY = "#2a78d6", "#eb6834", "#3a3a37", "#b5b4ad"
LF = (1.0, 4.0)
THETA = (4.0, 7.5)


# ------------------------------------------------------------------ loading
def load_eeg(d):
    df = pd.read_csv(os.path.join(d, "eeg.csv"))
    t = df.iloc[:, 0].to_numpy(float)
    names = [c.strip() for c in df.columns[1:]]
    X = df.iloc[:, 1:].to_numpy(float)
    dt = np.diff(t)
    good = dt < 0.5
    fs = float(good.sum() / dt[good].sum())
    return t, X, names, fs


def stage_hz(kind, meta):
    if kind.endswith("_sweep"):
        return -1.0
    if kind.startswith("plane_"):
        return float((meta.get("freq_delivered") or {}).get(kind.split("_")[1], 12.0))
    return 12.0


def labels(d, t):
    """Per sample: lit (luminance > 0.05), stage index, stimulus Hz, block id.
    A block is one run of the same lit state inside one stage (an 8 s ON or
    OFF block, or a run of lit cells / black cells in a plane scan)."""
    try:
        meta = json.load(open(os.path.join(d, "planes_meta.json")))
    except Exception:
        meta = {}
    n = len(t)
    lit = np.zeros(n, bool)
    stage = np.full(n, -1, np.int16)
    hz = np.zeros(n)
    blk = np.full(n, -1, np.int32)
    stages, nblk = [], 0
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        L = pd.read_csv(p)
        if len(L) < 10 or "luminance" not in L.columns:
            continue
        lt = L.iloc[:, 0].to_numpy(float)
        lum = L["luminance"].to_numpy(float)
        o = np.argsort(lt, kind="stable")
        lt, lum = lt[o], lum[o]
        ll = lum > 0.05
        bid = np.concatenate([[0], np.cumsum(np.diff(ll.astype(int)) != 0)]) + nblk
        nblk = int(bid[-1]) + 1
        sid = len(stages)
        f = stage_hz(kind, meta)
        stages.append(dict(name=kind, start=float(lt[0]), end=float(lt[-1]), hz=f))
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5) & (t <= lt[-1] + 0.05)
        lit[ok] = ll[idx[ok]]
        stage[ok] = sid
        hz[ok] = f
        blk[ok] = bid[idx[ok]]
    first = min(s["start"] for s in stages)
    last = max(s["end"] for s in stages)
    return lit, stage, hz, blk, stages, first, last


def line_cut(f, code, extra_mains=True):
    """Bins to drop from a band: the stimulus line, all harmonics, the
    subharmonic; mains 60 +- 2 Hz."""
    cut = np.abs(f - 60.0) <= 2.0 if extra_mains else np.zeros(len(f), bool)
    bases = [code] if code > 0 else [6.0, 7.2, 9.0, 12.0]
    for b in bases:
        for k in range(1, int(128 // b) + 1):
            cut |= np.abs(f - k * b) <= 0.75
        cut |= np.abs(f - b / 2) <= 0.6
    return cut


def rail_flat(x, fs):
    """Samples sitting on an ADC rail or in a flat run of >= 0.25 s."""
    rail = (x <= 0.5) | (x >= 1649.5)
    same = np.concatenate([[False], np.diff(x) == 0])
    w = int(round(0.25 * fs))
    run = np.convolve(same.astype(float), np.ones(w), mode="same") >= w - 0.5
    return rail | run


# ------------------------------------------------------------------ epochs
def epochs(d):
    t, X, names, fs = load_eeg(d)
    lit, stage, hz, blk, stages, first, last = labels(d, t)
    n = int(round(EPOCH_S * fs))
    sos_bb = butter(2, [1.0, 40.0], "bandpass", fs=fs, output="sos")
    sos_lf = butter(4, [1.0, 4.0], "bandpass", fs=fs, output="sos")
    sos_bl = butter(2, [0.5, 10.0], "bandpass", fs=fs, output="sos")
    Xbb = sosfiltfilt(sos_bb, X, axis=0)
    Xlf = sosfiltfilt(sos_lf, X, axis=0)
    badsmp = np.stack([rail_flat(X[:, i], fs) for i in range(X.shape[1])], axis=1)
    af = [names.index("AF7"), names.index("AF8")]
    oz = blinkmask.oz_index(names)
    ears = [names.index("TP9"), names.index("TP10")]
    blink100 = blinkmask.blink_mask(X, names, fs, 100.0)
    y = sosfiltfilt(sos_bl, X[:, af] - np.median(X[:, af], axis=0), axis=0)
    blink50 = (np.abs(y) > BLINK_STRICT_UV).any(axis=1)
    edges50 = np.concatenate([[False], np.diff(blink50.astype(int)) == 1])
    # block start times (first sample of each block, in time order)
    ub, first_idx = np.unique(blk, return_index=True)
    bstart = dict(zip(ub.tolist(), t[first_idx].tolist()))

    rows = []
    for sid, s in enumerate(stages):
        for s0 in np.arange(s["start"], s["end"] - EPOCH_S + 1e-6, EPOCH_S):
            i = int(np.searchsorted(t, s0))
            if i + n > len(t):
                break
            span = t[i + n - 1] - t[i]
            if span > EPOCH_S * 1.25 or span < EPOCH_S * 0.75:
                continue
            sl = slice(i, i + n)
            cov = (stage[sl] == sid).mean()
            share = lit[sl].mean()
            if cov < PURE:
                continue
            state = "lit" if share >= PURE else ("dark" if share <= 1 - PURE else "mixed")
            b = int(np.bincount(blk[sl][blk[sl] >= 0]).argmax()) if (blk[sl] >= 0).any() else -1
            seg = X[sl] - X[sl].mean(axis=0)
            f, P = welch(seg, fs=fs, nperseg=n, axis=0)             # Hann, one segment
            cut = line_cut(f, s["hz"])
            inlf = (f >= LF[0]) & (f <= LF[1])
            inth = (f >= THETA[0]) & (f <= THETA[1])
            peak = np.abs(Xbb[sl] - Xbb[sl].mean(axis=0)).max(axis=0)
            bad = (peak > ART_UV) | badsmp[sl].any(axis=0)
            r = dict(session=os.path.basename(d), stage=s["name"], sid=sid, hz=s["hz"],
                     t=float(t[i] + EPOCH_S / 2), minutes=float((t[i] + EPOCH_S / 2 - first) / 60),
                     state=state, lit_share=float(share), block=b,
                     t_since=float(t[i] - bstart.get(b, t[i])),
                     blink100=float(blink100[sl].mean()), blink50=float(blink50[sl].mean()),
                     edges50=int(edges50[sl].sum()),
                     af_bad=bool(bad[af].any()), any_bad=bool(bad.any()),
                     oz_bad=bool(bad[oz]) if oz is not None else True,
                     ear_bad=bool(bad[ears].any()))
            for ci, ch in enumerate(names):
                r[f"bad_{ch}"] = bool(bad[ci])
                r[f"lf_in_{ch}"] = float(np.log10(P[inlf, ci].mean()))
                kc = inlf & ~cut
                r[f"lf_cut_{ch}"] = float(np.log10(P[kc, ci].mean())) if kc.any() else np.nan
                kt = inth & ~cut
                r[f"th_cut_{ch}"] = float(np.log10(P[kt, ci].mean())) if kt.any() else np.nan
                r[f"lf_td_{ch}"] = float(np.log10(Xlf[sl, ci].var()))
            r["lf_in_AF"] = float(np.mean([r["lf_in_AF7"], r["lf_in_AF8"]]))
            r["lf_cut_AF"] = float(np.mean([r["lf_cut_AF7"], r["lf_cut_AF8"]]))
            r["th_cut_AF"] = float(np.mean([r["th_cut_AF7"], r["th_cut_AF8"]]))
            r["lf_td_AF"] = float(np.mean([r["lf_td_AF7"], r["lf_td_AF8"]]))
            r["lf_in_TP"] = float(np.mean([r["lf_in_TP9"], r["lf_in_TP10"]]))
            # low-resolution spectrum 0.5-20 Hz of the AF mean, for the figure
            r["_spec"] = np.log10(P[(f >= 0.5) & (f <= 20), :][:, af].mean(axis=1))
            rows.append(r)
    df = pd.DataFrame(rows)
    fspec = f[(f >= 0.5) & (f <= 20)]
    meta = dict(fs=fs, names=names, first=first, last=last, stages=stages,
                n_epochs=int(len(df)), fspec=fspec.tolist())
    return df, meta


# ------------------------------------------------------------------ statistics
def _blocks(v, b):
    ub, inv = np.unique(b, return_inverse=True)
    return ub, [v[inv == k] for k in range(len(ub))]


def contrast(va, ba, vb, bb, sa=None, sb=None, seed=0, stat=np.median):
    """stat(lit) - stat(dark) over epochs.  CI: block bootstrap (blocks
    resampled with replacement inside each state).  p_epoch: labels shuffled
    across epochs (the original analysis).  p_block: lit/dark labels shuffled
    across blocks within each stage (epochs keep their block)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    ok_a, ok_b = np.isfinite(va), np.isfinite(vb)
    va, ba, vb, bb = va[ok_a], np.asarray(ba)[ok_a], vb[ok_b], np.asarray(bb)[ok_b]
    if sa is not None:
        sa, sb = np.asarray(sa)[ok_a], np.asarray(sb)[ok_b]
    out = dict(n_lit=int(len(va)), n_dark=int(len(vb)),
               blocks_lit=int(len(np.unique(ba))), blocks_dark=int(len(np.unique(bb))))
    if len(va) < 4 or len(vb) < 4:
        out.update(diff=float("nan"), mean_diff=float("nan"), ci=[float("nan")] * 2,
                   p_epoch=float("nan"), p_block=float("nan"))
        return out
    rng = np.random.default_rng(seed)
    obs = float(stat(va) - stat(vb))
    out["diff"] = obs
    out["mean_diff"] = float(va.mean() - vb.mean())
    # block bootstrap
    _, ga = _blocks(va, ba)
    _, gb = _blocks(vb, bb)
    boots = np.empty(B_BOOT)
    for r in range(B_BOOT):
        ia = rng.integers(0, len(ga), len(ga))
        ib = rng.integers(0, len(gb), len(gb))
        boots[r] = stat(np.concatenate([ga[k] for k in ia])) - stat(np.concatenate([gb[k] for k in ib]))
    out["ci"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
    # epoch permutation
    pooled = np.concatenate([va, vb])
    perm = rng.permuted(np.tile(pooled, (B_PERM, 1)), axis=1)
    dp = stat(perm[:, :len(va)], axis=1) - stat(perm[:, len(va):], axis=1)
    out["p_epoch"] = float((np.sum(np.abs(dp) >= abs(obs)) + 1) / (B_PERM + 1))
    # block permutation, stratified by stage
    v = pooled
    b = np.concatenate([ba, bb])
    lab = np.concatenate([np.ones(len(va), bool), np.zeros(len(vb), bool)])
    s = np.concatenate([sa, sb]) if sa is not None else np.zeros(len(v), int)
    s = np.unique(np.asarray(s).astype(str), return_inverse=True)[1]
    ub, pos = np.unique(b, return_inverse=True)
    blab = np.zeros(len(ub), bool)
    bstage = np.zeros(len(ub), int)
    for k in range(len(ub)):
        m = pos == k
        blab[k] = lab[m][0]
        bstage[k] = s[m][0]
    stages_u = np.unique(bstage)
    cnt = 0
    for r in range(B_PERM):
        nl = blab.copy()
        for st in stages_u:
            idx = np.flatnonzero(bstage == st)
            nl[idx] = nl[idx][rng.permutation(len(idx))]
        el = nl[pos]
        if el.sum() < 2 or (~el).sum() < 2:
            continue
        d = stat(v[el]) - stat(v[~el])
        cnt += abs(d) >= abs(obs)
    out["p_block"] = float((cnt + 1) / (B_PERM + 1))
    return out


def paired_blocks(sub, col, seed=0):
    """Adjacent lit / dark blocks inside each stage, block means of col.
    Sign-flip permutation on the mean paired difference."""
    pairs = []
    for sid, g in sub.groupby("sid"):
        bl = g.groupby("block").agg(t=("t", "min"), state=("state", "first"), v=(col, "mean"),
                                    n=(col, "size")).sort_values("t")
        bl = bl[bl["n"] >= 1]
        st = bl["state"].to_list()
        vals = bl["v"].to_numpy()
        i = 0
        while i < len(bl) - 1:
            if st[i] != st[i + 1]:
                lit_v = vals[i] if st[i] == "lit" else vals[i + 1]
                dark_v = vals[i + 1] if st[i] == "lit" else vals[i]
                if np.isfinite(lit_v) and np.isfinite(dark_v):
                    pairs.append(lit_v - dark_v)
                i += 2
            else:
                i += 1
    pairs = np.array(pairs)
    out = dict(n_pairs=int(len(pairs)))
    if len(pairs) < 5:
        out.update(median=float("nan"), mean=float("nan"), ci=[float("nan")] * 2,
                   p_signflip=float("nan"), share_negative=float("nan"))
        return out
    rng = np.random.default_rng(seed)
    obs = pairs.mean()
    flips = rng.choice([-1.0, 1.0], size=(B_PERM, len(pairs)))
    null = (flips * pairs[None, :]).mean(axis=1)
    boot = np.array([np.median(rng.choice(pairs, len(pairs))) for _ in range(B_BOOT)])
    out.update(median=float(np.median(pairs)), mean=float(obs),
               ci=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
               p_signflip=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B_PERM + 1)),
               share_negative=float((pairs < 0).mean()))
    return out


def theil(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return dict(n=int(ok.sum()), slope_per_10min=float("nan"), ci=[float("nan")] * 2)
    sl, ic, lo, hi = stats.theilslopes(y[ok], x[ok], alpha=0.95)
    return dict(n=int(ok.sum()), slope_per_10min=float(sl * 10), ci=[float(lo * 10), float(hi * 10)])


def residualise(sub, col):
    """col minus a Theil-Sen trend over time fitted within each stage to lit
    and dark epochs together."""
    res = np.full(len(sub), np.nan)
    for sid, idx in sub.groupby("sid").indices.items():
        g = sub.iloc[idx]
        x, y = g["t"].to_numpy(), g[col].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 10:
            res[idx] = y - np.nanmean(y)
            continue
        sl, ic = stats.theilslopes(y[ok], x[ok])[:2]
        res[idx] = y - (sl * x + ic)
    return res


def group_tests(sub, label, seed=0):
    """All tests of the lit - dark gap on one set of epochs (one stimulus-
    frequency group of one or several sessions)."""
    sub = sub.copy()
    sub["bkey"] = sub["session"] + ":" + sub["block"].astype(str)
    sub["skey"] = sub["session"] + ":" + sub["sid"].astype(str)
    res = dict(label=label, n_lit_all=int((sub["state"] == "lit").sum()),
               n_dark_all=int((sub["state"] == "dark").sum()))

    def run(sel, col, seed):
        a, b = sub[sel & (sub["state"] == "lit")], sub[sel & (sub["state"] == "dark")]
        return contrast(a[col], a["bkey"], b[col], b["bkey"], a["skey"], b["skey"], seed=seed)

    ok = ~sub["af_bad"]
    res["AF_lf_in"] = run(ok, "lf_in_AF", seed)
    res["AF_lf_cut"] = run(ok, "lf_cut_AF", seed + 1)
    res["AF_lf_timedomain"] = run(ok, "lf_td_AF", seed + 2)
    res["AF_theta_cut"] = run(ok, "th_cut_AF", seed + 3)
    res["AF_lf_in_strict_any_channel_clean"] = run(~sub["any_bad"], "lf_in_AF", seed + 4)
    res["AF_lf_in_blinkfree_50uV"] = run(ok & (sub["blink50"] == 0), "lf_in_AF", seed + 5)
    res["AF_lf_in_blink100_le20pct"] = run(ok & (sub["blink100"] <= 0.2), "lf_in_AF", seed + 6)
    res["AF_lf_in_interior_t_since_ge_2s"] = run(ok & (sub["t_since"] >= 2.0 - 0.05), "lf_in_AF", seed + 7)
    res["AF_lf_in_first_epoch_of_block"] = run(ok & (sub["t_since"] < 2.0 - 0.05), "lf_in_AF", seed + 8)
    sub["lf_resid_AF"] = residualise(sub, "lf_in_AF")
    res["AF_lf_in_drift_residual"] = run(ok, "lf_resid_AF", seed + 9)
    res["AF_lf_in_paired_adjacent_blocks"] = paired_blocks(sub[ok], "lf_in_AF", seed + 10)
    res["AF_lf_in_paired_adjacent_blocks_blinkfree"] = paired_blocks(sub[ok & (sub["blink50"] == 0)], "lf_in_AF", seed + 11)
    for ch in ("AF7", "AF8", "AUX", "TP9", "TP10"):
        res[f"{ch}_lf_in"] = run(~sub[f"bad_{ch}"], f"lf_in_{ch}", seed + 12)
    # blinks: rate per minute and share of epochs touched, lit vs dark
    a, b = sub[ok & (sub["state"] == "lit")], sub[ok & (sub["state"] == "dark")]
    res["blinks50_per_min"] = dict(
        lit=float(a["edges50"].sum() / (len(a) * EPOCH_S) * 60) if len(a) else float("nan"),
        dark=float(b["edges50"].sum() / (len(b) * EPOCH_S) * 60) if len(b) else float("nan"),
        contrast_epochs=contrast(a["edges50"], a["bkey"], b["edges50"], b["bkey"], a["skey"], b["skey"],
                                 seed=seed + 13, stat=np.mean))
    res["blink50_share_of_epochs_touched"] = dict(lit=float((a["blink50"] > 0).mean()) if len(a) else float("nan"),
                                                  dark=float((b["blink50"] > 0).mean()) if len(b) else float("nan"))
    res["af_bad_share"] = dict(lit=float(sub.loc[sub["state"] == "lit", "af_bad"].mean()),
                               dark=float(sub.loc[sub["state"] == "dark", "af_bad"].mean()))
    # drift of the marker over minutes on task, lit and dark separately
    res["drift_theil_sen"] = {st: theil(sub.loc[ok & (sub["state"] == st), "minutes"].to_numpy(),
                                        sub.loc[ok & (sub["state"] == st), "lf_in_AF"].to_numpy())
                              for st in ("lit", "dark")}
    # per stage
    res["per_stage"] = {}
    for sname, g in sub[ok].groupby("stage"):
        a, b = g[g["state"] == "lit"], g[g["state"] == "dark"]
        if len(a) >= 4 and len(b) >= 4:
            c = contrast(a["lf_in_AF"], a["bkey"], b["lf_in_AF"], b["bkey"], a["skey"], b["skey"], seed=seed + 20)
            res["per_stage"][sname] = {k: c[k] for k in ("n_lit", "n_dark", "blocks_lit", "blocks_dark",
                                                        "diff", "ci", "p_epoch", "p_block")}
    return res


def gap_boot(sub, col="lf_in_AF", B=B_BOOT, seed=0):
    """Bootstrap draws of median(lit) - median(dark) with blocks resampled."""
    sub = sub[~sub["af_bad"] & sub["state"].isin(["lit", "dark"]) & np.isfinite(sub[col])]
    rng = np.random.default_rng(seed)
    ga = _blocks(sub.loc[sub["state"] == "lit", col].to_numpy(), sub.loc[sub["state"] == "lit", "block"].to_numpy())[1]
    gb = _blocks(sub.loc[sub["state"] == "dark", col].to_numpy(), sub.loc[sub["state"] == "dark", "block"].to_numpy())[1]
    out = np.empty(B)
    for r in range(B):
        ia = rng.integers(0, len(ga), len(ga))
        ib = rng.integers(0, len(gb), len(gb))
        out[r] = np.median(np.concatenate([ga[k] for k in ia])) - np.median(np.concatenate([gb[k] for k in ib]))
    return out


# ------------------------------------------------------------------ figure
def figure(frames, results, metas, path):
    fig, axes = plt.subplots(3, 3, figsize=(17, 16), gridspec_kw=dict(height_ratios=[1, 1.6, 1]))
    fig.patch.set_facecolor("white")
    short = {"calib": "white cal", "calib_red": "red cal", "calib_green": "green cal", "calib_blue": "blue cal",
             "calib_blue_sweep": "blue sweep", "plane_blue": "blue plane", "plane_red": "red plane",
             "plane_green": "green plane"}
    # row 1: block means over minutes on task, per session
    for j, (name, lab) in enumerate(SESSIONS):
        ax = axes[0, j]
        df = frames[name]
        sub = df[~df["af_bad"] & df["state"].isin(["lit", "dark"]) & (df["hz"] > 0)]
        bl = sub.groupby("block").agg(m=("minutes", "mean"), v=("lf_in_AF", "mean"), state=("state", "first"),
                                      hz=("hz", "first"), n=("lf_in_AF", "size"))
        for st, c in (("lit", C_LIT), ("dark", C_DARK)):
            g = bl[bl["state"] == st]
            ax.scatter(g["m"], g["v"], s=12 + 3 * g["n"], color=c, alpha=0.75, label=f"{st} block (size = epochs)",
                       edgecolor="none")
        ylo, yhi = bl["v"].min() - 0.05, bl["v"].max() + 0.05
        ax.set_ylim(ylo - 0.45, yhi)
        for s in metas[name]["stages"]:
            x0 = (s["start"] - metas[name]["first"]) / 60
            ax.axvline(x0, color=C_GREY, lw=0.6)
            ax.text(x0 + 0.1, ylo - 0.42, short.get(s["name"], s["name"]), fontsize=7, color=C_INK, rotation=90,
                    va="bottom")
        ax.set_title(f"{lab}: AF7/AF8 1-4 Hz log10 power, block means", loc="left", fontsize=10)
        ax.set_xlabel("minutes after the first flicker")
        ax.set_ylabel("log10 uV^2/Hz")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")
    # row 2 left: gap estimates with CI across variants, per group
    ax = axes[1, 0]
    variants = [("AF_lf_in", "all epochs"), ("AF_lf_in_blinkfree_50uV", "blink-free"),
                ("AF_lf_in_interior_t_since_ge_2s", "interior epochs"), ("AF_lf_in_drift_residual", "drift residual")]
    groups = [k for k in results["groups"]]
    yt, yl = [], []
    y = 0
    for gk in groups:
        r = results["groups"][gk]
        for vk, vl in variants:
            c = r[vk]
            if not np.isfinite(c["diff"]):
                y += 1
                continue
            col = C_INK if c["p_block"] < 0.05 else C_GREY
            ax.errorbar(c["diff"], y, xerr=[[c["diff"] - c["ci"][0]], [c["ci"][1] - c["diff"]]], fmt="o", color=col,
                        ms=4, capsize=2, lw=1)
            ax.text(0.42, y, f"p_blk {c['p_block']:.3f}", fontsize=7, va="center", color=col)
            yt.append(y)
            yl.append(f"{gk} | {vl}")
            y += 1
        y += 0.6
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.axvline(-0.163, color=C_DARK, lw=0.8, ls="--")
    ax.text(-0.163, y, "claimed -0.163", fontsize=7, color=C_DARK, ha="center")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("median(lit) - median(dark), log10 power; block bootstrap 95 %")
    ax.set_title("Lit - dark gap, AF7/AF8 1-4 Hz (dark grey = block-permutation p < 0.05)", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    # row 2 middle: adjacent-block pairs per group
    ax = axes[1, 1]
    xs, labs = [], []
    for i, gk in enumerate(groups):
        pr = results["groups"][gk]["AF_lf_in_paired_adjacent_blocks"]
        prb = results["groups"][gk]["AF_lf_in_paired_adjacent_blocks_blinkfree"]
        for k, (p, off, c) in enumerate(((pr, -0.15, C_INK), (prb, 0.15, C_LIT))):
            if np.isfinite(p["median"]):
                ax.errorbar(i + off, p["median"], yerr=[[p["median"] - p["ci"][0]], [p["ci"][1] - p["median"]]],
                            fmt="o", color=c, capsize=3)
                ax.text(i + off, p["ci"][1] + 0.01, f"n {p['n_pairs']}\np {p['p_signflip']:.3f}", fontsize=7, ha="center",
                        color=c)
        xs.append(i)
        labs.append(gk)
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7, rotation=20)
    ax.set_ylabel("lit block - adjacent dark block, log10 power")
    ax.set_title("Adjacent lit / dark block pairs (black: all, blue: blink-free), sign-flip p", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    # row 2 right: spectrum of lit minus dark per session, 12 Hz stages
    ax = axes[1, 2]
    for (name, lab), c in zip(SESSIONS, (C_LIT, C_INK, C_GREY)):
        df = frames[name]
        sub = df[~df["af_bad"] & (df["hz"] == 12.0) & df["state"].isin(["lit", "dark"])]
        if len(sub) < 10:
            continue
        S = np.stack(sub["_spec"].to_list())
        d = np.median(S[(sub["state"] == "lit").to_numpy()], axis=0) - np.median(S[(sub["state"] == "dark").to_numpy()], axis=0)
        ax.plot(metas[name]["fspec"], d, color=c, lw=1.4, label=lab)
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.axvspan(1, 4, color=C_DARK, alpha=0.12, lw=0)
    ax.axvline(12, color=C_GREY, lw=0.8, ls=":")
    ax.set_xlabel("Hz (0.5 Hz bins)")
    ax.set_ylabel("median lit - median dark, log10 PSD")
    ax.set_title("Where the gap sits: AF7/AF8 spectrum difference, 12 Hz stages", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    # row 3 left: blink rate lit vs dark
    ax = axes[2, 0]
    w = 0.38
    for i, gk in enumerate(groups):
        r = results["groups"][gk]["blinks50_per_min"]
        ax.bar(i - w / 2, r["lit"], w, color=C_LIT)
        ax.bar(i + w / 2, r["dark"], w, color=C_DARK)
        ax.text(i, max(r["lit"], r["dark"]) + 0.3, f"p_blk {r['contrast_epochs']['p_block']:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7, rotation=20)
    ax.set_ylabel("blink edges per minute (0.5-10 Hz AF > 50 uV)")
    ax.set_title("Blink rate while lit (blue) and dark (orange)", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    # row 3 middle: per channel
    ax = axes[2, 1]
    chans = ["AF7", "AF8", "AUX", "TP9", "TP10"]
    for i, gk in enumerate(groups):
        for k, ch in enumerate(chans):
            c = results["groups"][gk][f"{ch}_lf_in"]
            if np.isfinite(c["diff"]):
                col = C_INK if c["p_block"] < 0.05 else C_GREY
                ax.errorbar(k + (i - len(groups) / 2) * 0.12, c["diff"],
                            yerr=[[c["diff"] - c["ci"][0]], [c["ci"][1] - c["diff"]]], fmt="o", ms=3, color=col, capsize=2)
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans)
    ax.set_ylabel("median lit - median dark, 1-4 Hz log10 power")
    ax.set_title("Per channel (each group a column of dots; dark grey = block p < 0.05)", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    # row 3 right: epoch-level vs block-level p for pooled tonight
    ax = axes[2, 2]
    ax.axis("off")
    lines = ["Pooled tonight, AF7/AF8 1-4 Hz, lit - dark (log10):", ""]
    for gk in groups:
        r = results["groups"][gk]
        c = r["AF_lf_in"]
        lines.append(f"{gk}")
        lines.append(f"   all epochs      {c['diff']:+.3f}  [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  n {c['n_lit']}/{c['n_dark']} "
                     f"blocks {c['blocks_lit']}/{c['blocks_dark']}")
        lines.append(f"   p epoch-shuffle {c['p_epoch']:.4f}   p block-shuffle {c['p_block']:.4f}")
        b = r["AF_lf_in_blinkfree_50uV"]
        lines.append(f"   blink-free      {b['diff']:+.3f}  [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]  n {b['n_lit']}/{b['n_dark']}  p_blk {b['p_block']:.4f}")
        p = r["AF_lf_in_paired_adjacent_blocks"]
        lines.append(f"   adjacent pairs  median {p['median']:+.3f} n {p['n_pairs']} share<0 {p['share_negative']:.2f} p {p['p_signflip']:.4f}")
        lines.append("")
    ax.text(0, 1, "\n".join(lines), fontsize=7.5, family="monospace", va="top", transform=ax.transAxes)
    fig.suptitle("Check of the claim: frontal 1-4 Hz power is lower while a cell is lit than while the screen is black "
                 "(one subject, 2026-10-10 runs + 10-07 reference)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    frames, metas = {}, {}
    for name, lab in SESSIONS:
        d = os.path.join(ROOT, "runs", name)
        df, meta = epochs(d)
        frames[name], metas[name] = df, meta
        print(f"{name}: {len(df)} epochs, fs {meta['fs']:.2f}, stages {[s['name'] for s in meta['stages']]}")
        print(df.groupby(["hz", "state"]).size().to_string())
    results = dict(claim="AF7/AF8 1-4 Hz log10 power, median(lit) - median(dark), 12 Hz stages and 9 Hz blue plane",
                   methods=dict(epoch_s=EPOCH_S, epochs="2 s, non-overlapping, aligned to each stage start; pure when >= 90 % one state",
                                psd="Hann periodogram (welch, nperseg = epoch) of the demeaned raw signal; band = mean PSD 1-4 Hz inclusive",
                                time_domain="variance of a 4th-order 1-4 Hz band-passed signal per epoch",
                                artifact=f"epoch out when AF7 or AF8 exceed {ART_UV} uV peak after 1-40 Hz band-pass, sit on a rail or are flat 0.25 s",
                                blink_strict=f"0.5-10 Hz band-passed AF7/AF8 above {BLINK_STRICT_UV} uV (any sample) -> epoch not blink-free",
                                block="a run of the same lit state within one stage (8 s ON/OFF block or a run of lit / black cells)",
                                ci="block bootstrap, B = %d" % B_BOOT,
                                p_epoch="permutation of lit/dark labels over epochs (as in the original analysis)",
                                p_block="permutation of lit/dark labels over blocks within each stage, epochs keep their block",
                                paired="adjacent lit / dark blocks in each stage, block means, sign-flip test",
                                drift="Theil-Sen slope per 10 min with the Sen 95 % interval"),
                   sessions={}, groups={})
    for name, lab in SESSIONS:
        df = frames[name]
        res = dict(label=lab, fs=metas[name]["fs"], first=metas[name]["first"], last=metas[name]["last"],
                   n_epochs=metas[name]["n_epochs"], stages=metas[name]["stages"])
        g12 = df[(df["hz"] == 12.0) & df["state"].isin(["lit", "dark"])]
        res["12Hz_stages"] = group_tests(g12, f"{lab} 12 Hz", seed=1)
        results["groups"][f"{name} 12Hz"] = res["12Hz_stages"]
        g9 = df[(df["hz"] == 9.0) & df["state"].isin(["lit", "dark"])]
        if len(g9) > 20:
            res["9Hz_blue_plane"] = group_tests(g9, f"{lab} 9 Hz blue", seed=2)
            results["groups"][f"{name} 9Hz"] = res["9Hz_blue_plane"]
        results["sessions"][name] = res
    tonight = pd.concat([frames["vr_planes9"], frames["vr_planes10"]], ignore_index=True)
    p12 = tonight[(tonight["hz"] == 12.0) & tonight["state"].isin(["lit", "dark"])]
    p9 = tonight[(tonight["hz"] == 9.0) & tonight["state"].isin(["lit", "dark"])]
    results["pooled_tonight_12Hz"] = group_tests(p12, "tonight 12 Hz", seed=3)
    results["pooled_tonight_9Hz"] = group_tests(p9, "tonight 9 Hz blue", seed=4)
    results["groups"]["tonight 12Hz"] = results["pooled_tonight_12Hz"]
    results["groups"]["tonight 9Hz"] = results["pooled_tonight_9Hz"]
    # run 2 - run 1 and tonight - reference differences of the gap (block bootstrap)
    b1 = gap_boot(frames["vr_planes9"][frames["vr_planes9"]["hz"] == 12.0], seed=5)
    b2 = gap_boot(frames["vr_planes10"][frames["vr_planes10"]["hz"] == 12.0], seed=6)
    b8 = gap_boot(frames["vr_planes8"][frames["vr_planes8"]["hz"] == 12.0], seed=7)
    bt = gap_boot(p12, seed=8)
    results["gap_differences_12Hz"] = dict(
        run2_minus_run1=dict(diff=float(np.median(b2) - np.median(b1)), ci=[float(x) for x in np.percentile(b2 - b1, [2.5, 97.5])]),
        tonight_minus_reference=dict(diff=float(np.median(bt) - np.median(b8)), ci=[float(x) for x in np.percentile(bt - b8, [2.5, 97.5])]))
    # verdict numbers
    c = results["pooled_tonight_12Hz"]["AF_lf_in"]
    results["verdict"] = dict(
        pooled_tonight_12Hz_gap=c["diff"], ci=c["ci"], p_epoch=c["p_epoch"], p_block=c["p_block"],
        ratio_lit_over_dark=float(10 ** c["diff"]),
        blinkfree_gap=results["pooled_tonight_12Hz"]["AF_lf_in_blinkfree_50uV"]["diff"],
        blinkfree_p_block=results["pooled_tonight_12Hz"]["AF_lf_in_blinkfree_50uV"]["p_block"],
        paired_median=results["pooled_tonight_12Hz"]["AF_lf_in_paired_adjacent_blocks"]["median"],
        paired_p=results["pooled_tonight_12Hz"]["AF_lf_in_paired_adjacent_blocks"]["p_signflip"])
    with open(os.path.join(OUT, NAME + ".json"), "w") as fh:
        json.dump(jsonable(results), fh, indent=1)
    figure(frames, results, metas, os.path.join(OUT, NAME + ".png"))
    # console summary
    for gk, r in results["groups"].items():
        print(f"\n== {gk}")
        for vk in ("AF_lf_in", "AF_lf_cut", "AF_lf_timedomain", "AF_theta_cut", "AF_lf_in_strict_any_channel_clean",
                   "AF_lf_in_blinkfree_50uV", "AF_lf_in_blink100_le20pct", "AF_lf_in_interior_t_since_ge_2s",
                   "AF_lf_in_first_epoch_of_block", "AF_lf_in_drift_residual", "AF7_lf_in", "AF8_lf_in", "AUX_lf_in",
                   "TP9_lf_in", "TP10_lf_in"):
            c = r[vk]
            print(f"  {vk:36s} diff {c['diff']:+.3f} mean {c['mean_diff']:+.3f} ci [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
                  f"n {c['n_lit']}/{c['n_dark']} blk {c['blocks_lit']}/{c['blocks_dark']} p_epoch {c['p_epoch']:.4f} p_block {c['p_block']:.4f}")
        for vk in ("AF_lf_in_paired_adjacent_blocks", "AF_lf_in_paired_adjacent_blocks_blinkfree"):
            p = r[vk]
            print(f"  {vk:36s} median {p['median']:+.3f} mean {p['mean']:+.3f} ci [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] "
                  f"n_pairs {p['n_pairs']} share<0 {p['share_negative']:.2f} p {p['p_signflip']:.4f}")
        b = r["blinks50_per_min"]
        print(f"  blinks/min lit {b['lit']:.1f} dark {b['dark']:.1f} p_block {b['contrast_epochs']['p_block']:.4f}; "
              f"epochs touched lit {r['blink50_share_of_epochs_touched']['lit']:.2f} dark {r['blink50_share_of_epochs_touched']['dark']:.2f}; "
              f"af_bad share lit {r['af_bad_share']['lit']:.2f} dark {r['af_bad_share']['dark']:.2f}")
        print("  drift (per 10 min):", {k: (round(v['slope_per_10min'], 3), [round(x, 3) for x in v['ci']]) for k, v in r["drift_theil_sen"].items()})
        for sname, c in r["per_stage"].items():
            print(f"    stage {sname:18s} diff {c['diff']:+.3f} ci [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] n {c['n_lit']}/{c['n_dark']} "
                  f"blk {c['blocks_lit']}/{c['blocks_dark']} p_epoch {c['p_epoch']:.4f} p_block {c['p_block']:.4f}")
    print("\ngap differences:", json.dumps(jsonable(results["gap_differences_12Hz"])))
    print("verdict:", json.dumps(jsonable(results["verdict"])))


if __name__ == "__main__":
    main()
