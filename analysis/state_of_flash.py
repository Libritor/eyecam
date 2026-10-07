"""State of Flash: does the flicker make the subject drowsy?  (analysis, not scoring)

Every recorded session is an unplanned experiment on this: minutes of 12 Hz
flicker interleaved, position by position and block by block, with black.
This script reads all of them and asks three questions of the EEG, with the
flicker line and its harmonics cut out of every band first:

  acute   inside a session, is the EEG drowsier while a cell flickers than
          while a black cell is shown seconds before or after?  Each flicker
          epoch is paired with the nearest dark epoch within 45 s, so time on
          task is the same for both (reported for the calibration blocks, where
          ON and OFF strictly alternate, and for the scans)
  drift   does the marker climb with minutes of stimulation?  Theil-Sen slope
          per session, over flicker epochs and over dark epochs separately
  net     the minute before the first flicker against the minute after the
          last (the result on screen, no flicker)

Drowsiness markers per 2 s epoch, on Oz (AUX) and on the ears (TP9 / TP10),
after a 1-40 Hz band-pass:
  theta/alpha          log10 power 4-7.5 Hz over 8-10.5 Hz   (rises with drowsiness)
  (theta+alpha)/beta   over 15-23 + 25-30 Hz                  (rises with drowsiness)
  alpha                log10 power 8-10.5 Hz                  (rises when drowsy with
                                                              the eyes open; jumps when they close)
  blinks               blink rate per minute from AF7 / AF8   (blinkmask.blink_mask)
  closure              share of time with Oz alpha above 3x the eyes-open
                       baseline for >= 1 s                   (blinkmask.closure_mask)

An epoch is left out of a channel group's spectral markers when that group
exceeds 150 uV peak after the band-pass (movement, a loose electrode) or the
frontal channels show more than 20 % blink.  Colour-tagged stages (7.2 / 9 Hz
tags sit inside theta and alpha) are kept out of the spectral contrasts and
used for blinks only.  Everything is one subject, so the pooled tests are over
sessions, not people.

    python analysis/state_of_flash.py               # every session -> runs/state_of_flash/
    python analysis/state_of_flash.py vr_planes7    # one session
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

OUT = os.path.join(ROOT, "runs", "state_of_flash")
EPOCH_S, HOP_S = 2.0, 1.0
ARTIFACT_UV, BLINK_MAX = 150.0, 0.20
PAIR_WINDOW_S = 45.0
PRE, FLICKER, DARK, TAGGED, AUDIO, POST, IDLE = 0, 1, 2, 3, 4, 5, 6
STATE_NAME = {PRE: "before", FLICKER: "flicker", DARK: "dark", TAGGED: "colour tags",
              AUDIO: "audio", POST: "after", IDLE: "between stages"}
MARKERS = ["ta_oz", "tab_oz", "alpha_oz", "ta_ear", "tab_ear", "alpha_ear"]
MARKER_LABEL = {"ta_oz": "theta/alpha, Oz", "tab_oz": "(theta+alpha)/beta, Oz",
                "alpha_oz": "alpha power, Oz", "ta_ear": "theta/alpha, ears",
                "tab_ear": "(theta+alpha)/beta, ears", "alpha_ear": "alpha power, ears"}
# validated categorical pair (dataviz palette slots 1 and 2) + neutral ink
C_FLICKER, C_DARK, C_NONE, C_INK = "#2a78d6", "#eb6834", "#b5b4ad", "#3a3a37"


def ok_col(m):
    return "ok_oz" if m.endswith("_oz") else "ok_ear"


# ----------------------------------------------------------------- loading
def load_eeg(d):
    df = pd.read_csv(os.path.join(d, "eeg.csv"))
    t = df.iloc[:, 0].to_numpy(float)
    names = [c.strip() for c in df.columns[1:]]
    X = df.iloc[:, 1:].to_numpy(float)
    dt = np.diff(t)
    good = dt < 0.5
    fs = float(good.sum() / dt[good].sum()) if good.any() else 256.0
    return t, X, names, fs


def stim_freq(d):
    p = os.path.join(d, "calibration.json")
    if os.path.exists(p):
        try:
            c = json.load(open(p))
            f = (c.get("line") or {}).get("delivered") or c.get("stim_freq")
            if f:
                return float(f), True
        except (ValueError, TypeError):
            pass
    return 12.0, False


def timeline(d, t):
    """Per EEG sample: what was on the screen (state) and which stage it was."""
    state = np.full(len(t), PRE, np.int8)
    stage = np.full(len(t), 0, np.int8)        # 0 none, 1 calibration, 2 scan, 3 other
    first, last = np.inf, -np.inf
    for p in sorted(glob.glob(os.path.join(d, "*_log.csv"))):
        kind = os.path.basename(p).replace("_log.csv", "")
        try:
            L = pd.read_csv(p, header=0)
        except Exception:
            continue
        if len(L) < 10 or L.shape[1] < 5:
            continue
        lt = L.iloc[:, 0].to_numpy(float)
        gx = L.iloc[:, 1].to_numpy(float)
        lum = L.iloc[:, 3].to_numpy(float)
        audio = L.iloc[:, 11].to_numpy(float) if L.shape[1] > 11 else np.zeros(len(L))
        if kind in ("color", "mux", "smooth", "bwb"):
            s = np.where(gx >= 0, TAGGED, DARK)
            sg = 3
        elif kind in ("ccal", "bwb_cal"):
            s = np.where(lum > 0.05, TAGGED, DARK)
            sg = 3
        elif kind == "assr":
            s = np.where(audio > 0, AUDIO, DARK)
            sg = 3
        else:                                   # calib, cursor, sweep, plane_*, calib_<colour>
            s = np.where(lum > 0.05, FLICKER, DARK)
            sg = 1 if kind.startswith("calib") or kind == "sweep" else 2
        order = np.argsort(lt)
        lt, s = lt[order], s[order]
        idx = np.searchsorted(lt, t, side="right") - 1
        ok = (idx >= 0) & (t - lt[np.clip(idx, 0, len(lt) - 1)] < 0.5)
        state[ok] = s[idx[ok]]
        stage[ok] = sg
        first, last = min(first, lt[0]), max(last, lt[-1])
    if np.isfinite(first):
        state[(t > first) & (t < last) & (state == PRE)] = IDLE
        state[t >= last + 1.0] = POST
    return state, stage, first, last


# ----------------------------------------------------------------- epochs
def band_masks(f, f0):
    cut = np.zeros(len(f), bool)
    for k in (1, 2, 3):
        cut |= np.abs(f - k * f0) <= 0.75
    cut |= np.abs(f - f0 / 2) <= 0.6          # photic driving also shows at f0/2 (6 Hz, inside theta)
    theta = (f >= 4) & (f <= 7.5) & ~cut
    alpha = (f >= 8) & (f <= 10.5) & ~cut
    beta = (((f >= 15) & (f <= 23)) | ((f >= 25) & (f <= 30))) & ~cut
    return theta, alpha, beta


def epochs(t, X, names, fs, f0, state, stage, first, d):
    sos = butter(2, [1.0, min(40.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    n, hop = int(round(EPOCH_S * fs)), int(round(HOP_S * fs))
    starts = np.arange(0, len(t) - n, hop)
    span_ok = (t[starts + n - 1] - t[starts]) < EPOCH_S * 1.25
    starts = starts[span_ok]
    blink = blinkmask.blink_mask(X, names, fs, 100.0)
    # eyes-open alpha baseline from the calibration OFF blocks when there are any
    spans = blinkmask.calibration_off_spans(d)
    base = None
    if spans:
        try:
            base = blinkmask.alpha_baseline(t, Xf, names, fs, f0, spans)
        except Exception:
            base = None
    try:
        closure, _, _ = blinkmask.closure_mask(Xf, names, fs, f0, 3.0, 1.0, baseline=base)
    except Exception:
        closure = np.zeros(len(t), bool)
    oz = blinkmask.oz_index(names)
    ears = [i for i, nm in enumerate(names) if nm in ("TP9", "TP10")]
    seg = np.stack([Xf[s:s + n] for s in starts])           # (E, n, ch)
    seg = seg - seg.mean(axis=1, keepdims=True)
    p2p = np.abs(seg).max(axis=1)                            # (E, ch)
    # over the limit = movement or a loose electrode; under 1 uV = a flat (railed) channel
    art_oz = ((p2p[:, oz] > ARTIFACT_UV) | (p2p[:, oz] < 1.0)) if oz is not None else np.ones(len(starts), bool)
    art_ear = ((p2p[:, ears].max(axis=1) > ARTIFACT_UV) | (p2p[:, ears].min(axis=1) < 1.0)) if ears else np.ones(len(starts), bool)
    bl = np.array([blink[s:s + n].mean() for s in starts])
    cl = np.array([closure[s:s + n].mean() for s in starts])
    st = np.array([np.bincount(state[s:s + n].astype(int), minlength=8).argmax() for s in starts])
    pure = np.array([(state[s:s + n] == st[i]).mean() >= 0.8 for i, s in enumerate(starts)])
    sg = np.array([np.bincount(stage[s:s + n].astype(int), minlength=4).argmax() for s in starts])
    f, P = welch(seg, fs=fs, nperseg=n, axis=1)              # (E, F, ch)
    th, al, be = band_masks(f, f0)

    def bp(mask, chans):
        if not chans:
            return np.full(len(starts), np.nan)
        return P[:, mask][:, :, chans].mean(axis=(1, 2))
    ozc = [oz] if oz is not None else []
    out = dict(t=t[starts] + EPOCH_S / 2, state=st, pure=pure, stage=sg,
               art_oz=art_oz, art_ear=art_ear, blink=bl, closure=cl)
    for tag, ch in (("oz", ozc), ("ear", ears)):
        pt, pa, pb = bp(th, ch), bp(al, ch), bp(be, ch)
        out[f"ta_{tag}"] = np.log10(pt / pa)
        out[f"tab_{tag}"] = np.log10((pt + pa) / pb)
        out[f"alpha_{tag}"] = np.log10(pa)
    df = pd.DataFrame(out)
    df["minutes"] = (df["t"] - (first if np.isfinite(first) else df["t"].iloc[0])) / 60.0
    df["ok_oz"] = pure & ~art_oz & (bl <= BLINK_MAX)
    df["ok_ear"] = pure & ~art_ear & (bl <= BLINK_MAX)
    # blink events: rising edges of the blink mask, per epoch (for a rate)
    edges = np.flatnonzero(np.diff(blink.astype(int)) == 1)
    df["blink_events"] = [((edges >= s) & (edges < s + n)).sum() for s in starts]
    return df


# ----------------------------------------------------------------- per-session tests
def acute(df, stage_sel=None):
    """Mean (flicker - nearest dark within 45 s) per marker; n pairs (Oz group)."""
    res = {}
    for m in MARKERS:
        sel = df[ok_col(m)]
        if stage_sel is not None:
            sel = sel & (df["stage"] == stage_sel)
        fl = df[sel & (df["state"] == FLICKER)]
        dk = df[sel & (df["state"] == DARK)]
        if len(fl) < 10 or len(dk) < 10:
            res[m] = float("nan")
            continue
        ft, dt = fl["t"].to_numpy(), dk["t"].to_numpy()
        j = np.clip(np.searchsorted(dt, ft), 0, len(dt) - 1)
        jm = np.clip(j - 1, 0, len(dt) - 1)
        near = np.where(np.abs(dt[j] - ft) <= np.abs(dt[jm] - ft), j, jm)
        keep = np.abs(dt[near] - ft) <= PAIR_WINDOW_S
        if keep.sum() < 10:
            res[m] = float("nan")
            continue
        dlt = fl[m].to_numpy()[keep] - dk[m].to_numpy()[near[keep]]
        dlt = dlt[np.isfinite(dlt)]
        res[m] = float(dlt.mean()) if len(dlt) else float("nan")
        if m == "ta_oz":
            res["n"] = int(keep.sum())
    return res if any(np.isfinite(v) for k, v in res.items() if k != "n") else None


def drift(df, state_sel):
    res = {}
    for m in MARKERS:
        sel = df[ok_col(m)] & (df["state"] == state_sel) & (df["minutes"] >= 0)
        sub = df[sel]
        if len(sub) < 60 or sub["minutes"].max() - sub["minutes"].min() < 4:
            res[m] = float("nan")
            continue
        y, x = sub[m].to_numpy(), sub["minutes"].to_numpy()
        okm = np.isfinite(y)
        res[m] = float(stats.theilslopes(y[okm], x[okm])[0] * 10.0) if okm.sum() >= 30 else float("nan")
        if m == "ta_oz":
            res["n"] = int(okm.sum())
            res["span_min"] = float(x.max() - x.min())
    return res if any(np.isfinite(v) for k, v in res.items() if k not in ("n", "span_min")) else None


def net(df, first, last):
    res = {}
    for m in MARKERS:
        okm = df[ok_col(m)]
        pre = df[(df["t"] >= first - 70) & (df["t"] <= first - 10) & okm]
        post = df[(df["t"] >= last + 10) & (df["t"] <= last + 70) & okm]
        if len(pre) < 10 or len(post) < 10:
            res[m] = float("nan")
            continue
        res[m] = float(np.nanmedian(post[m]) - np.nanmedian(pre[m]))
        if m == "ta_oz":
            res["n_pre"], res["n_post"] = int(len(pre)), int(len(post))
    return res if any(np.isfinite(v) for k, v in res.items() if not k.startswith("n")) else None


def blink_rates(df):
    out = {}
    for s in (FLICKER, DARK, TAGGED, PRE, POST):
        sub = df[(df["state"] == s) & df["pure"]]
        if len(sub) >= 30:
            out[STATE_NAME[s]] = dict(per_min=float(sub["blink_events"].sum() / (len(sub) * HOP_S) * 60.0),
                                      closure=float(sub["closure"].mean()), epochs=int(len(sub)))
    return out


# ----------------------------------------------------------------- pooling
def pool(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)])
    if len(v) < 3:
        return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else float("nan"))
    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
    try:
        p = float(stats.wilcoxon(v).pvalue)
    except ValueError:
        p = float("nan")
    return dict(n=int(len(v)), mean=float(v.mean()), ci=[float(ci[0]), float(ci[1])],
                positive=int((v > 0).sum()), p_wilcoxon=p)


# ----------------------------------------------------------------- main
def analyse(d):
    name = os.path.basename(os.path.normpath(d))
    t, X, names, fs = load_eeg(d)
    f0, known = stim_freq(d)
    state, stage, first, last = timeline(d, t)
    if not np.isfinite(first):
        return None
    df = epochs(t, X, names, fs, f0, state, stage, first, d)
    res = dict(session=name, minutes=round((t[-1] - t[0]) / 60, 1), fs=round(fs, 1), f0=f0,
               f0_known=known, has_oz=blinkmask.oz_index(names) is not None,
               channels=names, epochs=int(len(df)),
               ok_oz=int(df["ok_oz"].sum()), ok_ear=int(df["ok_ear"].sum()),
               states={STATE_NAME[k]: int((df["state"] == k).sum()) for k in STATE_NAME},
               acute_all=acute(df), acute_calib=acute(df, 1), acute_scan=acute(df, 2),
               drift_flicker=drift(df, FLICKER), drift_dark=drift(df, DARK),
               net=net(df, first, last), blinks=blink_rates(df))
    return res, df


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    if argv:
        dirs = [os.path.join(ROOT, "runs", a) for a in argv]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "runs", "*"))
                      if os.path.isdir(d) and os.path.exists(os.path.join(d, "eeg.csv"))
                      and os.path.getsize(os.path.join(d, "eeg.csv")) >= 2_000_000
                      and not os.path.basename(d).startswith(("phantom", "gate", "colour_gate", "freeze")))
    results, courses = [], {}
    for d in dirs:
        try:
            r = analyse(d)
        except Exception as exc:
            print(f"{os.path.basename(d)}: {exc!r}")
            continue
        if r is None:
            print(f"{os.path.basename(d)}: no stimulus log")
            continue
        res, df = r
        results.append(res)
        courses[res["session"]] = df
        a = res["acute_all"] or {}
        bl = res["blinks"]
        print(f"{res['session']:14s} {res['minutes']:5.1f} min  f0 {res['f0']:.0f}{'' if res['f0_known'] else '?'} "
              f"Oz {'y' if res['has_oz'] else 'n'}  ok Oz {res['ok_oz']:4d} ears {res['ok_ear']:4d}  "
              f"flicker {res['states']['flicker']:4d} dark {res['states']['dark']:4d}  "
              f"acute theta/alpha Oz {a.get('ta_oz', float('nan')):+.3f} (n {a.get('n', 0)})  "
              f"blinks/min flicker {bl.get('flicker', {}).get('per_min', float('nan')):4.1f} dark {bl.get('dark', {}).get('per_min', float('nan')):4.1f}")
    # ---- pooled
    with_oz = [r for r in results if r["has_oz"] and r["f0_known"]]
    summary = {}
    for key in ("acute_all", "acute_calib", "acute_scan", "drift_flicker", "drift_dark", "net"):
        summary[key] = {m: pool([(r[key] or {}).get(m) for r in with_oz]) for m in MARKERS}
    blink = {}
    for st in ("flicker", "dark", "before", "after"):
        blink[st] = pool([r["blinks"].get(st, {}).get("per_min") for r in results]) if results else {}
    blink["flicker_minus_dark"] = pool([r["blinks"]["flicker"]["per_min"] - r["blinks"]["dark"]["per_min"]
                                        for r in results if "flicker" in r["blinks"] and "dark" in r["blinks"]])
    closure = pool([r["blinks"]["flicker"]["closure"] - r["blinks"]["dark"]["closure"]
                    for r in with_oz if "flicker" in r["blinks"] and "dark" in r["blinks"]])
    summary["blinks_per_min"] = blink
    summary["closure_flicker_minus_dark"] = closure
    json.dump(dict(sessions=results, pooled=summary, markers=MARKER_LABEL),
              open(os.path.join(OUT, "state_of_flash.json"), "w"), indent=1, default=float)
    print("\n==== pooled over sessions with an Oz channel and a known flicker frequency "
          f"(n = {len(with_oz)}); units: log10 power ratio; drift per 10 min")
    for key, label in (("acute_calib", "acute, calibration ON - OFF"), ("acute_scan", "acute, lit cell - black cell"),
                       ("drift_flicker", "drift over flicker epochs"), ("drift_dark", "drift over dark epochs"),
                       ("net", "after - before")):
        print(f"-- {label}")
        for m in MARKERS:
            s = summary[key][m]
            if s.get("n", 0) >= 3:
                print(f"   {MARKER_LABEL[m]:26s} mean {s['mean']:+.3f}  95% CI [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  "
                      f"positive {s['positive']}/{s['n']}  p {s['p_wilcoxon']:.3f}")
    b = summary["blinks_per_min"]
    print("-- blinks per minute: " + ", ".join(f"{k} {v['mean']:.1f} (n {v['n']})" for k, v in b.items() if v.get('n')))
    print(f"-- eye-closure share, flicker - dark: {closure.get('mean', float('nan')):+.3f} (n {closure.get('n', 0)}, p {closure.get('p_wilcoxon', float('nan')):.3f})")
    figures(results, courses, summary)
    return 0


# ----------------------------------------------------------------- figures
def figures(results, courses, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#d4d3cc", "axes.labelcolor": C_INK,
                         "xtick.color": C_INK, "ytick.color": C_INK, "text.color": C_INK,
                         "axes.titlesize": 8.5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    # 1. time courses: theta/alpha on Oz, the longest sessions with Oz
    def ok_flicker(r):
        df = courses[r["session"]]
        return int((df["ok_oz"] & (df["state"] == FLICKER) & np.isfinite(df["ta_oz"])).sum())
    longest = sorted([r for r in results if r["has_oz"] and r["f0_known"] and ok_flicker(r) >= 100],
                     key=lambda r: -r["minutes"])[:8]
    if longest:
        cols = 2
        rows_n = (len(longest) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(10, 2.0 * rows_n), sharey=True, squeeze=False)
        for ax, r in zip(axes.ravel(), longest):
            df = courses[r["session"]]
            for st, c, lab in ((FLICKER, C_FLICKER, "flicker on screen"), (DARK, C_DARK, "black on screen")):
                sub = df[df["ok_oz"] & (df["state"] == st)]
                if len(sub):
                    ax.plot(sub["minutes"], sub["ta_oz"], ".", ms=2.5, color=c, alpha=0.55, label=lab, rasterized=True)
                    ser = pd.Series(sub["ta_oz"].to_numpy(), index=pd.to_timedelta(sub["minutes"].to_numpy(), unit="m"))
                    roll = ser.rolling("30s", center=True, min_periods=8).median()
                    ax.plot(roll.index.total_seconds() / 60.0, roll.values, "-", lw=1.6, color=c)
            sub = df[df["ok_oz"] & df["state"].isin([PRE, POST, IDLE])]
            if len(sub):
                ax.plot(sub["minutes"], sub["ta_oz"], ".", ms=2.5, color=C_NONE, alpha=0.6, label="no stimulus", rasterized=True)
            ax.set_title(f"{r['session']}  ({r['minutes']:.0f} min, {r['f0']:.0f} Hz)", loc="left")
            ax.grid(True, color="#ecebe5", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("minutes since the first flicker")
        for ax in axes[:, 0]:
            ax.set_ylabel("log10 theta / alpha (Oz)")
        for ax in axes.ravel()[len(longest):]:
            ax.axis("off")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=3, frameon=False, markerscale=4)
        fig.suptitle("State of Flash: theta/alpha on Oz through the longest sessions (flicker line and harmonics cut out; 30 s running median)",
                     x=0.01, ha="left", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(OUT, "fig_time_courses.png"), dpi=140)
        plt.close(fig)
    # 2. effects per session: dot per session, pooled mean with CI
    keys = (("acute_calib", "calibration: ON - OFF block"), ("acute_scan", "scan: lit cell - black cell"),
            ("drift_flicker", "drift over flicker, per 10 min"), ("drift_dark", "drift over black, per 10 min"))
    marks = ("ta_oz", "tab_oz", "alpha_oz")
    with_oz = [r for r in results if r["has_oz"] and r["f0_known"]]
    fig, axes = plt.subplots(len(marks), len(keys), figsize=(11, 2.2 * len(marks)), squeeze=False)
    for i, m in enumerate(marks):
        for j, (key, label) in enumerate(keys):
            ax = axes[i, j]
            vals = [((r[key] or {}).get(m)) for r in with_oz]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            y = np.random.default_rng(0).uniform(-0.18, 0.18, len(vals))
            ax.axvline(0, color="#b5b4ad", lw=1)
            ax.plot(vals, y, "o", ms=5, mfc="none", mec=C_INK, mew=1)
            s = summary[key][m]
            if s.get("n", 0) >= 3:
                ax.errorbar(s["mean"], 0.5, xerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
                            fmt="o", ms=6, color=C_FLICKER, capsize=3, lw=1.5)
                ax.text(s["mean"], 0.72, f"mean {s['mean']:+.3f}, p {s['p_wilcoxon']:.2f}, {s['positive']}/{s['n']} > 0",
                        ha="center", va="bottom", fontsize=7.5, color=C_FLICKER)
            ax.set_ylim(-0.5, 1.1)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(label, loc="left")
            if j == 0:
                ax.set_ylabel(MARKER_LABEL[m], fontsize=8)
            ax.grid(True, axis="x", color="#ecebe5", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("State of Flash: each circle is one session (log10 power ratio; positive = drowsier with flicker); "
                 "blue = mean over sessions with 95 % CI, Wilcoxon p", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_effects.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
