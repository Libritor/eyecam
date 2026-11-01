"""State of Strobe, one eye or two.

The eye-exam sittings flickered one eye at a time with the other covered by the
subject (vr_exam1, 3, 5, 6, 8, 9); vr_exam2 ran the same tests with both eyes
open. The scan sittings of the paper are binocular (a 2-D browser page reaches
both eyes). Here the paper's strobe markers are read per eye segment of the
exam sittings (the span of that eye's exam logs):

  * blink rate on task, on the ON blocks and on the OFF blocks, blink amplitude,
    width and long-blink share (flash_eog's far-channel detector);
  * the release: blinks per minute in the first second after an ON block ends
    against seconds one to four;
  * Oz theta/alpha, ON epoch minus the nearest OFF epoch within 45 s, with the
    12 Hz line, its harmonics and 6 Hz cut (state_of_flash's epochs); the
    flicker-rate sweep (temporal) is left out, its 6 to 10 Hz rates sit in
    theta and alpha;
  * the flash SSVEP per eye from the exam read-out.

    python analysis/strobe_monocular.py [vr_exam6 vr_exam8 ...]

writes runs/state_of_flash/monocular.json and prints one row per segment.
"""
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import flash_eog as fe  # noqa: E402
from analysis import state_of_flash as sof  # noqa: E402

# vr_exam1 is left out (both ear electrodes at 100 to 280 uV under the strap, the far-channel blink
# detector needs them), vr_exam3 and vr_exam5 have under two minutes of flicker
RUNS = ["vr_exam2", "vr_exam6", "vr_exam8", "vr_exam9"]
# vr_exam2 was started as the left eye but run with both eyes open (docs/EXAM.md)
EYE_OVERRIDE = {("vr_exam2", "left"): "both"}
OUT = os.path.join(ROOT, "runs", "state_of_flash", "monocular.json")


def eye_segments(name, d):
    segs = {}
    for p in sorted(glob.glob(os.path.join(d, "exam_*_log.csv"))):
        stem = os.path.basename(p)[len("exam_"):-len("_log.csv")]
        eye, test = stem.split("_", 1)
        eye = EYE_OVERRIDE.get((name, eye), eye)
        lt = pd.read_csv(p, usecols=[0]).iloc[:, 0].to_numpy(float)
        if len(lt) < 10:
            continue
        s = segs.setdefault(eye, dict(t0=np.inf, t1=-np.inf, tests=[]))
        s["t0"], s["t1"] = min(s["t0"], lt[0]), max(s["t1"], lt[-1])
        s["tests"].append(dict(test=test, t0=float(lt[0]), t1=float(lt[-1])))
    for s in segs.values():
        s["tests"].sort(key=lambda x: x["t0"])
    return dict(sorted(segs.items(), key=lambda kv: kv[1]["t0"]))


def wall_start(d, t):
    """Clock time of the recording's first sample: eeg.csv's mtime minus its span."""
    end = os.path.getmtime(os.path.join(d, "eeg.csv"))
    return datetime.fromtimestamp(end - (t[-1] - t[0]))


def acute_ta(d, S, seg, sweep_spans):
    """Oz theta/alpha ON minus the nearest OFF within 45 s, inside one segment, sweep left out."""
    t, X, names, fs = sof.load_eeg(d)
    state, stage, first, _ = sof.timeline(d, t, S["f0"])
    for a, b in sweep_spans:
        state[(t >= a) & (t <= b + 1.0)] = sof.IDLE
    df = sof.epochs(t, X, names, fs, S["f0"], state, stage, first, d)
    df = df[(df["t"] >= seg["t0"]) & (df["t"] <= seg["t1"])]
    r = sof.acute(df.reset_index(drop=True))
    out = {}
    if r:
        out = dict(ta_oz=r.get("ta_oz"), tab_oz=r.get("tab_oz"), ta_ear=r.get("ta_ear"),
                   tab_ear=r.get("tab_ear"), alpha_oz=r.get("alpha_oz"), n_pairs=r.get("n"))
    ok = df[df["ok_oz"]]
    on, off = ok[ok["state"] == sof.FLICKER], ok[ok["state"] == sof.DARK]
    out["closure_on"] = float(on["closure"].mean()) if len(on) else None
    out["closure_off"] = float(off["closure"].mean()) if len(off) else None
    return out


def flash_ssvep(d):
    try:
        ro = json.load(open(os.path.join(d, "exam_readout.json")))
    except (OSError, ValueError):
        return {}
    out = {}
    for eye, r in (ro.get("eyes") or {}).items():
        fl = r.get("flash") or {}
        if fl:
            out[eye] = {k: fl.get(k) for k in ("amp", "p", "channel", "snr", "on", "off", "log10_snr_on",
                                                "log10_snr_off", "z") if k in fl}
    return out


def analyse(name):
    d = os.path.join(ROOT, "runs", name)
    S = fe.prepare(d)
    D = fe.detect_all(S)
    t = S["t"]
    start = wall_start(d, t)
    res = dict(session=name, start=start.strftime("%Y-%m-%d %H:%M"), f0=S["f0"],
               minutes=round(S["minutes"], 1), flash=flash_ssvep(d), segments=[])
    segs = eye_segments(name, d)
    sweep = [(x["t0"], x["t1"]) for s in segs.values() for x in s["tests"] if x["test"] == "temporal"]
    prev_end = None
    for eye, seg in segs.items():
        m = (t >= seg["t0"]) & (t <= seg["t1"])
        w = fe.window_stats(S, D, m, eye)
        dk = D["to_dark"][(D["to_dark"] >= seg["t0"]) & (D["to_dark"] <= seg["t1"])]
        rel = fe.release(S, D, dk)
        row = dict(eye=eye, tests=[x["test"] for x in seg["tests"]],
                   clock=(start.timestamp() + seg["t0"] - t[0]),
                   on_task_min=round((seg["t1"] - seg["t0"]) / 60, 2),
                   valid_min=w.get("minutes_valid"),
                   blinks_per_min=w.get("blink_per_min"), blinks_ci=w.get("blink_ci"),
                   n_blinks=w.get("n_blinks"),
                   lit_per_min=w.get("lit_per_min"), dark_per_min=w.get("dark_per_min"),
                   dark_minus_lit=w.get("dark_minus_lit"), dark_minus_lit_ci=w.get("dark_minus_lit_ci"),
                   amp_uv=w.get("amp_uv_median"), amp_iqr=w.get("amp_uv_iqr"),
                   fwhm_ms=w.get("fwhm_ms_median"), long_share=w.get("long_share"),
                   release=rel,
                   gap_before_min=None if prev_end is None else round((seg["t0"] - prev_end) / 60, 2))
        row["clock"] = datetime.fromtimestamp(row["clock"]).strftime("%H:%M")
        try:
            row["acute"] = acute_ta(d, S, seg, sweep)
        except Exception as exc:              # a segment with no usable Oz epochs
            row["acute"] = dict(error=str(exc))
        res["segments"].append(row)
        prev_end = seg["t1"]
    return res


def fmt(v, nd=1):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


# ----------------------------------------------------------------- figure
SCAN_ROWS = [("vr_planes8", "10-07 night, ref."), ("vr_planes9", "10-10 17:57"), ("vr_planes10", "10-10 19:50"),
             ("vr_vision1", "10-11 12:05 grey"), ("vr_planes11", "10-12 10:40"), ("vr_die1", "10-17 10:45 die")]
EXAM_ROWS = [("vr_exam2", "both", "10-27 00:40"), ("vr_exam6", "right", "10-28 18:12 R"),
             ("vr_exam6", "left", "10-28 18:29 L"), ("vr_exam8", "right", "10-29 15:50 R"),
             ("vr_exam9", "right", "10-29 16:21 R")]


def figure(out_dir):
    """Blinks per minute on the flicker blocks against the off blocks: the binocular scan
    sittings, the exam with both eyes open, and the exam one eye at a time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7})
    scans = json.load(open(os.path.join(ROOT, "runs", "state_of_flash", "flash_eog_all.json")))["sessions"]
    mono = {(r["session"], s["eye"]): s for r in json.load(open(OUT)) for s in r["segments"]}
    groups = [("scans, both eyes", [(lab, scans[n]["lit_dark_12hz_stages"]["lit_per_min"],
                                     scans[n]["lit_dark_12hz_stages"]["dark_per_min"]) for n, lab in SCAN_ROWS]),
              ("eye test, both eyes", [(lab, mono[(n, e)]["lit_per_min"], mono[(n, e)]["dark_per_min"])
                                       for n, e, lab in EXAM_ROWS if e == "both"]),
              ("eye test, one eye", [(lab, mono[(n, e)]["lit_per_min"], mono[(n, e)]["dark_per_min"])
                                     for n, e, lab in EXAM_ROWS if e != "both"])]
    c_on, c_off, ink, mute, grid = fe.C_LIT, fe.C_DARK, "#3a3a37", "#8a8a85", "#ecebe5"
    rows, y, ticks, labels, heads = [], 0.0, [], [], []
    for g, items in groups:
        heads.append((y, g))
        y -= 0.85
        for lab, on, off in items:
            rows.append((y, on, off))
            ticks.append(y)
            labels.append(lab)
            y -= 1.0
        y -= 0.35
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for yy, on, off in rows:
        ax.plot([on, off], [yy, yy], color="#c9c8c2", lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([off], [yy], s=26, color=c_off, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter([on], [yy], s=26, color=c_on, edgecolor="white", linewidth=1.0, zorder=4)
    for yy, g in heads:
        ax.text(-0.4, yy, g, ha="left", va="center", fontsize=7, color=ink, fontweight="bold")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.3, color=ink)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlabel("blinks per minute", color=ink)
    ax.xaxis.grid(True, color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(mute)
    ax.tick_params(axis="x", colors=ink, length=2)
    h_on = ax.scatter([], [], s=26, color=c_on, edgecolor="white", label="flicker on")
    h_off = ax.scatter([], [], s=26, color=c_off, edgecolor="white", label="flicker off (black or grey)")
    ax.legend(handles=[h_on, h_off], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
              fontsize=6.3, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    ax.set_xticks([0, 5, 10, 15])
    fig.tight_layout(pad=0.3)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "one_eye_blinks.pdf")
    fig.savefig(p)
    fig.savefig(os.path.join(ROOT, "runs", "state_of_flash", "one_eye_blinks.png"), dpi=200)   # preview, not for the paper
    print("wrote", p)


def main(argv):
    if argv[:1] == ["--figure"]:
        figure(argv[1] if len(argv) > 1 else os.path.join(ROOT, "runs", "state_of_flash"))
        return
    runs = argv or RUNS
    out = []
    for name in runs:
        if not os.path.exists(os.path.join(ROOT, "runs", name, "eeg.csv")):
            print(name, "no eeg.csv")
            continue
        try:
            r = analyse(name)
        except Exception as exc:
            print(name, "failed:", exc)
            continue
        out.append(r)
        print(f"== {name} {r['start']} ({r['minutes']} min)  flash: "
              + "; ".join(f"{e} amp {fmt(v.get('amp'), 2)} p {fmt(v.get('p'), 3)}" for e, v in r["flash"].items()))
        for s in r["segments"]:
            rel = s["release"] or {}
            ac = s["acute"]
            print(f"  {s['eye']:5s} {s['clock']} {s['on_task_min']:5.1f} min (valid {fmt(s['valid_min'])})"
                  f"  blinks {fmt(s['blinks_per_min'])}/min [{fmt((s['blinks_ci'] or [None])[0])}, {fmt((s['blinks_ci'] or [None, None])[1])}]"
                  f"  on {fmt(s['lit_per_min'])} off {fmt(s['dark_per_min'])}"
                  f"  release {fmt(rel.get('first_second_per_min'), 0)}/{fmt(rel.get('seconds_1_to_4_per_min'), 0)} (n {rel.get('n_transitions')})"
                  f"  amp {fmt(s['amp_uv'], 0)} uV fwhm {fmt(s['fwhm_ms'], 0)} ms long {fmt(s['long_share'], 2)}"
                  f"  ta_oz {fmt(ac.get('ta_oz'), 3)} (n {ac.get('n_pairs')}) closure {fmt(ac.get('closure_on'), 3)}/{fmt(ac.get('closure_off'), 3)}"
                  f"  gap {s['gap_before_min']}  tests {','.join(s['tests'])}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(fe.clean(out), open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
