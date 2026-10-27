"""Objective eye exam from the EEG: the plan of each test and the scoring of
its blocks (xr_session.py --mode exam).

The page (xr_stimulus.html, runExam) shows, per eye, four tests as ON/OFF
blocks at the session's flicker frequency with the eyes on the centre dot:

  acuity    a vertical black/white grating appearing and disappearing on a grey
            of the same mean luminance (pattern onset/offset), its bar width
            shrinking step by step (arcmin, set by --exam-sizes and --exam-ppd)
  contrast  the same grating at one bar width, its contrast falling step by step
  field     one sector of a grid over the patch flickering white/black at a time
  colour    the whole patch flickering red, green or blue against black

The driver logs every frame (exam_<eye>_<test>_log.csv, the usual columns plus
size, contrast, plane) and every block edge with the page's description of the
step, then calls score_stage() on each test: for every step, how much of the
flicker frequency (and its double) the EEG carries during that step's ON blocks
against the test's OFF blocks, by the exact-line periodogram and an exact
permutation test over the ON/OFF labels, plus the noise-corrected amplitude on
the test's best channel and the phase of the response against the logged
flicker edges. The thresholds are fitted afterwards by
analysis/exam_readout.py, which also draws the report.

Everything here is a response amplitude and its statistics. Nothing here is a
diagnosis.

The phase is relative, not an absolute latency: the EEG clock is rebuilt from
the sample index on the earliest-arriving samples, so it carries the stream's
transport delay (common to every step), and one flicker period at 15 Hz is
shorter than the visual pathway's delay. Compare it between sectors and
between the eyes; do not read it as milliseconds from the retina.

    python exam.py runs/<session>     re-scores a session from the blocks
                                      exam.json stores and redoes the read-out
"""
import math
import os
from itertools import combinations

import numpy as np
from scipy.signal import periodogram

import reconstruct

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
TESTS = ["acuity", "contrast", "field", "colour"]
EYES = ["right", "left", "both"]
LOG_EXTRA = ("size", "contrast", "plane")   # columns appended to the frame log
DISPLAY_GAMMA = 2.2                         # the page's grey and contrast scale
DISCARD_S = 0.5                             # dropped from every block start
MAX_RMS_UV = 60.0                           # a channel above this for a whole test is out of contact


def csv_header(path):
    try:
        with open(path) as f:
            line = f.readline().strip()
        return line.split(",")[1:] if line.startswith("time,") else None
    except OSError:
        return None


def plan(a):
    """The steps of each test from the driver's --exam-* arguments."""
    sizes = [float(v) for v in str(a.exam_sizes).split(",") if v.strip()]
    contrasts = [float(v) for v in str(a.exam_contrasts).split(",") if v.strip()]
    cols, rows = [int(v) for v in str(a.exam_field).lower().split("x")]
    if not sizes or not contrasts or cols < 1 or rows < 1:
        raise ValueError("--exam-sizes, --exam-contrasts and --exam-field must not be empty")
    return {
        "acuity": [dict(size=s, contrast=1.0) for s in sizes],
        "contrast": [dict(size=float(a.exam_contrast_size), contrast=c) for c in contrasts],
        "field": [dict(sector=i, cols=cols, rows=rows) for i in range(cols * rows)],
        "colour": [dict(plane=k) for k in range(3)],
    }


def _envelope_clock(t, fs):
    """Sample times rebuilt from the sample index: the EEG rows carry their UDP
    arrival stamp (bursts of several samples share one), so within a block the
    clock is index / fs anchored on the earliest-arriving samples."""
    k = np.arange(len(t))
    resid = t - k / fs
    return float(np.percentile(resid, 5)) + k / fs


def _phase_origin(edges, t0, f):
    """The block's flicker phase origin: the circular mean of the logged rising
    edges modulo the period, so a stray edge (a lone first frame, a dropped
    frame) cannot set the origin on its own."""
    if len(edges) == 0:
        return None
    per = 1.0 / f
    frac = ((edges - t0) / per) % 1.0
    z = np.mean(np.exp(2j * math.pi * frac))
    if abs(z) < 0.5:          # edges not periodic at f: no usable origin
        return None
    return float(t0 + (math.atan2(z.imag, z.real) / (2 * math.pi) % 1.0) * per)


def _line(seg, fs, f0, half=0.15, flank=(0.5, 2.0)):
    """(power at f0, flank noise power) from a hann periodogram, power per bin."""
    seg = seg - seg.mean()
    f, p = periodogram(seg, fs=fs, window="hann", detrend="constant", scaling="spectrum")
    h = max(half, (f[1] - f[0]) * 1.01)
    pk = p[np.abs(f - f0) <= h].mean()
    fl = (np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])
    return float(pk), (float(p[fl].mean()) if fl.any() else float("nan"))


def _perm_p(V, ons):
    """Max-over-channels mean ON-OFF difference and its exact permutation p."""
    n, n_on = len(ons), int(ons.sum())
    diff = V[ons].mean(0) - V[~ons].mean(0)
    best_c = int(np.argmax(diff))
    obs = float(diff[best_c])
    null = []
    if math.comb(n, n_on) <= 200000:
        for cmb in combinations(range(n), n_on):
            m = np.zeros(n, bool)
            m[list(cmb)] = True
            null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
    else:
        rng = np.random.default_rng(0)
        for _ in range(20000):
            m = np.zeros(n, bool)
            m[rng.choice(n, n_on, replace=False)] = True
            null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
    null = np.array(null)
    return obs, best_c, float(np.mean(null >= obs - 1e-12)), len(null)


def score_stage(session, blocks, hzs, metas, log_name, test, eye=""):
    """Per step of one test: ON blocks of that step against all OFF blocks of
    the test. blocks [(kind, t0, t1)], hzs the page's frequency per block,
    metas the page's step description per block (dict or None)."""
    path = os.path.join(session, "eeg.csv")
    names = csv_header(path) or CH_NAMES
    E = np.genfromtxt(path, delimiter=",", skip_header=1)
    if E.ndim != 2 or E.shape[0] < 10:
        return dict(test=test, eye=eye, ok=False, reason="no EEG", steps=[])
    et = E[:, 0]
    fs = reconstruct.infer_fs(et)
    L = np.genfromtxt(os.path.join(session, log_name), delimiter=",", skip_header=1)
    if L.ndim != 2:
        return dict(test=test, eye=eye, ok=False, reason="no frames logged", steps=[])
    lt, lfl = L[:, 0], L[:, 4]
    rise = lt[1:][(lfl[1:] == 1) & (lfl[:-1] == 0)]
    per = np.diff(rise)
    per = per[(per > 0.01) & (per < 0.5)]
    f_req = float(np.median([float(h) for h in hzs if h])) if any(hzs) else 15.0
    f_del = float(1 / np.median(per)) if len(per) >= 10 else f_req

    segs = []
    for (kind, t0, t1), hz, meta in zip(blocks, hzs, metas):
        sel = (et >= t0 + DISCARD_S) & (et <= t1)
        if sel.sum() < fs * 1.5:
            continue
        idx = np.where(sel)[0]
        e0 = rise[(rise >= t0) & (rise <= t1)]
        segs.append(dict(kind=kind, meta=dict(meta or {}), X=E[idx, 1:],
                         t=_envelope_clock(et[idx], fs),
                         edge=_phase_origin(e0, t0, f_del)))
    if not segs or not any(s["kind"] == "on" for s in segs) \
            or sum(s["kind"] == "off" for s in segs) < 2:
        return dict(test=test, eye=eye, ok=False, reason="too few scorable blocks",
                    steps=[], n_blocks=len(blocks))
    n_min = min(s["X"].shape[0] for s in segs)
    for s in segs:
        s["X"], s["t"] = s["X"][:n_min], s["t"][:n_min]
    nch = len(names)
    stds = np.array([[s["X"][:, c].std() for c in range(nch)] for s in segs])
    keep = stds <= 4 * np.median(stds, 0)      # a block drowned in artefact is left out
    # a channel out of contact for the whole test (sustained rms far above EEG:
    # vr_exam1's ear electrodes sat at 100-280 uV) must not be scored, or its
    # noise can be picked as the test's channel by chance
    dead = np.median(stds, 0) > MAX_RMS_UV
    keep[:, dead] = False
    V = np.zeros((len(segs), nch))             # the stat: log SNR at f plus at 2f
    A = np.full((len(segs), nch), np.nan)      # noise-corrected line amplitude
    Z = np.zeros((len(segs), nch), complex)    # phasor against the first flicker edge
    for j, s in enumerate(segs):
        for c in range(nch):
            x = s["X"][:, c]
            p1, n1 = _line(x, fs, f_del)
            p2, n2 = _line(x, fs, 2 * f_del)
            v = math.log(max(p1 / max(n1, 1e-12), 1e-9))
            if np.isfinite(n2):
                v += math.log(max(p2 / max(n2, 1e-12), 1e-9))
            if keep[j, c] and np.isfinite(v):
                V[j, c] = v
                A[j, c] = math.sqrt(2 * (max(p1 - n1, 0.0)
                                         + (max(p2 - n2, 0.0) if np.isfinite(n2) else 0.0)))
            if s["edge"] is not None and keep[j, c]:
                xd = x - x.mean()
                Z[j, c] = np.sum(xd * np.exp(-2j * math.pi * f_del * (s["t"] - s["edge"])))
    on = np.array([s["kind"] == "on" for s in segs])
    step_of = np.array([int(s["meta"].get("step", -1)) if s["kind"] == "on" else -1
                        for s in segs])
    # the test's channel: the largest ON-OFF difference of the stat over every step
    diff_all = V[on].mean(0) - V[~on].mean(0)
    diff_all[dead] = -np.inf
    if not np.isfinite(diff_all).any():
        return dict(test=test, eye=eye, ok=False, reason="every channel out of contact",
                    steps=[], dead=[names[c] for c in range(nch) if dead[c]])
    ch = int(np.argmax(diff_all))
    out_steps = []
    for st in sorted(set(step_of[on])):
        m_on = on & (step_of == st)
        rows = m_on | ~on
        Vs, ons = V[rows], m_on[rows]
        if ons.sum() < 1 or (~ons).sum() < 2:
            continue
        obs, best_c, p, n_null = _perm_p(Vs, ons)
        amps = A[m_on, ch]
        z = Z[m_on, ch]
        z = z[np.abs(z) > 0]
        phase = sd = delay = None
        if len(z):
            u = (z / np.abs(z)).mean()
            phase = math.degrees(math.atan2(u.imag, u.real))
            R = abs(u)
            sd = (math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))
                  if len(z) > 1 else None)
            delay = ((-phase / 360.0) % 1.0) * 1000.0 / f_del
        meta = next(s["meta"] for s in segs if s["kind"] == "on"
                    and int(s["meta"].get("step", -1)) == st)
        out_steps.append(dict(
            step=int(st), meta=meta, n_on=int(ons.sum()), n_off=int((~ons).sum()),
            stat=obs, p=p, n_null=n_null, best=names[best_c],
            amp=float(np.nanmedian(amps)) if np.isfinite(amps).any() else None,
            amp_sd=float(np.nanstd(amps, ddof=1)) if np.isfinite(amps).sum() > 1 else None,
            snr=float(math.exp(V[m_on, ch].mean())),
            amps={names[c]: (float(np.nanmedian(A[m_on, c]))
                             if np.isfinite(A[m_on, c]).any() else None) for c in range(nch)},
            phase_deg=phase, phase_sd_deg=sd, delay_ms=delay))
    off_amps = {names[c]: (float(np.nanmedian(A[~on, c])) if np.isfinite(A[~on, c]).any()
                           else None) for c in range(nch)}
    return dict(test=test, eye=eye, ok=True, names=names, channel=names[ch],
                freq=f_del, freq_requested=f_req, fs=float(fs),
                segment_s=float(n_min / fs), n_on=int(on.sum()), n_off=int((~on).sum()),
                noise_amp=off_amps[names[ch]], off_amps=off_amps, steps=out_steps,
                rms_uv={names[c]: float(np.median(stds[:, c])) for c in range(nch)},
                dead=[names[c] for c in range(nch) if dead[c]],
                blocks=[[k, float(t0), float(t1), hz, meta]
                        for (k, t0, t1), hz, meta in zip(blocks, hzs, metas)])


def rescore(session):
    """Score every test of <session>/exam.json again from the blocks it stores
    (after a change to the scoring), rewrite it and redo the read-out."""
    import json
    path = os.path.join(session, "exam.json")
    with open(path) as f:
        out = json.load(f)
    for eye, tests in out["results"].items():
        for test, res in tests.items():
            if not res.get("blocks"):
                print(f"{eye} {test}: no blocks stored, left as it was")
                continue
            bl = [(b[0], b[1], b[2]) for b in res["blocks"]]
            hz = [b[3] for b in res["blocks"]]
            meta = [b[4] for b in res["blocks"]]
            new = score_stage(session, bl, hz, meta, f"exam_{eye}_{test}_log.csv", test, eye=eye)
            new["delivery"] = res.get("delivery")
            tests[test] = new
            print(describe_stage(new))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
    import exam_readout
    print(exam_readout.run(session)["text"])


def step_label(meta, test=None):
    test = test or meta.get("test")
    if test == "acuity":
        return f"{meta.get('size', 0):.3g}'"
    if test == "contrast":
        return f"c {meta.get('contrast', 0):.3g}"
    if test == "field":
        return f"s{int(meta.get('sector', -1))}"
    if test == "colour":
        return ["red", "green", "blue"][int(meta.get("plane", 0))]
    return str(meta)


def describe_stage(res):
    """One line per test for the console."""
    if not res.get("ok"):
        return f"{res.get('eye')} {res.get('test')}: not scored ({res.get('reason')})"
    cells = []
    for s in res["steps"]:
        amp = f"{s['amp']:.2f}" if s["amp"] is not None else "-"
        cells.append(f"{step_label(s['meta'], res['test'])} {amp} p{s['p']:.3f}")
    return (f"{res['eye']} {res['test']}: {res['channel']} at {res['freq']:.2f} Hz, "
            f"noise {res['noise_amp'] if res['noise_amp'] is None else round(res['noise_amp'], 2)}; "
            + " | ".join(cells))


def merge(out, parts):
    """Put eyes from several sessions into one exam.json for a joint read-out:
    parts = [(session, eye, label), ...] takes <eye> of <session>'s exam.json
    and calls it <label> (an eye tested in two sittings, a restarted eye, a
    binocular run labelled 'both'). The merged folder has no EEG or logs, so
    it can be read out but not re-scored."""
    import json
    os.makedirs(out, exist_ok=True)
    merged = None
    for session, eye, label in parts:
        with open(os.path.join(session, "exam.json")) as f:
            e = json.load(f)
        if merged is None:
            merged = {k: v for k, v in e.items() if k not in ("eyes", "results", "delivery")}
            merged.update(eyes=[], results={}, delivery={}, sources={})
        if eye not in e["results"]:
            raise SystemExit(f"{session} has no eye '{eye}' (has {list(e['results'])})")
        merged["eyes"].append(label)
        merged["results"][label] = {t: dict(r, eye=label) for t, r in e["results"][eye].items()}
        merged["sources"][label] = dict(session=session, eye=eye)
        for k, v in e.get("delivery", {}).items():
            if k.startswith(eye + "/"):
                merged["delivery"][label + "/" + k.split("/", 1)[1]] = v
    with open(os.path.join(out, "exam.json"), "w") as f:
        json.dump(merged, f, indent=1)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
    import exam_readout
    print(exam_readout.run(out)["text"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        rescore(sys.argv[1])
    elif len(sys.argv) >= 4 and sys.argv[1] == "merge":
        parts = []
        for spec in sys.argv[3:]:
            session, eye = spec.split(":", 1)
            eye, _, label = eye.partition("=")
            parts.append((session, eye, label or eye))
        merge(sys.argv[2], parts)
    else:
        raise SystemExit("usage: python exam.py runs/<session>              (re-score from exam.json)\n"
                         "       python exam.py merge runs/<out> runs/<a>:<eye>[=<label>] ...")
