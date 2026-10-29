"""Objective eye exam from the EEG: the plan of each test and the scoring of
its blocks (xr_session.py --mode exam).

The page (xr_stimulus.html, runExam) shows, per eye, six tests as ON/OFF
blocks at the session's flicker frequency (the sweep at its own rates) with
the eyes on the centre dot, each over the whole window (what the eye-as-a-camera runs taught: the flash
response grows with the flickering area around the fixation, and a small
patch gives a line under the noise):

  flash     the whole window flickering white/black: the full-field flicker the
            camera's calibration uses, the strongest response the rig gives;
            it is the eye's reference (the channel and the amplitude every
            other test is read against) and the gate that the electrodes and
            the display deliver at all
  temporal  the same full-field flash at several rates, one step per rate
            (--exam-rates; the page delivers refresh / (2k) frames, so 6, 7.5,
            10, 15 and 30 Hz on a 60 Hz panel, 6, 7.2, 9, 12, 18 and 36 Hz at
            72 Hz): the response amplitude against rate, the eye's temporal
            tuning (its best rate and the rate where the response has fallen
            to half), and from the slope of the response phase against rate
            over the rates that respond the apparent latency (Regan, 1966),
            whose difference between the eyes is the pattern VEP's classic
            measure
  acuity    a vertical black/white grating whose bars swap places at the
            flicker rate (pattern reversal): every bar location swings
            black-white like the flash, the mean luminance never changes, and
            nothing swings once the bars are too fine to resolve; its bar
            width shrinks step by step (arcmin, set by --exam-sizes and
            --exam-ppd). --exam-style onset gives the older pattern
            onset/offset grating on a same-mean grey instead.
  vernier   coarse bars cut into horizontal strips; every other strip jumps
            sideways by an offset and back at the flicker rate, aligned <->
            misaligned (the vernier VEP of Norcia, Wesemann and Manny, 1999):
            the offset shrinks step by step (--exam-vernier, arcmin), a
            hyperacuity that normal eyes resolve below the bar width and that
            amblyopia degrades more than acuity; offsets are delivered to a
            quarter of a device pixel by the canvas's own anti-aliasing
  contrast  the same grating at one bar width, its contrast falling step by step
  csf       the contrast test again at other bar widths (--exam-csf-sizes,
            coarse and fine; --exam-csf-contrasts), which with the contrast
            test's own bar width gives three points of the contrast
            sensitivity function (log sensitivity against cycles per degree)
  field     one sector of a grid over the window flickering white/black at a
            time (2x2 quadrants by default, each a quarter of the window)
  gaze      the flash again while the fixation dot sits at the centre and
            then out toward each edge (looking up, down, left, right)
  meridian  the reversing grating at one bar width turned to several
            orientations (--exam-meridians, degrees from vertical,
            clockwise): a response that is weak along one meridian and strong
            along the one at right angles to it is the signature of
            uncorrected astigmatism (and, at fine bars, of the oblique effect)
  photometry  heterochromatic flicker photometry: the window swaps between a
            red field and a green field at the flicker rate while the share of
            the luminance given to red runs from 0.2 to 0.8 of the pair's
            total (--exam-mixes; the OFF screen is the steady mixture of the
            two, so neither the mean luminance nor the chromaticity changes).
            Where the two fields are equally bright TO THIS EYE the luminance
            response has its null; the red share at the null is the eye's
            isoluminant point (0.5 for the display's photometric standard), the
            response left at the null is the chromatic pathway's, and a shift
            of the null toward red is what a red-weak (protan) eye gives.
            --exam-photometry by adds the blue/yellow pair (tritan axis)
  colour    the whole window flickering red, green or blue against black

The driver logs every frame (exam_<eye>_<test>_log.csv, the usual columns plus
size, contrast, plane, angle, mix; the vernier test logs its offset in the size
column; in a reversal block the flicker column marks the frames
on which the bars swapped, so the logged edges are the reversals) and every
block edge with the page's description of the step, then calls score_stage()
on each test: for every step, how much of the flicker frequency (and its
double) the EEG carries during that step's ON blocks against the test's OFF
blocks, by the exact-line periodogram and an exact permutation test over the
ON/OFF labels, plus the noise-corrected amplitude on the eye's reference
channel (the flash test's best channel when the flash responded, else the
test's own best) and the phase of the response against the logged flicker
edges. The thresholds are fitted afterwards by analysis/exam_readout.py,
which also draws the report and gives every amplitude relative to the eye's
flash.

Everything here is a response amplitude and its statistics. Nothing here is a
diagnosis.

The phase is relative, not an absolute latency: the EEG clock is rebuilt from
the sample index on the earliest-arriving samples, so it carries the stream's
transport delay (common to every step), and one flicker period at 15 Hz is
shorter than the visual pathway's delay. Compare it between sectors and
between the eyes; do not read it as milliseconds from the retina.

The flicker sweep turns the phase into a latency the honest way: a response
delayed by tau has a phase that falls by 360 tau degrees per hertz of rate,
so the slope of the phase against rate over the rates that respond gives the
apparent latency (Regan, 1966) with the waveform's own constant offset taken
out (analysis/exam_readout.py, fit_latency). The transport delay stays in it,
the same for both eyes, so the difference between the eyes is the
measurement.

    python exam.py runs/<session>     re-scores a session from the blocks
                                      exam.json stores and redoes the read-out
"""
import math
import os
from itertools import combinations

import numpy as np
from scipy.signal import get_window, periodogram

import reconstruct

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
TESTS = ["flash", "temporal", "acuity", "vernier", "contrast", "csf", "field", "gaze",
         "meridian", "photometry", "colour"]
GRATING_TESTS = {"acuity", "contrast", "csf", "meridian"}   # reversing bars; the page draws them alike
PHOTOMETRY_PAIRS = {"rg": 0.2, "by": 0.08}  # pair -> total luminance of the two fields (of white,
#                                             linear): red at full is 0.2126 of white and blue 0.0722,
#                                             so a share of 0.8 for either stays within the primary
# gaze: the whole window flickers while the fixation dot sits at the centre or
# out toward an edge, so the stimulus falls on the retina to one side of the
# fovea (looking left, the window lies in the right visual field, from the
# fovea out to the window's far edge); offsets are fractions of the window's
# width and height from the centre (the driver's --exam-gaze sets the size)
GAZE_NAMES = ["centre", "up", "down", "left", "right"]
MULTI_RATE = {"temporal"}                   # tests whose steps differ in flicker rate
EYES = ["right", "left", "both"]
STYLES = ["reversal", "onset"]              # the grating tests' temporal pattern
LOG_EXTRA = ("size", "contrast", "plane", "angle", "mix")   # columns appended to the frame log
#                                            (angle: the bars' tilt; mix: photometry's red share)
DISPLAY_GAMMA = 2.2                         # the page's grey and contrast scale
DISCARD_S = 0.5                             # dropped from every block start
MAX_RMS_UV = 60.0                           # a channel above this for a whole test is noisy: kept only if it responds
OPEN_RMS_UV = 300.0                         # above this the input is open or railing: out of contact outright
FLASH_P = 0.05                              # the flash counts as a response below this


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
    rates = [float(v) for v in str(getattr(a, "exam_rates", "6,7.5,10,12,15,20,30")).split(",")
             if v.strip()]
    if not sizes or not contrasts or cols < 1 or rows < 1 or not rates or min(rates) <= 0:
        raise ValueError("--exam-sizes, --exam-contrasts, --exam-rates and --exam-field must "
                         "not be empty")
    if getattr(a, "exam_style", "reversal") not in STYLES:
        raise ValueError(f"--exam-style: one of {STYLES}")
    offsets = _floats(getattr(a, "exam_vernier", "8,4,2,1,0.5"))
    bar = float(getattr(a, "exam_vernier_bar", 20.0))
    csf_sizes = _floats(getattr(a, "exam_csf_sizes", "60,10"))
    csf_contrasts = _floats(getattr(a, "exam_csf_contrasts", "0.5,0.25,0.12,0.06"))
    meridians = _floats(getattr(a, "exam_meridians", "0,45,90,135"))
    m_size = float(getattr(a, "exam_meridian_size", 14.0))
    mixes = _floats(getattr(a, "exam_mixes", "0.2,0.35,0.5,0.65,0.8"))
    pairs = [p.strip() for p in str(getattr(a, "exam_photometry", "rg")).split(",") if p.strip()]
    if not offsets or min(offsets) <= 0 or bar <= 0 or not csf_sizes or not csf_contrasts \
            or not meridians or m_size <= 0 or not mixes or not all(0 < x < 1 for x in mixes) \
            or not pairs or any(p not in PHOTOMETRY_PAIRS for p in pairs):
        raise ValueError("--exam-vernier, --exam-vernier-bar, --exam-csf-sizes, --exam-csf-contrasts, "
                         "--exam-meridians, --exam-meridian-size must be positive, --exam-mixes "
                         f"between 0 and 1, --exam-photometry among {sorted(PHOTOMETRY_PAIRS)}")
    return {
        "flash": [dict(flash=1)],
        "temporal": [dict(rate=r) for r in rates],
        "acuity": [dict(size=s, contrast=1.0) for s in sizes],
        "vernier": [dict(offset=d, size=bar, contrast=1.0) for d in offsets],
        "contrast": [dict(size=float(a.exam_contrast_size), contrast=c) for c in contrasts],
        "csf": [dict(size=s, contrast=c) for s in csf_sizes for c in csf_contrasts],
        "field": [dict(sector=i, cols=cols, rows=rows) for i in range(cols * rows)],
        "gaze": [dict(gaze=i, name=n, gx=x, gy=y) for i, (n, x, y) in enumerate(gaze_positions(a))],
        "meridian": [dict(size=m_size, contrast=1.0, angle=th % 180.0) for th in meridians],
        "photometry": [dict(mix=x, pair=p, lum=PHOTOMETRY_PAIRS[p]) for p in pairs for x in mixes],
        "colour": [dict(plane=k) for k in range(3)],
    }


def _floats(text):
    return [float(v) for v in str(text).split(",") if v.strip()]


def gaze_positions(a):
    """(name, x, y) of the fixation dot for the gaze test: the centre and four
    offsets of --exam-gaze (fraction of the window's width / height; 0.42 puts
    the dot at 84 % of the way from the centre to the edge), y down."""
    g = float(getattr(a, "exam_gaze", 0.42))
    if not 0 < g < 0.5:
        raise ValueError("--exam-gaze: a fraction of the window between 0 and 0.5")
    return [("centre", 0.0, 0.0), ("up", 0.0, -g), ("down", 0.0, g), ("left", -g, 0.0), ("right", g, 0.0)]


def reps(a, test):
    """ON/OFF pairs per step: the flash has one step and gets --exam-flash
    pairs (an exact permutation test over 5 + 5 blocks reaches p = 0.004;
    4 + 4 floors at 0.014), every other test --exam-reps per step."""
    return int(getattr(a, "exam_flash", 5)) if test == "flash" else int(a.exam_reps)


def reference(flash_res):
    """The eye's reference from its scored flash test: (channel, amplitude)
    when the flash responded (p < FLASH_P), else (None, None)."""
    if not flash_res or not flash_res.get("ok") or not flash_res.get("steps"):
        return None, None
    s = flash_res["steps"][0]
    if s.get("p") is None or s["p"] >= FLASH_P or s.get("amp") is None:
        return None, None
    return flash_res.get("channel"), float(s["amp"])


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


def _line(seg, fs, f0, flank=(0.5, 2.0)):
    """(power at f0, flank noise power per bin). The line power is the
    hann-windowed projection on f0 itself, the quantity the periodogram gives
    at a grid frequency taken off the grid, so it does not depend on where f0
    falls between bins (the three-bin mean it replaces read a line on a bin
    0.83 as strong as one half a bin off, which the flicker sweep, whose
    rates fall on and off the grid, cannot carry); the noise is the mean of
    the hann periodogram's bins 0.5 to 2 Hz away from f0."""
    seg = seg - seg.mean()
    n = len(seg)
    w = get_window("hann", n)
    f, p = periodogram(seg, fs=fs, window=w, detrend="constant", scaling="spectrum")
    x = np.sum(w * seg * np.exp(-2j * math.pi * f0 * np.arange(n) / fs))
    pk = 2.0 * abs(x) ** 2 / w.sum() ** 2
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


def score_stage(session, blocks, hzs, metas, log_name, test, eye="", ref_ch=None, fs=None):
    """Per step of one test: ON blocks of that step against all OFF blocks of
    the test. blocks [(kind, t0, t1)], hzs the page's frequency per block,
    metas the page's step description per block (dict or None). ref_ch: the
    eye's reference channel (the flash test's) for the amplitudes and phases;
    without one, or with it out of contact, the test's own best channel. A
    test in MULTI_RATE (the flicker sweep) is scored step by step at each
    step's own rate, taken from its blocks' logged edges, the OFF blocks
    measured at that rate too. fs: the EEG sample rate to use (a re-score
    passes the run's stored value); None infers it from the file."""
    path = os.path.join(session, "eeg.csv")
    names = csv_header(path) or CH_NAMES
    E = np.genfromtxt(path, delimiter=",", skip_header=1)
    if E.ndim != 2 or E.shape[0] < 10:
        return dict(test=test, eye=eye, ok=False, reason="no EEG", steps=[])
    et = E[:, 0]
    fs = float(fs) if fs else reconstruct.infer_fs(et)
    L = np.genfromtxt(os.path.join(session, log_name), delimiter=",", skip_header=1)
    if L.ndim != 2:
        return dict(test=test, eye=eye, ok=False, reason="no frames logged", steps=[])
    lt, lfl = L[:, 0], L[:, 4]
    rise = lt[1:][(lfl[1:] == 1) & (lfl[:-1] == 0)]
    per = np.diff(rise)
    per = per[(per > 0.01) & (per < 0.5)]
    f_req = float(np.median([float(h) for h in hzs if h])) if any(hzs) else 15.0
    f_del = float(1 / np.median(per)) if len(per) >= 10 else f_req

    multi = test in MULTI_RATE
    segs = []
    for (kind, t0, t1), hz, meta in zip(blocks, hzs, metas):
        sel = (et >= t0 + DISCARD_S) & (et <= t1)
        if sel.sum() < fs * 1.5:
            continue
        idx = np.where(sel)[0]
        e0 = rise[(rise >= t0) & (rise <= t1)]
        f_b = f_del
        if multi:
            # a sweep block has its own rate: from its own edges, else the page's
            pb = np.diff(e0)
            pb = pb[(pb > 0.01) & (pb < 0.5)]
            f_b = float(1 / np.median(pb)) if len(pb) >= 5 else (float(hz) if hz else f_del)
        segs.append(dict(kind=kind, meta=dict(meta or {}), X=E[idx, 1:],
                         t=_envelope_clock(et[idx], fs), f=f_b,
                         edge=_phase_origin(e0, t0, f_b)))
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
    # two tiers. Above OPEN_RMS_UV the input is open or railing (mains rail to
    # rail) and is set aside outright. Between MAX_RMS_UV and OPEN_RMS_UV the
    # electrode is noisy but may still carry the response: in vr_exam6 the left
    # eye's Oz cup sat at 105 uV rms and gave a 12 Hz line SNR of 6-18 in every
    # ON block against under 1.5 in every OFF block (exact p 0.004), and the old
    # single threshold threw it away and read that eye on a forehead channel.
    # A noisy channel is therefore kept when its own ON/OFF statistic is
    # significant, decided below once the statistic exists.
    med_rms = np.median(stds, 0)
    noisy = med_rms > MAX_RMS_UV
    dead = med_rms > OPEN_RMS_UV
    keep[:, dead] = False
    on = np.array([s["kind"] == "on" for s in segs])
    step_of = np.array([int(s["meta"].get("step", -1)) if s["kind"] == "on" else -1
                        for s in segs])
    # the rate each step is scored at: the session's for every test but the
    # sweep, whose steps each have their own (the median over the step's ON blocks)
    f_step = {}
    for st in sorted(set(step_of[on])):
        f_step[st] = (float(np.median([sg["f"] for sg, m in zip(segs, on & (step_of == st)) if m]))
                      if multi else f_del)
    freqs = sorted(set(f_step.values()))

    def lines(x, f):
        """(the stat: log SNR at f plus at 2f; the noise-corrected line amplitude)"""
        p1, n1 = _line(x, fs, f)
        p2, n2 = _line(x, fs, 2 * f)
        v = math.log(max(p1 / max(n1, 1e-12), 1e-9))
        if np.isfinite(n2):
            v += math.log(max(p2 / max(n2, 1e-12), 1e-9))
        amp = math.sqrt(2 * (max(p1 - n1, 0.0) + (max(p2 - n2, 0.0) if np.isfinite(n2) else 0.0)))
        return v, amp
    VF, AF = {}, {}                            # per rate: the stat and the amplitude, block x channel
    for f in freqs:
        V = np.zeros((len(segs), nch))
        A = np.full((len(segs), nch), np.nan)
        for j, sg in enumerate(segs):
            for c in range(nch):
                if keep[j, c]:
                    v, amp = lines(sg["X"][:, c], f)
                    if np.isfinite(v):
                        V[j, c], A[j, c] = v, amp
        VF[f], AF[f] = V, A
    f_seg = [f_step.get(int(st), f_del) for st in step_of]       # each ON block's own rate
    Z = np.zeros((len(segs), nch), complex)    # phasor against the block's flicker edges, at its rate
    for j, sg in enumerate(segs):
        if sg["edge"] is None or not on[j]:
            continue
        for c in range(nch):
            if keep[j, c]:
                xd = sg["X"][:, c] - sg["X"][:, c].mean()
                Z[j, c] = np.sum(xd * np.exp(-2j * math.pi * f_seg[j] * (sg["t"] - sg["edge"])))
    # every block at its own rate (an OFF block: the mean over the test's
    # rates), for the channel rule; one rate for every test but the sweep
    V = np.array([VF[f_seg[j]][j] if on[j] else np.mean([VF[f][j] for f in freqs], 0)
                  for j in range(len(segs))])
    A_own = np.array([AF[f_seg[j]][j] if on[j] else np.mean([AF[f][j] for f in freqs], 0)
                      for j in range(len(segs))])
    # the test's own channel: the largest median ON amplitude among the
    # channels whose own ON/OFF difference of the stat is significant (exact
    # permutation per channel); the stat is a log SNR, which saturates on a
    # strong response and scatters around zero in OFF blocks, so its ON-OFF
    # difference alone picked an ear over a 1.5x stronger Oz on the phantom.
    # Without any significant channel: the largest ON-OFF difference of the stat.
    diff_all = V[on].mean(0) - V[~on].mean(0)
    ch_p = np.ones(nch)
    for c in range(nch):
        if not dead[c]:
            ch_p[c] = _perm_p(V[:, [c]], on)[2]
    # the noisy tier is out of contact unless its own response is significant
    dead = dead | (noisy & (ch_p >= 0.05))
    diff_all[dead] = -np.inf
    if not np.isfinite(diff_all).any():
        return dict(test=test, eye=eye, ok=False, reason="every channel out of contact",
                    steps=[], dead=[names[c] for c in range(nch) if dead[c]])
    amp_on = np.array([np.nanmedian(A_own[on, c]) if np.isfinite(A_own[on, c]).any() else -np.inf
                       for c in range(nch)])
    amp_on[dead] = -np.inf
    sig = (ch_p < 0.05) & ~dead
    best_ch = int(np.argmax(np.where(sig, amp_on, -np.inf))) if sig.any() else int(np.argmax(diff_all))
    ch = best_ch
    if ref_ch in names and not dead[names.index(ref_ch)]:
        ch = names.index(ref_ch)
    out_steps = []
    for st in sorted(set(step_of[on])):
        f_st = f_step[st]
        Vf, Af = VF[f_st], AF[f_st]
        m_on = on & (step_of == st)
        rows = m_on | ~on
        Vs, ons = Vf[rows], m_on[rows]
        if ons.sum() < 1 or (~ons).sum() < 2:
            continue
        # the step's family-wise test runs over the channels in contact only: a
        # noisy electrode that failed its own test must not widen the family
        # (it used to enter as a zero column, which could never win)
        live = np.where(~dead)[0]
        obs, best_i, p, n_null = _perm_p(Vs[:, live], ons)
        best_c = int(live[best_i])
        amps = Af[m_on, ch]
        z = Z[m_on, ch]
        z = z[np.abs(z) > 0]
        phase = sd = delay = None
        if len(z):
            u = (z / np.abs(z)).mean()
            phase = math.degrees(math.atan2(u.imag, u.real))
            R = abs(u)
            sd = (math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))
                  if len(z) > 1 else None)
            delay = ((-phase / 360.0) % 1.0) * 1000.0 / f_st
        meta = next(sg["meta"] for sg in segs if sg["kind"] == "on"
                    and int(sg["meta"].get("step", -1)) == st)
        off_c = Af[~on, ch]
        out_steps.append(dict(
            step=int(st), meta=meta, freq=f_st, n_on=int(ons.sum()), n_off=int((~ons).sum()),
            stat=obs, p=p, n_null=n_null, best=names[best_c],
            amp=float(np.nanmedian(amps)) if np.isfinite(amps).any() else None,
            amp_sd=float(np.nanstd(amps, ddof=1)) if np.isfinite(amps).sum() > 1 else None,
            snr=float(math.exp(Vf[m_on, ch].mean())),
            off_amp=float(np.nanmedian(off_c)) if np.isfinite(off_c).any() else None,
            amps={names[c]: (float(np.nanmedian(Af[m_on, c]))
                             if np.isfinite(Af[m_on, c]).any() else None) for c in range(nch)},
            phase_deg=phase, phase_sd_deg=sd, delay_ms=delay))
    off_amps = {names[c]: (float(np.nanmedian(A_own[~on, c])) if np.isfinite(A_own[~on, c]).any()
                           else None) for c in range(nch)}
    return dict(test=test, eye=eye, ok=True, names=names, channel=names[ch],
                channel_best=names[best_ch],
                channel_p={names[c]: float(ch_p[c]) for c in range(nch)},
                channel_amp_on={names[c]: (float(amp_on[c]) if np.isfinite(amp_on[c]) else None)
                                for c in range(nch)},
                reference=ref_ch if (ref_ch in names and ch == names.index(ref_ch)) else None,
                style=next((m.get("style") for m in metas if m and m.get("style")), None),
                freq=None if multi else f_del, freq_requested=None if multi else f_req,
                rates=freqs if multi else None, fs=float(fs),
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
        ref_ch = None
        for test in [t for t in TESTS if t in tests]:    # the flash first: it sets the reference
            res = tests[test]
            if not res.get("blocks"):
                print(f"{eye} {test}: no blocks stored, left as it was")
                continue
            bl = [(b[0], b[1], b[2]) for b in res["blocks"]]
            hz = [b[3] for b in res["blocks"]]
            meta = [b[4] for b in res["blocks"]]
            new = score_stage(session, bl, hz, meta, f"exam_{eye}_{test}_log.csv", test, eye=eye,
                              ref_ch=ref_ch, fs=res.get("fs"))
            new["delivery"] = res.get("delivery")
            tests[test] = new
            if test == "flash":
                ref_ch, _ = reference(new)
            print(describe_stage(new))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
    import exam_readout
    print(exam_readout.run(session)["text"])


def step_label(meta, test=None):
    test = test or meta.get("test")
    if test == "flash":
        return "flash"
    if test == "temporal":
        return f"{meta.get('hz') or meta.get('rate', 0):.3g} Hz"
    if test == "acuity":
        return f"{meta.get('size', 0):.3g}'"
    if test == "vernier":
        return f"offset {meta.get('offset', 0):.3g}'"
    if test == "contrast":
        return f"c {meta.get('contrast', 0):.3g}"
    if test == "csf":
        return f"{meta.get('size', 0):.3g}' c {meta.get('contrast', 0):.3g}"
    if test == "meridian":
        return f"{meta.get('angle', 0):g} deg"
    if test == "photometry":
        return f"{meta.get('pair', 'rg')} red {meta.get('mix', 0):.2f}" if meta.get("pair", "rg") == "rg" \
            else f"{meta.get('pair')} blue {meta.get('mix', 0):.2f}"
    if test == "field":
        return f"s{int(meta.get('sector', -1))}"
    if test == "gaze":
        return f"looking {meta.get('name') or meta.get('gaze')}"
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
    ch = res["channel"] + (f" (reference; own best {res['channel_best']})"
                           if res.get("channel_best") and res["channel_best"] != res["channel"] else "")
    where = f"at {res['freq']:.2f} Hz" if res.get("freq") else "rate by rate"
    return (f"{res['eye']} {res['test']}: {ch} {where}, "
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
