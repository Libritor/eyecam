"""Colour palette from the EEG: hybrid frequency-phase codes + FBCCA.

Stimulus (xr_stimulus.html, --mode bwb): class 0 is black and does not
flicker; every other class is a palette colour with its own code, and its cell
flickers (a bright tint of) that colour against black with that code. The
default palette is white and blue (the original black / white / blue scan).
A code is a frame-exact frequency AND a phase offset in whole frames (hybrid
frequency-phase modulation, as in the JFPM spellers decoded with FBCCA); the
phase restarts at each cell onset, so the response to a code has a known
phase in every window. Several colours can share one frequency and differ in
phase only: the colour comes from the code, not from the hue on screen.

Per window (one scan cell visit, or a slice of a calibration block) and per
code k, four features:
  fbcca_k  filter-bank CCA (Chen et al. 2015): canonical correlation between
           all EEG channels and sin/cos references at f_k and its harmonics in
           three sub-bands, weighted n^-1.25 + 0.25. Phase-blind, so it needs
           no training and survives a wrong latency guess.
  coh_k    phase-aware template match: the lock-in amplitude of each channel
           at f_k, 2 f_k relative to the code's own phase, projected onto the
           response measured for that code in calibration. Uses the phase
           half of the code and rejects the noise in quadrature.
A shrinkage LDA trained on the calibration windows (one block type per class,
on the same patch) turns the features into a class; it is scored by
leave-one-block-out cross-validation before the scan is decoded.

Timing: a cell's phase reference is its first frame, timed on the page's own
clock (page_t) mapped to the PC clock by a straight-line fit over every logged
frame. The PC receive time of a frame jitters with Wi-Fi; the page clock does
not, and phase codes are only as good as their time reference.

    python bwb.py --session runs/bwb1        # re-decode a recorded session
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfiltfilt

import config
from reconstruct import read_csv

# true colours (0..1) of the palette names: what the decoded image shows
PALETTE = {"black": (0, 0, 0), "white": (1, 1, 1), "red": (1, 0, 0), "green": (0, 1, 0),
           "blue": (0, 0, 1), "yellow": (1, 1, 0), "cyan": (0, 1, 1), "magenta": (1, 0, 1),
           "orange": (1, 0.5, 0), "purple": (0.5, 0, 1), "lime": (0.5, 1, 0),
           "pink": (1, 0.4, 0.7), "teal": (0, 0.6, 0.6), "brown": (0.55, 0.3, 0.1),
           "grey": (0.5, 0.5, 0.5)}
DEFAULT_COLOURS = ["white", "blue"]
CLASSES = ["black"] + DEFAULT_COLOURS
RGB = np.array([PALETTE[c] for c in CLASSES], float)


def classes_of(colours):
    """Class names and true RGB for a palette: black first, then one per code."""
    names = ["black"] + list(colours or DEFAULT_COLOURS)
    return names, np.array([PALETTE[c] for c in names], float)


def palette_codes(n, refresh=72.0, lo=12.0, hi=15.0):
    """n codes between lo and hi Hz: the frame-exact rates in that band
    (72 Hz panel: 6 frames = 12 Hz, 5 frames = 14.4 Hz; 90 Hz: 7 and 6
    frames), each colour added to the rate that keeps its phases furthest
    apart, phases spread evenly over that rate's whole-frame steps."""
    periods = [P for P in range(2, 64) if lo - 1e-6 <= refresh / P <= hi + 1e-6]
    if n > sum(periods):
        raise ValueError(f"{n} colours: only {sum(periods)} frame-exact codes between "
                         f"{lo:g} and {hi:g} Hz at {refresh:g} fps")
    count = dict.fromkeys(periods, 0)
    for _ in range(n):
        P = max(periods, key=lambda q: (q / (count[q] + 1) if count[q] < q else -1, q))
        count[P] += 1
    out = []
    for P in sorted(periods, reverse=True):       # lowest rate first
        out += [dict(hz=refresh / P, phaseDeg=360.0 * int(round(i * P / count[P])) / P)
                for i in range(count[P])]
    return out


LUMA = np.array([0.2126, 0.7152, 0.0722])     # linear-light luminance weights


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def tint_of(colour, tint):
    """Linear-light colour a cell flickers: the palette colour mixed with
    white, `tint` 0 = the pure colour, 1 = white. Brighter flicker drives a
    bigger SSVEP; the colour itself is carried by the code, not the hue."""
    c = _lin(PALETTE[colour])
    return c + tint * (1 - c)


def tint_hex(colour, tint):
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in _srgb(tint_of(colour, tint)))
FB_EDGES = [6.0, 14.0, 22.0]       # sub-band lower edges (Hz); upper edge 50 Hz
N_HARM = 3
LATENCY = 0.12                     # visual pathway delay (s)
SETTLE = 0.25                      # skip the SSVEP onset transient (s)


# ---------------------------------------------------------------- EEG
def load_eeg(session_dir, channels=""):
    """Uniformly resampled EEG: (t, X[n, ch], names, fs). Timestamps are first
    de-jittered (a linear fit per gap-free run), because phone/Wi-Fi arrival
    times come in bursts while the samples themselves are evenly spaced."""
    header, rows = read_csv(os.path.join(session_dir, "eeg.csv"))
    names = header[1:]
    t = np.array([float(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    if channels:
        idx = [names.index(c) if c in names else int(c) for c in channels.split(",")]
        X, names = X[:, idx], [names[i] for i in idx]
    gaps = np.flatnonzero(np.diff(t) > 0.5)
    starts, ends = np.r_[0, gaps + 1], np.r_[gaps + 1, len(t)]
    tt = t.copy()
    for a, b in zip(starts, ends):
        if b - a > 10:
            i = np.arange(b - a)
            tt[a:b] = np.polyval(np.polyfit(i, t[a:b], 1), i)
    fs = float(config.FS)
    grid = np.arange(tt[0], tt[-1], 1 / fs)
    U = np.column_stack([np.interp(grid, tt, X[:, c]) for c in range(X.shape[1])])
    # samples inside a stream gap are not data
    valid = np.ones(len(grid), bool)
    for a in gaps:
        valid &= ~((grid > tt[a]) & (grid < tt[a + 1]))
    U[~valid] = np.nan
    return grid, U, names, fs


class Filtered:
    """Zero-phase filter-bank and broadband copies of the whole recording."""

    def __init__(self, t, X, fs):
        self.t, self.fs = t, fs
        Xf = np.nan_to_num(X - np.nanmedian(X, axis=0))
        hi = min(50.0, 0.45 * fs)
        self.bands = [sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), Xf, axis=0)
                      for lo in FB_EDGES if lo < hi - 2]
        self.broad = sosfiltfilt(butter(4, [3.0, hi], "bandpass", fs=fs, output="sos"), Xf, axis=0)
        self.valid = np.isfinite(X).all(axis=1)

    def window(self, w0, w1):
        i0, i1 = np.searchsorted(self.t, [w0, w1])
        if i1 - i0 < self.fs * 0.8 or self.valid[i0:i1].mean() < 0.9:
            return None
        return slice(i0, i1)


# ---------------------------------------------------------------- features
def refs(t, t0, f, phi, n_harm):
    """sin/cos references of code (f, phi) whose phase restarts at t0."""
    x = t - t0
    cols = []
    for h in range(1, n_harm + 1):
        a = 2 * np.pi * h * f * x - h * phi
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def cca_rho(X, Y):
    """Largest canonical correlation between the columns of X and Y."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    return float(np.clip(np.linalg.svd(qx.T @ qy, compute_uv=False)[0], 0, 1))


def fbcca(F, sl, t0, f, phi):
    t = F.t[sl]
    nh = max(1, min(N_HARM, int(50.0 // f)))
    Y = refs(t, t0, f, phi, nh)
    return sum(((b + 1) ** -1.25 + 0.25) * cca_rho(band[sl], Y) ** 2
               for b, band in enumerate(F.bands))


def lockin(F, sl, t0, f, phi):
    """Complex amplitude per channel at f, 2f relative to the code phase."""
    x = F.broad[sl]
    t = F.t[sl] - t0
    z = []
    for h in (1, 2):
        e = np.exp(-1j * (2 * np.pi * h * f * t - h * phi))
        z.append(2 * (x * e[:, None]).mean(0))
    return np.concatenate(z)          # (2 * n_ch,)


def code_freqs(codes):
    """The distinct code frequencies, in order (FBCCA is phase-blind, so codes
    that share a frequency share one FBCCA score)."""
    out = []
    for c in codes:
        if not any(abs(c["hz"] - f) < 0.01 for f in out):
            out.append(c["hz"])
    return out


def raw_features(F, sl, t0, codes):
    """FBCCA score per code frequency and lock-in amplitudes per code
    (template-free)."""
    fb = [fbcca(F, sl, t0, f, 0.0) for f in code_freqs(codes)]
    zs = [lockin(F, sl, t0, c["hz"], np.deg2rad(c["phaseDeg"])) for c in codes]
    return fb, zs


def features(fb, zs, templates):
    """[fbcca per frequency..., coh per code...]: coh_k = lock-in projected on
    template k."""
    coh = [0.0 if T is None or not np.any(T) else
           float(np.real(np.vdot(T, z)) / np.linalg.norm(T)) for T, z in zip(templates, zs)]
    return np.array(list(fb) + coh)


# ---------------------------------------------------------------- classifier
class ShrinkLDA:
    def fit(self, X, y, shrink=0.3, n_cls=3):
        self.mu = np.array([X[y == k].mean(0) for k in range(n_cls)])
        self.sc = X.std(0) + 1e-9
        Z = (X - X.mean(0)) / self.sc
        R = np.concatenate([Z[y == k] - Z[y == k].mean(0) for k in range(n_cls)])
        S = np.cov(R.T) if len(R) > 1 else np.eye(X.shape[1])
        S = (1 - shrink) * S + shrink * np.eye(len(S)) * np.trace(S) / len(S)
        self.Si = np.linalg.pinv(S)
        self.m0 = X.mean(0)
        self.muz = (self.mu - self.m0) / self.sc
        return self

    def scores(self, X):
        Z = (np.atleast_2d(X) - self.m0) / self.sc
        return np.array([[-0.5 * (z - m) @ self.Si @ (z - m) for m in self.muz] for z in Z])

    def predict(self, X):
        return self.scores(X).argmax(1)


class ShrinkQDA:
    """Gaussian classes with their own spread, each shrunk halfway to the
    pooled (itself shrunk) covariance. Black is tight (no response) while a
    colour's response varies with attention and contact from visit to visit;
    a shared spread calls a weak but clear response 'black'."""

    def fit(self, X, y, shrink=0.3, n_cls=3, own=0.5):
        self.m0, self.sc = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.m0) / self.sc
        self.muz = np.array([Z[y == k].mean(0) for k in range(n_cls)])
        R = np.concatenate([Z[y == k] - self.muz[k] for k in range(n_cls)])
        S = np.cov(R.T) if len(R) > 1 else np.eye(X.shape[1])
        S = (1 - shrink) * S + shrink * np.eye(len(S)) * np.trace(S) / len(S)
        self.Si, self.ld = [], []
        for k in range(n_cls):
            Rk = Z[y == k] - self.muz[k]
            Sk = np.cov(Rk.T) if len(Rk) > 2 else S
            Sk = own * Sk + (1 - own) * S
            Sk = Sk + 1e-6 * np.eye(len(S))
            self.Si.append(np.linalg.pinv(Sk))
            self.ld.append(np.linalg.slogdet(Sk)[1])
        return self

    def scores(self, X, n=1):
        """Log likelihood of each class for the mean of n windows."""
        Z = (np.atleast_2d(X) - self.m0) / self.sc
        return np.array([[-0.5 * n * (z - m) @ Si @ (z - m) - 0.5 * ld
                          for m, Si, ld in zip(self.muz, self.Si, self.ld)] for z in Z])

    def predict(self, X):
        return self.scores(X).argmax(1)


def templates_from(zs_by_class, codes):
    """Mean phase-locked response to each code in its own calibration blocks
    (the blocks of class k + 1 for code k)."""
    out = []
    for k in range(len(codes)):
        z = zs_by_class.get(k + 1, [])
        out.append(np.mean([v[k] for v in z], axis=0) if z else None)
    return out


def calib_windows(F, blocks, win, codes):
    """Slice each calibration block into scan-length windows."""
    W = []
    for bi, b in enumerate(blocks):
        start = b["t0"] + LATENCY + SETTLE
        while start + win <= b["t1"] + LATENCY:
            sl = F.window(start, start + win)
            if sl is not None:
                W.append(dict(block=bi, cls=b["cls"], sl=sl, t0=b["t0"]))
            start += win
    return W


def train(F, blocks, win, codes, names=CLASSES):
    K = len(codes) + 1
    W = calib_windows(F, blocks, win, codes)
    if len({w["cls"] for w in W}) < K:
        raise ValueError("calibration has no usable windows for every class")
    zs = {}
    for w in W:
        w["fb"], w["z"] = raw_features(F, w["sl"], w["t0"], codes)
        zs.setdefault(w["cls"], []).append(w["z"])
    y = np.array([w["cls"] for w in W])
    blocks_id = np.array([w["block"] for w in W])
    # leave-one-block-out: templates AND classifier refit without the block
    pred = np.full(len(W), -1)
    held = np.full((len(W), K), np.nan)        # held-out LDA scores
    for bi in np.unique(blocks_id):
        tr = blocks_id != bi
        zs_tr = {}
        for w, keep in zip(W, tr):
            if keep:
                zs_tr.setdefault(w["cls"], []).append(w["z"])
        T = templates_from(zs_tr, codes)
        X = np.array([features(w["fb"], w["z"], T) for w in W])
        if len(set(y[tr])) < K:
            continue
        held[~tr] = ShrinkQDA().fit(X[tr], y[tr], n_cls=K).scores(X[~tr])
        pred[~tr] = held[~tr].argmax(1)
    ok = pred >= 0
    cv_acc = float((pred[ok] == y[ok]).mean()) if ok.any() else float("nan")
    cv_conf = confusion(y[ok], pred[ok], K)
    T = templates_from(zs, codes)
    X = np.array([features(w["fb"], w["z"], T) for w in W])
    model = ShrinkQDA().fit(X, y, n_cls=K)
    model.temp = temperature(held[ok], y[ok])
    means = {names[k]: X[y == k].mean(0).round(4).tolist() for k in range(K)}
    return model, T, dict(cv_accuracy=cv_acc, cv_confusion=cv_conf, n_windows=int(len(W)),
                          feature_means=means,
                          features=[f"fbcca_{f:g}Hz" for f in code_freqs(codes)]
                          + [f"coh_{c}" for c in names[1:]])


def confusion(y, p, K=3):
    C = np.zeros((K, K), int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C.tolist()


def page_clock(session_dir, log_names):
    """PC time as a straight line in page time, fitted over every logged frame:
    removes the Wi-Fi jitter in when each frame message reached the PC."""
    pts = []
    for name in log_names:
        path = os.path.join(session_dir, name)
        if os.path.exists(path):
            _, rows = read_csv(path)
            pts += [(float(r[0]), float(r[5])) for r in rows if len(r) > 5 and r[5]]
    if len(pts) < 20:
        return None
    t, pt = np.array(pts).T
    b, a = np.polyfit(pt, t, 1)
    return lambda x: a + b * np.asarray(x, float)


def scan_visits(session_dir, log_name="bwb_log.csv", clock=None):
    """(gx, gy, t_start, t_end) for each cell visit in the scan log; with a
    page clock, the times are the frames' own times on the page clock."""
    _, rows = read_csv(os.path.join(session_dir, log_name))
    t = np.array([float(r[0]) for r in rows])
    gx = np.array([int(r[1]) for r in rows])
    gy = np.array([int(r[2]) for r in rows])
    if clock is not None:
        t = clock([float(r[5]) for r in rows])
    cut = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0) | (np.diff(t) > 0.5))
    a, b = np.r_[0, cut + 1], np.r_[cut + 1, len(t)]
    return [(gx[i], gy[i], t[i], t[j - 1]) for i, j in zip(a, b) if gx[i] >= 0 and t[j - 1] - t[i] > 1.0]


def retime_blocks(session_dir, blocks, clock, log_name="bwb_cal_log.csv"):
    """Move each calibration block's onset to its first frame on the page
    clock. The block messages' own times carry Wi-Fi delay, so the frames are
    grouped into blocks by the rest gaps between them (no frames are logged
    in a rest) and each block takes the run of frames starting nearest it."""
    path = os.path.join(session_dir, log_name)
    if clock is None or not os.path.exists(path):
        return blocks
    _, rows = read_csv(path)
    tp = clock([float(r[5]) for r in rows])
    if len(tp) < 2:
        return blocks
    starts = tp[np.r_[0, np.flatnonzero(np.diff(tp) > 0.4) + 1]]
    out = []
    for b in blocks:
        j = int(np.argmin(np.abs(starts - b["t0"])))
        out.append(dict(b, t0=float(starts[j])) if abs(starts[j] - b["t0"]) < 0.5 else b)
    return out


def temperature(scores, y):
    """Divisor (>= 1) for the class scores that makes the held-out calibration
    windows' certainty honest: the smallest one whose mean certainty is no
    more than their accuracy, (right + 1) / (n + 2) so that a calibration with
    no held-out error cannot claim certainty it has not seen. A classifier
    fitted on a few blocks is overconfident, and a stopping rule on certainty
    needs the real number. (Least log loss was tried and is pulled far too
    high by the one or two confident held-out mistakes.)"""
    if len(y) < 4:
        return 1.0
    acc = ((scores.argmax(1) == y).sum() + 1) / (len(y) + 2)
    for tmp in np.geomspace(1, 256, 81):
        z = scores / tmp
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        if p.max(1).mean() <= acc:
            return float(tmp)
    return 256.0


def classify(model, feats):
    """Class and posterior of one cell from all its visits: the likelihood of
    the mean of n independent windows, on the calibrated temperature."""
    v = np.asarray(feats)
    s = model.scores(v.mean(axis=0), n=len(v))[0] / getattr(model, "temp", 1.0)
    p = np.exp(s - s.max())
    return int(np.argmax(p)), p / p.sum()


def decode_arrays(t, X, fs, codes, blocks, visits, spc, grid_w, grid_h, colours=None):
    """Train on the calibration blocks and decode every visited cell.
    visits: (gx, gy, t_start, t_end) per cell visit. Returns classes, posterior
    (grid_h, grid_w, K), seen mask, calibration report, and (model, templates,
    filtered EEG) for re-scoring further visits."""
    cls_names, _ = classes_of(colours)
    K = len(cls_names)
    if len(codes) != K - 1:
        raise ValueError(f"{len(codes)} codes for {K - 1} colours")
    F = Filtered(t, X, fs)
    win = max(1.0, spc - SETTLE)
    model, T, cal = train(F, blocks, win, codes, cls_names)
    feats = {}
    for gx, gy, t0, t1 in visits:
        f = visit_features(F, T, codes, t0, t1, win)
        if f is not None:
            feats.setdefault((gy, gx), []).append(f)
    cls = np.zeros((grid_h, grid_w), int)
    post = np.zeros((grid_h, grid_w, K))
    seen = np.zeros((grid_h, grid_w), bool)
    for (gy, gx), v in feats.items():
        cls[gy, gx], post[gy, gx] = classify(model, v)
        seen[gy, gx] = True
    return cls, post, seen, cal, (model, T, F)


def visit_features(F, T, codes, t0, t1, win):
    sl = F.window(t0 + LATENCY + SETTLE, min(t1, t0 + SETTLE + win) + LATENCY)
    return None if sl is None else features(*raw_features(F, sl, t0, codes), T)


def decode(session_dir, codes, blocks, spc, grid_w, grid_h, target=None, channels="",
           log_name="bwb_log.csv", colours=None):
    t, X, names, fs = load_eeg(session_dir, channels)
    clock = page_clock(session_dir, ["bwb_cal_log.csv", log_name])
    blocks = retime_blocks(session_dir, blocks, clock)
    cls_names, rgb_of = classes_of(colours)
    K = len(cls_names)
    cls, post, seen, cal, _ = decode_arrays(
        t, X, fs, codes, blocks, scan_visits(session_dir, log_name, clock),
        spc, grid_w, grid_h, colours)
    res = dict(channels=names, codes=codes, colours=cls_names, calibration=cal,
               cells_decoded=int(seen.sum()), classes=cls.tolist(),
               page_clock=clock is not None)
    if seen.any():
        res["certainty_min"] = float(post[seen].max(axis=1).min())
    if target is not None:
        tg = np.asarray(target)
        m = seen
        res["accuracy"] = float((cls[m] == tg[m]).mean()) if m.any() else float("nan")
        res["confusion"] = confusion(tg[m].ravel(), cls[m].ravel(), K)
        res["chance"] = 1 / K
    rgb = rgb_of[cls]
    rgb[~seen] = 0.5                     # undecoded cells grey
    if target is not None:
        comparison_image(target, cls, seen, rgb_of=rgb_of).save(
            os.path.join(session_dir, "bwb_comparison.png"))
    save_png(rgb, os.path.join(session_dir, "reconstruction_bwb.png"))
    np.save(os.path.join(session_dir, "reconstruction_bwb_post.npy"), post)
    with open(os.path.join(session_dir, "bwb_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    return cls, rgb, res


def comparison_image(target, cls, seen, cell=48, rgb_of=RGB):
    """Shown picture (left) next to the picture decoded from the EEG (right);
    wrongly decoded cells are outlined in red, undecoded cells are grey."""
    tg, cls = np.asarray(target), np.asarray(cls)
    gh, gw = tg.shape
    pad, gap, head, foot = 24, 48, 56, 40
    pw, ph = gw * cell, gh * cell
    W, H = 2 * pw + gap + 2 * pad, ph + head + foot + pad
    img = Image.new("RGB", (W, H), (34, 34, 34))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    panels = [("SHOWN", tg, None), ("DECODED FROM EEG", cls, seen)]
    wrong = 0
    for i, (label, grid, ok) in enumerate(panels):
        x0, y0 = pad + i * (pw + gap), head
        d.text((x0, 16), label, fill=(230, 230, 230), font=font)
        d.rectangle([x0 - 3, y0 - 3, x0 + pw + 2, y0 + ph + 2], outline=(110, 110, 110), width=3)
        for y in range(gh):
            for x in range(gw):
                col = (128, 128, 128) if ok is not None and not ok[y, x] else                     tuple(int(v * 255) for v in rgb_of[grid[y, x]])
                box = [x0 + x * cell, y0 + y * cell, x0 + (x + 1) * cell - 1, y0 + (y + 1) * cell - 1]
                d.rectangle(box, fill=col)
                if ok is not None and ok[y, x] and grid[y, x] != tg[y, x]:
                    d.rectangle(box, outline=(235, 40, 40), width=4)
                    wrong += 1
    n = int(np.asarray(seen).sum())
    d.text((pad, head + ph + 12),
           f"{n - wrong} of {n} cells right  ·  red outline = decoded wrong", fill=(200, 200, 200), font=small)
    return img


def save_png(rgb, path, scale=config.UPSCALE):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)


def summary(res):
    """The result that counts: cells of the shown picture decoded right."""
    k = len(res.get("colours", CLASSES))
    if "accuracy" not in res:
        return f"{res['cells_decoded']} cells decoded (no target to score against)"
    n = res["cells_decoded"]
    right = int(round(res["accuracy"] * n))
    return (f"{right} of {n} cells the right colour ({res['accuracy']:.0%}, chance {1 / k:.0%})"
            + ("; whole picture right" if right == n else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--channels", default="")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.session, "bwb_meta.json")))
    tg = np.load(os.path.join(args.session, "target_bwb.npy"))
    cls, _, res = decode(args.session, meta["codes"], meta["blocks"], meta["spc"],
                         tg.shape[1], tg.shape[0], target=tg, channels=args.channels,
                         colours=meta.get("colours"))
    print(summary(res))
    print(f"confusion (rows = shown {'/'.join(res['colours'])}, cols = decoded):",
          res.get("confusion"))


if __name__ == "__main__":
    main()
