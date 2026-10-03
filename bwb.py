"""Black / white / blue from the EEG: hybrid frequency-phase codes + FBCCA.

Stimulus (xr_stimulus.html, --mode bwb): a white cell flickers white/black with
code 0, a blue cell flickers blue/black with code 1, a black cell does not
flicker. A code is a frame-exact frequency AND a phase offset (hybrid
frequency-phase modulation, as in the JFPM spellers decoded with FBCCA); the
phase restarts at each cell onset, so the response to a code has a known
phase in every window.

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
A shrinkage LDA trained on the calibration windows (black / white / blue
blocks of the same patch) turns the features into a class; it is scored by
leave-one-block-out cross-validation before the scan is decoded.

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

CLASSES = ["black", "white", "blue"]
RGB = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1]], float)
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


def raw_features(F, sl, t0, codes):
    """FBCCA score and lock-in amplitudes for each code (template-free)."""
    fb = [fbcca(F, sl, t0, c["hz"], np.deg2rad(c["phaseDeg"])) for c in codes]
    zs = [lockin(F, sl, t0, c["hz"], np.deg2rad(c["phaseDeg"])) for c in codes]
    return fb, zs


def features(fb, zs, templates):
    """[fbcca_0, fbcca_1, coh_0, coh_1]: coh_k = lock-in projected on template k."""
    coh = [0.0 if T is None or not np.any(T) else
           float(np.real(np.vdot(T, z)) / np.linalg.norm(T)) for T, z in zip(templates, zs)]
    return np.array(list(fb) + coh)


# ---------------------------------------------------------------- classifier
class ShrinkLDA:
    def fit(self, X, y, shrink=0.3):
        self.mu = np.array([X[y == k].mean(0) for k in range(3)])
        self.sc = X.std(0) + 1e-9
        Z = (X - X.mean(0)) / self.sc
        R = np.concatenate([Z[y == k] - Z[y == k].mean(0) for k in range(3)])
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


def templates_from(zs_by_class, codes):
    """Mean phase-locked response to each code in its own calibration blocks
    (white blocks for code 0, blue blocks for code 1)."""
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


def train(F, blocks, win, codes):
    W = calib_windows(F, blocks, win, codes)
    if len({w["cls"] for w in W}) < 3:
        raise ValueError("calibration has no usable windows for every class")
    zs = {}
    for w in W:
        w["fb"], w["z"] = raw_features(F, w["sl"], w["t0"], codes)
        zs.setdefault(w["cls"], []).append(w["z"])
    y = np.array([w["cls"] for w in W])
    blocks_id = np.array([w["block"] for w in W])
    # leave-one-block-out: templates AND classifier refit without the block
    pred = np.full(len(W), -1)
    for bi in np.unique(blocks_id):
        tr = blocks_id != bi
        zs_tr = {}
        for w, keep in zip(W, tr):
            if keep:
                zs_tr.setdefault(w["cls"], []).append(w["z"])
        T = templates_from(zs_tr, codes)
        X = np.array([features(w["fb"], w["z"], T) for w in W])
        if len(set(y[tr])) < 3:
            continue
        pred[~tr] = ShrinkLDA().fit(X[tr], y[tr]).predict(X[~tr])
    ok = pred >= 0
    cv_acc = float((pred[ok] == y[ok]).mean()) if ok.any() else float("nan")
    cv_conf = confusion(y[ok], pred[ok])
    T = templates_from(zs, codes)
    X = np.array([features(w["fb"], w["z"], T) for w in W])
    model = ShrinkLDA().fit(X, y)
    means = {CLASSES[k]: X[y == k].mean(0).round(4).tolist() for k in range(3)}
    return model, T, dict(cv_accuracy=cv_acc, cv_confusion=cv_conf, n_windows=int(len(W)),
                          feature_means=means, features=["fbcca_white", "fbcca_blue",
                                                         "coh_white", "coh_blue"])


def confusion(y, p):
    C = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C.tolist()


def scan_visits(session_dir, log_name="bwb_log.csv"):
    """(gx, gy, t_start, t_end) for each cell visit in the scan log."""
    _, rows = read_csv(os.path.join(session_dir, log_name))
    t = np.array([float(r[0]) for r in rows])
    gx = np.array([int(r[1]) for r in rows])
    gy = np.array([int(r[2]) for r in rows])
    cut = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0) | (np.diff(t) > 0.5))
    a, b = np.r_[0, cut + 1], np.r_[cut + 1, len(t)]
    return [(gx[i], gy[i], t[i], t[j - 1]) for i, j in zip(a, b) if gx[i] >= 0 and t[j - 1] - t[i] > 1.0]


def decode(session_dir, codes, blocks, spc, grid_w, grid_h, target=None, channels="",
           log_name="bwb_log.csv"):
    t, X, names, fs = load_eeg(session_dir, channels)
    F = Filtered(t, X, fs)
    win = max(1.0, spc - SETTLE)
    model, T, cal = train(F, blocks, win, codes)
    feats = {}
    for gx, gy, t0, t1 in scan_visits(session_dir, log_name):
        sl = F.window(t0 + LATENCY + SETTLE, min(t1, t0 + SETTLE + win) + LATENCY)
        if sl is None:
            continue
        feats.setdefault((gy, gx), []).append(features(*raw_features(F, sl, t0, codes), T))
    cls = np.zeros((grid_h, grid_w), int)
    post = np.zeros((grid_h, grid_w, 3))
    seen = np.zeros((grid_h, grid_w), bool)
    for (gy, gx), v in feats.items():
        s = model.scores(np.mean(v, axis=0))[0]
        p = np.exp(s - s.max())
        post[gy, gx] = p / p.sum()
        cls[gy, gx] = int(np.argmax(s))
        seen[gy, gx] = True
    res = dict(channels=names, codes=codes, calibration=cal, cells_decoded=int(seen.sum()),
               classes=cls.tolist())
    if target is not None:
        tg = np.asarray(target)
        m = seen
        res["accuracy"] = float((cls[m] == tg[m]).mean()) if m.any() else float("nan")
        res["confusion"] = confusion(tg[m].ravel(), cls[m].ravel())
        res["chance"] = 1 / 3
    rgb = RGB[cls]
    rgb[~seen] = 0.5                     # undecoded cells grey
    if target is not None:
        comparison_image(target, cls, seen).save(os.path.join(session_dir, "bwb_comparison.png"))
    save_png(rgb, os.path.join(session_dir, "reconstruction_bwb.png"))
    np.save(os.path.join(session_dir, "reconstruction_bwb_post.npy"), post)
    with open(os.path.join(session_dir, "bwb_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    return cls, rgb, res


def comparison_image(target, cls, seen, cell=48):
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
                col = (128, 128, 128) if ok is not None and not ok[y, x] else                     tuple(int(v * 255) for v in RGB[grid[y, x]])
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
    cal = res["calibration"]
    s = f"calibration CV accuracy {cal['cv_accuracy']:.0%} ({cal['n_windows']} windows, chance 33%)"
    if "accuracy" in res:
        s += f"; scan {res['accuracy']:.0%} of {res['cells_decoded']} cells right"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--channels", default="")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.session, "bwb_meta.json")))
    tg = np.load(os.path.join(args.session, "target_bwb.npy"))
    cls, _, res = decode(args.session, meta["codes"], meta["blocks"], meta["spc"],
                         tg.shape[1], tg.shape[0], target=tg, channels=args.channels)
    print(summary(res))
    print("confusion (rows = shown black/white/blue, cols = decoded):", res.get("confusion"))


if __name__ == "__main__":
    main()
