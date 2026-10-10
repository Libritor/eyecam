"""Colour palette by frequency-phase codes, offline, on the synthetic subject.

Every colour is a code at 12 Hz or 14.4 Hz (the frame-exact rates between 12
and 15 Hz on a 72 Hz Quest) with a phase in whole frames: 60 degree steps at
12 Hz (6 frames a cycle), 72 degree steps at 14.4 Hz (5 frames). Black does
not flicker. The EEG is decoded by bwb.py itself (FBCCA + phase templates +
shrinkage LDA, trained on simulated calibration blocks), so this tests the
decoder that runs live.

Simulated subject: simulate.synth_eeg's background (as adaptive_passes.py) on
TP9 AF7 AF8 TP10 AUX, an SSVEP at the code's frequency and phase (+ 0.4 second
harmonic, 0.3 s entrainment) with a per-channel latency, the channel SNRs of
blink_gate.py, --scale 0.6 (the 2026-10-01 'NO' run), and a per-visit
log-normal gain. Two things the live run adds are modelled explicitly:

  timing wobble  every visit's response is shifted by N(0, sigma) seconds:
                 frame timing, EEG timestamps and the brain's own latency
                 drift. 10 ms is 43 degrees at 12 Hz.
  brightness     a cell flickers a tint of its colour (linear mix with white,
                 --tint 0 = the pure colour, 1 = white); response amplitude
                 ~ sqrt(luminance). This is an assumption, not a measurement.

Scored on the shown picture only: cells right and whether the whole picture
is right, after 1-3 fixed passes and with the quality rule (one pass, then 4 more visits at a time to the least certain
cells until every cell's posterior is >= --quality, cap 6 passes).

usage: python analysis/bwb_palette_sim.py --colours 4 --jitter-ms 5 [--trials 30]
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bwb                       # noqa: E402
from adaptive_passes import background, SNR   # noqa: E402

FS = 256.0
# palette order: the first colours are the most useful ones to have
ORDER = ["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange",
         "purple", "pink", "lime"]
codes_for = bwb.palette_codes            # the live session's own assignment


def gain_of(colour, tint):
    """SSVEP amplitude relative to white: sqrt of the tint's luminance (assumed)."""
    return float(np.sqrt(bwb.LUMA @ bwb.tint_of(colour, tint)))


class Subject:
    def __init__(self, rng, scale, jitter):
        self.rng, self.scale, self.jitter = rng, scale, jitter
        self.lat = 0.12 + 0.02 * rng.standard_normal(5)       # per-channel latency (s)
        a = np.exp(-1.0 / (0.3 * FS))
        self.env = lambda n: lfilter([1 - a], [1, -a], np.ones(n))

    def add(self, X, i0, n, code, g):
        """Add the response to `code` starting at sample i0 for n samples."""
        if code is None or g <= 0:
            return
        t = np.arange(n) / FS
        d = self.jitter * self.rng.standard_normal()
        g = g * self.scale * np.exp(0.3 * self.rng.standard_normal())
        f, phi = code["hz"], np.deg2rad(code["phaseDeg"])
        env = self.env(n)
        for c in range(5):
            x = 2 * np.pi * f * (t - self.lat[c] - d)
            X[i0:i0 + n, c] += SNR[c] * g * env * (np.sin(x - phi) + 0.4 * np.sin(2 * x - 2 * phi))


def trial(rng, n_col, jitter, tint, scale, spc, reps, on_s, quality, grid=(6, 4), cap=6,
          picture=None):
    colours = ORDER[:n_col]
    codes = codes_for(n_col)
    K = n_col + 1
    gains = [0.0] + [gain_of(c, tint) for c in colours]
    if picture is not None:              # a fixed class grid, e.g. the "NO" frame
        tgt = np.asarray(picture).ravel()
        P = tgt.size
    else:                                # every class at least once, rest random
        gw, gh = grid
        P = gw * gh
        tgt = np.concatenate([np.arange(K), rng.integers(0, K, P - K)])
        rng.shuffle(tgt)
    subj = Subject(rng, scale, jitter)
    order = np.concatenate([rng.permutation(K) for _ in range(reps)])
    n_cal = len(order) * (on_s + 1.0)
    n_vis = P * cap
    total = n_cal + n_vis * spc + 2.0
    N = int(total * FS)
    X = np.column_stack([background(rng, N) for _ in range(5)])
    t = np.arange(N) / FS
    blocks, s0 = [], 1.0
    for k in order:
        i0 = int(s0 * FS)
        subj.add(X, i0, int(on_s * FS), codes[k - 1] if k else None, gains[k])
        blocks.append(dict(cls=int(k), t0=s0, t1=s0 + on_s))
        s0 += on_s + 1.0
    pool = {p: [] for p in range(P)}          # cap visits per cell, in time order
    for v in range(cap):
        for p in range(P):
            i0 = int(s0 * FS)
            k = tgt[p]
            subj.add(X, i0, int(spc * FS), codes[k - 1] if k else None, gains[k])
            pool[p].append((s0, s0 + spc))
            s0 += spc
    F = bwb.Filtered(t, X, FS)
    win = max(1.0, spc - bwb.SETTLE)
    names, _ = bwb.classes_of(colours)
    model, T, cal = bwb.train(F, blocks, win, codes, names)
    feat = {p: [bwb.visit_features(F, T, codes, a, b, win) for a, b in pool[p]] for p in range(P)}

    def state(nv):
        out = [bwb.classify(model, [f for f in feat[p][:nv[p]] if f is not None]) for p in range(P)]
        cls = np.array([o[0] for o in out])
        cert = np.array([o[1].max() for o in out])
        return cls, cert

    fixed, conf = [], []
    for npass in (1, 2, 3):
        cls, _ = state([npass] * P)
        fixed.append(int((cls == tgt).sum()))
        conf.append(bwb.confusion(tgt, cls, K))
    nv = [1] * P
    claims = []
    while True:
        cls, cert = state(nv)
        claims.append((cert, cls == tgt))
        if cert.min() >= quality or sum(nv) >= cap * P:
            break
        for p in np.argsort(cert):
            if nv[p] < cap:
                nv[p] += 1
                if sum(nv) % 4 == 0:
                    break
    return dict(cv=cal["cv_accuracy"], fixed=fixed, adapt_visits=sum(nv),
                adapt_right=int((cls == tgt).sum()), adapt_cap=sum(nv) >= cap * P,
                cal_min=n_cal / 60, P=P, claims=claims[0], conf=conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colours", type=int, default=4, help="colour codes (black is extra)")
    ap.add_argument("--jitter-ms", type=float, default=5.0)
    ap.add_argument("--tint", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--spc", type=float, default=4.0, help="seconds per cell visit")
    ap.add_argument("--reps", type=int, default=4, help="calibration blocks per class")
    ap.add_argument("--on", type=float, default=8.0, help="seconds per calibration block")
    ap.add_argument("--quality", type=float, default=0.99)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", default="", help="a bwb target instead of random 6x4 cells, "
                    "e.g. pix:NO with --fg red,green --bg blue --border white")
    ap.add_argument("--fg", default=None)
    ap.add_argument("--bg", default=None)
    ap.add_argument("--border", default=None)
    a = ap.parse_args()
    picture = None
    if a.target:
        import targets
        picture = targets.bwb_target(a.target, fg=a.fg, bg=a.bg, border=a.border,
                                     colours=ORDER[:a.colours])
        names = ["black"] + ORDER[:a.colours]
        print(f"target {a.target} {picture.shape[1]}x{picture.shape[0]}:")
        for r in picture:
            print("  " + " ".join(names[v][0].upper() if v else "." for v in r))
    rng = np.random.default_rng(a.seed)
    R = [trial(rng, a.colours, a.jitter_ms / 1000, a.tint, a.scale, a.spc, a.reps, a.on,
               a.quality, picture=picture) for _ in range(a.trials)]
    P = R[0]["P"]
    codes = codes_for(a.colours)
    print(f"{a.colours} colours + black | jitter {a.jitter_ms:g} ms | tint {a.tint:g} | "
          f"{a.spc:g} s/visit | {a.trials} trials")
    print("  codes: " + ", ".join(f"{c}={d['hz']:.1f}Hz@{d['phaseDeg']:.0f}"
                                  for c, d in zip(ORDER, codes)))
    print("  brightness gain: " + ", ".join(f"{c} {gain_of(c, a.tint):.2f}" for c in ORDER[:a.colours]))
    print(f"  calibration takes {R[0]['cal_min']:.1f} min; scan results (chance {P / (a.colours + 1):.1f}/{P}):")
    for k in range(3):
        v = np.array([r["fixed"][k] for r in R])
        print(f"  fixed {k + 1} pass ({(k + 1) * P * a.spc / 60:.1f} min): {v.mean():.1f}/{P} right, "
              f"all {P}: {np.mean(v == P):.0%}")
    m = np.array([r["adapt_visits"] for r in R]) * a.spc / 60
    v = np.array([r["adapt_right"] for r in R])
    print(f"  quality {a.quality:.2f}: median {np.median(m):.1f} min, 90% by {np.quantile(m, .9):.1f}, "
          f"max {m.max():.1f} | {v.mean():.1f}/{P} right, all {P}: {np.mean(v == P):.0%}, "
          f"hit cap {np.mean([r['adapt_cap'] for r in R]):.0%}")
    cert = np.concatenate([r["claims"][0] for r in R])
    ok = np.concatenate([r["claims"][1] for r in R])
    print("  certainty check after one pass (claimed -> actually right):",
          "; ".join(f"{lo:.2f}-{hi:.2f}: {ok[(cert >= lo) & (cert < hi)].mean():.2f} "
                    f"(n={((cert >= lo) & (cert < hi)).sum()})"
                    for lo, hi in ((0, .7), (.7, .9), (.9, .99), (.99, 1.01))
                    if ((cert >= lo) & (cert < hi)).any()))


if __name__ == "__main__":
    main()
