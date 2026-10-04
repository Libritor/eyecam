"""Adaptive passes vs fixed passes, offline, on the synthetic subject.

Fixed:    every position once per pass, as the live scan does now.
Adaptive: one full pass, then mini-passes of --batch positions, each time the
          positions whose black/white call is least certain. Certainty uses the
          EEG only (never the target): |mean log score - Otsu threshold| over
          its standard error, with the within-class spread pooled across the
          image.

Both arms see the same visit model and are compared at equal numbers of
8 s visits. Each visit is 5 channels of simulate.synth_eeg's background
(1/f + white + 10 Hz alpha, vectorised here) plus a 12 Hz + 24 Hz SSVEP with
the 0.3 s entrainment lag, the channel SNRs of blink_gate.py, and a per-visit
log-normal gain (attention / contact varying from visit to visit). The pixel
is the paper's strict ratio on the centred 1700-sample window, weighted over
channels and averaged over visits, no Eq. 1, as in --legible.

Null: the shifted-EEG null at visit resolution. The visit schedule is kept
(for the adaptive arm, the one chosen live from the real EEG) and every visit
is given the score of the visit k later in time, circularly, for six k that
are not whole passes.

--scale 0.6 matches the 2026-10-01 'NO' run: 3 fixed passes give r ~0.91, ~43.5 of 45.

usage: python analysis/adaptive_passes.py [--trials 200] [--scale 0.6]
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config                    # noqa: E402
import reconstruct as R          # noqa: E402
import targets                   # noqa: E402
from nofigure import otsu        # noqa: E402

FS = config.FS
F0 = 12.0
SPC = 8.0
SNR = np.array([0.8, 0.2, 0.2, 0.8, 1.2])      # TP9 AF7 AF8 TP10 AUX, as blink_gate
W = np.array([0.25, 0.0, 0.0, 0.25, 0.5])
N = int(round(SPC * FS))
WIN = config.PAPER_WINDOW


def background(rng, n):
    """simulate.synth_eeg with snr=0: 1/f + 0.3 white + 10 Hz alpha."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    pink = np.fft.irfft(spec / np.sqrt(np.maximum(f, 0.5)), n)
    t = np.arange(n) / FS
    return (pink / pink.std() + 0.3 * rng.standard_normal(n)
            + 0.6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi)))


def visit(lum, rng, scale, jitter):
    """Score of one 8 s visit to a position of luminance lum."""
    t = np.arange(N) / FS
    a = np.exp(-1.0 / (0.3 * FS))
    env = lfilter([1 - a], [1, -a], np.full(N, lum))
    ssvep = env * (np.sin(2 * np.pi * F0 * t) + 0.4 * np.sin(4 * np.pi * F0 * t))
    g = scale * np.exp(jitter * rng.standard_normal())
    i0 = (N - WIN) // 2
    sc = [R.paper_strict_score((background(rng, N) + SNR[c] * g * ssvep)[i0:i0 + WIN], FS, F0)
          if W[c] > 0 else 0.0 for c in range(5)]
    return float(np.dot(W, sc) / W.sum())


def evaluate(scores, target):
    img = np.array([np.mean(s) for s in scores])
    tv = target.ravel()
    r = np.corrcoef(img, tv)[0, 1]
    acc = np.mean((img > otsu(img)) == (tv > 0.5))
    return r, acc


def uncertainty(scores):
    """|mean log score - threshold| / standard error, per position (EEG only)."""
    m = np.array([np.mean(np.log(s)) for s in scores])
    thr = otsu(m)
    hi = m > thr
    res = [np.log(s) - (m[hi].mean() if hi[i] else m[~hi].mean())
           for i, s in enumerate(scores)]
    sd = np.sqrt(np.mean(np.concatenate(res) ** 2))
    n = np.array([len(s) for s in scores])
    return np.abs(m - thr) / (sd / np.sqrt(n))


def run(target, budgets, rng, scale, jitter, adaptive, batch):
    """Returns ({budget: (r, acc)}, the visits in time order as (position, score))."""
    lum = target.ravel()
    P = len(lum)
    scores = [[visit(lum[p], rng, scale, jitter)] for p in range(P)]
    seq = [(p, scores[p][0]) for p in range(P)]
    used, out = P, {}
    for b in sorted(budgets):
        while used < b:
            k = min(batch if adaptive else P, b - used)
            if adaptive:
                pick = np.argsort(uncertainty(scores))[:k]
            else:
                pick = np.arange(P)[:k]
            for p in pick:
                scores[p].append(visit(lum[p], rng, scale, jitter))
                seq.append((p, scores[p][-1]))
            used += k
        out[b] = evaluate(scores, target)
    return out, seq


def shifted_null(seq, target, n_shifts=6):
    """r with every visit scored by the visit k later (circularly)."""
    P, B, W_ = target.size, len(seq), target.shape[1]
    rs = []
    for j in range(n_shifts):
        k = int(4 + j * (B - 8) / n_shifts)
        # a whole pass maps positions onto themselves, a whole row onto the
        # same column (vertical strokes line up): skip both
        while min(k % P, P - k % P) < 3 or min(k % W_, W_ - k % W_) < 2:
            k += 1
        scores = [[] for _ in range(P)]
        for i, (p, _) in enumerate(seq):
            scores[p].append(seq[(i + k) % B][1])
        rs.append(evaluate(scores, target)[0])
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pix:NO")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.6, help="SSVEP amplitude multiplier")
    ap.add_argument("--jitter", type=float, default=0.3, help="per-visit log gain SD")
    ap.add_argument("--batch", type=int, default=5, help="positions per adaptive mini-pass")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    target = targets.load_target(a.target)
    P = target.size
    budgets = [P, int(1.4 * P), int(1.6 * P), int(1.8 * P), 2 * P, int(2.5 * P), 3 * P]
    rng = np.random.default_rng(a.seed)
    res = {arm: {b: [] for b in budgets} for arm in ("fixed", "adaptive")}
    null = {"fixed": [], "adaptive": []}
    for _ in range(a.trials):
        for arm in res:
            out, seq = run(target, budgets, rng, a.scale, a.jitter,
                           arm == "adaptive", a.batch)
            for b, v in out.items():
                res[arm][b].append(v)
            null[arm] += shifted_null(seq[:2 * P], target)

    print(f"{a.target} {target.shape[1]}x{target.shape[0]}, {a.trials} trials, "
          f"scale {a.scale}, jitter {a.jitter}, batch {a.batch}; one visit = {SPC:.0f} s")
    print(f"{'visits':>6} {'minutes':>7} | {'fixed r':>7} {'right':>7} | "
          f"{'adapt r':>7} {'right':>7} | {'P(adapt 45/45)':>14} {'P(fixed 45/45)':>14}")
    for b in budgets:
        fr, fa = np.array(res["fixed"][b]).T
        ar, aa = np.array(res["adaptive"][b]).T
        print(f"{b:6d} {b * SPC / 60:7.1f} | {fr.mean():7.3f} {fa.mean() * P:7.1f} | "
              f"{ar.mean():7.3f} {aa.mean() * P:7.1f} | {np.mean(aa == 1):14.2f} "
              f"{np.mean(fa == 1):14.2f}")
    fixed3 = np.array(res["fixed"][3 * P])[:, 1].mean()
    hit = [b for b in budgets if np.array(res["adaptive"][b])[:, 1].mean() >= fixed3]
    if hit:
        print(f"adaptive matches 3 fixed passes ({fixed3 * P:.1f} right) at {hit[0]} visits "
              f"= {hit[0] * SPC / 60:.1f} min vs {3 * P * SPC / 60:.1f} min")
    for arm in null:
        v = np.array(null[arm])
        print(f"shifted null ({arm}, {2 * P} visits): r 95th pct {np.quantile(v, .95):.3f}, "
              f"max {v.max():.3f}")


if __name__ == "__main__":
    main()
