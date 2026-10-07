"""Certainty-driven scanning: one full pass, then the least certain squares
again, a few at a time, until every square is at least --certainty sure of its
colour (default 99 %) or a cap is reached.

Certainty comes from the EEG only, never from the target.

Black & white: each square is called white when its mean log paper score is
above the Otsu threshold of all squares' means; its certainty is
Phi(|mean - threshold| / (sd / sqrt(visits))). The spread sd is the larger of
  - the residual spread around the two classes' means, and
  - the spread between repeat visits to the same square.
The second is the guard: with no signal at all, Otsu still splits pure noise
into two tidy classes and the first alone would grow confident; repeat visits
to one square cannot fake agreement. So no stop is allowed before some squares
have been visited twice.

Colour: the colour decoder's own posterior (bwb.classify), whose certainty is
matched to held-out calibration accuracy.

Offline check: analysis/bwb_palette_sim.py (colour) and
analysis/adaptive_bw_sim.py (black & white).
"""
import os

import numpy as np
from scipy.stats import norm

from nofigure import otsu


def bw_certainty(scores, shape):
    """scores: {(gy, gx): [score per visit]}. Returns (white call, certainty,
    visits) grids and the number of squares with a repeat visit."""
    gh, gw = shape
    n = np.zeros(shape, int)
    m = np.full(shape, np.nan)
    for (gy, gx), v in scores.items():
        v = np.log(np.asarray(v, float))
        v = v[np.isfinite(v)]
        if len(v):
            n[gy, gx], m[gy, gx] = len(v), v.mean()
    seen = n > 0
    call = np.zeros(shape, bool)
    cert = np.zeros(shape)
    if seen.sum() < 3:
        return call, cert, n, 0
    thr = otsu(m[seen])
    call = seen & (m > thr)
    res, rep = [], []
    for (gy, gx), v in scores.items():
        v = np.log(np.asarray(v, float))
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        cls = m[seen & call].mean() if call[gy, gx] else m[seen & ~call].mean()
        res += list(v - cls)
        if len(v) > 1:
            rep += list(v - v.mean())
    sd = np.sqrt(np.mean(np.square(res)))
    repeats = int((n > 1).sum())
    if repeats:
        # within-square spread, corrected for each square's own mean
        k = sum(len(v) for v in scores.values() if len(v) > 1)
        sd = max(sd, np.sqrt(np.sum(np.square(rep)) / max(k - repeats, 1)))
    z = np.where(seen, np.abs(m - thr) / (sd / np.sqrt(np.maximum(n, 1))), 0.0)
    cert = np.where(seen, norm.cdf(z), 0.0)
    return call, cert, n, repeats


def bw_state(session, weights, stim_freq, method="paper", mask_blinks=False, mask_opts=None):
    """Live black & white certainty from the session's eeg.csv and cursor_log.csv."""
    import reconstruct
    info = {}
    grid, _ = reconstruct.run(session, stim_freq=stim_freq, method=method, weights=weights,
                              save=False, vblend=False, mask_blinks=mask_blinks,
                              mask_opts=mask_opts, info=info)
    scores = {}
    for gy, gx, v in info.get("scores", []):
        scores.setdefault((gy, gx), []).append(v)
    return bw_certainty(scores, grid.shape)


def colour_state(session, codes, blocks, spc, grid_w, grid_h, colours, channels=""):
    """Live colour certainty: (class, certainty, visits) grids."""
    import bwb
    t, X, _, fs = bwb.load_eeg(session, channels)
    clock = bwb.page_clock(session, ["bwb_cal_log.csv", "bwb_log.csv"])
    blocks = bwb.retime_blocks(session, blocks, clock)
    visits = bwb.scan_visits(session, "bwb_log.csv", clock)
    cls, post, seen, _, _ = bwb.decode_arrays(t, X, fs, codes, blocks, visits, spc,
                                              grid_w, grid_h, colours)
    n = np.zeros((grid_h, grid_w), int)
    for gx, gy, *_ in visits:
        n[gy, gx] += 1
    cert = np.where(seen, post.max(axis=2), 0.0)
    return cls, cert, n


def pick(cert, n, batch, cap_visits):
    """The `batch` least certain squares that are under the per-square cap,
    in an order that does not visit one square twice in a row."""
    order = np.argsort(cert, axis=None)
    out = []
    for i in order:
        gy, gx = np.unravel_index(i, cert.shape)
        if n[gy, gx] < cap_visits:
            out.append([int(gx), int(gy)])
        if len(out) >= batch:
            break
    return out
