"""claim_gates.py -- exact, anti-overclaim CLAIM GATES for eye-as-camera sessions.

Modelled on axiomatic-universe's FoundationSpec / refutation.dual_decide contract:
every claim gets an exact, deterministic, $0 gate; the gate is the ONLY thing
that can say PASS; anything the gate cannot decide is ABSTAIN (never bluff).
Verdicts are three-valued: PASS / FAIL / ABSTAIN, with the evidence recorded.

Gates (all computed from session artifacts only: eeg.csv, cursor_log.csv,
calibration.json, target.npy):

  G1 ssvep_present     exact max-over-channels permutation of calibration block
                       labels (Nichols & Holmes max-statistic FWE); PASS iff
                       p < 0.01. ABSTAIN if the block count cannot reach 0.01
                       (needs >= 5 ON + 5 OFF blocks: 1/C(10,5) = 0.004).
  G2 image_above_null  reconstruction r vs target compared with a CIRCULAR-SHIFT
                       null (EEG rolled >= 20 s; keeps autocorrelation, drift,
                       raster order). PASS iff p < 0.01 over >= 200 shifts.
                       G2b readable_image additionally r >= 0.6 (phantom bar).
  G3 frequency_specific same pipeline re-run at control bins {11,13,17,20} Hz
                       with the true stimulus bins ALSO excluded from the
                       denominator; PASS iff r(f0) - max_c r(fc) >= 0.2.
  G4 not_position_artifact |corr(grid,row)| and |corr(grid,col)| <= 0.3
                       (row == time in a serpentine raster: drift shows up
                       here). Also checks the TARGET is position-uncorrelated.
  G5 split_half        r(first-half-of-dwell recon, second-half recon) >= 0.3
                       -- reliability, NOT evidence of SSVEP on its own.
  G6 decoder_improves  (--decoder NAME) decoder r must beat the baseline r by
                       >= 0.1 on THIS session AND pass G2 AND not regress the
                       phantom session (--phantom runs/gate_full, r >= 0.6).

The image claim requires G1 AND G2 AND G3 AND G4 (G5 informational).
Usage:
  python claim_gates.py --session runs/xr_live6 [--n-shift 200] [--decoder row_detrend --phantom runs/gate_full]
Run from the eyecam repo root (imports reconstruct/config).
"""
import argparse
import contextlib
import io
import json
import os
import sys
from itertools import combinations
from math import comb

import numpy as np

sys.path.insert(0, os.getcwd())
import config       # noqa: E402
import reconstruct  # noqa: E402

CONTROL_FREQS = (11.0, 13.0, 17.0, 20.0)
P_GATE = 0.01
R_READABLE = config.GATE_R_MIN          # 0.6, the phantom bar
FREQ_MARGIN = 0.2
POS_MAX = 0.3
SPLIT_MIN = 0.3
DECODER_MARGIN = 0.1
MIN_SHIFT_S = 20.0
GUARD_HZ = 2.0   # denominator guard band around the true stimulus bins (G3)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


# ---------------------------------------------------------------- decoders
def dec_baseline(eeg_t, data, cur_t, gx, gy, fs, w):
    return quiet(reconstruct.reconstruct, eeg_t, data, cur_t, gx, gy, fs=fs, weights=w)


def dec_row_detrend(eeg_t, data, cur_t, gx, gy, fs, w):
    g = dec_baseline(eeg_t, data, cur_t, gx, gy, fs, w)
    return g - g.mean(axis=1, keepdims=True)


def dec_time_detrend(eeg_t, data, cur_t, gx, gy, fs, w):
    g = dec_baseline(eeg_t, data, cur_t, gx, gy, fs, w)
    H, W = g.shape
    order = np.zeros_like(g)
    for r in range(H):
        cols = range(W) if r % 2 == 0 else range(W - 1, -1, -1)
        for k, c in enumerate(cols):
            order[r, c] = r * W + k
    A = np.vstack([order.ravel(), np.ones(g.size)]).T
    coef = np.linalg.lstsq(A, g.ravel(), rcond=None)[0]
    return (g.ravel() - A @ coef).reshape(g.shape)


DECODERS = {"baseline": dec_baseline, "row_detrend": dec_row_detrend,
            "time_detrend": dec_time_detrend}


# ---------------------------------------------------------------- session io
class Session:
    def __init__(self, path):
        self.path = path
        (self.eeg_t, self.data, self.names,
         self.cur_t, self.gx, self.gy) = reconstruct.load_session(path)
        self.fs = quiet(reconstruct.infer_fs, self.eeg_t)
        cpath = os.path.join(path, "calibration.json")
        self.calib = json.load(open(cpath)) if os.path.exists(cpath) else None
        if self.calib:
            w = np.array([self.calib["weights"].get(n, 0.0) for n in self.names])
            self.w = w if w.sum() > 0 else np.ones(len(self.names))
        else:
            self.w = np.ones(len(self.names))
        tpath = os.path.join(path, "target.npy")
        self.target = np.load(tpath) if os.path.exists(tpath) else None


# ---------------------------------------------------------------- G1
def gate_ssvep_present(calib):
    if calib is None:
        return "ABSTAIN", {"reason": "no calibration.json"}
    chans = list(calib["channels"])
    on = {c: np.array(calib["channels"][c]["on_scores"], float) for c in chans}
    off = {c: np.array(calib["channels"][c]["off_scores"], float) for c in chans}
    n_on = min(len(on[c]) for c in chans)
    n_off = min(len(off[c]) for c in chans)
    n_on = min(n_on, len(on[chans[0]]))
    N = n_on + n_off
    min_p = 1.0 / comb(N, n_on) if N else 1.0
    if min_p > P_GATE:
        return "ABSTAIN", {"reason": f"underpowered: {n_on} ON/{n_off} OFF blocks, "
                                     f"min attainable p={min_p:.4f} > {P_GATE}"}
    allv = {c: np.concatenate([on[c][:n_on], off[c][:n_off]]) for c in chans}
    obs = {c: float(on[c][:n_on].mean() - off[c][:n_off].mean()) for c in chans}
    obs_max = max(obs.values())
    cnt = tot = 0
    for idx in combinations(range(N), n_on):
        m = np.zeros(N, bool)
        m[list(idx)] = True
        stat = max(allv[c][m].mean() - allv[c][~m].mean() for c in chans)
        tot += 1
        cnt += stat >= obs_max - 1e-12
    p = cnt / tot
    ev = {"p_max_stat": p, "n_perm": tot, "obs_diff_per_channel": obs,
          "best_channel": max(obs, key=obs.get), "n_on": n_on, "n_off": n_off}
    return ("PASS" if p < P_GATE else "FAIL"), ev


# ---------------------------------------------------------------- G2 (+G6)
def shift_null(s, decoder, n_shift, rng):
    n = len(s.data)
    lo, hi = int(MIN_SHIFT_S * s.fs), n - int(MIN_SHIFT_S * s.fs)
    rs = []
    for _ in range(n_shift):
        off = int(rng.integers(lo, hi))
        g = decoder(s.eeg_t, np.roll(s.data, off, axis=0), s.cur_t, s.gx, s.gy, s.fs, s.w)
        rs.append(corr(g, s.target))
    return np.array(rs)


def gate_image_above_null(s, decoder, n_shift, rng):
    if s.target is None:
        return "ABSTAIN", {"reason": "no target.npy"}, None
    grid = decoder(s.eeg_t, s.data, s.cur_t, s.gx, s.gy, s.fs, s.w)
    r = corr(grid, s.target)
    rs = shift_null(s, decoder, n_shift, rng)
    p = (1 + int((rs >= r).sum())) / (n_shift + 1)
    ev = {"r": r, "p_shift": p, "n_shift": n_shift,
          "null_p99": float(np.percentile(rs, 99)), "null_sd": float(rs.std())}
    if n_shift < 200:
        ev["note"] = "n_shift < 200: p resolution too coarse for 0.01 -> ABSTAIN"
        return "ABSTAIN", ev, grid
    return ("PASS" if p < P_GATE else "FAIL"), ev, grid


# ---------------------------------------------------------------- G3
def make_control_scorer(f_true, f_ctrl):
    """Relative power at f_ctrl (+2 f_ctrl) with BOTH the control bins and the
    TRUE stimulus bins removed from the denominator (otherwise a strong true
    SSVEP in the denominator makes control r artificially negative)."""
    from scipy.signal import welch

    def scorer(segment, fs, stim_freq=None):
        nperseg = int(round(fs))
        if len(segment) < nperseg:
            return np.nan
        freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
        df = freqs[1] - freqs[0]

        def bp(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            return psd[m].sum() * df
        nyq = fs / 2
        lo, hi = config.NOISE_BAND
        hi_eff = min(hi, nyq - 1)
        sig = 0.0
        excl = 0.0
        for f in (f_ctrl, 2 * f_ctrl):
            if f < nyq - 1:
                v = bp(f - 0.6, f + 0.6)
                sig += v
                if lo <= f <= hi_eff:
                    excl += v
        for f in (f_true, 2 * f_true):
            # +-GUARD_HZ leakage guard: a strong amplitude-modulated SSVEP
            # leaks into neighbouring 1 Hz bins; leaving those in the
            # denominator makes control r artificially NEGATIVE.
            if f < nyq - 1:
                excl += bp(max(lo, f - GUARD_HZ), min(hi_eff, f + GUARD_HZ))
        den = bp(lo, hi_eff) - excl
        return sig / den if den > 0 else np.nan
    return scorer


def gate_frequency_specific(s, decoder, r_true):
    if s.target is None:
        return "ABSTAIN", {"reason": "no target.npy"}
    orig = reconstruct.ssvep_score
    rc = {}
    try:
        for fc in CONTROL_FREQS:
            reconstruct.ssvep_score = make_control_scorer(config.STIM_FREQ_HZ, fc)
            g = decoder(s.eeg_t, s.data, s.cur_t, s.gx, s.gy, s.fs, s.w)
            rc[str(fc)] = corr(g, s.target)
    finally:
        reconstruct.ssvep_score = orig
    margin = r_true - max(rc.values())
    ev = {"r_true": r_true, "r_control": rc, "margin": margin}
    return ("PASS" if margin >= FREQ_MARGIN else "FAIL"), ev


# ---------------------------------------------------------------- G4
def gate_not_position_artifact(grid, target):
    H, W = grid.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ev = {"r_grid_row": corr(grid, yy), "r_grid_col": corr(grid, xx),
          "r_target_row": corr(target, yy), "r_target_col": corr(target, xx)}
    if max(abs(ev["r_target_row"]), abs(ev["r_target_col"])) > 0.2:
        return "ABSTAIN", dict(ev, reason="target itself is position-correlated; "
                                          "gate undecidable (choose another target)")
    ok = max(abs(ev["r_grid_row"]), abs(ev["r_grid_col"])) <= POS_MAX
    return ("PASS" if ok else "FAIL"), ev


# ---------------------------------------------------------------- G5
def gate_split_half(s, decoder):
    change = np.flatnonzero((np.diff(s.gx) != 0) | (np.diff(s.gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(s.gx)]))
    halves = []
    for which in (0, 1):
        keep = np.zeros(len(s.eeg_t), bool)
        for a, b in zip(starts, ends):
            ta, tb = s.cur_t[a], s.cur_t[b - 1]
            tm = 0.5 * (ta + tb)
            lo, hi = (ta, tm) if which == 0 else (tm, tb)
            keep |= (s.eeg_t >= lo) & (s.eeg_t < hi)
        halves.append(decoder(s.eeg_t[keep], s.data[keep], s.cur_t, s.gx, s.gy, s.fs, s.w))
    r = corr(halves[0], halves[1])
    return ("PASS" if r >= SPLIT_MIN else "FAIL"), {"r_split_half": r}


# ---------------------------------------------------------------- main
def run(session_path, decoder_name, n_shift, phantom_path, seed):
    rng = np.random.default_rng(seed)
    s = Session(session_path)
    decoder = DECODERS[decoder_name]
    out = {"session": session_path, "decoder": decoder_name, "fs": s.fs,
           "weights": dict(zip(s.names, map(float, s.w))), "gates": {}}

    v, ev = gate_ssvep_present(s.calib)
    out["gates"]["G1_ssvep_present"] = {"verdict": v, "evidence": ev}

    v, ev, grid = gate_image_above_null(s, decoder, n_shift, rng)
    out["gates"]["G2_image_above_null"] = {"verdict": v, "evidence": ev}
    r_true = ev.get("r")
    if grid is not None:
        out["gates"]["G2b_readable_image"] = {
            "verdict": "PASS" if (v == "PASS" and r_true >= R_READABLE) else
                       ("ABSTAIN" if v == "ABSTAIN" else "FAIL"),
            "evidence": {"r": r_true, "r_min": R_READABLE}}
        v, ev = gate_frequency_specific(s, decoder, r_true)
        out["gates"]["G3_frequency_specific"] = {"verdict": v, "evidence": ev}
        v, ev = gate_not_position_artifact(grid, s.target)
        out["gates"]["G4_not_position_artifact"] = {"verdict": v, "evidence": ev}
        v, ev = gate_split_half(s, decoder)
        out["gates"]["G5_split_half_reliability"] = {"verdict": v, "evidence": ev}

    if decoder_name != "baseline":
        base = DECODERS["baseline"](s.eeg_t, s.data, s.cur_t, s.gx, s.gy, s.fs, s.w)
        r_base = corr(base, s.target) if s.target is not None else None
        ev = {"r_decoder": r_true, "r_baseline": r_base,
              "delta": (r_true - r_base) if r_base is not None else None}
        verdict = "ABSTAIN"
        if r_base is not None:
            ok = ev["delta"] >= DECODER_MARGIN and \
                out["gates"]["G2_image_above_null"]["verdict"] == "PASS"
            if phantom_path:
                ph = Session(phantom_path)
                gph = decoder(ph.eeg_t, ph.data, ph.cur_t, ph.gx, ph.gy, ph.fs, ph.w)
                ev["r_phantom"] = corr(gph, ph.target)
                ok = ok and ev["r_phantom"] >= R_READABLE
            else:
                ev["note"] = "no --phantom given: no-regression check not run"
                ok = False
            verdict = "PASS" if ok else "FAIL"
        out["gates"]["G6_decoder_improves"] = {"verdict": verdict, "evidence": ev}

    core = ["G1_ssvep_present", "G2_image_above_null", "G3_frequency_specific",
            "G4_not_position_artifact"]
    vs = [out["gates"].get(k, {}).get("verdict", "ABSTAIN") for k in core]
    out["image_claim"] = ("PASS" if all(v == "PASS" for v in vs) else
                          "FAIL" if any(v == "FAIL" for v in vs) else "ABSTAIN")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--decoder", default="baseline", choices=sorted(DECODERS))
    ap.add_argument("--n-shift", type=int, default=200)
    ap.add_argument("--phantom", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = run(a.session, a.decoder, a.n_shift, a.phantom, a.seed)
    for k, g in out["gates"].items():
        ev = {kk: (round(vv, 3) if isinstance(vv, float) else vv)
              for kk, vv in g["evidence"].items() if kk != "obs_diff_per_channel"}
        print(f"  {g['verdict']:7s} {k}  {ev}")
    print(f"IMAGE CLAIM: {out['image_claim']}  ({a.session}, decoder={a.decoder})")
    path = a.out or os.path.join(a.session, "claim_gates.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    sys.exit(0 if out["image_claim"] == "PASS" else 1)


if __name__ == "__main__":
    main()
