"""ACT (Adaptive Chirplet Transform) SSVEP detector vs the Welch baseline.

Detector under test ("act"):
  For each calibration block / scan-cell visit, the artifact-clipped EEG
  segment (identical clipping to reconstruct.cell_score) is decomposed with
  ACTv14Skew (Vicol & Mann, analytic-gradient skew-normal chirplet engine,
  C:/Users/alexa/adaptive-chirplet-transform) on a dictionary CONSTRAINED to
  the stimulus: centre frequency seeds at 15 +- 0.2 Hz and 30 +- 0.2 Hz
  (refine bounds +- 0.4 Hz around the seed), chirp-rate seed 0 with refine
  bounds +- 0.5 Hz/s, Gaussian durations Dt in {128, 256, 512} samples
  (refine within x2), 8 centre-time seeds across the window, skew seeded 0
  (refine +- 1). OMP order 3, no backfit. Score:

      s_act = E_fit / (E_band(14-50 Hz) - E_fit)

  where E_fit = energy explained by the fitted atoms (from the decomposition
  certificate) and E_band is the segment's 14-50 Hz energy (rfft, Parseval).
  This is the same "stimulus power over broadband power" statistic as the
  Welch baseline (reconstruct.ssvep_score), with the numerator estimated by
  a frequency-constrained, time-adaptive chirplet fit instead of fixed 1-Hz
  Welch bins.

Ablation ("act1", pre-declared): the same engine with order 1 and the
fundamental only -- a single adaptive chirplet matched filter at 15 Hz.

Baseline ("welch"): reconstruct.cell_score / ssvep_score re-run here so the
control arm is computed identically.

Calibration statistic (both detectors): run_session.score_calibration's
robust d' = (med_on - med_off) / sqrt(0.5 (MAD_on^2 + MAD_off^2)), floor
0.05 |med_on|; ratio = med_on / med_off; rank = # ON blocks above max OFF;
passed = best d' >= 1.0 and ratio >= 1.3 and rank >= 5 (of 6).
Channel weights for reconstruction follow the same rule as run_session:
max(0, ratio - 1) if d' >= 0.5 else 0, normalised; uniform if all zero.

Reconstruction: reconstruct.reconstruct's per-visit segmentation, weighted
power-domain combine, Eq. 1 vertical kernel, Pearson r vs target.npy.

CONTROL ARM: everything re-run with the EEG sample matrix circularly shifted
by 1/3 of the recording (timestamps untouched); the control reconstruction
uses the REAL arm's channel weights (same decoder, misaligned data).

Sessions: xr_live6 (live human), webgate (phantom positive control),
xr_live2 (floating-electrode negative control; calibration + partial scan).

Usage:  python analysis/act_chirplet.py             (from C:/Users/alexa/eyecam)
        python analysis/act_chirplet.py --sessions xr_live6,webgate --no-recon
Writes analysis/act_chirplet_results.json and
analysis/<session>_act_{real,ctrl}_grid.npy. Nothing in runs/ is modified.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import convolve1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT_DIR = r"C:\Users\alexa\adaptive-chirplet-transform"
sys.path.insert(0, ROOT)
sys.path.insert(0, ACT_DIR)
import config          # noqa: E402
import reconstruct     # noqa: E402

try:
    from act import ACTv14Skew                       # noqa: E402
    from act.chirplet_v13_skew import skew_make_atom  # noqa: E402
    ACT_OK = True
    ACT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    ACT_OK = False
    ACT_IMPORT_ERROR = repr(exc)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")

F0 = config.STIM_FREQ_HZ            # 15 Hz
CONTROL_FRAC = 1.0 / 3.0
SESSION_MARGIN_S = 5.0              # EEG kept from first calib block -5 s to last cursor +5 s
CANON_L = (256, 512, 1024, 1536, 2048)
ACT_ORDER = 3
# "act" is the primary detector (declared before the first run); "act1" is a
# pre-declared ablation: one atom, fundamental only (a pure chirplet matched
# filter with the lowest noise floor).
METHODS = ("welch", "act", "act1")
FC_SEED_OFFSETS_HZ = (-0.2, 0.0, 0.2)   # seeds; refine radius = 2*step = 0.4 Hz
FC_STEP_HZ = 0.2
C_RADIUS_HZ_S = 0.5                     # chirp-rate refine radius (Hz/s)
LOGDT_STEP = 0.35                       # refine radius 0.7 -> Dt within x2
SKEW_STEP = 0.5                         # refine radius +-1
DT_SEEDS = (128, 256, 512)              # samples; for L=256 -> (64, 128)
N_TC = 8


# ----------------------------------------------------------------- helpers
def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_session(session_dir):
    eeg = np.loadtxt(os.path.join(session_dir, "eeg.csv"), delimiter=",",
                     skiprows=1)
    with open(os.path.join(session_dir, "eeg.csv")) as f:
        names = f.readline().strip().split(",")[1:]
    t = eeg[:, 0]
    data = eeg[:, 1:]
    cur = np.loadtxt(os.path.join(session_dir, "cursor_log.csv"),
                     delimiter=",", skiprows=1)
    cur_t = cur[:, 0]
    gx = cur[:, 1].astype(int)
    gy = cur[:, 2].astype(int)
    with open(os.path.join(session_dir, "calibration.json")) as f:
        calib = json.load(f)
    blocks = [(k, float(t0), float(t1)) for k, t0, t1 in calib["blocks"]]
    target = None
    tp = os.path.join(session_dir, "target.npy")
    if os.path.exists(tp):
        target = np.load(tp)
    # Freeze the analysis to the session span: the recorder can still be
    # appending to eeg.csv, which would otherwise move infer_fs and the
    # control shift between runs.
    t_lo = min(b[1] for b in blocks) - SESSION_MARGIN_S
    t_hi = max(cur_t.max(), max(b[2] for b in blocks)) + SESSION_MARGIN_S
    keep = (t >= t_lo) & (t <= t_hi)
    n_all = len(t)
    t, data = t[keep], data[keep]
    return t, data, names, cur_t, gx, gy, blocks, calib, target, n_all


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def clean_segment(seg, sigma_floor):
    """reconstruct.cell_score's artifact handling, returning the clipped
    segment (or None if blink/motion-dominated) instead of a score."""
    if len(seg) < 8:
        return None
    baseline = np.median(seg)
    dev = seg - baseline
    sigma = max(reconstruct.robust_sigma(dev), sigma_floor)
    bad = np.abs(dev) > config.ARTIFACT_Z * sigma
    if bad.mean() > config.ARTIFACT_DROP_FRAC:
        return None
    return baseline + np.clip(dev, -config.ARTIFACT_Z * sigma,
                              config.ARTIFACT_Z * sigma)


def dprime_stats(on, off):
    on, off = np.asarray(on, float), np.asarray(off, float)
    if len(on) < 3 or len(off) < 3:
        return dict(dprime=0.0, ratio=1.0, rank=0, n_on=int(len(on)),
                    n_off=int(len(off)), on=on.tolist(), off=off.tolist())
    spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
    spread = max(spread, 0.05 * abs(float(np.median(on))))
    return dict(
        dprime=float((np.median(on) - np.median(off)) / max(spread, 1e-12)),
        ratio=float(np.median(on) / max(np.median(off), 1e-12)),
        rank=int((on > off.max()).sum()),
        n_on=int(len(on)), n_off=int(len(off)),
        on=on.tolist(), off=off.tolist())


def weights_from_stats(stats, names):
    w = {}
    for n in names:
        s = stats[n]
        w[n] = max(0.0, s["ratio"] - 1.0) \
            if s["dprime"] >= config.CALIB_WEIGHT_DPRIME_MIN else 0.0
    tot = sum(w.values())
    if tot > 0:
        return {k: v / tot for k, v in w.items()}
    return {k: 1.0 / len(names) for k in names}


def passed_from_stats(stats, names):
    best = max(names, key=lambda n: stats[n]["dprime"])
    b = stats[best]
    ok = (b["dprime"] >= config.CALIB_DPRIME_MIN
          and b["ratio"] >= config.CALIB_RATIO_MIN
          and b["rank"] >= config.CALIB_RANK_MIN)
    return best, bool(ok)


# ----------------------------------------------------------------- scorers
def welch_scorer_factory(fs):
    def score(seg_clean):
        return reconstruct.ssvep_score(seg_clean, fs)
    return score


def canonical_length(n):
    ls = [L for L in CANON_L if L <= n]
    return max(ls) if ls else None


class ActSSVEP:
    """Frequency-constrained ACTv14Skew decomposition -> SSVEP power ratio."""

    def __init__(self, fs, order=ACT_ORDER, harmonic=True):
        self.fs = float(fs)
        self.order = int(order)
        self.harmonic = harmonic
        self._engines = {}
        self.n_transforms = 0
        self.wall = 0.0
        self.fc_hz_log = []      # fitted fc (Hz) of first atom, for sanity

    def engine(self, L):
        if L in self._engines:
            return self._engines[L]
        fs = self.fs
        cyc = L / fs                                   # cycles-per-L per Hz
        fc_step = FC_STEP_HZ * cyc
        c_step = (C_RADIUS_HZ_S / 2.0) * L / (2.0 * fs * fs)   # radius=2*step
        tc_step = L / N_TC
        eng = ACTv14Skew(
            length=L,
            tc_info=(0.0, 1.0, tc_step),
            fc_info=(F0 * cyc, F0 * cyc + 1e-9, fc_step),
            logDt_info=(5.0, 5.01, LOGDT_STEP),
            skew_info=(0.0, 0.01, SKEW_STEP),
            c_info=(0.0, 1e-9, c_step),
            backfit=False, newton=True, analytic_grad=True,
        )
        # Replace the (single-atom) arange dictionary with the constrained
        # multi-band dictionary. The refine bounds (+-2 grid steps) come
        # from the *_info steps above.
        tcs = (np.arange(N_TC) + 0.5) * tc_step
        bands = [F0] + ([2.0 * F0] if self.harmonic else [])
        fcs = [(f + d) * cyc for f in bands for d in FC_SEED_OFFSETS_HZ]
        dts = [d for d in DT_SEEDS if d <= L // 2]
        if not dts:
            dts = [L // 4, L // 2]
        if L <= 256:
            dts = [64, 128]
        params = np.array([(tc, fc, np.log(dt), 0.0, 0.0)
                           for tc in tcs for fc in fcs for dt in dts],
                          dtype=np.float64)
        atoms = np.stack([skew_make_atom(*p, length=L) for p in params]
                         ).astype(np.complex64)
        eng.dict_atoms = atoms
        eng.dict_params = params
        eng.dict_atoms_conj = np.ascontiguousarray(atoms.conj())
        self._engines[L] = eng
        return eng

    def band_energy(self, x, lo=None, hi=None):
        lo = config.NOISE_BAND[0] if lo is None else lo
        hi = config.NOISE_BAND[1] if hi is None else hi
        L = len(x)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(L, 1.0 / self.fs)
        hi_eff = min(hi, self.fs / 2 - 1)
        m = (f >= lo) & (f <= hi_eff)
        return float(2.0 * np.sum(np.abs(X[m]) ** 2) / L)

    def fit(self, x):
        """Return (E_fit, cert) for a median-removed window of canonical length."""
        eng = self.engine(len(x))
        t0 = time.perf_counter()
        _, cert = eng.transform(x, order=self.order)
        self.wall += time.perf_counter() - t0
        self.n_transforms += 1
        e_fit = float(cert.signal_energy - cert.final_residual_energy)
        return e_fit, cert

    def __call__(self, seg_clean):
        return self.score(seg_clean)

    def score(self, seg_clean):
        n = len(seg_clean)
        L = canonical_length(n)
        if L is None:
            return np.nan
        s = (n - L) // 2
        x = np.asarray(seg_clean[s:s + L], dtype=np.float64)
        x = x - np.median(x)
        e_fit, cert = self.fit(x)
        if cert.atoms:
            self.fc_hz_log.append(cert.atoms[0]["fc"] * self.fs / L)
        e_band = self.band_energy(x)
        denom = e_band - e_fit
        if denom <= 0:
            return np.nan
        return e_fit / denom


# ----------------------------------------------------------------- stages
def calibrate(t, data, names, fs, blocks, scorer):
    stats = {}
    for c, name in enumerate(names):
        sigma = reconstruct.hf_sigma(data[:, c])
        on, off = [], []
        for kind, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            seg = clean_segment(data[i0:i1, c], sigma)
            if seg is None:
                continue
            sc = scorer(seg)
            if np.isnan(sc):
                continue
            (on if kind == "on" else off).append(float(sc))
        stats[name] = dprime_stats(on, off)
    return stats


def reconstruct_grid(t, data, cur_t, gx, gy, fs, weights, scorer):
    n_ch = data.shape[1]
    w = np.asarray(weights, float)
    nperseg = int(round(fs))
    ch_sigma = [reconstruct.hf_sigma(data[:, c]) for c in range(n_ch)]
    grid_w, grid_h = gx.max() + 1, gy.max() + 1
    acc = np.zeros((grid_h, grid_w))
    cnt = np.zeros((grid_h, grid_w))
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(t, t0)
        i1 = np.searchsorted(t, t1)
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)
        num = den = 0.0
        for c in range(n_ch):
            if w[c] <= 0:
                continue
            seg = clean_segment(data[i0:i1, c], ch_sigma[c])
            if seg is None:
                continue
            sc = scorer(seg)
            if not np.isnan(sc):
                num += w[c] * sc
                den += w[c]
        if den > 0:
            acc[gy[s], gx[s]] += num / den
            cnt[gy[s], gx[s]] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).all():
        raise ValueError("no cell produced a score")
    if np.isnan(grid).any():
        grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    k = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, k / k.sum(), axis=0, mode="nearest")
    return grid, cnt


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ----------------------------------------------------------------- self-test
def self_test():
    """Synthetic check of the ACT dictionary's unit conversion: a 15 Hz tone
    of amplitude 1 uV in 8 uV white noise, 6 s at 256.7 Hz, must (a) be
    fitted by an atom whose fc is ~15 Hz and (b) score far above noise-only
    and above a 12 Hz tone (which lies outside the constrained band)."""
    fs = 256.7
    rng = np.random.default_rng(0)
    n = int(6 * fs)
    tt = np.arange(n) / fs
    noise = rng.normal(0, 8.0, n)
    act = ActSSVEP(fs)
    out = {}
    for label, f in (("tone15", 15.0), ("tone12", 12.0), ("noise", None)):
        x = noise + (2.0 * np.cos(2 * np.pi * f * tt + 0.3) if f else 0.0)
        s = act.score(x)
        fc_hz = act.fc_hz_log[-1]
        out[label] = (s, fc_hz)
        print(f"  self-test {label:7s}: score={s:.4f}  fitted fc={fc_hz:.2f} Hz")
    ok = (out["tone15"][0] > 2 * out["noise"][0]
          and out["tone15"][0] > 2 * out["tone12"][0]
          and abs(out["tone15"][1] - 15.0) < 0.5)
    print(f"  self-test {'PASS' if ok else 'FAIL'} "
          f"({act.n_transforms} transforms, {act.wall:.2f}s)")
    return ok


# ----------------------------------------------------------------- driver
def run_session(name, do_recon=True):
    sdir = os.path.join(RUNS, name)
    (t, data, names, cur_t, gx, gy, blocks, calib_json, target,
     n_all) = load_session(sdir)
    fs = reconstruct.infer_fs(t)
    shift = int(len(data) * CONTROL_FRAC)
    data_ctrl = np.roll(data, shift, axis=0)
    print(f"\n=== {name}: fs={fs:.2f} Hz, {len(data)} samples in session span "
          f"(of {n_all} rows in eeg.csv), {len(blocks)} calib blocks, "
          f"grid {gx.max()+1}x{gy.max()+1}, "
          f"target {None if target is None else target.shape}")
    print(f"control arm: EEG rolled by {shift} samples "
          f"({shift / fs:.1f} s of {len(data) / fs:.1f} s)")

    res = {"fs": fs, "names": names, "calib": {}, "recon": {}}
    scorers = {}
    for method in METHODS:
        for arm, d in (("real", data), ("ctrl", data_ctrl)):
            if method == "welch":
                sc = welch_scorer_factory(fs)
            elif method == "act":
                sc = ActSSVEP(fs, order=ACT_ORDER, harmonic=True)
            else:  # act1: single fundamental atom, no harmonic (ablation)
                sc = ActSSVEP(fs, order=1, harmonic=False)
            scorers[(method, arm)] = sc
            t0 = time.perf_counter()
            st = calibrate(t, d, names, fs, blocks, sc)
            best, ok = passed_from_stats(st, names)
            res["calib"][f"{method}:{arm}"] = dict(
                stats=st, best=best, passed=ok,
                weights=weights_from_stats(st, names))
            log(f"  {name} calib {method}/{arm}: {time.perf_counter()-t0:.1f}s")

    print("\ncalibration d-prime               REAL                        "
          "CONTROL (shifted)")
    print(f"{'detector / channel':22s} {'d-prime':>8s} {'ratio':>7s} "
          f"{'rank':>5s} {'n':>5s}   {'d-prime':>8s} {'ratio':>7s} "
          f"{'rank':>5s} {'n':>5s}")
    for method in METHODS:
        for n in names:
            r = res["calib"][f"{method}:real"]["stats"][n]
            c = res["calib"][f"{method}:ctrl"]["stats"][n]
            print(f"{method:6s} {n:15s} {r['dprime']:8.2f} {r['ratio']:7.2f} "
                  f"{r['rank']:5d} {r['n_on']}/{r['n_off']:<3d}   "
                  f"{c['dprime']:8.2f} {c['ratio']:7.2f} {c['rank']:5d} "
                  f"{c['n_on']}/{c['n_off']:<3d}")
        rr = res["calib"][f"{method}:real"]
        cc = res["calib"][f"{method}:ctrl"]
        print(f"{method:6s} best={rr['best']} passed={rr['passed']} "
              f"weights={ {k: round(v, 2) for k, v in rr['weights'].items()} }"
              f"   | ctrl best={cc['best']} passed={cc['passed']}")
    stored = {n: round(calib_json["channels"][n]["dprime"], 2) for n in names}
    print(f"stored calibration.json d': {stored} (passed={calib_json['passed']})")
    a = scorers[("act", "real")]
    if a.fc_hz_log:
        print(f"act fitted first-atom fc (Hz) on real calib: median "
              f"{np.median(a.fc_hz_log):.2f}, min {min(a.fc_hz_log):.2f}, "
              f"max {max(a.fc_hz_log):.2f}; {a.n_transforms} transforms "
              f"{a.wall:.1f}s")

    if not do_recon or target is None:
        return res

    print(f"\nreconstruction r vs target      REAL   CONTROL   real-ctrl"
          f"   (control uses the real arm's weights)")
    for method in METHODS:
        for wlabel in ("calib", "uniform"):
            if wlabel == "calib":
                wmap = res["calib"][f"{method}:real"]["weights"]
                w = [wmap[n] for n in names]
            else:
                w = [1.0] * len(names)
            rs = {}
            for arm, d in (("real", data), ("ctrl", data_ctrl)):
                sc = scorers[(method, arm)]
                t0 = time.perf_counter()
                try:
                    grid, cnt = reconstruct_grid(t, d, cur_t, gx, gy, fs, w, sc)
                    r = pearson(target, grid) if grid.shape == target.shape \
                        else float("nan")
                    if wlabel == "calib":
                        np.save(os.path.join(
                            HERE, f"{name}_{method}_{arm}_grid.npy"), grid)
                    ncells = int((cnt > 0).sum())
                except Exception as exc:
                    r, ncells = float("nan"), 0
                    log(f"  {name} recon {method}/{arm}/{wlabel} FAILED: {exc!r}")
                rs[arm] = r
                res["recon"][f"{method}:{wlabel}:{arm}"] = dict(r=r, cells=ncells)
                log(f"  {name} recon {method}/{arm}/{wlabel}: r={r:.3f} "
                    f"cells={ncells} {time.perf_counter()-t0:.1f}s")
            print(f"{method:6s} w={wlabel:8s} {rs['real']:12.3f} {rs['ctrl']:9.3f} "
                  f"{rs['real'] - rs['ctrl']:11.3f}")
    gpath = os.path.join(sdir, "reconstruction_grid.npy")
    if os.path.exists(gpath):
        g = np.load(gpath)
        if g.shape == target.shape:
            print(f"stored reconstruction_grid.npy r = {pearson(target, g):.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="xr_live6,webgate,xr_live2")
    ap.add_argument("--no-recon", action="store_true")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    print(f"ACT import: {'ok' if ACT_OK else 'FAILED ' + ACT_IMPORT_ERROR}")
    if not ACT_OK:
        print("ACT unavailable -- no fallback run; aborting.")
        sys.exit(2)
    if not args.skip_self_test:
        print("self-test (synthetic 15 Hz tone):")
        if not self_test():
            print("SELF-TEST FAILED: dictionary unit conversion wrong; aborting")
            sys.exit(3)

    results = {}
    t_start = time.perf_counter()
    for s in args.sessions.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            results[s] = run_session(s, do_recon=not args.no_recon)
        except Exception as exc:
            print(f"\n=== {s}: FAILED {exc!r}")
            results[s] = {"error": repr(exc)}

    print("\n\n=== SUMMARY (real / control) ===")
    print(f"{'session':9s} {'detector':9s} {'best d-prime':>22s}   "
          f"{'recon r (calib w)':>22s}   {'recon r (uniform w)':>22s}")
    for s, res in results.items():
        if "error" in res:
            continue
        names = res["names"]
        for method in METHODS:
            dr = max(res["calib"][f"{method}:real"]["stats"][n]["dprime"]
                     for n in names)
            dc = max(res["calib"][f"{method}:ctrl"]["stats"][n]["dprime"]
                     for n in names)
            rc = res["recon"].get(f"{method}:calib:real", {}).get("r", float("nan"))
            rcc = res["recon"].get(f"{method}:calib:ctrl", {}).get("r", float("nan"))
            ru = res["recon"].get(f"{method}:uniform:real", {}).get("r", float("nan"))
            ruc = res["recon"].get(f"{method}:uniform:ctrl", {}).get("r", float("nan"))
            print(f"{s:9s} {method:9s} {dr:9.2f} / {dc:8.2f}   "
                  f"{rc:9.3f} / {rcc:8.3f}   {ru:9.3f} / {ruc:8.3f}")
    print(f"total wall {time.perf_counter() - t_start:.0f}s")

    with open(os.path.join(HERE, "act_chirplet_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {os.path.join(HERE, 'act_chirplet_results.json')}")


if __name__ == "__main__":
    main()
