"""CCA / filter-bank-CCA SSVEP detector for the eye-as-a-camera pipeline.

Replaces the PSD-ratio score of reconstruct.ssvep_score with the standard
BCI detector (Lin et al. 2006; Chen et al. 2015 FBCCA):

  rho = max canonical correlation between the EEG segment (n x p channels)
        and the reference matrix Y (n x 6) = [sin, cos] at 15, 30, 45 Hz.

Variants scored (all pre-registered before the first run):
  cca       4-channel CCA, 5-50 Hz band-pass + 60 Hz notch, pixel = rho
  fbcca     filter-bank CCA, sub-bands 14-50 / 28-50 / 42-50 Hz, weights
            n^-1.25, pixel = sum_n w_n rho_n^2 (Chen et al. 2015)
  cca_ch    single-channel CCA per electrode (rho = multiple-R of the channel
            on the 6 references); per-channel d' like run_session; the grid
            is the calibration-weighted mean of per-channel rho (same weight
            rule as run_session: max(0, ratio-1) if d' >= 0.5, else 0)
  cca_best  cca_ch using only the channel with the highest calibration d'
  baseline  the shipped detector (reconstruct.cell_score / reconstruct())
            re-run here so its control arm is computed the same way

Calibration: per block (kind, t0, t1) from calibration.json, first
CALIB_DISCARD_S dropped, score = detector output on that segment; robust
d' = (med_on - med_off) / sqrt(0.5 (MAD_on^2 + MAD_off^2)) with the same
0.05*|med_on| floor as run_session.score_calibration; ratio; rank.

Reconstruction: per contiguous cursor-cell visit (searchsorted on arrival
time), detector output = pixel value, Eq. 1 vertical kernel, Pearson r vs
target.npy (computed on the kernel-blended grid before percentile
normalisation, exactly as reconstruct.run does).

CONTROL ARM: everything is re-run with the EEG sample matrix circularly
shifted by 1/3 of the recording (timestamps untouched), which destroys the
stimulus/EEG alignment while keeping every amplitude statistic.

Sessions: xr_live6 (live human), webgate (phantom positive control),
xr_live2 (floating-electrode negative control).

Usage:  python analysis/cca.py            (from C:/Users/alexa/eyecam)
        python analysis/cca.py --sessions xr_live6,webgate
Writes analysis/cca_results.json and analysis/<session>_<variant>_grid.npy.
Nothing in runs/ is modified.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
from scipy.ndimage import convolve1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config          # noqa: E402
import reconstruct     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")

F0 = config.STIM_FREQ_HZ           # 15 Hz
N_HARM = 3                         # 15, 30, 45 Hz
CCA_BAND = (5.0, 50.0)
FB_BANDS = [(14.0, 50.0), (28.0, 50.0), (42.0, 50.0)]
FB_WEIGHTS = np.array([n ** -1.25 for n in range(1, len(FB_BANDS) + 1)])
NOTCH_HZ = 60.0
MARGIN_S = 0.5                     # filter context on each side of a segment
CONTROL_FRAC = 1.0 / 3.0
SESSION_PAD_S = 5.0                # EEG kept: [first calib block - pad, scan end + pad]


# ----------------------------------------------------------------- helpers
def load_blocks(session_dir):
    with open(os.path.join(session_dir, "calibration.json")) as f:
        c = json.load(f)
    return [(k, float(t0), float(t1)) for k, t0, t1 in c["blocks"]], c


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def robust_dprime(on, off):
    """Verbatim copy of the d'/ratio/rank rule in run_session.score_calibration."""
    on, off = np.asarray(on, float), np.asarray(off, float)
    on, off = on[~np.isnan(on)], off[~np.isnan(off)]
    if len(on) < 3 or len(off) < 3:
        return dict(dprime=0.0, ratio=1.0, rank=0, n_on=len(on), n_off=len(off))
    spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
    spread = max(spread, 0.05 * abs(float(np.median(on))))
    return dict(
        dprime=float((np.median(on) - np.median(off)) / max(spread, 1e-12)),
        ratio=float(np.median(on) / max(np.median(off), 1e-12)),
        rank=int((on > off.max()).sum()),
        n_on=int(len(on)), n_off=int(len(off)),
        on=[float(v) for v in on], off=[float(v) for v in off],
    )


def references(n, fs, f0=F0, n_harm=N_HARM):
    t = np.arange(n) / fs
    cols = []
    for h in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * h * f0 * t))
        cols.append(np.cos(2 * np.pi * h * f0 * t))
    return np.column_stack(cols)


def cca_rho(X, Y):
    """First canonical correlation between column sets X (n x p), Y (n x q)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    if X.shape[0] <= max(X.shape[1], Y.shape[1]) + 2:
        return np.nan
    if not np.all(np.isfinite(X)) or np.any(X.std(axis=0) < 1e-12):
        return np.nan
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(min(1.0, s[0]))


_SOS_CACHE = {}


def band_sos(fs, lo, hi):
    key = (round(fs, 3), lo, hi)
    if key not in _SOS_CACHE:
        nyq = fs / 2
        sos = butter(4, [lo / nyq, min(hi, nyq - 1) / nyq], btype="band",
                     output="sos")
        b, a = iirnotch(NOTCH_HZ, 30.0, fs=fs)
        _SOS_CACHE[key] = np.vstack([tf2sos(b, a), sos])
    return _SOS_CACHE[key]


def clean_segment(seg, sigma_floor):
    """Per-channel median-subtract + robust 4-sigma clip (same rule as
    reconstruct.cell_score); returns (cleaned, keep_mask)."""
    out = np.empty_like(seg, dtype=float)
    keep = np.ones(seg.shape[1], dtype=bool)
    for c in range(seg.shape[1]):
        x = seg[:, c]
        dev = x - np.median(x)
        sigma = max(reconstruct.robust_sigma(dev), sigma_floor[c])
        bad = np.abs(dev) > config.ARTIFACT_Z * sigma
        if bad.mean() > config.ARTIFACT_DROP_FRAC:
            keep[c] = False
        out[:, c] = np.clip(dev, -config.ARTIFACT_Z * sigma,
                            config.ARTIFACT_Z * sigma)
    return out, keep


def extract(data, i0, i1, fs, sigma_floor, bands):
    """Segment [i0,i1) with MARGIN_S context, cleaned, filtered per band,
    trimmed. Returns dict band -> (n x p array), plus keep mask."""
    m = int(round(MARGIN_S * fs))
    j0, j1 = max(0, i0 - m), min(len(data), i1 + m)
    raw = data[j0:j1]
    cleaned, keep = clean_segment(raw, sigma_floor)
    a, b = i0 - j0, i1 - j0
    out = {}
    for lo, hi in bands:
        filt = sosfiltfilt(band_sos(fs, lo, hi), cleaned, axis=0)
        out[(lo, hi)] = filt[a:b]
    return out, keep


class Detector:
    """All CCA variants on one segment; returns dict of scalar scores."""

    def __init__(self, fs, names):
        self.fs = fs
        self.names = names
        self.bands = [CCA_BAND] + FB_BANDS

    def score(self, data, i0, i1, sigma_floor):
        n = i1 - i0
        segs, keep = extract(data, i0, i1, self.fs, sigma_floor, self.bands)
        Y = references(n, self.fs)
        out = {}
        X = segs[CCA_BAND]
        # multi-channel CCA on kept channels
        out["cca"] = cca_rho(X[:, keep], Y) if keep.any() else np.nan
        # single-channel CCA per electrode
        for c, nm in enumerate(self.names):
            out["ch:" + nm] = cca_rho(X[:, [c]], Y) if keep[c] else np.nan
        # filter-bank CCA
        if keep.any():
            acc = 0.0
            for w, band in zip(FB_WEIGHTS, FB_BANDS):
                r = cca_rho(segs[band][:, keep], Y)
                if np.isnan(r):
                    acc = np.nan
                    break
                acc += w * r * r
            out["fbcca"] = acc
        else:
            out["fbcca"] = np.nan
        return out


# ------------------------------------------------------------- pipeline
def segment_indices(eeg_t, t0, t1, fs, n_data):
    i0 = np.searchsorted(eeg_t, t0)
    i1 = np.searchsorted(eeg_t, t1)
    need = int(round(fs)) - (i1 - i0)
    if need > 0:  # same padding rule as reconstruct.reconstruct
        i0 = max(0, i0 - need // 2)
        i1 = min(n_data, i0 + int(round(fs)))
    return i0, i1


def calibrate(det, eeg_t, data, blocks, sigma_floor, baseline=True):
    """Per-block detector scores -> per-key robust d'."""
    fs = det.fs
    per_key = {}
    base_scores = {nm: {"on": [], "off": []} for nm in det.names}
    for kind, t0, t1 in blocks:
        i0 = np.searchsorted(eeg_t, t0 + config.CALIB_DISCARD_S)
        i1 = np.searchsorted(eeg_t, t1)
        if i1 - i0 < int(round(fs)):
            continue
        sc = det.score(data, i0, i1, sigma_floor)
        for k, v in sc.items():
            per_key.setdefault(k, {"on": [], "off": []})[kind].append(v)
        if baseline:
            for c, nm in enumerate(det.names):
                b = reconstruct.cell_score(data[i0:i1, c], fs, sigma_floor[c])
                base_scores[nm][kind].append(b)
    stats = {k: robust_dprime(v["on"], v["off"]) for k, v in per_key.items()}
    if baseline:
        stats.update({"base:" + nm: robust_dprime(v["on"], v["off"])
                      for nm, v in base_scores.items()})
    return stats


def channel_weights(stats, names):
    """run_session weight rule applied to the single-channel CCA d'."""
    w = {}
    for nm in names:
        s = stats["ch:" + nm]
        w[nm] = max(0.0, s["ratio"] - 1.0) \
            if s["dprime"] >= config.CALIB_WEIGHT_DPRIME_MIN else 0.0
    tot = sum(w.values())
    if tot > 0:
        w = {k: v / tot for k, v in w.items()}
    else:
        w = {k: 1.0 / len(names) for k in names}
    best = max(names, key=lambda nm: stats["ch:" + nm]["dprime"])
    return w, best


def reconstruct_grid(det, eeg_t, data, cur_t, gx, gy, sigma_floor, weights,
                     best):
    fs = det.fs
    grid_w, grid_h = gx.max() + 1, gy.max() + 1
    keys = ["cca", "fbcca", "cca_ch", "cca_best"]
    acc = {k: np.zeros((grid_h, grid_w)) for k in keys}
    cnt = {k: np.zeros((grid_h, grid_w)) for k in keys}
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    for s, e in zip(starts, ends):
        i0, i1 = segment_indices(eeg_t, cur_t[s], cur_t[e - 1], fs, len(data))
        if i1 - i0 < int(round(fs)):
            continue
        sc = det.score(data, i0, i1, sigma_floor)
        vals = {"cca": sc["cca"], "fbcca": sc["fbcca"]}
        num = den = 0.0
        for nm in det.names:
            v = sc["ch:" + nm]
            if not np.isnan(v) and weights[nm] > 0:
                num += weights[nm] * v
                den += weights[nm]
        vals["cca_ch"] = num / den if den > 0 else np.nan
        vals["cca_best"] = sc["ch:" + best]
        for k, v in vals.items():
            if not np.isnan(v):
                acc[k][gy[s], gx[s]] += v
                cnt[k][gy[s], gx[s]] += 1
    grids = {}
    kern = np.array(config.VERTICAL_KERNEL)
    for k in keys:
        with np.errstate(invalid="ignore"):
            g = np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), np.nan)
        if np.isnan(g).all():
            grids[k] = None
            continue
        if np.isnan(g).any():
            g = np.where(np.isnan(g), np.nanmedian(g), g)
        grids[k] = convolve1d(g, kern / kern.sum(), axis=0, mode="nearest")
    return grids


def pearson(a, b):
    if a is None or b is None or a.shape != b.shape:
        return np.nan
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def run_arm(session, eeg_t, data, names, cur_t, gx, gy, blocks, fs,
            calib_json, target, tag, save=True):
    det = Detector(fs, names)
    sigma_floor = [reconstruct.hf_sigma(data[:, c]) for c in range(data.shape[1])]
    stats = calibrate(det, eeg_t, data, blocks, sigma_floor)
    weights, best = channel_weights(stats, names)
    grids = reconstruct_grid(det, eeg_t, data, cur_t, gx, gy, sigma_floor,
                             weights, best)
    r = {k: pearson(target, g) for k, g in grids.items()}
    # baseline detector with the shipped calibration weights
    base_w = np.array([float(calib_json["weights"].get(n, 0.0)) for n in names])
    if base_w.sum() <= 0:
        base_w = np.ones(len(names))
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        base_grid = reconstruct.reconstruct(eeg_t, data, cur_t, gx, gy, fs=fs,
                                            weights=base_w)
    r["baseline"] = pearson(target, base_grid)
    grids["baseline"] = base_grid
    for k, g in grids.items():
        if g is not None and save:
            np.save(os.path.join(HERE, f"{session}_{k}_{tag}_grid.npy"), g)
    return dict(stats=stats, weights=weights, best=best, r=r)


def run_session(session, null_fracs=()):
    sdir = os.path.join(RUNS, session)
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(sdir)
    blocks, calib_json = load_blocks(sdir)
    # The live recorder may still be appending (xr_live6 was, at analysis
    # time, 13 min past scan end with the headset off): restrict the EEG to
    # the session window so fs, sigma floors and the control shift are
    # reproducible.
    t_lo = min(t0 for _, t0, _ in blocks) - SESSION_PAD_S
    t_hi = float(cur_t[-1]) + SESSION_PAD_S
    n_all = len(eeg_t)
    keep = (eeg_t >= t_lo) & (eeg_t <= t_hi)
    eeg_t, data = eeg_t[keep], data[keep]
    fs = reconstruct.infer_fs(eeg_t)
    target = np.load(os.path.join(sdir, "target.npy"))
    print(f"\n=== {session}: fs={fs:.2f} Hz, {len(eeg_t)} samples in session "
          f"window [{t_lo:.1f}, {t_hi:.1f}] (file has {n_all}), "
          f"{len(blocks)} calib blocks, grid {gy.max()+1}x{gx.max()+1}, "
          f"target {target.shape}")
    real = run_arm(session, eeg_t, data, names, cur_t, gx, gy, blocks, fs,
                   calib_json, target, "real")
    shift = int(len(data) * CONTROL_FRAC)
    data_ctrl = np.roll(data, shift, axis=0)
    ctrl = run_arm(session, eeg_t, data_ctrl, names, cur_t, gx, gy, blocks,
                   fs, calib_json, target, "ctrl")
    print(f"control arm: EEG rolled by {shift} samples "
          f"({shift/fs:.1f} s of {len(data)/fs:.1f} s)")
    nulls = []
    for frac in null_fracs:
        sh = int(len(data) * frac)
        arm = run_arm(session, eeg_t, np.roll(data, sh, axis=0), names, cur_t,
                      gx, gy, blocks, fs, calib_json, target, f"null{frac:.3f}",
                      save=False)
        nulls.append(dict(frac=frac, shift=sh, stats=arm["stats"], r=arm["r"]))

    # ---- calibration table
    print(f"\n{'calibration d-prime':<26}{'REAL':>32}   {'CONTROL (shifted)':>32}")
    print(f"{'detector / channel':<26}{'d-prime':>8}{'ratio':>8}{'rank':>8}"
          f"{'n':>8}   {'d-prime':>8}{'ratio':>8}{'rank':>8}{'n':>8}")
    rows = ([("baseline PSD " + nm, "base:" + nm) for nm in names]
            + [("cca_ch " + nm, "ch:" + nm) for nm in names]
            + [("CCA 4ch", "cca"), ("FBCCA 4ch", "fbcca")])
    for label, key in rows:
        a, b = real["stats"][key], ctrl["stats"][key]
        print(f"{label:<26}{a['dprime']:8.2f}{a['ratio']:8.2f}{a['rank']:8d}"
              f"{a['n_on']:>4}/{a['n_off']:<4}   "
              f"{b['dprime']:8.2f}{b['ratio']:8.2f}{b['rank']:8d}"
              f"{b['n_on']:>4}/{b['n_off']:<4}")
    print(f"stored calibration.json d': "
          + ", ".join(f"{nm} {calib_json['channels'][nm]['dprime']:.2f}"
                      for nm in names)
          + f"  (passed={calib_json['passed']})")
    print(f"cca_ch weights (real): "
          + ", ".join(f"{k} {v:.2f}" for k, v in real["weights"].items())
          + f"; best={real['best']}")

    # ---- reconstruction table
    print(f"\n{'reconstruction r vs target':<26}{'REAL':>10}{'CONTROL':>10}"
          f"{'real-ctrl':>12}")
    for k in ["baseline", "cca", "fbcca", "cca_ch", "cca_best"]:
        a, b = real["r"][k], ctrl["r"][k]
        print(f"{k:<26}{a:10.3f}{b:10.3f}{a-b:12.3f}")
    stored = np.load(os.path.join(sdir, "reconstruction_grid.npy"))
    print(f"stored reconstruction_grid.npy r = {pearson(target, stored):.3f}")

    if nulls:
        print(f"\nnull distribution over {len(nulls)} extra circular shifts "
              f"(fractions {', '.join(f'{n['frac']:.3f}' for n in nulls)}):")
        print(f"{'detector':<26}{'d-prime min':>12}{'mean':>8}{'max':>8}"
              f"{'r min':>10}{'mean':>8}{'max':>8}")
        for label, key, rk in [("baseline (best PSD ch)", "base", "baseline"),
                               ("cca_ch (best ch)", "ch", "cca_ch"),
                               ("CCA 4ch", "cca", "cca"),
                               ("FBCCA 4ch", "fbcca", "fbcca")]:
            if key in ("base", "ch"):
                ds = [max(n["stats"][k]["dprime"] for k in n["stats"]
                          if k.startswith(key + ":")) for n in nulls]
            else:
                ds = [n["stats"][key]["dprime"] for n in nulls]
            rs = [n["r"][rk] for n in nulls]
            print(f"{label:<26}{min(ds):12.2f}{np.mean(ds):8.2f}{max(ds):8.2f}"
                  f"{min(rs):10.3f}{np.mean(rs):8.3f}{max(rs):8.3f}")
    return dict(fs=fs, n_samples=int(len(eeg_t)), n_file=int(n_all),
                window=[t_lo, t_hi], shift=shift,
                real=real, control=ctrl, nulls=nulls,
                stored_dprime={nm: calib_json["channels"][nm]["dprime"]
                               for nm in names},
                stored_r=pearson(target, stored))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="xr_live6,webgate,xr_live2")
    ap.add_argument("--null-shifts", type=int, default=5,
                    help="extra circular shifts (evenly spaced, excluding the "
                         "1/3 control) for a null spread; 0 = off")
    args = ap.parse_args()
    k = args.null_shifts
    null_fracs = [(i + 1) / (k + 2) for i in range(k + 1)
                  if abs((i + 1) / (k + 2) - CONTROL_FRAC) > 1e-9][:k] if k else []
    results = {}
    for s in args.sessions.split(","):
        results[s] = run_session(s.strip(), null_fracs)

    # ---- summary
    print("\n\n=== SUMMARY (real / control) ===")
    print(f"{'session':<10}{'detector':<10}{'best d-prime':>26}"
          f"{'recon r':>22}")
    for s, res in results.items():
        for k in ["baseline", "cca", "fbcca", "cca_ch", "cca_best"]:
            if k == "baseline":
                dk = [x for x in res["real"]["stats"] if x.startswith("base:")]
                d_real = max(res["real"]["stats"][x]["dprime"] for x in dk)
                d_ctrl = max(res["control"]["stats"][x]["dprime"] for x in dk)
            elif k in ("cca_ch", "cca_best"):
                chs = [x for x in res["real"]["stats"] if x.startswith("ch:")]
                d_real = max(res["real"]["stats"][x]["dprime"] for x in chs)
                d_ctrl = max(res["control"]["stats"][x]["dprime"] for x in chs)
            else:
                d_real = res["real"]["stats"][k]["dprime"]
                d_ctrl = res["control"]["stats"][k]["dprime"]
            print(f"{s:<10}{k:<10}{d_real:12.2f} / {d_ctrl:<11.2f}"
                  f"{res['real']['r'][k]:10.3f} / {res['control']['r'][k]:<9.3f}")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(os.path.join(HERE, "cca_results.json"), "w") as f:
        json.dump(clean(results), f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'cca_results.json')}")


if __name__ == "__main__":
    main()
