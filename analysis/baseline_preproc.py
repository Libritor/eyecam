"""baseline_preproc.py -- reproduce the eyecam baseline and test PREPROCESSING
variants on the SAME scoring / reconstruction pipeline (reconstruct.py +
run_session.score_calibration), each with a circular-shift CONTROL ARM.

Runs (all under <eyecam>/runs/):
  xr_live6  real human session (dry electrodes, heavy 60 Hz on TP9/TP10)
  webgate   phantom subject over real UDP  -> POSITIVE control (r ~ 0.76)
  xr_live2  floating ear electrodes         -> NEGATIVE control

Variants:
  baseline            reference pipeline, weights recomputed by the pipeline rule
  a1_notch60          60 Hz IIR notch (Q=30) before scoring
  a2_bp1-45           1-45 Hz Butterworth(4) band-pass (sosfiltfilt) before scoring
  a_notch+bp          (a) both
  b_CAR               common-average reference across the 4 channels
  ab_notch+bp+CAR     (a)+(b)
  c_AF8only / c_TPonly / c_uniform   fixed channel weights (combine stage only)
  d_welch2s           2-s Welch segments (nperseg = 2*fs -> 0.5 Hz bins)
  ad_notch+bp+welch2s (a)+(d)

Control arm: the (preprocessed) EEG is circularly shifted by 1/3 of the
recording (rows), timestamps untouched, so cursor/calibration alignment is
broken; shifts k/12 (k=1..11) form an 11-sample null for r and best d'.

Determinism: the xr_live6 recorder was still appending to eeg.csv when this was
written, so the EEG is snapshotted exactly as the session saw it: rows with
t <= calib end + 0.7 s for calibration (own infer_fs), rows with t <= scan end
+ 0.7 s for reconstruction (own infer_fs). Note: d' is sensitive to infer_fs
at the 3rd decimal because the 14/50 Hz band-edge bins flip in/out of the
denominator (e.g. AF7 d' -1.19 vs -1.58 for fs 256.72 vs 257.28).

Usage:  python analysis/baseline_preproc.py [--runs xr_live6,webgate,xr_live2]
Output: printed tables + analysis/baseline_preproc_results.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import config  # noqa: E402
import reconstruct  # noqa: E402
import run_session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRUNC_MARGIN_S = 0.7


# --------------------------------------------------------------------------
# faithful copies of the reference scoring with nperseg as a parameter
# (verified equal to the library at nperseg = 1 s in verify_against_library)
# --------------------------------------------------------------------------

def ssvep_score(segment, fs, nperseg, stim_freq=None):
    if len(segment) < nperseg:
        return np.nan
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return psd[m].sum() * df

    nyquist = fs / 2
    f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
    f1 = 2.0 * f0
    lo, hi = config.NOISE_BAND
    hi_eff = min(hi, nyquist - 1)
    fund = band_power(f0 - 0.6, f0 + 0.6)
    sig_p = fund
    sig_in_band = fund if lo <= f0 <= hi_eff else 0.0
    if f1 < nyquist - 1:
        harm = band_power(f1 - 0.6, f1 + 0.6)
        sig_p += harm
        if lo <= f1 <= hi_eff:
            sig_in_band += harm
    total = band_power(lo, hi_eff)
    denom = total - sig_in_band
    if denom <= 0:
        return np.nan
    return sig_p / denom


def cell_score(seg, fs, sigma_floor, nperseg):
    if len(seg) < 8:
        return np.nan
    baseline = np.median(seg)
    dev = seg - baseline
    sigma = max(reconstruct.robust_sigma(dev), sigma_floor)
    bad = np.abs(dev) > config.ARTIFACT_Z * sigma
    if bad.mean() > config.ARTIFACT_DROP_FRAC:
        return np.nan
    clipped = baseline + np.clip(dev, -config.ARTIFACT_Z * sigma,
                                 config.ARTIFACT_Z * sigma)
    return ssvep_score(clipped, fs, nperseg)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def calib_stats(t, data, names, fs, blocks, nperseg, rank_min):
    """run_session.score_calibration without the file write; rank_min is
    min(config.CALIB_RANK_MIN, n_blocks-1) exactly as xr_session/run_session
    set it from --calib-blocks."""
    channels = {}
    weights = {}
    best = None
    n_nan = 0
    for c, name in enumerate(names):
        sigma = reconstruct.hf_sigma(data[:, c])
        scores = {"on": [], "off": []}
        for kind, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            sc = cell_score(data[i0:i1, c], fs, sigma, nperseg)
            if np.isnan(sc):
                n_nan += 1
            else:
                scores[kind].append(float(sc))
        on, off = np.array(scores["on"]), np.array(scores["off"])
        if len(on) < 3 or len(off) < 3:
            stats = dict(dprime=0.0, ratio=1.0, rank=0)
        else:
            spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
            spread = max(spread, 0.05 * abs(float(np.median(on))))
            stats = dict(
                dprime=float((np.median(on) - np.median(off))
                             / max(spread, 1e-12)),
                ratio=float(np.median(on) / max(np.median(off), 1e-12)),
                rank=int((on > off.max()).sum()),
            )
        channels[name] = stats
        if best is None or stats["dprime"] > channels[best]["dprime"]:
            best = name
        weights[name] = max(0.0, stats["ratio"] - 1.0) \
            if stats["dprime"] >= config.CALIB_WEIGHT_DPRIME_MIN else 0.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(names) for k in names}
    b = channels[best]
    passed = bool(b["dprime"] >= config.CALIB_DPRIME_MIN
                  and b["ratio"] >= config.CALIB_RATIO_MIN
                  and b["rank"] >= rank_min)
    return dict(channels=channels, best=best, passed=passed,
                weights=weights, n_nan_blocks=n_nan)


def reconstruct_grid(eeg_t, data, cur_t, gx, gy, fs, weights, nperseg):
    """reconstruct.reconstruct with nperseg as a parameter; returns
    (grid, n_cells_without_score)."""
    n_ch = data.shape[1]
    weights = np.asarray(weights, dtype=float)
    ch_sigma = [reconstruct.hf_sigma(data[:, c]) for c in range(n_ch)]
    grid_w, grid_h = gx.max() + 1, gy.max() + 1
    acc = np.zeros((grid_h, grid_w))
    cnt = np.zeros((grid_h, grid_w))
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    n_empty = 0
    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(eeg_t, t0)
        i1 = np.searchsorted(eeg_t, t1)
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)
        num = den = 0.0
        for c in range(n_ch):
            if weights[c] <= 0:
                continue
            sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], nperseg)
            if not np.isnan(sc):
                num += weights[c] * sc
                den += weights[c]
        if den > 0:
            acc[gy[s], gx[s]] += num / den
            cnt[gy[s], gx[s]] += 1
        else:
            n_empty += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).all():
        raise ValueError("no cell produced a score")
    if np.isnan(grid).any():
        grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    k = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, k / k.sum(), axis=0, mode="nearest")
    return grid, n_empty


# --------------------------------------------------------------------------
# preprocessing variants
# --------------------------------------------------------------------------

def preprocess(data, fs, pre):
    out = np.array(data, dtype=float, copy=True)
    if "notch" in pre:
        b, a = iirnotch(60.0, 30.0, fs)
        out = filtfilt(b, a, out, axis=0)
    if "bp" in pre:
        sos = butter(4, [1.0, 45.0], btype="bandpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out, axis=0)
    if "car" in pre:
        out = out - out.mean(axis=1, keepdims=True)
    return out


def fixed_weights(names, wmode):
    if wmode == "AF8":
        return np.array([1.0 if n == "AF8" else 0.0 for n in names])
    if wmode == "TP":
        return np.array([1.0 if n in ("TP9", "TP10") else 0.0 for n in names])
    if wmode == "AF":
        return np.array([1.0 if n in ("AF7", "AF8") else 0.0 for n in names])
    if wmode == "uniform":
        return np.ones(len(names))
    raise ValueError(wmode)


VARIANTS = [
    ("baseline",            dict()),
    ("a1_notch60",          dict(pre="notch")),
    ("a2_bp1-45",           dict(pre="bp")),
    ("a_notch+bp",          dict(pre="notch+bp")),
    ("b_CAR",               dict(pre="car")),
    ("ab_notch+bp+CAR",     dict(pre="notch+bp+car")),
    ("c_AF8only",           dict(wmode="AF8")),
    ("c_TPonly",            dict(wmode="TP")),
    ("c_uniform",           dict(wmode="uniform")),
    ("d_welch2s",           dict(seg_s=2.0)),
    ("ad_notch+bp+welch2s", dict(pre="notch+bp", seg_s=2.0)),
]

# Control shifts as fractions of the recording: 1/3 is the primary control
# (task spec); k/12, k=1..11 (which includes 1/3, 1/2, 2/3) form the null set.
PRIMARY_SHIFT = (4, 12)
SHIFTS = [(k, 12) for k in range(1, 12)]


# --------------------------------------------------------------------------
# session loading / verification
# --------------------------------------------------------------------------

def load_run(run):
    """Two run-time snapshots of eeg.csv, exactly as the session saw them:
    rows[:n_cal] (t <= calib end + 0.7 s) is what score_calibration scored,
    rows[:n_scan] (t <= scan end + 0.7 s) is what reconstruct.run used. Each
    snapshot has its own infer_fs, as at run time."""
    sess = os.path.join(ROOT, "runs", run)
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(sess)
    with open(os.path.join(sess, "calibration.json")) as f:
        calib = json.load(f)
    blocks = [tuple(b) for b in calib["blocks"]]
    n_on = sum(1 for b in blocks if b[0] == "on")
    rank_min = min(config.CALIB_RANK_MIN, n_on - 1)
    n_scan = int(np.searchsorted(eeg_t, cur_t[-1] + TRUNC_MARGIN_S,
                                 side="right"))
    n_cal = int(np.searchsorted(eeg_t, blocks[-1][2] + TRUNC_MARGIN_S,
                                side="right"))
    eeg_t, data = eeg_t[:n_scan], data[:n_scan]
    target = np.load(os.path.join(sess, "target.npy"))
    stored_grid = np.load(os.path.join(sess, "reconstruction_grid.npy"))
    fs_scan = reconstruct.infer_fs(eeg_t)
    fs_cal = reconstruct.infer_fs(eeg_t[:n_cal])
    prov = {}
    for fn in ("eeg.csv", "cursor_log.csv", "calib_log.csv",
               "calibration.json", "target.npy", "reconstruction_grid.npy"):
        p = os.path.join(sess, fn)
        st = os.stat(p)
        with open(p, "rb") as f:
            sha = hashlib.sha1(f.read()).hexdigest()[:12]
        prov[fn] = dict(bytes=st.st_size, sha1=sha,
                        mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(st.st_mtime)))
    return dict(run=run, sess=sess, eeg_t=eeg_t, data=data, names=names,
                cur_t=cur_t, gx=gx, gy=gy, calib=calib, blocks=blocks,
                target=target, stored_grid=stored_grid, fs=fs_scan,
                fs_cal=fs_cal, n_cal=n_cal, rank_min=rank_min, prov=prov)


def verify_against_library(S):
    """Local copies must equal the reference implementation at 1-s segments."""
    fs, fs_cal, names, n_cal = S["fs"], S["fs_cal"], S["names"], S["n_cal"]
    # calibration: library writes calibration.json, so give it a temp session
    tmp = tempfile.mkdtemp(prefix="eyecam_verify_")
    saved_rank = config.CALIB_RANK_MIN
    try:
        with open(os.path.join(tmp, "eeg.csv"), "w", newline="") as f:
            f.write("time," + ",".join(names) + "\n")
            for t, row in zip(S["eeg_t"][:n_cal], S["data"][:n_cal]):
                f.write(repr(float(t)) + ","
                        + ",".join(repr(float(v)) for v in row) + "\n")
        config.CALIB_RANK_MIN = S["rank_min"]
        lib = run_session.score_calibration(tmp, S["blocks"])
    finally:
        config.CALIB_RANK_MIN = saved_rank
        shutil.rmtree(tmp, ignore_errors=True)
    loc = calib_stats(S["eeg_t"][:n_cal], S["data"][:n_cal], names, fs_cal,
                      S["blocks"], int(round(fs_cal)), S["rank_min"])
    dd = max(abs(lib["channels"][n]["dprime"] - loc["channels"][n]["dprime"])
             for n in names)
    dw = max(abs(lib["weights"][n] - loc["weights"][n]) for n in names)
    dp = lib["passed"] == loc["passed"]
    # reconstruction
    w = np.array([lib["weights"][n] for n in names])
    g_lib = reconstruct.reconstruct(S["eeg_t"], S["data"], S["cur_t"],
                                    S["gx"], S["gy"], fs=fs, weights=w)
    g_loc, _ = reconstruct_grid(S["eeg_t"], S["data"], S["cur_t"], S["gx"],
                                S["gy"], fs, w, int(round(fs)))
    dg = float(np.abs(g_lib - g_loc).max())
    ok = dd < 1e-9 and dw < 1e-9 and dg < 1e-9 and dp
    print(f"  verify local==library: max|d'diff|={dd:.2e} "
          f"max|w diff|={dw:.2e} passed-eq={dp} max|grid diff|={dg:.2e} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise RuntimeError("local pipeline copy does not match the library")
    print(f"  library re-run (this snapshot): d' "
          + " ".join(f"{n}={lib['channels'][n]['dprime']:.3f}" for n in names)
          + f" passed={lib['passed']} weights={fmt_w(lib['weights'], names)}"
          f"  r={corr(S['target'], g_lib):.4f}")
    return lib


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate_variant(S, pre="", wmode="pipeline", seg_s=1.0):
    fs, fs_cal, names, n_cal = S["fs"], S["fs_cal"], S["names"], S["n_cal"]
    nperseg = int(round(fs * seg_s))
    nperseg_cal = int(round(fs_cal * seg_s))
    # filters run once over the whole scan snapshot; the calibration arm then
    # scores rows[:n_cal] of it (the run-time calibration snapshot)
    data = preprocess(S["data"], fs, pre) if pre else S["data"]
    n = len(data)

    def arm(d):
        return calib_stats(S["eeg_t"][:n_cal], d[:n_cal], names, fs_cal,
                           S["blocks"], nperseg_cal, S["rank_min"])

    real = arm(data)
    if wmode == "pipeline":
        w = np.array([real["weights"][nm] for nm in names])
    else:
        w = fixed_weights(names, wmode)
    grid, n_empty = reconstruct_grid(S["eeg_t"], data, S["cur_t"], S["gx"],
                                     S["gy"], fs, w, nperseg)
    out = dict(
        dprime={nm: real["channels"][nm]["dprime"] for nm in names},
        ratio={nm: real["channels"][nm]["ratio"] for nm in names},
        rank={nm: real["channels"][nm]["rank"] for nm in names},
        passed=real["passed"], best=real["best"],
        weights={nm: float(v) for nm, v in zip(names, w)},
        calib_nan_blocks=real["n_nan_blocks"],
        r=corr(S["target"], grid), empty_cells=n_empty,
        ctrl={},
    )
    # control arms: same weights, EEG circularly shifted (timestamps fixed)
    for num, den in SHIFTS:
        k = (n * num) // den
        d_shift = np.roll(data, k, axis=0)
        cal_c = arm(d_shift)
        grid_c, n_empty_c = reconstruct_grid(
            S["eeg_t"], d_shift, S["cur_t"], S["gx"], S["gy"], fs, w, nperseg)
        out["ctrl"][f"{num}/{den}"] = dict(
            dprime={nm: cal_c["channels"][nm]["dprime"] for nm in names},
            passed=cal_c["passed"], r=corr(S["target"], grid_c),
            empty_cells=n_empty_c)
    p = f"{PRIMARY_SHIFT[0]}/{PRIMARY_SHIFT[1]}"
    rs = np.array([c["r"] for c in out["ctrl"].values()])
    bds = np.array([max(c["dprime"].values()) for c in out["ctrl"].values()])
    out["ctrl_primary"] = out["ctrl"][p]
    out["null"] = dict(
        n=len(rs), r_mean=float(rs.mean()), r_sd=float(rs.std(ddof=1)),
        r_max=float(rs.max()),
        r_z=float((out["r"] - rs.mean()) / max(rs.std(ddof=1), 1e-12)),
        r_frac_ge=float((rs >= out["r"]).mean()),
        bestd_mean=float(bds.mean()), bestd_max=float(bds.max()),
        bestd_frac_ge=float((bds >= max(out["dprime"].values())).mean()),
        n_passed=int(sum(c["passed"] for c in out["ctrl"].values())),
    )
    return out


def fmt_d(dp, names):
    return " ".join(f"{dp[nm]:6.2f}" for nm in names)


def fmt_w(w, names):
    return "/".join(f"{w[nm]:.2f}" for nm in names)


def run_one(run):
    print(f"\n{'=' * 100}\nRUN {run}")
    t0 = time.time()
    S = load_run(run)
    names = S["names"]
    print(f"  EEG rows {len(S['eeg_t'])} (scan snapshot, t <= scan end + "
          f"{TRUNC_MARGIN_S} s) fs={S['fs']:.2f} Hz; calibration snapshot "
          f"{S['n_cal']} rows fs={S['fs_cal']:.2f} Hz (stored "
          f"{S['calib']['fs']}); grid {S['gx'].max() + 1}x{S['gy'].max() + 1}, "
          f"{len(S['blocks'])} calib blocks, rank_min={S['rank_min']}, "
          f"load {time.time() - t0:.1f}s")
    for fn, p in S["prov"].items():
        print(f"  input {fn:<24} {p['bytes']:>10} B  sha1 {p['sha1']}  "
              f"mtime {p['mtime']}")
    n_shift = (len(S["eeg_t"]) * PRIMARY_SHIFT[0]) // PRIMARY_SHIFT[1]
    print(f"  primary control shift 1/3 = {n_shift} rows = "
          f"{n_shift / S['fs']:.1f} s (calib block period "
          f"{config.CALIB_ON_S + config.CALIB_OFF_S:.0f} s; null shifts are "
          f"multiples of {len(S['eeg_t']) / 12 / S['fs']:.1f} s)")
    stored = S["calib"]["channels"]
    print("  stored calibration.json d': "
          + " ".join(f"{nm}={stored[nm]['dprime']:.2f}" for nm in names)
          + f"  passed={S['calib']['passed']}  weights="
          + fmt_w(S["calib"]["weights"], names))
    r_stored = corr(S["target"], S["stored_grid"])
    print(f"  stored reconstruction_grid.npy vs target: r={r_stored:.3f}"
          "  (file on disk; may have been rewritten by later re-runs)")
    verify_against_library(S)

    results = {}
    hdr = (f"  {'variant':<20} | {'d-prime real ' + ' '.join(names):<30} | "
           f"{'d-prime ctrl(1/3 shift)':<27} | pass | {'weights':<19} | "
           f"r_real | r_ctrl1/3 | null r mean+-sd [max] (n=11) | z | frac>= | empty")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for vname, kw in VARIANTS:
        t1 = time.time()
        res = evaluate_variant(S, **kw)
        results[vname] = res
        cp, nl = res["ctrl_primary"], res["null"]
        print(f"  {vname:<20} | {fmt_d(res['dprime'], names):<30} | "
              f"{fmt_d(cp['dprime'], names):<27} | "
              f"{'Y' if res['passed'] else 'n':^4} | "
              f"{fmt_w(res['weights'], names):<19} | "
              f"{res['r']:6.3f} | {cp['r']:9.3f} | "
              f"{nl['r_mean']:6.3f}+-{nl['r_sd']:.3f} [{nl['r_max']:6.3f}]"
              f"      | {nl['r_z']:5.2f} | {nl['r_frac_ge']:.2f}  | "
              f"{res['empty_cells']:>2}/{cp['empty_cells']:<2} "
              f"({time.time() - t1:.1f}s)")
    return S, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="xr_live6,webgate,xr_live2")
    args = ap.parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    import scipy
    print(f"python {sys.version.split()[0]} numpy {np.__version__} "
          f"scipy {scipy.__version__}")
    print(f"config: STIM {config.STIM_FREQ_HZ} Hz, NOISE_BAND "
          f"{config.NOISE_BAND}, ARTIFACT_Z {config.ARTIFACT_Z}, "
          f"DROP_FRAC {config.ARTIFACT_DROP_FRAC}, gate d'>={config.CALIB_DPRIME_MIN} "
          f"ratio>={config.CALIB_RATIO_MIN} rank>={config.CALIB_RANK_MIN}")

    all_results = {}
    for run in runs:
        S, res = run_one(run)
        all_results[run] = dict(fs=S["fs"], fs_cal=S["fs_cal"],
                                names=S["names"], inputs=S["prov"],
                                variants=res)

    # ---- cross-run summary -------------------------------------------------
    print(f"\n{'=' * 100}\nSUMMARY  r = corr(reconstruction, target); "
          f"ctrl = 1/3 circular shift; null = 11 shifts k/12 (max shown); "
          f"bestd = max channel d' (real / 1/3-shift ctrl / null max)")
    cols = [r for r in ("xr_live6", "webgate", "xr_live2") if r in all_results]
    head = f"  {'variant':<20}"
    for r in cols:
        head += f" | {r + ' r/ctrl/nullmax':<22} | {r + ' bestd/ctrl/nullmax':<26}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for vname, _ in VARIANTS:
        line = f"  {vname:<20}"
        for r in cols:
            v = all_results[r]["variants"][vname]
            nl = v["null"]
            bd = max(v["dprime"].values())
            bdc = max(v["ctrl_primary"]["dprime"].values())
            line += (f" | {v['r']:6.3f} {v['ctrl_primary']['r']:6.3f} "
                     f"{nl['r_max']:6.3f}   | {bd:6.2f} {bdc:6.2f} "
                     f"{nl['bestd_max']:6.2f} {'P' if v['passed'] else '-'}"
                     f"{nl['n_passed']:<2}      ")
        print(line)
    print("  (P = calibration gate passed on the real arm; the digit after "
          "P/- = how many of the 11 null shifts passed the gate.\n"
          "   Caveat: a circular shift close to a multiple of the 14-s "
          "ON/OFF period re-aligns calibration blocks, so on a short recording "
          "(webgate, ~90 s) some null shifts are partly aligned and the null "
          "d' max is inflated; the 1/3 shift itself is fully misaligned.)")

    if "xr_live6" in all_results:
        base = all_results["xr_live6"]["variants"]["baseline"]
        print("\n  xr_live6 verdicts vs baseline "
              f"(baseline r={base['r']:.3f}, best d'="
              f"{max(base['dprime'].values()):.2f}). Rule: IMPROVES r needs "
              f"r - baseline > 0.05, r > null max + 0.05, and webgate r >= "
              f"{config.GATE_R_MIN}:")
        for vname, _ in VARIANTS:
            v = all_results["xr_live6"]["variants"][vname]
            nl = v["null"]
            wg = all_results.get("webgate", {}).get("variants", {}).get(vname)
            wg_txt = (f"webgate r={wg['r']:.3f} (ctrl {wg['ctrl_primary']['r']:.3f})"
                      if wg else "webgate n/a")
            dr = v["r"] - base["r"]
            above_ctrl = v["r"] > nl["r_max"] + 0.05
            wg_ok = wg is not None and wg["r"] >= config.GATE_R_MIN
            if dr > 0.05 and above_ctrl and wg_ok:
                tag = "IMPROVES r"
            elif dr > 0.05 and not above_ctrl:
                tag = "higher r but NOT above null"
            elif dr > 0.05 and not wg_ok:
                tag = "higher r but FAILS webgate positive control"
            elif abs(dr) <= 0.05:
                tag = "same/noise"
            else:
                tag = "worse"
            print(f"    {vname:<20} r={v['r']:.3f} (d={dr:+.3f}; null "
                  f"{nl['r_mean']:.3f}+-{nl['r_sd']:.3f} max {nl['r_max']:.3f} "
                  f"z={nl['r_z']:.2f}) best d'={max(v['dprime'].values()):.2f} "
                  f"(ctrl {max(v['ctrl_primary']['dprime'].values()):.2f}, "
                  f"null max {nl['bestd_max']:.2f}) pass={v['passed']} | "
                  f"{wg_txt} -> {tag}")

    out_path = os.path.join(HERE, "baseline_preproc_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
