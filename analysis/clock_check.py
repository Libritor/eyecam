"""Which EEG clock is right?  Stimulus-independent test using the 60 Hz mains
line as a reference oscillator (TP9 carries the strongest mains pickup).

For each candidate clock, the Lomb-Scargle line sharpness at 60 Hz is measured
over the full 84 s calibration span and over the twelve 6 s block windows
(the lock-in's integration scale).  Candidates:
  arrival      raw UDP arrival stamps (what reconstruct.py uses via searchsorted)
  linear_env   lower-envelope line per segment (slope fitted)
  linear_cnt   per-segment line with the count-based rate (N-1)/(t_last-t_first)
  env_2s_1s    linear_env + rolling-min(+-2 s) residual, 1 s mean   [lock_in.py v1]
  env_3s_5s    linear_env + rolling-min(+-3 s) residual, 5 s mean
  env_cnt_3s_5s same tracking on the count-rate line

Usage: python analysis/clock_check.py --run xr_live6
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lombscargle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import config                       # noqa: E402
import reconstruct                  # noqa: E402
from lock_in import load_run, lower_envelope_line   # noqa: E402


def build_clock(t_arr, mode, fs_nominal=config.FS):
    n = len(t_arr)
    dt = np.diff(t_arr)
    cuts = np.flatnonzero(dt > 0.5) + 1
    bounds = np.concatenate(([0], cuts, [n]))
    t_hat = np.empty(n)
    rates = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        k = np.arange(s, e, dtype=float)
        if e - s < 20 * fs_nominal:
            line = k / fs_nominal + np.percentile(t_arr[s:e] - k / fs_nominal, 1)
        elif mode.startswith("linear_cnt") or mode.startswith("env_cnt"):
            rate = (e - s - 1) / (t_arr[e - 1] - t_arr[s])
            res = t_arr[s:e] - k / rate
            line = k / rate + np.percentile(res, 1)
            rates.append(rate)
        else:
            a, b = lower_envelope_line(k, t_arr[s:e])
            line = a + b * k
            rates.append(1.0 / b)
        if mode.startswith("env"):
            half = 2.0 if "2s" in mode else 3.0
            smooth = 1.0 if mode.endswith("_1s") else 5.0
            fs = rates[-1] if rates else fs_nominal
            r2 = t_arr[s:e] - line
            env = minimum_filter1d(r2, size=min(int(2 * half * fs) + 1, e - s), mode="nearest")
            env = uniform_filter1d(env, size=min(int(smooth * fs) + 1, e - s), mode="nearest")
            line = line + env
        t_hat[s:e] = line
    return t_hat, rates


def sharpness(t, x, fmin=58.0, fmax=62.0, df=0.005, core=0.03):
    freqs = np.arange(fmin, fmax + 1e-9, df)
    P = lombscargle(t - t[0], x - x.mean(), 2 * np.pi * freqs, normalize=False)
    near = np.abs(freqs - 60.0) <= 0.5
    k = int(np.argmax(np.where(near, P, -1)))
    band = np.abs(freqs - 60.0) <= 0.5
    core_m = np.abs(freqs - freqs[k]) <= core
    return float(freqs[k]), float(P[core_m].sum() / P[band].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="xr_live6")
    args = ap.parse_args()
    run = load_run(os.path.join(ROOT, "runs", args.run))
    t_arr, data = run["t_arr"], run["data"]
    c = int(np.argmax([reconstruct.robust_sigma(data[:, i]) for i in range(data.shape[1])]))
    print(f"run {args.run}: mains channel {run['names'][c]}; reference infer_fs={reconstruct.infer_fs(t_arr):.3f}")
    lg = run["logs"]["calib_log.csv"]
    s0, s1 = lg["ta"][0], lg["ta"][-1]
    blocks = [(t0 + 1.0, t1) for _, t0, t1 in run["blocks"]]
    x_all = data[:, c].astype(float)
    print(f"{'clock':<16} {'fs(seg)':>9} | {'84 s span: f_peak':>17} {'conc+-0.03Hz':>13} | "
          f"{'6 s blocks: median conc(+-0.2Hz)':>32} {'min':>6}")
    for mode in ("arrival", "linear_env", "linear_cnt", "env_2s_1s", "env_3s_5s", "env_cnt_3s_5s"):
        if mode == "arrival":
            t_hat, rates = t_arr, []
        else:
            t_hat, rates = build_clock(t_arr, mode)
        i0, i1 = np.searchsorted(t_hat, s0), np.searchsorted(t_hat, s1)
        x = x_all[i0:i1] - np.median(x_all[i0:i1])
        sig = reconstruct.robust_sigma(x)
        x = np.clip(x, -config.ARTIFACT_Z * sig, config.ARTIFACT_Z * sig)
        fpk, conc = sharpness(t_hat[i0:i1], x)
        concs = []
        for b0, b1 in blocks:
            j0, j1 = np.searchsorted(t_hat, b0), np.searchsorted(t_hat, b1)
            xb = x_all[j0:j1] - np.median(x_all[j0:j1])
            sb = reconstruct.robust_sigma(xb)
            xb = np.clip(xb, -config.ARTIFACT_Z * sb, config.ARTIFACT_Z * sb)
            concs.append(sharpness(t_hat[j0:j1], xb, df=0.02, core=0.2)[1])
        fs_txt = f"{max(rates, key=lambda r: r):9.3f}" if rates else "      n/a"
        print(f"{mode:<16} {fs_txt} | {fpk:17.3f} {conc:13.2f} | {np.median(concs):32.2f} {min(concs):6.2f}")


if __name__ == "__main__":
    main()
