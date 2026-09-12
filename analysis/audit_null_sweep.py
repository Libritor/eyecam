"""audit_null_sweep.py -- dense circular-shift null for xr_live6 (library only).

For baseline (pipeline weights = AF8) and b_CAR (CAR -> pipeline weights) on
the scan snapshot used by baseline_preproc.py, sweep the circular shift in
STEP_S steps over the whole recording and print: the r at every k/12 shift
(the script's null), the top-8 shifts by r, and the null summary excluding
shifts within EXCL_S of zero/wrap. Purpose: check whether the script's
11-shift null max (b_CAR 0.426) comes from a structurally near-aligned shift.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import reconstruct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_S = 1.0
EXCL_S = 20.0
TRUNC = 0.7
RUN = "xr_live6"


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    sess = os.path.join(ROOT, "runs", RUN)
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(sess)
    target = np.load(os.path.join(sess, "target.npy"))
    n_scan = int(np.searchsorted(eeg_t, cur_t[-1] + TRUNC, side="right"))
    t, d = eeg_t[:n_scan], data[:n_scan]
    fs = reconstruct.infer_fs(t)
    n = len(t)
    d_car = d - d.mean(axis=1, keepdims=True)
    w_af8 = np.array([1.0 if nm == "AF8" else 0.0 for nm in names])
    scan_t0 = cur_t[0] - t[0]
    row_s = (gx.max() + 1) * 4.0
    print(f"{RUN}: n={n} rows = {n / fs:.1f} s, fs={fs:.2f}; scan starts at "
          f"{scan_t0:.1f} s; one scan row = {row_s:.0f} s; calib ON/OFF period 14 s")

    def r_at(dd, w, k):
        g = reconstruct.reconstruct(t, np.roll(dd, k, axis=0), cur_t, gx, gy,
                                    fs=fs, weights=w)
        return corr(target, g)

    out = {}
    ks = np.arange(0, n, int(STEP_S * fs))
    for label, dd in (("baseline", d), ("b_CAR", d_car)):
        rs = np.array([r_at(dd, w_af8, int(k)) for k in ks])
        secs = ks / fs
        print(f"\n[{label}] real (shift 0) r={rs[0]:.3f}")
        print("  script k/12 shifts: " + "  ".join(
            f"{k}/12({(n * k // 12) / fs:.0f}s)={r_at(dd, w_af8, n * k // 12):.3f}"
            for k in range(1, 12)))
        keep = (secs >= EXCL_S) & (secs <= n / fs - EXCL_S)
        rk, sk = rs[keep], secs[keep]
        order = np.argsort(rk)[::-1][:8]
        print("  top-8 null shifts: " + "  ".join(
            f"{sk[i]:.0f}s={rk[i]:.3f}" for i in order))
        print(f"  null over {keep.sum()} shifts (excl +-{EXCL_S:.0f}s): mean "
              f"{rk.mean():+.3f} sd {rk.std(ddof=1):.3f} max {rk.max():.3f} "
              f"p95 {np.percentile(rk, 95):.3f} p99 {np.percentile(rk, 99):.3f} "
              f"frac>=real {(rk >= rs[0]).mean():.3f} "
              f"z={(rs[0] - rk.mean()) / rk.std(ddof=1):.2f}")
        # lag structure: is the null periodic at the scan-row period?
        ac = np.correlate(rk - rk.mean(), rk - rk.mean(), "full")[len(rk) - 1:]
        ac /= ac[0]
        lag_row = int(round(row_s / STEP_S))
        print(f"  null autocorr at lag 1 row ({row_s:.0f}s)={ac[lag_row]:.2f}, "
              f"2 rows={ac[2 * lag_row]:.2f}, 14 s={ac[14]:.2f}")
        out[label] = dict(shift_s=secs.tolist(), r=rs.tolist())
    with open(os.path.join(HERE, "audit_null_sweep_results.json"), "w") as f:
        json.dump(out, f)
    print(f"\nresults -> {os.path.join(HERE, 'audit_null_sweep_results.json')}")


if __name__ == "__main__":
    main()
