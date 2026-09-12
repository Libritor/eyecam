"""audit_baseline_preproc.py -- independent audit of analysis/baseline_preproc.py

Everything here goes through the LIBRARY only (reconstruct.reconstruct,
run_session.score_calibration via a temp session dir); none of the local
copies in baseline_preproc.py are imported. Checks:

 1. baseline r and 1/3-circular-shift control r for xr_live6 / webgate /
    xr_live2 on (a) the script's scan snapshot (t <= scan end + 0.7 s) and
    (b) the FULL eeg.csv as it stands now (post-recorder, 341k rows for
    xr_live6) -- shows whether the snapshot choice matters.
 2. b_CAR (common-average reference) and c_TPonly re-derived with the library:
    CAR -> library calibration (temp dir) for weights -> library reconstruct.
 3. a denser null (N_NULL evenly spaced circular shifts, excluding shifts within
    EXCL_S of zero / wrap) for baseline, b_CAR and c_TPonly on xr_live6, giving
    mean/sd/max/95th pct/frac>=real, to test the script's 11-shift null-max.
 4. compares each number against analysis/baseline_preproc_results.json.

Usage: python analysis/audit_baseline_preproc.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import config  # noqa: E402
import reconstruct  # noqa: E402
import run_session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRUNC = 0.7
N_NULL = 40
EXCL_S = 20.0


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def lib_calibration(t, data, names, blocks, rank_min):
    """run_session.score_calibration on an in-memory array via a temp dir."""
    tmp = tempfile.mkdtemp(prefix="eyecam_audit_")
    saved = config.CALIB_RANK_MIN
    try:
        with open(os.path.join(tmp, "eeg.csv"), "w", newline="") as f:
            f.write("time," + ",".join(names) + "\n")
            for tt, row in zip(t, data):
                f.write(repr(float(tt)) + "," +
                        ",".join(repr(float(v)) for v in row) + "\n")
        config.CALIB_RANK_MIN = rank_min
        return run_session.score_calibration(tmp, blocks)
    finally:
        config.CALIB_RANK_MIN = saved
        shutil.rmtree(tmp, ignore_errors=True)


def load(run):
    sess = os.path.join(ROOT, "runs", run)
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(sess)
    with open(os.path.join(sess, "calibration.json")) as f:
        calib = json.load(f)
    blocks = [tuple(b) for b in calib["blocks"]]
    n_on = sum(1 for b in blocks if b[0] == "on")
    rank_min = min(config.CALIB_RANK_MIN, n_on - 1)
    target = np.load(os.path.join(sess, "target.npy"))
    n_scan = int(np.searchsorted(eeg_t, cur_t[-1] + TRUNC, side="right"))
    n_cal = int(np.searchsorted(eeg_t, blocks[-1][2] + TRUNC, side="right"))
    return dict(run=run, eeg_t=eeg_t, data=data, names=names, cur_t=cur_t,
                gx=gx, gy=gy, calib=calib, blocks=blocks, rank_min=rank_min,
                target=target, n_scan=n_scan, n_cal=n_cal)


def recon_r(S, t, d, w, shift_rows=0):
    fs = reconstruct.infer_fs(t)
    dd = np.roll(d, shift_rows, axis=0) if shift_rows else d
    g = reconstruct.reconstruct(t, dd, S["cur_t"], S["gx"], S["gy"], fs=fs,
                                weights=w)
    return corr(S["target"], g)


def main():
    with open(os.path.join(HERE, "baseline_preproc_results.json")) as f:
        prev = json.load(f)
    print(f"python {sys.version.split()[0]} numpy {np.__version__}")
    summary = {}
    for run in ("xr_live6", "webgate", "xr_live2"):
        S = load(run)
        names = S["names"]
        t_all, d_all = S["eeg_t"], S["data"]
        t_s, d_s = t_all[:S["n_scan"]], d_all[:S["n_scan"]]
        pv = prev[run]["variants"]
        print(f"\n=== {run}: full rows {len(t_all)}, snapshot rows "
              f"{S['n_scan']}, calib rows {S['n_cal']}")

        # --- 1. stored calibration weights (what the session actually used)
        w_stored = np.array([S["calib"]["weights"][n] for n in names])
        # calibration re-derived by the library on the calibration snapshot
        cal = lib_calibration(t_all[:S["n_cal"]], d_all[:S["n_cal"]], names,
                              S["blocks"], S["rank_min"])
        w_lib = np.array([cal["weights"][n] for n in names])
        print("  library calibration d': " +
              " ".join(f"{n}={cal['channels'][n]['dprime']:.3f}" for n in names)
              + f" passed={cal['passed']}  weights={np.round(w_lib, 3)}"
              f"  (stored calibration.json weights {np.round(w_stored, 3)},"
              f" d' " + " ".join(f"{S['calib']['channels'][n]['dprime']:.3f}"
                                 for n in names) + ")")

        k_s = (len(t_s) * 4) // 12
        k_a = (len(t_all) * 4) // 12
        r_snap = recon_r(S, t_s, d_s, w_lib)
        r_snap_c = recon_r(S, t_s, d_s, w_lib, k_s)
        r_full = recon_r(S, t_all, d_all, w_lib)
        r_full_c = recon_r(S, t_all, d_all, w_lib, k_a)
        b = pv["baseline"]
        print(f"  baseline (library, snapshot): r={r_snap:.4f}  ctrl(1/3)="
              f"{r_snap_c:.4f}   | script: r={b['r']:.4f} ctrl="
              f"{b['ctrl_primary']['r']:.4f}  -> "
              f"{'MATCH' if abs(r_snap - b['r']) < 1e-6 and abs(r_snap_c - b['ctrl_primary']['r']) < 1e-6 else 'DIFF'}")
        print(f"  baseline (library, FULL file): r={r_full:.4f}  ctrl(1/3 of "
              f"full)={r_full_c:.4f}  fs_full={reconstruct.infer_fs(t_all):.2f}")
        summary[run] = dict(r_snapshot=r_snap, ctrl_snapshot=r_snap_c,
                            r_full=r_full, ctrl_full=r_full_c,
                            script_r=b["r"], script_ctrl=b["ctrl_primary"]["r"])

        # --- 2. b_CAR and c_TPonly re-derived through the library
        d_car = d_s - d_s.mean(axis=1, keepdims=True)
        cal_car = lib_calibration(t_s[:S["n_cal"]], d_car[:S["n_cal"]], names,
                                  S["blocks"], S["rank_min"])
        w_car = np.array([cal_car["weights"][n] for n in names])
        r_car = recon_r(S, t_s, d_car, w_car)
        r_car_c = recon_r(S, t_s, d_car, w_car, k_s)
        pc = pv["b_CAR"]
        print(f"  b_CAR (library): d' " +
              " ".join(f"{n}={cal_car['channels'][n]['dprime']:.2f}" for n in names)
              + f" w={np.round(w_car, 2)} r={r_car:.4f} ctrl={r_car_c:.4f}"
              f"   | script r={pc['r']:.4f} ctrl={pc['ctrl_primary']['r']:.4f} -> "
              f"{'MATCH' if abs(r_car - pc['r']) < 1e-6 and abs(r_car_c - pc['ctrl_primary']['r']) < 1e-6 else 'DIFF'}")
        w_tp = np.array([1.0 if n in ("TP9", "TP10") else 0.0 for n in names])
        r_tp = recon_r(S, t_s, d_s, w_tp)
        r_tp_c = recon_r(S, t_s, d_s, w_tp, k_s)
        pt = pv["c_TPonly"]
        print(f"  c_TPonly (library): r={r_tp:.4f} ctrl={r_tp_c:.4f}   | script "
              f"r={pt['r']:.4f} ctrl={pt['ctrl_primary']['r']:.4f} -> "
              f"{'MATCH' if abs(r_tp - pt['r']) < 1e-6 and abs(r_tp_c - pt['ctrl_primary']['r']) < 1e-6 else 'DIFF'}")
        summary[run].update(r_car=r_car, ctrl_car=r_car_c, r_tp=r_tp,
                            ctrl_tp=r_tp_c)

        # --- 3. denser null (N_NULL shifts), real weights held fixed as the script does
        fs_s = reconstruct.infer_fs(t_s)
        n = len(t_s)
        lo, hi = int(EXCL_S * fs_s), n - int(EXCL_S * fs_s)
        shifts = np.linspace(lo, hi, N_NULL).astype(int)
        nulls = {}
        for label, dd, ww in (("baseline", d_s, w_lib), ("b_CAR", d_car, w_car),
                              ("c_TPonly", d_s, w_tp)):
            rs = np.array([recon_r(S, t_s, dd, ww, int(k)) for k in shifts])
            real = {"baseline": r_snap, "b_CAR": r_car, "c_TPonly": r_tp}[label]
            nulls[label] = dict(mean=float(rs.mean()), sd=float(rs.std(ddof=1)),
                                max=float(rs.max()), p95=float(np.percentile(rs, 95)),
                                frac_ge=float((rs >= real).mean()), real=real,
                                z=float((real - rs.mean()) / rs.std(ddof=1)))
            nl = nulls[label]
            print(f"  null[{N_NULL} shifts, excl +-{EXCL_S:.0f}s] {label:<9}: real r="
                  f"{real:.3f}  null mean {nl['mean']:+.3f} sd {nl['sd']:.3f} "
                  f"max {nl['max']:.3f} p95 {nl['p95']:.3f} z={nl['z']:.2f} "
                  f"frac>=real {nl['frac_ge']:.3f} "
                  f"(script 11-shift max {pv[label]['null']['r_max']:.3f})")
        summary[run]["null40"] = nulls

    print("\n=== AUDIT TABLE (r = corr(recon, target); ctrl = 1/3 circular shift)")
    print(f"  {'run':<9} | {'script r':>8} {'script ctrl':>11} | {'lib snapshot r':>14} "
          f"{'ctrl':>7} | {'lib FULL-file r':>15} {'ctrl':>7} | {'b_CAR r':>7} {'ctrl':>7} "
          f"| {'TPonly r':>8} {'ctrl':>7}")
    for run, s in summary.items():
        print(f"  {run:<9} | {s['script_r']:8.3f} {s['script_ctrl']:11.3f} | "
              f"{s['r_snapshot']:14.3f} {s['ctrl_snapshot']:7.3f} | "
              f"{s['r_full']:15.3f} {s['ctrl_full']:7.3f} | {s['r_car']:7.3f} "
              f"{s['ctrl_car']:7.3f} | {s['r_tp']:8.3f} {s['ctrl_tp']:7.3f}")
    out = os.path.join(HERE, "audit_baseline_preproc_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
