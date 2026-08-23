"""Exact gate for the pipeline: simulate -> reconstruct -> verify.

PASS requires BOTH:
  1. aligned reconstruction correlates with the ground-truth target (r >= 0.60)
  2. control arm (EEG circularly shifted, same statistics) stays low (r <= 0.30)

Run:  python gate_test.py
"""

import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args):
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"step failed: {' '.join(args)}")
    return r.stdout


def corr(session):
    target = np.load(os.path.join(ROOT, session, "target.npy"))
    grid = np.load(os.path.join(ROOT, session, "reconstruction_grid.npy"))
    return float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])


def main():
    for d in ("runs/gate_real", "runs/gate_ctrl"):
        shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)

    run(["simulate.py", "--target", "text:NO", "--out-dir", "runs/gate_real",
         "--snr", "0.8", "--seed", "7"])
    run(["reconstruct.py", "--session", "runs/gate_real"])
    r_real = corr("runs/gate_real")

    run(["simulate.py", "--target", "text:NO", "--out-dir", "runs/gate_ctrl",
         "--snr", "0.8", "--seed", "7", "--shuffle-eeg"])
    run(["reconstruct.py", "--session", "runs/gate_ctrl"])
    r_ctrl = corr("runs/gate_ctrl")

    print(f"aligned:  r = {r_real:.3f}  (need >= 0.60)")
    print(f"control:  r = {r_ctrl:.3f}  (need <= 0.30)")
    ok = r_real >= 0.60 and r_ctrl <= 0.30
    print("GATE PASS" if ok else "GATE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
