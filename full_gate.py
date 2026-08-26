"""Full-stack gate: the complete session runner, live, with a phantom subject.

Runs run_session.py --source phantom --preset quick --auto --windowed
(~8.5 min real time: 90 s calibration + 6.4 min scan) and verifies:
  1. recorded EEG rate 256 +- 13 Hz (phantom pacing -> UDP -> recorder -> disk)
  2. calibration found the phantom's SNR asymmetry: TP9+TP10 combined weight
     >= 0.7, best d' >= CALIB_DPRIME_MIN, passed == true
  3. reconstruction vs target r >= GATE_R_MIN (0.6)
  4. run_session exit code 0 and a final done/ok event in session.json

Run:  python full_gate.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import config

ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(ROOT, "runs", "gate_full")


def main():
    shutil.rmtree(SESSION, ignore_errors=True)
    if os.path.exists(SESSION):
        sys.exit(f"FULL GATE ERROR: {SESSION} is locked — an orphaned "
                 "osc_acquire/phantom_subject from a previous run is still "
                 "holding files; kill leftover python processes and retry")

    proc = subprocess.Popen(
        [sys.executable, "run_session.py", "--source", "phantom",
         "--preset", "quick", "--target", "text:NO", "--auto", "--windowed",
         "--port", "5012", "--session", "runs/gate_full"],
        cwd=ROOT)
    try:
        proc.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        # kill the whole tree, or the recorder/phantom grandchildren keep
        # the UDP port and session files alive into the next run
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        proc.wait(timeout=10)

    checks = [("exit code 0", proc.returncode == 0, str(proc.returncode))]

    def guarded(name, fn):
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"missing/unreadable: {e!r}"
        checks.append((name, passed, detail))

    def check_rate():
        with open(os.path.join(SESSION, "eeg.csv"), newline="") as f:
            rows = list(csv.reader(f))[1:]
        t = np.array([float(r[0]) for r in rows])
        dt = np.diff(t)
        good = dt < 0.5
        rate = good.sum() / dt[good].sum()
        return abs(rate - 256) <= 13, f"{rate:.1f} Hz"

    def check_calib_passed():
        with open(os.path.join(SESSION, "calibration.json")) as f:
            calib = json.load(f)
        best_dp = calib["channels"][calib["best"]]["dprime"]
        return calib["passed"], f"best={calib['best']} d'={best_dp:.2f}"

    def check_calib_weights():
        with open(os.path.join(SESSION, "calibration.json")) as f:
            calib = json.load(f)
        w = calib["weights"]
        temporal = w.get("TP9", 0) + w.get("TP10", 0)
        return temporal >= 0.7, f"{temporal:.2f}"

    def check_reconstruction():
        target = np.load(os.path.join(SESSION, "target.npy"))
        grid = np.load(os.path.join(SESSION, "reconstruction_grid.npy"))
        r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
        return r >= config.GATE_R_MIN, f"r={r:.3f}"

    def check_done():
        with open(os.path.join(SESSION, "session.json")) as f:
            events = json.load(f)["events"]
        return any(e["stage"] == "done" and e["status"] == "ok"
                   for e in events), ""

    guarded("eeg rate 256+-13 Hz", check_rate)
    guarded("calibration passed", check_calib_passed)
    guarded("TP9+TP10 weight >= 0.7", check_calib_weights)
    guarded(f"reconstruction r >= {config.GATE_R_MIN}", check_reconstruction)
    guarded("done/ok logged", check_done)

    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}  {detail}")
    print("FULL GATE PASS" if ok else "FULL GATE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
