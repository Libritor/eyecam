"""Gate for a frozen computer mid-scan, no hardware.

On battery this laptop enters Modern Standby after 3 idle minutes and
freezes every process for 10-90 s while the headset page keeps scanning
(vr_planes7, 2026-10-07: five standbys, 135 s of holes in the EEG, blue and
green planes partly scanned blind). Two defences, both tested here:

  driver   holds the PC awake (SetThreadExecutionState) and logs any tick
           of its heartbeat that comes seconds late as "PC FROZE";
  page     treats a heartbeat missing for 3 s as a hold: the position is
           abandoned, shown as PAUSED, and shown again when the driver is
           back (the same path as a lost EEG stream).

The test runs the red plane alone (8x4, 8 s per position) with the phantom
subject in a headless browser, suspends the driver and its recorder for
FREEZE_S seconds some 20 s into the scan, and then checks the log and the
scan log: the freeze is reported, the page sent "paused" frames, the
position on screen at the freeze was shown again afterwards, and the plane
still decodes (r >= 0.6, above the time-shifted EEG).

Run:  python freeze_gate.py       (ports 8086 / 5004; ~6 min)
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from smooth_gate import browser_cmd  # noqa: E402

HTTP = int(os.environ.get("FREEZE_GATE_HTTP", 8086))
OSC = int(os.environ.get("FREEZE_GATE_OSC", 5004))
FREEZE_S = 8.0


def main():
    session = "runs/freeze_gate"
    full = os.path.join(ROOT, session)
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(os.path.join(full, "browser_profile"))
    log_path = os.path.join(ROOT, "runs", "freeze_gate_server.log")
    logf = open(log_path, "w")
    cmd = [sys.executable, "-u", "xr_session.py", "--mode", "planes", "--freq", "15",
           "--calib-blocks", "4", "--calib-on", "5", "--calib-off", "5",
           "--calib-style", "bw", "--calib-size", "1", "--plane-spc", "8",
           "--plane-passes", "1", "--plane-order", "0", "--plane-calib", "0",
           "--session", session, "--http-port", str(HTTP), "--osc-port", str(OSC),
           "--color-target", "eight", "--color-grid-w", "8", "--color-grid-h", "4",
           "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
    procs = []
    froze_at_row = None
    try:
        driver = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        procs.append(driver)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{HTTP}/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        procs.append(subprocess.Popen(
            browser_cmd(f"http://127.0.0.1:{HTTP}/?auto=1", os.path.join(full, "browser_profile")),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, "phantom_subject.py", "--session", session,
             "--host", "127.0.0.1", "--port", str(OSC), "--aux",
             "--tuning", "13.5:2", "--color-gain", "0.7,1,0.4", "--max-seconds", "600"],
            cwd=ROOT))

        def log_text():
            with open(log_path, errors="replace") as f:
                return f.read()

        deadline = time.time() + 240
        while "scanning the red plane" not in log_text():
            if driver.poll() is not None or time.time() > deadline:
                raise SystemExit("the red plane never started:\n" + log_text()[-800:])
            time.sleep(1.0)
        time.sleep(28.0)        # ~3 positions into the plane
        red_log = os.path.join(full, "plane_red_log.csv")
        with open(red_log) as f:
            froze_at_row = sum(1 for _ in f) - 1
        tree = [psutil.Process(driver.pid)]
        tree += tree[0].children(recursive=True)
        t0 = time.time()
        for p in tree:
            p.suspend()
        time.sleep(FREEZE_S)
        for p in tree:
            p.resume()
        print(f"driver + {len(tree) - 1} child process(es) frozen for "
              f"{time.time() - t0:.1f} s at scan-log row {froze_at_row}")
        code = driver.wait(timeout=600)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        logf.close()

    text = log_text()
    with open(os.path.join(full, "xr_session.json")) as f:
        res = json.load(f)
    rows = list(csv.reader(open(os.path.join(full, "plane_red_log.csv"))))[1:]
    cells = [(int(r[1]), int(r[2])) for r in rows]
    before = [c for c in cells[:froze_at_row] if c[0] >= 0]
    at_freeze = before[-1] if before else None
    after = cells[froze_at_row:]
    paused_rows = sum(1 for c in after if c[0] < 0)
    # the position on screen at the freeze must come back after the pause:
    # a later run of that cell that starts after at least one paused row
    redone = False
    seen_pause = False
    for c in after:
        if c[0] < 0:
            seen_pause = True
        elif seen_pause and c == at_freeze:
            redone = True
            break
    planes = res.get("planes", {})
    r_red = (planes.get("r_plane_raw") or [None])[0]
    nulls = planes.get("r_null") or []
    checks = {
        "driver exit 0": code == 0,
        "PC held awake": "holding this PC awake" in text,
        f"freeze reported in the log (>= {FREEZE_S - 1:.0f} s)": any(
            float(v) >= FREEZE_S - 1 for v in res.get("freezes", [])),
        "page sent paused frames after the freeze": paused_rows > 0,
        "frames buffered through the freeze logged as late": (res.get("late_frames") or 0) > 0,
        f"position {at_freeze} shown again after the pause": redone,
        "red plane r >= 0.6": (r_red or 0) >= 0.6,
        "above time-shifted EEG": (planes.get("r_all") or 0) > max(nulls or [1]),
    }
    print("---- freeze gate")
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"freezes logged: {res.get('freezes')}; late frames: {res.get('late_frames')}; "
          f"paused rows: {paused_rows}; "
          f"red r = {r_red}; nulls {[round(v, 2) for v in nulls]}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
