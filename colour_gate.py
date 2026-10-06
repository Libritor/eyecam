"""Full-stack gate for the two colour paths on more than black/white/blue,
no hardware:

  xr_session.py --mode smooth --smooth-rotate 1  (tags rotate every pass)
  xr_session.py --mode planes                    (R, G, B scanned in turn)
        <-WS/HTTP->  headless browser (stimulus page)
        ^ OSC /muse/eeg
  phantom_subject.py --tuning 13.5:2 --color-gain 0.7,1,0.4

The phantom responds best at 13.5 Hz (2 Hz wide) and weakly to blue, like a
person. The planes run at 15 Hz: the headless browser refreshes at 60 fps,
where 12 Hz is not frame-exact and becomes 10 Hz, which is the phantom's
alpha rhythm (two coherent 10 Hz waves add or cancel at random per dwell).
The smooth tags come from the sweep, 9 to 14 Hz. The target is 'eight': the
8 pure colours. PASS for each path: 90 % of the
coloured cells decoded with the right hue, 85 % of the pure-colour cells as
the right one of the eight (all three channels on the right side of half),
r >= 0.6 and above time-shifted EEG. 4 s per cell for smooth, 5 / 5 / 8 s
per plane; the live defaults are 8 s.

Run:  python colour_gate.py [smooth|planes]     (default: both)
      BROWSER=<path> to choose the browser; COLOUR_GATE_HTTP / COLOUR_GATE_OSC
      to run a second gate beside the first; COLOUR_GATE_EXTRA for more driver
      options (e.g. --smooth-rotate 0 as the fixed-tag control), COLOUR_GATE_SUFFIX
      for the session name
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HTTP = int(os.environ.get("COLOUR_GATE_HTTP", 8085))   # two gates at once: give
OSC = int(os.environ.get("COLOUR_GATE_OSC", 5003))     # the second other ports
PEAK, WIDTH = 13.5, 2.0
GAIN = "0.7,1,0.4"
from smooth_gate import browser_cmd  # noqa: E402


def run_one(mode):
    extra = os.environ.get("COLOUR_GATE_EXTRA", "").split()
    session = os.path.join("runs", f"colour_gate_{mode}" + os.environ.get("COLOUR_GATE_SUFFIX", ""))
    full = os.path.join(ROOT, session)
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(os.path.join(full, "browser_profile"))
    common = ["--session", session, "--http-port", str(HTTP), "--osc-port", str(OSC),
              "--color-target", "eight", "--color-grid-w", "8", "--color-grid-h", "4",
              "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
    if mode == "smooth":
        cmd = ["--mode", "smooth", "--color-reps", "3", "--color-on", "5",
               "--smooth-spc", "4", "--smooth-passes", "3", "--smooth-rotate", "1"]
    else:
        cmd = ["--mode", "planes", "--freq", "15", "--calib-blocks", "4",
               "--calib-on", "5", "--calib-off", "5", "--calib-style", "bw",
               "--calib-size", "1", "--plane-spc", "5,5,8", "--plane-passes", "1"]
    procs = []
    try:
        driver = subprocess.Popen([sys.executable, "xr_session.py"] + cmd + common + extra, cwd=ROOT)
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
             "--tuning", f"{PEAK}:{WIDTH}", "--color-gain", GAIN, "--max-seconds", "1200"],
            cwd=ROOT))
        code = driver.wait(timeout=1200)
        with open(os.path.join(full, "xr_session.json")) as f:
            res = json.load(f)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    sm = res.get("smooth" if mode == "smooth" else "planes", {})
    hue = sm.get("hue_right") or [0, 0]
    checks = {
        "driver exit 0": code == 0,
        "r >= 0.6": (sm.get("r_all") or 0) >= 0.6,
        "above time-shifted EEG": (sm.get("r_all") or 0) > max(sm.get("r_null") or [1]),
        "hue right on 90 % of the coloured cells": bool(hue[1]) and hue[0] >= 0.9 * hue[1],
        "85 % of the pure-colour cells right": bool(sm.get("pure_cells")) and sm.get("pure_right") >= 0.85 * sm.get("pure_cells"),
    }
    print(f"---- {mode}")
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    if "r_all" in sm:
        print("r = %.2f, planes %s, time-shifted %s, pure-colour cells %s of %s, hue %s of %s" % (
            sm["r_all"], [round(v, 2) for v in sm["r_planes"]],
            [round(v, 2) for v in sm.get("r_null", [])], sm.get("pure_right"),
            sm.get("pure_cells"), hue[0], hue[1]))
    if mode == "smooth":
        print("tags R, G, B:", sm.get("tags"), "|", sm.get("note"))
    return all(checks.values())


def main():
    modes = sys.argv[1:] or ["smooth", "planes"]
    ok = True
    for m in modes:
        ok &= run_one(m)
    print("COLOUR GATE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
