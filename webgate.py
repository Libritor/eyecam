"""Full-stack gate for the browser (Buzzkill-style) path, no hardware:

  xr_session.py (driver, DSP)  <-WS/HTTP->  headless Chrome (stimulus page)
        ^ OSC /muse/eeg 5000
        |
  phantom_subject.py  <- tails the driver's calib/cursor CSVs

PASS: xr_session exits 0 AND xr_session.json shows ok, calibPassed,
r >= 0.6, |measured flicker - 15| <= 0.5 Hz (browser rAF timing).

Run:  python webgate.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(ROOT, "runs", "webgate")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
HTTP, OSC = 8083, 5001


def main():
    shutil.rmtree(SESSION, ignore_errors=True)
    procs = []
    try:
        driver = subprocess.Popen(
            [sys.executable, "xr_session.py", "--session", "runs/webgate",
             "--http-port", str(HTTP), "--osc-port", str(OSC),
             "--grid-w", "8", "--grid-h", "6", "--spc", "0.8",
             "--calib-blocks", "5", "--target", "text:NO",
             "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
            + os.environ.get("WEBGATE_EXTRA", "").split(),
            cwd=ROOT)
        procs.append(driver)
        time.sleep(2)

        chrome_profile = os.path.join(SESSION, "chrome_profile")
        chrome = subprocess.Popen(
            [CHROME, "--headless=new", "--disable-gpu", "--mute-audio",
             "--autoplay-policy=no-user-gesture-required",
             f"--user-data-dir={chrome_profile}",
             f"http://127.0.0.1:{HTTP}/?auto=1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(chrome)

        phantom = subprocess.Popen(
            [sys.executable, "phantom_subject.py", "--session", "runs/webgate",
             "--host", "127.0.0.1", "--port", str(OSC),
             "--max-seconds", "600"],
            cwd=ROOT)
        procs.append(phantom)

        code = driver.wait(timeout=600)
        with open(os.path.join(SESSION, "xr_session.json")) as f:
            res = json.load(f)

        checks = [
            ("driver exit 0", code == 0, str(code)),
            ("session ok", res.get("ok") is True, str(res.get("error", ""))),
            ("calibration passed", res.get("calibPassed") is True, ""),
            ("r >= 0.6", (res.get("r") or 0) >= 0.6,
             f"r={res.get('r'):.3f}" if res.get("r") else "no r"),
            ("delivered flicker matches the frame-exact plan (±0.3 Hz)",
             abs((res.get("measuredFlickerHz") or 0)
                 - (res.get("stimFreqActual") or 15.0)) <= 0.3,
             f"measured {res.get('measuredFlickerHz'):.2f} Hz vs planned "
             f"{res.get('stimFreqActual') or 15.0:.2f} Hz"),
        ]
        ok = all(c[1] for c in checks)
        for name, passed, detail in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}  {detail}")
        print("WEB GATE PASS" if ok else "WEB GATE FAIL")
        return 0 if ok else 1
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
