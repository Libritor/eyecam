"""Full-stack gate for smooth-flicker colour (--mode smooth), no hardware:

  xr_session.py --mode smooth  <-WS/HTTP->  headless browser (stimulus page)
        ^ OSC /muse/eeg
  phantom_subject.py --tuning 11.5:1.5 --color-gain 0.7,1,0.4

The phantom responds best at 11.5 Hz and weakly to blue. PASS: the sweep puts
all three tags within 1.5 Hz of the peak with blue on the strongest, the
decoded mixtures-and-shades picture correlates with the one shown at
r >= 0.6, and time-shifted EEG stays below it.

Run:  python smooth_gate.py            (BROWSER=<path> to choose the browser)
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(ROOT, "runs", "smooth_gate")
HTTP, OSC = 8084, 5002
PEAK = 11.5
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Firefox.app/Contents/MacOS/firefox",
    "google-chrome", "chromium", "firefox",
]


def browser_cmd(url, profile):
    exe = os.environ.get("BROWSER") or next(
        (b for b in BROWSERS if os.path.exists(b) or shutil.which(b)), None)
    if not exe:
        raise SystemExit("no browser found: set BROWSER=<path to chrome or firefox>")
    if "firefox" in exe.lower():
        return [exe, "--headless", "--no-remote", "--profile", profile, url]
    return [exe, "--headless=new", "--disable-gpu", "--mute-audio",
            f"--user-data-dir={profile}", url]


def main():
    shutil.rmtree(SESSION, ignore_errors=True)
    os.makedirs(os.path.join(SESSION, "browser_profile"))
    procs = []
    try:
        driver = subprocess.Popen(
            [sys.executable, "xr_session.py", "--mode", "smooth",
             "--session", "runs/smooth_gate", "--http-port", str(HTTP),
             "--osc-port", str(OSC), "--color-target", "mix",
             "--color-reps", "3", "--color-on", "5", "--smooth-spc", "4",
             "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
            + os.environ.get("SMOOTH_GATE_EXTRA", "").split(), cwd=ROOT)
        procs.append(driver)
        for _ in range(60):                 # a browser does not retry a refused page
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{HTTP}/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        procs.append(subprocess.Popen(
            browser_cmd(f"http://127.0.0.1:{HTTP}/?auto=1",
                        os.path.join(SESSION, "browser_profile")),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [sys.executable, "phantom_subject.py", "--session", "runs/smooth_gate",
             "--host", "127.0.0.1", "--port", str(OSC), "--aux",
             "--tuning", f"{PEAK}:1.5", "--color-gain", "0.7,1,0.4",
             "--max-seconds", "900"], cwd=ROOT))
        code = driver.wait(timeout=900)
        with open(os.path.join(SESSION, "xr_session.json")) as f:
            res = json.load(f)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    sm = res.get("smooth", {})
    tags = sm.get("tags") or [0, 0, 0]
    checks = {
        "driver exit 0": code == 0,
        "stimulus delivered": bool(res.get("g0", {}).get("ok") and res.get("g0_scan", {}).get("ok")),
        "sweep found the response": bool(res.get("sweep", {}).get("passed")),
        "tags within 1.5 Hz of the peak": all(abs(t - PEAK) <= 1.5 for t in tags),
        "blue on the strongest tag": abs(tags[2] - PEAK) == min(abs(t - PEAK) for t in tags),
        "r >= 0.6": (sm.get("r_all") or 0) >= 0.6,
        "above time-shifted EEG": (sm.get("r_all") or 0) > max(sm.get("r_null") or [1]),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print("tags R, G, B:", tags, "|", sm.get("note"))
    if "r_all" in sm:
        print("r = %.2f, planes %s, time-shifted %s, pure-colour cells %s of %s" % (
            sm["r_all"], [round(v, 2) for v in sm["r_planes"]],
            [round(v, 2) for v in sm["r_null"]], sm.get("pure_right"), sm.get("pure_cells")))
    ok = all(checks.values())
    print("SMOOTH GATE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
