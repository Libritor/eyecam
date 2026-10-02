"""Run the two grey-scan variants back to back (large square, then the
paper-sized square). For each: start the driver, wait for live EEG and an
awake headset, push the page to the Quest browser once, wait for the end.

Launched detached; progress goes to runs/run_both.log.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEST = "340YC10GC20Y54"
IP = "10.0.0.185"
COMMON = ["--mode", "visual", "--freq", "12", "--target", "pix:NO", "--spc", "8",
          "--calib-blocks", "6", "--calib-on", "8", "--calib-off", "8",
          "--calib-style", "bw", "--calib-size", "1",
          "--signal-timeout", "3600", "--linger", "15"]
RUNS = [
    ("vr_pix_big", "big1", ["--passes", "2", "--patch", "3", "--board", "0.6"]),
    ("vr_pix_small", "small1", ["--passes", "3", "--patch", "1"]),
]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def adb(*args):
    try:
        return subprocess.run(["adb", "-s", QUEST, *args], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception as exc:
        return f"adb error {exc!r}"


def rows(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main():
    for name, token, extra in RUNS:
        session = os.path.join("runs", name)
        out = open(os.path.join(ROOT, "runs", f"{name}_server.log"), "w")
        err = open(os.path.join(ROOT, "runs", f"{name}_server.err"), "w")
        srv = subprocess.Popen(
            [sys.executable, "-u", "xr_session.py", "--session", session,
             "--page-token", token, *COMMON, *extra],
            cwd=ROOT, stdout=out, stderr=err)
        log(f"{name}: driver started (pid {srv.pid}); waiting for EEG + headset")
        eeg = os.path.join(ROOT, session, "eeg.csv")
        pushed = False
        last = rows(eeg)
        while srv.poll() is None:
            time.sleep(3)
            if pushed:
                continue
            now = rows(eeg)
            streaming, last = now - last > 20000, now      # ~3 s of 5-ch rows
            awake = "mWakefulness=Awake" in adb("shell", "dumpsys", "power")
            if streaming and awake:
                adb("shell", "setprop", "debug.oculus.refreshRate", "72")
                adb("shell", "setprop", "debug.oculus.guardian_pause", "1")
                res = adb("shell", "am start -a android.intent.action.VIEW "
                                   f"-d 'http://{IP}:8082/?k={token}' com.oculus.browser")
                log(f"{name}: page pushed ({res.strip()[:60]})")
                pushed = True
        log(f"{name}: driver exited with {srv.returncode}")
        time.sleep(3)
    log("ALL DONE")


if __name__ == "__main__":
    main()
