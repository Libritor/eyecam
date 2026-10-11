"""Gate for the photo run: the synthetic subject looks at one gallery photo
(12 x 12, black and white) through the grey certainty scan in a headless
browser, and analysis/photo_readout.py has to name it against the whole
gallery and agree with the seal. Mechanics and read-out end to end; the
phantom's picture is near-perfect, so a wrong name here is a layout, path or
template bug, not a weak signal. About four minutes.

    python photo_gate.py [--who lincoln] [--spc 3]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from smooth_gate import browser_cmd  # noqa: E402

HTTP, OSC = 8091, 5009            # never the live run's 8082 / 5000


def below_normal():
    """Run the gate and everything it spawns below normal priority, so a live
    headset run on the same PC is not disturbed."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
    else:
        os.nice(10)


def main():
    below_normal()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--who", default="lincoln", help="gallery member to show")
    ap.add_argument("--spc", type=float, default=3.0, help="seconds per position")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    session = "runs/photo_gate"
    full = os.path.join(ROOT, session)
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(os.path.join(full, "browser_profile"))
    subprocess.run([sys.executable, "targets_img/make_photo_gallery.py", "--n", str(a.n)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    import glob
    gallery = {os.path.basename(p).split("_")[1]: os.path.relpath(p, ROOT).replace(os.sep, "/")
               for p in sorted(glob.glob(os.path.join(ROOT, "targets_img", f"photo_*_{a.n}.png")))}
    if a.who not in gallery:
        raise SystemExit(f"{a.who} is not in the gallery {sorted(gallery)}")
    target = gallery[a.who]
    seal = os.path.join(full, "seal.json")
    json.dump(dict(name=a.who, file=target, n=a.n, levels=2, gallery=gallery), open(seal, "w"), indent=1)
    log_path = os.path.join(ROOT, "runs", "photo_gate_server.log")
    logf = open(log_path, "w")
    cmd = [sys.executable, "-u", "xr_session.py", "--mode", "visual", "--choice", "bw", "--freq", "15",
           "--calib-blocks", "4", "--calib-on", "5", "--calib-off", "5", "--calib-style", "bw",
           "--calib-size", "1", "--target", target, "--grid-w", str(a.n), "--grid-h", str(a.n),
           "--spc", str(a.spc), "--certainty-max-passes", "2", "--rest-every", "48",
           "--session", session, "--http-port", str(HTTP), "--osc-port", str(OSC),
           "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
    procs = []
    t0 = time.time()
    try:
        driver = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        procs.append(driver)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{HTTP}/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        procs.append(subprocess.Popen(browser_cmd(f"http://127.0.0.1:{HTTP}/?auto=1",
                                                  os.path.join(full, "browser_profile")),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([sys.executable, "phantom_subject.py", "--session", session,
                                       "--host", "127.0.0.1", "--port", str(OSC), "--aux",
                                       "--tuning", "13.5:2", "--max-seconds", "1500"], cwd=ROOT,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        code = driver.wait(timeout=1500)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        logf.close()
    text = open(log_path, errors="replace").read()
    keep = [l for l in text.splitlines() if any(k in l for k in (
        "scan:", "certainty", "grey image", "SESSION", "FAILED", "Traceback", "threshold right", "r = "))]
    print("\n".join(l[:200] for l in keep[-6:] if "websockets" not in l and "EOFError" not in l))
    ro = subprocess.run([sys.executable, "analysis/photo_readout.py", session, "--seal", seal],
                        cwd=ROOT, capture_output=True, text=True)
    print(ro.stdout[-1500:], ro.stderr[-800:])
    ok = "SESSION DONE" in text or "grey image" in text
    right = "-> RIGHT" in ro.stdout
    print(f"PHOTO GATE {'PASS' if (ok and right) else 'FAIL'}  exit {code}  {time.time() - t0:.0f} s")
    sys.exit(0 if (ok and right) else 1)


if __name__ == "__main__":
    main()
