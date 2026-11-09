"""Mind to Matter gate: the whole mode, live, on a phantom subject and a headless browser.

Runs xr_session.py --mode matter on media/test_creature.png (a bright creature on a dark
floor), the phantom at Oz following the logged luminance, and checks:
  1. the frame was captured and the creature found (find.json: >= 6 squares, a mask area
     between 5 and 60 % of the frame)
  2. the scan ran over exactly those squares and the brain's grid correlates with the frame's
     squares: r >= 0.6, above the shifted-EEG null
  3. the prints exist and are closed meshes (brain_relief, object_seen, base + mind layer)
  4. xr_session.json says ok

    python matter_gate.py                # ~4 min
    python matter_gate.py --verdict runs/matter_gate
"""
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

HTTP, OSC = 8791, 5011
R_MIN = 0.6


def verdict(full, code, t0=None):
    checks = {}
    try:
        with open(os.path.join(full, "xr_session.json")) as f:
            res = json.load(f)
    except OSError:
        res = {}
    checks["driver exit 0 and session ok"] = code == 0 and bool(res.get("ok"))
    fj = os.path.join(full, "find.json")
    find = json.load(open(fj)) if os.path.exists(fj) else {}
    checks["frame captured"] = os.path.exists(os.path.join(full, "frame.jpg"))
    checks["thing found: >= 6 squares"] = len(find.get("cells", [])) >= 6
    checks["thing found: mask 5-60 % of the frame"] = 0.05 <= float(find.get("mask_area", 0)) <= 0.6
    mt = res.get("matter", {})
    r = mt.get("r")
    nulls = mt.get("null_r") or []
    checks[f"brain grid vs frame squares r >= {R_MIN}"] = r is not None and r >= R_MIN
    checks["r above every shifted-EEG null"] = r is not None and (not nulls or r > max(nulls))
    ctl = find.get("controls") or {}
    gp = os.path.join(full, "reconstruction_grid.npy")
    if ctl and os.path.exists(gp):
        import numpy as np
        g = np.nan_to_num(np.load(gp))
        checks["white control scores above the black control"] = float(g[ctl["white"][1], ctl["white"][0]]) > float(g[ctl["black"][1], ctl["black"][0]])
    else:
        checks["controls present"] = False
    pr = os.path.join(full, "print")
    for name in ("brain_relief.stl", "object_seen.stl", "object_seen_base.stl", "object_seen_mind.stl"):
        rep = (mt.get("reports") or {}).get(name, {})
        checks[f"{name} written and closed"] = os.path.exists(os.path.join(pr, name)) and bool(rep.get("watertight", False))
    checks["preview.png written"] = os.path.exists(os.path.join(pr, "preview.png"))
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"found by {find.get('method')} ({find.get('mask_area', 0) * 100:.0f} % of the frame), grid "
          f"{find.get('gridW')}x{find.get('gridH')}, {len(find.get('cells', []))} squares; r = {r}; nulls = "
          + ", ".join(f"{v:.2f}" for v in nulls) + (f"  ({(time.time() - t0) / 60:.1f} min)" if t0 else ""))
    ok = all(checks.values())
    print("MATTER GATE", "PASSED" if ok else "FAILED")
    return ok


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verdict":
        full = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])
        with open(os.path.join(full, "xr_session.json")) as f:
            code = 0 if json.load(f).get("ok") else 1
        return 0 if verdict(full, code) else 1
    extra = os.environ.get("MATTER_GATE_EXTRA", "").split()
    media = os.environ.get("MATTER_GATE_MEDIA", os.path.join("media", "test_creature.png"))
    session = os.path.join("runs", "matter_gate" + os.environ.get("MATTER_GATE_SUFFIX", ""))
    full = os.path.join(ROOT, session)
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(os.path.join(full, "browser_profile"))
    cmd = ["--mode", "matter", "--matter-media", media, "--freq", "12", "--spc", "4", "--passes", "1",
           "--patch-side", "0.35", "--calib-blocks", "4", "--calib-on", "3", "--calib-off", "3",
           "--calib-style", "bw", "--calib-size", "1", "--matter-show", "2", "--matter-auto-pause", "2",
           "--matter-grid", "7x5", "--matter-cells", "16", "--recon-method", "line", "--matter-ring", "1", "--matter-ring-cells", "8",
           "--session", session, "--http-port", str(HTTP), "--osc-port", str(OSC),
           "--spectator", "", "--signal-timeout", "120", "--linger", "2"]
    procs = []
    t0 = time.time()
    try:
        driver = subprocess.Popen([sys.executable, "xr_session.py"] + cmd + extra, cwd=ROOT)
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
            [sys.executable, "phantom_subject.py", "--session", session, "--host", "127.0.0.1",
             "--port", str(OSC), "--aux", "--max-seconds", "1800"], cwd=ROOT))
        code = driver.wait(timeout=1800)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return 0 if verdict(full, code, t0) else 1


if __name__ == "__main__":
    sys.exit(main())
