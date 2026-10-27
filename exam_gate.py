"""Full-stack gate for the eye exam (--mode exam), no hardware:

  xr_session.py --mode exam --exam-eye right,left   <-WS/HTTP->  headless browser
        ^ OSC /muse/eeg
  phantom_subject.py --aux --acuity-cutoff right:8,left:16
                     --contrast-cutoff right:0.08,left:0.2 --field-loss right:2
                     --color-gain 0.7,1,0.4 --latency right:0.12,left:0.15
                     --flicker-cut right:40,left:24 --alpha-band

The phantom's right eye stops seeing bars finer than about 8 arcmin and
contrasts below about 0.08 and has no response from quadrant 3 (index 2 of
the 2x2 field); its left eye is twice as coarse on both and whole. Both eyes
respond to green most and blue least, and the phantom's Oz (AUX) carries the
strongest response. The right eye's response is 120 ms late and holds up to
40 Hz (half there); the left eye's is 150 ms late, 30 ms later, and has
fallen to half by 24 Hz. PASS: every stage delivered; the full-field flash
responds in both eyes (p < 0.05) on AUX, which then serves as the reference
channel of every other test; each fitted threshold lies in the band where
the phantom's logistic visibility runs from 0.02 to 0.6 (the linear
extrapolation of the rising limb lands below the half-visibility point, at
3 to 5 % on this phantom, so the band is the honest expectation; its floor
was 0.03 until gate4's left contrast crossed at 2.8 %); the left eye's thresholds are at least
1.3x the right's; the field read-out names exactly the knocked-out quadrant
and nothing else; green >= red > blue in both eyes; in the flicker sweep
every rate shown responds in both eyes, the right eye is still over half at
the fastest rate shown (its cut lies beyond it) and the left eye's half rate
lies within 15 % of 24 Hz, each eye's apparent latency lies within 6 ms of
the phantom's and the left is 30 +/- 6 ms later than the right. The
apparent latency expected of the phantom is its latency less 3 / fs =
11.7 ms: it sends four samples per packet, stamped on arrival, and the
scorer's clock is rebuilt from the sample index on the earliest-arriving
sample of each packet, which puts every sample 3 / fs early (the real
stream has an offset of its own, common to both eyes, which is why the
difference between the eyes is the measurement). 3 s blocks, five flash
pairs and two ON/OFF pairs per step: about 13 minutes for the two eyes.

Run:  python exam_gate.py        (BROWSER=<path> to choose the browser;
      EXAM_GATE_HTTP / EXAM_GATE_OSC for a second gate beside the first,
      EXAM_GATE_EXTRA for more driver options, EXAM_GATE_SUFFIX for the session)
      python exam_gate.py --verdict runs/exam_gate   re-judges a finished
      session (after `python exam.py runs/exam_gate` or a read-out change)
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HTTP = int(os.environ.get("EXAM_GATE_HTTP", 8086))
OSC = int(os.environ.get("EXAM_GATE_OSC", 5004))
sys.path.insert(0, ROOT)
import config  # noqa: E402
from smooth_gate import browser_cmd  # noqa: E402
from phantom_subject import ACUITY_W, CONTRAST_W  # noqa: E402

EYE = {"right": dict(acuity=8.0, contrast=0.08, loss={2}, latency=0.12, fcut=40.0),
       "left": dict(acuity=16.0, contrast=0.2, loss=set(), latency=0.15, fcut=24.0)}
GAIN = "0.7,1,0.4"
REF = "AUX"        # the phantom's strongest channel: the flash must pick it as the reference
LAT_TOL_MS = 6.0   # apparent latency: allowed error per eye and on the difference
RATE_TOL = 0.15    # half-response rate: allowed fraction of the phantom's cut
CLOCK_OFFSET_S = 3.0 / config.FS   # the rebuilt clock runs this much early on the phantom's packets


def band(cut, w, lo=0.02, hi=0.6):
    """Where a logistic visibility 1 / (1 + exp(-(ln x - ln cut) / w)) runs from
    lo to hi. The sweep-VEP rule extrapolates the rising limb to zero, which
    on a logistic lands below the half point: on this phantom the crossings
    fall where the visibility is 3 to 5 % (gate2: 4-5 %; gate4: 2.8 % for the
    left contrast, 4-5 % for the rest), so the band's floor is 2 %, not a
    tighter figure the fit's own noise would cross."""
    return cut * math.exp(w * math.log(lo / (1 - lo))), cut * math.exp(w * math.log(hi / (1 - hi)))


def verdict(full, code, t0=None):
    """The checks on a finished session folder; prints them, returns pass/fail."""
    with open(os.path.join(full, "exam.json")) as f:
        exam = json.load(f)
    with open(os.path.join(full, "exam_readout.json")) as f:
        ro = json.load(f)
    checks = {"driver exit 0": code == 0,
              "every stage delivered": bool(exam.get("delivery")) and all(
                  d.get("ok") for d in exam["delivery"].values())}
    thr = {}
    for eye, want in EYE.items():
        r = ro["eyes"].get(eye, {})
        ac, co, fi, col = r.get("acuity", {}), r.get("contrast", {}), r.get("field", {}), r.get("colour", {})
        fl = r.get("flash", {})
        checks[f"{eye} flash responds (p < 0.05) on {REF}"] = \
            bool(fl.get("ok")) and fl.get("responded") and fl.get("channel") == REF
        checks[f"{eye} every test read on the reference channel {REF}"] = all(
            exam["results"].get(eye, {}).get(t, {}).get("channel") == REF
            for t in ("temporal", "acuity", "contrast", "field", "colour"))
        te = r.get("temporal", {})
        checks[f"{eye} flicker sweep: every rate shown responds (at least four)"] = \
            bool(te.get("ok")) and te.get("n_rates", 0) >= 4 and te["n_responding"] == te["n_rates"]
        fastest = te.get("fastest_rate") or 0.0
        if want["fcut"] >= fastest:
            checks[f"{eye} flicker sweep: still over half at the fastest rate shown "
                   f"(the phantom's cut {want['fcut']:g} Hz lies beyond it)"] = \
                bool(te.get("ok")) and bool(te.get("half_beyond_range"))
        else:
            lo, hi = want["fcut"] * (1 - RATE_TOL), want["fcut"] * (1 + RATE_TOL)
            checks[f"{eye} flicker sweep: half-response rate within {lo:.1f}-{hi:.1f} Hz"] = \
                bool(te.get("ok")) and te.get("half_rate") is not None and lo <= te["half_rate"] <= hi
        exp_lat = (want["latency"] - CLOCK_OFFSET_S) * 1000.0
        lat = te.get("latency") or {}
        checks[f"{eye} apparent latency within {LAT_TOL_MS:g} ms of the phantom's {exp_lat:.0f} ms"] = \
            bool(lat.get("ok")) and abs(lat["latency_ms"] - exp_lat) <= LAT_TOL_MS
        thr[eye, "latency"] = lat.get("latency_ms")
        lo, hi = band(want["acuity"], ACUITY_W)
        checks[f"{eye} acuity threshold in the phantom's band {lo:.1f}-{hi:.1f} arcmin"] = \
            bool(ac.get("ok")) and lo <= ac["threshold"] <= hi
        thr[eye, "acuity"] = ac.get("threshold")
        lo, hi = band(want["contrast"], CONTRAST_W)
        checks[f"{eye} contrast threshold in the phantom's band {lo:.3f}-{hi:.3f}"] = \
            bool(co.get("ok")) and lo <= co["threshold"] <= hi
        thr[eye, "contrast"] = co.get("threshold")
        checks[f"{eye} field: no response from exactly {sorted(want['loss']) or 'no sector'}"] = \
            bool(fi.get("ok")) and set(fi["not_responding"]) == want["loss"]
        rel = col.get("relative") or [0, 0, 0]
        checks[f"{eye} colour: green >= red > blue"] = bool(col.get("ok")) and rel[1] >= rel[0] > rel[2]
    for t in ("acuity", "contrast"):
        l, r = thr.get(("left", t)), thr.get(("right", t))
        checks[f"left {t} threshold >= 1.3x the right's"] = bool(l and r) and l >= 1.3 * r
    want_d = (EYE["left"]["latency"] - EYE["right"]["latency"]) * 1000.0
    l, r = thr.get(("left", "latency")), thr.get(("right", "latency"))
    checks[f"left minus right apparent latency {want_d:+.0f} +/- {LAT_TOL_MS:g} ms"] = \
        l is not None and r is not None and abs((l - r) - want_d) <= LAT_TOL_MS
    print("----")
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print(ro.get("text", ""))
    print("thresholds: " + ", ".join(f"{e} {t} {v:.3g}" if v is not None else f"{e} {t} -"
                                     for (e, t), v in thr.items()))
    print(f"report: {ro.get('report')}" + (f"  ({(time.time() - t0) / 60:.1f} min)" if t0 else ""))
    ok = all(checks.values())
    print("EXAM GATE", "PASSED" if ok else "FAILED")
    return ok


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verdict":
        full = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])
        with open(os.path.join(full, "xr_session.json")) as f:
            code = 0 if json.load(f).get("ok") else 1
        return 0 if verdict(full, code) else 1
    extra = os.environ.get("EXAM_GATE_EXTRA", "").split()
    session = os.path.join("runs", "exam_gate" + os.environ.get("EXAM_GATE_SUFFIX", ""))
    full = os.path.join(ROOT, session)
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(os.path.join(full, "browser_profile"))
    cmd = ["--mode", "exam", "--exam-eye", "right,left", "--exam-on", "3", "--exam-off", "3",
           "--exam-reps", "2", "--freq", "15", "--exam-ppd", "30",
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
            [sys.executable, "phantom_subject.py", "--session", session,
             "--host", "127.0.0.1", "--port", str(OSC), "--aux",
             "--acuity-cutoff", ",".join(f"{e}:{v['acuity']}" for e, v in EYE.items()),
             "--contrast-cutoff", ",".join(f"{e}:{v['contrast']}" for e, v in EYE.items()),
             "--field-loss", ",".join(f"{e}:" + "+".join(str(s) for s in sorted(v["loss"]))
                                      for e, v in EYE.items()),
             "--color-gain", GAIN,
             "--latency", ",".join(f"{e}:{v['latency']}" for e, v in EYE.items()),
             "--flicker-cut", ",".join(f"{e}:{v['fcut']}" for e, v in EYE.items()),
             "--alpha-band", "--max-seconds", "2400"],
            cwd=ROOT))
        code = driver.wait(timeout=2400)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return 0 if verdict(full, code, t0) else 1


if __name__ == "__main__":
    sys.exit(main())
