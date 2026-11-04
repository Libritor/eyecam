"""A synthetic listener for hearing_decode.py: writes eeg.csv + events.csv for a full session
whose ASSR at each tag grows from zero at a known threshold, under dry-electrode-like noise
(1/f plus white, ~20 uV rms), stamped in bursts the way MuseLog's OSC packets arrive.

    python analysis/hearing_phantom.py runs/hearing_phantom [--thr 500:30,1000:25,2000:40,4000:35]
    python analysis/hearing_decode.py runs/hearing_phantom
"""
import argparse
import csv
import os

import numpy as np

FS = 256.0
TAGS = {500: 37.0, 1000: 39.0, 2000: 41.0, 4000: 43.0}
LEVELS = [60, 50, 40, 30, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--thr", default="500:30,1000:25,2000:40,4000:35")
    ap.add_argument("--gng-thr", type=float, default=30)
    ap.add_argument("--amp", type=float, default=0.6, help="ASSR rms at 30 dB above threshold, uV")
    ap.add_argument("--gng-seconds", type=float, default=120)
    ap.add_argument("--band-seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    thr = {int(k): float(v) for k, v in (p.split(":") for p in a.thr.split(","))}
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.session, exist_ok=True)
    blocks = [("baseline", 0, None, 30.0)]
    blocks += [("gng", k + 1, lv, a.gng_seconds) for k, lv in enumerate(LEVELS)]
    blocks += [("bands", k + 1, lv, a.band_seconds) for k, lv in enumerate(LEVELS)]
    blocks += [("baseline", 1, None, 10.0)]
    t0 = 1000.0
    total = sum(b[3] + 2 for b in blocks) + 5
    n = int(total * FS)
    t = np.arange(n) / FS
    w = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / FS)
    w[1:] /= np.sqrt(fr[1:])
    w[0] = 0
    pink = np.fft.irfft(w, n)
    noise = 18 * pink / pink.std() + 8 * rng.standard_normal(n)
    sig = np.zeros(n)
    ev = []
    cur = 2.0
    grow = lambda lv, th: a.amp * np.clip((lv - th) / 30.0, 0, None)  # noqa: E731
    for stage, k, lv, dur in blocks:
        i0, i1 = int(cur * FS), int((cur + dur) * FS)
        tt = t[i0:i1] - t[i0]
        if stage == "gng":
            sig[i0:i1] += np.sqrt(2) * grow(lv, a.gng_thr) * np.sin(2 * np.pi * 40 * tt + 1.0)
        if stage == "bands":
            for b, tag in TAGS.items():
                sig[i0:i1] += np.sqrt(2) * grow(lv, thr[b]) * np.sin(2 * np.pi * tag * tt + 0.3 * b / 500)
        ev.append([stage, k, "" if lv is None else lv, "", f"{t0 + cur + 0.08:.6f}", f"{t0 + cur + dur + 0.08:.6f}", "", ""])
        cur += dur + 2.0
    x = noise + sig
    # OSC arrival: packets of 12 samples stamped on arrival with jitter
    stamps = t0 + t.copy()
    for s in range(0, n, 12):
        stamps[s:s + 12] = t0 + t[min(s + 11, n - 1)] + abs(rng.normal(0.004, 0.003))
    stamps = np.maximum.accumulate(stamps + np.arange(n) * 1e-6)
    with open(os.path.join(a.session, "eeg.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time", "TP9", "AF7", "AF8", "TP10"])
        for i in range(n):
            v = x[i]
            wr.writerow([f"{stamps[i]:.6f}", f"{v + 800:.3f}", f"{rng.normal(800, 20):.3f}", f"{rng.normal(800, 20):.3f}",
                         f"{v + 0.3 * rng.standard_normal() * 10 + 800:.3f}"])
    with open(os.path.join(a.session, "events.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stage", "block", "level_db", "freq_hz", "t_on", "t_off", "kind", "note"])
        wr.writerows(ev)
    print(f"wrote {a.session}: {total / 60:.1f} min, thresholds {thr}, go/no-go threshold {a.gng_thr}")


if __name__ == "__main__":
    main()
