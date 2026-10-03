"""Gate for blink / eye-closure masking: simulate -> reconstruct -> verify.

Synthetic sessions of the pixel-font "NO" (8 s per position, 5 channels:
TP9 AF7 AF8 TP10 + Oz as AUX), built from simulate.py's model:
  clean   no blinks
  dirty   the same scan with 15 blinks/min (a frontal deflection, and no SSVEP
          while the lids are down) and 2 eye closures/min (2-4 s, no SSVEP,
          Oz alpha x6)
The masked arm runs like a live --mask-blinks session: a position more than
--mask-limit masked is shown again at the end of the scan (up to 3 rounds),
exactly what the live redo does. The unmasked arm gets no redos, as live.

PASS requires ALL of:
  1. clean, unmasked:            r >= 0.60 (the pipeline works at all)
  2. dirty, masked (+ redos):    r >= clean r - 0.05 (no worse than no blinks)
  3. dirty, masked, shifted EEG: max r <= 0.30 (masked nulls stay at chance)
and prints the dirty, unmasked r so the cost of NOT masking is visible.

Run:  python blink_gate.py
"""

import csv
import json
import os
import shutil
import sys

import numpy as np

import config
import reconstruct
import simulate
import targets

ROOT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]
SNR = [0.8, 0.2, 0.2, 0.8, 1.2]
SPC = 8.0
FS = config.FS
LIMIT = 0.35


def eye_events(dur, seed, blinks_per_min, closures_per_min):
    """Blink onsets and closure (start, length) on their own random streams,
    so appending redo visits never moves the earlier events."""
    rb, rc = np.random.default_rng(seed), np.random.default_rng(seed + 1000)
    blinks, t = [], 3.0
    while blinks_per_min:
        t += rb.exponential(60.0 / blinks_per_min)
        if t > dur:
            break
        blinks.append(t)
    closures, t = [], 5.0
    while closures_per_min:
        t += rc.exponential(60.0 / closures_per_min)
        k = rc.uniform(2.0, 4.0)
        if t > dur:
            break
        closures.append((t, k))
        t += k
    return blinks, closures


def build(out, visits, blinks_per_min, closures_per_min, seed=11):
    """Write eeg.csv / cursor_log.csv for a list of visits (gx, gy, lum); a
    (-1, -1) row before each redo marks it as a separate visit, as live."""
    target = targets.load_target("pix:NO")
    n_cell = int(round(SPC * FS))
    rows, lum = [], []
    for k, (gx, gy, lv, redo) in enumerate(visits):
        t0 = len(lum) / FS
        if redo:
            rows.append((t0, -1, -1, 0.0, 0))
        rows += [(t0 + i / FS, gx, gy, lv, 1 if lv > 0.05 else 0) for i in range(n_cell)]
        lum += [lv] * n_cell
    lum = np.array(lum)
    n = len(lum)
    tt = np.arange(n) / FS
    open_, blink, closed = np.ones(n), np.zeros(n), np.zeros(n)
    bl, cl = eye_events(n / FS, seed, blinks_per_min, closures_per_min)
    kb = int(0.3 * FS)
    for t in bl:
        i0 = int(t * FS)
        seg = slice(i0, min(n, i0 + kb))
        blink[seg] = np.sin(np.pi * np.arange(seg.stop - i0) / kb)
        open_[seg] = 0
    for t, k in cl:
        seg = slice(int(t * FS), min(n, int((t + k) * FS)))
        closed[seg] = 1
        open_[seg] = 0
    # SSVEP follows luminance x open eyes, with simulate.py's 0.3 s entrainment lag
    a = np.exp(-1.0 / (0.3 * FS))
    env = np.empty(n)
    acc = 0.0
    drive = lum * open_
    for i in range(n):
        acc = a * acc + (1 - a) * drive[i]
        env[i] = acc
    ssvep = env * (np.sin(2 * np.pi * config.STIM_FREQ_HZ * tt)
                   + 0.4 * np.sin(2 * np.pi * config.HARMONIC_HZ * tt))
    cols = []
    for c, name in enumerate(NAMES):
        # background per visit on its own seed: stable when redos are appended
        noise = np.concatenate([simulate.synth_eeg([(0, 0, 0, 0.0, 0)] * n_cell, FS, 0.0,
                                                   np.random.default_rng((seed, k, c)))
                                for k in range(len(visits))])
        alpha = 0.6 * np.sin(2 * np.pi * 10.0 * tt + c)
        if name == "AUX":
            noise = noise + 5.0 * closed * alpha             # eyes shut: Oz alpha x6
        gain = {"AF7": 150.0, "AF8": 150.0, "AUX": 5.0}.get(name, 40.0)
        cols.append(850.0 + 10.0 * (noise + SNR[c] * ssvep) + gain * blink)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eeg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + NAMES)
        for i in range(n):
            w.writerow([f"{tt[i]:.6f}"] + [f"{col[i]:.4f}" for col in cols])
    with open(os.path.join(out, "cursor_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "grid_x", "grid_y", "luminance", "flicker_on"])
        for r in rows:
            w.writerow([f"{r[0]:.6f}", r[1], r[2], f"{r[3]:.4f}", r[4]])
    np.save(os.path.join(out, "target.npy"), target)
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump({"weights": {"TP9": 0.25, "AF7": 0.0, "AF8": 0.0, "TP10": 0.25, "AUX": 0.5},
                   "blocks": []}, f)
    return float(1 - open_.mean())


def score(out, mask, nulls=False):
    kw = dict(calibration=os.path.join(out, "calibration.json"), method="paper",
              stim_freq=config.STIM_FREQ_HZ, mask_blinks=mask,
              mask_opts=dict(limit=LIMIT) if mask else None)
    info = {}
    reconstruct._MASKS.clear()
    _, r = reconstruct.run(out, save=False, info=info, **kw)
    rs = reconstruct.null_r(out, n_shifts=6, **kw) if nulls else []
    return r, rs, info


def main():
    target = targets.load_target("pix:NO")
    base = []
    for gy in range(target.shape[0]):
        xs = range(target.shape[1]) if gy % 2 == 0 else range(target.shape[1] - 1, -1, -1)
        base += [(gx, gy, float(target[gy, gx]), False) for gx in xs]
    clean, dirty, masked = (os.path.join(ROOT, "runs", d) for d in
                            ("gate_blink_clean", "gate_blink_dirty", "gate_blink_masked"))
    for d in (clean, dirty, masked):
        shutil.rmtree(d, ignore_errors=True)

    build(clean, base, 0, 0)
    shut = build(dirty, base, 15, 2)
    r_clean, _, _ = score(clean, False)
    r_dirty, _, _ = score(dirty, False)

    # masked arm: redo positions over the limit, like the live session
    visits, redos = list(base), 0
    for _ in range(3):
        build(masked, visits, 15, 2)
        _, _, info = score(masked, True)
        over = [v for v in info.get("visits", []) if not v["kept"]]
        if not over:
            break
        visits += [(v["gx"], v["gy"], float(target[v["gy"], v["gx"]]), True) for v in over]
        redos += len(over)
    r_masked, nulls, info = score(masked, True, nulls=True)
    m = info.get("mask", {})
    print(f"eyes shut {shut * 100:.0f}% of the scan; the mask covers "
          f"{m.get('masked_frac', 0) * 100:.0f}% (blinks {m.get('blink_frac', 0) * 100:.0f}%, "
          f"closures {m.get('closure_frac', 0) * 100:.0f}%); {redos} positions redone")
    print(f"clean, unmasked:          r = {r_clean:.3f}  (need >= 0.60)")
    print(f"blinks, unmasked:         r = {r_dirty:.3f}  (for reference)")
    print(f"blinks, masked + redos:   r = {r_masked:.3f}  (need >= {r_clean - 0.05:.3f})")
    print(f"blinks, masked, shifted:  max r = {max(nulls):.3f}  (need <= 0.30)")
    ok = r_clean >= 0.60 and r_masked >= r_clean - 0.05 and max(nulls) <= 0.30
    print("GATE PASS" if ok else "GATE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
