"""What limits the image quality of a session? Signal, noise, or setup.

Reads <session>/eeg.csv and the session's flicker ON / OFF blocks (the grey
calibration in calibration.json, or the black/white/blue calibration in
bwb_meta.json, whichever overlaps the recorded EEG) and reports per channel:

  SSVEP     exact-line SNR at the flicker frequency during ON vs OFF blocks
            (the same line detector the calibration gate uses), next to the
            values of the two published legible-"NO" runs
  noise     broadband RMS, high-frequency noise, mains (60 Hz) pickup
  artifacts share of 1 s windows with large swings (movement, jaw, pops)
  contact   alpha peak (8-12 Hz) as a sign the electrode reads real EEG

and ends with a verdict: which of these is the bottleneck.

    python diagnose.py --session runs/xr1
"""

import argparse
import json
import os

import numpy as np
from scipy.signal import periodogram, welch

import reconstruct as R

# line SNR medians during ON blocks in the published runs that read "NO"
# (docs/results/vr_pix_big and vr_pix_small, calibration.json -> line.channels)
REFERENCE_ON = {"vr_pix_big": {"TP9": 18.2, "TP10": 1.6, "AUX": 191.9},
                "vr_pix_small": {"TP9": 53.9, "TP10": 14.9, "AUX": 18.7}}


def load(session):
    header, rows = R.read_csv(os.path.join(session, "eeg.csv"))
    t = np.array([float(r[0]) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    return t, X, header[1:]


def blocks_for(session, t):
    """[(label, f0, t0, t1)] for ON ('on') and OFF ('off') blocks inside the EEG."""
    out = []
    cal = os.path.join(session, "calibration.json")
    if os.path.exists(cal):
        c = json.load(open(cal))
        f0 = float(c.get("line", {}).get("delivered") or 12.0)
        for kind, a, b in c.get("blocks", []):
            if a >= t[0] and b <= t[-1]:
                out.append((kind, f0, a, b))
    meta = os.path.join(session, "bwb_meta.json")
    if not out and os.path.exists(meta):
        m = json.load(open(meta))
        fw, fb = m["codes"][0]["hz"], m["codes"][1]["hz"]
        for b in m["blocks"]:
            if b["t0"] >= t[0] and b["t1"] <= t[-1]:
                if b["cls"] == 1:
                    out.append(("on", fw, b["t0"], b["t1"]))
                elif b["cls"] == 2:
                    out.append(("on_blue", fb, b["t0"], b["t1"]))
                else:
                    out += [("off", fw, b["t0"], b["t1"]), ("off_blue", fb, b["t0"], b["t1"])]
    return out


def band(f, p, lo, hi):
    return p[(f >= lo) & (f <= hi)].sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    t, X, names = load(a.session)
    fs = R.infer_fs(t)
    gaps = np.diff(t)
    print(f"session {a.session}: {len(t)} samples, {fs:.1f} Hz, {(t[-1] - t[0]) / 60:.1f} min, "
          f"{int((gaps > 0.25).sum())} stream gaps > 0.25 s")
    libmuse = np.median(X) > 300
    print("units:", "libmuse scale (phone app / repo)" if libmuse else "microvolts centred on 0 (muselsl)")
    scale = 1.0 if not libmuse else 1.0     # both are microvolts; only the offset differs

    blocks = blocks_for(a.session, t)
    print(f"flicker blocks inside this recording: {len(blocks)}\n")

    rows = []
    for c, n in enumerate(names):
        x = (X[:, c] - np.median(X[:, c])) * scale
        f, p = welch(x, fs=fs, nperseg=int(fs * 2))
        rms = np.sqrt(band(f, p, 1, 40) * (f[1] - f[0]))
        mains = band(f, p, 58, 62) / max(band(f, p, 40, 56), 1e-12)
        alpha = band(f, p, 8, 12) / max(band(f, p, 4, 7) + band(f, p, 13, 17), 1e-12) * 8 / 5
        win = int(fs)
        p2p = np.array([np.ptp(x[i:i + win]) for i in range(0, len(x) - win, win)])
        art = float(np.mean(p2p > 200))
        line = {}
        for kind, f0, a0, b0 in blocks:
            i0, i1 = np.searchsorted(t, [a0 + 0.5, b0])
            line.setdefault(kind, []).append(R.line_snr(x[i0:i1], fs, f0))
        med = {k: float(np.nanmedian(v)) for k, v in line.items()}
        rows.append((n, rms, mains, alpha, art, med))

    print(f"{'ch':5s} {'RMS 1-40Hz':>10s} {'mains 60Hz':>10s} {'alpha':>6s} {'artifact s':>10s} "
          f"{'SSVEP ON':>9s} {'OFF':>6s} {'blue ON':>8s} {'OFF':>6s}   published ON (big / small)")
    for n, rms, mains, alpha, art, med in rows:
        ref = " / ".join(f"{REFERENCE_ON[r].get(n, float('nan')):.1f}" for r in REFERENCE_ON)
        print(f"{n:5s} {rms:8.1f}uV {mains:10.1f}x {alpha:6.2f} {art * 100:9.0f}% "
              f"{med.get('on', np.nan):9.1f} {med.get('off', np.nan):6.1f} "
              f"{med.get('on_blue', np.nan):8.1f} {med.get('off_blue', np.nan):6.1f}   {ref}")

    # ---- verdict
    print("\nverdict:")
    best = max(rows, key=lambda r: r[5].get("on", 0) / max(r[5].get("off", 1), 1e-9))
    on, off = best[5].get("on", np.nan), best[5].get("off", np.nan)
    ref_best = max(max(v.values()) for v in REFERENCE_ON.values())
    ref_low = min(max(v.values()) for v in REFERENCE_ON.values())
    if np.isfinite(on):
        print(f"  SSVEP: best channel {best[0]} line SNR {on:.1f} ON vs {off:.1f} OFF; the published runs "
              f"reached {ref_low:.0f}-{ref_best:.0f} on their best channel "
              f"({ref_low / max(on, 1e-9):.0f}-{ref_best / max(on, 1e-9):.0f}x stronger).")
    aux = [r for r in rows if r[0] == "AUX"]
    if not aux:
        print("  Oz: no AUX column - the auxiliary (Oz) electrode is not being recorded.")
    else:
        an = aux[0]
        ratio = an[5].get("on", np.nan) / max(an[5].get("off", np.nan), 1e-9)
        if not (ratio > 3):
            print(f"  Oz: the AUX channel does not respond to the flicker (ON/OFF {ratio:.1f}x; "
                  "published: 340x and 27x). It is unplugged, not on occipital scalp, or has no contact.")
        if an[3] < 1.3:
            print(f"  Oz: AUX shows no alpha peak ({an[3]:.2f}); an electrode on the occipital scalp normally does.")
    noisy = [r[0] for r in rows if r[4] > 0.15]
    if noisy:
        print(f"  artifacts: {', '.join(noisy)} have large swings in >15% of seconds (movement, jaw, loose contact).")
    hum = [r[0] for r in rows if r[2] > 10]
    if hum:
        print(f"  mains: {', '.join(hum)} pick up strong 60 Hz (a high-impedance electrode).")
    print("  The image can only be as clean as the ON/OFF contrast above: with an SSVEP this close to the "
          "noise, no scoring method (paper ratio, line detector, FBCCA, Kalman) recovers the letter.")


if __name__ == "__main__":
    main()
