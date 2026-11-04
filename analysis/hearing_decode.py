"""Decode a hearing_session.py recording: the auditory steady-state response (ASSR) at each
tag rate, block by block, and an EEG threshold per band (Section V of the exam paper,
after Ali Sher's outline).

TP9 and TP10 are put on one clock (lock_in.eeg_clock), onto a 256 Hz grid, band-passed 1-70 Hz,
notched at 60 Hz and averaged (only the ones in contact: robust sd under --max-sd uV). Each
block is cut into 4 s epochs from 2 s after onset; an epoch over 150 uV peak to peak is dropped.
4 s holds a whole number of cycles of every tag, so contiguous epochs stay in phase and their
complex spectra are averaged coherently. At each tag the F statistic is the tag bin's power over
the mean of the K bins within +-6 Hz (other tags, their neighbours and 60 Hz left out), p from
F(2, 2K). A band's EEG threshold is the lowest level with a significant response provided every
louder level is significant too. The silent baseline blocks are scored the same way at every
tag as a false-alarm check.

    python analysis/hearing_decode.py runs/hearing1 [--quick]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from scipy.stats import f as fdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from lock_in import eeg_clock  # noqa: E402

FS = 256.0
EPOCH = 4.0
TAGS = {500: 37.0, 1000: 39.0, 2000: 41.0, 4000: 43.0}
GNG_TAG = 40.0


def grid_eeg(session, max_sd, channels=("TP9", "TP10")):
    """Samples kept on the headband's own clock (256 Hz, consecutive rows); arrival stamps are
    only used to find where a block starts. Resampling onto a stamp-derived grid smears the
    phase of a 40 Hz response between epochs; counting samples does not."""
    df = pd.read_csv(os.path.join(session, "eeg.csv"))
    t = df["time"].to_numpy(float)
    t_hat, fs_hat, _ = eeg_clock(t)
    t_hat = np.maximum.accumulate(t_hat)
    chans = [c for c in channels if c in df.columns]
    sos = butter(4, [1, 70], btype="band", fs=FS, output="sos")
    b, a = iirnotch(60.0, 30.0, FS)
    out, used = [], []
    for c in chans:
        x = df[c].to_numpy(float)
        x = filtfilt(b, a, sosfiltfilt(sos, x - np.median(x)))
        sd = 1.4826 * np.median(np.abs(x - np.median(x)))
        if sd <= max_sd:
            out.append(x)
            used.append(c)
    if not out:
        raise SystemExit(f"none of {channels} under {max_sd} uV robust sd: no contact")
    return t_hat, np.mean(out, 0), used


def block_spectrum(t_hat, x, t_on, t_off, pp_max=150.0):
    n = int(EPOCH * FS)
    i0 = np.searchsorted(t_hat, t_on + 2.0)
    i1 = np.searchsorted(t_hat, t_off)
    gaps = np.flatnonzero(np.diff(t_hat[i0:i1]) > 0.1) + i0 + 1     # stream gaps: restart the epoch count
    starts, s = [], i0
    for g in list(gaps) + [i1]:
        while s + n <= g:
            starts.append(s)
            s += n
        s = g
    specs = []
    for s in starts:
        seg = x[s:s + n]
        if np.ptp(seg) > pp_max:
            continue
        specs.append(np.fft.rfft(seg - seg.mean()) / n)
    if not specs:
        return None, 0
    return np.mean(specs, 0), len(specs)


def f_test(spec, tag, all_tags):
    f = np.fft.rfftfreq(int(EPOCH * FS), 1 / FS)
    k = int(round(tag * EPOCH))
    excl = set()
    for t in list(all_tags) + [60.0]:
        kk = int(round(t * EPOCH))
        excl.update({kk - 1, kk, kk + 1})
    nb = [j for j in range(int(round((tag - 6) * EPOCH)), int(round((tag + 6) * EPOCH)) + 1) if j not in excl and 0 < j < len(f)]
    p_sig = np.abs(spec[k]) ** 2
    p_noise = np.mean(np.abs(spec[nb]) ** 2)
    F = p_sig / p_noise
    return float(F), float(fdist.sf(F, 2, 2 * len(nb))), float(np.sqrt(2) * np.abs(spec[k])), len(nb)


def threshold(rows):
    """Lowest significant level such that every louder level is significant."""
    rows = sorted(rows, key=lambda r: -r["level_db"])
    thr = None
    for r in rows:
        if r["p"] < 0.05:
            thr = r["level_db"]
        else:
            break
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--quick", action="store_true", help="go/no-go result only")
    ap.add_argument("--max-sd", type=float, default=60.0)
    ap.add_argument("--channels", default="TP9,TP10", help="e.g. AF7,AF8 when the ear sites lost contact")
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(a.session, "events.csv"))
    grid, x, used = grid_eeg(a.session, a.max_sd, tuple(a.channels.split(",")))
    res = dict(channels=used, gng=[], bands={str(b): [] for b in TAGS}, baseline=[])
    for _, e in ev[ev["stage"] == "gng"].iterrows():
        spec, n = block_spectrum(grid, x, e.t_on, e.t_off)
        if spec is None:
            continue
        F, p, amp, K = f_test(spec, GNG_TAG, [GNG_TAG])
        res["gng"].append(dict(level_db=int(e.level_db), F=round(F, 2), p=round(p, 4), amp_uv=round(amp, 3), epochs=n))
        print(f"go/no-go {int(e.level_db):3d} dB: 40 Hz F {F:6.2f} p {p:.4f} amp {amp:.3f} uV ({n} epochs)")
    if res["gng"]:
        loud = sorted(res["gng"], key=lambda r: -r["level_db"])[:2]
        res["go"] = all(r["p"] < 0.05 for r in loud)
        print("GO: the two loudest levels respond" if res["go"] else
              "NO-GO: the loudest levels do not respond; fix contact (moisten the ear pads) and repeat")
    if a.quick:
        return
    for _, e in ev[ev["stage"] == "bands"].iterrows():
        spec, n = block_spectrum(grid, x, e.t_on, e.t_off)
        if spec is None:
            continue
        for band, tag in TAGS.items():
            F, p, amp, K = f_test(spec, tag, TAGS.values())
            res["bands"][str(band)].append(dict(level_db=int(e.level_db), F=round(F, 2), p=round(p, 4), amp_uv=round(amp, 3), epochs=n))
    for _, e in ev[ev["stage"] == "baseline"].iterrows():
        spec, n = block_spectrum(grid, x, e.t_on, e.t_off)
        if spec is None:
            continue
        for tag in list(TAGS.values()) + [GNG_TAG]:
            F, p, amp, K = f_test(spec, tag, list(TAGS.values()) + [GNG_TAG])
            res["baseline"].append(dict(tag=tag, F=round(F, 2), p=round(p, 4), epochs=n))
    res["eeg_threshold_db"] = {b: threshold(r) for b, r in res["bands"].items()}
    aud_p = os.path.join(a.session, "audiogram.json")
    if os.path.exists(aud_p):
        aud = json.load(open(aud_p))
        res["behavioural_threshold_db"] = {b: aud.get(b, {}).get("threshold_db") for b in res["bands"]}
    for b, rows in res["bands"].items():
        print(f"band {b:>4} Hz (tag {TAGS[int(b)]:.0f} Hz): " + "  ".join(
            f"{r['level_db']}dB F{r['F']:.1f}{'*' if r['p'] < 0.05 else ''}" for r in sorted(rows, key=lambda r: -r["level_db"]))
            + f"  -> EEG threshold {res['eeg_threshold_db'][b]}"
            + (f", behavioural {res.get('behavioural_threshold_db', {}).get(b)}" if "behavioural_threshold_db" in res else ""))
    fa = sum(r["p"] < 0.05 for r in res["baseline"])
    print(f"silence: {fa} of {len(res['baseline'])} tag tests significant (false alarms)")
    json.dump(res, open(os.path.join(a.session, "hearing_results.json"), "w"), indent=1)
    print("wrote", os.path.join(a.session, "hearing_results.json"))


if __name__ == "__main__":
    main()
