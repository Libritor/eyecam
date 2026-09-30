"""Fine-resolution ON-vs-OFF spectra for a G1 session: is there ANY peak at the
flicker frequency (and its harmonic) in any channel, plus alpha and mains sanity.

usage: python analysis/g1_fine.py runs/g1_oz2
"""
import io, os, sys
import numpy as np
from scipy.signal import welch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruct import infer_fs

sess = sys.argv[1] if len(sys.argv) > 1 else "runs/g1_oz2"
L = np.genfromtxt(os.path.join(sess, "calib_log.csv"), delimiter=",", skip_header=1)
t, lum, fl = L[:, 0], L[:, 3], L[:, 4]
edges = np.sum((fl[1:] == 1) & (fl[:-1] == 0))
on_time = np.sum(np.diff(t)[lum[:-1] > 0.5])
f0 = edges / max(on_time, 1e-9)
if f0 < 5: f0 = 15.0
blocks, cur, t0 = [], lum[0] > 0.5, t[0]
for i in range(1, len(t)):
    if (lum[i] > 0.5) != cur:
        blocks.append((cur, t0, t[i - 1])); cur, t0 = (lum[i] > 0.5), t[i]
blocks.append((cur, t0, t[-1]))
blocks = [(on, a, b) for on, a, b in blocks if b - a > 3]

hdr = io.open(os.path.join(sess, "eeg.csv"), encoding="utf-8", errors="replace").readline().strip().split(",")
E = np.genfromtxt(os.path.join(sess, "eeg.csv"), delimiter=",", skip_header=1)
et = E[:, 0]; fs = infer_fs(et); names = hdr[1:]
nper = int(round(fs * 4))  # 4 s Hann -> 0.25 Hz bins
def psd(seg):
    seg = seg - np.median(seg)
    # robust clip of gross artifacts (blinks/motion) before the PSD
    mad = np.median(np.abs(seg)) * 1.4826 + 1e-9
    seg = np.clip(seg, -6 * mad, 6 * mad)
    f, p = welch(seg, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, detrend="constant")
    return f, p
def band(f, p, lo, hi): return p[(f >= lo) & (f <= hi)].mean()
def at(f, p, x, w=0.13): return p[np.abs(f - x) <= w].mean()
def snr(f, p, x):  # peak vs mean of flanking bins +-(0.5..2 Hz)
    fl_ = ((np.abs(f - x) >= 0.5) & (np.abs(f - x) <= 2.0))
    return at(f, p, x) / p[fl_].mean()

print(f"flicker {f0:.2f} Hz, fs {fs:.2f}, {len(blocks)} blocks, bins 0.25 Hz")
print(f"{'ch':>5} | {'SNR@f ON':>9} {'SNR@f OFF':>9} | {'SNR@2f ON':>9} {'SNR@2f OFF':>10} | {'alpha ON/OFF':>12} | {'mains/eeg':>9} | {'hf uV':>6}")
for i, n in enumerate(names):
    Pon, Poff = [], []
    for on, a, b in blocks:
        sel = (et >= a + 1.0) & (et <= b)
        if sel.sum() < fs * 4: continue
        f, p = psd(E[sel, i + 1]); (Pon if on else Poff).append(p)
    Pon, Poff = np.mean(Pon, 0), np.mean(Poff, 0)
    mains = max(band(f, Pon, 59.5, 60.5), band(f, Pon, 49.5, 50.5)) / band(f, Pon, 20, 40)
    hf = np.sqrt(band(f, Pon, 14, 50) * 36)  # rms in 14-50 Hz band (approx)
    print(f"{n:>5} | {snr(f,Pon,f0):9.2f} {snr(f,Poff,f0):9.2f} | {snr(f,Pon,2*f0):9.2f} {snr(f,Poff,2*f0):10.2f} | "
          f"{band(f,Pon,8,12)/band(f,Poff,8,12):12.2f} | {mains:9.2f} | {hf:6.1f}")
    if n in ("AUX", "TP9", "TP10"):
        sel = (f >= f0 - 2) & (f <= f0 + 2)
        print("        f:   " + " ".join(f"{x:5.2f}" for x in f[sel]))
        print("        ON:  " + " ".join(f"{x:5.2f}" for x in Pon[sel] / Poff[sel].mean()))
        print("        OFF: " + " ".join(f"{x:5.2f}" for x in Poff[sel] / Poff[sel].mean()))
