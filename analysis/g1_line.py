"""Exact-frequency line detector for a G1 session.

Per block: one Hann periodogram over the whole block (9 s -> 0.11 Hz bins),
line power at f0 (+-0.15 Hz) over flanking power (0.5..2 Hz either side).
Blocks with gross motion (std > 4x the channel's median block std) are dropped.
ON vs OFF: Mann-Whitney per channel + exact max-over-channels permutation.

usage: python analysis/g1_line.py runs/g1_oz2 [f0]
"""
import io, os, sys, itertools
import numpy as np
from scipy.signal import periodogram
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruct import infer_fs

sess = sys.argv[1] if len(sys.argv) > 1 else "runs/g1_oz2"
L = np.genfromtxt(os.path.join(sess, "calib_log.csv"), delimiter=",", skip_header=1)
t, lum, fl = L[:, 0], L[:, 3], L[:, 4]
blocks, cur, t0 = [], lum[0] > 0.5, t[0]
for i in range(1, len(t)):
    if (lum[i] > 0.5) != cur:
        blocks.append((cur, t0, t[i - 1])); cur, t0 = (lum[i] > 0.5), t[i]
blocks.append((cur, t0, t[-1]))
blocks = [(on, a, b) for on, a, b in blocks if b - a > 3]
# delivered frequency = 1 / median rising-edge period on the PAGE clock (page_t,
# jitter-free), exactly as the driver's measured_flicker()
pt = L[:, 5]
ed = np.where((fl[1:] == 1) & (fl[:-1] == 0))[0] + 1
per = np.diff(pt[ed]); per = per[per < 0.5]
f0 = float(sys.argv[2]) if len(sys.argv) > 2 else (1.0 / float(np.median(per)) if len(per) > 10 else 15.0)
if not np.isfinite(f0) or f0 < 5: f0 = 15.0

hdr = io.open(os.path.join(sess, "eeg.csv"), encoding="utf-8", errors="replace").readline().strip().split(",")
E = np.genfromtxt(os.path.join(sess, "eeg.csv"), delimiter=",", skip_header=1)
et = E[:, 0]; fs = infer_fs(et); names = hdr[1:]

def line_snr(seg, f0, half=0.15, flank=(0.5, 2.0)):
    seg = seg - seg.mean()
    f, p = periodogram(seg, fs=fs, window="hann", detrend="constant")
    pk = p[np.abs(f - f0) <= half].mean()
    fk = p[(np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])].mean()
    return pk / fk

print(f"session {sess}: f0 = {f0:.3f} Hz (page-clock median edge period), fs {fs:.2f}, {len(blocks)} blocks")
kinds = [("on" if on else "off") for on, a, b in blocks]
stds = np.array([[np.std(E[(et >= a + 1) & (et <= b), i + 1]) for i in range(len(names))] for on, a, b in blocks])
keep = stds <= 4 * np.median(stds, 0)  # per channel, per block
S1 = np.full((len(blocks), len(names)), np.nan); S2 = S1.copy()
for k, (on, a, b) in enumerate(blocks):
    sel = (et >= a + 1.0) & (et <= b)
    for i in range(len(names)):
        if keep[k, i]:
            S1[k, i] = line_snr(E[sel, i + 1], f0); S2[k, i] = line_snr(E[sel, i + 1], 2 * f0)
print(f"{'blk':>3} {'kind':>4} " + " ".join(f"{n:>11s}" for n in names) + "    (SNR@f / SNR@2f; '-' = motion block dropped)")
for k in range(len(blocks)):
    print(f"{k:3d} {kinds[k]:>4} " + " ".join(("     -     " if not keep[k, i] else f"{S1[k,i]:5.2f}/{S2[k,i]:4.2f}") for i in range(len(names))))
on = np.array([kd == "on" for kd in kinds])
print(f"\n{'ch':>5} {'ON med':>7} {'OFF med':>7} {'MWU p':>7} | {'2f ON':>6} {'2f OFF':>6} {'MWU p':>7}")
stat = []
for i, n in enumerate(names):
    a1, b1 = S1[on, i], S1[~on, i]; a1, b1 = a1[np.isfinite(a1)], b1[np.isfinite(b1)]
    a2, b2 = S2[on, i], S2[~on, i]; a2, b2 = a2[np.isfinite(a2)], b2[np.isfinite(b2)]
    p1 = mannwhitneyu(a1, b1, alternative="greater").pvalue if len(a1) > 1 and len(b1) > 1 else np.nan
    p2 = mannwhitneyu(a2, b2, alternative="greater").pvalue if len(a2) > 1 and len(b2) > 1 else np.nan
    print(f"{n:>5} {np.median(a1):7.2f} {np.median(b1):7.2f} {p1:7.3f} | {np.median(a2):6.2f} {np.median(b2):6.2f} {p2:7.3f}")
# exact family-wise permutation on the combined log-SNR (f and 2f), max over channels
X = np.log(np.nan_to_num(S1, nan=1.0)) + np.log(np.nan_to_num(S2, nan=1.0))
def T(mask): return np.max(X[mask].mean(0) - X[~mask].mean(0))
obs = T(on); n_on = on.sum(); null = []
for comb in itertools.combinations(range(len(blocks)), n_on):
    m = np.zeros(len(blocks), bool); m[list(comb)] = True; null.append(T(m))
null = np.array(null); p_fw = np.mean(null >= obs - 1e-12)
best = names[int(np.argmax(X[on].mean(0) - X[~on].mean(0)))]
print(f"\nexact permutation (max-over-channels log-SNR f+2f): observed {obs:.3f}, best {best}, p_fw = {p_fw:.4f} (n_null {len(null)})")
