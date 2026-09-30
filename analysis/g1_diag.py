"""G1 session diagnostic: delivery check, per-block contact, ON/OFF SSVEP per channel.

usage: python analysis/g1_diag.py runs/g1_oz2
"""
import io, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruct import infer_fs, ssvep_score

sess = sys.argv[1] if len(sys.argv) > 1 else "runs/g1_oz2"

# ---- calibration.json (the driver's verdict) ----
cj = os.path.join(sess, "calibration.json")
if os.path.exists(cj):
    c = json.load(open(cj))
    p = c.get("permutation", {})
    print(f"VERDICT gate={c.get('gate')} passed={c.get('passed')} best={c.get('best')} "
          f"p_fw={p.get('p_fw')} decidable={p.get('decidable')} n_null={p.get('n_null')}")
    pc = p.get("p_channel", {})
    if not isinstance(pc, dict):
        pc = dict(zip(c.get("names", []), pc)) if isinstance(pc, list) else {}
    for n, v in c.get("channels", {}).items():
        if isinstance(v, dict):
            print(f"  {n:5s} d'={v.get('dprime', float('nan')):6.2f} on/off={v.get('ratio', float('nan')):5.2f} "
                  f"p={pc.get(n, float('nan')) if isinstance(pc, dict) else float('nan')}")
else:
    print("no calibration.json yet")

# ---- stimulus delivery ----
lg = os.path.join(sess, "calib_log.csv")
L = np.genfromtxt(lg, delimiter=",", skip_header=1)
if L.ndim == 1: L = L[None, :]
t, lum, fl = L[:, 0], L[:, 3], L[:, 4]
span = t[-1] - t[0]
edges = np.sum((fl[1:] == 1) & (fl[:-1] == 0))
on_time = np.sum(np.diff(t)[lum[:-1] > 0.5])
print(f"DELIVERY rows={len(t)} span={span:.1f}s fps={len(t)/max(span,1e-9):.1f} "
      f"edges={edges} delivered={edges/max(on_time,1e-9):.2f} Hz")

# ---- blocks from the log ----
blocks = []
cur, t0 = lum[0] > 0.5, t[0]
for i in range(1, len(t)):
    on = lum[i] > 0.5
    if on != cur:
        blocks.append((cur, t0, t[i - 1])); cur, t0 = on, t[i]
blocks.append((cur, t0, t[-1]))
blocks = [(on, a, b) for on, a, b in blocks if b - a > 3]
f_hz = edges / max(on_time, 1e-9)
if f_hz < 5:
    print("  (no flicker edges logged: scoring at nominal 15 Hz; session is VOID)")
    f_hz = 15.0

# ---- EEG ----
hdr = io.open(os.path.join(sess, "eeg.csv"), encoding="utf-8", errors="replace").readline().strip().split(",")
E = np.genfromtxt(os.path.join(sess, "eeg.csv"), delimiter=",", skip_header=1)
et = E[:, 0]; fs = infer_fs(et)
names = hdr[1:]
print(f"EEG fs={fs:.2f} rows={len(et)} channels={names}")
print(f"{'blk':>3} {'kind':>4} {'dur':>5}", " ".join(f"{n:>13s}" for n in names))
scores = {n: {"on": [], "off": []} for n in names}
for k, (on, a, b) in enumerate(blocks):
    sel = (et >= a + 1.0) & (et <= b)
    row = []
    for i, n in enumerate(names):
        seg = E[sel, i + 1]
        s = ssvep_score(seg, fs, stim_freq=f_hz) if len(seg) > fs * 2 else float("nan")
        scores[n]["on" if on else "off"].append(s)
        row.append(f"{s:6.3f}/{np.std(seg):5.0f}")
    print(f"{k:3d} {'ON' if on else 'OFF':>4} {b-a:5.1f}", " ".join(f"{r:>13s}" for r in row))
print("      (score / segment std µV)")
print(f"{'ch':>5} {'meanON':>8} {'meanOFF':>8} {'ratio':>6} {'dprime':>7}")
for n in names:
    on, off = np.array(scores[n]["on"]), np.array(scores[n]["off"])
    on, off = on[np.isfinite(on)], off[np.isfinite(off)]
    if len(on) and len(off):
        sd = np.sqrt((on.var(ddof=1) + off.var(ddof=1)) / 2) if len(on) > 1 and len(off) > 1 else np.nan
        print(f"{n:>5} {on.mean():8.4f} {off.mean():8.4f} {on.mean()/off.mean():6.2f} {(on.mean()-off.mean())/sd if sd else np.nan:7.2f}")
