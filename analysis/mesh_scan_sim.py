"""A 3D object scanned end to end on the synthetic subject.

The object (mesh3d.py) is unfolded into the scan grid; the synthetic subject
of bwb_palette_sim.py (5 channels, SSVEP at each code's frequency and phase,
per-visit timing wobble and gain, monochrome flicker) watches a calibration
and then the object's cells only, in the order the live session uses: one
serpentine pass over the object's cells, then the least certain cells,
--batch at a time, until every cell is --quality sure or has --cap visits
(--passes N instead: N fixed passes). Every step is decoded by bwb.py exactly
as live (bwb.decode_arrays on everything recorded so far), and the result is
painted back on the mesh.

--tint < 1 replaces the monochrome flicker with the older coloured one (each
colour mixed toward white, 0 = pure colour; response ~ sqrt(luminance), the
assumption of bwb_palette_sim.py), to compare the two.

--session DIR also writes the run to disk in the live session format (eeg.csv,
bwb_cal_log.csv and bwb_log.csv with Wi-Fi-like arrival delays on the page
clock, bwb_meta.json, target_bwb.npy) and re-decodes it with
`python bwb.py --session DIR`, the same path as a recorded headset session.

usage: python analysis/mesh_scan_sim.py mesh:die --out runs/sim_die
       python analysis/mesh_scan_sim.py mesh:globe --trials 5
       python analysis/mesh_scan_sim.py mesh:die --session runs/sim_die_session
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import adaptive                                          # noqa: E402
import bwb                                               # noqa: E402
import mesh3d                                            # noqa: E402
from bwb_palette_sim import FS, Subject, background, gain_of   # noqa: E402

REFRESH = 72.0


def scan(obj, colours, rng, jitter=0.005, scale=0.6, spc=4.0, reps=4, on_s=8.0,
         quality=0.99, batch=6, cap=6, passes=0, tint=1.0, log=print):
    """Simulate one session. Returns a dict with the decoded grids, the
    certainty timeline and everything needed to write it to disk."""
    names, _ = bwb.classes_of(colours)
    K = len(names)
    codes = bwb.palette_codes(len(colours), REFRESH)
    # response size per class: 1 for every colour when the flicker is white
    # (monochrome); dimmer colours respond less with a coloured flicker
    gain = [0.0] + [gain_of(c, tint) for c in colours]
    target = obj.target(colours)
    mask = obj.mask
    gh, gw = mask.shape
    P = int(mask.sum())
    order = np.concatenate([rng.permutation(K) for _ in range(reps)])
    N = int((len(order) * (on_s + 1.0) + P * cap * spc + 10) * FS)
    X = np.column_stack([background(rng, N) for _ in range(5)])
    t = np.arange(N) / FS
    subj = Subject(rng, scale, jitter)
    s = dict(t=1.0, blocks=[], visits=[], seps=[])
    for k in order:
        subj.add(X, int(s["t"] * FS), int(on_s * FS), codes[k - 1] if k else None, gain[k])
        s["blocks"].append(dict(cls=int(k), t0=s["t"], t1=s["t"] + on_s))
        s["t"] += on_s + 1.0

    def visit(cells):
        s["seps"].append(s["t"])
        for gx, gy in cells:
            k = target[gy, gx]
            subj.add(X, int(s["t"] * FS), int(spc * FS), codes[k - 1] if k else None, gain[k])
            s["visits"].append((gx, gy, s["t"], s["t"] + spc - 1 / REFRESH))
            s["t"] += spc

    def state():
        n_now = min(N, int((s["t"] + 1.0) * FS))
        cls, post, seen, cal, _ = bwb.decode_arrays(
            t[:n_now], X[:n_now], FS, codes, s["blocks"], s["visits"], spc, gw, gh, colours)
        n = np.zeros((gh, gw), int)
        for gx, gy, *_ in s["visits"]:
            n[gy, gx] += 1
        cert = np.where(mask, np.where(seen, post.max(axis=2), 0.0), 1.0)
        return cls, post, seen, cert, n, cal

    timeline = []
    if passes:
        visit(mesh3d.scan_cells(mask, passes))
        cls, post, seen, cert, n, cal = state()
    else:
        visit(mesh3d.scan_cells(mask))
        while True:
            cls, post, seen, cert, n, cal = state()
            right = int(((cls == target) & mask & seen).sum())
            low = int((cert < quality).sum())
            timeline.append(dict(visits=int(n.sum()), minutes=round(n.sum() * spc / 60, 2),
                                 right=right, min_certainty=round(float(cert.min()), 4), below=low))
            log(f"  {int(n.sum()):4d} visits ({n.sum() * spc / 60:5.1f} min): {right}/{P} right, "
                f"lowest certainty {cert.min():.1%}, {low} below {quality:.0%}")
            if low == 0:
                break
            cells = adaptive.pick(cert, np.where(mask, n, cap), batch, cap)
            if not cells:
                break
            visit(cells)
    right = int(((cls == target) & mask & seen).sum())
    return dict(cls=cls, post=post, seen=seen, cert=cert, n=n, target=target, codes=codes,
                blocks=s["blocks"], visits=s["visits"], seps=s["seps"], X=X, t=t,
                t_end=s["t"], right=right, P=P, cv=cal["cv_accuracy"], timeline=timeline,
                minutes=n.sum() * spc / 60, cal_minutes=len(order) * (on_s + 1) / 60,
                capped=bool((cert < quality).any()))


def write_session(d, obj, colours, run, spc, layout, spec, rng):
    """The run as a live session folder: page-clock frame logs whose PC
    arrival times carry Wi-Fi-like delays, as xr_session.py records them."""
    os.makedirs(d, exist_ok=True)
    n = int((run["t_end"] + 1.0) * FS)
    with open(os.path.join(d, "eeg.csv"), "w") as f:
        f.write("time,TP9,AF7,AF8,TP10,AUX\n")
        for i in range(n):
            f.write(f"{run['t'][i]:.6f}," + ",".join(f"{v:.4f}" for v in run["X"][i]) + "\n")
    head = "time,grid_x,grid_y,luminance,flicker_on,page_t,flicker_g,flicker_b,r,g,b,audio_hz\n"
    offset = 1000.0                                   # page clock = PC clock - offset

    def frames(f, gx, gy, t0, t1, cls):
        for pt in np.arange(t0, t1, 1 / REFRESH):
            arrive = pt + 0.004 + rng.exponential(0.012)
            f.write(f"{arrive:.6f},{gx},{gy},{1 if cls else 0},0,{pt - offset:.6f},0,0,{cls},0,0,0\n")

    with open(os.path.join(d, "bwb_cal_log.csv"), "w") as f:
        f.write(head)
        for b in run["blocks"]:
            frames(f, -1, -1, b["t0"], b["t1"], b["cls"])
    with open(os.path.join(d, "bwb_log.csv"), "w") as f:
        f.write(head)
        seps = list(run["seps"])
        for gx, gy, t0, t1 in run["visits"]:
            while seps and seps[0] <= t0:
                f.write(f"{seps[0] - 0.01:.6f},-1,-1,0,0,{seps.pop(0) - 0.01 - offset:.6f},0,0,0,0,0,0\n")
            frames(f, gx, gy, t0, t1 + 1e-6, run["target"][gy, gx])
    # block messages reach the PC late too; retime_blocks moves them back
    blocks = [dict(b, t0=b["t0"] + 0.02, t1=b["t1"] + 0.02) for b in run["blocks"]]
    with open(os.path.join(d, "bwb_meta.json"), "w") as f:
        json.dump(dict(codes=run["codes"], blocks=blocks, spc=spc, colours=colours,
                       show=bwb.show_of(colours), target=spec, mesh_layout=layout), f, indent=1)
    np.save(os.path.join(d, "target_bwb.npy"), run["target"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", nargs="?", default="mesh:die")
    ap.add_argument("--layout", default="atlas", choices=["atlas", "net"])
    ap.add_argument("--colours", default="", help="palette (default: the object's own)")
    ap.add_argument("--jitter-ms", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--tint", type=float, default=1.0,
                    help="1 = monochrome (live); < 1 = coloured flicker mixed toward white "
                         "(0 = pure colour), response ~ sqrt(luminance), as bwb_palette_sim")
    ap.add_argument("--spc", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--on", type=float, default=8.0)
    ap.add_argument("--quality", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--passes", type=int, default=0, help="fixed passes instead of the certainty rule")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="", help="write the 3D result of the first trial here")
    ap.add_argument("--session", default="", help="write the first trial as a live session "
                    "folder and re-decode it with bwb.py --session")
    a = ap.parse_args()
    spec = a.object if mesh3d.is_mesh(a.object) else "mesh:" + a.object
    user = [c.strip() for c in a.colours.split(",") if c.strip()] or None
    obj = mesh3d.from_spec(spec, a.layout, user)
    colours = user or obj.colours
    print(f"{obj.name}: {len(obj.cells)} cells, {obj.grid_w} x {obj.grid_h} grid ({obj.layout}), "
          f"{len(colours)} colours + black ({', '.join(colours)}) | jitter {a.jitter_ms:g} ms | "
          f"{a.spc:g} s a visit | tint {a.tint:g}")
    rng = np.random.default_rng(a.seed)
    runs = []
    for k in range(a.trials):
        t0 = time.time()
        print(f"trial {k + 1}:")
        r = scan(obj, colours, rng, a.jitter_ms / 1000, a.scale, a.spc, a.reps, a.on, a.quality,
                 a.batch, a.cap, a.passes, a.tint, log=print if k == 0 or a.trials <= 3 else (lambda *x: None))
        first = r["timeline"][0]["right"] if r["timeline"] else r["right"]
        print(f"  -> {r['right']}/{r['P']} cells right after {r['minutes']:.1f} min of scanning "
              f"(+{r['cal_minutes']:.1f} min calibration; first pass {first}/{r['P']}; "
              f"calibration CV {r['cv']:.0%}) [{time.time() - t0:.0f} s]")
        runs.append(r)
        if k == 0 and a.out:
            os.makedirs(a.out, exist_ok=True)
            summ = f"{r['right']} of {r['P']} cells right (simulated subject)"
            mesh3d.export_session(a.out, obj, r["target"], r["cls"], r["seen"], r["post"], colours, summ)
            print(f"  wrote {a.out}/mesh_comparison.png, mesh_viewer.html, mesh_decoded.obj")
        if k == 0 and a.session:
            write_session(a.session, obj, colours, r, a.spc, a.layout, spec, rng)
            print(f"  wrote session {a.session}; re-decoding with bwb.py --session:")
            subprocess.run([sys.executable, os.path.join(ROOT, "bwb.py"), "--session", a.session],
                           check=True)
    if a.trials > 1:
        right = np.array([r["right"] for r in runs])
        mins = np.array([r["minutes"] for r in runs])
        P = runs[0]["P"]
        print(f"{a.trials} trials: {right.mean():.1f}/{P} right on average, whole object right in "
              f"{np.mean(right == P):.0%}; scanning median {np.median(mins):.1f} min "
              f"(max {mins.max():.1f})")


if __name__ == "__main__":
    main()
