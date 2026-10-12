"""Where a planes session's cells came from, pass by pass: each plane decoded
from its first pass alone, its second pass alone, both passes, and both plus
the revisits (what the driver did), with the time-shifted null; and the
per-visit response of the lit cells against time on task, the SSVEP's own
fatigue meter.  Scoring as the driver's (reconstruct.run, method paper,
log-domain Otsu).

    python analysis/planes_passes.py runs/vr_planes10 [runs/vr_planes9 ...]

Writes <session>/<name>_passes.json and prints a table.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reconstruct, smoothcolor  # noqa: E402

NAMES = ["red", "green", "blue"]
NAMES_EEG = ["TP9", "AF7", "AF8", "TP10", "AUX"]


def visits_of(rows):
    """[(start_row, end_row, gx, gy)] in order; separator / rest rows end a visit."""
    out, start, key = [], None, None
    for i, r in enumerate(rows):
        k = (r[1], r[2])
        if k != key:
            if key is not None and key != ("-1", "-1"):
                out.append((start, i, int(key[0]), int(key[1])))
            start, key = i, k
    if key is not None and key != ("-1", "-1"):
        out.append((start, len(rows), int(key[0]), int(key[1])))
    return out


def write_rows(path, head, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def score(S, log, weights, cal, freq, target_plane, pure, shift=0.0, info=None):
    info = {} if info is None else info
    g, _ = reconstruct.run(S, calibration=cal, weights=weights, stim_freq=freq,
                           method="paper", vblend=False, cursor_log=log, save=False,
                           shift_s=shift, info=info)
    g = np.asarray(g, float)
    if g.shape != target_plane.shape:
        # a piece that visited only part of the grid (the revisits) comes back
        # as a smaller grid: put it on the full grid, unvisited cells NaN
        full = np.full(target_plane.shape, np.nan)
        full[:g.shape[0], :g.shape[1]] = g
        g = full
    seen = np.isfinite(g)
    if info.get("scores"):
        # only the cells this piece actually visited count (a revisit piece
        # leaves the rest of its grid at zero, which is not "dark")
        seen = np.zeros(g.shape, bool)
        for (gy, gx, _v) in info["scores"]:
            if 0 <= gy < g.shape[0] and 0 <= gx < g.shape[1]:
                seen[gy, gx] = True
        seen &= np.isfinite(g)
    on = np.zeros(g.shape, bool)
    on[seen] = g[seen] > smoothcolor.otsu_log(g[seen])
    right = int((on == target_plane)[pure & seen].sum())
    r = float(np.corrcoef(target_plane[seen].astype(float), g[seen])[0, 1]) if seen.sum() > 2 else float("nan")
    info["n_scored"] = int((pure & seen).sum())
    return right, r, info.get("scores", [])


def main(sessions):
    for S in sessions:
        name = os.path.basename(os.path.normpath(S))
        pcal = json.load(open(os.path.join(S, "planes_calib.json")))
        meta = json.load(open(os.path.join(S, "planes_meta.json")))
        target = np.load(os.path.join(S, "target_color.npy"))
        pure = ((target <= 0.25) | (target >= 0.75)).all(axis=2)
        gw, gh = target.shape[1], target.shape[0]
        cal = os.path.join(S, "calibration.json")
        out = {}
        print(f"\n{name}: passes {meta.get('passes')}, revisit {meta.get('revisit')}")
        print(f"{'plane':6s} {'piece':26s} {'right':>7s} {'r':>6s} {'null':>6s}  lit-cell score per visit")
        for k, pname in enumerate(NAMES):
            log = meta["logs"].get(pname)
            if not log:
                continue
            rows = list(csv.reader(open(os.path.join(S, log))))
            head, body = rows[0], rows[1:]
            vis = visits_of(body)
            n_cells = gw * gh
            passes = int(meta["passes"][k]) if isinstance(meta.get("passes"), list) else 1
            pieces = {}
            for p in range(passes):
                sel = vis[p * n_cells:(p + 1) * n_cells]
                pieces[f"pass {p + 1}"] = sel
            if passes > 1:
                pieces["both passes"] = vis[:passes * n_cells]
            if len(vis) > passes * n_cells:
                pieces["revisits alone"] = vis[passes * n_cells:]
                pieces["all (as run)"] = vis
            e = pcal.get(pname) or {}
            w = None
            if (e.get("passed") or e.get("use_weights")) and e.get("weights"):
                w = np.array([float(e["weights"].get(n, 0.0)) for n in NAMES_EEG])
                if w.sum() <= 0:
                    w = None
            freq = float(meta.get("freq_delivered", {}).get(pname, meta.get("freq", 12.0)))
            tplane = target[:, :, k] > 0.5
            out[pname] = {}
            for label, sel in pieces.items():
                if not sel:
                    continue
                sub = []
                for (s0, s1, gx, gy) in sel:
                    sub.extend(body[s0:s1])
                tmp = f"_pieces_{pname}_{label.replace(' ', '_')}.csv"
                write_rows(os.path.join(S, tmp), head, sub)
                inf = {}
                right, r, scores = score(S, tmp, w, "" if w is not None else cal, freq, tplane, pure, info=inf)
                n_scored = inf.get("n_scored", int(pure.sum()))
                nulls = [score(S, tmp, w, "" if w is not None else cal, freq, tplane, pure, shift=s)[1]
                         for s in (37.0, 91.0)]
                os.remove(os.path.join(S, tmp))
                lit = [v for (gy, gx, v) in scores if tplane[gy, gx]]
                dark = [v for (gy, gx, v) in scores if not tplane[gy, gx]]
                out[pname][label] = dict(right=right, of=n_scored, r=r, null=max(nulls),
                                         lit_median=float(np.median(lit)) if lit else None,
                                         dark_median=float(np.median(dark)) if dark else None,
                                         n_visits=len(sel))
                print(f"{pname:6s} {label:26s} {right:3d}/{n_scored:<3d} {r:+.2f} {max(nulls):+.2f}  "
                      f"lit {np.median(lit) if lit else float('nan'):.2f}  dark "
                      f"{np.median(dark) if dark else float('nan'):.2f}")
        json.dump(out, open(os.path.join(S, f"{name}_passes.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/vr_planes10"])
