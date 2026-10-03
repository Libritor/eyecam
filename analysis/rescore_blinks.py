"""Rescore recorded grey scans with and without blink / eye-closure masking.

Per session and arm it reports the correlation with the
target, the positions an automatic (Otsu) threshold gets right, the maximum r
of the circularly shifted EEG (chance; masked identically in the masked arm)
and, for the masked arm, the masked fraction of every position. Results are
also written to <session>/rescore_blinks.json. Arms:
  unmasked   as recorded
  masked     masked samples left out, positions over --mask-limit dropped (what a
             live --mask-blinks session does, where those positions are redone)
  keep-all   masked samples left out, nothing dropped: the fair rescore of a
             session recorded WITHOUT the live redo (a dropped position there has
             no second visit to replace it)

    python analysis/rescore_blinks.py runs/a runs/b
    python analysis/rescore_blinks.py runs/a --method upstream --blink-uv 80
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blinkmask          # noqa: E402
import nofigure           # noqa: E402
import reconstruct as R   # noqa: E402


def stim_freq(session):
    for name, path in (("calibration.json", ("line", "delivered")),
                       ("xr_session.json", ("stimFreqActual",))):
        p = os.path.join(session, name)
        if os.path.exists(p):
            v = json.load(open(p))
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if v:
                return float(v)
    return None


def arm(session, method, mask, opts, n_shifts):
    cal = os.path.join(session, "calibration.json")
    kw = dict(calibration=cal if os.path.exists(cal) else "", method=method,
              stim_freq=stim_freq(session), mask_blinks=mask, mask_opts=opts if mask else None)
    info = {}
    grid, r = R.run(session, save=False, info=info, **kw)
    target = np.load(os.path.join(session, "target.npy"))
    m = np.isfinite(grid)
    thr = nofigure.otsu(grid[m])
    right = int(np.sum((grid[m] > thr) == (target[m] > 0.5)))
    nulls = R.null_r(session, n_shifts=n_shifts, **kw)
    per_pos = {}
    for v in info.get("visits", []):
        per_pos.setdefault(f"{v['gx']},{v['gy']}", []).append(v["masked"])
    return dict(r=r, right=right, n=int(m.sum()), null_max=max(nulls) if nulls else None,
                nulls=nulls, mask=info.get("mask"),
                masked_per_position={k: round(float(np.mean(x)), 3) for k, x in per_pos.items()},
                dropped_visits=sum(not v["kept"] for v in info.get("visits", [])),
                visits=len(info.get("visits", [])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--method", choices=["paper", "upstream"], default="paper")
    ap.add_argument("--shifts", type=int, default=8, help="shifted-EEG nulls per arm")
    ap.add_argument("--blink-uv", type=float, default=None)
    ap.add_argument("--alpha-ratio", type=float, default=None)
    ap.add_argument("--closure-min-s", type=float, default=None)
    ap.add_argument("--margin-s", type=float, default=None)
    ap.add_argument("--mask-limit", dest="limit", type=float, default=None)
    a = ap.parse_args()
    opts = blinkmask.options(a)
    for s in a.sessions:
        if not os.path.exists(os.path.join(s, "cursor_log.csv")):
            print(f"{s}: no grey scan (cursor_log.csv) - skipped")
            continue
        res = {}
        try:
            for label, mask, o in (("unmasked", False, opts), ("masked", True, opts),
                                   ("keep-all", True, dict(opts, limit=1.0))):
                R._MASKS.clear()
                res[label] = arm(s, a.method, mask, o, a.shifts)
        except ValueError as exc:
            print(f"{s}: cannot score ({exc}) - is the EEG from the same run as the scan log?")
            continue
        print(f"\n== {s}  (method {a.method})")
        for label, x in res.items():
            nm = f"{x['null_max']:.2f}" if x["null_max"] is not None else "-"
            print(f"  {label:9s} r = {x['r']:+.3f}   threshold right {x['right']}/{x['n']}   "
                  f"shifted-EEG max r = {nm}"
                  + (f"   visits dropped {x['dropped_visits']}/{x['visits']}" if label == "masked" else "")
                  + ("   (masked samples left out, no position dropped)" if label == "keep-all" else ""))
        mk = res["masked"]["mask"] or {}
        if mk:
            print(f"  mask: {mk['masked_frac'] * 100:.1f}% of the recording (blinks "
                  f"{mk['blink_frac'] * 100:.1f}%, closed eyes {mk['closure_frac'] * 100:.1f}%)"
                  + (f"; {mk['note']}" if mk.get("note") else ""))
        pp = res["masked"]["masked_per_position"]
        worst = sorted(pp.items(), key=lambda kv: -kv[1])[:8]
        print("  most-masked positions (gx,gy: fraction): "
              + ", ".join(f"{k}: {v:.2f}" for k, v in worst))
        with open(os.path.join(s, "rescore_blinks.json"), "w") as f:
            json.dump(dict(method=a.method, options=opts, **res), f, indent=1)


if __name__ == "__main__":
    main()
