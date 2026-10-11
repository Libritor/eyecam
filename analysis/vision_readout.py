"""The acuity read-out: which way does the E point, and is the ring closed
(O) or missing its bottom (U)? Decided from the EEG picture alone
(reconstruction_grid.npy), by correlation with the optotype templates, on
the raw grid and on the Otsu-thresholded grid; then, if a seal is given,
scored against it. Layouts: the 12 x 7 pair of targets_img/make_ou_targets.py
(E in columns 1-5, ring in 7-11, rows 1-5) and a single 12 x 12 double-size
optotype (make_vision_targets.big / make_ou_targets.big).

    python analysis/vision_readout.py runs/<session> [--seal targets_img/vision_today_seal.json]

Writes <session>/vision_readout.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "targets_img"))
from make_vision_targets import E_RIGHT, C_RIGHT, rotations  # noqa: E402
from make_ou_targets import O_RING, U_RING, E_NAMES, E_WIN, RING_WIN  # noqa: E402
import smoothcolor  # noqa: E402

E_TEMPLATES = dict(zip(E_NAMES, rotations(E_RIGHT)))
RING_TEMPLATES = {"o": O_RING, "u": U_RING,
                  "gap down": rotations(C_RIGHT)[1], "gap right": C_RIGHT,
                  "gap left": rotations(C_RIGHT)[2], "gap up": rotations(C_RIGHT)[3]}


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def decide(win, templates):
    """{name: r} on the window, best name, margin to the runner-up."""
    rs = {k: corr(win, t) for k, t in templates.items()}
    order = sorted(rs, key=lambda k: -np.nan_to_num(rs[k], nan=-9))
    best = order[0]
    margin = rs[best] - rs[order[1]] if len(order) > 1 else float("nan")
    return rs, best, margin


def windows(g):
    """(E window 5x5, ring window 5x5, layout name); None for a missing one."""
    h, w = g.shape
    if (h, w) == (7, 12):
        return g[E_WIN], g[RING_WIN], "pair 12x7"
    if (h, w) == (12, 12):
        inner = g[1:11, 1:11]
        small = inner.reshape(5, 2, 5, 2).mean(axis=(1, 3))   # 2x2 blocks -> 5x5
        return small, small, "single 12x12 (the one window is tested as E and as ring)"
    raise SystemExit(f"unknown layout {w}x{h}: expected 12x7 (pair) or 12x12 (single)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session")
    ap.add_argument("--seal", default=None, help="json with the drawn answer (e, ring)")
    a = ap.parse_args()
    g = np.load(os.path.join(a.session, "reconstruction_grid.npy")).astype(float)
    g = np.nan_to_num(g, nan=np.nanmin(g))
    thr = smoothcolor.otsu_log(g)
    snapped = (g > thr).astype(float)
    out = dict(session=a.session, layout=None, otsu=thr)
    e_raw, r_raw, layout = windows(g)
    e_snap, r_snap, _ = windows(snapped)
    out["layout"] = layout
    print(f"{a.session}: {layout}")
    res_e = {}
    for tag, win in (("raw", e_raw), ("snapped", e_snap)):
        rs, best, margin = decide(win, E_TEMPLATES)
        res_e[tag] = dict(r=rs, best=best, margin=margin)
        print(f"  E ({tag:7s}): " + "  ".join(f"{k} {v:+.2f}" for k, v in rs.items())
              + f"  -> {best} (margin {margin:+.2f})")
    res_r = {}
    for tag, win in (("raw", r_raw), ("snapped", r_snap)):
        rs, best, margin = decide(win, RING_TEMPLATES)
        ou = "o" if rs["o"] >= rs["u"] else "u"
        res_r[tag] = dict(r=rs, best=best, margin=margin, o_or_u=ou,
                          ou_margin=abs(rs["o"] - rs["u"]))
        print(f"  ring ({tag:7s}): " + "  ".join(f"{k} {v:+.2f}" for k, v in rs.items())
              + f"  -> {best}; O/U: {ou.upper()} (margin {abs(rs['o'] - rs['u']):+.2f})")
    out["E"] = res_e
    out["ring"] = res_r
    # the picture as a whole against what was shown, for the record
    tp = os.path.join(a.session, "target.npy")
    if os.path.exists(tp):
        t = np.load(tp).astype(float)
        if t.shape == g.shape:
            out["r_picture"] = corr(g, t)
            print(f"  whole picture against the target shown: r {out['r_picture']:+.2f}")
    decision = dict(e=res_e["raw"]["best"], ring=res_r["raw"]["o_or_u"])
    out["decision"] = decision
    print(f"  DECISION (raw grid): E points {decision['e']}, ring is {decision['ring'].upper()}")
    if a.seal and os.path.exists(a.seal):
        seal = json.load(open(a.seal))
        out["seal"] = seal
        out["e_right"] = decision["e"] == seal.get("e")
        out["ring_right"] = decision["ring"] == seal.get("ring")
        print(f"  SEAL: E {seal.get('e')}, ring {str(seal.get('ring')).upper()} -> "
              f"E {'RIGHT' if out['e_right'] else 'WRONG'}, O/U {'RIGHT' if out['ring_right'] else 'WRONG'}")
    json.dump(out, open(os.path.join(a.session, "vision_readout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
