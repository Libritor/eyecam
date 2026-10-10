"""The live black & white certainty rule (adaptive.bw_certainty), offline on
the synthetic subject of adaptive_passes.py: one full pass of the pixel-font
"NO" (9 x 5, 8 s visits), then --batch of the least certain squares at a time
until every square is >= --certainty or each has --max-passes visits.

--scale 0 is the no-signal control: the rule must not claim certainty there.

usage: python analysis/adaptive_bw_sim.py [--scale 0.6] [--trials 40]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import adaptive                       # noqa: E402
import targets                        # noqa: E402
from adaptive_passes import visit, SPC   # noqa: E402


def trial(rng, target, scale, jitter, q, batch, max_passes):
    lum = target.ravel()
    gh, gw = target.shape
    P = lum.size
    scores = {}

    def look(p):
        gy, gx = divmod(p, gw)
        scores.setdefault((gy, gx), []).append(visit(lum[p], rng, scale, jitter))

    for p in range(P):
        look(p)
    first = None
    while True:
        call, cert, n, repeats = adaptive.bw_certainty(scores, target.shape)
        if first is None:
            first = (cert.copy(), call.copy())
        ready = repeats >= min(batch, P)
        cells = adaptive.pick(cert, n, batch, max_passes)
        if (ready and cert.min() >= q) or not cells:
            break
        for gx, gy in cells:
            look(gy * gw + gx)
    right = call == (target > 0.5)
    return dict(visits=int(n.sum()), right=int(right.sum()), P=P,
                capped=not cells, cert_min=float(cert.min()), first=first,
                stop_cert=cert, stop_right=right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pix:NO")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--certainty", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-passes", type=int, default=6)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    target = targets.load_target(a.target)
    rng = np.random.default_rng(a.seed)
    R = [trial(rng, target, a.scale, a.jitter, a.certainty, a.batch, a.max_passes)
         for _ in range(a.trials)]
    P = R[0]["P"]
    m = np.array([r["visits"] for r in R]) * SPC / 60
    v = np.array([r["right"] for r in R])
    print(f"{a.target} {target.shape[1]}x{target.shape[0]}, scale {a.scale}, certainty "
          f"{a.certainty}, batch {a.batch}, {a.trials} runs, {SPC:g} s visits")
    print(f"  one pass = {P * SPC / 60:.1f} min; stopped after median {np.median(m):.1f} min, "
          f"90% by {np.quantile(m, .9):.1f}, max {m.max():.1f}; hit the cap "
          f"{np.mean([r['capped'] for r in R]):.0%}")
    print(f"  {v.mean():.1f}/{P} right, whole picture right {np.mean(v == P):.0%}")
    stopped = [r for r in R if not r["capped"]]
    if stopped:
        c = np.concatenate([r["stop_cert"].ravel() for r in stopped])
        ok = np.concatenate([r["stop_right"].ravel() for r in stopped])
        print(f"  at a certainty stop: squares claimed >= {a.certainty:.0%} were right "
              f"{ok[c >= a.certainty].mean():.1%} (n={int((c >= a.certainty).sum())})")


if __name__ == "__main__":
    main()
