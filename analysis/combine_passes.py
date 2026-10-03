"""Combine grey-scan passes, within one session or across several sessions
of the same target, into one image.

    python analysis/combine_passes.py runs/a [runs/b ...] [--out fig.png]

Every pass of every session is scored per position with the paper's ratio
(channel weights from that session's own calibration); the image is the
mean over all passes. Prints the correlation with the target after each
added pass and writes a figure: shown / decoded / interpolated / black-white.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pix_runs_figure import analyse, stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    target, passes, labels = None, [], []
    for s in args.sessions:
        tgt, ps = analyse(s)
        if target is None:
            target = tgt
        elif tgt.shape != target.shape or not np.allclose(tgt, target):
            raise SystemExit(f"{s}: different target; cannot combine")
        for k, p in enumerate(ps):
            if np.isfinite(p).mean() < 0.5:
                continue        # an abandoned pass
            passes.append(p)
            labels.append(f"{os.path.basename(s)} pass {k + 1}")
    if not passes:
        raise SystemExit("no complete passes")

    binary = (target > 0.5).astype(float)
    for k in range(len(passes)):
        c = np.nanmean(np.stack(passes[:k + 1]), axis=0)
        st = stats(c, binary)
        print(f"+ {labels[k]:28s} -> r={st['r']:.2f}  AUC={st['auc']:.2f}  "
              f"{st['n'] - st['wrong']} of {st['n']} right")
    c = np.nanmean(np.stack(passes), axis=0)
    st = stats(c, binary)
    lo, hi = np.nanpercentile(c, [5, 95])
    n = np.clip((c - lo) / (hi - lo + 1e-12), 0, 1)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    panels = [("shown to the eye", target, "nearest"),
              (f"read from the EEG, {len(passes)} pass(es)\nr = {st['r']:.2f}", n, "nearest"),
              ("same values, interpolated", n, "bicubic"),
              (f"automatic black/white threshold\n{st['n'] - st['wrong']} of {st['n']} positions right",
               (c > st["thr"]).astype(float), "nearest")]
    for a, (ttl, im, itp) in zip(ax, panels):
        a.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation=itp)
        a.set_title(ttl, fontsize=10)
        a.set_xticks([]), a.set_yticks([])
    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.tight_layout()
    out = args.out or os.path.join(args.sessions[-1], "combined.png")
    fig.savefig(out, dpi=140)
    np.save(os.path.splitext(out)[0] + "_grid.npy", c)
    print("->", out)


if __name__ == "__main__":
    main()
