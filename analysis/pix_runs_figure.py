"""Figure + numbers for the pixel-font 'NO' runs: per pass and combined,
grey and black/white. usage: python analysis/pix_runs_figure.py runs/a runs/b"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import reconstruct as R          # noqa: E402
from decoder_bakeoff import visits, otsu, auc  # noqa: E402

F0 = 12.0


def analyse(session, shift_s=0.0):
    vis, names, fs = visits(session)
    tgt = np.load(os.path.join(session, "target.npy"))
    cal = json.load(open(os.path.join(session, "calibration.json")))
    w = np.array([float(cal["weights"].get(n, 0.0)) for n in names])
    w = w / w.sum()
    # score at the flicker frequency this session actually delivered
    f0 = float((cal.get("line") or {}).get("delivered") or F0)
    seen = {}
    passes = []
    # a visit cut short (pause, rest break, end of run) is not a measurement:
    # keep visits of at least 60 % of the typical dwell
    typical = np.median([len(segs[0]) for gx, gy, segs in vis if gx >= 0])
    for gx, gy, segs in vis:
        if gx < 0 or len(segs[0]) < 0.6 * typical:
            continue
        sc = np.array([R.paper_score(x, fs, f0) for x in segs])
        v = float(np.nansum(np.nan_to_num(sc) * w))
        k = seen.get((gx, gy), 0)
        seen[(gx, gy)] = k + 1
        while len(passes) <= k:
            passes.append(np.full(tgt.shape, np.nan))
        passes[k][gy, gx] = v
    return tgt, passes


def stats(g, tgt):
    m = np.isfinite(g)
    v, t = g[m], tgt[m]
    thr = otsu(v)
    return dict(r=float(np.corrcoef(v, t)[0, 1]), auc=auc(v, t),
                acc=float(np.mean((v > thr) == (t > 0.5))), thr=thr,
                wrong=int(np.sum((v > thr) != (t > 0.5))), n=int(m.sum()))


def main(sessions):
    rows = []
    for s in sessions:
        tgt, passes = analyse(s)
        name = os.path.basename(s)
        print(name, "passes:", len(passes))
        cum = []
        for k in range(len(passes)):
            st = stats(passes[k], tgt)
            c = np.nanmean(np.stack(passes[:k + 1]), axis=0)
            sc = stats(c, tgt)
            cum.append((c, sc))
            print(f"   pass {k + 1} alone: r={st['r']:.2f} AUC={st['auc']:.2f} right {st['acc'] * 100:.0f}%"
                  f"   | passes 1-{k + 1} averaged: r={sc['r']:.2f} AUC={sc['auc']:.2f} "
                  f"right {sc['acc'] * 100:.0f}% ({sc['wrong']} of {sc['n']} wrong)")
        rows.append((name, tgt, passes, cum))

    ncol = 2 + max(len(r[3]) for r in rows) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.9 * ncol, 2.3 * len(rows)),
                             squeeze=False)

    def norm(g):
        lo, hi = np.nanpercentile(g, [5, 95])
        return np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    labels = {"vr_pix_big": "large square", "vr_pix_small": "paper-sized square"}
    for i, (name, tgt, passes, cum) in enumerate(rows):
        panels = [("shown to the eye", tgt)]
        for k, (c, sc) in enumerate(cum):
            ttl = "pass 1" if k == 0 else f"passes 1-{k + 1} averaged"
            panels.append((f"{ttl}\nr = {sc['r']:.2f}", norm(c)))
        c, sc = cum[-1]
        panels.append((f"smoothed for display\n(bicubic, as in the paper)", norm(c)))
        panels.append((f"black / white\n{sc['n'] - sc['wrong']} of {sc['n']} positions right",
                       (c > sc["thr"]).astype(float)))
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]), ax.set_yticks([])
            if j >= len(panels):
                ax.axis("off")
                continue
            interp = "bicubic" if panels[j][0].startswith("smoothed") else "nearest"
            ax.imshow(panels[j][1], cmap="gray", vmin=0, vmax=1, interpolation=interp)
            ax.set_title(panels[j][0], fontsize=9)
            if j == 0:
                ax.set_ylabel(labels.get(name, name), fontsize=10)
    fig.suptitle("\"NO\" read from the brain: 12 Hz response at each of 45 gaze positions, "
                 "8 s per position per pass", fontsize=11)
    fig.tight_layout()
    out = os.path.join("runs", "figures", "fig11_pix_NO.png")
    fig.savefig(out, dpi=140)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
