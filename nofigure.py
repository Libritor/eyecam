"""The four-panel figure of the published "NO" result, for any grey scan:

  shown to the eye | read from the EEG, one value per position (r) |
  same values, interpolated for display | automatic black/white threshold

The threshold is Otsu's (it never sees the picture); the count of positions
it gets right is the honest headline number for a two-level target.

    python nofigure.py --session runs/<session>      # redraw from a session
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def otsu(v):
    """Threshold maximising the between-class variance of the values."""
    v = np.sort(np.asarray(v, float)[np.isfinite(v)])
    best, thr = -1.0, float(np.median(v))
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        s = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if s > best:
            best, thr = s, (v[i - 1] + v[i]) / 2
    return thr


def figure(grid, target, out, title="", subtitle=""):
    """Draw the figure; returns dict(r, right, n, thr)."""
    g = np.asarray(grid, float)
    t = np.asarray(target, float)
    m = np.isfinite(g)
    r = float(np.corrcoef(g[m], t[m])[0, 1]) if m.sum() > 2 and t[m].std() > 0 else float("nan")
    thr = otsu(g[m])
    bw = (g > thr).astype(float)
    right = int(np.sum((bw[m] > 0.5) == (t[m] > 0.5)))
    lo, hi = np.nanpercentile(g, [5, 95])
    norm = np.clip((g - lo) / (hi - lo + 1e-12), 0, 1)
    panels = [("shown to the eye", t, "nearest"),
              (f"read from the EEG, one value per position\nr = {r:.2f}", norm, "nearest"),
              ("same values, interpolated for display", norm, "bicubic"),
              (f"automatic black/white threshold\n{right} of {int(m.sum())} positions right", bw, "nearest")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (ttl, img, interp) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation=interp)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title + ("\n" + subtitle if subtitle else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return dict(r=r, right=right, n=int(m.sum()), thr=float(thr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    g = np.load(os.path.join(a.session, "reconstruction_grid.npy"))
    t = np.load(os.path.join(a.session, "target.npy"))
    meta = {}
    p = os.path.join(a.session, "xr_session.json")
    if os.path.exists(p):
        meta = json.load(open(p))
    res = figure(g, t, os.path.join(a.session, "reconstruction_figure.png"),
                 title=f"{os.path.basename(a.session)}: method {meta.get('method', '?')}")
    print(f"r = {res['r']:.3f}, threshold gets {res['right']} of {res['n']} positions right "
          f"-> {os.path.join(a.session, 'reconstruction_figure.png')}")


if __name__ == "__main__":
    main()
