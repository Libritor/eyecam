"""The step wedge read out: the 12 Hz response against the flicker brightness
(targets_img/wedge_12x4.png, 12 levels left to right, 4 rows). Says how
many grey levels this subject's eye separates, which decides --levels for
the photo gallery: the response is averaged down each column, the
column-to-column step is compared with the row-to-row scatter, and the
largest set of levels whose steps all clear the scatter is reported.

    python analysis/wedge_readout.py runs/<session>

Writes <session>/wedge_readout.json and <session>/wedge_readout.png.
"""
import argparse
import json
import os

import numpy as np


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def separable_levels(col_mean, col_sd, z=2.0):
    """Walk up the wedge, opening a new level whenever the response has risen
    by z standard errors over the last level opened."""
    levels = [0]
    for k in range(1, len(col_mean)):
        last = levels[-1]
        se = np.hypot(col_sd[k], col_sd[last]) / np.sqrt(2)
        if col_mean[k] - col_mean[last] > z * se:
            levels.append(k)
    return levels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    g = np.load(os.path.join(a.session, "reconstruction_grid.npy")).astype(float)
    t = np.load(os.path.join(a.session, "target.npy")).astype(float)
    assert g.shape == t.shape, (g.shape, t.shape)
    levels = t[0]                               # the drive level of each column
    col_mean = np.nanmean(g, axis=0)
    col_sd = np.nanstd(g, axis=0, ddof=1) if g.shape[0] > 1 else np.zeros(g.shape[1])
    rho = spearman(levels, col_mean)
    r = float(np.corrcoef(levels, col_mean)[0, 1])
    sep = separable_levels(col_mean, col_sd)
    print(f"{a.session}: {g.shape[1]} levels x {g.shape[0]} rows")
    print("  level     " + " ".join(f"{v:5.2f}" for v in levels))
    print("  response  " + " ".join(f"{v:5.2f}" for v in col_mean))
    print("  row sd    " + " ".join(f"{v:5.2f}" for v in col_sd))
    print(f"  Spearman rho {rho:+.2f}, Pearson r {r:+.2f} between drive level and response")
    print(f"  separable levels (each 2 se above the last): {len(sep)} at columns {sep} "
          f"-> use --levels {max(2, min(len(sep), 4))} for the photo gallery")
    out = dict(session=a.session, levels=levels.tolist(), response=col_mean.tolist(),
               row_sd=col_sd.tolist(), spearman=rho, pearson=r, separable=sep,
               suggested_levels=int(max(2, min(len(sep), 4))))
    json.dump(out, open(os.path.join(a.session, "wedge_readout.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(levels, col_mean, yerr=col_sd, fmt="o-", color="#2a78d6", ecolor="#9ab8de", capsize=3)
    for k in sep:
        ax.axvline(levels[k], color="#eb6834", lw=0.8, ls=":")
    ax.set_xlabel("flicker brightness (drive level)")
    ax.set_ylabel("response (whole-dwell ratio)")
    ax.set_title(f"step wedge: rho {rho:+.2f}, {len(sep)} separable levels")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.session, "wedge_readout.png"), dpi=110)
    print(f"  figure: {os.path.join(a.session, 'wedge_readout.png')}")


if __name__ == "__main__":
    main()
