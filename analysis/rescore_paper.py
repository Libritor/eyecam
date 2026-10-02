"""Re-score past grey scans with the paper's whole-dwell estimate and draw
target | old (line) | new (paper) side by side.

usage: python analysis/rescore_paper.py runs/vr_full2 runs/vr_fine1 ...
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reconstruct as R  # noqa: E402


def grid_for(session, method, shift_s=0.0):
    eeg_t, data, names, cur_t, gx, gy = R.load_session(session)
    cal = json.load(open(os.path.join(session, "calibration.json")))
    w = np.array([float(cal["weights"].get(n, 0.0)) for n in names])
    fs = R.infer_fs(eeg_t)
    if shift_s:
        data = np.roll(data, int(round(shift_s * fs)), axis=0)
    ok = cur_t <= eeg_t[-1] - 0.5          # positions the EEG actually covers
    tgt = np.load(os.path.join(session, "target.npy"))
    seen = np.zeros(tgt.shape, bool)
    seen[gy[ok], gx[ok]] = True
    g = R.reconstruct(eeg_t, data, cur_t[ok], gx[ok], gy[ok], fs=fs, weights=w,
                      stim_freq=12.0, method=method)
    full = np.full(tgt.shape, np.nan)
    full[:g.shape[0], :g.shape[1]] = g
    full[~seen] = np.nan
    r = float(np.corrcoef(full[seen], tgt[seen])[0, 1])
    return full, tgt, seen, r


def main(sessions):
    rows = []
    for s in sessions:
        row = {"run": os.path.basename(s)}
        for m in ("line", "welch", "paper"):
            g, tgt, seen, r = grid_for(s, m)
            span = None
            nulls = []
            eeg_t = R.load_session(s)[0]
            span = eeg_t[-1] - eeg_t[0]
            for k in range(12):
                sh = 31.0 + k * max(7.0, (span - 62.0) / 12)
                if sh > span - 10:
                    break
                nulls.append(grid_for(s, m, shift_s=sh)[3])
            row[m] = dict(grid=g, r=r, null_max=max(nulls), null_n=len(nulls))
        row.update(target=tgt, seen=seen)
        rows.append(row)
        print(f"{row['run']:10s} positions {int(seen.sum())}/{seen.size}  " +
              "  ".join(f"{m}: r={row[m]['r']:+.2f} (null max {row[m]['null_max']:+.2f})"
                        for m in ("line", "welch", "paper")))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)),
                             squeeze=False)
    for i, row in enumerate(rows):
        panels = [("shown to the eye", row["target"], None),
                  ("old scoring (line)", row["line"]["grid"], row["line"]),
                  ("paper scoring", row["paper"]["grid"], row["paper"])]
        for j, (name, g, info) in enumerate(panels):
            ax = axes[i][j]
            g = np.array(g, float)
            if info is not None:
                lo, hi = np.nanpercentile(g, [5, 95])
                g = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
            g = np.where(row["seen"], g, np.nan) if info is not None else g
            cm = plt.cm.gray.copy()
            cm.set_bad("#5a2a2a")
            ax.imshow(g, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            title = name if info is None else \
                f"{name}\nr = {info['r']:.2f} (controls up to {info['null_max']:.2f})"
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(row["run"], fontsize=9)
    fig.suptitle("Grey images from the brain, per position, with the paper's Eq. 1 "
                 "vertical blend (dark red = no EEG recorded)", fontsize=10)
    fig.tight_layout()
    out = os.path.join("runs", "figures", "fig9_paper_rescore.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
