"""Per-position decoder bake-off on past grey scans (no vertical blend):
paper score, ACT chirplet fit, a learned classifier (leave-one-run-out) and
binary thresholding. Writes runs/figures/fig10_decoder_bakeoff.png.

usage: python analysis/decoder_bakeoff.py runs/vr_full2 runs/vr_fine1 runs/vr_full3
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config            # noqa: E402
import reconstruct as R  # noqa: E402
import act_chirplet as AC  # noqa: E402

F0 = 12.0
AC.F0 = F0


def clean(seg, floor):
    base = np.median(seg)
    dev = seg - base
    sig = max(R.robust_sigma(dev), floor)
    return base + np.clip(dev, -config.ARTIFACT_Z * sig, config.ARTIFACT_Z * sig)


def visits(session):
    eeg_t, data, names, cur_t, gx, gy = R.load_session(session)
    fs = R.infer_fs(eeg_t)
    ok = cur_t <= eeg_t[-1] - 0.5
    cur_t, gx, gy = cur_t[ok], gx[ok], gy[ok]
    ch = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    st, en = np.r_[0, ch + 1], np.r_[ch + 1, len(gx)]
    floors = [R.hf_sigma(data[:, c]) for c in range(data.shape[1])]
    out = []
    for a, b in zip(st, en):
        i0, i1 = np.searchsorted(eeg_t, [cur_t[a], cur_t[b - 1]])
        segs = [clean(data[i0:i1, c], floors[c]) for c in range(data.shape[1])]
        out.append((int(gx[a]), int(gy[a]), segs))
    return out, names, fs


def auc(v, t):
    b, d = v[t > 0.5], v[t <= 0.5]
    return float(np.mean(b[:, None] > d[None, :]) + 0.5 * np.mean(b[:, None] == d[None, :]))


def otsu(v):
    s = np.sort(v)
    best, thr = -1, s[len(s) // 2]
    for k in range(1, len(s)):
        a, b = s[:k], s[k:]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best:
            best, thr = w, (s[k - 1] + s[k]) / 2
    return thr


def main(sessions):
    runs = []
    for s in sessions:
        vis, names, fs = visits(s)
        tgt = np.load(os.path.join(s, "target.npy"))
        cal = json.load(open(os.path.join(s, "calibration.json")))
        w = np.array([float(cal["weights"].get(n, 0.0)) for n in names])
        w = w / w.sum()
        act3 = AC.ActSSVEP(fs, order=3, harmonic=True)
        act1 = AC.ActSSVEP(fs, order=1, harmonic=False)
        rows = []
        for gx, gy, segs in vis:
            paper = np.array([R.paper_score(x, fs, F0) for x in segs])
            l1 = np.array([R.line_snr(x, fs, F0) for x in segs])
            l2 = np.array([R.line_snr(x, fs, 2 * F0) for x in segs])
            a3 = np.array([act3.score(x) for x in segs])
            a1 = np.array([act1.score(x) for x in segs])
            rows.append(dict(gx=gx, gy=gy, t=float(tgt[gy, gx] > 0.3), paper=paper,
                             l1=l1, l2=l2, act3=a3, act1=a1))
        runs.append(dict(name=os.path.basename(s), rows=rows, w=w, tgt=tgt, names=names))
        print(f"{runs[-1]['name']}: {len(rows)} positions, ACT fits {act3.n_transforms + act1.n_transforms}"
              f" in {act3.wall + act1.wall:.0f} s")

    def feats(run):
        X = np.array([np.r_[np.log(r["paper"] + 1e-9), np.log(r["l1"] + 1e-9),
                            np.log(r["l2"] + 1e-9)] for r in run["rows"]])
        X = np.nan_to_num(X)
        return (X - X.mean(0)) / (X.std(0) + 1e-9)      # per-run z-score, no labels

    results = {}
    for i, run in enumerate(runs):
        t = np.array([r["t"] for r in run["rows"]])
        sc = {}
        for key, label in (("paper", "paper score"), ("act3", "ACT (3 atoms, 12+24 Hz)"),
                           ("act1", "ACT (1 atom, 12 Hz)")):
            M = np.array([r[key] for r in run["rows"]])
            sc[label] = np.nansum(np.nan_to_num(M) * run["w"], axis=1)
        others = [r for j, r in enumerate(runs) if j != i]
        if others:
            Xtr = np.vstack([feats(o) for o in others])
            ytr = np.concatenate([[r["t"] for r in o["rows"]] for o in others])
            clf = LogisticRegression(C=0.3, max_iter=2000).fit(Xtr, ytr)
            sc["learned (trained on the other runs)"] = clf.decision_function(feats(run))
        res = {}
        for label, v in sc.items():
            thr = otsu(v)
            res[label] = dict(v=v, r=float(np.corrcoef(v, t)[0, 1]), auc=auc(v, t),
                              acc=float(np.mean((v > thr) == (t > 0.5))), thr=thr)
        results[run["name"]] = res
        print(run["name"])
        for label, d in res.items():
            print(f"   {label:38s} r={d['r']:+.2f}  AUC={d['auc']:.2f}  "
                  f"binary accuracy={d['acc'] * 100:.0f}%")

    labels = list(next(iter(results.values())).keys())
    fig, axes = plt.subplots(len(runs), 2 + 2 * len(labels) // 2 + len(labels) % 2 + 1,
                             figsize=(3.0 * (len(labels) + 2), 2.4 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        res = results[run["name"]]
        shape = run["tgt"].shape

        def img(v):
            g = np.full(shape, np.nan)
            for r, val in zip(run["rows"], v):
                g[r["gy"], r["gx"]] = val
            return g
        cm = plt.cm.gray.copy()
        cm.set_bad("#5a2a2a")
        panels = [("shown to the eye", (run["tgt"] > 0.3).astype(float))]
        for label in labels:
            v = res[label]["v"]
            lo, hi = np.percentile(v, [5, 95])
            panels.append((f"{label}\nr={res[label]['r']:.2f}, AUC {res[label]['auc']:.2f}",
                           img(np.clip((v - lo) / (hi - lo + 1e-12), 0, 1))))
        best = max(labels, key=lambda k: res[k]["auc"])
        panels.append((f"binary: {best.split(' (')[0]}\n{res[best]['acc'] * 100:.0f}% of positions right",
                       img((res[best]["v"] > res[best]["thr"]).astype(float))))
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]), ax.set_yticks([])
            if j < len(panels):
                ax.imshow(panels[j][1], cmap=cm, vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(panels[j][0], fontsize=8)
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(run["name"], fontsize=9)
    fig.tight_layout()
    out = os.path.join("runs", "figures", "fig10_decoder_bakeoff.png")
    fig.savefig(out, dpi=130)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
