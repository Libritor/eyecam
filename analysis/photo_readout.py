"""Who is it? The forced choice over the photo gallery, decided from the EEG
picture alone (reconstruction_grid.npy): the picture is correlated with every
target in the gallery, on the raw grid and on the Otsu-thresholded grid, and
the best match is named; then, if the seal is given, it is opened and the
answer scored. Also draws the figure: the EEG picture, the thresholded
picture, the gallery with its correlations, and the photo that was shown.

    python analysis/photo_readout.py runs/<session> --seal targets_img/photo_today_seal.json
    python analysis/photo_readout.py runs/<session> --gallery "targets_img/photo_*_12.png"

Writes <session>/photo_readout.json and <session>/photo_readout.png.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import smoothcolor  # noqa: E402


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_png(path):
    from PIL import Image
    return np.asarray(Image.open(os.path.join(ROOT, path)).convert("L"), float) / 255.0


def decide(g, targets):
    rs = {k: corr(g, t) for k, t in targets.items()}
    order = sorted(rs, key=lambda k: -np.nan_to_num(rs[k], nan=-9))
    margin = rs[order[0]] - rs[order[1]] if len(order) > 1 else float("nan")
    return rs, order[0], margin


def figure(session, g, snapped, targets, rs_raw, rs_snap, best, seal, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(targets)
    n = len(names)
    fig, axes = plt.subplots(2, max(n, 3), figsize=(3.0 * max(n, 3), 6.4))
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(g, cmap="gray", interpolation="bicubic")
    axes[0, 0].set_title("from the EEG", fontsize=11)
    axes[0, 1].imshow(snapped, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[0, 1].set_title("thresholded (Otsu)", fontsize=11)
    if seal and seal.get("name") in targets:
        axes[0, 2].imshow(targets[seal["name"]], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        right = best == seal["name"]
        axes[0, 2].set_title(f"shown: {seal['name']} ({'named right' if right else 'named wrong'})",
                             fontsize=11, color="#2a9d4f" if right else "#d13b2f")
    for k, name in enumerate(names):
        ax = axes[1, k]
        ax.imshow(targets[name], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        mark = "  <- named" if name == best else ""
        ax.set_title(f"{name}: r {rs_raw[name]:+.2f} (snapped {rs_snap[name]:+.2f}){mark}",
                     fontsize=10, fontweight="bold" if name == best else "normal")
    label = os.path.basename(os.path.normpath(session))
    fig.suptitle(f"{label}: who is it? the forced choice over {n} photos, from the EEG picture alone",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session")
    ap.add_argument("--seal", default=None, help="json with the drawn answer and the gallery files")
    ap.add_argument("--gallery", default=None, help="glob of the gallery PNGs (when there is no seal)")
    a = ap.parse_args()
    g = np.load(os.path.join(a.session, "reconstruction_grid.npy")).astype(float)
    g = np.nan_to_num(g, nan=np.nanmin(g))
    seal = json.load(open(a.seal)) if a.seal and os.path.exists(a.seal) else None
    if seal and seal.get("gallery"):
        files = seal["gallery"]
    elif a.gallery:
        files = {os.path.basename(p).split("_")[1]: os.path.relpath(p, ROOT) for p in sorted(glob.glob(a.gallery))}
    else:
        raise SystemExit("give --seal (with the gallery in it) or --gallery <glob>")
    targets = {k: load_png(v) for k, v in files.items()}
    bad = {k: t.shape for k, t in targets.items() if t.shape != g.shape}
    if bad:
        raise SystemExit(f"gallery shapes {bad} do not match the picture {g.shape}")
    thr = smoothcolor.otsu_log(g)
    snapped = (g > thr).astype(float)
    rs_raw, best, margin = decide(g, targets)
    rs_snap, best_snap, margin_snap = decide(snapped, targets)
    print(f"{a.session}: {g.shape[1]}x{g.shape[0]} picture against {len(targets)} photos")
    print("  raw grid:     " + "  ".join(f"{k} {v:+.2f}" for k, v in rs_raw.items())
          + f"  -> {best} (margin {margin:+.2f})")
    print("  snapped grid: " + "  ".join(f"{k} {v:+.2f}" for k, v in rs_snap.items())
          + f"  -> {best_snap} (margin {margin_snap:+.2f})")
    out = dict(session=a.session, otsu=thr, gallery=files, raw=dict(r=rs_raw, best=best, margin=margin),
               snapped=dict(r=rs_snap, best=best_snap, margin=margin_snap), decision=best)
    tp = os.path.join(a.session, "target.npy")
    if os.path.exists(tp):
        t = np.load(tp).astype(float)
        if t.shape == g.shape:
            out["r_picture"] = corr(g, t)
            out["cells_right"] = int((snapped == (t > 0.5)).sum())
            print(f"  whole picture against the target shown: r {out['r_picture']:+.2f}, "
                  f"{out['cells_right']} of {t.size} cells right after the threshold")
    print(f"  DECISION (raw grid): it is {best}")
    if seal:
        out["seal"] = seal
        out["right"] = best == seal.get("name")
        print(f"  SEAL: {seal.get('name')} -> {'RIGHT' if out['right'] else 'WRONG'}"
              f"{'' if best_snap == best else f' (the snapped grid said {best_snap})'}")
    json.dump(out, open(os.path.join(a.session, "photo_readout.json"), "w"), indent=1)
    png = os.path.join(a.session, "photo_readout.png")
    figure(a.session, g, snapped, targets, rs_raw, rs_snap, best, seal, png)
    print(f"  figure: {png}")


if __name__ == "__main__":
    main()
