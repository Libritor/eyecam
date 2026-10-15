"""A 3D object through the planes colour scan (the mode the sittings have
validated), instead of the frequency-phase scan mesh3d.py was written for.

The object's texture atlas (mesh3d.py) is a small grid of pure colours, and the
planes mode scans any colour image at grid size (`--mode planes --color-target
<png>`, quest_run.ps1 -Run cphoto). So:

  atlas   write the atlas as a PNG target plus a preview, and say how long the
          scan takes (18 s a cell at 5 / 5 / 8 s per plane, one pass)
  paint   after the session: snap the decoded planes grid to the object's
          palette (each plane at its own Otsu threshold, as the paper scores
          pure-colour cells) and paint it back on the mesh with mesh3d:
          mesh_comparison.png, mesh_decoded.obj / mesh_shown.obj, mesh_viewer.html

    python mesh_planes.py atlas mesh:die                 # targets_img/mesh_die_9x6.png
    .\\quest_run.ps1 -Run cphoto -Target targets_img/mesh_die_9x6.png
    python mesh_planes.py paint runs/vr_cphoto1 mesh:die

Only objects whose atlas has no holes can go through the planes scan (the
atlas layout of the cube objects, the donut's single sheet); the planes
driver visits every cell of the grid.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bwb                      # noqa: E402
import mesh3d                   # noqa: E402
import smoothcolor              # noqa: E402

PLANE_S = (5.0, 5.0, 8.0)       # quest_run.ps1 -Run cphoto: --plane-spc 5,5,8


def atlas_png(obj, path):
    """The atlas as an RGB PNG, one pixel a cell, holes black."""
    names, rgb = bwb.classes_of(obj.colours)
    tg = obj.target()
    img = (rgb[tg] * 255).astype(np.uint8)
    img[~obj.mask] = 0
    Image.fromarray(img, "RGB").save(path)
    return tg, names, rgb


def snap_to_palette(grid, names, rgb):
    """Decoded planes grid (gh, gw, 3 responses) -> class index per cell.
    Each plane is split at its own Otsu threshold on the log response (the
    paper's rule for pure-colour cells), which gives one of the 8 corner
    colours; that corner is then matched to the nearest palette colour.
    Returns (classes, margin) with margin the smallest distance of a cell's
    three responses to their thresholds, in units of each plane's range."""
    gh, gw, _ = grid.shape
    on = np.zeros((gh, gw, 3), bool)
    margin = np.full((gh, gw), np.inf)
    for k in range(3):
        v = grid[:, :, k]
        if np.nanstd(v) == 0:
            continue
        thr = smoothcolor.otsu_log(v)
        on[:, :, k] = v > thr
        span = max(np.nanpercentile(v, 98) - np.nanpercentile(v, 2), 1e-9)
        margin = np.minimum(margin, np.abs(v - thr) / span)
    corner = on.astype(float)
    d = ((corner[:, :, None, :] - rgb[None, None, :, :]) ** 2).sum(axis=3)
    return d.argmin(axis=2), np.where(np.isfinite(margin), margin, 0.0)


def cmd_atlas(a):
    obj = mesh3d.from_spec(a.object, a.layout)
    if not obj.mask.all():
        raise SystemExit(f"{obj.name} in the {obj.layout} layout has holes; the planes scan visits "
                         "every cell. Use --layout atlas.")
    os.makedirs(os.path.join(HERE, "targets_img"), exist_ok=True)
    stem = f"mesh_{obj.name}_{obj.grid_w}x{obj.grid_h}"
    out = a.out or os.path.join(HERE, "targets_img", stem + ".png")
    tg, names, rgb = atlas_png(obj, out)
    mesh3d.comparison(obj, tg, title=f"{obj.name}: the atlas the planes scan will show").save(
        os.path.splitext(out)[0] + "_preview.png")
    n = int(obj.mask.sum())
    used = sorted({names[int(c)] for c in tg[obj.mask]})
    print(f"{obj.name}: {n} cells on a {obj.grid_w} x {obj.grid_h} grid, colours {', '.join(used)}")
    print(f"one pass at {'/'.join(f'{s:g}' for s in PLANE_S)} s per plane: {n * sum(PLANE_S) / 60:.0f} min, "
          f"plus about 3 min of calibration")
    print(f"target: {out}\npreview: {os.path.splitext(out)[0] + '_preview.png'}")
    print(f"run:    .\\quest_run.ps1 -Run cphoto -Target {os.path.relpath(out, HERE)}")


def cmd_paint(a):
    obj = mesh3d.from_spec(a.object, a.layout)
    names, rgb = bwb.classes_of(obj.colours)
    grid = np.load(os.path.join(a.session, "reconstruction_planes_grid.npy"))
    if grid.shape[:2] != (obj.grid_h, obj.grid_w):
        raise SystemExit(f"session grid is {grid.shape[1]} x {grid.shape[0]}, "
                         f"{obj.name} needs {obj.grid_w} x {obj.grid_h}")
    shown = obj.target()
    tpath = os.path.join(a.session, "target_color.npy")
    if os.path.exists(tpath):
        tshown = np.load(tpath)
        shown_from_session = ((tshown[:, :, None, :] - rgb[None, None, :, :]) ** 2).sum(axis=3).argmin(axis=2)
        if not np.array_equal(shown_from_session[obj.mask], shown[obj.mask]):
            print("WARNING: the session's target_color.npy is not this object's atlas")
    decoded, margin = snap_to_palette(grid, names, rgb)
    seen = obj.mask & np.isfinite(grid).all(axis=2)
    # a certainty for the viewer: how far each cell's weakest plane sits from
    # its threshold, 0.5 at the threshold, 1 a full range away
    cert = np.clip(0.5 + margin, 0.5, 1.0)
    post = np.zeros((obj.grid_h, obj.grid_w, len(names)))
    for gy in range(obj.grid_h):
        for gx in range(obj.grid_w):
            post[gy, gx, decoded[gy, gx]] = cert[gy, gx]
    right = int((decoded[seen] == shown[seen]).sum())
    tot = int(seen.sum())
    summary = f"{right} of {tot} cells right through the planes scan"
    mesh3d.export_session(a.session, obj, shown, decoded, seen, post, obj.colours, summary)
    res = dict(object=obj.name, layout=obj.layout, cells=int(obj.mask.sum()), decoded=tot, right=right,
               per_colour={names[c]: dict(shown=int((shown[seen] == c).sum()),
                                          right=int(((shown[seen] == c) & (decoded[seen] == c)).sum()))
                           for c in sorted(set(shown[seen].tolist()))},
               source="reconstruction_planes_grid.npy, each plane split at its Otsu threshold")
    with open(os.path.join(a.session, "mesh_planes_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(summary + "; " + ", ".join(f"{k} {v['right']}/{v['shown']}" for k, v in res["per_colour"].items()))
    print(f"wrote {a.session}/mesh_comparison.png, mesh_decoded.obj, mesh_shown.obj, mesh_viewer.html, "
          "mesh_planes_result.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("atlas")
    p.add_argument("object", help="mesh:die, mesh:globe, mesh:donut, mesh:rubik, mesh:obj:<file>[:n]")
    p.add_argument("--layout", default="atlas", choices=["atlas", "net"])
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_atlas)
    p = sub.add_parser("paint")
    p.add_argument("session")
    p.add_argument("object")
    p.add_argument("--layout", default="atlas", choices=["atlas", "net"])
    p.set_defaults(fn=cmd_paint)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
