"""Is the object's relief in the G-code? Rebuild the printed surface from the
relief G-code itself (every extrusion move of the model, raft and support left
out, voxelised at 0.2 mm) and measure, at each decoded cell, how far the printed
surface stands from the flush level along the cell's normal: a cell must read
at the height its colour has in the relief map (die_print.RELIEF, or --relief).
Works for any mesh3d object printed by die_print.py: the die, the globe, the
donut, the Rubik's cube. Writes <print>/<name>_gcode_check.png + .pdf (six
views of the object as the G-code prints it) and .json, and prints the
per-cell table.

    python Tools/print_gcode_check.py --run runs/globe_shown --mesh mesh_shown.obj \
        --object globe --name globe_shown --out docs/results/globe_shown
    python Tools/print_gcode_check.py                     # the die, runs/vr_die1
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import die_print  # noqa: E402

VOX = 0.2                  # voxel edge, mm (the layer height)
LINE_W = 0.4               # nozzle line width, mm
SKIP = ("SUPPORT", "SKIRT", "RAFT", "PRIME", "BRIM")
TOL = 0.35                 # mm: a cell within this of its expected offset counts as printed right
SIDES = {(1, 0, 0): "+x", (-1, 0, 0): "-x", (0, 1, 0): "+y", (0, -1, 0): "-y", (0, 0, 1): "top (+z)",
         (0, 0, -1): "bottom (-z, on the raft)"}


def extrusion_segments(path):
    """[(x0, y0, x1, y1, z)] of the model's own extrusion moves (layers >= 0,
    support / raft / skirt types left out), from Cura G-code."""
    segs = []
    x = y = z = 0.0
    e_prev, relative, layer, typ = 0.0, False, None, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(";LAYER:"):
                layer = int(line[7:])
                continue
            if line.startswith(";TYPE:"):
                typ = line[6:].strip().upper()
                continue
            if line.startswith("G92"):
                m = re.search(r"E(-?[\d.]+)", line)
                if m:
                    e_prev = float(m.group(1))
                continue
            if line.startswith("M83"):
                relative = True
                continue
            if line.startswith("M82"):
                relative = False
                continue
            if not (line.startswith("G1") or line.startswith("G0")):
                continue
            nx, ny, nz = x, y, z
            m = re.search(r"\bX(-?[\d.]+)", line)
            if m:
                nx = float(m.group(1))
            m = re.search(r"\bY(-?[\d.]+)", line)
            if m:
                ny = float(m.group(1))
            m = re.search(r"\bZ(-?[\d.]+)", line)
            if m:
                nz = float(m.group(1))
            m = re.search(r"\bE(-?[\d.]+)", line)
            d = 0.0
            if m:
                e = float(m.group(1))
                d = e if relative else e - e_prev
                if not relative:
                    e_prev = e
            if d > 0 and layer is not None and layer >= 0 and not any(k in typ for k in SKIP) \
                    and (nx != x or ny != y):
                segs.append((x, y, nx, ny, nz))
            x, y, z = nx, ny, nz
    return np.array(segs, float)


def voxelise(segs):
    """Occupancy grid of the extruded lines (sampled every 0.1 mm, widened to
    the line width), its origin, and the model's bounds."""
    lo = np.array([segs[:, [0, 2]].min(), segs[:, [1, 3]].min(), segs[:, 4].min()]) - 2 * VOX
    hi = np.array([segs[:, [0, 2]].max(), segs[:, [1, 3]].max(), segs[:, 4].max()]) + 2 * VOX
    shape = np.ceil((hi - lo) / VOX).astype(int) + 1
    occ = np.zeros(shape, bool)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    n = np.maximum(2, np.ceil(L / 0.1).astype(int))
    xs, ys, zs = [], [], []
    for (x0, y0, x1, y1, z), k in zip(segs, n):
        t = np.linspace(0, 1, k)
        xs.append(x0 + (x1 - x0) * t); ys.append(y0 + (y1 - y0) * t); zs.append(np.full(k, z))
    P = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    I = np.round((P - lo) / VOX).astype(int)
    occ[I[:, 0], I[:, 1], I[:, 2]] = True
    from scipy.ndimage import binary_dilation
    r = int(round(LINE_W / 2 / VOX))           # half a line width sideways, one layer down
    st = np.zeros((2 * r + 1, 2 * r + 1, 2), bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    st[:, :, 1] = (xx ** 2 + yy ** 2) <= r * r
    st[r, r, 0] = True
    occ = binary_dilation(occ, structure=st)
    return occ, lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "vr_die1"))
    ap.add_argument("--mesh", default="mesh_decoded.obj")
    ap.add_argument("--object", default="die", help="the object's name: its relief map and orientation")
    ap.add_argument("--relief", default=None, help="override the relief map, 'blue:-1.2,green:0' (mm)")
    ap.add_argument("--name", default="die_decoded_eeg")
    ap.add_argument("--out", default=None, help="the print folder (default <run>/print)")
    ap.add_argument("--size", type=float, default=40.0)
    ap.add_argument("--depth", type=float, default=1.2)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "print")
    gcode = os.path.join(out, a.name + "_relief.gcode")
    stl = os.path.join(out, a.name + "_relief.stl")

    # the decoded cells on the base object, in the frame the slicer was handed
    verts, uvs, quads = die_print.load_obj(os.path.join(a.run, a.mesh))
    texture = os.path.join(a.run, os.path.splitext(a.mesh)[0] + ".png")
    cells = die_print.cell_colours(uvs, quads, texture)
    if die_print.Y_UP.get(a.object, True):
        verts = die_print.rotate_y_up(verts)
    scale = a.size / (2 * float(np.abs(verts).max()))
    base, old2new = die_print.merged_mesh(verts, quads, scale)
    rel = trimesh.load(stl, force="mesh")
    shift = np.array([-rel.bounds.mean(axis=0)[0], -rel.bounds.mean(axis=0)[1], -rel.bounds[0][2]])
    V = np.asarray(base.vertices) + shift
    FN = np.asarray(base.face_normals)
    expect = die_print.relief_map(a.object, {c for _, c in cells}, a.depth, a.relief)
    moved = sorted(c for c, d in expect.items() if d)

    segs = extrusion_segments(gcode)
    occ, lo = voxelise(segs)
    # the engine put the model at the bed centre: align the printed bounds to the mesh's
    pr_lo = lo + VOX * np.array(np.nonzero(occ)).min(axis=1)
    pr_hi = lo + VOX * np.array(np.nonzero(occ)).max(axis=1)
    mesh_lo, mesh_hi = rel.bounds[0] + shift, rel.bounds[1] + shift
    off = np.array([(pr_lo[0] + pr_hi[0]) / 2 - (mesh_lo[0] + mesh_hi[0]) / 2,
                    (pr_lo[1] + pr_hi[1]) / 2 - (mesh_lo[1] + mesh_hi[1]) / 2,
                    pr_hi[2] - mesh_hi[2]])
    print(f"G-code: {len(segs)} extrusion moves of the model, printed extent "
          f"{np.round(pr_hi - pr_lo, 1)} mm (mesh {np.round(mesh_hi - mesh_lo, 1)}), "
          f"placed at {np.round((pr_lo + pr_hi) / 2, 1)}; {occ.shape} voxels of {VOX} mm")

    def occupied(p):
        i = np.round((p - lo) / VOX).astype(int)
        if np.any(i < 0) or np.any(i >= occ.shape):
            return False
        return bool(occ[i[0], i[1], i[2]])

    # per cell: outermost printed material along the normal from each of its quads' centres
    ss = np.arange(-3.0, 3.0001, 0.05)
    per_cell = {}
    for q, ((vi, _), (cell, col)) in enumerate(zip(quads, cells)):
        ids = [old2new[x] for x in vi]
        c = V[ids].mean(axis=0) + off
        n = FN[2 * q]
        found = [s for s in ss if occupied(c + s * n)]
        meas = max(found) if found else float("nan")
        d = per_cell.setdefault(cell, dict(colour=col, meas=[], normal=np.zeros(3)))
        d["meas"].append(meas)
        d["normal"] += n
    rows = []
    for cell, d in sorted(per_cell.items()):
        n = d["normal"] / max(np.linalg.norm(d["normal"]), 1e-9)
        side = SIDES[tuple(int(v) for v in np.round(n))] if np.abs(n).max() > 0.9 else "edge"
        m = np.array(d["meas"], float)
        med = float(np.nanmedian(m))
        exp = expect[d["colour"]]
        rows.append(dict(cell=[int(cell[0]), int(cell[1])], side=side, colour=d["colour"],
                         expected_mm=exp, measured_mm=round(med, 2),
                         spread_mm=round(float(np.nanmax(m) - np.nanmin(m)), 2),
                         ok=bool(abs(med - exp) <= TOL)))
    raised = [r for r in rows if r["expected_mm"]]
    print(f"\n{'cell':>8} {'side':>22} {'colour':>7} {'expect':>7} {'G-code':>7} {'spread':>6}  ok")
    for r in rows:
        print(f"{str(tuple(r['cell'])):>8} {r['side']:>22} {r['colour']:>7} "
              f"{r['expected_mm']:>+7.2f} {r['measured_mm']:>+7.2f} {r['spread_mm']:>6.2f}  {'yes' if r['ok'] else 'NO'}")
    n_ok = sum(r["ok"] for r in rows)
    n_moved_ok = sum(r["ok"] for r in raised)
    print(f"\n{n_ok}/{len(rows)} cells print at their relief (within {TOL} mm); "
          f"{n_moved_ok}/{len(raised)} of the cells that stand off the flush level ({', '.join(moved)}). "
          f"The relief step is {a.depth} mm = {a.depth / VOX:.0f} layers.")

    # six views of the object as the G-code prints it: outermost printed voxel per column of the view
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
    idx = np.indices(occ.shape)
    views = [("top (+z)", 2, True, 0, 1), ("bottom (-z)", 2, False, 0, 1),
             ("+x", 0, True, 1, 2), ("-x", 0, False, 1, 2), ("+y", 1, True, 0, 2), ("-y", 1, False, 0, 2)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, (name, axis, outer_max, h, v) in zip(axes.flat, views):
        coord = np.where(occ, idx[axis], -1 if outer_max else 10 ** 6)
        depth = coord.max(axis=axis) if outer_max else coord.min(axis=axis)
        depth = np.where(occ.any(axis=axis), depth, np.nan).astype(float) * VOX
        if not outer_max:
            depth = -depth
        # a curved object has no flush face: the picture is the surface height
        # itself, which still shows the sunk and raised cells against their neighbours
        vals = depth[np.isfinite(depth)]
        ref = np.nanpercentile(vals, 90) if len(vals) else 0.0
        rel_d = depth - ref
        img = rel_d if h < v else rel_d.T
        im = ax.imshow(img.T, origin="lower", cmap="gray", vmin=-a.size / 3, vmax=a.depth * 1.25,
                       interpolation="nearest")
        ax.set_title(name, loc="left", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if ax is axes.flat[0]:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                         label="printed surface, mm from the outermost level")
    fig.suptitle(f"{a.name}_relief.gcode as it prints: {n_ok}/{len(rows)} cells at their relief, "
                 f"{n_moved_ok}/{len(raised)} of the sunk or raised ones", fontsize=8)
    png = os.path.join(out, f"{a.name}_gcode_check.png")
    fig.savefig(png, dpi=160)
    fig.savefig(os.path.join(out, f"{a.name}_gcode_check.pdf"))
    plt.close(fig)
    with open(os.path.join(out, f"{a.name}_gcode_check.json"), "w") as f:
        json.dump(dict(gcode=gcode, object=a.object, voxel_mm=VOX, tolerance_mm=TOL, relief_mm=expect,
                       cells=rows, n_ok=n_ok, n_cells=len(rows), n_moved_ok=n_moved_ok,
                       n_moved=len(raised)), f, indent=1)
    print("wrote", png)
    return 0 if n_moved_ok == len(raised) else 1


if __name__ == "__main__":
    sys.exit(main())
