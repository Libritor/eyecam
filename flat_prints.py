"""Two flat Mind to Matter prints for the Ender-3 Pro, both plain height fields
(no overhangs, no supports), as STL + G-code + preview:

  lithophane   the picture decoded from EEG (a run's reconstruction_grid.npy) as a
               white-PLA lithophane: thickness follows darkness, so the picture shows
               when it is held to a light. It is mirrored so it reads right from the
               smooth side, which is the side printed on the bed; printed solid.
  tactile      a literacy-free exam report: the left eye's four quadrants as a dome
               (height = each quadrant's response over the best; a silent quadrant
               stays low) and both eyes' red/green flicker photometry as ridges
               (x = red's share, y = response; left eye solid, right eye dashed) over a
               baseline with one peg under each eye's null.

    python flat_prints.py lithophane [--run runs/vr_vision1] [--slice] [--preview]
    python flat_prints.py tactile [--left runs/vr_exam_left_2026-11-03b] [--right runs/vr_exam_right_2026-11-03] [--slice] [--preview]

Outputs go to <run>/print/ (runs/ is never pushed: the report is the subject's own
exam data) and are copied to Downloads/mind_to_matter_prints/. Slicing reuses
die_print.py's CuraEngine recipe.
"""
import argparse
import json
import os
import shutil

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads", "mind_to_matter_prints")


# ── a height field as one closed solid ──────────────────────────────────────
def heightfield_solid(Z, res):
    """Z[iy, ix] in mm above z = 0 on a grid of res mm (x = ix * res, y = iy * res):
    top surface, flat bottom and four side walls, one watertight mesh."""
    h, w = Z.shape
    X, Y = np.meshgrid(np.arange(w) * res, np.arange(h) * res)
    n = h * w
    V = np.vstack([np.c_[X.ravel(), Y.ravel(), Z.ravel()], np.c_[X.ravel(), Y.ravel(), np.zeros(n)]])
    idx = np.arange(n).reshape(h, w)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    top = np.r_[np.c_[a, b, d], np.c_[a, d, c]]
    bottom = top[:, ::-1] + n
    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]
    p, q = ring, np.roll(ring, -1)
    sides = np.r_[np.c_[p, q + n, q], np.c_[p, p + n, q + n]]
    m = trimesh.Trimesh(V, np.vstack([top, bottom, sides]), process=True)
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


def check(m, name):
    print(f"{name}: {len(m.faces)} triangles, {m.extents[0]:.1f} x {m.extents[1]:.1f} x {m.extents[2]:.2f} mm, "
          f"watertight {m.is_watertight}, volume {m.volume / 1000:.1f} cm3")
    if not m.is_watertight:
        raise SystemExit(name + " is not watertight")


# ── 1. the lithophane ───────────────────────────────────────────────────────
def lithophane_height(grid, cell=6.0, frame=4.0, t_min=0.8, t_max=3.0, t_frame=3.6, res=0.25):
    """Brightness from the decoded grid (stretched between its 10th and 97th
    percentiles), thickness t_max at black down to t_min at white; returned in
    mesh order (rotated 180 degrees, so it reads right from the bed side)."""
    lo, hi = np.percentile(grid, 10), np.percentile(grid, 97)
    b = np.clip((grid - lo) / (hi - lo), 0, 1) ** 0.8
    k = int(round(cell / res)); f = int(round(frame / res))
    img = np.kron(t_max - (t_max - t_min) * b, np.ones((k, k)))
    T = np.pad(img, f, constant_values=t_frame)
    return T[::-1, ::-1], b


def lithophane(a):
    run = os.path.join(HERE, a.run)
    grid = np.load(os.path.join(run, "reconstruction_grid.npy"))
    Z, b = lithophane_height(grid)
    out = os.path.join(run, "print"); os.makedirs(out, exist_ok=True)
    name = f"lithophane_{os.path.basename(run.rstrip('/'))}"
    m = heightfield_solid(Z, 0.25)
    check(m, name)
    stl = os.path.join(out, name + ".stl"); m.export(stl)
    files = [stl]
    if a.preview:
        files.append(preview_lithophane(Z, grid, b, os.path.join(out, name + "_preview.png")))
    if a.slice:
        files.append(slice_flat(stl, {
            "layer_height": "0.12", "layer_height_0": "0.2",
            "infill_sparse_density": "100", "infill_pattern": "lines", "wall_line_count": "2",
            "top_layers": "0", "bottom_layers": "0",            # 100 % infill: one solid body
            "speed_print": "40", "adhesion_type": "brim", "brim_width": "5", "support_enable": "false"}))
    copy_out(files)


def preview_lithophane(Z, grid, b, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    T = Z[::-1, ::-1]                                       # as seen from the smooth side
    lamp = np.exp(-1.25 * T); lamp = (lamp - lamp.min()) / (lamp.max() - lamp.min())
    warm = LinearSegmentedColormap.from_list("lamp", ["#120804", "#7a3510", "#f0a040", "#fff3d6"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), dpi=150)
    axs[0].imshow(grid, cmap="gray", interpolation="nearest"); axs[0].set_title("decoded from EEG", fontsize=9)
    axs[1].imshow(T, cmap="gray_r", interpolation="nearest"); axs[1].set_title("thickness (dark = thick)", fontsize=9)
    axs[2].imshow(lamp, cmap=warm, interpolation="bilinear"); axs[2].set_title("held to a lamp, smooth side to you", fontsize=9)
    for ax in axs:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


# ── 2. the tactile report ───────────────────────────────────────────────────
def tactile_height(left, right, res=0.25, W=100.0, H=56.0):
    X, Y = np.meshgrid(np.arange(0, W + res / 2, res), np.arange(0, H + res / 2, res))
    Z = np.full(X.shape, 2.0)                                 # 2 mm plate
    cx, cy, R = 26.0, 28.0, 20.0                              # the dome
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy) / R
    fi = left["field"]
    quad = np.where(dy >= 0, 0, 2) + np.where(dx >= 0, 1, 0)  # s0 top-left .. s3 bottom-right (seen from above)
    rel = np.array([fi["rel"][s] if s in fi["responding"] else 0.0 for s in range(fi["n"])])
    dome = np.sqrt(np.clip(1 - r ** 2, 0, None))
    inside = r < 1
    Z += np.where(inside, 0.6 + 9.0 * rel[quad] * dome, 0)
    Z -= np.where(inside & ((np.abs(dx) < 0.5) | (np.abs(dy) < 0.5)), 1.0, 0)    # grooves between quadrants
    Z -= np.where(np.hypot(dx, dy) < 1.8, 1.5, 0)                                 # the fixation dimple
    px0, py0, pw, ph = 56.0, 12.0, 38.0, 32.0                 # the photometry panel
    Z += np.where((np.abs(Y - (py0 - 2)) < 0.6) & (X > px0 - 1) & (X < px0 + pw + 1), 1.0, 0)   # baseline
    for ro, dashed in ((left, False), (right, True)):
        d = ro["photometry"]["pairs"]["rg"]
        xs = np.linspace(0.2, 0.8, 240)
        ys = np.interp(xs, d["mixes"], d["amps"])
        ridge = np.zeros_like(Z)
        for i, (u, amp) in enumerate(zip(xs, ys)):
            if dashed and (i // 15) % 2:                       # ~2.4 mm dashes and gaps
                continue
            gx, gy = px0 + (u - 0.2) / 0.6 * pw, py0 + amp / 7.0 * ph
            ridge = np.maximum(ridge, 2.0 * np.exp(-((X - gx) ** 2 + (Y - gy) ** 2) / (2 * 0.8 ** 2)))
        Z += ridge
        gx = px0 + (d["null"] - 0.2) / 0.6 * pw                # a peg on the baseline under the null
        Z += np.where(np.hypot(X - gx, Y - (py0 - 2)) < 1.2, 1.8, 0)
    Z += np.where((X < 4) & (Y > H - 4), 1.5, 0)               # the orientation corner (top left)
    return np.clip(Z, 0.6, None)


def tactile(a):
    left = json.load(open(os.path.join(HERE, a.left, "exam_readout.json")))["eyes"]["left"]
    right = json.load(open(os.path.join(HERE, a.right, "exam_readout.json")))["eyes"]["right"]
    Z = tactile_height(left, right)
    out = os.path.join(HERE, a.left, "print"); os.makedirs(out, exist_ok=True)
    name = "tactile_exam_report"
    m = heightfield_solid(Z, 0.25)
    check(m, name)
    stl = os.path.join(out, name + ".stl"); m.export(stl)
    files = [stl]
    if a.preview:
        files.append(preview_tactile(Z, os.path.join(out, name + "_preview.png")))
    if a.slice:
        files.append(slice_flat(stl, {
            "layer_height": "0.2", "layer_height_0": "0.2", "infill_sparse_density": "15",
            "wall_line_count": "3", "top_layers": "5", "bottom_layers": "4",
            "adhesion_type": "skirt", "support_enable": "false"}))
    copy_out(files)


def preview_tactile(Z, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=32)
    rgb = ls.shade_rgb(np.broadcast_to([0.86, 0.86, 0.83], Z.shape + (3,)).copy(), Z, blend_mode="soft", vert_exag=5)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.imshow(rgb[::-1], interpolation="bilinear"); ax.set_axis_off()
    ax.set_title("tactile exam report, from above (100 x 56 mm)", fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


# ── slicing and copies ──────────────────────────────────────────────────────
def slice_flat(stl, extra):
    from die_print import slice_stl
    gcode = stl[:-4] + "_ender3pro.gcode"
    slice_stl(stl, gcode, "creality_ender3pro", extra)
    return gcode


def copy_out(files):
    os.makedirs(DOWNLOADS, exist_ok=True)
    for f in files:
        shutil.copy2(f, DOWNLOADS)
    print("copied to", DOWNLOADS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("lithophane")
    p.add_argument("--run", default="runs/vr_vision1")
    p = sub.add_parser("tactile")
    p.add_argument("--left", default="runs/vr_exam_left_2026-11-03b")
    p.add_argument("--right", default="runs/vr_exam_right_2026-11-03")
    for p in sub.choices.values():
        p.add_argument("--slice", action="store_true")
        p.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    {"lithophane": lithophane, "tactile": tactile}[a.what](a)


if __name__ == "__main__":
    main()
