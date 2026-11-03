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

  snellen      the letter chart as the brain responded (an exam session with the
               snellen test): the classic chart's rows, each line's letters raised in
               proportion to the response at that line (barely raised where the brain
               did not respond), 20/X beside every line, a bar under the finest line
               that responded
  ladder       the light ladder (an exam session with the intensity test): one column
               per light level, its height the response, a square below sized by the
               level; plus the same response as a lithophane strip (the brain's grey
               scale when held to a light)

    python flat_prints.py lithophane [--run runs/vr_vision1] [--slice] [--preview]
    python flat_prints.py snellen --session runs/<exam session> [--eye right] [--slice] [--preview]
    python flat_prints.py ladder --session runs/<exam session> [--eye right] [--slice] [--preview]
    python flat_prints.py tactile [--left runs/vr_exam_left_2026-11-03b] [--right runs/vr_exam_right_2026-11-03] [--slice] [--preview]

Outputs go to <run>/print/ (runs/ is never pushed: the report is the subject's own
exam data) and are copied to Downloads/mind_to_matter_prints/. Slicing reuses
die_print.py's CuraEngine recipe.
"""
import argparse
import json
import math
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


# ── 3 and 4: the letter chart and the light ladder, from an exam read-out ───
# Built from boxes on a plate (a height field fine enough for 3.5 mm letters
# would be millions of triangles); the boxes overlap the plate and each other,
# which every slicer unions. Letters and digits on a 5 x 5 and a 3 x 5 grid.
GLYPHS = {
    "C": [".####", "#....", "#....", "#....", ".####"], "D": ["####.", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "####.", "#....", "#####"], "F": ["#####", "#....", "####.", "#....", "#...."],
    "L": ["#....", "#....", "#....", "#....", "#####"], "O": [".###.", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "####.", "#....", "#...."], "T": ["#####", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "...#.", "..#..", ".#...", "#####"],
    "0": ["###", "#.#", "#.#", "#.#", "###"], "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"], "3": ["###", "..#", "###", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"], "5": ["###", "#..", "###", "..#", "###"],
    "6": ["###", "#..", "###", "#.#", "###"], "7": ["###", "..#", "..#", "..#", "..#"],
    "8": ["###", "#.#", "###", "#.#", "###"], "9": ["###", "#.#", "###", "..#", "###"],
    "/": ["..#", "..#", ".#.", "#..", "#.."],
}
CHART = {200: "E", 100: "FP", 70: "TOZ", 50: "LPED", 40: "PECFD", 30: "EDFCZP", 25: "FELOPZD",
         20: "DEFPOTEC", 15: "LEFODPCT", 13: "FDPLTCEO", 10: "PEZOLCFTD"}   # the classic chart's rows
BASE = 2.0                                                # plate thickness, mm


def box(x0, y0, x1, y1, z0, z1):
    b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return b


def glyph_boxes(ch, x, y_top, cell, z1):
    """Raised boxes for one character, its top-left corner at (x, y_top), y up;
    a row's touching cells are one box."""
    out = []
    for r, row in enumerate(GLYPHS[ch]):
        c = 0
        while c < len(row):
            if row[c] == "#":
                c0 = c
                while c < len(row) and row[c] == "#":
                    c += 1
                out.append(box(x + c0 * cell, y_top - (r + 1) * cell, x + c * cell, y_top - r * cell,
                               BASE - 0.05, z1))
            else:
                c += 1
    return out


def text_boxes(text, x, y_top, h, z1, gap=0.6):
    """A string of glyphs (cap height h); returns (boxes, width)."""
    out, cx = [], x
    for ch in text:
        if ch == " ":
            cx += h * 0.6
            continue
        cols = len(GLYPHS[ch][0])
        out += glyph_boxes(ch, cx, y_top, h / 5.0, z1)
        cx += cols * h / 5.0 + gap * h / 5.0 * 1.5
    return out, cx - x


def eye_readout(session, eye, test):
    ro = json.load(open(os.path.join(HERE, session, "exam_readout.json")))
    eyes = ro["eyes"]
    if eye is None:
        eye = next((e for e in ("right", "left", "both") if eyes.get(e, {}).get(test)), None)
    if eye is None or not eyes.get(eye, {}).get(test):
        raise SystemExit(f"{session}: no {test} read-out")
    return eye, eyes[eye][test]


def relief_of(amps, p, top=2.4, low=0.25):
    """Height above the plate per step: up to `top` mm in proportion to the
    response for a significant step, `low` (barely raised) for a silent one."""
    a = np.array([v if v is not None else 0.0 for v in amps], float)
    peak = max(float(np.nanmax(a)), 1e-9)
    return [low + (top - low) * max(0.0, v) / peak if pv < 0.05 and v > 0 else low
            for v, pv in zip(a, p)]


STANDARD_LINES = [200, 100, 70, 50, 40, 30, 25, 20, 15, 13, 10]


def chart_from_acuity(ac):
    """The reversing-bars acuity test as chart lines: a bar of w arcmin is the
    MAR of the line 20/(20 w); each delivered width goes to its nearest
    standard line (widths the panel delivered alike merge: the larger
    amplitude, the smaller p)."""
    rows = {}
    for w, amp, pv in zip(ac["sizes"], ac["amps"], ac["p"]):
        L = min(STANDARD_LINES, key=lambda s: abs(math.log(s) - math.log(20.0 * w)))
        a0, p0 = rows.get(L, (-1e9, 1.0))
        rows[L] = (max(a0, amp), min(p0, pv))
    lines = sorted(rows, reverse=True)
    out = dict(lines=[float(L) for L in lines], amps=[rows[L][0] for L in lines], p=[rows[L][1] for L in lines])
    if ac.get("finest_sig") is not None:
        out["finest_line"] = float(min(STANDARD_LINES, key=lambda s: abs(math.log(s) - math.log(20.0 * ac["finest_sig"]))))
    return out


def snellen_plaque(a):
    try:
        eye, sn = eye_readout(a.session, a.eye, "snellen")
    except SystemExit:                                 # the bars version of the chart (acuity at chart sizes)
        eye, ac = eye_readout(a.session, a.eye, "acuity")
        sn = chart_from_acuity(ac)
    tested = {round(L, 3): (amp, pv) for L, amp, pv in zip(sn["lines"], sn["amps"], sn["p"])}
    lines = sorted({round(L, 3) for L in sn["lines"]}, reverse=True)   # the lines tested, top to bottom
    rel = dict(zip(sn["lines"], relief_of(sn["amps"], sn["p"])))
    finest = sn.get("finest_line")
    lmax, lmin = max(lines), min(lines)
    def h_of(L):                                       # 18 mm at the top line, 3.5 mm at the bottom (log)
        return 3.5 + (18.0 - 3.5) * (math.log(L) - math.log(lmin)) / max(math.log(lmax) - math.log(lmin), 1e-9)
    rows = []
    for L in lines:
        letters = CHART.get(int(L)) if float(L).is_integer() and int(L) in CHART else None
        if letters is None:
            letters = ("EFPTOZLDC" * 3)[:max(1, min(9, int(round(2 + 6 * (math.log(lmax) - math.log(L)) /
                                                                  max(math.log(lmax) - math.log(lmin), 1e-9)))))]
        rows.append((L, letters, h_of(L)))
    margin, label_w = 6.0, 22.0
    widths = [len(t) * h * 1.6 - 0.6 * h for _, t, h in rows]
    W = max(widths) + label_w + 3 * margin
    H = sum(h * 1.55 for _, _, h in rows) + 2 * margin
    parts = [box(0, 0, W, H, 0, BASE)]
    y = H - margin
    for (L, letters, h), w in zip(rows, widths):
        key = round(L, 3)
        z = BASE + (rel.get(L) if key in tested else 0.2)
        x0 = margin + (max(widths) - w) / 2
        for k, ch in enumerate(letters):
            parts += glyph_boxes(ch, x0 + k * h * 1.6, y, h / 5.0, z)
        lab, lw = text_boxes(f"20/{L:g}", 0, 0, 3.0, BASE + 0.6)
        lab_h = 3.0
        for b in lab:                                  # the label right-aligned in its column, mid-line
            b.apply_translation((W - margin - lw, y - (h - lab_h) / 2, 0))
        parts += lab
        if finest is not None and abs(L - finest) < 1e-6:
            # the bar under the finest line the brain responded to (the chart's green line)
            t = min(1.2, 0.3 * h)                      # fits in the 0.55 h gap below the line
            parts.append(box(margin, y - 1.15 * h - t, margin + max(widths), y - 1.15 * h,
                             BASE - 0.05, BASE + 1.0))
        y -= h * 1.55
    m = trimesh.util.concatenate(parts)
    name = f"snellen_chart_{eye}"
    out = os.path.join(HERE, a.session, "print"); os.makedirs(out, exist_ok=True)
    stl = os.path.join(out, name + ".stl"); m.export(stl)
    print(f"{name}: {W:.0f} x {H:.0f} mm, {len(m.faces)} triangles, {len(parts)} boxes; lines tested "
          f"{', '.join(f'20/{L:g}' for L in sn['lines'])}; finest with a response "
          f"{'20/%g' % finest if finest else 'none'}")
    files = [stl]
    if a.preview:
        files.append(preview_boxes(parts, W, H, os.path.join(out, name + "_preview.png"),
                                   f"letter chart as the {eye} eye's brain responded (raised = responded)"))
    if a.slice:
        # 0.2 mm layers: the letters' detail is in x and y (the 20/10 line's strokes are 0.7 mm)
        files.append(slice_flat(stl, {"layer_height": "0.2", "layer_height_0": "0.2",
                                      "infill_sparse_density": "15", "wall_line_count": "2",
                                      "top_layers": "4", "bottom_layers": "3",
                                      "adhesion_type": "skirt", "support_enable": "false"}))
    copy_out(files)


def light_ladder(a):
    eye, li = eye_readout(a.session, a.eye, "intensity")
    order = np.argsort(li["levels"])                   # dim to bright, left to right
    levels = [li["levels"][i] for i in order]
    amps = [li["amps"][i] for i in order]
    ps = [li["p"][i] for i in order]
    rel = relief_of(amps, ps, top=6.0, low=0.3)
    n, cw, gap, margin = len(levels), 11.0, 2.0, 5.0
    W = n * cw + (n - 1) * gap + 2 * margin
    H = 46.0
    parts = [box(0, 0, W, H, 0, BASE)]
    lo, hi = math.log10(min(levels)), math.log10(max(levels))
    dimmest = li.get("dimmest_sig")
    for k, (lev, r) in enumerate(zip(levels, rel)):
        x0 = margin + k * (cw + gap)
        parts.append(box(x0, 16.0, x0 + cw, 40.0, BASE - 0.05, BASE + r))            # the response column
        s = 2.0 + 7.0 * (math.log10(lev) - lo) / max(hi - lo, 1e-9)                    # the light: a square
        parts.append(box(x0 + (cw - s) / 2, 10.5 - s / 2, x0 + (cw + s) / 2, 10.5 + s / 2,
                         BASE - 0.05, BASE + 0.8))
        if dimmest is not None and abs(lev - dimmest) < 1e-9:
            parts.append(box(x0, 41.5, x0 + cw, 43.0, BASE - 0.05, BASE + 1.0))      # mark: dimmest followed
    m = trimesh.util.concatenate(parts)
    name = f"light_ladder_{eye}"
    out = os.path.join(HERE, a.session, "print"); os.makedirs(out, exist_ok=True)
    stl = os.path.join(out, name + ".stl"); m.export(stl)
    # and the same response as a lithophane strip: the brain's grey scale when held to a light
    peak = max(max(v for v in amps if v is not None), 1e-9)
    b = np.array([[max(0.0, v) / peak if pv < 0.05 and v is not None else 0.0 for v, pv in zip(amps, ps)]])
    k = int(round(9.0 / 0.25))
    T = np.pad(np.kron(3.0 - 2.2 * b, np.ones((k, k))), int(round(3 / 0.25)), constant_values=3.6)
    strip = heightfield_solid(T[::-1, ::-1], 0.25)
    check(strip, name + "_lithophane")
    stl2 = os.path.join(out, name + "_lithophane.stl"); strip.export(stl2)
    print(f"{name}: {W:.0f} x {H:.0f} mm plaque, {n} levels {min(levels):.3g}-{max(levels):.3g} of white; "
          f"dimmest with a response {f'{dimmest:.3g}' if dimmest is not None else 'none'}")
    files = [stl, stl2]
    if a.preview:
        files.append(preview_boxes(parts, W, H, os.path.join(out, name + "_preview.png"),
                                   f"light ladder, {eye} eye: column height = the brain's response, "
                                   "square = the light level (dim left, bright right)"))
    if a.slice:
        files.append(slice_flat(stl, {"layer_height": "0.2", "layer_height_0": "0.2",
                                      "infill_sparse_density": "15", "wall_line_count": "3",
                                      "adhesion_type": "skirt", "support_enable": "false"}))
        files.append(slice_flat(stl2, {"layer_height": "0.12", "layer_height_0": "0.2",
                                       "infill_sparse_density": "100", "infill_pattern": "lines",
                                       "wall_line_count": "2", "top_layers": "0", "bottom_layers": "0",
                                       "adhesion_type": "brim", "brim_width": "5", "support_enable": "false"}))
    copy_out(files)


def preview_boxes(parts, W, H, out, title):
    """Top view of a box model: each box's top height drawn into a grid, hill-shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    res = 0.2
    Z = np.zeros((int(H / res) + 1, int(W / res) + 1))
    for b in parts:
        (x0, y0, _), (x1, y1, z1) = b.bounds
        i0, i1 = int(round(y0 / res)), int(round(y1 / res))
        j0, j1 = int(round(x0 / res)), int(round(x1 / res))
        Z[i0:i1, j0:j1] = np.maximum(Z[i0:i1, j0:j1], z1)
    ls = LightSource(azdeg=315, altdeg=35)
    # tinted by height (light = the plate, dark = the tallest) so flat tops show their height
    rel = np.clip((Z - BASE) / max(float(Z.max()) - BASE, 1e-9), 0, 1)
    tint = plt.get_cmap("YlOrBr")(0.08 + 0.85 * rel)[..., :3]
    rgb = ls.shade_rgb(tint, Z, blend_mode="soft", vert_exag=4 / res)
    fig, ax = plt.subplots(figsize=(6, 6 * H / W + 0.5), dpi=160)
    ax.imshow(rgb[::-1], interpolation="bilinear"); ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


# ── slicing and copies ──────────────────────────────────────────────────────
def slice_flat(stl, extra):
    from die_print import slice_stl
    gcode = stl[:-4] + "_ender3pro.gcode"
    slice_stl(stl, gcode, "creality_ender3pro", extra)
    return gcode


COPY = True                        # --no-copy turns this off (test and gate runs)


def copy_out(files):
    if not COPY:
        return
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
    for what in ("snellen", "ladder"):
        p = sub.add_parser(what)
        p.add_argument("--session", required=True, help="a session folder with exam_readout.json")
        p.add_argument("--eye", default=None, help="right / left / both (default: the first with the test)")
    for p in sub.choices.values():
        p.add_argument("--slice", action="store_true")
        p.add_argument("--preview", action="store_true")
        p.add_argument("--no-copy", action="store_true", help="leave Downloads alone (test runs)")
    a = ap.parse_args()
    global COPY
    COPY = not a.no_copy
    {"lithophane": lithophane, "tactile": tactile, "snellen": snellen_plaque, "ladder": light_ladder}[a.what](a)


if __name__ == "__main__":
    main()
