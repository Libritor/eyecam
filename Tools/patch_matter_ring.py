# -*- coding: utf-8 -*-
"""One-off (2026-11-09): the matter find scans a ring of background squares around the thing, two of
them shown black and white as controls, and stretches the grey levels to the display's range; the
prints map the brain's grid (which now covers the ring) back onto the thing's box. Idempotent."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "matter.py")
s = io.open(P, encoding="utf-8").read()
if "ring_cells" in s:
    print("already patched")
    raise SystemExit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('def find_object(frame_path, out_dir, grid="7x5", max_cells=20, depth="auto", name=""):',
    'def find_object(frame_path, out_dir, grid="7x5", max_cells=20, depth="auto", name="", ring=1, ring_cells=8):')
old_start = s.index("    bw, bh = x1 - x0, y1 - y0\n    gmax_c, gmax_r")
old_end = s.index("    # files\n    cv2.imwrite(os.path.join(out_dir, \"mask.png\")")
new_grid = '''    bw, bh = x1 - x0, y1 - y0
    gmax_c, gmax_r = [max(1, int(v)) for v in str(grid).lower().split("x")]
    cell = max(bw / gmax_c, bh / gmax_r)
    ocols = max(1, min(gmax_c, int(round(bw / cell))))
    orows = max(1, min(gmax_r, int(round(bh / cell))))
    cw, ch = bw / ocols, bh / orows
    # the grid: the thing's box plus a ring of background squares around it (clipped to the frame)
    ring = max(0, int(ring))
    gx0, gy0 = x0 - ring * cw, y0 - ring * ch
    cols, rows = ocols + 2 * ring, orows + 2 * ring
    cov = np.zeros((rows, cols)); lumc = np.zeros((rows, cols)); inframe = np.zeros((rows, cols), bool)
    for gy in range(rows):
        for gx in range(cols):
            ya, yb = int(round(gy0 + gy * ch)), int(round(gy0 + (gy + 1) * ch))
            xa, xb = int(round(gx0 + gx * cw)), int(round(gx0 + (gx + 1) * cw))
            ya_, yb_, xa_, xb_ = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
            if yb_ - ya_ < 0.6 * ch or xb_ - xa_ < 0.6 * cw:
                continue                                       # mostly off the frame: not shown
            inframe[gy, gx] = True
            region = mask[ya_:yb_, xa_:xb_]
            cov[gy, gx] = region.mean() if region.size else 0.0
            lumc[gy, gx] = lum[ya_:yb_, xa_:xb_].mean()      # the frame's own grey level there
    obj = inframe & (cov >= 0.3)
    if obj.sum() < 4:                        # a small thing: take its best-covered squares anyway
        obj = inframe & (cov >= max(0.05, np.sort(cov[inframe].ravel())[-min(4, int(inframe.sum()))]))
    if obj.sum() > max_cells:
        cy, cx = (rows - 1) / 2, (cols - 1) / 2
        rank = cov - 0.05 * np.hypot(*np.meshgrid(np.arange(cols) - cx, np.arange(rows) - cy)[::-1]) / max(rows, cols)
        thr = np.sort(rank[obj])[-max_cells]
        obj &= rank >= thr
    # background squares: spread around the ring, so the brain's picture has an outside
    bg_pool = [(gx, gy) for gy in range(rows) for gx in range(cols)
               if inframe[gy, gx] and not obj[gy, gx] and (gx < ring or gy < ring or gx >= cols - ring or gy >= rows - ring)]
    bg = []
    if bg_pool and ring_cells > 0:
        def ring_key(c):                      # walk the ring clockwise from the top left
            gx, gy = c
            if gy < ring: return (0, gx)
            if gx >= cols - ring: return (1, gy)
            if gy >= rows - ring: return (2, cols - gx)
            return (3, rows - gy)
        bg_pool.sort(key=ring_key)
        take = min(ring_cells, len(bg_pool))
        bg = [bg_pool[int(round(i * (len(bg_pool) - 1) / max(take - 1, 1)))] for i in range(take)]
        bg = list(dict.fromkeys(bg))
    # two controls among the background squares: one shown black (nothing to see), one white
    controls = {}
    if len(bg) >= 4:
        controls = {"black": list(bg[0]), "white": list(bg[len(bg) // 2])}
    scan = [(int(gx), int(gy)) for gy in range(rows) for gx in range(cols) if obj[gy, gx]] + [(int(a), int(b)) for a, b in bg]
    # the grey levels: the frame's luminance over the scanned squares, stretched to the display's
    # range (the brain tells black from grey from white, not 0.8 from 1.0)
    target = np.zeros((rows, cols))
    plain = [c for c in scan if list(c) not in controls.values()]
    if plain:
        vals = np.array([lumc[gy, gx] for gx, gy in plain])
        lo_, hi_ = np.percentile(vals, [5, 95])
        for gx, gy in plain:
            target[gy, gx] = float(np.clip((lumc[gy, gx] - lo_) / max(hi_ - lo_, 1e-6), 0, 1)) * 0.9 + 0.1
    if controls:
        target[controls["black"][1], controls["black"][0]] = 0.0
        target[controls["white"][1], controls["white"][0]] = 1.0
    cells = [[gx, gy] for gx, gy in scan]
    keep = np.zeros((rows, cols), bool)
    for gx, gy in scan:
        keep[gy, gx] = True
'''
s = s[:old_start] + new_grid + s[old_end:]
old_start = s.index("    n = 0\n    for gy in range(rows):\n        for gx in range(cols):\n            xa, ya = int(round(x0 + gx * cw))")
old_end = s.index("    label = name.strip()")
new_overlay = '''    n = 0
    for gx, gy in scan:
        xa, ya = int(round(gx0 + gx * cw)), int(round(gy0 + gy * ch))
        xb, yb = int(round(gx0 + (gx + 1) * cw)), int(round(gy0 + (gy + 1) * ch))
        n += 1
        is_obj = bool(obj[gy, gx])
        cv2.rectangle(ov, (xa, ya), (xb, yb), (255, 255, 255) if is_obj else (150, 150, 150), max(1, W // 600))
        tag = str(n)
        if controls.get("black") == [gx, gy]: tag = "B"
        if controls.get("white") == [gx, gy]: tag = "W"
        cv2.putText(ov, tag, (xa + 6, ya + int(ch * 0.35)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, W / 1600),
                    (255, 255, 255) if is_obj else (170, 170, 170), 1, cv2.LINE_AA)
'''
s = s[:old_start] + new_overlay + s[old_end:]
rep('''    found = dict(gridW=cols, gridH=rows, cells=cells, target=target.round(4).tolist(), coverage=cov.round(3).tolist(),
                 bbox=[int(x0), int(y0), int(x1), int(y1)], frame_size=[int(W), int(H)], method=method,''',
    '''    found = dict(gridW=cols, gridH=rows, cells=cells, target=target.round(4).tolist(), coverage=cov.round(3).tolist(),
                 object_cells=[[int(gx), int(gy)] for gy in range(rows) for gx in range(cols) if obj[gy, gx]],
                 background_cells=[[int(a), int(b)] for a, b in bg], controls=controls, ring=ring,
                 grid_bbox=[float(gx0), float(gy0), float(gx0 + cols * cw), float(gy0 + rows * ch)],
                 bbox=[int(x0), int(y0), int(x1), int(y1)], frame_size=[int(W), int(H)], method=method,''')
rep('''    cv2.putText(ov, (label + ": " if label else "") + f"{len(cells)} squares", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(ov, (label + ": " if label else "") + f"{len(cells)} squares", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (255, 255, 255), 2, cv2.LINE_AA)''',
    '''    head = (label + ": " if label else "") + f"{int(obj.sum())} squares + {len(bg)} around"
    cv2.putText(ov, head, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(ov, head, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (255, 255, 255), 2, cv2.LINE_AA)''')
rep('''    # the brain's picture at the frame's resolution over the object's box
    brain_hi = cv2.resize(bn.astype(np.float32), (Bw, Bh), interpolation=cv2.INTER_CUBIC)
    cell_px = max(2.0, min(Bw / gw, Bh / gh))''',
    '''    # the brain's picture at the frame's resolution: the grid covers the thing's box and its ring,
    # so it is laid over the grid's box and the thing's box is cut out of it
    gb = found.get("grid_bbox") or [x0, y0, x1, y1]
    Gw, Gh = max(1, int(round(gb[2] - gb[0]))), max(1, int(round(gb[3] - gb[1])))
    full = cv2.resize(bn.astype(np.float32), (Gw, Gh), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(round(x0 - gb[0])), int(round(y0 - gb[1]))
    brain_hi = np.zeros((Bh, Bw), np.float32)
    sx0, sy0 = max(0, ox), max(0, oy); sx1, sy1 = min(Gw, ox + Bw), min(Gh, oy + Bh)
    brain_hi[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = full[sy0:sy1, sx0:sx1]
    cell_px = max(2.0, min(Gw / gw, Gh / gh))''')
rep('''    lo, hi = (np.percentile(vals, 5), np.percentile(vals, 95)) if vals.size else (0.0, 1.0)
    bn = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1); bn[~visited] = 0.0''',
    '''    lo, hi = (np.percentile(vals, 5), np.percentile(vals, 95)) if vals.size else (0.0, 1.0)
    ctl = found.get("controls") or {}
    if ctl:   # the black and white squares the person also looked at are the yardsticks
        b_, w_ = float(g[ctl["black"][1], ctl["black"][0]]), float(g[ctl["white"][1], ctl["white"][0]])
        if b_ < hi: lo = max(lo, b_)
        if w_ > lo: hi = min(hi, w_)
    bn = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1); bn[~visited] = 0.0''')
rep('''    axs[0, 2].imshow(np.where(visited, tgt, np.nan), cmap="gray", vmin=0, vmax=1); axs[0, 2].set_title("the frame's squares (grey level shown)")
    axs[1, 0].imshow(np.where(visited, bn, np.nan), cmap="magma", vmin=0, vmax=1); axs[1, 0].set_title("the brain's squares (response, 0..1)")''',
    '''    axs[0, 2].imshow(np.where(visited, tgt, np.nan), cmap="gray", vmin=0, vmax=1); axs[0, 2].set_title("the frame's squares (grey level shown; B/W = controls)")
    axs[1, 0].imshow(np.where(visited, bn, np.nan), cmap="magma", vmin=0, vmax=1); axs[1, 0].set_title("the brain's squares (response, 0..1)")
    for ax in (axs[0, 2], axs[1, 0]):
        for tag, key in (("B", "black"), ("W", "white")):
            if ctl.get(key):
                ax.text(ctl[key][0], ctl[key][1], tag, ha="center", va="center", color="red", fontsize=12, fontweight="bold")''')
rep('''    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    fr = os.path.join(a.session, "frame.jpg")
    fd = find_object(fr, a.session, a.grid, a.cells, "auto", a.name)''',
    '''    ap.add_argument("--slice", action="store_true")
    ap.add_argument("--ring", type=int, default=1)
    ap.add_argument("--ring-cells", type=int, default=8)
    a = ap.parse_args()
    fr = os.path.join(a.session, "frame.jpg")
    fd = find_object(fr, a.session, a.grid, a.cells, "auto", a.name, a.ring, a.ring_cells)''')
io.open(P, "w", encoding="utf-8").write(s)
print("matter.py patched")
