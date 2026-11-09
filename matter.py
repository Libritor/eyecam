"""Mind to Matter: the thing you looked at, as your brain registered it, as an object to hold.

The session (xr_session.py --mode matter) plays a clip or shows a picture in the headset; the
person pulls the trigger on the thing they want; the frame comes here, the thing is found in it
(a depth model's nearest surface, a foreground cut, or the colour that differs from the edges,
whichever looks most like one object), a grid is laid over it, and every square of the grid is
then shown as a flickering patch at the calibrated rate while the person looks at it (the
eye-as-a-camera scan: the response amplitude of each square is the brain's grey level for it).
Afterwards this module turns the brain's grid into prints:

  brain_relief.stl       a plate: the thing's silhouette raised a little, and on it the brain's
                         response as height, square by square (a lithophane of what was registered)
  object_seen.stl        the thing in 3-D from the depth model (its nearer parts stand higher),
                         scaled by the brain's response: parts the brain registered stand full
                         height, parts it did not sink toward the plate
  object_seen_base.stl + object_seen_mind.stl
                         the same object in two bodies for two spools: the base, and a 1 mm skin
                         over the squares the brain registered (load both in Bambu Studio as one
                         object, give the skin the second filament)
  preview.png            frame, what was found, the frame's grid, the brain's grid, both reliefs

Everything is written to <session>/print/ and copied to Downloads/mind_to_matter_prints/<session>/.
Depth runs in the anaconda Python (matter_depth.py, Depth Anything V2 small, offline); without it
the object is a pillow relief of its own silhouette.
"""
import base64
import json
import os
import shutil
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_PY = os.environ.get("MATTER_DEPTH_PY", r"C:\Users\alexa\anaconda3\python.exe")
SLICER = os.environ.get("MATTER_SLICER", r"C:\Users\alexa\printer_cam\questbag\slice_h2c.py")
OUT_COPY = os.path.join(os.path.expanduser("~"), "Downloads", "mind_to_matter_prints")
PRINT_MM = 100.0        # the longest side of a print
RES_MM = 0.3            # height-field resolution


def log(*a):
    print("[matter]", *a, flush=True)


# ----------------------------------------------------------------------------- the frame
def save_capture(session, data_url, rect=None):
    """The page's JPEG (a data URL) -> <session>/frame.jpg, cropped to the drawn clip."""
    import cv2
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with open(os.path.join(session, "frame_full.jpg"), "wb") as f:
        f.write(raw)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if rect and len(rect) == 4 and rect[2] > 8 and rect[3] > 8:
        x, y, w, h = [int(round(float(v))) for v in rect]
        x, y = max(0, x), max(0, y)
        img = img[y:min(img.shape[0], y + h), x:min(img.shape[1], x + w)]
    out = os.path.join(session, "frame.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def depth_map(frame_path, session):
    """Relative nearness from Depth Anything (larger = nearer), or None."""
    out = os.path.join(session, "depth.npy")
    try:
        r = subprocess.run([DEPTH_PY, "-I", os.path.join(HERE, "matter_depth.py"), frame_path, out],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(out):
            log("depth unavailable:", (r.stderr or r.stdout).strip()[-300:])
            return None
        return np.load(out)
    except Exception as exc:
        log("depth unavailable:", repr(exc))
        return None


def srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(img_bgr):
    b, g, r = [srgb_to_linear(img_bgr[..., i].astype(np.float64)) for i in range(3)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _largest(mask):
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _clean(mask, px):
    from scipy import ndimage as ndi
    k = max(1, int(px))
    m = ndi.binary_opening(mask, iterations=k)
    m = ndi.binary_closing(m, iterations=k)
    m = ndi.binary_fill_holes(m)
    return _largest(m)


def _objectness(mask):
    """How much a mask looks like one thing someone was looking at: a fair share of the
    frame, filling its box, near the centre, not spilling over several edges."""
    H, W = mask.shape
    area = mask.mean()
    if area < 0.015 or area > 0.75 or not mask.any():
        return -1.0
    ys, xs = np.nonzero(mask)
    fill = mask.sum() / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    cx, cy = xs.mean() / W, ys.mean() / H
    centred = 1.0 - min(1.0, 2.0 * float(np.hypot(cx - 0.5, cy - 0.5)))
    touch = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    return 0.45 * fill + 0.4 * centred + 0.15 * min(1.0, area / 0.2) - 0.25 * max(0, touch - 1) - (0.3 if area > 0.5 else 0.0)


def _candidates(img, lum, depth):
    """(name, mask) for each way of finding the thing."""
    import cv2
    H, W = lum.shape
    out = []
    px = max(2, int(round(min(H, W) / 150)))
    dn = None
    if depth is not None and depth.shape == lum.shape:
        lo, hi = np.percentile(depth, [2, 98])
        dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        d8 = (dn * 255).astype(np.uint8)
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(("depth", _clean(d8 > t, px)))
    # a foreground cut from a centred box, seeded by the near surface when there is one
    scale = min(1.0, 480.0 / W)
    small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    x0, y0, x1, y1 = int(w * 0.12), int(h * 0.1), int(w * 0.88), int(h * 0.92)
    gc[y0:y1, x0:x1] = cv2.GC_PR_BGD
    if dn is not None:
        dns = cv2.resize(dn, (w, h), interpolation=cv2.INTER_AREA)
        gc[(dns > 0.7) & (gc == cv2.GC_PR_BGD)] = cv2.GC_PR_FGD
    else:
        gc[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        fg = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        out.append(("grabcut", _clean(fg, px)))
    except cv2.error as exc:
        log("grabcut failed:", str(exc)[:120])
    # whatever differs in colour from the edges of the frame
    ring = max(4, int(min(H, W) * 0.03))
    edge = np.concatenate([img[:ring].reshape(-1, 3), img[-ring:].reshape(-1, 3),
                           img[:, :ring].reshape(-1, 3), img[:, -ring:].reshape(-1, 3)]).astype(np.float64)
    med = np.median(edge, axis=0)
    dist = np.linalg.norm(img.astype(np.float64) - med, axis=2)
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("colour", _clean(d8 > t, px)))
    return out, dn


def find_object(frame_path, out_dir, grid="7x5", max_cells=20, depth="auto", name="", ring=1, ring_cells=8):
    """Find the thing in the frame and lay the scan grid over it. Writes mask.png, depth.png,
    overlay.jpg and find.json beside the frame; returns the find with the overlay as a data URL."""
    import cv2
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot read " + frame_path)
    H, W = img.shape[:2]
    lum = luminance(img)
    dep = depth_map(frame_path, out_dir) if depth != "off" else None
    cands, dn = _candidates(img, lum, dep)
    scored = sorted(((_objectness(m), nm, m) for nm, m in cands), key=lambda t: -t[0])
    for s, nm, m in scored:
        log(f"candidate {nm}: objectness {s:.2f}, {m.mean() * 100:.0f}% of the frame")
    score, method, mask = scored[0]
    if score < 0:
        # nothing object-like: take the middle of the frame
        mask = np.zeros((H, W), bool); mask[int(H * 0.25):int(H * 0.75), int(W * 0.25):int(W * 0.75)] = True
        method = "centre"
    ys, xs = np.nonzero(mask)
    pad_x, pad_y = int(0.04 * (xs.max() - xs.min() + 1)), int(0.04 * (ys.max() - ys.min() + 1))
    x0, x1 = max(0, xs.min() - pad_x), min(W, xs.max() + 1 + pad_x)
    y0, y1 = max(0, ys.min() - pad_y), min(H, ys.max() + 1 + pad_y)
    bw, bh = x1 - x0, y1 - y0
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
    # files
    cv2.imwrite(os.path.join(out_dir, "mask.png"), (mask * 255).astype(np.uint8))
    if dn is not None:
        cv2.imwrite(os.path.join(out_dir, "depth.png"), cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    ov = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (80, 255, 120), max(2, W // 400))
    n = 0
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
    label = name.strip()
    head = (label + ": " if label else "") + f"{int(obj.sum())} squares + {len(bg)} around"
    cv2.putText(ov, head, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(ov, head, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 1100), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    small = ov if W <= 1280 else cv2.resize(ov, (1280, int(H * 1280 / W)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    found = dict(gridW=cols, gridH=rows, cells=cells, target=target.round(4).tolist(), coverage=cov.round(3).tolist(),
                 object_cells=[[int(gx), int(gy)] for gy in range(rows) for gx in range(cols) if obj[gy, gx]],
                 background_cells=[[int(a), int(b)] for a, b in bg], controls=controls, ring=ring,
                 grid_bbox=[float(gx0), float(gy0), float(gx0 + cols * cw), float(gy0 + rows * ch)],
                 bbox=[int(x0), int(y0), int(x1), int(y1)], frame_size=[int(W), int(H)], method=method,
                 objectness=round(float(score), 3), depth_ok=bool(dn is not None), mask_area=round(float(mask.mean()), 4),
                 label=label, cell_px=[round(cw, 1), round(ch, 1)], frame=os.path.basename(frame_path))
    with open(os.path.join(out_dir, "find.json"), "w") as f:
        json.dump(found, f, indent=1)
    found["overlay_data_url"] = data_url
    return found


# ----------------------------------------------------------------------------- meshes
def write_stl(path, verts, faces):
    """Binary STL from vertex and face arrays."""
    v = np.asarray(verts, np.float64); f = np.asarray(faces, np.int64)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True); nrm = np.where(ln > 0, nrm / np.maximum(ln, 1e-12), 0)
    rec = np.zeros(len(f), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = nrm; rec["v"] = tri
    with open(path, "wb") as fh:
        fh.write(b"mind to matter".ljust(80, b"\0")); fh.write(struct.pack("<I", len(f))); fh.write(rec.tobytes())
    return len(f)


def heightfield_mesh(Z, res):
    """A watertight block: top surface Z[iy, ix] (mm) over a flat bottom at z = 0."""
    h, w = Z.shape
    X, Y = np.meshgrid(np.arange(w) * res, np.arange(h) * res)
    n = h * w
    V = np.vstack([np.c_[X.ravel(), Y.ravel(), Z.ravel()], np.c_[X.ravel(), Y.ravel(), np.zeros(n)]])
    idx = np.arange(n).reshape(h, w)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    top = np.r_[np.c_[a, d, b], np.c_[a, c, d]]
    bottom = np.r_[np.c_[a, b, d], np.c_[a, d, c]] + n
    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]
    p, q = ring, np.roll(ring, -1)
    sides = np.r_[np.c_[p, q, q + n], np.c_[p, q + n, p + n]]
    return V, np.vstack([top, bottom, sides])


def voxel_mesh(vol, res, origin=(0.0, 0.0, 0.0)):
    """Marching cubes over a boolean volume [ix, iy, iz] -> vertices (mm), faces."""
    from skimage.measure import marching_cubes
    from scipy import ndimage as ndi
    v = ndi.gaussian_filter(np.pad(vol, 2).astype(np.float32), 0.7)
    verts, faces, _, _ = marching_cubes(v, level=0.5, spacing=(res, res, res))
    verts = verts - 2 * res + np.asarray(origin)
    faces = faces[:, ::-1]
    # one vertex per position, no degenerate triangles: what a closed mesh needs
    q = np.round(verts / (res * 1e-3)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    verts = verts[first]; faces = inv.reshape(-1)[faces] if inv.ndim > 1 else inv[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return verts, faces[keep]


def mesh_report(path):
    try:
        import trimesh
        m = trimesh.load(path, force="mesh")
        return dict(triangles=int(len(m.faces)), watertight=bool(m.is_watertight),
                    extent_mm=[round(float(v), 1) for v in m.extents], volume_cm3=round(float(abs(m.volume)) / 1000, 1))
    except Exception as exc:
        return dict(note="trimesh not available: " + repr(exc)[:80])


# ----------------------------------------------------------------------------- the prints
def make_prints(session, grid, found, slice_h2c=False, print_mm=PRINT_MM, res=RES_MM):
    """The brain's grid (gridH x gridW scores from reconstruct) -> the prints listed in the module doc."""
    import cv2
    from scipy import ndimage as ndi
    out = os.path.join(session, "print"); os.makedirs(out, exist_ok=True)
    gw, gh = found["gridW"], found["gridH"]
    x0, y0, x1, y1 = found["bbox"]
    g = np.nan_to_num(np.asarray(grid, float), nan=0.0)
    if g.shape != (gh, gw):
        raise ValueError(f"brain grid {g.shape} does not match the find's {gh}x{gw}")
    visited = np.zeros((gh, gw), bool)
    for gx, gy in found["cells"]:
        visited[gy, gx] = True
    vals = g[visited]
    lo, hi = (np.percentile(vals, 5), np.percentile(vals, 95)) if vals.size else (0.0, 1.0)
    ctl = found.get("controls") or {}
    if ctl:   # the black and white squares the person also looked at are the yardsticks
        b_, w_ = float(g[ctl["black"][1], ctl["black"][0]]), float(g[ctl["white"][1], ctl["white"][0]])
        if b_ < hi: lo = max(lo, b_)
        if w_ > lo: hi = min(hi, w_)
    bn = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1); bn[~visited] = 0.0
    mask = cv2.imread(os.path.join(session, "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    mask_c = mask[y0:y1, x0:x1]
    Bh, Bw = mask_c.shape
    # the brain's picture at the frame's resolution: the grid covers the thing's box and its ring,
    # so it is laid over the grid's box and the thing's box is cut out of it
    gb = found.get("grid_bbox") or [x0, y0, x1, y1]
    Gw, Gh = max(1, int(round(gb[2] - gb[0]))), max(1, int(round(gb[3] - gb[1])))
    full = cv2.resize(bn.astype(np.float32), (Gw, Gh), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(round(x0 - gb[0])), int(round(y0 - gb[1]))
    brain_hi = np.zeros((Bh, Bw), np.float32)
    sx0, sy0 = max(0, ox), max(0, oy); sx1, sy1 = min(Gw, ox + Bw), min(Gh, oy + Bh)
    brain_hi[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = full[sy0:sy1, sx0:sx1]
    cell_px = max(2.0, min(Gw / gw, Gh / gh))
    brain_hi = np.clip(ndi.gaussian_filter(brain_hi, cell_px * 0.2), 0, 1)
    # the object's relief: the depth model's nearness, or a pillow from the silhouette
    dpath = os.path.join(session, "depth.npy")
    if found.get("depth_ok") and os.path.exists(dpath):
        dep = np.load(dpath)[y0:y1, x0:x1]
        inside = dep[mask_c]
        dlo, dhi = np.percentile(inside, [3, 97]) if inside.size else (0, 1)
        near = np.clip((dep - dlo) / max(dhi - dlo, 1e-6), 0, 1)
        near = ndi.gaussian_filter(near, cell_px * 0.1)
        relief_kind = "depth model"
    else:
        dt = ndi.distance_transform_edt(mask_c)
        near = np.sqrt(dt / max(dt.max(), 1e-6))
        relief_kind = "pillow"
    # print geometry: the box's longest side = print_mm, a 3 mm margin, res mm per sample
    scale = (print_mm - 6.0) / max(Bw, Bh)
    Wp, Hp = int(round(Bw * scale / res)), int(round(Bh * scale / res))
    marg = int(round(3.0 / res))
    def to_print(a, nearest=False):
        r = cv2.resize(a.astype(np.float32), (Wp, Hp), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        return np.pad(r, marg)
    M = to_print(mask_c, nearest=True) > 0.5
    M = ndi.binary_opening(M, iterations=2)
    B = np.clip(to_print(brain_hi), 0, 1) * M
    N = np.clip(to_print(near), 0, 1) * M
    base, plinth = 2.0, 0.8
    Z_relief = base + plinth * M + 5.0 * B
    Z_relief = ndi.gaussian_filter(Z_relief, 1.0)
    Z_obj = base + plinth * M + 10.0 * N * (0.35 + 0.65 * B)
    Z_obj = np.where(M, ndi.gaussian_filter(Z_obj, 1.0), base)
    files = {"stl": [], "other": []}
    V, F = heightfield_mesh(Z_relief[::-1], res)            # rows run bottom-up in a print
    p = os.path.join(out, "brain_relief.stl"); write_stl(p, V, F); files["stl"].append(p)
    V, F = heightfield_mesh(Z_obj[::-1], res)
    p = os.path.join(out, "object_seen.stl"); write_stl(p, V, F); files["stl"].append(p)
    # two bodies for two spools: the object with its skin lowered where the brain registered, and the skin
    seen = M & (B >= 0.5)
    skin = 1.0
    Z_base = Z_obj - skin * seen
    V, F = heightfield_mesh(Z_base[::-1], res)
    p = os.path.join(out, "object_seen_base.stl"); write_stl(p, V, F); files["stl"].append(p)
    if seen.any():
        zmax = float(Z_obj.max()) + res
        nz = int(np.ceil(zmax / res)) + 2
        zs = (np.arange(nz) + 0.5) * res
        Zb = Z_base[::-1].T[:, :, None]; Zt = Z_obj[::-1].T[:, :, None]; S = seen[::-1].T[:, :, None]
        vol = S & (zs[None, None, :] >= Zb) & (zs[None, None, :] < Zt)
        if vol.any():
            V, F = voxel_mesh(vol, res, origin=(0.0, 0.0, 0.0))
            p = os.path.join(out, "object_seen_mind.stl"); write_stl(p, V, F); files["stl"].append(p)
    # the preview
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    frame = cv2.cvtColor(cv2.imread(os.path.join(session, found.get("frame", "frame.jpg"))), cv2.COLOR_BGR2RGB)
    overlay = cv2.cvtColor(cv2.imread(os.path.join(session, "overlay.jpg")), cv2.COLOR_BGR2RGB)
    ls = LightSource(azdeg=315, altdeg=45)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), dpi=110)
    axs[0, 0].imshow(frame); axs[0, 0].set_title("what was on screen when the trigger was pulled")
    axs[0, 1].imshow(overlay); axs[0, 1].set_title(f"what was found ({found['method']}) and the {gw}x{gh} grid")
    tgt = np.asarray(found["target"], float)
    axs[0, 2].imshow(np.where(visited, tgt, np.nan), cmap="gray", vmin=0, vmax=1); axs[0, 2].set_title("the frame's squares (grey level shown; B/W = controls)")
    axs[1, 0].imshow(np.where(visited, bn, np.nan), cmap="magma", vmin=0, vmax=1); axs[1, 0].set_title("the brain's squares (response, 0..1)")
    for ax in (axs[0, 2], axs[1, 0]):
        ax.set_facecolor("#b8c4cc")            # so a white square shows against the unscanned ones
        for tag, key in (("B", "black"), ("W", "white")):
            if ctl.get(key):
                ax.text(ctl[key][0], ctl[key][1], tag, ha="center", va="center", color="red", fontsize=12, fontweight="bold")
    for ax, Z, t in ((axs[1, 1], Z_relief, "brain_relief.stl"), (axs[1, 2], Z_obj, f"object_seen.stl ({relief_kind})")):
        ax.imshow(ls.shade(Z, cmap=plt.cm.gray, vert_exag=1.5, blend_mode="overlay", dx=res, dy=res), origin="lower"); ax.set_title(t)
    for ax in axs.ravel(): ax.set_axis_off()
    r = None
    if tgt.shape == bn.shape and visited.sum() > 2:
        r = float(np.corrcoef(tgt[visited], g[visited])[0, 1])
    fig.suptitle(("Mind to Matter: " + found["label"] if found.get("label") else "Mind to Matter")
                 + (f"   r(brain, frame) = {r:.2f}" if r is not None else ""), fontsize=14)
    plt.tight_layout(); pv = os.path.join(out, "preview.png"); plt.savefig(pv, bbox_inches="tight"); plt.close(fig)
    files["other"].append(pv)
    np.save(os.path.join(out, "brain_grid_norm.npy"), bn)
    reports = {os.path.basename(p): mesh_report(p) for p in files["stl"]}
    for k, v in reports.items():
        log(k, v)
    if slice_h2c and os.path.exists(SLICER):
        for p in (os.path.join(out, "object_seen.stl"), os.path.join(out, "brain_relief.stl")):
            d = os.path.join(out, "h2c_" + os.path.splitext(os.path.basename(p))[0])
            try:
                rr = subprocess.run([DEPTH_PY, "-I", SLICER, p, "--out", d, "--set", "brim_type=no_brim"],
                                    capture_output=True, text=True, timeout=2700)
                line = next((l for l in rr.stdout.splitlines() if l.startswith("sliced")), rr.stdout.strip()[-200:])
                log(line)
                reports[os.path.basename(p)]["slice"] = line
                if os.path.exists(os.path.join(d, "sliced.3mf")):
                    dst = os.path.join(out, os.path.splitext(os.path.basename(p))[0] + "_h2c.3mf")
                    shutil.copy(os.path.join(d, "sliced.3mf"), dst); files["other"].append(dst)
            except Exception as exc:
                log("slice failed:", repr(exc)[:200])
    with open(os.path.join(out, "prints.json"), "w") as f:
        json.dump(dict(reports=reports, r=r, relief=relief_kind, print_mm=print_mm, res_mm=res,
                       seen_fraction=round(float(seen.sum() / max(M.sum(), 1)), 3)), f, indent=1)
    dst_dir = os.path.join(OUT_COPY, os.path.basename(os.path.normpath(session)))
    os.makedirs(dst_dir, exist_ok=True)
    for p in files["stl"] + files["other"] + [os.path.join(out, "prints.json")]:
        shutil.copy(p, dst_dir)
    files["copied_to"] = dst_dir
    files["reports"] = reports
    files["r"] = r
    return files


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="re-run the find and the prints on a recorded session")
    ap.add_argument("session")
    ap.add_argument("--find-only", action="store_true")
    ap.add_argument("--grid", default="7x5")
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--name", default="")
    ap.add_argument("--slice", action="store_true")
    ap.add_argument("--ring", type=int, default=1)
    ap.add_argument("--ring-cells", type=int, default=8)
    a = ap.parse_args()
    fr = os.path.join(a.session, "frame.jpg")
    fd = find_object(fr, a.session, a.grid, a.cells, "auto", a.name, a.ring, a.ring_cells)
    log(f"{fd['method']}: {len(fd['cells'])} squares on a {fd['gridW']}x{fd['gridH']} grid")
    if not a.find_only:
        gp = os.path.join(a.session, "reconstruction_grid.npy")
        grid = np.load(gp) if os.path.exists(gp) else np.asarray(fd["target"])
        print(make_prints(a.session, grid, fd, a.slice))
