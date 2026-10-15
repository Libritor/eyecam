"""3D objects for the colour scan: unfold to 2D, scan, paint the result back on.

The colour scan (bwb.py) decodes a 2D grid of cells. A 3D object is scanned by
making that grid its texture map:

  1. The object's surface is cut into charts: rectangular patches of n_u x n_v
     cells, each a map (u, v) in [0, 1]^2 -> R^3 (a cube face, a sphere face
     of a cube-sphere, the whole of a torus, ...). Every cell has a target
     colour, chosen by the object's recipe.
  2. The charts are packed into the scan grid (the "layout"): 'atlas' packs
     them tightly (a cube's six faces in 3 x 2 blocks, no wasted cells),
     'net' unfolds a cube-like object into the familiar cross so neighbouring
     cells on the object stay neighbours on the grid. Grid cells that belong
     to no chart are holes: never shown, never scanned (mask False).
  3. The subject scans the grid exactly like a 2D picture (monochrome
     frequency-phase codes, one cell at a time) and bwb.py decodes it.
  4. The decoded grid IS the object's texture: each cell's colour (and its
     certainty) is painted back on that cell's patch of surface, and the
     object is written out as a textured OBJ, rendered from several sides, and
     put in an interactive viewer (mesh_viewer.html) next to the object shown.

Objects (spec 'mesh:<name>[:n]', n = cells along a face edge):
  die     a white die with black pips and a red one (3 x 3 cells a face, 54)
  rubik   a scrambled Rubik's cube: six colours, 3 x 3 stickers a face (54)
  globe   the Earth on a sphere: ocean, land, ice (4 x 4 a face, 96 cells)
  donut   a torus: brown dough, pink icing, sprinkles (16 x 6 cells, 96)
  obj:<file.obj>[:n]   any star-shaped OBJ mesh, wrapped on a cube-sphere by
          casting rays from its centre; cell colours from its materials
          (Kd or a map_Kd texture) or vertex colours, snapped to the palette.

    python mesh3d.py die --out runs/die_preview     # preview an object
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bwb

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- geometry
# cube faces seen from outside: u to the right, v up; outward normals
# front +Z, right +X, back -Z, left -X, top +Y, bottom -Y
FACES = ["front", "right", "back", "left", "top", "bottom"]


def cube_point(face, u, v):
    """Point on the cube [-1, 1]^3 for face coordinates u, v in [0, 1]."""
    a, b = 2 * np.asarray(u, float) - 1, 2 * np.asarray(v, float) - 1
    one = np.ones_like(a)
    x, y, z = {"front": (a, b, one), "back": (-a, b, -one),
               "right": (one, b, -a), "left": (-one, b, a),
               "top": (a, one, -b), "bottom": (a, -one, b)}[face]
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def sphere_point(face, u, v):
    """Equal-angle cube-sphere: cells of a face span equal angles."""
    tu = (np.tan((2 * np.asarray(u, float) - 1) * np.pi / 4) + 1) / 2
    tv = (np.tan((2 * np.asarray(v, float) - 1) * np.pi / 4) + 1) / 2
    p = cube_point(face, tu, tv)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def rounded_point(face, u, v, r=0.18):
    """A cube with rounded edges (a die): the cube pulled toward the sphere
    near its edges only."""
    p = cube_point(face, u, v)
    inner = np.clip(p, -(1 - r), 1 - r)
    d = p - inner
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return inner + np.where(n > 0, d / np.maximum(n, 1e-9) * r, 0)


def torus_point(u, v, R=1.0, r=0.42):
    """u around the ring, v around the tube (v = 0.5 is the outer equator,
    v = 0.75 the top)."""
    a = 2 * np.pi * np.asarray(u, float)
    b = 2 * np.pi * (np.asarray(v, float) - 0.5)
    ring = R + r * np.cos(b)
    return np.stack(np.broadcast_arrays(ring * np.sin(a), r * np.sin(b), ring * np.cos(a)), axis=-1)


# ---------------------------------------------------------------- charts
@dataclass
class Chart:
    """n_u x n_v cells of surface. names[j][i] is cell (i, j)'s palette
    colour; row j = 0 is the top (v = 1), as in an image."""
    name: str
    nu: int
    nv: int
    fn: object                      # (u, v) -> (..., 3)
    names: list = field(default_factory=list)
    x0: int = 0                     # placement in the scan grid
    y0: int = 0

    def cell_uv(self, i, j):
        """(u0, u1, v0, v1) of cell (i, j)."""
        return i / self.nu, (i + 1) / self.nu, 1 - (j + 1) / self.nv, 1 - j / self.nv


class Object3D:
    """Charts + their placement in the scan grid."""

    def __init__(self, name, charts, colours, sub=1, layout="atlas", note=""):
        self.name, self.charts, self.sub, self.note = name, charts, sub, note
        used = {c for ch in charts for row in ch.names for c in row} - {"black"}
        self.colours = [c for c in colours if c in used] or list(colours)
        self.layout = place(charts, layout)
        hs = [ch.y0 + ch.nv for ch in charts]
        ws = [ch.x0 + ch.nu for ch in charts]
        self.grid_w, self.grid_h = max(ws), max(hs)
        self.mask = np.zeros((self.grid_h, self.grid_w), bool)
        self.cells = []                 # (gx, gy, chart index, i, j)
        for k, ch in enumerate(charts):
            for j in range(ch.nv):
                for i in range(ch.nu):
                    self.mask[ch.y0 + j, ch.x0 + i] = True
                    self.cells.append((ch.x0 + i, ch.y0 + j, k, i, j))
        self._norm = None

    # -- target
    def target(self, colours=None):
        """Class grid for the scan (holes 0, never visited)."""
        names, _ = bwb.classes_of(colours or self.colours)
        idx = {n: i for i, n in enumerate(names)}
        g = np.zeros((self.grid_h, self.grid_w), int)
        for gx, gy, k, i, j in self.cells:
            c = self.charts[k].names[j][i]
            if c not in idx:
                raise ValueError(f"{self.name} needs {c!r}: add it to --bwb-colours "
                                 f"(it has {', '.join(names[1:])})")
            g[gy, gx] = idx[c]
        return g

    # -- geometry
    def _points(self, ch, u, v):
        p = ch.fn(u, v)
        if self._norm is None:          # centre and fit in the unit sphere
            P = np.concatenate([c.fn(*np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)))
                                .reshape(-1, 3) for c in self.charts])
            ctr = (P.max(0) + P.min(0)) / 2
            self._norm = (ctr, 1 / np.linalg.norm(P - ctr, axis=1).max())
        ctr, s = self._norm
        return (p - ctr) * s

    def cell_quads(self, sub=None):
        """For every cell (in self.cells order): (sub, sub, 4, 3) sub-quads,
        corners counter-clockwise seen from outside."""
        s = sub or self.sub
        out = []
        for gx, gy, k, i, j in self.cells:
            ch = self.charts[k]
            u0, u1, v0, v1 = ch.cell_uv(i, j)
            uu, vv = np.meshgrid(np.linspace(u0, u1, s + 1), np.linspace(v0, v1, s + 1))
            P = self._points(ch, uu, vv)                       # (s+1, s+1, 3), [v][u]
            Q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], axis=2)
            out.append(Q)                                      # (s, s, 4, 3)
        return out

    def geometry_json(self, sub=None):
        """Compact geometry for mesh_view.js: per cell its grid position and
        its sub-quads (3 decimals)."""
        return dict(grid=[self.grid_w, self.grid_h], sub=sub or self.sub,
                    cells=[dict(gx=gx, gy=gy, chart=self.charts[k].name,
                                q=np.round(Q, 3).reshape(-1, 12).tolist())
                           for (gx, gy, k, i, j), Q in zip(self.cells, self.cell_quads(sub))],
                    charts=[dict(name=c.name, x=c.x0, y=c.y0, w=c.nu, h=c.nv) for c in self.charts])


def place(charts, layout="atlas"):
    """Set each chart's x0, y0. 'net' (six cube faces): the cross
             top
      left front right back
             bottom
    'atlas': rows of charts, tallest first, packed left to right into a
    width close to the square root of the total area (a cube: 3 x 2 faces)."""
    if layout == "net":
        if [c.name for c in charts] != FACES or len({(c.nu, c.nv) for c in charts}) != 1:
            raise ValueError("--mesh-layout net needs a cube-like object (six equal faces)")
        n = charts[0].nu
        pos = dict(top=(1, 0), left=(0, 1), front=(1, 1), right=(2, 1), back=(3, 1), bottom=(1, 2))
        for c in charts:
            c.x0, c.y0 = pos[c.name][0] * n, pos[c.name][1] * n
        return "net"
    if layout != "atlas":
        raise ValueError(f"unknown layout {layout!r} (atlas, net)")
    area = sum(c.nu * c.nv for c in charts)
    width = max(max(c.nu for c in charts), int(math.ceil(math.sqrt(area * 1.5))))
    # equal square charts: the most nearly square grid of whole charts
    if len({(c.nu, c.nv) for c in charts}) == 1:
        k = len(charts)
        cols = min(range(1, k + 1), key=lambda q: (abs(q * charts[0].nu - math.ceil(k / q) * charts[0].nv)
                                                   + (q * math.ceil(k / q) - k) * charts[0].nu, -q))
        width = cols * charts[0].nu
    x = y = row_h = 0
    for c in sorted(charts, key=lambda c: -c.nv):
        if x + c.nu > width:
            x, y, row_h = 0, y + row_h, 0
        c.x0, c.y0 = x, y
        x += c.nu
        row_h = max(row_h, c.nv)
    return "atlas"


# ---------------------------------------------------------------- objects
def _face_charts(n, fn, colour_of):
    """Six cube-face charts; colour_of(face, i, j, centre point) -> name."""
    out = []
    for f in FACES:
        ch = Chart(f, n, n, lambda u, v, f=f: fn(f, u, v))
        ch.names = [[colour_of(f, i, j, fn(f, (i + 0.5) / n, 1 - (j + 0.5) / n))
                     for i in range(n)] for j in range(n)]
        out.append(ch)
    return out


PIPS = {1: [(1, 1)], 2: [(0, 2), (2, 0)], 3: [(0, 2), (1, 1), (2, 0)],
        4: [(0, 0), (0, 2), (2, 0), (2, 2)], 5: [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        6: [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]}       # (row, col) on 3 x 3
DIE_FACE = dict(front=1, top=2, right=3, left=4, bottom=5, back=6)  # opposite faces sum to 7


def make_die(n=3, layout="atlas"):
    n = max(3, n)
    at = [1, n // 2, n - 2] if n > 3 else [0, 1, 2]

    def colour(face, i, j, p):
        k = DIE_FACE[face]
        if any((at[r], at[c]) == (j, i) for r, c in PIPS[k]):
            return "red" if k == 1 else "black"
        return "white"
    return Object3D("die", _face_charts(n, rounded_point, colour), ["white", "red"],
                    sub=3, layout=layout,
                    note="white die, black pips, a red one: opposite faces sum to 7")


def make_rubik(n=3, layout="atlas", seed=7):
    """A scrambled cube: stickers shuffled (nine of each colour), centres
    fixed, as a solved cube's face colours."""
    centre = dict(front="green", back="blue", right="red", left="orange", top="white",
                  bottom="yellow")
    rng = np.random.default_rng(seed)
    pool = [c for f in FACES for c in [centre[f]] * (n * n - 1)]
    rng.shuffle(pool)
    it = iter(pool)
    c0 = n // 2

    def colour(face, i, j, p):
        return centre[face] if (i, j) == (c0, c0) and n % 2 else next(it)
    return Object3D("rubik", _face_charts(n, cube_point, colour),
                    ["white", "yellow", "red", "orange", "blue", "green"], sub=1, layout=layout,
                    note="scrambled stickers, centres fixed: six colours")


# coarse continents (lon, lat), enough for 20-degree cells
CONTINENTS = [
    [(-168, 66), (-162, 70), (-140, 70), (-125, 70), (-95, 72), (-80, 73), (-65, 62), (-60, 55),
     (-56, 52), (-66, 45), (-70, 42), (-75, 38), (-76, 35), (-81, 31), (-80, 25), (-84, 30),
     (-90, 30), (-97, 27), (-97, 21), (-92, 18), (-88, 21), (-87, 16), (-83, 10), (-78, 8),
     (-85, 10), (-92, 14), (-105, 20), (-110, 24), (-117, 32), (-124, 40), (-124, 48),
     (-131, 54), (-140, 60), (-150, 61), (-160, 58), (-165, 62)],                # N. America
    [(-80, 9), (-75, 11), (-62, 11), (-52, 5), (-50, 0), (-35, -5), (-39, -14), (-41, -22),
     (-48, -26), (-53, -34), (-58, -38), (-63, -41), (-65, -46), (-68, -52), (-72, -53),
     (-74, -45), (-73, -37), (-71, -30), (-70, -18), (-76, -14), (-81, -5), (-80, 0),
     (-78, 2), (-77, 7)],                                                         # S. America
    [(-17, 21), (-16, 12), (-12, 7), (-8, 4), (0, 5), (9, 4), (10, -1), (12, -5), (13, -12),
     (12, -18), (15, -27), (18, -34), (20, -35), (27, -34), (33, -28), (35, -23), (40, -15),
     (40, -10), (39, -5), (42, 0), (51, 11), (44, 11), (43, 13), (38, 18), (34, 27), (32, 31),
     (25, 32), (20, 31), (10, 34), (10, 37), (0, 36), (-6, 36), (-10, 30), (-13, 27)],  # Africa
    [(-10, 36), (-9, 43), (-2, 44), (-5, 48), (0, 50), (5, 53), (8, 57), (5, 62), (15, 69),
     (25, 71), (40, 68), (45, 68), (60, 70), (70, 73), (80, 73), (100, 77), (115, 74),
     (140, 72), (160, 70), (180, 69), (180, 65), (170, 60), (162, 56), (156, 51), (155, 58),
     (143, 59), (140, 53), (135, 43), (129, 35), (127, 40), (122, 40), (121, 31), (118, 24),
     (110, 21), (108, 16), (109, 11), (105, 9), (103, 1), (100, 6), (98, 16), (94, 17),
     (92, 22), (88, 22), (80, 15), (77, 8), (73, 17), (72, 21), (66, 25), (57, 26), (56, 24),
     (59, 22), (52, 16), (43, 13), (39, 21), (35, 28), (34, 31), (36, 36), (28, 36), (26, 40),
     (23, 37), (20, 40), (16, 38), (12, 44), (8, 44), (3, 43), (-1, 37), (-6, 36)],  # Eurasia
    [(114, -22), (122, -17), (130, -12), (137, -12), (142, -11), (146, -19), (153, -26),
     (151, -34), (146, -39), (140, -38), (135, -35), (130, -32), (115, -34)],   # Australia
]
GREENLAND = [(-73, 78), (-60, 82), (-30, 83), (-20, 80), (-20, 70), (-40, 65), (-45, 60),
             (-50, 64), (-55, 70), (-68, 76)]


def _inside(lon, lat, poly):
    """Even-odd point-in-polygon on (lon, lat) arrays."""
    P = np.asarray(poly, float)
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    lon, lat = np.asarray(lon, float)[..., None], np.asarray(lat, float)[..., None]
    cross = ((y0 > lat) != (y1 > lat)) & (lon < (x1 - x0) * (lat - y0) / (y1 - y0 + 1e-12) + x0)
    return cross.sum(-1) % 2 == 1


def earth(p):
    """'blue' ocean, 'green' land, 'white' ice at unit vectors p (..., 3);
    lon 0 faces +Z (the front), east to the right."""
    p = np.asarray(p, float)
    lat = np.degrees(np.arcsin(np.clip(p[..., 1], -1, 1)))
    lon = np.degrees(np.arctan2(p[..., 0], p[..., 2]))
    land = np.zeros(lat.shape, bool)
    for poly in CONTINENTS:
        land |= _inside(lon, lat, poly)
    ice = (lat > 78) | (lat < -64) | _inside(lon, lat, GREENLAND)
    return np.where(ice, 2, np.where(land, 1, 0))


def make_globe(n=4, layout="atlas"):
    def colour(face, i, j, p):
        # majority of 5 x 5 samples over the cell, so a coastline cell takes
        # whichever covers more of it
        u = (i + (np.arange(5) + 0.5) / 5) / n
        v = 1 - (j + (np.arange(5) + 0.5) / 5) / n
        cls = earth(sphere_point(face, *np.meshgrid(u, v))).ravel()
        return ["blue", "green", "white"][int(np.bincount(cls, minlength=3).argmax())]
    return Object3D("globe", _face_charts(n, sphere_point, colour), ["blue", "green", "white"],
                    sub=3, layout=layout, note="the Earth: ocean, land, ice; Africa faces front")


def make_donut(n=6, layout="atlas", seed=3):
    """A torus, n cells around the tube and 8n/3 around the ring: brown
    dough underneath, pink icing on top, white and yellow sprinkles."""
    nv = max(4, n)
    nu = max(8, int(round(nv * 8 / 3)))
    rng = np.random.default_rng(seed)
    ch = Chart("torus", nu, nv, torus_point)
    names = []
    for j in range(nv):
        v = 1 - (j + 0.5) / nv                          # tube angle: 0.75 = top
        top = np.sin(2 * np.pi * (v - 0.5)) > -0.25
        row = []
        for i in range(nu):
            if not top:
                row.append("brown")
            else:
                r = rng.random()
                row.append("white" if r < 0.12 else "yellow" if r < 0.22 else "pink")
        names.append(row)
    ch.names = names
    return Object3D("donut", [ch], ["pink", "brown", "white", "yellow"], sub=3, layout=layout,
                    note="pink icing with sprinkles on brown dough")


# ---------------------------------------------------------------- OBJ import
def read_obj(path):
    """Triangles (F, 3, 3), per-triangle (3, 2) uv or None, per-triangle
    material, per-triangle vertex colours (F, 3, 3) or None, and materials
    {name: dict(kd, tex)}."""
    V, VT, VC, tris, tri_v, uvs, mats = [], [], [], [], [], [], []
    materials, cur = {}, None
    base = os.path.dirname(os.path.abspath(path))
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
            VC.append([float(x) for x in p[4:7]] if len(p) >= 7 else None)
        elif p[0] == "vt":
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl":
            cur = p[1] if len(p) > 1 else None
        elif p[0] == "mtllib":
            materials.update(read_mtl(os.path.join(base, " ".join(p[1:]))))
        elif p[0] == "f":
            idx = []
            for w in p[1:]:
                a = w.split("/")
                vi = int(a[0])
                ti = int(a[1]) if len(a) > 1 and a[1] else None
                idx.append((vi - 1 if vi > 0 else len(V) + vi,
                            None if ti is None else (ti - 1 if ti > 0 else len(VT) + ti)))
            for k in range(1, len(idx) - 1):             # fan
                tri = [idx[0], idx[k], idx[k + 1]]
                tris.append([V[a] for a, _ in tri])
                tri_v.append([a for a, _ in tri])
                uvs.append([VT[t] for _, t in tri] if all(t is not None for _, t in tri) else None)
                mats.append(cur)
    vc = None
    if VC and all(c is not None for c in VC):
        vc = np.asarray(VC, float)[np.asarray(tri_v, int)]
    return np.asarray(tris, float), uvs, mats, vc, materials


def read_mtl(path):
    out, cur = {}, None
    if not os.path.exists(path):
        return out
    base = os.path.dirname(path)
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "newmtl":
            cur = p[1]
            out[cur] = dict(kd=None, tex=None)
        elif cur and p[0] == "Kd":
            out[cur]["kd"] = [float(x) for x in p[1:4]]
        elif cur and p[0] == "map_Kd":
            tp = os.path.join(base, p[-1])
            if os.path.exists(tp):
                out[cur]["tex"] = np.asarray(Image.open(tp).convert("RGB"), float) / 255
    return out


def ray_hits(origin, dirs, T):
    """Farthest hit of each ray (origin + t d) on triangles T (F, 3, 3):
    (t, triangle index, barycentric b1, b2); t = nan on a miss."""
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    s = origin - T[:, 0]
    q = np.cross(s, e1)
    best_t = np.full(len(dirs), np.nan)
    best_f = np.zeros(len(dirs), int)
    best_b = np.zeros((len(dirs), 2))
    for a in range(0, len(dirs), 64):
        d = dirs[a:a + 64]
        p = np.cross(d[:, None, :], e2[None])                      # (R, F, 3)
        det = np.einsum("rfk,fk->rf", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        b1 = np.einsum("rfk,fk->rf", p, s) * inv
        b2 = np.einsum("rk,fk->rf", d, q) * inv
        t = np.einsum("fk,fk->f", e2, q)[None] * inv
        hit = ok & (b1 >= -1e-9) & (b2 >= -1e-9) & (b1 + b2 <= 1 + 1e-9) & (t > 1e-9)
        tt = np.where(hit, t, -np.inf)
        f = tt.argmax(1)
        r = np.arange(len(d))
        has = hit[r, f]
        best_t[a:a + 64] = np.where(has, tt[r, f], np.nan)
        best_f[a:a + 64] = f
        best_b[a:a + 64] = np.stack([b1[r, f], b2[r, f]], 1)
    return best_t, best_f, best_b


def make_obj(path, n=4, layout="atlas", colours=None):
    """Wrap a star-shaped OBJ mesh: the cube-sphere's rays from the mesh's
    centre land on its outermost surface; a cell's colour is the surface
    colour where its centre ray lands, snapped to the palette."""
    T, uvs, mats, vc, materials = read_obj(path)
    if not len(T):
        raise ValueError(f"{path}: no faces")
    pts = T.reshape(-1, 3)
    centre = (pts.max(0) + pts.min(0)) / 2

    def surface_rgb(f, b):
        w = np.array([1 - b[0] - b[1], b[0], b[1]])
        if vc is not None:
            return w @ vc[f]
        m = materials.get(mats[f]) if mats[f] else None
        if m and m["tex"] is not None and uvs[f] is not None:
            uv = w @ np.asarray(uvs[f])
            tex = m["tex"]
            x = int(np.clip(uv[0] % 1.0 * tex.shape[1], 0, tex.shape[1] - 1))
            y = int(np.clip((1 - uv[1] % 1.0) * tex.shape[0], 0, tex.shape[0] - 1))
            return tex[y, x]
        if m and m["kd"] is not None:
            return np.asarray(m["kd"])
        return None

    names_all = [c for c in bwb.PALETTE if c != "black"] if colours is None else list(colours)
    pal = ["black"] + names_all
    rgb_pal = np.array([bwb.PALETTE[c] for c in pal], float)
    has_colour = vc is not None or any(m["kd"] is not None or m["tex"] is not None
                                       for m in materials.values())
    cache = {}

    def radial(face, u, v):
        key = (face, np.asarray(u).tobytes(), np.asarray(v).tobytes())
        if key not in cache:
            d = sphere_point(face, u, v)
            sh = d.shape[:-1]
            t, f, b = ray_hits(centre, d.reshape(-1, 3), T)
            fill = np.nanmedian(t) if np.isfinite(t).any() else 1.0
            t = np.where(np.isfinite(t), t, fill)
            cache[key] = ((centre + d.reshape(-1, 3) * t[:, None]).reshape(sh + (3,)), f, b, sh)
        return cache[key]

    def fn_of(face):
        return lambda u, v: radial(face, u, v)[0]

    charts = []
    for face in FACES:
        ch = Chart(face, n, n, fn_of(face))
        uc = (np.arange(n) + 0.5) / n
        uu, vv = np.meshgrid(uc, 1 - uc)                           # row j = top first
        P, f, b, sh = radial(face, uu, vv)
        names = []
        for k in range(n * n):
            if has_colour:
                rgb = surface_rgb(f[k], b[k])
                rgb = rgb if rgb is not None else np.ones(3)
                names.append(pal[int(np.argmin(((rgb_pal - rgb) ** 2).sum(1)))])
            else:                                                  # height bands
                y = P.reshape(-1, 3)[k, 1]
                lo, hi = pts[:, 1].min(), pts[:, 1].max()
                band = int(np.clip((y - lo) / (hi - lo + 1e-9) * 3, 0, 2))
                names.append(["blue", "white", "red"][band])
        ch.names = [names[j * n:(j + 1) * n] for j in range(n)]
        charts.append(ch)
    used = [c for c in dict.fromkeys(x for ch in charts for row in ch.names for x in row)
            if c != "black"]
    return Object3D(os.path.splitext(os.path.basename(path))[0], charts, used, sub=3,
                    layout=layout, note=f"{os.path.basename(path)} wrapped on a cube-sphere")


OBJECTS = dict(die=make_die, rubik=make_rubik, globe=make_globe, donut=make_donut)


def is_mesh(spec):
    return isinstance(spec, str) and spec.startswith("mesh:")


def from_spec(spec, layout="atlas", colours=None):
    """'mesh:die', 'mesh:globe:5', 'mesh:obj:path/to/thing.obj[:n]'."""
    body = spec[5:] if is_mesh(spec) else spec
    if body.startswith("obj:"):
        path, n = body[4:], 4
        head, _, tail = path.rpartition(":")
        if head and tail.isdigit():
            path, n = head, int(tail)
        return make_obj(path, n, layout, colours)
    name, _, n = body.partition(":")
    if name not in OBJECTS:
        raise ValueError(f"unknown object {name!r}: {', '.join(OBJECTS)} or obj:<file.obj>")
    return OBJECTS[name](int(n), layout) if n else OBJECTS[name](layout=layout)


def scan_cells(mask, passes=1):
    """Visit order for a grid with holes: serpentine over the object's cells
    only, odd passes bottom to top (as the page's own full-grid passes)."""
    gh, gw = mask.shape
    out = []
    for ps in range(max(1, passes)):
        for ry in range(gh):
            gy = gh - 1 - ry if ps % 2 else ry
            xs = list(range(gw))
            if (ry + ps) % 2:
                xs.reverse()
            out += [[gx, gy] for gx in xs if mask[gy, gx]]
    return out


# ---------------------------------------------------------------- render
VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551],
                    [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]])


def certainty_rgb(c, k):
    """Certainty 1/k (chance) .. 1 on viridis."""
    x = np.clip((np.asarray(c, float) - 1 / k) / (1 - 1 / k), 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    f = (x - i)[..., None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f


def view_matrix(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry


LIGHT = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])


def render(obj, cell_rgb, yaw=-0.6, pitch=0.45, size=320, outline=(), quads=None,
           bg=(24, 24, 27), ss=2):
    """Shaded view of the object with each cell in cell_rgb[k] (0..1), cells
    in `outline` ringed red. Painter's algorithm with back-face culling."""
    quads = quads if quads is not None else obj.cell_quads()
    W = size * ss
    img = Image.new("RGB", (W, W), bg)
    d = ImageDraw.Draw(img)
    R = view_matrix(yaw, pitch)
    D, f = 3.4, W * 1.12
    items = []
    s = quads[0].shape[0]
    bad = set(outline)
    for k, Q in enumerate(quads):
        P = Q.reshape(-1, 4, 3) @ R.T
        n = np.cross(P[:, 1] - P[:, 0], P[:, 3] - P[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        for q in range(len(P)):
            c = P[q].mean(0)
            if n[q] @ (c - np.array([0, 0, D])) >= 0:     # faces away from the camera
                continue
            a, b = divmod(q, s)
            edges = (a == 0, b == s - 1, a == s - 1, b == 0)  # v-low, u-high, v-high, u-low
            items.append((c[2], k, P[q], n[q], edges))
    items.sort(key=lambda t: t[0])
    for z, k, P, n, edges in items:
        xy = [(W / 2 + f * x / (D - zz), W / 2 - f * y / (D - zz)) for x, y, zz in P]
        shade = 0.38 + 0.62 * max(0.0, float(n @ LIGHT))
        rgb = np.asarray(cell_rgb[k], float)
        col = tuple(int(255 * min(1, v * shade + 0.04 * shade)) for v in rgb)
        d.polygon(xy, fill=col)
        line = (235, 45, 45) if k in bad else tuple(int(v * 0.55) for v in col)
        width = 3 * ss if k in bad else max(1, ss // 2)
        for e, on in enumerate(edges):
            if on:
                d.line([xy[e], xy[(e + 1) % 4]], fill=line, width=width)
    return img.resize((size, size), Image.LANCZOS)


def atlas_image(obj, grid_rgb, cell=26, outline=(), bg=(24, 24, 27)):
    """The 2D scan grid: each chart outlined, holes left dark."""
    gh, gw = obj.grid_h, obj.grid_w
    img = Image.new("RGB", (gw * cell + 2, gh * cell + 2), bg)
    d = ImageDraw.Draw(img)
    for gy in range(gh):
        for gx in range(gw):
            if obj.mask[gy, gx]:
                box = [1 + gx * cell, 1 + gy * cell, gx * cell + cell, gy * cell + cell]
                d.rectangle(box, fill=tuple(int(255 * v) for v in grid_rgb[gy, gx]),
                            outline=(60, 60, 64))
    for c in obj.charts:
        d.rectangle([c.x0 * cell, c.y0 * cell, (c.x0 + c.nu) * cell + 1, (c.y0 + c.nv) * cell + 1],
                    outline=(170, 170, 176), width=2)
    for gx, gy in outline:
        d.rectangle([1 + gx * cell, 1 + gy * cell, gx * cell + cell, gy * cell + cell],
                    outline=(235, 45, 45), width=3)
    return img


VIEWS = [(-0.6, 0.45), (2.55, 0.35), (0.9, -0.75)]       # front-left-top, back-right, below


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def comparison(obj, shown, decoded=None, seen=None, cert=None, rgb_of=None, title=""):
    """Rows: SHOWN, DECODED FROM EEG (wrong cells ringed red), and CERTAINTY
    if given; each row three views of the object and its 2D scan grid.
    decoded=None: the shown object only (a preview)."""
    rgb_of = rgb_of if rgb_of is not None else bwb.classes_of(obj.colours)[1]
    preview = decoded is None
    shown = np.asarray(shown)
    decoded = shown if preview else np.asarray(decoded)
    seen = obj.mask.copy() if seen is None else np.asarray(seen) & obj.mask
    K = len(rgb_of)
    quads = obj.cell_quads()
    wrong = {k for k, (gx, gy, *_) in enumerate(obj.cells)
             if seen[gy, gx] and decoded[gy, gx] != shown[gy, gx]}
    wrong_cells = [obj.cells[k][:2] for k in sorted(wrong)]
    grey = np.array([0.5, 0.5, 0.5])
    rows = [("SHOWN", rgb_of[shown], ()),
            ("DECODED FROM EEG", np.where(seen[..., None], rgb_of[decoded], grey), wrong)]
    if preview:
        rows = rows[:1]
    elif cert is not None:
        rows.append(("CERTAINTY", np.where(seen[..., None], certainty_rgb(cert, K), grey), ()))
    vsz, pad, head = 250, 18, 46
    cell = max(10, min(30, int((vsz - 10) / max(obj.grid_h, obj.grid_w * 0.6))))
    at = atlas_image(obj, rows[0][1], cell=cell)
    W = pad + 3 * (vsz + pad) + at.width + pad
    H = head + len(rows) * (vsz + 34) + 40
    img = Image.new("RGB", (W, H), (34, 34, 36))
    d = ImageDraw.Draw(img)
    d.text((pad, 12), title or f"{obj.name}: scanned in 2D, painted back on the mesh",
           fill=(235, 235, 235), font=_font(22))
    for r, (label, grid, out) in enumerate(rows):
        y = head + r * (vsz + 34)
        d.text((pad, y), label, fill=(210, 210, 214), font=_font(17))
        cell_rgb = [grid[gy, gx] for gx, gy, *_ in obj.cells]
        for v, (yaw, pitch) in enumerate(VIEWS):
            img.paste(render(obj, cell_rgb, yaw, pitch, vsz, out, quads), (pad + v * (vsz + pad), y + 24))
        a = atlas_image(obj, grid, cell, outline=wrong_cells if out else ())
        img.paste(a, (pad + 3 * (vsz + pad), y + 24 + (vsz - a.height) // 2))
    n = int(seen[obj.mask].sum())
    tot = int(obj.mask.sum())
    if preview:
        d.text((pad, H - 30), f"{tot} cells on a {obj.grid_w} x {obj.grid_h} scan grid ({obj.layout})"
               "  ·  right: the 2D grid the subject scans, one cell at a time",
               fill=(200, 200, 204), font=_font(15))
        return img
    d.text((pad, H - 30), f"{n - len(wrong_cells)} of {n} cells right"
           + (f" ({tot - n} not decoded)" if n < tot else "")
           + "  ·  red ring = decoded wrong  ·  right: the 2D grid the subject scanned",
           fill=(200, 200, 204), font=_font(15))
    return img


# ---------------------------------------------------------------- export
def write_obj(obj, grid_rgb, path, px=32):
    """Textured OBJ + MTL + PNG. The texture is the scan grid itself (px
    pixels a cell); every chart is a vertex grid whose uvs land on its cells."""
    base = os.path.splitext(path)[0]
    name = os.path.basename(base)
    tex = Image.fromarray((np.clip(np.asarray(grid_rgb, float), 0, 1) * 255).astype(np.uint8), "RGB")
    tex.resize((obj.grid_w * px, obj.grid_h * px), Image.NEAREST).save(base + ".png")
    with open(base + ".mtl", "w") as f:
        f.write(f"newmtl scan\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {name}.png\n")
    s = obj.sub
    lines = [f"# {obj.name}: scan grid {obj.grid_w} x {obj.grid_h}, texture = decoded grid",
             f"mtllib {name}.mtl", "usemtl scan"]
    nv = 0
    for ch in obj.charts:
        U, V = ch.nu * s + 1, ch.nv * s + 1
        uu, vv = np.meshgrid(np.linspace(0, 1, U), np.linspace(0, 1, V))
        P = obj._points(ch, uu, vv)
        for y in range(V):
            for x in range(U):
                lines.append("v %.5f %.5f %.5f" % tuple(P[y, x]))
        for y in range(V):
            for x in range(U):
                tu = (ch.x0 + uu[y, x] * ch.nu) / obj.grid_w
                tv = 1 - (ch.y0 + (1 - vv[y, x]) * ch.nv) / obj.grid_h
                lines.append(f"vt {tu:.6f} {tv:.6f}")
        for y in range(V - 1):
            for x in range(U - 1):
                a = nv + y * U + x + 1
                q = [a, a + 1, a + 1 + U, a + U]
                lines.append("f " + " ".join(f"{i}/{i}" for i in q))
        nv += U * V
    with open(base + ".obj", "w") as f:
        f.write("\n".join(lines) + "\n")
    return base + ".obj"


def viewer_html(obj, payload, title):
    """Self-contained interactive viewer (mesh_view.js inlined)."""
    js = open(os.path.join(HERE, "mesh_view.js"), encoding="utf-8").read()
    tpl = open(os.path.join(HERE, "mesh_viewer_template.html"), encoding="utf-8").read()
    data = json.dumps(payload, separators=(",", ":"))
    return (tpl.replace("/*MESH_VIEW_JS*/", js)
               .replace("/*DATA*/null", data.replace("</", "<\\/"))
               .replace("{{TITLE}}", title))


def payload(obj, shown, decoded=None, seen=None, post=None, colours=None, summary=""):
    """Everything the viewer and the headset need: geometry, palette, and per
    cell the shown and decoded class and the certainty."""
    names, rgb = bwb.classes_of(colours or obj.colours)
    cells = []
    for gx, gy, *_ in obj.cells:
        c = dict(s=int(shown[gy, gx]))
        if decoded is not None and (seen is None or seen[gy, gx]):
            c["d"] = int(decoded[gy, gx])
            if post is not None:
                c["p"] = round(float(np.max(post[gy, gx])), 4)
        cells.append(c)
    return dict(name=obj.name, note=obj.note, layout=obj.layout, names=names,
                rgb=np.round(rgb, 3).tolist(), geom=obj.geometry_json(), cells=cells,
                summary=summary)


def export_session(session_dir, obj, shown, decoded, seen, post, colours, summary=""):
    """Write the 3D results next to the 2D ones: mesh_comparison.png, the
    decoded and shown objects as textured OBJs, and mesh_viewer.html."""
    names, rgb_of = bwb.classes_of(colours)
    cert = np.where(seen, np.asarray(post).max(axis=2), 0) if post is not None else None
    comparison(obj, shown, decoded, seen, cert, rgb_of,
               title=f"{obj.name}: shown vs decoded from EEG").save(
        os.path.join(session_dir, "mesh_comparison.png"))
    grey = np.array([0.5, 0.5, 0.5])
    dec_rgb = np.where(np.asarray(seen)[..., None], rgb_of[decoded], grey)
    dec_rgb[~obj.mask] = 0
    shown_rgb = rgb_of[shown]
    shown_rgb[~obj.mask] = 0
    write_obj(obj, dec_rgb, os.path.join(session_dir, "mesh_decoded.obj"))
    write_obj(obj, shown_rgb, os.path.join(session_dir, "mesh_shown.obj"))
    pl = payload(obj, shown, decoded, seen, post, colours, summary)
    with open(os.path.join(session_dir, "mesh_viewer.html"), "w", encoding="utf-8") as f:
        f.write(viewer_html(obj, pl, f"{obj.name} — EEG scan"))
    return pl


def export_from_session(session_dir, obj, target, colours, summary=""):
    """3D results from a decoded session folder (bwb.decode's
    bwb_result.json and reconstruction_bwb_post.npy)."""
    res = json.load(open(os.path.join(session_dir, "bwb_result.json")))
    post = np.load(os.path.join(session_dir, "reconstruction_bwb_post.npy"))
    seen = post.sum(axis=2) > 0
    return export_session(session_dir, obj, np.asarray(target), np.asarray(res["classes"]),
                          seen, post, colours, summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", help="die | rubik | globe | donut | obj:<file.obj>, optionally :n")
    ap.add_argument("--layout", default="atlas", choices=["atlas", "net"])
    ap.add_argument("--out", default="runs/mesh_preview")
    a = ap.parse_args()
    obj = from_spec(a.object, a.layout)
    os.makedirs(a.out, exist_ok=True)
    tg = obj.target()
    names, rgb_of = bwb.classes_of(obj.colours)
    comparison(obj, tg, title=f"{obj.name}: what will be scanned").save(
        os.path.join(a.out, "mesh_preview.png"))
    shown = rgb_of[tg]
    shown[~obj.mask] = 0
    write_obj(obj, shown, os.path.join(a.out, "mesh_shown.obj"))
    with open(os.path.join(a.out, "mesh_viewer.html"), "w", encoding="utf-8") as f:
        f.write(viewer_html(obj, payload(obj, tg, colours=obj.colours), f"{obj.name} — preview"))
    print(f"{obj.name}: {len(obj.cells)} cells on a {obj.grid_w} x {obj.grid_h} grid "
          f"({obj.layout}), colours {', '.join(obj.colours)}; {obj.note}")
    print(f"wrote {a.out}/mesh_preview.png, mesh_shown.obj, mesh_viewer.html")


if __name__ == "__main__":
    main()
