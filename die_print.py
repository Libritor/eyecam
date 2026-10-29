"""Turn a 3D object decoded from the EEG (runs/<run>/mesh_decoded.obj, any mesh3d.py
object: the die, the globe, the donut, the Rubik's cube) into files a 3D printer takes.

Two models come out:

  <name>.stl         the object exactly as the decoder produced it: the same geometry
                     as the object that was shown, scaled to --size mm across, duplicate
                     seam vertices merged so the surface is closed. The EEG result on
                     this model is the colour of its cells, which a single-colour
                     printer cannot reproduce (see <name>_cells.png, the painting guide).
  <name>_relief.stl  the same object with the decoded colours made physical: each colour
                     at its own height along the surface normal, with walls between
                     cells. The map is per object (RELIEF; --relief overrides it):
                       die    white flush, black sunk --depth, red raised --depth
                       globe  green land flush, blue ocean sunk, white ice raised
                       donut  brown dough flush, pink icing up half a step, sprinkles up one
                       rubik  six colours at six heights, white flush

With --slice each model is sliced with the CuraEngine shipped inside the installed
UltiMaker Cura (printer profile --printer, raft + tree support as in the user's Cura
set-up) and a .gcode is written beside the .stl. Objects other than the die are turned
so their +Y (the globe's north pole, the donut's icing) prints upward (--y-up / --no-y-up).

    python die_print.py                       # runs/vr_die1, 40 mm, STL only
    python die_print.py --slice --copy-to %USERPROFILE%\\Downloads
    python die_print.py --object mesh:globe --shown --slice --preview   # the globe as designed,
                                              # before any scan: runs/globe_shown/print/
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CURA_DIR = r"C:\Program Files\UltiMaker Cura 5.9.0"


def find_cura():
    """The folder of an installed UltiMaker Cura with its CuraEngine: $CURA_DIR,
    else the newest "UltiMaker Cura *" folder in Program Files, else None."""
    import glob
    cands = [os.environ.get("CURA_DIR", "")]
    cands += sorted(glob.glob(r"C:\Program Files\UltiMaker Cura *"), reverse=True) + [CURA_DIR]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "CuraEngine.exe")):
            return c
    return None


CURA_DIR = find_cura() or CURA_DIR

import bwb  # noqa: E402  (the scan's palette: the texture's cells are these colours exactly)

COLOURS = {name: tuple(int(round(255 * v)) for v in rgb) for name, rgb in bwb.PALETTE.items()}

# relief map per object: colour -> height in units of --depth (0 flush, negative sunk)
RELIEF = {
    "die": {"white": 0.0, "black": -1.0, "red": +1.0},
    "globe": {"green": 0.0, "blue": -1.0, "white": +1.0},
    "donut": {"brown": 0.0, "pink": +0.5, "white": +1.0, "yellow": +1.0},
    "rubik": {"white": 0.0, "yellow": +0.5, "orange": +1.0, "red": -0.5, "blue": -1.0, "green": +1.5},
}
Y_UP = {"die": False}        # the die's STL stays as the vr_die1 files were made; the rest turn +Y up


def relief_map(object_name, colours_present, depth, spec=None):
    """{colour: mm} for the colours in the texture. spec 'blue:-1.2,green:0' (mm)
    overrides the object's map; a colour in neither is flush."""
    units = dict(RELIEF.get(object_name or "", {}))
    mm = {c: depth * h for c, h in units.items()}
    for part in str(spec or "").split(","):
        if ":" in part:
            c, v = part.split(":", 1)
            mm[c.strip()] = float(v)
    return {c: float(mm.get(c, 0.0)) for c in colours_present}


def rotate_y_up(verts):
    """mesh3d's +Y (the top of the object) to +Z, which the slicer prints upward."""
    v = np.asarray(verts, float)
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


# ── OBJ with per-vertex texture coordinates ─────────────────────────────────
def load_obj(path):
    verts, uvs, quads = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(x) for x in t[1:4]])
            elif t[0] == "vt":
                uvs.append([float(x) for x in t[1:3]])
            elif t[0] == "f":
                vi = [int(x.split("/")[0]) - 1 for x in t[1:]]
                ti = [int(x.split("/")[1]) - 1 if "/" in x and x.split("/")[1] else vi[k] for k, x in enumerate(t[1:])]
                quads.append((vi, ti))
    return np.array(verts), np.array(uvs), quads


def classify(rgb):
    best, bd = None, 1e9
    for name, c in COLOURS.items():
        d = sum((int(a) - b) ** 2 for a, b in zip(rgb, c))
        if d < bd:
            best, bd = name, d
    return best


def texture_grid(texture, px=32):
    """The scan grid (columns, rows) a mesh3d texture encodes: write_obj paints
    px pixels a cell."""
    W, H = Image.open(texture).size
    return max(1, W // px), max(1, H // px)


def cell_colours(uvs, quads, texture, grid=None):
    """Colour of the texture under each quad's centre and the (column,row) cell it lies in."""
    im = Image.open(texture).convert("RGB")
    W, H = im.size
    grid = grid or texture_grid(texture)
    out = []
    for vi, ti in quads:
        uv = uvs[ti].mean(axis=0)
        u, v = float(uv[0]) % 1.0, float(uv[1]) % 1.0
        px = min(W - 1, int(u * W))
        py = min(H - 1, int((1.0 - v) * H))          # OBJ v runs upward, image rows run downward
        col = classify(im.getpixel((px, py)))
        cell = (min(grid[0] - 1, int(u * grid[0])), min(grid[1] - 1, int(v * grid[1])))
        out.append((cell, col))
    return out


# ── merged, scaled, closed base die ─────────────────────────────────────────
def merged_mesh(verts, quads, scale):
    """Triangulate, merge coincident seam vertices, orient outward. Returns (mesh, map old->merged)."""
    key = {}
    newv = []
    old2new = np.zeros(len(verts), dtype=int)
    for i, p in enumerate(verts):
        k = tuple(np.round(p * scale, 5))
        if k not in key:
            key[k] = len(newv)
            newv.append(np.array(k))
        old2new[i] = key[k]
    tris = []
    for vi, _ in quads:
        a, b, c, d = (old2new[x] for x in vi)
        tris.append((a, b, c))
        tris.append((a, c, d))
    m = trimesh.Trimesh(vertices=np.array(newv), faces=np.array(tris), process=False)
    trimesh.repair.fix_normals(m)
    return m, old2new


# ── relief: each cell offset along the surface normal, walls between cells ──
def relief_mesh(base, old2new, quads, cells, depth_of):
    V = np.asarray(base.vertices)
    N = np.asarray(base.vertex_normals)
    # quads in merged indices, grouped by cell
    cell_quads = defaultdict(list)
    for (vi, _), (cell, col) in zip(quads, cells):
        cell_quads[cell].append(tuple(old2new[x] for x in vi))
    cell_depth, cell_col = {}, {}
    for (vi, _), (cell, col) in zip(quads, cells):
        cell_depth[cell] = depth_of[col]
        cell_col[cell] = col
    # which cells touch each merged vertex, and each undirected edge
    vert_cells = defaultdict(set)
    edge_cells = defaultdict(list)
    for cell, qs in cell_quads.items():
        for q in qs:
            for k in range(4):
                vert_cells[q[k]].add(cell)
                e = tuple(sorted((q[k], q[(k + 1) % 4])))
                edge_cells[e].append(cell)

    out_v, out_f, out_c = [], [], []
    idx = {}

    def P(v, d):
        """Index of merged vertex v displaced by d mm along its normal."""
        k = (v, round(d, 6))
        if k not in idx:
            idx[k] = len(out_v)
            out_v.append(V[v] + N[v] * d)
        return idx[k]

    # cell surfaces
    for cell, qs in cell_quads.items():
        d = cell_depth[cell]
        for a, b, c, e in qs:
            A, B, C, E = P(a, d), P(b, d), P(c, d), P(e, d)
            out_f.append((A, B, C))
            out_f.append((A, C, E))
            out_c += [cell_col[cell], cell_col[cell]]
    # walls along edges shared by two cells of different depth
    for (p, q), cs in edge_cells.items():
        cs = list(dict.fromkeys(cs))
        if len(cs) != 2:
            continue
        da, db = cell_depth[cs[0]], cell_depth[cs[1]]
        if abs(da - db) < 1e-9:
            continue
        lo, hi = min(da, db), max(da, db)

        # every depth present at a corner becomes a vertex of the wall there, so the
        # walls meeting at that corner share vertices instead of leaving T-junctions
        def run(v):
            ds = sorted({cell_depth[c] for c in vert_cells[v] if lo < cell_depth[c] < hi})
            return [lo] + ds + [hi]
        rp, rq = run(p), run(q)
        # strip between the p side and the q side, climbing from lo to hi; winding is fixed below
        i = j = 0
        while i < len(rp) - 1 or j < len(rq) - 1:
            if j == len(rq) - 1 or (i < len(rp) - 1 and rp[i + 1] <= rq[j + 1]):
                out_f.append((P(p, rp[i]), P(q, rq[j]), P(p, rp[i + 1])))
                i += 1
            else:
                out_f.append((P(p, rp[i]), P(q, rq[j]), P(q, rq[j + 1])))
                j += 1
            out_c.append("wall")

    m = trimesh.Trimesh(vertices=np.array(out_v), faces=np.array(out_f), process=False)
    m.merge_vertices()
    keep = m.nondegenerate_faces()
    m.update_faces(keep)
    trimesh.repair.fix_normals(m)      # consistent winding, outward (the surface is closed)
    return m, np.array(out_c)[keep]


def preview_png(m, colours, out, title=None):
    """Three views of the relief object, cells in their decoded colours, walls grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rgb = {name: tuple(0.08 + 0.84 * v / 255.0 for v in c) for name, c in COLOURS.items()}
    rgb["wall"] = (0.55, 0.55, 0.6)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    N = np.asarray(m.face_normals)
    fig = plt.figure(figsize=(12, 4.2), dpi=110)
    fig.patch.set_facecolor("#1e1e1e")
    views = [(28, -50), (28, 130), (-28, 40)]
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        light = np.array([np.cos(np.radians(azim + 40)), np.sin(np.radians(azim + 40)), 0.8])
        light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.clip(N @ light, 0, 1)
        fc = np.array([rgb[c] for c in colours]) * shade[:, None]
        pc = Poly3DCollection(V[F], facecolors=np.clip(fc, 0, 1), edgecolors="none")
        ax.add_collection3d(pc)
        c = V.mean(axis=0); r = 24
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
        ax.set_facecolor("#1e1e1e")
    fig.suptitle(title or "die decoded from EEG, relief for printing: black sunk, red raised, white flush",
                 color="#dddddd", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def edge_report(m):
    """How many faces share each edge: 2 everywhere for a clean surface; 4 on the vertical
    lines where cells of a checkerboard meet corner to corner (a closed solid still)."""
    e = np.sort(m.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}


def report(m, name):
    print(f"{name}: {len(m.vertices)} vertices, {len(m.faces)} triangles, "
          f"extent {np.round(m.extents, 2)} mm, watertight={m.is_watertight}, "
          f"winding consistent={m.is_winding_consistent}, volume={m.volume:.0f} mm3, "
          f"faces per edge {edge_report(m)}")


def cells_png(texture, out, cell_px=48, grid=None):
    """The decoded cell grid, enlarged, as a painting guide for the printed object."""
    im = Image.open(texture).convert("RGB")
    grid = grid or texture_grid(texture)
    im = im.resize((grid[0] * cell_px, grid[1] * cell_px), Image.NEAREST)
    im.save(out)


# ── slicing with the CuraEngine inside the installed Cura ───────────────────
# CuraEngine on its own does not evaluate the Python formulas ("value") that the
# printer profiles use for most settings; the Cura front end does that before
# calling the engine. So the profile chain is resolved here the same way and
# written out as a definition whose every setting is a plain value.
import json
import math

CURA_RES = os.path.join(CURA_DIR, "share", "cura", "resources")


def _load_chain(name):
    chain = []
    while name:
        for sub in ("definitions", "extruders"):
            p = os.path.join(CURA_RES, sub, name + ".def.json")
            if os.path.exists(p):
                break
        else:
            raise SystemExit("Cura definition not found: " + name)
        d = json.load(open(p, encoding="utf-8"))
        chain.append(d)
        name = d.get("inherits")
    return chain[::-1]          # base first


def _collect(chain):
    """key -> merged setting entry; a later definition's 'value' or 'default_value' wins."""
    defs = {}

    def walk(tree):
        for k, v in tree.items():
            if not isinstance(v, dict):
                continue
            if v.get("type") != "category":
                e = defs.setdefault(k, {})
                e.update({a: b for a, b in v.items() if a != "children"})
            if "children" in v:
                walk(v["children"])
    for d in chain:
        if "settings" in d:
            walk(d["settings"])
        for k, v in d.get("overrides", {}).items():
            e = defs.setdefault(k, {})
            if "default_value" in v and "value" not in v:
                e.pop("value", None)
            e.update(v)
    return defs


class _Resolver:
    def __init__(self, defs, user):
        self.defs, self.user, self.cache, self.busy, self.failed = defs, user, {}, set(), []
        self.env = {"math": math, "resolveOrValue": self.get, "extruderValue": lambda n, k: self.get(k),
                    "extruderValues": lambda k: [self.get(k)], "defaultExtruderPosition": lambda: "0",
                    "anyExtruderWithMaterial": lambda *a: "0", "anyExtruderNrWithOrDefault": lambda *a: "0",
                    "valueFromContainer": lambda k, i: self.get(k), "valueFromExtruderContainer": lambda k, i: self.get(k),
                    "extruderValueFromDict": lambda k, d: self.get(k), "extruderValuesFromDict": lambda k, d: [self.get(k)]}

    def typed(self, key, v):
        t = self.defs.get(key, {}).get("type", "str")
        if isinstance(v, str):
            if t == "float":
                return float(v)
            if t == "int":
                return int(float(v))
            if t == "bool":
                return v.strip().lower() in ("1", "true", "yes")
        return v

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        if key in self.user:
            v = self.typed(key, self.user[key])
        else:
            e = self.defs[key]
            v = e.get("default_value")
            if "value" in e and key not in self.busy:
                self.busy.add(key)
                try:
                    v = eval(e["value"], self.env, _Namespace(self)) if isinstance(e["value"], str) else e["value"]
                except Exception as ex:
                    self.failed.append(f"{key}: {ex}")
                finally:
                    self.busy.discard(key)
        self.cache[key] = v
        return v


class _Namespace(dict):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def __missing__(self, key):
        if key in self.r.defs or key in self.r.user:
            return self.r.get(key)
        raise KeyError(key)


def resolved_definitions(printer, user, out_dir):
    """Write <out_dir>/<printer>_resolved.def.json (+ extruder) with every setting as a value."""
    pchain = _load_chain(printer)
    ext_id = None
    for d in pchain:
        ext_id = d.get("metadata", {}).get("machine_extruder_trains", {}).get("0", ext_id)
    echain = _load_chain(ext_id)
    pdefs, edefs = _collect(pchain), _collect(echain)
    alldefs = dict(pdefs)
    alldefs.update(edefs)
    r = _Resolver(alldefs, user)
    values = {k: r.get(k) for k in alldefs}
    for k in ("machine_start_gcode", "machine_end_gcode"):
        if isinstance(values.get(k), str):
            values[k] = re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), values[k])
    os.makedirs(out_dir, exist_ok=True)
    ext_name = printer + "_resolved_extruder"
    pj = {"version": 2, "name": printer + " (resolved)", "inherits": "fdmprinter",
          "metadata": {"type": "machine", "machine_extruder_trains": {"0": ext_name}},
          "overrides": {k: {"default_value": values[k]} for k in pdefs}}
    ej = {"version": 2, "name": ext_name, "inherits": "fdmextruder",
          "metadata": {"type": "extruder", "machine": printer, "position": "0"},
          "overrides": {k: {"default_value": values[k]} for k in edefs}}
    for src in ("definitions/fdmprinter.def.json", "definitions/fdmextruder.def.json"):
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(CURA_RES, src), dst)
    pp = os.path.join(out_dir, printer + "_resolved.def.json")
    json.dump(pj, open(pp, "w", encoding="utf-8"), indent=1)
    json.dump(ej, open(os.path.join(out_dir, ext_name + ".def.json"), "w", encoding="utf-8"), indent=1)
    if r.failed:
        print(f"  {len(r.failed)} formulas kept their default, e.g. " + "; ".join(r.failed[:3]))
    return pp, values


PRINT_SETTINGS = {
    "layer_height": "0.2", "layer_height_0": "0.2",
    "wall_line_count": "3", "top_layers": "5", "bottom_layers": "4",
    "infill_sparse_density": "15", "infill_pattern": "grid",
    "material_print_temperature": "200", "material_print_temperature_layer_0": "205",
    "material_bed_temperature": "60", "material_bed_temperature_layer_0": "60",
    "speed_print": "50", "retraction_enable": "true",
    "adhesion_type": "raft", "support_enable": "true", "support_structure": "tree",
    "support_type": "buildplate", "support_angle": "55",
    "minimum_support_area": "20",      # the 1.2 mm ledges of the relief cells bridge on their own
    "machine_center_is_zero": "false",
}


def slice_stl(stl, gcode, printer, extra):
    engine = os.path.join(CURA_DIR, "CuraEngine.exe")
    if not os.path.exists(engine):
        raise SystemExit("CuraEngine not found at " + engine)
    settings = dict(PRINT_SETTINGS)
    settings.update(extra)
    defs_dir = os.path.join(os.path.dirname(gcode), "_cura_defs")
    pdef, values = resolved_definitions(printer, settings, defs_dir)
    # with machine_center_is_zero false the engine itself shifts the model to the bed centre,
    # so hand it the die centred on x = y = 0 and standing on z = 0
    m = trimesh.load(stl, force="mesh")
    m.apply_translation([-m.bounds.mean(axis=0)[0], -m.bounds.mean(axis=0)[1], -m.bounds[0][2]])
    tmp = gcode + ".bed.stl"
    m.export(tmp)
    cmd = [engine, "slice", "-p", "-j", pdef, "-l", tmp, "-o", gcode]
    env = dict(os.environ, CURA_ENGINE_SEARCH_PATH=defs_dir)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    os.remove(tmp)
    errors = [l for l in (r.stdout + r.stderr).splitlines() if "[error]" in l]
    if r.returncode != 0 or not os.path.exists(gcode) or errors:
        print("\n".join(errors[:10]) or (r.stdout + r.stderr)[-3000:])
        raise SystemExit("CuraEngine failed for " + stl)
    return fix_header(gcode, values.get("material_diameter", 1.75))


def fix_header(gcode, filament_diameter):
    """Outside the Cura front end the engine leaves ;TIME, ;Filament used and the bounds as
    placeholders; fill them from the file itself so the printer's and Cura's previews show
    the real numbers."""
    text = open(gcode, encoding="utf-8", errors="replace").read()
    t = 0.0
    for m in re.finditer(r"^;TIME_ELAPSED:([\d.]+)", text, re.M):
        t = float(m.group(1))
    e_prev, e_total, relative = 0.0, 0.0, False
    lo, hi = [1e9] * 3, [-1e9] * 3
    pos = [0.0, 0.0, 0.0]
    in_print = False                      # bounds cover the print itself, not the start-up purge line
    for line in text.splitlines():
        if line.startswith(";LAYER:"):
            in_print = True
        if line.startswith("G92"):
            m = re.search(r"E(-?[\d.]+)", line)
            if m:
                e_prev = float(m.group(1))
            continue
        if line.startswith("M83"):
            relative = True
        elif line.startswith("M82"):
            relative = False
        if not (line.startswith("G1") or line.startswith("G0")):
            continue
        for ax, k in (("X", 0), ("Y", 1), ("Z", 2)):
            m = re.search(rf"\b{ax}(-?[\d.]+)", line)
            if m:
                pos[k] = float(m.group(1))
        m = re.search(r"\bE(-?[\d.]+)", line)
        if m:
            e = float(m.group(1))
            d = e if relative else e - e_prev
            if not relative:
                e_prev = e
            if d > 0:
                e_total += d
                if in_print:
                    for k in range(3):
                        lo[k] = min(lo[k], pos[k]); hi[k] = max(hi[k], pos[k])
    length_m = e_total / 1000.0
    vol = e_total * math.pi * (filament_diameter / 2) ** 2 / 1000.0   # cm3
    rep = {"TIME": f"{t:.0f}", "Filament used": f"{length_m:.5f}m",
           "MINX": f"{lo[0]:.3f}", "MINY": f"{lo[1]:.3f}", "MINZ": f"{lo[2]:.3f}",
           "MAXX": f"{hi[0]:.3f}", "MAXY": f"{hi[1]:.3f}", "MAXZ": f"{hi[2]:.3f}"}
    head, sep, rest = text.partition("\n;TARGET_MACHINE.NAME")
    for k, v in rep.items():
        head = re.sub(rf"^;{re.escape(k)}:.*$", f";{k}:{v}", head, count=1, flags=re.M)
    head += f"\n;Filament volume: {vol:.1f}cm3 (about {vol * 1.24:.0f} g of PLA)"
    open(gcode, "w", encoding="utf-8").write(head + sep + rest)
    head = open(gcode, encoding="utf-8", errors="replace").read(4000)
    info = {k: re.search(rf"^;{k}:(.*)$", head, re.M) for k in ("TIME", "Filament used", "Layer height", "LAYER_COUNT", "MINX", "MAXX", "MINY", "MAXY", "MINZ", "MAXZ")}
    info = {k: v.group(1).strip() for k, v in info.items() if v}
    t = float(info.get("TIME", "0"))
    print(f"{os.path.basename(gcode)}: {os.path.getsize(gcode)//1024} kB, layers {info.get('LAYER_COUNT','?')}, "
          f"estimated {t/3600:.1f} h, filament {info.get('Filament used','?')}, "
          f"footprint x {info.get('MINX','?')}..{info.get('MAXX','?')} y {info.get('MINY','?')}..{info.get('MAXY','?')} z to {info.get('MAXZ','?')} mm")
    return info


def make_print(run, mesh="mesh_decoded.obj", size=40.0, depth=1.2, name="die_decoded_eeg",
               out=None, slice_models=("relief",), printer="creality_ender3pro", extra=None,
               log=print, object_name="die", relief=None, y_up=None, preview=False):
    """The object in `run`/<mesh> as STL (plain and relief) and, for each model in
    slice_models ("plain", "relief"), G-code. object_name picks the relief map
    (RELIEF) and the printing orientation (Y_UP); relief 'colour:mm,...'
    overrides the map. Returns dict(stl=..., relief=..., gcode={model: path},
    info={model: slicer summary}, cells={colour: n}, relief_mm={colour: mm})."""
    out = out or os.path.join(run, "print")
    os.makedirs(out, exist_ok=True)
    obj = os.path.join(run, mesh)
    verts, uvs, quads = load_obj(obj)
    texture = os.path.join(run, os.path.splitext(mesh)[0] + ".png")
    cells = cell_colours(uvs, quads, texture)
    counts = defaultdict(set)
    for cell, col in cells:
        counts[col].add(cell)
    if y_up is None:
        y_up = Y_UP.get(object_name, True)
    if y_up:
        verts = rotate_y_up(verts)
    half = float(np.abs(verts).max())
    base, old2new = merged_mesh(verts, quads, size / (2 * half))
    report(base, f"{object_name} (as in the mesh)")
    stl = os.path.join(out, name + ".stl")
    base.export(stl)
    depth_of = relief_map(object_name, counts.keys(), depth, relief)
    rel, rel_col = relief_mesh(base, old2new, quads, cells, depth_of)
    desc = ", ".join(f"{c} {d:+.1f} mm" if d else f"{c} flush" for c, d in sorted(depth_of.items()))
    report(rel, f"{object_name} relief ({desc})")
    stl_rel = os.path.join(out, name + "_relief.stl")
    rel.export(stl_rel)
    cells_png(texture, os.path.join(out, name + "_cells.png"))
    res = dict(stl=stl, relief=stl_rel, gcode={}, info={}, relief_mm=depth_of, y_up=bool(y_up),
               cells={k: len(v) for k, v in sorted(counts.items())},
               watertight=dict(plain=bool(base.is_watertight), relief=bool(rel.is_watertight)),
               files=[stl, stl_rel, os.path.join(out, name + "_cells.png")])
    if preview:
        png = os.path.join(out, name + "_relief_preview.png")
        preview_png(rel, rel_col, png, title=f"{object_name}, relief for printing: {desc}")
        res["files"].append(png)
    for model in slice_models:
        src = stl if model == "plain" else stl_rel
        g = os.path.splitext(src)[0] + ".gcode"
        res["info"][model] = slice_stl(src, g, printer, dict(extra or {}))
        res["gcode"][model] = g
        res["files"].append(g)
    return res


def shown_object(spec, run):
    """Write the object as designed (mesh3d's target colours) to run/mesh_shown.obj
    and return (object name, 'mesh_shown.obj', the mesh3d object)."""
    import mesh3d
    obj = mesh3d.from_spec(spec)
    names, rgb = bwb.classes_of(obj.colours)
    os.makedirs(run, exist_ok=True)
    mesh3d.write_obj(obj, rgb[obj.target()], os.path.join(run, "mesh_shown.obj"))
    return obj.name, "mesh_shown.obj", obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="session folder holding the OBJ (default runs/vr_die1; "
                                                "with --object: runs/<object>_shown)")
    ap.add_argument("--mesh", default="mesh_decoded.obj", help="OBJ in the run folder (mesh_decoded.obj or mesh_shown.obj)")
    ap.add_argument("--object", default=None, help="the mesh3d object spec, e.g. mesh:globe: picks the relief "
                                                   "map and the orientation (default: the die)")
    ap.add_argument("--shown", action="store_true", help="with --object: build the object as designed into "
                                                         "the run folder (mesh_shown.obj) and print that")
    ap.add_argument("--size", type=float, default=40.0, help="width across, mm")
    ap.add_argument("--depth", type=float, default=1.2, help="relief step, mm (the object's map scales it)")
    ap.add_argument("--relief", default=None, help="override the relief map: 'blue:-1.2,green:0,white:1.2' (mm)")
    ap.add_argument("--y-up", dest="y_up", action="store_true", default=None,
                    help="turn the mesh so its +Y prints upward (default for every object but the die)")
    ap.add_argument("--no-y-up", dest="y_up", action="store_false")
    ap.add_argument("--name", default=None, help="file stem (default <object>_decoded_eeg, or <object>_shown)")
    ap.add_argument("--out", default=None, help="output folder (default <run>/print)")
    ap.add_argument("--slice", action="store_true", help="also write G-code with CuraEngine")
    ap.add_argument("--preview", action="store_true", help="render <name>_relief_preview.png (matplotlib)")
    ap.add_argument("--printer", default="creality_ender3pro")
    ap.add_argument("--set", action="append", default=[], help="extra CuraEngine setting key=value")
    ap.add_argument("--copy-to", default=None, help="folder to copy the STL and G-code files into")
    a = ap.parse_args()

    object_name, mesh = "die", a.mesh
    if a.object:
        import mesh3d
        object_name = mesh3d.from_spec(a.object).name
    run = a.run or os.path.join(HERE, "runs", f"{object_name}_shown" if a.shown else "vr_die1")
    if a.shown:
        if not a.object:
            raise SystemExit("--shown needs --object mesh:<name>")
        object_name, mesh, obj3d = shown_object(a.object, run)
        print(f"{object_name} as designed: {len(obj3d.cells)} cells on a {obj3d.grid_w} x {obj3d.grid_h} "
              f"atlas -> {os.path.join(run, mesh)}")
    name = a.name or f"{object_name}_{'shown' if a.shown else 'decoded_eeg'}"
    extra = dict(s.split("=", 1) for s in a.set)
    res = make_print(run, mesh=mesh, size=a.size, depth=a.depth, name=name, out=a.out,
                     slice_models=("plain", "relief") if a.slice else (), printer=a.printer,
                     extra=extra, object_name=object_name, relief=a.relief, y_up=a.y_up, preview=a.preview)
    print(f"{mesh}: cells " + ", ".join(f"{k} {v}" for k, v in res["cells"].items())
          + "; relief " + ", ".join(f"{c} {d:+.1f} mm" for c, d in sorted(res["relief_mm"].items())))
    for model, info in res["info"].items():
        if info:
            print(f"{model} G-code: about {float(info.get('TIME', 0) or 0) / 3600:.1f} h, "
                  f"{info.get('Filament used', '?')} of filament")
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        for f in res["files"]:
            shutil.copy2(f, a.copy_to)
        print("copied to", a.copy_to)
    print("wrote", a.out or os.path.join(run, "print"))


if __name__ == "__main__":
    main()
