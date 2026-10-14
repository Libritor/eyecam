// MeshView: a scanned 3D object drawn on a plain 2D canvas, one shaded quad
// per sub-patch of each scanned cell (painter's algorithm, back faces culled,
// cell borders drawn as the quads are). Same camera and light as mesh3d.py's
// render(). Shared by mesh_viewer.html (mesh3d.py inlines this file) and the
// headset's result screen (xr_stimulus.html loads /mesh_view.js).
"use strict";
class MeshView {
  // geom: mesh3d.Object3D.geometry_json(): {sub, cells: [{gx, gy, chart, q}]}
  constructor(canvas, geom) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.geom = geom;
    this.yaw = -0.6; this.pitch = 0.45; this.zoom = 1;
    this.quads = [];
    const s = geom.sub;
    geom.cells.forEach((c, k) => c.q.forEach((p, qi) => {
      const a = Math.floor(qi / s), b = qi % s;
      this.quads.push({ cell: k, p, e: [a === 0, b === s - 1, a === s - 1, b === 0] });
    }));
    this.drawn = [];
  }

  resize() {
    const r = this.cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
    if (this.cv.width !== w || this.cv.height !== h) { this.cv.width = w; this.cv.height = h; }
  }

  // o.colours: [r, g, b] (0..1) per cell; o.outline: Set of cells ringed red;
  // o.hi: a cell to highlight (-1 none); o.bg: background colour
  draw(o) {
    const ctx = this.ctx, W = this.cv.width, H = this.cv.height;
    ctx.fillStyle = o.bg || "#18181b"; ctx.fillRect(0, 0, W, H);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const R = [cy, 0, sy, sp * sy, cp, -sp * cy, -cp * sy, sp, cp * cy];
    const D = 3.4, f = Math.min(W, H) * 1.12 * this.zoom, L = [-0.45, 0.6, 0.66];
    const ln = Math.hypot(...L); L[0] /= ln; L[1] /= ln; L[2] /= ln;
    const items = [];
    for (const q of this.quads) {
      const v = new Array(12);
      for (let i = 0; i < 4; i++) {
        const x = q.p[3 * i], y = q.p[3 * i + 1], z = q.p[3 * i + 2];
        v[3 * i] = R[0] * x + R[1] * y + R[2] * z;
        v[3 * i + 1] = R[3] * x + R[4] * y + R[5] * z;
        v[3 * i + 2] = R[6] * x + R[7] * y + R[8] * z;
      }
      const ax = v[3] - v[0], ay = v[4] - v[1], az = v[5] - v[2];
      const bx = v[9] - v[0], by = v[10] - v[1], bz = v[11] - v[2];
      let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const nn = Math.hypot(nx, ny, nz) || 1; nx /= nn; ny /= nn; nz /= nn;
      const mx = (v[0] + v[3] + v[6] + v[9]) / 4, my = (v[1] + v[4] + v[7] + v[10]) / 4,
            mz = (v[2] + v[5] + v[8] + v[11]) / 4;
      if (nx * mx + ny * my + nz * (mz - D) >= 0) continue;     // faces away
      const xy = [];
      for (let i = 0; i < 4; i++) {
        const w = D - v[3 * i + 2];
        xy.push(W / 2 + f * v[3 * i] / w, H / 2 - f * v[3 * i + 1] / w);
      }
      items.push({ z: mz, q, xy, shade: 0.38 + 0.62 * Math.max(0, nx * L[0] + ny * L[1] + nz * L[2]) });
    }
    items.sort((a, b) => a.z - b.z);
    const out = o.outline || new Set(), lw = Math.max(1, Math.min(W, H) / 400);
    for (const it of items) {
      const k = it.q.cell, c = o.colours[k] || [0.5, 0.5, 0.5], s = it.shade;
      const lit = k === o.hi ? 0.35 : 0;
      const rgb = c.map(x => Math.round(255 * Math.min(1, (x * s + 0.04 * s) * (1 - lit) + lit)));
      const xy = it.xy;
      ctx.beginPath();
      ctx.moveTo(xy[0], xy[1]); ctx.lineTo(xy[2], xy[3]); ctx.lineTo(xy[4], xy[5]); ctx.lineTo(xy[6], xy[7]);
      ctx.closePath();
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fill();
      ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 1.2; ctx.stroke();   // hide seams
      const bad = out.has(k);
      ctx.strokeStyle = bad ? "#eb2d2d" : k === o.hi ? "#ffffff"
        : `rgb(${rgb.map(x => Math.round(x * 0.55)).join(",")})`;
      ctx.lineWidth = bad || k === o.hi ? 3 * lw : lw;
      ctx.beginPath();
      for (let e = 0; e < 4; e++) {
        if (!it.q.e[e]) continue;
        const n = (e + 1) % 4;
        ctx.moveTo(xy[2 * e], xy[2 * e + 1]); ctx.lineTo(xy[2 * n], xy[2 * n + 1]);
      }
      ctx.stroke();
    }
    this.drawn = items;
  }

  // the cell under canvas point (x, y) in CSS pixels, -1 if none
  pick(x, y) {
    const d = this.cv.width / (this.cv.getBoundingClientRect().width || 1);
    x *= d; y *= d;
    for (let i = this.drawn.length - 1; i >= 0; i--) {
      const p = this.drawn[i].xy;
      let inside = false;
      for (let a = 0, b = 3; a < 4; b = a++) {
        const xa = p[2 * a], ya = p[2 * a + 1], xb = p[2 * b], yb = p[2 * b + 1];
        if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) inside = !inside;
      }
      if (inside) return this.drawn[i].q.cell;
    }
    return -1;
  }
}
if (typeof module !== "undefined") module.exports = { MeshView };
