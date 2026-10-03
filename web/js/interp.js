// Image reconstruction helpers: Delaunay triangulation + linear interpolation of scattered
// samples (Fig. 6 look), grid resampling (Fig. 1/8/11 look), NaN filling, colormaps.

import { percentile } from './dsp.js';

// ------------------------------------------------------------- Delaunay (Bowyer-Watson)
export function delaunay(xs, ys) {
  const n = xs.length;
  if (n < 3) return [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (xs[i] < minX) minX = xs[i]; if (xs[i] > maxX) maxX = xs[i];
    if (ys[i] < minY) minY = ys[i]; if (ys[i] > maxY) maxY = ys[i];
  }
  const d = Math.max(maxX - minX, maxY - minY, 1) * 50, mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
  const X = Float64Array.from([...xs, mx - d, mx, mx + d]);
  const Y = Float64Array.from([...ys, my - d, my + d, my - d]);
  const mk = (a, b, c) => {
    const ax = X[a], ay = Y[a], bx = X[b], by = Y[b], cx = X[c], cy = Y[c];
    const D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(D) < 1e-12) return { a, b, c, x: 0, y: 0, r2: Infinity };
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D;
    return { a, b, c, x: ux, y: uy, r2: (ax - ux) ** 2 + (ay - uy) ** 2 };
  };
  let tris = [mk(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const px = X[i], py = Y[i];
    const keep = [], edges = new Map();
    for (const t of tris) {
      if ((px - t.x) ** 2 + (py - t.y) ** 2 < t.r2) {
        for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
          const k = u < v ? u * 1e6 + v : v * 1e6 + u;
          if (edges.has(k)) edges.delete(k); else edges.set(k, [u, v]);
        }
      } else keep.push(t);
    }
    for (const [u, v] of edges.values()) keep.push(mk(u, v, i));
    tris = keep;
  }
  return tris.filter(t => t.a < n && t.b < n && t.c < n).map(t => [t.a, t.b, t.c]);
}

// Rasterize linear interpolation over triangles into a w x h Float32Array (NaN outside hull).
// Points are given in output-pixel coordinates.
export function rasterizeTriangles(xs, ys, vals, tris, w, h) {
  const out = new Float32Array(w * h).fill(NaN);
  for (const [a, b, c] of tris) {
    const x1 = xs[a], y1 = ys[a], x2 = xs[b], y2 = ys[b], x3 = xs[c], y3 = ys[c];
    const den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    if (Math.abs(den) < 1e-9) continue;
    const x0 = Math.max(0, Math.floor(Math.min(x1, x2, x3))), xe = Math.min(w - 1, Math.ceil(Math.max(x1, x2, x3)));
    const y0 = Math.max(0, Math.floor(Math.min(y1, y2, y3))), ye = Math.min(h - 1, Math.ceil(Math.max(y1, y2, y3)));
    for (let y = y0; y <= ye; y++) for (let x = x0; x <= xe; x++) {
      const px = x + 0.5, py = y + 0.5;
      const l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den;
      const l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den;
      const l3 = 1 - l1 - l2;
      if (l1 >= -1e-6 && l2 >= -1e-6 && l3 >= -1e-6) out[y * w + x] = l1 * vals[a] + l2 * vals[b] + l3 * vals[c];
    }
  }
  return out;
}

// Gaussian splatting of scattered weighted samples (alternative to triangulation).
export function splat(xs, ys, vals, weights, w, h, sigma) {
  const num = new Float64Array(w * h), den = new Float64Array(w * h);
  const r = Math.ceil(sigma * 3), s2 = 2 * sigma * sigma;
  for (let i = 0; i < xs.length; i++) {
    const cx = xs[i], cy = ys[i], wi = weights ? weights[i] : 1;
    if (!(wi > 0) || !Number.isFinite(vals[i])) continue;
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(h - 1, Math.ceil(cy + r)); y++)
      for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(w - 1, Math.ceil(cx + r)); x++) {
        const g = wi * Math.exp(-((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) / s2);
        num[y * w + x] += g * vals[i]; den[y * w + x] += g;
      }
  }
  const out = new Float32Array(w * h);
  let dmax = 0; for (const d of den) if (d > dmax) dmax = d;
  for (let i = 0; i < out.length; i++) out[i] = den[i] > dmax * 1e-3 ? num[i] / den[i] : NaN;
  return out;
}

// Replace NaNs by the mean of valid 8-neighbours, iterating until filled.
export function fillNaN(grid, w, h) {
  const g = Float32Array.from(grid);
  let any = true, guard = 0;
  if (!g.some(Number.isFinite)) return g.fill(0);
  while (any && guard++ < w + h) {
    any = false;
    const src = Float32Array.from(g);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (Number.isFinite(src[i])) continue;
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const v = src[yy * w + xx];
        if (Number.isFinite(v)) { s += v; c++; }
      }
      if (c) g[i] = s / c; else any = true;
    }
  }
  return g;
}

const cub = (p0, p1, p2, p3, t) => p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));

// Sample a gw x gh grid at fractional grid coordinates (gx, gy), clamped at the edges.
export function sample(grid, gw, gh, gx, gy, method = 'bilinear') {
  const at = (x, y) => grid[Math.min(gh - 1, Math.max(0, y)) * gw + Math.min(gw - 1, Math.max(0, x))];
  const x0 = Math.floor(gx), tx = gx - x0, y0 = Math.floor(gy), ty = gy - y0;
  if (method === 'nearest') return at(Math.round(gx), Math.round(gy));
  if (method === 'bicubic') {
    const r = [];
    for (let j = -1; j <= 2; j++) r.push(cub(at(x0 - 1, y0 + j), at(x0, y0 + j), at(x0 + 1, y0 + j), at(x0 + 2, y0 + j), tx));
    return cub(r[0], r[1], r[2], r[3], ty);
  }
  const a = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
  const b = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
  return a * (1 - ty) + b * ty;
}

// Resample a gw x gh grid (cell centres) onto w x h output pixels.
export function resize(grid, gw, gh, w, h, method = 'bilinear') {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out[y * w + x] = sample(grid, gw, gh, (x + 0.5) * gw / w - 0.5, (y + 0.5) * gh / h - 0.5, method);
  return out;
}

export function rotateCW(grid, w, h) { // returns {grid, w: h, h: w}
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[x * h + (h - 1 - y)] = grid[y * w + x];
  return { grid: out, w: h, h: w };
}

export function gaussianBlur(grid, w, h, sigma) {
  if (!(sigma > 0)) return grid;
  const r = Math.ceil(sigma * 3), k = [];
  for (let i = -r; i <= r; i++) k.push(Math.exp(-i * i / (2 * sigma * sigma)));
  const pass = (src, horiz) => {
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0, c = 0;
      for (let i = -r; i <= r; i++) {
        const xx = horiz ? x + i : x, yy = horiz ? y : y + i;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const v = src[yy * w + xx];
        if (Number.isFinite(v)) { s += v * k[i + r]; c += k[i + r]; }
      }
      out[y * w + x] = c ? s / c : NaN;
    }
    return out;
  };
  return pass(pass(grid, true), false);
}

// Map to [0,1] using robust percentiles, then apply gain (paper: "multiplied by 2").
export function normalize(grid, { lo = 0.02, hi = 0.99, gain = 1, gamma = 1 } = {}) {
  const vals = Array.from(grid);
  const a = percentile(vals, lo), b = percentile(vals, hi);
  const out = new Float32Array(grid.length);
  const span = b - a || 1;
  for (let i = 0; i < grid.length; i++) {
    const v = grid[i];
    out[i] = Number.isFinite(v) ? Math.min(1, Math.max(0, Math.pow(Math.max(0, (v - a) / span), gamma) * gain)) : NaN;
  }
  return out;
}

// ------------------------------------------------------------------------- colormaps
const hex = s => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
const ANCHORS = {
  gray: ['#000000', '#ffffff'],
  viridis: ['#440154', '#472c7a', '#3b518b', '#2c718e', '#21908d', '#27ad81', '#5cc863', '#aadc32', '#fde725'],
  hot: ['#000000', '#7a0000', '#e01b00', '#ff7a00', '#ffd21a', '#ffffe0'],
  bluered: ['#0010ff', '#5a00c8', '#b0006e', '#ff0000'],      // Fig. 4 LED: blue (low) -> red (high)
  veillance: ['#ff0000', '#ffae00', '#00ff40', '#00c8ff', '#0030ff'], // Fig. 3: red bias, green mid, blue strong
  redglow: ['#000000', '#5a0000', '#ff0000', '#ff9a8a'],        // Fig. 4 top: bright red where SSVEP strong
};
export const CMAPS = Object.keys(ANCHORS);
const luts = {};
export function lut(name) {
  if (luts[name]) return luts[name];
  const a = (ANCHORS[name] || ANCHORS.gray).map(hex), L = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (a.length - 1), k = Math.min(a.length - 2, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) L[i * 3 + c] = a[k][c] + (a[k + 1][c] - a[k][c]) * f;
  }
  return (luts[name] = L);
}
export function colorOf(name, v) {
  const L = lut(name), i = Math.round(Math.min(1, Math.max(0, v)) * 255) * 3;
  return [L[i], L[i + 1], L[i + 2]];
}

// Paint a normalized [0,1] grid into an ImageData (NaN -> background colour).
export function toImageData(norm, w, h, cmap = 'gray', bg = [0, 0, 0]) {
  const img = new ImageData(w, h), L = lut(cmap), d = img.data;
  for (let i = 0; i < w * h; i++) {
    const v = norm[i];
    if (Number.isFinite(v)) {
      const k = Math.round(v * 255) * 3;
      d[i * 4] = L[k]; d[i * 4 + 1] = L[k + 1]; d[i * 4 + 2] = L[k + 2];
    } else { d[i * 4] = bg[0]; d[i * 4 + 1] = bg[1]; d[i * 4 + 2] = bg[2]; }
    d[i * 4 + 3] = 255;
  }
  return img;
}

export function drawGrid(canvas, norm, w, h, cmap) {
  canvas.width = w; canvas.height = h;
  canvas.getContext('2d').putImageData(toImageData(norm, w, h, cmap), 0, 0);
}

// Pearson correlation between two equal-length arrays (NaN-safe); used to score
// reconstructions against the ground-truth stimulus in tests / simulator runs.
export function correlation(a, b) {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) { n++; sa += a[i]; sb += b[i]; }
  if (n < 2) return NaN;
  const ma = sa / n, mb = sb / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) {
    num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db || 1);
}
