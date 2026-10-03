// Reconstruction: turn EEG + stimulus logs into images, exactly as described in the paper.
// Pure functions of (eeg segment | live store, run log, params) — used live and in Review.

import { spectrum, ssvepMetric, peakToPeak, percentile, butterLowpass, filtfilt, resampleUniform } from './dsp.js';
import { windowAt, indexAt } from './eeg-store.js';
import { fbcca, lockin, coherentProject } from './ssvep.js';
import { kalmanAt } from './kalman.js';
import { windowWeights } from './artifacts.js';
import { detrend } from './dsp.js';
import { correlation } from './interp.js';
import {
  sample, fillNaN, normalize, rotateCW, delaunay, rasterizeTriangles, splat, gaussianBlur,
} from './interp.js';

export const DEFAULTS = {
  metric: 'harmonic',   // see dsp.METRICS
  psd: 'welch',
  segSec: 2,
  bw: 0.25,
  harmonics: 2,
  latency: 0.12,        // visual pathway + transport delay (s)
  rejectUV: 250,        // peak-to-peak above this marks a window as artifact
};

// SSVEP magnitude of the window centred on t. Returns {v, bad, valid} or null if the data is
// missing. With an artifact mask (p.mask) only the clean samples are used; a window with less
// than p.minValid clean data is returned as bad with v = NaN. With a calibrated spatial filter
// (p.spatialW over p.spatialChans) the channels are first combined into one virtual channel.
export function metricAt(eeg, t, p) {
  const len = Math.max(32, Math.round(p.winSec * eeg.fs));
  const tc = t + (p.latency ?? DEFAULTS.latency);
  const chans = p.spatialW ? p.spatialChans : p.chans;
  // Kalman track: no window — the smoothed state at tc; masked data is bridged, so only a
  // stretch with almost nothing observed around it counts as bad
  if (p.metric === 'kalman') {
    const m = kalmanAt(eeg, tc, p);
    if (!m) return null;
    if (m.valid < (p.minValid ?? 0.4) / 2) return { v: NaN, z: m.z, bad: true, valid: m.valid };
    return { ...m, bad: false };
  }
  const w = windowAt(eeg, tc, len, chans);
  if (!w) return null;
  const i0 = indexAt(eeg, tc) - (len >> 1);
  // Mask on the imaging channels only: AF7/AF8 carry every blink, but blinks hardly touch the
  // 15 Hz band and their small spatial-filter weights scale them down further.
  const wts = windowWeights(eeg, i0, len, { ...p, maskChans: p.chans });
  let valid = 1;
  if (wts) {
    let s = 0; for (const v of wts) s += v;
    valid = s / len;
    if (valid < (p.minValid ?? 0.4)) return { v: NaN, bad: true, valid };
  }
  let bad = false;
  const masking = p.mask && p.mask.mode && p.mask.mode !== 'off';
  if (p.rejectUV > 0 && !masking) for (const x of w) if (peakToPeak(x) > p.rejectUV) bad = true;
  let wins = w;
  if (p.spatialW) {
    const v = new Float64Array(len);
    w.forEach((x, j) => { const k = p.spatialW[j]; for (let i = 0; i < len; i++) v[i] += k * x[i]; });
    wins = [v];
  }
  if (p.metric === 'fbcca') return { v: fbcca(eeg, i0, len, p, wts), bad, valid };
  if (p.metric === 'lockin' || p.metric === 'coherent') {
    const z = lockin(wins, eeg.times.subarray(i0, i0 + len), p.cycles || (tt => tt * p.freq), p.harmonics ?? 2, wts);
    return { v: z.power, z, bad, valid };
  }
  if (wts) wins = wins.map(x => { const d = detrend(x); for (let i = 0; i < len; i++) d[i] *= wts[i]; return d; });
  const spec = spectrum(wins, eeg.fs, { psd: p.psd, segSec: Math.min(p.segSec ?? 2, p.winSec) });
  let v = ssvepMetric(spec, p.freq, p);
  if (p.metric === 'power' && valid > 0) v /= valid;
  return { v, bad, valid };
}

// ------------------------------------------------------------------ Paper eq. (1)
// f(x) = 2x + x1 + x-1 + (x2 + x-2)/2 over overlapping scan lines. Missing neighbours
// (image edges, unfinished lines) are dropped and the weights renormalised, so edges
// are not darkened.
export function combineRows(grid, cols, rows) {
  const W = [[0, 2], [-1, 1], [1, 1], [-2, 0.5], [2, 0.5]];
  const out = new Float32Array(cols * rows).fill(NaN);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    let s = 0, ws = 0;
    for (const [dr, w] of W) {
      const rr = r + dr;
      if (rr < 0 || rr >= rows) continue;
      const v = grid[rr * cols + c];
      if (Number.isFinite(v)) { s += w * v; ws += w; }
    }
    if (ws >= 2 || (ws > 0 && Number.isFinite(grid[r * cols + c]))) out[r * cols + c] = s / ws;
  }
  return out;
}

// ------------------------------------------------------------------ Scan-line imaging
// run: { W, H, rows, cursor, freq, lines: [{row, y, t0, t1, x0, x1}], rotate }
// Positions are in stimulus pixels (screen px for the cursor mode; arbitrary units for the
// real-world rig, where W/H only set the aspect ratio).
export function emptyScanGrid(run, p) {
  const step = p.stepPx || run.cursor / 4;
  const cols = Math.floor(run.W / step) + 1, rows = run.rows;
  return { raw: new Float32Array(cols * rows).fill(NaN), bad: new Uint8Array(cols * rows), z: new Array(cols * rows).fill(null), cols, rows, step };
}

// Fill one scan line L = {row, t0, t1, x0, x1} of grid g: each column's value is the SSVEP
// over the window centred on the moment the cursor passed that x.
export function scanRow(eeg, run, L, p, g) {
  const { cols, step } = g, r = L.row;
  if (p.deconv) { deconvRow(eeg, run, L, p, g); return; }
  for (let c = 0; c < cols; c++) {
    g.raw[r * cols + c] = NaN; g.bad[r * cols + c] = 0; if (g.z) g.z[r * cols + c] = null;
    const x = Math.min(run.W, c * step);
    const a = (x - L.x0) / (L.x1 - L.x0);
    if (a < -0.001 || a > 1.001) continue;
    const m = metricAt(eeg, L.t0 + a * (L.t1 - L.t0), p);
    if (!m) continue;
    g.raw[r * cols + c] = m.v;
    if (m.z && g.z) g.z[r * cols + c] = m.z;
    if (m.bad) g.bad[r * cols + c] = 1;
  }
  // coherent: project this line's lock-in amplitudes onto its own response phase
  if ((p.metric === 'coherent' || (p.metric === 'kalman' && p.kalmanProject)) && g.z) {
    const zs = [];
    for (let c = 0; c < cols; c++) zs.push(g.bad[r * cols + c] && p.reject !== false ? null : g.z[r * cols + c]);
    const v = coherentProject(zs, p.harmonics ?? 2, (p.phaseSpan ?? 8) / (step / run.speed));
    for (let c = 0; c < cols; c++) if (g.z[r * cols + c]) g.raw[r * cols + c] = Number.isFinite(v[c]) ? v[c] : g.raw[r * cols + c];
  }
  if (p.lineRef) applyLineRef(eeg, L, p, g);
}

// Per-line reference patches: each line began with a full-flicker patch (refOn) and a
// no-flicker patch (refOff). Rescale the line so off -> 0 and on -> 1, cancelling slow drift
// of the SSVEP gain (drying electrodes, fatigue) between lines.
function applyLineRef(eeg, L, p, g) {
  if (!L.refOn || !L.refOff || p.metric === 'coherent' || p.kalmanProject || p.deconv) return;
  const at = ([a, b]) => {
    const win = Math.max(1, b - a - 1);
    const m = metricAt(eeg, (a + 1 + b) / 2 - (p.latency ?? DEFAULTS.latency), { ...p, winSec: win });
    return m && !m.bad ? m.v : NaN;
  };
  const on = at(L.refOn), off = at(L.refOff);
  if (!(on > off)) return;
  const { cols } = g, r = L.row;
  for (let c = 0; c < cols; c++) g.raw[r * cols + c] = (g.raw[r * cols + c] - off) / (on - off);
}

// Deconvolution of one scan line. Short-time coherent lock-in values m(t_k) every `hop` s
// are modelled as m = K u + b, where K[k][c] is the fraction of window k during which the
// cursor covered column c. Solving with a smoothness prior recovers detail finer than the
// cursor + analysis-window blur that the direct method is limited to.
export function deconvRow(eeg, run, L, p, g) {
  const { cols, step } = g, r = L.row;
  for (let c = 0; c < cols; c++) { g.raw[r * cols + c] = NaN; g.bad[r * cols + c] = 0; }
  const ws = p.deconvWin ?? 1, hop = p.deconvHop ?? 0.2, half = run.cursor / 2;
  const span = L.t1 - L.t0;
  const X = tau => L.x0 + Math.min(1, Math.max(0, (tau - L.t0) / span)) * (L.x1 - L.x0);
  const pp = { ...p, metric: 'lockin', winSec: ws, deconv: false };
  const ts = [], zs = [];
  for (let t = L.t0 - (p.winSec ?? ws) / 2; t <= L.t1 + (p.winSec ?? ws) / 2; t += hop) {
    const m = metricAt(eeg, t, pp);
    if (!m || (m.bad && p.reject !== false)) continue;
    ts.push(t); zs.push(m.z);
  }
  if (ts.length < 4) return;
  const meas = coherentProject(zs, p.harmonics ?? 2, (p.phaseSpan ?? 8) / hop);
  // No separate offset term: a projected lock-in value is zero-mean for a black image, and an
  // offset would be degenerate with a uniform shift of u (coverage sums are ~constant).
  const n = cols, KtK = new Float64Array(n * n), Ktm = new Float64Array(n);
  const sub = 9, kw = [];
  for (let j = 0; j < sub; j++) kw.push(0.5 - 0.5 * Math.cos(2 * Math.PI * (j + 0.5) / sub));
  const kws = kw.reduce((a, b) => a + b, 0);
  const row = new Float64Array(n);
  for (let k = 0; k < ts.length; k++) {
    row.fill(0);
    for (let j = 0; j < sub; j++) {
      const xc = X(ts[k] - ws / 2 + ws * (j + 0.5) / sub);
      const c0 = Math.max(0, Math.ceil((xc - half) / step)), c1 = Math.min(cols - 1, Math.floor((xc + half) / step));
      for (let c = c0; c <= c1; c++) row[c] += kw[j] / kws;
    }
    for (let i = 0; i < n; i++) { if (!row[i]) continue; Ktm[i] += row[i] * meas[k]; for (let j = 0; j < n; j++) KtK[i * n + j] += row[i] * row[j]; }
  }
  let dmean = 0; for (let i = 0; i < cols; i++) dmean += KtK[i * n + i]; dmean /= cols;
  const lam = (p.deconvLambda ?? 0.5) * dmean;
  for (let c = 1; c < cols - 1; c++) { // second-difference smoothness prior
    const idx = [c - 1, c, c + 1], w = [1, -2, 1];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) KtK[idx[a] * n + idx[b]] += lam * w[a] * w[b];
  }
  for (let i = 0; i < n; i++) KtK[i * n + i] += 1e-6 * dmean;
  const u = solve(KtK, Ktm, n);
  for (let c = 0; c < cols; c++) g.raw[r * cols + c] = u[c];
}

function solve(A, b, n) {
  const M = Float64Array.from(A), x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let piv = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k * n + i]) > Math.abs(M[piv * n + i])) piv = k;
    if (piv !== i) { for (let k = 0; k < n; k++) { const t = M[i * n + k]; M[i * n + k] = M[piv * n + k]; M[piv * n + k] = t; } const t = x[i]; x[i] = x[piv]; x[piv] = t; }
    const d = M[i * n + i] || 1e-12;
    for (let k = i + 1; k < n; k++) {
      const f = M[k * n + i] / d; if (!f) continue;
      for (let j = i; j < n; j++) M[k * n + j] -= f * M[i * n + j];
      x[k] -= f * x[i];
    }
  }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let j = i + 1; j < n; j++) s -= M[i * n + j] * x[j]; x[i] = s / (M[i * n + i] || 1e-12); }
  return x;
}

// One grid per pass (repeated scans of the image); within a pass a redo replaces the row.
export function scanPassGrids(eeg, run, p) {
  const byPass = new Map();
  for (const L of run.lines) {
    if (!(L.t1 > L.t0)) continue;
    const k = L.pass ?? 0;
    if (!byPass.has(k)) byPass.set(k, new Map());
    byPass.get(k).set(L.row, L);
  }
  return [...byPass.keys()].sort((a, b) => a - b).map(k => {
    const g = emptyScanGrid(run, p);
    for (const L of byPass.get(k).values()) scanRow(eeg, run, L, p, g);
    g.pass = k;
    return g;
  });
}

// Combine passes cell by cell: median (robust — a pass ruined by a head movement is outvoted)
// or mean. Artifact-flagged cells are left out when p.reject is on.
export function combinePasses(grids, p) {
  if (grids.length === 1) return grids[0];
  const { cols, rows, step } = grids[0];
  const out = { raw: new Float32Array(cols * rows).fill(NaN), bad: new Uint8Array(cols * rows), z: null, cols, rows, step, passes: grids.length };
  for (let i = 0; i < cols * rows; i++) {
    const v = [];
    for (const g of grids) if (Number.isFinite(g.raw[i]) && !(g.bad[i] && p.reject !== false)) v.push(g.raw[i]);
    if (!v.length) { out.bad[i] = 1; continue; }
    v.sort((a, b) => a - b);
    out.raw[i] = (p.passCombine || 'median') === 'mean' ? v.reduce((a, b) => a + b, 0) / v.length
      : v.length % 2 ? v[v.length >> 1] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  }
  return out;
}

export function scanGrid(eeg, run, p) {
  return combinePasses(scanPassGrids(eeg, run, p), p);
}

// Grid values ready for comparison: artifact cells removed, eq. (1) applied.
function prepared(g, p) {
  const v = Float32Array.from(g.raw);
  for (let i = 0; i < v.length; i++) if (g.bad[i] && p.reject !== false) v[i] = NaN;
  return p.combine === false ? v : combineRows(v, g.cols, g.rows);
}

// Split-half reliability (no ground truth needed). With >= 2 passes: even vs. odd passes
// (different times, so artifacts are independent). With one pass: alternate 0.5 s chunks of
// every window (weaker: an artifact longer than 0.5 s lands in both halves). Spearman-Brown
// gives the expected reliability of the full data.
export function reliability(eeg, run, p) {
  const nPass = new Set(run.lines.map(l => l.pass ?? 0)).size;
  let A, B, method;
  if (nPass >= 2) {
    const grids = scanPassGrids(eeg, run, p);
    A = combinePasses(grids.filter((_, i) => i % 2 === 0), p); B = combinePasses(grids.filter((_, i) => i % 2 === 1), p);
    method = 'even vs odd passes';
  } else {
    A = scanGrid(eeg, run, { ...p, half: 0, minValid: (p.minValid ?? 0.4) / 2 });
    B = scanGrid(eeg, run, { ...p, half: 1, minValid: (p.minValid ?? 0.4) / 2 });
    method = 'interleaved 0.5 s halves';
  }
  const r = correlation(prepared(A, p), prepared(B, p));
  return { r, full: r > -1 ? (2 * r) / (1 + r) : NaN, method };
}

// What an ideal observer would measure: mean stimulus brightness under the cursor at every
// grid point (run.truth is the displayed image's luminance, downsampled).
export function expectedGrid(run, g) {
  const T = run.truth;
  if (!T) return null;
  const sx = T.w / run.W, sy = T.h / run.H, half = run.cursor / 2;
  const out = new Float32Array(g.cols * g.rows);
  for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
    const X = Math.min(run.W, c * g.step), Y = run.rowY[r];
    const x0 = Math.max(0, Math.floor((X - half) * sx)), x1 = Math.min(T.w, Math.ceil((X + half) * sx));
    const y0 = Math.max(0, Math.floor((Y - half) * sy)), y1 = Math.min(T.h, Math.ceil((Y + half) * sy));
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { s += T.lum[y * T.w + x]; n++; }
    out[r * g.cols + c] = n ? s / n : 0;
  }
  return out;
}

// Validity: correlation of the measured grid with the ideal-observer grid. Only meaningful
// on a known target (e.g. the letter N) — that is how a pipeline is calibrated before being
// trusted on unknown scenes.
export function validity(eeg, run, p, g = null) {
  g = g || scanGrid(eeg, run, p);
  const E = expectedGrid(run, g);
  if (!E) return NaN;
  return correlation(prepared(g, p), p.combine === false ? E : combineRows(E, g.cols, g.rows));
}

// Try analysis variants and score each by split-half reliability (and validity on a known
// target). onProgress(i, n, row) is called after each candidate.
export async function autoTune(eeg, run, base, candidates, onProgress) {
  const rows = [];
  for (const [i, c] of candidates.entries()) {
    const p = { ...base, ...c.p };
    const rel = reliability(eeg, run, p);
    const val = run.truth ? validity(eeg, run, p) : NaN;
    const row = { label: c.label, p: c.p, rel: rel.r, relFull: rel.full, val, method: rel.method };
    rows.push(row);
    onProgress?.(i + 1, candidates.length, row);
    await new Promise(r => setTimeout(r, 0));
  }
  const key = r => (Number.isFinite(r.val) ? r.val : r.relFull);
  rows.sort((a, b) => (key(b) ?? -9) - (key(a) ?? -9));
  return rows;
}

export function renderScan(eeg, run, p, grid = null) {
  const g = grid || scanGrid(eeg, run, p);
  const { cols, rows, step } = g;
  const vals = Float32Array.from(g.raw);
  for (let i = 0; i < vals.length; i++) if (g.bad[i] && p.reject !== false && !p.deconv) vals[i] = NaN;
  if (p.rowBaseline) { // remove line-to-line drift: subtract each line's 10th percentile
    for (let r = 0; r < rows; r++) {
      const rowv = Array.from(vals.subarray(r * cols, (r + 1) * cols));
      const b = percentile(rowv, 0.1);
      if (Number.isFinite(b)) for (let c = 0; c < cols; c++) vals[r * cols + c] -= b;
    }
  }
  let comb = p.combine === false ? vals : combineRows(vals, cols, rows);
  const measured = comb.some(Number.isFinite);
  comb = fillNaN(comb, cols, rows);
  // row r sits at y = yFirst + r*dy (row centres of the scan lines)
  const ys = run.rowY;
  const yFirst = ys[0], dy = rows > 1 ? (ys[rows - 1] - ys[0]) / (rows - 1) : 1;
  const outW = p.outW || 480, outH = Math.max(1, Math.round(outW * run.H / run.W));
  let img = new Float32Array(outW * outH);
  for (let oy = 0; oy < outH; oy++) {
    const Y = (oy + 0.5) * run.H / outH, gr = rows > 1 ? (Y - yFirst) / dy : 0;
    for (let ox = 0; ox < outW; ox++) {
      const X = (ox + 0.5) * run.W / outW;
      img[oy * outW + ox] = sample(comb, cols, rows, X / step, gr, p.interp || 'bilinear');
    }
  }
  if (p.blur) img = gaussianBlur(img, outW, outH, p.blur);
  let norm = measured ? normalize(img, { lo: p.lo ?? 0.02, hi: p.hi ?? 0.99, gain: p.gain ?? 1, gamma: p.gamma ?? 1 }) : new Float32Array(img.length);
  let w = outW, h = outH;
  if (run.rotate) ({ grid: norm, w, h } = rotateCW(norm, w, h));
  return { img: norm, w, h, grid: g, combined: comb };
}

// ------------------------------------------------------------------ Free gaze (Fig. 6)
// run: { W, H, freq, gaze: {t:[], x:[], y:[], valid?:[]} }
export function gazeSamples(eeg, run, p) {
  const g = run.gaze;
  if (!g || g.t.length < 4) return [];
  const idx = g.t.map((_, i) => i).filter(i => Number.isFinite(g.x[i]) && Number.isFinite(g.y[i]) && (!g.valid || g.valid[i]));
  const t = idx.map(i => g.t[i]), X = idx.map(i => g.x[i]), Y = idx.map(i => g.y[i]);
  if (t.length < 4) return [];
  const fsg = 60;
  const rx = resampleUniform(t, X, fsg), ry = resampleUniform(t, Y, fsg);
  let xs = rx.v, ys = ry.v;
  if (p.lowpass !== false) { // paper: 1.5 Hz 6th-order Butterworth on the eye X/Y data
    const sos = butterLowpass(6, p.lowpassHz || 1.5, fsg);
    xs = filtfilt(sos, xs); ys = filtfilt(sos, ys);
  }
  const hop = p.hop || 0.25;
  const out = [];
  for (let tt = rx.t0; tt <= rx.t0 + (xs.length - 1) / fsg; tt += hop) {
    const i = Math.min(xs.length - 2, Math.round((tt - rx.t0) * fsg));
    const speed = Math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) * fsg;
    const m = metricAt(eeg, tt, p);
    if (!m || !Number.isFinite(m.v)) continue;
    let w = 1;
    if (p.speedWeight) w = Math.exp(-speed / (p.speedRef || 300));      // future-directions weighting
    if (m.bad) w *= p.reject === false ? 1 : 0;                          // blinks / artifacts
    out.push({ t: tt, x: xs[i], y: ys[i], v: m.v, w, speed });
  }
  return out;
}

export function renderGaze(samples, run, p) {
  const outW = p.outW || 480, outH = Math.round(outW * run.H / run.W), s = outW / run.W;
  const pts = samples.filter(q => q.w > (p.minWeight ?? 0.2));
  let img;
  if (pts.length >= 3 && (p.method || 'triangulate') === 'triangulate') {
    const seen = new Set(), xs = [], ys = [], vs = [];
    for (const q of pts) {
      const k = Math.round(q.x * s / 2) + ',' + Math.round(q.y * s / 2);
      if (seen.has(k)) continue; seen.add(k);
      xs.push(q.x * s); ys.push(q.y * s); vs.push(q.v);
    }
    img = xs.length >= 3 ? rasterizeTriangles(xs, ys, vs, delaunay(xs, ys), outW, outH) : new Float32Array(outW * outH).fill(NaN);
  } else {
    img = splat(pts.map(q => q.x * s), pts.map(q => q.y * s), pts.map(q => q.v), pts.map(q => q.w), outW, outH, (p.sigmaPx || 40) * s);
  }
  const norm = normalize(img, { lo: p.lo ?? 0.02, hi: p.hi ?? 0.99, gain: p.gain ?? 1 });
  return { img: norm, w: outW, h: outH };
}

// ------------------------------------------------------------------ Chunks (Fig. 12)
// run: { cols, rows, freq, cells: [{c, r, t0, t1}] } — fixation on each chunk.
export function renderChunks(eeg, run, p) {
  const grid = new Float32Array(run.cols * run.rows).fill(NaN);
  for (const cell of run.cells) {
    const settle = p.settle ?? 1;
    const t0 = cell.t0 + settle, t1 = cell.t1;
    if (t1 - t0 < 1) continue;
    const m = metricAt(eeg, (t0 + t1) / 2 - (p.latency ?? DEFAULTS.latency), { ...p, winSec: t1 - t0 });
    // a chunk is one fixation: keep it even if a blink flagged it (rejecting would leave a hole)
    if (m) grid[cell.r * run.cols + cell.c] = m.v;
  }
  const norm = normalize(grid, { lo: p.lo ?? 0, hi: p.hi ?? 1, gain: p.gain ?? 1 });
  const bs = p.block || 40, w = run.cols * bs, h = run.rows * bs, img = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = norm[Math.floor(y / bs) * run.cols + Math.floor(x / bs)];
  return { img, w, h, grid };
}

// ------------------------------------------------------------------ Timelines (Fig. 14)
// Trailing-window SSVEP metric every `hop` s between t0 and t1.
export function timeline(eeg, t0, t1, p) {
  const out = [];
  const half = p.winSec / 2;
  for (let t = t0 + p.winSec; t <= t1; t += p.hop || 0.5) {
    const m = metricAt(eeg, t - half - (p.latency ?? DEFAULTS.latency), p);
    if (m && Number.isFinite(m.v)) out.push({ t, v: m.v, bad: m.bad });
  }
  return out;
}

// Robust threshold (median + k * 1.4826 * MAD) and onset detection with refractory time.
export function detectEvents(series, { k = 2, refractory = 8, useBad = false } = {}) {
  const vals = series.filter(s => useBad || !s.bad).map(s => s.v);
  const med = percentile(vals, 0.5);
  const mad = percentile(vals.map(v => Math.abs(v - med)), 0.5) * 1.4826 || 1e-9;
  const thr = med + k * mad;
  const onsets = [];
  let above = false, last = -Infinity;
  for (const s of series) {
    const hit = s.v > thr && (useBad || !s.bad);
    if (hit && !above && s.t - last >= refractory) { onsets.push(s.t); last = s.t; }
    above = hit;
  }
  return { thr, med, mad, onsets };
}

// Hit rate of detections against user-indicated events (paper reports 89 %).
export function scoreEvents(onsets, marks, { before = 5, after = 5 } = {}) {
  let hits = 0;
  const matched = new Set();
  for (const e of marks) {
    const d = onsets.findIndex(o => o >= e - before && o <= e + after);
    if (d >= 0) { hits++; matched.add(d); }
  }
  return {
    hits, events: marks.length, hitRate: marks.length ? hits / marks.length : NaN,
    detections: onsets.length, falseAlarms: onsets.filter((_, i) => !matched.has(i)).length,
  };
}

// ------------------------------------------------------------------ Visual field (Fig. 4)
// Samples along a moving-stimulus path: path = {t:[], x:[], y:[]} (px or mm)
export function pathSamples(eeg, path, p) {
  const out = [];
  if (!path || path.t.length < 2) return out;
  const hop = p.hop || 0.25;
  let j = 0;
  for (let t = path.t[0]; t <= path.t[path.t.length - 1]; t += hop) {
    while (j < path.t.length - 2 && path.t[j + 1] < t) j++;
    const dt = path.t[j + 1] - path.t[j], a = dt > 0 ? Math.min(1, Math.max(0, (t - path.t[j]) / dt)) : 0;
    const x = path.x[j] + (path.x[j + 1] - path.x[j]) * a, y = path.y[j] + (path.y[j + 1] - path.y[j]) * a;
    const vx = dt > 0 ? (path.x[j + 1] - path.x[j]) / dt : 0;
    const m = metricAt(eeg, t, p);
    if (m && Number.isFinite(m.v)) out.push({ t, x, y, v: m.v, bad: m.bad, dir: vx >= 0 ? 1 : -1 });
  }
  return out;
}

// Mean response vs position along one axis, split by sweep direction (hysteresis).
export function profile(samples, key, lo, hi, bins = 30) {
  const mk = () => Array.from({ length: bins }, () => ({ s: 0, n: 0 }));
  const fwd = mk(), bwd = mk(), all = mk();
  for (const q of samples) {
    if (q.bad) continue;
    const b = Math.floor((q[key] - lo) / (hi - lo) * bins);
    if (b < 0 || b >= bins) continue;
    for (const arr of [all, q.dir > 0 ? fwd : bwd]) { arr[b].s += q.v; arr[b].n++; }
  }
  const fin = arr => arr.map(c => (c.n ? c.s / c.n : NaN));
  const centers = Array.from({ length: bins }, (_, i) => lo + (i + 0.5) * (hi - lo) / bins);
  return { centers, all: fin(all), fwd: fin(fwd), bwd: fin(bwd) };
}

// Full width at half maximum of a (possibly noisy) profile, in the profile's units.
export function fwhm(centers, vals) {
  const ok = vals.map((v, i) => [centers[i], v]).filter(([, v]) => Number.isFinite(v));
  if (ok.length < 3) return NaN;
  const base = percentile(ok.map(o => o[1]), 0.1), peak = Math.max(...ok.map(o => o[1]));
  const half = base + (peak - base) / 2;
  const above = ok.filter(o => o[1] >= half).map(o => o[0]);
  return above.length ? Math.max(...above) - Math.min(...above) : NaN;
}
