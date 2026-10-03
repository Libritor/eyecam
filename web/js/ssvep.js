// Advanced SSVEP detectors (beyond the paper's power ratio):
//
//   fbcca     Filter-bank CCA (Chen et al. 2015): canonical correlation between the EEG
//             channels and sin/cos references at the flicker frequency and harmonics, in
//             several sub-bands, combined with weights n^-1.25 + 0.25.
//   lockin    Lock-in demodulation against the *actual* flicker phase (logged per frame),
//             giving a complex amplitude per channel and harmonic.
//   coherent  Lock-in, then projected onto the response phase estimated per scan line, so
//             noise in quadrature is rejected and the estimate is unbiased and linear in
//             stimulus brightness (done at the scan level in recon.js).
//
// References use cycles(t): the stimulus' cumulative cycle count at page-clock time t. With
// a flicker log (raster runs) it is the exact displayed phase, so displays whose refresh is
// not a multiple of 2f (time-dithered flicker) are still demodulated correctly.

import { butterHighpass, butterLowpass, notch, sosState, sosRun, detrend } from './dsp.js';
import { indexAt } from './eeg-store.js';

// cumulative-cycles function from a run's flicker log {t: [], c: []}, else freq * t
export function makeCycles(run, freq) {
  const log = run?.flicker;
  if (!log || log.t.length < 2) return t => t * freq;
  const T = log.t, C = log.c;
  return t => {
    const n = T.length; // the log may still be growing during a live run
    if (n < 2) return t * freq;
    const slopeEnd = (C[n - 1] - C[0]) / (T[n - 1] - T[0]) || freq;
    if (t <= T[0]) return C[0] - (T[0] - t) * slopeEnd;
    if (t >= T[n - 1]) return C[n - 1] + (t - T[n - 1]) * slopeEnd;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    return C[lo] + (C[hi] - C[lo]) * (t - T[lo]) / (T[hi] - T[lo]);
  };
}

const HARM_W = [1, 0.5, 0.33, 0.25, 0.2];

// ------------------------------------------------------------------------- lock-in
// Weighted linear detrend (weights 0/1 mark usable samples).
function wdetrend(x, w) {
  const n = x.length, out = new Float64Array(n);
  let s0 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let k = 0; k < n; k++) if (w[k] > 0) { s0 += w[k]; sx += w[k] * k; sy += w[k] * x[k]; sxx += w[k] * k * k; sxy += w[k] * k * x[k]; }
  const den = s0 * sxx - sx * sx, b = den ? (s0 * sxy - sx * sy) / den : 0, a = s0 ? (sy - b * sx) / s0 : 0;
  for (let k = 0; k < n; k++) out[k] = w[k] > 0 ? x[k] - (a + b * k) : 0;
  return out;
}

// Complex amplitude per channel (rows) and harmonic (cols) of a window. Optional per-sample
// weights (artifact mask): masked samples are excluded from the fit and the normalisation.
export function lockin(wins, times, cycles, harmonics = 2, wts = null) {
  const n = times.length, cyc = new Float64Array(n);
  for (let k = 0; k < n; k++) cyc[k] = cycles(times[k]);
  let ws = 0; const hw = new Float64Array(n);
  for (let k = 0; k < n; k++) { hw[k] = (0.5 - 0.5 * Math.cos(2 * Math.PI * k / (n - 1))) * (wts ? wts[k] : 1); ws += hw[k]; }
  if (!(ws > 0)) ws = 1;
  const re = [], im = [];
  let power = 0;
  for (const x of wins) {
    const d = wts ? wdetrend(x, wts) : detrend(x), r = [], q = [];
    for (let h = 1; h <= harmonics; h++) {
      let sr = 0, si = 0;
      for (let k = 0; k < n; k++) { const a = 2 * Math.PI * h * cyc[k], v = d[k] * hw[k]; sr += v * Math.cos(a); si -= v * Math.sin(a); }
      sr *= 2 / ws; si *= 2 / ws;
      r.push(sr); q.push(si);
      power += (HARM_W[h - 1] ?? 0.2) * (sr * sr + si * si);
    }
    re.push(r); im.push(q);
  }
  return { re, im, power: power / wins.length };
}

// Phase alignment along a scan line: zs = consecutive lock-in results (null allowed).
// The response phase drifts slowly (latency wander, clock mapping), so it is tracked
// locally: for each sample, the |Z|-weighted mean phase of its ±span neighbours. Each
// sample is then projected onto that phase (rejecting the quadrature half of the noise).
export function coherentProject(zs, harmonics = 2, span = Infinity) {
  const first = zs.find(Boolean);
  if (!first) return zs.map(() => NaN);
  const n = zs.length, nch = first.re.length;
  // prefix sums of Z|Z| per channel/harmonic
  const K = nch * harmonics, pr = new Float64Array((n + 1) * K), pi = new Float64Array((n + 1) * K);
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) {
    const z = zs[i], c = Math.floor(k / harmonics), h = k % harmonics;
    let a = 0, b = 0;
    if (z) { const r = z.re[c][h], q = z.im[c][h], m = Math.hypot(r, q); a = r * m; b = q * m; }
    pr[(i + 1) * K + k] = pr[i * K + k] + a; pi[(i + 1) * K + k] = pi[i * K + k] + b;
  }
  const s = Number.isFinite(span) ? Math.max(1, Math.round(span)) : n;
  return zs.map((z, i) => {
    if (!z) return NaN;
    const lo = Math.max(0, i - s), hi = Math.min(n, i + s + 1);
    let v = 0;
    for (let c = 0; c < nch; c++) for (let h = 0; h < harmonics; h++) {
      const k = c * harmonics + h;
      const sr = pr[hi * K + k] - pr[lo * K + k], si = pi[hi * K + k] - pi[lo * K + k], m = Math.hypot(sr, si) || 1;
      v += (HARM_W[h] ?? 0.2) * (z.re[c][h] * sr / m + z.im[c][h] * si / m);
    }
    return v / nch;
  });
}

// ------------------------------------------------------------------------- FBCCA
const fbCache = new WeakMap();

// Causally band-pass the whole buffer into sub-bands (cached, extended incrementally as a
// live store grows). Causal filtering is fine: CCA with sin/cos references is phase-blind.
function filterBank(eeg, freq, nb, mains) {
  const key = `${eeg.fs}|${freq}|${nb}|${mains}|${eeg.channels.length}`;
  let c = fbCache.get(eeg);
  if (!c || c.key !== key || c.n > eeg.n || c.t0 !== eeg.times[0]) {
    const hi = Math.min(88, 0.42 * eeg.fs);
    const sos = [];
    for (let b = 1; b <= nb; b++) {
      const lo = Math.max(1, b * freq - 2);
      const s = [...butterHighpass(4, lo, eeg.fs), ...butterLowpass(4, hi, eeg.fs)];
      if (mains && mains < eeg.fs / 2) s.push(notch(mains, eeg.fs));
      sos.push(s);
    }
    c = { key, n: 0, t0: eeg.times[0], sos, hi, st: sos.map(s => eeg.channels.map(() => sosState(s))), data: sos.map(() => eeg.channels.map(() => new Float32Array(Math.max(1024, eeg.n)))) };
    fbCache.set(eeg, c);
  }
  if (c.n < eeg.n) {
    for (let b = 0; b < c.sos.length; b++) for (let ch = 0; ch < eeg.channels.length; ch++) {
      let out = c.data[b][ch];
      if (out.length < eeg.n) { const g = new Float32Array(Math.max(eeg.n, out.length * 2)); g.set(out.subarray(0, c.n)); out = c.data[b][ch] = g; }
      sosRun(c.sos[b], c.st[b][ch], eeg.data[ch], out, c.n, eeg.n);
    }
    c.n = eeg.n;
  }
  return c;
}

export function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j];
  }
  return L;
}
// solve L y = b (lower triangular), columns of B
export function forwardSub(L, n, B, m) {
  const Y = new Float64Array(n * m);
  for (let col = 0; col < m; col++) for (let i = 0; i < n; i++) {
    let s = B[i * m + col];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * Y[k * m + col];
    Y[i * m + col] = s / L[i * n + i];
  }
  return Y;
}

// Largest canonical correlation squared between X (p×n rows) and Y (q×n rows).
export function ccaRho2(X, Y) {
  const p = X.length, q = Y.length, n = X[0].length;
  const cen = A => A.map(r => { let m = 0; for (const v of r) m += v; m /= n; return Float64Array.from(r, v => v - m); });
  const Xc = cen(X), Yc = cen(Y);
  const cov = (A, B) => { const out = new Float64Array(A.length * B.length); for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) { let s = 0; const a = A[i], b = B[j]; for (let k = 0; k < n; k++) s += a[k] * b[k]; out[i * B.length + j] = s; } return out; };
  const Cxx = cov(Xc, Xc), Cyy = cov(Yc, Yc), Cxy = cov(Xc, Yc);
  let tr = 0; for (let i = 0; i < p; i++) tr += Cxx[i * p + i];
  for (let i = 0; i < p; i++) Cxx[i * p + i] += 1e-6 * tr / p + 1e-12;
  let ty = 0; for (let i = 0; i < q; i++) ty += Cyy[i * q + i];
  for (let i = 0; i < q; i++) Cyy[i * q + i] += 1e-6 * ty / q + 1e-12;
  const Lx = cholesky(Cxx, p), Ly = cholesky(Cyy, q);
  // T = Lx^-1 Cxy Ly^-T  (p×q)
  const A = forwardSub(Lx, p, Cxy, q);                 // Lx^-1 Cxy
  const At = new Float64Array(q * p); for (let i = 0; i < p; i++) for (let j = 0; j < q; j++) At[j * p + i] = A[i * q + j];
  const Tt = forwardSub(Ly, q, At, p);                 // Ly^-1 (Lx^-1 Cxy)^T = T^T (q×p)
  // M = T T^T (p×p), largest eigenvalue by power iteration
  const M = new Float64Array(p * p);
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < q; k++) s += Tt[k * p + i] * Tt[k * p + j]; M[i * p + j] = s; }
  let v = new Float64Array(p).fill(1), lam = 0;
  for (let it = 0; it < 60; it++) {
    const w = new Float64Array(p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) w[i] += M[i * p + j] * v[j];
    const nrm = Math.hypot(...w) || 1;
    lam = nrm; v = w.map(x => x / nrm);
  }
  return Math.min(1, lam);
}

// FBCCA score of the window of `len` samples starting at sample i0. Optional sample weights
// (artifact mask) drop masked samples; p.spatialW / p.spatialChans collapse the channels
// into one virtual channel first (calibrated spatial filter).
export function fbcca(eeg, i0, len, p, wts = null) {
  const nb = p.fbBands ?? 3;
  const fb = filterBank(eeg, p.freq, nb, p.mains ?? 60);
  const H = Math.max(1, Math.min(p.fbHarmonics ?? 4, Math.floor(fb.hi / p.freq)));
  const cycles = p.cycles || (t => t * p.freq);
  const idx = [];
  for (let k = 0; k < len; k++) if (!wts || wts[k] > 0) idx.push(i0 + k);
  if (idx.length < 16) return NaN;
  const m = idx.length, Y = [];
  for (let h = 1; h <= H; h++) {
    const s = new Float64Array(m), c = new Float64Array(m);
    for (let k = 0; k < m; k++) { const a = 2 * Math.PI * h * cycles(eeg.times[idx[k]]); s[k] = Math.sin(a); c[k] = Math.cos(a); }
    Y.push(s, c);
  }
  let score = 0;
  for (let b = 0; b < nb; b++) {
    let X;
    if (p.spatialW) {
      const v = new Float64Array(m);
      p.spatialChans.forEach((ch, j) => { const d = fb.data[b][ch], w = p.spatialW[j]; for (let k = 0; k < m; k++) v[k] += w * d[idx[k]]; });
      X = [v];
    } else X = p.chans.map(ch => Float64Array.from(idx, i => fb.data[b][ch][i]));
    score += (Math.pow(b + 1, -1.25) + 0.25) * ccaRho2(X, Y);
  }
  return score;
}

export { indexAt };
