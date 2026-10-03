// Kalman tracking of the SSVEP amplitude (metric 'kalman').
//
// The windowed detectors (paper metric, FBCCA, lock-in) estimate the response from a fixed
// window around each image column, so an artifact either costs the whole window or (with
// the mask) shrinks it, and every column is computed from scratch. Here the response is
// treated as a hidden state that evolves slowly, in the BCI-decoder sense:
//
//   state        x_k = complex amplitude of the response at harmonic h (re, im) per channel,
//                optionally with its rate of change (order 2)
//   transition   x_k = A x_{k-1} + w_k      random walk (order 1) or integrated random walk
//                                           (order 2: smooth trend, sharper equivalent kernel)
//   observation  z_k = x_k + q_k            z_k = 2/n Σ y·e^{-i2πh·cycles(t)} over one flicker
//                                           cycle of high-passed EEG (exact for any harmonic,
//                                           and the 2f image term averages out per cycle)
//
// Each cycle's measurement noise R_k comes from the local EEG variance, so muscle bursts and
// electrode pops are down-weighted in proportion to how noisy they are instead of being cut
// out with a hard threshold (on Muse data this beats the hard artifact mask). Masked samples (artifact mask, split-half selector) are simply
// not observed: the filter predicts through the gap and the smoother bridges it. A forward
// filter + backward Rauch–Tung–Striebel pass gives a zero-lag estimate at every cycle,
// computed once for the whole recording and then sampled wherever an image column needs it.
//
// The power is bias-corrected (|x|² minus its posterior variance), so the noise floor of a
// black region sits near zero instead of at the noise power.

import { butterHighpass, notch, filtfilt } from './dsp.js';
import { windowWeights } from './artifacts.js';

const HARM_W = [1, 0.5, 0.33, 0.25, 0.2];
const cache = new WeakMap();

// Smoothing time constant (s) for a given analysis window: chosen so that the smoothed
// estimate averages about as much data as a Hann window of winSec (equal noise), measured
// on the equivalent kernels (see tests.js). Order 2 keeps more of the edge sharpness.
export function kalmanTau(p) {
  if (p.kalmanTau > 0) return p.kalmanTau;
  return (p.winSec || 2) * ((p.kalmanOrder ?? 2) === 2 ? 0.30 : 0.17);
}

function paramKey(eeg, p) {
  const chans = p.spatialW ? p.spatialChans : p.chans;
  const m = p.mask;
  return [eeg.fs, p.freq, p.harmonics ?? 2, kalmanTau(p), p.kalmanOrder ?? 2, p.kalmanRobust ?? 1, p.mains ?? 60,
    chans.join(','), p.spatialW ? p.spatialW.map(w => w.toFixed(4)).join(',') : '',
    m?.mode ?? 'off', m?.k ?? '', m?.mode === 'calib' ? JSON.stringify(m.thr) : '', (p.chans || []).join(','), p.half ?? ''].join('|');
}

// Smoothed track for the whole recording (cached; recomputed as a live store grows).
export function kalmanTrack(eeg, p) {
  let per = cache.get(eeg);
  if (!per) { per = new Map(); cache.set(eeg, per); }
  const key = paramKey(eeg, p);
  const cycles = p.cycles || (t => t * p.freq);
  let c = per.get(key);
  // live: refresh at most every 0.5 s of new data (the smoother needs the future anyway)
  if (c && c.cycles === cycles && c.t0 === eeg.times[0] && (eeg.n - c.n < eeg.fs * 0.5 || c.n === eeg.n)) return c;
  if (per.size > 24) per.clear();
  c = computeTrack(eeg, p, cycles);
  c.cycles = cycles; c.t0 = eeg.times[0]; c.n = eeg.n;
  per.set(key, c);
  return c;
}

function computeTrack(eeg, p, cycles) {
  const n = eeg.n, fs = eeg.fs, H = Math.max(1, Math.min(p.harmonics ?? 2, 5));
  const order = (p.kalmanOrder ?? 2) === 2 ? 2 : 1;
  const chans = p.spatialW ? p.spatialChans : p.chans;
  const empty = { k: 0, t: new Float64Array(0) };
  if (n < fs * 2 || !chans?.length) return empty;

  // 1. signals: spatially filtered virtual channel or each analysis channel; drift and slow
  // artifacts removed (zero-phase, so the response phase is not shifted), mains notched
  const hp = Math.max(2, Math.min(0.5 * p.freq, 8));
  const sos = butterHighpass(4, hp, fs);
  const mains = p.mains ?? 60;
  if (mains && mains < fs / 2 - 1) sos.push(notch(mains, fs));
  let raw;
  if (p.spatialW) {
    const v = new Float64Array(n);
    chans.forEach((ch, j) => { const d = eeg.data[ch], w = p.spatialW[j]; for (let i = 0; i < n; i++) v[i] += w * (Number.isFinite(d[i]) ? d[i] : 0); });
    raw = [v];
  } else raw = chans.map(ch => Float64Array.from(eeg.data[ch].subarray(0, n), v => (Number.isFinite(v) ? v : 0)));
  const sig = raw.map(x => filtfilt(sos, x));
  const nanAt = new Uint8Array(n);
  for (const ch of chans) { const d = eeg.data[ch]; for (let i = 0; i < n; i++) if (!Number.isFinite(d[i])) nanAt[i] = 1; }
  const wts = windowWeights(eeg, 0, n, { ...p, maskChans: p.chans });

  // 2. one observation per flicker cycle
  const cyc = new Float64Array(n);
  for (let i = 0; i < n; i++) cyc[i] = cycles(eeg.times[i]);
  const bins = [];
  let i = 0;
  while (i < n) {
    const b = Math.floor(cyc[i]);
    let j = i; while (j < n && Math.floor(cyc[j]) === b) j++;
    // a bin spanning a data gap would mix phases unevenly: split it off
    const tooLong = eeg.times[j - 1] - eeg.times[i] > 2.5 / p.freq;
    bins.push({ i, j, skip: tooLong });
    i = j;
  }
  const K = bins.length, nch = sig.length, NC = nch * H;
  const t = new Float64Array(K), zr = new Float64Array(K * NC), zi = new Float64Array(K * NC);
  const varOwn = new Float64Array(K * nch), cnt = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const { i: a, j: b, skip } = bins[k];
    t[k] = (eeg.times[a] + eeg.times[b - 1]) / 2;
    if (skip) continue;
    let m = 0;
    for (let s = a; s < b; s++) if (!nanAt[s] && (!wts || wts[s] > 0)) m++;
    // need most of the cycle: a partial cycle no longer cancels the other harmonics
    if (m < 0.75 * (b - a) || m < 4) continue;
    cnt[k] = m;
    for (let ch = 0; ch < nch; ch++) {
      const x = sig[ch];
      let s1 = 0, s2 = 0;
      for (let s = a; s < b; s++) if (!nanAt[s] && (!wts || wts[s] > 0)) { s1 += x[s]; s2 += x[s] * x[s]; }
      varOwn[k * nch + ch] = Math.max(1e-6, s2 / m - (s1 / m) ** 2);
      for (let h = 1; h <= H; h++) {
        let re = 0, im = 0;
        for (let s = a; s < b; s++) if (!nanAt[s] && (!wts || wts[s] > 0)) {
          const ph = 2 * Math.PI * h * cyc[s];
          re += x[s] * Math.cos(ph); im -= x[s] * Math.sin(ph);
        }
        zr[k * NC + ch * H + h - 1] = 2 * re / m; zi[k * NC + ch * H + h - 1] = 2 * im / m;
      }
    }
  }

  // 3. measurement noise per cycle: local background variance (±1 s, observed cycles only);
  // a cycle far noisier than that (a sudden burst or pop) uses its own variance instead
  const half = Math.max(1, Math.round(p.freq));
  const robust = p.kalmanRobust ?? 1;
  const R = new Float64Array(K * nch);
  for (let ch = 0; ch < nch; ch++) {
    let s = 0, c = 0, lo = 0, hi = 0;
    for (let k = 0; k < K; k++) {
      while (hi < K && hi <= k + half) { if (cnt[hi]) { s += varOwn[hi * nch + ch]; c++; } hi++; }
      while (lo < k - half) { if (cnt[lo]) { s -= varOwn[lo * nch + ch]; c--; } lo++; }
      if (!cnt[k]) continue;
      const bg = c ? s / c : varOwn[k * nch + ch], own = varOwn[k * nch + ch];
      R[k * nch + ch] = 2 * (robust && own > 4 * bg ? own : bg) / cnt[k];
    }
  }

  // 4. Kalman filter + RTS smoother per channel. The covariance recursion depends only on R
  // and q, so it is shared by the re/im parts of every harmonic of that channel.
  const N = Math.max(1, kalmanTau(p) * p.freq); // time constant in cycles
  const xr = new Float64Array(K * NC), xi = new Float64Array(K * NC), Pv = new Float64Array(K * nch);
  for (let ch = 0; ch < nch; ch++) {
    const Rs = []; for (let k = 0; k < K; k++) if (cnt[k]) Rs.push(R[k * nch + ch]);
    if (!Rs.length) continue;
    Rs.sort((a, b) => a - b);
    const Rmed = Rs[Rs.length >> 1];
    const q = order === 1 ? Rmed / (N * N) : Rmed / N ** 4;
    smooth(K, t, p.freq, order, q, Rmed * 1e4, k => (cnt[k] ? R[k * nch + ch] : 0), H, ch, NC, zr, zi, xr, xi, Pv, nch);
  }
  return { k: K, t, xr, xi, P: Pv, nch, H, NC, cnt };
}

// Forward Kalman filter + backward RTS pass, scalar (order 1) or 2-state (order 2) model per
// component. Writes smoothed level into xr/xi and its variance into Pv.
function smooth(K, t, freq, order, q, P0, Rof, H, ch, NC, zr, zi, xr, xi, Pv, nch) {
  const comps = H * 2;
  const S = order;            // state size per component
  const SS = S * S;
  const xf = new Float64Array(K * comps * S), Pf = new Float64Array(K * SS), Pp = new Float64Array(K * SS);
  const x = new Float64Array(comps * S);
  let P = order === 1 ? [P0] : [P0, 0, 0, P0];
  let tPrev = t[0];
  for (let k = 0; k < K; k++) {
    const d = Math.max(1e-3, (t[k] - tPrev) * freq); tPrev = t[k]; // step in cycles
    // predict
    let Pq;
    if (order === 1) {
      Pq = [P[0] + q * d];
    } else {
      // F = [[1,d],[0,1]], Q = q[[d³/3, d²/2],[d²/2, d]]
      const [a, b, c, e] = P;
      const p00 = a + d * (b + c) + d * d * e, p01 = b + d * e, p10 = c + d * e, p11 = e;
      Pq = [p00 + q * d * d * d / 3, p01 + q * d * d / 2, p10 + q * d * d / 2, p11 + q * d];
      for (let j = 0; j < comps; j++) x[j * 2] += d * x[j * 2 + 1];
    }
    Pp.set(Pq, k * SS);
    const R = Rof(k);
    if (R > 0) {
      const sInn = Pq[0] + R;
      const G0 = Pq[0] / sInn, G1 = order === 2 ? Pq[2] / sInn : 0;
      for (let j = 0; j < comps; j++) {
        const h = j >> 1, z = (j & 1 ? zi : zr)[k * NC + ch * H + h];
        const inn = z - x[j * S];
        x[j * S] += G0 * inn;
        if (order === 2) x[j * 2 + 1] += G1 * inn;
      }
      P = order === 1 ? [(1 - G0) * Pq[0]]
        : [(1 - G0) * Pq[0], (1 - G0) * Pq[1], Pq[2] - G1 * Pq[0], Pq[3] - G1 * Pq[1]];
    } else P = Pq;
    xf.set(x, k * comps * S);
    Pf.set(P, k * SS);
  }
  // RTS backward pass
  const xs = Float64Array.from(xf.subarray((K - 1) * comps * S, K * comps * S));
  let Ps = Array.from(Pf.subarray((K - 1) * SS, K * SS));
  const out = (k, xsk, Psk) => {
    for (let j = 0; j < comps; j++) {
      const h = j >> 1;
      (j & 1 ? xi : xr)[k * NC + ch * H + h] = xsk[j * S];
    }
    Pv[k * nch + ch] = Psk[0];
  };
  out(K - 1, xs, Ps);
  for (let k = K - 2; k >= 0; k--) {
    const d = Math.max(1e-3, (t[k + 1] - t[k]) * freq);
    const pf = Pf.subarray(k * SS, k * SS + SS), pp = Pp.subarray((k + 1) * SS, (k + 1) * SS + SS);
    const xfk = xf.subarray(k * comps * S, (k + 1) * comps * S);
    if (order === 1) {
      const C = pf[0] / pp[0];
      for (let j = 0; j < comps; j++) xs[j] = xfk[j] + C * (xs[j] - xfk[j]);
      Ps = [pf[0] + C * C * (Ps[0] - pp[0])];
    } else {
      // C = Pf F^T Pp^-1
      const [a, b, c, e] = pf;
      const m00 = a + d * b, m01 = b, m10 = c + d * e, m11 = e;   // Pf F^T
      const det = pp[0] * pp[3] - pp[1] * pp[2] || 1e-300;
      const i00 = pp[3] / det, i01 = -pp[1] / det, i10 = -pp[2] / det, i11 = pp[0] / det;
      const C00 = m00 * i00 + m01 * i10, C01 = m00 * i01 + m01 * i11, C10 = m10 * i00 + m11 * i10, C11 = m10 * i01 + m11 * i11;
      for (let j = 0; j < comps; j++) {
        const l = xfk[j * 2], s = xfk[j * 2 + 1];
        const dl = xs[j * 2] - (l + d * s), ds = xs[j * 2 + 1] - s; // x_s(k+1) - F x_f(k)
        xs[j * 2] = l + C00 * dl + C01 * ds;
        xs[j * 2 + 1] = s + C10 * dl + C11 * ds;
      }
      // Ps = Pf + C (Ps' - Pp) C^T
      const D00 = Ps[0] - pp[0], D01 = Ps[1] - pp[1], D10 = Ps[2] - pp[2], D11 = Ps[3] - pp[3];
      const E00 = C00 * D00 + C01 * D10, E01 = C00 * D01 + C01 * D11, E10 = C10 * D00 + C11 * D10, E11 = C10 * D01 + C11 * D11;
      Ps = [a + E00 * C00 + E01 * C01, b + E00 * C10 + E01 * C11, c + E10 * C00 + E11 * C01, e + E10 * C10 + E11 * C11];
    }
    out(k, xs, Ps);
  }
}

// Smoothed amplitude at page-clock time tc: {v, z:{re,im,power}, valid} or null outside data.
export function kalmanAt(eeg, tc, p) {
  const c = kalmanTrack(eeg, p);
  if (!c.k || tc < c.t[0] || tc > c.t[c.k - 1]) return null;
  let lo = 0, hi = c.k - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c.t[m] <= tc) lo = m; else hi = m; }
  if (c.t[hi] - c.t[lo] > 0.5) return null; // inside a data gap
  const a = c.t[hi] > c.t[lo] ? (tc - c.t[lo]) / (c.t[hi] - c.t[lo]) : 0;
  const { nch, H, NC } = c;
  const re = [], im = [];
  let power = 0;
  for (let ch = 0; ch < nch; ch++) {
    const r = [], q = [];
    const P = c.P[lo * nch + ch] * (1 - a) + c.P[hi * nch + ch] * a;
    for (let h = 0; h < H; h++) {
      const j = ch * H + h;
      const vr = c.xr[lo * NC + j] * (1 - a) + c.xr[hi * NC + j] * a;
      const vi = c.xi[lo * NC + j] * (1 - a) + c.xi[hi * NC + j] * a;
      r.push(vr); q.push(vi);
      power += (HARM_W[h] ?? 0.2) * (vr * vr + vi * vi - 2 * P); // minus posterior variance (re + im)
    }
    re.push(r); im.push(q);
  }
  power /= nch;
  // fraction of observed cycles within ±tau (reported like the mask's valid fraction)
  const span = kalmanTau(p) * p.freq;
  let s = 0, m = 0;
  for (let k = Math.max(0, Math.round(lo - span)); k <= Math.min(c.k - 1, Math.round(hi + span)); k++) { m++; if (c.cnt[k]) s++; }
  return { v: power, z: { re, im, power }, valid: m ? s / m : 0 };
}
