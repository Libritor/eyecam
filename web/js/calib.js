// Calibration analysis.
//
// Input: a calibration recording with
//   blocks: flicker ON / OFF blocks (5 s each, fixating a central dot) with cycle numbers
//   tasks:  prompted artifacts — rest, blink, clench, swallow, move
// Output (all cross-validated where it matters):
//   1. per-channel SSVEP amplitude (µV) and ON/OFF power ratio
//   2. spatial filter: weights over all channels maximising ON/OFF power at the flicker
//      frequency (generalised eigenvector). Trained on odd cycles, tested on even, and vice
//      versa; adopted only if it beats the plain channel average on held-out data. On a Muse
//      AF7/AF8 carry little SSVEP but share the forehead-reference noise of TP9/TP10, so the
//      filter can learn to subtract it.
//   3. detectability d' of ON vs OFF per 1 s, the window needed for a clean image
//      (d' ≈ d'1s·√T ≥ 2.5) and a go / marginal / no-go verdict with a scan plan
//   4. personal artifact thresholds (p2p, EMG) separating prompted artifacts from clean rest

import { lockin, cholesky, forwardSub, makeCycles } from './ssvep.js';
import { indexAt } from './eeg-store.js';
import { blockFeatures } from './artifacts.js';
import { percentile, peakToPeak } from './dsp.js';

const HW = [1, 0.5];

function segPower(z, w) { // filtered (w over channels) or per-channel (w = null -> array)
  let p = 0;
  for (let h = 0; h < z.re[0].length; h++) {
    let sr = 0, si = 0;
    for (let c = 0; c < z.re.length; c++) { sr += w[c] * z.re[c][h]; si += w[c] * z.im[c][h]; }
    p += (HW[h] ?? 0.3) * (sr * sr + si * si);
  }
  return p;
}
const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
// d' with the pooled spread of ON and OFF (signal-plus-noise power varies more than noise alone)
const dprime = (on, off) => (mean(on) - mean(off)) / (Math.sqrt((sd(on) ** 2 + sd(off) ** 2) / 2) || 1e-12);

function stats(segs, w) {
  const on = segs.filter(s => s.on).map(s => segPower(s.z, w)), off = segs.filter(s => !s.on).map(s => segPower(s.z, w));
  if (on.length < 3 || off.length < 3) return { snr: NaN, d1: NaN };
  return { snr: mean(on) / mean(off), d1: dprime(on, off), on: mean(on), off: mean(off) };
}

// Generalised eigenvector: maximise w'Sw / w'Nw.
function gevp(segs, C) {
  const S = new Float64Array(C * C), N = new Float64Array(C * C);
  let nOn = 0, nOff = 0;
  for (const s of segs) {
    const M = s.on ? S : N; s.on ? nOn++ : nOff++;
    for (let h = 0; h < s.z.re[0].length; h++) for (let i = 0; i < C; i++) for (let j = 0; j < C; j++)
      M[i * C + j] += (HW[h] ?? 0.3) * (s.z.re[i][h] * s.z.re[j][h] + s.z.im[i][h] * s.z.im[j][h]);
  }
  for (let i = 0; i < C * C; i++) { S[i] /= nOn || 1; N[i] /= nOff || 1; }
  let tr = 0; for (let i = 0; i < C; i++) tr += N[i * C + i];
  for (let i = 0; i < C; i++) N[i * C + i] += 0.05 * tr / C; // shrinkage against overfitting
  const L = cholesky(N, C);
  const A = forwardSub(L, C, S, C);                       // L^-1 S
  const At = new Float64Array(C * C); for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) At[j * C + i] = A[i * C + j];
  const M = forwardSub(L, C, At, C);                       // L^-1 S L^-T (symmetric)
  let v = new Float64Array(C).fill(1 / Math.sqrt(C));
  for (let it = 0; it < 200; it++) {
    const w = new Float64Array(C);
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) w[i] += M[i * C + j] * v[j];
    const n = Math.hypot(...w) || 1; v = w.map(x => x / n);
  }
  // w = L^-T v  (back substitution)
  const w = new Float64Array(C);
  for (let i = C - 1; i >= 0; i--) { let s = v[i]; for (let k = i + 1; k < C; k++) s -= L[k * C + i] * w[k]; w[i] = s / L[i * C + i]; }
  const big = w.reduce((m, x) => (Math.abs(x) > Math.abs(m) ? x : m), 0) || 1;
  const l1 = w.reduce((s, x) => s + Math.abs(x), 0) || 1;
  return Array.from(w, x => (x / l1) * Math.sign(big));
}

export function analyzeCalibration(eeg, cal, { baseNames = ['TP9', 'TP10'], latency = 0.12, cursor = 100, letterPx = 1000 } = {}) {
  const fs = eeg.fs, names = eeg.channels, C = names.length;
  const cycles = makeCycles({ flicker: cal.flicker }, cal.freq);
  // ---- 1 s lock-in segments in every ON/OFF block (first second skipped: entrainment)
  let segs = [];
  for (const b of cal.blocks) {
    for (let t = b.t0 + 1 + latency; t + 1 <= b.t1 + latency + 1e-6; t += 1) {
      const i0 = indexAt(eeg, t);
      if (i0 + fs > eeg.n) continue;
      const wins = names.map((_, c) => eeg.data[c].subarray(i0, i0 + fs));
      if (wins.some(w => !Number.isFinite(w[0]))) continue;
      segs.push({ on: b.on, cycle: b.cycle, z: lockin(wins, eeg.times.subarray(i0, i0 + fs), cycles, 2), p2p: wins.map(peakToPeak) });
    }
  }
  // drop grossly contaminated segments (robust, per channel: > 3× the median p2p)
  const medP2p = names.map((_, c) => percentile(segs.map(s => s.p2p[c]), 0.5));
  const base = names.map(n => (baseNames.includes(n) ? 1 / baseNames.filter(b => names.includes(b)).length : 0));
  const clean = segs.filter(s => s.p2p.every((v, c) => !(base[c] > 0) || v < 3 * medP2p[c]));
  const used = clean.length >= 12 ? clean : segs;

  // ---- per channel
  const channels = names.map((n, c) => {
    const e = names.map((_, k) => (k === c ? 1 : 0));
    const st = stats(used, e);
    const z1 = s => s.z.re[c][0] ** 2 + s.z.im[c][0] ** 2;
    const amp = Math.sqrt(Math.max(0, mean(used.filter(s => s.on).map(z1)) - mean(used.filter(s => !s.on).map(z1))));
    return { name: n, snr: st.snr, d1: st.d1, ampUV: amp, p2pMedian: medP2p[c] };
  });

  // ---- baseline (power average over base channels) vs spatial filter, 2-fold CV by cycle parity
  const basePower = s => mean(names.map((_, c) => (base[c] > 0 ? segPower(s.z, names.map((_, k) => (k === c ? 1 : 0))) : NaN)).filter(Number.isFinite));
  const baseStats = set => {
    const on = set.filter(s => s.on).map(basePower), off = set.filter(s => !s.on).map(basePower);
    return { snr: mean(on) / mean(off), d1: dprime(on, off) };
  };
  const folds = [0, 1].map(k => ({ train: used.filter(s => s.cycle % 2 !== k), test: used.filter(s => s.cycle % 2 === k) }));
  const cv = folds.map(({ train, test }) => {
    const w = gevp(train, C);
    return { filt: stats(test, w), base: baseStats(test) };
  });
  const cvFilt = { snr: mean(cv.map(f => f.filt.snr)), d1: mean(cv.map(f => f.filt.d1)) };
  const cvBase = { snr: mean(cv.map(f => f.base.snr)), d1: mean(cv.map(f => f.base.d1)) };
  const weights = gevp(used, C);
  // single imaging channel (e.g. one ear has contact, the other does not): pick the best on
  // one fold, score it on the other, so the selection itself is cross-validated
  const imaging = names.map((n, c) => c).filter(c => /TP9|TP10|AUX/i.test(names[c]));
  const unit = c => names.map((_, k) => (k === c ? 1 : 0));
  const cvSingle = folds.map(({ train, test }) => {
    const pick = imaging.reduce((b, c) => (stats(train, unit(c)).d1 > stats(train, unit(b)).d1 ? c : b), imaging[0]);
    return { c: pick, ...stats(test, unit(pick)) };
  });
  const bestSingle = imaging.reduce((b, c) => (stats(used, unit(c)).d1 > stats(used, unit(b)).d1 ? c : b), imaging[0]);
  const cvOne = { snr: mean(cvSingle.map(f => f.snr)), d1: mean(cvSingle.map(f => f.d1)), name: names[bestSingle] };
  const useFilter = cvFilt.d1 > cvBase.d1 * 1.15 && cvFilt.snr > 1 && cvFilt.d1 >= cvOne.d1;
  const useSingle = !useFilter && imaging.length > 1 && cvOne.d1 > cvBase.d1 * 1.15;
  const best = useFilter ? cvFilt : useSingle ? cvOne : cvBase;

  // ---- detectability -> scan plan. Benchmarked on simulated Muse data (noisebench.html):
  // image validity ≈ 0.65 (a clearly readable N) needs d'(1 s)·√(window × passes) ≈ 3.5;
  // ≈ 2.4 gives only ≈ 0.38. Total data per pixel T = (3.5 / d')², split into passes of ≤ 6 s.
  const d1 = best.d1;
  const Tneed = d1 > 0 ? (3.5 / d1) ** 2 : Infinity;
  const verdict = !(d1 > 0) || Tneed > 20 ? 'nogo' : Tneed > 8 ? 'marginal' : 'go';
  const Ttot = Math.min(20, Math.max(2, Tneed));
  const passes = Math.max(1, Math.ceil(Ttot / 6));
  const T = Ttot / passes;
  const speed = Math.max(3, Math.round(cursor / T));
  const rows = 10, lineSec = letterPx / speed + cursor / speed + 1.5;
  const plan = { winSec: +T.toFixed(1), speed, rows, cursor, passes, totalSec: +Ttot.toFixed(1), minutes: rows * (lineSec + 3) / 60 };

  // ---- artifact thresholds
  const f = blockFeatures(eeg);
  const blockOf = t => Math.floor(indexAt(eeg, t) / f.bs);
  const blocksIn = (a, b) => { const out = []; for (let k = blockOf(a); k < Math.min(f.nb, blockOf(b)); k++) out.push(k); return out; };
  const cleanBlocks = [...cal.blocks.flatMap(b => blocksIn(b.t0 + 0.5, b.t1)), ...cal.tasks.filter(t => t.name === 'rest').flatMap(t => blocksIn(t.t0 + 0.5, t.t1))];
  const taskBlocks = name => cal.tasks.filter(t => t.name === name).flatMap(t => blocksIn(t.t0 + 0.5, t.t1));
  const thresholds = {}, artifactReport = [];
  names.forEach((n, c) => {
    const th = {};
    // p2p from head movement only: blinks are large but carry little 15 Hz energy, so they are
    // deliberately not used to set the threshold (masking them would waste data)
    for (const [feat, tasks] of [['p2p', ['move']], ['emg', ['clench', 'swallow']]]) {
      const cleanV = cleanBlocks.map(k => f[feat][c][k]).filter(Number.isFinite);
      const artV = tasks.flatMap(taskBlocks).map(k => f[feat][c][k]).filter(Number.isFinite);
      // clean level from robust statistics (rest data contains spontaneous artifacts too),
      // threshold half-way (geometrically) towards the prompted-artifact level
      const med = percentile(cleanV, 0.5), mad = percentile(cleanV.map(x => Math.abs(x - med)), 0.5) * 1.4826;
      const cl = med + 3 * mad, a75 = artV.length ? percentile(artV, 0.75) : NaN;
      th[feat] = a75 > cl ? Math.sqrt(cl * a75) : cl;
    }
    thresholds[n] = th;
    const flagged = k => !(f.p2p[c][k] <= th.p2p) || !(f.emg[c][k] <= th.emg);
    const rate = ks => (ks.length ? ks.filter(flagged).length / ks.length : NaN);
    artifactReport.push({
      name: n, p2pThr: th.p2p, emgThr: th.emg, cleanFlagged: rate(cleanBlocks),
      caught: Object.fromEntries(['blink', 'clench', 'swallow', 'move'].map(t => [t, rate(taskBlocks(t))])),
    });
  });

  return {
    created: new Date().toISOString(), freq: cal.freq, fs, channels: names,
    segments: { total: segs.length, used: used.length },
    perChannel: channels,
    pipelines: { base: cvBase, filter: cvFilt, single: cvOne, baseNames },
    pipeline: useFilter ? 'filter' : useSingle ? 'single' : 'base', bestChannel: useSingle ? cvOne.name : null,
    filter: { use: useFilter, weights: Object.fromEntries(names.map((n, c) => [n, weights[c]])), gainD1: cvFilt.d1 / cvBase.d1 },
    d1, verdict, plan, thresholds, artifactReport,
  };
}
