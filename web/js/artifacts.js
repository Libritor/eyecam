// Sample-level artifact masking.
//
// Whole-window rejection wastes most of the data on a Muse (a blink every few seconds, jaw
// EMG under TP9/TP10, electrode shifts), so instead the recording is cut into 0.25 s blocks
// and each block gets two features per channel:
//   p2p  peak-to-peak of the detrended block   -> electrode shifts, head movement, blinks
//   emg  mean power above 70 Hz (120 Hz notched) -> muscle (jaw clench, swallowing, talking)
// A block is masked if any analysis channel exceeds its threshold (dilated by one block),
// and the SSVEP estimators then ignore those samples instead of the whole window.
//
// Thresholds come either from a calibration (artifact prompts vs. clean rest; see calib.js)
// or, with no calibration, from the recording itself: median + k·MAD per channel/feature.

import { butterHighpass, notch, sosState, sosRun, percentile } from './dsp.js';

export const BLOCK_SEC = 0.25;
const cache = new WeakMap();

// Per-block features, computed incrementally as a live store grows (cached per eeg object).
export function blockFeatures(eeg) {
  const bs = Math.round(BLOCK_SEC * eeg.fs);
  const key = `${eeg.fs}|${eeg.channels.length}`;
  let c = cache.get(eeg);
  if (!c || c.key !== key || c.nb * bs > eeg.n || c.t0 !== eeg.times[0]) {
    const sos = butterHighpass(4, Math.min(70, eeg.fs * 0.3), eeg.fs);
    if (120 < eeg.fs / 2 - 2) sos.push(notch(120, eeg.fs, 8));
    c = {
      key, t0: eeg.times[0], bs, nb: 0, sos,
      st: eeg.channels.map(() => sosState(sos)), scratch: new Float32Array(bs),
      p2p: eeg.channels.map(() => []), emg: eeg.channels.map(() => []), masks: new Map(),
    };
    cache.set(eeg, c);
  }
  while ((c.nb + 1) * bs <= eeg.n) {
    const a = c.nb * bs;
    for (let ch = 0; ch < eeg.channels.length; ch++) {
      const x = eeg.data[ch].subarray(a, a + bs);
      // detrended peak-to-peak (NaN-safe)
      let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
      for (let k = 0; k < bs; k++) if (Number.isFinite(x[k])) { sx += k; sy += x[k]; sxx += k * k; sxy += k * x[k]; n++; }
      let p2p = NaN;
      if (n > 4) {
        const den = n * sxx - sx * sx, b = den ? (n * sxy - sx * sy) / den : 0, a0 = (sy - b * sx) / n;
        let lo = Infinity, hi = -Infinity;
        for (let k = 0; k < bs; k++) if (Number.isFinite(x[k])) { const r = x[k] - a0 - b * k; if (r < lo) lo = r; if (r > hi) hi = r; }
        p2p = hi - lo;
      }
      sosRun(c.sos, c.st[ch], x, c.scratch, 0, bs);
      let e = 0; for (let k = 0; k < bs; k++) e += c.scratch[k] * c.scratch[k];
      c.p2p[ch].push(p2p); c.emg[ch].push(e / bs);
    }
    c.nb++;
  }
  return c;
}

// Robust per-channel thresholds from the data itself.
export function autoThresholds(f, ch, k) {
  const t = {};
  for (const feat of ['p2p', 'emg']) {
    const v = f[feat][ch].filter(Number.isFinite);
    const med = percentile(v, 0.5), mad = percentile(v.map(x => Math.abs(x - med)), 0.5) * 1.4826;
    t[feat] = med + k * (mad || med * 0.5 || 1);
  }
  return t;
}

// Mask (1 = bad block) for the given analysis channels. p.mask = {mode:'off'|'auto'|'calib', k, thr:{chName:{p2p,emg}}}
export function maskInfo(eeg, p) {
  const m = p.mask;
  if (!m || m.mode === 'off' || !m.mode) return null;
  const f = blockFeatures(eeg);
  const chans = p.maskChans || p.chans || [];
  const key = `${m.mode}|${m.k}|${chans.join(',')}|${m.mode === 'calib' ? JSON.stringify(m.thr) : ''}`;
  let mi = f.masks.get(key);
  // auto thresholds are re-estimated as data accumulates (cheap); masks extend incrementally
  if (!mi || (m.mode === 'auto' && f.nb > mi.thrAt * 1.25 + 40)) {
    const thr = chans.map(ch => {
      const name = eeg.channels[ch];
      if (m.mode === 'calib' && m.thr?.[name]) return m.thr[name];
      return autoThresholds(f, ch, m.k ?? 5);
    });
    mi = { thr, thrAt: f.nb, raw: new Uint8Array(0), bad: new Uint8Array(0), nb: 0, bs: f.bs };
    f.masks.set(key, mi);
  }
  if (mi.nb < f.nb) {
    const raw = new Uint8Array(f.nb); raw.set(mi.raw);
    for (let b = mi.nb; b < f.nb; b++) {
      let bad = 0;
      chans.forEach((ch, i) => {
        const p2p = f.p2p[ch][b], emg = f.emg[ch][b];
        if (!(p2p <= mi.thr[i].p2p) || !(emg <= mi.thr[i].emg)) bad = 1; // NaN counts as bad
      });
      raw[b] = bad;
    }
    const dil = new Uint8Array(f.nb);
    for (let b = 0; b < f.nb; b++) dil[b] = raw[b] | (b > 0 ? raw[b - 1] : 0) | (b + 1 < f.nb ? raw[b + 1] : 0);
    mi.raw = raw; mi.bad = dil; mi.nb = f.nb;
  }
  return mi;
}

// Per-sample weights (0/1) for a window starting at sample i0, combining the artifact mask
// and an optional split-half selector (p.half = 0|1 keeps alternate 0.5 s chunks). Returns
// null when every sample counts.
export function windowWeights(eeg, i0, len, p) {
  const mi = maskInfo(eeg, p);
  const half = p.half;
  if (!mi && half == null) return null;
  const bs = mi ? mi.bs : Math.round(BLOCK_SEC * eeg.fs);
  const w = new Float32Array(len);
  for (let k = 0; k < len; k++) {
    const b = Math.floor((i0 + k) / bs);
    let v = 1;
    if (mi && b < mi.nb && mi.bad[b]) v = 0;
    if (half != null && (Math.floor(b / 2) & 1) !== half) v = 0;
    w[k] = v;
  }
  return w;
}

// Fraction of masked time over [t0, t1] (for reporting).
export function maskedFraction(eeg, p) {
  const mi = maskInfo(eeg, p);
  if (!mi || !mi.nb) return 0;
  let s = 0; for (let b = 0; b < mi.nb; b++) s += mi.bad[b];
  return s / mi.nb;
}

// Shared electrode-contact check (Signal monitor + Calibration gate use the same numbers).
// p2p: median over the last 5 s of 1 s peak-to-peak (max of 4 consecutive block values).
// emg: median power above 70 Hz — high means jaw/temple muscle or a poorly contacting
// electrode picking up interference.
export const EMG_WARN = 40, EMG_BAD = 100;
export function contactQuality(eeg, gate = 100) {
  if (eeg.n < eeg.fs * 2) return null;
  const f = blockFeatures(eeg), nb = f.nb, take = Math.min(nb, 20);
  return eeg.channels.map((name, c) => {
    const p2p = f.p2p[c].slice(nb - take, nb), emgs = f.emg[c].slice(nb - take, nb);
    const one = [];
    for (let i = 0; i + 4 <= p2p.length; i += 4) one.push(Math.max(...p2p.slice(i, i + 4)));
    const m = percentile(one, 0.5), emg = percentile(emgs, 0.5);
    const imaging = /TP9|TP10|AUX/i.test(name);
    let cls = !(m > 2) ? 'bad' : m <= gate * 0.6 ? 'ok' : m <= gate ? 'warn' : 'bad';
    if (emg > EMG_BAD) cls = 'bad'; else if (emg > EMG_WARN && cls === 'ok') cls = 'warn';
    const hint = !(m > 2) ? 'flat — no contact'
      : emg > EMG_WARN ? 'muscle/interference — relax jaw, re-wet, check pad'
      : m > gate ? 'large swings — re-seat / wet, keep still'
      : cls === 'ok' ? 'good' : 'fair — let it settle';
    return { name, p2p: m, emg, cls, imaging, hint };
  });
}
