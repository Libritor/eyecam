// Signal processing: FFT, PSD (periodogram / Welch), SSVEP metrics from the paper,
// Butterworth low-pass + zero-phase filtering, resampling helpers.

export function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

// In-place iterative radix-2 FFT. re/im are Float64Array of power-of-two length.
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, ang = -2 * Math.PI / len;
    for (let k = 0; k < half; k++) {
      const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
      for (let i = k; i < n; i += len) {
        const b = i + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[i] - xr; im[b] = im[i] - xi;
        re[i] += xr; im[i] += xi;
      }
    }
  }
}

const hannCache = new Map();
function hann(n) {
  let w = hannCache.get(n);
  if (!w) {
    w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1 || 1));
    hannCache.set(n, w);
  }
  return w;
}

// Remove mean and linear trend (returns a new Float64Array).
export function detrend(x) {
  const n = x.length, out = new Float64Array(n);
  if (n < 2) { if (n) out[0] = 0; return out; }
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += i; sy += x[i]; sxx += i * i; sxy += i * x[i]; }
  const den = n * sxx - sx * sx;
  const b = den ? (n * sxy - sx * sy) / den : 0, a = (sy - b * sx) / n;
  for (let i = 0; i < n; i++) out[i] = x[i] - (a + b * i);
  return out;
}

// One-sided PSD (units^2/Hz) of a single segment with a Hann window.
export function periodogram(x, fs, nfft) {
  const n = x.length;
  nfft = Math.max(nextPow2(n), nfft || 0);
  const w = hann(n), d = detrend(x);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  let wss = 0;
  for (let i = 0; i < n; i++) { re[i] = d[i] * w[i]; wss += w[i] * w[i]; }
  fft(re, im);
  const m = nfft / 2 + 1, f = new Float64Array(m), p = new Float64Array(m);
  const scale = 1 / (fs * wss);
  for (let k = 0; k < m; k++) {
    f[k] = k * fs / nfft;
    p[k] = (re[k] * re[k] + im[k] * im[k]) * scale * (k === 0 || k === nfft / 2 ? 1 : 2);
  }
  return { f, p, df: fs / nfft };
}

// Welch PSD: Hann segments of `seg` samples, 50 % overlap. Falls back to a periodogram
// when the data is shorter than one segment.
export function welch(x, fs, seg = 512, nfft) {
  const n = x.length;
  if (n <= seg) return periodogram(x, fs, nfft);
  const hop = Math.floor(seg / 2);
  const count = Math.floor((n - seg) / hop) + 1;
  let acc = null;
  for (let s = 0; s < count; s++) {
    const r = periodogram(x.subarray ? x.subarray(s * hop, s * hop + seg) : x.slice(s * hop, s * hop + seg), fs, nfft);
    if (!acc) acc = r; else for (let k = 0; k < r.p.length; k++) acc.p[k] += r.p[k];
  }
  for (let k = 0; k < acc.p.length; k++) acc.p[k] /= count;
  return acc;
}

// PSD averaged across channels (each channel is an array of samples).
export function spectrum(chans, fs, opt = {}) {
  const method = opt.psd || 'welch';
  const seg = Math.round((opt.segSec || 2) * fs);
  let acc = null;
  for (const x of chans) {
    const r = method === 'periodogram' ? periodogram(x, fs, opt.nfft) : welch(x, fs, seg, opt.nfft);
    if (!acc) acc = { f: r.f, p: Float64Array.from(r.p), df: r.df };
    else for (let k = 0; k < r.p.length; k++) acc.p[k] += r.p[k];
  }
  if (acc) for (let k = 0; k < acc.p.length; k++) acc.p[k] /= chans.length;
  return acc;
}

export function bandPower(spec, lo, hi) {
  let s = 0;
  for (let k = 0; k < spec.f.length; k++) if (spec.f[k] >= lo && spec.f[k] <= hi) s += spec.p[k];
  return s * spec.df;
}

// Power in a narrow band around f0 (always includes the nearest bin).
export function peakPower(spec, f0, bw = 0.25) {
  const half = Math.max(bw, spec.df / 2);
  let s = 0;
  for (let k = 0; k < spec.f.length; k++) if (Math.abs(spec.f[k] - f0) <= half) s += spec.p[k];
  if (s === 0) { const k = Math.round(f0 / spec.df); if (k < spec.p.length) s = spec.p[k]; }
  return s * spec.df;
}

export const METRICS = {
  harmonic: 'Paper §III: (P(f)+P(2f)) / P(band 14–50 Hz)',
  ratio: 'Paper §II: P(f) / P(rest of 1–50 Hz)',
  snr: 'SNR: P(f,2f) / neighbouring bins (±1–2 Hz)',
  power: 'Absolute power at f (µV²)',
  fbcca: 'FBCCA: filter-bank CCA, all analysis channels',
  lockin: 'Lock-in power vs. the logged flicker phase',
  coherent: 'Coherent lock-in (phase-tracked; best for short windows)',
  kalman: 'Kalman-tracked amplitude (RTS-smoothed; soft artifact weighting)',
};

// SSVEP magnitude from a spectrum, per the paper's definitions.
//   harmonic: (P(f) + P(2f) [+...]) / P(band)   band defaults to [f-1, 50] (14–50 for 15 Hz)
//   ratio:    P(f) / (P(1..50) - P(f))
export function ssvepMetric(spec, f, opt = {}) {
  const metric = opt.metric || 'harmonic';
  const bw = opt.bw ?? 0.25;
  const nyq = spec.f[spec.f.length - 1];
  const top = Math.min(opt.bandHi ?? 50, nyq);
  const harmonics = [];
  for (let h = 1; h <= (opt.harmonics ?? 2); h++) if (h * f + bw < nyq) harmonics.push(h * f);
  if (metric === 'power') return harmonics.reduce((s, hf) => s + peakPower(spec, hf, bw), 0);
  if (metric === 'ratio') {
    const pf = peakPower(spec, f, bw);
    const rest = bandPower(spec, 1, top) - pf;
    return rest > 0 ? pf / rest : 0;
  }
  if (metric === 'snr') {
    let sig = 0, noise = 0;
    for (const hf of harmonics) {
      sig += peakPower(spec, hf, bw);
      const nb = (bandPower(spec, hf - 2, hf - 1) + bandPower(spec, hf + 1, hf + 2)) / 2;
      noise += nb * (2 * Math.max(bw, spec.df / 2)) / 1; // same bandwidth as the signal estimate
    }
    return noise > 0 ? sig / noise : 0;
  }
  const lo = opt.bandLo ?? (f - 1);
  let sig = 0;
  for (const hf of harmonics) if (hf <= top) sig += peakPower(spec, hf, bw);
  const band = bandPower(spec, lo, top);
  return band > 0 ? sig / band : 0;
}

// Peak-to-peak of a detrended window; used to reject blinks / movement artifacts.
export function peakToPeak(x) {
  const d = detrend(x);
  let lo = Infinity, hi = -Infinity;
  for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return hi - lo;
}

export function std(x) {
  let m = 0; for (const v of x) m += v; m /= x.length || 1;
  let s = 0; for (const v of x) s += (v - m) * (v - m);
  return Math.sqrt(s / (x.length || 1));
}

export function percentile(values, q) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const i = (v.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return v[lo] + (v[hi] - v[lo]) * (i - lo);
}

// ---------------------------------------------------------------- Butterworth
// Low-pass Butterworth of even/odd order as second-order sections (bilinear transform,
// pre-warped). Returns [{b0,b1,b2,a1,a2}, ...] plus a first-order section for odd orders.
export function butterLowpass(order, fc, fs) {
  const K = 2 * fs, wc = K * Math.tan(Math.PI * fc / fs);
  const sos = [];
  for (let k = 0; k < Math.floor(order / 2); k++) {
    const theta = Math.PI * (2 * k + 1 + order) / (2 * order);   // pole angle, left half plane
    const A = -2 * wc * Math.cos(theta), B = wc * wc;            // s^2 + A s + B
    const a0 = K * K + A * K + B;
    sos.push({ b0: B / a0, b1: 2 * B / a0, b2: B / a0, a1: (2 * B - 2 * K * K) / a0, a2: (K * K - A * K + B) / a0 });
  }
  if (order % 2) { // (wc)/(s + wc)
    const a0 = K + wc;
    sos.push({ b0: wc / a0, b1: wc / a0, b2: 0, a1: (wc - K) / a0, a2: 0 });
  }
  return sos;
}

// High-pass Butterworth (bilinear, pre-warped) as second-order sections; even orders only.
export function butterHighpass(order, fc, fs) {
  const K = 2 * fs, wc = K * Math.tan(Math.PI * fc / fs);
  const sos = [];
  for (let k = 0; k < Math.floor(order / 2); k++) {
    const theta = Math.PI * (2 * k + 1 + order) / (2 * order);
    const A = -2 * wc * Math.cos(theta), B = wc * wc;          // s^2 / (s^2 + A s + B)
    const a0 = K * K + A * K + B;
    sos.push({ b0: K * K / a0, b1: -2 * K * K / a0, b2: K * K / a0, a1: (2 * B - 2 * K * K) / a0, a2: (K * K - A * K + B) / a0 });
  }
  return sos;
}

// Mains notch (RBJ biquad).
export function notch(f0, fs, Q = 30) {
  const w0 = 2 * Math.PI * f0 / fs, al = Math.sin(w0) / (2 * Q), c = Math.cos(w0), a0 = 1 + al;
  return { b0: 1 / a0, b1: -2 * c / a0, b2: 1 / a0, a1: -2 * c / a0, a2: (1 - al) / a0 };
}

// Streaming SOS filter with persistent state (for incremental filtering of a live buffer).
export function sosState(sos) { return sos.map(() => [0, 0]); }
export function sosRun(sos, st, x, out, from, to) {
  for (let i = from; i < to; i++) {
    let v = x[i];
    if (!Number.isFinite(v)) v = 0;
    for (let k = 0; k < sos.length; k++) {
      const s = sos[k], z = st[k], y = s.b0 * v + z[0];
      z[0] = s.b1 * v - s.a1 * y + z[1];
      z[1] = s.b2 * v - s.a2 * y;
      v = y;
    }
    out[i] = v;
  }
}

export function sosfilt(sos, x) {
  let y = Float64Array.from(x);
  for (const s of sos) {
    let z1 = 0, z2 = 0;
    // steady-state initial conditions for a step at y[0] (like scipy's lfilter_zi)
    const x0 = y[0], g = (s.b0 + s.b1 + s.b2) / (1 + s.a1 + s.a2), yss = g * x0;
    z1 = yss - s.b0 * x0; z2 = s.b2 * x0 - s.a2 * yss;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i], yi = s.b0 * xi + z1;
      z1 = s.b1 * xi - s.a1 * yi + z2;
      z2 = s.b2 * xi - s.a2 * yi;
      y[i] = yi;
    }
  }
  return y;
}

// Zero-phase forward/backward filtering with odd reflection padding.
export function filtfilt(sos, x) {
  const n = x.length;
  if (n < 4) return Float64Array.from(x);
  const pad = Math.min(n - 1, 3 * 2 * sos.length * 3);
  const ext = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) {
    ext[i] = 2 * x[0] - x[pad - i];
    ext[n + pad + i] = 2 * x[n - 1] - x[n - 2 - i];
  }
  for (let i = 0; i < n; i++) ext[pad + i] = x[i];
  let y = sosfilt(sos, ext);
  y.reverse(); y = sosfilt(sos, y); y.reverse();
  return y.slice(pad, pad + n);
}

// Linear resampling of irregular (t, v) onto a uniform grid starting at t[0].
export function resampleUniform(t, v, fs) {
  const n = t.length;
  if (n < 2) return { t0: t[0] || 0, fs, v: Float64Array.from(v) };
  const t0 = t[0], m = Math.max(2, Math.floor((t[n - 1] - t0) * fs) + 1);
  const out = new Float64Array(m);
  let j = 0;
  for (let i = 0; i < m; i++) {
    const ti = t0 + i / fs;
    while (j < n - 2 && t[j + 1] < ti) j++;
    const dt = t[j + 1] - t[j];
    const a = dt > 0 ? Math.min(1, Math.max(0, (ti - t[j]) / dt)) : 0;
    out[i] = v[j] + (v[j + 1] - v[j]) * a;
  }
  return { t0, fs, v: out };
}
