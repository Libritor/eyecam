// Growing multi-channel EEG buffer on the page clock (seconds, performance.now()/1000).
// The same {fs, channels, times, data, n} shape is used for saved session segments, so
// every reconstruction works identically live and offline.

export const perfNow = () => performance.now() / 1000;
// epoch seconds -> page clock seconds
export const epochToPerf = s => s - performance.timeOrigin / 1000;

export class EEGStore {
  constructor(fs = 256, channels = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']) {
    this.reset(fs, channels);
    this.listeners = new Set();
  }

  reset(fs, channels) {
    this.fs = fs;
    this.channels = channels.slice();
    this.cap = 1 << 16;
    this.n = 0;
    this.times = new Float64Array(this.cap);
    this.data = channels.map(() => new Float32Array(this.cap));
    this.received = 0;
  }

  _grow(need) {
    if (need <= this.cap) return;
    while (this.cap < need) this.cap *= 2;
    const t = new Float64Array(this.cap); t.set(this.times.subarray(0, this.n)); this.times = t;
    this.data = this.data.map(d => { const a = new Float32Array(this.cap); a.set(d.subarray(0, this.n)); return a; });
  }

  // times: array of page-clock seconds; rows: array of per-sample channel arrays
  push(times, rows) {
    const m = times.length;
    if (!m) return;
    this._grow(this.n + m);
    for (let i = 0; i < m; i++) {
      const t = times[i];
      if (this.n && t <= this.times[this.n - 1]) continue; // keep strictly increasing
      this.times[this.n] = t;
      const r = rows[i];
      for (let c = 0; c < this.data.length; c++) this.data[c][this.n] = r[c] ?? NaN;
      this.n++;
    }
    this.received += m;
    for (const fn of this.listeners) fn(this);
  }

  onData(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  get lastTime() { return this.n ? this.times[this.n - 1] : -Infinity; }

  // Copy of the samples in [t0, t1] as a standalone segment (for saving sessions).
  segment(t0, t1) { return segmentOf(this, t0, t1); }
}

// First index with times[i] >= t
export function indexAt(eeg, t) {
  let lo = 0, hi = eeg.n;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (eeg.times[mid] < t) lo = mid + 1; else hi = mid; }
  return lo;
}

// Window of `len` samples centred on time t for the given channel indices.
// Returns null if the data is not (yet) available or has a gap > 0.25 s.
export function windowAt(eeg, tCenter, len, chans) {
  const i0 = indexAt(eeg, tCenter) - (len >> 1);
  if (i0 < 0 || i0 + len > eeg.n) return null;
  const span = eeg.times[i0 + len - 1] - eeg.times[i0];
  if (span > (len - 1) / eeg.fs + 0.25) return null;
  const out = [];
  for (const c of chans) {
    const x = eeg.data[c].subarray(i0, i0 + len);
    if (!Number.isFinite(x[0]) || !Number.isFinite(x[len - 1])) return null;
    out.push(x);
  }
  return out;
}

// Window ending at time t
export function windowEnding(eeg, tEnd, len, chans) {
  return windowAt(eeg, tEnd - len / eeg.fs / 2, len, chans);
}

export function segmentOf(eeg, t0, t1) {
  const i0 = Math.max(0, indexAt(eeg, t0)), i1 = Math.min(eeg.n, indexAt(eeg, t1));
  return {
    fs: eeg.fs, channels: eeg.channels.slice(), n: Math.max(0, i1 - i0),
    times: eeg.times.slice(i0, i1), data: eeg.data.map(d => d.slice(i0, i1)),
  };
}

// Which channel indices to analyse, from a selection like ['TP9','TP10'] or 'auto'.
export function resolveChannels(eeg, sel) {
  const names = eeg.channels;
  if (Array.isArray(sel) && sel.length) {
    const idx = sel.map(s => names.indexOf(s)).filter(i => i >= 0);
    if (idx.length) return idx;
  }
  // auto: TP9 + TP10. The AUX (Oz) channel is used only when calibration shows it carries
  // the best SSVEP (it then sets Channels = AUX) or when listed explicitly: noise level alone
  // cannot tell an Oz electrode from an unconnected input that merely looks like EEG.
  const tp = ['TP9', 'TP10'].map(n => names.indexOf(n)).filter(i => i >= 0);
  return tp.length ? tp : [0];
}

// ---------------------------------------------------------------- (de)serialisation
function b64(buf) {
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function unb64(s, Type) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}
export function encodeSegment(seg) {
  return { fs: seg.fs, channels: seg.channels, n: seg.n, times: b64(seg.times), data: seg.data.map(b64) };
}
export function decodeSegment(o) {
  return { fs: o.fs, channels: o.channels, n: o.n, times: unb64(o.times, Float64Array), data: o.data.map(d => unb64(d, Float32Array)) };
}
export function segmentToCSV(seg) {
  const lines = ['time_s,' + seg.channels.join(',')];
  for (let i = 0; i < seg.n; i++) lines.push(seg.times[i].toFixed(5) + ',' + seg.data.map(d => d[i].toFixed(3)).join(','));
  return lines.join('\n');
}
