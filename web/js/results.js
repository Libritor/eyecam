// Result views for every experiment. The same view is shown right after a live run and in
// Review for a loaded session file; parameters can be changed and the image re-rendered.

import { h, form, lineChart, downloadCanvas, download, toast } from './ui.js';
import { app, saveSession, advancedParams } from './context.js';
import { maskedFraction } from './artifacts.js';
import { resolveChannels, segmentToCSV } from './eeg-store.js';
import { makeCycles } from './ssvep.js';
import { METRICS, percentile } from './dsp.js';
import { CMAPS, toImageData, colorOf, splat, normalize } from './interp.js';
import {
  renderScan, gazeSamples, renderGaze, renderChunks, timeline, detectEvents, scoreEvents,
  pathSamples, profile, fwhm, metricAt, DEFAULTS, reliability, validity, scanGrid, autoTune,
} from './recon.js';

const metricOpts = Object.entries(METRICS).map(([k, v]) => [k, v]);
const common = [
  { type: 'section', label: 'Analysis' },
  { key: 'chansSel', label: 'Channels', type: 'text', help: "'auto' = TP9+TP10; calibration may set the best single channel (e.g. AUX for an Oz electrode); or list e.g. TP10,AUX" },
  { key: 'metric', label: 'SSVEP metric', type: 'select', options: metricOpts },
  { key: 'winSec', label: 'Window', type: 'number', min: 0.25, step: 0.05, unit: 's' },
  { key: 'kalmanOrder', label: 'Kalman model', type: 'select', options: [['2', 'Smooth trend (order 2, sharper)'], ['1', 'Random walk (order 1)']], show: x => x.metric === 'kalman' },
  { key: 'kalmanTau', label: 'Kalman smoothing', type: 'number', min: 0, step: 0.1, unit: 's (0 = from window)', show: x => x.metric === 'kalman', help: 'Time constant of the forward–backward smoother. Larger = less noise, softer edges.' },
  { key: 'latency', label: 'Latency comp.', type: 'number', step: 0.01, unit: 's' },
  { key: 'psd', label: 'PSD', type: 'select', options: [['welch', 'Welch (2 s segments)'], ['periodogram', 'Periodogram']] },
  { key: 'maskMode', label: 'Artifact masking', type: 'select', options: [['off', 'Off (whole-window reject)'], ['auto', 'Auto (from this recording)'], ['calib', 'Calibrated thresholds']], help: 'Ignore only the contaminated 0.25 s blocks (blinks, jaw EMG, electrode shifts).' },
  { key: 'maskK', label: 'Auto threshold', type: 'number', min: 1, step: 0.5, unit: '× MAD', show: x => x.maskMode === 'auto' },
  { key: 'minValid', label: 'Min clean data', type: 'number', min: 0.05, max: 1, step: 0.05, unit: 'of window', show: x => x.maskMode !== 'off' },
  { key: 'spatial', label: 'Spatial filter', type: 'select', options: [['off', 'Off (average channels)'], ['calib', 'Calibrated (cross-validated)']] },
  { key: 'rejectUV', label: 'Whole-window reject', type: 'number', min: 0, unit: 'µVpp (mask off)', help: '0 = off', show: x => x.maskMode === 'off' },
];

function baseParams(session, extra) {
  const s = session.settings || app.settings;
  return {
    chansSel: s.chans ?? 'auto', metric: s.metric ?? DEFAULTS.metric, psd: s.psd ?? 'welch', latency: s.latency ?? DEFAULTS.latency,
    rejectUV: s.rejectUV ?? DEFAULTS.rejectUV, bw: s.bw ?? DEFAULTS.bw, harmonics: s.harmonics ?? 2, freq: session.run.freq,
    maskMode: s.mask ?? 'off', maskK: s.maskK ?? 5, minValid: s.minValid ?? 0.4, spatial: s.spatial ?? 'off',
    kalmanOrder: '2', kalmanTau: 0, kalmanProject: false, ...extra,
  };
}
function finalize(session, v) {
  const sel = v.chansSel === 'auto' ? null : String(v.chansSel).split(',').map(x => x.trim()).filter(Boolean);
  const adv = advancedParams({ mask: v.maskMode, maskK: v.maskK, minValid: v.minValid, spatial: v.spatial }, session.calib, session.eeg);
  return { ...v, mask: undefined, spatialW: undefined, spatialChans: undefined, ...adv, chans: resolveChannels(session.eeg, sel), segSec: Math.min(2, v.winSec || 2), cycles: makeCycles(session.run, session.run.freq), kalmanOrder: +(v.kalmanOrder ?? 2) };
}

function imgCanvas(res, cmap, cls = 'out') {
  const c = h('canvas', { class: cls }); c.width = res.w; c.height = res.h;
  c.getContext('2d').putImageData(toImageData(res.img, res.w, res.h, cmap), 0, 0);
  return c;
}

// Fig. 6-style axes around an image (pixel coordinates of the stimulus).
function withAxes(canvas, W, H) {
  const m = { l: 46, b: 26, t: 8, r: 8 };
  const out = h('canvas', { class: 'out' });
  out.width = canvas.width + m.l + m.r; out.height = canvas.height + m.t + m.b;
  const c = out.getContext('2d');
  c.fillStyle = '#fff'; c.fillRect(0, 0, out.width, out.height);
  c.drawImage(canvas, m.l, m.t);
  c.strokeStyle = '#000'; c.strokeRect(m.l, m.t, canvas.width, canvas.height);
  c.fillStyle = '#000'; c.font = '11px sans-serif';
  const step = v => { const e = Math.pow(10, Math.floor(Math.log10(v / 5))); return [1, 2, 2.5, 5, 10].map(k => k * e).find(k => v / k <= 8); };
  const sx = step(W), sy = step(H);
  for (let x = 0; x <= W; x += sx) { const px = m.l + x / W * canvas.width; c.fillText(String(x), px - 8, out.height - 8); c.fillRect(px, m.t + canvas.height, 1, 4); }
  for (let y = 0; y <= H; y += sy) { const py = m.t + y / H * canvas.height; c.fillText(String(y), 4, py + 4); c.fillRect(m.l - 4, py, 4, 1); }
  return out;
}

function lightPainting(samples, W, H, key = ['x', 'y'], cmap = 'bluered', { dot = 6, bg = '#000', scale = 1 } = {}) {
  const c = h('canvas', { class: 'out' }); c.width = Math.round(W * scale); c.height = Math.round(H * scale);
  const x = c.getContext('2d'); x.fillStyle = bg; x.fillRect(0, 0, c.width, c.height);
  samples = samples.filter(s => Number.isFinite(s.v));
  const vals = samples.map(s => s.v), lo = percentile(vals, 0.05), hi = percentile(vals, 0.99);
  x.globalCompositeOperation = 'lighter';
  for (const s of samples) {
    const t = Math.min(1, Math.max(0, (s.v - lo) / (hi - lo || 1)));
    const [r, g, b] = colorOf(cmap, t);
    const px = s[key[0]] * scale, py = s[key[1]] * scale, rad = dot * scale;
    const gr = x.createRadialGradient(px, py, 0, px, py, rad);
    gr.addColorStop(0, `rgba(${r},${g},${b},0.9)`); gr.addColorStop(1, `rgba(${r},${g},${b},0)`);
    x.fillStyle = gr; x.fillRect(px - rad, py - rad, rad * 2, rad * 2);
  }
  x.globalCompositeOperation = 'source-over';
  return c;
}

// ---------------------------------------------------------------- per-mode renderers
const VIEWS = {
  // Fig. 1 / 8 (cursor raster) and Fig. 11 (real-world rig)
  scan: {
    schema: [...common,
      { type: 'section', label: 'Image' },
      { key: 'stepPx', label: 'Column step', type: 'number', min: 1, unit: 'px' },
      { key: 'combine', label: 'Eq. (1) row combine', type: 'checkbox' },
      { key: 'phaseSpan', label: 'Phase tracking span', type: 'number', min: 1, unit: 's (coherent)', show: x => x.metric === 'coherent' || x.deconv || x.kalmanProject },
      { key: 'kalmanProject', label: 'Kalman: project on response phase', type: 'checkbox', show: x => x.metric === 'kalman' },
      { key: 'deconv', label: 'Deconvolve cursor blur', type: 'checkbox', help: 'Least-squares inversion of the cursor/window blur; helps only at high SNR.' },
      { key: 'deconvLambda', label: 'Deconv smoothness', type: 'number', min: 0, step: 0.1, show: x => x.deconv },
      { key: 'rowBaseline', label: 'Remove per-line baseline', type: 'checkbox' },
      { key: 'passCombine', label: 'Combine passes', type: 'select', options: [['median', 'Median (robust)'], ['mean', 'Mean']], show: () => true },
      { key: 'lineRef', label: 'Normalise by line reference patches', type: 'checkbox' },
      { key: 'reject', label: 'Drop artifact windows', type: 'checkbox' },
      { key: 'interp', label: 'Interpolation', type: 'select', options: ['bilinear', 'bicubic', 'nearest'] },
      { key: 'gain', label: 'Gain (paper: ×2 for faces)', type: 'number', min: 0.1, step: 0.1 },
      { key: 'gamma', label: 'Gamma', type: 'number', min: 0.2, step: 0.1 },
      { key: 'cmap', label: 'Colormap', type: 'select', options: CMAPS },
      { key: 'outW', label: 'Output width', type: 'number', min: 32, unit: 'px' },
    ],
    defaults: s => baseParams(s, {
      winSec: s.run.winSec, stepPx: Math.max(1, Math.round(s.run.cursor / 4)), combine: true, reject: true, phaseSpan: 8, deconv: false, deconvLambda: 0.5, rowBaseline: false,
      passCombine: 'median', lineRef: !!s.run.lineRef,
      interp: 'bilinear', gain: s.run.gain ?? 1, gamma: 1, cmap: 'gray', outW: 640,
    }),
    draw(out, s, p) {
      const r = renderScan(s.eeg, s.run, p);
      const lines = new Set(s.run.lines.map(l => l.row)).size;
      out.append(h('div', { class: 'pair' },
        s.reference ? h('figure', {}, h('img', { src: s.reference, class: 'out' }), h('figcaption', {}, 'Input viewed by subject')) : null,
        h('figure', {}, imgCanvas(r, p.cmap), h('figcaption', {}, 'Output from EEG (brainwaves)'))));
      const bad = r.grid.bad.reduce((a, b) => a + b, 0);
      const passes = new Set(s.run.lines.map(l => l.pass ?? 0)).size;
      let q = '';
      if (r.grid.cols * r.grid.rows <= 2500) { // quality scores (cheap enough to run every render)
        const rel = reliability(s.eeg, s.run, p);
        q += ` · split-half reliability ${fmt(rel.r)} (full ≈ ${fmt(rel.full)}, ${rel.method})`;
        if (s.run.truth) q += ` · validity vs. stimulus ${fmt(validity(s.eeg, s.run, p, r.grid))}`;
      }
      const masked = p.mask ? ` · ${(maskedFraction(s.eeg, { ...p, maskChans: p.chans }) * 100).toFixed(0)} % of time masked` : '';
      const chans = p.spatialW ? 'spatial filter (' + p.spatialChans.map((c, j) => `${s.eeg.channels[c]} ${p.spatialW[j].toFixed(2)}`).join(', ') + ')' : p.chans.map(i => s.eeg.channels[i]).join('+');
      return `${lines}/${s.run.rows} scan lines × ${passes} pass${passes > 1 ? 'es' : ''} · grid ${r.grid.cols}×${r.grid.rows} · ${bad} dropped windows${masked} · ${chans}${q}`;
    },
  },

  // Fig. 6: free viewing with eye position (tracker, mouse or guided cursor)
  gaze: {
    schema: [...common,
      { type: 'section', label: 'Eye data' },
      { key: 'hop', label: 'Sample every', type: 'number', min: 0.05, step: 0.05, unit: 's' },
      { key: 'lowpass', label: '1.5 Hz 6th-order Butterworth on X/Y', type: 'checkbox' },
      { key: 'lowpassHz', label: 'Low-pass cutoff', type: 'number', min: 0.1, step: 0.1, unit: 'Hz' },
      { key: 'speedWeight', label: 'Down-weight fast eye movements', type: 'checkbox' },
      { key: 'reject', label: 'Drop blink/artifact windows', type: 'checkbox' },
      { key: 'method', label: 'Interpolation', type: 'select', options: [['triangulate', 'Delaunay (paper)'], ['splat', 'Gaussian splat']] },
      { key: 'sigmaPx', label: 'Splat radius', type: 'number', min: 2, unit: 'px', show: v => v.method === 'splat' },
      { key: 'cmap', label: 'Colormap', type: 'select', options: CMAPS },
    ],
    defaults: s => baseParams(s, { winSec: s.run.winSec, hop: 0.25, lowpass: true, lowpassHz: 1.5, speedWeight: false, reject: true, method: 'triangulate', sigmaPx: 60, cmap: 'viridis', outW: 640 }),
    draw(out, s, p) {
      const samples = gazeSamples(s.eeg, s.run, p);
      const r = renderGaze(samples, s.run, p);
      out.append(h('div', { class: 'pair' },
        s.reference ? h('figure', {}, h('img', { src: s.reference, class: 'out' }), h('figcaption', {}, 'Flickering stimulus')) : null,
        h('figure', {}, withAxes(imgCanvas(r, p.cmap), s.run.W, s.run.H), h('figcaption', {}, 'Reconstructed from mind\'s eye'))));
      const g = h('canvas', { class: 'chart' }); out.append(g);
      requestAnimationFrame(() => lineChart(g, {
        series: [{ x: samples.map(q => q.t - samples[0]?.t), y: samples.map(q => q.v), color: '#3b82f6', label: 'SSVEP' }],
        xlabel: 'time (s)', ylabel: 'SSVEP',
      }));
      return `${samples.length} gaze samples (${samples.filter(q => q.w > 0.2).length} used) · gaze source: ${s.run.gazeSource}`;
    },
  },

  // Fig. 12: shutter glasses, fixate chunk by chunk
  chunks: {
    schema: [...common.filter(f => f.key !== 'winSec'),
      { type: 'section', label: 'Image' },
      { key: 'settle', label: 'Ignore first', type: 'number', min: 0, step: 0.5, unit: 's of each fixation' },
      { key: 'gain', label: 'Gain', type: 'number', min: 0.1, step: 0.1 },
      { key: 'cmap', label: 'Colormap', type: 'select', options: CMAPS },
    ],
    defaults: s => baseParams(s, { winSec: s.run.dwell, settle: 1, gain: 1, cmap: 'gray', block: 48, lo: 0, hi: 1 }),
    draw(out, s, p) {
      const r = renderChunks(s.eeg, s.run, p);
      out.append(h('div', { class: 'pair' },
        s.reference ? h('figure', {}, h('img', { src: s.reference, class: 'out' }), h('figcaption', {}, 'Reference')) : null,
        h('figure', {}, imgCanvas(r, p.cmap, 'out pixelated'), h('figcaption', {}, 'Obtained from the photographer\'s EEG'))));
      return `${s.run.cells.length}/${s.run.cols * s.run.rows} chunks fixated`;
    },
  },

  // Fig. 14: SSVEPVMP
  vmp: {
    schema: [...common,
      { type: 'section', label: 'Detection' },
      { key: 'hop', label: 'Hop', type: 'number', min: 0.1, step: 0.1, unit: 's' },
      { key: 'k', label: 'Threshold', type: 'number', step: 0.1, unit: '× MAD above median' },
      { key: 'refractory', label: 'Refractory', type: 'number', min: 0, unit: 's' },
      { key: 'before', label: 'Hit window before mark', type: 'number', min: 0, unit: 's' },
      { key: 'after', label: 'Hit window after mark', type: 'number', min: 0, unit: 's' },
      { key: 'band', label: 'Red band half-width', type: 'number', min: 0, unit: 's' },
    ],
    defaults: s => baseParams(s, { winSec: s.run.winSec, hop: 0.5, k: s.run.k ?? 2, refractory: s.run.refractory ?? 10, before: 8, after: 5, band: 3 }),
    draw(out, s, p) {
      const series = timeline(s.eeg, s.run.t0, s.run.t1, p);
      const det = detectEvents(series, { k: p.k, refractory: p.refractory, useBad: p.rejectUV <= 0 });
      const marks = s.run.marks || [];
      const sc = scoreEvents(det.onsets, marks, { before: p.before, after: p.after });
      const vals = series.map(q => q.v), lo = Math.min(...vals), hi = Math.max(...vals);
      const c = h('canvas', { class: 'chart tall' }); out.append(c);
      requestAnimationFrame(() => lineChart(c, {
        series: [{ x: series.map(q => (q.t - s.run.t0) / 60), y: series.map(q => (q.v - lo) / (hi - lo || 1)), color: '#2563eb', label: 'Normalized SSVEP power' }],
        bands: marks.map(m => ({ a: (m - s.run.t0 - p.band) / 60, b: (m - s.run.t0 + p.band) / 60, color: 'rgba(239,68,68,.45)' })),
        marks: det.onsets.map(o => ({ x: (o - s.run.t0) / 60, color: '#16a34a' })),
        hlines: [{ y: (det.thr - lo) / (hi - lo || 1), color: '#16a34a' }],
        xlabel: 'time (min)', ylabel: 'normalized SSVEP', ymin: 0, ymax: 1,
      }));
      if (s.snapshots?.length) {
        out.append(h('h4', {}, 'Visual memory (captured at high-SSVEP moments)'),
          h('div', { class: 'gallery' }, s.snapshots.map(sn => h('figure', {}, h('img', { src: sn.src }), h('figcaption', {}, `${((sn.t - s.run.t0) / 60).toFixed(2)} min`)))));
      }
      return `hit rate ${(sc.hitRate * 100).toFixed(0)} % (${sc.hits}/${sc.events} marked events) · ${sc.detections} detections, ${sc.falseAlarms} not near a mark · threshold ${det.thr.toPrecision(3)} (red = your marks, green = detections)`;
    },
  },

  // Fig. 4 (§II): on-screen sweep across the visual field
  field: {
    schema: [...common,
      { type: 'section', label: 'Rendering' },
      { key: 'hop', label: 'Sample every', type: 'number', min: 0.05, step: 0.05, unit: 's' },
      { key: 'cmap', label: 'Colormap', type: 'select', options: CMAPS },
      { key: 'dot', label: 'Light size', type: 'number', min: 1, unit: 'px' },
      { key: 'distCm', label: 'Viewing distance', type: 'number', min: 1, unit: 'cm' },
      { key: 'screenCm', label: 'Screen width', type: 'number', min: 1, unit: 'cm' },
    ],
    defaults: s => baseParams(s, { winSec: s.run.winSec, hop: 0.2, cmap: 'bluered', dot: 10, distCm: s.run.distCm ?? 50, screenCm: s.run.screenCm ?? 34, metric: 'ratio' }),
    draw(out, s, p) {
      const samples = pathSamples(s.eeg, s.run.path, p);
      const { W, H } = s.run, scale = Math.min(1, 900 / W);
      const paint = lightPainting(samples, W, H, ['x', 'y'], p.cmap, { dot: p.dot, scale });
      const ctx = paint.getContext('2d');
      if (s.run.attention === 'covert') { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; const fx = s.run.fix.x * scale, fy = s.run.fix.y * scale; ctx.beginPath(); ctx.moveTo(fx - 8, fy); ctx.lineTo(fx + 8, fy); ctx.moveTo(fx, fy - 8); ctx.lineTo(fx, fy + 8); ctx.stroke(); }
      out.append(h('figure', {}, paint, h('figcaption', {}, 'Long-exposure metaveillograph: stimulus path coloured by SSVEP (blue = weak, red = strong)')));
      const axis = s.run.pattern === 'vertical' ? 'y' : 'x', span = axis === 'x' ? W : H;
      const pr = profile(samples, axis, 0, span, 30);
      const c = h('canvas', { class: 'chart' }); out.append(c);
      requestAnimationFrame(() => lineChart(c, {
        series: [{ x: pr.centers, y: pr.fwd, color: '#ef4444', label: axis === 'x' ? 'moving right' : 'moving down' },
          { x: pr.centers, y: pr.bwd, color: '#3b82f6', label: axis === 'x' ? 'moving left' : 'moving up' },
          { x: pr.centers, y: pr.all, color: '#999', width: 1 }],
        xlabel: `${axis} position (px)`, ylabel: 'SSVEP',
      }));
      if (s.run.pattern === 'raster') {
        const sc = Math.min(1, 480 / W), w = Math.round(W * sc), hh = Math.round(H * sc);
        const img = normalize(splat(samples.map(q => q.x * sc), samples.map(q => q.y * sc), samples.map(q => q.v), samples.map(q => (q.bad ? 0 : 1)), w, hh, 30 * sc));
        out.append(h('figure', {}, imgCanvas({ img, w, h: hh }, 'hot'), h('figcaption', {}, 'Interpolated visual field map')));
      }
      const pxPerCm = W / p.screenCm, width = fwhm(pr.centers, pr.all);
      const deg = 2 * Math.atan(width / pxPerCm / 2 / p.distCm) * 180 / Math.PI;
      // hysteresis: shift between the two sweep directions' response centroids
      const cen = arr => { let s1 = 0, s0 = 0; arr.forEach((v, i) => { if (Number.isFinite(v)) { const w = Math.max(0, v - percentile(arr, 0.1)); s1 += w * pr.centers[i]; s0 += w; } }); return s0 ? s1 / s0 : NaN; };
      const hyst = cen(pr.fwd) - cen(pr.bwd);
      return `${samples.length} samples · ${s.run.attention} attention · response width (FWHM) ${width.toFixed(0)} px ≈ ${deg.toFixed(1)}° · hysteresis (direction shift) ${hyst.toFixed(0)} px`;
    },
  },

  // Fig. 4 top/bottom: plotter moving a flickering phone in front of the face (side view)
  plotter: {
    schema: [...common,
      { type: 'section', label: 'Rendering' },
      { key: 'hop', label: 'Sample every', type: 'number', min: 0.05, step: 0.05, unit: 's' },
      { key: 'cmap', label: 'Light colour', type: 'select', options: CMAPS },
      { key: 'dot', label: 'Light size', type: 'number', min: 0.5, unit: 'mm' },
    ],
    defaults: s => baseParams(s, { winSec: s.run.winSec, hop: 0.25, cmap: 'redglow', dot: 3, metric: 'ratio' }),
    draw(out, s, p) {
      const samples = pathSamples(s.eeg, { t: s.run.path.t, x: s.run.path.z, y: s.run.path.y }, p);
      const { z1, yLo, yHi } = s.run.plan;
      const scale = 4, W = z1 + 60, H = (yHi - yLo) + 40;
      const mapped = samples.map(q => ({ ...q, x: q.x + 50, y: (yHi - q.y) + 20 }));
      const paint = lightPainting(mapped, W, H, ['x', 'y'], p.cmap, { dot: p.dot, scale });
      const c = paint.getContext('2d');
      c.fillStyle = '#444'; c.beginPath(); c.ellipse(20 * scale, (yHi + 20) * scale, 26 * scale, 36 * scale, 0, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#ddd'; c.beginPath(); c.arc(44 * scale, (yHi + 12) * scale, 3 * scale, 0, Math.PI * 2); c.fill();
      out.append(h('figure', {}, paint, h('figcaption', {}, 'Side-view long exposure: eye at left, light brightness/colour = SSVEP')));
      const sc = 2, w = Math.round(W * sc), hh = Math.round(H * sc);
      const img = normalize(splat(mapped.map(q => q.x * sc), mapped.map(q => q.y * sc), mapped.map(q => q.v), mapped.map(q => (q.bad ? 0 : 1)), w, hh, 6 * sc));
      out.append(h('figure', {}, imgCanvas({ img, w, h: hh }, 'hot'), h('figcaption', {}, 'Interpolated metaveillograph (Fig. 4 bottom)')));
      return `${samples.length} samples over ${((s.run.t1 - s.run.t0) / 60).toFixed(1)} min · ${s.run.virtual ? 'virtual plotter' : 'GRBL plotter'}`;
    },
  },

  // Stimulus optimisation (§III: shape, colour, frequency optimised for response)
  tuner: {
    schema: [...common.filter(f => f.key !== 'winSec')],
    defaults: s => baseParams(s, { winSec: s.run.block }),
    draw(out, s, p) {
      const rows = new Map();
      for (const b of s.run.blocks) {
        const pp = { ...p, freq: b.freq, winSec: b.t1 - b.t0 - 1 };
        const snr = metricFor(s.eeg, (b.t0 + b.t1) / 2 + 0.5, { ...pp, metric: 'snr' });
        const har = metricFor(s.eeg, (b.t0 + b.t1) / 2 + 0.5, { ...pp, metric: 'harmonic' });
        const key = `${b.freq} Hz · ${b.colors}`;
        if (!rows.has(key)) rows.set(key, { freq: b.freq, colors: b.colors, snr: [], har: [] });
        if (snr != null) rows.get(key).snr.push(snr);
        if (har != null) rows.get(key).har.push(har);
      }
      const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
      const list = [...rows.entries()].map(([k, r]) => ({ k, ...r, m: mean(r.snr), mh: mean(r.har) })).sort((a, b) => b.m - a.m);
      out.append(h('table', { class: 'table' },
        h('tr', {}, h('th', {}, 'Stimulus'), h('th', {}, 'SNR (mean)'), h('th', {}, 'Harmonic ratio'), h('th', {}, 'Blocks')),
        list.map((r, i) => h('tr', { class: i === 0 ? 'best' : '' }, h('td', {}, r.k), h('td', {}, r.m.toFixed(2)), h('td', {}, r.mh.toFixed(3)), h('td', {}, r.snr.length)))));
      if (list[0]) {
        out.append(h('button', {
          onclick: () => { app.preferred = { freq: list[0].freq, colors: list[0].colors }; toast(`Preferred stimulus set: ${list[0].k}`); },
        }, `Use best (${list[0].k}) as default`));
      }
      return list[0] ? `best: ${list[0].k}` : 'no complete blocks';
    },
  },
};
const fmt = v => (Number.isFinite(v) ? v.toFixed(2) : '–');

// Auto-tune for scan-line sessions: score analysis variants by validity (known target) or
// split-half reliability and apply the best.
VIEWS.scan.extra = (session, values, rerender, setForm) => {
  const out = h('div', { class: 'info' });
  const btn = h('button', {}, 'Auto-tune analysis (cross-validated)');
  btn.onclick = async () => {
    btn.disabled = true;
    const metrics = ['harmonic', 'fbcca', 'coherent', 'kalman'];
    const masks = ['off', 'auto', ...(session.calib?.thresholds ? ['calib'] : [])];
    const spatials = ['off', ...(session.calib?.filter?.weights ? ['calib'] : [])];
    const cands = [];
    for (const metric of metrics) for (const maskMode of masks) for (const spatial of spatials)
      cands.push({ label: `${metric} · mask ${maskMode} · spatial ${spatial}`, p: { metric, maskMode, spatial } });
    const base = { ...values };
    const table = h('table', { class: 'table' });
    out.innerHTML = ''; out.append(h('b', {}, 'Auto-tune'), table);
    const rows = await autoTune(session.eeg, session.run, {}, cands.map(c => ({ ...c, p: finalize(session, { ...base, ...c.p }) })), (i, n) => { btn.textContent = `Scoring ${i}/${n}…`; });
    table.append(h('tr', {}, h('th', {}, 'Variant'), h('th', {}, session.run.truth ? 'Validity (vs. stimulus)' : ''), h('th', {}, 'Split-half r'), h('th', {}, 'Full-data reliability')));
    rows.forEach((r, i) => table.append(h('tr', { class: i === 0 ? 'best' : '' }, h('td', {}, r.label), h('td', {}, fmt(r.val)), h('td', {}, fmt(r.rel)), h('td', {}, fmt(r.relFull)))));
    const best = cands.find(c => c.label === rows[0].label);
    setForm(best.p);
    out.append(h('p', { class: 'muted small' }, `Applied: ${rows[0].label}. ${session.run.truth ? 'Ranked by validity against the displayed stimulus (a known target — this is the calibration of the pipeline); ' : 'Ranked by '}split-half reliability is shown for unknown scenes. Small differences (< 0.05) are within noise.`));
    btn.disabled = false; btn.textContent = 'Auto-tune analysis (cross-validated)';
    rerender();
  };
  return h('div', {}, btn, out);
};
VIEWS.rig = VIEWS.scan;
VIEWS.raster = VIEWS.scan;

function metricFor(eeg, t, p) { const m = metricAt(eeg, t - (p.latency ?? 0), p); return m ? m.v : null; }

export function showResult(container, session) {
  const view = VIEWS[session.mode];
  if (!view) { container.append(h('p', {}, `No viewer for mode ${session.mode}`)); return; }
  const values = view.defaults(session);
  const out = h('div', { class: 'result-out' });
  const stats = h('p', { class: 'stats' });
  const busy = h('span', { class: 'busy' }, '');
  let timer;
  const rerender = () => {
    busy.textContent = 'rendering…';
    clearTimeout(timer);
    timer = setTimeout(() => {
      out.innerHTML = '';
      try { stats.textContent = view.draw(out, session, finalize(session, values)) || ''; } catch (e) { console.error(e); stats.textContent = 'Error: ' + e.message; }
      busy.textContent = '';
    }, 30);
  };
  const dur = session.eeg.n / session.eeg.fs;
  const firstCanvas = () => out.querySelector('canvas');
  const theForm = form(view.schema, values, rerender);
  const extra = view.extra ? view.extra(session, values, rerender, obj => theForm.setAll(obj)) : null;
  const panel = h('div', { class: 'result' },
    h('div', { class: 'result-head' },
      h('h3', {}, session.title), busy,
      h('p', { class: 'muted' }, `${session.created.slice(0, 19).replace('T', ' ')} · ${session.source || ''} · ${(dur / 60).toFixed(1)} min EEG, ${session.eeg.channels.join(' ')}`)),
    h('div', { class: 'result-body' },
      h('div', { class: 'result-controls' }, theForm, extra,
        h('div', { class: 'btns' },
          h('button', { onclick: () => saveSession(session) }, 'Save session (.json)'),
          h('button', { onclick: () => { const c = firstCanvas(); if (c) downloadCanvas(`eyecam-${session.mode}.png`, c); } }, 'Save image (.png)'),
          h('button', { onclick: () => download(`eyecam-${session.mode}-eeg.csv`, segmentToCSV(session.eeg), 'text/csv') }, 'Export EEG (.csv)'))),
      h('div', { class: 'result-main' }, out, stats)));
  container.append(panel);
  rerender();
  return panel;
}
