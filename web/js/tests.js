// Self-tests: DSP correctness and end-to-end reconstructions on synthetic EEG whose SSVEP
// follows exactly what the simulated eye sees. Open /tests.html (also run headless).

import { periodogram, welch, bandPower, ssvepMetric, butterLowpass, sosfilt, filtfilt, spectrum } from './dsp.js';
import { delaunay, correlation, resize } from './interp.js';
import { synthesize, decodeAthena } from './sources.js';
import { maskInfo, maskedFraction } from './artifacts.js';
import { lockin } from './ssvep.js';
import { analyzeCalibration } from './calib.js';
import { renderScan, gazeSamples, renderGaze, renderChunks, timeline, detectEvents, scoreEvents, combineRows, pathSamples, profile, fwhm } from './recon.js';

const results = [];
const check = (name, ok, detail) => results.push({ name, ok: !!ok, detail });

// ---------------------------------------------------------------- DSP
{
  const fs = 256, n = 2048, x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * 15 * i / fs);
  const s = periodogram(x, fs);
  const pk = s.f[s.p.indexOf(Math.max(...s.p))];
  const pw = bandPower(s, 14, 16);
  check('periodogram peak at 15 Hz', Math.abs(pk - 15) < 0.2, `peak ${pk.toFixed(3)} Hz`);
  check('periodogram power = A²/2', Math.abs(pw - 0.5) < 0.03, `power ${pw.toFixed(4)}`);
  const w = welch(x, fs, 512);
  check('welch power = A²/2', Math.abs(bandPower(w, 14, 16) - 0.5) < 0.03, `power ${bandPower(w, 14, 16).toFixed(4)}`);
}
{
  const sos = butterLowpass(6, 1.5, 60);
  const amp = f => {
    const n = 6000, x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * f * i / 60);
    const y = sosfilt(sos, x); let m = 0; for (let i = n / 2; i < n; i++) m = Math.max(m, Math.abs(y[i])); return m;
  };
  check('butterworth -3 dB at fc', Math.abs(amp(1.5) - Math.SQRT1_2) < 0.02, `gain(1.5Hz)=${amp(1.5).toFixed(3)}`);
  check('butterworth passband / stopband', amp(0.2) > 0.99 && amp(6) < 0.001, `gain(0.2)=${amp(0.2).toFixed(4)} gain(6)=${amp(6).toExponential(2)}`);
  const step = new Float64Array(600).fill(3);
  const ff = filtfilt(sos, step);
  check('filtfilt preserves DC without edge transients', Math.max(...ff.map(v => Math.abs(v - 3))) < 1e-6, `max err ${Math.max(...ff.map(v => Math.abs(v - 3))).toExponential(2)}`);
}
{
  const xs = [], ys = [];
  for (let j = 0; j < 6; j++) for (let i = 0; i < 7; i++) { xs.push(i + 0.001 * j * i); ys.push(j + 0.0013 * i); }
  const t = delaunay(xs, ys);
  check('delaunay triangle count', t.length === 2 * 6 * 5, `${t.length} triangles (expected 60)`);
}
{
  const grid = new Float32Array([0, 0, 0, 0, 10, 0, 0, 0, 0]), out = combineRows(grid, 1, 9);
  check('eq.(1) row weights 2:1:0.5', Math.abs(out[4] / out[3] - 2) < 1e-6 && Math.abs(out[4] / out[2] - 4) < 1e-6,
    `centre ${out[4].toFixed(2)}, ±1 ${out[3].toFixed(2)}, ±2 ${out[2].toFixed(2)}`);
}
{
  const on = synthesize(0, 20, () => ({ s: 1, f: 15 }), { seed: 3 });
  const off = synthesize(0, 20, () => ({ s: 0, f: 15 }), { seed: 4 });
  const tp = [0, 3];
  const m = seg => ssvepMetric(spectrum(tp.map(c => seg.data[c].subarray(0, 1700)), 256), 15);
  check('SSVEP metric: flicker vs none', m(on) > 3 * m(off), `on ${m(on).toFixed(4)} off ${m(off).toFixed(4)}`);
}

// ---------------------------------------------------------------- ground-truth image
function testImage(W, H) {
  const c = new OffscreenCanvas(W, H), x = c.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, W, H);
  x.fillStyle = '#fff'; x.font = `bold ${Math.round(H * 0.8)}px Arial`; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText('NO', W / 2, H / 2 + H * 0.04);
  const d = x.getImageData(0, 0, W, H).data, lum = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) lum[i] = d[i * 4] / 255;
  return lum;
}
const W = 320, H = 160, lum = testImage(W, H);
const meanLum = (x0, y0, s) => {
  let t = 0, n = 0;
  for (let y = Math.max(0, Math.floor(y0)); y < Math.min(H, y0 + s); y += 2)
    for (let x = Math.max(0, Math.floor(x0)); x < Math.min(W, x0 + s); x += 2) { t += lum[y * W + x]; n++; }
  return n ? t / n * (n * 4) / (s * s) : 0;
};
const truthAt = (outW, outH) => resize(lum, W, H, outW, outH, 'bilinear');
const blurTruth = (outW, outH, r) => { // box-blurred truth at output res (what a cursor of size r can resolve)
  const t = truthAt(outW, outH), o = new Float32Array(outW * outH);
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) {
    let s = 0, n = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= outW || yy >= outH) continue; s += t[yy * outW + xx]; n++;
    }
    o[y * outW + x] = s / n;
  }
  return o;
};

// ---------------------------------------------------------------- Raster cursor (Fig. 1 / 8)
{
  const cursor = 40, rows = 16, win = 1.2, speed = cursor / win, lead = win / 2 + 0.5, rest = 1;
  const rowY = Array.from({ length: rows }, (_, r) => cursor / 2 + r * (H - cursor) / (rows - 1));
  const lines = []; let t = 10;
  for (let r = 0; r < rows; r++) {
    const t0 = t + lead, t1 = t0 + W / speed;
    lines.push({ row: r, y: rowY[r], t0, t1, x0: 0, x1: W });
    t = t1 + lead + rest;
  }
  const probe = tt => {
    const L = lines.find(l => tt >= l.t0 - lead && tt <= l.t1 + lead);
    if (!L) return { s: 0, f: 15 };
    const x = Math.min(W, Math.max(0, (tt - L.t0) / (L.t1 - L.t0) * W));
    return { s: meanLum(x - cursor / 2, L.y - cursor / 2, cursor), f: 15 };
  };
  const eeg = synthesize(0, t + 5, probe, { seed: 7, ssvepUV: 5 });
  const run = { W, H, rows, rowY, cursor, freq: 15, lines };
  const p = { chans: [0, 3, 4], freq: 15, metric: 'harmonic', winSec: win, psd: 'welch', segSec: 1, latency: 0.1, rejectUV: 0, outW: 160, bw: 0.5, harmonics: 2 };
  const r = renderScan(eeg, run, p);
  const c = correlation(r.img, blurTruth(r.w, r.h, 4));
  check('raster eye-camera reconstruction correlates with image', c > 0.6, `r = ${c.toFixed(3)} (${rows} lines, ${(t / 60).toFixed(1)} min synthetic EEG)`);
  window.__raster = r;
}

// ---------------------------------------------------------------- Free gaze + Butterworth (Fig. 6)
{
  const rowsG = 10, speed = 60, gaze = { t: [], x: [], y: [] };
  let t = 5;
  for (let r = 0; r < rowsG; r++) {
    const y = 8 + r * (H - 16) / (rowsG - 1);
    for (let x = 0; x <= W; x += 2) { gaze.t.push(t); gaze.x.push(x + (Math.random() - 0.5) * 6); gaze.y.push(y + (Math.random() - 0.5) * 6); t += 2 / speed; }
    t += 0.4;
  }
  const at = tt => {
    let i = gaze.t.findIndex(v => v >= tt); if (i < 0) return null;
    return [gaze.x[i], gaze.y[i]];
  };
  const probe = tt => { const g = at(tt); return g ? { s: meanLum(g[0] - 20, g[1] - 20, 40), f: 12 } : { s: 0, f: 12 }; };
  const eeg = synthesize(0, t + 5, probe, { seed: 11, ssvepUV: 5 });
  const run = { W, H, freq: 12, gaze };
  const p = { chans: [0, 3, 4], freq: 12, metric: 'harmonic', winSec: 1.2, psd: 'welch', segSec: 1, latency: 0.1, rejectUV: 0, outW: 160, hop: 0.1, bw: 0.5 };
  const s = gazeSamples(eeg, run, p);
  const r = renderGaze(s, run, p);
  const c = correlation(r.img, blurTruth(r.w, r.h, 6));
  check('free-gaze (Delaunay) reconstruction correlates with image', c > 0.4, `r = ${c.toFixed(3)} from ${s.length} gaze samples`);
  window.__gaze = r;
}

// ---------------------------------------------------------------- Chunks (Fig. 12)
{
  const cols = 8, rows = 4, dwell = 4, cells = []; let t = 5;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { cells.push({ c, r, t0: t, t1: t + dwell }); t += dwell + 1; }
  const probe = tt => {
    const cell = cells.find(q => tt >= q.t0 && tt <= q.t1);
    return cell ? { s: meanLum(cell.c * W / cols, cell.r * H / rows, W / cols), f: 15 } : { s: 0, f: 15 };
  };
  const eeg = synthesize(0, t + 3, probe, { seed: 5, ssvepUV: 4 });
  const r = renderChunks(eeg, { cols, rows, freq: 15, cells }, { chans: [0, 3, 4], freq: 15, metric: 'harmonic', psd: 'welch', segSec: 1, latency: 0.1, rejectUV: 0 });
  const truth = resize(lum, W, H, cols, rows, 'bilinear');
  const c = correlation(r.grid, truth);
  check('chunk (shutter-glasses) reconstruction correlates', c > 0.7, `r = ${c.toFixed(3)} (${cols}x${rows} chunks)`);
  window.__chunks = r;
}

// ---------------------------------------------------------------- SSVEPVMP events (Fig. 14)
{
  const episodes = []; let t = 30;
  while (t < 600) { episodes.push([t, t + 8 + Math.random() * 6]); t += 40 + Math.random() * 30; }
  const probe = tt => ({ s: episodes.some(([a, b]) => tt >= a && tt <= b) ? 1 : 0.25, f: 15 });
  const eeg = synthesize(0, 620, probe, { seed: 9, ssvepUV: 3 });
  const p = { chans: [0, 3, 4], freq: 15, metric: 'harmonic', winSec: 3, hop: 0.5, psd: 'welch', segSec: 1, latency: 0.1, rejectUV: 0 };
  const series = timeline(eeg, 0, 620, p);
  const det = detectEvents(series, { k: 2.5, refractory: 10, useBad: true });
  const marks = episodes.map(([a]) => a + 2);
  const sc = scoreEvents(det.onsets, marks, { before: 6, after: 6 });
  check('SSVEPVMP event hit rate', sc.hitRate >= 0.8, `hit ${(sc.hitRate * 100).toFixed(0)} % (${sc.hits}/${sc.events}), ${sc.falseAlarms} false alarms`);
}

// ---------------------------------------------------------------- Visual field sweep (Fig. 4)
{
  const path = { t: [], x: [], y: [] }; let t = 3, x = 0, dir = 1;
  for (let k = 0; k < 6 * 400; k++) { path.t.push(t); path.x.push(x); path.y.push(0); t += 0.05; x += dir * 4; if (x >= 800 || x <= 0) dir = -dir; }
  const sigma = 120;
  const probe = tt => { const i = Math.min(path.t.length - 1, Math.max(0, Math.round((tt - 3) / 0.05))); return { s: Math.exp(-((path.x[i] - 400) ** 2) / (2 * sigma * sigma)), f: 12 }; };
  const eeg = synthesize(0, t + 3, probe, { seed: 21, ssvepUV: 5 });
  const s = pathSamples(eeg, path, { chans: [0, 3, 4], freq: 12, metric: 'ratio', winSec: 1.5, hop: 0.2, psd: 'welch', segSec: 1, latency: 0.1, rejectUV: 0, bw: 0.5 });
  const pr = profile(s, 'x', 0, 800, 20);
  const width = fwhm(pr.centers, pr.all), peakAt = pr.centers[pr.all.indexOf(Math.max(...pr.all.filter(Number.isFinite)))];
  check('visual-field profile peaks at the fovea', Math.abs(peakAt - 400) < 120, `peak at ${peakAt.toFixed(0)} px (true 400), FWHM ${width.toFixed(0)} px (true ${(2.355 * sigma).toFixed(0)})`);
}

// ---------------------------------------------------------------- Muse S Athena decoder
// Fixture: a 2-packet notification decoded by muselsl's own athena.py (reference).
{
  const fx = await (await fetch('js/athena_fixture.json')).json();
  const u8 = Uint8Array.from(fx.hex.match(/../g).map(b => parseInt(b, 16)));
  const got = decodeAthena(u8).filter(b => b.type === 'eeg').map(b => b.rows);
  let maxErr = 0, shapeOk = got.length === fx.eeg.length;
  for (let i = 0; shapeOk && i < got.length; i++) {
    if (got[i].length !== fx.eeg[i].length) { shapeOk = false; break; }
    got[i].forEach((row, s) => row.forEach((v, c) => { maxErr = Math.max(maxErr, Math.abs(v - fx.eeg[i][s][c])); }));
  }
  check('Muse S Athena decoder matches muselsl', shapeOk && maxErr < 1e-3, `${got.length} EEG blocks, max |Δ| ${maxErr.toExponential(2)} µV`);
}

// ---------------------------------------------------------------- noise handling
{
  // 1. artifact mask: injected jaw clenches are masked, clean data mostly is not
  const events = [{ t: 20, type: 'clench', dur: 4 }, { t: 40, type: 'clench', dur: 4 }];
  const eeg = synthesize(0, 60, () => ({ s: 0, f: 15 }), { seed: 3, muse: true, events, channels: ['TP9', 'AF7', 'AF8', 'TP10'] });
  const p = { chans: [0, 3], mask: { mode: 'auto', k: 5 } };
  const mi = maskInfo(eeg, p), bs = mi.bs;
  const frac = (a, b) => { let s = 0, n = 0; for (let k = Math.floor(a * 256 / bs); k < Math.floor(b * 256 / bs); k++) { s += mi.bad[k]; n++; } return s / n; };
  const inClench = (frac(20.5, 24) + frac(40.5, 44)) / 2;
  check('artifact mask catches jaw clenches', inClench > 0.7, `${(inClench * 100).toFixed(0)} % of clench time masked, ${(maskedFraction(eeg, p) * 100).toFixed(0)} % of whole recording`);
}
{
  // 2. masked lock-in: amplitude estimate unbiased when 40 % of samples are masked
  const fs = 256, n = 1024, x = new Float32Array(n), t = new Float64Array(n), w = new Float32Array(n);
  for (let i = 0; i < n; i++) { t[i] = i / fs; x[i] = 2 * Math.sin(2 * Math.PI * 15 * t[i] + 0.7) + 30 * (i > 300 && i < 500 ? 1 : 0); w[i] = i > 280 && i < 520 ? 0 : 1; }
  const z = lockin([x], t, tt => 15 * tt, 1, w), amp = Math.hypot(z.re[0][0], z.im[0][0]);
  const z0 = lockin([x], t, tt => 15 * tt, 1), amp0 = Math.hypot(z0.re[0][0], z0.im[0][0]);
  check('masked lock-in recovers amplitude under a step artifact', Math.abs(amp - 2) < 0.3, `masked ${amp.toFixed(2)} µV vs. unmasked ${amp0.toFixed(2)} µV (true 2.00)`);
}
{
  // 3. spatial filter: strong shared reference noise is cancelled using AF7/AF8 -> adopted
  const blocks = [], tasks = []; let t = 2;
  for (let k = 0; k < 8; k++) for (const on of [true, false]) { blocks.push({ on, t0: t, t1: t + 5, cycle: k }); t += 5; }
  const eeg = synthesize(0, t + 2, tt => ({ s: blocks.some(b => b.on && tt >= b.t0 && tt < b.t1) ? 1 : 0, f: 15 }),
    { seed: 8, ssvepUV: 6, muse: true, refNoiseUV: 30, channels: ['TP9', 'AF7', 'AF8', 'TP10'] });
  const rep = analyzeCalibration(eeg, { freq: 15, blocks, tasks }, { latency: 0.1 });
  check('spatial filter cancels reference noise (cross-validated)', rep.filter.use && rep.filter.gainD1 > 1.2,
    `held-out d′ ${rep.pipelines.base.d1.toFixed(2)} → ${rep.pipelines.filter.d1.toFixed(2)} (×${rep.filter.gainD1.toFixed(2)}), weights ${Object.entries(rep.filter.weights).map(([n, w]) => n + ' ' + w.toFixed(2)).join(', ')}`);
}
const pass = results.filter(r => r.ok).length;
document.getElementById('out').textContent =
  results.map(r => `${r.ok ? 'PASS' : 'FAIL'}  ${r.name}  —  ${r.detail}`).join('\n') + `\n\n${pass}/${results.length} passed`;
document.title = `tests ${pass}/${results.length}`;
for (const [id, r] of [['c1', window.__raster], ['c2', window.__gaze], ['c3', window.__chunks]]) {
  if (!r) continue;
  const cv = document.getElementById(id); cv.width = r.w; cv.height = r.h;
  const im = new ImageData(r.w, r.h);
  for (let i = 0; i < r.w * r.h; i++) { const v = (Number.isFinite(r.img[i]) ? r.img[i] : 0) * 255; im.data.set([v, v, v, 255], i * 4); }
  cv.getContext('2d').putImageData(im, 0, 0);
}
