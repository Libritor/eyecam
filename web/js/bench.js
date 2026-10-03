// Benchmark of SSVEP detection / reconstruction methods on hard synthetic data:
// weak SSVEP (Muse TP9/TP10, no Oz), pink noise, alpha, blinks, mains, latency phase jitter.
// Score = correlation of the reconstruction with the ground-truth image.

import { synthesize } from './sources.js';
import { renderScan, scanGrid } from './recon.js';
import { correlation, resize } from './interp.js';
import { makeCycles } from './ssvep.js';

const q = new URLSearchParams(location.search);
const out = document.getElementById('out');
const log = s => { out.textContent += s + '\n'; };

function truthImage(W, H) {
  const c = new OffscreenCanvas(W, H), x = c.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, W, H); x.fillStyle = '#fff'; x.textAlign = 'center'; x.textBaseline = 'middle';
  if (q.get('img') === 'N') { x.font = `${Math.round(H * 0.92)}px Arial`; x.fillText('N', W / 2, H * 0.53); }
  else {
    x.font = `900 ${Math.round(H * 0.5)}px Arial`; x.fillText('NO', W / 2, H * 0.33);
    x.font = `900 ${Math.round(H * 0.26)}px Arial`; x.fillText('CAMERAS', W / 2, H * 0.78);
  }
  const d = x.getImageData(0, 0, W, H).data, lum = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) lum[i] = d[i * 4] > 128 ? 1 : 0;
  return lum;
}

function scenario({ W, H, cursor, speed, rows, ssvepUV, seed, phaseJitter }) {
  const lum = truthImage(W, H);
  const meanLum = (x0, y0, s) => {
    let t = 0, n = 0;
    for (let y = Math.max(0, Math.floor(y0)); y < Math.min(H, y0 + s); y++)
      for (let x = Math.max(0, Math.floor(x0)); x < Math.min(W, x0 + s); x++) { t += lum[y * W + x]; n++; }
    return n ? t / (s * s) : 0;
  };
  const win = cursor / speed, lead = win / 2 + 1;
  const rowY = Array.from({ length: rows }, (_, r) => cursor / 2 + r * (H - cursor) / (rows - 1));
  const lines = []; let t = 10;
  for (let r = 0; r < rows; r++) { const t0 = t + lead, t1 = t0 + W / speed; lines.push({ row: r, y: rowY[r], t0, t1, x0: 0, x1: W }); t = t1 + lead + 3; }
  const freq = 15;
  const probe = tt => {
    const L = lines.find(l => tt >= l.t0 - lead && tt <= l.t1 + lead);
    if (!L) return { s: 0, f: freq };
    const x = Math.min(W, Math.max(0, (tt - L.t0) * speed));
    return { s: meanLum(x - cursor / 2, L.y - cursor / 2, cursor), f: freq };
  };
  const eeg = synthesize(0, t + 5, probe, { seed, ssvepUV, noiseUV: 9, phaseJitter });
  const run = { W, H, rows, rowY, cursor, speed, freq, winSec: win, lines };
  return { eeg, run, lum, minutes: t / 60 };
}

const ALL_METHODS = [
  ['paper: (P15+P30)/P(14–50), Welch', { metric: 'harmonic' }],
  ['paper ratio, periodogram', { metric: 'harmonic', psd: 'periodogram' }],
  ['FBCCA', { metric: 'fbcca' }],
  ['lock-in power (incoherent)', { metric: 'lockin' }],
  ['coherent lock-in', { metric: 'coherent' }],
  ['coherent + line baseline', { metric: 'coherent', rowBaseline: true }],
  ['FBCCA + line baseline', { metric: 'fbcca', rowBaseline: true }],
  ['coherent + deconvolution', { metric: 'coherent', deconv: true, deconvLambda: 0.5 }],
  ['coherent + deconv + baseline', { metric: 'coherent', deconv: true, deconvLambda: 0.5, rowBaseline: true }],
];
const METHODS = q.get('methods') === 'short' ? ALL_METHODS.filter((_, i) => [0, 2, 4].includes(i)) : ALL_METHODS;

const configs = {
  quick: { W: 320, H: 180, cursor: 20, speed: 9, rows: 16 },    // the app's quick demo, scaled
  paper: { W: 320, H: 180, cursor: 20, speed: 3, rows: 48 },    // paper: 1700-sample window, 48 lines
  // single letter N, square, ~1000 px on screen -> scale 0.32 (100 px cursor = 32)
  N10: { W: 320, H: 320, cursor: 32, speed: 8, rows: 10 },      // 25 px/s on screen, 4 s window
  N7big: { W: 320, H: 320, cursor: 48, speed: 12, rows: 7 },    // 150 px cursor, 4 s window
  N10slow: { W: 320, H: 320, cursor: 32, speed: 4, rows: 10 },  // 8 s window
};
const levels = (q.get('uv') || '1.5,3').split(',').map(Number);
const which = (q.get('cfg') || 'quick,paper').split(',');
const canv = document.getElementById('imgs');

for (const name of which) {
  for (const uv of levels) {
    const sc = scenario({ ...configs[name], ssvepUV: uv, seed: 11, phaseJitter: +(q.get('jit') ?? 0.6) });
    log(`\n== ${name}: ${configs[name].rows} lines, window ${sc.run.winSec.toFixed(2)} s, SSVEP ${uv} µV, phase jitter ${q.get('jit') ?? 0.6} rad/√s, at Oz-equivalent (TP9/TP10 see ${(uv * 0.6).toFixed(1)} µV), ${sc.minutes.toFixed(1)} min ==`);
    const outW = 160, outH = Math.round(160 * sc.run.H / sc.run.W);
    const truth = resize(sc.lum, sc.run.W, sc.run.H, outW, outH, 'bilinear');
    for (const [label, m] of METHODS) {
      const t0 = performance.now();
      const p = {
        chans: [0, 3], freq: 15, winSec: sc.run.winSec, psd: 'welch', segSec: Math.min(2, sc.run.winSec), bw: 0.25, harmonics: 2,
        latency: 0.1, rejectUV: 250, outW, cycles: makeCycles(sc.run, 15), ...m,
      };
      const r = renderScan(sc.eeg, sc.run, p, scanGrid(sc.eeg, sc.run, p));
      const c = correlation(r.img, truth);
      log(`  ${label.padEnd(34)} r = ${c.toFixed(3)}   (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
      if (uv === levels[0]) {
        const cv = document.createElement('canvas'); cv.width = r.w; cv.height = r.h; cv.title = `${name} ${label}`;
        const im = new ImageData(r.w, r.h);
        for (let i = 0; i < r.w * r.h; i++) { const v = (Number.isFinite(r.img[i]) ? r.img[i] : 0) * 255; im.data.set([v, v, v, 255], i * 4); }
        cv.getContext('2d').putImageData(im, 0, 0);
        canv.append(cv);
      }
      await new Promise(res => setTimeout(res, 0));
    }
  }
}
document.title = 'BENCH DONE';
