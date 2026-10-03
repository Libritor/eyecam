// Noise-handling benchmark: simulated calibration + letter-N scan on Muse-like data
// (temporalis EMG bursts, electrode shifts, blinks through the FPz reference, common
// reference noise). Scores every option by validity (correlation with the ideal-observer
// grid) and checks that split-half reliability ranks options the same way.

import { synthesize } from './sources.js';
import { scanGrid, validity, reliability, expectedGrid } from './recon.js';
import { analyzeCalibration } from './calib.js';
import { makeCycles } from './ssvep.js';
import { maskedFraction } from './artifacts.js';

const q = new URLSearchParams(location.search);
const out = document.getElementById('out');
const log = s => { out.textContent += s + '\n'; };
const uv = +(q.get('uv') ?? 2), seed = +(q.get('seed') ?? 5), muse = q.get('muse') !== '0';

// ---------------------------------------------------------------- letter N, ideal grid
const W = 320, H = 320;
function letterN() {
  const c = new OffscreenCanvas(W, H), x = c.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, W, H); x.fillStyle = '#fff'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.font = `${Math.round(H * 0.92)}px Arial`; x.fillText('N', W / 2, H * 0.53);
  const d = x.getImageData(0, 0, W, H).data, lum = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) lum[i] = d[i * 4] > 128 ? 1 : 0;
  return lum;
}
const lum = letterN();
const meanLum = (x0, y0, s) => { let t = 0; for (let y = Math.max(0, Math.floor(y0)); y < Math.min(H, y0 + s); y++) for (let x = Math.max(0, Math.floor(x0)); x < Math.min(W, x0 + s); x++) t += lum[y * W + x]; return t / (s * s); };

// ---------------------------------------------------------------- 1. calibration recording
function calibration(seedC) {
  const freq = 15, blocks = [], tasks = [], events = [];
  let t = 2;
  for (let k = 0; k < 8; k++) for (const on of [true, false]) { blocks.push({ on, t0: t, t1: t + 5, cycle: k }); t += 5; }
  for (const [name, dur] of [['rest', 8], ['blink', 6], ['clench', 5], ['swallow', 5], ['move', 6]]) { t += 2.2; tasks.push({ name, t0: t, t1: t + dur }); events.push({ t, type: name, dur }); t += dur + 1.5; }
  const probe = tt => ({ s: blocks.some(b => b.on && tt >= b.t0 && tt < b.t1) ? 1 : 0, f: freq });
  const eeg = synthesize(0, t + 2, probe, { seed: seedC, ssvepUV: uv, muse, phaseJitter: 0.25, events, channels: ['TP9', 'AF7', 'AF8', 'TP10'] });
  return analyzeCalibration(eeg, { freq, blocks, tasks, flicker: null }, { latency: 0.1, cursor: 32, letterPx: 320 });
}

// ---------------------------------------------------------------- 2. N scan
function scan({ speed, passes, cursor = 32, rows = 10, seedS }) {
  const win = cursor / speed, lead = win / 2 + 1;
  const rowY = Array.from({ length: rows }, (_, r) => cursor / 2 + r * (H - cursor) / (rows - 1));
  const lines = []; let t = 5;
  for (let pass = 0; pass < passes; pass++) for (let r = 0; r < rows; r++) {
    const dir = pass % 2 ? -1 : 1, t0 = t + lead, t1 = t0 + W / speed;
    lines.push({ row: r, pass, y: rowY[r], t0, t1, x0: dir > 0 ? 0 : W, x1: dir > 0 ? W : 0 }); t = t1 + lead + 3;
  }
  const probe = tt => {
    const L = lines.find(l => tt >= l.t0 - lead && tt <= l.t1 + lead);
    if (!L) return { s: 0, f: 15 };
    const a = Math.min(1, Math.max(0, (tt - L.t0) / (L.t1 - L.t0))), x = L.x0 + a * (L.x1 - L.x0);
    return { s: meanLum(x - cursor / 2, L.y - cursor / 2, cursor), f: 15 };
  };
  const eeg = synthesize(0, t + 5, probe, { seed: seedS, ssvepUV: uv, muse, phaseJitter: 0.25, channels: ['TP9', 'AF7', 'AF8', 'TP10'] });
  const tw = 160, tl = new Float32Array(tw * tw);
  for (let y = 0; y < tw; y++) for (let x = 0; x < tw; x++) tl[y * tw + x] = lum[(y * 2) * W + x * 2];
  return { eeg, run: { W, H, rows, rowY, cursor, speed, freq: 15, winSec: win, lines, truth: { w: tw, h: tw, lum: tl } }, minutes: t / 60 };
}

const fmt = v => (Number.isFinite(v) ? v.toFixed(3) : '  –  ');
log(`Muse-like noise: ${muse ? 'ON (EMG bursts, electrode shifts, blinks via FPz, common reference noise)' : 'OFF'} · SSVEP ${uv} µV (TP9/TP10 see ${(uv * 0.6).toFixed(1)}) · seed ${seed}\n`);

let tc = performance.now();
const cal = calibration(seed + 100);
log(`(calibration synth+analysis ${((performance.now() - tc) / 1000).toFixed(1)} s)`);
log(`== Calibration: verdict ${cal.verdict.toUpperCase()}, d′/s base ${fmt(cal.pipelines.base.d1)} → filter ${fmt(cal.pipelines.filter.d1)} (held-out), filter ${cal.filter.use ? 'ADOPTED' : 'not adopted'}; weights ${Object.entries(cal.filter.weights).map(([n, w]) => `${n} ${w.toFixed(2)}`).join(', ')}`);
log(`   plan: window ${cal.plan.winSec} s, ${cal.plan.speed} px/s (of a 32 px cursor), ${cal.plan.passes} pass(es)`);
log(`   artifact catch: ` + cal.artifactReport.filter(a => /TP/.test(a.name)).map(a => `${a.name} clean-flagged ${(a.cleanFlagged * 100).toFixed(0)}% clench ${(a.caught.clench * 100).toFixed(0)}% move ${(a.caught.move * 100).toFixed(0)}%`).join(' | '));

const base = { chans: [0, 3], freq: 15, psd: 'welch', bw: 0.25, harmonics: 2, latency: 0.1, rejectUV: 250, minValid: 0.4, combine: true, reject: true };
const spatial = { spatialChans: Object.keys(cal.filter.weights).map((_, i) => i), spatialW: Object.values(cal.filter.weights) };
const variants = [
  ['paper metric, whole-window reject (old default)', { metric: 'harmonic' }],
  ['paper metric, no rejection at all', { metric: 'harmonic', rejectUV: 0 }],
  ['paper metric + auto mask', { metric: 'harmonic', mask: { mode: 'auto', k: 5 } }],
  ['paper metric + calibrated mask', { metric: 'harmonic', mask: { mode: 'calib', thr: cal.thresholds } }],
  ['paper metric + calib mask + spatial filter', { metric: 'harmonic', mask: { mode: 'calib', thr: cal.thresholds }, ...spatial }],
  ['FBCCA + calib mask', { metric: 'fbcca', mask: { mode: 'calib', thr: cal.thresholds } }],
  ['FBCCA + calib mask + spatial filter', { metric: 'fbcca', mask: { mode: 'calib', thr: cal.thresholds }, ...spatial }],
  ['coherent + calib mask + spatial filter', { metric: 'coherent', mask: { mode: 'calib', thr: cal.thresholds }, ...spatial }],
];

const CFGS = [{ label: '1 pass, 8 px/s (4 s window)', speed: 8, passes: 1 }, { label: '2 passes, 16 px/s (same total time)', speed: 16, passes: 2 }, { label: '2 passes, 8 px/s (double time)', speed: 8, passes: 2 }];
const pick = (arr, key) => (q.get(key) ? q.get(key).split(',').map(Number).map(i => arr[i]) : arr);
for (const cfg of pick(CFGS, 'cfg')) {
  const sc = scan({ ...cfg, seedS: seed });
  log(`\n== Scan: ${cfg.label}, ${sc.minutes.toFixed(1)} min ==   validity (vs. ideal) | split-half reliability`);
  for (const [label, v] of pick(variants, 'v')) {
    const tv = performance.now();
    const p = { ...base, ...v, winSec: sc.run.winSec, segSec: Math.min(2, sc.run.winSec), cycles: makeCycles(sc.run, 15) };
    const g = scanGrid(sc.eeg, sc.run, p);
    const val = validity(sc.eeg, sc.run, p, g), rel = reliability(sc.eeg, sc.run, p);
    const mf = p.mask ? ` (${(maskedFraction(sc.eeg, { ...p, maskChans: p.chans }) * 100).toFixed(0)}% masked)` : '';
    log(`  ${(label + mf).padEnd(58)} ${fmt(val)}   | ${fmt(rel.r)} (${rel.method})  [${((performance.now() - tv) / 1000).toFixed(1)} s]`);
    await new Promise(r => setTimeout(r, 0));
  }
}
document.title = 'NOISE DONE';
