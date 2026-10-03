// Shared application state: EEG store + source, hardware, analysis settings, sessions,
// audio cues and the stimulus image library.

import { EEGStore, resolveChannels, encodeSegment, decodeSegment } from './eeg-store.js';
import { EyeCamIO, Grbl } from './serial.js';
import { DEFAULTS } from './recon.js';
import { h, form, loadImage, readFile, stamp, download } from './ui.js';

export const app = {
  store: new EEGStore(),
  source: null,
  io: new EyeCamIO(),
  grbl: new Grbl(),
  gazeListeners: new Set(),
  sessions: [],
  settings: {
    chans: 'auto',            // 'auto' or comma list e.g. "TP9,TP10"
    metric: DEFAULTS.metric,
    psd: DEFAULTS.psd,
    latency: DEFAULTS.latency,
    rejectUV: DEFAULTS.rejectUV,
    bw: DEFAULTS.bw,
    harmonics: DEFAULTS.harmonics,
    mask: 'auto',             // artifact masking: 'off' | 'auto' (data-driven) | 'calib' (calibrated thresholds)
    maskK: 5,                 // auto thresholds: median + k·MAD
    minValid: 0.4,            // windows with less clean data than this are dropped
    spatial: 'off',           // 'off' | 'calib' (calibrated spatial filter)
  },
  calib: null,                // latest calibration report (calib.js), persisted in localStorage
  statusListeners: new Set(),
  status(msg) { for (const f of this.statusListeners) f(msg); },
  // The simulator's SSVEP follows whatever the running experiment registers here.
  probe(fn) { if (this.source?.kind === 'sim') this.source.setProbe(fn); },
  get connected() { return !!this.source; },
};

// Artifact-mask and spatial-filter parameters from settings + a calibration report, resolved
// against the channel names of a particular recording.
export function advancedParams(settings, calib, eeg) {
  const out = { minValid: settings.minValid ?? 0.4 };
  const mode = settings.mask || 'off';
  if (mode !== 'off') {
    out.mask = mode === 'calib' && calib?.thresholds ? { mode: 'calib', thr: calib.thresholds, k: settings.maskK ?? 5 } : { mode: 'auto', k: settings.maskK ?? 5 };
  }
  if (settings.spatial === 'calib' && calib?.filter?.weights) {
    const ws = Object.entries(calib.filter.weights).filter(([, w]) => Math.abs(w) > 1e-6);
    const idx = ws.map(([n]) => eeg.channels.indexOf(n));
    if (ws.length && idx.every(i => i >= 0)) { out.spatialChans = idx; out.spatialW = ws.map(([, w]) => w); }
  }
  return out;
}

// Analysis parameters for a given stimulus frequency and window length.
export function analysisParams(freq, winSec, eeg = app.store, extra = {}) {
  const s = app.settings;
  const sel = s.chans === 'auto' ? null : String(s.chans).split(',').map(x => x.trim()).filter(Boolean);
  return {
    chans: resolveChannels(eeg, sel), freq, winSec,
    metric: s.metric, psd: s.psd, segSec: Math.min(2, winSec), bw: s.bw, harmonics: s.harmonics,
    latency: s.latency, rejectUV: s.rejectUV, ...advancedParams(s, app.calib, eeg), ...extra,
  };
}

// ------------------------------------------------------------------ calibration persistence
const CALIB_KEY = 'eyecam.calib.v1';
export function setCalibration(report, { apply = true } = {}) {
  app.calib = report;
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(report)); } catch { /* private mode */ }
  if (apply && report) {
    app.settings.mask = 'calib';
    app.settings.spatial = report.filter?.use ? 'calib' : 'off';
    app.settings.calibApplied = report.created;
    app.settings.chans = report.pipeline === 'single' && report.bestChannel ? report.bestChannel : 'auto';
    app.refreshSettings?.();
  }
}
try { const c = localStorage.getItem(CALIB_KEY); if (c) app.calib = JSON.parse(c); } catch { /* ignore */ }

export function requireSource() {
  if (!app.source) throw new Error('Connect the Muse (or start the simulator) first — top right.');
}

// ------------------------------------------------------------------ sessions
export function makeSession(mode, run, t0, t1, extra = {}) {
  const s = {
    version: 1, mode, created: new Date().toISOString(), title: `${mode} ${stamp()}`,
    source: app.source?.info?.() || '', settings: { ...app.settings }, calib: app.calib, run,
    eeg: app.store.segment(t0 - 12, t1 + 12), ...extra,
  };
  app.sessions.unshift(s);
  return s;
}
export function saveSession(s) {
  const out = JSON.stringify({ ...s, eeg: encodeSegment(s.eeg) });
  download(`eyecam-${s.mode}-${s.created.replace(/[:.]/g, '-').slice(0, 19)}.json`, out, 'application/json');
}
export function parseSession(text) {
  const o = JSON.parse(text);
  if (!o.eeg || !o.mode) throw new Error('not an EyeCam session file');
  o.eeg = decodeSegment(o.eeg);
  return o;
}

export const STIM_PAIRS = { 'yellow/blue': ['#ffff00', '#0000ff'], 'white/black': ['#ffffff', '#000000'], 'red/black': ['#ff0000', '#000000'], 'green/black': ['#00ff00', '#000000'], 'white/gray': ['#ffffff', '#808080'] };
// Apply the stimulus chosen in the tuner (if any) to a mode's settings object.
export function applyPreferred(v) {
  if (!app.preferred) return v;
  v.freq = app.preferred.freq;
  const pair = STIM_PAIRS[app.preferred.colors];
  if (pair && 'colA' in v) [v.colA, v.colB] = pair;
  return v;
}

// ------------------------------------------------------------------ audio cues
let actx;
function ac() { actx ||= new AudioContext(); if (actx.state === 'suspended') actx.resume(); return actx; }
export function beep(freq = 880, dur = 0.06, gain = 0.25) {
  const a = ac(), o = a.createOscillator(), g = a.createGain();
  o.frequency.value = freq; g.gain.value = gain;
  g.gain.setTargetAtTime(0, a.currentTime + dur * 0.6, dur / 4);
  o.connect(g).connect(a.destination); o.start(); o.stop(a.currentTime + dur * 2);
}
export function say(text) {
  if (!('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; speechSynthesis.speak(u);
}

// ------------------------------------------------------------------ stimulus images
function textImage(lines, { w = 1600, h = 900, bg = '#000', fg = '#fff', sizes = [0.62], font = 'Arial Black, Arial, sans-serif' } = {}) {
  const c = h_canvas(w, h), x = c.getContext('2d');
  x.fillStyle = bg; x.fillRect(0, 0, w, h); x.fillStyle = fg; x.textAlign = 'center'; x.textBaseline = 'middle';
  const total = sizes.reduce((a, b) => a + b, 0) * h;
  let y = (h - total) / 2;
  lines.forEach((t, i) => {
    const px = sizes[i] * h;
    let fs = px; x.font = `900 ${fs}px ${font}`;
    while (x.measureText(t).width > w * 0.92 && fs > 10) { fs -= 4; x.font = `900 ${fs}px ${font}`; }
    x.fillText(t, w / 2, y + px / 2 + px * 0.04); y += px;
  });
  return c;
}
function h_canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

export const IMAGE_PRESETS = {
  'N (single letter)': () => textImage(['N'], { w: 900, h: 900, sizes: [0.92], font: 'Arial, Helvetica, sans-serif' }),
  'NO CAMERAS (Fig. 8)': () => textImage(['NO', 'CAMERAS'], { sizes: [0.56, 0.3] }),
  'NO (Fig. 6/7)': () => textImage(['NO'], { sizes: [0.8] }),
  'Test pattern': () => {
    const c = h_canvas(1600, 900), x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, 1600, 900); x.fillStyle = '#fff';
    x.beginPath(); x.arc(400, 450, 260, 0, Math.PI * 2); x.fill();
    x.fillRect(820, 190, 520, 140); x.fillRect(1000, 190, 140, 520);
    x.fillStyle = '#777'; x.fillRect(820, 600, 520, 110);
    return c;
  },
  'Gradient': () => {
    const c = h_canvas(1600, 900), x = c.getContext('2d'), g = x.createLinearGradient(0, 0, 1600, 0);
    g.addColorStop(0, '#000'); g.addColorStop(1, '#fff'); x.fillStyle = g; x.fillRect(0, 0, 1600, 900); return c;
  },
};

// Image picker widget: presets, file upload (e.g. a face photo, Fig. 1), webcam snapshot.
export function imagePicker(onPick, initial = 'NO CAMERAS (Fig. 8)') {
  const preview = h('canvas', { class: 'thumb' });
  const show = img => {
    preview.width = 320; preview.height = Math.round(320 * img.height / img.width);
    preview.getContext('2d').drawImage(img, 0, 0, preview.width, preview.height);
    onPick(img);
  };
  const sel = h('select', {}, Object.keys(IMAGE_PRESETS).map(k => h('option', { value: k, selected: k === initial }, k)), h('option', { value: '__file' }, 'Upload image…'), h('option', { value: '__cam' }, 'Webcam snapshot…'));
  const file = h('input', { type: 'file', accept: 'image/*', style: 'display:none' });
  file.onchange = async () => { if (file.files[0]) show(await loadImage(await readFile(file.files[0], 'dataurl'))); };
  sel.onchange = async () => {
    if (sel.value === '__file') file.click();
    else if (sel.value === '__cam') show(await webcamSnapshot());
    else show(IMAGE_PRESETS[sel.value]());
  };
  const el = h('div', { class: 'picker' }, h('label', { class: 'form-row' }, h('span', { class: 'lbl' }, 'Subject matter'), sel), file, preview);
  el.select = name => { sel.value = name; show(IMAGE_PRESETS[name]()); };
  el.load = img => show(img);
  queueMicrotask(() => show(IMAGE_PRESETS[initial]()));
  return el;
}

export async function webcamSnapshot() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
  const v = h('video', { autoplay: true, playsinline: true, muted: true });
  v.srcObject = stream; await v.play(); await new Promise(r => setTimeout(r, 1200));
  const c = h_canvas(v.videoWidth, v.videoHeight); c.getContext('2d').drawImage(v, 0, 0);
  stream.getTracks().forEach(t => t.stop());
  return c;
}

export function canvasToDataURL(img, maxW = 800) {
  const s = Math.min(1, maxW / img.width), c = h_canvas(Math.round(img.width * s), Math.round(img.height * s));
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

export function rotateCCW(img) {
  const c = h_canvas(img.height, img.width), x = c.getContext('2d');
  x.translate(0, img.width); x.rotate(-Math.PI / 2); x.drawImage(img, 0, 0);
  return c;
}

export { h, form, readFile };
