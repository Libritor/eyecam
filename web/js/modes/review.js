// Load saved sessions (or this browser session's runs) and re-render them with any
// analysis parameters. Also builds a synthetic demo session so the whole pipeline can be
// inspected without a headset.

import { app, parseSession, IMAGE_PRESETS, canvasToDataURL } from '../context.js';
import { h, toast, readFile } from '../ui.js';
import { showResult } from '../results.js';
import { synthesize } from '../sources.js';
import { flickerFrames, meanLum } from '../stage.js';

function demoSession() {
  const img = IMAGE_PRESETS['NO CAMERAS (Fig. 8)']();
  const W = 800, H = 450, cursor = 60, rows = 24, win = 2, speed = cursor / win;
  const frames = flickerFrames(img, W, H, [255, 255, 0], [0, 0, 255], { binary: true });
  const rowY = Array.from({ length: rows }, (_, r) => cursor / 2 + r * (H - cursor) / (rows - 1));
  const lead = win / 2 + 1, lines = [];
  let t = 5;
  for (let r = 0; r < rows; r++) { const t0 = t + lead, t1 = t0 + W / speed; lines.push({ row: r, y: rowY[r], t0, t1, x0: 0, x1: W }); t = t1 + lead + 2; }
  const probe = tt => {
    const L = lines.find(l => tt >= l.t0 - lead && tt <= l.t1 + lead);
    if (!L) return { s: 0, f: 15 };
    const x = Math.min(W, Math.max(0, (tt - L.t0) * speed));
    return { s: meanLum(frames, x - cursor / 2, L.y - cursor / 2, cursor), f: 15 };
  };
  const eeg = synthesize(0, t + 5, probe, { seed: 42, ssvepUV: 4 });
  return {
    version: 1, mode: 'raster', created: new Date().toISOString(), title: 'Demo — synthetic EEG, NO CAMERAS', source: 'synthetic',
    settings: { ...app.settings }, eeg, reference: canvasToDataURL(img),
    run: { W, H, rows, rowY, cursor, speed, freq: 15, winSec: win, lines, gain: 1 },
  };
}

export default {
  id: 'review', nav: 'Review & reprocess', title: 'Review & reprocess sessions', fig: 'offline',
  blurb: 'Every run is kept here for this browser session, and can be saved as a .json file (raw EEG + stimulus log). Load a file to re-render it with a different metric, window, channels, interpolation or colormap.',
  mount(el) {
    const out = h('div');
    const file = h('input', { type: 'file', accept: '.json,application/json', multiple: true });
    file.onchange = async () => {
      for (const f of file.files) {
        try { const s = parseSession(await readFile(f)); app.sessions.unshift(s); showResult(out, s); } catch (e) { toast(`${f.name}: ${e.message}`, 6000); }
      }
      renderList();
    };
    const list = h('div');
    const renderList = () => {
      list.innerHTML = '';
      if (!app.sessions.length) { list.append(h('p', { class: 'muted' }, 'No runs yet in this browser session.')); return; }
      for (const s of app.sessions) list.append(h('div', { class: 'hw-row' }, h('span', {}, s.title), h('span', { class: 'muted small' }, s.created.slice(11, 19)), h('button', { onclick: () => { out.innerHTML = ''; showResult(out, s); } }, 'Open')));
    };
    renderList();
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h3', {}, 'Load session files'), file,
        h('div', { class: 'btns' }, h('button', { onclick: () => { toast('Synthesising ≈ 10 min of EEG…'); setTimeout(() => { const s = demoSession(); app.sessions.unshift(s); renderList(); out.innerHTML = ''; showResult(out, s); }, 50); } }, 'Build a synthetic demo session')),
        h('p', { class: 'muted small' }, 'The demo simulates an observer whose SSVEP follows the flickering cursor over "NO CAMERAS" (4 µV SSVEP in ~9 µV pink noise, alpha, blinks, 60 Hz mains) — useful to learn the controls.')),
      h('div', { class: 'card' }, h('h3', {}, 'This session'), list)),
    out);
  },
};
