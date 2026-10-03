// EyeCam app shell: connection, global analysis settings, hardware, navigation.

import { app } from './context.js';
import { h, form, toast } from './ui.js';
import { MuseBLE, BridgeSource, Simulator } from './sources.js';
import { METRICS } from './dsp.js';
import { EyeCamIO } from './serial.js';

import monitor from './modes/monitor.js';
import calibrate from './modes/calibrate.js';
import raster from './modes/raster.js';
import freegaze from './modes/freegaze.js';
import field from './modes/field.js';
import rig from './modes/rig.js';
import chunks from './modes/chunks.js';
import vmp from './modes/vmp.js';
import tuner from './modes/tuner.js';
import veillance from './modes/veillance.js';
import review from './modes/review.js';

const MODES = [monitor, calibrate, raster, freegaze, field, rig, chunks, vmp, tuner, veillance, review];

// ------------------------------------------------------------------ connection
const statusEl = document.getElementById('conn-status');
const infoEl = document.getElementById('conn-info');
app.statusListeners.add(msg => { statusEl.textContent = msg; });

async function connect(kind) {
  try {
    await app.source?.disconnect?.();
    app.source = null;
    const onStatus = m => app.status(m);
    let src;
    if (kind === 'muse') {
      if (!MuseBLE.available()) throw new Error('Web Bluetooth is not available — use Chrome or Edge on http://localhost, or the Python bridge.');
      src = new MuseBLE(app.store, { aux: document.getElementById('aux').checked, onStatus });
    } else if (kind === 'bridge') {
      src = new BridgeSource(app.store, { onStatus, onGaze: (t, xy) => { for (const f of app.gazeListeners) f(t, xy); } });
    } else src = new Simulator(app.store, { onStatus });
    await src.connect();
    app.source = src;
    document.body.dataset.source = kind;
    toast(`Connected: ${src.info()}`);
  } catch (e) {
    console.error(e);
    app.status('not connected');
    toast(e.message || String(e), 6000);
  }
}
document.getElementById('btn-muse').onclick = () => connect('muse');
document.getElementById('btn-bridge').onclick = () => connect('bridge');
document.getElementById('btn-sim').onclick = () => connect('sim');
document.getElementById('btn-disc').onclick = async () => { await app.source?.disconnect?.(); app.source = null; delete document.body.dataset.source; app.status('not connected'); };

// sample-rate / packet health readout
let lastN = 0, lastT = performance.now();
setInterval(() => {
  const now = performance.now(), n = app.store.received;
  const rate = (n - lastN) / ((now - lastT) / 1000); lastN = n; lastT = now;
  infoEl.textContent = app.source ? `${app.source.info()} · ${rate.toFixed(0)} Hz` : '';
  document.getElementById('dot').className = 'dot ' + (app.source ? (rate > 200 ? 'ok' : rate > 0 ? 'warn' : 'bad') : '');
}, 1000);

// ------------------------------------------------------------------ settings drawer
const settingsSchema = [
  { type: 'section', label: 'SSVEP analysis (defaults for every experiment)' },
  { key: 'chans', label: 'Channels', type: 'text', help: "'auto' = TP9+TP10. Calibration sets the best single channel (e.g. AUX for an Oz electrode). Or list e.g. TP10,AUX" },
  { key: 'metric', label: 'Metric', type: 'select', options: Object.entries(METRICS) },
  { key: 'psd', label: 'PSD', type: 'select', options: [['welch', 'Welch (2 s segments, 50 % overlap)'], ['periodogram', 'Single periodogram']] },
  { key: 'bw', label: 'Peak half-width', type: 'number', min: 0.05, step: 0.05, unit: 'Hz' },
  { key: 'harmonics', label: 'Harmonics', type: 'number', min: 1, max: 4, step: 1, help: 'paper: fundamental + 2nd harmonic (15 + 30 Hz)' },
  { key: 'latency', label: 'Latency compensation', type: 'number', step: 0.01, unit: 's' },
  { type: 'section', label: 'Noise handling' },
  { key: 'mask', label: 'Artifact masking', type: 'select', options: [['off', 'Off (whole-window reject)'], ['auto', 'Auto (from the recording)'], ['calib', 'Calibrated thresholds']], help: 'Ignore only contaminated 0.25 s blocks: blinks, jaw EMG, electrode shifts.' },
  { key: 'maskK', label: 'Auto threshold', type: 'number', min: 1, step: 0.5, unit: '× MAD above median' },
  { key: 'minValid', label: 'Min clean data per window', type: 'number', min: 0.05, max: 1, step: 0.05 },
  { key: 'spatial', label: 'Spatial filter', type: 'select', options: [['off', 'Off'], ['calib', 'Calibrated (needs Calibration)']] },
  { key: 'rejectUV', label: 'Whole-window rejection', type: 'number', min: 0, unit: 'µV p-p, when masking is off (0 = off)' },
];
const settingsForm = form(settingsSchema, app.settings);
document.getElementById('settings-body').append(settingsForm);
app.refreshSettings = () => settingsForm.setAll(app.settings);

// ------------------------------------------------------------------ hardware
const hw = document.getElementById('hw-body');
function hwRow(label, dev, connectFn, extra) {
  const st = h('span', { class: 'muted' }, 'not connected');
  return h('div', { class: 'hw-row' }, h('b', {}, label), st,
    h('button', {
      onclick: async () => {
        try { await connectFn(); st.textContent = 'connected'; } catch (e) { toast(e.message, 5000); }
      },
    }, 'Connect…'),
    h('button', { onclick: async () => { await dev.close(); st.textContent = 'not connected'; } }, 'Close'), extra);
}
if (!EyeCamIO.available()) hw.append(h('p', { class: 'warn' }, 'Web Serial unavailable in this browser (use Chrome/Edge).'));
hw.append(
  hwRow('EyeCam Arduino (LED / lamp / shutter glasses)', app.io, () => app.io.connect(),
    h('span', {}, h('button', { onclick: () => app.io.color(255, 0, 0) }, 'LED red'), h('button', { onclick: () => app.io.color(0, 0, 255) }, 'LED blue'),
      h('button', { onclick: () => app.io.lampFlicker(15) }, 'Lamp 15 Hz'), h('button', { onclick: () => app.io.shutter(15) }, 'Glasses 15 Hz'),
      h('button', { onclick: () => app.io.stopAll() }, 'All off'))),
  hwRow('GRBL plotter (Fig. 4)', app.grbl, () => app.grbl.connect(),
    h('span', {}, h('button', { onclick: () => app.grbl.send('$H').catch(e => toast(e.message)) }, 'Home'),
      h('button', { onclick: () => app.grbl.send('G92 X0 Y0 Z0').catch(e => toast(e.message)) }, 'Zero here (eye)'),
      h('button', { onclick: () => app.grbl.reset() }, 'Reset'))),
  h('p', { class: 'muted' }, 'Firmware: firmware/eyecam_io/eyecam_io.ino. The flickering phone of Fig. 4: open /flicker.html on the phone (run server.py --lan).'),
);

// ------------------------------------------------------------------ navigation
const nav = document.getElementById('nav'), main = document.getElementById('main');
let current = null;
async function show(id) {
  const mode = MODES.find(m => m.id === id) || MODES[0];
  if (current?.unmount) { try { await current.unmount(); } catch (e) { console.error(e); } }
  app.probe(null);
  main.innerHTML = '';
  for (const b of nav.querySelectorAll('button')) b.classList.toggle('active', b.dataset.id === mode.id);
  current = mode;
  const head = h('div', { class: 'mode-head' }, h('h2', {}, mode.title), mode.fig ? h('span', { class: 'fig' }, mode.fig) : null);
  const body = h('div', { class: 'mode-body' });
  main.append(head, mode.blurb ? h('p', { class: 'blurb' }, mode.blurb) : null, body);
  try { await mode.mount(body); } catch (e) { console.error(e); body.append(h('p', { class: 'warn' }, e.message)); }
  if (location.hash !== '#' + mode.id) history.replaceState(null, '', '#' + mode.id);
}
for (const m of MODES) nav.append(h('button', { 'data-id': m.id, onclick: () => show(m.id) }, h('span', { class: 'nav-title' }, m.nav || m.title), m.fig ? h('span', { class: 'nav-fig' }, m.fig) : null));
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1) || 'monitor');

for (const id of ['settings', 'hw']) {
  document.getElementById(`btn-${id}`).onclick = () => document.getElementById(`${id}-panel`).classList.toggle('open');
}
window.eyecam = app; // handy in the console
