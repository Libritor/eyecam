// §II / Fig. 4: photograph a person's capacity to see (metaveillography). A flickering
// display (four light squares, 12 Hz) moves through the visual field while EEG is recorded;
// the SSVEP drives the colour of an RGB LED riding with the display (blue -> red), and a
// long exposure of the LED paints the visual field. Two rigs:
//   * on-screen sweep: the monitor itself is the moving display (no hardware)
//   * plotter: a GRBL 3D plotter carries a phone (flicker.html) in vertical passes at
//     5 mm/s while retreating 0.5 mm/s from 4 cm to 21 cm (side-view image of Fig. 4)

import { app, analysisParams, requireSource, makeSession } from '../context.js';
import { h, form, toast, sleep } from '../ui.js';
import { Stage, FlickerClock } from '../stage.js';
import { perfNow, windowEnding } from '../eeg-store.js';
import { spectrum, ssvepMetric, percentile } from '../dsp.js';
import { colorOf } from '../interp.js';
import { plotterPath, plannedPosition } from '../serial.js';
import { showResult } from '../results.js';

const LIGHTS = ['#ffffff', '#fff27a', '#8ff7ff', '#ffa3f2'];

// Live SSVEP -> [0,1] with running robust normalisation (for the LED colour).
function liveMeter(freq, winSec) {
  const hist = [];
  return () => {
    const st = app.store, p = analysisParams(freq, winSec);
    const w = windowEnding(st, st.lastTime, Math.round(winSec * st.fs), p.chans);
    if (!w) return null;
    const v = ssvepMetric(spectrum(w, st.fs, { psd: p.psd, segSec: Math.min(2, winSec) }), freq, p);
    hist.push(v); if (hist.length > 600) hist.shift();
    const lo = percentile(hist, 0.05), hi = percentile(hist, 0.95);
    return { v, n: hist.length > 8 ? Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1))) : 0.5 };
  };
}

export default {
  id: 'field', nav: 'Visual field (metaveillance)', title: 'Metaveillography of human vision', fig: 'Fig. 4, §II',
  blurb: 'Map where a person can see — "the sensing of sensing" applied to the eye. A flickering display is swept through the visual field; the SSVEP it evokes colours an LED that moves with it (blue = weak, red = strong), and the long exposure shows the visual field as a beam. Fixating straight ahead (covert attention) gives a narrow beam; tracking the display with the eyes (overt attention) gives a broad cone.',
  mount(el) {
    const v = {
      rigType: 'screen', freq: 12, attention: 'covert', pattern: 'horizontal', passes: 8, speed: 150, quad: 140, rasterRows: 6,
      winSec: 2, tint: false, distCm: 50, screenCm: Math.round(screen.width / 96 * 2.54 * 10) / 10,
      z0: 40, z1: 210, yLo: -50, yHi: 50, vy: 5, vz: 0.5, axisAway: 'X', axisVert: 'Z', virtual: true, lamp: false,
    };
    const info = h('div', { class: 'info' });
    const live = h('div');
    const results = h('div');
    const upd = () => {
      if (v.rigType === 'plotter') {
        const pl = plotterPath(v);
        info.innerHTML = `${pl.lines.length - 4} passes · ≈ <b>${(pl.duration / 60).toFixed(1)} min</b>. Stimulus: phone on the plotter showing <code>/flicker.html?f=${v.freq}&pattern=quad</code> (run <code>server.py --lan</code>), or the Arduino lamp.`;
      } else {
        const span = (v.pattern === 'vertical' ? screen.height : screen.width) - v.quad;
        const passes = v.pattern === 'raster' ? v.rasterRows : v.passes;
        info.innerHTML = `≈ <b>${(passes * span / v.speed / 60).toFixed(1)} min</b> · ${v.attention === 'covert' ? 'keep your eyes on the cross; attend to the moving squares out of the corner of your eye' : 'follow the moving squares with your eyes'}.`;
      }
    };
    const f = form([
      { key: 'rigType', label: 'Rig', type: 'select', options: [['screen', 'On-screen sweep (no hardware)'], ['plotter', '3D plotter + phone (Fig. 4)']] },
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'attention', label: 'Attention', type: 'select', options: [['covert', 'Covert: fixate ahead'], ['overt', 'Overt: track the display']] },
      { key: 'winSec', label: 'FFT window', type: 'number', min: 0.5, step: 0.5, unit: 's (paper: 10)' },
      { type: 'section', label: 'On-screen sweep', show: x => x.rigType === 'screen' },
      { key: 'pattern', label: 'Path', type: 'select', options: [['horizontal', 'Horizontal back-and-forth'], ['vertical', 'Vertical back-and-forth'], ['raster', '2-D raster (field map)']], show: x => x.rigType === 'screen' },
      { key: 'passes', label: 'Passes', type: 'number', min: 1, step: 1, show: x => x.rigType === 'screen' && x.pattern !== 'raster' },
      { key: 'rasterRows', label: 'Raster rows', type: 'number', min: 2, step: 1, show: x => x.rigType === 'screen' && x.pattern === 'raster' },
      { key: 'speed', label: 'Speed', type: 'number', min: 5, unit: 'px/s', show: x => x.rigType === 'screen' },
      { key: 'quad', label: 'Display size', type: 'number', min: 20, unit: 'px (2×2 squares)', show: x => x.rigType === 'screen' },
      { key: 'tint', label: 'Tint squares with SSVEP colour (paper)', type: 'checkbox', show: x => x.rigType === 'screen', help: 'Changes stimulus luminance with the response — a feedback loop, as in the paper.' },
      { key: 'distCm', label: 'Viewing distance', type: 'number', min: 1, unit: 'cm', show: x => x.rigType === 'screen' },
      { key: 'screenCm', label: 'Screen width', type: 'number', min: 1, unit: 'cm', show: x => x.rigType === 'screen' },
      { type: 'section', label: 'Plotter (mm, eye at origin)', show: x => x.rigType === 'plotter' },
      { key: 'z0', label: 'Start distance', type: 'number', unit: 'mm', show: x => x.rigType === 'plotter' },
      { key: 'z1', label: 'End distance', type: 'number', unit: 'mm', show: x => x.rigType === 'plotter' },
      { key: 'yLo', label: 'Lowest', type: 'number', unit: 'mm', show: x => x.rigType === 'plotter' },
      { key: 'yHi', label: 'Highest', type: 'number', unit: 'mm', show: x => x.rigType === 'plotter' },
      { key: 'vy', label: 'Vertical speed', type: 'number', min: 0.1, unit: 'mm/s', show: x => x.rigType === 'plotter' },
      { key: 'vz', label: 'Retreat speed', type: 'number', min: 0.01, step: 0.05, unit: 'mm/s', show: x => x.rigType === 'plotter' },
      { key: 'axisAway', label: 'Away-from-face axis', type: 'select', options: ['X', 'Y', 'Z'], show: x => x.rigType === 'plotter' },
      { key: 'axisVert', label: 'Vertical axis', type: 'select', options: ['X', 'Y', 'Z'], show: x => x.rigType === 'plotter' },
      { key: 'virtual', label: 'Virtual plotter (no GRBL)', type: 'checkbox', show: x => x.rigType === 'plotter' },
      { key: 'lamp', label: 'Flicker the Arduino lamp', type: 'checkbox', show: x => x.rigType === 'plotter' },
    ], v, upd);
    upd();
    const presets = h('div', { class: 'presets' },
      h('button', { onclick: () => { f.setAll({ rigType: 'plotter', freq: 12, winSec: 10, z0: 40, z1: 210, vy: 5, vz: 0.5 }); upd(); } }, 'Paper plotter settings'),
      h('button', { onclick: () => { f.setAll({ rigType: 'screen', attention: 'covert', pattern: 'horizontal' }); upd(); } }, 'Covert (narrow beam)'),
      h('button', { onclick: () => { f.setAll({ rigType: 'screen', attention: 'overt', pattern: 'horizontal' }); upd(); } }, 'Overt (broad cone)'));
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, presets, f, info,
        h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => (v.rigType === 'screen' ? runScreen() : runPlotter()).catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, h('h3', {}, 'Live'), live,
        h('p', { class: 'muted' }, 'Photograph it for real: connect the Arduino (Hardware), put the RGB LED on the moving display, and take a long exposure (a phone night mode, or the Long-exposure tool in "Camera metaveillance"). Red = strong SSVEP, blue = weak.'))),
    results);

    const ledOut = (n) => {
      const [r, g, b] = colorOf('bluered', n);
      if (app.io.connected) app.io.color(r, g, b);
      return `rgb(${r},${g},${b})`;
    };

    const runScreen = async () => {
      requireSource();
      const stage = await new Stage().open();
      try {
        const W = stage.w, H = stage.h, q = v.quad, m = q / 2 + 4;
        const fix = { x: W / 2, y: H / 2 };
        const path = { t: [], x: [], y: [] };
        const span = v.pattern === 'vertical' ? H - 2 * m : W - 2 * m;
        const passDur = span / v.speed;
        const rows = Math.max(2, Math.round(v.rasterRows));
        const nPass = v.pattern === 'raster' ? rows : Math.max(1, Math.round(v.passes));
        const posAt = t => {
          const k = Math.floor(t / passDur);
          if (k >= nPass) return null;
          const u = (t - k * passDur) / passDur, a = k % 2 ? 1 - u : u;
          if (v.pattern === 'vertical') return { x: W / 2, y: m + a * span };
          if (v.pattern === 'raster') return { x: m + a * span, y: m + k * (H - 2 * m) / (rows - 1) };
          return { x: m + a * span, y: H / 2 };
        };
        let pos = null;
        const sigma = v.attention === 'covert' ? 0.12 * W : 0.45 * W;
        app.probe(() => (pos ? { s: Math.exp(-((pos.x - fix.x) ** 2 + (pos.y - fix.y) ** 2) / (2 * sigma * sigma)), f: v.freq } : null));
        stage.message(`<div>${v.attention === 'covert' ? 'Keep your eyes on the cross' : 'Follow the flickering squares with your eyes'}</div><div class="small">SPACE to start · ESC to stop</div>`);
        stage.clear(); if (v.attention === 'covert') stage.cross(fix.x, fix.y);
        if ((await stage.waitKey([' ', 'Escape'])) === 'Escape') throw new Error('cancelled');
        stage.message('');
        const meter = liveMeter(v.freq, Math.min(v.winSec, 3));
        const clock = new FlickerClock(v.freq, stage.refresh);
        let tBegin = null, stop = false, col = '#fff', lastMeter = 0;
        const off = stage.onKey(e => { if (e.key === 'Escape') stop = true; });
        const lead = 1.5;
        await stage.loop((t, ts) => {
          if (stop) return false;
          if (tBegin == null) tBegin = t;
          const p = posAt(Math.max(0, t - tBegin - lead));
          if (!p) return false;
          pos = p;
          if (t - tBegin > lead) { path.t.push(t); path.x.push(p.x); path.y.push(p.y); }
          if (t - lastMeter > 0.25) { lastMeter = t; const r = meter(); if (r) col = ledOut(r.n); }
          const A = clock.advance(ts), c = stage.ctx;
          c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
          const s = q / 2;
          [[0, 0], [1, 0], [0, 1], [1, 1]].forEach(([i, j], k) => {
            c.fillStyle = A ? (v.tint ? col : LIGHTS[k]) : '#000';
            c.fillRect(p.x - s + i * s + 1, p.y - s + j * s + 1, s - 2, s - 2);
          });
          if (v.attention === 'covert') stage.cross(fix.x, fix.y);
        });
        off();
        stage.message('<div>Finishing…</div>');
        await sleep(v.winSec * 500 + 800);
        app.probe(null);
        await stage.close();
        const run = { W, H, freq: v.freq, winSec: v.winSec, path, fix, attention: v.attention, pattern: v.pattern, distCm: v.distCm, screenCm: v.screenCm };
        const session = makeSession('field', run, tBegin, perfNow(), { title: `Visual field — ${v.attention}, ${v.pattern}` });
        results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
      } catch (e) {
        app.probe(null); await stage.close();
        if (e.message !== 'cancelled') throw e;
      }
    };

    const runPlotter = async () => {
      requireSource();
      if (!v.virtual && !app.grbl.connected) throw new Error('Connect the GRBL plotter under Hardware, or tick "Virtual plotter".');
      const plan = plotterPath(v);
      const path = { t: [], z: [], y: [] };
      const cv = h('canvas', { width: 700, height: 360, style: 'width:100%;background:#000;border-radius:6px' });
      const bar = h('div', { class: 'bar' }, h('i'));
      const stopBtn = h('button', {}, 'Stop');
      live.innerHTML = ''; live.append(cv, bar, h('div', { class: 'btns' }, stopBtn));
      const c = cv.getContext('2d');
      const X = z => 30 + z / (v.z1 + 20) * 660, Y = y => 20 + (v.yHi - y) / (v.yHi - v.yLo) * 320;
      c.fillStyle = '#333'; c.beginPath(); c.arc(X(0) - 8, Y(0), 22, 0, 7); c.fill();
      let stop = false;
      stopBtn.onclick = async () => { stop = true; if (!v.virtual) await app.grbl.feedHold(); };
      const sig = z => (v.attention === 'covert' ? 8 + 0.1 * z : 20 + 0.45 * z);
      let cur = null;
      app.probe(() => (cur ? { s: Math.exp(-(cur.y ** 2) / (2 * sig(cur.z) ** 2)), f: v.freq } : null));
      if (v.lamp && app.io.connected) await app.io.lampFlicker(v.freq);
      const meter = liveMeter(v.freq, Math.min(v.winSec, 4));
      const tBegin = perfNow();
      let streaming = null;
      if (!v.virtual) {
        const map = { X: 'x', Y: 'y', Z: 'z' };
        streaming = app.grbl.stream(plan.lines, (i, n) => { bar.firstChild.style.width = (100 * i / n) + '%'; }).catch(e => toast(e.message));
        const poll = setInterval(() => {
          const p = app.grbl.pos; if (!p) return;
          cur = { z: p[map[v.axisAway]], y: p[map[v.axisVert]] };
        }, 100);
        streaming.finally(() => clearInterval(poll));
      }
      let lastLed = 0;
      while (!stop) {
        await sleep(100);
        const t = perfNow();
        if (v.virtual) {
          const p = plannedPosition(plan.moves, t - tBegin);
          cur = { z: p.z, y: p.y };
          bar.firstChild.style.width = Math.min(100, 100 * (t - tBegin) / plan.duration) + '%';
          if (p.done) break;
        } else if (app.grbl.state === 'Idle' && t - tBegin > 5 && path.t.length > 20) {
          await streaming; break;
        }
        if (!cur) continue;
        path.t.push(t); path.z.push(cur.z); path.y.push(cur.y);
        if (t - lastLed > 0.25) {
          lastLed = t;
          const r = meter();
          const col = r ? ledOut(r.n) : '#555';
          c.fillStyle = col; c.globalAlpha = 0.8; c.beginPath(); c.arc(X(cur.z), Y(cur.y), 3, 0, 7); c.fill(); c.globalAlpha = 1;
        }
      }
      app.probe(null);
      if (v.lamp && app.io.connected) await app.io.lampFlicker(0);
      await sleep(v.winSec * 500 + 800);
      const run = { freq: v.freq, winSec: v.winSec, plan: { z0: v.z0, z1: v.z1, yLo: v.yLo, yHi: v.yHi, vy: v.vy, vz: v.vz }, path, virtual: v.virtual, attention: v.attention, t0: tBegin, t1: perfNow() };
      const session = makeSession('plotter', run, tBegin, perfNow(), { title: `Metaveillograph — plotter${v.virtual ? ' (virtual)' : ''}` });
      results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
    };
  },
};
