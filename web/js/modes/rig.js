// Fig. 10 / 11: imaging the real world. A flickering light is slid along a horizontal pipe
// between two tripods, one scan line at a time, timed by a metronome; the participant
// watches the lit object. This page is the operator's console: metronome, cues, where the
// light should be right now, the Arduino lamp, and the reconstruction.

import { app, analysisParams, requireSource, makeSession, beep, say, canvasToDataURL } from '../context.js';
import { h, form, toast, sleep, readFile, loadImage } from '../ui.js';
import { perfNow } from '../eeg-store.js';
import { emptyScanGrid, scanRow, renderScan } from '../recon.js';
import { toImageData } from '../interp.js';
import { showResult } from '../results.js';

export default {
  id: 'rig', nav: 'Real-world rig', title: 'Imaging the real world — scan-line rig', fig: 'Fig. 10, 11',
  blurb: 'Two tripods ~6 ft apart hold a horizontal pipe at the current scan-line height; a flickering light slides along it at a steady, metronome-timed rate and lights the subject. The participant simply looks at the subject. Each line becomes one row of the image, reconstructed exactly like the on-screen raster.',
  mount(el) {
    const v = { lines: 20, bpm: 60, beatsPerLine: 16, leadBeats: 4, rest: 'key', restSec: 6, direction: 'ltr', freq: 15, footprint: 0.08, aspect: 0.6, lamp: true, speech: true };
    let reference = null, refLum = null;
    const info = h('div', { class: 'info' });
    const results = h('div');
    const op = h('div', { class: 'rig-op' });
    const upd = () => {
      const lineSec = v.beatsPerLine * 60 / v.bpm;
      info.innerHTML = `Line: <b>${lineSec.toFixed(1)} s</b> · window ${(v.footprint * lineSec).toFixed(2)} s · total ≈ <b>${(v.lines * (lineSec + v.leadBeats * 60 / v.bpm + 8) / 60).toFixed(0)} min</b>. Pipe heights: ${v.lines} evenly spaced lines from top to bottom of the subject.`;
    };
    const f = form([
      { key: 'lines', label: 'Scan lines', type: 'number', min: 2, step: 1 },
      { key: 'bpm', label: 'Metronome', type: 'number', min: 20, max: 240, unit: 'BPM' },
      { key: 'beatsPerLine', label: 'Beats per line', type: 'number', min: 2, step: 1 },
      { key: 'leadBeats', label: 'Count-in beats', type: 'number', min: 0, step: 1 },
      { key: 'direction', label: 'Direction', type: 'select', options: [['ltr', 'Always left → right'], ['snake', 'Alternate (boustrophedon)']] },
      { key: 'rest', label: 'Between lines', type: 'select', options: [['key', 'SPACE when pipe is moved'], ['auto', 'Automatic']] },
      { key: 'restSec', label: 'Automatic pause', type: 'number', min: 0, unit: 's', show: x => x.rest === 'auto' },
      { key: 'freq', label: 'Light flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'lamp', label: 'Drive Arduino lamp (pin 6)', type: 'checkbox' },
      { key: 'footprint', label: 'Light footprint', type: 'number', min: 0.01, max: 0.5, step: 0.01, unit: '× width', help: 'Width of the lit patch as a fraction of the scan width; sets the analysis window.' },
      { key: 'aspect', label: 'Aspect (width / height)', type: 'number', min: 0.1, step: 0.05 },
      { key: 'speech', label: 'Spoken cues', type: 'checkbox' },
    ], v, upd);
    upd();
    const refIn = h('input', { type: 'file', accept: 'image/*' });
    refIn.onchange = async () => {
      if (!refIn.files[0]) return;
      reference = await loadImage(await readFile(refIn.files[0], 'dataurl'));
      const c = document.createElement('canvas'); c.width = 200; c.height = Math.round(200 / (reference.width / reference.height));
      const x = c.getContext('2d'); x.drawImage(reference, 0, 0, c.width, c.height);
      const d = x.getImageData(0, 0, c.width, c.height).data;
      refLum = { w: c.width, h: c.height, lum: Float32Array.from({ length: c.width * c.height }, (_, i) => (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 765) };
      f.set('aspect', +(reference.width / reference.height).toFixed(3)); upd();
    };
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, f, h('label', { class: 'form-row' }, h('span', { class: 'lbl' }, 'Reference photo'), refIn), info,
        h('p', { class: 'muted small' }, 'Optional reference photo of the subject (Fig. 11 left) — shown beside the result, and used by the simulator as the scene.'),
        h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, h('h3', {}, 'Operator console'), op)),
    results);

    const run = async () => {
      requireSource();
      const beat = 60 / v.bpm, lineSec = v.beatsPerLine * beat, N = Math.round(v.lines);
      const W = 1000 * v.aspect, H = 1000;
      const rowY = Array.from({ length: N }, (_, r) => H * (r + 0.5) / N);
      const run = { W, H, rows: N, rowY, cursor: v.footprint * W, speed: W / lineSec, freq: v.freq, winSec: v.footprint * lineSec, lines: [], gain: 1, rig: { ...v } };
      const p = analysisParams(v.freq, run.winSec);
      const grid = emptyScanGrid(run, { stepPx: run.cursor / 4 });
      const lineEl = h('div', { class: 'rig-line' }, '');
      const cue = h('div', {}, '');
      const marker = h('i');
      const track = h('div', { class: 'rig-track' }, marker);
      const prev = h('canvas', { style: 'width:220px;background:#000;border-radius:4px' });
      const stopBtn = h('button', {}, 'Stop'), goBtn = h('button', { class: 'primary' }, 'Next line (SPACE)'), redoBtn = h('button', {}, 'Redo previous (R)');
      op.innerHTML = ''; op.append(lineEl, cue, track, h('div', { class: 'btns' }, goBtn, redoBtn, stopBtn), prev);
      let pos = null;
      app.probe(() => {
        if (!pos) return null;
        if (!refLum) return { s: 0.6, f: v.freq };
        const x = Math.floor(pos.x / W * refLum.w), y = Math.floor(pos.y / H * refLum.h), r = 3;
        let s = 0, n = 0;
        for (let yy = y - r; yy <= y + r; yy++) for (let xx = x - r; xx <= x + r; xx++) if (xx >= 0 && yy >= 0 && xx < refLum.w && yy < refLum.h) { s += refLum.lum[yy * refLum.w + xx]; n++; }
        return { s: n ? s / n : 0, f: v.freq };
      });
      let action = null;
      const onKey = e => { if (e.key === ' ') { e.preventDefault(); action = 'go'; } else if (e.key === 'r' || e.key === 'R') action = 'redo'; else if (e.key === 'Escape') action = 'stop'; };
      window.addEventListener('keydown', onKey);
      goBtn.onclick = () => (action = 'go'); redoBtn.onclick = () => (action = 'redo'); stopBtn.onclick = () => (action = 'stop');
      const waitAction = async (autoSec) => {
        action = null; const t = perfNow();
        while (!action) { await sleep(50); if (autoSec != null && perfNow() - t > autoSec) return 'go'; }
        return action;
      };
      const tStart = perfNow();
      try {
        let r = 0;
        while (r < N) {
          const ltr = v.direction === 'ltr' || r % 2 === 0;
          lineEl.textContent = `Line ${r + 1} / ${N}`;
          cue.innerHTML = `Set the pipe to height <b>${((r + 0.5) / N * 100).toFixed(0)} %</b> from the top, light at the <b>${ltr ? 'left' : 'right'}</b>. ${v.rest === 'key' ? 'SPACE when ready.' : ''}`;
          if (v.speech) say(`Line ${r + 1}. Pipe to ${((r + 0.5) / N * 100).toFixed(0)} percent. Light ${ltr ? 'left' : 'right'}.`);
          if (app.io.connected && v.lamp) await app.io.lampFlicker(0);
          const a = await waitAction(v.rest === 'auto' ? v.restSec : null);
          if (a === 'stop') break;
          if (a === 'redo' && r > 0) { r--; continue; }
          if (app.io.connected && v.lamp) await app.io.lampFlicker(v.freq);
          // count-in
          for (let b = v.leadBeats; b > 0; b--) { cue.innerHTML = `Starting in <b>${b}</b>…`; beep(b === 1 ? 1200 : 800, 0.05); await sleep(beat * 1000); if (action === 'stop') break; }
          if (action === 'stop') break;
          const t0 = perfNow(), t1 = t0 + lineSec;
          let nextBeat = t0;
          cue.innerHTML = `Slide the light <b>${ltr ? 'left → right' : 'right → left'}</b>, one step per beat`;
          while (perfNow() < t1 + beat * 0.5 && action !== 'stop') {
            const t = perfNow(), a2 = Math.min(1, Math.max(0, (t - t0) / lineSec)), u = ltr ? a2 : 1 - a2;
            marker.style.left = `${u * 100}%`;
            pos = { x: u * W, y: rowY[r] };
            if (t >= nextBeat) { beep(nextBeat === t0 || t >= t1 ? 1200 : 700, 0.04, 0.2); nextBeat += beat; }
            await sleep(15);
          }
          pos = null;
          if (action === 'stop') break;
          run.lines = run.lines.filter(l => l.row !== r);
          const L = { row: r, y: rowY[r], t0, t1, x0: ltr ? 0 : W, x1: ltr ? W : 0 };
          run.lines.push(L);
          await sleep(run.winSec * 500 + 600);
          scanRow(app.store, run, L, p, grid);
          const res = renderScan(app.store, run, { ...p, outW: 220, combine: false, reject: true }, grid);
          prev.width = res.w; prev.height = res.h; prev.getContext('2d').putImageData(toImageData(res.img, res.w, res.h, 'gray'), 0, 0);
          r++;
        }
      } finally {
        window.removeEventListener('keydown', onKey);
        app.probe(null);
        if (app.io.connected && v.lamp) await app.io.lampFlicker(0);
      }
      if (!run.lines.length) return toast('No scan lines recorded');
      const session = makeSession('rig', run, tStart, perfNow(), { reference: reference ? canvasToDataURL(reference) : null, title: `Real-world rig — ${run.lines.length} lines` });
      results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
    };
  },
};
