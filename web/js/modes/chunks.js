// Fig. 12: wearable photography in public. Modified LC shutter glasses flicker the whole
// world at 15 Hz; most of the view is occluded so the photographer sees only the spot in
// front of their gaze. The scene is divided into rectangular chunks and the photographer
// fixates each one for 8 s; each chunk's SSVEP becomes one pixel.

import { app, analysisParams, requireSource, makeSession, beep, say, imagePicker, canvasToDataURL, applyPreferred } from '../context.js';
import { h, form, toast, sleep } from '../ui.js';
import { Stage, FlickerClock, flickerFrames, meanLum, hexToRgb } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { showResult } from '../results.js';

export default {
  id: 'chunks', nav: 'Shutter-glasses chunks', title: 'Wearable eye camera — shutter glasses, chunk by chunk', fig: 'Fig. 12',
  blurb: 'Self-contained, for places where you cannot change the environment (the paper photographed a painting in a gallery). The shutter glasses flicker the world; you fixate each chunk of the scene in turn, cued by voice/beeps; one pixel per chunk. With no glasses you can emulate it on screen: only the current chunk of the image is visible and flickering.',
  mount(el) {
    let img = null;
    const v = applyPreferred({ cols: 9, rows: 7, dwell: 8, gap: 1.5, freq: 15, stim: 'screen', speech: true, colA: '#ffffff', colB: '#000000' });
    const info = h('div', { class: 'info' });
    const grid = h('div', { class: 'chunk-grid' });
    const results = h('div');
    const upd = () => { info.innerHTML = `${v.cols * v.rows} chunks × ${(v.dwell + v.gap).toFixed(1)} s ≈ <b>${(v.cols * v.rows * (v.dwell + v.gap) / 60).toFixed(1)} min</b>`; };
    const picker = imagePicker(i => { img = i; }, 'Test pattern');
    const f = form([
      { key: 'stim', label: 'Flicker source', type: 'select', options: [['glasses', 'LC shutter glasses (Arduino S)'], ['lamp', 'Flickering lamp (Arduino F)'], ['screen', 'On-screen emulation']] },
      { key: 'cols', label: 'Columns', type: 'number', min: 1, step: 1 },
      { key: 'rows', label: 'Rows', type: 'number', min: 1, step: 1 },
      { key: 'dwell', label: 'Fixation per chunk', type: 'number', min: 1, step: 0.5, unit: 's (paper: 8)' },
      { key: 'gap', label: 'Move time', type: 'number', min: 0, step: 0.5, unit: 's' },
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'colA', label: 'Colour A (screen)', type: 'color', show: x => x.stim === 'screen' },
      { key: 'colB', label: 'Colour B (screen)', type: 'color', show: x => x.stim === 'screen' },
      { key: 'speech', label: 'Spoken cues', type: 'checkbox' },
    ], v, upd);
    upd();
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, f, picker, h('p', { class: 'muted small' }, 'The image is the scene for on-screen emulation (and the simulator), and the reference shown beside the result.'), info,
        h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, h('h3', {}, 'Progress'), grid,
        h('p', { class: 'muted' }, 'Order: row by row, left to right. Glasses: the Arduino drives both LC lenses in phase across pins 2/3 and 4/5 (see firmware). Occlude all but a small central hole with tape to emulate the pointer.'))),
    results);

    const run = async () => {
      requireSource();
      const external = v.stim !== 'screen';
      if (external && !app.io.connected) throw new Error('Connect the EyeCam Arduino under Hardware (or choose on-screen emulation).');
      const C = Math.round(v.cols), R = Math.round(v.rows);
      grid.style.gridTemplateColumns = `repeat(${C}, 1fr)`;
      grid.innerHTML = ''; const cellsEl = [];
      for (let i = 0; i < C * R; i++) { const d = h('div'); cellsEl.push(d); grid.append(d); }
      const stage = external ? null : await new Stage().open();
      let frames = null, box = null, cur = null;
      if (stage) {
        const s = Math.min((stage.w - 40) / img.width, (stage.h - 40) / img.height);
        box = { W: Math.round(img.width * s), H: Math.round(img.height * s) };
        box.ox = Math.round((stage.w - box.W) / 2); box.oy = Math.round((stage.h - box.H) / 2);
        frames = flickerFrames(img, box.W, box.H, hexToRgb(v.colA), hexToRgb(v.colB));
      } else {
        const c = document.createElement('canvas'); c.width = 320; c.height = Math.round(320 * img.height / img.width);
        frames = flickerFrames(img, c.width, c.height, [255, 255, 255], [0, 0, 0]);
        box = { W: c.width, H: c.height };
      }
      const cw = box.W / C, ch = box.H / R;
      app.probe(() => (cur ? { s: 0.15 + 0.85 * meanLum(frames, cur.c * cw, cur.r * ch, Math.max(cw, ch)), f: v.freq } : null));
      const cells = [];
      let stop = false;
      const onKey = e => { if (e.key === 'Escape') stop = true; };
      window.addEventListener('keydown', onKey);
      const tStart = perfNow();
      try {
        if (stage) {
          stage.message('<div>Look at each visible patch until it disappears</div><div class="small">SPACE to start · ESC to stop</div>');
          if ((await stage.waitKey([' ', 'Escape'])) === 'Escape') stop = true;
          stage.message('');
        }
        const clock = stage ? new FlickerClock(v.freq, stage.refresh) : null;
        for (let r = 0; r < R && !stop; r++) for (let c = 0; c < C && !stop; c++) {
          const idx = r * C + c;
          cellsEl.forEach((d, i) => d.classList.toggle('cur', i === idx));
          if (v.speech) say(`Row ${r + 1}, column ${c + 1}`); else beep(600, 0.05);
          if (external) {
            if (v.stim === 'glasses') await app.io.shutter(0); else await app.io.lampFlicker(0);
            await sleep(v.gap * 1000);
            if (v.stim === 'glasses') await app.io.shutter(v.freq); else await app.io.lampFlicker(v.freq);
            cur = { c, r };
            const t0 = perfNow();
            await sleep(v.dwell * 1000);
            cells.push({ c, r, t0, t1: perfNow() });
          } else {
            const pos = { x: box.ox + (c + 0.5) * cw, y: box.oy + (r + 0.5) * ch };
            // move: show only a dot where the next chunk is
            const tm = perfNow();
            await stage.loop(() => { if (stop) return false; stage.clear(); stage.dot(pos.x, pos.y, 4, '#fff'); return perfNow() - tm < v.gap; });
            cur = { c, r };
            let t0 = null;
            await stage.loop((t, ts) => {
              if (stop) return false;
              if (t0 == null) t0 = t;
              if (t - t0 > v.dwell) return false;
              const A = clock.advance(ts), cx = stage.ctx;
              stage.clear();
              cx.drawImage(A ? frames.A : frames.B, c * cw, r * ch, cw, ch, box.ox + c * cw, box.oy + r * ch, cw, ch);
              stage.dot(pos.x, pos.y, 2, '#f33');
            });
            if (!stop) cells.push({ c, r, t0, t1: t0 + v.dwell });
          }
          cur = null;
          cellsEl[idx].classList.add('done');
          beep(900, 0.04, 0.12);
        }
      } finally {
        window.removeEventListener('keydown', onKey);
        if (external) { await app.io.shutter(0); await app.io.lampFlicker(0); }
        await sleep(800);
        app.probe(null);
        if (stage) await stage.close();
      }
      if (!cells.length) return toast('No chunks recorded');
      const run = { cols: C, rows: R, freq: v.freq, dwell: v.dwell, cells, stim: v.stim };
      const session = makeSession('chunks', run, tStart, perfNow(), { reference: canvasToDataURL(img), title: `Shutter-glasses camera — ${C}×${R}` });
      results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
    };
  },
};
