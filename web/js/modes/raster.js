// §III, Fig. 1 / 7 / 8 / 9: a flickering square cursor slides across the image, one scan
// line at a time; the SSVEP measured while it passes each point becomes that pixel.

import { app, analysisParams, requireSource, makeSession, imagePicker, canvasToDataURL, rotateCCW, beep, applyPreferred } from '../context.js';
import { h, form, toast, sleep } from '../ui.js';
import { Stage, FlickerClock, flickerFrames, meanLum, hexToRgb, measureRefresh } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { renderScan } from '../recon.js';
import { makeCycles } from '../ssvep.js';
import { toImageData } from '../interp.js';
import { showResult } from '../results.js';

const PRESETS = {
  'Letter N (≈8 min)': { image: 'N (single letter)', rows: 10, cursor: 100, speed: 25, freq: 15, colA: '#ffffff', colB: '#000000', levels: 'binary', rotate: false, gain: 1 },
  'Paper: NO CAMERAS': { image: 'NO CAMERAS (Fig. 8)', rows: 48, cursor: 100, speed: 15, freq: 15, colA: '#ffff00', colB: '#0000ff', levels: 'binary', rotate: false, gain: 1 },
  'Paper: face (upload photo)': { rows: 48, cursor: 100, speed: 15, freq: 15, colA: '#ffffff', colB: '#000000', levels: 'gray', rotate: true, gain: 2, upload: true },
  'Quick demo (≈6 min)': { image: 'NO CAMERAS (Fig. 8)', rows: 16, cursor: 100, speed: 45, freq: 15, colA: '#ffff00', colB: '#0000ff', levels: 'binary', rotate: false, gain: 1 },
};

export default {
  id: 'raster', nav: 'Eye camera (raster)', title: 'The eye as a camera — raster cursor', fig: 'Fig. 1, 7, 8, 9',
  blurb: 'A 100 px square cursor slides left→right across the image. Where the square covers non-black parts of the image it flashes (yellow/blue, 15 Hz); the subject just follows the white dot at its centre. For every position the PSD over the time the cursor takes to pass a point (1700 samples = 6.67 s at the paper\'s speed) gives (P15+P30)/P(14–50 Hz); 48 overlapping scan lines are merged with eq. (1) and interpolated into the image.',
  mount(el) {
    let img = null;
    const v = applyPreferred({ ...PRESETS['Letter N (≈8 min)'], threshold: 0.12, invert: false, maxW: 1600, rest: 'key', restSec: 4, outline: true, background: 'black', preview: true, passes: 1, lineRef: false, refSec: 3 });
    const info = h('div', { class: 'info' });
    const results = h('div');
    let refresh = null;
    measureRefresh().then(r => { refresh = r; estimate(); });
    const estimate = () => {
      if (!img) return;
      const iw = v.rotate ? img.height : img.width, ih = v.rotate ? img.width : img.height;
      const s = Math.min((screen.width - 40) / iw, (screen.height - 40) / ih, v.maxW / iw);
      const W = iw * s, win = v.cursor / v.speed, line = W / v.speed + win + 1.3 + (v.lineRef ? 2 * v.refSec : 0);
      info.innerHTML = `On this screen: image ≈ ${Math.round(W)}×${Math.round(ih * s)} px · each scan line ≈ <b>${line.toFixed(0)} s</b> · analysis window ${win.toFixed(2)} s (${Math.round(win * 256)} samples) · total ≈ <b>${(v.rows * v.passes * (line + 3) / 60).toFixed(0)} min</b> plus breaks.`;
      if (refresh && !FlickerClock.quality(v.freq, refresh).exact) {
        const exact = refresh / (2 * Math.max(1, Math.round(refresh / (2 * v.freq))));
        info.append(h('div', { class: 'warn', style: 'margin-top:6px' },
          `Your display runs at ${refresh.toFixed(1)} Hz, so ${v.freq} Hz can only be approximated (uneven frames blur the SSVEP). Nearest exact: ${exact.toFixed(2)} Hz. `,
          h('button', { onclick: () => { f.set('freq', +exact.toFixed(3)); estimate(); } }, `Use ${exact.toFixed(2)} Hz`)));
      }
    };
    const picker = imagePicker(i => { img = i; estimate(); }, v.image);
    const f = form([
      { type: 'section', label: 'Stimulus' },
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'colA', label: 'Colour A', type: 'color' }, { key: 'colB', label: 'Colour B', type: 'color' },
      { key: 'levels', label: 'Levels', type: 'select', options: [['binary', 'Binary: flash where non-black (paper)'], ['gray', 'Grayscale: flicker depth ∝ brightness']] },
      { key: 'threshold', label: 'Black threshold', type: 'number', min: 0, max: 1, step: 0.02, show: x => x.levels === 'binary' },
      { key: 'invert', label: 'Invert image', type: 'checkbox' },
      { key: 'cursor', label: 'Cursor size', type: 'number', min: 10, unit: 'px' },
      { key: 'speed', label: 'Cursor speed', type: 'number', min: 1, unit: 'px/s' },
      { key: 'rows', label: 'Scan lines', type: 'number', min: 1, step: 1 },
      { key: 'passes', label: 'Passes', type: 'number', min: 1, max: 4, step: 1, help: 'Repeat the whole image (alternating direction). Median-combined; enables split-half reliability.' },
      { key: 'lineRef', label: 'Reference patches before each line', type: 'checkbox', help: 'Full flicker then none at the start of every line, to correct slow SSVEP drift.' },
      { key: 'refSec', label: 'Reference duration', type: 'number', min: 1, step: 0.5, unit: 's each', show: x => x.lineRef },
      { key: 'rotate', label: 'Display rotated 90°, rotate back after (paper, faces)', type: 'checkbox' },
      { key: 'maxW', label: 'Max image width', type: 'number', min: 100, unit: 'px' },
      { key: 'background', label: 'Rest of image', type: 'select', options: [['black', 'Black (only the cursor is visible)'], ['dim', 'Dim static preview']] },
      { key: 'outline', label: 'Faint cursor outline', type: 'checkbox' },
      { type: 'section', label: 'Session' },
      { key: 'rest', label: 'Between lines', type: 'select', options: [['key', 'Wait for SPACE (avoid fatigue)'], ['auto', 'Automatic pause']] },
      { key: 'restSec', label: 'Automatic pause', type: 'number', min: 0, unit: 's', show: x => x.rest === 'auto' },
      { key: 'preview', label: 'Show partial image in breaks', type: 'checkbox' },
      { key: 'gain', label: 'Output gain', type: 'number', min: 0.1, step: 0.1, help: 'paper: face ×2' },
    ], v, estimate);
    const startBtn = () => h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start imaging');
    const bottomStart = startBtn();
    const presets = h('div', { class: 'presets' }, Object.entries(PRESETS).map(([k, p]) => h('button', {
      onclick: () => {
        f.setAll(Object.fromEntries(Object.entries(p).filter(([key]) => key in v)));
        if (p.image) picker.select(p.image);
        estimate();
        toast(p.upload ? 'Face settings loaded: choose "Upload image…" under Subject matter, then press Start imaging' : `${k} settings loaded — press Start imaging`);
        bottomStart.scrollIntoView({ behavior: 'smooth', block: 'center' });
        bottomStart.animate([{ boxShadow: '0 0 0 6px rgba(251,146,60,.8)' }, { boxShadow: '0 0 0 0 rgba(251,146,60,0)' }], { duration: 1200, iterations: 2 });
      },
    }, k)));
    const banner = h('div', { class: 'info', style: 'margin-bottom:10px' });
    const cal = app.calib;
    if (!cal) banner.append(h('b', {}, 'Not calibrated. '), 'Run ', h('a', { href: '#calibrate' }, 'Calibration'), ' first — it checks that your SSVEP is measurable and plans the scan (your last recording had almost no SSVEP).');
    else {
      const col = { go: 'var(--ok)', marginal: 'var(--warn)', nogo: 'var(--bad)' }[cal.verdict];
      banner.append(h('b', { style: `color:${col}` }, `Calibration: ${cal.verdict.toUpperCase()}`), ` (d′ ${cal.d1?.toFixed(2)}/s, ${cal.created.slice(0, 16).replace('T', ' ')}). `);
      if (cal.verdict !== 'nogo') banner.append(h('button', {
        onclick: () => {
          picker.select('N (single letter)');
          f.setAll({ freq: cal.freq, speed: cal.plan.speed, cursor: cal.plan.cursor, rows: cal.plan.rows, passes: cal.plan.passes, colA: '#ffffff', colB: '#000000', levels: 'binary' });
          estimate(); toast('Calibrated scan plan loaded — press Start imaging');
        },
      }, 'Use calibrated scan plan'));
      else banner.append('Imaging is not recommended until contact improves — recalibrate.');
    }
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, banner, h('p', { class: 'muted small', style: 'margin:0 0 6px' }, 'Presets load settings; Start imaging runs.'), presets,
        h('div', { class: 'btns', style: 'margin:0 0 10px' }, startBtn()), picker, f, info,
        h('div', { class: 'btns' }, bottomStart)),
      h('div', { class: 'card' }, h('h3', {}, 'How to run it'), h('ol', {},
        h('li', {}, 'Seat the subject ~50 cm from the screen in a dim room; check contact in Signal monitor.'),
        h('li', {}, 'Press Start: the screen goes fullscreen. Each scan line waits for SPACE, so the subject can rest between lines.'),
        h('li', {}, 'The subject follows the white dot at the centre of the moving square. Blink between lines if possible.'),
        h('li', {}, 'R during a break redoes the previous line; ESC finishes early (the image is built from the lines done).'),
        h('li', {}, 'After the run, change the analysis parameters and re-render; save the session to reprocess later.')),
        h('p', { class: 'muted' }, 'Paper settings: 100 px cursor at 15 px/s → 106 s per 1600 px scan line, 48 lines ≈ 85 min. The quick preset trades resolution for time.'))),
    results);

    const run = async () => {
      requireSource();
      if (!img) throw new Error('No image selected');
      const stage = await new Stage().open();
      try {
        const src = v.rotate ? rotateCCW(img) : img;
        const margin = 20;
        const s = Math.min((stage.w - 2 * margin) / src.width, (stage.h - 2 * margin) / src.height, v.maxW / src.width);
        const W = Math.round(src.width * s), H = Math.round(src.height * s);
        const ox = Math.round((stage.w - W) / 2), oy = Math.round((stage.h - H) / 2);
        const frames = flickerFrames(src, W, H, hexToRgb(v.colA), hexToRgb(v.colB), { binary: v.levels === 'binary', threshold: v.threshold, invert: v.invert });
        const c = Math.min(v.cursor, H), rows = Math.max(1, Math.round(v.rows)), passes = Math.max(1, Math.round(v.passes));
        const rowY = Array.from({ length: rows }, (_, r) => (rows === 1 ? H / 2 : c / 2 + r * (H - c) / (rows - 1)));
        const winSec = c / v.speed;
        // ground truth for validity scoring: displayed luminance, downsampled
        const tw = Math.min(160, W), th = Math.max(1, Math.round(tw * H / W)), tl = new Float32Array(tw * th);
        for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) tl[y * tw + x] = frames.lum[Math.floor(y * H / th) * W + Math.floor(x * W / tw)];
        const run = {
          W, H, rows, rowY, cursor: c, speed: v.speed, freq: v.freq, winSec, lines: [], passes, lineRef: v.lineRef,
          rotate: v.rotate, gain: v.gain, levels: v.levels, colors: [v.colA, v.colB], refresh: stage.refresh,
          flicker: { t: [], c: [] }, truth: { w: tw, h: th, lum: Array.from(tl) },
        };
        const clock = new FlickerClock(v.freq, stage.refresh);
        const q = FlickerClock.quality(v.freq, stage.refresh);
        let pos = null, refFlash = false;
        app.probe(() => (refFlash ? { s: 1, f: v.freq } : pos ? { s: meanLum(frames, pos.x - c / 2, pos.y - c / 2, c), f: v.freq } : null));
        const p = analysisParams(v.freq, winSec, app.store, { cycles: makeCycles(run, v.freq) });
        const tStart = perfNow();

        const draw = (x, y, A) => {
          const ctx = stage.ctx;
          ctx.fillStyle = '#000'; ctx.fillRect(0, 0, stage.w, stage.h);
          if (v.background === 'dim') { ctx.globalAlpha = 0.12; ctx.drawImage(frames.A, ox, oy); ctx.globalAlpha = 1; }
          const sx = Math.max(0, x - c / 2), sy = Math.max(0, y - c / 2), ex = Math.min(W, x + c / 2), ey = Math.min(H, y + c / 2);
          if (ex > sx && ey > sy) ctx.drawImage(A ? frames.A : frames.B, sx, sy, ex - sx, ey - sy, ox + sx, oy + sy, ex - sx, ey - sy);
          if (v.outline) { ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.strokeRect(ox + x - c / 2 + 0.5, oy + y - c / 2 + 0.5, c - 1, c - 1); }
          stage.dot(ox + x, oy + y, 3, '#fff');
        };

        // One scan line. Odd passes run right -> left (cancels latency/hysteresis bias). With
        // line references, the line starts with refSec of full flicker then refSec of none.
        const sweep = async (r, pass) => {
          const y = rowY[r], lead = winSec / 2 + 1, tail = winSec / 2 + 0.4, dur = W / v.speed;
          const dir = pass % 2 ? -1 : 1, xs = dir > 0 ? 0 : W, xe = dir > 0 ? W : 0;
          let t0 = null, t1 = null, aborted = false, refOn = null, refOff = null;
          const off = stage.onKey(e => { if (e.key === 'Escape') aborted = true; });
          if (v.lineRef) {
            let a = null;
            await stage.loop((t, ts) => { // full-flicker reference
              if (aborted) return false;
              if (a == null) a = t;
              if (t - a >= v.refSec) return false;
              refFlash = true;
              const A = clock.advance(ts); clock.log(run.flicker, t);
              const ctx = stage.ctx; stage.clear();
              ctx.fillStyle = A ? v.colA : v.colB; ctx.fillRect(ox + xs - c / 2, oy + y - c / 2, c, c);
              stage.dot(ox + xs, oy + y, 3, '#fff');
            });
            refOn = [a, a + v.refSec]; refFlash = false; a = null;
            await stage.loop(t => { // no-flicker reference
              if (aborted) return false;
              if (a == null) a = t;
              if (t - a >= v.refSec) return false;
              stage.clear(); stage.dot(ox + xs, oy + y, 3, '#fff');
            });
            refOff = [a, a + v.refSec];
          }
          await stage.loop((t, ts) => {
            if (aborted) return false;
            if (t0 == null) { t0 = t + lead; t1 = t0 + dur; }
            if (t > t1 + tail) return false;
            const x = t < t0 ? xs : t < t1 ? xs + dir * (t - t0) * v.speed : xe;
            pos = { x, y };
            const A = clock.advance(ts);
            clock.log(run.flicker, t);
            draw(x, y, A);
          });
          off(); pos = null; refFlash = false; stage.clear();
          if (aborted) return false;
          run.lines = run.lines.filter(l => !(l.row === r && (l.pass ?? 0) === pass));
          run.lines.push({ row: r, pass, y, t0, t1, x0: xs, x1: xe, refOn, refOff });
          return true;
        };

        const preview = () => {
          if (!v.preview || !run.lines.length) return '';
          const res = renderScan(app.store, run, { ...p, outW: 360, gain: v.gain, combine: true, reject: true, lineRef: v.lineRef });
          const cv = document.createElement('canvas'); cv.width = res.w; cv.height = res.h;
          cv.getContext('2d').putImageData(toImageData(res.img, res.w, res.h, 'gray'), 0, 0);
          return `<img src="${cv.toDataURL()}">`;
        };

        let done = false;
        for (let pass = 0; pass < passes && !done; pass++) {
          let r = 0;
          while (r < rows) {
            stage.clear();
            const head = `<div>${passes > 1 ? `Pass ${pass + 1} of ${passes} · ` : ''}Scan line ${r + 1} of ${rows}${pass % 2 ? ' (right → left)' : ''}</div><div class="small">Follow the white dot in the flashing square.<br>SPACE start line · R redo previous line · ESC finish${q.exact ? '' : `<br>(display ${stage.refresh.toFixed(1)} Hz: ${v.freq} Hz is time-dithered)`}</div>`;
            stage.message(head);
            if (run.lines.length) {
              await sleep(700); // let the tail of the previous line's EEG arrive
              stage.message(head + preview());
            }
            const key = v.rest === 'key'
              ? await stage.waitKey([' ', 'r', 'R', 'Escape'])
              : await Promise.race([sleep(v.restSec * 1000).then(() => ' '), stage.waitKey(['r', 'R', 'Escape'])]);
            if (key === 'Escape') { done = true; break; }
            if ((key === 'r' || key === 'R') && r > 0) { r--; continue; }
            stage.message('');
            beep(660, 0.05, 0.1);
            if (!(await sweep(r, pass))) { done = true; break; }
            r++;
          }
        }
        stage.message('<div>Finishing…</div>');
        await sleep(1200);
        app.probe(null);
        await stage.close();
        if (!run.lines.length) { toast('No complete scan lines — nothing to reconstruct'); return; }
        const session = makeSession('raster', run, tStart, perfNow(), { reference: canvasToDataURL(img), title: `Eye camera — ${run.lines.length} lines${passes > 1 ? `, ${passes} passes` : ''}` });
        results.innerHTML = '';
        showResult(results, session);
        results.scrollIntoView({ behavior: 'smooth' });
      } catch (e) {
        app.probe(null); await stage.close(); throw e;
      }
    };
  },
};
