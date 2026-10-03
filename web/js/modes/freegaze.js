// §III first method, Fig. 6/7: the whole image flickers; the observer rasters their eyes
// across it. Eye position (tracker, or a guiding cursor) is paired with the SSVEP computed
// over a sliding window; the scattered values are Delaunay-interpolated into an image.

import { app, analysisParams, requireSource, makeSession, imagePicker, canvasToDataURL } from '../context.js';
import { h, form, toast, sleep } from '../ui.js';
import { Stage, FlickerClock, flickerFrames, meanLum, hexToRgb } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { showResult } from '../results.js';

let webgazerLoaded = null;
function loadWebGazer() {
  webgazerLoaded ||= new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/webgazer@3.3.0/dist/webgazer.js';
    s.onload = () => res(window.webgazer); s.onerror = () => rej(new Error('could not load WebGazer (needs internet once)'));
    document.head.append(s);
  });
  return webgazerLoaded;
}

export default {
  id: 'freegaze', nav: 'Mind\'s eye (free gaze)', title: 'Pictures from the mind\'s eye — free viewing', fig: 'Fig. 6, 7',
  blurb: 'The first method in §III: the whole image flickers (12 Hz) with black, non-flickering contours. The observer rasters their gaze across it; for every eye position the relative 12 Hz power in a windowed FFT becomes a sample, and the samples are triangulated into an image. Eye position comes from a Tobii tracker (via the Python bridge), a webcam tracker (WebGazer), the mouse, or a guiding cursor that moves in a TV-style raster (no tracker needed). Eye X/Y is low-passed with a 1.5 Hz 6th-order Butterworth filter, as in the paper.',
  mount(el) {
    let img = null;
    const v = { freq: 12, colA: '#ffffff', colB: '#000000', levels: 'gray', gazeSource: 'guided', rows: 8, speed: 120, retrace: 1.2, limit: 180, winSec: 3, maxW: 1600 };
    const results = h('div');
    const picker = imagePicker(i => { img = i; }, 'NO (Fig. 6/7)');
    const f = form([
      { type: 'section', label: 'Stimulus' },
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'colA', label: 'Colour A', type: 'color' }, { key: 'colB', label: 'Colour B', type: 'color' },
      { key: 'levels', label: 'Levels', type: 'select', options: [['gray', 'Grayscale'], ['binary', 'Binary']] },
      { type: 'section', label: 'Eye position' },
      { key: 'gazeSource', label: 'Source', type: 'select', options: [['guided', 'Guiding cursor raster (no tracker)'], ['tobii', 'Tobii (python server.py --tobii)'], ['webgazer', 'Webcam (WebGazer)'], ['mouse', 'Mouse pointer']] },
      { key: 'rows', label: 'Raster rows', type: 'number', min: 1, step: 1, show: x => x.gazeSource === 'guided' },
      { key: 'speed', label: 'Cursor speed', type: 'number', min: 5, unit: 'px/s', show: x => x.gazeSource === 'guided' },
      { key: 'retrace', label: 'Retrace time', type: 'number', min: 0, step: 0.1, unit: 's', show: x => x.gazeSource === 'guided' },
      { key: 'limit', label: 'Time limit', type: 'number', min: 10, unit: 's', show: x => x.gazeSource !== 'guided' },
      { type: 'section', label: 'Analysis' },
      { key: 'winSec', label: 'FFT window', type: 'number', min: 0.5, step: 0.5, unit: 's (paper: 10)' },
    ], v);
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, picker, f,
        h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, h('h3', {}, 'Notes'), h('ul', {},
        h('li', {}, 'Guided: follow the white dot. It scans each row left→right, then jumps back (retrace) like a progressive TV raster.'),
        h('li', {}, 'Tobii / webcam / mouse: scan your eyes across the image yourself, in rows. Press ENTER or ESC to finish.'),
        h('li', {}, 'WebGazer first asks you to click 9 dots while looking at them (calibration). Accuracy is a few cm — the same limitation the paper lists.'),
        h('li', {}, 'The paper used a 10 s window; shorter windows give sharper pictures when you scan slowly. Change it after the run and re-render.')))),
    results);

    const run = async () => {
      requireSource();
      if (!img) throw new Error('No image selected');
      let wg = null;
      if (v.gazeSource === 'webgazer') {
        toast('Loading WebGazer…');
        wg = await loadWebGazer();
      }
      const stage = await new Stage().open({ cursor: v.gazeSource === 'mouse' || v.gazeSource === 'webgazer' });
      let tobiiOff = null, onMouse = null;
      try {
        const s = Math.min((stage.w - 40) / img.width, (stage.h - 40) / img.height, v.maxW / img.width);
        const W = Math.round(img.width * s), H = Math.round(img.height * s);
        const ox = Math.round((stage.w - W) / 2), oy = Math.round((stage.h - H) / 2);
        const frames = flickerFrames(img, W, H, hexToRgb(v.colA), hexToRgb(v.colB), { binary: v.levels === 'binary' });
        let meanAll = 0; for (const l of frames.lum) meanAll += l; meanAll /= frames.lum.length;
        const gaze = { t: [], x: [], y: [] };
        let cur = null; // latest eye position in image coords

        if (v.gazeSource === 'mouse') {
          onMouse = e => { cur = { x: e.clientX - ox, y: e.clientY - oy }; };
          window.addEventListener('mousemove', onMouse);
        } else if (v.gazeSource === 'tobii') {
          if (app.source?.kind !== 'bridge') toast('Tobii gaze arrives through the Python bridge — connect with "Python bridge" (server.py --muse --tobii)', 6000);
          const fn = (ts, xy) => {
            for (let i = 0; i < ts.length; i++) {
              const [nx, ny, valid] = xy[i];
              if (valid && nx != null) { gaze.t.push(ts[i]); gaze.x.push(nx * stage.w - ox); gaze.y.push(ny * stage.h - oy); cur = { x: nx * stage.w - ox, y: ny * stage.h - oy }; }
            }
          };
          app.gazeListeners.add(fn); tobiiOff = () => app.gazeListeners.delete(fn);
        } else if (wg) {
          await wg.setRegression('ridge').showVideoPreview(true).showPredictionPoints(true).begin();
          wg.setGazeListener(d => { if (d) cur = { x: d.x - ox, y: d.y - oy }; });
          // 9-point click calibration
          const pts = [0.1, 0.5, 0.9].flatMap(py => [0.1, 0.5, 0.9].map(px => [px * stage.w, py * stage.h]));
          for (const [px, py] of pts) {
            let clicks = 0;
            stage.clear('#111'); stage.dot(px, py, 12, '#e11'); stage.message('<div class="small">Look at the red dot and click it 5 times</div>');
            await new Promise(res => {
              const on = e => { if (Math.hypot(e.clientX - px, e.clientY - py) < 30 && ++clicks >= 5) { stage.el.removeEventListener('click', on); res(); } };
              stage.el.addEventListener('click', on);
            });
          }
          wg.showVideoPreview(false).showPredictionPoints(false);
          stage.message('');
          stage.el.style.cursor = 'none';
        }

        // guided raster path
        const rows = Math.max(1, Math.round(v.rows));
        const rowDur = W / v.speed, lineDur = rowDur + v.retrace;
        const guided = t => {
          const r = Math.floor(t / lineDur);
          if (r >= rows) return null;
          const u = t - r * lineDur, y = rows === 1 ? H / 2 : 10 + r * (H - 20) / (rows - 1);
          if (u < rowDur) return { x: u * v.speed, y };
          const a = (u - rowDur) / (v.retrace || 1), ny = rows === 1 ? y : 10 + Math.min(rows - 1, r + 1) * (H - 20) / (rows - 1);
          return { x: W * (1 - a), y: y + (ny - y) * a };
        };

        app.probe(() => (cur ? { s: 0.8 * meanLum(frames, cur.x - 40, cur.y - 40, 80) + 0.2 * meanAll, f: v.freq } : { s: meanAll * 0.2, f: v.freq }));
        stage.clear();
        stage.message('<div>Get ready</div><div class="small">' + (v.gazeSource === 'guided' ? 'Follow the white dot' : 'Scan the image row by row with your eyes; ENTER to finish') + ' · SPACE to start</div>');
        const k = await stage.waitKey([' ', 'Escape']);
        stage.message('');
        if (k === 'Escape') throw new Error('cancelled');
        const clock = new FlickerClock(v.freq, stage.refresh);
        let tBegin = null, done = false;
        const off = stage.onKey(e => { if (e.key === 'Enter' || e.key === 'Escape') done = true; });
        const lead = 2; // entrain before the raster starts
        await stage.loop((t, ts) => {
          if (done) return false;
          if (tBegin == null) tBegin = t;
          const A = clock.advance(ts), ctx = stage.ctx;
          ctx.fillStyle = '#000'; ctx.fillRect(0, 0, stage.w, stage.h);
          ctx.drawImage(A ? frames.A : frames.B, ox, oy);
          if (v.gazeSource === 'guided') {
            const g = guided(Math.max(0, t - tBegin - lead));
            if (!g) return false;
            cur = g;
            stage.dot(ox + g.x, oy + g.y, 5, '#fff'); stage.dot(ox + g.x, oy + g.y, 2, '#000');
          } else if (t - tBegin > v.limit) return false;
          if (cur && v.gazeSource !== 'tobii' && t - tBegin > lead) { gaze.t.push(t); gaze.x.push(cur.x); gaze.y.push(cur.y); }
        });
        off();
        stage.message('<div>Finishing…</div>');
        await sleep(v.winSec * 500 + 800);
        app.probe(null);
        await stage.close();
        if (gaze.t.length < 10) throw new Error('No eye positions were recorded');
        const run = { W, H, freq: v.freq, winSec: v.winSec, gaze, gazeSource: v.gazeSource, colors: [v.colA, v.colB] };
        const session = makeSession('gaze', run, tBegin, perfNow(), { reference: canvasToDataURL(img), title: `Mind's eye — ${v.gazeSource}` });
        results.innerHTML = '';
        showResult(results, session);
        results.scrollIntoView({ behavior: 'smooth' });
      } catch (e) {
        app.probe(null); await stage.close();
        if (e.message !== 'cancelled') throw e;
      } finally {
        if (onMouse) window.removeEventListener('mousemove', onMouse);
        tobiiOff?.();
        if (wg) { try { wg.end(); } catch { } }
      }
    };
  },
};
