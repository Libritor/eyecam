// Calibration: run before imaging.
//   1. Contact gate — live per-channel signal quality; imaging channels must settle.
//   2. SSVEP check — 8 × (5 s flicker on / 5 s off) while fixating a dot: measures your real
//      SSVEP, trains + cross-validates a spatial filter, predicts the scan you need.
//   3. Artifact prompts — rest, blink, clench, swallow, move: personal thresholds for masking.
//   4. Report — go / marginal / no-go, and one click to apply everything.

import { app, requireSource, setCalibration, say, beep } from '../context.js';
import { h, form, toast, sleep, download, readFile, stamp } from '../ui.js';
import { Stage, FlickerClock, measureRefresh, hexToRgb } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { contactQuality } from '../artifacts.js';
import { analyzeCalibration } from '../calib.js';
import { percentile } from '../dsp.js';

const TASKS = [
  ['rest', 'Keep still and relaxed — look at the dot', 8],
  ['blink', 'Blink normally, about 5 times', 6],
  ['clench', 'Clench your jaw firmly', 5],
  ['swallow', 'Swallow twice', 5],
  ['move', 'Turn your head slowly left and right', 6],
];

const pct = v => (Number.isFinite(v) ? `${(v * 100).toFixed(0)} %` : '–');
const f2 = v => (Number.isFinite(v) ? v.toFixed(2) : '–');

export function nearestExact(target, refresh) {
  const k = Math.max(1, Math.round(refresh / (2 * target)));
  return +(refresh / (2 * k)).toFixed(3);
}

export function reportView(rep) {
  const col = { go: 'var(--ok)', marginal: 'var(--warn)', nogo: 'var(--bad)' }[rep.verdict];
  const msg = {
    go: `GO — your SSVEP is clearly measurable. ${rep.plan.passes > 1 ? rep.plan.passes + ' passes × ' : ''}${rep.plan.winSec} s per pixel should give a readable letter N.`,
    marginal: `MARGINAL — measurable but weak. Expect a noisy image; use the slower plan below (and 2 passes), or improve contact first.`,
    nogo: `NO-GO — no reliable SSVEP at the electrodes. Imaging would only show noise. Fix contact (wet sensors, hair away from behind the ears, snug band), relax your jaw, then recalibrate.`,
  }[rep.verdict];
  const best = rep.filter.use ? rep.pipelines.filter : rep.pipeline === 'single' ? rep.pipelines.single : rep.pipelines.base;
  const which = rep.filter.use ? 'spatial filter' : rep.pipeline === 'single' ? `${rep.bestChannel} alone` : rep.pipelines.baseNames.join('+');
  return h('div', {},
    h('div', { class: 'card', style: `border-left:6px solid ${col};margin-bottom:10px` },
      h('h3', { style: `color:${col}` }, rep.verdict.toUpperCase()), h('p', {}, msg),
      h('p', { class: 'stats' }, `Detectability d′ = ${f2(rep.d1)} per second of data (cross-validated, ${which}) · ON/OFF power ${f2(best.snr)}× · ${rep.segments.used}/${rep.segments.total} clean 1 s segments`)),
    h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h4', {}, 'Per channel'),
        h('table', { class: 'table' }, h('tr', {}, h('th', {}, 'Channel'), h('th', {}, 'Contact p2p (1 s)'), h('th', {}, 'SSVEP amplitude'), h('th', {}, 'ON/OFF power'), h('th', {}, 'd′ / s')),
          rep.perChannel.map(c => h('tr', {}, h('td', {}, c.name), h('td', {}, `${f2(c.p2pMedian)} µV`), h('td', {}, `${f2(c.ampUV)} µV`), h('td', {}, `${f2(c.snr)}×`), h('td', {}, f2(c.d1))))),
        h('h4', {}, 'Spatial filter (2-fold cross-validated by cycle)'),
        h('p', { class: 'stats' }, `held-out d′: average of ${rep.pipelines.baseNames.join('+')} ${f2(rep.pipelines.base.d1)} → filter ${f2(rep.pipelines.filter.d1)} (×${f2(rep.filter.gainD1)}). ${rep.filter.use ? 'ADOPTED' : 'Not adopted (gain < 15 % on held-out data)'}.`),
        h('p', { class: 'stats' }, 'weights: ' + Object.entries(rep.filter.weights).map(([n, w]) => `${n} ${w.toFixed(2)}`).join(' · ')),
        rep.pipelines.single ? h('p', { class: 'stats' }, `best single channel: ${rep.pipelines.single.name}, held-out d′ ${f2(rep.pipelines.single.d1)}${rep.pipeline === 'single' ? ' — ADOPTED (Apply sets Channels to it)' : ''}`) : null),
      h('div', { class: 'card' }, h('h4', {}, 'Scan plan for the letter N'),
        h('p', {}, rep.verdict === 'nogo' ? 'Not recommended until contact improves.' :
          `window ${rep.plan.winSec} s · cursor ${rep.plan.cursor} px at ${rep.plan.speed} px/s · ${rep.plan.rows} lines × ${rep.plan.passes} pass${rep.plan.passes > 1 ? 'es' : ''} ≈ ${(rep.plan.minutes * rep.plan.passes).toFixed(0)} min at ${rep.freq} Hz`),
        h('h4', {}, 'Artifact thresholds (personal)'),
        h('table', { class: 'table' }, h('tr', {}, h('th', {}, 'Ch'), h('th', {}, 'p2p'), h('th', {}, 'EMG'), h('th', {}, 'clean flagged'), h('th', {}, 'blink'), h('th', {}, 'clench'), h('th', {}, 'swallow'), h('th', {}, 'move')),
          rep.artifactReport.map(a => h('tr', {}, h('td', {}, a.name), h('td', {}, `${a.p2pThr.toFixed(0)} µV`), h('td', {}, a.emgThr.toFixed(1)), h('td', {}, pct(a.cleanFlagged)),
            h('td', {}, pct(a.caught.blink)), h('td', {}, pct(a.caught.clench)), h('td', {}, pct(a.caught.swallow)), h('td', {}, pct(a.caught.move))))),
        h('p', { class: 'muted small' }, 'Caught = share of 0.25 s blocks during each prompt that would be masked. A blink barely affects 15 Hz, so a low blink catch rate is fine; clench/move matter.'))));
}

export default {
  id: 'calibrate', nav: 'Calibration', title: 'Calibration — check signal before imaging', fig: 'SNR',
  blurb: 'About 2½ minutes. First the electrodes must settle; then a square flashes on and off while you look at a dot (measures your SSVEP and trains a cross-validated spatial filter); then a few prompted artifacts set your personal masking thresholds. The report says whether imaging will work and how slowly to scan.',
  mount(el) {
    const v = { freq: 15, cycles: 8, on: 5, off: 5, size: 160, colA: '#ffffff', colB: '#000000', artifacts: true, gate: 100 };
    const live = h('div', { class: 'meters' });
    const gateMsg = h('p', { class: 'stats' }, '');
    const report = h('div');
    const startBtn = h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start calibration');
    const anyway = h('label', { class: 'small' }, h('input', { type: 'checkbox' }), ' start even if contact is poor');
    const f = form([
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.01, unit: 'Hz (snapped to your display)' },
      { key: 'size', label: 'Patch size', type: 'number', min: 20, unit: 'px (≥ imaging cursor)' },
      { key: 'colA', label: 'Colour A', type: 'color' }, { key: 'colB', label: 'Colour B', type: 'color' },
      { key: 'cycles', label: 'On/off cycles', type: 'number', min: 4, step: 2 },
      { key: 'artifacts', label: 'Include artifact prompts', type: 'checkbox' },
      { key: 'gate', label: 'Contact gate', type: 'number', min: 20, unit: 'µV p2p (TP9/TP10)' },
    ], v);
    measureRefresh().then(r => { f.set('freq', nearestExact(15, r)); });
    const loadIn = h('input', { type: 'file', accept: '.json', style: 'display:none' });
    loadIn.onchange = async () => { try { const rep = JSON.parse(await readFile(loadIn.files[0])); setCalibration(rep); showReport(rep); toast('Calibration loaded and applied'); } catch (e) { toast(e.message); } };
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h3', {}, '1 · Contact'), live, gateMsg, f, h('div', { class: 'btns' }, startBtn), anyway,
        h('div', { class: 'btns' }, h('button', { onclick: () => loadIn.click() }, 'Load calibration…'), loadIn)),
      h('div', { class: 'card' }, h('h3', {}, 'Tips for good contact'), h('ul', {},
        h('li', {}, 'Dampen the four sensors slightly (water or saline).'),
        h('li', {}, 'Push hair away from behind the ears; the rubber TP9/TP10 pads must touch skin.'),
        h('li', {}, 'Band snug, not tight; sit with your head supported and jaw relaxed.'),
        h('li', {}, 'Wait for TP9/TP10 to settle below the gate (green) — it often takes 1–2 minutes as the sensors wet. One stable ear is enough to start.'),
        h('li', {}, 'A channel that keeps flickering: if its hint says muscle/interference, relax your jaw and check the pad touches skin (not hair); if it says large swings, the pad is moving — re-seat it and tuck the band behind the ear.')),
        h('p', { class: 'muted' }, 'Last calibration: ', app.calib ? `${app.calib.created.slice(0, 16).replace('T', ' ')} — ${app.calib.verdict.toUpperCase()}${app.settings.calibApplied === app.calib.created ? ' (applied)' : ''}` : 'none'))),
    report);
    if (app.calib) showReport(app.calib);

    // ---- live contact gate
    // A channel whose contact flickers between good and bad must not restart the countdown
    // every time: each channel is judged by the share of the last 10 s it was not 'bad'.
    // Calibration can start when TP9 or TP10 is stable — the analysis then finds the best
    // channel (one ear is enough) and the artifact mask handles the noisy one.
    const HIST = 20, STABLE = 0.85; // 20 ticks × 0.5 s
    const hist = new Map();
    let alive = true;
    const tick = () => {
      if (!alive) return;
      const st = app.store;
      live.innerHTML = '';
      const cq = contactQuality(st, v.gate);
      if (cq) {
        const stable = [], flaky = [];
        for (const q of cq) {
          const hh = hist.get(q.name) || []; hh.push(q.cls !== 'bad'); if (hh.length > HIST) hh.shift(); hist.set(q.name, hh);
          const share = hh.filter(Boolean).length / hh.length, full = hh.length >= HIST;
          const isTP = /TP9|TP10/i.test(q.name);
          if (isTP && full && share >= STABLE) stable.push(q.name);
          else if (isTP && share > 0.3) flaky.push([q.name, share]);
          const shown = q.cls === 'bad' && share >= STABLE ? 'warn' : q.cls; // a brief dip is not red
          const flick = full && share < STABLE && share > 0.3 ? ` · unstable: good ${Math.round(share * 100)} % of the last 10 s` : '';
          live.append(h('div', { class: 'meter' }, h('span', {}, h('span', { class: 'dot ' + shown }), ' ', q.name, q.imaging ? '' : ' (ref)'),
            h('b', {}, Number.isFinite(q.p2p) ? `${q.p2p.toFixed(0)} µV` : '–'), h('span', { class: 'muted small' }, `EMG ${q.emg.toFixed(1)} · ${q.hint}${flick}`)));
        }
        const filled = Math.min(...[...hist.values()].map(x => x.length)) >= HIST;
        const gateOk = filled && stable.length > 0;
        gateMsg.textContent = !filled ? `Measuring contact… ${Math.ceil((HIST - Math.min(...[...hist.values()].map(x => x.length))) / 2)} s`
          : gateOk ? (stable.length === 2 ? 'Contact OK — ready.'
            : `Ready with ${stable[0]}. ${flaky.length ? flaky[0][0] + ' is unstable' : 'The other ear has no stable contact'}: calibration will use the better channel and down-weight the other. Fixing it still helps.`)
          : `Waiting for TP9 or TP10 to stay below ${v.gate} µV p2p and EMG 100 for most of 10 s…`;
        startBtn.disabled = !(gateOk || anyway.firstChild.checked);
      } else { gateMsg.textContent = app.source ? 'Collecting data…' : 'Connect the Muse (or the simulator) first.'; startBtn.disabled = !anyway.firstChild.checked; }
      setTimeout(tick, 500);
    };
    tick();
    anyway.firstChild.onchange = () => { startBtn.disabled = false; };
    this._stop = () => { alive = false; };

    function showReport(rep) {
      report.innerHTML = '';
      report.append(h('h3', {}, 'Calibration report'), reportView(rep),
        h('div', { class: 'btns' },
          h('button', { class: 'primary', onclick: () => { setCalibration(rep); toast('Applied: calibrated artifact masking' + (rep.filter.use ? ' + spatial filter' : '') + `; imaging at ${rep.freq} Hz`); } }, 'Apply to analysis'),
          h('button', { onclick: () => download(`eyecam-calibration-${stamp()}.json`, JSON.stringify(rep, null, 1), 'application/json') }, 'Save calibration'),
          h('a', { href: '#raster' }, h('button', {}, 'Go to imaging →'))));
    }

    const run = async () => {
      requireSource();
      const stage = await new Stage().open();
      const sim = app.source.kind === 'sim' ? app.source.gen : null;
      const cal = { freq: v.freq, blocks: [], tasks: [], flicker: { t: [], c: [] } };
      let flick = false;
      app.probe(() => (flick ? { s: 1, f: v.freq } : null));
      const t0 = perfNow();
      try {
        const [A, B] = [v.colA, v.colB];
        const clock = new FlickerClock(v.freq, stage.refresh);
        stage.clear(); stage.dot(stage.w / 2, stage.h / 2, 4);
        stage.message('<div>Look at the dot the whole time.</div><div class="small">A square will flash on and off around it. Keep still, jaw relaxed.<br>SPACE to start · ESC to cancel</div>');
        if ((await stage.waitKey([' ', 'Escape'])) === 'Escape') throw new Error('cancelled');
        stage.message('');
        let stop = false;
        const off = stage.onKey(e => { if (e.key === 'Escape') stop = true; });
        const cx = stage.w / 2, cy = stage.h / 2, s = v.size;
        await sleep(1500);
        for (let k = 0; k < v.cycles && !stop; k++) {
          for (const on of [true, false]) {
            let b0 = null;
            const dur = on ? v.on : v.off;
            flick = on;
            await stage.loop((t, ts) => {
              if (stop) return false;
              if (b0 == null) b0 = t;
              if (t - b0 >= dur) return false;
              const a = clock.advance(ts); clock.log(cal.flicker, t);
              stage.clear();
              if (on) { stage.ctx.fillStyle = a ? A : B; stage.ctx.fillRect(cx - s / 2, cy - s / 2, s, s); }
              stage.dot(cx, cy, 4, on ? '#f33' : '#fff');
            });
            cal.blocks.push({ on, t0: b0, t1: b0 + dur, cycle: k });
          }
        }
        flick = false;
        if (v.artifacts && !stop) {
          for (const [name, text, dur] of TASKS) {
            if (stop) break;
            stage.clear(); stage.dot(cx, cy, 4);
            stage.message(`<div>${text}</div><div class="small">starts in 2 s</div>`);
            say(text);
            await sleep(2200);
            beep(880, 0.06, 0.15);
            const a = perfNow();
            sim?.inject(name, dur);
            for (let left = dur; left > 0 && !stop; left -= 1) { stage.message(`<div>${text}</div><div class="small">${left} s</div>`); await sleep(1000); }
            cal.tasks.push({ name, t0: a, t1: perfNow() });
            beep(660, 0.06, 0.15);
            stage.message('<div>Relax</div>'); await sleep(1500);
          }
        }
        off();
        if (stop) throw new Error('cancelled');
        stage.message('<div>Analysing…</div>');
        await sleep(1200);
      } finally { app.probe(null); await stage.close(); }
      const seg = app.store.segment(t0 - 2, perfNow() + 1);
      const rep = analyzeCalibration(seg, cal, { latency: app.settings.latency, cursor: 100 });
      rep.source = app.source?.info?.() || '';
      setCalibration(rep, { apply: false });
      showReport(rep);
      report.scrollIntoView({ behavior: 'smooth' });
    };
  },
  unmount() { this._stop?.(); },
};
