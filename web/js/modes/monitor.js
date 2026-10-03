// Live signal check: raw traces, contact quality, spectrum, SSVEP meter, flicker test.

import { app, analysisParams, applyPreferred } from '../context.js';
import { h, form, lineChart, download, stamp } from '../ui.js';
import { spectrum, ssvepMetric, std } from '../dsp.js';
import { windowEnding, segmentToCSV } from '../eeg-store.js';
import { FlickerClock, measureRefresh } from '../stage.js';
import { contactQuality } from '../artifacts.js';

export default {
  id: 'monitor', nav: 'Signal monitor', title: 'Signal monitor', fig: 'setup',
  blurb: 'Check electrode contact and see the SSVEP appear. Put the Muse on with the extra electrode over the occipital lobe (Oz) plugged into the AUX port if you have one; otherwise TP9/TP10 behind the ears pick up the occipital response. Stare at the flicker patch: a clear peak at the flicker frequency (and its harmonic) should appear within a few seconds.',
  mount(el) {
    const v = applyPreferred({ freq: 15, colA: '#ffff00', colB: '#0000ff', win: 4, flicker: false });
    const traces = h('canvas', { class: 'chart tall' });
    const spec = h('canvas', { class: 'chart' });
    const meters = h('div', { class: 'meters' });
    const ssv = h('div', { class: 'meter' }, h('span', { class: 'muted' }, 'SSVEP'), h('b', {}, '–'), h('div', { class: 'bar' }, h('i')));
    const patch = h('canvas', { width: 220, height: 220, style: 'background:#000;border-radius:8px;display:block' });
    let rec = null;
    const recBtn = h('button', {
      onclick: () => {
        if (!rec) { rec = app.store.lastTime; recBtn.textContent = 'Stop & save CSV'; }
        else { download(`eyecam-raw-${stamp()}.csv`, segmentToCSV(app.store.segment(rec, app.store.lastTime + 1)), 'text/csv'); rec = null; recBtn.textContent = 'Record raw EEG'; }
      },
    }, 'Record raw EEG');

    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' },
        h('h3', {}, 'Flicker test'),
        form([
          { key: 'freq', label: 'Frequency', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
          { key: 'colA', label: 'Colour A', type: 'color' }, { key: 'colB', label: 'Colour B', type: 'color' },
          { key: 'win', label: 'Analysis window', type: 'number', min: 1, max: 20, unit: 's' },
          { key: 'flicker', label: 'Flicker on', type: 'checkbox' },
        ], v),
        patch, h('div', { class: 'info', id: 'refresh-info' }, 'measuring display refresh…'),
        h('div', { style: 'margin-top:10px' }, ssv),
        h('div', { class: 'btns' }, recBtn)),
      h('div', {},
        h('div', { class: 'card' }, h('h4', {}, 'Electrode contact (1 s peak-to-peak, same check as Calibration)'), meters),
        h('div', { class: 'card', style: 'margin-top:10px' }, h('h4', {}, 'Raw EEG (last 5 s)'), traces),
        h('div', { class: 'card', style: 'margin-top:10px' }, h('h4', {}, 'Spectrum of analysis channels'), spec))));

    let refresh = 60;
    measureRefresh().then(r => {
      refresh = r;
      const q = FlickerClock.quality(v.freq, r);
      const exact = [];
      for (let k = 1; k <= 12; k++) { const f = r / (2 * k); if (f >= 5 && f <= 40) exact.push(f.toFixed(2)); }
      el.querySelector('#refresh-info').textContent = `Display ${r.toFixed(1)} Hz. Exact square waves (equal on/off frames) at: ${exact.join(', ')} Hz` +
        (q.exact ? '.' : `. ${v.freq} Hz is approximated — pick one of these (or set the display to 60/120 Hz) for a cleaner SSVEP.`);
    });

    const clock = new FlickerClock(v.freq, refresh);
    const pc = patch.getContext('2d');
    let alive = true, lastSlow = 0;
    app.probe(() => (v.flicker ? { s: 1, f: v.freq } : null));
    const loop = ts => {
      if (!alive) return;
      clock.freq = v.freq; clock.refresh = refresh;
      const A = clock.advance(ts);
      pc.fillStyle = v.flicker ? (A ? v.colA : v.colB) : '#111'; pc.fillRect(0, 0, 220, 220);
      pc.fillStyle = '#fff'; pc.beginPath(); pc.arc(110, 110, 3, 0, 7); pc.fill();
      if (ts - lastSlow > 250) { lastSlow = ts; slow(); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    const slow = () => {
      const st = app.store;
      if (!st.n) return;
      // contact quality
      meters.innerHTML = '';
      for (const q of contactQuality(st) || []) {
        meters.append(h('div', { class: 'meter' }, h('span', {}, h('span', { class: 'dot ' + q.cls }), ' ', q.name, q.imaging ? '' : ' (ref)'),
          h('b', {}, Number.isFinite(q.p2p) ? `${q.p2p.toFixed(0)} µV` : '–'),
          h('span', { class: 'muted small' }, `EMG ${q.emg.toFixed(1)} · ${q.hint}`)));
      }
      // traces
      const n = Math.min(st.n, 5 * st.fs), i0 = st.n - n;
      const t = Array.from(st.times.subarray(i0, st.n), x => x - st.times[st.n - 1]);
      const series = st.channels.map((name, c) => {
        const d = st.data[c].subarray(i0, st.n); let m = 0; for (const x of d) m += x; m /= d.length || 1;
        return { x: t, y: Array.from(d, x => (x - m) / 100 + (st.channels.length - c)), color: ['#e11d48', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'][c % 5], width: 1, label: name };
      });
      lineChart(traces, { series, xlabel: 's', ylabel: 'channel (100 µV/div)', ymin: 0, ymax: st.channels.length + 1 });
      // spectrum + SSVEP
      const p = analysisParams(v.freq, v.win);
      const w = windowEnding(st, st.lastTime, Math.round(v.win * st.fs), p.chans);
      if (!w) return;
      const sp = spectrum(w, st.fs, { psd: p.psd, segSec: Math.min(2, v.win) });
      const keep = [...sp.f].map((f, i) => [f, sp.p[i]]).filter(([f]) => f >= 1 && f <= 60);
      lineChart(spec, {
        series: [{ x: keep.map(k => k[0]), y: keep.map(k => k[1]), color: '#c2410c', label: p.chans.map(i => st.channels[i]).join('+') }],
        marks: [{ x: v.freq, color: '#16a34a' }, { x: 2 * v.freq, color: '#16a34a' }], xlabel: 'Hz', ylabel: 'µV²/Hz (log)', logy: true,
      });
      const m = ssvepMetric(sp, v.freq, p);
      ssv.querySelector('b').textContent = `${m.toFixed(3)}  (${p.metric})`;
      ssv.querySelector('i').style.width = Math.min(100, m / (p.metric === 'snr' ? 20 : p.metric === 'power' ? 20 : 0.5) * 100) + '%';
    };
    this._stop = () => { alive = false; };
  },
  unmount() { this._stop?.(); },
};
