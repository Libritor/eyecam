// Fig. 13 / 14: SSVEPVMP — Steady-State Visually Evoked Potentials Visual Memory Prosthetic.
// The wearer's world is mediated to flicker at 15 Hz (shutter glasses, or a video
// see-through screen). A high SSVEP means the scene is engaging: the system captures a
// snapshot from the head-worn camera. The wearer marks interesting events; the hit rate of
// high-SSVEP detections against those marks is reported (paper: 89 %).

import { app, analysisParams, requireSource, makeSession, beep } from '../context.js';
import { h, form, toast, sleep, lineChart, download, stamp } from '../ui.js';
import { Stage, FlickerClock } from '../stage.js';
import { perfNow } from '../eeg-store.js';
import { metricAt, detectEvents } from '../recon.js';
import { showResult } from '../results.js';

export default {
  id: 'vmp', nav: 'SSVEPVMP memory prosthetic', title: 'SSVEPVMP — visual memory prosthetic', fig: 'Fig. 13, 14',
  blurb: 'Go about your day while your world flickers at 15 Hz. When something engages you, your SSVEP rises and the head-worn camera keeps a picture. Press SPACE (or the button) whenever something interesting happens; afterwards the normalized SSVEP trace is plotted with your marks as red bands and the detection hit rate is computed.',
  mount(el) {
    const v = { world: 'screen', freq: 15, winSec: 4, hop: 0.5, k: 2, refractory: 10, minutes: 30, snapshots: true, audio: false, simMarks: true };
    const chart = h('canvas', { class: 'chart tall' });
    const gallery = h('div', { class: 'gallery' });
    const stat = h('p', { class: 'stats' });
    const results = h('div');
    const markBtn = h('button', { class: 'primary big', disabled: true }, 'Mark interesting (SPACE)');
    const f = form([
      { key: 'world', label: 'World flicker', type: 'select', options: [['screen', 'Video see-through (webcam, fullscreen)'], ['glasses', 'LC shutter glasses (Arduino)'], ['lamp', 'Room lamp (Arduino)'], ['none', 'None (ambient)']] },
      { key: 'freq', label: 'Flicker', type: 'number', min: 1, max: 40, step: 0.5, unit: 'Hz' },
      { key: 'winSec', label: 'Window', type: 'number', min: 1, step: 0.5, unit: 's' },
      { key: 'hop', label: 'Update every', type: 'number', min: 0.1, step: 0.1, unit: 's' },
      { key: 'k', label: 'Detection threshold', type: 'number', step: 0.1, unit: '× MAD' },
      { key: 'refractory', label: 'Refractory', type: 'number', min: 0, unit: 's' },
      { key: 'minutes', label: 'Duration', type: 'number', min: 1, unit: 'min (paper: 30)' },
      { key: 'snapshots', label: 'Camera snapshot on detection', type: 'checkbox' },
      { key: 'audio', label: 'Record microphone', type: 'checkbox' },
      { key: 'simMarks', label: 'Simulator: auto-mark its engaging episodes', type: 'checkbox' },
    ], v);
    el.append(h('div', { class: 'cols' },
      h('div', { class: 'card' }, f, h('div', { class: 'btns' }, h('button', { class: 'primary big', onclick: () => run().catch(e => toast(e.message, 6000)) }, 'Start'))),
      h('div', { class: 'card' }, markBtn, chart, stat, gallery)),
    results);

    const run = async () => {
      requireSource();
      if ((v.world === 'glasses' || v.world === 'lamp') && !app.io.connected) throw new Error('Connect the EyeCam Arduino under Hardware.');
      const needCam = v.snapshots || v.world === 'screen';
      const media = needCam || v.audio ? await navigator.mediaDevices.getUserMedia({ video: needCam ? { width: 1280, height: 720 } : false, audio: v.audio }) : null;
      const video = h('video', { autoplay: true, playsinline: true, muted: true });
      if (media && needCam) { video.srcObject = media; await video.play(); }
      let rec = null; const chunks = [];
      if (v.audio && media) {
        rec = new MediaRecorder(new MediaStream(media.getAudioTracks()));
        rec.ondataavailable = e => chunks.push(e.data); rec.start(1000);
      }
      const snaps = [], marks = [], series = [], detections = [];
      const snap = t => {
        if (!needCam || !video.videoWidth) return;
        const c = document.createElement('canvas'); c.width = 480; c.height = Math.round(480 * video.videoHeight / video.videoWidth);
        c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
        const s = { t, src: c.toDataURL('image/jpeg', 0.8) };
        snaps.push(s);
        gallery.prepend(h('figure', {}, h('img', { src: s.src }), h('figcaption', {}, `${((t - t0) / 60).toFixed(2)} min`)));
      };
      const mark = () => { marks.push(perfNow()); beep(1000, 0.05, 0.15); };
      markBtn.disabled = false; markBtn.onclick = mark;

      // simulator: engaging episodes
      const episodes = [];
      for (let t = perfNow() + 20; t < perfNow() + v.minutes * 60; t += 35 + Math.random() * 40) episodes.push([t, t + 7 + Math.random() * 8]);
      app.probe(t => ({ s: episodes.some(([a, b]) => t >= a && t <= b) ? 1 : 0.25, f: v.freq }));
      const simMarkTimers = app.source.kind === 'sim' && v.simMarks ? episodes.map(([a]) => setTimeout(() => mark(), (a + 2 - perfNow()) * 1000)) : [];

      if (v.world === 'glasses') await app.io.shutter(v.freq);
      if (v.world === 'lamp') await app.io.lampFlicker(v.freq);
      const t0 = perfNow(), tEnd = t0 + v.minutes * 60;
      let stop = false, above = false, lastDet = -Infinity;
      const p = analysisParams(v.freq, v.winSec);
      const stage = v.world === 'screen' ? await new Stage().open() : null;
      const keyFn = e => { if (e.key === ' ') { e.preventDefault(); mark(); } if (e.key === 'Escape') stop = true; };
      window.addEventListener('keydown', keyFn);
      let loopDone = null;
      if (stage) {
        const clock = new FlickerClock(v.freq, stage.refresh);
        loopDone = stage.loop((t, ts) => {
          if (stop) return false;
          const A = clock.advance(ts), c = stage.ctx;
          if (A && video.videoWidth) {
            const s = Math.max(stage.w / video.videoWidth, stage.h / video.videoHeight);
            const w = video.videoWidth * s, hh = video.videoHeight * s;
            c.drawImage(video, (stage.w - w) / 2, (stage.h - hh) / 2, w, hh);
          } else stage.clear();
          c.fillStyle = 'rgba(0,0,0,.5)'; c.fillRect(8, 8, 250, 26);
          c.fillStyle = '#ddd'; c.font = '14px system-ui'; c.fillText(`marks ${marks.length} · captures ${snaps.length} · ESC ends`, 16, 26);
        });
      }
      try {
        while (!stop && perfNow() < tEnd) {
          await sleep(v.hop * 1000);
          const t = perfNow();
          const m = metricAt(app.store, t - v.winSec / 2 - p.latency - 0.15, p);
          if (!m) continue;
          series.push({ t, v: m.v, bad: m.bad });
          if (series.length > 30) {
            const det = detectEvents(series.slice(-Math.round(180 / v.hop)), { k: v.k, refractory: v.refractory, useBad: true });
            const hit = m.v > det.thr && !m.bad;
            if (hit && !above && t - lastDet >= v.refractory) { lastDet = t; detections.push(t); if (v.snapshots) snap(t); }
            above = hit;
          }
          if (series.length % 2 === 0) {
            const vals = series.map(s => s.v), lo = Math.min(...vals), hi = Math.max(...vals);
            lineChart(chart, {
              series: [{ x: series.map(s => (s.t - t0) / 60), y: series.map(s => (s.v - lo) / (hi - lo || 1)), color: '#2563eb', label: 'Normalized SSVEP' }],
              bands: marks.map(mk => ({ a: (mk - t0 - 3) / 60, b: (mk - t0 + 3) / 60, color: 'rgba(239,68,68,.45)' })),
              marks: detections.map(d => ({ x: (d - t0) / 60, color: '#16a34a' })), xlabel: 'time (min)', ymin: 0, ymax: 1,
            });
            stat.textContent = `${((t - t0) / 60).toFixed(1)} / ${v.minutes} min · ${marks.length} marks · ${detections.length} detections`;
          }
        }
      } finally {
        stop = true;
        await loopDone;
        window.removeEventListener('keydown', keyFn);
        simMarkTimers.forEach(clearTimeout);
        markBtn.disabled = true;
        if (stage) await stage.close();
        if (v.world === 'glasses') await app.io.shutter(0);
        if (v.world === 'lamp') await app.io.lampFlicker(0);
        app.probe(null);
        if (rec) { rec.stop(); await sleep(300); download(`eyecam-vmp-audio-${stamp()}.webm`, new Blob(chunks, { type: rec.mimeType })); }
        media?.getTracks().forEach(tr => tr.stop());
      }
      await sleep(600);
      const run = { freq: v.freq, winSec: v.winSec, k: v.k, refractory: v.refractory, t0, t1: perfNow(), marks: marks.filter(m => m <= perfNow()), detections, world: v.world };
      const session = makeSession('vmp', run, t0, perfNow(), { snapshots: snaps, title: `SSVEPVMP — ${((run.t1 - t0) / 60).toFixed(1)} min` });
      results.innerHTML = ''; showResult(results, session); results.scrollIntoView({ behavior: 'smooth' });
    };
  },
};
