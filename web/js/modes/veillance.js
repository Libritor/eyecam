// §I-A, Fig. 2 / 3: meta-sensing of cameras (sensing a sensor's capacity to sense) and
// long-exposure photography — used to photograph the brain-modulated LED of Fig. 4 too.
//
// Feedback loop (Fig. 2): the camera under test watches the scene; whatever light it sees
// above the ambient baseline drives the lamp (Arduino RGB LED), with a bias glow, so the
// lamp brightens only where the camera can see it. A second camera (or a phone on long
// exposure) photographs the lamp being waved around: that picture is the camera's
// sightfield. Colour mapping as Fig. 3: red = bias, green = medium, blue = strong veillance.

import { app } from '../context.js';
import { h, form, toast, downloadCanvas, stamp } from '../ui.js';
import { colorOf } from '../interp.js';

async function cameras() {
  try { (await navigator.mediaDevices.getUserMedia({ video: true })).getTracks().forEach(t => t.stop()); } catch { }
  return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
}

function camPanel(title, role) {
  const v = { device: '', mode: 'lighten', gain: 0.15 };
  const video = h('video', { class: 'preview', autoplay: true, playsinline: true, muted: true });
  const sel = h('select');
  let stream = null;
  const start = async () => {
    stream?.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: sel.value ? { exact: sel.value } : undefined, width: 1280, height: 720 } });
    video.srcObject = stream; await video.play();
  };
  cameras().then(list => { sel.innerHTML = ''; list.forEach((d, i) => sel.append(h('option', { value: d.deviceId }, d.label || `camera ${i + 1}`))); if (role === 'photo' && list[1]) sel.value = list[1].deviceId; });
  sel.onchange = () => start().catch(e => toast(e.message));
  const el = h('div', { class: 'card' }, h('h3', {}, title), h('label', { class: 'form-row' }, h('span', { class: 'lbl' }, 'Camera'), sel, h('button', { onclick: () => start().catch(e => toast(e.message)) }, 'Start')), video);
  return { el, video, v, stop: () => stream?.getTracks().forEach(t => t.stop()) };
}

export default {
  id: 'veillance', nav: 'Camera metaveillance', title: 'Metaveillance of cameras & long-exposure photography', fig: 'Fig. 2, 3',
  blurb: 'The sensing of sensors. Left: the camera under test drives a light (video feedback) so that the light glows only where the camera can see it. Right: a long-exposure camera photographs the light being waved around, revealing the camera\'s sightfield. The same long-exposure tool photographs the brain-modulated LED in the visual-field experiment.',
  mount(el) {
    const under = camPanel('Camera under test (feedback loop)', 'test');
    const photo = camPanel('Long-exposure camera', 'photo');
    const fb = { gain: 3, bias: 0.08, tau: 0.25, enabled: false, paint: true };
    const lx = { mode: 'lighten', running: false, gain: 0.08 };
    const swatch = h('div', { style: 'width:100%;height:60px;border-radius:8px;background:#200;margin-top:8px' });
    const meter = h('div', { class: 'bar' }, h('i'));
    const overlay = h('canvas', { width: 640, height: 360, style: 'width:100%;max-width:560px;background:#000;border-radius:6px;margin-top:8px' });
    const exposure = h('canvas', { width: 1280, height: 720, style: 'width:100%;max-width:560px;background:#000;border-radius:6px;margin-top:8px' });
    let baseline = null;
    under.el.append(form([
      { key: 'enabled', label: 'Drive the light', type: 'checkbox' },
      { key: 'gain', label: 'Loop gain', type: 'number', min: 0, step: 0.5 },
      { key: 'bias', label: 'Bias glow', type: 'number', min: 0, max: 1, step: 0.02 },
      { key: 'tau', label: 'Response time constant', type: 'number', min: 0, step: 0.05, unit: 's (hysteresis)' },
      { key: 'paint', label: 'Paint virtual sightfield', type: 'checkbox' },
    ], fb), h('div', { class: 'btns' },
      h('button', { onclick: () => { baseline = null; calib = 20; toast('Calibrating ambient — keep the light out of view'); } }, 'Calibrate ambient'),
      h('button', { onclick: () => overlay.getContext('2d').clearRect(0, 0, 640, 360) }, 'Clear painting'),
      h('button', { onclick: () => downloadCanvas(`sightfield-${stamp()}.png`, overlay) }, 'Save painting')),
    meter, swatch, overlay);
    photo.el.append(form([
      { key: 'mode', label: 'Accumulate', type: 'select', options: [['lighten', 'Lighten (max) — light painting'], ['add', 'Add (true long exposure)']] },
      { key: 'gain', label: 'Add gain', type: 'number', min: 0.01, step: 0.01, show: x => x.mode === 'add' },
    ], lx), h('div', { class: 'btns' },
      h('button', { class: 'primary', onclick: e => { lx.running = !lx.running; e.target.textContent = lx.running ? 'Stop exposure' : 'Open shutter'; } }, 'Open shutter'),
      h('button', { onclick: () => { const c = exposure.getContext('2d'); c.globalCompositeOperation = 'source-over'; c.fillStyle = '#000'; c.fillRect(0, 0, 1280, 720); } }, 'Clear'),
      h('button', { onclick: () => downloadCanvas(`long-exposure-${stamp()}.png`, exposure) }, 'Save photo')), exposure);
    el.append(h('div', { class: 'cols', style: 'grid-template-columns:1fr 1fr' }, under.el, photo.el));

    const small = document.createElement('canvas'); small.width = 64; small.height = 36;
    const sc = small.getContext('2d', { willReadFrequently: true });
    let level = 0, last = performance.now(), calib = 0, calAcc = null, alive = true;
    const loop = now => {
      if (!alive) return;
      const dt = (now - last) / 1000; last = now;
      const vid = under.video;
      if (vid.videoWidth) {
        sc.drawImage(vid, 0, 0, 64, 36);
        const d = sc.getImageData(0, 0, 64, 36).data, lum = new Float32Array(64 * 36);
        for (let i = 0; i < lum.length; i++) lum[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 765;
        if (calib > 0) { calAcc = calAcc ? calAcc.map((v, i) => Math.max(v, lum[i])) : lum; if (--calib === 0) baseline = calAcc, calAcc = null; }
        let best = 0, bi = 0;
        for (let i = 0; i < lum.length; i++) { const s = lum[i] - (baseline ? baseline[i] : 0); if (s > best) { best = s; bi = i; } }
        const target = Math.min(1, fb.bias + fb.gain * best);
        const a = fb.tau > 0 ? 1 - Math.exp(-dt / fb.tau) : 1;
        level += (target - level) * a;
        const [r, g, b] = colorOf('veillance', Math.min(1, Math.max(0, (level - fb.bias) / (1 - fb.bias || 1))));
        const bright = 0.25 + 0.75 * level;
        swatch.style.background = `rgb(${r * bright | 0},${g * bright | 0},${b * bright | 0})`;
        meter.firstChild.style.width = `${level * 100}%`;
        if (fb.enabled && app.io.connected) app.io.color(r * bright, g * bright, b * bright);
        if (fb.paint && best > 0.08) {
          const oc = overlay.getContext('2d'), x = (bi % 64 + 0.5) * 10, y = (Math.floor(bi / 64) + 0.5) * 10;
          oc.globalCompositeOperation = 'lighter'; oc.fillStyle = `rgba(${r},${g},${b},0.35)`; oc.beginPath(); oc.arc(x, y, 7, 0, 7); oc.fill();
        }
      }
      const pv = photo.video;
      if (lx.running && pv.videoWidth) {
        const c = exposure.getContext('2d');
        c.globalCompositeOperation = lx.mode === 'add' ? 'lighter' : 'lighten';
        c.globalAlpha = lx.mode === 'add' ? lx.gain : 1;
        c.drawImage(pv, 0, 0, 1280, 720);
        c.globalAlpha = 1; c.globalCompositeOperation = 'source-over';
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
    this._stop = () => { alive = false; under.stop(); photo.stop(); };
  },
  unmount() { this._stop?.(); },
};
