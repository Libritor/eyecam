// Fullscreen stimulus stage + frame-locked flicker clock.

// Measured display refresh rate (median frame interval over ~1 s).
export function measureRefresh(frames = 60) {
  return new Promise(resolve => {
    const ts = [];
    const step = t => {
      ts.push(t);
      if (ts.length < frames) requestAnimationFrame(step);
      else {
        const d = ts.slice(1).map((v, i) => v - ts[i]).sort((a, b) => a - b);
        resolve(1000 / d[d.length >> 1]);
      }
    };
    requestAnimationFrame(step);
  });
}

// Square-wave phase accumulator advanced in whole display frames, so the mean frequency
// is exact and the duty cycle is as even as the refresh rate allows (60 Hz / 15 Hz ->
// 2 frames on, 2 frames off).
export class FlickerClock {
  constructor(freq, refresh = 60) {
    this.freq = freq; this.refresh = refresh; this.phase = 0; this.cycles = 0; this.last = null; this.dropped = 0; this.frames = 0;
  }
  advance(tsMs) {
    if (this.last != null) {
      const fp = 1000 / this.refresh;
      const k = Math.max(1, Math.round((tsMs - this.last) / fp));
      if (k > 1 && k < 30) this.dropped += k - 1;
      const d = this.freq * k / this.refresh;
      this.cycles += d;
      this.phase = (this.phase + d) % 1;
    }
    this.last = tsMs; this.frames++;
    return this.phase < 0.5;
  }
  // Append (t, cumulative cycles) to a log every few frames; phase-locked detectors use it.
  log(rec, t, every = 3) {
    if (this.frames % every === 0) { rec.t.push(t); rec.c.push(this.cycles); }
  }
  static quality(freq, refresh) {
    const half = refresh / (2 * freq);
    return { framesPerHalf: half, exact: Math.abs(half - Math.round(half)) < 0.02 };
  }
}

export class Stage {
  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'stage';
    this.canvas = document.createElement('canvas');
    this.el.appendChild(this.canvas);
    this.hud = document.createElement('div');
    this.hud.className = 'stage-hud';
    this.el.appendChild(this.hud);
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.keys = new Set();
    this._onKey = e => { for (const fn of this.keys) fn(e); };
    this._onResize = () => this.fit();
  }
  async open({ fullscreen = true, cursor = false } = {}) {
    document.body.appendChild(this.el);
    this.el.style.cursor = cursor ? 'crosshair' : 'none';
    if (fullscreen && !document.fullscreenElement) {
      try { await this.el.requestFullscreen({ navigationUI: 'hide' }); } catch { /* windowed is fine */ }
    }
    this.fit();
    window.addEventListener('keydown', this._onKey);
    window.addEventListener('resize', this._onResize);
    this.refresh = await measureRefresh(45);
    this.fit();
    return this;
  }
  fit() {
    const dpr = window.devicePixelRatio || 1;
    this.w = this.el.clientWidth || innerWidth; this.h = this.el.clientHeight || innerHeight;
    this.canvas.width = Math.round(this.w * dpr); this.canvas.height = Math.round(this.h * dpr);
    this.canvas.style.width = this.w + 'px'; this.canvas.style.height = this.h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
  }
  onKey(fn) { this.keys.add(fn); return () => this.keys.delete(fn); }
  message(html) { this.hud.innerHTML = html || ''; this.hud.style.display = html ? 'flex' : 'none'; }
  // Frame loop: fn(tSec, tsMs) returns false to stop.
  loop(fn) {
    this.running = true;
    return new Promise(resolve => {
      const step = ts => {
        if (!this.running) return resolve();
        let r;
        try { r = fn(ts / 1000, ts); } catch (e) { console.error(e); this.running = false; return resolve(e); }
        if (r === false) { this.running = false; return resolve(); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }
  stop() { this.running = false; }
  clear(color = '#000') { this.ctx.fillStyle = color; this.ctx.fillRect(0, 0, this.w, this.h); }
  dot(x, y, r = 4, color = '#fff') { const c = this.ctx; c.fillStyle = color; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); }
  cross(x, y, s = 12, color = '#fff') {
    const c = this.ctx; c.strokeStyle = color; c.lineWidth = 2;
    c.beginPath(); c.moveTo(x - s, y); c.lineTo(x + s, y); c.moveTo(x, y - s); c.lineTo(x, y + s); c.stroke();
  }
  async close() {
    this.running = false;
    window.removeEventListener('keydown', this._onKey);
    window.removeEventListener('resize', this._onResize);
    if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch { } }
    this.el.remove();
  }
  // Resolves on the next key in `keys` (e.g. [' ', 'Escape']).
  waitKey(keys) {
    return new Promise(resolve => {
      const off = this.onKey(e => { if (keys.includes(e.key)) { e.preventDefault(); off(); resolve(e.key); } });
    });
  }
}

// Build the two flicker frames of an image: every pixel of luminance v shows v*colorA in
// phase A and v*colorB in phase B, so black stays black (no flicker) and bright areas
// flicker hardest — the paper's "flashes when over a non-black part of the image".
export function flickerFrames(img, w, h, colorA, colorB, { binary = false, threshold = 0.12, invert = false } = {}) {
  const src = document.createElement('canvas'); src.width = w; src.height = h;
  const sc = src.getContext('2d'); sc.drawImage(img, 0, 0, w, h);
  const id = sc.getImageData(0, 0, w, h), d = id.data;
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    let v = (0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2]) / 255;
    if (invert) v = 1 - v;
    if (binary) v = v > threshold ? 1 : 0;
    lum[i] = v;
  }
  const make = col => {
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const cx = c.getContext('2d'), o = cx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      o.data[i * 4] = col[0] * lum[i]; o.data[i * 4 + 1] = col[1] * lum[i]; o.data[i * 4 + 2] = col[2] * lum[i]; o.data[i * 4 + 3] = 255;
    }
    cx.putImageData(o, 0, 0);
    return c;
  };
  return { A: make(colorA), B: make(colorB), lum, w, h };
}

// Mean luminance of the image under a square (used by the simulator's "eye").
export function meanLum(frames, x0, y0, size) {
  const { lum, w, h } = frames;
  const xa = Math.max(0, Math.floor(x0)), xb = Math.min(w, Math.ceil(x0 + size));
  const ya = Math.max(0, Math.floor(y0)), yb = Math.min(h, Math.ceil(y0 + size));
  if (xb <= xa || yb <= ya) return 0;
  let s = 0, n = 0;
  const step = Math.max(1, Math.floor(size / 24));
  for (let y = ya; y < yb; y += step) for (let x = xa; x < xb; x += step) { s += lum[y * w + x]; n++; }
  return (s / n) * ((xb - xa) * (yb - ya)) / (size * size);
}

export const hexToRgb = s => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
