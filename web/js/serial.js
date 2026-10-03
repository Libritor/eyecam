// Web Serial devices: the EyeCam Arduino (RGB LED, flickering lamp, LC shutter glasses —
// firmware/eyecam_io/eyecam_io.ino) and a GRBL plotter (the Fig. 4 3D plotter).

class LineSerial {
  constructor(name) { this.name = name; this.port = null; this.lines = []; this.waiters = []; this.onLine = null; }
  static available() { return !!navigator.serial; }
  get connected() { return !!this.port; }
  async open(baudRate = 115200) {
    this.port = await navigator.serial.requestPort();
    await this.port.open({ baudRate });
    this.writer = this.port.writable.getWriter();
    this._read();
  }
  async _read() {
    const dec = new TextDecoder();
    let buf = '';
    try {
      this.reader = this.port.readable.getReader();
      for (;;) {
        const { value, done } = await this.reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.search(/\r?\n/)) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(buf[i] === '\r' ? i + 2 : i + 1);
          if (!line) continue;
          this.onLine?.(line);
          const w = this.waiters.findIndex(x => x.match(line));
          if (w >= 0) this.waiters.splice(w, 1)[0].resolve(line);
        }
      }
    } catch (e) { console.warn(this.name, 'read ended', e); }
  }
  async write(s) { if (this.writer) await this.writer.write(new TextEncoder().encode(s)); }
  waitFor(match, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const w = { match, resolve };
      this.waiters.push(w);
      setTimeout(() => { const i = this.waiters.indexOf(w); if (i >= 0) { this.waiters.splice(i, 1); reject(new Error(`${this.name}: timeout`)); } }, timeout);
    });
  }
  async close() {
    try { await this.reader?.cancel(); } catch { }
    try { this.writer?.releaseLock(); } catch { }
    try { await this.port?.close(); } catch { }
    this.port = null;
  }
}

export class EyeCamIO extends LineSerial {
  constructor() { super('Arduino'); this._last = 0; this._pendingColor = null; }
  async connect() { await this.open(115200); await new Promise(r => setTimeout(r, 1800)); await this.write('?\n'); }
  // Throttled colour updates (the paper's brain-modulated RGB LED)
  color(r, g, b) {
    this._pendingColor = [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))));
    const now = performance.now();
    if (now - this._last > 50) { this._last = now; this.write(`C ${this._pendingColor.join(' ')}\n`); }
  }
  lampFlicker(hz) { return this.write(`F ${hz}\n`); }         // hz = 0 -> lamp off
  lampSteady(on) { return this.write(`L ${on ? 1 : 0}\n`); }
  shutter(hz) { return this.write(`S ${hz}\n`); }             // hz = 0 -> glasses clear
  stopAll() { return this.write('X\n'); }
}

// GRBL: send one line, wait for ok/error (simple character-counting-free flow control).
export class Grbl extends LineSerial {
  constructor() { super('GRBL'); this.pos = null; this.state = '?'; }
  async connect() {
    await this.open(115200);
    this.onLine = line => {
      const m = line.match(/^<([^|>]+)\|(?:MPos|WPos):([-\d.]+),([-\d.]+),([-\d.]+)/);
      if (m) { this.state = m[1]; this.pos = { x: +m[2], y: +m[3], z: +m[4], t: performance.now() / 1000 }; }
    };
    await this.write('\r\n\r\n');
    await new Promise(r => setTimeout(r, 2000));
    this._poll = setInterval(() => this.write('?'), 100);
  }
  async send(line) {
    const p = this.waitFor(l => l === 'ok' || l.startsWith('error') || l.startsWith('ALARM'), 120000);
    await this.write(line + '\n');
    const r = await p;
    if (r !== 'ok') throw new Error(`GRBL ${r} on "${line}"`);
  }
  async stream(lines, onProgress) {
    this.abort = false;
    for (let i = 0; i < lines.length && !this.abort; i++) { await this.send(lines[i]); onProgress?.(i + 1, lines.length); }
  }
  async feedHold() { this.abort = true; await this.write('!'); }
  async reset() { this.abort = true; await this.write('\x18'); }
  async close() { clearInterval(this._poll); await super.close(); }
}

// Plotter path for the paper's Fig. 4 experiment: vertical passes at `vy` mm/s while the
// display retreats from the face at `vz` mm/s, from `z0` to `z1` mm.
export function plotterPath({ z0 = 40, z1 = 210, yLo = -50, yHi = 50, vy = 5, vz = 0.5, axisAway = 'X', axisVert = 'Z' } = {}) {
  const lines = ['G21', 'G90', 'G94'];
  const moves = []; // planned segments for the virtual plotter: {dy, dz, dur}
  let z = z0, y = yLo, up = true;
  lines.push(`G0 ${axisAway}${z0.toFixed(2)} ${axisVert}${yLo.toFixed(2)}`);
  const L = yHi - yLo;
  while (z < z1 - 1e-6) {
    const dz = Math.min(z1 - z, L * vz / vy);
    const ny = up ? yHi : yLo, dur = Math.abs(ny - y) / vy;
    const feed = Math.hypot(ny - y, dz) / dur * 60;
    lines.push(`G1 ${axisAway}${(z + dz).toFixed(3)} ${axisVert}${ny.toFixed(3)} F${feed.toFixed(1)}`);
    moves.push({ y0: y, y1: ny, z0: z, z1: z + dz, dur });
    z += dz; y = ny; up = !up;
  }
  return { lines, moves, duration: moves.reduce((s, m) => s + m.dur, 0) };
}

// Position along the planned path after `t` seconds (virtual plotter).
export function plannedPosition(moves, t) {
  for (const m of moves) {
    if (t <= m.dur) { const a = t / m.dur; return { y: m.y0 + (m.y1 - m.y0) * a, z: m.z0 + (m.z1 - m.z0) * a, done: false }; }
    t -= m.dur;
  }
  const m = moves[moves.length - 1];
  return { y: m.y1, z: m.z1, done: true };
}
