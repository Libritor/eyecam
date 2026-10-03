// EEG sources: Muse over Web Bluetooth, the Python LSL bridge, and a physiologically
// plausible simulator whose SSVEP depends on what the running experiment is displaying.

import { perfNow, epochToPerf } from './eeg-store.js';

// ================================================================ Muse (Web Bluetooth)
// Two protocols, detected from the GATT table on connect (as muselsl does):
//   legacy  — Muse 2016 / Muse 2 / Muse S gen 1: one characteristic per channel, 12-bit
//   athena  — Muse S Athena (gen 3): multiplexed DATA_1/DATA_2 packets, 14-bit EEG
//             (decoder ported from muselsl 2.5 athena.py / BrainFlow PR #779)
const MUSE_SERVICE = 0xfe8d;
const U = s => `273e${s}-4c4d-454d-96be-f03bac821358`;
const CONTROL = U('0001');
const TELEMETRY = U('000b');
const EEG_UUIDS = { TP9: U('0003'), AF7: U('0004'), AF8: U('0005'), TP10: U('0006'), AUX: U('0007') };
const ATHENA_DATA = [U('0013'), U('0014')];
const ATHENA_PRESET = 'p1041';
const ATHENA_EEG_SCALE = 1450 / 16383;
// tag -> [type, channels, samples, payload bytes, variable length]
const ATHENA_TAGS = {
  0x11: ['eeg', 4, 4, 28, false], 0x12: ['eeg', 8, 2, 28, false],
  0x34: ['optics', 4, 3, 30, false], 0x35: ['optics', 8, 2, 40, false], 0x36: ['optics', 16, 1, 40, false],
  0x47: ['acc_gyro', 6, 3, 36, false], 0x53: ['unknown', 2, 6, 24, false],
  0x88: ['battery', 1, 1, 0, true], 0x98: ['battery', 1, 1, 20, false],
};

function encodeCommand(cmd) {
  const b = new TextEncoder().encode(`X${cmd}\n`);
  b[0] = b.length - 1;
  return b;
}

function decodeEEG(dv) {
  const seq = dv.getUint16(0);
  const out = new Float32Array(12);
  for (let i = 0, k = 0; i < 18; i += 3) {
    const b0 = dv.getUint8(2 + i), b1 = dv.getUint8(3 + i), b2 = dv.getUint8(4 + i);
    out[k++] = 0.48828125 * (((b0 << 4) | (b1 >> 4)) - 0x800);
    out[k++] = 0.48828125 * ((((b1 & 0xf) << 8) | b2) - 0x800);
  }
  return { seq, samples: out };
}

// LSB-first bit field (BrainFlow custom_cast.h::extract_lsb_bits)
function lsbBits(u8, start, width) {
  let v = 0;
  for (let b = 0; b < width; b++) {
    const a = start + b;
    if ((u8[a >> 3] >> (a & 7)) & 1) v |= 1 << b;
  }
  return v;
}

// One Athena BLE notification -> [{type:'eeg', rows: [[ch...] per sample]} | {type:'battery', pct}]
export function decodeAthena(u8) {
  const blocks = [];
  let off = 0;
  while (u8.length - off >= 14) {
    const len = u8[off];
    if (len < 14 || off + len > u8.length) break;
    const pkt = u8.subarray(off, off + len);
    off += len;
    const payload = pkt.subarray(14);
    const emit = (tag, data) => {
      const cfg = ATHENA_TAGS[tag];
      if (!cfg) return;
      const [type, nch, ns] = cfg;
      if (type === 'eeg') {
        const rows = [];
        for (let s = 0; s < ns; s++) {
          const row = [];
          for (let c = 0; c < nch; c++) row.push((lsbBits(data, (s * nch + c) * 14, 14) - 8192) * ATHENA_EEG_SCALE);
          rows.push(row);
        }
        blocks.push({ type, rows });
      } else if (type === 'battery' && data.length >= 2) blocks.push({ type, pct: (data[0] | (data[1] << 8)) / 512 });
    };
    let p;
    const primary = ATHENA_TAGS[pkt[9]];
    if (primary) {
      const n = Math.min(primary[4] ? payload.length : primary[3], payload.length);
      emit(pkt[9], payload.subarray(0, n));
      p = n;
    } else p = payload.length;
    while (p + 5 <= payload.length) {
      const tag = payload[p], cfg = ATHENA_TAGS[tag];
      if (!cfg) break;
      const rem = payload.length - p - 5, n = cfg[4] ? rem : cfg[3];
      if (n <= 0 || n > rem) break;
      emit(tag, payload.subarray(p + 5, p + 5 + n));
      p += 5 + n;
    }
  }
  return blocks;
}

const sleepMs = ms => new Promise(r => setTimeout(r, ms));

export class MuseBLE {
  constructor(store, { aux = false, onStatus = () => {} } = {}) {
    this.store = store; this.aux = aux; this.onStatus = onStatus;
    this.kind = 'muse';
    this.battery = null;
    this.protocol = null;
    this.subs = [];
  }

  static available() { return !!navigator.bluetooth; }

  async connect() {
    this.device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [MUSE_SERVICE] }, { namePrefix: 'Muse' }],
      optionalServices: [MUSE_SERVICE],
    });
    this._onGattDisc ||= () => this._onDisconnect();
    this.device.addEventListener('gattserverdisconnected', this._onGattDisc);
    this.wanted = false; // auto-reconnect only once a first setup has succeeded
    try {
      await this._setup();
    } catch (e) {
      try { this.device.gatt.disconnect(); } catch { }
      throw e;
    }
    this.wanted = true;
  }

  _listen(ch, fn) {
    ch.addEventListener('characteristicvaluechanged', fn);
    this.subs.push([ch, fn]);
  }

  async _setup() {
    this.onStatus(`connecting to ${this.device.name}…`);
    for (const [ch, fn] of this.subs) ch.removeEventListener('characteristicvaluechanged', fn);
    this.subs = [];
    // Pick the protocol from the characteristics actually present. Windows can hand back
    // an incomplete (cached) GATT table on the first connection, so rediscover once on a
    // fresh connection before giving up, and report what was seen.
    let svc, byId;
    for (let attempt = 1; ; attempt++) {
      const server = await this.device.gatt.connect();
      svc = await server.getPrimaryService(MUSE_SERVICE);
      const chars = await svc.getCharacteristics();
      byId = new Map(chars.map(c => [c.uuid.toLowerCase(), c]));
      this.gattSeen = chars.map(c => c.uuid.slice(4, 8)).sort();
      console.info('Muse GATT characteristics:', this.gattSeen.join(' '));
      if (byId.has(ATHENA_DATA[0]) || byId.has(EEG_UUIDS.TP9)) break;
      // Only the control channel? Ask the headset what it is running: a Muse in its
      // bootloader (firmware-update mode) exposes no EEG characteristics at all.
      if (byId.has(CONTROL)) {
        const info = await this._queryVersion(byId.get(CONTROL));
        if (/"ap"\s*:\s*"bootloader"/.test(info)) {
          const bp = info.match(/"bp"\s*:\s*(\d+)/);
          throw new Error(`This Muse is in its BOOTLOADER (firmware-update mode${bp ? `, battery ${bp[1]} %` : ''}), so it has no EEG channels. ` +
            'Charge it fully, then open the official Muse app and let it finish/restore the firmware update; afterwards connect again.');
        }
      }
      if (attempt >= 3) {
        throw new Error(`Muse exposes neither Athena (273e0013) nor Muse 2 (273e0003) EEG characteristics — saw only: ${this.gattSeen.join(', ') || 'none'}. ` +
          'Probably a stale Windows Bluetooth cache: remove the Muse in Windows Settings > Bluetooth & devices (and in chrome://settings/content/bluetoothDevices), power-cycle it, then connect again — or use the Python bridge.');
      }
      this.onStatus(`incomplete Bluetooth service table (saw ${this.gattSeen.length} characteristics) — rediscovering (${attempt}/2)…`);
      try { this.device.gatt.disconnect(); } catch { }
      await sleepMs(1500);
    }
    this.control = byId.get(CONTROL) || await svc.getCharacteristic(CONTROL);
    this._offsets = []; this.packets = 0; this.lost = 0; this._lastEmitted = null;
    if (byId.has(ATHENA_DATA[0])) await this._setupAthena(ATHENA_DATA.map(u => byId.get(u)).filter(Boolean));
    else await this._setupLegacy(svc);
    this.onStatus(`streaming from ${this.device.name} (${this.protocol === 'athena' ? 'Muse S Athena' : 'Muse 2 / S gen 1'})`);
  }

  async _setupLegacy(svc) {
    this.protocol = 'legacy';
    this.channels = this.aux ? ['TP9', 'AF7', 'AF8', 'TP10', 'AUX'] : ['TP9', 'AF7', 'AF8', 'TP10'];
    this.store.reset(256, this.channels);
    this._pending = new Map();
    this._seqBase = 0; this._lastSeq = -1;
    for (const [i, name] of this.channels.entries()) {
      const ch = await svc.getCharacteristic(EEG_UUIDS[name]);
      this._listen(ch, e => this._onEEG(i, e.target.value));
      await ch.startNotifications();
    }
    try {
      const tel = await svc.getCharacteristic(TELEMETRY);
      this._listen(tel, e => { this.battery = e.target.value.getUint16(2) / 512; });
      await tel.startNotifications();
    } catch { /* optional */ }
    await this._cmd('h');
    await this._cmd(this.aux ? 'p20' : 'p21');
    await this._cmd('s');
    await this._cmd('d');
    clearInterval(this._keep);
    this._keep = setInterval(() => this._cmd('k').catch(() => {}), 5000);
  }

  // Same command sequence as muselsl Athena.connect() + start()
  async _setupAthena(dataChars) {
    this.protocol = 'athena';
    this.channels = ['TP9', 'AF7', 'AF8', 'TP10'];
    this.store.reset(256, this.channels);
    this._sampleIdx = 0;
    try { this._listen(this.control, () => {}); await this.control.startNotifications(); } catch { }
    for (const ch of dataChars) {
      this._listen(ch, e => this._onAthena(e.target.value));
      await ch.startNotifications();
    }
    for (const c of ['v6', 's', 'h', ATHENA_PRESET, 's']) { await this._cmd(c); await sleepMs(200); }
    await this._cmd('dc001'); await sleepMs(50);
    await this._cmd('dc001'); await sleepMs(100);
    await this._cmd('L1'); await sleepMs(300);
    await this._cmd('s'); await sleepMs(200);
    clearInterval(this._keep);
  }

  // Send 'v6' and collect the JSON the control channel answers with (≤ 1.5 s).
  async _queryVersion(ctrl) {
    let txt = '';
    const fn = e => { const d = new Uint8Array(e.target.value.buffer); txt += new TextDecoder().decode(d.subarray(1, 1 + d[0])); };
    try {
      ctrl.addEventListener('characteristicvaluechanged', fn);
      await ctrl.startNotifications();
      this.control = ctrl;
      await this._cmd('v6');
      await sleepMs(1500);
    } catch { /* best effort */ } finally { ctrl.removeEventListener('characteristicvaluechanged', fn); }
    console.info('Muse control reply:', txt);
    return txt;
  }

  async _cmd(c) {
    const b = encodeCommand(c);
    if (this.control.properties?.writeWithoutResponse && this.control.writeValueWithoutResponse) await this.control.writeValueWithoutResponse(b);
    else await this.control.writeValue(b);
  }

  _onAthena(dv) {
    const arrival = perfNow();
    let blocks;
    try { blocks = decodeAthena(new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength)); } catch (e) { console.warn('athena decode', e); return; }
    for (const b of blocks) {
      if (b.type === 'battery') { this.battery = b.pct; continue; }
      const n = b.rows.length, first = this._sampleIdx;
      this._sampleIdx += n;
      this.packets++;
      const off = this._clock(arrival, first + n - 1);
      this.store.push(b.rows.map((_, k) => off + (first + k) / 256), b.rows.map(r => r.slice(0, 4)));
    }
  }

  // Clock mapping: a packet cannot arrive before its last sample was taken, so the
  // minimum of (arrival - sampleTime) over a sliding 20 s window tracks the offset
  // (and slow crystal drift) with only positive BLE latency jitter removed.
  _clock(arrival, lastIdx) {
    const cand = arrival - lastIdx / 256;
    this._offsets.push([arrival, cand]);
    while (this._offsets.length && this._offsets[0][0] < arrival - 20) this._offsets.shift();
    let off = Infinity; for (const [, c] of this._offsets) if (c < off) off = c;
    return off;
  }

  _unwrap(seq) {
    if (this._lastSeq >= 0 && seq < this._lastSeq - 30000) this._seqBase += 65536;
    if (this._lastSeq >= 0 && seq > this._lastSeq + 30000) return this._seqBase - 65536 + seq; // late straggler
    this._lastSeq = Math.max(this._lastSeq, seq);
    return this._seqBase + seq;
  }

  _onEEG(chIndex, dv) {
    const arrival = perfNow();
    const { seq, samples } = decodeEEG(dv);
    const g = this._unwrap(seq);
    let p = this._pending.get(g);
    if (!p) { p = { rows: new Array(this.channels.length).fill(null), arrival, count: 0 }; this._pending.set(g, p); }
    if (!p.rows[chIndex]) p.count++;
    p.rows[chIndex] = samples;
    p.arrival = Math.min(p.arrival, arrival);
    // flush complete packets in order; incomplete ones once 4 newer packets exist
    const keys = [...this._pending.keys()].sort((a, b) => a - b);
    for (const k of keys) {
      const q = this._pending.get(k);
      const newest = keys[keys.length - 1];
      if (q.count < this.channels.length && newest - k < 4) break;
      this._pending.delete(k);
      this._emit(k, q);
    }
  }

  _emit(g, q) {
    this.packets++;
    if (this._lastEmitted != null && g > this._lastEmitted + 1) this.lost += g - this._lastEmitted - 1;
    this._lastEmitted = g;
    const off = this._clock(q.arrival, g * 12 + 11);
    const times = [], rows = [];
    for (let k = 0; k < 12; k++) {
      times.push(off + (g * 12 + k) / 256);
      rows.push(q.rows.map(r => (r ? r[k] : NaN)));
    }
    this.store.push(times, rows);
  }

  async _onDisconnect() {
    clearInterval(this._keep);
    if (!this.wanted) return;
    this.onStatus('Muse disconnected — reconnecting…');
    for (let attempt = 1; this.wanted && attempt <= 20; attempt++) {
      try { await this._setup(); return; } catch (e) {
        this.onStatus(`reconnect attempt ${attempt} failed (${e.message}); retrying…`);
        await sleepMs(1500);
      }
    }
    this.onStatus('Muse lost — click Connect Muse again');
  }

  async disconnect() {
    this.wanted = false;
    clearInterval(this._keep);
    try { await this._cmd('h'); } catch { }
    try { this.device?.gatt.disconnect(); } catch { }
  }

  info() {
    return `${this.device?.name || 'Muse'}${this.protocol === 'athena' ? ' (Athena)' : ''} · ${this.channels?.join(' ') || ''}` +
      (this.battery != null ? ` · battery ${this.battery.toFixed(0)}%` : '') +
      (this.lost ? ` · lost ${this.lost} pkts` : '');
  }
}

// ================================================================ Python LSL bridge
export class BridgeSource {
  constructor(store, { url = `ws://${location.hostname || 'localhost'}:8766`, onStatus = () => {}, onGaze = () => {} } = {}) {
    this.store = store; this.url = url; this.onStatus = onStatus; this.onGaze = onGaze;
    this.kind = 'bridge';
    this.remote = null;
  }
  connect() {
    return new Promise((resolve, reject) => {
      this.wanted = true;
      let first = true;
      const open = () => {
        const ws = new WebSocket(this.url);
        this.ws = ws;
        ws.onopen = () => { this.onStatus('bridge connected'); if (first) { first = false; resolve(); } };
        ws.onerror = () => { if (first) { first = false; reject(new Error(`cannot reach ${this.url} — start: python server.py --muse`)); } };
        ws.onclose = () => { if (this.wanted && !first) { this.onStatus('bridge closed — retrying'); setTimeout(open, 2000); } };
        ws.onmessage = ev => this._msg(JSON.parse(ev.data));
      };
      open();
    });
  }
  _msg(m) {
    if (m.type === 'info') {
      this.remote = m;
      if (m.fs && m.channels?.length) this.store.reset(m.fs, m.channels);
      this.onStatus(`bridge: ${m.source} ${m.channels?.join(' ') || ''}`);
    } else if (m.type === 'eeg') {
      this.store.push(m.t.map(epochToPerf), m.x);
    } else if (m.type === 'gaze') {
      this.onGaze(m.t.map(epochToPerf), m.xy);
    } else if (m.type === 'status') this.onStatus('bridge: ' + m.msg);
  }
  disconnect() { this.wanted = false; this.ws?.close(); }
  info() { return this.remote ? `${this.remote.source} · ${this.remote.channels.join(' ')}` : 'bridge (waiting for stream)'; }
}

// ================================================================ Simulator
export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Incremental EEG generator. drive(t) -> {s: 0..1 stimulus strength, f: Hz} (or null).
export class EEGGenerator {
  // muse: true adds the artifact profile measured on a real Muse recording — muscle bursts
  // on TP9/TP10 (temporalis), electrode shifts, blinks reaching every channel through the
  // forehead (FPz) reference, and noise common to all channels from that reference.
  constructor({ fs = 256, seed = 1, channels = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX'], ssvepUV = 4, noiseUV = 9, latency = 0.1, phaseJitter = 0, muse = false, refNoiseUV = 7 } = {}) {
    this.fs = fs; this.channels = channels; this.rand = mulberry32(seed); this.phaseJitter = phaseJitter;
    this.muse = muse; this.refNoiseUV = refNoiseUV; this.refPink = [0, 0, 0, 0, 0, 0, 0];
    this.emg = channels.map(() => ({ left: 0, amp: 0, hp: 0, prev: 0 })); this.step = channels.map(() => 0);
    this.forced = null;
    this.ssvepUV = ssvepUV; this.noiseUV = noiseUV; this.latency = latency;
    this.gain = channels.map(n => ({ TP9: 0.6, TP10: 0.6, AUX: 1.0, AF7: 0.15, AF8: 0.15 }[n] ?? 0.5));
    this.pink = channels.map(() => [0, 0, 0, 0, 0, 0, 0]);
    this.phase = 0; this.phase2 = Math.PI / 3; this.alphaPh = 0; this.alphaEnv = 0.5;
    this.blinkT = 3; this.blinkLeft = 0; this.history = [];
    this.i = 0;
  }
  gauss() { let u = 0, v = 0; while (!u) u = this.rand(); v = this.rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  pinkStep(c) { // Paul Kellet's refined pink filter
    const w = this.gauss(), b = this.pink[c];
    b[0] = 0.99886 * b[0] + w * 0.0555179; b[1] = 0.99332 * b[1] + w * 0.0750759;
    b[2] = 0.96900 * b[2] + w * 0.1538520; b[3] = 0.86650 * b[3] + w * 0.3104856;
    b[4] = 0.55000 * b[4] + w * 0.5329522; b[5] = -0.7616 * b[5] - w * 0.0168980;
    const out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362; b[6] = w * 0.115926;
    return out * 0.25;
  }
  // Force an artifact for `dur` seconds (calibration prompts in the simulator):
  // 'blink' | 'clench' | 'swallow' | 'move' | 'rest'
  inject(type, dur) { this.forced = { type, left: dur }; }
  pinkRef() {
    const w = this.gauss(), b = this.refPink;
    b[0] = 0.99886 * b[0] + w * 0.0555179; b[1] = 0.99332 * b[1] + w * 0.0750759;
    b[2] = 0.96900 * b[2] + w * 0.1538520; b[3] = 0.86650 * b[3] + w * 0.3104856;
    b[4] = 0.55000 * b[4] + w * 0.5329522; b[5] = -0.7616 * b[5] - w * 0.0168980;
    const out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362; b[6] = w * 0.115926;
    return out * 0.25;
  }
  // Generate one sample at time t using drive state from `latency` seconds earlier.
  sample(t, drive) {
    const dt = 1 / this.fs;
    const d = drive || { s: 0, f: 15 };
    const jit = this.phaseJitter ? this.phaseJitter * Math.sqrt(dt) * this.gauss() : 0;
    this.phase += 2 * Math.PI * d.f * dt + jit;
    this.phase2 += 4 * Math.PI * d.f * dt + 2 * jit;
    this.alphaPh += 2 * Math.PI * 10.2 * dt;
    this.alphaEnv += (this.rand() - 0.5) * 0.02; this.alphaEnv = Math.min(1, Math.max(0.1, this.alphaEnv));
    const fgain = 0.55 + 0.45 * Math.exp(-(((d.f - 15) / 6) ** 2));
    const ssvep = this.ssvepUV * fgain * d.s * (Math.sin(this.phase) + 0.45 * Math.sin(this.phase2));
    const alpha = 5 * this.alphaEnv * Math.sin(this.alphaPh);
    const line = 1.5 * Math.sin(2 * Math.PI * 60 * t);
    const F = this.forced;
    if (F) { F.left -= dt; if (F.left <= 0) this.forced = null; }
    this.blinkT -= dt;
    if (F?.type === 'blink' && this.blinkLeft <= 0 && this.blinkT > 0.7) this.blinkT = 0.7;
    if (this.blinkT <= 0) { this.blinkLeft = 0.3; this.blinkT = F?.type === 'blink' ? 0.9 : (this.muse ? 2 + this.rand() * 9 : 3 + this.rand() * 6); }
    let blink = 0;
    if (this.blinkLeft > 0) { blink = 140 * Math.sin(Math.PI * (0.3 - this.blinkLeft) / 0.3); this.blinkLeft -= dt; }
    const ref = this.muse ? this.refNoiseUV * this.pinkRef() : 0;
    const row = new Array(this.channels.length);
    for (let c = 0; c < row.length; c++) {
      const n = this.channels[c];
      const frontal = n === 'AF7' || n === 'AF8', temporal = n === 'TP9' || n === 'TP10';
      let art = 0;
      if (this.muse) {
        // muscle bursts (temporalis under TP9/TP10): broadband, high-passed white noise
        const e = this.emg[c];
        const wantEmg = F && (F.type === 'clench' || F.type === 'swallow') && (temporal || F.type === 'clench');
        if (e.left <= 0 && (wantEmg || (temporal && this.rand() < dt / 9))) { e.left = wantEmg ? Math.max(F.left, 0.3) : 0.4 + this.rand() * 2; e.amp = (temporal ? 30 : 15) * (0.5 + this.rand()); }
        if (e.left > 0) { const wv = this.gauss() * e.amp; e.hp = 0.6 * (e.hp + wv - e.prev); e.prev = wv; art += e.hp; e.left -= dt; }
        // electrode shifts / head movement: decaying steps
        const wantMove = F && (F.type === 'move' || (F.type === 'swallow' && this.rand() < dt * 2));
        if ((wantMove && this.rand() < dt * 3) || this.rand() < dt / 45) this.step[c] += (this.rand() < 0.5 ? -1 : 1) * (80 + this.rand() * 250);
        this.step[c] *= Math.exp(-dt / 1.2);
        art += this.step[c];
      }
      // blinks reach every channel through the forehead reference (inverted on TP)
      const blinkGain = frontal ? 1 : this.muse ? -0.35 : 0.08;
      row[c] = this.noiseUV * this.pinkStep(c) + this.gain[c] * ssvep + (frontal ? 0.3 : 1) * alpha + line
        + blink * blinkGain + art - ref;
    }
    this.i++;
    return row;
  }
}

// Offline synthesis (tests / demos): probe(t) -> {s, f}
export function synthesize(t0, duration, probe, opt = {}) {
  const gen = new EEGGenerator(opt);
  const fs = gen.fs, n = Math.floor(duration * fs);
  const times = new Float64Array(n), data = gen.channels.map(() => new Float32Array(n));
  const events = (opt.events || []).slice().sort((a, b) => a.t - b.t); // [{t, type, dur}] -> gen.inject
  let ev = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fs;
    times[i] = t;
    while (ev < events.length && events[ev].t <= t) { gen.inject(events[ev].type, events[ev].dur); ev++; }
    const row = gen.sample(t, probe(t - gen.latency));
    for (let c = 0; c < row.length; c++) data[c][i] = row[c];
  }
  return { fs, channels: gen.channels.slice(), n, times, data };
}

export class Simulator {
  constructor(store, { onStatus = () => {} } = {}) {
    this.store = store; this.onStatus = onStatus; this.kind = 'sim';
    this.probe = null;           // set by the running experiment: t -> {s, f}
    this.gen = new EEGGenerator({ seed: (Math.random() * 1e9) | 0, muse: true, ssvepUV: 3 });
    this.hist = [];
  }
  setProbe(fn) { this.probe = fn; }
  connect() {
    this.store.reset(256, this.gen.channels);
    this.t = perfNow();
    this.timer = setInterval(() => this._tick(), 40);
    this.onStatus('simulator running (SSVEP follows the stimulus on screen)');
    return Promise.resolve();
  }
  _tick() {
    const now = perfNow();
    const st = this.probe ? this.probe(now) : null;
    this.hist.push([now, st]);
    while (this.hist.length > 2 && this.hist[1][0] < now - 1) this.hist.shift();
    const times = [], rows = [];
    while (this.t + 1 / 256 <= now) {
      this.t += 1 / 256;
      // drive state from `latency` ago
      const target = this.t - this.gen.latency;
      let d = null;
      for (const [ht, hs] of this.hist) { if (ht <= target) d = hs; else break; }
      times.push(this.t); rows.push(this.gen.sample(this.t, d));
    }
    if (now - this.t > 2) this.t = now; // tab was throttled
    this.store.push(times, rows);
  }
  disconnect() { clearInterval(this.timer); }
  info() { return 'simulator · TP9 AF7 AF8 TP10 AUX(Oz)'; }
}
