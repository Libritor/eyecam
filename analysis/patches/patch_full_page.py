"""Patch xr_stimulus.html: multi-frequency phase, colour (frequency-tagged
RGB) scan, per-frequency colours in the sweep, 40 Hz-tagged music, extended
frame rows, auto-mode audio, colour result view."""
import io, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
h = io.open("xr_stimulus.html", encoding="utf-8").read()

# 1. phase for any half-period (frames)
a = '''let flickerT0 = 0;
function flickerOnAt(ts) {
  const half = halfFrames / refreshHz;               // seconds
  return Math.floor((ts / 1000 - flickerT0 + 0.5 / refreshHz) / half) % 2 === 0;
}'''
a2 = '''let flickerT0 = 0;
function phaseOn(ts, k) {                            // k = half period in frames
  const half = k / refreshHz;                        // seconds
  return Math.floor((ts / 1000 - flickerT0 + 0.5 / refreshHz) / half) % 2 === 0;
}
const flickerOnAt = ts => phaseOn(ts, halfFrames);'''
assert a in h; h = h.replace(a, a2)

# 2. runScan: colour mode
b = '''async function runScan(m) {
  hideResult(); title.textContent = ""; sub.textContent = "";
  const tok = running = {};
  await planFlicker(m.freq);
  const gw = m.gridW, gh = m.gridH, target = m.target;'''
b2 = '''async function runScan(m) {
  if (m.color) return runColorScan(m);
  hideResult(); title.textContent = ""; sub.textContent = "";
  const tok = running = {};
  await planFlicker(m.freq);
  const gw = m.gridW, gh = m.gridH, target = m.target;'''
assert b in h; h = h.replace(b, b2)

c = '''// ---- frequency sweep: frame-exact ON blocks at several frequencies ----'''
c2 = '''// ---- colour scan: each cell's R, G, B flicker at three frequencies ----
// (frequency-tagged colour: the line at f_R in the EEG is "how red", etc.)
async function runColorScan(m) {
  hideResult(); title.textContent = ""; sub.textContent = "";
  const tok = running = {};
  const plans = [];
  for (const f of m.freqs) { await planFlicker(f); plans.push({ hz: actualHz, k: halfFrames }); }
  send({ type: "color_plan", hzs: plans.map(p => p.hz), refresh: refreshHz });
  const gw = m.gridW, gh = m.gridH, target = m.target;   // target[y][x] = [r,g,b]
  const bw = Math.min(cv.width, cv.height * gw / gh) * 0.84;
  const bh = bw * gh / gw;
  const x0 = (cv.width - bw) / 2, y0 = (cv.height - bh) / 2;
  const cwid = bw / gw, chgt = bh / gh;
  for (let gy = 0; gy < gh && running === tok; gy++) {
    const xs = [...Array(gw).keys()];
    if (gy % 2 === 1) xs.reverse();
    for (const gx of xs) {
      const [r, g, b] = target[gy][gx];
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok) {
        const ts = await frameTs(); frameCount++;
        const t = now();
        const sR = r > 0.05 && phaseOn(ts, plans[0].k);
        const sG = g > 0.05 && phaseOn(ts, plans[1].k);
        const sB = b > 0.05 && phaseOn(ts, plans[2].k);
        black();
        if (r > 0.05 || g > 0.05 || b > 0.05) {
          ctx.fillStyle = `rgb(${sR ? Math.round(255 * r) : 0},${sG ? Math.round(255 * g) : 0},${sB ? Math.round(255 * b) : 0})`;
          ctx.fillRect(x0 + gx * cwid, y0 + gy * chgt, cwid, chgt);
        }
        dot(x0 + (gx + 0.5) * cwid, y0 + (gy + 0.5) * chgt);
        send({ type: "frame", stage: "color", pt: t, gx, gy, lum: r, fl: sR ? 1 : 0,
               flG: sG ? 1 : 0, flB: sB ? 1 : 0, r, g, b });
      }
      if (running !== tok) break;
    }
    send({ type: "row", y: gy });
  }
  black();
  send({ type: "scan_done" });
}

// ---- frequency sweep: frame-exact ON blocks at several frequencies ----'''
assert c in h; h = h.replace(c, c2)

# 3. sweep: per-frequency colours
d = '''      halfFrames = plan[f].k; actualHz = plan[f].hz;
      for (const on of [true, false]) {'''
d2 = '''      halfFrames = plan[f].k; actualHz = plan[f].hz;
      const [cOn, cOff] = (m.colorsPerFreq && m.colorsPerFreq[f]) || (m.colors || ["#ffff00", "#0000ff"]);
      for (const on of [true, false]) {'''
assert d in h; h = h.replace(d, d2)
e = '''  const [cOn, cOff] = m.colors || ["#ffff00", "#0000ff"];
  const plan = {};'''
e2 = '''  const plan = {};'''
assert e in h; h = h.replace(e, e2)

# 4. audio: auto mode never blocks; music option
f_ = '''    if (audioCtx.state === "suspended") {
      sub.textContent = "TAP anywhere once to enable sound";'''
f2 = '''    if (audioCtx.state === "suspended" && location.search.includes("auto=1")) return;
    if (audioCtx.state === "suspended") {
      sub.textContent = "TAP anywhere once to enable sound";'''
assert f_ in h; h = h.replace(f_, f2)

g = '''async function runAssr(m) {
  hideResult(); title.textContent = "EAR AS A MICROPHONE"; title.className = "";
  await ensureAudio();
  const tok = running = {};'''
g2 = '''// ---- music piece tagged at 40 Hz: the melody plays through every block;
// in ON blocks its amplitude is modulated at modFreq (depth 1), in OFF
// blocks it plays unmodulated at the same mean level. The ASSR at modFreq
// is "does the ear-as-level-meter follow the tag".
function startMusic(am) {
  const notes = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 293.66, 349.23,
                 440.0, 349.23, 293.66, 246.94, 329.63, 392.0, 493.88, 587.33,
                 493.88, 392.0, 349.23, 440.0, 523.25, 440.0, 349.23, 293.66];
  const dur = 0.32; let i = 0, next = audioCtx.currentTime + 0.1; const timers = [];
  function tick() {
    while (next < audioCtx.currentTime + 0.5) {
      const o = audioCtx.createOscillator(), env = audioCtx.createGain();
      o.type = "triangle"; o.frequency.value = notes[i % notes.length];
      env.gain.setValueAtTime(0.0001, next);
      env.gain.exponentialRampToValueAtTime(0.6, next + 0.02);
      env.gain.exponentialRampToValueAtTime(0.0001, next + dur * 0.95);
      o.connect(env); env.connect(am); o.start(next); o.stop(next + dur);
      next += dur; i++;
    }
  }
  const id = setInterval(tick, 100); tick();
  return () => clearInterval(id);
}
async function runAssr(m) {
  if (m.music) return runMusic(m);
  hideResult(); title.textContent = "EAR AS A MICROPHONE"; title.className = "";
  await ensureAudio();
  const tok = running = {};'''
assert g in h; h = h.replace(g, g2)

i_ = '''function showAssrResult(m) {'''
i2 = '''async function runMusic(m) {
  hideResult(); title.textContent = "EAR AS A LEVEL METER"; title.className = "";
  await ensureAudio();
  const tok = running = {};
  const am = audioCtx.createGain(); am.gain.value = 0.5;         // mean level
  const mod = audioCtx.createOscillator(); mod.frequency.value = m.modFreq;
  const modDepth = audioCtx.createGain(); modDepth.gain.value = 0.5;
  const modGate = audioCtx.createGain(); modGate.gain.value = 0;  // 1 = tagged
  const master = audioCtx.createGain(); master.gain.value = 0.8;
  mod.connect(modDepth); modDepth.connect(modGate); modGate.connect(am.gain);
  am.connect(master); master.connect(audioCtx.destination);
  mod.start();
  const stopMusic = startMusic(am);
  black(); dot(cv.width / 2, cv.height / 2);
  for (let b = 0; b < m.blocks && running === tok; b++) {
    for (const on of [true, false]) {
      send({ type: "block", b, on, phase: "start", hz: m.modFreq });
      sub.textContent = `listen — block ${b + 1}/${m.blocks} ` + (on ? "tagged" : "plain");
      modGate.gain.setValueAtTime(on ? 1 : 0, audioCtx.currentTime);
      const t0 = now();
      while (now() - t0 < (on ? m.onS : m.offS) && running === tok) {
        await sleep(50);
        send({ type: "frame", stage: "assr", pt: now(), gx: -1, gy: -1,
               lum: on ? 1 : 0, fl: 0, hzA: m.modFreq });
      }
      send({ type: "block", b, on, phase: "end", hz: m.modFreq });
    }
  }
  stopMusic(); mod.stop();
  send({ type: "assr_done" });
}

function showColorResult(m) {
  running = null; black();
  title.textContent = "COLOUR RESULT"; title.className = "";
  sub.textContent = m.text || "";
  document.getElementById("rimg").src = m.png;
  const rt = document.getElementById("rtext");
  rt.textContent = m.text || ""; rt.className = m.good ? "ok" : "warn";
  resultBox.style.display = "flex";
}

function showAssrResult(m) {'''
assert i_ in h; h = h.replace(i_, i2)

# 5. dispatch
j = '''      case "start_sweep": runSweep(m); break;'''
j2 = '''      case "start_sweep": runSweep(m); break;
      case "color_result": showColorResult(m); break;'''
assert j in h; h = h.replace(j, j2)
io.open("xr_stimulus.html", "w", encoding="utf-8", newline="\n").write(h)
print("patched xr_stimulus.html")
