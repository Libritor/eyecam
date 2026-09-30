"""Patch: --mode sweep (frequency sweep with exact-line detector + exact permutation)."""
import io, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

x = io.open("xr_session.py", encoding="utf-8").read()
if "\nimport math\n" not in x:
    x = x.replace("\nimport numpy as np\n", "\nimport math\nimport numpy as np\n", 1)

a = '''        elif typ == "block":
            if m["phase"] == "start":
                self._block_t0 = t
            else:
                self.blocks.append(("on" if m["on"] else "off",
                                    self._block_t0, t))'''
a2 = a + '''
                self.block_hz.append(m.get("hz"))'''
assert a in x; x = x.replace(a, a2)

b = '''        self.blocks = []
        self.frames = 0'''
b2 = '''        self.blocks = []
        self.block_hz = []          # per-block flicker frequency (sweep)
        self.frames = 0'''
assert b in x; x = x.replace(b, b2)

c = '''                              "assr_result", "arm", "start_alpha",
                              "alpha_result"):'''
c2 = '''                              "assr_result", "arm", "start_alpha",
                              "alpha_result", "start_sweep", "sweep_result"):'''
assert c in x; x = x.replace(c, c2)

d = '''            "start_calib", "start_scan", "start_assr", "start_alpha")'''
d2 = '''            "start_calib", "start_scan", "start_assr", "start_alpha",
            "start_sweep")'''
assert d in x; x = x.replace(d, d2)

e = '''    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha"],'''
e2 = '''    ap.add_argument("--sweep", default="7.5,10,12,15,20",
                    help="--mode sweep: flicker frequencies to test (Hz)")
    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",
                                       "sweep"],'''
assert e in x; x = x.replace(e, e2)

f_ = '''            if a.mode == "alpha":
                return await self.run_alpha(result, recorder)
'''
f2 = f_ + '''            if a.mode == "sweep":
                return await self.run_sweep(result, recorder)
'''
assert f_ in x; x = x.replace(f_, f2)

g = '''    async def run_alpha(self, result, recorder):'''
g2 = '''    async def run_sweep(self, result, recorder):
        """Frequency sweep: which flicker frequency (if any) evokes an SSVEP
        in THIS subject with THIS electrode? Frame-exact ON blocks at each
        frequency interleaved with OFF blocks, repeated; each frequency is
        scored with the exact-line detector and an exact permutation test."""
        a = self.args
        freqs = [float(v) for v in str(a.sweep).split(",") if v.strip()]
        self.spectate(type="stage", stage="sweep",
                      detail=",".join(f"{f:g}" for f in freqs))
        self.blocks, self.block_hz = [], []
        self.open_log("sweep_log.csv")
        colors = ["#ffffff", "#000000"] if a.calib_style == "bw" \\
            else ["#ffff00", "#0000ff"]
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.calib_blocks,
                        onS=a.calib_on, offS=a.calib_off, colors=colors,
                        size=a.calib_size)
        total = a.calib_blocks * len(freqs) * (a.calib_on + a.calib_off)
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        g0 = self.delivery_check("sweep_log.csv", total)
        result["g0"] = g0
        if not g0["ok"]:
            msg = "STIMULUS NOT DELIVERED: " + g0["reason"]
            print(msg)
            await self.send(cmd="msg", text=msg)
            result.update(ok=False, error="G0 " + g0["reason"])
            recorder.stop()
            await asyncio.sleep(a.linger)
            return 2
        res = score_sweep(self.session, self.blocks, self.block_hz)
        res["blocks"] = [list(b) + [h] for b, h in zip(self.blocks, self.block_hz)]
        json.dump(res, open(os.path.join(self.session, "sweep.json"), "w"),
                  indent=1)
        lines = [f"{r['freq']:g} Hz (delivered {r['delivered']:.2f}): best "
                 f"{r['best']} stat {r['stat']:+.2f} p={r['p']:.4f} "
                 f"(n_on {r['n_on']}, n_off {r['n_off']})"
                 for r in res["results"].values()]
        table = "   |   ".join(lines)
        if res["passed"]:
            verdict = (f"SSVEP at {res['best_freq']:g} Hz on "
                       f"{res['best_channel']} (p={res['p_min']:.4f})")
        else:
            verdict = (f"no frequency passes (min p={res.get('p_min', float('nan')):.3f}"
                       f" at {res.get('best_freq')} Hz)")
        print("SWEEP " + verdict)
        for line in lines:
            print("  " + line)
        result.update(ok=True, sweep=dict(passed=res["passed"],
                                          best_freq=res["best_freq"],
                                          p_min=res.get("p_min")))
        await self.send(cmd="sweep_result", passed=res["passed"],
                        verdict=verdict, table=table)
        self.spectate(type="log", msg="sweep " + verdict)
        recorder.stop()
        print(f"SWEEP SESSION DONE passed={res['passed']} -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_alpha(self, result, recorder):'''
assert g in x; x = x.replace(g, g2)

h_ = '''def score_alpha(session, blocks, band=(8.0, 12.0)):'''
scorer = '''def score_sweep(session, blocks, hzs, log_name="sweep_log.csv"):
    """Per frequency: exact-line SNR (f and 2f, whole-block periodogram) of
    the ON blocks at that frequency vs ALL OFF blocks scored at the same
    frequency; max-over-channels mean log-SNR difference, exact permutation.
    All segments are cut to one common length so ON and OFF are exchangeable."""
    from itertools import combinations
    from scipy.signal import periodogram
    path = os.path.join(session, "eeg.csv")
    names = csv_header(path) or CH_NAMES
    E = np.genfromtxt(path, delimiter=",", skip_header=1)
    et = E[:, 0]
    fs = reconstruct.infer_fs(et)
    L = np.genfromtxt(os.path.join(session, log_name), delimiter=",",
                      skip_header=1)
    lt, lfl, lpt = L[:, 0], L[:, 4], L[:, 5]

    def delivered(t0, t1, fallback):
        s = (lt >= t0) & (lt <= t1)
        fl, pt = lfl[s], lpt[s]
        ed = np.where((fl[1:] == 1) & (fl[:-1] == 0))[0] + 1
        per = np.diff(pt[ed])
        per = per[(per > 0) & (per < 1)]
        return float(1 / np.median(per)) if len(per) >= 5 else float(fallback)

    def line_snr(seg, f0, half=0.15, flank=(0.5, 2.0)):
        seg = seg - seg.mean()
        f, p = periodogram(seg, fs=fs, window="hann", detrend="constant")
        pk = p[np.abs(f - f0) <= half].mean()
        fk = p[(np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])].mean()
        return pk / max(fk, 1e-12)

    segs = []
    for (kind, t0, t1), hz in zip(blocks, hzs):
        sel = (et >= t0 + 1.0) & (et <= t1)
        if sel.sum() < fs * 3:
            continue
        fdel = delivered(t0, t1, hz) if kind == "on" else None
        segs.append((kind, round(float(hz), 2) if hz else None, fdel,
                     E[sel, 1:]))
    if not segs:
        return dict(names=names, freqs=[], results={}, passed=False,
                    best_freq=None, best_channel=None, p_min=float("nan"))
    n_min = min(s[3].shape[0] for s in segs)
    segs = [(k, hz, fd, X[:n_min]) for k, hz, fd, X in segs]
    stds = np.array([[s[3][:, c].std() for c in range(len(names))] for s in segs])
    keep = stds <= 4 * np.median(stds, 0)
    freqs = sorted(set(s[1] for s in segs if s[0] == "on" and s[1]))
    out = {}
    for f in freqs:
        rows = []
        for j, (kind, hz, fdel, X) in enumerate(segs):
            if kind == "on" and hz != f:
                continue
            f0 = fdel if kind == "on" else f
            v = np.zeros(len(names))
            for c in range(len(names)):
                if keep[j, c]:
                    v[c] = (np.log(line_snr(X[:, c], f0))
                            + np.log(line_snr(X[:, c], 2 * f0)))
            rows.append((kind == "on", v))
        on = np.array([r[0] for r in rows])
        V = np.array([r[1] for r in rows])
        if on.sum() < 2 or (~on).sum() < 2:
            continue
        diff = V[on].mean(0) - V[~on].mean(0)
        best_c = int(np.argmax(diff))
        obs = float(diff[best_c])
        n, n_on = len(on), int(on.sum())
        null = []
        if math.comb(n, n_on) <= 200000:
            for cmb in combinations(range(n), n_on):
                m = np.zeros(n, bool)
                m[list(cmb)] = True
                null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
        else:
            rng = np.random.default_rng(0)
            for _ in range(20000):
                m = np.zeros(n, bool)
                m[rng.choice(n, n_on, replace=False)] = True
                null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
        null = np.array(null)
        p = float(np.mean(null >= obs - 1e-12))
        per_ch = {names[c]: dict(on_med=float(np.exp(np.median(V[on, c]))),
                                 off_med=float(np.exp(np.median(V[~on, c]))),
                                 diff=float(diff[c]))
                  for c in range(len(names))}
        out[f"{f:g}"] = dict(
            freq=f, n_on=n_on, n_off=int((~on).sum()), best=names[best_c],
            stat=obs, p=p, n_null=len(null), channels=per_ch,
            delivered=float(np.median([s[2] for s in segs
                                       if s[0] == "on" and s[1] == f])))
    if not out:
        return dict(names=names, freqs=freqs, results={}, passed=False,
                    best_freq=None, best_channel=None, p_min=float("nan"))
    best_k = min(out, key=lambda k: out[k]["p"])
    p_min = out[best_k]["p"]
    return dict(names=names, freqs=freqs, results=out,
                best_freq=out[best_k]["freq"], best_channel=out[best_k]["best"],
                p_min=p_min, passed=bool(p_min < 0.01 / len(out)),
                segment_s=float(n_min / fs))


'''
assert h_ in x; x = x.replace(h_, scorer + h_)
io.open("xr_session.py", "w", encoding="utf-8", newline="\n").write(x)

h = io.open("xr_stimulus.html", encoding="utf-8").read()
p1 = '''async function planFlicker(nominal) {
  await measureRefresh();'''
p2 = '''let refreshMeasured = false;
async function planFlicker(nominal) {
  if (!refreshMeasured) { await measureRefresh(); refreshMeasured = true; }'''
assert p1 in h; h = h.replace(p1, p2)
q1 = '''      case "start_alpha": runAlpha(m); break;'''
q2 = '''      case "start_sweep": runSweep(m); break;
      case "sweep_result": showSweepResult(m); break;
      case "start_alpha": runAlpha(m); break;'''
assert q1 in h; h = h.replace(q1, q2)
r1 = '''async function runScan(m) {'''
r2 = '''// ---- frequency sweep: frame-exact ON blocks at several frequencies ----
async function runSweep(m) {
  hideResult(); title.textContent = "FREQUENCY SWEEP"; title.className = "";
  const tok = running = {};
  sub.textContent = "measuring display refresh…";
  const frac = m.size || 0.38, full = frac >= 1;
  const size = Math.min(cv.width, cv.height) * Math.min(frac, 1);
  const [cOn, cOff] = m.colors || ["#ffff00", "#0000ff"];
  const plan = {};
  for (const f of m.freqs) { await planFlicker(f); plan[f] = { hz: actualHz, k: halfFrames }; }
  let b = 0;
  const nBlocks = m.repeats * m.freqs.length;
  for (let r = 0; r < m.repeats && running === tok; r++) {
    for (const f of m.freqs) {
      if (running !== tok) break;
      halfFrames = plan[f].k; actualHz = plan[f].hz;
      for (const on of [true, false]) {
        frameCount = 0;
        send({ type: "block", b, on, phase: "start", hz: plan[f].hz });
        sub.textContent = `${on ? "ON" : "off"} ${plan[f].hz.toFixed(2)} Hz  (${b + 1}/${nBlocks}, ${halfFrames} frames on/off @ ${refreshHz} fps)`;
        const t0 = now(), dur = on ? m.onS : m.offS;
        while (now() - t0 < dur && running === tok) {
          await frame(); frameCount++;
          const t = now();
          const fl = on && flickerOn();
          black();
          if (on) {
            ctx.fillStyle = fl ? cOn : cOff;
            if (full) ctx.fillRect(0, 0, cv.width, cv.height);
            else ctx.fillRect((cv.width - size) / 2, (cv.height - size) / 2, size, size);
          }
          dot(cv.width / 2, cv.height / 2);
          send({ type: "frame", stage: "sweep", pt: t, gx: -1, gy: -1,
                 lum: on ? 1 : 0, fl: fl ? 1 : 0 });
        }
        send({ type: "block", b, on, phase: "end", hz: plan[f].hz });
      }
      b++;
    }
  }
  black();
  send({ type: "sweep_done" });
}
function showSweepResult(m) {
  running = null; black();
  title.textContent = m.passed ? "SSVEP FOUND" : "NO SSVEP AT ANY FREQUENCY";
  title.className = m.passed ? "ok" : "warn";
  sub.textContent = m.verdict + "   —   " + m.table;
}

async function runScan(m) {'''
assert r1 in h; h = h.replace(r1, r2)
io.open("xr_stimulus.html", "w", encoding="utf-8", newline="\n").write(h)
print("sweep mode added")
