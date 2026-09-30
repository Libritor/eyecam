"""Patch: frequency-multiplexed grey scan (3 cells per dwell), music-only mode,
fullscreen inside the click handler."""
import io, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- page ----------------
h = io.open("xr_stimulus.html", encoding="utf-8").read()
a = '''  await new Promise(res => {
    const done = () => { clearInterval(gp); res(); };
    addEventListener("pointerdown", done, { once: true });
    addEventListener("keydown", done, { once: true });'''
a2 = '''  const goFull = () => {  // must run INSIDE the user gesture to be honoured
    try {
      const el = document.documentElement;
      const req = el.requestFullscreen || el.webkitRequestFullscreen;
      if (req && !document.fullscreenElement) req.call(el).catch(() => {});
    } catch {}
  };
  await new Promise(res => {
    const done = () => { clearInterval(gp); res(); };
    addEventListener("pointerdown", () => { goFull(); done(); }, { once: true });
    addEventListener("keydown", () => { goFull(); done(); }, { once: true });'''
assert a in h; h = h.replace(a, a2)
b = '''  // go full screen on the arming gesture: the browser's URL bar / tab strip
  // is bright and sits right above the stimulus (it leaks into the SSVEP)
  try {
    const el = document.documentElement;
    const req = el.requestFullscreen || el.webkitRequestFullscreen;
    if (req && !document.fullscreenElement) await Promise.race([req.call(el).catch(() => {}), sleep(800)]);
  } catch {}'''
assert b in h; h = h.replace(b, "")
c = '''async function runScan(m) {
  if (m.color) return runColorScan(m);'''
c2 = '''async function runScan(m) {
  if (m.color) return runColorScan(m);
  if (m.mux) return runMuxScan(m);'''
assert c in h; h = h.replace(c, c2)
d = '''// ---- frequency sweep: frame-exact ON blocks at several frequencies ----'''
d2 = '''// ---- multiplexed grey scan: N neighbouring cells flicker at once, each
// with its own tag frequency; the line at tag k in the EEG is cell k ----
async function runMuxScan(m) {
  hideResult(); title.textContent = ""; sub.textContent = "";
  const tok = running = {};
  const plans = [];
  for (const f of m.freqs) { await planFlicker(f); plans.push({ hz: actualHz, k: halfFrames }); }
  send({ type: "color_plan", hzs: plans.map(p => p.hz), refresh: refreshHz });
  const gw = m.gridW, gh = m.gridH, target = m.target, n = plans.length;
  const bw = Math.min(cv.width, cv.height * gw / gh) * 0.84;
  const bh = bw * gh / gw;
  const x0b = (cv.width - bw) / 2, y0b = (cv.height - bh) / 2;
  const cwid = bw / gw, chgt = bh / gh;
  const nStrips = Math.floor(gw / n);
  for (let gy = 0; gy < gh && running === tok; gy++) {
    const strips = [...Array(nStrips).keys()];
    if (gy % 2 === 1) strips.reverse();
    for (const s of strips) {
      const x0 = s * n;
      const lums = plans.map((_, k) => target[gy][x0 + k]);
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok) {
        const ts = await frameTs(); frameCount++;
        const t = now();
        const st = plans.map((p, k) => lums[k] > 0.05 && phaseOn(ts, p.k));
        black();
        for (let k = 0; k < n; k++) {
          if (lums[k] > 0.05 && st[k]) {
            const shade = Math.round(255 * (0.3 + 0.7 * lums[k]));
            ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
            ctx.fillRect(x0b + (x0 + k) * cwid, y0b + gy * chgt, cwid, chgt);
          }
        }
        dot(x0b + (x0 + n / 2) * cwid, y0b + (gy + 0.5) * chgt);
        send({ type: "frame", stage: "mux", pt: t, gx: x0, gy, lum: lums[0], fl: st[0] ? 1 : 0,
               flG: st[1] ? 1 : 0, flB: st[2] ? 1 : 0, r: lums[0], g: lums[1] || 0, b: lums[2] || 0 });
      }
      if (running !== tok) break;
    }
    send({ type: "row", y: gy });
  }
  black();
  send({ type: "scan_done" });
}

// ---- frequency sweep: frame-exact ON blocks at several frequencies ----'''
assert d in h; h = h.replace(d, d2)
io.open("xr_stimulus.html", "w", encoding="utf-8", newline="\n").write(h)

# ---------------- reconstruct ----------------
r = io.open("reconstruct.py", encoding="utf-8").read()
e = '''def reconstruct_color(session_dir, freqs, weights=None, gains=None,'''
e2 = '''def reconstruct_mux(session_dir, freqs, weights=None, cursor_log="mux_log.csv",
                    out="", target=None, channels=""):
    """Multiplexed grey scan: each dwell shows n = len(freqs) neighbouring
    cells (gx..gx+n-1 of row gy), cell k tagged at freqs[k]. The line SNR at
    tag k (channel-weighted) is cell k's value; each tag's values are divided
    by their median over the scan so no tag's gain dominates."""
    out = out or os.path.join(session_dir, "reconstruction_mux.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    ch_sigma = [hf_sigma(data[:, c]) for c in range(n_ch)]
    n = len(freqs)
    gw, gh = gx.max() + n, gy.max() + 1
    vals = {k: [] for k in range(n)}
    cells = []
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    nperseg = int(round(fs))
    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
        need = nperseg - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nperseg)
        row = []
        for k, f0 in enumerate(freqs):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")
                if not np.isnan(sc):
                    num += w[c] * sc
                    den += w[c]
            row.append(num / den if den > 0 else np.nan)
        cells.append((gy[s], gx[s], row))
    grid = np.full((gh, gw), np.nan)
    med = [np.nanmedian([c[2][k] for c in cells]) or 1.0 for k in range(n)]
    for y, x, row in cells:
        for k in range(n):
            if x + k < gw and np.isfinite(row[k]):
                grid[y, x + k] = row[k] / max(med[k], 1e-9)
    if np.isnan(grid).all():
        raise ValueError("no cell produced a score")
    grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    kern = np.array(config.VERTICAL_KERNEL)
    grid = convolve1d(grid, kern / kern.sum(), axis=0, mode="nearest")
    np.save(os.path.join(session_dir, "reconstruction_mux_grid.npy"), grid)
    save_image(grid, out)
    rr = None
    if target is not None and np.asarray(target).shape == grid.shape:
        rr = float(np.corrcoef(np.asarray(target).ravel(), grid.ravel())[0, 1])
    return grid, rr


def reconstruct_color(session_dir, freqs, weights=None, gains=None,'''
assert e in r; r = r.replace(e, e2)
io.open("reconstruct.py", "w", encoding="utf-8", newline="\n").write(r)

# ---------------- driver ----------------
x = io.open("xr_session.py", encoding="utf-8").read()
f_ = '''            if a.mode == "extras":'''
f2 = '''            if a.mode == "mux":
                return await self.run_mux(result, recorder)
            if a.mode == "music":
                self.blocks, self.block_hz = [], []
                self.open_log("assr_log.csv")
                await self.send(cmd="start_assr", music=True, blocks=a.music_blocks,
                                onS=a.music_on, offS=a.music_off, modFreq=40.0)
                await self.wait_for("assr_done",
                                    timeout=a.music_blocks * (a.music_on + a.music_off) + 120)
                self.close_log()
                await asyncio.sleep(0.7)
                mus = score_sweep(self.session, self.blocks,
                                  [40.0] * len(self.blocks), log_name=None)
                mres = next(iter(mus["results"].values()), None)
                with open(os.path.join(self.session, "music.json"), "w") as f:
                    json.dump(mus, f, indent=1)
                mtext = (f"40 Hz tag: p={mres['p']:.4f} best {mres['best']} "
                         f"(ON {mres['channels'][mres['best']]['on_med']:.2f} / "
                         f"OFF {mres['channels'][mres['best']]['off_med']:.2f})"
                         if mres else "no scorable blocks")
                print("music " + mtext)
                await self.send(cmd="assr_result", passed=bool(mres and mres["p"] < config.CALIB_P_MAX), table=mtext)
                result.update(ok=True, music=mtext)
                recorder.stop()
                print(f"MUSIC DONE -> {self.session}")
                await asyncio.sleep(a.linger)
                return 0
            if a.mode == "extras":'''
assert f_ in x; x = x.replace(f_, f2)
g = '''    async def run_full_extras(self, result):'''
g2 = '''    async def run_mux(self, result, recorder):
        """White 3-tag calibration, then the multiplexed grey scan."""
        a = self.args
        freqs = [float(v) for v in str(a.color_freqs).split(",") if v.strip()]
        n = len(freqs)
        gw, gh = a.grid_w - (a.grid_w % n), a.grid_h
        target = targets.load_target(a.target, gw, gh)
        np.save(os.path.join(self.session, "target.npy"), target)
        names = csv_header(os.path.join(self.session, "eeg.csv")) or CH_NAMES
        # calibration: each tag frequency, full-panel white/black
        self.spectate(type="stage", stage="mux_calib", detail="")
        self.blocks, self.block_hz = [], []
        self.open_log("calib_log.csv")
        cols = {f"{f:g}": ["#ffffff", "#000000"] for f in freqs}
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.calib_blocks,
                        onS=a.calib_on, offS=a.calib_on, colorsPerFreq=cols, size=1.0)
        total = a.calib_blocks * n * 2 * a.calib_on
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        cal = score_sweep(self.session, self.blocks, self.block_hz, log_name="calib_log.csv")
        ordered = sorted(cal["results"].values(), key=lambda rr: rr["freq"])
        # channel weights: mean over tags of the per-channel ON-OFF log diff
        diff = np.zeros(len(names))
        for rr in ordered:
            diff += np.array([max(rr["channels"][nm]["diff"], 0.0) for nm in names])
        weights = diff / diff.sum() if diff.sum() > 0 else np.ones(len(names)) / len(names)
        cal["weights"] = {nm: float(v) for nm, v in zip(names, weights)}
        cal["names"] = names
        cal["passed"] = bool(min(rr["p"] for rr in ordered) < config.CALIB_P_MAX)
        with open(os.path.join(self.session, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=1)
        ctab = "  ".join(f"{rr['freq']:g}Hz p={rr['p']:.3f} {rr['best']}" for rr in ordered)
        print(f"mux calibration passed={cal['passed']}: {ctab}  weights="
              + " ".join(f"{nm}:{v:.2f}" for nm, v in zip(names, weights)))
        result["calib"] = dict(table=ctab, passed=cal["passed"], weights=cal["weights"])
        await self.send(cmd="calib_result", passed=cal["passed"], table=ctab)
        await asyncio.sleep(2.5)
        # scan
        self.spectate(type="stage", stage="mux_scan", detail="")
        self.color_hzs = []
        self.open_log("mux_log.csv")
        await self.send(cmd="start_scan", mux=True, gridW=gw, gridH=gh, spc=a.spc,
                        freqs=freqs, target=target.tolist())
        scan_s = (gw // n) * gh * a.spc
        await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        g0 = self.delivery_check("mux_log.csv", scan_s)
        result["g0"] = g0
        hz_del = self.measured_cols("mux_log.csv")
        for k in range(n):
            if not np.isfinite(hz_del[k]):
                hz_del[k] = self.color_hzs[k] if k < len(self.color_hzs) else freqs[k]
        print("mux scan delivered:", ["%.2f" % v for v in hz_del], "| G0", g0)
        grid, r = reconstruct.reconstruct_mux(self.session, hz_del, weights=weights,
                                              cursor_log="mux_log.csv", target=target)
        # shifted-EEG null
        eeg_t = np.genfromtxt(os.path.join(self.session, "eeg.csv"), delimiter=",",
                              skip_header=1, usecols=[0])
        fs = reconstruct.infer_fs(eeg_t)
        orig = reconstruct.load_session
        rs_null = []
        for k in range(8):
            shift = 31.0 + k * 29.0
            if shift > eeg_t[-1] - eeg_t[0] - 10:
                break
            def fake(session_dir, channels="", cursor_log="cursor_log.csv", _s=shift):
                e, d, nm, c, xx, yy = orig(session_dir, channels, cursor_log)
                return e, np.roll(d, int(round(_s * fs)), axis=0), nm, c, xx, yy
            reconstruct.load_session = fake
            try:
                _, rn = reconstruct.reconstruct_mux(
                    self.session, hz_del, weights=weights, cursor_log="mux_log.csv",
                    target=target, out=os.path.join(self.session, "_null.png"))
                if rn is not None:
                    rs_null.append(rn)
            finally:
                reconstruct.load_session = orig
        # restore the real reconstruction files (the null loop overwrote the grid)
        grid, r = reconstruct.reconstruct_mux(self.session, hz_del, weights=weights,
                                              cursor_log="mux_log.csv", target=target)
        print(f"mux image r={r:.3f}; shifted-EEG null r: "
              + ", ".join(f"{v:.2f}" for v in rs_null))
        result.update(ok=True, r=r, r_null=rs_null, r_null_max=max(rs_null) if rs_null else None,
                      freqs_delivered=hz_del, gridW=gw, gridH=gh)
        png = grid_to_data_url(reconstruct_norm(grid))
        await self.send(cmd="result", png=png, r=r or 0.0, flickerHz=hz_del[0])
        self.spectate(type="grid", w=gw, h=gh, scores=[float(v) for v in np.asarray(grid).ravel()])
        recorder.stop()
        print(f"MUX SESSION DONE r={r} -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_full_extras(self, result):'''
assert g in x; x = x.replace(g, g2)
i_ = '''                                       "sweep", "full", "extras"],'''
i2 = '''                                       "sweep", "full", "extras", "mux",
                                       "music"],'''
assert i_ in x; x = x.replace(i_, i2)
io.open("xr_session.py", "w", encoding="utf-8", newline="\n").write(x)
print("patched mux + music modes + fullscreen-in-click")
