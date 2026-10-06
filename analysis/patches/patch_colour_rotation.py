"""2026-10-02: colour stage redesign.

1. No colour gains: the per-colour gains estimated from the 90 s colour
   calibration were noisy (red 0.05 one day, 3.77 another) and lowered the
   whole-image correlation on all four colour runs. Channel weights stay.
2. Tag rotation: the colour scan is repeated --color-passes times and the
   colour-to-frequency assignment rotates each pass, so every colour spends
   one pass on each tag frequency. The decoder reads the per-pass assignment
   from color_passes.json and averages the planes, cancelling the subject's
   frequency preference instead of estimating it.
Run once from the repo root."""
import io


def patch(p, pairs):
    x = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert x.count(a) == 1, (p, a[:70], x.count(a))
        x = x.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="\n").write(x)


# ---------------- page: rotate the assignment every pass ----------------
patch("xr_stimulus.html", [
    ('''  send({ type: "color_plan", hzs: plans.map(p => p.hz), refresh: refreshHz });
  const gw = m.gridW, gh = m.gridH, target = m.target;   // target[y][x] = [r,g,b]
  const bw = Math.min(cv.width, cv.height * gw / gh) * 0.84;
  const bh = bw * gh / gw;
  const x0 = (cv.width - bw) / 2, y0 = (cv.height - bh) / 2;
  const cwid = bw / gw, chgt = bh / gh;
  for (let gy = 0; gy < gh && running === tok; gy++) {
    const xs = [...Array(gw).keys()];
    if (gy % 2 === 1) xs.reverse();
    for (let xi = 0; xi < xs.length; xi++) {
      const gx = xs[xi];
      if (await holdWhilePaused()) { xi--; continue; }
      const [r, g, b] = target[gy][gx];
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok && !paused) {
        const ts = await frameTs(); frameCount++;
        const t = now();
        const sR = r > 0.05 && phaseOn(ts, plans[0].k);
        const sG = g > 0.05 && phaseOn(ts, plans[1].k);
        const sB = b > 0.05 && phaseOn(ts, plans[2].k);''',
     '''  send({ type: "color_plan", hzs: plans.map(p => p.hz), refresh: refreshHz });
  const gw = m.gridW, gh = m.gridH, target = m.target;   // target[y][x] = [r,g,b]
  const bw = Math.min(cv.width, cv.height * gw / gh) * 0.84;
  const bh = bw * gh / gw;
  const x0 = (cv.width - bw) / 2, y0 = (cv.height - bh) / 2;
  const cwid = bw / gw, chgt = bh / gh;
  // Several passes; pass p gives colour c the tag plan (c + p) mod 3, so
  // over three passes every colour has been on every frequency once and the
  // subject's frequency preference cancels out of the colour image.
  const passes = Math.max(1, m.passes || 1);
  for (let ps = 0; ps < passes && running === tok; ps++) {
  const plan = [0, 1, 2].map(c => plans[(c + ps) % plans.length]);
  send({ type: "color_pass", pass: ps, plan: [0, 1, 2].map(c => (c + ps) % plans.length),
         hz: plan.map(p => p.hz) });
  await sleep(300);   // let the driver stamp the pass before the first frame
  for (let ry = 0; ry < gh && running === tok; ry++) {
    const gy = ps % 2 === 1 ? gh - 1 - ry : ry;
    const xs = [...Array(gw).keys()];
    if ((ry + ps) % 2 === 1) xs.reverse();
    if (passes > 1) sub.textContent = `colour pass ${ps + 1}/${passes}`;
    for (let xi = 0; xi < xs.length; xi++) {
      const gx = xs[xi];
      if (await holdWhilePaused()) { xi--; continue; }
      const [r, g, b] = target[gy][gx];
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok && !paused) {
        const ts = await frameTs(); frameCount++;
        const t = now();
        const sR = r > 0.05 && phaseOn(ts, plan[0].k);
        const sG = g > 0.05 && phaseOn(ts, plan[1].k);
        const sB = b > 0.05 && phaseOn(ts, plan[2].k);'''),
    ('''      if (paused) { xi--; continue; }
      if (running !== tok) break;
    }
    send({ type: "row", y: gy });
  }
  black();
  send({ type: "scan_done" });
}

// ---- multiplexed grey scan''',
     '''      if (paused) { xi--; continue; }
      if (running !== tok) break;
    }
    send({ type: "row", y: gy });
  }
  }
  black();
  send({ type: "scan_done" });
}

// ---- multiplexed grey scan'''),
])

# ---------------- driver ----------------
patch("xr_session.py", [
    ('''        elif typ == "color_plan":
            self.color_hzs = [float(v) for v in m.get("hzs", [])]
            print("colour flicker plan:", self.color_hzs, "@", m.get("refresh"), "fps")''',
     '''        elif typ == "color_plan":
            self.color_hzs = [float(v) for v in m.get("hzs", [])]
            print("colour flicker plan:", self.color_hzs, "@", m.get("refresh"), "fps")
        elif typ == "color_pass":
            # which tag frequency each of R, G, B uses from now on
            entry = dict(t=t, pass_index=int(m.get("pass", 0)),
                         plan=[int(v) for v in m.get("plan", [0, 1, 2])],
                         hz=[float(v) for v in m.get("hz", [])])
            self.color_passes.append(entry)
            with open(os.path.join(self.session, "color_passes.json"), "w") as f:
                json.dump(self.color_passes, f, indent=1)
            print(f"colour pass {entry['pass_index'] + 1}: R/G/B at "
                  + "/".join(f"{v:.2f}" for v in entry["hz"]) + " Hz")'''),
    ('''        self.color_hzs = []         # page's frame-exact plan for the colour scan''',
     '''        self.color_hzs = []         # page's frame-exact plan for the colour scan
        self.color_passes = []      # per pass: which frequency each colour used'''),
    ('''        self.color_hzs = []
        self.open_log("color_log.csv")
        await self.send(cmd="start_scan", color=True, gridW=gw, gridH=gh,
                        spc=a.color_spc, freqs=freqs, target=ctarget.tolist())
        scan_s = gw * gh * a.color_spc''',
     '''        self.color_hzs = []
        self.color_passes = []
        self.open_log("color_log.csv")
        await self.send(cmd="start_scan", color=True, gridW=gw, gridH=gh,
                        spc=a.color_spc, freqs=freqs, target=ctarget.tolist(),
                        passes=a.color_passes)
        scan_s = gw * gh * a.color_spc * max(1, a.color_passes)'''),
    ('''        hz_del = self.measured_cols("color_log.csv")
        for k in range(3):
            if not np.isfinite(hz_del[k]):
                hz_del[k] = (self.color_hzs[k] if k < len(self.color_hzs)
                             else freqs[k])
        print("colour scan delivered:", ["%.2f" % v for v in hz_del])
        try:
            grid, rgb, cres = reconstruct.reconstruct_color(
                self.session, hz_del, weights=weights, gains=gains,
                cursor_log="color_log.csv", target=ctarget)''',
     '''        # the three tag frequencies as the page delivered them; which colour
        # used which one in each pass is in color_passes.json
        hz_del = [self.color_hzs[k] if k < len(self.color_hzs) else freqs[k]
                  for k in range(3)]
        print("colour tags:", ["%.2f" % v for v in hz_del],
              f"over {max(1, a.color_passes)} pass(es)")
        try:
            # gains=None: the colour-calibration gains are reported but not
            # applied (they lowered every run's correlation)
            grid, rgb, cres = reconstruct.reconstruct_color(
                self.session, hz_del, weights=weights, gains=None,
                cursor_log="color_log.csv", target=ctarget)'''),
    ('''    ap.add_argument("--color-reps", type=int, default=3)''',
     '''    ap.add_argument("--color-reps", type=int, default=3)
    ap.add_argument("--color-passes", type=int, default=3,
                    help="colour scan passes; the colour-to-frequency "
                         "assignment rotates every pass")'''),
])

# ---------------- decoder ----------------
patch("reconstruct.py", [
    ('''    out = out or os.path.join(session_dir, "reconstruction_color.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)''',
     '''    out = out or os.path.join(session_dir, "reconstruction_color.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)
    # Per-pass colour-to-frequency assignment (tag rotation). Without the
    # file, colour k used freqs[k] throughout.
    passes = []
    pfile = os.path.join(session_dir, "color_passes.json")
    if os.path.exists(pfile):
        with open(pfile) as f:
            passes = sorted(json.load(f), key=lambda e: e["t"])

    def hz_for(t_visit):
        hz = list(freqs)
        for e in passes:
            if e["t"] <= t_visit + 0.05 and len(e.get("hz", [])) == 3:
                hz = [float(v) for v in e["hz"]]
        return hz'''),
    ('''        for k, f0 in enumerate(freqs):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")''',
     '''        for k, f0 in enumerate(hz_for(t0)):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")'''),
])
x = io.open("reconstruct.py", encoding="utf-8").read()
assert "import json" in x, "reconstruct.py needs json"

# ---------------- launcher defaults ----------------
patch("eyecam.py", [
    ('''             "--color-reps", "3", "--color-on", "6", "--color-spc", "6",''',
     '''             "--color-reps", "3", "--color-on", "6", "--color-spc", "6",
             "--color-passes", "3",'''),
])
print("patched")
