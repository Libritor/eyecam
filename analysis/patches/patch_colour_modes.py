"""Colour upgrade on the merged branch (Khalil smooth + Minu bwb + all-on-pc):

  * --mode smooth rotates the colour -> tag assignment every pass, so over
    three passes each colour has been on each tag once; the decoder divides
    every tag's response by that tag's mean response (the subject's
    frequency preference) before averaging, so the colour balance is the
    picture's, not the frequencies'.
  * --mode planes: the picture's R, G and B are scanned one after another as
    black/<colour> scans at the calibrated frequency and scored exactly like
    the "NO" (paper_score), then assembled into RGB. No colour is handicapped
    by a weak tag frequency.
  * colour targets 'eight' (the 8 pure colours) and 'hues' (12 hues + a
    shade row) so that decoding is tested on more than black/white/blue.

Run once from the repo root; it edits xr_stimulus.html, xr_session.py,
smoothcolor.py and targets.py in place and refuses to run twice."""
import io
import re

HERE = "."


def edit(path, pairs):
    x = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert x.count(old) == 1, (path, old[:60], x.count(old))
        x = x.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(x)
    print("patched", path)


# ------------------------------------------------------------------ page
page = io.open("xr_stimulus.html", encoding="utf-8").read()
assert "m.rotate" not in page, "already patched"

old_smooth = page[page.index("async function runSmoothScan(m) {"):page.index("// ---- multiplexed grey scan")]
new_smooth = r'''async function runSmoothScan(m) {
  hideResult(); title.textContent = ""; sub.textContent = "";
  const tok = running = {};
  if (!refreshMeasured) { await measureRefresh(); refreshMeasured = true; }
  const hz = m.freqs;
  send({ type: "color_plan", hzs: hz, refresh: refreshHz, smooth: 1 });
  const g = bwbGeom(m), gw = m.gridW, gh = m.gridH, target = m.target;   // target[y][x] = [r,g,b]
  const passes = Math.max(1, m.passes || 1);
  // rotation: pass p gives colour c the tag (c + p) mod 3, so over three
  // passes every colour has been on every tag once and the subject's
  // frequency preference cancels out of the colour balance
  const rotate = !!m.rotate && hz.length === 3;
  for (let ps = 0; ps < passes && running === tok; ps++) {
    const plan = [0, 1, 2].map(c => rotate ? (c + ps) % 3 : c);
    const tags = plan.map(k => hz[k]);
    send({ type: "color_pass", pass: ps, plan, hz: tags });
    await sleep(300);   // let the driver stamp the pass before the first frame
    for (let ry = 0; ry < gh && running === tok; ry++) {
      const gy = ps % 2 === 1 ? gh - 1 - ry : ry;
      const xs = [...Array(gw).keys()];
      if ((ry + ps) % 2 === 1) xs.reverse();
      if (passes > 1) sub.textContent = `pass ${ps + 1}/${passes}`;
      for (let xi = 0; xi < xs.length; xi++) {
        const gx = xs[xi];
        if (await holdWhilePaused()) { xi--; continue; }
        const rgb = target[gy][gx], lit = rgb.map(v => v > 0.02);
        const cx = g.x0 + (gx + 0.5) * g.cwid, cy = g.y0 + (gy + 0.5) * g.chgt;
        const t0 = now();
        flickerT0 = (await frameTs()) / 1000;
        while (now() - t0 < m.spc && running === tok && !halted()) {
          const ts = await frameTs(); frameCount++;
          const lvl = tags.map(f => smoothLevel(ts, f));
          black();
          if (lit.some(v => v)) {
            ctx.fillStyle = `rgb(${rgb.map((v, k) => lit[k] ? smoothByte(v, lvl[k]) : 0).join(",")})`;
            ctx.fillRect(cx - g.pw / 2, cy - g.ph / 2, g.pw, g.ph);
          }
          dot(cx, cy);
          const fl = lvl.map((v, k) => lit[k] && v > 0.5 ? 1 : 0);
          send({ type: "frame", stage: "color", pt: now(), gx, gy, lum: rgb[0], fl: fl[0],
                 flG: fl[1], flB: fl[2], r: rgb[0], g: rgb[1], b: rgb[2],
                 hzR: tags[0], hzG: tags[1], hzB: tags[2] });
        }
        if (halted()) { xi--; continue; }
        if (running !== tok) break;
      }
      send({ type: "row", y: gy });
    }
  }
  black();
  send({ type: "scan_done" });
}

'''
page = page.replace(old_smooth, new_smooth)

# the plain scan: an optional plane colour - the position flickers between
# black and that colour at the grey level the target gives it
page = page.replace(
    '''      const bwc = m.bw ? ["#ffffff", "#000000"] : null;
      const t0 = now();''',
    '''      const bwc = m.bw ? ["#ffffff", "#000000"] : null;
      const plane = m.plane || [1, 1, 1];   // --mode planes: this pass's colour
      const t0 = now();''')
page = page.replace(
    '''          ctx.fillStyle = bwc ? (fl ? `rgb(${shade},${shade},${shade})` : "#000000")
                              : (fl ? `rgb(${shade},${shade},0)` : `rgb(0,0,${shade})`);''',
    '''          ctx.fillStyle = bwc ? (fl ? `rgb(${plane.map(v => Math.round(v * shade)).join(",")})` : "#000000")
                              : (fl ? `rgb(${shade},${shade},0)` : `rgb(0,0,${shade})`);''')
assert page.count("m.plane") == 1 and page.count("plane.map") == 1
io.open("xr_stimulus.html", "w", encoding="utf-8", newline="\n").write(page)
print("patched xr_stimulus.html")

# ------------------------------------------------------------------ targets
edit("targets.py", [
    ("""    'mix'   = mixtures and shades: red, yellow, green, cyan, blue, magenta
              across, darker row by row, and a white-to-black row at the bottom
""", """    'mix'   = mixtures and shades: red, yellow, green, cyan, blue, magenta
              across, darker row by row, and a white-to-black row at the bottom
    'eight' = the 8 pure colours (black, red, green, yellow, blue, magenta,
              cyan, white) as blocks, every colour once per row, shuffled by
              row so no colour sits in one column
    'hues'  = 12 hues around the colour wheel across the top rows, then the
              8 pure colours, then a white-to-black shade row
"""),
    ("""    elif spec == "ring":""", """    elif spec == "eight":
        pure = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        for yg in range(grid_h):
            for xg in range(grid_w):
                g[yg, xg] = pure[(xg * 8 // grid_w + 3 * yg) % 8]
    elif spec == "hues":
        import colorsys
        pure = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        rows = max(grid_h - 2, 1)
        for yg in range(rows):
            for xg in range(grid_w):
                hue = ((xg + 0.5) / grid_w + yg / (2.0 * rows)) % 1.0
                g[yg, xg] = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        if grid_h > 1:
            for xg in range(grid_w):
                g[-2, xg] = pure[xg * 8 // grid_w]
            g[-1] = np.linspace(1, 0, grid_w)[:, None]
    elif spec == "ring":"""),
])

# ------------------------------------------------------------------ decoder
edit("smoothcolor.py", [
    ('''def visits(cur_t, gx, gy, spc):
    """(gx, gy, t0, t1) per cell visit. A stay much longer than spc is two
    passes meeting on one cell: it is cut into visits of spc."""
    cut = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    out = []
    for s, e in zip(np.r_[0, cut + 1], np.r_[cut + 1, len(gx)]):
        if gx[s] < 0 or gy[s] < 0:
            continue                               # pause marker
        t0, t1 = cur_t[s], cur_t[e - 1]
        n = max(1, int(round((t1 - t0) / spc))) if spc else 1
        for k in range(n):
            out.append((int(gx[s]), int(gy[s]), t0 + k * (t1 - t0) / n,
                        t0 + (k + 1) * (t1 - t0) / n))
    return out
''', '''def visits(cur_t, gx, gy, spc, with_index=False):
    """(gx, gy, t0, t1) per cell visit. A stay much longer than spc is two
    passes meeting on one cell: it is cut into visits of spc.
    with_index: also the log row the visit starts on."""
    cut = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    out = []
    for s, e in zip(np.r_[0, cut + 1], np.r_[cut + 1, len(gx)]):
        if gx[s] < 0 or gy[s] < 0:
            continue                               # pause marker
        t0, t1 = cur_t[s], cur_t[e - 1]
        n = max(1, int(round((t1 - t0) / spc))) if spc else 1
        for k in range(n):
            v = (int(gx[s]), int(gy[s]), t0 + k * (t1 - t0) / n,
                 t0 + (k + 1) * (t1 - t0) / n)
            out.append(v + (int(s),) if with_index else v)
    return out


def logged_tags(session_dir, log_name, tags):
    """The R, G, B tag of every log row (the page writes hz_r, hz_g, hz_b):
    with rotation (--smooth-rotate) the assignment changes every pass. Rows
    without tags (an older log) get the session's fixed tags."""
    from reconstruct import read_csv
    header, rows = read_csv(os.path.join(session_dir, log_name))
    out = np.tile(np.asarray(tags, float), (len(rows), 1))
    if "hz_r" in header:
        k = header.index("hz_r")
        for i, r in enumerate(rows):
            try:
                v = [float(r[k]), float(r[k + 1]), float(r[k + 2])]
            except (ValueError, IndexError):
                continue
            if all(x > 0 for x in v):
                out[i] = v
    return out
'''),
    ('''    w = np.ones(data.shape[1]) if weights is None else np.asarray(weights, float)
    sigma = [hf_sigma(data[:, c]) for c in range(data.shape[1])]
    gw, gh = gx.max() + 1, gy.max() + 1
    acc, cnt = np.zeros((gh, gw, 3)), np.zeros((gh, gw))
    for x, y, t0, t1 in visits(cur_t, gx, gy, spc):
        i0, i1 = np.searchsorted(eeg_t, [t0 + (SETTLE if t1 - t0 > 2 else 0), t1])
        if i1 - i0 < max(1.5, 0.6 * (t1 - t0)) * fs:
            continue                               # too short, or the stream dropped
        snr, den = np.zeros(3), 0.0
        for c in np.flatnonzero(w > 0):
            seg = clip_artifacts(data[i0:i1, c], sigma[c])
            if seg is None:
                continue
            v = [tag_snr(seg, fs, f0, [t for t in tags if t != f0]
                         + ([alpha] if alpha else [])) for f0 in tags]
            if np.all(np.isfinite(v)):
                snr += w[c] * np.array(v)
                den += w[c]
        if den > 0:
            acc[y, x] += snr / den
            cnt[y, x] += 1
    grid = np.sqrt(np.maximum(acc / np.maximum(cnt, 1)[:, :, None] - 1.0, 0.0))
''', '''    w = np.ones(data.shape[1]) if weights is None else np.asarray(weights, float)
    sigma = [hf_sigma(data[:, c]) for c in range(data.shape[1])]
    gw, gh = gx.max() + 1, gy.max() + 1
    row_tags = logged_tags(session_dir, log_name, tags)
    # one amplitude per (visit, colour): sqrt(SNR - 1) of the line at the
    # tag that colour used during that visit
    amp, where, used = [], [], []
    for x, y, t0, t1, s in visits(cur_t, gx, gy, spc, with_index=True):
        i0, i1 = np.searchsorted(eeg_t, [t0 + (SETTLE if t1 - t0 > 2 else 0), t1])
        if i1 - i0 < max(1.5, 0.6 * (t1 - t0)) * fs:
            continue                               # too short, or the stream dropped
        vt = list(row_tags[s])
        snr, den = np.zeros(3), 0.0
        for c in np.flatnonzero(w > 0):
            seg = clip_artifacts(data[i0:i1, c], sigma[c])
            if seg is None:
                continue
            v = [tag_snr(seg, fs, f0, [t for t in vt if t != f0]
                         + ([alpha] if alpha else [])) for f0 in vt]
            if np.all(np.isfinite(v)):
                snr += w[c] * np.array(v)
                den += w[c]
        if den > 0:
            amp.append(np.sqrt(np.maximum(snr / den - 1.0, 0.0)))
            where.append((y, x))
            used.append(vt)
    amp, used = np.asarray(amp).reshape(-1, 3), np.asarray(used).reshape(-1, 3)
    # tag gain: with rotation every colour has been on every tag equally
    # often, so a tag's mean amplitude over the scan is the subject's
    # response to that frequency, not the picture; divide it out
    rotated = len(amp) and all(
        abs(np.mean(used[:, c] == t) - 1.0 / 3) < 0.1 for c in range(3) for t in tags)
    if rotated:
        gain = {t: max(amp[used == t].mean(), 1e-6) for t in tags}
        mean_gain = np.mean(list(gain.values()))
        amp = amp / np.vectorize(lambda t: gain[t] / mean_gain)(used)
    acc, cnt = np.zeros((gh, gw, 3)), np.zeros((gh, gw))
    for a_, (y, x) in zip(amp, where):
        acc[y, x] += a_
        cnt[y, x] += 1
    grid = acc / np.maximum(cnt, 1)[:, :, None]
'''),
    ('''    res = dict(
        r_all=float(np.corrcoef(t.ravel(), rgb.ravel())[0, 1]),''',
     '''    res = dict(
        r_all=float(np.corrcoef(t.ravel(), rgb.ravel())[0, 1]),
        hue_right=hue_accuracy(rgb, t),'''),
    ('''def summary(res):
    text = "colour r = %.2f (R %.2f, G %.2f, B %.2f)" % (res["r_all"], *res["r_planes"])
    if "pure_cells" in res:
        text += f", {res['pure_right']} of {res['pure_cells']} pure-colour cells right"
''', '''def hue_accuracy(rgb, target, sat_min=0.25):
    """Share of the coloured cells (shown with some saturation) whose decoded
    hue is within 30 degrees of the hue shown: the colour-naming test."""
    import colorsys
    t, d = np.asarray(target, float), np.asarray(rgb, float)
    n = right = 0
    for y in range(t.shape[0]):
        for x in range(t.shape[1]):
            ht, st, vt = colorsys.rgb_to_hsv(*t[y, x])
            if st < sat_min or vt < 0.25:
                continue
            hd, sd, vd = colorsys.rgb_to_hsv(*np.clip(d[y, x], 0, 1))
            n += 1
            diff = abs(ht - hd) % 1.0
            right += min(diff, 1.0 - diff) <= 30 / 360.0 and sd >= sat_min
    return (right, n)


def summary(res):
    text = "colour r = %.2f (R %.2f, G %.2f, B %.2f)" % (res["r_all"], *res["r_planes"])
    if "pure_cells" in res:
        text += f", {res['pure_right']} of {res['pure_cells']} pure-colour cells right"
    hr = res.get("hue_right")
    if hr and hr[1]:
        text += f", hue right on {hr[0]} of {hr[1]} coloured cells"
'''),
])
