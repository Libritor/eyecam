"""Patch reconstruct.py + targets.py: line-detector cell score, colour
(frequency-tagged RGB) reconstruction, shifted-EEG null, colour targets."""
import io, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

x = io.open("reconstruct.py", encoding="utf-8").read()

# ---- imports ----
if "from scipy.signal import periodogram" not in x:
    assert "from scipy.signal import welch" in x
    x = x.replace("from scipy.signal import welch",
                  "from scipy.signal import welch, periodogram", 1)

# ---- load_session: selectable cursor log ----
a = '''def load_session(session_dir, channels=""):
    header, rows = read_csv(os.path.join(session_dir, "eeg.csv"))'''
a2 = '''def load_session(session_dir, channels="", cursor_log="cursor_log.csv"):
    header, rows = read_csv(os.path.join(session_dir, "eeg.csv"))'''
assert a in x; x = x.replace(a, a2)
b = '''    _, crows = read_csv(os.path.join(session_dir, "cursor_log.csv"))'''
b2 = '''    _, crows = read_csv(os.path.join(session_dir, cursor_log))'''
assert b in x; x = x.replace(b, b2)

# ---- line detector ----
c = '''def robust_sigma(x):'''
c2 = '''def line_snr(seg, fs, f0, half=0.15, flank=(0.5, 2.0)):
    """Exact-frequency line detector: mean periodogram power within +-half Hz
    of f0 (at least one bin) over the mean power 0.5-2 Hz either side.
    One Hann periodogram over the whole segment, so a 4 s cell gives 0.25 Hz
    bins: ~8x less noise per bin than the 1 Hz Welch bins of ssvep_score."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < fs * 1.5:
        return np.nan
    seg = seg - seg.mean()
    f, p = periodogram(seg, fs=fs, window="hann", detrend="constant")
    df = f[1] - f[0]
    h = max(half, df)
    pk = p[np.abs(f - f0) <= h].mean()
    fk = p[(np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])].mean()
    return pk / max(fk, 1e-12)


def robust_sigma(x):'''
assert c in x; x = x.replace(c, c2, 1)

# ---- cell_score: method ----
d = '''def cell_score(seg, fs, sigma_floor, stim_freq=None):'''
d2 = '''def cell_score(seg, fs, sigma_floor, stim_freq=None, method="welch"):'''
assert d in x; x = x.replace(d, d2)
e = '''    clipped = baseline + np.clip(dev, -config.ARTIFACT_Z * sigma,
                                 config.ARTIFACT_Z * sigma)
    return ssvep_score(clipped, fs, stim_freq)'''
e2 = '''    clipped = baseline + np.clip(dev, -config.ARTIFACT_Z * sigma,
                                 config.ARTIFACT_Z * sigma)
    if method == "line":
        f0 = config.STIM_FREQ_HZ if stim_freq is None else stim_freq
        return line_snr(clipped, fs, f0)
    return ssvep_score(clipped, fs, stim_freq)'''
assert e in x; x = x.replace(e, e2)

# ---- reconstruct: method ----
f_ = '''def reconstruct(eeg_t, data, cur_t, gx, gy, fs=None, weights=None,
                stim_freq=None):'''
f2 = '''def reconstruct(eeg_t, data, cur_t, gx, gy, fs=None, weights=None,
                stim_freq=None, method="welch"):'''
assert f_ in x; x = x.replace(f_, f2)
g = '''            sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], stim_freq)'''
g2 = '''            sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], stim_freq,
                            method)'''
assert g in x; x = x.replace(g, g2)

# ---- run: method, shift null, cursor log ----
h = '''def run(session_dir, channels="", calibration="", out="", stim_freq=None):
    out = out or os.path.join(session_dir, "reconstruction.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels)
    weights = None
    if calibration:
        weights = weights_from_calibration(calibration, names)
        print("channel weights:",
              {n: round(float(w), 3) for n, w in zip(names, weights)})
    grid = reconstruct(eeg_t, data, cur_t, gx, gy, weights=weights,
                       stim_freq=stim_freq)
    np.save(os.path.join(session_dir, "reconstruction_grid.npy"), grid)
    save_image(grid, out)
    print(f"reconstruction -> {out}")
'''
h2 = '''def run(session_dir, channels="", calibration="", out="", stim_freq=None,
        method="welch", shift_s=0.0, cursor_log="cursor_log.csv",
        weights=None, save=True):
    """shift_s != 0: circularly shift the EEG by that many seconds relative
    to the cursor log = a negative control (same data, wrong alignment)."""
    out = out or os.path.join(session_dir, "reconstruction.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    if weights is None and calibration:
        weights = weights_from_calibration(calibration, names)
        if not shift_s:
            print("channel weights:",
                  {n: round(float(w), 3) for n, w in zip(names, weights)})
    fs = infer_fs(eeg_t)
    if shift_s:
        data = np.roll(data, int(round(shift_s * fs)), axis=0)
    grid = reconstruct(eeg_t, data, cur_t, gx, gy, fs=fs, weights=weights,
                       stim_freq=stim_freq, method=method)
    if save:
        np.save(os.path.join(session_dir, "reconstruction_grid.npy"), grid)
        save_image(grid, out)
        print(f"reconstruction -> {out}")
'''
assert h in x; x = x.replace(h, h2)
i_ = '''            r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
            print(f"correlation with ground-truth target: r = {r:.3f}")
    return grid, r'''
i2 = '''            r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
            if save:
                print(f"correlation with ground-truth target: r = {r:.3f}")
    return grid, r


def null_r(session_dir, n_shifts=12, **kw):
    """r of the same reconstruction with the EEG circularly shifted by
    n_shifts different offsets (31 s .. ): the chance level for this session."""
    eeg_t = np.array([float(r[0]) for r in read_csv(
        os.path.join(session_dir, "eeg.csv"))[1]])
    span = eeg_t[-1] - eeg_t[0]
    rs = []
    for k in range(n_shifts):
        s = 31.0 + k * max(7.0, (span - 62.0) / max(n_shifts, 1))
        if s > span - 10:
            break
        _, r = run(session_dir, shift_s=s, save=False, **kw)
        if r is not None:
            rs.append(r)
    return rs


def reconstruct_color(session_dir, freqs, weights=None, gains=None,
                      cursor_log="color_log.csv", out="", target=None,
                      channels=""):
    """Frequency-tagged RGB: each cell flickers its R, G, B components at
    freqs[0..2]; the line SNR at each frequency (channel-weighted, divided by
    the per-frequency gain from the colour calibration) is that colour plane."""
    out = out or os.path.join(session_dir, "reconstruction_color.png")
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels,
                                                     cursor_log)
    fs = infer_fs(eeg_t)
    n_ch = data.shape[1]
    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)
    ch_sigma = [hf_sigma(data[:, c]) for c in range(n_ch)]
    gw, gh = gx.max() + 1, gy.max() + 1
    acc = np.zeros((gh, gw, 3))
    cnt = np.zeros((gh, gw))
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
        for k, f0 in enumerate(freqs):
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                sc = cell_score(data[i0:i1, c], fs, ch_sigma[c], f0, "line")
                if not np.isnan(sc):
                    num += w[c] * sc
                    den += w[c]
            if den > 0:
                acc[gy[s], gx[s], k] += num / den / max(gains[k], 1e-6)
        cnt[gy[s], gx[s]] += 1
    with np.errstate(invalid="ignore"):
        grid = acc / np.maximum(cnt, 1)[:, :, None]
    grid[cnt == 0] = np.nan
    for k in range(3):
        plane = grid[:, :, k]
        if np.isnan(plane).any():
            plane[np.isnan(plane)] = np.nanmedian(plane)
        kern = np.array(config.VERTICAL_KERNEL)
        grid[:, :, k] = convolve1d(plane, kern / kern.sum(), axis=0,
                                   mode="nearest")
    np.save(os.path.join(session_dir, "reconstruction_color_grid.npy"), grid)
    rgb = np.zeros_like(grid)
    for k in range(3):
        lo, hi = np.percentile(grid[:, :, k], 2), np.percentile(grid[:, :, k], 98)
        rgb[:, :, k] = np.clip((grid[:, :, k] - lo) / max(hi - lo, 1e-12), 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    img.resize((gw * config.UPSCALE, gh * config.UPSCALE),
               Image.BICUBIC).save(out)
    res = dict(out=out, r_planes=None, r_all=None)
    if target is not None and np.asarray(target).shape == grid.shape:
        t = np.asarray(target, float)
        res["r_planes"] = [float(np.corrcoef(t[:, :, k].ravel(),
                                             grid[:, :, k].ravel())[0, 1])
                           if t[:, :, k].std() > 0 else float("nan")
                           for k in range(3)]
        res["r_all"] = float(np.corrcoef(t.ravel(), grid.ravel())[0, 1])
        # colour identity per cell: does the dominant plane match the target's?
        dom_t = np.argmax(t, axis=2)
        dom_g = np.argmax(rgb, axis=2)
        m = t.max(axis=2) > 0.5
        res["hue_accuracy"] = float((dom_t[m] == dom_g[m]).mean()) if m.any() else float("nan")
    return grid, rgb, res'''
assert i_ in x; x = x.replace(i_, i2)
io.open("reconstruct.py", "w", encoding="utf-8", newline="\n").write(x)

# ---- targets: colour targets ----
t = io.open("targets.py", encoding="utf-8").read()
j = '''def load_target(spec, grid_w=None, grid_h=None):'''
j2 = '''def color_target(spec, grid_w=None, grid_h=None):
    """RGB target (grid_h, grid_w, 3) in [0,1].
    'flag'  = three vertical bands R | G | B
    'quad'  = quadrants R, G, B, white
    'text:X'= glyph X in yellow (R+G) on black  (tests two-plane cells)
    'ring'  = red disc, green ring, blue corners
    or an image path (RGB)."""
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    g = np.zeros((grid_h, grid_w, 3))
    if spec == "flag":
        w3 = grid_w / 3.0
        for xg in range(grid_w):
            g[:, xg, min(2, int(xg / w3))] = 1.0
    elif spec == "quad":
        h2, w2 = grid_h // 2, grid_w // 2
        g[:h2, :w2, 0] = 1.0
        g[:h2, w2:, 1] = 1.0
        g[h2:, :w2, 2] = 1.0
        g[h2:, w2:, :] = 1.0
    elif spec.startswith("text:"):
        gl = text_target(spec[5:], grid_w, grid_h)
        g[:, :, 0] = gl
        g[:, :, 1] = gl
    elif spec == "ring":
        yy, xx = np.mgrid[0:grid_h, 0:grid_w]
        cy, cx = (grid_h - 1) / 2, (grid_w - 1) / 2
        rr = np.sqrt(((yy - cy) / grid_h) ** 2 + ((xx - cx) / grid_w) ** 2)
        g[rr < 0.18, 0] = 1.0
        g[(rr >= 0.18) & (rr < 0.36), 1] = 1.0
        g[rr >= 0.36, 2] = 1.0
    else:
        img = Image.open(spec).convert("RGB").resize((grid_w, grid_h),
                                                     Image.LANCZOS)
        g = np.asarray(img, dtype=float) / 255.0
    return g


def load_target(spec, grid_w=None, grid_h=None):'''
assert j in t; t = t.replace(j, j2)
io.open("targets.py", "w", encoding="utf-8", newline="\n").write(t)
print("patched reconstruct.py + targets.py")
