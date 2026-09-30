"""Patch xr_session.py: --mode full = 12 Hz line-gated calibration -> grey
image scan (line-detector reconstruction + shifted-EEG null) -> colour
calibration (single-colour flicker per frequency) -> frequency-tagged colour
scan + colour reconstruction -> music piece tagged at 40 Hz (ASSR)."""
import io, os, re
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
x = io.open("xr_session.py", encoding="utf-8").read()

# ---- imports ----
for mod in ("base64", "io"):
    if not re.search(rf"^import {mod}\b", x, re.M):
        x = x.replace("\nimport json\n", f"\nimport json\nimport {mod}\n", 1)
if "from PIL import Image" not in x:
    x = x.replace("\nimport numpy as np\n", "\nimport numpy as np\nfrom PIL import Image\n", 1)

# ---- extended frame rows ----
a = '''                self.log_file.write(
                    f"{t:.6f},{m['gx']},{m['gy']},{m['lum']:.4f},"
                    f"{m['fl']},{m['pt']:.6f}\\n")
                self.frames += 1
            if m["stage"] == "scan" and self.frames % 12 == 0:'''
a2 = '''                self.log_file.write(
                    f"{t:.6f},{m['gx']},{m['gy']},{m['lum']:.4f},"
                    f"{m['fl']},{m['pt']:.6f},{m.get('flG', 0)},"
                    f"{m.get('flB', 0)},{m.get('r', 0):.3f},{m.get('g', 0):.3f},"
                    f"{m.get('b', 0):.3f},{m.get('hzA', 0):g}\\n")
                self.frames += 1
            if m["stage"] in ("scan", "color") and self.frames % 12 == 0:'''
assert a in x, "frame rows"; x = x.replace(a, a2)
b = '''        self.log_file.write(
            "time,grid_x,grid_y,luminance,flicker_on,page_t\\n")'''
b2 = '''        self.log_file.write(
            "time,grid_x,grid_y,luminance,flicker_on,page_t,"
            "flicker_g,flicker_b,r,g,b,audio_hz\\n")'''
assert b in x, "log header"; x = x.replace(b, b2)
c = '''        elif typ == "visibility":
            print(f"page visibility: {m.get('state')}")'''
c2 = '''        elif typ == "color_plan":
            self.color_hzs = [float(v) for v in m.get("hzs", [])]
            print("colour flicker plan:", self.color_hzs, "@", m.get("refresh"), "fps")
        elif typ == "visibility":
            print(f"page visibility: {m.get('state')}")'''
assert c in x, "color_plan"; x = x.replace(c, c2)
d = '''        self.block_hz = []          # per-block flicker frequency (sweep)'''
d2 = d + '''
        self.color_hzs = []         # page's frame-exact plan for the colour scan'''
assert d in x; x = x.replace(d, d2)

# ---- score_sweep: log optional + per-block values ----
e = '''    L = np.genfromtxt(os.path.join(session, log_name), delimiter=",",
                      skip_header=1)
    lt, lfl, lpt = L[:, 0], L[:, 4], L[:, 5]

    def delivered(t0, t1, fallback):
        s = (lt >= t0) & (lt <= t1)'''
e2 = '''    if log_name and os.path.exists(os.path.join(session, log_name)):
        L = np.genfromtxt(os.path.join(session, log_name), delimiter=",",
                          skip_header=1)
        lt, lfl, lpt = L[:, 0], L[:, 4], L[:, 5]
    else:
        lt = None

    def delivered(t0, t1, fallback):
        if lt is None:
            return float(fallback)
        s = (lt >= t0) & (lt <= t1)'''
assert e in x, "score_sweep log"; x = x.replace(e, e2)
f_ = '''    stds = np.array([[s[3][:, c].std() for c in range(len(names))] for s in segs])
    keep = stds <= 4 * np.median(stds, 0)
    freqs = sorted(set(s[1] for s in segs if s[0] == "on" and s[1]))
    out = {}'''
f2 = '''    stds = np.array([[s[3][:, c].std() for c in range(len(names))] for s in segs])
    keep = stds <= 4 * np.median(stds, 0)
    freqs = sorted(set(s[1] for s in segs if s[0] == "on" and s[1]))
    per_block = [dict(kind=k, hz=hz, delivered=fd,
                      snr={names[c]: float(line_snr(X[:, c], fd or hz or 1.0))
                           for c in range(len(names))})
                 for k, hz, fd, X in segs if (hz or fd)]
    out = {}'''
assert f_ in x, "per_block"; x = x.replace(f_, f2)
g = '''    best_k = min(out, key=lambda k: out[k]["p"])
    p_min = out[best_k]["p"]
    return dict(names=names, freqs=freqs, results=out,
                best_freq=out[best_k]["freq"], best_channel=out[best_k]["best"],
                p_min=p_min, passed=bool(p_min < 0.01 / len(out)),
                segment_s=float(n_min / fs))'''
g2 = '''    best_k = min(out, key=lambda k: out[k]["p"])
    p_min = out[best_k]["p"]
    return dict(names=names, freqs=freqs, results=out,
                best_freq=out[best_k]["freq"], best_channel=out[best_k]["best"],
                p_min=p_min, passed=bool(p_min < 0.01 / len(out)),
                segment_s=float(n_min / fs), per_block=per_block)'''
assert g in x, "sweep return"; x = x.replace(g, g2)

# ---- calibration: line-detector gate + weights ----
h = '''            calib = run_session.score_calibration(
                self.session, self.blocks, stim_freq=self.stim_freq)
            result["stimFreqActual"] = self.stim_freq'''
h2 = '''            calib = run_session.score_calibration(
                self.session, self.blocks, stim_freq=self.stim_freq)
            result["stimFreqActual"] = self.stim_freq
            # exact-line detector on the same blocks: the validated G1 test
            # (sweep2: 5 uV line at 10-12 Hz invisible to the 1 Hz-bin score)
            line = score_sweep(self.session, self.blocks,
                               [self.stim_freq] * len(self.blocks),
                               log_name="calib_log.csv")
            lres = next(iter(line["results"].values()), None)
            if lres:
                wl = {n: max(cc["diff"], 0.0) for n, cc in lres["channels"].items()}
                tot = sum(wl.values())
                calib["weights_welch"] = dict(calib["weights"])
                if tot > 0:
                    calib["weights"] = {n: v / tot for n, v in wl.items()}
                calib["line"] = dict(p=lres["p"], best=lres["best"],
                                     stat=lres["stat"], delivered=lres["delivered"],
                                     channels=lres["channels"],
                                     per_block=line.get("per_block"))
                calib["passed_welch"] = calib["passed"]
                calib["passed"] = bool(lres["p"] < config.CALIB_P_MAX)
                calib["gate"] = "line-permutation"
                calib["best"] = lres["best"]
                with open(os.path.join(self.session, "calibration.json"), "w") as f:
                    json.dump(calib, f, indent=1)'''
assert h in x, "calib gate"; x = x.replace(h, h2)
i_ = '''            table += (f"   |  gate: {calib.get('gate')}"
                      f"  p_fw={perm.get('p_fw', float('nan')):.4f}"
                      f" (n_null={perm.get('n_null', 0)})")'''
i2 = '''            table += (f"   |  welch p_fw={perm.get('p_fw', float('nan')):.4f}"
                      f" (n_null={perm.get('n_null', 0)})")
            if calib.get("line"):
                ln = calib["line"]
                table = (f"LINE {ln['delivered']:.2f} Hz p={ln['p']:.4f} "
                         f"best {ln['best']} stat {ln['stat']:+.2f}  ||  " + table)'''
assert i_ in x, "calib table"; x = x.replace(i_, i2)

# ---- grey reconstruction: method + null ----
j = '''            grid, r = reconstruct.run(
                self.session,
                calibration=os.path.join(self.session, "calibration.json"),
                stim_freq=self.stim_freq)
            flicker_hz = self.measured_flicker()
            result.update(r=r, measuredFlickerHz=flicker_hz)'''
j2 = '''            grid, r = reconstruct.run(
                self.session,
                calibration=os.path.join(self.session, "calibration.json"),
                stim_freq=self.stim_freq, method=a.recon_method)
            flicker_hz = self.measured_flicker()
            rs_null = reconstruct.null_r(
                self.session, n_shifts=8,
                calibration=os.path.join(self.session, "calibration.json"),
                stim_freq=self.stim_freq, method=a.recon_method)
            result.update(r=r, measuredFlickerHz=flicker_hz, method=a.recon_method,
                          r_null=rs_null,
                          r_null_max=(max(rs_null) if rs_null else None))
            print(f"grey image r={r:.3f} ({a.recon_method}); shifted-EEG null "
                  f"r: {', '.join(f'{v:.2f}' for v in rs_null)}")'''
assert j in x, "grey recon"; x = x.replace(j, j2)

# ---- full mode: extras after the grey result ----
k = '''            result["ok"] = True
            recorder.stop()  # stop at scan end, not after the linger
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")'''
k2 = '''            if a.mode == "full":
                await asyncio.sleep(4.0)  # let the subject see the image
                await self.run_full_extras(result)
            result["ok"] = True
            recorder.stop()  # stop at scan end, not after the linger
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")'''
assert k in x, "full hook"; x = x.replace(k, k2)

# ---- run_full_extras ----
l_ = '''    async def run_sweep(self, result, recorder):'''
l2 = '''    def measured_cols(self, log_name, cols=(4, 6, 7)):
        """Delivered frequency per flicker column from page-clock rising edges."""
        path = os.path.join(self.session, log_name)
        out = []
        try:
            L = np.genfromtxt(path, delimiter=",", skip_header=1)
        except OSError:
            return [float("nan")] * len(cols)
        if L.ndim == 1 or L.shape[0] < 10:
            return [float("nan")] * len(cols)
        for c in cols:
            v = L[:, c]
            ed = np.where((v[1:] == 1) & (v[:-1] == 0))[0] + 1
            per = np.diff(L[ed, 5])
            per = per[(per > 0.01) & (per < 1.0)]
            out.append(float(1 / np.median(per)) if len(per) >= 5 else float("nan"))
        return out

    async def run_full_extras(self, result):
        """Colour (frequency-tagged RGB) and music (40 Hz-tagged) stages."""
        a = self.args
        freqs = [float(v) for v in str(a.color_freqs).split(",") if v.strip()]
        names = csv_header(os.path.join(self.session, "eeg.csv")) or CH_NAMES
        calib = json.load(open(os.path.join(self.session, "calibration.json")))
        weights = np.array([float(calib["weights"].get(n, 0.0)) for n in names])
        if weights.sum() <= 0:
            weights = np.ones(len(names))

        # ---- colour calibration: full-field R, G, B flicker at f_R, f_G, f_B
        self.spectate(type="stage", stage="color_calib",
                      detail=",".join(f"{f:g}" for f in freqs))
        self.blocks, self.block_hz = [], []
        self.open_log("ccal_log.csv")
        cols = {f"{f:g}": [c, "#000000"] for f, c in
                zip(freqs, ("#ff0000", "#00ff00", "#0000ff"))}
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.color_reps,
                        onS=a.color_on, offS=a.color_on, colorsPerFreq=cols,
                        size=1.0)
        total = a.color_reps * len(freqs) * 2 * a.color_on
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        ccal = score_sweep(self.session, self.blocks, self.block_hz,
                           log_name="ccal_log.csv")
        ordered = sorted(ccal["results"].values(), key=lambda rr: rr["freq"])
        gains = []
        for k, f in enumerate(freqs):
            rr = ordered[k] if k < len(ordered) else None
            if rr is None:
                gains.append(1.0)
                continue
            num = sum(weights[i] * max(rr["channels"][n]["on_med"]
                                       - rr["channels"][n]["off_med"], 0.0)
                      for i, n in enumerate(names))
            gains.append(float(max(num / weights.sum(), 0.05)))
        ccal["gains"] = gains
        ccal["requested"] = freqs
        with open(os.path.join(self.session, "ccal.json"), "w") as f:
            json.dump(ccal, f, indent=1)
        ctab = "  ".join(f"{rr['freq']:g}Hz p={rr['p']:.3f} {rr['best']}"
                         for rr in ordered)
        print(f"colour calibration: {ctab}  gains={['%.2f' % g for g in gains]}")
        result["color_calib"] = dict(table=ctab, gains=gains,
                                     p=[rr["p"] for rr in ordered])
        await self.send(cmd="msg", text="colour calibration done: " + ctab)
        await asyncio.sleep(2.0)

        # ---- colour scan
        gw, gh = a.color_grid_w, a.color_grid_h
        ctarget = targets.color_target(a.color_target, gw, gh)
        np.save(os.path.join(self.session, "target_color.npy"), ctarget)
        self.spectate(type="stage", stage="color_scan", detail="")
        self.color_hzs = []
        self.open_log("color_log.csv")
        await self.send(cmd="start_scan", color=True, gridW=gw, gridH=gh,
                        spc=a.color_spc, freqs=freqs, target=ctarget.tolist())
        scan_s = gw * gh * a.color_spc
        await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        hz_del = self.measured_cols("color_log.csv")
        for k in range(3):
            if not np.isfinite(hz_del[k]):
                hz_del[k] = (self.color_hzs[k] if k < len(self.color_hzs)
                             else freqs[k])
        print("colour scan delivered:", ["%.2f" % v for v in hz_del])
        try:
            grid, rgb, cres = reconstruct.reconstruct_color(
                self.session, hz_del, weights=weights, gains=gains,
                cursor_log="color_log.csv", target=ctarget)
        except Exception as exc:  # keep the session alive for the music stage
            print("colour reconstruction failed:", repr(exc))
            cres = dict(r_all=None, r_planes=None, hue_accuracy=None,
                        error=repr(exc))
            rgb = None
        text = ("colour r=%s  (R %s, G %s, B %s)  hue accuracy %s" % (
            "n/a" if cres.get("r_all") is None else f"{cres['r_all']:.2f}",
            *(("n/a",) * 3 if not cres.get("r_planes") else
              tuple(f"{v:.2f}" for v in cres["r_planes"])),
            "n/a" if cres.get("hue_accuracy") is None
            else f"{cres['hue_accuracy']:.0%}"))
        print(text)
        result["color"] = dict(freqs_requested=freqs, freqs_delivered=hz_del,
                               gains=gains, **{k2: v for k2, v in cres.items()
                                               if k2 != "out"})
        png = rgb_to_data_url(rgb) if rgb is not None else ""
        await self.send(cmd="color_result", png=png, text=text,
                        good=bool(cres.get("r_all") and cres["r_all"] >= 0.6))
        self.spectate(type="log", msg=text)
        await asyncio.sleep(5.0)

        # ---- music piece tagged at 40 Hz (ASSR = ear as a level meter)
        self.spectate(type="stage", stage="music", detail="")
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
        # legacy 1 Hz-bin score too (it overwrites calibration.json: restore)
        cal_path = os.path.join(self.session, "calibration.json")
        cal_bytes = open(cal_path, "rb").read()
        try:
            legacy = run_session.score_calibration(self.session, self.blocks,
                                                   stim_freq=40.0)
            os.replace(cal_path, os.path.join(self.session, "assr_welch.json"))
        except Exception as exc:
            legacy = dict(passed=None, error=repr(exc))
        finally:
            open(cal_path, "wb").write(cal_bytes)
        mus["legacy_passed"] = legacy.get("passed")
        with open(os.path.join(self.session, "music.json"), "w") as f:
            json.dump(mus, f, indent=1)
        if mres:
            mtext = (f"40 Hz tag: p={mres['p']:.4f} best {mres['best']} "
                     f"(ON {mres['channels'][mres['best']]['on_med']:.2f} / "
                     f"OFF {mres['channels'][mres['best']]['off_med']:.2f})")
            passed = bool(mres["p"] < config.CALIB_P_MAX)
        else:
            mtext, passed = "music stage: no scorable blocks", False
        print("music " + mtext)
        result["music"] = dict(passed=passed, text=mtext,
                               p=mres["p"] if mres else None,
                               per_block=mus.get("per_block"))
        await self.send(cmd="assr_result", passed=passed, table=mtext)
        self.spectate(type="log", msg="music " + mtext)
        await asyncio.sleep(3.0)

    async def run_sweep(self, result, recorder):'''
assert l_ in x, "extras anchor"; x = x.replace(l_, l2, 1)

# ---- rgb_to_data_url helper (next to grid_to_data_url) ----
m_ = '''def grid_to_data_url(norm, scale=24):'''
m2 = '''def rgb_to_data_url(rgb, scale=24):
    """(h, w, 3) floats in [0,1] -> PNG data URL, nearest-neighbour upscaled."""
    arr = (np.clip(np.asarray(rgb, float), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_to_data_url(norm, scale=24):'''
assert m_ in x, "data url"; x = x.replace(m_, m2, 1)

# ---- CLI ----
n_ = '''    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",
                                       "sweep"],'''
n2 = '''    ap.add_argument("--recon-method", choices=["welch", "line"], default="line",
                    help="per-cell score: welch = 1 Hz-bin relative power "
                         "(paper), line = exact-frequency line SNR")
    ap.add_argument("--color-freqs", default="7.2,9,12",
                    help="--mode full: R,G,B tag frequencies (frame-exact at "
                         "72 fps: 7.2/9/12; at 240 fps: 8/10/12)")
    ap.add_argument("--color-reps", type=int, default=3)
    ap.add_argument("--color-on", type=float, default=6.0,
                    help="colour calibration ON = OFF seconds per block")
    ap.add_argument("--color-grid-w", type=int, default=6)
    ap.add_argument("--color-grid-h", type=int, default=4)
    ap.add_argument("--color-spc", type=float, default=5.0)
    ap.add_argument("--color-target", default="flag",
                    help="flag | quad | ring | text:X | image path")
    ap.add_argument("--music-blocks", type=int, default=6)
    ap.add_argument("--music-on", type=float, default=10.0)
    ap.add_argument("--music-off", type=float, default=10.0)
    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",
                                       "sweep", "full"],'''
assert n_ in x, "cli"; x = x.replace(n_, n2)
io.open("xr_session.py", "w", encoding="utf-8", newline="\n").write(x)
print("patched xr_session.py")
