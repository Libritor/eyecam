"""Driver side of patch_colour_modes.py: --smooth-rotate, --mode planes
(sequential R, G, B scans at the calibrated frequency, scored like the "NO")
and the new colour targets. Run once from the repo root."""
import io

P = "xr_session.py"
x = io.open(P, encoding="utf-8").read()
assert "run_planes" not in x, "already patched"


def rep(old, new):
    global x
    assert x.count(old) == 1, (old[:70], x.count(old))
    x = x.replace(old, new)


# ---- smooth: rotation flag through to the page and the meta file
rep('''        await self.send(cmd="start_scan", smooth=True, gridW=gw, gridH=gh,
                        spc=a.smooth_spc, passes=a.smooth_passes, freqs=tags,
                        target=ctarget.tolist(), patch=a.patch, board=a.board)''',
    '''        rotate = bool(a.smooth_rotate) and a.smooth_passes >= 3
        if a.smooth_rotate and not rotate:
            print("NOTE: tag rotation needs --smooth-passes 3 (or 6, 9...); "
                  "colours keep fixed tags this run")
        await self.send(cmd="start_scan", smooth=True, gridW=gw, gridH=gh,
                        spc=a.smooth_spc, passes=a.smooth_passes, freqs=tags,
                        rotate=rotate, target=ctarget.tolist(), patch=a.patch,
                        board=a.board)''')
rep('''                           spc=a.smooth_spc, passes=a.smooth_passes, balance=balance,
                           exponent=a.shade_exponent, alpha=alpha, note=note),''',
    '''                           spc=a.smooth_spc, passes=a.smooth_passes, balance=balance,
                           rotate=rotate, exponent=a.shade_exponent, alpha=alpha,
                           note=note),''')

# ---- planes mode: hooked in after the calibration, where bwb is chosen
rep('''            result["choice"] = choice or "bw"
            print(f"picture: {result['choice']}")
            if choice == "bwb":
                return await self.run_bwb(result, recorder)
''', '''            result["choice"] = choice or "bw"
            print(f"picture: {result['choice']}")
            if choice == "bwb":
                return await self.run_bwb(result, recorder)
            if a.mode == "planes":
                return await self.run_planes(result, recorder)
''')

rep('''    async def run_smooth(self, result, recorder):''',
    '''    async def run_planes(self, result, recorder):
        """Sequential colour planes: the picture's R, G and B are each
        scanned as a black/<colour> scan at the calibrated frequency, one
        after the other, and scored exactly like the "NO" (paper_score,
        calibration weights), then assembled into RGB. Every colour gets the
        one frequency this subject responds to best, so the colour balance
        is the picture's and not a frequency preference; the price is three
        scans instead of one."""
        import smoothcolor
        a = self.args
        cal_path = os.path.join(self.session, "calibration.json")
        ctarget = targets.color_target(a.color_target, a.color_grid_w, a.color_grid_h)
        gh, gw = ctarget.shape[:2]
        np.save(os.path.join(self.session, "target_color.npy"), ctarget)
        order = [int(v) for v in str(a.plane_order).split(",")]
        unit = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        names = ["red", "green", "blue"]
        logs = {}
        for k in order:
            plane = ctarget[:, :, k]
            if plane.max() <= 0.02:
                print(f"{names[k]} plane is empty in {a.color_target}: skipped")
                continue
            log = f"plane_{names[k]}_log.csv"
            logs[k] = log
            self.spectate(type="stage", stage="scan", detail=f"{names[k]} plane")
            print(f"scanning the {names[k]} plane ({gw}x{gh}, {a.plane_spc:g} s per "
                  f"position, {a.plane_passes} pass(es)) at {self.stim_freq:.1f} Hz")
            await self.send(cmd="msg", text=f"{names[k]} plane")
            await asyncio.sleep(1.5)
            self.open_log(log)
            await self.send(cmd="start_scan", gridW=gw, gridH=gh, spc=a.plane_spc,
                            freq=a.freq, target=plane.tolist(), patch=a.patch,
                            bw=True, plane=unit[k], passes=a.plane_passes,
                            board=a.board, restEvery=a.rest_every)
            scan_s = gw * gh * a.plane_spc * max(1, a.plane_passes)
            if a.rest_every > 0:
                scan_s += 600 * (gw * gh * max(1, a.plane_passes) // a.rest_every)
            await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
            self.close_log()
            await asyncio.sleep(0.7)
        with open(os.path.join(self.session, "planes_meta.json"), "w") as f:
            json.dump(dict(logs={names[k]: v for k, v in logs.items()}, spc=a.plane_spc,
                           passes=a.plane_passes, freq=self.stim_freq,
                           method=a.recon_method, target=a.color_target), f, indent=1)

        # ---- decode: each plane like the grey picture, then RGB
        print("reconstructing the colour planes...")

        def planes_grid(shift_s=0.0):
            grid = np.zeros((gh, gw, 3))
            for k, log in logs.items():
                g, _ = reconstruct.run(self.session, calibration=cal_path,
                                       stim_freq=self.stim_freq, method=a.recon_method,
                                       vblend=False, cursor_log=log, save=False,
                                       shift_s=shift_s)
                grid[:, :, k] = g
            return grid

        try:
            grid = await asyncio.to_thread(planes_grid)
            np.save(os.path.join(self.session, "reconstruction_planes_grid.npy"), grid)
            rgb = smoothcolor.to_rgb(grid)
            res = smoothcolor.metrics(rgb, ctarget)
            res["r_plane_raw"] = [float(np.corrcoef(ctarget[:, :, k].ravel(),
                                                    grid[:, :, k].ravel())[0, 1])
                                  if k in logs and ctarget[:, :, k].std() > 0 else None
                                  for k in range(3)]
            nulls = []
            for n in range(4):
                try:
                    nulls.append(smoothcolor.metrics(
                        smoothcolor.to_rgb(await asyncio.to_thread(planes_grid, 31.0 + 29.0 * n)),
                        ctarget)["r_all"])
                except Exception as exc:
                    print("null", n, repr(exc))
            res["r_null"] = nulls
            comp = smoothcolor.save_images(self.session, rgb, ctarget)
            os.replace(os.path.join(self.session, "reconstruction_smooth.png"),
                       os.path.join(self.session, "reconstruction_planes.png"))
            if comp:
                os.replace(comp, os.path.join(self.session, "planes_comparison.png"))
                comp = os.path.join(self.session, "planes_comparison.png")
            res["comparison"] = comp
            text = smoothcolor.summary(res)
            good = res["r_all"] >= 0.6 and res["r_all"] > max(nulls or [0])
        except Exception as exc:
            print("colour plane decoding failed:", repr(exc))
            res, rgb, comp, text, good = dict(error=repr(exc)), None, None, f"decoding failed: {exc}", False
        print(text)
        result["planes"] = res
        if comp:
            with open(comp, "rb") as f:
                png = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        else:
            png = rgb_to_data_url(rgb) if rgb is not None else ""
        await self.send(cmd="color_result", png=png,
                        text=f"{text}   [planes at {self.stim_freq:.1f} Hz]",
                        passed=bool(good), wide=bool(comp))
        result["ok"] = True
        recorder.stop()
        print(f"COLOUR PLANES DONE -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_smooth(self, result, recorder):''')

# ---- arguments
rep('''    ap.add_argument("--color-target", default="flag",
                    help="flag | quad | ring | mix | text:X | image path")''',
    '''    ap.add_argument("--color-target", default="flag",
                    help="flag | quad | ring | mix | eight | hues | text:X | image path")
    ap.add_argument("--plane-spc", type=float, default=8.0,
                    help="--mode planes: seconds per position per plane")
    ap.add_argument("--plane-passes", type=int, default=1,
                    help="--mode planes: passes per plane")
    ap.add_argument("--plane-order", default="2,0,1",
                    help="--mode planes: order of the planes (0 R, 1 G, 2 B); "
                         "the dimmest first, while the subject is fresh")''')
rep('''    ap.add_argument("--smooth-passes", type=int, default=1)''',
    '''    ap.add_argument("--smooth-passes", type=int, default=3)
    ap.add_argument("--smooth-rotate", type=int, default=1,
                    help="--mode smooth: 1 = the colour-to-tag assignment rotates "
                         "every pass (needs 3 passes or a multiple), so the "
                         "subject's frequency preference cancels out of the "
                         "colour balance; 0 = fixed tags")''')
rep('''                                       "music", "bwb", "smooth"],''',
    '''                                       "music", "bwb", "smooth", "planes"],''')
io.open(P, "w", encoding="utf-8", newline="\n").write(x)
print("patched", P)
