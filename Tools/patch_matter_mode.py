# -*- coding: utf-8 -*-
"""One-off (2026-11-09): add `--mode matter` (Mind to Matter) to xr_session.py and the watch / found
stages to xr_stimulus.html. Idempotent: run it again and it changes nothing."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rw(path, pairs):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        s = f.read()
    nl = "\r\n" if "\r\n" in s else "\n"
    n = 0
    for old, new in pairs:
        old_n, new_n = old.replace("\n", nl), new.replace("\n", nl)
        if new_n in s and old_n not in s:
            continue
        assert s.count(old_n) == 1, "not exactly once: " + old[:90]
        s = s.replace(old_n, new_n)
        n += 1
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    print(os.path.basename(path), n, "replacements")


RUN_MATTER = '''    async def run_matter(self, result, recorder, calib=None):
        """Mind to Matter: watch a clip (or a picture), pull the trigger on the thing, have it
        found in the frame, scan it square by square at the calibrated flicker while the eyes
        follow the dot, read the brain's picture of it back, and make the prints (matter.py)."""
        a = self.args
        import matter
        media = a.matter_media
        if not media or not os.path.exists(media):
            raise RuntimeError("--mode matter needs --matter-media <clip or picture>")
        name = os.path.basename(media)
        kind = "image" if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else "video"
        self.spectate(type="stage", stage="watch", detail=name)
        print(f"watch: {name} ({kind}); pull the trigger on the thing to print")
        await self.send(cmd="start_watch", src=f"/media/{name}", kind=kind,
                        autoPause=a.matter_auto_pause,
                        text="watch; when the thing you want to print is on screen, "
                             "<b>pull the trigger</b>")
        m = await self.wait_for("frame_capture", timeout=a.signal_timeout + 1800)
        frame_path = matter.save_capture(self.session, m["jpg"], m.get("rect"))
        print(f"frame captured at {float(m.get('vt') or 0):.1f} s -> {frame_path}")
        self.spectate(type="stage", stage="find", detail="")
        await self.send(cmd="msg", text="finding the thing you were looking at...")
        found = await asyncio.to_thread(matter.find_object, frame_path, self.session,
                                        a.matter_grid, a.matter_cells, a.matter_depth, a.matter_name)
        gw, gh = found["gridW"], found["gridH"]
        target = np.asarray(found["target"], float)
        np.save(os.path.join(self.session, "target.npy"), target)
        result.update(gridW=gw, gridH=gh)
        print(f"found by {found['method']}: {found['mask_area']:.0%} of the frame, grid {gw}x{gh}, "
              f"{len(found['cells'])} squares to scan"
              + ("" if found["depth_ok"] else "; no depth model: the 3-D will be a pillow relief"))
        await self.send(cmd="show_find", image=found["overlay_data_url"], seconds=a.matter_show,
                        title="FOUND",
                        text=(f"{found['label'] + ': ' if found['label'] else ''}"
                              f"{len(found['cells'])} squares, {a.spc:g} s each, {max(1, a.passes)} pass(es). "
                              "Look at the dot and follow it; each square flickers while you look at it."))
        await asyncio.sleep(a.matter_show + 1.5)
        # the scan: every square of the thing, each pass in a fresh random order
        rng = np.random.default_rng(1)
        cells = []
        for _ in range(max(1, a.passes)):
            order = rng.permutation(len(found["cells"]))
            cells += [list(found["cells"][i]) for i in order]
        scan_kw = dict(gridW=gw, gridH=gh, spc=a.spc, freq=a.freq, target=target.tolist(),
                       patch=a.patch, patchSide=a.patch_side, bw=True, board=a.board,
                       restEvery=a.rest_every)
        self.spectate(type="stage", stage="scan", detail=f"{len(cells)} squares")
        self.open_log("cursor_log.csv")
        await self.send(cmd="start_scan", cells=cells,
                        label=f"scanning {len(found['cells'])} squares", **scan_kw)
        await self.wait_for("scan_done", timeout=len(cells) * a.spc * 2 + 180)
        self.close_log()
        await asyncio.sleep(0.7)
        method = a.recon_method if a.recon_method in ("line", "paper", "welch") else "line"
        calib_path = os.path.join(self.session, "calibration.json")
        calib_arg = calib_path if os.path.exists(calib_path) else ""
        grid, r = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                  method=method, vblend=False)
        nulls = []   # the same scoring on misaligned EEG: what chance gives
        for k in range(8):
            try:
                _, r0 = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                        method=method, vblend=False, save=False,
                                        shift_s=23.0 * (k + 1) + 7.0)
                if r0 is not None:
                    nulls.append(float(r0))
            except Exception as exc:
                print("null:", repr(exc)[:120])
                break
        null_max = max(nulls) if nulls else None
        print(f"the brain's picture against the frame's squares: r = {r if r is not None else float('nan'):.3f}"
              + (f" (shifted-EEG null max {null_max:.3f} over {len(nulls)})" if null_max is not None else ""))
        files = await asyncio.to_thread(matter.make_prints, self.session, grid, found, a.matter_slice)
        png = grid_to_data_url(reconstruct_norm(np.nan_to_num(np.asarray(grid, float))))
        text = (f"the brain's picture: r = {(r or 0.0):.2f}"
                + (f" (null max {null_max:.2f})" if null_max is not None else "")
                + "   prints: " + ", ".join(os.path.basename(p) for p in files["stl"]))
        await self.send(cmd="result", png=png, r=r or 0.0, flickerHz=self.stim_freq, text=text)
        result.update(ok=True, matter=dict(
            found={k: v for k, v in found.items() if k != "overlay_data_url"}, r=r, null_r=nulls,
            null_max=null_max, files={k: v for k, v in files.items() if k != "reports"},
            reports=files.get("reports")))
        recorder.stop()
        print(f"MIND TO MATTER DONE -> {self.session}  (prints in {files.get('copied_to')})")
        await asyncio.sleep(a.linger)
        return 0

'''

MEDIA_ROUTE = '''MEDIA_FILES = {}   # name -> path of the clip or picture served at /media/<name> (--matter-media)
_MEDIA_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def http_media(request):
    """Serve --matter-media to the page with byte ranges (a <video> asks for them)."""
    name = request.path[len("/media/"):].split("?", 1)[0]
    path = MEDIA_FILES.get(name)
    if not path or not os.path.exists(path):
        return Response(404, "Not Found", Headers([("Content-Type", "text/plain"),
                                                   ("Content-Length", "9")]), b"not found")
    with open(path, "rb") as f:
        data = f.read()
    ctype = _MEDIA_TYPES.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
    total = len(data)
    rng = request.headers.get("Range", "")
    start, end = 0, total - 1
    status, reason = 200, "OK"
    if rng.startswith("bytes="):
        a_, _, b_ = rng[6:].partition("-")
        try:
            start = int(a_) if a_ else max(0, total - int(b_))
            end = int(b_) if (b_ and a_) else total - 1
            end = min(end, total - 1)
            if start <= end:
                status, reason = 206, "Partial Content"
        except ValueError:
            start, end = 0, total - 1
    body = data[start:end + 1]
    hdrs = [("Content-Type", ctype), ("Content-Length", str(len(body))), ("Accept-Ranges", "bytes"),
            ("Cache-Control", "no-store"), ("Connection", "close")]
    if status == 206:
        hdrs.append(("Content-Range", f"bytes {start}-{end}/{total}"))
    return Response(status, reason, Headers(hdrs), body)


def http_page(connection, request):
    if request.path.startswith("/ws"):
        return None
    if request.path.startswith("/media/"):
        return http_media(request)
'''

PAGE_JS = '''
// ---- Mind to Matter: watch a clip (or a picture), pull the trigger on the thing, hand the
// frame to the computer; then the thing as it was found, before the square-by-square scan
function drawMedia(el, c = cv, x = ctx) {
  const w = el.videoWidth || el.naturalWidth || el.width || 0, h = el.videoHeight || el.naturalHeight || el.height || 0;
  x.fillStyle = "#000"; x.fillRect(0, 0, c.width, c.height);
  if (!w || !h) return [0, 0, 0, 0];
  const s = Math.min(c.width / w, c.height / h), dw = w * s, dh = h * s, dx = (c.width - dw) / 2, dy = (c.height - dh) / 2;
  try { x.drawImage(el, dx, dy, dw, dh); } catch {}
  return [dx, dy, dw, dh];
}
async function runWatch(m) {
  hideResult(); title.textContent = m.title || "WATCH"; title.className = "ok";
  sub.innerHTML = m.text || "watch; when the thing you want to print is on screen, <b>pull the trigger</b>";
  const tok = running = {};
  black();
  let el;
  if (m.kind === "image") {
    el = new Image(); el.src = m.src;
    await new Promise(r => { el.onload = r; el.onerror = r; });
  } else {
    el = document.createElement("video");
    el.muted = true; el.loop = true; el.playsInline = true; el.preload = "auto";
    el.setAttribute("playsinline", ""); el.setAttribute("muted", "");
    el.src = m.src;
    try { await el.play(); } catch (e) { sub.textContent = "the clip could not start (" + e + "); pull the trigger to use the first frame"; }
  }
  const auto = location.search.includes("auto=1");
  let pressed = false;
  const press = () => { pressed = true; };
  await sleep(700);                      // the arming press may still be held
  addEventListener("pointerdown", press, { once: true });
  addEventListener("keydown", press, { once: true });
  const gp = setInterval(() => {
    try {
      for (const g of navigator.getGamepads ? navigator.getGamepads() : [])
        if (g && g.buttons.some(b => b && (b.pressed || b.value > 0.5))) pressed = true;
    } catch {}
  }, 100);
  const t0 = now();
  while (running === tok && !pressed) {
    await frame();
    drawMedia(el);
    if (auto && now() - t0 > (m.autoPause || 3)) pressed = true;
  }
  clearInterval(gp);
  removeEventListener("pointerdown", press); removeEventListener("keydown", press);
  if (running !== tok) return;
  if (el.pause) el.pause();
  drawMedia(el);
  // the frame as seen, at most 1280 px wide, as a JPEG the computer can take over the socket
  const cap = document.createElement("canvas");
  const s = Math.min(1, 1280 / cv.width);
  cap.width = Math.round(cv.width * s); cap.height = Math.round(cv.height * s);
  const rect = drawMedia(el, cap, cap.getContext("2d"));
  const jpg = cap.toDataURL("image/jpeg", 0.85);
  send({ type: "frame_capture", jpg, rect, vt: el.currentTime || 0, w: cap.width, h: cap.height });
  title.textContent = "PAUSED"; sub.textContent = "finding the thing you were looking at\\u2026";
}
async function showFind(m) {
  hideResult(); title.textContent = m.title || "FOUND"; title.className = "ok"; sub.innerHTML = m.text || "";
  const tok = running = {};
  const img = new Image(); img.src = m.image;
  await new Promise(r => { img.onload = r; img.onerror = r; });
  const t0 = now();
  while (running === tok && now() - t0 < (m.seconds || 5)) { await frame(); drawMedia(img); }
  if (running !== tok) return;
  black(); dot(cv.width / 2, cv.height / 2);
  sub.textContent = "look at the dot and follow it; each square flickers while you look at it";
}
'''

SESSION_PAIRS = [
    # the mode
    ('''                                       "music", "bwb", "smooth", "planes", "exam"],''',
     '''                                       "music", "bwb", "smooth", "planes", "exam", "matter"],'''),
    # its options
    ('''    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",''',
     '''    ap.add_argument("--matter-media", default="",
                    help="matter: the clip (mp4/webm) or picture the person watches; served to the page at /media/")
    ap.add_argument("--matter-grid", default="7x5",
                    help="matter: at most this many columns x rows over the thing (squares stay near square)")
    ap.add_argument("--matter-cells", type=int, default=20,
                    help="matter: at most this many squares are scanned (the ones most inside the thing)")
    ap.add_argument("--matter-show", type=float, default=6.0,
                    help="matter: seconds the found thing and its grid are shown before the scan")
    ap.add_argument("--matter-name", default="",
                    help="matter: what the thing is, for the overlay and the report (optional)")
    ap.add_argument("--matter-depth", default="auto", choices=["auto", "off"],
                    help="matter: use the depth model for the 3-D (auto) or a pillow relief (off)")
    ap.add_argument("--matter-auto-pause", type=float, default=3.0,
                    help="matter: with ?auto=1 the page pauses the clip after this many seconds")
    ap.add_argument("--matter-slice", action="store_true",
                    help="matter: slice the prints for the H2C at the end (minutes)")
    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",'''),
    # a reconnecting page gets the watch / found stage again
    ('''                              "alpha_result", "start_sweep", "sweep_result",
                              "start_bwb_calib"):
            self.last_cmd = msg  # re-issued to a page that reconnects''',
     '''                              "alpha_result", "start_sweep", "sweep_result",
                              "start_bwb_calib", "start_watch", "show_find"):
            self.last_cmd = msg  # re-issued to a page that reconnects'''),
    ('''            "start_calib", "start_scan", "start_assr", "start_alpha",
            "start_sweep", "start_bwb_calib", "start_exam")''',
     '''            "start_calib", "start_scan", "start_assr", "start_alpha",
            "start_sweep", "start_bwb_calib", "start_exam", "start_watch")'''),
    # the branch, after the calibration
    ('''            if a.mode == "planes":
                return await self.run_planes(result, recorder)
''',
     '''            if a.mode == "planes":
                return await self.run_planes(result, recorder)
            if a.mode == "matter":
                return await self.run_matter(result, recorder, calib)
'''),
    # the method
    ('''    async def certainty_scan(self, scan_kw, log_name, grid_w, grid_h, spc, state):''',
     RUN_MATTER + '''    async def certainty_scan(self, scan_kw, log_name, grid_w, grid_h, spc, state):'''),
    # the media route
    ('''def http_page(connection, request):
    if request.path.startswith("/ws"):
        return None
''', MEDIA_ROUTE),
    # register the media file before serving
    ('''    async with websockets.serve(ws_handler, "0.0.0.0", args.http_port,''',
     '''    if getattr(args, "matter_media", ""):
        MEDIA_FILES[os.path.basename(args.matter_media)] = os.path.abspath(args.matter_media)
    async with websockets.serve(ws_handler, "0.0.0.0", args.http_port,'''),
]

PAGE_PAIRS = [
    ('''      case "start_exam": runExam(m); break;''',
     '''      case "start_exam": runExam(m); break;
      case "start_watch": runWatch(m); break;
      case "show_find": showFind(m); break;'''),
    ('''const frame = () => new Promise(requestAnimationFrame);
const sleep = ms => new Promise(r => setTimeout(r, ms));''',
     '''const frame = () => new Promise(requestAnimationFrame);
const sleep = ms => new Promise(r => setTimeout(r, ms));
''' + PAGE_JS),
]

if __name__ == "__main__":
    rw(os.path.join(ROOT, "xr_session.py"), SESSION_PAIRS)
    rw(os.path.join(ROOT, "xr_stimulus.html"), PAGE_PAIRS)
