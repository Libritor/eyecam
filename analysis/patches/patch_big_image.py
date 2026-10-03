"""2026-10-03: larger test images.

- an image file that is already at scan resolution (a pixel drawing such as
  targets_img/mann_eyetap_12.png) sets the scan grid to its own size
- --rest-every N: after every N positions the page stops, says so, and waits
  for the trigger, so a 20-minute pass can be done in pieces
Run once from the repo root."""
import io


def patch(p, pairs):
    x = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert x.count(a) == 1, (p, a[:70], x.count(a))
        x = x.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="\n").write(x)


patch("xr_stimulus.html", [
    # a press helper shared by the rest breaks
    ('''// The driver pauses the scan when the EEG stream goes silent. The position''',
     '''// Wait for a trigger pull, click, key or controller button. In unattended
// runs (?auto=1) it returns after a second.
async function waitForPress() {
  if (location.search.includes("auto=1")) { await sleep(1000); return; }
  await sleep(600);   // ignore the press that may still be held
  await new Promise(res => {
    const done = () => { clearInterval(gp); res(); };
    addEventListener("pointerdown", done, { once: true });
    addEventListener("keydown", done, { once: true });
    const gp = setInterval(() => {
      try {
        for (const g of navigator.getGamepads ? navigator.getGamepads() : []) {
          if (g && g.buttons.some(b => b && (b.pressed || b.value > 0.5))) return done();
        }
      } catch {}
    }, 100);
  });
}

// The driver pauses the scan when the EEG stream goes silent. The position'''),
    ('''  const passes = Math.max(1, m.passes || 1);
  for (let ps = 0; ps < passes && running === tok; ps++)
  for (let ry = 0; ry < gh && running === tok; ry++) {''',
     '''  const passes = Math.max(1, m.passes || 1);
  const total = passes * gw * gh, restEvery = m.restEvery || 0;
  let nDone = 0;
  for (let ps = 0; ps < passes && running === tok; ps++)
  for (let ry = 0; ry < gh && running === tok; ry++) {'''),
    ('''               lum, fl: fl ? 1 : 0 });
      }
      if (paused) { xi--; continue; }
      if (running !== tok) break;''',
     '''               lum, fl: fl ? 1 : 0 });
      }
      if (paused) { xi--; continue; }
      if (running !== tok) break;
      nDone++;
      if (restEvery > 0 && nDone % restEvery === 0 && nDone < total) {
        // a rest break: mark it in the log so it is not scored as a position
        black();
        send({ type: "frame", stage: "rest", pt: now(), gx: -1, gy: -1, lum: 0, fl: 0 });
        title.textContent = "REST"; title.className = "ok";
        sub.textContent = `${nDone} of ${total} positions done. Blink, relax, ` +
                          `then pull the trigger to continue.`;
        await waitForPress();
        title.textContent = "";
        for (let n = 3; n > 0 && running === tok; n--) {
          sub.textContent = `continuing in ${n}`;
          black(); dot(x0 + (gx + 0.5) * cwid, y0 + (gy + 0.5) * chgt);
          await sleep(1000);
        }
        sub.textContent = passes > 1 ? `pass ${ps + 1}/${passes}` : "";
      }'''),
])

patch("xr_session.py", [
    ('''                            patch=a.patch, bw=(a.calib_style == "bw"),
                            passes=a.passes, board=a.board)
            scan_s = a.grid_w * a.grid_h * a.spc * max(1, a.passes)''',
     '''                            patch=a.patch, bw=(a.calib_style == "bw"),
                            passes=a.passes, board=a.board,
                            restEvery=a.rest_every)
            scan_s = a.grid_w * a.grid_h * a.spc * max(1, a.passes)
            if a.rest_every > 0:   # rest breaks are as long as the subject likes
                scan_s += 600 * (a.grid_w * a.grid_h * max(1, a.passes)
                                 // a.rest_every)'''),
    ('''    ap.add_argument("--vblend", action="store_true",''',
     '''    ap.add_argument("--rest-every", type=int, default=0,
                    help="grey scan: stop for a rest after this many positions "
                         "and wait for the trigger (0 = no breaks)")
    ap.add_argument("--vblend", action="store_true",'''),
    ('''    if args.target.startswith("pix"):
        import targets as _t
        _g = _t.load_target(args.target)''',
     '''    if os.path.isfile(args.target):
        # a drawing that is already at scan resolution: one position per pixel
        _w, _h = Image.open(args.target).size
        if _w <= 48 and _h <= 48:
            args.grid_w, args.grid_h = _w, _h
    if args.target.startswith("pix"):
        import targets as _t
        _g = _t.load_target(args.target)'''),
])
print("patched")
