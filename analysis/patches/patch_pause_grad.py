"""2026-10-01: pause/resume scans on EEG silence, skip starved visits,
grey-level text target. Run once from the repo root."""
import io


def patch(p, pairs):
    x = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert x.count(a) == 1, (p, a[:70], x.count(a))
        x = x.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="\n").write(x)


# ---------- page: pause / resume with the interrupted position redone ----------
patch("xr_stimulus.html", [
    ('''function black() { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height); }
''',
     '''function black() { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height); }

// The driver pauses the scan when the EEG stream goes silent. The position
// in progress is abandoned and done again from the start after the resume.
let paused = false;
async function holdWhilePaused() {
  if (!paused) return false;
  black();
  send({ type: "frame", stage: "paused", pt: now(), gx: -1, gy: -1, lum: 0, fl: 0 });
  const keep = sub.textContent;
  while (paused) {
    sub.textContent = "PAUSED - no EEG data from the Muse. Check MuseLog on the phone.";
    await sleep(200);
  }
  sub.textContent = "resuming...";
  await sleep(2000);
  sub.textContent = keep;
  return true;
}
'''),
    ('''    for (const gx of xs) {
      const lum = target[gy][gx];
      // fixed patch size''',
     '''    for (let xi = 0; xi < xs.length; xi++) {
      const gx = xs[xi];
      if (await holdWhilePaused()) { xi--; continue; }
      const lum = target[gy][gx];
      // fixed patch size'''),
    ('''      while (now() - t0 < m.spc && running === tok) {
        await frameTs(); frameCount++;
        const t = now();
        const fl = lum > 0.05 && flickerOn();''',
     '''      while (now() - t0 < m.spc && running === tok && !paused) {
        await frameTs(); frameCount++;
        const t = now();
        const fl = lum > 0.05 && flickerOn();'''),
    ('''          const shade = Math.round(255 * (0.3 + 0.7 * lum));
          ctx.fillStyle = bwc ?''',
     '''          // white/black: the flicker amplitude IS the grey level (a half-
          // grey position flickers at half the drive of a white one)
          const shade = Math.round(255 * (bwc ? lum : 0.3 + 0.7 * lum));
          ctx.fillStyle = bwc ?'''),
    ('''               lum, fl: fl ? 1 : 0 });
      }
      if (running !== tok) break;''',
     '''               lum, fl: fl ? 1 : 0 });
      }
      if (paused) { xi--; continue; }
      if (running !== tok) break;'''),
    ('''    for (const gx of xs) {
      const [r, g, b] = target[gy][gx];
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok) {''',
     '''    for (let xi = 0; xi < xs.length; xi++) {
      const gx = xs[xi];
      if (await holdWhilePaused()) { xi--; continue; }
      const [r, g, b] = target[gy][gx];
      const t0 = now();
      flickerT0 = (await frameTs()) / 1000;
      while (now() - t0 < m.spc && running === tok && !paused) {'''),
    ('''               flG: sG ? 1 : 0, flB: sB ? 1 : 0, r, g, b });
      }
      if (running !== tok) break;''',
     '''               flG: sG ? 1 : 0, flB: sB ? 1 : 0, r, g, b });
      }
      if (paused) { xi--; continue; }
      if (running !== tok) break;'''),
    ('''      case "msg": sub.textContent = m.text; break;''',
     '''      case "msg": sub.textContent = m.text; break;
      case "pause": paused = true; break;
      case "resume": paused = false; break;'''),
])

# ---------- driver: pause during scans, stop elsewhere ----------
patch("xr_session.py", [
    ('''                    if tail.rate() > 1:
                        self.eeg_silent = 0
                    else:
                        self.eeg_silent = getattr(self, "eeg_silent", 0) + 1
                        if self.eeg_silent == 10:
                            print("EEG STREAM LOST (10 s without samples)")
                            self.eeg_lost = True''',
     '''                    # scans PAUSE on a silent stream (3 s) and resume when
                    # it is back; the interrupted position is redone. Block
                    # stages cannot be redone piecemeal: they stop at 10 s.
                    if tail.rate() > 100:
                        self.eeg_good = getattr(self, "eeg_good", 0) + 1
                        self.eeg_silent = 0
                        if self.paused and self.eeg_good >= 3:
                            self.paused = False
                            print("EEG stream back: resuming scan")
                            await self.send(cmd="resume")
                    elif tail.rate() < 1:
                        self.eeg_good = 0
                        self.eeg_silent = getattr(self, "eeg_silent", 0) + 1
                        if (self.scan_active and not self.paused
                                and self.eeg_silent >= 3):
                            self.paused = True
                            print("EEG STREAM SILENT: scan paused")
                            await self.send(cmd="pause")
                        elif not self.scan_active and self.eeg_silent == 10:
                            print("EEG STREAM LOST (10 s without samples)")
                            self.eeg_lost = True'''),
    ('''    async def wait_for(self, typ, timeout=600):
        while True:
            m = await asyncio.wait_for(self.queue.get(), timeout)''',
     '''    paused = False
    scan_active = False

    async def wait_for(self, typ, timeout=600):
        self.scan_active = (typ == "scan_done")
        self.eeg_lost = False
        while True:
            try:
                m = await asyncio.wait_for(self.queue.get(),
                                           5.0 if self.paused else timeout)
            except asyncio.TimeoutError:
                if self.paused:
                    continue  # a paused scan may wait as long as it takes
                raise'''),
    ('''            if m.get("type") == typ:
                return m
            if getattr(self, "eeg_lost", False) and typ != "go":''',
     '''            if m.get("type") == typ:
                self.scan_active = False
                return m
            if getattr(self, "eeg_lost", False) and typ != "go":'''),
])

# ---------- reconstruction: ignore pause markers and EEG-starved visits ----------
patch("reconstruct.py", [
    ('''    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(eeg_t, t0)
        i1 = np.searchsorted(eeg_t, t1)
''',
     '''    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(eeg_t, t0)
        i1 = np.searchsorted(eeg_t, t1)
        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
'''),
    ('''    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
''',
     '''    for s, e in zip(starts, ends):
        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
'''),
])

# ---------- targets: a text target with grey levels ----------
patch("targets.py", [
    ('''    if spec.startswith("text:"):
        g = text_target(spec[5:], grid_w, grid_h)''',
     '''    if spec.startswith("grad:"):
        # letters at descending grey levels (first white, last half grey):
        # 'grad:NO' = white N, half-grey O
        txt = spec[5:]
        g = (text_target(txt, grid_w, grid_h) > 0.5).astype(float)
        n = max(len(txt.replace(" ", "")), 1)
        edges = np.linspace(0, g.shape[1], n + 1).round().astype(int)
        for k in range(n):
            g[:, edges[k]:edges[k + 1]] *= 1.0 - 0.5 * k / max(n - 1, 1)
        return g
    if spec.startswith("text:"):
        g = text_target(spec[5:], grid_w, grid_h)'''),
])
print("patched")
