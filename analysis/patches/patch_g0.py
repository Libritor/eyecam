"""Patch: G0 stimulus-delivery gate (page hidden/throttled => session void)."""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

x = open("xr_session.py").read()
a = '''            await self.wait_for(
                "calib_done",
                timeout=a.calib_blocks * (a.calib_on + a.calib_off + 2) + 60)
            self.close_log()
            await asyncio.sleep(0.7)'''
a2 = '''            await self.wait_for(
                "calib_done",
                timeout=a.calib_blocks * (a.calib_on + a.calib_off + 2) + 60)
            self.close_log()
            await asyncio.sleep(0.7)
            # G0: was the stimulus actually delivered? A background/hidden
            # browser tab throttles rAF to ~1 fps: no flicker edges, a few
            # rows per block. Such a session is VOID, not a null result.
            g0 = self.delivery_check("calib_log.csv",
                                     a.calib_blocks * (a.calib_on + a.calib_off))
            result["g0"] = g0
            if not g0["ok"]:
                msg = ("STIMULUS NOT DELIVERED: " + g0["reason"] +
                       " — keep the page visible & in the foreground")
                print(msg)
                await self.send(cmd="msg", text=msg)
                result.update(ok=False, error="G0 " + g0["reason"])
                await asyncio.sleep(a.linger)
                return 2'''
assert a in x
x = x.replace(a, a2)

b = '''    def measured_flicker(self):'''
b2 = '''    def delivery_check(self, log_name, duration_s):
        """Rows per second and flicker edges in a stimulus log."""
        path = os.path.join(self.session, log_name)
        rows, edges, last = 0, 0, 0
        try:
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\\n").split(",")
                    if len(parts) < 5:
                        continue
                    rows += 1
                    fl = int(parts[4])
                    if fl == 1 and last == 0:
                        edges += 1
                    last = fl
        except OSError:
            pass
        fps = rows / max(duration_s, 1e-9)
        if fps < 20:
            reason = f"page ran at {fps:.1f} fps (hidden/background tab?)"
        elif edges < 10:
            reason = f"only {edges} flicker edges logged"
        else:
            reason = ""
        return dict(ok=not reason, rows=rows, fps=round(fps, 1),
                    edges=edges, reason=reason)

    def measured_flicker(self):'''
assert b in x
x = x.replace(b, b2)
open("xr_session.py", "w").write(x)

h = open("xr_stimulus.html").read()
c = '''black(); connect();'''
c2 = '''black(); connect();
// a hidden tab is throttled to ~1 fps: tell the driver so a void session
// is never mistaken for a negative result
document.addEventListener("visibilitychange", () =>
  send({ type: "visibility", state: document.visibilityState }));'''
assert c in h
h = h.replace(c, c2)
open("xr_stimulus.html", "w").write(h)
print("patched G0")
