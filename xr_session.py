"""Browser-driven eye-as-camera session (the Buzzkill architecture).

The stimulus renders in ANY browser — Quest Browser for VR, desktop Chrome
for testing — served by this driver. EEG arrives over OSC (Mind Monitor,
MuseLog, or phantom_subject.py). All DSP is the gated laptop stack:
run_session's Recorder/EEGTail/score_calibration and reconstruct.py, unchanged.

    python xr_session.py --session runs/xr1 --preset quick
    # headset/desktop: open  http://<PC_IP>:8082/        (?auto=1 = no input)
    # phone: Mind Monitor or MuseLog -> OSC /muse/eeg -> <PC_IP>:5000

Cursor/calib logs use the standard 5-column schema (+page_t), so
phantom_subject.py --session <same dir> closes a hardware-free loop.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import socket
import sys
import time

import numpy as np
from PIL import Image

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter

import config
import reconstruct
import run_session
import targets
from run_session import EEGTail, Recorder

try:
    import websockets
    from websockets.datastructures import Headers
    from websockets.http11 import Response
except ImportError:
    raise SystemExit("pip install websockets")

ROOT = os.path.dirname(os.path.abspath(__file__))


class Driver:
    def __init__(self, args):
        self.args = args
        self.session = os.path.join(ROOT, args.session) \
            if not os.path.isabs(args.session) else args.session
        os.makedirs(self.session, exist_ok=True)
        self.page = None            # the active subject page
        self.queue = asyncio.Queue()
        self.log_file = None
        self.blocks = []
        self.frames = 0
        self.spec = None
        if args.spectator:
            host, port = args.spectator.rsplit(":", 1)
            self.spec = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                         (host, int(port)))
        self.seq = 0
        self.sid = os.path.basename(self.session)

    # ---- plumbing ----

    def spectate(self, **msg):
        if not self.spec:
            return
        self.seq += 1
        msg.update(v=1, sid=self.sid, seq=self.seq, t=local_clock())
        try:
            self.spec[0].sendto(json.dumps(msg).encode(), self.spec[1])
        except OSError:
            pass

    async def send(self, **msg):
        if self.page is not None:
            try:
                await self.page.send(json.dumps(msg))
            except Exception:
                pass

    def open_log(self, name):
        self.log_file = open(os.path.join(self.session, name), "w",
                             buffering=1)
        self.log_file.write(
            "time,grid_x,grid_y,luminance,flicker_on,page_t\n")

    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def handle(self, m):
        """Fast path for page messages (called from the WS receive loop)."""
        t = local_clock()
        typ = m.get("type")
        if typ == "frame":
            if self.log_file:
                self.log_file.write(
                    f"{t:.6f},{m['gx']},{m['gy']},{m['lum']:.4f},"
                    f"{m['fl']},{m['pt']:.6f}\n")
                self.frames += 1
            if m["stage"] == "scan" and self.frames % 12 == 0:
                self.spectate(type="cursor", gx=m["gx"], gy=m["gy"],
                              lum=m["lum"], fl=m["fl"])
        elif typ == "block":
            if m["phase"] == "start":
                self._block_t0 = t
            else:
                self.blocks.append(("on" if m["on"] else "off",
                                    self._block_t0, t))
        else:
            self.queue.put_nowait(m)

    async def wait_for(self, typ, timeout=600):
        while True:
            m = await asyncio.wait_for(self.queue.get(), timeout)
            if m.get("type") == typ:
                return m

    # ---- session flow ----

    async def run(self):
        a = self.args
        target = targets.load_target(a.target, a.grid_w, a.grid_h)
        np.save(os.path.join(self.session, "target.npy"), target)

        recorder = Recorder("osc", self.session, a.osc_port)
        recorder.start()
        tail = EEGTail(os.path.join(self.session, "eeg.csv"))
        osc_target = f"{local_ip()}:{a.osc_port}"
        result = dict(mode="browser", gridW=a.grid_w, gridH=a.grid_h,
                      secondsPerCell=a.spc, calibBlocks=a.calib_blocks)
        try:
            print("waiting for a browser to open the stimulus page...")
            while self.page is None:
                await asyncio.sleep(0.2)
            self.spectate(type="hello", mode="browser", ch=4, fs=256.0,
                          gridW=a.grid_w, gridH=a.grid_h, freq=a.freq,
                          refresh=-1)

            # signal check
            self.spectate(type="stage", stage="signal_check", detail="")
            stable_since = None
            deadline = time.monotonic() + a.signal_timeout
            while True:
                tail.poll()
                rate = tail.rate()
                stable = rate > 40
                nowm = time.monotonic()
                if stable and stable_since is None:
                    stable_since = nowm
                if not stable:
                    stable_since = None
                ready = stable_since and nowm - stable_since >= 3
                await self.send(cmd="signal", rate=rate, stable=stable,
                                ready=bool(ready), oscTarget=osc_target)
                self.spectate(type="eeg", rate=rate, rms=[])
                if ready:
                    break
                if nowm > deadline:
                    raise TimeoutError(
                        f"no stable EEG stream on udp:{a.osc_port} "
                        f"(send /muse/eeg to {osc_target})")
                await asyncio.sleep(0.25)
            print(f"signal ok ({rate:.0f} Hz); calibration...")

            # calibration (page-driven)
            self.spectate(type="stage", stage="calibrate", detail="")
            self.blocks = []
            self.open_log("calib_log.csv")
            await self.send(cmd="start_calib", blocks=a.calib_blocks,
                            onS=config.CALIB_ON_S, offS=config.CALIB_OFF_S,
                            freq=a.freq)
            await self.wait_for("calib_done",
                                timeout=a.calib_blocks * 16 + 60)
            self.close_log()
            await asyncio.sleep(0.7)
            calib = run_session.score_calibration(self.session, self.blocks)
            table = "   ".join(
                f"{n} d'={calib['channels'][n]['dprime']:.1f}"
                f" w={calib['weights'][n]:.2f}"
                for n in calib["names"])
            print(f"calibration passed={calib['passed']} {table}")
            result.update(calibPassed=calib["passed"], best=calib["best"],
                          weights=calib["weights"])
            await self.send(cmd="calib_result", passed=calib["passed"],
                            table=table)
            self.spectate(type="calib", detail=table, passed=calib["passed"],
                          best=calib["best"],
                          weights=list(calib["weights"].values()))
            await asyncio.sleep(2.5)
            if not calib["passed"] and a.require_pass:
                raise RuntimeError("calibration gate failed (no SSVEP)")

            # scan (page-driven serpentine)
            self.spectate(type="stage", stage="scan", detail="")
            self.open_log("cursor_log.csv")
            await self.send(cmd="start_scan", gridW=a.grid_w, gridH=a.grid_h,
                            spc=a.spc, freq=a.freq, target=target.tolist())
            scan_s = a.grid_w * a.grid_h * a.spc
            await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
            self.close_log()
            await asyncio.sleep(0.7)

            # reconstruct with the unchanged laptop stack
            print("reconstructing...")
            grid, r = reconstruct.run(
                self.session,
                calibration=os.path.join(self.session, "calibration.json"))
            flicker_hz = self.measured_flicker()
            result.update(r=r, measuredFlickerHz=flicker_hz)
            png = grid_to_data_url(reconstruct_norm(grid))
            await self.send(cmd="result", png=png, r=r or 0.0,
                            flickerHz=flicker_hz)
            flat = [float(v) for v in np.asarray(grid).ravel()]
            self.spectate(type="grid", w=a.grid_w, h=a.grid_h, scores=flat)
            self.spectate(type="stage", stage="done",
                          detail=f"r={r:.3f}" if r else "")
            result["ok"] = True
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")
            await asyncio.sleep(a.linger)
            return 0
        except Exception as e:
            result.update(ok=False, error=repr(e))
            print(f"SESSION FAILED: {e!r}")
            await self.send(cmd="msg", text=f"session failed: {e}")
            return 1
        finally:
            with open(os.path.join(self.session, "xr_session.json"),
                      "w") as f:
                json.dump(result, f, indent=1)
            recorder.stop()

    def measured_flicker(self):
        """Median rising-edge period from the page's own clock (jitter-free)."""
        edges = []
        for name in ("cursor_log.csv", "calib_log.csv"):
            path = os.path.join(self.session, name)
            if not os.path.exists(path):
                continue
            last = 0
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) < 6:
                        continue
                    fl = int(parts[4])
                    if fl == 1 and last == 0:
                        edges.append(float(parts[5]))
                    last = fl
        periods = sorted(b - a for a, b in zip(edges, edges[1:])
                         if 0.01 < b - a < 0.5)
        if len(periods) < 5:
            return -1.0
        return 1.0 / periods[len(periods) // 2]


def reconstruct_norm(grid):
    g = np.asarray(grid)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - lo) / max(hi - lo, 1e-12), 0, 1)


def grid_to_data_url(norm, scale=24):
    img = Image.fromarray((norm * 255).astype(np.uint8), "L")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def http_page(connection, request):
    if request.path.startswith("/ws"):
        return None
    with open(os.path.join(ROOT, "xr_stimulus.html"), "rb") as f:
        body = f.read()
    return Response(200, "OK", Headers([
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Connection", "close")]), body)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join("runs", "xr1"))
    ap.add_argument("--http-port", type=int, default=8082)
    ap.add_argument("--osc-port", type=int, default=5000)
    ap.add_argument("--preset", choices=list(config.PRESETS), default="")
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--spc", type=float, default=4.0)
    ap.add_argument("--calib-blocks", type=int, default=config.CALIB_BLOCKS)
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--freq", type=float, default=config.STIM_FREQ_HZ)
    ap.add_argument("--spectator", default="127.0.0.1:8090",
                    help="'' disables the spectator mirror")
    ap.add_argument("--require-pass", action="store_true")
    ap.add_argument("--signal-timeout", type=float, default=600)
    ap.add_argument("--linger", type=float, default=10,
                    help="seconds to keep serving the result before exit")
    args = ap.parse_args()
    if args.preset:
        args.grid_w, args.grid_h, args.spc = config.PRESETS[args.preset]
    # rank can never exceed the ON-block count (same clamp as run_session)
    config.CALIB_RANK_MIN = min(config.CALIB_RANK_MIN, args.calib_blocks - 1)

    driver = Driver(args)

    async def ws_handler(ws):
        if not getattr(ws, "request", None) or \
                ws.request.path.startswith("/ws"):
            driver.page = ws
            print("stimulus page connected")
            try:
                async for raw in ws:
                    try:
                        driver.handle(json.loads(raw))
                    except (ValueError, KeyError):
                        pass
            finally:
                if driver.page is ws:
                    driver.page = None

    async with websockets.serve(ws_handler, "0.0.0.0", args.http_port,
                                process_request=http_page,
                                max_size=2 ** 22):
        print(f"stimulus page:  http://{local_ip()}:{args.http_port}/  "
              f"(?auto=1 for unattended)")
        print(f"EEG OSC in:     {local_ip()}:{args.osc_port}  "
              "(Mind Monitor / MuseLog / phantom)")
        code = await driver.run()
    sys.exit(code)


if __name__ == "__main__":
    asyncio.run(main())
