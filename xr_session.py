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

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def csv_header(path):
    """Column names of a CSV (minus 'time'), or None if not written yet."""
    try:
        with open(path) as f:
            line = f.readline().strip()
        return line.split(",")[1:] if line.startswith("time,") else None
    except OSError:
        return None


class MultiRecorder:
    """Several OSC recorders (one per headband/port), merged into ONE eeg.csv
    on the PC arrival clock: device 0 is the time base; every other device is
    resampled onto it by nearest arrival timestamp (both streams are ~256 Hz
    on the same PC clock, so this is a <2 ms alignment). Columns: device-0
    names, then B1_<name>, B2_<name>, ... — the calibration weights then pick
    the best channels across headbands automatically."""

    def __init__(self, session, ports):
        from tailer import CsvTail
        self.session = session
        self.ports = ports
        self.recs = []
        self.tails = []
        for i, port in enumerate(ports):
            sub = os.path.join(session, f"dev{i}")
            os.makedirs(sub, exist_ok=True)
            self.recs.append(Recorder("osc", sub, port))
            self.tails.append(CsvTail(os.path.join(sub, "eeg.csv")))
        self.names = [None] * len(ports)
        self.buf = [[] for _ in ports]  # per device: list of (t, vals)
        self.out = None
        self.merged = 0

    def start(self):
        for r in self.recs:
            r.start()

    def alive(self):
        return all(r.alive() for r in self.recs)

    def stop(self):
        for r in self.recs:
            r.stop()
        if self.out:
            self.out.close()

    def stderr_tail(self):
        return " | ".join(f"udp:{p}: {r.stderr_tail()[-120:]}"
                          for p, r in zip(self.ports, self.recs))

    def poll_merge(self):
        import bisect
        for i, tail in enumerate(self.tails):
            if self.names[i] is None:
                self.names[i] = csv_header(
                    os.path.join(self.session, f"dev{i}", "eeg.csv"))
            for r in tail.poll():
                try:
                    self.buf[i].append((float(r[0]), [float(v) for v in r[1:]]))
                except (ValueError, IndexError):
                    pass
        if any(n is None for n in self.names) or any(not b for b in self.buf):
            return
        if self.out is None:
            cols = list(self.names[0])
            for i in range(1, len(self.names)):
                cols += [f"B{i}_{n}" for n in self.names[i]]
            self.out = open(os.path.join(self.session, "eeg.csv"), "w",
                            buffering=1)
            self.out.write("time," + ",".join(cols) + "\n")
            print(f"merged EEG columns: {cols}")
        # merge device-0 rows older than 0.35 s (lets the others catch up)
        cutoff = local_clock() - 0.35
        others = [(np.array([t for t, _ in b]), b) for b in self.buf[1:]]
        n_done = 0
        for t0, vals in self.buf[0]:
            if t0 > cutoff:
                break
            row = list(vals)
            for times, b in others:
                j = bisect.bisect_left(times, t0)
                cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
                k = min(cands, key=lambda k: abs(times[k] - t0)) if cands else None
                if k is not None and abs(times[k] - t0) < 0.06:
                    row += b[k][1]
                else:  # no sample from that device near t0 (dropout)
                    row += [0.0] * len(b[0][1])
            self.out.write(f"{t0:.6f}," + ",".join(f"{v:.4f}" for v in row) + "\n")
            n_done += 1
        del self.buf[0][:n_done]
        self.merged += n_done
        keep = local_clock() - 5.0  # trim other-device buffers
        for i in range(1, len(self.buf)):
            self.buf[i] = [(t, v) for t, v in self.buf[i] if t >= keep]


def channel_quality(tail):
    """Per-channel contact verdict from the last 2 s: 1=good, 0=bad.
    Catches the observed live failure mode (floating ear electrode: huge
    std, samples at the ADC rails) while accepting both raw-Muse microvolt
    scales and the phantom's unit-scale synthetic data."""
    cutoff = local_clock() - 2.0
    win = [r for r in tail.rows if r[0] >= cutoff]
    if len(win) < 32:
        return [0, 0, 0, 0]
    width = min(len(r[1]) for r in win)
    arr = np.array([r[1][:width] for r in win])
    out = []
    for c in range(arr.shape[1]):
        x = arr[:, c]
        std = float(np.std(x))
        med = float(np.median(x))
        if float(np.max(np.abs(x))) > 100:  # raw Muse scale (~0..1682 uV)
            rail = float(np.mean((x < 5) | (x > 1670)))
        else:
            rail = float(np.mean(np.abs(x - med) > 500))
        out.append(1 if (0.5 < std < 350 and rail < 0.10) else 0)
    while len(out) < 4:
        out.append(0)
    return out


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
        self.stim_freq = args.freq  # replaced by the page's delivered value

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
        if msg.get("cmd") in ("start_calib", "start_scan", "start_assr",
                              "signal", "calib_result", "result",
                              "assr_result"):
            self.last_cmd = msg  # re-issued to a page that reconnects
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
        elif typ == "freq":
            # the page reports the frame-exact frequency it actually delivers
            self.stim_freq = float(m["hz"])
            print(f"page flicker: {self.stim_freq:.2f} Hz "
                  f"(refresh {m.get('refresh')} fps, "
                  f"{m.get('halfFrames')} frames on/off)")
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

        merge_task = None
        ports = [int(p) for p in str(a.osc_port).split(",") if p.strip()]
        if len(ports) == 1:
            recorder = Recorder("osc", self.session, ports[0])
        else:
            recorder = MultiRecorder(self.session, ports)
        recorder.start()
        await asyncio.sleep(1.5)
        if not recorder.alive():
            err = recorder.stderr_tail()
            if "10048" in err:  # port held by an orphaned recorder
                print("udp port busy — retrying once in 3 s")
                await asyncio.sleep(3)
                recorder.start()
                await asyncio.sleep(1.5)
            if not recorder.alive():
                raise RuntimeError(
                    f"EEG recorder failed on udp:{a.osc_port}: "
                    + recorder.stderr_tail()[-200:])
        merged_path = os.path.join(self.session, "eeg.csv")
        tail = EEGTail(merged_path)
        osc_target = ", ".join(f"{local_ip()}:{p}" for p in ports)
        merge_task = None
        if len(ports) > 1:
            async def merge_loop():
                while True:
                    recorder.poll_merge()
                    await asyncio.sleep(0.25)
            merge_task = asyncio.create_task(merge_loop())

        def names():
            return csv_header(merged_path) or CH_NAMES

        def ears_ok(q):
            n = names()
            return any(q[i] for i in range(min(len(q), len(n)))
                       if "TP9" in n[i] or "TP10" in n[i])
        result = dict(mode="browser", gridW=a.grid_w, gridH=a.grid_h,
                      secondsPerCell=a.spc, calibBlocks=a.calib_blocks)
        hb_task = None
        try:
            print("waiting for a browser to open the stimulus page...")
            wait_deadline = time.monotonic() + a.signal_timeout
            while self.page is None:
                if time.monotonic() > wait_deadline:
                    raise TimeoutError("no stimulus page connected")
                await asyncio.sleep(0.2)
            self.spectate(type="hello", mode="browser", ch=4, fs=256.0,
                          gridW=a.grid_w, gridH=a.grid_h, freq=a.freq,
                          refresh=-1)

            # persistent Muse status badge: rate + per-electrode contact,
            # pushed to the page once a second through EVERY stage
            async def heartbeat():
                while True:
                    tail.poll()
                    await self.send(cmd="eeg", rate=tail.rate(),
                                    q=channel_quality(tail), names=names())
                    self.spectate(type="eeg", rate=tail.rate(), rms=[])
                    await asyncio.sleep(1.0)
            hb_task = asyncio.create_task(heartbeat())

            # signal check: rate stable AND at least one EAR electrode good
            # (TP9/TP10 carry the SSVEP; a session without them is doomed)
            self.spectate(type="stage", stage="signal_check", detail="")
            stable_since = None
            deadline = time.monotonic() + a.signal_timeout
            while True:
                tail.poll()
                rate = tail.rate()
                q = channel_quality(tail)
                ears = ears_ok(q)
                stable = rate > 40
                nowm = time.monotonic()
                if stable and ears and stable_since is None:
                    stable_since = nowm
                if not (stable and ears):
                    stable_since = None
                ready = stable_since and nowm - stable_since >= 3
                if not recorder.alive():
                    raise RuntimeError("EEG recorder died: "
                                       + recorder.stderr_tail()[-200:])
                await self.send(cmd="signal", rate=rate, stable=stable,
                                earsOk=ears, ready=bool(ready),
                                oscTarget=osc_target)
                if ready:
                    break
                if nowm > deadline:
                    raise TimeoutError(
                        f"no usable EEG on udp:{a.osc_port} "
                        f"(rate {rate:.0f} Hz, ears_ok={ears_ok}; "
                        f"send /muse/eeg to {osc_target})")
                await asyncio.sleep(0.25)
            print(f"signal ok ({rate:.0f} Hz, ears good)")

            if a.mode == "assr":
                return await self.run_assr(result)

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
            calib = run_session.score_calibration(
                self.session, self.blocks, stim_freq=self.stim_freq)
            result["stimFreqActual"] = self.stim_freq
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
                calibration=os.path.join(self.session, "calibration.json"),
                stim_freq=self.stim_freq)
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
            if hb_task:
                hb_task.cancel()
            if merge_task:
                merge_task.cancel()
                recorder.poll_merge()  # flush what is already buffered
            with open(os.path.join(self.session, "xr_session.json"),
                      "w") as f:
                json.dump(result, f, indent=1)
            recorder.stop()

    async def run_assr(self, result):
        """'Ear as a microphone' v1: 40 Hz amplitude-modulated tone in ON/OFF
        blocks; the auditory steady-state response (ASSR) at the modulation
        frequency is scored with the same machinery as visual calibration.
        Honest scope: this detects whether EEG tracks the sound envelope on
        a minutes timescale — one audio pixel, not real-time audio."""
        a = self.args
        self.spectate(type="stage", stage="assr", detail="")
        self.blocks = []
        await self.send(cmd="start_assr", blocks=a.calib_blocks,
                        onS=config.CALIB_ON_S, offS=config.CALIB_OFF_S,
                        modFreq=40.0)
        await self.wait_for("assr_done", timeout=a.calib_blocks * 16 + 120)
        await asyncio.sleep(0.7)
        assr = run_session.score_calibration(self.session, self.blocks,
                                             stim_freq=40.0)
        os.replace(os.path.join(self.session, "calibration.json"),
                   os.path.join(self.session, "assr.json"))
        table = "   ".join(
            f"{n} d'={assr['channels'][n]['dprime']:.1f}"
            for n in assr["names"])
        print(f"ASSR passed={assr['passed']} {table}")
        result.update(ok=True, assrPassed=assr["passed"],
                      assrBest=assr["best"], assrTable=table)
        await self.send(cmd="assr_result", passed=assr["passed"],
                        table=table)
        self.spectate(type="calib", detail="ASSR " + table,
                      passed=assr["passed"], best=assr["best"], weights=[])
        await asyncio.sleep(a.linger)
        return 0

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
    ap.add_argument("--osc-port", default="5000",
                    help="UDP port, or comma list for several headbands "
                         "(e.g. 5000,5001 -> merged 8-channel eeg.csv)")
    ap.add_argument("--preset", choices=list(config.PRESETS), default="")
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--spc", type=float, default=4.0)
    ap.add_argument("--calib-blocks", type=int, default=config.CALIB_BLOCKS)
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--mode", choices=["visual", "assr"], default="visual",
                    help="assr = 'ear as a microphone' 40 Hz tone blocks")
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
            # One subject page at a time. A second tab (or a spectator who
            # opened the stimulus URL) must never take over a running stage:
            # Quest Browser pauses background tabs, which stalls the flicker.
            if driver.page is not None:
                print("extra stimulus page connected — ignored "
                      "(close duplicate tabs)")
                try:
                    await ws.send(json.dumps(dict(
                        cmd="msg", text="another tab is running the session "
                        "— close this one")))
                    async for _ in ws:
                        pass
                finally:
                    return
            driver.page = ws
            print("stimulus page connected")
            try:
                # a reconnecting page resumes the current stage from its start
                last = getattr(driver, "last_cmd", None)
                if last is not None:
                    if last.get("cmd") in ("start_calib", "start_assr"):
                        driver.blocks = []  # stage restarts on the new page
                    await ws.send(json.dumps(last))
                async for raw in ws:
                    try:
                        driver.handle(json.loads(raw))
                    except (ValueError, KeyError):
                        pass
            finally:
                if driver.page is ws:
                    driver.page = None
                    print("stimulus page disconnected")

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
