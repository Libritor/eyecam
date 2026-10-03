"""EyeCam server: serves the web app and bridges LSL EEG (and optional Tobii gaze)
to the browser over a WebSocket.

    python server.py                  # web app only (connect the Muse via Web Bluetooth in the page)
    python server.py --muse           # also launch `muselsl stream` and bridge its LSL EEG outlet
    python server.py --lsl            # bridge an LSL EEG stream someone else runs (BlueMuse, mind2motor, ...)
    python server.py --tobii          # also bridge a Tobii eye tracker (needs tobii_research) for Fig. 6 mode
    python server.py --lan            # listen on all interfaces so a phone can open /flicker.html

The browser talks to ws://<host>:<ws-port>. Messages (JSON):
    {"type":"info", "source":..., "fs":256, "channels":[...]}
    {"type":"eeg",  "t":[epoch_s,...], "x":[[ch0,ch1,...],...]}
    {"type":"gaze", "t":[epoch_s,...], "xy":[[x01,y01,valid],...]}     (display-normalized 0..1)
    {"type":"status", "msg":"..."}
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")


# --------------------------------------------------------------------------- HTTP
class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".json": "application/json", ".wasm": "application/wasm"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the console for bridge status
        pass


def serve_http(host: str, port: int) -> http.server.ThreadingHTTPServer:
    httpd = http.server.ThreadingHTTPServer((host, port), functools.partial(Handler, directory=WEB))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# --------------------------------------------------------------------------- Bridge
class Bridge:
    """Fan-out of EEG/gaze chunks to every connected WebSocket client."""

    def __init__(self):
        self.clients: set = set()
        self.info = {"type": "info", "source": "none", "fs": 0, "channels": []}
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, msg: dict):
        if self.loop is None or not self.clients:
            return
        data = json.dumps(msg, separators=(",", ":"))
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    async def _send(self, ws, data):
        try:
            await ws.send(data)
        except Exception:
            self.clients.discard(ws)

    def status(self, msg: str):
        print(f"[bridge] {msg}", flush=True)
        self.publish({"type": "status", "msg": msg})

    async def handler(self, ws, path=None):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps(self.info))
            async for _ in ws:  # the page never needs to send anything; just drain
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(ws)


def lsl_thread(bridge: Bridge, stop: threading.Event):
    try:
        import pylsl
    except ImportError:
        bridge.status("pylsl is not installed: pip install pylsl")
        return
    # LSL timestamps are on pylsl.local_clock(); the page wants epoch seconds.
    while not stop.is_set():
        bridge.status("waiting for an LSL EEG stream (type=EEG)...")
        streams = []
        while not streams and not stop.is_set():
            streams = pylsl.resolve_byprop("type", "EEG", timeout=2.0)
        if stop.is_set():
            return
        info = streams[0]
        inlet = pylsl.StreamInlet(info, max_buflen=60, max_chunklen=12,
                                  processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        names = []
        try:
            ch = inlet.info().desc().child("channels").child("channel")
            for _ in range(info.channel_count()):
                names.append(ch.child_value("label") or f"ch{len(names)}")
                ch = ch.next_sibling()
        except Exception:
            pass
        if len(names) != info.channel_count():
            names = ["TP9", "AF7", "AF8", "TP10", "AUX"][: info.channel_count()]
        bridge.info = {"type": "info", "source": f"LSL:{info.name()}", "fs": info.nominal_srate(),
                       "channels": names}
        bridge.publish(bridge.info)
        bridge.status(f"streaming {info.name()} ({info.channel_count()} ch @ {info.nominal_srate():g} Hz)")
        last = time.time()
        while not stop.is_set():
            chunk, ts = inlet.pull_chunk(timeout=0.05)
            if ts:
                off = time.time() - pylsl.local_clock()
                bridge.publish({"type": "eeg", "t": [t + off for t in ts],
                                "x": [[round(v, 3) for v in row] for row in chunk]})
                last = time.time()
            elif time.time() - last > 5:
                bridge.status("EEG stream went quiet; re-resolving")
                break


def muse_supervisor(bridge: Bridge, stop: threading.Event, address: str | None, model: str | None):
    """Keep `muselsl stream` alive; it exits after 3 s without data, so restart it."""
    while not stop.is_set():
        args = [sys.executable, "-m", "muselsl", "stream"]
        if address:
            args += ["--address", address]
        if model:
            args += ["--model", model]
        bridge.status("launching: " + " ".join(args[1:]))
        proc = subprocess.Popen(args)
        while proc.poll() is None and not stop.is_set():
            time.sleep(0.5)
        if proc.poll() is None:
            proc.terminate()
        if not stop.is_set():
            bridge.status("muselsl exited; restarting in 2 s")
            time.sleep(2)


def tobii_thread(bridge: Bridge):
    try:
        import tobii_research as tr
    except ImportError:
        bridge.status("--tobii given but tobii_research is not installed (pip install tobii-research)")
        return
    trackers = tr.find_all_eyetrackers()
    if not trackers:
        bridge.status("no Tobii eye tracker found")
        return
    et = trackers[0]
    bridge.status(f"Tobii {et.model} ({et.serial_number}) connected")
    buf, lock = [], threading.Lock()

    def cb(g):
        l, r = g["left_gaze_point_on_display_area"], g["right_gaze_point_on_display_area"]
        lv, rv = g["left_gaze_point_validity"], g["right_gaze_point_validity"]
        pts = [p for p, v in ((l, lv), (r, rv)) if v]
        if pts:
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
        else:
            x = y = float("nan")
        with lock:
            buf.append((time.time(), x, y, 1 if pts else 0))

    et.subscribe_to(tr.EYETRACKER_GAZE_DATA, cb, as_dictionary=True)
    while True:
        time.sleep(0.05)
        with lock:
            out, buf[:] = list(buf), []
        if out:
            bridge.publish({"type": "gaze", "t": [o[0] for o in out],
                            "xy": [[o[1] if o[1] == o[1] else None, o[2] if o[2] == o[2] else None, o[3]]
                                   for o in out]})


async def ws_main(bridge: Bridge, host: str, port: int):
    import websockets
    bridge.loop = asyncio.get_running_loop()
    async with websockets.serve(bridge.handler, host, port, max_size=None):
        await asyncio.Future()


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    ap.add_argument("--ws-port", type=int, default=8766, help="WebSocket bridge port (default 8766)")
    ap.add_argument("--muse", action="store_true", help="launch `muselsl stream` and bridge it")
    ap.add_argument("--address", help="Muse MAC address for --muse")
    ap.add_argument("--model", help="muselsl --model value (e.g. muse2, athena) for --muse")
    ap.add_argument("--lsl", action="store_true", help="bridge an existing LSL EEG stream")
    ap.add_argument("--tobii", action="store_true", help="bridge a Tobii eye tracker")
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces (phone flicker page)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    host = "0.0.0.0" if a.lan else "127.0.0.1"
    serve_http(host, a.port)
    url = f"http://localhost:{a.port}/"
    print(f"EyeCam running at {url}", flush=True)
    if a.lan:
        print(f"  phone / second device: http://{lan_ip()}:{a.port}/flicker.html", flush=True)

    bridge = Bridge()
    stop = threading.Event()
    want_bridge = a.muse or a.lsl or a.tobii
    if a.muse:
        threading.Thread(target=muse_supervisor, args=(bridge, stop, a.address, a.model), daemon=True).start()
    if a.muse or a.lsl:
        threading.Thread(target=lsl_thread, args=(bridge, stop), daemon=True).start()
    if a.tobii:
        threading.Thread(target=tobii_thread, args=(bridge,), daemon=True).start()

    if not a.no_browser:
        webbrowser.open(url)
    try:
        if want_bridge:
            try:
                import websockets  # noqa: F401
            except ImportError:
                print("websockets is not installed: pip install websockets  (bridge disabled)")
                raise
            print(f"  bridge: ws://localhost:{a.ws_port}  (choose 'Python bridge' in the page)", flush=True)
            asyncio.run(ws_main(bridge, host, a.ws_port))
        else:
            while True:
                time.sleep(3600)
    except (KeyboardInterrupt, ImportError):
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
