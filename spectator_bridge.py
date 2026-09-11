"""Spectator bridge for EyeCamXR ("shared XR").

Receives fire-and-forget UDP JSON from the Quest app (port 8090), keeps the
latest session state, and serves it to any number of browser spectators over
WebSocket (port 8080) alongside the static spectator page. Late joiners get a
full state replay. Also writes the phantom-schema CSVs (calib_log.csv /
cursor_log.csv) from incoming cursor messages, so phantom_subject.py can run
against the REAL on-device app with no headband:

    python spectator_bridge.py --session runs/quest_phantom
    python phantom_subject.py --session runs/quest_phantom \
        --host <QUEST_IP> --port 5005

Spectators (PC browser or a second headset's browser): http://<PC_IP>:8080
"""

import argparse
import asyncio
import json
import os
import socket
import time

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

UDP_PORT = 8090
WS_PORT = 8080

STATE = {
    "hello": None,
    "stage": None,
    "calib": None,
    "grid": None,
    "eeg": None,
    "rows": {},      # y -> row msg
    "log": [],
}
CLIENTS = set()
LOOP = None


class BridgeProtocol(asyncio.DatagramProtocol):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.calib_f = None
        self.cursor_f = None
        self.session_id = None

    def csv_for(self, msg):
        """Mirror cursor messages into phantom-schema CSVs."""
        if not self.csv_dir:
            return None
        stage = (STATE.get("stage") or {}).get("stage", "")
        name = "calib_log.csv" if stage == "calibrate" else "cursor_log.csv"
        attr = "calib_f" if name == "calib_log.csv" else "cursor_f"
        f = getattr(self, attr)
        if f is None:
            f = open(os.path.join(self.csv_dir, name), "w", buffering=1)
            f.write("time,grid_x,grid_y,luminance,flicker_on\n")
            setattr(self, attr, f)
        return f

    def datagram_received(self, data, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        mtype = msg.get("type")
        if msg.get("sid") and msg["sid"] != self.session_id:
            # new session: reset state
            self.session_id = msg["sid"]
            STATE["rows"].clear()
            STATE["log"].clear()
            for k in ("hello", "stage", "calib", "grid", "eeg"):
                STATE[k] = None
            for attr in ("calib_f", "cursor_f"):
                f = getattr(self, attr)
                if f:
                    f.close()
                    setattr(self, attr, None)

        if mtype in ("hello", "stage", "calib", "grid", "eeg"):
            STATE[mtype] = msg
        elif mtype == "row":
            STATE["rows"][msg["y"]] = msg
        elif mtype == "log":
            STATE["log"].append(msg)
            STATE["log"][:] = STATE["log"][-50:]
        elif mtype == "cursor":
            f = self.csv_for(msg)
            if f:
                f.write(f"{msg['t']:.6f},{msg['gx']},{msg['gy']},"
                        f"{msg['lum']:.4f},{msg['fl']}\n")

        payload = json.dumps(msg)
        for ws in list(CLIENTS):
            asyncio.ensure_future(send_safe(ws, payload))


async def send_safe(ws, payload):
    try:
        await ws.send(payload)
    except Exception:
        CLIENTS.discard(ws)


async def ws_handler(ws):
    CLIENTS.add(ws)
    try:
        # full replay for late joiners
        for key in ("hello", "stage", "calib", "eeg", "grid"):
            if STATE[key]:
                await ws.send(json.dumps(STATE[key]))
        for row in STATE["rows"].values():
            await ws.send(json.dumps(row))
        for entry in STATE["log"][-10:]:
            await ws.send(json.dumps(entry))
        async for _ in ws:
            pass  # spectators don't talk back
    finally:
        CLIENTS.discard(ws)


async def http_page(path, request_headers):
    """Serve spectator.html on plain GET (websockets' process_request hook)."""
    target = getattr(path, "path", path)  # websockets>=12 passes a Request
    if isinstance(target, str) and not target.startswith("/ws"):
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "spectator.html")
        with open(page, "rb") as f:
            body = f.read()
        return (200, [("Content-Type", "text/html; charset=utf-8"),
                      ("Content-Length", str(len(body)))], body)
    return None


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="",
                    help="dir for phantom-schema CSV mirroring (optional)")
    ap.add_argument("--udp-port", type=int, default=UDP_PORT)
    ap.add_argument("--ws-port", type=int, default=WS_PORT)
    args = ap.parse_args()
    if args.session:
        os.makedirs(args.session, exist_ok=True)

    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        lambda: BridgeProtocol(args.session or None),
        local_addr=("0.0.0.0", args.udp_port))

    async with websockets.serve(ws_handler, "0.0.0.0", args.ws_port,
                                process_request=http_page):
        ips = ", ".join(local_ips()) or "?"
        print(f"EyeCamXR spectator bridge")
        print(f"  Quest sends UDP JSON to  {ips}:{args.udp_port}")
        print(f"  spectators open          http://{ips.split(', ')[0]}:{args.ws_port}")
        if args.session:
            print(f"  phantom CSVs mirrored to {args.session}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
