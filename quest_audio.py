"""Ear as a Microphone through the Quest: the clip is played by the headset's
browser (WebAudio) instead of the PC speakers, and timed on the PC clock.

ear_stimulus.html is served on --quest-port and pushed to the Quest browser over
USB (adb reverse, as quest_run.ps1 does for the visual runs). A trigger pull
unlocks audio; the PC then sends "play" for each repetition and the page answers
with the scheduled start on its own clock plus the output latency WebAudio
reports. The page clock is mapped onto the PC clock by ping-pong (the exchange
with the shortest round trip, a few ms over USB), re-done before every
repetition so drift cannot build up. events.csv gets the same t_on / t_off as
the speaker path, so analysis/ear_mic_decode.py is unchanged.

Caveat for the paper: the Quest's speakers sit a few centimetres from TP9 and
TP10. Part of what comes back may be the sound induced into the electrodes
rather than the brain's response. Control: the same session with the Muse on
the desk under the headset and nobody wearing it (--reps 6 is enough); it
should decode nothing.
"""
import asyncio
import json
import os
import queue
import subprocess
import threading
import time

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

try:
    from pylsl import local_clock          # the clock osc_acquire.py stamps with
except ImportError:
    local_clock = time.perf_counter

ROOT = os.path.dirname(os.path.abspath(__file__))


class QuestPlayer:
    def __init__(self, clip_path, port=8083, page=os.path.join(ROOT, "ear_stimulus.html")):
        self.clip_path, self.port, self.page = clip_path, port, page
        self.q = queue.Queue()
        self.ws = None
        self.loop = None
        self.ready = threading.Event()
        self.info = {}
        self.offset = None          # page clock (s) minus PC clock (s)
        self.rtt = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, "0.0.0.0", self.port, process_request=self._http, max_size=None):
            await asyncio.Future()

    def _http(self, connection, request):
        if request.path.startswith("/ws"):
            return None
        if request.path.startswith("/clip.wav"):
            with open(self.clip_path, "rb") as f:
                body = f.read()
            ctype = "audio/wav"
        else:
            with open(self.page, "rb") as f:
                body = f.read()
            ctype = "text/html; charset=utf-8"
        return Response(200, "OK", Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                                            ("Cache-Control", "no-store"), ("Connection", "close")]), body)

    async def _handler(self, ws):
        self.ws = ws
        try:
            async for msg in ws:
                m = json.loads(msg)
                m["_t_rx"] = local_clock()
                if m.get("type") == "ready":
                    self.info = m
                    self.ready.set()
                self.q.put(m)
        finally:
            if self.ws is ws:
                self.ws = None

    def send(self, obj):
        if self.ws is None:
            raise RuntimeError("the headset page is not connected")
        asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(obj)), self.loop).result(5)

    def wait(self, type_, timeout, rep=None):
        t_end = time.monotonic() + timeout
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no '{type_}' from the headset page in {timeout:.0f} s")
            try:
                m = self.q.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if m.get("type") == type_ and (rep is None or m.get("rep") == rep):
                return m

    def wait_ready(self, timeout, nudge_every=20.0):
        """Wait for the page's "ready"; every `nudge_every` s, if the page is open
        but its audio context stayed suspended, resume it through the browser's
        DevTools port (a DevTools evaluation counts as a user gesture)."""
        t_end = time.monotonic() + timeout
        while not self.ready.is_set():
            if self.ready.wait(min(nudge_every, max(0.0, t_end - time.monotonic()))):
                break
            if time.monotonic() >= t_end:
                break
            note = devtools_nudge(self.port)
            if note:
                print("  " + note)
        return self.ready.is_set()

    def sync_clock(self, n=8):
        """(round trip s, page clock minus PC clock s) from the shortest of n ping-pongs."""
        best = None
        for _ in range(n):
            t0 = local_clock()
            self.send(dict(cmd="ping", t_pc=t0))
            m = self.wait("pong", 5)
            t1 = m["_t_rx"]
            rtt = t1 - t0
            off = m["t_page"] / 1000 - (t0 + t1) / 2
            if best is None or rtt < best[0]:
                best = (rtt, off)
            time.sleep(0.03)
        self.rtt, self.offset = best
        return best

    def page_to_pc(self, t_page_ms):
        return t_page_ms / 1000 - self.offset

    def play(self, rep, timeout):
        """Play one repetition; (t_on, t_off) on the PC clock, at the headset's output."""
        self.sync_clock()
        self.send(dict(cmd="play", rep=rep))
        m = self.wait("play", 5, rep)
        lat = float(m.get("outputLatency") or 0.0)
        t_on = self.page_to_pc(m["t_page_start"]) + lat
        e = self.wait("ended", timeout, rep)
        t_off = self.page_to_pc(e["t_page"]) + lat
        return t_on, t_off, dict(output_latency_s=lat, rtt_s=self.rtt, offset_s=self.offset)

    def text(self, s):
        try:
            self.send(dict(cmd="text", text=s))
        except Exception:                                    # noqa: BLE001
            pass

    def done(self):
        try:
            self.send(dict(cmd="done"))
        except Exception:                                    # noqa: BLE001
            pass


def devtools_nudge(port, quest_serial=None, dev_port=9222):
    """Through the Quest browser's DevTools socket: keep one tab of our page,
    close the rest, bring it in front, and if its audio context is suspended
    (the trigger pull did not count as a user gesture) resume it with a
    user-gesture evaluation. Returns a one-line note, or "" when nothing to do."""
    import urllib.request
    base = ["adb"] + (["-s", quest_serial] if quest_serial else [])
    try:
        subprocess.run(base + ["forward", f"tcp:{dev_port}", "localabstract:chrome_devtools_remote"],
                       check=False, capture_output=True, timeout=10)
        tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{dev_port}/json", timeout=4).read().decode())
    except Exception:                                        # noqa: BLE001
        return ""
    ours = [t for t in tabs if t.get("type") == "page" and f":{port}/" in t.get("url", "")]
    if not ours:
        return "the page is not open in the headset browser"
    keep = ours[0]
    for t in ours[1:]:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{dev_port}/json/close/{t['id']}", timeout=4).read()
        except Exception:                                    # noqa: BLE001
            pass
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{dev_port}/json/activate/{keep['id']}", timeout=4).read()
    except Exception:                                        # noqa: BLE001
        pass
    probe = ("JSON.stringify({sub: document.getElementById('sub').textContent, "
             "ctx: (typeof ctx !== 'undefined' && ctx) ? ctx.state : 'none'})")

    async def go():
        async with websockets.connect(keep["webSocketDebuggerUrl"], max_size=None) as ws:
            async def ev(i, expr, gesture=False):
                await ws.send(json.dumps(dict(id=i, method="Runtime.evaluate",
                                              params=dict(expression=expr, returnByValue=True,
                                                          awaitPromise=True, userGesture=gesture))))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == i:
                        r = m.get("result", {}).get("result", {})
                        return r.get("value", r.get("description"))
            st = json.loads(await ev(1, probe))
            if st.get("ctx") == "suspended":
                await ev(2, "ctx.resume().then(() => ctx.state)", gesture=True)
                return f"headset page was '{st.get('sub')}' with audio suspended: resumed it through DevTools"
            return f"headset page: '{st.get('sub')}' (audio {st.get('ctx')})"

    try:
        return asyncio.run(go())
    except Exception as e:                                   # noqa: BLE001
        return f"devtools: {e}"


def adb_open(port):
    """Push the page to the Quest browser over USB; the Quest is named in case
    the MuseLog phone is on USB too. Returns the serial, or None without adb."""
    try:
        out = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        print(f"adb not found: open http://<this PC's IP>:{port}/ in the headset browser yourself")
        return None
    serial = next((line.split()[0] for line in out.splitlines() if "model:Quest" in line), None)
    base = ["adb"] + (["-s", serial] if serial else [])
    subprocess.run(base + ["reverse", f"tcp:{port}", f"tcp:{port}"], check=False, capture_output=True)
    subprocess.run(base + ["shell", "input", "keyevent", "KEYCODE_WAKEUP"], check=False, capture_output=True)
    time.sleep(1)
    subprocess.run(base + ["shell", f"am start -a android.intent.action.VIEW -d 'http://127.0.0.1:{port}/' "
                           f"-f 0x10000000 com.oculus.browser"], check=False, capture_output=True)
    return serial
