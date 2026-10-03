"""Keep a Muse connected to THIS PC over Bluetooth, published on LSL by muselsl.

`muselsl stream` exits as soon as the headset goes 3 s without data (a brief
contact glitch is enough), so it is relaunched until stopped. It is started with
--lsltime so its sample timestamps are on pylsl's local_clock(), the clock the
stimulus page and the recorders use.

The aux input (the Oz electrode on the charging port) only streams in an aux
preset: p20 on a classic Muse (2016 / Muse 2 / Muse S). muselsl's own default
is p21, aux OFF. Athena uses a different preset family (the Interaxon SDK's
1022 is not in muselsl's list), so no preset is forced there.

Used by xr_session.py --muse and lsl_to_osc.py --start-muselsl.
"""

import subprocess
import sys
import threading
import time


class MuseStream:
    def __init__(self, address=None, model="auto", log=print, preset=None):
        self.cmd = [sys.executable, "-m", "muselsl", "stream", "--lsltime", "--model", model]
        if address:
            self.cmd += ["--address", address]
        if preset:          # e.g. 20 (or "p20"): classic Muse with the aux (Oz) input ON
            # muselsl's command line takes the preset as a plain number
            self.cmd += ["--preset", str(preset).lower().lstrip("p")]
        self.log = log
        self.stop_event = threading.Event()
        self.proc = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        import tempfile
        while not self.stop_event.is_set():
            self.log("muse: connecting over Bluetooth (headset on; Muse phone app closed)...")
            # keep muselsl's own messages: when it fails, say WHY instead of looping blind
            out = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
            t0 = time.monotonic()
            self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT)
            while self.proc.poll() is None and not self.stop_event.is_set():
                time.sleep(0.5)
            if self.stop_event.is_set():
                out.close()
                break
            out.seek(0)
            lines = [l.strip() for l in out.read().splitlines()
                     if l.strip() and "netif" not in l and "ifindex" not in l
                     and "INFO|" not in l]
            out.close()
            why = " | ".join(lines[-3:]) if lines else "no message"
            if time.monotonic() - t0 < 15:
                self.log(f"muse: muselsl stopped after {time.monotonic() - t0:.0f} s: {why}")
            else:
                self.log("muse: connection dropped")
            self.log("muse: reconnecting in 2 s")
            self.stop_event.wait(2)

    def stop(self):
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
