"""The 40 Hz-tagged music of the stimulus page, played on THIS computer's
default audio output (headphones) instead of the headset's speakers.

Same signal as xr_stimulus.html runMusic(): a triangle-wave melody whose
gain is 0.5 + 0.5*sin(2*pi*40*t) in tagged blocks and a flat 0.5 in plain
blocks (same mean level). The driver flips `tagged` on the page's block
messages, so block timing still comes from the one clock the scoring uses.

    python pc_audio.py          # 4 s plain, 4 s tagged: a level check
"""
import threading
import time

import numpy as np
import sounddevice as sd

FS = 48000
NOTES = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 293.66, 349.23,
         440.0, 349.23, 293.66, 246.94, 329.63, 392.0, 493.88, 587.33,
         493.88, 392.0, 349.23, 440.0, 523.25, 440.0, 349.23, 293.66]
NOTE_S = 0.32


def melody_loop():
    """One pass of the 24-note melody (7.68 s), triangle wave, per-note
    attack/decay as on the page (exponential 0.0001 -> 0.6 in 20 ms, back
    to 0.0001 by 95 % of the note)."""
    n = int(round(NOTE_S * FS))
    t = np.arange(n) / FS
    a_end, d_end = 0.02, NOTE_S * 0.95
    env = np.full(n, 1e-4)
    a = t < a_end
    env[a] = 1e-4 * (0.6 / 1e-4) ** (t[a] / a_end)
    d = (t >= a_end) & (t < d_end)
    env[d] = 0.6 * (1e-4 / 0.6) ** ((t[d] - a_end) / (d_end - a_end))
    out = []
    for f in NOTES:
        ph = (t * f) % 1.0
        tri = 4.0 * np.abs(ph - 0.5) - 1.0
        out.append(tri * env)
    return np.concatenate(out).astype(np.float32)


class PcAudio:
    def __init__(self, mod_hz=40.0, master=0.8):
        self.mod_hz = float(mod_hz)
        self.master = float(master)
        self.loop = melody_loop()
        self.pos = 0           # sample counter (melody position + AM phase)
        self.playing = False
        self.tagged = False
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=FS, channels=2, dtype="float32",
                                      callback=self._cb, blocksize=0,
                                      latency="low")
        self.device = sd.query_devices(self.stream.device, "output")["name"]
        self.stream.start()

    def _cb(self, out, frames, _time, _status):
        with self.lock:
            playing, tagged, pos = self.playing, self.tagged, self.pos
            self.pos += frames
        if not playing:
            out[:] = 0
            return
        idx = (pos + np.arange(frames)) % len(self.loop)
        gain = np.full(frames, 0.5, dtype=np.float32)
        if tagged:
            tt = (pos + np.arange(frames)) / FS
            gain += 0.5 * np.sin(2 * np.pi * self.mod_hz * tt).astype(np.float32)
        y = self.loop[idx] * gain * self.master
        out[:, 0] = y
        out[:, 1] = y

    def block(self, tagged):
        with self.lock:
            self.tagged = bool(tagged)
            self.playing = True

    def stop(self):
        with self.lock:
            self.playing = False

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.stream.stop()
        self.stream.close()


if __name__ == "__main__":
    a = PcAudio()
    print("output device:", a.device)
    a.block(False)
    print("plain 4 s")
    time.sleep(4)
    a.block(True)
    print("tagged (40 Hz flutter) 4 s")
    time.sleep(4)
    a.close()
