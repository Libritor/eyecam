"""Ear as a Microphone, session runner.

Plays one clip of a song N times through the PC's speakers while MuseLog streams
the Muse (TP9, AF7, AF8, TP10 + AUX on the top of the head) to this PC over OSC,
and logs the PC-clock time each repetition started. analysis/ear_mic_decode.py
then rebuilds the clip's loudness envelope from the EEG and writes it back out as
sound.

Phone (MuseLog): OSC format SnowballArcade, full-rate ON, Target IP = this PC's
Wi-Fi address (printed at start), port 5000, Start Streaming.

Sound: use the PC SPEAKERS, not earbuds. Earbuds sit next to TP9 / TP10 and their
drivers induce the audio straight into the electrodes, which would "work" for the
wrong reason. Speakers across the desk cannot do that.

The design is repetition: the same 30 s clip 24 times (about 14 min). Each
repetition is decoded from EEG the decoder never trained on, and the average of
those held-out reconstructions is the "brain envelope" of the clip.

    python ear_mic_session.py --song "C:\\path\\song.mp3" --start 40 --clip 30 --reps 24
    python ear_mic_session.py --song song.mp3 --session runs/ear_mic1 --auto

Writes <session>/eeg.csv (osc_acquire.py), events.csv (rep, t_on, t_off, gap_s),
stimulus.json and clip.wav (the exact samples played).
"""

import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import sys
import time
from math import gcd

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

try:
    from pylsl import local_clock          # the clock osc_acquire.py stamps with
except ImportError:
    local_clock = time.perf_counter

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NAMES5 = ["TP9", "AF7", "AF8", "TP10", "AUX"]


def hold_awake():
    """The laptop sleeps after 3 idle minutes on battery (memory note); hold it."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


def read_audio(path):
    """(mono float32, fs). soundfile for wav / flac / mp3; anything else (m4a, aac,
    opus, webm) through the ffmpeg that imageio-ffmpeg bundles."""
    try:
        data, fs = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1), int(fs)
    except Exception as e:                                   # noqa: BLE001
        try:
            import imageio_ffmpeg
        except ImportError:
            raise SystemExit(f"cannot read {path} ({e}); pip install imageio-ffmpeg for m4a/aac/opus") from None
        fs = 48000
        out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-f", "f32le",
                              "-ac", "1", "-ar", str(fs), "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy(), fs


def load_clip(path, start_s, clip_s, fs_out):
    x, fs = read_audio(path)
    a, b = int(start_s * fs), int((start_s + clip_s) * fs)
    if b > len(x):
        raise SystemExit(f"{path} is {len(x) / fs:.1f} s long; --start {start_s} + --clip {clip_s} runs past the end")
    x = x[a:b].copy()
    if fs != fs_out:
        g = gcd(int(fs), int(fs_out))
        x = resample_poly(x, fs_out // g, fs // g).astype(np.float32)
    n = int(0.02 * fs_out)                                   # 20 ms fades, no clicks
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    x *= 0.9 / (np.abs(x).max() + 1e-9)
    return x


class Recorder:
    """osc_acquire.py in a child process, stopped through a stop file (as run_session)."""

    def __init__(self, session, port):
        self.stop_file = os.path.join(session, "stop_recorder")
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        self.proc = subprocess.Popen([PY, os.path.join(ROOT, "osc_acquire.py"), "--session", session,
                                      "--port", str(port), "--stop-file", self.stop_file])

    def stop(self):
        open(self.stop_file, "w").close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


def tail_eeg(path, seconds=5.0):
    """(rate Hz over the last `seconds`, channel names, robust sd per channel in the last 2 s)."""
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return 0.0, [], []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 600_000))
        chunk = f.read().decode("utf-8", "replace").splitlines()
    with open(path, "r") as f:
        names = f.readline().strip().split(",")[1:]
    rows = []
    for line in chunk[1:]:
        parts = line.split(",")
        if len(parts) == len(names) + 1:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
    if len(rows) < 10:
        return 0.0, names, []
    arr = np.array(rows)
    now = local_clock()
    recent = arr[arr[:, 0] > now - seconds]
    rate = len(recent) / seconds
    last2 = arr[arr[:, 0] > now - 2.0]
    sds = []
    if len(last2) > 20:
        x = last2[:, 1:] - np.median(last2[:, 1:], axis=0)
        sds = (1.4826 * np.median(np.abs(x), axis=0)).tolist()
    return rate, names, sds


def signal_check(eeg_path, ip_text, auto, timeout_s=180):
    print(f"\nwaiting for MuseLog at {ip_text} (UDP 5000) ...")
    stable_since = None
    t0 = time.monotonic()
    while True:
        rate, names, sds = tail_eeg(eeg_path)
        ok = rate >= 200 and len(names) >= 5
        if ok:
            stable_since = stable_since or time.monotonic()
        else:
            stable_since = None
        q = "  ".join(f"{n} {s:5.1f}" for n, s in zip(names, sds)) if sds else "no rows yet"
        flag = "" if ok else ("  <- need 5 channels (AUX plugged in, aux preset) and full rate" if names and (len(names) < 5 or rate < 200) else "")
        print(f"  {rate:6.1f} Hz  {q}{flag}")
        if stable_since and time.monotonic() - stable_since >= 10:
            print("stream stable for 10 s: " + ", ".join(names))
            if sds:
                bad = [n for n, s in zip(names, sds) if s > 150]
                if bad:
                    print(f"  WARNING: very noisy (>150 uV robust sd): {', '.join(bad)}. Reseat those electrodes.")
            break
        if auto and time.monotonic() - t0 > timeout_s:
            raise SystemExit("no stable 5-channel stream; check the phone's Target IP, port 5000, full rate, Start Streaming")
        time.sleep(1.0)
    if not auto:
        input("\nSit still, eyes open, looking at one spot. Press Enter to start the clips ... ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="wav / flac / mp3 file (m4a / aac / opus through imageio-ffmpeg)")
    ap.add_argument("--start", type=float, default=30.0, help="seconds into the song the clip begins")
    ap.add_argument("--clip", type=float, default=30.0, help="clip length in seconds")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--gap", type=float, default=2.5, help="silence between repetitions (s)")
    ap.add_argument("--jitter", type=float, default=1.0, help="random extra gap, uniform 0..jitter (s)")
    ap.add_argument("--baseline", type=float, default=10.0, help="silence before the first and after the last clip (s)")
    ap.add_argument("--session", default=os.path.join("runs", "ear_mic1"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default=None, help="sounddevice output index or name (default: system default)")
    ap.add_argument("--fs", type=int, default=48000, help="playback sample rate")
    ap.add_argument("--auto", action="store_true", help="no Enter prompt; abort if no stream in 3 min")
    ap.add_argument("--no-recorder", action="store_true", help="play only (recorder already running elsewhere)")
    args = ap.parse_args()

    from osc_acquire import local_ips
    ips = [ip for ip in local_ips() if ip.startswith(("192.", "172.", "10."))] or local_ips()
    ip_text = ", ".join(ips) or "this PC's Wi-Fi IP"

    os.makedirs(args.session, exist_ok=True)
    clip = load_clip(args.song, args.start, args.clip, args.fs)
    sf.write(os.path.join(args.session, "clip.wav"), clip, args.fs, subtype="PCM_16")
    if args.device is not None:
        sd.default.device = (None, int(args.device) if args.device.isdigit() else args.device)
    dev = sd.query_devices(kind="output")
    print(f"output device: {dev['name']}  (reported latency {dev['default_low_output_latency'] * 1000:.0f} ms, constant; the decoder absorbs it)")
    print(f"clip: {args.clip:.0f} s of {os.path.basename(args.song)} from {args.start:.0f} s, {args.reps} repetitions, "
          f"about {(args.reps * (args.clip + args.gap + args.jitter / 2) + 2 * args.baseline) / 60:.1f} min")

    hold_awake()
    rec = None if args.no_recorder else Recorder(args.session, args.port)
    try:
        signal_check(os.path.join(args.session, "eeg.csv"), ip_text, args.auto)
        meta = dict(song=os.path.abspath(args.song), start_s=args.start, clip_s=args.clip, fs=args.fs,
                    reps=args.reps, gap_s=args.gap, jitter_s=args.jitter, baseline_s=args.baseline,
                    device=dev["name"], output_latency_s=float(dev["default_low_output_latency"]),
                    clock="pylsl.local_clock" if local_clock is not time.perf_counter else "time.perf_counter",
                    started=datetime.datetime.now().isoformat(timespec="seconds"))
        json.dump(meta, open(os.path.join(args.session, "stimulus.json"), "w"), indent=1)

        ev_path = os.path.join(args.session, "events.csv")
        with open(ev_path, "w", newline="", buffering=1) as f:
            w = csv.writer(f)
            w.writerow(["rep", "t_on", "t_off", "gap_s"])
            print(f"\nbaseline {args.baseline:.0f} s of silence")
            time.sleep(args.baseline)
            for r in range(1, args.reps + 1):
                gap = args.gap + random.uniform(0, args.jitter)
                t_on = local_clock()
                sd.play(clip, args.fs, blocking=False)
                sd.wait()
                t_off = local_clock()
                w.writerow([r, f"{t_on:.6f}", f"{t_off:.6f}", f"{gap:.3f}"])
                print(f"  rep {r:2d}/{args.reps}  played {t_off - t_on:5.2f} s  next in {gap:.1f} s")
                time.sleep(gap)
            print(f"baseline {args.baseline:.0f} s of silence")
            time.sleep(args.baseline)
    except KeyboardInterrupt:
        print("\nstopped early; what was recorded is kept")
    finally:
        sd.stop()
        if rec:
            rec.stop()
    print(f"\ndone. Decode with:\n  python analysis/ear_mic_decode.py {args.session}")


if __name__ == "__main__":
    main()
