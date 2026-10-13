"""Phantom session for the Ear-as-a-Microphone decoder: the gate before a human run.

Builds runs/<name>/ with eeg.csv, events.csv, stimulus.json and clip.wav exactly as
ear_mic_session.py writes them. The EEG is pink noise with a DC offset, drift,
60 Hz mains and blinks on AF7/AF8, plus (unless --no-response) a cortical
envelope-following response: the clip's 1-8 Hz envelope convolved with a
three-peak kernel (50, 110, 200 ms), strongest on AUX (top of the head),
inverted on TP9/TP10, weak on AF7/AF8, delayed by the speaker latency. Arrival
times are stamped like Wi-Fi OSC: bundles of 12 samples, 20 ms latency, jitter
and occasional stalls, so the decoder's clock rebuild is exercised too.

    python analysis/ear_mic_phantom.py --song song.mp3 --snr-db -15
    python analysis/ear_mic_decode.py runs/ear_mic_phantom
    python analysis/ear_mic_phantom.py --song song.mp3 --no-response --session runs/ear_mic_phantom_null
    python analysis/ear_mic_decode.py runs/ear_mic_phantom_null      # must NOT beat chance
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ear_mic_session import load_clip, NAMES5  # noqa: E402
from ear_mic_decode import song_envelope, band_env, FS_D  # noqa: E402

FS = 256.0


def pink(n, rng):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def kernel():
    t = np.arange(0, 0.35, 1 / FS)
    g = lambda c, w, a: a * np.exp(-0.5 * ((t - c) / w) ** 2)   # noqa: E731
    return g(0.05, 0.012, -0.6) + g(0.11, 0.02, 1.0) + g(0.20, 0.03, -0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--start", type=float, default=40.0)
    ap.add_argument("--clip", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--snr-db", type=float, default=-15.0, help="response vs noise RMS in 1-8 Hz on AUX")
    ap.add_argument("--no-response", action="store_true")
    ap.add_argument("--session", default=os.path.join("runs", "ear_mic_phantom"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.session, exist_ok=True)

    fs_play = 48000
    clip = load_clip(args.song, args.start, args.clip, fs_play)
    sf.write(os.path.join(args.session, "clip.wav"), clip, fs_play, subtype="PCM_16")
    env = band_env(song_envelope(clip.astype(np.float64), fs_play))          # 64 Hz, z
    t64 = np.arange(len(env)) / FS_D

    # timeline
    baseline, gap, jit, latency = 10.0, 2.5, 1.0, 0.09
    t_on, tcur = [], baseline
    for _ in range(args.reps):
        t_on.append(tcur)
        tcur += args.clip + gap + rng.uniform(0, jit)
    total = tcur + baseline
    n = int(total * FS)
    t = np.arange(n) / FS

    # noise: pink, 1-8 Hz RMS 8 uV; mains; drift; DC like the real headband
    gains = dict(TP9=-0.6, AF7=0.3, AF8=0.3, TP10=-0.6, AUX=1.0)
    sos_band = butter(2, [1.0, 8.0], "bandpass", fs=FS, output="sos")
    X = np.zeros((n, 5))
    for c in range(5):
        pn = pink(n, rng)
        pn *= 8.0 / sosfiltfilt(sos_band, pn).std()
        drift = np.cumsum(rng.standard_normal(n)) * 0.02
        X[:, c] = 800.0 + 50 * rng.standard_normal() + pn + 5.0 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6)) + drift
    # blinks: 0.25 / s, +120 uV on AF7/AF8, -20 uV on the far channels
    for tb in np.sort(rng.uniform(0, total, int(0.25 * total))):
        pulse = np.exp(-0.5 * ((t - tb) / 0.065) ** 2)
        X[:, 1] += 120 * pulse
        X[:, 2] += 120 * pulse
        X[:, 0] -= 20 * pulse
        X[:, 3] -= 20 * pulse
        X[:, 4] -= 20 * pulse

    if not args.no_response:
        resp = np.zeros(n)
        env256 = np.interp(np.arange(int(args.clip * FS)) / FS, t64, env)
        for on in t_on:
            i0 = int(round((on + latency) * FS))
            resp[i0:i0 + len(env256)] += env256
        resp = np.convolve(resp, kernel())[:n]
        amp = 8.0 * 10 ** (args.snr_db / 20) / sosfiltfilt(sos_band, resp).std()
        for c, name in enumerate(NAMES5):
            X[:, c] += gains[name] * amp * resp

    # arrival stamps: bundles of 12 samples, 20 ms latency, jitter, stalls
    t0 = 50000.0
    arr = np.empty(n)
    for s in range(0, n, 12):
        e = min(s + 12, n)
        stamp = t0 + (e - 1) / 256.03 + 0.02 + rng.exponential(0.008)
        if rng.uniform() < 0.002:
            stamp += 0.3
        arr[s:e] = stamp
    arr = np.maximum.accumulate(arr)

    with open(os.path.join(args.session, "eeg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + NAMES5)
        for i in range(n):
            w.writerow([f"{arr[i]:.6f}"] + [f"{v:.4f}" for v in X[i]])
    with open(os.path.join(args.session, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep", "t_on", "t_off", "gap_s"])
        for r, on in enumerate(t_on, 1):
            w.writerow([r, f"{t0 + on:.6f}", f"{t0 + on + args.clip:.6f}", f"{gap:.3f}"])
    json.dump(dict(song=os.path.abspath(args.song), start_s=args.start, clip_s=args.clip, fs=fs_play, reps=args.reps,
                   gap_s=gap, jitter_s=jit, baseline_s=baseline, device="phantom", output_latency_s=latency,
                   phantom=dict(snr_db=None if args.no_response else args.snr_db, response=not args.no_response,
                                gains=gains, seed=args.seed)),
              open(os.path.join(args.session, "stimulus.json"), "w"), indent=1)
    print(f"phantom written to {args.session}: {total / 60:.1f} min, {args.reps} reps, "
          f"{'no response' if args.no_response else f'response at {args.snr_db} dB on AUX'}")


if __name__ == "__main__":
    main()
