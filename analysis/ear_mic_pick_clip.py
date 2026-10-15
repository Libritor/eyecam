"""Which 30 s of a song to play for the Ear-as-a-Microphone run.

The decoder reads back the 1-8 Hz loudness envelope, so the easiest clip is
the one whose envelope swings the most (a loud, rhythmic chorus), not the
quiet verse. Every window of --clip seconds (hop 5 s) is scored by the
modulation depth of its envelope: sd of the 1-8 Hz band-passed envelope over
the mean of the positive envelope. The best five are printed with the
--start value to give ear_mic_session.py.

    python analysis/ear_mic_pick_clip.py "C:\\Users\\alexa\\Downloads\\Sia - Unstoppable.mp3"
    python analysis/ear_mic_pick_clip.py song.m4a --clip 30
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ear_mic_session import read_audio  # noqa: E402
from ear_mic_decode import song_envelope, band_env, FS_D  # noqa: E402
from scipy.signal import butter, sosfiltfilt, resample_poly  # noqa: E402
from math import gcd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("--clip", type=float, default=30.0)
    ap.add_argument("--hop", type=float, default=5.0)
    args = ap.parse_args()
    x, fs = read_audio(args.song)
    dur = len(x) / fs
    if fs > 16000:                                           # the envelope needs nothing above 8 kHz
        g = gcd(int(fs), 16000)
        x, fs = resample_poly(x.astype(np.float64), 16000 // g, int(fs) // g), 16000
    env_pos = song_envelope(x.astype(np.float64), fs)
    env_bp = sosfiltfilt(butter(2, [1.0, 8.0], "bandpass", fs=FS_D, output="sos"), env_pos)
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    level = np.sqrt(np.maximum(c[int(fs):] - c[:-int(fs)], 0) / fs)   # 1 s moving RMS
    rows = []
    n = int(args.clip * FS_D)
    for s in np.arange(0.0, dur - args.clip + 1e-9, args.hop):
        a = int(s * FS_D)
        if a + n > len(env_pos):
            break
        depth = env_bp[a:a + n].std() / (env_pos[a:a + n].mean() + 1e-12)
        loud = 20 * np.log10(level[int(s * fs):int((s + args.clip) * fs)].mean() + 1e-12)
        rows.append((depth, loud, s))
    rows.sort(reverse=True)
    print(f"{os.path.basename(args.song)}: {dur / 60:.0f} min {dur % 60:.0f} s, windows of {args.clip:.0f} s")
    print("  modulation  level   --start")
    for depth, loud, s in rows[:5]:
        print(f"  {depth:8.3f}  {loud:6.1f} dB  {s:5.0f}   ({int(s // 60)}:{int(s % 60):02d} to {int((s + args.clip) // 60)}:{int((s + args.clip) % 60):02d})")
    best = rows[0][2]
    print(f'\nrun: python ear_mic_session.py --song "{args.song}" --start {best:.0f} --clip {args.clip:.0f} --reps 24 --session runs/ear_mic1')


if __name__ == "__main__":
    main()
