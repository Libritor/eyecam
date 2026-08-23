"""Generate a synthetic eye-as-camera session (cursor log + EEG) so the whole
pipeline can be validated end-to-end without hardware.

Model: the observer fixates the guided cursor; when the cursor sits on a bright
part of the target it flickers, entraining an SSVEP at STIM_FREQ_HZ whose
amplitude is proportional to local luminance (with a short entrainment lag).
EEG = 1/f background noise + 10 Hz alpha + SSVEP fundamental + 30 Hz harmonic.

Usage:
    python simulate.py --target "text:NO" --out-dir runs/sim --snr 0.8 --seed 7
"""

import argparse
import csv
import os

import numpy as np

import config
import targets


def raster_cursor_log(target, seconds_per_cell, fs):
    """Serpentine raster over the grid; one row per EEG sample."""
    grid_h, grid_w = target.shape
    rows = []
    t = 0.0
    dt = 1.0 / fs
    n_cell = int(round(seconds_per_cell * fs))
    for gy in range(grid_h):
        xs = range(grid_w) if gy % 2 == 0 else range(grid_w - 1, -1, -1)
        for gx in xs:
            lum = float(target[gy, gx])
            for _ in range(n_cell):
                rows.append((t, gx, gy, lum, 1 if lum > 0.05 else 0))
                t += dt
    return rows


def synth_eeg(cursor_rows, fs, snr, rng):
    n = len(cursor_rows)
    t = np.arange(n) / fs

    # 1/f background noise, band-limited white on top.
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    shaping = 1.0 / np.sqrt(np.maximum(freqs, 0.5))
    pink = np.fft.irfft(spec * shaping, n)
    pink = pink / np.std(pink)
    noise = pink + 0.3 * rng.standard_normal(n)

    alpha = 0.6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))

    # SSVEP amplitude envelope: luminance under cursor, smoothed with a ~0.3 s
    # exponential lag to mimic entrainment/de-entrainment.
    lum = np.array([r[3] for r in cursor_rows])
    tau = 0.3
    a = np.exp(-1.0 / (tau * fs))
    env = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = a * acc + (1 - a) * lum[i]
        env[i] = acc

    f0, f1 = config.STIM_FREQ_HZ, config.HARMONIC_HZ
    ssvep = env * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * f1 * t))
    return noise + alpha + snr * ssvep


def write_session(out_dir, cursor_rows, eeg, target):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cursor_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "grid_x", "grid_y", "luminance", "flicker_on"])
        for r in cursor_rows:
            w.writerow([f"{r[0]:.6f}", r[1], r[2], f"{r[3]:.4f}", r[4]])
    with open(os.path.join(out_dir, "eeg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "SIM"])
        dt = 1.0 / config.FS
        for i, v in enumerate(eeg):
            w.writerow([f"{i * dt:.6f}", f"{v:.6f}"])
    np.save(os.path.join(out_dir, "target.npy"), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--out-dir", default=os.path.join("runs", "sim"))
    ap.add_argument("--snr", type=float, default=0.8,
                    help="SSVEP amplitude relative to unit-variance background")
    ap.add_argument("--seconds-per-cell", type=float,
                    default=config.SECONDS_PER_CELL)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle-eeg", action="store_true",
                    help="control arm: destroy EEG/cursor correspondence")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    target = targets.load_target(args.target)
    cursor_rows = raster_cursor_log(target, args.seconds_per_cell, config.FS)
    eeg = synth_eeg(cursor_rows, config.FS, args.snr, rng)
    if args.shuffle_eeg:
        # Circular shift by a third of the session: preserves the EEG's own
        # statistics but breaks alignment with the cursor path.
        eeg = np.roll(eeg, len(eeg) // 3)
    write_session(args.out_dir, cursor_rows, eeg, target)
    dur = len(eeg) / config.FS
    print(f"simulated session: {len(eeg)} samples ({dur:.0f} s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
