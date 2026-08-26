"""Synthetic real-time subject for full-stack testing without hardware.

Tails the session's calib_log.csv / cursor_log.csv as the stimulus writes
them, and sends real /muse/eeg OSC packets whose SSVEP follows what is
actually on screen:
  - amplitude follows the logged luminance, but drops to zero if flicker_on
    stops toggling (a stimulus that renders a static patch produces no SSVEP);
  - carrier frequency is MEASURED from the flicker_on rising-edge intervals,
    so a stimulus flickering at the wrong rate (bad half_period math, an
    unexpected display refresh) shifts the injected SSVEP off the 15 Hz
    analysis bin and fails the gate.
SNR is deliberately asymmetric across channels (strong on TP9/TP10, weak on
AF7/AF8) so calibration must find the right channels, and each channel
carries a slow large-amplitude baseline drift (ordinary for dry electrodes)
so the artifact statistics are exercised too.

Pacing: Windows sleep granularity is ~15.6 ms, so per-sample pacing at 256 Hz
is unreliable. Samples are sent in 4-packet bursts on a 64 Hz absolute
schedule — which also mimics real Wi-Fi aggregation of the phone stream.

Spawned by run_session.py --source phantom; can also run standalone.
"""

import argparse
import ctypes
import math
import os
import socket
import struct
import sys
import time
from collections import deque

import numpy as np

import config
from tailer import CsvTail

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
CH_SNR = [0.8, 0.2, 0.2, 0.8]
DC_OFFSET = 800.0  # Muse-like raw baseline (robust stats must absorb it)


def osc_eeg_packet(vals):
    addr = b"/muse/eeg\x00\x00\x00"
    tags = b",ffff\x00\x00\x00"
    return addr + tags + b"".join(struct.pack(">f", float(v)) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seconds", type=float, default=3600)
    args = ap.parse_args()

    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)

    tails = [CsvTail(os.path.join(args.session, "calib_log.csv")),
             CsvTail(os.path.join(args.session, "cursor_log.csv"))]

    rng = np.random.default_rng(args.seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    fs = config.FS
    burst = 4
    burst_dt = burst / fs
    a_env = math.exp(-1.0 / (0.3 * fs))   # SSVEP entrainment lag ~0.3 s
    a_pink = 0.98                          # AR(1) fast background noise
    pink_gain = 0.199                      # -> ~unit-variance
    a_drift = 0.999935                     # slow electrode wander, tau ~60 s
    drift_gain = 0.0913                    # -> stationary std ~8x HF noise

    lum = 0.0
    env = 0.0
    pink = np.zeros(4)
    drift = np.zeros(4)
    n = 0
    sent = 0

    # Flicker observed from the logs: rising-edge periods give the actual
    # on-screen alternation rate; a stale toggle means no flicker is shown.
    last_fl = 0
    last_rise_t = None
    last_toggle_pc = -1e9
    periods = deque(maxlen=8)
    phase = 0.0

    t_start = time.perf_counter()
    next_t = t_start

    print(f"phantom subject -> {args.host}:{args.port} "
          f"(TP9/TP10 snr {CH_SNR[0]}, AF7/AF8 snr {CH_SNR[1]})")
    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break
            if time.perf_counter() - t_start > args.max_seconds:
                break
            for tail in tails:
                for row in tail.poll():
                    try:
                        t_row = float(row[0])
                        lum = float(row[3])
                        fl = int(row[4])
                    except (IndexError, ValueError):
                        continue
                    if fl != last_fl:
                        last_toggle_pc = time.perf_counter()
                        if fl == 1:
                            if last_rise_t is not None:
                                period = t_row - last_rise_t
                                if 0.01 < period < 0.5:
                                    periods.append(period)
                            last_rise_t = t_row
                        last_fl = fl

            flicker_live = time.perf_counter() - last_toggle_pc < 0.15
            drive = lum if flicker_live else 0.0
            f_meas = (1.0 / np.median(periods) if len(periods) >= 2
                      else config.STIM_FREQ_HZ)

            for _ in range(burst):
                t = n / fs
                env = a_env * env + (1 - a_env) * drive
                phase += 2 * math.pi * f_meas / fs
                ssvep = env * (math.sin(phase) + 0.4 * math.sin(2 * phase))
                alpha = 0.6 * math.sin(2 * math.pi * 10.0 * t)
                pink = a_pink * pink + pink_gain * rng.standard_normal(4)
                drift = a_drift * drift + drift_gain * rng.standard_normal(4)
                white = 0.45 * rng.standard_normal(4)
                vals = [DC_OFFSET + drift[c] + pink[c] + white[c] + alpha
                        + CH_SNR[c] * ssvep for c in range(4)]
                sock.sendto(osc_eeg_packet(vals), (args.host, args.port))
                sent += 1
                n += 1

            next_t += burst_dt
            delay = next_t - time.perf_counter()
            if delay > 0.002:
                time.sleep(delay - 0.002)
            while time.perf_counter() < next_t:
                pass  # short spin for sub-ms precision
    except KeyboardInterrupt:
        pass
    finally:
        if sys.platform == "win32":
            ctypes.windll.winmm.timeEndPeriod(1)
    elapsed = time.perf_counter() - t_start
    print(f"phantom sent {sent} samples in {elapsed:.1f} s "
          f"({sent / max(elapsed, 1e-9):.1f} Hz effective)")


if __name__ == "__main__":
    main()
