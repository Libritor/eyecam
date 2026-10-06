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

Optional, for testing blink / eye-closure masking (all off by default):
  --aux        a 5th channel, Oz: the strongest SSVEP and the alpha that rises
               when the eyes close
  --blinks N   N blinks per minute: a ~0.3 s frontal deflection (AF7/AF8 large,
               TP9/TP10 small) and no SSVEP while the lids are down
  --closures N N eye closures per minute, 2-4 s: no SSVEP, alpha x6 on Oz

Smooth flicker (--mode smooth) is a sine, so its on/off edges are too coarse to
measure its frequency: the phantom takes it from the hz_r/hz_g/hz_b columns
and responds pi/4 as strongly as to on/off flicker. For testing that mode:
  --tuning PEAK:WIDTH   response falls off away from PEAK Hz (Gaussian, Hz)
  --color-gain R,G,B    response to each colour of a colour scan
  --response-gamma G    response = brightness ** G (below 1: shades crowd up)
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
    tags = b"," + b"f" * len(vals)
    tags += b"\x00" * (4 - len(tags) % 4)   # OSC strings: NUL-padded to 4 bytes
    return addr + tags + b"".join(struct.pack(">f", float(v)) for v in vals)


BLINK_S = 0.3
BLINK_GAIN = [40.0, 150.0, 150.0, 40.0, 5.0]   # TP9 AF7 AF8 TP10 AUX


class Eyes:
    """Random blinks and closures: open() is 0 while the eyes are shut."""

    def __init__(self, rng, fs, blinks_per_min, closures_per_min):
        self.rng, self.fs = rng, fs
        self.b_rate = blinks_per_min / 60.0
        self.c_rate = closures_per_min / 60.0
        self.next_b = self._gap(self.b_rate)
        self.next_c = self._gap(self.c_rate)
        self.blink_end = self.closure_end = -1.0
        self.blink_t0 = 0.0

    def _gap(self, rate):
        return self.rng.exponential(1 / rate) + 3.0 if rate > 0 else float("inf")

    def step(self, t):
        if t >= self.next_b:
            self.blink_t0, self.blink_end = t, t + BLINK_S
            self.next_b = t + self._gap(self.b_rate)
        if t >= self.next_c and t > self.blink_end:
            self.closure_end = t + self.rng.uniform(2.0, 4.0)
            self.next_c = self.closure_end + self._gap(self.c_rate)

    def blink(self, t):
        """Frontal deflection shape (0..1) at time t."""
        if t < self.blink_end:
            return math.sin(math.pi * (t - self.blink_t0) / BLINK_S)
        return 0.0

    def closed(self, t):
        return t < self.closure_end

    def open(self, t):
        return 0.0 if (t < self.blink_end or t < self.closure_end) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--aux", action="store_true",
                    help="add an Oz channel (5th column, AUX)")
    ap.add_argument("--blinks", type=float, default=0.0,
                    help="blinks per minute (frontal spikes + SSVEP dropout)")
    ap.add_argument("--closures", type=float, default=0.0,
                    help="eye closures per minute (2-4 s, alpha up, no SSVEP)")
    ap.add_argument("--tuning", default="",
                    help="PEAK:WIDTH in Hz: response curve over flicker frequency")
    ap.add_argument("--color-gain", default="1,1,1",
                    help="response to R,G,B in a colour scan (color_log / smooth_log)")
    ap.add_argument("--response-gamma", type=float, default=1.0,
                    help="response = brightness ** gamma")
    args = ap.parse_args()
    nch = 5 if args.aux else 4
    peak, width = ([float(v) for v in args.tuning.split(":")] if args.tuning
                   else (0.0, 0.0))
    color_gain = [float(v) for v in args.color_gain.split(",")]

    def tuning(f):
        return math.exp(-0.5 * ((f - peak) / width) ** 2) if width > 0 and f > 0 else 1.0
    snr = CH_SNR + ([1.2] if args.aux else [])

    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)

    tails = [(n, CsvTail(os.path.join(args.session, n))) for n in
             ("calib_log.csv", "cursor_log.csv", "sweep_log.csv",
              "ccal_log.csv", "color_log.csv", "assr_log.csv", "mux_log.csv",
              "bwb_cal_log.csv", "bwb_log.csv", "smooth_log.csv",
              "plane_red_log.csv", "plane_green_log.csv", "plane_blue_log.csv")]
    # --mode planes: one black/<colour> scan per plane; the colour gain of
    # that plane applies to the whole scan
    plane_of = {"plane_red_log.csv": 0, "plane_green_log.csv": 1, "plane_blue_log.csv": 2}
    cur_plane = None

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
    pink = np.zeros(nch)
    drift = np.zeros(nch)
    eyes = Eyes(rng, fs, args.blinks, args.closures)
    n = 0
    sent = 0

    # Flicker observed from the logs: rising-edge periods give the actual
    # on-screen alternation rate; a stale toggle means no flicker is shown.
    # Three trackers: the primary flicker (column 4), and the colour-scan
    # G / B taggings (columns 6, 7); plus the audio drive (columns 3 + 11).
    class Tracker:
        def __init__(self):
            self.last_fl = 0
            self.last_rise_t = None
            self.last_toggle_pc = -1e9
            self.periods = deque(maxlen=8)
            self.phase = 0.0
            self.env = 0.0
            self.drive = 0.0

        def see(self, t_row, fl):
            if fl != self.last_fl:
                self.last_toggle_pc = time.perf_counter()
                if fl == 1:
                    if self.last_rise_t is not None:
                        period = t_row - self.last_rise_t
                        if 0.01 < period < 0.5:
                            self.periods.append(period)
                    self.last_rise_t = t_row
                self.last_fl = fl

        def live(self):
            return time.perf_counter() - self.last_toggle_pc < 0.15

        def freq(self, fallback):
            return (1.0 / np.median(self.periods) if len(self.periods) >= 2
                    else fallback)

    trk = [Tracker(), Tracker(), Tracker()]
    comp = [0.0, 0.0, 0.0]          # r, g, b of the current cell
    hz_s = [0.0, 0.0, 0.0]          # smooth-flicker frequency per colour (0 = on/off)
    in_cell = False
    audio_hz, audio_lum, audio_pc = 0.0, 0.0, -1e9
    audio_phase, audio_env = 0.0, 0.0
    last_fl = 0
    last_rise_t = None
    last_toggle_pc = -1e9
    periods = deque(maxlen=8)
    phase = 0.0

    t_start = time.perf_counter()
    next_t = t_start

    print(f"phantom subject -> {args.host}:{args.port} "
          f"(TP9/TP10 snr {CH_SNR[0]}, AF7/AF8 snr {CH_SNR[1]}"
          + (", Oz snr 1.2" if args.aux else "")
          + (f"; {args.blinks:g} blinks/min" if args.blinks else "")
          + (f"; {args.closures:g} closures/min" if args.closures else "") + ")")
    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break
            if time.perf_counter() - t_start > args.max_seconds:
                break
            for log_name, tail in tails:
                for row in tail.poll():
                    try:
                        t_row = float(row[0])
                        lum = float(row[3])
                        fl = int(row[4])
                    except (IndexError, ValueError):
                        continue
                    trk[0].see(t_row, fl)
                    cur_plane = plane_of.get(log_name, None)
                    if len(row) >= 12:
                        try:
                            trk[1].see(t_row, int(row[6]))
                            trk[2].see(t_row, int(row[7]))
                            comp = [float(row[8]), float(row[9]), float(row[10])]
                            hz_a = float(row[11])
                        except ValueError:
                            hz_a = 0.0
                        if hz_a > 0:
                            audio_hz, audio_lum = hz_a, lum
                            audio_pc = time.perf_counter()
                    else:
                        comp = [lum, 0.0, 0.0]
                    try:
                        hz_s = [float(v) for v in row[12:15]] if len(row) >= 15 \
                            else [0.0, 0.0, 0.0]
                        in_cell = int(row[1]) >= 0 and log_name in (
                            "color_log.csv", "smooth_log.csv")
                    except ValueError:
                        hz_s = [0.0, 0.0, 0.0]

            # primary drive = luminance column gated by a live primary toggle
            f_meas = hz_s[0] or trk[0].freq(config.STIM_FREQ_HZ)
            f_g = hz_s[1] or trk[1].freq(0.0)
            f_b = hz_s[2] or trk[2].freq(0.0)
            for k, (v, f_k) in enumerate(zip((lum, comp[1], comp[2]), (f_meas, f_g, f_b))):
                gain = tuning(f_k) * (math.pi / 4 if hz_s[k] else 1.0)
                if in_cell:
                    gain *= color_gain[k]
                elif k == 0 and cur_plane is not None:
                    gain *= color_gain[cur_plane]
                trk[k].drive = gain * max(v, 0.0) ** args.response_gamma \
                    if trk[k].live() else 0.0
            audio_live = time.perf_counter() - audio_pc < 0.3
            a_drive = audio_lum if audio_live else 0.0

            for _ in range(burst):
                t = n / fs
                env = a_env * env + (1 - a_env) * trk[0].drive
                phase += 2 * math.pi * f_meas / fs
                ssvep = env * (math.sin(phase) + 0.4 * math.sin(2 * phase))
                for k, f_k in ((1, f_g), (2, f_b)):
                    trk[k].env = a_env * trk[k].env + (1 - a_env) * trk[k].drive
                    if f_k > 0:
                        trk[k].phase += 2 * math.pi * f_k / fs
                        ssvep += trk[k].env * (math.sin(trk[k].phase)
                                               + 0.4 * math.sin(2 * trk[k].phase))
                audio_env = a_env * audio_env + (1 - a_env) * a_drive
                if audio_hz > 0:
                    audio_phase += 2 * math.pi * audio_hz / fs
                    ssvep += 0.6 * audio_env * math.sin(audio_phase)
                alpha = 0.6 * math.sin(2 * math.pi * 10.0 * t)
                pink = a_pink * pink + pink_gain * rng.standard_normal(nch)
                drift = a_drift * drift + drift_gain * rng.standard_normal(nch)
                white = 0.45 * rng.standard_normal(nch)
                eyes.step(t)
                ssvep *= eyes.open(t)           # no visual input, no SSVEP
                blink = eyes.blink(t)
                a_oz = 6.0 if eyes.closed(t) else 1.0
                vals = [DC_OFFSET + drift[c] + pink[c] + white[c]
                        + alpha * (a_oz if c == 4 else 1.0)
                        + snr[c] * ssvep + BLINK_GAIN[c] * blink for c in range(nch)]
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
