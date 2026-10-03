"""Record an EEG LSL stream to <session>/eeg.csv.

Works with any tool that puts a Muse on the lab streaming layer:
  - muselsl  (pip install muselsl; `muselsl stream` in another terminal,
    or let xr_session.py --muse / run_session.py --source lsl start it)
  - Petal Metrics app (choose LSL output)
  - BlueMuse (older Windows option)

Start this BEFORE stimulus.py so the whole scan is covered:
    python acquire.py --session runs/live1
Ctrl+C to stop.

Timestamps are written on pylsl's local_clock(), the clock the stimulus uses,
and values on the libmuse scale (0..1650 uV, resting near 850) exactly as the
phone apps send them, so the contact check and analysis see the same numbers.
muselsl stamps samples with time.time() unless started with --lsltime; such
wall-clock stamps are detected and converted. The recorder waits for the
stream to appear and re-attaches if it disappears (muselsl relaunching after a
contact glitch), so one eeg.csv covers the whole session.
"""

import argparse
import csv
import os
import time

from pylsl import StreamInlet, local_clock, resolve_byprop


def to_libmuse(v):
    """muselsl microvolts (0.48828125 * (count - 2048)) -> the libmuse scale the
    phone apps stream and the rest of this repo expects (count * 1650 / 4095:
    0..1650 uV, resting near 850). The contact check and every recorded
    eeg.csv are then identical in form to the phone path."""
    return (v / 0.48828125 + 2048.0) * 1650.0 / 4095.0


def channel_names(info):
    names = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        label = ch.child_value("label") or f"ch{len(names)}"
        # muselsl calls the aux input "Right AUX"; osc_acquire.py calls it AUX
        names.append("AUX" if label.replace(" ", "").upper() in ("RIGHTAUX", "AUXR", "AUX") else label)
        ch = ch.next_sibling()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join("runs", "live1"))
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to record; 0 = until Ctrl+C")
    ap.add_argument("--stop-file", default="",
                    help="exit cleanly when this file appears")
    ap.add_argument("--append", action="store_true",
                    help="append to existing CSV (crash-resume)")
    ap.add_argument("--max-seconds", type=float, default=14400,
                    help="absolute runtime cap (orphan protection)")
    ap.add_argument("--no-aux", action="store_true",
                    help="record only TP9, AF7, AF8, TP10 (nothing on the aux port)")
    args = ap.parse_args()
    os.makedirs(args.session, exist_ok=True)

    path = os.path.join(args.session, "eeg.csv")
    n = 0
    t0 = None
    names = None
    mode = "a" if args.append else "w"
    had_data = args.append and os.path.exists(path) and os.path.getsize(path) > 0
    t_launch = time.monotonic()

    def stopping():
        return (args.stop_file and os.path.exists(args.stop_file)) or \
            time.monotonic() - t_launch > args.max_seconds

    def attach():
        print("resolving EEG stream on LSL (muselsl / Petal / mind2motor)...", flush=True)
        while not stopping():
            streams = resolve_byprop("type", "EEG", timeout=2)
            if streams:
                inlet = StreamInlet(streams[0], max_chunklen=12, recover=False)
                info = inlet.info()
                print(f"stream: {info.name()} @ {info.nominal_srate()} Hz, "
                      f"channels: {channel_names(info)}", flush=True)
                return inlet
        return None

    with open(path, mode, newline="", buffering=1) as f:
        w = csv.writer(f)
        try:
            while not stopping():
                inlet = attach()
                if inlet is None:
                    break
                found = channel_names(inlet.info())
                keep = [i for i, nm in enumerate(found) if not (args.no_aux and nm == "AUX")]
                if names is None:
                    names = [found[i] for i in keep]
                    if not had_data:
                        w.writerow(["time"] + names)
                elif [found[i] for i in keep] != names:
                    print(f"stream channels changed to {found}; keeping columns {names}", flush=True)
                keep = keep[:len(names)]
                offset = None
                last = time.monotonic()
                while not stopping():
                    try:
                        chunk, ts = inlet.pull_chunk(timeout=1.0)
                    except Exception:  # stream lost (muselsl exited)
                        break
                    if ts:
                        last = time.monotonic()
                        if offset is None:
                            # same-machine LSL clock, or wall-clock stamps (muselsl default)
                            lag = ts[-1] - local_clock()
                            offset = (time.time() - local_clock()) if abs(lag) > 3600 \
                                else -inlet.time_correction() if abs(lag) > 1 else 0.0
                    for row, t in zip(chunk, ts):
                        t -= offset
                        w.writerow([f"{t:.6f}"] + [f"{to_libmuse(row[i]):.4f}" for i in keep])
                        n += 1
                        t0 = t0 or t
                        if args.duration and t - t0 >= args.duration:
                            raise KeyboardInterrupt
                    if time.monotonic() - last > 4:
                        print("EEG stream silent for 4 s; re-attaching", flush=True)
                        break
        except KeyboardInterrupt:
            pass
    print(f"{n} samples -> {path}")


if __name__ == "__main__":
    main()
