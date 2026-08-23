"""Record an EEG LSL stream to <session>/eeg.csv.

Works with any tool that puts a Muse on the lab streaming layer:
  - muselsl  (pip install muselsl; `muselsl stream` in another terminal)
  - Petal Metrics app (choose LSL output)
  - BlueMuse (older Windows option)

Start this BEFORE stimulus.py so the whole scan is covered:
    python acquire.py --session runs/live1
Ctrl+C to stop.
"""

import argparse
import csv
import os

from pylsl import StreamInlet, resolve_byprop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join("runs", "live1"))
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to record; 0 = until Ctrl+C")
    args = ap.parse_args()
    os.makedirs(args.session, exist_ok=True)

    print("resolving EEG stream on LSL (start muselsl/Petal first)...")
    streams = resolve_byprop("type", "EEG", timeout=15)
    if not streams:
        raise SystemExit("no EEG stream found — is the Muse streaming to LSL?")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    info = inlet.info()
    n_ch = info.channel_count()
    names = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(n_ch):
        names.append(ch.child_value("label") or f"ch{len(names)}")
        ch = ch.next_sibling()
    print(f"stream: {info.name()} @ {info.nominal_srate()} Hz, channels: {names}")

    path = os.path.join(args.session, "eeg.csv")
    n = 0
    t0 = None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + names)
        try:
            while True:
                chunk, ts = inlet.pull_chunk(timeout=1.0)
                for row, t in zip(chunk, ts):
                    w.writerow([f"{t:.6f}"] + [f"{v:.4f}" for v in row])
                    n += 1
                    t0 = t0 or t
                    if args.duration and t - t0 >= args.duration:
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
    print(f"{n} samples -> {path}")


if __name__ == "__main__":
    main()
