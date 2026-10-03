"""Feed an LSL EEG stream into anything that only takes OSC (osc_acquire.py).

xr_session.py can now read the Muse directly (--muse); this bridge is only needed
for tools that listen for OSC, or to feed a second program from the same headset.

xr_session.py only listens for OSC over UDP (Mind Monitor / MuseLog on a phone).
This bridge reads the Muse's LSL stream instead -- from `muselsl stream`, which it
can launch and keep alive for you -- and re-sends every sample as the same
/muse/eeg OSC message the phone app would, so nothing else has to change.

    # terminal 1 (the session, as usual; any free UDP port)
    python xr_session.py --mode visual --osc-port 5011 ...
    # terminal 2 (connects to the Muse and feeds the session)
    python lsl_to_osc.py --port 5011 --start-muselsl

Timing: osc_acquire.py stamps each sample with its ARRIVAL time, so samples are
not forwarded in muselsl's bursts of 12 but re-paced to their own sample
timestamps plus a fixed delay (--delay, default 0.15 s). A constant delay only
shifts the response in time; the reconstruction scores a whole dwell at once.

Channels: TP9, AF7, AF8, TP10, plus muselsl's 5th "Right AUX" column, which is
where an Oz electrode on the aux port appears (osc_acquire names it AUX). Use
--no-aux when nothing is plugged into the aux port.
"""

import argparse
import collections
import sys
import threading
import time

try:
    from pylsl import StreamInlet, resolve_byprop
    from pythonosc.udp_client import SimpleUDPClient
except ImportError as e:
    raise SystemExit(f"{e}: pip install pylsl python-osc muselsl")
from muse_stream import MuseStream


def resolve(stop):
    while not stop.is_set():
        streams = resolve_byprop("type", "EEG", timeout=2.0)
        if streams:
            info = streams[0]
            print(f"LSL: '{info.name()}' {info.channel_count()} ch @ {info.nominal_srate():.0f} Hz")
            return StreamInlet(info, max_buflen=10)
        print("LSL: waiting for an EEG stream (muselsl stream / mind2motor run_muse.py --live)...")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5011, help="xr_session.py --osc-port (default 5011)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--delay", type=float, default=0.15, help="fixed forwarding delay in s")
    ap.add_argument("--no-aux", action="store_true", help="send only TP9, AF7, AF8, TP10")
    ap.add_argument("--start-muselsl", action="store_true", help="launch and supervise `muselsl stream`")
    ap.add_argument("--address", default=None, help="Muse MAC address for --start-muselsl")
    ap.add_argument("--model", default="auto", choices=["auto", "athena", "legacy"],
                    help="Muse protocol for --start-muselsl (Athena = Muse S Gen 3)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # status lines visible at once in any terminal

    stop = threading.Event()
    preset = None if args.no_aux or args.model == "athena" else "20"   # aux (Oz) ON
    muse = MuseStream(args.address, args.model, preset=preset).start() if args.start_muselsl else None

    osc = SimpleUDPClient(args.host, args.port)
    print(f"forwarding to OSC {args.host}:{args.port} as /muse/eeg")
    queue = collections.deque()
    anchor = None            # (wall time, sample timestamp) pairing for pacing
    sent, last_report, last_data = 0, time.monotonic(), time.monotonic()
    inlet = None
    try:
        while True:
            if inlet is None:
                inlet = resolve(stop)
                anchor, last_data = None, time.monotonic()
                n_ch = inlet.info().channel_count()
                keep = 4 if args.no_aux else min(n_ch, 5)
            chunk, stamps = inlet.pull_chunk(timeout=0.0, max_samples=256)
            now = time.perf_counter()
            if stamps:
                last_data = time.monotonic()
                for s, ts in zip(chunk, stamps):
                    # re-anchor on start, after a gap, or if the stream drifted > 0.5 s
                    if anchor is None or abs((anchor[0] + ts - anchor[1] + args.delay) - now) > 0.5 + args.delay:
                        anchor = (now, ts)
                        queue.clear()
                    queue.append((anchor[0] + ts - anchor[1] + args.delay, s[:keep]))
            while queue and queue[0][0] <= now:
                _, s = queue.popleft()
                osc.send_message("/muse/eeg", [float(v) for v in s])
                sent += 1
            if time.monotonic() - last_data > 5:
                print("no EEG for 5 s -- reconnecting to the LSL stream")
                inlet = None
                continue
            if time.monotonic() - last_report > 5:
                print(f"  forwarded {sent} samples ({keep} channels), queue {len(queue)}")
                last_report = time.monotonic()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if muse:
            muse.stop()
        print(f"stopped after {sent} samples")


if __name__ == "__main__":
    main()
