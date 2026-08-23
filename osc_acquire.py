"""Record EEG streamed from the MuseLog app (OSC over Wi-Fi) to <session>/eeg.csv.

MuseLog (museai repo) sends UDP OSC to a target IP/port set in its
"OSC Streaming Settings" dialog. Formats it can emit:
  /muse/eeg      raw EEG floats [TP9, AF7, AF8, TP10]   (SnowballArcade format,
                 decimated x4 -> ~64 Hz unless full-rate is enabled in the app)
  /person{n}/eeg averaged band powers [delta, theta, alpha, beta]  (legacy)
  /eeg           4 raw floats (oldest convention, port 6001 tools)

This recorder accepts all three; only raw EEG rows go into eeg.csv (band powers
go to bandpower.csv for reference). Timestamps use the same LSL clock as
stimulus.py, so reconstruct.py aligns them directly.

On the phone, set Target IP to THIS PC's Wi-Fi address and port to 5000, then
Start Streaming. Check the printed packet counter to confirm arrival.

Usage:
    python osc_acquire.py --session runs/live1 [--port 5000]
Ctrl+C to stop.
"""

import argparse
import csv
import os
import socket
import time

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter

from pythonosc.osc_message import OscMessage
from pythonosc.osc_bundle import OscBundle


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def iter_messages(data):
    """Yield OscMessage objects from a datagram (handles bundles too)."""
    if OscBundle.dgram_is_bundle(data):
        for item in OscBundle(data):
            if isinstance(item, OscMessage):
                yield item
    else:
        yield OscMessage(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join("runs", "live1"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to record; 0 = until Ctrl+C")
    args = ap.parse_args()
    os.makedirs(args.session, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"listening for MuseLog OSC on UDP {args.port}")
    print(f"in the app's OSC settings, set Target IP to one of: "
          f"{', '.join(local_ips()) or 'unknown'}  port {args.port}")

    eeg_path = os.path.join(args.session, "eeg.csv")
    bp_path = os.path.join(args.session, "bandpower.csv")
    n_eeg = n_bp = n_other = 0
    t_start = None
    last_report = time.monotonic()

    with open(eeg_path, "w", newline="") as fe, \
         open(bp_path, "w", newline="") as fb:
        we = csv.writer(fe)
        we.writerow(["time", "TP9", "AF7", "AF8", "TP10"])
        wb = csv.writer(fb)
        wb.writerow(["time", "address", "delta", "theta", "alpha", "beta"])
        try:
            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    if time.monotonic() - last_report > 5 and n_eeg == 0:
                        print("  ... no packets yet (check app IP/port, "
                              "same Wi-Fi, Start Streaming pressed)")
                        last_report = time.monotonic()
                    continue
                t = local_clock()
                t_start = t_start or t
                for msg in iter_messages(data):
                    addr, p = msg.address, list(msg.params)
                    if addr in ("/muse/eeg", "/eeg") and len(p) >= 4:
                        we.writerow([f"{t:.6f}"] + [f"{float(v):.4f}"
                                                    for v in p[:4]])
                        n_eeg += 1
                    elif addr.startswith("/person") and addr.endswith("/eeg"):
                        wb.writerow([f"{t:.6f}", addr] +
                                    [f"{float(v):.4f}" for v in p[:4]])
                        n_bp += 1
                    else:
                        n_other += 1
                if time.monotonic() - last_report > 5:
                    print(f"  eeg rows: {n_eeg}  bandpower: {n_bp}  "
                          f"other addrs: {n_other}")
                    last_report = time.monotonic()
                if args.duration and t_start and t - t_start >= args.duration:
                    break
        except KeyboardInterrupt:
            pass

    print(f"{n_eeg} raw EEG rows -> {eeg_path}")
    print(f"{n_bp} band-power rows -> {bp_path}")
    if n_eeg == 0 and n_bp > 0:
        print("NOTE: only band powers arrived — switch the app's OSC Format "
              "to SnowballArcade so raw /muse/eeg is included.")


if __name__ == "__main__":
    main()
