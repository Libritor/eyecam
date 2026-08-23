"""Gate for the MuseLog-OSC ingest path. Two checks:

1. UDP round-trip: send synthetic /muse/eeg packets at ~64 Hz to a live
   osc_acquire.py process; recorded rows must match what was sent.
2. Decimated-rate reconstruction: the simulation gate's 256 Hz session,
   naively decimated x4 (exactly what the MuseLog app does), must still
   reconstruct (r >= 0.50).

Run:  python osc_gate.py
"""

import csv
import os
import shutil
import socket
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PORT = 5010


def osc_eeg_packet(vals):
    addr = b"/muse/eeg\x00\x00\x00"
    tags = b",ffff\x00\x00\x00"
    return addr + tags + b"".join(struct.pack(">f", v) for v in vals)


def check_roundtrip():
    session = os.path.join(ROOT, "runs", "gate_osc_rt")
    shutil.rmtree(session, ignore_errors=True)
    proc = subprocess.Popen(
        [PY, "osc_acquire.py", "--session", "runs/gate_osc_rt",
         "--port", str(PORT), "--duration", "4"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)  # let it bind

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = []
    for i in range(64 * 4):
        vals = [float(i), float(i) + 0.1, float(i) + 0.2, float(i) + 0.3]
        sock.sendto(osc_eeg_packet(vals), ("127.0.0.1", PORT))
        sent.append(vals)
        time.sleep(1 / 64)
    proc.wait(timeout=30)

    with open(os.path.join(session, "eeg.csv"), newline="") as f:
        rows = list(csv.reader(f))[1:]
    got = len(rows)
    ok_count = got >= len(sent) * 0.95
    first = [float(v) for v in rows[0][1:]] if rows else []
    # Row 0 should be one of the first packets, channels intact and ordered.
    ok_vals = (len(first) == 4 and
               abs(first[1] - first[0] - 0.1) < 1e-3 and
               abs(first[3] - first[0] - 0.3) < 1e-3)
    print(f"round-trip: sent {len(sent)}, recorded {got}, "
          f"channel order {'intact' if ok_vals else 'BROKEN'}")
    return ok_count and ok_vals


def check_decimated_reconstruction():
    src = os.path.join(ROOT, "runs", "gate_real")
    if not os.path.exists(os.path.join(src, "eeg.csv")):
        subprocess.run([PY, "gate_test.py"], cwd=ROOT, check=False)
    dst = os.path.join(ROOT, "runs", "gate_osc64")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    shutil.copy(os.path.join(src, "cursor_log.csv"), dst)
    shutil.copy(os.path.join(src, "target.npy"), dst)
    with open(os.path.join(src, "eeg.csv"), newline="") as f:
        rows = list(csv.reader(f))
    with open(os.path.join(dst, "eeg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(rows[1::4])  # the app's x4 decimation, no anti-alias filter

    subprocess.run([PY, "reconstruct.py", "--session", "runs/gate_osc64"],
                   cwd=ROOT, check=True, capture_output=True)
    target = np.load(os.path.join(dst, "target.npy"))
    grid = np.load(os.path.join(dst, "reconstruction_grid.npy"))
    r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])
    print(f"64 Hz decimated reconstruction: r = {r:.3f}  (need >= 0.50)")
    return r >= 0.50


def main():
    ok1 = check_roundtrip()
    ok2 = check_decimated_reconstruction()
    print("OSC GATE PASS" if ok1 and ok2 else "OSC GATE FAIL")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
