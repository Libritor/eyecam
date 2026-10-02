"""Record a Muse headband straight over this computer's Bluetooth to
<session>/eeg.csv. No phone app (MuseLog, Mind Monitor) in the loop.

Works with the Bluetooth LE Muse models (Muse 2016 "MU-02", Muse 2, Muse S).
The headband must NOT be connected to a phone: it only talks to one device.

    python muse_ble.py --scan                       # list headbands in range
    python muse_ble.py --session runs/live1         # first Muse found
    python muse_ble.py --session runs/live1 --muse Muse-1E3D

Output matches osc_acquire.py: columns time,TP9,AF7,AF8,TP10[,AUX] on the
same clock as the stimulus logs, values on the libmuse scale (0..1650 uV,
resting level about 850), so the rest of the stack does not change.

Protocol (the one muselsl and Mind Monitor use): each EEG channel is its own
GATT characteristic and notifies 20-byte packets, a 16-bit packet counter
followed by twelve 12-bit samples, 256 samples per second. Commands go to the
control characteristic as [length, ascii..., newline]: a preset ("p20" = the
auxiliary electrode on, "p21" = off), then "d" to start and "k" to keep the
stream alive.
"""
import argparse
import asyncio
import os
import sys
import time

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter

from bleak import BleakClient, BleakScanner

CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
EEG_CHARS = [
    ("TP9", "273e0003-4c4d-454d-96be-f03bac821358"),
    ("AF7", "273e0004-4c4d-454d-96be-f03bac821358"),
    ("AF8", "273e0005-4c4d-454d-96be-f03bac821358"),
    ("TP10", "273e0006-4c4d-454d-96be-f03bac821358"),
    ("AUX", "273e0007-4c4d-454d-96be-f03bac821358"),   # right auxiliary input
]
FS = 256.0
PER_PACKET = 12
SCALE = 1650.0 / 4095.0          # 12-bit count -> libmuse microvolts


def cmd(text):
    body = text.encode("ascii") + b"\n"
    return bytes([len(body)]) + body


def unpack(data):
    """20 bytes -> (packet counter, twelve 12-bit samples)."""
    seq = (data[0] << 8) | data[1]
    out = []
    for k in range(0, 18, 3):
        a, b, c = data[2 + k], data[3 + k], data[4 + k]
        out.append((a << 4) | (b >> 4))
        out.append(((b & 0x0F) << 8) | c)
    return seq, out


async def find(name_filter, timeout=10.0):
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    muses = []
    for dev, adv in found.values():
        name = dev.name or adv.local_name or ""
        if "muse" not in name.lower():
            continue
        if name_filter and name_filter.lower() not in (name + " " + dev.address).lower():
            continue
        muses.append((adv.rssi, name, dev))
    muses.sort(key=lambda m: -m[0])
    return muses


class Writer:
    """Assembles the per-channel packets into rows and gives every sample a
    time. Packets arrive in bursts, so sample times come from the packet
    counter, anchored to the earliest arrivals (a packet cannot arrive
    before it was sampled) instead of from each burst's arrival time."""

    def __init__(self, path, names, append):
        self.names = names
        new = not (append and os.path.exists(path) and os.path.getsize(path) > 0)
        self.f = open(path, "a" if append else "w", newline="", buffering=1)
        if new:
            self.f.write("time," + ",".join(names) + "\n")
        self.pending = {}            # packet counter -> {channel: samples}
        self.last_seq = None
        self.t_end = None            # estimated time of the newest written sample
        self.rows = 0
        self.lost = 0

    def add(self, name, seq, samples, t_arr):
        slot = self.pending.setdefault(seq, {})
        slot[name] = samples
        if len(slot) == len(self.names):
            del self.pending[seq]
            self.emit(seq, slot, t_arr)
        if len(self.pending) > 16:   # a channel's packet was lost: let it go
            for old in sorted(self.pending)[:8]:
                del self.pending[old]
                self.lost += 1

    def emit(self, seq, slot, t_arr):
        step = PER_PACKET / FS
        if self.last_seq is None or self.t_end is None:
            self.t_end = t_arr
        else:
            gap = (seq - self.last_seq) & 0xFFFF
            if gap == 0 or gap > 2000:
                return               # duplicate, or a stale packet after a wrap
            if gap > 1:
                self.lost += gap - 1
            est = self.t_end + gap * step
            if t_arr - est > 1.0:    # the link stalled: restart the clock here
                est = t_arr
            elif est > t_arr:
                # the clock ran ahead of an arrival, which cannot be: pull it
                # back, by less than one sample so times stay increasing
                est = max(t_arr, est - 0.003)
            else:
                # lagging: packets early in a burst always look late, so
                # close only 0.2 % of the lag per packet (tracks real drift,
                # ignores burst structure)
                est += 0.002 * (t_arr - est)
            self.t_end = est
        self.last_seq = seq
        cols = [slot[n] for n in self.names]
        lines = []
        for i in range(PER_PACKET):
            t = self.t_end - (PER_PACKET - 1 - i) / FS
            lines.append(f"{t:.6f}," + ",".join(f"{c[i] * SCALE:.4f}" for c in cols))
        self.f.write("\n".join(lines) + "\n")
        self.rows += PER_PACKET

    def close(self):
        self.f.close()


async def record(args):
    eeg_path = os.path.join(args.session, "eeg.csv")
    chars = EEG_CHARS if not args.no_aux else EEG_CHARS[:4]
    names = [n for n, _ in chars]
    by_uuid = {u: n for n, u in chars}
    writer = Writer(eeg_path, names, args.append)
    t_launch = time.monotonic()

    def stop_now():
        return ((args.stop_file and os.path.exists(args.stop_file))
                or time.monotonic() - t_launch > args.max_seconds
                or (args.duration and time.monotonic() - t_launch > args.duration))

    try:
        while not stop_now():
            muses = await find(args.muse, timeout=6.0)
            if not muses:
                print("no Muse in range (is it on, and not connected to a phone?)",
                      flush=True)
                await asyncio.sleep(1.0)
                continue
            rssi, name, dev = muses[0]
            print(f"connecting to {name} [{dev.address}] rssi {rssi}", flush=True)
            gone = asyncio.Event()
            try:
                async with BleakClient(dev, disconnected_callback=lambda _c: gone.set()) as client:
                    def on_eeg(sender, data):
                        if len(data) != 20:
                            return
                        seq, samples = unpack(bytes(data))
                        writer.add(by_uuid[sender.uuid.lower()], seq, samples, local_clock())

                    for _, uuid in chars:
                        await client.start_notify(uuid, on_eeg)
                    await client.write_gatt_char(CONTROL, cmd("h"), response=False)
                    await client.write_gatt_char(
                        CONTROL, cmd("p21" if args.no_aux else args.preset), response=False)
                    await client.write_gatt_char(CONTROL, cmd("d"), response=False)
                    print(f"streaming {names} at {FS:g} Hz", flush=True)
                    last_k = last_report = time.monotonic()
                    last_rows = writer.rows
                    while not stop_now() and not gone.is_set():
                        await asyncio.sleep(0.25)
                        now = time.monotonic()
                        if now - last_k > 2.0:
                            await client.write_gatt_char(CONTROL, cmd("k"), response=False)
                            last_k = now
                        if now - last_report > 5.0:
                            rate = (writer.rows - last_rows) / (now - last_report)
                            print(f"  rows {writer.rows}  rate {rate:.0f}/s  "
                                  f"lost packets {writer.lost}", flush=True)
                            if rate < 1:           # connected but silent: ask again
                                await client.write_gatt_char(CONTROL, cmd("d"), response=False)
                            last_report, last_rows = now, writer.rows
                    if not gone.is_set():
                        await client.write_gatt_char(CONTROL, cmd("h"), response=False)
            except Exception as exc:
                print(f"Bluetooth link error: {exc!r}", file=sys.stderr, flush=True)
            if not stop_now():
                print("link lost: reconnecting", flush=True)
                writer.last_seq = writer.t_end = None
                writer.pending.clear()
                await asyncio.sleep(1.0)
    finally:
        writer.close()
        print(f"{writer.rows} EEG rows -> {eeg_path}", flush=True)


async def scan(args):
    muses = await find(args.muse, timeout=8.0)
    if not muses:
        print("no Muse found. Turn it on, and disconnect it from any phone app.")
    for rssi, name, dev in muses:
        print(f"{name}  {dev.address}  rssi {rssi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join("runs", "live1"))
    ap.add_argument("--muse", default="",
                    help="part of the headband's name or address (default: strongest)")
    ap.add_argument("--preset", default="p20",
                    help="p20 = auxiliary electrode on (default), p21 = off")
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=14400)
    # accepted so the launcher can treat every recorder alike
    ap.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.scan:
        asyncio.run(scan(args))
        return
    os.makedirs(args.session, exist_ok=True)
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
