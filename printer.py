"""The 3D printer link: pre-connect before the scan, print the moment it ends.

A Marlin printer on USB (the Creality Ender-3 Pro and most hobby printers)
takes G-code over a serial port, one line at a time: each line carries a line
number and a checksum, the printer answers "ok" when it has taken it, and
asks for a line again ("Resend: N") when it arrived damaged. This module
speaks that protocol.

The session (xr_session.py --print-port) uses it in two phases:

  pre-connect   before any scanning: find the printer, read its firmware and
                temperatures, and keep the connection open with a status poll
                every few seconds for the whole scan, so a printer that is
                switched off or unplugged is reported at once and not found
                out after half an hour of scanning.
  print         when the decoded object is sliced: the connection is handed
                to a separate print process (`printer.py print ...`), which
                streams the G-code to the end. It does not depend on the
                session, which finishes and exits as usual.

The print process writes <folder>/print_status.json while it runs (state,
progress, layer, temperatures) and stops cleanly, heaters off, when
<folder>/STOP appears (`printer.py cancel <folder>`).

    python printer.py ports                          # list serial ports, find printers
    python printer.py check --port COM5              # pre-connect test, prints firmware and temps
    python printer.py print part.gcode --port COM5   # stream a file (foreground)
    python printer.py status runs/<session>/print    # how a running print is doing
    python printer.py cancel runs/<session>/print    # stop it, heaters off

--port sim is a simulated Marlin printer in this process, for tests.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

BAUD = 115200
STATUS = "print_status.json"
STOP = "STOP"
COOL_DOWN = ["M104 S0", "M140 S0", "M107", "M84"]   # hot end off, bed off, fan off, motors off


# ---------------------------------------------------------------- transports
class SimMarlin:
    """A Marlin printer that lives in this process: answers M115 and M105,
    checks line numbers and checksums, asks for a resend now and then, heats
    up (M109 / M190) over a few simulated seconds, and keeps the moves it is
    given. `speed` scales its time (0 = instant)."""

    def __init__(self, resend_rate=0.0, speed=0.0, seed=0):
        self.rx = []
        self.lock = threading.Lock()
        self.expect = 1
        self.rng = random.Random(seed)
        self.resend_rate, self.speed = resend_rate, speed
        self.temps = dict(T=21.0, B=20.0, Tt=0.0, Bt=0.0)
        self.moves = 0
        self.taken = []
        self.is_open = True
        self._out("start")
        self._out("echo:Marlin 1.1.9 (simulated)")

    def _out(self, line):
        with self.lock:
            self.rx.append(line)

    def _temps(self):
        t = self.temps
        return f"T:{t['T']:.1f} /{t['Tt']:.1f} B:{t['B']:.1f} /{t['Bt']:.1f}"

    def write(self, data):
        line = data.decode().strip()
        m = re.match(r"N(\d+) (.*)\*(\d+)$", line)
        if m and m.group(2).startswith("M110"):
            cmd = m.group(2)
        elif m:
            n, cmd, cs = int(m.group(1)), m.group(2), int(m.group(3))
            if checksum(f"N{n} {cmd}") != cs or n != self.expect:
                self._out(f"Error:Line Number is not Last Line Number+1, Last Line: {self.expect - 1}")
                self._out(f"Resend: {self.expect}")
                self._out("ok")
                return
            if self.resend_rate and self.rng.random() < self.resend_rate:
                self._out("Error:checksum mismatch, Last Line: %d" % (self.expect - 1))
                self._out(f"Resend: {self.expect}")
                self._out("ok")
                return
            self.expect = n + 1
        else:
            cmd = line
        word = cmd.split()[0] if cmd.split() else ""
        if word == "M110":
            mm = re.search(r"N(\d+)", cmd)
            self.expect = int(mm.group(1)) + 1 if mm else 1
        elif word == "M115":
            self._out("FIRMWARE_NAME:Marlin 1.1.9 (simulated) SOURCE_CODE_URL:github.com/MarlinFirmware/Marlin "
                      "PROTOCOL_VERSION:1.0 MACHINE_TYPE:Ender-3 Pro EXTRUDER_COUNT:1")
        elif word == "M105":
            self._out("ok " + self._temps())
            return
        elif word in ("M104", "M109", "M140", "M190"):
            s = re.search(r"S(-?[\d.]+)", cmd)
            target = float(s.group(1)) if s else 0.0
            key = "T" if word in ("M104", "M109") else "B"
            self.temps[key + "t"] = target
            if word in ("M109", "M190"):
                for _ in range(3):                      # heating reports, no "ok" yet
                    self.temps[key] += (target - self.temps[key]) / 2
                    self._out(self._temps())
                    time.sleep(0.2 * self.speed)
                self.temps[key] = target
        elif word in ("G0", "G1"):
            self.moves += 1
            if self.speed:
                time.sleep(0.002 * self.speed)
        self.taken.append(cmd)
        self._out("ok")

    def readline(self):
        for _ in range(50):
            with self.lock:
                if self.rx:
                    return (self.rx.pop(0) + "\n").encode()
            time.sleep(0.001)
        return b""

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False


def open_transport(port, baud=BAUD):
    if port == "sim":
        return SimMarlin()
    if port.startswith("sim:"):                       # sim:<resend rate>:<speed>
        _, r, s = (port + "::").split(":")[:3]
        return SimMarlin(float(r or 0), float(s or 0))
    import serial
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout, ser.write_timeout = port, baud, 0.5, 5
    ser.open()
    return ser


def checksum(s):
    c = 0
    for ch in s.encode():
        c ^= ch
    return c


# ---------------------------------------------------------------- printer
class PrinterError(RuntimeError):
    pass


class Printer:
    """One Marlin printer. Thread-safe: the keep-alive poll and the caller
    share the connection through one lock."""

    def __init__(self, port, baud=BAUD, log=print):
        self.port, self.baud, self.log = port, baud, log
        self.ser = None
        self.lock = threading.RLock()
        self.line_no = 0
        self.firmware = ""
        self.temps = {}
        self.last_ok = 0.0
        self.lost = None                 # why the connection was lost, if it was
        self._poll = None
        self._stop_poll = threading.Event()

    # -- connection
    def connect(self, boot_wait=6.0):
        """Open the port and wait for the printer to answer. Opening a USB
        serial port restarts most printer boards, so the first seconds are
        the board booting."""
        self.ser = open_transport(self.port, self.baud)
        t_end = time.time() + boot_wait
        while time.time() < t_end:                      # boot chatter, if any
            line = self._read()
            if line is None:
                break
        for attempt in range(3):
            try:
                self.command("M110 N0", timeout=5)
                break
            except PrinterError:
                if attempt == 2:
                    raise PrinterError(f"{self.port}: no answer from a printer (is it on, and is "
                                       "this its port? Cura or another program may be holding it)")
        self.line_no = 0
        fw = self.command("M115", timeout=5)
        self.firmware = next((l for l in fw if "FIRMWARE_NAME" in l), " ".join(fw)).strip()
        self.read_temps()
        self.last_ok = time.time()
        return self

    def close(self):
        self.stop_keepalive()
        with self.lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    # -- one command, with line number and checksum
    def _read(self):
        raw = self.ser.readline()
        if not raw:
            return None
        return raw.decode(errors="replace").strip()

    def _send(self, cmd, numbered=True):
        if numbered and not cmd.startswith("M110"):
            self.line_no += 1
            body = f"N{self.line_no} {cmd}"
        elif cmd.startswith("M110"):
            body = f"N0 {cmd}"
        else:
            body = cmd
        self.ser.write((f"{body}*{checksum(body)}\n").encode())

    def command(self, cmd, timeout=10.0, sent=None):
        """Send one command and wait for its "ok"; returns the lines the
        printer answered with. Heating (M109 / M190) and busy reports keep
        the wait alive past `timeout`."""
        with self.lock:
            if self.ser is None:
                raise PrinterError("not connected")
            self._send(cmd)
            if sent is not None:
                sent[self.line_no] = cmd
            lines = []
            resend = None
            deadline = time.time() + timeout
            while True:
                line = self._read()
                if line is None:
                    if time.time() > deadline:
                        raise PrinterError(f"no 'ok' for {cmd!r} within {timeout:g} s")
                    continue
                if line.startswith("T:") or " T:" in line or line.startswith("echo:busy"):
                    self._parse_temps(line)
                    deadline = time.time() + max(timeout, 30)   # heating or busy: still alive
                    if not line.startswith("ok"):
                        continue
                low = line.lower()
                if low.startswith("resend:") or low.startswith("rs "):
                    resend = int(re.findall(r"\d+", line)[0])
                    continue                    # its "ok" follows; raise after it
                if "halted" in low or "kill" in low or ("error:" in low and "thermal" in low):
                    raise PrinterError("printer stopped itself: " + line)
                lines.append(line)
                if low.startswith("ok"):
                    self._parse_temps(line)
                    self.last_ok = time.time()
                    if resend is not None:
                        raise _Resend(resend)
                    return lines

    def read_temps(self):
        self.command("M105", timeout=5)
        return self.temps

    def _parse_temps(self, line):
        for key, val, tgt in re.findall(r"\b([TB]\d?):\s*(-?[\d.]+)\s*/\s*(-?[\d.]+)", line):
            self.temps[key[0]] = (float(val), float(tgt))

    def describe(self):
        t = self.temps
        temp = ", ".join(f"{'hot end' if k == 'T' else 'bed'} {v[0]:.0f} C" for k, v in sorted(t.items(), reverse=True))
        return f"{self.firmware or 'unknown firmware'} on {self.port}; {temp or 'no temperatures'}"

    # -- keep-alive during the scan
    def start_keepalive(self, every=5.0, on_lost=None):
        """Poll the temperatures every few seconds from a thread. If the
        printer stops answering, `lost` says why and on_lost(reason) runs."""
        self._stop_poll.clear()

        def loop():
            while not self._stop_poll.wait(every):
                try:
                    self.read_temps()
                except Exception as exc:
                    self.lost = repr(exc)
                    if on_lost:
                        on_lost(self.lost)
                    return
        self._poll = threading.Thread(target=loop, daemon=True)
        self._poll.start()

    def stop_keepalive(self):
        self._stop_poll.set()
        if self._poll is not None:
            self._poll.join(timeout=10)
            self._poll = None


class _Resend(Exception):
    def __init__(self, n):
        super().__init__(n)
        self.n = n


# ---------------------------------------------------------------- streaming
def gcode_lines(path):
    """The commands of a G-code file: comments and blank lines dropped,
    with the layer each command belongs to (from Cura's ;LAYER: marks)."""
    out, layer = [], -1
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            if raw.startswith(";LAYER:"):
                try:
                    layer = int(raw.split(":")[1])
                except ValueError:
                    pass
            cmd = raw.split(";", 1)[0].strip()
            if cmd:
                out.append((cmd, layer))
    return out


def gcode_info(path):
    head = open(path, encoding="utf-8", errors="replace").read(6000)
    get = lambda k: (re.search(rf"^;{k}:(.*)$", head, re.M) or [None, None])[1]
    t = get("TIME")
    return dict(seconds=float(t) if t and t.strip().replace(".", "", 1).isdigit() else None,
                layers=int(get("LAYER_COUNT")) if (get("LAYER_COUNT") or "").strip().isdigit() else None,
                filament=(get("Filament used") or "").strip() or None)


def stream(printer, path, folder, log=print, status_every=2.0):
    """Send every line of `path` and wait for each "ok"; resend on request.
    Stops cleanly (heaters off) when folder/STOP appears. Writes
    folder/print_status.json as it goes. Returns the final state."""
    lines = gcode_lines(path)
    info = gcode_info(path)
    n_layers = info["layers"] or max([l for _, l in lines] + [0]) + 1
    total = len(lines)
    stop_file = os.path.join(folder, STOP)
    t0, last = time.time(), 0.0
    sent = {}
    state = dict(state="printing", file=os.path.abspath(path), port=printer.port,
                 firmware=printer.firmware, lines=total, done=0, layer=0, layers=n_layers,
                 started=time.strftime("%Y-%m-%d %H:%M:%S"), estimate_s=info["seconds"],
                 filament=info["filament"], pid=os.getpid())

    def write_status(**kw):
        state.update(kw, elapsed_s=round(time.time() - t0), temps=printer.temps,
                     progress=round(state["done"] / max(total, 1), 4))
        tmp = os.path.join(folder, STATUS + ".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, os.path.join(folder, STATUS))

    write_status()
    log(f"printing {os.path.basename(path)}: {total} lines, {n_layers} layers"
        + (f", about {info['seconds'] / 3600:.1f} h" if info["seconds"] else ""))
    i = 0
    try:
        while i < total:
            if os.path.exists(stop_file):
                log("STOP file found: stopping the print")
                cool_down(printer, log)
                write_status(state="cancelled")
                return "cancelled"
            cmd, layer = lines[i]
            try:
                printer.command(cmd, timeout=60, sent=sent)
            except _Resend as r:
                # the printer missed line r.n (lines[i] went out as line_no):
                # go back to it and number it r.n again
                i = max(0, i - (printer.line_no - r.n))
                printer.line_no = r.n - 1
                state["resends"] = state.get("resends", 0) + 1
                continue
            i += 1
            state["done"], state["layer"] = i, max(layer, 0)
            if time.time() - last > status_every:
                last = time.time()
                write_status()
    except PrinterError as exc:
        log("PRINT FAILED: " + str(exc))
        try:
            cool_down(printer, log)
        except Exception:
            pass
        write_status(state="failed", error=str(exc))
        return "failed"
    write_status(state="done")
    log(f"print finished in {(time.time() - t0) / 60:.0f} min")
    return "done"


def cool_down(printer, log=print):
    for c in COOL_DOWN:
        try:
            printer.command(c, timeout=10)
        except Exception as exc:
            log(f"{c}: {exc}")
    try:
        t = printer.read_temps()
        log("heaters off: " + ", ".join(f"{'nozzle' if k == 'T' else 'bed'} {v[0]:.0f} C, target {v[1]:.0f}"
                                        for k, v in sorted(t.items(), reverse=True)))
    except Exception as exc:
        log(f"could not read the temperatures after turning the heaters off: {exc}")


def start_print_process(gcode, port, folder, baud=BAUD):
    """Run `printer.py print` as its own process, detached from this one, so
    the print goes on when the session that started it exits. Returns the
    process and its log file."""
    os.makedirs(folder, exist_ok=True)
    logf = os.path.join(folder, "print_log.txt")
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "print", os.path.abspath(gcode),
           "--port", port, "--baud", str(baud), "--folder", os.path.abspath(folder)]
    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    out = open(logf, "a")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True,
                         start_new_session=(sys.platform != "win32"))
    return p, logf


# ---------------------------------------------------------------- finding it
def serial_ports():
    """(device, description) of every serial port; Bluetooth links last."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    ports = [(p.device, p.description or "") for p in list_ports.comports()]
    return sorted(ports, key=lambda p: ("bluetooth" in p[1].lower(), p[0]))


def find_printer(log=print):
    """The first USB serial port with a Marlin printer behind it."""
    for dev, desc in serial_ports():
        if "bluetooth" in desc.lower():
            continue
        log(f"trying {dev} ({desc})...")
        p = Printer(dev, log=log)
        try:
            p.connect(boot_wait=4.0)
            return p
        except Exception as exc:
            log(f"  {dev}: {exc}")
            p.close()
    raise PrinterError("no printer found on any USB serial port: is it switched on and "
                       "plugged into this PC? (python printer.py ports lists the ports)")


def connect(port, log=print):
    if port == "auto":
        return find_printer(log)
    return Printer(port, log=log).connect()


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports", help="list serial ports")
    p = sub.add_parser("check", help="pre-connect test: firmware and temperatures")
    p.add_argument("--port", default="auto")
    p.add_argument("--baud", type=int, default=BAUD)
    p = sub.add_parser("print", help="stream a G-code file")
    p.add_argument("gcode")
    p.add_argument("--port", default="auto")
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--folder", default=None, help="where print_status.json goes (default: beside the file)")
    p = sub.add_parser("status", help="how a print is doing")
    p.add_argument("folder")
    p = sub.add_parser("cancel", help="stop a print, heaters off")
    p.add_argument("folder")
    a = ap.parse_args()

    if a.cmd == "ports":
        ports = serial_ports()
        for dev, desc in ports:
            print(f"{dev:8s} {desc}")
        if not ports:
            print("no serial ports")
        return 0
    if a.cmd == "check":
        p = connect(a.port)
        print("printer ready:", p.describe())
        p.close()
        return 0
    if a.cmd == "print":
        folder = a.folder or os.path.dirname(os.path.abspath(a.gcode))
        os.makedirs(folder, exist_ok=True)
        stop = os.path.join(folder, STOP)
        if os.path.exists(stop):
            os.remove(stop)
        ts = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
        try:
            p = connect(a.port, log=ts)
        except Exception as exc:
            ts("could not connect:", exc)
            with open(os.path.join(folder, STATUS), "w") as f:
                json.dump(dict(state="failed", error=f"could not connect: {exc}", port=a.port), f, indent=1)
            return 1
        ts("connected:", p.describe())
        try:
            state = stream(p, a.gcode, folder, log=ts)
        except KeyboardInterrupt:
            ts("interrupted: heaters off")
            cool_down(p, ts)
            state = "cancelled"
        p.close()
        return 0 if state == "done" else 1
    if a.cmd == "status":
        path = os.path.join(a.folder, STATUS)
        if not os.path.exists(path):
            print("no print has started in", a.folder)
            return 1
        s = json.load(open(path))
        eta = ""
        if s.get("state") == "printing" and s.get("progress"):
            left = s["elapsed_s"] / s["progress"] - s["elapsed_s"]
            eta = f", about {left / 60:.0f} min left"
        print(f"{s.get('state')}: {100 * s.get('progress', 0):.1f} % (layer {s.get('layer')} of "
              f"{s.get('layers')}){eta}; temps {s.get('temps')}; {s.get('error', '')}")
        return 0
    if a.cmd == "cancel":
        open(os.path.join(a.folder, STOP), "w").close()
        print(f"asked the print in {a.folder} to stop (it turns the heaters off)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
