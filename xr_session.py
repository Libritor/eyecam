"""Browser-driven eye-as-camera session (the Buzzkill architecture).

The stimulus renders in ANY browser — Quest Browser for VR, desktop Chrome
for testing — served by this driver. EEG arrives over OSC (Mind Monitor,
MuseLog, or phantom_subject.py). All DSP is the gated laptop stack:
run_session's Recorder/EEGTail/score_calibration and reconstruct.py, unchanged.

    python xr_session.py --session runs/xr1 --preset quick
    # headset/desktop: open  http://<PC_IP>:8082/        (?auto=1 = no input)
    # phone: Mind Monitor or MuseLog -> OSC /muse/eeg -> <PC_IP>:5000

Cursor/calib logs use the standard 5-column schema (+page_t), so
phantom_subject.py --session <same dir> closes a hardware-free loop.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import socket
import sys
import time

import math
import numpy as np
from PIL import Image

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter

import config
import reconstruct
import run_session
import targets
from run_session import EEGTail, Recorder

try:
    import websockets
    from websockets.datastructures import Headers
    from websockets.http11 import Response
except ImportError:
    raise SystemExit("pip install websockets")

ROOT = os.path.dirname(os.path.abspath(__file__))

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def csv_header(path):
    """Column names of a CSV (minus 'time'), or None if not written yet."""
    try:
        with open(path) as f:
            line = f.readline().strip()
        return line.split(",")[1:] if line.startswith("time,") else None
    except OSError:
        return None


class MultiRecorder:
    """Several OSC recorders (one per headband/port), merged into ONE eeg.csv
    on the PC arrival clock: device 0 is the time base; every other device is
    resampled onto it by nearest arrival timestamp (both streams are ~256 Hz
    on the same PC clock, so this is a <2 ms alignment). Columns: device-0
    names, then B1_<name>, B2_<name>, ... — the calibration weights then pick
    the best channels across headbands automatically."""

    def __init__(self, session, ports):
        from tailer import CsvTail
        self.session = session
        self.ports = ports
        self.recs = []
        self.tails = []
        for i, port in enumerate(ports):
            sub = os.path.join(session, f"dev{i}")
            os.makedirs(sub, exist_ok=True)
            self.recs.append(Recorder("osc", sub, port))
            self.tails.append(CsvTail(os.path.join(sub, "eeg.csv")))
        self.names = [None] * len(ports)
        self.buf = [[] for _ in ports]  # per device: list of (t, vals)
        self.out = None
        self.merged = 0

    def start(self):
        for r in self.recs:
            r.start()

    def alive(self):
        return all(r.alive() for r in self.recs)

    def stop(self):
        for r in self.recs:
            r.stop()
        if self.out:
            self.out.close()

    def stderr_tail(self):
        return " | ".join(f"udp:{p}: {r.stderr_tail()[-120:]}"
                          for p, r in zip(self.ports, self.recs))

    def poll_merge(self):
        import bisect
        for i, tail in enumerate(self.tails):
            if self.names[i] is None:
                self.names[i] = csv_header(
                    os.path.join(self.session, f"dev{i}", "eeg.csv"))
            for r in tail.poll():
                try:
                    self.buf[i].append((float(r[0]), [float(v) for v in r[1:]]))
                except (ValueError, IndexError):
                    pass
        if any(n is None for n in self.names) or any(not b for b in self.buf):
            return
        if self.out is None:
            cols = list(self.names[0])
            for i in range(1, len(self.names)):
                cols += [f"B{i}_{n}" for n in self.names[i]]
            self.out = open(os.path.join(self.session, "eeg.csv"), "w",
                            buffering=1)
            self.out.write("time," + ",".join(cols) + "\n")
            print(f"merged EEG columns: {cols}")
        # merge device-0 rows older than 0.35 s (lets the others catch up)
        cutoff = local_clock() - 0.35
        others = [(np.array([t for t, _ in b]), b) for b in self.buf[1:]]
        n_done = 0
        for t0, vals in self.buf[0]:
            if t0 > cutoff:
                break
            row = list(vals)
            for times, b in others:
                j = bisect.bisect_left(times, t0)
                cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
                k = min(cands, key=lambda k: abs(times[k] - t0)) if cands else None
                if k is not None and abs(times[k] - t0) < 0.06:
                    row += b[k][1]
                else:  # no sample from that device near t0 (dropout)
                    row += [0.0] * len(b[0][1])
            self.out.write(f"{t0:.6f}," + ",".join(f"{v:.4f}" for v in row) + "\n")
            n_done += 1
        del self.buf[0][:n_done]
        self.merged += n_done
        keep = local_clock() - 5.0  # trim other-device buffers
        for i in range(1, len(self.buf)):
            self.buf[i] = [(t, v) for t, v in self.buf[i] if t >= keep]


def channel_quality(tail):
    """Per-channel contact verdict from the last 2 s: 1=good, 0=bad.
    Catches the observed live failure mode (floating ear electrode: huge
    std, samples at the ADC rails) while accepting both raw-Muse microvolt
    scales and the phantom's unit-scale synthetic data."""
    cutoff = local_clock() - 2.0
    win = [r for r in tail.rows if r[0] >= cutoff]
    if len(win) < 32:
        return [0, 0, 0, 0]
    width = min(len(r[1]) for r in win)
    arr = np.array([r[1][:width] for r in win])
    out = []
    for c in range(arr.shape[1]):
        x = arr[:, c]
        std = float(np.std(x))
        med = float(np.median(x))
        if float(np.max(np.abs(x))) > 100:  # raw Muse scale (~0..1682 uV)
            rail = float(np.mean((x < 5) | (x > 1670)))
        else:
            rail = float(np.mean(np.abs(x - med) > 500))
        out.append(1 if (0.5 < std < 350 and rail < 0.10) else 0)
    while len(out) < 4:
        out.append(0)
    return out


def score_sweep(session, blocks, hzs, log_name="sweep_log.csv"):
    """Per frequency: exact-line SNR (f and 2f, whole-block periodogram) of
    the ON blocks at that frequency vs ALL OFF blocks scored at the same
    frequency; max-over-channels mean log-SNR difference, exact permutation.
    All segments are cut to one common length so ON and OFF are exchangeable."""
    from itertools import combinations
    from scipy.signal import periodogram
    path = os.path.join(session, "eeg.csv")
    names = csv_header(path) or CH_NAMES
    E = np.genfromtxt(path, delimiter=",", skip_header=1)
    et = E[:, 0]
    fs = reconstruct.infer_fs(et)
    if log_name and os.path.exists(os.path.join(session, log_name)):
        L = np.genfromtxt(os.path.join(session, log_name), delimiter=",",
                          skip_header=1)
        lt, lfl, lpt = L[:, 0], L[:, 4], L[:, 5]
    else:
        lt = None

    def delivered(t0, t1, fallback):
        if lt is None:
            return float(fallback)
        s = (lt >= t0) & (lt <= t1)
        fl, pt = lfl[s], lpt[s]
        ed = np.where((fl[1:] == 1) & (fl[:-1] == 0))[0] + 1
        per = np.diff(pt[ed])
        per = per[(per > 0) & (per < 1)]
        return float(1 / np.median(per)) if len(per) >= 5 else float(fallback)

    def line_snr(seg, f0, half=0.15, flank=(0.5, 2.0)):
        seg = seg - seg.mean()
        f, p = periodogram(seg, fs=fs, window="hann", detrend="constant")
        h = max(half, (f[1] - f[0]) * 1.01)   # always >= 1 bin (short blocks)
        pk = p[np.abs(f - f0) <= h].mean()
        fl_ = (np.abs(f - f0) >= flank[0]) & (np.abs(f - f0) <= flank[1])
        if not fl_.any():
            return np.nan
        return pk / max(p[fl_].mean(), 1e-12)

    segs = []
    for (kind, t0, t1), hz in zip(blocks, hzs):
        sel = (et >= t0 + 1.0) & (et <= t1)
        if sel.sum() < fs * 3:
            continue
        fdel = delivered(t0, t1, hz) if kind == "on" else None
        segs.append((kind, round(float(hz), 2) if hz else None, fdel,
                     E[sel, 1:]))
    if not segs:
        return dict(names=names, freqs=[], results={}, passed=False,
                    best_freq=None, best_channel=None, p_min=float("nan"))
    n_min = min(s[3].shape[0] for s in segs)
    segs = [(k, hz, fd, X[:n_min]) for k, hz, fd, X in segs]
    stds = np.array([[s[3][:, c].std() for c in range(len(names))] for s in segs])
    keep = stds <= 4 * np.median(stds, 0)
    freqs = sorted(set(s[1] for s in segs if s[0] == "on" and s[1]))
    per_block = [dict(kind=k, hz=hz, delivered=fd,
                      snr={names[c]: float(np.nan_to_num(line_snr(X[:, c], fd or hz or 1.0), nan=0.0))
                           for c in range(len(names))})
                 for k, hz, fd, X in segs if (hz or fd)]
    out = {}
    for f in freqs:
        rows = []
        for j, (kind, hz, fdel, X) in enumerate(segs):
            if kind == "on" and hz != f:
                continue
            f0 = fdel if kind == "on" else f
            v = np.zeros(len(names))
            for c in range(len(names)):
                if keep[j, c]:
                    s1, s2 = line_snr(X[:, c], f0), line_snr(X[:, c], 2 * f0)
                    val = np.log(max(s1, 1e-9)) + (np.log(max(s2, 1e-9))
                                                   if np.isfinite(s2) else 0.0)
                    v[c] = val if np.isfinite(val) else 0.0
            rows.append((kind == "on", v))
        on = np.array([r[0] for r in rows])
        V = np.array([r[1] for r in rows])
        if on.sum() < 2 or (~on).sum() < 2:
            continue
        diff = V[on].mean(0) - V[~on].mean(0)
        if not np.isfinite(diff).any():
            continue
        diff = np.where(np.isfinite(diff), diff, -np.inf)
        best_c = int(np.argmax(diff))
        obs = float(diff[best_c])
        n, n_on = len(on), int(on.sum())
        null = []
        if math.comb(n, n_on) <= 200000:
            for cmb in combinations(range(n), n_on):
                m = np.zeros(n, bool)
                m[list(cmb)] = True
                null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
        else:
            rng = np.random.default_rng(0)
            for _ in range(20000):
                m = np.zeros(n, bool)
                m[rng.choice(n, n_on, replace=False)] = True
                null.append(np.max(V[m].mean(0) - V[~m].mean(0)))
        null = np.array(null)
        p = float(np.mean(null >= obs - 1e-12)) if np.isfinite(obs) else 1.0
        per_ch = {names[c]: dict(on_med=float(np.exp(np.median(V[on, c]))),
                                 off_med=float(np.exp(np.median(V[~on, c]))),
                                 diff=float(diff[c]))
                  for c in range(len(names))}
        out[f"{f:g}"] = dict(
            freq=f, n_on=n_on, n_off=int((~on).sum()), best=names[best_c],
            stat=obs, p=p, n_null=len(null), channels=per_ch,
            delivered=float(np.median([s[2] for s in segs
                                       if s[0] == "on" and s[1] == f])))
    if not out:
        return dict(names=names, freqs=freqs, results={}, passed=False,
                    best_freq=None, best_channel=None, p_min=float("nan"))
    best_k = min(out, key=lambda k: out[k]["p"])
    p_min = out[best_k]["p"]
    return dict(names=names, freqs=freqs, results=out,
                best_freq=out[best_k]["freq"], best_channel=out[best_k]["best"],
                p_min=p_min, passed=bool(p_min < 0.01 / len(out)),
                segment_s=float(n_min / fs), per_block=per_block)


def score_alpha(session, blocks, band=(8.0, 12.0)):
    """Median eyes-closed vs eyes-open alpha rms per channel over blocks.
    blocks: [(kind, t0, t1)] with kind 'on' = eyes CLOSED."""
    from scipy.signal import welch
    path = os.path.join(session, "eeg.csv")
    names = csv_header(path) or CH_NAMES
    E = np.genfromtxt(path, delimiter=",", skip_header=1)
    et = E[:, 0]
    fs = reconstruct.infer_fs(et)
    out = {}
    fgrid = None
    for i, n in enumerate(names):
        vals = {"on": [], "off": []}
        psds = {"on": [], "off": []}
        for kind, t0, t1 in blocks:
            sel = (et >= t0 + 2.0) & (et <= t1)  # 2 s: blink/settle after the cue
            seg = E[sel, i + 1]
            if len(seg) < fs * 4:
                continue
            seg = seg - np.median(seg)
            mad = np.median(np.abs(seg)) * 1.4826 + 1e-9
            seg = np.clip(seg, -6 * mad, 6 * mad)
            f, p = welch(seg, fs=fs, window="hann", nperseg=int(fs * 2),
                         noverlap=int(fs))
            fgrid = f
            psds[kind].append(p)
            vals[kind].append(np.sqrt(p[(f >= band[0]) & (f <= band[1])].mean()
                                      * (band[1] - band[0])))
        c, o = (np.median(vals["on"]) if vals["on"] else np.nan,
                np.median(vals["off"]) if vals["off"] else np.nan)
        # individual alpha frequency: the 0.5 Hz bin (7.5-13 Hz) with the
        # largest closed/open POWER ratio of the median spectra; a narrow
        # alpha peak is diluted 3-4x by the broadband 8-12 Hz rms above
        Pc = np.median(psds["on"], 0) if psds["on"] else None
        Po = np.median(psds["off"], 0) if psds["off"] else None
        if Pc is not None and Po is not None:
            sel = (fgrid >= 7.5) & (fgrid <= 13.0)
            r = Pc[sel] / np.maximum(Po[sel], 1e-12)
            k = int(np.argmax(r))
            iaf = float(fgrid[sel][k])
            nb = (np.abs(fgrid - iaf) <= 0.5)
            peak_ratio = float(Pc[nb].mean() / max(Po[nb].mean(), 1e-12))
            peak_uv = float(np.sqrt(Pc[nb].sum() * 0.5))
        else:
            iaf, peak_ratio, peak_uv = float("nan"), float("nan"), float("nan")
        out[n] = dict(closed_uv=float(c), open_uv=float(o),
                      ratio=float(c / o) if o else float("nan"),
                      iaf_hz=iaf, peak_ratio=peak_ratio, peak_uv=peak_uv,
                      n_closed=len(vals["on"]), n_open=len(vals["off"]))
    aux = [n for n in names if "AUX" in n]
    aux_ok = bool(aux and out[aux[0]]["peak_ratio"] >= 3.0)
    best = max(out, key=lambda n: out[n]["peak_ratio"] if np.isfinite(out[n]["peak_ratio"]) else -1)
    verdict = ((f"AUX sees occipital alpha: {out[aux[0]]['iaf_hz']:.1f} Hz peak "
                f"x{out[aux[0]]['peak_ratio']:.1f} power with eyes closed") if aux_ok
               else (f"AUX shows no eyes-closed alpha peak (x{out[aux[0]]['peak_ratio']:.1f}) "
                     "— reseat/wet the Oz electrode" if aux
                     else f"no AUX channel; best {best}"))
    return dict(names=names, channels=out, aux_ok=aux_ok, best=best,
                verdict=verdict, band=list(band))


class Driver:
    def __init__(self, args):
        self.args = args
        self.session = os.path.join(ROOT, args.session) \
            if not os.path.isabs(args.session) else args.session
        os.makedirs(self.session, exist_ok=True)
        self.page = None            # the active subject page
        self.queue = asyncio.Queue()
        self.log_file = None
        self.blocks = []
        self.block_hz = []          # per-block flicker frequency (sweep)
        self.color_hzs = []         # page's frame-exact plan for the colour scan
        self.block_cls = []         # black/white/blue class per calibration block
        self.bwb_codes = []         # page's delivered (Hz, phase) codes
        self.frames = 0
        self.spec = None
        if args.spectator:
            host, port = args.spectator.rsplit(":", 1)
            self.spec = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                         (host, int(port)))
        self.seq = 0
        self.sid = os.path.basename(self.session)
        self.stim_freq = args.freq  # replaced by the page's delivered value
        self.auto = False           # ?auto=1 on the page skips the arm gate
        self.last_cmd = None
        self.pages = []             # every connected stimulus page (oldest first)
        self.vis = {}               # ws -> "visible" | "hidden" | None

    # ---- plumbing ----

    def spectate(self, **msg):
        if not self.spec:
            return
        self.seq += 1
        msg.update(v=1, sid=self.sid, seq=self.seq, t=local_clock())
        try:
            self.spec[0].sendto(json.dumps(msg).encode(), self.spec[1])
        except OSError:
            pass

    async def send(self, **msg):
        if msg.get("cmd") in ("start_calib", "start_scan", "start_assr",
                              "signal", "calib_result", "result",
                              "assr_result", "arm", "start_alpha",
                              "alpha_result", "start_sweep", "sweep_result",
                              "start_bwb_calib"):
            self.last_cmd = msg  # re-issued to a page that reconnects
        if self.page is not None:
            try:
                await self.page.send(json.dumps(msg))
            except Exception:
                pass

    def open_log(self, name, append=False):
        path = os.path.join(self.session, name)
        add = append and os.path.exists(path)
        self.log_file = open(path, "a" if add else "w", buffering=1)
        if not add:
            self.log_file.write(
                "time,grid_x,grid_y,luminance,flicker_on,page_t,"
                "flicker_g,flicker_b,r,g,b,audio_hz\n")

    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def handle(self, m):
        """Fast path for page messages (called from the WS receive loop)."""
        t = local_clock()
        typ = m.get("type")
        if typ == "assr_done" and getattr(self, "pc_audio", None):
            self.pc_audio.stop()
        if typ == "frame":
            if m["stage"] == "scan":
                pos = (m["gx"], m["gy"])
                if pos != getattr(self, "cur_pos", None):
                    self.cur_pos, self.cur_t0, self.cur_masked = pos, t, set()
            elif m["stage"] in ("paused", "redo", "sep"):
                self.cur_pos = None
            if self.log_file:
                self.log_file.write(
                    f"{t:.6f},{m['gx']},{m['gy']},{m['lum']:.4f},"
                    f"{m['fl']},{m['pt']:.6f},{m.get('flG', 0)},"
                    f"{m.get('flB', 0)},{m.get('r', 0):.3f},{m.get('g', 0):.3f},"
                    f"{m.get('b', 0):.3f},{m.get('hzA', 0):g}\n")
                self.frames += 1
            if m["stage"] in ("scan", "color") and self.frames % 12 == 0:
                self.spectate(type="cursor", gx=m["gx"], gy=m["gy"],
                              lum=m["lum"], fl=m["fl"])
        elif typ == "block":
            pc = getattr(self, "pc_audio", None)
            if pc and m["phase"] == "start" and abs(float(m.get("hz") or 0) - 40.0) < 0.01:
                pc.block(m["on"])      # music on this PC's headphones
            if m["phase"] == "start":
                self._block_t0 = t
            else:
                self.blocks.append(("on" if m["on"] else "off",
                                    self._block_t0, t))
                self.block_hz.append(m.get("hz"))
                self.block_cls.append(m.get("cls"))
        elif typ == "hello":
            self.auto = bool(m.get("auto"))
            print(f"page: {str(m.get('ua', ''))[:70]}  auto={self.auto}")
            self.queue.put_nowait(m)
        elif typ == "bwb_plan":
            self.bwb_codes = [dict(hz=float(c["hz"]), phaseDeg=float(c["phaseDeg"]))
                              for c in m.get("codes", [])]
            print("colour codes delivered:",
                  ", ".join(f"{c['hz']:.2f} Hz @ {c['phaseDeg']:.0f} deg" for c in self.bwb_codes),
                  "@", m.get("refresh"), "fps")
            seen = [(round(c["hz"], 2), round(c["phaseDeg"]) % 360) for c in self.bwb_codes]
            if len(set(seen)) < len(seen):
                print("WARNING: two colours landed on the same code at this refresh rate; "
                      "they cannot be told apart (pass --bwb-codes for this panel)")
        elif typ == "color_plan":
            self.color_hzs = [float(v) for v in m.get("hzs", [])]
            print("colour flicker plan:", self.color_hzs, "@", m.get("refresh"), "fps")
        elif typ == "visibility":
            print(f"page visibility: {m.get('state')}")
            self.spectate(type="log", msg=f"page {m.get('state')}")
        elif typ == "freq":
            # the page reports the frame-exact frequency it actually delivers
            self.stim_freq = float(m["hz"])
            print(f"page flicker: {self.stim_freq:.2f} Hz "
                  f"(refresh {m.get('refresh')} fps, "
                  f"{m.get('halfFrames')} frames on/off)")
        else:
            self.queue.put_nowait(m)

    # ---- which tab is the subject page? ----
    # Quest Browser pauses background tabs (rAF ~1 fps), so only a VISIBLE
    # page may drive a stage. The newest visible tab wins between stages;
    # a running stage keeps its page unless that page goes hidden.
    def mid_run(self):
        return bool(self.last_cmd) and self.last_cmd.get("cmd") in (
            "start_calib", "start_scan", "start_assr", "start_alpha",
            "start_sweep", "start_bwb_calib")

    async def adopt(self, ws):
        old = self.page
        self.page = ws
        print(f"active page: #{self.pages.index(ws) + 1} of {len(self.pages)}")
        if old is not None and old is not ws:
            try:
                await old.send(json.dumps(dict(
                    cmd="msg", text="another tab took over — close this one")))
            except Exception:
                pass
        if self.last_cmd is not None:
            if self.mid_run():
                self.blocks = []  # the stage restarts on the new page
                self.block_hz, self.block_cls = [], []
            try:
                await ws.send(json.dumps(self.last_cmd))
            except Exception:
                pass

    async def consider(self, ws):
        """Called on a page's hello and every visibility change."""
        visible = self.vis.get(ws) == "visible"
        if self.page is ws:
            if not visible:
                print("active page went hidden")
                for other in reversed(self.pages):
                    if other is not ws and self.vis.get(other) == "visible":
                        await self.adopt(other)
                        break
            return
        if not visible:
            print(f"page #{self.pages.index(ws) + 1} is hidden — ignored "
                  "until it becomes visible")
            return
        if self.page is None or self.vis.get(self.page) != "visible"                 or not self.mid_run():
            await self.adopt(ws)
        else:
            print("extra visible page while a stage is running — ignored")
            try:
                await ws.send(json.dumps(dict(
                    cmd="msg", text="another tab is running the session "
                    "— close this one")))
            except Exception:
                pass

    paused = False
    scan_active = False

    async def wait_for(self, typ, timeout=600):
        self.scan_active = (typ == "scan_done")
        self.eeg_lost = False
        while True:
            try:
                m = await asyncio.wait_for(self.queue.get(),
                                           5.0 if self.paused else timeout)
            except asyncio.TimeoutError:
                if self.paused:
                    continue  # a paused scan may wait as long as it takes
                raise
            if m.get("type") == typ:
                self.scan_active = False
                return m
            if getattr(self, "eeg_lost", False) and typ != "go":
                await self.send(cmd="msg", text="EEG stream lost - session "
                                                "stopped. Check the phone.")
                raise RuntimeError("EEG stream lost during " + typ)

    # ---- session flow ----

    async def run(self):
        a = self.args
        target = targets.load_target(a.target, a.grid_w, a.grid_h)
        np.save(os.path.join(self.session, "target.npy"), target)

        merge_task = None
        ports = [int(p) for p in str(a.osc_port).split(",") if p.strip()]
        if a.muse or a.lsl:
            # Muse on this PC's Bluetooth (muselsl) or any LSL EEG stream:
            # recorded straight from LSL, no phone app and no OSC
            ports = []
            recorder = Recorder("lsl", self.session, 0,
                                extra=["--no-aux"] if a.no_aux else [])
        elif len(ports) == 1:
            recorder = Recorder("osc", self.session, ports[0])
        else:
            recorder = MultiRecorder(self.session, ports)
        eeg_src = "LSL" if (a.muse or a.lsl) else f"udp:{a.osc_port}"
        recorder.start()
        await asyncio.sleep(1.5)
        if not recorder.alive():
            err = recorder.stderr_tail()
            if "10048" in err:  # port held by an orphaned recorder
                print("udp port busy — retrying once in 3 s")
                await asyncio.sleep(3)
                recorder.start()
                await asyncio.sleep(1.5)
            if not recorder.alive():
                raise RuntimeError(
                    f"EEG recorder failed on {eeg_src}: "
                    + recorder.stderr_tail()[-200:])
        merged_path = os.path.join(self.session, "eeg.csv")
        tail = EEGTail(merged_path)
        osc_target = ", ".join(f"{local_ip()}:{p}" for p in ports) or             ("the Muse over this PC's Bluetooth" if a.muse else "an LSL EEG stream")
        merge_task = None
        if len(ports) > 1:
            async def merge_loop():
                while True:
                    recorder.poll_merge()
                    await asyncio.sleep(0.25)
            merge_task = asyncio.create_task(merge_loop())

        def names():
            return csv_header(merged_path) or CH_NAMES

        def ears_ok(q):
            n = names()
            return any(q[i] for i in range(min(len(q), len(n)))
                       if "TP9" in n[i] or "TP10" in n[i])
        result = dict(mode="browser", gridW=a.grid_w, gridH=a.grid_h,
                      secondsPerCell=a.spc, calibBlocks=a.calib_blocks)
        hb_task = bw_task = None
        try:
            print("waiting for a browser to open the stimulus page...")
            wait_deadline = time.monotonic() + a.signal_timeout
            while self.page is None:
                if time.monotonic() > wait_deadline:
                    raise TimeoutError("no stimulus page connected")
                await asyncio.sleep(0.2)
            self.spectate(type="hello", mode="browser", ch=4, fs=256.0,
                          gridW=a.grid_w, gridH=a.grid_h, freq=a.freq,
                          refresh=-1)

            # persistent Muse status badge: rate + per-electrode contact,
            # pushed to the page once a second through EVERY stage
            async def heartbeat():
                while True:
                    tail.poll()
                    await self.send(cmd="eeg", rate=tail.rate(),
                                    q=channel_quality(tail), names=names())
                    self.spectate(type="eeg", rate=tail.rate(), rms=[])
                    # a scan with no EEG behind it is wasted subject time:
                    # stop the stage once the stream has been silent 10 s
                    # scans PAUSE on a silent stream (3 s) and resume when
                    # it is back; the interrupted position is redone. Block
                    # stages cannot be redone piecemeal: they stop at 10 s.
                    if tail.rate() > 100:
                        self.eeg_good = getattr(self, "eeg_good", 0) + 1
                        self.eeg_silent = 0
                        if self.paused and self.eeg_good >= 3:
                            self.paused = False
                            print("EEG stream back: resuming scan")
                            await self.send(cmd="resume")
                    elif tail.rate() < 1:
                        self.eeg_good = 0
                        self.eeg_silent = getattr(self, "eeg_silent", 0) + 1
                        if (self.scan_active and not self.paused
                                and self.eeg_silent >= 3):
                            self.paused = True
                            print("EEG STREAM SILENT: scan paused")
                            await self.send(cmd="pause")
                        elif not self.scan_active and self.eeg_silent == 10:
                            print("EEG STREAM LOST (10 s without samples)")
                            self.eeg_lost = True
                    await asyncio.sleep(1.0)
            hb_task = asyncio.create_task(heartbeat())

            # --mask-blinks: a position whose dwell is more than --mask-limit
            # blinks / closed eyes is redone, through the same path as a
            # stream drop (the page abandons it and shows it again)
            async def blink_watch():
                import blinkmask
                errors = set()
                while True:
                    await asyncio.sleep(0.25)
                    try:
                        await blink_step(blinkmask)
                    except Exception as exc:          # never silent, never fatal
                        if repr(exc) not in errors:
                            errors.add(repr(exc))
                            print("blink watch:", repr(exc))

            async def blink_step(blinkmask):
                pos = getattr(self, "cur_pos", None)
                if not (self.scan_active and pos and not self.paused):
                    return
                tail.poll()
                rows = [r for r in tail.rows if r[0] >= local_clock() - 4.0]
                width = min((len(r[1]) for r in rows), default=0)
                if len(rows) < 256 or width < 4:
                    return
                tt = np.array([r[0] for r in rows])
                X = np.array([r[1][:width] for r in rows])
                ok, _ = blinkmask.compute(tt, X, names()[:width], 256.0, self.stim_freq,
                                          self.mask_opts, alpha_base=self.alpha_base)
                for ts in tt[(~ok) & (tt >= self.cur_t0)]:
                    self.cur_masked.add(round(float(ts), 4))
                if len(self.cur_masked) / 256.0 > self.mask_opts["limit"] * a.spc:
                    self.cur_pos = None
                    redos = self.redo_count.get(pos, 0)
                    if redos >= a.max_redo:   # never loop forever on one position
                        print(f"position {pos}: still over the limit after {redos} "
                              "redo(s): keeping it (it is dropped from scoring)")
                        return
                    self.redo_count[pos] = redos + 1
                    print(f"position {pos}: {len(self.cur_masked) / 256.0:.1f} s of "
                          f"blinks / closed eyes > {self.mask_opts['limit']:.0%} of the "
                          f"dwell: redoing it ({redos + 1}/{a.max_redo})")
                    await self.send(cmd="redo")
            if a.mask_blinks:
                import blinkmask
                self.mask_opts = blinkmask.options(a)
                self.alpha_base = None
                self.redo_count = {}
                bw_task = asyncio.create_task(blink_watch())

            # signal check: rate stable AND at least one EAR electrode good
            # (TP9/TP10 carry the SSVEP; a session without them is doomed)
            self.spectate(type="stage", stage="signal_check", detail="")
            stable_since = None
            deadline = time.monotonic() + a.signal_timeout
            while True:
                tail.poll()
                rate = tail.rate()
                q = channel_quality(tail)
                ears = ears_ok(q)
                stable = rate > 40
                nowm = time.monotonic()
                if stable and ears and stable_since is None:
                    stable_since = nowm
                if not (stable and ears):
                    stable_since = None
                ready = stable_since and nowm - stable_since >= 3
                if not recorder.alive():
                    raise RuntimeError("EEG recorder died: "
                                       + recorder.stderr_tail()[-200:])
                await self.send(cmd="signal", rate=rate, stable=stable,
                                earsOk=ears, ready=bool(ready),
                                oscTarget=osc_target)
                if ready:
                    break
                if nowm > deadline:
                    raise TimeoutError(
                        f"no usable EEG on {eeg_src} "
                        f"(rate {rate:.0f} Hz, ears_ok={ears_ok}; "
                        f"source: {osc_target})")
                await asyncio.sleep(0.25)
            print(f"signal ok ({rate:.0f} Hz, ears good)")

            # arm gate: the calibration must not start before the headset
            # is on and the subject is looking at the dot. Without ?auto=1
            # the page waits for a controller trigger / click, counts down
            # 3 s and replies "go".
            if not self.auto:
                await self.send(cmd="arm")
                print("armed: put the headset on and pull the trigger to start")
                self.spectate(type="stage", stage="armed", detail="")
                await self.wait_for("go", timeout=a.signal_timeout)
                print("go")

            if a.mode == "assr":
                return await self.run_assr(result)
            if a.mode == "alpha":
                return await self.run_alpha(result, recorder)
            if a.mode == "sweep":
                return await self.run_sweep(result, recorder)
            if a.mode == "mux":
                return await self.run_mux(result, recorder)
            if a.mode == "bwb":
                return await self.run_bwb(result, recorder)
            if a.mode == "music":
                self.blocks, self.block_hz = [], []
                self.open_log("assr_log.csv")
                await self.send(cmd="start_assr", music=True, blocks=a.music_blocks,
                                onS=a.music_on, offS=a.music_off, modFreq=40.0,
                                mute=bool(getattr(self, "pc_audio", None)))
                await self.wait_for("assr_done",
                                    timeout=a.music_blocks * (a.music_on + a.music_off) + 120)
                self.close_log()
                await asyncio.sleep(0.7)
                mus = score_sweep(self.session, self.blocks,
                                  [40.0] * len(self.blocks), log_name=None)
                mres = next(iter(mus["results"].values()), None)
                with open(os.path.join(self.session, "music.json"), "w") as f:
                    json.dump(mus, f, indent=1)
                mtext = (f"40 Hz tag: p={mres['p']:.4f} best {mres['best']} "
                         f"(ON {mres['channels'][mres['best']]['on_med']:.2f} / "
                         f"OFF {mres['channels'][mres['best']]['off_med']:.2f})"
                         if mres else "no scorable blocks")
                print("music " + mtext)
                await self.send(cmd="assr_result", passed=bool(mres and mres["p"] < config.CALIB_P_MAX), table=mtext)
                result.update(ok=True, music=mtext)
                recorder.stop()
                print(f"MUSIC DONE -> {self.session}")
                await asyncio.sleep(a.linger)
                return 0
            if a.mode == "extras":
                # colour + music only, reusing a copied calibration.json
                await self.run_full_extras(result)
                result["ok"] = True
                recorder.stop()
                print(f"EXTRAS DONE -> {self.session}")
                await asyncio.sleep(a.linger)
                return 0

            # calibration (page-driven)
            self.spectate(type="stage", stage="calibrate", detail="")
            self.blocks = []
            self.open_log("calib_log.csv")
            colors = ["#ffffff", "#000000"] if a.calib_style == "bw"                 else ["#ffff00", "#0000ff"]
            result.update(calibStyle=a.calib_style, calibSize=a.calib_size)
            await self.send(cmd="start_calib", blocks=a.calib_blocks,
                            onS=a.calib_on, offS=a.calib_off, freq=a.freq,
                            colors=colors, size=a.calib_size)
            await self.wait_for(
                "calib_done",
                timeout=a.calib_blocks * (a.calib_on + a.calib_off + 2) + 60)
            self.close_log()
            await asyncio.sleep(0.7)
            # G0: was the stimulus actually delivered? A background/hidden
            # browser tab throttles rAF to ~1 fps: no flicker edges, a few
            # rows per block. Such a session is VOID, not a null result.
            g0 = self.delivery_check("calib_log.csv",
                                     a.calib_blocks * (a.calib_on + a.calib_off))
            result["g0"] = g0
            if not g0["ok"]:
                msg = ("STIMULUS NOT DELIVERED: " + g0["reason"] +
                       " — keep the page visible & in the foreground")
                print(msg)
                await self.send(cmd="msg", text=msg)
                result.update(ok=False, error="G0 " + g0["reason"])
                await asyncio.sleep(a.linger)
                return 2
            calib = run_session.score_calibration(
                self.session, self.blocks, stim_freq=self.stim_freq)
            result["stimFreqActual"] = self.stim_freq
            # exact-line detector on the same blocks: the validated G1 test
            # (sweep2: 5 uV line at 10-12 Hz invisible to the 1 Hz-bin score)
            line = score_sweep(self.session, self.blocks,
                               [self.stim_freq] * len(self.blocks),
                               log_name="calib_log.csv")
            lres = next(iter(line["results"].values()), None)
            if lres:
                wl = {n: max(cc["diff"], 0.0) for n, cc in lres["channels"].items()}
                tot = sum(wl.values())
                calib["weights_welch"] = dict(calib["weights"])
                if tot > 0:
                    calib["weights"] = {n: v / tot for n, v in wl.items()}
                calib["line"] = dict(p=lres["p"], best=lres["best"],
                                     stat=lres["stat"], delivered=lres["delivered"],
                                     channels=lres["channels"],
                                     per_block=line.get("per_block"))
                calib["passed_welch"] = calib["passed"]
                calib["passed"] = bool(lres["p"] < config.CALIB_P_MAX)
                calib["gate"] = "line-permutation"
                calib["best"] = lres["best"]
                with open(os.path.join(self.session, "calibration.json"), "w") as f:
                    json.dump(calib, f, indent=1)
            perm = calib.get("permutation", {})
            pch = perm.get("p_channel") or [float("nan")] * len(calib["names"])
            table = "   ".join(
                f"{n} d'={calib['channels'][n]['dprime']:.1f}"
                f" p={pch[i]:.3f} w={calib['weights'][n]:.2f}"
                for i, n in enumerate(calib["names"]))
            table += (f"   |  welch p_fw={perm.get('p_fw', float('nan')):.4f}"
                      f" (n_null={perm.get('n_null', 0)})")
            if calib.get("line"):
                ln = calib["line"]
                table = (f"LINE {ln['delivered']:.2f} Hz p={ln['p']:.4f} "
                         f"best {ln['best']} stat {ln['stat']:+.2f}  ||  " + table)
            print(f"calibration passed={calib['passed']} {table}")
            result.update(calibPassed=calib["passed"], best=calib["best"],
                          weights=calib["weights"])
            await self.send(cmd="calib_result", passed=calib["passed"],
                            table=table)
            self.spectate(type="calib", detail=table, passed=calib["passed"],
                          best=calib["best"],
                          weights=list(calib["weights"].values()))
            await asyncio.sleep(2.5)
            if a.mode == "calib":
                # G1-only session: the question is whether an SSVEP exists.
                result.update(ok=True, gate=calib.get("gate"),
                              p_fw=calib.get("permutation", {}).get("p_fw"))
                recorder.stop()
                print(f"G1 SESSION DONE passed={calib['passed']} "
                      f"-> {self.session}")
                await asyncio.sleep(a.linger)
                return 0
            if not calib["passed"] and a.require_pass:
                raise RuntimeError("calibration gate failed (no SSVEP)")

            if a.mask_blinks:
                import blinkmask
                try:
                    hdr, rows_ = reconstruct.read_csv(os.path.join(self.session, "eeg.csv"))
                    tt_ = np.array([float(x[0]) for x in rows_])
                    X_ = np.array([[float(v) for v in x[1:]] for x in rows_])
                    self.alpha_base = blinkmask.alpha_baseline(
                        tt_, X_, hdr[1:], 256.0, self.stim_freq,
                        [(t0, t1) for k, t0, t1 in calib.get("blocks", []) if k == "off"])
                    print("blink mask: eyes-open alpha baseline",
                          "from the calibration OFF blocks" if self.alpha_base
                          else "unavailable (no Oz channel): eye-closure detection off")
                except Exception as exc:
                    print("blink mask baseline:", repr(exc))

            # which picture: black & white "NO" (the original scan) or the
            # black / white / blue frequency-phase scan - asked on the page
            choice = a.choice
            if choice is None and a.mode == "visual":
                if self.auto:
                    choice = "bw"
                else:
                    note = (f"calibration found the {self.stim_freq:.1f} Hz response "
                            f"(best {calib['best']})" if calib["passed"] else
                            "calibration did NOT find a clear response - expect noise")
                    print("choose the picture on the stimulus page (button or key 1 / 2)")
                    await self.send(cmd="choose", passed=bool(calib["passed"]), note=note,
                                    options=[dict(value="bw", label="Black & white",
                                                  detail=f"the original scan: {a.target}, "
                                                         f"{a.passes} pass(es), {a.spc:g} s per position"),
                                             dict(value="bwb", label="Colour",
                                                  detail=f"{a.bwb_colours} by frequency-phase codes; "
                                                         f"{a.bwb_target}, {a.bwb_passes} pass(es)")])
                    m = await self.wait_for("choice", timeout=a.signal_timeout)
                    choice = m.get("value", "bw")
            result["choice"] = choice or "bw"
            print(f"picture: {result['choice']}")
            if choice == "bwb":
                return await self.run_bwb(result, recorder)

            # scan (page-driven serpentine)
            self.spectate(type="stage", stage="scan", detail="")
            scan_kw = dict(gridW=a.grid_w, gridH=a.grid_h, spc=a.spc, freq=a.freq,
                           target=target.tolist(), patch=a.patch,
                           bw=(a.calib_style == "bw"), board=a.board)
            bw_cert = None
            if a.certainty > 0:
                import adaptive
                cal_w = reconstruct.weights_from_calibration(
                    os.path.join(self.session, "calibration.json"), names())
                mk_live = {}
                if a.mask_blinks:
                    import blinkmask
                    mk_live = dict(mask_blinks=True, mask_opts=blinkmask.options(a))
                P = a.grid_w * a.grid_h

                def bw_state():
                    call, cert, n, repeats = adaptive.bw_state(
                        self.session, cal_w, self.stim_freq, **mk_live)
                    bw_state.call = call
                    # no stop before some squares have a repeat visit (adaptive.py)
                    return cert, n, repeats >= min(a.certainty_batch, P)
                print(f"scan: one pass, then the least certain squares until every "
                      f"square is {a.certainty:.0%} certain (--passes is not used)")
                timeline, cert, n = await self.certainty_scan(
                    scan_kw, "cursor_log.csv", a.grid_w, a.grid_h, a.spc, bw_state)
                tb = np.asarray(target) > 0.5
                right = int((bw_state.call == tb).sum())
                bw_cert = dict(right=right, n=int(tb.size), min_certainty=float(cert.min()),
                               minutes=timeline[-1]["minutes"], visits=int(n.sum()))
                result["certainty"] = dict(bw_cert, timeline=timeline)
                print(f"certainty call: {right} of {tb.size} squares right, lowest "
                      f"certainty {cert.min():.1%}, {n.sum() * a.spc / 60:.1f} min of scanning")
            else:
                self.open_log("cursor_log.csv")
                await self.send(cmd="start_scan", passes=a.passes, **scan_kw)
                scan_s = a.grid_w * a.grid_h * a.spc * max(1, a.passes)
                await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
                self.close_log()
            await asyncio.sleep(0.7)

            # reconstruct: the chosen method (default: the paper exactly -
            # 1700-sample window, (f + 2f) / 14-50 Hz, Eq. 1), plus the two
            # comparisons, so every run says what each choice costs
            print("reconstructing...")
            cal_path = os.path.join(self.session, "calibration.json")
            vblend = not a.no_eq1
            mk = {}
            if a.mask_blinks:
                import blinkmask
                mk = dict(mask_blinks=True, mask_opts=blinkmask.options(a))
            grid, r = reconstruct.run(self.session, calibration=cal_path,
                                      stim_freq=self.stim_freq, method=a.recon_method,
                                      vblend=vblend, **mk)
            compare = {}
            for label, kw in (("paper, no Eq. 1", dict(method="paper", vblend=False)),
                              ("upstream repo method", dict(method="upstream"))):
                try:
                    compare[label] = reconstruct.run(self.session, calibration=cal_path,
                                                     stim_freq=self.stim_freq, save=False,
                                                     **kw, **mk)[1]
                except Exception as exc:
                    compare[label] = None
                    print(f"{label}: {exc!r}")
            flicker_hz = self.measured_flicker()
            rs_null = reconstruct.null_r(
                self.session, n_shifts=8, calibration=cal_path,
                stim_freq=self.stim_freq, method=a.recon_method, vblend=vblend, **mk)
            import nofigure
            fig = nofigure.figure(
                grid, target, os.path.join(self.session, "reconstruction_figure.png"),
                title=(f"{a.target}: {self.stim_freq:.1f} Hz, {a.spc:g} s per position, "
                       f"{a.passes} pass(es); method {a.recon_method}"
                       + (" + Eq. 1" if vblend and a.recon_method == "paper" else "")),
                subtitle=("other methods on the same EEG: " + ", ".join(
                    f"{k} r = {v:.2f}" for k, v in compare.items() if v is not None)
                    + (f";  time-shifted EEG (chance) up to r = {max(rs_null):.2f}"
                       if rs_null else "")))
            result.update(r=r, measuredFlickerHz=flicker_hz, method=a.recon_method,
                          eq1=vblend, r_compare=compare, threshold_right=fig["right"],
                          mask_blinks=bool(a.mask_blinks),
                          threshold_n=fig["n"], r_null=rs_null,
                          r_null_max=(max(rs_null) if rs_null else None))
            print(f"grey image r={r:.3f} ({a.recon_method}{', Eq. 1' if vblend else ''}); "
                  + ", ".join(f"{k} r={v:.3f}" for k, v in compare.items() if v is not None)
                  + f"; threshold right {fig['right']}/{fig['n']}; shifted-EEG null "
                  f"r: {', '.join(f'{v:.2f}' for v in rs_null)}")
            with open(os.path.join(self.session, "reconstruction_figure.png"), "rb") as f:
                png = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            text = (f"r = {r:.2f}  -  automatic threshold: "
                    f"{fig['right']} of {fig['n']} positions right")
            if bw_cert:
                text = (f"{bw_cert['right']} of {bw_cert['n']} squares right at "
                        f"{a.certainty:.0%} certainty ({bw_cert['minutes']:.1f} min)  -  " + text)
            await self.send(cmd="result", png=png, r=r or 0.0, flickerHz=flicker_hz,
                            wide=True, text=text)
            flat = [float(v) for v in np.asarray(grid).ravel()]
            self.spectate(type="grid", w=a.grid_w, h=a.grid_h, scores=flat)
            self.spectate(type="stage", stage="done",
                          detail=f"r={r:.3f}" if r else "")
            if a.mode == "full":
                await asyncio.sleep(4.0)  # let the subject see the image
                await self.run_full_extras(result)
            result["ok"] = True
            recorder.stop()  # stop at scan end, not after the linger
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")
            await asyncio.sleep(a.linger)
            return 0
        except Exception as e:
            result.update(ok=False, error=repr(e))
            print(f"SESSION FAILED: {e!r}")
            await self.send(cmd="msg", text=f"session failed: {e}")
            return 1
        finally:
            if hb_task:
                hb_task.cancel()
            if bw_task:
                bw_task.cancel()
            if merge_task:
                merge_task.cancel()
                recorder.poll_merge()  # flush what is already buffered
            with open(os.path.join(self.session, "xr_session.json"),
                      "w") as f:
                json.dump(result, f, indent=1)
            recorder.stop()

    async def certainty_scan(self, scan_kw, log_name, grid_w, grid_h, spc, state):
        """One full pass, then the least certain squares again, --certainty-batch
        at a time, until every square is at least --certainty sure (adaptive.py)
        or each has --certainty-max-passes visits. state() -> (certainty grid,
        visits grid, may_stop). Returns the per-batch timeline."""
        import adaptive
        a = self.args
        P = grid_w * grid_h
        self.open_log(log_name)
        await self.send(cmd="start_scan", passes=1, **scan_kw)
        await self.wait_for("scan_done", timeout=P * spc * 2 + 120)
        self.close_log()
        timeline = []
        while True:
            await asyncio.sleep(0.7)                  # the last visit's EEG lands
            cert, n, may_stop = await asyncio.to_thread(state)
            low = int((cert < a.certainty).sum())
            timeline.append(dict(visits=int(n.sum()), minutes=round(n.sum() * spc / 60, 2),
                                 min_certainty=round(float(cert.min()), 4), below=low))
            print(f"certainty after {int(n.sum())} visits ({n.sum() * spc / 60:.1f} min): "
                  f"lowest {cert.min():.1%}, {low} of {P} squares below {a.certainty:.0%}")
            if may_stop and low == 0:
                print("every square is certain enough: scan done")
                break
            cells = adaptive.pick(cert, n, a.certainty_batch, a.certainty_max_passes)
            if not cells:
                print(f"every square has {a.certainty_max_passes} visits: stopping "
                      f"with {low} squares below {a.certainty:.0%}")
                break
            self.open_log(log_name, append=True)
            await self.send(cmd="start_scan", cells=cells, spc=spc,
                            label=f"checking {len(cells)} uncertain squares "
                                  f"({low} below {a.certainty:.0%})",
                            **{k: v for k, v in scan_kw.items() if k != "spc"})
            await self.wait_for("scan_done", timeout=len(cells) * spc * 2 + 60)
            self.close_log()
        with open(os.path.join(self.session, "certainty.json"), "w") as f:
            json.dump(dict(certainty=a.certainty, batch=a.certainty_batch,
                           max_passes=a.certainty_max_passes, timeline=timeline,
                           final=np.asarray(cert).round(4).tolist(),
                           visits=np.asarray(n).tolist()), f, indent=1)
        return timeline, cert, n

    def measured_cols(self, log_name, cols=(4, 6, 7)):
        """Delivered frequency per flicker column from page-clock rising edges."""
        path = os.path.join(self.session, log_name)
        out = []
        try:
            L = np.genfromtxt(path, delimiter=",", skip_header=1)
        except OSError:
            return [float("nan")] * len(cols)
        if L.ndim == 1 or L.shape[0] < 10:
            return [float("nan")] * len(cols)
        for c in cols:
            v = L[:, c]
            ed = np.where((v[1:] == 1) & (v[:-1] == 0))[0] + 1
            per = np.diff(L[ed, 5])
            per = per[(per > 0.01) & (per < 1.0)]
            out.append(float(1 / np.median(per)) if len(per) >= 5 else float("nan"))
        return out

    async def run_mux(self, result, recorder):
        """White 3-tag calibration, then the multiplexed grey scan."""
        a = self.args
        freqs = [float(v) for v in str(a.color_freqs).split(",") if v.strip()]
        n = len(freqs)
        gw, gh = a.grid_w - (a.grid_w % n), a.grid_h
        target = targets.load_target(a.target, gw, gh)
        np.save(os.path.join(self.session, "target.npy"), target)
        names = csv_header(os.path.join(self.session, "eeg.csv")) or CH_NAMES
        # calibration: each tag frequency, full-panel white/black
        self.spectate(type="stage", stage="mux_calib", detail="")
        self.blocks, self.block_hz = [], []
        self.open_log("calib_log.csv")
        cols = {f"{f:g}": ["#ffffff", "#000000"] for f in freqs}
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.calib_blocks,
                        onS=a.calib_on, offS=a.calib_on, colorsPerFreq=cols, size=1.0)
        total = a.calib_blocks * n * 2 * a.calib_on
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        cal = score_sweep(self.session, self.blocks, self.block_hz, log_name="calib_log.csv")
        ordered = sorted(cal["results"].values(), key=lambda rr: rr["freq"])
        # channel weights: mean over tags of the per-channel ON-OFF log diff
        diff = np.zeros(len(names))
        for rr in ordered:
            diff += np.array([max(rr["channels"][nm]["diff"], 0.0) for nm in names])
        weights = diff / diff.sum() if diff.sum() > 0 else np.ones(len(names)) / len(names)
        cal["weights"] = {nm: float(v) for nm, v in zip(names, weights)}
        cal["names"] = names
        cal["passed"] = bool(min(rr["p"] for rr in ordered) < config.CALIB_P_MAX)
        with open(os.path.join(self.session, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=1)
        ctab = "  ".join(f"{rr['freq']:g}Hz p={rr['p']:.3f} {rr['best']}" for rr in ordered)
        print(f"mux calibration passed={cal['passed']}: {ctab}  weights="
              + " ".join(f"{nm}:{v:.2f}" for nm, v in zip(names, weights)))
        result["calib"] = dict(table=ctab, passed=cal["passed"], weights=cal["weights"])
        await self.send(cmd="calib_result", passed=cal["passed"], table=ctab)
        await asyncio.sleep(2.5)
        # scan
        self.spectate(type="stage", stage="mux_scan", detail="")
        self.color_hzs = []
        self.open_log("mux_log.csv")
        await self.send(cmd="start_scan", mux=True, gridW=gw, gridH=gh, spc=a.spc,
                        freqs=freqs, target=target.tolist(), patch=a.patch)
        scan_s = (gw // n) * gh * a.spc
        await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        g0 = self.delivery_check("mux_log.csv", scan_s)
        result["g0"] = g0
        hz_del = self.measured_cols("mux_log.csv")
        for k in range(n):
            if not np.isfinite(hz_del[k]):
                hz_del[k] = self.color_hzs[k] if k < len(self.color_hzs) else freqs[k]
        print("mux scan delivered:", ["%.2f" % v for v in hz_del], "| G0", g0)
        grid, r = reconstruct.reconstruct_mux(self.session, hz_del, weights=weights,
                                              cursor_log="mux_log.csv", target=target,
                                              stride=gw // n)
        # shifted-EEG null
        eeg_t = np.genfromtxt(os.path.join(self.session, "eeg.csv"), delimiter=",",
                              skip_header=1, usecols=[0])
        fs = reconstruct.infer_fs(eeg_t)
        orig = reconstruct.load_session
        rs_null = []
        for k in range(8):
            shift = 31.0 + k * 29.0
            if shift > eeg_t[-1] - eeg_t[0] - 10:
                break
            def fake(session_dir, channels="", cursor_log="cursor_log.csv", _s=shift):
                e, d, nm, c, xx, yy = orig(session_dir, channels, cursor_log)
                return e, np.roll(d, int(round(_s * fs)), axis=0), nm, c, xx, yy
            reconstruct.load_session = fake
            try:
                _, rn = reconstruct.reconstruct_mux(
                    self.session, hz_del, weights=weights, cursor_log="mux_log.csv",
                    target=target, out=os.path.join(self.session, "_null.png"),
                    stride=gw // n)
                if rn is not None:
                    rs_null.append(rn)
            finally:
                reconstruct.load_session = orig
        # restore the real reconstruction files (the null loop overwrote the grid)
        grid, r = reconstruct.reconstruct_mux(self.session, hz_del, weights=weights,
                                              cursor_log="mux_log.csv", target=target,
                                              stride=gw // n)
        print(f"mux image r={r:.3f}; shifted-EEG null r: "
              + ", ".join(f"{v:.2f}" for v in rs_null))
        result.update(ok=True, r=r, r_null=rs_null, r_null_max=max(rs_null) if rs_null else None,
                      freqs_delivered=hz_del, gridW=gw, gridH=gh)
        png = grid_to_data_url(reconstruct_norm(grid))
        await self.send(cmd="result", png=png, r=r or 0.0, flickerHz=hz_del[0])
        self.spectate(type="grid", w=gw, h=gh, scores=[float(v) for v in np.asarray(grid).ravel()])
        recorder.stop()
        print(f"MUX SESSION DONE r={r} -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_full_extras(self, result):
        """Colour (frequency-tagged RGB) and music (40 Hz-tagged) stages."""
        a = self.args
        freqs = [float(v) for v in str(a.color_freqs).split(",") if v.strip()]
        names = csv_header(os.path.join(self.session, "eeg.csv")) or CH_NAMES
        calib = json.load(open(os.path.join(self.session, "calibration.json")))
        weights = np.array([float(calib["weights"].get(n, 0.0)) for n in names])
        if weights.sum() <= 0:
            weights = np.ones(len(names))

        # ---- colour calibration: full-field R, G, B flicker at f_R, f_G, f_B
        self.spectate(type="stage", stage="color_calib",
                      detail=",".join(f"{f:g}" for f in freqs))
        self.blocks, self.block_hz = [], []
        self.open_log("ccal_log.csv")
        cols = {f"{f:g}": [c, "#000000"] for f, c in
                zip(freqs, ("#ff0000", "#00ff00", "#0000ff"))}
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.color_reps,
                        onS=a.color_on, offS=a.color_on, colorsPerFreq=cols,
                        size=1.0)
        total = a.color_reps * len(freqs) * 2 * a.color_on
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        ccal = score_sweep(self.session, self.blocks, self.block_hz,
                           log_name="ccal_log.csv")
        ordered = sorted(ccal["results"].values(), key=lambda rr: rr["freq"])
        gains = []
        for k, f in enumerate(freqs):
            rr = ordered[k] if k < len(ordered) else None
            if rr is None:
                gains.append(1.0)
                continue
            num = sum(weights[i] * max(rr["channels"][n]["on_med"]
                                       - rr["channels"][n]["off_med"], 0.0)
                      for i, n in enumerate(names))
            g_ = num / weights.sum()
            gains.append(float(g_) if np.isfinite(g_) and g_ > 0.05 else 0.05)
        ccal["gains"] = gains
        ccal["requested"] = freqs
        with open(os.path.join(self.session, "ccal.json"), "w") as f:
            json.dump(ccal, f, indent=1)
        ctab = "  ".join(f"{rr['freq']:g}Hz p={rr['p']:.3f} {rr['best']}"
                         for rr in ordered)
        print(f"colour calibration: {ctab}  gains={['%.2f' % g for g in gains]}")
        result["color_calib"] = dict(table=ctab, gains=gains,
                                     p=[rr["p"] for rr in ordered])
        await self.send(cmd="msg", text="colour calibration done: " + ctab)
        await asyncio.sleep(2.0)

        # ---- colour scan
        gw, gh = a.color_grid_w, a.color_grid_h
        ctarget = targets.color_target(a.color_target, gw, gh)
        np.save(os.path.join(self.session, "target_color.npy"), ctarget)
        self.spectate(type="stage", stage="color_scan", detail="")
        self.color_hzs = []
        self.open_log("color_log.csv")
        await self.send(cmd="start_scan", color=True, gridW=gw, gridH=gh,
                        spc=a.color_spc, freqs=freqs, target=ctarget.tolist())
        scan_s = gw * gh * a.color_spc
        await self.wait_for("scan_done", timeout=scan_s * 2 + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        hz_del = self.measured_cols("color_log.csv")
        for k in range(3):
            if not np.isfinite(hz_del[k]):
                hz_del[k] = (self.color_hzs[k] if k < len(self.color_hzs)
                             else freqs[k])
        print("colour scan delivered:", ["%.2f" % v for v in hz_del])
        try:
            grid, rgb, cres = reconstruct.reconstruct_color(
                self.session, hz_del, weights=weights, gains=gains,
                cursor_log="color_log.csv", target=ctarget)
        except Exception as exc:  # keep the session alive for the music stage
            print("colour reconstruction failed:", repr(exc))
            cres = dict(r_all=None, r_planes=None, hue_accuracy=None,
                        error=repr(exc))
            rgb = None
        text = ("colour r=%s  (R %s, G %s, B %s)  hue accuracy %s" % (
            "n/a" if cres.get("r_all") is None else f"{cres['r_all']:.2f}",
            *(("n/a",) * 3 if not cres.get("r_planes") else
              tuple(f"{v:.2f}" for v in cres["r_planes"])),
            "n/a" if cres.get("hue_accuracy") is None
            else f"{cres['hue_accuracy']:.0%}"))
        print(text)
        result["color"] = dict(freqs_requested=freqs, freqs_delivered=hz_del,
                               gains=gains, **{k2: v for k2, v in cres.items()
                                               if k2 != "out"})
        png = rgb_to_data_url(rgb) if rgb is not None else ""
        await self.send(cmd="color_result", png=png, text=text,
                        good=bool(cres.get("r_all") and cres["r_all"] >= 0.6))
        self.spectate(type="log", msg=text)
        await asyncio.sleep(5.0)

        # ---- music piece tagged at 40 Hz (ASSR = ear as a level meter)
        self.spectate(type="stage", stage="music", detail="")
        self.blocks, self.block_hz = [], []
        self.open_log("assr_log.csv")
        await self.send(cmd="start_assr", music=True, blocks=a.music_blocks,
                        onS=a.music_on, offS=a.music_off, modFreq=40.0,
                                mute=bool(getattr(self, "pc_audio", None)))
        await self.wait_for("assr_done",
                            timeout=a.music_blocks * (a.music_on + a.music_off) + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        mus = score_sweep(self.session, self.blocks,
                          [40.0] * len(self.blocks), log_name=None)
        mres = next(iter(mus["results"].values()), None)
        # legacy 1 Hz-bin score too (it overwrites calibration.json: restore)
        cal_path = os.path.join(self.session, "calibration.json")
        cal_bytes = open(cal_path, "rb").read()
        try:
            legacy = run_session.score_calibration(self.session, self.blocks,
                                                   stim_freq=40.0)
            os.replace(cal_path, os.path.join(self.session, "assr_welch.json"))
        except Exception as exc:
            legacy = dict(passed=None, error=repr(exc))
        finally:
            open(cal_path, "wb").write(cal_bytes)
        mus["legacy_passed"] = legacy.get("passed")
        with open(os.path.join(self.session, "music.json"), "w") as f:
            json.dump(mus, f, indent=1)
        if mres:
            mtext = (f"40 Hz tag: p={mres['p']:.4f} best {mres['best']} "
                     f"(ON {mres['channels'][mres['best']]['on_med']:.2f} / "
                     f"OFF {mres['channels'][mres['best']]['off_med']:.2f})")
            passed = bool(mres["p"] < config.CALIB_P_MAX)
        else:
            mtext, passed = "music stage: no scorable blocks", False
        print("music " + mtext)
        result["music"] = dict(passed=passed, text=mtext,
                               p=mres["p"] if mres else None,
                               per_block=mus.get("per_block"))
        await self.send(cmd="assr_result", passed=passed, table=mtext)
        self.spectate(type="log", msg="music " + mtext)
        await asyncio.sleep(3.0)

    async def run_bwb(self, result, recorder):
        """Colour palette: calibrate one frequency-phase code per colour and the
        no-flicker black class on one scan-sized patch, then scan a class
        picture and decode every cell with FBCCA + phase templates (bwb.py).
        Each colour flickers a bright tint of itself (--bwb-tint); the colour
        is read from the code, not from the hue."""
        import random
        import bwb
        a = self.args
        colours = [c.strip() for c in str(a.bwb_colours).split(",") if c.strip()]
        bad = [c for c in colours if c not in bwb.PALETTE or c == "black"]
        if bad or len(set(colours)) < len(colours):
            raise ValueError(f"--bwb-colours: unknown or repeated {bad or colours}; "
                             f"choose from {', '.join(c for c in bwb.PALETTE if c != 'black')}")
        if a.bwb_codes:
            codes = []
            for part in str(a.bwb_codes).split(","):
                hz, _, deg = part.partition(":")
                codes.append(dict(hz=float(hz), phaseDeg=float(deg or 0)))
        else:
            codes = bwb.palette_codes(len(colours), a.bwb_refresh)
        if len(codes) != len(colours):
            raise ValueError(f"{len(codes)} codes for {len(colours)} colours "
                             f"(--bwb-codes needs one Hz:deg per --bwb-colours entry)")
        names = ["black"] + colours
        show = ["#000000"] + [bwb.tint_hex(c, a.bwb_tint) for c in colours]
        print("colour codes requested:", ", ".join(
            f"{c} {d['hz']:.2f} Hz @ {d['phaseDeg']:.0f} deg ({h})"
            for c, d, h in zip(colours, codes, show[1:])))
        ctarget = targets.bwb_target(a.bwb_target, a.bwb_grid_w, a.bwb_grid_h,
                                     fg=a.bwb_fg, bg=a.bwb_bg, colours=colours,
                                     border=a.bwb_border)
        gh, gw = ctarget.shape
        np.save(os.path.join(self.session, "target_bwb.npy"), ctarget)
        geom = dict(gridW=gw, gridH=gh, patch=a.patch, board=a.board)

        # ---- calibration: every class once per round, shuffled, on one patch
        order = []
        for _ in range(a.bwb_reps):
            rnd = list(range(len(names)))
            random.shuffle(rnd)
            order += rnd
        self.spectate(type="stage", stage="bwb_calib", detail=",".join(colours))
        self.blocks, self.block_hz, self.block_cls = [], [], []
        self.bwb_codes = []
        self.open_log("bwb_cal_log.csv")
        await self.send(cmd="start_bwb_calib", codes=codes, order=order, names=names,
                        show=show, onS=a.bwb_on, restS=1.0, **geom)
        await self.wait_for("bwb_calib_done",
                            timeout=len(order) * (a.bwb_on + 1.0) * 2 + 120)
        self.close_log()
        delivered = self.bwb_codes or codes
        blocks = [dict(cls=int(c), t0=t0, t1=t1)
                  for (_, t0, t1), c in zip(self.blocks, self.block_cls) if c is not None]
        print(f"colour calibration: {len(blocks)} blocks")

        # ---- scan
        self.spectate(type="stage", stage="bwb_scan", detail=a.bwb_target)
        scan_kw = dict(bwb=True, codes=codes, spc=a.bwb_spc, show=show,
                       target=ctarget.tolist(), **geom)
        if a.certainty > 0:
            import adaptive

            def colour_state():
                _, cert, n = adaptive.colour_state(
                    self.session, self.bwb_codes or codes, blocks, a.bwb_spc, gw, gh,
                    colours, a.bwb_channels)
                return cert, n, True
            print(f"scan: one pass, then the least certain squares until every "
                  f"square is {a.certainty:.0%} certain (--bwb-passes is not used)")
            timeline, cert, n = await self.certainty_scan(
                scan_kw, "bwb_log.csv", gw, gh, a.bwb_spc, colour_state)
            result["certainty"] = dict(min_certainty=float(cert.min()), visits=int(n.sum()),
                                       minutes=timeline[-1]["minutes"], timeline=timeline)
        else:
            self.open_log("bwb_log.csv")
            await self.send(cmd="start_scan", passes=a.bwb_passes, **scan_kw)
            await self.wait_for("scan_done",
                                timeout=gw * gh * a.bwb_spc * a.bwb_passes * 2 + 120)
            self.close_log()
        await asyncio.sleep(0.7)
        with open(os.path.join(self.session, "bwb_meta.json"), "w") as f:
            json.dump(dict(codes=delivered, blocks=blocks, spc=a.bwb_spc, colours=colours,
                           show=show, tint=a.bwb_tint), f, indent=1)

        # ---- decode
        print(f"decoding {len(names)} colours...")
        try:
            cls, rgb, res = await asyncio.to_thread(
                bwb.decode, self.session, delivered, blocks, a.bwb_spc, gw, gh,
                ctarget, a.bwb_channels, "bwb_log.csv", colours)
            text = bwb.summary(res)
            good = res.get("accuracy", 0) >= 0.6
        except Exception as exc:
            print("black/white/blue decoding failed:", repr(exc))
            res, rgb, text, good = dict(error=repr(exc)), None, f"decoding failed: {exc}", False
        print(text)
        if "confusion" in res:
            print(f"confusion (rows = shown {'/'.join(names)}, cols = decoded):",
                  res["confusion"])
        result["bwb"] = {k: v for k, v in res.items() if k != "classes"}
        comp = os.path.join(self.session, "bwb_comparison.png")
        if os.path.exists(comp):          # shown vs decoded, side by side
            with open(comp, "rb") as f:
                png = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        else:
            png = rgb_to_data_url(rgb) if rgb is not None else ""
        await self.send(cmd="color_result", png=png, text=text, good=bool(good),
                        wide=os.path.exists(comp))
        self.spectate(type="log", msg=text)
        result["ok"] = True
        recorder.stop()
        print(f"BLACK/WHITE/BLUE DONE -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_sweep(self, result, recorder):
        """Frequency sweep: which flicker frequency (if any) evokes an SSVEP
        in THIS subject with THIS electrode? Frame-exact ON blocks at each
        frequency interleaved with OFF blocks, repeated; each frequency is
        scored with the exact-line detector and an exact permutation test."""
        a = self.args
        freqs = [float(v) for v in str(a.sweep).split(",") if v.strip()]
        self.spectate(type="stage", stage="sweep",
                      detail=",".join(f"{f:g}" for f in freqs))
        self.blocks, self.block_hz = [], []
        self.open_log("sweep_log.csv")
        colors = ["#ffffff", "#000000"] if a.calib_style == "bw" \
            else ["#ffff00", "#0000ff"]
        await self.send(cmd="start_sweep", freqs=freqs, repeats=a.calib_blocks,
                        onS=a.calib_on, offS=a.calib_off, colors=colors,
                        size=a.calib_size)
        total = a.calib_blocks * len(freqs) * (a.calib_on + a.calib_off)
        await self.wait_for("sweep_done", timeout=total + 120)
        self.close_log()
        await asyncio.sleep(0.7)
        g0 = self.delivery_check("sweep_log.csv", total)
        result["g0"] = g0
        if not g0["ok"]:
            msg = "STIMULUS NOT DELIVERED: " + g0["reason"]
            print(msg)
            await self.send(cmd="msg", text=msg)
            result.update(ok=False, error="G0 " + g0["reason"])
            recorder.stop()
            await asyncio.sleep(a.linger)
            return 2
        res = score_sweep(self.session, self.blocks, self.block_hz)
        res["blocks"] = [list(b) + [h] for b, h in zip(self.blocks, self.block_hz)]
        json.dump(res, open(os.path.join(self.session, "sweep.json"), "w"),
                  indent=1)
        lines = [f"{r['freq']:g} Hz (delivered {r['delivered']:.2f}): best "
                 f"{r['best']} stat {r['stat']:+.2f} p={r['p']:.4f} "
                 f"(n_on {r['n_on']}, n_off {r['n_off']})"
                 for r in res["results"].values()]
        table = "   |   ".join(lines)
        if res["passed"]:
            verdict = (f"SSVEP at {res['best_freq']:g} Hz on "
                       f"{res['best_channel']} (p={res['p_min']:.4f})")
        else:
            verdict = (f"no frequency passes (min p={res.get('p_min', float('nan')):.3f}"
                       f" at {res.get('best_freq')} Hz)")
        print("SWEEP " + verdict)
        for line in lines:
            print("  " + line)
        result.update(ok=True, sweep=dict(passed=res["passed"],
                                          best_freq=res["best_freq"],
                                          p_min=res.get("p_min")))
        await self.send(cmd="sweep_result", passed=res["passed"],
                        verdict=verdict, table=table)
        self.spectate(type="log", msg="sweep " + verdict)
        recorder.stop()
        print(f"SWEEP SESSION DONE passed={res['passed']} -> {self.session}")
        await asyncio.sleep(a.linger)
        return 0

    async def run_alpha(self, result, recorder):
        """Electrode check: eyes-open / eyes-closed blocks. Occipital alpha
        (8-12 Hz) rises several-fold with eyes closed at Oz; a channel that
        shows no rise is not on occipital scalp (or not coupled). Stimulus-
        free, so it validates the aux electrode independently of SSVEP."""
        a = self.args
        self.spectate(type="stage", stage="alpha", detail="")
        self.blocks = []
        await self.send(cmd="start_alpha", blocks=a.calib_blocks,
                        openS=a.calib_on, closedS=a.calib_off)
        await self.wait_for("alpha_done",
                            timeout=a.calib_blocks * (a.calib_on + a.calib_off + 3) + 60)
        await asyncio.sleep(0.7)
        res = score_alpha(self.session, self.blocks)
        res["blocks"] = [list(b) for b in self.blocks]  # (kind, t0, t1) on the EEG clock
        json.dump(res, open(os.path.join(self.session, "alpha.json"), "w"),
                  indent=1)
        table = "   ".join(
            f"{n} {v['iaf_hz']:.1f}Hz x{v['peak_ratio']:.1f}"
            for n, v in res["channels"].items())
        verdict = res["verdict"]
        print(f"ALPHA CHECK {verdict}: {table}")
        result.update(ok=True, alpha=res)
        await self.send(cmd="alpha_result", verdict=verdict, table=table,
                        good=res["aux_ok"])
        self.spectate(type="log", msg="alpha " + table)
        recorder.stop()
        await asyncio.sleep(a.linger)
        return 0

    async def run_assr(self, result):
        """'Ear as a microphone' v1: 40 Hz amplitude-modulated tone in ON/OFF
        blocks; the auditory steady-state response (ASSR) at the modulation
        frequency is scored with the same machinery as visual calibration.
        Honest scope: this detects whether EEG tracks the sound envelope on
        a minutes timescale — one audio pixel, not real-time audio."""
        a = self.args
        self.spectate(type="stage", stage="assr", detail="")
        self.blocks = []
        await self.send(cmd="start_assr", blocks=a.calib_blocks,
                        onS=config.CALIB_ON_S, offS=config.CALIB_OFF_S,
                        modFreq=40.0)
        await self.wait_for("assr_done", timeout=a.calib_blocks * 16 + 120)
        await asyncio.sleep(0.7)
        assr = run_session.score_calibration(self.session, self.blocks,
                                             stim_freq=40.0)
        os.replace(os.path.join(self.session, "calibration.json"),
                   os.path.join(self.session, "assr.json"))
        table = "   ".join(
            f"{n} d'={assr['channels'][n]['dprime']:.1f}"
            for n in assr["names"])
        print(f"ASSR passed={assr['passed']} {table}")
        result.update(ok=True, assrPassed=assr["passed"],
                      assrBest=assr["best"], assrTable=table)
        await self.send(cmd="assr_result", passed=assr["passed"],
                        table=table)
        self.spectate(type="calib", detail="ASSR " + table,
                      passed=assr["passed"], best=assr["best"], weights=[])
        await asyncio.sleep(a.linger)
        return 0

    def delivery_check(self, log_name, duration_s):
        """Rows per second and flicker edges in a stimulus log."""
        path = os.path.join(self.session, log_name)
        rows, edges, last = 0, 0, 0
        try:
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) < 5:
                        continue
                    rows += 1
                    fl = int(parts[4])
                    if fl == 1 and last == 0:
                        edges += 1
                    last = fl
        except OSError:
            pass
        fps = rows / max(duration_s, 1e-9)
        if fps < 20:
            reason = f"page ran at {fps:.1f} fps (hidden/background tab?)"
        elif edges < 10:
            reason = f"only {edges} flicker edges logged"
        else:
            reason = ""
        return dict(ok=not reason, rows=rows, fps=round(fps, 1),
                    edges=edges, reason=reason)

    def measured_flicker(self):
        """Median rising-edge period from the page's own clock (jitter-free)."""
        edges = []
        for name in ("cursor_log.csv", "calib_log.csv"):
            path = os.path.join(self.session, name)
            if not os.path.exists(path):
                continue
            last = 0
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) < 6:
                        continue
                    fl = int(parts[4])
                    if fl == 1 and last == 0:
                        edges.append(float(parts[5]))
                    last = fl
        periods = sorted(b - a for a, b in zip(edges, edges[1:])
                         if 0.01 < b - a < 0.5)
        if len(periods) < 5:
            return -1.0
        return 1.0 / periods[len(periods) // 2]


def reconstruct_norm(grid):
    g = np.asarray(grid)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - lo) / max(hi - lo, 1e-12), 0, 1)


def rgb_to_data_url(rgb, scale=24):
    """(h, w, 3) floats in [0,1] -> PNG data URL, nearest-neighbour upscaled."""
    arr = (np.clip(np.asarray(rgb, float), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_to_data_url(norm, scale=24):
    img = Image.fromarray((norm * 255).astype(np.uint8), "L")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def http_page(connection, request):
    if request.path.startswith("/ws"):
        return None
    with open(os.path.join(ROOT, "xr_stimulus.html"), "rb") as f:
        body = f.read()
    return Response(200, "OK", Headers([
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Connection", "close")]), body)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None,
                    help="session folder (default: a new runs/<date-time>_<mode>, "
                         "so a run never overwrites another run's EEG)")
    ap.add_argument("--http-port", type=int, default=8082)
    ap.add_argument("--osc-port", default="5000",
                    help="UDP port, or comma list for several headbands "
                         "(e.g. 5000,5001 -> merged 8-channel eeg.csv)")
    ap.add_argument("--muse", action="store_true",
                    help="connect to the Muse over THIS PC's Bluetooth (muselsl) "
                         "and record it directly; no phone app, no OSC")
    ap.add_argument("--muse-model", default="auto", choices=["auto", "athena", "legacy"],
                    help="Muse protocol for --muse (athena = Muse S Gen 3)")
    ap.add_argument("--muse-address", default=None, help="Muse MAC address for --muse")
    ap.add_argument("--muse-preset", default=None,
                    help="--muse: muselsl preset (default p20 = aux/Oz ON on a classic "
                         "Muse; none with --no-aux or --muse-model athena)")
    ap.add_argument("--lsl", action="store_true",
                    help="record an EEG stream another program already puts on LSL "
                         "(muselsl stream, mind2motor run_muse.py --live, Petal)")
    ap.add_argument("--no-aux", action="store_true",
                    help="with --muse/--lsl: drop the aux column (no Oz electrode)")
    ap.add_argument("--preset", choices=list(config.PRESETS), default="")
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--spc", type=float, default=4.0)
    ap.add_argument("--calib-blocks", type=int, default=config.CALIB_BLOCKS)
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--sweep", default="7.5,10,12,15,20",
                    help="--mode sweep: flicker frequencies to test (Hz)")
    ap.add_argument("--page-token", default="",
                    help="only pages opened with ?k=<token> may drive the "
                         "session (stale tabs on other devices are ignored)")
    ap.add_argument("--legible", action="store_true",
                    help="the settings of the published legible 'NO', with a bigger "
                         "flicker: --target pix:NO --freq 12 --spc 8 --passes 3 "
                         "--patch 1.5 --calib-style bw --calib-size 1 --calib-blocks 6 "
                         "--calib-on 8 --calib-off 8 (any of these given explicitly wins)")
    ap.add_argument("--mask-blinks", action="store_true",
                    help="leave blinks (AF7/AF8) and closed eyes (Oz alpha) out of "
                         "every position's spectrum; redo a position that is more "
                         "than --mask-limit masked (off by default)")
    ap.add_argument("--blink-uv", type=float, default=None,
                    help="blink threshold on AF7/AF8 (0.5-10 Hz), uV (default 100)")
    ap.add_argument("--alpha-ratio", type=float, default=None,
                    help="eyes closed when Oz 8-12 Hz power exceeds this x the "
                         "eyes-open baseline (default 3)")
    ap.add_argument("--closure-min-s", type=float, default=None,
                    help="minimum duration of a closure, s (default 1)")
    ap.add_argument("--margin-s", type=float, default=None,
                    help="masked margin each side of a blink / closure, s (default 0.2)")
    ap.add_argument("--max-redo", type=int, default=2,
                    help="--mask-blinks: redo one position at most this many times")
    ap.add_argument("--mask-limit", dest="limit", type=float, default=None,
                    help="redo / drop a position more than this fraction masked "
                         "(default 0.35)")
    ap.add_argument("--choice", choices=["bw", "bwb"], default=None,
                    help="--mode visual: picture after calibration (default: ask on "
                         "the page; ?auto=1 takes bw)")
    ap.add_argument("--no-eq1", action="store_true",
                    help="paper method without step 3 (Eq. 1 vertical blend)")
    ap.add_argument("--recon-method", choices=["welch", "line", "paper", "upstream"],
                    default="paper",
                    help="per-cell score: paper = Mann 2019 exactly (1700-sample "
                         "window, (f0 + 2 f0) / all 14-50 Hz power, Eq. 1); upstream "
                         "= this repo as published (whole dwell, clipping, no Eq. 1); "
                         "welch = 1 Hz bins; line = exact-frequency line SNR")
    ap.add_argument("--pc-audio", action="store_true",
                    help="music stage: play the tagged music on THIS "
                         "computer's default output (headphones) and mute "
                         "the page")
    ap.add_argument("--board", type=float, default=0.84,
                    help="grey scan: board size as a fraction of the screen; "
                         "shrink it (0.6) with a large --patch so the square "
                         "is never clipped at the edge positions")
    ap.add_argument("--vblend", action="store_true",
                    help="apply the paper's Eq. 1 vertical blend. It is for "
                         "overlapping scan lines; on discrete positions it "
                         "only smears rows together, so it is off by default")
    ap.add_argument("--passes", type=int, default=1,
                    help="repeat the grey scan this many times; repeat visits "
                         "to a position are averaged (the paper integrates "
                         "over 48 overlapping scan lines)")
    ap.add_argument("--color-freqs", default="7.2,9,12",
                    help="--mode full: R,G,B tag frequencies (frame-exact at "
                         "72 fps: 7.2/9/12; at 240 fps: 8/10/12)")
    ap.add_argument("--color-reps", type=int, default=3)
    ap.add_argument("--color-on", type=float, default=6.0,
                    help="colour calibration ON = OFF seconds per block")
    ap.add_argument("--color-grid-w", type=int, default=6)
    ap.add_argument("--color-grid-h", type=int, default=4)
    ap.add_argument("--color-spc", type=float, default=5.0)
    ap.add_argument("--color-target", default="flag",
                    help="flag | quad | ring | text:X | image path")
    ap.add_argument("--patch", type=float, default=1.0,
                    help="mux scan: flicker patch size in units of a 6x4-grid "
                         "cell, independent of the grid (0 = the grid cell)")
    ap.add_argument("--music-blocks", type=int, default=6)
    ap.add_argument("--music-on", type=float, default=10.0)
    ap.add_argument("--music-off", type=float, default=10.0)
    ap.add_argument("--mode", choices=["visual", "assr", "calib", "alpha",
                                       "sweep", "full", "extras", "mux",
                                       "music", "bwb"],
                    default="visual",
                    help="calib = G1 only (calibration, no scan); "
                         "assr = 'ear as a microphone' 40 Hz tone blocks")
    ap.add_argument("--calib-style", choices=["yb", "bw"], default="yb",
                    help="flicker colours: yb = yellow/blue (chromatic+luminance), "
                         "bw = white/black (max luminance contrast)")
    ap.add_argument("--calib-size", type=float, default=0.38,
                    help="flicker square as a fraction of the short screen side; "
                         ">= 1 = full field")
    ap.add_argument("--calib-on", type=float, default=config.CALIB_ON_S)
    ap.add_argument("--calib-off", type=float, default=config.CALIB_OFF_S)
    ap.add_argument("--freq", type=float, default=config.STIM_FREQ_HZ)
    ap.add_argument("--spectator", default="127.0.0.1:8090",
                    help="'' disables the spectator mirror")
    ap.add_argument("--require-pass", action="store_true")
    ap.add_argument("--signal-timeout", type=float, default=600)
    ap.add_argument("--certainty", type=float, default=0.99,
                    help="black & white and colour scans: one pass, then the least "
                         "certain squares again until every square is at least this "
                         "sure of its colour (0 = fixed --passes / --bwb-passes)")
    ap.add_argument("--certainty-batch", type=int, default=6,
                    help="squares revisited between certainty checks")
    ap.add_argument("--certainty-max-passes", type=int, default=6,
                    help="most visits any one square gets")
    ap.add_argument("--bwb-colours", default="white,red,green,blue",
                    help="--mode bwb: palette colours besides black, in code order "
                         "(white red green blue yellow cyan magenta orange purple pink "
                         "lime teal brown grey). Up to 11 at 72 fps")
    ap.add_argument("--bwb-codes", default="",
                    help="frequency-phase codes 'Hz:deg,...', one per colour (snapped to "
                         "whole display frames; the page reports what it delivers). "
                         "Default: assigned between 12 and 15 Hz for --bwb-refresh, e.g. "
                         "white 12:0, red 12:180, green 14.4:0, blue 14.4:144 at 72 fps")
    ap.add_argument("--bwb-refresh", type=float, default=72.0,
                    help="panel rate the default codes are planned for")
    ap.add_argument("--bwb-tint", type=float, default=0.5,
                    help="how far each colour's flicker is mixed toward white "
                         "(0 = pure colour, 1 = white): brighter flicker, bigger response")
    ap.add_argument("--bwb-target", default="bands",
                    help="bands | checker | text:X | pix:X | image path")
    ap.add_argument("--bwb-fg", default=None,
                    help="text:/pix: targets: glyph colour (default the first palette "
                         "colour); pix: takes one per letter, e.g. red,green")
    ap.add_argument("--bwb-border", default=None,
                    help="text:/pix: targets: colour of the one-cell frame (default the "
                         "background colour)")
    ap.add_argument("--bwb-bg", default=None,
                    help="text:/pix: targets: background colour (default the second palette "
                         "colour, with black rows); setting fg/bg adds a background margin")
    ap.add_argument("--bwb-grid-w", type=int, default=6)
    ap.add_argument("--bwb-grid-h", type=int, default=4)
    ap.add_argument("--bwb-spc", type=float, default=4.0,
                    help="seconds per cell in the black/white/blue scan")
    ap.add_argument("--bwb-passes", type=int, default=1)
    ap.add_argument("--bwb-reps", type=int, default=4,
                    help="calibration blocks per class")
    ap.add_argument("--bwb-on", type=float, default=8.0,
                    help="seconds per calibration block")
    ap.add_argument("--bwb-channels", default="",
                    help="channels to decode (default: all, e.g. TP9,TP10,AUX)")
    ap.add_argument("--linger", type=float, default=10,
                    help="seconds to keep serving the result before exit")
    args = ap.parse_args()
    if args.legible:
        legible = dict(target="pix:NO", freq=12.0, spc=8.0, passes=3, patch=1.5,
                       calib_style="bw", calib_size=1.0, calib_blocks=6,
                       calib_on=8.0, calib_off=8.0)
        given = {act.dest for act in ap._actions
                 if any(o == x or x.startswith(o + "=") for o in act.option_strings
                        for x in sys.argv[1:])}
        for k, v in legible.items():
            if k not in given:          # a flag typed on the command line wins
                setattr(args, k, v)
    if args.session is None:
        args.session = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S") + "_" + args.mode)
    if not args.vblend:
        config.VERTICAL_KERNEL = [1.0]
    if args.target.startswith("pix"):
        import targets as _t
        _g = _t.load_target(args.target)
        args.grid_h = max(args.grid_h, _g.shape[0]) if args.grid_h != 8 else _g.shape[0]
        args.grid_w = max(args.grid_w, _g.shape[1]) if args.grid_w != 12 else _g.shape[1]
    if args.preset:
        args.grid_w, args.grid_h, args.spc = config.PRESETS[args.preset]
    # rank can never exceed the ON-block count (same clamp as run_session)
    config.CALIB_RANK_MIN = min(config.CALIB_RANK_MIN, args.calib_blocks - 1)

    driver = Driver(args)
    if args.pc_audio:
        import pc_audio
        driver.pc_audio = pc_audio.PcAudio(mod_hz=40.0)
        print("music plays on this PC:", driver.pc_audio.device)

    async def ws_handler(ws):
        if not getattr(ws, "request", None) or                 ws.request.path.startswith("/ws"):
            driver.pages.append(ws)
            driver.vis[ws] = None
            print(f"stimulus page connected (#{len(driver.pages)})")
            try:
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        continue
                    typ = m.get("type")
                    if typ == "hello":
                        tok = getattr(args, "page_token", "") or ""
                        if tok and tok not in str(m.get("q", "")):
                            print("page without the session token ignored "
                                  f"(open the URL with ?k={tok})")
                            driver.vis[ws] = "hidden"   # never adopted
                            try:
                                await ws.send(json.dumps(dict(
                                    cmd="msg", text=f"stale tab: open ?k={tok}")))
                            except Exception:
                                pass
                            continue
                        driver.vis[ws] = m.get("vis", "visible")
                        await driver.consider(ws)
                    elif typ == "visibility":
                        driver.vis[ws] = m.get("state")
                        await driver.consider(ws)
                    if driver.page is ws:
                        try:
                            driver.handle(m)
                        except (ValueError, KeyError):
                            pass
            except websockets.ConnectionClosed:
                pass  # tab closed / headset slept: normal, not an error
            finally:
                if ws in driver.pages:
                    driver.pages.remove(ws)
                driver.vis.pop(ws, None)
                if driver.page is ws:
                    driver.page = None
                    print("stimulus page disconnected")
                    for other in reversed(driver.pages):
                        if driver.vis.get(other) == "visible":
                            await driver.adopt(other)
                            break

    async with websockets.serve(ws_handler, "0.0.0.0", args.http_port,
                                process_request=http_page,
                                max_size=2 ** 22):
        print(f"stimulus page:  http://{local_ip()}:{args.http_port}/  "
              f"(?auto=1 for unattended)"
              + (f"  token: ?k={args.page_token}" if args.page_token else ""))
        muse = None
        if args.muse:
            from muse_stream import MuseStream
            preset = args.muse_preset
            if preset is None and not args.no_aux and args.muse_model != "athena":
                preset = "20"           # classic Muse: aux (Oz) input ON
            muse = MuseStream(args.muse_address, args.muse_model, preset=preset).start()
            print("EEG in:         Muse over this PC's Bluetooth (muselsl -> LSL)"
                  + (f", preset {preset}" if preset else "")
                  + ("" if preset or args.no_aux else
                     "; Athena: aux preset not set - check AUX with --mode alpha"))
        elif args.lsl:
            print("EEG in:         LSL EEG stream (start muselsl / mind2motor first)")
        else:
            print(f"EEG OSC in:     {local_ip()}:{args.osc_port}  "
                  "(Mind Monitor / MuseLog / phantom)")
        try:
            code = await driver.run()
        finally:
            if muse:
                muse.stop()
    sys.exit(code)


if __name__ == "__main__":
    asyncio.run(main())
