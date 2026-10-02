"""One-command eye-as-a-camera session: signal check -> SSVEP calibration ->
raster scan -> reconstruction, in a single pygame window.

    python run_session.py --source osc     --preset quick --target "text:NO"
    python run_session.py --source lsl     --preset standard
    python run_session.py --source phantom --preset quick --auto --windowed

Sources: osc = MuseLog phone app streaming to this PC (start the recorder's
printed IP/port in the app); lsl = muselsl over laptop Bluetooth; phantom =
hardware-free synthetic subject (full-stack test).

Everything is logged to <session>/session.json; exit code 0 means every stage
completed cleanly.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np
import pygame
from PIL import Image, ImageDraw

import config
import reconstruct
import targets
from stimulus import run_raster
from tailer import CsvTail

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# infrastructure
# --------------------------------------------------------------------------

class SessionLog:
    def __init__(self, path, params):
        self.path = path
        self.data = {"params": params, "events": []}
        self.save()

    def event(self, stage, status, **kw):
        self.data["events"].append(
            dict(stage=stage, status=status, t=local_clock(), **kw))
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1)


class EEGTail:
    """Inline (no thread) follower of eeg.csv for rate/RMS displays."""

    def __init__(self, path, window_s=2.0):
        self.tail = CsvTail(path)
        self.window_s = window_s
        self.rows = deque(maxlen=4096)

    def poll(self):
        for r in self.tail.poll():
            try:
                self.rows.append((float(r[0]), [float(v) for v in r[1:]]))
            except (ValueError, IndexError):
                pass

    def _window(self):
        cutoff = local_clock() - self.window_s
        return [row for row in self.rows if row[0] >= cutoff]

    def rate(self):
        return len(self._window()) / self.window_s

    def rms(self):
        win = self._window()
        if len(win) < 8:
            return []
        arr = np.array([r[1] for r in win])
        return list(arr.std(axis=0))


class Recorder:
    def __init__(self, source, session, port, muse=""):
        self.session = session
        self.stop_file = os.path.join(session, "stop_recorder")
        self.err_path = os.path.join(session, "recorder.stderr.log")
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        if source in ("osc", "phantom"):
            self.cmd = [PY, "osc_acquire.py", "--session", session,
                        "--port", str(port), "--stop-file", self.stop_file]
        elif source == "ble":
            # the headband straight over this computer's Bluetooth
            self.cmd = [PY, "muse_ble.py", "--session", session,
                        "--stop-file", self.stop_file]
            if muse:
                self.cmd += ["--muse", muse]
        else:
            self.cmd = [PY, "acquire.py", "--session", session,
                        "--stop-file", self.stop_file]
        self.proc = None

    def start(self, append=False):
        cmd = self.cmd + (["--append"] if append else [])
        # keep stderr: a bind failure (port already held by an orphaned
        # recorder) must be diagnosable, not swallowed by DEVNULL
        err = open(self.err_path, "a")
        self.proc = subprocess.Popen(cmd, cwd=ROOT,
                                     stdout=subprocess.DEVNULL, stderr=err)

    def stderr_tail(self):
        try:
            with open(self.err_path) as f:
                return f.read()[-600:].strip()
        except OSError:
            return ""

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if not self.alive():
            return
        open(self.stop_file, "w").close()
        try:
            self.proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


def spawn_phantom(session, port):
    stop_file = os.path.join(session, "stop_phantom")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    proc = subprocess.Popen(
        [PY, "phantom_subject.py", "--session", session,
         "--port", str(port), "--stop-file", stop_file],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, stop_file


def draw_lines(screen, lines, color=(220, 220, 220), y0=80):
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont("consolas", 26)
    y = y0
    for text, col in lines:
        surf = font.render(text, True, col or color)
        screen.blit(surf, (80, y))
        y += 36
    pygame.display.flip()


def pump_keys():
    """Return set of pressed keys this frame; raises SystemExit on QUIT."""
    keys = set()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            keys.add("quit")
        elif ev.type == pygame.KEYDOWN:
            keys.add(ev.key)
    return keys


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_signal_check(screen, tail, recorder, args):
    from osc_acquire import local_ips
    ips = ", ".join(local_ips()) or "unknown"
    stable_since = None
    dead_since = None
    deadline = time.monotonic() + 90 if args.auto else None
    clock = pygame.time.Clock()
    while True:
        tail.poll()
        rate = tail.rate()
        rms = tail.rms()
        stable = rate > 40
        now = time.monotonic()
        if stable and stable_since is None:
            stable_since = now
        if not stable:
            stable_since = None
        ready = stable_since is not None and now - stable_since >= 3

        # A dead recorder can never recover here — fail fast with its stderr
        # (e.g. WinError 10048: port held by an orphaned recorder).
        if recorder.alive():
            dead_since = None
        elif dead_since is None:
            dead_since = now
        elif now - dead_since > 3:
            print("recorder died; stderr tail:\n" + recorder.stderr_tail())
            return "fail", rate

        keys = pump_keys()
        if "quit" in keys or pygame.K_ESCAPE in keys:
            return "abort", rate
        if pygame.K_SPACE in keys and (ready or args.force):
            return "ok", rate
        if args.auto and ready:
            return "ok", rate
        if deadline and now > deadline:
            return "fail", rate

        rms_txt = "  ".join(f"{v:7.2f}" for v in rms) if rms else "--"
        lines = [
            ("SIGNAL CHECK", (255, 255, 255)),
            ("", None),
            (f"source: {args.source}   port: {args.port}", None),
            (f"this PC: {ips}   <- set as Target IP in MuseLog", None),
            ("", None),
            (f"sample rate: {rate:6.1f} Hz   "
             f"({'STABLE' if stable else 'waiting for stream...'})",
             (0, 220, 0) if stable else (220, 160, 0)),
            (f"channel RMS: {rms_txt}", None),
            (f"recorder: {'running' if recorder.alive() else 'DEAD'}",
             (0, 220, 0) if recorder.alive() else (255, 60, 60)),
            ("", None),
            ("SPACE = continue (enabled when stable 3 s)   ESC = abort",
             (150, 150, 150)),
        ]
        draw_lines(screen, lines)
        clock.tick(10)


def stage_calibrate(screen, session, log):
    """Alternating fullscreen flicker ON / black OFF blocks; score channels."""
    calib_path = os.path.join(session, "calib_log.csv")
    half_period = max(1, int(round(config.DISPLAY_FPS
                                   / config.STIM_FREQ_HZ / 2)))
    W, H = screen.get_size()
    patch = pygame.Rect(0, 0, min(600, W - 100), min(600, H - 100))
    patch.center = screen.get_rect().center

    blocks = []
    clock = pygame.time.Clock()
    frame = 0
    f = open(calib_path, "w", newline="", buffering=1)
    f.write("time,grid_x,grid_y,luminance,flicker_on\n")
    try:
        for b in range(config.CALIB_BLOCKS):
            for kind, dur, lum in (("on", config.CALIB_ON_S, 1.0),
                                   ("off", config.CALIB_OFF_S, 0.0)):
                t0 = local_clock()
                for _ in range(int(dur * config.DISPLAY_FPS)):
                    keys = pump_keys()
                    if "quit" in keys or pygame.K_ESCAPE in keys:
                        return None
                    flicker = (frame // half_period) % 2 == 0 and lum > 0
                    screen.fill((0, 0, 0))
                    if lum > 0:
                        pygame.draw.rect(
                            screen,
                            (255, 255, 0) if flicker else (0, 0, 255), patch)
                    pygame.draw.circle(screen, (255, 255, 255),
                                       screen.get_rect().center, 4)
                    pygame.display.flip()
                    f.write(f"{local_clock():.6f},-1,-1,{lum:.1f},"
                            f"{int(flicker)}\n")
                    frame += 1
                    clock.tick(config.DISPLAY_FPS)
                blocks.append((kind, t0, local_clock()))
    finally:
        f.close()

    draw_lines(screen, [("scoring calibration...", None)])
    time.sleep(0.7)  # let the last packets land in eeg.csv
    return score_calibration(session, blocks)


def permutation_gate(block_scores, kinds, p_max=None, max_exact=250000, seed=0):
    """Exact family-wise permutation test (PLAN_NEXT Rank A #1).

    statistic = max over channels of mean(ON) - mean(OFF) block scores;
    null = every C(N, n_on) relabeling of the blocks all channels scored
    (Monte Carlo 20k if the enumeration is too large). p >= 1/n_null by
    construction; the gate is 'decidable' only when 1/n_null <= p_max."""
    from itertools import combinations
    from math import comb
    p_max = config.CALIB_P_MAX if p_max is None else p_max
    S = np.array(block_scores, dtype=float)          # (channels, blocks)
    valid = ~np.isnan(S).any(axis=0)
    S = S[:, valid]
    labels = np.array([k == "on" for k, v in zip(kinds, valid) if v])
    N, n_on = len(labels), int(labels.sum())
    out = {"n_blocks_used": int(N), "n_on": n_on, "p_max": p_max}
    if N < 4 or n_on == 0 or n_on == N or S.shape[0] == 0:
        out.update(decidable=False, p_fw=1.0, p_channel=[], stat=0.0, n_null=0)
        return out

    def stat(mask):
        return S[:, mask].mean(axis=1) - S[:, ~mask].mean(axis=1)

    obs = stat(labels)
    total = comb(N, n_on)
    if total <= max_exact:
        n_null, exact = total, True
        masks = (np.isin(np.arange(N), on_idx)
                 for on_idx in combinations(range(N), n_on))
    else:
        n_null, exact = 20000, False
        rng = np.random.default_rng(seed)
        masks = (np.isin(np.arange(N), rng.choice(N, n_on, replace=False))
                 for _ in range(n_null))
    null = np.empty((n_null, S.shape[0]))
    for i, m in enumerate(masks):
        null[i] = stat(m)
    null_fw = null.max(axis=1)
    out.update(decidable=bool(1.0 / n_null <= p_max), exact=exact,
               n_null=int(n_null), stat=float(obs.max()),
               p_fw=float((null_fw >= obs.max()).mean()),
               p_channel=[float((null[:, ch] >= obs[ch]).mean())
                          for ch in range(S.shape[0])])
    return out


def score_calibration(session, blocks, stim_freq=None):
    header, rows = reconstruct.read_csv(os.path.join(session, "eeg.csv"))
    names = header[1:]
    t = np.array([float(r[0]) for r in rows])
    data = np.array([[float(v) for v in r[1:]] for r in rows])
    fs = reconstruct.infer_fs(t)

    def mad_sigma(x):
        return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12

    result = {"names": names, "fs": round(fs, 2), "channels": {},
              "blocks": [[k, t0, t1] for k, t0, t1 in blocks]}
    weights = {}
    best = None
    block_scores = []  # per channel, per block (NaN if dropped)
    for c, name in enumerate(names):
        sigma = reconstruct.hf_sigma(data[:, c])
        scores = {"on": [], "off": []}
        per_block = []
        for kind, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            sc = reconstruct.cell_score(data[i0:i1, c], fs, sigma, stim_freq)
            per_block.append(float(sc))
            if not np.isnan(sc):
                scores[kind].append(float(sc))
        block_scores.append(per_block)
        on, off = np.array(scores["on"]), np.array(scores["off"])
        if len(on) < 3 or len(off) < 3:
            stats = dict(dprime=0.0, ratio=1.0, rank=0)
        else:
            # Scale-aware floor: a MAD of <=6 block scores can quantize to
            # ~0 and make d' explode; cap d' near 20 instead.
            spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
            spread = max(spread, 0.05 * abs(float(np.median(on))))
            stats = dict(
                dprime=float((np.median(on) - np.median(off))
                             / max(spread, 1e-12)),
                ratio=float(np.median(on) / max(np.median(off), 1e-12)),
                rank=int((on > off.max()).sum()),
            )
        stats["on_scores"] = scores["on"]
        stats["off_scores"] = scores["off"]
        result["channels"][name] = stats
        if best is None or stats["dprime"] > result["channels"][best]["dprime"]:
            best = name
        # Weight by the bounded on/off ratio (d' only gates admission): the
        # d' of a handful of blocks is too noisy to apportion weight by.
        weights[name] = max(0.0, stats["ratio"] - 1.0) \
            if stats["dprime"] >= config.CALIB_WEIGHT_DPRIME_MIN else 0.0

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(names) for k in names}  # fallback: uniform

    b = result["channels"][best]
    result["best"] = best
    legacy = bool(b["dprime"] >= config.CALIB_DPRIME_MIN
                  and b["ratio"] >= config.CALIB_RATIO_MIN
                  and b["rank"] >= config.CALIB_RANK_MIN)
    perm = permutation_gate(block_scores, [k for k, _, _ in blocks])
    result["permutation"] = perm
    result["passed_dprime_legacy"] = legacy
    if perm["decidable"]:
        # d'/ratio/rank are reported only; the exact test decides.
        result["passed"] = bool(perm["p_fw"] < perm["p_max"])
        result["gate"] = "permutation"
        if perm["p_channel"]:
            result["best"] = names[int(np.argmin(perm["p_channel"]))]
    else:
        result["passed"] = legacy
        result["gate"] = "dprime-legacy (too few blocks for exact p)"
    result["weights"] = weights
    with open(os.path.join(session, "calibration.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def stage_confirm(screen, calib, args):
    lines = [("CALIBRATION RESULTS", (255, 255, 255)), ("", None)]
    for name, s in calib["channels"].items():
        w = calib["weights"][name]
        lines.append((f"{name:>5}: d'={s['dprime']:6.2f}  "
                      f"on/off={s['ratio']:5.2f}  rank={s['rank']}/"
                      f"{config.CALIB_BLOCKS}  weight={w:.2f}", None))
    lines.append(("", None))
    if calib["passed"]:
        lines.append((f"SSVEP detected (best: {calib['best']}). "
                      "SPACE = start scan, ESC = abort", (0, 220, 0)))
    else:
        lines.append(("WARNING: no reliable SSVEP contrast. A scan will "
                      "likely fail.", (255, 60, 60)))
        lines.append(("Check electrode contact / darken room. "
                      "O = override and scan anyway, ESC = abort",
                      (255, 60, 60)))
    draw_lines(screen, lines)

    if args.auto:
        return "ok"
    while True:
        keys = pump_keys()
        if "quit" in keys or pygame.K_ESCAPE in keys:
            return "abort"
        if calib["passed"] and pygame.K_SPACE in keys:
            return "ok"
        if not calib["passed"] and pygame.K_o in keys:
            return "override"
        pygame.time.wait(50)


class ScanControls:
    """Watchdog + row logging hooks passed to stimulus.run_raster."""

    def __init__(self, tail, recorder, nominal_rate, log, auto=False):
        self.tail = tail
        self.recorder = recorder
        self.nominal = max(nominal_rate, 1.0)
        self.log = log
        self.auto = auto  # unattended: recover + log, but never block on keys
        self._pause = False
        self._unhealthy_since = None
        self._frame = 0
        self._respawned = 0.0

    def on_frame(self):
        self._frame += 1
        if self._frame % 30:
            return
        self.tail.poll()
        now = time.monotonic()
        if not self.recorder.alive() and now - self._respawned > 10:
            self.log.event("scan", "recorder_respawn")
            self.recorder.start(append=True)
            self._respawned = now
            if not self.auto:
                self._pause = True
        if self.tail.rate() < 0.25 * self.nominal:
            if self._unhealthy_since is None:
                self._unhealthy_since = now
            elif now - self._unhealthy_since > 3:
                self.log.event("scan", "stream_dropout",
                               rate=self.tail.rate())
                if not self.auto:
                    self._pause = True
                self._unhealthy_since = None
        else:
            self._unhealthy_since = None

    def on_row_end(self, gy):
        self.log.event("scan", "row_done", row=int(gy))

    def should_pause(self):
        p = self._pause
        self._pause = False
        return p


def build_compare_png(session, target, grid):
    def to_img(arr, label):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        norm = np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1)
        img = Image.fromarray((norm * 255).astype(np.uint8), "L").convert("RGB")
        img = img.resize((arr.shape[1] * config.UPSCALE,
                          arr.shape[0] * config.UPSCALE), Image.BICUBIC)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width - 1, 23], fill=(20, 20, 20))
        d.text((8, 5), label, fill=(255, 255, 255))
        return img

    a = to_img(target, "target")
    b = to_img(grid, "reconstruction from EEG")
    out = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)),
                    (0, 0, 0))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 8, 0))
    path = os.path.join(session, "compare.png")
    out.save(path)
    return path


def stage_show_result(screen, session, compare_path, r, args):
    img = pygame.image.load(compare_path)
    W, H = screen.get_size()
    scale = min((W - 80) / img.get_width(), (H - 160) / img.get_height(), 1.0)
    img = pygame.transform.smoothscale(
        img, (int(img.get_width() * scale), int(img.get_height() * scale)))
    screen.fill((0, 0, 0))
    screen.blit(img, img.get_rect(center=(W // 2, H // 2 - 20)))
    font = pygame.font.SysFont("consolas", 28)
    msg = f"r = {r:.3f}" if r is not None else "no ground truth"
    surf = font.render(f"{msg}    (any key to exit)", True, (220, 220, 220))
    screen.blit(surf, (80, H - 80))
    pygame.display.flip()
    if args.auto:
        return
    while True:
        keys = pump_keys()
        if keys:
            return
        pygame.time.wait(50)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["osc", "lsl", "ble", "phantom"],
                    default="osc",
                    help="ble = the Muse straight over this computer's "
                         "Bluetooth (muse_ble.py), no phone app")
    ap.add_argument("--preset", choices=list(config.PRESETS), default="quick")
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--session", default="")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="no keypresses (phantom gate / unattended)")
    ap.add_argument("--force", action="store_true",
                    help="allow SPACE past an unstable signal check")
    ap.add_argument("--seconds-per-cell", type=float, default=0,
                    help="override the preset dwell")
    ap.add_argument("--calib-blocks", type=int, default=0,
                    help="override CALIB_BLOCKS (testing)")
    args = ap.parse_args()
    if args.calib_blocks > 0:
        config.CALIB_BLOCKS = args.calib_blocks
        config.CALIB_RANK_MIN = min(config.CALIB_RANK_MIN,
                                    args.calib_blocks - 1)

    grid_w, grid_h, spc = config.PRESETS[args.preset]
    if args.seconds_per_cell > 0:
        spc = args.seconds_per_cell
    session = args.session or os.path.join(
        "runs", f"session_{time.strftime('%Y%m%d_%H%M%S')}")
    session = os.path.join(ROOT, session) if not os.path.isabs(session) \
        else session
    os.makedirs(session, exist_ok=True)

    target = targets.load_target(args.target, grid_w, grid_h)
    np.save(os.path.join(session, "target.npy"), target)

    log = SessionLog(os.path.join(session, "session.json"),
                     dict(source=args.source, preset=args.preset,
                          target=args.target, grid=[grid_w, grid_h],
                          seconds_per_cell=spc, port=args.port))

    if sys.platform == "win32":
        # keep display+system awake for the scan; restored in finally
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)

    pygame.init()
    flags = 0 if args.windowed else pygame.FULLSCREEN
    screen = pygame.display.set_mode(
        (1280, 720) if args.windowed else (0, 0), flags, vsync=1)
    pygame.display.set_caption("eyecam session")

    recorder = Recorder(args.source, session, args.port)
    phantom = None
    phantom_stop = None
    exit_code = 0
    try:
        recorder.start()
        log.event("source_start", "ok")
        if args.source == "phantom":
            phantom, phantom_stop = spawn_phantom(session, args.port)

        tail = EEGTail(os.path.join(session, "eeg.csv"))
        status, rate = stage_signal_check(screen, tail, recorder, args)
        log.event("signal_check", status, rate=round(rate, 1))
        if status != "ok":
            return 2

        calib = stage_calibrate(screen, session, log)
        if calib is None:
            log.event("calibrate", "abort")
            return 3
        log.event("calibrate", "ok", passed=calib["passed"],
                  best=calib["best"],
                  dprime=round(calib["channels"][calib["best"]]["dprime"], 2))

        status = stage_confirm(screen, calib, args)
        log.event("confirm", status)
        if status == "abort":
            return 3

        controls = ScanControls(tail, recorder, rate, log, auto=args.auto)
        scan_status = run_raster(
            screen, target, spc,
            os.path.join(session, "cursor_log.csv"), controls,
            auto=args.auto)
        log.event("scan", scan_status)
        if scan_status == "aborted":
            exit_code = 4  # partial data still gets reconstructed below

        draw_lines(screen, [("reconstructing...", None)])
        time.sleep(0.7)  # let the final packets land
        grid, r = reconstruct.run(
            session, calibration=os.path.join(session, "calibration.json"))
        log.event("reconstruct", "ok", r=None if r is None else round(r, 3))

        compare = build_compare_png(session, target, grid)
        stage_show_result(screen, session, compare, r, args)
        log.event("done", "ok")
        print(f"session -> {session}")
        if r is not None:
            print(f"reconstruction vs target: r = {r:.3f}")
        return exit_code
    except Exception as e:
        log.event("error", "exception", detail=repr(e))
        raise
    finally:
        recorder.stop()
        if phantom is not None:
            open(phantom_stop, "w").close()
            try:
                phantom.wait(timeout=4)
            except subprocess.TimeoutExpired:
                phantom.terminate()
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
