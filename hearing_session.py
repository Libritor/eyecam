"""Hearing test on the Muse headband (ASSR), after Ali Sher's "Muse Hearing Test: Code
Outline" (27 Oct 2026) and Section V of the exam paper.

Three stages, the EEG recorded throughout by osc_acquire.py from MuseLog (OSC, port 5000):

  1. go/no-go   a 1 kHz tone amplitude-modulated at 40 Hz, blocks at 60, 50, 40, 30, 20 dB
                nominal (--gng-seconds each, 120 in the paper). Continue only if the loudest
                levels give a 40 Hz response (analysis/hearing_decode.py --quick).
  2. audiogram  behavioural thresholds at 500, 1000, 2000, 4000 Hz: 1 s tones at random
                0.8-2.0 s gaps from 40 dB, down 10 dB after a tone heard, up 5 after a miss,
                threshold = the lowest level heard on two ascending runs; ~20 % silent catch
                trials count false alarms. Press SPACE when you hear a tone.
  3. four-band  music (or pink noise) split into octave bands at 500, 1k, 2k, 4k Hz, each band
                modulated at its own rate (37, 39, 41, 43 Hz, full depth, envelope starts at 0),
                every band at the same nominal level, blocks at 60, 50, 40, 30, 20 dB
                (--band-seconds each, 60 in the paper), listened to passively.

Levels: 100 dB nominal = a full-scale sine; a band at L dB has an RMS of 10^((L-100)/20)/sqrt2 of
full scale. Nothing is acoustically calibrated, so thresholds are relative to this PC, this
volume and these speakers: set the system volume once and do not touch it.

Speakers, not earbuds: earbuds sit on TP9/TP10 and can induce the sound into the electrodes.

    python hearing_session.py --session runs/hearing1 [--music song.mp3] [--stages gng,audiogram,bands]
    python hearing_session.py --session runs/hearing1 --stages bands --attenuate 2000:20   (simulated loss)

Writes <session>/eeg.csv, events.csv (stage, block, level_db, band/freq, t_on, t_off),
audiogram.json, stimulus.json.
"""
import argparse
import csv
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ear_mic_session import Recorder, hold_awake, local_clock, read_audio, tail_eeg  # noqa: E402

FS = 48000
LEVELS = [60, 50, 40, 30, 20]
BANDS = [500, 1000, 2000, 4000]
TAGS = [37.0, 39.0, 41.0, 43.0]
MAX_BAND_DB = 85


def amp_for(level_db):
    """RMS (fraction of full scale) of a band at a nominal level."""
    return 10 ** ((level_db - 100) / 20) / np.sqrt(2)


def ramp(x, ms=20):
    n = int(ms / 1000 * FS)
    r = np.linspace(0, 1, n)
    x[:n] *= r
    x[-n:] *= r[::-1]
    return x


def am(x, fm, t):
    return x * (1 + np.sin(2 * np.pi * fm * t - np.pi / 2))     # full depth, envelope starts at 0


def go_nogo_block(level, seconds):
    t = np.arange(int(seconds * FS)) / FS
    tone = np.sin(2 * np.pi * 1000 * t)
    x = am(tone, 40.0, t)
    x *= amp_for(level) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return ramp(x.astype(np.float32))


def band_source(music, seconds, seed):
    n = int(seconds * FS)
    if music is not None:
        y = np.resize(music, n)
    else:                                                      # pink noise
        rng = np.random.default_rng(seed)
        w = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / FS)
        w[1:] /= np.sqrt(f[1:])
        w[0] = 0
        y = np.fft.irfft(w, n)
    return y.astype(np.float64)


def four_band_block(src, level, attenuate):
    t = np.arange(len(src)) / FS
    out = np.zeros_like(src)
    for fc, fm in zip(BANDS, TAGS):
        sos = butter(4, [fc / np.sqrt(2), fc * np.sqrt(2)], btype="band", fs=FS, output="sos")
        b = sosfiltfilt(sos, src)
        b = am(b, fm, t)
        lv = min(level - attenuate.get(fc, 0), MAX_BAND_DB)
        b *= amp_for(lv) / (np.sqrt(np.mean(b ** 2)) + 1e-12)
        out += b
    peak = np.abs(out).max()
    if peak > 0.99:
        raise SystemExit(f"block at {level} dB would clip (peak {peak:.2f}); lower the levels")
    return ramp(out.astype(np.float32))


def play(x):
    t_on = local_clock()
    sd.play(x, FS, blocking=True)
    return t_on, local_clock()


def check_stream(eeg_path, auto, timeout_s=180):
    print("\nwaiting for MuseLog (OSC, UDP 5000, full rate) ...")
    t0 = time.monotonic()
    stable = None
    while True:
        rate, names, sds = tail_eeg(eeg_path)
        ok = rate >= 200 and len(names) >= 4
        stable = (stable or time.monotonic()) if ok else None
        q = "  ".join(f"{n} {s:5.1f}" for n, s in zip(names, sds)) if sds else "no rows yet"
        print(f"  {rate:6.1f} Hz  {q}")
        if stable and time.monotonic() - stable >= 10:
            bad = [n for n, s in zip(names, sds) if n in ("TP9", "TP10") and s > 60]
            print("stream stable." + (f"  TP9/TP10 noisy ({', '.join(bad)} > 60 uV): press the ear pads in." if bad else ""))
            return
        if auto and time.monotonic() - t0 > timeout_s:
            raise SystemExit("no stable stream from MuseLog")
        time.sleep(1.0)


def key_pressed_since(t_start):
    """Windows console: True if SPACE was pressed since t_start (drains the buffer)."""
    import msvcrt
    hit = False
    while msvcrt.kbhit():
        if msvcrt.getwch() == " ":
            hit = True
    return hit


def audiogram(writer, log):
    import msvcrt
    results = {}
    print("\nAUDIOGRAM: press SPACE every time you hear a beep, as soon as you hear it.")
    input("Press Enter to start ... ")
    while msvcrt.kbhit():
        msvcrt.getwch()
    for f in BANDS:
        level, heard_asc, last_dir, trials, fa, catches = 40, {}, None, 0, 0, 0
        while trials < 40:
            time.sleep(random.uniform(0.8, 2.0))
            if random.random() < 0.2:                           # catch trial: silence
                catches += 1
                t0 = local_clock()
                time.sleep(1.0 + 1.5)
                if key_pressed_since(t0):
                    fa += 1
                writer.writerow(["audiogram", f"catch_{catches}", "", f, f"{t0:.6f}", f"{local_clock():.6f}", "catch", "fa" if fa else ""])
                continue
            t = np.arange(FS) / FS
            x = ramp((amp_for(level) * np.sqrt(2) * np.sin(2 * np.pi * f * t)).astype(np.float32))
            while msvcrt.kbhit():
                msvcrt.getwch()
            t_on, t_off = play(x)
            time.sleep(1.0)
            heard = key_pressed_since(t_on)
            trials += 1
            writer.writerow(["audiogram", trials, level, f, f"{t_on:.6f}", f"{t_off:.6f}", "tone", "heard" if heard else "missed"])
            if heard:
                if last_dir == "up":                            # heard on an ascending run
                    heard_asc[level] = heard_asc.get(level, 0) + 1
                    if heard_asc[level] >= 2:
                        break
                level -= 10
                last_dir = "down"
            else:
                level += 5
                last_dir = "up"
            if level > 80:
                break
        thr = min([lv for lv, n in heard_asc.items() if n >= 2], default=None)
        results[f] = dict(threshold_db=thr, trials=trials, catch_trials=catches, false_alarms=fa)
        print(f"  {f} Hz: threshold {thr} dB nominal ({trials} tones, {fa}/{catches} false alarms)")
        log(f"audiogram {f} {thr}")
    return results


def quest_run(a, stages, levels, band_levels, att, music, session):
    """Render go/no-go and four-band blocks into one clip at 22.05 kHz (bands reach 5.7 kHz),
    play it once through the Quest browser, and log each block at its offset in the clip."""
    global FS
    import soundfile as sf
    import subprocess
    from quest_audio import QuestPlayer, adb_open
    FS = 22050
    gap = np.zeros(int(2.0 * FS), np.float32)
    parts, plan, t = [], [], 0.0
    if "gng" in stages:
        for k, lv in enumerate(levels, 1):
            x = go_nogo_block(lv, a.gng_seconds)
            plan.append(("gng", k, lv, 1000, t, t + len(x) / FS, "tone40", ""))
            parts += [x, gap]
            t += (len(x) + len(gap)) / FS
    if "bands" in stages:
        mus = None
        if music is not None:
            from scipy.signal import resample_poly
            mus = resample_poly(music, 147, 320)             # 48 kHz -> 22.05 kHz
        for k, lv in enumerate(band_levels, 1):
            x = four_band_block(band_source(mus, a.band_seconds, k), lv, att)
            plan.append(("bands", k, lv, "500/1000/2000/4000", t, t + len(x) / FS, "four_band",
                         ";".join(f"{b}:-{v:g}" for b, v in att.items())))
            parts += [x, gap]
            t += (len(x) + len(gap)) / FS
    clip = np.concatenate(parts)
    path = os.path.join(session, "clip.wav")
    sf.write(path, clip, FS, subtype="PCM_16")
    print(f"clip: {len(clip) / FS / 60:.1f} min, {os.path.getsize(path) / 1e6:.0f} MB")
    out = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True).stdout
    qs = next((l.split()[0] for l in out.splitlines() if "model:Quest" in l), None)
    if qs:
        subprocess.run(["adb", "-s", qs, "shell", "am", "force-stop", "com.oculus.browser"], capture_output=True)  # no stale tab
    player = QuestPlayer(path, a.quest_port)
    serial = adb_open(a.quest_port)
    print(f"page pushed to the Quest ({serial or 'open http://127.0.0.1:%d/ in the headset' % a.quest_port}); "
          "in the headset, pull the trigger to start")
    if not player.wait_ready(1800):
        raise SystemExit("the headset page never reported ready")
    player.text("silence first: sit still, jaw relaxed, eyes on one spot")
    return player, plan, len(clip) / FS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=os.path.join("runs", "hearing1"))
    ap.add_argument("--music", help="track for the four-band stage (default: pink noise)")
    ap.add_argument("--music-start", type=float, default=30.0)
    ap.add_argument("--stages", default="gng,audiogram,bands")
    ap.add_argument("--gng-seconds", type=float, default=120.0)
    ap.add_argument("--band-seconds", type=float, default=60.0)
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--band-levels", default=None, help="levels for the four-band stage (default: --levels)")
    ap.add_argument("--attenuate", default="", help="simulated loss, e.g. 2000:20 (band Hz : dB)")
    ap.add_argument("--baseline", type=float, default=30.0, help="silence before the first block (s)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--player", choices=["pc", "quest"], default="pc",
                    help="quest: the headset's browser plays the whole sequence (go/no-go + bands) as one clip, "
                         "timed on the PC clock (quest_audio.py); the button audiogram is skipped")
    ap.add_argument("--quest-port", type=int, default=8083)
    ap.add_argument("--no-recorder", action="store_true")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    levels = [int(v) for v in a.levels.split(",")]
    band_levels = [int(v) for v in a.band_levels.split(",")] if a.band_levels else levels
    att = {int(k): float(v) for k, v in (p.split(":") for p in a.attenuate.split(",") if p)}
    if a.device is not None:
        sd.default.device = (None, int(a.device) if a.device.isdigit() else a.device)
    os.makedirs(a.session, exist_ok=True)
    music = None
    if a.music:
        x, fs = read_audio(a.music)
        from scipy.signal import resample_poly
        from math import gcd
        if fs != FS:
            g = gcd(fs, FS)
            x = resample_poly(x, FS // g, fs // g)
        music = x[int(a.music_start * FS):]
    dev = sd.query_devices(kind="output")
    meta = dict(stages=stages, levels_db=levels, band_levels_db=band_levels, bands_hz=BANDS, tags_hz=TAGS, gng_seconds=a.gng_seconds,
                band_seconds=a.band_seconds, attenuate=att, music=os.path.abspath(a.music) if a.music else "pink noise",
                device=dev["name"], reference="100 dB nominal = full-scale sine; not acoustically calibrated",
                design="Ali Sher, Muse Hearing Test: Code Outline, 27 Oct 2026",
                started=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(os.path.join(a.session, "stimulus.json"), "w"), indent=1)
    print(f"output: {dev['name']}.  Set the volume once (about half) and leave it.  Stages: {', '.join(stages)}")
    est = (len(levels) * a.gng_seconds if "gng" in stages else 0) + (len(band_levels) * a.band_seconds if "bands" in stages else 0)
    print(f"about {est / 60 + (4 if 'audiogram' in stages else 0) + a.baseline / 60:.0f} min")

    hold_awake()
    rec = None if a.no_recorder else Recorder(a.session, a.port)
    lines = []
    log = lambda s: lines.append(f"{time.strftime('%H:%M:%S')} {s}")   # noqa: E731
    try:
        check_stream(os.path.join(a.session, "eeg.csv"), a.auto)
        if not a.auto:
            input("\nSit still, jaw relaxed, eyes open on one spot, and stay alert. Enter to start ... ")
        with open(os.path.join(a.session, "events.csv"), "w", newline="", buffering=1) as f:
            w = csv.writer(f)
            w.writerow(["stage", "block", "level_db", "freq_hz", "t_on", "t_off", "kind", "note"])
            if a.player == "pc":
                t0 = local_clock()
                print(f"\nsilence {a.baseline:.0f} s")
                time.sleep(a.baseline)
                w.writerow(["baseline", 0, "", "", f"{t0:.6f}", f"{local_clock():.6f}", "silence", ""])
            if a.player == "quest":
                player, plan, dur = quest_run(a, stages, levels, band_levels, att, music, a.session)
                t0 = local_clock()
                print(f"\nsilence {a.baseline:.0f} s")
                time.sleep(a.baseline)
                w.writerow(["baseline", 0, "", "", f"{t0:.6f}", f"{local_clock():.6f}", "silence", ""])
                print(f"playing {dur / 60:.1f} min through the headset: tone blocks, then the tagged hiss. Just listen.")
                player.text("listen: a buzzing tone, then a hiss, getting quieter")
                t_on, t_off, info = player.play(1, timeout=dur + 30)
                for st, k, lv, fq, a0, a1, kind, note in plan:
                    w.writerow([st, k, lv, fq, f"{t_on + a0:.6f}", f"{t_on + a1:.6f}", kind, note])
                print(f"played {t_off - t_on:.0f} s (latency {info['output_latency_s'] * 1000:.0f} ms, clock rtt {info['rtt_s'] * 1000:.1f} ms)")
                player.text("done: thank you")
                player.done()
                stages = []                                      # nothing else to run here
            if "gng" in stages:
                print("\nGO/NO-GO: a buzzing 1 kHz tone, getting quieter. Just listen.")
                for k, lv in enumerate(levels, 1):
                    t_on, t_off = play(go_nogo_block(lv, a.gng_seconds))
                    w.writerow(["gng", k, lv, 1000, f"{t_on:.6f}", f"{t_off:.6f}", "tone40", ""])
                    print(f"  block {k}: {lv} dB, {t_off - t_on:.0f} s")
                    time.sleep(2.0)
            if "audiogram" in stages:
                res = audiogram(w, log)
                json.dump(res, open(os.path.join(a.session, "audiogram.json"), "w"), indent=1)
            if "bands" in stages:
                print("\nFOUR BANDS: tagged music, getting quieter. Just listen, stay alert.")
                for k, lv in enumerate(band_levels, 1):
                    x = four_band_block(band_source(music, a.band_seconds, k), lv, att)
                    t_on, t_off = play(x)
                    w.writerow(["bands", k, lv, "500/1000/2000/4000", f"{t_on:.6f}", f"{t_off:.6f}", "four_band",
                                ";".join(f"{b}:-{v:g}" for b, v in att.items())])
                    print(f"  block {k}: {lv} dB, {t_off - t_on:.0f} s")
                    time.sleep(2.0)
            t1 = local_clock()
            time.sleep(10)
            w.writerow(["baseline", 1, "", "", f"{t1:.6f}", f"{local_clock():.6f}", "silence", ""])
    except KeyboardInterrupt:
        print("\nstopped early; what was recorded is kept")
    finally:
        sd.stop()
        if rec:
            rec.stop()
    print(f"\ndone. Decode with:\n  python analysis/hearing_decode.py {a.session}")


if __name__ == "__main__":
    main()
