# eyecam — the human eye as a camera

Replication + extension scaffold for **Mann et al., "The Human Eye as a
Camera," IEEE HealthCom 2019** ([wearcam.org/eyecam.pdf](http://wearcam.org/eyecam.pdf)),
which grew out of the WearSys'19 abstract *"Eye itself as a camera: Sensors,
integrity, and trust"* (doi 10.1145/3325424.3330210).

**Idea:** a stimulus flickering at 15 Hz entrains an SSVEP (steady-state visual
evoked potential) in visual cortex. Record EEG while a guided cursor rasters
over a scene; the relative EEG power at 15 Hz (+ 30 Hz harmonic) at each cursor
position is a pixel. The eye itself becomes the sensor — the image is
reconstructed from brainwaves alone.

## One command (the full stack)

```
python run_session.py --source osc --preset quick --target "text:NO"
```

Six stages in one pygame window: recorder spawn → live signal check →
**90 s SSVEP calibration** (6× flicker-ON/black-OFF blocks; per-channel robust
d′ picks the channel weights and hard-gates a doomed scan) → guided-cursor
raster scan (SPACE pauses at row ends, ESC keeps partial data; auto-pause on
focus loss / stream dropout; recorder auto-respawn) → calibration-weighted
reconstruction with blink/drift-immune artifact rejection → result beside the
target with r. Everything logs to `<session>/session.json`.

Presets: `quick` 12×8 @4 s ≈ 6.4 min · `standard` 16×12 @5 s ≈ 16 min ·
`fine` 24×18 @6 s ≈ 43 min. Sources: `osc` (MuseLog phone app), `lsl`
(muselsl over laptop Bluetooth), `phantom` (hardware-free synthetic subject).

## Pipeline

| file | role |
|---|---|
| `config.py` | shared parameters (15 Hz stim, presets, calibration thresholds, Eq. 1 kernel) |
| `run_session.py` | the six-stage orchestrator above |
| `stimulus.py` | pygame guided-cursor flicker raster (eye-tracker-free mode of the paper); logs cursor position on the LSL clock |
| `osc_acquire.py` | records MuseLog OSC (`/muse/eeg` etc.) on UDP 5000 to CSV |
| `acquire.py` | records any LSL EEG stream (muselsl / Petal Metrics / BlueMuse) to CSV |
| `phantom_subject.py` | synthetic subject: tails the live stimulus logs, injects SSVEP at the *measured* flicker rate over real UDP |
| `tailer.py` | Windows-safe follow-a-growing-CSV reader |
| `reconstruct.py` | per cell × channel Welch PSD → (P15 + P30)/P(14–50) → weighted combine → Eq. 1 blend → PNG |
| `simulate.py` | offline synthetic session — validates the math without hardware |
| `gate_test.py` / `osc_gate.py` / `full_gate.py` | exact gates: offline sim · UDP ingest · full live stack |

Gates (2026-08-26): sim r = 0.92 vs shifted control r = 0.00; OSC round-trip
248/256 with 64 Hz decimated r = 0.886; full-stack phantom gate per
`full_gate.py`. The code also survived a 23-agent adversarial review
(13 confirmed findings, all fixed: gap-aware fs inference, drift-immune
artifact stats, flicker-verified phantom, auto-mode deadlocks, tailer
truncation, orphan protection).

## Run the simulation (no hardware)

```
python gate_test.py
python simulate.py --target "text:NO CAMERAS" --out-dir runs/demo
python reconstruct.py --session runs/demo
```

## Run a live session — option A: MuseLog app (OSC over Wi-Fi)

Uses your own MuseLog app (`museai`) as the
acquisition hub — no PC Bluetooth needed.

1. Phone and PC on the **same network** (DHCP moves the PC's IP around —
   the signal-check screen and `osc_acquire.py` print the current one).
2. `python run_session.py --source osc --preset quick --target "text:NO"`
3. In MuseLog: connect the Muse → OSC Streaming Settings → Target IP = the
   IP shown on the signal-check screen, Port = 5000, Format =
   **SnowballArcade** (the one that includes raw `/muse/eeg`), enable
   **Full-rate raw EEG** → Start Streaming. The on-screen sample rate must
   go green; SPACE continues.

The old manual three-terminal flow (osc_acquire + stimulus + reconstruct
separately) still works for debugging; decimated ~64 Hz streams are handled
(sample rate is inferred, gap-aware — verified by `python osc_gate.py`).

## Run a live session — option B: muselsl (LSL over PC Bluetooth)

1. `muselsl stream` in another terminal (muselsl is installed; pair the Muse
   in Windows Bluetooth settings first). Petal Metrics with LSL output also
   works.
2. `python run_session.py --source lsl --preset quick --target "text:NO"`

Timestamps everywhere are on the same LSL clock, so no manual sync needed.

## What the paper used that this scaffold approximates

- **Oz electrode:** they modified the Muse with an auxiliary occipital (Oz)
  electrode in a 3D-printed holder — that is where SSVEP is strongest. Stock
  TP9/TP10 will show the effect but weaker; expect to need longer dwells
  (`--seconds-per-cell 6`+) and a darkened room.
- **Head stabilization:** chin rest + high chair back.
- **Display:** they flag LCD pixel-onset jitter as an SNR limiter; use a
  high-refresh (120/144 Hz) display or strobe an LED for spectral purity.
- **Eye tracker (optional):** Tobii, used only to reject saccades/blinks in
  guided-cursor mode.

## Extension angle (ours, not theirs)

The paper's Future Directions explicitly propose VEP estimation via the
**chirplet transform** — i.e., ACT. Two concrete upgrades:

1. **Chirp-coded stimulus:** sweep the flicker (e.g., 12→18 Hz) and matched-
   filter the EEG with ACT instead of a fixed-bin FFT — better noise rejection,
   and the chirp signature is unambiguous evidence of entrainment.
2. **Code-division pixels:** give different scene regions different chirp
   codes and demix with ACT — several pixels per dwell instead of one, directly
   attacking the paper's acquisition-speed bottleneck (~106 s per scan line).
