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

## Pipeline

| file | role |
|---|---|
| `config.py` | shared parameters (15 Hz stim, 256 Hz Muse, 14–50 Hz noise band, Eq. 1 kernel) |
| `stimulus.py` | pygame guided-cursor flicker raster (eye-tracker-free mode of the paper); logs cursor position on the LSL clock |
| `acquire.py` | records any LSL EEG stream (muselsl / Petal Metrics / BlueMuse) to CSV |
| `reconstruct.py` | Welch PSD per cell → (P15 + P30)/P(14–50) → Eq. 1 vertical blend → PNG |
| `simulate.py` | synthetic session (1/f noise + alpha + luminance-driven SSVEP) — validates everything without hardware |
| `gate_test.py` | **exact gate**: sim reconstruction r ≥ 0.60 AND shifted-EEG control r ≤ 0.30 |

Verified 2026-08-23: `python gate_test.py` → aligned r = 0.928, control
r = −0.010, **GATE PASS**. "NO CAMERAS" sim reconstructs at r = 0.733.

## Run the simulation (no hardware)

```
python gate_test.py
python simulate.py --target "text:NO CAMERAS" --out-dir runs/demo
python reconstruct.py --session runs/demo
```

## Run a live session (Muse)

1. Stream the Muse to LSL — one of:
   - `pip install muselsl` then `muselsl stream` (Windows: needs the bleak
     backend; pair the Muse in Windows Bluetooth settings first)
   - Petal Metrics desktop app with LSL output enabled
2. Terminal A: `python acquire.py --session runs/live1`
3. Terminal B: `python stimulus.py --target "text:NO" --session runs/live1 --seconds-per-cell 6 --windowed`
   (drop `--windowed` for the real run; fullscreen + vsync gives cleaner flicker)
4. `python reconstruct.py --session runs/live1 --channels TP9,TP10`

Timestamps from both files are on the same LSL clock, so no manual sync needed.

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
