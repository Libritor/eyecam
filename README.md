# EyeCam: the human eye as a camera

This is a working build of Mann et al., *The Human Eye as a Camera* (IEEE HealthCom 2019).
A flickering stimulus evokes an SSVEP (steady-state visual evoked potential) in the visual
cortex. EyeCam reads it with a Muse EEG headband and turns it into pictures of what the
subject is looking at. Every experiment in the paper has its own mode.

## Step by step: taking a picture with your eye

Total time is about 20 minutes: 5 to set up, 3 to calibrate, and 8–16 to scan a letter N.

### 1. Prepare the headset
1. **Charge it fully.** A nearly flat Muse can drop into its bootloader (firmware-update
   mode) and stop sending EEG. The app detects this and tells you.
2. **Close the Muse phone app.** The headset accepts only one connection at a time.
3. **Wet the four sensors slightly** (water or saline). Dry sensors are the main cause of noise.
4. Put it on **snug**, forehead sensors on bare skin. **Push hair away from behind your
   ears** so the rubber TP9/TP10 pads touch skin. These two channels carry the image.

### 2. Prepare yourself and the room
- Sit about 50 cm from the screen with your **head supported** (headrest or a wall behind
  you). The paper used a chin rest.
- **Relax your jaw.** TP9/TP10 sit on the jaw muscle, and clenching, talking or
  swallowing floods the signal.
- Dim the room. Turn the screen brightness up.

### 3. Start the app
```
cd eyecam
python server.py            # or double-click start.bat; opens http://localhost:8765
```
Use **Chrome or Edge**, since only they support Web Bluetooth and Web Serial.

### 4. Connect
Click **Connect Muse** (top right) and pick your Muse (e.g. *Muse-0357*). The status
should read **"streaming from Muse-… (Muse 2 / S gen 1)"** at about 256 Hz.
If it won't connect, see *Troubleshooting* below. The fallback is
`python server.py --muse` and then the **Python bridge** button.

### 5. Check the signal (*Signal monitor* tab)
- All four contact dots should be **green**. TP9 and TP10 should read below roughly 60 µV.
  If not, re-wet, adjust the headset and wait 1–2 minutes for the sensors to settle.
- Optional: tick **Flicker on** and stare at the patch for 10 s. A spike at the flicker
  frequency should appear in the spectrum.

### 6. Calibrate (*Calibration* tab) — about 2½ minutes
1. Wait until the panel says **"Contact OK — ready"** or **"Ready with TP9"** (or TP10),
   then press **Start calibration**. One ear with stable contact is enough. A channel that
   keeps flickering no longer restarts the countdown. The analysis uses the better ear and
   down-weights the other.
2. The screen goes fullscreen. Press **SPACE**, then **keep your eyes on the dot**. A
   square flashes on and off around it for 80 s. Stay still.
3. Follow the spoken prompts: rest, blink, clench your jaw, swallow, turn your head.
4. Read the **report**:
   - **GO**: your brain response is measurable. Continue.
   - **MARGINAL**: it works but slowly. Use the plan it gives (2+ passes), or improve
     contact first.
   - **NO-GO**: no usable signal. **Don't scan yet.** Fix contact (step 1) and recalibrate.
5. Click **Apply to analysis**. This sets the analysis to the **Kalman-tracked SSVEP**
   with masking off, plus the spatial filter if calibration found that it helps.
   Optionally click **Save calibration**. Calibration is remembered by this browser.

### 7. Take the picture (*Eye camera (raster)* tab)
1. Click **Use calibrated scan plan**. This loads the letter **N**, the exact flicker
   frequency for your display, and the cursor speed and number of passes.
   (Without calibration: preset **Letter N**, and click **Use 14.49 Hz** if a refresh-rate
   warning appears.)
2. Press **Start imaging**. The screen goes fullscreen.
3. For each scan line, press **SPACE**, then **follow the white dot** inside the flashing
   square with your eyes until it stops. Rest and blink between lines.
   - **R** in a break redoes the previous line.
   - **ESC** finishes early. The image is built from the lines you completed.

### 8. Read and save the result
- The page shows the letter you looked at next to the **image reconstructed from your EEG**.
  Below it:
  - **validity**: how well the image matches the shown letter; above about 0.6 is readable.
  - **split-half reliability**: how consistent the passes are.
  - how much data was **masked** as artifact.
- After calibration the image uses the **Kalman tracker** (see
  [Kalman tracking](#kalman-tracking-of-the-ssvep)). To compare it with the paper's method,
  set **SSVEP metric** → *Paper §III* and **Artifact masking** → *Calibrated thresholds*.
  The image re-renders at once.
- Click **Auto-tune analysis** to try FBCCA, coherent lock-in, the Kalman tracker,
  masking and the spatial filter on your own recording, and apply whichever scores best.
- Click **Save session (.json)**, which keeps raw EEG plus the stimulus log, and
  **Save image (.png)**.

### 9. Reprocess later
**Review & reprocess** → load the `.json` → change any analysis setting and re-render.

### No headset?
Click **Simulator**. It produces synthetic Muse-like EEG (including realistic artifacts)
whose SSVEP follows whatever flickers on screen, so every step above works end to end.
**Review → Build a synthetic demo session** shows a finished reconstruction straight away.

### Troubleshooting
| Symptom | Cause and fix |
|---|---|
| "This Muse is in its BOOTLOADER" | Firmware-update mode, usually after a flat battery. Charge it, open the official Muse app to finish or restore the firmware, close the app, connect again. |
| "No Characteristics matching UUID …" / headset disconnects while connecting | Windows has a stale Bluetooth cache. Remove the Muse in Windows Bluetooth settings and at `chrome://settings/content/bluetoothDevices`, power-cycle it, reconnect. Or use `python server.py --muse`. |
| Contact never turns green | Re-wet the sensors, clear hair behind the ears, tighten the band, wait 1–2 min. |
| One ear (e.g. TP10) keeps flickering red/green | You can still calibrate once the other ear is stable. To fix it, read the channel's hint. *Large swings* means the pad is moving: re-seat it on bare skin just above and behind the ear and make the band snug. *Muscle/interference* means relax your jaw, or the pad is touching hair. A pad that never settles may need a drop of saline. |
| Calibration says NO-GO | The signal is too weak to image. Improve contact. For the biggest gain, add an Oz electrode on the AUX port (see below). |
| Warning that the display can't make 15 Hz | Your monitor's refresh (e.g. 144.9 Hz) isn't a multiple of 30. Click **Use 14.49 Hz**, or set Windows to 60/120 Hz. |
| Image is just noise | Check validity. If it's below about 0.3, scan slower or use more passes (more data per pixel), and recalibrate after fixing contact. |
| "Quick demo" seems to do nothing | Presets only *load settings*. Press **Start imaging**. |

### Three ways to get EEG in

| Button | How | When |
|---|---|---|
| Connect Muse | Web Bluetooth, straight from the page. Detects the protocol: legacy (preset `p21`, or `p20` with AUX) or Muse S **Athena** (multiplexed 14-bit packets, preset `p1041`) | Muse 2016 / Muse 2 / Muse S gen 1 / Muse S Athena |
| Python bridge | `python server.py --muse` runs `muselsl stream`, and the server forwards its LSL outlet over a WebSocket (`--model athena` forces the Athena protocol) | Web Bluetooth won't pair, or you already use muselsl |
| Python bridge | `python server.py --lsl` joins any LSL `type=EEG` stream (BlueMuse, `mind2motor/run_muse.py --live`, …) | Something else already owns the headset |

For better signal, plug an extra electrode into the Muse's micro-USB AUX port, place it
over **Oz** (back of the head, on the midline), and tick **AUX/Oz** before connecting.
That is the paper's setup. The Muse S Athena has no AUX port, so it always uses TP9/TP10. Run Calibration afterwards: if
the Oz channel carries the strongest SSVEP, calibration selects it (Channels = AUX).
*Channels = auto* otherwise means TP9 + TP10, which pick up a weaker occipital SSVEP.

## Calibrate first (noise handling)

On a Muse, signal quality is the bottleneck. A first real recording had almost no SSVEP,
and only 15 % of its 4 s windows were artifact-free. So **Calibration** (about 2½ min) runs
before imaging:

1. **Contact gate.** TP9 or TP10 must stay below 100 µV peak-to-peak per second (and low
   EMG) for at least 85 % of the last 10 s. One stable ear is enough.
2. **SSVEP check.** 8 × (5 s flicker / 5 s off) while fixating a dot. Measures your SSVEP
   amplitude and detectability d′ per channel, then trains a **spatial filter**: a
   generalized eigenvector over all channels that can subtract the forehead-reference noise
   shared by TP9/TP10 and AF7/AF8. The filter is cross-validated (train on odd cycles, test
   on even) and adopted only if it wins on held-out data.
3. **Artifact prompts.** Rest, blink, clench, swallow and turn your head. These set
   personal thresholds for the **artifact mask**, which ignores only contaminated 0.25 s
   blocks (muscle above 70 Hz, electrode shifts) instead of rejecting whole windows.
4. **Report.** GO / MARGINAL / NO-GO, plus a scan plan for the letter N. A readable N
   needs d′·√(seconds per pixel) ≈ 3.5, a figure taken from `noisebench.html`.

Imaging then offers **repeat passes** (alternating direction, median-combined) and optional
**per-line reference patches**. Every scan-line result reports **split-half reliability**
(even vs odd passes; needs no ground truth) and **validity** against the displayed stimulus.
**Auto-tune** scores metric × mask × spatial-filter variants and applies the best.

What the simulated benchmark showed (`noisebench.html`, Muse-like artifacts):
- Scan time and SSVEP strength dominate everything else.
- Masking gives a small, mixed gain (about +0.04 validity on average).
- The spatial filter helps a lot (×6.7 d′) only when reference noise dominates.
  Calibration measures whether that is true for you.
- FBCCA helps at very weak signals.
- The Kalman tracker (below) beat all of these.

### Kalman tracking of the SSVEP

The windowed metrics compute each image column from its own fixed window. When an artifact
hits, the whole window is lost, or the mask shrinks it. Metric **Kalman-tracked amplitude**
(`web/js/kalman.js`) instead treats the SSVEP as a hidden state that changes slowly, like
the intended velocity in a BCI decoder:

- **Observation.** Once per flicker cycle (1/15 s), the EEG is demodulated against the
  logged flicker phase. The result is the response's complex amplitude at the fundamental
  and its 2nd harmonic.
- **Model.** The amplitude follows a smooth trend (an integrated random walk). Each
  cycle's measurement noise comes from the local EEG variance, so muscle bursts and
  electrode pops are trusted less in proportion to how noisy they are. Nothing is cut
  out with a hard threshold.
- **Smoothing.** A forward Kalman filter plus a backward Rauch–Tung–Striebel pass gives a
  zero-lag estimate at every cycle. It runs once over the whole recording. The power is
  bias-corrected (|x|² minus its posterior variance), so dark areas sit near zero.

Benchmark: `noisebench.html`, Muse-like artifacts, 10-line letter N, 3 seeds × 3 scan
plans per level. Validity is the correlation with the shown letter:

| SSVEP at Oz | Paper metric + calibrated mask (previous default) | FBCCA + mask + spatial filter | **Kalman, mask off** |
|---|---|---|---|
| 1 µV | 0.16 | 0.05 | **0.28** |
| 2 µV | 0.55 | 0.35 | **0.72** |
| 3 µV | 0.78 | 0.70 | **0.89** |

- **Accuracy.** Kalman beat the previous default in 24 of 27 scans (mean +0.13 validity) and
  in all 9 at 3 µV. Split-half reliability roughly doubled at 2–3 µV. At 1 µV it helps, but
  the letter is still barely readable.
- **Speed.** It is about 10× faster than FBCCA.
- **Default after calibration**, with masking off. Adding the hard mask on top, plus a
  spatial filter that calibration had not adopted, made it worse (0.45 at 2 µV), because
  masked gaps throw away data the soft weighting already handles. A spatial filter that
  calibration does adopt (it won on held-out data) is still used.
- **Settings.** *Kalman smoothing* is the time constant; 0 derives it from the window, and
  larger means less noise but softer edges. *Kalman model* is order 2 (sharper, the
  default) or order 1. *Project on response phase* is an option on scan views.
- **Limits.**
  - These numbers come from the simulator; real Muse data is not tested yet. Auto-tune
    will show whether the gain holds for your recordings.
  - It does not beat the cursor-size blur. It gives a cleaner image for the same scan
    time.

## The experiments (paper → mode)

| Paper | Mode | What it does |
|---|---|---|
| §III, Fig. 1, 7, 8, 9 | **Eye camera (raster)** | A 100 px cursor slides across the image at 15 px/s, flashing yellow/blue at 15 Hz wherever the image is non-black. Each pixel is (P15+P30)/P(14–50 Hz), from the PSD over the 1700 samples (6.67 s) the cursor takes to pass it. The 48 overlapping scan lines are merged with eq. (1), f = 2x + x₁ + x₋₁ + (x₂+x₋₂)/2, then interpolated. Presets: *NO CAMERAS* (≈85 min, as in the paper), *face* (displayed rotated 90°, rotated back afterwards, ×2 gain), and *quick demo*. You can rest between lines (SPACE); R redoes a line. |
| §III, Fig. 6 | **Mind's eye (free gaze)** | The whole image flickers at 12 Hz while the eyes raster across it. Eye position comes from a Tobii (`server.py --tobii`), a webcam (WebGazer), the mouse, or a guiding cursor (no tracker). A 1.5 Hz 6th-order Butterworth filter smooths eye X/Y. The result is Delaunay-interpolated and shown in viridis with pixel axes. Optional speed weighting and blink rejection come from the paper's future-work section. |
| §II, Fig. 4 | **Visual field (metaveillance)** | Four light squares flicker at 12 Hz and move through the visual field. The live SSVEP sets an RGB LED from blue (weak) to red (strong). *On-screen sweep* needs no hardware; covert vs overt attention gives a narrow beam vs a broad cone, and it reports width in degrees and left/right hysteresis. The *3D plotter* option drives GRBL through vertical passes at 5 mm/s while retreating 0.5 mm/s from 4 to 21 cm. It then renders the side-view long exposure and the interpolated metaveillograph (a virtual plotter is included). |
| Fig. 10, 11 | **Real-world rig** | Operator console for the two-tripod rig: metronome, spoken cues, pipe height per line, a marker showing where the light should be right now, and the Arduino flicker lamp. Reconstruction is the same as the raster mode. |
| Fig. 12 | **Shutter-glasses chunks** | The scene is split into chunks, and each is fixated for 8 s under 15 Hz shutter glasses (Arduino-driven), cued by voice. You get one pixel per chunk. There is also an on-screen emulation. |
| Fig. 13, 14 | **SSVEPVMP memory prosthetic** | The world flickers at 15 Hz (shutter glasses, flicker lamp, or a webcam video see-through). The SSVEP is tracked continuously, and moments of high SSVEP trigger a head-camera snapshot and are logged as detections. SPACE marks the moments you find interesting. The result is the normalized SSVEP trace with red bands at your marks and a hit rate (the paper reports 89 %). |
| §III ("optimized…") | **Stimulus tuner** | Blocks of frequency × colour pairs, ranked by SNR. You can make the best one the default. |
| §I-A, Fig. 2, 3 | **Camera metaveillance** | A video-feedback loop: whatever the camera under test sees drives the lamp (red bias → green → blue), with a time constant that produces the paper's response/"desponse" hysteresis. There is also a long-exposure camera (lighten/add) for photographing the lamp, or the brain-modulated LED of Fig. 4. |
| — | **Review & reprocess** | Load saved sessions (`.json` holds raw EEG + stimulus log) and re-render them with any metric, window, channel set, interpolation or colormap. Export PNG / CSV. |

**Analysis** (top bar) sets the defaults every mode uses: channels, metric (the paper's
§III harmonic ratio, its §II "12 Hz vs rest" ratio, neighbour-bin SNR, absolute power,
FBCCA, lock-in, coherent lock-in or the Kalman tracker), PSD method, harmonics, latency compensation and artifact rejection. Every result
view can override them afterwards.

## Hardware (optional)

- **Arduino**: flash `firmware/eyecam_io/eyecam_io.ino`, then *Hardware → Connect*. It
  drives:
  - the brain-modulated RGB LED (pins 9/10/11),
  - a flickering lamp through a MOSFET (pin 6),
  - LC shutter glasses, with each lens across pins 2/3 and 4/5 and alternating polarity.

  Serial protocol: `C r g b`, `F hz`, `L 0|1`, `S hz`, `X`.
- **GRBL plotter**: *Hardware → Connect*. Jog the display to the eye position, click
  *Zero here (eye)*, and pick the axis mapping in the Visual-field mode.
- **Phone as the moving display (Fig. 4)**: run `python server.py --lan` and open the
  printed `http://<pc-ip>:8765/flicker.html?f=12&pattern=quad` on the phone. `flicker.html`
  also works as a full-screen light source for the rig.

## Files

```
server.py                  static server + LSL/Tobii → WebSocket bridge (+ muselsl supervisor)
firmware/eyecam_io/        Arduino sketch for LED / lamp / shutter glasses
web/index.html             the app
web/flicker.html           standalone flicker source (phone / 2nd screen)
web/tests.html             DSP, decoder, masking, spatial-filter and reconstruction self-tests
web/bench.html             SSVEP metric benchmark; web/noisebench.html: calibration + noise-handling benchmark
web/js/artifacts.js        0.25 s artifact masking (auto or calibrated thresholds)
web/js/kalman.js           Kalman/RTS tracking of the SSVEP amplitude (metric 'kalman')
web/js/calib.js            calibration analysis: SSVEP d′, cross-validated spatial filter, thresholds, scan plan
web/smoke.html             drives the whole UI with the simulator (every mode, live)
web/js/dsp.js              FFT, Welch/periodogram, SSVEP metrics, Butterworth, filtfilt
web/js/recon.js            eq. (1), scan-line / gaze / chunk / timeline / field reconstruction
web/js/interp.js           Delaunay, interpolation, colormaps
web/js/sources.js          Muse Web Bluetooth, Python bridge client, simulator
web/js/stage.js            fullscreen stimulus stage + frame-locked flicker clock
web/js/modes/*.js          one file per experiment
```

## Tested / not tested

- `tests.html` passes 19/19. It checks the FFT/PSD scaling, the Butterworth response
  (−3 dB at 1.5 Hz), filtfilt, Delaunay and the eq. (1) weights. It also runs end-to-end
  reconstructions from synthetic EEG: raster r = 0.80, free-gaze r = 0.76 and chunks
  r = 0.74 against ground truth, a VMP hit rate of 100 %, and the field-profile peak.
  The Kalman tracker is checked on an amplitude step through a muscle burst (0.31 / 2.35 /
  1.74 µV for true 0 / 2 / 2).
- `smoke.html` runs every mode live against the simulator with no errors.
- The Python bridge was tested with a fake LSL Muse stream.
- **Not yet tested with a real headset.** The Web Bluetooth path follows muselsl's protocol
  exactly. If it will not pair, use `python server.py --muse`, the same muselsl route that
  `mind2motor` already uses on this machine.

Practical notes:
- LCD pixel timing makes the flicker less pure; the paper points this out too.
  Frequencies that divide your refresh rate evenly give exact square waves. Signal monitor
  lists them (at 60 Hz: 6, 7.5, 10, 12, 15, 20, 30 Hz).
- A dry-electrode Muse without Oz gives a weaker SSVEP than the paper's Oz setup. If
  images look noisy, use slower cursors or longer windows, and try the tuner.
