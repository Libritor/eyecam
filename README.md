# eyecam: the human eye as a camera

Reconstructing what a person is looking at from their brainwaves, with a
consumer EEG headband, on a laptop or inside a VR headset.

This is a replication and extension of **Mann et al., "The Human Eye as a
Camera," IEEE HealthCom 2019** ([wearcam.org/eyecam.pdf](http://wearcam.org/eyecam.pdf)),
which grew out of the WearSys'19 abstract *"Eye itself as a camera: Sensors,
integrity, and trust"* (doi 10.1145/3325424.3330210). It was built in Steve
Mann's lab at the University of Toronto by Alexander Vicol.

![Colour flag and grey image reconstructed from EEG in VR](docs/figures/fig5_run2_images.png)

*Run of 2026-09-29, Quest 3S browser, Muse 2 with an Oz auxiliary electrode.
Bottom row: a red/green/blue flag shown with each colour flickering at its own
frequency, and the colour image rebuilt from three spectral lines in the EEG;
all 24 cells come back with the right dominant colour.*

## The idea, and the paper behind it

A light that flickers at a steady rate makes visual cortex oscillate at the
same rate. That response, the **steady-state visual evoked potential (SSVEP)**,
shows up in EEG over the back of the head as a narrow spectral line at the
flicker frequency, and it is larger when the flickering patch is brighter.

Mann and colleagues turned that into a camera. A small flickering square is
moved over a scene while the subject follows it with their eyes. At each
position the strength of the EEG line is one pixel. The eye and brain are the
sensor; the picture is assembled from brainwaves alone. The paper used a Muse
headband modified with an extra occipital (Oz) electrode in a 3D-printed
holder, a 100 by 100 pixel square, about 6.7 s per point, 15 Hz flicker (moved
up from 12 Hz to stay clear of the alpha rhythm), a chin rest, and 48
overlapping scan lines blended by a small vertical kernel (their Eq. 1). It
also names the chirplet transform as a future direction for estimating the
evoked response.

The point of the original work is as much about integrity and trust as about
imaging: if the eye itself is the camera, the record of what was seen is tied
to a living observer.

## What this repository adds

- **A complete, gated pipeline** from stimulus to image, with a hardware-free
  synthetic subject as a positive control and shifted-EEG reconstructions as a
  negative control for every result.
- **A browser stimulus that runs anywhere**, including the Meta Quest browser,
  driven by a Python process on a PC. No app install on the headset.
- **An exact-frequency line detector** with an exact permutation test, which is
  about eight times more sensitive than 1 Hz-bin band power and is what first
  found the response on a Muse.
- **Subject calibration tools**: an electrode check (eyes-closed alpha), a
  frequency sweep that finds where this subject actually responds, and a
  delivery check that voids a run if the flicker was not really on screen.
- **Frequency-tagged colour**: red, green and blue flicker at three different
  frequencies at once, and the three EEG lines become the three colour planes.
- **A 40 Hz-tagged music stage** for the auditory steady-state response
  ("ear as a level meter").
- An honest record of what works and what does not yet.

## What you need

| item | notes |
|---|---|
| Muse 2 / Muse S / Muse Athena | any model that streams raw EEG |
| **Oz auxiliary electrode** | the Interaxon aux cup on micro-USB (Muse 2/S) or USB-C (Athena). This matters: without it we never detected a response. Place it two finger-widths above the bump at the back of the skull, on the midline, hair parted |
| Phone with **Mind Monitor** (or the lab's MuseLog app) | streams `/muse/eeg` over OSC/UDP, with the aux channel enabled |
| PC with Python 3.10+ | runs the driver and the analysis |
| Display | a laptop screen, or a Quest headset on the same Wi-Fi |

```
pip install -r requirements.txt
```

**Safety.** The stimulus is full-contrast flicker between 7 and 20 Hz. Do not
use it with anyone who has photosensitive epilepsy or a history of seizures,
and stop if you feel unwell.

## Try it with no hardware

The phantom subject reads the stimulus log and injects a synthetic response
into the same UDP port a real headband would use, so the whole stack runs
end to end:

```
python gate_test.py                       # offline maths check
python xr_session.py --mode full --session runs/demo --osc-port 5001 --freq 15 \
    --grid-w 6 --grid-h 4 --spc 2 --calib-blocks 5 --calib-on 5 --calib-off 5 \
    --color-freqs 7.5,10,15 --color-grid-w 4 --color-grid-h 3 --color-spc 3
python phantom_subject.py --session runs/demo --port 5001       # second terminal
# then open http://localhost:8082/?auto=1 in a browser
```

## Run it for real

1. Put the Muse on, attach the Oz cup, and start Mind Monitor. Set OSC
   streaming to the PC's IP, port 5000. The driver prints the address to use.
2. Start a session on the PC (examples below). It serves the stimulus page on
   port 8082.
3. Open `http://<PC-IP>:8082/` on the display: a laptop browser tab, or the
   Quest browser. The page shows live electrode contact, waits for good signal,
   then says READY.
4. Look at the white dot and click, press a key, or pull the controller
   trigger. The click also switches the page to full screen.

Recommended order for a new subject:

```
# 1. Is the Oz electrode on occipital scalp? (3 min, eyes open/closed on beeps)
python xr_session.py --mode alpha --session runs/alpha1 --calib-blocks 4 --calib-on 20 --calib-off 20

# 2. Which flicker frequency does this person respond to? (5 min)
python xr_session.py --mode sweep --sweep 7.5,10,12,15,20 --calib-blocks 4 \
    --calib-on 8 --calib-off 8 --calib-style bw --calib-size 1 --session runs/sweep1

# 3. Full session at the best frequency: calibration gate, grey image,
#    colour calibration, colour image, music (13 min)
python xr_session.py --mode full --session runs/full1 --freq 12 \
    --grid-w 8 --grid-h 6 --spc 6 --calib-blocks 6 --calib-on 8 --calib-off 8 \
    --calib-style bw --calib-size 1 --color-freqs 7.2,9,12 --target text:NO
```

### Modes of `xr_session.py`

| `--mode` | what it runs |
|---|---|
| `alpha` | eyes-open / eyes-closed blocks; reports the alpha peak per channel |
| `sweep` | full-field flicker at several frequencies, scored per frequency |
| `calib` | the calibration gate only |
| `visual` | calibration gate, then a single-square scan and reconstruction |
| `full` | `visual` plus colour calibration, colour scan and tagged music |
| `extras` | colour and music only, reusing an existing `calibration.json` |
| `mux` | three squares per dwell, each with its own tag frequency |
| `music` / `assr` | the 40 Hz auditory stage on its own |

Useful options: `--freq` flicker frequency, `--grid-w/--grid-h` positions,
`--spc` seconds per position, `--patch` square size independent of the grid
(`0` = one grid cell, `1.0` = a 6 by 4 cell), `--calib-style bw` white/black
flicker, `--color-freqs` the three colour tags, `--page-token` to ignore stale
browser tabs, `--osc-port 5000,5001` to merge two headbands.

### Running in a Quest headset

```
adb shell setprop debug.oculus.refreshRate 72     # 12 Hz = 3 frames on / 3 off
adb shell setprop debug.oculus.guardian_pause 1   # the boundary dialog hides the browser
python xr_session.py --mode full --page-token vr1 ...
adb shell am start -a android.intent.action.VIEW -d "http://<PC-IP>:8082/?k=vr1" com.oculus.browser
```

Push the page after the headset is on the head; a sleeping Quest does not load
it. Pick tag frequencies that are a whole number of frames at the panel rate:
at 72 Hz that is 7.2, 9 and 12 Hz; at 90 Hz it is 7.5, 9, 11.25 and 15 Hz. The
page measures the panel rate and reports the frequency it actually delivers,
and scoring uses that.

## How a result is decided

- **Line detector.** For each block or scan position, one Hann periodogram over
  the whole window; the score is the power at the delivered flicker frequency
  divided by the power 0.5 to 2 Hz either side.
- **Calibration gate.** Flicker-ON blocks against black OFF blocks, statistic is
  the best channel's mean log-ratio difference, significance by enumerating
  every relabelling of the blocks. With 6 + 6 blocks the smallest possible p is
  0.0011; the gate is p < 0.01. The per-channel differences become the channel
  weights for the image.
- **Delivery gate.** If the page logged fewer than 20 frames per second or too
  few flicker edges, the run is void, not negative. A hidden browser tab is
  throttled to one frame per second and looks exactly like "no response".
- **Image null.** Every reconstruction is repeated with the EEG circularly
  shifted by several offsets. The image counts only if it beats those.
- **Decoder comparison.** `analysis/decoders_compare.py` scores the same scan
  with the paper's band-power ratio, the line detector and CCA, each against
  its own nulls.

## Results so far (one subject)

| finding | evidence |
|---|---|
| The Oz cup sees occipital cortex | eyes-closed alpha peak at 10.5 Hz, 8.4 times the eyes-open power on Oz, 2.5 times behind the ear |
| This subject responds at 7.5 to 12 Hz, not at 15 or 20 Hz | laptop sweep: Oz line ratio 9.0 at 10.4 Hz and 5.9 at 11.4 Hz against 1.0 with the screen off; nothing at 20 Hz |
| The response is reproducible in the Quest browser | 12 Hz calibration gate passed at p = 0.0011 in three of three runs on 2026-09-29 |
| Frequency-tagged colour works | flag planes correlate at R 0.75, G 0.86, B 0.90; 24 of 24 cells with the right hue; shifted nulls reach at most 58% |
| Grey images are weakly above chance | 8 by 6 positions at 6 s: r = 0.41 to 0.49, nulls up to 0.48. Not yet legible |
| Only the fixated square counts | with three tagged squares on screen, the one under fixation responds and the two a third of the panel away do not |
| Smaller squares are not higher resolution | 12 by 8 cell-sized squares at 8 s give a line below the noise flanks; the response scales with flickering area |
| 40 Hz tagged music | not detected through headset speakers (p = 0.13 to 0.69) |

![Response curve and spectrum](docs/figures/fig3_response_curve.png)

![Calibration blocks in VR](docs/figures/fig6_run2_calibration_blocks.png)

The practical consequence: Mann's choice of 15 Hz is right for staying clear of
alpha but is on the weak side of this subject's response curve, and a stock
Muse without the Oz electrode showed no detectable response at all. Measure the
subject first. Resolution has to come from more overlapping positions of a
large square and longer dwell, as in the paper, not from shrinking the square.

Raw EEG recordings are not in the repository. Figures are in `docs/figures`
and the small result files for each session are in `docs/results`.

## Things that will waste your evening

- A browser tab that is not in front draws one frame per second. Keep the page
  visible; the driver follows whichever tab is visible and ignores the rest.
- An orphaned recorder holding UDP 5000 silently eats the stream.
- Counting frames drifts when frames drop. The flicker phase is taken from the
  vsync timestamp with a whole-frame half period instead.
- A refresh rate measured as a mean over a few frames read 208 Hz on a 240 Hz
  panel. The page uses the median frame interval after a warm-up.
- Full screen must be requested inside the click handler or the browser
  refuses it. A controller button seen through the Gamepad API starts the run
  but cannot request full screen.
- Text targets are thresholded so every lit square flickers at full contrast;
  the grey edge cells of an anti-aliased glyph give weaker lines.
- Neck tension puts broadband muscle noise on the Oz channel. Support the head.
- For audio the cup belongs at Cz, the top of the head, with earbuds.

## Files

| file | role |
|---|---|
| `xr_session.py` | browser-path driver: recorder, gates, all modes, scoring, reconstruction, WebSocket and HTTP server |
| `xr_stimulus.html` | the stimulus page: frame-exact flicker, scans, colour and multiplexed scans, tagged music, arm gate |
| `reconstruct.py` | per-position scoring (band power or line detector), grey, colour and multiplexed reconstruction, shifted nulls |
| `run_session.py` | the original laptop orchestrator in pygame; also the permutation gate and legacy calibration score |
| `osc_acquire.py` | records `/muse/eeg` on UDP to CSV, 4 to 8 channels |
| `phantom_subject.py` | synthetic subject for end-to-end tests, including colour tags and audio |
| `targets.py` | text, image and colour targets |
| `config.py` | shared parameters |
| `spectator_bridge.py`, `spectator.html` | live mirror of a session for a second screen or headset |
| `analysis/decoders_compare.py` | band power vs line detector vs CCA with nulls |
| `analysis/g1_diag.py`, `g1_fine.py`, `g1_line.py` | calibration diagnostics |
| `analysis/make_result_figures.py`, `make_run2_figures.py` | the figures in `docs/figures` |
| `analysis/cca.py`, `act_chirplet.py`, `lock_in.py`, `baseline_preproc.py` | the earlier decoder campaign on stock-Muse recordings |
| `analysis/patches/` | one-shot patch scripts, kept as a record of how the code changed |
| `claim_gates.py`, `gate_test.py`, `osc_gate.py`, `full_gate.py`, `webgate.py` | pass/fail gates for the maths, the UDP ingest and the full stack |
| `PLAN_NEXT.md` | research plan, campaign verdicts and the dated status log |
| `EYECAMXR_CHECKLIST.md` | checklist for the native Quest app, which lives in a separate Unity project |

## The laptop path without a browser

`run_session.py` is the original six-stage flow in one pygame window: recorder,
signal check, calibration, guided-cursor scan, reconstruction, result.

```
python run_session.py --source osc --preset quick --target "text:NO"     # Mind Monitor / MuseLog over Wi-Fi
python run_session.py --source lsl --preset quick --target "text:NO"     # muselsl over PC Bluetooth
python run_session.py --source phantom --preset quick                    # no hardware
```

Presets: `quick` 12 by 8 at 4 s, `standard` 16 by 12 at 5 s, `fine` 24 by 18
at 6 s. This path still uses 15 Hz and the band-power score by default; for a
new subject use the browser path's sweep first.

## Where this is going

- Large square on a fine, overlapping grid with 8 to 10 s per position, and two
  passes averaged, to make the grey image legible.
- The inverse geometry: the whole panel flickers through the image as a mask
  while the subject fixates position by position.
- Chirp-coded flicker matched with the adaptive chirplet transform, the
  direction the paper itself points to.
- Audio with a vertex electrode and earbuds.
- A note for the paper on security: every gate here is a public function of the
  stimulus log, and the phantom subject passes all of them. Anything a verifier
  can check from the stimulus alone, a forger who knows the stimulus can
  synthesize, so a brain response is a liveness signal only when the sensor
  itself is attested.

## Citing

S. Mann et al., "The Human Eye as a Camera," IEEE HealthCom 2019.
S. Mann et al., "Eye itself as a camera: Sensors, integrity, and trust,"
WearSys 2019, doi 10.1145/3325424.3330210.
