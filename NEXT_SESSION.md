# Next session

Written 2026-10-06. Branch `colour` (= `all-on-pc` + Khalil's smooth colour +
Minu's analysis, merged; see README "Colour beyond black, white and blue").

## Tomorrow (2026-10-11): three runs, one launcher

Everything below is phantom-checked. Laptop on mains, Quest on the USB
cable, MuseLog streaming to this PC on port 5000, Oz wet, and look at the
ear badges on the page before pulling the trigger (a loose TP10 cost blue's
second pass on 2026-10-10). `quest_run.ps1` starts the driver hidden, pushes
the page to the Quest and prints the log; each run begins when the headset
is worn and the trigger pulled.

1. **Vision pair, about 15 min** (`.\quest_run.ps1 -Run vision`). DONE 12:28,
   `runs/vr_vision1`: E up and ring U both named right from the EEG alone
   (r 0.89, 80 of 84 cells; 168 visits, 22 min, lowest certainty 65%).
   The read-out as planned: a tumbling
   E and a ring that is either closed (O) or open at the bottom (U), side by
   side on a 12 by 7 grid, 8 s per position, one pass plus the least certain
   positions once more. The launcher draws one of the eight combinations at
   random (`targets_img/make_ou_targets.py --draw`) and seals the answer in
   `targets_img/vision_today_seal.json`. When it is done:
   `python analysis/vision_readout.py runs/vr_vision1 --seal targets_img/vision_today_seal.json`
   decides which way the E points (4 choices) and O or U (2 choices) from the
   EEG picture alone, then opens the seal and says right or wrong. This is
   the acuity read-out for the paper: no letters named, no button pressed.
2. **The eight colours, about 25 min**. DONE 2026-10-12 10:40-11:13,
   `runs/vr_planes11`: 23 of 32, hue 16 of 24, r 0.75; red 32, green 31,
   blue 23 (lost on the right half of the field with the electrodes
   holding; Oz alone 25). Oz d' 1.8 on white, the weakest sitting. As planned: (`.\quest_run.ps1 -Run eight`): the
   benchmark, with blue once and green twice (the current default). Then
   `python analysis/planes_figures.py runs/vr_planes11 --label planes11`.
3. **The officer, about 33 min** (`.\quest_run.ps1 -Run officer`, after a
   break): `targets_img/officer_8x10.png`, 80 positions in four colours,
   5 / 5 / 8 s per position, one pass each, a rest break every 40. Same
   figure script.

4. **A real photograph, who is it?** Two short runs for another sitting, or
   today if the eyes allow. First the step wedge, about 7 min
   (`.\quest_run.ps1 -Run wedge`, then `python analysis/wedge_readout.py
   runs/vr_wedge1`): the response against flicker brightness, which says
   how many grey levels the eye separates. Then the photo, about 25 min
   (`.\quest_run.ps1 -Run photo`): one of the four portraits in
   `targets_img/photos/` (Mann's paper portrait, Lincoln, Einstein, Tesla),
   cut to 12 by 12 in black and white the way the 2019 paper thresholded
   its Fig. 1, drawn at random and sealed by
   `targets_img/make_photo_gallery.py --draw`. Afterwards
   `python analysis/photo_readout.py runs/vr_photo1 --seal targets_img/photo_today_seal.json`
   names the person from the EEG picture alone by correlation against the
   whole gallery, then opens the seal. On the clean targets the four are
   told apart 100% of the time with a quarter of the cells flipped, so a
   picture as good as last night's colour runs should name the right one.
   `--levels 3` on the gallery script if the wedge separates three shades.
   Gate: `python photo_gate.py` (phantom, about five minutes).

Do the vision pair first (short, fresh eyes), the eight colours second,
the officer after a real break; on 2026-10-10 the blink rate had doubled by
the second run of the evening, so stop when the eyes say so and keep the
officer for another day if needed. A second ring size (`ou_big_o_12.png`,
`ou_big_u_12.png`, 12 by 12, double-size optotype) is there for a
threshold series later.

## 2026-10-10, two hours later: the second run (`runs/vr_planes10`)

Same code with `--plane-passes 1,2,2` (green and blue scanned twice), the
run that was to buy the eight missing cells. It did not: colour r 0.66
(R 0.79, G 0.82, B 0.44), 20 of 32 cells (R 29, G 31, B 23), hue 16 of 24,
null 0.42. `analysis/planes_passes.py` decodes each plane pass by pass:

- green: pass 1 28, pass 2 26, both averaged 31 of 32. The second pass works.
- blue: pass 1 24 (r 0.55), pass 2 21 (r 0.01: lit cells 0.20 against dark
  0.18, no response at all), both 19, with the revisits 23. Averaging a dead
  pass with a live one cost five cells.
- red: 29 (32 two hours earlier); its response fell from 0.45 to 0.15 within
  its three minutes, at minute 20-23 of the run.

Why: not the start of the run. The white calibration gave Oz d' 4.4 (5.6
before) and the colour calibrations matched the first run's (red +5.8, green
+3.9, blue +5.0 against +5.0, +4.3, +5.5). The run then faded in two ways,
`docs/state_of_flash.md` and `docs/figures/state_of_flash_ssvep_fatigue.png`:

- The right ear electrode (TP10) came loose at minute 11.2 of the run (51 %
  of its epochs over 150 uV in minutes 10-15, 23 uV RMS against 7 before),
  exactly as blue's second pass began; it carries 0.15 of blue's weights and
  the dark cells' 9 Hz score rose with it (0.14 -> 0.18). With the ear
  channels dropped from the decode the two blue passes read 26 of 32 (19
  with them). The blue response on Oz alone still fell from 0.34 to 0.19
  between the passes: a real fading on a clean contact, cause open. An
  "alpha rise on the ears" reported before this check was that electrode;
  alpha on every electrode that held was flat.
- The Oz contact: 1-40 Hz RMS about 10 uV from minute 1 to 20, then rising
  to 80 uV by minute 30, mains pick-up 20 -> 50 uV. Red and green were
  scanned in those minutes. Wet Oz again before a second run, or run the
  colours first.

Default changed to `--plane-passes 1,2,1`: green twice, blue once plus its
revisits (blue is scanned first; 5 min saved). A loose-channel guard in the
decode (drop a channel whose RMS triples against its calibration) would make
a second blue pass safe again; until then check the ear badges on the page
before any second run. Session: about 25 min. A second run in the same
evening is a different subject state: treat it as one.

The subject's report after this evening (not drowsy; eyes irritated; slight
forehead ache) against the EEG is in `docs/state_of_flash.md`: blink rate
doubled between the runs with the blink shape unchanged (irritated eyes,
supported), no forehead muscle tension or effort theta (the ache is not in the
EEG; the headset's facial interface is the likely cause), no drowsiness sign.

## 2026-10-10: the best colour run so far (`runs/vr_planes9`)

The merged code (Minu's certainty scanning and HFPM palette, commit a4eac48)
with every fix on; laptop on mains, Quest on the cable, MuseLog over Wi-Fi.
White calibration Oz d' 5.6 (p 0.001). Blue sweep: 6 Hz +4.2, 7.2 Hz +4.4,
9 Hz +5.5, 12 Hz +4.1 (all p <= 0.007), so blue was scanned at 9 Hz with the
patch doubled, 10 s per position, and its ten least certain positions were
shown again. Red and green calibrations p 0.014 each. 21 min from the white
calibration to the end of the green plane (blue 7 min with the revisits).

Result: colour r 0.79 (R 0.90, G 0.79, B 0.69); 24 of 32 pure-colour cells
right (R 32, G 28, B 28); hue right on 19 of 24; shifted-EEG null 0.32.
`vr_planes8` three days earlier: 18 of 32 (R 29, G 27, B 24), hue 14 of 24.
Figures: `runs/vr_planes9/planes9_colour_picture.png` and
`planes9_planes_figure.png` (`analysis/planes_figures.py`).

What carried it (decoding the blue log again in pieces):

- Blue from the first pass alone (9 Hz, patch 2x): 27 of 32, r 0.63. With
  the ten revisits: 28, r 0.67. With the white calibration's weights instead
  of blue's own: 28, r 0.66. The frequency and the patch did the work; the
  revisits added one cell; the per-colour weights changed nothing on a day
  when Oz carried every plane.
- Red 29 -> 32 and green 27 -> 28 with no change to their settings: the Oz
  contact (white calibration d' 5.6 against 0.8 on `vr_planes8`). Wet Oz.
- The driver printed "PC FROZE for 5 s" right after it scored the white
  calibration; the EEG has no hole there (the recorder is its own process).
  That was the driver scoring, not standby. The heartbeat now tells the two
  apart by CPU time and prints "driver busy" instead (`freeze_gate.py`
  still passes for a real suspension).

Next: the same planes command. The eight wrong cells were four green and four
blue; the second run that evening (above) showed a second pass of green buys
its cells and a second pass of blue does not, so the default is now
`--plane-passes 1,2,1` (a rest break falls between green's passes with
`--rest-every 32`), on top of blue's ten revisits. Session: about 25 min.

## 2026-10-07: the first complete three-plane run, and what it found

`runs/vr_planes7` (Quest 3S, MuseLog over Wi-Fi, 12 Hz, `eight`, 10 / 6 / 6 s
per position): calibration Oz d' 5.7; red plane r 0.76 and 29 of 32 right,
green 0.60 and 22, blue 0.27 and 17; 11 of 32 pure-colour cells, because the
colour is only right where all three planes are. The diagnosis is in
`runs/vr_planes7/planes7_diagnosis.png`:

- The laptop, on battery, entered Modern Standby five times during the scan
  (Windows event log, Kernel-Power 506 / 507) and froze the driver and the
  recorder for 11 to 64 s each time while the headset kept scanning: 135 s
  of holes, ten positions of blue and green scanned blind. Red had none and
  was the plane that worked. Fixed: the driver holds the PC awake and logs
  any freeze as "PC FROZE"; the page treats a heartbeat missing for 3 s as
  a hold and shows the position again (`freeze_gate.py`). Keep the laptop
  on mains anyway.
- The black/white calibration's channel weights put 0.57 on the ears, which
  carry the white response but only noise for blue: blue on the clean cells
  r 0.21 with them, 0.55 from Oz alone. Fixed: `--plane-calib 4` (default)
  runs four black/colour blocks per plane before the scans and decodes each
  plane with its own weights when its own response is found (p < 0.05); a
  colour without a response is announced on the page before five minutes
  are spent on it.
- One very strong cell dragged the automatic threshold above the other blue
  cells. Fixed: Otsu on the log of the scores for the colour planes (blue
  17 -> 21 of 32 on the same EEG, red unchanged); the result line now counts
  each plane: "(R 29, G 24, B 21)".

Next run: the planes command below, laptop on mains, phone screen on. The
blue fixes are on by default (`--plane-patch 1,1,2`, `--plane-sweep 6,7.2,9,12`
for blue, `--plane-revisit 0,0,10`): blue's calibration is a 3 min sweep, it is
scanned at its best frequency with a double-size patch, and its ten least
certain positions are shown again (about 2 min); green gets a second pass
(`--plane-passes 1,2,1`, about 3 min, since 2026-10-10). Session: about 25 min.

## The colour session, in order (one day, one electrode placement)

Wet the Oz electrode and check it on the page badge before anything else: the
two voided colour runs so far were both a floating Oz.

1. **Planes, eight colours** (3 scans, about 20 min with `--plane-spc 6,6,10`
   on an 8 by 4 grid plus 3 min of per-colour calibration; the scoring that
   gave the legible "NO").
   ```
   python xr_session.py --muse --mode planes --color-target eight --color-grid-w 8 --color-grid-h 4 \
       --freq 12 --calib-blocks 6 --calib-on 8 --calib-off 8 --calib-style bw --calib-size 1 \
       --plane-spc 6,6,10 --rest-every 32
   ```
   (`--source osc` instead of `--muse` with MuseLog on the phone; add
   `--plane-sweep ''` or `--plane-revisit 0` to switch the blue fixes off.)
2. **Smooth with rotating tags, eight colours** (sweep 3.6 min + 3 passes of
   32 cells at 8 s = 13 min).
   ```
   python xr_session.py --muse --mode smooth --color-target eight --color-grid-w 8 --color-grid-h 4 \
       --smooth-passes 3 --smooth-rotate 1
   ```
3. Compare: hue right on how many of the 24 coloured cells, pure-colour cells
   right of 32, r per plane, both above their time-shifted nulls. The better
   path becomes the default; then `hues` (12 hues) on that path, then a face
   with colour (`--color-target targets_img/<face>.png`).
4. Optional, 5 minutes: the **phase check** (full-field 12 Hz at 0 / 120 / 240
   degrees). Phase stability measured on 2026-10-04 was PLV 0.80 to 0.93 after
   removing the sub-mHz drift, so a phase-coded palette at one frequency is
   the next coding to try if both paths above are weak on blue.

## Still on the list from before

- Step wedge (`targets_img/wedge_12x4.png`): the response against brightness.
- Landolt rings (`targets_img/landolt_c_chart_12.png`, `--rest-every 36`).
- Gaze check from AF7 minus AF8; Oz-amplitude scoring (0.96 / 0.92 / 0.90
  against 0.91 / 0.87 / 0.79 on three sessions, confirm on a new run).
- Paper-style overlapping scan of the high-contrast portrait (new scan mode).

## Built, tested on the phantom only

- `--mode planes`, `--smooth-rotate`, targets `eight` and `hues`, the hue
  metric (`colour_gate.py`).
- Link watch: the run pauses and says so on the page when the computer
  changes network, the page loses the driver, or the EEG stops; it resumes
  by itself when EEG returns. Tested by pulling the stream.
- Frozen computer: the PC is held awake; a heartbeat missing for 3 s holds
  the scan and the position is shown again (`freeze_gate.py`: driver and
  recorder suspended 8 s mid-scan). Per-colour calibration (`--plane-calib`)
  and the log-domain threshold for the colour planes (`colour_gate.py planes`).
- `--rest-every N`, `--color-passes` rotation in `--mode full`, `muse_ble.py`
  / `eyecam.py` (Muse straight to the PC over Bluetooth).

## Blink and closed-eye masking: tested, leave off

- Blinks are about 1 % of our scan recordings; the closed-eye detector flags
  59 % of truly closed and 31 % of truly open time; masked re-scores of three
  sessions: 0.784 to 0.781, 0.778 to 0.779, 0.722 to 0.689.

## 2026-10-16: the portrait runs, decided

Bose is the black-and-white portrait; the other four go in colour, one a
sitting, as a pair of scans (brightness at 12 x 12 by the grey scan, colour
at 6 x 6 by the planes; `analysis/colour_combine.py` puts them together).
Targets and credits: `targets_img/*_luma_*.png`, `*_chroma_*.png`,
`targets_img/photos_more/README.md`. Order, each after the wedge
(`.\quest_run.ps1 -Run wedge`, 7 min, which says how many grey levels the
four-level targets may use; `--levels 2` on make_colour_pair.py if only two):

1. Bose, grey, 16 x 16 four levels, 34 min a pass:
   `.\quest_run.ps1 -Run grey -Target targets_img\bose_luma_16x16.png`
2. Mann (colour headshot), 19 + 11 min:
   `.\quest_run.ps1 -Run grey -Target targets_img\mann_c_luma_12x12.png`
   `.\quest_run.ps1 -Run cphoto -Target targets_img\mann_c_chroma_6x6.png`
   `python analysis/colour_combine.py runs\vr_grey2 runs\vr_cphoto1 --name mann_c`
3. Lincoln (colourised 1863 Gardner), same three commands with `lincoln_c`.
4. Einstein (colourised, DonkeyHotey CC BY 2.0), `einstein_c`.
5. Tesla (coloured by Ivar van Wooning, CC BY-SA 3.0), `tesla_c`.

Session numbers: the launcher takes the next free `vr_greyN` / `vr_cphotoN`;
check `runs\` before the combine. The officer (`-Run officer`, 33 min) and
the sealed plate (`targets_img/make_plate.py --draw`, 30 min) stay as the
corner-colour runs. Colourised colours are a colourist's, say so in the paper.

## 2026-10-17, 00:30: vr_grey1 (Bose 16 x 16), stopped at the first rest

Calibration strong (AUX line SNR 5-14 on, 0.1-2 off; electrode clean through
the scan), but the scan response faded to the OFF level after about two
minutes and never followed the grey; identical white cells swung 20-fold.
Started at 00:35, so fatigue is the first suspect. Second: on a square board
the 6 x 4 patch rule drew a tall, narrow rectangle (112 x 168 px of a 1280 x
800 browser window, 70% of the NO run's patch area). New `--patch-side`
(square, fraction of the board's shorter side) and `quest_run.ps1 -Run grey`
now uses 0.3 (about 5 cells of 16, 2.2 x the area). Redo Bose rested:
`.\quest_run.ps1 -Run grey -Target targets_img\bose_luma_16x16.png`
(add `-PatchSide 0.4` for a bigger one). The mid-run checks live in the
scratchpad only (snr_compare / timecourse / first_minutes), to be made a tool
(`analysis/midrun_check.py`) if this happens again.

## 2026-10-17, 01:34: Ear as a Microphone, first real head (runs/ear_mic1)

Alice ch. 1 clip (31.5 s) x 24, played from the Quest browser (--player quest,
new tonight; the trigger pull left the audio context suspended, fixed in the
page and backed by a DevTools resume in quest_audio.py). Result: averaged
reconstruction r 0.18 (null 95% 0.09, p 0.0006), single trial r 0.037
(p 0.0006); rolled-EEG control r 0.078 (p 0.044, so the clip's own rhythm
gives a little by itself). Channels: TP9 0.16 and TP10 0.11 carry it, AUX and
frontals nothing; model-free lag TP9 +344 ms (neural-looking), TP10 -47 ms
(not). Phantom was 0.65. CAVEAT: the Quest speakers sit beside TP9/TP10.
Control still to run: same session with the Muse on the desk under the
headset, nobody wearing it, 6 plays:
    python ear_mic_session.py --song targets_audio/alice_ch1_rabbit_hole.wav --start 0 --clip 31.5 --reps 6 --session runs/ear_mic_desk --player quest
    python analysis/ear_mic_decode.py runs/ear_mic_desk
If that decodes anything, the headset-speaker result is pickup, and the paper
needs the PC-speaker run (--player pc) instead.
Attempted at 01:55 (runs/ear_mic_desk, --quest-auto, prox_close broadcast so
the Quest stays awake on the desk): the Muse stream stopped 85 s in, before the
first full clip, so nothing decodable; the phone's MuseLog recording was closed
at 02:01. Re-run it with the Muse on and MuseLog streaming for the whole 4.5 min.
Afterwards: adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
(done tonight) to give the wear sensor back.

## 2026-10-17, 11:52: Ear as a Microphone, second real head, morning (runs/ear_mic2)

Same Alice clip x 24 from the headset speakers, rested, all five electrodes
clean (top one 39 uV). Averaged reconstruction r 0.30 (null 0.09), one play
0.073. The half-period control arm also rose (0.16, p 0.006): the start and
stop of each play. Diagnosed and fixed: leaving the first and last 3 s of every
play out (analysis/ear_mic_decode.py --trim-s, now the default) gives r 0.36
(single 0.086, 23 plays) with the control at chance (0.035, p 0.29); a
quarter-period control is at chance too (0.063). Gap jitter raised to 0-3 s
for the next sessions. Still open: the desk control (speaker pickup) and a
PC-speaker daytime run for the paper.
