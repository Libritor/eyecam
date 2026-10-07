# Next session

Written 2026-10-06. Branch `colour` (= `all-on-pc` + Khalil's smooth colour +
Minu's analysis, merged; see README "Colour beyond black, white and blue").

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
certain positions are shown again (about 2 min). Session: about 22 min.

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
