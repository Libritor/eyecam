# Next session

Written 2026-10-06. Branch `colour` (= `all-on-pc` + Khalil's smooth colour +
Minu's analysis, merged; see README "Colour beyond black, white and blue").

## The colour session, in order (one day, one electrode placement)

Wet the Oz electrode and check it on the page badge before anything else: the
two voided colour runs so far were both a floating Oz.

1. **Planes, eight colours** (3 scans, about 20 min with `--plane-spc 6,6,10`
   on an 8 by 4 grid; the scoring that gave the legible "NO").
   ```
   python xr_session.py --muse --mode planes --color-target eight --color-grid-w 8 --color-grid-h 4 \
       --freq 12 --calib-blocks 6 --calib-on 8 --calib-off 8 --calib-style bw --calib-size 1 \
       --plane-spc 6,6,10 --rest-every 32
   ```
   (`--source osc` instead of `--muse` with MuseLog on the phone.)
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
- `--rest-every N`, `--color-passes` rotation in `--mode full`, `muse_ble.py`
  / `eyecam.py` (Muse straight to the PC over Bluetooth).

## Blink and closed-eye masking: tested, leave off

- Blinks are about 1 % of our scan recordings; the closed-eye detector flags
  59 % of truly closed and 31 % of truly open time; masked re-scores of three
  sessions: 0.784 to 0.781, 0.778 to 0.779, 0.722 to 0.689.
