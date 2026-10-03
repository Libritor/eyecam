# Next session

Written 2026-10-03. Branch `all-on-pc`.

## Runs, in order

1. **Step wedge** (`targets_img/wedge_12x4.png`, 48 positions, 6.5 min per pass, two passes).
   Gives the response curve: 12 Hz response against flicker brightness.
   ```
   python xr_session.py --mode visual --freq 12 --target targets_img/wedge_12x4.png --spc 8 --passes 2 \
       --calib-blocks 6 --calib-on 8 --calib-off 8 --calib-style bw --calib-size 1
   ```
2. **Landolt rings** (`targets_img/landolt_c_chart_12.png`, 144 positions, 19 min per pass).
   Add `--rest-every 36`. Score: gap direction of each of the four rings.
3. **Phase check, 5 minutes** (to build first): full-field blocks at 12 Hz with
   starting phase 0, 120 and 240 degrees. Question: does the decoded phase
   follow the commanded phase? If yes, colours can all flicker at 12 Hz and
   differ only in phase.
4. **A face with some colour.** Method depends on step 3: phase-coded palette
   at 12 Hz, or the rotated red/green/blue tags (`--color-passes 3`).
   Brightness can be varied along with frequency if needed.

## Before each long scan

- Run the collaborator's `diagnose.py` (branch `meny's-branch`) right after
  calibration, or read the calibration line: Oz ON response was 192 and 19 on
  the two good days, 7 on the weaker one.
- Wet the Oz electrode; glasses arms outside the Muse band.
- Phone on a charger, MuseLog streaming to this PC's current address.

## Built but not yet tried on a person

- `--rest-every N` rest breaks, `--color-passes` tag rotation, pause on EEG loss
- `muse_ble.py` / `eyecam.py` (Muse straight to the PC over Bluetooth)

## Offered, waiting for a decision

- Gaze check from the forehead electrodes (AF7 minus AF8 follows left/right
  moves of the dot; every move classified correctly on `vr_pix_small`).
- Scoring by Oz amplitude at the flicker frequency instead of the power ratio
  (0.96 / 0.92 / 0.90 against 0.91 / 0.87 / 0.79 on three sessions; found
  after the fact, confirm on a new run).
- Paper-style scan of the high-contrast portrait (cursor shows the image under
  it, overlapping positions). Needs a new scan mode; about 100 min for 16x16.
- `meny's-branch`: take `diagnose.py`, the corrected Eq. 1 weights and the
  black/white/blue classifier; leave blink masking off (see below).

## Blink and closed-eye masking: tested, leave off

- Blinks are about 1 % of our scan recordings.
- The closed-eye detector, checked against the eyes-open / eyes-closed test
  recording: flags 59 % of truly closed time and 31 % of truly open time.
- Inside scan positions it flags 1 to 6 % of the time, and the 12 Hz response
  is not lower in those stretches.
- Re-scoring three sessions with masking: 0.784 to 0.781, 0.778 to 0.779,
  0.722 to 0.689.
