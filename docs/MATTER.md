# Mind to Matter: the thing you looked at, as your brain registered it, printed

`xr_session.py --mode matter` closes a loop: a 3-D thing is shown flat (a clip or a picture
in the headset), the brain's response to it is read square by square through the EEG, and
the result goes back to 3-D as an object you can print and hold. The same run is an eye test:
a square the brain did not register is a square the eye did not deliver.

## What happens in a sitting

1. **Calibration.** The full window flickers white/black at 12 Hz in five ON/OFF pairs. This
   is the one rate that responded in every exam sitting (see below); it picks the channel the
   rest is read on and gives the amplitude the squares are read against.
2. **Watch.** The clip plays (muted, looping) or the picture sits on screen. When the thing to
   print is on screen, pull the trigger. The page pauses, captures exactly what was on screen
   and sends the frame to the computer.
3. **Find.** The computer finds the thing in the frame three ways and keeps the one that looks
   most like a single object: the depth model's nearest surface (Depth Anything V2, offline),
   a foreground cut seeded from it, or whatever differs in colour from the frame's edges. A
   grid of at most 7 x 5 near-square cells is laid over the thing's box; the cells at least
   30 % inside it are the squares to scan (at most `--matter-cells`, 20). Around the box runs
   a ring of background squares (`--matter-ring 1`), of which `--matter-ring-cells` (8),
   spread round the ring, are scanned too, so the brain's picture has an outside; two of
   them are shown plain black and plain white as controls, the yardsticks the rest is read
   against (and the eye test's floor and ceiling). Each square's grey level is the frame's
   mean luminance inside it, stretched over the scanned squares to the display's range,
   because the brain tells black from grey from white, not 0.8 from 1.0 (the first gate run
   with the raw levels of a bright creature, 0.73 to 1.0, read back flat). The frame with
   the outline and numbered squares is shown for a few seconds.
4. **Scan.** Each square in turn, in a fresh random order per pass: the fixation dot moves to
   the square and a patch there flickers at 12 Hz between black and the square's grey level
   for `--spc` seconds (8 s in the runs that gave legible pictures). The eyes follow the dot.
   This is the eye-as-a-camera scan with the frame's squares as the target.
5. **Read back.** Each visit is scored at the delivered flicker rate (`--recon-method`: the
   paper's scorer by default, the one the legible 8-second pictures used; `line`, the exact-line
   detector, reads 4-second dwells better and is what the gate uses). The brain's grid is the
   response per square. Its correlation with the frame's grey levels is taken over the squares
   that were shown, controls included, and reported with eight shifted-EEG nulls (the same
   scoring on misaligned EEG), which is what chance gives.
6. **Matter.** `matter.py` writes to `<session>/print/` and copies to
   `Downloads/mind_to_matter_prints/<session>/`:
   - `brain_relief.stl`: the thing's silhouette as a plinth, the brain's response as height on
     it, square by square (a lithophane of what was registered)
   - `object_seen.stl`: the thing in 3-D from the depth model (nearer parts stand higher),
     scaled by the brain's response: registered parts at full height, the rest sunk
   - `object_seen_base.stl` + `object_seen_mind.stl`: the same object in two bodies for two
     spools, the base and a 1 mm skin over the squares the brain registered. In Bambu Studio
     load both as one object with several parts and give the skin the second filament.
   - `preview.png`: the frame, what was found, the frame's squares, the brain's squares, and
     both reliefs shaded
   With `--matter-slice` the two single-body prints are sliced for the H2C as well.

## Running it on the headset

Muse S with the AUX electrode at Oz, MuseLog streaming to this computer on port 5000 (the
MuseLog aux stream is the clean one; Mind Monitor's was noisier). The clip can be any mp4 the
Quest browser plays (a phone recording is fine).

    python xr_session.py --mode matter --source osc --matter-media "C:\path\to\dinosaur.mp4" ^
        --matter-name dinosaur --freq 12 --spc 8 --passes 2 --patch-side 0.35 ^
        --calib-style bw --calib-size 1 --recon-method line --session runs/vr_matter_1

Open `http://<this computer's IP>:8082/` in the Quest browser (the driver prints the URL),
put the headset on, pull the trigger at the dot. Twenty squares at 8 s and two passes is a
little over five minutes of scanning. `--matter-grid 5x4 --matter-cells 12` halves it.

Afterwards: `python matter.py runs/vr_matter_1 --name dinosaur` redoes the find and the
prints from the recorded session (add `--slice` for the H2C files).

## Why 12 Hz, why one square at a time, why a big patch

From the exam and eye-as-a-camera sittings on the Quest (72 fps), on this subject:

- **12 Hz** (3 frames on, 3 off) responded in every sitting: flash gate p = 0.004 to 0.012,
  temporal sweep p = 0.011 with the largest amplitude of all rates. **18 Hz** responded but
  weaker (p = 0.022, 0.7 of the 12 Hz amplitude). **6, 7.2 and 9 Hz** did not reach
  significance in the November sweep (p = 0.08, 0.41, 0.40; 7.2 Hz had as much at OFF as at
  ON), and 9 Hz passed only once in earlier runs. On the laptop panel 10.4 and 11.5 Hz were
  strongest, 15 Hz weak, 20 Hz nothing. So the scan uses 12 Hz, and a second tag, if one is
  ever needed, should be 18 Hz, not a lower rate.
- **One square at a time, eyes on it.** The multiplexed runs (several patches at once, each
  with its own frequency) scored at chance even with a passed calibration: only the fixated
  patch responds on this rig (vr_mux2: the middle tag p = 0.03, the outer tags nothing). The
  quadrant test shows large sectors do respond in the periphery (3 of 4 quadrants at about a
  third of the full-field amplitude), but a 7 x 5 cell is far smaller than a quadrant. Hence
  the moving dot: the person looks at each square while it flickers.
- **A big patch.** A small square gives a line under the noise even at 8 s; the legible
  "NO" came from large patches (`--patch 3 --board 0.6`, r = 0.87). `--patch-side 0.35`
  draws the square at 35 % of the board's shorter side, whatever the grid.

## The gate

`python matter_gate.py` runs the whole mode on `media/test_creature.png` with the phantom
subject at Oz and a headless browser (about 4 minutes) and checks: frame captured, the
creature found (6 or more squares, 5 to 60 % of the frame), the brain's grid against the
frame's squares r >= 0.6 and above every shifted-EEG null, all four prints written and
closed, the session ok. `python media/make_test_clip.py` remakes the test still and an H.264
clip of the same creature walking.

## Honest limits

- This is the brain's registration of each square's flicker, read at Oz. It is not
  recognition: whether the person knew it was a dinosaur is a different measurement (an
  oddball on the name would be the way).
- Resolution is the grid, 20 squares or so. Finer grids cost time linearly and the smallest
  useful square is set by the signal, not the display.
- Flicker at 12 Hz over a large patch is in the photosensitive range. Ask about
  photosensitive epilepsy before anyone else sits, keep the room lit, and stop on request.
- The depth model gives relative depth of a picture, not the thing's true shape; the 3-D is
  a relief of the view, which is what was seen.
