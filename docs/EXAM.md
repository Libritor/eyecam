# The eye exam: acuity, contrast, field and colour from the EEG, per eye

`xr_session.py --mode exam` turns the eye-as-a-camera rig (a Muse headband with
the AUX electrode at Oz, the Quest browser as the display, the fixation dot)
into an objective eye test. Nothing is asked of the person but to look at the
dot with one eye while the other is covered. Each test is a run of ON/OFF
blocks at the session's flicker frequency, and each step of a test is scored by
how much of that frequency appears in the EEG during its ON blocks against the
test's OFF blocks, the same exact-line detector and exact permutation test the
calibration and the frequency sweep use.

| test | what is shown (ON) | what varies | OFF | reads out |
|---|---|---|---|---|
| acuity | a vertical black/white grating appearing and disappearing on a grey of the same mean luminance (pattern onset/offset) | bar width, coarse to fine (`--exam-sizes`, arcmin) | the grey alone | the bar width at which the response is extrapolated to vanish (sweep-VEP rule), and the finest bar still giving a significant response |
| contrast | the same grating at one bar width (`--exam-contrast-size`) | Michelson contrast (`--exam-contrasts`) | the grey alone | the contrast at which the response is extrapolated to vanish, as log sensitivity |
| field | one sector of a grid over the patch flickering white/black, the eyes on the centre dot | the sector (`--exam-field`, 4x4) | black | amplitude and phase per sector; sectors without a significant response; with two eyes, the sector-by-sector ratio and phase difference |
| colour | the whole patch flickering red, green or blue against black | the plane | black | amplitude per plane, R:G:B |

Defaults: 4 s ON / 4 s OFF, two ON/OFF pairs per step, patch 0.6 of the short
screen side; eight bar widths, six contrasts, sixteen sectors, three planes:
about 7 minutes per eye. `--exam-eye right,left` runs the right eye and then
the left with a "cover your other eye" screen between them; `both` tests
binocularly.

## Running it

Phantom, no hardware (the gate, about 13 minutes):

```
python exam_gate.py
```

On the Quest:

```
.\quest_run.ps1 -Run exam -Ppd 20
```

The run writes under `runs/<session>/`: `exam_<eye>_<test>_log.csv` (one row per
frame, the usual columns plus `size`, `contrast`, `plane`), `exam.json` (every
step's amplitude, p, phase and the page's description of what it showed),
`exam_readout.json`, `exam_report.pdf` and `.png`. `python analysis/exam_readout.py
runs/<session>` redoes the thresholds and the report from `exam.json`.

## The arcmin scale

Bar widths are specified in arcmin and converted on the page with
`--exam-ppd`, the browser window's CSS pixels per degree of visual angle, then
rounded to whole device pixels; the width actually delivered is reported in
every block and used in the fits. The default of 30 px/deg is for the headless
gate; for the headset it has to be measured once (the width of the browser
window in CSS pixels divided by the angle it subtends). Until it is, the acuity
numbers are relative: the comparison between the eyes and between sittings
holds, the Snellen-style equivalent does not. The grey and the contrast steps
assume a display gamma of 2.2.

## How the thresholds are decided

For acuity and contrast the response amplitude (the noise-corrected line
amplitude at the flicker frequency and its double, on the channel with the
largest ON/OFF difference over the whole test) is taken against log bar width
or log contrast. The fitted limb starts one step below the finest (faintest)
step that still gives a significant response (p < 0.05), where the response has
gone, runs through that step and on toward the peak while the amplitude keeps
rising by at least a tenth of the peak per step; a line through those points
is extrapolated to zero amplitude. That crossing is the threshold (Norcia and
Tyler, 1985, for the sweep VEP; Campbell and Maffei, 1970, for contrast). The
anchor below matters: in the phantom gate one step inside the limb was lost to
noise, and a line through the saturating top of the curve alone landed four
times too low. A threshold below half the finest step shown, or a finest step
that still responds, is marked as beyond the range shown. The response phase is taken against the first logged flicker edge
of each block and is reported relative to the mean over the responding sectors,
so it is a within-eye and between-eye comparison, not an absolute latency (one
flicker period at 15 Hz is 67 ms, and the pathway's delay is longer than that).

## What the phantom gate checks

`exam_gate.py` runs the full stack against `phantom_subject.py` with a logistic
visibility in log bar width and log contrast (`--acuity-cutoff`,
`--contrast-cutoff`, per eye), a dead sector (`--field-loss`) and the colour
gains: right eye cutoff 8 arcmin and contrast 0.08 with sector 6 dead; left eye
16 arcmin and 0.2, whole; green 1, red 0.7, blue 0.4. Because a line through the
rising limb crosses zero below the half-visibility point, the gate asks that
each fitted threshold lie where the phantom's visibility runs from 0.03 to
0.6, that the left eye's thresholds be at least 1.3x the right's, that the
field names exactly the dead sector, and that green >= red > blue in both eyes.

Result on 26 October 2026 (`runs/exam_gate2`, 13.7 min, headless Chrome at
60 fps, 15.01 Hz delivered): all twelve checks pass. Right eye: acuity threshold
3.9 arcmin (finest bar with a response 6'), contrast threshold 0.031 (faintest
with a response 0.06), 15/16 sectors, sector 6 silent, R:G:B 0.70:1.00:0.42.
Left eye: 7.8 arcmin (finest 10'), contrast 0.067 (faintest 0.12), 16/16
sectors, R:G:B 0.68:1.00:0.36. Response phase 42 to 47 degrees in every
responding sector of both eyes (spread 0 to 3 degrees), random in the silent
one. A first gate on the same protocol had lost the left eye's 0.12 contrast
step to noise (two 3 s blocks at a response of 0.13 of full), which is what
put the anchor step into the limb fit.

## The first sitting (27 October 2026) and what it changed

`runs/vr_exam1` (right eye with the left covered, then the left eye, stopped
during its field test) and `runs/vr_exam2` (restarted as the left eye but done
with both eyes open), Quest at 12 Hz, 4 s blocks, patch 0.6 of the browser
window's short side, merged into `runs/vr_exam_2026-10-27`. No threshold could
be fitted: the only clear 12 Hz response on Oz was the full-patch colour
flicker (right eye, log10 line SNR +0.75 in ON blocks against -0.22 in OFF);
the gratings gave a weak one at the coarse bars, the sectors and the contrast
steps none. Both ear electrodes sat at 100 to 280 uV rms for the whole sitting
under the headset strap, and Oz, clean for the right eye (13 to 18 uV), rose to
50 to 77 uV afterwards.

What changed because of it: the signal check's rail test had passed those
ears, so the exam now measures each channel's rms before every eye and asks on
the headset for a fix (press the ear pads in, wet the top electrode, pull the
trigger) until every channel is under 60 uV; the scorer sets aside any channel
above that for a whole test, so a dead ear cannot be picked as the test's
channel by chance; the read-out refuses a threshold when only one step
responds, says "below the finest bar shown" instead of a Snellen number when
the line crosses zero outside the range, and names the electrodes that were
out of contact; the exam's own trigger pull now takes the page full screen and
the Quest run uses a patch the full height of the window. Still open for the
next sitting: whether pattern onset/offset at 12 Hz (3 frames on, 3 off at
72 fps) drives the response enough on a bigger patch, or whether the grating
tests need a slower rate (`--freq 6` is frame-exact on the Quest and keeps its
second harmonic at 12 Hz) or more repeats. `python exam.py merge <out>
<session>:<eye>[=<label>] ...` joins eyes from several sessions into one
report; `python exam.py <session>` re-scores one.

## What this is and is not

These are response amplitudes from one occipital electrode against a flickering
display, with a stimulus whose absolute scale depends on a measured pixels-per-
degree. They say how the eye's response changes with bar width, contrast,
place in the field and colour, and how the two eyes differ. They are not a
refraction, a visual field test or a diagnosis; the next steps (a blur study
with trial lenses against a letter chart, then a normative cohort) are what
would tie them to clinical measures.
