# The eye exam: acuity, vernier, contrast and its function, field, gaze, meridians, flicker photometry, colour and the temporal response from the EEG, per eye

`xr_session.py --mode exam` turns the eye-as-a-camera rig (a Muse headband with
the AUX electrode at Oz, the Quest browser as the display, the fixation dot)
into an objective eye test. Nothing is asked of the person but to look at the
dot with one eye while the other is covered. Each test is a run of ON/OFF
blocks at the session's flicker frequency, and each step of a test is scored by
how much of that frequency appears in the EEG during its ON blocks against the
test's OFF blocks, the same exact-line detector and exact permutation test the
calibration and the frequency sweep use.

Every stimulus fills the whole browser window. That is the first lesson of the
eye-as-a-camera runs carried over: the full-field white/black flicker the
camera calibrates on gives the strongest response the rig has (Oz line SNR in
the tens to hundreds), a patch gives less, and a small square gives a line
under the flanking noise even at eight seconds. The second lesson is that the
flash itself is the best reference: every eye starts with it, which says
whether the electrodes and the display deliver at all before any finer test is
read, picks the channel the rest of that eye is read on, and gives the
amplitude the rest is given relative to, so an electrode that drifts between
the two eyes cancels out of their comparison.

| test | what is shown (ON) | what varies | OFF | reads out |
|---|---|---|---|---|
| flash | the whole window flickering white/black (the camera's calibration stimulus) | nothing; five ON/OFF pairs (`--exam-flash`) | black | the response on every channel; the eye's reference channel and amplitude, or "no response" |
| temporal | the same full-field flash at several rates, one step per rate | the rate (`--exam-rates`, 6 to 30 Hz asked; the panel delivers refresh / 2k frames: 6, 7.5, 10, 15, 30 Hz at 60 fps, 6, 7.2, 9, 12, 18, 36 Hz at 72) | black | amplitude per rate with the OFF blocks' line power taken out, the best rate and the rate where the response has fallen to half; the apparent latency from the slope of the phase against rate over the rates that respond, and between the eyes its difference |
| acuity | a vertical black/white grating whose bars swap places at the flicker rate (pattern reversal): every bar location swings black/white like the flash, the mean luminance never changes, and nothing swings once the bars are too fine to resolve | bar width, coarse to fine (`--exam-sizes`, arcmin) | a grey of the same mean | the bar width at which the response is extrapolated to vanish (sweep-VEP rule), and the finest bar still giving a significant response |
| vernier | coarse vertical bars (`--exam-vernier-bar`, 20') cut into horizontal strips three bars high; every other strip jumps sideways by the step's offset and back at the flicker rate (the vernier VEP of Norcia, Wesemann and Manny, 1999) | the offset, coarse to fine (`--exam-vernier`, arcmin; delivered to a quarter of a device pixel through the canvas's anti-aliasing) | the aligned bars alone, nothing moving | the offset at which the response is extrapolated to vanish (the same limb rule), and the finest offset still giving a response; a hyperacuity, which amblyopia degrades more than acuity |
| contrast | the same reversing grating at one bar width (`--exam-contrast-size`) | Michelson contrast (`--exam-contrasts`) | the grey alone | the contrast at which the response is extrapolated to vanish, as log sensitivity |
| csf | the contrast test again at other bar widths (`--exam-csf-sizes`, 60' and 10') | the contrast at each (`--exam-csf-contrasts`) | the grey alone | a contrast threshold per bar width; with the contrast test's own point, log sensitivity against cycles per degree (a bar of w arcmin is 30 / w cpd) and the bar width where it peaks |
| field | one quadrant of the window flickering white/black, the eyes on the centre dot | the quadrant (`--exam-field`, 2x2; finer grids allowed) | black | amplitude and phase per sector; sectors without a significant response; with two eyes, the sector-by-sector ratio and phase difference, also with each eye over its own flash |
| gaze | the whole window flickering white/black while the fixation dot sits at the centre or out toward an edge (up, down, left, right; `--exam-gaze`, 0.42 of the window from the centre), so the stimulus falls to one side of the fovea | the dot's position; the dot moves 1.2 s before each step's blocks | black | amplitude and phase per position, each over the centre position and over the best; positions without a significant response; with two eyes, the position-by-position ratio |
| meridian | the reversing grating at one bar width (`--exam-meridian-size`, 14') turned to several orientations | the bars' tilt, degrees clockwise from vertical (`--exam-meridians`, 0, 45, 90, 135) | the grey alone | amplitude per orientation over the best; the weakest meridian and its share of the best; orientations without a response |
| photometry | the window swapping between a red field and a green field at the flicker rate (heterochromatic flicker photometry); the pair's luminance is fixed and the share given to red is the step's variable; `--exam-photometry by` adds blue/yellow | the first field's share, 0.2 to 0.8 (`--exam-mixes`) | the steady mixture of the two fields: the same mean luminance and chromaticity, no exchange | amplitude against the share; the null fitted as the exchange's luminance modulation for an eye whose two fields match at share p0, so p0 is the eye's isoluminant point (0.5 for the display's standard observer; toward red for a red-weak eye); the response left at the null, which is the chromatic pathway's; between the eyes, the difference of the nulls |
| colour | the whole window flickering red, green or blue against black | the plane | black | amplitude per plane, R:G:B |

`--exam-style onset` gives the older grating stimulus (the grating appearing
and disappearing on the same-mean grey, pattern onset/offset) for a
comparison; reversal is the default because at every bar location it is the
flash, which is what the rig is known to see. In a reversal block the logged
flicker edges are the frames on which the bars swapped, so the response is
scored and phased at the reversal rate, the session frequency.

Defaults: 4 s ON / 4 s OFF, five pairs of flash, two ON/OFF pairs per step of
the others, the whole window (`--exam-patch 1`); seven rates asked (five or
six delivered), eight bar widths, five vernier offsets, six contrasts, two
more bar widths at four contrasts, four quadrants, five gaze positions, four
meridians, five red/green shares, three planes: about 15.5 minutes per eye.
`--exam-tests` picks a subset in the fixed order (the seven tests of the
28 October protocol, `flash,temporal,acuity,contrast,field,gaze,colour`,
take 9.5 minutes per eye).

## The tests added on 29 October 2026

Four tests that the rig could show and had not: each is a known VEP
measure, each is read with the same exact test per step and the same limb
rule, and each has a dial in the phantom so the gate can check it.

**Vernier.** Coarse bars (20', well above any acuity limit here) are cut
into horizontal strips three bars high, and every other strip jumps
sideways by the step's offset and back at the flicker rate; OFF is the same
bars aligned and still. Nothing changes but the alignment, so the response
is to the offset alone: the vernier VEP (Norcia, Wesemann and Manny, 1999),
whose threshold in a normal eye lies well below the acuity limit (a
hyperacuity) and which amblyopia degrades more than acuity does. The offset
is delivered to a quarter of a device pixel: the shifted strip is drawn at
a fractional position and the browser's anti-aliasing renders the edge
between the pixels. On the headset at about 20 px/deg a quarter pixel is
0.75', so the finest offset asked (0.5') lands there; a normal eye's vernier
threshold is finer than that, and the read-out will say "below the finest
offset shown" rather than give a number. The test is for the eye that does
not get there.

**CSF.** The contrast test at 20' is one point of the contrast sensitivity
function; the csf test repeats it at 60' (0.5 cycles per degree) and 10'
(3 cpd, near where the function peaks) with four contrasts each, and the
read-out joins the three thresholds into log sensitivity against spatial
frequency with the peak named. The shape, not the single number, is what
early cataract, glaucoma and amblyopia change. A fine bar width beyond the
eye's acuity gives no threshold there, which is the honest reading. The
phantom's contrast cut does not depend on bar width, so the gate runs the
fine width at 14', inside its left eye's acuity, and asks every csf
threshold to land in the contrast band.

**Meridian.** The reversing grating at 14' turned to 0, 45, 90 and 135
degrees (clockwise from vertical; the page rotates the drawn grating about
the window's centre and clips it to the window). With spherical blur every
orientation suffers alike; an astigmatic blur leaves the bars along one
meridian sharp and those at right angles smeared, so one orientation
responds strongly and the one 90 degrees from it weakly. The read-out
gives each orientation over the best and names the weakest and its share.
Two cautions: the normal eye's oblique effect makes 45 and 135 a little
weaker at fine bars, and the bar width must sit between the eye's acuity
and the blur (14' is the default; `--exam-meridian-size` moves it). The
phantom dials an axis and a gain (`--astigmatism right:90:0.4`: 0.4 of
full along the horizontal bars, rising as cos² to full for vertical).

**Flicker photometry.** Heterochromatic flicker photometry is the oldest
colour-vision measure that needs no words: a red field and a green field
exchanged at a rate the chromatic pathway cannot follow give a luminance
flicker whose depth is the difference in brightness of the two fields to
this eye, and the setting at which the flicker vanishes is the eye's
isoluminant point. The test runs it as a sweep: the pair's luminance is held
(0.2 of white, red at most 0.75 of its primary) while the share given to red
runs from 0.2 to 0.8 (`--exam-mixes`), the display's photometric match
being 0.5 (sRGB primaries at 0.2126 and 0.7152 of white). OFF is the steady
mixture of the two fields, so the mean luminance and the chromaticity never
change; only the exchange comes and goes. The read-out fits amplitude =
a·M(share; p0) + b over p0, where M is the exchange's luminance modulation
for an eye whose red efficiency is scaled to match at p0; p0 is the null
(the eye's isoluminant share), b the chromatic response left there, and the
null's shift from 0.5 is the reading: a protan eye (red-weak) needs more
red, so its null lies above 0.5; a deutan's lies a little below. Two
limits: the display's primaries are taken as sRGB, which is not measured,
so the absolute null carries the display's error while the difference
between the two eyes does not; and at 12 to 15 Hz the chromatic response is
small but not zero (b), which is why the null is fitted rather than read
off the dimmest step. `--exam-photometry rg,by` adds a blue/yellow exchange
(the tritan axis, the pair's luminance 0.08 of white so blue stays within
its primary). Not built: the Ishihara-style figure in colour noise, which
is a reading task.

The gaze test (added 28 October 2026 after the first sitting that measured a
human response) answers "do all parts of the eye respond?" in the only way a
single occipital electrode can: the whole window flickers while the eye
fixates the centre, then a dot 84 % of the way to the top, bottom, left and
right edges of the window. Looking left, the window lies in the right visual
field from the fovea out to the window's far edge, and so on, which puts the
flicker on retina the central-fixation tests never reach. The read-out gives
each position's amplitude over the centre's and over the best, the positions
without a response, and between the eyes the ratio position by position. The
quadrant test (field) covers the central field at fixation; the gaze test
reaches outward. Both are coarse: with one electrode the response is dominated
by the central few degrees (cortical magnification), so a peripheral loss shows
only as a missing or weak position, never as a map.
`--exam-eye right,left` runs the right eye and then the left with a "cover
your other eye" screen between them; `both` tests binocularly. When the flash
gives no response (p >= 0.05) the page asks for an electrode check (wet the
top electrode, press the ear pads in) and the flash is tried again, twice at
most; the exam then goes on and the report says that the eye had no reference.

## Running it

Phantom, no hardware (the gate, about 25 minutes with the eleven tests):

```
python exam_gate.py
```

On the Quest:

```
.\quest_run.ps1 -Run exam -Ppd 20
```

The run writes under `runs/<session>/`: `exam_<eye>_<test>_log.csv` (one row per
frame, the usual columns plus `size`, `contrast`, `plane`, `angle` (the bars'
tilt) and `mix` (photometry's first-field share); the vernier log carries its
offset in the `size` column; in a reversal block
`flicker_on` marks the frames on which the bars swapped), `exam.json` (every
step's amplitude, p, phase and the page's description of what it showed, the
flash attempts per eye),
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
amplitude at the flicker frequency and its double, on the eye's reference
channel: the one the full-field flash responded on most, or, without a flash
response, the channel with the largest ON/OFF difference over the whole test)
is taken against log bar width or log contrast. The fitted limb starts one step below the finest (faintest)
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
The flicker sweep, below, is what turns the phase into a latency.

## The flicker sweep: temporal tuning and apparent latency

The second test of each eye is the flash again at several rates, one step
per rate (`--exam-rates 6,7.5,10,12,15,20,30`; the page shows each at the
nearest rate it can hold with a whole number of frames per half period and
drops a rate that repeats one, so a 60 fps panel delivers 6, 7.5, 10, 15 and
30 Hz and the Quest's 72 fps panel 6, 7.2, 9, 12, 18 and 36). Each step is
scored at its own rate, the OFF blocks measured at that rate too, and the
line power is the hann-windowed projection on the rate itself rather than a
mean over the periodogram's nearest bins, which read a rate on a bin 0.83 as
strong as one half a bin off (the same for every step of a single-rate test,
so the thresholds never saw it; the sweep cannot carry it).

The read-out gives, per rate, the amplitude with the OFF blocks' line power
taken out in quadrature (a step inside the alpha band is read against OFF
blocks that carry the eyes-open alpha; when the flicker blocks the alpha the
correction over-subtracts, so a dip at the alpha rate is read with that in
mind), the best rate, and the rate above it where the response has fallen to
half, interpolated in log rate between the bracketing steps (a rate without
a response counts as nothing; a response still over half at the fastest rate
shown is said so). This is the eye's temporal tuning: in the camera runs the
subject's response was strongest at 12 Hz, weak at 15 and gone at 20, which
is what set the session rate, and the rate an eye stops following is a
measure in its own right.

The apparent latency comes from the phase. A response delayed by tau has a
phase that falls by 360 tau degrees per hertz of rate, so over the rates that
respond the slope of the phase against rate gives tau with the waveform's own
constant offset taken out (Regan, 1966). The fit (`fit_latency` in
`analysis/exam_readout.py`) first searches tau from 30 to 300 ms for the
delay whose residual phases agree most, which fixes each phase's turn count
(the panel's rates repeat only beyond 1.6 s, so the search has one answer),
then puts a line through the unwrapped phases weighted by the relative
amplitude; its slope is the latency and its scatter the standard error. Three
responding rates are needed; on synthetic phases with 3 degrees of noise the
fit is good to half a millisecond with five rates and to two milliseconds
with the four rates up to 12 Hz an eye that stops at 12 would give. The
stream's transport delay stays in the number (on the phantom the rebuilt
EEG clock runs 3 / fs = 11.7 ms early, so a 120 ms eye reads 108), and it is
the same for both eyes, so the difference between the eyes is the
measurement, the one the clinical pattern VEP is used for: an eye whose
response comes later than the other's. Each step's single-rate delay
(`delay_ms`) stays ambiguous by whole periods and is not the latency.

## What the phantom gate checks

`exam_gate.py` runs the full stack against `phantom_subject.py` with a logistic
visibility in log bar width and log contrast (`--acuity-cutoff`,
`--contrast-cutoff`, per eye), a dead sector (`--field-loss`), the colour
gains, a latency and a flicker cut per eye (`--latency`, `--flicker-cut`: the
response is half at that rate) and the alpha as a 1.5 Hz band rather than a
pure line (`--alpha-band`): right eye cutoff 8 arcmin and contrast 0.08 with
quadrant 3 dead, 120 ms late, holding to 40 Hz; left eye 16 arcmin and 0.2,
whole, 150 ms late, half gone by 24 Hz; green 1, red 0.7, blue 0.4; the
phantom's Oz (AUX) carries the strongest response. Because a line through
the rising limb crosses zero below the half-visibility point, the gate asks
that the flash respond in both eyes on AUX and that every other test be read
on that channel, that each fitted threshold lie where the phantom's
visibility runs from 0.02 to 0.6, that the left eye's thresholds be at least
1.3x the right's, that the field names exactly the dead quadrant, that green
>= red > blue in both eyes, and of the flicker sweep that every rate shown
responds in both eyes, that the right eye is still over half at the fastest
rate shown and the left eye's half rate lies within 15 % of 24 Hz, that each
eye's apparent latency lies within 6 ms of the phantom's (its latency less
the 11.7 ms packet offset) and that the left is 30 +/- 6 ms later than the
right, and of the gaze test that the whole window responds at every fixation
position in both eyes: twenty-five checks. The 29 October tests add dials
and checks: the phantom's vernier cut-off (`--vernier-cutoff right:1,left:2`
arcmin, the same logistic with width 0.3), an astigmatic axis and gain for
the right eye (`--astigmatism right:90:0.4`) and none for the left, and the
red/green match (`--isolum right:0.5,left:0.65`: the left eye is red-weak,
its two fields matching when red carries 0.65 of the luminance; the response
left at the null is 0.1 of the full exchange). The gate asks that each eye's
vernier threshold lie in its band and the left's be at least 1.3x the
right's; that the csf test be scored at both of its bar widths and every
threshold there lie in the contrast band (the phantom's contrast cut is the
same at every bar width); that the right eye's weakest meridian be the 90
degree one at no more than 0.6 of the best, and the left eye's every
meridian respond within 0.7 of the best; and that each eye's red/green null
lie within 0.07 of the dialled share, the left's 0.15 +/- 0.07 above the
right's: thirty-nine checks. The band's floor: the extrapolated crossing lands where the
phantom's visibility is 3 to 5 % in every gate so far (2.8 % once, the left
contrast of the first run of the new protocol, which is why the floor is 2 %
and not the 3 % first used).

Result on 26 October 2026 with the earlier protocol (`runs/exam_gate2`, patch
0.6, pattern onset/offset, 4x4 field, no flash; 13.7 min, headless Chrome at
60 fps, 15.01 Hz delivered): all twelve checks pass. Right eye: acuity threshold
3.9 arcmin (finest bar with a response 6'), contrast threshold 0.031 (faintest
with a response 0.06), 15/16 sectors, sector 6 silent, R:G:B 0.70:1.00:0.42.
Left eye: 7.8 arcmin (finest 10'), contrast 0.067 (faintest 0.12), 16/16
sectors, R:G:B 0.68:1.00:0.36. Response phase 42 to 47 degrees in every
responding sector of both eyes (spread 0 to 3 degrees), random in the silent
one. A first gate on the same protocol had lost the left eye's 0.12 contrast
step to noise (two 3 s blocks at a response of 0.13 of full), which is what
put the anchor step into the limb fit.

Result on 27 October 2026 with the present protocol (`runs/exam_gate4`,
10.2 min, headless Chrome at 60 fps, 15.00 Hz delivered; the whole window,
flash first, reversing bars, 2x2 field): all sixteen checks pass. Both eyes'
flash responds on AUX (p 0.004, line SNR about 3000 and 4900) and every test
is read on it. Right eye: acuity threshold 3.8 arcmin (finest bar with a
response 6'), contrast threshold 0.027 (faintest with a response 0.06), 3/4
quadrants, quadrant 3 silent, R:G:B 0.73:1.00:0.41. Left eye: 7.9 arcmin
(finest 10'), 0.058 (faintest 0.12), 4/4 quadrants, R:G:B 0.71:1.00:0.43;
flash amplitude left/right 0.99, field amplitude left/right 0.97 (0.98 with
each eye over its own flash), phase difference 0. The left contrast
threshold, 0.058, lies where the phantom's visibility is 2.8 %, which is
what set the band's floor at 2 %. The channel rule was changed by the first
run of this protocol (`exam_gate3`, stopped): the flash had picked TP10 over
a 1.5x stronger AUX because the log-SNR statistic's ON-OFF difference
saturates on a strong response; the test's own channel is now the largest
median ON amplitude among the channels whose own ON/OFF difference is
significant.
A second run under the final code (`runs/exam_gate5`, 10.4 min) passes
all sixteen again: right eye 4.0 arcmin and 0.026 with quadrant 3 silent,
left eye 6.8 arcmin and 0.087; the left contrast crossing moved from 2.8 %
to about 6 % of the phantom's visibility between the two runs, which is the
fit's own spread over two 3 s blocks per step.

Result on 27 October 2026 with the flicker sweep added and the line power
taken as the projection on the rate itself (`runs/exam_gate6`, 12.9 min,
five rates delivered at 60 fps: 6, 7.5, 10, 15, 30 Hz): all twenty-three
checks pass. Right eye: flash 1.20 on AUX (line SNR 6650), every rate
responds, still over half at 30 Hz (0.79 of the 6 Hz amplitude; the
phantom's cut is 40), apparent latency 109 +/- 1 ms against the 108 expected
(120 less the 11.7 ms packet offset), acuity 3.9 arcmin, the contrast line
crossing at 0.026 below the faintest step shown (0.03 responded this time),
quadrant 3 silent, R:G:B 0.78:1.00:0.41. Left eye: flash 1.03, every rate
responds, half by 22.3 Hz (phantom 24; the log-linear interpolation between
15 and 30 Hz of a fourth-order low-pass reads low by about 3 %), apparent
latency 138 +/- 1 ms against 138, acuity 8.0 arcmin (finest bar with a
response 14'), contrast 0.102, 4/4 quadrants, R:G:B 0.67:1.00:0.41. Between
the eyes: latency left minus right +29 +/- 1 ms (dialled +30), sweep
amplitude 0.99x, flash 0.86x, field 0.89x (1.04x over each eye's own
flash). The phases of the right eye's sweep sat within 7 degrees of the
108 ms model at every rate (33, -17, -116, 45, 173 degrees at 6, 7.5, 10,
15, 30 Hz against 36, -22, -120, 45, 180). The exact projection reads the
flash 1.20 where the three-bin mean read 1.04, and the ON/OFF statistic's
OFF values scatter a little more with it (one sample of the noise spectrum
instead of a mean of three), so a step at the edge of detection can go
either way: the left eye's 10' step, significant in gates 4 and 5, was not
here, and the threshold moved from 6.8-8.0 to 8.0 arcmin, inside the band.

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
the Quest run uses a patch the full height of the window. `python exam.py
merge <out> <session>:<eye>[=<label>] ...` joins eyes from several sessions
into one report; `python exam.py <session>` re-scores one.

## What the eye-as-a-camera runs changed in the protocol (27 October 2026)

The sitting's one clear response was the full-window colour flicker, which is
the stimulus closest to the camera's calibration; the gratings on a 0.6 patch
and the sixteen small sectors gave nothing a threshold could be fitted to. The
camera runs had already measured this: full-field flicker gives an Oz line SNR
of 20 to 190 in a good sitting, a fixated patch a few units, a cell-sized
square a line below the flanks even at eight seconds, and only the fixated
region responds at all. So the exam now (1) fills the window with every
stimulus; (2) opens each eye with the full-field flash itself, scored like the
camera's calibration gate (five ON/OFF pairs, exact permutation to p = 0.004),
retried after an electrode check when it fails, and used as the eye's
reference channel and amplitude; (3) shows the gratings as pattern reversal,
so that where a bar is resolved the retina sees the same black/white swing as
in the flash, with the mean luminance constant (the standard sweep-VEP
stimulus; Norcia and Tyler, 1985), instead of a half-swing to grey for half
the time; and (4) tests the field as four quadrants, each a quarter of the
window touching the fixation, instead of sixteen small sectors. The response
frequency stays the session's (12 Hz on the Quest, where the subject's
response is strongest); in a reversal block each bar location swings at half
that rate and the response locks to the swaps. Added the same afternoon:
(5) the flicker sweep as the second test, the flash at several rates,
because the flash is the one stimulus the sitting answered and the camera
runs had found the subject's rate tuning by sweeping it; it adds the apparent
latency, which the single-rate phase could not give.

## What this is and is not

These are response amplitudes from one occipital electrode against a flickering
display, with a stimulus whose absolute scale depends on a measured pixels-per-
degree. They say how the eye's response changes with bar width, contrast,
place in the field and colour, and how the two eyes differ. They are not a
refraction, a visual field test or a diagnosis; the next steps (a blur study
with trial lenses against a letter chart, then a normative cohort) are what
would tie them to clinical measures.

Result on 28 October 2026 with the gaze test added (`runs/exam_gate`, 17.9 min,
headless Chrome at 60 fps, 15.00 Hz delivered): all twenty-five checks pass.
Right eye: acuity 3.8', contrast 0.029, quadrant 3 silent, latency 108 ms,
over half at 30 Hz, gaze 5/5 with every position 0.94 to 0.98 of the centre;
left eye: 7.8', 0.082, 4/4 quadrants, 138 ms, half by 22.5 Hz, gaze 5/5 at
0.90 to 0.96 of the centre; latency difference +30 ms; gaze amplitude
left/right 0.89 to 0.92 at every position. The first run with the test
(`exam_gate_run_1858.log`) failed only the two gaze checks, because the
phantom follows a fixed list of exam logs and the new log was not on it: the
phantom was blind through the gaze blocks while every other check held. That
is what the gate is for.

Result on 29 October 2026 with the four new tests (`runs/exam_gate`, 29.7 min,
headless Chrome at 60 fps, 15.00 Hz delivered; the gate's meridian bars at
20' and its csf bar widths 60' and 14'): all thirty-nine checks pass. Right
eye: vernier threshold 0.37' (finest offset with a response 1.00'; dialled
cut 1', band 0.31 to 1.13), CSF log sensitivity 1.64 at 0.5 cpd, 1.52 at
1.5 and 1.52 at 2.1 cpd (contrast thresholds 0.023, 0.030 and 0.030 against
a cut of 0.08), weakest meridian 90 degrees at 0.47 of the best (dialled
0.4 along that axis), red/green null at a red share of 0.49 (dialled 0.50)
with 14 % of the response left at the null (dialled 0.1); the earlier
measures as before (latency 109 ms, acuity 3.7', contrast 0.030, quadrant 3
silent, gaze 0.92 to 0.99 of the centre). Left eye: vernier 0.88' (finest
2.00'; cut 2', band 0.62 to 2.26), CSF 1.21, 1.27 and 1.02 (thresholds
0.062, 0.053 and 0.095 against 0.2), every meridian within 0.90 of the best,
null at 0.65 (dialled 0.65) with 16 % left; latency 139 ms, acuity 7.9',
contrast 0.053, 4/4 quadrants. Between the eyes: vernier 2.36x, CSF 2.7x,
1.8x and 3.2x at the three bar widths, null shift +0.16 (dialled +0.15),
latency +30 ms. The vernier sweep shows the page's quarter-pixel offsets
delivered: the right eye's response fell from full at 2' through half at 1'
to nothing at 0.5', the phantom's logistic in the log offset. The two
widest bar widths of the right eye's CSF cross below the faintest contrast
shown, which the read-out marks as beyond the range: with a contrast cut of
0.08 and steps down to 0.06, the limb's last point is near the floor.
