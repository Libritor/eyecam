# Eye as a camera on consumer hardware: the Fletcher angle and the road to an ICCE paper

Written 2026-10-11, after the two colour runs of 2026-10-10.

## Who Rich Fletcher is, in the terms that matter here

Research Scientist at MIT, founder and head of the Mobile Technology Group
(Mechanical Engineering), Assistant Professor of Psychiatry at UMass Medical
School, Research Scientist in Psychiatry at MGH, Instructor at Harvard
Medical School. Twenty years of wireless sensors and RFID; MIT PhD on
low-cost electromagnetic tagging (Gershenfeld's group). His group builds
low-cost mobile and wearable sensors, diagnostic algorithms and AI for
public health and behavioural medicine, and runs clinical studies in the US
and in India with community health workers (phone-based pulmonary and
thermal screening, PPG hypertension, anaemia, the ASHA kit). With Rosalind
Picard he patented the "Just In Time Intervention" (JITAI) framework for
mobile behavioural interventions. Funding from NIH, Gates, USAID, Vodafone,
Tata Trust. Application areas on his CV: precision diagnostics, global and
public health, behavioural medicine and mHealth, mental health and
psychophysiology.

So "Fletcherized" means: consumer-priced hardware, no specialist operator,
an objective number that a community health worker can act on, a behavioural
or mental-health reading as a by-product, and a path to a field study.

## What this project already has that fits

| Fletcher theme | what exists in this repo | state |
|---|---|---|
| consumer-priced sensing | Muse 2 ($250) + one aux cup at Oz, Quest 3S ($300), a laptop; no amplifier, no lab | done, every result in README |
| objective, literacy-free readout | the image is read from the EEG while the subject only looks at a dot; no button, no naming | done ("NO" r 0.91; eight colours 24 of 32) |
| colour as a measurement | per-colour calibration gives a red / green / blue response profile before any scan (`planes_calib.json`: tonight R +5.8, G +3.9, B +5.0; blue 0.4x red in power on 2026-10-07) | done; not yet a screening claim |
| acuity as a measurement | tumbling-E chart, big E and Landolt rings as targets (`targets_img/`), the step wedge for contrast | targets built, not yet run on a person |
| psychophysiology by-product | State of Flash: blink suppression while a cell flickers and release when it goes dark; blink rate doubled between the two runs of one evening; no drowsiness markers (`docs/state_of_flash.md`) | done on one subject |
| JITAI hook | the blink suppress-release pattern and the blink-rate rise are measurable live from the same headband; a "rest your eyes" prompt is a page message away | not built |
| field robustness | awake hold, freeze detection and redo, link-watch, loose-electrode diagnosis, phantom gates | done |
| electromagnetic tagging | the frequency tag on each pixel is to the visual field what his chipless RFID tags were to a reader: the same word, a one-line aside at most | |

## The Fletcherized paper (working title)

"A consumer EEG headband and a consumer VR headset as an objective visual
screener: reading images, colours and eye strain from the SSVEP."

Contributions, in the order a reviewer will weigh them:

1. First steady-state visual evoked imaging on a consumer VR headset with a
   four-channel consumer headband plus one occipital cup: a legible word
   (r 0.91), and eight colours by three calibrated planes (24 of 32 cells,
   hue right in 19 of 24). Delivered-frequency bookkeeping (72 fps, 12 Hz
   frame-exact; 9 Hz from the blue sweep) as method.
2. A per-colour response profile from the calibration alone (red, green,
   blue line statistics in a few minutes) as a candidate objective
   colour-vision screen. Honest scope: our planes are not isoluminant, so
   the profile mixes luminance and chromatic response; separating a colour
   deficiency needs isoluminant red/green and more than one subject. Stated
   as future work with the stimulus already in hand (`--mode smooth`
   sampled-sinusoid flicker can carry any pair at matched luminance).
3. An objective acuity read-out: the tumbling E and Landolt C read from the
   EEG at known cell sizes, the sweep-VEP idea without the gaze scan as the
   next step.
4. Eye strain from the same recording: blink suppression under flicker and
   its release, blink rate as a between-session state, with the method note
   that on a Muse under a headset the blink is read through the Fpz
   reference on the far channels, not on AF7/AF8. One subject; the study
   design in `docs/state_of_flash.md`.
5. Engineering for the field: what broke (standby freezes, a hidden browser
   tab, a loose ear electrode at minute 11, an orphaned recorder) and the
   guards that now catch each, with the phantom gates as the regression test.

ICCE fit: consumer electronics, wearables, AR/VR, health and wellness tracks
all apply; the hardware is the kind the conference is about.

## Where to submit, honestly

- ICCE 2027 (Las Vegas, 8-11 January 2027): the paper deadline was
  15 September 2026 and has passed; acceptances go out 31 October 2026.
- ICCE-Berlin 2027 (September 2027, with IFA): no 2027 call yet; the 2025
  call closed 30 May 2025, so expect a late-May 2027 deadline. The right
  target if the data are in by April 2027.
- ICCE 2028 (Las Vegas, January 2028): deadline about September 2027.
- Sooner, smaller: an IEEE GEM / ICCE-Asia late-news or work-in-progress
  slot if one opens, with the colour result and the eye-strain readout as a
  two-figure short paper.

## A real colour picture

The planes mode already takes an image path as the colour target
(`targets.color_target`: the PNG is resized to the grid, so a PNG drawn at
the grid size is used pixel for pixel). What the eye can deliver is a
pixel picture in the eight pure colours, not a photograph: on the 8 by 4
grid one cell in four was wrong tonight, so a face has to survive a quarter
of its cells being off, which a cartoon with big flat regions does and a
downsampled photo does not (the grey portrait work of 2026-10-03 reached the
same conclusion for luminance).

`targets_img/make_officer.py` draws a police officer at 8 by 10 cells in
four colours (blue cap with a yellow badge, yellow face with black eyes and
mouth, blue uniform with a badge, on white; 80 positions, lit / dark
balanced in every plane) and a 12 by 14 version (168 positions) for a longer
sitting. Command, about 33 minutes with a rest break every 40 cells:

```
python xr_session.py --source osc --mode planes --color-target targets_img/officer_8x10.png \
    --color-grid-w 8 --color-grid-h 10 --freq 12 --calib-blocks 6 --calib-on 8 --calib-off 8 \
    --calib-style bw --calib-size 1 --plane-spc 5,5,8 --plane-passes 1 --rest-every 40
```

Expect the cap, face and uniform as blocks with a few wrong cells; the badge
and the mouth are single cells and may or may not come through. The 12 by 14
version is a second sitting on another day (about 60 min at the same
dwells), the one for the paper's hero figure if the 8 by 10 works.

## Next steps, in order

1. One clean colour run of `eight` on a fresh day, Oz wet, ear badges
   checked on the page before the trigger (the TP10 lesson), blue once:
   the benchmark number for the paper, and the day's colour profile.
2. The officer at 8 by 10 on the same day or the next: the "real thing in
   colour" figure.
3. A loose-channel guard in the decode (drop a channel whose RMS triples
   against its calibration; `analysis/planes_passes.py` showed blue's two
   passes read 26 of 32 without the ears against 19 with them). Small
   change, gated on the phantom with an injected artifact.
4. The tumbling-E run (targets exist): the acuity read-out for contribution 3.
5. Two more subjects (Minu, Khalil) for the colour profile and the blink
   readout: n = 3 is the floor for a screening claim at a conference.
6. The eye-strain prompt (JITAI): a page message when the blink-release
   pattern says the eyes are drying; a day of work, and the sentence that
   makes the paper Fletcher's kind.
7. Write the paper from the figures that exist (`docs/figures/fig17_vr_planes9_colour.png`,
   `state_of_flash_ssvep_fatigue.png`, `state_of_flash_eyes.png`), aim at
   ICCE-Berlin 2027 (May), with ICCE 2028 as the fallback.
