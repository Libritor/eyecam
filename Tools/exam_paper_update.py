"""Rewrite the protocol paragraphs of docs/paper/exam2027/main.tex for the
full-window / flash-reference / reversal / quadrant exam, with the gate
numbers read from a passed gate session's exam_readout.json.

    python Tools/exam_paper_update.py runs/exam_gate4 docs/paper/exam2027/main.tex
"""
import json
import os
import sys

import numpy as np

session, tex = sys.argv[1], sys.argv[2]
ro = json.load(open(os.path.join(session, "exam_readout.json"), encoding="utf-8"))
R, L = ro["eyes"]["right"], ro["eyes"]["left"]
dead = R["field"]["not_responding"]
phases = []
for eye in (R, L):
    f = eye["field"]
    phases += [f["phase_deg"][k] for k in f["responding"] if f["phase_deg"][k] is not None]
ph_lo, ph_hi = (min(phases), max(phases)) if phases else (float("nan"), float("nan"))
n_checks = 16

s = open(tex, encoding="utf-8").read()
reps = [
    # abstract
    ("A hardware-free phantom with set cut-offs and a dead sector passes all twelve checks of a full-stack gate. "
     "The first sitting on a person fitted no threshold because two electrodes had come loose unnoticed; "
     "it produced the contact check a device in a health worker's hands must have.",
     "A hardware-free phantom with set cut-offs and a dead quadrant passes all sixteen checks of a full-stack gate. "
     "The first sitting on a person fitted no threshold because two electrodes had come loose unnoticed and the "
     "patch was small; it produced the contact check, and the full-window flash reference, a device in a health "
     "worker's hands must have."),
    # method
    ("Each test is ON/OFF blocks of flicker (4~s, two pairs per step, 12~Hz on the 72~Hz display): a grating "
     "appearing and disappearing on a grey of the same mean luminance with bars from 40 to 3.5~arcmin (acuity) "
     "or contrast from 1 to 0.03 (contrast), one sector of a $4\\times4$ grid at a time (field), the patch in "
     "red, green or blue (colour). Each step's ON blocks are scored against all the test's OFF blocks by the "
     "line power at the flicker frequency and its double, with an exact permutation test; a channel above "
     "60~$\\mu$V rms through a test is set aside. Thresholds follow the sweep-VEP rule: a line through the limb "
     "from the step below the finest responding one up the rise, extrapolated to zero. Seven minutes per eye.",
     "Every stimulus fills the window: the camera runs had shown the response grows with the flickering area "
     "and a small square gives nothing. Each eye opens with the window flickering white/black, the camera's "
     "calibration stimulus (five ON/OFF pairs of 4~s at 12~Hz on the 72~Hz display); it must respond, and the "
     "channel and amplitude it responds with are the reference for the eye's tests. Then, two pairs per step: "
     "a black/white grating whose bars swap places at the flicker rate, so each resolved bar swings like the "
     "flash while the mean luminance never changes (pattern reversal \\cite{norcia1985sweep}), bars from 40 to "
     "3.5~arcmin (acuity) or contrast from 1 to 0.03 (contrast); one quadrant of the window at a time (field); "
     "the window in red, green or blue (colour). Each step's ON blocks are scored against all the test's OFF "
     "blocks by the line power at the flicker frequency and its double, with an exact permutation test; a "
     "channel above 60~$\\mu$V rms through a test is set aside. Thresholds follow the sweep-VEP rule: a line "
     "through the limb from the step below the finest responding one up the rise, extrapolated to zero. Six "
     "and a half minutes per eye."),
    # gate
    ("The gate dials in a right eye at 8~arcmin and contrast 0.08 with one dead sector, a left eye twice as "
     "coarse and whole, gains 0.7/1/0.4 for red/green/blue, and asks twelve things: every stage delivered, each "
     "threshold inside the band where the phantom's visibility runs from 0.03 to 0.6, the left eye at least "
     "$1.3\\times$ coarser, exactly the dead sector named, green~$\\geq$~red~$>$~blue. All twelve pass "
     "(Fig.~\\ref{fig:gate}): right eye 3.9~arcmin and 0.031 with sector six silent, left eye 7.8~arcmin and "
     "0.067, response phase within $42$ to $47^\\circ$ in every responding sector.",
     "The gate dials in a right eye at 8~arcmin and contrast 0.08 with one dead quadrant, a left eye twice as "
     "coarse and whole, gains 0.7/1/0.4 for red/green/blue, Oz the strongest channel, and asks sixteen things: "
     "every stage delivered, the flash responding in both eyes on Oz and every test read on it, each threshold "
     "inside the band where the phantom's visibility runs from 0.02 to 0.6, the left eye at least $1.3\\times$ "
     "coarser, exactly the dead quadrant named, green~$\\geq$~red~$>$~blue. All sixteen pass "
     f"(Fig.~\\ref{{fig:gate}}): right eye {R['acuity']['threshold']:.1f}~arcmin and "
     f"{R['contrast']['threshold']:.3f} with quadrant {', '.join(str(k + 1) for k in dead)} silent, left eye "
     f"{L['acuity']['threshold']:.1f}~arcmin and {L['contrast']['threshold']:.3f}, response phase within "
     f"${ph_lo:.0f}$ to ${ph_hi:.0f}^\\circ$ in every responding quadrant."),
    # gate figure caption
    ("(c)~The right eye's field, the cross its dead sector.",
     "(c)~The right eye's field, the cross its dead quadrant."),
    # first sitting
    ("The exam now measures every channel before each eye and, when one is above 60~$\\mu$V, tells the subject "
     "on the headset which electrode to press in and waits; the read-out names any channel set aside.",
     "The exam now measures every channel before each eye and, when one is above 60~$\\mu$V, tells the subject "
     "on the headset which electrode to press in and waits; the read-out names any channel set aside. And the "
     "sitting's one response was the stimulus closest to the camera's: so the exam now fills the window, opens "
     "each eye with the flash as its reference, swaps the grating's bars instead of fading them, and tests the "
     "field as quadrants (Fig.~\\ref{fig:diagram})."),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
open(tex, "w", encoding="utf-8").write(s)
print("main.tex updated; gate numbers:",
      f"right {R['acuity']['threshold']:.2f}' / {R['contrast']['threshold']:.3f}, dead {dead}; "
      f"left {L['acuity']['threshold']:.2f}' / {L['contrast']['threshold']:.3f}; phases {ph_lo:.0f}-{ph_hi:.0f}")
