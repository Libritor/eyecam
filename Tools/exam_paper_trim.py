"""Trim docs/paper/exam2027/main.tex back to two pages after the protocol
rewrite (Tools/exam_paper_update.py): shorter method, sitting and outlook
sentences, slightly smaller gate and photo figures.

    python Tools/exam_paper_trim.py docs/paper/exam2027/main.tex
"""
import sys

tex = sys.argv[1]
s = open(tex, encoding="utf-8").read()
reps = [
    ("come loose unnoticed and the patch was small; it produced the contact check, and the full-window flash "
     "reference, a device in a health worker's hands must have.",
     "come loose unnoticed and the stimulus was small; it produced the contact check and the full-window flash "
     "reference a device in a health worker's hands must have."),
    ("Each eye opens with the window flickering white/black, the camera's calibration stimulus (five ON/OFF "
     "pairs of 4~s at 12~Hz on the 72~Hz display); it must respond, and the channel and amplitude it responds "
     "with are the reference for the eye's tests. Then, two pairs per step: a black/white grating whose bars "
     "swap places at the flicker rate, so each resolved bar swings like the flash while the mean luminance "
     "never changes (pattern reversal \\cite{norcia1985sweep}), bars from 40 to 3.5~arcmin (acuity) or "
     "contrast from 1 to 0.03 (contrast); one quadrant of the window at a time (field); the window in red, "
     "green or blue (colour).",
     "Each eye opens with the window flickering white/black, the camera's calibration stimulus (five ON/OFF "
     "pairs of 4~s at 12~Hz on the 72~Hz display), which must respond and whose channel and amplitude are the "
     "reference for the eye's tests. Then, two pairs per step: a grating whose bars swap places at the flicker "
     "rate, each resolved bar swinging like the flash at a constant mean luminance (pattern reversal "
     "\\cite{norcia1985sweep}), 40 to 3.5~arcmin (acuity) or contrast 1 to 0.03 (contrast); one quadrant at a "
     "time (field); the window red, green or blue (colour)."),
    ("through a logistic visibility in bar width and contrast with a cut-off per eye, dead sectors and colour gains.",
     "through a logistic visibility in bar width and contrast with a cut-off per eye, dead quadrants and colour gains."),
    ("gains 0.7/1/0.4 for red/green/blue, Oz the strongest channel, and asks sixteen things:",
     "gains 0.7/1/0.4 for red/green/blue, and asks sixteen things:"),
    ("And the sitting's one response was the stimulus closest to the camera's: so the exam now fills the "
     "window, opens each eye with the flash as its reference, swaps the grating's bars instead of fading them, "
     "and tests the field as quadrants (Fig.~\\ref{fig:diagram}).",
     "Its one response was the stimulus closest to the camera's; hence the full window, the flash reference "
     "and the reversing bars of Section~\\ref{sec:rig}."),
    ("That is the hook for a just-in-time adaptive intervention \\cite{nahum2018jitai}: the device that has "
     "just measured an eye can notice the eyes are tired and say so.",
     "That is the hook for a just-in-time adaptive intervention \\cite{nahum2018jitai}: the device that has "
     "just measured an eye can say the eyes are tired."),
    ("Open before a second sitting: the headset's pixels per degree (the arcmin scale) and the flicker rate of "
     "the pattern tests.",
     "Open before a second sitting: the headset's pixels per degree (the arcmin scale)."),
    ("\\includegraphics[width=\\columnwidth]{exam_gate.pdf}",
     "\\includegraphics[width=0.92\\columnwidth]{exam_gate.pdf}"),
    ("\\includegraphics[width=\\columnwidth]{exam_photos.pdf}",
     "\\includegraphics[width=0.95\\columnwidth]{exam_photos.pdf}"),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
if "\\label{sec:rig}" not in s:
    s = s.replace("\\section{The rig and the exam}", "\\section{The rig and the exam}\\label{sec:rig}")
open(tex, "w", encoding="utf-8").write(s)
print("trimmed")
