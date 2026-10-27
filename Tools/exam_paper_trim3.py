"""Third trim of docs/paper/exam2027/main.tex to hold two pages after the
flicker sweep went in: smaller figures, tighter sweep, gate, companion and
outlook sentences.

    python Tools/exam_paper_trim3.py docs/paper/exam2027
"""
import os
import sys

d = sys.argv[1]
tex = os.path.join(d, "main.tex")
s = open(tex, encoding="utf-8").read()
reps = [
    ("\\includegraphics[width=\\columnwidth]{exam_diagram.pdf}",
     "\\includegraphics[width=0.9\\columnwidth]{exam_diagram.pdf}"),
    ("\\includegraphics[width=0.8\\columnwidth]{exam_photos.pdf}",
     "\\includegraphics[width=0.7\\columnwidth]{exam_photos.pdf}"),
    ("\\includegraphics[width=0.74\\columnwidth]{exam_gate.pdf}",
     "\\includegraphics[width=0.64\\columnwidth]{exam_gate.pdf}"),
    ("The flash then runs at six rates from 6 to 36~Hz: the amplitude against rate is the eye's temporal "
     "tuning, and the slope of the response phase against rate its apparent latency \\cite{regan1966}, whose "
     "difference between the eyes is what the clinical pattern VEP is read for.",
     "The flash then runs at six rates, 6 to 36~Hz: amplitude against rate is the eye's temporal tuning, the "
     "slope of phase against rate its apparent latency \\cite{regan1966}, whose difference between the eyes "
     "is the clinical pattern VEP's measure."),
    ("green~$\\geq$~red~$>$~blue, every rate responding, the left eye's half rate within 15\\% of 24~Hz, "
     "each latency within 6~ms of the phantom's and the left 30~ms later.",
     "green~$\\geq$~red~$>$~blue, every rate responding, the half rate and the latencies within 15\\% and "
     "6~ms of the dialled."),
    ("A headband records more than the visual response \\cite{picard1997affective}. In companion sittings on "
     "the same rig (in preparation) the forehead electrodes gave the blink rate, which stops while a patch "
     "flickers, bursts when it goes dark, and doubled between two sittings of one evening, an old marker of "
     "fatigue and eye strain \\cite{stern1994blink}. That is the hook for a just-in-time adaptive intervention "
     "\\cite{nahum2018jitai}: the device that has just measured an eye can say the eyes are tired. We call it "
     "an eye-strain estimate against the subject's own earlier minutes, not detection and not emotion.",
     "A headband records more than the visual response \\cite{picard1997affective}. In companion sittings on "
     "the same rig the forehead electrodes gave the blink rate, which stops while a patch flickers, bursts "
     "when it goes dark, and doubled between two sittings of one evening, an old marker of fatigue "
     "\\cite{stern1994blink}: the hook for a just-in-time adaptive intervention \\cite{nahum2018jitai}, an "
     "eye-strain estimate against the subject's own earlier minutes, not detection and not emotion."),
    ("Established: the apparatus, an exact test per step, a threshold rule from the sweep VEP, and a phantom "
     "gate that catches failures without a volunteer. Not established: any measurement on a person. Open "
     "before a second sitting: the headset's pixels per degree (the arcmin scale). Then a blur study with "
     "trial lenses against a letter chart, a normative cohort, and a worker-facing flow in which the device "
     "checks its own contact and the report is one picture.",
     "Established: the apparatus, an exact test per step, the sweep-VEP threshold rule, the latency from the "
     "phase slope, and a phantom gate that catches failures without a volunteer. Not established: any "
     "measurement on a person. Open: the headset's pixels per degree, then a blur study with trial lenses "
     "against a letter chart, a normative cohort, and a worker-facing flow with the report as one picture."),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
open(tex, "w", encoding="utf-8").write(s)
print("trimmed a third time")
