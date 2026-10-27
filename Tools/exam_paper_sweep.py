"""Add the flicker sweep (temporal tuning and apparent latency) to the two-page
paper docs/paper/exam2027: the abstract, the method and the gate paragraph,
with the gate numbers read from a passed gate session's exam_readout.json,
and the Regan (1966) reference in refs.bib.

    python Tools/exam_paper_sweep.py runs/exam_gate6 docs/paper/exam2027
"""
import json
import os
import sys

session, d = sys.argv[1], sys.argv[2]
ro = json.load(open(os.path.join(session, "exam_readout.json"), encoding="utf-8"))
R, L = ro["eyes"]["right"], ro["eyes"]["left"]
dead = R["field"]["not_responding"]
diff = ro["compare"]["latency_diff_ms"]
half = L["temporal"]["half_rate"]
n_rates = R["temporal"]["n_rates"]


def contrast_phrase(E):
    """The contrast threshold, or, when the faintest step shown still responds,
    where the line crosses and the fact that it is below the range."""
    c = E["contrast"]
    if c.get("beyond_range"):
        return (f"a contrast line crossing at {c['threshold']:.3f}, below the faintest step shown "
                f"({min(c['contrasts']):.2g})")
    return f"contrast {c['threshold']:.3f}"


tex = os.path.join(d, "main.tex")
s = open(tex, encoding="utf-8").read()
reps = [
    # abstract
    ("Acuity, contrast, field and colour come from the steady-state visual evoked potential, an exact test "
     "per step and the sweep-VEP threshold rule.",
     "Acuity, contrast, field, colour and the apparent latency come from the steady-state visual evoked "
     "potential: an exact test per step, the sweep-VEP threshold rule and the slope of the response phase "
     "over flicker rate."),
    ("passes all sixteen checks of a full-stack gate.",
     "passes all twenty-three checks of a full-stack gate."),
    # method
    ("which must respond and whose channel and amplitude are the reference for the eye's tests. Then, two "
     "pairs per step:",
     "which must respond and whose channel and amplitude are the reference for the eye's tests. The flash "
     "then runs at six rates from 6 to 36~Hz: the amplitude against rate is the eye's temporal tuning, and "
     "the slope of the response phase against rate its apparent latency \\cite{regan1966}, whose difference "
     "between the eyes is what the clinical pattern VEP is read for. Then, two pairs per step:"),
    ("Six and a half minutes per eye.", "Eight minutes per eye."),
    # gate
    ("through a logistic visibility in bar width and contrast with a cut-off per eye, dead quadrants and "
     "colour gains.",
     "through a logistic visibility in bar width and contrast with a cut-off per eye, dead quadrants, colour "
     "gains, and a latency and a flicker cut per eye."),
    ("The gate dials in a right eye at 8~arcmin and contrast 0.08 with one dead quadrant, a left eye twice as "
     "coarse and whole, gains 0.7/1/0.4 for red/green/blue, and asks sixteen things: every stage delivered, "
     "the flash responding in both eyes on Oz and every test read on it, each threshold inside the band where "
     "the phantom's visibility runs from 0.02 to 0.6, the left eye at least $1.3\\times$ coarser, exactly the "
     "dead quadrant named, green~$\\geq$~red~$>$~blue. All sixteen pass (Fig.~\\ref{fig:gate}): right eye "
     "3.8~arcmin and 0.027 with quadrant 3 silent, left eye 7.9~arcmin and 0.058.",
     "The gate dials in a right eye at 8~arcmin and contrast 0.08 with one dead quadrant, 120~ms late and "
     "holding to 40~Hz, a left eye twice as coarse and whole, 150~ms late and half gone by 24~Hz, gains "
     "0.7/1/0.4 for red/green/blue, and asks twenty-three things: every stage delivered, the flash responding "
     "in both eyes on Oz and every test read on it, each threshold inside the band where the phantom's "
     "visibility runs from 0.02 to 0.6, the left eye at least $1.3\\times$ coarser, exactly the dead quadrant "
     "named, green~$\\geq$~red~$>$~blue, every rate responding, the left eye's half rate within 15\\% of "
     "24~Hz, each latency within 6~ms of the phantom's and the left 30~ms later. All twenty-three pass "
     f"(Fig.~\\ref{{fig:gate}}): right eye {R['acuity']['threshold']:.1f}~arcmin, quadrant "
     f"{', '.join(str(k + 1) for k in dead)} silent, {contrast_phrase(R)}; left eye "
     f"{L['acuity']['threshold']:.1f}~arcmin and {contrast_phrase(L)}, half rate "
     f"{half:.1f}~Hz, its response {diff:.0f}~ms later than the right's."),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
open(tex, "w", encoding="utf-8").write(s)

bib = os.path.join(d, "refs.bib")
b = open(bib, encoding="utf-8").read()
if "regan1966" not in b:
    b = b.rstrip("\n") + """

@article{regan1966,
  author = {David Regan},
  title = {Some characteristics of average steady-state and transient responses evoked by modulated light},
  journal = {Electroencephalography and Clinical Neurophysiology},
  volume = {20},
  number = {3},
  pages = {238--248},
  year = {1966}
}
"""
    open(bib, "w", encoding="utf-8").write(b)
print("main.tex + refs.bib updated; gate numbers:",
      f"right {R['acuity']['threshold']:.2f}' / {R['contrast']['threshold']:.3f}, dead {dead}; "
      f"left {L['acuity']['threshold']:.2f}' / {L['contrast']['threshold']:.3f}, half {half:.1f} Hz; "
      f"latency diff {diff:.1f} ms; {n_rates} rates")
