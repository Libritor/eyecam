"""Second trim of docs/paper/exam2027 (main.tex + refs.bib) to hold two pages:
smaller page-2 figures, shorter sitting and gate sentences, 'and others' for
the long iCalm author list, no URL note on the Mann reference.

    python Tools/exam_paper_trim2.py docs/paper/exam2027
"""
import os
import re
import sys

d = sys.argv[1]
tex = os.path.join(d, "main.tex")
s = open(tex, encoding="utf-8").read()
reps = [
    ("\\includegraphics[width=0.92\\columnwidth]{exam_gate.pdf}",
     "\\includegraphics[width=0.8\\columnwidth]{exam_gate.pdf}"),
    ("\\includegraphics[width=0.95\\columnwidth]{exam_photos.pdf}",
     "\\includegraphics[width=0.86\\columnwidth]{exam_photos.pdf}"),
    ("and the session's signal check had passed them because it was written for electrodes at the amplifier "
     "rails; Oz held for the right eye and drifted after: the quiet failure, not the wrong number, that a "
     "consumer wearable brings to the field \\cite{fletcher2010wearable}.",
     "and the session's signal check, written for electrodes at the amplifier rails, had passed them: the "
     "quiet failure, not the wrong number, a consumer wearable brings to the field \\cite{fletcher2010wearable}."),
    ("The exam now measures every channel before each eye and, when one is above 60~$\\mu$V, tells the subject "
     "on the headset which electrode to press in and waits; the read-out names any channel set aside.",
     "The exam now measures every channel before each eye, tells the subject on the headset which electrode to "
     "press in when one is above 60~$\\mu$V, and names any channel set aside."),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
s = re.sub(r", response phase within \$\d+\$ to \$\d+\^\\circ\$ in every responding quadrant\.", ".", s)
open(tex, "w", encoding="utf-8").write(s)

bib = os.path.join(d, "refs.bib")
b = open(bib, encoding="utf-8").read()
m = re.search(r"@article\{fletcher2010icalm,.*?\n\}", b, re.S)
if m:
    entry = m.group(0)
    new = re.sub(r"author\s*=\s*\{[^}]*\}", "author = {Richard R. Fletcher and others}", entry)
    b = b.replace(entry, new)
m = re.search(r"@inproceedings\{mann2019eye,.*?\n\}", b, re.S)
if m:
    entry = m.group(0)
    new = re.sub(r"\s*note\s*=\s*\{[^}]*\},?", "", entry)
    b = b.replace(entry, new)
open(bib, "w", encoding="utf-8").write(b)
print("trimmed again")
