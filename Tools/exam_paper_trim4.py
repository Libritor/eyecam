"""Fourth trim of docs/paper/exam2027 (one reference spilled onto a third
page): 'and others' for the JITAI author list, slightly smaller page-2
figures, one shorter sentence in the sitting.

    python Tools/exam_paper_trim4.py docs/paper/exam2027
"""
import os
import re
import sys

d = sys.argv[1]
tex = os.path.join(d, "main.tex")
s = open(tex, encoding="utf-8").read()
reps = [
    ("\\includegraphics[width=0.7\\columnwidth]{exam_photos.pdf}",
     "\\includegraphics[width=0.66\\columnwidth]{exam_photos.pdf}"),
    ("\\includegraphics[width=0.64\\columnwidth]{exam_gate.pdf}",
     "\\includegraphics[width=0.6\\columnwidth]{exam_gate.pdf}"),
    ("The exam now measures every channel before each eye, tells the subject on the headset which electrode "
     "to press in when one is above 60~$\\mu$V, and names any channel set aside.",
     "The exam now measures every channel before each eye, tells the subject which electrode to press in "
     "when one is above 60~$\\mu$V, and names any channel set aside."),
]
for old, new in reps:
    if old not in s:
        raise SystemExit("NOT FOUND: " + old[:70])
    s = s.replace(old, new)
open(tex, "w", encoding="utf-8").write(s)

bib = os.path.join(d, "refs.bib")
b = open(bib, encoding="utf-8").read()
m = re.search(r"@article\{nahum2018jitai,.*?\n\}", b, re.S)
if m:
    entry = m.group(0)
    b = b.replace(entry, re.sub(r"author\s*=\s*\{[^}]*\}", "author = {Inbal Nahum-Shani and others}", entry))
open(bib, "w", encoding="utf-8").write(b)
print("trimmed a fourth time")
