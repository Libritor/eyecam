State of Strobe: IEEE ICCE 2027 draft (about 75% complete), companion to the Eye as a Camera in XR paper
Authors as drafted: Alexander Vicol and Steve Mann (U of T / MannLab).

Upload this whole folder as a zip to Overleaf (New Project > Upload Project).
Main document: main.tex. Compiler: pdfLaTeX. Bibliography: BibTeX (refs.bib, IEEEtran.bst).
IEEEtran.cls and IEEEtran.bst are included. The draft runs 5 pages with seven figures and one table.

Marks in the text:
  red  [TODO: ...]  open items (the subject's report after the two daytime sittings; acknowledgements)
  (no notes for other authors in this version)

Data behind the paper (one subject):
  21 sessions, 2026-09-14 to 2026-10-12, for the pooled drowsiness markers (analysis/state_of_flash.py)
  five sittings in depth: vr_planes8 (10-07 night, reference), vr_planes9 (10-10 17:57), vr_planes10 (10-10 19:50),
  vr_vision1 (10-11 12:05, grey acuity scan), vr_planes11 (10-12 10:40)
  blinks: analysis/flash_eog.py tag=_all ...   frontal: analysis/flash_frontal_theta.py tag=_all ...
  SSVEP fatigue: analysis/flash_fatigue.py runs/vr_planes9 runs/vr_planes10 runs/vr_planes11
  the five-sittings figure: analysis/strobe_summary.py
  all outputs under runs/state_of_flash/ in the eyecam repository (runs/ is not in git)

Figures (figures/):
  effects.png            drowsiness markers over 21 sessions
  time_courses_two.png   theta/alpha through two long sessions (crop of time_courses.png)
  blink_rate.png         blink rate per minute through the five sittings (crop of flash_eog_all.png)
  blink_release.png      blinks around the moment a cell goes dark (crop)
  lit_dark_rest.png      blink rate on lit cells, black cells, rest breaks (crop)
  forehead.png           the frontalis family, two evening sittings and the reference night
  ssvep_fatigue.png      every lit cell's SSVEP against minutes on task, three colour sittings
  summary.png            the five sittings at a glance
  confounds.png, blink_duration.png   not placed
