Eye as a Camera in XR: IEEE ICCE 2027 draft (about 75% complete)
Authors as drafted: Vicol, Herreraviyella, Adefris, Chaghouri, Sher (U of T / MannLab), Fletcher (MIT), Mann (U of T / MannLab); confirm affiliations, emails and order.

Upload this whole folder as a zip to Overleaf (New Project > Upload Project).
Main document: main.tex. Compiler: pdfLaTeX. Bibliography: BibTeX (refs.bib, IEEEtran.bst).
IEEEtran.cls and IEEEtran.bst are included so the project builds without relying on Overleaf's copies.

ICCE 2027 rules (icce.org/2027/submission-guidelines): IEEE conference template, US letter,
2 pages or up to 6 pages, no author biographies, EDAS (edas.info/N35272), PDF eXpress only for
the final version. Demos and posters are 2-page abstracts in IEEE Xplore (the 2027 page still
shows the 2022 deadlines; confirm on EDAS). The draft runs exactly 6 pages, the ICCE maximum, with all twelve figures; adding anything means cutting something.

Marks in the text:
  red  [TODO: ...]  open items for the first author (results pending, photos, emails, names)
  blue [RF: ...]    places for Rich Fletcher's additions (field context, JITAI, screening design)

Figures (figures/), every one a vector PDF with embedded TrueType text (selectable, searchable, no resolution limit),
drawn by analysis/icce_figures.py and analysis/planes_figures.py in the eyecam repository:
  response_curve.pdf       the subject's response by frequency (laptop sweep) and the spectrum at 10.4 Hz
  no_word.pdf              "NO" read from the EEG, per pass, both square sizes
  colour_planes9.pdf       eight colours, first colour sitting (24 of 32)
  planes9_per_plane.pdf    the same sitting plane by plane
  acuity_vision1.pdf       the sealed E and ring read from the EEG
  gallery_16.pdf           the photograph gallery (crops over 16x16 four-level targets) and the colour officer
  blink_release.pdf        blink burst when the cell goes dark, two sittings and the reference night
  ssvep_fatigue_left.pdf   the SSVEP of every lit cell through the two sittings of 2026-10-10
  die.pdf                  a die read from the eye: the cube shown and decoded, three views and the atlas (analysis/die_paper_figure.py; the renders are rasters, the labels text)
  setup.jpg                the set-up photograph: the Quest 3S over the Muse, the Oz cup held at the back of the head (a photograph, raster by nature)
  ear_mic.pdf              the ear as a microphone: the clip's loudness envelope and the brain's (r 0.36), and every play held out (analysis/ear_mic_paper_figure.py)
  The photographs inside gallery_16.pdf are raster by nature; every label, axis and number is text.
  blink_rate.png, colour_planes11.png, officer_illustration.png, officer_poster_12x14.png, ssvep_fatigue.png: spare rasters, not placed.

Sources: all figures come from the eyecam repository (github.com/Libritor/eyecam, branch colour),
docs/figures and docs/results; the numbers in the text are in README.md and NEXT_SESSION.md there.
