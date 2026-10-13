Ear as a Hydrophone: IEEE ICCE 2027 draft (about 70% complete), companion to the Eye as a Camera in XR paper
Authors as drafted: Alexander Vicol and Steve Mann (U of T / MannLab).

Upload this whole folder as a zip to Overleaf (New Project > Upload Project).
Main document: main.tex. Compiler: pdfLaTeX. Bibliography: BibTeX (refs.bib, IEEEtran.bst).
IEEEtran.cls and IEEEtran.bst are included. The draft runs 4 pages with seven figures and one table.

Open items (red [TODO] in the text):
  - who played the hydraulophone, what was played, how far the ear was in the water, and whether the lead
    in the water in the photograph is a hydrophone feeding the amplifier (Methods)
  - acknowledgements
  - a recording with the optical array switched off (to confirm the 64 Hz line is the headband's optical clock)

Data (one listener, Prof. Mann, 2026-10-04 16:37 to 16:44, 6.3 min): Downloads\SteveEarAsAHydraulophone.csv
  MuseLog CSV, 82 columns: 4 EEG channels at 256 Hz plus the 16 optical channels (730 / 850 nm, red, ambient;
  inner and outer; left and right), which ARE populated in this file (unlike the 2026-10-09 harbour files).
  Analysis (eyecam repo): analysis/hydrophone_field.py <csv> <out_dir>   EEG spectra, spectrogram ridge search,
                                                                         alpha / level time course, ASSR budget
                          analysis/hydrophone_optics.py <csv> <out_dir>  pulse, breathing, HbO / HbR by Beer-Lambert
  Both write their numbers as json next to the figures.

Figures (figures/): photo_dock.jpg (the screenshot), fig_spectra, fig_spectrogram, fig_timecourse, fig_optics, fig_budget.
