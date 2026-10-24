# The die decoded from EEG (run vr_die1, 17 October 2026)

Printable files made by `die_print.py` from `runs/vr_die1/mesh_decoded.obj` (6 faces x 9 x 9 cells, 51 of 54 cells decoded right).

- `die_decoded_eeg.stl` — the decoded die as-is, 40 mm, watertight.
- `die_decoded_eeg_relief.stl` — the same die with the cells decoded black sunk 1.2 mm and the red pip raised 1.2 mm.
- `die_decoded_eeg.gcode`, `die_decoded_eeg_relief.gcode` — sliced with CuraEngine 5.9 for a Creality Ender-3 Pro: 0.2 mm layers, 3 walls, 15 % grid infill, 205 / 60 C, raft, tree support.
- `die_decoded_eeg_cells.png` — the 54-cell painting guide (9 x 6 atlas).
- `die_decoded_eeg_relief_preview.png` — render of the relief version.
