# The globe as designed, ready to print (29 October 2026)

The second object after the die: `mesh:globe` from `mesh3d.py`, the Earth on
a cube-sphere with 4 x 4 cells a face (96 cells, a 12 x 8 atlas; blue ocean
69 cells, green land 22, white ice 5; Africa faces front). These are the files
for the object as designed, made by
`python die_print.py --object mesh:globe --shown --slice --preview`, before any
sitting has scanned it; a scanned globe goes through the same step from its
`mesh_decoded.obj` (`quest_run.ps1 -Run globe -PrintPort COMn` does it the
moment the scan is decoded).

- `globe_shown.stl` — the globe as designed, 40 mm across, north pole up, watertight.
- `globe_shown_relief.stl` — the same globe with the colours made physical: ocean
  sunk 1.2 mm, land flush, ice caps and Greenland raised 1.2 mm, walls between cells.
- `globe_shown.gcode`, `globe_shown_relief.gcode` — sliced with CuraEngine 5.9 for a
  Creality Ender-3 Pro: 0.2 mm layers, 3 walls, 15 % grid infill, 200 / 60 C, raft,
  tree support (the sphere needs it below the equator; the supports land on the
  Antarctic cap). Plain 1.7 h and 8.8 m of filament; relief 212 layers, 1.8 h,
  9.3 m (about 28 g of PLA).
- `globe_shown_cells.png` — the 12 x 8 atlas as a painting guide (one face = 4 x 4 cells:
  front, right, back, left on the top row; top, bottom on the second).
- `globe_shown_relief_preview.png` — three views of the relief globe.
- `globe_shown_gcode_check.png` / `.pdf` / `.json` — `Tools/print_gcode_check.py` rebuilt the
  printed surface from the relief G-code and measured every cell along its normal:
  96 of 96 cells at their height within 0.35 mm, the 74 sunk or raised ones among them.
