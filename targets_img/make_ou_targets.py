"""Acuity read-out targets, one scan position per pixel: a tumbling E and a
ring that is either closed (O) or has its whole bottom missing (the "U" of
an O/U test; three cells of difference), side by side on a 12 x 7 grid, so
one scan of 84 positions reads both tests at once. The read-out
(analysis/vision_readout.py) decides which way the E points (4 choices) and
whether the ring is O or U (2 choices) from the EEG picture alone.

  ou_pair_E<right|down|left|up>_<o|u>.png   the 8 combinations, 12 x 7
  ou_big_o_12.png, ou_big_u_12.png           one ring at double size, 12 x 12
  vision_today.png + vision_today_seal.json  one combination drawn at random
                                             (--draw), the seal says which

    python targets_img/make_ou_targets.py          # all targets + a sheet
    python targets_img/make_ou_targets.py --draw   # tomorrow's sealed pair

Run (about 15 min with the certainty scan capped at two visits per cell):
    python xr_session.py --source osc --mode visual --choice bw \\
        --target targets_img/vision_today.png --grid-w 12 --grid-h 7 \\
        --freq 12 --calib-blocks 6 --calib-on 8 --calib-off 8 --calib-style bw \\
        --calib-size 1 --spc 8 --certainty-max-passes 2 --rest-every 42
then  python analysis/vision_readout.py runs/<session> --seal targets_img/vision_today_seal.json
"""
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_vision_targets import E_RIGHT, C_RIGHT, rotations, tile  # noqa: E402

E_NAMES = ["right", "down", "left", "up"]          # the open side of the E
O_RING = np.array([[0, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 1, 1, 1, 0]], float)
U_RING = O_RING.copy()
U_RING[4, :] = 0                                   # the whole bottom row missing
RINGS = {"o": O_RING, "u": U_RING}
# u = the O with its bottom missing: three cells of difference, sturdier
# than a Landolt gap of one cell (rotations(C_RIGHT)[1]) when a quarter of
# the cells may come out wrong; the read-out still reports the Landolt gaps

# the pair layout on 12 x 7: one dark row above and below, the E in columns
# 1-5, a dark column, the ring in columns 7-11
E_WIN = (slice(1, 6), slice(1, 6))
RING_WIN = (slice(1, 6), slice(7, 12))


def pair(e_k, ring):
    g = np.zeros((7, 12))
    g[E_WIN] = rotations(E_RIGHT)[e_k]
    g[RING_WIN] = RINGS[ring]
    return g


def big(glyph):
    g = np.zeros((12, 12))
    g[1:11, 1:11] = np.kron(glyph, np.ones((2, 2)))
    return g


def save(a, name):
    Image.fromarray((a * 255).astype(np.uint8), mode="L").save(os.path.join(HERE, name + ".png"))


def main(draw=False):
    if draw:
        e_k, ring = random.randrange(4), random.choice(["o", "u"])
        a = pair(e_k, ring)
        save(a, "vision_today")
        seal = dict(e=E_NAMES[e_k], ring=ring, file="vision_today.png",
                    drawn=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(seal, open(os.path.join(HERE, "vision_today_seal.json"), "w"), indent=1)
        print("drawn: targets_img/vision_today.png (12 x 7); the answer is sealed in "
              "targets_img/vision_today_seal.json, do not open it before the read-out")
        return
    tiles = []
    for e_k in range(4):
        for ring in ("o", "u"):
            a = pair(e_k, ring)
            name = f"ou_pair_E{E_NAMES[e_k]}_{ring}"
            save(a, name)
            tiles.append(tile(a, f"E {E_NAMES[e_k]}, ring {ring.upper()}", size=240))
            print(f"{name}: 12x7, {int(a.sum())} lit of 84")
    for ring in ("o", "u"):
        a = big(RINGS[ring])
        save(a, f"ou_big_{ring}_12")
        tiles.append(tile(a, f"big ring {ring.upper()} (12 x 12)", size=240))
        print(f"ou_big_{ring}_12: 12x12, {int(a.sum())} lit of 144")
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 250 + 10, rows * 272 + 10), (90, 90, 90))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % cols) * 250, 10 + (i // cols) * 272))
    sheet.save(os.path.join(HERE, "ou_preview.png"))
    print("sheet: targets_img/ou_preview.png")


if __name__ == "__main__":
    main(draw="--draw" in sys.argv)
