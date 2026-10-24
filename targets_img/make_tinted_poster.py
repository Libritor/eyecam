"""A grey portrait tinted into the eight corner colours, the way the officer
illustration is drawn: a colour poster the planes scan can read, made from a
photograph that has no colour of its own. The brightness of each cell is cut
into four bands and each band is given a colour (default: darkest black for
the suit and the eyes, the face band yellow, the two brightest white for hair,
collar and wall), with an optional red band for a tie or lips. This is tinting,
not colour recovery: the paper must say the colours are assigned.

    python targets_img/make_tinted_poster.py targets_img/photos_more/bose.png bose --n 12
    python targets_img/make_tinted_poster.py targets_img/photos_more/bose.png bose --n 16 --colours black,yellow,white,white
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_photo_gallery as g  # noqa: E402

CORNER = dict(black=(0, 0, 0), red=(255, 0, 0), green=(0, 255, 0), yellow=(255, 255, 0),
              blue=(0, 0, 255), magenta=(255, 0, 255), cyan=(0, 255, 255), white=(255, 255, 255))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("name", help="crop name in make_photo_gallery.CROPS")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--colours", default="black,yellow,white,white", help="one corner colour per band, darkest first")
    ap.add_argument("--mode", default="portrait", choices=["bands", "portrait"])
    ap.add_argument("--head", default="0.50,0.50,0.28,0.38,0.27,0.80",
                    help="portrait mode: head ellipse cx,cy,rx,ry, hair line y, chin y (crop coordinates)")
    ap.add_argument("--wall", default="white")
    ap.add_argument("--hair", default="white")
    ap.add_argument("--face", default="yellow")
    ap.add_argument("--suit", default="black")
    ap.add_argument("--feature-bands", type=int, default=1, help="darkest bands inside the face that become black features")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cols = a.colours.split(",")
    assert len(cols) == a.levels, "one colour per band"
    q, head = g.target_from_photo(a.photo, a.name, a.n, a.levels)
    band = np.round(q * (a.levels - 1)).astype(int)
    img = np.zeros((a.n, a.n, 3), np.uint8)
    if a.mode == "bands":
        for k, c in enumerate(cols):
            img[band == k] = CORNER[c]
    else:
        # portrait regions in crop coordinates (0..1): wall, head ellipse (face
        # below the hair line, hair above), suit below the chin with a collar;
        # the darkest band inside the face becomes the features
        cx, cy, rx, ry, hair_y, chin_y = (float(v) for v in a.head.split(","))
        yy, xx = (np.arange(a.n) + 0.5) / a.n, (np.arange(a.n) + 0.5) / a.n
        X, Y = np.meshgrid(xx, yy)
        head_m = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        suit = (Y > chin_y) & ~head_m | ((Y > chin_y - 0.1) & (np.abs(X - 0.5) > 0.32))
        collar = (Y > chin_y - 0.02) & (Y < chin_y + 0.12) & (np.abs(X - 0.5) < 0.07)
        img[:] = CORNER[a.wall]
        img[suit] = CORNER[a.suit]
        img[collar] = CORNER["white"]
        img[head_m & (Y < hair_y)] = CORNER[a.hair]
        face = head_m & (Y >= hair_y)
        img[face] = CORNER[a.face]
        img[face & (band < a.feature_bands)] = CORNER["black"]
    stem = a.out or os.path.join(HERE, f"{a.name}_tinted_{a.n}x{a.n}.png")
    Image.fromarray(img, "RGB").save(stem)
    S = 360
    prev = Image.new("RGB", (2 * S + 30, S + 20), (70, 70, 70))
    prev.paste(head.convert("RGB").resize((S, S)), (10, 10))
    big = Image.fromarray(img, "RGB").resize((S, S), Image.NEAREST)
    d = ImageDraw.Draw(big)
    for i in range(1, a.n):
        p = i * S // a.n
        d.line([(p, 0), (p, S)], fill=(110, 110, 110))
        d.line([(0, p), (S, p)], fill=(110, 110, 110))
    prev.paste(big, (S + 20, 10))
    prev.save(os.path.splitext(stem)[0] + "_preview.png")
    names = {v: k for k, v in CORNER.items()}
    flat = [tuple(int(v) for v in px) for px in img.reshape(-1, 3)]
    used = {names[c]: flat.count(c) for c in sorted(set(flat))}
    print(f"{stem}: {a.n} x {a.n}, {a.mode} -> {', '.join(f'{c} {n}' for c, n in used.items())}; "
          f"about {a.n * a.n * 18 / 60:.0f} min through the planes; preview {os.path.splitext(stem)[0] + '_preview.png'}")


if __name__ == "__main__":
    main()
