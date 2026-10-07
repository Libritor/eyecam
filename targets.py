"""Target images for stimulus and simulation.

A target is a float array in [0, 1], shape (GRID_H, GRID_W): the luminance the
flickering cursor takes on at each scan position. Text targets echo the paper's
"NO" / "NO CAMERAS" demonstrations.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


def text_target(text="NO", grid_w=None, grid_h=None):
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    # Render large, then downsample to the scan grid.
    W, H = grid_w * 16, grid_h * 16
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    size = H
    font = None
    while size > 8:
        try:
            font = ImageFont.truetype("arialbd.ttf", size)
        except OSError:
            font = ImageFont.load_default()
            break
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= W * 0.9 and bbox[3] - bbox[1] <= H * 0.9:
            break
        size = int(size * 0.9)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (H - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    small = img.resize((grid_w, grid_h), Image.LANCZOS)
    return np.asarray(small, dtype=float) / 255.0


def image_target(path, grid_w=None, grid_h=None):
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    img = Image.open(path).convert("L").resize((grid_w, grid_h), Image.LANCZOS)
    return np.asarray(img, dtype=float) / 255.0


def color_target(spec, grid_w=None, grid_h=None):
    """RGB target (grid_h, grid_w, 3) in [0,1].
    'flag'  = three vertical bands R | G | B
    'quad'  = quadrants R, G, B, white
    'text:X'= glyph X in yellow (R+G) on black  (tests two-plane cells)
    'ring'  = red disc, green ring, blue corners
    or an image path (RGB)."""
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    g = np.zeros((grid_h, grid_w, 3))
    if spec == "flag":
        w3 = grid_w / 3.0
        for xg in range(grid_w):
            g[:, xg, min(2, int(xg / w3))] = 1.0
    elif spec == "quad":
        h2, w2 = grid_h // 2, grid_w // 2
        g[:h2, :w2, 0] = 1.0
        g[:h2, w2:, 1] = 1.0
        g[h2:, :w2, 2] = 1.0
        g[h2:, w2:, :] = 1.0
    elif spec.startswith("text:"):
        gl = text_target(spec[5:], grid_w, grid_h)
        g[:, :, 0] = gl
        g[:, :, 1] = gl
    elif spec == "ring":
        yy, xx = np.mgrid[0:grid_h, 0:grid_w]
        cy, cx = (grid_h - 1) / 2, (grid_w - 1) / 2
        rr = np.sqrt(((yy - cy) / grid_h) ** 2 + ((xx - cx) / grid_w) ** 2)
        g[rr < 0.18, 0] = 1.0
        g[(rr >= 0.18) & (rr < 0.36), 1] = 1.0
        g[rr >= 0.36, 2] = 1.0
    else:
        img = Image.open(spec).convert("RGB").resize((grid_w, grid_h),
                                                     Image.LANCZOS)
        g = np.asarray(img, dtype=float) / 255.0
    return g


# 4x5 pixel glyphs: a scalable font downsampled to a handful of scan
# positions is an unreadable blob; these stay letters at one cell per pixel.
PIX = {
    "N": ["X..X", "XX.X", "X.XX", "X..X", "X..X"],
    "O": [".XX.", "X..X", "X..X", "X..X", ".XX."],
    "H": ["X..X", "X..X", "XXXX", "X..X", "X..X"],
    "I": ["XXX", ".X.", ".X.", ".X.", "XXX"],
    "E": ["XXXX", "X...", "XXX.", "X...", "XXXX"],
    "Y": ["X..X", "X..X", ".XX.", ".X..", ".X.."],
    "S": [".XXX", "X...", ".XX.", "...X", "XXX."],
    "T": ["XXX", ".X.", ".X.", ".X.", ".X."],
    "L": ["X...", "X...", "X...", "X...", "XXXX"],
    "A": [".XX.", "X..X", "XXXX", "X..X", "X..X"],
    "+": ["...", ".X.", "XXX", ".X.", "..."],
}


def pix_target(text, levels=False):
    """Letters from PIX, one blank column between them, no margin. The scan
    grid IS this bitmap (use pix_size() for --grid-w/--grid-h).
    levels=True: first letter white, last letter half grey."""
    chars = [c for c in text.upper() if c in PIX]
    if not chars:
        raise ValueError(f"no pixel glyphs for {text!r} (have {''.join(PIX)})")
    cols = []
    for k, c in enumerate(chars):
        lvl = 1.0 - 0.5 * k / max(len(chars) - 1, 1) if levels else 1.0
        g = np.array([[lvl if ch == "X" else 0.0 for ch in row] for row in PIX[c]])
        cols.append(g)
        if k < len(chars) - 1:
            cols.append(np.zeros((5, 1)))
    return np.hstack(cols)


def pix_letters(text):
    """Same layout as pix_target: the index of the letter each lit cell
    belongs to (0, 1, ...), -1 where unlit."""
    chars = [c for c in text.upper() if c in PIX]
    cols = []
    for k, c in enumerate(chars):
        cols.append(np.array([[k if ch == "X" else -1 for ch in row] for row in PIX[c]]))
        if k < len(chars) - 1:
            cols.append(np.full((5, 1), -1))
    return np.hstack(cols)


def load_target(spec, grid_w=None, grid_h=None, binarize=None):
    """spec: 'text:NO CAMERAS' or a path to an image file.
    binarize: threshold at 0.5 so every cell is fully on or off (default: yes
    for text targets, whose grey edge cells are just anti-aliasing and would
    flicker at reduced contrast; no for images)."""
    if spec.startswith("pix:") or spec.startswith("pixgrad:"):
        g = pix_target(spec.split(":", 1)[1], levels=spec.startswith("pixgrad:"))
        if (grid_h, grid_w) not in ((None, None), g.shape):
            # centre the bitmap on a larger board (dark margin around it)
            out = np.zeros((grid_h, grid_w))
            y0, x0 = (grid_h - g.shape[0]) // 2, (grid_w - g.shape[1]) // 2
            if y0 < 0 or x0 < 0:
                raise ValueError(f"{spec} needs at least a "
                                 f"{g.shape[1]}x{g.shape[0]} grid")
            out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
            return out
        return g
    if spec.startswith("grad:"):
        # letters at descending grey levels (first white, last half grey):
        # 'grad:NO' = white N, half-grey O
        txt = spec[5:]
        g = (text_target(txt, grid_w, grid_h) > 0.5).astype(float)
        n = max(len(txt.replace(" ", "")), 1)
        edges = np.linspace(0, g.shape[1], n + 1).round().astype(int)
        for k in range(n):
            g[:, edges[k]:edges[k + 1]] *= 1.0 - 0.5 * k / max(n - 1, 1)
        return g
    if spec.startswith("text:"):
        g = text_target(spec[5:], grid_w, grid_h)
        return (g > 0.5).astype(float) if (binarize is None or binarize) else g
    g = image_target(spec, grid_w, grid_h)
    return (g > 0.5).astype(float) if binarize else g


BWB_NAMES = {"black": 0, "white": 1, "blue": 2}


def bwb_target(spec, grid_w=6, grid_h=4, fg=None, bg=None, colours=None, border=None):
    """Class grid (grid_h, grid_w) for the colour-palette scan: 0 = black,
    k = colours[k - 1] (default colours: white, blue, i.e. the original
    black / white / blue scan).
    'bands'   = one vertical band per class: black | colour 1 | colour 2 ...
    'checker' = the classes on diagonals (every class in every row)
    'text:X'  = glyph X in colour 1 on colour 2, a black row above and below
    'pix:X'   = pixel-font X in colour 1 on colour 2, black rows above and below
                (grid set by the font: 7 rows)
    or an image path, each cell snapped to the nearest palette colour.
    fg / bg (palette names, or 'black') recolour a text:/pix: glyph and its
    background; the glyph then gets a one-cell margin all round instead of the
    black rows, in `border` if given, else the background (e.g. fg='blue',
    bg='white': a blue letter on white). For pix: targets fg may name one
    colour per letter: fg='red,green', bg='blue', border='white' is a red N
    and a green O on blue inside a white frame, 11 x 7 cells."""
    import bwb
    names, pal = bwb.classes_of(colours)
    K = len(names)
    idx = {n: i for i, n in enumerate(names)}
    if spec == "bands":
        g = np.zeros((grid_h, grid_w), int)
        for x in range(grid_w):
            g[:, x] = min(K - 1, int(K * x / grid_w))
        return g
    if spec == "checker":
        yy, xx = np.mgrid[0:grid_h, 0:grid_w]
        return (xx + yy) % K
    if spec.startswith("text:") or spec.startswith("pix:"):
        # glyph rows plus one black row above and below (so black appears too)
        fgs = [c.strip() for c in fg.split(",")] if fg else []
        custom = bool(fgs) or bg is not None or border is not None
        for c in fgs + [bg, border]:
            if c is not None and c not in idx:
                raise ValueError(f"{c!r} is not in the palette {names}")
        f_cls = [idx[c] for c in fgs] or [1]
        b_cls = idx[bg] if bg else min(2, K - 1)
        rows = grid_h - 2
        cols = grid_w - 2 if custom else grid_w
        if spec.startswith("text:"):
            gl = text_target(spec[5:], cols, max(1, rows))
            g = np.where(np.asarray(gl) > 0.5, f_cls[0], b_cls)
        else:
            letter = pix_letters(spec[4:])
            g = np.where(letter >= 0, np.take(f_cls, letter % len(f_cls)), b_cls)
        if custom:
            return np.pad(g, 1, constant_values=b_cls if border is None else idx[border])
        return np.pad(g, ((1, 1), (0, 0)), constant_values=0)
    img = np.asarray(Image.open(spec).convert("RGB").resize((grid_w, grid_h), Image.BOX), float) / 255
    return np.argmin(((img[:, :, None, :] - pal) ** 2).sum(-1), axis=2)
