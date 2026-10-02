"""2026-10-01: pixel-font targets ('pix:NO', 'pixgrad:NO') that are legible
at scan resolution, and no Eq. 1 vertical blend for discrete positions.
Run once from the repo root."""
import io


def patch(p, pairs):
    x = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert x.count(a) == 1, (p, a[:70], x.count(a))
        x = x.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="\n").write(x)


patch("targets.py", [
    ('''def load_target(spec, grid_w=None, grid_h=None, binarize=None):''',
     '''# 4x5 pixel glyphs: a scalable font downsampled to a handful of scan
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


def load_target(spec, grid_w=None, grid_h=None, binarize=None):'''),
    ('''    if spec.startswith("grad:"):''',
     '''    if spec.startswith("pix:") or spec.startswith("pixgrad:"):
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
    if spec.startswith("grad:"):'''),
])

patch("xr_session.py", [
    ('''    ap.add_argument("--passes", type=int, default=1,''',
     '''    ap.add_argument("--vblend", action="store_true",
                    help="apply the paper's Eq. 1 vertical blend. It is for "
                         "overlapping scan lines; on discrete positions it "
                         "only smears rows together, so it is off by default")
    ap.add_argument("--passes", type=int, default=1,'''),
    ('''    args = ap.parse_args()
    if args.preset:''',
     '''    args = ap.parse_args()
    if not args.vblend:
        config.VERTICAL_KERNEL = [1.0]
    if args.target.startswith("pix"):
        import targets as _t
        _g = _t.load_target(args.target)
        args.grid_h = max(args.grid_h, _g.shape[0]) if args.grid_h != 8 else _g.shape[0]
        args.grid_w = max(args.grid_w, _g.shape[1]) if args.grid_w != 12 else _g.shape[1]
    if args.preset:'''),
])
print("patched")
