"""Paper figures as vector PDFs with real text, from the analysis scripts that
draw them as PNGs.

  run   runs any figure-drawing script under a hook: every PNG it saves with
        matplotlib is also saved as a PDF beside it, same layout, TrueType
        text embedded (pdf.fonttype 42), so the text stays selectable.
  cut   cuts panels out of such a PDF by page fractions (x0,y0,x1,y1 from the
        top-left, as a PNG crop would be) and composes them left to right on
        a new page, keeping the vectors (PyMuPDF show_pdf_page with a clip).
  check prints the page size and the embedded text of each PDF given.

    python analysis/paper_pdf_figures.py run analysis/flash_fatigue.py runs/vr_planes9 runs/vr_planes10
    python analysis/paper_pdf_figures.py cut runs/state_of_flash/fig_effects.pdf out.pdf 0.01,0.09,0.26,0.41 0.49,0.09,0.74,0.41
    python analysis/paper_pdf_figures.py check docs/paper/strobe2027/figures/*.pdf
"""
import os
import runpy
import sys


def hook():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    orig = Figure.savefig

    def savefig(self, fname, *a, **kw):
        out = orig(self, fname, *a, **kw)
        if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png"):
            kw2 = {k: v for k, v in kw.items() if k != "dpi"}
            orig(self, str(fname)[:-4] + ".pdf", *a, **kw2)
        return out
    Figure.savefig = savefig


def run(script, args):
    hook()
    sys.argv = [script] + list(args)
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


def cut(src, out, rects, gap_pt=12.0):
    import fitz
    doc = fitz.open(src)
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    clips = [fitz.Rect(x0 * W, y0 * H, x1 * W, y1 * H) for (x0, y0, x1, y1) in rects]
    width = sum(c.width for c in clips) + gap_pt * (len(clips) - 1)
    height = max(c.height for c in clips)
    new = fitz.open()
    pg = new.new_page(width=width, height=height)
    x = 0.0
    for c in clips:
        target = fitz.Rect(x, 0, x + c.width, c.height)
        pg.show_pdf_page(target, doc, 0, clip=c)
        x += c.width + gap_pt
    new.save(out, garbage=3, deflate=True)
    new.close(); doc.close()
    return out


def check(paths):
    import fitz
    for p in paths:
        d = fitz.open(p); pg = d[0]
        txt = pg.get_text().strip().replace("\n", " | ")
        fonts = sorted({f[3] for f in pg.get_fonts()})
        line = f"{os.path.basename(p):28s} {pg.rect.width:5.0f}x{pg.rect.height:4.0f} pt  text chars {len(txt):4d}  fonts {fonts[:2]}  e.g. {txt[:60]}"
        print(line.encode("ascii", "replace").decode())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3:])
    elif cmd == "cut":
        rects = [tuple(float(v) for v in r.split(",")) for r in sys.argv[4:]]
        print("wrote", cut(sys.argv[2], sys.argv[3], rects))
    elif cmd == "check":
        check(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
