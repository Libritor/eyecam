"""The eye-exam paper's figures as separate images, one plot per file, sized
for the IEEE column (3.45 in) or half column (1.68 in), text embedded
(pdf.fonttype 42). Replaces the composite figures of exam_diagram.py,
exam_photo_figure.py, exam_paper_figure.py, exam_gate_new_figure.py and
exam_human_figure.py, reusing their drawing functions where the panel reads
well at the new size.

    python analysis/exam_paper_panels.py docs/paper/exam2027/figures [part ...]

Inputs: runs/exam_gate (the 39-check gate, 29 October 2026), runs/vr_exam6 (the fourth human sitting),
runs/vr_exam_left_2026-11-03b and runs/vr_exam_right_2026-11-03 (the sixth sitting's photometry), the rig
photo of the ICCE paper. Writes fig_*.pdf (and fig_rig.jpg) into the folder given.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exam_paper_figure import EYE_STYLE, colour, curve  # noqa: E402
from exam_gate_new_figure import csf, photometry  # noqa: E402

FULL, HALF = 3.45, 1.68
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.5})
RIGHT_H, LEFT_H = "#2a78d6", "#eb6834"        # the human figure's eye colours
ALPHA = 0.05
GAMMA = 2.2


def readout(session):
    with open(os.path.join(ROOT, session, "exam_readout.json")) as f:
        return json.load(f)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ── the rig and the data path ────────────────────────────────────────────────
def rig_photo(out):
    rig = Image.open(os.path.join(ROOT, "docs", "paper", "icce2027", "figures", "setup.jpg")).convert("RGB")
    w, h = rig.size
    rig = rig.crop((int(w * 0.08), int(h * 0.30), int(w * 0.92), int(h * 0.82)))
    rig.save(os.path.join(out, "fig_rig.jpg"), quality=92)
    print("wrote", os.path.join(out, "fig_rig.jpg"))


def box(ax, x, y, w, h, title, body, fc="#f4f6f8", ec="#1f4e79"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h * 0.80, title, ha="center", va="center", fontsize=7, weight="bold")
    ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center", fontsize=6.3, linespacing=1.25)


def arrow(ax, p, q, text="", off=(0, 0.03), both=False, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7,
                                 lw=0.8, color="#333", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha, va="bottom",
                fontsize=6, color="#333")


def pipeline(out):
    """The loop the exam runs: subject, headband, phone, PC, headset, subject."""
    fig = plt.figure(figsize=(FULL, 1.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, h = 0.27, 0.38
    xs = [0.01, 0.365, 0.72]
    y1, y2 = 0.58, 0.04
    box(ax, xs[0], y1, w, h, "Subject", "one eye covered,\nlooks at the dot", fc="#fff8e6", ec="#b26a00")
    box(ax, xs[1], y1, w, h, "Muse headband", "4 dry pads + a cup\nat Oz, 256 Hz")
    box(ax, xs[2], y1, w, h, "Phone", "MuseLog app:\nBluetooth in,\nOSC out")
    box(ax, xs[0], y2, w, h, "Quest 3S", "stimulus page in\nits browser,\n12 Hz at 72 fps")
    box(ax, xs[1], y2, w, h, "PC driver", "logs every frame,\nscores each test")
    box(ax, xs[2], y2, w, h, "Read-out", "thresholds, maps,\nreport", fc="#eef5ee", ec="#2e7d32")
    ym1, ym2 = y1 + h / 2, y2 + h / 2
    arrow(ax, (xs[0] + w, ym1), (xs[1], ym1), "EEG")
    arrow(ax, (xs[1] + w, ym1), (xs[2], ym1))
    arrow(ax, (xs[2] + 0.04, y1), (xs[1] + w - 0.03, y2 + h), "Wi-Fi, UDP", off=(0.045, -0.035), ha="left")
    arrow(ax, (xs[1], ym2), (xs[0] + w, ym2), "USB")
    arrow(ax, (xs[1] + w, ym2), (xs[2], ym2))
    arrow(ax, (xs[0] + w / 2, y2 + h), (xs[0] + w / 2, y1), "shows", off=(0.012, -0.02), ha="left")
    save(fig, out, "fig_pipeline.pdf")


# ── the stimuli ──────────────────────────────────────────────────────────────
def byte(L):
    return round(255 * max(0.0, min(1.0, L)) ** (1 / GAMMA)) / 255


def grating(c, period_px, angle_deg=0.0, n=260, aspect=1.3):
    """A square-wave grating of contrast c (mean luminance 0.5), bars at
    angle_deg from vertical, as an RGB image of n x n*aspect pixels."""
    hgt, wid = n, int(n * aspect)
    yy, xx = np.mgrid[0:hgt, 0:wid]
    th = math.radians(angle_deg)
    u = xx * math.cos(th) + yy * math.sin(th)
    hi, lo = byte(0.5 * (1 + c)), byte(0.5 * (1 - c))
    v = np.where((u // period_px) % 2 == 0, hi, lo)
    return np.repeat(v[:, :, None], 3, axis=2)


def window(ax, img=None, fill=None, dot=(0.5, 0.5)):
    W, H = 1.3, 1.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="black", ec="none"))
    if fill is not None:
        ax.add_patch(Rectangle((0, 0), W, H, fc=fill, ec="none", zorder=2))
    if img is not None:
        ax.imshow(img, extent=(0, W, 0, H), interpolation="nearest", zorder=2)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
    cx, cy = dot[0] * W, dot[1] * H
    ax.add_patch(Circle((cx, cy), 0.055, fc="black", ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy), 0.033, fc="white", ec="none", zorder=6))
    ax.add_patch(Rectangle((0, 0), W, H, fc="none", ec="#777", lw=0.5, zorder=7))
    return W, H


def vernier_img(n=260, aspect=1.3, bars=6, shift=0.45):
    """Coarse bars cut into strips three bars high; every other strip shifted
    sideways (exaggerated here so the offset is visible in print)."""
    hgt, wid = n, int(n * aspect)
    period = wid / bars
    strip = int(round(3 * period / 2))
    yy, xx = np.mgrid[0:hgt, 0:wid]
    off = np.where((yy // strip) % 2 == 1, shift * period / 2, 0.0)
    v = np.where(((xx + off) // (period / 2)) % 2 == 0, 1.0, 0.0)
    return np.repeat(v[:, :, None], 3, axis=2)


def stimuli(out):
    kinds = [
        ("flash", "Flash: white/\nblack; sweep\n6 to 36 Hz"),
        ("acuity", "Acuity: bars\nswap places,\n40 to 3.5'"),
        ("contrast", "Contrast, CSF:\n1 to 0.03 at\n3 bar widths"),
        ("vernier", "Vernier: strips\njump by\n8 to 0.5'"),
        ("field", "Field: one\nquadrant\nat a time"),
        ("gaze", "Gaze: dot at\ncentre, then\nnear each edge"),
        ("meridian", "Meridians:\nbars at 0, 45,\n90, 135 deg"),
        ("photometry", "Photometry:\nred/green swap,\nred 0.2 to 0.8"),
        ("colour", "Colour: red,\ngreen or blue\non black"),
    ]
    fig = plt.figure(figsize=(7.1, 1.12))
    n = len(kinds)
    gap = 0.012
    wpan = (1 - gap * (n + 1)) / n
    for j, (kind, label) in enumerate(kinds):
        ax = fig.add_axes([gap + j * (wpan + gap), 0.40, wpan, 0.58])
        if kind == "flash":
            window(ax, fill="white")
        elif kind == "acuity":
            window(ax, img=grating(1.0, 9))
        elif kind == "contrast":
            window(ax, img=grating(0.3, 20))
        elif kind == "vernier":
            window(ax, img=vernier_img())
        elif kind == "field":
            W, H = window(ax)
            ax.add_patch(Rectangle((0, H / 2), W / 2, H / 2, fc="white", ec="none", zorder=4))
        elif kind == "gaze":
            window(ax, fill="white", dot=(0.5, 0.5 + 0.5 * 0.84))
        elif kind == "meridian":
            window(ax, img=grating(1.0, 12, angle_deg=45))
        elif kind == "photometry":
            W, H = window(ax)
            ax.add_patch(Polygon([(0, 0), (0, H), (W, H)], closed=True, fc="#d81e1e", ec="none", zorder=3))
            ax.add_patch(Polygon([(0, 0), (W, 0), (W, H)], closed=True, fc="#1fa31f", ec="none", zorder=3))
        elif kind == "colour":
            window(ax, fill="#00c000")
        ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                linespacing=1.15)
    save(fig, out, "fig_stimuli.pdf")


# ── the phantom gate ─────────────────────────────────────────────────────────
def plot_axes(width=FULL, height=1.45, left=0.14, bottom=0.25, right=0.98, top=0.95):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    return fig, ax


def field_map(ax, fig, ro, eye):
    d = ro["eyes"][eye]["field"]
    rows, cols = d["rows"], d["cols"]
    rel = np.array([np.nan if v is None else v for v in d["rel"]], float).reshape(rows, cols)
    p = np.array(d["p"], float)
    im = ax.imshow(np.nan_to_num(rel), cmap="magma", vmin=0, vmax=1)
    for k in range(d["n"]):
        y, x = divmod(k, cols)
        if p[k] < 0.05 and np.isfinite(rel.flat[k]):
            ax.text(x, y, f"{rel.flat[k]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if rel.flat[k] < 0.55 else "black")
        else:
            ax.text(x, y, "×", ha="center", va="center", fontsize=10, color="#e74c3c")
    ax.plot(cols / 2 - 0.5, rows / 2 - 0.5, "o", ms=3.5, mfc="white", mec="black", mew=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=6)


def meridian_plot(ax, ro):
    for th in (0.5, 1.0):
        ax.add_patch(plt.Circle((0, 0), th, fill=False, color="#ddd", lw=0.5))
    for eye, r in ro["eyes"].items():
        d = r.get("meridian", {})
        if not d.get("ok"):
            continue
        st = EYE_STYLE.get(eye, dict(color="#555", marker="o"))
        ang = np.array(d["angles"], float); p = np.array(d["p"], float)
        rel = np.array([np.nan if v is None else v for v in d["rel"]], float)
        off = 0.0 if eye == "right" else 0.07
        for a_, r_, p_ in zip(ang, rel, p):
            dx, dy = math.sin(math.radians(a_)), math.cos(math.radians(a_))
            r_ = float(np.clip(np.nan_to_num(r_), 0, 1.1))
            ox, oy = -dy * off, dx * off
            ax.plot([-dx * r_ + ox, dx * r_ + ox], [-dy * r_ + oy, dy * r_ + oy], color=st["color"],
                    lw=2.0 if p_ < 0.05 else 0.9, ls="-" if p_ < 0.05 else "--", solid_capstyle="round")
        ax.plot([], [], color=st["color"], lw=2.0, label=f"{eye} eye")
        if eye == "right":
            for a_ in ang:
                dx, dy = math.sin(math.radians(a_)), math.cos(math.radians(a_))
                ax.text(dx * 1.33, dy * 1.33, f"{a_:g}°", ha="center", va="center", fontsize=6.3)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def gate(out):
    ro = readout("runs/exam_gate")
    fig, ax = plot_axes()
    curve(ax, ro, "acuity", "sizes", "bar width (arcmin)", math.e)
    ax.set_xticks([40, 20, 10, 5, 3.5]); ax.set_xticklabels(["40", "20", "10", "5", "3.5"]); ax.minorticks_off()
    save(fig, out, "fig_gate_acuity.pdf")
    fig, ax = plot_axes()
    curve(ax, ro, "contrast", "contrasts", "contrast", 10)
    ax.set_xticks([1, 0.3, 0.1, 0.03]); ax.set_xticklabels(["1", "0.3", "0.1", "0.03"]); ax.minorticks_off()
    save(fig, out, "fig_gate_contrast.pdf")
    fig, ax = plot_axes(HALF, 1.35, left=0.04, bottom=0.05, right=0.80, top=0.97)
    field_map(ax, fig, ro, "right")
    save(fig, out, "fig_gate_field.pdf")
    fig, ax = plot_axes(HALF, 1.35, left=0.27, bottom=0.17, right=0.98, top=0.95)
    colour(ax, ro)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, fontsize=6, ncol=1,
              handles=[Patch(facecolor="white", edgecolor=EYE_STYLE[e]["color"], label=f"{e} eye")
                       for e in ("right", "left")])
    ax.set_ylim(0, 1.45)
    save(fig, out, "fig_gate_colour.pdf")

    fig, ax = plot_axes()
    curve(ax, ro, "vernier", "offsets", "vernier offset (arcmin)", math.e)
    ax.set_xticks([8, 4, 2, 1, 0.5]); ax.set_xticklabels(["8", "4", "2", "1", "0.5"]); ax.minorticks_off()
    save(fig, out, "fig_gate_vernier.pdf")
    fig, ax = plot_axes(HALF, 1.45, left=0.27, bottom=0.25, right=0.97, top=0.95)
    csf(ax, ro)
    ax.set_ylabel("log sensitivity")
    save(fig, out, "fig_gate_csf.pdf")
    fig, ax = plot_axes(HALF, 1.45, left=0.02, bottom=0.02, right=0.98, top=0.98)
    meridian_plot(ax, ro)
    ax.legend(frameon=False, loc="lower left", fontsize=6, handlelength=1.2, bbox_to_anchor=(-0.04, -0.04))
    save(fig, out, "fig_gate_meridian.pdf")
    fig, ax = plot_axes()
    photometry(ax, ro)
    ax.set_ylim(-0.08, 1.45)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2)
    save(fig, out, "fig_gate_photometry.pdf")


# ── the fourth human sitting ─────────────────────────────────────────────────
def human(out):
    ro = readout("runs/vr_exam6")
    eyes = {"right": ro["eyes"]["right"], "left": ro["eyes"]["left"]}
    col = {"right": RIGHT_H, "left": LEFT_H}
    ch = ["TP9", "AF7", "AF8", "TP10", "AUX"]

    fig, ax = plot_axes(HALF, 1.5, left=0.25, bottom=0.17, right=0.98, top=0.97)
    x = np.arange(len(ch))
    for k, (eye, r) in enumerate(eyes.items()):
        fl = r["flash"]
        amps = [fl["amps_ch"].get(c, 0.0) for c in ch]
        ps = [fl["channel_p"].get(c, 1.0) for c in ch]
        bars = ax.bar(x + (k - 0.5) * 0.38, amps, 0.36, color=col[eye])
        for b, p in zip(bars, ps):
            if p >= ALPHA:
                b.set_hatch("////"); b.set_alpha(0.45); b.set_edgecolor(col[eye])
    ax.set_xticks(x); ax.set_xticklabels(["TP9", "AF7", "AF8", "TP10", "Oz"], fontsize=6)
    ax.set_ylabel("flash amplitude (a.u.)")
    ax.legend(handles=[Patch(color=RIGHT_H, label="right eye"), Patch(color=LEFT_H, label="left eye")],
              frameon=False, fontsize=6, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_human_flash.pdf")

    fig, ax = plot_axes(HALF, 1.5, left=0.06, bottom=0.04, right=0.98, top=0.96)
    ax.set_xlim(0, 2.2); ax.set_ylim(0, 4.75); ax.set_aspect("equal"); ax.axis("off")
    cmap = matplotlib.colormaps["magma"]
    for k, (eye, r) in enumerate(eyes.items()):
        fi = r["field"]
        y0 = 2.45 - k * 2.4
        for s in range(fi["n"]):
            cx, cy = s % fi["cols"], s // fi["cols"]
            v = fi["rel"][s] if fi["rel"][s] is not None else float("nan")
            resp = s in fi["responding"]
            face = cmap(0.15 + 0.8 * min(max(v, 0), 1)) if resp and np.isfinite(v) else "#f2f2f2"
            ax.add_patch(plt.Rectangle((0.2 + cx, y0 + 1.0 - cy * 1.0), 0.95, 0.95, facecolor=face,
                                       edgecolor="#333", lw=0.5))
            ax.text(0.2 + cx + 0.475, y0 + 1.0 - cy + 0.475, f"{v:.2f}" if resp else "×",
                    ha="center", va="center", fontsize=7, color="#0b0b0b" if resp and v > 0.5 else ("white" if resp else "#0b0b0b"))
        ax.text(0.2 + 1.0, y0 + 2.08, f"{eye} eye", ha="center", fontsize=7, color=col[eye])
    save(fig, out, "fig_human_quadrants.pdf")

    fig, ax = plot_axes()
    for eye, r in eyes.items():
        te = r["temporal"]
        rates, amps, ps = np.array(te["rates"]), np.array(te["amps_corr"]), np.array(te["p"])
        ax.plot(rates, amps, color=col[eye], lw=1.0, zorder=1, label=f"{eye} eye")
        sig = ps < ALPHA
        ax.plot(rates[sig], amps[sig], "o", color=col[eye], ms=3.5)
        ax.plot(rates[~sig], amps[~sig], "o", mfc="white", color=col[eye], ms=3.5)
    ax.set_xscale("log"); ax.set_xticks([6, 7.2, 9, 12, 18, 36])
    ax.set_xticklabels(["6", "7.2", "9", "12", "18", "36"]); ax.minorticks_off()
    ax.set_xlabel("flicker rate (Hz)"); ax.set_ylabel("amplitude on Oz (a.u.)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_human_sweep.pdf")

    fig, ax = plot_axes()
    for eye, r in eyes.items():
        ac = r["acuity"]
        sizes, amps, ps = np.array(ac["sizes"]), np.array(ac["amps"]), np.array(ac["p"])
        sig = ps < ALPHA
        ax.plot(sizes[sig], amps[sig], "o", color=col[eye], ms=3.5, label=f"{eye} eye")
        ax.plot(sizes[~sig], amps[~sig], "o", mfc="white", color=col[eye], ms=3.5)
        fit = ac.get("fit") or {}
        if ac.get("ok") and fit.get("ok") and fit.get("x_used"):
            xl = np.linspace(math.log(ac["threshold"]), max(fit["x_used"]), 20)
            ax.plot(np.exp(xl), fit["a"] + fit["b"] * xl, color=col[eye], lw=1.2)
            ax.axvline(ac["threshold"], color=col[eye], lw=0.8, ls=(0, (3, 2)))
            ax.text(ac["threshold"] * 1.08, 0.3, f"{ac['threshold']:.1f}'", color=col[eye], fontsize=6.5)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xticks([40, 20, 10, 5, 3]); ax.set_xticklabels(["40", "20", "10", "5", "3"]); ax.minorticks_off()
    ax.set_xlabel("bar width (arcmin, placeholder scale)"); ax.set_ylabel("amplitude (a.u.)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_human_bars.pdf")


# ── the sixth human sitting (3 November 2026) ────────────────────────────────
SIXTH = {"left": "runs/vr_exam_left_2026-11-03b", "right": "runs/vr_exam_right_2026-11-03"}


def _photometry_half(ax, eyes, col, ylim, ylabel, legend="lower right"):
    """One half-column photometry panel: per eye (readout, colour) the step
    amplitudes (open: p >= 0.05), the fitted null model and the null (dashed);
    the display's standard match at 0.5 dotted."""
    from exam_readout import photometry_modulation
    for eye, ro in eyes.items():
        d = ro["eyes"][eye]["photometry"]["pairs"]["rg"]
        x, a, p = np.array(d["mixes"], float), np.array(d["amps"], float), np.array(d["p"], float)
        sig = p < ALPHA
        ax.plot(x[sig], a[sig], "o", color=col[eye], ms=3.0, label=eye)
        ax.plot(x[~sig], a[~sig], "o", mfc="white", color=col[eye], ms=3.0)
        if d.get("fit_ok"):
            xx = np.linspace(0.12, 0.88, 120)
            ax.plot(xx, [d["a"] * photometry_modulation(v, d["null"]) + d["b"] for v in xx],
                    color=col[eye], lw=0.8)
            ax.axvline(d["null"], color=col[eye], lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0.5, color="#888", lw=0.5, ls=":")
    ax.set_xlim(0.1, 0.9); ax.set_ylim(*ylim)
    ax.set_xticks([0.2, 0.35, 0.5, 0.65, 0.8]); ax.set_xticklabels(["0.2", "", "0.5", "", "0.8"])
    ax.set_xlabel("red's share"); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc=legend, handlelength=0.8, fontsize=6, borderaxespad=0.1,
              ncol=2 if legend == "upper center" else 1, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def human6(out):
    """Red/green flicker photometry as a half-column pair: the synthetic subject
    of the gate (fig_gate_photometry_half.pdf, nulls dialled at 0.5 and 0.65)
    and the person in the sixth sitting (fig_human_photometry.pdf, Oz)."""
    gate_ro = readout("runs/exam_gate")
    fig, ax = plot_axes(HALF, 1.4, left=0.25, bottom=0.25, right=0.97, top=0.95)
    _photometry_half(ax, {"right": gate_ro, "left": gate_ro},
                     {e: EYE_STYLE[e]["color"] for e in ("right", "left")}, (-0.05, 1.45), "amplitude (a.u.)",
                     legend="upper center")
    save(fig, out, "fig_gate_photometry_half.pdf")
    fig, ax = plot_axes(HALF, 1.4, left=0.25, bottom=0.25, right=0.97, top=0.95)
    _photometry_half(ax, {e: readout(SIXTH[e]) for e in ("right", "left")},
                     {"right": RIGHT_H, "left": LEFT_H}, (0, 8.2), "amplitude on Oz (a.u.)")
    save(fig, out, "fig_human_photometry.pdf")


LAPTOP_CHART = "runs/lap_chart_2026-11-03"     # the bars at chart sizes, laptop panel at 1.5 m
LIGHT_LADDER = "runs/vr_light_2026-11-03c"     # the light ladder, four repeats per level


def _limb_panel(ax, key_x, test, ro, logf, col):
    """Amplitude against the step variable for both eyes (open: p >= 0.05), the
    fitted limb down to its zero crossing and the threshold (dashed)."""
    for eye in ("right", "left"):
        d = ro["eyes"][eye][test]
        x, a, p = np.array(d[key_x], float), np.array(d["amps"], float), np.array(d["p"], float)
        sig = p < ALPHA
        ax.plot(x[sig], a[sig], "o", color=col[eye], ms=3.0, label=eye)
        ax.plot(x[~sig], a[~sig], "o", mfc="white", color=col[eye], ms=3.0)
        fit = d.get("fit") or {}
        if d.get("ok") and fit.get("ok"):
            xs = np.array(fit["x_used"])
            xx = np.linspace(min(fit["x_thr_log"], xs.min()), xs.max(), 30)
            ax.plot(logf(xx), fit["a"] + fit["b"] * xx, color=col[eye], lw=0.8)
            ax.axvline(d["threshold"], color=col[eye], lw=0.6, ls=(0, (3, 2)))
    ax.set_xscale("log"); ax.axhline(0, color="#999", lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def human7(out):
    """The afternoon of the sixth day: the chart as reversing bars on the laptop
    (physical units) and the light ladder in the headset, both eyes each."""
    col = {"right": RIGHT_H, "left": LEFT_H}
    fig, ax = plot_axes(HALF, 1.45, left=0.25, bottom=0.27, right=0.97, top=0.95)
    _limb_panel(ax, "sizes", "acuity", readout(LAPTOP_CHART), np.exp, col)
    ax.invert_xaxis()
    ticks = [10, 5, 2.5, 1, 0.62]
    ax.set_xticks(ticks); ax.set_xticklabels(["200", "100", "50", "20", "12"]); ax.minorticks_off()
    ax.set_xlabel("bars as chart line 20/X"); ax.set_ylabel("amplitude on Oz (a.u.)")
    ax.legend(frameon=False, loc="lower left", fontsize=6, handlelength=0.8)
    save(fig, out, "fig_human_laptop_bars.pdf")
    fig, ax = plot_axes(HALF, 1.45, left=0.25, bottom=0.27, right=0.91, top=0.95)
    _limb_panel(ax, "levels", "intensity", readout(LIGHT_LADDER), lambda v: 10 ** v, col)
    ax.set_xticks([0.01, 0.1, 1]); ax.set_xticklabels(["1%", "10%", "100%"]); ax.minorticks_off()
    ax.set_xlabel("flash level (of white)"); ax.set_ylabel("amplitude on Oz (a.u.)")
    ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=0.8)
    save(fig, out, "fig_human_ladder.pdf")


def gate_pair(out):
    """The gate's acuity and contrast curves as a half-column pair (the paper's
    sixth-day additions needed the room)."""
    ro = readout("runs/exam_gate")
    fig, ax = plot_axes(HALF, 1.45, left=0.24, bottom=0.27, right=0.97, top=0.95)
    curve(ax, ro, "acuity", "sizes", "bar width (arcmin)", math.e)
    ax.set_xticks([40, 10, 3.5]); ax.set_xticklabels(["40", "10", "3.5"]); ax.minorticks_off()
    if ax.get_legend():
        ax.get_legend().remove()
    save(fig, out, "fig_gate_acuity_half.pdf")
    fig, ax = plot_axes(HALF, 1.45, left=0.24, bottom=0.27, right=0.97, top=0.95)
    curve(ax, ro, "contrast", "contrasts", "contrast", 10)
    ax.set_xticks([1, 0.1, 0.03]); ax.set_xticklabels(["1", "0.1", "0.03"]); ax.minorticks_off()
    if ax.get_legend():
        ax.legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=0.8)
    save(fig, out, "fig_gate_contrast_half.pdf")


PARTS = {"rig": rig_photo, "pipeline": pipeline, "stimuli": stimuli, "gate": gate, "human": human,
         "human6": human6, "human7": human7, "gate_pair": gate_pair}


def main():
    """python analysis/exam_paper_panels.py [out_dir] [part ...]; parts: rig,
    pipeline, stimuli, gate, human, human6 (default: all)."""
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "paper", "exam2027", "figures")
    os.makedirs(out, exist_ok=True)
    # the imported modules set smaller fonts at import; the human plots come last with ours
    for name in (sys.argv[2:] or PARTS):
        PARTS[name](out)


if __name__ == "__main__":
    main()
