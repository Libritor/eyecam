"""The first sitting's raw EEG for the paper: (a) each electrode's rms through
every test of the sitting against the 60 uV contact limit; (b) the Oz spectrum
in ON against OFF blocks of the test with the clearest response (the right
eye's colour flicker) and (c) of the right eye's acuity test. Vector PDF.

    python analysis/exam_sitting_figure.py docs/paper/exam2027/figures/exam_sitting.pdf runs/vr_exam1 runs/vr_exam2
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import reconstruct  # noqa: E402
import exam  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5,
                     "font.family": "DejaVu Sans", "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 6, "legend.fontsize": 5.5,
                     "axes.linewidth": 0.5})
ORDER = ["acuity", "contrast", "field", "colour"]
CH = {"TP9": "#8e44ad", "TP10": "#c0392b", "AUX": "#1f4e79"}


def blocks_from_log(path):
    L = np.genfromtxt(path, delimiter=",", skip_header=1)
    lt, lum = L[:, 0], L[:, 3]
    on = lum == 1
    starts = np.where(on[1:] & ~on[:-1])[0] + 1
    ends = np.where(~on[1:] & on[:-1])[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if len(ends) < len(starts):
        ends = np.r_[ends, len(lt) - 1]
    out = []
    for b, (i0, i1) in enumerate(zip(starts, ends)):
        out.append(("on", lt[i0], lt[i1 - 1]))
        j1 = starts[b + 1] if b + 1 < len(starts) else len(lt)
        if j1 - 1 > i1:
            out.append(("off", lt[i1], lt[j1 - 1]))
    return out, (lt[0], lt[-1])


def load(session):
    E = np.genfromtxt(os.path.join(session, "eeg.csv"), delimiter=",", skip_header=1)
    names = exam.csv_header(os.path.join(session, "eeg.csv"))
    return E, names, reconstruct.infer_fs(E[:, 0])


def rms_table(sessions):
    """[(label, {channel: rms})] in the order the tests were run: eyes in the
    session's order, tests in ORDER; eye_names relabels an eye (the restarted
    'left' run of the first sitting was done with both eyes open)."""
    rows = []
    for session, eye_names in sessions:
        E, names, fs = load(session)
        logs = [f for f in os.listdir(session) if f.startswith("exam_") and f.endswith("_log.csv")]
        eyes = []
        for log in sorted(logs, key=lambda f: os.path.getmtime(os.path.join(session, f))):
            eye = log.split("_")[1]
            if eye not in eyes:
                eyes.append(eye)
        for eye in eyes:
            for test in ORDER:
                log = f"exam_{eye}_{test}_log.csv"
                if log not in logs:
                    continue
                _, (t0, t1) = blocks_from_log(os.path.join(session, log))
                seg = E[(E[:, 0] >= t0) & (E[:, 0] <= t1), 1:]
                if len(seg) < 256:
                    continue
                rms = {names[c]: float(np.sqrt(np.mean((seg[:, c] - np.median(seg[:, c])) ** 2)))
                       for c in range(len(names))}
                rows.append((f"{eye_names.get(eye, eye)}\n{test}", rms))
    return rows


def spectra(session, log, fs_hint=None, channel="AUX"):
    E, names, fs = load(session)
    c = names.index(channel) + 1
    blocks, _ = blocks_from_log(os.path.join(session, log))
    P = {"on": [], "off": []}
    f = None
    for kind, t0, t1 in blocks:
        sel = (E[:, 0] >= t0 + 0.5) & (E[:, 0] <= t1)
        if sel.sum() < fs * 2.5:
            continue
        x = E[sel, c][: int(fs * 2.5)]
        x = x - x.mean()
        f, p = periodogram(x, fs=fs, window="hann", detrend="constant", scaling="spectrum")
        P[kind].append(p)
    return f, {k: np.mean(v, 0) for k, v in P.items() if v}


def main():
    out = sys.argv[1]
    s1, s2 = sys.argv[2], sys.argv[3]
    rows = rms_table([(s1, {}), (s2, {"left": "both"})])
    fig = plt.figure(figsize=(3.5, 3.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], left=0.14, right=0.98, top=0.95,
                          bottom=0.13, hspace=0.85, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    labels = []
    for i, (lab, rms) in enumerate(rows):
        for j, ch in enumerate(CH):
            if ch in rms:
                ax.bar(i + (j - 1) * 0.27, rms[ch], width=0.26, color=CH[ch], label=ch if i == 0 else None)
        labels.append(lab)
    ax.axhline(exam.MAX_RMS_UV, color="#333", lw=0.6, ls="--", label="60 µV contact limit")
    ax.set_yscale("log"); ax.set_ylim(5, 400)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=4.6)
    ax.set_ylabel("rms (µV)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=7, weight="bold")
    for k, (log, ttl, lab) in enumerate([("exam_right_colour_log.csv", "right eye, colour flicker", "(b)"),
                                         ("exam_right_acuity_log.csv", "right eye, acuity gratings", "(c)")]):
        ax = fig.add_subplot(gs[1, k])
        f, P = spectra(s1, log)
        sel = (f >= 5) & (f <= 30)
        for kind, col, name in (("on", "#1f4e79", "ON blocks"), ("off", "#999", "OFF blocks")):
            if kind in P:
                ax.plot(f[sel], 10 * np.log10(P[kind][sel] + 1e-12), color=col, lw=0.8, label=name)
        for fx in (12, 24):
            ax.axvline(fx, color="#c0392b", lw=0.5, ls=":")
        ax.set_xlabel("frequency (Hz)")
        if k == 0:
            ax.set_ylabel("Oz power (dB)")
        ax.set_title(ttl, loc="left", fontsize=6)
        ax.legend(frameon=False, handlelength=1.2, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.25 if k == 0 else -0.12, 1.12, lab, transform=ax.transAxes, fontsize=7, weight="bold")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    main()
