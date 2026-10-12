"""Ear as a Hydrophone, the optical channels: what the Muse S Athena's
fNIRS / PPG stream carried during the hydraulophone recording. MuseLog
writes the optical samples on the EEG rows (held between optical samples),
so the series are taken at the row rate, low-passed and decimated. From the
short (inner) and long (outer) detector pairs at 730 and 850 nm:
  pulse     band-pass 0.7-3 Hz, the spectral peak as heart rate, and the
            rate per 20 s window
  breathing band-pass 0.1-0.5 Hz on the same channel
  haemo     modified Beer-Lambert on 730 / 850 for dHbO and dHbR (relative
            units), smoothed over 10 s, left and right, inner and outer

    python analysis/hydrophone_optics.py <csv> <out_dir>

Writes fig_optics.png and hydrophone_optics.json into <out_dir>.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, sosfiltfilt, welch

FS = 256.0
DEC = 8                      # 256 -> 32 Hz
FO = FS / DEC
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe5"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def series(d, name):
    v = pd.to_numeric(d[name], errors="coerce").interpolate().bfill().ffill().to_numpy()
    return decimate(v, DEC, ftype="fir", zero_phase=True)


def main():
    path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    d = pd.read_csv(path, low_memory=False)
    t = None
    out = dict(file=os.path.basename(path), fs_optical_after_decimation=FO, channels={})
    S = {}
    for side in ("LEFT", "RIGHT"):
        for sep in ("INNER", "OUTER"):
            for wl in ("730nm", "850nm"):
                S[(side, sep, wl)] = series(d, f"{wl}_{side}_{sep}")
    n = min(len(v) for v in S.values())
    t = np.arange(n) / FO
    # pulse and breathing on the 850 nm inner channels (the strongest)
    sos_p = butter(3, [0.7, 3.0], btype="band", fs=FO, output="sos")
    sos_b = butter(3, [0.1, 0.5], btype="band", fs=FO, output="sos")
    pulse = {}
    for side in ("LEFT", "RIGHT"):
        v = S[(side, "INNER", "850nm")][:n]
        y = sosfiltfilt(sos_p, v - v.mean())
        f, P = welch(y, fs=FO, nperseg=int(60 * FO))
        sel = (f > 0.7) & (f < 3.0)
        fpk = float(f[sel][np.argmax(P[sel])])
        nb = P[sel]; prom = float(P[sel].max() / np.median(nb))
        rates = []
        W = int(20 * FO)
        for k in range(n // W):
            seg = y[k * W:(k + 1) * W]
            fw, Pw = welch(seg, fs=FO, nperseg=len(seg))
            sw = (fw > 0.7) & (fw < 3.0)
            rates.append(float(fw[sw][np.argmax(Pw[sw])] * 60))
        b = sosfiltfilt(sos_b, v - v.mean())
        fb, Pb = welch(b, fs=FO, nperseg=int(120 * FO))
        sb = (fb > 0.1) & (fb < 0.5)
        pulse[side] = dict(peak_hz=fpk, bpm=fpk * 60, prominence=prom, rel_amplitude=float(y.std() / v.mean()),
                           bpm_per_20s=rates, breathing_hz=float(fb[sb][np.argmax(Pb[sb])]), spectrum=(f[sel].tolist(), P[sel].tolist()), trace=y)
        out["channels"][side] = {k: v for k, v in pulse[side].items() if k not in ("spectrum", "trace")}
        print(f"{side} 850 nm inner: pulse peak {fpk:.2f} Hz = {fpk*60:.0f} bpm (prominence x{prom:.1f}), rel. amplitude {pulse[side]['rel_amplitude']*100:.2f} %, "
              f"per-20 s rates {np.round(rates, 0).astype(int).tolist()}, breathing {pulse[side]['breathing_hz']:.2f} Hz")
    # haemodynamics by modified Beer-Lambert, relative units
    E = np.array([[0.39, 1.10], [1.06, 0.69]])   # rows 730, 850; columns HbO, HbR (approximate extinction, scaled)
    k10 = int(10 * FO)
    sm = lambda z: np.convolve(z, np.ones(k10) / k10, mode="same")
    haemo = {}
    for side in ("LEFT", "RIGHT"):
        for sep in ("INNER", "OUTER"):
            a730 = -np.log(np.clip(S[(side, sep, "730nm")][:n], 1e-3, None) / S[(side, sep, "730nm")][:n].mean())
            a850 = -np.log(np.clip(S[(side, sep, "850nm")][:n], 1e-3, None) / S[(side, sep, "850nm")][:n].mean())
            hbo, hbr = np.linalg.solve(E, np.vstack([a730, a850]))
            haemo[(side, sep)] = (sm(hbo), sm(hbr))
            out["channels"][f"{side}_{sep}_haemo"] = dict(dHbO_first_to_last_min=float(np.mean(sm(hbo)[-int(60 * FO):]) - np.mean(sm(hbo)[:int(60 * FO)])),
                                                           dHbR_first_to_last_min=float(np.mean(sm(hbr)[-int(60 * FO):]) - np.mean(sm(hbr)[:int(60 * FO)])),
                                                           dHbO_sd=float(sm(hbo).std()), dHbR_sd=float(sm(hbr).std()))
            print(f"{side} {sep}: dHbO first->last minute {out['channels'][f'{side}_{sep}_haemo']['dHbO_first_to_last_min']:+.4f}, dHbR {out['channels'][f'{side}_{sep}_haemo']['dHbR_first_to_last_min']:+.4f} (relative units)")
    json.dump(out, open(os.path.join(out_dir, "hydrophone_optics.json"), "w"), indent=1, default=float)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2))
    ax = axes[0, 0]
    for side, col in (("LEFT", C1), ("RIGHT", C2)):
        f, P = pulse[side]["spectrum"]
        ax.plot(np.array(f) * 60, P, color=col, lw=1.4, label=f"{side.lower()}, {pulse[side]['bpm']:.0f} bpm")
    ax.set_xlabel("beats per minute", fontsize=9, color=MUTED); ax.set_ylabel("power", fontsize=9, color=MUTED); ax.legend(fontsize=8, frameon=False)
    style(ax, "a. pulse in the 850 nm inner channel")
    ax = axes[0, 1]
    seg = slice(int(120 * FO), int(130 * FO))
    for side, col, off in (("LEFT", C1, 0.0), ("RIGHT", C2, 1.0)):
        y = pulse[side]["trace"][seg]; y = y / (y.std() + 1e-9)
        ax.plot(t[seg] - t[seg][0], y + off * 4, color=col, lw=1.0)
    ax.set_xlabel("seconds (from minute 2)", fontsize=9, color=MUTED); ax.set_ylabel("pulse, normalised", fontsize=9, color=MUTED); ax.set_yticks([])
    style(ax, "b. ten seconds of the pulse, both temples")
    ax = axes[1, 0]
    for side, col in (("LEFT", C1), ("RIGHT", C2)):
        r = pulse[side]["bpm_per_20s"]
        ax.plot(np.arange(len(r)) * 20 / 60 + 1 / 6, r, "o-", color=col, ms=3, lw=1, label=side.lower())
    ax.set_xlabel("minutes", fontsize=9, color=MUTED); ax.set_ylabel("heart rate, bpm", fontsize=9, color=MUTED); ax.legend(fontsize=8, frameon=False)
    style(ax, "c. heart rate per 20 s")
    ax = axes[1, 1]
    for side, col in (("LEFT", C1), ("RIGHT", C2)):
        hbo, hbr = haemo[(side, "INNER")]
        ax.plot(t / 60, hbo, color=col, lw=1.2, label=f"{side.lower()} dHbO")
        ax.plot(t / 60, hbr, color=col, lw=1.2, ls="--", label=f"{side.lower()} dHbR")
    ax.set_xlabel("minutes", fontsize=9, color=MUTED); ax.set_ylabel("relative change, 10 s smoothing", fontsize=9, color=MUTED); ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax, "d. haemoglobin, inner detectors")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_optics.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
