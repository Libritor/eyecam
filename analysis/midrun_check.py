"""Is a grey scan working, while it runs? Reads eeg.csv, cursor_log.csv and
calibration.json of a session in progress and answers three questions with
one figure: is the electrode still good, is the flicker still driving the eye
(12 Hz response per 8 s window and per cell against the calibration ON and
OFF blocks), and does the response follow the grey shown (identical cells
should agree; brighter cells should respond more).

    python analysis/midrun_check.py runs/vr_grey1            # figure + verdict
    python analysis/midrun_check.py runs/vr_grey1 --first 16 # the first N cells spelled out

Writes <session>/midrun_check.png. The SNR here is the 12 Hz line over its
neighbours (0.5-3 Hz away) in one 8 s window; the driver's own scores are in
reconstruct.py, this is the quick look.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CHANS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
FS = 256.0


def spectrum(x):
    x = x - x.mean()
    x = x * np.hanning(len(x))
    P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    return np.fft.rfftfreq(len(x), 1 / FS), P


def snr(x, f0):
    f, P = spectrum(x)
    k = int(np.argmin(np.abs(f - f0)))
    nb = (np.abs(f - f0) > 0.5) & (np.abs(f - f0) < 3.0)
    return float(P[k - 1:k + 2].sum() / 3 / P[nb].mean())


def dwells(cur, t, eeg, f0, chans):
    """One row per cell shown for at least 5 s: its grey and the SNR per channel."""
    scan = cur[cur.grid_x >= 0].reset_index(drop=True)
    key = scan.grid_x.astype(int) * 1000 + scan.grid_y.astype(int)
    brk = np.where(np.diff(key.to_numpy()) != 0)[0] + 1
    bounds = np.concatenate([[0], brk, [len(scan)]])
    rows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = scan.iloc[s:e]
        t0, t1 = seg.time.iloc[0], seg.time.iloc[-1]
        m = (t >= t0 + 0.5) & (t <= t1)
        if t1 - t0 < 5 or m.sum() < 3 * FS:
            continue
        rows.append(dict(x=int(seg.grid_x.iloc[0]), y=int(seg.grid_y.iloc[0]), lum=float(seg.luminance.iloc[0]),
                         t0=t0, t1=t1, **{c: snr(eeg[c].to_numpy()[m], f0) for c in chans}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--freq", type=float, default=12.0)
    ap.add_argument("--first", type=int, default=0, help="print the first N cells one by one")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    eeg = pd.read_csv(os.path.join(a.session, "eeg.csv"))
    cur = pd.read_csv(os.path.join(a.session, "cursor_log.csv"))
    cal = json.load(open(os.path.join(a.session, "calibration.json")))
    chans = [c for c in CHANS if c in eeg.columns]
    best = cal.get("best", "AUX") if cal.get("best", "AUX") in chans else chans[-1]
    t = eeg["time"].to_numpy()
    tgt = np.load(os.path.join(a.session, "target.npy")) if os.path.exists(os.path.join(a.session, "target.npy")) else None

    # calibration blocks
    on, off = [], []
    for kind, b0, b1 in cal["blocks"]:
        m = (t >= b0 + 0.5) & (t <= b1)
        (on if kind == "on" else off).append(snr(eeg[best].to_numpy()[m], a.freq))
    print(f"calibration, {best} {a.freq:g} Hz SNR: ON {np.round(on, 1).tolist()}  OFF {np.round(off, 1).tolist()}")

    # time course, 8 s windows every 4 s
    t0 = t[0]
    tc = []
    for s in np.arange(t0, t[-1] - 8, 4.0):
        m = (t >= s) & (t < s + 8)
        if m.sum() < 6 * FS:
            continue
        v = eeg[best].to_numpy()[m]
        tc.append(dict(t=(s - t0) / 60, snr=snr(v, a.freq), rail=float(np.mean((v <= 0) | (v >= 1680))), sd=float(v.std())))
    tc = pd.DataFrame(tc)

    # cells
    df = dwells(cur, t, eeg, a.freq, chans)
    if len(df) == 0:
        print("no cell shown for 5 s yet"); return
    scan_t0, scan_t1 = (df.t0.min() - t0) / 60, (df.t1.max() - t0) / 60
    seg_early = tc[(tc.t >= scan_t0) & (tc.t < scan_t0 + 2)]
    seg_late = tc[(tc.t >= scan_t0 + 2) & (tc.t <= scan_t1)]
    print(f"scan: {len(df)} cells, {scan_t1 - scan_t0:.1f} min; {best} SNR median first 2 min {seg_early.snr.median():.1f}, "
          f"after {seg_late.snr.median():.1f} (calibration ON {np.median(on):.1f}, OFF {np.median(off):.1f})")
    scan_tc = tc[(tc.t >= scan_t0) & (tc.t <= scan_t1)]
    print(f"electrode during the scan: {best} at rail {100 * scan_tc.rail.mean():.1f}% of samples, sd {scan_tc.sd.median():.0f} uV")
    r = np.corrcoef(df[best], df.lum)[0, 1] if df.lum.std() > 0 else float("nan")
    print(f"response vs grey over {len(df)} cells: r {r:+.2f}")
    for lv, g in df.groupby("lum"):
        print(f"  grey {lv:.2f}: n {len(g):3d}, {best} SNR median {g[best].median():5.1f}, "
              f"spread (90th/10th pct) {g[best].quantile(.9) / max(g[best].quantile(.1), 1e-3):.0f}x")
    if a.first:
        print(df.head(a.first)[["x", "y", "lum"] + chans].round(2).to_string(index=False))
    in_on = float(np.mean(df[best] >= min(on)))
    verdict = ("DRIVING: most cells at calibration strength" if in_on > 0.6 else
               "WEAK: few cells at calibration strength" if in_on > 0.2 else
               "NOT DRIVING: the eye is not responding to the patch")
    follow = f"grey followed (r {r:+.2f})" if r > 0.3 else f"grey NOT followed (r {r:+.2f})"
    print(f"verdict: {verdict}; {follow}; {100 * in_on:.0f}% of cells in the calibration ON band")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ax = axs[0, 0]
    for kind, b0, b1 in cal["blocks"]:
        ax.axvspan((b0 - t0) / 60, (b1 - t0) / 60, color="tab:green" if kind == "on" else "tab:red", alpha=0.12)
    ax.axvspan(scan_t0, scan_t1, color="tab:blue", alpha=0.08)
    ax.semilogy(tc.t, tc.snr, "k.-", ms=3, lw=0.8)
    ax.axhline(1, color="grey", lw=0.5)
    ax.set_ylabel(f"{a.freq:g} Hz SNR, {best}"); ax.set_xlabel("minutes since recording start")
    ax.set_title("response in time (green/red = calibration on/off, blue = scan)", fontsize=10)
    ax2 = ax.twinx(); ax2.plot(tc.t, tc.rail * 100, color="tab:red", lw=0.8, alpha=0.6); ax2.set_ylabel("at rail, %", color="tab:red")
    ax = axs[0, 1]
    ax.axhspan(min(on), max(on), color="tab:green", alpha=0.15, label="calibration ON")
    ax.axhspan(min(off), max(off), color="tab:red", alpha=0.15, label="calibration OFF")
    ax.scatter(np.arange(len(df)), df[best], c=df.lum, cmap="gray", vmin=0, vmax=1, edgecolors="k", s=30)
    ax.set_yscale("log"); ax.set_xlabel("cell in scan order"); ax.set_ylabel(f"{a.freq:g} Hz SNR, {best}")
    ax.set_title("response per cell (shade = the grey shown)", fontsize=10); ax.legend(fontsize=8)
    ax = axs[1, 0]
    levels = sorted(df.lum.unique())
    ax.boxplot([df[df.lum == lv][best] for lv in levels], positions=range(len(levels)), widths=0.5)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{lv:.2f}\nn={int((df.lum == lv).sum())}" for lv in levels])
    ax.set_yscale("log"); ax.set_xlabel("grey shown"); ax.set_ylabel(f"{a.freq:g} Hz SNR, {best}")
    ax.axhspan(min(off), max(off), color="tab:red", alpha=0.1); ax.axhspan(min(on), max(on), color="tab:green", alpha=0.1)
    ax.set_title(f"does the response follow the grey? r {r:+.2f}", fontsize=10)
    ax = axs[1, 1]
    if tgt is not None:
        gh, gw = tgt.shape
        img = np.full((gh, gw, 3), 0.25)
        img[..., :] = np.stack([tgt * 0.35 + 0.1] * 3, axis=2)
        snr_grid = np.full((gh, gw), np.nan)
        for _, rr in df.iterrows():
            snr_grid[int(rr.y), int(rr.x)] = rr[best]
        lo, hi = np.log(max(min(off), 0.05)), np.log(max(on))
        v = np.clip((np.log(np.nan_to_num(snr_grid, nan=0.05)) - lo) / (hi - lo), 0, 1)
        seen = ~np.isnan(snr_grid)
        for yy in range(gh):
            for xx in range(gw):
                if seen[yy, xx]:
                    img[yy, xx] = [v[yy, xx], v[yy, xx], v[yy, xx]]
        ax.imshow(img, interpolation="nearest")
        ax.set_title("cells visited: brightness = response strength (dim = target, not yet shown)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{a.session}: {verdict}; {follow}", fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(a.session, "midrun_check.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
