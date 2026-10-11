"""State of Flash, the SSVEP's own fatigue meter: the response of every lit
cell against minutes on task, for planes sessions, next to the Oz
electrode's own health (1-40 Hz RMS and 60 Hz mains pick-up per minute,
which climb when the cup dries) so a fading response can be told from a
fading contact.  Per-visit scores are the driver's (reconstruct.run, method
paper: power at f and 2f over the 14-50 Hz floor, blended over channels
with the plane's own weights), dark cells drawn grey.

    python analysis/flash_fatigue.py runs/vr_planes9 runs/vr_planes10

Writes runs/state_of_flash/fig_ssvep_fatigue.png and
runs/state_of_flash/ssvep_fatigue.json.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reconstruct  # noqa: E402
from analysis.planes_passes import visits_of, NAMES, NAMES_EEG  # noqa: E402

COL = {"red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4"}


def plane_scores(S, meta, pcal, k, pname):
    log = meta["logs"][pname]
    rows = list(csv.reader(open(os.path.join(S, log))))
    body = rows[1:]
    vis = visits_of(body)
    e = pcal.get(pname) or {}
    w = None
    if (e.get("passed") or e.get("use_weights")) and e.get("weights"):
        w = np.array([float(e["weights"].get(n, 0.0)) for n in NAMES_EEG])
        if w.sum() <= 0:
            w = None
    freq = float(meta.get("freq_delivered", {}).get(pname, meta.get("freq", 12.0)))
    info = {}
    reconstruct.run(S, calibration="" if w is not None else os.path.join(S, "calibration.json"),
                    weights=w, stim_freq=freq, method="paper", vblend=False, cursor_log=log,
                    save=False, info=info)
    sc = info.get("scores", [])
    if len(sc) != len(vis):
        print(f"  {pname}: {len(sc)} scores for {len(vis)} visits; aligning by order")
    out = []
    for (s0, s1, gx, gy), (gy2, gx2, v) in zip(vis, sc):
        t0 = float(body[s0][0])
        out.append(dict(t=t0, gx=gx, gy=gy, score=float(v), plane=pname, freq=freq))
    return out


def oz_health(S, first_t, fs=256.0):
    t = np.loadtxt(os.path.join(S, "eeg.csv"), delimiter=",", skiprows=1, usecols=0)
    x = np.loadtxt(os.path.join(S, "eeg.csv"), delimiter=",", skiprows=1, usecols=5)
    sos = butter(4, [1, 40], btype="band", fs=fs, output="sos")
    sos60 = butter(4, [58, 62], btype="band", fs=fs, output="sos")
    xb = sosfiltfilt(sos, x - np.median(x))
    x60 = sosfiltfilt(sos60, x - np.median(x))
    mins = np.floor((t - first_t) / 60.0)
    out = []
    for m in np.unique(mins):
        sel = mins == m
        if sel.sum() < fs * 30:
            continue
        seg = xb[sel]
        rms = float(np.sqrt(np.mean(np.clip(seg, -200, 200) ** 2)))
        line = float(np.sqrt(np.mean(x60[sel] ** 2)))
        out.append(dict(minute=float(m) + 0.5, rms=rms, mains=line))
    return out


def main(sessions):
    res = {}
    fig, axs = plt.subplots(len(sessions), 2, figsize=(14, 3.6 * len(sessions)),
                            gridspec_kw=dict(width_ratios=[3, 1.4]), squeeze=False)
    for i, S in enumerate(sessions):
        name = os.path.basename(os.path.normpath(S))
        meta = json.load(open(os.path.join(S, "planes_meta.json")))
        pcal = json.load(open(os.path.join(S, "planes_calib.json")))
        target = np.load(os.path.join(S, "target_color.npy"))
        calib = json.load(open(os.path.join(S, "calibration.json")))
        crow = list(csv.reader(open(os.path.join(S, "calib_log.csv"))))[1:]
        first_t = float(crow[0][0])
        visits = []
        for k, pname in enumerate(NAMES):
            if pname in meta.get("logs", {}):
                visits += plane_scores(S, meta, pcal, k, pname)
        for v in visits:
            v["lit"] = bool(target[v["gy"], v["gx"], NAMES.index(v["plane"])] > 0.5)
            v["min"] = (v["t"] - first_t) / 60.0
        health = oz_health(S, first_t)
        res[name] = dict(visits=visits, oz=health,
                         calib_oz_dprime=calib.get("channels", {}).get("AUX", {}).get("dprime"),
                         plane_calib={p: dict(stat=pcal[p].get("stat"), p=pcal[p].get("p"))
                                      for p in NAMES if p in pcal})
        ax = axs[i, 0]
        for pname in NAMES:
            lit = [v for v in visits if v["plane"] == pname and v["lit"]]
            dark = [v for v in visits if v["plane"] == pname and not v["lit"]]
            ax.scatter([v["min"] for v in dark], [v["score"] for v in dark], s=12, c="#bbbbbb",
                       label="dark cell" if pname == "red" else None, zorder=1)
            ax.scatter([v["min"] for v in lit], [v["score"] for v in lit], s=18, c=COL[pname],
                       label=f"{pname} lit ({lit[0]['freq']:g} Hz)" if lit else None, zorder=2)
            # running median of the lit cells, 8 visits wide
            if len(lit) >= 8:
                lit.sort(key=lambda v: v["min"])
                m = np.array([v["min"] for v in lit]); s = np.array([v["score"] for v in lit])
                med = [np.median(s[max(0, j - 4):j + 4]) for j in range(len(s))]
                ax.plot(m, med, c=COL[pname], lw=2, zorder=3)
        dark_all = [v["score"] for v in visits if not v["lit"]]
        ax.axhline(np.median(dark_all), c="#888888", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("minutes since the first flicker (white calibration)")
        ax.set_ylabel("visit response (f + 2f over the floor)")
        oz = res[name]["calib_oz_dprime"]
        pc = res[name]["plane_calib"]
        ax.set_title(f"{name}: white calibration Oz d' {oz:.1f}; colour calibrations "
                     + ", ".join(f"{p} +{pc[p]['stat']:.1f}" for p in NAMES if p in pc), loc="left")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax2 = axs[i, 1]
        hm = [h["minute"] for h in health]
        ax2.plot(hm, [h["rms"] for h in health], c="#333333", label="Oz 1-40 Hz RMS (uV)")
        ax2.plot(hm, [h["mains"] for h in health], c="#eb6834", label="Oz 60 Hz mains (uV)")
        ax2.set_xlabel("minutes since the first flicker")
        ax2.set_yscale("log")
        ax2.set_xlim(-1, max(hm) + 1 if hm else 1)
        ax2.set_title("Oz electrode health per minute", loc="left")
        ax2.legend(fontsize=8)
    fig.suptitle("State of Flash: the SSVEP response of each lit cell through the session, "
                 "with the Oz contact's own health", fontsize=12)
    fig.tight_layout()
    os.makedirs("runs/state_of_flash", exist_ok=True)
    p = "runs/state_of_flash/fig_ssvep_fatigue.png"
    fig.savefig(p, dpi=110)
    json.dump(res, open("runs/state_of_flash/ssvep_fatigue.json", "w"), indent=1)
    print(p)
    # the numbers: lit-cell median per plane per quarter of the session, dark median
    for name, r in res.items():
        vs = r["visits"]
        print(f"\n{name}: dark-cell median {np.median([v['score'] for v in vs if not v['lit']]):.2f}")
        for pname in NAMES:
            lit = sorted([v for v in vs if v["plane"] == pname and v["lit"]], key=lambda v: v["min"])
            if not lit:
                continue
            q = np.array_split(lit, 4)
            print(f"  {pname:5s} lit median by quarter of its scan: "
                  + "  ".join(f"min {qq[0]['min']:.0f}-{qq[-1]['min']:.0f}: {np.median([v['score'] for v in qq]):.2f}"
                              for qq in q if len(qq)))
        oz = r["oz"]
        print(f"  Oz RMS first 5 min {np.median([h['rms'] for h in oz[:5]]):.1f} uV, last 5 min "
              f"{np.median([h['rms'] for h in oz[-5:]]):.1f} uV; mains first {np.median([h['mains'] for h in oz[:5]]):.2f}, "
              f"last {np.median([h['mains'] for h in oz[-5:]]):.2f} uV")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/vr_planes9", "runs/vr_planes10"])
