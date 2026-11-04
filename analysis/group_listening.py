"""Seven listeners, one hydrophone: the 17 October 2026 group sessions at Love Quay. A hydrophone
in Lake Ontario fed a speaker on the pier while swimmers jumped in, someone kicked the water and
someone hit it with a stick; up to seven Muse headbands were recorded at once by one MuseLog
phone, so every headband is on the same clock.

Per headband and session (CLOCK_TIME on the phone's clock):
  * head motion: gyroscope magnitude, 1-s bins, robust z (median/MAD of the session);
  * frontal transients: AF7/AF8 band-passed 0.5-8 Hz, 1-s rms, robust z (blinks and startle);
  * contact: share of the session each electrode's horseshoe (HSI) was 1 or 2.
Group measures per session:
  * shared moments: bins where at least 2 headbands exceed z 2.5 together (fixed before looking),
    against 1,000 surrogates that shift every headband's series independently by a random lag
    (circular, at least 5 s): the observed count over the surrogate distribution;
  * inter-subject correlation (ISC): mean pairwise Pearson r of the z series, against the same
    surrogates;
  * the timing of shared moments: count, intervals, and the moments themselves.

    python analysis/group_listening.py runs/oct17_group docs/paper/hydrophone2027/figures
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 6.5, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5})
BIN = 1.0
FS = 256.0
Z = 2.5
RNG = np.random.default_rng(0)


def rz(x):
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / mad


def headband(path, t0):
    d = pd.read_csv(path, low_memory=False)
    t = (pd.to_datetime(d["CLOCK_TIME"]) - t0).dt.total_seconds().to_numpy()
    out = {}
    g = d[["GYRO_X", "GYRO_Y", "GYRO_Z"]].apply(pd.to_numeric, errors="coerce")
    ok = g.notna().all(1).to_numpy()
    out["gyro"] = (t[ok], np.linalg.norm(g[ok].to_numpy(), axis=1))
    e = d[["AF7_RAW", "AF8_RAW"]].apply(pd.to_numeric, errors="coerce")
    ok = e.notna().all(1).to_numpy()
    te, x = t[ok], e[ok].to_numpy().mean(1)
    if len(x) > 4 * FS:
        x = sosfiltfilt(butter(2, [0.5, 8], "band", fs=FS, output="sos"), x - np.median(x))
    out["frontal"] = (te, np.abs(x))
    hsi = {}
    for c in ("TP9", "AF7", "AF8", "TP10"):
        h = pd.to_numeric(d[f"{c}_CONNECTION_STRENGTH(HSI)"], errors="coerce").dropna()
        hsi[c] = round(float((h <= 2).mean()), 2) if len(h) else None
    out["hsi_good"] = hsi
    out["span"] = (float(np.nanmin(t)), float(np.nanmax(t)))
    return out


def binned(t, v, edges):
    idx = np.digitize(t, edges) - 1
    s = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        m = idx == k
        if m.any():
            s[k] = np.sqrt(np.mean(v[m] ** 2))
    return s


def group_stats(Zm, need):
    """Zm: headbands x bins (NaN allowed). Shared-moment count and ISC, with shift surrogates."""
    def shared(M):
        return np.nansum(M > Z, axis=0) >= need

    def isc(M):
        rs = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                ok = np.isfinite(M[i]) & np.isfinite(M[j])
                if ok.sum() > 40:
                    rs.append(np.corrcoef(M[i, ok], M[j, ok])[0, 1])
        return float(np.mean(rs)) if rs else np.nan
    obs_n, obs_isc = int(shared(Zm).sum()), isc(Zm)
    n = Zm.shape[1]
    min_shift = int(5 / BIN)
    sur_n, sur_isc = [], []
    for _ in range(1000):
        S = np.array([np.roll(r, RNG.integers(min_shift, max(min_shift + 1, n - min_shift))) for r in Zm])
        sur_n.append(int(shared(S).sum()))
        if len(sur_isc) < 300:
            sur_isc.append(isc(S))
    sur_n, sur_isc = np.array(sur_n), np.array(sur_isc)
    return dict(shared_moments=obs_n, shared_surrogate_mean=round(float(sur_n.mean()), 2),
                shared_surrogate_95=float(np.percentile(sur_n, 95)),
                shared_p=round(float((np.sum(sur_n >= obs_n) + 1) / (len(sur_n) + 1)), 4),
                isc=round(obs_isc, 3), isc_surrogate_95=round(float(np.nanpercentile(sur_isc, 95)), 3),
                isc_p=round(float((np.sum(sur_isc >= obs_isc) + 1) / (len(sur_isc) + 1)), 4)), shared(Zm)


def main():
    src, out = sys.argv[1], sys.argv[2]
    files = [f for f in sorted(glob.glob(os.path.join(src, "muse_session_20261017_*_Muse*.csv"))) if os.path.getsize(f) > 100000 and "154110" not in f]
    sessions = defaultdict(list)
    for f in files:
        sessions[re.search(r"20261017_(\d{6})_", f).group(1)].append(f)
    res = dict(sessions=[])
    fig, axes = plt.subplots(len(sessions), 1, figsize=(7.0, 0.95 * len(sessions) + 0.3), sharex=False)
    for row, (stamp, fs_) in enumerate(sorted(sessions.items())):
        t0 = pd.Timestamp(f"2026-10-17 {stamp[:2]}:{stamp[2:4]}:{stamp[4:]}")
        hbs = {re.search(r"(Muse\d)", f).group(1): headband(f, t0) for f in fs_}
        end = min(max(h["span"][1] for h in hbs.values()), 240)
        edges = np.arange(0, end + BIN, BIN)
        need = 2
        entry = dict(session=f"15:41 17 Oct".replace("15:41", f"{stamp[:2]}:{stamp[2:4]}"), stamp=stamp,
                     headbands=sorted(hbs), seconds=round(end, 1), need=need,
                     contact={k: v["hsi_good"] for k, v in sorted(hbs.items())})
        tc = (edges[:-1] + edges[1:]) / 2
        for sig in ("gyro", "frontal"):
            M = []
            used = []
            for k in sorted(hbs):
                t, v = hbs[k][sig]
                if len(v) < 50:
                    continue
                used.append(k)
                s = binned(t, v, edges)
                s[(tc < hbs[k]["span"][0]) | (tc > hbs[k]["span"][1])] = np.nan
                M.append(rz(s))
            M = np.array(M)
            st, sh = group_stats(M, need)
            times = tc[sh]
            moments = []
            for x in times:
                if not moments or x - moments[-1] > 1.0:
                    moments.append(round(float(x), 2))
            st["moments_s"] = moments
            st["headbands_used"] = used
            st["moment_intervals_s"] = [round(b - a, 2) for a, b in zip(moments, moments[1:])]
            entry[sig] = st
            if sig == "frontal":
                ax = axes[row] if len(sessions) > 1 else axes
                frac = np.nansum(M > Z, axis=0) / np.sum(np.isfinite(M), axis=0).clip(1)
                ax.fill_between(tc, 0, frac, color="#1f4e79", lw=0, step="mid", alpha=0.85)
                for x in moments:
                    ax.axvline(x, color="#c0392b", lw=0.5, alpha=0.7)
                ax.set_ylim(0, 1); ax.set_xlim(0, end)
                ax.set_ylabel(f"{stamp[:2]}:{stamp[2:4]}\n{len(used)} listeners", fontsize=5.5)
                ax.text(end * 0.995, 0.92, f"shared {st['shared_moments']} vs {st['shared_surrogate_mean']:.1f} by chance (p {st['shared_p']}); ISC {st['isc']} (p {st['isc_p']})",
                        ha="right", va="top", fontsize=5.2)
                ax.spines[["top", "right"]].set_visible(False)
        res["sessions"].append(entry)
        g, fr = entry["gyro"], entry["frontal"]
        print(f"{stamp} {len(hbs)} heads {end:.0f}s | head motion: shared {g['shared_moments']} vs {g['shared_surrogate_mean']} (95th {g['shared_surrogate_95']}, p {g['shared_p']}), ISC {g['isc']} (95th {g['isc_surrogate_95']}, p {g['isc_p']}), moments {g['moments_s']} | frontal: shared {fr['shared_moments']} vs {fr['shared_surrogate_mean']} p {fr['shared_p']}, ISC {fr['isc']} p {fr['isc_p']}")
    (axes[-1] if len(sessions) > 1 else axes).set_xlabel("seconds into the session")
    fig.suptitle("share of listeners with a frontal EEG transient (AF7/AF8 0.5-8 Hz, z > 2.5), 1-s bins; red: shared moments", fontsize=6)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(os.path.join(out, "fig_group.pdf"))
    fig.savefig(os.path.join(out, "fig_group_preview.png"), dpi=200)
    json.dump(res, open(os.path.join(out, "group_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
