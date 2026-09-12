"""Signal-quality and statistics audit of eyecam live sessions.

Reproducible: python analysis/quality_stats.py [--runs xr_live6,xr_live2,webgate]

Sections (all numbers printed; nothing is hard-coded from memory):
  A. Per-channel impedance proxies over calib+scan: 60 Hz line/floor ratio,
     60 Hz / broadband ratio, std, robust sigma, HF sigma, rail fraction.
  B. Time-resolved contact quality (10 s windows) per stage + trend.
  C. Stimulus check: measured flicker Hz per ON block and per scan cell.
  D. Calibration re-score (mirrors run_session.score_calibration), exact
     permutation p over the ON/OFF block labels, bootstrap CI, Mann-Whitney.
     CONTROL ARM: same with EEG circularly shifted by 1/3 of the recording.
  E. Per-block 15 Hz power time series + 2 s sliding 15 Hz SNR trace.
  F. PSD around 15 Hz, ON vs OFF, per channel (0.5 Hz bins): is there a peak?
  G. Reconstruction r (baseline weights) real vs control; split-half
     reliability of the cell scores; SNR gain needed for r >= 0.6.
  H. Proposed live-badge thresholds evaluated on every session.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.signal import welch
from scipy.stats import mannwhitneyu, spearmanr

EYECAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EYECAM)
import config  # noqa: E402
import reconstruct  # noqa: E402

RAIL_HI = 1450.0  # observed ADC ceiling in these logs (Muse via MuseLog bridge)
RAIL_LO = 0.0
WIN_S = 10.0


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_run(run):
    d = os.path.join(EYECAM, "runs", run)
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(d)
    _, crows = reconstruct.read_csv(os.path.join(d, "cursor_log.csv"))
    cur_lum = np.array([float(r[3]) for r in crows])
    cur_fl = np.array([int(float(r[4])) for r in crows])
    _, krows = reconstruct.read_csv(os.path.join(d, "calib_log.csv"))
    cal_t = np.array([float(r[0]) for r in krows])
    cal_fl = np.array([int(float(r[4])) for r in krows])
    with open(os.path.join(d, "calibration.json")) as f:
        calib = json.load(f)
    blocks = [(k, float(t0), float(t1)) for k, t0, t1 in calib["blocks"]]
    target = np.load(os.path.join(d, "target.npy"))
    fs = reconstruct.infer_fs(eeg_t)
    # The recorder may still be appending (xr_live6 grew while this was
    # written) and the post-session tail is rail-saturated once the headset
    # is off. Everything except the time-resolved stage table uses the
    # session window [first sample, scan end + 2 s] so numbers are stable
    # and the control shift never drags post-session garbage into the
    # calibration window.
    keep = eeg_t <= cur_t[-1] + 2.0
    return dict(run=run, dir=d, t_full=eeg_t, x_full=data,
                t=eeg_t[keep], x=data[keep], names=names, fs=fs,
                cur_t=cur_t, gx=gx, gy=gy, cur_lum=cur_lum, cur_fl=cur_fl,
                cal_t=cal_t, cal_fl=cal_fl, calib=calib, blocks=blocks,
                target=target)


def circ_shift(x):
    """CONTROL ARM: circular shift by 1/3 of the (session-window) recording."""
    n = x.shape[0]
    return np.roll(x, n // 3, axis=0)


# ----------------------------------------------------------------------------
# A. impedance proxies
# ----------------------------------------------------------------------------
def psd_05(seg, fs):
    """Welch PSD with 0.5 Hz bins (2 s Hann segments)."""
    nper = int(round(2 * fs))
    if len(seg) < nper:
        return None, None
    return welch(seg, fs=fs, nperseg=nper)


def band_mean(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(p[m].mean()) if m.any() else np.nan


def line60(f, p):
    """60 Hz line / local floor (median PSD 45-55 & 65-75 Hz), and
    60 Hz line / mean broadband PSD (14-50 Hz, the decoder's noise band)."""
    p60 = band_mean(f, p, 59.5, 60.5)
    floor = np.median(np.concatenate((p[(f >= 45) & (f <= 55)],
                                      p[(f >= 65) & (f <= 75)])))
    bb = band_mean(f, p, 14, 50)
    return p60 / max(floor, 1e-20), p60 / max(bb, 1e-20)


def proxies(seg, fs):
    seg = np.asarray(seg, float)
    out = dict(n=len(seg))
    if len(seg) < 2 * fs:
        return None
    out["std"] = float(seg.std())
    out["rsig"] = float(reconstruct.robust_sigma(seg))
    out["hfsig"] = float(reconstruct.hf_sigma(seg))
    out["rail"] = float(((seg <= RAIL_LO) | (seg >= RAIL_HI)).mean())
    f, p = psd_05(seg, fs)
    out["r60_floor"], out["r60_bb"] = line60(f, p)
    out["bb14_50"] = band_mean(f, p, 14, 50)      # uV^2/Hz
    out["p15"] = band_mean(f, p, 14.5, 15.5)
    return out


# ----------------------------------------------------------------------------
# D. calibration scoring (mirrors run_session.score_calibration)
# ----------------------------------------------------------------------------
def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12


def dprime_stats(on, off):
    on, off = np.asarray(on, float), np.asarray(off, float)
    if len(on) < 3 or len(off) < 3:
        return dict(dprime=0.0, ratio=1.0, rank=0)
    spread = np.sqrt(0.5 * (mad_sigma(on) ** 2 + mad_sigma(off) ** 2))
    spread = max(spread, 0.05 * abs(float(np.median(on))))
    return dict(dprime=float((np.median(on) - np.median(off)) / max(spread, 1e-12)),
                ratio=float(np.median(on) / max(np.median(off), 1e-12)),
                rank=int((on > off.max()).sum()))


def block_scores(t, x, fs, blocks, stim_freq=None, per_block_freq=None):
    """Per-channel ON/OFF decoder scores per block (cell_score, 1 s discard)."""
    n_ch = x.shape[1]
    out = []
    for c in range(n_ch):
        sigma = reconstruct.hf_sigma(x[:, c])
        sc = {"on": [], "off": []}
        for bi, (kind, t0, t1) in enumerate(blocks):
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            f0 = stim_freq
            if per_block_freq is not None and kind == "on":
                f0 = per_block_freq[bi]
            s = reconstruct.cell_score(x[i0:i1, c], fs, sigma, f0)
            sc[kind].append(float(s) if not np.isnan(s) else np.nan)
        out.append(sc)
    return out


def perm_test(on, off, stat="dprime"):
    """Exact permutation over the block labels: all C(n, n_on) relabelings."""
    on = np.asarray(on, float)
    off = np.asarray(off, float)
    on = on[~np.isnan(on)]
    off = off[~np.isnan(off)]
    allv = np.concatenate((on, off))
    n, k = len(allv), len(on)
    obs = dprime_stats(on, off)[stat]
    vals = []
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        vals.append(dprime_stats(allv[m], allv[~m])[stat])
    vals = np.array(vals)
    p_one = float((vals >= obs - 1e-12).mean())
    p_two = float((np.abs(vals) >= abs(obs) - 1e-12).mean())
    return obs, p_one, p_two, len(vals)


def bootstrap_dprime(on, off, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    on = np.asarray(on, float)
    off = np.asarray(off, float)
    on = on[~np.isnan(on)]
    off = off[~np.isnan(off)]
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = dprime_stats(rng.choice(on, len(on)), rng.choice(off, len(off)))["dprime"]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


# ----------------------------------------------------------------------------
# F. PSD ON vs OFF
# ----------------------------------------------------------------------------
def block_psd(t, x, fs, blocks, kind, c, nper_s=2.0):
    """Average Welch PSD over all blocks of `kind` (post 1 s discard)."""
    nper = int(round(nper_s * fs))
    acc, n = None, 0
    for k, t0, t1 in blocks:
        if k != kind:
            continue
        i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
        i1 = np.searchsorted(t, t1)
        seg = x[i0:i1, c]
        if len(seg) < nper:
            continue
        # same artifact clipping as the decoder
        base = np.median(seg)
        dev = seg - base
        sig = max(reconstruct.robust_sigma(dev), reconstruct.hf_sigma(x[:, c]))
        seg = base + np.clip(dev, -config.ARTIFACT_Z * sig, config.ARTIFACT_Z * sig)
        f, p = welch(seg, fs=fs, nperseg=nper)
        acc = p if acc is None else acc + p
        n += 1
    return f, acc / max(n, 1), n


def peak_snr(f, p, f0=15.0, half=0.5, nb=(1.5, 3.5)):
    """P(f0 +- half) / mean P in the flanks [f0-nb1, f0-nb0] U [f0+nb0, f0+nb1]."""
    sig = band_mean(f, p, f0 - half, f0 + half)
    flank = np.concatenate((p[(f >= f0 - nb[1]) & (f <= f0 - nb[0])],
                            p[(f >= f0 + nb[0]) & (f <= f0 + nb[1])]))
    return sig / max(flank.mean(), 1e-20)


def is_local_max(f, p, f0=15.0, span=3.0):
    m = (f >= f0 - span) & (f <= f0 + span)
    i = np.argmin(np.abs(f - f0))
    return bool(p[i] >= p[m].max() - 1e-20)


# ----------------------------------------------------------------------------
# G. reconstruction + split-half reliability
# ----------------------------------------------------------------------------
def visits(cur_t, gx, gy):
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    return starts, ends


def split_half_grids(t, x, fs, cur_t, gx, gy, weights):
    """Two grids from the first / second half of every cell dwell (same
    scoring as reconstruct.reconstruct, same vertical kernel)."""
    n_ch = x.shape[1]
    ch_sigma = [reconstruct.hf_sigma(x[:, c]) for c in range(n_ch)]
    W, H = gx.max() + 1, gy.max() + 1
    grids = [np.full((H, W), np.nan), np.full((H, W), np.nan)]
    nper = int(round(fs))
    for s, e in zip(*visits(cur_t, gx, gy)):
        i0 = np.searchsorted(t, cur_t[s])
        i1 = np.searchsorted(t, cur_t[e - 1])
        mid = (i0 + i1) // 2
        for h, (a, b) in enumerate(((i0, mid), (mid, i1))):
            if b - a < nper:
                continue
            num = den = 0.0
            for c in range(n_ch):
                if weights[c] <= 0:
                    continue
                sc = reconstruct.cell_score(x[a:b, c], fs, ch_sigma[c])
                if not np.isnan(sc):
                    num += weights[c] * sc
                    den += weights[c]
            if den > 0:
                grids[h][gy[s], gx[s]] = num / den
    k = np.array(config.VERTICAL_KERNEL)
    out = []
    for g in grids:
        g = np.where(np.isnan(g), np.nanmedian(g), g)
        out.append(reconstruct.convolve1d(g, k / k.sum(), axis=0, mode="nearest"))
    return out


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def spearman_brown(rho_half, k):
    return k * rho_half / (1 + (k - 1) * rho_half)


# ----------------------------------------------------------------------------
# main per-run audit
# ----------------------------------------------------------------------------
def audit(run, results):
    R = load_run(run)
    t, x, names, fs = R["t"], R["x"], R["names"], R["fs"]
    tf, xf = R["t_full"], R["x_full"]
    blocks = R["blocks"]
    cal0, cal1 = blocks[0][1], blocks[-1][2]
    scan0, scan1 = R["cur_t"][0], R["cur_t"][-1]
    n_on = sum(1 for b in blocks if b[0] == "on")
    n_off = len(blocks) - n_on
    print("\n" + "=" * 78)
    print(f"RUN {run}: fs={fs:.2f} Hz, file n={len(tf)} ({tf[-1]-tf[0]:.1f} s), "
          f"session-window n={len(t)} ({t[-1]-t[0]:.1f} s); "
          f"calib {cal0-tf[0]:.0f}..{cal1-tf[0]:.0f} s ({n_on} ON/{n_off} OFF), "
          f"scan {scan0-tf[0]:.0f}..{scan1-tf[0]:.0f} s, post-scan tail in file {tf[-1]-scan1:.0f} s")
    res = dict(fs=fs, n_file=len(tf), n_session=len(t))

    # ---- A. impedance proxies over calib+scan -------------------------------
    print("\n[A] impedance proxies over calib+scan window (per channel)")
    print(f"  r60_floor = P(60Hz)/median P(45-55,65-75); r60_bb = P(60Hz)/mean P(14-50)")
    print(f"  {'ch':>5} {'std':>8} {'rsig':>7} {'hfsig':>7} {'rail%':>6} "
          f"{'r60_floor':>9} {'r60_bb':>8} {'bb14-50':>9} {'P15':>9}")
    m = (t >= cal0) & (t <= scan1)
    A = {}
    for c, nm in enumerate(names):
        pr = proxies(x[m, c], fs)
        A[nm] = pr
        print(f"  {nm:>5} {pr['std']:8.1f} {pr['rsig']:7.1f} {pr['hfsig']:7.2f} "
              f"{100*pr['rail']:6.2f} {pr['r60_floor']:9.1f} {pr['r60_bb']:8.2f} "
              f"{pr['bb14_50']:9.3f} {pr['p15']:9.3f}")
    res["proxies"] = A

    # ---- B. time-resolved contact quality -----------------------------------
    print(f"\n[B] time-resolved contact quality, {WIN_S:.0f} s windows, per stage "
          f"(median over windows in stage; 'post' = whole file tail after the scan)")
    stages = [("pre", tf[0], cal0), ("calib", cal0, cal1)]
    third = (scan1 - scan0) / 3
    stages += [("scan1/3", scan0, scan0 + third), ("scan2/3", scan0 + third, scan0 + 2 * third),
               ("scan3/3", scan0 + 2 * third, scan1), ("post", scan1, tf[-1])]
    edges = np.arange(tf[0], tf[-1], WIN_S)
    wins = []
    for w0 in edges:
        mm = (tf >= w0) & (tf < w0 + WIN_S)
        if mm.sum() < 2 * fs:
            continue
        row = dict(t=w0 - tf[0], abs_t=w0)
        for c, nm in enumerate(names):
            row[nm] = proxies(xf[mm, c], fs)
        wins.append(row)
    B = {}
    for key, label in (("r60_floor", "60Hz/floor"), ("hfsig", "HF sigma uV"),
                       ("rail", "rail frac"), ("std", "std uV")):
        print(f"  {label:>12} " + " ".join(f"{s[0]:>9}" for s in stages))
        for nm in names:
            vals = []
            for sname, s0, s1 in stages:
                v = [w[nm][key] for w in wins if s0 <= w["abs_t"] < s1 and w[nm]]
                vals.append(float(np.median(v)) if v else np.nan)
            B[f"{nm}:{key}"] = dict(zip([s[0] for s in stages], vals))
            print(f"  {nm:>12} " + " ".join(f"{v:9.3f}" for v in vals))
    # trend inside calib+scan
    print("  trend within calib+scan (Spearman rho vs time; slope per minute):")
    for key in ("r60_floor", "hfsig"):
        for nm in names:
            ww = [w for w in wins if cal0 <= w["abs_t"] < scan1 and w[nm]]
            tt = np.array([w["t"] for w in ww]) / 60.0
            vv = np.array([w[nm][key] for w in ww])
            rho, p = spearmanr(tt, vv)
            slope = np.polyfit(tt, vv, 1)[0]
            B[f"{nm}:{key}:trend"] = dict(rho=float(rho), p=float(p), slope_per_min=float(slope))
            print(f"    {nm:>5} {key:>9}: rho={rho:+.2f} (p={p:.3f}) slope={slope:+.3f}/min "
                  f"first={vv[0]:.2f} last={vv[-1]:.2f}")
    res["stages"] = B
    # when did the first rail hit occur?
    railrow = ((xf <= RAIL_LO) | (xf >= RAIL_HI)).any(axis=1)
    idx = np.flatnonzero(railrow)
    if len(idx):
        first_rail = tf[idx[0]] - tf[0]
        rail_in_session = railrow[(tf >= cal0) & (tf <= scan1)].mean()
        print(f"  first rail sample at +{first_rail:.0f} s (scan ends +{scan1-tf[0]:.0f} s); "
              f"rail rows within calib+scan: {100*rail_in_session:.3f}% ; "
              f"whole file {100*railrow.mean():.2f}%")
        res["rail_first_s"] = float(first_rail)
        res["rail_in_session_frac"] = float(rail_in_session)
    else:
        print("  no rail samples anywhere in the file")
        res["rail_in_session_frac"] = 0.0

    # ---- C. stimulus check -------------------------------------------------
    print("\n[C] measured flicker frequency (toggle count / 2 / duration)")
    fmeas = []
    for bi, (k, t0, t1) in enumerate(blocks):
        mm = (R["cal_t"] >= t0) & (R["cal_t"] < t1)
        fl = R["cal_fl"][mm]
        tt = R["cal_t"][mm]
        tog = int((np.diff(fl) != 0).sum())
        fr = len(tt) / max(tt[-1] - tt[0], 1e-9)
        fhz = tog / 2.0 / (t1 - t0)
        fmeas.append(fhz if k == "on" else None)
        if k == "on":
            print(f"  block {bi:2d} ON : {fhz:6.2f} Hz  (page frame rate {fr:5.1f} fps)")
    on_f = [f for f in fmeas if f is not None]
    res["flicker_hz_on_blocks"] = on_f
    print(f"  ON blocks: mean {np.mean(on_f):.2f} Hz, min {min(on_f):.2f}, max {max(on_f):.2f}")
    # scan cells
    st, en = visits(R["cur_t"], R["gx"], R["gy"])
    cell_f = []
    for s, e in zip(st, en):
        fl = R["cur_fl"][s:e]
        tog = int((np.diff(fl) != 0).sum())
        dur = R["cur_t"][e - 1] - R["cur_t"][s]
        if tog >= 4 and dur > 1:
            cell_f.append(tog / 2.0 / dur)
    cell_f = np.array(cell_f)
    if len(cell_f):
        print(f"  scan: {len(cell_f)} flickering cell visits, flicker Hz median {np.median(cell_f):.2f} "
              f"[min {cell_f.min():.2f}, max {cell_f.max():.2f}], "
              f"{100*(np.abs(cell_f-15)>0.4).mean():.0f}% of cells off by >0.4 Hz")
        res["flicker_hz_scan"] = dict(n=int(len(cell_f)), median=float(np.median(cell_f)),
                                      min=float(cell_f.min()), max=float(cell_f.max()),
                                      frac_off_by_0p4=float((np.abs(cell_f - 15) > 0.4).mean()))

    # ---- D. calibration statistics: real vs control ------------------------
    print("\n[D] calibration contrast: decoder score per block, d' (run_session formula),")
    print("    exact permutation p over block labels, bootstrap 95% CI, Mann-Whitney U")
    xs = circ_shift(x)
    D = {}
    for arm, xx in (("real", x), ("control", xs)):
        bs = block_scores(t, xx, fs, blocks)
        D[arm] = {}
        print(f"  --- {arm} arm ---")
        print(f"  {'ch':>5} {'dprime':>7} {'ratio':>6} {'rank':>4} {'p_perm1':>8} {'p_perm2':>8} "
              f"{'nperm':>5} {'boot95lo':>8} {'boot95hi':>8} {'P(d<=0)':>7} {'MWU_p':>7} {'json_d':>7}")
        for c, nm in enumerate(names):
            on, off = bs[c]["on"], bs[c]["off"]
            obs, p1, p2, nperm = perm_test(on, off)
            lo, hi, pneg = bootstrap_dprime(on, off)
            st_ = dprime_stats(on, off)
            onv = np.array(on)[~np.isnan(on)]
            offv = np.array(off)[~np.isnan(off)]
            mwu = mannwhitneyu(onv, offv, alternative="greater").pvalue
            jd = R["calib"]["channels"][nm]["dprime"] if arm == "real" else float("nan")
            D[arm][nm] = dict(dprime=obs, ratio=st_["ratio"], rank=st_["rank"], p_perm_one=p1,
                              p_perm_two=p2, n_perm=nperm, boot_lo=lo, boot_hi=hi,
                              p_boot_le0=pneg, mwu_p=float(mwu), on=on, off=off)
            print(f"  {nm:>5} {obs:7.2f} {st_['ratio']:6.2f} {st_['rank']:4d} {p1:8.3f} {p2:8.3f} "
                  f"{nperm:5d} {lo:8.2f} {hi:8.2f} {pneg:7.2f} {mwu:7.3f} {jd:7.2f}")
        # family-wise: best-of-4 permutation (max d' over channels under relabeling)
        allsc = [(np.array(bs[c]["on"]), np.array(bs[c]["off"])) for c in range(len(names))]
        nb = len(blocks)
        maxd = []
        for idx in itertools.combinations(range(nb), n_on):
            msk = np.zeros(nb, bool)
            msk[list(idx)] = True
            best = -np.inf
            for on, off in allsc:
                allv = np.concatenate((on, off))  # on blocks first, then off
                best = max(best, dprime_stats(allv[msk], allv[~msk])["dprime"])
            maxd.append(best)
        maxd = np.array(maxd)
        obs_best = max(D[arm][nm]["dprime"] for nm in names)
        p_fw = float((maxd >= obs_best - 1e-12).mean())
        D[arm]["best_of_4_p"] = p_fw
        D[arm]["best_dprime"] = float(obs_best)
        print(f"  best-of-4-channels d'={obs_best:.2f}: family-wise permutation p={p_fw:.3f} "
              f"(null 95th pct of max d' = {np.percentile(maxd, 95):.2f})")
    res["calib"] = D

    # f-tracked variant: score ON blocks at their measured flicker Hz
    bs_tr = block_scores(t, x, fs, blocks, per_block_freq=fmeas)
    bs_tr_c = block_scores(t, xs, fs, blocks, per_block_freq=fmeas)
    print("  f-tracked (ON blocks scored at measured flicker Hz): d' real / control")
    res["calib_ftracked"] = {}
    for c, nm in enumerate(names):
        dr = dprime_stats(bs_tr[c]["on"], bs_tr[c]["off"])["dprime"]
        dc = dprime_stats(bs_tr_c[c]["on"], bs_tr_c[c]["off"])["dprime"]
        _, p1, _, _ = perm_test(bs_tr[c]["on"], bs_tr[c]["off"])
        res["calib_ftracked"][nm] = dict(dprime=dr, control=dc, p_perm_one=p1)
        print(f"    {nm:>5}: d'={dr:6.2f} (p={p1:.3f})  control d'={dc:6.2f}")

    # ---- E. per-block 15 Hz power time series ------------------------------
    print("\n[E] per-block 15 Hz: decoder score | absolute P15 (uV^2/Hz, 0.5 Hz bins) | peakSNR15")
    E = {}
    hdr = "  " + f"{'ch':>5} " + " ".join(f"{b[0][:2]}{i:<3d}" for i, b in enumerate(blocks))
    print("  decoder score (P15+P30)/(P14-50 - inband):")
    print(hdr)
    bs = block_scores(t, x, fs, blocks)
    for c, nm in enumerate(names):
        row = []
        io = iff = 0
        for k, _, _ in blocks:
            if k == "on":
                row.append(bs[c]["on"][io]); io += 1
            else:
                row.append(bs[c]["off"][iff]); iff += 1
        E[f"{nm}:score"] = row
        print(f"  {nm:>5} " + " ".join(f"{v:5.3f}" for v in row))
    print("  peak SNR at 15 Hz = P(14.5-15.5)/mean P(flanks 11.5-13.5 & 16.5-18.5):")
    print(hdr)
    for c, nm in enumerate(names):
        row = []
        for k, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            f, p = psd_05(x[i0:i1, c], fs)
            row.append(peak_snr(f, p))
        E[f"{nm}:peaksnr"] = row
        on = [v for v, b in zip(row, blocks) if b[0] == "on"]
        off = [v for v, b in zip(row, blocks) if b[0] == "off"]
        _, p1, _, _ = perm_test(on, off)
        print(f"  {nm:>5} " + " ".join(f"{v:5.2f}" for v in row) +
              f"   med ON {np.median(on):.2f} OFF {np.median(off):.2f} perm p={p1:.3f}")
        E[f"{nm}:peaksnr_perm_p"] = p1
    # sliding trace (2 s windows, 1 s hop) across calibration for best json channel
    best = R["calib"]["best"]
    cb = names.index(best)
    print(f"  2 s sliding peakSNR15 trace for {best} across calibration "
          f"(| marks block boundaries, ON blocks in [ ]):")
    trace = []
    line = "  "
    nper2 = int(round(2 * fs))
    for bi, (k, t0, t1) in enumerate(blocks):
        vals = []
        w0 = t0
        while w0 + 2 < t1:
            i0 = np.searchsorted(t, w0)
            i1 = min(i0 + nper2, len(t))  # sample-index window: arrival stamps clump
            f, p = psd_05(x[i0:i1, cb], fs)
            vals.append(peak_snr(f, p) if p is not None else np.nan)
            w0 += 1
        trace.append(vals)
        s = " ".join(f"{v:.1f}" for v in vals)
        line += (f"[{s}]" if k == "on" else f" {s} ") + "|"
    print(line)
    res["e"] = E

    # ---- F. PSD ON vs OFF -------------------------------------------------
    print("\n[F] mean PSD (uV^2/Hz, 0.5 Hz bins, decoder clipping) ON vs OFF around 15 & 30 Hz")
    fgrid = [12, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 18, 29, 29.5, 30, 30.5, 31]
    F = {}
    print("  " + f"{'ch':>5} {'arm':>4} " + " ".join(f"{f:>6.1f}" for f in fgrid) +
          "   snr15 lmax15  snr30")
    for c, nm in enumerate(names):
        F[nm] = {}
        for k in ("on", "off"):
            f, p, nblk = block_psd(t, x, fs, blocks, k, c)
            vals = [float(p[np.argmin(np.abs(f - g))]) for g in fgrid]
            s15, s30 = peak_snr(f, p, 15.0), peak_snr(f, p, 30.0)
            lm = is_local_max(f, p, 15.0)
            F[nm][k] = dict(psd=dict(zip(map(str, fgrid), vals)), snr15=s15, snr30=s30,
                            localmax15=lm)
            print(f"  {nm:>5} {k:>4} " + " ".join(f"{v:6.3f}" for v in vals) +
                  f"   {s15:5.2f}  {str(lm):>5}  {s30:5.2f}")
        on15 = F[nm]["on"]["psd"]["15"]; off15 = F[nm]["off"]["psd"]["15"]
        onfl = np.mean([F[nm]["on"]["psd"][k] for k in ("13", "13.5", "16.5", "17")])
        offl = np.mean([F[nm]["off"]["psd"][k] for k in ("13", "13.5", "16.5", "17")])
        F[nm]["on_off_15"] = on15 / max(off15, 1e-20)
        F[nm]["on_off_flank"] = onfl / max(offl, 1e-20)
        print(f"  {nm:>5} ON/OFF at 15 Hz = {on15/max(off15,1e-20):.2f}, ON/OFF at flanks = "
              f"{onfl/max(offl,1e-20):.2f}  -> stimulus-specific ratio "
              f"{(on15/max(off15,1e-20))/(onfl/max(offl,1e-20)):.2f}")
    # control arm PSD SNR for the best channel
    f, p, _ = block_psd(t, xs, fs, blocks, "on", cb)
    f2, p2, _ = block_psd(t, xs, fs, blocks, "off", cb)
    print(f"  control (shifted) {best}: snr15 ON {peak_snr(f, p):.2f} OFF {peak_snr(f2, p2):.2f}")
    res["psd"] = F

    # ---- G. reconstruction real vs control, reliability, SNR gain ----------
    print("\n[G] reconstruction r vs target (baseline calibration weights) real vs control")
    weights = reconstruct.weights_from_calibration(os.path.join(R["dir"], "calibration.json"), names)
    G = {}
    for arm, xx in (("real", x), ("control", xs)):
        g = reconstruct.reconstruct(t, xx, R["cur_t"], R["gx"], R["gy"], fs=fs, weights=weights)
        G[arm] = corr(g, R["target"])
    print(f"  weights {dict(zip(names, np.round(weights, 3)))}")
    print(f"  r_real = {G['real']:.3f}   r_control = {G['control']:.3f}")
    # per-channel single-electrode r (real / control)
    print("  single-channel r (real / control):", end="")
    G["single"] = {}
    for c, nm in enumerate(names):
        w1 = np.zeros(len(names)); w1[c] = 1
        gr = reconstruct.reconstruct(t, x, R["cur_t"], R["gx"], R["gy"], fs=fs, weights=w1)
        gc = reconstruct.reconstruct(t, xs, R["cur_t"], R["gx"], R["gy"], fs=fs, weights=w1)
        G["single"][nm] = (corr(gr, R["target"]), corr(gc, R["target"]))
        print(f"  {nm} {G['single'][nm][0]:+.3f}/{G['single'][nm][1]:+.3f}", end="")
    print()
    # split-half reliability (needs >= 2 s dwell so each half holds a 1 s PSD segment)
    dwell = float(np.median(np.diff(R["cur_t"][visits(R["cur_t"], R["gx"], R["gy"])[0]])))
    r_obs = G["real"]
    tgt = R["target"].ravel()
    p_bright = float((tgt > 0.5).mean())
    q = p_bright * (1 - p_bright)
    print(f"  target bright fraction (>0.5) = {p_bright:.3f}, p(1-p) = {q:.3f}; "
          f"median cell dwell {dwell:.2f} s")
    if dwell >= 2.0:
        h1, h2 = split_half_grids(t, x, fs, R["cur_t"], R["gx"], R["gy"], weights)
        rho_hh = corr(h1, h2)
        rel_full = spearman_brown(rho_hh, 2) if rho_hh > 0 else float("nan")
        r1, r2 = corr(h1, R["target"]), corr(h2, R["target"])
        h1c, h2c = split_half_grids(t, xs, fs, R["cur_t"], R["gx"], R["gy"], weights)
        rho_hh_c = corr(h1c, h2c)
        G["split_half"] = dict(rho_half=rho_hh, rho_half_control=rho_hh_c, rel_full=rel_full,
                               r_half1=r1, r_half2=r2)
        print(f"  split-half (first/second half of every dwell): rho_hh = {rho_hh:+.3f} "
              f"(control {rho_hh_c:+.3f}); Spearman-Brown full-dwell reliability = {rel_full:.3f}; "
              f"half-grid r vs target = {r1:+.3f}, {r2:+.3f}")
        if rho_hh_c >= rho_hh:
            print("  CAVEAT: control split-half reliability >= real -> within-dwell agreement is "
                  "noise autocorrelation (non-stationary artifact), not repeatable stimulus "
                  "response; route 1 is not interpretable here")
        if rho_hh > 0 and r_obs > 0 and rho_hh > rho_hh_c:
            r_ceiling = min(r_obs / np.sqrt(rel_full), 1.0)
            rel_needed = (0.6 / r_ceiling) ** 2
            if rel_needed < 1:
                k_needed = rel_needed * (1 - rel_full) / (rel_full * (1 - rel_needed))
                print(f"  route 1 (reliability): r_ceiling = r_obs/sqrt(rel) = {r_ceiling:.3f}; "
                      f"reliability needed for r>=0.6 = {rel_needed:.3f}; Spearman-Brown "
                      f"SNR-power multiplier k = {k_needed:.1f}x (= {np.sqrt(k_needed):.1f}x in d')")
                G["route1"] = dict(r_ceiling=float(r_ceiling), rel_needed=float(rel_needed),
                                   k_power=float(k_needed))
            else:
                print(f"  route 1: r_ceiling {r_ceiling:.3f} < 0.6 -> no amount of averaging "
                      f"reaches 0.6 (systematic error, not noise)")
                G["route1"] = dict(r_ceiling=float(r_ceiling), rel_needed=float(rel_needed))
        else:
            print("  route 1: not usable (rho_hh <= 0, r_obs <= 0, or control >= real)")
            G["route1"] = None
    else:
        print(f"  split-half: dwell {dwell:.2f} s < 2 s -> halves shorter than one 1 s PSD "
              f"segment; split-half not computable for this run")
        G["split_half"] = None
        G["route1"] = None
    # route 2: block d' -> cell d' -> r via two-class model, ceiling from webgate
    d_best = D["real"][best]["dprime"]
    d_ctrl = D["control"][best]["dprime"]
    on_len = np.mean([t1 - t0 - config.CALIB_DISCARD_S for k, t0, t1 in blocks if k == "on"])
    ceil = 0.8  # phantom ceiling (webgate r ~0.76-0.82 at d' ~18); stated assumption
    d_need = (0.6 / ceil) / np.sqrt(q * (1 - (0.6 / ceil) ** 2))
    G["route2"] = {}
    for label, d_blk in (("real d'", d_best), ("excess d' (real - control)", d_best - d_ctrl)):
        d_cell = d_blk * np.sqrt(dwell / on_len) if d_blk > 0 else 0.0
        r_pred = ceil * d_cell * np.sqrt(q) / np.sqrt(d_cell ** 2 * q + 1)
        gain = d_need / d_cell if d_cell > 0 else float("inf")
        print(f"  route 2 (two-class model, ceiling {ceil}) from {label} of {best}: block d'={d_blk:.2f} "
              f"at {on_len:.1f} s -> cell d' at {dwell:.1f} s dwell = {d_cell:.2f} -> predicted "
              f"r = {r_pred:.3f} (observed {r_obs:.3f}); cell d' needed for r>=0.6 = {d_need:.2f} "
              f"-> gain {gain:.1f}x in d' ({gain**2:.1f}x in SNR power)")
        G["route2"][label] = dict(d_block=float(d_blk), d_cell=float(d_cell), r_pred=float(r_pred),
                                  d_cell_needed=float(d_need),
                                  gain_dprime=float(gain) if np.isfinite(gain) else None)
    res["recon"] = G
    results[run] = res
    return res


def thresholds(results):
    print("\n" + "=" * 78)
    print("[H] proposed live-badge thresholds (evaluated on calib+scan window of each run)")
    # Thresholds are set from the three sessions below: 'hard' rules catch a
    # floating electrode (xr_live2 ears), 'soft' rules separate the two
    # xr_live6 forehead channels (the only ones with a non-negative d') from
    # its ear channels. The phantom has no mains/impedance physics, so it can
    # only prove the rules do not reject clean data.
    rules = [
        ("HARD rail fraction (0 or 1450) per channel", "rail", 0.01, "lt"),
        ("HARD HF sigma (uV) per channel", "hfsig", 100.0, "lt"),
        ("SOFT HF sigma (uV) per channel", "hfsig", 40.0, "lt"),
        ("SOFT 60 Hz line/floor ratio per channel", "r60_floor", 1.0e4, "lt"),
        ("SOFT std (uV) per channel", "std", 60.0, "lt"),
    ]
    runs = list(results)
    for label, key, thr, op in rules:
        print(f"  {label}: threshold {'<' if op=='lt' else '>'} {thr}")
        for run in runs:
            pr = results[run]["proxies"]
            vals = {nm: pr[nm][key] for nm in pr}
            ok = {nm: (v < thr if op == "lt" else v > thr) for nm, v in vals.items()}
            print(f"    {run:>9}: " + "  ".join(f"{nm}={v:.3g}{'' if ok[nm] else '!'}" for nm, v in vals.items())
                  + f"   -> {sum(ok.values())}/4 channels pass")
    print("  stimulus rule: measured flicker within 15 +- 0.3 Hz in every ON block")
    for run in runs:
        f = results[run]["flicker_hz_on_blocks"]
        bad = [round(v, 2) for v in f if abs(v - 15) > 0.3]
        print(f"    {run:>9}: ON-block Hz {[round(v,2) for v in f]} -> {'PASS' if not bad else 'FAIL '+str(bad)}")
    print("  calibration rule: best-of-4 family-wise permutation p < 0.05 AND real d' > control d' + 1")
    for run in runs:
        D = results[run]["calib"]
        print(f"    {run:>9}: best d' real {D['real']['best_dprime']:.2f} (fw p={D['real']['best_of_4_p']:.3f}) "
              f"vs control best {D['control']['best_dprime']:.2f} (fw p={D['control']['best_of_4_p']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="xr_live6,xr_live2,webgate")
    ap.add_argument("--out", default=os.path.join(EYECAM, "analysis", "quality_stats_results.json"))
    args = ap.parse_args()
    results = {}
    for run in args.runs.split(","):
        audit(run.strip(), results)
    thresholds(results)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(args.out, "w") as f:
        json.dump(clean(results), f, indent=1)
    print(f"\nresults json -> {args.out}")


if __name__ == "__main__":
    main()
