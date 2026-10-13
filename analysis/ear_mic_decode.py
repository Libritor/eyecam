"""Ear as a Microphone, decoder.

Reconstructs the loudness envelope (1-8 Hz) of the clip the listener heard from
the Muse channels, writes the reconstruction back out as sound, and says whether
it beats chance. A Muse cannot give back the waveform; the slow envelope is what
scalp EEG follows, so that is what is reconstructed and re-sounded.

Method
  1. EEG clock. eeg.csv times are UDP arrival stamps; lock_in.eeg_clock rebuilds
     a sample-index clock per contiguous stream segment (as the SSVEP analyses).
  2. EEG. Per segment: median removal, 1-8 Hz zero-phase band-pass, clip at 4
     robust sigma (blinks), then interpolated onto one 64 Hz grid (NaN in gaps).
     Derived channel Cz-M = AUX - mean(TP9, TP10), the vertex-to-mastoid
     auditory montage, is reported for single-channel comparisons.
  3. Stimulus. Eight ERB-spaced sub-band envelopes of clip.wav (|Hilbert|^0.6,
     summed), 64 Hz, 1-8 Hz band-pass, z-scored.
  4. Decoder. Ridge regression from the EEG at lags -125..+500 ms (all channels
     at once) to the envelope. Leave-one-repetition-out: every repetition is
     predicted by a model that never saw it; the ridge strength is chosen by an
     inner leave-one-out on the training repetitions only.
  5. Scores. Pearson r per held-out repetition (single trial) and r of the
     average of the held-out predictions (the brain envelope of the clip). Null
     for both: every circular shift of the envelope by 3 s or more. Control
     arm: the whole analysis with the EEG rolled by half the recording
     (alignment broken, statistics kept); it must come out at chance.
  6. Audio. brain_envelope_noise.wav (the averaged reconstruction modulating
     white noise), true_envelope_noise.wav (the same for the real envelope, to
     compare like with like) and brain_envelope_carrier.wav (the clip's own fine
     structure under the brain envelope). The z-scored reconstruction is mapped
     to loudness with only the true envelope's mean and sd.

    python analysis/ear_mic_decode.py runs/ear_mic1
    python analysis/ear_mic_decode.py runs/ear_mic1 --channels TP9,TP10,AUX
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from lock_in import eeg_clock  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FS_D = 64                       # decoder rate
BAND = (1.0, 8.0)               # envelope-following band
LAG_MIN_S, LAG_MAX_S = -0.125, 0.5
LAGS = np.arange(int(round(LAG_MIN_S * FS_D)), int(round(LAG_MAX_S * FS_D)) + 1)
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
NULL_MIN_SHIFT_S = 3.0
RAW5 = ["TP9", "AF7", "AF8", "TP10", "AUX"]


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


# ---------------------------------------------------------------- stimulus
def erb_edges(lo, hi, n):
    e = lambda f: 21.4 * np.log10(4.37e-3 * f + 1)            # noqa: E731
    ie = lambda v: (10 ** (v / 21.4) - 1) / 4.37e-3            # noqa: E731
    return ie(np.linspace(e(lo), e(hi), n + 1))


def song_envelope(x, fs, n_bands=8):
    """Positive loudness envelope of x at FS_D (sub-band |Hilbert|^0.6 summed, 20 Hz low-passed)."""
    env = np.zeros(len(x))
    for lo, hi in zip(*(lambda e: (e[:-1], e[1:]))(erb_edges(100.0, min(8000.0, 0.45 * fs), n_bands))):
        y = sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), x)
        env += np.abs(hilbert(y)) ** 0.6
    env = sosfiltfilt(butter(2, 20.0, "lowpass", fs=fs, output="sos"), env)
    t = np.arange(len(x)) / fs
    grid = np.arange(int(len(x) / fs * FS_D)) / FS_D
    return np.interp(grid, t, env)


def band_env(env_pos):
    return zscore(sosfiltfilt(butter(2, BAND, "bandpass", fs=FS_D, output="sos"), env_pos))


# ---------------------------------------------------------------- EEG
def load_eeg(session):
    df = pd.read_csv(os.path.join(session, "eeg.csv"))
    names = [c for c in df.columns if c != "time"]
    t = df["time"].to_numpy(float)
    X = df[names].to_numpy(float)
    t_hat, fs_hat, diag = eeg_clock(t)
    return t_hat, fs_hat, X, names, diag


def eeg_to_grid(t_hat, X, fs_hat):
    """Band-passed, clipped EEG on a global FS_D grid; NaN where the stream was absent."""
    n = len(t_hat)
    cuts = np.flatnonzero(np.diff(t_hat) > 0.5) + 1
    bounds = np.concatenate(([0], cuts, [n]))
    g0 = t_hat[0]
    grid = g0 + np.arange(int((t_hat[-1] - g0) * FS_D)) / FS_D
    Y = np.full((len(grid), X.shape[1]), np.nan)
    sos = butter(2, BAND, "bandpass", fs=fs_hat, output="sos")
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 4 * fs_hat:
            continue
        seg = X[s:e] - np.median(X[s:e], axis=0)
        seg = sosfiltfilt(sos, seg, axis=0)
        sd = 1.4826 * np.median(np.abs(seg - np.median(seg, axis=0)), axis=0) + 1e-9
        seg = np.clip(seg, -4 * sd, 4 * sd)
        ts = np.maximum.accumulate(t_hat[s:e])
        m = (grid >= ts[0]) & (grid <= ts[-1])
        for c in range(X.shape[1]):
            Y[m, c] = np.interp(grid[m], ts, seg[:, c])
    return grid, Y


def rep_segment(Y, grid, t_on, n_env):
    """EEG rows covering envelope samples 0..n_env-1 at every lag; None if missing."""
    i0 = int(round((t_on - grid[0]) * FS_D))
    a, b = i0 + LAGS[0], i0 + n_env + LAGS[-1]
    if a < 0 or b > len(Y):
        return None
    seg = Y[a:b]
    if np.isnan(seg).any():
        return None
    return (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-9)


def design(seg, n_env):
    """Rows t, columns (lag, channel): seg[t + lag]."""
    return np.concatenate([seg[L - LAGS[0]: L - LAGS[0] + n_env] for L in LAGS], axis=1)


# ---------------------------------------------------------------- decoder
def ridge_loo(Xs, y):
    """Leave-one-rep-out ridge, inner LOO for alpha. Returns held-out predictions, r per rep, alphas, full weights."""
    G = [x.T @ x for x in Xs]
    c = [x.T @ y for x in Xs]
    Gs, cs = sum(G), sum(c)
    n, d = len(Xs), Gs.shape[0]
    I = np.eye(d)

    def solve(Gm, cm, a):
        return np.linalg.solve(Gm + a * np.trace(Gm) / d * I, cm)

    preds, rs, chosen = [], [], []
    for i in range(n):
        Gi, ci = Gs - G[i], cs - c[i]
        best = (-np.inf, ALPHAS[0])
        for a in ALPHAS:
            rr = [pearson(Xs[j] @ solve(Gi - G[j], ci - c[j], a), y) for j in range(n) if j != i]
            if np.mean(rr) > best[0]:
                best = (np.mean(rr), a)
        w = solve(Gi, ci, best[1])
        p = Xs[i] @ w
        preds.append(p)
        rs.append(pearson(p, y))
        chosen.append(best[1])
    a_full = max(set(chosen), key=chosen.count)
    return np.array(preds), np.array(rs), chosen, solve(Gs, cs, a_full), a_full


def shift_null(preds, y, fn):
    """fn(preds, y_shifted) for every circular shift of y by >= NULL_MIN_SHIFT_S."""
    n = len(y)
    k = int(NULL_MIN_SHIFT_S * FS_D)
    return np.array([fn(preds, np.roll(y, s)) for s in range(k, n - k)])


def p_value(obs, null):
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def run_decoder(Y, grid, onsets, y, cols):
    n_env = len(y)
    segs = [(r, rep_segment(Y[:, cols], grid, t_on, n_env)) for r, t_on in onsets]
    used = [(r, s) for r, s in segs if s is not None]
    if len(used) < 4:
        return None
    Xs = [design(s, n_env) for _, s in used]
    preds, rs, chosen, w, a_full = ridge_loo(Xs, y)
    avg = preds.mean(axis=0)
    r_avg = pearson(avg, y)
    null_single = shift_null(preds, y, lambda P, ys: np.mean([pearson(p, ys) for p in P]))
    null_avg = shift_null(avg, y, pearson)
    return dict(reps_used=[r for r, _ in used], reps_dropped=[r for r, s in segs if s is None],
                r_single=rs.tolist(), r_single_mean=float(rs.mean()), r_single_sd=float(rs.std()),
                p_single=p_value(rs.mean(), null_single), null_single_95=float(np.percentile(null_single, 95)),
                r_avg=r_avg, p_avg=p_value(r_avg, null_avg), null_avg_95=float(np.percentile(null_avg, 95)),
                alpha=a_full, alphas_chosen=chosen, preds=preds, avg=avg, weights=w)


def evoked_lags(Y, grid, onsets, y, names, chans):
    """Model-free check: average EEG over reps, correlate with the envelope at each lag."""
    n_env = len(y)
    cols = [names.index(c) for c in chans]
    segs = [rep_segment(Y[:, cols], grid, t_on, n_env) for _, t_on in onsets]
    segs = [s for s in segs if s is not None]
    if not segs:
        return None
    ev = np.mean(segs, axis=0)
    out = {}
    for j, c in enumerate(chans):
        out[c] = [pearson(ev[L - LAGS[0]: L - LAGS[0] + n_env, j], y) for L in LAGS]
    return out


# ---------------------------------------------------------------- audio
def to_audio(env_z, mu, sd, fs_a, carrier=None, seed=0):
    e = np.maximum(mu + sd * env_z, 0.0)
    t64 = np.arange(len(e)) / FS_D
    ta = np.arange(int(len(e) / FS_D * fs_a)) / fs_a
    ea = np.interp(ta, t64, e)
    if carrier is None:
        carrier = np.random.default_rng(seed).standard_normal(len(ta))
    y = carrier[:len(ea)] * ea[:len(carrier)]
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def flat_carrier(x, fs):
    env = sosfiltfilt(butter(2, 8.0, "lowpass", fs=fs, output="sos"), np.abs(hilbert(x)))
    return x / (np.maximum(env, 0) + 0.05 * env.max())


# ---------------------------------------------------------------- figure
def figure(session, res, ctrl, single, ev, y, names_used, out_stem):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    t = np.arange(len(y)) / FS_D
    a = ax[0, 0]
    a.plot(t, y, color="0.35", lw=1.0, label="song envelope (1-8 Hz, z)")
    a.plot(t, zscore(res["avg"]), color="#c0392b", lw=1.0,
           label=f"brain envelope, mean of {len(res['reps_used'])} held-out reconstructions (r = {res['r_avg']:.2f})")
    a.set_xlabel("time in clip (s)")
    a.set_ylabel("z")
    a.set_xlim(0, t[-1])
    a.legend(loc="upper right", fontsize=8)
    a.set_title("a. the clip's loudness envelope and its reconstruction from EEG", fontsize=10, loc="left")

    a = ax[0, 1]
    reps = res["reps_used"]
    a.axhspan(-res["null_single_95"], res["null_single_95"], color="0.9", label="95% of circular-shift null (mean r)")
    a.plot(reps, res["r_single"], "o", color="#c0392b", ms=4, label=f"aligned, mean r = {res['r_single_mean']:.3f}, p = {res['p_single']:.3g}")
    if ctrl:
        a.plot(ctrl["reps_used"], ctrl["r_single"], "x", color="0.4", ms=4,
               label=f"control: EEG rolled half a period off every clip, mean r = {ctrl['r_single_mean']:.3f}, p = {ctrl['p_single']:.2g}")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("repetition")
    a.set_ylabel("single-trial r (held out)")
    a.legend(fontsize=7, loc="upper right")
    a.set_title("b. every repetition decoded by a model that never saw it", fontsize=10, loc="left")

    a = ax[1, 0]
    labels = list(single.keys())
    vals = [single[k]["r_avg"] for k in labels]
    ps = [single[k]["p_avg"] for k in labels]
    bars = a.bar(labels, vals, color=["#c0392b" if p < 0.05 else "0.6" for p in ps])
    for b_, p in zip(bars, ps):
        a.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01, f"p={p:.2g}", ha="center", fontsize=7)
    a.axhline(0, color="k", lw=0.5)
    a.set_ylabel("r of averaged reconstruction")
    a.set_title("c. one channel at a time (red = beats the shift null)", fontsize=10, loc="left")

    a = ax[1, 1]
    if ev:
        for c, rr in ev.items():
            a.plot(LAGS / FS_D * 1000, rr, lw=1.0, label=c)
        a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("EEG lag after the sound (ms)")
        a.set_ylabel("r, repetition-averaged EEG vs envelope")
        a.legend(fontsize=7, ncol=2)
    a.set_title("d. model-free: averaged EEG against the envelope by lag", fontsize=10, loc="left")
    fig.suptitle(f"Ear as a Microphone, {os.path.basename(os.path.normpath(session))}: decoder on {', '.join(names_used)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--channels", default="", help="comma list for the multichannel decoder (default: all of TP9,AF7,AF8,TP10,AUX present)")
    ap.add_argument("--no-control", action="store_true", help="skip the rolled-EEG control arm")
    ap.add_argument("--audio-fs", type=int, default=16000)
    args = ap.parse_args()
    S = args.session

    meta = json.load(open(os.path.join(S, "stimulus.json")))
    ev_df = pd.read_csv(os.path.join(S, "events.csv"))
    onsets = [(int(r.rep), float(r.t_on)) for r in ev_df.itertuples()]
    clip, fs_clip = sf.read(os.path.join(S, "clip.wav"), dtype="float64", always_2d=True)
    clip = clip.mean(axis=1)
    env_pos = song_envelope(clip, fs_clip)
    y = band_env(env_pos)
    n_env = len(y)

    t_hat, fs_hat, X, names, diag = load_eeg(S)
    grid, Y = eeg_to_grid(t_hat, X, fs_hat)
    if all(c in names for c in ("AUX", "TP9", "TP10")):
        czm = Y[:, names.index("AUX")] - 0.5 * (Y[:, names.index("TP9")] + Y[:, names.index("TP10")])
        Y = np.concatenate([Y, czm[:, None]], axis=1)
        names = names + ["Cz-M"]
    want = [c for c in (args.channels.split(",") if args.channels else RAW5) if c in names]
    cols = [names.index(c) for c in want]
    jitter = t_hat - np.asarray(pd.read_csv(os.path.join(S, "eeg.csv"), usecols=["time"])["time"], float)
    print(f"EEG: {len(t_hat)} rows, fs_hat {fs_hat:.3f} Hz, {diag.get('n_segments')} segment(s), "
          f"arrival jitter 95% {np.percentile(np.abs(jitter), 95) * 1000:.0f} ms; channels {names}")
    print(f"clip: {n_env / FS_D:.1f} s, {len(onsets)} repetitions logged; decoder channels {want}")

    res = run_decoder(Y, grid, onsets, y, cols)
    if res is None:
        raise SystemExit("fewer than 4 repetitions have complete EEG; nothing to decode")
    print(f"\nALIGNED   single-trial r = {res['r_single_mean']:.3f} +- {res['r_single_sd']:.3f} over {len(res['reps_used'])} reps "
          f"(null 95% {res['null_single_95']:.3f}, p = {res['p_single']:.3g})")
    print(f"          averaged reconstruction r = {res['r_avg']:.3f} (null 95% {res['null_avg_95']:.3f}, p = {res['p_avg']:.3g}); ridge alpha {res['alpha']}")
    if res["reps_dropped"]:
        print(f"          dropped (stream gap / outside recording): reps {res['reps_dropped']}")

    ctrl = None
    if not args.no_control:
        # Roll the EEG by about half the recording, rounded to an odd number of half
        # repetition periods, so every window lands half a period off any response.
        # The song's own bar structure can still give a small r at such a shift; the
        # circular-shift null contains exactly those shifts, so the control is one
        # draw from the null and must sit inside it, not at zero.
        period = float(np.median(np.diff([t for _, t in onsets]))) if len(onsets) > 1 else n_env / FS_D
        roll_s = (round(len(Y) / FS_D / 2 / period - 0.5) + 0.5) * period
        Yc = np.roll(Y, int(round(roll_s * FS_D)), axis=0)
        ctrl = run_decoder(Yc, grid, onsets, y, cols)
        if ctrl:
            print(f"CONTROL   EEG rolled by {roll_s:.1f} s ({roll_s / period:.1f} repetition periods): "
                  f"single-trial r = {ctrl['r_single_mean']:.3f} (p = {ctrl['p_single']:.2g}), "
                  f"averaged r = {ctrl['r_avg']:.3f} (p = {ctrl['p_avg']:.2g})")

    single = {}
    for c in [n for n in names if n in RAW5 + ["Cz-M"]]:
        r1 = run_decoder(Y, grid, onsets, y, [names.index(c)])
        if r1:
            single[c] = dict(r_single_mean=r1["r_single_mean"], p_single=r1["p_single"], r_avg=r1["r_avg"], p_avg=r1["p_avg"])
            print(f"  {c:5s} alone: single-trial r {r1['r_single_mean']:+.3f} (p {r1['p_single']:.2g}), averaged r {r1['r_avg']:+.3f} (p {r1['p_avg']:.2g})")
    ev = evoked_lags(Y, grid, onsets, y, names, list(single))
    ev_best = {}
    if ev:
        for c, rr in ev.items():
            k = int(np.argmax(np.abs(rr)))
            ev_best[c] = dict(lag_ms=float(LAGS[k] / FS_D * 1000), r=float(rr[k]))
        print("  model-free best lag: " + ", ".join(f"{c} {v['r']:+.2f} at {v['lag_ms']:.0f} ms" for c, v in ev_best.items()))

    # audio
    mu, sd = float(env_pos.mean()), float(env_pos.std())
    fa = args.audio_fs
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_clip), fa)
    clip_a = resample_poly(clip, fa // g, int(fs_clip) // g)
    sf.write(os.path.join(S, "brain_envelope_noise.wav"), to_audio(zscore(res["avg"]), mu, sd, fa), fa)
    sf.write(os.path.join(S, "true_envelope_noise.wav"), to_audio(y, mu, sd, fa), fa)
    sf.write(os.path.join(S, "brain_envelope_carrier.wav"), to_audio(zscore(res["avg"]), mu, sd, fa, carrier=flat_carrier(clip_a, fa)), fa)
    np.save(os.path.join(S, "brain_envelope_64hz.npy"), res["avg"])

    out = dict(session=S, decoder_channels=want, fs_hat=fs_hat, n_segments=diag.get("n_segments"),
               arrival_jitter_95_ms=float(np.percentile(np.abs(jitter), 95) * 1000),
               clip_s=n_env / FS_D, reps_logged=len(onsets), band_hz=BAND, lags_ms=[LAG_MIN_S * 1000, LAG_MAX_S * 1000],
               aligned={k: v for k, v in res.items() if k not in ("preds", "avg", "weights")},
               control={k: v for k, v in ctrl.items() if k not in ("preds", "avg", "weights")} if ctrl else None,
               single_channel=single, evoked_best_lag=ev_best,
               audio=dict(brain_noise="brain_envelope_noise.wav", true_noise="true_envelope_noise.wav",
                          brain_carrier="brain_envelope_carrier.wav",
                          mapping="loudness = max(mu + sd * z_reconstruction, 0) with mu, sd of the true positive envelope"),
               verdict=("reconstruction beats chance" if res["p_avg"] < 0.01 and res["p_single"] < 0.01 and (ctrl is None or ctrl["p_avg"] >= 0.01)
                        else "not above chance" if res["p_avg"] >= 0.01 or res["p_single"] >= 0.01
                        else "aligned beats chance BUT the rolled control does too (p < 0.01): suspect the pipeline"))
    json.dump(out, open(os.path.join(S, "ear_mic_results.json"), "w"), indent=1)
    figure(S, res, ctrl, single, ev, y, want, os.path.join(S, "ear_mic_figure"))
    print(f"\nverdict: {out['verdict']}")
    print(f"wrote ear_mic_results.json, ear_mic_figure.pdf/.png, brain_envelope_noise.wav, true_envelope_noise.wav, brain_envelope_carrier.wav in {S}")


if __name__ == "__main__":
    main()
