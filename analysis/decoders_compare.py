"""Post-hoc decoder comparison for one scan session.

For each per-cell scoring method the image is reconstructed from the same
EEG + cursor log with the same calibration weights, correlated with the
target, and compared with a shifted-EEG null (same data, wrong alignment).

methods: welch = 1 Hz-bin relative power (Mann 2019 / run_session)
         line  = exact-frequency line SNR (whole-cell Hann periodogram)
         cca   = canonical correlation with sin/cos references at f, 2f, 3f
                 over all weighted channels jointly (standard SSVEP CCA)
         act   = ACT chirplet scorer if analysis/act_chirplet exposes one

usage: python analysis/decoders_compare.py runs/<session> [--freq 12]
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import welch, periodogram
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, reconstruct  # noqa: E402


def cca_rho(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    ok = np.isfinite(X).all(0) & (X.std(0) > 1e-9)
    X = X[:, ok]
    if X.shape[1] == 0 or not np.isfinite(X).all():
        return np.nan
    try:
        qx, _ = np.linalg.qr(X)
        qy, _ = np.linalg.qr(Y)
        s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan
    return float(s[0]) if len(s) else np.nan


def refs(n, fs, f0, n_harm=3):
    t = np.arange(n) / fs
    cols = []
    for k in range(1, n_harm + 1):
        cols += [np.sin(2 * np.pi * k * f0 * t), np.cos(2 * np.pi * k * f0 * t)]
    return np.stack(cols, 1)


def clean(seg, sigma_floor):
    base = np.median(seg)
    dev = seg - base
    sig = max(reconstruct.robust_sigma(dev), sigma_floor)
    if (np.abs(dev) > config.ARTIFACT_Z * sig).mean() > config.ARTIFACT_DROP_FRAC:
        return None
    return base + np.clip(dev, -config.ARTIFACT_Z * sig, config.ARTIFACT_Z * sig)


def grid_for(method, eeg_t, data, cur_t, gx, gy, fs, w, f0, ch_sigma):
    n_ch = data.shape[1]
    gw, gh = gx.max() + 1, gy.max() + 1
    acc, cnt = np.zeros((gh, gw)), np.zeros((gh, gw))
    change = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(gx)]))
    nper = int(round(fs))
    for s, e in zip(starts, ends):
        i0, i1 = np.searchsorted(eeg_t, cur_t[s]), np.searchsorted(eeg_t, cur_t[e - 1])
        need = nper - (i1 - i0)
        if need > 0:
            i0 = max(0, i0 - need // 2)
            i1 = min(len(data), i0 + nper)
        if i1 - i0 < fs * 1.5:
            continue
        if method == "cca":
            cols, ws = [], []
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                seg = clean(data[i0:i1, c], ch_sigma[c])
                if seg is not None:
                    cols.append(seg * w[c])
                    ws.append(w[c])
            if not cols:
                continue
            X = np.stack(cols, 1)
            val = cca_rho(X, refs(X.shape[0], fs, f0))
        else:
            num = den = 0.0
            for c in range(n_ch):
                if w[c] <= 0:
                    continue
                seg = clean(data[i0:i1, c], ch_sigma[c])
                if seg is None:
                    continue
                if method == "welch":
                    v = reconstruct.ssvep_score(seg, fs, f0)
                elif method == "line":
                    v = reconstruct.line_snr(seg, fs, f0)
                elif method == "act":
                    v = ACT(seg, fs, f0)
                else:
                    raise ValueError(method)
                if v is not None and np.isfinite(v):
                    num += w[c] * v
                    den += w[c]
            if den <= 0:
                continue
            val = num / den
        acc[gy[s], gx[s]] += val
        cnt[gy[s], gx[s]] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if np.isnan(grid).all():
        return None
    grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)
    k = np.array(config.VERTICAL_KERNEL)
    return convolve1d(grid, k / k.sum(), axis=0, mode="nearest")


ACT = None
try:  # optional: an ACT chirplet per-segment scorer, if the module offers one
    from analysis import act_chirplet as _act  # noqa
    for name in ("act_score", "score_segment", "chirplet_score"):
        if hasattr(_act, name):
            ACT = getattr(_act, name)
            break
except Exception:
    ACT = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--freq", type=float, default=None,
                    help="stimulus frequency (default: calibration.json line "
                         "delivered, else measured from cursor_log edges)")
    ap.add_argument("--cursor-log", default="cursor_log.csv")
    ap.add_argument("--target", default="target.npy")
    ap.add_argument("--shifts", type=int, default=8)
    ap.add_argument("--methods", default="welch,line,cca" + (",act" if ACT else ""))
    a = ap.parse_args()
    sess = a.session
    eeg_t, data, names, cur_t, gx, gy = reconstruct.load_session(sess, cursor_log=a.cursor_log)
    keep = (gx >= 0) & (gy >= 0)          # calibration rows carry grid -1
    cur_t, gx, gy = cur_t[keep], gx[keep], gy[keep]
    if len(gx) == 0:
        sys.exit("no scan cells in " + a.cursor_log)
    fs = reconstruct.infer_fs(eeg_t)
    cal = json.load(open(os.path.join(sess, "calibration.json")))
    w = np.array([float(cal["weights"].get(n, 0.0)) for n in names])
    if w.sum() <= 0:
        w = np.ones(len(names))
    f0 = a.freq
    if f0 is None:
        f0 = cal.get("line", {}).get("delivered")
    if f0 is None:
        L = np.genfromtxt(os.path.join(sess, a.cursor_log), delimiter=",", skip_header=1)
        ed = np.where((L[1:, 4] == 1) & (L[:-1, 4] == 0))[0] + 1
        per = np.diff(L[ed, 5]); per = per[(per > 0.01) & (per < 1)]
        f0 = float(1 / np.median(per))
    target = np.load(os.path.join(sess, a.target))
    ch_sigma = [reconstruct.hf_sigma(data[:, c]) for c in range(data.shape[1])]
    span = eeg_t[-1] - eeg_t[0]
    shifts = [31.0 + k * max(7.0, (span - 62.0) / max(a.shifts, 1)) for k in range(a.shifts)]
    shifts = [s for s in shifts if s < span - 10]
    print(f"session {sess}: f0 {f0:.2f} Hz, fs {fs:.2f}, weights "
          + " ".join(f"{n}:{v:.2f}" for n, v in zip(names, w)))
    print(f"{'method':>7} {'r':>7} {'null mean':>9} {'null max':>8} {'z':>6}  verdict")
    out = {}
    for m in a.methods.split(","):
        g = grid_for(m, eeg_t, data, cur_t, gx, gy, fs, w, f0, ch_sigma)
        if g is None:
            print(f"{m:>7}  no cells"); continue
        r = float(np.corrcoef(target.ravel(), g.ravel())[0, 1])
        rn = []
        for s in shifts:
            d2 = np.roll(data, int(round(s * fs)), axis=0)
            gn = grid_for(m, eeg_t, d2, cur_t, gx, gy, fs, w, f0, ch_sigma)
            if gn is not None:
                rn.append(float(np.corrcoef(target.ravel(), gn.ravel())[0, 1]))
        mu, sd = (np.mean(rn), np.std(rn) + 1e-9) if rn else (np.nan, np.nan)
        z = (r - mu) / sd if rn else np.nan
        verdict = "REAL" if (r >= 0.6 and rn and r > max(rn) and z > 2) else \
                  ("above null" if (rn and r > max(rn)) else "chance")
        print(f"{m:>7} {r:7.3f} {mu:9.3f} {max(rn) if rn else float('nan'):8.3f} {z:6.2f}  {verdict}")
        reconstruct.save_image(g, os.path.join(sess, f"reconstruction_{m}.png"))
        out[m] = dict(r=r, null=rn, z=float(z), verdict=verdict)
    json.dump(dict(f0=f0, results=out), open(os.path.join(sess, "decoders_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
