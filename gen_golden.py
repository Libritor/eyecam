"""Generate golden parity vectors for the Unity C# DSP port (EyeCamXR).

Every expected value is produced by the *actual* laptop-stack code
(reconstruct.py, run_session.score_calibration, scipy.signal.welch), so the
Unity EditMode tests prove bin-exact parity, not agreement between two ports.

Output: JsonUtility-friendly JSON (flat arrays, no dicts, NaN encoded as
NAN_SENTINEL) written into the Unity project's Tests/Golden folder.

Run:  python gen_golden.py
"""

import json
import os
import shutil

import numpy as np
from scipy.signal import welch

import config
import reconstruct
import run_session
import simulate
import targets

OUT_DIR = (r"MuseLogMR"
           r"\Assets\EyeCamXR\Tests\Golden")
def b64(x):
    """Exact float64 transport: base64 of little-endian doubles. Survives any
    JSON parser bit-for-bit, NaN included (JsonUtility numeric parsing does
    not guarantee last-ulp round-trips)."""
    import base64
    arr = np.asarray(x, dtype="<f8").ravel()
    return base64.b64encode(arr.tobytes()).decode("ascii")


def enc(x):
    """Doubles (scalars and arrays) -> base64; ints/strings/bools stay plain."""
    if isinstance(x, (float, np.floating)):
        return b64([float(x)])
    if isinstance(x, np.ndarray):
        return b64(x)
    if isinstance(x, (list, tuple)):
        if x and all(isinstance(v, (float, np.floating)) for v in x):
            return b64(x)
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def dump(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(enc(obj), f, allow_nan=False)
    print(f"  {name}: {os.path.getsize(path)} bytes")


def sig_with_ssvep(rng, n, fs, snr, f0=15.0):
    t = np.arange(n) / fs
    noise = rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * 10.0 * t)
    return noise + snr * (np.sin(2 * np.pi * f0 * t)
                          + 0.4 * np.sin(2 * np.pi * 2 * f0 * t))


def welch_cases(rng):
    cases = []
    for name, fs, n in [("fs256_pow2", 256.0, 1280),
                        ("fs64_decimated", 64.0, 640),
                        ("fs251_odd_nonpow2", 251.0, 1255),
                        ("fs250p7_rounds_251", 250.7, 1300)]:
        nperseg = int(round(fs))
        x = sig_with_ssvep(rng, n, fs, 0.8)
        freqs, psd = welch(x, fs=fs, nperseg=nperseg)
        cases.append(dict(name=name, fs=fs, nperseg=nperseg,
                          x=x.tolist(), psd=psd.tolist(),
                          df=float(freqs[1] - freqs[0])))
    dump("welch_cases.json", dict(cases=cases))


def ssvep_cases(rng):
    cases = []
    specs = [("strong_15hz", 256.0, 1280, 1.5),
             ("noise_only", 256.0, 1280, 0.0),
             ("weak", 256.0, 1280, 0.15),
             ("fs64", 64.0, 640, 0.8),
             ("fs251", 251.0, 1255, 0.8)]
    for name, fs, n, snr in specs:
        x = sig_with_ssvep(rng, n, fs, snr)
        cases.append(dict(name=name, fs=fs, x=x.tolist(),
                          score=reconstruct.ssvep_score(x, fs)))
    # too-short segment -> NaN
    x = sig_with_ssvep(rng, 100, 256.0, 1.0)
    cases.append(dict(name="too_short", fs=256.0, x=x.tolist(),
                      score=reconstruct.ssvep_score(x, 256.0)))
    dump("ssvep_cases.json", dict(cases=cases))


def cell_cases(rng):
    cases = []
    fs = 256.0
    base = sig_with_ssvep(rng, 1280, fs, 0.8) + 800.0
    floor_sig = reconstruct.hf_sigma(base)

    clean = base.copy()
    cases.append(dict(name="clean", fs=fs, sigmaFloor=floor_sig,
                      x=clean.tolist(),
                      score=reconstruct.cell_score(clean, fs, floor_sig)))

    blink = base.copy()
    blink[:400] += 400.0  # >20% of samples far out -> drop
    cases.append(dict(name="blink_drop", fs=fs, sigmaFloor=floor_sig,
                      x=blink.tolist(),
                      score=reconstruct.cell_score(blink, fs, floor_sig)))

    spikes = base.copy()
    spikes[::50] += 60.0  # ~2% clipped but kept
    cases.append(dict(name="clipped_kept", fs=fs, sigmaFloor=floor_sig,
                      x=spikes.tolist(),
                      score=reconstruct.cell_score(spikes, fs, floor_sig)))

    short = base[:6]
    cases.append(dict(name="too_short", fs=fs, sigmaFloor=floor_sig,
                      x=short.tolist(),
                      score=reconstruct.cell_score(short, fs, floor_sig)))
    dump("cell_cases.json", dict(cases=cases))


def inferfs_cases():
    cases = []
    t = np.arange(0, 60, 1 / 256.0)
    cases.append(dict(name="uniform256", t=t.tolist(),
                      fs=reconstruct.infer_fs(t)))
    t2 = np.concatenate([t[:7680], t[7680:] + 25.0])  # 25 s dead-air gap
    cases.append(dict(name="gap25s", t=t2.tolist(),
                      fs=reconstruct.infer_fs(t2)))
    # burst-clumped: 4 samples share ~the same stamp, bursts at 64 Hz
    nb = 960
    tb = np.repeat(np.arange(nb) / 64.0, 4) + np.tile(
        [0.0, 1e-4, 2e-4, 3e-4], nb)
    cases.append(dict(name="burst_clumped", t=tb.tolist(),
                      fs=reconstruct.infer_fs(tb)))
    dump("inferfs_cases.json", dict(cases=cases))


def calib_case(rng):
    """Exercise the actual run_session.score_calibration via a temp session."""
    fs = 256
    names = ["TP9", "AF7", "AF8", "TP10"]
    snrs = [0.8, 0.15, 0.1, 0.9]
    blocks = []
    t_all, rows = [], []
    t = 100.0  # arbitrary epoch
    for b in range(config.CALIB_BLOCKS):
        for kind, dur, lum in (("on", config.CALIB_ON_S, 1.0),
                               ("off", config.CALIB_OFF_S, 0.0)):
            n = int(dur * fs)
            tt = t + np.arange(n) / fs
            blocks.append((kind, float(tt[0]), float(tt[-1])))
            for c in range(4):
                pass  # filled below
            t_all.append(tt)
            rows.append(lum)
            t = float(tt[-1]) + 1 / fs
    t_cat = np.concatenate(t_all)
    n_tot = len(t_cat)
    data = np.zeros((n_tot, 4))
    tsec = t_cat - t_cat[0]
    for c in range(4):
        noise = rng.standard_normal(n_tot) + 0.3 * np.sin(2 * np.pi * 10 * tsec)
        env = np.zeros(n_tot)
        i = 0
        for tt, lum in zip(t_all, rows):
            env[i:i + len(tt)] = lum
            i += len(tt)
        data[:, c] = 800 + noise + snrs[c] * env * (
            np.sin(2 * np.pi * 15 * tsec) + 0.4 * np.sin(2 * np.pi * 30 * tsec))

    tmp = os.path.join("runs", "golden_calib")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "eeg.csv"), "w") as f:
        f.write("time," + ",".join(names) + "\n")
        for i in range(n_tot):
            f.write(f"{t_cat[i]:.6f}," +
                    ",".join(f"{v:.4f}" for v in data[i]) + "\n")
    calib = run_session.score_calibration(tmp, blocks)

    # reload the CSV so C# sees the exact rounded values Python scored
    header, rrows = reconstruct.read_csv(os.path.join(tmp, "eeg.csv"))
    t_r = [float(r[0]) for r in rrows]
    ch_flat = []
    for c in range(4):
        ch_flat.extend(float(r[1 + c]) for r in rrows)

    dump("calib_case.json", dict(
        t=t_r, nCh=4, chFlat=ch_flat, names=names,
        blockIsOn=[1 if k == "on" else 0 for k, _, _ in blocks],
        blockT0=[t0 for _, t0, _ in blocks],
        blockT1=[t1 for _, _, t1 in blocks],
        rankMin=config.CALIB_RANK_MIN,
        expFs=calib["fs"],
        expDprime=[calib["channels"][n]["dprime"] for n in names],
        expRatio=[calib["channels"][n]["ratio"] for n in names],
        expRank=[calib["channels"][n]["rank"] for n in names],
        expWeights=[calib["weights"][n] for n in names],
        expBest=calib["best"],
        expPassed=calib["passed"]))


def recon_case(rng):
    """Small end-to-end reconstruction with 2 channels + weights."""
    grid_w, grid_h, spc, fs = 8, 6, 0.6, 256
    target = targets.text_target("N", grid_w, grid_h)
    cursor_rows = simulate.raster_cursor_log(target, spc, fs)
    cur_t = np.array([r[0] for r in cursor_rows]) + 500.0
    gx = np.array([r[1] for r in cursor_rows])
    gy = np.array([r[2] for r in cursor_rows])
    n = len(cursor_rows)
    eeg_t = cur_t.copy()  # sample-aligned, simplest case

    data = np.zeros((n, 2))
    for c, snr in enumerate([1.0, 0.3]):
        eeg = simulate.synth_eeg(cursor_rows, fs, snr,
                                 np.random.default_rng(1000 + c))
        data[:, c] = eeg + 800.0
    weights = np.array([0.7, 0.3])

    grid = reconstruct.reconstruct(eeg_t, data, cur_t, gx, gy, weights=weights)
    norm = np.clip((grid - np.percentile(grid, 2))
                   / max(np.percentile(grid, 98) - np.percentile(grid, 2),
                         1e-12), 0, 1)
    r = float(np.corrcoef(target.ravel(), grid.ravel())[0, 1])

    ch_flat = []
    for c in range(2):
        ch_flat.extend(data[:, c].tolist())
    dump("recon_case.json", dict(
        eegT=eeg_t.tolist(), nCh=2, chFlat=ch_flat,
        weights=weights.tolist(),
        curT=cur_t.tolist(), gx=gx.astype(int).tolist(),
        gy=gy.astype(int).tolist(),
        gridW=grid_w, gridH=grid_h,
        expGridFlat=grid.ravel().tolist(),
        expNormFlat=norm.ravel().tolist(),
        targetFlat=target.ravel().tolist(),
        expR=r))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    print("golden vectors ->", OUT_DIR)
    welch_cases(rng)
    ssvep_cases(rng)
    cell_cases(rng)
    inferfs_cases()
    calib_case(rng)
    recon_case(rng)
    print("done")


if __name__ == "__main__":
    main()
