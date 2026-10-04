"""Smooth-flicker colour: R, G and B tagged at three close frequencies.

Stimulus (xr_stimulus.html, --mode smooth): every frame the brightness of each
colour channel follows a sine wave (sampled sinusoidal stimulation), so a tag
can be ANY frequency, not only refresh / (2 * whole frames). All three tags
then fit inside the band where this subject responds, instead of one tag at
12 Hz and two at weak frequencies (7.2 / 9 Hz at 72 fps).

The three channels flicker together in the cell. The strength of the EEG line
at a channel's tag is how much of that colour the cell holds, so mixtures
(yellow = red + green lines) and shades (a weaker line) need no extra coding.

Before the scan a white smooth-flicker sweep measures the response at several
frequencies. choose_tags() places three tags where the response is strong,
unevenly spaced so that no mixing product (2a - b, a + b - c) of the visual
system lands on another tag, clear of the alpha peak, and gives the strongest
tag to the dimmest colour (blue, then red, then green). If the sweep finds no
clear response the DEFAULT_TAGS are kept.

    python smoothcolor.py --session runs/<session>       # re-decode a session
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
from PIL import Image
from scipy.signal import periodogram

import config
from reconstruct import hf_sigma, infer_fs, load_session, robust_sigma

# ---------------------------------------------------------------- constants
# MULTIPLIER: colour balance. One factor per finished plane (R, G, B), applied
# after each plane is scaled to its own range. It shifts the balance of the
# picture only; it cannot make a weak plane less noisy. Leave at 1 unless a
# value has been settled over several sessions (--color-balance).
COLOR_BALANCE = (1.0, 1.0, 1.0)

# MULTIPLIER: brightness curve. shade = response ** SHADE_EXPONENT. The
# response grows less than in proportion to brightness, so shades come out too
# bright; an exponent above 1 pulls them back down. 1 = no correction. The one
# grey-level run so far (2026-10-02: black 0.09, half grey 0.42, white 0.53)
# puts half grey at 0.87 of the white response, which would need about 5; one
# run is not enough to make that the default (--shade-exponent).
SHADE_EXPONENT = 1.0

# A sine has pi/4 of the fundamental of an on/off flicker with the same
# brightness range (21 % less). It is the same for all three colours, so no
# multiplier restores it; (4/pi)^2 = 1.62 times the dwell does.
SINE_VS_SQUARE = np.pi / 4
DWELL_FACTOR = (4 / np.pi) ** 2

LUMA = (0.2126, 0.7152, 0.0722)     # relative brightness of R, G, B (Rec. 709)
DEFAULT_TAGS = (11.2, 13.2, 12.0)   # R, G, B in Hz when the sweep is not used
MIN_SEP = 0.8        # Hz between two tags
COLLIDE_TOL = 0.35   # Hz a mixing product must keep from every tag
ALPHA_GUARD = 0.6    # Hz a tag must keep from the alpha peak
SETTLE = 0.25        # s skipped at each cell onset
LINE_HALF = 0.15     # Hz half-width of a tag line
FLANK = (0.5, 2.0)   # Hz either side of a tag used as its noise reference
RANGE_FLOOR = 1.0    # a plane spanning less than this is noise: do not stretch it


# ---------------------------------------------------------------- tags
def mixing_products(tags):
    """Frequencies where the visual system's nonlinearity puts energy when
    the tags are shown together: 2a - b and a + b - c."""
    out = [2 * a - b for a in tags for b in tags if a != b]
    for i, c in enumerate(tags):
        rest = [t for j, t in enumerate(tags) if j != i]
        out += [a + b - c for a, b in combinations(rest, 2)]
    return out


def tags_ok(tags, alpha=None):
    """'' when the tags can be shown together, else the reason they cannot."""
    for a, b in combinations(tags, 2):
        if abs(a - b) < MIN_SEP - 1e-9:
            return f"{a:g} and {b:g} Hz are closer than {MIN_SEP} Hz"
    for p in mixing_products(tags):
        for t in tags:
            if abs(p - t) < COLLIDE_TOL - 1e-9:
                return f"a mixing product at {p:.2f} Hz falls on the {t:g} Hz tag"
    if alpha is not None:
        for t in tags:
            if abs(t - alpha) < ALPHA_GUARD - 1e-9:
                return f"{t:g} Hz is on the alpha peak ({alpha:.1f} Hz)"
    return ""


def choose_tags(freqs, strengths, alpha=None, step=0.1):
    """Three tags inside the swept band where the response (interpolated
    between the swept frequencies) is strong: the triple whose weakest tag is
    strongest. Returns ([f_R, f_G, f_B], [strength per tag]) or (None, reason).
    The strongest tag goes to the dimmest colour."""
    order = np.argsort(freqs)
    fr, st = np.asarray(freqs, float)[order], np.asarray(strengths, float)[order]
    if len(fr) < 2:
        return None, "the sweep needs at least two frequencies"
    cand = np.round(np.arange(fr[0], fr[-1] + step / 2, step), 3)
    s = np.interp(cand, fr, np.maximum(st, 0.0))
    best = None
    for trio in combinations(range(len(cand)), 3):
        tags = [float(cand[i]) for i in trio]
        if tags[2] - tags[0] < 2 * MIN_SEP or tags_ok(tags, alpha):
            continue
        key = (min(s[list(trio)]), sum(s[list(trio)]))
        if best is None or key > best[0]:
            best = (key, trio)
    if best is None:
        return None, "no three tags fit in the swept band"
    trio = sorted(best[1], key=lambda i: -s[i])          # strongest first
    out, strength = [0.0] * 3, [0.0] * 3
    for i, colour in zip(trio, np.argsort(LUMA)):        # dimmest colour first
        out[colour], strength[colour] = float(cand[i]), float(s[i])
    return out, strength


def alpha_peak(session_dir, blocks, channel=None):
    """Alpha peak (Hz) in the flicker-OFF blocks, or None when no peak stands
    out. channel = column index into the EEG channels (default: the last)."""
    E = np.genfromtxt(os.path.join(session_dir, "eeg.csv"), delimiter=",", skip_header=1)
    et, fs = E[:, 0], infer_fs(E[:, 0])
    x = E[:, 1:][:, -1 if channel is None else channel]
    acc, n = None, 0
    for kind, t0, t1 in blocks:
        seg = x[(et >= t0 + 1.0) & (et <= t1)]
        if kind != "off" or len(seg) < 3 * fs:
            continue
        seg = seg[:int(3 * fs)]
        f, p = periodogram(seg - seg.mean(), fs=fs, window="hann")
        acc, n = (p if acc is None else acc + p), n + 1
    if not n:
        return None
    band, ref = (f >= 8) & (f <= 13), (f >= 6) & (f <= 16)
    k = np.flatnonzero(band)[np.argmax(acc[band])]
    return float(f[k]) if acc[k] > 4 * np.median(acc[ref]) else None


def tags_from_sweep(res, alpha=None):
    """(tags [R, G, B], weights per channel, note) from score_sweep() output.
    The swept tags replace DEFAULT_TAGS only when the sweep found a response."""
    names = res.get("names", [])
    rows = sorted(res.get("results", {}).values(), key=lambda r: r["freq"])
    diff = np.zeros(len(names))
    for r in rows:
        diff += np.array([max(r["channels"][n]["diff"], 0.0) for n in names])
    weights = diff / diff.sum() if diff.sum() > 0 else np.ones(len(names)) / max(len(names), 1)
    if not res.get("passed"):
        return list(DEFAULT_TAGS), weights, "sweep found no clear response: default tags kept"
    tags, info = choose_tags([r["freq"] for r in rows], [r["stat"] for r in rows], alpha)
    if tags is None:
        return list(DEFAULT_TAGS), weights, f"{info}: default tags kept"
    return tags, weights, "tags from the sweep, strongest to blue: response " + \
        ", ".join(f"{c} {v:.2f}" for c, v in zip("RGB", info))


# ---------------------------------------------------------------- decoding
def tag_snr(seg, fs, f0, avoid=()):
    """Line at f0 over the power 0.5-2 Hz either side, with the other tags
    and the alpha peak (avoid) left out of the noise reference: a strong
    neighbouring colour, or alpha, would otherwise count as noise and pull
    this one down."""
    f, p = periodogram(seg - seg.mean(), fs=fs, window="hann", detrend="constant")
    h = max(LINE_HALF, (f[1] - f[0]) * 1.01)
    d = np.abs(f - f0)
    flank = (d >= FLANK[0]) & (d <= FLANK[1])
    for a in avoid:
        flank &= np.abs(f - a) > COLLIDE_TOL
    if not flank.any():
        return np.nan
    return p[d <= h].mean() / max(p[flank].mean(), 1e-12)


def clip_artifacts(seg, sigma_floor):
    """reconstruct.cell_score's artifact handling: None for a segment that is
    mostly blink / movement, else the segment with its outliers clipped."""
    base = np.median(seg)
    dev = seg - base
    lim = config.ARTIFACT_Z * max(robust_sigma(dev), sigma_floor)
    if (np.abs(dev) > lim).mean() > config.ARTIFACT_DROP_FRAC:
        return None
    return base + np.clip(dev, -lim, lim)


def visits(cur_t, gx, gy, spc):
    """(gx, gy, t0, t1) per cell visit. A stay much longer than spc is two
    passes meeting on one cell: it is cut into visits of spc."""
    cut = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0))
    out = []
    for s, e in zip(np.r_[0, cut + 1], np.r_[cut + 1, len(gx)]):
        if gx[s] < 0 or gy[s] < 0:
            continue                               # pause marker
        t0, t1 = cur_t[s], cur_t[e - 1]
        n = max(1, int(round((t1 - t0) / spc))) if spc else 1
        for k in range(n):
            out.append((int(gx[s]), int(gy[s]), t0 + k * (t1 - t0) / n,
                        t0 + (k + 1) * (t1 - t0) / n))
    return out


def decode(session_dir, tags, weights=None, spc=0.0, shift_s=0.0, channels="",
           alpha=None, log_name="smooth_log.csv"):
    """Response per cell and colour, (grid_h, grid_w, 3): the amplitude of the
    line at each tag, sqrt(SNR - 1), averaged over channels and visits.
    shift_s rolls the EEG against the scan: a chance-level control."""
    eeg_t, data, names, cur_t, gx, gy = load_session(session_dir, channels, log_name)
    fs = infer_fs(eeg_t)
    if shift_s:
        data = np.roll(data, int(round(shift_s * fs)), axis=0)
    w = np.ones(data.shape[1]) if weights is None else np.asarray(weights, float)
    sigma = [hf_sigma(data[:, c]) for c in range(data.shape[1])]
    gw, gh = gx.max() + 1, gy.max() + 1
    acc, cnt = np.zeros((gh, gw, 3)), np.zeros((gh, gw))
    for x, y, t0, t1 in visits(cur_t, gx, gy, spc):
        i0, i1 = np.searchsorted(eeg_t, [t0 + (SETTLE if t1 - t0 > 2 else 0), t1])
        if i1 - i0 < max(1.5, 0.6 * (t1 - t0)) * fs:
            continue                               # too short, or the stream dropped
        snr, den = np.zeros(3), 0.0
        for c in np.flatnonzero(w > 0):
            seg = clip_artifacts(data[i0:i1, c], sigma[c])
            if seg is None:
                continue
            v = [tag_snr(seg, fs, f0, [t for t in tags if t != f0]
                         + ([alpha] if alpha else [])) for f0 in tags]
            if np.all(np.isfinite(v)):
                snr += w[c] * np.array(v)
                den += w[c]
        if den > 0:
            acc[y, x] += snr / den
            cnt[y, x] += 1
    grid = np.sqrt(np.maximum(acc / np.maximum(cnt, 1)[:, :, None] - 1.0, 0.0))
    for k in range(3):                              # cells never scored
        plane = grid[:, :, k]
        plane[cnt == 0] = np.median(plane[cnt > 0]) if (cnt > 0).any() else 0.0
    return grid


def to_rgb(grid, balance=COLOR_BALANCE, exponent=SHADE_EXPONENT):
    """Response grid -> picture in [0, 1]: each plane scaled to its own range,
    then the brightness curve and the colour balance."""
    rgb = np.zeros_like(grid)
    for k in range(3):
        lo, hi = np.percentile(grid[:, :, k], [2, 98])
        plane = np.clip((grid[:, :, k] - lo) / max(hi - lo, RANGE_FLOOR), 0, 1)
        rgb[:, :, k] = np.clip(plane ** exponent * balance[k], 0, 1)
    return rgb


def metrics(rgb, target):
    """Agreement of the decoded picture with the one shown."""
    t = np.asarray(target, float)
    if t.shape != rgb.shape:
        return {}
    res = dict(
        r_all=float(np.corrcoef(t.ravel(), rgb.ravel())[0, 1]),
        r_planes=[float(np.corrcoef(t[:, :, k].ravel(), rgb[:, :, k].ravel())[0, 1])
                  if t[:, :, k].std() > 0 else float("nan") for k in range(3)],
        mean_abs_error=float(np.abs(t - rgb).mean()))
    # cells shown in one of the 8 pure colours (every channel fully on or off)
    pure = ((t <= 0.25) | (t >= 0.75)).all(axis=2)
    if pure.any():
        same = ((rgb > 0.5) == (t > 0.5)).all(axis=2)
        res["pure_cells"] = int(pure.sum())
        res["pure_right"] = int(same[pure].sum())
    return res


def summary(res):
    text = "colour r = %.2f (R %.2f, G %.2f, B %.2f)" % (res["r_all"], *res["r_planes"])
    if "pure_cells" in res:
        text += f", {res['pure_right']} of {res['pure_cells']} pure-colour cells right"
    if res.get("r_null"):
        text += f"; time-shifted EEG reaches {max(res['r_null']):.2f}"
    return text


def save_images(session_dir, rgb, target=None, scale=24):
    def big(a):
        img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="RGB")
        return img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    big(rgb).save(os.path.join(session_dir, "reconstruction_smooth.png"))
    if target is None or np.asarray(target).shape != rgb.shape:
        return None
    left, right = big(np.asarray(target, float)), big(rgb)
    gap = scale
    both = Image.new("RGB", (left.width + gap + right.width, left.height), (40, 40, 40))
    both.paste(left, (0, 0))
    both.paste(right, (left.width + gap, 0))
    path = os.path.join(session_dir, "smooth_comparison.png")   # shown | decoded
    both.save(path)
    return path


def run(session_dir, tags, weights=None, spc=0.0, target=None, channels="",
        balance=COLOR_BALANCE, exponent=SHADE_EXPONENT, alpha=None, n_null=6):
    """Decode a recorded scan, save the picture and score it against the
    target and against time-shifted EEG."""
    grid = decode(session_dir, tags, weights, spc, channels=channels, alpha=alpha)
    np.save(os.path.join(session_dir, "reconstruction_smooth_grid.npy"), grid)
    rgb = to_rgb(grid, balance, exponent)
    res = dict(tags=[float(t) for t in tags], balance=list(balance), exponent=exponent)
    if target is not None:
        res.update(metrics(rgb, target))
        eeg_t = np.genfromtxt(os.path.join(session_dir, "eeg.csv"), delimiter=",",
                              skip_header=1, usecols=[0])
        span, nulls = eeg_t[-1] - eeg_t[0], []
        for k in range(n_null):
            shift = 31.0 + 29.0 * k
            if shift > span - 10:
                break
            m = metrics(to_rgb(decode(session_dir, tags, weights, spc, shift, channels,
                                      alpha), balance, exponent), target)
            nulls.append(m.get("r_all"))
        res["r_null"] = nulls
    res["comparison"] = save_images(session_dir, rgb, target)
    return grid, rgb, res


def main():
    ap = argparse.ArgumentParser(description="re-decode a --mode smooth session")
    ap.add_argument("--session", required=True)
    ap.add_argument("--channels", default="")
    ap.add_argument("--color-balance", default="",
                    help="R,G,B multipliers on the finished planes (default: as recorded)")
    ap.add_argument("--shade-exponent", type=float, default=None)
    a = ap.parse_args()
    with open(os.path.join(a.session, "smooth_meta.json")) as f:
        meta = json.load(f)
    tpath = os.path.join(a.session, "target_color.npy")
    target = np.load(tpath) if os.path.exists(tpath) else None
    balance = ([float(v) for v in a.color_balance.split(",")] if a.color_balance
               else meta.get("balance", COLOR_BALANCE))
    exponent = meta.get("exponent", SHADE_EXPONENT) if a.shade_exponent is None \
        else a.shade_exponent
    weights = None if a.channels else meta.get("weights")
    _, _, res = run(a.session, meta["tags"], weights, meta.get("spc", 0.0), target,
                    a.channels, balance, exponent, meta.get("alpha"))
    print(summary(res) if "r_all" in res else "decoded (no target to compare with)")
    print("->", os.path.join(a.session, "reconstruction_smooth.png"))


if __name__ == "__main__":
    main()
