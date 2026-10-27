"""The eye exam read out: thresholds from the per-step responses in
<session>/exam.json (written by xr_session.py --mode exam) and the report.

  flash     the full-field flicker's response on every channel: the eye's
            reference channel and amplitude (p < 0.05), which every other
            test's amplitude is also given relative to (rel_flash), so an
            electrode that drifts between the eyes cancels out of their
            comparison; no flash response = the sitting has no reference
  temporal  the flash at several rates: the noise-corrected amplitude per rate
            with the OFF blocks' line power taken out in quadrature (alpha under
            a 10 Hz step), the best rate, the rate above it where the response
            has fallen to half (log-linear between the bracketing rates; a
            rate without a response counts as nothing), and the apparent
            latency from the slope of the response phase against rate over the
            rates that respond (Regan, 1966; fit_latency), whose difference
            between the eyes is the measurement
  acuity    response amplitude against log bar width; a line through the limb
            that falls to the noise floor (from the step just below the finest
            bar still giving a significant response, through it, and up toward
            the peak while the amplitude rises), extrapolated to zero amplitude
            (the sweep-VEP rule of Norcia and Tyler, 1985) -> threshold bar
            width in arcmin, with its logMAR and a Snellen-style equivalent (a
            1-arcmin bar is the 20/20 detail), and the finest bar with a
            significant response
  contrast  amplitude against log contrast at one bar width, the same rule
            (Campbell and Maffei, 1970) -> threshold contrast, log sensitivity
  field     amplitude and phase per sector, each against the test's OFF blocks;
            sectors without a significant response are listed; with two eyes,
            the sector-by-sector amplitude ratio and phase difference
  colour    amplitude per plane, red : green : blue

    python analysis/exam_readout.py runs/<session>

Writes <session>/exam_readout.json, exam_report.pdf (vector, text embedded)
and exam_report.png. These are response measurements from one electrode
against a flickering display; they are not a clinical diagnosis.
"""
import argparse
import json
import math
import os
import textwrap

import numpy as np

TESTS = ["flash", "temporal", "acuity", "contrast", "field", "colour"]
ALPHA = 0.05
LAT_WINDOW_S = (0.03, 0.30)   # apparent latencies searched (the pathway's 80-150 ms plus the rig's offset)
RISE_FRAC = 0.1        # a limb point must add this fraction of the peak amplitude


def read_flash(stage):
    out = dict(ok=False, responded=False)
    if not stage or not stage.get("ok") or not stage.get("steps"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    s = stage["steps"][0]
    amps_ch = {k: v for k, v in (s.get("amps") or {}).items()}
    out.update(ok=True, p=float(s["p"]), amp=s.get("amp"), amp_sd=s.get("amp_sd"),
               snr=s.get("snr"), channel=stage.get("channel"), best=s.get("best"),
               amps_ch=amps_ch, off_amps=stage.get("off_amps") or {},
               channel_p=stage.get("channel_p") or {},
               rms_uv=stage.get("rms_uv") or {}, dead=stage.get("dead") or [],
               freq=stage.get("freq"), n_on=s.get("n_on"), n_off=s.get("n_off"),
               responded=bool(s["p"] < ALPHA and s.get("amp")))
    return out


def fit_latency(rates, phases_deg, weights=None):
    """Apparent latency from the response phase against rate (Regan, 1966): a
    response delayed by tau has phase = phi0 - 360 f tau (mod 360), with phi0
    the waveform's own constant offset. A grid over tau (LAT_WINDOW_S) picks
    the delay whose residual phases agree most (largest weighted mean
    resultant), which fixes each phase's turn count; a weighted line through
    the unwrapped phases against rate then gives tau and its standard error.
    Needs three rates. The rates' spacing must not alias within the window:
    the panel's rates (6, 7.5, 10, 15, 30 or 6, 7.2, 9, 12, 18, 36 Hz) repeat
    only beyond 1.6 s."""
    f = np.asarray(rates, float)
    ph = np.radians(np.asarray(phases_deg, float))
    n = len(f)
    if n < 3:
        return dict(ok=False, reason="fewer than three rates respond", n=int(n))
    w = np.ones(n) if weights is None else np.clip(np.asarray(weights, float), 1e-3, None)
    taus = np.arange(LAT_WINDOW_S[0], LAT_WINDOW_S[1] + 1e-9, 0.0001)
    E = np.exp(1j * (ph[None, :] + 2 * np.pi * f[None, :] * taus[:, None]))
    R = np.abs((E * w[None, :]).sum(1)) / w.sum()
    k = int(np.argmax(R))
    tau0 = float(taus[k])
    resid = ph + 2 * np.pi * f * tau0
    z = (w * np.exp(1j * resid)).sum()
    phi0 = math.atan2(z.imag, z.real)
    unwrapped = phi0 + np.angle(np.exp(1j * (resid - phi0))) - 2 * np.pi * f * tau0
    X = np.column_stack([np.ones(n), f])
    XtW = X.T * w[None, :]
    G = np.linalg.inv(XtW @ X)
    beta = G @ (XtW @ unwrapped)
    tau = -beta[1] / (2 * np.pi)
    res = unwrapped - X @ beta
    dof = n - 2
    se = None
    if dof > 0:
        s2 = float((w * res ** 2).sum() / dof)
        se = math.sqrt(G[1, 1] * s2) / (2 * np.pi)
    return dict(ok=True, latency_ms=float(tau * 1000), latency_se_ms=None if se is None else float(se * 1000),
                offset_deg=float(math.degrees(beta[0])), R=float(R[k]), n=int(n),
                rates=f.tolist(), phase_unwrapped_deg=np.degrees(unwrapped).tolist(),
                resid_deg=np.degrees(res).tolist(), weights=w.tolist(),
                at_window_edge=bool(k == 0 or k == len(taus) - 1))


def read_temporal(stage):
    """The flicker sweep: amplitude against rate, the half-response rate and
    the apparent latency."""
    out = dict(ok=False)
    if not stage or not stage.get("ok") or not stage.get("steps"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    steps = [st for st in stage["steps"] if st.get("amp") is not None]
    steps.sort(key=lambda st: st.get("freq") or st["meta"].get("hz") or 0.0)
    f = np.array([st.get("freq") or st["meta"].get("hz") or np.nan for st in steps], float)
    amp = np.array([float(st["amp"]) for st in steps])
    p = np.array([float(st["p"]) for st in steps])
    off = np.array([st["off_amp"] if st.get("off_amp") is not None else np.nan for st in steps], float)
    ph = np.array([st["phase_deg"] if st.get("phase_deg") is not None else np.nan for st in steps], float)
    sd = np.array([st["phase_sd_deg"] if st.get("phase_sd_deg") is not None else np.nan for st in steps], float)
    ok_f = np.isfinite(f)
    f, amp, p, off, ph, sd = f[ok_f], amp[ok_f], p[ok_f], off[ok_f], ph[ok_f], sd[ok_f]
    # the OFF blocks' line power (alpha under a 10 Hz step) taken out in quadrature
    corr = np.sqrt(np.clip(amp ** 2 - np.nan_to_num(off) ** 2, 0.0, None))
    sig = (p < ALPHA) & (corr > 0)
    out.update(rates=f.tolist(), amps=amp.tolist(), off_amps=off.tolist(), amps_corr=corr.tolist(),
               p=p.tolist(), phase_deg=ph.tolist(), phase_sd_deg=sd.tolist(),
               n_rates=int(len(f)), n_responding=int(sig.sum()),
               responding=f[sig].tolist(), not_responding=f[~sig].tolist(),
               channel=stage.get("channel"), noise_amp=stage.get("noise_amp"))
    if len(f) < 2:
        out["reason"] = "fewer than two scored rates"
        return out
    if not sig.any():
        out["reason"] = "no rate responds"
        return out
    top = float(corr[sig].max())
    rel = corr / top
    rel_eff = np.where(sig, rel, 0.0)          # a rate without a response counts as nothing
    i_best = int(np.flatnonzero(sig)[np.argmax(corr[sig])])
    half, beyond = None, True
    for i in range(i_best + 1, len(f)):
        if rel_eff[i] < 0.5:
            x0, x1 = math.log(f[i - 1]), math.log(f[i])
            y0, y1 = rel_eff[i - 1], rel_eff[i]
            half = float(math.exp(x0 + (y0 - 0.5) / (y0 - y1) * (x1 - x0)))
            beyond = False
            break
    out.update(ok=True, relative=rel.tolist(), best_rate=float(f[i_best]), half_rate=half,
               half_beyond_range=beyond, fastest_rate=float(f.max()), slowest_rate=float(f.min()))
    use = sig & np.isfinite(ph)
    if use.sum() >= 3:
        fit = fit_latency(f[use], ph[use], weights=np.clip(rel[use], 0.05, 1.0))
    else:
        fit = dict(ok=False, reason="fewer than three rates respond", n=int(use.sum()))
    out["latency"] = fit
    if fit["ok"]:
        out["latency_ms"] = fit["latency_ms"]
        out["latency_se_ms"] = fit["latency_se_ms"]
    return out


def _rel_flash(d, ref_amp, key):
    """Add key + '_rel_flash': the test's amplitudes over the eye's flash amplitude."""
    if ref_amp and d.get(key) is not None:
        d[key + "_rel_flash"] = [float(v) / ref_amp if v is not None and np.isfinite(v) else None
                                 for v in d[key]]


def fit_limb(xlog, amp, p):
    """Line through the limb of amplitude vs log size (or log contrast) that
    falls to the noise floor: from the step just below the finest/faintest step
    with a significant response (the anchor, where the response has gone),
    through that step and up toward the peak while the amplitude keeps rising;
    extrapolated to zero amplitude (the sweep-VEP rule). A step lost to noise
    inside the limb then costs little, and a fit from the saturating top of
    the curve alone, which lands far too low, is avoided."""
    order = np.argsort(xlog)              # ascending: the threshold side first
    x, y, pp = xlog[order], amp[order], p[order]
    sig = np.where((pp < ALPHA) & (y > 0))[0]
    if len(sig) == 0:
        return dict(ok=False, reason="no step with a significant response")
    i0 = int(sig[0])
    peak = float(np.nanmax(y))
    used = [i0]
    for i in range(i0 + 1, len(x)):
        if y[i] - y[used[-1]] >= RISE_FRAC * peak:
            used.append(i)
        else:
            break
    note = ""
    beyond = False
    if i0 > 0:
        used = [i0 - 1] + used            # the step below, where the response has gone
    else:
        beyond = True                     # the finest step shown still responds
        note = "the finest step shown still responds: threshold extrapolated below the range shown"
    if len(used) < 2:
        return dict(ok=False, reason="the finest step shown still responds at full "
                                      "strength: threshold beyond the range shown",
                    finest_sig=float(x[i0]), beyond_range=True)
    if sum(pp[i] < ALPHA for i in used) < 2:
        # one responding step among silent ones is not a limb: a sweep that
        # works responds at every coarse step down to its threshold
        return dict(ok=False, reason="only one step responds: no limb to fit",
                    finest_sig=float(x[i0]))
    xs, ys = x[used], y[used]
    b, a = np.polyfit(xs, ys, 1)
    if b <= 0:
        return dict(ok=False, reason="amplitude does not rise along the limb",
                    finest_sig=float(x[i0]))
    return dict(ok=True, a=float(a), b=float(b), used=[int(i) for i in used],
                x_used=xs.tolist(), y_used=ys.tolist(), x_thr_log=float(-a / b),
                finest_sig=float(x[i0]), note=note, beyond_range=beyond)


def _steps(stage, key):
    steps = [s for s in stage.get("steps", []) if s.get("amp") is not None]
    x = np.array([float(s["meta"][key]) for s in steps])
    a = np.array([float(s["amp"]) for s in steps])
    p = np.array([float(s["p"]) for s in steps])
    sd = np.array([float(s["amp_sd"]) if s.get("amp_sd") is not None else 0.0 for s in steps])
    return steps, x, a, p, sd


def read_acuity(stage):
    out = dict(ok=False)
    if not stage or not stage.get("ok"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    steps, size, amp, p, sd = _steps(stage, "size")
    out.update(sizes=size.tolist(), amps=amp.tolist(), p=p.tolist(), amp_sd=sd.tolist(),
               noise_amp=stage.get("noise_amp"), channel=stage.get("channel"), freq=stage.get("freq"))
    if len(size) < 2:
        out["reason"] = "fewer than two scored steps"
        return out
    fit = fit_limb(np.log(size), amp, p)
    out["fit"] = fit
    if fit.get("finest_sig") is not None:
        out["finest_sig"] = float(math.exp(fit["finest_sig"]))
    if not fit["ok"]:
        out["reason"] = fit["reason"]
        return out
    thr = float(math.exp(fit["x_thr_log"]))
    out.update(ok=True, threshold=thr, logmar=float(math.log10(thr)),
               snellen=f"20/{20 * thr:.0f}",
               beyond_range=bool(fit.get("beyond_range") or thr < 0.5 * float(size.min())),
               note=fit.get("note", ""))
    return out


def read_contrast(stage):
    out = dict(ok=False)
    if not stage or not stage.get("ok"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    steps, c, amp, p, sd = _steps(stage, "contrast")
    out.update(contrasts=c.tolist(), amps=amp.tolist(), p=p.tolist(), amp_sd=sd.tolist(),
               noise_amp=stage.get("noise_amp"), channel=stage.get("channel"), freq=stage.get("freq"),
               size=float(steps[0]["meta"].get("size", 0)) if steps else None)
    if len(c) < 2:
        out["reason"] = "fewer than two scored steps"
        return out
    fit = fit_limb(np.log10(c), amp, p)
    out["fit"] = fit
    if fit.get("finest_sig") is not None:
        out["faintest_sig"] = float(10 ** fit["finest_sig"])
    if not fit["ok"]:
        out["reason"] = fit["reason"]
        return out
    thr = float(10 ** fit["x_thr_log"])
    out.update(ok=True, threshold=thr, log_cs=float(-math.log10(thr)),
               beyond_range=bool(fit.get("beyond_range") or thr < 0.5 * float(c.min())),
               note=fit.get("note", ""))
    return out


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def read_field(stage):
    out = dict(ok=False)
    if not stage or not stage.get("ok") or not stage.get("steps"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    s0 = stage["steps"][0]["meta"]
    cols, rows = int(s0.get("cols", 4)), int(s0.get("rows", 4))
    n = cols * rows
    amp = np.full(n, np.nan)
    p = np.ones(n)
    ph = np.full(n, np.nan)
    for s in stage["steps"]:
        k = int(s["meta"].get("sector", -1))
        if 0 <= k < n:
            amp[k] = s["amp"] if s["amp"] is not None else np.nan
            p[k] = s["p"]
            ph[k] = s["phase_deg"] if s.get("phase_deg") is not None else np.nan
    resp = (p < ALPHA) & np.isfinite(amp) & (amp > 0)
    ref = float(np.nanmax(amp[resp])) if resp.any() else float("nan")
    rel = amp / ref if np.isfinite(ref) and ref > 0 else amp * np.nan
    # phase relative to the amplitude-weighted circular mean of the responding sectors
    if resp.any() and np.isfinite(ph[resp]).any():
        w = np.nan_to_num(amp[resp])
        z = np.sum(w * np.exp(1j * np.radians(np.nan_to_num(ph[resp]))))
        mean_ph = math.degrees(math.atan2(z.imag, z.real))
    else:
        mean_ph = 0.0
    ph_rel = _wrap(ph - mean_ph)
    out.update(ok=True, cols=cols, rows=rows, n=n, amp=amp.tolist(), p=p.tolist(),
               rel=rel.tolist(), phase_deg=ph.tolist(), phase_rel_deg=ph_rel.tolist(),
               responding=[int(k) for k in np.where(resp)[0]],
               not_responding=[int(k) for k in np.where(~resp)[0]],
               n_responding=int(resp.sum()), channel=stage.get("channel"), freq=stage.get("freq"),
               noise_amp=stage.get("noise_amp"))
    return out


def read_colour(stage):
    out = dict(ok=False)
    if not stage or not stage.get("ok") or not stage.get("steps"):
        out["reason"] = (stage or {}).get("reason", "not run")
        return out
    amps, ps = [np.nan] * 3, [1.0] * 3
    for s in stage["steps"]:
        k = int(s["meta"].get("plane", -1))
        if 0 <= k < 3:
            amps[k] = s["amp"] if s["amp"] is not None else np.nan
            ps[k] = s["p"]
    amps = np.array(amps, float)
    top = float(np.nanmax(amps)) if np.isfinite(amps).any() else float("nan")
    rel = (amps / top) if np.isfinite(top) and top > 0 else amps * np.nan
    out.update(ok=True, amps=amps.tolist(), p=ps, relative=rel.tolist(),
               channel=stage.get("channel"), freq=stage.get("freq"), noise_amp=stage.get("noise_amp"))
    return out


def eye_text(eye, r):
    bits = []
    fl = r.get("flash")
    if fl:
        if fl.get("ok") and fl["responded"]:
            s = f"flash {fl['channel']} {fl['amp']:.2f} (p {fl['p']:.3f}, line SNR {fl['snr']:.1f})"
            bits.append(s)
        elif fl.get("ok"):
            bits.append(f"NO RESPONSE TO THE FULL-FIELD FLASH (p {fl['p']:.2f}): the finer tests "
                        "below cannot be read as the eye's")
        else:
            bits.append(f"flash: {fl.get('reason')}")
    te = r.get("temporal")
    if te:
        if te.get("ok"):
            s = (f"flicker sweep {te['slowest_rate']:.3g}-{te['fastest_rate']:.3g} Hz: "
                 f"{te['n_responding']}/{te['n_rates']} rates respond, best {te['best_rate']:.3g} Hz")
            s += (f", half by {te['half_rate']:.1f} Hz" if te.get("half_rate") is not None
                  else f", still over half at {te['fastest_rate']:.3g} Hz")
            lat = te.get("latency") or {}
            if lat.get("ok"):
                s += f"; apparent latency {lat['latency_ms']:.0f}"
                if lat.get("latency_se_ms") is not None:
                    s += f" +/- {lat['latency_se_ms']:.0f}"
                s += f" ms ({lat['n']} rates, R {lat['R']:.2f})"
            else:
                s += f"; latency: {lat.get('reason', '-')}"
            bits.append(s)
        else:
            bits.append(f"flicker sweep: {te.get('reason')}")
    ac = r.get("acuity")
    if ac:
        if ac.get("ok") and ac.get("beyond_range"):
            s = (f"acuity threshold below the finest bar shown ({min(ac['sizes']):.1f}'; "
                 f"the line crosses zero at {ac['threshold']:.1f}', not a measurement)")
            bits.append(s)
        elif ac.get("ok"):
            s = (f"acuity threshold {ac['threshold']:.1f}' (logMAR {ac['logmar']:.2f}, "
                 f"{ac['snellen']} equivalent, extrapolated)")
            if ac.get("finest_sig") is not None:
                s += f", finest bar with a response {ac['finest_sig']:.1f}'"
            bits.append(s)
        else:
            s = f"acuity: {ac.get('reason')}"
            if ac.get("finest_sig") is not None:
                s += f" (finest bar with a response {ac['finest_sig']:.1f}')"
            bits.append(s)
    co = r.get("contrast")
    if co:
        if co.get("ok") and co.get("beyond_range"):
            bits.append(f"contrast threshold below the faintest step shown ({min(co['contrasts']):.3g}; "
                        f"the line crosses zero at {co['threshold']:.3g}, not a measurement)")
        elif co.get("ok"):
            s = f"contrast threshold {co['threshold']:.3f} (log sensitivity {co['log_cs']:.2f}"
            if co.get("faintest_sig") is not None:
                s += f"; faintest with a response {co['faintest_sig']:.3g}"
            bits.append(s + ")")
        else:
            bits.append(f"contrast: {co.get('reason')}")
    fi = r.get("field")
    if fi:
        if fi.get("ok"):
            s = f"field {fi['n_responding']}/{fi['n']} sectors respond"
            if fi["not_responding"]:
                s += " (none from " + ", ".join(f"s{k + 1}" for k in fi["not_responding"]) + ")"
            bits.append(s)
        else:
            bits.append(f"field: {fi.get('reason')}")
    col = r.get("colour")
    if col:
        if col.get("ok"):
            rel = col["relative"]
            bits.append("colour R:G:B " + ":".join("-" if not np.isfinite(v) else f"{v:.2f}" for v in rel))
        else:
            bits.append(f"colour: {col.get('reason')}")
    ch = next((r[t].get("channel") for t in TESTS if r.get(t, {}).get("channel")), None)
    fq = next((r[t].get("freq") for t in TESTS if r.get(t, {}).get("freq")), None)
    dead = r.get("dead") or {}
    if dead:
        bits.append("out of contact: " + ", ".join(f"{k} {v:.0f} uV" for k, v in dead.items()))
    head = f"{eye.upper()} eye"
    if ch:
        head += f" ({ch}" + (f", {fq:.2f} Hz" if fq else "") + ")"
    return head + ": " + "; ".join(bits)


def compare_eyes(eyes):
    """Left against right when both were tested."""
    if "left" not in eyes or "right" not in eyes:
        return {}
    L, R = eyes["left"], eyes["right"]
    out = {}
    for t, key in (("flash", "amp"), ("acuity", "threshold"), ("contrast", "threshold")):
        a, b = L.get(t, {}), R.get(t, {})
        if a.get("ok") and b.get("ok") and a.get(key) and b.get(key) and b[key] > 0:
            out[f"{t}_ratio_lr"] = float(a[key] / b[key])
    fa, fb = L.get("field", {}), R.get("field", {})
    if fa.get("ok") and fb.get("ok") and fa["n"] == fb["n"]:
        la, lb = np.array(fa["amp"], float), np.array(fb["amp"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2(la / lb)
        ratio[~np.isfinite(ratio)] = np.nan
        dph = _wrap(np.array(fa["phase_rel_deg"], float) - np.array(fb["phase_rel_deg"], float))
        both = set(fa["responding"]) & set(fb["responding"])
        only_l = sorted(set(fa["responding"]) - set(fb["responding"]))
        only_r = sorted(set(fb["responding"]) - set(fa["responding"]))
        out.update(field_log2_ratio_lr=ratio.tolist(), field_phase_diff_deg=dph.tolist(),
                   field_only_left=only_l, field_only_right=only_r,
                   field_ratio_median=float(np.nanmedian(ratio[list(both)])) if both else None,
                   field_phase_diff_median=float(np.nanmedian(dph[list(both)])) if both else None)
        # the same ratio with each eye's sectors over its own flash: an
        # electrode that changed between the eyes cancels
        if fa.get("rel_flash") and fb.get("rel_flash") and both:
            ra = np.array([v if v is not None else np.nan for v in fa["rel_flash"]], float)
            rb = np.array([v if v is not None else np.nan for v in fb["rel_flash"]], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.log2(ra / rb)
            rr[~np.isfinite(rr)] = np.nan
            out["field_ratio_median_rel_flash"] = (float(np.nanmedian(rr[list(both)]))
                                                   if np.isfinite(rr[list(both)]).any() else None)
    ta, tb = L.get("temporal", {}), R.get("temporal", {})
    la, lb = ta.get("latency") or {}, tb.get("latency") or {}
    if la.get("ok") and lb.get("ok"):
        out["latency_diff_ms"] = float(la["latency_ms"] - lb["latency_ms"])
        if la.get("latency_se_ms") is not None and lb.get("latency_se_ms") is not None:
            out["latency_diff_se_ms"] = float(math.hypot(la["latency_se_ms"], lb["latency_se_ms"]))
    if ta.get("ok") and tb.get("ok"):
        # amplitude ratio at the rates both eyes were shown (within 1 %)
        ratios, rates = [], []
        for i, fa_ in enumerate(ta["rates"]):
            for j, fb_ in enumerate(tb["rates"]):
                if abs(fa_ - fb_) <= 0.01 * fb_ and ta["amps_corr"][i] > 0 and tb["amps_corr"][j] > 0:
                    ratios.append(math.log2(ta["amps_corr"][i] / tb["amps_corr"][j]))
                    rates.append(fa_)
        if ratios:
            out.update(temporal_rates=rates, temporal_log2_ratio_lr=ratios,
                       temporal_ratio_median=float(np.median(ratios)))
        out["half_rate_left"], out["half_rate_right"] = ta.get("half_rate"), tb.get("half_rate")
    return out


def compare_text(c):
    if not c:
        return ""
    bits = []
    if "flash_ratio_lr" in c:
        bits.append(f"flash amplitude left/right {c['flash_ratio_lr']:.2f}x")
    if "latency_diff_ms" in c:
        s = f"apparent latency left minus right {c['latency_diff_ms']:+.0f}"
        if c.get("latency_diff_se_ms") is not None:
            s += f" +/- {c['latency_diff_se_ms']:.0f}"
        bits.append(s + " ms")
    if "temporal_ratio_median" in c:
        bits.append(f"flicker-sweep amplitude left/right median {2 ** c['temporal_ratio_median']:.2f}x")
    if "acuity_ratio_lr" in c:
        bits.append(f"acuity threshold left/right {c['acuity_ratio_lr']:.2f}x")
    if "contrast_ratio_lr" in c:
        bits.append(f"contrast threshold left/right {c['contrast_ratio_lr']:.2f}x")
    if "field_log2_ratio_lr" in c:
        med = c.get("field_ratio_median")
        s = "field amplitude left/right " + (f"median {2 ** med:.2f}x" if med is not None else "-")
        if c["field_only_left"]:
            s += "; responding in the left eye only: " + ", ".join(f"s{k + 1}" for k in c["field_only_left"])
        if c["field_only_right"]:
            s += "; responding in the right eye only: " + ", ".join(f"s{k + 1}" for k in c["field_only_right"])
        if c.get("field_ratio_median_rel_flash") is not None:
            s += f" (each over its own flash: {2 ** c['field_ratio_median_rel_flash']:.2f}x)"
        if c.get("field_phase_diff_median") is not None:
            s += f"; phase difference median {c['field_phase_diff_median']:+.0f} deg"
        bits.append(s)
    return "Between the eyes: " + "; ".join(bits) if bits else ""


def draw(session, exam, eyes, compare, text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                         "font.family": "DejaVu Sans", "axes.titlesize": 9})
    names = exam["eyes"]
    tests = [t for t in TESTS if t in exam["tests"]]
    extra = 1 if compare.get("field_log2_ratio_lr") else 0
    nrow, ncol = len(names) + extra, max(len(tests), 2)
    width = 3.1 * ncol
    summary = textwrap.wrap(text.replace("\n", "  |  "), int(width * 12))   # ~12 chars/in at 7 pt
    top_in = 0.4 + 0.14 * len(summary) + 0.45          # title, summary lines, room for the panel titles
    bottom_in = 0.9                                     # x labels and the footer
    height = 2.9 * nrow + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(nrow, ncol, left=0.07, right=0.97, top=1 - top_in / height,
                          bottom=bottom_in / height, hspace=0.55, wspace=0.4)
    fig.text(0.07, 1 - 0.12 / height, f"Eye exam from the EEG - {os.path.basename(os.path.normpath(session))}",
             fontsize=11, weight="bold", va="top")
    fig.text(0.07, 1 - 0.4 / height, "\n".join(summary), fontsize=7, va="top", linespacing=1.35)
    for i, eye in enumerate(names):
        r = eyes[eye]
        for j, t in enumerate(tests):
            ax = fig.add_subplot(gs[i, j])
            d = r.get(t, {})
            if t == "flash":
                _draw_flash(ax, d)
                ttl = (f"{eye}: flash, {d['channel']} p {d['p']:.3f}" + ("" if d["responded"] else " (no response)")
                       if d.get("ok") else f"{eye}: flash, not scored")
            elif t == "temporal":
                _draw_temporal(ax, d)
                lat = d.get("latency") or {}
                ttl = (f"{eye}: flicker sweep, latency {lat['latency_ms']:.0f} ms" if lat.get("ok")
                       else f"{eye}: flicker sweep, no latency" if d.get("ok")
                       else f"{eye}: flicker sweep, not scored")
            elif t == "acuity":
                _draw_curve(ax, d, "sizes", "bar width (arcmin)", invert=True, logbase=math.e)
                ttl = (f"{eye}: acuity, below the finest bar shown" if d.get("ok") and d.get("beyond_range")
                       else f"{eye}: acuity, threshold {d['threshold']:.1f}' ({d['snellen']} eq.)"
                       if d.get("ok") else f"{eye}: acuity, no threshold")
            elif t == "contrast":
                _draw_curve(ax, d, "contrasts", "contrast", invert=True, logbase=10)
                ttl = (f"{eye}: contrast, below the faintest step shown" if d.get("ok") and d.get("beyond_range")
                       else f"{eye}: contrast, threshold {d['threshold']:.3f} (logCS {d['log_cs']:.2f})"
                       if d.get("ok") else f"{eye}: contrast, no threshold")
            elif t == "field":
                _draw_field(ax, d, fig)
                ttl = (f"{eye}: field, {d['n_responding']}/{d['n']} sectors respond"
                       if d.get("ok") else f"{eye}: field, not scored")
            else:
                _draw_colour(ax, d)
                ttl = f"{eye}: colour planes" if d.get("ok") else f"{eye}: colour, not scored"
            ax.set_title(ttl, loc="left")
    if extra:
        c = compare
        ax = fig.add_subplot(gs[nrow - 1, 0])
        fa = eyes["left"]["field"]
        m = np.array(c["field_log2_ratio_lr"], float).reshape(fa["rows"], fa["cols"])
        im = ax.imshow(m, cmap="RdBu_r", vmin=-2, vmax=2)
        for k in range(fa["n"]):
            v = m.flat[k]
            ax.text(k % fa["cols"], k // fa["cols"], "-" if not np.isfinite(v) else f"{2 ** v:.2f}x",
                    ha="center", va="center", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("field amplitude, left / right", loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log2 ratio")
        ax = fig.add_subplot(gs[nrow - 1, 1])
        m = np.array(c["field_phase_diff_deg"], float).reshape(fa["rows"], fa["cols"])
        im = ax.imshow(m, cmap="twilight_shifted", vmin=-180, vmax=180)
        for k in range(fa["n"]):
            v = m.flat[k]
            ax.text(k % fa["cols"], k // fa["cols"], "-" if not np.isfinite(v) else f"{v:+.0f}",
                    ha="center", va="center", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("response phase, left minus right (deg)", loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="deg")
        if ncol > 2:
            ax = fig.add_subplot(gs[nrow - 1, 2:])
            ax.axis("off")
            ax.text(0, 1, textwrap.fill(compare_text(c), 70), va="top", fontsize=7.5, linespacing=1.4)
    fig.text(0.07, 0.45 / height, textwrap.fill(
        "Response amplitudes at the flicker frequency and its double from one EEG channel (the eye's "
        "reference: the full-field flash's channel), noise-corrected, against the test's OFF blocks; "
        "open markers: not significant (p >= 0.05, exact permutation). Thresholds by linear "
        "extrapolation of the limb that falls to the noise floor, to zero amplitude (sweep-VEP rule). "
        "Not a clinical diagnosis.", int(width * 13)),
        fontsize=6.5, color="#555", va="top", linespacing=1.3)
    pdf = os.path.join(session, "exam_report.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(session, "exam_report.png"), dpi=130)
    plt.close(fig)
    return pdf


def _draw_curve(ax, d, xkey, xlabel, invert, logbase):
    if not d.get(xkey):
        ax.text(0.5, 0.5, d.get("reason", "not run"), ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    x = np.array(d[xkey], float); a = np.array(d["amps"], float); p = np.array(d["p"], float)
    sd = np.array(d.get("amp_sd", [0] * len(x)), float)
    sig = p < ALPHA
    ax.errorbar(x[sig], a[sig], yerr=sd[sig], fmt="o", ms=4, color="#1f4e79", ecolor="#9db2c8",
                capsize=2, lw=1, label="significant")
    ax.errorbar(x[~sig], a[~sig], yerr=sd[~sig], fmt="o", ms=4, mfc="white", color="#1f4e79",
                ecolor="#9db2c8", capsize=2, lw=1, label="p >= 0.05")
    ax.axhline(0, color="#888", lw=0.6)
    if d.get("noise_amp") is not None:
        ax.axhline(d["noise_amp"], color="#888", lw=0.6, ls=":", label="OFF-block amplitude")
    fit = d.get("fit") or {}
    if fit.get("ok"):
        xs = np.array(fit["x_used"])
        # the line is drawn over the fitted points and down to its zero crossing,
        # but never past half the finest step shown (a crossing far below the
        # range is not a measurement and would squash the axis)
        floor = math.log(0.5 * float(x.min())) if logbase == math.e else math.log10(0.5 * float(x.min()))
        lo = max(min(fit["x_thr_log"], xs.min()), floor); hi = xs.max()
        xx = np.linspace(lo, hi, 50)
        ax.plot(logbase ** xx, fit["a"] + fit["b"] * xx, color="#c0392b", lw=1.2, label="limb fit")
        if not d.get("beyond_range"):
            ax.axvline(d["threshold"], color="#c0392b", lw=0.8, ls="--")
    ax.set_xscale("log")
    if invert:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("response amplitude (a.u.)")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_temporal(ax, d):
    """Amplitude against flicker rate (filled: significant), the half-response
    rate, and on the right axis the unwrapped response phase with the fitted
    line whose slope is the apparent latency."""
    if not d.get("rates"):
        ax.text(0.5, 0.5, d.get("reason", "not run"), ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    f = np.array(d["rates"], float); a = np.array(d["amps_corr"], float); p = np.array(d["p"], float)
    sig = p < ALPHA
    ax.plot(f, a, "-", color="#9db2c8", lw=0.8)
    ax.plot(f[sig], a[sig], "o", ms=4, color="#1f4e79", label="significant")
    ax.plot(f[~sig], a[~sig], "o", ms=4, mfc="white", color="#1f4e79", label="p >= 0.05")
    ax.axhline(0, color="#888", lw=0.6)
    if d.get("half_rate") is not None:
        ax.axvline(d["half_rate"], color="#c0392b", lw=0.8, ls="--", label=f"half at {d['half_rate']:.1f} Hz")
    ax.set_xscale("log")
    ax.set_xticks(f); ax.set_xticklabels([f"{v:.3g}" for v in f], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("flicker rate (Hz)"); ax.set_ylabel("response amplitude (a.u.)")
    ax.set_ylim(0, max(float(np.nanmax(a)) if np.isfinite(a).any() else 1.0, 1e-6) * 1.3)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.spines[["top"]].set_visible(False)
    lat = d.get("latency") or {}
    if lat.get("ok"):
        ax2 = ax.twinx()
        fr = np.array(lat["rates"], float); pu = np.array(lat["phase_unwrapped_deg"], float)
        ax2.plot(fr, pu, "s", ms=3, color="#7f8c8d")
        xx = np.linspace(fr.min(), fr.max(), 20)
        ax2.plot(xx, lat["offset_deg"] - 360.0 * xx * lat["latency_ms"] / 1000.0, color="#7f8c8d", lw=0.8)
        ax2.set_ylabel("phase (deg)", color="#7f8c8d", fontsize=7, labelpad=1)
        ax2.tick_params(axis="y", colors="#7f8c8d", labelsize=6.5)
        ax2.spines[["top"]].set_visible(False)
        txt = f"apparent latency {lat['latency_ms']:.0f}"
        if lat.get("latency_se_ms") is not None:
            txt += f" +/- {lat['latency_se_ms']:.0f}"
        ax.text(0.03, 0.05, txt + f" ms (R {lat['R']:.2f})", transform=ax.transAxes, fontsize=6.5)


def _draw_flash(ax, d):
    """Per channel: the full-field flash's ON amplitude (bars) and the OFF
    amplitude (ticks); the reference channel filled, out-of-contact channels
    hatched."""
    if not d.get("ok"):
        ax.text(0.5, 0.5, d.get("reason", "not run"), ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    names = list(d["amps_ch"].keys())
    on = np.array([d["amps_ch"][n] if d["amps_ch"][n] is not None else 0.0 for n in names], float)
    off = np.array([d["off_amps"].get(n) if d["off_amps"].get(n) is not None else np.nan for n in names], float)
    x = np.arange(len(names))
    bars = ax.bar(x, on, width=0.6, color="#1f4e79", alpha=0.35)
    for n, b in zip(names, bars):
        if n == d.get("channel") and d["responded"]:
            b.set_alpha(1.0)
        elif d.get("channel_p", {}).get(n, 1.0) < ALPHA:
            b.set_alpha(0.6)
        if n in d.get("dead", []):
            b.set_hatch("///"); b.set_alpha(0.25)
    ax.plot(x, off, "_", color="#c0392b", ms=14, mew=1.5, label="OFF blocks")
    labels = []
    for n in names:
        rms = d.get("rms_uv", {}).get(n)
        labels.append(n + (f"\n{rms:.0f} uV" if rms is not None else ""))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("flash amplitude (a.u.)")
    ax.set_ylim(0, max(float(np.nanmax(on)) if np.isfinite(on).any() else 1.0, 1e-6) * 1.25)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_field(ax, d, fig):
    if not d.get("ok"):
        ax.text(0.5, 0.5, d.get("reason", "not run"), ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    rows, cols = d["rows"], d["cols"]
    rel = np.array(d["rel"], float).reshape(rows, cols)
    im = ax.imshow(np.nan_to_num(rel), cmap="magma", vmin=0, vmax=1)
    ph = np.array(d["phase_rel_deg"], float)
    p = np.array(d["p"], float)
    for k in range(d["n"]):
        y, x = divmod(k, cols)
        if p[k] < ALPHA and np.isfinite(rel.flat[k]):
            ax.text(x, y, f"{rel.flat[k]:.2f}\n{ph[k]:+.0f}°" if np.isfinite(ph[k]) else f"{rel.flat[k]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if rel.flat[k] < 0.55 else "black")
        else:
            ax.text(x, y, "×", ha="center", va="center", fontsize=12, color="#e74c3c")
    ax.plot(cols / 2 - 0.5, rows / 2 - 0.5, "o", ms=5, mfc="white", mec="black")   # the fixation dot
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="amplitude / best sector")


def _draw_colour(ax, d):
    if not d.get("ok"):
        ax.text(0.5, 0.5, d.get("reason", "not run"), ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    rel = np.nan_to_num(np.array(d["relative"], float))
    p = np.array(d["p"], float)
    bars = ax.bar(["red", "green", "blue"], rel, color=["#d62728", "#2ca02c", "#1f77b4"], width=0.6)
    for b, pv in zip(bars, p):
        if pv >= ALPHA:
            b.set_hatch("///"); b.set_alpha(0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("amplitude / best plane")
    ax.spines[["top", "right"]].set_visible(False)


def run(session):
    with open(os.path.join(session, "exam.json")) as f:
        exam = json.load(f)
    eyes = {}
    for eye in exam["eyes"]:
        res = exam["results"].get(eye, {})
        r = {}
        ref_amp = None
        if "flash" in exam["tests"]:
            r["flash"] = read_flash(res.get("flash"))
            if r["flash"]["responded"]:
                ref_amp = float(r["flash"]["amp"])
        if "temporal" in exam["tests"]:
            r["temporal"] = read_temporal(res.get("temporal"))
            _rel_flash(r["temporal"], ref_amp, "amps")
        if "acuity" in exam["tests"]:
            r["acuity"] = read_acuity(res.get("acuity"))
            _rel_flash(r["acuity"], ref_amp, "amps")
        if "contrast" in exam["tests"]:
            r["contrast"] = read_contrast(res.get("contrast"))
            _rel_flash(r["contrast"], ref_amp, "amps")
        if "field" in exam["tests"]:
            r["field"] = read_field(res.get("field"))
            _rel_flash(r["field"], ref_amp, "amp")
            if "amp_rel_flash" in r["field"]:
                r["field"]["rel_flash"] = r["field"].pop("amp_rel_flash")
        if "colour" in exam["tests"]:
            r["colour"] = read_colour(res.get("colour"))
            _rel_flash(r["colour"], ref_amp, "amps")
        r["ref_amp"] = ref_amp
        # electrodes the scorer set aside as out of contact, with their rms (the worst test)
        dead = {}
        for st in res.values():
            for ch in st.get("dead", []) if isinstance(st, dict) else []:
                dead[ch] = max(dead.get(ch, 0.0), float(st.get("rms_uv", {}).get(ch, 0.0)))
        r["dead"] = dead
        eyes[eye] = r
    compare = compare_eyes(eyes)
    lines = [eye_text(e, eyes[e]) for e in exam["eyes"]]
    ct = compare_text(compare)
    if ct:
        lines.append(ct)
    text = "\n".join(lines)
    out = dict(session=session, ppd=exam.get("ppd"), freq=exam.get("freq"), eyes=eyes,
               compare=compare, text=text,
               delivery_ok=all(d.get("ok") for d in exam.get("delivery", {}).values()))
    pdf = None
    try:
        pdf = draw(session, exam, eyes, compare, text)
    except Exception as exc:   # the numbers stand even if the picture fails
        out["figure_error"] = repr(exc)
    out["report"] = pdf
    with open(os.path.join(session, "exam_readout.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session")
    a = ap.parse_args()
    out = run(a.session)
    print(out["text"])
    if out.get("figure_error"):
        print("figure failed:", out["figure_error"])
    else:
        print("report:", out["report"])


if __name__ == "__main__":
    main()
