"""One-shot patch: exact permutation calibration gate (PLAN_NEXT Rank A #1),
G1-only session mode, configurable block lengths, recorder stop at scan end."""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

c = open("config.py").read()
if "CALIB_P_MAX" not in c:
    c = c.replace(
        "CALIB_WEIGHT_DPRIME_MIN = 0.5  # channels below this get zero weight",
        "CALIB_WEIGHT_DPRIME_MIN = 0.5  # channels below this get zero weight\n"
        "CALIB_P_MAX = 0.01         # exact family-wise permutation p for 'passed'")
    open("config.py", "w").write(c)

s = open("run_session.py").read()
a = "def score_calibration(session, blocks, stim_freq=None):"
PERM = '''def permutation_gate(block_scores, kinds, p_max=None, max_exact=250000, seed=0):
    """Exact family-wise permutation test (PLAN_NEXT Rank A #1).

    statistic = max over channels of mean(ON) - mean(OFF) block scores;
    null = every C(N, n_on) relabeling of the blocks all channels scored
    (Monte Carlo 20k if the enumeration is too large). p >= 1/n_null by
    construction; the gate is 'decidable' only when 1/n_null <= p_max."""
    from itertools import combinations
    from math import comb
    p_max = config.CALIB_P_MAX if p_max is None else p_max
    S = np.array(block_scores, dtype=float)          # (channels, blocks)
    valid = ~np.isnan(S).any(axis=0)
    S = S[:, valid]
    labels = np.array([k == "on" for k, v in zip(kinds, valid) if v])
    N, n_on = len(labels), int(labels.sum())
    out = {"n_blocks_used": int(N), "n_on": n_on, "p_max": p_max}
    if N < 4 or n_on == 0 or n_on == N or S.shape[0] == 0:
        out.update(decidable=False, p_fw=1.0, p_channel=[], stat=0.0, n_null=0)
        return out

    def stat(mask):
        return S[:, mask].mean(axis=1) - S[:, ~mask].mean(axis=1)

    obs = stat(labels)
    total = comb(N, n_on)
    if total <= max_exact:
        n_null, exact = total, True
        masks = (np.isin(np.arange(N), on_idx)
                 for on_idx in combinations(range(N), n_on))
    else:
        n_null, exact = 20000, False
        rng = np.random.default_rng(seed)
        masks = (np.isin(np.arange(N), rng.choice(N, n_on, replace=False))
                 for _ in range(n_null))
    null = np.empty((n_null, S.shape[0]))
    for i, m in enumerate(masks):
        null[i] = stat(m)
    null_fw = null.max(axis=1)
    out.update(decidable=bool(1.0 / n_null <= p_max), exact=exact,
               n_null=int(n_null), stat=float(obs.max()),
               p_fw=float((null_fw >= obs.max()).mean()),
               p_channel=[float((null[:, ch] >= obs[ch]).mean())
                          for ch in range(S.shape[0])])
    return out


'''
assert a in s and "def permutation_gate" not in s
s = s.replace(a, PERM + a)

b = '''    weights = {}
    best = None
    for c, name in enumerate(names):
        sigma = reconstruct.hf_sigma(data[:, c])
        scores = {"on": [], "off": []}
        for kind, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            sc = reconstruct.cell_score(data[i0:i1, c], fs, sigma, stim_freq)
            if not np.isnan(sc):
                scores[kind].append(float(sc))'''
b2 = '''    weights = {}
    best = None
    block_scores = []  # per channel, per block (NaN if dropped)
    for c, name in enumerate(names):
        sigma = reconstruct.hf_sigma(data[:, c])
        scores = {"on": [], "off": []}
        per_block = []
        for kind, t0, t1 in blocks:
            i0 = np.searchsorted(t, t0 + config.CALIB_DISCARD_S)
            i1 = np.searchsorted(t, t1)
            sc = reconstruct.cell_score(data[i0:i1, c], fs, sigma, stim_freq)
            per_block.append(float(sc))
            if not np.isnan(sc):
                scores[kind].append(float(sc))
        block_scores.append(per_block)'''
assert b in s
s = s.replace(b, b2)

d = '''    b = result["channels"][best]
    result["best"] = best
    result["passed"] = bool(b["dprime"] >= config.CALIB_DPRIME_MIN
                            and b["ratio"] >= config.CALIB_RATIO_MIN
                            and b["rank"] >= config.CALIB_RANK_MIN)
    result["weights"] = weights'''
d2 = '''    b = result["channels"][best]
    result["best"] = best
    legacy = bool(b["dprime"] >= config.CALIB_DPRIME_MIN
                  and b["ratio"] >= config.CALIB_RATIO_MIN
                  and b["rank"] >= config.CALIB_RANK_MIN)
    perm = permutation_gate(block_scores, [k for k, _, _ in blocks])
    result["permutation"] = perm
    result["passed_dprime_legacy"] = legacy
    if perm["decidable"]:
        # d'/ratio/rank are reported only; the exact test decides.
        result["passed"] = bool(perm["p_fw"] < perm["p_max"])
        result["gate"] = "permutation"
        if perm["p_channel"]:
            result["best"] = names[int(np.argmin(perm["p_channel"]))]
    else:
        result["passed"] = legacy
        result["gate"] = "dprime-legacy (too few blocks for exact p)"
    result["weights"] = weights'''
assert d in s
s = s.replace(d, d2)
open("run_session.py", "w").write(s)
print("run_session.py patched")

x = open("xr_session.py").read()
e = '''    ap.add_argument("--mode", choices=["visual", "assr"], default="visual",
                    help="assr = 'ear as a microphone' 40 Hz tone blocks")'''
e2 = '''    ap.add_argument("--mode", choices=["visual", "assr", "calib"],
                    default="visual",
                    help="calib = G1 only (calibration, no scan); "
                         "assr = 'ear as a microphone' 40 Hz tone blocks")
    ap.add_argument("--calib-on", type=float, default=config.CALIB_ON_S)
    ap.add_argument("--calib-off", type=float, default=config.CALIB_OFF_S)'''
assert e in x
x = x.replace(e, e2)

f = '''            await self.send(cmd="start_calib", blocks=a.calib_blocks,
                            onS=config.CALIB_ON_S, offS=config.CALIB_OFF_S,
                            freq=a.freq)
            await self.wait_for("calib_done",
                                timeout=a.calib_blocks * 16 + 60)'''
f2 = '''            await self.send(cmd="start_calib", blocks=a.calib_blocks,
                            onS=a.calib_on, offS=a.calib_off, freq=a.freq)
            await self.wait_for(
                "calib_done",
                timeout=a.calib_blocks * (a.calib_on + a.calib_off + 2) + 60)'''
assert f in x
x = x.replace(f, f2)

g = '''            table = "   ".join(
                f"{n} d'={calib['channels'][n]['dprime']:.1f}"
                f" w={calib['weights'][n]:.2f}"
                for n in calib["names"])
            print(f"calibration passed={calib['passed']} {table}")'''
g2 = '''            perm = calib.get("permutation", {})
            pch = perm.get("p_channel") or [float("nan")] * len(calib["names"])
            table = "   ".join(
                f"{n} d'={calib['channels'][n]['dprime']:.1f}"
                f" p={pch[i]:.3f} w={calib['weights'][n]:.2f}"
                for i, n in enumerate(calib["names"]))
            table += (f"   |  gate: {calib.get('gate')}"
                      f"  p_fw={perm.get('p_fw', float('nan')):.4f}"
                      f" (n_null={perm.get('n_null', 0)})")
            print(f"calibration passed={calib['passed']} {table}")'''
assert g in x
x = x.replace(g, g2)

h = '''            await asyncio.sleep(2.5)
            if not calib["passed"] and a.require_pass:
                raise RuntimeError("calibration gate failed (no SSVEP)")'''
h2 = '''            await asyncio.sleep(2.5)
            if a.mode == "calib":
                # G1-only session: the question is whether an SSVEP exists.
                result.update(ok=True, gate=calib.get("gate"),
                              p_fw=calib.get("permutation", {}).get("p_fw"))
                recorder.stop()
                print(f"G1 SESSION DONE passed={calib['passed']} "
                      f"-> {self.session}")
                await asyncio.sleep(a.linger)
                return 0
            if not calib["passed"] and a.require_pass:
                raise RuntimeError("calibration gate failed (no SSVEP)")'''
assert h in x
x = x.replace(h, h2)

i = '''            result["ok"] = True
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")'''
i2 = '''            result["ok"] = True
            recorder.stop()  # stop at scan end, not after the linger
            print(f"SESSION DONE r={r} flicker={flicker_hz:.2f} Hz "
                  f"-> {self.session}")'''
assert i in x
x = x.replace(i, i2)
open("xr_session.py", "w").write(x)
print("xr_session.py patched")
