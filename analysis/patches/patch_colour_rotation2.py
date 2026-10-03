"""Second half of patch_colour_rotation.py: reconstruct.py (within
reconstruct_color only) and eyecam.py. Run once from the repo root."""
import io

p = "reconstruct.py"
x = io.open(p, encoding="utf-8").read()
start = x.index("def reconstruct_color(")
end = x.index("\ndef ", start + 10)
body = x[start:end]

a = '''    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)'''
b = '''    w = np.ones(n_ch) if weights is None else np.asarray(weights, float)
    gains = np.ones(3) if gains is None else np.asarray(gains, float)
    # Per-pass colour-to-frequency assignment (tag rotation). Without the
    # file, colour k used freqs[k] throughout.
    passes = []
    pfile = os.path.join(session_dir, "color_passes.json")
    if os.path.exists(pfile):
        with open(pfile) as f:
            passes = sorted(json.load(f), key=lambda e: e["t"])

    def hz_for(t_visit):
        hz = list(freqs)
        for e in passes:
            if e["t"] <= t_visit + 0.05 and len(e.get("hz", [])) == 3:
                hz = [float(v) for v in e["hz"]]
        return hz'''
assert body.count(a) == 1
body = body.replace(a, b)
a2 = '''        for k, f0 in enumerate(freqs):
            num = den = 0.0'''
b2 = '''        for k, f0 in enumerate(hz_for(t0)):
            num = den = 0.0'''
assert body.count(a2) == 1
body = body.replace(a2, b2)
x = x[:start] + body + x[end:]
io.open(p, "w", encoding="utf-8", newline="\n").write(x)

p = "eyecam.py"
x = io.open(p, encoding="utf-8").read()
a = '''             "--color-reps", "3", "--color-on", "6", "--color-spc", "6",'''
b = a + '''\n             "--color-passes", "3",'''
assert x.count(a) == 1
io.open(p, "w", encoding="utf-8", newline="\n").write(x.replace(a, b))
print("patched")
