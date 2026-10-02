"""Second half of patch_pause_grad.py (reconstruct.py + targets.py); the
first half (page + driver) applied before the reconstruct pattern proved
non-unique. Run once from the repo root."""
import io

GUARD = '''        if gx[s] < 0 or gy[s] < 0:
            continue  # pause marker, not a position
        if t1 - t0 > 1.0 and (i1 - i0) < 0.6 * (t1 - t0) * fs:
            continue  # the stream dropped during this visit (it is redone)
'''
p = "reconstruct.py"
x = io.open(p, encoding="utf-8").read()
n = 0
for a in ('''        t0, t1 = cur_t[s], cur_t[e - 1]
        i0 = np.searchsorted(eeg_t, t0)
        i1 = np.searchsorted(eeg_t, t1)
''', '''        t0, t1 = cur_t[s], cur_t[e - 1]
        i0, i1 = np.searchsorted(eeg_t, t0), np.searchsorted(eeg_t, t1)
'''):
    n += x.count(a)
    x = x.replace(a, a + GUARD)
assert n >= 2, n
io.open(p, "w", encoding="utf-8", newline="\n").write(x)

p = "targets.py"
x = io.open(p, encoding="utf-8").read()
a = '''    if spec.startswith("text:"):
        g = text_target(spec[5:], grid_w, grid_h)'''
b = '''    if spec.startswith("grad:"):
        # letters at descending grey levels (first white, last half grey):
        # 'grad:NO' = white N, half-grey O
        txt = spec[5:]
        g = (text_target(txt, grid_w, grid_h) > 0.5).astype(float)
        n = max(len(txt.replace(" ", "")), 1)
        edges = np.linspace(0, g.shape[1], n + 1).round().astype(int)
        for k in range(n):
            g[:, edges[k]:edges[k + 1]] *= 1.0 - 0.5 * k / max(n - 1, 1)
        return g
''' + a
assert x.count(a) == 1
io.open(p, "w", encoding="utf-8", newline="\n").write(x.replace(a, b))
print("patched", n, "loops")
