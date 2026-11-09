# -*- coding: utf-8 -*-
"""One-off (2026-11-09): the matter mode reports the brain-vs-frame correlation over the squares that were
actually shown (reconstruct's whole-grid r dilutes it with squares nobody saw); the nulls likewise."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "xr_session.py")
s = io.open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(old, new):
    global s
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if new in s and old not in s:
        return
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''        grid, r = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                  method=method, vblend=False)
        nulls = []   # the same scoring on misaligned EEG: what chance gives
        for k in range(8):
            try:
                _, r0 = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                        method=method, vblend=False, save=False,
                                        shift_s=23.0 * (k + 1) + 7.0)
                if r0 is not None:
                    nulls.append(float(r0))
            except Exception as exc:
                print("null:", repr(exc)[:120])
                break
        null_max = max(nulls) if nulls else None''',
     '''        grid, r_full = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                       method=method, vblend=False)
        shown = np.zeros((gh, gw), bool)
        for gx_, gy_ in found["cells"]:
            shown[gy_, gx_] = True

        def r_shown(g_):
            g_ = np.nan_to_num(np.asarray(g_, float))
            if shown.sum() < 3 or np.std(g_[shown]) == 0 or np.std(target[shown]) == 0:
                return None
            return float(np.corrcoef(target[shown], g_[shown])[0, 1])
        r = r_shown(grid)   # over the squares the person was shown, the controls included
        nulls = []   # the same scoring on misaligned EEG: what chance gives
        for k in range(8):
            try:
                g0, _ = reconstruct.run(self.session, calibration=calib_arg, stim_freq=self.stim_freq,
                                        method=method, vblend=False, save=False,
                                        shift_s=23.0 * (k + 1) + 7.0)
                r0 = r_shown(g0)
                if r0 is not None:
                    nulls.append(r0)
            except Exception as exc:
                print("null:", repr(exc)[:120])
                break
        null_max = max(nulls) if nulls else None''')
rep('''        result.update(ok=True, matter=dict(
            found={k: v for k, v in found.items() if k != "overlay_data_url"}, r=r, null_r=nulls,''',
     '''        result.update(ok=True, matter=dict(
            found={k: v for k, v in found.items() if k != "overlay_data_url"}, r=r, r_full_grid=r_full,
            method=method, null_r=nulls,''')
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("xr_session patched")
