"""Ali Sher's EEG-to-music clips: how closely each condition's output follows the rhythm the
listener heard. Onset strength (positive spectral flux of the log STFT, 512-sample frames,
hop 160 at 16 kHz = 10 ms), z-scored; similarity = the largest Pearson r between the heard
clip's onset curve and the output's within +-60 ms of lag. Conditions: ControlNet_EEG (the
real system), No-EEG_baseline, Classify-then-play, Label-shuffle_null.

    python analysis/ali_music_similarity.py docs/results/ear_music_alisher
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft

CONDS = ["ControlNet_EEG", "No-EEG_baseline", "Classify-then-play", "Label-shuffle_null"]


def onset(path):
    x, fs = sf.read(path)
    x = x.mean(1) if x.ndim > 1 else x
    _, _, Z = stft(x, fs=fs, nperseg=512, noverlap=352)
    S = np.log1p(np.abs(Z))
    fl = np.maximum(np.diff(S, axis=1), 0).sum(0)
    return (fl - fl.mean()) / (fl.std() + 1e-9)


def main():
    root = sys.argv[1]
    res = {c: [] for c in CONDS}
    names = []
    for real in sorted(glob.glob(os.path.join(root, "clips", "*", "*_REAL.wav"))):
        base = real[:-len("_REAL.wav")]
        names.append(os.path.basename(base))
        r = onset(real)
        for c in CONDS:
            o = onset(base + "_" + c + ".wav")
            n = min(len(r), len(o))
            res[c].append(round(float(max(np.corrcoef(np.roll(o[:n], k), r[:n])[0, 1] for k in range(-6, 7))), 3))
    wins = sum(a > b for a, b in zip(res["ControlNet_EEG"], res["Label-shuffle_null"]))
    out = dict(examples=names, similarity=res, mean={c: round(float(np.mean(v)), 3) for c, v in res.items()},
               controlnet_beats_shuffle=f"{wins}/{len(names)}", sign_test_p_one_sided=round(0.5 ** len(names) * sum(
                   __import__("math").comb(len(names), k) for k in range(wins, len(names) + 1)), 4),
               note="Classify-then-play outputs for both fast examples are the heard file itself (MD5), hence 1.0")
    json.dump(out, open(os.path.join(root, "similarity.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
