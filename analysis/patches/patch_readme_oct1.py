"""2026-10-01: lead the README with the legible "NO". Run once from the repo root."""
import io

p = "README.md"
x = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global x
    assert x.count(a) == 1, (a[:60], x.count(a))
    x = x.replace(a, b)


rep('''![Colour flag and grey image reconstructed from EEG in VR](docs/figures/fig5_run2_images.png)

*Run of 2026-09-29, Quest 3S browser, Muse 2 with an Oz auxiliary electrode.
Bottom row: a red/green/blue flag shown with each colour flickering at its own
frequency, and the colour image rebuilt from three spectral lines in the EEG;
all 24 cells come back with the right dominant colour.*
''',
    '''![The word NO shown to the eye and read back from the EEG](docs/figures/fig0_NO_from_eeg.png)

**The word "NO", read back from brainwaves.** The subject looked at 45
positions in turn while a square flickered at 12 Hz wherever the picture was
white. The second panel is nothing but the strength of the 12 Hz line in the
EEG at each position. It correlates with the picture at **r = 0.91**, and an
automatic black/white threshold, which never sees the picture, gets **43 of
45 positions right**. The same reconstruction from time-shifted EEG reaches at
most r = 0.23.

*Run of 2026-10-01: Quest 3S browser, Muse (2016) with an auxiliary electrode
at Oz, 8 s per position, three passes, 18 minutes of scanning. A second run
the same day with a larger square and two passes gave r = 0.87 and again 43 of
45. One subject, one day.*

| after | pass 1 | passes 1-2 | passes 1-3 |
|---|---|---|---|
| correlation with the picture | 0.79 | 0.88 | 0.91 |
| positions right after thresholding | 38 of 45 | 43 of 45 | 43 of 45 |

What made the difference, after a month of images that were only weakly above
chance, was getting four ordinary things right at once:

1. **A picture that is a picture at scan resolution.** A word in a scalable
   font, squeezed onto 48 positions, is a grey blob before any brain is
   involved. The target is now a 4 by 5 pixel font, one position per pixel.
2. **The paper's own scoring.** One spectrum over the whole dwell, power at
   the flicker frequency plus its harmonic, divided by the 14 to 50 Hz power.
3. **No vertical blend.** The paper's Eq. 1 is for overlapping scan lines; on
   discrete positions it only smears rows together.
4. **Repeat passes**, averaged per position.

![Colour flag and grey image reconstructed from EEG in VR](docs/figures/fig5_run2_images.png)

*Run of 2026-09-29, same setup. Bottom row: a red/green/blue flag shown with
each colour flickering at its own frequency, and the colour image rebuilt from
three spectral lines in the EEG; all 24 cells come back with the right
dominant colour.*
''')

rep('''- **An exact-frequency line detector** with an exact permutation test, which is
  about eight times more sensitive than 1 Hz-bin band power and is what first
  found the response on a Muse.''',
    '''- **A legible grey image on consumer hardware**: the paper's whole-dwell
  power ratio as the pixel value, a pixel-font target, repeat passes, and a
  scan that pauses and redoes a position when the EEG stream drops.
- **An exact-frequency line detector** with an exact permutation test, which is
  about eight times more sensitive than 1 Hz-bin band power and is what first
  found the response on a Muse. It decides the calibration gate; the image
  itself is scored with the paper's ratio.''')

rep('''- **Line detector.** For each block or scan position, one Hann periodogram over
  the whole window; the score is the power at the delivered flicker frequency
  divided by the power 0.5 to 2 Hz either side.''',
    '''- **Pixel value (the paper's ratio).** For each scan position, one Hann
  periodogram over the whole dwell; the pixel is the power at the flicker
  frequency plus its first harmonic, divided by the power from 14 to 50 Hz
  with the signal bins and the mains line left out.
- **Line detector.** For each calibration block, one Hann periodogram over
  the whole window; the score is the power at the delivered flicker frequency
  divided by the power 0.5 to 2 Hz either side.''')

rep('''subject first. Resolution has to come from more overlapping positions of a
large square and longer dwell, as in the paper, not from shrinking the square.''',
    '''subject first. Shrinking the square does not buy resolution, and on the one
day it was tried a larger square did not buy signal either; what moved the
grey image from weakly above chance to legible was a target that survives the
scan grid, the paper's scoring, no vertical blend, and repeat passes. Nearly
all of the image comes from the Oz electrode: on its own it gives r = 0.92,
the ear electrodes alone give 0.36.''')

rep('''- Large square on a fine, overlapping grid with 8 to 10 s per position, and two
  passes averaged, to make the grey image legible.
''',
    '''- Repeat the legible "NO" on another day and another subject, then scale it:
  more letters, more positions, grey levels (`pixgrad:`), and a real
  photograph.
''')

rep('''- Audio with a vertex electrode and earbuds.
''',
    '''- Audio with a continuous 40 Hz modulated tone or click train; tagged music
  through headphones with a vertex electrode gave nothing.
''')

io.open(p, "w", encoding="utf-8", newline="\n").write(x)
print("README patched")
