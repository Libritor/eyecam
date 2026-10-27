# Ear as a microphone, music: Ali Sher's EEG-to-music clips

Six short rhythm clips, 3.84 s each at 16 kHz mono, were played to one listener
(participant P00, session 1, blocks 02 and 03) in Ali Sher's EEG-to-music
experiment. The listener's EEG goes into a music generator as a control signal
and the generator writes a clip of its own. For every clip the set holds what the
listener heard and four outputs:

| condition | what it is |
|---|---|
| `ControlNet_EEG` | the EEG-conditioned generator, the real system |
| `No-EEG_baseline` | the same generator with its control input removed |
| `Classify-then-play` | a classifier names the clip's rhythm class from the EEG and a stored example of that class is played; no generation |
| `Label-shuffle_null` | the EEG labels shuffled before generating: what chance gives |

The clips arrived as `audio_examples.zip` on 15 October 2026 with no scores
beside them, so this folder is for listening. The question the set asks is whether
the EEG-conditioned output resembles the heard clip more than the no-EEG and
shuffled-label outputs do; a listening test or a rhythm-similarity score is the
next step and has not been done.

## Files

- `clips/block02_real_P00_s1/`, `clips/block03_real_P00_s1/`: the thirty original
  clips as delivered, `ex<n>_<rhythm>_REAL.wav` is what was heard and the four
  `ex<n>_<rhythm>_<condition>.wav` are the outputs.
- `INPUT_all_heard.wav`: the six heard clips in a row, 1.5 s apart.
- `OUTPUT_<condition>.wav`: the six outputs of one condition in the same order and
  spacing, so any two of these files line up sample for sample with the input.
- `PAIRED_input_then_<condition>.wav`: each heard clip followed half a second later
  by its output under that condition, 1.5 s between examples; the one to start
  with is `PAIRED_input_then_ControlNet_EEG.wav`.
- `README_io.txt`: the timelines of every stitched file, the per-example map with
  MD5s, and the list of identical files.

## Order of the examples

| # | example | block | starts in the single files | heard / output in the paired files |
|---|---|---|---|---|
| 1 | fast, syncopated | 02 | 0:00.0 | 0:00.0 / 0:04.3 |
| 2 | fast, straight | 02 | 0:05.3 | 0:09.7 / 0:14.0 |
| 3 | M2, as labelled | 02 | 0:10.7 | 0:19.4 / 0:23.7 |
| 4 | slow, syncopated | 03 | 0:16.0 | 0:29.0 / 0:33.4 |
| 5 | slow, syncopated, second | 03 | 0:21.4 | 0:38.7 / 0:43.1 |
| 6 | slow, straight | 03 | 0:26.7 | 0:48.4 / 0:52.7 |

## A caveat about the set

An MD5 check of the zip found several files that are the same bytes (the full
list is in `README_io.txt`). For both fast examples the classify-then-play output
is the heard clip itself. The no-EEG baseline uses one file for the fast and the
slow version of each rhythm. The two slow syncopated clips the listener heard are
one file, and that file also serves as the classify-then-play output for the slow
straight example. So between examples only the EEG-conditioned output and the
shuffled-label null differ every time, and those two are the pair to compare
against the input.

Credit: the experiment, the model and the clips are Ali Sher's (Mersivity Lab,
University of Toronto); this folder only packages them beside the ear-as-a-
microphone decoding so the two read-outs of the ear sit together.
