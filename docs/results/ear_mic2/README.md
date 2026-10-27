# Ear as a microphone, session 2 (the reported one)

The first two paragraphs of *Alice's Adventures in Wonderland* (LibriVox, read by
Kristen McQuillin, public domain), 31.5 s, played 24 times from the Quest 3S's own
speakers through its browser (`quest_audio.py`), 2.5 s apart with up to 1 s of
jitter, at 11:57 on 17 October 2026. The listener wore a Muse S with the AUX
electrode on top of the head: TP9, AF7, AF8, TP10 and AUX at 256 Hz, streamed from
the phone. Recorded by `ear_mic_session.py`, decoded by
`analysis/ear_mic_decode.py` (defaults: 1 to 8 Hz, lags -125 to +500 ms, ridge,
leave-one-play-out, the first and last 3 s of every play left out of the scoring,
25.5 s decoded per play).

## Numbers

| measure | result | null and p |
|---|---|---|
| averaged reconstruction, 23 held-out plays | **r 0.357** | shuffled null 95th percentile 0.097, p 0.0008 |
| one play, mean over the 24 | r 0.086 (sd 0.059) | null 95th 0.022, p 0.0008 |
| control: EEG rolled off the clips | r 0.035 | p 0.29, chance |
| each electrode alone, averaged | TP10 0.20, AF8 0.16, AUX 0.15, TP9 0.14, AF7 0.02 | Cz-M (AUX minus the mastoid mean) 0.15 |
| reference, 64-channel research caps, one play of continuous speech | 0.10 to 0.19 | linear 0.10 to 0.12, deep networks 0.19 |

The night before (`../ear_mic1`, top electrode off, tired subject) the same
protocol gave r 0.18 with a control of 0.078, p 0.044.

## The sounds

The decoder gives back loudness only, so each envelope is put on a carrier to be
heard. `loudness = max(mu + sd * z, 0)` with mu and sd from the true positive
envelope.

| file | what it is |
|---|---|
| `clip.wav` | the clip as played, 31.5 s at 48 kHz |
| `published/clip.wav` | the clip cut to the 25.5 s the decoder scores |
| `true_envelope_noise.wav` | the clip's true loudness on white noise: the most a perfect read-out could give back |
| `brain_envelope_noise.wav` | the loudness read from the EEG on the same noise: the reconstructed audio |
| `brain_envelope_carrier.wav` | the brain's loudness on the clip's own sound: the words are the recording with its own loudness flattened, only the rise and fall is from the EEG |
| `published/env.json` | the two envelopes (true and read from the EEG) sampled on the scored 25.5 s, as drawn on the listening page |
| `player.html` | a stand-alone listening page for the four sounds |
| `ear_mic_figure.pdf`, `.png` | panel a the two envelopes; b each play against the shuffled null with the shifted-EEG control as crosses; c each channel alone; d the averaged EEG against the envelope by lag |
| `ear_mic_results.json` | every number above with the per-play values |
| `stimulus.json`, `events.csv` | the session's timing: clip, repetitions, the Quest clock sync, play onsets |

The raw EEG (`runs/ear_mic2/eeg.csv`, 21 MB) is not in the repository, as for the
eye sessions.

## What is still open

The strongest electrodes sit beside the ears, where an auditory response should be
and also where the headset's speakers are, so part of the signal could be the sound
induced into the electrodes rather than the brain following it. The control that
settles it is the same session with the headband on the desk under the headset and
nobody wearing it. Two such recordings exist (`runs/ear_mic_desk`, 01:56 the same
night, and `runs/ear_mic_desk2`, 11:44 the same morning) but neither is usable: the
first lost the stream six times and the second logged only three plays, and the
decoder needs at least four complete ones. The desk control therefore has not been
run, and the r 0.36 above should be read with that caveat until it is.
