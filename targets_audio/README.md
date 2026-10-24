# Audio targets for the Ear-as-a-Microphone runs

Public-domain sound that can be published with the paper, unlike a song.

## alice_ch1_rabbit_hole.wav

The first two paragraphs of *Alice's Adventures in Wonderland* (Lewis Carroll,
1865), chapter 1, "Down the Rabbit-Hole", from "Alice was beginning to get very
tired of sitting by her sister on the bank" to "when suddenly a White Rabbit
with pink eyes ran close by her." 31.5 s, mono, 44.1 kHz, 16 bit.

Read by Kristen McQuillin for LibriVox (2006), public domain. Source: Internet
Archive item `alice_in_wonderland_librivox`, file `wonderland_ch_01.mp3`,
16.0 s to 47.5 s (the LibriVox preamble and the chapter heading are cut off).
https://archive.org/details/alice_in_wonderland_librivox
https://librivox.org/alices-adventures-in-wonderland-by-lewis-carroll/

Run:

    python ear_mic_session.py --song targets_audio/alice_ch1_rabbit_hole.wav --start 0 --clip 31.5 --reps 24 --session runs/ear_mic1
    python analysis/ear_mic_decode.py runs/ear_mic1

Envelope modulation depth (analysis/ear_mic_pick_clip.py) 0.40, against 0.46
for a dance remix and 0.09 for a choir piece.
