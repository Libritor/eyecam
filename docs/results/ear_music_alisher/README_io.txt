audio_examples: INPUT (what the participant heard) vs OUTPUT (what the model produced)
format: 16000 Hz, 1 ch, 16-bit PCM; every clip is 3.84 s

INPUT  = the *_REAL.wav clip for each example (the stimulus played to participant P00, session 1)
OUTPUT = the generated clip for the same example under each condition:
  ControlNet_EEG     - EEG-conditioned model (the real system)
  No-EEG_baseline    - same generator with no EEG input
  Classify-then-play - classify the EEG, then play a stored clip for that class
  Label-shuffle_null - EEG labels shuffled (null control)

Timeline shared by INPUT_all_heard.wav and every OUTPUT_<condition>.wav:
  00:00.00  block02_real_P00_s1  ex0_fast_synco
  00:05.34  block02_real_P00_s1  ex1_fast_straight
  00:10.68  block02_real_P00_s1  ex2_M2
  00:16.02  block03_real_P00_s1  ex0_slow_synco
  00:21.36  block03_real_P00_s1  ex1_slow_synco
  00:26.70  block03_real_P00_s1  ex2_slow_straight

Timeline of PAIRED_input_then_<condition>.wav (same for all four):
  00:00.00 input  ->  00:04.34 output   block02_real_P00_s1  ex0_fast_synco
  00:09.68 input  ->  00:14.02 output   block02_real_P00_s1  ex1_fast_straight
  00:19.36 input  ->  00:23.70 output   block02_real_P00_s1  ex2_M2
  00:29.04 input  ->  00:33.38 output   block03_real_P00_s1  ex0_slow_synco
  00:38.72 input  ->  00:43.06 output   block03_real_P00_s1  ex1_slow_synco
  00:48.40 input  ->  00:52.74 output   block03_real_P00_s1  ex2_slow_straight

Identical files (same MD5) found in the zip:
  block02_real_P00_s1\ex0_fast_synco_Classify-then-play.wav  ==  block02_real_P00_s1\ex0_fast_synco_REAL.wav
  block02_real_P00_s1\ex0_fast_synco_No-EEG_baseline.wav  ==  block03_real_P00_s1\ex0_slow_synco_No-EEG_baseline.wav
  block02_real_P00_s1\ex1_fast_straight_Classify-then-play.wav  ==  block02_real_P00_s1\ex1_fast_straight_REAL.wav
  block02_real_P00_s1\ex1_fast_straight_No-EEG_baseline.wav  ==  block03_real_P00_s1\ex1_slow_synco_No-EEG_baseline.wav
  block02_real_P00_s1\ex2_M2_No-EEG_baseline.wav  ==  block03_real_P00_s1\ex2_slow_straight_No-EEG_baseline.wav
  block03_real_P00_s1\ex0_slow_synco_Classify-then-play.wav  ==  block03_real_P00_s1\ex1_slow_synco_Classify-then-play.wav
  block03_real_P00_s1\ex0_slow_synco_REAL.wav  ==  block03_real_P00_s1\ex1_slow_synco_REAL.wav  ==  block03_real_P00_s1\ex2_slow_straight_Classify-then-play.wav

Per-example map (example -> input file -> output files):
  block02_real_P00_s1 / ex0_fast_synco
    input : input/block02_ex0_fast_synco__INPUT_heard.wav   (md5 72b5ab9d)
    output: output/block02_ex0_fast_synco__OUTPUT_ControlNet_EEG.wav   (md5 a271d12c)
    output: output/block02_ex0_fast_synco__OUTPUT_No-EEG_baseline.wav   (md5 1420872e)
    output: output/block02_ex0_fast_synco__OUTPUT_Classify-then-play.wav   (md5 72b5ab9d)  <-- IDENTICAL to input
    output: output/block02_ex0_fast_synco__OUTPUT_Label-shuffle_null.wav   (md5 9756da71)
  block02_real_P00_s1 / ex1_fast_straight
    input : input/block02_ex1_fast_straight__INPUT_heard.wav   (md5 59506e4a)
    output: output/block02_ex1_fast_straight__OUTPUT_ControlNet_EEG.wav   (md5 5d2eed9b)
    output: output/block02_ex1_fast_straight__OUTPUT_No-EEG_baseline.wav   (md5 bb425966)
    output: output/block02_ex1_fast_straight__OUTPUT_Classify-then-play.wav   (md5 59506e4a)  <-- IDENTICAL to input
    output: output/block02_ex1_fast_straight__OUTPUT_Label-shuffle_null.wav   (md5 5b799bf5)
  block02_real_P00_s1 / ex2_M2
    input : input/block02_ex2_M2__INPUT_heard.wav   (md5 750ff87d)
    output: output/block02_ex2_M2__OUTPUT_ControlNet_EEG.wav   (md5 1b089a68)
    output: output/block02_ex2_M2__OUTPUT_No-EEG_baseline.wav   (md5 0cd9d0b2)
    output: output/block02_ex2_M2__OUTPUT_Classify-then-play.wav   (md5 29730475)
    output: output/block02_ex2_M2__OUTPUT_Label-shuffle_null.wav   (md5 c0e5784a)
  block03_real_P00_s1 / ex0_slow_synco
    input : input/block03_ex0_slow_synco__INPUT_heard.wav   (md5 7d890c6c)
    output: output/block03_ex0_slow_synco__OUTPUT_ControlNet_EEG.wav   (md5 f88c5735)
    output: output/block03_ex0_slow_synco__OUTPUT_No-EEG_baseline.wav   (md5 1420872e)
    output: output/block03_ex0_slow_synco__OUTPUT_Classify-then-play.wav   (md5 29d56e0c)
    output: output/block03_ex0_slow_synco__OUTPUT_Label-shuffle_null.wav   (md5 ed42d82f)
  block03_real_P00_s1 / ex1_slow_synco
    input : input/block03_ex1_slow_synco__INPUT_heard.wav   (md5 7d890c6c)
    output: output/block03_ex1_slow_synco__OUTPUT_ControlNet_EEG.wav   (md5 36fbdbb6)
    output: output/block03_ex1_slow_synco__OUTPUT_No-EEG_baseline.wav   (md5 bb425966)
    output: output/block03_ex1_slow_synco__OUTPUT_Classify-then-play.wav   (md5 29d56e0c)
    output: output/block03_ex1_slow_synco__OUTPUT_Label-shuffle_null.wav   (md5 e300829d)
  block03_real_P00_s1 / ex2_slow_straight
    input : input/block03_ex2_slow_straight__INPUT_heard.wav   (md5 ae8997dc)
    output: output/block03_ex2_slow_straight__OUTPUT_ControlNet_EEG.wav   (md5 0e04a05d)
    output: output/block03_ex2_slow_straight__OUTPUT_No-EEG_baseline.wav   (md5 0cd9d0b2)
    output: output/block03_ex2_slow_straight__OUTPUT_Classify-then-play.wav   (md5 7d890c6c)
    output: output/block03_ex2_slow_straight__OUTPUT_Label-shuffle_null.wav   (md5 23186372)
