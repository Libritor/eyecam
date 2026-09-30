# EyeCam / EarMic forward plan — 2026-09-12

Scope: eye-as-camera (EyeCamXR, Quest 3S + Muse) and the ear-as-microphone extension. Every number below comes from the five decoder reports, the audit, and the four research reports; claims marked `survives=false` in adversarial verification are replaced by their corrected forms.

---

## 1. What the live data says

### 1.1 Session facts (xr_live6 = the only live human session with clean contact during calib+scan)

| Quantity | Value | Source |
|---|---|---|
| Calibration d' (TP9 / AF7 / AF8 / TP10) | −0.322 / −1.193 / **+1.056** / −0.164; `passed=False`; weights AF8 = 1.0, ears = 0 | calibration.json, reproduced to 4 decimals by 4 scripts |
| Is AF8 d' = 1.06 real? | exact permutation p = 0.210 one-sided (924 relabelings), best-of-4 family-wise **p = 0.389**, bootstrap 95% CI [−1.30, 7.29], MWU p = 0.120; null 95th pct of max d' = 2.34 | quality_stats.py |
| Control arm (EEG rolled 1/3, timestamps fixed) | TP9 d' = **1.33** (p = 0.045); in baseline_preproc the same control reached 1.84 and **passed the calibration gate** | baseline_preproc.py, quality_stats.py |
| 15 Hz line in ON-block PSD | local maximum at 15 Hz: False on all 4 channels; AF8 ON/OFF at 15 Hz = 1.60 but flanks (13–13.5, 16.5–17 Hz) = 1.27 → stimulus-specific ratio only 1.25 (broadband bright-vs-dark contrast, not a line) | quality_stats.py |
| Reconstruction r vs target | **0.189** (0.192–0.194 on the full file); 1/3-shift control −0.272; circular-shift null mean 0.026, sd 0.251, **p = 0.36**; 11-shift null z = 1.0; dense 452-shift null z = 0.97, frac ≥ real 0.21 | all reports + audit |
| Image = time drift | r(grid, scan row) = **+0.73** (row ≡ time: r(gy, t) = 0.993); target itself position-uncorrelated (r_row −0.02) | gates report |
| Frequency specificity | control bins 11/13/17/20 Hz give r = 0.22 / 0.00 / **0.28** / 0.09 → margin r(15) − max r(control) = **−0.085** (17 Hz beats 15 Hz) | gates report |
| Split-half reliability | 0.067 (0.10 in gates run) — noise autocorrelation | quality_stats.py |
| Lock-in sensitivity (injected stimulus-locked 15 Hz into real live6 background) | detected at 0.25 µV (full-window coherent), 0.5 µV (τ = 1 s), 1 µV (Welch) → **no stimulus-locked 15 Hz response ≥ ~0.25 µV exists at any Muse site** | lock_in.py |
| Ear contact | HF σ TP9 214 µV / TP10 113 µV (AF7 31 / AF8 23); 60 Hz line/floor TP9 4.18e4, TP10 1.66e4; rails 0 % during calib+scan; ears did not improve over the session; AF8 degraded during scan (HF σ 18.6 → 24.5, ρ = +0.46, p = 0.001) | quality_stats.py |
| Flicker | ON blocks 14.13, 14.99, 14.98, 14.97, 14.97, 14.97 Hz (block 0 ran at 85 fps); cells median 15.00 Hz; xr_session.json `measuredFlickerHz = 14.43` is pulled down by block 0. Rescoring at measured Hz: AF8 d' 0.73 (p = 0.209) — no rescue | quality_stats.py, act_chirplet.py, cca.py |
| Timestamps | UDP arrival stamps burst-clumped: p50 jitter 77 ms, p99 145 ms, 98.5 % of samples off by > 1/4 flicker period; sample-index clock rebuild restores 60 Hz mains coherence 56 % → 93 % | lock_in.py, clock_check.py |

Note on the audit's root-cause hypothesis ("14.43 Hz lands in the 14 Hz bin"): the per-block measurement contradicts it — five of six ON blocks and all scan cells ran at 15.00 Hz; only block 0 was off, and dropping/rescoring it does not rescue the contrast. The signal is absent, not misbinned.

Controls: webgate phantom d' 18.82 / 13.46 / 3.54 / 18.84, r = 0.782–0.796 (control −0.02 to −0.05, z 4.2–4.9) — pipeline detects SSVEP when present. gate_full phantom r = 0.849, shift-null p = 0.005, permutation p = 0.0011. xr_live2 (floating electrodes) r = −0.388, best d' −0.24, TP9 rails 36.8 %, split-half 0.39 → a *reliable artifact*. xr_live4: best channel AF7 uncorrected p = 0.014. Three live sessions, zero G1 passes.

Provenance hazards found during analysis (must be fixed before the next session): the recorder (`--linger 900`) kept appending to eeg.csv 13 min post-scan with TP9 at the 1450 rail (254k → 341k rows); `infer_fs` moved 256.72 → 257.30 Hz and AF7 d' moved −1.19 → −1.58 because the 14 Hz / 50 Hz band-edge Welch bins flip in and out of the denominator — the reference score is discretely sensitive to fs at the 3rd decimal; a sibling process rewrote `runs/xr_live6/reconstruction_grid.npy` with a 14.40 Hz variant (on-disk r now 0.214) and regenerated `runs/webgate` mid-analysis.

### 1.2 Decoder verdicts (xr_live6 real / control; webgate positive control)

| Decoder | live6 r (1/3-shift ctrl) | live6 best d' (ctrl) | webgate r | Verdict |
|---|---|---|---|---|
| Baseline Welch relative power (`reconstruct.cell_score`) | 0.189 (−0.272) | AF8 1.06 (TP9 1.33–1.84) | 0.782–0.796 | reference; inside its own null |
| Preprocessing variants: 60 Hz notch, 1–45 bp, CAR, AF8/TP-only/uniform weights, 2-s Welch | 0.12–0.32; null max up to 0.44 | 0.78–1.27; null max 1.62–2.42 | 0.78–0.80; **CAR 0.636** | SAME. CAR's +0.134 is the best of 11 post-hoc variants, r(shift) peaks at −40 s (0.39) not 0, and it damages the phantom |
| CCA 4-ch / FBCCA (Lin 2006 / Chen 2015) | 0.124 (0.058) / 0.022 (−0.040) | −0.73 / −1.72; single-ch TP9 1.42 but xr_live2 gives 1.31 by chance | 0.675 / 0.679 | WORSE-to-SAME |
| ACTv14Skew chirplet OMP | 0.100 (−0.134) | 0.39 (0.41) | 0.682 (Welch 0.782) | WORSE — free tc/Dt/fc/skew/chirp fit noise (OFF scores 0.06–0.19 vs Welch 0.05–0.09) |
| Software lock-in on rebuilt clock | 0.120 (0.42; 20-shift null 0.12 ± 0.17) | −0.14 (ctrl 1.93, passes gate) | 0.744 | SAME — 2–4× lower detection threshold, finds nothing |
| Row-detrend / linear-time-detrend | 0.133 / 0.300 (p 0.11 / 0.06) | — | 0.668 | FAIL G6 (Δ −0.06 / p ≥ 0.01) |

Audit (`analysis/audit_baseline_preproc.py`, `audit_null_sweep.py`): baseline reproduces bit-exactly, no leakage, no preprocessing variant legitimately improves r or detects calibration contrast. Conclusion: **the problem is acquisition (no SSVEP reaching TP9/TP10/AF7/AF8), not decoding.** No decoder work can change the verdict until G1 (§6) passes on a live session.

---

## 2. Decoder + preprocessing changes, ranked by evidence

Rank A = validated on the phantom positive control, fixes a demonstrated failure, and cannot manufacture signal. Rank B = infrastructure that is stimulus-independently validated. Rank C = only after G1 passes. Rank X = do not implement (evidence negative).

### Rank A — implement before the next live session

1. **Exact permutation calibration gate** (`run_session.score_calibration`, lines 279–340). Keep d'/ratio/rank as *reported* fields only. Decide `passed` by: statistic = mean(on) − mean(off) per channel, family-wise max over channels, null = all C(n_on+n_off, n_on) relabelings, **PASS iff p < 0.01**. ABSTAIN when 1/C(N, n_on) ≥ 0.01 (6/6 → 1/924; 5/5 → 1/252; 4/4 → 1/70 cannot decide; 3/3 → 0.05). Today: live6 p = 0.431, live4 > 0.014, live2 p = 0.66, phantom 0.0011. `CALIB_DPRIME_MIN = 1.0` would have admitted AF8 at p = 0.055 uncorrected. Ref: Nichols & Holmes max-statistic, https://www.fil.ion.ucl.ac.uk/spm/doc/papers/NicholsHolmes.pdf; Combrisson & Jerbi, https://pubmed.ncbi.nlm.nih.gov/25596422/.
2. **Circular-shift null in `reconstruct.run`**: ≥ 200 shifts of the EEG matrix by random offsets ≥ 20 s, same weights, same pipeline; p = (1 + #null ≥ r)/(N+1); PASS iff p < 0.01. Never the pixel-permutation null (sd 0.10 vs 0.25 on live6, p = 0.032 vs 0.36). Also record the 1/3-shift control r in xr_session.json.
3. **Frequency-specificity control** in `reconstruct.ssvep_score`: add `control_freq` re-runs at 11/13/17/20 Hz with the denominator = 14–50 Hz minus control bins minus a **±2 Hz guard around f0 and 2f0** (without the guard the phantom's control r reads −0.78/−0.68/−0.72/−0.72; with it −0.08/+0.12/+0.10/−0.06). PASS iff r(f0) − max r(fc) ≥ 0.2 and r(f0) > 0. Today: live6 −0.085 FAIL, phantom +0.73 PASS.
4. **Randomized cell visiting order** in `start_scan` (xr_stimulus.html; `cursor_log.csv` already records gx,gy per sample, so `reconstruct.py` needs no change) plus the check |corr(grid,row)| ≤ 0.3 and |corr(grid,col)| ≤ 0.3. This turns the drift-as-image confound (live6 r_row 0.73) from a check into a design guarantee, and blocks the block-design confound for any later learner (Li et al. TPAMI 2021, https://engineering.purdue.edu/~qobi/papers/tpami2021.pdf; Ahmed et al. CVPR 2021).
5. **No uniform-weight fallback into an image.** `score_calibration` line 330 falls back to uniform weights when no channel is admitted, and `xr_session.py` line 288 reconstructs anyway unless `--require-pass`. Make the scan and reconstruction ABSTAIN when G1 is not PASS (default on, not opt-in). The lock-in and ACT runs both produced live6 "images" from this fallback.
6. **Session-window snapshot + recorder stop.** Stop the recorder on `scan_done` (not `--linger 900`); write sha1 + row count of eeg.csv, calib_log.csv, cursor_log.csv into xr_session.json; every offline script must slice rows to t ≤ stage end + 0.7 s (the rule that reproduced stored d' to 3 decimals and r = 0.1888). Add `runs/<session>/FROZEN` marker; no script may write into a frozen run (the 08:36 sibling rewrite is the counter-example).
7. **Band-edge robustness** in `ssvep_score`: the 14 Hz and 50 Hz edge bins flip membership with a 0.2 % fs change. Build the denominator mask with a half-bin margin (`freqs > lo + 0.5*df`, `freqs < hi_eff − 0.5*df`) and re-run gate_full/webgate as no-regression (phantom must stay ≥ 0.6; expect ≈ 0.78–0.85).
8. **Per-block measured flicker** in calibration.json (rising-edge period per ON block from calib_log); drop blocks with |f − f0| > 0.3 Hz (block 0 at 14.13 Hz; xr_live2 block 6 at 11.59 Hz). Store per-block Hz, not a session mean.

### Rank B — infrastructure, stimulus-independently validated

9. **Sample-index clock** (from `analysis/lock_in.py`/`clock_check.py`): per contiguous segment, lower-envelope line fit of arrival time vs sample index, ±3 s rolling-min residual smoothed over 5 s; chosen by the 60 Hz mains metric (93 % vs 56 % coherence at 6 s; 94 % linear over 84 s), not by outcome. Use it in `reconstruct.infer_fs`/segmentation. It changes nothing for 1-s Welch (< 0.06 cycle at 15 Hz) but is the precondition for any coherent detector and for the G7 latency window.
10. **Lock-in as a sensitivity instrument, not a decoder.** Ship `lock_in.py`'s injection curve as a per-session report: "smallest stimulus-locked amplitude this session would have detected" (live6: 0.25 µV full-window, 0.5 µV τ = 1 s, 1 µV Welch). This converts a null result into a bound the paper can print.

### Rank C — only after G1 PASS on ≥ 1 live session

11. Learned decoders under the G8 sealed protocol (§6). First candidate: EEGNet-class compact CNN on 1-s windows, binary ON/OFF target at the block frequency, trained on calibration blocks of *train* sessions only (Waytowich et al. 2018, https://iopscience.iop.org/article/10.1088/1741-2552/aae5d8).
12. Trained/spatially filtered CCA (TRCA-style) — needs a calibration that contains SSVEP (cca.py note).

### Rank X — negative evidence, do not implement as primary path

- CAR (phantom 0.796 → 0.636, spreads SSVEP into AF7/AF8); 60 Hz notch and 1–45 Hz band-pass (no effect: scoring band already excludes mains); 2-s Welch; TP-only/uniform weights (inside their nulls); CCA/FBCCA as primary detector; ACT on a weak stationary tone (the fixed-bin periodogram is already near the matched filter); row/time detrend as a "decoder".
- Synthetic-injection sensitivity for CCA vs PSD ratio is a reasonable side experiment (cca.py note) but does not change acquisition priority.

### ASSR-path code fixes (`--mode assr` exists: xr_session.py lines 319–348, xr_stimulus.html lines 214–283)

- `modDepth` gain is 0.5 → set AM depth to 1.0 (Picton 2003: amplitude grows with modulation depth, https://pubmed.ncbi.nlm.nih.gov/12790346/).
- `score_calibration(..., stim_freq=40.0)` uses `NOISE_BAND = (14, 50)`: in a combined session the 15/30/45 Hz visual lines sit in the ASSR denominator, and 45 Hz is 5 Hz from 40 Hz. Add a `noise_bins` argument: neighbours ±2..10 Hz around f, excluding 29–31, 44–46 and 59–61 Hz; use ≥ 2-s windows (0.5 Hz bins) for the 40 Hz path.
- Add `--mode both` (visual 15 Hz + auditory 40 Hz concurrently) and a level-coding block (§4).
- Log the audio output level; calibrate SPL once with a meter and store it in xr_session.json.

---

## 3. Signal-quality protocol changes

### 3.1 Live badge thresholds (from quality_stats.py [H]; computed per 10-s window, per channel, from the raw stream)

| Class | Rule | Live6 today | Live2 today |
|---|---|---|---|
| HARD (block scan) | rail fraction (samples at 0 or 1450) < 1 % | 0 % all ch (pass) | TP9 36.8 %, TP10 8.1 %, AF7 1.01 % (fail) |
| HARD | HF σ < 100 µV | TP9 214, TP10 113 **fail**; AF7/AF8 pass | TP9 693, TP10 360 fail |
| SOFT (warn, down-weight) | HF σ < 40 µV; 60 Hz line/floor < 1e4 (0.5 Hz Welch, floor = median 45–55 & 65–75 Hz); std < 60 µV | AF7/AF8 pass all; ears fail all | forehead fails std |
| STIMULUS | every ON block within f0 ± 0.3 Hz; `slowFrames/totalFrames < 1 %`; lock Quest refresh, log fps | block 0 fails (14.13 Hz) | block 6 fails (11.59 Hz) |
| CALIBRATION (proceed-to-scan) | family-wise permutation p < 0.05 **and** real best d' > 1/3-shift control best d' + 1; requires ≥ 5+5 blocks | p = 0.389, ctrl 1.33 > real 1.06 → fail | fail |
| CLAIM (paper) | G1 p < 0.01 (§6) | fail | fail |

Rail values on MuseLog are 0 and 1450 (not 1682). The phantom cannot validate contact thresholds (no impedance physics); it only shows the rules don't reject clean data (4/4 pass).

### 3.2 Electrode / setup protocol (before every live session)

1. Ears first — the ASSR dipole is frontocentral-vs-mastoid, so TP9/TP10 are the channels both the camera and the microphone need, and they carried nothing in live6 (weights 0). Alcohol wipe + saline/water on the rubber ear sensors, clear hair, 2-min settle; log Muse HSI per second and require HSI = 1 on TP9/TP10 for the whole calibration; exclude HSI > 1 segments. Literature caveat to state in the paper: 17 of the Muse recordings in the actiCHamp comparison were non-physiological at TP9/TP10 (https://pmc.ncbi.nlm.nih.gov/articles/PMC11679099/).
2. **Aux Oz electrode** via the Interaxon aux cup (checklist presets PRESET_20 / PRESET_1022), as in Mann et al. HealthCom 2019 (http://wearcam.org/eyecam.pdf). This is the single highest-leverage change: every published SSVEP-from-behind-the-ear result is weaker than Oz, and the live sessions found nothing at any stock site.
3. Chromatic stimulus for the behind-ear channels (Floriano, Diez, Bastos-Filho, Sensors 2018, https://doi.org/10.3390/s18020615: > 80 % CCA detection at TP9/TP10 with suitable colour/frequency). The research reports disagree on the exact colour pair/frequency (one says green-blue 30–40 Hz, another green-red at medium frequencies) — read the primary before choosing.
4. Calibration design: 8–10 ON/OFF blocks of 10 s (not 6 × 7 s), so G1 is decidable at p < 0.01 with margin; darkened room; chin rest or fixed head; in-ear or open-back headphones so ear cups do not lift TP9/TP10.
5. Muse stream check on the phantom before any ASSR session: confirm no onboard low-pass/notch attenuates 40–80 Hz (spec only states 256 Hz / 12-bit, https://ifelldh.tec.mx/sites/g/files/vgjovo1101/files/Muse_2_Specifications.pdf).
6. Do not use 20 Hz flicker (2nd harmonic = 40 Hz aliases into the ASSR bin); keep 15 Hz (harmonics 30/45/60; discard 60 Hz = mains).

---

## 4. Combined eye-camera + ear-microphone protocol

### 4.1 What is decodable on a Muse (honest statement for the paper)

- **Decodable (physical measurement):** presence and amplitude of a 40 Hz-tagged carrier band = a scalar level / modulation-depth readout per band, integration ~10–90 s per estimate (Picton et al. 2003: mean 22 s, range 2–92 s to p < 0.01 with a clinical montage, https://pubmed.ncbi.nlm.nih.gov/14570657/; expect longer with dry ear electrodes: ear-EEG detection ~20 % lower than scalp, thresholds elevated 0.8–7.5 dB, https://pmc.ncbi.nlm.nih.gov/articles/PMC6291863/). Up to 4 bands simultaneously via MASTER if carriers are ≥ 1 octave apart at ≤ 60 dB SPL (John et al. 1998, https://pubmed.ncbi.nlm.nih.gov/9547921/). Attention state (auditory vs visual) via ASSR/SSVEP amplitude shift (Saupe et al. 2009, https://pmc.ncbi.nlm.nih.gov/articles/PMC2791035/).
- **Marginal:** attended-talker selection at ≥ 60 s windows — 12 dry in-ear electrodes reach 61.1 %, 19 wet around-ear 67.2 %, 32 wet scalp 83.4 % (Geirnaert et al. 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12644565/); 4 dry Muse channels likely chance-to-60 %.
- **Not decodable:** speech content, melody, timbre, song identity, any waveform. Postolache et al. ICASSP 2025 needed 124 wet channels at 1 kHz and lifted CLAP only 0.38 → 0.60 on within-song test chunks (Pearson barely above the no-EEG generator; OOD song mostly non-significant), https://arxiv.org/pdf/2405.09062; Daly 2023 59.2 % EEG-only rank accuracy, https://repository.essex.ac.uk/34603/; Brain2Music is fMRI, https://arxiv.org/abs/2307.11078. A generative decoder trained on Muse data would mostly emit its prior.
- No peer-reviewed ASSR-on-Muse study exists. Nearest: 16-dry-electrode OpenBCI at 125 Hz shows a 40 Hz line during 2-min stimulation (https://www.biorxiv.org/content/10.1101/2025.03.14.639935). Closest "ear as level meter" template: Sergeeva, Christensen & Kidmose 2024 — 40 Hz AM on a 1 kHz sub-band of natural speech, level-coded ASSR from ear electrodes, 180 min per subject (https://pubmed.ncbi.nlm.nih.gov/38579741/).

Paper framing: "eye as camera + ear as level-meter / spectrum analyser" (SSVEP photometer + ASSR level meter, each with an explicit integration-time constant and calibration curve). "Ear as microphone" in the literal sense is not defensible.

### 4.2 Session design (per subject ~35 min; n ≥ 10; pre-register; blocks 60 s, ≥ 5 repeats, counterbalanced)

| Block | Stimulus | Purpose / gate |
|---|---|---|
| 0 | 60 s eyes-open rest in silence + 60 s eyes-closed | noise floor; alpha check; per-channel badge |
| 1 | Visual only: 15 Hz frame-locked square wave (Quest 3S: 6 frames @ 90 Hz or 8 @ 120 Hz), current calibration + scan | G1 (must pass before any scan), G2–G5 |
| 2 | Auditory only: 500 Hz (as in the current `runAssr`; Saupe used 500 Hz) or 1 kHz carrier, **100 % sinusoidal AM at 40 Hz**, ~60 dB SPL, binaural, 7–10 s ON/OFF blocks | GA1 (ASSR detected, exact permutation, same statistic as G1) + time-to-p < 0.01 per channel |
| 2b (optional) | MASTER: 0.5/1/2/4 kHz carriers at 37/39/41/43 Hz, ≤ 60 dB SPL | 4-band spectrum analyser; rates ≥ 1 Hz apart and away from 45 Hz |
| 3 | Both, attend visual (count brief flicker dimmings) | SSVEP unchanged by tone (Porcu et al. 2014: no cross-modal competition, https://www.sciencedirect.com/science/article/abs/pii/S1053811914002791) |
| 4 | Both, attend auditory (count brief AM gaps) | attention-modulation index ASSR_attendA − ASSR_attendV > 0 (Saupe 2009) |
| 5 | Level coding: 40 Hz tone at 30/40/50/60 dB SPL (or Sergeeva-style 5 dB steps on a 1 kHz speech sub-band) | ASSR amplitude vs level curve = the "level meter" calibration |

Analysis: 1–2 s FFT windows (1 Hz bins), Hanning; per-bin SNR = P(f) / mean P in ±2..10 Hz neighbours excluding 30/45/60 Hz; F-test / Hotelling T² across windows (Picton 2003 found the frequency-domain statistics equally efficient); Picton-style time-domain averaging over 25 ms epochs for faster detection; report time-to-p < 0.01 per subject and channel. Compare TP9-Fpz, TP10-Fpz, bipolar TP9-TP10, AF7/AF8 (expect frontals weak: Fpz sits near the ASSR maximum and partly cancels).

Frequency bookkeeping (256 Hz, Nyquist 128): visual 15/30/45/60/75/90, auditory 40/80/120, possible IM 25/55 (40 ± 15), 10/70 (40 ± 30), 5/85 (45 ∓ 40). Discard 60 Hz (mains). Pre-registered nulls: no power at 25/55 Hz IM bins for unrelated streams (Giani et al. 2012, https://pubmed.ncbi.nlm.nih.gov/22305992/); no reduction of the 15 Hz SSVEP when the tone is added; positive attention-modulation index. Phantom first: drive the phantom with summed 15 Hz + 40 Hz at realistic amplitudes (SSVEP ~1 µV, ASSR ~50–300 nV) to confirm the raw stream does not attenuate 40–80 Hz and that detection times match theory.

---

## 5. Paper sections (only claims that survived verification)

### 5.1 Retitle "The brain as an encryption machine" → "Security and privacy implications of an HMD eye-as-camera"

Position the brain as (a) a stimulus-response channel usable for freshness/presence checks and (b) a noisy, slowly drifting, low-entropy-rate biometric source — never as a key generator. Banned phrases: "unique", "unforgeable", "cannot be faked", "inherent liveness", "revocable".

Evidence that may be cited (with the corrected wording):

- **Permanence.** The most extensive longitudinal EEG-biometric study is Maiorana & Campisi (IEEE TIFS 2018): 45 subjects, 5–6 sessions over ~3 years, verification EER with aging degradation. CEREBRE (Ruiz-Blondet et al. 2016, https://ieeexplore.ieee.org/document/7435286; 2017 follow-up https://www.sciencedirect.com/science/article/abs/pii/S0167865517301940) reports 100 % *closed-set identification* of 20 returning participants at 48–516 days with 26–30-channel lab EEG and 50-trial averaged ERPs; Brainprint 2015's 82–97 % is a rank-based score with 50 % chance. Perfect closed-set ID among 50 certifies ≈ log₂(50) ≈ 5.6 bits of discriminability; none of these report EER or unknown-attacker verification, so none is evidence of cryptographic entropy. On the largest longitudinal set (345 subjects, 6,007 sessions, 5 years) EER goes 6.7 % (1 day) → 14.3 % (1 year), and a simulated Muse-2 4-channel montage gives 18.18 % EER (13.07 % with 4 samples) (https://arxiv.org/abs/2501.17866).
- **SSVEP as biometric.** Cross-session SSVEP identification reaches 92.92 % / 86.30 % with EER 3.92 % / 4.09 % (DRFNet, IEEE JBHI, https://pubmed.ncbi.nlm.nih.gov/40889321/) on two gel-electrode two-session datasets (30 subjects, 6-s trials; 54 subjects, 4-s trials), trained on the enrolled subjects. Deep models and occipital montages are not required: Piciucco et al. BIOSIG 2017 (https://ieeexplore.ieee.org/document/8053521/) reached 94.5–100 % cross-session (25 subjects, ~15 days apart) with MFCC/AR template matching, and frontal/central 7-electrode montages (94–96 %) beat occipital (~89–91 %). No dry-electrode cross-session SSVEP biometric exists; a ~4 % EER is ~4–5 bits per attempt, not key-grade.
- **Reproducibility.** NeuroIDBench (https://arxiv.org/abs/2402.08656): known-attacker EER 2.87 % → unknown-attacker 4.6 %; multi-session average EERs 21 % and 30 %; datasets mostly < 70 subjects, single session.
- **Consumer EEG.** Arias-Cabarcos et al. USENIX Security 2021: consumer-grade EER 8.5 % vs medical 1.9 %, 52 subjects (https://www.usenix.org/conference/usenixsecurity21/presentation/arias-cabarcos). Passthoughts (Berkeley): ~99 % only with per-subject task + threshold selection, N = 15, random split pooling two days (Chuang et al. 2013, https://people.ischool.berkeley.edu/~chuang/pubs/usec13.pdf); 99.82 % in-ear was N = 7 with custom-fit wet AgCl earpieces on an OpenBCI amplifier, single session (https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00354/full); consumer in-ear reached 72 %/80 % and is "not immediately viable" (https://biosense.berkeley.edu/biosense/files/2018/03/EMBC2016.pdf); the program's own paper says "passthoughts remains confined, for now, to the lab" (https://www.nspw.org/papers/2017/nspw2017-merrill.pdf).
- **Twins / task-invariance.** MZ twins distinguishable but markedly more similar; a person-identifying component is shared across tasks and weeks (https://pmc.ncbi.nlm.nih.gov/articles/PMC9553892/) — good for permanence, bad for revocability.
- **Fuzzy extractors.** Correct formal tool (Dodis et al., https://arxiv.org/abs/cs/0602007): key bits ≈ m − (n−k) − 2 log(1/ε); naive constructions leak under re-enrollment (Boyen, https://eprint.iacr.org/2004/358); high fuzzy min-entropy is not sufficient for some source families (Fuller-Reyzin-Smith, https://eprint.iacr.org/2014/961); the only realistic route is computational reusable extractors (Canetti et al., https://link.springer.com/article/10.1007/s00145-020-09367-8). No EEG key-generation paper reports a defensible min-entropy at cryptographic scale from multi-session data; "entropy 0.968" (KeyEncoder) is a normalized Shannon value, not bits.
- **Attacks.** Resting-state authenticators accept impostor recordings at FAR 0.784–0.896 (https://arxiv.org/html/2609.01856); hill-climbing with score access 100 % in < 10,000 attempts (Maiorana et al.); cross-modal inference raises EER 180–360 % (https://dl.acm.org/doi/10.1145/3374137); GAN/VAE EEG and post-mortem EEG undermine "inherent liveness" (https://pmc.ncbi.nlm.nih.gov/articles/PMC11824856/); consumer BCI P300 leaks PINs (Martinovic et al. 2012).
- **Freshness gate (the one HMD-novel, supported idea).** A fresh random code encoded in the flicker and verified by SSVEP decoding is a *freshness/presence* check layered on a separate identity factor, not a security primitive on its own. Kiser, Cantürk & Volosyak 2026 (https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1741655/full): 8 digits over a 3-symbol alphabet, 21 subjects, 95–99 % symbol accuracy with gel O1/Oz/O2 on a 360 Hz display — no EER, no cross-session, no attack test. The 2026 adversarial paper (https://arxiv.org/html/2609.01856) recommends fresh-stimulus challenge-response for RSVP/P300 only, unevaluated, and does not study SSVEP. The current eyecam stack (fixed 15 Hz, no code) does not implement it; an attacker who can see the flicker and write the EEG channel passes by emitting a sinusoid — which is exactly what `phantom_subject.py` does.
- **Key derivation, if included at all:** a design only — fuzzy extractor, raw EEG never stored, a worked entropy budget with the observed within/across-session noise showing the budget is small or negative, labeled future work.
- **Ethics/law paragraph (write now):** neural data is sensitive under Colorado HB 24-1058, California SB 1223, Montana SB 163, Connecticut, the OPC Canada PIPEDA bulletin (2026-02-10, https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/pipeda-compliance-help/pipeda-interpretation-bulletins/interpretations_10_sensible/) and UNESCO's Nov 2025 Recommendation; Chile v. Emotiv ordered deletion. REB consent must cover secondary identity/biometric use; store only sketches/features; deletion on request; no template sharing.
- **Minimum evidence bar** before any empirical biometric claim: ≥ 20 subjects, ≥ 2 sessions ≥ 7 days apart, verification metrics (EER, FRR at FAR 1 % / 0.1 %), unknown-attacker protocol, released code. Today's data meets none of it; do not analyze it for identity.

### 5.2 "Proof of humanness" → "Cortical challenge–response is not a one-way function: what an HMD brain-CAPTCHA can and cannot prove"

Correct framing (replaces the PUF framing, which did not survive): the scheme is an interactive challenge–response **test** for class membership (humanness) with a public verifier — a CAPTCHA in the sense of von Ahn, Blum, Hopper & Langford (EUROCRYPT 2003, https://link.springer.com/chapter/10.1007/3-540-39200-9_18), whose only admissible hardness assumption is a hard AI problem between challenge and accepting response. It is not a PUF: a PUF must be instance-unique and unpredictable, whereas a humanness oracle must be class-generic (and Strong PUFs were themselves introduced as "physical one-way functions"). If PUF literature is cited, the relevant model is the malicious/bad-PUF setting (Ostrovsky–Scafuro–Visconti–Wadia EUROCRYPT 2013; van Dijk–Rührmair), where the prover may substitute a simulator for the oracle. The dominant attack is zero-CRP synthesis/presentation (ISO/IEC 30107, https://www.iso.org/standard/83828.html) and relay/human-farm, not Rührmair's CRP-learning modeling attack.

**Proposition (survived, tightened):** if the verifier accepts iff S(c, r) ≥ τ where S is a public statistic whose maximizer over r is efficiently computable from c alone (true for CCA, template correlation, reconvolution likelihood — the maximizer is c ⊛ k for any plausible kernel k), and the adversary can read c and write r, then the adversary that outputs the template plus noise calibrated to the honest score distribution is accepted with probability ≥ the honest human (honest humans correlate only r ≈ 0.38–0.51 with the verifier's own linear forward model, Nagel & Spüler 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC6197660/). Soundness therefore requires at least one of: (i) a secret the adversary lacks — an enrolled kernel/latency, which is a low-entropy, non-revocable biometric transmitted in the clear in every honest response and recoverable from ~40–126 s of eavesdropped unlabeled EEG (Thielen et al. 2021, https://iopscience.iop.org/article/10.1088/1741-2552/abecef), so it changes the goal to proof-of-enrolled-identity; (ii) an authenticated channel from a physical sensor the adversary cannot write to (attested sensor; the Muse BLE/OSC link is not one); (iii) the challenge unobservable to anyone who can write the EEG channel until the response window closes (sealed, attested optical path), which additionally needs high-entropy fresh codes and rate limiting; or (iv) a hard AI problem between c and r — an ordinary CAPTCHA using EEG as the answer channel, inheriting CAPTCHA's breakage (Searles et al. USENIX Security 2023: bots 99.8 % in < 1 s vs humans 50–84 % in 9–15 s on distorted text, https://www.usenix.org/system/files/usenixsecurity23-searles.pdf). None of (i)–(iv) is "brain hardness".

**Linear-model statement (corrected form):** to first order the c-VEP/SSVEP response is a linear superposition of ~250–300 ms transient responses (Capilla et al. 2011, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014543; Thielen et al. 2015, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133797; Nagel & Spüler 2018), with documented second-order departures (rate-dependent adaptation, distinct short/long-flash kernels, harmonics/intermodulation); the linear model explains ~15–25 % of single-trial variance. Thielen 2021's zero-training result (81.3 % at 2.1 s vs 85.2 % supervised, converging to ~88 %) fits the kernel *unsupervised on the subject's own EEG* — it does **not** show that a generic cross-subject kernel passes a subject-specific verifier; that is untested. What follows: an *unenrolled* correlation/CCA verifier has no generate/verify asymmetry (both cheap); an *enrolled* verifier reduces to secrecy of a leaky low-entropy kernel. Existing SSVEP GANs (Aznan 2019; Kwon & Im 2022; TEGAN 2024) were validated only as augmentation, not as verifier-fooling forgeries.

**Our own stack (corrected form — Figure 1 caption):** `phantom_subject.py` is a white-box positive-control fixture that reads the stimulus state from the verifier's own calib/cursor logs and injects sin f + 0.4 sin 2f at author-chosen SNR over unauthenticated UDP; it passes every gate (r = 0.787–0.849 vs threshold 0.6). The single live human session was refused by the calibration gate (best d' 1.06, family-wise p = 0.39, control 1.33 ≥ real) and its post-hoc r = 0.19 lies inside the shift null; adding ~0.5–1.0 µV of stimulus-locked 15 Hz to that real background flips the gate (r = 0.72 at 1 µV). Report this as a **design-time no-liveness property plus a completeness failure (n = 1, gate-refused)** — not as "the machine out-performs the human" and not as an attack demonstration; no human/machine separation can be claimed in either direction yet. It was shown on the PC/browser OSC path; the on-device Quest BLE path is still [TODO] in `paper/eyecamxr/main.tex`.

Timing: Face Flashing's asymmetry (honest = speed of light, forgery slower; https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2018_03B-5_Tang_paper-updated.pdf) is inverted for cortex (honest ≈ 100 ms with single-trial SD ≈ 1.7 ms, Chen et al. PNAS 2015; forgery = microseconds), so latency bounds catch only relay, and only with enrolled latency.

Trust boundary on our hardware: the Quest 3S side can be attested (Meta Attestation API: nonce-bound token with app hash, bootloader lock, OS validity, device ID — attests nothing about sensors; https://developers.meta.com/horizon/documentation/native/ps-attestation-api/); the Muse side cannot (no secure element; reverse-engineered BLE, https://github.com/alexandrebarachant/muse-lsl; unauthenticated OSC on UDP 5000/5005). The one genuinely new HMD assumption: the challenge is a private optical event inside a sealed cavity; if c reaches the adversary neither digitally (attestation + just-in-time encrypted delivery) nor optically (would need a photodiode inside the facial interface), a blind EEG-writer's success is bounded by (1/N)^k for an N-ary decoy decision over k rounds (N = 63 Gold codes separable in 2.1 s per Thielen 2021; N = 63, k = 2 → 2.5e-4 in ~5 s). Against an adversary who sees c the bound is 1 regardless of N — state it. Proof-of-personhood (uniqueness) is out of reach; position the scheme as a session-time presence check a personhood credential could invoke (https://arxiv.org/abs/2408.07892; https://whitepaper.world.org/achieving-proof-of-human).

Threat-model table (paper-ready):

| # | Adversary | Reads c? | Writes EEG? | Naive verifier (corr > τ) | Tier 1: attested app + sealed cavity, N-ary decoys, k rounds | Tier 2: + attested sensor | Residual attacker cost |
|---|---|---|---|---|---|---|---|
| 1 | Replay of prior recording | no | yes | PASS if same code | FAIL, (1/N)^k | FAIL | — |
| 2 | Blind synthesizer | no | yes | PASS (any 15 Hz sinusoid) | FAIL, (1/N)^k | FAIL | — |
| 3 | Digital interception of c (malware on HMD/PC) | yes | yes | PASS | FAIL if Horizon OS/app attestation holds; PASS on rooted/emulated device | FAIL | break OS integrity |
| 4 | Photodiode in cavity + generic-kernel FIR (our phantom with optical input) | optically | yes | PASS | **PASS (honest residual)** | FAIL | one modified attested headset per bot (~$300 + labour); rate-limit by device ID |
| 5 | Electrode-side injection on attested sensor | yes | at electrodes | PASS | n/a | PASS unless signed impedance/PPG/1/f gates hold | physical PAI per device |
| 6 | Remote relay to distant human | yes | via human | PASS | FAIL only with enrolled latency (SD ~1.7 ms) + verifier frame clock | same | RTT vs latency window |
| 7 | Local human farm (worker wears rig) | yes | via human | PASS | PASS (bona fide) | PASS | ~wage × 3–5 s per solve; ~10× CAPTCHA-farm cost, not categorical |
| 8 | Coerced / unwitting wearer | yes | via human | PASS | PASS | PASS | consent control, not crypto |
| 9 | Enrolled-kernel theft | yes | yes | PASS | irrelevant (no secret kernel) | irrelevant | — |
| 10 | Hill-climbing on scores | yes | yes | PASS | mitigated: accept/reject only + rate limit | same | — |

Tier 1 is sound only against 1–3 and 6 (with enrollment); Tier 2 adds 4; nothing addresses 7–8. Prior art to cite as lacking exactly this analysis: Kiser et al. 2026 (above); US12423396B2 uses face/voice/emoji mimicry, not EEG. The sound-system hardware direction is an "attested EEG" front end (secure element signing hash(block)‖counter‖nonce plus signed impedance/PPG telemetry) — the analogue of C2PA camera signing (https://spec.c2pa.org/specifications/specifications/1.4/attestations/attestation.html) and the Orb's secure element.

---

## 6. Register every claim as a New Axiom-style gate

Contract copied from axuniv (`foundation_spec.py`, `refutation.dual_decide`, `gate_synthesis.MetaGate`): every claim is a card `{claim_id, statement, gate (exact computation), benchmark (frozen sessions + sha1), null, threshold, lookup_baseline (must FAIL held-out), preregistration_hash, verdict ∈ {PASS, FAIL, ABSTAIN}, refutation ∈ {TRUE, OPPOSITE_SUPPORTED, OPEN}}`. `gate = None` ⇒ ABSTAIN, never bluff; "verified" requires a populated gate_ref; thresholds are committed (git tag; SHA quoted in the paper) **before** the session that tests them; verdict precedence: any FAIL → claim FAIL; else any ABSTAIN → ABSTAIN; else PASS. d', ratio, rank never decide — they are reported only. The paper may cite only PASS sessions and must list every FAIL/ABSTAIN session (n recorded / n passed).

Implementation: copy scratchpad `claim_gates.py` → `claim_gates.py`; call `claim_gates.run()` in `xr_session.py` right after `reconstruct.run` (line ~288) and merge into xr_session.json; make `full_gate.py`/`webgate.py` assert IMAGE CLAIM == PASS on the phantom (CI catches regressions); make G1 FAIL/ABSTAIN block `start_scan` by default. Run webgate with ≥ 5+5 calibration blocks so G1 is decidable.

| Gate | Claim | Exact test | PASS | ABSTAIN | Today |
|---|---|---|---|---|---|
| G0 delivery | evidence admissible | fs within 256 ± 13 excl. gaps > 0.5 s; every ON block within f0 ± 0.3 Hz (per-block, from calib_log); slowFrames < 1 %; ≥ 95 % cells with ≥ 1 full 1-s segment; badge HARD rules (§3.1) over calib+scan; inputs frozen (sha1) | all hold | — (G0 FAIL ⇒ all others ABSTAIN) | live6: block 0 flicker + ear HF σ fail |
| G1 human_ssvep_detected | a human SSVEP exists | max-over-channels mean(on)−mean(off) of `cell_score`, exact enumeration of all C(N, n_on) relabelings | p < 0.01 | 1/C(N,n_on) ≥ 0.01 (needs ≥ 5+5) | live6 0.431 FAIL; live4 > 0.014 FAIL; live2 0.66 FAIL; phantom 0.0011 PASS. Refutation: OPPOSITE_SUPPORTED when p ≥ 0.5 and best ratio < 1.1 |
| G2 image_above_null | reconstruction beats chance | ≥ 200 circular shifts ≥ 20 s, same weights; p = (1+#≥r)/(N+1) | p < 0.01 | N < 200 | live6 p = 0.36 FAIL; phantom 0.005 PASS |
| G2b readable | image readable | G2 PASS and r ≥ `GATE_R_MIN` 0.6 | — | — | live6 FAIL; phantom 0.849 PASS |
| G3 frequency_specific | image is at f0 | control bins 11/13/17/20 Hz, ±2 Hz guard around f0, 2f0 | r(f0) − max r(fc) ≥ 0.2 and r(f0) > 0 | — | live6 −0.085 FAIL; phantom +0.73 PASS |
| G4 not_position_or_time | not drift | \|corr(grid,row)\| ≤ 0.3 and \|corr(grid,col)\| ≤ 0.3 | — | target itself \|corr\| > 0.2 | live6 +0.73 FAIL; phantom −0.05 PASS |
| G5 split_half (informational) | within-dwell reliability | first vs second half of each dwell | r ≥ 0.3 | — | live6 0.10; live2 0.39 + G3/G4 FAIL = "reliable artifact" |
| IMAGE CLAIM | G0 ∧ G1 ∧ G2 ∧ G3 ∧ G4 | | | | phantom PASS; 0 of 3 live |
| G6 decoder_improves_live_r | decoder D beats baseline | D frozen (git hash) before sealed sessions; on every sealed session: r_D − r_baseline ≥ 0.10, G2 through D p < 0.01, G3/G4 on D's output, sham (stimulus-OFF / other f0) \|r\| < null p95, phantom ≥ 0.6, shuffled-EEG ≤ 0.3 | all | G1 FAIL on that session | row_detrend FAIL (Δ −0.06); time_detrend FAIL (p 0.06) |
| G7 proof_of_presence | live visual system entrained to a post-issue nonce | nonce = K = 8 block frequencies i.i.d. from the frame-exact alphabet (Quest 120 Hz {12,15,20,24}; laptop 60 Hz {12,15,20,30}), issued after t_issue; decoder = argmax over alphabet of `cell_score` on admitted channels; ≥ 7/8 correct (binomial p = 25/65536 = 3.8e-4). Falsification arm: replay of any pre-t_issue session must FAIL; blind phantom (no calib_log access) must FAIL; informed phantom must PASS (proves the protocol, not humanness). Claim wording on PASS: presence, never identity; EEG link assumed trusted | ≥ 7/8 | no admitted channel | phantom-passable now; live pending G1 |
| G8 learned-decoder protocol | AI used without overclaiming | session-level (never segment-level) splits, subject-level for cross-subject claims; sealed test ≥ 2 live sessions recorded after model hash commit, randomized cell order, targets from an unseen family (blobs/noise, not text); inference = EEG window only; lookup/memorizer baseline must FAIL held-out; shift-null p; permitted wording "improved held-out live r from a to b (p < 0.01, n sessions)" | per G6 | — | not started |
| GA0 assr_delivery | audio path admissible | AM depth 1.0, SPL logged, phantom shows 40 Hz line un-attenuated | — | — | pending |
| GA1 assr_detected | ASSR exists | same statistic as G1 with f = 40 Hz, denominator excluding 29–31/44–46/59–61 Hz; report time-to-p < 0.01 | p < 0.01 | < 5+5 blocks | pending |
| GA2 level_curve | ear as level meter | ASSR amplitude vs SPL monotone (Spearman ρ > 0, p < 0.01 across ≥ 4 levels × ≥ 5 repeats) | — | — | pending |
| GA3 no_crossmodal_cost | 15 Hz SSVEP unchanged by tone | G1 statistic in "both" ≥ G1 statistic in "visual only" − null p95 | — | — | pending |
| GA4 attention_index | ASSR_attendA − ASSR_attendV > 0 | permutation over block labels | p < 0.05 | — | pending |
| GA5 IM_null (pre-registered) | no 25/55 Hz IM | power at 25/55 Hz within neighbour null | — | — | pending |
| GS1..GS3 security claims | any sentence in §5 that asserts a measured property | must point to G7 ledger rows (E2 blind phantom FAIL, E3 informed phantom PASS, E4 photodiode PASS, E5 relay ROC, E6 physiology-gate pass counts) | — | no experiment → sentence is design/future work only | E3 done (r 0.787–0.849); E2/E4/E5/E6 pending |

---

## 7. Execution order for the next sessions

**S0 — Hygiene (no human, ~half day).** (a) Freeze `runs/xr_live6`, `xr_live2`, `xr_live4`, `gate_full`, `webgate` with sha1 manifests; move the 14.40 Hz variant files out of `xr_live6`. (b) Implement Rank A items 1–8 (§2) + badge thresholds (§3.1) + `claim_gates.py` wiring (§6). (c) Recorder stops on `scan_done`; `--require-pass` semantics become default. (d) Git-init the repo if still none (report: "the repo currently has no remote"), tag `prereg-v1` with the gate thresholds, quote the SHA in the paper.

**S1 — Phantom regression (no human).** `full_gate.py`, `webgate.py --calib-blocks 6`, Unity PlayMode phantom: IMAGE CLAIM must PASS on all (expect r ≈ 0.78–0.85, G1 p = 0.0011, G3 margin ≈ +0.7). Add: 40 Hz phantom (GA0), 15 + 40 Hz summed phantom at ~1 µV / ~50–300 nV, blind phantom for G7 (must FAIL), informed phantom for G7 (must PASS). Go/no-go: any regression → fix before S2.

**S2 — G1-only live session (first human, ~15 min).** Aux Oz electrode + ear prep + HSI = 1 gate + chromatic stimulus + 8–10 blocks × 10 s ON/OFF, darkened room, chin rest; no scan. Gates: G0, G1; also print the lock-in sensitivity bound (µV) and the per-block Hz. Go/no-go: **G1 p < 0.01 on ≥ 1 channel** → S3; else iterate contact/electrode (Oz vs TP9/TP10 vs AF7/AF8 ranked by p) — do not scan, do not touch decoders. Repeat S2 up to 3 times before reconsidering hardware (OpenBCI/wet electrodes).

**S3 — Image session (only after S2 passes).** Quick preset with randomized cell order; G2–G5; report r with shift-null p, control-bin margin, r_row. Go/no-go: IMAGE CLAIM PASS on one subject → the paper's first honest live figure. Predicted requirement (quality_stats two-class model, validated on webgate 0.757 pred vs 0.782 obs): cell d' ≈ 2.52 (block d' ≈ 3.1) for r ≥ 0.6, i.e. ≥ 2.5–3× today's best.

**S4 — ASSR-only session (`--mode assr`, AM depth 1.0, 500 Hz / 40 Hz, ~60 dB SPL, 8–10 blocks).** GA0–GA1 + time-to-p per channel (benchmark 22 s mean, 2–92 s range with a clinical montage). Go/no-go: GA1 PASS on TP9 or TP10 → S5; else the ear-electrode problem is confirmed for both modalities and the "ear" paper line is a phantom-validated protocol with a reported human null and sensitivity bound.

**S5 — Combined session (`--mode both`, §4.2 blocks 0–5).** GA2–GA5 and G1 under tone. Deliverable: two integration-time constants and one level-calibration curve per subject.

**S6 — G7 live.** Nonce-driven calibration blocks (per-block `freq` list in `start_calib`, xr_session.py line 250; `flickerOn(t, m.freq)` already accepts it); log t_issue; run replay and blind-phantom falsification arms; then E4 (photodiode inside the facial interface driving a 15-tap FIR → OSC 5005) and E5 (20/50/100/200 ms relay delays, ROC with/without enrolled latency) and E6 (physiology-gate pass counts). All six results go in the paper, including the ones that pass when they should not.

**S7 — Decoders (only after ≥ 1 IMAGE CLAIM PASS).** Commit model hash, declare sealed sessions, record ≥ 2 new sessions with randomized order and unseen target family, evaluate G6/G8 with the Welch baseline as control arm. Existing xr_live runs are train/exploratory only.

**S8 — Scale.** n ≥ 10 subjects, per-subject reporting, counterbalanced §4.2 protocol, REB consent covering neural + biometric secondary use, on-device verification preferred so raw EEG never leaves the HMD. The current n = 1 (r = 0.19) is a pilot, not a result, and should be described as such in every venue.

Key file paths: `reconstruct.py` (ssvep_score L80, cell_score L123, reconstruct L145, run L226), `run_session.py` (score_calibration L279–340), `xr_session.py` (start_calib L250, reconstruct.run L288, run_assr L319–348, `--mode assr` L422), `xr_stimulus.html` (runAssr L228, modDepth L236), `config.py` (CALIB_* L39–46, GATE_R_MIN L47, NOISE_BAND L21), analysis scripts `analysis\{baseline_preproc,cca,act_chirplet,lock_in,clock_check,quality_stats,audit_baseline_preproc,audit_null_sweep}.py`, gate prototype `claim_gates.py`.

---
## Status update 2026-09-14 (after the first full VR run)

**What is now established (n=1, Alexander, Oz aux cup + Muse 2, Mind Monitor OSC):**
- Oz aux electrode validated: eyes-closed alpha peak 10.5 Hz, x8.4 power on AUX (`--mode alpha`, runs/alpha1).
- SSVEP exists but only at 7.5-12 Hz for this subject: laptop full-field sweep (runs/sweep2) AUX line SNR 9.0 @10.4 Hz (p=0.017), 5.9 @11.4 Hz (p=0.004); 15 Hz weak, 20 Hz absent. Two earlier valid nulls at 15/20 Hz are explained by this response curve (sensitivity floor ~1-2 uV).
- In the Quest browser (72 fps, 12 Hz = 3 frames): full-panel 12 Hz calibration p=0.014 with the last 4 ON blocks at 4-10x line SNR on AUX (runs/vr_full1); full-field blue @12 Hz p=0.005 and green @9 Hz p=0.009 (colour calibration), red @7.2 Hz nothing.
- NOT established: the per-cell scan. Grey "NO" 12x8 @3.5 s/cell is at chance for Welch, line and CCA scoring (r 0.17-0.26 vs null max 0.18-0.40). The colour flag's blue plane r=0.72 exceeds 6/6 shifted nulls (max 0.52) but R/G planes are chance: suggestive only.
- Audio: 40 Hz-tagged music via Quest speakers, p=0.69 (no ASSR). Needs louder/closer sound (earbuds) and longer blocks.

**Why the scan fails while calibration passes:** the calibration flicker is the full panel; a scan cell is ~1/100 of it and gets 3.5 s. The SSVEP scales with stimulated area; the line detector needs ~2 uV in the cell window. Options, in order of expected payoff:
1. Bigger cells, longer dwell: 6x4 grid at 8 s (3.2 min) at 12 Hz — the colour run's 6x4 @5 s already showed a plane above null.
2. Keep the flicker LARGE and move the *image* instead: flicker the whole panel but mask it by the target (Mann's "pixel-by-pixel" the other way round: bright cells flicker in place, all at once, while the subject fixates each cell in turn) — the peripheral flicker drives the SSVEP, foveation modulates it. Must be tested against the phantom and the shifted null (it changes the physics: expect a dominant common-mode response).
3. Frequency-multiplex several cells per dwell (2-3 tags in the 8-12 Hz band) to cut scan time.
4. Per-cell adaptive dwell: keep flickering until the line SNR crosses a threshold or a time cap.

**Tooling that now exists:** `--mode full|calib|sweep|alpha|assr`, line-detector gate + weights, `--recon-method line`, colour scan (`reconstruct.reconstruct_color`, `targets.color_target`), music tag, phantom with colour+audio drives, `analysis/decoders_compare.py`, `analysis/g1_{diag,fine,line}.py`, `--page-token`, gamepad/keyboard arm gate, timestamp-locked flicker, median-vsync refresh measurement.

