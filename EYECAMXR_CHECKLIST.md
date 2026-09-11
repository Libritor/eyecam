# EyeCamXR — on-device bring-up checklist (the parts that need hands)

Unity project: `MuseLogMR`
APK output: `EyeCamXR.apk` in the project root (package `com.mannlab.eyecamxr`).

## 0. Install

```powershell
adb install -r MuseLogMR\EyeCamXR.apk
adb shell monkey -p com.mannlab.eyecamxr 1        # launch
adb logcat -s Unity | Select-String "\[EyeCam\]"  # watch
adb shell ip route                                 # Quest's IP (for phantom/phone)
```

Session artifacts land in
`/sdcard/Android/data/com.mannlab.eyecamxr/files/EyeCamXR/session_*/`
(`adb pull` that folder: eeg.csv, cursor_log.csv, calib_log.csv,
calibration.json, session.json, recon.png, target.csv).

## 1. Timing gate (no headband needed)

1. Boot the app; logcat must show `[EyeCam] refresh requested=120 actual=120`
   (90 is acceptable — 15 Hz stays frame-exact; note it in session.json).
2. Run a **phantom session against the real headset**:
   - PC: `python spectator_bridge.py --session runs/quest_phantom`
   - PC: `python phantom_subject.py --session runs/quest_phantom --host <QUEST_IP> --port 5005`
   - Headset: start the session (source OSC). The bridge mirrors the Quest's
     cursor stream into the phantom's input CSVs, closing the loop.
3. Pull session.json and check: `measuredFlickerHz` within **0.1 Hz** of 15.0,
   `slowFrames/totalFrames < 1%`, `r >= 0.6`, `calibPassed: true`.

## 2. Real Muse session

- **Native BLE (all-in-one)**: pair nothing — the app scans and connects to
  the first Muse it finds (Muse 2 / Muse S / Athena; model logged with the
  chosen preset). With an aux electrode plugged in, set the aux toggle
  (PlayerPrefs `eyecam_aux_electrode=1`) → PRESET_20 (5 ch) on classic
  Muses / PRESET_1022 (8 ch) on Athena; the true channel count is probed at
  runtime and recorded.
- **Phone path**: MuseLog app → OSC Streaming → IP shown on the headset's
  signal-check screen, port 5005, SnowballArcade format, full-rate ON.
- Protocol: seated, headset stable, room lighting irrelevant (black-void VR).
  Calibration (~90 s at 6 blocks) tells you in advance whether the montage
  carries SSVEP; the confirm screen refuses weak sessions honestly.

## 3. Spectators ("shared XR")

- PC: `python spectator_bridge.py` → open `http://<PC_IP>:8080` on any
  browser/projector; a second headset opens the same URL in Quest Browser.
- Live: stage banner, EEG rate, calibration verdict, cursor mirror with
  flicker state, and the reconstruction building up row by row.

## 4. Aux ("5th electrode") hardware options

- Interaxon auxiliary cup electrode: micro-USB (Muse 2016/2/S) or USB-C
  (Athena / Muse 2 2024). Aux input: 1.45 mV pp, AC-coupled (pins 3/4).
- Or MannLab's 3D-printed Oz holder from the HealthCom 2019 paper.
- Place at Oz (inion); the calibration d′ table quantifies the gain over
  TP9/TP10 per subject — that comparison is itself unpublished data.
