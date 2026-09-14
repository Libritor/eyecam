"""Shared parameters for the eye-as-a-camera pipeline.

Values follow Mann et al., "The Human Eye as a Camera," IEEE HealthCom 2019
(wearcam.org/eyecam.pdf) unless noted.
"""

# --- Stimulus ---
STIM_FREQ_HZ = 15.0        # paper moved from 12 Hz to 15 Hz (above most alpha)
HARMONIC_HZ = 30.0         # first harmonic, included in the SSVEP power estimate
DISPLAY_FPS = 60           # flicker = 2 frames on / 2 frames off at 60 Hz -> 15 Hz
CURSOR_PX = 100            # flashing square size on screen (paper: 100x100 px)

# --- Scan geometry ---
GRID_W = 32                # horizontal scan positions per line
GRID_H = 24                # number of scan lines (paper used 48 overlapping lines)
SECONDS_PER_CELL = 1.0     # dwell per grid cell (paper: ~6.67 s per point; raise
                           # this for real sessions, 1.0 s is for fast simulation)

# --- EEG ---
FS = 256                   # Muse sampling rate (Hz)
NOISE_BAND = (14.0, 50.0)  # denominator band for relative SSVEP power (paper)
PSD_NPERSEG = 256          # Welch segment: 1 s -> 1 Hz bins, 15 Hz on-bin

# --- Reconstruction ---
VERTICAL_KERNEL = [0.5, 0.5, 1.0, 0.5, 0.5]  # Eq. 1 of the paper: overlapping
                                             # scan lines blended as
                                             # f(x) = (2x + x+-1 + x+-2)/2
UPSCALE = 12               # bicubic upscale factor for the output image
ARTIFACT_Z = 4.0           # clip samples beyond this many robust sigmas;
ARTIFACT_DROP_FRAC = 0.2   # drop a cell-channel if > this fraction clipped

# --- Session runner ---
# preset name -> (grid_w, grid_h, seconds_per_cell)
PRESETS = {
    "quick": (12, 8, 4.0),      # ~6.4 min scan
    "standard": (16, 12, 5.0),  # ~16 min
    "fine": (24, 18, 6.0),      # ~43 min
}
CALIB_BLOCKS = 6           # ON/OFF block pairs in the calibration stage
CALIB_ON_S = 7.0
CALIB_OFF_S = 7.0
CALIB_DISCARD_S = 1.0      # drop from each block start (entrainment transient)
CALIB_DPRIME_MIN = 1.0     # best channel must reach this to proceed
CALIB_RATIO_MIN = 1.3      # ... and this on/off score ratio
CALIB_RANK_MIN = 5         # ... and >= this many ON blocks above max OFF block
CALIB_WEIGHT_DPRIME_MIN = 0.5  # channels below this get zero weight
CALIB_P_MAX = 0.01         # exact family-wise permutation p for 'passed'
GATE_R_MIN = 0.6           # full-stack phantom gate threshold
