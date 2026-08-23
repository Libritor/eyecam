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
