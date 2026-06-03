"""
Vision app configuration — ALL flags and constants live here.
To change any behaviour: edit this file only. No other vision file should be modified.
"""

# ---------------------------------------------------------------------------
# Feature enable/disable flags
# ---------------------------------------------------------------------------
ENABLE_LIPS_DETECTION  = True
ENABLE_EYES_DETECTION  = True
ENABLE_FACE_DETECTION  = True
ENABLE_HAND_DETECTION  = True

# ---------------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------------
DEFAULT_INPUT_SOURCE = "webcam"   # "webcam" | "video" | "image"
WEBCAM_INDEX         = 0
VIDEO_FPS            = 30          # synthetic timestamp rate for VIDEO running mode

# ---------------------------------------------------------------------------
# MediaPipe landmarker confidences
# ---------------------------------------------------------------------------
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE  = 0.5
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE  = 0.5

# ---------------------------------------------------------------------------
# Lips constants
# ---------------------------------------------------------------------------
MAR_OPEN_THRESHOLD   = 0.55    # MAR above this → mouth open / yawning
MAR_CLOSE_THRESHOLD  = 0.35    # hysteresis: below this → mouth closed
SMILE_CORNER_THRESHOLD = 0.02  # fraction of mouth width; corners below upper lip = smile

# ---------------------------------------------------------------------------
# Eyes constants
# ---------------------------------------------------------------------------
# Calibration — the module samples the user's own open-eye EAR for this many
# frames at startup, then derives a personal threshold.  Works for small eyes,
# large eyes, and glasses wearers without any manual tuning.
CALIBRATION_FRAMES     = 60     # frames to sample (≈ 2 s at 30 fps); keep eyes open

# Personal thresholds are derived as a fraction of the calibrated baseline:
#   closed_threshold = baseline × EAR_CLOSED_RATIO
#   A typical open-eye EAR is 0.28–0.38; 0.70 of that gives ~0.20–0.27.
#   Reduce this ratio (e.g. 0.65) if blinks are still missed; raise it
#   (e.g. 0.75) if normal blinking triggers false drowsiness.
EAR_CLOSED_RATIO       = 0.70

# Hard-floor: even after calibration the threshold never goes below this.
# Prevents a miscalibration (eyes partly closed during warmup) from setting
# a threshold so low that nothing ever triggers.
EAR_FLOOR              = 0.15

# Hard-ceiling: prevents an unreasonably high threshold from triggering on
# every partial blink (e.g. if calibration captured a wide-eyed expression).
EAR_CEILING            = 0.28

DROWSY_FRAMES_TRIGGER  = 20    # consecutive closed frames before drowsiness alert fires
DROWSY_CLEAR_FRAMES    = 5     # consecutive open frames to clear drowsiness alert

# Legacy fallback used only when calibration cannot collect enough valid frames
# (e.g. no face detected during warmup).  Kept here so it's easy to adjust.
EAR_CLOSED_THRESHOLD   = 0.25

# ---------------------------------------------------------------------------
# Face constants
# ---------------------------------------------------------------------------
EMOTION_INFERENCE_EVERY  = 5   # run DeepFace every N frames (1 = every frame)
EMOTION_HISTORY_SECONDS  = 5   # rolling window length (seconds) for recent-mood

# ---------------------------------------------------------------------------
# Hand gesture map
# Key   : (non_thumb_finger_count: int, thumb_extended: bool)
# Value : (label: str, tag: str)   — `tag` is a short ASCII label drawn on the
#          OpenCV frame (Hershey fonts cannot render emojis, so keep tags ASCII).
#
# TO CHANGE A GESTURE: edit only this dict. Nothing else changes.
# ---------------------------------------------------------------------------
GESTURE_MAP = {
    (0, False): ("Fist",      "OK"),
    (1, False): ("One",       "1"),
    (2, False): ("Peace",     "V"),
    (3, False): ("Three",     "3"),
    (4, False): ("Four",      "4"),
    (5, False): ("Open Hand", "HI"),
    (1, True):  ("Thumbs Up", "+1"),
    (5, True):  ("High Five", "5"),
}

# ---------------------------------------------------------------------------
# Letter / hand-sign detection map
# Key   : letter string
# Value : (non_thumb_count, thumb_extended) — same key format as GESTURE_MAP
#
# TO ADD LETTERS: add entries here only. Set ENABLE_LETTER_DETECTION = True.
# ---------------------------------------------------------------------------
ENABLE_LETTER_DETECTION = True

LETTER_GESTURE_MAP = {
    "A": (0, True),
    "B": (4, False),
    "I": (1, False),
    "L": (1, True),
    "U": (2, False),
    "W": (3, False),
    "Y": (5, True),
}

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
ENABLE_GESTURE_GAME = True   # offer the gesture game option in hand mode
GAME_HOLD_SECONDS   = 1.0    # seconds the correct gesture must be held to score
MAX_HANDS           = 2      # maximum simultaneous hands to track