"""
Central MediaPipe landmark index definitions and skeleton connections.

This is the SINGLE place that knows about model landmark numbering.
If you switch models or want to tweak which points a feature uses,
edit ONLY this file — every vision module imports its indices from here.

Indices follow the MediaPipe Tasks models:
  - Face Landmarker : 468 base landmarks (478 with refined irises)
  - Hand Landmarker : 21 landmarks per hand
Reference: https://ai.google.dev/edge/mediapipe/solutions/guide
"""

# ---------------------------------------------------------------------------
# Face Landmarker — mouth / lips
# ---------------------------------------------------------------------------
FACE_LIP_TOP          = 13    # top-lip centre
FACE_LIP_BOTTOM       = 14    # bottom-lip centre
FACE_LIP_LEFT_CORNER  = 61
FACE_LIP_RIGHT_CORNER = 291

# ---------------------------------------------------------------------------
# Face Landmarker — eyes (6 points each, ordered for the EAR formula)
#   p1=outer, p2=top-1, p3=top-2, p4=inner, p5=bottom-2, p6=bottom-1
# ---------------------------------------------------------------------------
FACE_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
FACE_RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ---------------------------------------------------------------------------
# Face Landmarker — head pose reference points
# ---------------------------------------------------------------------------
FACE_NOSE_TIP         = 1
FACE_LEFT_EYE_CORNER  = 33
FACE_RIGHT_EYE_CORNER = 263

# ---------------------------------------------------------------------------
# Hand Landmarker — finger landmark ids
#   order: thumb, index, middle, ring, pinky
# ---------------------------------------------------------------------------
HAND_WRIST   = 0
HAND_TIP_IDS = [4, 8, 12, 16, 20]
HAND_PIP_IDS = [3, 6, 10, 14, 18]

# Note: the hand skeleton is drawn using the official Tasks API connection set
# (mediapipe.tasks.python.vision.HandLandmarksConnections.HAND_CONNECTIONS),
# so no manual connection list is needed here.
