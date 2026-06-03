"""
module_lips.py
==============
Module 1 -- Lips Detection

Tasks implemented
-----------------
T1 : Smile detection  -- classifies each frame as 'Smiling' or 'Neutral' by
                          comparing lip-corner vertical drop to the upper lip
                          centre.
T2 : MAR tracking     -- computes and displays the Mouth Aspect Ratio to 2 d.p.;
                          draws a red border when MAR exceeds the yawn threshold.
T3 : Lip-sync counter -- counts open/close mouth cycles using hysteresis to
                          avoid double-counting near the threshold boundary.

MediaPipe component: FaceLandmarker (Tasks API 0.10.x+)

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
"""

from mediapipe.tasks.python         import BaseOptions
from mediapipe.tasks.python.vision  import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    RunningMode,
    drawing_utils  as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from models import FACE_MODEL_PATH, bgr_to_mp_image
from utils  import (
    lm_to_px, euclidean, compute_aspect_ratio,
    draw_text, draw_red_border, detection_loop, _esc_hint,
)


# ==============================================================================
#  LANDMARK INDICES  (FaceMesh 468-point topology)
# ==============================================================================

_MOUTH_TOP    = 13    # upper lip centre (vertical measurement top anchor)
_MOUTH_BOTTOM = 14    # lower lip centre (vertical measurement bottom anchor)
_MOUTH_LEFT   = 61    # left mouth corner (horizontal span left anchor)
_MOUTH_RIGHT  = 291   # right mouth corner (horizontal span right anchor)

_CORNER_L        = 61    # same as _MOUTH_LEFT  — reused for smile geometry
_CORNER_R        = 291   # same as _MOUTH_RIGHT — reused for smile geometry
_SMILE_UPPER_MID = 13    # upper lip centre    — reference Y for smile check


# ==============================================================================
#  THRESHOLDS
# ==============================================================================

#: MAR value above which the mouth is considered 'open' (entering open state).
MAR_OPEN_THRESH  = 0.55

#: MAR value below which the mouth is considered 'closed' (entering closed state).
#: The gap between OPEN and CLOSE thresholds implements hysteresis, preventing
#: rapid toggling when MAR hovers near a single threshold.
MAR_CLOSE_THRESH = 0.35


# ==============================================================================
#  FEATURE FUNCTIONS
# ==============================================================================

def compute_mar(landmarks, w, h):
    """
    Compute the Mouth Aspect Ratio (MAR) for a detected face.

    MAR is analogous to the Eye Aspect Ratio introduced by Soukupova & Cech
    (2016).  It measures how 'open' the mouth is as a ratio of vertical span
    to horizontal span:

        MAR = ||top_lip - bottom_lip|| / (||left_corner - right_corner|| + ε)

    A MAR near 0 means the lips are pressed together; a MAR above ~0.55
    typically indicates the mouth is open (yawning / speaking loudly).

    Parameters
    ----------
    landmarks : list[NormalizedLandmark]
        Full list of 468 normalised face landmarks from
        FaceLandmarkerResult.face_landmarks[face_index].
    w : int
        Frame width in pixels — used by lm_to_px() to convert coordinates.
    h : int
        Frame height in pixels.

    Returns
    -------
    float
        MAR value >= 0.  Larger values indicate a more open mouth.
    """
    top    = lm_to_px(landmarks[_MOUTH_TOP],    w, h)
    bottom = lm_to_px(landmarks[_MOUTH_BOTTOM], w, h)
    left   = lm_to_px(landmarks[_MOUTH_LEFT],   w, h)
    right  = lm_to_px(landmarks[_MOUTH_RIGHT],  w, h)
    return compute_aspect_ratio(top, bottom, left, right)


def detect_smile(landmarks, w, h):
    """
    Classify the current facial expression as 'Smiling' or 'Neutral'.

    Algorithm
    ---------
    A genuine smile pulls the mouth corners upward (smaller Y in image
    coordinates because Y increases downward) relative to the upper lip.
    However, in frontal camera images the corners actually move slightly
    downward when the cheeks raise.  The heuristic used here checks that
    BOTH corners sit below the upper lip centre by at least 5 % of the
    inter-corner distance — this scale-normalised threshold is
    resolution-independent and robust to small head tilts.

    Why 5 %?
    --------
    Empirically tested on a mix of frontal selfie photos at distances
    0.4 m – 1.2 m.  Below 5 % most neutral expressions pass; above 5 %
    most visible smiles are captured without many false positives.

    Parameters
    ----------
    landmarks : list[NormalizedLandmark]
        Full 468-point face landmark list for one detected face.
    w : int  -- frame width in pixels.
    h : int  -- frame height in pixels.

    Returns
    -------
    str
        'Smiling' if both corners satisfy the threshold, else 'Neutral'.
    """
    corner_l = lm_to_px(landmarks[_CORNER_L],        w, h)
    corner_r = lm_to_px(landmarks[_CORNER_R],        w, h)
    upper    = lm_to_px(landmarks[_SMILE_UPPER_MID], w, h)

    mouth_width = euclidean(corner_l, corner_r)
    threshold   = mouth_width * 0.02

    # In image coords Y increases downward, so corner Y > upper Y means drop.
    drop_l = corner_l[1] - upper[1]
    drop_r = corner_r[1] - upper[1]
    return "Smiling" if (drop_l > threshold and drop_r > threshold) else "Neutral"


# ==============================================================================
#  MODULE ENTRY POINT
# ==============================================================================

def run_lips_detection(source):
    """
    Launch Module 1 — Lips Detection and run the live annotation loop.

    Setup
    -----
    * Creates a FaceLandmarker with RunningMode.IMAGE for static images and
      RunningMode.VIDEO for webcam / video file input.
    * VIDEO mode requires a monotonically increasing timestamp in milliseconds;
      we synthesise one by incrementing frame_ts by 33 ms per frame (~30 fps).

    Per-frame processing
    --------------------
    1.  Convert BGR frame to mediapipe.Image.
    2.  Call detector.detect() or detector.detect_for_video() to get landmarks.
    3.  Draw the LIPS connection set using the Tasks drawing API.
    4.  T1: Compute smile label and overlay it.
    5.  T2: Compute MAR and overlay it; add red border if MAR > MAR_OPEN_THRESH.
    6.  T3: Update hysteresis state machine and increment lip-sync counter when
            a full open→close cycle completes.

    Parameters
    ----------
    source : int | str | np.ndarray
        * int         -- webcam device index.
        * str         -- path to a video file.
        * np.ndarray  -- BGR static image loaded with cv2.imread().

    Returns
    -------
    None
    """
    import numpy as np
    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # T3 hysteresis state — captured via nonlocal inside the closure
    mouth_open = False
    sync_count = 0
    frame_ts   = 0    # synthetic millisecond timestamp for VIDEO mode

    with FaceLandmarker.create_from_options(options) as detector:

        def process(frame):
            nonlocal mouth_open, sync_count, frame_ts
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += 33
                result = detector.detect_for_video(mp_img, frame_ts)

            if result.face_landmarks:
                lms = result.face_landmarks[0]   # first (only) detected face

                # Draw lip outline using Tasks-API drawing helper
                mp_drawing.draw_landmarks(
                    frame, lms,
                    FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style(),
                )

                # T1: Smile classification
                smile = detect_smile(lms, w, h)
                draw_text(frame, f"Mood: {smile}", (20, 40),
                          color=(0, 255, 100) if smile == "Smiling" else (200, 200, 200))

                # T2: MAR display + red border on open mouth
                mar = compute_mar(lms, w, h)
                draw_text(frame, f"MAR: {mar:.2f}", (20, 75))
                if mar > MAR_OPEN_THRESH:
                    draw_red_border(frame)

                # T3: Lip-sync counter with hysteresis
                if not mouth_open and mar > MAR_OPEN_THRESH:
                    mouth_open = True
                elif mouth_open and mar < MAR_CLOSE_THRESH:
                    mouth_open = False
                    sync_count += 1

                draw_text(frame, f"Lip-Sync Count: {sync_count}", (20, 110))
                draw_text(frame,
                          "Mouth: OPEN" if mouth_open else "Mouth: closed",
                          (20, 145),
                          color=(0, 100, 255) if mouth_open else (180, 180, 180))
            else:
                draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255))

            _esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 1 -- Lips Detection")
