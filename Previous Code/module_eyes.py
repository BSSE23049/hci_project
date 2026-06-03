"""
module_eyes.py
==============
Module 2 -- Eyes Detection

Tasks implemented
-----------------
T1 : Blink counter    -- counts blink events; a blink = EAR drops below
                          EAR_THRESH then rises back above it.
T2 : EAR display      -- computes and shows left and right Eye Aspect Ratios
                          to three decimal places each frame.
T3 : Drowsiness alert -- triggers a red banner after DROWSY_FRAMES_ENTER
                          consecutive low-EAR frames; clears after
                          DROWSY_FRAMES_CLEAR consecutive normal-EAR frames.

MediaPipe component: FaceLandmarker (Tasks API 0.10.x+)

Reference
---------
Soukupova T. & Cech J. (2016). Real-time Eye Blink Detection using Facial
Landmarks.  CVWW 2016.

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
    lm_to_px, euclidean,
    draw_text, draw_red_banner, detection_loop, _esc_hint,
)


# ==============================================================================
#  LANDMARK INDICES
# ==============================================================================

# Six-point EAR geometry per eye (Soukupova & Cech 2016)
# p1 = outer corner, p2 = upper outer, p3 = upper inner,
# p4 = inner corner, p5 = lower inner, p6 = lower outer
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]


# ==============================================================================
#  THRESHOLDS
# ==============================================================================

#: EAR below this value indicates the eye is closed (or nearly so).
EAR_THRESH = 0.25

#: Number of consecutive closed-eye frames before drowsiness is flagged.
DROWSY_FRAMES_ENTER = 20

#: Number of consecutive open-eye frames required to clear the drowsy alert.
#: Using a smaller clear count than the enter count means the alert resets
#: quickly once the person opens their eyes, without flickering.
DROWSY_FRAMES_CLEAR = 5


# ==============================================================================
#  FEATURE FUNCTION
# ==============================================================================

def compute_ear(landmarks, eye_indices, w, h):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye using 6 landmarks.

    Formula (Soukupova & Cech 2016)
    --------------------------------
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

    where:
        p1, p4  -- horizontal end-points (outer/inner eye corners)
        p2, p6  -- first vertical pair  (upper outer / lower outer)
        p3, p5  -- second vertical pair (upper inner / lower inner)

    Two vertical distances are averaged to account for the fact that the
    eye lid does not open and close symmetrically at a single point.
    The denominator is normalised by twice the horizontal span so EAR is
    scale-invariant (same threshold works for near and far faces).

    A fully open eye typically has EAR ≈ 0.28 – 0.35.
    A closed eye has EAR ≈ 0.10 – 0.20.

    Parameters
    ----------
    landmarks   : list[NormalizedLandmark]
        Full 468-point face landmark list for one detected face.
    eye_indices : list[int]
        Six FaceMesh landmark indices in the order [p1, p2, p3, p4, p5, p6].
    w : int  -- frame width in pixels.
    h : int  -- frame height in pixels.

    Returns
    -------
    float
        EAR value >= 0.  Lower means more closed.
    """
    pts    = [lm_to_px(landmarks[i], w, h) for i in eye_indices]
    v1     = euclidean(pts[1], pts[5])   # p2 – p6
    v2     = euclidean(pts[2], pts[4])   # p3 – p5
    h_dist = euclidean(pts[0], pts[3])   # p1 – p4
    return (v1 + v2) / (2.0 * h_dist + 1e-6)


# ==============================================================================
#  MODULE ENTRY POINT
# ==============================================================================

def run_eyes_detection(source):
    """
    Launch Module 2 — Eyes Detection and run the live annotation loop.

    Setup
    -----
    Creates a FaceLandmarker in IMAGE or VIDEO RunningMode depending on the
    source type.  State variables for blink counting, drowsiness counters, and
    the synthetic video timestamp are maintained in the enclosing scope and
    accessed via nonlocal inside the process() closure.

    Per-frame processing
    --------------------
    1.  Convert BGR → mediapipe.Image and run the landmarker.
    2.  Draw left and right eye connection sets.
    3.  T2: Compute EAR for both eyes; display individually.
    4.  T1: If average EAR < EAR_THRESH, mark eye as closed.
            When EAR rises back above threshold, increment blink counter.
    5.  T3: Maintain drowsy_frames and normal_frames counters.
            Set/clear drowsy_alert based on DROWSY_FRAMES_ENTER / _CLEAR.
            If alert is active, draw the red banner.

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

    blink_count   = 0
    eye_closed    = False
    drowsy_frames = 0
    normal_frames = 0
    drowsy_alert  = False
    frame_ts      = 0

    with FaceLandmarker.create_from_options(options) as detector:

        def process(frame):
            nonlocal blink_count, eye_closed
            nonlocal drowsy_frames, normal_frames, drowsy_alert, frame_ts

            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += 33
                result = detector.detect_for_video(mp_img, frame_ts)

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                # Draw both eye outlines
                for conn in [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                             FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE]:
                    mp_drawing.draw_landmarks(
                        frame, lms, conn,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style(),
                    )

                # T2: Per-eye EAR
                ear_l = compute_ear(lms, _LEFT_EYE_IDX,  w, h)
                ear_r = compute_ear(lms, _RIGHT_EYE_IDX, w, h)
                avg   = (ear_l + ear_r) / 2.0

                draw_text(frame, f"EAR L: {ear_l:.3f}", (20, 40))
                draw_text(frame, f"EAR R: {ear_r:.3f}", (20, 75))

                # T1: Blink detection
                if avg < EAR_THRESH:
                    eye_closed = True
                elif eye_closed:
                    blink_count += 1
                    eye_closed   = False

                draw_text(frame, f"Blinks: {blink_count}", (20, 110),
                          color=(0, 220, 255))

                # T3: Drowsiness counters
                if avg < EAR_THRESH:
                    drowsy_frames += 1
                    normal_frames  = 0
                else:
                    normal_frames += 1
                    drowsy_frames  = 0

                if drowsy_frames >= DROWSY_FRAMES_ENTER:
                    drowsy_alert = True
                if normal_frames >= DROWSY_FRAMES_CLEAR:
                    drowsy_alert = False

                if drowsy_alert:
                    draw_red_banner(frame, "DROWSY! Wake Up!")
            else:
                draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255))

            _esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 2 -- Eyes Detection")
