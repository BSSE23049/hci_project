"""
lips_module.py
==============
Module 1 — Lips Detection (MediaPipe Face Landmarker, Tasks API).

Features
--------
T1 Smile detection   : 'Smiling' vs 'Neutral' from lip-corner drop.
T2 MAR tracking      : Mouth Aspect Ratio; red border when mouth is open/yawning.
T3 Lip-sync counter  : counts open→close cycles using hysteresis.

Landmark indices come from vision/landmarks.py.
Thresholds come from vision/vision_config.py.
"""

import numpy as np

from mediapipe.tasks.python.vision import (
    RunningMode,
    FaceLandmarksConnections,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from vision.vision_config import (
    MAR_OPEN_THRESHOLD,
    MAR_CLOSE_THRESHOLD,
    SMILE_CORNER_THRESHOLD,
    VIDEO_FPS,
)
from vision.landmarks import (
    FACE_LIP_TOP, FACE_LIP_BOTTOM, FACE_LIP_LEFT_CORNER, FACE_LIP_RIGHT_CORNER,
)
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.vision_utils import (
    lm_to_px, euclidean, compute_aspect_ratio,
    draw_text, draw_red_border, draw_esc_hint, detection_loop,
)


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def compute_mar(landmarks, w: int, h: int) -> float:
    """
    Compute the Mouth Aspect Ratio (MAR) for a detected face.

        MAR = ||top_lip - bottom_lip|| / (||left_corner - right_corner|| + eps)

    Parameters
    ----------
    landmarks : list
        Face landmark list (each item exposes .x/.y in [0,1]).
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    float
        MAR value >= 0. Larger means a more open mouth.
    """
    top    = lm_to_px(landmarks[FACE_LIP_TOP],          w, h)
    bottom = lm_to_px(landmarks[FACE_LIP_BOTTOM],       w, h)
    left   = lm_to_px(landmarks[FACE_LIP_LEFT_CORNER],  w, h)
    right  = lm_to_px(landmarks[FACE_LIP_RIGHT_CORNER], w, h)
    return compute_aspect_ratio(top, bottom, left, right)


def detect_smile(landmarks, w: int, h: int) -> bool:
    """
    Classify the expression as smiling or neutral.

    Both mouth corners must sit below the upper-lip centre by at least
    SMILE_CORNER_THRESHOLD * mouth_width (scale-invariant).

    Parameters
    ----------
    landmarks : list
        Face landmark list.
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    bool
        True if smiling, else False.
    """
    corner_l = lm_to_px(landmarks[FACE_LIP_LEFT_CORNER],  w, h)
    corner_r = lm_to_px(landmarks[FACE_LIP_RIGHT_CORNER], w, h)
    upper    = lm_to_px(landmarks[FACE_LIP_TOP],          w, h)

    mouth_width = euclidean(corner_l, corner_r)
    threshold   = mouth_width * SMILE_CORNER_THRESHOLD

    drop_l = corner_l[1] - upper[1]   # Y grows downward, so positive = drop
    drop_r = corner_r[1] - upper[1]
    return drop_l > threshold and drop_r > threshold


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_lips(source) -> None:
    """
    Launch Module 1 — Lips Detection and run the live annotation loop.

    Parameters
    ----------
    source : int | str | numpy.ndarray
        Webcam index, video path, or a static BGR image.

    Returns
    -------
    None
    """
    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    mouth_open = False
    sync_count = 0
    frame_ts   = 0
    ts_step    = int(1000 / max(VIDEO_FPS, 1))

    with make_face_landmarker(running_mode, num_faces=1) as detector:

        def process(frame):
            nonlocal mouth_open, sync_count, frame_ts
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += ts_step
                result = detector.detect_for_video(mp_img, frame_ts)

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, lms,
                    FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style(),
                )

                # T1: smile
                smiling = detect_smile(lms, w, h)
                draw_text(frame, f"Mood: {'Smiling' if smiling else 'Neutral'}",
                          (20, 40),
                          color=(0, 255, 100) if smiling else (200, 200, 200))

                # T2: MAR + open-mouth border
                mar = compute_mar(lms, w, h)
                draw_text(frame, f"MAR: {mar:.2f}", (20, 75))
                if mar > MAR_OPEN_THRESHOLD:
                    draw_red_border(frame)

                # T3: lip-sync hysteresis counter
                if not mouth_open and mar > MAR_OPEN_THRESHOLD:
                    mouth_open = True
                elif mouth_open and mar < MAR_CLOSE_THRESHOLD:
                    mouth_open = False
                    sync_count += 1

                draw_text(frame, f"Lip-Sync Count: {sync_count}", (20, 110))
                draw_text(frame, "Mouth: OPEN" if mouth_open else "Mouth: closed",
                          (20, 145),
                          color=(0, 100, 255) if mouth_open else (180, 180, 180))
            else:
                draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255))

            draw_esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 1 -- Lips Detection")