"""
face_module.py
==============
Module 3 — Face Detection (MediaPipe Face Landmarker, Tasks API + DeepFace).

Features
--------
T1 Emotion recognition : DeepFace.analyze() on the face ROI every
                         EMOTION_INFERENCE_EVERY frames; dominant emotion shown.
T2 Head pose           : nose-tip vs inter-eye midpoint geometry → 5 directions.
T3 Recent mood         : rolling EMOTION_HISTORY_SECONDS window, most-common label.

The face ROI is derived directly from the landmarks (no separate Haar cascade).
DeepFace requires the `tf-keras` package on TensorFlow >= 2.16.
Landmark indices come from vision/landmarks.py; constants from vision_config.py.
"""

import time
import collections

import numpy as np

from mediapipe.tasks.python.vision import RunningMode

from vision.vision_config import (
    EMOTION_INFERENCE_EVERY,
    EMOTION_HISTORY_SECONDS,
    VIDEO_FPS,
)
from vision.landmarks import (
    FACE_NOSE_TIP, FACE_LEFT_EYE_CORNER, FACE_RIGHT_EYE_CORNER,
)
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.vision_utils import (
    lm_to_px, euclidean, draw_text, draw_esc_hint, detection_loop,
)


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def classify_head_pose(landmarks, w: int, h: int) -> str:
    """
    Estimate coarse head orientation from facial geometry.

    Compares the nose tip to the inter-eye midpoint, normalised by the
    inter-eye distance so thresholds are scale-invariant.

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
    str
        One of "Left", "Right", "Up", "Down", "Forward".
    """
    nose  = lm_to_px(landmarks[FACE_NOSE_TIP],        w, h)
    eye_l = lm_to_px(landmarks[FACE_LEFT_EYE_CORNER], w, h)
    eye_r = lm_to_px(landmarks[FACE_RIGHT_EYE_CORNER], w, h)

    mid_x    = (eye_l[0] + eye_r[0]) // 2
    mid_y    = (eye_l[1] + eye_r[1]) // 2
    dx       = nose[0] - mid_x
    dy       = nose[1] - mid_y
    eye_dist = euclidean(eye_l, eye_r)

    if abs(dx) > abs(dy):
        if dx < -eye_dist * 0.18:
            return "Left"
        if dx > eye_dist * 0.18:
            return "Right"
    else:
        if dy < -eye_dist * 0.12:
            return "Up"
        if dy > eye_dist * 0.12:
            return "Down"
    return "Forward"


def bbox_from_landmarks(landmarks, w: int, h: int, pad: float = 0.10) -> tuple:
    """
    Compute a padded face bounding box from the landmark extent.

    Parameters
    ----------
    landmarks : list
        Face landmark list.
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.
    pad : float
        Fraction of box size to pad on each side.

    Returns
    -------
    tuple[int, int, int, int]
        (x, y, w, h) clamped to the frame.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs) * w, max(xs) * w
    y_min, y_max = min(ys) * h, max(ys) * h

    bw, bh = x_max - x_min, y_max - y_min
    x_min -= bw * pad; x_max += bw * pad
    y_min -= bh * pad; y_max += bh * pad

    x = max(0, int(x_min))
    y = max(0, int(y_min))
    return x, y, min(w - x, int(x_max - x_min)), min(h - y, int(y_max - y_min))


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_face(source) -> None:
    """
    Launch Module 3 — Face Detection and run the live annotation loop.

    Parameters
    ----------
    source : int | str | numpy.ndarray
        Webcam index, video path, or a static BGR image.

    Returns
    -------
    None
    """
    # Lazy DeepFace import so the app still starts if it is unavailable
    try:
        from deepface import DeepFace
        df_ok = True
    except Exception:
        DeepFace = None
        df_ok = False
        print("  [WARNING] DeepFace unavailable — emotion recognition disabled.")
        print("            Fix: pip install deepface tf-keras")

    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    frame_idx       = 0
    frame_ts        = 0
    ts_step         = int(1000 / max(VIDEO_FPS, 1))
    last_emotion    = "N/A"
    last_confidence = 0.0
    emotion_history = collections.deque()   # (timestamp, emotion)

    with make_face_landmarker(running_mode, num_faces=1) as detector:

        def process(frame):
            nonlocal frame_idx, frame_ts, last_emotion, last_confidence
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += ts_step
                result = detector.detect_for_video(mp_img, frame_ts)

            if result.face_landmarks:
                lms = result.face_landmarks[0]
                bx, by, bw, bh = bbox_from_landmarks(lms, w, h)

                # Draw the face box
                import cv2
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)

                # T1: emotion (throttled)
                if df_ok and frame_idx % EMOTION_INFERENCE_EVERY == 0 and bw > 0 and bh > 0:
                    roi = frame[by:by + bh, bx:bx + bw]
                    if roi.size:
                        try:
                            analysis = DeepFace.analyze(
                                roi, actions=["emotion"],
                                enforce_detection=False, silent=True,
                            )
                            emo = analysis[0]["emotion"]
                            last_emotion    = max(emo, key=emo.get)
                            last_confidence = emo[last_emotion]
                            emotion_history.append((time.time(), last_emotion))
                        except Exception:
                            pass

                draw_text(frame, f"Emotion: {last_emotion} ({last_confidence:.0f}%)",
                          (20, 40))

                # T3: rolling recent-mood
                now = time.time()
                while emotion_history and now - emotion_history[0][0] > EMOTION_HISTORY_SECONDS:
                    emotion_history.popleft()
                recent = (
                    collections.Counter(e for _, e in emotion_history).most_common(1)[0][0]
                    if emotion_history else "N/A"
                )
                draw_text(frame, f"Recent Mood ({EMOTION_HISTORY_SECONDS}s): {recent}",
                          (20, 75), color=(255, 200, 80))

                # T2: head pose
                draw_text(frame, f"Head Pose: {classify_head_pose(lms, w, h)}",
                          (20, 110), color=(100, 255, 255))
            else:
                draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255))

            frame_idx += 1
            draw_esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 3 -- Face Detection")