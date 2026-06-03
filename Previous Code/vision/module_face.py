"""
module_face.py
==============
Module 3 -- Face Detection

Tasks implemented
-----------------
T1 : Emotion recognition  -- DeepFace.analyze() is called every EMOTION_INTERVAL
                              frames on the Haar-detected face ROI; dominant
                              emotion + confidence are displayed.
T2 : Head pose estimation -- FaceLandmarker geometry (nose tip vs inter-eye
                              midpoint) classifies orientation as
                              Left / Right / Up / Down / Forward.
T3 : Emotion history log  -- rolling 5-second deque; most-frequent label is
                              shown as 'Recent Mood'.

MediaPipe component  : FaceLandmarker (Tasks API 0.10.x+)
Haar cascade         : cv2.data.haarcascades / haarcascade_frontalface_default.xml
Optional dependency  : deepface  (pip install deepface)

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
"""

import time
import collections

import cv2

from mediapipe.tasks.python         import BaseOptions
from mediapipe.tasks.python.vision  import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from models import FACE_MODEL_PATH, bgr_to_mp_image
from utils  import lm_to_px, euclidean, draw_text, detection_loop, _esc_hint


# ==============================================================================
#  LANDMARK INDICES
# ==============================================================================

_NOSE_TIP    = 1     # tip of the nose — primary head-pose marker
_EYE_L_OUTER = 33    # left eye outer corner
_EYE_R_OUTER = 263   # right eye outer corner


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

#: Run DeepFace inference only every N frames to maintain real-time performance.
#: DeepFace can take 50–200 ms per call; throttling keeps the UI responsive.
EMOTION_INTERVAL = 5


# ==============================================================================
#  FEATURE FUNCTIONS
# ==============================================================================

def classify_head_pose(landmarks, w, h):
    """
    Estimate the yaw and pitch direction of the head from facial geometry.

    Algorithm
    ---------
    1.  Locate the nose tip and both eye outer corners in pixel coordinates.
    2.  Compute the inter-eye midpoint (proxy for the 'neutral' nose position
        when the head faces forward).
    3.  Measure dx and dy from midpoint to nose tip.
    4.  Normalise dx and dy by the inter-eye distance so the thresholds are
        scale-invariant (work the same for a face 30 cm away and 1 m away).
    5.  Compare to empirically tuned ratios:
            |dx| > 0.18 × inter-eye  → turning Left or Right
            |dy| > 0.12 × inter-eye  → tilting Up or Down
        The dy threshold is smaller (0.12) because vertical head movement
        produces a smaller nose displacement than lateral rotation.

    Why geometry instead of a 3-D pose model?
    ------------------------------------------
    A full PnP-based head pose requires a 3-D face model and intrinsic camera
    parameters.  The geometry approach here is model-free, requires no camera
    calibration, and is robust enough for coarse 5-direction classification
    which is all that HCI applications typically need.

    Parameters
    ----------
    landmarks : list[NormalizedLandmark]
        Full 468-point face landmark list for one detected face.
    w : int  -- frame width in pixels.
    h : int  -- frame height in pixels.

    Returns
    -------
    str
        One of: 'Left', 'Right', 'Up', 'Down', 'Forward'.
    """
    nose  = lm_to_px(landmarks[_NOSE_TIP],    w, h)
    eye_l = lm_to_px(landmarks[_EYE_L_OUTER], w, h)
    eye_r = lm_to_px(landmarks[_EYE_R_OUTER], w, h)

    mid_x    = (eye_l[0] + eye_r[0]) // 2
    mid_y    = (eye_l[1] + eye_r[1]) // 2
    dx       = nose[0] - mid_x
    dy       = nose[1] - mid_y
    eye_dist = euclidean(eye_l, eye_r)

    if abs(dx) > abs(dy):
        if dx < -eye_dist * 0.18:
            return "Left"
        if dx >  eye_dist * 0.18:
            return "Right"
    else:
        if dy < -eye_dist * 0.12:
            return "Up"
        if dy >  eye_dist * 0.12:
            return "Down"
    return "Forward"


# ==============================================================================
#  MODULE ENTRY POINT
# ==============================================================================

def run_face_detection(source):
    """
    Launch Module 3 — Face Detection and run the live annotation loop.

    Setup
    -----
    * Imports DeepFace lazily so the application starts fast even when the
      library is not installed.  If DeepFace is missing, T1 (emotion) is
      disabled but T2 and T3 still work.
    * Creates a FaceLandmarker and an OpenCV Haar cascade for face bounding-box
      detection (the Haar box is used as the DeepFace ROI).
    * Maintains emotion_history as a collections.deque of (timestamp, emotion)
      tuples for the rolling 5-second window.

    Per-frame processing
    --------------------
    1.  Convert to greyscale for Haar; draw bounding boxes.
    2.  T1: Every EMOTION_INTERVAL frames, pass the first Haar ROI to
            DeepFace.analyze(); cache the result in last_emotion /
            last_confidence for display on all other frames.
    3.  Display cached emotion + confidence.
    4.  T3: Prune entries older than 5 s from emotion_history; compute
            the most common label with collections.Counter.
    5.  T2: Run FaceLandmarker; display classified head pose.

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

    # Lazy DeepFace import — kept lazy so the app starts even if not installed
    try:
        from deepface import DeepFace
        _df_ok = True
    except ImportError:
        DeepFace = None
        _df_ok   = False
        print("  [WARNING] DeepFace not installed -- emotion recognition disabled.")

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

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frame_idx       = 0
    frame_ts        = 0
    last_emotion    = "N/A"
    last_confidence = 0.0
    emotion_history = collections.deque()   # (unix_timestamp, emotion_str)

    with FaceLandmarker.create_from_options(options) as detector:

        def process(frame):
            nonlocal frame_idx, frame_ts, last_emotion, last_confidence
            h, w = frame.shape[:2]
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mp_img = bgr_to_mp_image(frame)

            # Haar face bounding boxes
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                              (255, 200, 0), 2)

            # T1: Emotion — only recomputed every EMOTION_INTERVAL frames
            if _df_ok and (frame_idx % EMOTION_INTERVAL == 0) and len(faces):
                fx, fy, fw, fh = faces[0]
                roi = frame[fy:fy + fh, fx:fx + fw]
                try:
                    analysis        = DeepFace.analyze(roi, actions=["emotion"],
                                                       enforce_detection=False,
                                                       silent=True)
                    emo_data        = analysis[0]["emotion"]
                    last_emotion    = max(emo_data, key=emo_data.get)
                    last_confidence = emo_data[last_emotion]
                    emotion_history.append((time.time(), last_emotion))
                except Exception:
                    pass

            draw_text(frame, f"Emotion: {last_emotion} ({last_confidence:.0f}%)",
                      (20, 40))

            # T3: Rolling 5-second emotion summary
            now = time.time()
            while emotion_history and now - emotion_history[0][0] > 5.0:
                emotion_history.popleft()
            recent = (
                collections.Counter(e for _, e in emotion_history).most_common(1)[0][0]
                if emotion_history else "N/A"
            )
            draw_text(frame, f"Recent Mood (5s): {recent}", (20, 75),
                      color=(255, 200, 80))

            # T2: Head pose via FaceLandmarker
            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += 33
                result = detector.detect_for_video(mp_img, frame_ts)

            if result.face_landmarks:
                lms = result.face_landmarks[0]
                draw_text(frame, f"Head Pose: {classify_head_pose(lms, w, h)}",
                          (20, 110), color=(100, 255, 255))
            else:
                draw_text(frame, "Head Pose: N/A", (20, 110), color=(180, 180, 180))

            frame_idx += 1
            _esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 3 -- Face Detection")
