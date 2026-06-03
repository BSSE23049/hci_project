"""
eyes_module.py
==============
Module 2 — Eyes Detection (MediaPipe Face Landmarker, Tasks API).

Features
--------
T1 Blink counter     : a blink = EAR drops below personal threshold then rises.
T2 EAR display       : per-eye Eye Aspect Ratio shown each frame.
T3 Drowsiness alert  : red banner after N consecutive low-EAR frames.

Personal calibration
--------------------
Fixed thresholds fail for small-eyed users whose baseline EAR is naturally
lower than the textbook 0.28–0.35 range.  On startup the module samples the
user's own open-eye EAR for CALIBRATION_FRAMES frames, then sets:

    closed_threshold = clamp(baseline × EAR_CLOSED_RATIO,
                             EAR_FLOOR, EAR_CEILING)

This self-calibration requires no manual tuning and works correctly for
small eyes, large eyes, and glasses wearers.

Reference: Soukupova & Cech (2016), Real-time Eye Blink Detection.
Landmark indices come from vision/landmarks.py; all tunables from vision_config.py.
"""

import numpy as np

from mediapipe.tasks.python.vision import (
    RunningMode,
    FaceLandmarksConnections,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from vision.vision_config import (
    CALIBRATION_FRAMES,
    EAR_CLOSED_RATIO,
    EAR_FLOOR,
    EAR_CEILING,
    EAR_CLOSED_THRESHOLD,   # fallback only
    DROWSY_FRAMES_TRIGGER,
    DROWSY_CLEAR_FRAMES,
    VIDEO_FPS,
)
from vision.landmarks import FACE_LEFT_EYE, FACE_RIGHT_EYE
from vision.mp_tasks import make_face_landmarker, bgr_to_mp_image
from vision.vision_utils import (
    lm_to_px, euclidean,
    draw_text, draw_red_banner, draw_esc_hint, detection_loop,
)


# ---------------------------------------------------------------------------
# EAR computation
# ---------------------------------------------------------------------------

def compute_ear(landmarks, eye_indices: list, w: int, h: int) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye using 6 landmarks.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    landmarks : list
        Face landmark list from FaceLandmarker.
    eye_indices : list[int]
        Six landmark indices in order [p1, p2, p3, p4, p5, p6].
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    float
        EAR value >= 0. Lower = more closed.
    """
    pts    = [lm_to_px(landmarks[i], w, h) for i in eye_indices]
    v1     = euclidean(pts[1], pts[5])
    v2     = euclidean(pts[2], pts[4])
    h_dist = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h_dist + 1e-6)


# ---------------------------------------------------------------------------
# Personal calibration
# ---------------------------------------------------------------------------

def _calibrate_threshold(samples: list) -> float:
    """
    Derive a personal closed-eye threshold from open-eye EAR samples.

    Rejects obvious outliers (bottom 10 % of samples, which may include
    frames captured mid-blink) before computing the baseline mean.  The
    threshold is then clamped to [EAR_FLOOR, EAR_CEILING] so a bad
    calibration session can't produce an unusable value.

    Parameters
    ----------
    samples : list[float]
        Raw EAR values collected while the user's eyes were open.

    Returns
    -------
    float
        Personal closed-eye threshold.
    """
    if not samples:
        return EAR_CLOSED_THRESHOLD   # nothing collected — use config fallback

    arr = sorted(samples)
    # drop bottom 10 % (blink frames captured during calibration)
    trim = max(1, int(len(arr) * 0.10))
    baseline = float(np.mean(arr[trim:]))

    threshold = baseline * EAR_CLOSED_RATIO
    return float(np.clip(threshold, EAR_FLOOR, EAR_CEILING))


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_eyes(source) -> None:
    """
    Launch Module 2 — Eyes Detection with personal EAR calibration.

    Calibration phase
    -----------------
    The first CALIBRATION_FRAMES valid face frames are used to measure the
    user's natural open-eye EAR.  A green progress bar and instruction text
    are shown during this phase.  After calibration the derived threshold is
    displayed so the user can confirm it looks sensible.

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

    # ---- state ----
    cal_samples: list  = []            # EAR samples during calibration
    calibrated         = is_image      # skip calibration for static images
    closed_thresh      = EAR_CLOSED_THRESHOLD  # replaced after calibration

    blink_count   = 0
    eye_closed    = False
    drowsy_frames = 0
    normal_frames = 0
    drowsy_alert  = False
    frame_ts      = 0
    ts_step       = int(1000 / max(VIDEO_FPS, 1))

    with make_face_landmarker(running_mode, num_faces=1) as detector:

        def process(frame):
            nonlocal cal_samples, calibrated, closed_thresh
            nonlocal blink_count, eye_closed
            nonlocal drowsy_frames, normal_frames, drowsy_alert, frame_ts

            import cv2
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += ts_step
                result = detector.detect_for_video(mp_img, frame_ts)

            # ------------------------------------------------------------------
            # CALIBRATION PHASE
            # ------------------------------------------------------------------
            if not calibrated:
                if result.face_landmarks:
                    lms   = result.face_landmarks[0]
                    ear_l = compute_ear(lms, FACE_LEFT_EYE,  w, h)
                    ear_r = compute_ear(lms, FACE_RIGHT_EYE, w, h)
                    avg   = (ear_l + ear_r) / 2.0
                    if avg > 0.05:   # discard frames where face barely visible
                        cal_samples.append(avg)

                collected = len(cal_samples)

                # Progress bar
                bar_w = int(w * 0.6)
                bx    = (w - bar_w) // 2
                by    = h // 2 + 20
                progress = min(collected / CALIBRATION_FRAMES, 1.0)

                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16),
                              (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by),
                              (bx + int(bar_w * progress), by + 16),
                              (0, 200, 80), -1)

                draw_text(frame, "Calibrating — keep eyes OPEN",
                          (bx, h // 2 - 10), color=(0, 255, 150), scale=0.75)
                draw_text(frame,
                          f"{collected}/{CALIBRATION_FRAMES} frames",
                          (bx, by + 38), color=(200, 200, 200), scale=0.6)

                if collected >= CALIBRATION_FRAMES:
                    closed_thresh = _calibrate_threshold(cal_samples)
                    calibrated    = True
                    print(f"[Eyes] Calibration complete. "
                          f"Baseline EAR ≈ {np.mean(cal_samples):.3f}  |  "
                          f"Closed threshold = {closed_thresh:.3f}")

                draw_esc_hint(frame)
                return frame

            # ------------------------------------------------------------------
            # DETECTION PHASE
            # ------------------------------------------------------------------
            if result.face_landmarks:
                lms = result.face_landmarks[0]

                # Draw eye outlines using the official Tasks drawing API
                for conn in (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                             FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE):
                    mp_drawing.draw_landmarks(
                        frame, lms, conn,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style(),
                    )

                # T2: per-eye EAR
                ear_l = compute_ear(lms, FACE_LEFT_EYE,  w, h)
                ear_r = compute_ear(lms, FACE_RIGHT_EYE, w, h)
                avg   = (ear_l + ear_r) / 2.0

                draw_text(frame, f"EAR L: {ear_l:.3f}", (20, 40))
                draw_text(frame, f"EAR R: {ear_r:.3f}", (20, 75))
                draw_text(frame, f"Threshold: {closed_thresh:.3f}",
                          (20, 110), color=(180, 180, 180), scale=0.55)

                # T1: blink detection (using personal threshold)
                if avg < closed_thresh:
                    eye_closed = True
                elif eye_closed:
                    blink_count += 1
                    eye_closed   = False
                draw_text(frame, f"Blinks: {blink_count}", (20, 135),
                          color=(0, 220, 255))

                # T3: drowsiness with enter/clear hysteresis
                if avg < closed_thresh:
                    drowsy_frames += 1
                    normal_frames  = 0
                else:
                    normal_frames += 1
                    drowsy_frames  = 0

                if drowsy_frames >= DROWSY_FRAMES_TRIGGER:
                    drowsy_alert = True
                if normal_frames >= DROWSY_CLEAR_FRAMES:
                    drowsy_alert = False

                if drowsy_alert:
                    draw_red_banner(frame, "DROWSY! Wake Up!")
            else:
                draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255))

            draw_esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 2 -- Eyes Detection")
