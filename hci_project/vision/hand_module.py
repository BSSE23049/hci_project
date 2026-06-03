"""
hand_module.py
==============
Module 4 — Hand Detection (MediaPipe Hand Landmarker, Tasks API).

Features
--------
T1 Finger counter : counts extended fingers; thumb uses chirality-aware X test.
T2 Gesture label  : maps (non-thumb count, thumb_up) → named gesture via config.
T3 Letter sign    : optional ASL-style letter via config LETTER_GESTURE_MAP.
T4 Gesture game   : hold a random target gesture to score points.

Landmark indices come from vision/landmarks.py.
Gesture/letter maps and game settings come from vision_config.py.
"""

import random
import time

import cv2
import numpy as np

from mediapipe.tasks.python.vision import (
    RunningMode,
    HandLandmarksConnections,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from vision.vision_config import (
    GESTURE_MAP,
    LETTER_GESTURE_MAP,
    ENABLE_LETTER_DETECTION,
    GAME_HOLD_SECONDS,
    MAX_HANDS,
    VIDEO_FPS,
)
from vision.landmarks import HAND_TIP_IDS, HAND_PIP_IDS, HAND_WRIST
from vision.mp_tasks import make_hand_landmarker, bgr_to_mp_image
from vision.vision_utils import lm_to_px, draw_text, draw_esc_hint, detection_loop


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def count_fingers(landmarks, handedness: str, w: int, h: int) -> tuple:
    """
    Count extended fingers on one detected hand.

    Non-thumb fingers are extended when the tip is above the PIP joint.
    The thumb extends sideways; its direction is mirrored by handedness.

    Parameters
    ----------
    landmarks : list
        21 hand landmark objects (.x/.y in [0,1]).
    handedness : str
        "Left" or "Right".
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    tuple[int, bool]
        (non_thumb_count 0-4, thumb_extended).
    """
    pts = [lm_to_px(landmarks[i], w, h) for i in range(21)]

    if handedness == "Right":
        thumb_up = pts[HAND_TIP_IDS[0]][0] < pts[HAND_PIP_IDS[0]][0]
    else:
        thumb_up = pts[HAND_TIP_IDS[0]][0] > pts[HAND_PIP_IDS[0]][0]

    fingers_up = sum(
        1 for i in range(1, 5)
        if pts[HAND_TIP_IDS[i]][1] < pts[HAND_PIP_IDS[i]][1]
    )
    return fingers_up, thumb_up


def classify_gesture(fingers_up: int, thumb_up: bool) -> tuple:
    """
    Map a finger configuration to a gesture label + short tag from GESTURE_MAP.

    Reads ONLY from config GESTURE_MAP — no hardcoded gesture names.

    Parameters
    ----------
    fingers_up : int
        Non-thumb extended finger count (0-4).
    thumb_up : bool
        Whether the thumb is extended.

    Returns
    -------
    tuple[str, str]
        (label, tag), or ("Unknown", "?") if not found.
    """
    return GESTURE_MAP.get(
        (fingers_up, thumb_up),
        GESTURE_MAP.get((fingers_up, False), ("Unknown", "?")),
    )


def detect_letter(fingers_up: int, thumb_up: bool):
    """
    Detect an ASL-style letter from the current hand pose.

    Returns None immediately if ENABLE_LETTER_DETECTION is False.
    Reads ONLY from config LETTER_GESTURE_MAP.

    Parameters
    ----------
    fingers_up : int
        Non-thumb extended finger count.
    thumb_up : bool
        Whether the thumb is extended.

    Returns
    -------
    str | None
        Matched letter, or None.
    """
    if not ENABLE_LETTER_DETECTION:
        return None
    key = (fingers_up, thumb_up)
    for letter, gesture_key in LETTER_GESTURE_MAP.items():
        if gesture_key == key:
            return letter
    return None


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_hands(source, game_mode: bool = False) -> None:
    """
    Launch Module 4 — Hand Detection and run the live annotation loop.

    Parameters
    ----------
    source : int | str | numpy.ndarray
        Webcam index, video path, or a static BGR image.
    game_mode : bool
        If True, show the gesture-matching game HUD.

    Returns
    -------
    None
    """
    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    game_targets = [label for label, _ in GESTURE_MAP.values()]
    target       = random.choice(game_targets)
    score        = 0
    hold_start   = None
    frame_ts     = 0
    ts_step      = int(1000 / max(VIDEO_FPS, 1))

    with make_hand_landmarker(running_mode, num_hands=MAX_HANDS) as detector:

        def process(frame):
            nonlocal target, score, hold_start, frame_ts
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += ts_step
                result = detector.detect_for_video(mp_img, frame_ts)

            current_gestures = []

            if result.hand_landmarks:
                for idx, hand_lms in enumerate(result.hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_lms,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_hand_connections_style(),
                    )

                    if result.handedness and idx < len(result.handedness):
                        handedness = result.handedness[idx][0].display_name
                    else:
                        handedness = "Right"

                    # T1 + T2: fingers and gesture
                    fingers_up, thumb_up = count_fingers(hand_lms, handedness, w, h)
                    total = fingers_up + (1 if thumb_up else 0)
                    label, tag = classify_gesture(fingers_up, thumb_up)
                    current_gestures.append(label)

                    wrist = lm_to_px(hand_lms[HAND_WRIST], w, h)
                    draw_text(frame, f"{handedness}: {total} fingers",
                              (wrist[0] - 60, wrist[1] + 30), color=(50, 255, 50))
                    draw_text(frame, f"{label} [{tag}]",
                              (wrist[0] - 60, wrist[1] + 60), color=(255, 200, 50))

                    # T3: letter sign
                    letter = detect_letter(fingers_up, thumb_up)
                    if letter:
                        draw_text(frame, f"Letter: {letter}",
                                  (wrist[0] - 60, wrist[1] + 90),
                                  color=(255, 160, 0))

            # T4: gesture game HUD
            if game_mode:
                gy = h - 90
                draw_text(frame, f"TARGET: {target}", (20, gy),
                          color=(0, 200, 255), scale=0.8)
                draw_text(frame, f"Score: {score}", (20, gy + 35),
                          color=(0, 255, 180), scale=0.8)

                if target in current_gestures:
                    if hold_start is None:
                        hold_start = time.time()
                    progress = min((time.time() - hold_start) / GAME_HOLD_SECONDS, 1.0)

                    bx, by, bw_px, bh_px = 20, gy + 55, 200, 12
                    cv2.rectangle(frame, (bx, by), (bx + bw_px, by + bh_px),
                                  (60, 60, 60), -1)
                    cv2.rectangle(frame, (bx, by),
                                  (bx + int(bw_px * progress), by + bh_px),
                                  (0, 220, 120), -1)

                    if progress >= 1.0:
                        score += 1
                        target = random.choice(game_targets)
                        hold_start = None
                else:
                    hold_start = None

            draw_esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 4 -- Hand Detection")