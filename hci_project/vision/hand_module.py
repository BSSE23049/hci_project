"""
hand_module.py
==============
Module 4 -- Hand Detection (MediaPipe Hand Landmarker, Tasks API).

Tasks
-----
T1 Finger counter : counts extended fingers; thumb uses palm-facing X-axis check.
T2 Gesture label  : maps (non-thumb count, thumb_up) -> label + tag via config.
T3 Letter sign    : optional ASL-style letter lookup via config LETTER_GESTURE_MAP.
T4 Gesture game   : hold the target gesture for GAME_HOLD_SECONDS to score.

Palm-facing vs back-facing
--------------------------
The thumb check is calibrated for the NATURAL 'palm facing the camera' pose:
    Right hand: thumb tip is to the RIGHT of the PIP joint  -> tip_x > pip_x
    Left  hand: thumb tip is to the LEFT  of the PIP joint  -> tip_x < pip_x
(Back-facing is the opposite sign.)

All gesture/letter maps and game settings come from vision/vision_config.py.
Landmark indices come from vision/landmarks.py.
"""

import random
import time

import cv2
import numpy as np

from mediapipe.tasks.python.vision import (
    RunningMode,
    HandLandmarksConnections,
    drawing_utils  as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from vision.vision_config import (
    GESTURE_MAP,
    LETTER_GESTURE_MAP,
    ENABLE_LETTER_DETECTION,
    ENABLE_GESTURE_GAME,
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
    Count extended fingers on one detected hand (palm facing the camera).

    Non-thumb fingers: extended when fingertip Y < PIP Y (tip above knuckle).
    Thumb: uses X-axis with chirality correction for palm-facing pose.
        Right hand -- thumb tip is to the RIGHT of PIP when extended (tip_x > pip_x)
        Left  hand -- thumb tip is to the LEFT  of PIP when extended (tip_x < pip_x)

    Parameters
    ----------
    landmarks : list
        21 hand landmark objects (.x/.y in [0,1]) from the Tasks API result.
    handedness : str
        "Left" or "Right" from HandLandmarkerResult.
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    tuple[int, bool]
        (non_thumb_fingers_extended 0-4, thumb_extended).
    """
    pts = [lm_to_px(landmarks[i], w, h) for i in range(21)]

    # Palm-facing thumb check (flip signs vs back-facing)
    if handedness == "Right":
        thumb_up = pts[HAND_TIP_IDS[0]][0] > pts[HAND_PIP_IDS[0]][0]
    else:
        thumb_up = pts[HAND_TIP_IDS[0]][0] < pts[HAND_PIP_IDS[0]][0]

    # Non-thumb fingers: tip Y < PIP Y -> extended
    fingers_up = sum(
        1 for i in range(1, 5)
        if pts[HAND_TIP_IDS[i]][1] < pts[HAND_PIP_IDS[i]][1]
    )
    return fingers_up, thumb_up


def classify_gesture(fingers_up: int, thumb_up: bool) -> tuple:
    """
    Map a finger configuration to a gesture label + tag from config GESTURE_MAP.

    Tries the exact key first, then falls back to (fingers_up, False) so
    partially ambiguous poses still produce a sensible label.
    Contains NO hardcoded gesture names — all labels come from GESTURE_MAP.

    Parameters
    ----------
    fingers_up : int
        Non-thumb extended finger count (0-4).
    thumb_up : bool
        Whether the thumb is extended.

    Returns
    -------
    tuple[str, str]
        (label, tag), or ("Unknown", "?") if not in the map.
    """
    return GESTURE_MAP.get(
        (fingers_up, thumb_up),
        GESTURE_MAP.get((fingers_up, False), ("Unknown", "?")),
    )


def detect_letter(fingers_up: int, thumb_up: bool):
    """
    Detect an ASL-style letter from the current hand pose via config LETTER_GESTURE_MAP.

    Returns None immediately if ENABLE_LETTER_DETECTION is False.

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
    for letter, key in LETTER_GESTURE_MAP.items():
        if key == (fingers_up, thumb_up):
            return letter
    return None


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_hands(source) -> None:
    """
    Launch Module 4 -- Hand Detection and run the live annotation loop.

    Game state (target, score, hold timer) is kept as nonlocal variables
    in the enclosing scope -- no separate class needed.

    If ENABLE_GESTURE_GAME is True in vision_config.py, the game HUD is
    always shown. Set it to False to hide the game entirely.

    Parameters
    ----------
    source : int | str | numpy.ndarray
        Webcam device index, video path, or a static BGR image.

    Returns
    -------
    None
    """
    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    # Game state -- inline, no class
    game_labels    = [label for label, _ in GESTURE_MAP.values()]
    target_gesture = random.choice(game_labels)
    score          = 0
    hold_start     = None
    frame_ts       = 0
    ts_step        = int(1000 / max(VIDEO_FPS, 1))

    with make_hand_landmarker(running_mode, num_hands=MAX_HANDS) as detector:

        def process(frame):
            nonlocal target_gesture, score, hold_start, frame_ts

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

                    # Draw skeleton with the official Tasks drawing API
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_hand_connections_style(),
                    )

                    # Handedness label from result
                    if result.handedness and idx < len(result.handedness):
                        handedness = result.handedness[idx][0].display_name
                    else:
                        handedness = "Right"

                    # T1: finger count (palm-facing)
                    fingers_up, thumb_up = count_fingers(hand_lms, handedness, w, h)
                    total = fingers_up + (1 if thumb_up else 0)

                    # T2: gesture label from config
                    gesture_name, tag = classify_gesture(fingers_up, thumb_up)
                    current_gestures.append(gesture_name)

                    # Annotate near the wrist
                    wrist = lm_to_px(hand_lms[HAND_WRIST], w, h)
                    draw_text(frame, f"{handedness}: {total} fingers",
                              (wrist[0] - 60, wrist[1] + 30), color=(50, 255, 50))
                    draw_text(frame, f"{gesture_name} [{tag}]",
                              (wrist[0] - 60, wrist[1] + 60), color=(255, 200, 50))

                    # T3: letter sign
                    letter = detect_letter(fingers_up, thumb_up)
                    if letter:
                        draw_text(frame, f"Letter: {letter}",
                                  (wrist[0] - 60, wrist[1] + 90),
                                  color=(255, 160, 0))

            # T4: gesture game HUD
            if ENABLE_GESTURE_GAME:
                gy = h - 90
                draw_text(frame, f"TARGET: {target_gesture}", (20, gy),
                          color=(0, 200, 255), scale=0.8)
                draw_text(frame, f"Score: {score}", (20, gy + 35),
                          color=(0, 255, 180), scale=0.8)

                if target_gesture in current_gestures:
                    if hold_start is None:
                        hold_start = time.time()
                    progress = min((time.time() - hold_start) / GAME_HOLD_SECONDS, 1.0)

                    bx, by, bw_px, bh_px = 20, gy + 55, 200, 12
                    cv2.rectangle(frame, (bx, by),
                                  (bx + bw_px, by + bh_px), (60, 60, 60), -1)
                    cv2.rectangle(frame, (bx, by),
                                  (bx + int(bw_px * progress), by + bh_px),
                                  (0, 220, 120), -1)

                    if progress >= 1.0:
                        score          += 1
                        target_gesture  = random.choice(game_labels)
                        hold_start      = None
                else:
                    hold_start = None

            draw_esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 4 -- Hand Detection")
