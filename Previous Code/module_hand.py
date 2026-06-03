"""
module_hand.py
==============
Module 4 -- Hand Detection

Tasks implemented
-----------------
T1 : Finger counter  -- counts extended fingers per hand using tip vs PIP
                         landmark Y-comparison; thumb uses X-axis with
                         chirality-aware direction.
T2 : Gesture label   -- maps (non-thumb count, thumb_up) to a named gesture
                         and a short display tag.
T3 : Gesture game    -- displays a random target gesture; awards 1 point when
                         the user holds the correct gesture for HOLD_REQUIRED
                         seconds; shows a progress bar.

MediaPipe component: HandLandmarker (Tasks API 0.10.x+)

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
"""

import random
import time

import cv2

from mediapipe.tasks.python         import BaseOptions
from mediapipe.tasks.python.vision  import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils  as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from models import HAND_MODEL_PATH, bgr_to_mp_image
from utils  import lm_to_px, draw_text, detection_loop, _esc_hint


# ==============================================================================
#  LANDMARK INDICES
# ==============================================================================

#: Fingertip landmark indices (thumb=4, index=8, middle=12, ring=16, pinky=20)
_FINGER_TIPS = [4,  8,  12, 16, 20]

#: Corresponding PIP (Proximal InterPhalangeal) joint indices used as reference
#: for the 'is finger extended' check.
_FINGER_PIPS = [3,  6,  10, 14, 18]


# ==============================================================================
#  GESTURE LOOKUP TABLE
# ==============================================================================

# Key: (non_thumb_fingers_extended: int, thumb_up: bool)
# Value: (gesture_label: str, short_display_tag: str)
_GESTURE_MAP = {
    (0, False): ("Fist",      "OK"),
    (0, True):  ("Thumbs Up", "+1"),
    (1, False): ("One",       "1"),
    (2, False): ("Peace",     "V"),
    (3, False): ("Three",     "3"),
    (4, False): ("Four",      "4"),
    (4, True):  ("Open Hand", "HI"),
    (5, True):  ("High Five", "5"),
}

#: Subset of gesture labels used as valid game targets
_GAME_GESTURES = ["Fist", "One", "Peace", "Three", "Four", "Open Hand", "Thumbs Up"]

#: Seconds the user must hold the correct gesture to score a point
HOLD_REQUIRED = 1.0


# ==============================================================================
#  FEATURE FUNCTIONS
# ==============================================================================

def count_fingers(landmarks, handedness, w, h):
    """
    Count how many fingers are extended on one detected hand.

    Algorithm
    ---------
    For each of the four non-thumb fingers (index, middle, ring, pinky) the
    test is simple: if the fingertip Y-coordinate is ABOVE (smaller Y in image
    coordinates) the corresponding PIP joint, the finger is extended.  This
    works because in a canonical 'show your hand' pose the fingertips point
    upward relative to the knuckles when extended.

    The THUMB is special for two reasons:
      1. It extends sideways, not upward, so the Y test fails.
      2. The direction of sideways extension is mirrored between left and right
         hands (right thumb tip moves to the LEFT when extended; left thumb tip
         moves to the RIGHT).

    For the right hand  : thumb is up when tip_x > PIP_x (tip is to the left).
    For the left hand   : thumb is up when tip_x < PIP_x (tip is to the right).

    Why PIP and not MCP?
    --------------------
    The MCP (knuckle) joint lies close to the palm and can appear above a
    slightly curled finger.  The PIP joint is further along the finger and
    gives a more reliable bent-vs-straight signal.

    Parameters
    ----------
    landmarks   : list[NormalizedLandmark]
        21 hand landmarks from HandLandmarkerResult.hand_landmarks[hand_index].
    handedness  : str
        'Left' or 'Right' from HandLandmarkerResult.handedness[hand_index][0].display_name.
    w : int  -- frame width in pixels.
    h : int  -- frame height in pixels.

    Returns
    -------
    tuple (int, bool)
        * int  -- number of non-thumb fingers extended (0–4).
        * bool -- True if thumb is extended.
    """
    pts = [lm_to_px(landmarks[i], w, h) for i in range(21)]

    # Thumb chirality-aware X-axis check
    # This is work for back side of hand
    # =======================================
    # If you convert change sign (Palm Facing)
    # ======================================
    if handedness == "Right":
        thumb_up = pts[_FINGER_TIPS[0]][0] < pts[_FINGER_PIPS[0]][0]
    else:
        thumb_up = pts[_FINGER_TIPS[0]][0] > pts[_FINGER_PIPS[0]][0]

    # Non-thumb fingers: tip Y < PIP Y  →  extended
    fingers_up = sum(
        1 for i in range(1, 5)
        if pts[_FINGER_TIPS[i]][1] < pts[_FINGER_PIPS[i]][1]
    )
    return fingers_up, thumb_up


def classify_gesture(fingers_up, thumb_up):
    """
    Map a finger-count configuration to a named gesture and short tag.

    The lookup first tries the exact key (fingers_up, thumb_up).  If that
    fails it falls back to (fingers_up, False) so partially ambiguous
    configurations still produce a reasonable label.  If neither key is in
    the table it returns ('Unknown', '?').

    Parameters
    ----------
    fingers_up : int   -- non-thumb extended finger count (0–4).
    thumb_up   : bool  -- True if the thumb is extended.

    Returns
    -------
    tuple (str, str)
        * str -- human-readable gesture label (e.g. 'Peace').
        * str -- short display tag (e.g. 'V').
    """
    return _GESTURE_MAP.get(
        (fingers_up, thumb_up),
        _GESTURE_MAP.get((fingers_up, False), ("Unknown", "?"))
    )


# ==============================================================================
#  MODULE ENTRY POINT
# ==============================================================================

def run_hand_detection(source):
    """
    Launch Module 4 — Hand Detection and run the live annotation loop.

    Setup
    -----
    Creates a HandLandmarker that can track up to 2 hands simultaneously.
    Game state (target gesture, score, hold timer) is kept in the enclosing
    scope and accessed via nonlocal inside process().

    Per-frame processing
    --------------------
    For each detected hand:
        1.  Draw the full hand skeleton with Tasks drawing API.
        2.  Determine handedness from result.handedness.
        3.  T1: Count fingers with count_fingers().
        4.  T2: Classify gesture with classify_gesture(); annotate near wrist.

    Game (T3):
        5.  Show current target and score in the lower-left.
        6.  If the target gesture appears among current_gestures:
              - Start or continue a hold timer.
              - Draw a progress bar (width proportional to hold fraction).
              - When progress reaches 1.0: increment score, pick new target,
                reset hold_start.
        7.  If the target gesture is NOT seen, reset hold_start.

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

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    target_gesture = random.choice(_GAME_GESTURES)
    score          = 0
    hold_start     = None
    frame_ts       = 0

    with HandLandmarker.create_from_options(options) as detector:

        def process(frame):
            nonlocal target_gesture, score, hold_start, frame_ts
            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            if is_image:
                result = detector.detect(mp_img)
            else:
                frame_ts += 33
                result = detector.detect_for_video(mp_img, frame_ts)

            current_gestures = []

            if result.hand_landmarks:
                for idx, hand_lms in enumerate(result.hand_landmarks):

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_hand_connections_style(),
                    )

                    # Handedness string from result
                    if result.handedness and idx < len(result.handedness):
                        handedness = result.handedness[idx][0].display_name
                    else:
                        handedness = "Right"

                    # T1: Finger count
                    fingers_up, thumb_up = count_fingers(hand_lms, handedness, w, h)
                    total = fingers_up + (1 if thumb_up else 0)

                    # T2: Gesture label
                    gesture_name, tag = classify_gesture(fingers_up, thumb_up)
                    current_gestures.append(gesture_name)

                    # Annotate near the wrist landmark
                    wrist = lm_to_px(hand_lms[0], w, h)
                    draw_text(frame, f"{handedness}: {total} fingers",
                              (wrist[0] - 60, wrist[1] + 30), color=(50, 255, 50))
                    draw_text(frame, f"{gesture_name} [{tag}]",
                              (wrist[0] - 60, wrist[1] + 60), color=(255, 200, 50))

            # T3: Gesture Game HUD
            game_y = h - 90
            draw_text(frame, f"TARGET: {target_gesture}", (20, game_y),
                      color=(0, 200, 255), scale=0.8)
            draw_text(frame, f"Score: {score}", (20, game_y + 35),
                      color=(0, 255, 180), scale=0.8)

            if target_gesture in current_gestures:
                if hold_start is None:
                    hold_start = time.time()
                progress = min((time.time() - hold_start) / HOLD_REQUIRED, 1.0)

                # Progress bar
                bx, by, bw_px, bh_px = 20, game_y + 55, 200, 12
                cv2.rectangle(frame, (bx, by),
                              (bx + bw_px, by + bh_px), (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by),
                              (bx + int(bw_px * progress), by + bh_px),
                              (0, 220, 120), -1)

                if progress >= 1.0:
                    score         += 1
                    target_gesture = random.choice(_GAME_GESTURES)
                    hold_start     = None
            else:
                hold_start = None

            _esc_hint(frame)
            return frame

        detection_loop(source, process, "Module 4 -- Hand Detection")
