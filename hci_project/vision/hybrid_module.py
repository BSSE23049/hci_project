"""
hybrid_module.py
================
Module 5 — Hybrid Mode (MediaPipe Face + Hand Landmarker, Tasks API).

Runs any combination of the four feature sets (lips, eyes, face, hands)
on a single live frame.  Which features are active is controlled entirely
by HYBRID_ACTIVE_MODULES in vision_config.py — no code changes needed.

Design notes
------------
* The FaceLandmarker is shared across lips / eyes / face so only one
  inference call is made per frame for all three face feature sets.
* The HandLandmarker is only opened when "hands" is active.
* Text is stacked top-left with colour-coded section headers so every
  active feature's stats are visible at a glance.
* Eye calibration runs in the background; other features keep annotating
  the frame normally during the calibration period.
* contextlib.ExitStack manages detector lifetimes so both landmarkers
  are always cleanly closed regardless of how the loop exits.
"""

import collections
import random
import time
from contextlib import ExitStack

import cv2
import numpy as np

from mediapipe.tasks.python.vision import (
    RunningMode,
    FaceLandmarksConnections,
    HandLandmarksConnections,
    drawing_utils  as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

from vision.vision_config import (
    HYBRID_ACTIVE_MODULES,
    # lips
    MAR_OPEN_THRESHOLD, MAR_CLOSE_THRESHOLD, SMILE_CORNER_THRESHOLD,
    # eyes
    CALIBRATION_FRAMES, EAR_CLOSED_RATIO, EAR_FLOOR, EAR_CEILING,
    EAR_CLOSED_THRESHOLD, DROWSY_FRAMES_TRIGGER, DROWSY_CLEAR_FRAMES,
    # face
    EMOTION_INFERENCE_EVERY, EMOTION_HISTORY_SECONDS,
    # hands
    GESTURE_MAP, LETTER_GESTURE_MAP, ENABLE_LETTER_DETECTION,
    ENABLE_GESTURE_GAME, GAME_HOLD_SECONDS, MAX_HANDS,
    # shared
    VIDEO_FPS,
)
from vision.landmarks import (
    FACE_LIP_TOP, FACE_LIP_BOTTOM, FACE_LIP_LEFT_CORNER, FACE_LIP_RIGHT_CORNER,
    FACE_LEFT_EYE, FACE_RIGHT_EYE,
    FACE_NOSE_TIP, FACE_LEFT_EYE_CORNER, FACE_RIGHT_EYE_CORNER,
    HAND_TIP_IDS, HAND_PIP_IDS, HAND_WRIST,
)
from vision.mp_tasks import make_face_landmarker, make_hand_landmarker, bgr_to_mp_image
from vision.vision_utils import (
    lm_to_px, draw_text, draw_red_border, draw_red_banner, draw_esc_hint,
    detection_loop,
)
# Pure-logic helpers reused from the individual modules
from vision.lips_module import compute_mar, detect_smile
from vision.eyes_module import compute_ear, _calibrate_threshold
from vision.face_module import classify_head_pose, bbox_from_landmarks
from vision.hand_module import count_fingers, classify_gesture, detect_letter


# ---------------------------------------------------------------------------
# Colour per section header (BGR)
# ---------------------------------------------------------------------------
_HEADER_COLOR = {
    "lips":  (200, 180, 255),
    "eyes":  (150, 255, 180),
    "face":  (100, 215, 255),
    "hands": (255, 210, 120),
}

_TEXT_X     = 12   # left-edge x for all section text
_LINE_H     = 24   # pixels between data lines
_HEADER_H   = 27   # pixels below a section header before the first data line
_GAP_H      = 6    # extra gap between sections


def run_hybrid(source) -> None:
    """
    Launch Module 5 — Hybrid Mode, all HYBRID_ACTIVE_MODULES on one panel.

    Active modules are read from vision_config.HYBRID_ACTIVE_MODULES.
    Any non-empty subset of ["lips", "eyes", "face", "hands"] is valid.

    Parameters
    ----------
    source : int | str | numpy.ndarray
        Webcam device index, video file path, or a static BGR image.
    """
    active = {m.lower().strip() for m in HYBRID_ACTIVE_MODULES}
    active &= {"lips", "eyes", "face", "hands"}   # ignore unknown entries

    if not active:
        print("[Hybrid] HYBRID_ACTIVE_MODULES is empty — nothing to run.")
        print("         Set at least one of: lips, eyes, face, hands.")
        return

    active_list = [m for m in ["lips", "eyes", "face", "hands"] if m in active]
    print(f"[Hybrid] Active features: {', '.join(active_list)}")

    is_image     = isinstance(source, np.ndarray)
    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO
    ts_step      = int(1000 / max(VIDEO_FPS, 1))

    need_face = bool(active & {"lips", "eyes", "face"})
    need_hand = "hands" in active

    # ------------------------------------------------------------------
    # Optional DeepFace (face emotion)
    # ------------------------------------------------------------------
    DeepFace = None
    df_ok    = False
    if "face" in active:
        try:
            from deepface import DeepFace as _df
            DeepFace = _df
            df_ok    = True
        except Exception:
            print("  [Hybrid] DeepFace unavailable — emotion recognition disabled.")

    # ------------------------------------------------------------------
    # Per-feature state (persists across frames via nonlocal)
    # ------------------------------------------------------------------

    # Lips
    mouth_open = False
    sync_count = 0

    # Eyes
    cal_samples: list = []
    calibrated        = is_image   # static images skip calibration
    closed_thresh     = EAR_CLOSED_THRESHOLD
    blink_count       = 0
    eye_closed_state  = False
    drowsy_frames     = 0
    normal_frames     = 0
    drowsy_alert      = False

    # Face
    face_frame_idx  = 0
    last_emotion    = "N/A"
    last_conf       = 0.0
    emo_history     = collections.deque()   # (timestamp, emotion)

    # Hands / game
    game_labels    = [label for label, _ in GESTURE_MAP.values()]
    target_gesture = random.choice(game_labels)
    game_score     = 0
    hold_start     = None

    # Shared timestamps for detect_for_video
    face_ts = 0
    hand_ts = 0

    # ------------------------------------------------------------------
    # Build window title from active features
    # ------------------------------------------------------------------
    tag          = " + ".join(m.capitalize() for m in active_list)
    window_title = f"Module 5 -- Hybrid Mode  [{tag}]"

    # ------------------------------------------------------------------
    # Open detectors via ExitStack (both closed cleanly on any exit path)
    # ------------------------------------------------------------------
    with ExitStack() as stack:
        face_det = (
            stack.enter_context(make_face_landmarker(running_mode, num_faces=1))
            if need_face else None
        )
        hand_det = (
            stack.enter_context(make_hand_landmarker(running_mode, num_hands=MAX_HANDS))
            if need_hand else None
        )

        # --------------------------------------------------------------
        # Per-frame processing closure
        # --------------------------------------------------------------
        def process(frame):
            nonlocal mouth_open, sync_count
            nonlocal cal_samples, calibrated, closed_thresh
            nonlocal blink_count, eye_closed_state, drowsy_frames, normal_frames, drowsy_alert
            nonlocal face_frame_idx, last_emotion, last_conf
            nonlocal target_gesture, game_score, hold_start
            nonlocal face_ts, hand_ts

            h, w = frame.shape[:2]
            mp_img = bgr_to_mp_image(frame)

            # ---- run face detector once for all face features --------
            face_result = None
            lms         = None
            if need_face and face_det is not None:
                if is_image:
                    face_result = face_det.detect(mp_img)
                else:
                    face_ts    += ts_step
                    face_result = face_det.detect_for_video(mp_img, face_ts)
                if face_result and face_result.face_landmarks:
                    lms = face_result.face_landmarks[0]

            # ---- run hand detector -----------------------------------
            hand_result = None
            if need_hand and hand_det is not None:
                if is_image:
                    hand_result = hand_det.detect(mp_img)
                else:
                    hand_ts    += ts_step
                    hand_result = hand_det.detect_for_video(mp_img, hand_ts)

            # Running Y cursor for stacked text (resets every frame)
            y = 35

            # ===========================================================
            # LIPS block
            # ===========================================================
            if "lips" in active:
                draw_text(frame, "[ LIPS ]", (_TEXT_X, y),
                          color=_HEADER_COLOR["lips"], scale=0.55, thickness=1)
                y += _HEADER_H

                if lms:
                    # Draw lip mesh
                    mp_drawing.draw_landmarks(
                        frame, lms,
                        FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style(),
                    )

                    smiling = detect_smile(lms, w, h)
                    mar     = compute_mar(lms, w, h)

                    mood_color = (0, 255, 100) if smiling else (200, 200, 200)
                    draw_text(frame,
                              f"Mood: {'Smiling' if smiling else 'Neutral'}",
                              (_TEXT_X + 4, y), color=mood_color, scale=0.55, thickness=1)
                    y += _LINE_H

                    draw_text(frame, f"MAR: {mar:.2f}",
                              (_TEXT_X + 4, y), scale=0.55, thickness=1)
                    y += _LINE_H

                    # hysteresis open/close counter
                    if not mouth_open and mar > MAR_OPEN_THRESHOLD:
                        mouth_open = True
                    elif mouth_open and mar < MAR_CLOSE_THRESHOLD:
                        mouth_open = False
                        sync_count += 1

                    state_color = (0, 100, 255) if mouth_open else (170, 170, 170)
                    draw_text(frame,
                              f"Sync:{sync_count}  {'OPEN' if mouth_open else 'closed'}",
                              (_TEXT_X + 4, y), color=state_color, scale=0.55, thickness=1)
                    y += _LINE_H

                    if mar > MAR_OPEN_THRESHOLD:
                        draw_red_border(frame)
                else:
                    draw_text(frame, "No face", (_TEXT_X + 4, y),
                              color=(0, 0, 255), scale=0.55, thickness=1)
                    y += _LINE_H

                y += _GAP_H

            # ===========================================================
            # EYES block
            # ===========================================================
            if "eyes" in active:
                draw_text(frame, "[ EYES ]", (_TEXT_X, y),
                          color=_HEADER_COLOR["eyes"], scale=0.55, thickness=1)
                y += _HEADER_H

                if not calibrated:
                    # Collect calibration samples while other features still run
                    if lms:
                        ear_l = compute_ear(lms, FACE_LEFT_EYE,  w, h)
                        ear_r = compute_ear(lms, FACE_RIGHT_EYE, w, h)
                        avg   = (ear_l + ear_r) / 2.0
                        if avg > 0.05:
                            cal_samples.append(avg)

                    n   = len(cal_samples)
                    pct = min(n / CALIBRATION_FRAMES, 1.0)
                    draw_text(frame, f"Cal: {n}/{CALIBRATION_FRAMES}",
                              (_TEXT_X + 4, y), color=(0, 255, 150), scale=0.55, thickness=1)
                    y += _LINE_H

                    # Inline progress bar under the text
                    bar_w = 120
                    bx    = _TEXT_X + 4
                    by    = y
                    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 8), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bx, by),
                                  (bx + int(bar_w * pct), by + 8), (0, 200, 80), -1)
                    y += 14

                    if n >= CALIBRATION_FRAMES:
                        closed_thresh = _calibrate_threshold(cal_samples)
                        calibrated    = True

                elif lms:
                    # Draw eye contours
                    for conn in (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                                 FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE):
                        mp_drawing.draw_landmarks(
                            frame, lms, conn,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style(),
                        )

                    ear_l = compute_ear(lms, FACE_LEFT_EYE,  w, h)
                    ear_r = compute_ear(lms, FACE_RIGHT_EYE, w, h)
                    avg   = (ear_l + ear_r) / 2.0

                    # Blink detection
                    if avg < closed_thresh:
                        eye_closed_state = True
                    elif eye_closed_state:
                        blink_count     += 1
                        eye_closed_state = False

                    # Drowsiness tracking
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

                    draw_text(frame,
                              f"EAR L:{ear_l:.2f}  R:{ear_r:.2f}",
                              (_TEXT_X + 4, y), scale=0.55, thickness=1)
                    y += _LINE_H

                    alert_color = (0, 60, 255) if drowsy_alert else (0, 220, 255)
                    draw_text(frame,
                              f"Blinks:{blink_count}  {'DROWSY' if drowsy_alert else 'Awake'}",
                              (_TEXT_X + 4, y), color=alert_color, scale=0.55, thickness=1)
                    y += _LINE_H

                    if drowsy_alert:
                        draw_red_banner(frame, "DROWSY! Wake Up!")

                else:
                    draw_text(frame, "No face", (_TEXT_X + 4, y),
                              color=(0, 0, 255), scale=0.55, thickness=1)
                    y += _LINE_H

                y += _GAP_H

            # ===========================================================
            # FACE block
            # ===========================================================
            if "face" in active:
                draw_text(frame, "[ FACE ]", (_TEXT_X, y),
                          color=_HEADER_COLOR["face"], scale=0.55, thickness=1)
                y += _HEADER_H

                if lms:
                    bx_f, by_f, bw_f, bh_f = bbox_from_landmarks(lms, w, h)
                    cv2.rectangle(frame,
                                  (bx_f, by_f), (bx_f + bw_f, by_f + bh_f),
                                  (255, 200, 0), 2)

                    # Throttled DeepFace call
                    if df_ok and face_frame_idx % EMOTION_INFERENCE_EVERY == 0:
                        if bw_f > 0 and bh_f > 0:
                            roi = frame[by_f:by_f + bh_f, bx_f:bx_f + bw_f]
                            if roi.size:
                                try:
                                    result = DeepFace.analyze(
                                        roi, actions=["emotion"],
                                        enforce_detection=False, silent=True,
                                    )
                                    emo          = result[0]["emotion"]
                                    last_emotion = max(emo, key=emo.get)
                                    last_conf    = emo[last_emotion]
                                    emo_history.append((time.time(), last_emotion))
                                except Exception:
                                    pass

                    # Rolling recent-mood window
                    now = time.time()
                    while emo_history and now - emo_history[0][0] > EMOTION_HISTORY_SECONDS:
                        emo_history.popleft()
                    recent = (
                        collections.Counter(e for _, e in emo_history).most_common(1)[0][0]
                        if emo_history else "N/A"
                    )

                    draw_text(frame,
                              f"Emo: {last_emotion} ({last_conf:.0f}%)",
                              (_TEXT_X + 4, y), scale=0.55, thickness=1)
                    y += _LINE_H
                    draw_text(frame, f"Mood: {recent}",
                              (_TEXT_X + 4, y), color=(255, 200, 80), scale=0.55, thickness=1)
                    y += _LINE_H
                    draw_text(frame,
                              f"Pose: {classify_head_pose(lms, w, h)}",
                              (_TEXT_X + 4, y), color=(100, 255, 255), scale=0.55, thickness=1)
                    y += _LINE_H
                else:
                    draw_text(frame, "No face", (_TEXT_X + 4, y),
                              color=(0, 0, 255), scale=0.55, thickness=1)
                    y += _LINE_H

                face_frame_idx += 1
                y += _GAP_H

            # ===========================================================
            # HANDS block
            # ===========================================================
            if "hands" in active:
                draw_text(frame, "[ HANDS ]", (_TEXT_X, y),
                          color=_HEADER_COLOR["hands"], scale=0.55, thickness=1)
                y += _HEADER_H

                current_gestures = []

                if hand_result and hand_result.hand_landmarks:
                    for idx, hand_lms in enumerate(hand_result.hand_landmarks):
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

                        # Handedness
                        if hand_result.handedness and idx < len(hand_result.handedness):
                            handedness = hand_result.handedness[idx][0].display_name
                        else:
                            handedness = "Right"

                        fingers_up, thumb_up = count_fingers(hand_lms, handedness, w, h)
                        total         = fingers_up + (1 if thumb_up else 0)
                        gesture_name, tag = classify_gesture(fingers_up, thumb_up)
                        current_gestures.append(gesture_name)

                        # Annotate near the wrist (self-positioned, not in left column)
                        wrist = lm_to_px(hand_lms[HAND_WRIST], w, h)
                        wrist_x = max(wrist[0] - 60, 0)
                        draw_text(frame,
                                  f"{handedness}: {total} fingers",
                                  (wrist_x, wrist[1] + 30), color=(50, 255, 50))
                        draw_text(frame,
                                  f"{gesture_name} [{tag}]",
                                  (wrist_x, wrist[1] + 60), color=(255, 200, 50))

                        letter = detect_letter(fingers_up, thumb_up)
                        if letter:
                            draw_text(frame, f"Letter: {letter}",
                                      (wrist_x, wrist[1] + 90), color=(255, 160, 0))
                else:
                    draw_text(frame, "No hands", (_TEXT_X + 4, y),
                              color=(150, 150, 150), scale=0.55, thickness=1)
                    y += _LINE_H

                # Gesture game HUD — always at the bottom when enabled
                if ENABLE_GESTURE_GAME:
                    gy = h - 90
                    draw_text(frame, f"TARGET: {target_gesture}", (20, gy),
                              color=(0, 200, 255), scale=0.8)
                    draw_text(frame, f"Score: {game_score}", (20, gy + 35),
                              color=(0, 255, 180), scale=0.8)

                    if target_gesture in current_gestures:
                        if hold_start is None:
                            hold_start = time.time()
                        progress = min(
                            (time.time() - hold_start) / GAME_HOLD_SECONDS, 1.0
                        )
                        bx_g, by_g, bw_g, bh_g = 20, gy + 55, 200, 12
                        cv2.rectangle(frame,
                                      (bx_g, by_g), (bx_g + bw_g, by_g + bh_g),
                                      (60, 60, 60), -1)
                        cv2.rectangle(frame,
                                      (bx_g, by_g),
                                      (bx_g + int(bw_g * progress), by_g + bh_g),
                                      (0, 220, 120), -1)
                        if progress >= 1.0:
                            game_score    += 1
                            target_gesture = random.choice(game_labels)
                            hold_start     = None
                    else:
                        hold_start = None

            draw_esc_hint(frame)
            return frame

        # ----------------------------------------------------------------
        # Run the shared detection loop
        # ----------------------------------------------------------------
        detection_loop(source, process, window_title)
