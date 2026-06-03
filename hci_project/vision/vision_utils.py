"""
vision_utils.py
===============
Shared helpers for the vision app: geometry, drawing, source management,
and the master detection loop used by every detection module.

Centralising these removes duplicated code from each module and guarantees
consistent behaviour (ESC always returns to the menu, video files replay,
static images stay open until a key is pressed).

Adapted from the proven "Previous Code" reference implementation; tunables
live in vision_config.py where applicable.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def euclidean(p1, p2) -> float:
    """
    Euclidean (L2) distance between two 2-D points.

    Parameters
    ----------
    p1 : tuple[float, float]
        First point (x, y).
    p2 : tuple[float, float]
        Second point (x, y).

    Returns
    -------
    float
        Non-negative distance.
    """
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def lm_to_px(landmark, w: int, h: int) -> tuple:
    """
    Convert a normalised MediaPipe landmark (.x/.y in [0,1]) to pixel coords.

    Parameters
    ----------
    landmark : object
        Landmark exposing .x and .y floats.
    w : int
        Frame width in pixels.
    h : int
        Frame height in pixels.

    Returns
    -------
    tuple[int, int]
        (x_px, y_px).
    """
    return int(landmark.x * w), int(landmark.y * h)


def compute_aspect_ratio(top, bottom, left, right) -> float:
    """
    Generic vertical-to-horizontal aspect ratio (drives both EAR and MAR).

        ratio = ||top - bottom|| / (||left - right|| + epsilon)

    Parameters
    ----------
    top, bottom : tuple[float, float]
        Vertical extent points.
    left, right : tuple[float, float]
        Horizontal extent points.

    Returns
    -------
    float
        Aspect ratio >= 0. Larger means more open.
    """
    return euclidean(top, bottom) / (euclidean(left, right) + 1e-6)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text(frame, text: str, pos: tuple, color=(0, 255, 0),
              scale: float = 0.7, thickness: int = 2) -> None:
    """
    Render anti-aliased text with a black drop-shadow for readability.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, mutated in-place.
    text : str
        Text to draw.
    pos : tuple[int, int]
        Baseline position (x, y).
    color : tuple
        BGR foreground colour.
    scale : float
        Font scale.
    thickness : int
        Stroke thickness.

    Returns
    -------
    None
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_red_border(frame, thickness: int = 10) -> None:
    """
    Draw a solid red rectangle around the whole frame (threshold warning).

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, mutated in-place.
    thickness : int
        Border width in pixels.

    Returns
    -------
    None
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)


def draw_red_banner(frame, message: str = "ALERT!") -> None:
    """
    Overlay a semi-transparent red banner with centred white text.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, mutated in-place.
    message : str
        Text to display.

    Returns
    -------
    None
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cy = h // 2
    cv2.rectangle(overlay, (0, cy - 40), (w, cy + 40), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw = cv2.getTextSize(message, font, 1.2, 3)[0][0]
    cv2.putText(frame, message, ((w - tw) // 2, cy + 12), font, 1.2,
                (255, 255, 255), 3, cv2.LINE_AA)


def draw_esc_hint(frame) -> None:
    """
    Draw a small greyed-out 'ESC = Menu' hint in the bottom-right corner.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, mutated in-place.

    Returns
    -------
    None
    """
    h, w = frame.shape[:2]
    draw_text(frame, "ESC = Menu", (w - 145, h - 15),
              color=(200, 200, 200), scale=0.5, thickness=1)


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

def open_source(source):
    """
    Open a cv2.VideoCapture from a webcam index or a file path.

    Parameters
    ----------
    source : int | str | cv2.VideoCapture
        Webcam index, video file path, or an already-open capture.

    Returns
    -------
    cv2.VideoCapture

    Raises
    ------
    RuntimeError
        If the source cannot be opened.
    """
    if isinstance(source, cv2.VideoCapture):
        return source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def read_frame(cap_or_image):
    """
    Read one BGR frame from a capture, or return a copy of a static image.

    Parameters
    ----------
    cap_or_image : cv2.VideoCapture | numpy.ndarray
        Capture object or static image.

    Returns
    -------
    tuple[bool, numpy.ndarray | None]
        (ok, frame).
    """
    if isinstance(cap_or_image, np.ndarray):
        return True, cap_or_image.copy()
    return cap_or_image.read()


# ---------------------------------------------------------------------------
# Master detection loop
# ---------------------------------------------------------------------------

def detection_loop(source, process_fn, window_title: str = "Detection") -> None:
    """
    Master frame-processing loop shared by all detection modules.

    Each module supplies a `process_fn(frame) -> annotated_frame`; this loop
    handles grabbing frames, displaying them, replaying finished videos, and
    keyboard control (ESC = back to menu, 'q' = quit the whole app).

    Parameters
    ----------
    source : int | str | numpy.ndarray | cv2.VideoCapture
        Input source. Static images (ndarray) are shown until a key is pressed.
    process_fn : callable
        Signature: annotated = process_fn(bgr_frame). Returns a BGR ndarray.
    window_title : str
        OpenCV window title.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        When the user presses 'q'.
    """
    is_image = isinstance(source, np.ndarray)
    cap = None if is_image else open_source(source)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = read_frame(source if is_image else cap)
        if not ok or frame is None:
            if is_image:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # replay video from start
            continue

        result = process_fn(frame)
        cv2.imshow(window_title, result)

        key = cv2.waitKey(0 if is_image else 30) & 0xFF
        if key == 27:            # ESC -> back to menu
            break
        if key == ord('q'):      # q -> quit the whole application
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        if is_image:
            break

    if cap:
        cap.release()
    cv2.destroyWindow(window_title)