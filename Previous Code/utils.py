"""
utils.py
========
Shared utility functions for the Multi-Modal Human Detection System.

Contains:
- Euclidean distance computation
- Landmark-to-pixel conversion
- Generic aspect ratio computation
- OpenCV drawing helpers (text, red border, red banner)
- Video/image source management (open_source, read_frame)
- Master detection loop
- ESC hint overlay

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
"""

import cv2
import numpy as np


# ==============================================================================
#  GEOMETRY HELPERS
# ==============================================================================

def euclidean(p1, p2):
    """
    Compute the Euclidean (straight-line) distance between two 2-D points.

    The function subtracts the coordinate vectors and takes the L2 norm,
    which is the standard formula: sqrt((x2-x1)^2 + (y2-y1)^2).
    It is used throughout the codebase whenever a physical distance between
    two landmarks is needed (e.g. eye width, mouth height).

    Parameters
    ----------
    p1 : tuple (x, y)  -- first point in pixel or normalised coordinates.
    p2 : tuple (x, y)  -- second point in pixel or normalised coordinates.

    Returns
    -------
    float  -- non-negative distance value.
    """
    return float(np.linalg.norm(
        np.array(p1, dtype=float) - np.array(p2, dtype=float)
    ) )


def lm_to_px(landmark, w, h):
    """
    Convert a normalised MediaPipe landmark to integer pixel coordinates.

    MediaPipe reports every landmark in the range [0.0, 1.0] relative to the
    frame dimensions.  To draw or measure anything in pixel space we must
    multiply by the actual width and height.

    This function is compatible with both the legacy proto NormalizedLandmark
    (mp.solutions.*) and the new Tasks API NormalizedLandmark
    (mediapipe.tasks.python.components.containers.landmark) because both
    expose .x and .y float attributes.

    Parameters
    ----------
    landmark : NormalizedLandmark  -- has .x (float) and .y (float) in [0,1].
    w        : int                 -- frame width  in pixels.
    h        : int                 -- frame height in pixels.

    Returns
    -------
    tuple (int, int)  -- (x_px, y_px) clipped to frame dimensions.
    """
    return int(landmark.x * w), int(landmark.y * h)


def compute_aspect_ratio(top, bottom, left, right):
    """
    Compute a generic vertical-to-horizontal Aspect Ratio.

    The formula is:
        ratio = ||top - bottom|| / (||left - right|| + epsilon)

    A small epsilon (1e-6) prevents division by zero when the horizontal span
    collapses (e.g. a face seen at extreme profile).

    This single formula drives BOTH:
      * EAR (Eye Aspect Ratio)  -- Soukupova & Cech, 2016
      * MAR (Mouth Aspect Ratio) -- analogous formulation for lip openness

    A larger ratio means the feature is MORE open (eye wide open / mouth agape).

    Parameters
    ----------
    top    : (x, y)  -- upper landmark (e.g. upper eyelid, upper lip centre).
    bottom : (x, y)  -- lower landmark (e.g. lower eyelid, lower lip centre).
    left   : (x, y)  -- leftmost landmark (e.g. eye inner corner, mouth corner).
    right  : (x, y)  -- rightmost landmark (e.g. eye outer corner, mouth corner).

    Returns
    -------
    float  -- aspect ratio >= 0.
    """
    return euclidean(top, bottom) / (euclidean(left, right) + 1e-6)


# ==============================================================================
#  DRAWING HELPERS
# ==============================================================================

def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.7, thickness=2):
    """
    Render anti-aliased text with a black drop-shadow on a BGR frame.

    The drop-shadow is drawn one pixel down and one pixel right of the main
    text, making the label readable against any background colour (bright
    skin, dark clothing, complex textures).

    Parameters
    ----------
    frame     : np.ndarray  -- BGR image; mutated in-place.
    text      : str         -- the string to display.
    pos       : (x, y)      -- pixel position of the text baseline (bottom-left
                               of the first character).
    color     : (B, G, R)   -- foreground colour tuple. Default green.
    scale     : float       -- cv2 font scale multiplier. Default 0.7.
    thickness : int         -- stroke width in pixels. Default 2.

    Returns
    -------
    None
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Drop-shadow (offset +1 in both axes, black, one pixel thicker)
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Foreground text
    cv2.putText(frame, text, pos,
                font, scale, color, thickness, cv2.LINE_AA)


def draw_red_border(frame, thickness=10):
    """
    Draw a solid red rectangle along the entire frame border.

    Used as a visual warning when a threshold is exceeded (e.g. mouth open /
    MAR above the yawn threshold in Module 1).

    Parameters
    ----------
    frame     : np.ndarray  -- BGR image; mutated in-place.
    thickness : int         -- border width in pixels. Default 10.

    Returns
    -------
    None
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)


def draw_red_banner(frame, message="ALERT!"):
    """
    Overlay a semi-transparent red banner with centred white bold text.

    The banner spans the full width of the frame and is positioned at the
    vertical centre.  A weighted blend (0.6 overlay + 0.4 original) keeps
    the background partially visible so the user can still see landmarks.

    Parameters
    ----------
    frame   : np.ndarray  -- BGR image; mutated in-place.
    message : str         -- text to display inside the banner.

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
    cv2.putText(frame, message, ((w - tw) // 2, cy + 12),
                font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def _esc_hint(frame):
    """
    Draw a small greyed-out 'ESC = Menu' hint in the bottom-right corner.

    This passive reminder tells users how to return to the main menu without
    interrupting the live feed.

    Parameters
    ----------
    frame : np.ndarray  -- BGR image; mutated in-place.

    Returns
    -------
    None
    """
    h, w = frame.shape[:2]
    draw_text(frame, "ESC = Menu", (w - 145, h - 15),
              color=(200, 200, 200), scale=0.5, thickness=1)


# ==============================================================================
#  SOURCE MANAGEMENT
# ==============================================================================

def open_source(source):
    """
    Open a cv2.VideoCapture from an integer webcam index or a file path string.

    If an already-open VideoCapture is passed in it is returned unchanged,
    making the function idempotent and safe to call multiple times.

    Parameters
    ----------
    source : int | str | cv2.VideoCapture
        * int            -- webcam device index (0 = default camera).
        * str            -- path to a video file (.mp4, .avi, …).
        * VideoCapture   -- already-open capture object; returned as-is.

    Returns
    -------
    cv2.VideoCapture  -- open capture object.

    Raises
    ------
    RuntimeError  -- if the source string/index cannot be opened.
    """
    if isinstance(source, cv2.VideoCapture):
        return source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def read_frame(cap_or_image):
    """
    Read one BGR frame from a VideoCapture, or return a copy of a static image.

    Wrapping both cases in a single function lets the detection loop handle
    webcam/video and static images with identical logic.

    Parameters
    ----------
    cap_or_image : cv2.VideoCapture | np.ndarray
        * VideoCapture  -- .read() is called; returns (ok, frame).
        * np.ndarray    -- treated as a static image; always returns (True, copy).

    Returns
    -------
    tuple (bool, np.ndarray | None)
        * bool         -- True if a valid frame was obtained.
        * np.ndarray   -- BGR image, or None on failure.
    """
    if isinstance(cap_or_image, np.ndarray):
        return True, cap_or_image.copy()
    return cap_or_image.read()


# ==============================================================================
#  DETECTION LOOP
# ==============================================================================

def detection_loop(source, process_fn, window_title="Detection"):
    """
    Master frame-processing loop shared by all four detection modules.

    Design rationale
    ----------------
    All four modules (Lips, Eyes, Face, Hand) follow the same pattern:
      1. Grab a frame.
      2. Run inference + annotation (process_fn).
      3. Display the result.
      4. Poll for keyboard input.

    Centralising this loop removes ~50 lines of duplicated code from each
    module and ensures consistent behaviour (e.g. ESC always goes to menu,
    'q' always quits, video replays at end-of-file, static images stay open).

    Behaviour by source type
    ------------------------
    * Webcam / video file : runs at ~30 fps (cv2.waitKey(30)).
    * Static image        : waits indefinitely (cv2.waitKey(0)) then exits
                            after one frame or when a key is pressed.
    * End of video file   : seeks back to frame 0 and replays automatically.

    Key bindings
    ------------
    * ESC (27) -- break out of the loop; returns to the calling module's
                  run_* function, which returns to the main menu.
    * 'q'      -- release the capture, destroy all windows, and raise
                  SystemExit to terminate the entire application.

    Parameters
    ----------
    source       : int | str | np.ndarray | cv2.VideoCapture
                   Input source; passed to open_source() if not already open.
    process_fn   : callable
                   Signature: annotated_frame = process_fn(bgr_frame).
                   Must return the (possibly annotated) BGR ndarray.
    window_title : str
                   OpenCV window title shown in the OS title bar.

    Returns
    -------
    None

    Raises
    ------
    SystemExit  -- when the user presses 'q'.
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
        if key == 27:           # ESC -> back to menu
            break
        if key == ord('q'):     # q   -> quit entire application
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        if is_image:
            break

    if cap:
        cap.release()
    cv2.destroyWindow(window_title)
