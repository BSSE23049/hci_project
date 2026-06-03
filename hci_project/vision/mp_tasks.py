"""
mp_tasks.py
===========
MediaPipe Tasks API infrastructure (the current, non-deprecated API):
  - canonical paths for the two .task model bundles
  - automatic first-run download from Google's public CDN
  - BGR -> mediapipe.Image conversion
  - factory helpers that build a FaceLandmarker / HandLandmarker using the
    confidence and count settings from vision_config.py

Keeping model management + conversion here means every detection module can
build a landmarker without re-declaring options or knowing download details.
Reference: https://ai.google.dev/edge/mediapipe/solutions/guide
"""

import urllib.request
from pathlib import Path

import cv2

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat

from vision.vision_config import (
    MAX_HANDS,
    FACE_DETECTION_CONFIDENCE,
    FACE_TRACKING_CONFIDENCE,
    HAND_DETECTION_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
)

# ---------------------------------------------------------------------------
# Model paths + URLs
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
}

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model(filename: str) -> Path:
    """
    Return the local path to a model bundle, downloading it on first use.

    Parameters
    ----------
    filename : str
        Key in MODEL_URLS, e.g. "face_landmarker.task".

    Returns
    -------
    pathlib.Path
        Local path to the .task file.

    Raises
    ------
    RuntimeError
        If the file is missing and cannot be downloaded.
    """
    path = MODELS_DIR / filename
    if path.exists():
        return path

    url = MODEL_URLS[filename]
    print(f"  [mp_tasks] Downloading {filename} (first run only) ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  [mp_tasks] Saved to {path}")
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Could not download {filename} from {url}\n"
            f"Place the file manually at: {path}\nError: {exc}"
        ) from exc
    return path


def ensure_models() -> None:
    """
    Ensure both required .task model bundles are present (download if missing).

    Returns
    -------
    None
    """
    for name in MODEL_URLS:
        ensure_model(name)


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def bgr_to_mp_image(frame) -> MpImage:
    """
    Convert an OpenCV BGR frame to a mediapipe.Image (SRGB) for the Tasks API.

    Parameters
    ----------
    frame : numpy.ndarray
        uint8 BGR image (H, W, 3).

    Returns
    -------
    mediapipe.Image
        SRGB image ready for detect()/detect_for_video().
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return MpImage(image_format=ImageFormat.SRGB, data=rgb)


# ---------------------------------------------------------------------------
# Landmarker factories
# ---------------------------------------------------------------------------

def make_face_landmarker(running_mode: RunningMode, num_faces: int = 1):
    """
    Build a FaceLandmarker using confidences from vision_config.py.

    Parameters
    ----------
    running_mode : RunningMode
        RunningMode.IMAGE for static images, RunningMode.VIDEO for streams.
    num_faces : int
        Maximum number of faces to detect.

    Returns
    -------
    FaceLandmarker
        A landmarker created from options (use as a context manager).
    """
    ensure_model("face_landmarker.task")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=running_mode,
        num_faces=num_faces,
        min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
        min_face_presence_confidence=FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
    )
    return FaceLandmarker.create_from_options(options)


def make_hand_landmarker(running_mode: RunningMode, num_hands: int = MAX_HANDS):
    """
    Build a HandLandmarker using confidences from vision_config.py.

    Parameters
    ----------
    running_mode : RunningMode
        RunningMode.IMAGE for static images, RunningMode.VIDEO for streams.
    num_hands : int
        Maximum number of hands to detect.

    Returns
    -------
    HandLandmarker
        A landmarker created from options (use as a context manager).
    """
    ensure_model("hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)