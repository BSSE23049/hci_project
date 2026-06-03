"""
models.py
=========
Model download management and MediaPipe image conversion helpers.

Responsibilities
----------------
1. Define canonical local paths for both .task model bundles.
2. Download models from Google's public CDN on first run only.
3. Convert OpenCV BGR frames to the mediapipe.Image format required by
   the new Tasks API (0.10.x+).

Why separate?
-------------
Model management and format conversion are infrastructure concerns that
every module needs but none should own.  Keeping them here means any
module can call ensure_models() and bgr_to_mp_image() without importing
the full detection logic of another module.

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
"""

import urllib.request
from pathlib import Path
import os

import cv2

from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat


# ==============================================================================
#  MODEL PATHS & URLS
# ==============================================================================

#: Local cache directory — created automatically if it does not exist.
_current_dir = Path(__file__).parent
_MODEL_DIR = _current_dir / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

#: Public CDN URLs for the two required model bundles.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

#: Full local file paths (exported so other modules can reference them).
FACE_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
HAND_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"


# ==============================================================================
#  DOWNLOAD HELPERS
# ==============================================================================

def _download_model(url, dest):
    """
    Download a single MediaPipe .task model bundle if it is not cached locally.

    The function is intentionally a no-op when the file already exists so that
    subsequent application starts are instant (no network round-trip).

    Why .task files?
    ----------------
    MediaPipe Tasks API packages the model weights, pre/post-processing ops,
    and metadata into a single self-contained .task bundle (a TFLite flatbuffer
    with extra metadata).  This replaces the older approach of shipping raw
    .tflite weights alongside Python glue code.

    Parameters
    ----------
    url  : str   -- publicly accessible HTTPS URL of the model bundle.
    dest : Path  -- absolute local path where the bundle should be saved.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError  -- if the download fails (network error, HTTP error, etc.).
                     The error message includes a manual-placement hint so users
                     can resolve the issue without re-running the downloader.
    """
    if dest.exists():
        return   # already cached — skip download
    print(f"  [INFO] Downloading {dest.name}  (first-time only) ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [INFO] Saved to {dest}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model from {url}\n"
            f"Place the file manually at: {dest}\n"
            f"Error: {exc}"
        )


def ensure_models():
    """
    Ensure both required model .task files are present, downloading if needed.

    Called once at application startup (in main.py) before any module is
    selected.  Downloading here — rather than lazily inside each run_* function
    — gives the user a single, clear progress message instead of unexpected
    pauses mid-session.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    RuntimeError  -- propagated from _download_model() if either file cannot
                     be fetched.
    """
    _download_model(_FACE_MODEL_URL, FACE_MODEL_PATH)
    _download_model(_HAND_MODEL_URL, HAND_MODEL_PATH)


# ==============================================================================
#  IMAGE FORMAT CONVERSION
# ==============================================================================

def bgr_to_mp_image(frame):
    """
    Convert an OpenCV BGR numpy array to a mediapipe.Image object (SRGB).

    Why this conversion is necessary
    ---------------------------------
    The new MediaPipe Tasks API (0.10.x+) requires inputs as
    mediapipe.tasks.python.vision.core.image.Image rather than raw numpy
    arrays.  Under the hood MediaPipe expects RGB byte order, but OpenCV
    stores pixels as BGR.  Skipping the colour-channel swap produces correct
    detections but wrong colours on any drawn overlays.

    The conversion pipeline is:
        OpenCV BGR ndarray
            |  cv2.cvtColor(BGR -> RGB)
            v
        RGB ndarray
            |  MpImage(image_format=SRGB, data=rgb)
            v
        mediapipe.Image  (ready for .detect() / .detect_for_video())

    Parameters
    ----------
    frame : np.ndarray  -- uint8 BGR image from cv2.VideoCapture.read() or
                           cv2.imread().  Shape (H, W, 3).

    Returns
    -------
    MpImage  -- mediapipe.Image wrapping the RGB pixel data; ready to be
                passed directly to FaceLandmarker.detect() or
                HandLandmarker.detect_for_video().
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return MpImage(image_format=ImageFormat.SRGB, data=rgb)
