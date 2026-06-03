"""
vision_main.py
==============
Vision app entry point: console menu, input-source selection, and dispatch
to the four detection modules. Feature flags are checked against vision_config.py.

Navigation
----------
  Main menu   : 1-4 pick a module, 0 to quit.
  Source menu : 1 = webcam, 2 = video file, 3 = image, 0 = back.
  In window   : ESC returns to the menu; 'q' quits the whole app.
"""

from pathlib import Path

import cv2

from vision.vision_config import (
    ENABLE_LIPS_DETECTION,
    ENABLE_EYES_DETECTION,
    ENABLE_FACE_DETECTION,
    ENABLE_HAND_DETECTION,
    ENABLE_GESTURE_GAME,
    WEBCAM_INDEX,
)
from vision.mp_tasks import ensure_models
from vision.lips_module import run_lips
from vision.eyes_module import run_eyes
from vision.face_module import run_face
from vision.hand_module import run_hands


# ---------------------------------------------------------------------------
# Input source selection
# ---------------------------------------------------------------------------

def _choose_source():
    """
    Prompt the user to choose an input source.

    Returns the type expected by the detection modules:
      webcam -> int (device index), video -> str path, image -> numpy.ndarray.

    Returns
    -------
    int | str | numpy.ndarray | None
        Source, or None to go back / on error.
    """
    print("\nSelect Input Source:")
    print("  1. Live Webcam")
    print("  2. Video File")
    print("  3. Image File")
    print("  0. Back")
    choice = input("Enter 0-3: ").strip()

    if choice == "1":
        return WEBCAM_INDEX
    if choice == "2":
        p = input("Enter video file path: ").strip().strip('"')
        if not Path(p).is_file():
            print(f"[ERROR] File not found: {p}")
            return None
        return p
    if choice == "3":
        p = input("Enter image file path: ").strip().strip('"')
        img = cv2.imread(p)
        if img is None:
            print(f"[ERROR] Cannot read image: {p}")
            return None
        return img
    return None


# ---------------------------------------------------------------------------
# Module dispatch (each entry checks its ENABLE_* flag)
# ---------------------------------------------------------------------------

def _dispatch_lips(source):
    """Run lips detection if enabled, else print a notice."""
    if not ENABLE_LIPS_DETECTION:
        print("[INFO] Lips detection is disabled in vision_config.py.")
        return
    run_lips(source)


def _dispatch_eyes(source):
    """Run eyes detection if enabled, else print a notice."""
    if not ENABLE_EYES_DETECTION:
        print("[INFO] Eyes detection is disabled in vision_config.py.")
        return
    run_eyes(source)


def _dispatch_face(source):
    """Run face detection if enabled, else print a notice."""
    if not ENABLE_FACE_DETECTION:
        print("[INFO] Face detection is disabled in vision_config.py.")
        return
    run_face(source)


def _dispatch_hands(source):
    """Run hand detection if enabled, optionally enabling the gesture game."""
    if not ENABLE_HAND_DETECTION:
        print("[INFO] Hand detection is disabled in vision_config.py.")
        return
    game = False
    if ENABLE_GESTURE_GAME:
        game = input("Enable Gesture Game? (y/n): ").strip().lower() == "y"
    run_hands(source, game_mode=game)


#: Module registry: menu key -> (display name, dispatch function)
MODULES = {
    "1": ("Lips Detection", _dispatch_lips),
    "2": ("Eyes Detection", _dispatch_eyes),
    "3": ("Face Detection", _dispatch_face),
    "4": ("Hand Detection", _dispatch_hands),
}


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main menu loop for the vision app.

    Ensures models are present, then repeatedly shows the module menu,
    asks for an input source, and dispatches to the chosen module.

    Returns
    -------
    None
    """
    print("\n" + "=" * 45)
    print("        HCI Vision System (NEXUS)")
    print("=" * 45)

    print("\n  Checking model files ...")
    try:
        ensure_models()
        print("  Models ready.")
    except RuntimeError as exc:
        print(f"  [FATAL] {exc}")
        return

    while True:
        print("\nSelect Detection Mode:")
        for key, (name, _) in MODULES.items():
            print(f"  {key}. {name}")
        print("  0. Exit")
        mode = input("Enter 0-4: ").strip()

        if mode == "0":
            print("Goodbye.")
            break
        if mode not in MODULES:
            print("[WARN] Invalid selection.")
            continue

        name, dispatch = MODULES[mode]
        source = _choose_source()
        if source is None:
            continue

        try:
            dispatch(source)
        except SystemExit:
            print("\n[INFO] Quit requested. Exiting.")
            break
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted — returning to menu.")

        print("\n[INFO] Back at main menu. (ESC inside a window returns here; 'q' quits.)")