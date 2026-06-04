"""
vision_main.py
==============
Vision app entry point: console menu, input-source selection, and dispatch
to the four detection modules.

Dynamic menu
------------
The menu is built at runtime from the ENABLE_* flags in vision_config.py.
Setting any flag to False removes that option from the menu completely and
re-numbers the remaining options 1..n automatically.  No other file changes.

Navigation
----------
  Main menu   : 1-n to pick a module, 0 to quit.
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
    ENABLE_HYBRID_MODE,
    WEBCAM_INDEX,
)
from vision.mp_tasks import ensure_models
from vision.lips_module   import run_lips
from vision.eyes_module   import run_eyes
from vision.face_module   import run_face
from vision.hand_module   import run_hands
from vision.hybrid_module import run_hybrid


# ---------------------------------------------------------------------------
# Module definitions — (enabled_flag, display_name, dispatch_function)
# To add a new module: append one tuple here and add its flag to vision_config.py
# ---------------------------------------------------------------------------

def _dispatch_lips(source):
    """Run lips detection."""
    run_lips(source)


def _dispatch_eyes(source):
    """Run eyes detection."""
    run_eyes(source)


def _dispatch_face(source):
    """Run face detection."""
    run_face(source)


def _dispatch_hands(source):
    """Run hand detection. Game is shown when ENABLE_GESTURE_GAME=True in config."""
    run_hands(source)


def _dispatch_hybrid(source):
    """Run hybrid mode with all HYBRID_ACTIVE_MODULES on one panel."""
    run_hybrid(source)


def _build_menu() -> dict:
    """
    Build the active module menu by reading flags from vision_config at call time.

    Reading from the config module object (not from imported names) means the
    menu reflects whatever the flags are at the moment this function runs.
    Only modules whose ENABLE_* flag is True are included; they are numbered
    sequentially from 1 so the menu is always compact and gap-free.

    Returns
    -------
    dict[str, tuple[str, callable]]
        {str_key: (display_name, dispatch_fn)} for every enabled module.
        Empty dict if all modules are disabled.
    """
    import vision.vision_config as cfg

    # Ordered list: (config_flag, display_name, dispatch_function)
    # To add a new module: append a tuple here + add its flag to vision_config.py
    all_modules = [
        (cfg.ENABLE_LIPS_DETECTION, "Lips Detection",  _dispatch_lips),
        (cfg.ENABLE_EYES_DETECTION, "Eyes Detection",  _dispatch_eyes),
        (cfg.ENABLE_FACE_DETECTION, "Face Detection",  _dispatch_face),
        (cfg.ENABLE_HAND_DETECTION, "Hand Detection",  _dispatch_hands),
        (cfg.ENABLE_HYBRID_MODE,    "Hybrid Mode",     _dispatch_hybrid),
    ]

    menu = {}
    key  = 1
    for enabled, name, fn in all_modules:
        if enabled:
            menu[str(key)] = (name, fn)
            key += 1
    return menu


# ---------------------------------------------------------------------------
# Input source selection
# ---------------------------------------------------------------------------

def _choose_source():
    """
    Prompt the user to choose an input source.

    Returns the type expected by detection modules:
      webcam -> int, video -> str path, image -> numpy.ndarray.

    Returns
    -------
    int | str | numpy.ndarray | None
        Source, or None if the user chose Back or an error occurred.
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
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main menu loop for the vision app.

    Builds the active module list from ENABLE_* flags, ensures models are
    present, then repeatedly shows the menu and dispatches to the chosen
    module.  Disabled modules are hidden and the numbering is compact.

    Returns
    -------
    None
    """
    print("\n" + "=" * 45)
    print("        HCI Vision System (NEXUS)")
    print("=" * 45)

    # Build menu once — flags are read at import time so this is stable
    modules = _build_menu()

    if not modules:
        print("[ERROR] All detection modules are disabled in vision_config.py.")
        print("        Set at least one ENABLE_* flag to True and re-run.")
        return

    print("\n  Checking model files ...")
    try:
        ensure_models()
        print("  Models ready.")
    except RuntimeError as exc:
        print(f"  [FATAL] {exc}")
        return

    max_key = max(int(k) for k in modules)

    while True:
        print("\nSelect Detection Mode:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  0. Exit")
        mode = input(f"Enter 0-{max_key}: ").strip()

        if mode == "0":
            print("Goodbye.")
            break

        if mode not in modules:
            print("[WARN] Invalid selection.")
            continue

        name, dispatch = modules[mode]
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
