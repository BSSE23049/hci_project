"""
main.py
=======
Application entry point for the Multi-Modal Human Detection System.

Responsibilities
----------------
1. Ensure both MediaPipe .task model bundles are present (downloading on first run).
2. Present the module-selection main menu.
3. For each selection, present the input-source sub-menu and dispatch to the
   appropriate run_* function.

Navigation
----------
  Main menu  : 1–4 to pick a module, 'q' to quit.
  Source menu: 1 = webcam, 2 = video file, 3 = static image, 0 = back.
  In window  : ESC returns to the main menu; 'q' quits the application.

Course  : Human Computer Interaction (SE305T / MD445T) — Spring-26
Institute: Information Technology University (ITU), Lahore
Assignment: 4  |  CLO-3
Groups  : BSSE23 & BSCE22
Instructor: Dr. Muhammad Asif
"""

from pathlib import Path

import cv2

from models       import ensure_models
from module_lips  import run_lips_detection
from module_eyes  import run_eyes_detection
from module_face  import run_face_detection
from module_hand  import run_hand_detection


# ==============================================================================
#  INPUT SOURCE SELECTION
# ==============================================================================

def choose_source():
    """
    Interactive console sub-menu for selecting the input source.

    Presents three options:
        1  -> Live Webcam Feed   : opens device index 0.
        2  -> Video File Upload  : prompts for a file-system path.
        3  -> Image Upload       : prompts for a file-system path; loads the
                                   image into memory as a BGR ndarray so the
                                   detection loop treats it as a static frame.
        0  -> Back               : returns None, causing the caller to skip
                                   module dispatch and loop back to the main menu.

    Why return different types?
    ---------------------------
    The detection_loop() function in utils.py checks isinstance() to decide
    whether the source is a static image (ndarray), video file (str), or
    live camera (int).  Returning the correct Python type here means no
    extra flag variables are needed anywhere downstream.

    Parameters
    ----------
    None

    Returns
    -------
    int | str | np.ndarray | None
        * int          -- 0 (webcam device index).
        * str          -- absolute or relative path to a video file.
        * np.ndarray   -- BGR image array (H×W×3 uint8).
        * None         -- user chose 'Back' or an error occurred.
    """
    print("\n+----------------------------------+")
    print("|      SELECT INPUT SOURCE         |")
    print("+----------------------------------+")
    print("|  1. Live Webcam Feed             |")
    print("|  2. Video File Upload            |")
    print("|  3. Image Upload                 |")
    print("|  0. Back                         |")
    print("+----------------------------------+")
    choice = input("  Enter choice: ").strip()

    if choice == "1":
        return 0   # webcam index

    elif choice == "2":
        path = input("  Enter video file path: ").strip().strip('"')
        if not Path(path).is_file():
            print(f"  [ERROR] File not found: {path}")
            return None
        return path

    elif choice == "3":
        path = input("  Enter image file path: ").strip().strip('"')
        img = cv2.imread(path)
        if img is None:
            print(f"  [ERROR] Cannot read image: {path}")
            return None
        return img

    return None   # '0' or unrecognised input -> back to main menu


# ==============================================================================
#  MAIN MENU
# ==============================================================================

#: Module registry: key -> (display_name, run_function)
MODULES = {
    "1": ("Lips Detection",  run_lips_detection),
    "2": ("Eyes Detection",  run_eyes_detection),
    "3": ("Face Detection",  run_face_detection),
    "4": ("Hand Detection",  run_hand_detection),
}


def main():
    """
    Top-level application entry point.

    Flow
    ----
    1. Print the application banner.
    2. Call ensure_models() to download .task files if not cached.
    3. Loop:
         a. Print the module menu.
         b. Read user input.
         c. If 'q', exit cleanly.
         d. If valid module number, call choose_source().
         e. If source is not None, dispatch to the module's run_* function.
         f. Catch RuntimeError (source open failure) and KeyboardInterrupt
            gracefully so the menu loop continues.

    Why a separate menu loop instead of argparse?
    ---------------------------------------------
    The assignment targets a lab environment where students run the script
    interactively.  A looping console menu lets users try multiple modules
    and input types in a single session without restarting the process.
    It also makes it easy to add new modules by extending the MODULES dict.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("\n+==========================================+")
    print("|   Multi-Modal Human Detection System    |")
    print("|   HCI Assignment 4 -- ITU Lahore        |")
    print("+==========================================+")

    print("\n  Checking model files ...")
    try:
        ensure_models()
    except RuntimeError as exc:
        print(f"\n  [FATAL] {exc}")
        return

    while True:
        print("\n+----------------------------------+")
        print("|       DETECTION MODE MENU        |")
        print("+----------------------------------+")
        for key, (name, _) in MODULES.items():
            print(f"|  {key}. {name:<30}|")
        print("|  q. Quit                         |")
        print("+----------------------------------+")

        choice = input("  Select module: ").strip().lower()

        if choice == "q":
            print("  Goodbye!")
            break

        if choice not in MODULES:
            print("  Invalid choice -- please enter 1, 2, 3, 4, or q.")
            continue

        module_name, module_fn = MODULES[choice]
        print(f"\n  -- {module_name} selected --")

        source = choose_source()
        if source is None:
            continue   # user pressed 0 (back) or an error occurred

        try:
            module_fn(source)
        except RuntimeError as exc:
            print(f"  [ERROR] {exc}")
        except KeyboardInterrupt:
            print("\n  Interrupted -- returning to menu.")


# ==============================================================================
if __name__ == "__main__":
    main()
