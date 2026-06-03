"""
main_modular.py
================
Entry point for the MODULAR build of NEXUS — the Student Wellbeing
Advisor (HCI Text + Voice Assessment, SE305T, Spring-26).

Handled here
------------
  1. Print the startup banner.
  2. Initialise optional output modules (TTS).
     (Whisper / sounddevice load lazily inside voice_handler.)
  3. Start an ESC-key monitor thread on Windows.
  4. Run the menu-driven main loop:
        [1] Text Input Mode
        [2] Voice Input Mode (Whisper-based)
        [3] Hybrid Mode
        [4] Offline session replay (Stage 4 STUDENT_LOG)
        [0] Exit
"""

import time
import threading
import platform

from chatbot_modular.voice_handler import initialize_tts, initialize_voice_components
from chatbot_modular.input_modes   import run_text_mode, run_voice_mode, run_hybrid_mode
from chatbot_modular.demo          import run_demo_tests


MSVCRT_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import msvcrt
        MSVCRT_AVAILABLE = True
    except ImportError:
        pass


def start_esc_monitor(exit_event: threading.Event) -> threading.Thread:
    """Daemon thread that watches the ESC key on Windows."""
    def _monitor(event: threading.Event) -> None:
        if not MSVCRT_AVAILABLE:
            return
        while not event.is_set():
            try:
                if msvcrt.kbhit():
                    if msvcrt.getch() == b"\x1b":
                        print("\n\n  [ESC] Exit key pressed — shutting down...")
                        event.set()
                        break
            except Exception:
                break
            time.sleep(0.05)

    thread = threading.Thread(target=_monitor, args=(exit_event,), daemon=True)
    thread.start()
    return thread


def print_banner() -> None:
    line = "=" * 60
    print(f"\n{line}")
    print("   INFORMATION TECHNOLOGY UNIVERSITY (ITU) — LAHORE")
    print("   NEXUS — AI-Powered Student Wellbeing Advisor")
    print("   HCI Text + Voice Assessment | SE305T | Spring-26")
    print("   Instructor : Dr. Muhammad Asif")
    print(line)
    print("   Wellbeing scale  : THRIVING / CONTENT / NEUTRAL /")
    print("                      STRESSED / DISTRESSED / CRISIS")
    print("   Support classes  : ACADEMIC / WELLBEING / FINANCIAL /")
    print("                      TECHNICAL / SOCIAL / ADMIN")
    print(line)


def print_menu() -> None:
    print("\n  SELECT INPUT MODE:")
    print("  [1] Text Input Mode")
    print("  [2] Voice Input Mode (Whisper)")
    print("  [3] Hybrid Mode  (Text + Voice)")
    print("  [4] Offline Session Replay (Stage 4 STUDENT_LOG)")
    print("  [0] Exit NEXUS")
    print("-" * 60)


def main() -> None:
    print_banner()
    print("\n  [Init] Loading modules...")

    tts_engine = initialize_tts()
    print(f"  [Init] Text-to-Speech : {'Ready' if tts_engine else 'Unavailable'}")

    recognizer, microphone = initialize_voice_components()
    print(f"  [Init] Legacy STT      : {'Ready' if recognizer else 'Unavailable'}")

    print("  [Init] Wellbeing engine: Ready (6-tier scale)")
    print("  [Init] Support engine  : Ready (6 categories, multi-label)")
    print("  [Init] Report engine   : Ready")
    print("\n  NEXUS online. Press ESC or type '0' / 'exit' to quit.\n")

    exit_event = threading.Event()
    start_esc_monitor(exit_event)

    while not exit_event.is_set():
        print_menu()
        try:
            choice = input("  Enter choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  [NEXUS] Goodbye.")
            break

        if exit_event.is_set():
            break

        if choice == "0" or choice.lower() in ("exit", "quit"):
            print("\n  [NEXUS] Take care of yourself. Goodbye.")
            break
        elif choice == "1":
            run_text_mode(None, tts_engine, exit_event)
        elif choice == "2":
            run_voice_mode(None, tts_engine, recognizer, microphone, exit_event)
        elif choice == "3":
            run_hybrid_mode(None, tts_engine, recognizer, microphone, exit_event)
        elif choice == "4":
            run_demo_tests(None, None)
        else:
            print("  [!] Invalid choice. Please enter 0, 1, 2, 3, or 4.")

    exit_event.set()


if __name__ == "__main__":
    main()
