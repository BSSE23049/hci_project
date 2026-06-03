"""
chatbot_modular/input_modes.py
================================
NEXUS interactive runtime modes — Text, Voice, Hybrid.

Each runner opens a NEXUS session, captures input turn by turn, runs
the pipeline, and on exit prints the counsellor intelligence report.

Differences from the old chatbot
--------------------------------
  * No "is this spam?" stage — every student message is processed.
  * Session state is accumulated across turns (so trajectory and
    transition logic actually do something).
  * On exit, generate_intelligence_report() is called automatically
    so the counsellor has a record of the session.

Exit conditions (all modes)
    - User types/says "exit", "quit", "back", or "menu"
    - ESC key is pressed (exit_event is set by background thread)
    - Ctrl+C is pressed (KeyboardInterrupt)

Public API
----------
    run_text_mode(classifier, tts_engine, exit_event)
    run_voice_mode(classifier, tts_engine, rec, mic, exit_event)
    run_hybrid_mode(classifier, tts_engine, rec, mic, exit_event)
"""

import threading

from chatbot_modular.pipeline         import new_session, process_turn
from chatbot_modular.voice_handler    import (
    speech_to_text,
    nexus_capture_audio,
    nexus_transcribe,
    SD_AVAILABLE,
    NP_AVAILABLE,
)
from chatbot_modular.report_generator import generate_intelligence_report


_EXIT_WORDS = {"exit", "quit", "back", "menu", "stop", "bye", "goodbye"}


def _print_report(state):
    """Print the counsellor intelligence report for `state` (if there are turns)."""
    if not state["session_log"]:
        return
    print("\n  Generating counsellor intelligence report...")
    generate_intelligence_report(
        state["session_log"],
        state["wellbeing_log"],
        state["support_log"],
        state["transition_log"],
    )


def _ingest_turn(text: str, source: str, state: dict, tts_engine) -> None:
    """Build the input dict for one turn and run it through the pipeline."""
    state["turn"] += 1
    input_dict = {
        "text":       text,
        "source":     source,
        "turn":       state["turn"],
        "word_count": len(text.split()),
    }
    process_turn(input_dict, state, tts_engine)


# ---------------------------------------------------------------------------
# TEXT MODE
# ---------------------------------------------------------------------------
def run_text_mode(classifier, tts_engine, exit_event: threading.Event) -> None:
    """Continuous typed-input session until the user exits."""
    print("\n  [TEXT MODE] Type your message. Type 'exit' to end the session.")
    state = new_session()

    while not exit_event.is_set():
        try:
            raw = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if exit_event.is_set() or not raw:
            if not raw:
                continue
            break

        if raw.lower() in _EXIT_WORDS:
            print("  [Ending session]")
            break

        _ingest_turn(raw, "text", state, tts_engine)

    _print_report(state)


# ---------------------------------------------------------------------------
# VOICE MODE  (Whisper-based)
# ---------------------------------------------------------------------------
def run_voice_mode(
    classifier,
    tts_engine,
    recognizer,
    microphone,
    exit_event: threading.Event,
) -> None:
    """Voice-input session using Whisper (with a typed fallback per turn)."""
    if not (SD_AVAILABLE and NP_AVAILABLE):
        print("\n  [VOICE MODE] Voice input unavailable on this machine.")
        print("  Install sounddevice + numpy + openai-whisper to enable.")
        return

    print("\n  [VOICE MODE] Speak your message. Say or type 'exit' to end.")
    state = new_session()

    while not exit_event.is_set():
        audio  = nexus_capture_audio()
        result = nexus_transcribe(audio)
        text   = result["text"].strip()
        source = "voice"

        if not text:
            try:
                fallback = input(
                    "  [Voice failed] Type a message | 'retry' | 'exit'\n  > "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not fallback:
                continue
            low = fallback.lower()
            if low in _EXIT_WORDS:
                break
            if low in ("retry", "again", "listen"):
                continue
            text   = fallback
            source = "text"

        if text.lower() in _EXIT_WORDS:
            print("  [Ending session]")
            break

        _ingest_turn(text, source, state, tts_engine)

    _print_report(state)


# ---------------------------------------------------------------------------
# HYBRID MODE
# ---------------------------------------------------------------------------
def run_hybrid_mode(
    classifier,
    tts_engine,
    recognizer,
    microphone,
    exit_event: threading.Event,
) -> None:
    """Hybrid mode — start in text, type 'voice' to switch, 'text' to switch back."""
    voice_ok = SD_AVAILABLE and NP_AVAILABLE
    mode     = "text"
    state    = new_session()

    def _wants_voice(s):
        return s.lower().strip() in {"voice", "voice mode", "switch to voice"}

    def _wants_text(s):
        return s.lower().strip() in {"text", "text mode", "switch to text"}

    def _wants_exit(s):
        return s.lower().strip() in _EXIT_WORDS

    print("\n  [HYBRID MODE] Starting in TEXT mode.")
    if voice_ok:
        print("  Type 'voice' to switch to voice input, 'text' to switch back.")
    else:
        print("  [Note] Voice unavailable on this machine.")
    print("  Type 'exit' to end the session.\n")

    while not exit_event.is_set():
        if mode == "text":
            try:
                raw = input("  [TEXT] You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not raw:
                continue
            if _wants_exit(raw):
                print("  [Ending session]")
                break
            if _wants_voice(raw):
                if voice_ok:
                    mode = "voice"
                    print("  [Switched to VOICE mode]")
                else:
                    print("  [Voice unavailable]")
                continue
            _ingest_turn(raw, "text", state, tts_engine)

        else:  # voice mode
            print("  [VOICE] Listening — say 'text' to switch back.")
            audio  = nexus_capture_audio()
            result = nexus_transcribe(audio)
            text   = result["text"].strip()

            if not text:
                try:
                    alt = input(
                        "  [Voice failed] Type message | 'text' | 'retry' | 'exit'\n  > "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if not alt:
                    continue
                if _wants_exit(alt):
                    break
                if _wants_text(alt):
                    mode = "text"
                    print("  [Switched to TEXT mode]")
                    continue
                if alt.lower() in ("retry", "voice", "listen", "again"):
                    continue
                _ingest_turn(alt, "text", state, tts_engine)
                continue

            if _wants_exit(text):
                break
            if _wants_text(text):
                mode = "text"
                print("  [Switched to TEXT mode]")
                continue

            _ingest_turn(text, "voice", state, tts_engine)

    _print_report(state)
