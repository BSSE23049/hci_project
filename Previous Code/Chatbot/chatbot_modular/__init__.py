"""
chatbot_modular/__init__.py
============================
Package initialiser for NEXUS — the modular Student Wellbeing Advisor.
"""

# Reconfigure stdout to UTF-8 so the wellbeing emojis (🌟😊😐😟😢🆘)
# render correctly on Windows consoles (which default to cp1252).
# Wrapped in a try because some environments (e.g. piped non-TTY
# stdouts on older Pythons) lack the reconfigure method.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


_DOCSTRING_TAIL = """

Package layout
--------------
constants.py          -- Data only: wellbeing scale, support keywords,
                          response matrix, risk score config.
utils.py              -- Shared helper: _kw_match (word-boundary aware).
spam_detector.py      -- Wellbeing engine (assess_wellbeing,
                          compute_trajectory, check_and_alert).
                          (filename kept for import compatibility)
intent_classifier.py  -- Support classifier (classify_support_need,
                          log_support_transition). (filename kept)
response_engine.py    -- nexus_respond + generate_response shim.
voice_handler.py      -- nexus_capture_audio, nexus_transcribe,
                          nexus_get_input + TTS helpers.
report_generator.py   -- generate_intelligence_report (Stage 5).
pipeline.py           -- new_session + process_turn (and process_query
                          legacy shim).
input_modes.py        -- Live interactive runners (text / voice / hybrid).
demo.py               -- Stage-4 offline STUDENT_LOG replay.

Top-level re-exports (so callers can write
``from chatbot_modular import nexus_respond``).
"""

# Stage 1 — input
from chatbot_modular.voice_handler import (
    nexus_capture_audio,
    nexus_transcribe,
    nexus_get_input,
    initialize_tts,
    speak_text,
    speech_to_text,
    initialize_voice_components,
)

# Stage 2 — wellbeing engine
from chatbot_modular.spam_detector import (
    assess_wellbeing,
    compute_trajectory,
    check_and_alert,
    # legacy shims
    build_spam_classifier,
    is_spam,
)

# Stage 3 — support classifier + response
from chatbot_modular.intent_classifier import (
    classify_support_need,
    log_support_transition,
    identify_intent,           # legacy shim
)
from chatbot_modular.response_engine import (
    nexus_respond,
    generate_response,         # legacy shim
)

# Stage 5 — intelligence report
from chatbot_modular.report_generator import generate_intelligence_report

# Pipeline + interactive runners
from chatbot_modular.pipeline    import new_session, process_turn, process_query
from chatbot_modular.input_modes import run_text_mode, run_voice_mode, run_hybrid_mode
from chatbot_modular.demo        import run_demo_tests, STUDENT_LOG


__all__ = [
    # Stage 1
    "nexus_capture_audio", "nexus_transcribe", "nexus_get_input",
    "initialize_tts", "speak_text",
    "speech_to_text", "initialize_voice_components",
    # Stage 2
    "assess_wellbeing", "compute_trajectory", "check_and_alert",
    # Stage 3
    "classify_support_need", "log_support_transition",
    "nexus_respond",
    # Stage 5
    "generate_intelligence_report",
    # Pipeline
    "new_session", "process_turn", "process_query",
    # Interactive runners
    "run_text_mode", "run_voice_mode", "run_hybrid_mode",
    # Demo
    "run_demo_tests", "STUDENT_LOG",
    # Legacy shims
    "build_spam_classifier", "is_spam",
    "identify_intent", "generate_response",
]
