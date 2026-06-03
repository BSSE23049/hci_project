"""
chatbot_modular/pipeline.py
=============================
Core per-turn processing pipeline for NEXUS.

Every student utterance — whether typed or transcribed — passes
through process_turn(), which chains the four stages of the brief
into a single call:

  Stage A: Wellbeing engine  ->  assess_wellbeing(text)
  Stage B: At-risk alerter   ->  check_and_alert(wb, turn)
  Stage C: Support classifier->  classify_support_need(text)
  Stage D: Transition logger ->  log_support_transition(...)
  Stage E: Empathetic response -> nexus_respond(text, primary, tier)

The accumulated logs (session_log, wellbeing_log, support_log,
transition_log) are kept in a single SessionState dict so callers
can feed it straight into generate_intelligence_report() at the end.

Public API
----------
    new_session()                              -> dict
    process_turn(input_dict, state, tts)       -> dict
    process_query(text, classifier, tts)       -> bool   (legacy)
"""

from typing import Dict

from chatbot_modular.spam_detector     import assess_wellbeing, check_and_alert
from chatbot_modular.intent_classifier import classify_support_need, log_support_transition
from chatbot_modular.response_engine   import nexus_respond
from chatbot_modular.voice_handler     import speak_text


def new_session() -> Dict:
    """
    Build an empty NEXUS session-state dict.

    The session state is just four parallel lists plus a turn counter.
    Keeping them in one dict means callers pass a single argument
    around instead of four — and the intelligence report can simply
    do `**state` at the end if it wants.
    """
    return {
        "turn":           0,
        "session_log":    [],   # list of input dicts: {text, source, turn, word_count}
        "wellbeing_log":  [],   # list of wellbeing dicts
        "support_log":    [],   # list of classifier-result dicts (per turn)
        "transition_log": [],   # list of transition-event dicts
    }


def process_turn(input_dict: Dict, state: Dict, tts_engine=None) -> Dict:
    """
    Run one full NEXUS turn on an already-captured input dict.

    Args:
        input_dict: dict from nexus_get_input() (or hand-built for
                    offline replay). Must have 'text', 'source',
                    'turn', 'word_count'.
        state:      session state dict from new_session().
        tts_engine: optional pyttsx3 engine — when provided, NEXUS
                    will speak its reply.

    Returns:
        dict with the per-turn artefacts:
          'wellbeing'   — assess_wellbeing result
          'support'     — classify_support_need result
          'transition'  — transition event or None
          'response'    — empathetic NEXUS reply (str)
          'alerted'     — True if a CRISIS alert was printed
    """
    text  = input_dict["text"]
    turn  = input_dict["turn"]

    # --- Stage A: wellbeing ----------------------------------------------
    wb = assess_wellbeing(text)
    state["wellbeing_log"].append(wb)

    # --- Stage B: at-risk alert (printed inline if triggered) ------------
    alerted = check_and_alert(wb, turn)

    # --- Stage C: multi-label support classification ---------------------
    support = classify_support_need(text)
    state["support_log"].append(support)

    # --- Stage D: transition logger (records change in primary) ----------
    transition = log_support_transition(
        state["transition_log"], support["primary"], turn,
    )

    # --- Stage E: empathetic response ------------------------------------
    response = nexus_respond(text, support["primary"], wb["tier"])

    # --- Bookkeeping & user-facing output --------------------------------
    state["session_log"].append(input_dict)
    state["turn"] = turn

    all_detected = support["all_detected"] or ["NONE"]
    print(
        f"Turn {turn}: {wb['emoji']} {wb['tier']:<10s} | "
        f"primary={support['primary']:<9s} | all={all_detected}"
    )
    print(f"NEXUS: {response}\n")

    speak_text(response, tts_engine)

    return {
        "wellbeing":  wb,
        "support":    support,
        "transition": transition,
        "response":   response,
        "alerted":    alerted,
    }


# ---------------------------------------------------------------------------
# Legacy single-call shim
# ---------------------------------------------------------------------------
def process_query(query: str, classifier=None, tts_engine=None) -> bool:
    """
    Backwards-compatible single-shot interface.

    Older callers (main_modular.py menu, the legacy demo) pass a
    plain text query and expect a True/False outcome (legacy semantics:
    True = handled, False = blocked-as-spam). NEXUS has no spam stage
    so this function always handles the input — we just bundle a
    minimal one-turn session and call process_turn().

    A fresh session is built each call so legacy callers don't have
    to manage state. That means trajectory / transition / report logic
    is only meaningful when callers use new_session() + process_turn()
    directly.
    """
    state = new_session()
    state["turn"] = 1
    input_dict = {
        "text":       query,
        "source":     "text",
        "turn":       1,
        "word_count": len(query.split()),
    }
    process_turn(input_dict, state, tts_engine)
    return True
