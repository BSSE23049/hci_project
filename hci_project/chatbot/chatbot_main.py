"""
Chatbot app entry point.
Presents a console menu and runs either the University Chatbot or
NEXUS Wellbeing Advisor conversation loop.
"""

from chatbot.chatbot_config import (
    CHATBOT_MODE,
    DEFAULT_INPUT_MODE,
    SPAM_BLOCKED_RESPONSE,
    AUDIO_RECORD_SECONDS,
    AUDIO_SAMPLE_RATE,
    WHISPER_MODEL,
    ENABLE_UNIVERSITY_MODE,
    ENABLE_NEXUS_MODE,
    ENABLE_MODE_SWITCH,
    ENABLE_INPUT_SWITCH,
    ENABLE_OFFLINE_REPLAY,
)
from chatbot.spam_module       import is_spam
from chatbot.intent_module     import classify_intent
from chatbot.response_module   import generate_university_response, speak_response
from chatbot.nexus_wellbeing   import assess_wellbeing, check_and_alert
from chatbot.nexus_support     import classify_support_need, log_support_transition, nexus_respond
from chatbot.nexus_report      import generate_intelligence_report

# Offline NEXUS test log (Assessment Stage 4)
STUDENT_LOG = [
    "Hi, I need some help please.",
    "I have a major assignment due tomorrow and I have not started.",
    "My laptop also broke yesterday so I cannot access my files.",
    "To be honest I have been struggling a lot lately, not just academically.",
    "I have not been sleeping, I feel completely hopeless about everything.",
    "I think I might need to talk to someone but I do not know who.",
    "Also I got an email saying my fees are overdue and I cannot register.",
    "Sorry for dumping all this. I just feel very alone right now.",
    "Actually, my friend just texted. I feel a tiny bit better now.",
    "Thank you for listening. I will try to contact the counsellor.",
]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _record_and_transcribe() -> str:
    """
    Record audio from the microphone and transcribe it with Whisper.

    Returns
    -------
    str
        Transcribed text, or "" on failure.
    """
    try:
        from shared.audio_utils import capture_audio, transcribe
        audio = capture_audio(seconds=AUDIO_RECORD_SECONDS, sample_rate=AUDIO_SAMPLE_RATE)
        result = transcribe(audio, sample_rate=AUDIO_SAMPLE_RATE, model_name=WHISPER_MODEL)
        text = result["text"]
        if text:
            print(f"[Recognised] {text}")
        return text
    except Exception as exc:
        print(f"[WARN] Voice input failed: {exc}")
        return ""


def _get_input_text(input_mode: str, turn: int) -> tuple:
    """
    Collect user input according to the selected input mode.

    When voice capture returns no text (mic error or silence), falls back
    to a typed prompt for that turn so the session is not lost.

    Parameters
    ----------
    input_mode : str
        "text", "voice", or "hybrid".
    turn : int
        Current turn number (1-based), used in prompts.

    Returns
    -------
    tuple[str, str]
        (user_text, source) where source is "text" or "voice".
    """
    if input_mode == "text":
        text = input("You: ").strip()
        return text, "text"

    if input_mode == "voice":
        text = _record_and_transcribe()
        if not text:
            print("[Voice failed] Please type your message instead.")
            text = input(f"You (turn {turn}): ").strip()
            return text, "text"
        return text, "voice"

    # Hybrid: let the user choose each turn
    choice = input("  [ENTER] = type  |  [V + ENTER] = voice  >  ").strip().lower()
    if choice == "v":
        text = _record_and_transcribe()
        if not text:
            print("[Voice failed] Please type your message instead.")
            text = input("You: ").strip()
            return text, "text"
        return text, "voice"
    text = input("You: ").strip()
    return text, "text"


# ---------------------------------------------------------------------------
# University chatbot conversation loop
# ---------------------------------------------------------------------------

def run_university_chatbot(input_mode: str) -> None:
    """
    Run the university information chatbot conversation loop.

    Steps each turn:
    1. Get input (text/voice/hybrid)
    2. Spam check
    3. Intent classification + confidence display
    4. Response generation
    5. TTS output

    Exits when user types "exit" or "quit".

    Parameters
    ----------
    input_mode : str
        "text", "voice", or "hybrid".

    Returns
    -------
    None
    """
    print("\n[University Chatbot] Type 'exit' to quit.\n")
    turn = 0

    while True:
        turn += 1
        text, source = _get_input_text(input_mode, turn)

        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Spam gate
        flagged, reason = is_spam(text)
        if flagged:
            print(f"[SPAM] {reason}")
            print(f"NEXUS: {SPAM_BLOCKED_RESPONSE}\n")
            continue

        # Intent detection
        result = classify_intent(text)
        intent     = result["intent"]
        confidence = result["confidence"]
        pattern    = result["pattern"]
        print(f"[Intent: {intent} | Score: {confidence} | Keywords: {pattern}]")

        # Response
        response = generate_university_response(intent, text)
        print(f"NEXUS: {response}\n")
        speak_response(response)


# ---------------------------------------------------------------------------
# NEXUS wellbeing conversation loop
# ---------------------------------------------------------------------------

def run_nexus_chatbot(input_mode: str) -> None:
    """
    Run the NEXUS wellbeing advisor conversation loop.

    Maintains session_log, wellbeing_log, support_log, transition_log.
    Prints an intelligence report when the session ends.

    Exits when user types "exit" or "quit".

    Parameters
    ----------
    input_mode : str
        "text", "voice", or "hybrid".

    Returns
    -------
    None
    """
    print("\n[NEXUS Wellbeing Advisor] Type 'exit' to end session.\n")

    session_log: list    = []
    wellbeing_log: list  = []
    support_log: list    = []
    transition_log: list = []
    turn = 0

    while True:
        turn += 1
        text, source = _get_input_text(input_mode, turn)

        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break

        word_count = len(text.split())

        # Wellbeing assessment
        wb_result = assess_wellbeing(text)
        print(f"  {wb_result['emoji']} {wb_result['tier']}  (score: {wb_result['score']:.3f})")
        check_and_alert(wb_result, turn)

        # Support classification
        sup_result = classify_support_need(text)
        print(f"  [{sup_result['primary']}] all: {sup_result['all_detected']}")

        # Transition logging
        transition_log = log_support_transition(transition_log, sup_result["primary"], turn)

        # NEXUS response
        response = nexus_respond(text, sup_result["primary"], wb_result["tier"])
        print(f"NEXUS: {response}\n")
        speak_response(response)

        # Append to logs
        session_log.append({
            "text":       text,
            "source":     source,
            "turn":       turn,
            "word_count": word_count,
        })
        wellbeing_log.append(wb_result)
        support_log.append({"primary": sup_result["primary"]})

    generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log)


# ---------------------------------------------------------------------------
# Offline session replay (Stage 4)
# ---------------------------------------------------------------------------

def run_offline_replay() -> None:
    """
    Process the hardcoded STUDENT_LOG through the full NEXUS pipeline.

    Prints per-turn results then generates the full intelligence report.
    Requires no hardware (no webcam, no microphone).

    Returns
    -------
    None
    """
    print("\n" + "=" * 59)
    print("  NEXUS Offline Session Replay")
    print("=" * 59 + "\n")

    session_log: list    = []
    wellbeing_log: list  = []
    support_log: list    = []
    transition_log: list = []

    for i, text in enumerate(STUDENT_LOG):
        turn = i + 1
        print(f"--- Turn {turn} ---")
        print(f"Student: {text}")

        wb_result  = assess_wellbeing(text)
        sup_result = classify_support_need(text)

        print(
            f"Turn {turn}: {wb_result['emoji']} {wb_result['tier']} | "
            f"{sup_result['primary']} | all: {sup_result['all_detected']}"
        )

        check_and_alert(wb_result, turn)

        transition_log = log_support_transition(transition_log, sup_result["primary"], turn)
        response       = nexus_respond(text, sup_result["primary"], wb_result["tier"])
        print(f"NEXUS: {response}\n")

        session_log.append({
            "text":       text,
            "source":     "text",
            "turn":       turn,
            "word_count": len(text.split()),
        })
        wellbeing_log.append(wb_result)
        support_log.append({"primary": sup_result["primary"]})

    generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def _build_chatbot_menu(current_mode: str, current_input: str) -> list:
    """
    Build the active chatbot menu items by reading flags from chatbot_config
    at call time (reads module attributes, not imported names, so patching
    the config module is reflected immediately).

    Items whose flag is False are excluded; remaining items are numbered 1..n.
    "Exit" is always the last item.

    Parameters
    ----------
    current_mode : str
        Current chatbot mode ("university" or "nexus").
    current_input : str
        Current input mode ("text", "voice", or "hybrid").

    Returns
    -------
    list[dict]
        [{"key": str, "label": str, "action": str}, ...]
    """
    import chatbot.chatbot_config as cfg

    items = []

    # Start conversation — shown when at least one mode is enabled
    if cfg.ENABLE_UNIVERSITY_MODE or cfg.ENABLE_NEXUS_MODE:
        items.append({"label": "Start conversation", "action": "start"})

    # Switch mode — only when both modes exist and the option is on
    if cfg.ENABLE_MODE_SWITCH and cfg.ENABLE_UNIVERSITY_MODE and cfg.ENABLE_NEXUS_MODE:
        other = "nexus" if current_mode == "university" else "university"
        label = f"Switch to {'NEXUS' if other == 'nexus' else 'University'} mode"
        items.append({"label": label, "action": "switch_mode"})

    # Switch input — only when the option is on
    if cfg.ENABLE_INPUT_SWITCH:
        items.append({"label": f"Switch input  (current: {current_input.upper()})",
                      "action": "switch_input"})

    # Offline replay — only in NEXUS mode when the flag allows it
    if cfg.ENABLE_OFFLINE_REPLAY and current_mode == "nexus" and cfg.ENABLE_NEXUS_MODE:
        items.append({"label": "Run offline session replay", "action": "replay"})

    # Exit is always last
    items.append({"label": "Exit", "action": "exit"})

    # Assign sequential keys starting at 1
    for i, item in enumerate(items, start=1):
        item["key"] = str(i)

    return items


def main() -> None:
    """
    Main menu loop for the chatbot app.

    Builds the menu dynamically from ENABLE_* flags in chatbot_config.py.
    Disabled options are hidden and the numbering is always compact (1..n).

    Returns
    -------
    None
    """
    import chatbot.chatbot_config as cfg

    # Validate that at least one mode is enabled
    if not cfg.ENABLE_UNIVERSITY_MODE and not cfg.ENABLE_NEXUS_MODE:
        print("[ERROR] Both chatbot modes are disabled in chatbot_config.py.")
        print("        Set ENABLE_UNIVERSITY_MODE or ENABLE_NEXUS_MODE to True.")
        return

    # Determine starting mode — respect CHATBOT_MODE, fall back to whichever is enabled
    if cfg.CHATBOT_MODE == "nexus" and cfg.ENABLE_NEXUS_MODE:
        current_mode = "nexus"
    elif cfg.ENABLE_UNIVERSITY_MODE:
        current_mode = "university"
    else:
        current_mode = "nexus"

    current_input = DEFAULT_INPUT_MODE

    while True:
        mode_display  = "UNIVERSITY CHATBOT" if current_mode == "university" else "NEXUS WELLBEING ADVISOR"

        print("\n" + "=" * 45)
        print("         HCI Chatbot System")
        print(f"  Mode : {mode_display}")
        print(f"  Input: {current_input.upper()}")
        print("=" * 45)

        menu = _build_chatbot_menu(current_mode, current_input)
        for item in menu:
            print(f"  {item['key']}. {item['label']}")
        print()

        valid_keys = {item["key"]: item["action"] for item in menu}
        choice = input("Select: ").strip()

        if choice not in valid_keys:
            print("[WARN] Invalid selection.")
            continue

        action = valid_keys[choice]

        if action == "start":
            if current_mode == "university" and cfg.ENABLE_UNIVERSITY_MODE:
                run_university_chatbot(current_input)
            elif current_mode == "nexus" and cfg.ENABLE_NEXUS_MODE:
                run_nexus_chatbot(current_input)

        elif action == "switch_mode":
            if current_mode == "university" and cfg.ENABLE_NEXUS_MODE:
                current_mode = "nexus"
                print("[Switched to NEXUS Wellbeing Advisor]")
            elif current_mode == "nexus" and cfg.ENABLE_UNIVERSITY_MODE:
                current_mode = "university"
                print("[Switched to University Chatbot]")

        elif action == "switch_input":
            modes = ["text", "voice", "hybrid"]
            current_input = modes[(modes.index(current_input) + 1) % len(modes)]
            print(f"[Input mode: {current_input.upper()}]")

        elif action == "replay":
            run_offline_replay()

        elif action == "exit":
            print("Goodbye.")
            break
