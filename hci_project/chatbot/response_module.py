"""
Response generation module for the university chatbot.
Supports static rule-based responses and optional LLM-powered responses.
TTS runs in a subprocess so pyttsx3 is never initialised more than once
per process, avoiding "run loop already started" errors on repeated calls.
"""

import re
import sys
import subprocess
import time

from chatbot.chatbot_config import (
    UNIVERSITY_INTENTS,
    UNKNOWN_INTENT_RESPONSE,
    USE_LLM,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from shared.llm_utils import query_ollama

_UNIVERSITY_SYSTEM_PROMPT = (
    "You are a helpful and friendly university information assistant. "
    "You answer student questions about admissions, fees, courses, schedules, "
    "the library, accommodation, and transport. "
    "Keep responses concise (2-3 sentences), factual, and polite. "
    "If you are unsure, direct the student to the relevant university office."
)

# Inline pyttsx3 script executed in a fresh subprocess each call.
_TTS_SUBPROCESS_SCRIPT = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.setProperty('rate', 160)\n"
    "e.setProperty('volume', 0.9)\n"
    "v = e.getProperty('voices')\n"
    "if v: e.setProperty('voice', v[0].id)\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)


def generate_university_response(intent: str, user_text: str) -> str:
    """
    Generate a response for the detected university chatbot intent.

    If USE_LLM is True, attempts to call the local Ollama server.
    Falls back to the static response from UNIVERSITY_INTENTS if Ollama is
    unavailable or returns None.

    Parameters
    ----------
    intent : str
        Detected intent name, or "Unknown".
    user_text : str
        The original user input text.

    Returns
    -------
    str
        Response string to display (and optionally speak) to the user.
    """
    if intent == "Unknown":
        return UNKNOWN_INTENT_RESPONSE

    static_response = UNIVERSITY_INTENTS[intent]["response_static"]

    if not USE_LLM:
        return static_response

    system_prompt = (
        f"{_UNIVERSITY_SYSTEM_PROMPT}\n\n"
        f"The student is asking about: {intent}. "
        f"Focus your answer on {intent.lower()} information only."
    )
    llm_response = query_ollama(
        prompt=user_text,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return llm_response if llm_response else static_response


def speak_response(text: str) -> None:
    """
    Speak the response text aloud using pyttsx3 via a subprocess.

    Running pyttsx3 in a fresh subprocess on every call prevents the
    "run loop already started" error that occurs when pyttsx3.init() is
    called multiple times in the same process.

    Falls back silently if pyttsx3 is not installed.

    Parameters
    ----------
    text : str
        Text to synthesise and speak.
    """
    try:
        import pyttsx3  # noqa: F401 — availability check only
    except ImportError:
        return

    # Keep only the first 3 non-separator lines, strip markdown symbols
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    spoken_lines = []
    for ln in lines:
        if re.match(r"^[-|:=\s]+$", ln):
            continue
        spoken_lines.append(ln)
        if len(spoken_lines) >= 3:
            break
    spoken = ". ".join(spoken_lines)[:300] # Limit length to avoid excessively long TTS calls
    clean  = re.sub(r"[*_`#|]", " ", spoken) # Remove common markdown symbols that don't speak well
    clean  = re.sub(r":{1,}",   ",", clean) # Replace colons (e.g. in "Fees:") with commas for better speech flow
    clean  = re.sub(r"-{2,}",   " ", clean) # Replace multiple dashes (e.g. in "Course Schedule ---") with space
    clean  = re.sub(r"\s+",     " ", clean).strip() # Collapse multiple spaces and trim

    if not clean:
        return

    print("  [TTS] Speaking response...")
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _TTS_SUBPROCESS_SCRIPT, clean],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = proc.communicate(timeout=90)
            if proc.returncode != 0 and stderr:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                if err_msg:
                    print(f"  [TTS] Error: {err_msg.splitlines()[-1]}")
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.communicate(timeout=2)
            except Exception:
                pass
            print("  [TTS] Subprocess timed out.")
    except Exception as exc:
        print(f"  [TTS] Launch error: {exc}")
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass