"""
Response generation module for the university chatbot.
Supports static rule-based responses and optional LLM-powered responses.
Imports all config values from chatbot_config.py.
"""

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

    # Build a context-specific system prompt that names the intent
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
    Speak the response text aloud using pyttsx3 text-to-speech.

    Falls back to a printed notice if pyttsx3 is not installed.

    Parameters
    ----------
    text : str
        Text to synthesise and speak.

    Returns
    -------
    None
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        print(f"[TTS unavailable] {text}")
    except Exception as exc:
        print(f"[TTS error: {exc}] {text}")
