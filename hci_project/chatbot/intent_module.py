"""
Intent classification module for the university chatbot.

Two modes controlled by INTENT_CLASSIFICATION_MODE in chatbot_config.py:
  "rule_based" — keyword-match scoring (fast, fully offline)
  "ai"         — LLM classifies the intent; falls back to rule_based if the
                 LLM is unavailable or returns an unrecognised label.
"""

from chatbot.chatbot_config import (
    UNIVERSITY_INTENTS,
    UNKNOWN_INTENT_RESPONSE,
    INTENT_CLASSIFICATION_MODE,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)

_INTENT_NAMES = list(UNIVERSITY_INTENTS.keys())

_AI_SYSTEM_PROMPT = (
    "You are an intent classifier for a university information chatbot. "
    "Given a student message, reply with ONLY the intent name from this list: "
    + ", ".join(_INTENT_NAMES) + ". "
    "If no intent matches, reply with exactly: Unknown. "
    "Output the intent name only — no explanation, no punctuation."
)


# ---------------------------------------------------------------------------
# Rule-based classifier (keyword match)
# ---------------------------------------------------------------------------

def _classify_intent_rule_based(text: str) -> dict:
    """Classify intent by counting keyword matches. Fast and offline."""
    lower = text.lower()
    words = lower.split()

    all_scores: dict = {}
    matched_per_intent: dict = {}

    for intent_name, intent_data in UNIVERSITY_INTENTS.items():
        matched = []
        for kw in intent_data["keywords"]:
            if kw in lower or any(kw in w for w in words):
                matched.append(kw)
        all_scores[intent_name] = len(matched)
        matched_per_intent[intent_name] = matched

    best_intent = max(all_scores, key=all_scores.get)
    best_score  = all_scores[best_intent]

    if best_score == 0:
        return {
            "intent":     "Unknown",
            "confidence": 0,
            "pattern":    "",
            "all_scores": all_scores,
        }

    return {
        "intent":     best_intent,
        "confidence": best_score,
        "pattern":    ", ".join(matched_per_intent[best_intent]),
        "all_scores": all_scores,
    }


# ---------------------------------------------------------------------------
# AI classifier (LLM-based)
# ---------------------------------------------------------------------------

def _classify_intent_ai(text: str) -> dict:
    """Classify intent via LLM. Falls back to rule_based if LLM is down."""
    from shared.llm_utils import query_ollama

    response = query_ollama(
        prompt=text,
        system_prompt=_AI_SYSTEM_PROMPT,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    if response:
        intent = response.strip().rstrip(".")
        if intent in UNIVERSITY_INTENTS:
            return {
                "intent":     intent,
                "confidence": 1,
                "pattern":    "ai-classified",
                "all_scores": {},
            }

    # LLM unavailable or returned an unrecognised label — fall back
    return _classify_intent_rule_based(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> dict:
    """
    Classify the user's intent.

    Dispatches to the LLM or keyword-match classifier based on
    INTENT_CLASSIFICATION_MODE in chatbot_config.py.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    dict with keys:
        "intent"     : str  — top intent name, or "Unknown"
        "confidence" : int  — match score (rule) or 1 (ai)
        "pattern"    : str  — matched keywords (rule) or "ai-classified"
        "all_scores" : dict — {intent: score} (rule) or {} (ai)
    """
    if INTENT_CLASSIFICATION_MODE == "ai":
        return _classify_intent_ai(text)
    return _classify_intent_rule_based(text)