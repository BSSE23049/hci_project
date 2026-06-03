"""
Intent classification module for the university chatbot.
Scores each intent by counting keyword matches in the user's text.
All intent definitions are imported from chatbot_config.py.
"""

from chatbot.chatbot_config import UNIVERSITY_INTENTS, UNKNOWN_INTENT_RESPONSE


def classify_intent(text: str) -> dict:
    """
    Classify the user's intent by counting keyword matches.

    Matching uses substring containment so partial words match
    (e.g. "fees" matches keyword "fee").

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    dict with keys:
        "intent"     : str  — top-scoring intent name, or "Unknown"
        "confidence" : int  — raw keyword match count (score)
        "pattern"    : str  — comma-joined list of matched keywords
        "all_scores" : dict — {intent_name: score} for every intent
    """
    lower = text.lower()
    words = lower.split()

    all_scores: dict = {}
    matched_per_intent: dict = {}

    for intent_name, intent_data in UNIVERSITY_INTENTS.items():
        matched = []
        for kw in intent_data["keywords"]:
            # Match if the keyword is a substring of any word, or contained in the full text
            if kw in lower or any(kw in w for w in words):
                matched.append(kw)
        all_scores[intent_name]     = len(matched)
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
