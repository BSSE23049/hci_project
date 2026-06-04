"""
NEXUS support classification and transition logging module.
Classifies student messages by support category and tracks topic shifts.
All keyword lists and response rules are imported from chatbot_config.py.
"""

from chatbot.chatbot_config import NEXUS_SUPPORT_KEYWORDS, NEXUS_RESPONSE_RULES


def classify_support_need(text: str) -> dict:
    """
    Classify the primary support need from the student's message.

    Scores all categories by counting keyword matches in the text.

    Parameters
    ----------
    text : str
        User input message.

    Returns
    -------
    dict with keys:
        "primary"      : str       — highest-scoring category name
        "all_detected" : list[str] — all categories with score > 0
        "scores"       : dict      — {category: match_count}
    """
    lower = text.lower()
    scores: dict = {}

    for category, keywords in NEXUS_SUPPORT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        scores[category] = count

    all_detected = [cat for cat, score in scores.items() if score > 0]
    best_cat     = max(scores, key=scores.get)
    primary      = best_cat if scores[best_cat] > 0 else "GENERAL"

    #========= HARD CODE FOR EMERGENCY ONLY =========

    # if("fee" in lower or "fees" in lower):
    #     primary = "FINANCIAL"

    return {
        "primary":      primary,
        "all_detected": all_detected,
        "scores":       scores,
    }


def log_support_transition(support_log: list, new_primary: str, turn_number: int) -> list:
    """
    Append a transition entry when the primary support category changes.

    No entry is added if the category is the same as the last logged category.

    Parameters
    ----------
    support_log : list[dict]
        Existing log of transition events (may be empty).
    new_primary : str
        The primary support category detected this turn.
    turn_number : int
        1-based turn index.

    Returns
    -------
    list[dict]
        Updated support_log (new list reference with possible new entry appended).
    """
    if support_log and support_log[-1]["curr"] == new_primary:
        return support_log

    prev = support_log[-1]["curr"] if support_log else "START"
    is_escalation = (new_primary == "WELLBEING" and prev != "WELLBEING")

    support_log = list(support_log)
    support_log.append({
        "prev":         prev,
        "curr":         new_primary,
        "turn":         turn_number,
        "is_escalation": is_escalation,
    })
    return support_log


def nexus_respond(text: str, support_need: str, wellbeing_tier: str) -> str:
    """
    Select the most appropriate NEXUS response by walking NEXUS_RESPONSE_RULES.

    Rules are matched top to bottom; the first matching rule wins.
    A wildcard "*" matches any value for its position.

    This function contains NO hardcoded response strings —
    all responses come from NEXUS_RESPONSE_RULES in chatbot_config.py.

    Parameters
    ----------
    text : str
        The user's message (available for future context-aware rules).
    support_need : str
        Primary support category (e.g. "ACADEMIC").
    wellbeing_tier : str
        Current wellbeing tier (e.g. "STRESSED").

    Returns
    -------
    str
        The first matching response string from the rules list.
    """
    for cat, tier, response in NEXUS_RESPONSE_RULES:
        cat_match  = (cat  == support_need or cat  == "*")
        tier_match = (tier == wellbeing_tier or tier == "*")
        if cat_match and tier_match:
            return response

    return "Thank you for sharing. I am here to help — can you tell me more?"
