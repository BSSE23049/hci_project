"""
NEXUS support classification and transition logging module.
Classifies student messages by support category and tracks topic shifts.
All keyword lists and response rules are imported from chatbot_config.py.
"""

from email.mime import text

from chatbot.chatbot_config import (
    NEXUS_SUPPORT_KEYWORDS,
    NEXUS_RESPONSE_RULES,
    USE_LLM,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from shared.llm_utils import query_ollama
# ---------------------------------------------------------------------------
# NEXUS LLM system prompt
# Used by nexus_respond() when USE_LLM = True.
# Edit this string to change the advisor's tone, scope, or instructions
# without touching any logic file.
# ---------------------------------------------------------------------------
NEXUS_LLM_SYSTEM_PROMPT = (
    "You are NEXUS, a compassionate and professional university student wellbeing advisor. "
    "Your role is to support students facing academic, emotional, financial, technical, "
    "social, and administrative challenges. "
    "Always respond with warmth and empathy — validate the student's feelings first, then "
    "offer practical guidance and refer to relevant university support services where appropriate. "
    "Keep your response concise: 2 to 3 sentences only. "
    "Never dismiss or minimise a student's concerns. "
    "If the student appears to be in crisis, always include the university counselling line number."
)


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
    Generate a NEXUS wellbeing response.

    When USE_LLM = False  : walks NEXUS_RESPONSE_RULES and returns the first
                            matching static string (wildcard "*" matches any field).
    When USE_LLM = True   : calls Ollama with NEXUS_LLM_SYSTEM_PROMPT enriched
                            with the detected support category and wellbeing tier.
                            Falls back to the rule-based static response if Ollama
                            is unavailable or returns nothing.

    Parameters
    ----------
    text : str
        The student's raw message — sent as the user prompt to Ollama.
    support_need : str
        Primary support category detected this turn (e.g. "ACADEMIC").
    wellbeing_tier : str
        Current wellbeing tier (e.g. "STRESSED").

    Returns
    -------
    str
        Dynamic LLM response, or the matching static rule response as fallback.
    """
    # Always compute the rule-based response first — used as fallback
    static_response = "Thank you for sharing. I am here to help — can you tell me more?"
    for cat, tier, response in NEXUS_RESPONSE_RULES:
        cat_match  = (cat  == support_need or cat  == "*")
        tier_match = (tier == wellbeing_tier or tier == "*")
        if cat_match and tier_match:
            static_response = response
            break

    if not USE_LLM:
        return static_response

    # Build context-aware system prompt (same pattern as university chatbot)
    system_prompt = (
        f"{NEXUS_LLM_SYSTEM_PROMPT}\n\n"
        f"Student wellbeing tier : {wellbeing_tier}.\n"
        f"Primary support need   : {support_need}.\n"
        f"Tailor your response specifically to this wellbeing level and support category."
    )

    llm_response = query_ollama(
        prompt=text,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return llm_response if llm_response else static_response
