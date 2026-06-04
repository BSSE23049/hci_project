"""
NEXUS wellbeing assessment engine.
Uses VADER sentiment analysis to map user text to a wellbeing tier.
All scale definitions and constants are imported from chatbot_config.py.
"""

import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from chatbot.chatbot_config import NEXUS_WELLBEING_SCALE, NEXUS_CRISIS_LINE


def _get_analyser():
    """
    Lazily load and return the VADER SentimentIntensityAnalyzer.

    Returns
    -------
    nltk.sentiment.SentimentIntensityAnalyzer
    """
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def assess_wellbeing(text: str) -> dict:
    """
    Assess the student's current wellbeing tier from their message text.

    Uses VADER compound score and maps it to the NEXUS_WELLBEING_SCALE.
    The scale is walked top to bottom; the first matching tier is used.

    Parameters
    ----------
    text : str
        User input message.

    Returns
    -------
    dict with keys:
        "tier"      : str   — wellbeing tier name (e.g. "CONTENT")
        "score"     : float — VADER compound score (-1.0 to 1.0)
        "emoji"     : str   — tier emoji
        "is_at_risk": bool  — True only for "CRISIS" tier
    """
    analyser = _get_analyser()
    scores   = analyser.polarity_scores(text)
    compound = scores["compound"]

    # ====== EMERGENCY BOOST (for testing only) ======
    # if ("fee" in text.lower() or "fees" in text.lower()):
    #     compound = 0.32 # artificially boost score 
    

    matched_tier  = "NEUTRAL"
    matched_emoji = "😐"

    for tier_name, low, high, emoji in NEXUS_WELLBEING_SCALE:
        if low <= compound < high:
            matched_tier  = tier_name
            matched_emoji = emoji
            break

    return {
        "tier":       matched_tier,
        "score":      compound,
        "emoji":      matched_emoji,
        "is_at_risk": matched_tier == "CRISIS",
    }


def compute_trajectory(wellbeing_log: list) -> dict:
    """
    Analyse the trajectory of the student's wellbeing across the session.

    Splits the log into two halves and compares mean scores to detect
    whether the student is improving, declining, or fluctuating.

    Parameters
    ----------
    wellbeing_log : list[dict]
        List of assess_wellbeing() result dicts, one per turn.

    Returns
    -------
    dict with keys:
        "trend"        : str       — "improving" | "declining" | "fluctuating"
        "lowest_tier"  : str       — tier name with the lowest compound score
        "at_risk_turns": list[int] — 0-based turn indices where is_at_risk == True
    """
    if not wellbeing_log:
        return {"trend": "fluctuating", "lowest_tier": "N/A", "at_risk_turns": []}

    n     = len(wellbeing_log)
    mid   = n // 2 if n > 1 else 1
    first_half  = wellbeing_log[:mid]
    second_half = wellbeing_log[mid:] if n > 1 else wellbeing_log

    def _mean_score(half: list) -> float:
        if not half:
            return 0.0
        return sum(e["score"] for e in half) / len(half)

    first_avg  = _mean_score(first_half)
    second_avg = _mean_score(second_half)

    if second_avg > first_avg + 0.1:
        trend = "improving"
    elif second_avg < first_avg - 0.1:
        trend = "declining"
    else:
        trend = "fluctuating"

    lowest_entry = min(wellbeing_log, key=lambda e: e["score"])
    lowest_tier  = lowest_entry["tier"]

    at_risk_turns = [i for i, e in enumerate(wellbeing_log) if e["is_at_risk"]]

    return {
        "trend":         trend,
        "lowest_tier":   lowest_tier,
        "at_risk_turns": at_risk_turns,
    }


def check_and_alert(wellbeing_result: dict, turn_number: int) -> bool:
    """
    Print a crisis alert if the student's wellbeing tier is CRISIS.

    Parameters
    ----------
    wellbeing_result : dict
        Return value of assess_wellbeing().
    turn_number : int
        1-based turn index for display in the alert.

    Returns
    -------
    bool
        True if an alert was triggered, False otherwise.
    """
    if not wellbeing_result["is_at_risk"]:
        return False

    print()
    print("=" * 60)
    print("  *** CRISIS ALERT — Turn {} ***".format(turn_number))
    print("  This student may be in immediate distress.")
    print(f"  Please contact the counselling line NOW: {NEXUS_CRISIS_LINE}")
    print("  Counsellor: review this session transcript immediately.")
    print("=" * 60)
    print()
    return True
