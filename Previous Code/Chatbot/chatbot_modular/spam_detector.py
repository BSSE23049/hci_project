"""
chatbot_modular/spam_detector.py
=================================
NEXUS Wellbeing Engine — 6-Tier Emotional State Tracker

Originally this module held the spam classifier for the university Q&A
bot. NEXUS is a different product (a student wellbeing advisor): every
student message is treated as legitimate emotional data, so there is no
spam stage at all. The filename is retained for import-compatibility
but the contents now implement the wellbeing engine described in
Stage 2 of the assessment brief.

Responsibilities
----------------
  1. assess_wellbeing(text)
     Use VADER (nltk.sentiment.SentimentIntensityAnalyzer) to score
     the text and map the compound score onto the 6-tier wellbeing
     scale defined in constants.WELLBEING_SCALE.

  2. compute_trajectory(wellbeing_log)
     Given the per-turn wellbeing log, decide whether the session is
     improving, declining, or fluctuating, and surface the lowest
     tier reached plus the indices of at-risk turns.

  3. check_and_alert(wellbeing_result, turn_number)
     If the student's tier is CRISIS, print the prominent counsellor
     alert and return True. Otherwise return False silently.

Public API
----------
    assess_wellbeing(text)     -> dict
    compute_trajectory(log)    -> dict
    check_and_alert(wb, turn)  -> bool
"""

from typing import List, Dict

from chatbot_modular.constants import (
    WELLBEING_SCALE,
    AT_RISK_TIERS,
    CRISIS_ALERT_TEMPLATE,
)

# --- Lazy VADER import ------------------------------------------------------
# Importing nltk + downloading the lexicon at module load slows every script
# that touches this package (including the demo and the unit tests). The
# analyzer is built on first use instead. If nltk is missing we fall back
# to a tiny lexicon-based estimator so the module still works.
_sia = None
_VADER_AVAILABLE = True


def _get_analyzer():
    """Return a cached VADER SentimentIntensityAnalyzer, building on first call."""
    global _sia, _VADER_AVAILABLE
    if _sia is not None or not _VADER_AVAILABLE:
        return _sia
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            _sia = SentimentIntensityAnalyzer()
        except LookupError:
            # VADER lexicon not downloaded yet — fetch quietly, then retry.
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            _sia = SentimentIntensityAnalyzer()
    except Exception:
        _VADER_AVAILABLE = False
        _sia = None
    return _sia


# --- Fallback lexicon (only used when nltk is unavailable) ------------------
# A deliberately small list — enough to keep the demo runnable on machines
# without nltk installed, not a serious substitute for VADER.
_FALLBACK_POS = {
    "happy", "good", "great", "better", "thanks", "thank", "glad", "love",
    "okay", "fine", "calm", "hopeful", "relieved",
}
_FALLBACK_NEG = {
    "sad", "bad", "worse", "hopeless", "anxious", "depressed", "panic",
    "cry", "afraid", "overwhelmed", "lonely", "stressed", "stress",
    "tired", "alone", "fail", "failed", "broke", "broken", "struggling",
    "hate", "angry", "fear",
}


def _fallback_compound(text: str) -> float:
    """Cheap polarity estimate used when VADER is not available."""
    tokens = [t.strip(".,!?;:\"'()").lower() for t in text.split()]
    pos = sum(1 for t in tokens if t in _FALLBACK_POS)
    neg = sum(1 for t in tokens if t in _FALLBACK_NEG)
    if pos == 0 and neg == 0:
        return 0.0
    # squash to roughly [-1, 1]
    score = (pos - neg) / (pos + neg + 1)
    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Q2.1 — 6-Tier Wellbeing Classifier
# ---------------------------------------------------------------------------

def assess_wellbeing(text: str) -> Dict:
    """
    Classify a single student utterance onto the 6-tier wellbeing scale.

    Steps
    -----
      1. Run VADER on the text to get the compound polarity score.
      2. Walk WELLBEING_SCALE (top → bottom) and find the band that
         contains the score: lower_bound <= score < upper_bound.
         The top band (THRIVING) uses lower <= score (inf upper bound);
         the bottom band (CRISIS) uses score < upper (−inf lower bound).
      3. Build the return dict and flip is_at_risk=True when the tier
         lands in AT_RISK_TIERS (currently only "CRISIS").

    Args:
        text: Raw student message (any case, punctuation OK).

    Returns:
        dict with keys:
          'tier'       (str)   — tier name e.g. "STRESSED"
          'score'      (float) — VADER compound score, -1.0 to 1.0
          'emoji'      (str)   — visual emoji for the tier
          'is_at_risk' (bool)  — True only on the CRISIS tier
    """
    if not text or not text.strip():
        # An empty input is treated as neutral so the session log stays
        # well-formed even if STT returned silence.
        return {
            "tier":       "NEUTRAL",
            "score":      0.0,
            "emoji":      "😐",
            "is_at_risk": False,
        }

    sia = _get_analyzer()
    if sia is not None:
        score = sia.polarity_scores(text)["compound"]
    else:
        score = _fallback_compound(text)

    # Default to NEUTRAL — every realistic compound score will hit a band
    # below, but this keeps the function total even for NaN edge cases.
    tier_name, emoji = "NEUTRAL", "😐"
    for name, lower, upper, em in WELLBEING_SCALE:
        if lower <= score < upper:
            tier_name, emoji = name, em
            break

    return {
        "tier":       tier_name,
        "score":      round(float(score), 4),
        "emoji":      emoji,
        "is_at_risk": tier_name in AT_RISK_TIERS,
    }


# ---------------------------------------------------------------------------
# Q2.2 — Wellbeing Trajectory
# ---------------------------------------------------------------------------

def compute_trajectory(wellbeing_log: List[Dict]) -> Dict:
    """
    Summarise how the student's wellbeing moved across the session.

    Algorithm (from the brief)
    --------------------------
      1. Split the log in half.
      2. Compute the mean compound score for each half.
      3. If second_avg > first_avg + 0.1 -> 'improving'
         If second_avg < first_avg - 0.1 -> 'declining'
         Otherwise                       -> 'fluctuating'
      4. Find the lowest tier (most negative midpoint) reached.
      5. Collect the 0-based indices of every at-risk turn.

    Args:
        wellbeing_log: list of dicts produced by assess_wellbeing().

    Returns:
        dict with keys:
          'trend'         (str)
          'lowest_tier'   (str)
          'at_risk_turns' (list[int])
    """
    if not wellbeing_log:
        return {"trend": "fluctuating", "lowest_tier": "NEUTRAL", "at_risk_turns": []}

    n = len(wellbeing_log)
    half = max(1, n // 2)
    first_half  = wellbeing_log[:half]
    second_half = wellbeing_log[half:] if n > 1 else wellbeing_log

    first_avg  = sum(r["score"] for r in first_half)  / len(first_half)
    second_avg = sum(r["score"] for r in second_half) / max(1, len(second_half))

    if second_avg > first_avg + 0.1:
        trend = "improving"
    elif second_avg < first_avg - 0.1:
        trend = "declining"
    else:
        trend = "fluctuating"

    # Lowest tier reached = the entry with the smallest compound score.
    # We use the *score* to compare (not tier name) so the answer is
    # robust even if a future tier list is reordered.
    lowest_entry = min(wellbeing_log, key=lambda r: r["score"])
    lowest_tier  = lowest_entry["tier"]

    at_risk_turns = [i for i, r in enumerate(wellbeing_log) if r.get("is_at_risk")]

    return {
        "trend":         trend,
        "lowest_tier":   lowest_tier,
        "at_risk_turns": at_risk_turns,
    }


# ---------------------------------------------------------------------------
# Q2.3 — At-Risk Alert System
# ---------------------------------------------------------------------------

def check_and_alert(wellbeing_result: Dict, turn_number: int) -> bool:
    """
    Print a prominent at-risk alert when the student hits the CRISIS tier.

    The alert message is templated in constants.CRISIS_ALERT_TEMPLATE so
    counselling teams can re-brand it without touching code. The turn
    number is woven into the message so the at-risk message can be
    located later in the session transcript.

    Args:
        wellbeing_result: dict from assess_wellbeing()
        turn_number:      1-based turn index

    Returns:
        True  -- alert was triggered (is_at_risk was True)
        False -- nothing to do
    """
    if wellbeing_result.get("is_at_risk"):
        print(CRISIS_ALERT_TEMPLATE.format(turn=turn_number))
        return True
    return False


# ---------------------------------------------------------------------------
# Backwards-compat shims
# ---------------------------------------------------------------------------
# Older callers (main_modular.py and demo.py) used to import
# build_spam_classifier() and is_spam(). NEXUS has no spam stage, but
# we keep these symbols as no-ops so an unmodified import does not
# break the rest of the package while it migrates.

def build_spam_classifier():
    """No-op kept for legacy import compatibility. NEXUS has no spam stage."""
    return None


def is_spam(query, classifier):
    """Always returns (False, 0.0, 'NEXUS has no spam stage')."""
    return False, 0.0, "NEXUS has no spam stage"
