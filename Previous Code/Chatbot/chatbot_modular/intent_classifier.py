"""
chatbot_modular/intent_classifier.py
======================================
NEXUS Support Classifier — Stage 3 of the assessment brief.

Originally this module produced a single-label "intent" from 8 generic
university topics. NEXUS works differently: a single student message
can simultaneously belong to MULTIPLE support categories (e.g. an
ACADEMIC question can carry strong WELLBEING signals at the same time).
This module therefore performs MULTI-LABEL classification and exposes
the per-category scores so the response engine and the intelligence
report can both make informed decisions.

The filename is preserved for backwards-compatible imports; the
mandated function names from the brief (classify_support_need and
log_support_transition) live here.

Responsibilities
----------------
  1. classify_support_need(text)
     Score every support category and return:
       * primary       — the highest-scoring category
       * all_detected  — every category with score > 0
       * scores        — full category -> score dict

  2. log_support_transition(support_log, new_primary, turn_number)
     Append a transition event to support_log whenever the primary
     category changes from one turn to the next. Mark
     is_escalation=True when the transition lands in WELLBEING from
     any other category — counsellors care most about that move.

Public API
----------
    classify_support_need(text) -> dict
    log_support_transition(support_log, new_primary, turn_number) -> dict or None
"""

from typing import Dict, List, Optional

from chatbot_modular.constants import SUPPORT_KEYWORDS, ESCALATION_TARGET
from chatbot_modular.utils import _kw_match


# ---------------------------------------------------------------------------
# Q3.1 — Multi-Label Support Classifier
# ---------------------------------------------------------------------------

def classify_support_need(text: str) -> Dict:
    """
    Score the message against every support category and return all hits.

    Scoring
    -------
    Each keyword that matches the lowercased query contributes +1 to its
    category's score. Multi-word keywords are matched as substrings;
    short keywords (<= 3 chars) use word boundaries so that, e.g., "hr"
    doesn't match inside "her" or "their".

    Return shape (from the brief)
    -----------------------------
        {
            'primary':      'ACADEMIC',                # highest-scoring
            'all_detected': ['ACADEMIC', 'WELLBEING'], # every score > 0
            'scores':       {'ACADEMIC': 3, 'WELLBEING': 1, ...}
        }

    Edge cases
    ----------
      * Empty / whitespace-only text -> primary 'NONE', empty lists.
      * No category matches          -> primary 'NONE', empty all_detected.
      * Ties on the top score        -> first category in
                                        SUPPORT_KEYWORDS dict order wins;
                                        every tied category appears in
                                        all_detected.

    Args:
        text: Student's raw message (any case, any punctuation).

    Returns:
        dict with keys 'primary', 'all_detected', 'scores'.
    """
    if not text or not text.strip():
        return {"primary": "NONE", "all_detected": [], "scores": {}}

    q = text.lower().strip()
    scores: Dict[str, int] = {}

    for category, keywords in SUPPORT_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            kw_low = kw.lower()
            if " " in kw_low:
                # Multi-word keyword: plain substring match
                if kw_low in q:
                    hits += 1
            else:
                # Single-word keyword: boundary-aware match
                if _kw_match(kw_low, q):
                    hits += 1
        if hits > 0:
            scores[category] = hits

    if not scores:
        return {"primary": "NONE", "all_detected": [], "scores": {}}

    top_score   = max(scores.values())
    # all_detected = every category that landed at least one hit, ordered
    # by score descending (then by SUPPORT_KEYWORDS insertion order).
    all_detected = sorted(
        scores.keys(),
        key=lambda c: (-scores[c], list(SUPPORT_KEYWORDS.keys()).index(c)),
    )
    # primary = first tied winner under the same ordering
    primary = next(c for c in all_detected if scores[c] == top_score)

    return {
        "primary":      primary,
        "all_detected": all_detected,
        "scores":       scores,
    }


# ---------------------------------------------------------------------------
# Q3.2 — Support Transition Logger
# ---------------------------------------------------------------------------

def log_support_transition(
    support_log: List[Dict],
    new_primary: str,
    turn_number: int,
) -> Optional[Dict]:
    """
    Append a transition event to support_log when the primary category changes.

    Why track transitions?
    ----------------------
    The most clinically meaningful signal in a session is often the
    SHIFT — a student who opens with TECHNICAL questions and then moves
    to WELLBEING is more urgent than a student stable in either
    category. The intelligence report uses these events directly.

    Event shape (from the brief):
        {
            'prev': 'ACADEMIC',     # category from the previous turn
            'curr': 'WELLBEING',    # category from the current turn
            'turn': 4,              # 1-based turn number
            'is_escalation': True,  # transition INTO WELLBEING from elsewhere
        }

    Logic
    -----
      * The first turn has no previous category, so the function returns
        None and leaves support_log untouched (no transition to log).
      * If the previous primary equals new_primary, no event is logged.
      * Transitions into WELLBEING from any other (non-WELLBEING)
        category are flagged is_escalation=True. Movement OUT of
        WELLBEING is logged but NOT marked as escalation.
      * 'NONE' is treated as a real previous state — moving from NONE
        into WELLBEING still counts as an escalation.

    Args:
        support_log: list of previous transition events (mutated).
        new_primary: primary category for the current turn.
        turn_number: 1-based turn number.

    Returns:
        The newly appended event dict, or None if no transition fired.
    """
    # Find the previous primary by walking back through support_log.
    # support_log only contains events at the moment of change, so the
    # latest event's 'curr' is the most recent primary on record.
    prev_primary = None
    for event in reversed(support_log):
        if "curr" in event:
            prev_primary = event["curr"]
            break

    if prev_primary is None:
        # First turn: nothing to compare against — just seed the log
        # with the current category so future calls can detect changes.
        seed = {
            "prev": None,
            "curr": new_primary,
            "turn": turn_number,
            "is_escalation": False,
        }
        support_log.append(seed)
        return None

    if prev_primary == new_primary:
        return None   # no transition

    event = {
        "prev": prev_primary,
        "curr": new_primary,
        "turn": turn_number,
        "is_escalation": (
            new_primary == ESCALATION_TARGET and prev_primary != ESCALATION_TARGET
        ),
    }
    support_log.append(event)
    return event


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------
# main_modular.py and demo.py still import identify_intent() from this
# module. The shim re-routes them to the multi-label classifier so
# unmodified callers continue to receive sensible output instead of an
# ImportError. The "confidence" value is synthesised from the top score.

def identify_intent(query):
    """
    Legacy single-label interface that wraps the multi-label classifier.

    Returns the same 3-tuple shape the old module produced:
        (primary_category, confidence, matched_keywords)
    """
    result = classify_support_need(query)
    primary = result["primary"]
    if primary == "NONE":
        return "unknown", 0.0, []
    scores = result["scores"]
    total  = sum(scores.values()) or 1
    confidence = round(min(0.99, max(0.30, scores[primary] / total)), 2)
    # We don't track which exact keywords matched (the new classifier
    # only stores counts), so return an empty list for legacy callers.
    return primary, confidence, []
