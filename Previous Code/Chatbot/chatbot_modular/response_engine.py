"""
chatbot_modular/response_engine.py
====================================
NEXUS Empathetic Response Generator — Stage 3, Q3.3.

Responsibility
--------------
Produce a context-appropriate response for the student given:
  * the original text,
  * the PRIMARY support category (from classify_support_need),
  * the wellbeing tier (from assess_wellbeing).

The response is empathetic — never robotic — and is selected from
constants.RESPONSE_MATRIX, which encodes every (category, tier)
combination called out in the brief plus graceful fallbacks for the
rest of the grid.

Lookup order
------------
  1. Exact (category, tier) match
        e.g. ("WELLBEING", "CRISIS")
  2. Category-wildcard match
        e.g. ("FINANCIAL", "*")  — used for FINANCIAL + any tier
  3. Tier-only wildcard
        e.g. ("*", "CRISIS")     — generic crisis response for any cat
  4. DEFAULT_RESPONSE
        — generic empathetic fallback when nothing else applies

The original `generate_response` symbol is kept as a thin wrapper so
legacy callers continue to work without changes.

Public API
----------
    nexus_respond(text, support_need, wellbeing) -> str
    generate_response(intent, query, matched_keywords) -> str   (legacy)
"""

from chatbot_modular.constants import RESPONSE_MATRIX, DEFAULT_RESPONSE


def nexus_respond(text: str, support_need: str, wellbeing: str) -> str:
    """
    Return the empathetic response NEXUS should give for this turn.

    Args:
        text:         The student's raw message. Currently not used for
                      lookup (the matrix is keyed on category+tier), but
                      kept in the signature for the brief and to leave
                      room for future text-aware tuning.
        support_need: Primary support category, e.g. 'WELLBEING'.
                      Use 'NONE' (or empty string) when no category fired.
        wellbeing:    Wellbeing tier string, e.g. 'CRISIS'.

    Returns:
        A plain-text response string ready for display or TTS.
    """
    cat  = (support_need or "NONE").upper()
    tier = (wellbeing    or "NEUTRAL").upper()

    # 1. Exact (category, tier) match
    if (cat, tier) in RESPONSE_MATRIX:
        return RESPONSE_MATRIX[(cat, tier)]

    # 2. Category-wildcard — applies for any tier under that category
    if (cat, "*") in RESPONSE_MATRIX:
        return RESPONSE_MATRIX[(cat, "*")]

    # 3. Tier-only wildcard — applies across any category
    if ("*", tier) in RESPONSE_MATRIX:
        return RESPONSE_MATRIX[("*", tier)]

    # 4. Generic fallback
    return DEFAULT_RESPONSE


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------
def generate_response(intent: str, query: str, matched_keywords) -> str:
    """
    Legacy single-arg interface that routes into nexus_respond.

    The old pipeline passed (intent, query, matched_keywords). We treat
    'intent' as the support category and assume NEUTRAL tier — that is
    a sensible default for any old call site that hasn't been migrated
    yet. New code should call nexus_respond() directly.
    """
    return nexus_respond(query, intent, "NEUTRAL")
