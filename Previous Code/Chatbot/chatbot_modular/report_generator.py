"""
chatbot_modular/report_generator.py
=====================================
NEXUS Intelligence Report — Stage 5 of the assessment brief.

Builds the counsellor-facing handover document at the end of a
session. Nothing here is shown to the student: the report is a
clinical artefact for the wellbeing team, summarising what NEXUS
observed and recommending whether action is needed.

Public API
----------
    generate_intelligence_report(session_log, wellbeing_log,
                                 support_log, transition_log) -> dict

The function PRINTS the report to console (as required by the brief)
and returns a dict with the computed fields so callers (and unit
tests) can inspect them programmatically.
"""

from collections import Counter
from typing import List, Dict

from chatbot_modular.constants import (
    RISK_SCORE_CONFIG,
    ACTION_THRESHOLDS,
    REPORT_HEADER,
    REPORT_FOOTER,
)
from chatbot_modular.spam_detector import compute_trajectory


def _recommended_action(risk_score: int) -> str:
    """Map a 0–100 risk score onto NO_ACTION / FOLLOW_UP / URGENT_REFERRAL."""
    # ACTION_THRESHOLDS is ordered high->low. The first threshold the
    # score clears wins.
    for threshold, label in ACTION_THRESHOLDS:
        if risk_score >= threshold:
            return label
    return "NO_ACTION"


def generate_intelligence_report(
    session_log:    List[Dict],
    wellbeing_log:  List[Dict],
    support_log:    List[Dict],
    transition_log: List[Dict],
) -> Dict:
    """
    Print the NEXUS counsellor report and return its computed fields.

    Inputs
    ------
      session_log:    list of input dicts from nexus_get_input(), one
                      per turn. Each dict must carry 'source' and
                      'word_count' so we can split voice vs text and
                      compute the verbose-bonus on the risk score.
      wellbeing_log:  list of dicts from assess_wellbeing(), one per
                      turn — used for trajectory + at-risk counts.
      support_log:    list of per-turn classifier results (each with a
                      'primary' key). Used for category frequencies.
      transition_log: list of transition events from
                      log_support_transition().

    Risk score formula (from the brief)
    ----------------------------------
        base_risk           = 20
        wellbeing_penalty   = abs(avg_score) * 40
        at_risk_penalty     = at_risk_count  * 15
        escalation_penalty  = escalation_count * 10
        verbose_bonus       = 5  if avg words/turn > 20 else 0
        risk_score          = clip(round(sum), 0, 100)

    Returns:
        dict with every field that was printed to the console — handy
        for unit tests and CSV export.
    """
    cfg = RISK_SCORE_CONFIG

    # ---- Session counts ---------------------------------------------------
    total_turns = len(session_log)
    voice_turns = sum(1 for s in session_log if s.get("source") == "voice")
    text_turns  = sum(1 for s in session_log if s.get("source") == "text")

    # ---- Wellbeing trajectory --------------------------------------------
    trajectory     = compute_trajectory(wellbeing_log) if wellbeing_log else {
        "trend": "fluctuating", "lowest_tier": "NEUTRAL", "at_risk_turns": [],
    }
    trend          = trajectory["trend"]
    lowest_tier    = trajectory["lowest_tier"]
    at_risk_turns  = trajectory["at_risk_turns"]
    at_risk_count  = len(at_risk_turns)
    # turn indices in trajectory are 0-based; counsellors think in 1-based
    at_risk_turns_1b = [i + 1 for i in at_risk_turns]

    # ---- Support category frequencies ------------------------------------
    primaries = [s.get("primary", "NONE") for s in support_log
                 if s.get("primary") and s.get("primary") != "NONE"]
    category_counts = Counter(primaries)
    ranked = category_counts.most_common()

    # ---- Escalation events -----------------------------------------------
    escalations = [e for e in transition_log if e.get("is_escalation")]
    escalation_count = len(escalations)

    # ---- Risk score ------------------------------------------------------
    if wellbeing_log:
        avg_score = sum(r["score"] for r in wellbeing_log) / len(wellbeing_log)
    else:
        avg_score = 0.0
    avg_words = (
        sum(s.get("word_count", 0) for s in session_log) / total_turns
        if total_turns else 0
    )

    risk = (
        cfg["base_risk"]
        + abs(avg_score) * cfg["wellbeing_weight"]
        + at_risk_count  * cfg["at_risk_weight"]
        + escalation_count * cfg["escalation_weight"]
    )
    if avg_words > cfg["verbose_threshold"]:
        risk += cfg["verbose_bonus"]
    risk_score = max(cfg["score_floor"], min(cfg["score_ceiling"], round(risk)))
    action     = _recommended_action(risk_score)

    # ---- Print the report -------------------------------------------------
    print(REPORT_HEADER)
    print(f"  Session turns total      : {total_turns}")
    print(f"    Voice turns            : {voice_turns}")
    print(f"    Text turns             : {text_turns}")
    print(f"  Wellbeing trajectory     : {trend.upper()}")
    print(f"  Lowest wellbeing tier    : {lowest_tier}")
    print(f"  At-risk alerts triggered : {at_risk_count} "
          f"(turns: {at_risk_turns_1b})")
    print(f"  Avg wellbeing score      : {avg_score:.3f}")
    print(f"  Avg words per turn       : {avg_words:.1f}")
    print(f"  Support categories detected (ranked by frequency):")
    if ranked:
        for category, count in ranked:
            print(f"     - {category:<12s}{count}")
    else:
        print("     - (none)")
    print(f"  Escalation events (into WELLBEING): {escalation_count}")
    for event in escalations:
        print(f"     - turn {event['turn']}: {event['prev']} -> {event['curr']}")
    print(f"  Risk score               : {risk_score}/100")
    print(f"  Recommended action       : {action}")
    print(REPORT_FOOTER)

    return {
        "total_turns":      total_turns,
        "voice_turns":      voice_turns,
        "text_turns":       text_turns,
        "trend":            trend,
        "lowest_tier":      lowest_tier,
        "at_risk_turns":    at_risk_turns_1b,
        "at_risk_count":    at_risk_count,
        "avg_score":        avg_score,
        "avg_words":        avg_words,
        "category_counts":  dict(category_counts),
        "escalation_count": escalation_count,
        "risk_score":       risk_score,
        "action":           action,
    }
