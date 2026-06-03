"""
NEXUS intelligence report generator.
Computes risk score and prints a formatted counsellor report to the console.
All thresholds are imported from chatbot_config.py.
"""

from chatbot.chatbot_config import NEXUS_RISK_URGENT, NEXUS_RISK_FOLLOW_UP


def generate_intelligence_report(
    session_log: list,
    wellbeing_log: list,
    support_log: list,
    transition_log: list,
) -> None:
    """
    Compute session statistics and print a formatted NEXUS intelligence report.

    Risk score formula (exact):
        base_risk          = 20
        wellbeing_penalty  = abs(avg_wellbeing_score) * 40
        at_risk_penalty    = at_risk_count * 15
        escalation_penalty = escalation_count * 10
        word_count_factor  = 5 if avg_words_per_turn > 20 else 0
        risk_score         = clamp(round(sum of above), 0, 100)

    Action thresholds (from config):
        >= NEXUS_RISK_URGENT    → "URGENT_REFERRAL"
        >= NEXUS_RISK_FOLLOW_UP → "FOLLOW_UP"
        else                    → "NO_ACTION"

    Parameters
    ----------
    session_log : list[dict]
        One entry per turn: {"text", "source", "turn", "word_count"}.
    wellbeing_log : list[dict]
        One entry per turn from assess_wellbeing().
    support_log : list[dict]
        One entry per turn from classify_support_need().
    transition_log : list[dict]
        One entry per topic-shift from log_support_transition().

    Returns
    -------
    None
    """
    total_turns  = len(session_log)
    voice_count  = sum(1 for e in session_log if e.get("source") == "voice")
    text_count   = total_turns - voice_count

    # --- Wellbeing trajectory ---
    from chatbot.nexus_wellbeing import compute_trajectory
    trajectory_data = compute_trajectory(wellbeing_log)
    trend        = trajectory_data["trend"].upper()
    lowest_tier  = trajectory_data["lowest_tier"]
    at_risk_turns = trajectory_data["at_risk_turns"]

    # Find emoji for lowest tier
    from chatbot.chatbot_config import NEXUS_WELLBEING_SCALE
    tier_emoji = {t[0]: t[3] for t in NEXUS_WELLBEING_SCALE}
    lowest_display = f"{tier_emoji.get(lowest_tier, '')} {lowest_tier}"

    # --- Support category frequency ---
    cat_counts: dict = {}
    for entry in support_log:
        cat = entry.get("primary", "GENERAL")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)

    # --- Escalation events ---
    escalation_events = [e for e in transition_log if e.get("is_escalation")]
    escalation_count  = len(escalation_events)

    # --- Risk score ---
    if wellbeing_log:
        avg_wellbeing_score = sum(e["score"] for e in wellbeing_log) / len(wellbeing_log)
    else:
        avg_wellbeing_score = 0.0

    at_risk_count = len(at_risk_turns)

    if session_log:
        avg_words_per_turn = sum(e.get("word_count", 0) for e in session_log) / len(session_log)
    else:
        avg_words_per_turn = 0.0

    base_risk          = 20
    wellbeing_penalty  = abs(avg_wellbeing_score) * 40
    at_risk_penalty    = at_risk_count * 15
    escalation_penalty = escalation_count * 10
    word_count_factor  = 5 if avg_words_per_turn > 20 else 0

    risk_score = base_risk + wellbeing_penalty + at_risk_penalty + escalation_penalty + word_count_factor
    risk_score = max(0, min(100, round(risk_score)))

    if risk_score >= NEXUS_RISK_URGENT:
        action = "URGENT_REFERRAL"
    elif risk_score >= NEXUS_RISK_FOLLOW_UP:
        action = "FOLLOW_UP"
    else:
        action = "NO_ACTION"

    # --- Print report ---
    SEP = "=" * 59
    print()
    print(SEP)
    print("        NEXUS STUDENT INTELLIGENCE REPORT")
    print("        Code: NX-2B  —  Counsellor Eyes Only")
    print(SEP)
    print(f"Session Turns     : {total_turns}  (Voice: {voice_count} | Text: {text_count})")
    print("-" * 59)
    print(f"Wellbeing Trajectory  : {trend}")
    print(f"Lowest Tier Reached   : {lowest_display}")
    print(f"At-Risk Alerts        : {at_risk_count}  "
          f"(turns: {[t + 1 for t in at_risk_turns]})")
    print("-" * 59)
    print("Support Categories (by frequency):")
    for rank, (cat, count) in enumerate(sorted_cats, start=1):
        print(f"  {rank}. {cat:<12} — {count} turn{'s' if count != 1 else ''}")
    if not sorted_cats:
        print("  (none detected)")
    print(f"Escalation Events     : {escalation_count} transitions into WELLBEING")
    for ev in escalation_events:
        print(f"  (Turn {ev['turn']}: {ev['prev']} → {ev['curr']})")
    print("-" * 59)
    print(f"Risk Score            : {risk_score}")
    print(f"Recommended Action    : {action}")
    print(SEP)
    print()
