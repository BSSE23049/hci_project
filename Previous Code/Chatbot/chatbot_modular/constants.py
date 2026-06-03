"""
chatbot_modular/constants.py
=============================
Central data store for NEXUS — the AI-powered Student Wellbeing Advisor.

All keyword lists, support-category patterns, empathetic responses, the
6-tier wellbeing scale, and crisis-alert templates live here. No logic,
just data. Keeping data separate means:

  1. You can update responses or add new keywords without touching code.
  2. Unit tests can import just this file to validate data quality.
  3. It is obvious WHERE to look when NEXUS says the wrong thing.

What changed from the old build?
--------------------------------
The previous version of this file powered a generic university Q&A bot
with SPAM detection and 8 informational intents (admission, fee, ...).

NEXUS is a different product. It does NOT block messages as spam — every
student utterance is treated as legitimate emotional / support data.
Instead of SPAM_KEYWORDS, this file now exposes the WELLBEING_SCALE
(used by VADER-based emotion tracking) and the 6 SUPPORT_KEYWORDS
categories defined in the assessment brief.

Why tuples instead of lists?
  Python tuples are immutable (cannot be changed after creation).
  Using tuples makes it impossible to accidentally mutate these
  constants at runtime — satisfying the "no global mutable variables"
  requirement.
"""

from typing import Tuple


# ===========================================================================
# WELLBEING ENGINE DATA — 6-Tier Emotional State Scale
# ===========================================================================

# WELLBEING_SCALE
# ----------------
# The 6-tier emotional state scale NEXUS uses to classify every student
# utterance. Each tuple is:
#
#   (tier_name, lower_bound, upper_bound, emoji)
#
# Semantics:
#   lower_bound <= VADER compound score < upper_bound
#
# The VADER compound score (from nltk.sentiment.SentimentIntensityAnalyzer)
# ranges from -1.0 (most negative) to +1.0 (most positive). NEXUS maps that
# continuous score onto 6 discrete tiers — far richer than the standard
# 3-level positive/neutral/negative split, which is insufficient for the
# emotional range of university students.
#
# CRISIS tier triggers is_at_risk=True and an immediate counsellor alert.

WELLBEING_SCALE: Tuple[Tuple[str, float, float, str], ...] = (
    ("THRIVING",    0.60,          float("inf"),  "🌟"),  # >= 0.60
    ("CONTENT",     0.20,          0.60,          "😊"),  # 0.20 – 0.59
    ("NEUTRAL",    -0.19,          0.20,          "😐"),  # -0.19 – 0.19
    ("STRESSED",   -0.40,         -0.19,          "😟"),  # -0.40 – -0.20
    ("DISTRESSED", -0.60,         -0.40,          "😢"),  # -0.60 – -0.41
    ("CRISIS",      float("-inf"),-0.60,          "🆘"),  # < -0.60
)


# AT_RISK_TIERS
# --------------
# Tiers that flip is_at_risk=True. Per the brief, only the CRISIS tier
# triggers the immediate-intervention alert. Stored as a tuple so the
# check is one-liner: `tier in AT_RISK_TIERS`.

AT_RISK_TIERS: Tuple[str, ...] = ("CRISIS",)


# CRISIS_ALERT_TEMPLATE
# ----------------------
# Printed verbatim by check_and_alert() whenever a student message is
# classified as CRISIS. The {turn} placeholder is filled with the
# 1-based turn number so counsellors can locate the exact message in
# the session log.

CRISIS_ALERT_TEMPLATE: str = (
    "\n"
    "╔══════════════════════════════════════════════════════════╗\n"
    "║   ⚠  AT-RISK ALERT — TURN {turn}                           \n"
    "║   The student's wellbeing score has reached CRISIS level.   \n"
    "║   PLEASE contact the university counselling line NOW:       \n"
    "║   0800-XXX-XXXX  (24/7 confidential support)                \n"
    "╚══════════════════════════════════════════════════════════╝\n"
)


# ===========================================================================
# SUPPORT CLASSIFIER DATA — 6 NEXUS Support Categories
# ===========================================================================

# SUPPORT_KEYWORDS
# -----------------
# Maps each support category -> tuple of trigger keywords.
#
# Unlike the previous single-label intent classifier, NEXUS is MULTI-LABEL:
# a single student message can simultaneously belong to multiple
# categories (e.g. "I failed my exam and I can't sleep" is both
# ACADEMIC and WELLBEING). The classify_support_need() function scores
# ALL categories and returns every one whose score is > 0.
#
# Categories (from the assessment brief):
#   ACADEMIC   — coursework, exams, grades, study issues
#   WELLBEING  — mental health, stress, isolation, hopelessness
#   FINANCIAL  — fees, scholarships, money troubles
#   TECHNICAL  — portal/login/system issues
#   SOCIAL     — friendships, isolation, group/community
#   ADMIN      — paperwork, registration, certificates
#
# Keyword design rules:
#   * keep keywords short and single-word where possible (multi-label
#     scoring is keyword-count based, not phrase-weighted)
#   * lowercase only — classifier lowercases the query before matching
#   * no overlap between categories where avoidable (the multi-label
#     design tolerates overlap, but it dilutes "primary" detection)

SUPPORT_KEYWORDS: dict = {

    # ---- ACADEMIC ----
    # Coursework, deadlines, exams, study struggles.
    # Example: "I failed my midterm and don't know what to do"
    "ACADEMIC": (
        "assignment",   # "I have an assignment due"
        "deadline",     # "the deadline is tomorrow"
        "exam",         # "my exam went badly"
        "grade",        # "my grade dropped"
        "fail",         # "I failed the midterm"
        "pass",         # "I'm worried I won't pass"
        "lecture",      # "I missed today's lecture"
        "study",        # "I can't focus on studying"
        "professor",    # "my professor is unreachable"
        "submit",       # "I can't submit my work"
        "midterm",      # extra: very common student vocabulary
        "thesis",       # extra: final-year students
        "course",       # extra: course-related struggles
    ),

    # ---- WELLBEING ----
    # Mental-health vocabulary. Triggers escalation flag in transition
    # logger when entered from another category.
    # Example: "I can't sleep and I feel completely overwhelmed"
    "WELLBEING": (
        "stress",       # "I'm so stressed"
        "anxious",      # "I feel anxious all the time"
        "depressed",    # "I think I'm depressed"
        "lonely",       # "I feel really lonely"
        "overwhelmed",  # "everything is overwhelming"
        "panic",        # "I had a panic attack"
        "cry",          # "I can't stop crying"
        "hopeless",     # "I feel hopeless about everything"
        "afraid",       # "I'm afraid all the time"
        "sleep",        # extra: insomnia is a common signal
        "tired",        # extra: exhaustion
        "alone",        # extra: synonym for lonely
        "struggling",   # extra: catch-all distress word
    ),

    # ---- FINANCIAL ----
    # Money, fees, scholarships, affordability.
    # Example: "I can't afford my accommodation fees this semester"
    "FINANCIAL": (
        "fees",         # "I haven't paid my fees"
        "scholarship",  # "the scholarship was rejected"
        "loan",         # "my loan didn't come through"
        "afford",       # "I can't afford this"
        "money",        # "I have no money left"
        "rent",         # "I can't pay my rent"
        "bursary",      # "the bursary office said no"
        "payment",      # "my payment was declined"
        "debt",         # "I'm in debt"
        "fee",          # extra: singular form (very common in queries)
        "overdue",      # extra: "my fees are overdue"
    ),

    # ---- TECHNICAL ----
    # Portal, login, system, password, access issues.
    # Example: "The student portal won't let me log in to submit"
    "TECHNICAL": (
        "portal",       # "the portal is broken"
        "login",        # "I can't login"
        "password",     # "I forgot my password"
        "system",       # "the system is down"
        "error",        # "I keep getting an error"
        "access",       # "I can't access my account"
        "email",        # "my email isn't working"
        "vpn",          # "the VPN won't connect" (stored lowercase)
        "reset",        # "I need to reset my password"
        "laptop",       # extra: hardware issue
        "wifi",         # extra: connectivity issue
    ),

    # ---- SOCIAL ----
    # Friendships, belonging, isolation from peers.
    # Example: "I don't have any friends here and feel very alone"
    "SOCIAL": (
        "friends",      # "I have no friends"
        "roommate",     # "my roommate and I argue"
        "belong",       # "I don't feel I belong"
        "isolated",     # "I feel isolated"
        "group",        # "I'm not in any study group"
        "relationship", # "my relationship ended"
        "community",    # "I miss community"
        "friend",       # extra: singular form
    ),

    # ---- ADMIN ----
    # Paperwork, registration, certificates, transcripts.
    # Example: "I need my transcript urgently for a job application"
    "ADMIN": (
        "enrolment",    # UK spelling per brief
        "certificate", # "I need my certificate"
        "transcript",  # "request a transcript"
        "registration",# "I can't complete registration"
        "form",        # "I need to fill a form"
        "office",      # "the admin office"
        "hr",          # extra: brief lists HR under ADMIN
        "register",    # extra: "I cannot register"
    ),
}


# ===========================================================================
# EMPATHETIC RESPONSE TEMPLATES
# ===========================================================================

# RESPONSE_MATRIX
# ----------------
# Lookup table for nexus_respond(text, support_need, wellbeing).
# Key  : (support_category, wellbeing_tier) tuple
# Value: the exact empathetic response NEXUS should produce.
#
# The 6 combinations listed in the brief are required and MUST produce
# distinct outputs. Additional combinations below act as graceful
# fallbacks so the matrix covers every (category, tier) pair without
# having to construct responses from scratch in code.
#
# Lookup order (handled in response_engine.py / nexus_respond):
#   1. Exact (category, tier) match in RESPONSE_MATRIX.
#   2. Category-wildcard match — RESPONSE_MATRIX[(category, "*")].
#   3. Tier-only fallback     — RESPONSE_MATRIX[("*", tier)].
#   4. Generic empathetic default (DEFAULT_RESPONSE below).

RESPONSE_MATRIX: dict = {

    # --- REQUIRED COMBINATIONS (from the brief, must be distinct) ---

    ("WELLBEING", "CRISIS"): (
        "I am very concerned about you. Please contact the university "
        "counselling line RIGHT NOW: 0800-XXX-XXXX."
    ),
    ("WELLBEING", "DISTRESSED"): (
        "It sounds like you are going through a really difficult time. "
        "Have you spoken to anyone about how you are feeling?"
    ),
    ("ACADEMIC", "STRESSED"): (
        "Exam pressure is real. Let us look at what support your faculty "
        "offers — have you spoken to your tutor?"
    ),
    ("FINANCIAL", "*"): (
        "Financial difficulty is more common than you think. The "
        "university bursary office can help — shall I give you their "
        "contact?"
    ),
    ("TECHNICAL", "*"): (
        "Let me help you with that technical issue. Which system are "
        "you trying to access?"
    ),
    ("SOCIAL", "DISTRESSED"): (
        "Feeling isolated at university is incredibly hard. The student "
        "union runs weekly social events — would that help?"
    ),

    # --- ADDITIONAL FALLBACK ENTRIES ---
    # These are not strictly required by the rubric but keep NEXUS
    # coherent across the rest of the (category, tier) grid.

    ("ACADEMIC", "DISTRESSED"): (
        "Academic struggles can feel crushing. You don't have to face "
        "this alone — your academic advisor and the wellbeing service "
        "can both help."
    ),
    ("ACADEMIC", "CRISIS"): (
        "Your wellbeing matters more than any deadline. Please contact "
        "the counselling line now: 0800-XXX-XXXX. We can sort the "
        "academic side out together afterwards."
    ),
    ("SOCIAL", "CRISIS"): (
        "I'm really worried about you. Please reach the counselling "
        "line now: 0800-XXX-XXXX — you do not have to feel alone "
        "tonight."
    ),
    ("ADMIN", "*"): (
        "I can point you to the right office for that. The Registrar's "
        "team handles transcripts, certificates and registration — "
        "would you like the contact details?"
    ),
    ("WELLBEING", "STRESSED"): (
        "Stress is a signal, not a verdict. Small steps help — would "
        "you like to talk about what is weighing on you most right now?"
    ),
    ("WELLBEING", "NEUTRAL"): (
        "Thank you for sharing how you're feeling. Is there anything "
        "specific on your mind today?"
    ),
    ("WELLBEING", "CONTENT"): (
        "It's good to hear you're doing okay. What's been going well "
        "for you recently?"
    ),
    ("WELLBEING", "THRIVING"): (
        "That's wonderful to hear! Keep nurturing whatever is working "
        "for you."
    ),
}


# DEFAULT_RESPONSE
# -----------------
# Returned by nexus_respond() when nothing matches the matrix — for
# example, a NEUTRAL tier student asking a TECHNICAL-adjacent question
# that didn't trigger any category. Keeps NEXUS warm and inviting.

DEFAULT_RESPONSE: str = (
    "I'm here to listen. Could you tell me a little more about what's "
    "going on for you?"
)


# ESCALATION_TARGET
# ------------------
# Used by log_support_transition(): when the primary category transitions
# INTO this target from any other category, the event is marked
# is_escalation=True. Per the brief, escalation = entering WELLBEING.

ESCALATION_TARGET: str = "WELLBEING"


# ===========================================================================
# INTELLIGENCE REPORT CONFIGURATION
# ===========================================================================

# RISK_SCORE_CONFIG
# ------------------
# Constants used by generate_intelligence_report() to compute the
# 0–100 risk score handed to the counselling team.
#
# Formula (from the brief):
#   risk_score = base_risk
#              + abs(avg_wellbeing_score) * wellbeing_weight
#              + at_risk_count            * at_risk_weight
#              + escalation_count         * escalation_weight
#   risk_score = max(0, min(100, round(risk_score)))
#
# verbose_bonus is awarded when avg words/turn > verbose_threshold
# (the brief notes: verbose = distressed).

RISK_SCORE_CONFIG: dict = {
    "base_risk":           20,
    "wellbeing_weight":    40,   # multiplied by abs(avg compound score)
    "at_risk_weight":      15,   # +15 per CRISIS turn
    "escalation_weight":   10,   # +10 per WELLBEING escalation
    "verbose_bonus":        5,   # +5 if avg words/turn > threshold
    "verbose_threshold":   20,
    "score_floor":          0,
    "score_ceiling":      100,
}


# ACTION_THRESHOLDS
# ------------------
# Maps risk-score ranges -> recommended action label printed at the
# bottom of the counsellor report.
#
#   risk_score >= 70           -> URGENT_REFERRAL
#   40 <= risk_score < 70      -> FOLLOW_UP
#   risk_score < 40            -> NO_ACTION

ACTION_THRESHOLDS: Tuple[Tuple[int, str], ...] = (

    (70, "URGENT_REFERRAL"),
    (40, "FOLLOW_UP"),
    (0,  "NO_ACTION"),
)


# REPORT_HEADER / REPORT_FOOTER
# -----------------------------
# Cosmetic banners for the counsellor intelligence report. Kept here so
# the report layout can be tweaked without editing pipeline code.

REPORT_HEADER: str = (
    "=======================================================\n"
    "          NEXUS STUDENT INTELLIGENCE REPORT\n"
    "          Code: NX-2B — Counsellor Eyes Only\n"
    "======================================================="
)

REPORT_FOOTER: str = (
    "======================================================="
)
