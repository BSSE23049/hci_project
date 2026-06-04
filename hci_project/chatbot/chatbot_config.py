"""
Chatbot app configuration — ALL flags, constants, keyword lists, intent definitions,
NEXUS wellbeing scale, response rules, and risk thresholds live here.

To change any behaviour: edit this file only.
"""

# ---------------------------------------------------------------------------
# Top-level mode switch
# ---------------------------------------------------------------------------
CHATBOT_MODE       = "university"   # "university" | "nexus"
DEFAULT_INPUT_MODE = "hybrid"       # "text" | "voice" | "hybrid"

# ---------------------------------------------------------------------------
# Menu feature flags
# Setting any flag to False hides that option from the main menu entirely
# and re-numbers the remaining options automatically. No other file changes.
# ---------------------------------------------------------------------------
ENABLE_UNIVERSITY_MODE  = True   # show / hide University Chatbot option
ENABLE_NEXUS_MODE       = True   # show / hide NEXUS Wellbeing Advisor option
ENABLE_MODE_SWITCH      = True   # show / hide "Switch mode" menu item
ENABLE_INPUT_SWITCH     = True   # show / hide "Switch input" menu item
ENABLE_OFFLINE_REPLAY   = True   # show / hide offline NEXUS session replay option

# ---------------------------------------------------------------------------
# LLM flags
# ---------------------------------------------------------------------------
USE_LLM         = True
LLM_PROVIDER    = "ollama"
OLLAMA_MODEL    = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Whisper / audio flags
# ---------------------------------------------------------------------------
WHISPER_MODEL        = "base"
AUDIO_RECORD_SECONDS = 7
AUDIO_SAMPLE_RATE    = 16000

# ---------------------------------------------------------------------------
# Intent classification mode
# "rule_based" — keyword matching only (fast, offline, no LLM needed)
# "ai"         — LLM classifies intent; falls back to rule_based if LLM is down
# ---------------------------------------------------------------------------
INTENT_CLASSIFICATION_MODE = "rule_based"   # "rule_based" | "ai"

# ---------------------------------------------------------------------------
# University chatbot — intent dictionary
# TO ADD A NEW INTENT: add one entry here. Nothing else changes.
# ---------------------------------------------------------------------------
UNIVERSITY_INTENTS = {
    "Admission": {
        "keywords": [
            "admission", "apply", "application", "enroll", "enrollment",
            "register", "registration", "entry", "intake", "join", "accept",
            "accepted", "criteria", "requirement", "eligibility",
        ],
        "response_static": (
            "For admissions, please visit the official university admissions portal "
            "or contact the Admissions Office directly. Requirements vary by programme. "
            "The typical intake periods are January and September each year."
        ),
    },
    "Fee": {
        "keywords": [
            "fee", "fees", "charges", "tuition", "cost", "payment", "pay",
            "invoice", "bursary", "scholarship", "financial", "fund", "funding",
            "loan", "money", "price", "rate", "afford", "expensive",
        ],
        "response_static": (
            "Fee structures are published on the university finance page. "
            "Payment plans, bursaries, and student loans are available — "
            "please contact the Finance Office for personalised advice."
        ),
    },
    "Courses": {
        "keywords": [
            "course", "courses", "module", "modules", "subject", "subjects",
            "programme", "programs", "degree", "diploma", "undergraduate",
            "postgraduate", "masters", "phd", "curriculum", "syllabus", "unit",
            "credit", "credits", "class", "classes", "study", "major", "minor",
        ],
        "response_static": (
            "The full course catalogue is available on the university website. "
            "You can browse by faculty or use the course search tool. "
            "Your academic advisor can also help you select the right modules."
        ),
    },
    "Schedule": {
        "keywords": [
            "schedule", "timetable", "lecture", "lectures", "time", "times",
            "when", "calendar", "date", "dates", "semester", "term", "week",
            "exam", "exams", "assessment", "deadline", "due", "slot", "session",
            "morning", "afternoon", "evening", "online", "venue", "room",
        ],
        "response_static": (
            "Class timetables and exam schedules are published on the student portal. "
            "Log in with your student credentials to view your personalised timetable. "
            "Contact your faculty office for any scheduling queries."
        ),
    },
    "Library": {
        "keywords": [
            "library", "book", "books", "borrow", "return", "journal", "journals",
            "resource", "resources", "database", "databases", "research", "article",
            "reading", "reserve", "catalogue", "online resource", "e-book",
        ],
        "response_static": (
            "The university library is open Monday–Friday 08:00–22:00 and weekends 09:00–18:00. "
            "You can search for books and journals through the online catalogue. "
            "Students get 10 book loans and free access to all digital databases."
        ),
    },
    "Hostel": {
        "keywords": [
            "hostel", "accommodation", "residence", "dorm", "dormitory",
            "room", "housing", "live", "stay", "on-campus",
            "off-campus", "residential", "flat", "apartment",
        ],
        "response_static": (
            "On-campus accommodation applications open in March for the following academic year. "
            "Visit the Student Housing Office website to apply or check availability. "
            "Off-campus housing listings are also available on the student noticeboard."
        ),
    },
    "Transport": {
        "keywords": [
            "transport", "bus", "shuttle", "train", "taxi", "travel",
            "parking", "car", "commute", "route", "directions", "map",
            "how to get", "getting there", "navigate",
        ],
        "response_static": (
            "The university runs free shuttle buses on campus. Public bus routes 15 and 22 stop "
            "at the main gate. Student parking permits are available from Campus Security. "
            "Visit the transport page on the university website for maps and routes."
        ),
    },
}

# ---------------------------------------------------------------------------
# Spam and domain filtering
# ---------------------------------------------------------------------------
SPAM_KEYWORDS = [
    "buy now", "click here", "win a prize", "free money", "make money fast",
    "casino", "lottery", "bitcoin", "crypto invest", "earn online",
    "cheap pills", "discount viagra", "you have won", "limited offer",
    "act now", "urgent offer", "nude", "xxx", "porn", "hack", "crack",
    "cheat", "pirate", "warez", "keygen", "serial key",
]

DOMAIN_KEYWORDS = [
    "university", "college", "campus", "student", "students", "course",
    "lecture", "module", "programme", "degree", "exam", "assignment",
    "fee", "fees", "tuition", "cost", "price", "payment", "pay",
    "admission", "apply", "enroll", "register", "registration",
    "tutor", "faculty", "department", "library", "hostel", "dorm",
    "study", "academic", "semester", "result", "grade", "marks", "portal",
    "application", "scholarship", "bursary", "transport", "parking",
    "research", "thesis", "dissertation", "project", "class", "classroom",
    "schedule", "timetable", "lecture", "exam", "deadline", "room",
    "help", "support", "information", "info", "question", "ask",
    "wellbeing", "mental", "stress", "feeling", "health", "counsellor",
]

SPAM_BLOCKED_RESPONSE   = "This query has been classified as spam and cannot be processed."
UNKNOWN_INTENT_RESPONSE = (
    "Sorry, I could not understand your request. "
    "Please ask about admissions, fees, courses, schedules, the library, "
    "accommodation, or transport. Type 'exit' to quit."
)

# ---------------------------------------------------------------------------
# NEXUS wellbeing scale
# Each entry: (tier_name, low_inclusive, high_exclusive, emoji)
# Order matters — first match wins.
# ---------------------------------------------------------------------------
NEXUS_WELLBEING_SCALE = [
    ("THRIVING",    0.60,            float("inf"),   "🌟"),
    ("CONTENT",     0.20,            0.60,           "😊"),
    ("NEUTRAL",    -0.19,            0.20,           "😐"),
    ("STRESSED",   -0.40,           -0.19,           "😟"),
    ("DISTRESSED", -0.60,           -0.40,           "😢"),
    ("CRISIS",      float("-inf"),  -0.60,           "🆘"),
]

# ---------------------------------------------------------------------------
# NEXUS support keyword dictionary
# TO ADD A CATEGORY: add one entry. The classifier rebuilds automatically.
# ---------------------------------------------------------------------------
NEXUS_SUPPORT_KEYWORDS = {
    "ACADEMIC": [
        "assignment", "exam", "exams", "study", "fail", "failing", "grade",
        "grades", "marks", "deadline", "submit", "submission", "tutor",
        "lecturer", "project", "thesis", "dissertation", "plagiarism",
        "extension", "deferred", "module", "course", "class",
    ],
    "FINANCIAL": [
        "fee", "fees", "money", "debt", "loan", "bursary", "scholarship",
        "afford", "overdue", "invoice", "payment", "register", "financial",
        "broke", "poor", "fund", "funding", "sponsor", "cost", "expensive",
    ],
    "TECHNICAL": [
        "laptop", "computer", "wifi", "internet", "portal", "login",
        "password", "account", "access", "broken", "error", "crash",
        "software", "hardware", "printer", "email", "system", "network",
        "download", "upload", "file", "lost files",
    ],
    "SOCIAL": [
        "friend", "friends", "alone", "lonely", "bully", "bullying",
        "harassed", "harassment", "relationship", "roommate", "conflict",
        "argue", "argument", "social", "group", "club", "society",
        "introvert", "shy", "fitting in", "belonging",
    ],
    "ADMIN": [
        "admin", "administration", "registration", "enroll", "enrollment",
        "document", "certificate", "transcript", "letter", "office",
        "staff", "form", "application", "apply", "rule", "policy",
        "complaint", "appeal", "deadline", "withdrawal",
    ],
    "WELLBEING": [
        "hopeless", "depressed", "depression", "anxious", "anxiety",
        "lonely", "alone", "isolated", "overwhelmed",
        "stress", "mental health", "wellbeing", "sad", "crying", "hurt",
        "pain", "suffer", "struggling", "can't cope", "cannot cope",
        "suicidal", "suicide", "self-harm", "helpless", "worthless",
        "tired", "exhausted", "sleep", "insomnia", "panic",
    ],
}

# ---------------------------------------------------------------------------
# NEXUS response rules
# Format: (support_category, wellbeing_tier, response_text)
# Rules are checked top to bottom; first match wins.
# Use "*" as wildcard for category or tier.
# TO CHANGE A RESPONSE: edit the string. TO ADD A RULE: add a tuple.
# ---------------------------------------------------------------------------
NEXUS_RESPONSE_RULES = [
    (
        "WELLBEING", "CRISIS",
        "I am very concerned about you right now. Please contact the university counselling line "
        "IMMEDIATELY: 0800-XXX-XXXX. You do not have to face this alone — help is available "
        "right now, 24 hours a day."
    ),
    (
        "WELLBEING", "DISTRESSED",
        "It sounds like you are going through a really difficult time. "
        "Have you spoken to anyone about how you are feeling? "
        "The university counselling service offers free, confidential sessions — "
        "I can give you their contact details if you would like."
    ),
    (
        "WELLBEING", "STRESSED",
        "You sound stressed, which is very understandable. "
        "Many students feel this way, especially around deadlines. "
        "Would you like to talk about what is weighing on you most right now?"
    ),
    (
        "ACADEMIC", "CRISIS",
        "Academic pressure combined with how you are feeling right now is a serious concern. "
        "Please reach out to student support immediately: 0800-XXX-XXXX. "
        "Your wellbeing matters more than any assignment."
    ),
    (
        "ACADEMIC", "STRESSED",
        "Exam and assignment pressure is real and difficult. "
        "Let us look at what support your faculty offers — have you spoken to your tutor? "
        "Extensions and deferred assessments may be available."
    ),
    (
        "ACADEMIC", "DISTRESSED",
        "I can see you are really struggling academically right now. "
        "Please speak to your academic advisor or student support team — "
        "they can arrange extensions, tutoring, or other accommodations."
    ),
    (
        "FINANCIAL", "*",
        "Financial difficulty is more common among students than you might think. "
        "The university bursary and financial aid office can help — "
        "shall I give you their contact details? Emergency funds may also be available."
    ),
    (
        "TECHNICAL", "*",
        "I can help you with that technical issue. "
        "Which system are you trying to access? "
        "The IT helpdesk is also available at helpdesk@university.ac.uk or extension 1234."
    ),
    (
        "SOCIAL", "DISTRESSED",
        "Feeling isolated at university is incredibly hard. "
        "The student union runs weekly social events and there are many clubs and societies. "
        "Would connecting with others help? I can share some upcoming events with you."
    ),
    (
        "SOCIAL", "*",
        "Social wellbeing is just as important as academic success. "
        "The student union and campus community hubs are great places to meet people. "
        "Is there a specific social challenge I can help you with?"
    ),
    (
        "ADMIN", "*",
        "For administrative matters, the student services office can assist you directly. "
        "Most admin requests can also be submitted via the student portal. "
        "What specific document or process do you need help with?"
    ),
    (
        "*", "THRIVING",
        "It is wonderful to hear you are doing well! "
        "Is there anything specific I can help you with today?"
    ),
    (
        "*", "*",
        "Thank you for sharing that with me. I am here to help — "
        "can you tell me a little more about what you need? "
        "I want to make sure I point you in the right direction."
    ),
]

# ---------------------------------------------------------------------------
# Risk score thresholds (used by nexus_report.py)
# ---------------------------------------------------------------------------
NEXUS_RISK_URGENT    = 70
NEXUS_RISK_FOLLOW_UP = 40
NEXUS_CRISIS_LINE    = "0800-XXX-XXXX"
