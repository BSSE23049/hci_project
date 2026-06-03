"""
Spam detection module for the university chatbot.
Uses keyword matching and a domain relevance filter.
All keyword lists are imported from chatbot_config.py.
"""

from chatbot.chatbot_config import SPAM_KEYWORDS, DOMAIN_KEYWORDS


def is_spam(text: str) -> tuple:
    """
    Determine whether the input text is spam.

    Step 1 — Basic spam check: matches any phrase in SPAM_KEYWORDS.
    Step 2 — Domain filter: if step 1 passes, checks that at least one word
              from DOMAIN_KEYWORDS appears. If not, the text is off-topic spam.

    Parameters
    ----------
    text : str
        The raw user input string.

    Returns
    -------
    tuple[bool, str]
        (is_spam_flag, reason_string).
        is_spam_flag is True if the text is classified as spam.
        reason_string explains why it was flagged, or "" if clean.
    """
    lower = text.lower()

    # Step 1: explicit spam keyword match
    for phrase in SPAM_KEYWORDS:
        if phrase in lower:
            return True, f"matched spam keyword: '{phrase}'"

    # Step 2: domain relevance filter
    has_domain_word = any(word in lower for word in DOMAIN_KEYWORDS)
    if not has_domain_word:
        return True, "off-topic: no university context detected"

    return False, ""
