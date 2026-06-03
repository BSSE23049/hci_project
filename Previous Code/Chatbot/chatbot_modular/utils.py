"""
chatbot_modular/utils.py
=========================
Shared utility functions used by multiple chatbot modules.

Currently contains one function: _kw_match(), a smarter keyword
matcher that avoids false-positive substring hits for short words.

Why is this in its own file?
  spam_detector.py, intent_classifier.py, and response_engine.py ALL
  need the same keyword matching logic.  Without a shared utility,
  we would copy-paste the same function three times -- making future
  bug fixes harder.  One file, one function, imported everywhere.
"""

import re


def _kw_match(kw: str, text: str) -> bool:
    """
    Test whether keyword ``kw`` appears in ``text``.

    The function uses TWO different strategies depending on keyword length:

    Strategy A -- Word-boundary regex  (for keywords <= 3 characters)
    ---------------------------------------------------------------
    Short keywords like "bs", "ms", "cs", "ai" are dangerous as plain
    substrings because they appear INSIDE longer words:

        "bs"  is found inside  "subscribe"   <- false positive!
        "ms"  is found inside  "semester"    <- false positive!
        "cs"  is found inside  "access"      <- false positive!

    Word boundaries (\\b in regex) anchor the match to word edges.
    So r'\\bbs\\b' matches the standalone word "bs" but NOT "subscribe".

    Example:
        _kw_match("bs",  "subscribe now")         -> False  correct
        _kw_match("bs",  "I study bs program")    -> True   correct
        _kw_match("ms",  "semester schedule")     -> False  correct
        _kw_match("ms",  "apply for ms program")  -> True   correct

    Strategy B -- Plain substring containment  (for keywords > 3 characters)
    -------------------------------------------------------------------------
    Longer keywords are specific enough that accidental substring hits
    are rare.  Using plain `in` is faster and also handles plurals:

        "scholarship" in "scholarships"  -> True   handles plural
        "course"      in "courses"       -> True   handles plural
        "exam"        in "examination"   -> True   handles word family

    This is the key reason INTENT matching correctly classifies:
        "Are any scholarships available?" -> FEE intent
    even though the keyword list has "scholarship" (singular).

    Args:
        kw:   Keyword string.  Must already be lowercased.
        text: Target text string.  Must already be lowercased.

    Returns:
        True  -- keyword found in text.
        False -- keyword not found in text.
    """
    # Short keywords: require word boundaries to prevent false matches
    if len(kw) <= 3:
        # re.search returns a Match object (truthy) or None (falsy)
        # re.escape handles any special regex chars in kw (e.g. if kw = "c++")
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))

    # Longer keywords: plain substring match (also covers plurals)
    return kw in text
