"""Correct common misspellings of podcast host names."""

import re

CANONICAL_FULL_NAME = "Luen Warneke"
CANONICAL_FIRST_NAME = "Luen"
CHERRY_JUDGE_FULL_NAME = "Cherry Judge"

# Full name: misheard first name + optional "and" + misheard surname.
_FULL_NAME_PATTERN = re.compile(
    r"\b(?:Lewyn|Lewin|Luan|Luen|Lil)\s+(?:and\s+)?"
    r"(?:Warnakie|Warneke|Warnecke|Warnocky|Warnicky|Warrneke|Wernake|Warrnake)\b",
    re.IGNORECASE,
)

# "Cherry Lewin" / "Cherry Lewyn" when Luen is addressed alongside Cherry.
_CHERRY_AND_LUEN_PATTERN = re.compile(
    r"\b[Cc]herry\s+(?:Lewyn|Lewin|Luan|Lil)\b",
    re.IGNORECASE,
)

# Always title-case the host name, including double spaces from transcripts.
_CHERRY_JUDGE_PATTERN = re.compile(r"\b[Cc]herry\s+[Jj]udge\b", re.IGNORECASE)

# Possessive first-name misspellings: "Lewyn's world" -> "Luen's world".
_POSSESSIVE_FIRST_NAME_PATTERN = re.compile(
    r"\b(?:Lewyn|Lewin|Luan|Lil)'s\b",
    re.IGNORECASE,
)

# "Lil and thought..." when Whisper splits the name from the surname.
_LIL_AND_PATTERN = re.compile(r"\bLil\s+and\b", re.IGNORECASE)

# Standalone first-name misspellings after full names are corrected.
_FIRST_NAME_PATTERN = re.compile(
    r"\b(?:Lewyn|Lewin|Luan|Lil|Loon|Lohan)\b",
    re.IGNORECASE,
)

NAME_PROMPT_NOTE = (
    f'Always spell guest names exactly as "{CANONICAL_FULL_NAME}" and '
    f'"{CHERRY_JUDGE_FULL_NAME}" (never "Lewyn Warnakie", "cherry judge", etc.).'
)


def fix_names(text: str) -> str:
    """Replace common transcript and AI misspellings with the correct name."""
    if not text:
        return text

    corrected = _FULL_NAME_PATTERN.sub(CANONICAL_FULL_NAME, text)
    corrected = _CHERRY_AND_LUEN_PATTERN.sub(f"Cherry, {CANONICAL_FIRST_NAME}", corrected)
    corrected = _CHERRY_JUDGE_PATTERN.sub(CHERRY_JUDGE_FULL_NAME, corrected)
    corrected = _POSSESSIVE_FIRST_NAME_PATTERN.sub(f"{CANONICAL_FIRST_NAME}'s", corrected)
    corrected = _LIL_AND_PATTERN.sub(CANONICAL_FIRST_NAME, corrected)
    corrected = _FIRST_NAME_PATTERN.sub(CANONICAL_FIRST_NAME, corrected)
    return corrected
