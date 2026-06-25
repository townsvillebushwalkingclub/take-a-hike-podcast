"""Correct common misspellings of podcast host names."""

import re

CANONICAL_FULL_NAME = "Luen Warneke"
CANONICAL_FIRST_NAME = "Luen"
CHERRY_JUDGE_FULL_NAME = "Cherry Judge"
BLAIR_WOODCOCK_FIRST_NAME = "Blair"

# Host first name: Blaise -> Blair.
_BLAISE_PATTERN = re.compile(r"\bBlaise\b", re.IGNORECASE)

# Mentioned name: Mikkel -> Mickle.
_MIKKEL_POSSESSIVE_PATTERN = re.compile(r"\bMikkel's\b", re.IGNORECASE)
_MIKKEL_PATTERN = re.compile(r"\bMikkel\b", re.IGNORECASE)

# Mentioned name: Wilfrid Karnal -> Wilfred Karnoll.
WILFRED_KARNOLL_FULL_NAME = "Wilfred Karnoll"
_WILFRID_KARNAL_PATTERN = re.compile(r"\bWilfrid\s+Karnal\b", re.IGNORECASE)
_WILFRID_POSSESSIVE_PATTERN = re.compile(r"\bWilfrid['']s\b", re.IGNORECASE)

# "Luen, you're..." misheard as a surname.
_LUEN_YOURE_PATTERN = re.compile(r"\bLuen\s+Yorke\b", re.IGNORECASE)

# Full name: misheard first name + optional "and" + misheard surname.
_FULL_NAME_PATTERN = re.compile(
    r"\b(?:Llewyn|Lewyn|Lewin|Luan|Luen|Lil)\s+(?:and\s+)?"
    r"(?:Warnakie|Warneke|Warnecke|Warnakey|Warnocky|Warnicky|Warrneke|Wernake|Warrnake)\b",
    re.IGNORECASE,
)

# "Cherry Lewin" / "Cherry Lewyn" when Luen is addressed alongside Cherry.
_CHERRY_AND_LUEN_PATTERN = re.compile(
    r"\b[Cc]herry\s+(?:Llewyn|Lewyn|Lewin|Luan|Lil)\b",
    re.IGNORECASE,
)

# Always title-case the host name, including double spaces from transcripts.
_CHERRY_JUDGE_PATTERN = re.compile(r"\b[Cc]herry\s+[Jj]udge\b", re.IGNORECASE)

# Possessive first-name misspellings: "Lewyn's world" -> "Luen's world".
_POSSESSIVE_FIRST_NAME_PATTERN = re.compile(
    r"\b(?:Llewyn|Lewyn|Lewin|Luan|Lil)'s\b",
    re.IGNORECASE,
)

# "Lil and thought..." when Whisper splits the name from the surname.
_LIL_AND_PATTERN = re.compile(r"\bLil\s+and\b", re.IGNORECASE)

# Standalone first-name misspellings after full names are corrected.
_FIRST_NAME_PATTERN = re.compile(
    r"\b(?:Llewyn|Lewyn|Lewin|Luan|Lil|Llewit|Lorne|Luwin|Loon|Lohan)\b",
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

    corrected = _LUEN_YOURE_PATTERN.sub("Luen you're", text)
    corrected = _BLAISE_PATTERN.sub(BLAIR_WOODCOCK_FIRST_NAME, corrected)
    corrected = _MIKKEL_POSSESSIVE_PATTERN.sub("Mickle's", corrected)
    corrected = _MIKKEL_PATTERN.sub("Mickle", corrected)
    corrected = _WILFRID_KARNAL_PATTERN.sub(WILFRED_KARNOLL_FULL_NAME, corrected)
    corrected = _WILFRID_POSSESSIVE_PATTERN.sub("Wilfred's", corrected)
    corrected = _FULL_NAME_PATTERN.sub(CANONICAL_FULL_NAME, corrected)
    corrected = _CHERRY_AND_LUEN_PATTERN.sub(f"Cherry, {CANONICAL_FIRST_NAME}", corrected)
    corrected = _CHERRY_JUDGE_PATTERN.sub(CHERRY_JUDGE_FULL_NAME, corrected)
    corrected = _POSSESSIVE_FIRST_NAME_PATTERN.sub(f"{CANONICAL_FIRST_NAME}'s", corrected)
    corrected = _LIL_AND_PATTERN.sub(CANONICAL_FIRST_NAME, corrected)
    corrected = _FIRST_NAME_PATTERN.sub(CANONICAL_FIRST_NAME, corrected)
    return corrected
