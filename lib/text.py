"""Normalize typographic punctuation and podcast branding text."""

import re

from lib.names import fix_names

CANONICAL_LISTNR_PRODUCTION = "LiSTNR production"

# Whisper mishears the podcast network intro in many ways.
_LISTNR_PRODUCTION_PATTERN = re.compile(
    r"\b(A\s+)?"
    r"(?:"
    r"list[-\s]*n[-\s]*a(?:[-\s]*production)?|"
    r"list[-\s]*(?:snuff|snap|nuff)(?:[-\s]*production)?|"
    r"list\s+nap(?:\s+production)?|"
    r"list\s+of\s+our(?:\s+production)?"
    r")\b",
    re.IGNORECASE,
)

_LISTENER_PRODUCTION_PATTERN = re.compile(
    r"\bA\s+listener\s+production\b",
    re.IGNORECASE,
)

# Common Whisper mishearings of hiking and regional terms.
# More specific patterns must appear before shorter/overlapping ones.
_TERM_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bTaken\s+A\s+Hike\b", re.IGNORECASE), "Take A Hike"),
    (re.compile(r"\bMisty\s+Mermans\b", re.IGNORECASE), "Misty Mountains"),
    (re.compile(r"\bRarunanan\s+National\s+Park\b", re.IGNORECASE), "Wooroonooran National Park"),
    (re.compile(r"\bMitcher\s+Creek\b", re.IGNORECASE), "Mitcha Creek"),
    (re.compile(r"\bTupan\s+Falls\b", re.IGNORECASE), "Tipun Falls"),
    (re.compile(r"\bNadroia\s+Falls\b", re.IGNORECASE), "Nandroya Falls"),
    (re.compile(r"\bChunder\s+Rock\b", re.IGNORECASE), "Chunda Rock"),
    (re.compile(r"\bChunder\s+Bay\b", re.IGNORECASE), "Chunda Bay"),
    (re.compile(r"\bGireme\s+National\s+Park\b", re.IGNORECASE), "Girramay National Park"),
    (re.compile(r"\bWilfrid\s+Karnal\b", re.IGNORECASE), "Wilfred Karnoll"),
    (re.compile(r"\bno-seum\s+mesh\b", re.IGNORECASE), "No-see-um mesh"),
    (re.compile(r"\bMount\s+Halifax\s+track,\s+George\b", re.IGNORECASE), "Mount Halifax track, Gorge"),
    (re.compile(r"\bHuendon\b", re.IGNORECASE), "Hughenden"),
    (re.compile(r"\bHuindon\b", re.IGNORECASE), "Hughenden"),
    (re.compile(r"\bHewnan\b", re.IGNORECASE), "Hughenden"),
    (re.compile(r"\bwonderstories\.space\b", re.IGNORECASE), "wanderstories.space"),
    (re.compile(r"\bWonder\s+Stories\b", re.IGNORECASE), "Wanderstories"),
    (re.compile(r"\bWander\s+Stories\b", re.IGNORECASE), "Wanderstories"),
    (re.compile(r"\bwonderstories\b", re.IGNORECASE), "wanderstories"),
    (re.compile(r"\bKillie\s+Moon\s+Creek\b", re.IGNORECASE), "Killymoon Creek"),
    (re.compile(r"\bKilly\s+Moon\s+Creek\b", re.IGNORECASE), "Killymoon Creek"),
    (re.compile(r"\bKillie\s+Moon\b", re.IGNORECASE), "Killymoon"),
    (re.compile(r"\bKilly\s+Moon\b", re.IGNORECASE), "Killymoon"),
    (re.compile(r"\bPlumer\s+Dam\b", re.IGNORECASE), "Paluma Dam"),
    (re.compile(r"\bPadrama\s+Falls\b", re.IGNORECASE), "Jourama Falls"),
    (re.compile(r"\bJarama\s+Falls\b", re.IGNORECASE), "Jourama Falls"),
    (re.compile(r"\bJarama\b", re.IGNORECASE), "Jourama"),
    (re.compile(r"\bDharama\s+Falls\b", re.IGNORECASE), "Jourama Falls"),
    (re.compile(r"\bDyerite\s+Falls\b", re.IGNORECASE), "Diorite Falls"),
    (re.compile(r"\bGuzuazu\s+Falls\b", re.IGNORECASE), "Iguazu Falls"),
    (re.compile(r"\bWallerman\s+Falls\b", re.IGNORECASE), "Wallaman Falls"),
    (re.compile(r"\bWolloman\s+Falls\b", re.IGNORECASE), "Wallaman Falls"),
    (re.compile(r"\bWalliman\s+Falls\b", re.IGNORECASE), "Wallaman Falls"),
    (re.compile(r"\bMagnet\s+Island\b", re.IGNORECASE), "Magnetic Island"),
    (re.compile(r"\bMy\s+County\s+Toyota\b", re.IGNORECASE), "Mike Carney Toyota"),
    (re.compile(r"\bThorsbourne\s+Trail\b", re.IGNORECASE), "Thorsborne Trail"),
    (re.compile(r"\bThor['']s\s+Born\s+Trail\b", re.IGNORECASE), "Thorsborne Trail"),
    (re.compile(r"\bMid\s+Trooper\s+Falls\b", re.IGNORECASE), "Moochoopa Falls"),
    (re.compile(r"\bMatrupa\b", re.IGNORECASE), "Moochoopa"),
    (re.compile(r"\bFive\s+Inch\s+Bay\b", re.IGNORECASE), "Five Beach Bay"),
    (re.compile(r"\bright\s+here\s+on\s+listener\b", re.IGNORECASE), "right here on LiSTNR"),
    (re.compile(r"\bMount\s+Stewart\b", re.IGNORECASE), "Mount Stuart"),
    (re.compile(r"\bMount\s+Peterbott\b", re.IGNORECASE), "Mount Pieter Botte"),
    (re.compile(r"\bMount\s+Elliott\b", re.IGNORECASE), "Mount Elliot"),
    (re.compile(r"\bLake\s+Tinneroo\b", re.IGNORECASE), "Lake Tinaroo"),
    (re.compile(r"\bMingler\s+Range\b", re.IGNORECASE), "Mingela Range"),
    (re.compile(r"\bEast\s+Sinafa\s+Range\b", re.IGNORECASE), "Eastern Arthur Range"),
    (re.compile(r"\bEast\s+Sinafa'?s\b", re.IGNORECASE), "Eastern Arthur Range"),
    (re.compile(r"\bMangatree\s+Car\s+Park\b", re.IGNORECASE), "Mango Tree Car Park"),
    (re.compile(r"\bMitch\s+Nissen\b", re.IGNORECASE), "Mitch Nissan"),
    (re.compile(r"\bGod\s+Zone\b", re.IGNORECASE), "Godzone"),
    (re.compile(r"\bGringan\s+National\s+Park\b", re.IGNORECASE), "Girringun National Park"),
    (re.compile(r"\bGringan\b", re.IGNORECASE), "Grin and Bear"),
    (re.compile(r"\bPalooma\b", re.IGNORECASE), "Paluma"),
    (re.compile(r"\bDinema\b", re.IGNORECASE), "Dyneema"),
    (re.compile(r"\bFarnworth\s+Queensland\b", re.IGNORECASE), "Far North Queensland"),
    (re.compile(r"\bfar\s+north\s+Queensland\b", re.IGNORECASE), "Far North Queensland"),
    (re.compile(r"\bpack\s+raft\b", re.IGNORECASE), "packraft"),
    (re.compile(r"\bKwondong\b", re.IGNORECASE), "Quandong"),
    (re.compile(r"\bIntentant\b", re.IGNORECASE), "Hint Hint Hint"),
    (re.compile(r"\bBartlefrier\b", re.IGNORECASE), "Bartle Frere"),
    (re.compile(r"\bBartle\s+Free\b", re.IGNORECASE), "Bartle Frere"),
    (re.compile(r"\bBartle\s+Freer\b", re.IGNORECASE), "Bartle Frere"),
    (re.compile(r"\bonedistory\.space\b", re.IGNORECASE), "wanderstories.space"),
    (re.compile(r"\bTazzy\b", re.IGNORECASE), "Tassie"),
    (re.compile(r"\bGroviral\b", re.IGNORECASE), "go viral"),
    (re.compile(r"\bPlowdy\s+Creek\b", re.IGNORECASE), "Cloudy Creek"),
    (re.compile(r"\bAthel\s+Creek\b", re.IGNORECASE), "Ethel Creek"),
    (re.compile(r"\bMangal\s+Tree\s+Car\s+Park\b", re.IGNORECASE), "Mango Tree Car Park"),
    (re.compile(r"\bMagno\s+Tree\s+Car\s+Park\b", re.IGNORECASE), "Mango Tree Car Park"),
    (re.compile(r"\bKilimune\s+Creek\b", re.IGNORECASE), "Killymoon Creek"),
    (re.compile(r"\bMingola\s+Range\b", re.IGNORECASE), "Mingela Range"),
    (re.compile(r"\bKranda\b", re.IGNORECASE), "Kuranda"),
    (re.compile(r"\bCrayander\b", re.IGNORECASE), "Kuranda"),
    (re.compile(r"\bTe\s+Adaroa\b", re.IGNORECASE), "Te Araroa"),
    (re.compile(r"\bzack-mark\b", re.IGNORECASE), "ZACH MACH"),
    (re.compile(r"\bgodzones\b", re.IGNORECASE), "Godzone"),
    (re.compile(r"\bnightcorp\b", re.IGNORECASE), "Nitecore"),
    (re.compile(r"\bnightcore\b", re.IGNORECASE), "Nitecore"),
    (re.compile(r"\bYongeboro\b", re.IGNORECASE), "Yungaburra"),
    (re.compile(r"\bHuntingfield\s+bath\b", re.IGNORECASE), "Huntingfield Bay"),
    (re.compile(r"\bHinch\s+and\s+Brook\b", re.IGNORECASE), "Hinchinbrook"),
    (re.compile(r"\bmord\s+bay\b", re.IGNORECASE), "Maud Bay"),
    (re.compile(r"\bLovers\s+bay\b", re.IGNORECASE), "Lovers Bay"),
    (re.compile(r"\bJoyce\s+bay\b", re.IGNORECASE), "Joyce Bay"),
    (re.compile(r"\bhunting\s+field\b", re.IGNORECASE), "Huntingfield"),
    (re.compile(r"\bBig\s+Bristol\s+Creek\b", re.IGNORECASE), "Big Crystal Creek"),
    (re.compile(r"\bPaloma\b", re.IGNORECASE), "Paluma"),
    (re.compile(r"\bPluma\b", re.IGNORECASE), "Paluma"),
    (re.compile(r"\bLake\s+Tineru\b", re.IGNORECASE), "Lake Tinaroo"),
    (re.compile(r"\bCasawaries\b", re.IGNORECASE), "Cassowaries"),
    (re.compile(r"\bCasawary\b", re.IGNORECASE), "Cassowary"),
    (re.compile(r"\bHelifax\b", re.IGNORECASE), "Halifax"),
    (re.compile(r"\bTimilabeech\b", re.IGNORECASE), "Toomulla Beach"),
    (re.compile(r"\bTamula\s+Beach\b", re.IGNORECASE), "Toomulla Beach"),
    (re.compile(r"\bTumala\b", re.IGNORECASE), "Toomulla"),
    (re.compile(r"\bSteat\s+Park\b", re.IGNORECASE), "State Park"),
    (re.compile(r"\bSteat\s+Forest\b", re.IGNORECASE), "State Forest"),
    (re.compile(r"\bunderstory\.space\b", re.IGNORECASE), "wanderstories.space"),
    (re.compile(r"\bunderstories\b", re.IGNORECASE), "Wanderstories"),
    (re.compile(r"\bWallamann\s+falls\b", re.IGNORECASE), "Wallaman Falls"),
    (re.compile(r"\bWallamann\b", re.IGNORECASE), "Wallaman"),
)

# Curly/smart quotes, dashes, and other common Unicode punctuation → ASCII.
_CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201a": "'",  # single low-9 quote
        "\u201b": "'",  # single high-reversed-9 quote
        "\u2032": "'",  # prime
        "\u2035": "'",  # reversed prime
        "\u02bc": "'",  # modifier letter apostrophe
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u201e": '"',  # double low-9 quote
        "\u2033": '"',  # double prime
        "\u00ab": '"',  # guillemet left
        "\u00bb": '"',  # guillemet right
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00a0": " ",  # non-breaking space
        "\u2009": " ",  # thin space
        "\u202f": " ",  # narrow no-break space
        "\u200b": "",   # zero-width space
        "\ufeff": "",   # byte order mark
    }
)

TEXT_PROMPT_NOTE = (
    "Use plain ASCII punctuation only: straight apostrophes ('), straight quotes (\"), "
    "hyphens (-), and three dots (...) - not curly quotes, em dashes, or ellipsis characters."
)


def normalize_text(text: str) -> str:
    """Replace typographic punctuation with standard ASCII characters."""
    if not text:
        return text
    return text.translate(_CHAR_REPLACEMENTS).replace("\u2026", "...")


def fix_terms(text: str) -> str:
    """Correct common Whisper mishearings of hiking and place names."""
    if not text:
        return text
    for pattern, replacement in _TERM_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def fix_branding(text: str) -> str:
    """Correct misheard LiSTNR production intro lines from Whisper."""

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        return f"{prefix}{CANONICAL_LISTNR_PRODUCTION}"

    if not text:
        return text
    text = _LISTENER_PRODUCTION_PATTERN.sub(CANONICAL_LISTNR_PRODUCTION, text)
    return _LISTNR_PRODUCTION_PATTERN.sub(_replace, text)


def clean_text(text: str) -> str:
    """Normalize punctuation, branding, terms, then fix known name misspellings."""
    return fix_names(fix_terms(fix_branding(normalize_text(text))))
