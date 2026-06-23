"""Normalize typographic punctuation to plain ASCII."""

from lib.names import fix_names

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


def clean_text(text: str) -> str:
    """Normalize punctuation, then fix known name misspellings."""
    return fix_names(normalize_text(text))
