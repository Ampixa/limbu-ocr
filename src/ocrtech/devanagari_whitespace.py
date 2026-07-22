"""Conservative cleanup for Devanagari converter whitespace artifacts."""

from __future__ import annotations

import unicodedata


DEVANAGARI_VIRAMA = "\u094d"
DEVANAGARI_CONSONANT_RANGES = (
    ("\u0915", "\u0939"),
    ("\u0958", "\u095f"),
    ("\u0978", "\u097f"),
)


def _is_devanagari_consonant(char: str) -> bool:
    return any(start <= char <= end for start, end in DEVANAGARI_CONSONANT_RANGES)


def normalize_devanagari_whitespace(text: str) -> str:
    """Remove only known converter-artifact runs of literal spaces.

    A space run is removed when it is either directly between a Devanagari
    virama and consonant, or directly before any Unicode nonspacing mark.
    All other characters and whitespace are preserved verbatim.
    """

    normalized: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != " ":
            normalized.append(text[index])
            index += 1
            continue

        run_end = index + 1
        while run_end < len(text) and text[run_end] == " ":
            run_end += 1

        previous = text[index - 1] if index else ""
        following = text[run_end] if run_end < len(text) else ""
        after_virama_before_consonant = previous == DEVANAGARI_VIRAMA and bool(following) and _is_devanagari_consonant(following)
        before_combining_mark = bool(following) and unicodedata.category(following) == "Mn"

        if not (after_virama_before_consonant or before_combining_mark):
            normalized.append(text[index:run_end])
        index = run_end

    return "".join(normalized)
