"""Unicode normalization and dictionary coverage helpers."""

from __future__ import annotations

import string
import unicodedata


DEVANAGARI_START = "\u0900"
DEVANAGARI_END = "\u097f"
DEVANAGARI_EXTENDED_START = "\ua8e0"
DEVANAGARI_EXTENDED_END = "\ua8ff"
PRESERVED_FORMAT_CHARS = {"\u200c", "\u200d"}
ASCII_PRINTABLE = set(string.printable)


def is_devanagari(char: str) -> bool:
    return DEVANAGARI_START <= char <= DEVANAGARI_END or DEVANAGARI_EXTENDED_START <= char <= DEVANAGARI_EXTENDED_END


def normalize_ocr_text(text: str, *, collapse_spaces: bool = False) -> str:
    """Normalize OCR text while preserving Devanagari combining marks."""

    normalized = unicodedata.normalize("NFC", text.replace("\u00a0", " "))
    kept: list[str] = []
    previous_space = False
    for char in normalized:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"} and char not in {"\n", "\t", "\r"} | PRESERVED_FORMAT_CHARS:
            continue
        if collapse_spaces and char in {" ", "\t"}:
            if previous_space:
                continue
            char = " "
            previous_space = True
        else:
            previous_space = char == " "
        kept.append(char)
    return "".join(kept)


def script_counts(text: str) -> dict[str, int]:
    counts = {"devanagari": 0, "latin": 0, "digit": 0, "punctuation": 0, "other": 0}
    for char in normalize_ocr_text(text):
        if char.isspace():
            continue
        if is_devanagari(char):
            counts["devanagari"] += 1
        elif "LATIN" in unicodedata.name(char, ""):
            counts["latin"] += 1
        elif char.isdigit():
            counts["digit"] += 1
        elif unicodedata.category(char).startswith("P") or char in string.punctuation:
            counts["punctuation"] += 1
        else:
            counts["other"] += 1
    return counts


def default_allowed_characters() -> set[str]:
    chars = set(ASCII_PRINTABLE)
    for codepoint in range(ord(DEVANAGARI_START), ord(DEVANAGARI_END) + 1):
        chars.add(chr(codepoint))
    for codepoint in range(ord(DEVANAGARI_EXTENDED_START), ord(DEVANAGARI_EXTENDED_END) + 1):
        chars.add(chr(codepoint))
    chars.update(PRESERVED_FORMAT_CHARS)
    chars.update({"।", "॥", "–", "—", "‘", "’", "“", "”", "…", "₹", "₨"})
    return chars


def unsupported_characters(text: str, allowed: set[str] | None = None) -> list[str]:
    allowed_chars = allowed or default_allowed_characters()
    unsupported = sorted({char for char in normalize_ocr_text(text) if char not in allowed_chars and not char.isspace()})
    return unsupported


def dictionary_from_texts(texts: list[str], *, include_ascii: bool = True) -> list[str]:
    chars: set[str] = set()
    if include_ascii:
        chars.update(ch for ch in string.printable if ch not in {"\x0b", "\x0c", "\r"})
    for text in texts:
        chars.update(char for char in normalize_ocr_text(text) if char not in {"\n", "\r", "\t"})
    return sorted(chars)
