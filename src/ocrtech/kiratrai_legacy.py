"""Convert canonical-new and Sikkim Herald legacy Kirat Rai text to Unicode.

SIL publishes a TECkit mapping for the canonical 2021 ``kirat rai font new``
encoding. Sikkim Herald PDFs use a different, older layout hidden behind per-PDF
font subsets and ASCII extraction values. Four initial subsets plus the two
2024/2025 subsets audited in 2026 share one stable old-to-new remap derived by
exact glyph-outline command and advance-width identity. The Herald converter
applies that premap before the SIL rules.

That map is expressed with TECkit ``ByteClass``/``UniClass`` declarations plus a
handful of explicit multi-byte ligature rules. A ``[class] > [class]`` rule maps
the byte at position *n* of the byte class to the codepoint at position *n* of the
paired Unicode class. This module parses that map natively (no ``teckit_compile``
dependency, which is unavailable on this machine) and applies the forward
Legacy->Unicode pass.

The two layouts must not be auto-detected from text: even Herald strings without
the formerly conspicuous ``f R x F I L`` bytes are globally permuted. Callers must
select the canonical-new or Herald converter explicitly. The later pair adds one
exact match, extracted ``T`` -> canonical ``Z`` -> U+16D6C KIRAT RAI VOWEL SIGN
EU. It also establishes that extracted ``C`` has an empty outline and that the
custom glyph stored under ``:`` is a visible double quotation mark. These
edition-specific findings are pinned by PDF/font hashes in the converter-layout
audit; they are not inferred from ASCII character names.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataValidationError

DEFAULT_KIRATRAI_LEGACY_MAP = Path(__file__).with_name("maps") / "kiratraifontnew.map"

# Bytes absent from SIL's canonical-new map. They remain a compatibility constant;
# in Herald PDFs these ASCII values belong to the separate permuted layout below.
UNMAPPED_LEGACY_BYTES: frozenset[str] = frozenset("fRxFIL\\")

# Sikkim Herald extracted ASCII -> canonical ``kiratraifontnew`` byte. Derived by
# exact RecordingPen outline-command and advance-width identity against the font
# embedded in Unicode proposal L2/22-043R. Shared by four PDF subsets.
KIRATRAI_HERALD_PREMAP: dict[str, str] = {
    "D": "q",
    "F": "g",
    "G": "P",
    "H": "j",
    "I": "W",
    "J": "J",
    "K": "Q",
    "L": "$",
    "O": "O",
    "R": "w",
    "S": "G",
    "T": "Z",
    "U": "o",
    "a": "k",
    "b": "A",
    "c": "D",
    "d": "K",
    "e": "m",
    "f": "N",
    "g": "s",
    "h": "a",
    "i": "v",
    "j": "b",
    "k": "r",
    "l": "i",
    "m": "c",
    "n": "p",
    "o": "n",
    "p": "t",
    "q": "d",
    "r": "e",
    "s": "C",
    "t": "h",
    "u": "u",
    "v": "B",
    "w": "l",
    "x": "T",
    "y": "y",
    "z": "z",
}

# Confirmed non-ink glyphs in the audited Herald layout. The two 2024/2025 font
# subsets give C an empty RecordingPen command stream; backslash was already proven
# blank in the earlier four-subset corpus. They normalize to an ordinary space.
KIRATRAI_HERALD_BLANK_GLYPHS: frozenset[str] = frozenset("\\C")

# The old font stores a visible opening/closing double-quotation glyph under the
# extracted ASCII colon code. Original-resolution review of the 2024-09-26 PDF,
# page 3, includes bboxes (75.239998, 566.760010, 81.175995, 584.615967) and
# (211.019241, 566.760010, 216.955246, 584.615967). This is a punctuation
# correction, not a script-letter inference.
KIRATRAI_HERALD_PUNCTUATION_MAP: dict[str, str] = {":": '"'}

# Confirmed literal values in the audited Herald PDFs.
KIRATRAI_HERALD_PASSTHROUGH: frozenset[str] = frozenset(" \t\r\n0123456789(),-/.;")

_KIRATRAI_CODEPOINT_RE = re.compile(r"[\U00016D40-\U00016D7F]")
_BYTE_TOKEN_RE = re.compile(r"0x([0-9A-Fa-f]{2})")
_UNI_TOKEN_RE = re.compile(r"U\+([0-9A-Fa-f]{4,6})")
_BYTECLASS_RE = re.compile(r"^ByteClass\s*\[([^\]]+)\]\s*=\s*\((.*)\)\s*$")
_UNICLASS_RE = re.compile(r"^UniClass\s*\[([^\]]+)\]\s*=\s*\((.*)\)\s*$")
_CLASS_RULE_RE = re.compile(r"^\[([^\]]+)\]\s*<?>\s*\[([^\]]+)\]\s*$")


def _expand_byte_tokens(body: str) -> tuple[int, ...]:
    """Expand a TECkit byte-class body into an ordered list of byte values.

    Supports both explicit ``0xNN`` tokens and ``0xNN .. 0xMM`` ranges.
    """
    values: list[int] = []
    tokens = re.split(r"\s+", body.strip())
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token:
            index += 1
            continue
        if index + 2 < len(tokens) and tokens[index + 1] == "..":
            start = int(token, 16)
            end = int(tokens[index + 2], 16)
            if end < start:
                raise DataValidationError(f"invalid byte range in map: {token}..{tokens[index + 2]}")
            values.extend(range(start, end + 1))
            index += 3
            continue
        values.append(int(token, 16))
        index += 1
    return tuple(values)


def _expand_uni_tokens(body: str) -> tuple[int, ...]:
    """Expand a TECkit Unicode-class body into an ordered list of codepoints."""
    values: list[int] = []
    tokens = re.split(r"\s+", body.strip())
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token:
            index += 1
            continue
        if index + 2 < len(tokens) and tokens[index + 1] == "..":
            start = int(_UNI_TOKEN_RE.match(token).group(1), 16)
            end = int(_UNI_TOKEN_RE.match(tokens[index + 2]).group(1), 16)
            if end < start:
                raise DataValidationError(f"invalid unicode range in map: {token}..{tokens[index + 2]}")
            values.extend(range(start, end + 1))
            index += 3
            continue
        match = _UNI_TOKEN_RE.match(token)
        if match is None:
            raise DataValidationError(f"unparseable unicode token in map: {token!r}")
        values.append(int(match.group(1), 16))
        index += 1
    return tuple(values)


@dataclass(frozen=True, slots=True)
class KiratRaiLegacyConversion:
    legacy_text: str
    unicode_text: str
    kiratrai_char_count: int
    replacement_count: int
    unmapped_codepoints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_text": self.legacy_text,
            "unicode_text": self.unicode_text,
            "kiratrai_char_count": self.kiratrai_char_count,
            "replacement_count": self.replacement_count,
            "unmapped_codepoints": self.unmapped_codepoints,
        }


class KiratRaiLegacyConverter:
    """Native reader/applier for SIL's ``kiratraifontnew.map`` TECkit mapping.

    Rules are flattened to ``(source_bytes, target_codepoints)`` pairs and applied
    longest-match-first so that explicit multi-byte ligature rules (e.g. ``//`` ->
    double danda) take precedence over the single-byte class rules.
    """

    def __init__(self, rules: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
        if not rules:
            raise DataValidationError("Kirat Rai legacy converter requires at least one mapping rule")
        # De-duplicate while preserving the longest-source-first ordering.
        seen: set[tuple[int, ...]] = set()
        ordered: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for source, target in sorted(rules, key=lambda item: len(item[0]), reverse=True):
            if source in seen:
                continue
            seen.add(source)
            ordered.append((source, target))
        self._rules = ordered

    @classmethod
    def from_map_file(cls, path: str | Path = DEFAULT_KIRATRAI_LEGACY_MAP) -> "KiratRaiLegacyConverter":
        map_path = Path(path)
        if not map_path.is_file():
            raise DataValidationError(f"Kirat Rai legacy map does not exist: {map_path}")
        byte_classes: dict[str, tuple[int, ...]] = {}
        uni_classes: dict[str, tuple[int, ...]] = {}
        rules: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        in_byte_pass = False
        # First parse class declarations; collect rule lines for a second pass so
        # class rules can reference classes regardless of declaration order.
        rule_lines: list[str] = []
        for raw_line in map_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("pass("):
                in_byte_pass = lowered == "pass(byte_unicode)"
                continue
            if not in_byte_pass:
                continue
            byte_match = _BYTECLASS_RE.match(line)
            if byte_match:
                byte_classes[byte_match.group(1).strip()] = _expand_byte_tokens(byte_match.group(2))
                continue
            uni_match = _UNICLASS_RE.match(line)
            if uni_match:
                uni_classes[uni_match.group(1).strip()] = _expand_uni_tokens(uni_match.group(2))
                continue
            rule_lines.append(line)

        for line in rule_lines:
            class_rule = _CLASS_RULE_RE.match(line)
            if class_rule:
                left_name = class_rule.group(1).strip()
                right_name = class_rule.group(2).strip()
                left = byte_classes.get(left_name)
                right = uni_classes.get(right_name)
                if left is None:
                    raise DataValidationError(f"class rule references unknown byte class: {left_name!r}")
                if right is None:
                    raise DataValidationError(f"class rule references unknown unicode class: {right_name!r}")
                if len(left) != len(right):
                    raise DataValidationError(
                        f"class rule length mismatch for [{left_name}]>[{right_name}]: {len(left)} bytes vs {len(right)} codepoints"
                    )
                for byte_value, codepoint in zip(left, right):
                    rules.append(((byte_value,), (codepoint,)))
                continue
            if "<>" in line or ">" in line:
                left_text, right_text = line.split("<>", 1) if "<>" in line else line.split(">", 1)
                source = _BYTE_TOKEN_RE.findall(left_text)
                target = _UNI_TOKEN_RE.findall(right_text)
                if source and target:
                    rules.append(
                        (
                            tuple(int(value, 16) for value in source),
                            tuple(int(value, 16) for value in target),
                        )
                    )
        return cls(rules)

    def convert(self, text: str) -> KiratRaiLegacyConversion:
        output: list[str] = []
        unmapped: list[str] = []
        replacements = 0
        index = 0
        length = len(text)
        while index < length:
            matched = False
            for source, target in self._rules:
                if self._matches(text, index, source):
                    output.extend(chr(value) for value in target)
                    replacements += 1
                    index += len(source)
                    matched = True
                    break
            if matched:
                continue
            char = text[index]
            output.append(char)
            code = ord(char)
            if not (0x16D40 <= code <= 0x16D7F):
                unmapped.append(f"U+{code:04X}")
            index += 1
        converted = unicodedata.normalize("NFC", "".join(output))
        return KiratRaiLegacyConversion(
            legacy_text=text,
            unicode_text=converted,
            kiratrai_char_count=len(_KIRATRAI_CODEPOINT_RE.findall(converted)),
            replacement_count=replacements,
            unmapped_codepoints=sorted(set(unmapped)),
        )

    @staticmethod
    def _matches(text: str, index: int, source: tuple[int, ...]) -> bool:
        if index + len(source) > len(text):
            return False
        return all(ord(text[index + offset]) == code for offset, code in enumerate(source))


class KiratRaiHeraldConverter:
    """Convert the permuted Sikkim Herald layout through the canonical SIL map."""

    def __init__(self, canonical: KiratRaiLegacyConverter) -> None:
        self._canonical = canonical

    @classmethod
    def from_map_file(cls, path: str | Path = DEFAULT_KIRATRAI_LEGACY_MAP) -> "KiratRaiHeraldConverter":
        return cls(KiratRaiLegacyConverter.from_map_file(path))

    def convert(self, text: str) -> KiratRaiLegacyConversion:
        output: list[str] = []
        canonical_run: list[str] = []
        unmapped: set[str] = set()
        replacements = 0

        def flush() -> None:
            nonlocal replacements
            if not canonical_run:
                return
            result = self._canonical.convert("".join(canonical_run))
            output.append(result.unicode_text)
            replacements += result.replacement_count
            unmapped.update(result.unmapped_codepoints)
            canonical_run.clear()

        for char in text:
            remapped = KIRATRAI_HERALD_PREMAP.get(char)
            if remapped is not None:
                canonical_run.append(remapped)
                continue
            punctuation = KIRATRAI_HERALD_PUNCTUATION_MAP.get(char)
            if punctuation is not None:
                canonical_run.append(punctuation)
                continue
            if char in KIRATRAI_HERALD_BLANK_GLYPHS:
                canonical_run.append(" ")
                continue
            if char in KIRATRAI_HERALD_PASSTHROUGH:
                canonical_run.append(char)
                continue
            if 0x16D40 <= ord(char) <= 0x16D7F:
                flush()
                output.append(char)
                continue
            flush()
            output.append(char)
            unmapped.add(f"U+{ord(char):04X}")
        flush()

        converted = unicodedata.normalize("NFC", "".join(output))
        return KiratRaiLegacyConversion(
            legacy_text=text,
            unicode_text=converted,
            kiratrai_char_count=len(_KIRATRAI_CODEPOINT_RE.findall(converted)),
            replacement_count=replacements,
            unmapped_codepoints=sorted(unmapped),
        )


def convert_kiratrai_legacy_text(
    text: str, map_path: str | Path = DEFAULT_KIRATRAI_LEGACY_MAP
) -> KiratRaiLegacyConversion:
    return KiratRaiLegacyConverter.from_map_file(map_path).convert(text)


def convert_kiratrai_herald_text(
    text: str, map_path: str | Path = DEFAULT_KIRATRAI_LEGACY_MAP
) -> KiratRaiLegacyConversion:
    """Convert Sikkim Herald's older/permuted Kirat Rai layout to Unicode."""

    return KiratRaiHeraldConverter.from_map_file(map_path).convert(text)
