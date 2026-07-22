"""Convert JG Lepcha legacy-encoded text to Unicode Lepcha (Róng script).

The Sikkim Herald Lepcha edition is typeset in *Jason Glavy's* ``JG Lepcha``
custom-encoded TrueType font: a font that stuffs the Lepcha (Róng) glyph
repertoire into a Times-New-Roman shell, addressing each glyph by a single
legacy byte (Latin code point). SIL publishes a TECkit mapping
(``silnrsi/wsresources`` -> ``scripts/Lepc/legacy/jg-lepcha/mappings/
JGLepcha.map``) that converts that legacy byte stream to the Lepcha Unicode
block (U+1C00-U+1C4F).

The map is a *two-pass* TECkit conversion:

``pass(Byte_Unicode)``
    ``ByteClass``/``UniClass`` declarations plus explicit single-byte and
    composite (multi-codepoint) rules map the legacy bytes to Lepcha codepoints
    in the order the bytes appear in the font's *visual* layout. This pass also
    contains the one context rule ``0x61 / ^[DepVow] _ > U+1C26`` (a bare ``a``
    after a dependent vowel is the independent vowel A, not the dep. vowel AA).

``pass(Unicode)``
    A reordering pass that rewrites runs of ``C (.) (R) (Y) (V) (F) (^)`` into
    the canonical Lepcha storage order. The legacy font keys some dependent
    vowels and final-consonant signs *before* their base consonant (visual
    order); this pass moves the base consonant to the front so the output is in
    logical/Unicode order, not visual order.

This module parses the TECkit map natively (no ``teckit_compile`` dependency,
which is unavailable on this machine) and applies both passes. It mirrors the
structure of :mod:`ocrtech.kiratrai_legacy`.

NOTE on the Sikkim Herald source: in the *current* Sikkim Herald Lepcha PDFs the
Lepcha body text is rasterised into image strips, so there is no recoverable
legacy byte stream for the body columns (only the English masthead/colophon and
inter-word punctuation are live text). This converter is therefore validated
against the SIL map's own round-trip rules and a synthetic JG Lepcha byte
sample; it will convert any genuine JG-Lepcha-encoded byte stream should one
become available (e.g. an older live-text edition or the source DTP files).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataValidationError

DEFAULT_LEPCHA_LEGACY_MAP = Path("data/mappings/lepcha-legacy/JGLepcha.map")

LEPCHA_LO, LEPCHA_HI = 0x1C00, 0x1C4F
_LEPCHA_CODEPOINT_RE = re.compile(r"[ᰀ-ᱏ]")

_BYTE_TOKEN_RE = re.compile(r"0x([0-9A-Fa-f]{2})")
_UNI_TOKEN_RE = re.compile(r"U\+([0-9A-Fa-f]{4,6})")
_BYTECLASS_RE = re.compile(r"^ByteClass\s*\[([^\]]+)\]\s*=\s*\((.*)\)\s*$")
_UNICLASS_RE = re.compile(r"^UniClass\s*\[([^\]]+)\]\s*=\s*\((.*)\)\s*$")
# Pass(Unicode) classes are declared as bare ``Class[name] = (...)``.
_PLAINCLASS_RE = re.compile(r"^Class\s*\[([^\]]+)\]\s*=\s*\((.*)\)\s*$")
# A simple ``[byteclass] <> [uniclass]`` Pass-1 rule.
_CLASS_RULE_RE = re.compile(r"^\[([^\]]+)\]\s*<?>\s*\[([^\]]+)\]\s*$")
# A Pass-2 reorder rule: ``[Cls]=v [Cls]=c ... <> @c @v ...``
_BOUND_CLASS_RE = re.compile(r"\[([^\]]+)\]\s*=\s*([A-Za-z]\w*)")
_VAR_REF_RE = re.compile(r"@([A-Za-z]\w*)")
# The single context rule: ``0x61 / ^[DepVow] _ > U+1C26``
_CONTEXT_RULE_RE = re.compile(
    r"^0x([0-9A-Fa-f]{2})\s*/\s*\^\s*\[([^\]]+)\]\s*_\s*>\s*U\+([0-9A-Fa-f]{4,6})\s*$"
)


def _split_tokens(body: str) -> list[str]:
    # The JG Lepcha map writes ranges both spaced (``0x00 .. 0x1F``) and compact
    # (``U+1C00..U+1C23``). Normalise the compact form to the spaced form so the
    # range handling below sees a uniform ``A .. B`` token stream.
    body = re.sub(r"\s*\.\.\s*", " .. ", body)
    return [t for t in re.split(r"\s+", body.strip()) if t]


def _expand_byte_tokens(body: str) -> tuple[int, ...]:
    """Expand a TECkit byte-class body into an ordered tuple of byte values.

    Supports explicit ``0xNN`` tokens and ``0xNN .. 0xMM`` ranges.
    """
    values: list[int] = []
    tokens = _split_tokens(body)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index + 2 < len(tokens) and tokens[index + 1] == "..":
            start = int(token, 16)
            end = int(tokens[index + 2], 16)
            if end < start:
                raise DataValidationError(
                    f"invalid byte range in map: {token}..{tokens[index + 2]}"
                )
            values.extend(range(start, end + 1))
            index += 3
            continue
        if not _BYTE_TOKEN_RE.fullmatch(token):
            raise DataValidationError(f"unparseable byte token in map: {token!r}")
        values.append(int(token, 16))
        index += 1
    return tuple(values)


def _expand_uni_tokens(body: str) -> tuple[int, ...]:
    """Expand a TECkit Unicode-class body into an ordered tuple of codepoints."""
    values: list[int] = []
    tokens = _split_tokens(body)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index + 2 < len(tokens) and tokens[index + 1] == "..":
            start_match = _UNI_TOKEN_RE.fullmatch(token)
            end_match = _UNI_TOKEN_RE.fullmatch(tokens[index + 2])
            if start_match is None or end_match is None:
                raise DataValidationError(
                    f"invalid unicode range in map: {token}..{tokens[index + 2]}"
                )
            start = int(start_match.group(1), 16)
            end = int(end_match.group(1), 16)
            if end < start:
                raise DataValidationError(
                    f"invalid unicode range in map: {token}..{tokens[index + 2]}"
                )
            values.extend(range(start, end + 1))
            index += 3
            continue
        match = _UNI_TOKEN_RE.fullmatch(token)
        if match is None:
            raise DataValidationError(f"unparseable unicode token in map: {token!r}")
        values.append(int(match.group(1), 16))
        index += 1
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _ReorderRule:
    """A Pass(Unicode) contextual reorder rule.

    ``slots`` is an ordered list of ``(class_name, var_name)`` describing the LHS
    match sequence; ``output_vars`` is the ordered list of variable names on the
    RHS. Each slot matches exactly one input codepoint that belongs to the named
    class; the rule rewrites the matched run by emitting the bound codepoints in
    the RHS order.
    """

    slots: tuple[tuple[str, str], ...]
    output_vars: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LepchaLegacyConversion:
    legacy_text: str
    unicode_text: str
    lepcha_char_count: int
    replacement_count: int
    unmapped_codepoints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_text": self.legacy_text,
            "unicode_text": self.unicode_text,
            "lepcha_char_count": self.lepcha_char_count,
            "replacement_count": self.replacement_count,
            "unmapped_codepoints": self.unmapped_codepoints,
        }


class LepchaLegacyConverter:
    """Native two-pass reader/applier for SIL's ``JGLepcha.map`` TECkit mapping.

    Pass 1 (``Byte_Unicode``) flattens every byte-class and explicit rule into
    ``(source_bytes, target_codepoints)`` pairs, applied longest-match-first so
    composite/conjunct multi-byte rules win over single-byte class rules. The one
    context rule (bare ``a`` after a dependent vowel -> independent A) is applied
    as a guarded special case.

    Pass 2 (``Unicode``) applies the contextual reorder rules, longest-LHS-first,
    to rewrite visual-order runs into canonical Lepcha storage order.
    """

    def __init__(
        self,
        byte_rules: list[tuple[tuple[int, ...], tuple[int, ...]]],
        reorder_rules: list[_ReorderRule],
        uni_classes: dict[str, frozenset[int]],
        context_rule: tuple[int, frozenset[int], int] | None,
    ) -> None:
        if not byte_rules:
            raise DataValidationError(
                "Lepcha legacy converter requires at least one Byte_Unicode rule"
            )
        seen: set[tuple[int, ...]] = set()
        ordered: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for source, target in sorted(byte_rules, key=lambda item: len(item[0]), reverse=True):
            if source in seen:
                continue
            seen.add(source)
            ordered.append((source, target))
        self._byte_rules = ordered
        # Longest LHS first so the most specific reorder pattern matches.
        self._reorder_rules = sorted(
            reorder_rules, key=lambda r: len(r.slots), reverse=True
        )
        self._uni_classes = uni_classes
        self._context_rule = context_rule

    @classmethod
    def from_map_file(
        cls, path: str | Path = DEFAULT_LEPCHA_LEGACY_MAP
    ) -> "LepchaLegacyConverter":
        map_path = Path(path)
        if not map_path.is_file():
            raise DataValidationError(f"Lepcha legacy map does not exist: {map_path}")

        # Join backslash line continuations, then strip comments per line.
        raw = map_path.read_text(encoding="utf-8-sig")
        raw = re.sub(r"\\\s*\n", " ", raw)

        byte_classes: dict[str, tuple[int, ...]] = {}
        uni_classes_ordered: dict[str, tuple[int, ...]] = {}
        plain_classes: dict[str, frozenset[int]] = {}
        byte_rules: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        reorder_rules: list[_ReorderRule] = []
        context_rule: tuple[int, frozenset[int], int] | None = None

        current_pass = ""
        byte_rule_lines: list[str] = []

        for raw_line in raw.splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("pass("):
                current_pass = lowered
                continue

            if current_pass == "pass(byte_unicode)":
                byte_match = _BYTECLASS_RE.match(line)
                if byte_match:
                    byte_classes[byte_match.group(1).strip()] = _expand_byte_tokens(
                        byte_match.group(2)
                    )
                    continue
                uni_match = _UNICLASS_RE.match(line)
                if uni_match:
                    uni_classes_ordered[uni_match.group(1).strip()] = _expand_uni_tokens(
                        uni_match.group(2)
                    )
                    continue
                byte_rule_lines.append(line)
                continue

            if current_pass == "pass(unicode)":
                plain_match = _PLAINCLASS_RE.match(line)
                if plain_match:
                    plain_classes[plain_match.group(1).strip()] = frozenset(
                        _expand_uni_tokens(plain_match.group(2))
                    )
                    continue
                rule = cls._parse_reorder_rule(line)
                if rule is not None:
                    reorder_rules.append(rule)
                continue

        # Second pass over Byte_Unicode rule lines so class rules can reference
        # classes regardless of declaration order.
        for line in byte_rule_lines:
            ctx = _CONTEXT_RULE_RE.match(line)
            if ctx:
                cls_name = ctx.group(2).strip()
                cls_bytes = byte_classes.get(cls_name)
                if cls_bytes is None:
                    raise DataValidationError(
                        f"context rule references unknown byte class: {cls_name!r}"
                    )
                # The context class names a *byte* class but the run-time check is
                # against already-converted Unicode codepoints. Map the byte class
                # to its Unicode equivalents via the paired UniClass.
                cls_codepoints = uni_classes_ordered.get(cls_name)
                if cls_codepoints is None:
                    raise DataValidationError(
                        f"context rule class {cls_name!r} has no UniClass"
                    )
                context_rule = (
                    int(ctx.group(1), 16),
                    frozenset(cls_codepoints),
                    int(ctx.group(3), 16),
                )
                continue

            class_rule = _CLASS_RULE_RE.match(line)
            if class_rule:
                left_name = class_rule.group(1).strip()
                right_name = class_rule.group(2).strip()
                left = byte_classes.get(left_name)
                right = uni_classes_ordered.get(right_name)
                if left is None:
                    raise DataValidationError(
                        f"class rule references unknown byte class: {left_name!r}"
                    )
                if right is None:
                    raise DataValidationError(
                        f"class rule references unknown unicode class: {right_name!r}"
                    )
                if len(left) != len(right):
                    raise DataValidationError(
                        f"class rule length mismatch for [{left_name}]>[{right_name}]: "
                        f"{len(left)} bytes vs {len(right)} codepoints"
                    )
                for byte_value, codepoint in zip(left, right):
                    byte_rules.append(((byte_value,), (codepoint,)))
                continue

            if "<>" in line or ">" in line:
                left_text, right_text = (
                    line.split("<>", 1) if "<>" in line else line.split(">", 1)
                )
                source = _BYTE_TOKEN_RE.findall(left_text)
                target = _UNI_TOKEN_RE.findall(right_text)
                if source and target:
                    byte_rules.append(
                        (
                            tuple(int(value, 16) for value in source),
                            tuple(int(value, 16) for value in target),
                        )
                    )

        return cls(byte_rules, reorder_rules, plain_classes, context_rule)

    @staticmethod
    def _parse_reorder_rule(line: str) -> _ReorderRule | None:
        if "<>" not in line:
            return None
        left, right = line.split("<>", 1)
        slots = tuple(
            (cls.strip(), var.strip())
            for cls, var in _BOUND_CLASS_RE.findall(left)
        )
        output_vars = tuple(_VAR_REF_RE.findall(right))
        if not slots or not output_vars:
            return None
        # Only keep well-formed permutation rules where every output var is bound.
        bound = {var for _, var in slots}
        if any(var not in bound for var in output_vars):
            return None
        return _ReorderRule(slots=slots, output_vars=output_vars)

    # ----- Pass 1: Byte_Unicode -------------------------------------------------

    def _apply_byte_pass(self, text: str) -> tuple[str, int, list[str]]:
        output: list[str] = []
        unmapped: list[str] = []
        replacements = 0
        index = 0
        length = len(text)
        while index < length:
            # Context rule: bare 0x61 after a (previously emitted) dependent vowel
            # becomes the independent vowel A (U+1C26) rather than dep. vowel AA.
            if self._context_rule is not None:
                trigger_byte, prev_classes, replacement = self._context_rule
                if (
                    ord(text[index]) == trigger_byte
                    and output
                    and ord(output[-1]) in prev_classes
                ):
                    output.append(chr(replacement))
                    replacements += 1
                    index += 1
                    continue

            matched = False
            for source, target in self._byte_rules:
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
            if code > 0x7F and not (LEPCHA_LO <= code <= LEPCHA_HI):
                unmapped.append(f"U+{code:04X}")
            index += 1
        return "".join(output), replacements, unmapped

    # ----- Pass 2: Unicode reorder ---------------------------------------------

    def _apply_reorder_pass(self, text: str) -> str:
        output: list[str] = []
        index = 0
        length = len(text)
        while index < length:
            applied = False
            for rule in self._reorder_rules:
                bound = self._match_reorder(text, index, rule)
                if bound is not None:
                    output.extend(bound[var] for var in rule.output_vars)
                    index += len(rule.slots)
                    applied = True
                    break
            if applied:
                continue
            output.append(text[index])
            index += 1
        return "".join(output)

    def _match_reorder(
        self, text: str, index: int, rule: _ReorderRule
    ) -> dict[str, str] | None:
        if index + len(rule.slots) > len(text):
            return None
        bound: dict[str, str] = {}
        for offset, (cls_name, var_name) in enumerate(rule.slots):
            members = self._uni_classes.get(cls_name)
            if members is None:
                return None
            ch = text[index + offset]
            if ord(ch) not in members:
                return None
            bound[var_name] = ch
        return bound

    @staticmethod
    def _matches(text: str, index: int, source: tuple[int, ...]) -> bool:
        if index + len(source) > len(text):
            return False
        return all(ord(text[index + offset]) == code for offset, code in enumerate(source))

    # ----- public --------------------------------------------------------------

    def convert(self, text: str) -> LepchaLegacyConversion:
        stage1, replacements, unmapped = self._apply_byte_pass(text)
        stage2 = self._apply_reorder_pass(stage1)
        converted = unicodedata.normalize("NFC", stage2)
        return LepchaLegacyConversion(
            legacy_text=text,
            unicode_text=converted,
            lepcha_char_count=len(_LEPCHA_CODEPOINT_RE.findall(converted)),
            replacement_count=replacements,
            unmapped_codepoints=sorted(set(unmapped)),
        )


def convert_lepcha_legacy_text(
    text: str, map_path: str | Path = DEFAULT_LEPCHA_LEGACY_MAP
) -> LepchaLegacyConversion:
    return LepchaLegacyConverter.from_map_file(map_path).convert(text)
