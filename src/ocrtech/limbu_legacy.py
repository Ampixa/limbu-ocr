"""Convert Limbu legacy Namdhinggo/Sirijonga text to Unicode Limbu."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataValidationError


DEFAULT_LIMBU_LEGACY_MAP = Path("data/mappings/limbu-legacy/Limbu.map")
_BYTE_RULE_RE = re.compile(r"0x([0-9A-Fa-f]{2})")
_UNICODE_RULE_RE = re.compile(r"U\+([0-9A-Fa-f]{4,6})")
_LIMBU_CODEPOINT_RE = re.compile(r"[\u1900-\u194F]")
_VOWEL_RANGE = range(0x1920, 0x1929)
_SUBJOINED_RANGE = range(0x1929, 0x192C)
_KEMPHRENG = "\u193A"


@dataclass(frozen=True, slots=True)
class LimbuLegacyConversion:
    legacy_text: str
    unicode_text: str
    limbu_char_count: int
    replacement_count: int
    unmapped_codepoints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_text": self.legacy_text,
            "unicode_text": self.unicode_text,
            "limbu_char_count": self.limbu_char_count,
            "replacement_count": self.replacement_count,
            "unmapped_codepoints": self.unmapped_codepoints,
        }


class LimbuLegacyConverter:
    """Small native reader for SIL's Limbu TECkit text map.

    The upstream map is a TECkit map for Namdhinggo SIL L custom-encoded text.
    This converter implements the forward Byte_Unicode rules used by the
    Gorkhapatra PDF extraction path.
    """

    def __init__(self, rules: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
        if not rules:
            raise DataValidationError("Limbu legacy converter requires at least one mapping rule")
        # Stable longest-match-first ordering. For equal-length sources, the first
        # rule seen wins; this lets an override map (parsed first) take precedence
        # over a same-source rule in the canonical map without mutating the latter.
        seen: set[tuple[int, ...]] = set()
        ordered: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        indexed = sorted(enumerate(rules), key=lambda item: (-len(item[1][0]), item[0]))
        for _index, (src, tgt) in indexed:
            if src in seen:
                continue
            seen.add(src)
            ordered.append((src, tgt))
        self._rules = ordered

    @staticmethod
    def _parse_byte_unicode_rules(
        map_path: Path,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        rules: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        in_byte_pass = False
        for raw_line in map_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("Pass("):
                in_byte_pass = line == "Pass(Byte_Unicode)"
                continue
            if not in_byte_pass:
                continue
            if "<>" not in line and ">" not in line:
                continue
            left, right = line.split("<>", 1) if "<>" in line else line.split(">", 1)
            source = tuple(int(value, 16) for value in _BYTE_RULE_RE.findall(left))
            target = tuple(int(value, 16) for value in _UNICODE_RULE_RE.findall(right))
            if source and target:
                rules.append((source, target))
        return rules

    @classmethod
    def from_map_file(
        cls,
        path: str | Path = DEFAULT_LIMBU_LEGACY_MAP,
        *,
        override_map_path: str | Path | None = None,
    ) -> "LimbuLegacyConverter":
        map_path = Path(path)
        if not map_path.is_file():
            raise DataValidationError(f"Limbu legacy map does not exist: {map_path}")
        # Override rules are parsed FIRST so that, on an equal-length source-byte
        # tie, the override wins (see __init__: first rule seen per source wins).
        rules: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        if override_map_path is not None:
            override_path = Path(override_map_path)
            if not override_path.is_file():
                raise DataValidationError(
                    f"Limbu legacy override map does not exist: {override_path}"
                )
            rules.extend(cls._parse_byte_unicode_rules(override_path))
        rules.extend(cls._parse_byte_unicode_rules(map_path))
        return cls(rules)

    def convert(self, text: str) -> LimbuLegacyConversion:
        output: list[str] = []
        unmapped: list[str] = []
        replacements = 0
        index = 0
        while index < len(text):
            code = ord(text[index])
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
            output.append(text[index])
            if code > 0x7F and not (0x1900 <= code <= 0x194F):
                unmapped.append(f"U+{code:04X}")
            index += 1
        converted = _reorder_limbu_vowels_and_subjoined("".join(output))
        converted = unicodedata.normalize("NFC", converted)
        return LimbuLegacyConversion(
            legacy_text=text,
            unicode_text=converted,
            limbu_char_count=len(_LIMBU_CODEPOINT_RE.findall(converted)),
            replacement_count=replacements,
            unmapped_codepoints=sorted(set(unmapped)),
        )

    @staticmethod
    def _matches(text: str, index: int, source: tuple[int, ...]) -> bool:
        if index + len(source) > len(text):
            return False
        return all(ord(text[index + offset]) == code for offset, code in enumerate(source))


def convert_limbu_legacy_text(text: str, map_path: str | Path = DEFAULT_LIMBU_LEGACY_MAP) -> LimbuLegacyConversion:
    return LimbuLegacyConverter.from_map_file(map_path).convert(text)


def convert_gorkhapatra_limbu_pdf_text(
    pdf_text_json_path: str | Path,
    output_dir: str | Path,
    *,
    map_path: str | Path = DEFAULT_LIMBU_LEGACY_MAP,
) -> dict[str, Any]:
    source = Path(pdf_text_json_path)
    if not source.is_file():
        raise DataValidationError(f"PDF text JSON does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid PDF text JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("spans"), list):
        raise DataValidationError(f"PDF text JSON must contain a spans array: {source}")
    converter = LimbuLegacyConverter.from_map_file(map_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    for span in payload["spans"]:
        if not isinstance(span, dict) or not span.get("is_target_font"):
            continue
        conversion = converter.convert(str(span.get("raw_text") or ""))
        row = {
            "span_index": span.get("span_index"),
            "block_index": span.get("block_index"),
            "line_index": span.get("line_index"),
            "font": span.get("font"),
            "bbox": span.get("bbox"),
            **conversion.to_dict(),
        }
        rows.append(row)
        if conversion.unicode_text.strip():
            lines.append(conversion.unicode_text.rstrip())
    unicode_text = "\n".join(lines)
    result = {
        "source_pdf_text_json": str(source),
        "map_path": str(Path(map_path)),
        "sample_id": payload.get("sample_id"),
        "candidate_language": payload.get("candidate_language"),
        "pdf_page_number": payload.get("pdf_page_number"),
        "target_span_count": len(rows),
        "unicode_text": unicode_text,
        "limbu_char_count": len(_LIMBU_CODEPOINT_RE.findall(unicode_text)),
        "rows": rows,
    }
    sample_id = str(payload.get("sample_id") or source.stem)
    json_path = out / f"{sample_id}.limbu-unicode.json"
    text_path = out / f"{sample_id}.limbu-unicode.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(unicode_text + "\n", encoding="utf-8")
    result["json_path"] = str(json_path)
    result["text_path"] = str(text_path)
    return result


def _reorder_limbu_vowels_and_subjoined(text: str) -> str:
    chars = list(text)
    index = 0
    while index < len(chars) - 1:
        current = ord(chars[index])
        nxt = ord(chars[index + 1])
        if current in _VOWEL_RANGE and nxt in _SUBJOINED_RANGE:
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            index += 2
            continue
        if index < len(chars) - 2 and current in _VOWEL_RANGE and chars[index + 1] == _KEMPHRENG and ord(chars[index + 2]) in _SUBJOINED_RANGE:
            vowel = chars[index]
            subjoined = chars[index + 2]
            chars[index : index + 3] = [subjoined, vowel, _KEMPHRENG]
            index += 3
            continue
        index += 1
    return "".join(chars)
