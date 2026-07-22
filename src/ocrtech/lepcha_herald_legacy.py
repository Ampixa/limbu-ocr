"""Convert the *Sikkim Herald live-text* Lepcha font to Unicode Lepcha (Róng).

This is a DIFFERENT font from Jason Glavy's ``JG Lepcha`` (handled by
:mod:`ocrtech.lepcha_legacy`). A cluster of 2021-2022 Sikkim Herald editions
typeset the Lepcha body as live text using an anonymised CFF-subset font family
(PDF FontName ``TT<hex>O00``). Each subset reassigns Latin glyph CODES, but the
named body layout (``TT21B1O00`` / ``TT106DO00`` / ``TTDA7O00`` / ``TT70AO00`` —
verified outline-identical across the 4 editions) shares ONE byte->glyph layout.

Critically, the byte->glyph code is the *identity Latin* encoding (byte 0x56 'V'
draws the glyph named "V"), but the glyph OUTLINE is a Lepcha letter, NOT a Latin
V. So the recoverable PDF "text" is Latin garbage; the real Lepcha content lives
in the glyph shapes. SIL's JG Lepcha map decodes a *different* font and yields
structurally-valid but UNFAITHFUL output here (a sibling agent proved this with a
per-byte A/B render: ``data/sikkim-herald/lepcha/_evalpack_demo/font_mismatch_byte_AB.png``).

The map below was derived by GLYPH-SHAPE IDENTITY: each body-font glyph was
rendered and shape-matched (NCC/IoU, best-shift) against the 74 assigned Lepcha
Unicode codepoints rendered in Noto Sans Lepcha + Mingzat, anchored on the
font's alphabetic consonant structure (the uppercase Latin letters A.. map to the
Lepcha base-consonant series in Unicode order) and confirmed by positional
statistics (pre-base vs post-base) plus round-trip rendering against the printed
crops. See ``outputs/lepcha-map-derivation/`` for the derivation + contact sheets.

Structure handled:
- base consonants (independent glyphs);
- LA-conjunct consonants (KLA/GLA/PLA/FLA/BLA/MLA/HLA) — single bytes;
- the independent vowel A;
- PRE-BASE dependent vowel signs (I, O, OO): in the legacy stream they are keyed
  to the LEFT of (before) their base consonant; Unicode stores them AFTER the
  base, so a reorder pass moves the base to the front of the cluster;
- POST-BASE vowel signs (AA, U, UU, E);
- final consonant signs (-ng/-k/-m/-l/-n/-p/-r/-t) and RAN/NUKTA;
- digits 0-9.

Output is NFC, canonical Lepcha storage order
``C (subjoined) (vowel) (final) (ran)``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataValidationError

DEFAULT_HERALD_MAP = Path(__file__).with_name("maps") / "sikkim_herald_lepcha.json"

LEPCHA_LO, LEPCHA_HI = 0x1C00, 0x1C4F
_LEPCHA_CODEPOINT_RE = re.compile(r"[ᰀ-ᱏ]")

# Pre-base dependent vowel signs (keyed before the base in the legacy visual
# stream; Unicode stores them after the base). I / O / OO.
PRE_BASE_VOWELS = frozenset({0x1C27, 0x1C28, 0x1C29})

# Codepoint classes for canonical ordering within a cluster.
_SUBJOINED = frozenset({0x1C24, 0x1C25})            # subjoined YA, RA
_VOWEL_SIGNS = frozenset(range(0x1C26, 0x1C2D))      # AA, I, O, OO, U, UU, E
_FINAL_SIGNS = frozenset(range(0x1C2D, 0x1C36))      # K, M, L, N, P, R, T, NYIN-DO, KANG
_RAN = 0x1C36
_NUKTA = 0x1C37
_BASES = frozenset(
    list(range(0x1C00, 0x1C24)) + [0x1C4D, 0x1C4E, 0x1C4F]
)  # consonants + independent vowel A + TTA/TTHA/DDA


@dataclass(frozen=True, slots=True)
class HeraldLepchaConversion:
    legacy_text: str
    unicode_text: str
    lepcha_char_count: int
    replacement_count: int
    unmapped_bytes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_text": self.legacy_text,
            "unicode_text": self.unicode_text,
            "lepcha_char_count": self.lepcha_char_count,
            "replacement_count": self.replacement_count,
            "unmapped_bytes": self.unmapped_bytes,
        }


class HeraldLepchaConverter:
    """Byte->Unicode converter for the Sikkim Herald live-text Lepcha body font.

    ``byte_map`` maps a single legacy byte value (the Latin code) to a tuple of
    Lepcha codepoints. Conversion is a single byte pass (no multi-byte rules in
    this font) followed by a per-cluster reorder that (a) moves pre-base vowel
    signs after the base and (b) sorts the dependent signs into canonical Lepcha
    storage order.
    """

    def __init__(self, byte_map: dict[int, tuple[int, ...]]) -> None:
        if not byte_map:
            raise DataValidationError("Herald Lepcha converter requires a non-empty map")
        self._byte_map = dict(byte_map)

    @classmethod
    def from_map_file(cls, path: str | Path = DEFAULT_HERALD_MAP) -> "HeraldLepchaConverter":
        map_path = Path(path)
        if not map_path.is_file():
            raise DataValidationError(f"Herald Lepcha map does not exist: {map_path}")
        raw = json.loads(map_path.read_text(encoding="utf-8"))
        entries = raw.get("map")
        if not isinstance(entries, dict):
            raise DataValidationError(f"Herald Lepcha map missing 'map' object: {map_path}")
        byte_map: dict[int, tuple[int, ...]] = {}
        for byte_hex, target in entries.items():
            try:
                b = int(byte_hex, 16)
            except ValueError as exc:
                raise DataValidationError(
                    f"invalid byte key in Herald map: {byte_hex!r}"
                ) from exc
            if not (0 <= b <= 0xFF):
                raise DataValidationError(f"byte key out of range in Herald map: {byte_hex!r}")
            cps = tuple(int(u, 16) for u in target)
            for cp in cps:
                if not (LEPCHA_LO <= cp <= LEPCHA_HI):
                    raise DataValidationError(
                        f"Herald map target U+{cp:04X} outside Lepcha block for byte {byte_hex}"
                    )
            byte_map[b] = cps
        return cls(byte_map)

    # ----- byte pass -----------------------------------------------------------

    def _byte_pass(self, text: str) -> tuple[list[int | str], int, list[str]]:
        """Map each input char (legacy byte) to codepoints; pass spaces through.

        Returns a token list where ints are Lepcha codepoints and str tokens are
        verbatim pass-through characters (spaces, untouched ASCII).
        """
        out: list[int | str] = []
        unmapped: list[str] = []
        replacements = 0
        for ch in text:
            code = ord(ch)
            if ch == " " or ch == "\n" or ch == "\t":
                out.append(ch)
                continue
            target = self._byte_map.get(code)
            if target is not None:
                out.extend(target)
                replacements += 1
            else:
                # Unmapped legacy byte (layout/punctuation glyph not in the map).
                out.append(ch)
                unmapped.append(f"0x{code:02X}" if code <= 0xFF else f"U+{code:04X}")
        return out, replacements, unmapped

    # ----- reorder pass --------------------------------------------------------

    @staticmethod
    def _canonical_cluster(base: int, signs: list[int]) -> list[int]:
        """Order a base + its dependent signs into canonical Lepcha storage order.

        Order: base, subjoined (YA/RA), nukta, vowel sign, final consonant sign,
        RAN. (Lepcha storage order per Unicode: C + subjoin + vowel + final +
        ran; nukta sits with the base.)
        """
        subjoined = [s for s in signs if s in _SUBJOINED]
        nukta = [s for s in signs if s == _NUKTA]
        vowels = [s for s in signs if s in _VOWEL_SIGNS]
        finals = [s for s in signs if s in _FINAL_SIGNS]
        ran = [s for s in signs if s == _RAN]
        other = [
            s
            for s in signs
            if s not in _SUBJOINED
            and s != _NUKTA
            and s not in _VOWEL_SIGNS
            and s not in _FINAL_SIGNS
            and s != _RAN
        ]
        return [base] + subjoined + nukta + vowels + finals + ran + other

    def _reorder(self, tokens: list[int | str]) -> list[int | str]:
        """Reorder visual-order codepoint runs into logical Lepcha clusters.

        Walks the token stream. A syllable = optional leading PRE-BASE vowel
        sign(s) (I/O/OO, keyed left of the base in the legacy stream), then the
        base, then trailing dependent signs (post-base vowels, finals, ran,
        nukta). Each syllable is emitted in canonical storage order
        ``base + subjoined + nukta + vowel + final + ran``. Crucially, a trailing
        sign-run STOPS at the next pre-base vowel, which begins the next syllable.
        """
        out: list[int | str] = []
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            if isinstance(tok, str):
                out.append(tok)
                i += 1
                continue

            # Collect any leading pre-base vowel signs for the upcoming base.
            pre: list[int] = []
            while (
                i < n
                and isinstance(tokens[i], int)
                and tokens[i] in PRE_BASE_VOWELS
                and tokens[i] not in _BASES
            ):
                pre.append(tokens[i])  # type: ignore[arg-type]
                i += 1

            if i >= n or not isinstance(tokens[i], int):
                # No base follows the pre-vowel(s); emit them verbatim (degenerate).
                out.extend(pre)
                continue

            cur = tokens[i]
            if cur not in _BASES:
                # A stray dependent sign with no base (and not a pre-vowel).
                out.extend(pre)
                out.append(cur)
                i += 1
                continue

            base = cur
            i += 1
            # Trailing dependent signs, stopping at the next base OR the next
            # pre-base vowel (which starts a new syllable).
            post: list[int] = []
            while (
                i < n
                and isinstance(tokens[i], int)
                and tokens[i] not in _BASES
                and tokens[i] not in PRE_BASE_VOWELS
            ):
                post.append(tokens[i])  # type: ignore[arg-type]
                i += 1
            out.extend(self._canonical_cluster(base, pre + post))
        return out

    # ----- public --------------------------------------------------------------

    def convert(self, text: str) -> HeraldLepchaConversion:
        tokens, replacements, unmapped = self._byte_pass(text)
        reordered = self._reorder(tokens)
        chars = []
        for t in reordered:
            chars.append(chr(t) if isinstance(t, int) else t)
        converted = unicodedata.normalize("NFC", "".join(chars))
        return HeraldLepchaConversion(
            legacy_text=text,
            unicode_text=converted,
            lepcha_char_count=len(_LEPCHA_CODEPOINT_RE.findall(converted)),
            replacement_count=replacements,
            unmapped_bytes=sorted(set(unmapped)),
        )


def convert_herald_lepcha_text(
    text: str, map_path: str | Path = DEFAULT_HERALD_MAP
) -> HeraldLepchaConversion:
    return HeraldLepchaConverter.from_map_file(map_path).convert(text)
