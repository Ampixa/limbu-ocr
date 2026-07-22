"""Decode the audited Sikkim Herald ``Khema`` PUA text layer.

The 2022 Herald PDFs re-encode legacy Khema font bytes as ``U+F0xx``.  The
byte semantics and pre-base medial ordering come from Unicode proposal
L2/22-096.  This module is deliberately font-profile-specific: callers must
first attest the embedded font against the audited Herald profiles.

No font program or outline data is stored here.  The tables are functional
character-decoding data only.
"""

from __future__ import annotations

from dataclasses import dataclass

PUA_OFFSET = 0xF000

# L2/22-096 sections 5.1--5.5 and the Unicode 16 Gurung Khema allocation.
PROPOSAL_LEGACY_TO_UNICODE = {
    "c": "\U00016100",
    "s": "\U00016101",
    "v": "\U00016102",
    "u": "\U00016103",
    "q": "\U00016104",
    "m": "\U00016105",
    "r": "\U00016106",
    "i": "\U00016107",
    "h": "\U00016108",
    "G": "\U00016109",
    "x": "\U0001610A",
    "H": "\U0001610B",
    "J": "\U0001610C",
    "F": "\U0001610D",
    "K": "\U0001610E",
    "j": "\U0001610F",
    "t": "\U00016110",
    "y": "\U00016111",
    "b": "\U00016112",
    "w": "\U00016113",
    "g": "\U00016114",
    "k": "\U00016115",
    "L": "\U00016116",
    "a": "\U00016117",
    "e": "\U00016118",
    "d": "\U00016119",
    "o": "\U0001611A",
    "M": "\U0001611B",
    "n": "\U0001611C",
    "z": "\U0001611D",
    "f": "\U0001611E",
    "l": "\U0001611F",
    "W": "\U00016120",
    "p": "\U00016121",
    "Z": "\U00016122",
    "P": "\U00016123",
    "C": "\U00016124",
    "Q": "\U00016125",
    "R": "\U00016126",
    "B": "\U00016127",
    "S": "\U00016128",
    "T": "\U00016129",
    "E": "\U0001612A",
    "U": "\U0001612B",
    "I": "\U0001612C",
    "\u00e1": "\U0001612D",
    **{str(value): chr(0x16130 + value) for value in range(10)},
}

# The exact Herald Khema profile uses three duplicate/alias slots.  Their
# printed outlines match the proposal's canonical base-letter glyphs, not the
# proposal semantics of the corresponding Latin byte.
HERALD_ALIASES = {
    "C": PROPOSAL_LEGACY_TO_UNICODE["b"],
    "N": PROPOSAL_LEGACY_TO_UNICODE["g"],
    "X": PROPOSAL_LEGACY_TO_UNICODE["s"],
}

HERALD_ALIAS_CANONICAL_KEYS = {
    "C": "b",
    "N": "g",
    "X": "s",
}

# L2/22-096 unifies these two legacy signs with existing combining marks.
UNIFIED_MARKS = {
    "O": "\u030c",  # COMBINING CARON
    "V": "\u032d",  # COMBINING CIRCUMFLEX ACCENT BELOW
}

# The proposal documents Latin punctuation use.  Brackets also occur as
# literal, visibly bracket-shaped glyphs in the pinned April edition.
LITERAL_CHARACTERS = frozenset(" \t:.,<>;_\u2018\u2019()?\\/-![]")

BASE_KEYS = frozenset("csvuqmrihGxHJFKjtybwgkLaedoMnzCNX")
PREBASE_MEDIAL_KEYS = frozenset("TE")


@dataclass(frozen=True)
class DecodeResult:
    """One in-memory decode result; artifacts must not serialize ``text``."""

    text: str | None
    source_units: int
    output_units: int
    double_danda_pairs: int
    reordered_prebase_medials: int
    unresolved: tuple[dict[str, object], ...]

    @property
    def accepted(self) -> bool:
        return self.text is not None and not self.unresolved


def _source_byte(character: str) -> str | None:
    code = ord(character)
    if PUA_OFFSET <= code <= PUA_OFFSET + 0xFF:
        return chr(code - PUA_OFFSET)
    return None


def _map_base(key: str) -> str:
    if key in HERALD_ALIASES:
        return HERALD_ALIASES[key]
    return PROPOSAL_LEGACY_TO_UNICODE[key]


def decode_herald_pua(text: str) -> DecodeResult:
    """Decode one same-font span/line and fail closed on ambiguous sequences.

    A legacy ``DD`` run is the proposal's double danda and becomes U+0965.
    ``T`` and ``E`` are visually pre-base medials, so a directly following
    consonant is emitted first.  Any single ``D`` or non-adjacent pre-base
    medial remains unresolved rather than receiving a guessed label.
    """

    source: list[str | None] = [_source_byte(character) for character in text]
    output: list[str] = []
    unresolved: list[dict[str, object]] = []
    double_danda_pairs = 0
    reordered_prebase_medials = 0
    index = 0
    while index < len(source):
        key = source[index]
        if key is None:
            unresolved.append({
                "source_index": index,
                "reason": "source_code_outside_u_f0xx",
                "codepoint": f"U+{ord(text[index]):04X}",
            })
            index += 1
            continue
        if key == "D":
            if index + 1 < len(source) and source[index + 1] == "D":
                output.append("\u0965")
                double_danda_pairs += 1
                index += 2
            else:
                unresolved.append({
                    "source_index": index,
                    "reason": "isolated_legacy_danda_unit",
                    "source_code": "U+0044",
                })
                index += 1
            continue
        if key in PREBASE_MEDIAL_KEYS:
            following = source[index + 1] if index + 1 < len(source) else None
            if following not in BASE_KEYS:
                unresolved.append({
                    "source_index": index,
                    "reason": "prebase_medial_not_immediately_followed_by_base",
                    "source_code": f"U+{ord(key):04X}",
                    "following_source_code": (
                        None if following is None else f"U+{ord(following):04X}"
                    ),
                })
                index += 1
                continue
            output.extend((_map_base(following), PROPOSAL_LEGACY_TO_UNICODE[key]))
            reordered_prebase_medials += 1
            index += 2
            continue
        if key in HERALD_ALIASES:
            output.append(HERALD_ALIASES[key])
        elif key in PROPOSAL_LEGACY_TO_UNICODE:
            output.append(PROPOSAL_LEGACY_TO_UNICODE[key])
        elif key in UNIFIED_MARKS:
            output.append(UNIFIED_MARKS[key])
        elif key in LITERAL_CHARACTERS:
            output.append(key)
        else:
            unresolved.append({
                "source_index": index,
                "reason": "unmapped_legacy_source_code",
                "source_code": f"U+{ord(key):04X}",
            })
        index += 1

    decoded = "".join(output)
    if unresolved:
        decoded_or_none = None
        output_units = 0
    else:
        decoded_or_none = decoded
        output_units = len(decoded)
    return DecodeResult(
        text=decoded_or_none,
        source_units=len(text),
        output_units=output_units,
        double_danda_pairs=double_danda_pairs,
        reordered_prebase_medials=reordered_prebase_medials,
        unresolved=tuple(unresolved),
    )


def is_allowed_output_character(character: str) -> bool:
    """Return whether a decoded unit belongs to the audited output alphabet."""

    code = ord(character)
    return (
        0x16100 <= code <= 0x16139
        or character in UNIFIED_MARKS.values()
        or character == "\u0965"
        or character in LITERAL_CHARACTERS
    )
