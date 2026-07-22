"""Hash-pinned recovery for the Janaki text layer in the Videha sample PDF.

The verified Videha issue uses a Janaki CID font whose incomplete PDF
``ToUnicode`` table emits U+FFFD for many conjunct glyphs. PyMuPDF retains the
corresponding glyph ID in ``Page.get_texttrace()`` output, so those characters
can be recovered without redistributing or rendering the font program.

This module is intentionally sample-scoped. It rejects any PDF, embedded-font
set, or replacement glyph ID that is not part of the audited profile. The
recovered text is still Devanagari-coded Janaki input; callers must pass it to
``nepal_ttf2utf.convert_tirhuta`` for Unicode Tirhuta conversion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VIDEHA_SAMPLE_PDF_SHA256 = (
    "91ec43fdc5ccd22cf449457f94e159650b944fea5cf35c7baec89a695d146722"
)
VIDEHA_SAMPLE_PAGE_COUNT = 152
VIDEHA_SAMPLE_EXPECTED_REPLACEMENTS = 3306

# PyMuPDF ``Document.extract_font`` SHA-256 values for the only two Janaki
# fonts in the hash-pinned sample. The Type0/Identity-H face carries the CID
# glyph IDs; the small WinAnsi face carries punctuation and display material.
VIDEHA_SAMPLE_JANAKI_FONT_SHA256 = frozenset(
    {
        "b51da8d0c99bf8cc0e7ee85f18681272b0f57eb80f277838f4e2cdcaa5253755",
        "1e3da463c92b8563d4f22db4c0f31b366668988da5008dccdff68f96a44e3501",
    }
)

# Separately dated issue from the same CC BY-SA Internet Archive item. It uses
# the same Janaki 1.000 glyph system but embeds a larger subset than issue 001.
VIDEHA_2008_04_15_PDF_SHA256 = (
    "740782ecf5bfa9466727029bcb7733d9c8b046c36d848b598ddc60efc1c51bd2"
)
VIDEHA_2008_04_15_PAGE_COUNT = 300
VIDEHA_2008_04_15_EXPECTED_REPLACEMENTS = 15476
VIDEHA_2008_04_15_JANAKI_FONT_SHA256 = frozenset(
    {
        "c64600a4edc0fa153717d66d2524c1665562eee47dd489848578e3cec1c56861",
        "d8863d057541d5cecb862fd43e93114a9a20c6d5de519fc30f3c990962a8b18b",
    }
)


# Functional decoding data only: no glyph outlines or font tables are stored.
# Each key is a Janaki glyph ID observed where the sample's ToUnicode map emits
# U+FFFD. Values are the unique Devanagari input sequences that shape to the
# glyph in Janaki 1.000. The two half forms (167 and 172) are the corresponding
# base consonant plus virama.
JANAKI_GID_TO_DEVANAGARI: Mapping[int, str] = {
    56: "ट्ट",
    57: "ट्ठ",
    58: "ट्य",
    60: "ठ्य",
    63: "द्म",
    64: "द्य",
    65: "ह्ण",
    66: "ह्न",
    67: "ह्म",
    68: "ह्य",
    69: "ह्व",
    70: "क्ष्म",
    71: "क्ष्य",
    74: "स्थ्य",
    75: "क्ख",
    76: "क्च",
    77: "क्ट",
    78: "क्त",
    81: "क्म",
    82: "क्ल",
    83: "क्व",
    84: "क्श",
    85: "ख्न",
    86: "ख्य",
    88: "ग्न",
    89: "च्च",
    90: "च्य",
    92: "ज्व",
    94: "त्क",
    96: "त्थ",
    97: "त्न",
    98: "त्प",
    100: "त्म",
    101: "त्य",
    102: "त्व",
    103: "त्स",
    105: "ध्य",
    107: "न्ध",
    108: "न्य",
    109: "न्स",
    113: "फ्ल",
    114: "ब्य",
    116: "भ्य",
    118: "म्य",
    120: "म्स",
    121: "म्ह",
    122: "ल्द",
    123: "ल्प",
    124: "ल्ब",
    126: "ल्म",
    127: "ल्य",
    128: "ल्व",
    129: "ल्स",
    130: "ल्ह",
    131: "व्य",
    134: "श्च",
    135: "श्व",
    136: "ष्क",
    137: "ष्ट",
    138: "ष्ठ",
    139: "ष्ण",
    140: "ष्प",
    142: "ष्म",
    143: "ष्य",
    146: "स्क",
    147: "स्ख",
    148: "स्न",
    149: "स्म",
    150: "स्य",
    151: "स्ल",
    152: "स्व",
    153: "स्स",
    156: "रु",
    157: "रू",
    167: "ण्",
    172: "न्",
    213: "क्र",
    214: "ङ्ग",
    216: "ह्ल",
    217: "क्य",
    218: "क्स",
    219: "घ्न",
    221: "च्छ",
    222: "ज्ज",
    223: "ज्य",
    224: "ञ्च",
    230: "ग्र",
    231: "घ्र",
    237: "ट्र",
    239: "ड्र",
    241: "त्र",
    243: "ध्र",
    245: "प्र",
    246: "फ्र",
    247: "ब्र",
    248: "भ्र",
    249: "म्र",
    250: "व्र",
    251: "श्र",
    252: "स्र",
    253: "ह्र",
    258: "तृ",
    260: "हृ",
    263: "क्क",
    264: "क्न",
    266: "ग्य",
    271: "ड्ड",
    272: "ण्ट",
    274: "ण्ड",
    276: "ण्ण",
    277: "त्त",
    278: "थ्य",
    279: "थ्व",
    282: "द्द",
    283: "द्ध",
    285: "द्भ",
    286: "द्व",
    287: "ध्व",
    289: "न्ग",
    290: "न्त",
    291: "न्द",
    292: "न्न",
    293: "न्म",
    295: "न्ह",
    296: "प्त",
    297: "प्न",
    298: "प्प",
    299: "प्ल",
    304: "ब्द",
    305: "ब्ध",
    309: "म्ब",
    310: "म्भ",
    311: "म्म",
    313: "ल्क",
    315: "ल्ल",
    317: "श्न",
    318: "श्म",
    319: "श्य",
    320: "स्ट",
    321: "स्त",
    322: "स्थ",
    323: "स्प",
    340: "ग्ध",
    351: "ण्य",
    352: "ण्व",
    367: "म्प",
    369: "म्ल",
    370: "य्य",
    373: "श्ल",
    404: "र्ग",
    419: "णे",
    424: "ने",
    457: "नै",
    481: "णो",
    486: "नो",
    519: "नौ",
    533: "क्षे",
    541: "नु",
    593: "द्र",
    596: "ग्रे",
    603: "ट्रे",
    607: "त्रे",
    611: "प्रे",
    617: "श्रे",
}


# Functional extension for the separately hash-pinned 15 April 2008 issue.
# Thirty-three entries uniquely shape to the named GID in Janaki 1.000 over an
# exhaustive bounded search of 555,885 one-to-three-consonant clusters. GID 155
# is a contextual half form: त्र्क shapes to [155, 12]. No font program or
# outline data is stored here.
JANAKI_GID_EXTENSION_2008_04_15: Mapping[int, str] = {
    51: "ङ्क",
    53: "ङ्घ",
    106: "न्थ",
    111: "प्य",
    119: "म्व",
    155: "त्र्",
    215: "ड्य",
    220: "घ्य",
    226: "ञ्ज",
    235: "ज्र",
    244: "न्र",
    259: "भृ",
    267: "ग्व",
    273: "ण्ठ",
    281: "द्घ",
    294: "न्व",
    301: "प्स",
    303: "ब्ज",
    306: "ब्ब",
    308: "म्न",
    312: "य्व",
    314: "ल्ग",
    316: "श्छ",
    324: "स्फ",
    332: "ग्ग",
    344: "ग्ल",
    345: "ङ्म",
    355: "न्ख",
    359: "न्फ",
    388: "त्म्य",
    414: "ञे",
    539: "णु",
    612: "फ्रे",
    613: "ब्रे",
}

JANAKI_GID_TO_DEVANAGARI_2008_04_15: Mapping[int, str] = {
    **JANAKI_GID_TO_DEVANAGARI,
    **JANAKI_GID_EXTENSION_2008_04_15,
}

VIDEHA_2008_04_15_EXPECTED_REPLACEMENT_GIDS = frozenset(
    {
        51, 53, 56, 57, 58, 60, 63, 64, 66, 67, 68, 69, 70, 71, 74, 75,
        76, 77, 78, 82, 83, 84, 86, 88, 89, 90, 92, 94, 96, 97, 98, 100,
        101, 102, 103, 105, 106, 107, 108, 109, 111, 113, 114, 116, 118,
        119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 134,
        135, 136, 137, 138, 139, 140, 142, 143, 146, 148, 149, 150, 151,
        152, 153, 155, 156, 157, 167, 172, 213, 214, 215, 216, 217, 218,
        220, 221, 222, 223, 224, 226, 230, 231, 235, 237, 239, 241, 243,
        244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 258, 259, 260,
        263, 264, 266, 267, 271, 272, 273, 274, 277, 278, 279, 281, 282,
        283, 285, 286, 287, 289, 290, 291, 292, 293, 294, 295, 296, 297,
        298, 299, 301, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313,
        314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 332, 340,
        344, 345, 351, 352, 355, 359, 367, 369, 370, 373, 388, 404, 414,
        419, 424, 457, 481, 486, 519, 533, 539, 541, 593, 596, 603, 607,
        611, 612, 613, 617,
    }
)


class VidehaProfileError(ValueError):
    """Raised when the input falls outside the audited Videha profile."""


class UnknownJanakiGlyphError(VidehaProfileError):
    """Raised when U+FFFD is paired with an unaudited Janaki glyph ID."""


@dataclass(frozen=True)
class TraceRecovery:
    """Recovered Devanagari-coded text and replacement accounting."""

    text: str
    replacement_count: int
    recovered_count: int
    recovered_gids: tuple[int, ...]


def sha256_path(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gid_map_sha256(mapping: Mapping[int, str] = JANAKI_GID_TO_DEVANAGARI) -> str:
    """Return the canonical digest of the functional GID recovery map."""
    payload = json.dumps(
        {str(gid): text for gid, text in sorted(mapping.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_profile_fingerprints(
    *,
    pdf_sha256: str,
    janaki_font_sha256: Iterable[str],
    page_count: int,
) -> None:
    """Reject inputs that do not exactly match the audited sample profile."""
    if pdf_sha256 != VIDEHA_SAMPLE_PDF_SHA256:
        raise VidehaProfileError(
            f"unsupported Videha PDF SHA-256: {pdf_sha256}; expected {VIDEHA_SAMPLE_PDF_SHA256}"
        )
    actual_fonts = frozenset(janaki_font_sha256)
    if actual_fonts != VIDEHA_SAMPLE_JANAKI_FONT_SHA256:
        raise VidehaProfileError(
            "unsupported Janaki font fingerprint set: "
            f"{sorted(actual_fonts)}; expected {sorted(VIDEHA_SAMPLE_JANAKI_FONT_SHA256)}"
        )
    if page_count != VIDEHA_SAMPLE_PAGE_COUNT:
        raise VidehaProfileError(
            f"unsupported page count: {page_count}; expected {VIDEHA_SAMPLE_PAGE_COUNT}"
        )


def recover_janaki_trace_chars(
    chars: Sequence[Sequence[object]],
    *,
    mapping: Mapping[int, str] = JANAKI_GID_TO_DEVANAGARI,
) -> TraceRecovery:
    """Recover one Janaki ``get_texttrace`` character sequence.

    PyMuPDF trace characters begin with ``(unicode_codepoint, glyph_id, ...)``.
    Ordinary Unicode values pass through. U+FFFD must have an explicitly
    audited glyph ID; otherwise conversion stops rather than guessing.
    """
    output: list[str] = []
    recovered_gids: list[int] = []
    replacements = 0
    for index, trace_char in enumerate(chars):
        if len(trace_char) < 2:
            raise VidehaProfileError(
                f"trace character {index} has fewer than two fields"
            )
        try:
            codepoint = int(trace_char[0])
            gid = int(trace_char[1])
        except (TypeError, ValueError) as exc:
            raise VidehaProfileError(
                f"trace character {index} has invalid codepoint/GID"
            ) from exc
        if codepoint != 0xFFFD:
            try:
                output.append(chr(codepoint))
            except ValueError as exc:
                raise VidehaProfileError(
                    f"trace character {index} has invalid Unicode codepoint {codepoint}"
                ) from exc
            continue
        replacements += 1
        recovered = mapping.get(gid)
        if recovered is None:
            raise UnknownJanakiGlyphError(
                f"unaudited Janaki replacement glyph ID {gid} at trace character {index}"
            )
        output.append(recovered)
        recovered_gids.append(gid)
    return TraceRecovery(
        text="".join(output),
        replacement_count=replacements,
        recovered_count=len(recovered_gids),
        recovered_gids=tuple(recovered_gids),
    )
