"""Canonical writing-system profiles for Newari / Nepal Bhasa OCR.

The profiles distinguish source-image appearance from OCR transcription encoding.
That distinction matters for Bhujimol and Ranjana: neither may be treated as a
Prachalit visual model merely because its transcription can use an encoded script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


ScriptCodepointRange = tuple[int, int]

DEVANAGARI_RANGES: tuple[ScriptCodepointRange, ...] = (
    (0x0900, 0x097F),
    (0xA8E0, 0xA8FF),
)
NEWA_RANGES: tuple[ScriptCodepointRange, ...] = ((0x11400, 0x1147F),)


@dataclass(frozen=True, slots=True)
class NewariWritingSystemProfile:
    """Static encoding and routing semantics for one Newari writing system."""

    profile_id: str
    display_name: str
    aliases: tuple[str, ...]
    source_unicode_status: str
    source_unicode_script: str | None
    source_unicode_ranges: tuple[ScriptCodepointRange, ...]
    transcription_unicode_script: str
    transcription_unicode_ranges: tuple[ScriptCodepointRange, ...]
    route_script: str
    visual_profile_required: bool
    input_policy: str
    output_policy: str


NEWARI_WRITING_SYSTEM_PROFILES: tuple[NewariWritingSystemProfile, ...] = (
    NewariWritingSystemProfile(
        profile_id="newa_prachalit",
        display_name="Newar / Prachalit",
        aliases=(
            "newa",
            "newar",
            "newar script",
            "newari script",
            "nepal lipi",
            "nepaalalipi",
            "prachalit",
            "prachalit newa",
        ),
        source_unicode_status="encoded",
        source_unicode_script="Newa",
        source_unicode_ranges=NEWA_RANGES,
        transcription_unicode_script="Newa",
        transcription_unicode_ranges=NEWA_RANGES,
        route_script="newa",
        visual_profile_required=True,
        input_policy="unicode_text_or_source_image",
        output_policy="native_newa_unicode",
    ),
    NewariWritingSystemProfile(
        profile_id="bhujimol",
        display_name="Bhujimol / Bhujinmol",
        aliases=("bhujimol", "bhujinmol", "bhujimol script", "bhujinmol script"),
        source_unicode_status="no_separate_script_encoding",
        source_unicode_script="Newa",
        source_unicode_ranges=NEWA_RANGES,
        transcription_unicode_script="Newa",
        transcription_unicode_ranges=NEWA_RANGES,
        route_script="bhujimol",
        visual_profile_required=True,
        input_policy="source_image_or_newa_text_with_explicit_bhujimol_style",
        output_policy="newa_unicode_transcription_with_source_style_metadata",
    ),
    NewariWritingSystemProfile(
        profile_id="ranjana",
        display_name="Ranjana",
        aliases=("ranjana", "ranjana script", "ranjana lipi", "rañjana", "rañjanā"),
        source_unicode_status="unencoded",
        source_unicode_script=None,
        source_unicode_ranges=(),
        transcription_unicode_script="Newa",
        transcription_unicode_ranges=NEWA_RANGES,
        route_script="ranjana",
        visual_profile_required=True,
        input_policy="source_image_or_audited_legacy_font_mapping_only",
        output_policy="newa_unicode_transcription_with_source_image_and_ranjana_style_metadata",
    ),
    NewariWritingSystemProfile(
        profile_id="devanagari",
        display_name="Devanagari",
        aliases=("devanagari", "deva", "devanagari script", "देवनागरी"),
        source_unicode_status="encoded",
        source_unicode_script="Devanagari",
        source_unicode_ranges=DEVANAGARI_RANGES,
        transcription_unicode_script="Devanagari",
        transcription_unicode_ranges=DEVANAGARI_RANGES,
        route_script="devanagari",
        visual_profile_required=True,
        input_policy="unicode_text_or_source_image",
        output_policy="native_devanagari_unicode",
    ),
)


def _normalize_alias(value: str) -> str:
    return re.sub(r"[\s_/-]+", " ", value.strip().casefold())


def _build_alias_index() -> dict[str, NewariWritingSystemProfile]:
    aliases: dict[str, NewariWritingSystemProfile] = {}
    for profile in NEWARI_WRITING_SYSTEM_PROFILES:
        values = (profile.profile_id, profile.display_name, *profile.aliases)
        for value in values:
            normalized = _normalize_alias(value)
            previous = aliases.get(normalized)
            if previous is not None and previous.profile_id != profile.profile_id:
                raise RuntimeError(
                    f"ambiguous Newari writing-system alias {value!r}: "
                    f"{previous.profile_id!r} and {profile.profile_id!r}"
                )
            aliases[normalized] = profile
    return aliases


NEWARI_WRITING_SYSTEM_ALIASES = _build_alias_index()
NEWARI_WRITING_SYSTEM_BY_ID = {profile.profile_id: profile for profile in NEWARI_WRITING_SYSTEM_PROFILES}

# These keys score a recognizer's Unicode transcription, not the visual identity
# of the source glyphs. In particular, Ranjana remains unencoded.
NEWARI_TRANSCRIPTION_ROUTING_RANGES: dict[str, tuple[ScriptCodepointRange, ...]] = {
    "newar": NEWA_RANGES,
    "newar-script": NEWA_RANGES,
    "newa": NEWA_RANGES,
    "prachalit": NEWA_RANGES,
    "newa/prachalit": NEWA_RANGES,
    "newa-prachalit": NEWA_RANGES,
    "bhujimol": NEWA_RANGES,
    "bhujinmol": NEWA_RANGES,
    "ranjana": NEWA_RANGES,
    "ranjana-newa-transcription": NEWA_RANGES,
}


def resolve_newari_writing_system(value: str) -> NewariWritingSystemProfile:
    """Resolve a user/config alias to one of the four supported profiles."""

    profile = NEWARI_WRITING_SYSTEM_ALIASES.get(_normalize_alias(value))
    if profile is None:
        supported = ", ".join(profile.profile_id for profile in NEWARI_WRITING_SYSTEM_PROFILES)
        raise ValueError(f"unsupported Newari writing system {value!r}; supported profiles: {supported}")
    return profile
