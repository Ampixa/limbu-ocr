"""Audits for multilingual OCR language registries."""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import re
import struct
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError


UNICODE_RANGE_RE = re.compile(r"^U\+[0-9A-F]{4,6}(?:\.\.U\+[0-9A-F]{4,6})?$")

LANGUAGE_REQUIRED_FIELDS = (
    "id",
    "name",
    "priority",
    "counts_toward_nepal_language_target",
    "first_target",
    "scripts",
    "font_candidates",
    "paddle_dictionary",
    "normalization",
    "corpora",
    "validation_sources",
    "gorkhapatra_language_labels",
)

SCRIPT_REQUIRED_FIELDS = ("script", "primary", "status", "unicode_ranges")
FONT_PATH_STATUSES = {"pending_render_audit", "exists_local", "schema_audit_pending"}
FONT_INVENTORY_SUFFIXES = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
CMAP_READABLE_FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}


@dataclass(slots=True)
class TargetLanguageReadiness:
    language_id: str
    name: str
    priority: int
    counts_toward_target: bool
    primary_scripts: list[str]
    gorkhapatra_labels: list[str]
    corpus_statuses: dict[str, int]
    validation_statuses: dict[str, int]
    synthesis_text_samples: int
    synthesis_text_audits: list[str]
    gates: dict[str, bool]
    missing_actions: list[str] = field(default_factory=list)

    @property
    def stage(self) -> str:
        if self.gates.get("synthesis_text_audit_passed"):
            return "synthesis_text_ready"
        if self.gates.get("corpus_candidate_declared") and self.gates.get("validation_source_declared"):
            return "source_candidates_declared"
        return "registry_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_id": self.language_id,
            "name": self.name,
            "priority": self.priority,
            "counts_toward_target": self.counts_toward_target,
            "primary_scripts": self.primary_scripts,
            "gorkhapatra_labels": self.gorkhapatra_labels,
            "corpus_statuses": self.corpus_statuses,
            "validation_statuses": self.validation_statuses,
            "synthesis_text_samples": self.synthesis_text_samples,
            "synthesis_text_audits": self.synthesis_text_audits,
            "gates": self.gates,
            "stage": self.stage,
            "missing_actions": self.missing_actions,
        }


@dataclass(slots=True)
class LanguageReadinessAudit:
    registry_path: str
    passed: bool
    target_language_count: int
    synthesis_ready_language_count: int
    required_synthesis_ready_language_count: int
    first_target_ids: list[str]
    languages: list[TargetLanguageReadiness]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": self.registry_path,
            "passed": self.passed,
            "target_language_count": self.target_language_count,
            "synthesis_ready_language_count": self.synthesis_ready_language_count,
            "required_synthesis_ready_language_count": self.required_synthesis_ready_language_count,
            "first_target_ids": self.first_target_ids,
            "warnings": self.warnings,
            "errors": self.errors,
            "languages": [language.to_dict() for language in self.languages],
        }


@dataclass(slots=True)
class FontCandidateAudit:
    language_id: str
    script: str
    font_name: str
    font_path: str
    status: str
    path_exists: bool
    supported_font_file: bool
    required_codepoints: int
    covered_codepoints: int
    coverage_ratio: float
    missing_sample: list[str] = field(default_factory=list)
    render_sample_path: str | None = None
    render_status: str = "not_requested"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_id": self.language_id,
            "script": self.script,
            "font_name": self.font_name,
            "font_path": self.font_path,
            "status": self.status,
            "path_exists": self.path_exists,
            "supported_font_file": self.supported_font_file,
            "required_codepoints": self.required_codepoints,
            "covered_codepoints": self.covered_codepoints,
            "coverage_ratio": self.coverage_ratio,
            "missing_sample": self.missing_sample,
            "render_sample_path": self.render_sample_path,
            "render_status": self.render_status,
            "error": self.error,
        }


@dataclass(slots=True)
class FontRenderabilityAudit:
    registry_path: str
    passed: bool
    language_count: int
    script_count: int
    candidate_count: int
    render_requested: bool
    pillow_available: bool
    passing_scripts: list[str] = field(default_factory=list)
    failing_scripts: list[str] = field(default_factory=list)
    candidates: list[FontCandidateAudit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": self.registry_path,
            "passed": self.passed,
            "language_count": self.language_count,
            "script_count": self.script_count,
            "candidate_count": self.candidate_count,
            "render_requested": self.render_requested,
            "pillow_available": self.pillow_available,
            "passing_scripts": self.passing_scripts,
            "failing_scripts": self.failing_scripts,
            "warnings": self.warnings,
            "errors": self.errors,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class FontInventoryItem:
    source_root: str
    path: str
    archive_member: str | None
    filename: str
    suffix: str
    sha256: str
    size_bytes: int
    cmap_readable: bool
    total_codepoints: int
    matched_scripts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root": self.source_root,
            "path": self.path,
            "archive_member": self.archive_member,
            "filename": self.filename,
            "suffix": self.suffix,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "cmap_readable": self.cmap_readable,
            "total_codepoints": self.total_codepoints,
            "matched_scripts": self.matched_scripts,
            "error": self.error,
        }


@dataclass(slots=True)
class FontInventoryAudit:
    registry_path: str
    source_roots: list[str]
    passed: bool
    font_count: int
    readable_font_count: int
    archive_count: int
    duplicate_sha_count: int
    items: list[FontInventoryItem]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None
    index_csv_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": self.registry_path,
            "source_roots": self.source_roots,
            "passed": self.passed,
            "font_count": self.font_count,
            "readable_font_count": self.readable_font_count,
            "archive_count": self.archive_count,
            "duplicate_sha_count": self.duplicate_sha_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(slots=True)
class LanguageRegistryAudit:
    registry_path: str
    registry_id: str
    passed: bool
    language_count: int
    counted_nepal_language_count: int
    first_target_ids: list[str]
    primary_scripts: dict[str, list[str]]
    missing_local_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": self.registry_path,
            "registry_id": self.registry_id,
            "passed": self.passed,
            "language_count": self.language_count,
            "counted_nepal_language_count": self.counted_nepal_language_count,
            "first_target_ids": self.first_target_ids,
            "primary_scripts": self.primary_scripts,
            "missing_local_paths": self.missing_local_paths,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(slots=True)
class PaddleDictionaryAudit:
    dictionary_path: str
    passed: bool
    character_count: int
    sha256: str
    required_ranges: list[str] = field(default_factory=list)
    required_character_count: int = 0
    missing_required_characters: list[str] = field(default_factory=list)
    label_file_paths: list[str] = field(default_factory=list)
    label_character_count: int = 0
    missing_label_characters: list[str] = field(default_factory=list)
    duplicate_characters: list[str] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dictionary_path": self.dictionary_path,
            "passed": self.passed,
            "character_count": self.character_count,
            "sha256": self.sha256,
            "required_ranges": self.required_ranges,
            "required_character_count": self.required_character_count,
            "missing_required_characters": self.missing_required_characters,
            "label_file_paths": self.label_file_paths,
            "label_character_count": self.label_character_count,
            "missing_label_characters": self.missing_label_characters,
            "duplicate_characters": self.duplicate_characters,
            "invalid_lines": self.invalid_lines,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def audit_language_registry(
    registry_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    min_nepal_languages: int = 10,
    require_limbu_first: bool = True,
    verify_local_paths: bool = True,
) -> LanguageRegistryAudit:
    path = Path(registry_path)
    payload = _load_json_object(path)
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise DataValidationError(f"language registry must contain a non-empty languages list: {path}")

    errors: list[str] = []
    warnings: list[str] = []
    missing_local_paths: list[str] = []
    ids: set[str] = set()
    priorities: set[int] = set()
    counted_ids: list[str] = []
    first_target_ids: list[str] = []
    primary_scripts: dict[str, list[str]] = {}

    for index, language in enumerate(languages):
        prefix = f"languages[{index}]"
        if not isinstance(language, dict):
            errors.append(f"{prefix}: entry must be an object")
            continue
        _validate_language_entry(prefix, language, errors, warnings)

        language_id = language.get("id")
        if isinstance(language_id, str):
            if language_id in ids:
                errors.append(f"{prefix}: duplicate id {language_id!r}")
            ids.add(language_id)

        priority = language.get("priority")
        if isinstance(priority, int):
            if priority in priorities:
                errors.append(f"{prefix}: duplicate priority {priority}")
            priorities.add(priority)

        if language.get("counts_toward_nepal_language_target") is True:
            if isinstance(language_id, str):
                counted_ids.append(language_id)
        elif language_id == "english":
            pass
        elif language.get("counts_toward_nepal_language_target") is False:
            warnings.append(f"{prefix}: non-English language is not counted toward Nepal target")

        if language.get("first_target") is True and isinstance(language_id, str):
            first_target_ids.append(language_id)

        scripts = language.get("scripts")
        if isinstance(language_id, str) and isinstance(scripts, list):
            primary_scripts[language_id] = [
                str(script.get("script")) for script in scripts if isinstance(script, dict) and script.get("primary") is True
            ]
            if not primary_scripts[language_id]:
                errors.append(f"{prefix}: at least one script must be primary")

        if verify_local_paths:
            missing_local_paths.extend(_collect_missing_local_paths(prefix, language))

    if len(counted_ids) < min_nepal_languages:
        errors.append(f"registry counts {len(counted_ids)} Nepal languages; minimum is {min_nepal_languages}")

    if "english" in counted_ids:
        errors.append("english must not count toward the Nepal-language target")

    if require_limbu_first:
        _validate_limbu_first(languages, errors)

    errors.extend(f"missing local path: {path_value}" for path_value in missing_local_paths)

    audit = LanguageRegistryAudit(
        registry_path=str(path),
        registry_id=str(payload.get("registry_id", "")),
        passed=not errors,
        language_count=len(languages),
        counted_nepal_language_count=len(counted_ids),
        first_target_ids=first_target_ids,
        primary_scripts=primary_scripts,
        missing_local_paths=missing_local_paths,
        warnings=warnings,
        errors=errors,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "language-registry-audit.json"
        md_path = out / "language-registry-audit.md"
        json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_language_registry_markdown(md_path, audit, languages)
        audit.summary_json_path = str(json_path)
        audit.summary_md_path = str(md_path)

    return audit


def audit_paddle_dictionary(
    dictionary_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    required_ranges: list[str] | None = None,
    label_file_paths: list[str | Path] | None = None,
    allow_space_char: bool = True,
) -> PaddleDictionaryAudit:
    path = Path(dictionary_path)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise DataValidationError(f"PaddleOCR dictionary does not exist: {path}") from exc
    text = data.decode("utf-8")
    lines = text.splitlines()
    characters: list[str] = []
    invalid_lines: list[str] = []
    duplicate_characters: list[str] = []
    seen: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []

    for index, line in enumerate(lines, start=1):
        if line == "":
            invalid_lines.append(f"line {index}: empty dictionary row")
            continue
        if len(line) != 1:
            invalid_lines.append(f"line {index}: expected one character, found {len(line)}")
            continue
        if line in seen:
            duplicate_characters.append(_format_character_for_report(line))
            continue
        seen.add(line)
        characters.append(line)

    if not characters:
        errors.append("dictionary contains no usable characters")
    if invalid_lines:
        errors.append(f"dictionary has {len(invalid_lines)} invalid lines")
    if duplicate_characters:
        errors.append(f"dictionary has {len(duplicate_characters)} duplicate characters")

    required_range_values = required_ranges or []
    required_codepoints = _expand_unicode_ranges(required_range_values)
    required_characters = {chr(codepoint) for codepoint in required_codepoints}
    missing_required = sorted(required_characters.difference(seen), key=ord)
    if missing_required:
        errors.append(f"dictionary is missing {len(missing_required)} required Unicode-range characters")

    label_file_values = label_file_paths or []
    label_characters = _read_paddle_label_characters(label_file_values)
    label_required_characters = set(label_characters)
    if allow_space_char and " " in label_required_characters and " " not in seen:
        label_required_characters.remove(" ")
        warnings.append("space label characters are covered by PaddleOCR use_space_char=true")
    missing_label = sorted(label_required_characters.difference(seen), key=ord)
    if missing_label:
        errors.append(f"dictionary is missing {len(missing_label)} label characters")

    if text and not text.endswith("\n"):
        warnings.append("dictionary file has no trailing newline")

    audit = PaddleDictionaryAudit(
        dictionary_path=str(path),
        passed=not errors,
        character_count=len(characters),
        sha256=hashlib.sha256(data).hexdigest(),
        required_ranges=list(required_range_values),
        required_character_count=len(required_characters),
        missing_required_characters=[_format_character_for_report(char) for char in missing_required],
        label_file_paths=[str(Path(value)) for value in label_file_values],
        label_character_count=len(label_characters),
        missing_label_characters=[_format_character_for_report(char) for char in missing_label],
        duplicate_characters=duplicate_characters,
        invalid_lines=invalid_lines,
        warnings=warnings,
        errors=errors,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "paddle-dictionary-audit.json"
        md_path = out / "paddle-dictionary-audit.md"
        json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_paddle_dictionary_markdown(md_path, audit)
        audit.summary_json_path = str(json_path)
        audit.summary_md_path = str(md_path)

    return audit


def audit_language_readiness(
    registry_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    synthesis_text_audit_paths: list[str | Path] | None = None,
    min_target_languages: int = 10,
    min_synthesis_ready_languages: int = 1,
    require_limbu_synthesis: bool = True,
) -> LanguageReadinessAudit:
    if min_target_languages < 1:
        raise DataValidationError("min_target_languages must be >= 1")
    if min_synthesis_ready_languages < 0:
        raise DataValidationError("min_synthesis_ready_languages must be >= 0")

    path = Path(registry_path)
    payload = _load_json_object(path)
    languages_payload = payload.get("languages")
    if not isinstance(languages_payload, list) or not languages_payload:
        raise DataValidationError(f"language registry must contain a non-empty languages list: {path}")

    synthesis_evidence = _load_synthesis_text_audit_evidence(synthesis_text_audit_paths or [])
    target_readiness: list[TargetLanguageReadiness] = []
    first_target_ids: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    for language in languages_payload:
        if not isinstance(language, dict):
            continue
        language_id = str(language.get("id") or "")
        if language.get("first_target") is True:
            first_target_ids.append(language_id)
        if language.get("counts_toward_nepal_language_target") is not True:
            continue
        readiness = _language_readiness(language, synthesis_evidence.get(language_id, []))
        target_readiness.append(readiness)

    target_readiness.sort(key=lambda item: item.priority)
    synthesis_ready_count = sum(1 for item in target_readiness if item.gates.get("synthesis_text_audit_passed"))

    if len(target_readiness) < min_target_languages:
        errors.append(f"registry has {len(target_readiness)} target languages; minimum is {min_target_languages}")
    if synthesis_ready_count < min_synthesis_ready_languages:
        errors.append(
            f"{synthesis_ready_count} target languages have passed synthesis-text audits; "
            f"minimum is {min_synthesis_ready_languages}"
        )
    if require_limbu_synthesis:
        limbu = next((item for item in target_readiness if item.language_id == "limbu"), None)
        if limbu is None:
            errors.append("Limbu is missing from target languages")
        elif not limbu.gates.get("synthesis_text_audit_passed"):
            errors.append("Limbu must have a passed synthesis-text audit before broader synthesis expansion")

    for item in target_readiness:
        if item.stage != "synthesis_text_ready":
            warnings.append(f"{item.language_id}: {item.stage}; next actions: {'; '.join(item.missing_actions)}")

    audit = LanguageReadinessAudit(
        registry_path=str(path),
        passed=not errors,
        target_language_count=len(target_readiness),
        synthesis_ready_language_count=synthesis_ready_count,
        required_synthesis_ready_language_count=min_synthesis_ready_languages,
        first_target_ids=first_target_ids,
        languages=target_readiness,
        warnings=warnings,
        errors=errors,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "language-readiness-audit.json"
        md_path = out / "language-readiness-audit.md"
        json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_language_readiness_markdown(md_path, audit)
        audit.summary_json_path = str(json_path)
        audit.summary_md_path = str(md_path)

    return audit


def audit_font_renderability(
    registry_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    primary_only: bool = True,
    min_coverage_ratio: float = 0.95,
    render_samples: bool = False,
) -> FontRenderabilityAudit:
    path = Path(registry_path)
    payload = _load_json_object(path)
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise DataValidationError(f"language registry must contain a non-empty languages list: {path}")
    if min_coverage_ratio < 0 or min_coverage_ratio > 1:
        raise DataValidationError("min_coverage_ratio must be between 0 and 1")

    out = Path(out_dir) if out_dir is not None else None
    sample_dir = out / "font-samples" if out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    if sample_dir is not None and render_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)

    pillow_available = _pillow_available()
    warnings: list[str] = []
    errors: list[str] = []
    candidates: list[FontCandidateAudit] = []
    script_keys: set[str] = set()
    passing_scripts: list[str] = []
    failing_scripts: list[str] = []

    if render_samples and not pillow_available:
        warnings.append("Pillow is not installed; sample rendering was requested but skipped")

    for language in languages:
        if not isinstance(language, dict):
            continue
        language_id = str(language.get("id", ""))
        font_candidates = [item for item in language.get("font_candidates", []) if isinstance(item, dict)]
        scripts = [item for item in language.get("scripts", []) if isinstance(item, dict)]
        for script in scripts:
            if primary_only and script.get("primary") is not True:
                continue
            script_name = str(script.get("script", ""))
            unicode_ranges = script.get("unicode_ranges", [])
            if not isinstance(unicode_ranges, list):
                unicode_ranges = []
            required = _expand_unicode_ranges(unicode_ranges)
            script_key = f"{language_id}:{script_name}"
            script_keys.add(script_key)
            script_candidates = [
                _audit_font_candidate(
                    language_id,
                    script_name,
                    required,
                    candidate,
                    sample_dir=sample_dir,
                    render_sample=render_samples and pillow_available,
                )
                for candidate in font_candidates
            ]
            candidates.extend(script_candidates)
            if any(candidate.coverage_ratio >= min_coverage_ratio for candidate in script_candidates):
                passing_scripts.append(script_key)
            else:
                failing_scripts.append(script_key)
                errors.append(f"{script_key}: no font candidate covers at least {min_coverage_ratio:.2%} of required codepoints")

    audit = FontRenderabilityAudit(
        registry_path=str(path),
        passed=not errors,
        language_count=len(languages),
        script_count=len(script_keys),
        candidate_count=len(candidates),
        render_requested=render_samples,
        pillow_available=pillow_available,
        passing_scripts=passing_scripts,
        failing_scripts=failing_scripts,
        candidates=candidates,
        warnings=warnings,
        errors=errors,
    )

    if out is not None:
        json_path = out / "font-renderability-audit.json"
        md_path = out / "font-renderability-audit.md"
        json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_font_renderability_markdown(md_path, audit)
        audit.summary_json_path = str(json_path)
        audit.summary_md_path = str(md_path)

    return audit


def inventory_fonts(
    registry_path: str | Path,
    roots: list[str | Path],
    out_dir: str | Path | None = None,
    *,
    min_script_coverage_ratio: float = 0.95,
) -> FontInventoryAudit:
    path = Path(registry_path)
    payload = _load_json_object(path)
    if min_script_coverage_ratio < 0 or min_script_coverage_ratio > 1:
        raise DataValidationError("min_script_coverage_ratio must be between 0 and 1")
    if not roots:
        raise DataValidationError("at least one font inventory root is required")
    script_requirements = _registry_script_requirements(payload)
    warnings: list[str] = []
    errors: list[str] = []
    items: list[FontInventoryItem] = []
    archive_count = 0
    source_roots: list[str] = []

    for root_value in roots:
        root = Path(root_value).expanduser()
        source_roots.append(str(root))
        if not root.exists():
            errors.append(f"font inventory root does not exist: {root}")
            continue
        if root.is_file():
            if root.suffix.lower() == ".zip":
                archive_count += 1
                items.extend(_inventory_font_zip(root, source_root=root, script_requirements=script_requirements, min_script_coverage_ratio=min_script_coverage_ratio))
            elif root.suffix.lower() in FONT_INVENTORY_SUFFIXES:
                items.append(_inventory_font_path(root, source_root=root, script_requirements=script_requirements, min_script_coverage_ratio=min_script_coverage_ratio))
            else:
                warnings.append(f"skipping unsupported inventory file: {root}")
            continue
        for child in sorted(root.rglob("*")):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix == ".zip":
                archive_count += 1
                items.extend(_inventory_font_zip(child, source_root=root, script_requirements=script_requirements, min_script_coverage_ratio=min_script_coverage_ratio))
            elif suffix in FONT_INVENTORY_SUFFIXES:
                items.append(_inventory_font_path(child, source_root=root, script_requirements=script_requirements, min_script_coverage_ratio=min_script_coverage_ratio))

    hashes: dict[str, int] = {}
    for item in items:
        hashes[item.sha256] = hashes.get(item.sha256, 0) + 1
    duplicate_sha_count = sum(1 for count in hashes.values() if count > 1)
    if not items:
        errors.append("no font files found in inventory roots")

    audit = FontInventoryAudit(
        registry_path=str(path),
        source_roots=source_roots,
        passed=not errors,
        font_count=len(items),
        readable_font_count=sum(1 for item in items if item.cmap_readable),
        archive_count=archive_count,
        duplicate_sha_count=duplicate_sha_count,
        items=items,
        warnings=warnings,
        errors=errors,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "font-inventory.json"
        md_path = out / "font-inventory.md"
        csv_path = out / "font-inventory.csv"
        json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_font_inventory_csv(csv_path, audit)
        _write_font_inventory_markdown(md_path, audit)
        audit.summary_json_path = str(json_path)
        audit.summary_md_path = str(md_path)
        audit.index_csv_path = str(csv_path)

    return audit


def _load_json_object(path: Path, *, label: str = "language registry") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_language_entry(prefix: str, language: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    for field_name in LANGUAGE_REQUIRED_FIELDS:
        if field_name not in language:
            errors.append(f"{prefix}: missing required field {field_name!r}")

    if "iso_639_3" not in language:
        warnings.append(f"{prefix}: iso_639_3 missing; set null only after confirming no code exists")
    elif language["iso_639_3"] is not None and not _is_non_empty_string(language["iso_639_3"]):
        errors.append(f"{prefix}: iso_639_3 must be a non-empty string or null")

    if not _is_non_empty_string(language.get("id")):
        errors.append(f"{prefix}: id must be a non-empty string")
    if not _is_non_empty_string(language.get("name")):
        errors.append(f"{prefix}: name must be a non-empty string")
    if not isinstance(language.get("priority"), int):
        errors.append(f"{prefix}: priority must be an integer")
    if not isinstance(language.get("counts_toward_nepal_language_target"), bool):
        errors.append(f"{prefix}: counts_toward_nepal_language_target must be boolean")
    if not isinstance(language.get("first_target"), bool):
        errors.append(f"{prefix}: first_target must be boolean")
    if not _is_non_empty_string(language.get("paddle_dictionary")):
        errors.append(f"{prefix}: paddle_dictionary must be a non-empty string")
    if not isinstance(language.get("gorkhapatra_language_labels"), list):
        errors.append(f"{prefix}: gorkhapatra_language_labels must be a list")

    _validate_scripts(prefix, language.get("scripts"), errors)
    _validate_non_empty_list_of_objects(prefix, "font_candidates", language.get("font_candidates"), errors)
    _validate_non_empty_list_of_objects(prefix, "corpora", language.get("corpora"), errors)
    _validate_non_empty_list_of_objects(prefix, "validation_sources", language.get("validation_sources"), errors)
    _validate_normalization(prefix, language.get("normalization"), errors)


def _validate_scripts(prefix: str, scripts: Any, errors: list[str]) -> None:
    if not isinstance(scripts, list) or not scripts:
        errors.append(f"{prefix}: scripts must be a non-empty list")
        return
    for script_index, script in enumerate(scripts):
        script_prefix = f"{prefix}.scripts[{script_index}]"
        if not isinstance(script, dict):
            errors.append(f"{script_prefix}: script entry must be an object")
            continue
        for field_name in SCRIPT_REQUIRED_FIELDS:
            if field_name not in script:
                errors.append(f"{script_prefix}: missing required field {field_name!r}")
        if not _is_non_empty_string(script.get("script")):
            errors.append(f"{script_prefix}: script must be a non-empty string")
        if not isinstance(script.get("primary"), bool):
            errors.append(f"{script_prefix}: primary must be boolean")
        if not _is_non_empty_string(script.get("status")):
            errors.append(f"{script_prefix}: status must be a non-empty string")
        ranges = script.get("unicode_ranges")
        if not isinstance(ranges, list):
            errors.append(f"{script_prefix}: unicode_ranges must be a list")
            continue
        if not ranges:
            if script.get("status") != "unencoded":
                errors.append(f"{script_prefix}: unicode_ranges may be empty only when status is 'unencoded'")
            continue
        for range_index, value in enumerate(ranges):
            if not isinstance(value, str) or not UNICODE_RANGE_RE.match(value):
                errors.append(f"{script_prefix}.unicode_ranges[{range_index}]: invalid Unicode range {value!r}")


def _validate_non_empty_list_of_objects(prefix: str, field_name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{prefix}: {field_name} must be a non-empty list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{prefix}.{field_name}[{index}]: entry must be an object")


def _validate_normalization(prefix: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: normalization must be an object")
        return
    if value.get("unicode_form") not in {"NFC", "NFKC", "none"}:
        errors.append(f"{prefix}: normalization.unicode_form must be NFC, NFKC, or none")
    if not _is_non_empty_string(value.get("policy")):
        errors.append(f"{prefix}: normalization.policy must be a non-empty string")


def _validate_limbu_first(languages: list[Any], errors: list[str]) -> None:
    first = languages[0] if languages else {}
    if not isinstance(first, dict):
        errors.append("first language entry must be Limbu")
        return
    if first.get("id") != "limbu":
        errors.append("first language entry must have id 'limbu'")
    if first.get("priority") != 1:
        errors.append("Limbu must have priority 1")
    if first.get("first_target") is not True:
        errors.append("Limbu must be marked first_target=true")
    scripts = first.get("scripts")
    if isinstance(scripts, list):
        primary_scripts = [script.get("script") for script in scripts if isinstance(script, dict) and script.get("primary") is True]
        if "Limbu / Sirijonga" not in primary_scripts:
            errors.append("Limbu must have primary script 'Limbu / Sirijonga'")


def _collect_missing_local_paths(prefix: str, language: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in ("corpora", "validation_sources", "font_candidates"):
        values = language.get(field_name)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            path_value = item.get("path")
            status = item.get("status")
            if not _should_verify_local_path(path_value, status, item.get("host")):
                continue
            candidate = Path(path_value)
            if not candidate.exists():
                missing.append(f"{prefix}.{field_name}[{index}].path={path_value}")
    return missing


def _load_synthesis_text_audit_evidence(paths: list[str | Path]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for value in paths:
        path = Path(value)
        payload = _load_json_object(path, label="synthesis text audit")
        if payload.get("passed") is not True:
            raise DataValidationError(f"synthesis text audit did not pass: {path}")
        language_counts = payload.get("language_counts")
        if not isinstance(language_counts, dict):
            raise DataValidationError(f"synthesis text audit missing language_counts object: {path}")
        for language_id, count in language_counts.items():
            if not isinstance(count, int):
                raise DataValidationError(f"synthesis text audit language count must be integer: {path}: {language_id}")
            evidence.setdefault(str(language_id), []).append(
                {
                    "path": str(path),
                    "sample_count": count,
                    "script_counts": payload.get("script_counts") if isinstance(payload.get("script_counts"), dict) else {},
                }
            )
    return evidence


def _language_readiness(language: dict[str, Any], synthesis_evidence: list[dict[str, Any]]) -> TargetLanguageReadiness:
    language_id = str(language.get("id") or "")
    scripts = [item for item in language.get("scripts", []) if isinstance(item, dict)]
    primary_scripts = [str(item.get("script")) for item in scripts if item.get("primary") is True]
    corpora = [item for item in language.get("corpora", []) if isinstance(item, dict)]
    validation_sources = [item for item in language.get("validation_sources", []) if isinstance(item, dict)]
    labels = [str(label) for label in language.get("gorkhapatra_language_labels", []) if str(label).strip()]
    synthesis_samples = sum(int(item["sample_count"]) for item in synthesis_evidence)
    gates = {
        "primary_script_declared": bool(primary_scripts),
        "encoded_primary_script": any(str(item.get("status")) == "encoded" for item in scripts if item.get("primary") is True),
        "font_candidate_declared": bool(language.get("font_candidates")),
        "paddle_dictionary_declared": _is_non_empty_string(language.get("paddle_dictionary")),
        "corpus_candidate_declared": any(_path_is_evidence_candidate(item.get("path")) for item in corpora),
        "validation_source_declared": any(_path_is_evidence_candidate(item.get("path")) for item in validation_sources),
        "gorkhapatra_labels_declared": bool(labels),
        "synthesis_text_audit_passed": synthesis_samples > 0,
    }
    missing_actions: list[str] = []
    if not gates["encoded_primary_script"]:
        missing_actions.append("verify encoded primary script and Unicode range")
    if not gates["font_candidate_declared"]:
        missing_actions.append("add at least one font candidate")
    if not gates["corpus_candidate_declared"]:
        missing_actions.append("find and audit real text corpus candidates")
    if not gates["validation_source_declared"]:
        missing_actions.append("find real document validation sources")
    if not gates["gorkhapatra_labels_declared"]:
        missing_actions.append("add Gorkhapatra/Naya Nepal language labels or justify absence")
    if not gates["synthesis_text_audit_passed"]:
        missing_actions.append("convert verified text and run audit-synthesis-text-manifest")

    return TargetLanguageReadiness(
        language_id=language_id,
        name=str(language.get("name") or ""),
        priority=int(language.get("priority")) if isinstance(language.get("priority"), int) else 9999,
        counts_toward_target=language.get("counts_toward_nepal_language_target") is True,
        primary_scripts=primary_scripts,
        gorkhapatra_labels=labels,
        corpus_statuses=_status_counts(corpora),
        validation_statuses=_status_counts(validation_sources),
        synthesis_text_samples=synthesis_samples,
        synthesis_text_audits=[str(item["path"]) for item in synthesis_evidence],
        gates=gates,
        missing_actions=missing_actions,
    )


def _path_is_evidence_candidate(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(value.strip()) and value.strip() != "TBD"


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _audit_font_candidate(
    language_id: str,
    script_name: str,
    required: set[int],
    candidate: dict[str, Any],
    *,
    sample_dir: Path | None,
    render_sample: bool,
) -> FontCandidateAudit:
    font_name = str(candidate.get("name", ""))
    font_path = str(candidate.get("path", ""))
    status = str(candidate.get("status", ""))
    path = Path(font_path) if font_path and not font_path.startswith(("TBD", "system_or_installable_font")) else None
    path_exists = path.exists() if path is not None else False
    supported_font_file = path is not None and path.suffix.lower() in {".ttf", ".otf", ".ttc"}
    covered: set[int] = set()
    error: str | None = None
    if path_exists and supported_font_file and path is not None:
        try:
            covered = _read_font_codepoints(path)
        except (OSError, ValueError, struct.error) as exc:
            error = str(exc)
    elif font_path == "TBD":
        error = "font path is TBD"
    elif not path_exists:
        error = "font path does not exist"
    elif not supported_font_file:
        error = "unsupported font file type"

    covered_required = required.intersection(covered)
    missing = sorted(required.difference(covered))[:20]
    coverage_ratio = len(covered_required) / len(required) if required else 0.0
    audit = FontCandidateAudit(
        language_id=language_id,
        script=script_name,
        font_name=font_name,
        font_path=font_path,
        status=status,
        path_exists=path_exists,
        supported_font_file=bool(supported_font_file),
        required_codepoints=len(required),
        covered_codepoints=len(covered_required),
        coverage_ratio=coverage_ratio,
        missing_sample=[_format_codepoint(codepoint) for codepoint in missing],
        error=error,
    )
    if render_sample:
        _render_font_sample(audit, required, path, sample_dir)
    return audit


def _registry_script_requirements(payload: dict[str, Any]) -> dict[str, set[int]]:
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise DataValidationError("language registry must contain a non-empty languages list")
    requirements: dict[str, set[int]] = {}
    for language in languages:
        if not isinstance(language, dict):
            continue
        language_id = str(language.get("id") or "")
        for script in language.get("scripts", []):
            if not isinstance(script, dict):
                continue
            script_name = str(script.get("script") or "")
            ranges = script.get("unicode_ranges")
            if not isinstance(ranges, list):
                continue
            key = f"{language_id}:{script_name}"
            requirements[key] = _expand_unicode_ranges(ranges)
    return requirements


def _inventory_font_zip(
    archive_path: Path,
    *,
    source_root: Path,
    script_requirements: dict[str, set[int]],
    min_script_coverage_ratio: float,
) -> list[FontInventoryItem]:
    items: list[FontInventoryItem] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(archive.namelist()):
                suffix = Path(name).suffix.lower()
                if suffix not in FONT_INVENTORY_SUFFIXES:
                    continue
                try:
                    data = archive.read(name)
                except (KeyError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
                    items.append(
                        _failed_font_inventory_item(
                            source_root=source_root,
                            path=archive_path,
                            archive_member=name,
                            data=b"",
                            suffix=suffix,
                            error=f"could not read archive member: {exc}",
                        )
                    )
                    continue
                items.append(
                    _inventory_font_bytes(
                        data,
                        path=archive_path,
                        source_root=source_root,
                        archive_member=name,
                        script_requirements=script_requirements,
                        min_script_coverage_ratio=min_script_coverage_ratio,
                    )
                )
    except (OSError, zipfile.BadZipFile) as exc:
        items.append(
            _failed_font_inventory_item(
                source_root=source_root,
                path=archive_path,
                archive_member=None,
                data=b"",
                suffix=".zip",
                error=f"could not read font archive: {exc}",
            )
        )
    return items


def _inventory_font_path(
    path: Path,
    *,
    source_root: Path,
    script_requirements: dict[str, set[int]],
    min_script_coverage_ratio: float,
) -> FontInventoryItem:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _failed_font_inventory_item(
            source_root=source_root,
            path=path,
            archive_member=None,
            data=b"",
            suffix=path.suffix.lower(),
            error=f"could not read font file: {exc}",
        )
    return _inventory_font_bytes(
        data,
        path=path,
        source_root=source_root,
        archive_member=None,
        script_requirements=script_requirements,
        min_script_coverage_ratio=min_script_coverage_ratio,
    )


def _inventory_font_bytes(
    data: bytes,
    *,
    path: Path,
    source_root: Path,
    archive_member: str | None,
    script_requirements: dict[str, set[int]],
    min_script_coverage_ratio: float,
) -> FontInventoryItem:
    suffix = Path(archive_member or path.name).suffix.lower()
    filename = Path(archive_member).name if archive_member is not None else path.name
    sha = hashlib.sha256(data).hexdigest()
    codepoints: set[int] = set()
    error: str | None = None
    cmap_readable = suffix in CMAP_READABLE_FONT_SUFFIXES
    if cmap_readable:
        try:
            codepoints = _read_font_codepoints_from_bytes(data)
        except (OSError, ValueError, struct.error) as exc:
            error = str(exc)
            cmap_readable = False
    else:
        error = "cmap coverage not implemented for compressed web font type"

    return FontInventoryItem(
        source_root=str(source_root),
        path=str(path),
        archive_member=archive_member,
        filename=filename,
        suffix=suffix,
        sha256=sha,
        size_bytes=len(data),
        cmap_readable=cmap_readable,
        total_codepoints=len(codepoints),
        matched_scripts=_matched_font_scripts(codepoints, script_requirements, min_script_coverage_ratio) if codepoints else [],
        error=error,
    )


def _failed_font_inventory_item(
    *,
    source_root: Path,
    path: Path,
    archive_member: str | None,
    data: bytes,
    suffix: str,
    error: str,
) -> FontInventoryItem:
    filename = Path(archive_member).name if archive_member is not None else path.name
    return FontInventoryItem(
        source_root=str(source_root),
        path=str(path),
        archive_member=archive_member,
        filename=filename,
        suffix=suffix,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        cmap_readable=False,
        total_codepoints=0,
        matched_scripts=[],
        error=error,
    )


def _matched_font_scripts(
    codepoints: set[int],
    script_requirements: dict[str, set[int]],
    min_script_coverage_ratio: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for key, required in sorted(script_requirements.items()):
        if not required:
            continue
        covered = len(required.intersection(codepoints))
        ratio = covered / len(required)
        if ratio >= min_script_coverage_ratio:
            language_id, script_name = key.split(":", 1)
            matches.append(
                {
                    "language_id": language_id,
                    "script": script_name,
                    "required_codepoints": len(required),
                    "covered_codepoints": covered,
                    "coverage_ratio": ratio,
                }
            )
    return matches


def _expand_unicode_ranges(values: list[Any]) -> set[int]:
    codepoints: set[int] = set()
    for value in values:
        if not isinstance(value, str) or not UNICODE_RANGE_RE.match(value):
            continue
        if ".." in value:
            start_text, end_text = value.split("..", 1)
            start = int(start_text.removeprefix("U+"), 16)
            end = int(end_text.removeprefix("U+"), 16)
            codepoints.update(range(start, end + 1))
        else:
            codepoints.add(int(value.removeprefix("U+"), 16))
    return {codepoint for codepoint in codepoints if not unicodedata.category(chr(codepoint)).startswith("C")}


def _read_paddle_label_characters(paths: list[str | Path]) -> set[str]:
    characters: set[str] = set()
    for value in paths:
        path = Path(value)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise DataValidationError(f"PaddleOCR label file does not exist: {path}") from exc
        for index, line in enumerate(lines, start=1):
            if not line:
                continue
            if "\t" not in line:
                raise DataValidationError(f"PaddleOCR label row missing tab separator: {path}:{index}")
            _, label = line.split("\t", 1)
            characters.update(char for char in unicodedata.normalize("NFC", label) if char not in {"\n", "\r", "\t"})
    return characters


def _format_character_for_report(char: str) -> str:
    codepoint = ord(char)
    name = unicodedata.name(char, "UNKNOWN")
    display = "SPACE" if char == " " else char
    return f"U+{codepoint:04X} {display} {name}"


def _read_font_codepoints(path: Path) -> set[int]:
    data = path.read_bytes()
    return _read_font_codepoints_from_bytes(data)


def _read_font_codepoints_from_bytes(data: bytes) -> set[int]:
    offsets = _font_face_offsets(data)
    codepoints: set[int] = set()
    for offset in offsets:
        codepoints.update(_read_face_codepoints(data, offset))
    return codepoints


def _font_face_offsets(data: bytes) -> list[int]:
    if len(data) < 12:
        raise ValueError("font file is too small")
    if data[:4] == b"ttcf":
        if len(data) < 12:
            raise ValueError("TTC header is truncated")
        count = struct.unpack_from(">L", data, 8)[0]
        offsets_end = 12 + count * 4
        if len(data) < offsets_end:
            raise ValueError("TTC face offset table is truncated")
        return [struct.unpack_from(">L", data, 12 + index * 4)[0] for index in range(count)]
    return [0]


def _read_face_codepoints(data: bytes, face_offset: int) -> set[int]:
    if face_offset + 12 > len(data):
        raise ValueError("font face offset is outside file")
    table_count = struct.unpack_from(">H", data, face_offset + 4)[0]
    table_dir = face_offset + 12
    cmap_offset: int | None = None
    cmap_length: int | None = None
    for index in range(table_count):
        record = table_dir + index * 16
        if record + 16 > len(data):
            raise ValueError("font table directory is truncated")
        tag = data[record : record + 4]
        if tag == b"cmap":
            cmap_offset = struct.unpack_from(">L", data, record + 8)[0]
            cmap_length = struct.unpack_from(">L", data, record + 12)[0]
            break
    if cmap_offset is None or cmap_length is None:
        return set()
    if cmap_offset + cmap_length > len(data):
        raise ValueError("cmap table is outside file")
    return _read_cmap_codepoints(data, cmap_offset)


def _read_cmap_codepoints(data: bytes, cmap_offset: int) -> set[int]:
    if cmap_offset + 4 > len(data):
        raise ValueError("cmap header is truncated")
    subtable_count = struct.unpack_from(">H", data, cmap_offset + 2)[0]
    subtables: list[tuple[int, int, int]] = []
    for index in range(subtable_count):
        record = cmap_offset + 4 + index * 8
        if record + 8 > len(data):
            raise ValueError("cmap encoding records are truncated")
        platform_id, encoding_id, sub_offset = struct.unpack_from(">HHL", data, record)
        subtables.append((platform_id, encoding_id, cmap_offset + sub_offset))
    preferred = sorted(subtables, key=lambda item: _cmap_preference(item[0], item[1]))
    all_codepoints: set[int] = set()
    for _, _, subtable_offset in preferred:
        if subtable_offset + 2 > len(data):
            continue
        fmt = struct.unpack_from(">H", data, subtable_offset)[0]
        try:
            if fmt == 12:
                all_codepoints.update(_read_cmap_format_12(data, subtable_offset))
            elif fmt == 4:
                all_codepoints.update(_read_cmap_format_4(data, subtable_offset))
        except (ValueError, struct.error):
            continue
        if all_codepoints:
            return all_codepoints
    return all_codepoints


def _cmap_preference(platform_id: int, encoding_id: int) -> int:
    if platform_id == 3 and encoding_id == 10:
        return 0
    if platform_id == 0:
        return 1
    if platform_id == 3 and encoding_id in {1, 0}:
        return 2
    return 3


def _read_cmap_format_12(data: bytes, offset: int) -> set[int]:
    if offset + 16 > len(data):
        raise ValueError("cmap format 12 header is truncated")
    group_count = struct.unpack_from(">L", data, offset + 12)[0]
    codepoints: set[int] = set()
    for index in range(group_count):
        record = offset + 16 + index * 12
        if record + 12 > len(data):
            raise ValueError("cmap format 12 groups are truncated")
        start, end, _glyph = struct.unpack_from(">LLL", data, record)
        if start <= end:
            codepoints.update(range(start, end + 1))
    return codepoints


def _read_cmap_format_4(data: bytes, offset: int) -> set[int]:
    if offset + 14 > len(data):
        raise ValueError("cmap format 4 header is truncated")
    length = struct.unpack_from(">H", data, offset + 2)[0]
    if offset + length > len(data):
        raise ValueError("cmap format 4 table is truncated")
    seg_count_x2 = struct.unpack_from(">H", data, offset + 6)[0]
    seg_count = seg_count_x2 // 2
    end_codes_offset = offset + 14
    start_codes_offset = end_codes_offset + seg_count * 2 + 2
    id_delta_offset = start_codes_offset + seg_count * 2
    id_range_offset_offset = id_delta_offset + seg_count * 2
    glyph_array_offset = id_range_offset_offset + seg_count * 2
    if glyph_array_offset > offset + length:
        raise ValueError("cmap format 4 arrays are truncated")

    codepoints: set[int] = set()
    for index in range(seg_count):
        end_code = struct.unpack_from(">H", data, end_codes_offset + index * 2)[0]
        start_code = struct.unpack_from(">H", data, start_codes_offset + index * 2)[0]
        id_delta = struct.unpack_from(">h", data, id_delta_offset + index * 2)[0]
        id_range_offset = struct.unpack_from(">H", data, id_range_offset_offset + index * 2)[0]
        if start_code == 0xFFFF and end_code == 0xFFFF:
            continue
        for codepoint in range(start_code, end_code + 1):
            if id_range_offset == 0:
                glyph_id = (codepoint + id_delta) & 0xFFFF
            else:
                glyph_offset = id_range_offset_offset + index * 2 + id_range_offset + (codepoint - start_code) * 2
                if glyph_offset + 2 > offset + length:
                    glyph_id = 0
                else:
                    glyph_id = struct.unpack_from(">H", data, glyph_offset)[0]
                    if glyph_id:
                        glyph_id = (glyph_id + id_delta) & 0xFFFF
            if glyph_id:
                codepoints.add(codepoint)
    return codepoints


def _render_font_sample(audit: FontCandidateAudit, required: set[int], path: Path | None, sample_dir: Path | None) -> None:
    if path is None or sample_dir is None or not path.exists():
        audit.render_status = "skipped_missing_font"
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        audit.render_status = "skipped_pillow_missing"
        return
    sample_text = _sample_text(required)
    if not sample_text:
        audit.render_status = "skipped_empty_sample"
        return
    try:
        font = ImageFont.truetype(str(path), 48)
        image = Image.new("RGB", (1200, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 42), sample_text, fill="black", font=font)
        filename = f"{_slug(audit.language_id)}-{_slug(audit.script)}-{_slug(audit.font_name)}.png"
        output = sample_dir / filename
        image.save(output)
        audit.render_sample_path = str(output)
        audit.render_status = "rendered"
    except (OSError, ValueError) as exc:
        audit.render_status = "failed"
        audit.error = f"{audit.error}; render failed: {exc}" if audit.error else f"render failed: {exc}"


def _sample_text(required: set[int]) -> str:
    chars = [chr(codepoint) for codepoint in sorted(required) if unicodedata.category(chr(codepoint))[0] in {"L", "M", "N"}]
    return "".join(chars[:24])


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def _format_codepoint(codepoint: int) -> str:
    return f"U+{codepoint:04X}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "sample"


def _should_verify_local_path(path_value: Any, status: Any, host: Any = None) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    if isinstance(host, str) and host and host != getpass.getuser():
        return False
    if status not in {"exists_local", "local_verified_directory", "schema_audit_pending"}:
        return False
    if path_value.startswith(("TBD", "huggingface:", "system_or_installable_font")):
        return False
    return path_value.startswith("/")


def _write_language_registry_markdown(path: Path, audit: LanguageRegistryAudit, languages: list[Any]) -> None:
    lines = [
        "# Language Registry Audit",
        "",
        f"- Registry: `{audit.registry_id}`",
        f"- Passed: `{audit.passed}`",
        f"- Languages: `{audit.language_count}`",
        f"- Nepal-language targets counted: `{audit.counted_nepal_language_count}`",
        f"- First targets: `{', '.join(audit.first_target_ids) or 'none'}`",
        "",
        "## Languages",
        "",
        "| Priority | ID | Name | Counted | Primary scripts | Validation labels |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    sorted_languages = sorted(
        [language for language in languages if isinstance(language, dict)],
        key=lambda language: language.get("priority") if isinstance(language.get("priority"), int) else 9999,
    )
    for language in sorted_languages:
        language_id = str(language.get("id", ""))
        primary_scripts = ", ".join(audit.primary_scripts.get(language_id, []))
        labels = ", ".join(str(label) for label in language.get("gorkhapatra_language_labels", []))
        lines.append(
            "| "
            f"{language.get('priority', '')} | "
            f"`{language_id}` | "
            f"{language.get('name', '')} | "
            f"`{language.get('counts_toward_nepal_language_target', '')}` | "
            f"{primary_scripts} | "
            f"{labels} |"
        )

    if audit.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    if audit.missing_local_paths:
        lines.extend(["", "## Missing Local Paths", ""])
        lines.extend(f"- `{missing}`" for missing in audit.missing_local_paths)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_language_readiness_markdown(path: Path, audit: LanguageReadinessAudit) -> None:
    lines = [
        "# Nepal Language Readiness Audit",
        "",
        f"- Passed: `{audit.passed}`",
        f"- Registry: `{audit.registry_path}`",
        f"- Target languages: `{audit.target_language_count}`",
        f"- Synthesis-ready languages: `{audit.synthesis_ready_language_count}`",
        f"- Required synthesis-ready languages: `{audit.required_synthesis_ready_language_count}`",
        f"- First targets: `{', '.join(audit.first_target_ids) or 'none'}`",
        "",
        "## Target Languages",
        "",
        "| Priority | ID | Primary scripts | Stage | Synthesis samples | Missing next action |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for language in audit.languages:
        missing = "; ".join(language.missing_actions) if language.missing_actions else "none"
        lines.append(
            "| "
            f"{language.priority} | "
            f"`{language.language_id}` | "
            f"{', '.join(language.primary_scripts)} | "
            f"`{language.stage}` | "
            f"{language.synthesis_text_samples} | "
            f"{missing} |"
        )

    lines.extend(["", "## Gate Details", ""])
    for language in audit.languages:
        lines.extend(
            [
                f"### {language.priority}. {language.language_id}",
                "",
                f"- name: {language.name}",
                f"- Gorkhapatra labels: {', '.join(language.gorkhapatra_labels) or 'none'}",
                f"- corpus statuses: `{json.dumps(language.corpus_statuses, ensure_ascii=False, sort_keys=True)}`",
                f"- validation statuses: `{json.dumps(language.validation_statuses, ensure_ascii=False, sort_keys=True)}`",
                f"- synthesis audits: {', '.join(f'`{item}`' for item in language.synthesis_text_audits) or 'none'}",
                "",
            ]
        )
    if audit.errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_font_renderability_markdown(path: Path, audit: FontRenderabilityAudit) -> None:
    lines = [
        "# Font Renderability Audit",
        "",
        f"- Passed: `{audit.passed}`",
        f"- Scripts checked: `{audit.script_count}`",
        f"- Font candidates checked: `{audit.candidate_count}`",
        f"- Render requested: `{audit.render_requested}`",
        f"- Pillow available: `{audit.pillow_available}`",
        "",
        "## Candidate Coverage",
        "",
        "| Language | Script | Font | Exists | Coverage | Render | Error |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for candidate in audit.candidates:
        error = candidate.error or ""
        lines.append(
            "| "
            f"`{candidate.language_id}` | "
            f"{candidate.script} | "
            f"{candidate.font_name} | "
            f"`{candidate.path_exists}` | "
            f"{candidate.coverage_ratio:.1%} | "
            f"`{candidate.render_status}` | "
            f"{error} |"
        )
    if audit.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_font_inventory_csv(path: Path, audit: FontInventoryAudit) -> None:
    fieldnames = [
        "source_root",
        "path",
        "archive_member",
        "filename",
        "suffix",
        "sha256",
        "size_bytes",
        "cmap_readable",
        "total_codepoints",
        "matched_scripts",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in audit.items:
            writer.writerow(
                {
                    "source_root": item.source_root,
                    "path": item.path,
                    "archive_member": item.archive_member or "",
                    "filename": item.filename,
                    "suffix": item.suffix,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "cmap_readable": item.cmap_readable,
                    "total_codepoints": item.total_codepoints,
                    "matched_scripts": ";".join(f"{match['language_id']}:{match['script']}={match['coverage_ratio']:.3f}" for match in item.matched_scripts),
                    "error": item.error or "",
                }
            )


def _write_font_inventory_markdown(path: Path, audit: FontInventoryAudit) -> None:
    lines = [
        "# Font Inventory",
        "",
        f"- Passed: `{audit.passed}`",
        f"- Registry: `{audit.registry_path}`",
        f"- Source roots: `{', '.join(audit.source_roots)}`",
        f"- Fonts found: `{audit.font_count}`",
        f"- Cmap-readable fonts: `{audit.readable_font_count}`",
        f"- ZIP archives scanned: `{audit.archive_count}`",
        f"- Duplicate SHA groups: `{audit.duplicate_sha_count}`",
        "",
        "## Fonts",
        "",
        "| File | Type | SHA-256 | Codepoints | Matched Scripts | Error |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in audit.items:
        member = f"!{item.archive_member}" if item.archive_member else ""
        location = f"`{item.path}{member}`"
        sha = item.sha256[:16]
        scripts = ", ".join(f"{match['language_id']}:{match['script']} {match['coverage_ratio']:.1%}" for match in item.matched_scripts) or ""
        lines.append(
            "| "
            f"{location} | "
            f"`{item.suffix}` | "
            f"`{sha}` | "
            f"{item.total_codepoints} | "
            f"{scripts} | "
            f"{item.error or ''} |"
        )
    if audit.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paddle_dictionary_markdown(path: Path, audit: PaddleDictionaryAudit) -> None:
    lines = [
        "# PaddleOCR Dictionary Audit",
        "",
        f"- Passed: `{audit.passed}`",
        f"- Dictionary: `{audit.dictionary_path}`",
        f"- SHA-256: `{audit.sha256}`",
        f"- Characters: `{audit.character_count}`",
        f"- Required ranges: `{', '.join(audit.required_ranges) or 'none'}`",
        f"- Required characters: `{audit.required_character_count}`",
        f"- Label files: `{', '.join(audit.label_file_paths) or 'none'}`",
        f"- Label characters: `{audit.label_character_count}`",
        f"- Missing required characters: `{len(audit.missing_required_characters)}`",
        f"- Missing label characters: `{len(audit.missing_label_characters)}`",
    ]
    if audit.missing_required_characters:
        lines.extend(["", "## Missing Required Characters", ""])
        lines.extend(f"- {character}" for character in audit.missing_required_characters)
    if audit.missing_label_characters:
        lines.extend(["", "## Missing Label Characters", ""])
        lines.extend(f"- {character}" for character in audit.missing_label_characters)
    if audit.invalid_lines:
        lines.extend(["", "## Invalid Lines", ""])
        lines.extend(f"- {line}" for line in audit.invalid_lines)
    if audit.duplicate_characters:
        lines.extend(["", "## Duplicate Characters", ""])
        lines.extend(f"- {character}" for character in audit.duplicate_characters)
    if audit.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
