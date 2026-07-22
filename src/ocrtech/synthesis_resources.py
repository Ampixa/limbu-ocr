"""Audits for real-text resources that may feed synthetic OCR generation."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, read_jsonl, sha256_file, sha256_text, write_manifest
from .normalization import normalize_ocr_text


TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".txt", ".tsv", ".xml", ".html", ".htm", ".php"})
STRUCTURED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".tsv", ".xml", ".xlsx"})
FONT_SUFFIXES = frozenset({".ttf", ".otf", ".ttc", ".woff", ".woff2"})
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".xlsx"})
LIMDIC_JSON_FIELDS = ("id", "headword_limbu", "desc_devanagari", "group", "definition", "ipa")
LIMDIC_FIREBASE_FIELDS = ("dId", "desc", "group", "mean")
LIMDIC_TEXT_FIELDS = ("headword_limbu", "desc_devanagari", "definition")
LIMDIC_FIREBASE_TEXT_FIELDS = ("dId", "desc", "mean")
TAMANG_NEPTAM_FIELDS = ("sentence_id", "nepali_sentences", "translation_tamang", "sentence_type", "Tense", "polarity")
TAMANG_NEPTAM_TEXT_FIELDS = ("translation_tamang", "nepali_sentences")
TAMANG_DICTIONARY_FIELDS = ("headword", "pos", "definition")
TAMANG_DICTIONARY_TEXT_FIELDS = ("headword", "definition")
MAGAR_DICTIONARY_FIELDS = ("headword", "ipa", "pos", "variant", "definition")
MAGAR_DICTIONARY_TEXT_FIELDS = ("headword", "variant", "definition")
BIBLE_BRAIN_MANIFEST_FIELDS = ("bible_id", "book_id", "chapter", "fileset_id", "fileset_type", "path", "status")
BIBLE_BRAIN_PLAIN_VERSE_FIELDS = (
    "book_id",
    "book_name",
    "chapter",
    "verse_start",
    "verse_end",
    "verse_text",
)
LIMBU_TAGGED_SENTENCE_FIELDS = ("sentence", "source", "script", "license")
TOOLKIT_PARALLEL_SUPPORTED_SUFFIXES = frozenset({".csv", ".jsonl", ".tsv"})
RENDER_DEGRADATION_PROFILES = frozenset(
    {
        "clean",
        "scan",
        "low_light",
        "uneven_light",
        "glare_light",
        "camera_left",
        "camera_right",
        "camera_top",
        "camera_bottom",
        "phone_photo",
    }
)
LIGHTING_DEGRADATION_PROFILES = frozenset({"low_light", "uneven_light", "glare_light", "phone_photo"})
CAMERA_DEGRADATION_PROFILES = frozenset({"camera_left", "camera_right", "camera_top", "camera_bottom", "phone_photo"})
SYNTHESIS_TEXT_REQUIRED_FIELDS = (
    "sample_id",
    "dataset",
    "split",
    "language",
    "script",
    "text",
    "text_sha256",
    "source_path",
    "source_sha256",
    "source_format",
    "source_row_id",
    "source_field",
    "source_schema",
    "source_schema_verified",
    "unicode_blocks",
    "metadata",
)


@dataclass(slots=True)
class TextSampleAudit:
    path: str
    relative_path: str
    suffix: str
    size_bytes: int
    sha256: str
    decoded_chars: int
    line_count: int
    unicode_blocks: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "suffix": self.suffix,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "decoded_chars": self.decoded_chars,
            "line_count": self.line_count,
            "unicode_blocks": self.unicode_blocks,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ResourceRootAudit:
    root: str
    label: str | None
    exists: bool
    total_files: int
    total_size_bytes: int
    suffix_counts: dict[str, int]
    text_file_count: int
    structured_file_count: int
    font_file_count: int
    document_file_count: int
    sampled_files: list[TextSampleAudit]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "label": self.label,
            "exists": self.exists,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "suffix_counts": self.suffix_counts,
            "text_file_count": self.text_file_count,
            "structured_file_count": self.structured_file_count,
            "font_file_count": self.font_file_count,
            "document_file_count": self.document_file_count,
            "sampled_files": [sample.to_dict() for sample in self.sampled_files],
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class SynthesisResourceAudit:
    roots: list[ResourceRootAudit]
    root_count: int
    existing_root_count: int
    total_files: int
    total_size_bytes: int
    output_json_path: str
    output_md_path: str
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.existing_root_count == self.root_count and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "root_count": self.root_count,
            "existing_root_count": self.existing_root_count,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "warnings": self.warnings,
            "roots": [root.to_dict() for root in self.roots],
        }


@dataclass(slots=True)
class SynthesisTextEntry:
    sample_id: str
    dataset: str
    split: str
    language: str
    script: str
    text: str
    text_sha256: str
    source_path: str
    source_sha256: str
    source_format: str
    source_row_id: str
    source_field: str
    source_schema: str
    source_schema_verified: bool
    unicode_blocks: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "language": self.language,
            "script": self.script,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "source_row_id": self.source_row_id,
            "source_field": self.source_field,
            "source_schema": self.source_schema,
            "source_schema_verified": self.source_schema_verified,
            "unicode_blocks": self.unicode_blocks,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SynthesisTextPrepareSummary:
    manifest_path: str
    summary_json_path: str
    summary_md_path: str
    sample_count: int
    source_count: int
    rejected_count: int
    duplicate_count: int
    field_counts: dict[str, int]
    script_counts: dict[str, int]
    source_paths: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "sample_count": self.sample_count,
            "source_count": self.source_count,
            "rejected_count": self.rejected_count,
            "duplicate_count": self.duplicate_count,
            "field_counts": self.field_counts,
            "script_counts": self.script_counts,
            "source_paths": self.source_paths,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class SynthesisTextManifestAudit:
    manifest_path: str
    output_json_path: str
    output_md_path: str
    sample_count: int
    language_counts: dict[str, int]
    script_counts: dict[str, int]
    source_schema_counts: dict[str, int]
    license_status_counts: dict[str, int]
    claim_evidence_eligible_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "sample_count": self.sample_count,
            "language_counts": self.language_counts,
            "script_counts": self.script_counts,
            "source_schema_counts": self.source_schema_counts,
            "license_status_counts": self.license_status_counts,
            "claim_evidence_eligible_count": self.claim_evidence_eligible_count,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class SynthesisTextPromotionAudit:
    manifest_path: str
    output_json_path: str
    output_md_path: str
    sample_count: int
    basic_audit_passed: bool
    overlap_count: int
    overlap_counts_by_source: dict[str, int]
    split_policy_counts: dict[str, int]
    license_status_counts: dict[str, int]
    require_reviewed_license: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overlap_examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.basic_audit_passed and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "sample_count": self.sample_count,
            "basic_audit_passed": self.basic_audit_passed,
            "overlap_count": self.overlap_count,
            "overlap_counts_by_source": self.overlap_counts_by_source,
            "split_policy_counts": self.split_policy_counts,
            "license_status_counts": self.license_status_counts,
            "require_reviewed_license": self.require_reviewed_license,
            "issues": self.issues,
            "warnings": self.warnings,
            "overlap_examples": self.overlap_examples,
        }


@dataclass(slots=True)
class SynthesisTextSplitSummary:
    train_manifest: str
    eval_manifest: str
    train_count: int
    eval_count: int
    group_count: int
    group_by: str
    seed: int
    eval_ratio: float
    split_policy: str
    train_promotion_passed: bool
    eval_promotion_passed: bool
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    @property
    def passed(self) -> bool:
        return self.train_count > 0 and self.eval_count > 0 and self.train_promotion_passed and self.eval_promotion_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "group_count": self.group_count,
            "group_by": self.group_by,
            "seed": self.seed,
            "eval_ratio": self.eval_ratio,
            "split_policy": self.split_policy,
            "train_promotion_passed": self.train_promotion_passed,
            "eval_promotion_passed": self.eval_promotion_passed,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class RenderedLineSummary:
    manifest_path: str
    label_path: str
    summary_json_path: str
    summary_md_path: str
    image_dir: str
    sample_count: int
    skipped_count: int
    script_counts: dict[str, int]
    degradation_counts: dict[str, int]
    font_path: str | None
    font_sha256: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "label_path": self.label_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
            "image_dir": self.image_dir,
            "sample_count": self.sample_count,
            "skipped_count": self.skipped_count,
            "script_counts": self.script_counts,
            "degradation_counts": self.degradation_counts,
            "font_path": self.font_path,
            "font_sha256": self.font_sha256,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RenderedLineManifestAudit:
    manifest_path: str
    output_json_path: str
    output_md_path: str
    sample_count: int
    script_counts: dict[str, int]
    font_counts: dict[str, int]
    claim_evidence_eligible_count: int
    font_readiness_report: str | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "sample_count": self.sample_count,
            "script_counts": self.script_counts,
            "font_counts": self.font_counts,
            "claim_evidence_eligible_count": self.claim_evidence_eligible_count,
            "font_readiness_report": self.font_readiness_report,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RenderedLineSplitSummary:
    train_manifest: str
    train_label_path: str
    train_audit_path: str
    eval_manifest: str
    eval_label_path: str
    eval_audit_path: str
    train_count: int
    eval_count: int
    train_audit_passed: bool
    eval_audit_passed: bool
    output_json_path: str
    output_md_path: str
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.train_count > 0 and self.eval_count > 0 and self.train_audit_passed and self.eval_audit_passed and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "train_manifest": self.train_manifest,
            "train_label_path": self.train_label_path,
            "train_audit_path": self.train_audit_path,
            "eval_manifest": self.eval_manifest,
            "eval_label_path": self.eval_label_path,
            "eval_audit_path": self.eval_audit_path,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "train_audit_passed": self.train_audit_passed,
            "eval_audit_passed": self.eval_audit_passed,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RenderedDegradationSplitAudit:
    train_manifest: str
    eval_manifest: str
    output_json_path: str
    output_md_path: str
    train_count: int
    eval_count: int
    expected_profiles: list[str]
    train_profile_counts: dict[str, int]
    eval_profile_counts: dict[str, int]
    train_text_hash_count: int
    eval_text_hash_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.train_count > 0 and self.eval_count > 0 and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "expected_profiles": self.expected_profiles,
            "train_profile_counts": self.train_profile_counts,
            "eval_profile_counts": self.eval_profile_counts,
            "train_text_hash_count": self.train_text_hash_count,
            "eval_text_hash_count": self.eval_text_hash_count,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def audit_synthesis_resources(
    roots: list[str | Path],
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    max_files_per_root: int = 0,
    max_text_samples_per_root: int = 20,
    sample_bytes: int = 64_000,
) -> SynthesisResourceAudit:
    if not roots:
        raise DataValidationError("at least one synthesis resource root is required")
    if labels is not None and len(labels) not in {0, len(roots)}:
        raise DataValidationError("labels must be omitted or match the number of roots")
    if max_files_per_root < 0:
        raise DataValidationError("max_files_per_root must be >= 0")
    if max_text_samples_per_root < 0:
        raise DataValidationError("max_text_samples_per_root must be >= 0")
    if sample_bytes < 1:
        raise DataValidationError("sample_bytes must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root_audits: list[ResourceRootAudit] = []
    warnings: list[str] = []
    label_values = labels or []
    for index, root_value in enumerate(roots):
        label = label_values[index] if label_values else None
        root = Path(root_value).expanduser()
        root_audit = _audit_resource_root(
            root,
            label=label,
            max_files=max_files_per_root,
            max_text_samples=max_text_samples_per_root,
            sample_bytes=sample_bytes,
        )
        root_audits.append(root_audit)
        warnings.extend(f"{root}: {warning}" for warning in root_audit.warnings)

    json_path = out / "synthesis-resource-audit.json"
    md_path = out / "synthesis-resource-audit.md"
    audit = SynthesisResourceAudit(
        roots=root_audits,
        root_count=len(root_audits),
        existing_root_count=sum(1 for root in root_audits if root.exists),
        total_files=sum(root.total_files for root in root_audits),
        total_size_bytes=sum(root.total_size_bytes for root in root_audits),
        output_json_path=str(json_path),
        output_md_path=str(md_path),
        warnings=warnings,
    )
    _write_synthesis_resource_audit(audit)
    return audit


def audit_synthesis_text_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    require_languages: list[str] | None = None,
    require_scripts: list[str] | None = None,
    min_samples: int = 1,
    allow_claim_evidence: bool = False,
) -> SynthesisTextManifestAudit:
    if min_samples < 0:
        raise DataValidationError("min_samples must be >= 0")
    path = Path(manifest_path)
    if not path.is_file():
        raise DataValidationError(f"synthesis text manifest does not exist: {path}")

    issues: list[str] = []
    warnings: list[str] = []
    language_counts: dict[str, int] = {}
    script_counts: dict[str, int] = {}
    source_schema_counts: dict[str, int] = {}
    license_status_counts: dict[str, int] = {}
    claim_count = 0
    sample_count = 0
    seen_sample_ids: set[str] = set()

    for row_number, row in enumerate(read_jsonl(path), start=1):
        sample_count += 1
        sample_id = str(row.get("sample_id") or f"row-{row_number}")
        prefix = f"{sample_id}: "
        missing = [field for field in SYNTHESIS_TEXT_REQUIRED_FIELDS if field not in row]
        if missing:
            issues.append(f"{prefix}missing required fields: {', '.join(missing)}")

        if sample_id in seen_sample_ids:
            issues.append(f"{prefix}duplicate sample_id")
        seen_sample_ids.add(sample_id)

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{prefix}text must be a non-empty string")
            text_value = "" if text is None else str(text)
        else:
            text_value = text

        text_hash = row.get("text_sha256")
        if not isinstance(text_hash, str) or not text_hash:
            issues.append(f"{prefix}missing text_sha256")
        elif sha256_text(text_value) != text_hash:
            issues.append(f"{prefix}text_sha256 mismatch")

        unicode_blocks = row.get("unicode_blocks")
        if not isinstance(unicode_blocks, dict):
            issues.append(f"{prefix}unicode_blocks must be an object")
        elif dict(sorted(unicode_blocks.items())) != dict(sorted(_unicode_block_counts(text_value).items())):
            issues.append(f"{prefix}unicode_blocks mismatch")

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            issues.append(f"{prefix}metadata must be an object")
            metadata = {}

        language = str(row.get("language") or "")
        script = str(row.get("script") or "")
        schema = str(row.get("source_schema") or "")
        license_status = str(metadata.get("license_status") or "")
        for field_name in (
            "sample_id",
            "dataset",
            "split",
            "language",
            "script",
            "source_path",
            "source_sha256",
            "source_format",
            "source_row_id",
            "source_field",
            "source_schema",
        ):
            if not str(row.get(field_name) or "").strip():
                issues.append(f"{prefix}missing or empty field: {field_name}")

        if row.get("source_schema_verified") is not True:
            issues.append(f"{prefix}source_schema_verified must be true")
        if metadata.get("synthesis_training_candidate") is not True:
            issues.append(f"{prefix}synthesis_training_candidate must be true")
        if not license_status:
            issues.append(f"{prefix}metadata.license_status is required")
        if metadata.get("claim_evidence_eligible") is not False:
            claim_count += 1
            if not allow_claim_evidence:
                issues.append(f"{prefix}synthesis text must have claim_evidence_eligible=false")

        language_counts[language or "missing"] = language_counts.get(language or "missing", 0) + 1
        script_counts[script or "missing"] = script_counts.get(script or "missing", 0) + 1
        source_schema_counts[schema or "missing"] = source_schema_counts.get(schema or "missing", 0) + 1
        license_status_counts[license_status or "missing"] = license_status_counts.get(license_status or "missing", 0) + 1

    if sample_count < min_samples:
        issues.append(f"sample count {sample_count} is below required minimum {min_samples}")
    if sample_count == 0:
        issues.append("synthesis text manifest is empty")

    for language in require_languages or []:
        if language not in language_counts:
            issues.append(f"required language missing: {language}")
    for script in require_scripts or []:
        if script not in script_counts:
            issues.append(f"required script missing: {script}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = SynthesisTextManifestAudit(
        manifest_path=str(path),
        output_json_path=str(out / "synthesis-text-manifest-audit.json"),
        output_md_path=str(out / "synthesis-text-manifest-audit.md"),
        sample_count=sample_count,
        language_counts=dict(sorted(language_counts.items())),
        script_counts=dict(sorted(script_counts.items())),
        source_schema_counts=dict(sorted(source_schema_counts.items())),
        license_status_counts=dict(sorted(license_status_counts.items())),
        claim_evidence_eligible_count=claim_count,
        issues=issues,
        warnings=warnings,
    )
    _write_synthesis_text_manifest_audit(audit)
    return audit


def audit_synthesis_text_promotion(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    exclude_paths: list[str | Path] | None = None,
    require_reviewed_license: bool = False,
) -> SynthesisTextPromotionAudit:
    path = Path(manifest_path)
    if not path.is_file():
        raise DataValidationError(f"synthesis text manifest does not exist: {path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    basic_audit = audit_synthesis_text_manifest(path, out / "basic-manifest-audit")
    rows = list(read_jsonl(path))
    issues = list(basic_audit.issues)
    warnings = list(basic_audit.warnings)
    split_policy_counts: dict[str, int] = {}
    license_status_counts: dict[str, int] = {}
    candidate_hashes: dict[str, str] = {}

    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or f"row-{row_number}")
        text_hash = _row_text_sha256(row)
        if text_hash:
            candidate_hashes[text_hash] = sample_id
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        split_policy = str(metadata.get("split_policy") or "").strip()
        license_status = str(metadata.get("license_status") or "").strip()
        if not split_policy:
            issues.append(f"{sample_id}: metadata.split_policy is required before synthesis promotion")
            split_policy = "missing"
        if require_reviewed_license and license_status in {"", "pending_review"}:
            issues.append(f"{sample_id}: reviewed license_status is required before synthesis promotion")
        license_status_counts[license_status or "missing"] = license_status_counts.get(license_status or "missing", 0) + 1
        split_policy_counts[split_policy] = split_policy_counts.get(split_policy, 0) + 1

    overlap_counts_by_source: dict[str, int] = {}
    overlap_examples: list[dict[str, str]] = []
    for exclude_value in exclude_paths or []:
        exclude_path = Path(exclude_value).expanduser()
        blocked_hashes = _collect_text_hashes_from_exclusion(exclude_path)
        overlap_hashes = sorted(set(candidate_hashes) & set(blocked_hashes))
        if overlap_hashes:
            overlap_counts_by_source[str(exclude_path)] = len(overlap_hashes)
            issues.append(f"text overlap with excluded source {exclude_path}: {len(overlap_hashes)} hash(es)")
            for text_hash in overlap_hashes[:10]:
                overlap_examples.append(
                    {
                        "candidate_sample_id": candidate_hashes[text_hash],
                        "text_sha256": text_hash,
                        "excluded_source": str(exclude_path),
                    }
                )

    audit = SynthesisTextPromotionAudit(
        manifest_path=str(path),
        output_json_path=str(out / "synthesis-text-promotion-audit.json"),
        output_md_path=str(out / "synthesis-text-promotion-audit.md"),
        sample_count=len(rows),
        basic_audit_passed=basic_audit.passed,
        overlap_count=sum(overlap_counts_by_source.values()),
        overlap_counts_by_source=dict(sorted(overlap_counts_by_source.items())),
        split_policy_counts=dict(sorted(split_policy_counts.items())),
        license_status_counts=dict(sorted(license_status_counts.items())),
        require_reviewed_license=require_reviewed_license,
        issues=issues,
        warnings=warnings,
        overlap_examples=overlap_examples,
    )
    _write_synthesis_text_promotion_audit(audit)
    return audit


def split_synthesis_text_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    eval_ratio: float = 0.15,
    seed: int = 13,
    group_by: str = "text_sha256",
    train_name: str = "train.jsonl",
    eval_name: str = "eval.jsonl",
    exclude_paths: list[str | Path] | None = None,
    require_reviewed_license: bool = False,
) -> SynthesisTextSplitSummary:
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise DataValidationError("eval_ratio must be between 0 and 1")
    rows = list(read_jsonl(Path(manifest_path)))
    if len(rows) < 2:
        raise DataValidationError("split-synthesis-text requires at least two rows")
    groups = _group_synthesis_text_rows(rows, group_by=group_by)
    if len(groups) < 2:
        raise DataValidationError("split-synthesis-text requires at least two unique text groups")
    rng = random.Random(seed)
    group_items = sorted(groups.items())
    rng.shuffle(group_items)
    eval_group_count = max(1, round(len(group_items) * eval_ratio))
    eval_group_count = min(eval_group_count, len(group_items) - 1)
    eval_keys = {key for key, _ in group_items[:eval_group_count]}

    split_policy = f"split_synthesis_text:{group_by}:seed={seed}:eval_ratio={eval_ratio:g}"
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for group_key, group_rows in groups.items():
        target = eval_rows if group_key in eval_keys else train_rows
        split_name = "eval" if group_key in eval_keys else "train"
        target.extend(_with_synthesis_split(row, split_name, split_policy) for row in group_rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / train_name
    eval_path = out / eval_name
    _write_synthesis_text_rows(train_rows, train_path)
    _write_synthesis_text_rows(eval_rows, eval_path)
    train_promotion = audit_synthesis_text_promotion(
        train_path,
        out / "train-promotion-audit",
        exclude_paths=exclude_paths,
        require_reviewed_license=require_reviewed_license,
    )
    eval_promotion = audit_synthesis_text_promotion(
        eval_path,
        out / "eval-promotion-audit",
        exclude_paths=[train_path, *(exclude_paths or [])],
        require_reviewed_license=require_reviewed_license,
    )
    if not train_promotion.passed:
        raise DataValidationError(f"train synthesis split failed promotion audit: {train_promotion.output_json_path}")
    if not eval_promotion.passed:
        raise DataValidationError(f"eval synthesis split failed promotion audit: {eval_promotion.output_json_path}")
    warnings: list[str] = []
    if not train_rows:
        warnings.append("train split is empty")
    if not eval_rows:
        warnings.append("eval split is empty")
    summary = SynthesisTextSplitSummary(
        train_manifest=str(train_path),
        eval_manifest=str(eval_path),
        train_count=len(train_rows),
        eval_count=len(eval_rows),
        group_count=len(groups),
        group_by=group_by,
        seed=seed,
        eval_ratio=eval_ratio,
        split_policy=split_policy,
        train_promotion_passed=train_promotion.passed,
        eval_promotion_passed=eval_promotion.passed,
        warnings=warnings,
        summary_json_path=str(out / "synthesis-text-split-summary.json"),
        summary_md_path=str(out / "synthesis-text-split-summary.md"),
    )
    _write_synthesis_text_split_summary(summary)
    return summary


def prepare_limbu_limdic_text(
    sources: list[str | Path],
    output_dir: str | Path,
    *,
    split: str = "train",
    dataset: str = "limbu-limdic",
    min_text_chars: int = 1,
) -> SynthesisTextPrepareSummary:
    if not sources:
        raise DataValidationError("at least one Limdic source path is required")
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []
    for source_value in sources:
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            raise DataValidationError(f"Limdic source path does not exist: {source_path}")
        source_paths.append(str(source_path))
        source_hash = sha256_file(source_path)
        rows, schema_name, source_format = _load_limdic_rows(source_path)
        for row_index, row in enumerate(rows, start=1):
            row_id = str(row.get("id") or row.get("_firebase_key") or row_index)
            for field_name, raw_text in _limdic_text_fields(row):
                text = _clean_synthesis_text(raw_text)
                if len(text) < min_text_chars:
                    rejected_count += 1
                    continue
                script = _dominant_script(text)
                key = (field_name, script, sha256_text(text))
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sample_id = f"limdic-{len(entries) + 1:06d}"
                entries.append(
                    SynthesisTextEntry(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        language="limbu",
                        script=script,
                        text=text,
                        text_sha256=sha256_text(text),
                        source_path=str(source_path),
                        source_sha256=source_hash,
                        source_format=source_format,
                        source_row_id=row_id,
                        source_field=field_name,
                        source_schema=schema_name,
                        source_schema_verified=True,
                        unicode_blocks=_unicode_block_counts(text),
                        metadata={
                            "source": "LTK Limdic",
                            "source_kind": "real_text_inventory_derived",
                            "claim_evidence_eligible": False,
                            "synthesis_training_candidate": True,
                            "license_status": "pending_review",
                            "normalization": "NFC_control_chars_removed",
                            "group": row.get("group"),
                        },
                    )
                )
    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "limbu-limdic-text.jsonl"
    summary_json_path = out / "limbu-limdic-text-summary.json"
    summary_md_path = out / "limbu-limdic-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(sources),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def prepare_tamang_text(
    sources: list[str | Path],
    output_dir: str | Path,
    *,
    split: str = "train",
    dataset: str = "tamang-real-text",
    min_text_chars: int = 1,
    limit_per_source: int = 0,
) -> SynthesisTextPrepareSummary:
    if not sources:
        raise DataValidationError("at least one Tamang source path is required")
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    if limit_per_source < 0:
        raise DataValidationError("limit_per_source must be >= 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []

    for source_value in sources:
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            raise DataValidationError(f"Tamang source path does not exist: {source_path}")
        source_paths.append(str(source_path))
        source_hash = sha256_file(source_path)
        rows, schema_name, source_format = _load_tamang_rows(source_path)
        emitted_for_source = 0
        for row_index, row in enumerate(rows, start=1):
            row_id = str(row.get("sentence_id") or row.get("row_id") or row_index)
            for field_name, raw_text in _tamang_text_fields(row, schema_name):
                text = _clean_synthesis_text(raw_text)
                if len(text) < min_text_chars:
                    rejected_count += 1
                    continue
                script = _dominant_script(text)
                text_hash = sha256_text(text)
                key = (field_name, script, text_hash)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sample_id = f"tamang-{len(entries) + 1:07d}"
                entries.append(
                    SynthesisTextEntry(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        language="tamang",
                        script=script,
                        text=text,
                        text_sha256=text_hash,
                        source_path=str(source_path),
                        source_sha256=source_hash,
                        source_format=source_format,
                        source_row_id=row_id,
                        source_field=field_name,
                        source_schema=schema_name,
                        source_schema_verified=True,
                        unicode_blocks=_unicode_block_counts(text),
                        metadata={
                            "source": _tamang_source_name(schema_name),
                            "source_kind": "real_text_inventory_derived",
                            "claim_evidence_eligible": False,
                            "synthesis_training_candidate": True,
                            "license_status": "pending_review",
                            "normalization": "NFC_control_chars_removed",
                            "sentence_type": row.get("sentence_type"),
                            "tense": row.get("Tense"),
                            "polarity": row.get("polarity"),
                            "pos": row.get("pos"),
                        },
                    )
                )
                emitted_for_source += 1
                if limit_per_source and emitted_for_source >= limit_per_source:
                    break
            if limit_per_source and emitted_for_source >= limit_per_source:
                break

    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "tamang-text.jsonl"
    summary_json_path = out / "tamang-text-summary.json"
    summary_md_path = out / "tamang-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(sources),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def prepare_magar_text(
    sources: list[str | Path],
    output_dir: str | Path,
    *,
    split: str = "train",
    dataset: str = "magar-real-text",
    min_text_chars: int = 1,
    limit_per_source: int = 0,
) -> SynthesisTextPrepareSummary:
    if not sources:
        raise DataValidationError("at least one Magar source path is required")
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    if limit_per_source < 0:
        raise DataValidationError("limit_per_source must be >= 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []

    for source_value in sources:
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            raise DataValidationError(f"Magar source path does not exist: {source_path}")
        source_paths.append(str(source_path))
        source_hash = sha256_file(source_path)
        rows, schema_name, source_format = _load_magar_rows(source_path)
        emitted_for_source = 0
        for row_index, row in enumerate(rows, start=1):
            row_id = str(row.get("row_id") or row_index)
            for field_name, raw_text in _magar_text_fields(row):
                text = _clean_synthesis_text(raw_text)
                if len(text) < min_text_chars:
                    rejected_count += 1
                    continue
                script = _dominant_script(text)
                text_hash = sha256_text(text)
                key = (field_name, script, text_hash)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sample_id = f"magar-{len(entries) + 1:07d}"
                entries.append(
                    SynthesisTextEntry(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        language="magar",
                        script=script,
                        text=text,
                        text_sha256=text_hash,
                        source_path=str(source_path),
                        source_sha256=source_hash,
                        source_format=source_format,
                        source_row_id=row_id,
                        source_field=field_name,
                        source_schema=schema_name,
                        source_schema_verified=True,
                        unicode_blocks=_unicode_block_counts(text),
                        metadata={
                            "source": "LTK Western Magar dictionary",
                            "source_kind": "real_text_inventory_derived",
                            "claim_evidence_eligible": False,
                            "synthesis_training_candidate": True,
                            "license_status": "pending_review",
                            "normalization": "NFC_control_chars_removed",
                            "pos": row.get("pos"),
                            "ipa": row.get("ipa"),
                        },
                    )
                )
                emitted_for_source += 1
                if limit_per_source and emitted_for_source >= limit_per_source:
                    break
            if limit_per_source and emitted_for_source >= limit_per_source:
                break

    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "magar-text.jsonl"
    summary_json_path = out / "magar-text-summary.json"
    summary_md_path = out / "magar-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(sources),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def prepare_bible_brain_text(
    manifests: list[str | Path],
    output_dir: str | Path,
    *,
    languages: list[str],
    split: str = "train",
    dataset: str = "bible-brain-nepal",
    min_text_chars: int = 1,
    limit_per_manifest: int = 0,
) -> SynthesisTextPrepareSummary:
    if not manifests:
        raise DataValidationError("at least one Bible Brain text_manifest.jsonl path is required")
    if len(languages) != len(manifests):
        raise DataValidationError("--language must be provided once for each --manifest")
    if any(not str(language).strip() for language in languages):
        raise DataValidationError("Bible Brain language ids must be non-empty")
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    if limit_per_manifest < 0:
        raise DataValidationError("limit_per_manifest must be >= 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []

    for manifest_index, manifest_value in enumerate(manifests, start=1):
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_file():
            raise DataValidationError(f"Bible Brain text manifest does not exist: {manifest_path}")
        language = languages[manifest_index - 1]
        source_paths.append(str(manifest_path))
        source_hash = sha256_file(manifest_path)
        emitted_for_manifest = 0
        for row_index, row in enumerate(read_jsonl(manifest_path), start=1):
            _validate_bible_brain_manifest_row(row, manifest_path=manifest_path, row_index=row_index)
            if row.get("status") != "downloaded":
                rejected_count += 1
                continue
            chapter_path = Path(str(row["path"])).expanduser()
            if not chapter_path.is_file():
                raise DataValidationError(f"Bible Brain chapter path does not exist at {manifest_path}:{row_index}: {chapter_path}")
            chapter_hash = sha256_file(chapter_path)
            chapter_rows, chapter_schema = _load_bible_brain_chapter_text_rows(chapter_path, fileset_type=str(row["fileset_type"]))
            for chapter_row_id, field_name, raw_text, row_metadata in chapter_rows:
                text = _clean_synthesis_text(raw_text)
                if len(text) < min_text_chars:
                    rejected_count += 1
                    continue
                script = _dominant_script(text)
                text_hash = sha256_text(text)
                key = (language, script, text_hash)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sample_id = f"bible-brain-{len(entries) + 1:07d}"
                metadata = {
                    "source": "Bible Brain Nepal",
                    "source_kind": "real_text_inventory_derived",
                    "claim_evidence_eligible": False,
                    "synthesis_training_candidate": True,
                    "license_status": "pending_review",
                    "normalization": "NFC_control_chars_removed",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": source_hash,
                    "manifest_row_number": row_index,
                    "bible_id": row["bible_id"],
                    "book_id": row["book_id"],
                    "chapter": row["chapter"],
                    "fileset_id": row["fileset_id"],
                    "fileset_type": row["fileset_type"],
                    "chapter_path": str(chapter_path),
                    "chapter_sha256": chapter_hash,
                    "chapter_schema": chapter_schema,
                }
                metadata.update(row_metadata)
                entries.append(
                    SynthesisTextEntry(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        language=language,
                        script=script,
                        text=text,
                        text_sha256=text_hash,
                        source_path=str(chapter_path),
                        source_sha256=chapter_hash,
                        source_format="json",
                        source_row_id=chapter_row_id,
                        source_field=field_name,
                        source_schema=chapter_schema,
                        source_schema_verified=True,
                        unicode_blocks=_unicode_block_counts(text),
                        metadata=metadata,
                    )
                )
                emitted_for_manifest += 1
                if limit_per_manifest and emitted_for_manifest >= limit_per_manifest:
                    break
            if limit_per_manifest and emitted_for_manifest >= limit_per_manifest:
                break

    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "bible-brain-text.jsonl"
    summary_json_path = out / "bible-brain-text-summary.json"
    summary_md_path = out / "bible-brain-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(manifests),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def prepare_limbu_unicode_text(
    sources: list[str | Path],
    output_dir: str | Path,
    *,
    split: str = "train",
    dataset: str = "limbu-unicode-real-text",
    min_text_chars: int = 1,
    min_limbu_chars: int = 1,
    limit_per_source: int = 0,
) -> SynthesisTextPrepareSummary:
    if not sources:
        raise DataValidationError("at least one Limbu Unicode source path is required")
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    if min_limbu_chars < 1:
        raise DataValidationError("min_limbu_chars must be at least 1")
    if limit_per_source < 0:
        raise DataValidationError("limit_per_source must be >= 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str]] = set()
    source_paths: list[str] = []

    for source_value in sources:
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            raise DataValidationError(f"Limbu Unicode source path does not exist: {source_path}")
        source_paths.append(str(source_path))
        source_hash = sha256_file(source_path)
        rows, schema_name, source_format = _load_limbu_unicode_rows(source_path)
        emitted_for_source = 0
        for row_index, row in enumerate(rows, start=1):
            text = _clean_synthesis_text(row.get("text"))
            if len(text) < min_text_chars or _limbu_char_count(text) < min_limbu_chars:
                rejected_count += 1
                continue
            script = _dominant_script(text)
            text_hash = sha256_text(text)
            key = (script, text_hash)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            sample_id = f"limbu-unicode-{len(entries) + 1:07d}"
            metadata = {
                "source": row.get("source") or "Limbu Unicode text",
                "source_kind": "real_text_inventory_derived",
                "claim_evidence_eligible": False,
                "synthesis_training_candidate": True,
                "license_status": row.get("license_status") or "pending_review",
                "normalization": "NFC_control_chars_removed",
                "source_script_label": row.get("source_script_label"),
                "line_number": row.get("line_number"),
            }
            entries.append(
                SynthesisTextEntry(
                    sample_id=sample_id,
                    dataset=dataset,
                    split=split,
                    language="limbu",
                    script=script,
                    text=text,
                    text_sha256=text_hash,
                    source_path=str(source_path),
                    source_sha256=source_hash,
                    source_format=source_format,
                    source_row_id=str(row.get("row_id") or row_index),
                    source_field=str(row.get("source_field") or "text"),
                    source_schema=schema_name,
                    source_schema_verified=True,
                    unicode_blocks=_unicode_block_counts(text),
                    metadata=metadata,
                )
            )
            emitted_for_source += 1
            if limit_per_source and emitted_for_source >= limit_per_source:
                break

    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "limbu-unicode-text.jsonl"
    summary_json_path = out / "limbu-unicode-text-summary.json"
    summary_md_path = out / "limbu-unicode-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(sources),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def prepare_toolkit_parallel_text(
    sources: list[str | Path],
    output_dir: str | Path,
    *,
    text_fields: list[str],
    split: str = "train",
    dataset: str = "nepal-toolkit-parallel-text",
    min_text_chars: int = 1,
    limit_per_source: int = 0,
    row_id_field: str | None = None,
    metadata_fields: list[str] | None = None,
    license_status: str = "pending_review",
    split_policy: str | None = None,
) -> SynthesisTextPrepareSummary:
    if not sources:
        raise DataValidationError("at least one toolkit source path is required")
    field_languages = _parse_toolkit_text_fields(text_fields)
    if min_text_chars < 1:
        raise DataValidationError("min_text_chars must be at least 1")
    if limit_per_source < 0:
        raise DataValidationError("limit_per_source must be >= 0")
    if not license_status.strip():
        raise DataValidationError("license_status must be non-empty")
    split_policy_value = str(split_policy or "").strip()

    metadata_field_names = [field.strip() for field in (metadata_fields or []) if field.strip()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[SynthesisTextEntry] = []
    warnings: list[str] = []
    rejected_count = 0
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []

    for source_value in sources:
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            raise DataValidationError(f"toolkit source path does not exist: {source_path}")
        source_paths.append(str(source_path))
        source_hash = sha256_file(source_path)
        rows, schema_name, source_format = _load_toolkit_parallel_rows(
            source_path,
            required_fields=list(field_languages),
            row_id_field=row_id_field,
            metadata_fields=metadata_field_names,
        )
        emitted_for_source = 0
        for row_index, row in enumerate(rows, start=1):
            row_id = str(row.get(row_id_field or "") or row.get("id") or row.get("ref") or row_index)
            for field_name, language in field_languages.items():
                text = _clean_synthesis_text(row.get(field_name))
                if len(text) < min_text_chars:
                    rejected_count += 1
                    continue
                script = _dominant_script(text)
                text_hash = sha256_text(text)
                key = (language, field_name, text_hash)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sample_id = f"toolkit-parallel-{len(entries) + 1:07d}"
                metadata = {
                    "source": "Nepal language toolkit parallel corpus",
                    "source_kind": "real_text_inventory_derived",
                    "claim_evidence_eligible": False,
                    "synthesis_training_candidate": True,
                    "license_status": license_status,
                    "normalization": "NFC_control_chars_removed",
                }
                if split_policy_value:
                    metadata["split_policy"] = split_policy_value
                for metadata_field in metadata_field_names:
                    metadata[metadata_field] = row.get(metadata_field)
                entries.append(
                    SynthesisTextEntry(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        language=language,
                        script=script,
                        text=text,
                        text_sha256=text_hash,
                        source_path=str(source_path),
                        source_sha256=source_hash,
                        source_format=source_format,
                        source_row_id=row_id,
                        source_field=field_name,
                        source_schema=schema_name,
                        source_schema_verified=True,
                        unicode_blocks=_unicode_block_counts(text),
                        metadata=metadata,
                    )
                )
                emitted_for_source += 1
                if limit_per_source and emitted_for_source >= limit_per_source:
                    break
            if limit_per_source and emitted_for_source >= limit_per_source:
                break

    if not entries:
        warnings.append("no synthesis text rows emitted")
    manifest_path = out / "toolkit-parallel-text.jsonl"
    summary_json_path = out / "toolkit-parallel-text-summary.json"
    summary_md_path = out / "toolkit-parallel-text-summary.md"
    _write_synthesis_text_manifest(entries, manifest_path)
    summary = _synthesis_text_summary(
        entries,
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        source_count=len(sources),
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        source_paths=source_paths,
        warnings=warnings,
    )
    _write_synthesis_text_summary(summary)
    return summary


def audit_rendered_synthesis_lines(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    label_file: str | Path | None = None,
    require_font: bool = False,
    font_readiness_report: str | Path | None = None,
) -> RenderedLineManifestAudit:
    path = Path(manifest_path)
    if not path.is_file():
        raise DataValidationError(f"rendered synthesis manifest does not exist: {path}")
    label_path = Path(label_file) if label_file else None
    if label_path is not None and not label_path.is_file():
        raise DataValidationError(f"label_file does not exist: {label_path}")
    readiness_path = Path(font_readiness_report) if font_readiness_report else None
    readiness_hashes = _load_font_readiness_hashes(readiness_path) if readiness_path is not None else set()

    entries = load_manifest(path)
    issues: list[str] = []
    warnings: list[str] = []
    script_counts: dict[str, int] = {}
    font_counts: dict[str, int] = {}
    claim_count = 0

    label_rows = _read_paddle_line_labels(label_path) if label_path is not None else None
    if label_rows is not None and len(label_rows) != len(entries):
        issues.append(f"label row count {len(label_rows)} does not match manifest sample count {len(entries)}")

    for index, entry in enumerate(entries, start=1):
        metadata = entry.metadata
        prefix = f"{entry.sample_id}: "
        image_path = Path(entry.image_path)
        if not image_path.is_file():
            issues.append(f"{prefix}missing rendered image: {image_path}")
        elif entry.sha256:
            actual_hash = sha256_file(image_path)
            if actual_hash != entry.sha256:
                issues.append(f"{prefix}image sha256 mismatch: manifest={entry.sha256} actual={actual_hash}")
        else:
            issues.append(f"{prefix}missing image sha256")

        script = str(metadata.get("script") or "unknown")
        font_hash = metadata.get("font_sha256")
        script_counts[script] = script_counts.get(script, 0) + 1
        font_key = str(font_hash or "missing")
        font_counts[font_key] = font_counts.get(font_key, 0) + 1

        if metadata.get("claim_evidence_eligible") is not False:
            claim_count += 1
            issues.append(f"{prefix}rendered synthetic line must have claim_evidence_eligible=false")
        if metadata.get("source_kind") != "rendered_synthesis_text_line":
            issues.append(f"{prefix}source_kind must be rendered_synthesis_text_line")
        if metadata.get("document_type") != "synthetic_line":
            issues.append(f"{prefix}document_type must be synthetic_line")
        if metadata.get("input_format") != "image":
            issues.append(f"{prefix}input_format must be image")
        if metadata.get("source_schema_verified") is not True:
            issues.append(f"{prefix}source_schema_verified must be true")
        if require_font and not font_hash:
            issues.append(f"{prefix}font_sha256 is required")
        if require_font and not metadata.get("font_path"):
            issues.append(f"{prefix}font_path is required")
        if readiness_path is not None and not font_hash:
            issues.append(f"{prefix}font_sha256 is required when font_readiness_report is supplied")
        if readiness_path is not None and font_hash and str(font_hash) not in readiness_hashes:
            issues.append(f"{prefix}font_sha256 is not present in font readiness evidence: {font_hash}")
        for field_name in (
            "source_text_manifest",
            "source_text_sample_id",
            "source_text_sha256",
            "source_path",
            "source_sha256",
            "source_schema",
            "source_field",
            "text_sha256",
            "sample_sha256",
        ):
            if not metadata.get(field_name):
                issues.append(f"{prefix}missing metadata field: {field_name}")
        if "synthetic" not in _metadata_slices(metadata):
            issues.append(f"{prefix}slices must include synthetic")
        if "rendered_text" not in _metadata_slices(metadata):
            issues.append(f"{prefix}slices must include rendered_text")
        text_hash = sha256_text(entry.text)
        if metadata.get("text_sha256") and metadata.get("text_sha256") != text_hash:
            issues.append(f"{prefix}text_sha256 mismatch")
        if metadata.get("source_text_sha256") and metadata.get("source_text_sha256") != text_hash:
            warnings.append(f"{prefix}source_text_sha256 differs from rendered text hash")
        if metadata.get("sample_sha256") and entry.sha256:
            if "degradation_profile" in metadata or "degradation" in metadata:
                profile = str(metadata.get("degradation_profile") or metadata.get("degradation") or "")
                expected_sample_hash = sha256_text(f"{entry.sha256}\n{text_hash}\n{font_hash or ''}\n{profile}\n")
            else:
                expected_sample_hash = sha256_text(f"{entry.sha256}\n{text_hash}\n{font_hash or ''}\n")
            if metadata.get("sample_sha256") != expected_sample_hash:
                issues.append(f"{prefix}sample_sha256 mismatch")
        profile = str(metadata.get("degradation_profile") or metadata.get("degradation") or "")
        if profile in LIGHTING_DEGRADATION_PROFILES:
            parameters = metadata.get("degradation_parameters")
            if not isinstance(parameters, dict):
                issues.append(f"{prefix}lighting degradation requires degradation_parameters object")
            else:
                if profile in {"low_light", "phone_photo"}:
                    if not isinstance(parameters.get("brightness"), int | float):
                        issues.append(f"{prefix}lighting degradation missing numeric brightness")
                    if not isinstance(parameters.get("contrast"), int | float):
                        issues.append(f"{prefix}lighting degradation missing numeric contrast")
                if profile in {"uneven_light", "phone_photo"}:
                    if not parameters.get("lighting_gradient"):
                        issues.append(f"{prefix}lighting degradation missing lighting_gradient")
                    if not isinstance(parameters.get("lighting_blend"), int | float):
                        issues.append(f"{prefix}lighting degradation missing numeric lighting_blend")
                if profile == "glare_light":
                    if not parameters.get("glare_center"):
                        issues.append(f"{prefix}lighting degradation missing glare_center")
                    if not isinstance(parameters.get("glare_blend"), int | float):
                        issues.append(f"{prefix}lighting degradation missing numeric glare_blend")
        if profile in CAMERA_DEGRADATION_PROFILES:
            parameters = metadata.get("degradation_parameters")
            if not isinstance(parameters, dict):
                issues.append(f"{prefix}camera degradation requires degradation_parameters object")
            else:
                if parameters.get("camera_transform") != "perspective":
                    issues.append(f"{prefix}camera degradation must record camera_transform=perspective")
                for field_name in ("source_quad", "output_quad", "perspective_coefficients"):
                    if not parameters.get(field_name):
                        issues.append(f"{prefix}camera degradation missing {field_name}")
        if label_rows is not None and index <= len(label_rows):
            label_image, label_text = label_rows[index - 1]
            if label_image != entry.image_path:
                issues.append(f"{prefix}label image path mismatch")
            if label_text != _paddle_safe_label(entry.text):
                issues.append(f"{prefix}label text mismatch")

    if not entries:
        issues.append("rendered synthesis manifest is empty")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = RenderedLineManifestAudit(
        manifest_path=str(path),
        output_json_path=str(out / "rendered-line-manifest-audit.json"),
        output_md_path=str(out / "rendered-line-manifest-audit.md"),
        sample_count=len(entries),
        script_counts=dict(sorted(script_counts.items())),
        font_counts=dict(sorted(font_counts.items())),
        claim_evidence_eligible_count=claim_count,
        font_readiness_report=str(readiness_path) if readiness_path is not None else None,
        issues=issues,
        warnings=warnings,
    )
    _write_rendered_line_manifest_audit(audit)
    return audit


def render_synthesis_text_lines(
    synthesis_text_manifest: str | Path,
    output_dir: str | Path,
    *,
    font_path: str | Path | None = None,
    limit: int = 0,
    scripts: list[str] | None = None,
    font_size: int = 36,
    padding: int = 16,
    degradation_profiles: list[str] | None = None,
    degradation_seed: int = 13,
    split: str = "train",
    dataset: str = "synthesis-rendered-lines",
) -> RenderedLineSummary:
    if limit < 0:
        raise DataValidationError("limit must be >= 0")
    if font_size < 1:
        raise DataValidationError("font_size must be >= 1")
    if padding < 0:
        raise DataValidationError("padding must be >= 0")
    profiles = _validate_render_degradation_profiles(degradation_profiles)
    source_manifest = Path(synthesis_text_manifest)
    if not source_manifest.is_file():
        raise DataValidationError(f"synthesis text manifest does not exist: {source_manifest}")
    selected_scripts = set(scripts or [])
    font = Path(font_path).expanduser() if font_path else None
    if font is not None and not font.is_file():
        raise DataValidationError(f"font_path does not exist: {font}")
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise DataValidationError("line rendering requires Pillow. Install ocr-tech[eval].") from exc

    out = Path(output_dir)
    images_dir = out / "images"
    manifests_dir = out / "manifests"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if font:
        # Use HarfBuzz/libraqm complex-script shaping when available (required for correct
        # Brahmic stacking/reordering — Devanagari conjuncts, Limbu subjoined consonants);
        # fall back to basic layout otherwise.
        _layout_engine = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
        pillow_font = ImageFont.truetype(str(font), font_size, layout_engine=_layout_engine)
    else:
        pillow_font = ImageFont.load_default()
    font_hash = sha256_file(font) if font else None
    entries: list[ManifestEntry] = []
    script_counts: dict[str, int] = {}
    degradation_counts: dict[str, int] = {}
    skipped_count = 0
    for row in read_jsonl(source_manifest):
        if limit and len(entries) >= limit:
            break
        text = str(row.get("text") if row.get("text") is not None else "")
        script = str(row.get("script") or "unknown")
        if selected_scripts and script not in selected_scripts:
            skipped_count += 1
            continue
        if not text.strip():
            skipped_count += 1
            continue
        text_hash = sha256_text(text)
        for profile in profiles:
            if limit and len(entries) >= limit:
                break
            sample_id = f"rendered-line-{len(entries) + 1:06d}"
            target = images_dir / f"{sample_id}.png"
            rng = random.Random(f"{degradation_seed}:{row.get('sample_id')}:{text_hash}:{profile}")
            degradation_metadata = _render_line_image(
                Image,
                ImageDraw,
                ImageEnhance,
                ImageFilter,
                target,
                text,
                pillow_font,
                padding=padding,
                degradation_profile=profile,
                rng=rng,
            )
            image_hash = sha256_file(target)
            script_counts[script] = script_counts.get(script, 0) + 1
            degradation_counts[profile] = degradation_counts.get(profile, 0) + 1
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "source_text_manifest": str(source_manifest),
                    "source_text_sample_id": row.get("sample_id"),
                    "source_text_sha256": row.get("text_sha256") or text_hash,
                    "source_path": row.get("source_path"),
                    "source_sha256": row.get("source_sha256"),
                    "source_schema": row.get("source_schema"),
                    "source_schema_verified": bool(row.get("source_schema_verified")),
                    "source_field": row.get("source_field"),
                    "source_row_id": row.get("source_row_id"),
                    "language": row.get("language") or "unknown",
                    "script": script,
                    "input_format": "image",
                    "document_type": "synthetic_line",
                    "source_kind": "rendered_synthesis_text_line",
                    "font_path": str(font) if font else None,
                    "font_sha256": font_hash,
                    "font_size": font_size,
                    "padding": padding,
                    "claim_evidence_eligible": False,
                    "degradation": profile,
                    "degradation_profile": profile,
                    "degradation_seed": degradation_seed,
                    "degradation_parameters": degradation_metadata,
                    "slices": sorted(
                        {
                            "synthetic",
                            "line_crop",
                            "rendered_text",
                            str(row.get("language") or "unknown"),
                            script,
                            profile,
                        }
                    ),
                    "text_sha256": text_hash,
                    "sample_sha256": sha256_text(f"{image_hash}\n{text_hash}\n{font_hash or ''}\n{profile}\n"),
                }
            )
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    dataset=dataset,
                    split=split,
                    image_path=str(target),
                    text=text,
                    sha256=image_hash,
                    metadata=metadata,
                )
            )
    warnings: list[str] = []
    if not entries:
        warnings.append("no line images rendered")
    manifest_path = manifests_dir / "rendered-line-manifest.jsonl"
    label_path = labels_dir / f"{split}.txt"
    summary_json_path = out / "rendered-line-summary.json"
    summary_md_path = out / "rendered-line-summary.md"
    write_manifest(entries, manifest_path)
    _write_paddle_line_labels(entries, label_path)
    summary = RenderedLineSummary(
        manifest_path=str(manifest_path),
        label_path=str(label_path),
        summary_json_path=str(summary_json_path),
        summary_md_path=str(summary_md_path),
        image_dir=str(images_dir),
        sample_count=len(entries),
        skipped_count=skipped_count,
        script_counts=dict(sorted(script_counts.items())),
        degradation_counts=dict(sorted(degradation_counts.items())),
        font_path=str(font) if font else None,
        font_sha256=font_hash,
        warnings=warnings,
    )
    _write_rendered_line_summary(summary)
    return summary


def render_synthesis_text_split(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    font_path: str | Path | None = None,
    limit_per_split: int = 0,
    scripts: list[str] | None = None,
    font_size: int = 36,
    padding: int = 16,
    degradation_profiles: list[str] | None = None,
    degradation_seed: int = 13,
    dataset: str = "synthesis-rendered-lines",
    require_font: bool = False,
    font_readiness_report: str | Path | None = None,
) -> RenderedLineSplitSummary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_summary = render_synthesis_text_lines(
        train_manifest,
        out / "train",
        font_path=font_path,
        limit=limit_per_split,
        scripts=scripts,
        font_size=font_size,
        padding=padding,
        split="train",
        dataset=dataset,
        degradation_profiles=degradation_profiles,
        degradation_seed=degradation_seed,
    )
    eval_summary = render_synthesis_text_lines(
        eval_manifest,
        out / "eval",
        font_path=font_path,
        limit=limit_per_split,
        scripts=scripts,
        font_size=font_size,
        padding=padding,
        split="eval",
        dataset=dataset,
        degradation_profiles=degradation_profiles,
        degradation_seed=degradation_seed,
    )
    train_audit = audit_rendered_synthesis_lines(
        train_summary.manifest_path,
        out / "train-rendered-audit",
        label_file=train_summary.label_path,
        require_font=require_font,
        font_readiness_report=font_readiness_report,
    )
    eval_audit = audit_rendered_synthesis_lines(
        eval_summary.manifest_path,
        out / "eval-rendered-audit",
        label_file=eval_summary.label_path,
        require_font=require_font,
        font_readiness_report=font_readiness_report,
    )
    warnings = [f"train: {warning}" for warning in train_summary.warnings]
    warnings.extend(f"eval: {warning}" for warning in eval_summary.warnings)
    summary = RenderedLineSplitSummary(
        train_manifest=train_summary.manifest_path,
        train_label_path=train_summary.label_path,
        train_audit_path=train_audit.output_json_path,
        eval_manifest=eval_summary.manifest_path,
        eval_label_path=eval_summary.label_path,
        eval_audit_path=eval_audit.output_json_path,
        train_count=train_summary.sample_count,
        eval_count=eval_summary.sample_count,
        train_audit_passed=train_audit.passed,
        eval_audit_passed=eval_audit.passed,
        output_json_path=str(out / "rendered-line-split-summary.json"),
        output_md_path=str(out / "rendered-line-split-summary.md"),
        warnings=warnings,
    )
    _write_rendered_line_split_summary(summary)
    return summary


def audit_rendered_degradation_split(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_profiles: list[str] | None = None,
) -> RenderedDegradationSplitAudit:
    profiles = _validate_render_degradation_profiles(expected_profiles) if expected_profiles else []
    train_path = Path(train_manifest)
    eval_path = Path(eval_manifest)
    if not train_path.is_file():
        raise DataValidationError(f"train rendered manifest does not exist: {train_path}")
    if not eval_path.is_file():
        raise DataValidationError(f"eval rendered manifest does not exist: {eval_path}")

    train_entries = load_manifest(train_path)
    eval_entries = load_manifest(eval_path)
    train_stats = _rendered_degradation_stats(train_entries, expected_profiles=profiles, split_name="train")
    eval_stats = _rendered_degradation_stats(eval_entries, expected_profiles=profiles, split_name="eval")
    issues = [*train_stats["issues"], *eval_stats["issues"]]
    warnings = [*train_stats["warnings"], *eval_stats["warnings"]]

    train_hashes = set(train_stats["text_profiles"])
    eval_hashes = set(eval_stats["text_profiles"])
    overlap = sorted(train_hashes & eval_hashes)
    if overlap:
        issues.append(f"train/eval text_sha256 overlap count={len(overlap)} first={overlap[0]}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = RenderedDegradationSplitAudit(
        train_manifest=str(train_path),
        eval_manifest=str(eval_path),
        output_json_path=str(out / "rendered-degradation-split-audit.json"),
        output_md_path=str(out / "rendered-degradation-split-audit.md"),
        train_count=len(train_entries),
        eval_count=len(eval_entries),
        expected_profiles=profiles,
        train_profile_counts=dict(sorted(train_stats["profile_counts"].items())),
        eval_profile_counts=dict(sorted(eval_stats["profile_counts"].items())),
        train_text_hash_count=len(train_hashes),
        eval_text_hash_count=len(eval_hashes),
        issues=issues,
        warnings=warnings,
    )
    _write_rendered_degradation_split_audit(audit)
    return audit


def _load_font_readiness_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"font_readiness_report does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"font_readiness_report is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"font_readiness_report must be a JSON object: {path}")
    if payload.get("passed") is not True:
        raise DataValidationError(f"font_readiness_report did not pass: {path}")
    manifest_value = payload.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise DataValidationError(f"font_readiness_report missing manifest_path: {path}")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file():
        raise DataValidationError(f"font readiness manifest does not exist: {manifest_path}")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"font readiness manifest is invalid JSON: {manifest_path}: {exc}") from exc
    assets = manifest_payload.get("assets") if isinstance(manifest_payload, dict) else None
    if not isinstance(assets, list) or not assets:
        raise DataValidationError(f"font readiness manifest must contain assets: {manifest_path}")
    hashes: set[str] = set()
    for item in assets:
        if not isinstance(item, dict):
            continue
        target = item.get("target_path")
        sha = item.get("sha256")
        if isinstance(target, str) and Path(target).suffix.lower() in FONT_SUFFIXES and isinstance(sha, str) and sha:
            hashes.add(sha)
    if not hashes:
        raise DataValidationError(f"font readiness manifest contains no font asset hashes: {manifest_path}")
    return hashes


def _audit_resource_root(
    root: Path,
    *,
    label: str | None,
    max_files: int,
    max_text_samples: int,
    sample_bytes: int,
) -> ResourceRootAudit:
    if not root.exists():
        return ResourceRootAudit(
            root=str(root),
            label=label,
            exists=False,
            total_files=0,
            total_size_bytes=0,
            suffix_counts={},
            text_file_count=0,
            structured_file_count=0,
            font_file_count=0,
            document_file_count=0,
            sampled_files=[],
            warnings=["root does not exist"],
        )
    if not root.is_dir():
        return ResourceRootAudit(
            root=str(root),
            label=label,
            exists=False,
            total_files=0,
            total_size_bytes=0,
            suffix_counts={},
            text_file_count=0,
            structured_file_count=0,
            font_file_count=0,
            document_file_count=0,
            sampled_files=[],
            warnings=["root is not a directory"],
        )
    suffix_counts: dict[str, int] = {}
    sampled_files: list[TextSampleAudit] = []
    warnings: list[str] = []
    total_files = 0
    total_size_bytes = 0
    text_file_count = 0
    structured_file_count = 0
    font_file_count = 0
    document_file_count = 0
    text_candidates: list[Path] = []
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        if max_files and total_files >= max_files:
            warnings.append(f"stopped after max_files_per_root={max_files}")
            break
        total_files += 1
        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            warnings.append(f"cannot stat {file_path}: {type(exc).__name__}: {exc}")
            continue
        total_size_bytes += size_bytes
        suffix = file_path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix in TEXT_SUFFIXES:
            text_file_count += 1
            text_candidates.append(file_path)
        if suffix in STRUCTURED_SUFFIXES:
            structured_file_count += 1
        if suffix in FONT_SUFFIXES:
            font_file_count += 1
        if suffix in DOCUMENT_SUFFIXES:
            document_file_count += 1
    for file_path in sorted(text_candidates, key=lambda path: _text_sample_priority(path, root))[:max_text_samples]:
        try:
            sampled_files.append(_audit_text_sample(file_path, root=root, sample_bytes=sample_bytes))
        except OSError as exc:
            warnings.append(f"cannot sample {file_path}: {type(exc).__name__}: {exc}")
    return ResourceRootAudit(
        root=str(root),
        label=label,
        exists=True,
        total_files=total_files,
        total_size_bytes=total_size_bytes,
        suffix_counts=dict(sorted(suffix_counts.items())),
        text_file_count=text_file_count,
        structured_file_count=structured_file_count,
        font_file_count=font_file_count,
        document_file_count=document_file_count,
        sampled_files=sampled_files,
        warnings=warnings,
    )


def _load_limdic_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_limdic_json_rows(path)
    if suffix == ".tsv":
        return _load_limdic_tsv_rows(path)
    raise DataValidationError(f"unsupported Limdic source suffix for {path}: {suffix}")


def _load_limbu_unicode_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value:
                    continue
                rows.append(
                    {
                        "row_id": line_number,
                        "line_number": line_number,
                        "text": value,
                        "source_field": "line",
                        "source": "Limbu Unicode plaintext",
                        "license_status": "pending_review",
                        "source_script_label": "Sirijunga-Unicode",
                    }
                )
        return rows, "limbu_unicode_plaintext_lines_v1", "txt"
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise DataValidationError(f"Limbu Unicode TSV has no header: {path}")
            missing = [field for field in LIMBU_TAGGED_SENTENCE_FIELDS if field not in reader.fieldnames]
            if missing:
                raise DataValidationError(f"Limbu Unicode TSV missing fields: {', '.join(missing)}")
            rows = []
            for row_index, row in enumerate(reader, start=1):
                rows.append(
                    {
                        "row_id": row_index,
                        "line_number": row_index + 1,
                        "text": row.get("sentence"),
                        "source_field": "sentence",
                        "source": row.get("source") or "Limbu Unicode tagged sentences",
                        "license_status": row.get("license") or "pending_review",
                        "source_script_label": row.get("script"),
                    }
                )
        return rows, "limbu_unicode_tagged_sentences_tsv_v1", "tsv"
    raise DataValidationError(f"unsupported Limbu Unicode source suffix for {path}: {suffix}")


def _validate_bible_brain_manifest_row(row: dict[str, Any], *, manifest_path: Path, row_index: int) -> None:
    if not isinstance(row, dict):
        raise DataValidationError(f"Bible Brain manifest row {manifest_path}:{row_index} must be an object")
    missing = [field for field in BIBLE_BRAIN_MANIFEST_FIELDS if field not in row]
    if missing:
        raise DataValidationError(
            f"Bible Brain manifest row {manifest_path}:{row_index} missing fields: {', '.join(missing)}"
        )
    if not isinstance(row["path"], str) or not row["path"].strip():
        raise DataValidationError(f"Bible Brain manifest row {manifest_path}:{row_index} has invalid path")
    if row["fileset_type"] not in {"text_plain", "text_json"}:
        raise DataValidationError(
            f"Bible Brain manifest row {manifest_path}:{row_index} has unsupported fileset_type: {row['fileset_type']}"
        )


def _load_bible_brain_chapter_text_rows(
    path: Path,
    *,
    fileset_type: str,
) -> tuple[list[tuple[str, str, Any, dict[str, Any]]], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid Bible Brain chapter JSON {path}: {exc}") from exc
    if fileset_type == "text_plain":
        return _load_bible_brain_plain_rows(payload, path)
    if fileset_type == "text_json":
        return _load_bible_brain_sofria_rows(payload, path)
    raise DataValidationError(f"unsupported Bible Brain fileset_type for {path}: {fileset_type}")


def _load_bible_brain_plain_rows(payload: Any, path: Path) -> tuple[list[tuple[str, str, Any, dict[str, Any]]], str]:
    if not isinstance(payload, list):
        raise DataValidationError(f"Bible Brain text_plain chapter must be a list: {path}")
    rows: list[tuple[str, str, Any, dict[str, Any]]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise DataValidationError(f"Bible Brain text_plain row {path}:{index} must be an object")
        missing = [field for field in BIBLE_BRAIN_PLAIN_VERSE_FIELDS if field not in item]
        if missing:
            raise DataValidationError(f"Bible Brain text_plain row {path}:{index} missing fields: {', '.join(missing)}")
        verse_start = item.get("verse_start")
        verse_end = item.get("verse_end")
        row_id = f"{item.get('book_id')}.{item.get('chapter')}.{verse_start}-{verse_end}"
        rows.append(
            (
                row_id,
                "verse_text",
                item.get("verse_text"),
                {
                    "book_name": item.get("book_name"),
                    "book_name_alt": item.get("book_name_alt"),
                    "chapter_alt": item.get("chapter_alt"),
                    "verse_start": verse_start,
                    "verse_end": verse_end,
                    "verse_start_alt": item.get("verse_start_alt"),
                    "verse_end_alt": item.get("verse_end_alt"),
                },
            )
        )
    return rows, "bible_brain_text_plain_json_v1"


def _load_bible_brain_sofria_rows(payload: Any, path: Path) -> tuple[list[tuple[str, str, Any, dict[str, Any]]], str]:
    if not isinstance(payload, dict):
        raise DataValidationError(f"Bible Brain text_json chapter must be an object: {path}")
    schema = payload.get("schema")
    sequence = payload.get("sequence")
    metadata = payload.get("metadata")
    if not isinstance(schema, dict) or schema.get("structure") != "nested":
        raise DataValidationError(f"Bible Brain text_json chapter has unsupported schema: {path}")
    if not isinstance(sequence, dict):
        raise DataValidationError(f"Bible Brain text_json chapter missing sequence object: {path}")
    if not isinstance(metadata, dict):
        raise DataValidationError(f"Bible Brain text_json chapter missing metadata object: {path}")

    book_id = _nested_str(metadata, "document", "bookCode") or _nested_str(metadata, "document", "id") or path.stem
    document_title = _nested_str(metadata, "document", "h") or _nested_str(metadata, "document", "toc")
    rows: list[tuple[str, str, Any, dict[str, Any]]] = []
    for index, item in enumerate(_iter_sofria_text_items(sequence), start=1):
        rows.append(
            (
                f"{book_id}.content.{index}",
                "content",
                item["text"],
                {
                    "book_title": document_title,
                    "sequence_type": item.get("sequence_type"),
                    "block_type": item.get("block_type"),
                    "block_subtype": item.get("block_subtype"),
                    "content_index": item.get("content_index"),
                },
            )
        )
    if not rows:
        raise DataValidationError(f"Bible Brain text_json chapter contains no string content rows: {path}")
    return rows, "bible_brain_sofria_text_json_v1"


def _iter_sofria_text_items(node: Any, *, sequence_type: str | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(node, dict):
        current_type = str(node.get("type") or "")
        current_sequence_type = current_type if "blocks" in node or "sequence" in node or "sequences" in node else str(sequence_type or "")
        content = node.get("content")
        if isinstance(content, list):
            for index, value in enumerate(content):
                if isinstance(value, str):
                    items.append(
                        {
                            "text": value,
                            "sequence_type": current_sequence_type,
                            "block_type": node.get("type"),
                            "block_subtype": node.get("subtype"),
                            "content_index": index,
                        }
                    )
                else:
                    items.extend(_iter_sofria_text_items(value, sequence_type=current_sequence_type))
        for key in ("blocks", "sequences"):
            child = node.get(key)
            if isinstance(child, list):
                for value in child:
                    items.extend(_iter_sofria_text_items(value, sequence_type=current_sequence_type))
            elif isinstance(child, dict):
                items.extend(_iter_sofria_text_items(child, sequence_type=current_sequence_type))
        child_sequence = node.get("sequence")
        if isinstance(child_sequence, dict):
            child_sequence_type = str(child_sequence.get("type") or current_sequence_type)
            items.extend(_iter_sofria_text_items(child_sequence, sequence_type=child_sequence_type))
        elif isinstance(child_sequence, list):
            for value in child_sequence:
                items.extend(_iter_sofria_text_items(value, sequence_type=current_sequence_type))
    elif isinstance(node, list):
        for value in node:
            items.extend(_iter_sofria_text_items(value, sequence_type=sequence_type))
    return items


def _nested_str(payload: dict[str, Any], *keys: str) -> str | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _validate_render_degradation_profiles(profiles: list[str] | None) -> list[str]:
    values = [str(profile).strip() for profile in (profiles or ["clean"]) if str(profile).strip()]
    if not values:
        return ["clean"]
    invalid = [profile for profile in values if profile not in RENDER_DEGRADATION_PROFILES]
    if invalid:
        valid = ", ".join(sorted(RENDER_DEGRADATION_PROFILES))
        raise DataValidationError(f"unsupported degradation profile(s): {', '.join(invalid)}; valid profiles: {valid}")
    return values


def _rendered_degradation_stats(
    entries: list[ManifestEntry],
    *,
    expected_profiles: list[str],
    split_name: str,
) -> dict[str, Any]:
    profile_counts: dict[str, int] = {}
    text_profiles: dict[str, dict[str, int]] = {}
    issues: list[str] = []
    warnings: list[str] = []
    expected = set(expected_profiles)
    for entry in entries:
        metadata = entry.metadata
        text_hash = str(metadata.get("text_sha256") or sha256_text(entry.text))
        profile = metadata.get("degradation_profile") or metadata.get("degradation")
        if not isinstance(profile, str) or not profile:
            issues.append(f"{split_name}:{entry.sample_id}: missing degradation_profile")
            continue
        if expected and profile not in expected:
            issues.append(f"{split_name}:{entry.sample_id}: unexpected degradation_profile={profile}")
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        per_text = text_profiles.setdefault(text_hash, {})
        per_text[profile] = per_text.get(profile, 0) + 1
        if metadata.get("claim_evidence_eligible") is not False:
            issues.append(f"{split_name}:{entry.sample_id}: rendered degradation row must have claim_evidence_eligible=false")

    if not entries:
        issues.append(f"{split_name}: rendered manifest is empty")
    if expected:
        for text_hash, counts in sorted(text_profiles.items()):
            observed = set(counts)
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            duplicates = sorted(profile for profile, count in counts.items() if count != 1)
            if missing:
                issues.append(f"{split_name}:{text_hash}: missing degradation profiles: {', '.join(missing)}")
            if extra:
                issues.append(f"{split_name}:{text_hash}: unexpected degradation profiles: {', '.join(extra)}")
            if duplicates:
                issues.append(f"{split_name}:{text_hash}: duplicate degradation profile rows: {', '.join(duplicates)}")
    elif entries:
        warnings.append(f"{split_name}: expected degradation profiles not supplied; checked only overlap and metadata presence")
    return {
        "profile_counts": profile_counts,
        "text_profiles": text_profiles,
        "issues": issues,
        "warnings": warnings,
    }


def _render_line_image(
    Image: Any,
    ImageDraw: Any,
    ImageEnhance: Any,
    ImageFilter: Any,
    path: Path,
    text: str,
    font: Any,
    *,
    padding: int,
    degradation_profile: str,
    rng: random.Random,
) -> dict[str, Any]:
    probe = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0]) + padding * 2
    height = max(1, bbox[3] - bbox[1]) + padding * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((padding - bbox[0], padding - bbox[1]), text, fill="black", font=font)
    parameters = _apply_render_degradation(Image, ImageEnhance, ImageFilter, image, degradation_profile, rng)
    image.save(path)
    return parameters


def _apply_render_degradation(
    Image: Any,
    ImageEnhance: Any,
    ImageFilter: Any,
    image: Any,
    profile: str,
    rng: random.Random,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {"profile": profile}
    if profile == "clean":
        return parameters

    if profile in {"low_light", "phone_photo"}:
        brightness = 0.62 if profile == "low_light" else 0.82
        contrast = 1.12 if profile == "low_light" else 1.08
        image.paste(ImageEnhance.Brightness(image).enhance(brightness))
        image.paste(ImageEnhance.Contrast(image).enhance(contrast))
        parameters.update({"brightness": brightness, "contrast": contrast})

    if profile in {"uneven_light", "phone_photo"}:
        overlay = Image.new("RGB", image.size, "white")
        pixels = overlay.load()
        width, height = image.size
        direction = rng.choice(("left_to_right", "right_to_left", "top_to_bottom"))
        for y in range(height):
            for x in range(width):
                if direction == "top_to_bottom":
                    ratio = y / max(1, height - 1)
                elif direction == "right_to_left":
                    ratio = 1.0 - (x / max(1, width - 1))
                else:
                    ratio = x / max(1, width - 1)
                value = int(235 + ratio * 20)
                pixels[x, y] = (value, value, value)
        image.paste(Image.blend(image, overlay, 0.28))
        parameters.update({"lighting_gradient": direction, "lighting_blend": 0.28})

    if profile == "glare_light":
        overlay = Image.new("RGB", image.size, "white")
        pixels = overlay.load()
        width, height = image.size
        center_x = int(width * (0.35 + rng.random() * 0.3))
        center_y = int(height * (0.25 + rng.random() * 0.5))
        radius = max(1.0, min(width, height) * (0.45 + rng.random() * 0.2))
        for y in range(height):
            for x in range(width):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                ratio = max(0.0, 1.0 - distance / radius)
                value = int(235 + ratio * 20)
                pixels[x, y] = (value, value, value)
        glare_blend = 0.34
        image.paste(Image.blend(image, overlay, glare_blend))
        parameters.update(
            {
                "glare_center": [center_x, center_y],
                "glare_radius": round(radius, 6),
                "glare_blend": glare_blend,
            }
        )

    if profile in {"scan", "phone_photo"}:
        blur_radius = 0.45 if profile == "scan" else 0.7
        image.paste(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        image.paste(ImageEnhance.Contrast(image).enhance(0.92))
        parameters.update({"blur_radius": blur_radius, "contrast": 0.92})

    if profile in CAMERA_DEGRADATION_PROFILES:
        direction = profile.removeprefix("camera_")
        strength = 0.10
        if profile == "phone_photo":
            direction = rng.choice(("left", "right", "top", "bottom"))
            strength = 0.06 + rng.random() * 0.05
        original_width, original_height = image.size
        horizontal = max(2.0, original_width * strength)
        vertical = max(1.0, original_height * (strength / 2.0))
        source_quad = _camera_source_quad(original_width, original_height, direction=direction, horizontal=horizontal, vertical=vertical)
        output_quad = [
            [0.0, 0.0],
            [float(original_width), 0.0],
            [float(original_width), float(original_height)],
            [0.0, float(original_height)],
        ]
        coefficients = _perspective_coefficients(source_quad, output_quad)
        transformed = image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor="white",
        )
        image.paste(transformed)
        parameters.update(
            {
                "camera_angle": direction,
                "camera_transform": "perspective",
                "perspective_strength": round(strength, 6),
                "source_quad": source_quad,
                "output_quad": output_quad,
                "perspective_coefficients": [round(value, 8) for value in coefficients],
            }
        )

    return parameters


def _camera_source_quad(
    width: int,
    height: int,
    *,
    direction: str,
    horizontal: float,
    vertical: float,
) -> list[list[float]]:
    right = float(width)
    bottom = float(height)
    if direction == "left":
        return [
            [horizontal, vertical],
            [right, 0.0],
            [right, bottom],
            [horizontal, bottom - vertical],
        ]
    if direction == "right":
        return [
            [0.0, 0.0],
            [right - horizontal, vertical],
            [right - horizontal, bottom - vertical],
            [0.0, bottom],
        ]
    if direction == "top":
        return [
            [horizontal / 2.0, vertical],
            [right - (horizontal / 2.0), vertical],
            [right, bottom],
            [0.0, bottom],
        ]
    if direction == "bottom":
        return [
            [0.0, 0.0],
            [right, 0.0],
            [right - (horizontal / 2.0), bottom - vertical],
            [horizontal / 2.0, bottom - vertical],
        ]
    raise DataValidationError(f"unsupported camera direction: {direction}")


def _perspective_coefficients(source_quad: list[list[float]], output_quad: list[list[float]]) -> tuple[float, ...]:
    if len(source_quad) != 4 or len(output_quad) != 4:
        raise DataValidationError("perspective transform requires four source and output points")
    matrix: list[list[float]] = []
    values: list[float] = []
    for (source_x, source_y), (output_x, output_y) in zip(source_quad, output_quad, strict=True):
        matrix.append([output_x, output_y, 1.0, 0.0, 0.0, 0.0, -source_x * output_x, -source_x * output_y])
        matrix.append([0.0, 0.0, 0.0, output_x, output_y, 1.0, -source_y * output_x, -source_y * output_y])
        values.extend([source_x, source_y])
    return tuple(_solve_linear_system(matrix, values))


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise DataValidationError("linear system matrix must be square")
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row_index: abs(augmented[row_index][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise DataValidationError("perspective transform matrix is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor:
                augmented[row_index] = [
                    current - factor * pivot_current
                    for current, pivot_current in zip(augmented[row_index], augmented[column], strict=True)
                ]
    return [augmented[row_index][-1] for row_index in range(size)]


def _write_paddle_line_labels(entries: list[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry.image_path}\t{_paddle_safe_label(entry.text)}\n")


def _read_paddle_line_labels(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.rstrip("\n")
            if not value:
                continue
            if "\t" not in value:
                raise DataValidationError(f"invalid PaddleOCR label row at {path}:{line_number}: missing tab")
            image_path, text = value.split("\t", 1)
            if not image_path:
                raise DataValidationError(f"invalid PaddleOCR label row at {path}:{line_number}: missing image path")
            rows.append((image_path, text))
    return rows


def _paddle_safe_label(text: str) -> str:
    return normalize_ocr_text(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _metadata_slices(metadata: dict[str, Any]) -> set[str]:
    value = metadata.get("slices")
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return set()


def _write_rendered_line_summary(summary: RenderedLineSummary) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rendered Synthesis Line Images",
        "",
        f"- passed: {summary.passed}",
        f"- manifest: `{summary.manifest_path}`",
        f"- labels: `{summary.label_path}`",
        f"- image dir: `{summary.image_dir}`",
        f"- samples: {summary.sample_count}",
        f"- skipped: {summary.skipped_count}",
        f"- font: `{summary.font_path or 'Pillow default'}`",
        f"- font sha256: `{summary.font_sha256 or 'none'}`",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Scripts",
        "",
    ]
    lines.extend(f"- `{script}`: {count}" for script, count in summary.script_counts.items()) if summary.script_counts else lines.append("- none")
    lines.extend(["", "## Degradations", ""])
    lines.extend(f"- `{profile}`: {count}" for profile, count in summary.degradation_counts.items()) if summary.degradation_counts else lines.append("- none")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rendered_line_split_summary(summary: RenderedLineSplitSummary) -> None:
    json_path = Path(summary.output_json_path)
    md_path = Path(summary.output_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rendered Synthesis Line Split",
        "",
        f"- passed: {summary.passed}",
        f"- train manifest: `{summary.train_manifest}`",
        f"- train labels: `{summary.train_label_path}`",
        f"- train audit: `{summary.train_audit_path}`",
        f"- train samples: {summary.train_count}",
        f"- eval manifest: `{summary.eval_manifest}`",
        f"- eval labels: `{summary.eval_label_path}`",
        f"- eval audit: `{summary.eval_audit_path}`",
        f"- eval samples: {summary.eval_count}",
        f"- train audit passed: {summary.train_audit_passed}",
        f"- eval audit passed: {summary.eval_audit_passed}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthesis_text_manifest_audit(audit: SynthesisTextManifestAudit) -> None:
    json_path = Path(audit.output_json_path)
    md_path = Path(audit.output_md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Synthesis Text Manifest Audit",
        "",
        f"- passed: {audit.passed}",
        f"- manifest: `{audit.manifest_path}`",
        f"- samples: {audit.sample_count}",
        f"- claim eligible rows: {audit.claim_evidence_eligible_count}",
        f"- issues: {len(audit.issues)}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Languages",
        "",
    ]
    lines.extend(f"- `{language}`: {count}" for language, count in audit.language_counts.items()) if audit.language_counts else lines.append("- none")
    lines.extend(["", "## Scripts", ""])
    lines.extend(f"- `{script}`: {count}" for script, count in audit.script_counts.items()) if audit.script_counts else lines.append("- none")
    lines.extend(["", "## Source Schemas", ""])
    lines.extend(f"- `{schema}`: {count}" for schema, count in audit.source_schema_counts.items()) if audit.source_schema_counts else lines.append("- none")
    lines.extend(["", "## License Status", ""])
    lines.extend(f"- `{status}`: {count}" for status, count in audit.license_status_counts.items()) if audit.license_status_counts else lines.append("- none")
    if audit.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit.issues)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthesis_text_promotion_audit(audit: SynthesisTextPromotionAudit) -> None:
    json_path = Path(audit.output_json_path)
    md_path = Path(audit.output_md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Synthesis Text Promotion Audit",
        "",
        f"- passed: {audit.passed}",
        f"- manifest: `{audit.manifest_path}`",
        f"- samples: {audit.sample_count}",
        f"- basic manifest audit passed: {audit.basic_audit_passed}",
        f"- overlap count: {audit.overlap_count}",
        f"- require reviewed license: {audit.require_reviewed_license}",
        f"- issues: {len(audit.issues)}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Split Policies",
        "",
    ]
    lines.extend(f"- `{policy}`: {count}" for policy, count in audit.split_policy_counts.items()) if audit.split_policy_counts else lines.append("- none")
    lines.extend(["", "## License Status", ""])
    lines.extend(f"- `{status}`: {count}" for status, count in audit.license_status_counts.items()) if audit.license_status_counts else lines.append("- none")
    lines.extend(["", "## Overlaps", ""])
    lines.extend(f"- `{source}`: {count}" for source, count in audit.overlap_counts_by_source.items()) if audit.overlap_counts_by_source else lines.append("- none")
    if audit.overlap_examples:
        lines.extend(["", "## Overlap Examples", ""])
        for example in audit.overlap_examples:
            lines.append(
                f"- `{example['candidate_sample_id']}` overlaps `{example['excluded_source']}` "
                f"on `{example['text_sha256']}`"
            )
    if audit.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit.issues)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthesis_text_split_summary(summary: SynthesisTextSplitSummary) -> None:
    if summary.summary_json_path is None or summary.summary_md_path is None:
        raise DataValidationError("synthesis text split summary paths are required")
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Synthesis Text Split",
        "",
        f"- passed: {summary.passed}",
        f"- train manifest: `{summary.train_manifest}`",
        f"- eval manifest: `{summary.eval_manifest}`",
        f"- train samples: {summary.train_count}",
        f"- eval samples: {summary.eval_count}",
        f"- groups: {summary.group_count}",
        f"- group by: `{summary.group_by}`",
        f"- seed: {summary.seed}",
        f"- eval ratio: {summary.eval_ratio:g}",
        f"- split policy: `{summary.split_policy}`",
        f"- train promotion audit: `{'pass' if summary.train_promotion_passed else 'fail'}`",
        f"- eval promotion audit: `{'pass' if summary.eval_promotion_passed else 'fail'}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rendered_degradation_split_audit(audit: RenderedDegradationSplitAudit) -> None:
    json_path = Path(audit.output_json_path)
    md_path = Path(audit.output_md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rendered Degradation Split Audit",
        "",
        f"- passed: {audit.passed}",
        f"- train manifest: `{audit.train_manifest}`",
        f"- eval manifest: `{audit.eval_manifest}`",
        f"- train samples: {audit.train_count}",
        f"- eval samples: {audit.eval_count}",
        f"- train text hashes: {audit.train_text_hash_count}",
        f"- eval text hashes: {audit.eval_text_hash_count}",
        f"- expected profiles: `{', '.join(audit.expected_profiles) if audit.expected_profiles else 'not supplied'}`",
        f"- issues: {len(audit.issues)}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Train Profiles",
        "",
    ]
    lines.extend(f"- `{profile}`: {count}" for profile, count in audit.train_profile_counts.items()) if audit.train_profile_counts else lines.append("- none")
    lines.extend(["", "## Eval Profiles", ""])
    lines.extend(f"- `{profile}`: {count}" for profile, count in audit.eval_profile_counts.items()) if audit.eval_profile_counts else lines.append("- none")
    if audit.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit.issues)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rendered_line_manifest_audit(audit: RenderedLineManifestAudit) -> None:
    json_path = Path(audit.output_json_path)
    md_path = Path(audit.output_md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rendered Synthesis Line Manifest Audit",
        "",
        f"- passed: {audit.passed}",
        f"- manifest: `{audit.manifest_path}`",
        f"- samples: {audit.sample_count}",
        f"- claim eligible rows: {audit.claim_evidence_eligible_count}",
        f"- issues: {len(audit.issues)}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Scripts",
        "",
    ]
    lines.extend(f"- `{script}`: {count}" for script, count in audit.script_counts.items()) if audit.script_counts else lines.append("- none")
    lines.extend(["", "## Fonts", ""])
    lines.extend(f"- `{font}`: {count}" for font, count in audit.font_counts.items()) if audit.font_counts else lines.append("- none")
    if audit.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit.issues)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_limdic_json_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid Limdic JSON {path}: {exc}") from exc
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise DataValidationError(f"Limdic JSON list row {index} must be an object")
            missing = [field for field in LIMDIC_JSON_FIELDS if field not in item]
            if missing:
                raise DataValidationError(f"Limdic JSON list row {index} missing fields: {', '.join(missing)}")
            rows.append(dict(item))
        return rows, "limdic_dictionary_json_v1", "json"
    if isinstance(payload, dict) and isinstance(payload.get("Dic"), dict):
        rows = []
        for key, value in sorted(payload["Dic"].items()):
            if not isinstance(value, dict):
                raise DataValidationError(f"Limdic Firebase row {key} must be an object")
            missing = [field for field in LIMDIC_FIREBASE_FIELDS if field not in value]
            if missing:
                raise DataValidationError(f"Limdic Firebase row {key} missing fields: {', '.join(missing)}")
            row = dict(value)
            row["_firebase_key"] = str(key)
            rows.append(row)
        return rows, "limdic_firebase_json_v1", "json"
    raise DataValidationError(f"unsupported Limdic JSON schema: {path}")


def _load_limdic_tsv_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise DataValidationError(f"Limdic TSV has no header: {path}")
        missing = [field for field in LIMDIC_JSON_FIELDS if field not in reader.fieldnames]
        if missing:
            raise DataValidationError(f"Limdic TSV missing fields: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    return rows, "limdic_dictionary_tsv_v1", "tsv"


def _limdic_text_fields(row: dict[str, Any]) -> list[tuple[str, Any]]:
    if "headword_limbu" in row:
        return [(field, row.get(field)) for field in LIMDIC_TEXT_FIELDS]
    return [(field, row.get(field)) for field in LIMDIC_FIREBASE_TEXT_FIELDS]


def _load_tamang_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_tamang_csv_rows(path)
    if suffix == ".tsv":
        return _load_tamang_tsv_rows(path)
    if suffix == ".json":
        return _load_tamang_json_rows(path)
    raise DataValidationError(f"unsupported Tamang source suffix: {path}")


def _load_tamang_csv_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"Tamang CSV has no header: {path}")
        missing = [field for field in TAMANG_NEPTAM_FIELDS if field not in reader.fieldnames]
        if missing:
            raise DataValidationError(f"Tamang NepTam CSV missing fields: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    return rows, "tamang_neptam_parallel_csv_v1", "csv"


def _load_tamang_tsv_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise DataValidationError(f"Tamang TSV has no header: {path}")
        missing = [field for field in TAMANG_DICTIONARY_FIELDS if field not in reader.fieldnames]
        if missing:
            raise DataValidationError(f"Tamang dictionary TSV missing fields: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    return rows, "tamang_western_dictionary_tsv_v1", "tsv"


def _load_tamang_json_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid Tamang JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise DataValidationError(f"unsupported Tamang JSON schema: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise DataValidationError(f"Tamang dictionary JSON row {index} must be an object")
        missing = [field for field in TAMANG_DICTIONARY_FIELDS if field not in item]
        if missing:
            raise DataValidationError(f"Tamang dictionary JSON row {index} missing fields: {', '.join(missing)}")
        row = dict(item)
        row["row_id"] = str(index)
        rows.append(row)
    return rows, "tamang_western_dictionary_json_v1", "json"


def _tamang_text_fields(row: dict[str, Any], schema_name: str) -> list[tuple[str, Any]]:
    if schema_name == "tamang_neptam_parallel_csv_v1":
        return [(field, row.get(field)) for field in TAMANG_NEPTAM_TEXT_FIELDS]
    return [(field, row.get(field)) for field in TAMANG_DICTIONARY_TEXT_FIELDS]


def _tamang_source_name(schema_name: str) -> str:
    if schema_name == "tamang_neptam_parallel_csv_v1":
        return "LTK NepTam"
    return "LTK Western Tamang dictionary"


def _load_magar_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return _load_magar_tsv_rows(path)
    if suffix == ".json":
        return _load_magar_json_rows(path)
    raise DataValidationError(f"unsupported Magar source suffix: {path}")


def _load_magar_tsv_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise DataValidationError(f"Magar TSV has no header: {path}")
        missing = [field for field in MAGAR_DICTIONARY_FIELDS if field not in reader.fieldnames]
        if missing:
            raise DataValidationError(f"Magar dictionary TSV missing fields: {', '.join(missing)}")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(reader, start=1):
            value = dict(row)
            value["row_id"] = str(index)
            rows.append(value)
    return rows, "magar_western_dictionary_tsv_v1", "tsv"


def _load_magar_json_rows(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid Magar JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise DataValidationError(f"unsupported Magar JSON schema: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise DataValidationError(f"Magar dictionary JSON row {index} must be an object")
        missing = [field for field in MAGAR_DICTIONARY_FIELDS if field not in item]
        if missing:
            raise DataValidationError(f"Magar dictionary JSON row {index} missing fields: {', '.join(missing)}")
        row = dict(item)
        row["row_id"] = str(index)
        rows.append(row)
    return rows, "magar_western_dictionary_json_v1", "json"


def _magar_text_fields(row: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(field, row.get(field)) for field in MAGAR_DICTIONARY_TEXT_FIELDS]


def _row_text_sha256(row: dict[str, Any]) -> str | None:
    value = row.get("text_sha256")
    if isinstance(value, str) and value:
        return value
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("text_sha256")
        if isinstance(value, str) and value:
            return value
    text = row.get("text")
    if isinstance(text, str):
        return sha256_text(text)
    return None


def _collect_text_hashes_from_exclusion(path: Path) -> set[str]:
    if not path.exists():
        raise DataValidationError(f"excluded text source does not exist: {path}")
    if path.is_dir():
        hashes: set[str] = set()
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".md", ".txt"}:
                hashes.update(_collect_text_hashes_from_exclusion(child))
        return hashes
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _collect_text_hashes_from_jsonl(path)
    if suffix == ".json":
        return _collect_text_hashes_from_reference_json(path)
    if suffix in {".md", ".txt"}:
        text = _clean_synthesis_text(path.read_text(encoding="utf-8"))
        return {sha256_text(text)} if text else set()
    raise DataValidationError(f"unsupported excluded text source suffix for {path}: {suffix}")


def _collect_text_hashes_from_jsonl(path: Path) -> set[str]:
    hashes: set[str] = set()
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        text_hash = _row_text_sha256(row)
        if text_hash:
            hashes.add(text_hash)
    return hashes


def _collect_text_hashes_from_reference_json(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid excluded reference JSON {path}: {exc}") from exc
    hashes: set[str] = set()
    for text in _reference_text_values(payload):
        cleaned = _clean_synthesis_text(text)
        if cleaned:
            hashes.add(sha256_text(cleaned))
    return hashes


def _reference_text_values(payload: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(payload, dict):
        return values
    for key in ("text", "markdown"):
        value = payload.get(key)
        if isinstance(value, str):
            values.append(value)
    reading_order = payload.get("reading_order")
    if isinstance(reading_order, list):
        for item in reading_order:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                values.append(str(item["text"]))
    tables = payload.get("tables")
    if isinstance(tables, list):
        values.extend(_reference_table_text_values(tables))
    figures = payload.get("figures")
    if isinstance(figures, list):
        for figure in figures:
            if isinstance(figure, dict) and isinstance(figure.get("caption"), str):
                values.append(str(figure["caption"]))
    return values


def _reference_table_text_values(tables: list[Any]) -> list[str]:
    values: list[str] = []
    for table in tables:
        if isinstance(table, list):
            for row in table:
                if isinstance(row, list):
                    values.extend(str(cell) for cell in row if cell is not None)
        elif isinstance(table, dict):
            cells = table.get("cells")
            if isinstance(cells, list):
                for cell in cells:
                    if isinstance(cell, dict):
                        for key in ("text", "value", "content"):
                            value = cell.get(key)
                            if isinstance(value, str):
                                values.append(value)
                                break
    return values


def _group_synthesis_text_rows(rows: list[dict[str, Any]], *, group_by: str) -> dict[str, list[dict[str, Any]]]:
    if group_by not in {"text_sha256", "source_path", "source_row_id", "sample_id"}:
        raise DataValidationError("group_by must be one of: text_sha256, source_path, source_row_id, sample_id")
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows, start=1):
        key = _synthesis_text_group_key(row, group_by=group_by, row_number=index)
        groups.setdefault(key, []).append(row)
    return groups


def _synthesis_text_group_key(row: dict[str, Any], *, group_by: str, row_number: int) -> str:
    if group_by == "text_sha256":
        value = _row_text_sha256(row)
    elif group_by == "source_path":
        value = row.get("source_path")
    elif group_by == "source_row_id":
        value = row.get("source_row_id")
    elif group_by == "sample_id":
        value = row.get("sample_id")
    else:
        raise DataValidationError("group_by must be one of: text_sha256, source_path, source_row_id, sample_id")
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"row {row_number} cannot be grouped by {group_by}: missing value")
    return value


def _with_synthesis_split(row: dict[str, Any], split: str, split_policy: str) -> dict[str, Any]:
    updated = dict(row)
    updated["split"] = split
    metadata = dict(updated.get("metadata") if isinstance(updated.get("metadata"), dict) else {})
    metadata["split_policy"] = split_policy
    updated["metadata"] = metadata
    return updated


def _write_synthesis_text_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_toolkit_text_fields(values: list[str]) -> dict[str, str]:
    if not values:
        raise DataValidationError("at least one --text-field FIELD=language value is required")
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise DataValidationError(f"invalid text field mapping {value!r}; expected FIELD=language")
        field_name, language = value.split("=", 1)
        field_name = field_name.strip()
        language = language.strip()
        if not field_name or not language:
            raise DataValidationError(f"invalid text field mapping {value!r}; expected FIELD=language")
        if field_name in parsed:
            raise DataValidationError(f"duplicate text field mapping for {field_name}")
        parsed[field_name] = language
    return parsed


def _load_toolkit_parallel_rows(
    path: Path,
    *,
    required_fields: list[str],
    row_id_field: str | None,
    metadata_fields: list[str],
) -> tuple[list[dict[str, Any]], str, str]:
    suffix = path.suffix.lower()
    if suffix not in TOOLKIT_PARALLEL_SUPPORTED_SUFFIXES:
        raise DataValidationError(f"unsupported toolkit source suffix for {path}: {suffix}")
    if suffix == ".jsonl":
        return _load_toolkit_parallel_jsonl_rows(
            path,
            required_fields=required_fields,
            row_id_field=row_id_field,
            metadata_fields=metadata_fields,
        )
    delimiter = "\t" if suffix == ".tsv" else ","
    source_format = "tsv" if suffix == ".tsv" else "csv"
    return _load_toolkit_parallel_delimited_rows(
        path,
        delimiter=delimiter,
        source_format=source_format,
        required_fields=required_fields,
        row_id_field=row_id_field,
        metadata_fields=metadata_fields,
    )


def _load_toolkit_parallel_jsonl_rows(
    path: Path,
    *,
    required_fields: list[str],
    row_id_field: str | None,
    metadata_fields: list[str],
) -> tuple[list[dict[str, Any]], str, str]:
    rows: list[dict[str, Any]] = []
    all_required = _toolkit_required_fields(required_fields, row_id_field=row_id_field, metadata_fields=metadata_fields)
    for row_index, row in enumerate(read_jsonl(path), start=1):
        if not isinstance(row, dict):
            raise DataValidationError(f"toolkit JSONL row {path}:{row_index} must be an object")
        missing = [field for field in all_required if field not in row]
        if missing:
            raise DataValidationError(f"toolkit JSONL row {path}:{row_index} missing fields: {', '.join(missing)}")
        rows.append(dict(row))
    if not rows:
        raise DataValidationError(f"toolkit JSONL source is empty: {path}")
    return rows, "toolkit_parallel_jsonl_explicit_fields_v1", "jsonl"


def _load_toolkit_parallel_delimited_rows(
    path: Path,
    *,
    delimiter: str,
    source_format: str,
    required_fields: list[str],
    row_id_field: str | None,
    metadata_fields: list[str],
) -> tuple[list[dict[str, Any]], str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataValidationError(f"toolkit {source_format.upper()} has no header: {path}")
        normalized_fieldnames = [field.strip() for field in reader.fieldnames]
        all_required = _toolkit_required_fields(
            required_fields,
            row_id_field=row_id_field,
            metadata_fields=metadata_fields,
        )
        missing = [field for field in all_required if field not in normalized_fieldnames]
        if missing:
            raise DataValidationError(f"toolkit {source_format.upper()} missing fields: {', '.join(missing)}")
        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(reader, start=1):
            value = {str(key).strip(): item for key, item in row.items() if key is not None}
            value["row_number"] = str(row_index)
            rows.append(value)
    return rows, f"toolkit_parallel_{source_format}_explicit_fields_v1", source_format


def _toolkit_required_fields(
    required_fields: list[str],
    *,
    row_id_field: str | None,
    metadata_fields: list[str],
) -> list[str]:
    fields = [field for field in required_fields if field]
    if row_id_field:
        fields.append(row_id_field)
    fields.extend(metadata_fields)
    return sorted(set(fields))


def _clean_synthesis_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(item) for item in value if item is not None)
    else:
        text = str(value)
    text = _strip_html(text)
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return normalize_ocr_text(text.strip(), collapse_spaces=True)


def _strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(" ".join(self._chunks).split())

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)


def _dominant_script(text: str) -> str:
    counts = _unicode_block_counts(text)
    has_limbu = counts.get("Limbu", 0) > 0
    has_devanagari = counts.get("Devanagari", 0) > 0
    has_latin = counts.get("Basic Latin", 0) > 0 or counts.get("Latin-1 Supplement", 0) > 0
    if has_limbu and has_devanagari:
        return "Mixed_Limbu_Devanagari"
    if has_limbu and has_latin:
        return "Mixed_Limbu_Latin"
    if has_devanagari and has_latin:
        return "Mixed_Devanagari_Latin"
    if has_limbu:
        return "Limbu"
    if has_devanagari:
        return "Devanagari"
    if has_latin:
        return "Latin"
    return "Other"


def _write_synthesis_text_manifest(entries: list[SynthesisTextEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def _synthesis_text_summary(
    entries: list[SynthesisTextEntry],
    *,
    manifest_path: Path,
    summary_json_path: Path,
    summary_md_path: Path,
    source_count: int,
    rejected_count: int,
    duplicate_count: int,
    source_paths: list[str],
    warnings: list[str],
) -> SynthesisTextPrepareSummary:
    field_counts: dict[str, int] = {}
    script_counts: dict[str, int] = {}
    for entry in entries:
        field_counts[entry.source_field] = field_counts.get(entry.source_field, 0) + 1
        script_counts[entry.script] = script_counts.get(entry.script, 0) + 1
    return SynthesisTextPrepareSummary(
        manifest_path=str(manifest_path),
        summary_json_path=str(summary_json_path),
        summary_md_path=str(summary_md_path),
        sample_count=len(entries),
        source_count=source_count,
        rejected_count=rejected_count,
        duplicate_count=duplicate_count,
        field_counts=dict(sorted(field_counts.items())),
        script_counts=dict(sorted(script_counts.items())),
        source_paths=source_paths,
        warnings=warnings,
    )


def _write_synthesis_text_summary(summary: SynthesisTextPrepareSummary) -> None:
    json_path = Path(summary.summary_json_path)
    md_path = Path(summary.summary_md_path)
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    title = _synthesis_text_summary_title(Path(summary.manifest_path).name)
    lines = [
        f"# {title}",
        "",
        f"- passed: {summary.passed}",
        f"- manifest: `{summary.manifest_path}`",
        f"- samples: {summary.sample_count}",
        f"- sources: {summary.source_count}",
        f"- rejected: {summary.rejected_count}",
        f"- duplicates skipped: {summary.duplicate_count}",
        f"- warnings: {len(summary.warnings)}",
        "",
        "## Fields",
        "",
    ]
    lines.extend(f"- `{field}`: {count}" for field, count in summary.field_counts.items()) if summary.field_counts else lines.append("- none")
    lines.extend(["", "## Scripts", ""])
    lines.extend(f"- `{script}`: {count}" for script, count in summary.script_counts.items()) if summary.script_counts else lines.append("- none")
    lines.extend(["", "## Sources", ""])
    lines.extend(f"- `{source}`" for source in summary.source_paths)
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthesis_text_summary_title(manifest_name: str) -> str:
    if "bible-brain" in manifest_name:
        return "Bible Brain Text Manifest"
    if "limbu-limdic" in manifest_name:
        return "Limbu Limdic Text Manifest"
    if "limbu-unicode" in manifest_name:
        return "Limbu Unicode Text Manifest"
    if "tamang" in manifest_name:
        return "Tamang Text Manifest"
    if "magar" in manifest_name:
        return "Magar Text Manifest"
    return "Synthesis Text Manifest"


def _audit_text_sample(file_path: Path, *, root: Path, sample_bytes: int) -> TextSampleAudit:
    warnings: list[str] = []
    data = file_path.read_bytes()[:sample_bytes]
    text = data.decode("utf-8", errors="replace")
    if "\ufffd" in text:
        warnings.append("utf8_replacement_char_seen")
    return TextSampleAudit(
        path=str(file_path),
        relative_path=str(file_path.relative_to(root)),
        suffix=file_path.suffix.lower() or "<none>",
        size_bytes=file_path.stat().st_size,
        sha256=sha256_file(file_path),
        decoded_chars=len(text),
        line_count=text.count("\n") + (1 if text else 0),
        unicode_blocks=_unicode_block_counts(text),
        warnings=warnings,
    )


def _text_sample_priority(file_path: Path, root: Path) -> tuple[int, int, str]:
    relative = file_path.relative_to(root)
    parts = set(relative.parts)
    suffix = file_path.suffix.lower()
    score = 0
    if "data" in parts:
        score -= 50
    if suffix in {".tsv", ".csv", ".json", ".jsonl", ".xml", ".txt", ".md"}:
        score -= 20
    if suffix in {".html", ".htm", ".php"}:
        score += 10
    if {"apk", "decompiled"} & parts:
        score += 100
    if "res" in parts or "assets" in parts:
        score += 20
    try:
        size = file_path.stat().st_size
    except OSError:
        size = 0
    return (score, -size, str(relative))


def _unicode_block_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        block = _unicode_block_name(ord(char))
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def _limbu_char_count(text: str) -> int:
    return sum(1 for char in text if 0x1900 <= ord(char) <= 0x194F)


def _unicode_block_name(codepoint: int) -> str:
    if 0x0900 <= codepoint <= 0x097F:
        return "Devanagari"
    if 0x1900 <= codepoint <= 0x194F:
        return "Limbu"
    if 0x16D40 <= codepoint <= 0x16D7F:
        return "Kirat Rai"
    if 0x1C50 <= codepoint <= 0x1C7F:
        return "Ol Chiki"
    if 0x0041 <= codepoint <= 0x007A:
        return "Basic Latin"
    if codepoint <= 0x007F:
        return "ASCII punctuation/control"
    if 0x0080 <= codepoint <= 0x00FF:
        return "Latin-1 Supplement"
    return "Other"


def _write_synthesis_resource_audit(audit: SynthesisResourceAudit) -> None:
    json_path = Path(audit.output_json_path)
    md_path = Path(audit.output_md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Synthesis Resource Audit",
        "",
        f"- passed: {audit.passed}",
        f"- roots: {audit.existing_root_count}/{audit.root_count}",
        f"- files: {audit.total_files}",
        f"- size: {_format_bytes(audit.total_size_bytes)}",
        f"- warnings: {len(audit.warnings)}",
        "",
        "## Roots",
        "",
    ]
    for root in audit.roots:
        title = root.label or root.root
        lines.extend(
            [
                f"### {title}",
                "",
                f"- path: `{root.root}`",
                f"- exists: {root.exists}",
                f"- files: {root.total_files}",
                f"- size: {_format_bytes(root.total_size_bytes)}",
                f"- text files: {root.text_file_count}",
                f"- structured files: {root.structured_file_count}",
                f"- font files: {root.font_file_count}",
                f"- document files: {root.document_file_count}",
                "",
                "Suffix counts:",
                "",
            ]
        )
        if root.suffix_counts:
            lines.extend(f"- `{suffix}`: {count}" for suffix, count in root.suffix_counts.items())
        else:
            lines.append("- none")
        lines.extend(["", "Sampled text files:", ""])
        if root.sampled_files:
            for sample in root.sampled_files:
                top_blocks = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(sample.unicode_blocks.items(), key=lambda item: item[1], reverse=True)[:5]
                )
                lines.append(
                    f"- `{sample.relative_path}` sha256={sample.sha256[:12]} "
                    f"chars={sample.decoded_chars} blocks={top_blocks or 'none'}"
                )
        else:
            lines.append("- none")
        if root.warnings:
            lines.extend(["", "Warnings:", ""])
            lines.extend(f"- {warning}" for warning in root.warnings)
        lines.append("")
    if audit.warnings:
        lines.extend(["## All Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{value} B"
