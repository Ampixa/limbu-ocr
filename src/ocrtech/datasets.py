"""Manifest audit and leakage-safe split helpers."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, read_jsonl, sha256_path, sha256_text, write_manifest
from .normalization import script_counts, unsupported_characters
from .validation import check_leakage


@dataclass(slots=True)
class ManifestAudit:
    manifest_path: str
    passed: bool
    sample_count: int
    dataset_counts: dict[str, int]
    split_counts: dict[str, int]
    slice_counts: dict[str, int]
    script_counts: dict[str, int]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "dataset_counts": self.dataset_counts,
            "split_counts": self.split_counts,
            "slice_counts": self.slice_counts,
            "script_counts": self.script_counts,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class SplitSummary:
    train_manifest: str
    eval_manifest: str
    train_count: int
    eval_count: int
    group_count: int
    group_by: str
    seed: int
    leakage_passed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "group_count": self.group_count,
            "group_by": self.group_by,
            "seed": self.seed,
            "leakage_passed": self.leakage_passed,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class MergeSummary:
    output_manifest: str
    input_manifests: list[str]
    sample_count: int
    dataset_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_manifest": self.output_manifest,
            "input_manifests": self.input_manifests,
            "sample_count": self.sample_count,
            "dataset_counts": self.dataset_counts,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RebalanceSummary:
    output_manifest: str
    source_manifest: str
    original_count: int
    rebalanced_count: int
    target_count: int
    slice_counts_before: dict[str, int]
    slice_counts_after: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_manifest": self.output_manifest,
            "source_manifest": self.source_manifest,
            "original_count": self.original_count,
            "rebalanced_count": self.rebalanced_count,
            "target_count": self.target_count,
            "slice_counts_before": self.slice_counts_before,
            "slice_counts_after": self.slice_counts_after,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class FilterSummary:
    output_manifest: str
    source_manifest: str
    source_count: int
    selected_count: int
    sample_id_filters: list[str] = field(default_factory=list)
    slice_filters: list[str] = field(default_factory=list)
    document_type_filters: list[str] = field(default_factory=list)
    degradation_filters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_manifest": self.output_manifest,
            "source_manifest": self.source_manifest,
            "source_count": self.source_count,
            "selected_count": self.selected_count,
            "sample_id_filters": self.sample_id_filters,
            "slice_filters": self.slice_filters,
            "document_type_filters": self.document_type_filters,
            "degradation_filters": self.degradation_filters,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RecognizerCorpusProfile:
    manifest_path: str
    sample_count: int
    dataset_counts: dict[str, int]
    slice_counts: dict[str, int]
    script_counts: dict[str, int]
    script_sample_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "dataset_counts": self.dataset_counts,
            "slice_counts": self.slice_counts,
            "script_counts": self.script_counts,
            "script_sample_counts": self.script_sample_counts,
        }


@dataclass(slots=True)
class RecognizerCorpusAudit:
    train_manifest: str
    eval_manifest: str
    passed: bool
    train_profile: RecognizerCorpusProfile
    eval_profile: RecognizerCorpusProfile
    leakage_passed: bool
    require_eval_real: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "passed": self.passed,
            "train_profile": self.train_profile.to_dict(),
            "eval_profile": self.eval_profile.to_dict(),
            "leakage_passed": self.leakage_passed,
            "require_eval_real": self.require_eval_real,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class RecognizerCorpusBuildSummary:
    output_manifest: str
    input_manifests: list[str]
    source_count: int
    output_count: int
    target_counts: dict[str, int]
    bucket_counts_before: dict[str, int]
    bucket_counts_after: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_manifest": self.output_manifest,
            "input_manifests": self.input_manifests,
            "source_count": self.source_count,
            "output_count": self.output_count,
            "target_counts": self.target_counts,
            "bucket_counts_before": self.bucket_counts_before,
            "bucket_counts_after": self.bucket_counts_after,
            "warnings": self.warnings,
        }


def audit_manifest(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    verify_hashes: bool = True,
    require_slices: list[str] | None = None,
    strict_chars: bool = False,
) -> ManifestAudit:
    path = Path(manifest_path)
    rows = list(read_jsonl(path))
    entries: list[ManifestEntry] = []
    issues: list[str] = []
    warnings: list[str] = []
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    image_hashes: set[str] = set()
    sample_hashes: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    slice_counts: Counter[str] = Counter()
    aggregate_scripts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        try:
            entry = ManifestEntry.from_dict(row)
        except DataValidationError as exc:
            issues.append(f"row {row_index}: {exc}")
            continue
        entries.append(entry)
        if entry.sample_id in sample_ids:
            issues.append(f"duplicate sample_id: {entry.sample_id}")
        sample_ids.add(entry.sample_id)
        if entry.image_path in image_paths:
            warnings.append(f"duplicate image_path: {entry.image_path}")
        image_paths.add(entry.image_path)
        input_path = Path(entry.image_path)
        if not input_path.exists():
            issues.append(f"missing image_path for {entry.sample_id}: {input_path}")
        elif verify_hashes:
            actual = sha256_path(input_path)
            if entry.sha256 and entry.sha256 != actual:
                issues.append(f"sha256 mismatch for {entry.sample_id}: manifest={entry.sha256} actual={actual}")
            if actual in image_hashes:
                warnings.append(f"duplicate image sha256: {actual}")
            image_hashes.add(actual)
        elif entry.sha256:
            if entry.sha256 in image_hashes:
                warnings.append(f"duplicate image sha256: {entry.sha256}")
            image_hashes.add(entry.sha256)

        text_sha = sha256_text(entry.text)
        metadata_text_sha = entry.metadata.get("text_sha256") if entry.metadata else None
        if metadata_text_sha and metadata_text_sha != text_sha:
            issues.append(f"text_sha256 mismatch for {entry.sample_id}")
        sample_sha = _entry_sample_signature(entry)
        if sample_sha in sample_hashes:
            warnings.append(f"duplicate sample signature: {sample_sha}")
        sample_hashes.add(sample_sha)
        if strict_chars:
            unsupported = unsupported_characters(entry.text)
            if unsupported:
                issues.append(f"unsupported characters for {entry.sample_id}: {unsupported}")
        dataset_counts[entry.dataset] += 1
        split_counts[entry.split] += 1
        for slice_name in entry_slices(entry):
            slice_counts[slice_name] += 1
        aggregate_scripts.update(script_counts(entry.text))

    for required in require_slices or []:
        if slice_counts.get(required, 0) == 0:
            issues.append(f"required slice missing: {required}")

    audit = ManifestAudit(
        manifest_path=str(path),
        passed=not issues,
        sample_count=len(entries),
        dataset_counts=dict(sorted(dataset_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        slice_counts=dict(sorted(slice_counts.items())),
        script_counts=dict(sorted(aggregate_scripts.items())),
        issues=issues,
        warnings=warnings,
    )
    if output_dir is not None:
        _write_audit(audit, Path(output_dir))
    return audit


def build_recognizer_corpus(
    manifest_paths: list[str | Path],
    output_manifest: str | Path,
    *,
    target_latin_only: int = 0,
    target_devanagari_only: int = 0,
    target_mixed: int = 0,
    target_real: int = 0,
    target_synthetic: int = 0,
    seed: int = 13,
) -> RecognizerCorpusBuildSummary:
    if not manifest_paths:
        raise DataValidationError("build-recognizer-corpus requires at least one input manifest")
    targets = {
        "latin_only": target_latin_only,
        "devanagari_only": target_devanagari_only,
        "mixed_devanagari_latin": target_mixed,
        "real": target_real,
        "synthetic": target_synthetic,
    }
    invalid = [name for name, value in targets.items() if value < 0]
    if invalid:
        raise DataValidationError(f"recognizer corpus targets must be non-negative: {', '.join(invalid)}")
    if all(value == 0 for value in targets.values()):
        raise DataValidationError("build-recognizer-corpus requires at least one positive target")

    entries: list[ManifestEntry] = []
    seen_sample_ids: set[str] = set()
    seen_signatures: set[str] = set()
    warnings: list[str] = []
    for manifest_path in manifest_paths:
        for entry in load_manifest(manifest_path):
            if entry.sample_id in seen_sample_ids:
                raise DataValidationError(f"duplicate sample_id across recognizer corpus inputs: {entry.sample_id}")
            seen_sample_ids.add(entry.sample_id)
            signature = _entry_sample_signature(entry)
            if signature in seen_signatures:
                warnings.append(f"duplicate sample signature across recognizer corpus inputs: {entry.sample_id}")
            seen_signatures.add(signature)
            entries.append(entry)
    if not entries:
        raise DataValidationError("build-recognizer-corpus requires non-empty input manifests")

    output_entries = list(entries)
    rng = random.Random(seed)
    duplicate_index = 0
    counts_before = _recognizer_bucket_counts(output_entries)
    for bucket_name, target in targets.items():
        if target <= 0:
            continue
        current = _recognizer_bucket_counts(output_entries).get(bucket_name, 0)
        if current >= target:
            continue
        bucket = [entry for entry in output_entries if _entry_in_recognizer_bucket(entry, bucket_name)]
        if not bucket:
            raise DataValidationError(f"recognizer corpus cannot satisfy missing bucket: {bucket_name}")
        deficit = target - current
        warnings.append(f"upsampled recognizer bucket {bucket_name} by {deficit} samples")
        for _ in range(deficit):
            source = rng.choice(bucket)
            duplicate_index += 1
            output_entries.append(_duplicate_recognizer_corpus_entry(source, bucket_name=bucket_name, duplicate_index=duplicate_index))

    out_path = Path(output_manifest)
    write_manifest(output_entries, out_path)
    summary = RecognizerCorpusBuildSummary(
        output_manifest=str(out_path),
        input_manifests=[str(Path(path)) for path in manifest_paths],
        source_count=len(entries),
        output_count=len(output_entries),
        target_counts={key: value for key, value in targets.items() if value > 0},
        bucket_counts_before=_recognizer_bucket_counts(entries),
        bucket_counts_after=_recognizer_bucket_counts(output_entries),
        warnings=warnings,
    )
    _write_recognizer_corpus_build_summary(summary, out_path)
    return summary


def audit_recognizer_corpus(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path | None = None,
    *,
    min_train_samples: int = 1000,
    min_eval_samples: int = 100,
    min_train_english: int = 200,
    min_train_devanagari: int = 200,
    min_train_mixed: int = 25,
    min_eval_english: int = 50,
    min_eval_devanagari: int = 50,
    min_eval_mixed: int = 10,
    min_train_latin_only: int = 50,
    min_train_devanagari_only: int = 50,
    min_eval_latin_only: int = 10,
    min_eval_devanagari_only: int = 10,
    min_train_real: int = 100,
    min_train_synthetic: int = 100,
    require_eval_real: bool = False,
) -> RecognizerCorpusAudit:
    thresholds = {
        "min_train_samples": min_train_samples,
        "min_eval_samples": min_eval_samples,
        "min_train_english": min_train_english,
        "min_train_devanagari": min_train_devanagari,
        "min_train_mixed": min_train_mixed,
        "min_eval_english": min_eval_english,
        "min_eval_devanagari": min_eval_devanagari,
        "min_eval_mixed": min_eval_mixed,
        "min_train_latin_only": min_train_latin_only,
        "min_train_devanagari_only": min_train_devanagari_only,
        "min_eval_latin_only": min_eval_latin_only,
        "min_eval_devanagari_only": min_eval_devanagari_only,
        "min_train_real": min_train_real,
        "min_train_synthetic": min_train_synthetic,
    }
    invalid_thresholds = [name for name, value in thresholds.items() if value < 0]
    if invalid_thresholds:
        raise DataValidationError(f"recognizer corpus thresholds must be non-negative: {', '.join(invalid_thresholds)}")

    train_path = Path(train_manifest)
    eval_path = Path(eval_manifest)
    train_entries = load_manifest(train_path)
    eval_entries = load_manifest(eval_path)
    train_profile = _recognizer_corpus_profile(train_path, train_entries)
    eval_profile = _recognizer_corpus_profile(eval_path, eval_entries)
    leakage = check_leakage([train_path], eval_path)

    issues: list[str] = []
    warnings: list[str] = []
    _append_minimum_issue(issues, "train samples", train_profile.sample_count, min_train_samples)
    _append_minimum_issue(issues, "eval samples", eval_profile.sample_count, min_eval_samples)
    _append_minimum_issue(issues, "train latin-script samples", train_profile.script_sample_counts.get("contains_latin", 0), min_train_english)
    _append_minimum_issue(issues, "train devanagari-script samples", train_profile.script_sample_counts.get("contains_devanagari", 0), min_train_devanagari)
    _append_minimum_issue(issues, "train mixed-script samples", train_profile.script_sample_counts.get("mixed_devanagari_latin", 0), min_train_mixed)
    _append_minimum_issue(issues, "eval latin-script samples", eval_profile.script_sample_counts.get("contains_latin", 0), min_eval_english)
    _append_minimum_issue(issues, "eval devanagari-script samples", eval_profile.script_sample_counts.get("contains_devanagari", 0), min_eval_devanagari)
    _append_minimum_issue(issues, "eval mixed-script samples", eval_profile.script_sample_counts.get("mixed_devanagari_latin", 0), min_eval_mixed)
    _append_minimum_issue(issues, "train latin-only samples", train_profile.script_sample_counts.get("latin_only", 0), min_train_latin_only)
    _append_minimum_issue(issues, "train devanagari-only samples", train_profile.script_sample_counts.get("devanagari_only", 0), min_train_devanagari_only)
    _append_minimum_issue(issues, "eval latin-only samples", eval_profile.script_sample_counts.get("latin_only", 0), min_eval_latin_only)
    _append_minimum_issue(issues, "eval devanagari-only samples", eval_profile.script_sample_counts.get("devanagari_only", 0), min_eval_devanagari_only)
    _append_minimum_issue(issues, "train real slice samples", train_profile.slice_counts.get("real", 0), min_train_real)
    _append_minimum_issue(issues, "train synthetic slice samples", train_profile.slice_counts.get("synthetic", 0), min_train_synthetic)
    if require_eval_real:
        if eval_profile.slice_counts.get("real", 0) == 0:
            issues.append("eval real slice samples 0 < required 1 for claim-grade evaluation")
        if eval_profile.slice_counts.get("synthetic", 0) > 0:
            issues.append("eval manifest contains synthetic slice samples; claim-grade evaluation must use real held-out documents")
    if not leakage.passed:
        issues.append(f"train/eval leakage detected: {leakage.overlaps}")
    if eval_profile.slice_counts.get("hard_eval", 0) == 0:
        warnings.append("eval manifest has no hard_eval slice; recognizer admission should still use a separate hard gate")
    if train_profile.script_sample_counts.get("latin_only", 0) == 0:
        warnings.append("train manifest has no latin-only samples; English preservation is unlikely")
    if train_profile.script_sample_counts.get("devanagari_only", 0) == 0:
        warnings.append("train manifest has no devanagari-only samples; Nepali isolation is unlikely")

    audit = RecognizerCorpusAudit(
        train_manifest=str(train_path),
        eval_manifest=str(eval_path),
        passed=not issues,
        train_profile=train_profile,
        eval_profile=eval_profile,
        leakage_passed=leakage.passed,
        require_eval_real=require_eval_real,
        issues=issues,
        warnings=warnings,
    )
    if output_dir is not None:
        _write_recognizer_corpus_audit(audit, Path(output_dir))
    return audit


def split_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    eval_ratio: float = 0.15,
    seed: int = 13,
    stratify_by: str = "slices",
    group_by: str = "sample_signature",
    train_name: str = "train.jsonl",
    eval_name: str = "eval.jsonl",
) -> SplitSummary:
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise DataValidationError("eval_ratio must be between 0 and 1")
    entries = load_manifest(manifest_path)
    if len(entries) < 2:
        raise DataValidationError("split-manifest requires at least two entries")
    groups = _group_entries(entries, group_by=group_by)
    if len(groups) < 2:
        raise DataValidationError("split-manifest requires at least two unique sample groups")
    rng = random.Random(seed)
    strata: dict[str, list[tuple[str, list[ManifestEntry]]]] = defaultdict(list)
    for group_key, group_entries in groups.items():
        strata[_stratum_key(group_entries, stratify_by)].append((group_key, group_entries))

    eval_group_keys: set[str] = set()
    for stratum, stratum_groups in sorted(strata.items()):
        shuffled = list(stratum_groups)
        rng.shuffle(shuffled)
        target = max(1, round(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
        for group_key, _ in shuffled[:target]:
            eval_group_keys.add(group_key)

    if not eval_group_keys:
        eval_group_keys.add(sorted(groups)[0])
    if len(eval_group_keys) == len(groups):
        eval_group_keys.remove(sorted(eval_group_keys)[-1])

    train_entries: list[ManifestEntry] = []
    eval_entries: list[ManifestEntry] = []
    for group_key, group_entries in groups.items():
        target = eval_entries if group_key in eval_group_keys else train_entries
        split_name = "eval" if group_key in eval_group_keys else "train"
        target.extend(_with_split(entry, split_name) for entry in group_entries)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / train_name
    eval_path = out / eval_name
    write_manifest(train_entries, train_path)
    write_manifest(eval_entries, eval_path)
    leakage = check_leakage([train_path], eval_path)
    warnings: list[str] = []
    if not leakage.passed:
        raise DataValidationError(f"generated split has leakage: {leakage.to_dict()}")
    if not eval_entries:
        warnings.append("eval split is empty")
    if not train_entries:
        warnings.append("train split is empty")
    summary = SplitSummary(
        train_manifest=str(train_path),
        eval_manifest=str(eval_path),
        train_count=len(train_entries),
        eval_count=len(eval_entries),
        group_count=len(groups),
        group_by=group_by,
        seed=seed,
        leakage_passed=leakage.passed,
        warnings=warnings,
    )
    _write_split_summary(summary, out)
    return summary


def merge_manifests(
    manifest_paths: list[str | Path],
    output_manifest: str | Path,
) -> MergeSummary:
    if len(manifest_paths) < 2:
        raise DataValidationError("merge-manifests requires at least two input manifests")
    merged: list[ManifestEntry] = []
    sample_ids: set[str] = set()
    sample_signatures: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    warnings: list[str] = []
    ordered_inputs = [str(Path(path)) for path in manifest_paths]

    for manifest_path in manifest_paths:
        entries = load_manifest(manifest_path)
        for entry in entries:
            if entry.sample_id in sample_ids:
                raise DataValidationError(f"duplicate sample_id across merged manifests: {entry.sample_id}")
            sample_ids.add(entry.sample_id)
            signature = _entry_sample_signature(entry)
            if signature in sample_signatures:
                warnings.append(f"duplicate sample signature across merged manifests: {entry.sample_id}")
            sample_signatures.add(signature)
            dataset_counts[entry.dataset] += 1
            merged.append(entry)
    out_path = Path(output_manifest)
    write_manifest(merged, out_path)
    summary = MergeSummary(
        output_manifest=str(out_path),
        input_manifests=ordered_inputs,
        sample_count=len(merged),
        dataset_counts=dict(sorted(dataset_counts.items())),
        warnings=warnings,
    )
    _write_merge_summary(summary, out_path)
    return summary


def rebalance_manifest(
    manifest_path: str | Path,
    output_manifest: str | Path,
    *,
    slices: list[str],
    target_count: int | None = None,
    seed: int = 13,
) -> RebalanceSummary:
    if not slices:
        raise DataValidationError("rebalance-manifest requires at least one --slice")
    requested_slices = [slice_name.strip() for slice_name in slices if slice_name.strip()]
    if not requested_slices:
        raise DataValidationError("rebalance-manifest requires at least one non-empty --slice")
    entries = load_manifest(manifest_path)
    if not entries:
        raise DataValidationError("rebalance-manifest requires a non-empty manifest")
    buckets: dict[str, list[ManifestEntry]] = {slice_name: [] for slice_name in requested_slices}
    for entry in entries:
        entry_slice_set = set(entry_slices(entry))
        for slice_name in requested_slices:
            if slice_name in entry_slice_set:
                buckets[slice_name].append(entry)
    missing = [slice_name for slice_name, bucket in buckets.items() if not bucket]
    if missing:
        raise DataValidationError(f"rebalance-manifest missing requested slices: {', '.join(missing)}")
    before_counts = {slice_name: len(bucket) for slice_name, bucket in buckets.items()}
    target = target_count if target_count is not None else max(before_counts.values())
    if target <= 0:
        raise DataValidationError("rebalance-manifest target_count must be positive")
    rng = random.Random(seed)
    rebalanced = list(entries)
    duplicate_index = 0
    warnings: list[str] = []
    for slice_name, bucket in buckets.items():
        deficit = target - len(bucket)
        if deficit <= 0:
            continue
        warnings.append(f"upsampled slice {slice_name} by {deficit} samples")
        for _ in range(deficit):
            source = rng.choice(bucket)
            duplicate_index += 1
            rebalanced.append(_duplicate_entry(source, slice_name=slice_name, duplicate_index=duplicate_index))
    out_path = Path(output_manifest)
    write_manifest(rebalanced, out_path)
    after_counts = Counter[str]()
    for entry in rebalanced:
        for slice_name in requested_slices:
            if slice_name in set(entry_slices(entry)):
                after_counts[slice_name] += 1
    summary = RebalanceSummary(
        output_manifest=str(out_path),
        source_manifest=str(Path(manifest_path)),
        original_count=len(entries),
        rebalanced_count=len(rebalanced),
        target_count=target,
        slice_counts_before=dict(sorted(before_counts.items())),
        slice_counts_after=dict(sorted(after_counts.items())),
        warnings=warnings,
    )
    _write_rebalance_summary(summary, out_path)
    return summary


def filter_manifest(
    manifest_path: str | Path,
    output_manifest: str | Path,
    *,
    sample_ids: list[str] | None = None,
    slices: list[str] | None = None,
    document_types: list[str] | None = None,
    degradations: list[str] | None = None,
    limit: int | None = None,
) -> FilterSummary:
    entries = load_manifest(manifest_path)
    requested_sample_ids = [value for value in (sample_ids or []) if value]
    requested_slices = [value for value in (slices or []) if value]
    requested_document_types = [value for value in (document_types or []) if value]
    requested_degradations = [value for value in (degradations or []) if value]
    if not any([requested_sample_ids, requested_slices, requested_document_types, requested_degradations, limit is not None]):
        raise DataValidationError("filter-manifest requires at least one filter or --limit")

    selected = [
        entry
        for entry in entries
        if _matches_entry_filters(
            entry,
            sample_ids=requested_sample_ids,
            slices=requested_slices,
            document_types=requested_document_types,
            degradations=requested_degradations,
        )
    ]
    if requested_sample_ids:
        selected_by_id = {entry.sample_id for entry in selected}
        missing = [sample_id for sample_id in requested_sample_ids if sample_id not in selected_by_id]
        if missing:
            raise DataValidationError(f"requested sample_ids not found after filtering: {', '.join(missing)}")
    if limit is not None:
        if limit <= 0:
            raise DataValidationError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise DataValidationError("filter-manifest produced zero samples")

    out_path = Path(output_manifest)
    write_manifest(selected, out_path)
    summary = FilterSummary(
        output_manifest=str(out_path),
        source_manifest=str(Path(manifest_path)),
        source_count=len(entries),
        selected_count=len(selected),
        sample_id_filters=requested_sample_ids,
        slice_filters=requested_slices,
        document_type_filters=requested_document_types,
        degradation_filters=requested_degradations,
    )
    _write_filter_summary(summary, out_path)
    return summary


def entry_slices(entry: ManifestEntry) -> list[str]:
    metadata = entry.metadata or {}
    values: set[str] = set()
    for key in ("slice", "script", "document_type", "degradation", "language", "input_format"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    for key in ("slices", "scripts", "tags"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.update(str(item) for item in value if str(item))
        elif isinstance(value, str) and value:
            values.add(value)
    return sorted(values)


def _recognizer_bucket_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        for bucket_name in ("latin_only", "devanagari_only", "mixed_devanagari_latin", "real", "synthetic"):
            if _entry_in_recognizer_bucket(entry, bucket_name):
                counts[bucket_name] += 1
    return dict(sorted(counts.items()))


def _entry_in_recognizer_bucket(entry: ManifestEntry, bucket_name: str) -> bool:
    counts = script_counts(entry.text)
    has_devanagari = counts.get("devanagari", 0) > 0
    has_latin = counts.get("latin", 0) > 0
    if bucket_name == "latin_only":
        return has_latin and not has_devanagari
    if bucket_name == "devanagari_only":
        return has_devanagari and not has_latin
    if bucket_name == "mixed_devanagari_latin":
        return has_latin and has_devanagari
    if bucket_name in {"real", "synthetic"}:
        return bucket_name in set(entry_slices(entry))
    raise DataValidationError(f"unknown recognizer corpus bucket: {bucket_name}")


def _duplicate_recognizer_corpus_entry(entry: ManifestEntry, *, bucket_name: str, duplicate_index: int) -> ManifestEntry:
    metadata = dict(entry.metadata or {})
    metadata["recognizer_corpus_source_sample_id"] = entry.sample_id
    metadata["recognizer_corpus_bucket"] = bucket_name
    metadata["recognizer_corpus_duplicate"] = True
    return ManifestEntry(
        sample_id=f"{entry.sample_id}~recognizer-{bucket_name}-{duplicate_index:06d}",
        dataset=entry.dataset,
        split=entry.split,
        image_path=entry.image_path,
        text=entry.text,
        sha256=entry.sha256,
        metadata=metadata,
    )


def _recognizer_corpus_profile(path: Path, entries: list[ManifestEntry]) -> RecognizerCorpusProfile:
    dataset_counts: Counter[str] = Counter()
    slice_counts: Counter[str] = Counter()
    aggregate_scripts: Counter[str] = Counter()
    script_sample_counts: Counter[str] = Counter()
    for entry in entries:
        dataset_counts[entry.dataset] += 1
        for slice_name in entry_slices(entry):
            slice_counts[slice_name] += 1
        counts = script_counts(entry.text)
        aggregate_scripts.update(counts)
        has_devanagari = counts.get("devanagari", 0) > 0
        has_latin = counts.get("latin", 0) > 0
        if has_latin:
            script_sample_counts["contains_latin"] += 1
        if has_devanagari:
            script_sample_counts["contains_devanagari"] += 1
        if has_latin and has_devanagari:
            script_sample_counts["mixed_devanagari_latin"] += 1
        elif has_latin:
            script_sample_counts["latin_only"] += 1
        elif has_devanagari:
            script_sample_counts["devanagari_only"] += 1
        else:
            script_sample_counts["other_only"] += 1
    return RecognizerCorpusProfile(
        manifest_path=str(path),
        sample_count=len(entries),
        dataset_counts=dict(sorted(dataset_counts.items())),
        slice_counts=dict(sorted(slice_counts.items())),
        script_counts=dict(sorted(aggregate_scripts.items())),
        script_sample_counts=dict(sorted(script_sample_counts.items())),
    )


def _append_minimum_issue(issues: list[str], label: str, actual: int, required: int) -> None:
    if actual < required:
        issues.append(f"{label} {actual} < required {required}")


def _group_entries(entries: list[ManifestEntry], *, group_by: str = "sample_signature") -> dict[str, list[ManifestEntry]]:
    if group_by not in {"sample_signature", "text_sha256", "image_sha256", "image_path", "sample_id"}:
        raise DataValidationError("group_by must be one of: sample_signature, text_sha256, image_sha256, image_path, sample_id")
    groups: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        groups[_entry_group_key(entry, group_by)].append(entry)
    return dict(groups)


def _entry_group_key(entry: ManifestEntry, group_by: str) -> str:
    metadata = entry.metadata or {}
    if group_by == "sample_signature":
        return _entry_sample_signature(entry)
    if group_by == "text_sha256":
        value = metadata.get("text_sha256")
        return str(value) if value else sha256_text(entry.text)
    if group_by == "image_sha256":
        return str(entry.sha256 or entry.image_path)
    if group_by == "image_path":
        return entry.image_path
    if group_by == "sample_id":
        return entry.sample_id
    raise DataValidationError("group_by must be one of: sample_signature, text_sha256, image_sha256, image_path, sample_id")


def _entry_sample_signature(entry: ManifestEntry) -> str:
    metadata = entry.metadata or {}
    sample_hash = metadata.get("sample_sha256")
    if isinstance(sample_hash, str) and sample_hash:
        return sample_hash
    image_hash = entry.sha256 or entry.image_path
    text_hash = metadata.get("text_sha256") if isinstance(metadata.get("text_sha256"), str) else sha256_text(entry.text)
    return sha256_text(f"{image_hash}\n{text_hash}")


def _stratum_key(entries: list[ManifestEntry], stratify_by: str) -> str:
    if stratify_by == "none":
        return "all"
    if stratify_by == "dataset":
        return entries[0].dataset
    if stratify_by == "document_type":
        return str((entries[0].metadata or {}).get("document_type") or "unknown")
    if stratify_by != "slices":
        raise DataValidationError("stratify_by must be one of: slices, dataset, document_type, none")
    slices = sorted({slice_name for entry in entries for slice_name in entry_slices(entry)})
    return ",".join(slices) if slices else "unsliced"


def _with_split(entry: ManifestEntry, split: str) -> ManifestEntry:
    metadata = dict(entry.metadata)
    metadata["original_split"] = entry.split
    return ManifestEntry(
        sample_id=entry.sample_id,
        dataset=entry.dataset,
        split=split,
        image_path=entry.image_path,
        text=entry.text,
        sha256=entry.sha256,
        metadata=metadata,
    )


def _duplicate_entry(entry: ManifestEntry, *, slice_name: str, duplicate_index: int) -> ManifestEntry:
    metadata = dict(entry.metadata or {})
    metadata["rebalance_source_sample_id"] = entry.sample_id
    metadata["rebalance_slice"] = slice_name
    metadata["rebalance_duplicate"] = True
    return ManifestEntry(
        sample_id=f"{entry.sample_id}~rebalance-{slice_name}-{duplicate_index:06d}",
        dataset=entry.dataset,
        split=entry.split,
        image_path=entry.image_path,
        text=entry.text,
        sha256=entry.sha256,
        metadata=metadata,
    )


def _matches_entry_filters(
    entry: ManifestEntry,
    *,
    sample_ids: list[str],
    slices: list[str],
    document_types: list[str],
    degradations: list[str],
) -> bool:
    metadata = entry.metadata or {}
    if sample_ids and entry.sample_id not in set(sample_ids):
        return False
    entry_slice_set = set(entry_slices(entry))
    if slices and not set(slices).issubset(entry_slice_set):
        return False
    document_type = str(metadata.get("document_type") or "")
    if document_types and document_type not in set(document_types):
        return False
    degradation = str(metadata.get("degradation") or "")
    if degradations and degradation not in set(degradations):
        return False
    return True


def _write_audit(audit: ManifestAudit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest-audit.json").write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Manifest Audit",
        "",
        f"Passed: `{'yes' if audit.passed else 'no'}`",
        f"Samples: `{audit.sample_count}`",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.issues) if audit.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in audit.warnings) if audit.warnings else lines.append("- none")
    (output_dir / "manifest-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recognizer_corpus_audit(audit: RecognizerCorpusAudit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recognizer-corpus-audit.json"
    md_path = output_dir / "recognizer-corpus-audit.md"
    audit.summary_json_path = str(json_path)
    audit.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Recognizer Corpus Audit",
        "",
        f"Passed: `{'yes' if audit.passed else 'no'}`",
        f"Train manifest: `{audit.train_manifest}`",
        f"Eval manifest: `{audit.eval_manifest}`",
        f"Leakage check: `{'pass' if audit.leakage_passed else 'fail'}`",
        f"Require real eval: `{'yes' if audit.require_eval_real else 'no'}`",
        "",
        "## Profiles",
        "",
        "| split | samples | latin | devanagari | mixed | real | synthetic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _recognizer_profile_row("train", audit.train_profile),
        _recognizer_profile_row("eval", audit.eval_profile),
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.issues) if audit.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in audit.warnings) if audit.warnings else lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
        ]
    )
    lines.extend(_recognizer_corpus_next_actions(audit))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recognizer_corpus_build_summary(summary: RecognizerCorpusBuildSummary, output_manifest: Path) -> None:
    summary_json = output_manifest.with_name(f"{output_manifest.stem}-recognizer-corpus-build.json")
    summary_md = output_manifest.with_name(f"{output_manifest.stem}-recognizer-corpus-build.md")
    summary_json.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Recognizer Corpus Build",
        "",
        f"Output manifest: `{summary.output_manifest}`",
        f"Source samples: `{summary.source_count}`",
        f"Output samples: `{summary.output_count}`",
        "",
        "## Targets",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.target_counts.items()) if summary.target_counts else lines.append("- none")
    lines.extend(["", "## Bucket Counts Before", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.bucket_counts_before.items()) if summary.bucket_counts_before else lines.append("- none")
    lines.extend(["", "## Bucket Counts After", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.bucket_counts_after.items()) if summary.bucket_counts_after else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in summary.warnings) if summary.warnings else lines.append("- none")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recognizer_profile_row(split: str, profile: RecognizerCorpusProfile) -> str:
    scripts = profile.script_sample_counts
    slices = profile.slice_counts
    return (
        f"| {split} | {profile.sample_count} | {scripts.get('contains_latin', 0)} | "
        f"{scripts.get('contains_devanagari', 0)} | {scripts.get('mixed_devanagari_latin', 0)} | "
        f"{slices.get('real', 0)} | {slices.get('synthetic', 0)} |"
    )


def _recognizer_corpus_next_actions(audit: RecognizerCorpusAudit) -> list[str]:
    if audit.passed:
        return ["- corpus passes the configured recognizer training gate"]
    actions: list[str] = []
    issue_text = "\n".join(audit.issues)
    if "latin-script" in issue_text:
        actions.append("- add or upsample English/Latin line crops before training")
    if "devanagari-script" in issue_text:
        actions.append("- add or upsample Nepali/Devanagari line crops before training")
    if "mixed-script" in issue_text:
        actions.append("- add mixed Nepali/English samples so the recognizer sees script transitions")
    if "latin-only" in issue_text:
        actions.append("- add Latin-only English samples to protect English preservation")
    if "devanagari-only" in issue_text:
        actions.append("- add Devanagari-only Nepali samples to prevent mixed-script-only overfitting")
    if "real slice" in issue_text:
        actions.append("- add real scanned/captured Nepali data; synthetic-only training is not enough")
    if "synthetic slice" in issue_text:
        actions.append("- add synthetic coverage for controllable character and layout diversity")
    if "claim-grade evaluation" in issue_text or "real held-out documents" in issue_text:
        actions.append("- replace eval with held-out real document crops/pages; keep synthetic rows in train or ablation splits")
    if "leakage" in issue_text:
        actions.append("- regenerate train/eval split with leakage-safe grouping before training")
    if not actions:
        actions.append("- inspect failed thresholds and rebuild the training manifest")
    return actions


def _write_split_summary(summary: SplitSummary, output_dir: Path) -> None:
    (output_dir / "split-summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Manifest Split",
        "",
        f"Train manifest: `{summary.train_manifest}`",
        f"Eval manifest: `{summary.eval_manifest}`",
        f"Train samples: `{summary.train_count}`",
        f"Eval samples: `{summary.eval_count}`",
        f"Sample groups: `{summary.group_count}`",
        f"Group by: `{summary.group_by}`",
        f"Leakage check: `{'pass' if summary.leakage_passed else 'fail'}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.warnings) if summary.warnings else lines.append("- none")
    (output_dir / "split-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_merge_summary(summary: MergeSummary, output_manifest: Path) -> None:
    summary_json = output_manifest.with_name(f"{output_manifest.stem}-merge.json")
    summary_md = output_manifest.with_name(f"{output_manifest.stem}-merge.md")
    summary_json.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Manifest Merge",
        "",
        f"Output manifest: `{summary.output_manifest}`",
        f"Samples: `{summary.sample_count}`",
        "",
        "## Input Manifests",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.input_manifests)
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in summary.warnings) if summary.warnings else lines.append("- none")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rebalance_summary(summary: RebalanceSummary, output_manifest: Path) -> None:
    summary_json = output_manifest.with_name(f"{output_manifest.stem}-rebalance.json")
    summary_md = output_manifest.with_name(f"{output_manifest.stem}-rebalance.md")
    summary_json.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Manifest Rebalance",
        "",
        f"Source manifest: `{summary.source_manifest}`",
        f"Output manifest: `{summary.output_manifest}`",
        f"Original samples: `{summary.original_count}`",
        f"Rebalanced samples: `{summary.rebalanced_count}`",
        f"Target count: `{summary.target_count}`",
        "",
        "## Slice Counts Before",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.slice_counts_before.items())
    lines.extend(["", "## Slice Counts After", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.slice_counts_after.items())
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in summary.warnings) if summary.warnings else lines.append("- none")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_filter_summary(summary: FilterSummary, output_manifest: Path) -> None:
    summary_json = output_manifest.with_name(f"{output_manifest.stem}-filter.json")
    summary_md = output_manifest.with_name(f"{output_manifest.stem}-filter.md")
    summary_json.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Manifest Filter",
        "",
        f"Source manifest: `{summary.source_manifest}`",
        f"Output manifest: `{summary.output_manifest}`",
        f"Source samples: `{summary.source_count}`",
        f"Selected samples: `{summary.selected_count}`",
        f"Sample IDs: `{', '.join(summary.sample_id_filters)}`",
        f"Slices: `{', '.join(summary.slice_filters)}`",
        f"Document types: `{', '.join(summary.document_type_filters)}`",
        f"Degradations: `{', '.join(summary.degradation_filters)}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.warnings) if summary.warnings else lines.append("- none")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
