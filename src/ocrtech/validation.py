"""Validation gates for defensible OCR quality claims."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .manifest import ManifestEntry, load_manifest, sha256_file, sha256_text


LOWER_IS_BETTER = {
    "cer",
    "wer",
    "text_cer",
    "text_wer",
    "document_reconstruction_cer",
    "document_reconstruction_wer",
    "figure_count_error",
    "figure_caption_cer",
}
HIGHER_IS_BETTER = {
    "quality_score",
    "document_reconstruction_similarity",
    "figure_detection_precision",
    "figure_detection_recall",
    "figure_detection_f1",
    "reading_order_pair_accuracy",
    "table_cell_precision",
    "table_cell_recall",
    "table_cell_f1",
    "table_exact_match",
    "table_structure_similarity",
    "table_teds_structure",
    "table_teds_text",
}


@dataclass(slots=True)
class ClaimGate:
    baseline: str
    metric: str
    direction: str
    slice: str | None = None
    slices: list[str] = field(default_factory=list)
    min_absolute_margin: float = 0.0
    min_relative_improvement: float = 0.0
    min_pairs: int = 1
    min_pair_win_rate: float | None = None
    max_p_value: float | None = None
    require_ci_above_zero: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimGate":
        metric = str(data.get("metric") or "")
        direction = str(data.get("direction") or metric_direction(metric))
        if direction not in {"lower", "higher"}:
            raise ValidationError(f"gate direction must be lower or higher: {direction}")
        baseline = str(data.get("baseline") or "")
        if not baseline:
            raise ValidationError("gate baseline is required")
        if not metric:
            raise ValidationError("gate metric is required")
        slices_payload = data.get("slices")
        if slices_payload is None:
            slices = [str(data["slice"])] if data.get("slice") else []
        elif isinstance(slices_payload, list):
            slices = [str(item) for item in slices_payload if str(item)]
        else:
            raise ValidationError("gate slices must be a list when provided")
        min_pair_win_rate = _optional_rate(data.get("min_pair_win_rate"), "min_pair_win_rate")
        return cls(
            baseline=baseline,
            metric=metric,
            direction=direction,
            slice=data.get("slice"),
            slices=slices,
            min_absolute_margin=float(data.get("min_absolute_margin", 0.0)),
            min_relative_improvement=float(data.get("min_relative_improvement", 0.0)),
            min_pairs=int(data.get("min_pairs", 1)),
            min_pair_win_rate=min_pair_win_rate,
            max_p_value=float(data["max_p_value"]) if data.get("max_p_value") is not None else None,
            require_ci_above_zero=bool(data.get("require_ci_above_zero", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "metric": self.metric,
            "direction": self.direction,
            "slice": self.slice,
            "slices": self.slices,
            "min_absolute_margin": self.min_absolute_margin,
            "min_relative_improvement": self.min_relative_improvement,
            "min_pairs": self.min_pairs,
            "min_pair_win_rate": self.min_pair_win_rate,
            "max_p_value": self.max_p_value,
            "require_ci_above_zero": self.require_ci_above_zero,
        }


@dataclass(slots=True)
class ValidationConfig:
    candidate: str = "ours"
    min_samples: int = 30
    required_slices: dict[str, int] = field(default_factory=dict)
    gates: list[ClaimGate] = field(default_factory=list)
    require_full_eval_manifest_coverage: bool = False
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95
    random_seed: int = 13

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ValidationConfig":
        if path is None:
            return default_validation_config()
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid validation config JSON {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("validation config must be a JSON object")
        return cls(
            candidate=str(payload.get("candidate", "ours")),
            min_samples=int(payload.get("min_samples", 30)),
            required_slices={str(key): int(value) for key, value in dict(payload.get("required_slices", {})).items()},
            gates=[ClaimGate.from_dict(item) for item in payload.get("gates", [])],
            require_full_eval_manifest_coverage=bool(payload.get("require_full_eval_manifest_coverage", False)),
            bootstrap_iterations=int(payload.get("bootstrap_iterations", 1000)),
            bootstrap_confidence=float(payload.get("bootstrap_confidence", 0.95)),
            random_seed=int(payload.get("random_seed", 13)),
        )


@dataclass(slots=True)
class LeakageReport:
    train_manifests: list[str]
    eval_manifest: str | None
    overlaps: dict[str, list[str]]
    warnings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(self.overlaps.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "train_manifests": self.train_manifests,
            "eval_manifest": self.eval_manifest,
            "overlaps": self.overlaps,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GateResult:
    gate: ClaimGate
    passed: bool
    pairs: int
    candidate_mean: float | None
    baseline_mean: float | None
    absolute_improvement: float | None
    relative_improvement: float | None
    improvement_ci_low: float | None
    improvement_ci_high: float | None
    win_count: int
    loss_count: int
    tie_count: int
    win_rate: float | None
    pair_win_rate: float | None
    sign_test_p_value: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.to_dict(),
            "passed": self.passed,
            "pairs": self.pairs,
            "candidate_mean": self.candidate_mean,
            "baseline_mean": self.baseline_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "improvement_ci_low": self.improvement_ci_low,
            "improvement_ci_high": self.improvement_ci_high,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "win_rate": self.win_rate,
            "pair_win_rate": self.pair_win_rate,
            "sign_test_p_value": self.sign_test_p_value,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ValidationReport:
    claim_status: str
    passed: bool
    sample_count: int
    slice_counts: dict[str, int]
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]]
    missing_requirements: list[str]
    leakage: LeakageReport
    gates: list[GateResult]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_status": self.claim_status,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "slice_counts": self.slice_counts,
            "aggregate_metrics": self.aggregate_metrics,
            "missing_requirements": self.missing_requirements,
            "leakage": self.leakage.to_dict(),
            "gates": [gate.to_dict() for gate in self.gates],
            "provenance": self.provenance,
        }


def validate_claim(
    benchmark_report: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    train_manifests: list[str | Path] | None = None,
    candidate_model_config: str | Path | None = None,
    fallback_model_config: str | Path | None = None,
) -> ValidationReport:
    config = ValidationConfig.from_path(config_path)
    results = _load_benchmark_results(benchmark_report)
    eval_entries = load_manifest(eval_manifest) if eval_manifest else []
    enriched = [_enrich_result(item, eval_entries) for item in results]
    candidate_rows = [item for item in enriched if item["baseline"] == config.candidate and item["status"] == "ok"]
    sample_count = len({sample_key(item) for item in candidate_rows})
    slice_counts = _slice_counts(candidate_rows)
    leakage = check_leakage(train_manifests or [], eval_manifest)

    missing: list[str] = []
    if sample_count < config.min_samples:
        missing.append(f"candidate sample count {sample_count} < required {config.min_samples}")
    for slice_name, required_count in config.required_slices.items():
        actual = slice_counts.get(slice_name, 0)
        if actual < required_count:
            missing.append(f"slice {slice_name!r} count {actual} < required {required_count}")
    missing.extend(_full_eval_manifest_coverage_failures(enriched, config, eval_entries))
    if not leakage.passed:
        missing.append("train/eval leakage detected")
    gate_results = [_evaluate_gate(enriched, config, gate) for gate in config.gates]
    for gate_result in gate_results:
        if not gate_result.passed:
            missing.append(gate_result.reason or f"gate failed: {gate_result.gate.metric} vs {gate_result.gate.baseline}")

    passed = not missing and bool(config.gates)
    report = ValidationReport(
        claim_status="validated" if passed else "not_validated",
        passed=passed,
        sample_count=sample_count,
        slice_counts=slice_counts,
        aggregate_metrics=_aggregate_metrics(enriched),
        missing_requirements=missing,
        leakage=leakage,
        gates=gate_results,
        provenance=_validation_provenance(
            benchmark_report,
            config_path,
            eval_manifest,
            train_manifests or [],
            candidate_model_config,
            fallback_model_config,
        ),
    )
    _write_validation_report(report, Path(output_dir))
    return report


def check_leakage(train_manifests: list[str | Path], eval_manifest: str | Path | None) -> LeakageReport:
    if eval_manifest is None or not train_manifests:
        return LeakageReport([str(item) for item in train_manifests], str(eval_manifest) if eval_manifest else None, {})
    eval_entries = load_manifest(eval_manifest)
    train_entries: list[ManifestEntry] = []
    for manifest in train_manifests:
        train_entries.extend(load_manifest(manifest))
    overlaps = {
        "sample_id": sorted({entry.sample_id for entry in train_entries} & {entry.sample_id for entry in eval_entries}),
        "image_sha256": _hash_overlap(train_entries, eval_entries, "sha256"),
        "sample_sha256": _sample_hash_overlap(train_entries, eval_entries),
        "text_sha256": _text_hash_overlap(train_entries, eval_entries),
        "image_path": sorted({entry.image_path for entry in train_entries} & {entry.image_path for entry in eval_entries}),
    }
    warnings = {"text_sha256": overlaps.pop("text_sha256")}
    return LeakageReport([str(item) for item in train_manifests], str(eval_manifest), overlaps, warnings)


def _validation_provenance(
    benchmark_report: str | Path,
    config_path: str | Path | None,
    eval_manifest: str | Path | None,
    train_manifests: list[str | Path],
    candidate_model_config: str | Path | None,
    fallback_model_config: str | Path | None,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_report)
    payload: dict[str, Any] = {
        "benchmark_report": {
            "path": str(benchmark_path),
            "sha256": _sha256_if_file(benchmark_path),
        },
        "validation_config": None,
        "eval_manifest": None,
        "candidate_model_config": None,
        "fallback_model_config": None,
        "train_manifests": [],
    }
    if config_path is not None:
        config = Path(config_path)
        payload["validation_config"] = {"path": str(config), "sha256": _sha256_if_file(config)}
    if eval_manifest is not None:
        eval_path = Path(eval_manifest)
        payload["eval_manifest"] = {"path": str(eval_path), "sha256": _sha256_if_file(eval_path)}
    if candidate_model_config is not None:
        candidate_path = Path(candidate_model_config)
        payload["candidate_model_config"] = {"path": str(candidate_path), "sha256": _sha256_if_file(candidate_path)}
    if fallback_model_config is not None:
        fallback_path = Path(fallback_model_config)
        payload["fallback_model_config"] = {"path": str(fallback_path), "sha256": _sha256_if_file(fallback_path)}
    for manifest in train_manifests:
        path = Path(manifest)
        payload["train_manifests"].append({"path": str(path), "sha256": _sha256_if_file(path)})
    return payload


def _sha256_if_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def _optional_rate(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    rate = float(value)
    if rate < 0 or rate > 1:
        raise ValidationError(f"{field_name} must be between 0 and 1")
    return rate


def default_validation_config() -> ValidationConfig:
    return ValidationConfig(
        candidate="ours",
        min_samples=30,
        required_slices={"nepali": 10, "english": 10, "table": 5, "reading_order": 5, "figure": 5},
        require_full_eval_manifest_coverage=False,
        gates=[
            ClaimGate(baseline="tesseract", metric="cer", direction="lower", slice="nepali", min_relative_improvement=0.05, min_pair_win_rate=0.55),
            ClaimGate(baseline="tesseract", metric="wer", direction="lower", slice="nepali", min_relative_improvement=0.02, min_pair_win_rate=0.55),
            ClaimGate(baseline="stock-paddle", metric="cer", direction="lower", slice="nepali", min_relative_improvement=0.02, min_pair_win_rate=0.55),
            ClaimGate(baseline="stock-paddle", metric="cer", direction="lower", slice="english", min_relative_improvement=0.0, require_ci_above_zero=False),
            ClaimGate(baseline="stock-paddle", metric="wer", direction="lower", slice="english", min_relative_improvement=0.0, require_ci_above_zero=False),
            ClaimGate(baseline="glm-ocr", metric="cer", direction="lower", slice="nepali", min_relative_improvement=0.02, min_pair_win_rate=0.55),
            ClaimGate(baseline="surya", metric="table_cell_f1", direction="higher", slice="table", min_absolute_margin=0.01, min_pair_win_rate=0.55),
            ClaimGate(baseline="surya", metric="table_teds_text", direction="higher", slice="table", min_absolute_margin=0.01, min_pair_win_rate=0.55),
            ClaimGate(baseline="paddleocr-vl", metric="reading_order_pair_accuracy", direction="higher", slice="reading_order", min_absolute_margin=0.01, min_pair_win_rate=0.55),
            ClaimGate(baseline="paddleocr-vl", metric="document_reconstruction_similarity", direction="higher", slice="figure", min_absolute_margin=0.01, min_pair_win_rate=0.55),
            ClaimGate(baseline="surya", metric="figure_detection_f1", direction="higher", slice="figure", min_absolute_margin=0.0, require_ci_above_zero=False),
            ClaimGate(baseline="surya", metric="figure_caption_cer", direction="lower", slice="figure", min_absolute_margin=0.01, min_pair_win_rate=0.55),
        ],
    )


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    raise ValidationError(f"Unknown metric direction for {metric!r}; specify direction explicitly")


def sample_key(result: dict[str, Any]) -> str:
    return str(result.get("sample_id") or result.get("input_path") or "")


def _load_benchmark_results(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValidationError("benchmark report must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValidationError(f"benchmark row {index} must be an object")
        rows.append(dict(item))
    return rows


def _enrich_result(row: dict[str, Any], eval_entries: list[ManifestEntry]) -> dict[str, Any]:
    enriched = dict(row)
    entry = _entry_for_input(str(row.get("input_path", "")), eval_entries)
    if entry:
        enriched["sample_id"] = row.get("sample_id") or entry.sample_id
        existing_slices = row.get("slices") if isinstance(row.get("slices"), list) else []
        enriched["slices"] = sorted({*existing_slices, *_slices_for_entry(entry)})
    else:
        enriched.setdefault("slices", [])
    return enriched


def _entry_for_input(input_path: str, entries: list[ManifestEntry]) -> ManifestEntry | None:
    if not entries:
        return None
    path = Path(input_path)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for entry in entries:
        entry_path = Path(entry.image_path)
        try:
            resolved_entry = entry_path.resolve()
        except OSError:
            resolved_entry = entry_path
        if resolved == resolved_entry or input_path == entry.image_path:
            return entry
    return None


def _slices_for_entry(entry: ManifestEntry) -> list[str]:
    values: set[str] = set()
    metadata = entry.metadata or {}
    for key in ("slice", "script", "document_type", "degradation", "language"):
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


def _slice_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = sample_key(row)
        for slice_name in row.get("slices", []) or []:
            pair = (key, str(slice_name))
            if pair in seen:
                continue
            seen.add(pair)
            counts[str(slice_name)] = counts.get(str(slice_name), 0) + 1
    return counts


def _full_eval_manifest_coverage_failures(
    rows: list[dict[str, Any]],
    config: ValidationConfig,
    eval_entries: list[ManifestEntry],
) -> list[str]:
    if not config.require_full_eval_manifest_coverage:
        return []
    if not eval_entries:
        return ["eval manifest is required when require_full_eval_manifest_coverage is true"]
    eval_sample_ids = {entry.sample_id for entry in eval_entries}
    required_baselines = {config.candidate, *(gate.baseline for gate in config.gates)}
    failures: list[str] = []
    for baseline in sorted(required_baselines):
        ok_sample_ids = {
            str(row.get("sample_id"))
            for row in rows
            if row.get("baseline") == baseline and row.get("status") == "ok" and str(row.get("sample_id") or "")
        }
        missing = sorted(eval_sample_ids - ok_sample_ids)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
            failures.append(
                f"baseline {baseline!r} is missing successful rows for {len(missing)} eval manifest samples: {preview}{suffix}"
            )
    return failures


def _evaluate_gate(rows: list[dict[str, Any]], config: ValidationConfig, gate: ClaimGate) -> GateResult:
    gate_slices = gate.slices or ([gate.slice] if gate.slice else [])
    candidate = _rows_by_sample(rows, config.candidate, gate.metric, gate_slices)
    baseline = _rows_by_sample(rows, gate.baseline, gate.metric, gate_slices)
    shared = sorted(set(candidate) & set(baseline))
    if not shared:
        return GateResult(gate, False, 0, None, None, None, None, None, None, 0, 0, 0, None, None, None, f"no paired samples for {config.candidate} vs {gate.baseline} on {gate.metric}")
    candidate_values = [candidate[key] for key in shared]
    baseline_values = [baseline[key] for key in shared]
    improvements = [
        _absolute_improvement(candidate_value, baseline_value, gate.direction)
        for candidate_value, baseline_value in zip(candidate_values, baseline_values, strict=True)
    ]
    candidate_mean = sum(candidate_values) / len(candidate_values)
    baseline_mean = sum(baseline_values) / len(baseline_values)
    absolute = _absolute_improvement(candidate_mean, baseline_mean, gate.direction)
    relative = _relative_improvement(absolute, baseline_mean)
    ci_low, ci_high = _bootstrap_ci(improvements, config.bootstrap_iterations, config.bootstrap_confidence, config.random_seed)
    wins, losses, ties = _paired_outcome_counts(candidate_values, baseline_values, gate.direction)
    non_ties = wins + losses
    win_rate = (wins / non_ties) if non_ties else None
    pair_win_rate = wins / len(shared)
    sign_test_p_value = _sign_test_p_value(wins, losses)
    reasons: list[str] = []
    if len(shared) < gate.min_pairs:
        reasons.append(f"paired samples {len(shared)} < required {gate.min_pairs}")
    if absolute < gate.min_absolute_margin:
        reasons.append(f"absolute improvement {absolute:.6f} < required {gate.min_absolute_margin:.6f}")
    if relative < gate.min_relative_improvement:
        reasons.append(f"relative improvement {relative:.6f} < required {gate.min_relative_improvement:.6f}")
    if gate.min_pair_win_rate is not None and pair_win_rate < gate.min_pair_win_rate:
        reasons.append(f"paired win rate {pair_win_rate:.6f} < required {gate.min_pair_win_rate:.6f}")
    if gate.max_p_value is not None:
        if sign_test_p_value is None:
            reasons.append("sign test p-value is unavailable because all paired outcomes are ties")
        elif sign_test_p_value > gate.max_p_value:
            reasons.append(f"sign test p-value {sign_test_p_value:.6f} > required {gate.max_p_value:.6f}")
    if gate.require_ci_above_zero and ci_low <= 0:
        reasons.append(f"bootstrap CI low {ci_low:.6f} <= 0")
    return GateResult(
        gate=gate,
        passed=not reasons,
        pairs=len(shared),
        candidate_mean=candidate_mean,
        baseline_mean=baseline_mean,
        absolute_improvement=absolute,
        relative_improvement=relative,
        improvement_ci_low=ci_low,
        improvement_ci_high=ci_high,
        win_count=wins,
        loss_count=losses,
        tie_count=ties,
        win_rate=win_rate,
        pair_win_rate=pair_win_rate,
        sign_test_p_value=sign_test_p_value,
        reason="; ".join(reasons) if reasons else None,
    )


def gate_has_quantitative_evidence(item: dict[str, Any]) -> bool:
    gate = item.get("gate")
    gate_payload = gate if isinstance(gate, dict) else {}
    pairs = item.get("pairs")
    if isinstance(pairs, bool) or not isinstance(pairs, int | float) or int(pairs) <= 0:
        return False
    pair_count = int(pairs)
    for key in ("candidate_mean", "baseline_mean", "absolute_improvement"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            return False
    counts: list[int] = []
    for key in ("win_count", "loss_count", "tie_count"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        count = int(value)
        if count < 0:
            return False
        counts.append(count)
    if sum(counts) != pair_count:
        return False
    min_pairs = gate_payload.get("min_pairs")
    if min_pairs is not None:
        if isinstance(min_pairs, bool) or not isinstance(min_pairs, int | float):
            return False
        if pair_count < int(min_pairs):
            return False
    min_absolute_margin = gate_payload.get("min_absolute_margin")
    if min_absolute_margin is not None and not _numeric_threshold_passes(item.get("absolute_improvement"), min_absolute_margin):
        return False
    min_relative_improvement = gate_payload.get("min_relative_improvement")
    if min_relative_improvement is not None and not _numeric_threshold_passes(item.get("relative_improvement"), min_relative_improvement):
        return False
    min_pair_win_rate = gate_payload.get("min_pair_win_rate")
    if min_pair_win_rate is not None and not _numeric_threshold_passes(item.get("pair_win_rate"), min_pair_win_rate):
        return False
    max_p_value = gate_payload.get("max_p_value")
    if max_p_value is not None and not _numeric_max_threshold_passes(item.get("sign_test_p_value"), max_p_value):
        return False
    if bool(gate_payload.get("require_ci_above_zero", True)):
        ci_low = item.get("improvement_ci_low")
        if isinstance(ci_low, bool) or not isinstance(ci_low, int | float) or not math.isfinite(float(ci_low)) or float(ci_low) <= 0:
            return False
    return True


def _numeric_threshold_passes(value: Any, threshold: Any) -> bool:
    if isinstance(value, bool) or isinstance(threshold, bool):
        return False
    if not isinstance(value, int | float) or not isinstance(threshold, int | float):
        return False
    if not math.isfinite(float(value)) or not math.isfinite(float(threshold)):
        return False
    return float(value) >= float(threshold)


def _numeric_max_threshold_passes(value: Any, threshold: Any) -> bool:
    if isinstance(value, bool) or isinstance(threshold, bool):
        return False
    if not isinstance(value, int | float) or not isinstance(threshold, int | float):
        return False
    if not math.isfinite(float(value)) or not math.isfinite(float(threshold)):
        return False
    return float(value) <= float(threshold)


def _rows_by_sample(rows: list[dict[str, Any]], baseline: str, metric: str, slice_names: list[str] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    required_slices = {str(item) for item in slice_names or [] if str(item)}
    for row in rows:
        if row.get("baseline") != baseline or row.get("status") != "ok":
            continue
        row_slices = {str(item) for item in row.get("slices") or []}
        if required_slices and not required_slices.issubset(row_slices):
            continue
        metrics = row.get("metrics") or {}
        if metric not in metrics:
            continue
        value = metrics[metric]
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values[sample_key(row)] = float(value)
    return values


def _absolute_improvement(candidate_value: float, baseline_value: float, direction: str) -> float:
    if direction == "lower":
        return baseline_value - candidate_value
    return candidate_value - baseline_value


def _relative_improvement(absolute_improvement: float, baseline_mean: float) -> float:
    denominator = abs(baseline_mean)
    if denominator == 0:
        return math.inf if absolute_improvement > 0 else 0.0
    return absolute_improvement / denominator


def _bootstrap_ci(values: list[float], iterations: int, confidence: float, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or iterations <= 0:
        mean = sum(values) / len(values)
        return mean, mean
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = 1 - confidence
    low_index = max(0, min(len(means) - 1, int((alpha / 2) * len(means))))
    high_index = max(0, min(len(means) - 1, int((1 - alpha / 2) * len(means)) - 1))
    return means[low_index], means[high_index]


def _paired_outcome_counts(candidate_values: list[float], baseline_values: list[float], direction: str) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for candidate_value, baseline_value in zip(candidate_values, baseline_values, strict=True):
        improvement = _absolute_improvement(candidate_value, baseline_value, direction)
        if improvement > 0:
            wins += 1
        elif improvement < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def _sign_test_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    numerator = sum(math.comb(trials, success_count) for success_count in range(wins, trials + 1))
    return numerator / (2**trials)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = str(row.get("baseline") or "")
        if not baseline:
            continue
        slices = [*({str(item) for item in row.get("slices") or []} or {"all"})]
        slices.append("all")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for slice_name in slices:
            slice_bucket = grouped.setdefault(baseline, {}).setdefault(slice_name, {})
            for metric_name, value in metrics.items():
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    slice_bucket.setdefault(str(metric_name), []).append(float(value))
    aggregates: dict[str, dict[str, dict[str, float | int]]] = {}
    for baseline, slice_map in grouped.items():
        aggregates[baseline] = {}
        for slice_name, metric_map in slice_map.items():
            aggregates[baseline][slice_name] = {
                metric_name: sum(values) / len(values)
                for metric_name, values in sorted(metric_map.items())
                if values
            }
            if metric_map:
                first_values = next(iter(metric_map.values()))
                aggregates[baseline][slice_name]["sample_count"] = len(first_values)
    return aggregates


def _hash_overlap(train_entries: list[ManifestEntry], eval_entries: list[ManifestEntry], attr: str) -> list[str]:
    train_values = {getattr(entry, attr) for entry in train_entries if getattr(entry, attr)}
    eval_values = {getattr(entry, attr) for entry in eval_entries if getattr(entry, attr)}
    return sorted(str(item) for item in train_values & eval_values)


def _text_hash_overlap(train_entries: list[ManifestEntry], eval_entries: list[ManifestEntry]) -> list[str]:
    def text_hash(entry: ManifestEntry) -> str:
        value = entry.metadata.get("text_sha256") if entry.metadata else None
        return str(value) if value else sha256_text(entry.text)

    return sorted({text_hash(entry) for entry in train_entries} & {text_hash(entry) for entry in eval_entries})


def _sample_hash_overlap(train_entries: list[ManifestEntry], eval_entries: list[ManifestEntry]) -> list[str]:
    def sample_hash(entry: ManifestEntry) -> str:
        value = entry.metadata.get("sample_sha256") if entry.metadata else None
        if value:
            return str(value)
        image_hash = entry.sha256 or ""
        return sha256_text(f"{image_hash}\n{sha256_text(entry.text)}")

    return sorted({sample_hash(entry) for entry in train_entries} & {sample_hash(entry) for entry in eval_entries})


def _write_validation_report(report: ValidationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR Claim Validation",
        "",
        f"Status: `{report.claim_status}`",
        f"Samples: `{report.sample_count}`",
        f"Leakage check: `{'pass' if report.leakage.passed else 'fail'}`",
        "",
        "## Missing Requirements",
        "",
    ]
    if report.missing_requirements:
        lines.extend(f"- {item}" for item in report.missing_requirements)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| baseline | slice | metric | pairs | wins | losses | ties | win_rate | pair_win_rate | p_value | candidate | baseline_mean | improvement | ci_low | pass | reason |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for gate in report.gates:
        lines.append(
            "| {baseline} | {slice} | {metric} | {pairs} | {wins} | {losses} | {ties} | {win_rate} | {pair_win_rate} | {p_value} | {candidate} | {baseline_mean} | {improvement} | {ci_low} | {passed} | {reason} |".format(
                baseline=gate.gate.baseline,
                slice=",".join(gate.gate.slices or ([gate.gate.slice] if gate.gate.slice else [])),
                metric=gate.gate.metric,
                pairs=gate.pairs,
                wins=gate.win_count,
                losses=gate.loss_count,
                ties=gate.tie_count,
                win_rate=f"{gate.win_rate:.4f}" if gate.win_rate is not None else "",
                pair_win_rate=f"{gate.pair_win_rate:.4f}" if gate.pair_win_rate is not None else "",
                p_value=f"{gate.sign_test_p_value:.6f}" if gate.sign_test_p_value is not None else "",
                candidate=f"{gate.candidate_mean:.6f}" if gate.candidate_mean is not None else "",
                baseline_mean=f"{gate.baseline_mean:.6f}" if gate.baseline_mean is not None else "",
                improvement=f"{gate.absolute_improvement:.6f}" if gate.absolute_improvement is not None else "",
                ci_low=f"{gate.improvement_ci_low:.6f}" if gate.improvement_ci_low is not None else "",
                passed="yes" if gate.passed else "no",
                reason=(gate.reason or "").replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Aggregate Metrics", ""])
    for baseline, slice_map in sorted(report.aggregate_metrics.items()):
        lines.append(f"### {baseline}")
        lines.append("")
        lines.append("| slice | sample_count | metric | mean |")
        lines.append("| --- | ---: | --- | ---: |")
        for slice_name, metrics in sorted(slice_map.items()):
            sample_count = metrics.get("sample_count", "")
            for metric_name, value in sorted(metrics.items()):
                if metric_name == "sample_count":
                    continue
                lines.append(f"| {slice_name} | {sample_count} | {metric_name} | {float(value):.6f} |")
        lines.append("")
    (output_dir / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
