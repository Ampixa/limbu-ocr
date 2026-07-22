"""Portfolio-level claim review across multiple benchmark experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .errors import ClaimReviewError
from .manifest import ManifestEntry, load_manifest, sha256_file
from .models import ModelCard, audit_model_card, component_baseline_names, inspect_routed_calibration_collapse, model_admission_status, trained_recognizer_provenance_status
from .validation import gate_has_quantitative_evidence, sample_key


@dataclass(slots=True)
class RequiredValidationGate:
    baseline: str
    metric: str
    slices: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequiredValidationGate":
        baseline = str(data.get("baseline") or "")
        metric = str(data.get("metric") or "")
        if not baseline:
            raise ClaimReviewError("required_validation_gates baseline is required")
        if not metric:
            raise ClaimReviewError("required_validation_gates metric is required")
        return cls(baseline=baseline, metric=metric, slices=_string_list(data.get("slices")))

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline, "metric": self.metric, "slices": self.slices}


@dataclass(slots=True)
class RequiredExperiment:
    name: str
    datasets: list[str] = field(default_factory=list)
    eval_manifest_patterns: list[str] = field(default_factory=list)
    require_validation_status: str | None = None
    required_validation_config: str | None = None
    require_baselines: list[str] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    required_metric_slices: dict[str, list[str]] = field(default_factory=dict)
    required_output_artifacts: list[str] = field(default_factory=list)
    required_output_artifact_slices: dict[str, list[str]] = field(default_factory=dict)
    required_validation_gates: list[RequiredValidationGate] = field(default_factory=list)
    require_no_errors: bool = True
    required_slices: list[str] = field(default_factory=list)
    require_candidate_fallback_metrics: bool = False
    max_candidate_fallback_trigger_rate: float | None = None
    max_candidate_fallback_trigger_rate_by_slice: dict[str, float] = field(default_factory=dict)
    max_candidate_fallback_failed_rate: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequiredExperiment":
        name = str(data.get("name") or "")
        if not name:
            raise ClaimReviewError("required_experiment.name is required")
        datasets = _string_list(data.get("datasets"))
        eval_manifest_patterns = _string_list(data.get("eval_manifest_patterns"))
        if not datasets and not eval_manifest_patterns:
            raise ClaimReviewError(f"required experiment {name!r} must define datasets or eval_manifest_patterns")
        require_validation_status = data.get("require_validation_status")
        if require_validation_status is not None:
            require_validation_status = str(require_validation_status)
        required_validation_config = data.get("required_validation_config")
        if required_validation_config is not None:
            required_validation_config = str(required_validation_config)
        require_baselines = _string_list(data.get("require_baselines"))
        required_metrics = _string_list(data.get("required_metrics"))
        required_metric_slices = _metric_slice_requirements(data.get("required_metric_slices"))
        required_output_artifacts = _string_list(data.get("required_output_artifacts"))
        required_output_artifact_slices = _metric_slice_requirements(data.get("required_output_artifact_slices"))
        required_validation_gates = _required_validation_gates(data.get("required_validation_gates"))
        _validate_required_validation_gates(
            name,
            require_validation_status,
            require_baselines,
            required_metrics,
            required_validation_gates,
        )
        required_slices = _string_list(data.get("required_slices"))
        require_candidate_fallback_metrics = bool(data.get("require_candidate_fallback_metrics", False))
        max_candidate_fallback_trigger_rate = _optional_rate(data.get("max_candidate_fallback_trigger_rate"), "max_candidate_fallback_trigger_rate")
        max_candidate_fallback_trigger_rate_by_slice = _optional_rate_mapping(
            data.get("max_candidate_fallback_trigger_rate_by_slice"),
            "max_candidate_fallback_trigger_rate_by_slice",
        )
        max_candidate_fallback_failed_rate = _optional_rate(data.get("max_candidate_fallback_failed_rate"), "max_candidate_fallback_failed_rate")
        return cls(
            name=name,
            datasets=datasets,
            eval_manifest_patterns=eval_manifest_patterns,
            require_validation_status=require_validation_status,
            required_validation_config=required_validation_config,
            require_baselines=require_baselines,
            required_metrics=required_metrics,
            required_metric_slices=required_metric_slices,
            required_output_artifacts=required_output_artifacts,
            required_output_artifact_slices=required_output_artifact_slices,
            required_validation_gates=required_validation_gates,
            require_no_errors=bool(data.get("require_no_errors", True)),
            required_slices=required_slices,
            require_candidate_fallback_metrics=require_candidate_fallback_metrics,
            max_candidate_fallback_trigger_rate=max_candidate_fallback_trigger_rate,
            max_candidate_fallback_trigger_rate_by_slice=max_candidate_fallback_trigger_rate_by_slice,
            max_candidate_fallback_failed_rate=max_candidate_fallback_failed_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datasets": self.datasets,
            "eval_manifest_patterns": self.eval_manifest_patterns,
            "require_validation_status": self.require_validation_status,
            "required_validation_config": self.required_validation_config,
            "require_baselines": self.require_baselines,
            "required_metrics": self.required_metrics,
            "required_metric_slices": self.required_metric_slices,
            "required_output_artifacts": self.required_output_artifacts,
            "required_output_artifact_slices": self.required_output_artifact_slices,
            "required_validation_gates": [gate.to_dict() for gate in self.required_validation_gates],
            "require_no_errors": self.require_no_errors,
            "required_slices": self.required_slices,
            "require_candidate_fallback_metrics": self.require_candidate_fallback_metrics,
            "max_candidate_fallback_trigger_rate": self.max_candidate_fallback_trigger_rate,
            "max_candidate_fallback_trigger_rate_by_slice": self.max_candidate_fallback_trigger_rate_by_slice,
            "max_candidate_fallback_failed_rate": self.max_candidate_fallback_failed_rate,
        }


@dataclass(slots=True)
class ClaimReviewConfig:
    required_experiments: list[RequiredExperiment]
    require_shared_candidate_model: bool = True
    require_fallback_model_present: bool = False
    require_candidate_fallback_provenance: bool = False
    require_shared_fallback_model: bool = True
    require_model_audit_passed: bool = True
    require_fallback_model_audit_passed: bool = True
    require_trained_recognizer: bool = False
    require_model_admission: bool = False

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ClaimReviewConfig":
        if path is None:
            return default_claim_review_config()
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ClaimReviewError(f"invalid claim review config JSON {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ClaimReviewError("claim review config must be a JSON object")
        required = payload.get("required_experiments", [])
        if not isinstance(required, list):
            raise ClaimReviewError("required_experiments must be a list")
        experiments = [RequiredExperiment.from_dict(item) for item in required]
        if not experiments:
            raise ClaimReviewError("claim review config must define at least one required experiment")
        return cls(
            required_experiments=experiments,
            require_shared_candidate_model=bool(payload.get("require_shared_candidate_model", True)),
            require_fallback_model_present=bool(payload.get("require_fallback_model_present", False)),
            require_candidate_fallback_provenance=bool(payload.get("require_candidate_fallback_provenance", False)),
            require_shared_fallback_model=bool(payload.get("require_shared_fallback_model", True)),
            require_model_audit_passed=bool(payload.get("require_model_audit_passed", True)),
            require_fallback_model_audit_passed=bool(payload.get("require_fallback_model_audit_passed", True)),
            require_trained_recognizer=bool(payload.get("require_trained_recognizer", False)),
            require_model_admission=bool(payload.get("require_model_admission", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_experiments": [item.to_dict() for item in self.required_experiments],
            "require_shared_candidate_model": self.require_shared_candidate_model,
            "require_fallback_model_present": self.require_fallback_model_present,
            "require_candidate_fallback_provenance": self.require_candidate_fallback_provenance,
            "require_shared_fallback_model": self.require_shared_fallback_model,
            "require_model_audit_passed": self.require_model_audit_passed,
            "require_fallback_model_audit_passed": self.require_fallback_model_audit_passed,
            "require_trained_recognizer": self.require_trained_recognizer,
            "require_model_admission": self.require_model_admission,
        }


@dataclass(slots=True)
class ClaimReviewCheck:
    name: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(slots=True)
class ExperimentEvidence:
    experiment_path: str
    experiment_id: str
    eval_manifest: str
    eval_manifest_sha256: str | None
    recorded_eval_manifest_sha256: str | None
    eval_entries: list[ManifestEntry]
    datasets: list[str]
    slice_counts: dict[str, int]
    baselines: list[str]
    benchmark_report: str
    benchmark_rows: list[dict[str, Any]]
    benchmark_status_counts: dict[str, int]
    validation_status: str | None
    validation_report: str | None
    validation_config: str | None
    validation_config_sha256: str | None
    train_manifests: list[dict[str, str | None]]
    candidate_model_config: str | None
    candidate_model_config_sha256: str | None
    fallback_model_config: str | None
    fallback_model_config_sha256: str | None
    fallback_min_quality_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_path": self.experiment_path,
            "experiment_id": self.experiment_id,
            "eval_manifest": self.eval_manifest,
            "eval_manifest_sha256": self.eval_manifest_sha256,
            "recorded_eval_manifest_sha256": self.recorded_eval_manifest_sha256,
            "datasets": self.datasets,
            "slice_counts": self.slice_counts,
            "baselines": self.baselines,
            "benchmark_report": self.benchmark_report,
            "benchmark_status_counts": self.benchmark_status_counts,
            "validation_status": self.validation_status,
            "validation_report": self.validation_report,
            "validation_config": self.validation_config,
            "validation_config_sha256": self.validation_config_sha256,
            "train_manifests": self.train_manifests,
            "candidate_model_config": self.candidate_model_config,
            "candidate_model_config_sha256": self.candidate_model_config_sha256,
            "fallback_model_config": self.fallback_model_config,
            "fallback_model_config_sha256": self.fallback_model_config_sha256,
            "fallback_min_quality_score": self.fallback_min_quality_score,
        }


@dataclass(slots=True)
class ClaimReviewReport:
    passed: bool
    config: dict[str, Any]
    model_config: str | None
    experiments: list[ExperimentEvidence]
    checks: list[ClaimReviewCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "config": self.config,
            "model_config": self.model_config,
            "experiments": [item.to_dict() for item in self.experiments],
            "checks": [item.to_dict() for item in self.checks],
        }


def review_claim(
    experiment_reports: list[str | Path],
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    model_config: str | Path | None = None,
) -> ClaimReviewReport:
    if not experiment_reports:
        raise ClaimReviewError("claim review requires at least one experiment report")
    config = ClaimReviewConfig.from_path(config_path)
    evidence = [_load_experiment_evidence(path) for path in experiment_reports]
    checks: list[ClaimReviewCheck] = []
    checks.append(_shared_candidate_model_check(evidence, model_config, config))
    checks.append(_fallback_model_check(evidence, config))
    checks.append(_candidate_fallback_provenance_check(evidence, model_config, config))
    checks.append(_trained_recognizer_check(evidence, model_config, config))
    checks.append(_model_admission_review_check(evidence, model_config, config))
    checks.append(_candidate_calibration_review_check(evidence, model_config))
    checks.append(_candidate_baseline_equivalence_check(evidence, model_config))
    checks.extend(_required_experiment_checks(evidence, config))
    report = ClaimReviewReport(
        passed=all(check.status != "fail" for check in checks),
        config=config.to_dict(),
        model_config=str(model_config) if model_config else None,
        experiments=evidence,
        checks=checks,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_claim_review(report, out)
    return report


def default_claim_review_config() -> ClaimReviewConfig:
    return ClaimReviewConfig(
        required_experiments=[
            RequiredExperiment(
                name="internal-nepali-pack",
                datasets=["ocrtech-eval-pack"],
                require_validation_status="validated",
                require_baselines=["candidate", "tesseract", "stock-paddle", "glm-ocr", "paddleocr-vl"],
                required_metrics=[
                    "cer",
                    "wer",
                    "table_cell_f1",
                    "reading_order_pair_accuracy",
                    "figure_detection_f1",
                    "figure_caption_cer",
                ],
                required_slices=["nepali", "english", "table", "reading_order", "figure", "scan"],
            )
        ]
    )


def _load_experiment_evidence(path: str | Path) -> ExperimentEvidence:
    experiment_path = Path(path)
    _require_existing_file(experiment_path, "experiment report")
    try:
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClaimReviewError(f"invalid experiment JSON {experiment_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClaimReviewError(f"experiment report must be a JSON object: {experiment_path}")

    eval_manifest = str(payload.get("eval_manifest") or "")
    if not eval_manifest:
        raise ClaimReviewError(f"experiment report missing eval_manifest: {experiment_path}")
    _require_existing_file(Path(eval_manifest), f"eval_manifest in {experiment_path}")
    eval_entries = load_manifest(eval_manifest)
    eval_manifest_sha256 = sha256_file(Path(eval_manifest)) if Path(eval_manifest).exists() else None
    datasets = sorted({entry.dataset for entry in eval_entries})
    slice_counts = _slice_counts(eval_entries)

    benchmark_report = str(payload.get("benchmark_report") or "")
    if not benchmark_report:
        raise ClaimReviewError(f"experiment report missing benchmark_report: {experiment_path}")
    _require_existing_file(Path(benchmark_report), f"benchmark_report in {experiment_path}")
    benchmark_rows = _load_benchmark_report(benchmark_report)
    baselines = sorted({str(item.get("baseline") or "") for item in benchmark_rows if str(item.get("baseline") or "")})
    benchmark_status_counts = _status_counts(benchmark_rows)
    environment = dict(payload.get("environment") or {})

    return ExperimentEvidence(
        experiment_path=str(experiment_path),
        experiment_id=str(payload.get("experiment_id") or experiment_path.stem),
        eval_manifest=eval_manifest,
        eval_manifest_sha256=eval_manifest_sha256,
        recorded_eval_manifest_sha256=str(environment["eval_manifest_sha256"]) if environment.get("eval_manifest_sha256") else None,
        eval_entries=eval_entries,
        datasets=datasets,
        slice_counts=slice_counts,
        baselines=baselines,
        benchmark_report=benchmark_report,
        benchmark_rows=benchmark_rows,
        benchmark_status_counts={str(key): int(value) for key, value in benchmark_status_counts.items()},
        validation_status=str(payload["validation_status"]) if payload.get("validation_status") is not None else None,
        validation_report=str(payload["validation_report"]) if payload.get("validation_report") is not None else None,
        validation_config=str(environment["validation_config"]) if environment.get("validation_config") else None,
        validation_config_sha256=str(environment["validation_config_sha256"]) if environment.get("validation_config_sha256") else None,
        train_manifests=_environment_manifest_hashes(environment.get("train_manifests")),
        candidate_model_config=str(environment["candidate_model_config"]) if environment.get("candidate_model_config") else None,
        candidate_model_config_sha256=str(environment["candidate_model_config_sha256"]) if environment.get("candidate_model_config_sha256") else None,
        fallback_model_config=str(environment["fallback_model_config"]) if environment.get("fallback_model_config") else None,
        fallback_model_config_sha256=str(environment["fallback_model_config_sha256"]) if environment.get("fallback_model_config_sha256") else None,
        fallback_min_quality_score=_optional_float(environment.get("fallback_min_quality_score")),
    )


def _shared_candidate_model_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
    config: ClaimReviewConfig,
) -> ClaimReviewCheck:
    candidate_refs = [item for item in experiments if any(baseline in {"candidate", "ours", "ocrtech"} for baseline in item.baselines)]
    details: list[str] = []
    shas = sorted({item.candidate_model_config_sha256 for item in candidate_refs if item.candidate_model_config_sha256})
    missing = [item.experiment_id for item in candidate_refs if not item.candidate_model_config_sha256]
    if missing:
        return ClaimReviewCheck(
            "candidate-model",
            "fail",
            "candidate experiments are missing candidate_model_config_sha256",
            details=missing,
        )
    if config.require_shared_candidate_model and len(shas) > 1:
        return ClaimReviewCheck("candidate-model", "fail", "candidate experiments do not share one model config", details=shas)

    if model_config is not None:
        model_path = Path(model_config)
        if not model_path.exists():
            return ClaimReviewCheck("candidate-model", "fail", f"explicit model config does not exist: {model_path}")
        explicit_sha = sha256_file(model_path)
        if shas and explicit_sha not in shas:
            return ClaimReviewCheck(
                "candidate-model",
                "fail",
                "explicit model config does not match experiment evidence",
                details=[f"explicit_sha256={explicit_sha}", *shas],
            )
        if config.require_model_audit_passed:
            audit = audit_model_card(model_path)
            details.extend(audit.warnings)
            if not audit.passed:
                return ClaimReviewCheck("candidate-model", "fail", "explicit model card failed audit", details=[*audit.issues, *details])
            return ClaimReviewCheck("candidate-model", "pass", f"explicit model card audited: {audit.model_id}", details=details)
        return ClaimReviewCheck("candidate-model", "pass", "explicit model config matches experiment evidence")

    if candidate_refs and config.require_model_audit_passed:
        model_paths = sorted({item.candidate_model_config for item in candidate_refs if item.candidate_model_config})
        if len(model_paths) != 1:
            return ClaimReviewCheck(
                "candidate-model",
                "fail",
                "claim review requires one explicit candidate model path across candidate experiments",
                details=model_paths,
            )
        model_path = Path(model_paths[0])
        if not model_path.exists():
            return ClaimReviewCheck("candidate-model", "fail", f"candidate model config does not exist: {model_path}")
        actual_sha = sha256_file(model_path)
        if actual_sha not in shas:
            return ClaimReviewCheck(
                "candidate-model",
                "fail",
                "candidate model config sha256 does not match experiment evidence",
                details=[f"path={model_path}", f"actual_sha256={actual_sha}", *shas],
            )
        audit = audit_model_card(model_path)
        details.extend(audit.warnings)
        if not audit.passed:
            return ClaimReviewCheck("candidate-model", "fail", "candidate model card failed audit", details=[*audit.issues, *details])
        return ClaimReviewCheck(
            "candidate-model",
            "pass",
            f"candidate model card hash-checked and audited: {audit.model_id}",
            details=[f"path={model_path}", f"sha256={actual_sha}", *details],
        )

    if candidate_refs:
        return ClaimReviewCheck("candidate-model", "pass", "candidate experiments share one model config")
    return ClaimReviewCheck("candidate-model", "warn", "no candidate baselines found in supplied experiments")


def _fallback_model_check(experiments: list[ExperimentEvidence], config: ClaimReviewConfig) -> ClaimReviewCheck:
    candidate_refs = [item for item in experiments if any(baseline in {"candidate", "ours", "ocrtech"} for baseline in item.baselines)]
    missing_required = [
        item.experiment_id
        for item in candidate_refs
        if not item.fallback_model_config and not item.fallback_model_config_sha256
    ]
    if config.require_fallback_model_present and missing_required:
        return ClaimReviewCheck(
            "fallback-model",
            "fail",
            "candidate experiments are missing required fallback_model_config provenance",
            details=missing_required,
        )
    fallback_refs = [item for item in candidate_refs if item.fallback_model_config or item.fallback_model_config_sha256]
    if not fallback_refs:
        return ClaimReviewCheck("fallback-model", "pass", "no fallback model config recorded in candidate experiments")
    missing_path = [item.experiment_id for item in fallback_refs if not item.fallback_model_config]
    if missing_path:
        return ClaimReviewCheck("fallback-model", "fail", "fallback model sha256 is recorded but fallback_model_config path is missing", details=missing_path)
    missing_sha = [item.experiment_id for item in fallback_refs if not item.fallback_model_config_sha256]
    if missing_sha:
        return ClaimReviewCheck("fallback-model", "fail", "fallback model config path is recorded but sha256 is missing", details=missing_sha)
    shas = sorted({item.fallback_model_config_sha256 for item in fallback_refs if item.fallback_model_config_sha256})
    if config.require_shared_fallback_model and len(shas) > 1:
        return ClaimReviewCheck("fallback-model", "fail", "candidate experiments do not share one fallback model config", details=shas)

    details: list[str] = []
    model_paths = sorted({item.fallback_model_config for item in fallback_refs if item.fallback_model_config})
    for model_path_text in model_paths:
        model_path = Path(model_path_text)
        matching_shas = sorted({item.fallback_model_config_sha256 for item in fallback_refs if item.fallback_model_config == model_path_text})
        if not model_path.exists():
            return ClaimReviewCheck("fallback-model", "fail", f"fallback model config does not exist: {model_path}")
        actual_sha = sha256_file(model_path)
        if actual_sha not in matching_shas:
            return ClaimReviewCheck(
                "fallback-model",
                "fail",
                "fallback model config sha256 does not match experiment evidence",
                details=[f"path={model_path}", f"actual_sha256={actual_sha}", *matching_shas],
            )
        if config.require_fallback_model_audit_passed:
            audit = audit_model_card(model_path)
            details.extend(f"{model_path}: {warning}" for warning in audit.warnings)
            if not audit.passed:
                return ClaimReviewCheck(
                    "fallback-model",
                    "fail",
                    "fallback model card failed audit",
                    details=[f"{model_path}: {issue}" for issue in audit.issues] + details,
                )
            details.append(f"{model_path}: audited model_id={audit.model_id}")
    return ClaimReviewCheck("fallback-model", "pass", "fallback model configs are present, hash-checked, and audited", details=details)


def _candidate_fallback_provenance_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
    config: ClaimReviewConfig,
) -> ClaimReviewCheck:
    if not config.require_candidate_fallback_provenance:
        return ClaimReviewCheck("candidate-fallback-provenance", "pass", "candidate fallback provenance is not required by config")
    model_path = _single_candidate_model_path(experiments, model_config)
    if model_path is None:
        return ClaimReviewCheck("candidate-fallback-provenance", "fail", "candidate fallback provenance requires one candidate model path")
    if not model_path.exists():
        return ClaimReviewCheck("candidate-fallback-provenance", "fail", f"candidate model config does not exist: {model_path}")
    card = ModelCard.from_path(model_path)
    payload = card.provenance.get("fallback_model_card")
    if not isinstance(payload, dict):
        return ClaimReviewCheck("candidate-fallback-provenance", "fail", "candidate model card missing provenance.fallback_model_card")
    path_value = payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return ClaimReviewCheck("candidate-fallback-provenance", "fail", "candidate fallback_model_card.path is missing")
    fallback_path = Path(path_value)
    if not fallback_path.is_absolute():
        fallback_path = model_path.parent / fallback_path
    if not fallback_path.exists():
        return ClaimReviewCheck("candidate-fallback-provenance", "fail", f"candidate fallback model card does not exist: {fallback_path}")
    declared_sha = payload.get("sha256")
    actual_sha = sha256_file(fallback_path)
    if isinstance(declared_sha, str) and declared_sha and declared_sha != actual_sha:
        return ClaimReviewCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate fallback model card sha256 mismatch",
            details=[f"expected={declared_sha}", f"actual={actual_sha}", f"path={fallback_path}"],
        )
    fallback_refs = [item for item in experiments if item.fallback_model_config_sha256]
    mismatches = [
        f"{item.experiment_id}: experiment_fallback_sha256={item.fallback_model_config_sha256} candidate_fallback_sha256={actual_sha}"
        for item in fallback_refs
        if item.fallback_model_config_sha256 != actual_sha
    ]
    if mismatches:
        return ClaimReviewCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate fallback model card does not match experiment fallback_model_config",
            details=mismatches,
        )
    audit = audit_model_card(fallback_path)
    if not audit.passed:
        return ClaimReviewCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate fallback model card failed audit",
            details=[*audit.issues, *audit.warnings],
        )
    return ClaimReviewCheck(
        "candidate-fallback-provenance",
        "pass",
        "candidate fallback model provenance matches reviewed experiment fallback configs",
        details=[f"fallback_model_card={fallback_path}", f"sha256={actual_sha}", *audit.warnings],
    )


def _trained_recognizer_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
    config: ClaimReviewConfig,
) -> ClaimReviewCheck:
    if not config.require_trained_recognizer:
        return ClaimReviewCheck("trained-recognizer", "pass", "trained recognizer provenance is not required by config")
    model_path = _single_candidate_model_path(experiments, model_config)
    if model_path is None:
        return ClaimReviewCheck("trained-recognizer", "fail", "trained recognizer review requires one candidate model path")
    if not model_path.exists():
        return ClaimReviewCheck("trained-recognizer", "fail", f"candidate model config does not exist: {model_path}")
    card = ModelCard.from_path(model_path)
    status = trained_recognizer_provenance_status(card, model_path)
    if not status.passed:
        return ClaimReviewCheck(
            "trained-recognizer",
            "fail",
            "candidate model lacks trained local recognizer provenance",
            details=status.details,
        )
    return ClaimReviewCheck("trained-recognizer", "pass", "candidate includes trained local recognizer provenance", details=status.details)


def _model_admission_review_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
    config: ClaimReviewConfig,
) -> ClaimReviewCheck:
    if not config.require_model_admission:
        return ClaimReviewCheck("model-admission", "pass", "model admission validation is not required by config")
    model_path = _single_candidate_model_path(experiments, model_config)
    if model_path is None:
        return ClaimReviewCheck("model-admission", "fail", "model admission review requires one candidate model path")
    if not model_path.exists():
        return ClaimReviewCheck("model-admission", "fail", f"candidate model config does not exist: {model_path}")
    card = ModelCard.from_path(model_path)
    status = model_admission_status(card, model_path)
    if not status.passed:
        return ClaimReviewCheck("model-admission", "fail", "candidate model has not passed admission validation", details=status.details)
    return ClaimReviewCheck("model-admission", "pass", "candidate model passed admission validation", details=status.details)


def _single_candidate_model_path(experiments: list[ExperimentEvidence], model_config: str | Path | None) -> Path | None:
    if model_config is not None:
        return Path(model_config)
    candidate_refs = [item for item in experiments if any(baseline in {"candidate", "ours", "ocrtech"} for baseline in item.baselines)]
    distinct_paths = sorted({item.candidate_model_config for item in candidate_refs if item.candidate_model_config})
    if len(distinct_paths) != 1:
        return None
    return Path(distinct_paths[0])


def _required_experiment_checks(experiments: list[ExperimentEvidence], config: ClaimReviewConfig) -> list[ClaimReviewCheck]:
    checks: list[ClaimReviewCheck] = []
    for required in config.required_experiments:
        matches = [item for item in experiments if _matches_required_experiment(item, required)]
        if not matches:
            checks.append(
                ClaimReviewCheck(
                    f"required:{required.name}",
                    "fail",
                    "no experiment matches the required dataset/pattern constraints",
                    details=[*required.datasets, *required.eval_manifest_patterns],
                )
            )
            continue
        if len(matches) > 1:
            checks.append(
                ClaimReviewCheck(
                    f"required:{required.name}",
                    "fail",
                    "multiple experiments match one required slot; tighten the claim config",
                    details=[item.experiment_path for item in matches],
                )
            )
            continue
        evidence = matches[0]
        checks.append(_evaluate_required_experiment(evidence, required))
    return checks


def _candidate_calibration_review_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
) -> ClaimReviewCheck:
    candidate_refs = [item for item in experiments if any(baseline in {"candidate", "ours", "ocrtech"} for baseline in item.baselines)]
    if not candidate_refs:
        return ClaimReviewCheck("candidate-calibration", "warn", "no candidate experiments found; calibration review skipped")
    model_path = Path(model_config) if model_config is not None else None
    if model_path is None:
        distinct_paths = sorted({item.candidate_model_config for item in candidate_refs if item.candidate_model_config})
        if len(distinct_paths) != 1:
            return ClaimReviewCheck("candidate-calibration", "warn", "candidate calibration review skipped because no single model path is available")
        model_path = Path(distinct_paths[0])
    if not model_path.exists():
        return ClaimReviewCheck("candidate-calibration", "fail", f"candidate model config does not exist: {model_path}")
    card = ModelCard.from_path(model_path)
    if card.backend not in {"quality_select_composite", "quality_ranked_ensemble", "script_select_composite"}:
        return ClaimReviewCheck("candidate-calibration", "pass", "candidate backend does not require routed-threshold calibration review")
    calibration = card.provenance.get("calibration")
    if not isinstance(calibration, dict):
        return ClaimReviewCheck("candidate-calibration", "fail", f"{card.backend} candidate is missing provenance.calibration")
    eval_payload = calibration.get("eval_manifest")
    if not isinstance(eval_payload, dict):
        return ClaimReviewCheck("candidate-calibration", "fail", f"{card.backend} calibration is missing eval_manifest provenance")
    calibration_sha = eval_payload.get("sha256")
    if not isinstance(calibration_sha, str) or not calibration_sha:
        return ClaimReviewCheck("candidate-calibration", "fail", f"{card.backend} calibration eval manifest is missing sha256")
    overlaps = [item.experiment_id for item in candidate_refs if item.eval_manifest_sha256 and item.eval_manifest_sha256 == calibration_sha]
    if overlaps:
        return ClaimReviewCheck(
            "candidate-calibration",
            "fail",
            f"{card.backend} calibration eval manifest matches a reviewed experiment eval manifest",
            details=overlaps,
        )
    threshold = calibration.get("selected_threshold")
    details = []
    if isinstance(threshold, int | float):
        details.append(f"selected_threshold={float(threshold):.6f}")
    collapse_reason, collapse_details = inspect_routed_calibration_collapse(card, model_path)
    if collapse_reason is not None:
        return ClaimReviewCheck("candidate-calibration", "fail", collapse_reason, details=[*details, *collapse_details])
    return ClaimReviewCheck("candidate-calibration", "pass", f"{card.backend} calibration provenance is separate from reviewed eval manifests", details=details)


def _candidate_baseline_equivalence_check(
    experiments: list[ExperimentEvidence],
    model_config: str | Path | None,
) -> ClaimReviewCheck:
    candidate_refs = [item for item in experiments if any(baseline in {"candidate", "ours", "ocrtech"} for baseline in item.baselines)]
    if not candidate_refs:
        return ClaimReviewCheck("candidate-distinctness", "warn", "no candidate experiments found; baseline-equivalence review skipped")
    model_path = Path(model_config) if model_config is not None else None
    if model_path is None:
        distinct_paths = sorted({item.candidate_model_config for item in candidate_refs if item.candidate_model_config})
        if len(distinct_paths) != 1:
            return ClaimReviewCheck("candidate-distinctness", "warn", "candidate distinctness review skipped because no single model path is available")
        model_path = Path(distinct_paths[0])
    if not model_path.exists():
        return ClaimReviewCheck("candidate-distinctness", "fail", f"candidate model config does not exist: {model_path}")
    card = ModelCard.from_path(model_path)
    component_baselines = _component_baselines_for_card(card)
    if not component_baselines:
        return ClaimReviewCheck("candidate-distinctness", "pass", "candidate backend does not require component-baseline distinctness review")

    checked: list[str] = []
    for evidence in candidate_refs:
        candidate_baseline = _candidate_baseline_name(evidence.baselines)
        if candidate_baseline is None:
            continue
        for component_baseline in component_baselines:
            if component_baseline not in evidence.baselines:
                continue
            paired = _paired_metric_outcomes(
                evidence.benchmark_rows,
                candidate_baseline,
                component_baseline,
                metric="cer",
                slice_name="all",
            )
            if paired is None:
                continue
            checked.append(f"{evidence.experiment_id}:{component_baseline}")
            if paired["pairs"] > 0 and paired["win_count"] == 0 and paired["loss_count"] == 0:
                return ClaimReviewCheck(
                    "candidate-distinctness",
                    "fail",
                    f"candidate is indistinguishable from component baseline {component_baseline} on reviewed holdout CER",
                    details=[
                        f"experiment={evidence.experiment_id}",
                        f"pairs={paired['pairs']}",
                        f"candidate_mean={paired['candidate_mean']:.6f}",
                        f"baseline_mean={paired['baseline_mean']:.6f}",
                        "metric=cer",
                        "slice=all",
                    ],
                )
    if checked:
        return ClaimReviewCheck(
            "candidate-distinctness",
            "pass",
            "candidate is measurably distinct from reviewed component baselines on holdout CER",
            details=checked,
        )
    return ClaimReviewCheck(
        "candidate-distinctness",
        "warn",
        "no reviewed experiment includes component baselines needed for distinctness checks",
        details=component_baselines,
    )


def _component_baselines_for_card(card: ModelCard) -> list[str]:
    return component_baseline_names(card)


def _candidate_baseline_name(baselines: list[str]) -> str | None:
    for name in ("candidate", "ours", "ocrtech"):
        if name in baselines:
            return name
    return None


def _paired_metric_outcomes(
    rows: list[dict[str, Any]],
    candidate_baseline: str,
    baseline: str,
    *,
    metric: str,
    slice_name: str,
    tolerance: float = 1e-12,
) -> dict[str, float | int] | None:
    candidate_values = _rows_by_sample(rows, candidate_baseline, metric, slice_name)
    baseline_values = _rows_by_sample(rows, baseline, metric, slice_name)
    shared = sorted(set(candidate_values) & set(baseline_values))
    if not shared:
        return None
    candidate_series = [candidate_values[key] for key in shared]
    baseline_series = [baseline_values[key] for key in shared]
    wins = 0
    losses = 0
    ties = 0
    for candidate_value, baseline_value in zip(candidate_series, baseline_series, strict=True):
        delta = baseline_value - candidate_value
        if delta > tolerance:
            wins += 1
        elif delta < -tolerance:
            losses += 1
        else:
            ties += 1
    return {
        "pairs": len(shared),
        "candidate_mean": sum(candidate_series) / len(candidate_series),
        "baseline_mean": sum(baseline_series) / len(baseline_series),
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
    }


def _rows_by_sample(rows: list[dict[str, Any]], baseline: str, metric: str, slice_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row.get("baseline") != baseline or row.get("status") != "ok":
            continue
        if slice_name != "all" and slice_name not in set(row.get("slices") or []):
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values[sample_key(row)] = float(value)
    return values


def _matches_required_experiment(evidence: ExperimentEvidence, required: RequiredExperiment) -> bool:
    if required.datasets and not (set(required.datasets) & set(evidence.datasets)):
        return False
    if required.eval_manifest_patterns and not any(fnmatch(evidence.eval_manifest, pattern) for pattern in required.eval_manifest_patterns):
        return False
    return True


def _evaluate_required_experiment(evidence: ExperimentEvidence, required: RequiredExperiment) -> ClaimReviewCheck:
    details: list[str] = [f"experiment={evidence.experiment_id}", f"eval_manifest={evidence.eval_manifest}"]
    failures: list[str] = []
    if not evidence.recorded_eval_manifest_sha256:
        failures.append("experiment missing environment.eval_manifest_sha256")
    elif evidence.recorded_eval_manifest_sha256 != evidence.eval_manifest_sha256:
        failures.append(
            "experiment eval_manifest sha256 does not match environment.eval_manifest_sha256"
        )
    failures.extend(_benchmark_eval_manifest_binding_failures(evidence))
    if required.require_validation_status is not None and evidence.validation_status != required.require_validation_status:
        failures.append(
            f"validation_status {evidence.validation_status or 'not_run'} != required {required.require_validation_status}"
        )
    if required.required_validation_config is not None:
        failures.extend(_required_validation_config_failures(evidence, required.required_validation_config))
    if required.require_validation_status is not None:
        if not evidence.validation_report:
            failures.append("validation_report is missing")
        elif not Path(evidence.validation_report).exists():
            failures.append(f"validation_report does not exist: {evidence.validation_report}")
        else:
            failures.extend(_validation_report_failures(evidence.validation_report, required.require_validation_status, evidence))
            failures.extend(_validation_gate_failures(evidence.validation_report, required.required_validation_gates))
    missing_baselines = [baseline for baseline in required.require_baselines if baseline not in evidence.baselines]
    if missing_baselines:
        failures.append(f"missing baselines: {', '.join(missing_baselines)}")
    missing_baseline_samples = _missing_required_baseline_sample_coverage(evidence, required.require_baselines)
    if missing_baseline_samples:
        failures.append(f"missing baseline sample coverage: {', '.join(missing_baseline_samples)}")
    if required.require_no_errors and evidence.benchmark_status_counts.get("error", 0) > 0:
        failures.append(f"benchmark has {evidence.benchmark_status_counts['error']} error rows")
    missing_slices = [slice_name for slice_name in required.required_slices if evidence.slice_counts.get(slice_name, 0) < 1]
    if missing_slices:
        failures.append(f"missing slices: {', '.join(missing_slices)}")
    metric_baselines = required.require_baselines or (["candidate"] if "candidate" in evidence.baselines else evidence.baselines)
    missing_metrics = _missing_required_metrics(evidence.benchmark_rows, metric_baselines, required.required_metrics)
    if missing_metrics:
        failures.append(f"missing metrics: {', '.join(missing_metrics)}")
    missing_metric_slices = _missing_required_metric_slices(evidence.benchmark_rows, metric_baselines, required.required_metric_slices)
    if missing_metric_slices:
        failures.append(f"missing metric slice coverage: {', '.join(missing_metric_slices)}")
    missing_paired_metrics = _missing_required_paired_metric_coverage(
        evidence.benchmark_rows,
        metric_baselines,
        required.required_metrics,
        required.required_metric_slices,
    )
    if missing_paired_metrics:
        failures.append(f"missing paired metric coverage: {', '.join(missing_paired_metrics)}")
    missing_artifacts = _missing_required_output_artifacts(evidence, required)
    if missing_artifacts:
        failures.append(f"missing output artifacts: {', '.join(missing_artifacts)}")
    failures.extend(_candidate_fallback_policy_failures(evidence, required))
    details.append(f"datasets={', '.join(evidence.datasets)}")
    details.append(f"baselines={', '.join(evidence.baselines)}")
    if required.required_metrics:
        details.append(f"required_metrics={', '.join(required.required_metrics)}")
    if required.required_validation_config is not None:
        details.append(f"required_validation_config={required.required_validation_config}")
    if required.required_metric_slices:
        details.append(
            "required_metric_slices="
            + ", ".join(f"{metric}@{'+'.join(slices)}" for metric, slices in sorted(required.required_metric_slices.items()))
        )
    if required.required_output_artifacts:
        details.append(f"required_output_artifacts={', '.join(required.required_output_artifacts)}")
    if required.required_output_artifact_slices:
        details.append(
            "required_output_artifact_slices="
            + ", ".join(f"{artifact}@{'+'.join(slices)}" for artifact, slices in sorted(required.required_output_artifact_slices.items()))
        )
    if required.required_validation_gates:
        details.append(
            "required_validation_gates="
            + ", ".join(
                f"{gate.baseline}:{gate.metric}@{'+'.join(gate.slices) if gate.slices else 'any'}"
                for gate in required.required_validation_gates
            )
        )
    if required.require_candidate_fallback_metrics:
        details.append("require_candidate_fallback_metrics=true")
    if required.max_candidate_fallback_trigger_rate is not None:
        details.append(f"max_candidate_fallback_trigger_rate={required.max_candidate_fallback_trigger_rate:.6f}")
    if required.max_candidate_fallback_trigger_rate_by_slice:
        details.append(
            "max_candidate_fallback_trigger_rate_by_slice="
            + ", ".join(f"{slice_name}:{rate:.6f}" for slice_name, rate in sorted(required.max_candidate_fallback_trigger_rate_by_slice.items()))
        )
    if required.max_candidate_fallback_failed_rate is not None:
        details.append(f"max_candidate_fallback_failed_rate={required.max_candidate_fallback_failed_rate:.6f}")
    if failures:
        return ClaimReviewCheck(f"required:{required.name}", "fail", "; ".join(failures), details=details)
    summary = "required experiment evidence is present and passes configured checks"
    return ClaimReviewCheck(f"required:{required.name}", "pass", summary, details=details)


def _load_benchmark_report(path: str | Path) -> list[dict[str, Any]]:
    _require_existing_file(Path(path), "benchmark report")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClaimReviewError(f"invalid benchmark report JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ClaimReviewError(f"benchmark report must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ClaimReviewError(f"benchmark row {index} must be an object: {path}")
        rows.append(dict(item))
    return rows


def _benchmark_eval_manifest_binding_failures(evidence: ExperimentEvidence) -> list[str]:
    entries_by_sample = {entry.sample_id: entry for entry in evidence.eval_entries}
    failures: list[str] = []
    for row in evidence.benchmark_rows:
        if row.get("status") != "ok":
            continue
        sample_id_value = row.get("sample_id")
        if not isinstance(sample_id_value, str) or not sample_id_value:
            continue
        entry = entries_by_sample.get(sample_id_value)
        if entry is None:
            _append_once(failures, f"benchmark row sample_id not in eval_manifest: {sample_id_value}")
            continue
        input_path_value = row.get("input_path")
        if not isinstance(input_path_value, str) or not input_path_value:
            _append_once(failures, f"benchmark row missing input_path for sample_id: {sample_id_value}")
            continue
        if not _same_path(input_path_value, entry.image_path):
            _append_once(failures, f"benchmark row input_path does not match eval_manifest for sample_id: {sample_id_value}")
        row_slices = row.get("slices")
        if not isinstance(row_slices, list):
            _append_once(failures, f"benchmark row slices missing or invalid for sample_id: {sample_id_value}")
            continue
        if {str(item) for item in row_slices if str(item)} != set(_entry_slices(entry)):
            _append_once(failures, f"benchmark row slices do not match eval_manifest for sample_id: {sample_id_value}")
    return failures


def _append_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _missing_required_baseline_sample_coverage(evidence: ExperimentEvidence, required_baselines: list[str]) -> list[str]:
    if not required_baselines:
        return []
    required_samples = {entry.sample_id for entry in evidence.eval_entries}
    if not required_samples:
        return []
    missing: list[str] = []
    for baseline in required_baselines:
        ok_samples = {
            str(row.get("sample_id"))
            for row in evidence.benchmark_rows
            if row.get("baseline") == baseline
            and row.get("status") == "ok"
            and isinstance(row.get("sample_id"), str)
            and row.get("sample_id")
        }
        baseline_missing = sorted(required_samples - ok_samples)
        if not baseline_missing:
            continue
        preview = ", ".join(baseline_missing[:5])
        suffix = f" (+{len(baseline_missing) - 5} more)" if len(baseline_missing) > 5 else ""
        missing.append(f"{baseline}:{preview}{suffix}")
    return missing


def _same_path(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    try:
        if left_path.exists() and right_path.exists():
            return left_path.resolve() == right_path.resolve()
    except OSError:
        pass
    return str(left_path) == str(right_path)


def _required_validation_config_failures(evidence: ExperimentEvidence, required_validation_config: str) -> list[str]:
    config_path = Path(required_validation_config)
    if not config_path.exists():
        return [f"required validation config does not exist: {required_validation_config}"]
    if not config_path.is_file():
        return [f"required validation config is not a file: {required_validation_config}"]
    expected_sha = sha256_file(config_path)
    failures: list[str] = []
    if not evidence.validation_config:
        failures.append(f"experiment missing required validation_config: {required_validation_config}")
    if not evidence.validation_config_sha256:
        failures.append(f"experiment missing validation_config_sha256 for required config: {required_validation_config}")
    elif evidence.validation_config_sha256 != expected_sha:
        failures.append(
            f"experiment validation_config sha256 does not match required config {required_validation_config}: "
            f"expected={expected_sha} actual={evidence.validation_config_sha256}"
        )
    return failures


def _validation_report_failures(path: str | Path, required_status: str, evidence: ExperimentEvidence) -> list[str]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"validation_report invalid JSON: {report_path}: {exc}"]
    if not isinstance(payload, dict):
        return [f"validation_report must be a JSON object: {report_path}"]
    claim_status = str(payload.get("claim_status") or "")
    if claim_status != required_status:
        return [f"validation_report claim_status {claim_status or 'not_run'} != required {required_status}"]
    if required_status == "validated" and payload.get("passed") is not True:
        return ["validation_report passed is not true"]
    failures: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["validation_report missing provenance"]
    benchmark_payload = provenance.get("benchmark_report")
    if not isinstance(benchmark_payload, dict):
        failures.append("validation_report missing provenance.benchmark_report")
    else:
        benchmark_sha = benchmark_payload.get("sha256")
        expected_benchmark_sha = sha256_file(Path(evidence.benchmark_report))
        if benchmark_sha != expected_benchmark_sha:
            failures.append("validation_report benchmark_report sha256 does not match experiment benchmark_report")
    eval_payload = provenance.get("eval_manifest")
    if not isinstance(eval_payload, dict):
        failures.append("validation_report missing provenance.eval_manifest")
    else:
        eval_sha = eval_payload.get("sha256")
        if eval_sha != evidence.eval_manifest_sha256:
            failures.append("validation_report eval_manifest sha256 does not match experiment eval_manifest")
    config_payload = provenance.get("validation_config")
    if evidence.validation_config_sha256 is not None:
        if not isinstance(config_payload, dict):
            failures.append("validation_report missing provenance.validation_config")
        elif config_payload.get("sha256") != evidence.validation_config_sha256:
            failures.append("validation_report validation_config sha256 does not match experiment validation_config")
    elif config_payload not in (None, {}):
        failures.append("validation_report validation_config provenance is present but experiment has no validation_config")
    validation_train_manifests = _environment_manifest_hashes(provenance.get("train_manifests"))
    if _manifest_hash_signature(validation_train_manifests) != _manifest_hash_signature(evidence.train_manifests):
        failures.append("validation_report train_manifests provenance does not match experiment train_manifests")
    _append_optional_provenance_hash_failure(
        failures,
        provenance,
        "candidate_model_config",
        evidence.candidate_model_config_sha256,
        "candidate_model_config",
    )
    _append_optional_provenance_hash_failure(
        failures,
        provenance,
        "fallback_model_config",
        evidence.fallback_model_config_sha256,
        "fallback_model_config",
    )
    return failures


def _append_optional_provenance_hash_failure(
    failures: list[str],
    provenance: dict[str, Any],
    key: str,
    expected_sha256: str | None,
    label: str,
) -> None:
    payload = provenance.get(key)
    if expected_sha256 is not None:
        if not isinstance(payload, dict):
            failures.append(f"validation_report missing provenance.{key}")
        elif payload.get("sha256") != expected_sha256:
            failures.append(f"validation_report {label} sha256 does not match experiment {label}")
    elif payload not in (None, {}):
        failures.append(f"validation_report {label} provenance is present but experiment has no {label}")


def _validation_gate_failures(path: str | Path, required_gates: list[RequiredValidationGate]) -> list[str]:
    if not required_gates:
        return []
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"validation_report invalid JSON: {report_path}: {exc}"]
    if not isinstance(payload, dict):
        return [f"validation_report must be a JSON object: {report_path}"]
    gate_results = payload.get("gates")
    if not isinstance(gate_results, list):
        return ["validation_report missing gates"]
    failures: list[str] = []
    for required in required_gates:
        if not any(_validation_gate_matches(item, required) for item in gate_results):
            slice_suffix = "@" + "+".join(required.slices) if required.slices else ""
            failures.append(f"missing passing validation gate: {required.baseline}:{required.metric}{slice_suffix}")
    return failures


def _validation_gate_matches(item: Any, required: RequiredValidationGate) -> bool:
    if not isinstance(item, dict) or item.get("passed") is not True:
        return False
    gate = item.get("gate")
    if not isinstance(gate, dict):
        return False
    if gate.get("baseline") != required.baseline or gate.get("metric") != required.metric:
        return False
    if not required.slices:
        return gate_has_quantitative_evidence(item)
    gate_slices = gate.get("slices")
    if not isinstance(gate_slices, list):
        gate_slice = gate.get("slice")
        gate_slices = [gate_slice] if gate_slice else []
    return set(required.slices).issubset({str(slice_name) for slice_name in gate_slices}) and gate_has_quantitative_evidence(item)


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ClaimReviewError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ClaimReviewError(f"{label} is not a file: {path}")


def _missing_required_metrics(rows: list[dict[str, Any]], baselines: list[str], required_metrics: list[str]) -> list[str]:
    missing: list[str] = []
    if not required_metrics:
        return missing
    for baseline in baselines:
        ok_rows = [row for row in rows if row.get("baseline") == baseline and row.get("status") == "ok"]
        for metric in required_metrics:
            if not any(_row_has_finite_metric(row, metric) for row in ok_rows):
                missing.append(f"{baseline}:{metric}")
    return missing


def _missing_required_metric_slices(rows: list[dict[str, Any]], baselines: list[str], requirements: dict[str, list[str]]) -> list[str]:
    missing: list[str] = []
    if not requirements:
        return missing
    for baseline in baselines:
        ok_rows = [row for row in rows if row.get("baseline") == baseline and row.get("status") == "ok"]
        for metric, slices in requirements.items():
            for slice_name in slices:
                if not any(slice_name in set(row.get("slices") or []) and _row_has_finite_metric(row, metric) for row in ok_rows):
                    missing.append(f"{baseline}:{metric}@{slice_name}")
    return missing


def _missing_required_paired_metric_coverage(
    rows: list[dict[str, Any]],
    baselines: list[str],
    required_metrics: list[str],
    required_metric_slices: dict[str, list[str]],
) -> list[str]:
    missing: list[str] = []
    if not required_metrics:
        return missing
    candidate = _candidate_baseline_name(baselines)
    if candidate is None:
        return missing
    comparison_baselines = [baseline for baseline in baselines if baseline != candidate]
    for baseline in comparison_baselines:
        for metric in required_metrics:
            slice_names = required_metric_slices.get(metric) or ["all"]
            for slice_name in slice_names:
                if _paired_metric_outcomes(rows, candidate, baseline, metric=metric, slice_name=slice_name) is None:
                    missing.append(f"{candidate}:{baseline}:{metric}@{slice_name}")
    return missing


def _missing_required_output_artifacts(evidence: ExperimentEvidence, required: RequiredExperiment) -> list[str]:
    if not required.required_output_artifacts and not required.required_output_artifact_slices:
        return []
    candidate = _candidate_baseline_name(evidence.baselines)
    if candidate is None:
        return ["candidate output artifacts require a candidate/ours baseline"]
    ok_rows = [row for row in evidence.benchmark_rows if row.get("baseline") == candidate and row.get("status") == "ok"]
    if not ok_rows:
        return [f"candidate output artifacts have no ok rows for {candidate}"]
    benchmark_dir = Path(evidence.benchmark_report).parent
    missing: list[str] = []
    for row in ok_rows:
        output_dir = benchmark_dir / candidate / Path(str(row.get("input_path") or "")).stem
        sample = sample_key(row)
        for artifact in required.required_output_artifacts:
            artifact_issue = _output_artifact_issue(output_dir, artifact, row)
            if artifact_issue is not None:
                missing.append(f"{candidate}:{sample}:{artifact} ({artifact_issue})")
        row_slices = {str(item) for item in row.get("slices", [])}
        for artifact, slices in required.required_output_artifact_slices.items():
            for slice_name in slices:
                if slice_name not in row_slices:
                    continue
                artifact_issue = _output_artifact_issue(output_dir, artifact, row)
                if artifact_issue is not None:
                    missing.append(f"{candidate}:{sample}:{artifact}@{slice_name} ({artifact_issue})")
    return missing


def _output_artifact_issue(output_dir: Path, artifact: str, row: dict[str, Any] | None = None) -> str | None:
    records = _row_output_artifacts(row)
    if artifact == "document_json":
        path, record_issue = _single_artifact_path(records, "document_json", output_dir / "document.json")
        if record_issue is not None:
            return record_issue
        return _json_file_issue(path, expected_type=dict, required_keys=("pages", "tables", "figures"))
    if artifact == "document_md":
        path, record_issue = _single_artifact_path(records, "document_md", output_dir / "document.md")
        if record_issue is not None:
            return record_issue
        return _nonempty_file_issue(path)
    if artifact == "quality_json":
        path, record_issue = _single_artifact_path(records, "quality_json", output_dir / "quality.json")
        if record_issue is not None:
            return record_issue
        return _json_file_issue(path, expected_type=dict)
    if artifact == "table_csv_html":
        document, document_issue = _document_json_payload(output_dir, records)
        if document_issue is not None:
            return f"document_json {document_issue}"
        tables = document.get("tables") if document is not None else None
        if not isinstance(tables, list) or not tables:
            return "document_json has no tables"
        if records is not None:
            csv_paths, csv_issue = _artifact_path_list(records, "tables_csv")
            html_paths, html_issue = _artifact_path_list(records, "tables_html")
            if csv_issue is not None:
                return f"csv {csv_issue}"
            if html_issue is not None:
                return f"html {html_issue}"
            return _table_artifact_match_issue(tables, csv_paths, html_paths)
        tables_dir = output_dir / "tables"
        if not tables_dir.is_dir():
            return "tables directory missing"
        csv_issue = _any_nonempty_file_issue(tables_dir, "*.csv")
        html_issue = _any_nonempty_file_issue(tables_dir, "*.html")
        if csv_issue is not None:
            return f"csv {csv_issue}"
        if html_issue is not None:
            return f"html {html_issue}"
        table_ids = _object_ids(tables, "table_id")
        if table_ids:
            csv_stems = {path.stem for path in tables_dir.glob("*.csv") if path.is_file()}
            html_stems = {path.stem for path in tables_dir.glob("*.html") if path.is_file()}
            missing_csv = sorted(table_ids - csv_stems)
            missing_html = sorted(table_ids - html_stems)
            if missing_csv:
                return f"csv missing document tables: {', '.join(missing_csv[:5])}"
            if missing_html:
                return f"html missing document tables: {', '.join(missing_html[:5])}"
        return None
    if artifact == "figure_metadata":
        document, document_issue = _document_json_payload(output_dir, records)
        if document_issue is not None:
            return f"document_json {document_issue}"
        figures = document.get("figures") if document is not None else None
        if not isinstance(figures, list) or not figures:
            return "document_json has no figures"
        metadata_path, record_issue = _single_artifact_path(records, "figure_metadata", output_dir / "figures" / "metadata.json")
        if record_issue is not None:
            return record_issue
        metadata_issue = _json_file_issue(metadata_path, expected_type=list, require_nonempty=True)
        if metadata_issue is not None:
            return metadata_issue
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        figure_ids = _object_ids(figures, "figure_id")
        metadata_ids = _object_ids(metadata_payload, "figure_id")
        if figure_ids and metadata_ids and not figure_ids.issubset(metadata_ids):
            missing = sorted(figure_ids - metadata_ids)
            return f"metadata missing document figures: {', '.join(missing[:5])}"
        return None
    if artifact == "figure_images":
        document, document_issue = _document_json_payload(output_dir, records)
        if document_issue is not None:
            return f"document_json {document_issue}"
        figures = document.get("figures") if document is not None else None
        if not isinstance(figures, list) or not figures:
            return "document_json has no figures"
        if records is not None:
            image_files, image_issue = _artifact_path_list(records, "figure_images")
            if image_issue is not None:
                return image_issue
            return _figure_image_match_issue(figures, image_files)
        figures_dir = output_dir / "figures"
        if not figures_dir.is_dir():
            return "figures directory missing"
        image_files = [
            path
            for path in figures_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not image_files:
            return "no image files"
        if not any(path.stat().st_size > 0 for path in image_files):
            return "image files are empty"
        image_names = {path.name for path in image_files}
        declared_names = _declared_figure_image_names(figures)
        if declared_names and not declared_names.issubset(image_names):
            missing = sorted(declared_names - image_names)
            return f"image files missing document figures: {', '.join(missing[:5])}"
        return None
    return "unknown artifact type"


def _row_output_artifacts(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = row.get("output_artifacts")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return {}
    return payload


def _single_artifact_path(records: dict[str, Any] | None, key: str, fallback_path: Path) -> tuple[Path, str | None]:
    if records is None:
        return fallback_path, None
    record = records.get(key)
    if not isinstance(record, dict):
        return fallback_path, "missing artifact record"
    return _verified_artifact_record_path(record)


def _artifact_path_list(records: dict[str, Any], key: str) -> tuple[list[Path], str | None]:
    payload = records.get(key)
    if not isinstance(payload, list) or not payload:
        return [], "missing artifact records"
    paths: list[Path] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            return [], f"artifact record {index} is not an object"
        path, issue = _verified_artifact_record_path(item)
        if issue is not None:
            return [], f"artifact record {index} {issue}"
        paths.append(path)
    return paths, None


def _verified_artifact_record_path(record: dict[str, Any]) -> tuple[Path, str | None]:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        return Path(), "missing path in artifact record"
    path = Path(path_value)
    if not path.is_file():
        return path, "missing"
    size_value = record.get("size_bytes")
    if not isinstance(size_value, int):
        return path, "missing size_bytes in artifact record"
    actual_size = path.stat().st_size
    if actual_size != size_value:
        return path, f"size mismatch: expected {size_value}, actual {actual_size}"
    sha_value = record.get("sha256")
    if not isinstance(sha_value, str) or not sha_value:
        return path, "missing sha256 in artifact record"
    actual_sha = sha256_file(path)
    if actual_sha != sha_value:
        return path, f"sha256 mismatch: expected {sha_value}, actual {actual_sha}"
    return path, None


def _table_artifact_match_issue(tables: list[Any], csv_paths: list[Path], html_paths: list[Path]) -> str | None:
    if not any(path.stat().st_size > 0 for path in csv_paths):
        return "csv empty *.csv"
    if not any(path.stat().st_size > 0 for path in html_paths):
        return "html empty *.html"
    table_ids = _object_ids(tables, "table_id")
    if table_ids:
        csv_stems = {path.stem for path in csv_paths}
        html_stems = {path.stem for path in html_paths}
        missing_csv = sorted(table_ids - csv_stems)
        missing_html = sorted(table_ids - html_stems)
        if missing_csv:
            return f"csv missing document tables: {', '.join(missing_csv[:5])}"
        if missing_html:
            return f"html missing document tables: {', '.join(missing_html[:5])}"
    return None


def _figure_image_match_issue(figures: list[Any], image_files: list[Path]) -> str | None:
    if not image_files:
        return "no image files"
    if not any(path.stat().st_size > 0 for path in image_files):
        return "image files are empty"
    image_names = {path.name for path in image_files}
    declared_names = _declared_figure_image_names(figures)
    if declared_names and not declared_names.issubset(image_names):
        missing = sorted(declared_names - image_names)
        return f"image files missing document figures: {', '.join(missing[:5])}"
    return None


def _document_json_payload(output_dir: Path, records: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    path, record_issue = _single_artifact_path(records, "document_json", output_dir / "document.json")
    if record_issue is not None:
        return None, record_issue
    issue = _json_file_issue(path, expected_type=dict, required_keys=("pages", "tables", "figures"))
    if issue is not None:
        return None, issue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "expected dict"
    return payload, None


def _object_ids(items: list[Any], key: str) -> set[str]:
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _declared_figure_image_names(figures: list[Any]) -> set[str]:
    names: set[str] = set()
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        image_path = figure.get("image_path")
        if isinstance(image_path, str) and image_path:
            names.add(Path(image_path).name)
    return names


def _json_file_issue(
    path: Path,
    *,
    expected_type: type[dict[str, Any]] | type[list[Any]],
    required_keys: tuple[str, ...] = (),
    require_nonempty: bool = False,
) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid json: {exc}"
    if not isinstance(payload, expected_type):
        return f"expected {expected_type.__name__}"
    if require_nonempty and not payload:
        return "empty payload"
    if isinstance(payload, dict):
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            return f"missing keys: {', '.join(missing_keys)}"
    return None


def _nonempty_file_issue(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    return None


def _any_nonempty_file_issue(directory: Path, pattern: str) -> str | None:
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if not files:
        return f"missing {pattern}"
    if not any(path.stat().st_size > 0 for path in files):
        return f"empty {pattern}"
    return None


def _candidate_fallback_policy_failures(evidence: ExperimentEvidence, required: RequiredExperiment) -> list[str]:
    if (
        not required.require_candidate_fallback_metrics
        and required.max_candidate_fallback_trigger_rate is None
        and not required.max_candidate_fallback_trigger_rate_by_slice
        and required.max_candidate_fallback_failed_rate is None
    ):
        return []
    candidate = _candidate_baseline_name(evidence.baselines)
    if candidate is None:
        return ["candidate fallback policy requires a candidate/ours baseline"]
    ok_rows = [row for row in evidence.benchmark_rows if row.get("baseline") == candidate and row.get("status") == "ok"]
    if not ok_rows:
        return [f"candidate fallback policy has no ok rows for {candidate}"]
    failures: list[str] = []
    if required.require_candidate_fallback_metrics:
        missing_triggered = [sample_key(row) for row in ok_rows if not _row_has_finite_metric(row, "fallback_triggered")]
        missing_failed = [sample_key(row) for row in ok_rows if not _row_has_finite_metric(row, "fallback_failed")]
        missing_primary_quality = [
            sample_key(row)
            for row in ok_rows
            if evidence.fallback_min_quality_score is not None and not _row_has_finite_metric(row, "fallback_primary_quality_score")
        ]
        missing_min_quality = [
            sample_key(row)
            for row in ok_rows
            if evidence.fallback_min_quality_score is not None and not _row_has_finite_metric(row, "fallback_min_quality_score")
        ]
        if missing_triggered:
            failures.append(f"missing candidate fallback_triggered metrics on {len(missing_triggered)} rows")
        if missing_failed:
            failures.append(f"missing candidate fallback_failed metrics on {len(missing_failed)} rows")
        if missing_primary_quality:
            failures.append(f"missing candidate fallback_primary_quality_score metrics on {len(missing_primary_quality)} rows")
        if missing_min_quality:
            failures.append(f"missing candidate fallback_min_quality_score metrics on {len(missing_min_quality)} rows")
    if required.max_candidate_fallback_trigger_rate is not None:
        trigger_rate = _mean_metric(ok_rows, "fallback_triggered")
        if trigger_rate is None:
            failures.append("candidate fallback_triggered metrics are missing for trigger-rate limit")
        elif trigger_rate > required.max_candidate_fallback_trigger_rate:
            failures.append(
                f"candidate fallback trigger rate {trigger_rate:.6f} exceeds max {required.max_candidate_fallback_trigger_rate:.6f}"
            )
    for slice_name, max_rate in sorted(required.max_candidate_fallback_trigger_rate_by_slice.items()):
        slice_rows = [row for row in ok_rows if slice_name in {str(item) for item in row.get("slices", [])}]
        if not slice_rows:
            failures.append(f"candidate fallback slice trigger-rate limit has no ok rows for slice {slice_name}")
            continue
        trigger_rate = _mean_metric(slice_rows, "fallback_triggered")
        if trigger_rate is None:
            failures.append(f"candidate fallback_triggered metrics are missing for slice {slice_name}")
        elif trigger_rate > max_rate:
            failures.append(
                f"candidate fallback trigger rate {trigger_rate:.6f} on slice {slice_name} exceeds max {max_rate:.6f}"
            )
    if required.max_candidate_fallback_failed_rate is not None:
        failed_rate = _mean_metric(ok_rows, "fallback_failed")
        if failed_rate is None:
            failures.append("candidate fallback_failed metrics are missing for failure-rate limit")
        elif failed_rate > required.max_candidate_fallback_failed_rate:
            failures.append(
                f"candidate fallback failure rate {failed_rate:.6f} exceeds max {required.max_candidate_fallback_failed_rate:.6f}"
            )
    return failures


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _row_has_finite_metric(row: dict[str, Any], metric: str) -> bool:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    value = metrics.get(metric)
    return isinstance(value, int | float) and math.isfinite(float(value))


def _optional_rate(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ClaimReviewError(f"{field_name} must be a number between 0 and 1")
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ClaimReviewError(f"{field_name} must be a number between 0 and 1")
    return rate


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _optional_rate_mapping(value: Any, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ClaimReviewError(f"{field_name} must be an object mapping slice names to rates")
    result: dict[str, float] = {}
    for key, item in value.items():
        slice_name = str(key)
        if not slice_name:
            raise ClaimReviewError(f"{field_name} slice names must be non-empty")
        rate = _optional_rate(item, f"{field_name}.{slice_name}")
        if rate is None:
            raise ClaimReviewError(f"{field_name}.{slice_name} must be a number between 0 and 1")
        result[slice_name] = rate
    return result


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        status = str(item.get("status") or "")
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _slice_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        for slice_name in _entry_slices(entry):
            pair = (entry.sample_id, slice_name)
            if pair in seen:
                continue
            seen.add(pair)
            counts[slice_name] = counts.get(slice_name, 0) + 1
    return dict(sorted(counts.items()))


def _entry_slices(entry: ManifestEntry) -> list[str]:
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


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ClaimReviewError("expected a list of strings")
    result = [str(item) for item in value if str(item)]
    if len(result) != len(value):
        raise ClaimReviewError("list items must be non-empty strings")
    return result


def _environment_manifest_hashes(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    manifests: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        sha = item.get("sha256")
        manifests.append(
            {
                "path": str(path) if path is not None else None,
                "sha256": str(sha) if sha is not None else None,
            }
        )
    return manifests


def _manifest_hash_signature(manifests: list[dict[str, str | None]]) -> list[tuple[str | None, str | None]]:
    return sorted(
        ((item.get("path"), item.get("sha256")) for item in manifests),
        key=lambda item: (item[0] or "", item[1] or ""),
    )


def _metric_slice_requirements(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ClaimReviewError("required_metric_slices must be an object mapping metric names to slice lists")
    result: dict[str, list[str]] = {}
    for metric, slices in value.items():
        metric_name = str(metric)
        if not metric_name:
            raise ClaimReviewError("required_metric_slices metric names must be non-empty strings")
        parsed_slices = _string_list(slices)
        if not parsed_slices:
            raise ClaimReviewError(f"required_metric_slices for {metric_name!r} must define at least one slice")
        result[metric_name] = parsed_slices
    return result


def _required_validation_gates(value: Any) -> list[RequiredValidationGate]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ClaimReviewError("required_validation_gates must be a list")
    gates: list[RequiredValidationGate] = []
    for item in value:
        if not isinstance(item, dict):
            raise ClaimReviewError("required_validation_gates items must be objects")
        gates.append(RequiredValidationGate.from_dict(item))
    return gates


def _validate_required_validation_gates(
    experiment_name: str,
    require_validation_status: str | None,
    require_baselines: list[str],
    required_metrics: list[str],
    required_validation_gates: list[RequiredValidationGate],
) -> None:
    if not required_validation_gates:
        return
    if require_validation_status is None:
        raise ClaimReviewError(f"required experiment {experiment_name!r} defines required_validation_gates without require_validation_status")
    baseline_set = set(require_baselines)
    missing_baselines = sorted({gate.baseline for gate in required_validation_gates if gate.baseline not in baseline_set})
    if missing_baselines:
        raise ClaimReviewError(
            f"required experiment {experiment_name!r} has validation gate baselines not listed in require_baselines: {', '.join(missing_baselines)}"
        )
    metric_set = set(required_metrics)
    missing_metrics = sorted({gate.metric for gate in required_validation_gates if gate.metric not in metric_set})
    if missing_metrics:
        raise ClaimReviewError(
            f"required experiment {experiment_name!r} has validation gate metrics not listed in required_metrics: {', '.join(missing_metrics)}"
        )


def _write_claim_review(report: ClaimReviewReport, output_dir: Path) -> None:
    (output_dir / "claim-review.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR Claim Review",
        "",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Model config: `{report.model_config or ''}`",
        "",
        "| check | status | summary |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        summary = check.summary.replace("|", "\\|")
        lines.append(f"| {check.name} | {check.status} | {summary} |")
        for detail in check.details:
            escaped_detail = detail.replace("|", "\\|")
            lines.append(f"|  |  | {escaped_detail} |")
    lines.extend(["", "## Experiments", ""])
    for experiment in report.experiments:
        lines.append(f"- `{experiment.experiment_id}` datasets=`{', '.join(experiment.datasets)}` baselines=`{', '.join(experiment.baselines)}`")
    (output_dir / "claim-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
