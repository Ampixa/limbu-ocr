"""Model artifact cards and audit helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import sha256_file
from .validation import gate_has_quantitative_evidence


@dataclass(slots=True)
class ArtifactRef:
    path: str
    required: bool = True
    sha256: str | None = None
    kind: str = "file"
    size_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            path=str(data.get("path") or ""),
            required=bool(data.get("required", True)),
            sha256=data.get("sha256"),
            kind=str(data.get("kind") or "file"),
            size_bytes=int(data["size_bytes"]) if isinstance(data.get("size_bytes"), int) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "required": self.required,
            "sha256": self.sha256,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
        }


@dataclass(slots=True)
class ModelCard:
    model_id: str
    backend: str
    base_model: str
    created_at_utc: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelCard":
        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Invalid model card JSON {config_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataValidationError(f"model card must be a JSON object: {config_path}")
        artifacts_payload = payload.get("artifacts", [])
        if not isinstance(artifacts_payload, list):
            raise DataValidationError("model card artifacts must be a list")
        card = cls(
            model_id=str(payload.get("model_id") or ""),
            backend=str(payload.get("backend") or "paddleocr"),
            base_model=str(payload.get("base_model") or ""),
            created_at_utc=str(payload.get("created_at_utc") or ""),
            artifacts=[ArtifactRef.from_dict(item) for item in artifacts_payload],
            backend_kwargs=dict(payload.get("backend_kwargs") or payload.get("paddleocr_kwargs") or {}),
            provenance=dict(payload.get("provenance") or {}),
            metrics=dict(payload.get("metrics") or {}),
            notes=payload.get("notes"),
        )
        card.validate_shape()
        return card

    def validate_shape(self) -> None:
        if not self.model_id:
            raise DataValidationError("model card requires model_id")
        if self.backend not in {"paddleocr", "sidecar", "tesseract", "surya", "hf_vision_encoder_decoder", "text_correction_composite", "quality_select_composite", "quality_ranked_ensemble", "script_select_composite", "line_align_composite", "table_cell_refine_composite", "figure_caption_refine_composite"}:
            raise DataValidationError(f"unsupported model backend: {self.backend}")
        if not isinstance(self.backend_kwargs, dict):
            raise DataValidationError("backend_kwargs must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "base_model": self.base_model,
            "created_at_utc": self.created_at_utc,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "backend_kwargs": self.backend_kwargs,
            "provenance": self.provenance,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ModelAudit:
    model_config: str
    model_id: str
    backend: str
    passed: bool
    total_artifact_size_bytes: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "model_id": self.model_id,
            "backend": self.backend,
            "passed": self.passed,
            "total_artifact_size_bytes": self.total_artifact_size_bytes,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class TrainedRecognizerProvenance:
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelAdmissionStatus:
    passed: bool
    details: list[str] = field(default_factory=list)


def write_model_card(
    output_path: str | Path,
    *,
    model_id: str,
    backend: str = "paddleocr",
    base_model: str = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams"
    ),
    artifact_paths: list[str | Path] | None = None,
    paddleocr_kwargs: dict[str, Any] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    train_manifest: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    training_summary: str | Path | None = None,
    metrics_report: str | Path | None = None,
    admission_validation_report: str | Path | None = None,
    fallback_model_card: str | Path | None = None,
    artifact_base_path: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    artifact_base = Path(artifact_base_path) if artifact_base_path is not None else None
    artifacts = [_artifact_ref(Path(item), base_path=artifact_base) for item in artifact_paths or []]
    provenance_payload: dict[str, Any] = dict(provenance or {})
    if train_manifest:
        train_path = Path(train_manifest)
        provenance_payload["train_manifest"] = {"path": str(train_path), "sha256": sha256_file(train_path) if train_path.exists() else None}
    if eval_manifest:
        eval_path = Path(eval_manifest)
        provenance_payload["eval_manifest"] = {"path": str(eval_path), "sha256": sha256_file(eval_path) if eval_path.exists() else None}
    if training_summary:
        summary_path = Path(training_summary)
        provenance_payload["training_summary"] = {"path": str(summary_path), "sha256": sha256_file(summary_path) if summary_path.exists() else None}
    metrics: dict[str, Any] = {}
    if metrics_report:
        metrics_path = Path(metrics_report)
        metrics["report"] = {"path": str(metrics_path), "sha256": sha256_file(metrics_path) if metrics_path.exists() else None}
    if admission_validation_report:
        admission_path = Path(admission_validation_report)
        metrics["admission_validation"] = {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path) if admission_path.exists() else None,
            "required_status": "validated",
        }
    if fallback_model_card:
        fallback_path = Path(fallback_model_card)
        provenance_payload["fallback_model_card"] = {
            "path": str(fallback_path),
            "sha256": sha256_file(fallback_path) if fallback_path.exists() else None,
        }
    card = ModelCard(
        model_id=model_id,
        backend=backend,
        base_model=base_model,
        created_at_utc=datetime.now(UTC).isoformat(),
        artifacts=artifacts,
        backend_kwargs=backend_kwargs or paddleocr_kwargs or {},
        provenance=provenance_payload,
        metrics=metrics,
        notes=notes,
    )
    card.validate_shape()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def audit_model_card(model_config: str | Path, output_dir: str | Path | None = None) -> ModelAudit:
    config_path = Path(model_config)
    card = ModelCard.from_path(config_path)
    issues: list[str] = []
    warnings: list[str] = []
    total_artifact_size_bytes = 0
    for artifact in card.artifacts:
        path = _resolve_artifact_path(artifact.path, config_path)
        if not artifact.path:
            issues.append("artifact path is empty")
            continue
        if artifact.required and not path.exists():
            issues.append(f"required artifact missing: {artifact.path}")
            continue
        if path.exists() and artifact.sha256 and path.is_file():
            actual = sha256_file(path)
            if actual != artifact.sha256:
                issues.append(f"artifact sha256 mismatch for {artifact.path}: manifest={artifact.sha256} actual={actual}")
        if path.exists() and artifact.kind == "directory" and not path.is_dir():
            issues.append(f"artifact expected directory but is not: {artifact.path}")
        if path.exists() and artifact.kind == "file" and not path.is_file():
            issues.append(f"artifact expected file but is not: {artifact.path}")
        if path.exists():
            actual_size = _artifact_size_bytes(path)
            total_artifact_size_bytes += actual_size
            if artifact.size_bytes is not None and actual_size != artifact.size_bytes:
                issues.append(f"artifact size mismatch for {artifact.path}: manifest={artifact.size_bytes} actual={actual_size}")
    if card.backend == "paddleocr" and not card.artifacts:
        issues.append("paddleocr candidate requires at least one recorded artifact")
    if card.backend == "paddleocr" and not card.backend_kwargs:
        warnings.append("paddleocr backend has no backend_kwargs; installed PaddleOCR defaults will be used")
    if card.backend == "tesseract" and not card.backend_kwargs:
        warnings.append("tesseract backend has no backend_kwargs; installed Tesseract defaults will be used")
    if card.backend == "surya" and not card.backend_kwargs:
        warnings.append("surya backend has no backend_kwargs; installed Surya defaults will be used")
    if card.backend == "hf_vision_encoder_decoder" and not card.artifacts:
        issues.append("hf_vision_encoder_decoder candidate requires at least one recorded artifact")
    if card.backend == "hf_vision_encoder_decoder" and not card.backend_kwargs:
        warnings.append("hf_vision_encoder_decoder backend has no backend_kwargs; default generation settings will be used")
    if card.backend == "text_correction_composite" and not card.artifacts:
        issues.append("text_correction_composite candidate requires at least one recorded artifact")
    if card.backend == "text_correction_composite":
        if "base_engine" not in card.backend_kwargs:
            warnings.append("text_correction_composite backend has no base_engine; default tesseract will be used")
        if "model_dir" not in card.backend_kwargs:
            issues.append("text_correction_composite candidate requires backend_kwargs.model_dir")
    if card.backend == "quality_select_composite":
        if "primary_engine" not in card.backend_kwargs:
            issues.append("quality_select_composite candidate requires backend_kwargs.primary_engine")
        if "secondary_engine" not in card.backend_kwargs:
            issues.append("quality_select_composite candidate requires backend_kwargs.secondary_engine")
        if "secondary_quality_margin" not in card.backend_kwargs:
            warnings.append("quality_select_composite backend has no secondary_quality_margin; default 0.0 will be used")
    if card.backend == "quality_ranked_ensemble":
        if "primary_engine" not in card.backend_kwargs:
            issues.append("quality_ranked_ensemble candidate requires backend_kwargs.primary_engine")
        alternatives = card.backend_kwargs.get("alternative_engines")
        if not isinstance(alternatives, list) or not alternatives:
            issues.append("quality_ranked_ensemble candidate requires backend_kwargs.alternative_engines")
    if card.backend == "script_select_composite":
        if "primary_engine" not in card.backend_kwargs:
            issues.append("script_select_composite candidate requires backend_kwargs.primary_engine")
        if "secondary_engine" not in card.backend_kwargs:
            issues.append("script_select_composite candidate requires backend_kwargs.secondary_engine")
        if "primary_script_threshold" not in card.backend_kwargs:
            warnings.append("script_select_composite backend has no primary_script_threshold; default 0.2 will be used")
        routing_signal_engine = card.backend_kwargs.get("routing_signal_engine")
        if routing_signal_engine is not None and routing_signal_engine not in {"primary", "secondary", "both"}:
            issues.append("script_select_composite routing_signal_engine must be primary, secondary, or both")
    if card.backend == "line_align_composite":
        if "primary_engine" not in card.backend_kwargs:
            issues.append("line_align_composite candidate requires backend_kwargs.primary_engine")
        if "secondary_engine" not in card.backend_kwargs:
            issues.append("line_align_composite candidate requires backend_kwargs.secondary_engine")
        if "append_unmatched_primary" not in card.backend_kwargs:
            warnings.append("line_align_composite backend has no append_unmatched_primary; default true will be used")
    if card.backend == "table_cell_refine_composite":
        if "structure_engine" not in card.backend_kwargs:
            warnings.append("table_cell_refine_composite backend has no structure_engine; default surya will be used")
        if "crop_engine" not in card.backend_kwargs:
            warnings.append("table_cell_refine_composite backend has no crop_engine; default tesseract will be used")
        elif card.backend_kwargs.get("crop_engine") == "candidate":
            crop_kwargs = card.backend_kwargs.get("crop_engine_kwargs")
            if not isinstance(crop_kwargs, dict) or not crop_kwargs.get("model_config"):
                issues.append("table_cell_refine_composite crop_engine=candidate requires backend_kwargs.crop_engine_kwargs.model_config")
        replacement_policy = card.backend_kwargs.get("replacement_policy")
        if replacement_policy is not None and replacement_policy not in {"empty", "guarded", "always"}:
            issues.append("table_cell_refine_composite replacement_policy must be empty, guarded, or always")
        confidence_delta = card.backend_kwargs.get("min_replacement_confidence_delta")
        if confidence_delta is not None:
            if isinstance(confidence_delta, bool) or not isinstance(confidence_delta, int | float) or not math.isfinite(float(confidence_delta)):
                issues.append("table_cell_refine_composite min_replacement_confidence_delta must be a finite number")
            elif float(confidence_delta) < 0:
                issues.append("table_cell_refine_composite min_replacement_confidence_delta must be non-negative")
    if card.backend == "figure_caption_refine_composite":
        if "primary_engine" not in card.backend_kwargs:
            warnings.append("figure_caption_refine_composite backend has no primary_engine; default surya will be used")
        if "caption_engine" not in card.backend_kwargs:
            warnings.append("figure_caption_refine_composite backend has no caption_engine; default tesseract will be used")
        elif card.backend_kwargs.get("caption_engine") == "candidate":
            caption_kwargs = card.backend_kwargs.get("caption_engine_kwargs")
            if not isinstance(caption_kwargs, dict) or not caption_kwargs.get("model_config"):
                issues.append("figure_caption_refine_composite caption_engine=candidate requires backend_kwargs.caption_engine_kwargs.model_config")
        _append_numeric_kwarg_issue(
            issues,
            card.backend_kwargs,
            "min_caption_iou",
            "figure_caption_refine_composite min_caption_iou",
            min_value=0.0,
            max_value=1.0,
        )
        _append_numeric_kwarg_issue(
            issues,
            card.backend_kwargs,
            "min_score_delta",
            "figure_caption_refine_composite min_score_delta",
            min_value=0.0,
        )
        _append_numeric_kwarg_issue(
            issues,
            card.backend_kwargs,
            "max_gap_multiplier",
            "figure_caption_refine_composite max_gap_multiplier",
            min_value=0.0,
            min_inclusive=False,
        )
        _append_numeric_kwarg_issue(
            issues,
            card.backend_kwargs,
            "table_name_max_distance",
            "figure_caption_refine_composite table_name_max_distance",
            min_value=0.0,
        )
        lexicon_path = card.backend_kwargs.get("table_name_lexicon_path")
        if lexicon_path is not None:
            if not isinstance(lexicon_path, str) or not lexicon_path:
                issues.append("figure_caption_refine_composite table_name_lexicon_path must be a non-empty string")
            elif not _resolve_artifact_path(lexicon_path, config_path).is_file():
                issues.append(f"figure_caption_refine_composite table_name_lexicon_path does not exist: {lexicon_path}")
    audit = ModelAudit(str(config_path), card.model_id, card.backend, not issues, total_artifact_size_bytes, issues, warnings)
    if output_dir is not None:
        _write_model_audit(audit, Path(output_dir))
    return audit


def _append_numeric_kwarg_issue(
    issues: list[str],
    kwargs: dict[str, Any],
    key: str,
    label: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    value = kwargs.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        issues.append(f"{label} must be a finite number")
        return
    numeric = float(value)
    if min_value is not None:
        too_low = numeric < min_value if min_inclusive else numeric <= min_value
        if too_low:
            comparator = ">=" if min_inclusive else ">"
            issues.append(f"{label} must be {comparator} {min_value:g}")
            return
    if max_value is not None:
        too_high = numeric > max_value if max_inclusive else numeric >= max_value
        if too_high:
            comparator = "<=" if max_inclusive else "<"
            issues.append(f"{label} must be {comparator} {max_value:g}")
            return


def _artifact_ref(path: Path, *, base_path: Path | None = None) -> ArtifactRef:
    hash_path = path if path.is_absolute() or base_path is None else base_path / path
    if hash_path.exists() and hash_path.is_file():
        return ArtifactRef(path=str(path), required=True, sha256=sha256_file(hash_path), kind="file", size_bytes=hash_path.stat().st_size)
    if hash_path.exists() and hash_path.is_dir():
        return ArtifactRef(path=str(path), required=True, sha256=None, kind="directory", size_bytes=_artifact_size_bytes(hash_path))
    kind = "directory" if path.suffix == "" else "file"
    return ArtifactRef(path=str(path), required=True, sha256=None, kind=kind)


def _artifact_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _resolve_artifact_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return config_path.parent / candidate


def _write_model_audit(audit: ModelAudit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model-audit.json").write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Model Audit",
        "",
        f"Model: `{audit.model_id}`",
        f"Backend: `{audit.backend}`",
        f"Passed: `{'yes' if audit.passed else 'no'}`",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in audit.issues) if audit.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in audit.warnings) if audit.warnings else lines.append("- none")
    (output_dir / "model-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_routed_calibration_collapse(card: ModelCard, model_config: str | Path) -> tuple[str | None, list[str]]:
    if card.backend not in {"quality_select_composite", "script_select_composite"}:
        return None, []
    report_payload = card.metrics.get("report")
    if not isinstance(report_payload, dict):
        return "missing calibration metrics report reference", []
    report_path_value = report_payload.get("path")
    if not isinstance(report_path_value, str) or not report_path_value:
        return "missing calibration metrics report path", []
    report_path = _resolve_artifact_path(report_path_value, Path(model_config))
    if not report_path.exists():
        return f"calibration metrics report does not exist: {report_path}", []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid calibration metrics report JSON {report_path}: {exc}", []
    if not isinstance(payload, dict):
        return f"calibration metrics report must be a JSON object: {report_path}", []
    trial = payload.get("selected_trial")
    if not isinstance(trial, dict):
        return f"calibration metrics report missing selected_trial: {report_path}", []
    pairs = trial.get("pairs")
    if isinstance(pairs, bool) or not isinstance(pairs, int | float) or int(pairs) <= 0:
        return f"calibration selected_trial has invalid pairs value: {pairs!r}", []
    pair_count = int(pairs)
    if card.backend == "quality_select_composite":
        picks = trial.get("candidate_picks")
        if isinstance(picks, bool) or not isinstance(picks, int | float):
            return f"quality_select_composite calibration missing candidate_picks: {report_path}", []
        pick_count = int(picks)
        details = _collapse_details(payload, report_path, pair_count, pick_count, "candidate_picks")
        if pick_count == 0:
            return "quality_select_composite collapses to the primary engine on every calibration sample", details
        if pick_count == pair_count:
            return "quality_select_composite collapses to the secondary/candidate engine on every calibration sample", details
        return None, details
    picks = trial.get("secondary_picks")
    if isinstance(picks, bool) or not isinstance(picks, int | float):
        return f"script_select_composite calibration missing secondary_picks: {report_path}", []
    pick_count = int(picks)
    details = _collapse_details(payload, report_path, pair_count, pick_count, "secondary_picks")
    if pick_count == 0:
        return "script_select_composite collapses to the primary engine on every calibration sample", details
    if pick_count == pair_count:
        return "script_select_composite collapses to the secondary engine on every calibration sample", details
    return None, details


def _collapse_details(payload: dict[str, Any], report_path: Path, pairs: int, picks: int, label: str) -> list[str]:
    details = [f"report={report_path}", f"pairs={pairs}", f"{label}={picks}"]
    threshold = payload.get("selected_threshold")
    if isinstance(threshold, int | float) and math.isfinite(float(threshold)):
        details.append(f"selected_threshold={float(threshold):.6f}")
    metric = payload.get("metric")
    if isinstance(metric, str) and metric:
        details.append(f"metric={metric}")
    return details


def component_baseline_names(card: ModelCard) -> list[str]:
    kwargs = card.backend_kwargs
    engines: list[str] = []
    if card.backend in {"quality_select_composite", "script_select_composite", "line_align_composite"}:
        for key in ("primary_engine", "secondary_engine"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                engines.append(value)
    elif card.backend == "table_cell_refine_composite":
        for key in ("structure_engine", "crop_engine"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                engines.append(value)
    elif card.backend == "figure_caption_refine_composite":
        for key in ("primary_engine", "caption_engine"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                engines.append(value)
    elif card.backend == "quality_ranked_ensemble":
        primary = kwargs.get("primary_engine")
        if isinstance(primary, str) and primary:
            engines.append(primary)
        alternatives = kwargs.get("alternative_engines")
        if isinstance(alternatives, list):
            for item in alternatives:
                if isinstance(item, dict):
                    engine_name = item.get("engine")
                    if isinstance(engine_name, str) and engine_name:
                        engines.append(engine_name)
    normalized = [_normalize_component_baseline_name(name) for name in engines]
    return sorted({name for name in normalized if name})


def _normalize_component_baseline_name(name: str) -> str:
    value = name.strip().lower()
    if value in {"paddleocr", "paddle"}:
        return "stock-paddle"
    return value


TRAINED_RECOGNIZER_BACKENDS = {"paddleocr", "hf_vision_encoder_decoder"}


def trained_recognizer_provenance_status(card: ModelCard, model_config: str | Path) -> TrainedRecognizerProvenance:
    passed, details = _model_card_has_trained_recognizer(card, Path(model_config), seen=set())
    return TrainedRecognizerProvenance(passed=passed, details=details)


def model_admission_status(card: ModelCard, model_config: str | Path) -> ModelAdmissionStatus:
    leaf_status = _candidate_leaf_model_admission_status(card, Path(model_config))
    if leaf_status is not None:
        return leaf_status
    payload = card.metrics.get("admission_validation")
    if not isinstance(payload, dict):
        return ModelAdmissionStatus(False, ["missing metrics.admission_validation"])
    path_value = payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return ModelAdmissionStatus(False, ["metrics.admission_validation.path is missing"])
    report_path = _resolve_artifact_path(path_value, Path(model_config))
    if not report_path.exists():
        return ModelAdmissionStatus(False, [f"admission validation report does not exist: {report_path}"])
    expected_sha = payload.get("sha256")
    if isinstance(expected_sha, str) and expected_sha:
        actual_sha = sha256_file(report_path)
        if actual_sha != expected_sha:
            return ModelAdmissionStatus(False, [f"admission validation sha256 mismatch: expected={expected_sha} actual={actual_sha}"])
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ModelAdmissionStatus(False, [f"invalid admission validation JSON {report_path}: {exc}"])
    if not isinstance(report, dict):
        return ModelAdmissionStatus(False, [f"admission validation report must be a JSON object: {report_path}"])
    required_status = str(payload.get("required_status") or "validated")
    claim_status = str(report.get("claim_status") or "")
    passed = bool(report.get("passed"))
    details = [f"report={report_path}", f"claim_status={claim_status or '<missing>'}", f"required_status={required_status}", f"passed={passed}"]
    binding_passed, binding_details = _admission_validation_model_binding(card, report, report_path)
    details.extend(binding_details)
    gate_evidence_passed, gate_evidence_details = _admission_validation_gate_evidence(report)
    details.extend(gate_evidence_details)
    details.extend(_admission_validation_failure_details(report))
    if claim_status == required_status and passed and binding_passed and gate_evidence_passed:
        return ModelAdmissionStatus(True, details)
    return ModelAdmissionStatus(False, details)


def _candidate_leaf_model_admission_status(card: ModelCard, card_path: Path) -> ModelAdmissionStatus | None:
    candidate_leaf_refs = _candidate_leaf_model_config_refs(card, card_path)
    if not candidate_leaf_refs:
        return None
    failures: list[str] = [f"model_id={card.model_id}", f"backend={card.backend}"]
    for label, nested_path in candidate_leaf_refs:
        prefix = [f"{label}={nested_path}"]
        if not nested_path.exists():
            failures.extend([*prefix, f"candidate leaf model card does not exist: {nested_path}"])
            continue
        try:
            nested_card = ModelCard.from_path(nested_path)
        except DataValidationError as exc:
            failures.extend([*prefix, f"candidate leaf model card is invalid: {exc}"])
            continue
        nested_status = model_admission_status(nested_card, nested_path)
        details = [*prefix, *nested_status.details]
        if nested_status.passed:
            return ModelAdmissionStatus(True, details)
        failures.extend(details)
    return ModelAdmissionStatus(False, failures)


def _admission_validation_model_binding(
    card: ModelCard,
    report: dict[str, Any],
    report_path: Path,
) -> tuple[bool, list[str]]:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        return False, ["admission validation missing provenance"]
    candidate_payload = provenance.get("candidate_model_config")
    if not isinstance(candidate_payload, dict):
        return False, ["admission validation missing provenance.candidate_model_config"]
    path_value = candidate_payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return False, ["admission validation candidate_model_config.path is missing"]
    candidate_path = _resolve_report_provenance_path(path_value, report_path)
    details = [f"admission_candidate_model_config={candidate_path}"]
    if not candidate_path.exists():
        return False, [*details, f"admission candidate model config does not exist: {candidate_path}"]
    try:
        admitted_card = ModelCard.from_path(candidate_path)
    except DataValidationError as exc:
        return False, [*details, f"admission candidate model config is invalid: {exc}"]
    mismatches = _stable_model_card_mismatches(card, admitted_card)
    if mismatches:
        return False, [*details, *mismatches]
    return True, [*details, "admission candidate model identity matches current model card"]


def _resolve_report_provenance_path(path_value: str, report_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    candidate = report_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _stable_model_card_mismatches(current: ModelCard, admitted: ModelCard) -> list[str]:
    mismatches: list[str] = []
    for field_name in ("model_id", "backend", "base_model"):
        if getattr(current, field_name) != getattr(admitted, field_name):
            mismatches.append(
                f"admission candidate model {field_name} mismatch: current={getattr(current, field_name)!r} "
                f"admitted={getattr(admitted, field_name)!r}"
            )
    current_artifacts = _stable_artifact_identity(current)
    admitted_artifacts = _stable_artifact_identity(admitted)
    if current_artifacts != admitted_artifacts:
        mismatches.append("admission candidate model artifacts do not match current model card")
    return mismatches


def _stable_artifact_identity(card: ModelCard) -> list[tuple[str, str | None, str, bool]]:
    return sorted((artifact.path, artifact.sha256, artifact.kind, artifact.required) for artifact in card.artifacts)


def _admission_validation_failure_details(report: dict[str, Any]) -> list[str]:
    details: list[str] = []
    missing = report.get("missing_requirements")
    if isinstance(missing, list):
        for item in missing[:10]:
            if isinstance(item, str) and item:
                details.append(f"missing_requirement={item}")
    gates = report.get("gates")
    if isinstance(gates, list):
        failed_count = 0
        for item in gates:
            if not isinstance(item, dict) or bool(item.get("passed")):
                continue
            failed_count += 1
            details.append(_admission_gate_detail(item))
            if failed_count >= 10:
                break
    return details


def _admission_validation_gate_evidence(report: dict[str, Any]) -> tuple[bool, list[str]]:
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        return False, ["admission validation missing gates"]
    details: list[str] = []
    for index, item in enumerate(gates):
        if not isinstance(item, dict):
            return False, [f"admission validation gate {index} is not an object"]
        if item.get("passed") is not True:
            return False, [f"admission validation gate {index} is not passed"]
        if not gate_has_quantitative_evidence(item):
            return False, [f"admission validation gate {index} lacks quantitative passing evidence"]
        details.append(f"admission_gate_{index}=quantitative_pass")
    return True, details


def _admission_gate_detail(gate_result: dict[str, Any]) -> str:
    gate = gate_result.get("gate")
    gate_payload = gate if isinstance(gate, dict) else {}
    metric = str(gate_payload.get("metric") or "<missing>")
    baseline = str(gate_payload.get("baseline") or "<missing>")
    slices = gate_payload.get("slices")
    if isinstance(slices, list) and slices:
        slice_label = ",".join(str(item) for item in slices)
    else:
        slice_label = str(gate_payload.get("slice") or "all")
    reason = gate_result.get("reason")
    parts = [
        f"failed_gate={metric}",
        f"baseline={baseline}",
        f"slice={slice_label}",
    ]
    for key in ("candidate_mean", "baseline_mean", "absolute_improvement", "relative_improvement", "pairs", "win_count", "loss_count", "tie_count"):
        value = gate_result.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if isinstance(reason, str) and reason:
        parts.append(f"reason={reason}")
    return " ".join(parts)


def _model_card_has_trained_recognizer(card: ModelCard, card_path: Path, *, seen: set[Path]) -> tuple[bool, list[str]]:
    resolved_card_path = card_path.resolve()
    if resolved_card_path in seen:
        return False, [f"cycle in trained recognizer provenance at {card_path}"]
    seen.add(resolved_card_path)
    details = [f"model_id={card.model_id}", f"backend={card.backend}"]
    if card.backend in TRAINED_RECOGNIZER_BACKENDS:
        missing = _trained_recognizer_missing_fields(card, card_path)
        if missing:
            return False, [*details, *missing]
        return True, [
            *details,
            "train_manifest provenance hash verified",
            "eval_manifest provenance hash verified",
            "training_summary provenance hash verified",
            "artifacts present",
        ]
    candidate_leaf_refs = _candidate_leaf_model_config_refs(card, card_path)
    if candidate_leaf_refs:
        failures: list[str] = []
        for label, nested_path in candidate_leaf_refs:
            nested_passed, nested_details = _check_nested_trained_recognizer_model_card(
                nested_path,
                seen=seen,
                details=[*details, f"{label}={nested_path}"],
            )
            if nested_passed:
                return True, nested_details
            failures.extend(nested_details)
        return False, failures

    nested = card.provenance.get("trained_recognizer_model_card")
    if not isinstance(nested, dict):
        return False, [*details, "missing provenance.trained_recognizer_model_card"]
    nested_path_value = nested.get("path")
    nested_sha = nested.get("sha256")
    if not isinstance(nested_path_value, str) or not nested_path_value:
        return False, [*details, "trained_recognizer_model_card.path is missing"]
    nested_path = Path(nested_path_value)
    if not nested_path.is_absolute():
        nested_path = card_path.parent / nested_path
    if not nested_path.exists():
        return False, [*details, f"trained recognizer model card does not exist: {nested_path}"]
    if isinstance(nested_sha, str) and nested_sha:
        actual_sha = sha256_file(nested_path)
        if actual_sha != nested_sha:
            return False, [*details, f"trained recognizer model card sha256 mismatch: expected={nested_sha} actual={actual_sha}"]
    nested_card = ModelCard.from_path(nested_path)
    nested_passed, nested_details = _model_card_has_trained_recognizer(nested_card, nested_path, seen=seen)
    return nested_passed, [*details, f"trained_recognizer_model_card={nested_path}", *nested_details]


def _candidate_leaf_model_config_refs(card: ModelCard, card_path: Path) -> list[tuple[str, Path]]:
    refs: list[tuple[str, Path]] = []
    if card.backend == "table_cell_refine_composite" and card.backend_kwargs.get("crop_engine") == "candidate":
        model_config = _nested_model_config_value(card.backend_kwargs.get("crop_engine_kwargs"))
        if model_config is not None:
            refs.append(("crop_engine_candidate_model_config", _resolve_artifact_path(model_config, card_path)))
    if card.backend == "figure_caption_refine_composite" and card.backend_kwargs.get("caption_engine") == "candidate":
        model_config = _nested_model_config_value(card.backend_kwargs.get("caption_engine_kwargs"))
        if model_config is not None:
            refs.append(("caption_engine_candidate_model_config", _resolve_artifact_path(model_config, card_path)))
    return refs


def _nested_model_config_value(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("model_config")
    return value if isinstance(value, str) and value else None


def _check_nested_trained_recognizer_model_card(
    nested_path: Path,
    *,
    seen: set[Path],
    details: list[str],
) -> tuple[bool, list[str]]:
    if not nested_path.exists():
        return False, [*details, f"trained recognizer model card does not exist: {nested_path}"]
    try:
        nested_card = ModelCard.from_path(nested_path)
    except DataValidationError as exc:
        return False, [*details, f"trained recognizer model card is invalid: {exc}"]
    nested_passed, nested_details = _model_card_has_trained_recognizer(nested_card, nested_path, seen=seen)
    return nested_passed, [*details, *nested_details]


def _trained_recognizer_missing_fields(card: ModelCard, card_path: Path) -> list[str]:
    missing: list[str] = []
    if not card.artifacts:
        missing.append("trained recognizer has no artifacts")
    missing.extend(_trained_recognizer_provenance_file_issues(card, card_path, "train_manifest"))
    missing.extend(_trained_recognizer_provenance_file_issues(card, card_path, "eval_manifest"))
    missing.extend(_trained_recognizer_provenance_file_issues(card, card_path, "training_summary"))
    return missing


def _trained_recognizer_provenance_file_issues(card: ModelCard, card_path: Path, key: str) -> list[str]:
    payload = card.provenance.get(key)
    if not isinstance(payload, dict):
        return [f"trained recognizer missing {key} path/sha256 provenance"]
    path_value = payload.get("path")
    expected_sha = payload.get("sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(expected_sha, str) or not expected_sha:
        return [f"trained recognizer missing {key} path/sha256 provenance"]
    path = _resolve_artifact_path(path_value, card_path)
    if not path.is_file():
        return [f"trained recognizer {key} does not exist: {path}"]
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        return [f"trained recognizer {key} sha256 mismatch: expected={expected_sha} actual={actual_sha}"]
    return []
