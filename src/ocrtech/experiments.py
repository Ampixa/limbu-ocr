"""Reproducible experiment runner for benchmark and validation jobs."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .benchmark import BenchmarkResult, run_benchmark
from .errors import BenchmarkError
from .manifest import ManifestEntry, load_manifest, sha256_file
from .preflight import run_claim_preflight
from .validation import ValidationReport, validate_claim


@dataclass(slots=True)
class ExperimentReport:
    experiment_id: str
    output_dir: str
    eval_manifest: str
    references_dir: str | None
    baselines: list[str]
    input_count: int
    datasets: list[str]
    slice_counts: dict[str, int]
    benchmark_report: str
    benchmark_status_counts: dict[str, int]
    preflight_report: str | None = None
    preflight_passed: bool | None = None
    validation_report: str | None = None
    validation_status: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "output_dir": self.output_dir,
            "eval_manifest": self.eval_manifest,
            "references_dir": self.references_dir,
            "baselines": self.baselines,
            "input_count": self.input_count,
            "datasets": self.datasets,
            "slice_counts": self.slice_counts,
            "benchmark_report": self.benchmark_report,
            "benchmark_status_counts": self.benchmark_status_counts,
            "preflight_report": self.preflight_report,
            "preflight_passed": self.preflight_passed,
            "validation_report": self.validation_report,
            "validation_status": self.validation_status,
            "environment": self.environment,
        }


def run_experiment(
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    baselines: list[str],
    references_dir: str | Path | None = None,
    validation_config: str | Path | None = None,
    train_manifests: list[str | Path] | None = None,
    candidate_model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    fallback_min_quality_score: float | None = None,
    gorkhapatra_review_audits: list[str | Path] | None = None,
    run_validation: bool = True,
    run_preflight: bool = False,
    require_claim_ready_eval_pack: bool = True,
    require_trained_recognizer: bool = False,
    require_model_admission: bool = False,
) -> ExperimentReport:
    if not baselines:
        raise BenchmarkError("run-experiment requires at least one baseline")
    eval_path = Path(eval_manifest)
    entries = load_manifest(eval_path)
    if not entries:
        raise BenchmarkError("eval manifest contains no inputs")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs = Path(references_dir) if references_dir else _infer_references_dir(eval_path)
    inputs = [entry.image_path for entry in entries]
    datasets = sorted({entry.dataset for entry in entries})
    slice_counts = _slice_counts(entries)

    preflight_report_path: str | None = None
    preflight_passed: bool | None = None
    if run_preflight:
        preflight_dir = out / "preflight"
        preflight = run_claim_preflight(
            eval_path,
            preflight_dir,
            baselines=baselines,
            references_dir=refs,
            candidate_model_config=candidate_model_config,
            fallback_model_config=fallback_model_config,
            train_manifests=train_manifests or [],
            validation_config=validation_config,
            gorkhapatra_review_audits=gorkhapatra_review_audits or [],
            require_claim_ready_eval_pack=require_claim_ready_eval_pack,
            require_trained_recognizer=require_trained_recognizer,
            require_model_admission=require_model_admission,
        )
        preflight_report_path = str(preflight_dir / "preflight.json")
        preflight_passed = preflight.passed
        if not preflight.passed:
            raise BenchmarkError(f"preflight failed; see {preflight_report_path}")

    benchmark_dir = out / "benchmark"
    results = run_benchmark(
        inputs,
        benchmark_dir,
        baselines=baselines,
        references_dir=refs,
        eval_manifest=eval_path,
        candidate_model_config=candidate_model_config,
        fallback_engine=fallback_engine,
        fallback_model_config=fallback_model_config,
        low_confidence_threshold=low_confidence_threshold,
        fallback_min_quality_score=fallback_min_quality_score,
    )
    status_counts = _status_counts(results)
    validation_status = None
    validation_report_path: str | None = None
    if run_validation:
        validation_dir = out / "validation"
        validation = validate_claim(
            benchmark_dir / "report.json",
            validation_dir,
            config_path=validation_config,
            eval_manifest=eval_path,
            train_manifests=train_manifests or [],
            candidate_model_config=candidate_model_config,
            fallback_model_config=fallback_model_config,
        )
        validation_status = validation.claim_status
        validation_report_path = str(validation_dir / "validation.json")
    report = ExperimentReport(
        experiment_id=out.name or "experiment",
        output_dir=str(out),
        eval_manifest=str(eval_path),
        references_dir=str(refs) if refs else None,
        baselines=baselines,
        input_count=len(inputs),
        datasets=datasets,
        slice_counts=slice_counts,
        benchmark_report=str(benchmark_dir / "report.json"),
        benchmark_status_counts=status_counts,
        preflight_report=preflight_report_path,
        preflight_passed=preflight_passed,
        validation_report=validation_report_path,
        validation_status=validation_status,
        environment=_environment(
            eval_path,
            refs,
            validation_config,
            train_manifests or [],
            candidate_model_config,
            fallback_engine,
            fallback_model_config,
            low_confidence_threshold,
            fallback_min_quality_score,
        ),
    )
    _write_experiment_report(report, out)
    return report


def _infer_references_dir(eval_manifest: Path) -> Path | None:
    candidate = eval_manifest.parent.parent / "references"
    return candidate if candidate.exists() else None


def _status_counts(results: list[BenchmarkResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return dict(sorted(counts.items()))


def _slice_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        metadata = entry.metadata or {}
        values: set[str] = set()
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
        for slice_name in sorted(values):
            pair = (entry.sample_id, slice_name)
            if pair in seen:
                continue
            seen.add(pair)
            counts[slice_name] = counts.get(slice_name, 0) + 1
    return dict(sorted(counts.items()))


def _environment(
    eval_manifest: Path,
    references_dir: Path | None,
    validation_config: str | Path | None,
    train_manifests: list[str | Path],
    candidate_model_config: str | Path | None,
    fallback_engine: str | None,
    fallback_model_config: str | Path | None,
    low_confidence_threshold: float,
    fallback_min_quality_score: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ocrtech_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "eval_manifest_sha256": sha256_file(eval_manifest) if eval_manifest.exists() else None,
        "references_dir": str(references_dir) if references_dir else None,
        "validation_config": str(validation_config) if validation_config else None,
        "validation_config_sha256": sha256_file(Path(validation_config)) if validation_config and Path(validation_config).exists() else None,
        "candidate_model_config": str(candidate_model_config) if candidate_model_config else None,
        "candidate_model_config_sha256": sha256_file(Path(candidate_model_config)) if candidate_model_config and Path(candidate_model_config).exists() else None,
        "fallback_engine": fallback_engine,
        "fallback_model_config": str(fallback_model_config) if fallback_model_config else None,
        "fallback_model_config_sha256": sha256_file(Path(fallback_model_config)) if fallback_model_config and Path(fallback_model_config).exists() else None,
        "low_confidence_threshold": low_confidence_threshold,
        "fallback_min_quality_score": fallback_min_quality_score,
        "train_manifests": [],
    }
    for manifest in train_manifests:
        path = Path(manifest)
        payload["train_manifests"].append({"path": str(path), "sha256": sha256_file(path) if path.exists() else None})
    return payload


def _write_experiment_report(report: ExperimentReport, output_dir: Path) -> None:
    (output_dir / "experiment.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR Experiment",
        "",
        f"Experiment: `{report.experiment_id}`",
        f"Eval manifest: `{report.eval_manifest}`",
        f"References: `{report.references_dir or ''}`",
        f"Inputs: `{report.input_count}`",
        f"Datasets: `{', '.join(report.datasets)}`",
        f"Baselines: `{', '.join(report.baselines)}`",
        f"Preflight: `{report.preflight_passed if report.preflight_passed is not None else 'not_run'}`",
        f"Benchmark report: `{report.benchmark_report}`",
        f"Validation status: `{report.validation_status or 'not_run'}`",
        "",
        "## Benchmark Status",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in report.benchmark_status_counts.items())
    if not report.benchmark_status_counts:
        lines.append("- none")
    lines.extend(["", "## Slice Coverage", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in report.slice_counts.items())
    if not report.slice_counts:
        lines.append("- none")
    (output_dir / "experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
