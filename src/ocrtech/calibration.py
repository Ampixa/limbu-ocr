"""Calibration helpers for routed OCR candidates."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BenchmarkError
from .experiments import ExperimentReport
from .manifest import sha256_file
from .models import write_model_card
from .validation import metric_direction, sample_key


@dataclass(slots=True)
class ThresholdTrial:
    threshold: float
    pairs: int
    candidate_picks: int
    chosen_mean: float
    primary_mean: float
    absolute_improvement: float
    relative_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "pairs": self.pairs,
            "candidate_picks": self.candidate_picks,
            "chosen_mean": self.chosen_mean,
            "primary_mean": self.primary_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
        }


@dataclass(slots=True)
class CalibrationReport:
    experiment_report: str
    benchmark_report: str
    eval_manifest: str
    candidate_baseline: str
    primary_baseline: str
    metric: str
    direction: str
    selected_threshold: float
    selected_trial: ThresholdTrial
    trials: list[ThresholdTrial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_report": self.experiment_report,
            "benchmark_report": self.benchmark_report,
            "eval_manifest": self.eval_manifest,
            "candidate_baseline": self.candidate_baseline,
            "primary_baseline": self.primary_baseline,
            "metric": self.metric,
            "direction": self.direction,
            "selected_threshold": self.selected_threshold,
            "selected_trial": self.selected_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


@dataclass(slots=True)
class ScriptThresholdTrial:
    threshold: float
    pairs: int
    secondary_picks: int
    chosen_mean: float
    primary_mean: float
    absolute_improvement: float
    relative_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "pairs": self.pairs,
            "secondary_picks": self.secondary_picks,
            "chosen_mean": self.chosen_mean,
            "primary_mean": self.primary_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
        }


@dataclass(slots=True)
class ScriptCalibrationReport:
    experiment_report: str
    benchmark_report: str
    eval_manifest: str
    primary_baseline: str
    secondary_baseline: str
    metric: str
    direction: str
    script: str
    selected_threshold: float
    selected_trial: ScriptThresholdTrial
    trials: list[ScriptThresholdTrial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_report": self.experiment_report,
            "benchmark_report": self.benchmark_report,
            "eval_manifest": self.eval_manifest,
            "primary_baseline": self.primary_baseline,
            "secondary_baseline": self.secondary_baseline,
            "metric": self.metric,
            "direction": self.direction,
            "script": self.script,
            "selected_threshold": self.selected_threshold,
            "selected_trial": self.selected_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def calibrate_quality_router(
    experiment_report: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    primary_engine: str,
    secondary_engine: str,
    primary_engine_kwargs: dict[str, Any] | None = None,
    secondary_engine_kwargs: dict[str, Any] | None = None,
    candidate_baseline: str = "candidate",
    primary_baseline: str = "tesseract",
    metric: str = "cer",
    direction: str | None = None,
    threshold_start: float = -0.20,
    threshold_end: float = 0.20,
    threshold_step: float = 0.01,
    base_model: str = "tesseract-ocr",
    notes: str | None = None,
) -> tuple[CalibrationReport, Path]:
    experiment = _load_experiment_report(experiment_report)
    report_rows = _load_benchmark_rows(experiment.benchmark_report)
    resolved_direction = direction or metric_direction(metric)
    trials = _trial_thresholds(
        report_rows,
        candidate_baseline=candidate_baseline,
        primary_baseline=primary_baseline,
        metric=metric,
        direction=resolved_direction,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )
    if not trials:
        raise BenchmarkError("quality router calibration produced no valid threshold trials")
    selected = _select_best_trial(trials, resolved_direction)
    calibration = CalibrationReport(
        experiment_report=str(Path(experiment_report)),
        benchmark_report=experiment.benchmark_report,
        eval_manifest=experiment.eval_manifest,
        candidate_baseline=candidate_baseline,
        primary_baseline=primary_baseline,
        metric=metric,
        direction=resolved_direction,
        selected_threshold=selected.threshold,
        selected_trial=selected,
        trials=trials,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "calibration.md").write_text(_render_calibration_markdown(calibration), encoding="utf-8")
    model_path = write_model_card(
        out / "model-card.json",
        model_id=model_id,
        backend="quality_select_composite",
        base_model=base_model,
        backend_kwargs={
            "primary_engine": primary_engine,
            "secondary_engine": secondary_engine,
            "primary_engine_kwargs": primary_engine_kwargs or {},
            "secondary_engine_kwargs": secondary_engine_kwargs or {},
            "secondary_quality_margin": selected.threshold,
        },
        provenance={
            "calibration": {
                "experiment_report": {"path": str(Path(experiment_report)), "sha256": _maybe_sha256(Path(experiment_report))},
                "benchmark_report": {"path": experiment.benchmark_report, "sha256": _maybe_sha256(Path(experiment.benchmark_report))},
                "eval_manifest": {"path": experiment.eval_manifest, "sha256": _maybe_sha256(Path(experiment.eval_manifest))},
                "candidate_baseline": candidate_baseline,
                "primary_baseline": primary_baseline,
                "metric": metric,
                "direction": resolved_direction,
                "selected_threshold": selected.threshold,
                "threshold_start": threshold_start,
                "threshold_end": threshold_end,
                "threshold_step": threshold_step,
            }
        },
        metrics_report=out / "calibration.json",
        notes=notes,
    )
    return calibration, model_path


def calibrate_script_router(
    experiment_report: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    primary_engine: str,
    secondary_engine: str,
    primary_engine_kwargs: dict[str, Any] | None = None,
    secondary_engine_kwargs: dict[str, Any] | None = None,
    primary_baseline: str = "tesseract",
    secondary_baseline: str = "surya",
    metric: str = "cer",
    direction: str | None = None,
    script: str = "devanagari",
    threshold_start: float = 0.0,
    threshold_end: float = 0.6,
    threshold_step: float = 0.05,
    routing_granularity: str = "document",
    secondary_structure_backfill: bool = False,
    base_model: str = "tesseract+surya",
    notes: str | None = None,
) -> tuple[ScriptCalibrationReport, Path]:
    experiment = _load_experiment_report(experiment_report)
    report_rows = _load_benchmark_rows(experiment.benchmark_report)
    resolved_direction = direction or metric_direction(metric)
    trials = _trial_script_thresholds(
        experiment,
        report_rows,
        primary_baseline=primary_baseline,
        secondary_baseline=secondary_baseline,
        metric=metric,
        direction=resolved_direction,
        script=script,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )
    if not trials:
        raise BenchmarkError("script router calibration produced no valid threshold trials")
    selected = _select_best_script_trial(trials, resolved_direction)
    calibration = ScriptCalibrationReport(
        experiment_report=str(Path(experiment_report)),
        benchmark_report=experiment.benchmark_report,
        eval_manifest=experiment.eval_manifest,
        primary_baseline=primary_baseline,
        secondary_baseline=secondary_baseline,
        metric=metric,
        direction=resolved_direction,
        script=script,
        selected_threshold=selected.threshold,
        selected_trial=selected,
        trials=trials,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "calibration.md").write_text(_render_script_calibration_markdown(calibration), encoding="utf-8")
    model_path = write_model_card(
        out / "model-card.json",
        model_id=model_id,
        backend="script_select_composite",
        base_model=base_model,
        backend_kwargs={
            "primary_engine": primary_engine,
            "secondary_engine": secondary_engine,
            "primary_engine_kwargs": primary_engine_kwargs or {},
            "secondary_engine_kwargs": secondary_engine_kwargs or {},
            "script": script,
            "primary_script_threshold": selected.threshold,
            "routing_granularity": routing_granularity,
            "secondary_structure_backfill": secondary_structure_backfill,
        },
        provenance={
            "calibration": {
                "experiment_report": {"path": str(Path(experiment_report)), "sha256": _maybe_sha256(Path(experiment_report))},
                "benchmark_report": {"path": experiment.benchmark_report, "sha256": _maybe_sha256(Path(experiment.benchmark_report))},
                "eval_manifest": {"path": experiment.eval_manifest, "sha256": _maybe_sha256(Path(experiment.eval_manifest))},
                "primary_baseline": primary_baseline,
                "secondary_baseline": secondary_baseline,
                "metric": metric,
                "direction": resolved_direction,
                "script": script,
                "selected_threshold": selected.threshold,
                "threshold_start": threshold_start,
                "threshold_end": threshold_end,
                "threshold_step": threshold_step,
                "routing_granularity": routing_granularity,
                "secondary_structure_backfill": secondary_structure_backfill,
            }
        },
        metrics_report=out / "calibration.json",
        notes=notes,
    )
    return calibration, model_path


def _load_experiment_report(path: str | Path) -> ExperimentReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BenchmarkError("experiment report must be a JSON object")
    return ExperimentReport(
        experiment_id=str(raw.get("experiment_id") or ""),
        output_dir=str(raw.get("output_dir") or ""),
        eval_manifest=str(raw.get("eval_manifest") or ""),
        references_dir=str(raw["references_dir"]) if raw.get("references_dir") is not None else None,
        baselines=[str(item) for item in raw.get("baselines", [])],
        input_count=int(raw.get("input_count") or 0),
        datasets=[str(item) for item in raw.get("datasets", [])],
        slice_counts={str(key): int(value) for key, value in dict(raw.get("slice_counts") or {}).items()},
        benchmark_report=str(raw.get("benchmark_report") or ""),
        benchmark_status_counts={str(key): int(value) for key, value in dict(raw.get("benchmark_status_counts") or {}).items()},
        validation_report=str(raw["validation_report"]) if raw.get("validation_report") is not None else None,
        validation_status=str(raw["validation_status"]) if raw.get("validation_status") is not None else None,
        environment=dict(raw.get("environment") or {}),
    )


def _load_benchmark_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise BenchmarkError("benchmark report must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BenchmarkError(f"benchmark row {index} must be an object")
        rows.append(dict(item))
    return rows


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _script_ratio(text: str, script: str) -> float:
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    normalized = script.strip().lower()
    if normalized == "devanagari":
        matches = sum(1 for char in non_space if _DEVANAGARI_RE.match(char))
        return matches / len(non_space)
    if normalized in {"latin", "english"}:
        matches = sum(1 for char in non_space if _LATIN_RE.match(char))
        return matches / len(non_space)
    raise BenchmarkError(f"unsupported script calibration target: {script!r}")


def _trial_script_thresholds(
    experiment: ExperimentReport,
    rows: list[dict[str, Any]],
    *,
    primary_baseline: str,
    secondary_baseline: str,
    metric: str,
    direction: str,
    script: str,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> list[ScriptThresholdTrial]:
    if threshold_step <= 0:
        raise BenchmarkError("threshold_step must be positive")
    primary_rows = {sample_key(row): row for row in rows if row.get("baseline") == primary_baseline and row.get("status") == "ok"}
    secondary_rows = {sample_key(row): row for row in rows if row.get("baseline") == secondary_baseline and row.get("status") == "ok"}
    shared = sorted(set(primary_rows) & set(secondary_rows))
    if not shared:
        raise BenchmarkError(f"no paired rows for primary={primary_baseline} vs secondary={secondary_baseline} on metric={metric}")
    output_root = Path(experiment.output_dir) / "benchmark" / primary_baseline
    thresholds = _threshold_grid(threshold_start, threshold_end, threshold_step)
    ratios: dict[str, float] = {}
    for sample_id in shared:
        input_path = Path(primary_rows[sample_id]["input_path"])
        document_path = output_root / input_path.stem / "document.md"
        if not document_path.exists():
            raise BenchmarkError(f"primary benchmark output is missing document.md for {sample_id}: {document_path}")
        ratios[sample_id] = _script_ratio(document_path.read_text(encoding="utf-8"), script)
    trials: list[ScriptThresholdTrial] = []
    for threshold in thresholds:
        chosen_metrics: list[float] = []
        primary_metrics: list[float] = []
        secondary_picks = 0
        for sample_id in shared:
            primary_row = primary_rows[sample_id]
            secondary_row = secondary_rows[sample_id]
            metric_primary = (primary_row.get("metrics") or {}).get(metric)
            metric_secondary = (secondary_row.get("metrics") or {}).get(metric)
            if not _finite_number(metric_primary) or not _finite_number(metric_secondary):
                continue
            if ratios[sample_id] > threshold:
                chosen_metrics.append(float(metric_secondary))
                secondary_picks += 1
            else:
                chosen_metrics.append(float(metric_primary))
            primary_metrics.append(float(metric_primary))
        if not chosen_metrics:
            continue
        chosen_mean = sum(chosen_metrics) / len(chosen_metrics)
        primary_mean = sum(primary_metrics) / len(primary_metrics)
        absolute_improvement = _absolute_improvement(chosen_mean, primary_mean, direction)
        relative_improvement = _relative_improvement(absolute_improvement, primary_mean)
        trials.append(
            ScriptThresholdTrial(
                threshold=threshold,
                pairs=len(chosen_metrics),
                secondary_picks=secondary_picks,
                chosen_mean=chosen_mean,
                primary_mean=primary_mean,
                absolute_improvement=absolute_improvement,
                relative_improvement=relative_improvement,
            )
        )
    return trials


def _trial_thresholds(
    rows: list[dict[str, Any]],
    *,
    candidate_baseline: str,
    primary_baseline: str,
    metric: str,
    direction: str,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> list[ThresholdTrial]:
    if threshold_step <= 0:
        raise BenchmarkError("threshold_step must be positive")
    paired: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    candidate_rows = {sample_key(row): row for row in rows if row.get("baseline") == candidate_baseline and row.get("status") == "ok"}
    primary_rows = {sample_key(row): row for row in rows if row.get("baseline") == primary_baseline and row.get("status") == "ok"}
    for key in sorted(set(candidate_rows) & set(primary_rows)):
        candidate_row = candidate_rows[key]
        primary_row = primary_rows[key]
        if metric not in (candidate_row.get("metrics") or {}) or metric not in (primary_row.get("metrics") or {}):
            continue
        candidate_quality = (candidate_row.get("metrics") or {}).get("quality_score")
        primary_quality = (primary_row.get("metrics") or {}).get("quality_score")
        if not _finite_number(candidate_quality) or not _finite_number(primary_quality):
            continue
        metric_candidate = (candidate_row.get("metrics") or {}).get(metric)
        metric_primary = (primary_row.get("metrics") or {}).get(metric)
        if not _finite_number(metric_candidate) or not _finite_number(metric_primary):
            continue
        paired[key] = (candidate_row, primary_row)
    if not paired:
        raise BenchmarkError(f"no paired rows for candidate={candidate_baseline} vs primary={primary_baseline} on metric={metric}")

    thresholds = _threshold_grid(threshold_start, threshold_end, threshold_step)
    trials: list[ThresholdTrial] = []
    for threshold in thresholds:
        chosen_metrics: list[float] = []
        primary_metrics: list[float] = []
        candidate_picks = 0
        for candidate_row, primary_row in paired.values():
            candidate_quality = float(candidate_row["metrics"]["quality_score"])
            primary_quality = float(primary_row["metrics"]["quality_score"])
            metric_candidate = float(candidate_row["metrics"][metric])
            metric_primary = float(primary_row["metrics"][metric])
            if candidate_quality - primary_quality > threshold:
                chosen_metrics.append(metric_candidate)
                candidate_picks += 1
            else:
                chosen_metrics.append(metric_primary)
            primary_metrics.append(metric_primary)
        chosen_mean = sum(chosen_metrics) / len(chosen_metrics)
        primary_mean = sum(primary_metrics) / len(primary_metrics)
        absolute_improvement = _absolute_improvement(chosen_mean, primary_mean, direction)
        relative_improvement = _relative_improvement(absolute_improvement, primary_mean)
        trials.append(
            ThresholdTrial(
                threshold=threshold,
                pairs=len(chosen_metrics),
                candidate_picks=candidate_picks,
                chosen_mean=chosen_mean,
                primary_mean=primary_mean,
                absolute_improvement=absolute_improvement,
                relative_improvement=relative_improvement,
            )
        )
    return trials


def _threshold_grid(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    limit = end + (step / 2)
    while current <= limit:
        values.append(round(current, 10))
        current += step
    return values


def _select_best_trial(trials: list[ThresholdTrial], direction: str) -> ThresholdTrial:
    if direction == "lower":
        return min(trials, key=lambda item: (item.chosen_mean, item.candidate_picks, -item.threshold))
    return max(trials, key=lambda item: (item.chosen_mean, -item.candidate_picks, item.threshold))


def _select_best_script_trial(trials: list[ScriptThresholdTrial], direction: str) -> ScriptThresholdTrial:
    if direction == "lower":
        return min(trials, key=lambda item: (item.chosen_mean, item.secondary_picks, item.threshold))
    return max(trials, key=lambda item: (item.chosen_mean, -item.secondary_picks, -item.threshold))


def _absolute_improvement(candidate_value: float, baseline_value: float, direction: str) -> float:
    if direction == "lower":
        return baseline_value - candidate_value
    return candidate_value - baseline_value


def _relative_improvement(absolute_improvement: float, baseline_mean: float) -> float:
    denominator = abs(baseline_mean)
    if denominator == 0:
        return math.inf if absolute_improvement > 0 else 0.0
    return absolute_improvement / denominator


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _maybe_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() and path.is_file() else None


def _render_calibration_markdown(report: CalibrationReport) -> str:
    lines = [
        "# Quality Router Calibration",
        "",
        f"Experiment report: `{report.experiment_report}`",
        f"Benchmark report: `{report.benchmark_report}`",
        f"Eval manifest: `{report.eval_manifest}`",
        f"Candidate baseline: `{report.candidate_baseline}`",
        f"Primary baseline: `{report.primary_baseline}`",
        f"Metric: `{report.metric}`",
        f"Direction: `{report.direction}`",
        f"Selected threshold: `{report.selected_threshold:.6f}`",
        "",
        "| threshold | pairs | candidate_picks | chosen_mean | primary_mean | improvement | relative |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in report.trials:
        lines.append(
            f"| {trial.threshold:.6f} | {trial.pairs} | {trial.candidate_picks} | {trial.chosen_mean:.6f} | {trial.primary_mean:.6f} | {trial.absolute_improvement:.6f} | {trial.relative_improvement:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _render_script_calibration_markdown(report: ScriptCalibrationReport) -> str:
    lines = [
        "# Script Router Calibration",
        "",
        f"Experiment report: `{report.experiment_report}`",
        f"Benchmark report: `{report.benchmark_report}`",
        f"Eval manifest: `{report.eval_manifest}`",
        f"Primary baseline: `{report.primary_baseline}`",
        f"Secondary baseline: `{report.secondary_baseline}`",
        f"Metric: `{report.metric}`",
        f"Direction: `{report.direction}`",
        f"Script: `{report.script}`",
        f"Selected threshold: `{report.selected_threshold:.6f}`",
        "",
        "| threshold | pairs | secondary_picks | chosen_mean | primary_mean | improvement | relative |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in report.trials:
        lines.append(
            f"| {trial.threshold:.6f} | {trial.pairs} | {trial.secondary_picks} | {trial.chosen_mean:.6f} | {trial.primary_mean:.6f} | {trial.absolute_improvement:.6f} | {trial.relative_improvement:.6f} |"
        )
    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class EnsembleTrial:
    biases: dict[str, float]
    pairs: int
    alternative_picks: int
    chosen_mean: float
    primary_mean: float
    absolute_improvement: float
    relative_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "biases": self.biases,
            "pairs": self.pairs,
            "alternative_picks": self.alternative_picks,
            "chosen_mean": self.chosen_mean,
            "primary_mean": self.primary_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
        }


@dataclass(slots=True)
class EnsembleCalibrationReport:
    primary_experiment_report: str
    variant_experiments: dict[str, str]
    benchmark_reports: dict[str, str]
    eval_manifest: str
    primary_label: str
    metric: str
    direction: str
    selected_biases: dict[str, float]
    selected_trial: EnsembleTrial
    trials: list[EnsembleTrial] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_experiment_report": self.primary_experiment_report,
            "variant_experiments": self.variant_experiments,
            "benchmark_reports": self.benchmark_reports,
            "eval_manifest": self.eval_manifest,
            "primary_label": self.primary_label,
            "metric": self.metric,
            "direction": self.direction,
            "selected_biases": self.selected_biases,
            "selected_trial": self.selected_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def calibrate_tesseract_psm_ensemble(
    primary_experiment_report: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    variant_experiment_reports: dict[str, str | Path],
    metric: str = "cer",
    direction: str | None = None,
    bias_start: float = -0.12,
    bias_end: float = 0.12,
    bias_step: float = 0.02,
    language: str = "nep+eng",
    selection_margin: float = 0.0,
    notes: str | None = None,
) -> tuple[EnsembleCalibrationReport, Path]:
    if not variant_experiment_reports:
        raise BenchmarkError("tesseract ensemble calibration requires at least one variant experiment report")
    primary_experiment = _load_experiment_report(primary_experiment_report)
    resolved_direction = direction or metric_direction(metric)
    primary_rows = _load_variant_rows(primary_experiment.benchmark_report, baseline="tesseract")
    variant_rows: dict[str, dict[str, dict[str, Any]]] = {}
    variant_reports: dict[str, str] = {}
    for label, path in sorted(variant_experiment_reports.items()):
        experiment = _load_experiment_report(path)
        if experiment.eval_manifest != primary_experiment.eval_manifest:
            raise BenchmarkError(f"variant experiment {path} does not use the same eval manifest as the primary experiment")
        variant_rows[label] = _load_variant_rows(experiment.benchmark_report, baseline="candidate")
        variant_reports[label] = experiment.benchmark_report
    shared = set(primary_rows)
    for values in variant_rows.values():
        shared &= set(values)
    if not shared:
        raise BenchmarkError("no shared samples across primary and variant experiment reports")
    shared_ids = sorted(shared)
    bias_values = _threshold_grid(bias_start, bias_end, bias_step)
    trials: list[EnsembleTrial] = []
    labels = sorted(variant_rows)
    import itertools

    for combo in itertools.product(bias_values, repeat=len(labels)):
        biases = dict(zip(labels, combo, strict=True))
        chosen_metrics: list[float] = []
        primary_metrics: list[float] = []
        alternative_picks = 0
        for sample_id in shared_ids:
            primary = primary_rows[sample_id]
            scored = [(primary["metrics"].get("quality_score", -1.0), primary, None)]
            for label in labels:
                row = variant_rows[label][sample_id]
                scored.append((float(row["metrics"].get("quality_score", -1.0)) + biases[label], row, label))
            selected_quality, selected_row, selected_label = max(scored, key=lambda item: item[0])
            if selected_label is not None and selected_quality <= float(primary["metrics"].get("quality_score", -1.0)) + selection_margin:
                selected_row = primary
                selected_label = None
            chosen_metrics.append(float(selected_row["metrics"][metric]))
            primary_metrics.append(float(primary["metrics"][metric]))
            if selected_label is not None:
                alternative_picks += 1
        chosen_mean = sum(chosen_metrics) / len(chosen_metrics)
        primary_mean = sum(primary_metrics) / len(primary_metrics)
        absolute_improvement = _absolute_improvement(chosen_mean, primary_mean, resolved_direction)
        relative_improvement = _relative_improvement(absolute_improvement, primary_mean)
        trials.append(
            EnsembleTrial(
                biases=biases,
                pairs=len(shared_ids),
                alternative_picks=alternative_picks,
                chosen_mean=chosen_mean,
                primary_mean=primary_mean,
                absolute_improvement=absolute_improvement,
                relative_improvement=relative_improvement,
            )
        )
    if not trials:
        raise BenchmarkError("tesseract ensemble calibration produced no trials")
    selected = _select_best_ensemble_trial(trials, resolved_direction)
    report = EnsembleCalibrationReport(
        primary_experiment_report=str(Path(primary_experiment_report)),
        variant_experiments={label: str(path) for label, path in variant_experiment_reports.items()},
        benchmark_reports={"primary": primary_experiment.benchmark_report, **variant_reports},
        eval_manifest=primary_experiment.eval_manifest,
        primary_label="psm6",
        metric=metric,
        direction=resolved_direction,
        selected_biases=selected.biases,
        selected_trial=selected,
        trials=trials,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "calibration.md").write_text(_render_ensemble_calibration_markdown(report), encoding="utf-8")
    alternatives = [
        {"label": label, "engine": "tesseract", "engine_kwargs": {"language": language, "psm": int(label.removeprefix("psm"))}, "quality_bias": bias}
        for label, bias in sorted(selected.biases.items())
    ]
    model_path = write_model_card(
        out / "model-card.json",
        model_id=model_id,
        backend="quality_ranked_ensemble",
        base_model="tesseract-ocr",
        backend_kwargs={
            "primary_engine": "tesseract",
            "primary_engine_kwargs": {"language": language, "psm": 6},
            "alternative_engines": alternatives,
            "selection_margin": selection_margin,
        },
        provenance={
            "calibration": {
                "eval_manifest": {"path": primary_experiment.eval_manifest, "sha256": _maybe_sha256(Path(primary_experiment.eval_manifest))},
                "primary_experiment_report": {"path": str(Path(primary_experiment_report)), "sha256": _maybe_sha256(Path(primary_experiment_report))},
                "variant_experiments": {
                    label: {"path": str(path), "sha256": _maybe_sha256(Path(path))}
                    for label, path in variant_experiment_reports.items()
                },
                "metric": metric,
                "direction": resolved_direction,
                "selected_biases": selected.biases,
                "selection_margin": selection_margin,
                "bias_start": bias_start,
                "bias_end": bias_end,
                "bias_step": bias_step,
            }
        },
        metrics_report=out / "calibration.json",
        notes=notes,
    )
    return report, model_path


def _load_variant_rows(path: str | Path, *, baseline: str) -> dict[str, dict[str, Any]]:
    rows = _load_benchmark_rows(path)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("baseline") != baseline or row.get("status") != "ok":
            continue
        selected[sample_key(row)] = row
    if not selected:
        raise BenchmarkError(f"no benchmark rows found for baseline={baseline} in {path}")
    return selected


def _select_best_ensemble_trial(trials: list[EnsembleTrial], direction: str) -> EnsembleTrial:
    if direction == "lower":
        return min(trials, key=lambda item: (item.chosen_mean, item.alternative_picks, sorted(item.biases.items())))
    return max(trials, key=lambda item: (item.chosen_mean, -item.alternative_picks, sorted(item.biases.items())))


def _render_ensemble_calibration_markdown(report: EnsembleCalibrationReport) -> str:
    labels = sorted(report.selected_biases)
    lines = [
        "# Quality Ranked Ensemble Calibration",
        "",
        f"Primary experiment report: `{report.primary_experiment_report}`",
        f"Eval manifest: `{report.eval_manifest}`",
        f"Metric: `{report.metric}`",
        f"Direction: `{report.direction}`",
        f"Selected biases: `{report.selected_biases}`",
        "",
        "| biases | pairs | alternative_picks | chosen_mean | primary_mean | improvement | relative |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in report.trials:
        bias_text = ", ".join(f"{label}={trial.biases[label]:.2f}" for label in labels)
        lines.append(
            f"| {bias_text} | {trial.pairs} | {trial.alternative_picks} | {trial.chosen_mean:.6f} | {trial.primary_mean:.6f} | {trial.absolute_improvement:.6f} | {trial.relative_improvement:.6f} |"
        )
    return "\n".join(lines) + "\n"
