"""Benchmark runner for OCR engines and structured output."""

from __future__ import annotations

import json
import multiprocessing
import queue
import signal
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on some platforms.
    resource = None  # type: ignore[assignment]

from .engines import OcrEngine, create_engine
from .errors import BenchmarkError, EngineUnavailableError, ParseError
from .external_outputs import document_from_markdown
from .manifest import ManifestEntry, load_manifest, sha256_file
from .pipeline import export_document, parse_document
from .quality import evaluate_document_quality
from .references import load_reference, score_document
from .schemas import Document
from .structure import build_document
from .validation import metric_direction, sample_key


@dataclass(slots=True)
class BaselineOutput:
    text: str
    document: Document | None = None


@dataclass(slots=True)
class BenchmarkResult:
    baseline: str
    input_path: str
    status: str
    latency_seconds: float
    sample_id: str | None = None
    slices: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    resource_metrics: dict[str, float] = field(default_factory=dict)
    output_artifacts: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "input_path": self.input_path,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "sample_id": self.sample_id,
            "slices": self.slices,
            "metrics": self.metrics,
            "resource_metrics": self.resource_metrics,
            "output_artifacts": self.output_artifacts,
            "error": self.error,
        }


@dataclass(slots=True)
class PairedMetricSummary:
    baseline: str
    slice_name: str
    metric: str
    direction: str
    pairs: int
    candidate_mean: float
    baseline_mean: float
    absolute_improvement: float
    relative_improvement: float
    win_count: int
    loss_count: int
    tie_count: int
    win_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "slice": self.slice_name,
            "metric": self.metric,
            "direction": self.direction,
            "pairs": self.pairs,
            "candidate_mean": self.candidate_mean,
            "baseline_mean": self.baseline_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "win_rate": self.win_rate,
        }


@dataclass(slots=True)
class BenchmarkSummary:
    report_path: str
    candidate: str
    baselines: list[str]
    metrics: list[str]
    aggregate_metrics: dict[str, dict[str, dict[str, float | int]]]
    paired_metrics: list[PairedMetricSummary]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_path": self.report_path,
            "candidate": self.candidate,
            "baselines": self.baselines,
            "metrics": self.metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "paired_metrics": [item.to_dict() for item in self.paired_metrics],
        }


def run_benchmark(
    inputs: list[str | Path],
    output_dir: str | Path,
    *,
    baselines: list[str],
    references_dir: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    candidate_model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    fallback_min_quality_score: float | None = None,
    resume_existing: bool = False,
    sample_timeout_seconds: int | None = None,
    capture_gpu_metrics: bool = False,
) -> list[BenchmarkResult]:
    if not inputs:
        raise BenchmarkError("benchmark requires at least one input")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference_root = Path(references_dir) if references_dir else None
    eval_entries = _load_eval_entries(eval_manifest)
    results: list[BenchmarkResult] = []
    for baseline in baselines:
        runner: Callable[[Path, Path], BaselineOutput] | None = None
        for input_item in inputs:
            input_path = Path(input_item)
            eval_entry = _entry_for_input(input_path, eval_entries)
            start = time.perf_counter()
            resource_before = _resource_snapshot(capture_gpu_metrics=capture_gpu_metrics)
            output_dir_for_input = out / baseline / input_path.stem
            document_path = output_dir_for_input / "document.json"
            try:
                explicit_reference_path = None
                if eval_entry is not None and isinstance(eval_entry.metadata.get("reference_path"), str):
                    explicit_reference_path = str(eval_entry.metadata["reference_path"])
                if resume_existing and document_path.exists():
                    payload = json.loads(document_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise BenchmarkError(f"document JSON must be an object: {document_path}")
                    document = Document.from_dict(payload)
                    elapsed = 0.0
                else:
                    if sample_timeout_seconds is None:
                        if runner is None:
                            runner = _make_baseline_runner(
                                baseline,
                                candidate_model_config=candidate_model_config,
                                fallback_engine=fallback_engine,
                                fallback_model_config=fallback_model_config,
                                low_confidence_threshold=low_confidence_threshold,
                                fallback_min_quality_score=fallback_min_quality_score,
                            )
                        output = runner(input_path, output_dir_for_input)
                    else:
                        output = _call_with_timeout(
                            lambda: _run_baseline(
                                baseline,
                                input_path,
                                output_dir_for_input,
                                candidate_model_config=candidate_model_config,
                                fallback_engine=fallback_engine,
                                fallback_model_config=fallback_model_config,
                                low_confidence_threshold=low_confidence_threshold,
                                fallback_min_quality_score=fallback_min_quality_score,
                            ),
                            sample_timeout_seconds,
                            description=f"{baseline} on {input_path}",
                        )
                    elapsed = time.perf_counter() - start
                    if output.document is None:
                        raise BenchmarkError(f"baseline {baseline!r} did not return a document for {input_path}")
                    document = output.document
                metrics = score_document(document, document.text, load_reference(input_path, reference_root, explicit_path=explicit_reference_path))
                metrics.update(_quality_metrics(document))
                results.append(
                    BenchmarkResult(
                        baseline=baseline,
                        input_path=str(input_path),
                        status="ok",
                        latency_seconds=elapsed,
                        sample_id=eval_entry.sample_id if eval_entry else None,
                        slices=_slices_for_entry(eval_entry),
                        metrics=metrics,
                        resource_metrics=_resource_delta(resource_before, capture_gpu_metrics=capture_gpu_metrics),
                        output_artifacts=_output_artifacts(output_dir_for_input),
                    )
                )
                _write_reports(results, out)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                results.append(
                    BenchmarkResult(
                        baseline=baseline,
                        input_path=str(input_path),
                        status="error",
                        latency_seconds=elapsed,
                        sample_id=eval_entry.sample_id if eval_entry else None,
                        slices=_slices_for_entry(eval_entry),
                        resource_metrics=_resource_delta(resource_before, capture_gpu_metrics=capture_gpu_metrics),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                _write_reports(results, out)
    _write_reports(results, out)
    return results


def _call_with_timeout(
    func: Callable[[], BaselineOutput],
    timeout_seconds: int | None,
    *,
    description: str,
) -> BaselineOutput:
    if timeout_seconds is None:
        return func()
    if timeout_seconds < 1:
        raise BenchmarkError("sample_timeout_seconds must be positive when provided")
    if "fork" in multiprocessing.get_all_start_methods():
        return _call_with_process_timeout(func, timeout_seconds, description=description)
    if not hasattr(signal, "SIGALRM"):
        return func()

    def _timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"timed out after {timeout_seconds}s while running {description}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _call_with_process_timeout(
    func: Callable[[], BaselineOutput],
    timeout_seconds: int,
    *,
    description: str,
) -> BaselineOutput:
    context = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue[dict[str, object]] = context.Queue(maxsize=1)
    process = context.Process(target=_run_timeout_child, args=(func, result_queue))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(5)
        raise TimeoutError(f"timed out after {timeout_seconds}s while running {description}")
    try:
        payload = result_queue.get_nowait()
    except queue.Empty as exc:
        exit_code = process.exitcode
        raise BenchmarkError(f"timeout worker exited without a result while running {description}; exit_code={exit_code}") from exc
    status = payload.get("status")
    if status == "ok":
        output = payload.get("output")
        if not isinstance(output, BaselineOutput):
            raise BenchmarkError(f"timeout worker returned invalid output while running {description}")
        return output
    if status == "error":
        error_type = str(payload.get("error_type") or "Exception")
        message = str(payload.get("message") or "")
        child_traceback = str(payload.get("traceback") or "")
        raise BenchmarkError(f"{error_type}: {message}\nChild traceback:\n{child_traceback}")
    raise BenchmarkError(f"timeout worker returned invalid status while running {description}: {status!r}")


def _run_timeout_child(func: Callable[[], BaselineOutput], result_queue: multiprocessing.Queue[dict[str, object]]) -> None:
    try:
        result_queue.put({"status": "ok", "output": func()})
    except BaseException as exc:  # pragma: no cover - exercised through parent process.
        result_queue.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def rescore_benchmark_report(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    references_dir: str | Path | None = None,
    eval_manifest: str | Path | None = None,
) -> list[BenchmarkResult]:
    source_report = Path(report_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_benchmark_rows(source_report)
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    reference_root = Path(references_dir) if references_dir else None
    eval_entries = _load_eval_entries(eval_manifest)
    benchmark_dir = source_report.parent
    rescored: list[BenchmarkResult] = []
    for row in rows:
        baseline = str(row.get("baseline") or "")
        input_path = Path(str(row.get("input_path") or ""))
        eval_entry = _entry_for_input(input_path, eval_entries)
        sample_id = str(row.get("sample_id") or (eval_entry.sample_id if eval_entry else "")) or None
        slices = [str(item) for item in row.get("slices", [])] if isinstance(row.get("slices"), list) else _slices_for_entry(eval_entry)
        latency = float(row.get("latency_seconds") or 0.0)
        if row.get("status") != "ok":
            rescored.append(
                BenchmarkResult(
                    baseline=baseline,
                    input_path=str(input_path),
                    status=str(row.get("status") or "error"),
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=str(row.get("error") or ""),
                )
            )
            continue
        document_path = benchmark_dir / baseline / input_path.stem / "document.json"
        if not document_path.exists():
            rescored.append(
                BenchmarkResult(
                    baseline=baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=f"missing document output for rescore: {document_path}",
                )
            )
            continue
        try:
            _copy_existing_output_for_rescore(document_path.parent, out / baseline / input_path.stem)
            payload = json.loads(document_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise BenchmarkError(f"document JSON must be an object: {document_path}")
            document = Document.from_dict(payload)
            explicit_reference_path = None
            if eval_entry is not None and isinstance(eval_entry.metadata.get("reference_path"), str):
                explicit_reference_path = str(eval_entry.metadata["reference_path"])
            metrics = score_document(document, document.text, load_reference(input_path, reference_root, explicit_path=explicit_reference_path))
            metrics.update(_quality_metrics(document))
            rescored.append(
                BenchmarkResult(
                    baseline=baseline,
                    input_path=str(input_path),
                    status="ok",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    metrics=metrics,
                    output_artifacts=_output_artifacts(out / baseline / input_path.stem),
                )
            )
        except Exception as exc:
            rescored.append(
                BenchmarkResult(
                    baseline=baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    _write_reports(rescored, out)
    return rescored


def derive_table_postprocess_benchmark(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    source_baseline: str,
    candidate_baseline: str = "candidate",
    references_dir: str | Path | None = None,
    eval_manifest: str | Path | None = None,
) -> list[BenchmarkResult]:
    source_report = Path(report_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_benchmark_rows(source_report)
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    reference_root = Path(references_dir) if references_dir else None
    eval_entries = _load_eval_entries(eval_manifest)
    benchmark_dir = source_report.parent
    derived: list[BenchmarkResult] = []
    for row in rows:
        baseline = str(row.get("baseline") or "")
        if baseline != source_baseline:
            continue
        input_path = Path(str(row.get("input_path") or ""))
        eval_entry = _entry_for_input(input_path, eval_entries)
        sample_id = str(row.get("sample_id") or (eval_entry.sample_id if eval_entry else "")) or None
        slices = [str(item) for item in row.get("slices", [])] if isinstance(row.get("slices"), list) else _slices_for_entry(eval_entry)
        latency = float(row.get("latency_seconds") or 0.0)
        source_output_dir = benchmark_dir / source_baseline / input_path.stem
        source_document_path = source_output_dir / "document.json"
        target_source_dir = out / source_baseline / input_path.stem
        target_candidate_dir = out / candidate_baseline / input_path.stem
        if row.get("status") != "ok":
            derived.append(
                BenchmarkResult(
                    baseline=source_baseline,
                    input_path=str(input_path),
                    status=str(row.get("status") or "error"),
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=str(row.get("error") or ""),
                )
            )
            derived.append(
                BenchmarkResult(
                    baseline=candidate_baseline,
                    input_path=str(input_path),
                    status=str(row.get("status") or "error"),
                    latency_seconds=0.0,
                    sample_id=sample_id,
                    slices=slices,
                    error=f"source baseline {source_baseline} failed",
                )
            )
            continue
        if not source_document_path.exists():
            error = f"missing source document output: {source_document_path}"
            derived.append(
                BenchmarkResult(
                    baseline=source_baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=error,
                )
            )
            derived.append(
                BenchmarkResult(
                    baseline=candidate_baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=0.0,
                    sample_id=sample_id,
                    slices=slices,
                    error=error,
                )
            )
            continue
        try:
            _copy_existing_output_for_rescore(source_output_dir, target_source_dir)
            _copy_existing_output_for_rescore(source_output_dir, target_candidate_dir)
            payload = json.loads(source_document_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise BenchmarkError(f"document JSON must be an object: {source_document_path}")
            source_document = Document.from_dict(payload)
            candidate_document, _stats = _derive_table_postprocessed_document(source_document, source_baseline=source_baseline)
            export_document(candidate_document, target_candidate_dir, input_path)
            explicit_reference_path = None
            if eval_entry is not None and isinstance(eval_entry.metadata.get("reference_path"), str):
                explicit_reference_path = str(eval_entry.metadata["reference_path"])
            reference = load_reference(input_path, reference_root, explicit_path=explicit_reference_path)
            source_metrics = score_document(source_document, source_document.text, reference)
            source_metrics.update(_quality_metrics(source_document))
            candidate_metrics = score_document(candidate_document, candidate_document.text, reference)
            candidate_metrics.update(_quality_metrics(candidate_document))
            derived.append(
                BenchmarkResult(
                    baseline=source_baseline,
                    input_path=str(input_path),
                    status="ok",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    metrics=source_metrics,
                    output_artifacts=_output_artifacts(target_source_dir),
                )
            )
            derived.append(
                BenchmarkResult(
                    baseline=candidate_baseline,
                    input_path=str(input_path),
                    status="ok",
                    latency_seconds=0.0,
                    sample_id=sample_id,
                    slices=slices,
                    metrics=candidate_metrics,
                    output_artifacts=_output_artifacts(target_candidate_dir),
                )
            )
        except Exception as exc:
            derived.append(
                BenchmarkResult(
                    baseline=source_baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=latency,
                    sample_id=sample_id,
                    slices=slices,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            derived.append(
                BenchmarkResult(
                    baseline=candidate_baseline,
                    input_path=str(input_path),
                    status="error",
                    latency_seconds=0.0,
                    sample_id=sample_id,
                    slices=slices,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    if not derived:
        raise BenchmarkError(f"benchmark report has no rows for source baseline {source_baseline!r}")
    _write_reports(derived, out)
    return derived


def _derive_table_postprocessed_document(document: Document, *, source_baseline: str) -> tuple[Document, dict[str, object]]:
    from .tables import infer_repeated_header_colspans, normalize_table_grid_structure

    tables = []
    inferred_colspans = 0
    dropped_columns = 0
    merged_rows = 0
    changed_tables = 0
    for table in document.tables:
        current = table
        current, inferred = infer_repeated_header_colspans(current)
        current, changed = normalize_table_grid_structure(current)
        if inferred or changed:
            changed_tables += 1
        inferred_colspans += inferred
        dropped_columns += int(current.metadata.get("normalized_table_grid_dropped_columns") or 0)
        merged_rows += int(current.metadata.get("normalized_table_grid_merged_rows") or 0)
        tables.append(current)
    metadata = dict(document.metadata)
    metadata["derived_table_postprocess"] = {
        "source_baseline": source_baseline,
        "inferred_table_header_colspans": inferred_colspans,
        "normalized_table_grid_dropped_columns": dropped_columns,
        "normalized_table_grid_merged_rows": merged_rows,
        "changed_tables": changed_tables,
    }
    derived = Document(
        source_path=document.source_path,
        pages=document.pages,
        tables=tables,
        figures=document.figures,
        metadata=metadata,
    )
    return derived, metadata["derived_table_postprocess"]


def _copy_existing_output_for_rescore(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def summarize_benchmark_report(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    candidate: str = "candidate",
    baselines: list[str] | None = None,
    metrics: list[str] | None = None,
) -> BenchmarkSummary:
    rows = _load_benchmark_rows(report_path)
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    aggregate_metrics = _aggregate_metrics(rows)
    available_baselines = sorted({str(row.get("baseline") or "") for row in rows if str(row.get("baseline") or "")})
    if candidate not in available_baselines:
        raise BenchmarkError(f"candidate baseline {candidate!r} is not present in benchmark report")
    compare_baselines = baselines or [name for name in available_baselines if name != candidate]
    missing = [name for name in compare_baselines if name not in available_baselines]
    if missing:
        raise BenchmarkError(f"requested baselines missing from benchmark report: {', '.join(missing)}")
    paired_metrics: list[PairedMetricSummary] = []
    selected_metrics = metrics or ["cer"]
    slices = _distinct_slices(rows)
    for baseline in compare_baselines:
        for metric in selected_metrics:
            direction = metric_direction(metric)
            for slice_name in slices:
                summary = _summarize_paired_metric(rows, candidate, baseline, metric, direction, slice_name)
                if summary is not None:
                    paired_metrics.append(summary)
    summary = BenchmarkSummary(
        report_path=str(Path(report_path)),
        candidate=candidate,
        baselines=compare_baselines,
        metrics=selected_metrics,
        aggregate_metrics=aggregate_metrics,
        paired_metrics=paired_metrics,
    )
    _write_benchmark_summary(summary, Path(output_dir))
    return summary


def _run_baseline(
    baseline: str,
    input_path: Path,
    output_dir: Path,
    *,
    candidate_model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    fallback_min_quality_score: float | None = None,
) -> BaselineOutput:
    return _make_baseline_runner(
        baseline,
        candidate_model_config=candidate_model_config,
        fallback_engine=fallback_engine,
        fallback_model_config=fallback_model_config,
        low_confidence_threshold=low_confidence_threshold,
        fallback_min_quality_score=fallback_min_quality_score,
    )(input_path, output_dir)


def _make_baseline_runner(
    baseline: str,
    *,
    candidate_model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    fallback_min_quality_score: float | None = None,
) -> Callable[[Path, Path], BaselineOutput]:
    name = baseline.lower()
    if name in {"ours", "ocrtech", "candidate"}:
        if fallback_engine or fallback_model_config:
            engine_name = "candidate" if candidate_model_config else "auto"
            return lambda input_path, output_dir: _run_candidate_with_fallback(
                input_path,
                output_dir,
                engine_name=engine_name,
                candidate_model_config=candidate_model_config,
                fallback_engine=fallback_engine,
                fallback_model_config=fallback_model_config,
                low_confidence_threshold=low_confidence_threshold,
                fallback_min_quality_score=fallback_min_quality_score,
            )
        engine_name = "candidate" if candidate_model_config else "auto"
        engine_kwargs = {"model_config": candidate_model_config} if engine_name == "candidate" else {}
        engine = create_engine(engine_name, **engine_kwargs)
        return lambda input_path, output_dir: _run_reusable_engine(engine, input_path, output_dir)
    if name in {"tesseract", "stock-paddle", "paddleocr", "paddle", "surya"}:
        engine_name = "paddleocr" if name in {"stock-paddle", "paddleocr", "paddle"} else name
        engine = create_engine(engine_name)
        return lambda input_path, output_dir: _run_reusable_engine(engine, input_path, output_dir)
    if name in {"glm-ocr", "paddleocr-vl"}:
        return lambda input_path, output_dir: _run_external_command(name, input_path, output_dir)
    raise BenchmarkError(f"Unknown baseline: {baseline}")


def _run_candidate_with_fallback(
    input_path: Path,
    output_dir: Path,
    *,
    engine_name: str,
    candidate_model_config: str | Path | None,
    fallback_engine: str | None,
    fallback_model_config: str | Path | None,
    low_confidence_threshold: float,
    fallback_min_quality_score: float | None,
) -> BaselineOutput:
    document = parse_document(
        input_path,
        output_dir,
        engine_name=engine_name,
        model_config=candidate_model_config,
        fallback_engine=fallback_engine,
        fallback_model_config=fallback_model_config,
        low_confidence_threshold=low_confidence_threshold,
        fallback_min_quality_score=fallback_min_quality_score,
    )
    return BaselineOutput(text=document.text, document=document)


def _run_reusable_engine(engine: OcrEngine, input_path: Path, output_dir: Path) -> BaselineOutput:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_output = engine.recognize(input_path)
    document = build_document(str(input_path), engine_output)
    quality = evaluate_document_quality(document)
    document.metadata["quality"] = quality.to_dict()
    export_document(document, output_dir, input_path)
    return BaselineOutput(text=document.text, document=document)


def _run_external_command(name: str, input_path: Path, output_dir: Path) -> BaselineOutput:
    env_key = "OCRTECH_GLM_OCR_CMD" if name == "glm-ocr" else "OCRTECH_PADDLEOCR_VL_CMD"
    template = os.environ.get(env_key)
    if not template:
        raise EngineUnavailableError(f"{name} requires {env_key} command template")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [part.format(input=str(input_path), out=str(output_dir)) for part in shlex.split(template)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ParseError(f"{name} failed: {completed.stderr.strip()}")
    output_text = completed.stdout.strip()
    if output_text:
        (output_dir / "stdout.txt").write_text(output_text + "\n", encoding="utf-8")
    if completed.stderr.strip():
        (output_dir / "stderr.txt").write_text(completed.stderr.strip() + "\n", encoding="utf-8")
    text_file = output_dir / "document.md"
    if text_file.exists():
        output_text = text_file.read_text(encoding="utf-8")
    elif output_text:
        text_file.write_text(output_text + ("\n" if not output_text.endswith("\n") else ""), encoding="utf-8")
    document_file = output_dir / "document.json"
    document = None
    if document_file.exists():
        payload = json.loads(document_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            document = Document.from_dict(payload)
            if not output_text:
                output_text = document.text
    elif text_file.exists():
        document = document_from_markdown(input_path, text_file.read_text(encoding="utf-8"), metadata={"engine": name, "command": template})
        quality = evaluate_document_quality(document)
        document.metadata["quality"] = quality.to_dict()
        export_document(document, output_dir, input_path)
        output_text = (output_dir / "document.md").read_text(encoding="utf-8")
    return BaselineOutput(text=output_text, document=document)


def _quality_metrics(document: Document) -> dict[str, float]:
    quality_payload = document.metadata.get("quality") if document.metadata else None
    if isinstance(quality_payload, dict):
        score = quality_payload.get("quality_score")
        metrics = quality_payload.get("metrics")
    else:
        quality = evaluate_document_quality(document)
        score = quality.quality_score
        metrics = quality.metrics
    out: dict[str, float] = {}
    if isinstance(score, int | float):
        out["quality_score"] = float(score)
    if isinstance(metrics, dict):
        line_count = metrics.get("line_count")
        low = metrics.get("low_confidence_line_count")
        missing = metrics.get("missing_confidence_line_count")
        if isinstance(line_count, int | float) and line_count:
            if isinstance(low, int | float):
                out["low_confidence_line_rate"] = float(low) / float(line_count)
            if isinstance(missing, int | float):
                out["missing_confidence_line_rate"] = float(missing) / float(line_count)
    fallback = document.metadata.get("fallback") if document.metadata else None
    if isinstance(fallback, dict):
        triggered = bool(fallback.get("triggered"))
        out["fallback_triggered"] = 1.0 if triggered else 0.0
        out["fallback_succeeded"] = 1.0 if triggered and fallback.get("outcome") == "success" else 0.0
        out["fallback_failed"] = 1.0 if triggered and fallback.get("outcome") == "error" else 0.0
        primary_confidence = fallback.get("primary_average_confidence")
        threshold = fallback.get("threshold")
        min_quality_score = fallback.get("min_quality_score")
        primary_quality_score = fallback.get("primary_quality_score")
        if isinstance(primary_confidence, int | float) and math.isfinite(float(primary_confidence)):
            out["fallback_primary_average_confidence"] = float(primary_confidence)
        if isinstance(threshold, int | float) and math.isfinite(float(threshold)):
            out["fallback_threshold"] = float(threshold)
        if isinstance(primary_quality_score, int | float) and math.isfinite(float(primary_quality_score)):
            out["fallback_primary_quality_score"] = float(primary_quality_score)
        if isinstance(min_quality_score, int | float) and math.isfinite(float(min_quality_score)):
            out["fallback_min_quality_score"] = float(min_quality_score)
    return out


def _output_artifacts(output_dir: Path) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for key, filename in (
        ("document_json", "document.json"),
        ("document_md", "document.md"),
        ("quality_json", "quality.json"),
    ):
        artifact = _file_artifact(output_dir / filename)
        if artifact is not None:
            artifacts[key] = artifact
    tables_dir = output_dir / "tables"
    if tables_dir.is_dir():
        csv_files = [_file_artifact(path) for path in sorted(tables_dir.glob("*.csv")) if path.is_file()]
        html_files = [_file_artifact(path) for path in sorted(tables_dir.glob("*.html")) if path.is_file()]
        artifacts["tables_csv"] = [item for item in csv_files if item is not None]
        artifacts["tables_html"] = [item for item in html_files if item is not None]
    figures_dir = output_dir / "figures"
    if figures_dir.is_dir():
        metadata = _file_artifact(figures_dir / "metadata.json")
        if metadata is not None:
            artifacts["figure_metadata"] = metadata
        image_files = [
            path
            for path in sorted(figures_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        artifacts["figure_images"] = [item for item in (_file_artifact(path) for path in image_files) if item is not None]
    return artifacts


def _file_artifact(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resource_snapshot(*, capture_gpu_metrics: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if resource is None:
        pass
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = float(usage.ru_maxrss)
        if max_rss > 0 and math.isfinite(max_rss):
            if sys.platform == "darwin":
                max_rss_mb = max_rss / (1024.0 * 1024.0)
            else:
                max_rss_mb = max_rss / 1024.0
            metrics["process_max_rss_mb"] = max_rss_mb
    if capture_gpu_metrics:
        metrics.update(_gpu_resource_snapshot())
    return metrics


def _resource_delta(before: dict[str, float], *, capture_gpu_metrics: bool = False) -> dict[str, float]:
    after = _resource_snapshot(capture_gpu_metrics=capture_gpu_metrics)
    if not after:
        return {}
    metrics = dict(after)
    before_rss = before.get("process_max_rss_mb")
    after_rss = after.get("process_max_rss_mb")
    if before_rss is not None and after_rss is not None:
        metrics["process_max_rss_delta_mb"] = max(0.0, after_rss - before_rss)
    before_gpu = before.get("gpu_memory_used_mb")
    after_gpu = after.get("gpu_memory_used_mb")
    if before_gpu is not None and after_gpu is not None:
        metrics["gpu_memory_used_delta_mb"] = max(0.0, after_gpu - before_gpu)
    return metrics


def _gpu_resource_snapshot() -> dict[str, float]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    command = [
        nvidia_smi,
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    used_values: list[float] = []
    total_values: list[float] = []
    utilization_values: list[float] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            used, total, utilization = (float(part) for part in parts)
        except ValueError:
            continue
        if math.isfinite(used):
            used_values.append(used)
        if math.isfinite(total):
            total_values.append(total)
        if math.isfinite(utilization):
            utilization_values.append(utilization)
    metrics: dict[str, float] = {}
    if used_values:
        metrics["gpu_memory_used_mb"] = max(used_values)
    if total_values:
        metrics["gpu_memory_total_mb"] = max(total_values)
    if utilization_values:
        metrics["gpu_utilization_percent"] = max(utilization_values)
    return metrics


def _write_reports(results: list[BenchmarkResult], out: Path) -> None:
    payload = [result.to_dict() for result in results]
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR Benchmark",
        "",
        "| baseline | sample | input | slices | status | latency_s | rss_mb | gpu_mem_mb | CER | WER | order | table_f1 | fallback | fallback_failed | error |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        metrics = result.metrics
        resources = result.resource_metrics
        lines.append(
            "| {baseline} | {sample} | {input_path} | {slices} | {status} | {latency:.3f} | {rss} | {gpu_mem} | {cer} | {wer} | {order} | {table_f1} | {fallback} | {fallback_failed} | {error} |".format(
                baseline=result.baseline,
                sample=result.sample_id or "",
                input_path=result.input_path,
                slices=",".join(result.slices),
                status=result.status,
                latency=result.latency_seconds,
                rss=f"{resources['process_max_rss_mb']:.1f}" if "process_max_rss_mb" in resources else "",
                gpu_mem=f"{resources['gpu_memory_used_mb']:.1f}" if "gpu_memory_used_mb" in resources else "",
                cer=f"{metrics['cer']:.4f}" if "cer" in metrics else "",
                wer=f"{metrics['wer']:.4f}" if "wer" in metrics else "",
                order=f"{metrics['reading_order_pair_accuracy']:.4f}" if "reading_order_pair_accuracy" in metrics else "",
                table_f1=f"{metrics['table_cell_f1']:.4f}" if "table_cell_f1" in metrics else "",
                fallback=f"{metrics['fallback_triggered']:.0f}" if "fallback_triggered" in metrics else "",
                fallback_failed=f"{metrics['fallback_failed']:.0f}" if "fallback_failed" in metrics else "",
                error=(result.error or "").replace("|", "\\|"),
            )
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_eval_entries(eval_manifest: str | Path | None) -> list[ManifestEntry]:
    if eval_manifest is None:
        return []
    return load_manifest(eval_manifest)


def _load_benchmark_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid benchmark report JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise BenchmarkError("benchmark report must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BenchmarkError(f"benchmark row {index} must be an object")
        rows.append(dict(item))
    return rows


def _entry_for_input(input_path: Path, entries: list[ManifestEntry]) -> ManifestEntry | None:
    if not entries:
        return None
    try:
        resolved_input = input_path.resolve()
    except OSError:
        resolved_input = input_path
    for entry in entries:
        entry_path = Path(entry.image_path)
        try:
            resolved_entry = entry_path.resolve()
        except OSError:
            resolved_entry = entry_path
        if resolved_input == resolved_entry or str(input_path) == entry.image_path:
            return entry
    return None


def _slices_for_entry(entry: ManifestEntry | None) -> list[str]:
    if entry is None:
        return []
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


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = str(row.get("baseline") or "")
        if not baseline:
            continue
        slices = sorted({str(item) for item in row.get("slices") or []}) or ["all"]
        slices.append("all")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for slice_name in slices:
            slice_bucket = grouped.setdefault(baseline, {}).setdefault(slice_name, {})
            latency_value = row.get("latency_seconds")
            if isinstance(latency_value, int | float) and math.isfinite(float(latency_value)):
                slice_bucket.setdefault("latency_seconds", []).append(float(latency_value))
            resource_metrics = row.get("resource_metrics")
            if isinstance(resource_metrics, dict):
                for metric_name, value in resource_metrics.items():
                    if isinstance(value, int | float) and math.isfinite(float(value)):
                        slice_bucket.setdefault(str(metric_name), []).append(float(value))
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


def _distinct_slices(rows: list[dict[str, Any]]) -> list[str]:
    values = {"all"}
    for row in rows:
        for slice_name in row.get("slices") or []:
            text = str(slice_name)
            if text:
                values.add(text)
    return sorted(values)


def _summarize_paired_metric(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    metric: str,
    direction: str,
    slice_name: str,
) -> PairedMetricSummary | None:
    candidate_values = _rows_by_sample(rows, candidate, metric, slice_name)
    baseline_values = _rows_by_sample(rows, baseline, metric, slice_name)
    shared = sorted(set(candidate_values) & set(baseline_values))
    if not shared:
        return None
    candidate_series = [candidate_values[key] for key in shared]
    baseline_series = [baseline_values[key] for key in shared]
    candidate_mean = sum(candidate_series) / len(candidate_series)
    baseline_mean = sum(baseline_series) / len(baseline_series)
    absolute_improvement = _absolute_improvement(candidate_mean, baseline_mean, direction)
    relative_improvement = _relative_improvement(absolute_improvement, baseline_mean)
    wins, losses, ties = _paired_outcome_counts(candidate_series, baseline_series, direction)
    non_ties = wins + losses
    return PairedMetricSummary(
        baseline=baseline,
        slice_name=slice_name,
        metric=metric,
        direction=direction,
        pairs=len(shared),
        candidate_mean=candidate_mean,
        baseline_mean=baseline_mean,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        win_count=wins,
        loss_count=losses,
        tie_count=ties,
        win_rate=(wins / non_ties) if non_ties else None,
    )


def _rows_by_sample(rows: list[dict[str, Any]], baseline: str, metric: str, slice_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row.get("baseline") != baseline or row.get("status") != "ok":
            continue
        if slice_name != "all" and slice_name not in set(row.get("slices") or []):
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


def _write_benchmark_summary(summary: BenchmarkSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Benchmark Summary",
        "",
        f"Report: `{summary.report_path}`",
        f"Candidate: `{summary.candidate}`",
        "",
        "## Paired Comparisons",
        "",
        "| baseline | slice | metric | direction | pairs | candidate | baseline | improvement | relative | wins | losses | ties | win_rate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if summary.paired_metrics:
        for item in summary.paired_metrics:
            lines.append(
                "| {baseline} | {slice} | {metric} | {direction} | {pairs} | {candidate_mean:.6f} | {baseline_mean:.6f} | {improvement:.6f} | {relative:.6f} | {wins} | {losses} | {ties} | {win_rate} |".format(
                    baseline=item.baseline,
                    slice=item.slice_name,
                    metric=item.metric,
                    direction=item.direction,
                    pairs=item.pairs,
                    candidate_mean=item.candidate_mean,
                    baseline_mean=item.baseline_mean,
                    improvement=item.absolute_improvement,
                    relative=item.relative_improvement,
                    wins=item.win_count,
                    losses=item.loss_count,
                    ties=item.tie_count,
                    win_rate=f"{item.win_rate:.4f}" if item.win_rate is not None else "",
                )
            )
    else:
        lines.append("| none |  |  |  | 0 |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Aggregate Metrics", ""])
    for baseline, slice_map in sorted(summary.aggregate_metrics.items()):
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
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
