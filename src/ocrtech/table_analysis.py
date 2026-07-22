"""Cell-level table analysis for benchmark outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BenchmarkError, DataValidationError
from .manifest import ManifestEntry, load_manifest
from .metrics import bbox_iou, cer
from .references import ReferenceTable, load_reference
from .schemas import Document
from .tables import table_to_grid


@dataclass(slots=True)
class TableCellSystemSummary:
    system: str
    reference_tables: int = 0
    predicted_tables: int = 0
    duplicate_overlapping_tables: int = 0
    reference_cells: int = 0
    exact_cells: int = 0
    missing_cells: int = 0
    total_cer: float = 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact_cells / self.reference_cells if self.reference_cells else 0.0

    @property
    def mean_cer(self) -> float:
        return self.total_cer / self.reference_cells if self.reference_cells else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "reference_tables": self.reference_tables,
            "predicted_tables": self.predicted_tables,
            "duplicate_overlapping_tables": self.duplicate_overlapping_tables,
            "reference_cells": self.reference_cells,
            "exact_cells": self.exact_cells,
            "missing_cells": self.missing_cells,
            "exact_rate": self.exact_rate,
            "mean_cer": self.mean_cer,
        }


@dataclass(slots=True)
class TableCellPairSummary:
    baseline: str
    cells: int = 0
    candidate_better: int = 0
    baseline_better: int = 0
    ties: int = 0
    candidate_exact_baseline_not: int = 0
    baseline_exact_candidate_not: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "cells": self.cells,
            "candidate_better": self.candidate_better,
            "baseline_better": self.baseline_better,
            "ties": self.ties,
            "candidate_exact_baseline_not": self.candidate_exact_baseline_not,
            "baseline_exact_candidate_not": self.baseline_exact_candidate_not,
        }


@dataclass(slots=True)
class TableCellSampleSummary:
    sample_id: str
    system: str
    reference_tables: int
    predicted_tables: int
    duplicate_overlapping_tables: int
    reference_shapes: list[list[int]]
    predicted_shapes: list[list[int]]
    reference_cells: int
    exact_cells: int
    missing_cells: int
    mean_cer: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "system": self.system,
            "reference_tables": self.reference_tables,
            "predicted_tables": self.predicted_tables,
            "duplicate_overlapping_tables": self.duplicate_overlapping_tables,
            "reference_shapes": self.reference_shapes,
            "predicted_shapes": self.predicted_shapes,
            "reference_cells": self.reference_cells,
            "exact_cells": self.exact_cells,
            "missing_cells": self.missing_cells,
            "exact_rate": self.exact_cells / self.reference_cells if self.reference_cells else 0.0,
            "mean_cer": self.mean_cer,
        }


@dataclass(slots=True)
class TableCellAnalysis:
    benchmark_report: str
    eval_manifest: str
    candidate: str
    systems: dict[str, TableCellSystemSummary]
    paired: dict[str, TableCellPairSummary]
    samples: list[TableCellSampleSummary] = field(default_factory=list)
    top_failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_report": self.benchmark_report,
            "eval_manifest": self.eval_manifest,
            "candidate": self.candidate,
            "systems": {key: value.to_dict() for key, value in sorted(self.systems.items())},
            "paired": {key: value.to_dict() for key, value in sorted(self.paired.items())},
            "samples": [item.to_dict() for item in self.samples],
            "top_failures": self.top_failures,
            "warnings": self.warnings,
        }


def analyze_table_cells(
    benchmark_report: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    candidate: str = "candidate",
    baselines: list[str] | None = None,
    top_n: int = 25,
) -> TableCellAnalysis:
    report_path = Path(benchmark_report)
    rows = _load_report_rows(report_path)
    entries = {entry.sample_id: entry for entry in load_manifest(eval_manifest)}
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    systems = sorted({str(row.get("baseline") or "") for row in rows if row.get("status") == "ok"})
    if candidate not in systems:
        raise BenchmarkError(f"candidate system {candidate!r} is not present in benchmark report")
    selected_baselines = baselines or [name for name in systems if name != candidate]
    missing = [name for name in selected_baselines if name not in systems]
    if missing:
        raise BenchmarkError(f"requested systems missing from benchmark report: {', '.join(missing)}")

    docs = _load_documents(report_path, rows)
    system_summaries = {system: TableCellSystemSummary(system=system) for system in systems}
    pair_summaries = {baseline: TableCellPairSummary(baseline=baseline) for baseline in selected_baselines}
    sample_summaries: list[TableCellSampleSummary] = []
    top_failures: list[dict[str, Any]] = []
    warnings: list[str] = []

    rows_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        baseline = str(row.get("baseline") or "")
        if sample_id and baseline and row.get("status") == "ok":
            rows_by_sample[sample_id].add(baseline)

    for sample_id, present_systems in sorted(rows_by_sample.items()):
        entry = entries.get(sample_id)
        if entry is None:
            warnings.append(f"{sample_id}: missing eval manifest entry")
            continue
        reference = load_reference(Path(entry.image_path), explicit_path=entry.metadata.get("reference_path"))
        if reference is None or not reference.tables:
            continue
        reference_cells = _reference_cells(reference.tables)
        if not reference_cells:
            continue
        predictions: dict[str, dict[tuple[int, int, int], str]] = {}
        for system in systems:
            document = docs.get((system, sample_id))
            if document is None:
                continue
            _update_table_summary(system_summaries[system], document, reference.tables)
            predictions[system] = _predicted_cells(document)
            _update_system_summary(system_summaries[system], reference_cells, predictions[system])
            sample_summaries.append(_sample_summary(sample_id, system, document, reference.tables, reference_cells, predictions[system]))
        candidate_cells = predictions.get(candidate)
        if candidate_cells is None:
            warnings.append(f"{sample_id}: missing candidate document")
            continue
        for baseline in selected_baselines:
            baseline_cells = predictions.get(baseline)
            if baseline_cells is None:
                continue
            _update_pair_summary(pair_summaries[baseline], reference_cells, candidate_cells, baseline_cells)
        top_failures.extend(_sample_failures(sample_id, reference_cells, candidate_cells, selected_baselines, predictions))

    top_failures = sorted(top_failures, key=lambda item: float(item["candidate_cer"]), reverse=True)[:top_n]
    analysis = TableCellAnalysis(
        benchmark_report=str(report_path),
        eval_manifest=str(eval_manifest),
        candidate=candidate,
        systems=system_summaries,
        paired=pair_summaries,
        samples=sample_summaries,
        top_failures=top_failures,
        warnings=warnings,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table-cell-analysis.json").write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(analysis, out / "table-cell-analysis.md")
    return analysis


def _load_report_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _load_documents(report_path: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, str], Document]:
    benchmark_dir = report_path.parent
    docs: dict[tuple[str, str], Document] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = str(row.get("baseline") or "")
        sample_id = str(row.get("sample_id") or "")
        input_path = Path(str(row.get("input_path") or ""))
        if not baseline or not sample_id or not input_path.name:
            continue
        document_path = benchmark_dir / baseline / input_path.stem / "document.json"
        if not document_path.exists():
            continue
        try:
            payload = json.loads(document_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"invalid document JSON {document_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataValidationError(f"document JSON must be an object: {document_path}")
        docs[(baseline, sample_id)] = Document.from_dict(payload)
    return docs


def _reference_cells(tables: list[ReferenceTable]) -> dict[tuple[int, int, int], str]:
    cells: dict[tuple[int, int, int], str] = {}
    for table_index, table in enumerate(tables):
        for row_index, row in enumerate(table.grid):
            for col_index, value in enumerate(row):
                cells[(table_index, row_index, col_index)] = str(value)
    return cells


def _predicted_cells(document: Document) -> dict[tuple[int, int, int], str]:
    cells: dict[tuple[int, int, int], str] = {}
    tables = sorted(document.tables, key=lambda item: (item.page_index, item.bbox.y, item.bbox.x))
    for table_index, table in enumerate(tables):
        grid = table_to_grid(table)
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                cells[(table_index, row_index, col_index)] = str(value)
    return cells


def _update_table_summary(summary: TableCellSystemSummary, document: Document, reference_tables: list[ReferenceTable]) -> None:
    summary.reference_tables += len(reference_tables)
    summary.predicted_tables += len(document.tables)
    summary.duplicate_overlapping_tables += _duplicate_overlapping_table_count(document.tables)


def _duplicate_overlapping_table_count(tables: list[Any], *, iou_threshold: float = 0.9) -> int:
    duplicates = 0
    kept: list[Any] = []
    for table in sorted(tables, key=lambda item: (item.page_index, item.bbox.y, item.bbox.x)):
        duplicate = False
        for existing in kept:
            if table.page_index == existing.page_index and bbox_iou(table.bbox, existing.bbox) >= iou_threshold:
                duplicate = True
                break
        if duplicate:
            duplicates += 1
        else:
            kept.append(table)
    return duplicates


def _update_system_summary(
    summary: TableCellSystemSummary,
    reference_cells: dict[tuple[int, int, int], str],
    predicted_cells: dict[tuple[int, int, int], str],
) -> None:
    for key, reference_text in reference_cells.items():
        predicted_text = predicted_cells.get(key, "")
        score = cer(predicted_text, reference_text)
        summary.reference_cells += 1
        summary.total_cer += score
        if not predicted_text:
            summary.missing_cells += 1
        if score == 0.0:
            summary.exact_cells += 1


def _sample_summary(
    sample_id: str,
    system: str,
    document: Document,
    reference_tables: list[ReferenceTable],
    reference_cells: dict[tuple[int, int, int], str],
    predicted_cells: dict[tuple[int, int, int], str],
) -> TableCellSampleSummary:
    exact_cells = 0
    missing_cells = 0
    total_cer = 0.0
    for key, reference_text in reference_cells.items():
        predicted_text = predicted_cells.get(key, "")
        score = cer(predicted_text, reference_text)
        total_cer += score
        if score == 0.0:
            exact_cells += 1
        if not predicted_text:
            missing_cells += 1
    return TableCellSampleSummary(
        sample_id=sample_id,
        system=system,
        reference_tables=len(reference_tables),
        predicted_tables=len(document.tables),
        duplicate_overlapping_tables=_duplicate_overlapping_table_count(document.tables),
        reference_shapes=[_grid_shape(table.grid) for table in reference_tables],
        predicted_shapes=[_grid_shape(table_to_grid(table)) for table in sorted(document.tables, key=lambda item: (item.page_index, item.bbox.y, item.bbox.x))],
        reference_cells=len(reference_cells),
        exact_cells=exact_cells,
        missing_cells=missing_cells,
        mean_cer=total_cer / len(reference_cells) if reference_cells else 0.0,
    )


def _grid_shape(grid: list[list[str]]) -> list[int]:
    return [len(grid), max((len(row) for row in grid), default=0)]


def _update_pair_summary(
    summary: TableCellPairSummary,
    reference_cells: dict[tuple[int, int, int], str],
    candidate_cells: dict[tuple[int, int, int], str],
    baseline_cells: dict[tuple[int, int, int], str],
) -> None:
    for key, reference_text in reference_cells.items():
        candidate_score = cer(candidate_cells.get(key, ""), reference_text)
        baseline_score = cer(baseline_cells.get(key, ""), reference_text)
        summary.cells += 1
        if candidate_score < baseline_score:
            summary.candidate_better += 1
        elif baseline_score < candidate_score:
            summary.baseline_better += 1
        else:
            summary.ties += 1
        if candidate_score == 0.0 and baseline_score != 0.0:
            summary.candidate_exact_baseline_not += 1
        if baseline_score == 0.0 and candidate_score != 0.0:
            summary.baseline_exact_candidate_not += 1


def _sample_failures(
    sample_id: str,
    reference_cells: dict[tuple[int, int, int], str],
    candidate_cells: dict[tuple[int, int, int], str],
    baselines: list[str],
    predictions: dict[str, dict[tuple[int, int, int], str]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key, reference_text in reference_cells.items():
        candidate_text = candidate_cells.get(key, "")
        candidate_score = cer(candidate_text, reference_text)
        if candidate_score == 0.0:
            continue
        baseline_payload: dict[str, Any] = {}
        for baseline in baselines:
            baseline_cells = predictions.get(baseline)
            if baseline_cells is None:
                continue
            baseline_text = baseline_cells.get(key, "")
            baseline_payload[baseline] = {
                "text": baseline_text,
                "cer": cer(baseline_text, reference_text),
            }
        table_index, row_index, col_index = key
        failures.append(
            {
                "sample_id": sample_id,
                "table_index": table_index,
                "row": row_index,
                "col": col_index,
                "reference": reference_text,
                "candidate": candidate_text,
                "candidate_cer": candidate_score,
                "baselines": baseline_payload,
            }
        )
    return failures


def _write_markdown(analysis: TableCellAnalysis, path: Path) -> None:
    lines = [
        "# Table Cell Analysis",
        "",
        f"Benchmark report: `{analysis.benchmark_report}`",
        f"Eval manifest: `{analysis.eval_manifest}`",
        f"Candidate: `{analysis.candidate}`",
        "",
        "## Systems",
        "",
        "| system | reference_tables | predicted_tables | duplicate_tables | reference_cells | exact_cells | exact_rate | missing_cells | mean_cer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in sorted(analysis.systems.values(), key=lambda item: item.system):
        lines.append(
            f"| {summary.system} | {summary.reference_tables} | {summary.predicted_tables} | {summary.duplicate_overlapping_tables} | "
            f"{summary.reference_cells} | {summary.exact_cells} | {summary.exact_rate:.6f} | {summary.missing_cells} | {summary.mean_cer:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired",
            "",
            "| baseline | cells | candidate_better | baseline_better | ties | candidate_exact_baseline_not | baseline_exact_candidate_not |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in sorted(analysis.paired.values(), key=lambda item: item.baseline):
        lines.append(
            f"| {summary.baseline} | {summary.cells} | {summary.candidate_better} | {summary.baseline_better} | {summary.ties} | {summary.candidate_exact_baseline_not} | {summary.baseline_exact_candidate_not} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Table Shape",
            "",
            "| sample | system | reference_tables | predicted_tables | reference_shapes | predicted_shapes | exact_cells | missing_cells | mean_cer |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(analysis.samples, key=lambda value: (value.sample_id, value.system)):
        lines.append(
            "| {sample_id} | {system} | {reference_tables} | {predicted_tables} | {reference_shapes} | {predicted_shapes} | {exact_cells} | {missing_cells} | {mean_cer:.6f} |".format(
                sample_id=item.sample_id,
                system=item.system,
                reference_tables=item.reference_tables,
                predicted_tables=item.predicted_tables,
                reference_shapes="; ".join(f"{shape[0]}x{shape[1]}" for shape in item.reference_shapes),
                predicted_shapes="; ".join(f"{shape[0]}x{shape[1]}" for shape in item.predicted_shapes),
                exact_cells=item.exact_cells,
                missing_cells=item.missing_cells,
                mean_cer=item.mean_cer,
            )
        )
    lines.extend(["", "## Top Candidate Cell Failures", ""])
    if analysis.top_failures:
        lines.extend(
            [
                "| sample | table | row | col | reference | candidate | candidate_cer |",
                "| --- | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for item in analysis.top_failures:
            lines.append(
                "| {sample_id} | {table_index} | {row} | {col} | {reference} | {candidate} | {candidate_cer:.6f} |".format(
                    sample_id=item["sample_id"],
                    table_index=item["table_index"],
                    row=item["row"],
                    col=item["col"],
                    reference=str(item["reference"]).replace("|", "\\|"),
                    candidate=str(item["candidate"]).replace("|", "\\|"),
                    candidate_cer=float(item["candidate_cer"]),
                )
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in analysis.warnings) if analysis.warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
