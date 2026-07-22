"""Paper-oriented exports from benchmark summary artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BenchmarkError


DEFAULT_PAPER_METRICS = ["cer", "table_cell_f1", "reading_order_pair_accuracy"]


@dataclass(slots=True)
class PaperTableRow:
    system: str
    display_name: str
    sample_count: int | None
    metrics: dict[str, float]
    sources: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "display_name": self.display_name,
            "sample_count": self.sample_count,
            "metrics": self.metrics,
            "sources": self.sources,
        }


@dataclass(slots=True)
class PaperBenchmarkTable:
    title: str
    slice_name: str
    metrics: list[str]
    rows: list[PaperTableRow]
    source_summaries: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "slice": self.slice_name,
            "metrics": self.metrics,
            "rows": [row.to_dict() for row in self.rows],
            "source_summaries": self.source_summaries,
        }


def export_paper_benchmark_table(
    summary_paths: list[str | Path],
    output_dir: str | Path,
    *,
    slice_name: str = "all",
    metrics: list[str] | None = None,
    systems: list[str] | None = None,
    display_names: dict[str, str] | None = None,
    source_preferences: dict[str, str] | None = None,
    title: str | None = None,
) -> PaperBenchmarkTable:
    if not summary_paths:
        raise BenchmarkError("paper benchmark table requires at least one summary artifact")
    selected_metrics = metrics or list(DEFAULT_PAPER_METRICS)
    ordered_systems = list(dict.fromkeys(systems or []))
    filter_systems = set(ordered_systems)
    labels = dict(display_names or {})
    preferred_sources = {
        system: str(Path(path))
        for system, path in dict(source_preferences or {}).items()
    }

    rows_by_system: dict[str, PaperTableRow] = {}
    seen_order: list[str] = []
    source_summaries: list[str] = []
    for summary_path in summary_paths:
        payload = _load_summary_json(summary_path)
        normalized_summary_path = str(Path(summary_path))
        source_summaries.append(normalized_summary_path)
        aggregate_metrics = payload.get("aggregate_metrics")
        if not isinstance(aggregate_metrics, dict):
            raise BenchmarkError(f"summary {summary_path} missing aggregate_metrics object")
        for system_name, slice_map in aggregate_metrics.items():
            system = str(system_name)
            if filter_systems and system not in filter_systems:
                continue
            preferred_source = preferred_sources.get(system)
            if preferred_source and preferred_source != normalized_summary_path:
                continue
            if not isinstance(slice_map, dict):
                continue
            slice_metrics = slice_map.get(slice_name)
            if not isinstance(slice_metrics, dict):
                continue
            row = rows_by_system.get(system)
            sample_count = _optional_int(slice_metrics.get("sample_count"), field_name=f"{system}.{slice_name}.sample_count")
            if row is None:
                row = PaperTableRow(
                    system=system,
                    display_name=labels.get(system, system),
                    sample_count=sample_count,
                    metrics={},
                    sources=[],
                )
                rows_by_system[system] = row
                seen_order.append(system)
            elif sample_count is not None:
                _merge_sample_count(row, sample_count, summary_path, slice_name)
            for metric in selected_metrics:
                value = slice_metrics.get(metric)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                    raise BenchmarkError(
                        f"summary {summary_path} has non-finite {metric!r} for system {system!r} slice {slice_name!r}"
                    )
                _merge_metric(row, metric, float(value), summary_path, slice_name)
            row.sources.append(str(Path(summary_path)))

    if not rows_by_system:
        raise BenchmarkError(f"no systems found for slice {slice_name!r} in the provided summaries")
    if ordered_systems:
        missing = [system for system in ordered_systems if system not in rows_by_system]
        if missing:
            raise BenchmarkError(f"requested systems missing from provided summaries: {', '.join(missing)}")
        final_systems = ordered_systems
    else:
        final_systems = seen_order
    table = PaperBenchmarkTable(
        title=title or f"Benchmark Comparison ({slice_name})",
        slice_name=slice_name,
        metrics=selected_metrics,
        rows=[rows_by_system[system] for system in final_systems],
        source_summaries=source_summaries,
    )
    _write_paper_benchmark_table(table, Path(output_dir))
    return table


def _load_summary_json(path: str | Path) -> dict[str, Any]:
    raw_path = Path(path)
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"summary artifact not found: {raw_path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid summary JSON {raw_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"summary artifact must be a JSON object: {raw_path}")
    return payload


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkError(f"{field_name} must be numeric, got {type(value).__name__}")
    return int(value)


def _merge_sample_count(row: PaperTableRow, sample_count: int, source_path: str | Path, slice_name: str) -> None:
    if row.sample_count is None:
        row.sample_count = sample_count
        return
    if row.sample_count != sample_count:
        raise BenchmarkError(
            f"conflicting sample_count for system {row.system!r} slice {slice_name!r}: "
            f"{row.sample_count} vs {sample_count} in {source_path}"
        )


def _merge_metric(row: PaperTableRow, metric: str, value: float, source_path: str | Path, slice_name: str) -> None:
    previous = row.metrics.get(metric)
    if previous is None:
        row.metrics[metric] = value
        return
    if not math.isclose(previous, value, rel_tol=1e-9, abs_tol=1e-9):
        raise BenchmarkError(
            f"conflicting aggregate value for system {row.system!r} slice {slice_name!r} metric {metric!r}: "
            f"{previous} vs {value} in {source_path}"
        )


def _write_paper_benchmark_table(table: PaperBenchmarkTable, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper-table.json").write_text(
        json.dumps(table.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = ["system", "sample_count", *table.metrics]
    lines = [
        "# Paper Benchmark Table",
        "",
        f"Title: {table.title}",
        f"Slice: `{table.slice_name}`",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if index > 0 else "---" for index, _ in enumerate(columns)) + " |",
    ]
    for row in table.rows:
        values = [row.display_name, "" if row.sample_count is None else str(row.sample_count)]
        for metric in table.metrics:
            metric_value = row.metrics.get(metric)
            values.append("" if metric_value is None else f"{metric_value:.6f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Source Summaries", ""])
    for source in table.source_summaries:
        lines.append(f"- `{source}`")
    (output_dir / "paper-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
