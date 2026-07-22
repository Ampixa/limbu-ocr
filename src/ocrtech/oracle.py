"""Oracle analysis over benchmark reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    _load_benchmark_rows,
    summarize_benchmark_report,
)
from .errors import BenchmarkError
from .validation import metric_direction, sample_key


@dataclass(slots=True)
class OracleReport:
    benchmark_report: str
    oracle_name: str
    metric: str
    direction: str
    systems: list[str]
    sample_count: int
    chosen_counts: dict[str, int]
    oracle_report_path: str
    oracle_summary_path: str
    oracle_choices_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_report": self.benchmark_report,
            "oracle_name": self.oracle_name,
            "metric": self.metric,
            "direction": self.direction,
            "systems": self.systems,
            "sample_count": self.sample_count,
            "chosen_counts": self.chosen_counts,
            "oracle_report_path": self.oracle_report_path,
            "oracle_summary_path": self.oracle_summary_path,
            "oracle_choices_path": self.oracle_choices_path,
        }


def build_benchmark_oracle(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    metric: str = "cer",
    systems: list[str] | None = None,
    direction: str | None = None,
    oracle_name: str = "oracle",
) -> tuple[OracleReport, BenchmarkSummary]:
    rows = _load_benchmark_rows(report_path)
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    compare_systems = systems or sorted({str(row.get("baseline") or "") for row in rows if str(row.get("baseline") or "")})
    compare_systems = [name for name in compare_systems if name]
    if len(compare_systems) < 2:
        raise BenchmarkError("benchmark oracle requires at least two systems")
    missing = [name for name in compare_systems if not any(row.get("baseline") == name for row in rows)]
    if missing:
        raise BenchmarkError(f"requested systems missing from benchmark report: {', '.join(missing)}")
    target_direction = direction or metric_direction(metric)
    oracle_rows, chosen_counts, choices = _select_oracle_rows(rows, compare_systems, metric, target_direction, oracle_name)
    if not oracle_rows:
        raise BenchmarkError(f"benchmark oracle found no shared comparable rows for metric {metric!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined_rows = [dict(row) for row in rows] + [result.to_dict() for result in oracle_rows]
    oracle_report_path = out / "oracle-report.json"
    oracle_report_path.write_text(json.dumps(combined_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    oracle_choices_path = out / "oracle-choices.json"
    oracle_choices_path.write_text(json.dumps(choices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = summarize_benchmark_report(
        oracle_report_path,
        out / "oracle-summary",
        candidate=oracle_name,
        baselines=compare_systems,
        metrics=[metric],
    )
    report = OracleReport(
        benchmark_report=str(Path(report_path)),
        oracle_name=oracle_name,
        metric=metric,
        direction=target_direction,
        systems=compare_systems,
        sample_count=len(oracle_rows),
        chosen_counts=chosen_counts,
        oracle_report_path=str(oracle_report_path),
        oracle_summary_path=str(out / "oracle-summary" / "summary.json"),
        oracle_choices_path=str(oracle_choices_path),
    )
    (out / "oracle-analysis.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Benchmark Oracle",
        "",
        f"Source report: `{report_path}`",
        f"Oracle: `{oracle_name}`",
        f"Metric: `{metric}`",
        f"Direction: `{target_direction}`",
        f"Samples: `{len(oracle_rows)}`",
        "",
        "## Chosen Counts",
        "",
        "| system | chosen |",
        "| --- | ---: |",
    ]
    for system, count in sorted(chosen_counts.items()):
        lines.append(f"| {system} | {count} |")
    lines.extend(
        [
            "",
            f"Oracle report: `{oracle_report_path}`",
            f"Oracle choices: `{oracle_choices_path}`",
            f"Oracle summary: `{out / 'oracle-summary' / 'summary.json'}`",
        ]
    )
    (out / "oracle-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, summary


def _select_oracle_rows(
    rows: list[dict[str, Any]],
    systems: list[str],
    metric: str,
    direction: str,
    oracle_name: str,
) -> tuple[list[BenchmarkResult], dict[str, int], list[dict[str, Any]]]:
    rows_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = str(row.get("baseline") or "")
        if baseline not in systems:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        sample = sample_key(row)
        rows_by_sample.setdefault(sample, {})[baseline] = row
    oracle_rows: list[BenchmarkResult] = []
    chosen_counts = {system: 0 for system in systems}
    choices: list[dict[str, Any]] = []
    for sample, sample_rows in sorted(rows_by_sample.items()):
        if any(system not in sample_rows for system in systems):
            continue
        chosen_system, chosen_row = _best_row(sample_rows, systems, metric, direction)
        chosen_counts[chosen_system] += 1
        oracle_row = dict(chosen_row)
        oracle_row["baseline"] = oracle_name
        oracle_row["error"] = None
        metrics = dict(chosen_row.get("metrics") or {})
        oracle_row["metrics"] = metrics
        choices.append(
            {
                "sample_key": sample,
                "sample_id": chosen_row.get("sample_id"),
                "input_path": chosen_row.get("input_path"),
                "chosen_system": chosen_system,
                "metric": metric,
                "metric_value": float(metrics[metric]),
                "direction": direction,
            }
        )
        oracle_result = BenchmarkResult(
            baseline=str(oracle_row["baseline"]),
            input_path=str(oracle_row.get("input_path") or ""),
            status=str(oracle_row.get("status") or "ok"),
            latency_seconds=float(oracle_row.get("latency_seconds") or 0.0),
            sample_id=oracle_row.get("sample_id"),
            slices=list(oracle_row.get("slices") or []),
            metrics=metrics,
            error=None,
        )
        oracle_result.slices = list(oracle_row.get("slices") or [])
        oracle_rows.append(oracle_result)
    return oracle_rows, {system: count for system, count in chosen_counts.items() if count}, choices


def _best_row(
    sample_rows: dict[str, dict[str, Any]],
    systems: list[str],
    metric: str,
    direction: str,
) -> tuple[str, dict[str, Any]]:
    best_system = systems[0]
    best_row = sample_rows[best_system]
    best_value = float((best_row.get("metrics") or {})[metric])
    for system in systems[1:]:
        row = sample_rows[system]
        value = float((row.get("metrics") or {})[metric])
        if _is_better(value, best_value, direction):
            best_system = system
            best_row = row
            best_value = value
            continue
        if value == best_value:
            if _tie_break_quality(row) > _tie_break_quality(best_row):
                best_system = system
                best_row = row
                best_value = value
            elif _tie_break_quality(row) == _tie_break_quality(best_row):
                row_latency = float(row.get("latency_seconds") or 0.0)
                best_latency = float(best_row.get("latency_seconds") or 0.0)
                if row_latency < best_latency:
                    best_system = system
                    best_row = row
                    best_value = value
    return best_system, best_row


def _is_better(candidate: float, incumbent: float, direction: str) -> bool:
    if direction == "lower":
        return candidate < incumbent
    return candidate > incumbent


def _tie_break_quality(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    value = metrics.get("quality_score")
    if isinstance(value, int | float):
        return float(value)
    return float("-inf")
