"""Error analysis utilities for benchmark reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import _load_benchmark_rows
from .errors import BenchmarkError
from .manifest import ManifestEntry, load_manifest, write_manifest
from .validation import metric_direction, sample_key


@dataclass(slots=True)
class BenchmarkErrorAnalysis:
    benchmark_report: str
    system: str
    metric: str
    direction: str
    baselines: list[str]
    sample_count: int
    top_failures_path: str
    analysis_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_report": self.benchmark_report,
            "system": self.system,
            "metric": self.metric,
            "direction": self.direction,
            "baselines": self.baselines,
            "sample_count": self.sample_count,
            "top_failures_path": self.top_failures_path,
            "analysis_path": self.analysis_path,
        }


@dataclass(slots=True)
class FailureManifestExport:
    source_manifest: str
    top_failures_path: str
    output_manifest: str
    selected_count: int
    split: str | None
    summary_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_manifest": self.source_manifest,
            "top_failures_path": self.top_failures_path,
            "output_manifest": self.output_manifest,
            "selected_count": self.selected_count,
            "split": self.split,
            "summary_path": self.summary_path,
        }


def analyze_benchmark_errors(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    system: str,
    baselines: list[str] | None = None,
    metric: str = "cer",
    direction: str | None = None,
    top_n: int = 20,
) -> BenchmarkErrorAnalysis:
    if top_n < 1:
        raise BenchmarkError("top_n must be at least 1")
    rows = _load_benchmark_rows(report_path)
    if not rows:
        raise BenchmarkError("benchmark report has no rows")
    target_direction = direction or metric_direction(metric)
    available_systems = sorted({str(row.get("baseline") or "") for row in rows if str(row.get("baseline") or "")})
    if system not in available_systems:
        raise BenchmarkError(f"system {system!r} is not present in benchmark report")
    compare_baselines = baselines or []
    missing = [name for name in compare_baselines if name not in available_systems]
    if missing:
        raise BenchmarkError(f"requested baselines missing from benchmark report: {', '.join(missing)}")

    rows_by_system = _metric_rows_by_system(rows, metric)
    system_rows = rows_by_system.get(system, {})
    if not system_rows:
        raise BenchmarkError(f"system {system!r} has no ok rows with metric {metric!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    top_failures = _top_failures(system_rows, rows_by_system, compare_baselines, metric, target_direction, top_n)
    slice_summary = _slice_summary(system_rows, rows_by_system, compare_baselines, metric, target_direction)
    baseline_summary = _baseline_summary(system_rows, rows_by_system, compare_baselines, metric, target_direction)
    top_failure_slice_counts = _top_failure_slice_counts(top_failures)

    top_failures_path = out / "top-failures.json"
    top_failures_path.write_text(json.dumps(top_failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "benchmark_report": str(Path(report_path)),
        "system": system,
        "metric": metric,
        "direction": target_direction,
        "baselines": compare_baselines,
        "sample_count": len(system_rows),
        "baseline_summary": baseline_summary,
        "slice_summary": slice_summary,
        "top_failure_slice_counts": top_failure_slice_counts,
        "top_failures_path": str(top_failures_path),
    }
    analysis_path = out / "error-analysis.json"
    analysis_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out / "error-analysis.md", payload, top_failures)
    return BenchmarkErrorAnalysis(
        benchmark_report=str(Path(report_path)),
        system=system,
        metric=metric,
        direction=target_direction,
        baselines=compare_baselines,
        sample_count=len(system_rows),
        top_failures_path=str(top_failures_path),
        analysis_path=str(analysis_path),
    )


def export_failure_manifest(
    source_manifest: str | Path,
    top_failures_path: str | Path,
    output_manifest: str | Path,
    *,
    split: str | None = None,
) -> FailureManifestExport:
    entries = load_manifest(source_manifest)
    entries_by_sample = {entry.sample_id: entry for entry in entries}
    try:
        failure_rows = json.loads(Path(top_failures_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid top failures JSON {top_failures_path}: {exc}") from exc
    if not isinstance(failure_rows, list):
        raise BenchmarkError("top failures file must contain a JSON list")
    selected: list[ManifestEntry] = []
    missing: list[str] = []
    seen: set[str] = set()
    for rank, item in enumerate(failure_rows, start=1):
        if not isinstance(item, dict):
            raise BenchmarkError(f"top failure row {rank} must be an object")
        sample_id = str(item.get("sample_id") or item.get("sample_key") or "")
        if not sample_id:
            raise BenchmarkError(f"top failure row {rank} missing sample_id/sample_key")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        source = entries_by_sample.get(sample_id)
        if source is None:
            missing.append(sample_id)
            continue
        selected.append(_failure_entry(source, failure=item, rank=rank, split=split))
    if missing:
        raise BenchmarkError(f"top failure samples missing from source manifest: {', '.join(missing)}")
    if not selected:
        raise BenchmarkError("failure manifest export selected zero samples")
    out_path = Path(output_manifest)
    write_manifest(selected, out_path)
    summary_path = out_path.with_name(f"{out_path.stem}-failure-manifest.json")
    export = FailureManifestExport(
        source_manifest=str(Path(source_manifest)),
        top_failures_path=str(Path(top_failures_path)),
        output_manifest=str(out_path),
        selected_count=len(selected),
        split=split,
        summary_path=str(summary_path),
    )
    summary_path.write_text(json.dumps(export.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_failure_manifest_markdown(out_path.with_name(f"{out_path.stem}-failure-manifest.md"), export, selected)
    return export


def _metric_rows_by_system(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, dict[str, Any]]]:
    by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = str(row.get("baseline") or "")
        if not baseline:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            continue
        by_system.setdefault(baseline, {})[sample_key(row)] = row
    return by_system


def _failure_entry(source: ManifestEntry, *, failure: dict[str, Any], rank: int, split: str | None) -> ManifestEntry:
    metadata = dict(source.metadata or {})
    if split:
        existing_slices = metadata.get("slices")
        slice_values = list(existing_slices) if isinstance(existing_slices, list) else []
        if split not in {str(item) for item in slice_values}:
            slice_values.append(split)
        metadata["slices"] = slice_values
    metric_name, metric_value = _failure_metric(failure)
    metadata["failure_analysis"] = {
        "rank": rank,
        "metric": metric_name,
        "metric_value": metric_value,
        "worst_gap": failure.get("worst_gap"),
        "comparisons": failure.get("comparisons") if isinstance(failure.get("comparisons"), dict) else {},
    }
    return ManifestEntry(
        sample_id=source.sample_id,
        dataset=source.dataset,
        split=split or source.split,
        image_path=source.image_path,
        text=source.text,
        sha256=source.sha256,
        metadata=metadata,
    )


def _failure_metric(failure: dict[str, Any]) -> tuple[str, float]:
    for key, value in failure.items():
        if key in {"badness", "worst_gap"}:
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return key, float(value)
    raise BenchmarkError("top failure row has no finite metric value")


def _write_failure_manifest_markdown(path: Path, export: FailureManifestExport, selected: list[ManifestEntry]) -> None:
    lines = [
        "# Failure Manifest Export",
        "",
        f"Source manifest: `{export.source_manifest}`",
        f"Top failures: `{export.top_failures_path}`",
        f"Output manifest: `{export.output_manifest}`",
        f"Selected samples: `{export.selected_count}`",
        f"Split override: `{export.split or ''}`",
        "",
        "| rank | sample_id | metric | value | image_path |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for entry in selected:
        failure = (entry.metadata or {}).get("failure_analysis") or {}
        lines.append(
            "| {rank} | {sample_id} | {metric} | {value:.6f} | {image_path} |".format(
                rank=int(failure.get("rank") or 0),
                sample_id=entry.sample_id,
                metric=str(failure.get("metric") or ""),
                value=float(failure.get("metric_value") or 0.0),
                image_path=entry.image_path,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_failures(
    system_rows: dict[str, dict[str, Any]],
    rows_by_system: dict[str, dict[str, dict[str, Any]]],
    baselines: list[str],
    metric: str,
    direction: str,
    top_n: int,
) -> list[dict[str, Any]]:
    ranked = [_failure_record(sample, row, rows_by_system, baselines, metric, direction) for sample, row in system_rows.items()]
    ranked.sort(key=_failure_sort_key, reverse=True)
    return ranked[:top_n]


def _top_failure_slice_counts(top_failures: list[dict[str, Any]]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for item in top_failures:
        slices = item.get("slices")
        if not isinstance(slices, list):
            continue
        for slice_name in slices:
            text = str(slice_name)
            if text:
                counts[text] = counts.get(text, 0) + 1
    return [{"slice": slice_name, "count": count} for slice_name, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def _failure_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    worst_gap = item.get("worst_gap")
    badness = float(item["badness"])
    if isinstance(worst_gap, int | float) and math.isfinite(float(worst_gap)):
        gap = float(worst_gap)
        return (1.0 if gap > 0 else 0.0, gap if gap > 0 else badness, badness)
    return (0.0, badness, badness)


def _failure_record(
    sample: str,
    row: dict[str, Any],
    rows_by_system: dict[str, dict[str, dict[str, Any]]],
    baselines: list[str],
    metric: str,
    direction: str,
) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    value = float(metrics[metric])
    comparisons: dict[str, dict[str, float]] = {}
    gaps: list[float] = []
    for baseline in baselines:
        baseline_row = rows_by_system.get(baseline, {}).get(sample)
        if baseline_row is None:
            continue
        baseline_value = float((baseline_row.get("metrics") or {})[metric])
        gap = _worse_than_baseline(value, baseline_value, direction)
        gaps.append(gap)
        comparisons[baseline] = {
            "baseline_value": baseline_value,
            "worse_than_baseline": gap,
        }
    return {
        "sample_key": sample,
        "sample_id": row.get("sample_id"),
        "input_path": row.get("input_path"),
        "slices": sorted({str(item) for item in row.get("slices") or []}),
        metric: value,
        "badness": _badness(value, direction),
        "worst_gap": max(gaps) if gaps else None,
        "comparisons": comparisons,
    }


def _slice_summary(
    system_rows: dict[str, dict[str, Any]],
    rows_by_system: dict[str, dict[str, dict[str, Any]]],
    baselines: list[str],
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    slices = sorted({slice_name for row in system_rows.values() for slice_name in _row_slices(row)})
    summaries = []
    for slice_name in slices:
        keys = sorted(sample for sample, row in system_rows.items() if slice_name in _row_slices(row))
        system_values = [float((system_rows[key].get("metrics") or {})[metric]) for key in keys]
        item: dict[str, Any] = {
            "slice": slice_name,
            "sample_count": len(system_values),
            "system_mean": _mean(system_values),
            "system_max": max(system_values) if system_values else None,
        }
        baseline_items = {}
        for baseline in baselines:
            shared = [key for key in keys if key in rows_by_system.get(baseline, {})]
            if not shared:
                continue
            baseline_values = [float((rows_by_system[baseline][key].get("metrics") or {})[metric]) for key in shared]
            shared_system_values = [float((system_rows[key].get("metrics") or {})[metric]) for key in shared]
            baseline_items[baseline] = {
                "pairs": len(shared),
                "baseline_mean": _mean(baseline_values),
                "mean_worse_than_baseline": _worse_than_baseline(_mean(shared_system_values), _mean(baseline_values), direction),
            }
        item["baselines"] = baseline_items
        summaries.append(item)
    summaries.sort(key=lambda item: _badness(float(item["system_mean"]), direction), reverse=True)
    return summaries


def _baseline_summary(
    system_rows: dict[str, dict[str, Any]],
    rows_by_system: dict[str, dict[str, dict[str, Any]]],
    baselines: list[str],
    metric: str,
    direction: str,
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for baseline in baselines:
        shared = sorted(set(system_rows) & set(rows_by_system.get(baseline, {})))
        if not shared:
            continue
        system_values = [float((system_rows[key].get("metrics") or {})[metric]) for key in shared]
        baseline_values = [float((rows_by_system[baseline][key].get("metrics") or {})[metric]) for key in shared]
        gaps = [_worse_than_baseline(system_value, baseline_value, direction) for system_value, baseline_value in zip(system_values, baseline_values, strict=True)]
        summary[baseline] = {
            "pairs": len(shared),
            "system_mean": _mean(system_values),
            "baseline_mean": _mean(baseline_values),
            "mean_worse_than_baseline": _mean(gaps),
            "system_worse_count": sum(1 for gap in gaps if gap > 0),
            "system_better_count": sum(1 for gap in gaps if gap < 0),
            "tie_count": sum(1 for gap in gaps if gap == 0),
        }
    return summary


def _row_slices(row: dict[str, Any]) -> set[str]:
    values = {str(item) for item in row.get("slices") or [] if str(item)}
    values.add("all")
    return values


def _badness(value: float, direction: str) -> float:
    if direction == "lower":
        return value
    return -value


def _worse_than_baseline(system_value: float, baseline_value: float, direction: str) -> float:
    if direction == "lower":
        return system_value - baseline_value
    return baseline_value - system_value


def _mean(values: list[float]) -> float:
    if not values:
        raise BenchmarkError("cannot average an empty metric series")
    return sum(values) / len(values)


def _write_markdown(path: Path, payload: dict[str, Any], top_failures: list[dict[str, Any]]) -> None:
    metric = str(payload["metric"])
    lines = [
        "# Benchmark Error Analysis",
        "",
        f"Report: `{payload['benchmark_report']}`",
        f"System: `{payload['system']}`",
        f"Metric: `{metric}`",
        f"Direction: `{payload['direction']}`",
        f"Samples: `{payload['sample_count']}`",
        "",
        "## Baseline Summary",
        "",
        "| baseline | pairs | system_mean | baseline_mean | mean_worse_than_baseline | system_worse | system_better | ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_summary = payload.get("baseline_summary")
    if isinstance(baseline_summary, dict) and baseline_summary:
        for baseline, item in sorted(baseline_summary.items()):
            lines.append(
                "| {baseline} | {pairs} | {system_mean:.6f} | {baseline_mean:.6f} | {gap:.6f} | {worse} | {better} | {ties} |".format(
                    baseline=baseline,
                    pairs=int(item["pairs"]),
                    system_mean=float(item["system_mean"]),
                    baseline_mean=float(item["baseline_mean"]),
                    gap=float(item["mean_worse_than_baseline"]),
                    worse=int(item["system_worse_count"]),
                    better=int(item["system_better_count"]),
                    ties=int(item["tie_count"]),
                )
            )
    else:
        lines.append("| none | 0 |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Hardest Samples",
            "",
            f"| sample | slices | {metric} | worst_gap | input |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in top_failures:
        gap = item["worst_gap"]
        lines.append(
            "| {sample} | {slices} | {value:.6f} | {gap} | {input_path} |".format(
                sample=item.get("sample_id") or item["sample_key"],
                slices=",".join(item.get("slices") or []),
                value=float(item[metric]),
                gap=f"{float(gap):.6f}" if gap is not None else "",
                input_path=item.get("input_path") or "",
            )
        )
    top_failure_counts = payload.get("top_failure_slice_counts")
    lines.extend(["", "## Top Failure Slice Counts", "", "| slice | count |", "| --- | ---: |"])
    if isinstance(top_failure_counts, list) and top_failure_counts:
        for item in top_failure_counts:
            if isinstance(item, dict):
                lines.append(f"| {item.get('slice', '')} | {int(item.get('count') or 0)} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", f"Top failures JSON: `{payload['top_failures_path']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
