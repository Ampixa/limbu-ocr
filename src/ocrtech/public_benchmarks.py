"""Public benchmark manifest readiness checks."""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .datasets import entry_slices
from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, sha256_file, sha256_text, write_manifest
from .normalization import normalize_ocr_text
from .schemas import TableCell


@dataclass(frozen=True, slots=True)
class PublicBenchmarkSpec:
    dataset: str
    title: str
    url: str
    min_samples: int
    required_slices: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "title": self.title,
            "url": self.url,
            "min_samples": self.min_samples,
            "required_slices": list(self.required_slices),
            "notes": self.notes,
        }


PUBLIC_BENCHMARKS: dict[str, PublicBenchmarkSpec] = {
    "opendatalab/OmniDocBench": PublicBenchmarkSpec(
        dataset="opendatalab/OmniDocBench",
        title="OmniDocBench",
        url="https://huggingface.co/datasets/opendatalab/OmniDocBench",
        min_samples=100,
        required_slices=("public", "document_parsing"),
        notes="Real-world document parsing benchmark; use official annotations/scorer when available.",
    ),
    "allenai/olmOCR-bench": PublicBenchmarkSpec(
        dataset="allenai/olmOCR-bench",
        title="olmOCR-bench",
        url="https://huggingface.co/datasets/allenai/olmOCR-bench",
        min_samples=100,
        required_slices=("public", "pdf", "markdown"),
        notes="PDF-to-Markdown benchmark with unit-test style assertions.",
    ),
    "PaddlePaddle/Real5-OmniDocBench": PublicBenchmarkSpec(
        dataset="PaddlePaddle/Real5-OmniDocBench",
        title="Real5-OmniDocBench",
        url="https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench",
        min_samples=100,
        required_slices=("public", "real_world", "document_parsing"),
        notes="Physical reconstruction benchmark with scanning, warping, screen-photo, illumination, and skew scenarios.",
    ),
}


@dataclass(slots=True)
class PublicBenchmarkAudit:
    manifest_path: str
    passed: bool
    benchmark: str
    sample_count: int
    dataset_counts: dict[str, int]
    slice_counts: dict[str, int]
    spec: dict[str, Any] | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "passed": self.passed,
            "benchmark": self.benchmark,
            "sample_count": self.sample_count,
            "dataset_counts": self.dataset_counts,
            "slice_counts": self.slice_counts,
            "spec": self.spec,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class PublicBenchmarkPrepareReport:
    benchmark: str
    manifest_path: str
    references_dir: str
    sample_count: int
    filters: dict[str, str | int | bool | None]
    summary_json_path: str
    summary_md_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "manifest_path": self.manifest_path,
            "references_dir": self.references_dir,
            "sample_count": self.sample_count,
            "filters": self.filters,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


def list_public_benchmarks() -> list[PublicBenchmarkSpec]:
    return [PUBLIC_BENCHMARKS[key] for key in sorted(PUBLIC_BENCHMARKS)]


def prepare_public_benchmark(
    benchmark: str,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    language: str | None = None,
    subset: str | None = None,
    data_source: str | None = None,
    require_tables: bool = False,
    annotation_path: str | Path | None = None,
    local_image_root: str | Path | None = None,
) -> PublicBenchmarkPrepareReport:
    if benchmark != "opendatalab/OmniDocBench":
        raise DataValidationError(f"unsupported public benchmark converter: {benchmark}")
    return _prepare_omnidocbench(
        output_dir,
        limit=limit,
        language=language,
        subset=subset,
        data_source=data_source,
        require_tables=require_tables,
        annotation_path=annotation_path,
        local_image_root=local_image_root,
    )


def audit_public_benchmark_manifest(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    benchmark: str | None = None,
    min_samples: int | None = None,
    require_reference_paths: bool = True,
) -> PublicBenchmarkAudit:
    path = Path(manifest_path)
    entries = load_manifest(path)
    if not entries:
        raise DataValidationError("public benchmark audit requires a non-empty manifest")
    benchmark_name = benchmark or _infer_benchmark_name(entries)
    spec = PUBLIC_BENCHMARKS.get(benchmark_name)
    required_count = min_samples if min_samples is not None else (spec.min_samples if spec else 1)
    issues: list[str] = []
    warnings: list[str] = []
    dataset_counts: dict[str, int] = {}
    slice_counts: dict[str, int] = {}
    for entry in entries:
        dataset_counts[entry.dataset] = dataset_counts.get(entry.dataset, 0) + 1
        for slice_name in entry_slices(entry):
            slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
        metadata = entry.metadata or {}
        image_path = Path(entry.image_path)
        if not image_path.is_file():
            issues.append(f"{entry.sample_id}: image_path does not exist: {entry.image_path}")
        elif entry.sha256:
            actual_image_sha = sha256_file(image_path)
            if actual_image_sha != entry.sha256:
                issues.append(f"{entry.sample_id}: image sha256 mismatch: expected {entry.sha256}, actual {actual_image_sha}")
        text_sha = metadata.get("text_sha256")
        if isinstance(text_sha, str) and text_sha and text_sha != sha256_text(entry.text):
            issues.append(f"{entry.sample_id}: metadata.text_sha256 does not match entry text")
        if metadata.get("public_benchmark") is not True:
            issues.append(f"{entry.sample_id}: metadata.public_benchmark must be true")
        if not metadata.get("source_url"):
            issues.append(f"{entry.sample_id}: metadata.source_url is required")
        if not metadata.get("benchmark_name"):
            issues.append(f"{entry.sample_id}: metadata.benchmark_name is required")
        if require_reference_paths:
            reference_path = metadata.get("reference_path")
            if not isinstance(reference_path, str) or not reference_path:
                issues.append(f"{entry.sample_id}: metadata.reference_path is required")
            elif not Path(reference_path).exists():
                issues.append(f"{entry.sample_id}: reference_path does not exist: {reference_path}")
    if len(entries) < required_count:
        issues.append(f"sample_count {len(entries)} is below required minimum {required_count}")
    if spec is None:
        warnings.append(f"benchmark {benchmark_name!r} is not in the public benchmark registry")
    else:
        for required_slice in spec.required_slices:
            if slice_counts.get(required_slice, 0) == 0:
                issues.append(f"required public benchmark slice missing: {required_slice}")
    audit = PublicBenchmarkAudit(
        manifest_path=str(path),
        passed=not issues,
        benchmark=benchmark_name,
        sample_count=len(entries),
        dataset_counts=dict(sorted(dataset_counts.items())),
        slice_counts=dict(sorted(slice_counts.items())),
        spec=spec.to_dict() if spec else None,
        issues=issues,
        warnings=warnings,
    )
    if output_dir is not None:
        _write_public_benchmark_audit(audit, Path(output_dir))
    return audit


def _prepare_omnidocbench(
    output_dir: str | Path,
    *,
    limit: int | None,
    language: str | None,
    subset: str | None,
    data_source: str | None,
    require_tables: bool,
    annotation_path: str | Path | None,
    local_image_root: str | Path | None,
) -> PublicBenchmarkPrepareReport:
    if limit is not None and limit < 1:
        raise DataValidationError("limit must be positive when provided")
    benchmark = "opendatalab/OmniDocBench"
    out = Path(output_dir)
    manifest_dir = out / "manifests"
    image_dir = out / "images"
    reference_dir = out / "references"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    rows, repo_files, download_file = _load_omnidocbench_source(annotation_path, local_image_root)
    entries: list[ManifestEntry] = []
    for row_index, row in enumerate(rows):
        if limit is not None and len(entries) >= limit:
            break
        if not isinstance(row, dict):
            raise DataValidationError(f"OmniDocBench row {row_index} must be an object")
        page_info = row.get("page_info")
        if not isinstance(page_info, dict):
            raise DataValidationError(f"OmniDocBench row {row_index} is missing page_info")
        attributes = page_info.get("page_attribute")
        if not isinstance(attributes, dict):
            raise DataValidationError(f"OmniDocBench row {row_index} is missing page_info.page_attribute")
        if language and str(attributes.get("language")) != language:
            continue
        if subset and str(attributes.get("subset")) != subset:
            continue
        if data_source and str(attributes.get("data_source")) != data_source:
            continue
        layout_dets = row.get("layout_dets")
        if not isinstance(layout_dets, list):
            raise DataValidationError(f"OmniDocBench row {row_index} is missing layout_dets")
        tables = _omnidocbench_reference_tables(layout_dets)
        if require_tables and not tables:
            continue
        image_name = page_info.get("image_path")
        if not isinstance(image_name, str) or not image_name:
            raise DataValidationError(f"OmniDocBench row {row_index} page_info.image_path must be a non-empty string")
        dataset_image_path = image_name if image_name in repo_files else f"images/{image_name}"
        if dataset_image_path not in repo_files:
            raise DataValidationError(f"OmniDocBench row {row_index} image file is missing from dataset repo: {dataset_image_path}")
        source_image = download_file(dataset_image_path)
        target_image = image_dir / Path(dataset_image_path).name
        if source_image.resolve() != target_image.resolve():
            shutil.copy2(source_image, target_image)
        sample_id = f"omnidocbench-{row_index:05d}-{Path(image_name).stem}"
        reference_path = reference_dir / f"{sample_id}.ref.json"
        reference_text = _omnidocbench_reference_text(layout_dets)
        reference_payload = {
            "text": reference_text,
            "tables": tables,
            "metadata": {
                "benchmark_name": benchmark,
                "source_image_path": dataset_image_path,
                "page_info": page_info,
            },
        }
        reference_path.write_text(json.dumps(reference_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        image_hash = sha256_file(target_image)
        text_hash = sha256_text(reference_text)
        slices = _omnidocbench_slices(attributes, tables=tables, layout_dets=layout_dets)
        entries.append(
            ManifestEntry(
                sample_id=sample_id,
                dataset=benchmark,
                split="eval",
                image_path=str(target_image),
                text=reference_text,
                sha256=image_hash,
                metadata={
                    "benchmark_name": benchmark,
                    "public_benchmark": True,
                    "source_url": PUBLIC_BENCHMARKS[benchmark].url,
                    "source_image_path": dataset_image_path,
                    "reference_path": str(reference_path),
                    "slices": slices,
                    "text_sha256": text_hash,
                    "sample_sha256": sha256_text(f"{image_hash}\n{text_hash}"),
                    "page_attribute": attributes,
                    "page_no": page_info.get("page_no"),
                    "width": page_info.get("width"),
                    "height": page_info.get("height"),
                },
            )
        )
    if not entries:
        raise DataValidationError("No OmniDocBench rows matched the requested filters")
    manifest_path = manifest_dir / "omnidocbench-eval.jsonl"
    write_manifest(entries, manifest_path)
    report = PublicBenchmarkPrepareReport(
        benchmark=benchmark,
        manifest_path=str(manifest_path),
        references_dir=str(reference_dir),
        sample_count=len(entries),
        filters={
            "limit": limit,
            "language": language,
            "subset": subset,
            "data_source": data_source,
            "require_tables": require_tables,
        },
        summary_json_path=str(out / "public-benchmark-prepare.json"),
        summary_md_path=str(out / "public-benchmark-prepare.md"),
    )
    _write_public_benchmark_prepare_report(report)
    return report


def _load_omnidocbench_source(
    annotation_path: str | Path | None,
    local_image_root: str | Path | None,
) -> tuple[list[dict[str, Any]], set[str], Any]:
    if annotation_path is not None:
        annotation = Path(annotation_path)
        if not annotation.exists():
            raise DataValidationError(f"OmniDocBench annotation path does not exist: {annotation}")
        root = Path(local_image_root) if local_image_root is not None else annotation.parent
        files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}

        def local_download(name: str) -> Path:
            path = root / name
            if not path.exists():
                raise DataValidationError(f"OmniDocBench local image file missing: {path}")
            return path

        rows = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise DataValidationError("OmniDocBench annotation JSON must be a list")
        return rows, files, local_download
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise DataValidationError("OmniDocBench preparation requires huggingface_hub. Install ocr-tech[datasets].") from exc
    benchmark = "opendatalab/OmniDocBench"
    try:
        annotation = Path(hf_hub_download(repo_id=benchmark, repo_type="dataset", filename="OmniDocBench.json"))
        files = set(list_repo_files(repo_id=benchmark, repo_type="dataset"))
    except Exception as exc:
        raise DataValidationError(f"Failed to inspect OmniDocBench dataset repo: {exc}") from exc

    def hf_download(name: str) -> Path:
        try:
            return Path(hf_hub_download(repo_id=benchmark, repo_type="dataset", filename=name))
        except Exception as exc:
            raise DataValidationError(f"Failed to download OmniDocBench file {name!r}: {exc}") from exc

    rows = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise DataValidationError("OmniDocBench annotation JSON must be a list")
    return rows, files, hf_download


def _omnidocbench_reference_text(layout_dets: list[Any]) -> str:
    parts: list[str] = []
    for item in sorted(_valid_layout_items(layout_dets), key=lambda value: int(value.get("order", 0) or 0)):
        category = str(item.get("category_type") or "")
        if category == "table" and isinstance(item.get("html"), str):
            text = _html_to_text(str(item["html"]))
        else:
            text = str(item.get("text") or "")
        text = normalize_ocr_text(text, collapse_spaces=True).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _omnidocbench_reference_tables(layout_dets: list[Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for item in sorted(_valid_layout_items(layout_dets), key=lambda value: int(value.get("order", 0) or 0)):
        if item.get("category_type") != "table" or not isinstance(item.get("html"), str):
            continue
        cells = _table_cells_from_html(str(item["html"]))
        if cells:
            tables.append(
                {
                    "cells": [cell.to_dict() for cell in cells],
                    "metadata": {"anno_id": item.get("anno_id"), "order": item.get("order")},
                }
            )
    return tables


def _valid_layout_items(layout_dets: list[Any]) -> list[dict[str, Any]]:
    return [item for item in layout_dets if isinstance(item, dict) and not bool(item.get("ignore", False))]


def _omnidocbench_slices(attributes: dict[str, Any], *, tables: list[dict[str, Any]], layout_dets: list[Any]) -> list[str]:
    slices = ["public", "document_parsing"]
    language = str(attributes.get("language") or "")
    if language:
        slices.append(language)
    if language == "english":
        slices.append("english")
    if "chinese" in language:
        slices.append("chinese")
    subset = str(attributes.get("subset") or "")
    if subset:
        slices.append(subset)
    source = str(attributes.get("data_source") or "")
    if source:
        slices.append(source)
    categories = {str(item.get("category_type") or "") for item in _valid_layout_items(layout_dets)}
    if tables:
        slices.append("table")
    if any("figure" in category for category in categories):
        slices.append("figure")
    if any("equation" in category for category in categories):
        slices.append("equation")
    return sorted(set(slices))


class _PlainHtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"tr", "br", "p", "div"}:
            self.parts.append("\n")
        if tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", html.unescape("".join(self.parts)))).strip()
        return "\n".join(line.strip() for line in value.splitlines() if line.strip())


def _html_to_text(value: str) -> str:
    parser = _PlainHtmlTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: list[TableCell] = []
        self._row = -1
        self._col = 0
        self._occupied: set[tuple[int, int]] = set()
        self._cell_text: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row += 1
            self._col = 0
            return
        if tag not in {"td", "th"} or self._row < 0:
            return
        while (self._row, self._col) in self._occupied:
            self._col += 1
        attrs_dict = {key.lower(): value for key, value in attrs}
        self._rowspan = _positive_int(attrs_dict.get("rowspan"), default=1)
        self._colspan = _positive_int(attrs_dict.get("colspan"), default=1)
        self._cell_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag not in {"td", "th"} or self._cell_text is None:
            return
        text = normalize_ocr_text(html.unescape("".join(self._cell_text)), collapse_spaces=True).strip()
        self.cells.append(TableCell(row=self._row, col=self._col, text=text, rowspan=self._rowspan, colspan=self._colspan))
        for row_offset in range(self._rowspan):
            for col_offset in range(self._colspan):
                if row_offset or col_offset:
                    self._occupied.add((self._row + row_offset, self._col + col_offset))
        self._col += self._colspan
        self._cell_text = None
        self._rowspan = 1
        self._colspan = 1

    def handle_data(self, data: str) -> None:
        if self._cell_text is not None:
            self._cell_text.append(data)


def _positive_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _table_cells_from_html(value: str) -> list[TableCell]:
    parser = _TableHtmlParser()
    parser.feed(value)
    parser.close()
    return parser.cells


def _infer_benchmark_name(entries: list[ManifestEntry]) -> str:
    names = sorted({str(entry.metadata.get("benchmark_name") or entry.dataset) for entry in entries})
    if len(names) != 1:
        raise DataValidationError(f"public benchmark manifest must contain exactly one benchmark, got: {names}")
    return names[0]


def _write_public_benchmark_audit(audit: PublicBenchmarkAudit, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "public-benchmark-audit.json"
    md_path = out / "public-benchmark-audit.md"
    audit.summary_json_path = str(json_path)
    audit.summary_md_path = str(md_path)
    json_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_public_benchmark_audit(audit), encoding="utf-8")


def _render_public_benchmark_audit(audit: PublicBenchmarkAudit) -> str:
    lines = [
        "# Public Benchmark Audit",
        "",
        f"Status: `{'pass' if audit.passed else 'fail'}`",
        f"Benchmark: `{audit.benchmark}`",
        f"Samples: `{audit.sample_count}`",
        "",
        "## Datasets",
        "",
    ]
    for dataset, count in audit.dataset_counts.items():
        lines.append(f"- `{dataset}`: `{count}`")
    lines.extend(["", "## Slices", ""])
    for slice_name, count in audit.slice_counts.items():
        lines.append(f"- `{slice_name}`: `{count}`")
    if audit.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit.issues)
    if audit.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.warnings)
    return "\n".join(lines) + "\n"


def _write_public_benchmark_prepare_report(report: PublicBenchmarkPrepareReport) -> None:
    json_path = Path(report.summary_json_path)
    md_path = Path(report.summary_md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_public_benchmark_prepare_report(report), encoding="utf-8")


def _render_public_benchmark_prepare_report(report: PublicBenchmarkPrepareReport) -> str:
    lines = [
        "# Public Benchmark Prepare Report",
        "",
        f"Benchmark: `{report.benchmark}`",
        f"Samples: `{report.sample_count}`",
        f"Manifest: `{report.manifest_path}`",
        f"References: `{report.references_dir}`",
        "",
        "## Filters",
        "",
    ]
    for name, value in report.filters.items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"
