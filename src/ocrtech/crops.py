"""Extract recognizer training crops from rendered eval-pack references."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, sha256_path, sha256_text, write_manifest


SUPPORTED_CROP_TYPES = {"line", "table_cell", "figure_caption"}


@dataclass(slots=True)
class CropManifestSummary:
    manifest_path: str
    source_manifest: str
    crop_count: int
    source_sample_count: int
    crop_type_counts: dict[str, int]
    skipped_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "source_manifest": self.source_manifest,
            "crop_count": self.crop_count,
            "source_sample_count": self.source_sample_count,
            "crop_type_counts": self.crop_type_counts,
            "skipped_count": self.skipped_count,
            "warnings": self.warnings,
        }


def extract_crop_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    crop_types: list[str] | None = None,
    split: str = "train",
    dataset_name: str = "ocrtech-crops",
    min_text_length: int = 1,
) -> CropManifestSummary:
    if min_text_length < 1:
        raise DataValidationError("min_text_length must be at least 1")
    requested = [item.replace("-", "_") for item in (crop_types or ["line", "table_cell", "figure_caption"])]
    unknown = sorted(set(requested) - SUPPORTED_CROP_TYPES)
    if unknown:
        raise DataValidationError(f"unsupported crop types: {', '.join(unknown)}")
    source_entries = load_manifest(manifest_path)
    if not source_entries:
        raise DataValidationError("extract-crops requires a non-empty source manifest")

    out = Path(output_dir)
    images_dir = out / "images"
    manifests_dir = out / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    crop_entries: list[ManifestEntry] = []
    warnings: list[str] = []
    skipped_count = 0
    for source_entry in source_entries:
        reference_path = _reference_path(source_entry)
        reference = _load_reference(reference_path)
        layout = _reference_layout(reference, reference_path)
        figure_caption_lines = _figure_caption_line_ids(reference)
        page_by_index = _layout_pages(layout)
        if layout.get("unavailable_pages"):
            warnings.append(f"{source_entry.sample_id}: skipped unavailable layout pages {layout['unavailable_pages']}")
        for page_index, page_layout in page_by_index.items():
            page_image = _page_image_path(Path(source_entry.image_path), page_index)
            if not page_image.exists():
                raise DataValidationError(f"missing page image for {source_entry.sample_id} page {page_index}: {page_image}")
            page_crops = _page_crops(page_layout, requested, figure_caption_lines, min_text_length=min_text_length)
            for crop_index, crop in enumerate(page_crops):
                crop_path = images_dir / f"{source_entry.sample_id}-p{page_index:04d}-{crop['kind']}-{crop_index:04d}.png"
                _write_crop(page_image, crop_path, crop["bbox"])
                crop_text = str(crop["text"])
                crop_sha = sha256_path(crop_path)
                text_sha = sha256_text(crop_text)
                crop_entries.append(
                    ManifestEntry(
                        sample_id=f"{source_entry.sample_id}::{crop['kind']}::{page_index:04d}::{crop_index:04d}",
                        dataset=dataset_name,
                        split=split,
                        image_path=str(crop_path),
                        text=crop_text,
                        sha256=crop_sha,
                        metadata={
                            "slices": sorted({*(_entry_slices(source_entry)), "crop", str(crop["kind"])}),
                            "source_sample_id": source_entry.sample_id,
                            "source_manifest": str(manifest_path),
                            "source_image_path": source_entry.image_path,
                            "source_reference_path": str(reference_path),
                            "page_index": page_index,
                            "crop_type": crop["kind"],
                            "bbox": crop["bbox"],
                            "text_sha256": text_sha,
                            "sample_sha256": sha256_text(f"{crop_sha}\n{text_sha}"),
                        },
                    )
                )
            skipped_count += int(page_layout.get("skipped_crop_count", 0))

    if not crop_entries:
        raise DataValidationError("no crops extracted; source references may be missing supported layout metadata")
    manifest_out = manifests_dir / "crops.jsonl"
    write_manifest(crop_entries, manifest_out)
    counts = Counter(str(entry.metadata.get("crop_type")) for entry in crop_entries)
    summary = CropManifestSummary(
        manifest_path=str(manifest_out),
        source_manifest=str(manifest_path),
        crop_count=len(crop_entries),
        source_sample_count=len(source_entries),
        crop_type_counts=dict(sorted(counts.items())),
        skipped_count=skipped_count,
        warnings=warnings,
    )
    (out / "crop-manifest-summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(summary, out / "crop-manifest-summary.md")
    return summary


def _reference_path(entry: ManifestEntry) -> Path:
    value = entry.metadata.get("reference_path") if entry.metadata else None
    if not value:
        raise DataValidationError(f"missing reference_path metadata for {entry.sample_id}")
    path = Path(str(value))
    if not path.exists():
        raise DataValidationError(f"missing reference file for {entry.sample_id}: {path}")
    return path


def _load_reference(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid reference JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"reference must be an object: {path}")
    return payload


def _reference_layout(reference: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    metadata = reference.get("metadata")
    if not isinstance(metadata, dict):
        raise DataValidationError(f"reference metadata must be an object: {reference_path}")
    layout = metadata.get("layout")
    if not isinstance(layout, dict):
        raise DataValidationError(f"reference has no renderer layout metadata: {reference_path}")
    pages = layout.get("pages")
    if not isinstance(pages, list):
        raise DataValidationError(f"reference layout pages must be a list: {reference_path}")
    return layout


def _figure_caption_line_ids(reference: dict[str, Any]) -> set[str]:
    line_ids: set[str] = set()
    figures = reference.get("figures", [])
    if not isinstance(figures, list):
        raise DataValidationError("reference figures must be a list")
    for figure in figures:
        if not isinstance(figure, dict):
            raise DataValidationError("reference figure entries must be objects")
        caption = str(figure.get("caption") or "")
        for line_id in figure.get("line_ids", []):
            if caption and isinstance(line_id, str):
                line_ids.add(line_id)
    return line_ids


def _layout_pages(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for raw_page in layout.get("pages", []):
        if not isinstance(raw_page, dict):
            raise DataValidationError("layout page entries must be objects")
        page_index = raw_page.get("page_index")
        if not isinstance(page_index, int):
            raise DataValidationError("layout page_index must be an integer")
        pages[page_index] = raw_page
    return pages


def _page_image_path(source_path: Path, page_index: int) -> Path:
    if source_path.is_dir():
        return source_path / f"page-{page_index + 1:04d}.png"
    if page_index != 0:
        raise DataValidationError(f"single image source cannot provide page {page_index}: {source_path}")
    return source_path


def _page_crops(
    page_layout: dict[str, Any],
    crop_types: list[str],
    figure_caption_lines: set[str],
    *,
    min_text_length: int,
) -> list[dict[str, Any]]:
    crops: list[dict[str, Any]] = []
    skipped = 0
    if "line" in crop_types or "figure_caption" in crop_types:
        lines = page_layout.get("lines", [])
        if not isinstance(lines, list):
            raise DataValidationError("layout lines must be a list")
        for line in lines:
            if not isinstance(line, dict):
                raise DataValidationError("layout line entries must be objects")
            text = str(line.get("text") or "").strip()
            line_id = str(line.get("line_id") or "")
            if not text or len(text) < min_text_length or text.startswith("[Figure:"):
                skipped += 1
                continue
            if "line" in crop_types:
                crops.append({"kind": "line", "text": text, "bbox": _bbox(line.get("bbox"))})
            if "figure_caption" in crop_types and line_id in figure_caption_lines:
                crops.append({"kind": "figure_caption", "text": text, "bbox": _bbox(line.get("bbox"))})
    if "table_cell" in crop_types:
        cells = page_layout.get("table_cells", [])
        if not isinstance(cells, list):
            raise DataValidationError("layout table_cells must be a list")
        for cell in cells:
            if not isinstance(cell, dict):
                raise DataValidationError("layout table cell entries must be objects")
            text = str(cell.get("text") or "").strip()
            if not text or len(text) < min_text_length:
                skipped += 1
                continue
            crops.append({"kind": "table_cell", "text": text, "bbox": _bbox(cell.get("text_bbox") or cell.get("bbox"))})
    page_layout["skipped_crop_count"] = skipped
    return crops


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"crop bbox must be a four-item list, got: {value!r}")
    coordinates = [int(item) for item in value]
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise DataValidationError(f"invalid crop bbox: {coordinates}")
    return coordinates


def _write_crop(source_image: Path, crop_path: Path, bbox: list[int]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("crop extraction requires Pillow. Install ocr-tech[eval].") from exc
    with Image.open(source_image) as image:
        width, height = image.size
        x1, y1, x2, y2 = bbox
        clamped = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            raise DataValidationError(f"crop bbox outside image bounds for {source_image}: {bbox}")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(tuple(clamped)).save(crop_path)


def _entry_slices(entry: ManifestEntry) -> list[str]:
    value = entry.metadata.get("slices") if entry.metadata else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _write_summary_markdown(summary: CropManifestSummary, path: Path) -> None:
    lines = [
        "# Crop Manifest",
        "",
        f"Manifest: `{summary.manifest_path}`",
        f"Source manifest: `{summary.source_manifest}`",
        f"Source samples: `{summary.source_sample_count}`",
        f"Crops: `{summary.crop_count}`",
        f"Skipped candidates: `{summary.skipped_count}`",
        "",
        "## Crop Types",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.crop_type_counts.items())
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
