"""Generate and audit held-out evaluation packs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, sha256_path, sha256_text, write_manifest
from .references import load_reference


CLAIM_REQUIRED_SLICES = {
    "nepali",
    "english",
    "mixed_script",
    "multi_page",
    "table",
    "complex_table",
    "reading_order",
    "form",
    "figure",
    "scan",
    "blur",
    "low_contrast",
    "skew",
}


@dataclass(slots=True)
class PageTemplate:
    name: str
    pages: list[list[str]]
    reference_text: str
    reading_order: list[str]
    tables: list[list[list[str]]]
    figures: list[dict[str, Any]]
    slices: list[str]
    document_type: str
    reference_tables: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class RenderedInput:
    path: Path
    layout: dict[str, Any] | None


@dataclass(slots=True)
class EvalPackSummary:
    manifest_path: str
    sample_count: int
    input_format: str
    slices: dict[str, int]
    claim_ready: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "input_format": self.input_format,
            "slices": self.slices,
            "claim_ready": self.claim_ready,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class EvalPackAudit:
    passed: bool
    claim_ready: bool
    sample_count: int
    slices: dict[str, int]
    reference_counts: dict[str, int]
    issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "claim_ready": self.claim_ready,
            "sample_count": self.sample_count,
            "slices": self.slices,
            "reference_counts": self.reference_counts,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def create_eval_pack(
    output_dir: str | Path,
    *,
    count_per_template: int = 1,
    input_format: str = "text",
    degradations: list[str] | None = None,
    seed: int = 13,
    font_path: str | Path | None = None,
    templates: list[str] | None = None,
    variant_offset: int = 0,
) -> EvalPackSummary:
    if count_per_template < 1:
        raise DataValidationError("count_per_template must be at least 1")
    if variant_offset < 0:
        raise DataValidationError("variant_offset must be non-negative")
    if input_format not in {"text", "image"}:
        raise DataValidationError("input_format must be text or image")
    degradation_names = degradations or ["clean"]
    unknown = sorted(set(degradation_names) - {"clean", "scan", "blur", "low_contrast", "skew"})
    if unknown:
        raise DataValidationError(f"unsupported degradations: {unknown}")
    out = Path(output_dir)
    inputs_dir = out / "inputs"
    refs_dir = out / "references"
    manifests_dir = out / "manifests"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    entries: list[ManifestEntry] = []
    selected_templates = _select_templates(templates)
    for template in selected_templates:
        for copy_index in range(count_per_template):
            variant_index = variant_offset + copy_index
            materialized = _materialize_template(template, variant_index)
            for degradation in degradation_names:
                sample_id = f"eval-{template.name}-{variant_index:03d}-{degradation}"
                rendered_input = _write_input(
                    inputs_dir,
                    sample_id,
                    materialized,
                    input_format=input_format,
                    degradation=degradation,
                    rng=rng,
                    font_path=Path(font_path) if font_path else None,
                )
                input_path = rendered_input.path
                reference_path = refs_dir / f"{sample_id}.json"
                _write_reference(reference_path, materialized, layout=rendered_input.layout)
                image_sha256 = sha256_path(input_path)
                text_sha256 = sha256_text(materialized.reference_text)
                slices = sorted({*materialized.slices, degradation, input_format})
                entries.append(
                    ManifestEntry(
                        sample_id=sample_id,
                        dataset="ocrtech-eval-pack",
                        split="eval",
                        image_path=str(input_path),
                        text=materialized.reference_text,
                        sha256=image_sha256,
                        metadata={
                            "slices": slices,
                            "script": "mixed_script" if "mixed_script" in slices else materialized.slices[0],
                            "document_type": materialized.document_type,
                            "degradation": degradation,
                            "input_format": input_format,
                            "reference_path": str(reference_path),
                            "text_sha256": text_sha256,
                            "sample_sha256": sha256_text(f"{image_sha256}\n{text_sha256}"),
                            "claim_evidence_eligible": input_format == "image",
                        },
                    )
                )
    manifest_path = manifests_dir / "ocrtech-eval-pack.jsonl"
    write_manifest(entries, manifest_path)
    summary = _summary(manifest_path, entries, input_format)
    (out / "eval-pack.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(summary, out / "eval-pack.md")
    return summary


def audit_eval_pack(manifest_path: str | Path, output_dir: str | Path | None = None) -> EvalPackAudit:
    entries = load_manifest(manifest_path)
    issues: list[str] = []
    warnings: list[str] = []
    if not entries:
        issues.append("manifest has no entries")
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    slices: dict[str, int] = {}
    reference_counts = {"parsed": 0, "text": 0, "markdown": 0, "reading_order": 0, "tables": 0, "figures": 0}
    input_formats: set[str] = set()
    manifest_dir = Path(manifest_path).parent
    for entry in entries:
        if entry.sample_id in sample_ids:
            issues.append(f"duplicate sample_id: {entry.sample_id}")
        sample_ids.add(entry.sample_id)
        if entry.sha256:
            if entry.sha256 in image_hashes:
                warnings.append(f"duplicate input sha256: {entry.sha256}")
            image_hashes.add(entry.sha256)
        input_path = Path(entry.image_path)
        if not input_path.exists():
            issues.append(f"missing input file for {entry.sample_id}: {input_path}")
        reference_path = entry.metadata.get("reference_path") if entry.metadata else None
        if not reference_path:
            issues.append(f"missing reference_path metadata for {entry.sample_id}")
        else:
            resolved_reference = _resolve_manifest_path(str(reference_path), manifest_dir)
            if not resolved_reference.exists():
                issues.append(f"missing reference file for {entry.sample_id}: {reference_path}")
            else:
                try:
                    reference = load_reference(input_path, explicit_path=resolved_reference)
                except Exception as exc:
                    issues.append(f"invalid reference for {entry.sample_id}: {exc}")
                else:
                    if reference is None:
                        issues.append(f"reference did not load for {entry.sample_id}: {reference_path}")
                    else:
                        reference_counts["parsed"] += 1
                        if reference.text is not None:
                            reference_counts["text"] += 1
                        if reference.markdown is not None:
                            reference_counts["markdown"] += 1
                        if reference.reading_order:
                            reference_counts["reading_order"] += 1
                        if reference.tables:
                            reference_counts["tables"] += 1
                        if reference.figures:
                            reference_counts["figures"] += 1
        input_format = str(entry.metadata.get("input_format", "")) if entry.metadata else ""
        if input_format:
            input_formats.add(input_format)
        for slice_name in _entry_slices(entry):
            slices[slice_name] = slices.get(slice_name, 0) + 1
    missing_slices = sorted(CLAIM_REQUIRED_SLICES - set(slices))
    if missing_slices:
        warnings.append(f"missing claim slices: {', '.join(missing_slices)}")
    if input_formats != {"image"}:
        warnings.append("pack is not claim-ready for OCR SOTA unless every input is a rendered image")
    claim_ready = not issues and not missing_slices and input_formats == {"image"}
    audit = EvalPackAudit(
        passed=not issues,
        claim_ready=claim_ready,
        sample_count=len(entries),
        slices=dict(sorted(slices.items())),
        reference_counts=reference_counts,
        issues=issues,
        warnings=warnings,
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval-pack-audit.json").write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_audit_markdown(audit, out / "eval-pack-audit.md")
    return audit


def _summary(manifest_path: Path, entries: list[ManifestEntry], input_format: str) -> EvalPackSummary:
    slices: dict[str, int] = {}
    for entry in entries:
        for slice_name in _entry_slices(entry):
            slices[slice_name] = slices.get(slice_name, 0) + 1
    missing = sorted(CLAIM_REQUIRED_SLICES - set(slices))
    warnings: list[str] = []
    if input_format != "image":
        warnings.append("text input packs are for pipeline smoke tests, not OCR SOTA evidence")
    if missing:
        warnings.append(f"missing claim slices: {', '.join(missing)}")
    return EvalPackSummary(
        manifest_path=str(manifest_path),
        sample_count=len(entries),
        input_format=input_format,
        slices=dict(sorted(slices.items())),
        claim_ready=input_format == "image" and not missing,
        warnings=warnings,
    )


def _write_reference(path: Path, template: PageTemplate, *, layout: dict[str, Any] | None = None) -> None:
    payload = {
        "text": template.reference_text,
        "reading_order": template.reading_order,
        "tables": template.reference_tables or template.tables,
        "figures": template.figures,
        "metadata": {
            "template": template.name,
            "document_type": template.document_type,
            "slices": template.slices,
            "page_count": len(template.pages),
        },
    }
    if layout is not None:
        payload["metadata"]["layout"] = layout
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_input(
    inputs_dir: Path,
    sample_id: str,
    template: PageTemplate,
    *,
    input_format: str,
    degradation: str,
    rng: random.Random,
    font_path: Path | None,
) -> RenderedInput:
    if input_format == "text":
        return RenderedInput(path=_write_text_sidecar(inputs_dir, sample_id, template), layout=None)
    if len(template.pages) == 1:
        path = inputs_dir / f"{sample_id}.png"
        page_layout = _render_page_image(path, template, page_index=0, degradation=degradation, rng=rng, font_path=font_path)
        layout = _render_layout([page_layout], degradation=degradation)
        return RenderedInput(path=path, layout=layout)
    bundle_dir = inputs_dir / sample_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    page_layouts: list[dict[str, Any] | None] = []
    for page_index, _lines in enumerate(template.pages):
        page_path = bundle_dir / f"page-{page_index + 1:04d}.png"
        page_layouts.append(_render_page_image(page_path, template, page_index=page_index, degradation=degradation, rng=rng, font_path=font_path))
    return RenderedInput(path=bundle_dir, layout=_render_layout(page_layouts, degradation=degradation))


def _write_text_sidecar(inputs_dir: Path, sample_id: str, template: PageTemplate) -> Path:
    if len(template.pages) == 1:
        path = inputs_dir / f"{sample_id}.txt"
        path.write_text("\n".join(template.pages[0]) + "\n", encoding="utf-8")
        return path
    bundle_dir = inputs_dir / sample_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pages": [
            {
                "lines": [
                    {"text": line, "bbox": [0, 40 * line_index, max(50, len(line) * 9), 24], "line_id": f"p{page_index}-l{line_index}"}
                    for line_index, line in enumerate(lines)
                ],
                "metadata": {"template": template.name, "page_index": page_index},
            }
            for page_index, lines in enumerate(template.pages)
        ],
        "tables": [],
        "figures": [],
        "metadata": {"template": template.name, "document_type": template.document_type},
    }
    (bundle_dir / "document.ocr.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bundle_dir


def _render_page_image(
    path: Path,
    template: PageTemplate,
    *,
    page_index: int,
    degradation: str,
    rng: random.Random,
    font_path: Path | None,
) -> dict[str, Any] | None:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    except ImportError as exc:
        raise DataValidationError("image eval packs require Pillow. Install ocr-tech[eval].") from exc
    width = 1240
    line_height = 42
    margin = 72
    lines = template.pages[page_index]
    table_grid = _page_table_grid(template, page_index) or []
    prose_lines = list(lines)
    if table_grid:
        prose_lines = lines[: max(0, len(lines) - len(table_grid))]
    height = max(900, margin * 2 + line_height * (len(prose_lines) + len(table_grid) + 5))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(ImageFont, font_path, size=28)
    caption_font = _load_font(ImageFont, font_path, size=24)
    line_layout: list[dict[str, Any]] = []
    table_cell_layout: list[dict[str, Any]] = []
    y = margin
    for line_index, line in enumerate(prose_lines):
        draw.text((margin, y), line, fill=(20, 20, 20), font=font)
        line_layout.append(
            {
                "line_id": f"p{page_index}-l{line_index}",
                "text": line,
                "bbox": _padded_text_bbox(draw, (margin, y), line, font, image_size=(width, height)),
            }
        )
        y += line_height
    if table_grid:
        y += 12
        y, table_cell_layout = _draw_table_grid(draw, table_grid, x=margin, y=y, font=caption_font, page_width=width - margin * 2)
    layout: dict[str, Any] | None = {
        "page_index": page_index,
        "width": width,
        "height": height,
        "lines": line_layout,
        "table_cells": table_cell_layout,
    }
    if degradation == "scan":
        image = ImageEnhance.Contrast(image).enhance(0.82)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    elif degradation == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
    elif degradation == "low_contrast":
        image = ImageEnhance.Contrast(image).enhance(0.55)
    elif degradation == "skew":
        image = image.rotate(rng.uniform(-2.0, 2.0), expand=True, fillcolor="white")
        layout = None
    image.save(path)
    return layout


def _render_layout(page_layouts: list[dict[str, Any] | None], *, degradation: str) -> dict[str, Any]:
    pages = [page_layout for page_layout in page_layouts if page_layout is not None]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "ocrtech.evalpack",
        "degradation": degradation,
        "pages": pages,
    }
    if len(pages) != len(page_layouts):
        payload["unavailable_pages"] = [
            index for index, page_layout in enumerate(page_layouts) if page_layout is None
        ]
        payload["unavailable_reason"] = "skew degradation rotates rendered pixels; pre-rotation bboxes are not valid"
    return payload


def _page_table_grid(template: PageTemplate, page_index: int) -> list[list[str]] | None:
    if not template.tables:
        return None
    if template.document_type == "table" and page_index == 0:
        return template.tables[0]
    if template.document_type == "multi_page_report" and page_index == 0:
        return template.tables[0]
    return None


def _draw_table_grid(draw: Any, grid: list[list[str]], *, x: int, y: int, font: Any, page_width: int) -> tuple[int, list[dict[str, Any]]]:
    row_count = len(grid)
    col_count = max((len(row) for row in grid), default=0)
    if row_count == 0 or col_count == 0:
        return y, []
    row_height = 56
    col_width = max(160, page_width // col_count)
    table_width = col_width * col_count
    table_height = row_height * row_count
    cells: list[dict[str, Any]] = []
    for row_index in range(row_count + 1):
        line_y = y + row_index * row_height
        draw.line((x, line_y, x + table_width, line_y), fill=(80, 80, 80), width=2 if row_index in {0, 1, row_count} else 1)
    for col_index in range(col_count + 1):
        line_x = x + col_index * col_width
        draw.line((line_x, y, line_x, y + table_height), fill=(80, 80, 80), width=2 if col_index in {0, col_count} else 1)
    for row_index, row in enumerate(grid):
        for col_index in range(col_count):
            value = row[col_index] if col_index < len(row) else ""
            cell_x1 = x + col_index * col_width
            cell_y1 = y + row_index * row_height
            cell_x2 = cell_x1 + col_width
            cell_y2 = cell_y1 + row_height
            text_xy = (cell_x1 + 12, cell_y1 + 12)
            draw.text(text_xy, value, fill=(20, 20, 20), font=font)
            cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "text": value,
                    "bbox": [cell_x1 + 2, cell_y1 + 2, cell_x2 - 2, cell_y2 - 2],
                    "text_bbox": _padded_text_bbox(draw, text_xy, value, font, image_size=(x + page_width, y + table_height)),
                }
            )
    return y + table_height, cells


def _padded_text_bbox(draw: Any, xy: tuple[int, int], text: str, font: Any, *, image_size: tuple[int, int]) -> list[int]:
    if text:
        bbox = draw.textbbox(xy, text, font=font)
    else:
        x, y = xy
        bbox = (x, y, x + 1, y + 1)
    x1, y1, x2, y2 = bbox
    width, height = image_size
    padding = 6
    return [
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(width, int(x2) + padding),
        min(height, int(y2) + padding),
    ]


def _load_font(ImageFont: Any, font_path: Path | None, *, size: int) -> Any:
    candidates: list[Path] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"),
            Path("/System/Library/Fonts/Supplemental/Kohinoor Devanagari.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _entry_slices(entry: ManifestEntry) -> list[str]:
    value = entry.metadata.get("slices") if entry.metadata else []
    if isinstance(value, list):
        return sorted(str(item) for item in value if str(item))
    if isinstance(value, str) and value:
        return [value]
    return []


def _resolve_manifest_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def _write_summary_markdown(summary: EvalPackSummary, path: Path) -> None:
    lines = [
        "# OCR Eval Pack",
        "",
        f"Manifest: `{summary.manifest_path}`",
        f"Samples: `{summary.sample_count}`",
        f"Input format: `{summary.input_format}`",
        f"Claim ready: `{'yes' if summary.claim_ready else 'no'}`",
        "",
        "## Slices",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.slices.items())
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary.warnings) if summary.warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_markdown(audit: EvalPackAudit, path: Path) -> None:
    lines = [
        "# Eval Pack Audit",
        "",
        f"Passed: `{'yes' if audit.passed else 'no'}`",
        f"Claim ready: `{'yes' if audit.claim_ready else 'no'}`",
        f"Samples: `{audit.sample_count}`",
        "",
        "## References",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in audit.reference_counts.items())
    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    lines.extend(f"- {issue}" for issue in audit.issues) if audit.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in audit.warnings) if audit.warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_template(template: PageTemplate, copy_index: int) -> PageTemplate:
    if copy_index == 0:
        return template
    pools = _variant_values(copy_index)
    text = _render_variant_text(template.reference_text, pools)
    pages = [[_render_variant_text(line, pools) for line in page] for page in template.pages]
    tables = [[[_render_variant_text(cell, pools) for cell in row] for row in table] for table in template.tables]
    reference_tables = _render_reference_tables(template.reference_tables, pools)
    figures = [
        {
            **figure,
            "caption": _render_variant_text(str(figure.get("caption") or ""), pools),
        }
        for figure in template.figures
    ]
    return replace(template, pages=pages, reference_text=text, tables=tables, reference_tables=reference_tables, figures=figures)


def _render_variant_text(text: str, pools: dict[str, str]) -> str:
    rendered = text
    for key, value in pools.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _render_reference_tables(value: Any, pools: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _render_variant_text(value, pools)
    if isinstance(value, list):
        return [_render_reference_tables(item, pools) for item in value]
    if isinstance(value, dict):
        return {key: _render_reference_tables(item, pools) for key, item in value.items()}
    return value


def _variant_values(copy_index: int) -> dict[str, str]:
    departments = ["Department of Records", "Department of Revenue", "Central Archives Office", "Land Survey Office"]
    english_lines = [
        "The quick brown fox jumps over the lazy dog.",
        "Please verify the attached procurement details.",
        "This report mixes structured fields with running text.",
        "Payment clearance requires dual-language review.",
    ]
    nepali_lines = [
        "यो कागजात नेपाली र English दुबै भाषामा लेखिएको छ।",
        "यो प्रतिवेदनमा मिश्रित लिपि र तालिका दुबै समावेश छन्।",
        "दुवै भाषाको शुद्धता छुट्टै जाँच गर्नुपर्ने छ।",
        "यो सूचना प्रणालीले पढ्ने क्रममा स्थिरता देखाउनुपर्छ।",
    ]
    amounts = ["१२,५००.००", "१८,७५०.००", "९,४००.००", "२२,१००.००"]
    iso_dates = ["2026-06-14", "2026-07-01", "2026-07-15", "2026-08-03"]
    nepali_dates = ["२०८३-०२-०१", "२०८३-०२-१५", "२०८३-०३-०५", "२०८३-०३-२०"]
    form_names = ["आशा गुरुङ", "नवीन श्रेष्ठ", "सविता केसी", "रमेश अधिकारी"]
    addresses = ["Pokhara-08", "Kathmandu-11", "Lalitpur-05", "Biratnagar-03"]
    vendors_a = ["Himalayan Trade", "Everest Supply", "Kathmandu Medical", "Janaki Logistics"]
    vendors_b = ["Pokhara Supply", "Bharatpur Health", "Mithila Pharma", "Seti Traders"]
    approvers = ["Sita Rana", "Nabin Karki", "Asha Shrestha", "Ramesh Thapa"]
    ministries = ["Ministry of Health", "Ministry of Education", "Ministry of Finance", "Ministry of Agriculture"]
    fiscal_years = ["FY 2025/26", "FY 2026/27", "FY 2024/25", "FY 2023/24"]
    figure_year_ranges = ["2022 to 2026", "2021 to 2025", "2020 to 2024", "2019 to 2023"]
    figure_summaries = [
        "राजस्व क्रमशः बढेको छ।",
        "अन्तिम दुई वर्षमा वृद्धिदर तीव्र भएको छ।",
        "मध्यम अवधिमा स्थिर वृद्धि देखिएको छ।",
        "समीक्षा अवधिमा उतारचढाव भए पनि समग्र प्रवृत्ति सकारात्मक छ।",
    ]
    qty_masks = ["५००", "६५०", "७२०", "८००"]
    qty_gloves = ["८००", "९५०", "११००", "१२५०"]
    ledger_name_a = ["राम", "हरी", "किरण", "सोम"]
    ledger_name_b = ["सीता", "गीता", "माया", "रीता"]
    ledger_amount_a = ["१५००", "१८००", "२१००", "२४००"]
    ledger_amount_b = ["२३००", "२६००", "२९००", "३२००"]
    pending_words = ["Pending", "Cleared", "Review", "Approved"]
    variant = copy_index % 4
    generation = copy_index // 4
    amount_delta = generation * 137
    ledger_a_value = int(ledger_amount_a[variant]) + amount_delta
    ledger_b_value = int(ledger_amount_b[variant]) + amount_delta
    total_amount = str(ledger_a_value + ledger_b_value)
    return {
        "department": f"{departments[variant]} Unit {copy_index:03d}",
        "english_line": english_lines[variant],
        "nepali_line": nepali_lines[variant],
        "amount": _amount_variant(amounts[variant], amount_delta),
        "iso_date": _date_variant(iso_dates[variant], generation),
        "nepali_date": _nepali_date_variant(nepali_dates[variant], generation),
        "form_name": f"{form_names[variant]} {copy_index:03d}",
        "address": addresses[variant],
        "vendor_a": f"{vendors_a[variant]}-{copy_index:03d}",
        "vendor_b": f"{vendors_b[variant]}-{copy_index:03d}",
        "approver": approvers[variant],
        "ministry": ministries[variant],
        "fiscal_year": f"{fiscal_years[variant]} Batch {copy_index:03d}",
        "figure_year_range": f"{figure_year_ranges[variant]} batch {copy_index:03d}",
        "figure_summary": f"{figure_summaries[variant]} Batch {copy_index:03d}.",
        "qty_masks": _nepali_digits(int(qty_masks[variant]) + amount_delta),
        "qty_gloves": _nepali_digits(int(qty_gloves[variant]) + amount_delta),
        "ledger_name_a": f"{ledger_name_a[variant]}-{copy_index:03d}",
        "ledger_name_b": f"{ledger_name_b[variant]}-{copy_index:03d}",
        "ledger_amount_a": _nepali_digits(ledger_a_value),
        "ledger_amount_b": _nepali_digits(ledger_b_value),
        "ledger_total": total_amount,
        "pending_word": pending_words[variant],
        "phone": f"98{variant + 1}000000{variant}",
    }


def _amount_variant(amount: str, delta: int) -> str:
    if not delta:
        return amount
    normalized = amount.replace(",", "").replace("०", "0").replace("१", "1").replace("२", "2").replace("३", "3").replace("४", "4").replace("५", "5").replace("६", "6").replace("७", "7").replace("८", "8").replace("९", "9")
    integer_part, _, decimal_part = normalized.partition(".")
    value = int(integer_part) + delta
    return f"{_nepali_digits(value)}.{decimal_part or '००'}"


def _date_variant(date_text: str, generation: int) -> str:
    if not generation:
        return date_text
    year, month, day = date_text.split("-")
    day_value = ((int(day) + generation - 1) % 28) + 1
    return f"{year}-{month}-{day_value:02d}"


def _nepali_date_variant(date_text: str, generation: int) -> str:
    if not generation:
        return date_text
    year, month, day = date_text.split("-")
    day_ascii = int(_ascii_digits(day))
    day_value = ((day_ascii + generation - 1) % 28) + 1
    return f"{year}-{month}-{_nepali_digits_text(f'{day_value:02d}')}"


def _ascii_digits(text: str) -> str:
    return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))


def _nepali_digits(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "०१२३४५६७८९"))


def _nepali_digits_text(value: str) -> str:
    return value.translate(str.maketrans("0123456789", "०१२३४५६७८९"))


def _templates() -> list[PageTemplate]:
    return [
        PageTemplate(
            name="mixed-paragraph",
            pages=[[
                "नेपाल सरकार",
                "{department}",
                "{nepali_line}",
                "{english_line}",
                "रकम रु. {amount} paid on {iso_date}.",
            ]],
            reference_text="नेपाल सरकार\n{department}\n{nepali_line}\n{english_line}\nरकम रु. {amount} paid on {iso_date}.",
            reading_order=["p0-l0", "p0-l1", "p0-l2", "p0-l3", "p0-l4"],
            tables=[],
            figures=[],
            slices=["nepali", "english", "mixed_script", "reading_order"],
            document_type="paragraph",
        ),
        PageTemplate(
            name="table-ledger",
            pages=[[
                "दैनिक हिसाब",
                "नाम        रकम        Remarks",
                "{ledger_name_a}        {ledger_amount_a}       Paid",
                "{ledger_name_b}       {ledger_amount_b}       {pending_word}",
                "Total      {ledger_total}       Verified",
            ]],
            reference_text=(
                "दैनिक हिसाब\n"
                "नाम रकम Remarks\n"
                "{ledger_name_a} {ledger_amount_a} Paid\n"
                "{ledger_name_b} {ledger_amount_b} {pending_word}\n"
                "Total {ledger_total} Verified"
            ),
            reading_order=["p0-l0", "p0-l1", "p0-l2", "p0-l3", "p0-l4"],
            tables=[
                [
                    ["नाम", "रकम", "Remarks"],
                    ["{ledger_name_a}", "{ledger_amount_a}", "Paid"],
                    ["{ledger_name_b}", "{ledger_amount_b}", "{pending_word}"],
                    ["Total", "{ledger_total}", "Verified"],
                ]
            ],
            figures=[],
            slices=["nepali", "english", "mixed_script", "table", "reading_order"],
            document_type="table",
        ),
        PageTemplate(
            name="complex-table-ledger",
            pages=[[
                "समेकित हिसाब",
                "Ledger Summary        Ledger Summary        Status",
                "नाम        रकम        Remarks",
                "{ledger_name_a}        {ledger_amount_a}       Paid",
                "{ledger_name_b}       {ledger_amount_b}       {pending_word}",
                "Total      {ledger_total}       Verified",
            ]],
            reference_text=(
                "समेकित हिसाब\n"
                "Ledger Summary Status\n"
                "नाम रकम Remarks\n"
                "{ledger_name_a} {ledger_amount_a} Paid\n"
                "{ledger_name_b} {ledger_amount_b} {pending_word}\n"
                "Total {ledger_total} Verified"
            ),
            reading_order=["p0-l0", "p0-l1", "p0-l2", "p0-l3", "p0-l4", "p0-l5"],
            tables=[
                [
                    ["Ledger Summary", "Ledger Summary", "Status"],
                    ["नाम", "रकम", "Remarks"],
                    ["{ledger_name_a}", "{ledger_amount_a}", "Paid"],
                    ["{ledger_name_b}", "{ledger_amount_b}", "{pending_word}"],
                    ["Total", "{ledger_total}", "Verified"],
                ]
            ],
            reference_tables=[
                {
                    "cells": [
                        {"row": 0, "col": 0, "text": "Ledger Summary", "rowspan": 1, "colspan": 2},
                        {"row": 0, "col": 2, "text": "Status", "rowspan": 1, "colspan": 1},
                        {"row": 1, "col": 0, "text": "नाम", "rowspan": 1, "colspan": 1},
                        {"row": 1, "col": 1, "text": "रकम", "rowspan": 1, "colspan": 1},
                        {"row": 1, "col": 2, "text": "Remarks", "rowspan": 1, "colspan": 1},
                        {"row": 2, "col": 0, "text": "{ledger_name_a}", "rowspan": 1, "colspan": 1},
                        {"row": 2, "col": 1, "text": "{ledger_amount_a}", "rowspan": 1, "colspan": 1},
                        {"row": 2, "col": 2, "text": "Paid", "rowspan": 1, "colspan": 1},
                        {"row": 3, "col": 0, "text": "{ledger_name_b}", "rowspan": 1, "colspan": 1},
                        {"row": 3, "col": 1, "text": "{ledger_amount_b}", "rowspan": 1, "colspan": 1},
                        {"row": 3, "col": 2, "text": "{pending_word}", "rowspan": 1, "colspan": 1},
                        {"row": 4, "col": 0, "text": "Total", "rowspan": 1, "colspan": 1},
                        {"row": 4, "col": 1, "text": "{ledger_total}", "rowspan": 1, "colspan": 1},
                        {"row": 4, "col": 2, "text": "Verified", "rowspan": 1, "colspan": 1},
                    ]
                }
            ],
            figures=[],
            slices=["nepali", "english", "mixed_script", "table", "complex_table", "reading_order"],
            document_type="table",
        ),
        PageTemplate(
            name="form-notice",
            pages=[[
                "निवेदन फाराम",
                "नाम: {form_name}",
                "Address: {address}",
                "मिति: {nepali_date}",
                "Phone: {phone}",
                "हस्ताक्षर: __________",
            ]],
            reference_text="निवेदन फाराम\nनाम: {form_name}\nAddress: {address}\nमिति: {nepali_date}\nPhone: {phone}\nहस्ताक्षर: __________",
            reading_order=["p0-l0", "p0-l1", "p0-l2", "p0-l3", "p0-l4", "p0-l5"],
            tables=[],
            figures=[],
            slices=["nepali", "english", "mixed_script", "form", "reading_order"],
            document_type="form",
        ),
        PageTemplate(
            name="figure-caption",
            pages=[[
                "वार्षिक प्रतिवेदन",
                "[Figure: revenue chart]",
                "चित्र १: Revenue growth from {figure_year_range}.",
                "निष्कर्ष: {figure_summary}",
            ]],
            reference_text="वार्षिक प्रतिवेदन\nचित्र १: Revenue growth from {figure_year_range}.\nनिष्कर्ष: {figure_summary}",
            reading_order=["p0-l0", "p0-l1", "p0-l2", "p0-l3"],
            tables=[],
            figures=[
                {
                    "figure_id": "fig-revenue-chart",
                    "page_index": 0,
                    "caption": "चित्र १: Revenue growth from {figure_year_range}.",
                    "line_ids": ["p0-l1", "p0-l2"],
                }
            ],
            slices=["nepali", "english", "mixed_script", "figure", "reading_order"],
            document_type="figure",
        ),
        PageTemplate(
            name="multi-page-brief",
            pages=[
                [
                    "Procurement Status Report",
                    "नेपाल सरकार - {ministry}",
                    "{fiscal_year} mixed-script status summary.",
                    "Item        Qty        Vendor",
                    "Masks       {qty_masks}       {vendor_a}",
                    "Gloves      {qty_gloves}       {vendor_b}",
                ],
                [
                    "[Figure: monthly delivery chart]",
                    "चित्र २: Delivery trend by month.",
                    "निष्कर्ष: April पछि आपूर्ति बढेको छ।",
                    "Approved by: {approver}",
                    "मिति: {nepali_date}",
                ],
            ],
            reference_text=(
                "Procurement Status Report\n"
                "नेपाल सरकार - {ministry}\n"
                "{fiscal_year} mixed-script status summary.\n"
                "Item Qty Vendor\n"
                "Masks {qty_masks} {vendor_a}\n"
                "Gloves {qty_gloves} {vendor_b}\n"
                "चित्र २: Delivery trend by month.\n"
                "निष्कर्ष: April पछि आपूर्ति बढेको छ।\n"
                "Approved by: {approver}\n"
                "मिति: {nepali_date}"
            ),
            reading_order=[
                "p0-l0",
                "p0-l1",
                "p0-l2",
                "p0-l3",
                "p0-l4",
                "p0-l5",
                "p1-l0",
                "p1-l1",
                "p1-l2",
                "p1-l3",
                "p1-l4",
            ],
            tables=[
                [
                    ["Item", "Qty", "Vendor"],
                    ["Masks", "{qty_masks}", "{vendor_a}"],
                    ["Gloves", "{qty_gloves}", "{vendor_b}"],
                ]
            ],
            figures=[
                {
                    "figure_id": "fig-delivery-trend",
                    "page_index": 1,
                    "caption": "चित्र २: Delivery trend by month.",
                    "line_ids": ["p1-l0", "p1-l1"],
                }
            ],
            slices=["nepali", "english", "mixed_script", "multi_page", "table", "figure", "form", "reading_order"],
            document_type="multi_page_report",
        ),
    ]


def _select_templates(names: list[str] | None) -> list[PageTemplate]:
    templates = _templates()
    requested = [name for name in (names or []) if name]
    if not requested:
        return templates
    by_name = {template.name: template for template in templates}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise DataValidationError(f"unknown eval-pack templates: {', '.join(missing)}")
    return [by_name[name] for name in requested]
