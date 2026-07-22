"""Reference document loading and scoring helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, write_manifest
from .markdown import render_document_markdown
from .metrics import bbox_iou, cer, detection_score, reading_order_pair_accuracy, table_cell_score, table_structure_similarity, table_teds_structure, table_teds_text, wer
from .normalization import normalize_ocr_text
from .schemas import BBox, Document, Figure, TableCell
from .tables import cells_to_grid, table_to_grid


@dataclass(slots=True)
class ReferenceTable:
    cells: list[TableCell] = field(default_factory=list)

    @property
    def grid(self) -> list[list[str]]:
        return cells_to_grid(self.cells)


@dataclass(slots=True)
class ReferenceFigure:
    caption: str | None = None
    bbox: BBox | None = None
    page_index: int = 0
    figure_id: str | None = None
    line_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceDocument:
    text: str | None = None
    markdown: str | None = None
    reading_order: list[str] | None = None
    tables: list[ReferenceTable] = field(default_factory=list)
    figures: list[ReferenceFigure] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferencePackReport:
    references_dir: str
    source_manifest: str
    rewritten_manifest: str | None
    sample_count: int
    summary_json_path: str
    summary_md_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "references_dir": self.references_dir,
            "source_manifest": self.source_manifest,
            "rewritten_manifest": self.rewritten_manifest,
            "sample_count": self.sample_count,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class ReferenceAuditReport:
    manifest_path: str
    output_dir: str
    sample_count: int
    parsed_count: int
    text_reference_count: int
    markdown_reference_count: int
    reading_order_reference_count: int
    table_reference_count: int
    figure_reference_count: int
    claim_ready_reference_count: int
    require_claim_ready: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest_path": self.manifest_path,
            "output_dir": self.output_dir,
            "sample_count": self.sample_count,
            "parsed_count": self.parsed_count,
            "text_reference_count": self.text_reference_count,
            "markdown_reference_count": self.markdown_reference_count,
            "reading_order_reference_count": self.reading_order_reference_count,
            "table_reference_count": self.table_reference_count,
            "figure_reference_count": self.figure_reference_count,
            "claim_ready_reference_count": self.claim_ready_reference_count,
            "require_claim_ready": self.require_claim_ready,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


def load_reference(input_path: Path, reference_root: Path | None = None, *, explicit_path: str | Path | None = None) -> ReferenceDocument | None:
    if explicit_path is not None:
        explicit_candidate = Path(explicit_path)
        if _is_json_reference_path(explicit_candidate):
            payload = _load_reference_json_candidates([explicit_candidate])
            if payload is not None:
                return _reference_from_payload(payload)
        else:
            text_reference = _load_reference_text_candidates([explicit_candidate])
            if text_reference is not None:
                return _reference_from_text_path(text_reference[0], text_reference[1])
    payload = _load_reference_json(input_path, reference_root)
    if payload is not None:
        return _reference_from_payload(payload)
    text_reference = _load_reference_text(input_path, reference_root)
    if text_reference is not None:
        return _reference_from_text_path(text_reference[0], text_reference[1])
    return None


def audit_references(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    require_claim_ready: bool = False,
) -> ReferenceAuditReport:
    manifest = Path(manifest_path)
    entries = load_manifest(manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    warnings: list[str] = []
    parsed_count = 0
    text_count = 0
    markdown_count = 0
    reading_order_count = 0
    table_count = 0
    figure_count = 0
    claim_ready_count = 0
    for entry in entries:
        input_path = _resolve_manifest_path(entry.image_path, manifest.parent)
        explicit = entry.metadata.get("reference_path") if entry.metadata else None
        explicit_path = _resolve_manifest_path(explicit, manifest.parent) if isinstance(explicit, str) and explicit else None
        if isinstance(explicit, str) and explicit and not explicit_path.exists():
            issues.append(f"{entry.sample_id}: reference_path does not exist: {explicit}")
            continue
        try:
            reference = load_reference(input_path, explicit_path=explicit_path)
        except Exception as exc:
            issues.append(f"{entry.sample_id}: reference failed to parse: {exc}")
            continue
        if reference is None:
            issues.append(f"{entry.sample_id}: no reference found")
            continue
        parsed_count += 1
        if reference.text is not None:
            text_count += 1
        if reference.markdown is not None:
            markdown_count += 1
        if reference.reading_order:
            reading_order_count += 1
        if reference.tables:
            table_count += 1
        if reference.figures:
            figure_count += 1
        metadata = reference.metadata or {}
        claim_ready = metadata.get("reference_status") == "verified" and metadata.get("claim_evidence_eligible") is True
        if claim_ready:
            claim_ready_count += 1
        elif require_claim_ready:
            issues.append(f"{entry.sample_id}: reference is not claim-ready")
        if reference.text is None and reference.markdown is None and not reference.reading_order and not reference.tables and not reference.figures:
            warnings.append(f"{entry.sample_id}: reference parsed but has no scoreable fields")
    report = ReferenceAuditReport(
        manifest_path=str(manifest),
        output_dir=str(out),
        sample_count=len(entries),
        parsed_count=parsed_count,
        text_reference_count=text_count,
        markdown_reference_count=markdown_count,
        reading_order_reference_count=reading_order_count,
        table_reference_count=table_count,
        figure_reference_count=figure_count,
        claim_ready_reference_count=claim_ready_count,
        require_claim_ready=require_claim_ready,
        issues=issues,
        warnings=warnings,
        summary_json_path=str(out / "reference-audit.json"),
        summary_md_path=str(out / "reference-audit.md"),
    )
    _write_reference_audit(report)
    return report


def materialize_references(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rewritten_manifest_path: str | Path | None = None,
) -> ReferencePackReport:
    entries = load_manifest(manifest_path)
    if not entries:
        raise DataValidationError("reference materialization requires a non-empty manifest")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rewritten_entries: list[ManifestEntry] = []
    for entry in entries:
        sample_dir = out / entry.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        reference_path = sample_dir / "document.md"
        reference_path.write_text(_reference_text_from_manifest_entry(entry), encoding="utf-8")
        metadata = dict(entry.metadata or {})
        metadata["reference_path"] = str(reference_path)
        rewritten_entries.append(
            ManifestEntry(
                sample_id=entry.sample_id,
                dataset=entry.dataset,
                split=entry.split,
                image_path=entry.image_path,
                text=entry.text,
                sha256=entry.sha256,
                metadata=metadata,
            )
        )
    rewritten_manifest = Path(rewritten_manifest_path) if rewritten_manifest_path is not None else None
    if rewritten_manifest is not None:
        rewritten_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(rewritten_entries, rewritten_manifest)
    summary_json = out / "reference-pack.json"
    summary_md = out / "reference-pack.md"
    report = ReferencePackReport(
        references_dir=str(out),
        source_manifest=str(Path(manifest_path)),
        rewritten_manifest=str(rewritten_manifest) if rewritten_manifest is not None else None,
        sample_count=len(entries),
        summary_json_path=str(summary_json),
        summary_md_path=str(summary_md),
    )
    summary_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                "# Reference Pack",
                "",
                f"Source manifest: `{manifest_path}`",
                f"References dir: `{out}`",
                f"Samples: `{len(entries)}`",
                f"Rewritten manifest: `{rewritten_manifest or ''}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _write_reference_audit(report: ReferenceAuditReport) -> None:
    json_path = Path(report.summary_json_path or Path(report.output_dir) / "reference-audit.json")
    md_path = Path(report.summary_md_path or Path(report.output_dir) / "reference-audit.md")
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Reference Audit",
        "",
        f"- passed: {report.passed}",
        f"- manifest: `{report.manifest_path}`",
        f"- samples: {report.sample_count}",
        f"- parsed references: {report.parsed_count}",
        f"- text references: {report.text_reference_count}",
        f"- markdown references: {report.markdown_reference_count}",
        f"- reading-order references: {report.reading_order_reference_count}",
        f"- table references: {report.table_reference_count}",
        f"- figure references: {report.figure_reference_count}",
        f"- claim-ready references: {report.claim_ready_reference_count}",
        f"- require claim-ready: {report.require_claim_ready}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in report.issues) if report.issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_document(document: Document | None, text: str, reference: ReferenceDocument | None) -> dict[str, float]:
    if reference is None:
        return {}
    metrics: dict[str, float] = {}
    reconstruction_text_similarity: float | None = None
    reconstruction_components: list[float] = []
    if reference.text is not None:
        pred = text.strip()
        ref = reference.text.strip()
        metrics["cer"] = cer(pred, ref)
        metrics["wer"] = wer(pred, ref)
        metrics["text_cer"] = metrics["cer"]
        metrics["text_wer"] = metrics["wer"]
    if document is not None and reference.markdown is not None:
        predicted_markdown = render_document_markdown(document).strip()
        reference_markdown = reference.markdown.strip()
        reconstruction_cer = cer(predicted_markdown, reference_markdown)
        metrics["document_reconstruction_cer"] = reconstruction_cer
        metrics["document_reconstruction_wer"] = wer(predicted_markdown, reference_markdown)
        reconstruction_text_similarity = max(0.0, 1.0 - reconstruction_cer)
        metrics["document_reconstruction_text_similarity"] = reconstruction_text_similarity
        reconstruction_components.append(reconstruction_text_similarity)
    if document is not None and reference.reading_order:
        predicted_order = _document_reading_order(document)
        reading_order_score = reading_order_pair_accuracy(predicted_order, reference.reading_order)
        metrics["reading_order_pair_accuracy"] = reading_order_score
        if reconstruction_text_similarity is not None:
            reconstruction_components.append(reading_order_score)
    if document is not None and reference.tables:
        table_scores = []
        structure_scores: list[float] = []
        teds_structure_scores: list[float] = []
        teds_text_scores: list[float] = []
        exact_matches: list[float] = []
        predicted_tables = sorted(document.tables, key=lambda item: (item.page_index, item.bbox.y, item.bbox.x))
        table_matches = _match_tables(predicted_tables, reference.tables)
        for reference_table, predicted_table in table_matches:
            predicted_grid = table_to_grid(predicted_table) if predicted_table is not None else []
            predicted_cells = predicted_table.cells if predicted_table is not None else []
            table_scores.append(table_cell_score(predicted_grid, reference_table.grid))
            structure_scores.append(table_structure_similarity(predicted_cells, reference_table.cells))
            teds_structure_scores.append(table_teds_structure(predicted_cells, reference_table.cells))
            teds_text_scores.append(table_teds_text(predicted_cells, reference_table.cells))
            exact_matches.append(1.0 if _table_cells_exact(predicted_cells, reference_table.cells) else 0.0)
        if table_scores:
            metrics["table_cell_precision"] = sum(item.precision for item in table_scores) / len(table_scores)
            metrics["table_cell_recall"] = sum(item.recall for item in table_scores) / len(table_scores)
            metrics["table_cell_f1"] = sum(item.f1 for item in table_scores) / len(table_scores)
            metrics["table_exact_match"] = sum(exact_matches) / len(exact_matches)
            metrics["table_structure_similarity"] = sum(structure_scores) / len(structure_scores)
            metrics["table_teds_structure"] = sum(teds_structure_scores) / len(teds_structure_scores)
            metrics["table_teds_text"] = sum(teds_text_scores) / len(teds_text_scores)
            if reconstruction_text_similarity is not None:
                reconstruction_components.append(min(metrics["table_cell_f1"], metrics["table_teds_text"]))
    if document is not None and reference.figures:
        figure_metrics = _score_figures(document.figures, reference.figures)
        metrics.update(figure_metrics)
        if reconstruction_text_similarity is not None:
            if "figure_detection_f1" in figure_metrics:
                reconstruction_components.append(figure_metrics["figure_detection_f1"])
            if "figure_caption_cer" in figure_metrics:
                reconstruction_components.append(max(0.0, 1.0 - figure_metrics["figure_caption_cer"]))
    if reconstruction_text_similarity is not None:
        metrics["document_reconstruction_similarity"] = min(reconstruction_components) if reconstruction_components else reconstruction_text_similarity
    return metrics


def _match_tables(predicted_tables: list[Any], reference_tables: list[ReferenceTable]) -> list[tuple[ReferenceTable, Any | None]]:
    matches: list[tuple[ReferenceTable, Any | None]] = []
    unused = set(range(len(predicted_tables)))
    for reference_table in reference_tables:
        best_index: int | None = None
        best_score: tuple[float, float, float] | None = None
        for index in sorted(unused):
            predicted_table = predicted_tables[index]
            score = _table_match_score(predicted_table, reference_table)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            matches.append((reference_table, None))
            continue
        unused.remove(best_index)
        matches.append((reference_table, predicted_tables[best_index]))
    return matches


def _table_match_score(predicted_table: Any, reference_table: ReferenceTable) -> tuple[float, float, float]:
    predicted_cells = predicted_table.cells if predicted_table is not None else []
    predicted_grid = table_to_grid(predicted_table) if predicted_table is not None else []
    cell_score = table_cell_score(predicted_grid, reference_table.grid)
    return (
        table_teds_text(predicted_cells, reference_table.cells),
        table_teds_structure(predicted_cells, reference_table.cells),
        cell_score.f1,
    )


def _document_reading_order(document: Document) -> list[str]:
    ordered: list[str] = []
    for page in sorted(document.pages, key=lambda item: item.page_index):
        for block in sorted(page.blocks, key=lambda item: item.order):
            if block.line_ids:
                ordered.extend(line_id for line_id in block.line_ids if line_id)
            else:
                ordered.append(block.block_id)
    return ordered


def _load_reference_json(input_path: Path, reference_root: Path | None) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if reference_root:
        candidates.extend(
            [
                reference_root / f"{input_path.stem}.json",
                reference_root / input_path.name / "reference.json",
                reference_root / input_path.stem / "reference.json",
            ]
        )
    candidates.extend([input_path.with_suffix(input_path.suffix + ".ref.json"), input_path.with_suffix(".ref.json")])
    return _load_reference_json_candidates(candidates)


def _resolve_manifest_path(path_value: str | None, manifest_dir: Path) -> Path:
    if not path_value:
        return Path("")
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def _load_reference_json_candidates(candidates: list[Path]) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"Invalid reference JSON {candidate}: {exc}") from exc
            if not isinstance(payload, dict):
                raise DataValidationError(f"Reference JSON must be an object: {candidate}")
            return payload
    return None


def _load_reference_text(input_path: Path, reference_root: Path | None) -> tuple[str, Path] | None:
    candidates: list[Path] = []
    if reference_root:
        candidates.extend(
            [
                reference_root / f"{input_path.stem}.txt",
                reference_root / input_path.name / "document.md",
                reference_root / input_path.stem / "document.md",
            ]
        )
    candidates.extend([input_path.with_suffix(input_path.suffix + ".ref.txt"), input_path.with_suffix(".ref.txt")])
    return _load_reference_text_candidates(candidates)


def _load_reference_text_candidates(candidates: list[Path]) -> tuple[str, Path] | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), candidate
    return None


def _reference_from_text_path(text: str, path: Path) -> ReferenceDocument:
    markdown = text if _is_markdown_reference_path(path) else None
    return ReferenceDocument(text=text, markdown=markdown)


def _is_markdown_reference_path(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"} or path.name == "document.md"


def _reference_text_from_manifest_entry(entry: ManifestEntry) -> str:
    text = entry.text.rstrip("\n")
    return text + "\n"


def _is_json_reference_path(path: Path) -> bool:
    return path.suffix.lower() == ".json" or path.name.endswith(".ref.json") or path.name == "reference.json"


def _reference_from_payload(payload: dict[str, Any]) -> ReferenceDocument:
    text = payload.get("text")
    if text is not None and not isinstance(text, str):
        raise DataValidationError("reference text must be a string")
    markdown = _reference_markdown_from_payload(payload)
    reading_order = _coerce_reference_reading_order(payload.get("reading_order"))
    tables_payload = payload.get("tables", [])
    if not isinstance(tables_payload, list):
        raise DataValidationError("reference tables must be a list")
    tables = [_coerce_reference_table(item) for item in tables_payload]
    figures_payload = payload.get("figures", [])
    if not isinstance(figures_payload, list):
        raise DataValidationError("reference figures must be a list")
    figures = [_coerce_reference_figure(item) for item in figures_payload]
    return ReferenceDocument(
        text=normalize_ocr_text(text) if text is not None else None,
        markdown=normalize_ocr_text(markdown) if markdown is not None else None,
        reading_order=reading_order,
        tables=tables,
        figures=figures,
        metadata=dict(payload.get("metadata") or {}),
    )


def _coerce_reference_reading_order(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DataValidationError("reference reading_order must be a list")
    ordered: list[str] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            item_id = item.strip()
        elif isinstance(item, dict):
            raw_id = item.get("id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise DataValidationError(f"reference reading_order item {index} missing id")
            raw_text = item.get("text")
            if raw_text is not None and not isinstance(raw_text, str):
                raise DataValidationError(f"reference reading_order item {index} text must be a string")
            item_id = raw_id.strip()
        else:
            raise DataValidationError("reference reading_order must contain strings or objects with id")
        if not item_id:
            raise DataValidationError(f"reference reading_order item {index} is empty")
        ordered.append(item_id)
    return ordered


def _reference_markdown_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("markdown", "document_markdown", "reconstruction_markdown"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise DataValidationError(f"reference {key} must be a string")
        return value
    return None


def _coerce_reference_table(value: Any) -> ReferenceTable:
    if isinstance(value, dict):
        if "cells" in value and _looks_like_cell_objects(value["cells"]):
            return ReferenceTable(cells=[_coerce_reference_table_cell(item) for item in value["cells"]])
        value = value.get("grid") or value.get("cells")
    if not isinstance(value, list):
        raise DataValidationError("reference table must be a grid or object with grid/cells")
    cells: list[TableCell] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise DataValidationError("reference table row must be a list")
        for col_index, cell in enumerate(row):
            cells.append(TableCell(row=row_index, col=col_index, text=normalize_ocr_text(str(cell))))
    return ReferenceTable(cells=cells)


def _looks_like_cell_objects(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) and {"row", "col", "text"} <= set(item) for item in value)


def _coerce_reference_table_cell(value: Any) -> TableCell:
    if not isinstance(value, dict):
        raise DataValidationError("reference table cell must be an object")
    return TableCell(
        row=int(value.get("row", 0)),
        col=int(value.get("col", 0)),
        text=normalize_ocr_text(str(value.get("text", ""))),
        rowspan=int(value.get("rowspan", 1)),
        colspan=int(value.get("colspan", 1)),
    )


def _table_cells_exact(predicted: list[TableCell], reference: list[TableCell]) -> bool:
    return _cell_signature(predicted) == _cell_signature(reference)


def _cell_signature(cells: list[TableCell]) -> list[tuple[int, int, int, int, str]]:
    return sorted(
        (
            cell.row,
            cell.col,
            cell.rowspan,
            cell.colspan,
            normalize_ocr_text(cell.text, collapse_spaces=True),
        )
        for cell in cells
    )


def _coerce_reference_figure(value: Any) -> ReferenceFigure:
    if not isinstance(value, dict):
        raise DataValidationError("reference figure must be an object")
    bbox_payload = value.get("bbox")
    line_ids = value.get("line_ids", [])
    if not isinstance(line_ids, list) or not all(isinstance(item, str) for item in line_ids):
        raise DataValidationError("reference figure line_ids must be a list of strings")
    caption = value.get("caption")
    if caption is not None and not isinstance(caption, str):
        raise DataValidationError("reference figure caption must be a string")
    return ReferenceFigure(
        caption=normalize_ocr_text(caption) if caption is not None else None,
        bbox=BBox.from_any(bbox_payload) if bbox_payload is not None else None,
        page_index=int(value.get("page_index", 0)),
        figure_id=value.get("figure_id"),
        line_ids=line_ids,
    )


def _score_figures(predicted: list[Figure], reference: list[ReferenceFigure]) -> dict[str, float]:
    matches = _match_figures(predicted, reference)
    score = detection_score(len(predicted), len(reference), len(matches))
    metrics: dict[str, float] = {
        "figure_detection_precision": score.precision,
        "figure_detection_recall": score.recall,
        "figure_detection_f1": score.f1,
    }
    caption_errors: list[float] = []
    for predicted_index, reference_index in matches:
        ref_caption = reference[reference_index].caption
        pred_caption = predicted[predicted_index].caption or predicted[predicted_index].summary
        if ref_caption is not None:
            caption_errors.append(cer(pred_caption or "", ref_caption))
    if caption_errors:
        metrics["figure_caption_cer"] = sum(caption_errors) / len(caption_errors)
    return metrics


def _match_figures(predicted: list[Figure], reference: list[ReferenceFigure]) -> list[tuple[int, int]]:
    used_predicted: set[int] = set()
    matches: list[tuple[int, int]] = []
    for ref_index, ref in enumerate(reference):
        best: tuple[float, int] | None = None
        for pred_index, pred in enumerate(predicted):
            if pred_index in used_predicted or pred.page_index != ref.page_index:
                continue
            score = _figure_match_score(pred, ref)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, pred_index)
        if best is not None:
            used_predicted.add(best[1])
            matches.append((best[1], ref_index))
    return matches


def _figure_match_score(predicted: Figure, reference: ReferenceFigure) -> float:
    if reference.bbox is not None:
        iou = bbox_iou(predicted.bbox, reference.bbox)
        if iou >= 0.5:
            return iou
    if reference.caption and predicted.caption:
        caption_error = cer(predicted.caption, reference.caption)
        if caption_error <= 0.25:
            return 1.0 - caption_error
    if reference.line_ids:
        source_ids = set(predicted.metadata.get("line_ids", [])) if isinstance(predicted.metadata, dict) else set()
        if source_ids & set(reference.line_ids):
            return 0.75
    return 0.0
