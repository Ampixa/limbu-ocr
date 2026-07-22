#!/usr/bin/env python3
"""Apply simple geometry reading-order strategies to existing OCR documents."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from ocrtech.linearization import column_major_indices
from ocrtech.markdown import render_document_markdown
from ocrtech.manifest import sha256_file
from ocrtech.schemas import Block, Document, TextLine, union_bbox


REPORT_JSON = "reading-order-strategy-application.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _document_paths(documents_dir: Path) -> dict[str, Path]:
    if not documents_dir.is_dir():
        raise ValueError(f"documents directory does not exist: {documents_dir}")
    paths: dict[str, Path] = {}
    for child in sorted(documents_dir.iterdir()):
        if not child.is_dir():
            continue
        document_path = child / "document.json"
        if document_path.is_file():
            paths[child.name] = document_path
    if not paths:
        raise ValueError(f"documents directory contains no sample document.json files: {documents_dir}")
    return paths


def _has_two_column_shape(lines: list[TextLine], image_width: float | None) -> bool:
    if len(lines) < 8 or not image_width or image_width <= 0:
        return False
    midpoint = image_width / 2.0
    left = [line for line in lines if line.bbox.x + line.bbox.w / 2.0 < midpoint]
    right = [line for line in lines if line.bbox.x + line.bbox.w / 2.0 >= midpoint]
    if len(left) < 3 or len(right) < 3:
        return False
    left_median = median(line.bbox.x + line.bbox.w / 2.0 for line in left)
    right_median = median(line.bbox.x + line.bbox.w / 2.0 for line in right)
    return (right_median - left_median) >= image_width * 0.22


def _line_key(line: TextLine, *, mode: str, image_width: float | None, row_height: float) -> tuple[float, float, float, str]:
    x_center = line.bbox.x + line.bbox.w / 2.0
    y_center = line.bbox.y + line.bbox.h / 2.0
    width = image_width or 0.0
    if mode == "Columns LTR":
        return (0 if x_center < width / 2.0 else 1, y_center, x_center, line.line_id or "")
    if mode == "Columns RTL":
        return (0 if x_center >= width / 2.0 else 1, y_center, x_center, line.line_id or "")
    if mode == "Rows RTL":
        return (math.floor(y_center / row_height), -x_center, y_center, line.line_id or "")
    return (math.floor(y_center / row_height), x_center, y_center, line.line_id or "")


def _sort_top_anchored_rows(lines: list[TextLine], *, mode: str, row_height: float) -> list[TextLine]:
    if not lines:
        return []
    row_tolerance = max(12.0, row_height * 0.75)
    rows: list[list[TextLine]] = []
    row_anchors: list[float] = []
    for line in sorted(lines, key=lambda item: (item.bbox.y, item.bbox.x, item.line_id or "")):
        anchor = line.bbox.y
        if not rows or abs(anchor - row_anchors[-1]) > row_tolerance:
            rows.append([line])
            row_anchors.append(anchor)
            continue
        rows[-1].append(line)
        row_anchors[-1] = sum(item.bbox.y for item in rows[-1]) / len(rows[-1])

    rows.sort(key=lambda row: min(line.bbox.y for line in row))
    sorted_lines: list[TextLine] = []
    reverse_x = mode == "Rows RTL Top"
    for row in rows:
        sorted_lines.extend(
            sorted(
                row,
                key=lambda line: (
                    -(line.bbox.x + line.bbox.w / 2.0) if reverse_x else (line.bbox.x + line.bbox.w / 2.0),
                    line.bbox.y,
                    line.line_id or "",
                ),
            )
        )
    return sorted_lines


def _resolved_mode(mode: str, lines: list[TextLine], image_width: float | None) -> str:
    if mode != "Auto":
        return mode
    return "Columns LTR" if _has_two_column_shape(lines, image_width) else "Rows LTR"


def _block_key(block: Block, line_positions: dict[str, int]) -> tuple[int, int, str]:
    if block.line_ids:
        positions = [line_positions[line_id] for line_id in block.line_ids if line_id in line_positions]
        if positions:
            return (min(positions), block.order, block.block_id)
    return (10**9, block.order, block.block_id)


def _apply_to_document(document: Document, mode: str) -> tuple[Document, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    for page in document.pages:
        lines = list(page.text_lines)
        heights = [line.bbox.h for line in lines if line.bbox.h > 0]
        row_height = max(12.0, float(median(heights)) if heights else 24.0)
        actual_mode = _resolved_mode(mode, lines, page.width)
        if actual_mode == "Column Major":
            order = column_major_indices(
                [(line.bbox.x, line.bbox.y, line.bbox.w, line.bbox.h) for line in lines]
            )
            sorted_lines = [lines[index] for index in order]
        elif actual_mode in {"Rows LTR Top", "Rows RTL Top"}:
            sorted_lines = _sort_top_anchored_rows(lines, mode=actual_mode, row_height=row_height)
        else:
            sorted_lines = sorted(
                lines,
                key=lambda line: _line_key(line, mode=actual_mode, image_width=page.width, row_height=row_height),
            )
        page.text_lines = sorted_lines
        lines_by_id = {line.line_id or "": line for line in sorted_lines}
        line_positions = {line.line_id or "": index for index, line in enumerate(sorted_lines)}
        for block in page.blocks:
            if not block.line_ids:
                continue
            block.line_ids = sorted(
                block.line_ids,
                key=lambda line_id: line_positions.get(line_id, 10**9),
            )
            block_lines = [lines_by_id[line_id] for line_id in block.line_ids if line_id in lines_by_id]
            if block_lines:
                block.text = "\n".join(line.text for line in block_lines)
                block.bbox = union_bbox([line.bbox for line in block_lines])
        page.blocks = sorted(page.blocks, key=lambda block: _block_key(block, line_positions))
        for order, block in enumerate(page.blocks, start=1):
            block.order = order
        page.metadata = dict(page.metadata or {})
        page.metadata["reading_order_strategy"] = {
            "requested_mode": mode,
            "resolved_mode": actual_mode,
            "row_height": row_height,
            "line_count": len(sorted_lines),
        }
        reports.append(
            {
                "page_index": page.page_index,
                "requested_mode": mode,
                "resolved_mode": actual_mode,
                "row_height": row_height,
                "line_count": len(sorted_lines),
            }
        )
    document.metadata = dict(document.metadata or {})
    document.metadata["reading_order_strategy"] = {"requested_mode": mode}
    return document, reports


def _write_document_artifacts(document_dir: Path, document: Document) -> None:
    document_dir.mkdir(parents=True, exist_ok=True)
    (document_dir / "document.json").write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (document_dir / "document.md").write_text(render_document_markdown(document), encoding="utf-8")
    (document_dir / "document.body.md").write_text(render_document_markdown(document, include_structural_roles=False), encoding="utf-8")


def apply_strategy(args: argparse.Namespace) -> dict[str, Any]:
    source_documents_dir = args.source_documents_dir
    out = args.out
    if source_documents_dir.resolve() == out.resolve():
        raise ValueError("--out must not be the same directory as --source-documents-dir")
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise FileExistsError(f"output directory exists and is not empty: {out}")
        if out.resolve() in {Path.cwd().resolve(), Path("/")}:
            raise ValueError(f"refusing to remove unsafe output directory: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = _document_paths(source_documents_dir)
    pages: list[dict[str, Any]] = []
    for sample_id, document_path in sorted(source_paths.items()):
        document = Document.from_dict(_read_json_object(document_path))
        document, page_reports = _apply_to_document(document, args.mode)
        target_dir = out / sample_id
        _write_document_artifacts(target_dir, document)
        pages.append(
            {
                "sample_id": sample_id,
                "source_document": str(document_path),
                "source_document_sha256": sha256_file(document_path),
                "document_json": str(target_dir / "document.json"),
                "document_json_sha256": sha256_file(target_dir / "document.json"),
                "page_reports": page_reports,
            }
        )
    report = {
        "tool": "tools/apply_page_reading_order_strategy.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_documents_dir": str(source_documents_dir),
        "out": str(out),
        "mode": args.mode,
        "document_count": len(source_paths),
        "pages": pages,
    }
    report_path = out / REPORT_JSON
    report["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-documents-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["Auto", "Rows LTR", "Rows RTL", "Rows LTR Top", "Rows RTL Top", "Columns LTR", "Columns RTL", "Column Major"],
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = apply_strategy(args)
    print(report["report_json"])
    print(f"documents={report['document_count']} mode={report['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
