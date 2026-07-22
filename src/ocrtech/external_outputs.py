"""Helpers for normalizing external OCR baseline outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import BBox, Block, Document, Page, Table, TableCell, TextLine


_PAGE_BREAK_PATTERN = re.compile(r"(?:\n\s*<!--\s*pagebreak\s*-->\s*\n)|\f", re.IGNORECASE)
_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def bundle_page_paths(input_path: Path) -> list[Path]:
    """Return page image paths for a directory bundle or the single input path."""
    if not input_path.is_dir():
        return [input_path]
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    page_paths = [path for path in sorted(input_path.iterdir()) if path.is_file() and path.suffix.lower() in allowed_suffixes]
    if not page_paths:
        raise ValueError(f"page bundle directory has no supported image files: {input_path}")
    return page_paths


def materialize_external_markdown(
    source_path: Path,
    output_dir: Path,
    markdown_text: str,
    *,
    metadata: dict[str, Any] | None = None,
    raw_payload: Any | None = None,
) -> Document:
    """Write normalized markdown and document JSON for an external OCR result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _clean_markdown(markdown_text)
    document = document_from_markdown(source_path, cleaned, metadata=metadata)
    (output_dir / "document.md").write_text(cleaned + ("\n" if cleaned and not cleaned.endswith("\n") else ""), encoding="utf-8")
    (output_dir / "document.json").write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if raw_payload is not None:
        (output_dir / "raw-output.json").write_text(
            json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return document


def document_from_markdown(source_path: Path, markdown_text: str, *, metadata: dict[str, Any] | None = None) -> Document:
    """Build a minimal structured document from markdown output."""
    pages: list[Page] = []
    tables: list[Table] = []
    markdown_pages = [part.strip("\n") for part in _PAGE_BREAK_PATTERN.split(markdown_text)] or [markdown_text]
    for page_index, page_markdown in enumerate(markdown_pages):
        page = _page_from_markdown(page_markdown, page_index=page_index, global_tables=tables)
        pages.append(page)
    return Document(source_path=str(source_path), pages=pages, tables=tables, figures=[], metadata=dict(metadata or {}))


def _clean_markdown(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _page_from_markdown(markdown_text: str, *, page_index: int, global_tables: list[Table]) -> Page:
    text_lines: list[TextLine] = []
    blocks: list[Block] = []
    lines = markdown_text.splitlines()
    order = 0
    y = 0.0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            y += 20
            continue
        table_result = _parse_markdown_table(lines, start=index)
        if table_result is not None:
            table, consumed, table_text = table_result
            table.page_index = page_index
            global_tables.append(table)
            blocks.append(
                Block(
                    block_id=f"p{page_index}-b{order}",
                    block_type="table",
                    page_index=page_index,
                    bbox=table.bbox,
                    order=order,
                    text=table_text,
                    table_id=table.table_id,
                )
            )
            order += 1
            y = table.bbox.bottom + 24
            index += consumed
            continue
        text = _strip_markdown_prefix(stripped)
        bbox = BBox(0, y, max(50, len(text) * 9), 22)
        line_id = f"p{page_index}-l{len(text_lines)}"
        text_lines.append(TextLine(text=text, bbox=bbox, confidence=None, page_index=page_index, line_id=line_id))
        block_type = "title" if stripped.startswith("#") else "caption" if _looks_like_caption(text) else "text"
        blocks.append(
            Block(
                block_id=f"p{page_index}-b{order}",
                block_type=block_type,
                page_index=page_index,
                bbox=bbox,
                order=order,
                text=text,
                line_ids=[line_id],
            )
        )
        order += 1
        y += 28
        index += 1
    return Page(page_index=page_index, text_lines=text_lines, blocks=blocks, metadata={"source": "external-markdown"})


def _parse_markdown_table(lines: list[str], *, start: int) -> tuple[Table, int, str] | None:
    if start + 1 >= len(lines):
        return None
    header_line = lines[start].strip()
    separator_line = lines[start + 1].strip()
    if "|" not in header_line or not _TABLE_SEPARATOR_PATTERN.match(separator_line):
        return None
    data_lines: list[str] = [lines[start], lines[start + 1]]
    index = start + 2
    while index < len(lines):
        candidate = lines[index].strip()
        if not candidate or "|" not in candidate:
            break
        data_lines.append(lines[index])
        index += 1
    rows = [_split_markdown_row(item) for item in data_lines if "|" in item]
    if len(rows) < 2:
        return None
    header = rows[0]
    body = rows[2:] if len(rows) > 2 else []
    cells: list[TableCell] = []
    row_count = 0
    if header:
        for col_index, value in enumerate(header):
            cells.append(TableCell(row=0, col=col_index, text=value))
        row_count = 1
    for body_row in body:
        for col_index, value in enumerate(body_row):
            cells.append(TableCell(row=row_count, col=col_index, text=value))
        row_count += 1
    col_count = max((len(row) for row in [header, *body]), default=1)
    table_height = max(32.0, 28.0 * max(1, row_count))
    table_width = max(120.0, 120.0 * col_count)
    table_text = "\n".join(line.rstrip() for line in data_lines)
    table = Table(
        table_id=f"table-{start}",
        page_index=0,
        bbox=BBox(0, start * 28.0, table_width, table_height),
        cells=cells,
        html=None,
        caption=None,
        metadata={"source": "external-markdown"},
    )
    return table, len(data_lines), table_text


def _split_markdown_row(line: str) -> list[str]:
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [part.strip() for part in trimmed.split("|")]


def _strip_markdown_prefix(text: str) -> str:
    value = text.lstrip("#").strip() if text.startswith("#") else text
    if value.startswith("- "):
        return value[2:].strip()
    if value.startswith("* "):
        return value[2:].strip()
    return value


def _looks_like_caption(text: str) -> bool:
    lowered = text.casefold()
    return lowered.startswith("figure ") or lowered.startswith("fig. ") or lowered.startswith("चित्र ")
