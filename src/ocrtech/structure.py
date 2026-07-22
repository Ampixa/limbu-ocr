"""Convert OCR lines into blocks, tables, figures, and reading order."""

from __future__ import annotations

import re

from .engines import EngineOutput
from .metrics import bbox_iou
from .normalization import normalize_ocr_text
from .schemas import Block, Document, Figure, Page, Table, TextLine, union_bbox
from .tables import infer_table_from_lines, looks_like_table_line


FIGURE_MARKER_RE = re.compile(r"^\s*(?:\[(?:figure|chart|image|चित्र)[^\]]*\]|(?:figure|fig\.?|chart|चित्र)\s*[:：]?)", re.IGNORECASE)
CAPTION_RE = re.compile(r"^\s*(?:caption|figure|fig\.?|चित्र|तालिका)\s*[\s\d०-९IVXivx.-]*[:：]", re.IGNORECASE)
FRONT_MATTER_STRUCTURAL_ROLES = {"page_marker"}
FRONT_MATTER_DIGIT_PREFIX_RE = re.compile(r"^\s*[०-९]\s+\S")


def sort_lines_reading_order(lines: list[TextLine]) -> list[TextLine]:
    if not lines:
        return []
    heights = [line.bbox.h for line in lines if line.bbox.h > 0]
    line_height = sorted(heights)[len(heights) // 2] if heights else 20
    bucket_size = max(8, line_height * 0.75)
    return sorted(lines, key=lambda line: (round(line.bbox.y / bucket_size), line.bbox.x, line.bbox.y))


def build_document(source_path: str, output: EngineOutput) -> Document:
    pages: list[Page] = []
    tables = list(output.tables)
    figures = list(output.figures)
    preserve_output_line_order = bool(output.metadata.get("preserve_line_order"))
    for raw_page in output.pages:
        preserve_page_line_order = preserve_output_line_order or bool((raw_page.metadata or {}).get("preserve_line_order"))
        if preserve_page_line_order:
            sorted_lines = list(raw_page.text_lines)
        else:
            sorted_lines = sort_lines_reading_order(raw_page.text_lines)
        for index, line in enumerate(sorted_lines):
            if line.line_id is None:
                line.line_id = f"p{raw_page.page_index}-l{index}"
        if preserve_page_line_order:
            pages.append(
                Page(
                    page_index=raw_page.page_index,
                    width=raw_page.width,
                    height=raw_page.height,
                    text_lines=sorted_lines,
                    blocks=_preserved_line_order_blocks(raw_page.page_index, sorted_lines),
                    metadata=raw_page.metadata,
                )
            )
            continue
        blocks: list[Block] = []
        represented_figure_ids: set[str] = set()
        order = 0
        index = 0
        while index < len(sorted_lines):
            line = sorted_lines[index]
            structural_role = _structural_role(line)
            if structural_role is not None:
                text = normalize_ocr_text(line.text).strip()
                if text:
                    blocks.append(
                        Block(
                            block_id=f"p{raw_page.page_index}-b{order}",
                            block_type="text",
                            page_index=raw_page.page_index,
                            bbox=line.bbox,
                            order=order,
                            text=text,
                            confidence=line.confidence,
                            line_ids=[line.line_id or ""],
                            metadata={"structural_role": structural_role},
                        )
                    )
                    order += 1
                index += 1
                continue
            if looks_like_figure_marker(line.text):
                marker_line = line
                caption_line: TextLine | None = None
                index += 1
                if index < len(sorted_lines) and _looks_like_caption_after_marker(marker_line, sorted_lines[index]):
                    caption_line = sorted_lines[index]
                    index += 1
                figure_lines = [marker_line] + ([caption_line] if caption_line is not None else [])
                figure_id = f"p{raw_page.page_index}-f{len(figures)}"
                caption = normalize_ocr_text(caption_line.text).strip() if caption_line is not None else None
                summary = normalize_ocr_text(marker_line.text).strip()
                figure = Figure(
                    figure_id=figure_id,
                    page_index=raw_page.page_index,
                    bbox=union_bbox([item.bbox for item in figure_lines]),
                    caption=caption,
                    summary=summary,
                    confidence=_average_confidence(figure_lines),
                    metadata={"line_ids": [item.line_id or "" for item in figure_lines]},
                )
                figures.append(figure)
                represented_figure_ids.add(figure_id)
                blocks.append(
                    Block(
                        block_id=f"p{raw_page.page_index}-b{order}",
                        block_type="figure",
                        page_index=raw_page.page_index,
                        bbox=figure.bbox,
                        order=order,
                        figure_id=figure_id,
                        confidence=figure.confidence,
                        line_ids=[item.line_id or "" for item in figure_lines],
                        text=caption or summary,
                    )
                )
                order += 1
                continue
            if looks_like_table_line(line.text):
                table_lines = [line]
                index += 1
                while index < len(sorted_lines) and looks_like_table_line(sorted_lines[index].text):
                    table_lines.append(sorted_lines[index])
                    index += 1
                table_id = f"p{raw_page.page_index}-t{len(tables)}"
                table = infer_table_from_lines(table_lines, table_id)
                existing_table = _find_overlapping_table(tables, table)
                represented_table = existing_table or table
                if existing_table is None:
                    tables.append(table)
                blocks.append(
                    Block(
                        block_id=f"p{raw_page.page_index}-b{order}",
                        block_type="table",
                        page_index=raw_page.page_index,
                        bbox=represented_table.bbox,
                        order=order,
                        table_id=represented_table.table_id,
                        confidence=represented_table.confidence,
                        line_ids=[line.line_id or "" for line in table_lines],
                    )
                )
                order += 1
                continue

            text_lines = [line]
            index += 1
            while (
                index < len(sorted_lines)
                and not looks_like_table_line(sorted_lines[index].text)
                and not looks_like_figure_marker(sorted_lines[index].text)
                and _structural_role(sorted_lines[index]) is None
            ):
                previous = text_lines[-1]
                current = sorted_lines[index]
                vertical_gap = current.bbox.y - previous.bbox.bottom
                if vertical_gap > max(previous.bbox.h, 24) * 1.5:
                    break
                text_lines.append(current)
                index += 1
            text = "\n".join(normalize_ocr_text(item.text).strip() for item in text_lines if item.text.strip())
            if not text.strip():
                continue
            confidence = _average_confidence(text_lines)
            block_type = "title" if order == 0 and len(text) <= 120 and len(text_lines) == 1 else "text"
            blocks.append(
                Block(
                    block_id=f"p{raw_page.page_index}-b{order}",
                    block_type=block_type,
                    page_index=raw_page.page_index,
                    bbox=union_bbox([item.bbox for item in text_lines]),
                    order=order,
                    text=text,
                    confidence=confidence,
                    line_ids=[item.line_id or "" for item in text_lines],
                )
            )
            order += 1

        for figure in [item for item in figures if item.page_index == raw_page.page_index and item.figure_id not in represented_figure_ids]:
            blocks.append(
                Block(
                    block_id=f"p{raw_page.page_index}-b{order}",
                    block_type="figure",
                    page_index=raw_page.page_index,
                    bbox=figure.bbox,
                    order=order,
                    figure_id=figure.figure_id,
                    confidence=figure.confidence,
                    text=figure.caption or figure.summary or "",
                )
            )
            order += 1
        blocks = sorted(blocks, key=lambda block: _block_sort_key(raw_page.height, block))
        for order_index, block in enumerate(blocks):
            block.order = order_index
        pages.append(
            Page(
                page_index=raw_page.page_index,
                width=raw_page.width,
                height=raw_page.height,
                text_lines=sorted_lines,
                blocks=blocks,
                metadata=raw_page.metadata,
            )
        )
    return Document(source_path=source_path, pages=pages, tables=tables, figures=figures, metadata=dict(output.metadata))


def _preserved_line_order_blocks(page_index: int, lines: list[TextLine]) -> list[Block]:
    blocks: list[Block] = []
    for order, line in enumerate(lines):
        metadata: dict[str, str] | None = None
        structural_role = _structural_role(line)
        if structural_role is not None:
            metadata = {"structural_role": structural_role}
        text = normalize_ocr_text(line.text)
        blocks.append(
            Block(
                block_id=f"p{page_index}-b{order}",
                block_type="text",
                page_index=page_index,
                bbox=line.bbox,
                order=order,
                text=text,
                confidence=line.confidence,
                line_ids=[line.line_id or ""],
                metadata=metadata,
            )
        )
    return blocks


def _structural_role(line: TextLine) -> str | None:
    role = (line.metadata or {}).get("structural_role")
    if not isinstance(role, str):
        return None
    normalized = role.strip()
    if not normalized or normalized == "text":
        return None
    return normalized


def _block_structural_role(block: Block) -> str | None:
    role = (block.metadata or {}).get("structural_role")
    if not isinstance(role, str):
        return None
    normalized = role.strip()
    return normalized or None


def _is_bottom_digit_prefixed_block(page_height: float | int | None, block: Block) -> bool:
    if page_height is None or page_height <= 0:
        return False
    if block.bbox.y < float(page_height) * 0.80:
        return False
    return bool(FRONT_MATTER_DIGIT_PREFIX_RE.match(normalize_ocr_text(block.text or "")))


def _block_sort_key(page_height: float | int | None, block: Block) -> tuple[int, float, float, int]:
    role = _block_structural_role(block)
    if role in FRONT_MATTER_STRUCTURAL_ROLES:
        front_matter_rank = 0
    elif _is_bottom_digit_prefixed_block(page_height, block):
        front_matter_rank = 1
    else:
        front_matter_rank = 2
    return (front_matter_rank, block.bbox.y, block.bbox.x, block.order)


def looks_like_figure_marker(text: str) -> bool:
    return bool(FIGURE_MARKER_RE.match(text.strip()))


def looks_like_caption(text: str) -> bool:
    return bool(CAPTION_RE.match(text.strip()))


def _looks_like_caption_after_marker(marker_line: TextLine, candidate: TextLine) -> bool:
    if looks_like_caption(candidate.text):
        return True
    if looks_like_figure_marker(candidate.text) or looks_like_table_line(candidate.text):
        return False
    if not candidate.text.strip():
        return False
    vertical_gap = candidate.bbox.y - marker_line.bbox.bottom
    max_gap = max(marker_line.bbox.h, candidate.bbox.h, 24) * 1.5
    return 0 <= vertical_gap <= max_gap


def _find_overlapping_table(tables: list[Table], candidate: Table, *, iou_threshold: float = 0.9) -> Table | None:
    for table in tables:
        if table.page_index != candidate.page_index:
            continue
        if bbox_iou(table.bbox, candidate.bbox) >= iou_threshold:
            return table
    return None


def _average_confidence(lines: list[TextLine]) -> float | None:
    confidences = [item.confidence for item in lines if item.confidence is not None]
    return sum(confidences) / len(confidences) if confidences else None
