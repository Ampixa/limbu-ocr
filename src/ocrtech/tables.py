"""Table reconstruction and export helpers."""

from __future__ import annotations

import csv
import html
import io
import re
from pathlib import Path

from dataclasses import replace

from .normalization import normalize_ocr_text
from .schemas import BBox, Table, TableCell, TextLine, union_bbox


PIPE_ROW_RE = re.compile(r"^\s*\|?.+\|.+\|?\s*$")
MULTISPACE_RE = re.compile(r"\s{2,}")
LIST_CONTINUATION_RE = re.compile(r"^\s*(?:[-–—•]|\(?[A-Za-z0-9]{1,3}[\).]|[०-९]{1,3}[\).])\s+")
MAX_NORMALIZED_GRID_WIDTH = 12


def looks_like_table_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\t" in stripped:
        return True
    if PIPE_ROW_RE.match(stripped):
        return True
    return len(MULTISPACE_RE.split(stripped)) >= 3


def split_table_row(text: str) -> list[str]:
    stripped = text.strip()
    if "\t" in stripped:
        return [part.strip() for part in stripped.split("\t")]
    if "|" in stripped:
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        return [part for part in parts if part]
    return [part.strip() for part in MULTISPACE_RE.split(stripped) if part.strip()]


def infer_table_from_lines(lines: list[TextLine], table_id: str) -> Table:
    cells: list[TableCell] = []
    for row_index, line in enumerate(lines):
        for col_index, text in enumerate(split_table_row(line.text)):
            cells.append(TableCell(row=row_index, col=col_index, text=text, bbox=line.bbox, confidence=line.confidence))
    confidence_values = [line.confidence for line in lines if line.confidence is not None]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    return Table(
        table_id=table_id,
        page_index=lines[0].page_index if lines else 0,
        bbox=union_bbox([line.bbox for line in lines]),
        cells=cells,
        confidence=confidence,
    )


def table_to_grid(table: Table) -> list[list[str]]:
    return cells_to_grid(table.cells)


def cells_to_grid(cells: list[TableCell]) -> list[list[str]]:
    if not cells:
        return []
    rows = max(cell.row + cell.rowspan for cell in cells)
    cols = max(cell.col + cell.colspan for cell in cells)
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        grid[cell.row][cell.col] = cell.text
    return grid


def grid_to_csv(grid: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(grid)
    return output.getvalue()


def grid_to_html(grid: list[list[str]]) -> str:
    rows = []
    for row in grid:
        cells = "".join(f"<td>{html.escape(value)}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def cells_to_html(cells: list[TableCell]) -> str:
    if not cells:
        return "<table>\n</table>"
    rows = max(cell.row + cell.rowspan for cell in cells)
    cols = max(cell.col + cell.colspan for cell in cells)
    by_row: dict[int, dict[int, TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, {})[cell.col] = cell
    covered: set[tuple[int, int]] = set()
    rendered_rows: list[str] = []
    for row_index in range(rows):
        rendered_cells: list[str] = []
        col_index = 0
        while col_index < cols:
            if (row_index, col_index) in covered:
                col_index += 1
                continue
            cell = by_row.get(row_index, {}).get(col_index)
            if cell is None:
                rendered_cells.append("<td></td>")
                col_index += 1
                continue
            attrs = []
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            for row_offset in range(cell.rowspan):
                for col_offset in range(cell.colspan):
                    position = (row_index + row_offset, col_index + col_offset)
                    if position != (row_index, col_index):
                        covered.add(position)
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            rendered_cells.append(f"<td{attr_text}>{html.escape(cell.text)}</td>")
            col_index += cell.colspan
        rendered_rows.append("<tr>" + "".join(rendered_cells) + "</tr>")
    return "<table>\n" + "\n".join(rendered_rows) + "\n</table>"


def infer_repeated_header_colspans(table: Table, *, max_header_rows: int = 2) -> tuple[Table, int]:
    """Collapse adjacent identical header cells into colspan cells.

    This is intentionally narrow: it only looks at the first few rows and only
    collapses a repeated run when a following row contains distinct detail cells
    beneath the run. That avoids rewriting ordinary repeated body values.
    """

    if max_header_rows < 1 or not table.cells:
        return table, 0
    rows: dict[int, list[TableCell]] = {}
    for cell in table.cells:
        rows.setdefault(cell.row, []).append(cell)
    changed = 0
    replaced_cells: list[TableCell] = []
    removed: set[int] = set()
    cell_ids = {id(cell): index for index, cell in enumerate(table.cells)}
    for row_index in sorted(rows):
        row_cells = sorted(rows[row_index], key=lambda cell: cell.col)
        if row_index >= max_header_rows or not _has_distinct_detail_row(rows, row_index):
            continue
        scan_index = 0
        while scan_index < len(row_cells):
            cell = row_cells[scan_index]
            normalized = normalize_ocr_text(cell.text, collapse_spaces=True).strip()
            if not normalized or cell.colspan != 1 or cell.rowspan != 1:
                scan_index += 1
                continue
            run = [cell]
            next_index = scan_index + 1
            while next_index < len(row_cells):
                next_cell = row_cells[next_index]
                next_text = normalize_ocr_text(next_cell.text, collapse_spaces=True).strip()
                if next_cell.col != run[-1].col + run[-1].colspan:
                    break
                if next_cell.colspan != 1 or next_cell.rowspan != 1 or next_text != normalized:
                    break
                run.append(next_cell)
                next_index += 1
            if len(run) < 2 or not _run_has_distinct_children(rows, row_index, run, normalized):
                scan_index += 1
                continue
            boxes = [item.bbox for item in run if item.bbox is not None]
            bbox = union_bbox(boxes) if boxes else cell.bbox
            confidence_values = [item.confidence for item in run if item.confidence is not None]
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else cell.confidence
            replaced_cells.append(replace(cell, colspan=len(run), bbox=bbox, confidence=confidence))
            removed.update(cell_ids[id(item)] for item in run)
            changed += len(run) - 1
            scan_index = next_index
    if not changed:
        return table, 0
    merged_cells: list[TableCell] = []
    inserted_by_position = {(cell.row, cell.col): cell for cell in replaced_cells}
    for index, cell in enumerate(table.cells):
        replacement = inserted_by_position.get((cell.row, cell.col))
        if replacement is not None:
            merged_cells.append(replacement)
            continue
        if index in removed:
            continue
        merged_cells.append(cell)
    metadata = dict(table.metadata)
    metadata["inferred_repeated_header_colspans"] = changed
    return replace(table, cells=merged_cells, html=cells_to_html(merged_cells), metadata=metadata), changed


def normalize_table_grid_structure(table: Table) -> tuple[Table, int]:
    """Remove obvious non-content columns from a reconstructed table.

    The rules are intentionally conservative and reference-free:
    - drop columns that are entirely empty
    - drop columns that only contain separator glyphs
    - drop exact duplicate adjacent columns
    """

    if any(cell.rowspan > 1 or cell.colspan > 1 for cell in table.cells):
        metadata = dict(table.metadata)
        metadata["normalized_table_grid_skipped_reason"] = "explicit_spans"
        return replace(table, metadata=metadata), 0
    grid = table_to_grid(table)
    if not grid:
        return table, 0
    width = max(len(row) for row in grid)
    if width <= 1:
        return table, 0
    if width > MAX_NORMALIZED_GRID_WIDTH:
        return table, 0
    padded = [row + [""] * (width - len(row)) for row in grid]
    keep: list[int] = []
    dropped = 0
    previous_kept_values: list[str] | None = None
    for col_index in range(width):
        values = [normalize_ocr_text(row[col_index], collapse_spaces=True).strip() for row in padded]
        non_empty = [value for value in values if value]
        if not non_empty:
            dropped += 1
            continue
        if all(_is_separator_cell(value) for value in non_empty):
            dropped += 1
            continue
        if previous_kept_values is not None and values == previous_kept_values and any(values):
            dropped += 1
            continue
        keep.append(col_index)
        previous_kept_values = values
    if not keep:
        return table, 0
    normalized_grid = [[row[col_index] for col_index in keep] for row in padded]
    normalized_grid, merged_rows = _merge_sparse_bullet_continuation_rows(normalized_grid)
    if dropped == 0 and merged_rows == 0:
        return table, 0
    cells = cells_from_grid(normalized_grid)
    metadata = dict(table.metadata)
    if dropped:
        metadata["normalized_table_grid_dropped_columns"] = dropped
    if merged_rows:
        metadata["normalized_table_grid_merged_rows"] = merged_rows
    normalized = replace(table, cells=cells, html=cells_to_html(cells), metadata=metadata)
    return normalized, dropped + merged_rows


def _is_separator_cell(value: str) -> bool:
    return bool(value) and all(char in {"-", "–", "—", "_", " "} for char in value)


def _merge_sparse_bullet_continuation_rows(grid: list[list[str]]) -> tuple[list[list[str]], int]:
    if len(grid) < 2:
        return grid, 0
    output: list[list[str]] = []
    merged = 0
    for row in grid:
        non_empty = [(index, normalize_ocr_text(value, collapse_spaces=True).strip()) for index, value in enumerate(row) if value.strip()]
        if output and len(non_empty) == 1:
            source_col, text = non_empty[0]
            target_col = _continuation_target_column(output[-1], source_col=source_col, text=text)
            if target_col is not None:
                output[-1][target_col] = "\n".join(part for part in [output[-1][target_col].strip(), text] if part)
                merged += 1
                continue
        if non_empty:
            output.append(list(row))
        elif not output:
            output.append(list(row))
        else:
            merged += 1
    return output, merged


def _continuation_target_column(previous_row: list[str], *, source_col: int, text: str) -> int | None:
    if source_col != 0 or not _looks_like_list_continuation(text):
        return None
    candidates = [
        (index, normalize_ocr_text(value, collapse_spaces=True).strip())
        for index, value in enumerate(previous_row)
        if index > source_col and value.strip()
    ]
    list_candidates = [item for item in candidates if _looks_like_list_continuation(item[1])]
    if list_candidates:
        return list_candidates[0][0]
    return None


def _looks_like_list_continuation(value: str) -> bool:
    return bool(LIST_CONTINUATION_RE.match(value))


def _has_distinct_detail_row(rows: dict[int, list[TableCell]], row_index: int) -> bool:
    current_cols = {cell.col for cell in rows.get(row_index, []) if cell.text.strip()}
    next_cols = {cell.col for cell in rows.get(row_index + 1, []) if cell.text.strip()}
    return len(next_cols) >= max(2, len(current_cols))


def _run_has_distinct_children(rows: dict[int, list[TableCell]], row_index: int, run: list[TableCell], normalized: str) -> bool:
    child_by_col = {cell.col: cell for cell in rows.get(row_index + 1, [])}
    child_texts: list[str] = []
    for col in range(run[0].col, run[-1].col + 1):
        child = child_by_col.get(col)
        if child is None:
            return False
        child_text = normalize_ocr_text(child.text, collapse_spaces=True).strip()
        if not child_text or child_text == normalized:
            return False
        child_texts.append(child_text)
    return len(set(child_texts)) > 1


def table_to_markdown(table: Table) -> str:
    grid = table_to_grid(table)
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    padded = [row + [""] * (width - len(row)) for row in grid]
    header = padded[0]
    separator = ["---"] * width
    body = padded[1:]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(value.replace("\n", " ") for value in row) + " |"

    return "\n".join([render_row(header), render_row(separator), *[render_row(row) for row in body]])


def write_table_files(table: Table, out_dir: Path) -> Table:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = table_to_grid(table)
    csv_text = grid_to_csv(grid)
    html_text = table.html or cells_to_html(table.cells)
    csv_path = out_dir / f"{table.table_id}.csv"
    html_path = out_dir / f"{table.table_id}.html"
    csv_path.write_text(csv_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    table.csv_path = str(csv_path)
    table.html_path = str(html_path)
    table.html = html_text
    return table


def cells_from_grid(grid: list[list[str]]) -> list[TableCell]:
    cells: list[TableCell] = []
    for row_index, row in enumerate(grid):
        for col_index, text in enumerate(row):
            cells.append(TableCell(row=row_index, col=col_index, text=text, bbox=BBox(0, 0, 0, 0)))
    return cells
