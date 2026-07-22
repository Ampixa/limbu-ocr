"""Deterministic OCR and document-structure metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .normalization import normalize_ocr_text
from .schemas import BBox, TableCell


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    left_seq = list(left)
    right_seq = list(right)
    if not left_seq:
        return len(right_seq)
    if not right_seq:
        return len(left_seq)
    previous = list(range(len(right_seq) + 1))
    for i, left_item in enumerate(left_seq, start=1):
        current = [i]
        for j, right_item in enumerate(right_seq, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(predicted: str, reference: str, *, normalized: bool = True) -> float:
    pred = normalize_ocr_text(predicted) if normalized else predicted
    ref = normalize_ocr_text(reference) if normalized else reference
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / len(ref)


def wer(predicted: str, reference: str, *, normalized: bool = True) -> float:
    pred = normalize_ocr_text(predicted, collapse_spaces=True) if normalized else predicted
    ref = normalize_ocr_text(reference, collapse_spaces=True) if normalized else reference
    pred_words = pred.split()
    ref_words = ref.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(pred_words, ref_words) / len(ref_words)


def exact_line_accuracy(predicted: Iterable[str], reference: Iterable[str], *, normalized: bool = True) -> float:
    predicted_lines = list(predicted)
    reference_lines = list(reference)
    if not reference_lines:
        return 1.0 if not predicted_lines else 0.0
    matches = 0
    for pred, ref in zip(predicted_lines, reference_lines, strict=False):
        if normalized:
            pred = normalize_ocr_text(pred, collapse_spaces=True)
            ref = normalize_ocr_text(ref, collapse_spaces=True)
        if pred == ref:
            matches += 1
    return matches / len(reference_lines)


def reading_order_pair_accuracy(predicted_ids: list[str], reference_ids: list[str]) -> float:
    if len(reference_ids) < 2:
        return 1.0
    pred_position = {item: index for index, item in enumerate(predicted_ids)}
    correct = 0
    total = 0
    for i, left in enumerate(reference_ids):
        for right in reference_ids[i + 1 :]:
            if left not in pred_position or right not in pred_position:
                total += 1
                continue
            total += 1
            if pred_position[left] < pred_position[right]:
                correct += 1
    return correct / total if total else 1.0


@dataclass(slots=True)
class TableScore:
    precision: float
    recall: float
    f1: float
    exact_match: float
    structure_similarity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_match": self.exact_match,
            "structure_similarity": self.structure_similarity,
        }


def table_cell_score(predicted: list[list[str]], reference: list[list[str]]) -> TableScore:
    pred_counter = Counter((r, c, normalize_ocr_text(value, collapse_spaces=True)) for r, row in enumerate(predicted) for c, value in enumerate(row))
    ref_counter = Counter((r, c, normalize_ocr_text(value, collapse_spaces=True)) for r, row in enumerate(reference) for c, value in enumerate(row))
    overlap = sum((pred_counter & ref_counter).values())
    predicted_total = sum(pred_counter.values())
    reference_total = sum(ref_counter.values())
    precision = overlap / predicted_total if predicted_total else (1.0 if reference_total == 0 else 0.0)
    recall = overlap / reference_total if reference_total else (1.0 if predicted_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = 1.0 if predicted == reference else 0.0
    return TableScore(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact,
        structure_similarity=table_structure_similarity(predicted, reference),
    )


TableLike = list[list[str]] | list[TableCell]


@dataclass(slots=True)
class _TableTreeNode:
    label: str
    children: list["_TableTreeNode"] = field(default_factory=list)


def table_structure_similarity(predicted: TableLike, reference: TableLike) -> float:
    """TEDS-style table-structure similarity over rows, columns, and occupied cells.

    For rectangular grid inputs, this scores row count, per-row column count,
    and non-empty cell coordinates. For TableCell inputs, it also scores
    rowspans and colspans. Text content is ignored.
    """

    pred_tokens = _table_structure_tokens(predicted)
    ref_tokens = _table_structure_tokens(reference)
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    distance = edit_distance(pred_tokens, ref_tokens)
    return max(0.0, 1.0 - distance / len(ref_tokens))


def table_tree_similarity(predicted: TableLike, reference: TableLike, *, include_text: bool = False) -> float:
    """Ordered table-tree edit similarity over table/tr/td nodes.

    This is a deterministic TEDS-style score. Structure mode compares the table
    DOM shape and rowspan/colspan attributes. Text mode also compares normalized
    cell text in td node labels.
    """

    pred_tree = _table_tree(predicted, include_text=include_text)
    ref_tree = _table_tree(reference, include_text=include_text)
    pred_size = _tree_size(pred_tree)
    ref_size = _tree_size(ref_tree)
    if ref_size == 0:
        return 1.0 if pred_size == 0 else 0.0
    distance = _tree_edit_distance(pred_tree, ref_tree)
    return max(0.0, 1.0 - distance / max(pred_size, ref_size))


def table_teds_structure(predicted: TableLike, reference: TableLike) -> float:
    return table_tree_similarity(predicted, reference, include_text=False)


def table_teds_text(predicted: TableLike, reference: TableLike) -> float:
    return table_tree_similarity(predicted, reference, include_text=True)


def _table_structure_tokens(table: TableLike) -> list[str]:
    if _is_cell_list(table):
        return _cell_structure_tokens(table)
    grid = table
    tokens: list[str] = [f"rows:{len(grid)}"]
    for row_index, row in enumerate(grid):
        tokens.append(f"row:{row_index}:cols:{len(row)}")
        for col_index, value in enumerate(row):
            if normalize_ocr_text(value, collapse_spaces=True):
                tokens.append(f"cell:{row_index}:{col_index}:rs:1:cs:1")
    return tokens


def _cell_structure_tokens(cells: list[TableCell]) -> list[str]:
    if not cells:
        return []
    rows = max(cell.row + cell.rowspan for cell in cells)
    cols = max(cell.col + cell.colspan for cell in cells)
    tokens = [f"rows:{rows}", f"cols:{cols}"]
    for cell in sorted(cells, key=lambda item: (item.row, item.col, item.rowspan, item.colspan)):
        if normalize_ocr_text(cell.text, collapse_spaces=True):
            tokens.append(f"cell:{cell.row}:{cell.col}:rs:{cell.rowspan}:cs:{cell.colspan}")
    return tokens


def _is_cell_list(value: TableLike) -> bool:
    return bool(value) and all(isinstance(item, TableCell) for item in value)


def _table_tree(table: TableLike, *, include_text: bool) -> _TableTreeNode:
    cells = _cells_from_table_like(table)
    if not cells:
        return _TableTreeNode("table")
    rows = max(cell.row + cell.rowspan for cell in cells)
    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, []).append(cell)
    row_nodes: list[_TableTreeNode] = []
    for row_index in range(rows):
        cell_nodes = []
        for cell in sorted(by_row.get(row_index, []), key=lambda item: item.col):
            label = f"td:rs:{cell.rowspan}:cs:{cell.colspan}"
            if include_text:
                text = normalize_ocr_text(cell.text, collapse_spaces=True)
                label = f"{label}:text:{text}"
            cell_nodes.append(_TableTreeNode(label))
        row_nodes.append(_TableTreeNode("tr", cell_nodes))
    return _TableTreeNode("table", row_nodes)


def _cells_from_table_like(table: TableLike) -> list[TableCell]:
    if _is_cell_list(table):
        return list(table)
    cells: list[TableCell] = []
    for row_index, row in enumerate(table):
        for col_index, value in enumerate(row):
            cells.append(TableCell(row=row_index, col=col_index, text=str(value)))
    return cells


def _tree_size(node: _TableTreeNode) -> int:
    return 1 + sum(_tree_size(child) for child in node.children)


def _tree_edit_distance(left: _TableTreeNode, right: _TableTreeNode) -> int:
    relabel_cost = 0 if left.label == right.label else 1
    return relabel_cost + _forest_edit_distance(left.children, right.children)


def _forest_edit_distance(left: list[_TableTreeNode], right: list[_TableTreeNode]) -> int:
    previous = [0]
    for node in right:
        previous.append(previous[-1] + _tree_size(node))
    for left_index, left_node in enumerate(left, start=1):
        current = [previous[0] + _tree_size(left_node)]
        for right_index, right_node in enumerate(right, start=1):
            delete_cost = previous[right_index] + _tree_size(left_node)
            insert_cost = current[right_index - 1] + _tree_size(right_node)
            replace_cost = previous[right_index - 1] + _tree_edit_distance(left_node, right_node)
            current.append(min(delete_cost, insert_cost, replace_cost))
        previous = current
    return previous[-1]


def bbox_iou(left: BBox, right: BBox) -> float:
    inter_left = max(left.x, right.x)
    inter_top = max(left.y, right.y)
    inter_right = min(left.right, right.right)
    inter_bottom = min(left.bottom, right.bottom)
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    intersection = inter_w * inter_h
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class DetectionScore:
    precision: float
    recall: float
    f1: float
    matches: int
    predicted: int
    reference: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "matches": self.matches,
            "predicted": self.predicted,
            "reference": self.reference,
        }


def detection_score(predicted_count: int, reference_count: int, matches: int) -> DetectionScore:
    precision = matches / predicted_count if predicted_count else (1.0 if reference_count == 0 else 0.0)
    recall = matches / reference_count if reference_count else (1.0 if predicted_count == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScore(precision=precision, recall=recall, f1=f1, matches=matches, predicted=predicted_count, reference=reference_count)
