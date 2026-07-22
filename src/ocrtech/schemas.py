"""Stable JSON-compatible schemas for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError


JsonDict = dict[str, Any]


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataValidationError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


@dataclass(slots=True)
class BBox:
    """Axis-aligned bounding box in page coordinates."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        self.x = _number(self.x, "bbox.x")
        self.y = _number(self.y, "bbox.y")
        self.w = _number(self.w, "bbox.w")
        self.h = _number(self.h, "bbox.h")
        if self.w < 0 or self.h < 0:
            raise DataValidationError(f"bbox width/height must be non-negative: {self.to_list()}")

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    def to_dict(self) -> JsonDict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_any(cls, value: Any) -> "BBox":
        if isinstance(value, BBox):
            return value
        if isinstance(value, dict):
            return cls(value.get("x"), value.get("y"), value.get("w"), value.get("h"))
        if isinstance(value, list | tuple) and len(value) == 4:
            return cls(value[0], value[1], value[2], value[3])
        raise DataValidationError(f"bbox must be a dict or 4-item list, got {value!r}")


def union_bbox(boxes: list[BBox]) -> BBox:
    if not boxes:
        return BBox(0, 0, 0, 0)
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BBox(left, top, right - left, bottom - top)


def validate_confidence(value: float | int | None, field_name: str = "confidence") -> float | None:
    if value is None:
        return None
    number = _number(value, field_name)
    if number < 0 or number > 1:
        raise DataValidationError(f"{field_name} must be between 0 and 1, got {number}")
    return number


@dataclass(slots=True)
class TextLine:
    text: str
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))
    confidence: float | None = None
    page_index: int = 0
    line_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise DataValidationError(f"text line must be a string, got {type(self.text).__name__}")
        self.bbox = BBox.from_any(self.bbox)
        self.confidence = validate_confidence(self.confidence)
        if self.page_index < 0:
            raise DataValidationError(f"page_index must be non-negative, got {self.page_index}")

    def to_dict(self) -> JsonDict:
        return {
            "line_id": self.line_id,
            "page_index": self.page_index,
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict, page_index: int = 0) -> "TextLine":
        return cls(
            text=data.get("text", ""),
            bbox=BBox.from_any(data.get("bbox", [0, 0, 0, 0])),
            confidence=data.get("confidence"),
            page_index=int(data.get("page_index", page_index)),
            line_id=data.get("line_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class TableCell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    bbox: BBox | None = None
    confidence: float | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row < 0 or self.col < 0:
            raise DataValidationError("table cell row/col must be non-negative")
        if self.rowspan < 1 or self.colspan < 1:
            raise DataValidationError("table cell rowspan/colspan must be at least 1")
        if not isinstance(self.text, str):
            raise DataValidationError("table cell text must be a string")
        if self.bbox is not None:
            self.bbox = BBox.from_any(self.bbox)
        self.confidence = validate_confidence(self.confidence)

    def to_dict(self) -> JsonDict:
        return {
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "text": self.text,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TableCell":
        bbox = data.get("bbox")
        return cls(
            row=int(data.get("row", 0)),
            col=int(data.get("col", 0)),
            text=str(data.get("text", "")),
            rowspan=int(data.get("rowspan", 1)),
            colspan=int(data.get("colspan", 1)),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Table:
    table_id: str
    page_index: int
    bbox: BBox
    cells: list[TableCell] = field(default_factory=list)
    html: str | None = None
    csv_path: str | None = None
    html_path: str | None = None
    caption: str | None = None
    confidence: float | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = BBox.from_any(self.bbox)
        self.confidence = validate_confidence(self.confidence)
        if not self.table_id:
            raise DataValidationError("table_id is required")

    def to_dict(self) -> JsonDict:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "bbox": self.bbox.to_list(),
            "cells": [cell.to_dict() for cell in self.cells],
            "html": self.html,
            "csv_path": self.csv_path,
            "html_path": self.html_path,
            "caption": self.caption,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Table":
        return cls(
            table_id=str(data.get("table_id") or data.get("id") or "table-0"),
            page_index=int(data.get("page_index", 0)),
            bbox=BBox.from_any(data.get("bbox", [0, 0, 0, 0])),
            cells=[TableCell.from_dict(item) for item in data.get("cells", [])],
            html=data.get("html"),
            csv_path=data.get("csv_path"),
            html_path=data.get("html_path"),
            caption=data.get("caption"),
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Figure:
    figure_id: str
    page_index: int
    bbox: BBox
    caption: str | None = None
    summary: str | None = None
    image_path: str | None = None
    confidence: float | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.figure_id:
            raise DataValidationError("figure_id is required")
        self.bbox = BBox.from_any(self.bbox)
        self.confidence = validate_confidence(self.confidence)

    def to_dict(self) -> JsonDict:
        return {
            "figure_id": self.figure_id,
            "page_index": self.page_index,
            "bbox": self.bbox.to_list(),
            "caption": self.caption,
            "summary": self.summary,
            "image_path": self.image_path,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Figure":
        return cls(
            figure_id=str(data.get("figure_id") or data.get("id") or "figure-0"),
            page_index=int(data.get("page_index", 0)),
            bbox=BBox.from_any(data.get("bbox", [0, 0, 0, 0])),
            caption=data.get("caption"),
            summary=data.get("summary"),
            image_path=data.get("image_path"),
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Block:
    block_id: str
    block_type: str
    page_index: int
    bbox: BBox
    order: int
    text: str = ""
    confidence: float | None = None
    line_ids: list[str] = field(default_factory=list)
    table_id: str | None = None
    figure_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_type not in {"text", "title", "table", "figure", "caption"}:
            raise DataValidationError(f"unsupported block_type: {self.block_type}")
        self.bbox = BBox.from_any(self.bbox)
        self.confidence = validate_confidence(self.confidence)

    def to_dict(self) -> JsonDict:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "page_index": self.page_index,
            "bbox": self.bbox.to_list(),
            "order": self.order,
            "text": self.text,
            "confidence": self.confidence,
            "line_ids": self.line_ids,
            "table_id": self.table_id,
            "figure_id": self.figure_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Block":
        return cls(
            block_id=str(data.get("block_id") or data.get("id") or "block-0"),
            block_type=str(data.get("block_type", "text")),
            page_index=int(data.get("page_index", 0)),
            bbox=BBox.from_any(data.get("bbox", [0, 0, 0, 0])),
            order=int(data.get("order", 0)),
            text=str(data.get("text", "")),
            confidence=data.get("confidence"),
            line_ids=list(data.get("line_ids", [])),
            table_id=data.get("table_id"),
            figure_id=data.get("figure_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Page:
    page_index: int
    width: float | None = None
    height: float | None = None
    text_lines: list[TextLine] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "text_lines": [line.to_dict() for line in self.text_lines],
            "blocks": [block.to_dict() for block in self.blocks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Page":
        page_index = int(data.get("page_index", data.get("index", 0)))
        return cls(
            page_index=page_index,
            width=data.get("width"),
            height=data.get("height"),
            text_lines=[TextLine.from_dict(item, page_index) for item in data.get("text_lines", data.get("lines", []))],
            blocks=[Block.from_dict(item) for item in data.get("blocks", [])],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Document:
    source_path: str
    pages: list[Page] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": "1.0",
            "source_path": self.source_path,
            "pages": [page.to_dict() for page in self.pages],
            "tables": [table.to_dict() for table in self.tables],
            "figures": [figure.to_dict() for figure in self.figures],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Document":
        return cls(
            source_path=str(data.get("source_path", "")),
            pages=[Page.from_dict(item) for item in data.get("pages", [])],
            tables=[Table.from_dict(item) for item in data.get("tables", [])],
            figures=[Figure.from_dict(item) for item in data.get("figures", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    @property
    def text(self) -> str:
        lines: list[str] = []
        for page in sorted(self.pages, key=lambda item: item.page_index):
            for block in sorted(page.blocks, key=lambda item: item.order):
                if block.text:
                    lines.append(block.text)
        return "\n".join(lines)


def source_name(path: str | Path) -> str:
    return Path(path).name
