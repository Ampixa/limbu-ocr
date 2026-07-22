"""Quality and confidence reporting for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import Document


@dataclass(slots=True)
class QualityIssue:
    severity: str
    code: str
    message: str
    sample: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "message": self.message, "sample": self.sample}


@dataclass(slots=True)
class DocumentQuality:
    passed: bool
    quality_score: float
    metrics: dict[str, float | int | None]
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "quality_score": self.quality_score,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def evaluate_document_quality(
    document: Document,
    *,
    min_line_confidence: float = 0.80,
    min_quality_score: float = 0.70,
) -> DocumentQuality:
    lines = [line for page in document.pages for line in page.text_lines]
    blocks = [block for page in document.pages for block in page.blocks]
    confidences = [line.confidence for line in lines if line.confidence is not None]
    low_confidence = [line for line in lines if line.confidence is not None and line.confidence < min_line_confidence]
    missing_confidence = [line for line in lines if line.confidence is None]
    empty_lines = [line for line in lines if not line.text.strip()]
    tables_without_cells = [table for table in document.tables if not table.cells]
    blocks_without_lines = [block for block in blocks if block.block_type in {"text", "title", "table"} and not block.line_ids]

    metrics: dict[str, float | int | None] = {
        "page_count": len(document.pages),
        "line_count": len(lines),
        "block_count": len(blocks),
        "table_count": len(document.tables),
        "figure_count": len(document.figures),
        "average_line_confidence": sum(confidences) / len(confidences) if confidences else None,
        "minimum_line_confidence": min(confidences) if confidences else None,
        "low_confidence_line_count": len(low_confidence),
        "missing_confidence_line_count": len(missing_confidence),
        "empty_text_line_count": len(empty_lines),
        "tables_without_cells": len(tables_without_cells),
        "blocks_without_lines": len(blocks_without_lines),
    }
    issues: list[QualityIssue] = []
    if not document.pages:
        issues.append(QualityIssue("error", "no_pages", "document contains no pages"))
    if not lines:
        issues.append(QualityIssue("error", "no_text_lines", "document contains no OCR text lines"))
    if blocks and not document.text.strip():
        issues.append(QualityIssue("error", "empty_document_text", "document blocks contain no text"))
    if missing_confidence:
        issues.append(
            QualityIssue(
                "warning",
                "missing_line_confidence",
                f"{len(missing_confidence)} text lines have no confidence score",
                sample=missing_confidence[0].line_id,
            )
        )
    if low_confidence:
        issues.append(
            QualityIssue(
                "warning",
                "low_line_confidence",
                f"{len(low_confidence)} text lines are below confidence threshold {min_line_confidence}",
                sample=low_confidence[0].line_id,
            )
        )
    if tables_without_cells:
        issues.append(
            QualityIssue(
                "warning",
                "table_without_cells",
                f"{len(tables_without_cells)} tables have no cells",
                sample=tables_without_cells[0].table_id,
            )
        )
    if blocks_without_lines:
        issues.append(
            QualityIssue(
                "warning",
                "block_without_lines",
                f"{len(blocks_without_lines)} text/table blocks have no source line ids",
                sample=blocks_without_lines[0].block_id,
            )
        )

    score = _quality_score(metrics, len(lines))
    if score < min_quality_score:
        issues.append(QualityIssue("error", "quality_score_below_threshold", f"quality score {score:.4f} < required {min_quality_score:.4f}"))
    passed = not any(issue.severity == "error" for issue in issues)
    return DocumentQuality(passed=passed, quality_score=score, metrics=metrics, issues=issues)


def _quality_score(metrics: dict[str, float | int | None], line_count: int) -> float:
    if line_count <= 0:
        return 0.0
    penalty = 0.0
    low = int(metrics["low_confidence_line_count"] or 0)
    missing = int(metrics["missing_confidence_line_count"] or 0)
    empty = int(metrics["empty_text_line_count"] or 0)
    tables_without_cells = int(metrics["tables_without_cells"] or 0)
    blocks_without_lines = int(metrics["blocks_without_lines"] or 0)
    penalty += low / line_count * 0.35
    penalty += missing / line_count * 0.20
    penalty += empty / line_count * 0.20
    penalty += min(0.15, tables_without_cells * 0.05)
    penalty += min(0.10, blocks_without_lines * 0.02)
    confidence = metrics.get("average_line_confidence")
    if isinstance(confidence, int | float):
        penalty += max(0.0, 0.95 - float(confidence)) * 0.20
    return round(max(0.0, min(1.0, 1.0 - penalty)), 6)
