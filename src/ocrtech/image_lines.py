"""Image-derived line box OCR helpers for page-level pipelines."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engines import SCRIPT_CODEPOINT_RANGES, _script_codepoint_count, _script_ratio
from .errors import EngineUnavailableError, ParseError
from .manifest import sha256_file
from .normalization import normalize_ocr_text
from .schemas import BBox, Block, Document, Page, TextLine

READING_ORDER_MODES = (
    "top_to_bottom",
    "column_major",
    "wide_column_right_to_left",
    "wide_column_right_to_left_split_tall_last",
    "auto_layout",
    "auto_layout_split_tall_last",
)
FRONT_MATTER_STRUCTURAL_ROLES = {"page_marker"}
FRONT_MATTER_DIGIT_PREFIX_RE = re.compile(r"^\s*[०-९]\s+\S")
_DEVANAGARI_DIGITS = set("०१२३४५६७८९")
_ASCII_DIGITS = set("0123456789")
_IMAGE_LINE_REQUIRE_SCRIPT_KEYS = {"any", "limbu_or_devanagari", *SCRIPT_CODEPOINT_RANGES}


@dataclass(frozen=True)
class ImageLineDetectionConfig:
    threshold: str = "otsu"
    bbox_source: str = "ink"
    reading_order: str = "auto_layout"
    horizontal_kernel: int = 23
    vertical_kernel: int = 3
    dilation_iterations: int = 1
    min_width: int = 35
    min_height: int = 10
    min_area: int = 100
    max_height: int = 140
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    detector_padding: int = 2
    crop_padding: int = 12
    rescue_detector_passes: tuple[dict[str, Any], ...] = ()
    merge_iou_threshold: float = 0.80
    split_tall_components: bool = False
    split_tall_row_min_ink: int = 2
    split_tall_max_row_gap: int = 4
    split_wide_components: bool = False
    split_wide_col_min_ink: int = 2
    split_wide_max_col_gap: int = 24
    split_wide_min_width: int = 600
    split_detected_row_components: bool = False
    split_detected_row_col_min_ink: int = 2
    split_detected_row_max_col_gap: int = 24
    split_detected_row_min_width: int = 600
    split_detected_row_min_segment_width: int = 40
    split_detected_tall_components: bool = False
    split_detected_tall_row_min_ink: int = 20
    split_detected_tall_max_row_gap: int = 4
    split_detected_tall_min_height: int = 90
    split_detected_tall_min_segment_height: int = 24
    merge_same_row_components: bool = False
    merge_same_row_y_tolerance: float = 8.0
    merge_same_row_max_gap: float = 120.0
    merge_same_row_max_center_delta: float = 300.0
    merge_same_row_max_width: float = 420.0
    merge_same_row_auto_fragmented_top_to_bottom: bool = False
    merge_same_row_auto_min_reduction_ratio: float = 0.18
    merge_same_row_auto_min_reduction_count: int = 8

    def __post_init__(self) -> None:
        if self.merge_same_row_y_tolerance < 0:
            raise ParseError(
                f"image-line merge_same_row_y_tolerance must be non-negative, got {self.merge_same_row_y_tolerance}"
            )
        if self.split_wide_col_min_ink <= 0:
            raise ParseError(f"image-line split_wide_col_min_ink must be positive, got {self.split_wide_col_min_ink}")
        if self.min_aspect_ratio < 0:
            raise ParseError(f"image-line min_aspect_ratio must be non-negative, got {self.min_aspect_ratio}")
        if self.max_aspect_ratio < 0:
            raise ParseError(f"image-line max_aspect_ratio must be non-negative, got {self.max_aspect_ratio}")
        if self.split_wide_max_col_gap < 0:
            raise ParseError(
                f"image-line split_wide_max_col_gap must be non-negative, got {self.split_wide_max_col_gap}"
            )
        if self.split_wide_min_width <= 0:
            raise ParseError(f"image-line split_wide_min_width must be positive, got {self.split_wide_min_width}")
        if self.split_detected_row_col_min_ink <= 0:
            raise ParseError(
                "image-line split_detected_row_col_min_ink must be positive, "
                f"got {self.split_detected_row_col_min_ink}"
            )
        if self.split_detected_row_max_col_gap < 0:
            raise ParseError(
                "image-line split_detected_row_max_col_gap must be non-negative, "
                f"got {self.split_detected_row_max_col_gap}"
            )
        if self.split_detected_row_min_width <= 0:
            raise ParseError(
                "image-line split_detected_row_min_width must be positive, "
                f"got {self.split_detected_row_min_width}"
            )
        if self.split_detected_row_min_segment_width <= 0:
            raise ParseError(
                "image-line split_detected_row_min_segment_width must be positive, "
                f"got {self.split_detected_row_min_segment_width}"
            )
        if self.split_detected_tall_row_min_ink <= 0:
            raise ParseError(
                "image-line split_detected_tall_row_min_ink must be positive, "
                f"got {self.split_detected_tall_row_min_ink}"
            )
        if self.split_detected_tall_max_row_gap < 0:
            raise ParseError(
                "image-line split_detected_tall_max_row_gap must be non-negative, "
                f"got {self.split_detected_tall_max_row_gap}"
            )
        if self.split_detected_tall_min_height <= 0:
            raise ParseError(
                "image-line split_detected_tall_min_height must be positive, "
                f"got {self.split_detected_tall_min_height}"
            )
        if self.split_detected_tall_min_segment_height <= 0:
            raise ParseError(
                "image-line split_detected_tall_min_segment_height must be positive, "
                f"got {self.split_detected_tall_min_segment_height}"
            )
        if self.merge_same_row_max_gap < 0:
            raise ParseError(f"image-line merge_same_row_max_gap must be non-negative, got {self.merge_same_row_max_gap}")
        if self.merge_same_row_max_center_delta <= 0:
            raise ParseError(
                "image-line merge_same_row_max_center_delta must be positive, "
                f"got {self.merge_same_row_max_center_delta}"
            )
        if self.merge_same_row_max_width <= 0:
            raise ParseError(f"image-line merge_same_row_max_width must be positive, got {self.merge_same_row_max_width}")
        if not 0 <= self.merge_same_row_auto_min_reduction_ratio <= 1:
            raise ParseError(
                "image-line merge_same_row_auto_min_reduction_ratio must be between 0 and 1, "
                f"got {self.merge_same_row_auto_min_reduction_ratio}"
            )
        if self.merge_same_row_auto_min_reduction_count < 0:
            raise ParseError(
                "image-line merge_same_row_auto_min_reduction_count must be non-negative, "
                f"got {self.merge_same_row_auto_min_reduction_count}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "bbox_source": self.bbox_source,
            "reading_order": self.reading_order,
            "horizontal_kernel": self.horizontal_kernel,
            "vertical_kernel": self.vertical_kernel,
            "dilation_iterations": self.dilation_iterations,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_area": self.min_area,
            "max_height": self.max_height,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "detector_padding": self.detector_padding,
            "crop_padding": self.crop_padding,
            "rescue_detector_passes": list(self.rescue_detector_passes),
            "merge_iou_threshold": self.merge_iou_threshold,
            "split_tall_components": self.split_tall_components,
            "split_tall_row_min_ink": self.split_tall_row_min_ink,
            "split_tall_max_row_gap": self.split_tall_max_row_gap,
            "split_wide_components": self.split_wide_components,
            "split_wide_col_min_ink": self.split_wide_col_min_ink,
            "split_wide_max_col_gap": self.split_wide_max_col_gap,
            "split_wide_min_width": self.split_wide_min_width,
            "split_detected_row_components": self.split_detected_row_components,
            "split_detected_row_col_min_ink": self.split_detected_row_col_min_ink,
            "split_detected_row_max_col_gap": self.split_detected_row_max_col_gap,
            "split_detected_row_min_width": self.split_detected_row_min_width,
            "split_detected_row_min_segment_width": self.split_detected_row_min_segment_width,
            "split_detected_tall_components": self.split_detected_tall_components,
            "split_detected_tall_row_min_ink": self.split_detected_tall_row_min_ink,
            "split_detected_tall_max_row_gap": self.split_detected_tall_max_row_gap,
            "split_detected_tall_min_height": self.split_detected_tall_min_height,
            "split_detected_tall_min_segment_height": self.split_detected_tall_min_segment_height,
            "merge_same_row_components": self.merge_same_row_components,
            "merge_same_row_y_tolerance": self.merge_same_row_y_tolerance,
            "merge_same_row_max_gap": self.merge_same_row_max_gap,
            "merge_same_row_max_center_delta": self.merge_same_row_max_center_delta,
            "merge_same_row_max_width": self.merge_same_row_max_width,
            "merge_same_row_auto_fragmented_top_to_bottom": self.merge_same_row_auto_fragmented_top_to_bottom,
            "merge_same_row_auto_min_reduction_ratio": self.merge_same_row_auto_min_reduction_ratio,
            "merge_same_row_auto_min_reduction_count": self.merge_same_row_auto_min_reduction_count,
        }


@dataclass(frozen=True)
class ImageLineFilterConfig:
    drop_empty: bool = False
    min_confidence: float | None = None
    require_script: str = "any"
    script_ratio_threshold: float = 0.20
    min_width_ratio: float | None = None
    max_width_ratio: float | None = None
    min_height_ratio: float | None = None
    max_height_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.min_confidence is not None and not 0 <= self.min_confidence <= 1:
            raise ParseError(f"image-line min_confidence must be between 0 and 1, got {self.min_confidence}")
        if self.require_script not in _IMAGE_LINE_REQUIRE_SCRIPT_KEYS:
            raise ParseError(f"unsupported image-line require_script: {self.require_script!r}")
        if not 0 <= self.script_ratio_threshold <= 1:
            raise ParseError(f"image-line script_ratio_threshold must be between 0 and 1, got {self.script_ratio_threshold}")
        for name, value in (
            ("min_width_ratio", self.min_width_ratio),
            ("max_width_ratio", self.max_width_ratio),
            ("min_height_ratio", self.min_height_ratio),
            ("max_height_ratio", self.max_height_ratio),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ParseError(f"image-line {name} must be between 0 and 1, got {value}")

    @property
    def enabled(self) -> bool:
        return bool(
            self.drop_empty
            or self.min_confidence is not None
            or self.require_script != "any"
            or self.min_width_ratio is not None
            or self.max_width_ratio is not None
            or self.min_height_ratio is not None
            or self.max_height_ratio is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "drop_empty": self.drop_empty,
            "min_confidence": self.min_confidence,
            "require_script": self.require_script,
            "script_ratio_threshold": self.script_ratio_threshold,
            "min_width_ratio": self.min_width_ratio,
            "max_width_ratio": self.max_width_ratio,
            "min_height_ratio": self.min_height_ratio,
            "max_height_ratio": self.max_height_ratio,
        }


@dataclass(frozen=True)
class DetectedLineBox:
    line_id: str
    bbox: BBox
    area: int
    centroid: tuple[float, float]
    source_pass: str = "primary"


@dataclass(frozen=True)
class ReadingOrderSelection:
    requested: str
    selected: str
    reason: str
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
            "features": self.features,
        }


def build_image_line_document(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    engine: object,
    model_config: str | Path | None,
    detection_config: ImageLineDetectionConfig,
    filter_config: ImageLineFilterConfig | None = None,
    allow_empty_lines: bool = False,
) -> tuple[Document, dict[str, Any]]:
    """Detect line-like boxes in an image, recognize crops, and build a document.

    The function records both kept and removed line decisions in metadata so callers
    can export an auditable OCR bundle. It does not use reference text or boxes.
    """

    source = Path(input_path)
    out = Path(output_dir)
    if not source.is_file():
        raise ParseError(f"image-line OCR input must be an image file: {source}")
    filter_config = filter_config or ImageLineFilterConfig()
    effective_allow_empty = allow_empty_lines or filter_config.drop_empty
    detected, reading_order_selection = detect_line_boxes_with_audit(source, detection_config)
    if not detected:
        raise ParseError(f"image-line detector found no line boxes: {source}")
    crops_dir = out / "image-line-crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    text_lines: list[TextLine] = []
    blocks: list[Block] = []
    crop_rows: list[dict[str, Any]] = []
    removal_counts: dict[str, int] = {}
    empty_line_ids: list[str] = []
    kept_count = 0
    removed_count = 0
    image_module = _pillow_image_module()
    with image_module.open(source) as raw_image:
        image = raw_image.convert("RGB")
        image_width = image.width
        image_height = image.height
        for order, line_box in enumerate(detected):
            crop_path = crops_dir / f"{order + 1:04d}-{_safe_name(line_box.line_id)}.png"
            crop_xyxy = _crop_box(line_box.bbox, width=image_width, height=image_height, padding=detection_config.crop_padding)
            image.crop(crop_xyxy).save(crop_path)
            text, confidence, raw_prediction_lines = _recognize_crop(engine, crop_path)
            if not text:
                empty_line_ids.append(line_box.line_id)
                if not effective_allow_empty:
                    raise ParseError(
                        f"recognizer returned empty text for detected line {line_box.line_id}; "
                        "enable image-line empty-line handling or filtering to continue"
                    )
            reasons, script_profile = _line_removal_reasons(
                text,
                confidence,
                line_box.bbox,
                image_width=image_width,
                image_height=image_height,
                filter_config=filter_config,
            )
            structural_profile = _line_structural_profile(
                text,
                line_box.bbox,
                image_width=image_width,
                image_height=image_height,
            )
            kept = not reasons
            if kept:
                kept_count += 1
                line_id = f"img-line-{kept_count:04d}"
                metadata = {
                    "image_line_detector": "morph_line_detector_v1",
                    "original_detected_line_id": line_box.line_id,
                    "crop_path": str(crop_path),
                    "crop_sha256": sha256_file(crop_path),
                    "crop_xyxy": list(crop_xyxy),
                    "crop_padding_px": detection_config.crop_padding,
                    "detected_area": line_box.area,
                    "detected_centroid": list(line_box.centroid),
                    "detector_source_pass": line_box.source_pass,
                    "reading_order": reading_order_selection.selected,
                    "requested_reading_order": reading_order_selection.requested,
                    "reading_order_selection": reading_order_selection.to_dict(),
                    "model_config": str(model_config) if model_config is not None else None,
                    "raw_prediction_lines": raw_prediction_lines,
                    "image_line_filter": filter_config.to_dict(),
                    "image_line_filter_reasons": reasons,
                    **structural_profile,
                }
                text_line = TextLine(
                    text=text,
                    bbox=line_box.bbox,
                    confidence=confidence,
                    page_index=0,
                    line_id=line_id,
                    metadata=metadata,
                )
                text_lines.append(text_line)
                blocks.append(
                    Block(
                        block_id=f"block-{line_id}",
                        block_type="text",
                        page_index=0,
                        bbox=line_box.bbox,
                        order=len(blocks),
                        text=text,
                        confidence=confidence,
                        line_ids=[line_id],
                        metadata={
                            "image_line_detector": "morph_line_detector_v1",
                            **structural_profile,
                        },
                    )
                )
            else:
                removed_count += 1
                for reason in reasons:
                    removal_counts[reason] = removal_counts.get(reason, 0) + 1
            crop_rows.append(
                {
                    "detected_line_id": line_box.line_id,
                    "line_id": text_lines[-1].line_id if kept and text_lines else None,
                    "kept": kept,
                    "removal_reasons": reasons,
                    "crop_path": str(crop_path),
                    "crop_sha256": sha256_file(crop_path),
                    "crop_xyxy": list(crop_xyxy),
                    "detected_bbox": line_box.bbox.to_list(),
                    "detected_area": line_box.area,
                    "detector_source_pass": line_box.source_pass,
                    "reading_order": reading_order_selection.selected,
                    "requested_reading_order": reading_order_selection.requested,
                    "prediction_text": text,
                    "confidence": confidence,
                    **script_profile,
                    **structural_profile,
                }
            )
    run_metadata = {
        "mode": "image_line",
        "detector": "morph_line_detector_v1",
        "detector_parameters": detection_config.to_dict(),
        "filter": filter_config.to_dict(),
        "filter_enabled": filter_config.enabled,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "model_config": str(model_config) if model_config is not None else None,
        "reading_order": reading_order_selection.selected,
        "requested_reading_order": reading_order_selection.requested,
        "reading_order_selection": reading_order_selection.to_dict(),
        "detected_line_count": len(detected),
        "kept_line_count": kept_count,
        "removed_line_count": removed_count,
        "empty_line_count": len(empty_line_ids),
        "empty_line_ids": empty_line_ids,
        "removal_counts": removal_counts,
        "detector_source_pass_counts": _detector_source_pass_counts(crop_rows),
        "kept_detector_source_pass_counts": _detector_source_pass_counts(crop_rows, kept=True),
        "removed_detector_source_pass_counts": _detector_source_pass_counts(crop_rows, kept=False),
        "structural_role_counts": _structural_role_counts(crop_rows),
        "kept_structural_role_counts": _structural_role_counts(crop_rows, kept=True),
        "removed_structural_role_counts": _structural_role_counts(crop_rows, kept=False),
        "crop_manifest": crop_rows,
        "duration_seconds": round(time.time() - started, 3),
    }
    blocks = _sort_blocks_for_reading_order(blocks, page_height=image_height)
    page = Page(
        page_index=0,
        width=image_width,
        height=image_height,
        text_lines=text_lines,
        blocks=blocks,
        metadata={
            "ocr_output_kind": "image_line_detector",
            "image_line_ocr": {key: value for key, value in run_metadata.items() if key != "crop_manifest"},
        },
    )
    document = Document(
        source_path=str(source),
        pages=[page],
        metadata={
            "engine": "image_line_detector",
            "ocr_output_kind": "image_line_detector",
            "image_line_ocr": {key: value for key, value in run_metadata.items() if key != "crop_manifest"},
        },
    )
    return document, run_metadata


def _block_structural_role(block: Block) -> str | None:
    metadata = block.metadata if isinstance(block.metadata, dict) else {}
    role = metadata.get("structural_role")
    if not isinstance(role, str):
        return None
    normalized = role.strip()
    return normalized or None


def _is_bottom_digit_prefixed_block(block: Block, *, page_height: float | int | None) -> bool:
    if page_height is None or page_height <= 0:
        return False
    if block.bbox.y < float(page_height) * 0.80:
        return False
    return bool(FRONT_MATTER_DIGIT_PREFIX_RE.match(normalize_ocr_text(block.text or "")))


def _block_sort_key(block: Block, *, page_height: float | int | None) -> tuple[int, float, float, int]:
    role = _block_structural_role(block)
    if role in FRONT_MATTER_STRUCTURAL_ROLES:
        front_matter_rank = 0
    elif _is_bottom_digit_prefixed_block(block, page_height=page_height):
        front_matter_rank = 1
    else:
        front_matter_rank = 2
    return (front_matter_rank, block.bbox.y, block.bbox.x, block.order)


def _sort_blocks_for_reading_order(blocks: list[Block], *, page_height: float | int | None) -> list[Block]:
    ordered = sorted(blocks, key=lambda block: _block_sort_key(block, page_height=page_height))
    for order, block in enumerate(ordered):
        block.order = order
    return ordered


def _detector_source_pass_counts(crop_rows: list[dict[str, Any]], *, kept: bool | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in crop_rows:
        if kept is not None and row.get("kept") is not kept:
            continue
        source_pass = row.get("detector_source_pass")
        key = str(source_pass) if source_pass else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _structural_role_counts(crop_rows: list[dict[str, Any]], *, kept: bool | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in crop_rows:
        if kept is not None and row.get("kept") is not kept:
            continue
        role = row.get("structural_role")
        key = str(role) if role else "text"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _line_structural_profile(
    text: str,
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if _looks_like_page_marker(text, bbox, image_width=image_width, image_height=image_height):
        return {
            "structural_role": "page_marker",
            "structural_role_reason": "bottom_margin_digit_only_small_box",
        }
    return {
        "structural_role": "text",
        "structural_role_reason": None,
    }


def _looks_like_page_marker(
    text: str,
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 3:
        return False
    if not all((char in _DEVANAGARI_DIGITS or char in _ASCII_DIGITS) for char in stripped):
        return False
    if image_width <= 0 or image_height <= 0 or bbox.w <= 0 or bbox.h <= 0:
        return False
    center_y_ratio = (bbox.y + bbox.h / 2.0) / image_height
    width_ratio = bbox.w / image_width
    height_ratio = bbox.h / image_height
    aspect_ratio = bbox.w / bbox.h
    return (
        center_y_ratio >= 0.82
        and width_ratio <= 0.05
        and height_ratio <= 0.04
        and 0.15 <= aspect_ratio <= 1.25
    )


def detect_line_boxes_with_audit(
    image_path: str | Path,
    config: ImageLineDetectionConfig,
) -> tuple[list[DetectedLineBox], ReadingOrderSelection]:
    cv2, np = _cv2_and_numpy()
    image_module = _pillow_image_module()
    with image_module.open(image_path) as image:
        gray = np.array(image.convert("L"))
    height, width = gray.shape
    requested_order = config.reading_order.strip().lower()
    gate_split_tall = requested_order == "auto_layout_split_tall_last" and config.split_tall_components
    gate_same_row_merge = config.merge_same_row_auto_fragmented_top_to_bottom and not config.merge_same_row_components
    boxes = _detect_line_boxes_for_config(
        gray,
        config,
        cv2=cv2,
        np=np,
        split_tall_override=False if gate_split_tall else None,
        merge_same_row_override=False if gate_same_row_merge else None,
    )
    selection = _resolve_reading_order_selection(boxes, image_width=width, reading_order=config.reading_order)
    if gate_split_tall and selection.selected == "wide_column_right_to_left_split_tall_last":
        boxes = _detect_line_boxes_for_config(
            gray,
            config,
            cv2=cv2,
            np=np,
            merge_same_row_override=False if gate_same_row_merge else None,
        )
        selection = _resolve_reading_order_selection(boxes, image_width=width, reading_order=config.reading_order)
    if gate_same_row_merge:
        merged_boxes = _merge_same_row_line_boxes(
            boxes,
            y_tolerance=config.merge_same_row_y_tolerance,
            max_gap=config.merge_same_row_max_gap,
            max_center_delta=config.merge_same_row_max_center_delta,
            max_width=config.merge_same_row_max_width,
        )
        decision = _same_row_auto_merge_decision(
            boxes,
            merged_boxes,
            selection=selection,
            min_reduction_ratio=config.merge_same_row_auto_min_reduction_ratio,
            min_reduction_count=config.merge_same_row_auto_min_reduction_count,
        )
        selection = _selection_with_same_row_merge_decision(selection, decision)
        if decision["applied"]:
            boxes = merged_boxes
    sorted_boxes = _sort_detected_boxes(boxes, image_width=width, reading_order=selection.selected)
    return [
        DetectedLineBox(
            line_id=f"img-line-{index:04d}",
            bbox=item.bbox,
            area=item.area,
            centroid=item.centroid,
            source_pass=item.source_pass,
        )
        for index, item in enumerate(sorted_boxes, start=1)
    ], selection


def _detect_line_boxes_for_config(
    gray: Any,
    config: ImageLineDetectionConfig,
    *,
    cv2: Any,
    np: Any,
    split_tall_override: bool | None = None,
    merge_same_row_override: bool | None = None,
) -> list[DetectedLineBox]:
    primary_config = _image_line_detection_pass_config(config)
    if split_tall_override is not None:
        primary_config["split_tall_components"] = split_tall_override
    boxes = _detect_line_boxes_single_pass(gray, primary_config, source_pass="primary", cv2=cv2, np=np)
    for index, overrides in enumerate(config.rescue_detector_passes, start=1):
        pass_config = dict(primary_config)
        pass_config.update(_normalize_rescue_detector_pass(overrides))
        if split_tall_override is not None:
            pass_config["split_tall_components"] = split_tall_override
        boxes = _merge_detected_line_boxes(
            boxes,
            _detect_line_boxes_single_pass(gray, pass_config, source_pass=f"rescue-{index}", cv2=cv2, np=np),
            merge_iou_threshold=config.merge_iou_threshold,
        )
    merge_same_row = config.merge_same_row_components if merge_same_row_override is None else merge_same_row_override
    if merge_same_row:
        boxes = _merge_same_row_line_boxes(
            boxes,
            y_tolerance=config.merge_same_row_y_tolerance,
            max_gap=config.merge_same_row_max_gap,
            max_center_delta=config.merge_same_row_max_center_delta,
            max_width=config.merge_same_row_max_width,
        )
    if config.split_detected_row_components:
        boxes = _split_detected_row_line_boxes(gray, boxes, primary_config, cv2=cv2, np=np)
    if config.split_detected_tall_components:
        boxes = _split_detected_tall_line_boxes(gray, boxes, primary_config, cv2=cv2, np=np)
    return boxes


def _image_line_detection_pass_config(config: ImageLineDetectionConfig) -> dict[str, object]:
    if config.merge_iou_threshold <= 0 or config.merge_iou_threshold > 1:
        raise ParseError(f"image-line merge_iou_threshold must be in (0, 1], got {config.merge_iou_threshold}")
    return {
        "threshold": config.threshold,
        "bbox_source": config.bbox_source,
        "horizontal_kernel": config.horizontal_kernel,
        "vertical_kernel": config.vertical_kernel,
        "dilation_iterations": config.dilation_iterations,
        "min_width": config.min_width,
        "min_height": config.min_height,
        "min_area": config.min_area,
        "max_height": config.max_height,
        "min_aspect_ratio": config.min_aspect_ratio,
        "max_aspect_ratio": config.max_aspect_ratio,
        "detector_padding": config.detector_padding,
        "split_tall_components": config.split_tall_components,
        "split_tall_row_min_ink": config.split_tall_row_min_ink,
        "split_tall_max_row_gap": config.split_tall_max_row_gap,
        "split_wide_components": config.split_wide_components,
        "split_wide_col_min_ink": config.split_wide_col_min_ink,
        "split_wide_max_col_gap": config.split_wide_max_col_gap,
        "split_wide_min_width": config.split_wide_min_width,
        "split_detected_row_components": config.split_detected_row_components,
        "split_detected_row_col_min_ink": config.split_detected_row_col_min_ink,
        "split_detected_row_max_col_gap": config.split_detected_row_max_col_gap,
        "split_detected_row_min_width": config.split_detected_row_min_width,
        "split_detected_row_min_segment_width": config.split_detected_row_min_segment_width,
        "split_detected_tall_components": config.split_detected_tall_components,
        "split_detected_tall_row_min_ink": config.split_detected_tall_row_min_ink,
        "split_detected_tall_max_row_gap": config.split_detected_tall_max_row_gap,
        "split_detected_tall_min_height": config.split_detected_tall_min_height,
        "split_detected_tall_min_segment_height": config.split_detected_tall_min_segment_height,
    }


def _normalize_rescue_detector_pass(overrides: dict[str, Any]) -> dict[str, object]:
    allowed = {
        "threshold",
        "bbox_source",
        "horizontal_kernel",
        "vertical_kernel",
        "dilation_iterations",
        "min_width",
        "min_height",
        "min_area",
        "max_height",
        "min_aspect_ratio",
        "max_aspect_ratio",
        "detector_padding",
        "split_tall_components",
        "split_tall_row_min_ink",
        "split_tall_max_row_gap",
        "split_wide_components",
        "split_wide_col_min_ink",
        "split_wide_max_col_gap",
        "split_wide_min_width",
        "split_detected_row_components",
        "split_detected_row_col_min_ink",
        "split_detected_row_max_col_gap",
        "split_detected_row_min_width",
        "split_detected_row_min_segment_width",
        "split_detected_tall_components",
        "split_detected_tall_row_min_ink",
        "split_detected_tall_max_row_gap",
        "split_detected_tall_min_height",
        "split_detected_tall_min_segment_height",
        "region_min_x_ratio",
        "region_max_x_ratio",
        "region_min_y_ratio",
        "region_max_y_ratio",
    }
    normalized: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in allowed:
            raise ParseError(f"unsupported image-line rescue detector key {key!r}; allowed keys are {sorted(allowed)}")
        if key == "bbox_source":
            if value not in {"ink", "dilated"}:
                raise ParseError("image-line rescue detector bbox_source must be 'ink' or 'dilated'")
            normalized[key] = value
        elif key == "threshold":
            normalized[key] = value
        elif key in {
            "split_tall_components",
            "split_wide_components",
            "split_detected_row_components",
            "split_detected_tall_components",
        }:
            if isinstance(value, bool):
                normalized[key] = value
            elif str(value).strip().lower() in {"1", "true", "yes", "on"}:
                normalized[key] = True
            elif str(value).strip().lower() in {"0", "false", "no", "off"}:
                normalized[key] = False
            else:
                raise ParseError(f"image-line rescue detector key {key!r} must be boolean: {value!r}")
        elif key in {
            "min_aspect_ratio",
            "max_aspect_ratio",
            "region_min_x_ratio",
            "region_max_x_ratio",
            "region_min_y_ratio",
            "region_max_y_ratio",
        }:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ParseError(f"image-line rescue detector key {key!r} must be a number: {value!r}") from exc
            if key.startswith("region_") and not 0 <= numeric_value <= 1:
                raise ParseError(f"image-line rescue detector key {key!r} must be between 0 and 1: {value!r}")
            if key in {"min_aspect_ratio", "max_aspect_ratio"} and numeric_value < 0:
                raise ParseError(f"image-line rescue detector key {key!r} must be non-negative: {value!r}")
            normalized[key] = numeric_value
        else:
            try:
                numeric_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ParseError(f"image-line rescue detector key {key!r} must be an integer: {value!r}") from exc
            if key in {
                "split_tall_max_row_gap",
                "split_wide_max_col_gap",
                "split_detected_row_max_col_gap",
                "split_detected_tall_max_row_gap",
            }:
                if numeric_value < 0:
                    raise ParseError(f"image-line rescue detector key {key!r} must be non-negative: {value!r}")
            elif numeric_value <= 0:
                raise ParseError(f"image-line rescue detector key {key!r} must be positive: {value!r}")
            normalized[key] = numeric_value
    return normalized


def _detect_line_boxes_single_pass(
    gray: Any,
    pass_config: dict[str, object],
    *,
    source_pass: str,
    cv2: Any,
    np: Any,
) -> list[DetectedLineBox]:
    binary = _threshold_image(gray, str(pass_config["threshold"]), cv2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, int(pass_config["horizontal_kernel"])), max(1, int(pass_config["vertical_kernel"]))),
    )
    dilated = cv2.dilate(binary, kernel, iterations=max(1, int(pass_config["dilation_iterations"])))
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, 8)
    height, width = gray.shape
    boxes: list[DetectedLineBox] = []
    bbox_source = str(pass_config["bbox_source"]).strip().lower()
    min_aspect_ratio = float(pass_config.get("min_aspect_ratio") or 0.0)
    max_aspect_ratio = float(pass_config.get("max_aspect_ratio") or 0.0)
    if bbox_source not in {"ink", "dilated"}:
        raise ParseError(f"bbox_source must be 'ink' or 'dilated', got {pass_config['bbox_source']!r}")
    for component_index in range(1, component_count):
        x, y, w, h, area = (int(value) for value in stats[component_index])
        if (
            bool(pass_config.get("split_tall_components"))
            and w >= int(pass_config["min_width"])
            and h > int(pass_config["max_height"])
            and area >= int(pass_config["min_area"])
        ):
            boxes.extend(
                box
                for box in _split_tall_component_boxes(
                    binary,
                    labels,
                    component_index,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    image_width=width,
                    image_height=height,
                    pass_config=pass_config,
                    source_pass=f"{source_pass}-split-tall",
                    np=np,
                )
                if _line_box_passes_region_filter(box.bbox, image_width=width, image_height=height, pass_config=pass_config)
            )
            continue
        if (
            bool(pass_config.get("split_wide_components"))
            and w >= int(pass_config.get("split_wide_min_width", 600))
            and h >= int(pass_config["min_height"])
            and h <= int(pass_config["max_height"])
            and area >= int(pass_config["min_area"])
        ):
            split_boxes = _split_wide_component_boxes(
                binary,
                labels,
                component_index,
                x=x,
                y=y,
                w=w,
                h=h,
                image_width=width,
                image_height=height,
                pass_config=pass_config,
                source_pass=f"{source_pass}-split-wide",
                np=np,
            )
            if len(split_boxes) > 1:
                boxes.extend(
                    box
                    for box in split_boxes
                    if _line_box_passes_region_filter(box.bbox, image_width=width, image_height=height, pass_config=pass_config)
                )
                continue
        if (
            w < int(pass_config["min_width"])
            or h < int(pass_config["min_height"])
            or area < int(pass_config["min_area"])
            or h > int(pass_config["max_height"])
        ):
            continue
        if bbox_source == "ink":
            region = (labels[y : y + h, x : x + w] == component_index) & (binary[y : y + h, x : x + w] > 0)
            ys, xs = np.where(region)
            if len(xs) == 0 or len(ys) == 0:
                continue
            ink_left = x + int(xs.min())
            ink_top = y + int(ys.min())
            ink_right = x + int(xs.max()) + 1
            ink_bottom = y + int(ys.max()) + 1
            x, y, w, h = ink_left, ink_top, ink_right - ink_left, ink_bottom - ink_top
            if w < int(pass_config["min_width"]) or h < int(pass_config["min_height"]) or h > int(pass_config["max_height"]):
                continue
        aspect_ratio = w / h if h > 0 else 0.0
        if min_aspect_ratio and aspect_ratio < min_aspect_ratio:
            continue
        if max_aspect_ratio and aspect_ratio > max_aspect_ratio:
            continue
        padding = int(pass_config["detector_padding"])
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + w + padding)
        bottom = min(height, y + h + padding)
        if right <= left or bottom <= top:
            continue
        bbox = BBox(left, top, right - left, bottom - top)
        if not _line_box_passes_region_filter(bbox, image_width=width, image_height=height, pass_config=pass_config):
            continue
        boxes.append(
            DetectedLineBox(
                line_id=f"img-line-{len(boxes) + 1:04d}",
                bbox=bbox,
                area=area,
                centroid=(float(centroids[component_index][0]), float(centroids[component_index][1])),
                source_pass=source_pass,
            )
        )
    return boxes


def _line_box_passes_region_filter(
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    pass_config: dict[str, object],
) -> bool:
    center_x_ratio = (bbox.x + bbox.w / 2.0) / image_width if image_width > 0 else 0.0
    center_y_ratio = (bbox.y + bbox.h / 2.0) / image_height if image_height > 0 else 0.0
    min_x = float(pass_config.get("region_min_x_ratio", 0.0))
    max_x = float(pass_config.get("region_max_x_ratio", 1.0))
    min_y = float(pass_config.get("region_min_y_ratio", 0.0))
    max_y = float(pass_config.get("region_max_y_ratio", 1.0))
    if min_x > max_x:
        raise ParseError("image-line region_min_x_ratio must be <= region_max_x_ratio")
    if min_y > max_y:
        raise ParseError("image-line region_min_y_ratio must be <= region_max_y_ratio")
    return min_x <= center_x_ratio <= max_x and min_y <= center_y_ratio <= max_y


def _split_detected_row_line_boxes(
    gray: Any,
    boxes: list[DetectedLineBox],
    pass_config: dict[str, object],
    *,
    cv2: Any,
    np: Any,
) -> list[DetectedLineBox]:
    col_min_ink = int(pass_config.get("split_detected_row_col_min_ink", 2))
    max_col_gap = int(pass_config.get("split_detected_row_max_col_gap", 24))
    min_row_width = int(pass_config.get("split_detected_row_min_width", 600))
    min_segment_width = int(pass_config.get("split_detected_row_min_segment_width", 40))
    if col_min_ink <= 0:
        raise ParseError("image-line split_detected_row_col_min_ink must be positive")
    if max_col_gap < 0:
        raise ParseError("image-line split_detected_row_max_col_gap must be non-negative")
    if min_row_width <= 0:
        raise ParseError("image-line split_detected_row_min_width must be positive")
    if min_segment_width <= 0:
        raise ParseError("image-line split_detected_row_min_segment_width must be positive")

    binary = _threshold_image(gray, str(pass_config["threshold"]), cv2)
    height, width = gray.shape
    padding = int(pass_config["detector_padding"])
    min_height = int(pass_config["min_height"])
    min_area = int(pass_config["min_area"])
    max_height = int(pass_config["max_height"])

    output: list[DetectedLineBox] = []
    for box in boxes:
        if box.bbox.w < min_row_width:
            output.append(box)
            continue
        x = max(0, int(math.floor(box.bbox.x)))
        y = max(0, int(math.floor(box.bbox.y)))
        right = min(width, int(math.ceil(box.bbox.right)))
        bottom = min(height, int(math.ceil(box.bbox.bottom)))
        if right <= x or bottom <= y:
            output.append(box)
            continue

        crop = binary[y:bottom, x:right]
        col_profile = (crop > 0).sum(axis=0)
        active_cols = col_profile >= col_min_ink
        col_groups: list[tuple[int, int]] = []
        start: int | None = None
        last_active: int | None = None
        for col_index, is_active in enumerate(active_cols):
            if is_active:
                if start is None:
                    start = col_index
                last_active = col_index
                continue
            if start is None or last_active is None:
                continue
            upcoming = active_cols[col_index + 1 : col_index + 1 + max_col_gap]
            if bool(upcoming.any()):
                continue
            col_groups.append((start, last_active))
            start = None
            last_active = None
        if start is not None and last_active is not None:
            col_groups.append((start, last_active))
        if len(col_groups) <= 1:
            output.append(box)
            continue

        split_boxes: list[DetectedLineBox] = []
        for left_col, right_col in col_groups:
            expanded_left = max(0, left_col - padding)
            expanded_right = min(right - x, right_col + 1 + padding)
            slice_mask = crop[:, expanded_left:expanded_right] > 0
            ys, xs = np.where(slice_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            ink_left = x + expanded_left + int(xs.min())
            ink_top = y + int(ys.min())
            ink_right = x + expanded_left + int(xs.max()) + 1
            ink_bottom = y + int(ys.max()) + 1
            line_width = ink_right - ink_left
            line_height = ink_bottom - ink_top
            if line_width < min_segment_width or line_height < min_height or line_height > max_height:
                continue
            if line_width * line_height < min_area:
                continue
            left = max(0, ink_left - padding)
            top = max(0, ink_top - padding)
            split_right = min(width, ink_right + padding)
            split_bottom = min(height, ink_bottom + padding)
            if split_right <= left or split_bottom <= top:
                continue
            split_bbox = BBox(left, top, split_right - left, split_bottom - top)
            if not _line_box_passes_region_filter(
                split_bbox,
                image_width=width,
                image_height=height,
                pass_config=pass_config,
            ):
                continue
            split_boxes.append(
                DetectedLineBox(
                    line_id=f"{box.line_id}-row-split-{len(split_boxes) + 1:02d}",
                    bbox=split_bbox,
                    area=int(line_width * line_height),
                    centroid=(float(ink_left + line_width / 2), float(ink_top + line_height / 2)),
                    source_pass=f"{box.source_pass}-row-split",
                )
            )
        if len(split_boxes) > 1:
            output.extend(split_boxes)
        else:
            output.append(box)
    return output


def _split_detected_tall_line_boxes(
    gray: Any,
    boxes: list[DetectedLineBox],
    pass_config: dict[str, object],
    *,
    cv2: Any,
    np: Any,
) -> list[DetectedLineBox]:
    row_min_ink = int(pass_config.get("split_detected_tall_row_min_ink", 20))
    max_row_gap = int(pass_config.get("split_detected_tall_max_row_gap", 4))
    min_box_height = int(pass_config.get("split_detected_tall_min_height", 90))
    min_segment_height = int(pass_config.get("split_detected_tall_min_segment_height", 24))
    if row_min_ink <= 0:
        raise ParseError("image-line split_detected_tall_row_min_ink must be positive")
    if max_row_gap < 0:
        raise ParseError("image-line split_detected_tall_max_row_gap must be non-negative")
    if min_box_height <= 0:
        raise ParseError("image-line split_detected_tall_min_height must be positive")
    if min_segment_height <= 0:
        raise ParseError("image-line split_detected_tall_min_segment_height must be positive")

    binary = _threshold_image(gray, str(pass_config["threshold"]), cv2)
    height, width = gray.shape
    padding = int(pass_config["detector_padding"])
    min_width = int(pass_config["min_width"])
    min_area = int(pass_config["min_area"])
    max_height = int(pass_config["max_height"])

    output: list[DetectedLineBox] = []
    for box in boxes:
        if box.bbox.h < min_box_height:
            output.append(box)
            continue
        x = max(0, int(math.floor(box.bbox.x)))
        y = max(0, int(math.floor(box.bbox.y)))
        right = min(width, int(math.ceil(box.bbox.right)))
        bottom = min(height, int(math.ceil(box.bbox.bottom)))
        if right <= x or bottom <= y:
            output.append(box)
            continue

        crop = binary[y:bottom, x:right]
        row_profile = (crop > 0).sum(axis=1)
        active_rows = row_profile >= row_min_ink
        row_groups: list[tuple[int, int]] = []
        start: int | None = None
        last_active: int | None = None
        for row_index, is_active in enumerate(active_rows):
            if is_active:
                if start is None:
                    start = row_index
                last_active = row_index
                continue
            if start is None or last_active is None:
                continue
            upcoming = active_rows[row_index + 1 : row_index + 1 + max_row_gap]
            if bool(upcoming.any()):
                continue
            row_groups.append((start, last_active))
            start = None
            last_active = None
        if start is not None and last_active is not None:
            row_groups.append((start, last_active))
        if len(row_groups) <= 1:
            output.append(box)
            continue

        split_boxes: list[DetectedLineBox] = []
        for top_row, bottom_row in row_groups:
            expanded_top = max(0, top_row - padding)
            expanded_bottom = min(bottom - y, bottom_row + 1 + padding)
            slice_mask = crop[expanded_top:expanded_bottom, :] > 0
            ys, xs = np.where(slice_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            ink_left = x + int(xs.min())
            ink_top = y + expanded_top + int(ys.min())
            ink_right = x + int(xs.max()) + 1
            ink_bottom = y + expanded_top + int(ys.max()) + 1
            line_width = ink_right - ink_left
            line_height = ink_bottom - ink_top
            if line_width < min_width or line_height < min_segment_height or line_height > max_height:
                continue
            if line_width * line_height < min_area:
                continue
            left = max(0, ink_left - padding)
            top = max(0, ink_top - padding)
            split_right = min(width, ink_right + padding)
            split_bottom = min(height, ink_bottom + padding)
            if split_right <= left or split_bottom <= top:
                continue
            split_bbox = BBox(left, top, split_right - left, split_bottom - top)
            if not _line_box_passes_region_filter(
                split_bbox,
                image_width=width,
                image_height=height,
                pass_config=pass_config,
            ):
                continue
            split_boxes.append(
                DetectedLineBox(
                    line_id=f"{box.line_id}-tall-split-{len(split_boxes) + 1:02d}",
                    bbox=split_bbox,
                    area=int(line_width * line_height),
                    centroid=(float(ink_left + line_width / 2), float(ink_top + line_height / 2)),
                    source_pass=f"{box.source_pass}-tall-split",
                )
            )
        if len(split_boxes) > 1:
            output.extend(split_boxes)
        else:
            output.append(box)
    return output


def _split_wide_component_boxes(
    binary: Any,
    labels: Any,
    component_index: int,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    pass_config: dict[str, object],
    source_pass: str,
    np: Any,
) -> list[DetectedLineBox]:
    col_min_ink = int(pass_config.get("split_wide_col_min_ink", 2))
    max_col_gap = int(pass_config.get("split_wide_max_col_gap", 24))
    if col_min_ink <= 0:
        raise ParseError(f"image-line split_wide_col_min_ink must be positive, got {col_min_ink}")
    if max_col_gap < 0:
        raise ParseError(f"image-line split_wide_max_col_gap must be non-negative, got {max_col_gap}")

    component_mask = labels[y : y + h, x : x + w] == component_index
    ink_mask = component_mask & (binary[y : y + h, x : x + w] > 0)
    col_profile = ink_mask.sum(axis=0)
    active_cols = col_profile >= col_min_ink
    col_groups: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for col_index, is_active in enumerate(active_cols):
        if is_active:
            if start is None:
                start = col_index
            last_active = col_index
            continue
        if start is None or last_active is None:
            continue
        upcoming = active_cols[col_index + 1 : col_index + 1 + max_col_gap]
        if bool(upcoming.any()):
            continue
        col_groups.append((start, last_active))
        start = None
        last_active = None
    if start is not None and last_active is not None:
        col_groups.append((start, last_active))
    if len(col_groups) <= 1:
        return []

    padding = int(pass_config["detector_padding"])
    min_width = int(pass_config["min_width"])
    min_height = int(pass_config["min_height"])
    min_area = int(pass_config["min_area"])
    max_height = int(pass_config["max_height"])
    boxes: list[DetectedLineBox] = []
    for left_col, right_col in col_groups:
        expanded_left = max(0, left_col - padding)
        expanded_right = min(w, right_col + 1 + padding)
        slice_mask = ink_mask[:, expanded_left:expanded_right]
        ys, xs = np.where(slice_mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        ink_left = x + expanded_left + int(xs.min())
        ink_top = y + int(ys.min())
        ink_right = x + expanded_left + int(xs.max()) + 1
        ink_bottom = y + int(ys.max()) + 1
        line_width = ink_right - ink_left
        line_height = ink_bottom - ink_top
        if line_width < min_width or line_height < min_height or line_height > max_height:
            continue
        if line_width * line_height < min_area:
            continue
        left = max(0, ink_left - padding)
        top = max(0, ink_top - padding)
        right = min(image_width, ink_right + padding)
        bottom = min(image_height, ink_bottom + padding)
        if right <= left or bottom <= top:
            continue
        boxes.append(
            DetectedLineBox(
                line_id=f"img-line-{len(boxes) + 1:04d}",
                bbox=BBox(left, top, right - left, bottom - top),
                area=int(line_width * line_height),
                centroid=(float(ink_left + line_width / 2), float(ink_top + line_height / 2)),
                source_pass=source_pass,
            )
        )
    return boxes


def _split_tall_component_boxes(
    binary: Any,
    labels: Any,
    component_index: int,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    pass_config: dict[str, object],
    source_pass: str,
    np: Any,
) -> list[DetectedLineBox]:
    row_min_ink = int(pass_config.get("split_tall_row_min_ink", 2))
    max_row_gap = int(pass_config.get("split_tall_max_row_gap", 4))
    if row_min_ink <= 0:
        raise ParseError(f"image-line split_tall_row_min_ink must be positive, got {row_min_ink}")
    if max_row_gap < 0:
        raise ParseError(f"image-line split_tall_max_row_gap must be non-negative, got {max_row_gap}")

    component_mask = labels[y : y + h, x : x + w] == component_index
    ink_mask = component_mask & (binary[y : y + h, x : x + w] > 0)
    row_profile = ink_mask.sum(axis=1)
    active_rows = row_profile >= row_min_ink
    row_groups: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for row_index, is_active in enumerate(active_rows):
        if is_active:
            if start is None:
                start = row_index
            last_active = row_index
            continue
        if start is None or last_active is None:
            continue
        upcoming = active_rows[row_index + 1 : row_index + 1 + max_row_gap]
        if bool(upcoming.any()):
            continue
        row_groups.append((start, last_active))
        start = None
        last_active = None
    if start is not None and last_active is not None:
        row_groups.append((start, last_active))

    padding = int(pass_config["detector_padding"])
    min_width = int(pass_config["min_width"])
    min_height = int(pass_config["min_height"])
    max_height = int(pass_config["max_height"])
    boxes: list[DetectedLineBox] = []
    for top_row, bottom_row in row_groups:
        expanded_top = max(0, top_row - padding)
        expanded_bottom = min(h, bottom_row + 1 + padding)
        slice_mask = ink_mask[expanded_top:expanded_bottom, :]
        ys, xs = np.where(slice_mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        ink_left = x + int(xs.min())
        ink_top = y + expanded_top + int(ys.min())
        ink_right = x + int(xs.max()) + 1
        ink_bottom = y + expanded_top + int(ys.max()) + 1
        line_width = ink_right - ink_left
        line_height = ink_bottom - ink_top
        if line_width < min_width or line_height < min_height or line_height > max_height:
            continue
        left = max(0, ink_left - padding)
        top = max(0, ink_top - padding)
        right = min(image_width, ink_right + padding)
        bottom = min(image_height, ink_bottom + padding)
        if right <= left or bottom <= top:
            continue
        boxes.append(
            DetectedLineBox(
                line_id=f"img-line-{len(boxes) + 1:04d}",
                bbox=BBox(left, top, right - left, bottom - top),
                area=int(line_width * line_height),
                centroid=(float(ink_left + line_width / 2), float(ink_top + line_height / 2)),
                source_pass=source_pass,
            )
        )
    return boxes


def _merge_detected_line_boxes(
    existing: list[DetectedLineBox],
    candidates: list[DetectedLineBox],
    *,
    merge_iou_threshold: float,
) -> list[DetectedLineBox]:
    if merge_iou_threshold <= 0 or merge_iou_threshold > 1:
        raise ParseError(f"image-line merge_iou_threshold must be in (0, 1], got {merge_iou_threshold}")
    merged = list(existing)
    for candidate in candidates:
        if any(_bbox_iou(candidate.bbox, item.bbox) >= merge_iou_threshold for item in merged):
            continue
        merged.append(candidate)
    return merged


def _merge_same_row_line_boxes(
    boxes: list[DetectedLineBox],
    *,
    y_tolerance: float,
    max_gap: float,
    max_center_delta: float,
    max_width: float,
) -> list[DetectedLineBox]:
    if y_tolerance < 0:
        raise ParseError(f"image-line merge_same_row_y_tolerance must be non-negative, got {y_tolerance}")
    if max_gap < 0:
        raise ParseError(f"image-line merge_same_row_max_gap must be non-negative, got {max_gap}")
    if max_center_delta <= 0:
        raise ParseError(f"image-line merge_same_row_max_center_delta must be positive, got {max_center_delta}")
    if max_width <= 0:
        raise ParseError(f"image-line merge_same_row_max_width must be positive, got {max_width}")
    if len(boxes) < 2:
        return list(boxes)

    ordered = sorted(boxes, key=lambda item: (_bbox_center_y(item.bbox), item.bbox.x, item.bbox.w))
    used: set[int] = set()
    merged: list[DetectedLineBox] = []
    for seed_index, seed in enumerate(ordered):
        if seed_index in used:
            continue
        group = [seed]
        used.add(seed_index)
        changed = True
        while changed:
            changed = False
            group_bbox = _union_bbox([item.bbox for item in group])
            for candidate_index, candidate in enumerate(ordered):
                if candidate_index in used:
                    continue
                candidate_bbox = candidate.bbox
                if not _same_row_candidate(group_bbox, candidate_bbox, y_tolerance=y_tolerance):
                    continue
                horizontal_gap = _horizontal_gap(group_bbox, candidate_bbox)
                union_bbox = _union_bbox([group_bbox, candidate_bbox])
                if horizontal_gap > max_gap:
                    continue
                if abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(group_bbox)) > max_center_delta:
                    continue
                if union_bbox.w > max_width:
                    continue
                group.append(candidate)
                used.add(candidate_index)
                changed = True
                break
        if len(group) == 1:
            merged.append(seed)
            continue
        bbox = _union_bbox([item.bbox for item in group])
        merged.append(
            DetectedLineBox(
                line_id=seed.line_id,
                bbox=bbox,
                area=sum(item.area for item in group),
                centroid=(_bbox_center_x(bbox), _bbox_center_y(bbox)),
                source_pass="row-merge",
            )
        )
    return merged


def _same_row_auto_merge_decision(
    before: list[DetectedLineBox],
    after: list[DetectedLineBox],
    *,
    selection: ReadingOrderSelection,
    min_reduction_ratio: float,
    min_reduction_count: int,
) -> dict[str, Any]:
    before_count = len(before)
    after_count = len(after)
    reduction_count = max(0, before_count - after_count)
    reduction_ratio = reduction_count / before_count if before_count else 0.0
    selected_top_to_bottom = selection.selected == "top_to_bottom"
    threshold_met = reduction_count >= min_reduction_count and reduction_ratio >= min_reduction_ratio
    applied = selected_top_to_bottom and threshold_met
    if applied:
        reason = "top-to-bottom layout with high same-row fragment reduction"
    elif not selected_top_to_bottom:
        reason = f"reading order {selection.selected!r} is not top_to_bottom"
    else:
        reason = "same-row fragment reduction below threshold"
    return {
        "enabled": True,
        "applied": applied,
        "reason": reason,
        "before_line_count": before_count,
        "after_line_count": after_count,
        "reduction_count": reduction_count,
        "reduction_ratio": reduction_ratio,
        "min_reduction_count": min_reduction_count,
        "min_reduction_ratio": min_reduction_ratio,
        "required_reading_order": "top_to_bottom",
        "selected_reading_order": selection.selected,
    }


def _selection_with_same_row_merge_decision(
    selection: ReadingOrderSelection,
    decision: dict[str, Any],
) -> ReadingOrderSelection:
    features = dict(selection.features)
    features["same_row_auto_merge"] = decision
    return ReadingOrderSelection(
        requested=selection.requested,
        selected=selection.selected,
        reason=selection.reason,
        features=features,
    )


def _same_row_candidate(left: BBox, right: BBox, *, y_tolerance: float) -> bool:
    if abs(_bbox_center_y(left) - _bbox_center_y(right)) <= y_tolerance:
        return True
    overlap = max(0.0, min(left.bottom, right.bottom) - max(left.y, right.y))
    shorter_height = min(left.h, right.h)
    return shorter_height > 0 and overlap / shorter_height >= 0.35


def _horizontal_gap(left: BBox, right: BBox) -> float:
    if left.right < right.x:
        return right.x - left.right
    if right.right < left.x:
        return left.x - right.right
    return 0.0


def _union_bbox(boxes: list[BBox]) -> BBox:
    if not boxes:
        raise RuntimeError("cannot union an empty bbox list")
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BBox(left, top, right - left, bottom - top)


def _bbox_center_x(bbox: BBox) -> float:
    return bbox.x + bbox.w / 2.0


def _bbox_center_y(bbox: BBox) -> float:
    return bbox.y + bbox.h / 2.0


def _bbox_iou(left: BBox, right: BBox) -> float:
    inter_w = max(0.0, min(left.right, right.right) - max(left.x, right.x))
    inter_h = max(0.0, min(left.bottom, right.bottom) - max(left.y, right.y))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union_area = left.w * left.h + right.w * right.h - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _cv2_and_numpy() -> tuple[Any, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise EngineUnavailableError("OpenCV is required for image-line OCR mode") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise EngineUnavailableError("NumPy is required for image-line OCR mode") from exc
    return cv2, np


def _pillow_image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise EngineUnavailableError("Pillow is required for image-line OCR mode") from exc
    return Image


def _threshold_image(gray: Any, mode: str, cv2: Any) -> Any:
    if mode == "otsu":
        _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    try:
        value = int(mode)
    except ValueError as exc:
        raise ParseError(f"threshold must be 'otsu' or an integer 0..255, got {mode!r}") from exc
    if value < 0 or value > 255:
        raise ParseError(f"threshold integer must be 0..255, got {value}")
    _threshold, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    return binary


def _resolve_reading_order_selection(
    boxes: list[DetectedLineBox],
    *,
    image_width: int,
    reading_order: str,
) -> ReadingOrderSelection:
    requested = reading_order.strip().lower()
    if requested not in READING_ORDER_MODES:
        raise ParseError(f"reading_order must be one of {', '.join(READING_ORDER_MODES)}, got {reading_order!r}")
    features = _reading_order_layout_features(boxes, image_width=image_width)
    auto_modes = {"auto_layout", "auto_layout_split_tall_last"}
    if requested not in auto_modes:
        return ReadingOrderSelection(
            requested=requested,
            selected=requested,
            reason="explicit reading-order mode",
            features=features,
        )
    if (
        features["line_count"] >= 4
        and features["image_width"] >= 1200
        and features["x_span_ratio"] >= 0.55
        and features["width_iqr_ratio"] <= 0.55
    ):
        selected = "wide_column_right_to_left_split_tall_last" if requested == "auto_layout_split_tall_last" else "wide_column_right_to_left"
        return ReadingOrderSelection(
            requested=requested,
            selected=selected,
            reason="wide page with column-like regular line widths",
            features=features,
        )
    return ReadingOrderSelection(
        requested=requested,
        selected="top_to_bottom",
        reason="layout features did not meet wide-column heuristic",
        features=features,
    )


def _reading_order_layout_features(boxes: list[DetectedLineBox], *, image_width: int) -> dict[str, Any]:
    if not boxes:
        return {
            "image_width": image_width,
            "line_count": 0,
            "x_min": None,
            "x_max": None,
            "x_span": 0.0,
            "x_span_ratio": 0.0,
            "median_line_width": 0.0,
            "width_q25": 0.0,
            "width_q75": 0.0,
            "width_iqr": 0.0,
            "width_iqr_ratio": 0.0,
            "x_cluster_count": 0,
        }
    _cv2, np = _cv2_and_numpy()
    centers = sorted(box.bbox.x + box.bbox.w / 2.0 for box in boxes)
    widths = [box.bbox.w for box in boxes]
    median_width = float(np.median(widths))
    width_q25 = float(np.percentile(widths, 25))
    width_q75 = float(np.percentile(widths, 75))
    width_iqr = width_q75 - width_q25
    x_min = min(centers)
    x_max = max(centers)
    cluster_gap = max(80.0, min(float(image_width) * 0.18, median_width * 1.5))
    cluster_count = 1
    previous = centers[0]
    for center in centers[1:]:
        if center - previous > cluster_gap:
            cluster_count += 1
        previous = center
    x_span = x_max - x_min
    return {
        "image_width": image_width,
        "line_count": len(boxes),
        "x_min": x_min,
        "x_max": x_max,
        "x_span": x_span,
        "x_span_ratio": x_span / image_width if image_width > 0 else 0.0,
        "median_line_width": median_width,
        "width_q25": width_q25,
        "width_q75": width_q75,
        "width_iqr": width_iqr,
        "width_iqr_ratio": width_iqr / median_width if median_width > 0 else 0.0,
        "x_cluster_gap": cluster_gap,
        "x_cluster_count": cluster_count,
    }


def _sort_detected_boxes(
    boxes: list[DetectedLineBox],
    *,
    image_width: int,
    reading_order: str,
) -> list[DetectedLineBox]:
    normalized = reading_order.strip().lower()
    if normalized == "top_to_bottom":
        return sorted(boxes, key=lambda item: (item.bbox.y, item.bbox.x, item.bbox.h, item.bbox.w))
    if normalized in {"wide_column_right_to_left", "wide_column_right_to_left_split_tall_last"}:
        band_width = max(600.0, float(image_width) * 0.45)
        return sorted(
            boxes,
            key=lambda item: (
                1 if normalized.endswith("_split_tall_last") and "split-tall" in item.source_pass else 0,
                -int((item.bbox.x + item.bbox.w / 2.0) // band_width),
                item.bbox.y,
                -item.bbox.x,
                item.bbox.h,
                item.bbox.w,
            ),
        )
    if normalized != "column_major":
        raise ParseError(
            "reading_order must be 'top_to_bottom', 'column_major', "
            "'wide_column_right_to_left', 'wide_column_right_to_left_split_tall_last', "
            "'auto_layout', or 'auto_layout_split_tall_last', "
            f"got {reading_order!r}"
        )
    if not boxes:
        return []
    _cv2, np = _cv2_and_numpy()
    median_width = float(np.median([box.bbox.w for box in boxes]))
    center_threshold = max(24.0, min(float(image_width) * 0.18, median_width * 1.2))
    columns: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda item: (item.bbox.x + item.bbox.w / 2.0, item.bbox.y)):
        center = box.bbox.x + box.bbox.w / 2.0
        best_index: int | None = None
        best_distance = float("inf")
        for index, column in enumerate(columns):
            column_center = float(column["center"])
            distance = abs(center - column_center)
            overlap = max(0.0, min(box.bbox.right, float(column["right"])) - max(box.bbox.x, float(column["left"])))
            if overlap > 0 or distance <= center_threshold:
                if distance < best_distance:
                    best_index = index
                    best_distance = distance
        if best_index is None:
            columns.append({"left": box.bbox.x, "right": box.bbox.right, "center": center, "boxes": [box]})
            continue
        column = columns[best_index]
        column_boxes = column["boxes"]
        if not isinstance(column_boxes, list):
            raise RuntimeError("internal column grouping state is invalid")
        column_boxes.append(box)
        column["left"] = min(float(column["left"]), box.bbox.x)
        column["right"] = max(float(column["right"]), box.bbox.right)
        column["center"] = sum(item.bbox.x + item.bbox.w / 2.0 for item in column_boxes) / len(column_boxes)
    ordered: list[DetectedLineBox] = []
    for column in sorted(columns, key=lambda item: (float(item["center"]), float(item["left"]))):
        column_boxes = column["boxes"]
        if not isinstance(column_boxes, list):
            raise RuntimeError("internal column grouping state is invalid")
        ordered.extend(sorted(column_boxes, key=lambda item: (item.bbox.y, item.bbox.x, item.bbox.h, item.bbox.w)))
    return ordered


def _crop_box(bbox: BBox, *, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    left = max(0, math.floor(bbox.x) - padding)
    top = max(0, math.floor(bbox.y) - padding)
    right = min(width, math.ceil(bbox.right) + padding)
    bottom = min(height, math.ceil(bbox.bottom) + padding)
    if right <= left or bottom <= top:
        raise ParseError(f"clipped crop is empty for bbox {bbox.to_list()} in {width}x{height}")
    return left, top, right, bottom


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "line"


def _recognize_crop(engine: object, crop_path: Path) -> tuple[str, float | None, list[dict[str, Any]]]:
    try:
        output = engine.recognize(crop_path)
    except EngineUnavailableError:
        raise
    except Exception as exc:
        raise ParseError(f"image-line crop recognition failed for {crop_path}: {type(exc).__name__}: {exc}") from exc
    lines: list[TextLine] = []
    for page in output.pages:
        lines.extend(page.text_lines)
    if not lines:
        return "", None, []
    ordered = sorted(lines, key=lambda line: (line.bbox.y, line.bbox.x, line.line_id or ""))
    text = normalize_ocr_text(" ".join(line.text for line in ordered if line.text))
    confidences = [line.confidence for line in ordered if line.confidence is not None]
    confidence = sum(confidences) / len(confidences) if confidences else None
    return text, confidence, [line.to_dict() for line in ordered]


def _line_removal_reasons(
    text: str,
    confidence: float | None,
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    filter_config: ImageLineFilterConfig,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    profile = _script_profile(text, script_ratio_threshold=filter_config.script_ratio_threshold)
    width_ratio = bbox.w / image_width if image_width else None
    height_ratio = bbox.h / image_height if image_height else None
    if filter_config.drop_empty and not text.strip():
        reasons.append("empty_text")
    if filter_config.min_confidence is not None:
        if confidence is None:
            reasons.append("missing_confidence")
        elif confidence < filter_config.min_confidence:
            reasons.append("low_confidence")
    if filter_config.require_script != "any":
        script_class = str(profile["script_class"])
        if filter_config.require_script == "sirijonga" and script_class not in {"limbu_sirijonga", "mixed_limbu_devanagari"}:
            reasons.append("script_not_sirijonga")
        elif filter_config.require_script == "devanagari" and script_class not in {"devanagari_limbu", "mixed_limbu_devanagari"}:
            reasons.append("script_not_devanagari")
        elif filter_config.require_script == "limbu_or_devanagari" and script_class == "other":
            reasons.append("script_not_limbu_or_devanagari")
        elif filter_config.require_script not in {"sirijonga", "devanagari", "limbu_or_devanagari"}:
            ratio = _script_ratio(text, filter_config.require_script)
            profile["required_script"] = filter_config.require_script
            profile["required_script_ratio"] = ratio
            if ratio < filter_config.script_ratio_threshold:
                reasons.append(f"script_not_{_script_reason_key(filter_config.require_script)}")
    if filter_config.min_width_ratio is not None:
        if width_ratio is None:
            reasons.append("missing_image_width")
        elif width_ratio < filter_config.min_width_ratio:
            reasons.append("width_ratio_below_min")
    if filter_config.max_width_ratio is not None:
        if width_ratio is None:
            reasons.append("missing_image_width")
        elif width_ratio > filter_config.max_width_ratio:
            reasons.append("width_ratio_above_max")
    if filter_config.min_height_ratio is not None:
        if height_ratio is None:
            reasons.append("missing_image_height")
        elif height_ratio < filter_config.min_height_ratio:
            reasons.append("height_ratio_below_min")
    if filter_config.max_height_ratio is not None:
        if height_ratio is None:
            reasons.append("missing_image_height")
        elif height_ratio > filter_config.max_height_ratio:
            reasons.append("height_ratio_above_max")
    return reasons, {
        **profile,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
    }


def _script_profile(text: str, *, script_ratio_threshold: float) -> dict[str, Any]:
    non_space = [char for char in text if not char.isspace()]
    total = len(non_space)
    limbu_count = _script_codepoint_count(non_space, script="limbu")
    devanagari_count = _script_codepoint_count(non_space, script="devanagari")
    latin_count = _script_codepoint_count(non_space, script="latin")
    limbu_ratio = limbu_count / total if total else 0.0
    devanagari_ratio = devanagari_count / total if total else 0.0
    latin_ratio = latin_count / total if total else 0.0
    has_limbu = limbu_ratio >= script_ratio_threshold
    has_devanagari = devanagari_ratio >= script_ratio_threshold
    if has_limbu and has_devanagari:
        script_class = "mixed_limbu_devanagari"
    elif has_limbu:
        script_class = "limbu_sirijonga"
    elif has_devanagari:
        script_class = "devanagari_limbu"
    else:
        script_class = "other"
    return {
        "script_class": script_class,
        "non_space_count": total,
        "limbu_codepoint_count": limbu_count,
        "devanagari_codepoint_count": devanagari_count,
        "latin_codepoint_count": latin_count,
        "limbu_ratio": limbu_ratio,
        "devanagari_ratio": devanagari_ratio,
        "latin_ratio": latin_ratio,
    }


def _script_reason_key(script: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", script.strip().lower()).strip("_") or "script"
