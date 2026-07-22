"""End-to-end parsing pipeline and file exports."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import platform
import re
import shutil
import socket
from pathlib import Path
from datetime import UTC, datetime

from .capture import audit_limbu_capture
from .engines import create_engine
from .errors import EngineUnavailableError, ParseError
from .image_lines import ImageLineDetectionConfig, ImageLineFilterConfig, build_image_line_document
from .markdown import render_document_markdown
from .normalization import normalize_ocr_text
from .quality import evaluate_document_quality
from .schemas import Document, TextLine
from .structure import build_document
from .tables import write_table_files

_CROPPABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_LIMBU_CODEPOINT_START = 0x1900
_LIMBU_CODEPOINT_END = 0x194F
_DEVANAGARI_CODEPOINT_START = 0x0900
_DEVANAGARI_CODEPOINT_END = 0x097F
_LIMBU_SCRIPT_CLASSES = ("limbu_sirijonga", "devanagari_limbu", "mixed_limbu_devanagari", "other")
_LIMBU_REQUIRED_OCR_ARTIFACTS = (
    "document.json",
    "document.md",
    "document.body.md",
    "quality.json",
    "limbu-pipeline-audit.json",
    "limbu-line-audit.jsonl",
    "limbu-post-correction-audit.json",
    "limbu-post-correction-lines.jsonl",
    "limbu-review-queue.tsv",
    "limbu-review-dashboard.html",
    "line-crops/manifest.jsonl",
    "line-crops/summary.json",
)
_LIMBU_REQUIRED_REVIEW_CORRECTION_ARTIFACTS = (
    "document.json",
    "document.md",
    "document.body.md",
    "limbu-correction-audit.json",
    "limbu-correction-pairs.jsonl",
)


def parse_limbu_document(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    engine_name: str = "auto",
    model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    sirijonga_low_confidence_threshold: float | None = None,
    devanagari_low_confidence_threshold: float | None = None,
    mixed_script_low_confidence_threshold: float | None = None,
    other_script_low_confidence_threshold: float | None = None,
    fallback_min_quality_score: float | None = None,
    script_ratio_threshold: float = 0.20,
    capture_prep_metadata: str | Path | None = None,
    post_correction_profile: str | Path | None = None,
    line_detection_mode: str = "engine",
    image_line_threshold: str = "otsu",
    image_line_bbox_source: str = "ink",
    image_line_reading_order: str = "auto_layout",
    image_line_horizontal_kernel: int = 23,
    image_line_vertical_kernel: int = 3,
    image_line_dilation_iterations: int = 1,
    image_line_min_width: int = 35,
    image_line_min_height: int = 10,
    image_line_min_area: int = 100,
    image_line_max_height: int = 140,
    image_line_min_aspect_ratio: float = 0.0,
    image_line_max_aspect_ratio: float = 0.0,
    image_line_detector_padding: int = 2,
    image_line_crop_padding: int = 12,
    image_line_rescue_detector_passes: tuple[dict[str, object], ...] = (),
    image_line_merge_iou_threshold: float = 0.80,
    image_line_split_tall_components: bool = False,
    image_line_split_tall_row_min_ink: int = 2,
    image_line_split_tall_max_row_gap: int = 4,
    image_line_split_wide_components: bool = False,
    image_line_split_wide_col_min_ink: int = 2,
    image_line_split_wide_max_col_gap: int = 24,
    image_line_split_wide_min_width: int = 600,
    image_line_split_detected_row_components: bool = False,
    image_line_split_detected_row_col_min_ink: int = 2,
    image_line_split_detected_row_max_col_gap: int = 24,
    image_line_split_detected_row_min_width: int = 600,
    image_line_split_detected_row_min_segment_width: int = 40,
    image_line_split_detected_tall_components: bool = False,
    image_line_split_detected_tall_row_min_ink: int = 20,
    image_line_split_detected_tall_max_row_gap: int = 4,
    image_line_split_detected_tall_min_height: int = 90,
    image_line_split_detected_tall_min_segment_height: int = 24,
    image_line_merge_same_row_components: bool = False,
    image_line_merge_same_row_y_tolerance: float = 8.0,
    image_line_merge_same_row_max_gap: float = 120.0,
    image_line_merge_same_row_max_center_delta: float = 300.0,
    image_line_merge_same_row_max_width: float = 420.0,
    image_line_merge_same_row_auto_fragmented_top_to_bottom: bool = False,
    image_line_merge_same_row_auto_min_reduction_ratio: float = 0.18,
    image_line_merge_same_row_auto_min_reduction_count: int = 8,
    image_line_allow_empty_lines: bool = False,
    image_line_filter_drop_empty: bool = False,
    image_line_filter_min_confidence: float | None = None,
    image_line_filter_require_script: str = "any",
    image_line_filter_min_width_ratio: float | None = None,
    image_line_filter_max_width_ratio: float | None = None,
    image_line_filter_min_height_ratio: float | None = None,
    image_line_filter_max_height_ratio: float | None = None,
    run_context: dict[str, object] | None = None,
) -> Document:
    """Parse a Limbu document and emit script-aware review/audit artifacts.

    This wrapper is intentionally conservative: it does not pretend to solve
    language correction yet. It makes mixed Devanagari/Sirijonga behavior visible
    line by line so weak lines can be reviewed instead of silently entering the
    digitized text.
    """

    out = Path(output_dir)
    capture_audit = _audit_capture_prep_for_limbu_ocr(Path(input_path), out, capture_prep_metadata)
    script_confidence_thresholds = _limbu_script_confidence_thresholds(
        low_confidence_threshold=low_confidence_threshold,
        sirijonga_low_confidence_threshold=sirijonga_low_confidence_threshold,
        devanagari_low_confidence_threshold=devanagari_low_confidence_threshold,
        mixed_script_low_confidence_threshold=mixed_script_low_confidence_threshold,
        other_script_low_confidence_threshold=other_script_low_confidence_threshold,
    )
    normalized_line_detection_mode = line_detection_mode.strip().lower().replace("-", "_")
    image_line_metadata: dict[str, object] | None = None
    if normalized_line_detection_mode == "engine":
        document = parse_document(
            input_path,
            out,
            engine_name=engine_name,
            model_config=model_config,
            fallback_engine=fallback_engine,
            fallback_model_config=fallback_model_config,
            low_confidence_threshold=low_confidence_threshold,
            fallback_min_quality_score=fallback_min_quality_score,
        )
    elif normalized_line_detection_mode == "image_line":
        if fallback_engine is not None or fallback_model_config is not None or fallback_min_quality_score is not None:
            raise ParseError("image-line Limbu OCR mode does not support fallback engine options")
        engine_kwargs = {"model_config": model_config} if engine_name in {"candidate", "ours"} else {}
        engine = create_engine(engine_name, **engine_kwargs)
        document, image_line_metadata = build_image_line_document(
            input_path,
            output_dir=out,
            engine=engine,
            model_config=model_config,
            detection_config=ImageLineDetectionConfig(
                threshold=image_line_threshold,
                bbox_source=image_line_bbox_source,
                reading_order=image_line_reading_order,
                horizontal_kernel=image_line_horizontal_kernel,
                vertical_kernel=image_line_vertical_kernel,
                dilation_iterations=image_line_dilation_iterations,
                min_width=image_line_min_width,
                min_height=image_line_min_height,
                min_area=image_line_min_area,
                max_height=image_line_max_height,
                min_aspect_ratio=image_line_min_aspect_ratio,
                max_aspect_ratio=image_line_max_aspect_ratio,
                detector_padding=image_line_detector_padding,
                crop_padding=image_line_crop_padding,
                rescue_detector_passes=image_line_rescue_detector_passes,
                merge_iou_threshold=image_line_merge_iou_threshold,
                split_tall_components=image_line_split_tall_components,
                split_tall_row_min_ink=image_line_split_tall_row_min_ink,
                split_tall_max_row_gap=image_line_split_tall_max_row_gap,
                split_wide_components=image_line_split_wide_components,
                split_wide_col_min_ink=image_line_split_wide_col_min_ink,
                split_wide_max_col_gap=image_line_split_wide_max_col_gap,
                split_wide_min_width=image_line_split_wide_min_width,
                split_detected_row_components=image_line_split_detected_row_components,
                split_detected_row_col_min_ink=image_line_split_detected_row_col_min_ink,
                split_detected_row_max_col_gap=image_line_split_detected_row_max_col_gap,
                split_detected_row_min_width=image_line_split_detected_row_min_width,
                split_detected_row_min_segment_width=image_line_split_detected_row_min_segment_width,
                split_detected_tall_components=image_line_split_detected_tall_components,
                split_detected_tall_row_min_ink=image_line_split_detected_tall_row_min_ink,
                split_detected_tall_max_row_gap=image_line_split_detected_tall_max_row_gap,
                split_detected_tall_min_height=image_line_split_detected_tall_min_height,
                split_detected_tall_min_segment_height=image_line_split_detected_tall_min_segment_height,
                merge_same_row_components=image_line_merge_same_row_components,
                merge_same_row_y_tolerance=image_line_merge_same_row_y_tolerance,
                merge_same_row_max_gap=image_line_merge_same_row_max_gap,
                merge_same_row_max_center_delta=image_line_merge_same_row_max_center_delta,
                merge_same_row_max_width=image_line_merge_same_row_max_width,
                merge_same_row_auto_fragmented_top_to_bottom=image_line_merge_same_row_auto_fragmented_top_to_bottom,
                merge_same_row_auto_min_reduction_ratio=image_line_merge_same_row_auto_min_reduction_ratio,
                merge_same_row_auto_min_reduction_count=image_line_merge_same_row_auto_min_reduction_count,
            ),
            filter_config=ImageLineFilterConfig(
                drop_empty=image_line_filter_drop_empty,
                min_confidence=image_line_filter_min_confidence,
                require_script=image_line_filter_require_script,
                script_ratio_threshold=script_ratio_threshold,
                min_width_ratio=image_line_filter_min_width_ratio,
                max_width_ratio=image_line_filter_max_width_ratio,
                min_height_ratio=image_line_filter_min_height_ratio,
                max_height_ratio=image_line_filter_max_height_ratio,
            ),
            allow_empty_lines=image_line_allow_empty_lines,
        )
        document.metadata["quality"] = evaluate_document_quality(document).to_dict()
    else:
        raise ParseError("line_detection_mode must be 'engine' or 'image_line'")
    annotate_limbu_document(
        document,
        low_confidence_threshold=low_confidence_threshold,
        script_confidence_thresholds=script_confidence_thresholds,
        script_ratio_threshold=script_ratio_threshold,
        capture_prep_metadata=capture_prep_metadata,
        capture_prep_audit=capture_audit,
    )
    post_correction_audit = _apply_limbu_post_correction_profile(
        document,
        post_correction_profile,
        script_ratio_threshold=script_ratio_threshold,
    )
    if post_correction_audit.get("changed_count"):
        _refresh_blocks_from_lines(document)
        document.metadata["quality"] = evaluate_document_quality(document).to_dict()
    audit = annotate_limbu_document(
        document,
        low_confidence_threshold=low_confidence_threshold,
        script_confidence_thresholds=script_confidence_thresholds,
        script_ratio_threshold=script_ratio_threshold,
        capture_prep_metadata=capture_prep_metadata,
        capture_prep_audit=capture_audit,
    )
    _attach_limbu_post_correction_metadata(document, post_correction_audit, audit)
    audit["post_correction"] = _post_correction_summary(post_correction_audit)
    document.metadata["limbu_pipeline"] = audit
    export_document(document, out, Path(input_path))
    _write_limbu_pipeline_artifacts(audit, out)
    _write_limbu_post_correction_artifacts(post_correction_audit, out)
    if image_line_metadata is not None:
        _write_limbu_image_line_artifacts(image_line_metadata, out)
    _write_limbu_line_crop_artifacts(document, out, Path(input_path))
    _write_limbu_review_dashboard(out)
    expected_artifacts = [
        "document.json",
        "document.md",
        "document.body.md",
        "quality.json",
        "limbu-pipeline-audit.json",
        "limbu-line-audit.jsonl",
        "limbu-post-correction-audit.json",
        "limbu-post-correction-lines.jsonl",
        "limbu-review-queue.tsv",
        "limbu-review-dashboard.html",
        "line-crops/manifest.jsonl",
        "line-crops/summary.json",
    ]
    if image_line_metadata is not None:
        expected_artifacts.extend(["image-line-ocr-run.json", "image-line-crops/manifest.jsonl"])
    if capture_audit is not None:
        expected_artifacts.extend(
            [
                "capture-prep-audit/limbu-capture-audit.json",
                "capture-prep-audit/limbu-capture-audit.md",
            ]
        )
    _write_limbu_output_manifest(
        out,
        pipeline_stage="ocr",
        document=document,
        source_path=Path(input_path),
        run_context=run_context,
        provenance=_limbu_run_provenance(
            model_config=model_config,
            fallback_model_config=fallback_model_config,
            capture_prep_metadata=capture_prep_metadata,
            post_correction_profile=post_correction_profile,
        ),
        expected_artifacts=expected_artifacts,
    )
    return document


def parse_document(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    engine_name: str = "auto",
    model_config: str | Path | None = None,
    fallback_engine: str | None = None,
    fallback_model_config: str | Path | None = None,
    low_confidence_threshold: float = 0.80,
    fallback_min_quality_score: float | None = None,
) -> Document:
    source = Path(input_path)
    out = Path(output_dir)
    if not source.exists():
        raise ParseError(f"Input does not exist: {source}")
    out.mkdir(parents=True, exist_ok=True)

    engine_kwargs = {"model_config": model_config} if engine_name in {"candidate", "ours"} else {}
    engine = create_engine(engine_name, **engine_kwargs)
    engine_output = engine.recognize(source)
    primary_document = build_document(str(source), engine_output)
    primary_quality = evaluate_document_quality(primary_document)
    avg_confidence = engine_output.average_confidence
    fallback_decision = _fallback_decision(
        primary_engine=engine_name,
        fallback_engine=fallback_engine,
        fallback_model_config=fallback_model_config,
        primary_average_confidence=avg_confidence,
        low_confidence_threshold=low_confidence_threshold,
        primary_quality_score=primary_quality.quality_score,
        fallback_min_quality_score=fallback_min_quality_score,
    )
    if fallback_decision is not None and fallback_decision["triggered"]:
        try:
            fallback_kwargs = {"model_config": fallback_model_config} if fallback_model_config is not None else {}
            fallback = create_engine(str(fallback_decision["fallback_engine"]), **fallback_kwargs)
            fallback_output = fallback.recognize(source)
            fallback_decision["outcome"] = "success"
            fallback_output.metadata["fallback"] = fallback_decision
            document = build_document(str(source), fallback_output)
        except EngineUnavailableError as exc:
            fallback_decision["outcome"] = "error"
            fallback_decision["error"] = str(exc)
            engine_output.metadata["fallback"] = fallback_decision
            engine_output.metadata["fallback_error"] = str(exc)
            primary_document.metadata["fallback"] = fallback_decision
            primary_document.metadata["fallback_error"] = str(exc)
            primary_document.metadata["quality"] = primary_quality.to_dict()
            export_document(primary_document, out, source)
            return primary_document
        except Exception as exc:
            fallback_decision["outcome"] = "error"
            fallback_decision["error"] = f"{type(exc).__name__}: {exc}"
            engine_output.metadata["fallback"] = fallback_decision
            engine_output.metadata["fallback_error"] = f"{type(exc).__name__}: {exc}"
            primary_document.metadata["fallback"] = fallback_decision
            primary_document.metadata["fallback_error"] = f"{type(exc).__name__}: {exc}"
            primary_document.metadata["quality"] = primary_quality.to_dict()
            export_document(primary_document, out, source)
            return primary_document
    else:
        document = primary_document
        if fallback_decision is not None:
            document.metadata["fallback"] = fallback_decision

    quality = evaluate_document_quality(document)
    document.metadata["quality"] = quality.to_dict()
    export_document(document, out, source)
    return document


def apply_limbu_review_corrections(
    document_path: str | Path,
    review_queue_tsv: str | Path,
    output_dir: str | Path,
    *,
    accepted_statuses: tuple[str, ...] = ("accepted", "approved", "corrected", "verified", "done"),
    run_context: dict[str, object] | None = None,
) -> Document:
    """Apply accepted Limbu review-queue corrections and emit auditable pairs."""

    doc_path = Path(document_path)
    review_path = Path(review_queue_tsv)
    out = Path(output_dir)
    if not doc_path.is_file():
        raise ParseError(f"document JSON does not exist: {doc_path}")
    if not review_path.is_file():
        raise ParseError(f"Limbu review queue TSV does not exist: {review_path}")
    try:
        payload = json.loads(doc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid document JSON {doc_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"document JSON must be an object: {doc_path}")
    document = Document.from_dict(payload)
    accepted = {status.strip().lower() for status in accepted_statuses if status.strip()}
    if not accepted:
        raise ParseError("accepted_statuses must contain at least one non-empty status")
    script_ratio_threshold = _limbu_document_script_ratio_threshold(document)

    line_lookup, positional_lookup = _document_line_indexes(document)
    review_rows = _read_limbu_review_rows(review_path)
    applied_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    seen_targets: set[str] = set()
    review_sha = _sha256_file(review_path)
    for row_number, row in enumerate(review_rows, start=2):
        status = str(row.get("review_status") or "").strip().lower()
        target_key = _review_target_key(row)
        if status not in accepted:
            skipped_rows.append(
                {
                    "row_number": row_number,
                    "line_id": row.get("line_id"),
                    "page_index": row.get("page_index"),
                    "line_index": row.get("line_index"),
                    "status": status or "blank",
                    "reason": "review_status_not_accepted",
                }
            )
            continue
        if target_key in seen_targets:
            raise ParseError(f"duplicate accepted review row for {target_key} in {review_path}")
        seen_targets.add(target_key)
        line = _resolve_review_line(row, line_lookup, positional_lookup, review_path=review_path, row_number=row_number)
        corrected_text = normalize_ocr_text(str(row.get("corrected_text") or ""))
        if not corrected_text.strip():
            raise ParseError(f"accepted review row {row_number} has empty corrected_text")
        original_text = line.text
        line.text = corrected_text
        post_correction = {
            "status": "human_review_applied",
            "method": "limbu_review_queue_tsv",
            "source_review_queue_path": str(review_path),
            "source_review_queue_sha256": review_sha,
            "review_status": status,
            "reviewer": row.get("reviewer") or "",
            "notes": row.get("notes") or "",
            "original_text": original_text,
            "corrected_text": corrected_text,
        }
        limbu_metadata = line.metadata.setdefault("limbu_pipeline", {})
        if isinstance(limbu_metadata, dict):
            limbu_metadata["original_needs_review"] = limbu_metadata.get("needs_review")
            limbu_metadata["needs_review"] = False
            limbu_metadata["review_resolved"] = True
            limbu_metadata["post_correction"] = post_correction
        script_class = str(row.get("script_class") or "").strip()
        if not script_class:
            script_class = str(_limbu_script_profile(corrected_text, script_ratio_threshold=script_ratio_threshold)["script_class"])
        _validate_limbu_post_correction_script_class(
            script_class,
            label=f"accepted review row {row_number} script_class",
        )
        applied_rows.append(
            {
                "row_number": row_number,
                "page_index": line.page_index,
                "line_id": line.line_id,
                "script_class": script_class,
                "review_status": status,
                "reviewer": row.get("reviewer") or "",
                "original_text": original_text,
                "corrected_text": corrected_text,
                "changed": original_text != corrected_text,
            }
        )

    _refresh_blocks_from_lines(document)
    correction_audit = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "correction_stage": "human_review_queue_application",
        "document_path": str(doc_path),
        "document_sha256": _sha256_file(doc_path),
        "review_queue_path": str(review_path),
        "review_queue_sha256": review_sha,
        "accepted_statuses": sorted(accepted),
        "script_ratio_threshold": script_ratio_threshold,
        "review_rows": len(review_rows),
        "applied_count": len(applied_rows),
        "changed_count": sum(1 for row in applied_rows if row.get("changed")),
        "skipped_count": len(skipped_rows),
        "applied_rows": applied_rows,
        "skipped_rows": skipped_rows,
        "claim_scope": "review-applied operational OCR text; not ground truth unless the review protocol is separately audited",
    }
    document.metadata["limbu_post_correction"] = correction_audit
    export_document(document, out, Path(document.source_path or doc_path))
    _write_limbu_correction_artifacts(correction_audit, out)
    _write_limbu_output_manifest(
        out,
        pipeline_stage="review_correction",
        document=document,
        source_path=Path(document.source_path or doc_path),
        run_context=run_context,
        provenance=_limbu_run_provenance(
            input_document=doc_path,
            review_queue=review_path,
        ),
        expected_artifacts=[
            "document.json",
            "document.md",
            "document.body.md",
            "limbu-correction-audit.json",
            "limbu-correction-pairs.jsonl",
        ],
    )
    return document


def audit_limbu_post_correction_profile(
    profile_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Audit a deterministic Limbu post-correction profile for shape and admission evidence."""

    source = Path(profile_path)
    issues: list[str] = []
    warnings: list[str] = []
    profile: dict[str, object] | None = None
    try:
        profile = _load_limbu_post_correction_profile(source)
    except ParseError as exc:
        issues.append(str(exc))
    report = _limbu_post_correction_profile_audit_report(source, profile, issues, warnings)
    out = Path(output_dir) if output_dir is not None else source.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "limbu-post-correction-profile-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "limbu-post-correction-profile-audit.md").write_text(
        _render_limbu_post_correction_profile_audit_markdown(report),
        encoding="utf-8",
    )
    return report


def derive_limbu_post_correction_profile(
    correction_pair_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    profile_id: str,
    min_support: int = 1,
    script_scope: tuple[str, ...] = ("limbu_sirijonga", "devanagari_limbu", "mixed_limbu_devanagari"),
) -> dict[str, object]:
    """Derive an experimental deterministic Limbu correction profile from reviewed pairs."""

    normalized_profile_id = str(profile_id or "").strip()
    if not normalized_profile_id:
        raise ParseError("profile_id is required")
    if min_support < 1:
        raise ParseError(f"min_support must be >= 1, got {min_support}")
    normalized_script_scope = tuple(str(item).strip() for item in script_scope if str(item).strip())
    _validate_limbu_post_correction_script_classes(list(normalized_script_scope), label="derivation script_scope")
    paths = [Path(path) for path in correction_pair_paths]
    if not paths:
        raise ParseError("at least one Limbu correction-pairs JSONL path is required")

    source_files: list[dict[str, object]] = []
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    clean_by_noisy: dict[str, set[str]] = {}
    total_rows = 0
    skipped_rows: list[dict[str, object]] = []
    for path in paths:
        rows = _read_limbu_correction_pair_rows(path)
        source_files.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "row_count": len(rows),
            }
        )
        total_rows += len(rows)
        for row_number, row in enumerate(rows, start=1):
            noisy_text = normalize_ocr_text(str(row.get("noisy_text") or ""))
            clean_text = normalize_ocr_text(str(row.get("clean_text") or ""))
            if not noisy_text or not clean_text:
                skipped_rows.append(
                    {
                        "path": str(path),
                        "row_number": row_number,
                        "reason": "empty_noisy_or_clean_text",
                    }
                )
                continue
            if noisy_text == clean_text:
                skipped_rows.append(
                    {
                        "path": str(path),
                        "row_number": row_number,
                        "reason": "unchanged_pair",
                        "text": noisy_text,
                    }
                )
                continue
            if _count_limbu_or_devanagari(clean_text) == 0:
                skipped_rows.append(
                    {
                        "path": str(path),
                        "row_number": row_number,
                        "reason": "clean_text_has_no_limbu_or_devanagari",
                    }
                )
                continue
            script_class = str(row.get("script_class") or "").strip()
            if not script_class:
                script_class = str(_limbu_script_profile(clean_text, script_ratio_threshold=0.20)["script_class"])
            _validate_limbu_post_correction_script_class(
                script_class,
                label=f"correction pair {path}:{row_number} script_class",
            )
            key = (noisy_text, clean_text)
            candidate = candidates.setdefault(
                key,
                {
                    "source": noisy_text,
                    "replacement": clean_text,
                    "support": 0,
                    "script_classes": set(),
                    "sample_ids": [],
                    "source_paths": set(),
                },
            )
            candidate["support"] = int(candidate["support"]) + 1
            if script_class:
                script_classes = candidate["script_classes"]
                if isinstance(script_classes, set):
                    script_classes.add(script_class)
            sample_id = row.get("sample_id")
            if sample_id:
                sample_ids = candidate["sample_ids"]
                if isinstance(sample_ids, list):
                    sample_ids.append(str(sample_id))
            source_paths = candidate["source_paths"]
            if isinstance(source_paths, set):
                source_paths.add(str(path))
            clean_by_noisy.setdefault(noisy_text, set()).add(clean_text)

    conflicts = [
        {
            "source": noisy,
            "replacements": sorted(replacements),
        }
        for noisy, replacements in sorted(clean_by_noisy.items())
        if len(replacements) > 1
    ]
    if conflicts:
        conflict_preview = "; ".join(
            f"{item['source']!r} -> {item['replacements']!r}" for item in conflicts[:5]
        )
        raise ParseError(f"conflicting Limbu correction pairs cannot form a deterministic profile: {conflict_preview}")

    rules: list[dict[str, object]] = []
    for index, ((source, replacement), candidate) in enumerate(
        sorted(candidates.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))),
        start=1,
    ):
        support = int(candidate.get("support") or 0)
        if support < min_support:
            skipped_rows.append(
                {
                    "reason": "support_below_minimum",
                    "source": source,
                    "replacement": replacement,
                    "support": support,
                    "min_support": min_support,
                }
            )
            continue
        script_classes = candidate.get("script_classes")
        script_class_list = sorted(str(item) for item in script_classes) if isinstance(script_classes, set) else []
        scoped_script_classes = [item for item in script_class_list if item in normalized_script_scope]
        if not scoped_script_classes:
            scoped_script_classes = list(normalized_script_scope)
        rule_hash = hashlib.sha256(f"{source}\0{replacement}".encode("utf-8")).hexdigest()[:12]
        sample_ids = candidate.get("sample_ids")
        sampled_ids = sample_ids[:5] if isinstance(sample_ids, list) else []
        rules.append(
            {
                "id": f"review-derived-{index:04d}-{rule_hash}",
                "match": "exact",
                "source": source,
                "replacement": replacement,
                "script_class": scoped_script_classes,
                "source_note": (
                    f"derived from {support} accepted review correction pair(s); "
                    f"sample_ids={','.join(sampled_ids)}"
                ),
            }
        )

    if not rules:
        raise ParseError("no usable changed Limbu correction pairs met the derivation criteria")

    profile = {
        "profile_id": normalized_profile_id,
        "language": "limbu",
        "script_scope": list(normalized_script_scope),
        "derivation": {
            "artifact": "limbu-post-correction-profile-derivation-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_pair_files": source_files,
            "min_support": min_support,
            "total_rows": total_rows,
            "usable_changed_pair_count": sum(int(candidate.get("support") or 0) for candidate in candidates.values()),
            "skipped_count": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "rule_count": len(rules),
            "conflict_count": len(conflicts),
            "claim_scope": "profile derivation from accepted review corrections; automatic use remains experimental until held-out admission evidence is attached",
        },
        "admission": {
            "status": "experimental",
            "claim_scope": "derived profile is not claim-ready without frozen held-out evaluation evidence",
        },
        "rules": rules,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / "limbu-post-correction-profile.json"
    derivation_path = out / "limbu-post-correction-profile-derivation.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    derivation_report = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-post-correction-profile-derivation-report-v1",
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "profile_id": normalized_profile_id,
        **profile["derivation"],
    }
    derivation_path.write_text(json.dumps(derivation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "limbu-post-correction-profile-derivation.md").write_text(
        _render_limbu_post_correction_profile_derivation_markdown(derivation_report),
        encoding="utf-8",
    )
    return {
        "profile_path": str(profile_path),
        "derivation_path": str(derivation_path),
        "profile": profile,
        "derivation": derivation_report,
    }


def score_limbu_post_correction_profile(
    profile_path: str | Path,
    correction_pair_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    frozen_eval_pack: str | None = None,
    script_ratio_threshold: float = 0.20,
) -> dict[str, object]:
    """Score a Limbu post-correction profile on held-out noisy/clean pairs."""

    source = Path(profile_path)
    profile = _load_limbu_post_correction_profile(source)
    admission_issues: list[str] = []
    admission_warnings: list[str] = []
    profile_admission = _audit_limbu_post_correction_admission(source, profile, admission_issues, admission_warnings)
    if admission_issues:
        raise ParseError("Limbu post-correction profile admission failed: " + "; ".join(admission_issues))
    paths = [Path(path) for path in correction_pair_paths]
    if not paths:
        raise ParseError("at least one held-out Limbu correction-pairs JSONL path is required")

    source_files: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    before_edits_total = 0
    after_edits_total = 0
    ref_len_total = 0
    changed_count = 0
    improved_count = 0
    regressed_count = 0
    unchanged_count = 0
    for path in paths:
        pair_rows = _read_limbu_correction_pair_rows(path)
        source_files.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "row_count": len(pair_rows),
            }
        )
        for row_number, row in enumerate(pair_rows, start=1):
            noisy_text = normalize_ocr_text(str(row.get("noisy_text") or ""))
            clean_text = normalize_ocr_text(str(row.get("clean_text") or ""))
            if not clean_text:
                raise ParseError(f"held-out Limbu correction pair has empty clean_text: {path}:{row_number}")
            script_class = str(row.get("script_class") or "").strip()
            if not script_class:
                script_class = str(_limbu_script_profile(clean_text, script_ratio_threshold=script_ratio_threshold)["script_class"])
            _validate_limbu_post_correction_script_class(
                script_class,
                label=f"held-out correction pair {path}:{row_number} script_class",
            )
            line = TextLine(
                text=noisy_text,
                line_id=str(row.get("sample_id") or f"{path.name}:{row_number}"),
                metadata={"limbu_pipeline": {"script_class": script_class}},
            )
            corrected_text, applied_rules = _apply_limbu_post_correction_rules(
                noisy_text,
                profile["rules"],
                line=line,
                script_ratio_threshold=script_ratio_threshold,
            )
            before_edits = _levenshtein_distance(list(noisy_text), list(clean_text))
            after_edits = _levenshtein_distance(list(corrected_text), list(clean_text))
            ref_len = len(list(clean_text))
            before_edits_total += before_edits
            after_edits_total += after_edits
            ref_len_total += ref_len
            if corrected_text != noisy_text:
                changed_count += 1
            if after_edits < before_edits:
                improved_count += 1
            elif after_edits > before_edits:
                regressed_count += 1
            else:
                unchanged_count += 1
            rows.append(
                {
                    "source_pair_file": str(path),
                    "row_number": row_number,
                    "sample_id": row.get("sample_id") or f"{path.name}:{row_number}",
                    "script_class": script_class,
                    "noisy_text": noisy_text,
                    "clean_text": clean_text,
                    "corrected_text": corrected_text,
                    "applied_rules": applied_rules,
                    "changed": corrected_text != noisy_text,
                    "before_codepoint_edits": before_edits,
                    "after_codepoint_edits": after_edits,
                    "reference_codepoint_length": ref_len,
                    "codepoint_edit_delta": after_edits - before_edits,
                    "status": "improved" if after_edits < before_edits else "regressed" if after_edits > before_edits else "unchanged",
                }
            )

    if not rows:
        raise ParseError("held-out Limbu post-correction eval has no rows")
    before_cer = before_edits_total / ref_len_total if ref_len_total else 0.0
    after_cer = after_edits_total / ref_len_total if ref_len_total else 0.0
    passed = after_edits_total <= before_edits_total and regressed_count == 0
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "limbu-post-correction-profile-eval-lines.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-post-correction-profile-eval-v1",
        "profile_path": str(source),
        "profile_sha256": _sha256_file(source),
        "profile_id": profile["profile_id"],
        "profile_admission": profile_admission,
        "profile_audit_warnings": admission_warnings,
        "frozen_eval_pack": frozen_eval_pack,
        "source_pair_files": source_files,
        "metric": "text_cer_codepoint_micro",
        "before": before_cer,
        "after": after_cer,
        "absolute_improvement": before_cer - after_cer,
        "sample_count": len(rows),
        "reference_codepoint_count": ref_len_total,
        "before_codepoint_edits": before_edits_total,
        "after_codepoint_edits": after_edits_total,
        "changed_count": changed_count,
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "unchanged_count": unchanged_count,
        "passed": passed,
        "claim_ready_admission_candidate": bool(passed and frozen_eval_pack),
        "lines_path": str(rows_path),
        "lines_sha256": _sha256_file(rows_path),
        "claim_scope": "held-out Limbu post-correction profile evaluation; attach this JSON hash to profile admission before treating automatic correction as validated",
    }
    report_path = out / "limbu-post-correction-profile-eval.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    eval_run_sha256 = _sha256_file(report_path)
    result = {
        **report,
        "eval_run_path": str(report_path),
        "eval_run_sha256": eval_run_sha256,
    }
    (out / "limbu-post-correction-profile-eval.md").write_text(
        _render_limbu_post_correction_profile_eval_markdown(result),
        encoding="utf-8",
    )
    return result


def admit_limbu_post_correction_profile(
    profile_path: str | Path,
    eval_run_path: str | Path,
    output_dir: str | Path,
    *,
    output_filename: str = "limbu-post-correction-profile-admitted.json",
) -> dict[str, object]:
    """Attach validated held-out eval evidence to a Limbu post-correction profile."""

    source = Path(profile_path)
    eval_path = Path(eval_run_path)
    if not source.is_file():
        raise ParseError(f"Limbu post-correction profile does not exist: {source}")
    if not eval_path.is_file():
        raise ParseError(f"Limbu post-correction profile eval JSON does not exist: {eval_path}")
    try:
        raw_profile = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid Limbu post-correction profile JSON {source}: {exc}") from exc
    if not isinstance(raw_profile, dict):
        raise ParseError(f"Limbu post-correction profile must be a JSON object: {source}")
    profile = _load_limbu_post_correction_profile(source)
    try:
        eval_run = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid Limbu post-correction profile eval JSON {eval_path}: {exc}") from exc
    if not isinstance(eval_run, dict):
        raise ParseError(f"Limbu post-correction profile eval must be a JSON object: {eval_path}")
    if eval_run.get("artifact") != "limbu-post-correction-profile-eval-v1":
        raise ParseError(f"unexpected Limbu post-correction eval artifact: {eval_run.get('artifact')!r}")
    source_sha = _sha256_file(source)
    if eval_run.get("profile_sha256") != source_sha:
        raise ParseError(
            "Limbu post-correction eval profile_sha256 does not match source profile: "
            f"eval={eval_run.get('profile_sha256')!r} profile={source_sha}"
        )
    if eval_run.get("profile_id") != profile["profile_id"]:
        raise ParseError(
            "Limbu post-correction eval profile_id does not match source profile: "
            f"eval={eval_run.get('profile_id')!r} profile={profile['profile_id']!r}"
        )
    if eval_run.get("passed") is not True:
        raise ParseError("Limbu post-correction eval did not pass; refusing validated admission")
    if eval_run.get("regressed_count") != 0:
        raise ParseError("Limbu post-correction eval has regressed held-out rows; refusing validated admission")
    metric = str(eval_run.get("metric") or "").strip()
    frozen_eval_pack = str(eval_run.get("frozen_eval_pack") or "").strip()
    before = eval_run.get("before")
    after = eval_run.get("after")
    if not metric:
        raise ParseError("Limbu post-correction eval is missing metric")
    if not frozen_eval_pack:
        raise ParseError("Limbu post-correction eval is missing frozen_eval_pack")
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        raise ParseError("Limbu post-correction eval requires numeric before and after metrics")
    if float(after) > float(before):
        raise ParseError("Limbu post-correction eval after metric is worse than before")

    admitted = dict(raw_profile)
    admitted["admission"] = {
        "status": "validated",
        "eval_run": str(eval_path.resolve()),
        "eval_run_sha256": _sha256_file(eval_path),
        "frozen_eval_pack": frozen_eval_pack,
        "metric": metric,
        "before": before,
        "after": after,
        "claim_scope": eval_run.get("claim_scope"),
        "source_profile_path": str(source),
        "source_profile_sha256": source_sha,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    admitted_path = out / output_filename
    admitted_path.write_text(json.dumps(admitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_dir = out / "audit"
    audit = audit_limbu_post_correction_profile(admitted_path, audit_dir)
    report = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-post-correction-profile-admission-v1",
        "source_profile": str(source),
        "source_profile_sha256": source_sha,
        "eval_run": str(eval_path),
        "eval_run_sha256": _sha256_file(eval_path),
        "admitted_profile": str(admitted_path),
        "admitted_profile_sha256": _sha256_file(admitted_path),
        "audit_dir": str(audit_dir),
        "audit_passed": audit.get("passed"),
        "audit_claim_ready": audit.get("claim_ready"),
        "metric": metric,
        "before": before,
        "after": after,
        "frozen_eval_pack": frozen_eval_pack,
        "claim_scope": "validated Limbu post-correction profile admission from held-out eval evidence",
    }
    (out / "limbu-post-correction-profile-admission.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "limbu-post-correction-profile-admission.md").write_text(
        _render_limbu_post_correction_profile_admission_markdown(report),
        encoding="utf-8",
    )
    return report


def build_limbu_correction_pair_pack(
    correction_pair_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    pack_id: str,
    heldout_fraction: float = 0.20,
    min_heldout: int = 1,
) -> dict[str, object]:
    """Build a frozen, hashed Limbu correction-pair pack with train/heldout splits."""

    normalized_pack_id = str(pack_id or "").strip()
    if not normalized_pack_id:
        raise ParseError("pack_id is required")
    if heldout_fraction <= 0 or heldout_fraction >= 1:
        raise ParseError(f"heldout_fraction must be between 0 and 1, got {heldout_fraction}")
    if min_heldout < 1:
        raise ParseError(f"min_heldout must be >= 1, got {min_heldout}")
    paths = [Path(path) for path in correction_pair_paths]
    if not paths:
        raise ParseError("at least one Limbu correction-pairs JSONL path is required")

    source_files: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()
    for path in paths:
        pair_rows = _read_limbu_correction_pair_rows(path)
        source_file = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "row_count": len(pair_rows),
        }
        source_files.append(source_file)
        for row_number, row in enumerate(pair_rows, start=1):
            noisy_text = normalize_ocr_text(str(row.get("noisy_text") or ""))
            clean_text = normalize_ocr_text(str(row.get("clean_text") or ""))
            if not noisy_text or not clean_text:
                raise ParseError(f"Limbu correction pair pack row needs non-empty noisy_text and clean_text: {path}:{row_number}")
            if _count_limbu_or_devanagari(clean_text) == 0:
                raise ParseError(f"Limbu correction pair pack clean_text has no Limbu or Devanagari text: {path}:{row_number}")
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                sample_id = f"{path.name}:{row_number}"
            if sample_id in seen_sample_ids:
                raise ParseError(f"duplicate Limbu correction pair sample_id {sample_id!r}")
            seen_sample_ids.add(sample_id)
            script_class = str(row.get("script_class") or "").strip()
            if not script_class:
                script_class = str(_limbu_script_profile(clean_text, script_ratio_threshold=0.20)["script_class"])
            _validate_limbu_post_correction_script_class(
                script_class,
                label=f"correction pair pack row {path}:{row_number} script_class",
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "language": "limbu",
                    "script_class": script_class,
                    "noisy_text": noisy_text,
                    "clean_text": clean_text,
                    "changed": noisy_text != clean_text,
                    "source": "limbu_review_correction_pair_pack",
                    "source_pair_file": str(path),
                    "source_pair_file_sha256": source_file["sha256"],
                    "source_row_number": row_number,
                    "review_status": row.get("review_status"),
                    "reviewer": row.get("reviewer"),
                    "claim_eligible": False,
                }
            )
    if len(rows) < 2:
        raise ParseError("Limbu correction pair pack needs at least two rows to create train and heldout splits")

    heldout_count = max(min_heldout, math.ceil(len(rows) * heldout_fraction))
    if heldout_count >= len(rows):
        heldout_count = len(rows) - 1
    sorted_rows = sorted(rows, key=lambda row: _limbu_correction_pair_split_key(row, normalized_pack_id))
    heldout_ids = {str(row["sample_id"]) for row in sorted_rows[:heldout_count]}
    train_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        split = "heldout" if row["sample_id"] in heldout_ids else "train"
        packed_row = {**row, "split": split, "pack_id": normalized_pack_id}
        canonical_rows.append(_limbu_correction_pair_pack_digest_row(packed_row))
        if split == "heldout":
            heldout_rows.append(packed_row)
        else:
            train_rows.append(packed_row)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train-limbu-correction-pairs.jsonl"
    heldout_path = out / "heldout-limbu-correction-pairs.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    heldout_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in heldout_rows),
        encoding="utf-8",
    )
    pack_content_sha256 = _limbu_correction_pair_pack_content_digest(canonical_rows)
    report = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-correction-pair-pack-v1",
        "pack_id": normalized_pack_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "language": "limbu",
        "frozen_eval_pack": True,
        "claim_eligible_default": False,
        "pack_content_sha256": pack_content_sha256,
        "source_pair_files": source_files,
        "total_rows": len(rows),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "changed_count": sum(1 for row in rows if row.get("changed")),
        "train_pairs": str(train_path),
        "train_pairs_sha256": _sha256_file(train_path),
        "heldout_pairs": str(heldout_path),
        "heldout_pairs_sha256": _sha256_file(heldout_path),
        "split_method": {
            "method": "sha256_sorted_by_pack_id_and_sample",
            "heldout_fraction": heldout_fraction,
            "min_heldout": min_heldout,
        },
        "claim_scope": "frozen Limbu correction-pair pack for post-correction derivation and held-out scoring; not OCR page ground truth",
    }
    (out / "limbu-correction-pair-pack.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "limbu-correction-pair-pack.md").write_text(
        _render_limbu_correction_pair_pack_markdown(report),
        encoding="utf-8",
    )
    return report


def audit_limbu_correction_pair_pack(
    pack_dir: str | Path,
    audit_dir: str | Path | None = None,
) -> dict[str, object]:
    """Audit a frozen Limbu correction-pair pack for stale or tampered artifacts."""

    root = Path(pack_dir)
    issues: list[str] = []
    warnings: list[str] = []
    summary_path = root / "limbu-correction-pair-pack.json"
    summary = _read_json_object(summary_path, issues, label="limbu-correction-pair-pack.json")
    if summary is None:
        report = _limbu_correction_pair_pack_audit_report(root, summary_path, None, issues, warnings)
        _write_limbu_correction_pair_pack_audit(report, Path(audit_dir) if audit_dir is not None else root)
        return report
    if summary.get("artifact") != "limbu-correction-pair-pack-v1":
        issues.append(f"unexpected correction pair pack artifact: {summary.get('artifact')!r}")
    if summary.get("frozen_eval_pack") is not True:
        issues.append("Limbu correction pair pack is not declared frozen")
    if summary.get("claim_eligible_default") is not False:
        warnings.append("Limbu correction pair pack should keep claim_eligible_default=false")

    train_path = _resolve_pack_artifact_path(summary.get("train_pairs"), root, "train-limbu-correction-pairs.jsonl")
    heldout_path = _resolve_pack_artifact_path(summary.get("heldout_pairs"), root, "heldout-limbu-correction-pairs.jsonl")
    train_rows = _read_limbu_pack_jsonl_rows(train_path, issues, label="train-limbu-correction-pairs.jsonl")
    heldout_rows = _read_limbu_pack_jsonl_rows(heldout_path, issues, label="heldout-limbu-correction-pairs.jsonl")
    all_rows = train_rows + heldout_rows

    if train_path.is_file() and summary.get("train_pairs_sha256") != _sha256_file(train_path):
        issues.append("train-limbu-correction-pairs.jsonl sha256 mismatch")
    if heldout_path.is_file() and summary.get("heldout_pairs_sha256") != _sha256_file(heldout_path):
        issues.append("heldout-limbu-correction-pairs.jsonl sha256 mismatch")
    if isinstance(summary.get("train_count"), int) and summary.get("train_count") != len(train_rows):
        issues.append(f"train_count mismatch: summary={summary.get('train_count')} actual={len(train_rows)}")
    if isinstance(summary.get("heldout_count"), int) and summary.get("heldout_count") != len(heldout_rows):
        issues.append(f"heldout_count mismatch: summary={summary.get('heldout_count')} actual={len(heldout_rows)}")
    if isinstance(summary.get("total_rows"), int) and summary.get("total_rows") != len(all_rows):
        issues.append(f"total_rows mismatch: summary={summary.get('total_rows')} actual={len(all_rows)}")
    changed_count = sum(1 for row in all_rows if row.get("changed") is True)
    if isinstance(summary.get("changed_count"), int) and summary.get("changed_count") != changed_count:
        issues.append(f"changed_count mismatch: summary={summary.get('changed_count')} actual={changed_count}")

    _audit_limbu_correction_pair_pack_rows(train_rows, expected_split="train", summary=summary, issues=issues)
    _audit_limbu_correction_pair_pack_rows(heldout_rows, expected_split="heldout", summary=summary, issues=issues)
    sample_ids = [str(row.get("sample_id") or "") for row in all_rows]
    duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_id and sample_ids.count(sample_id) > 1})
    if duplicate_ids:
        issues.append(f"duplicate Limbu correction pair sample_id values: {duplicate_ids}")
    if not train_rows:
        issues.append("Limbu correction pair pack train split is empty")
    if not heldout_rows:
        issues.append("Limbu correction pair pack heldout split is empty")

    digest_rows = [_limbu_correction_pair_pack_digest_row(row) for row in all_rows]
    recomputed_pack_sha = _limbu_correction_pair_pack_content_digest(digest_rows)
    if summary.get("pack_content_sha256") != recomputed_pack_sha:
        issues.append(
            "pack_content_sha256 mismatch: "
            f"summary={summary.get('pack_content_sha256')!r} actual={recomputed_pack_sha}"
        )

    for source in summary.get("source_pair_files") or []:
        if not isinstance(source, dict):
            issues.append("source_pair_files entries must be objects")
            continue
        source_path_value = source.get("path")
        if not isinstance(source_path_value, str) or not source_path_value:
            issues.append("source_pair_files entry missing path")
            continue
        source_path = Path(source_path_value)
        if not source_path.is_file():
            warnings.append(f"source correction-pair file is not available for hash replay: {source_path}")
            continue
        if source.get("sha256") != _sha256_file(source_path):
            issues.append(f"source correction-pair file sha256 mismatch: {source_path}")

    report = _limbu_correction_pair_pack_audit_report(
        root,
        summary_path,
        summary,
        issues,
        warnings,
        recomputed_pack_sha256=recomputed_pack_sha,
        train_count=len(train_rows),
        heldout_count=len(heldout_rows),
        total_rows=len(all_rows),
    )
    _write_limbu_correction_pair_pack_audit(report, Path(audit_dir) if audit_dir is not None else root)
    return report


def audit_limbu_output(
    output_dir: str | Path,
    audit_dir: str | Path | None = None,
    *,
    require_capture_prep: bool = False,
    min_capture_prepared_width: int | None = None,
    min_capture_prepared_height: int | None = None,
    min_capture_prepared_entropy: float | None = None,
    min_capture_prepared_luminance_stddev: float | None = None,
    min_capture_prepared_edge_stddev: float | None = None,
    require_capture_metadata_path_self: bool = False,
    require_no_pending_review: bool = False,
    require_reviewer_for_corrections: bool = False,
    require_no_dropped_image_lines: bool = False,
    min_line_count: int | None = None,
    min_average_line_confidence: float | None = None,
    min_quality_score: float | None = None,
    required_scripts: tuple[str, ...] = (),
    required_script_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Audit a Limbu OCR/correction output directory for stale or missing artifacts."""

    out = Path(output_dir)
    issues: list[str] = []
    warnings: list[str] = []
    normalized_required_scripts = _normalize_required_scripts(required_scripts)
    normalized_required_script_counts = _normalize_required_script_counts(required_script_counts)
    normalized_min_capture_prepared_width = _normalize_optional_non_negative_number(
        min_capture_prepared_width,
        "min_capture_prepared_width",
    )
    normalized_min_capture_prepared_height = _normalize_optional_non_negative_number(
        min_capture_prepared_height,
        "min_capture_prepared_height",
    )
    normalized_min_capture_prepared_entropy = _normalize_optional_non_negative_number(
        min_capture_prepared_entropy,
        "min_capture_prepared_entropy",
    )
    normalized_min_capture_prepared_luminance_stddev = _normalize_optional_non_negative_number(
        min_capture_prepared_luminance_stddev,
        "min_capture_prepared_luminance_stddev",
    )
    normalized_min_capture_prepared_edge_stddev = _normalize_optional_non_negative_number(
        min_capture_prepared_edge_stddev,
        "min_capture_prepared_edge_stddev",
    )
    normalized_min_line_count = _normalize_optional_min_int(min_line_count, "min_line_count")
    normalized_min_average_line_confidence = _normalize_optional_ratio(
        min_average_line_confidence,
        "min_average_line_confidence",
    )
    normalized_min_quality_score = _normalize_optional_ratio(min_quality_score, "min_quality_score")
    policy = {
        "require_capture_prep": require_capture_prep,
        "min_capture_prepared_width": normalized_min_capture_prepared_width,
        "min_capture_prepared_height": normalized_min_capture_prepared_height,
        "min_capture_prepared_entropy": normalized_min_capture_prepared_entropy,
        "min_capture_prepared_luminance_stddev": normalized_min_capture_prepared_luminance_stddev,
        "min_capture_prepared_edge_stddev": normalized_min_capture_prepared_edge_stddev,
        "require_capture_metadata_path_self": require_capture_metadata_path_self,
        "require_no_pending_review": require_no_pending_review,
        "require_reviewer_for_corrections": require_reviewer_for_corrections,
        "require_no_dropped_image_lines": require_no_dropped_image_lines,
        "min_line_count": normalized_min_line_count,
        "min_average_line_confidence": normalized_min_average_line_confidence,
        "min_quality_score": normalized_min_quality_score,
        "required_scripts": list(normalized_required_scripts),
        "required_script_counts": normalized_required_script_counts,
    }
    manifest_path = out / "limbu-output-manifest.json"
    manifest = _read_json_object(manifest_path, issues, label="limbu-output-manifest.json")
    if manifest is None:
        report = _limbu_output_audit_report(out, manifest_path, None, issues, warnings, policy)
        _write_limbu_output_audit(report, Path(audit_dir) if audit_dir is not None else out)
        return report

    if manifest.get("pipeline_id") != "limbu-first-ocr-pipeline-v1":
        issues.append(f"unexpected pipeline_id: {manifest.get('pipeline_id')!r}")
    stage = str(manifest.get("pipeline_stage") or "")
    if stage not in {"ocr", "review_correction"}:
        issues.append(f"unexpected pipeline_stage: {stage!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        issues.append("limbu-output-manifest.json artifacts must be a list")
        artifacts = []
    _audit_limbu_required_manifest_artifacts(out, artifacts, stage, issues)
    _audit_limbu_manifest_artifacts(out, artifacts, manifest, issues)
    document = _audit_limbu_document(out, manifest, stage, issues, warnings)
    _audit_limbu_manifest_provenance(manifest, document, stage, issues, warnings)
    capture_quality_policy = {
        "min_prepared_width": normalized_min_capture_prepared_width,
        "min_prepared_height": normalized_min_capture_prepared_height,
        "min_prepared_entropy": normalized_min_capture_prepared_entropy,
        "min_prepared_luminance_stddev": normalized_min_capture_prepared_luminance_stddev,
        "min_prepared_edge_stddev": normalized_min_capture_prepared_edge_stddev,
        "require_metadata_path_self": require_capture_metadata_path_self,
    }
    if stage == "ocr":
        _audit_limbu_ocr_stage(out, document, manifest, issues, warnings, capture_quality_policy=capture_quality_policy)
    elif stage == "review_correction":
        _audit_limbu_review_correction_stage(out, document, manifest, issues, warnings)
    _audit_limbu_output_policy(
        out,
        stage=stage,
        document=document,
        require_capture_prep=require_capture_prep,
        capture_quality_policy=capture_quality_policy,
        require_no_pending_review=require_no_pending_review,
        require_reviewer_for_corrections=require_reviewer_for_corrections,
        require_no_dropped_image_lines=require_no_dropped_image_lines,
        min_line_count=normalized_min_line_count,
        min_average_line_confidence=normalized_min_average_line_confidence,
        min_quality_score=normalized_min_quality_score,
        required_scripts=normalized_required_scripts,
        required_script_counts=normalized_required_script_counts,
        issues=issues,
    )

    run_context = manifest.get("run_context")
    if not isinstance(run_context, dict):
        issues.append("limbu-output-manifest.json run_context must be an object")
    else:
        if not run_context.get("command"):
            warnings.append("run_context.command is missing; CLI provenance is incomplete")
        if not run_context.get("hostname"):
            warnings.append("run_context.hostname is missing")

    report = _limbu_output_audit_report(out, manifest_path, manifest, issues, warnings, policy)
    _write_limbu_output_audit(report, Path(audit_dir) if audit_dir is not None else out)
    return report


def _normalize_required_scripts(required_scripts: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not required_scripts:
        return ()
    normalized: list[str] = []
    for raw_script in required_scripts:
        script = str(raw_script).strip()
        if not script:
            continue
        if script not in _LIMBU_SCRIPT_CLASSES:
            raise ParseError(f"unsupported required Limbu script class: {script!r}")
        if script not in normalized:
            normalized.append(script)
    return tuple(normalized)


def _normalize_required_script_counts(required_script_counts: dict[str, int] | None) -> dict[str, int]:
    if not required_script_counts:
        return {}
    normalized: dict[str, int] = {}
    for raw_script, raw_count in required_script_counts.items():
        script = str(raw_script).strip()
        if script not in _LIMBU_SCRIPT_CLASSES:
            raise ParseError(f"unsupported required Limbu script class: {script!r}")
        if isinstance(raw_count, bool):
            raise ParseError(f"required script count for {script!r} must be an integer, got {raw_count!r}")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ParseError(f"required script count for {script!r} must be an integer, got {raw_count!r}") from exc
        if count < 1:
            raise ParseError(f"required script count for {script!r} must be positive, got {count}")
        normalized[script] = count
    return dict(sorted(normalized.items()))


def _normalize_optional_non_negative_number(value: float | None, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ParseError(f"{label} must be a finite non-negative number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ParseError(f"{label} must be a finite non-negative number, got {value!r}")
    return int(value) if isinstance(value, int) else normalized


def _normalize_optional_min_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ParseError(f"{label} must be an integer, got {value!r}")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"{label} must be an integer, got {value!r}") from exc
    if normalized < 1:
        raise ParseError(f"{label} must be positive, got {normalized}")
    return normalized


def _normalize_optional_ratio(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ParseError(f"{label} must be a finite number between 0 and 1, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"{label} must be a finite number between 0 and 1, got {value!r}") from exc
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ParseError(f"{label} must be a finite number between 0 and 1, got {value!r}")
    return normalized


def _normalize_required_ratio(value: float, label: str) -> float:
    normalized = _normalize_optional_ratio(value, label)
    if normalized is None:
        raise ParseError(f"{label} must be a finite number between 0 and 1, got {value!r}")
    return normalized


def _limbu_document_script_ratio_threshold(document: Document, *, default: float = 0.20) -> float:
    metadata = document.metadata.get("limbu_pipeline")
    if not isinstance(metadata, dict) or "script_ratio_threshold" not in metadata:
        return default
    return _normalize_required_ratio(
        metadata.get("script_ratio_threshold"),
        "document.json metadata.limbu_pipeline.script_ratio_threshold",
    )


def annotate_limbu_document(
    document: Document,
    *,
    low_confidence_threshold: float = 0.80,
    script_confidence_thresholds: dict[str, float] | None = None,
    script_ratio_threshold: float = 0.20,
    capture_prep_metadata: str | Path | None = None,
    capture_prep_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    normalized_low_confidence_threshold = _normalize_required_ratio(
        low_confidence_threshold,
        "low_confidence_threshold",
    )
    normalized_script_ratio_threshold = _normalize_required_ratio(
        script_ratio_threshold,
        "script_ratio_threshold",
    )
    normalized_thresholds = _normalize_limbu_script_confidence_thresholds(
        low_confidence_threshold=normalized_low_confidence_threshold,
        script_confidence_thresholds=script_confidence_thresholds,
    )

    rows: list[dict[str, object]] = []
    script_counts = {
        "limbu_sirijonga": 0,
        "devanagari_limbu": 0,
        "mixed_limbu_devanagari": 0,
        "other": 0,
    }
    review_count = 0
    line_count = 0
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            line_count += 1
            profile = _limbu_script_profile(line.text, script_ratio_threshold=normalized_script_ratio_threshold)
            review_threshold = _limbu_review_confidence_threshold(
                profile["script_class"],
                low_confidence_threshold=normalized_low_confidence_threshold,
                script_confidence_thresholds=normalized_thresholds,
            )
            review_reasons = _limbu_review_reasons(
                line.text,
                line.confidence,
                profile["script_class"],
                low_confidence_threshold=review_threshold,
            )
            needs_review = bool(review_reasons)
            if needs_review:
                review_count += 1
            script_class = str(profile["script_class"])
            script_counts[script_class if script_class in script_counts else "other"] += 1
            line.metadata["limbu_pipeline"] = {
                **profile,
                "needs_review": needs_review,
                "review_reasons": review_reasons,
                "review_confidence_threshold": review_threshold,
                "review_confidence_threshold_source": "script_effective",
                "post_correction": {
                    "status": "not_configured",
                    "reason": "no Limbu-specific correction model has been attached to this pipeline run",
                },
            }
            rows.append(
                {
                    "page_index": page.page_index,
                    "line_index": line_index,
                    "line_id": line.line_id or f"p{page.page_index}-l{line_index}",
                    "text": line.text,
                    "confidence": line.confidence,
                    "review_confidence_threshold": review_threshold,
                    "bbox": line.bbox.to_list(),
                    "needs_review": needs_review,
                    "review_reasons": review_reasons,
                    **profile,
                }
            )
    capture_metadata = _load_capture_prep_metadata(capture_prep_metadata)
    if capture_metadata is not None and capture_prep_audit is not None:
        capture_metadata["audit"] = _capture_audit_summary(capture_prep_audit)
    return {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "source_path": document.source_path,
        "capture_prep": capture_metadata,
        "line_count": line_count,
        "review_line_count": review_count,
        "low_confidence_threshold": normalized_low_confidence_threshold,
        "script_confidence_thresholds": normalized_thresholds,
        "script_ratio_threshold": normalized_script_ratio_threshold,
        "script_counts": script_counts,
        "rows": rows,
        "claim_scope": "operational OCR output triage; not a claim-grade CER evaluation without reference scoring",
    }


def _audit_capture_prep_for_limbu_ocr(
    input_path: Path,
    output_dir: Path,
    capture_prep_metadata: str | Path | None,
) -> dict[str, object] | None:
    if capture_prep_metadata is None:
        return None
    audit_dir = output_dir / "capture-prep-audit"
    report = audit_limbu_capture(capture_prep_metadata, audit_dir)
    report["audit_path"] = str(audit_dir / "limbu-capture-audit.json")
    report["audit_markdown_path"] = str(audit_dir / "limbu-capture-audit.md")
    if not report.get("passed"):
        issues = report.get("issues")
        issue_text = "; ".join(str(issue) for issue in issues) if isinstance(issues, list) else "unknown issue"
        raise ParseError(f"capture prep audit failed for {capture_prep_metadata}: {issue_text}")
    capture_metadata = _load_capture_prep_metadata(capture_prep_metadata)
    if capture_metadata is not None:
        _validate_capture_prep_input_binding(input_path, capture_metadata)
    return report


def _limbu_script_confidence_thresholds(
    *,
    low_confidence_threshold: float,
    sirijonga_low_confidence_threshold: float | None,
    devanagari_low_confidence_threshold: float | None,
    mixed_script_low_confidence_threshold: float | None,
    other_script_low_confidence_threshold: float | None,
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if sirijonga_low_confidence_threshold is not None:
        overrides["limbu_sirijonga"] = sirijonga_low_confidence_threshold
    if devanagari_low_confidence_threshold is not None:
        overrides["devanagari_limbu"] = devanagari_low_confidence_threshold
    if mixed_script_low_confidence_threshold is not None:
        overrides["mixed_limbu_devanagari"] = mixed_script_low_confidence_threshold
    if other_script_low_confidence_threshold is not None:
        overrides["other"] = other_script_low_confidence_threshold
    return _normalize_limbu_script_confidence_thresholds(
        low_confidence_threshold=low_confidence_threshold,
        script_confidence_thresholds=overrides,
    )


def _normalize_limbu_script_confidence_thresholds(
    *,
    low_confidence_threshold: float,
    script_confidence_thresholds: dict[str, float] | None,
) -> dict[str, float]:
    normalized_low_confidence_threshold = _normalize_required_ratio(
        low_confidence_threshold,
        "low_confidence_threshold",
    )
    defaults = {
        "limbu_sirijonga": normalized_low_confidence_threshold,
        "devanagari_limbu": normalized_low_confidence_threshold,
        "mixed_limbu_devanagari": normalized_low_confidence_threshold,
        "other": normalized_low_confidence_threshold,
    }
    if script_confidence_thresholds is None:
        return defaults
    for key, value in script_confidence_thresholds.items():
        if key not in defaults:
            raise ParseError(f"unknown Limbu script confidence threshold key: {key}")
        defaults[key] = _normalize_required_ratio(value, f"{key} confidence threshold")
    return defaults


def _limbu_review_confidence_threshold(
    script_class: object,
    *,
    low_confidence_threshold: float,
    script_confidence_thresholds: dict[str, float],
) -> float:
    key = str(script_class)
    return float(script_confidence_thresholds.get(key, low_confidence_threshold))


def _validate_capture_prep_input_binding(input_path: Path, capture_metadata: dict[str, object]) -> None:
    prepared_path_value = capture_metadata.get("prepared_image_path")
    if not isinstance(prepared_path_value, str) or not prepared_path_value:
        raise ParseError("capture prep metadata missing prepared_image_path")
    prepared_path = _resolve_capture_metadata_artifact_path(
        prepared_path_value,
        capture_metadata.get("path"),
    )
    try:
        if input_path.resolve() != prepared_path.resolve():
            raise ParseError(f"OCR input {input_path} does not match capture prepared_image_path {prepared_path}")
    except OSError as exc:
        raise ParseError(f"could not resolve OCR input/capture prepared path: {exc}") from exc


def _resolve_capture_metadata_artifact_path(path_value: str, metadata_path_value: object) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    if isinstance(metadata_path_value, str) and metadata_path_value:
        return Path(metadata_path_value).parent / path
    return path


def _capture_audit_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "path": report.get("audit_path"),
        "markdown_path": report.get("audit_markdown_path"),
        "passed": report.get("passed"),
        "issues": report.get("issues", []),
        "warnings": report.get("warnings", []),
    }


def _load_capture_prep_metadata(capture_prep_metadata: str | Path | None) -> dict[str, object] | None:
    if capture_prep_metadata is None:
        return None
    path = Path(capture_prep_metadata)
    if not path.is_file():
        raise ParseError(f"capture prep metadata does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid capture prep metadata JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"capture prep metadata must be a JSON object: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "source_path": payload.get("source_path"),
        "raw_copy_path": payload.get("raw_copy_path"),
        "prepared_image_path": payload.get("prepared_image_path"),
        "source_sha256": payload.get("source_sha256"),
        "prepared_sha256": payload.get("prepared_sha256"),
        "operations": payload.get("operations", []),
    }


def _apply_limbu_post_correction_profile(
    document: Document,
    profile_path: str | Path | None,
    *,
    script_ratio_threshold: float,
) -> dict[str, object]:
    lines = [(page, line_index, line) for page in document.pages for line_index, line in enumerate(page.text_lines)]
    if profile_path is None:
        rows = [
            _limbu_post_correction_row(
                page_index=page.page_index,
                line_index=line_index,
                line=line,
                original_text=line.text,
                corrected_text=line.text,
                status="not_configured",
                applied_rules=[],
            )
            for page, line_index, line in lines
        ]
        return {
            "pipeline_id": "limbu-first-ocr-pipeline-v1",
            "artifact": "limbu-post-correction-audit-v1",
            "status": "not_configured",
            "profile_path": None,
            "profile_sha256": None,
            "profile_id": None,
            "line_count": len(rows),
            "changed_count": 0,
            "rules_loaded": 0,
            "rows": rows,
            "claim_scope": "no automatic Limbu post-correction was applied; review queue remains the correction gate",
        }

    profile_source = Path(profile_path)
    profile = _load_limbu_post_correction_profile(profile_source)
    admission_issues: list[str] = []
    admission_warnings: list[str] = []
    admission = _audit_limbu_post_correction_admission(profile_source, profile, admission_issues, admission_warnings)
    if admission_issues:
        raise ParseError("Limbu post-correction profile admission failed: " + "; ".join(admission_issues))
    rows: list[dict[str, object]] = []
    changed_count = 0
    for page, line_index, line in lines:
        original_text = line.text
        corrected_text, applied_rules = _apply_limbu_post_correction_rules(
            original_text,
            profile["rules"],
            line=line,
            script_ratio_threshold=script_ratio_threshold,
        )
        if corrected_text != original_text:
            line.text = corrected_text
            changed_count += 1
        rows.append(
            _limbu_post_correction_row(
                page_index=page.page_index,
                line_index=line_index,
                line=line,
                original_text=original_text,
                corrected_text=corrected_text,
                status="applied" if applied_rules else "no_change",
                applied_rules=applied_rules,
            )
        )
    return {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-post-correction-audit-v1",
        "status": "applied",
        "profile_path": str(profile_source),
        "profile_sha256": _sha256_file(profile_source),
        "profile_id": profile["profile_id"],
        "language": profile["language"],
        "script_scope": profile["script_scope"],
        "profile_admission": admission,
        "profile_audit_warnings": admission_warnings,
        "line_count": len(rows),
        "changed_count": changed_count,
        "rules_loaded": len(profile["rules"]),
        "rows": rows,
        "claim_scope": "deterministic Limbu post-correction profile output; not ground truth and not an accuracy claim without held-out scoring",
    }


def _load_limbu_post_correction_profile(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ParseError(f"Limbu post-correction profile does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid Limbu post-correction profile JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"Limbu post-correction profile must be a JSON object: {path}")
    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        raise ParseError("Limbu post-correction profile requires profile_id")
    language = str(payload.get("language") or "").strip().lower()
    if language != "limbu":
        raise ParseError(f"Limbu post-correction profile language must be 'limbu', got {language!r}")
    script_scope_payload = payload.get("script_scope", ["limbu_sirijonga", "devanagari_limbu", "mixed_limbu_devanagari"])
    if not isinstance(script_scope_payload, list) or not script_scope_payload:
        raise ParseError("Limbu post-correction profile script_scope must be a non-empty list")
    script_scope = [str(item).strip() for item in script_scope_payload if str(item).strip()]
    _validate_limbu_post_correction_script_classes(script_scope, label="script_scope")
    rules_payload = payload.get("rules")
    if not isinstance(rules_payload, list) or not rules_payload:
        raise ParseError("Limbu post-correction profile requires at least one rule")
    rules = [_normalize_limbu_post_correction_rule(item, index=index) for index, item in enumerate(rules_payload)]
    rule_ids = [str(rule.get("id") or "") for rule in rules]
    duplicate_rule_ids = sorted({rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1})
    if duplicate_rule_ids:
        raise ParseError(f"Limbu post-correction profile has duplicate rule ids: {duplicate_rule_ids}")
    for rule in rules:
        if isinstance(rule, dict) and not rule.get("script_class"):
            rule["script_class"] = script_scope
    admission = payload.get("admission")
    if admission is not None and not isinstance(admission, dict):
        raise ParseError("Limbu post-correction profile admission must be an object when present")
    return {
        "profile_id": profile_id,
        "language": language,
        "script_scope": script_scope,
        "rules": rules,
        "admission": dict(admission or {}),
    }


def _limbu_post_correction_profile_audit_report(
    profile_path: Path,
    profile: dict[str, object] | None,
    issues: list[str],
    warnings: list[str],
) -> dict[str, object]:
    admission = _audit_limbu_post_correction_admission(profile_path, profile, issues, warnings)
    rule_count = len(profile.get("rules", [])) if isinstance(profile, dict) and isinstance(profile.get("rules"), list) else 0
    script_scope = profile.get("script_scope", []) if isinstance(profile, dict) else []
    return {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-post-correction-profile-audit-v1",
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_file(profile_path) if profile_path.is_file() else None,
        "profile_id": profile.get("profile_id") if isinstance(profile, dict) else None,
        "language": profile.get("language") if isinstance(profile, dict) else None,
        "script_scope": script_scope,
        "rule_count": rule_count,
        "admission": admission,
        "claim_ready": bool(admission.get("claim_ready")) and not issues,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def _audit_limbu_post_correction_admission(
    profile_path: Path,
    profile: dict[str, object] | None,
    issues: list[str],
    warnings: list[str],
) -> dict[str, object]:
    if not isinstance(profile, dict):
        return {"status": "invalid", "claim_ready": False}
    raw = profile.get("admission")
    if not isinstance(raw, dict) or not raw:
        warnings.append("Limbu post-correction profile has no admission evidence; automatic corrections are experimental")
        return {
            "status": "experimental",
            "claim_ready": False,
            "reason": "missing_admission_evidence",
        }
    status = str(raw.get("status") or "experimental").strip().lower()
    if status not in {"experimental", "validated"}:
        issues.append(f"Limbu post-correction profile admission.status must be experimental or validated, got {status!r}")
    eval_run = str(raw.get("eval_run") or "").strip()
    eval_run_sha = str(raw.get("eval_run_sha256") or "").strip()
    frozen_eval_pack = str(raw.get("frozen_eval_pack") or "").strip()
    metric = str(raw.get("metric") or "").strip()
    before = raw.get("before")
    after = raw.get("after")
    if status == "validated":
        if not eval_run:
            issues.append("validated Limbu post-correction profile requires admission.eval_run")
        if not eval_run_sha:
            issues.append("validated Limbu post-correction profile requires admission.eval_run_sha256")
        if not frozen_eval_pack:
            issues.append("validated Limbu post-correction profile requires admission.frozen_eval_pack")
        if not metric:
            issues.append("validated Limbu post-correction profile requires admission.metric")
        if not isinstance(before, int | float):
            issues.append("validated Limbu post-correction profile requires numeric admission.before")
        if not isinstance(after, int | float):
            issues.append("validated Limbu post-correction profile requires numeric admission.after")
        if isinstance(before, int | float) and isinstance(after, int | float) and float(after) > float(before):
            issues.append("validated Limbu post-correction profile admission.after must be <= admission.before")
    resolved_eval_run: str | None = None
    if eval_run:
        eval_path = Path(eval_run)
        if not eval_path.is_absolute():
            eval_path = profile_path.parent / eval_path
        resolved_eval_run = str(eval_path)
        if not eval_path.is_file():
            issues.append(f"Limbu post-correction profile admission eval_run is missing: {eval_path}")
        elif eval_run_sha and _sha256_file(eval_path) != eval_run_sha:
            issues.append(f"Limbu post-correction profile admission eval_run_sha256 mismatch: {eval_path}")
    return {
        "status": status,
        "claim_ready": status == "validated" and not issues,
        "eval_run": eval_run or None,
        "resolved_eval_run": resolved_eval_run,
        "eval_run_sha256": eval_run_sha or None,
        "frozen_eval_pack": frozen_eval_pack or None,
        "metric": metric or None,
        "before": before,
        "after": after,
        "claim_scope": raw.get("claim_scope"),
    }


def _render_limbu_post_correction_profile_audit_markdown(report: dict[str, object]) -> str:
    admission = report.get("admission") if isinstance(report.get("admission"), dict) else {}
    lines = [
        "# Limbu Post-Correction Profile Audit",
        "",
        f"Profile: `{report.get('profile_path')}`",
        f"Profile ID: `{report.get('profile_id')}`",
        f"Passed: `{'yes' if report.get('passed') else 'no'}`",
        f"Claim ready: `{'yes' if report.get('claim_ready') else 'no'}`",
        f"Admission status: `{admission.get('status')}`",
        f"Rules: `{report.get('rule_count')}`",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues")
    lines.extend(f"- {issue}" for issue in issues) if isinstance(issues, list) and issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings")
    lines.extend(f"- {warning}" for warning in warnings) if isinstance(warnings, list) and warnings else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _render_limbu_post_correction_profile_derivation_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Limbu Post-Correction Profile Derivation",
        "",
        f"Profile: `{report.get('profile_path')}`",
        f"Profile ID: `{report.get('profile_id')}`",
        f"Profile SHA-256: `{report.get('profile_sha256')}`",
        f"Rules: `{report.get('rule_count')}`",
        f"Source rows: `{report.get('total_rows')}`",
        f"Skipped rows: `{report.get('skipped_count')}`",
        f"Min support: `{report.get('min_support')}`",
        "",
        "## Source Pair Files",
        "",
    ]
    source_files = report.get("source_pair_files")
    if isinstance(source_files, list) and source_files:
        for item in source_files:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('path')}` rows=`{item.get('row_count')}` sha256=`{item.get('sha256')}`"
                )
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_scope") or "experimental profile derivation only"), ""])
    return "\n".join(lines)


def _render_limbu_post_correction_profile_eval_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Limbu Post-Correction Profile Eval",
        "",
        f"Profile: `{report.get('profile_path')}`",
        f"Profile ID: `{report.get('profile_id')}`",
        f"Profile SHA-256: `{report.get('profile_sha256')}`",
        f"Eval JSON: `{report.get('eval_run_path')}`",
        f"Eval JSON SHA-256: `{report.get('eval_run_sha256')}`",
        f"Frozen eval pack: `{report.get('frozen_eval_pack')}`",
        f"Passed: `{'yes' if report.get('passed') else 'no'}`",
        f"Admission candidate: `{'yes' if report.get('claim_ready_admission_candidate') else 'no'}`",
        f"Metric: `{report.get('metric')}`",
        f"Before: `{report.get('before')}`",
        f"After: `{report.get('after')}`",
        f"Absolute improvement: `{report.get('absolute_improvement')}`",
        f"Samples: `{report.get('sample_count')}`",
        f"Regressed rows: `{report.get('regressed_count')}`",
        "",
        "## Source Pair Files",
        "",
    ]
    source_files = report.get("source_pair_files")
    if isinstance(source_files, list) and source_files:
        for item in source_files:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('path')}` rows=`{item.get('row_count')}` sha256=`{item.get('sha256')}`"
                )
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_scope") or "held-out profile eval only"), ""])
    return "\n".join(lines)


def _render_limbu_post_correction_profile_admission_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Limbu Post-Correction Profile Admission",
        "",
        f"Source profile: `{report.get('source_profile')}`",
        f"Source profile SHA-256: `{report.get('source_profile_sha256')}`",
        f"Eval JSON: `{report.get('eval_run')}`",
        f"Eval JSON SHA-256: `{report.get('eval_run_sha256')}`",
        f"Admitted profile: `{report.get('admitted_profile')}`",
        f"Admitted profile SHA-256: `{report.get('admitted_profile_sha256')}`",
        f"Audit passed: `{'yes' if report.get('audit_passed') else 'no'}`",
        f"Audit claim ready: `{'yes' if report.get('audit_claim_ready') else 'no'}`",
        f"Frozen eval pack: `{report.get('frozen_eval_pack')}`",
        f"Metric: `{report.get('metric')}`",
        f"Before: `{report.get('before')}`",
        f"After: `{report.get('after')}`",
        "",
        "## Claim Boundary",
        "",
        str(report.get("claim_scope") or "profile admission from held-out eval evidence"),
        "",
    ]
    return "\n".join(lines)


def _render_limbu_correction_pair_pack_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Limbu Correction Pair Pack",
        "",
        f"Pack ID: `{report.get('pack_id')}`",
        f"Frozen eval pack: `{'yes' if report.get('frozen_eval_pack') else 'no'}`",
        f"Pack content SHA-256: `{report.get('pack_content_sha256')}`",
        f"Total rows: `{report.get('total_rows')}`",
        f"Train rows: `{report.get('train_count')}`",
        f"Heldout rows: `{report.get('heldout_count')}`",
        f"Changed rows: `{report.get('changed_count')}`",
        f"Train pairs: `{report.get('train_pairs')}`",
        f"Heldout pairs: `{report.get('heldout_pairs')}`",
        "",
        "## Source Pair Files",
        "",
    ]
    source_files = report.get("source_pair_files")
    if isinstance(source_files, list) and source_files:
        for item in source_files:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('path')}` rows=`{item.get('row_count')}` sha256=`{item.get('sha256')}`"
                )
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_scope") or "correction-pair pack only"), ""])
    return "\n".join(lines)


def _limbu_correction_pair_pack_audit_report(
    pack_dir: Path,
    summary_path: Path,
    summary: dict[str, object] | None,
    issues: list[str],
    warnings: list[str],
    *,
    recomputed_pack_sha256: str | None = None,
    train_count: int = 0,
    heldout_count: int = 0,
    total_rows: int = 0,
) -> dict[str, object]:
    return {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-correction-pair-pack-audit-v1",
        "pack_dir": str(pack_dir),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path) if summary_path.is_file() else None,
        "pack_id": summary.get("pack_id") if isinstance(summary, dict) else None,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "train_count": train_count,
        "heldout_count": heldout_count,
        "total_rows": total_rows,
        "summary_pack_content_sha256": summary.get("pack_content_sha256") if isinstance(summary, dict) else None,
        "recomputed_pack_content_sha256": recomputed_pack_sha256,
        "claim_scope": "artifact integrity audit for a frozen Limbu correction-pair pack",
    }


def _write_limbu_correction_pair_pack_audit(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "limbu-correction-pair-pack-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Limbu Correction Pair Pack Audit",
        "",
        f"Pack dir: `{report.get('pack_dir')}`",
        f"Pack ID: `{report.get('pack_id')}`",
        f"Passed: `{'yes' if report.get('passed') else 'no'}`",
        f"Train rows: `{report.get('train_count')}`",
        f"Heldout rows: `{report.get('heldout_count')}`",
        f"Summary pack SHA-256: `{report.get('summary_pack_content_sha256')}`",
        f"Recomputed pack SHA-256: `{report.get('recomputed_pack_content_sha256')}`",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues")
    lines.extend(f"- {issue}" for issue in issues) if isinstance(issues, list) and issues else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings")
    lines.extend(f"- {warning}" for warning in warnings) if isinstance(warnings, list) and warnings else lines.append("- none")
    lines.append("")
    (output_dir / "limbu-correction-pair-pack-audit.md").write_text("\n".join(lines), encoding="utf-8")


def _normalize_limbu_post_correction_rule(raw_rule: object, *, index: int) -> dict[str, object]:
    if not isinstance(raw_rule, dict):
        raise ParseError(f"Limbu post-correction rule {index} must be an object")
    rule_id = str(raw_rule.get("id") or "").strip()
    if not rule_id:
        raise ParseError(f"Limbu post-correction rule {index} requires id")
    match = str(raw_rule.get("match") or "exact").strip().lower()
    if match not in {"exact", "token"}:
        raise ParseError(f"Limbu post-correction rule {rule_id!r} match must be exact or token")
    source = normalize_ocr_text(str(raw_rule.get("source") or ""))
    replacement = normalize_ocr_text(str(raw_rule.get("replacement") or ""))
    if not source:
        raise ParseError(f"Limbu post-correction rule {rule_id!r} requires non-empty source")
    if not replacement:
        raise ParseError(f"Limbu post-correction rule {rule_id!r} requires non-empty replacement")
    if _count_limbu_or_devanagari(replacement) == 0:
        raise ParseError(f"Limbu post-correction rule {rule_id!r} replacement must contain Limbu or Devanagari text")
    script_class_payload = raw_rule.get("script_class")
    script_classes: list[str] = []
    if isinstance(script_class_payload, list):
        script_classes = [str(item).strip() for item in script_class_payload if str(item).strip()]
        _validate_limbu_post_correction_script_classes(script_classes, label=f"rule {rule_id!r} script_class")
    elif script_class_payload is not None:
        script_classes = [str(script_class_payload).strip()]
        _validate_limbu_post_correction_script_classes(script_classes, label=f"rule {rule_id!r} script_class")
    return {
        "id": rule_id,
        "match": match,
        "source": source,
        "replacement": replacement,
        "script_class": script_classes,
        "source_note": str(raw_rule.get("source_note") or ""),
    }


def _validate_limbu_post_correction_script_classes(script_classes: list[str], *, label: str) -> None:
    if not script_classes:
        raise ParseError(f"Limbu post-correction profile {label} must contain at least one script class")
    unsupported = sorted({item for item in script_classes if item not in _LIMBU_SCRIPT_CLASSES})
    if unsupported:
        raise ParseError(
            f"Limbu post-correction profile {label} has unsupported script classes: "
            f"{unsupported}; allowed={list(_LIMBU_SCRIPT_CLASSES)}"
        )


def _validate_limbu_post_correction_script_class(script_class: str, *, label: str) -> None:
    _validate_limbu_post_correction_script_classes([script_class], label=label)


def _apply_limbu_post_correction_rules(
    text: str,
    rules: object,
    *,
    line: TextLine,
    script_ratio_threshold: float,
) -> tuple[str, list[str]]:
    if not isinstance(rules, list):
        return text, []
    corrected = normalize_ocr_text(text)
    applied: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if not _limbu_post_correction_rule_applies_to_line(rule, line, script_ratio_threshold=script_ratio_threshold):
            continue
        next_text = _apply_limbu_post_correction_rule(corrected, rule)
        if next_text != corrected:
            corrected = next_text
            applied.append(str(rule.get("id") or "unknown"))
    return corrected, applied


def _limbu_post_correction_rule_applies_to_line(
    rule: dict[str, object],
    line: TextLine,
    *,
    script_ratio_threshold: float,
) -> bool:
    script_classes = rule.get("script_class")
    if not script_classes:
        return True
    profile = line.metadata.get("limbu_pipeline")
    script_class = profile.get("script_class") if isinstance(profile, dict) else None
    if not script_class:
        script_class = _limbu_script_profile(line.text, script_ratio_threshold=script_ratio_threshold)["script_class"]
    return str(script_class) in {str(item) for item in script_classes if str(item)}


def _apply_limbu_post_correction_rule(text: str, rule: dict[str, object]) -> str:
    source = str(rule.get("source") or "")
    replacement = str(rule.get("replacement") or "")
    match = str(rule.get("match") or "exact")
    if match == "exact":
        return replacement if text == source else text
    if match == "token":
        parts = re.split(r"(\s+)", text)
        changed = False
        for index, part in enumerate(parts):
            if part == source:
                parts[index] = replacement
                changed = True
        return "".join(parts) if changed else text
    return text


def _limbu_post_correction_row(
    *,
    page_index: int,
    line_index: int,
    line: TextLine,
    original_text: str,
    corrected_text: str,
    status: str,
    applied_rules: list[str],
) -> dict[str, object]:
    limbu_metadata = line.metadata.get("limbu_pipeline")
    if not isinstance(limbu_metadata, dict):
        limbu_metadata = {}
    return {
        "page_index": page_index,
        "line_index": line_index,
        "line_id": line.line_id or f"p{page_index}-l{line_index}",
        "script_class": limbu_metadata.get("script_class"),
        "confidence": line.confidence,
        "status": status,
        "changed": original_text != corrected_text,
        "applied_rules": applied_rules,
        "original_text": original_text,
        "corrected_text": corrected_text,
    }


def _attach_limbu_post_correction_metadata(
    document: Document,
    post_correction_audit: dict[str, object],
    pipeline_audit: dict[str, object],
) -> None:
    rows = post_correction_audit.get("rows")
    if not isinstance(rows, list):
        rows = []
    by_key = {_line_row_key(row): row for row in rows if isinstance(row, dict) and _line_row_key(row)}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"line_id:{line.line_id}" if line.line_id else f"position:{page.page_index}:{line_index}"
            row = by_key.get(key)
            if row is None:
                continue
            limbu_metadata = line.metadata.setdefault("limbu_pipeline", {})
            if isinstance(limbu_metadata, dict):
                limbu_metadata["post_correction"] = _line_post_correction_metadata(row, post_correction_audit)
    audit_rows = pipeline_audit.get("rows")
    if isinstance(audit_rows, list):
        for row in audit_rows:
            if not isinstance(row, dict):
                continue
            source = by_key.get(_line_row_key(row) or "")
            if source is not None:
                row["post_correction"] = _line_post_correction_metadata(source, post_correction_audit)


def _line_post_correction_metadata(row: dict[str, object], audit: dict[str, object]) -> dict[str, object]:
    return {
        "status": row.get("status"),
        "method": "limbu_deterministic_profile" if audit.get("profile_path") else "not_configured",
        "profile_id": audit.get("profile_id"),
        "profile_path": audit.get("profile_path"),
        "profile_sha256": audit.get("profile_sha256"),
        "changed": row.get("changed"),
        "applied_rules": row.get("applied_rules", []),
        "original_text": row.get("original_text"),
        "corrected_text": row.get("corrected_text"),
    }


def _post_correction_summary(audit: dict[str, object]) -> dict[str, object]:
    return {
        "status": audit.get("status"),
        "profile_id": audit.get("profile_id"),
        "profile_path": audit.get("profile_path"),
        "profile_sha256": audit.get("profile_sha256"),
        "rules_loaded": audit.get("rules_loaded"),
        "line_count": audit.get("line_count"),
        "changed_count": audit.get("changed_count"),
        "profile_admission": audit.get("profile_admission"),
        "profile_audit_warnings": audit.get("profile_audit_warnings"),
        "claim_scope": audit.get("claim_scope"),
    }


def _write_limbu_post_correction_artifacts(audit: dict[str, object], output_dir: Path) -> None:
    rows = audit.get("rows")
    if not isinstance(rows, list):
        rows = []
    summary = {key: value for key, value in audit.items() if key != "rows"}
    (output_dir / "limbu-post-correction-audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "limbu-post-correction-lines.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if isinstance(row, dict)),
        encoding="utf-8",
    )


def _count_limbu_or_devanagari(text: str) -> int:
    chars = [char for char in text if not char.isspace()]
    return _count_range(chars, _LIMBU_CODEPOINT_START, _LIMBU_CODEPOINT_END) + _count_range(
        chars,
        _DEVANAGARI_CODEPOINT_START,
        _DEVANAGARI_CODEPOINT_END,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_limbu_output_manifest(
    output_dir: Path,
    *,
    pipeline_stage: str,
    document: Document,
    source_path: Path,
    run_context: dict[str, object] | None,
    provenance: dict[str, object] | None,
    expected_artifacts: list[str],
) -> None:
    artifact_names = [*expected_artifacts, "figures/metadata.json"]
    if (output_dir / "tables").is_dir():
        artifact_names.extend(str(path.relative_to(output_dir)) for path in sorted((output_dir / "tables").glob("*")) if path.is_file())
    if (output_dir / "figures").is_dir():
        artifact_names.extend(
            str(path.relative_to(output_dir))
            for path in sorted((output_dir / "figures").glob("*"))
            if path.is_file() and path.name != "metadata.json"
        )
    if (output_dir / "line-crops").is_dir():
        artifact_names.extend(
            str(path.relative_to(output_dir))
            for path in sorted((output_dir / "line-crops").glob("*.png"))
            if path.is_file()
        )
    if (output_dir / "image-line-crops").is_dir():
        artifact_names.extend(
            str(path.relative_to(output_dir))
            for path in sorted((output_dir / "image-line-crops").glob("*.png"))
            if path.is_file()
        )
    artifacts: list[dict[str, object]] = []
    seen: set[str] = set()
    for name in artifact_names:
        if name in seen:
            continue
        seen.add(name)
        path = output_dir / name
        artifacts.append(_artifact_manifest_entry(path, output_dir=output_dir))
    manifest = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "pipeline_stage": pipeline_stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path) if source_path.is_file() else None,
        "document_source_path": document.source_path,
        "output_dir": str(output_dir),
        "run_context": _default_run_context(run_context),
        "provenance": provenance or {},
        "document": {
            "page_count": len(document.pages),
            "line_count": sum(len(page.text_lines) for page in document.pages),
            "table_count": len(document.tables),
            "figure_count": len(document.figures),
        },
        "artifacts": artifacts,
        "missing_artifacts": [item["path"] for item in artifacts if not item["exists"]],
        "claim_scope": "operational Limbu OCR artifact manifest; claim-grade accuracy still requires frozen page eval references",
    }
    (output_dir / "limbu-output-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_limbu_image_line_artifacts(metadata: dict[str, object], output_dir: Path) -> None:
    crop_rows = metadata.get("crop_manifest")
    if not isinstance(crop_rows, list):
        crop_rows = []
    crops_dir = output_dir / "image-line-crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_manifest_path = crops_dir / "manifest.jsonl"
    crop_manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in crop_rows if isinstance(row, dict)),
        encoding="utf-8",
    )
    run = {key: value for key, value in metadata.items() if key != "crop_manifest"}
    run["crop_manifest_path"] = str(crop_manifest_path)
    run["crop_manifest_sha256"] = _sha256_file(crop_manifest_path)
    (output_dir / "image-line-ocr-run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _limbu_run_provenance(
    *,
    model_config: str | Path | None = None,
    fallback_model_config: str | Path | None = None,
    capture_prep_metadata: str | Path | None = None,
    post_correction_profile: str | Path | None = None,
    input_document: str | Path | None = None,
    review_queue: str | Path | None = None,
) -> dict[str, object]:
    entries: dict[str, object] = {}
    for key, value in (
        ("model_config", model_config),
        ("fallback_model_config", fallback_model_config),
        ("capture_prep_metadata", capture_prep_metadata),
        ("post_correction_profile", post_correction_profile),
        ("input_document", input_document),
        ("review_queue", review_queue),
    ):
        if value is not None:
            entries[key] = _input_provenance_entry(Path(value))
    return entries


def _input_provenance_entry(path: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256_file(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _write_limbu_line_crop_artifacts(document: Document, output_dir: Path, source_path: Path) -> None:
    crops_dir = output_dir / "line-crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = crops_dir / "manifest.jsonl"
    summary_path = crops_dir / "summary.json"
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    missing_bbox_count = 0
    crop_count = 0
    page_images = _line_crop_page_images(source_path, len(document.pages), warnings)
    pil_image = None
    try:
        from PIL import Image

        pil_image = Image
    except ImportError as exc:
        warnings.append(f"Pillow is required for line crop export: {exc}")
    for page in document.pages:
        page_image_path = page_images.get(page.page_index)
        for line_index, line in enumerate(page.text_lines):
            line_id = line.line_id or f"p{page.page_index}-l{line_index}"
            crop_path: Path | None = None
            crop_sha256: str | None = None
            crop_warning: str | None = None
            bbox = line.bbox.to_list()
            if line.bbox.w <= 0 or line.bbox.h <= 0:
                missing_bbox_count += 1
                crop_warning = "line bbox has non-positive width or height"
            elif pil_image is None:
                crop_warning = "Pillow unavailable"
            elif page_image_path is None:
                crop_warning = "page image unavailable for line crop"
            else:
                crop_path = crops_dir / f"p{page.page_index:04d}-l{line_index:04d}-{_safe_line_crop_id(line_id)}.png"
                try:
                    _write_line_crop(page_image_path, crop_path, line.bbox.to_list())
                    crop_sha256 = _sha256_file(crop_path)
                    crop_count += 1
                except Exception as exc:
                    crop_path = None
                    crop_warning = f"{type(exc).__name__}: {exc}"
            if crop_warning is not None:
                warnings.append(f"{line_id}: {crop_warning}")
            limbu_metadata = line.metadata.get("limbu_pipeline")
            if not isinstance(limbu_metadata, dict):
                limbu_metadata = {}
            rows.append(
                {
                    "page_index": page.page_index,
                    "line_index": line_index,
                    "line_id": line_id,
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": bbox,
                    "script_class": limbu_metadata.get("script_class"),
                    "needs_review": limbu_metadata.get("needs_review"),
                    "review_reasons": limbu_metadata.get("review_reasons", []),
                    "crop_path": str(crop_path) if crop_path is not None else None,
                    "crop_sha256": crop_sha256,
                    "source_image_path": str(page_image_path) if page_image_path is not None else None,
                    "source_image_sha256": _sha256_file(page_image_path) if page_image_path is not None and page_image_path.is_file() else None,
                    "warning": crop_warning,
                }
            )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "artifact": "limbu-line-crops-v1",
        "source_path": str(source_path),
        "line_count": len(rows),
        "crop_count": crop_count,
        "missing_bbox_count": missing_bbox_count,
        "warning_count": len(warnings),
        "warnings": warnings,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "claim_scope": "line crop lineage for review/training; not oracle recognizer ground truth without human reference labels",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_limbu_review_dashboard(output_dir: Path) -> None:
    issues: list[str] = []
    review_rows = _read_tsv_rows(output_dir / "limbu-review-queue.tsv", issues, label="limbu-review-queue.tsv")
    crop_rows = _read_jsonl_objects(output_dir / "line-crops" / "manifest.jsonl", issues, label="line-crops/manifest.jsonl")
    crop_by_key: dict[str, dict[str, object]] = {}
    for row in crop_rows:
        key = _line_row_key(row)
        if key:
            crop_by_key[key] = row
    cards: list[str] = []
    for row in review_rows:
        key = _line_row_key(row)
        crop_row = crop_by_key.get(key or "")
        cards.append(_limbu_review_dashboard_card(row, crop_row, output_dir=output_dir))
    issue_markup = ""
    if issues:
        issue_items = "\n".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
        issue_markup = f'<section class="issues"><h2>Dashboard Build Issues</h2><ul>{issue_items}</ul></section>'
    empty_markup = ""
    if not cards:
        empty_markup = '<section class="empty">No lines require review for this run.</section>'
    body = "\n".join(cards)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Limbu OCR Review Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: #d8dee8;
      --accent: #0f766e;
      --warn: #9f1239;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px;
    }}
    .review-card {{
      display: grid;
      grid-template-columns: minmax(240px, 42%) 1fr;
      gap: 16px;
      margin: 0 0 14px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    .crop {{
      min-height: 96px;
      display: grid;
      align-content: center;
      border: 1px solid var(--line);
      background: #fdfdfd;
      overflow: auto;
    }}
    .crop img {{
      max-width: 100%;
      height: auto;
      image-rendering: auto;
      display: block;
    }}
    .missing-crop {{
      padding: 14px;
      color: var(--warn);
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f7fafc;
      color: var(--muted);
      white-space: nowrap;
    }}
    .reasons {{
      color: var(--warn);
    }}
    dl {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 6px 10px;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    .copy-row {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }}
    .tsv {{
      width: 100%;
      box-sizing: border-box;
      resize: vertical;
      min-height: 62px;
      font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--ink);
    }}
    .issues, .empty {{
      margin-bottom: 14px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    @media (max-width: 760px) {{
      .review-card {{
        grid-template-columns: 1fr;
      }}
      dl {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Limbu OCR Review Dashboard</h1>
    <p>Review rows mirror <code>limbu-review-queue.tsv</code>; apply accepted edits with <code>ocrtech apply-limbu-review-corrections</code>.</p>
  </header>
  <main>
    {issue_markup}
    {empty_markup}
    {body}
  </main>
</body>
</html>
"""
    (output_dir / "limbu-review-dashboard.html").write_text(page, encoding="utf-8")


def _limbu_review_dashboard_card(
    review_row: dict[str, str],
    crop_row: dict[str, object] | None,
    *,
    output_dir: Path,
) -> str:
    line_id = str(review_row.get("line_id") or "")
    page_index = str(review_row.get("page_index") or "")
    line_index = str(review_row.get("line_index") or "")
    text = str(review_row.get("text") or "")
    script_class = str(review_row.get("script_class") or "")
    confidence = str(review_row.get("confidence") or "")
    threshold = str(review_row.get("review_confidence_threshold") or "")
    reasons = str(review_row.get("review_reasons") or "")
    crop_markup = _limbu_dashboard_crop_markup(crop_row, output_dir=output_dir)
    tsv_row = "\t".join(
        [
            page_index,
            line_index,
            line_id,
            script_class,
            confidence,
            threshold,
            reasons,
            text,
            "accepted",
            text,
            "",
            "",
        ]
    )
    return f"""<article class="review-card" data-line-id="{html.escape(line_id, quote=True)}">
  <div class="crop">{crop_markup}</div>
  <div class="details">
    <div class="meta">
      <span class="pill">page {html.escape(page_index)}</span>
      <span class="pill">line {html.escape(line_index)}</span>
      <span class="pill">{html.escape(script_class)}</span>
      <span class="pill">confidence {html.escape(confidence)}</span>
      <span class="pill">threshold {html.escape(threshold)}</span>
      <span class="pill reasons">{html.escape(reasons)}</span>
    </div>
    <dl>
      <dt>line_id</dt><dd><code>{html.escape(line_id)}</code></dd>
      <dt>OCR text</dt><dd>{html.escape(text)}</dd>
      <dt>review_status</dt><dd><code>pending</code> or <code>accepted</code></dd>
      <dt>corrected_text</dt><dd>Replace the OCR text in <code>limbu-review-queue.tsv</code> before applying corrections.</dd>
    </dl>
    <div class="copy-row">
      <textarea class="tsv" readonly>{html.escape(tsv_row)}</textarea>
    </div>
  </div>
</article>"""


def _limbu_dashboard_crop_markup(crop_row: dict[str, object] | None, *, output_dir: Path) -> str:
    if crop_row is None:
        return '<div class="missing-crop">No matching crop manifest row.</div>'
    crop_path = crop_row.get("crop_path")
    warning = crop_row.get("warning")
    if not crop_path:
        message = str(warning or "No crop was emitted for this line.")
        return f'<div class="missing-crop">{html.escape(message)}</div>'
    href = _dashboard_artifact_href(str(crop_path), output_dir=output_dir)
    alt = f"line crop for {crop_row.get('line_id') or 'review line'}"
    return f'<a href="{html.escape(href, quote=True)}"><img src="{html.escape(href, quote=True)}" alt="{html.escape(alt, quote=True)}"></a>'


def _dashboard_artifact_href(path_value: str, *, output_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            return path.as_uri()
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _line_row_key(row: dict[str, object]) -> str | None:
    line_id = row.get("line_id")
    if line_id:
        return f"line_id:{line_id}"
    page_index = row.get("page_index")
    line_index = row.get("line_index")
    if page_index is None or line_index is None:
        return None
    return f"position:{page_index}:{line_index}"


def _line_crop_page_images(source_path: Path, page_count: int, warnings: list[str]) -> dict[int, Path]:
    if source_path.is_dir():
        page_paths = [
            path
            for path in sorted(source_path.iterdir())
            if path.is_file() and path.suffix.lower() in _CROPPABLE_IMAGE_SUFFIXES
        ]
        if not page_paths:
            warnings.append(f"page bundle has no croppable image files: {source_path}")
            return {}
        return {index: path for index, path in enumerate(page_paths[:page_count])}
    if source_path.suffix.lower() not in _CROPPABLE_IMAGE_SUFFIXES:
        warnings.append(f"source image format cannot be cropped: {source_path.suffix or '<none>'}")
        return {}
    return {0: source_path}


def _write_line_crop(source_image: Path, crop_path: Path, bbox: list[float]) -> None:
    from PIL import Image

    left = int(round(max(0, bbox[0])))
    top = int(round(max(0, bbox[1])))
    right = int(round(max(left, bbox[0] + bbox[2])))
    bottom = int(round(max(top, bbox[1] + bbox[3])))
    with Image.open(source_image) as image:
        right = min(right, image.width)
        bottom = min(bottom, image.height)
        if right <= left or bottom <= top:
            raise ParseError(f"line bbox is outside source image bounds: {bbox}")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(crop_path)


def _safe_line_crop_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-") or "line"


def _artifact_manifest_entry(path: Path, *, output_dir: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path.relative_to(output_dir)),
        "exists": exists,
        "sha256": _sha256_file(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _default_run_context(run_context: dict[str, object] | None) -> dict[str, object]:
    context = dict(run_context or {})
    context.setdefault("hostname", socket.gethostname())
    context.setdefault("platform", platform.platform())
    context.setdefault("python_version", platform.python_version())
    return context


def _read_json_object(path: Path, issues: list[str], *, label: str) -> dict[str, object] | None:
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"{label} is invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        issues.append(f"{label} must contain a JSON object")
        return None
    return payload


def _audit_limbu_manifest_artifacts(
    output_dir: Path,
    artifacts: list[object],
    manifest: dict[str, object],
    issues: list[str],
) -> None:
    recomputed_missing: list[str] = []
    seen_paths: set[str] = set()
    for index, raw_artifact in enumerate(artifacts):
        if not isinstance(raw_artifact, dict):
            issues.append(f"artifact entry {index} must be an object")
            continue
        artifact_path_value = raw_artifact.get("path")
        if not isinstance(artifact_path_value, str) or not artifact_path_value:
            issues.append(f"artifact entry {index} has empty path")
            continue
        artifact_rel = Path(artifact_path_value)
        if artifact_rel.is_absolute() or ".." in artifact_rel.parts:
            issues.append(f"artifact path must be relative and stay inside output dir: {artifact_path_value}")
            continue
        if artifact_path_value in seen_paths:
            issues.append(f"duplicate artifact path in output manifest: {artifact_path_value}")
            continue
        seen_paths.add(artifact_path_value)
        artifact_path = output_dir / artifact_rel
        exists = artifact_path.is_file()
        if bool(raw_artifact.get("exists")) != exists:
            issues.append(f"artifact exists flag mismatch for {artifact_path_value}")
        if not exists:
            recomputed_missing.append(artifact_path_value)
            continue
        recorded_sha = raw_artifact.get("sha256")
        if not isinstance(recorded_sha, str) or not recorded_sha:
            issues.append(f"artifact missing sha256: {artifact_path_value}")
        else:
            actual_sha = _sha256_file(artifact_path)
            if actual_sha != recorded_sha:
                issues.append(f"artifact sha256 mismatch for {artifact_path_value}: manifest={recorded_sha} actual={actual_sha}")
        recorded_size = raw_artifact.get("size_bytes")
        if not isinstance(recorded_size, int):
            issues.append(f"artifact missing integer size_bytes: {artifact_path_value}")
        elif artifact_path.stat().st_size != recorded_size:
            issues.append(f"artifact size mismatch for {artifact_path_value}: manifest={recorded_size} actual={artifact_path.stat().st_size}")
    recorded_missing = manifest.get("missing_artifacts")
    if not isinstance(recorded_missing, list):
        issues.append("limbu-output-manifest.json missing_artifacts must be a list")
    else:
        recorded_missing_strings = [str(item) for item in recorded_missing]
        if sorted(recorded_missing_strings) != sorted(recomputed_missing):
            issues.append(
                "missing_artifacts mismatch: "
                f"manifest={sorted(recorded_missing_strings)} recomputed={sorted(recomputed_missing)}"
            )


def _audit_limbu_required_manifest_artifacts(
    output_dir: Path,
    artifacts: list[object],
    stage: str,
    issues: list[str],
) -> None:
    if stage == "ocr":
        required = _LIMBU_REQUIRED_OCR_ARTIFACTS
    elif stage == "review_correction":
        required = _LIMBU_REQUIRED_REVIEW_CORRECTION_ARTIFACTS
    else:
        return
    manifest_paths = {
        str(artifact.get("path"))
        for artifact in artifacts
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }
    for required_path in required:
        if required_path not in manifest_paths:
            issues.append(f"required {stage} artifact is missing from limbu-output-manifest.json: {required_path}")
        elif not (output_dir / required_path).is_file():
            issues.append(f"required {stage} artifact is missing from output directory: {required_path}")


def _audit_limbu_manifest_provenance(
    manifest: dict[str, object],
    document: Document | None,
    stage: str,
    issues: list[str],
    warnings: list[str],
) -> None:
    provenance = manifest.get("provenance")
    if provenance is None:
        warnings.append("limbu-output-manifest.json missing provenance; input model/profile hashes are incomplete")
        return
    if not isinstance(provenance, dict):
        issues.append("limbu-output-manifest.json provenance must be an object")
        return
    for key, raw_entry in provenance.items():
        if not isinstance(raw_entry, dict):
            issues.append(f"provenance.{key} must be an object")
            continue
        _audit_input_provenance_entry(str(key), raw_entry, issues)
    if stage == "ocr" and document is not None:
        candidate_model = document.metadata.get("candidate_model")
        if isinstance(candidate_model, dict) and candidate_model.get("model_config") and "model_config" not in provenance:
            warnings.append("OCR document records candidate_model.model_config but manifest provenance lacks model_config")
        _audit_limbu_image_line_model_config_provenance(document, provenance, issues)


def _audit_limbu_image_line_model_config_provenance(
    document: Document,
    provenance: dict[object, object],
    issues: list[str],
) -> None:
    if not _document_uses_image_line_ocr(document):
        return
    image_line_metadata = document.metadata.get("image_line_ocr")
    if not isinstance(image_line_metadata, dict):
        return
    model_config_value = image_line_metadata.get("model_config")
    if not isinstance(model_config_value, str) or not model_config_value:
        return
    raw_entry = provenance.get("model_config")
    if not isinstance(raw_entry, dict):
        issues.append("image-line OCR records model_config but manifest provenance lacks model_config")
        return
    provenance_path_value = raw_entry.get("path")
    if not isinstance(provenance_path_value, str) or not provenance_path_value:
        issues.append("provenance.model_config.path is missing for image-line OCR model_config")
        return
    model_config_path = Path(model_config_value)
    provenance_path = Path(provenance_path_value)
    try:
        if model_config_path.resolve() != provenance_path.resolve():
            issues.append(
                "image-line OCR model_config does not match manifest provenance.model_config.path: "
                f"document={model_config_value!r} provenance={provenance_path_value!r}"
            )
    except OSError as exc:
        issues.append(f"image-line OCR model_config path cannot be resolved for provenance comparison: {exc}")
        return
    if not model_config_path.is_file():
        return
    recorded_sha = raw_entry.get("sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha:
        issues.append("provenance.model_config.sha256 is missing for image-line OCR model_config")
    elif _sha256_file(model_config_path) != recorded_sha:
        issues.append(f"image-line OCR model_config sha256 mismatch: {model_config_path}")


def _manifest_provenance_entry(manifest: dict[str, object], provenance_key: str) -> dict[str, object] | None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return None
    raw_entry = provenance.get(provenance_key)
    if isinstance(raw_entry, dict):
        return raw_entry
    return None


def _audit_manifest_provenance_binding(
    manifest: dict[str, object],
    *,
    provenance_key: str,
    audit_label: str,
    audit_path: object,
    audit_sha: object,
    issues: list[str],
) -> None:
    raw_entry = _manifest_provenance_entry(manifest, provenance_key)
    if raw_entry is None:
        issues.append(f"manifest provenance lacks {provenance_key} for {audit_label}")
        return
    provenance_path_value = raw_entry.get("path")
    if not isinstance(provenance_path_value, str) or not provenance_path_value:
        issues.append(f"provenance.{provenance_key}.path is missing for {audit_label}")
    elif not isinstance(audit_path, str) or not audit_path:
        issues.append(f"{audit_label} path is missing")
    else:
        try:
            if Path(provenance_path_value).resolve() != Path(audit_path).resolve():
                issues.append(
                    f"manifest provenance.{provenance_key}.path does not match {audit_label} path: "
                    f"provenance={provenance_path_value!r} audit={audit_path!r}"
                )
        except OSError as exc:
            issues.append(f"could not resolve manifest provenance path for {provenance_key}: {exc}")
    provenance_sha = raw_entry.get("sha256")
    if not isinstance(provenance_sha, str) or not provenance_sha:
        issues.append(f"provenance.{provenance_key}.sha256 is missing for {audit_label}")
    elif not isinstance(audit_sha, str) or not audit_sha:
        issues.append(f"{audit_label} sha256 is missing")
    elif provenance_sha != audit_sha:
        issues.append(
            f"manifest provenance.{provenance_key}.sha256 does not match {audit_label} sha256: "
            f"provenance={provenance_sha} audit={audit_sha}"
        )


def _audit_input_provenance_entry(
    label: str,
    entry: dict[str, object],
    issues: list[str],
) -> None:
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        issues.append(f"provenance.{label}.path is missing")
        return
    path = Path(path_value)
    exists = path.is_file()
    if bool(entry.get("exists")) != exists:
        issues.append(f"provenance.{label}.exists mismatch for {path}")
    if not exists:
        return
    recorded_sha = entry.get("sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha:
        issues.append(f"provenance.{label}.sha256 is missing")
    else:
        actual_sha = _sha256_file(path)
        if actual_sha != recorded_sha:
            issues.append(f"provenance.{label}.sha256 mismatch for {path}: manifest={recorded_sha} actual={actual_sha}")
    recorded_size = entry.get("size_bytes")
    if not isinstance(recorded_size, int):
        issues.append(f"provenance.{label}.size_bytes is missing")
    elif path.stat().st_size != recorded_size:
        issues.append(f"provenance.{label}.size_bytes mismatch for {path}: manifest={recorded_size} actual={path.stat().st_size}")


def _audit_limbu_document(
    output_dir: Path,
    manifest: dict[str, object],
    stage: str,
    issues: list[str],
    warnings: list[str],
) -> Document | None:
    document_payload = _read_json_object(output_dir / "document.json", issues, label="document.json")
    if document_payload is None:
        return None
    try:
        document = Document.from_dict(document_payload)
    except Exception as exc:
        issues.append(f"document.json failed schema load: {type(exc).__name__}: {exc}")
        return None
    _audit_limbu_manifest_source_binding(manifest, document, issues, warnings)
    expected = manifest.get("document")
    if not isinstance(expected, dict):
        issues.append("limbu-output-manifest.json document must be an object")
        return document
    actual_counts = {
        "page_count": len(document.pages),
        "line_count": sum(len(page.text_lines) for page in document.pages),
        "table_count": len(document.tables),
        "figure_count": len(document.figures),
    }
    for key, actual in actual_counts.items():
        if expected.get(key) != actual:
            issues.append(f"document count mismatch for {key}: manifest={expected.get(key)!r} actual={actual}")
    if stage == "ocr" and not isinstance(document.metadata.get("limbu_pipeline"), dict):
        issues.append("document.json missing metadata.limbu_pipeline for OCR stage")
    if stage == "review_correction" and not isinstance(document.metadata.get("limbu_post_correction"), dict):
        issues.append("document.json missing metadata.limbu_post_correction for review_correction stage")
    if not document.pages:
        warnings.append("document.json has no pages")
    return document


def _audit_limbu_manifest_source_binding(
    manifest: dict[str, object],
    document: Document,
    issues: list[str],
    warnings: list[str],
) -> None:
    source_path_value = manifest.get("source_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        issues.append("limbu-output-manifest.json source_path is missing")
    else:
        source_path = Path(source_path_value)
        if source_path.is_file():
            recorded_sha = manifest.get("source_sha256")
            if not isinstance(recorded_sha, str) or not recorded_sha:
                issues.append("limbu-output-manifest.json source_sha256 is missing for source file")
            elif _sha256_file(source_path) != recorded_sha:
                issues.append(f"limbu-output-manifest.json source_sha256 mismatch for {source_path}")
        elif source_path.exists():
            if manifest.get("source_sha256") is not None:
                issues.append("limbu-output-manifest.json source_sha256 must be null for non-file source_path")
        else:
            warnings.append(f"limbu-output-manifest.json source_path is not available for hash replay: {source_path}")
    document_source_path = document.source_path
    manifest_document_source = manifest.get("document_source_path")
    if document_source_path:
        if manifest_document_source != document_source_path:
            issues.append(
                "limbu-output-manifest.json document_source_path does not match document.json source_path: "
                f"manifest={manifest_document_source!r} document={document_source_path!r}"
            )
        if isinstance(source_path_value, str) and source_path_value:
            try:
                if Path(source_path_value).resolve() != Path(document_source_path).resolve():
                    issues.append(
                        "limbu-output-manifest.json source_path does not match document.json source_path: "
                        f"manifest={source_path_value!r} document={document_source_path!r}"
                    )
            except OSError as exc:
                warnings.append(f"could not resolve manifest/document source paths for comparison: {exc}")
    elif manifest_document_source not in (None, ""):
        issues.append(f"limbu-output-manifest.json document_source_path should be empty when document.json source_path is empty: {manifest_document_source!r}")


def _audit_limbu_ocr_stage(
    output_dir: Path,
    document: Document | None,
    manifest: dict[str, object],
    issues: list[str],
    warnings: list[str],
    *,
    capture_quality_policy: dict[str, object],
) -> None:
    audit_payload = _read_json_object(output_dir / "limbu-pipeline-audit.json", issues, label="limbu-pipeline-audit.json")
    line_rows = _read_jsonl_objects(output_dir / "limbu-line-audit.jsonl", issues, label="limbu-line-audit.jsonl")
    review_rows = _read_tsv_rows(output_dir / "limbu-review-queue.tsv", issues, label="limbu-review-queue.tsv")
    post_correction_audit = _read_json_object(
        output_dir / "limbu-post-correction-audit.json",
        issues,
        label="limbu-post-correction-audit.json",
    )
    post_correction_rows = _read_jsonl_objects(
        output_dir / "limbu-post-correction-lines.jsonl",
        issues,
        label="limbu-post-correction-lines.jsonl",
    )
    crop_summary = _read_json_object(output_dir / "line-crops" / "summary.json", issues, label="line-crops/summary.json")
    crop_rows = _read_jsonl_objects(output_dir / "line-crops" / "manifest.jsonl", issues, label="line-crops/manifest.jsonl")
    dashboard_path = output_dir / "limbu-review-dashboard.html"
    dashboard_text = ""
    if dashboard_path.is_file():
        dashboard_text = dashboard_path.read_text(encoding="utf-8")
    else:
        issues.append(f"limbu-review-dashboard.html is missing: {dashboard_path}")
    if audit_payload is None:
        return
    expected_line_count = audit_payload.get("line_count")
    if isinstance(expected_line_count, int) and len(line_rows) != expected_line_count:
        issues.append(f"limbu-line-audit.jsonl row count mismatch: audit={expected_line_count} actual={len(line_rows)}")
    if isinstance(expected_line_count, int) and len(post_correction_rows) != expected_line_count:
        issues.append(
            f"limbu-post-correction-lines.jsonl row count mismatch: audit={expected_line_count} actual={len(post_correction_rows)}"
        )
    expected_review_count = audit_payload.get("review_line_count")
    if isinstance(expected_review_count, int) and len(review_rows) != expected_review_count:
        issues.append(f"limbu-review-queue.tsv row count mismatch: audit={expected_review_count} actual={len(review_rows)}")
    if isinstance(expected_review_count, int) and dashboard_text:
        dashboard_count = dashboard_text.count('class="review-card"')
        if dashboard_count != expected_review_count:
            issues.append(f"limbu-review-dashboard.html card count mismatch: audit={expected_review_count} actual={dashboard_count}")
    if document is not None:
        actual_line_count = sum(len(page.text_lines) for page in document.pages)
        if isinstance(expected_line_count, int) and actual_line_count != expected_line_count:
            issues.append(f"document line count mismatch: audit={expected_line_count} actual={actual_line_count}")
        _audit_limbu_line_audit_replay(document, audit_payload, line_rows, issues)
    if audit_payload.get("claim_scope") is None:
        warnings.append("limbu-pipeline-audit.json missing claim_scope")
    capture_prep = audit_payload.get("capture_prep")
    if isinstance(capture_prep, dict):
        _audit_manifest_provenance_binding(
            manifest,
            provenance_key="capture_prep_metadata",
            audit_label="limbu-pipeline-audit.json capture_prep",
            audit_path=capture_prep.get("path"),
            audit_sha=capture_prep.get("sha256"),
            issues=issues,
        )
        capture_audit = capture_prep.get("audit")
        if not isinstance(capture_audit, dict):
            issues.append("limbu-pipeline-audit.json capture_prep missing audit summary")
        elif capture_audit.get("passed") is not True:
            issues.append(f"capture prep audit did not pass: {capture_audit.get('issues')!r}")
        capture_audit_path = output_dir / "capture-prep-audit" / "limbu-capture-audit.json"
        if not capture_audit_path.is_file():
            issues.append(f"capture prep audit artifact is missing: {capture_audit_path}")
        else:
            capture_audit_payload = _read_json_object(capture_audit_path, issues, label="capture-prep-audit/limbu-capture-audit.json")
            if capture_audit_payload is not None and capture_audit_payload.get("passed") is not True:
                issues.append(f"capture-prep-audit/limbu-capture-audit.json did not pass: {capture_audit_payload.get('issues')!r}")
        _audit_limbu_capture_prep_replay(output_dir, capture_prep, issues, capture_quality_policy=capture_quality_policy)
    if post_correction_audit is not None:
        if post_correction_audit.get("line_count") != len(post_correction_rows):
            issues.append(
                "limbu-post-correction-lines.jsonl row count mismatch: "
                f"audit={post_correction_audit.get('line_count')!r} actual={len(post_correction_rows)}"
            )
        changed_count = sum(1 for row in post_correction_rows if row.get("changed"))
        if post_correction_audit.get("changed_count") != changed_count:
            issues.append(
                "limbu post-correction changed_count mismatch: "
                f"audit={post_correction_audit.get('changed_count')!r} actual={changed_count}"
            )
        profile_path_value = post_correction_audit.get("profile_path")
        profile_sha = post_correction_audit.get("profile_sha256")
        if profile_path_value:
            _audit_manifest_provenance_binding(
                manifest,
                provenance_key="post_correction_profile",
                audit_label="limbu-post-correction-audit.json",
                audit_path=profile_path_value,
                audit_sha=profile_sha,
                issues=issues,
            )
            profile_path = Path(str(profile_path_value))
            if not profile_path.is_file():
                issues.append(f"Limbu post-correction profile is missing: {profile_path}")
            elif not isinstance(profile_sha, str) or _sha256_file(profile_path) != profile_sha:
                issues.append(f"Limbu post-correction profile sha256 mismatch: {profile_path}")
            profile_admission = post_correction_audit.get("profile_admission")
            if not isinstance(profile_admission, dict):
                warnings.append("Limbu post-correction profile admission evidence is missing from audit")
            elif profile_admission.get("claim_ready") is not True:
                warnings.append("Limbu post-correction profile is not claim-ready; automatic corrections are experimental")
        elif _manifest_provenance_entry(manifest, "post_correction_profile") is not None:
            issues.append("OCR manifest provenance has post_correction_profile but limbu-post-correction-audit.json has no profile_path")
        if document is not None:
            _audit_limbu_post_correction_line_metadata_replay(
                document,
                post_correction_audit,
                post_correction_rows,
                issues,
            )
    required_review_columns = {"page_index", "line_index", "line_id", "review_status", "corrected_text"}
    if review_rows:
        missing = required_review_columns - set(review_rows[0])
        if missing:
            issues.append(f"limbu-review-queue.tsv missing columns: {sorted(missing)}")
        _audit_limbu_review_queue_alignment(line_rows, review_rows, issues)
        if dashboard_text:
            _audit_limbu_review_dashboard_alignment(output_dir, review_rows, crop_rows, dashboard_text, issues)
    if crop_summary is not None:
        if crop_summary.get("line_count") != len(crop_rows):
            issues.append(f"line crop manifest row count mismatch: summary={crop_summary.get('line_count')!r} actual={len(crop_rows)}")
        crop_count = sum(1 for row in crop_rows if row.get("crop_path"))
        if crop_summary.get("crop_count") != crop_count:
            issues.append(f"line crop count mismatch: summary={crop_summary.get('crop_count')!r} actual={crop_count}")
        for row in crop_rows:
            crop_path_value = row.get("crop_path")
            if not crop_path_value:
                continue
            crop_path = Path(str(crop_path_value))
            if not crop_path.is_file():
                issues.append(f"line crop file missing: {crop_path}")
                continue
            recorded_sha = row.get("crop_sha256")
            if not isinstance(recorded_sha, str) or not recorded_sha:
                issues.append(f"line crop row missing crop_sha256: {crop_path}")
            elif _sha256_file(crop_path) != recorded_sha:
                issues.append(f"line crop sha256 mismatch: {crop_path}")
    _audit_limbu_image_line_artifacts(output_dir, document, issues)


def _audit_limbu_image_line_artifacts(output_dir: Path, document: Document | None, issues: list[str]) -> None:
    run_path = output_dir / "image-line-ocr-run.json"
    crop_manifest_path = output_dir / "image-line-crops" / "manifest.jsonl"
    uses_image_line = _document_uses_image_line_ocr(document)
    if not uses_image_line and not run_path.exists() and not crop_manifest_path.exists():
        return
    run = _read_json_object(run_path, issues, label="image-line-ocr-run.json")
    crop_rows = _read_jsonl_objects(crop_manifest_path, issues, label="image-line-crops/manifest.jsonl")
    if run is None:
        if uses_image_line:
            issues.append("image-line OCR metadata is present but image-line-ocr-run.json is missing or invalid")
        return
    _audit_limbu_image_line_run_metadata(output_dir, document, run, issues)
    recorded_manifest_path = run.get("crop_manifest_path")
    if not isinstance(recorded_manifest_path, str) or not recorded_manifest_path:
        issues.append("image-line-ocr-run.json missing crop_manifest_path")
    else:
        try:
            if Path(recorded_manifest_path).resolve() != crop_manifest_path.resolve():
                issues.append(
                    "image-line-ocr-run.json crop_manifest_path does not match image-line-crops/manifest.jsonl: "
                    f"{recorded_manifest_path!r}"
                )
        except OSError as exc:
            issues.append(f"image-line-ocr-run.json crop_manifest_path cannot be resolved: {recorded_manifest_path}: {exc}")
    recorded_manifest_sha = run.get("crop_manifest_sha256")
    if not isinstance(recorded_manifest_sha, str) or not recorded_manifest_sha:
        issues.append("image-line-ocr-run.json missing crop_manifest_sha256")
    elif crop_manifest_path.is_file() and _sha256_file(crop_manifest_path) != recorded_manifest_sha:
        issues.append("image-line-crops/manifest.jsonl sha256 mismatch against image-line-ocr-run.json")
    detected_line_count = run.get("detected_line_count")
    kept_line_count = run.get("kept_line_count")
    removed_line_count = run.get("removed_line_count")
    if not all(isinstance(value, int) for value in (detected_line_count, kept_line_count, removed_line_count)):
        issues.append("image-line-ocr-run.json must record integer detected_line_count, kept_line_count, and removed_line_count")
        return
    kept_rows = sum(1 for row in crop_rows if row.get("kept") is True)
    removed_rows = sum(1 for row in crop_rows if row.get("kept") is False)
    if detected_line_count != len(crop_rows):
        issues.append(f"image-line crop manifest row count mismatch: run={detected_line_count!r} actual={len(crop_rows)}")
    if kept_line_count != kept_rows:
        issues.append(f"image-line kept row count mismatch: run={kept_line_count!r} actual={kept_rows}")
    if removed_line_count != removed_rows:
        issues.append(f"image-line removed row count mismatch: run={removed_line_count!r} actual={removed_rows}")
    if document is not None:
        document_line_count = sum(len(page.text_lines) for page in document.pages)
        if kept_line_count != document_line_count:
            issues.append(f"image-line kept line count mismatch against document: run={kept_line_count!r} document={document_line_count}")
    _audit_limbu_image_line_detector_source_pass_counts(run, crop_rows, issues)
    document_lines_by_id: dict[str, TextLine] = {}
    if document is not None:
        for page in document.pages:
            for line in page.text_lines:
                if line.line_id:
                    if line.line_id in document_lines_by_id:
                        issues.append(f"duplicate image-line document line_id: {line.line_id}")
                    document_lines_by_id[line.line_id] = line
    kept_manifest_line_ids: set[str] = set()
    for row in crop_rows:
        crop_path_value = row.get("crop_path")
        if not isinstance(crop_path_value, str) or not crop_path_value:
            issues.append(f"image-line crop manifest row missing crop_path: {row.get('detected_line_id')!r}")
            continue
        crop_path = Path(crop_path_value)
        if not crop_path.is_file():
            issues.append(f"image-line crop file missing: {crop_path}")
            continue
        recorded_sha = row.get("crop_sha256")
        if not isinstance(recorded_sha, str) or not recorded_sha:
            issues.append(f"image-line crop row missing crop_sha256: {crop_path}")
        elif _sha256_file(crop_path) != recorded_sha:
            issues.append(f"image-line crop sha256 mismatch: {crop_path}")
        if row.get("kept") is True:
            line_id = row.get("line_id")
            if not isinstance(line_id, str) or not line_id:
                issues.append(f"kept image-line crop manifest row missing line_id: {row.get('detected_line_id')!r}")
                continue
            kept_manifest_line_ids.add(line_id)
            if document is not None:
                line = document_lines_by_id.get(line_id)
                if line is None:
                    issues.append(f"kept image-line crop row references missing document line_id: {line_id}")
                    continue
                _audit_limbu_image_line_document_row(line_id, row, line, issues)
        elif row.get("line_id") not in (None, ""):
            issues.append(f"removed image-line crop row should not record a kept line_id: {row.get('detected_line_id')!r}")
    if document is not None:
        extra_document_ids = sorted(set(document_lines_by_id) - kept_manifest_line_ids)
        if extra_document_ids:
            issues.append(f"document image-line lines missing from crop manifest: {extra_document_ids}")


def _audit_limbu_image_line_detector_source_pass_counts(
    run: dict[str, object],
    crop_rows: list[dict[str, object]],
    issues: list[str],
) -> None:
    expected = {
        "detector_source_pass_counts": _limbu_detector_source_pass_counts(crop_rows),
        "kept_detector_source_pass_counts": _limbu_detector_source_pass_counts(crop_rows, kept=True),
        "removed_detector_source_pass_counts": _limbu_detector_source_pass_counts(crop_rows, kept=False),
    }
    for key, value in expected.items():
        actual = run.get(key)
        if actual is None:
            continue
        if not isinstance(actual, dict):
            issues.append(f"image-line-ocr-run.json {key} must be an object")
        elif actual != value:
            issues.append(f"image-line detector source pass count mismatch for {key}: run={actual!r} actual={value!r}")


def _limbu_detector_source_pass_counts(
    crop_rows: list[dict[str, object]],
    *,
    kept: bool | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in crop_rows:
        if kept is not None and row.get("kept") is not kept:
            continue
        source_pass = row.get("detector_source_pass")
        key = str(source_pass) if source_pass else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _audit_limbu_image_line_document_row(
    line_id: str,
    row: dict[str, object],
    line: TextLine,
    issues: list[str],
) -> None:
    metadata = line.metadata
    expected_pairs = {
        "original_detected_line_id": row.get("detected_line_id"),
        "crop_path": row.get("crop_path"),
        "crop_sha256": row.get("crop_sha256"),
        "crop_xyxy": row.get("crop_xyxy"),
        "detected_area": row.get("detected_area"),
        "detector_source_pass": row.get("detector_source_pass"),
        "reading_order": row.get("reading_order"),
        "requested_reading_order": row.get("requested_reading_order"),
    }
    for metadata_key, expected in expected_pairs.items():
        actual = metadata.get(metadata_key)
        if not _limbu_audit_values_equal(actual, expected):
            issues.append(
                "image-line document metadata mismatch for "
                f"{line_id}.{metadata_key}: document={actual!r} manifest={expected!r}"
            )
    detected_bbox = row.get("detected_bbox")
    if not _limbu_audit_values_equal(line.bbox.to_list(), detected_bbox):
        issues.append(f"image-line document bbox mismatch for {line_id}: document={line.bbox.to_list()!r} manifest={detected_bbox!r}")
    prediction_text = row.get("prediction_text")
    if not _limbu_audit_values_equal(line.text, prediction_text):
        issues.append(f"image-line document text mismatch for {line_id}: document={line.text!r} manifest={prediction_text!r}")
    confidence = row.get("confidence")
    if not _limbu_audit_values_equal(line.confidence, confidence):
        issues.append(f"image-line document confidence mismatch for {line_id}: document={line.confidence!r} manifest={confidence!r}")


def _audit_limbu_image_line_run_metadata(
    output_dir: Path,
    document: Document | None,
    run: dict[str, object],
    issues: list[str],
) -> None:
    source_path_value = run.get("source_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        issues.append("image-line-ocr-run.json missing source_path")
    else:
        source_path = Path(source_path_value)
        if not source_path.is_file():
            issues.append(f"image-line source_path is missing: {source_path}")
        else:
            source_sha = run.get("source_sha256")
            if not isinstance(source_sha, str) or not source_sha:
                issues.append("image-line-ocr-run.json missing source_sha256")
            elif _sha256_file(source_path) != source_sha:
                issues.append(f"image-line source_sha256 mismatch: {source_path}")
    model_config_value = run.get("model_config")
    if isinstance(model_config_value, str) and model_config_value:
        model_config_path = Path(model_config_value)
        if not model_config_path.is_file():
            issues.append(f"image-line model_config is missing: {model_config_path}")
    expected_metadata = {
        key: value
        for key, value in run.items()
        if key not in {"crop_manifest_path", "crop_manifest_sha256"}
    }
    if document is None:
        return
    document_metadata = document.metadata.get("image_line_ocr")
    if not isinstance(document_metadata, dict):
        issues.append("document.json metadata.image_line_ocr is missing for image-line output")
    else:
        _audit_limbu_image_line_metadata_object("document.metadata.image_line_ocr", document_metadata, expected_metadata, issues)
    if document.source_path and source_path_value and document.source_path != source_path_value:
        issues.append(
            "document.json source_path does not match image-line-ocr-run.json source_path: "
            f"document={document.source_path!r} run={source_path_value!r}"
        )
    for page in document.pages:
        page_metadata = page.metadata.get("image_line_ocr")
        if not isinstance(page_metadata, dict):
            issues.append(f"document page {page.page_index} metadata.image_line_ocr is missing for image-line output")
            continue
        _audit_limbu_image_line_metadata_object(
            f"document.pages[{page.page_index}].metadata.image_line_ocr",
            page_metadata,
            expected_metadata,
            issues,
        )


def _audit_limbu_image_line_metadata_object(
    label: str,
    metadata: dict[str, object],
    expected: dict[str, object],
    issues: list[str],
) -> None:
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if not _limbu_audit_values_equal(actual, expected_value):
            issues.append(f"{label} {key} mismatch: document={actual!r} run={expected_value!r}")


def _audit_limbu_capture_prep_replay(
    output_dir: Path,
    capture_prep: dict[str, object],
    issues: list[str],
    *,
    capture_quality_policy: dict[str, object],
) -> None:
    metadata_path_value = capture_prep.get("path")
    if not isinstance(metadata_path_value, str) or not metadata_path_value:
        issues.append("limbu-pipeline-audit.json capture_prep missing metadata path")
        return
    metadata_path = Path(metadata_path_value)
    if not metadata_path.is_file():
        issues.append(f"capture prep metadata path is not available for replay audit: {metadata_path}")
        return
    embedded_sha = capture_prep.get("sha256")
    if not isinstance(embedded_sha, str) or not embedded_sha:
        issues.append("limbu-pipeline-audit.json capture_prep missing metadata sha256")
    else:
        actual_sha = _sha256_file(metadata_path)
        if actual_sha != embedded_sha:
            issues.append(f"capture prep metadata sha256 mismatch: embedded={embedded_sha} actual={actual_sha}")
    replay = audit_limbu_capture(
        metadata_path,
        output_dir / "capture-prep-replay-audit",
        **capture_quality_policy,
    )
    if replay.get("passed") is not True:
        issues.append(f"capture prep replay audit did not pass: {replay.get('issues')!r}")


def _audit_limbu_line_audit_replay(
    document: Document,
    audit_payload: dict[str, object],
    line_rows: list[dict[str, object]],
    issues: list[str],
) -> None:
    try:
        low_threshold = _normalize_required_ratio(
            audit_payload.get("low_confidence_threshold"),
            "limbu-pipeline-audit.json low_confidence_threshold",
        )
        script_ratio_threshold = _normalize_required_ratio(
            audit_payload.get("script_ratio_threshold"),
            "limbu-pipeline-audit.json script_ratio_threshold",
        )
    except ParseError as exc:
        issues.append(str(exc))
        return
    try:
        script_thresholds = _normalize_limbu_script_confidence_thresholds(
            low_confidence_threshold=low_threshold,
            script_confidence_thresholds=_audit_script_confidence_thresholds(audit_payload),
        )
    except ParseError as exc:
        issues.append(f"limbu-pipeline-audit.json script_confidence_thresholds invalid: {exc}")
        return
    rows_by_key: dict[str, dict[str, object]] = {}
    for row in line_rows:
        key = _line_row_key(row)
        if key is None:
            issues.append("limbu-line-audit.jsonl row is missing line_id and page/line position")
            continue
        if key in rows_by_key:
            issues.append(f"duplicate limbu-line-audit.jsonl row key: {key}")
            continue
        rows_by_key[key] = row
    expected_keys: set[str] = set()
    recomputed_script_counts = {
        "limbu_sirijonga": 0,
        "devanagari_limbu": 0,
        "mixed_limbu_devanagari": 0,
        "other": 0,
    }
    recomputed_review_count = 0
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"line_id:{line.line_id}" if line.line_id else f"position:{page.page_index}:{line_index}"
            expected_keys.add(key)
            row = rows_by_key.get(key)
            if row is None:
                issues.append(f"limbu-line-audit.jsonl missing row for {key}")
                continue
            profile = _limbu_script_profile(line.text, script_ratio_threshold=script_ratio_threshold)
            script_class = str(profile["script_class"])
            recomputed_script_counts[script_class if script_class in recomputed_script_counts else "other"] += 1
            review_threshold = _limbu_review_confidence_threshold(
                script_class,
                low_confidence_threshold=low_threshold,
                script_confidence_thresholds=script_thresholds,
            )
            review_reasons = _limbu_review_reasons(
                line.text,
                line.confidence,
                script_class,
                low_confidence_threshold=review_threshold,
            )
            needs_review = bool(review_reasons)
            if needs_review:
                recomputed_review_count += 1
            _audit_limbu_line_metadata_replay(
                key,
                line,
                profile=profile,
                review_threshold=review_threshold,
                review_reasons=review_reasons,
                needs_review=needs_review,
                issues=issues,
            )
            _compare_limbu_line_audit_row(
                key,
                row,
                page_index=page.page_index,
                line_index=line_index,
                line=line,
                profile=profile,
                review_threshold=review_threshold,
                review_reasons=review_reasons,
                needs_review=needs_review,
                issues=issues,
            )
    extra_keys = sorted(set(rows_by_key) - expected_keys)
    if extra_keys:
        issues.append(f"limbu-line-audit.jsonl contains rows not present in document.json: {extra_keys}")
    if audit_payload.get("review_line_count") != recomputed_review_count:
        issues.append(
            "limbu-pipeline-audit.json review_line_count mismatch after replay: "
            f"audit={audit_payload.get('review_line_count')!r} recomputed={recomputed_review_count}"
        )
    if audit_payload.get("script_counts") != recomputed_script_counts:
        issues.append(
            "limbu-pipeline-audit.json script_counts mismatch after replay: "
            f"audit={audit_payload.get('script_counts')!r} recomputed={recomputed_script_counts}"
        )


def _audit_script_confidence_thresholds(audit_payload: dict[str, object]) -> dict[str, float] | None:
    raw = audit_payload.get("script_confidence_thresholds")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ParseError("script_confidence_thresholds must be an object")
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ParseError(f"{key} confidence threshold must be numeric, got {value!r}")
        thresholds[str(key)] = float(value)
    return thresholds


def _compare_limbu_line_audit_row(
    key: str,
    row: dict[str, object],
    *,
    page_index: int,
    line_index: int,
    line: TextLine,
    profile: dict[str, object],
    review_threshold: float,
    review_reasons: list[str],
    needs_review: bool,
    issues: list[str],
) -> None:
    expected_scalars: dict[str, object] = {
        "page_index": page_index,
        "line_index": line_index,
        "line_id": line.line_id or f"p{page_index}-l{line_index}",
        "text": line.text,
        "confidence": line.confidence,
        "bbox": line.bbox.to_list(),
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "review_confidence_threshold": review_threshold,
        **profile,
    }
    for field, expected in expected_scalars.items():
        actual = row.get(field)
        if not _limbu_audit_values_equal(actual, expected):
            issues.append(f"limbu-line-audit.jsonl {field} mismatch for {key}: audit={actual!r} recomputed={expected!r}")


def _audit_limbu_line_metadata_replay(
    key: str,
    line: TextLine,
    *,
    profile: dict[str, object],
    review_threshold: float,
    review_reasons: list[str],
    needs_review: bool,
    issues: list[str],
) -> None:
    limbu_metadata = line.metadata.get("limbu_pipeline")
    if not isinstance(limbu_metadata, dict):
        issues.append(f"document.json line metadata.limbu_pipeline is missing for {key}")
        return
    expected = {
        **profile,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "review_confidence_threshold": review_threshold,
        "review_confidence_threshold_source": "script_effective",
    }
    for field, expected_value in expected.items():
        actual = limbu_metadata.get(field)
        if not _limbu_audit_values_equal(actual, expected_value):
            issues.append(
                f"document.json line metadata.limbu_pipeline.{field} mismatch for {key}: "
                f"metadata={actual!r} recomputed={expected_value!r}"
            )


def _audit_limbu_post_correction_line_metadata_replay(
    document: Document,
    post_correction_audit: dict[str, object],
    post_correction_rows: list[dict[str, object]],
    issues: list[str],
) -> None:
    document_metadata = _document_limbu_line_metadata_by_limbu_key(document)
    rows_by_key: dict[str, dict[str, object]] = {}
    for row in post_correction_rows:
        key = _line_row_key(row)
        if key is None:
            issues.append("limbu-post-correction-lines.jsonl row is missing line_id and page/line position")
            continue
        if key in rows_by_key:
            issues.append(f"duplicate limbu-post-correction-lines.jsonl row key: {key}")
            continue
        rows_by_key[key] = row
    for key, row in sorted(rows_by_key.items()):
        limbu_metadata = document_metadata.get(key)
        if not isinstance(limbu_metadata, dict):
            issues.append(f"document.json line metadata.limbu_pipeline is missing for post-correction row {key}")
            continue
        expected_metadata = _line_post_correction_metadata(row, post_correction_audit)
        actual_metadata = limbu_metadata.get("post_correction")
        if not _limbu_audit_values_equal(actual_metadata, expected_metadata):
            issues.append(
                f"document.json line metadata.limbu_pipeline.post_correction mismatch for {key}: "
                f"metadata={actual_metadata!r} post_correction_row={expected_metadata!r}"
            )
        corrected_text = row.get("corrected_text")
        line_text = _document_line_text_by_limbu_key(document).get(key)
        if isinstance(corrected_text, str) and line_text != corrected_text:
            issues.append(
                f"limbu-post-correction-lines.jsonl corrected_text mismatch for {key}: "
                f"document={line_text!r} post_correction_row={corrected_text!r}"
            )


def _document_limbu_line_metadata_by_limbu_key(document: Document) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"line_id:{line.line_id}" if line.line_id else f"position:{page.page_index}:{line_index}"
            metadata = line.metadata.get("limbu_pipeline")
            if isinstance(metadata, dict):
                rows[key] = metadata
    return rows


def _document_line_text_by_limbu_key(document: Document) -> dict[str, str]:
    rows: dict[str, str] = {}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"line_id:{line.line_id}" if line.line_id else f"position:{page.page_index}:{line_index}"
            rows[key] = line.text
    return rows


def _limbu_audit_values_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, int | float) and not isinstance(actual, bool) and abs(float(actual) - expected) <= 1e-12
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_limbu_audit_values_equal(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected, strict=True))
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(_limbu_audit_values_equal(actual[key], expected[key]) for key in expected)
    return actual == expected


def _audit_limbu_review_queue_alignment(
    line_rows: list[dict[str, object]],
    review_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    expected_rows = [row for row in line_rows if row.get("needs_review") is True]
    expected_by_id: dict[str, dict[str, object]] = {}
    for row in expected_rows:
        line_id = str(row.get("line_id") or "")
        if not line_id:
            issues.append("limbu-line-audit.jsonl needs_review row is missing line_id")
            continue
        if line_id in expected_by_id:
            issues.append(f"duplicate needs_review line_id in limbu-line-audit.jsonl: {line_id}")
            continue
        expected_by_id[line_id] = row
    actual_by_id: dict[str, dict[str, str]] = {}
    for row in review_rows:
        line_id = str(row.get("line_id") or "")
        if not line_id:
            issues.append("limbu-review-queue.tsv row is missing line_id")
            continue
        if line_id in actual_by_id:
            issues.append(f"duplicate line_id in limbu-review-queue.tsv: {line_id}")
            continue
        actual_by_id[line_id] = row
    missing_ids = sorted(set(expected_by_id) - set(actual_by_id))
    extra_ids = sorted(set(actual_by_id) - set(expected_by_id))
    if missing_ids:
        issues.append(f"limbu-review-queue.tsv missing needs_review line_ids: {missing_ids}")
    if extra_ids:
        issues.append(f"limbu-review-queue.tsv contains rows not marked needs_review: {extra_ids}")
    for line_id in sorted(set(expected_by_id) & set(actual_by_id)):
        expected = expected_by_id[line_id]
        actual = actual_by_id[line_id]
        comparisons = {
            "page_index": str(expected.get("page_index")),
            "line_index": str(expected.get("line_index")),
            "script_class": str(expected.get("script_class")),
            "review_confidence_threshold": str(expected.get("review_confidence_threshold")),
            "text": str(expected.get("text")),
            "review_reasons": ",".join(str(item) for item in expected.get("review_reasons", []) if item is not None)
            if isinstance(expected.get("review_reasons"), list)
            else str(expected.get("review_reasons") or ""),
        }
        expected_confidence = expected.get("confidence")
        actual_confidence = actual.get("confidence")
        if not _review_queue_confidence_matches(actual_confidence, expected_confidence):
            issues.append(
                f"limbu-review-queue.tsv confidence mismatch for line_id {line_id}: "
                f"queue={actual_confidence!r} line_audit={expected_confidence!r}"
            )
        for field, expected_value in comparisons.items():
            actual_value = str(actual.get(field) or "")
            if actual_value != expected_value:
                issues.append(
                    f"limbu-review-queue.tsv {field} mismatch for line_id {line_id}: "
                    f"queue={actual_value!r} line_audit={expected_value!r}"
                )


def _audit_limbu_review_dashboard_alignment(
    output_dir: Path,
    review_rows: list[dict[str, str]],
    crop_rows: list[dict[str, object]],
    dashboard_text: str,
    issues: list[str],
) -> None:
    crop_by_key: dict[str, dict[str, object]] = {}
    for row in crop_rows:
        key = _line_row_key(row)
        if key:
            crop_by_key[key] = row
    for row in review_rows:
        line_id = str(row.get("line_id") or "")
        if not line_id:
            continue
        card = _dashboard_card_html_for_line_id(dashboard_text, line_id)
        if card is None:
            issues.append(f"limbu-review-dashboard.html missing review card for line_id {line_id!r}")
            continue
        _audit_limbu_review_dashboard_card(output_dir, row, crop_by_key.get(_line_row_key(row) or ""), card, issues)


def _dashboard_card_html_for_line_id(dashboard_text: str, line_id: str) -> str | None:
    escaped_line_id = html.escape(line_id, quote=True)
    marker = f'data-line-id="{escaped_line_id}"'
    marker_index = dashboard_text.find(marker)
    if marker_index < 0:
        return None
    start = dashboard_text.rfind("<article", 0, marker_index)
    end = dashboard_text.find("</article>", marker_index)
    if start < 0 or end < 0:
        return None
    return dashboard_text[start : end + len("</article>")]


def _audit_limbu_review_dashboard_card(
    output_dir: Path,
    review_row: dict[str, str],
    crop_row: dict[str, object] | None,
    card: str,
    issues: list[str],
) -> None:
    line_id = str(review_row.get("line_id") or "")
    expected_fragments = {
        "page_index": f"page {review_row.get('page_index') or ''}",
        "line_index": f"line {review_row.get('line_index') or ''}",
        "script_class": str(review_row.get("script_class") or ""),
        "confidence": f"confidence {review_row.get('confidence') or ''}",
        "review_confidence_threshold": f"threshold {review_row.get('review_confidence_threshold') or ''}",
        "review_reasons": str(review_row.get("review_reasons") or ""),
        "text": str(review_row.get("text") or ""),
        "tsv_row": _limbu_review_dashboard_tsv_row(review_row),
    }
    for field, expected in expected_fragments.items():
        if html.escape(expected) not in card:
            issues.append(
                f"limbu-review-dashboard.html {field} mismatch for line_id {line_id}: "
                f"expected fragment {expected!r}"
            )
    expected_crop_fragment = _limbu_review_dashboard_expected_crop_fragment(crop_row, output_dir=output_dir)
    if expected_crop_fragment and expected_crop_fragment not in card:
        issues.append(
            f"limbu-review-dashboard.html crop link mismatch for line_id {line_id}: "
            f"expected fragment {expected_crop_fragment!r}"
        )


def _limbu_review_dashboard_tsv_row(review_row: dict[str, str]) -> str:
    text = str(review_row.get("text") or "")
    return "\t".join(
        [
            str(review_row.get("page_index") or ""),
            str(review_row.get("line_index") or ""),
            str(review_row.get("line_id") or ""),
            str(review_row.get("script_class") or ""),
            str(review_row.get("confidence") or ""),
            str(review_row.get("review_confidence_threshold") or ""),
            str(review_row.get("review_reasons") or ""),
            text,
            "accepted",
            text,
            "",
            "",
        ]
    )


def _limbu_review_dashboard_expected_crop_fragment(crop_row: dict[str, object] | None, *, output_dir: Path) -> str | None:
    if crop_row is None:
        return html.escape("No matching crop manifest row.")
    crop_path = crop_row.get("crop_path")
    if not crop_path:
        return html.escape(str(crop_row.get("warning") or "No crop was emitted for this line."))
    href = _dashboard_artifact_href(str(crop_path), output_dir=output_dir)
    return f'src="{html.escape(href, quote=True)}"'


def _review_queue_confidence_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return str(actual or "") == ""
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return str(actual or "") == str(expected)
    try:
        return abs(float(str(actual)) - float(expected)) <= 1e-12
    except ValueError:
        return False


def _audit_limbu_review_correction_stage(
    output_dir: Path,
    document: Document | None,
    manifest: dict[str, object],
    issues: list[str],
    warnings: list[str],
) -> None:
    audit_payload = _read_json_object(output_dir / "limbu-correction-audit.json", issues, label="limbu-correction-audit.json")
    pair_rows = _read_jsonl_objects(output_dir / "limbu-correction-pairs.jsonl", issues, label="limbu-correction-pairs.jsonl")
    if audit_payload is None:
        return
    applied_count = audit_payload.get("applied_count")
    if isinstance(applied_count, int) and len(pair_rows) != applied_count:
        issues.append(f"limbu-correction-pairs.jsonl row count mismatch: audit={applied_count} actual={len(pair_rows)}")
    changed_count = audit_payload.get("changed_count")
    if isinstance(changed_count, int) and isinstance(applied_count, int) and changed_count > applied_count:
        issues.append(f"changed_count cannot exceed applied_count: changed={changed_count} applied={applied_count}")
    if audit_payload.get("claim_scope") is None:
        warnings.append("limbu-correction-audit.json missing claim_scope")
    _audit_limbu_review_correction_manifest_provenance(manifest, audit_payload, issues)
    if document is None:
        return
    corrected_lines = [
        line
        for page in document.pages
        for line in page.text_lines
        if isinstance(line.metadata.get("limbu_pipeline"), dict)
        and isinstance(line.metadata["limbu_pipeline"].get("post_correction"), dict)
        and line.metadata["limbu_pipeline"]["post_correction"].get("status") == "human_review_applied"
    ]
    if isinstance(applied_count, int) and len(corrected_lines) != applied_count:
        issues.append(f"human_review_applied line count mismatch: audit={applied_count} actual={len(corrected_lines)}")
    _audit_limbu_review_correction_replay(document, audit_payload, pair_rows, issues)


def _audit_limbu_review_correction_manifest_provenance(
    manifest: dict[str, object],
    audit_payload: dict[str, object],
    issues: list[str],
) -> None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return
    expected = {
        "input_document": ("document_path", "document_sha256"),
        "review_queue": ("review_queue_path", "review_queue_sha256"),
    }
    for provenance_key, (audit_path_key, audit_sha_key) in expected.items():
        raw_entry = provenance.get(provenance_key)
        if not isinstance(raw_entry, dict):
            issues.append(f"review_correction manifest provenance lacks {provenance_key}")
            continue
        provenance_path_value = raw_entry.get("path")
        audit_path_value = audit_payload.get(audit_path_key)
        if not isinstance(provenance_path_value, str) or not provenance_path_value:
            issues.append(f"provenance.{provenance_key}.path is missing for review_correction stage")
        elif not isinstance(audit_path_value, str) or not audit_path_value:
            issues.append(f"limbu-correction-audit.json {audit_path_key} is missing")
        else:
            try:
                if Path(provenance_path_value).resolve() != Path(audit_path_value).resolve():
                    issues.append(
                        f"review_correction manifest provenance.{provenance_key}.path does not match "
                        f"limbu-correction-audit.json {audit_path_key}: "
                        f"provenance={provenance_path_value!r} audit={audit_path_value!r}"
                    )
            except OSError as exc:
                issues.append(f"could not resolve review_correction provenance path for {provenance_key}: {exc}")
        provenance_sha = raw_entry.get("sha256")
        audit_sha = audit_payload.get(audit_sha_key)
        if not isinstance(provenance_sha, str) or not provenance_sha:
            issues.append(f"provenance.{provenance_key}.sha256 is missing for review_correction stage")
        elif not isinstance(audit_sha, str) or not audit_sha:
            issues.append(f"limbu-correction-audit.json {audit_sha_key} is missing")
        elif provenance_sha != audit_sha:
            issues.append(
                f"review_correction manifest provenance.{provenance_key}.sha256 does not match "
                f"limbu-correction-audit.json {audit_sha_key}: provenance={provenance_sha} audit={audit_sha}"
            )


def _audit_limbu_output_policy(
    output_dir: Path,
    *,
    stage: str,
    document: Document | None,
    require_capture_prep: bool,
    capture_quality_policy: dict[str, object],
    require_no_pending_review: bool,
    require_reviewer_for_corrections: bool,
    require_no_dropped_image_lines: bool,
    min_line_count: int | None,
    min_average_line_confidence: float | None,
    min_quality_score: float | None,
    required_scripts: tuple[str, ...],
    required_script_counts: dict[str, int],
    issues: list[str],
) -> None:
    require_capture_quality = any(
        key.startswith("min_") and value is not None
        for key, value in capture_quality_policy.items()
    )
    require_output_quality = (
        min_line_count is not None
        or min_average_line_confidence is not None
        or min_quality_score is not None
    )
    if (
        not require_capture_prep
        and not require_capture_quality
        and not require_no_pending_review
        and not require_reviewer_for_corrections
        and not require_no_dropped_image_lines
        and not require_output_quality
        and not required_scripts
        and not required_script_counts
    ):
        return
    limbu_metadata: dict[str, object] = {}
    if document is not None and isinstance(document.metadata.get("limbu_pipeline"), dict):
        limbu_metadata = document.metadata["limbu_pipeline"]
    if require_capture_prep:
        capture_prep = limbu_metadata.get("capture_prep")
        if not isinstance(capture_prep, dict):
            issues.append("strict Limbu output policy requires capture_prep metadata")
        else:
            capture_audit = capture_prep.get("audit")
            if not isinstance(capture_audit, dict) or capture_audit.get("passed") is not True:
                issues.append("strict Limbu output policy requires passing capture_prep audit")
    if require_capture_quality and not isinstance(limbu_metadata.get("capture_prep"), dict):
        issues.append("strict Limbu output policy requires capture_prep metadata for capture quality policy")
    if require_no_pending_review:
        if stage == "review_correction":
            correction_audit = _read_json_object(
                output_dir / "limbu-correction-audit.json",
                issues,
                label="limbu-correction-audit.json",
            )
            skipped_count = correction_audit.get("skipped_count") if isinstance(correction_audit, dict) else None
            if skipped_count != 0:
                issues.append(f"strict Limbu output policy requires no skipped review rows: skipped_count={skipped_count!r}")
        else:
            review_line_count = limbu_metadata.get("review_line_count")
            if review_line_count != 0:
                issues.append(f"strict Limbu output policy requires no pending review lines: review_line_count={review_line_count!r}")
    if require_reviewer_for_corrections:
        _audit_limbu_correction_reviewer_policy(output_dir, stage=stage, issues=issues)
    if require_no_dropped_image_lines:
        _audit_limbu_image_line_drop_policy(output_dir, document, issues)
    if require_output_quality:
        _audit_limbu_output_quality_policy(
            document,
            min_line_count=min_line_count,
            min_average_line_confidence=min_average_line_confidence,
            min_quality_score=min_quality_score,
            issues=issues,
        )
    if required_scripts or required_script_counts:
        script_counts = limbu_metadata.get("script_counts")
        if not isinstance(script_counts, dict):
            issues.append("strict Limbu output policy requires script_counts metadata")
            return
        for script in required_scripts:
            count = script_counts.get(script)
            if not isinstance(count, int) or count <= 0:
                issues.append(f"strict Limbu output policy requires observed script {script!r}: count={count!r}")
        for script, required_count in required_script_counts.items():
            count = script_counts.get(script)
            if not isinstance(count, int) or count < required_count:
                issues.append(
                    f"strict Limbu output policy requires at least {required_count} lines for script {script!r}: "
                    f"count={count!r}"
            )


def _audit_limbu_correction_reviewer_policy(output_dir: Path, *, stage: str, issues: list[str]) -> None:
    if stage != "review_correction":
        return
    correction_audit = _read_json_object(
        output_dir / "limbu-correction-audit.json",
        issues,
        label="limbu-correction-audit.json",
    )
    if correction_audit is None:
        return
    review_queue_path = Path(str(correction_audit.get("review_queue_path") or ""))
    if not review_queue_path.is_file():
        issues.append(f"strict Limbu output policy requires source review queue: {review_queue_path}")
        return
    recorded_review_sha = correction_audit.get("review_queue_sha256")
    if not isinstance(recorded_review_sha, str) or _sha256_file(review_queue_path) != recorded_review_sha:
        issues.append(f"strict Limbu output policy review_queue_sha256 mismatch for {review_queue_path}")
        return
    accepted_statuses = {
        str(item).strip().lower()
        for item in correction_audit.get("accepted_statuses", [])
        if str(item).strip()
    }
    if not accepted_statuses:
        issues.append("strict Limbu output policy requires accepted_statuses in limbu-correction-audit.json")
        return
    try:
        review_rows = _read_limbu_review_rows(review_queue_path)
    except Exception as exc:
        issues.append(f"strict Limbu output policy could not read review queue: {type(exc).__name__}: {exc}")
        return
    missing_reviewers: list[str] = []
    for row_number, row in enumerate(review_rows, start=2):
        status = str(row.get("review_status") or "").strip().lower()
        if status not in accepted_statuses:
            continue
        reviewer = str(row.get("reviewer") or "").strip()
        if not reviewer:
            line_id = row.get("line_id") or f"row-{row_number}"
            missing_reviewers.append(str(line_id))
    if missing_reviewers:
        issues.append(
            "strict Limbu output policy requires reviewer for accepted correction rows: "
            f"missing_reviewers={missing_reviewers}"
        )


def _audit_limbu_output_quality_policy(
    document: Document | None,
    *,
    min_line_count: int | None,
    min_average_line_confidence: float | None,
    min_quality_score: float | None,
    issues: list[str],
) -> None:
    if document is None:
        issues.append("strict Limbu output policy requires document.json for output quality policy")
        return
    actual_line_count = sum(len(page.text_lines) for page in document.pages)
    if min_line_count is not None and actual_line_count < min_line_count:
        issues.append(
            f"strict Limbu output policy requires at least {min_line_count} OCR lines: "
            f"line_count={actual_line_count}"
        )
    quality = document.metadata.get("quality")
    if not isinstance(quality, dict):
        if min_average_line_confidence is not None or min_quality_score is not None:
            issues.append("strict Limbu output policy requires document.metadata.quality for output quality policy")
        return
    metrics = quality.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    if min_average_line_confidence is not None:
        average_confidence = metrics.get("average_line_confidence")
        if not isinstance(average_confidence, int | float) or not math.isfinite(float(average_confidence)):
            issues.append("strict Limbu output policy requires finite quality.metrics.average_line_confidence")
        elif float(average_confidence) < min_average_line_confidence:
            issues.append(
                "strict Limbu output policy requires average line confidence "
                f">= {min_average_line_confidence}: average_line_confidence={float(average_confidence)}"
            )
    if min_quality_score is not None:
        quality_score = quality.get("quality_score")
        if not isinstance(quality_score, int | float) or not math.isfinite(float(quality_score)):
            issues.append("strict Limbu output policy requires finite quality.quality_score")
        elif float(quality_score) < min_quality_score:
            issues.append(
                f"strict Limbu output policy requires quality score >= {min_quality_score}: "
                f"quality_score={float(quality_score)}"
            )


def _audit_limbu_image_line_drop_policy(output_dir: Path, document: Document | None, issues: list[str]) -> None:
    run_path = output_dir / "image-line-ocr-run.json"
    uses_image_line = _document_uses_image_line_ocr(document)
    if not uses_image_line and not run_path.exists():
        return
    run = _read_json_object(run_path, issues, label="image-line-ocr-run.json")
    if run is None:
        if uses_image_line:
            issues.append("strict Limbu output policy requires image-line OCR run metadata")
        return
    removed_line_count = run.get("removed_line_count")
    if removed_line_count != 0:
        issues.append(
            "strict Limbu output policy requires no dropped image-line OCR rows: "
            f"removed_line_count={removed_line_count!r}"
        )
    detected_line_count = run.get("detected_line_count")
    kept_line_count = run.get("kept_line_count")
    if not isinstance(detected_line_count, int) or not isinstance(kept_line_count, int):
        issues.append("image-line-ocr-run.json must record integer detected_line_count and kept_line_count")
    elif isinstance(removed_line_count, int) and detected_line_count != kept_line_count + removed_line_count:
        issues.append(
            "image-line-ocr-run.json line counts do not balance: "
            f"detected={detected_line_count!r} kept={kept_line_count!r} removed={removed_line_count!r}"
        )


def _document_uses_image_line_ocr(document: Document | None) -> bool:
    if document is None:
        return False
    if document.metadata.get("ocr_output_kind") == "image_line_detector":
        return True
    if isinstance(document.metadata.get("image_line_ocr"), dict):
        return True
    for page in document.pages:
        if page.metadata.get("ocr_output_kind") == "image_line_detector":
            return True
        if isinstance(page.metadata.get("image_line_ocr"), dict):
            return True
    return any(
        line.metadata.get("image_line_detector") == "morph_line_detector_v1"
        or isinstance(line.metadata.get("image_line_ocr"), dict)
        for page in document.pages
        for line in page.text_lines
    )


def _audit_limbu_review_correction_replay(
    document: Document,
    audit_payload: dict[str, object],
    pair_rows: list[dict[str, object]],
    issues: list[str],
) -> None:
    source_document_path = Path(str(audit_payload.get("document_path") or ""))
    review_queue_path = Path(str(audit_payload.get("review_queue_path") or ""))
    if not source_document_path.is_file():
        issues.append(f"limbu-correction-audit.json source document is missing: {source_document_path}")
        return
    if not review_queue_path.is_file():
        issues.append(f"limbu-correction-audit.json review queue is missing: {review_queue_path}")
        return
    recorded_doc_sha = audit_payload.get("document_sha256")
    if not isinstance(recorded_doc_sha, str) or _sha256_file(source_document_path) != recorded_doc_sha:
        issues.append(f"limbu-correction-audit.json document_sha256 mismatch for {source_document_path}")
        return
    recorded_review_sha = audit_payload.get("review_queue_sha256")
    if not isinstance(recorded_review_sha, str) or _sha256_file(review_queue_path) != recorded_review_sha:
        issues.append(f"limbu-correction-audit.json review_queue_sha256 mismatch for {review_queue_path}")
        return
    try:
        source_payload = json.loads(source_document_path.read_text(encoding="utf-8"))
        if not isinstance(source_payload, dict):
            raise ParseError("source document must contain a JSON object")
        source_document = Document.from_dict(source_payload)
        review_rows = _read_limbu_review_rows(review_queue_path)
        accepted_statuses = tuple(str(item) for item in audit_payload.get("accepted_statuses", []) if str(item).strip())
        expected = _replay_limbu_review_corrections(
            source_document,
            review_rows,
            accepted_statuses=accepted_statuses,
            review_path=review_queue_path,
        )
    except Exception as exc:
        issues.append(f"could not replay Limbu review corrections: {type(exc).__name__}: {exc}")
        return
    for field in ("review_rows", "applied_count", "changed_count", "skipped_count", "script_ratio_threshold"):
        if audit_payload.get(field) != expected[field]:
            issues.append(
                f"limbu-correction-audit.json {field} mismatch after replay: "
                f"audit={audit_payload.get(field)!r} replay={expected[field]!r}"
            )
    actual_lines = _document_text_by_key(document)
    expected_lines = _document_text_by_key(expected["document"])
    if actual_lines != expected_lines:
        issues.append("document.json text does not match replayed Limbu review corrections")
    _audit_limbu_review_correction_line_metadata_replay(document, expected["document"], issues)
    expected_pairs = _expected_limbu_correction_pair_rows(expected["applied_rows"])
    if len(pair_rows) != len(expected_pairs):
        issues.append(f"limbu-correction-pairs.jsonl row count mismatch after replay: actual={len(pair_rows)} replay={len(expected_pairs)}")
    for index, expected_pair in enumerate(expected_pairs):
        if index >= len(pair_rows):
            break
        actual_pair = pair_rows[index]
        if actual_pair != expected_pair:
            issues.append(f"limbu-correction-pairs.jsonl row {index + 1} mismatch after replay: actual={actual_pair!r} replay={expected_pair!r}")


def _audit_limbu_review_correction_line_metadata_replay(
    document: Document,
    expected_document: Document,
    issues: list[str],
) -> None:
    actual_by_key = _document_limbu_line_metadata_by_key(document)
    expected_by_key = _document_limbu_line_metadata_by_key(expected_document)
    for key, expected_metadata in sorted(expected_by_key.items()):
        expected_post = expected_metadata.get("post_correction")
        if not isinstance(expected_post, dict) or expected_post.get("status") != "human_review_applied":
            continue
        actual_metadata = actual_by_key.get(key)
        if not isinstance(actual_metadata, dict):
            issues.append(f"document.json line metadata.limbu_pipeline missing after review replay for {key}")
            continue
        comparisons = {
            "needs_review": expected_metadata.get("needs_review"),
            "review_resolved": expected_metadata.get("review_resolved"),
            "original_needs_review": expected_metadata.get("original_needs_review"),
            "post_correction": expected_post,
        }
        for field, expected_value in comparisons.items():
            actual = actual_metadata.get(field)
            if not _limbu_audit_values_equal(actual, expected_value):
                issues.append(
                    f"document.json line metadata.limbu_pipeline.{field} mismatch after review replay for {key}: "
                    f"metadata={actual!r} replay={expected_value!r}"
                )


def _document_limbu_line_metadata_by_key(document: Document) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"page:{page.page_index}:line:{line_index}:line_id:{line.line_id or ''}"
            metadata = line.metadata.get("limbu_pipeline")
            if isinstance(metadata, dict):
                rows[key] = metadata
    return rows


def _replay_limbu_review_corrections(
    source_document: Document,
    review_rows: list[dict[str, str]],
    *,
    accepted_statuses: tuple[str, ...],
    review_path: Path,
) -> dict[str, object]:
    accepted = {status.strip().lower() for status in accepted_statuses if status.strip()}
    if not accepted:
        raise ParseError("accepted_statuses must contain at least one non-empty status")
    script_ratio_threshold = _limbu_document_script_ratio_threshold(source_document)
    line_lookup, positional_lookup = _document_line_indexes(source_document)
    applied_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    seen_targets: set[str] = set()
    review_sha = _sha256_file(review_path)
    for row_number, row in enumerate(review_rows, start=2):
        status = str(row.get("review_status") or "").strip().lower()
        target_key = _review_target_key(row)
        if status not in accepted:
            skipped_rows.append(
                {
                    "row_number": row_number,
                    "line_id": row.get("line_id"),
                    "page_index": row.get("page_index"),
                    "line_index": row.get("line_index"),
                    "status": status or "blank",
                    "reason": "review_status_not_accepted",
                }
            )
            continue
        if target_key in seen_targets:
            raise ParseError(f"duplicate accepted review row for {target_key} in {review_path}")
        seen_targets.add(target_key)
        line = _resolve_review_line(row, line_lookup, positional_lookup, review_path=review_path, row_number=row_number)
        corrected_text = normalize_ocr_text(str(row.get("corrected_text") or ""))
        if not corrected_text.strip():
            raise ParseError(f"accepted review row {row_number} has empty corrected_text")
        original_text = line.text
        line.text = corrected_text
        post_correction = {
            "status": "human_review_applied",
            "method": "limbu_review_queue_tsv",
            "source_review_queue_path": str(review_path),
            "source_review_queue_sha256": review_sha,
            "review_status": status,
            "reviewer": row.get("reviewer") or "",
            "notes": row.get("notes") or "",
            "original_text": original_text,
            "corrected_text": corrected_text,
        }
        limbu_metadata = line.metadata.setdefault("limbu_pipeline", {})
        if isinstance(limbu_metadata, dict):
            limbu_metadata["original_needs_review"] = limbu_metadata.get("needs_review")
            limbu_metadata["needs_review"] = False
            limbu_metadata["review_resolved"] = True
            limbu_metadata["post_correction"] = post_correction
        script_class = str(row.get("script_class") or "").strip()
        if not script_class:
            script_class = str(_limbu_script_profile(corrected_text, script_ratio_threshold=script_ratio_threshold)["script_class"])
        _validate_limbu_post_correction_script_class(
            script_class,
            label=f"accepted review row {row_number} script_class",
        )
        applied_rows.append(
            {
                "row_number": row_number,
                "page_index": line.page_index,
                "line_id": line.line_id,
                "script_class": script_class,
                "review_status": status,
                "reviewer": row.get("reviewer") or "",
                "original_text": original_text,
                "corrected_text": corrected_text,
                "changed": original_text != corrected_text,
            }
        )
    _refresh_blocks_from_lines(source_document)
    return {
        "document": source_document,
        "review_rows": len(review_rows),
        "applied_count": len(applied_rows),
        "changed_count": sum(1 for row in applied_rows if row.get("changed")),
        "skipped_count": len(skipped_rows),
        "script_ratio_threshold": script_ratio_threshold,
        "applied_rows": applied_rows,
        "skipped_rows": skipped_rows,
    }


def _document_text_by_key(document: Document) -> dict[str, str]:
    rows: dict[str, str] = {}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            key = f"page:{page.page_index}:line:{line_index}:line_id:{line.line_id or ''}"
            rows[key] = line.text
    return rows


def _expected_limbu_correction_pair_rows(applied_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in applied_rows:
        rows.append(
            {
                "sample_id": row.get("line_id") or f"review-row-{row.get('row_number')}",
                "language": "limbu",
                "script_class": row.get("script_class"),
                "noisy_text": row.get("original_text"),
                "clean_text": row.get("corrected_text"),
                "source": "limbu_review_queue",
                "review_status": row.get("review_status"),
                "reviewer": row.get("reviewer"),
            }
        )
    return rows


def _read_jsonl_objects(path: Path, issues: list[str], *, label: str) -> list[dict[str, object]]:
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{label}:{line_number} invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{label}:{line_number} row must be a JSON object")
            continue
        rows.append(payload)
    return rows


def _read_limbu_correction_pair_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ParseError(f"Limbu correction-pairs JSONL does not exist: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid Limbu correction-pairs JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ParseError(f"Limbu correction-pairs row must be an object: {path}:{line_number}")
        noisy_text = payload.get("noisy_text")
        clean_text = payload.get("clean_text")
        if not isinstance(noisy_text, str) or not isinstance(clean_text, str):
            raise ParseError(f"Limbu correction-pairs row requires string noisy_text and clean_text: {path}:{line_number}")
        rows.append(payload)
    return rows


def _read_limbu_pack_jsonl_rows(path: Path, issues: list[str], *, label: str) -> list[dict[str, object]]:
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{label}:{line_number} invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{label}:{line_number} row must be a JSON object")
            continue
        rows.append(payload)
    return rows


def _resolve_pack_artifact_path(path_value: object, root: Path, fallback_name: str) -> Path:
    if isinstance(path_value, str) and path_value.strip():
        path = Path(path_value)
        if path.is_absolute() or path.exists():
            return path
        candidate = root / path.name
        if candidate.exists():
            return candidate
        return path
    return root / fallback_name


def _audit_limbu_correction_pair_pack_rows(
    rows: list[dict[str, object]],
    *,
    expected_split: str,
    summary: dict[str, object],
    issues: list[str],
) -> None:
    pack_id = summary.get("pack_id")
    for index, row in enumerate(rows, start=1):
        label = f"{expected_split} row {index}"
        if not row.get("sample_id"):
            issues.append(f"{label} missing sample_id")
        if row.get("language") != "limbu":
            issues.append(f"{label} language must be 'limbu'")
        script_class = str(row.get("script_class") or "").strip()
        if script_class not in _LIMBU_SCRIPT_CLASSES:
            issues.append(
                f"{label} script_class has unsupported value: "
                f"{script_class!r}; allowed={list(_LIMBU_SCRIPT_CLASSES)}"
            )
        if row.get("pack_id") != pack_id:
            issues.append(f"{label} pack_id mismatch: row={row.get('pack_id')!r} summary={pack_id!r}")
        if row.get("split") != expected_split:
            issues.append(f"{label} split mismatch: {row.get('split')!r}")
        noisy_text = row.get("noisy_text")
        clean_text = row.get("clean_text")
        if not isinstance(noisy_text, str) or not noisy_text:
            issues.append(f"{label} missing non-empty noisy_text")
        if not isinstance(clean_text, str) or not clean_text:
            issues.append(f"{label} missing non-empty clean_text")
        elif _count_limbu_or_devanagari(clean_text) == 0:
            issues.append(f"{label} clean_text has no Limbu or Devanagari text")
        expected_changed = isinstance(noisy_text, str) and isinstance(clean_text, str) and noisy_text != clean_text
        if row.get("changed") != expected_changed:
            issues.append(f"{label} changed flag mismatch")
        if row.get("claim_eligible") is not False:
            issues.append(f"{label} claim_eligible must be false")


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            substitution_cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _limbu_correction_pair_split_key(row: dict[str, object], pack_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "pack_id": pack_id,
                "sample_id": row.get("sample_id"),
                "noisy_text": row.get("noisy_text"),
                "clean_text": row.get("clean_text"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _limbu_correction_pair_pack_digest_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "pack_id": row.get("pack_id"),
        "sample_id": row.get("sample_id"),
        "language": row.get("language"),
        "script_class": row.get("script_class"),
        "split": row.get("split"),
        "noisy_text_sha256": hashlib.sha256(str(row.get("noisy_text") or "").encode("utf-8")).hexdigest(),
        "clean_text_sha256": hashlib.sha256(str(row.get("clean_text") or "").encode("utf-8")).hexdigest(),
        "changed": row.get("changed"),
        "claim_eligible": row.get("claim_eligible"),
    }


def _limbu_correction_pair_pack_content_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_tsv_rows(path: Path, issues: list[str], *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames is None:
            issues.append(f"{label} has no header")
            return []
        return [dict(row) for row in reader]


def _limbu_output_audit_report(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, object] | None,
    issues: list[str],
    warnings: list[str],
    policy: dict[str, object],
) -> dict[str, object]:
    return {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "audit_stage": "limbu_output_bundle",
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "pipeline_stage": manifest.get("pipeline_stage") if isinstance(manifest, dict) else None,
        "policy": policy,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def _write_limbu_output_audit(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "limbu-output-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Limbu Output Audit",
        "",
        f"Output dir: `{report.get('output_dir')}`",
        f"Pipeline stage: `{report.get('pipeline_stage')}`",
        f"Passed: `{'yes' if report.get('passed') else 'no'}`",
        f"Policy: `{json.dumps(report.get('policy', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues")
    if isinstance(issues, list) and issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    (output_dir / "limbu-output-audit.md").write_text("\n".join(lines), encoding="utf-8")


def _document_line_indexes(document: Document) -> tuple[dict[str, TextLine], dict[tuple[int, int], TextLine]]:
    line_lookup: dict[str, TextLine] = {}
    positional_lookup: dict[tuple[int, int], TextLine] = {}
    for page in document.pages:
        for line_index, line in enumerate(page.text_lines):
            if line.line_id:
                if line.line_id in line_lookup:
                    raise ParseError(f"duplicate line_id in document: {line.line_id}")
                line_lookup[line.line_id] = line
            positional_lookup[(page.page_index, line_index)] = line
    return line_lookup, positional_lookup


def _read_limbu_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"page_index", "line_index", "line_id", "review_status", "corrected_text"}
        if reader.fieldnames is None:
            raise ParseError(f"Limbu review queue TSV has no header: {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ParseError(f"Limbu review queue TSV missing columns {missing}: {path}")
        return [dict(row) for row in reader]


def _review_target_key(row: dict[str, str]) -> str:
    line_id = str(row.get("line_id") or "").strip()
    if line_id:
        return f"line_id:{line_id}"
    return f"position:{row.get('page_index')}:{row.get('line_index')}"


def _resolve_review_line(
    row: dict[str, str],
    line_lookup: dict[str, TextLine],
    positional_lookup: dict[tuple[int, int], TextLine],
    *,
    review_path: Path,
    row_number: int,
) -> TextLine:
    line_id = str(row.get("line_id") or "").strip()
    if line_id:
        line = line_lookup.get(line_id)
        if line is None:
            raise ParseError(f"review row {row_number} references unknown line_id {line_id!r}: {review_path}")
        return line
    try:
        page_index = int(str(row.get("page_index") or ""))
        line_index = int(str(row.get("line_index") or ""))
    except ValueError as exc:
        raise ParseError(f"review row {row_number} needs integer page_index and line_index when line_id is empty") from exc
    line = positional_lookup.get((page_index, line_index))
    if line is None:
        raise ParseError(f"review row {row_number} references missing page/line {page_index}/{line_index}: {review_path}")
    return line


def _refresh_blocks_from_lines(document: Document) -> None:
    by_line_id = {
        line.line_id: line
        for page in document.pages
        for line in page.text_lines
        if line.line_id
    }
    for page in document.pages:
        for block in page.blocks:
            if block.block_type not in {"text", "title", "caption", "figure"}:
                continue
            lines = [by_line_id[line_id] for line_id in block.line_ids if line_id in by_line_id]
            if not lines:
                continue
            block.text = "\n".join(normalize_ocr_text(line.text).strip() for line in lines if line.text.strip())


def _write_limbu_correction_artifacts(audit: dict[str, object], output_dir: Path) -> None:
    applied_rows = audit.get("applied_rows")
    if not isinstance(applied_rows, list):
        applied_rows = []
    (output_dir / "limbu-correction-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "limbu-correction-pairs.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row.get("line_id") or f"review-row-{row.get('row_number')}",
                    "language": "limbu",
                    "script_class": row.get("script_class"),
                    "noisy_text": row.get("original_text"),
                    "clean_text": row.get("corrected_text"),
                    "source": "limbu_review_queue",
                    "review_status": row.get("review_status"),
                    "reviewer": row.get("reviewer"),
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in applied_rows
            if isinstance(row, dict)
        ),
        encoding="utf-8",
    )


def _limbu_script_profile(text: str, *, script_ratio_threshold: float) -> dict[str, object]:
    non_space = [char for char in text if not char.isspace()]
    total = len(non_space)
    limbu_count = _count_range(non_space, _LIMBU_CODEPOINT_START, _LIMBU_CODEPOINT_END)
    devanagari_count = _count_range(non_space, _DEVANAGARI_CODEPOINT_START, _DEVANAGARI_CODEPOINT_END)
    latin_count = sum(1 for char in non_space if ("A" <= char <= "Z") or ("a" <= char <= "z"))
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


def _limbu_review_reasons(
    text: str,
    confidence: float | None,
    script_class: object,
    *,
    low_confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_text")
    if confidence is None:
        reasons.append("missing_confidence")
    elif confidence < low_confidence_threshold:
        reasons.append("low_confidence")
    if script_class == "mixed_limbu_devanagari":
        reasons.append("mixed_devanagari_sirijonga")
    elif script_class == "other":
        reasons.append("no_limbu_or_devanagari_script")
    return reasons


def _count_range(chars: list[str], start: int, end: int) -> int:
    return sum(1 for char in chars if start <= ord(char) <= end)


def _write_limbu_pipeline_artifacts(audit: dict[str, object], output_dir: Path) -> None:
    rows = audit.get("rows")
    if not isinstance(rows, list):
        rows = []
    review_rows = [row for row in rows if isinstance(row, dict) and row.get("needs_review")]
    (output_dir / "limbu-pipeline-audit.json").write_text(
        json.dumps({key: value for key, value in audit.items() if key != "rows"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "limbu-line-audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if isinstance(row, dict)),
        encoding="utf-8",
    )
    tsv_path = output_dir / "limbu-review-queue.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "page_index",
                "line_index",
                "line_id",
                "script_class",
                "confidence",
                "review_confidence_threshold",
                "review_reasons",
                "text",
                "review_status",
                "corrected_text",
                "reviewer",
                "notes",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "page_index": row.get("page_index"),
                    "line_index": row.get("line_index"),
                    "line_id": row.get("line_id"),
                    "script_class": row.get("script_class"),
                    "confidence": row.get("confidence"),
                    "review_confidence_threshold": row.get("review_confidence_threshold"),
                    "review_reasons": ",".join(str(item) for item in row.get("review_reasons", [])),
                    "text": row.get("text"),
                    "review_status": "pending",
                    "corrected_text": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def _fallback_decision(
    *,
    primary_engine: str,
    fallback_engine: str | None,
    fallback_model_config: str | Path | None,
    primary_average_confidence: float | None,
    low_confidence_threshold: float,
    primary_quality_score: float | None = None,
    fallback_min_quality_score: float | None = None,
) -> dict[str, object] | None:
    if not fallback_engine and fallback_model_config is None:
        return None
    resolved_fallback_engine = fallback_engine or "candidate"
    decision: dict[str, object] = {
        "configured": True,
        "triggered": False,
        "outcome": "not_triggered",
        "primary_engine": primary_engine,
        "fallback_engine": resolved_fallback_engine,
        "fallback_model_config": str(fallback_model_config) if fallback_model_config is not None else None,
        "primary_average_confidence": primary_average_confidence,
        "threshold": low_confidence_threshold,
        "primary_quality_score": primary_quality_score,
        "min_quality_score": fallback_min_quality_score,
    }
    if fallback_model_config is None and resolved_fallback_engine.lower() == primary_engine.lower():
        decision["reason"] = "same_engine"
        return decision
    if primary_average_confidence is not None and primary_average_confidence < low_confidence_threshold:
        decision["triggered"] = True
        decision["outcome"] = "triggered"
        decision["reason"] = "average_confidence_below_threshold"
        return decision
    if fallback_min_quality_score is not None and primary_quality_score is not None and primary_quality_score < fallback_min_quality_score:
        decision["triggered"] = True
        decision["outcome"] = "triggered"
        decision["reason"] = "quality_score_below_threshold"
        return decision
    if primary_average_confidence is None and fallback_min_quality_score is None:
        decision["reason"] = "confidence_not_available"
        return decision
    decision["reason"] = "confidence_and_quality_above_threshold" if fallback_min_quality_score is not None else "confidence_above_threshold"
    return decision


def export_document(document: Document, output_dir: Path, source_path: Path) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for table in document.tables:
        write_table_files(table, tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for figure in document.figures:
        _export_figure_image(figure, figures_dir, source_path)
    (figures_dir / "metadata.json").write_text(
        json.dumps([figure.to_dict() for figure in document.figures], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quality = document.metadata.get("quality")
    if isinstance(quality, dict):
        (output_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "document.json").write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "document.md").write_text(render_document_markdown(document), encoding="utf-8")
    (output_dir / "document.body.md").write_text(
        render_document_markdown(document, include_structural_roles=False),
        encoding="utf-8",
    )


def _export_figure_image(figure: object, figures_dir: Path, source_path: Path) -> None:
    image_path = getattr(figure, "image_path", None)
    if image_path:
        source_figure = Path(str(image_path))
        if not source_figure.is_absolute():
            source_base = source_path if source_path.is_dir() else source_path.parent
            source_figure = source_base / source_figure
        if source_figure.exists():
            target = figures_dir / source_figure.name
            shutil.copy2(source_figure, target)
            setattr(figure, "image_path", str(target))
            return
        _record_figure_export_warning(figure, f"figure image_path does not exist: {source_figure}")
    _crop_figure_from_source(figure, figures_dir, source_path)


def _crop_figure_from_source(figure: object, figures_dir: Path, source_path: Path) -> None:
    source_image = _source_image_for_figure(figure, source_path)
    if source_image is None:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        _record_figure_export_warning(figure, f"Pillow is required to crop figure image: {exc}")
        return
    try:
        bbox = getattr(figure, "bbox")
        left = max(0, int(round(bbox.x)))
        top = max(0, int(round(bbox.y)))
        right = max(left, int(round(bbox.right)))
        bottom = max(top, int(round(bbox.bottom)))
        with Image.open(source_image) as image:
            right = min(right, image.width)
            bottom = min(bottom, image.height)
            if right <= left or bottom <= top:
                _record_figure_export_warning(figure, f"figure bbox is outside source image bounds: {bbox.to_list()}")
                return
            target = figures_dir / f"{getattr(figure, 'figure_id', 'figure')}.png"
            image.crop((left, top, right, bottom)).save(target)
        setattr(figure, "image_path", str(target))
        metadata = getattr(figure, "metadata")
        if isinstance(metadata, dict):
            metadata["export_source"] = "source_crop"
    except Exception as exc:
        _record_figure_export_warning(figure, f"{type(exc).__name__}: {exc}")


def _source_image_for_figure(figure: object, source_path: Path) -> Path | None:
    if source_path.is_dir():
        page_paths = [
            path
            for path in sorted(source_path.iterdir())
            if path.is_file() and path.suffix.lower() in _CROPPABLE_IMAGE_SUFFIXES
        ]
        if not page_paths:
            _record_figure_export_warning(figure, f"page bundle has no croppable image files: {source_path}")
            return None
        page_index = getattr(figure, "page_index", 0)
        if not isinstance(page_index, int) or page_index < 0 or page_index >= len(page_paths):
            _record_figure_export_warning(figure, f"figure page_index is outside page bundle bounds: {page_index}")
            return None
        return page_paths[page_index]
    if source_path.suffix.lower() not in _CROPPABLE_IMAGE_SUFFIXES:
        _record_figure_export_warning(figure, f"source image format cannot be cropped: {source_path.suffix or '<none>'}")
        return None
    return source_path


def _record_figure_export_warning(figure: object, message: str) -> None:
    metadata = getattr(figure, "metadata", None)
    if isinstance(metadata, dict):
        warnings = metadata.setdefault("export_warnings", [])
        if isinstance(warnings, list):
            warnings.append(message)
