#!/usr/bin/env python3
"""Run a claim-grade page-level OCR evaluation from exported document JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import platform
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex
from rapidfuzz.distance import Levenshtein

from ocrtech.linearization import linearize_texts_column_major
from ocrtech.manifest import ManifestEntry, load_manifest, read_jsonl, sha256_file
from ocrtech.metrics import bbox_iou, detection_score
from ocrtech.references import ReferenceDocument, load_reference, score_document
from ocrtech.schemas import BBox, Document, TextLine


DEFAULT_NORMALIZATION = "NFC_preserve_whitespace"
PAGE_OCR_OUTPUT_KINDS = {"live_detector", "oracle_line_boxes", "sidecar_fixture", "unknown"}
CLAIM_SAFE_DOMAINS = {"real_print", "real_handwritten"}
CLAIM_SAFE_GT_STATUSES = {"human_verified", "machine_converted_spot_checked"}
UNCERTAINTY_BOOTSTRAP_SAMPLES = 200
UNCERTAINTY_BOOTSTRAP_SEED = 1729
MIN_CLAIM_SCORING_UNITS = 30
SOURCE_DOCUMENT_KEYS = ("source_document", "source_doc", "source_pdf", "source_image", "source_xml", "document", "doc_id")
PAGE_KEYS = ("page", "page_index", "page_number", "source_page", "page_id")
BBOX_KEYS = ("bbox", "bounding_box", "coords", "polygon")
CLAIM_ELIGIBLE_KEYS = ("claim_evidence_eligible", "claim_eligible", "paper_claim_eligible")
HUMAN_REVIEW_EVIDENCE_KEYS = (
    "reviewed_by",
    "reviewer",
    "native_reviewer",
    "gt_reviewer",
    "review_artifact",
    "gt_review_artifact",
    "review_batch",
    "double_reviewed_by",
    "inter_annotator_agreement",
    "correction_rate",
)
PROVENANCE_SIGNAL_KEYS = (
    "label_provenance",
    "gt_source",
    "source_gt_status",
    "source_label_status",
    "reference_status",
    "assisted_source",
    "source_type",
    "data_origin",
    "evidence_type",
    "dataset_type",
    "slices",
)


@dataclass
class PageEvalRow:
    sample_id: str
    input_path: str
    input_sha256: str
    document_path: str
    document_sha256: str
    document_source_path: str
    language: str
    script: str
    domain: str
    gt_status: str
    split: str
    claim_eligible: bool
    source_document: str = ""
    source_page: str = ""
    source_bbox: str = ""
    reference_path: str = ""
    reference_sha256: str = ""
    reference_status: str = ""
    reference_claim_eligible: str = ""
    slices: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    text_codepoint_edits: int = 0
    text_codepoint_ref_len: int = 0
    text_grapheme_edits: int = 0
    text_grapheme_ref_len: int = 0
    text_exact: bool = False
    prediction_text: str = ""
    reference_text: str = ""
    reference_fields: list[str] = field(default_factory=list)
    line_alignment: dict[str, float | int] = field(default_factory=dict)
    ocr_output_kind: str = "unknown"


@dataclass(frozen=True)
class LeakageRow:
    sample_id: str
    image_path: str
    image_sha256: str
    text_sha256: str
    sample_sha256: str
    source_sample_id: str = ""


@dataclass(frozen=True)
class ReferenceLine:
    line_id: str
    text: str
    bbox: BBox
    page_index: int = 0


@dataclass(frozen=True)
class LineAlignmentRow:
    sample_id: str
    kind: str
    reference_line_id: str
    predicted_line_id: str
    reference_bbox: str
    predicted_bbox: str
    page_index: int
    iou: float
    reference_text: str
    predicted_text: str
    text_codepoint_edits: int
    text_codepoint_ref_len: int
    text_cer_codepoint: float
    text_grapheme_edits: int
    text_grapheme_ref_len: int
    text_cer_grapheme: float


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def _graphemes(value: str) -> list[str]:
    return regex.findall(r"\X", _nfc(value))


def _codepoint_id(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def _read_script_inventory(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"script inventory file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"script inventory file must contain a JSON object: {path}")
    return payload


def _required_codepoints(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("required_codepoints", [])
    if not isinstance(raw, list):
        raise ValueError("script inventory required_codepoints must be a list")
    required: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"script inventory codepoint entry must be a string: {item!r}")
        value = item.strip()
        if not value:
            continue
        if value.upper().startswith("U+"):
            try:
                required.add(f"U+{int(value[2:], 16):04X}")
            except ValueError as exc:
                raise ValueError(f"script inventory codepoint entry is not valid hex: {item!r}") from exc
            continue
        normalized = _nfc(value)
        if len(normalized) != 1:
            raise ValueError(f"script inventory codepoint entry must be one character or U+XXXX: {item!r}")
        required.add(_codepoint_id(normalized))
    return sorted(required)


def _required_graphemes(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("required_graphemes", [])
    if not isinstance(raw, list):
        raise ValueError("script inventory required_graphemes must be a list")
    required: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"script inventory grapheme entry must be a string: {item!r}")
        value = _nfc(item)
        if value.strip():
            required.add(value)
    return sorted(required)


def _script_inventory_summary(texts: list[str], inventory_file: Path | None) -> dict[str, Any]:
    payload = _read_script_inventory(inventory_file)
    text = _nfc("\n".join(texts))
    codepoints = sorted({_codepoint_id(ch) for ch in text if not ch.isspace()})
    graphemes = sorted({cluster for cluster in _graphemes(text) if cluster.strip()})
    required_cps = _required_codepoints(payload)
    required_graphs = _required_graphemes(payload)
    categories = [unicodedata.category(ch) for ch in text if not ch.isspace()]
    return {
        "inventory_file": str(inventory_file) if inventory_file else None,
        "inventory_file_sha256": sha256_file(inventory_file) if inventory_file else None,
        "observed_codepoint_count": len(codepoints),
        "observed_grapheme_count": len(graphemes),
        "observed_codepoints": codepoints,
        "observed_graphemes": graphemes,
        "required_codepoints": required_cps,
        "required_graphemes": required_graphs,
        "missing_required_codepoints": sorted(set(required_cps) - set(codepoints)),
        "missing_required_graphemes": sorted(set(required_graphs) - set(graphemes)),
        "digit_codepoint_count": sum(1 for ch in text if not ch.isspace() and unicodedata.category(ch) == "Nd"),
        "punctuation_codepoint_count": sum(1 for category in categories if category.startswith("P")),
        "combining_mark_codepoint_count": sum(1 for category in categories if category.startswith("M")),
    }


def _dictionary_codepoints(dictionary: str) -> set[str] | None:
    path = Path(str(dictionary or ""))
    if not path.exists() or not path.is_file():
        return None
    codepoints: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = _nfc(line.strip())
        if not token:
            continue
        for ch in token:
            if not ch.isspace():
                codepoints.add(_codepoint_id(ch))
    return codepoints


def _dictionary_coverage(texts: list[str], dictionary: str) -> dict[str, Any]:
    dictionary_path = Path(str(dictionary or ""))
    gt_codepoints = sorted({_codepoint_id(ch) for text in texts for ch in _nfc(text) if not ch.isspace()})
    dictionary_codepoints = _dictionary_codepoints(dictionary)
    if dictionary_codepoints is None:
        return {
            "dictionary": str(dictionary or ""),
            "dictionary_file_sha256": None,
            "checked": False,
            "gt_codepoint_count": len(gt_codepoints),
            "missing_gt_codepoints": [],
            "missing_gt_codepoint_count": 0,
            "reason": "dictionary file is not readable",
        }
    missing = sorted(set(gt_codepoints) - dictionary_codepoints)
    return {
        "dictionary": str(dictionary or ""),
        "dictionary_file_sha256": sha256_file(dictionary_path),
        "checked": True,
        "dictionary_codepoint_count": len(dictionary_codepoints),
        "gt_codepoint_count": len(gt_codepoints),
        "missing_gt_codepoints": missing,
        "missing_gt_codepoint_count": len(missing),
    }


def _normalization_summary(rows: list[PageEvalRow]) -> dict[str, Any]:
    prediction_changed = sum(1 for row in rows if row.prediction_text != _nfc(row.prediction_text))
    reference_changed = sum(1 for row in rows if row.reference_text != _nfc(row.reference_text))
    return {
        "profile": DEFAULT_NORMALIZATION,
        "unit": "page",
        "total": len(rows),
        "prediction_changed": prediction_changed,
        "reference_changed": reference_changed,
        "any_changed": sum(
            1
            for row in rows
            if row.prediction_text != _nfc(row.prediction_text) or row.reference_text != _nfc(row.reference_text)
        ),
    }


def _text_hash(value: str) -> str:
    return hashlib.sha256(_nfc(value).encode("utf-8")).hexdigest()


def _sample_hash(image_sha256: str, text_sha256: str) -> str:
    return hashlib.sha256(f"{image_sha256}\0{text_sha256}".encode("utf-8")).hexdigest()


def _artifact_identity(value: str) -> dict[str, Any]:
    value = str(value or "")
    if not value:
        return {"value": "", "verified": False, "reason": "missing"}
    path = Path(value)
    if path.exists() and path.is_file():
        return {
            "value": value,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "verified": True,
        }
    return {"value": value, "verified": False, "reason": "not a readable local file"}


def _artifact_identities(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "checkpoint": _artifact_identity(args.checkpoint),
        "config": _artifact_identity(args.config),
        "dictionary": _artifact_identity(args.dictionary),
    }


def _ocr_output_kind(args: argparse.Namespace) -> str:
    value = str(getattr(args, "ocr_output_kind", "") or "").strip()
    if not value:
        return "unknown"
    if value not in PAGE_OCR_OUTPUT_KINDS:
        raise ValueError(
            f"unsupported OCR output kind: {value!r}; expected one of {sorted(PAGE_OCR_OUTPUT_KINDS)}"
        )
    return value


def _resolve_manifest_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _validate_manifest_identity(manifest: list[ManifestEntry]) -> None:
    sample_ids = [entry.sample_id for entry in manifest if entry.sample_id]
    duplicate_sample_ids = _duplicate_values(sample_ids)
    if duplicate_sample_ids:
        raise ValueError(f"duplicate eval manifest sample_id values: {duplicate_sample_ids[:20]}")
    image_paths = [entry.image_path for entry in manifest if entry.image_path]
    duplicate_image_paths = _duplicate_values(image_paths)
    if duplicate_image_paths:
        raise ValueError(f"duplicate eval manifest image_path values: {duplicate_image_paths[:20]}")


def _validate_manifest_image_hash(entry: ManifestEntry, input_path: Path) -> str:
    actual = sha256_file(input_path)
    if entry.sha256 and entry.sha256 != actual:
        raise ValueError(
            f"eval manifest sha256 mismatch for {entry.sample_id}: "
            f"manifest={entry.sha256} actual={actual}"
        )
    return actual


def _document_candidates(entry: ManifestEntry, input_path: Path, documents_dir: Path) -> list[Path]:
    return [
        documents_dir / entry.sample_id / "document.json",
        documents_dir / input_path.stem / "document.json",
        documents_dir / f"{entry.sample_id}.json",
        documents_dir / f"{input_path.stem}.json",
    ]


def _load_document(entry: ManifestEntry, input_path: Path, documents_dir: Path) -> tuple[Document | None, Path | None]:
    for candidate in _document_candidates(entry, input_path, documents_dir):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"document JSON is invalid for {entry.sample_id}: {candidate}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"document JSON must contain an object for {entry.sample_id}: {candidate}")
        return Document.from_dict(payload), candidate
    return None, None


def _page_output_audit_filename(args: argparse.Namespace) -> str:
    return str(getattr(args, "output_audit_filename", "") or "limbu-output-audit.json")


def _page_output_audit_passed_field(args: argparse.Namespace) -> str:
    return str(getattr(args, "output_audit_passed_field", "") or "passed")


def _requires_page_output_audit(args: argparse.Namespace, ocr_output_kind: str) -> bool:
    return bool(getattr(args, "require_output_audit", False)) or (bool(getattr(args, "require_claim_safe", False)) and ocr_output_kind == "live_detector")


def _read_page_output_audit(
    sample_id: str,
    output_dir: Path,
    *,
    filename: str,
    passed_field: str,
) -> dict[str, Any]:
    audit_path = output_dir / filename
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "exists": audit_path.is_file(),
        "passed": False,
        "sha256": None,
        "size_bytes": None,
        "issue": "",
    }
    if not audit_path.is_file():
        row["issue"] = f"missing page output audit: {audit_path}"
        return row
    row["sha256"] = sha256_file(audit_path)
    row["size_bytes"] = audit_path.stat().st_size
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        row["issue"] = f"page output audit is invalid JSON: {audit_path}: {exc}"
        return row
    if not isinstance(payload, dict):
        row["issue"] = f"page output audit must contain a JSON object: {audit_path}"
        return row
    passed = payload.get(passed_field)
    row["passed"] = passed is True
    if passed is not True:
        row["issue"] = f"page output audit did not pass: {audit_path} {passed_field}={passed!r}"
    return row


def _reference_for(entry: ManifestEntry, input_path: Path, manifest_dir: Path) -> ReferenceDocument | None:
    explicit = entry.metadata.get("reference_path") if entry.metadata else None
    explicit_path = _resolve_manifest_path(explicit, manifest_dir) if isinstance(explicit, str) and explicit else None
    return load_reference(input_path, explicit_path=explicit_path)


def _reference_path_for(entry: ManifestEntry, input_path: Path, manifest_dir: Path) -> Path | None:
    explicit = entry.metadata.get("reference_path") if entry.metadata else None
    candidates: list[Path] = []
    if isinstance(explicit, str) and explicit:
        candidates.append(_resolve_manifest_path(explicit, manifest_dir))
    candidates.extend(
        [
            input_path.with_suffix(input_path.suffix + ".ref.json"),
            input_path.with_suffix(".ref.json"),
            input_path.with_suffix(input_path.suffix + ".ref.txt"),
            input_path.with_suffix(".ref.txt"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _reference_payload_for(entry: ManifestEntry, input_path: Path, manifest_dir: Path) -> dict[str, Any] | None:
    explicit = entry.metadata.get("reference_path") if entry.metadata else None
    candidates: list[Path] = []
    if isinstance(explicit, str) and explicit:
        candidates.append(_resolve_manifest_path(explicit, manifest_dir))
    candidates.extend([input_path.with_suffix(input_path.suffix + ".ref.json"), input_path.with_suffix(".ref.json")])
    for candidate in candidates:
        if not candidate.exists() or candidate.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"reference JSON is invalid for {entry.sample_id}: {candidate}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"reference JSON must contain an object for {entry.sample_id}: {candidate}")
        return payload
    return None


def _reference_fields(reference: ReferenceDocument | None) -> list[str]:
    if reference is None:
        return []
    fields: list[str] = []
    if reference.text is not None:
        fields.append("text")
    if reference.markdown is not None:
        fields.append("markdown")
    if reference.reading_order:
        fields.append("reading_order")
    if reference.tables:
        fields.append("tables")
    if reference.figures:
        fields.append("figures")
    return fields


def _reference_lines(payload: dict[str, Any] | None) -> list[ReferenceLine]:
    if payload is None:
        return []
    candidates = payload.get("reading_order")
    if candidates is None:
        candidates = payload.get("lines")
    if not isinstance(candidates, list):
        return []
    rows: list[ReferenceLine] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id") or item.get("line_id") or f"ref-{index:04d}"
        raw_bbox = item.get("bbox")
        if raw_bbox is None:
            continue
        try:
            bbox = BBox.from_any(raw_bbox)
        except Exception:
            continue
        if bbox.w <= 0 or bbox.h <= 0:
            continue
        rows.append(
            ReferenceLine(
                line_id=str(raw_id),
                text=str(item.get("text") or ""),
                bbox=bbox,
                page_index=int(item.get("page_index", 0)),
            )
        )
    return rows


def _predicted_lines(document: Document) -> list[TextLine]:
    lines: list[TextLine] = []
    for page in document.pages:
        lines.extend(page.text_lines)
    return lines


def _predicted_reading_order_ids(document: Document) -> list[str]:
    ordered: list[str] = []
    for page in sorted(document.pages, key=lambda item: item.page_index):
        for block in sorted(page.blocks, key=lambda item: item.order):
            ordered.extend(line_id for line_id in block.line_ids if line_id)
    return ordered


def _line_structural_role(line: TextLine) -> str:
    metadata = line.metadata if isinstance(line.metadata, dict) else {}
    role = metadata.get("structural_role")
    if not isinstance(role, str) or not role.strip():
        return "text"
    return role.strip()


def _is_structural_line(line: TextLine) -> bool:
    return _line_structural_role(line) not in {"", "text"}


def _metric_safe_role(role: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in role.strip().lower())
    return cleaned.strip("_") or "unknown"


def _matched_reading_order_counts(predicted_positions: list[int]) -> tuple[int, int]:
    correct = 0
    total = 0
    for left_index, left_position in enumerate(predicted_positions):
        for right_position in predicted_positions[left_index + 1 :]:
            total += 1
            if left_position < right_position:
                correct += 1
    return correct, total


def _align_lines(
    sample_id: str,
    reference_lines: list[ReferenceLine],
    predicted_lines: list[TextLine],
    iou_threshold: float,
    *,
    predicted_reading_order_ids: list[str] | None = None,
) -> tuple[dict[str, float | int], list[LineAlignmentRow]]:
    unmatched_predicted = set(range(len(predicted_lines)))
    rows: list[LineAlignmentRow] = []
    matched_ious: list[float] = []
    matched_cp_edits = 0
    matched_cp_len = 0
    matched_gr_edits = 0
    matched_gr_len = 0
    matched_exact = 0
    structural_prediction_count = sum(1 for predicted in predicted_lines if _is_structural_line(predicted))
    structural_role_counts: dict[str, int] = {}
    for predicted in predicted_lines:
        if not _is_structural_line(predicted):
            continue
        role_key = _metric_safe_role(_line_structural_role(predicted))
        structural_role_counts[role_key] = structural_role_counts.get(role_key, 0) + 1
    matched_structural_count = 0
    matched_structural_exact = 0
    fallback_positions = {
        predicted.line_id or f"__predicted_index_{index}": index
        for index, predicted in enumerate(predicted_lines)
    }
    reading_order_positions = {
        line_id: index for index, line_id in enumerate(predicted_reading_order_ids or [])
    }
    matched_predicted_positions: list[int] = []
    for reference in reference_lines:
        best_index: int | None = None
        best_iou = 0.0
        for predicted_index in sorted(unmatched_predicted):
            predicted = predicted_lines[predicted_index]
            if predicted.page_index != reference.page_index:
                continue
            overlap = bbox_iou(predicted.bbox, reference.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = predicted_index
        if best_index is None or best_iou < iou_threshold:
            cp_edits, cp_len, gr_edits, gr_len = _score_text_micro("", reference.text)
            rows.append(
                LineAlignmentRow(
                    sample_id=sample_id,
                    kind="reference_unmatched",
                    reference_line_id=reference.line_id,
                    predicted_line_id="",
                    reference_bbox=_bbox_text(reference.bbox),
                    predicted_bbox="",
                    page_index=reference.page_index,
                    iou=best_iou,
                    reference_text=reference.text,
                    predicted_text="",
                    text_codepoint_edits=cp_edits,
                    text_codepoint_ref_len=cp_len,
                    text_cer_codepoint=cp_edits / max(cp_len, 1),
                    text_grapheme_edits=gr_edits,
                    text_grapheme_ref_len=gr_len,
                    text_cer_grapheme=gr_edits / max(gr_len, 1),
                )
            )
            continue
        predicted = predicted_lines[best_index]
        unmatched_predicted.remove(best_index)
        matched_ious.append(best_iou)
        cp_edits, cp_len, gr_edits, gr_len = _score_text_micro(predicted.text, reference.text)
        matched_cp_edits += cp_edits
        matched_cp_len += cp_len
        matched_gr_edits += gr_edits
        matched_gr_len += gr_len
        matched_exact += int(_nfc(predicted.text) == _nfc(reference.text))
        if _is_structural_line(predicted):
            matched_structural_count += 1
            matched_structural_exact += int(_nfc(predicted.text) == _nfc(reference.text))
        predicted_key = predicted.line_id or f"__predicted_index_{best_index}"
        matched_predicted_positions.append(
            reading_order_positions.get(predicted_key, fallback_positions.get(predicted_key, best_index))
        )
        rows.append(
            LineAlignmentRow(
                sample_id=sample_id,
                kind="matched",
                reference_line_id=reference.line_id,
                predicted_line_id=predicted.line_id or "",
                reference_bbox=_bbox_text(reference.bbox),
                predicted_bbox=_bbox_text(predicted.bbox),
                page_index=reference.page_index,
                iou=best_iou,
                reference_text=reference.text,
                predicted_text=predicted.text,
                text_codepoint_edits=cp_edits,
                text_codepoint_ref_len=cp_len,
                text_cer_codepoint=cp_edits / max(cp_len, 1),
                text_grapheme_edits=gr_edits,
                text_grapheme_ref_len=gr_len,
                text_cer_grapheme=gr_edits / max(gr_len, 1),
            )
        )
    for predicted_index in sorted(unmatched_predicted):
        predicted = predicted_lines[predicted_index]
        cp_edits, cp_len, gr_edits, gr_len = _score_text_micro(predicted.text, "")
        rows.append(
            LineAlignmentRow(
                sample_id=sample_id,
                kind="prediction_unmatched",
                reference_line_id="",
                predicted_line_id=predicted.line_id or "",
                reference_bbox="",
                predicted_bbox=_bbox_text(predicted.bbox),
                page_index=predicted.page_index,
                iou=0.0,
                reference_text="",
                predicted_text=predicted.text,
                text_codepoint_edits=cp_edits,
                text_codepoint_ref_len=cp_len,
                text_cer_codepoint=1.0 if predicted.text else 0.0,
                text_grapheme_edits=gr_edits,
                text_grapheme_ref_len=gr_len,
                text_cer_grapheme=1.0 if predicted.text else 0.0,
            )
        )
    score = detection_score(len(predicted_lines), len(reference_lines), len(matched_ious))
    unmatched_reference_count = len(reference_lines) - len(matched_ious)
    unmatched_prediction_count = len(unmatched_predicted)
    blank_prediction_count = sum(1 for predicted in predicted_lines if not _nfc(predicted.text).strip())
    blank_unmatched_prediction_count = sum(
        1 for predicted_index in unmatched_predicted if not _nfc(predicted_lines[predicted_index].text).strip()
    )
    matched_order_correct, matched_order_pairs = _matched_reading_order_counts(matched_predicted_positions)
    alignment = {
        "line_reference_count": len(reference_lines),
        "line_prediction_count": len(predicted_lines),
        "line_match_count": len(matched_ious),
        "line_unmatched_reference_count": unmatched_reference_count,
        "line_unmatched_prediction_count": unmatched_prediction_count,
        "line_blank_prediction_count": blank_prediction_count,
        "line_blank_unmatched_prediction_count": blank_unmatched_prediction_count,
        "line_missed_reference_rate": unmatched_reference_count / max(len(reference_lines), 1),
        "line_extra_prediction_rate": unmatched_prediction_count / max(len(predicted_lines), 1),
        "line_blank_prediction_rate": blank_prediction_count / max(len(predicted_lines), 1),
        "line_detection_precision": score.precision,
        "line_detection_recall": score.recall,
        "line_detection_f1": score.f1,
        "line_mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "matched_line_text_codepoint_edits": matched_cp_edits,
        "matched_line_text_codepoint_ref_len": matched_cp_len,
        "matched_line_text_cer_codepoint_micro": matched_cp_edits / max(matched_cp_len, 1),
        "matched_line_text_grapheme_edits": matched_gr_edits,
        "matched_line_text_grapheme_ref_len": matched_gr_len,
        "matched_line_text_cer_grapheme_micro": matched_gr_edits / max(matched_gr_len, 1),
        "matched_line_text_exact_count": matched_exact,
        "matched_line_text_exact_accuracy": matched_exact / max(len(matched_ious), 1),
        "matched_line_reading_order_correct_pairs": matched_order_correct,
        "matched_line_reading_order_total_pairs": matched_order_pairs,
        "matched_line_reading_order_pair_accuracy": matched_order_correct / matched_order_pairs
        if matched_order_pairs
        else 1.0,
        "structural_line_prediction_count": structural_prediction_count,
        "structural_line_matched_count": matched_structural_count,
        "structural_line_unmatched_count": structural_prediction_count - matched_structural_count,
        "structural_line_match_rate": matched_structural_count / max(structural_prediction_count, 1),
        "structural_line_exact_text_match_count": matched_structural_exact,
        "structural_line_exact_text_match_rate": matched_structural_exact / max(matched_structural_count, 1),
    }
    for role, count in sorted(structural_role_counts.items()):
        alignment[f"structural_role_{role}_prediction_count"] = count
    return alignment, rows


def _score_text_micro(prediction: str, reference: str) -> tuple[int, int, int, int]:
    pred_nfc = _nfc(prediction)
    ref_nfc = _nfc(reference)
    pred_graph = _graphemes(pred_nfc)
    ref_graph = _graphemes(ref_nfc)
    return (
        Levenshtein.distance(pred_nfc, ref_nfc),
        len(ref_nfc),
        Levenshtein.distance(pred_graph, ref_graph),
        len(ref_graph),
    )


def _entry_slices(entry: ManifestEntry) -> list[str]:
    metadata = entry.metadata or {}
    slices = metadata.get("slices")
    if isinstance(slices, list):
        return [str(item) for item in slices]
    return [entry.split]


def _metadata_value(entry: ManifestEntry, key: str, fallback: str) -> str:
    metadata = entry.metadata or {}
    value = metadata.get(key)
    if value in (None, ""):
        return fallback
    return str(value)


def _entry_language(entry: ManifestEntry, fallback: str) -> str:
    return _metadata_value(entry, "language", fallback)


def _entry_script(entry: ManifestEntry, fallback: str) -> str:
    return _metadata_value(entry, "script", fallback)


def _entry_domain(entry: ManifestEntry, fallback: str) -> str:
    return _metadata_value(entry, "domain", fallback)


def _entry_gt_status(entry: ManifestEntry, fallback: str) -> str:
    return _metadata_value(entry, "gt_status", fallback)


def _entry_split(entry: ManifestEntry) -> str:
    metadata = entry.metadata or {}
    value = metadata.get("split") or metadata.get("split_role") or entry.split
    return str(value)


def _entry_claim_eligible(entry: ManifestEntry, fallback: bool) -> bool:
    metadata = entry.metadata or {}
    value = next((metadata.get(key) for key in CLAIM_ELIGIBLE_KEYS if metadata.get(key) is not None), None)
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _metadata_signal_text(metadata: dict[str, Any]) -> str:
    values: list[str] = []
    for key in PROVENANCE_SIGNAL_KEYS:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif isinstance(value, dict):
            values.extend(f"{item_key}:{item_value}" for item_key, item_value in value.items())
        else:
            values.append(str(value))
    return " ".join(values).lower().replace("_", " ").replace("-", " ")


def _metadata_contradictions(sample_id: str, metadata: dict[str, Any], domain: str, gt_status: str) -> list[str]:
    signal_text = _metadata_signal_text(metadata)
    if not signal_text:
        return []
    contradictions: list[str] = []
    if domain in CLAIM_SAFE_DOMAINS and any(term in signal_text for term in ("synthetic", "transliterated", "transliteration")):
        contradictions.append(f"{sample_id}: synthetic/transliterated provenance is declared as real domain {domain}")
    if gt_status == "human_verified" and any(
        term in signal_text
        for term in ("machine converted", "machine", "synthetic", "transliterated", "transliteration")
    ):
        contradictions.append(f"{sample_id}: machine/synthetic provenance is declared as human_verified GT")
    return contradictions


def _read_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"summary/datasheet file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"summary/datasheet file is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"summary/datasheet file must contain a JSON object: {path}")
    return payload


def _read_spotcheck_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"native spot-check calibration file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"native spot-check calibration file is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"native spot-check calibration file must contain a JSON object: {path}")
    return payload


def _spotcheck_calibration_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_evidence_eligible": payload.get("claim_evidence_eligible"),
        "recommended_pack_gt_status": payload.get("recommended_pack_gt_status"),
        "language": payload.get("language"),
        "script": payload.get("script"),
        "domain": payload.get("domain"),
        "source_gt_status": payload.get("source_gt_status"),
        "counts": payload.get("counts", {}),
        "machine_label_metrics": payload.get("machine_label_metrics", {}),
        "eligibility_reasons": payload.get("eligibility_reasons", []),
    }


def _leakage_rows_from_manifest(path: Path, manifest_dir: Path | None = None) -> list[LeakageRow]:
    rows: list[LeakageRow] = []
    for payload in read_jsonl(path):
        metadata = dict(payload.get("metadata") or {})
        sample_id = str(payload.get("sample_id") or payload.get("id") or metadata.get("sample_id") or "")
        image_path_value = str(payload.get("image_path") or payload.get("image") or metadata.get("image_path") or "")
        text_value = payload.get("text") if payload.get("text") is not None else metadata.get("text")
        text = str(text_value or "")
        image_path = _resolve_manifest_path(image_path_value, manifest_dir or path.parent)
        image_sha256 = str(
            payload.get("sha256")
            or payload.get("image_sha256")
            or metadata.get("image_sha256")
            or metadata.get("sha256")
            or metadata.get("source_image_sha256")
            or (sha256_file(image_path) if image_path.exists() else "")
        )
        text_sha256 = str(payload.get("text_sha256") or metadata.get("text_sha256") or (_text_hash(text) if text else ""))
        sample_sha256 = str(
            payload.get("sample_sha256")
            or metadata.get("sample_sha256")
            or (_sample_hash(image_sha256, text_sha256) if image_sha256 and text_sha256 else "")
        )
        source_sample_id = str(payload.get("source_sample_id") or metadata.get("source_sample_id") or sample_id)
        rows.append(
            LeakageRow(
                sample_id=sample_id,
                image_path=image_path_value,
                image_sha256=image_sha256,
                text_sha256=text_sha256,
                sample_sha256=sample_sha256,
                source_sample_id=source_sample_id,
            )
        )
    return rows


def _overlap(left: list[LeakageRow], right: list[LeakageRow], field_name: str) -> list[str]:
    left_values = {str(getattr(row, field_name)) for row in left if getattr(row, field_name)}
    right_values = {str(getattr(row, field_name)) for row in right if getattr(row, field_name)}
    return sorted(left_values & right_values)


def _leakage_report(train_manifests: list[Path], eval_manifest: Path) -> dict[str, Any]:
    eval_rows = _leakage_rows_from_manifest(eval_manifest, eval_manifest.parent)
    train_rows: list[LeakageRow] = []
    for manifest in train_manifests:
        train_rows.extend(_leakage_rows_from_manifest(manifest, manifest.parent))
    overlaps = {
        "sample_id": _overlap(train_rows, eval_rows, "sample_id"),
        "source_sample_id": _overlap(train_rows, eval_rows, "source_sample_id"),
        "image_sha256": _overlap(train_rows, eval_rows, "image_sha256"),
        "sample_sha256": _overlap(train_rows, eval_rows, "sample_sha256"),
        "image_path": _overlap(train_rows, eval_rows, "image_path"),
        "text_sha256": _overlap(train_rows, eval_rows, "text_sha256"),
    }
    return {
        "train_manifests": [str(path) for path in train_manifests],
        "checked": bool(train_manifests),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "overlaps": overlaps,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _confidence_interval(values: list[float]) -> dict[str, float]:
    return {
        "low": _percentile(values, 0.025),
        "high": _percentile(values, 0.975),
    }


def _uncertainty(rows: list[PageEvalRow]) -> dict[str, Any]:
    n = len(rows)
    cp_points: list[float] = []
    gr_points: list[float] = []
    exact_points: list[float] = []
    if n:
        rng = random.Random(UNCERTAINTY_BOOTSTRAP_SEED)
        for _ in range(UNCERTAINTY_BOOTSTRAP_SAMPLES):
            cp_edits = cp_len = gr_edits = gr_len = 0
            exact = text_refs = 0
            for _ in range(n):
                row = rows[rng.randrange(n)]
                cp_edits += row.text_codepoint_edits
                cp_len += row.text_codepoint_ref_len
                gr_edits += row.text_grapheme_edits
                gr_len += row.text_grapheme_ref_len
                if "text" in row.reference_fields:
                    text_refs += 1
                    exact += int(row.text_exact)
            cp_points.append(cp_edits / max(cp_len, 1))
            gr_points.append(gr_edits / max(gr_len, 1))
            exact_points.append(exact / max(text_refs, 1))
    return {
        "method": "deterministic_page_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_samples": UNCERTAINTY_BOOTSTRAP_SAMPLES,
        "seed": UNCERTAINTY_BOOTSTRAP_SEED,
        "scoring_units": n,
        "effective_sample_size": n,
        "min_claim_scoring_units": MIN_CLAIM_SCORING_UNITS,
        "claim_sample_size_ok": n >= MIN_CLAIM_SCORING_UNITS,
        "metrics": {
            "text_cer_codepoint_micro": _confidence_interval(cp_points),
            "text_cer_grapheme_micro": _confidence_interval(gr_points),
            "text_exact_accuracy": _confidence_interval(exact_points),
        },
    }


def _aggregate(rows: list[PageEvalRow]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.metrics.items():
            metric_values.setdefault(key, []).append(value)
        for key, value in row.line_alignment.items():
            metric_values.setdefault(key, []).append(float(value))
    cp_edits = sum(row.text_codepoint_edits for row in rows)
    cp_len = sum(row.text_codepoint_ref_len for row in rows)
    gr_edits = sum(row.text_grapheme_edits for row in rows)
    gr_len = sum(row.text_grapheme_ref_len for row in rows)
    text_reference_pages = sum(1 for row in rows if "text" in row.reference_fields)
    text_exact_pages = sum(1 for row in rows if "text" in row.reference_fields and row.text_exact)
    zero_length_references = sum(1 for row in rows if "text" in row.reference_fields and len(_nfc(row.reference_text)) == 0)
    matched_cp_edits = sum(int(row.line_alignment.get("matched_line_text_codepoint_edits", 0) or 0) for row in rows)
    matched_cp_len = sum(int(row.line_alignment.get("matched_line_text_codepoint_ref_len", 0) or 0) for row in rows)
    matched_gr_edits = sum(int(row.line_alignment.get("matched_line_text_grapheme_edits", 0) or 0) for row in rows)
    matched_gr_len = sum(int(row.line_alignment.get("matched_line_text_grapheme_ref_len", 0) or 0) for row in rows)
    matched_exact_count = sum(int(row.line_alignment.get("matched_line_text_exact_count", 0) or 0) for row in rows)
    matched_line_count = sum(int(row.line_alignment.get("line_match_count", 0) or 0) for row in rows)
    line_reference_count = sum(int(row.line_alignment.get("line_reference_count", 0) or 0) for row in rows)
    line_prediction_count = sum(int(row.line_alignment.get("line_prediction_count", 0) or 0) for row in rows)
    line_unmatched_reference_count = sum(
        int(row.line_alignment.get("line_unmatched_reference_count", 0) or 0) for row in rows
    )
    line_unmatched_prediction_count = sum(
        int(row.line_alignment.get("line_unmatched_prediction_count", 0) or 0) for row in rows
    )
    line_blank_prediction_count = sum(int(row.line_alignment.get("line_blank_prediction_count", 0) or 0) for row in rows)
    line_blank_unmatched_prediction_count = sum(
        int(row.line_alignment.get("line_blank_unmatched_prediction_count", 0) or 0) for row in rows
    )
    line_detection_micro = detection_score(line_prediction_count, line_reference_count, matched_line_count)
    matched_order_correct = sum(
        int(row.line_alignment.get("matched_line_reading_order_correct_pairs", 0) or 0) for row in rows
    )
    matched_order_pairs = sum(
        int(row.line_alignment.get("matched_line_reading_order_total_pairs", 0) or 0) for row in rows
    )
    structural_prediction_count = sum(int(row.line_alignment.get("structural_line_prediction_count", 0) or 0) for row in rows)
    structural_matched_count = sum(int(row.line_alignment.get("structural_line_matched_count", 0) or 0) for row in rows)
    structural_unmatched_count = sum(int(row.line_alignment.get("structural_line_unmatched_count", 0) or 0) for row in rows)
    structural_exact_count = sum(int(row.line_alignment.get("structural_line_exact_text_match_count", 0) or 0) for row in rows)
    structural_role_totals: dict[str, int] = {}
    for row in rows:
        for key, value in row.line_alignment.items():
            if key.startswith("structural_role_") and key.endswith("_prediction_count"):
                structural_role_totals[key] = structural_role_totals.get(key, 0) + int(value or 0)
    result = {
        "scored_pages": len(rows),
        "metric_means": {key: sum(values) / len(values) for key, values in sorted(metric_values.items()) if values},
        "zero_length_text_references": zero_length_references,
        "text_codepoint_edits": cp_edits,
        "text_codepoint_ref_len": cp_len,
        "text_cer_codepoint_micro": cp_edits / max(cp_len, 1),
        "text_grapheme_edits": gr_edits,
        "text_grapheme_ref_len": gr_len,
        "text_cer_grapheme_micro": gr_edits / max(gr_len, 1),
        "text_reference_pages": text_reference_pages,
        "text_exact_pages": text_exact_pages,
        "text_exact_accuracy": text_exact_pages / max(text_reference_pages, 1),
        "matched_line_text_codepoint_edits": matched_cp_edits,
        "matched_line_text_codepoint_ref_len": matched_cp_len,
        "matched_line_text_cer_codepoint_micro": matched_cp_edits / max(matched_cp_len, 1),
        "matched_line_text_grapheme_edits": matched_gr_edits,
        "matched_line_text_grapheme_ref_len": matched_gr_len,
        "matched_line_text_cer_grapheme_micro": matched_gr_edits / max(matched_gr_len, 1),
        "matched_line_text_exact_count": matched_exact_count,
        "matched_line_text_exact_accuracy": matched_exact_count / max(matched_line_count, 1),
        "line_reference_count": line_reference_count,
        "line_prediction_count": line_prediction_count,
        "line_match_count": matched_line_count,
        "line_unmatched_reference_count": line_unmatched_reference_count,
        "line_unmatched_prediction_count": line_unmatched_prediction_count,
        "line_blank_prediction_count": line_blank_prediction_count,
        "line_blank_unmatched_prediction_count": line_blank_unmatched_prediction_count,
        "line_detection_precision_micro": line_detection_micro.precision,
        "line_detection_recall_micro": line_detection_micro.recall,
        "line_detection_f1_micro": line_detection_micro.f1,
        "line_missed_reference_rate": line_unmatched_reference_count / max(line_reference_count, 1),
        "line_extra_prediction_rate": line_unmatched_prediction_count / max(line_prediction_count, 1),
        "line_blank_prediction_rate": line_blank_prediction_count / max(line_prediction_count, 1),
        "line_blank_unmatched_prediction_rate": line_blank_unmatched_prediction_count / max(line_prediction_count, 1),
        "matched_line_reading_order_correct_pairs": matched_order_correct,
        "matched_line_reading_order_total_pairs": matched_order_pairs,
        "matched_line_reading_order_pair_accuracy": matched_order_correct / matched_order_pairs
        if matched_order_pairs
        else 1.0,
        "structural_line_prediction_count": structural_prediction_count,
        "structural_line_matched_count": structural_matched_count,
        "structural_line_unmatched_count": structural_unmatched_count,
        "structural_line_match_rate": structural_matched_count / max(structural_prediction_count, 1),
        "structural_line_exact_text_match_count": structural_exact_count,
        "structural_line_exact_text_match_rate": structural_exact_count / max(structural_matched_count, 1),
        "uncertainty": _uncertainty(rows),
    }
    result.update(structural_role_totals)
    return result


def _slice_metrics(rows: list[PageEvalRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[PageEvalRow]] = {}
    for row in rows:
        key = (row.language, row.script, row.domain, row.gt_status, row.split)
        groups.setdefault(key, []).append(row)
    slices: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        language, script, domain, gt_status, split = key
        metrics = _aggregate(group)
        metrics.update(
            {
                "language": language,
                "script": script,
                "domain": domain,
                "gt_status": gt_status,
                "split": split,
            }
        )
        slices.append(metrics)
    return slices


def _coverage(rows: list[PageEvalRow]) -> dict[str, int]:
    fields = ("text", "markdown", "reading_order", "tables", "figures", "line_bboxes")
    coverage = {field: sum(1 for row in rows if field in row.reference_fields) for field in fields}
    coverage["line_alignment_scored"] = sum(1 for row in rows if row.line_alignment.get("line_reference_count", 0))
    return coverage


def _reference_claim_ready(row: PageEvalRow) -> bool:
    return row.reference_status == "verified" and row.reference_claim_eligible.strip().lower() == "true"


def _reference_claim_readiness(rows: list[PageEvalRow]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    claim_eligible_counts: dict[str, int] = {}
    not_ready: list[str] = []
    for row in rows:
        status = row.reference_status or "(missing)"
        claim_eligible = row.reference_claim_eligible or "(missing)"
        status_counts[status] = status_counts.get(status, 0) + 1
        claim_eligible_counts[claim_eligible] = claim_eligible_counts.get(claim_eligible, 0) + 1
        if not _reference_claim_ready(row):
            not_ready.append(row.sample_id)
    return {
        "ready_count": len(rows) - len(not_ready),
        "scored_pages": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "claim_eligible_counts": dict(sorted(claim_eligible_counts.items())),
        "not_ready_count": len(not_ready),
        "not_ready_sample_ids": not_ready[:50],
    }


def _metadata_coverage(entries: list[ManifestEntry]) -> dict[str, int]:
    def missing(key: str) -> int:
        return sum(1 for entry in entries if (entry.metadata or {}).get(key) in (None, ""))

    def has_any(metadata: dict[str, Any], keys: tuple[str, ...]) -> bool:
        return any(metadata.get(key) not in (None, "", [], {}) for key in keys)

    def missing_any(keys: tuple[str, ...]) -> int:
        return sum(1 for entry in entries if not has_any(entry.metadata or {}, keys))

    def missing_human_review_evidence() -> int:
        return sum(
            1
            for entry in entries
            if (entry.metadata or {}).get("gt_status") == "human_verified"
            and not has_any(entry.metadata or {}, HUMAN_REVIEW_EVIDENCE_KEYS)
        )

    return {
        "missing_source_document": missing_any(SOURCE_DOCUMENT_KEYS),
        "missing_page": missing_any(PAGE_KEYS),
        "missing_bbox": missing_any(BBOX_KEYS),
        "missing_claim_eligibility": missing_any(CLAIM_ELIGIBLE_KEYS),
        "missing_human_review_evidence": missing_human_review_evidence(),
        "missing_language": missing("language"),
        "missing_script": missing("script"),
        "missing_domain": missing("domain"),
        "missing_gt_status": missing("gt_status"),
    }


def _metadata_value_for_keys(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return str(value)
    return ""


def _same_source_path(document_source: str, input_path: Path) -> bool:
    if not document_source:
        return False
    source = Path(document_source)
    try:
        if source.resolve() == input_path.resolve():
            return True
    except OSError:
        if document_source == str(input_path):
            return True
    if source.is_file() and input_path.is_file():
        return sha256_file(source) == sha256_file(input_path)
    return False


def _document_source_path_mismatch(sample_id: str, document: Document, input_path: Path) -> str | None:
    if not _same_source_path(document.source_path, input_path):
        return f"{sample_id}: document.source_path {document.source_path!r} does not match eval input {str(input_path)!r}"
    return None


def _bbox_text(bbox: BBox | None) -> str:
    if bbox is None:
        return ""
    return json.dumps(bbox.to_list(), ensure_ascii=False, separators=(",", ":"))


def _warnings(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    leakage: dict[str, Any],
    summary: dict[str, Any],
    spotcheck_calibration: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    metadata_contradictions: list[str],
) -> list[str]:
    warnings: list[str] = []
    if metrics.get("ocr_output_kind") == "unknown":
        warnings.append("OCR output kind is not recorded")
    page_output_audit = metrics.get("page_output_audit") if isinstance(metrics.get("page_output_audit"), dict) else {}
    if page_output_audit.get("required") and page_output_audit.get("passed_count") != page_output_audit.get("expected_count"):
        warnings.append(
            "page output audits are not all passing: "
            f"{page_output_audit.get('passed_count')}/{page_output_audit.get('expected_count')} "
            f"filename={page_output_audit.get('filename')!r}"
        )
    if metrics["missing_documents"]:
        warnings.append(f"missing document outputs: {metrics['missing_documents']}/{metrics['manifest_samples']}")
    if metrics["missing_references"]:
        warnings.append(f"missing references: {metrics['missing_references']}/{metrics['manifest_samples']}")
    if metrics["scored_pages"] < metrics["manifest_samples"]:
        warnings.append(f"incomplete page scoring coverage: {metrics['scored_pages']}/{metrics['manifest_samples']}")
    document_source_path_mismatches = metrics.get("document_source_path_mismatches")
    if isinstance(document_source_path_mismatches, list) and document_source_path_mismatches:
        warnings.append(
            "document source_path mismatches detected: "
            + "; ".join(str(item) for item in document_source_path_mismatches[:10])
            + (f"; ... {len(document_source_path_mismatches) - 10} more" if len(document_source_path_mismatches) > 10 else "")
        )
    if not args.eval_pack_version:
        warnings.append("eval pack version not recorded")
    if not str(args.system_name or "").strip():
        warnings.append("system name not recorded")
    if not str(args.model_name or "").strip():
        warnings.append("model name not recorded")
    for name, identity in artifacts.items():
        if not identity.get("verified"):
            warnings.append(f"{name} artifact is not hash-verifiable: {identity.get('reason') or 'unverified'}")
    if not args.frozen_eval_pack:
        warnings.append("eval pack is not declared frozen")
    if not args.unseen_for_selection:
        warnings.append("eval pack is not declared unseen for model/checkpoint selection")
    if args.selected_checkpoint_on_test:
        warnings.append("checkpoint/model selection on this test pack is declared")
    if not args.claim_eligible:
        warnings.append("eval pack is not declared claim eligible")
    if metrics["claim_eligible_rows"] != metrics["manifest_samples"]:
        warnings.append(f"not all manifest rows are claim eligible: {metrics['claim_eligible_rows']}/{metrics['manifest_samples']}")
    if args.summary_file is None:
        warnings.append("eval pack summary/datasheet missing")
    if summary and not any(key in summary for key in ("dataset", "datasheet", "eval_pack", "version", "eval_pack_version")):
        warnings.append("eval pack summary lacks dataset/version/eval_pack metadata")
    if summary:
        summary_pack_content_sha256 = _summary_pack_content_sha256(summary)
        pack_content_sha256 = metrics.get("pack_content_sha256")
        if not summary_pack_content_sha256:
            warnings.append("eval pack summary missing pack_content_sha256")
        elif summary_pack_content_sha256 != pack_content_sha256:
            warnings.append(
                "pack content digest mismatch: "
                f"summary={summary_pack_content_sha256} actual={pack_content_sha256}"
            )
    metadata_coverage = metrics.get("metadata_coverage") or {}
    for key, count in metadata_coverage.items():
        if count:
            warnings.append(f"{key}: {count}/{metrics['manifest_samples']}")
    if metadata_contradictions:
        warnings.append(
            "metadata contradictions detected: "
            + "; ".join(metadata_contradictions[:10])
            + (f"; ... {len(metadata_contradictions) - 10} more" if len(metadata_contradictions) > 10 else "")
        )
    script_inventory = metrics.get("script_inventory") if isinstance(metrics.get("script_inventory"), dict) else {}
    if not script_inventory.get("inventory_file"):
        warnings.append("script inventory file missing")
    if script_inventory.get("missing_required_codepoints"):
        warnings.append(f"missing required script codepoints: {script_inventory['missing_required_codepoints'][:20]}")
    if script_inventory.get("missing_required_graphemes"):
        warnings.append(f"missing required script graphemes: {script_inventory['missing_required_graphemes'][:20]}")
    dictionary_coverage = metrics.get("dictionary_coverage") if isinstance(metrics.get("dictionary_coverage"), dict) else {}
    if dictionary_coverage.get("checked") and dictionary_coverage.get("missing_gt_codepoints"):
        warnings.append(f"GT codepoints missing from dictionary: {dictionary_coverage['missing_gt_codepoints'][:20]}")
    if int(metrics.get("zero_length_text_references", 0) or 0):
        warnings.append(f"zero-length text references present: {metrics['zero_length_text_references']}/{metrics['text_reference_pages']}")
    reference_readiness = metrics.get("reference_claim_readiness") if isinstance(metrics.get("reference_claim_readiness"), dict) else {}
    if reference_readiness.get("ready_count") != metrics.get("scored_pages"):
        warnings.append(
            "references are not all claim-ready: "
            f"{reference_readiness.get('ready_count', 0)}/{metrics.get('scored_pages', 0)} "
            "have reference_status=verified and claim_evidence_eligible=true; "
            f"not_ready={reference_readiness.get('not_ready_sample_ids', [])[:10]}"
        )
    uncertainty = metrics.get("uncertainty") if isinstance(metrics.get("uncertainty"), dict) else {}
    effective_sample_size = int(uncertainty.get("effective_sample_size", metrics.get("scored_pages", 0)) or 0)
    min_claim_scoring_units = int(uncertainty.get("min_claim_scoring_units", MIN_CLAIM_SCORING_UNITS) or MIN_CLAIM_SCORING_UNITS)
    if effective_sample_size < min_claim_scoring_units:
        warnings.append(f"claim-facing sample size too small: {effective_sample_size} scoring units, minimum {min_claim_scoring_units}")
    slices = metrics.get("slices") or []
    for row in slices:
        if not isinstance(row, dict):
            continue
        count = int(row.get("scored_pages", 0) or 0)
        if count < min_claim_scoring_units:
            warnings.append(
                "slice sample size too small: "
                f"{row.get('language')}/{row.get('script')}/{row.get('domain')}/{row.get('gt_status')}/{row.get('split')} "
                f"has {count} scoring units, minimum {min_claim_scoring_units}"
            )
    languages = {str(row.get("language")) for row in slices if isinstance(row, dict)}
    scripts = {str(row.get("script")) for row in slices if isinstance(row, dict)}
    domains = {str(row.get("domain")) for row in slices if isinstance(row, dict)}
    gt_statuses = {str(row.get("gt_status")) for row in slices if isinstance(row, dict)}
    splits = {str(row.get("split")) for row in slices if isinstance(row, dict)}
    if len(languages) > 1:
        warnings.append(f"mixed languages: {sorted(languages)}")
    if len(scripts) > 1:
        warnings.append(f"mixed scripts: {sorted(scripts)}")
    if len(domains) > 1:
        warnings.append(f"mixed eval domains: {sorted(domains)}")
    if len(gt_statuses) > 1:
        warnings.append(f"mixed GT statuses: {sorted(gt_statuses)}")
    if len(splits) > 1:
        warnings.append(f"mixed split statuses: {sorted(splits)}")
    if not splits <= {"eval", "test", "heldout", "held_out", "validation"}:
        warnings.append(f"unexpected split status for claim evaluation: {sorted(splits)}")
    if not domains <= CLAIM_SAFE_DOMAINS:
        warnings.append(f"not claim-safe real domain: {sorted(domains)}")
    if not gt_statuses <= CLAIM_SAFE_GT_STATUSES:
        warnings.append(f"not human-verified/spot-checked GT: {sorted(gt_statuses)}")
    if "machine_converted_spot_checked" in gt_statuses:
        if args.spotcheck_calibration_file is None:
            warnings.append("machine_converted_spot_checked GT declared without native spot-check calibration artifact")
        else:
            if not spotcheck_calibration.get("claim_evidence_eligible"):
                warnings.append("native spot-check calibration artifact is not claim evidence eligible")
            if spotcheck_calibration.get("recommended_pack_gt_status") != "machine_converted_spot_checked":
                warnings.append("native spot-check calibration does not recommend machine_converted_spot_checked GT")
            if spotcheck_calibration.get("source_gt_status") != "machine_converted":
                warnings.append("native spot-check calibration source_gt_status is not machine_converted")
            for field, values in (("language", languages), ("script", scripts), ("domain", domains)):
                value = spotcheck_calibration.get(field)
                if not isinstance(value, str) or not value:
                    warnings.append(f"native spot-check calibration missing {field}")
                elif value not in values:
                    warnings.append(f"native spot-check calibration {field} mismatch: {value!r} not in {sorted(values)}")
    coverage = metrics["reference_coverage"]
    if not coverage["text"]:
        warnings.append("no text references scored")
    if not coverage["reading_order"]:
        warnings.append("no reading-order references scored")
    if not coverage["line_alignment_scored"]:
        warnings.append("no line-bbox alignment references scored")
    if not (coverage["tables"] or coverage["figures"]):
        warnings.append("no layout detection references scored")
    leakage_overlaps = leakage.get("overlaps", {})
    hard_leakage = {key: values for key, values in leakage_overlaps.items() if key != "text_sha256" and values}
    text_overlap = leakage_overlaps.get("text_sha256") or []
    if hard_leakage:
        warnings.append("train/dev/eval leakage detected: " + json.dumps({key: values[:10] for key, values in hard_leakage.items()}, ensure_ascii=False))
    if text_overlap:
        warnings.append(f"train/dev/eval text overlap detected: {len(text_overlap)} text hashes")
    if not args.train_manifest:
        warnings.append("train/dev leakage check not run")
    return warnings


def _write_rows(path: Path, rows: list[PageEvalRow]) -> None:
    metric_names = sorted({name for row in rows for name in row.metrics})
    alignment_names = sorted({name for row in rows for name in row.line_alignment})
    fields = [
        "sample_id",
        "input_path",
        "input_sha256",
        "document_path",
        "document_sha256",
        "document_source_path",
        "ocr_output_kind",
        "language",
        "script",
        "domain",
        "gt_status",
        "split",
        "claim_eligible",
        "source_document",
        "source_page",
        "source_bbox",
        "reference_fields",
        "text_codepoint_edits",
        "text_codepoint_ref_len",
        "text_cer_codepoint",
        "text_grapheme_edits",
        "text_grapheme_ref_len",
        "text_cer_grapheme",
        "text_exact",
        *metric_names,
        *alignment_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "sample_id": row.sample_id,
                "input_path": row.input_path,
                "input_sha256": row.input_sha256,
                "document_path": row.document_path,
                "document_sha256": row.document_sha256,
                "document_source_path": row.document_source_path,
                "ocr_output_kind": row.ocr_output_kind,
                "language": row.language,
                "script": row.script,
                "domain": row.domain,
                "gt_status": row.gt_status,
                "split": row.split,
                "claim_eligible": row.claim_eligible,
                "source_document": row.source_document,
                "source_page": row.source_page,
                "source_bbox": row.source_bbox,
                "reference_fields": ";".join(row.reference_fields),
                "text_codepoint_edits": row.text_codepoint_edits,
                "text_codepoint_ref_len": row.text_codepoint_ref_len,
                "text_cer_codepoint": row.text_codepoint_edits / max(row.text_codepoint_ref_len, 1),
                "text_grapheme_edits": row.text_grapheme_edits,
                "text_grapheme_ref_len": row.text_grapheme_ref_len,
                "text_cer_grapheme": row.text_grapheme_edits / max(row.text_grapheme_ref_len, 1),
                "text_exact": row.text_exact,
            }
            payload.update(row.metrics)
            payload.update(row.line_alignment)
            writer.writerow(payload)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page_pack_content_digest(rows: list[PageEvalRow]) -> str:
    canonical_rows = [
        {
            "sample_id": row.sample_id,
            "input_sha256": row.input_sha256,
            "reference_sha256": row.reference_sha256,
            "reference_text_sha256": _text_sha256(row.reference_text),
            "language": row.language,
            "script": row.script,
            "domain": row.domain,
            "gt_status": row.gt_status,
            "split": row.split,
            "claim_eligible": bool(row.claim_eligible),
            "source_document": row.source_document,
            "source_page": row.source_page,
            "source_bbox": row.source_bbox,
            "reference_fields": ";".join(row.reference_fields),
            "reference_status": row.reference_status,
            "reference_claim_eligible": row.reference_claim_eligible,
        }
        for row in rows
    ]
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(canonical_rows, key=lambda item: item["sample_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary_pack_content_sha256(summary: dict[str, Any]) -> str:
    value = summary.get("pack_content_sha256")
    if isinstance(value, str) and value:
        return value
    eval_pack = summary.get("eval_pack")
    if isinstance(eval_pack, dict):
        nested = eval_pack.get("pack_content_sha256")
        if isinstance(nested, str) and nested:
            return nested
    return ""


def _write_text_pairs(path: Path, rows: list[PageEvalRow]) -> None:
    fields = [
        "sample_id",
        "input_path",
        "document_path",
        "prediction_text_sha256",
        "reference_text_sha256",
        "prediction_text",
        "reference_text",
        "text_codepoint_edits",
        "text_codepoint_ref_len",
        "text_grapheme_edits",
        "text_grapheme_ref_len",
        "text_exact",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "input_path": row.input_path,
                    "document_path": row.document_path,
                    "prediction_text_sha256": _text_sha256(row.prediction_text),
                    "reference_text_sha256": _text_sha256(row.reference_text),
                    "prediction_text": row.prediction_text,
                    "reference_text": row.reference_text,
                    "text_codepoint_edits": row.text_codepoint_edits,
                    "text_codepoint_ref_len": row.text_codepoint_ref_len,
                    "text_grapheme_edits": row.text_grapheme_edits,
                    "text_grapheme_ref_len": row.text_grapheme_ref_len,
                    "text_exact": row.text_exact,
                }
            )


def _write_reference_artifacts(path: Path, rows: list[PageEvalRow]) -> None:
    fields = [
        "sample_id",
        "input_path",
        "reference_path",
        "reference_sha256",
        "reference_fields",
        "reference_text_sha256",
        "reference_status",
        "reference_claim_eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "input_path": row.input_path,
                    "reference_path": row.reference_path,
                    "reference_sha256": row.reference_sha256,
                    "reference_fields": ";".join(row.reference_fields),
                    "reference_text_sha256": _text_sha256(row.reference_text),
                    "reference_status": row.reference_status,
                    "reference_claim_eligible": row.reference_claim_eligible,
                }
            )


def _write_missing(path: Path, missing_documents: list[str], missing_references: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "sample_id"])
        writer.writeheader()
        for sample_id in missing_documents:
            writer.writerow({"kind": "missing_document", "sample_id": sample_id})
        for sample_id in missing_references:
            writer.writerow({"kind": "missing_reference", "sample_id": sample_id})


def _confusions(rows: list[PageEvalRow], limit: int = 200) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        prediction = _nfc(row.prediction_text)
        reference = _nfc(row.reference_text)
        for opcode in Levenshtein.opcodes(prediction, reference):
            tag, i1, i2, j1, j2 = opcode
            if tag == "equal":
                continue
            left = prediction[i1:i2] or "<eps>"
            right = reference[j1:j2] or "<eps>"
            key = f"{left}->{right}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit])


def _write_worst(path: Path, rows: list[PageEvalRow], limit: int) -> list[PageEvalRow]:
    worst = sorted(rows, key=lambda row: row.text_grapheme_edits / max(row.text_grapheme_ref_len, 1), reverse=True)[:limit]
    fields = [
        "sample_id",
        "input_path",
        "input_sha256",
        "document_path",
        "document_sha256",
        "document_source_path",
        "language",
        "script",
        "domain",
        "gt_status",
        "split",
        "claim_eligible",
        "source_document",
        "source_page",
        "source_bbox",
        "reference_path",
        "reference_sha256",
        "reference_status",
        "reference_claim_eligible",
        "text_cer_grapheme",
        "text_cer_codepoint",
        "reference_fields",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in worst:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "input_path": row.input_path,
                    "input_sha256": row.input_sha256,
                    "document_path": row.document_path,
                    "document_sha256": row.document_sha256,
                    "document_source_path": row.document_source_path,
                    "language": row.language,
                    "script": row.script,
                    "domain": row.domain,
                    "gt_status": row.gt_status,
                    "split": row.split,
                    "claim_eligible": row.claim_eligible,
                    "source_document": row.source_document,
                    "source_page": row.source_page,
                    "source_bbox": row.source_bbox,
                    "reference_path": row.reference_path,
                    "reference_sha256": row.reference_sha256,
                    "reference_status": row.reference_status,
                    "reference_claim_eligible": row.reference_claim_eligible,
                    "text_cer_grapheme": row.text_grapheme_edits / max(row.text_grapheme_ref_len, 1),
                    "text_cer_codepoint": row.text_codepoint_edits / max(row.text_codepoint_ref_len, 1),
                    "reference_fields": ";".join(row.reference_fields),
                }
            )
    return worst


def _write_review_csv(path: Path, rows: list[PageEvalRow]) -> None:
    fields = [
        "sample_id",
        "input_path",
        "document_path",
        "document_source_path",
        "text_cer_grapheme",
        "text_cer_codepoint",
        "reference_fields",
        "review_status",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "input_path": row.input_path,
                    "document_path": row.document_path,
                    "document_source_path": row.document_source_path,
                    "text_cer_grapheme": row.text_grapheme_edits / max(row.text_grapheme_ref_len, 1),
                    "text_cer_codepoint": row.text_codepoint_edits / max(row.text_codepoint_ref_len, 1),
                    "reference_fields": ";".join(row.reference_fields),
                    "review_status": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def _write_review_html(path: Path, rows: list[PageEvalRow]) -> None:
    cards = []
    for row in rows:
        image_uri = Path(row.input_path).resolve().as_uri()
        cards.append(
            f"""
<section>
  <h2>{html.escape(row.sample_id)}</h2>
  <img src="{html.escape(image_uri)}" alt="{html.escape(row.sample_id)}">
  <dl>
    <dt>Document JSON</dt><dd>{html.escape(row.document_path)}</dd>
    <dt>Document source</dt><dd>{html.escape(row.document_source_path)}</dd>
    <dt>Language / script</dt><dd>{html.escape(row.language)} / {html.escape(row.script)}</dd>
    <dt>Domain / GT / split</dt><dd>{html.escape(row.domain)} / {html.escape(row.gt_status)} / {html.escape(row.split)}</dd>
    <dt>Text grapheme CER</dt><dd>{row.text_grapheme_edits / max(row.text_grapheme_ref_len, 1):.6f}</dd>
    <dt>Text codepoint CER</dt><dd>{row.text_codepoint_edits / max(row.text_codepoint_ref_len, 1):.6f}</dd>
    <dt>Reference fields</dt><dd>{html.escape(';'.join(row.reference_fields))}</dd>
  </dl>
</section>
"""
        )
    path.write_text(
        """<!doctype html>
<html><head><meta charset="utf-8"><style>
body{font-family:system-ui,sans-serif;margin:24px}section{border-bottom:1px solid #ddd;padding:16px 0}
img{max-width:100%;border:1px solid #ddd}dt{font-weight:700}dd{margin:0 0 8px 0}
</style></head><body><h1>Page OCR Eval Review Sample</h1>"""
        + "\n".join(cards)
        + "</body></html>\n",
        encoding="utf-8",
    )


def _write_line_alignment(path: Path, rows: list[LineAlignmentRow]) -> None:
    fields = [
        "sample_id",
        "kind",
        "reference_line_id",
        "predicted_line_id",
        "reference_bbox",
        "predicted_bbox",
        "page_index",
        "iou",
        "reference_text",
        "predicted_text",
        "text_codepoint_edits",
        "text_codepoint_ref_len",
        "text_cer_codepoint",
        "text_grapheme_edits",
        "text_grapheme_ref_len",
        "text_cer_grapheme",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "kind": row.kind,
                    "reference_line_id": row.reference_line_id,
                    "predicted_line_id": row.predicted_line_id,
                    "reference_bbox": row.reference_bbox,
                    "predicted_bbox": row.predicted_bbox,
                    "page_index": row.page_index,
                    "iou": row.iou,
                    "reference_text": row.reference_text,
                    "predicted_text": row.predicted_text,
                    "text_codepoint_edits": row.text_codepoint_edits,
                    "text_codepoint_ref_len": row.text_codepoint_ref_len,
                    "text_cer_codepoint": row.text_cer_codepoint,
                    "text_grapheme_edits": row.text_grapheme_edits,
                    "text_grapheme_ref_len": row.text_grapheme_ref_len,
                    "text_cer_grapheme": row.text_cer_grapheme,
                }
            )


def _write_metrics_md(path: Path, metrics: dict[str, Any], warnings: list[str]) -> None:
    uncertainty = metrics.get("uncertainty") if isinstance(metrics.get("uncertainty"), dict) else {}
    intervals = uncertainty.get("metrics") if isinstance(uncertainty.get("metrics"), dict) else {}
    graph_interval = intervals.get("text_cer_grapheme_micro") if isinstance(intervals.get("text_cer_grapheme_micro"), dict) else {}
    inventory = metrics.get("script_inventory") if isinstance(metrics.get("script_inventory"), dict) else {}
    normalization = metrics.get("normalization_summary") if isinstance(metrics.get("normalization_summary"), dict) else {}
    dictionary_coverage = metrics.get("dictionary_coverage") if isinstance(metrics.get("dictionary_coverage"), dict) else {}
    page_output_audit = metrics.get("page_output_audit") if isinstance(metrics.get("page_output_audit"), dict) else {}
    reference_readiness = metrics.get("reference_claim_readiness") if isinstance(metrics.get("reference_claim_readiness"), dict) else {}
    lines = [
        "# Page OCR Eval Metrics",
        "",
        f"- manifest samples: {metrics['manifest_samples']}",
        f"- scored pages: {metrics['scored_pages']}",
        f"- missing documents: {metrics['missing_documents']}",
        f"- missing references: {metrics['missing_references']}",
        f"- OCR output kind: {metrics.get('ocr_output_kind')}",
        f"- page output audit required: {page_output_audit.get('required', False)}",
        f"- page output audit passed: {page_output_audit.get('passed_count', 0)}/{page_output_audit.get('expected_count', 0)}",
        f"- text codepoint CER micro: {metrics['text_cer_codepoint_micro']:.6f}",
        f"- text grapheme CER micro: {metrics['text_cer_grapheme_micro']:.6f}",
        f"- text exact accuracy: {metrics['text_exact_accuracy']:.6f}",
        f"- text codepoint edits/ref: {metrics['text_codepoint_edits']}/{metrics['text_codepoint_ref_len']}",
        f"- text grapheme edits/ref: {metrics['text_grapheme_edits']}/{metrics['text_grapheme_ref_len']}",
        f"- text exact pages/ref pages: {metrics['text_exact_pages']}/{metrics['text_reference_pages']}",
        f"- zero-length text references: {metrics.get('zero_length_text_references')}",
        f"- reference claim-ready: {reference_readiness.get('ready_count')}/{reference_readiness.get('scored_pages')}",
        f"- reference status counts: {reference_readiness.get('status_counts')}",
        f"- reference claim-eligible counts: {reference_readiness.get('claim_eligible_counts')}",
        f"- normalization profile: {normalization.get('profile')}",
        f"- normalization changed reference/prediction/any: {normalization.get('reference_changed')}/{normalization.get('prediction_changed')}/{normalization.get('any_changed')} of {normalization.get('total')}",
        f"- dictionary coverage checked: {dictionary_coverage.get('checked')}",
        f"- dictionary reference codepoints/dictionary codepoints: {dictionary_coverage.get('gt_codepoint_count')}/{dictionary_coverage.get('dictionary_codepoint_count')}",
        f"- dictionary missing reference codepoints: {dictionary_coverage.get('missing_gt_codepoints')}",
        f"- line detection F1 mean: {metrics['metric_means'].get('line_detection_f1', 0.0):.6f}",
        f"- line detection precision/recall/F1 micro: {metrics.get('line_detection_precision_micro', 0.0):.6f}/{metrics.get('line_detection_recall_micro', 0.0):.6f}/{metrics.get('line_detection_f1_micro', 0.0):.6f}",
        f"- line reference/prediction/match counts: {metrics.get('line_reference_count')}/{metrics.get('line_prediction_count')}/{metrics.get('line_match_count')}",
        f"- missed/extra/blank line rates: {metrics.get('line_missed_reference_rate', 0.0):.6f}/{metrics.get('line_extra_prediction_rate', 0.0):.6f}/{metrics.get('line_blank_prediction_rate', 0.0):.6f}",
        f"- unmatched reference/prediction counts: {metrics.get('line_unmatched_reference_count')}/{metrics.get('line_unmatched_prediction_count')}",
        f"- blank prediction/unmatched blank prediction counts: {metrics.get('line_blank_prediction_count')}/{metrics.get('line_blank_unmatched_prediction_count')}",
        f"- line mean IoU: {metrics['metric_means'].get('line_mean_iou', 0.0):.6f}",
        f"- matched-line text codepoint CER micro: {metrics.get('matched_line_text_cer_codepoint_micro', 0.0):.6f}",
        f"- matched-line text grapheme CER micro: {metrics.get('matched_line_text_cer_grapheme_micro', 0.0):.6f}",
        f"- matched-line exact accuracy: {metrics.get('matched_line_text_exact_accuracy', 0.0):.6f}",
        f"- matched-line reading-order pair accuracy: {metrics.get('matched_line_reading_order_pair_accuracy', 0.0):.6f}",
        f"- matched-line reading-order correct/total pairs: {metrics.get('matched_line_reading_order_correct_pairs')}/{metrics.get('matched_line_reading_order_total_pairs')}",
        f"- matched-line text codepoint edits/ref: {metrics.get('matched_line_text_codepoint_edits')}/{metrics.get('matched_line_text_codepoint_ref_len')}",
        f"- matched-line text grapheme edits/ref: {metrics.get('matched_line_text_grapheme_edits')}/{metrics.get('matched_line_text_grapheme_ref_len')}",
        f"- structural line prediction/matched/unmatched counts: {metrics.get('structural_line_prediction_count')}/{metrics.get('structural_line_matched_count')}/{metrics.get('structural_line_unmatched_count')}",
        f"- structural line match/exact-text rates: {metrics.get('structural_line_match_rate', 0.0):.6f}/{metrics.get('structural_line_exact_text_match_rate', 0.0):.6f}",
        f"- uncertainty method: {uncertainty.get('method')}",
        f"- uncertainty scoring units: {uncertainty.get('effective_sample_size')}/{uncertainty.get('min_claim_scoring_units')}",
        f"- text grapheme CER 95% CI: {_fmt_metric(graph_interval.get('low'))}..{_fmt_metric(graph_interval.get('high'))}",
        f"- script inventory file: {inventory.get('inventory_file')}",
        f"- observed codepoints/graphemes: {inventory.get('observed_codepoint_count')}/{inventory.get('observed_grapheme_count')}",
        f"- missing required codepoints: {inventory.get('missing_required_codepoints')}",
        f"- missing required graphemes: {inventory.get('missing_required_graphemes')}",
        "",
        "## Mean Page Metrics",
        "",
    ]
    lines.extend([f"- {key}: {value:.6f}" for key, value in metrics["metric_means"].items()] or ["- none"])
    lines.extend(["", "## Slices", ""])
    for row in metrics["slices"]:
        lines.append(
            f"- {row['language']} / {row['script']} / {row['domain']} / {row['gt_status']} / {row['split']}: "
            f"pages={row['scored_pages']} text_graph_CER={row['text_cer_grapheme_micro']:.6f} "
            f"text_cp_CER={row['text_cer_codepoint_micro']:.6f} "
            f"line_F1={row['metric_means'].get('line_detection_f1', 0.0):.6f}"
        )
    lines.extend(["", "## Reference Coverage", ""])
    lines.extend([f"- {key}: {value}" for key, value in metrics["reference_coverage"].items()])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_metric(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.6f}"
    return str(value)


def _write_claim_safety_report(path: Path, eval_run: dict[str, Any], metrics: dict[str, Any]) -> None:
    warnings = eval_run.get("warnings") or []
    machine = eval_run.get("machine") if isinstance(eval_run.get("machine"), dict) else {}
    coverage = metrics.get("reference_coverage") if isinstance(metrics.get("reference_coverage"), dict) else {}
    metadata_coverage = metrics.get("metadata_coverage") if isinstance(metrics.get("metadata_coverage"), dict) else {}
    leakage = metrics.get("leakage") if isinstance(metrics.get("leakage"), dict) else {}
    metric_means = metrics.get("metric_means") if isinstance(metrics.get("metric_means"), dict) else {}
    uncertainty = metrics.get("uncertainty") if isinstance(metrics.get("uncertainty"), dict) else {}
    intervals = uncertainty.get("metrics") if isinstance(uncertainty.get("metrics"), dict) else {}
    graph_interval = intervals.get("text_cer_grapheme_micro") if isinstance(intervals.get("text_cer_grapheme_micro"), dict) else {}
    inventory = metrics.get("script_inventory") if isinstance(metrics.get("script_inventory"), dict) else {}
    normalization = metrics.get("normalization_summary") if isinstance(metrics.get("normalization_summary"), dict) else {}
    dictionary_coverage = metrics.get("dictionary_coverage") if isinstance(metrics.get("dictionary_coverage"), dict) else {}
    page_output_audit = eval_run.get("page_output_audit") if isinstance(eval_run.get("page_output_audit"), dict) else {}
    reference_readiness = metrics.get("reference_claim_readiness") if isinstance(metrics.get("reference_claim_readiness"), dict) else {}
    lines = [
        "# Eval Claim Safety Report",
        "",
        f"- eval mode: {eval_run.get('eval_mode')}",
        f"- claim safe: {eval_run.get('claim_safe')}",
        f"- created at: {eval_run.get('created_at')}",
        f"- machine platform: {machine.get('platform')}",
        f"- machine python: {machine.get('python')}",
        f"- command: {eval_run.get('command')}",
        "",
        "## What Was Scored",
        "",
        f"- system: {eval_run.get('system_name')}",
        f"- model: {eval_run.get('model_name')}",
        f"- checkpoint: {eval_run.get('checkpoint') or '(not recorded)'}",
        f"- config: {eval_run.get('config') or '(not recorded)'}",
        f"- dictionary: {eval_run.get('dictionary') or '(not recorded)'}",
        f"- artifact identity: {json.dumps(eval_run.get('artifact_identity', {}), ensure_ascii=False, sort_keys=True)}",
        f"- eval manifest: {eval_run.get('eval_manifest')}",
        f"- documents dir: {eval_run.get('documents_dir')}",
        f"- OCR output kind: {eval_run.get('ocr_output_kind')}",
        f"- page output audit required: {eval_run.get('require_output_audit')}",
        f"- page output audit passed: {page_output_audit.get('passed_count', 0)}/{page_output_audit.get('expected_count', 0)}",
        f"- manifest samples: {metrics.get('manifest_samples')}",
        f"- scored pages: {metrics.get('scored_pages')}",
        f"- missing documents: {metrics.get('missing_documents')}",
        f"- missing references: {metrics.get('missing_references')}",
        f"- review CSV: sample-review.csv",
        f"- review HTML: sample-review.html",
        "",
        "## Ground Truth And Pack",
        "",
        f"- eval pack version: {eval_run.get('eval_pack_version') or '(missing)'}",
        f"- frozen eval pack: {eval_run.get('frozen_eval_pack')}",
        f"- unseen for selection: {eval_run.get('unseen_for_selection')}",
        f"- selected checkpoint on test: {eval_run.get('selected_checkpoint_on_test')}",
        f"- languages: {eval_run.get('languages')}",
        f"- scripts: {eval_run.get('scripts')}",
        f"- domains: {eval_run.get('domains')}",
        f"- GT statuses: {eval_run.get('gt_statuses')}",
        f"- splits: {eval_run.get('splits')}",
        f"- reference coverage: {coverage}",
        f"- reference claim-ready: {reference_readiness.get('ready_count')}/{reference_readiness.get('scored_pages')}",
        f"- reference status counts: {reference_readiness.get('status_counts')}",
        f"- reference claim-eligible counts: {reference_readiness.get('claim_eligible_counts')}",
        f"- metadata coverage: {metadata_coverage}",
        f"- script inventory file: {inventory.get('inventory_file')}",
        f"- observed codepoints/graphemes: {inventory.get('observed_codepoint_count')}/{inventory.get('observed_grapheme_count')}",
        f"- missing required codepoints: {inventory.get('missing_required_codepoints')}",
        f"- missing required graphemes: {inventory.get('missing_required_graphemes')}",
        f"- metadata contradictions: {metrics.get('metadata_contradictions', [])}",
        f"- normalization profile: {normalization.get('profile')}",
        f"- normalization changed reference/prediction/any: {normalization.get('reference_changed')}/{normalization.get('prediction_changed')}/{normalization.get('any_changed')} of {normalization.get('total')}",
        f"- dictionary coverage checked: {dictionary_coverage.get('checked')}",
        f"- dictionary reference codepoints/dictionary codepoints: {dictionary_coverage.get('gt_codepoint_count')}/{dictionary_coverage.get('dictionary_codepoint_count')}",
        f"- dictionary missing reference codepoints: {dictionary_coverage.get('missing_gt_codepoints')}",
        f"- leakage checked: {leakage.get('checked')}",
        "",
        "## Metrics",
        "",
        f"- text codepoint CER micro: {_fmt_metric(metrics.get('text_cer_codepoint_micro'))}",
        f"- text grapheme CER micro: {_fmt_metric(metrics.get('text_cer_grapheme_micro'))}",
        f"- text exact accuracy: {_fmt_metric(metrics.get('text_exact_accuracy'))}",
        f"- line detection F1 mean: {_fmt_metric(metric_means.get('line_detection_f1'))}",
        f"- line detection precision/recall/F1 micro: {_fmt_metric(metrics.get('line_detection_precision_micro'))}/{_fmt_metric(metrics.get('line_detection_recall_micro'))}/{_fmt_metric(metrics.get('line_detection_f1_micro'))}",
        f"- line reference/prediction/match counts: {metrics.get('line_reference_count')}/{metrics.get('line_prediction_count')}/{metrics.get('line_match_count')}",
        f"- missed/extra/blank line rates: {_fmt_metric(metrics.get('line_missed_reference_rate'))}/{_fmt_metric(metrics.get('line_extra_prediction_rate'))}/{_fmt_metric(metrics.get('line_blank_prediction_rate'))}",
        f"- unmatched reference/prediction counts: {metrics.get('line_unmatched_reference_count')}/{metrics.get('line_unmatched_prediction_count')}",
        f"- blank prediction/unmatched blank prediction counts: {metrics.get('line_blank_prediction_count')}/{metrics.get('line_blank_unmatched_prediction_count')}",
        f"- line mean IoU: {_fmt_metric(metric_means.get('line_mean_iou'))}",
        f"- matched-line text codepoint CER micro: {_fmt_metric(metrics.get('matched_line_text_cer_codepoint_micro'))}",
        f"- matched-line text grapheme CER micro: {_fmt_metric(metrics.get('matched_line_text_cer_grapheme_micro'))}",
        f"- matched-line exact accuracy: {_fmt_metric(metrics.get('matched_line_text_exact_accuracy'))}",
        f"- matched-line reading-order pair accuracy: {_fmt_metric(metrics.get('matched_line_reading_order_pair_accuracy'))}",
        f"- reading-order pair accuracy: {_fmt_metric(metric_means.get('reading_order_pair_accuracy'))}",
        f"- text codepoint edits/ref: {metrics.get('text_codepoint_edits')}/{metrics.get('text_codepoint_ref_len')}",
        f"- text grapheme edits/ref: {metrics.get('text_grapheme_edits')}/{metrics.get('text_grapheme_ref_len')}",
        f"- text exact pages/ref pages: {metrics.get('text_exact_pages')}/{metrics.get('text_reference_pages')}",
        f"- matched-line text codepoint edits/ref: {metrics.get('matched_line_text_codepoint_edits')}/{metrics.get('matched_line_text_codepoint_ref_len')}",
        f"- matched-line text grapheme edits/ref: {metrics.get('matched_line_text_grapheme_edits')}/{metrics.get('matched_line_text_grapheme_ref_len')}",
        f"- matched-line reading-order correct/total pairs: {metrics.get('matched_line_reading_order_correct_pairs')}/{metrics.get('matched_line_reading_order_total_pairs')}",
        f"- structural line prediction/matched/unmatched counts: {metrics.get('structural_line_prediction_count')}/{metrics.get('structural_line_matched_count')}/{metrics.get('structural_line_unmatched_count')}",
        f"- structural line match/exact-text rates: {_fmt_metric(metrics.get('structural_line_match_rate'))}/{_fmt_metric(metrics.get('structural_line_exact_text_match_rate'))}",
        f"- zero-length text references: {metrics.get('zero_length_text_references')}",
        f"- uncertainty method: {uncertainty.get('method')}",
        f"- uncertainty scoring units: {uncertainty.get('effective_sample_size')}/{uncertainty.get('min_claim_scoring_units')}",
        f"- text grapheme CER 95% CI: {_fmt_metric(graph_interval.get('low'))}..{_fmt_metric(graph_interval.get('high'))}",
        "",
        "## Slices",
        "",
    ]
    for row in metrics.get("slices", []):
        if not isinstance(row, dict):
            continue
        row_means = row.get("metric_means") if isinstance(row.get("metric_means"), dict) else {}
        lines.append(
            f"- {row.get('language')} / {row.get('script')} / {row.get('domain')} / {row.get('gt_status')} / {row.get('split')}: "
            f"pages={row.get('scored_pages')} text_graph_CER={_fmt_metric(row.get('text_cer_grapheme_micro'))} "
            f"text_cp_CER={_fmt_metric(row.get('text_cer_codepoint_micro'))} "
            f"exact={_fmt_metric(row.get('text_exact_accuracy'))} line_F1={_fmt_metric(row_means.get('line_detection_f1', 0.0))}"
        )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Review Artifacts",
            "",
            "- command.txt",
            "- page-metrics.csv",
            "- reference-artifacts.tsv",
            "- text-pairs.tsv",
            "- missing.tsv",
            "- worst-pages.csv",
            "- sample-review.csv",
            "- sample-review.html",
            "- confusions.json",
            "- line-alignment.csv",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.eval_manifest)
    if not manifest:
        raise ValueError(f"eval manifest contains no rows: {args.eval_manifest}")
    _validate_manifest_identity(manifest)
    summary = _read_summary(args.summary_file)
    spotcheck_calibration = _read_spotcheck_calibration(args.spotcheck_calibration_file)
    manifest_dir = args.eval_manifest.parent
    rows: list[PageEvalRow] = []
    line_alignment_rows: list[LineAlignmentRow] = []
    reference_texts: list[str] = []
    metadata_contradictions: list[str] = []
    document_source_path_mismatches: list[str] = []
    missing_documents: list[str] = []
    missing_references: list[str] = []
    linearization_fallback_samples: list[str] = []
    ocr_output_kind = _ocr_output_kind(args)
    require_output_audit = _requires_page_output_audit(args, ocr_output_kind)
    output_audit_filename = _page_output_audit_filename(args)
    output_audit_passed_field = _page_output_audit_passed_field(args)
    output_audit_rows: list[dict[str, Any]] = []
    for entry in manifest:
        input_path = _resolve_manifest_path(entry.image_path, manifest_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"eval manifest input is missing for {entry.sample_id}: {input_path}")
        input_sha256 = _validate_manifest_image_hash(entry, input_path)
        entry_domain = _entry_domain(entry, args.domain)
        entry_gt_status = _entry_gt_status(entry, args.gt_status)
        metadata_contradictions.extend(_metadata_contradictions(entry.sample_id, entry.metadata or {}, entry_domain, entry_gt_status))
        document, document_path = _load_document(entry, input_path, args.documents_dir)
        if document is None or document_path is None:
            missing_documents.append(entry.sample_id)
            continue
        if require_output_audit:
            output_audit_rows.append(
                _read_page_output_audit(
                    entry.sample_id,
                    document_path.parent,
                    filename=output_audit_filename,
                    passed_field=output_audit_passed_field,
                )
            )
        source_path_mismatch = _document_source_path_mismatch(entry.sample_id, document, input_path)
        if source_path_mismatch:
            document_source_path_mismatches.append(source_path_mismatch)
        reference = _reference_for(entry, input_path, manifest_dir)
        if reference is None:
            missing_references.append(entry.sample_id)
            continue
        reference_path = _reference_path_for(entry, input_path, manifest_dir)
        reference_payload = _reference_payload_for(entry, input_path, manifest_dir)
        if isinstance(reference_payload, dict) and isinstance(reference_payload.get("metadata"), dict):
            metadata_contradictions.extend(
                _metadata_contradictions(entry.sample_id, reference_payload["metadata"], entry_domain, entry_gt_status)
            )
        reference_lines = _reference_lines(reference_payload)
        predicted_lines = _predicted_lines(document)
        line_alignment, line_rows = _align_lines(
            entry.sample_id,
            reference_lines,
            predicted_lines,
            args.line_iou_threshold,
            predicted_reading_order_ids=_predicted_reading_order_ids(document),
        )
        line_alignment_rows.extend(line_rows)
        reference_fields = _reference_fields(reference)
        if reference_lines:
            reference_fields.append("line_bboxes")
        metrics = score_document(document, document.text, reference)
        predicted_page_text = document.text
        reference_page_text = reference.text
        if args.page_linearization == "column-major":
            if reference_lines:
                reference_page_text = linearize_texts_column_major(
                    [((line.bbox.x, line.bbox.y, line.bbox.w, line.bbox.h), line.text) for line in reference_lines]
                )
                predicted_page_text = linearize_texts_column_major(
                    [((line.bbox.x, line.bbox.y, line.bbox.w, line.bbox.h), line.text) for line in predicted_lines]
                )
            else:
                # No reference line geometry: column-major is undefined for this
                # sample; fall back to native and record it, never silently.
                linearization_fallback_samples.append(entry.sample_id)
        cp_edits = cp_len = gr_edits = gr_len = 0
        if reference_page_text is not None:
            reference_texts.append(reference.text if reference.text is not None else reference_page_text)
            cp_edits, cp_len, gr_edits, gr_len = _score_text_micro(predicted_page_text, reference_page_text)
            text_exact = _nfc(predicted_page_text) == _nfc(reference_page_text)
        else:
            text_exact = False
        rows.append(
            PageEvalRow(
                sample_id=entry.sample_id,
                input_path=str(input_path),
                input_sha256=input_sha256,
                document_path=str(document_path),
                document_sha256=sha256_file(document_path),
                document_source_path=document.source_path,
                ocr_output_kind=ocr_output_kind,
                language=_entry_language(entry, args.language),
                script=_entry_script(entry, args.script),
                domain=entry_domain,
                gt_status=entry_gt_status,
                split=_entry_split(entry),
                claim_eligible=_entry_claim_eligible(entry, args.claim_eligible),
                source_document=_metadata_value_for_keys(entry.metadata or {}, SOURCE_DOCUMENT_KEYS),
                source_page=_metadata_value_for_keys(entry.metadata or {}, PAGE_KEYS),
                source_bbox=_metadata_value_for_keys(entry.metadata or {}, BBOX_KEYS),
                reference_path=str(reference_path) if reference_path else "",
                reference_sha256=sha256_file(reference_path) if reference_path else "",
                reference_status=str((reference.metadata or {}).get("reference_status") or ""),
                reference_claim_eligible=str((reference.metadata or {}).get("claim_evidence_eligible"))
                if (reference.metadata or {}).get("claim_evidence_eligible") is not None
                else "",
                slices=_entry_slices(entry),
                metrics=metrics,
                text_codepoint_edits=cp_edits,
                text_codepoint_ref_len=cp_len,
                text_grapheme_edits=gr_edits,
                text_grapheme_ref_len=gr_len,
                text_exact=text_exact,
                prediction_text=document.text,
                reference_text=reference.text or "",
                reference_fields=reference_fields,
                line_alignment=line_alignment,
            )
        )
    metrics = _aggregate(rows)
    metrics["manifest_samples"] = len(manifest)
    metrics["ocr_output_kind"] = ocr_output_kind
    metrics["page_output_audit"] = {
        "required": require_output_audit,
        "filename": output_audit_filename,
        "passed_field": output_audit_passed_field,
        "expected_count": metrics["scored_pages"] if require_output_audit else 0,
        "checked_count": len(output_audit_rows),
        "passed_count": sum(1 for row in output_audit_rows if row.get("passed") is True),
        "missing_count": sum(1 for row in output_audit_rows if row.get("exists") is not True),
        "failed_count": sum(1 for row in output_audit_rows if row.get("exists") is True and row.get("passed") is not True),
        "issues": [str(row.get("issue")) for row in output_audit_rows if row.get("issue")],
        "artifacts": output_audit_rows,
    }
    metrics["missing_documents"] = len(missing_documents)
    metrics["missing_references"] = len(missing_references)
    metrics["document_source_path_mismatch_count"] = len(document_source_path_mismatches)
    metrics["document_source_path_mismatches"] = document_source_path_mismatches
    metrics["claim_eligible_rows"] = sum(1 for entry in manifest if _entry_claim_eligible(entry, args.claim_eligible))
    metrics["slices"] = _slice_metrics(rows)
    metrics["reference_coverage"] = _coverage(rows)
    metrics["reference_claim_readiness"] = _reference_claim_readiness(rows)
    metrics["metadata_coverage"] = _metadata_coverage(manifest)
    metrics["metadata_contradictions"] = metadata_contradictions
    metrics["normalization_summary"] = _normalization_summary(rows)
    metrics["pack_content_sha256"] = _page_pack_content_digest(rows)
    script_inventory = _script_inventory_summary(reference_texts, args.script_inventory_file)
    metrics["script_inventory"] = script_inventory
    metrics["dictionary_coverage"] = _dictionary_coverage(reference_texts, args.dictionary)
    leakage = _leakage_report(args.train_manifest or [], args.eval_manifest)
    metrics["leakage"] = leakage
    if spotcheck_calibration:
        metrics["spotcheck_calibration"] = _spotcheck_calibration_summary(spotcheck_calibration)
    artifacts = _artifact_identities(args)
    warnings = _warnings(args, metrics, leakage, summary, spotcheck_calibration, artifacts, metadata_contradictions)
    claim_safe = not warnings
    if args.require_claim_safe and not claim_safe:
        raise ValueError("claim-safe page eval requested but warnings are present: " + "; ".join(warnings))
    out = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"output directory exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    languages = sorted({row.language for row in rows})
    scripts = sorted({row.script for row in rows})
    domains = sorted({row.domain for row in rows})
    gt_statuses = sorted({row.gt_status for row in rows})
    splits = sorted({row.split for row in rows})
    eval_run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_mode": "page_level_ocr",
        "page_linearization": args.page_linearization,
        "page_linearization_fallback_samples": linearization_fallback_samples,
        "system_name": args.system_name,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "dictionary": args.dictionary,
        "artifact_identity": artifacts,
        "eval_manifest": str(args.eval_manifest),
        "eval_manifest_sha256": sha256_file(args.eval_manifest),
        "summary_file": str(args.summary_file) if args.summary_file else None,
        "summary_file_sha256": sha256_file(args.summary_file) if args.summary_file else None,
        "script_inventory_file": str(args.script_inventory_file) if args.script_inventory_file else None,
        "script_inventory_file_sha256": sha256_file(args.script_inventory_file) if args.script_inventory_file else None,
        "spotcheck_calibration_file": str(args.spotcheck_calibration_file) if args.spotcheck_calibration_file else None,
        "spotcheck_calibration_file_sha256": sha256_file(args.spotcheck_calibration_file) if args.spotcheck_calibration_file else None,
        "spotcheck_calibration": metrics.get("spotcheck_calibration"),
        "train_manifests": [{"path": str(path), "sha256": sha256_file(path)} for path in (args.train_manifest or [])],
        "documents_dir": str(args.documents_dir),
        "ocr_output_kind": ocr_output_kind,
        "require_output_audit": require_output_audit,
        "output_audit_filename": output_audit_filename,
        "output_audit_passed_field": output_audit_passed_field,
        "page_output_audit": metrics["page_output_audit"],
        "eval_pack_version": args.eval_pack_version,
        "pack_content_sha256": metrics["pack_content_sha256"],
        "frozen_eval_pack": args.frozen_eval_pack,
        "unseen_for_selection": args.unseen_for_selection,
        "selected_checkpoint_on_test": args.selected_checkpoint_on_test,
        "claim_eligible_default": args.claim_eligible,
        "language_default": args.language,
        "script_default": args.script,
        "domain_default": args.domain,
        "gt_status_default": args.gt_status,
        "line_iou_threshold": args.line_iou_threshold,
        "normalization_profile": DEFAULT_NORMALIZATION,
        "languages": languages,
        "scripts": scripts,
        "domains": domains,
        "gt_statuses": gt_statuses,
        "splits": splits,
        "script_inventory": metrics["script_inventory"],
        "claim_safe": claim_safe,
        "warnings": warnings,
        "machine": {"platform": platform.platform(), "python": sys.version},
        "command": " ".join(sys.argv),
    }
    (out / "page-eval-run.json").write_text(json.dumps(eval_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_metrics_md(out / "metrics.md", metrics, warnings)
    _write_claim_safety_report(out / "claim-safety.md", eval_run, metrics)
    _write_rows(out / "page-metrics.csv", rows)
    _write_reference_artifacts(out / "reference-artifacts.tsv", rows)
    _write_text_pairs(out / "text-pairs.tsv", rows)
    _write_missing(out / "missing.tsv", missing_documents, missing_references)
    (out / "confusions.json").write_text(json.dumps(_confusions(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    worst = _write_worst(out / "worst-pages.csv", rows, args.review_sample_size)
    _write_review_csv(out / "sample-review.csv", worst)
    _write_review_html(out / "sample-review.html", worst)
    _write_line_alignment(out / "line-alignment.csv", line_alignment_rows)
    (out / "command.txt").write_text(eval_run["command"] + "\n", encoding="utf-8")
    return {"metrics": metrics, "eval_run": eval_run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument(
        "--ocr-output-kind",
        choices=sorted(PAGE_OCR_OUTPUT_KINDS),
        default="unknown",
        help="What kind of OCR document outputs are being scored: live detector boxes, oracle line boxes, sidecar fixture, or unknown.",
    )
    parser.add_argument("--require-output-audit", action="store_true", help="require every scored page output directory to contain a passing output audit JSON")
    parser.add_argument("--output-audit-filename", default="limbu-output-audit.json", help="per-page output audit JSON filename under each document output directory")
    parser.add_argument("--output-audit-passed-field", default="passed", help="boolean JSON field that must be true in each per-page output audit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--script-inventory-file", type=Path)
    parser.add_argument("--spotcheck-calibration-file", type=Path)
    parser.add_argument("--train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--language", default="unknown")
    parser.add_argument("--script", default="unknown")
    parser.add_argument("--domain", default="unknown")
    parser.add_argument("--gt-status", default="unknown")
    parser.add_argument("--eval-pack-version", default="")
    parser.add_argument("--frozen-eval-pack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unseen-for-selection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selected-checkpoint-on-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--claim-eligible", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--line-iou-threshold", type=float, default=0.5)
    parser.add_argument("--review-sample-size", type=int, default=50)
    parser.add_argument(
        "--page-linearization",
        choices=["native", "column-major"],
        default="native",
        help=(
            "How page text is linearized before page-level CER: 'native' compares "
            "document.text to reference.text as stored (v1 behavior); 'column-major' "
            "re-linearizes BOTH the reference lines and the predicted lines with the "
            "shared ocrtech.linearization.column_major_indices (page-eval v2). Scores "
            "are only comparable within one linearization."
        ),
    )
    parser.add_argument("--require-claim-safe", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_eval(args)
    metrics = result["metrics"]
    print(
        f"wrote {args.out} page_text_graph_CER={metrics['text_cer_grapheme_micro']:.3f} "
        f"page_text_cp_CER={metrics['text_cer_codepoint_micro']:.3f} scored={metrics['scored_pages']}/{metrics['manifest_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
