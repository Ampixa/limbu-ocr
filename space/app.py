from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps


APP_ROOT = Path(__file__).resolve().parent
SIRIJONGA_MODEL_DIR = APP_ROOT / "models" / "limbu-sirijonga-ft-v2" / "artifacts"
DEVANAGARI_MODEL_DIR = APP_ROOT / "models" / "deva-v2" / "artifacts"
DETECTOR_MODEL_PATH = APP_ROOT / "models" / "line-detector" / "best.pt"

LIMBU_RE = re.compile(r"[\u1900-\u194F]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


@dataclass(slots=True)
class Recognition:
    text: str
    confidence: float | None
    engine: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
        }


@dataclass(slots=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def x_center(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class PaddleDetection:
    box: Box
    seed_text: str
    seed_confidence: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.as_dict(),
            "seed_text": self.seed_text,
            "seed_confidence": self.seed_confidence,
        }


def _ensure_models_present(*, include_detector: bool = False) -> None:
    required = [
        SIRIJONGA_MODEL_DIR / "inference.json",
        SIRIJONGA_MODEL_DIR / "inference.pdiparams",
        SIRIJONGA_MODEL_DIR / "inference.yml",
        SIRIJONGA_MODEL_DIR / "limbu-sirijonga-v2.txt",
        DEVANAGARI_MODEL_DIR / "inference.json",
        DEVANAGARI_MODEL_DIR / "inference.pdiparams",
        DEVANAGARI_MODEL_DIR / "inference.yml",
        DEVANAGARI_MODEL_DIR / "devanagari-v2.txt",
    ]
    if include_detector:
        required.append(DETECTOR_MODEL_PATH)
    missing = [str(path.relative_to(APP_ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing bundled OCR artifacts: " + ", ".join(missing))


def _normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _save_temp_image(image: Image.Image) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="limbu-ocr-", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    _normalize_image(image).save(tmp_path)
    return tmp_path


@lru_cache(maxsize=1)
def _text_recognition_class() -> type[Any]:
    try:
        from paddleocr._models.text_recognition import TextRecognition
    except ImportError as exc:
        raise RuntimeError("PaddleOCR TextRecognition is unavailable in this runtime") from exc
    return TextRecognition


@lru_cache(maxsize=2)
def _recognizer(engine: str) -> Any:
    _ensure_models_present()
    cls = _text_recognition_class()
    if engine == "sirijonga":
        model_dir = SIRIJONGA_MODEL_DIR
    elif engine == "devanagari":
        model_dir = DEVANAGARI_MODEL_DIR
    else:
        raise ValueError(f"unknown recognizer engine: {engine}")
    return cls(
        model_name="PP-OCRv5_mobile_rec",
        model_dir=str(model_dir),
        device=os.environ.get("LIMBU_OCR_PADDLE_DEVICE", "cpu"),
    )


@lru_cache(maxsize=1)
def _detector() -> Any:
    _ensure_models_present(include_detector=True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is unavailable in this runtime") from exc
    return YOLO(str(DETECTOR_MODEL_PATH))


def _result_mapping(item: object) -> dict[str, object] | None:
    payload: object = item
    if not isinstance(payload, dict):
        payload = getattr(item, "json", None)
    if not isinstance(payload, dict):
        return None
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _recognize_with_engine(image_path: Path, engine: str) -> Recognition:
    raw_result = _recognizer(engine).predict(str(image_path))
    best_text = ""
    best_confidence: float | None = None
    for item in raw_result if isinstance(raw_result, list) else [raw_result]:
        payload = _result_mapping(item)
        if payload is None:
            continue
        text = payload.get("rec_text") or payload.get("text") or payload.get("transcription") or ""
        score = payload.get("rec_score") or payload.get("score") or payload.get("confidence")
        confidence = float(score) if score is not None else None
        text = str(text).strip()
        if not text:
            continue
        if best_confidence is None or (confidence is not None and confidence > best_confidence):
            best_text = text
            best_confidence = confidence
    return Recognition(text=best_text, confidence=best_confidence, engine=engine)


def _script_score(recognition: Recognition) -> float:
    text = recognition.text or ""
    confidence = recognition.confidence if recognition.confidence is not None else 0.0
    limbu_chars = len(LIMBU_RE.findall(text))
    deva_chars = len(DEVANAGARI_RE.findall(text))
    if recognition.engine == "sirijonga":
        return confidence + min(limbu_chars, 12) * 0.06 - min(deva_chars, 12) * 0.04
    if recognition.engine == "devanagari":
        return confidence + min(deva_chars, 12) * 0.035 - min(limbu_chars, 12) * 0.08
    return confidence


def _recognize_line_path(image_path: Path, mode: str) -> tuple[Recognition, list[Recognition]]:
    if mode == "Sirijonga":
        rec = _recognize_with_engine(image_path, "sirijonga")
        return rec, [rec]
    if mode == "Devanagari":
        rec = _recognize_with_engine(image_path, "devanagari")
        return rec, [rec]
    candidates = [
        _recognize_with_engine(image_path, "sirijonga"),
        _recognize_with_engine(image_path, "devanagari"),
    ]
    return max(candidates, key=_script_score), candidates


def _engine_key(mode: str) -> str:
    if mode == "Sirijonga":
        return "sirijonga"
    if mode == "Devanagari":
        return "devanagari"
    raise ValueError(f"unsupported Paddle seed recognizer: {mode}")


@lru_cache(maxsize=8)
def _paddle_page_ocr(
    seed_engine: str,
    text_det_limit_side_len: int,
    text_det_thresh: float,
    text_det_box_thresh: float,
    text_det_unclip_ratio: float,
) -> Any:
    _ensure_models_present()
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is unavailable in this runtime") from exc
    if seed_engine == "sirijonga":
        model_dir = SIRIJONGA_MODEL_DIR
    elif seed_engine == "devanagari":
        model_dir = DEVANAGARI_MODEL_DIR
    else:
        raise ValueError(f"unsupported Paddle seed engine: {seed_engine}")
    return PaddleOCR(
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(model_dir),
        text_det_limit_side_len=max(256, int(text_det_limit_side_len)),
        text_det_thresh=float(text_det_thresh),
        text_det_box_thresh=float(text_det_box_thresh),
        text_det_unclip_ratio=float(text_det_unclip_ratio),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _coerce_point_pair(value: object) -> tuple[float, float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _paddle_box_from_value(value: object, *, image_width: int, image_height: int, confidence: float | None) -> Box | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple) and len(value) == 4 and all(isinstance(item, int | float) for item in value):
        left, top, right, bottom = [float(item) for item in value]
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
    elif isinstance(value, list | tuple):
        points = [_coerce_point_pair(point) for point in value]
        clean_points = [point for point in points if point is not None]
        if not clean_points:
            return None
        left = min(point[0] for point in clean_points)
        top = min(point[1] for point in clean_points)
        right = max(point[0] for point in clean_points)
        bottom = max(point[1] for point in clean_points)
    else:
        return None
    box = Box(
        left=max(0, int(math.floor(left))),
        top=max(0, int(math.floor(top))),
        right=min(image_width, int(math.ceil(right))),
        bottom=min(image_height, int(math.ceil(bottom))),
        confidence=float(confidence) if confidence is not None else 1.0,
    )
    return box if box.width > 0 and box.height > 0 else None


def _as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def _first_payload_list(payload: dict[str, object], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        values = _as_list(payload.get(key))
        if values:
            return values
    return []


def _paddle_detections_from_result(raw_result: object, *, image_width: int, image_height: int) -> list[PaddleDetection]:
    candidates = raw_result
    if isinstance(raw_result, list) and len(raw_result) == 1 and isinstance(raw_result[0], list):
        candidates = raw_result[0]
    if not isinstance(candidates, list):
        candidates = [candidates]

    detections: list[PaddleDetection] = []
    for item in candidates:
        payload = _result_mapping(item)
        if payload is None:
            continue
        boxes = _first_payload_list(payload, ("rec_polys", "dt_polys", "rec_boxes", "dt_boxes"))
        texts = _first_payload_list(payload, ("rec_texts", "texts"))
        scores = _first_payload_list(payload, ("rec_scores", "scores"))
        for index, raw_box in enumerate(boxes):
            score: float | None = None
            if index < len(scores) and scores[index] is not None:
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = None
            box = _paddle_box_from_value(raw_box, image_width=image_width, image_height=image_height, confidence=score)
            if box is None:
                continue
            seed_text = str(texts[index]).strip() if index < len(texts) and texts[index] is not None else ""
            detections.append(PaddleDetection(box=box, seed_text=seed_text, seed_confidence=score))
    return detections


def _predict_paddle_page_detections(
    image: Image.Image,
    *,
    seed_recognizer: str,
    text_det_limit_side_len: int,
    text_det_thresh: float,
    text_det_box_thresh: float,
    text_det_unclip_ratio: float,
) -> list[PaddleDetection]:
    normalized = _normalize_image(image)
    page_path = _save_temp_image(normalized)
    try:
        ocr = _paddle_page_ocr(
            _engine_key(seed_recognizer),
            int(text_det_limit_side_len),
            float(text_det_thresh),
            float(text_det_box_thresh),
            float(text_det_unclip_ratio),
        )
        raw_result = ocr.predict(str(page_path))
    finally:
        page_path.unlink(missing_ok=True)
    return _paddle_detections_from_result(raw_result, image_width=normalized.width, image_height=normalized.height)


def recognize_line(image: Image.Image | None, mode: str) -> tuple[str, str]:
    if image is None:
        return "", "Upload a cropped text line image."
    path = _save_temp_image(image)
    try:
        best, candidates = _recognize_line_path(path, mode)
    finally:
        path.unlink(missing_ok=True)
    details = {
        "selected_engine": best.engine,
        "selected_confidence": best.confidence,
        "candidates": [candidate.as_dict() for candidate in candidates],
    }
    return best.text, json.dumps(details, ensure_ascii=False, indent=2)


def _detect_boxes(image_path: Path, confidence: float) -> list[Box]:
    result = _detector().predict(
        source=str(image_path),
        imgsz=1280,
        conf=float(confidence),
        verbose=False,
    )[0]
    boxes: list[Box] = []
    raw_boxes = getattr(result, "boxes", None)
    if raw_boxes is None:
        return boxes
    xyxy_values = raw_boxes.xyxy.cpu().numpy() if hasattr(raw_boxes.xyxy, "cpu") else np.asarray(raw_boxes.xyxy)
    conf_values = raw_boxes.conf.cpu().numpy() if hasattr(raw_boxes.conf, "cpu") else np.asarray(raw_boxes.conf)
    for xyxy, conf in zip(xyxy_values, conf_values, strict=False):
        left, top, right, bottom = [int(round(float(value))) for value in xyxy[:4]]
        if right <= left or bottom <= top:
            continue
        boxes.append(Box(left=left, top=top, right=right, bottom=bottom, confidence=float(conf)))
    return boxes


def _order_boxes(boxes: list[Box], mode: str, image_width: int) -> list[Box]:
    if mode == "Two columns LTR":
        return sorted(boxes, key=lambda box: (0 if box.x_center < image_width / 2 else 1, box.y_center, box.x_center))
    if mode == "Two columns RTL":
        return sorted(boxes, key=lambda box: (0 if box.x_center >= image_width / 2 else 1, box.y_center, box.x_center))
    row_height = max(12, int(np.median([box.height for box in boxes])) if boxes else 24)
    return sorted(boxes, key=lambda box: (math.floor(box.y_center / row_height), box.x_center))


def _crop_box(image: Image.Image, box: Box, padding: int) -> Image.Image:
    left = max(0, box.left - padding)
    top = max(0, box.top - padding)
    right = min(image.width, box.right + padding)
    bottom = min(image.height, box.bottom + padding)
    return image.crop((left, top, right, bottom))


def _draw_boxes(image: Image.Image, boxes: list[Box]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for index, box in enumerate(boxes, start=1):
        draw.rectangle((box.left, box.top, box.right, box.bottom), outline=(28, 120, 92), width=3)
        draw.text((box.left + 4, max(0, box.top - 18)), str(index), fill=(28, 120, 92))
    return annotated


def _box_iou(left: Box, right: Box) -> float:
    overlap_left = max(left.left, right.left)
    overlap_top = max(left.top, right.top)
    overlap_right = min(left.right, right.right)
    overlap_bottom = min(left.bottom, right.bottom)
    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    intersection = overlap_width * overlap_height
    if intersection <= 0:
        return 0.0
    left_area = max(1, left.width * left.height)
    right_area = max(1, right.width * right.height)
    return intersection / float(left_area + right_area - intersection)


def _merge_overlapping_boxes(boxes: list[Box], threshold: float = 0.80) -> list[Box]:
    merged: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        if any(_box_iou(box, kept) >= threshold for kept in merged):
            continue
        merged.append(box)
    return merged


def _cv2_threshold(gray: Image.Image, threshold: int) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is unavailable; deterministic page OCR requires opencv-python-headless") from exc
    pixels = np.asarray(gray)
    if threshold <= 0:
        _value, binary = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _value, binary = cv2.threshold(pixels, int(threshold), 255, cv2.THRESH_BINARY_INV)
    return cv2, binary


def _detect_image_line_boxes(
    image: Image.Image,
    *,
    threshold: int,
    horizontal_kernel: int,
    vertical_kernel: int,
    dilation_iterations: int,
    min_width: int,
    min_height: int,
    max_height: int,
    min_area: int,
    padding: int,
) -> list[Box]:
    normalized = _normalize_image(image)
    gray = ImageOps.grayscale(normalized)
    cv2, binary = _cv2_threshold(gray, threshold)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, int(horizontal_kernel)), max(1, int(vertical_kernel))),
    )
    dilated = cv2.dilate(binary, kernel, iterations=max(1, int(dilation_iterations)))
    contours, _hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = int(width * height)
        if width < min_width or height < min_height or area < min_area:
            continue
        if max_height > 0 and height > max_height:
            continue
        left = max(0, int(x) - padding)
        top = max(0, int(y) - padding)
        right = min(normalized.width, int(x + width) + padding)
        bottom = min(normalized.height, int(y + height) + padding)
        if right <= left or bottom <= top:
            continue
        boxes.append(Box(left=left, top=top, right=right, bottom=bottom, confidence=1.0))
    return _merge_overlapping_boxes(boxes)


def _has_two_column_shape(boxes: list[Box], image_width: int) -> bool:
    if len(boxes) < 8 or image_width <= 0:
        return False
    midpoint = image_width / 2.0
    left = [box for box in boxes if box.x_center < midpoint]
    right = [box for box in boxes if box.x_center >= midpoint]
    if len(left) < 3 or len(right) < 3:
        return False
    left_median = float(np.median([box.x_center for box in left]))
    right_median = float(np.median([box.x_center for box in right]))
    return (right_median - left_median) >= image_width * 0.22


def _order_image_line_boxes(boxes: list[Box], mode: str, image_width: int) -> list[Box]:
    if mode == "Auto":
        mode = "Columns LTR" if _has_two_column_shape(boxes, image_width) else "Rows LTR"
    if mode == "Columns LTR":
        return sorted(boxes, key=lambda box: (0 if box.x_center < image_width / 2 else 1, box.y_center, box.x_center))
    if mode == "Columns RTL":
        return sorted(boxes, key=lambda box: (0 if box.x_center >= image_width / 2 else 1, box.y_center, box.x_center))
    row_height = max(12, int(np.median([box.height for box in boxes])) if boxes else 24)
    return sorted(boxes, key=lambda box: (math.floor(box.y_center / row_height), box.x_center))


def _recognize_page_core(
    image: Image.Image,
    *,
    mode: str,
    reading_order: str,
    threshold: int,
    horizontal_kernel: int,
    vertical_kernel: int,
    dilation_iterations: int,
    min_width: int,
    min_height: int,
    max_height: int,
    min_area: int,
    crop_padding: int,
    max_lines: int,
    artifact_dir: Path | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    normalized = _normalize_image(image)
    boxes = _detect_image_line_boxes(
        normalized,
        threshold=threshold,
        horizontal_kernel=horizontal_kernel,
        vertical_kernel=vertical_kernel,
        dilation_iterations=dilation_iterations,
        min_width=min_width,
        min_height=min_height,
        max_height=max_height,
        min_area=min_area,
        padding=max(0, int(crop_padding // 2)),
    )
    boxes = _order_image_line_boxes(boxes, reading_order, normalized.width)
    if max_lines > 0:
        boxes = boxes[: int(max_lines)]

    lines: list[str] = []
    gallery: list[tuple[Image.Image, str]] = []
    rows: list[dict[str, Any]] = []
    crop_dir = artifact_dir / "crops" / f"page-{page_index + 1:04d}" if artifact_dir is not None else None
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="limbu-page-image-line-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, box in enumerate(boxes, start=1):
            crop = _crop_box(normalized, box, int(crop_padding))
            crop_path = tmp_dir / f"line-{index:04d}.png"
            crop.save(crop_path)
            best, candidates = _recognize_line_path(crop_path, mode)
            if best.text:
                lines.append(best.text)
            saved_crop_path: str | None = None
            if crop_dir is not None:
                saved_crop = crop_dir / f"line-{index:04d}.png"
                crop.save(saved_crop)
                saved_crop_path = str(saved_crop.relative_to(artifact_dir))
            rows.append(
                {
                    "page_index": page_index,
                    "index": index,
                    "box": box.as_dict(),
                    "crop_path": saved_crop_path,
                    "selected_engine": best.engine,
                    "selected_confidence": best.confidence,
                    "text": best.text,
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
            caption = f"{index}. {best.text}" if best.text else f"{index}. <empty>"
            gallery.append((crop, caption))

    annotated = _draw_boxes(normalized, boxes)
    annotated_path: str | None = None
    if artifact_dir is not None:
        page_dir = artifact_dir / "annotated-pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        target = page_dir / f"page-{page_index + 1:04d}.png"
        annotated.save(target)
        annotated_path = str(target.relative_to(artifact_dir))
    return {
        "text": "\n".join(lines),
        "annotated": annotated,
        "annotated_path": annotated_path,
        "gallery": gallery,
        "rows": rows,
        "line_count": len(boxes),
        "detector": "deterministic image-line projection",
        "reading_order": reading_order,
        "threshold": threshold,
        "horizontal_kernel": horizontal_kernel,
        "vertical_kernel": vertical_kernel,
        "dilation_iterations": dilation_iterations,
    }


def _recognize_page_paddle_core(
    image: Image.Image,
    *,
    mode: str,
    seed_recognizer: str,
    reading_order: str,
    text_det_limit_side_len: int,
    text_det_thresh: float,
    text_det_box_thresh: float,
    text_det_unclip_ratio: float,
    crop_padding: int,
    max_lines: int,
    artifact_dir: Path | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    normalized = _normalize_image(image)
    detections = _predict_paddle_page_detections(
        normalized,
        seed_recognizer=seed_recognizer,
        text_det_limit_side_len=text_det_limit_side_len,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=text_det_unclip_ratio,
    )
    ordered_boxes = _order_image_line_boxes([detection.box for detection in detections], reading_order, normalized.width)
    detection_by_box = {(detection.box.left, detection.box.top, detection.box.right, detection.box.bottom): detection for detection in detections}
    ordered_detections = [
        detection_by_box[(box.left, box.top, box.right, box.bottom)]
        for box in ordered_boxes
        if (box.left, box.top, box.right, box.bottom) in detection_by_box
    ]
    if max_lines > 0:
        ordered_detections = ordered_detections[: int(max_lines)]

    lines: list[str] = []
    gallery: list[tuple[Image.Image, str]] = []
    rows: list[dict[str, Any]] = []
    crop_dir = artifact_dir / "crops" / f"page-{page_index + 1:04d}" if artifact_dir is not None else None
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="limbu-page-paddle-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, detection in enumerate(ordered_detections, start=1):
            crop = _crop_box(normalized, detection.box, int(crop_padding))
            crop_path = tmp_dir / f"line-{index:04d}.png"
            crop.save(crop_path)
            best, candidates = _recognize_line_path(crop_path, mode)
            if best.text:
                lines.append(best.text)
            saved_crop_path: str | None = None
            if crop_dir is not None:
                saved_crop = crop_dir / f"line-{index:04d}.png"
                crop.save(saved_crop)
                saved_crop_path = str(saved_crop.relative_to(artifact_dir))
            rows.append(
                {
                    "page_index": page_index,
                    "index": index,
                    "box": detection.box.as_dict(),
                    "crop_path": saved_crop_path,
                    "paddle_seed_recognizer": seed_recognizer,
                    "paddle_seed_text": detection.seed_text,
                    "paddle_seed_confidence": detection.seed_confidence,
                    "selected_engine": best.engine,
                    "selected_confidence": best.confidence,
                    "text": best.text,
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
            caption = f"{index}. {best.text}" if best.text else f"{index}. <empty>"
            gallery.append((crop, caption))

    boxes = [detection.box for detection in ordered_detections]
    annotated = _draw_boxes(normalized, boxes)
    annotated_path: str | None = None
    if artifact_dir is not None:
        page_dir = artifact_dir / "annotated-pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        target = page_dir / f"page-{page_index + 1:04d}.png"
        annotated.save(target)
        annotated_path = str(target.relative_to(artifact_dir))
    return {
        "text": "\n".join(lines),
        "annotated": annotated,
        "annotated_path": annotated_path,
        "gallery": gallery,
        "rows": rows,
        "line_count": len(ordered_detections),
        "detector": "PaddleOCR PP-OCRv5 full-page text detector",
        "reading_order": reading_order,
        "seed_recognizer": seed_recognizer,
        "text_det_limit_side_len": text_det_limit_side_len,
        "text_det_thresh": text_det_thresh,
        "text_det_box_thresh": text_det_box_thresh,
        "text_det_unclip_ratio": text_det_unclip_ratio,
    }


def recognize_page_paddle(
    image: Image.Image | None,
    mode: str,
    seed_recognizer: str,
    reading_order: str,
    text_det_limit_side_len: int,
    text_det_thresh: float,
    text_det_box_thresh: float,
    text_det_unclip_ratio: float,
    crop_padding: int,
    max_lines: int,
) -> tuple[str, Image.Image | None, list[tuple[Image.Image, str]], str]:
    if image is None:
        return "", None, [], "Upload a page image."

    result = _recognize_page_paddle_core(
        image,
        mode=mode,
        seed_recognizer=seed_recognizer,
        reading_order=reading_order,
        text_det_limit_side_len=text_det_limit_side_len,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=text_det_unclip_ratio,
        crop_padding=crop_padding,
        max_lines=max_lines,
    )
    details = {
        "detector": result["detector"],
        "reading_order": result["reading_order"],
        "line_count": result["line_count"],
        "recognition_mode": mode,
        "paddle_seed_recognizer": result["seed_recognizer"],
        "paddle": {
            "text_det_limit_side_len": result["text_det_limit_side_len"],
            "text_det_thresh": result["text_det_thresh"],
            "text_det_box_thresh": result["text_det_box_thresh"],
            "text_det_unclip_ratio": result["text_det_unclip_ratio"],
        },
        "segmentation": {
            "crop_padding": crop_padding,
        },
        "rows": result["rows"],
    }
    return result["text"], result["annotated"], result["gallery"], json.dumps(details, ensure_ascii=False, indent=2)


def recognize_page(
    image: Image.Image | None,
    mode: str,
    reading_order: str,
    threshold: int,
    horizontal_kernel: int,
    vertical_kernel: int,
    dilation_iterations: int,
    min_width: int,
    min_height: int,
    max_height: int,
    min_area: int,
    crop_padding: int,
    max_lines: int,
) -> tuple[str, Image.Image | None, list[tuple[Image.Image, str]], str]:
    if image is None:
        return "", None, [], "Upload a page image."

    result = _recognize_page_core(
        image,
        mode=mode,
        reading_order=reading_order,
        threshold=threshold,
        horizontal_kernel=horizontal_kernel,
        vertical_kernel=vertical_kernel,
        dilation_iterations=dilation_iterations,
        min_width=min_width,
        min_height=min_height,
        max_height=max_height,
        min_area=min_area,
        crop_padding=crop_padding,
        max_lines=max_lines,
    )
    details = {
        "detector": result["detector"],
        "reading_order": result["reading_order"],
        "line_count": result["line_count"],
        "recognition_mode": mode,
        "segmentation": {
            "threshold": result["threshold"],
            "horizontal_kernel": result["horizontal_kernel"],
            "vertical_kernel": result["vertical_kernel"],
            "dilation_iterations": result["dilation_iterations"],
            "min_width": min_width,
            "min_height": min_height,
            "max_height": max_height,
            "min_area": min_area,
            "crop_padding": crop_padding,
        },
        "rows": result["rows"],
    }
    return result["text"], result["annotated"], result["gallery"], json.dumps(details, ensure_ascii=False, indent=2)


def recognize_page_yolo(
    image: Image.Image | None,
    mode: str,
    detector_confidence: float,
    reading_order: str,
    crop_padding: int,
    max_lines: int,
) -> tuple[str, Image.Image | None, list[tuple[Image.Image, str]], str]:
    if image is None:
        return "", None, [], "Upload a page image."

    normalized = _normalize_image(image)
    page_path = _save_temp_image(normalized)
    try:
        boxes = _detect_boxes(page_path, detector_confidence)
    finally:
        page_path.unlink(missing_ok=True)

    boxes = _order_boxes(boxes, reading_order, normalized.width)
    if max_lines > 0:
        boxes = boxes[: int(max_lines)]

    lines: list[str] = []
    gallery: list[tuple[Image.Image, str]] = []
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="limbu-page-ocr-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, box in enumerate(boxes, start=1):
            crop = _crop_box(normalized, box, int(crop_padding))
            crop_path = tmp_dir / f"line-{index:04d}.png"
            crop.save(crop_path)
            best, candidates = _recognize_line_path(crop_path, mode)
            if best.text:
                lines.append(best.text)
            rows.append(
                {
                    "index": index,
                    "box": box.as_dict(),
                    "selected_engine": best.engine,
                    "selected_confidence": best.confidence,
                    "text": best.text,
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
            caption = f"{index}. {best.text}" if best.text else f"{index}. <empty>"
            gallery.append((crop, caption))

    text = "\n".join(lines)
    annotated = _draw_boxes(normalized, boxes)
    details = {
        "detector": "YOLO cdc-grade1-allbooks-v1",
        "detector_confidence": detector_confidence,
        "reading_order": reading_order,
        "line_count": len(boxes),
        "recognition_mode": mode,
        "rows": rows,
        "warning": "Prototype page OCR. Detector was trained on rendered/augmented CDC pages; verify output before using it as text.",
    }
    return text, annotated, gallery, json.dumps(details, ensure_ascii=False, indent=2)


def _uploaded_path(item: object) -> Path:
    if isinstance(item, str | Path):
        return Path(item)
    name = getattr(item, "name", None)
    if isinstance(name, str):
        return Path(name)
    path = getattr(item, "path", None)
    if isinstance(path, str):
        return Path(path)
    raise ValueError(f"unsupported uploaded file object: {type(item).__name__}")


def _render_pdf_pages(path: Path, *, dpi: int, max_pages: int) -> list[tuple[str, Image.Image]]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PDF input requires pypdfium2 in the Space runtime") from exc
    pages: list[tuple[str, Image.Image]] = []
    pdf = pdfium.PdfDocument(str(path))
    limit = len(pdf) if max_pages <= 0 else min(len(pdf), max_pages)
    scale = max(72, int(dpi)) / 72.0
    for page_index in range(limit):
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        pages.append((f"{path.name}:p{page_index + 1}", bitmap.to_pil().convert("RGB")))
    return pages


def _load_book_pages(files: list[object] | None, *, pdf_dpi: int, max_pages: int) -> list[tuple[str, Image.Image]]:
    if not files:
        return []
    pages: list[tuple[str, Image.Image]] = []
    for item in files:
        path = _uploaded_path(item)
        suffix = path.suffix.lower()
        remaining = 0 if max_pages <= 0 else max_pages - len(pages)
        if max_pages > 0 and remaining <= 0:
            break
        if suffix == ".pdf":
            pages.extend(_render_pdf_pages(path, dpi=pdf_dpi, max_pages=remaining))
            continue
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"unsupported book input file type: {path.name}")
        with Image.open(path) as image:
            pages.append((path.name, _normalize_image(image)))
    return pages[:max_pages] if max_pages > 0 else pages


def _zip_directory(source_dir: Path) -> Path:
    zip_path = Path(tempfile.NamedTemporaryFile(prefix="limbu-book-ocr-", suffix=".zip", delete=False).name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return zip_path


def recognize_book_paddle(
    files: list[object] | None,
    mode: str,
    seed_recognizer: str,
    reading_order: str,
    text_det_limit_side_len: int,
    text_det_thresh: float,
    text_det_box_thresh: float,
    text_det_unclip_ratio: float,
    crop_padding: int,
    max_lines_per_page: int,
    max_pages: int,
    pdf_dpi: int,
) -> tuple[str, str | None, str]:
    pages = _load_book_pages(files, pdf_dpi=pdf_dpi, max_pages=max_pages)
    if not pages:
        return "", None, "Upload page images or a PDF."

    artifact_dir = Path(tempfile.mkdtemp(prefix="limbu-book-paddle-artifacts-"))
    page_payloads: list[dict[str, Any]] = []
    page_texts: list[str] = []
    try:
        for page_index, (source_name, image) in enumerate(pages):
            page_result = _recognize_page_paddle_core(
                image,
                mode=mode,
                seed_recognizer=seed_recognizer,
                reading_order=reading_order,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                crop_padding=crop_padding,
                max_lines=max_lines_per_page,
                artifact_dir=artifact_dir,
                page_index=page_index,
            )
            page_text = page_result["text"]
            page_texts.append(page_text)
            page_payloads.append(
                {
                    "page_index": page_index,
                    "source_name": source_name,
                    "line_count": page_result["line_count"],
                    "annotated_path": page_result["annotated_path"],
                    "text": page_text,
                    "lines": page_result["rows"],
                }
            )
        book_text = "\n\n".join(text for text in page_texts if text.strip())
        payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "system": "limbu-book-ocr-paddle-v1",
            "detector": "PaddleOCR PP-OCRv5 full-page text detector",
            "recognition_mode": mode,
            "paddle_seed_recognizer": seed_recognizer,
            "reading_order": reading_order,
            "page_count": len(page_payloads),
            "line_count": sum(int(page["line_count"]) for page in page_payloads),
            "paddle": {
                "text_det_limit_side_len": text_det_limit_side_len,
                "text_det_thresh": text_det_thresh,
                "text_det_box_thresh": text_det_box_thresh,
                "text_det_unclip_ratio": text_det_unclip_ratio,
            },
            "segmentation": {
                "crop_padding": crop_padding,
            },
            "pages": page_payloads,
        }
        (artifact_dir / "book.txt").write_text(book_text + ("\n" if book_text else ""), encoding="utf-8")
        (artifact_dir / "book.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        zip_path = _zip_directory(artifact_dir)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    details = {
        "page_count": len(page_payloads),
        "line_count": sum(int(page["line_count"]) for page in page_payloads),
        "artifact_zip": str(zip_path),
        "warning": "Paddle-first operational OCR output with crop evidence; benchmark claims still require frozen-page scoring.",
    }
    return book_text, str(zip_path), json.dumps(details, ensure_ascii=False, indent=2)


def recognize_book(
    files: list[object] | None,
    mode: str,
    reading_order: str,
    threshold: int,
    horizontal_kernel: int,
    vertical_kernel: int,
    dilation_iterations: int,
    min_width: int,
    min_height: int,
    max_height: int,
    min_area: int,
    crop_padding: int,
    max_lines_per_page: int,
    max_pages: int,
    pdf_dpi: int,
) -> tuple[str, str | None, str]:
    pages = _load_book_pages(files, pdf_dpi=pdf_dpi, max_pages=max_pages)
    if not pages:
        return "", None, "Upload page images or a PDF."

    artifact_dir = Path(tempfile.mkdtemp(prefix="limbu-book-ocr-artifacts-"))
    page_payloads: list[dict[str, Any]] = []
    page_texts: list[str] = []
    try:
        for page_index, (source_name, image) in enumerate(pages):
            page_result = _recognize_page_core(
                image,
                mode=mode,
                reading_order=reading_order,
                threshold=threshold,
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                dilation_iterations=dilation_iterations,
                min_width=min_width,
                min_height=min_height,
                max_height=max_height,
                min_area=min_area,
                crop_padding=crop_padding,
                max_lines=max_lines_per_page,
                artifact_dir=artifact_dir,
                page_index=page_index,
            )
            page_text = page_result["text"]
            page_texts.append(page_text)
            page_payloads.append(
                {
                    "page_index": page_index,
                    "source_name": source_name,
                    "line_count": page_result["line_count"],
                    "annotated_path": page_result["annotated_path"],
                    "text": page_text,
                    "lines": page_result["rows"],
                }
            )
        book_text = "\n\n".join(text for text in page_texts if text.strip())
        payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "system": "limbu-book-ocr-v1",
            "detector": "deterministic image-line projection",
            "recognition_mode": mode,
            "reading_order": reading_order,
            "page_count": len(page_payloads),
            "line_count": sum(int(page["line_count"]) for page in page_payloads),
            "segmentation": {
                "threshold": threshold,
                "horizontal_kernel": horizontal_kernel,
                "vertical_kernel": vertical_kernel,
                "dilation_iterations": dilation_iterations,
                "min_width": min_width,
                "min_height": min_height,
                "max_height": max_height,
                "min_area": min_area,
                "crop_padding": crop_padding,
            },
            "pages": page_payloads,
        }
        (artifact_dir / "book.txt").write_text(book_text + ("\n" if book_text else ""), encoding="utf-8")
        (artifact_dir / "book.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        zip_path = _zip_directory(artifact_dir)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    details = {
        "page_count": len(page_payloads),
        "line_count": sum(int(page["line_count"]) for page in page_payloads),
        "artifact_zip": str(zip_path),
        "warning": "This is operational OCR output with crop evidence, not a benchmark-quality claim.",
    }
    return book_text, str(zip_path), json.dumps(details, ensure_ascii=False, indent=2)


def health() -> str:
    status: dict[str, Any] = {
        "sirijonga_model": SIRIJONGA_MODEL_DIR.exists(),
        "devanagari_model": DEVANAGARI_MODEL_DIR.exists(),
        "line_detector": DETECTOR_MODEL_PATH.exists(),
        "paddle_device": os.environ.get("LIMBU_OCR_PADDLE_DEVICE", "cpu"),
    }
    try:
        _ensure_models_present()
        status["artifact_check"] = "pass"
    except Exception as exc:
        status["artifact_check"] = f"fail: {type(exc).__name__}: {exc}"
    return json.dumps(status, ensure_ascii=False, indent=2)


CSS = """
.output-markdown textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
"""


with gr.Blocks(title="Limbu OCR", css=CSS) as demo:
    gr.Markdown(
        """
# Limbu OCR

OCR research prototype with a Limbu-specific Sirijonga recognizer and a generic
Devanagari route intended for future Devanagari-written Limbu validation. The primary
page and book paths use PaddleOCR full-page text detection and preserve crop evidence
beside draft Unicode output.
"""
    )

    with gr.Tab("Line OCR"):
        with gr.Row():
            line_image = gr.Image(type="pil", label="Cropped text line")
            with gr.Column():
                line_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                line_button = gr.Button("Recognize Line", variant="primary")
        line_text = gr.Textbox(label="Unicode text", lines=4)
        line_details = gr.Code(label="Details", language="json")
        line_button.click(recognize_line, inputs=[line_image, line_mode], outputs=[line_text, line_details])

    with gr.Tab("Paddle Page OCR"):
        with gr.Row():
            paddle_page_image = gr.Image(type="pil", label="Page image")
            with gr.Column():
                paddle_page_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                paddle_page_seed = gr.Radio(["Sirijonga", "Devanagari"], value="Sirijonga", label="Paddle seed recognizer")
                paddle_page_order = gr.Radio(
                    ["Auto", "Rows LTR", "Columns LTR", "Columns RTL"],
                    value="Rows LTR",
                    label="Reading order",
                )
                paddle_page_limit = gr.Slider(512, 2048, value=1280, step=64, label="Paddle detector side limit")
                paddle_page_thresh = gr.Slider(0.10, 0.90, value=0.30, step=0.05, label="Paddle det threshold")
                paddle_page_box_thresh = gr.Slider(0.10, 0.90, value=0.60, step=0.05, label="Paddle box threshold")
                paddle_page_unclip = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Paddle unclip ratio")
                paddle_page_crop_padding = gr.Slider(0, 32, value=12, step=1, label="Crop padding")
                paddle_page_max_lines = gr.Slider(0, 300, value=0, step=1, label="Max lines (0 = no cap)")
                paddle_page_button = gr.Button("Run Paddle Page OCR", variant="primary")
        paddle_page_text = gr.Textbox(label="Draft Unicode text", lines=16)
        paddle_page_annotated = gr.Image(type="pil", label="Paddle detected lines")
        paddle_page_gallery = gr.Gallery(label="Line crops", columns=2, height=420)
        paddle_page_details = gr.Code(label="Details", language="json")
        paddle_page_button.click(
            recognize_page_paddle,
            inputs=[
                paddle_page_image,
                paddle_page_mode,
                paddle_page_seed,
                paddle_page_order,
                paddle_page_limit,
                paddle_page_thresh,
                paddle_page_box_thresh,
                paddle_page_unclip,
                paddle_page_crop_padding,
                paddle_page_max_lines,
            ],
            outputs=[paddle_page_text, paddle_page_annotated, paddle_page_gallery, paddle_page_details],
        )

    with gr.Tab("Paddle Book OCR"):
        with gr.Row():
            paddle_book_files = gr.Files(
                label="Page images or PDF",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"],
            )
            with gr.Column():
                paddle_book_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                paddle_book_seed = gr.Radio(["Sirijonga", "Devanagari"], value="Sirijonga", label="Paddle seed recognizer")
                paddle_book_order = gr.Radio(
                    ["Auto", "Rows LTR", "Columns LTR", "Columns RTL"],
                    value="Rows LTR",
                    label="Reading order",
                )
                paddle_book_limit = gr.Slider(512, 2048, value=1280, step=64, label="Paddle detector side limit")
                paddle_book_thresh = gr.Slider(0.10, 0.90, value=0.30, step=0.05, label="Paddle det threshold")
                paddle_book_box_thresh = gr.Slider(0.10, 0.90, value=0.60, step=0.05, label="Paddle box threshold")
                paddle_book_unclip = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Paddle unclip ratio")
                paddle_book_crop_padding = gr.Slider(0, 32, value=12, step=1, label="Crop padding")
                paddle_book_max_lines = gr.Slider(0, 300, value=0, step=1, label="Max lines per page (0 = no cap)")
                paddle_book_max_pages = gr.Slider(0, 80, value=20, step=1, label="Max pages (0 = no cap)")
                paddle_book_pdf_dpi = gr.Slider(120, 360, value=240, step=10, label="PDF render DPI")
                paddle_book_button = gr.Button("Run Paddle Book OCR", variant="primary")
        paddle_book_text = gr.Textbox(label="Unicode text", lines=18)
        paddle_book_zip = gr.File(label="OCR artifact zip")
        paddle_book_details = gr.Code(label="Details", language="json")
        paddle_book_button.click(
            recognize_book_paddle,
            inputs=[
                paddle_book_files,
                paddle_book_mode,
                paddle_book_seed,
                paddle_book_order,
                paddle_book_limit,
                paddle_book_thresh,
                paddle_book_box_thresh,
                paddle_book_unclip,
                paddle_book_crop_padding,
                paddle_book_max_lines,
                paddle_book_max_pages,
                paddle_book_pdf_dpi,
            ],
            outputs=[paddle_book_text, paddle_book_zip, paddle_book_details],
        )

    with gr.Tab("Image-Line Page Fallback"):
        with gr.Row():
            page_image = gr.Image(type="pil", label="Page image")
            with gr.Column():
                page_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                reading_order = gr.Radio(
                    ["Auto", "Rows LTR", "Columns LTR", "Columns RTL"],
                    value="Auto",
                    label="Reading order",
                )
                threshold = gr.Slider(0, 255, value=0, step=1, label="Ink threshold (0 = Otsu)")
                horizontal_kernel = gr.Slider(3, 81, value=23, step=2, label="Horizontal merge")
                vertical_kernel = gr.Slider(1, 15, value=3, step=1, label="Vertical merge")
                dilation_iterations = gr.Slider(1, 4, value=1, step=1, label="Dilation passes")
                min_width = gr.Slider(5, 300, value=35, step=5, label="Min line width")
                min_height = gr.Slider(4, 80, value=10, step=1, label="Min line height")
                max_height = gr.Slider(0, 260, value=140, step=5, label="Max line height (0 = no cap)")
                min_area = gr.Slider(20, 5000, value=100, step=20, label="Min line area")
                crop_padding = gr.Slider(0, 32, value=12, step=1, label="Crop padding")
                max_lines = gr.Slider(0, 300, value=180, step=1, label="Max lines (0 = no cap)")
                page_button = gr.Button("Run Image-Line Page OCR", variant="secondary")
        page_text = gr.Textbox(label="Draft Unicode text", lines=16)
        annotated_image = gr.Image(type="pil", label="Detected lines")
        crop_gallery = gr.Gallery(label="Line crops", columns=2, height=420)
        page_details = gr.Code(label="Details", language="json")
        page_button.click(
            recognize_page,
            inputs=[
                page_image,
                page_mode,
                reading_order,
                threshold,
                horizontal_kernel,
                vertical_kernel,
                dilation_iterations,
                min_width,
                min_height,
                max_height,
                min_area,
                crop_padding,
                max_lines,
            ],
            outputs=[page_text, annotated_image, crop_gallery, page_details],
        )

    with gr.Tab("Image-Line Book Fallback"):
        with gr.Row():
            book_files = gr.Files(
                label="Page images or PDF",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"],
            )
            with gr.Column():
                book_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                book_order = gr.Radio(
                    ["Auto", "Rows LTR", "Columns LTR", "Columns RTL"],
                    value="Auto",
                    label="Reading order",
                )
                book_threshold = gr.Slider(0, 255, value=0, step=1, label="Ink threshold (0 = Otsu)")
                book_horizontal_kernel = gr.Slider(3, 81, value=23, step=2, label="Horizontal merge")
                book_vertical_kernel = gr.Slider(1, 15, value=3, step=1, label="Vertical merge")
                book_dilation_iterations = gr.Slider(1, 4, value=1, step=1, label="Dilation passes")
                book_min_width = gr.Slider(5, 300, value=35, step=5, label="Min line width")
                book_min_height = gr.Slider(4, 80, value=10, step=1, label="Min line height")
                book_max_height = gr.Slider(0, 260, value=140, step=5, label="Max line height (0 = no cap)")
                book_min_area = gr.Slider(20, 5000, value=100, step=20, label="Min line area")
                book_crop_padding = gr.Slider(0, 32, value=12, step=1, label="Crop padding")
                book_max_lines = gr.Slider(0, 300, value=180, step=1, label="Max lines per page (0 = no cap)")
                book_max_pages = gr.Slider(0, 80, value=20, step=1, label="Max pages (0 = no cap)")
                pdf_dpi = gr.Slider(120, 360, value=240, step=10, label="PDF render DPI")
                book_button = gr.Button("Run Image-Line Book OCR", variant="secondary")
        book_text = gr.Textbox(label="Unicode text", lines=18)
        book_zip = gr.File(label="OCR artifact zip")
        book_details = gr.Code(label="Details", language="json")
        book_button.click(
            recognize_book,
            inputs=[
                book_files,
                book_mode,
                book_order,
                book_threshold,
                book_horizontal_kernel,
                book_vertical_kernel,
                book_dilation_iterations,
                book_min_width,
                book_min_height,
                book_max_height,
                book_min_area,
                book_crop_padding,
                book_max_lines,
                book_max_pages,
                pdf_dpi,
            ],
            outputs=[book_text, book_zip, book_details],
        )

    with gr.Tab("YOLO Probe"):
        with gr.Row():
            yolo_image = gr.Image(type="pil", label="Page image")
            with gr.Column():
                yolo_mode = gr.Radio(["Auto", "Sirijonga", "Devanagari"], value="Auto", label="Recognizer")
                detector_conf = gr.Slider(0.05, 0.90, value=0.50, step=0.05, label="Detector confidence")
                yolo_order = gr.Radio(
                    ["Rows LTR", "Two columns LTR", "Two columns RTL"],
                    value="Rows LTR",
                    label="Reading order",
                )
                yolo_crop_padding = gr.Slider(0, 24, value=6, step=1, label="Crop padding")
                yolo_max_lines = gr.Slider(0, 200, value=120, step=1, label="Max lines (0 = no cap)")
                yolo_button = gr.Button("Run YOLO Probe", variant="secondary")
        yolo_text = gr.Textbox(label="Draft Unicode text", lines=16)
        yolo_annotated = gr.Image(type="pil", label="Detected lines")
        yolo_gallery = gr.Gallery(label="Line crops", columns=2, height=420)
        yolo_details = gr.Code(label="Details", language="json")
        yolo_button.click(
            recognize_page_yolo,
            inputs=[yolo_image, yolo_mode, detector_conf, yolo_order, yolo_crop_padding, yolo_max_lines],
            outputs=[yolo_text, yolo_annotated, yolo_gallery, yolo_details],
        )

    with gr.Tab("Runtime"):
        health_button = gr.Button("Check OCR Runtime")
        health_output = gr.Code(label="Status", language="json")
        health_button.click(health, outputs=health_output)
        demo.load(health, outputs=health_output)


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
