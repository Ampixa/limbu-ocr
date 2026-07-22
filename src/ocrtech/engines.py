"""Optional OCR engine adapters."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Protocol

from .errors import DataValidationError, EngineUnavailableError, ParseError
from .metrics import bbox_iou, edit_distance
from .models import ModelCard, audit_model_card
from .newari_scripts import NEWARI_TRANSCRIPTION_ROUTING_RANGES
from .normalization import normalize_ocr_text
from .schemas import BBox, Figure, Page, Table, TableCell, TextLine, validate_confidence


@dataclass(slots=True)
class EngineOutput:
    pages: list[Page]
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def average_confidence(self) -> float | None:
        values = [
            line.confidence
            for page in self.pages
            for line in page.text_lines
            if line.confidence is not None
        ]
        return sum(values) / len(values) if values else None


class OcrEngine(Protocol):
    name: str

    def recognize(self, input_path: Path) -> EngineOutput:
        """Recognize an image or document and return page text lines."""


def _synthetic_lines_from_text(text: str, page_index: int = 0) -> list[TextLine]:
    lines: list[TextLine] = []
    y = 0
    for index, raw_line in enumerate(text.splitlines()):
        line_text = normalize_ocr_text(raw_line)
        if not line_text.strip():
            y += 28
            continue
        line = TextLine(
            text=line_text,
            bbox=BBox(0, y, max(50, len(line_text) * 9), 22),
            confidence=1.0,
            page_index=page_index,
            line_id=f"p{page_index}-l{index}",
        )
        lines.append(line)
        y += 28
    return lines


def sidecar_candidates(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return [
            input_path / "document.ocr.json",
            input_path / "document.txt",
            input_path / "document.md",
        ]
    candidates = [
        input_path.with_suffix(input_path.suffix + ".ocr.json"),
        input_path.with_suffix(".ocr.json"),
        input_path.with_suffix(input_path.suffix + ".txt"),
        input_path.with_suffix(".txt"),
    ]
    if input_path.suffix.lower() in {".txt", ".md"}:
        candidates.insert(0, input_path)
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def has_sidecar(input_path: Path) -> bool:
    return any(candidate.exists() for candidate in sidecar_candidates(input_path))


def _bundle_page_paths(input_path: Path) -> list[Path]:
    if not input_path.is_dir():
        return []
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    page_paths = [path for path in sorted(input_path.iterdir()) if path.is_file() and path.suffix.lower() in allowed_suffixes]
    if not page_paths:
        raise ParseError(f"page bundle directory has no supported image files: {input_path}")
    return page_paths


def _pages_from_lines(lines: list[TextLine], *, metadata: dict[str, object] | None = None) -> list[Page]:
    grouped: dict[int, list[TextLine]] = {}
    for line in lines:
        grouped.setdefault(line.page_index, []).append(line)
    pages: list[Page] = []
    for page_index in sorted(grouped):
        pages.append(Page(page_index=page_index, text_lines=grouped[page_index], metadata=dict(metadata or {})))
    return pages


def _merge_engine_outputs(outputs: list[EngineOutput], *, engine_name: str) -> EngineOutput:
    pages: list[Page] = []
    tables: list[Table] = []
    figures: list[Figure] = []
    metadata: dict[str, object] = {"engine": engine_name}
    for output in outputs:
        pages.extend(output.pages)
        tables.extend(output.tables)
        figures.extend(output.figures)
    return EngineOutput(pages=pages, tables=tables, figures=figures, metadata=metadata)


class SidecarEngine:
    """Deterministic engine for fixtures and pre-OCR text sidecars."""

    name = "sidecar"

    def recognize(self, input_path: Path) -> EngineOutput:
        for candidate in sidecar_candidates(input_path):
            if not candidate.exists():
                continue
            if candidate.suffix == ".json" or candidate.name.endswith(".ocr.json"):
                return self._read_json(candidate)
            return self._read_text(candidate)
        raise EngineUnavailableError(
            f"No OCR sidecar found for {input_path}. Expected one of: "
            + ", ".join(str(path) for path in sidecar_candidates(input_path))
        )

    def _read_text(self, path: Path) -> EngineOutput:
        text = path.read_text(encoding="utf-8")
        lines = _synthetic_lines_from_text(text)
        page = Page(page_index=0, width=None, height=None, text_lines=lines, metadata={"sidecar_path": str(path)})
        return EngineOutput(pages=[page], metadata={"engine": self.name, "sidecar_path": str(path)})

    def _read_json(self, path: Path) -> EngineOutput:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Invalid OCR sidecar JSON {path}: {exc}") from exc
        raw_pages = data.get("pages")
        if not isinstance(raw_pages, list):
            raise DataValidationError(f"OCR sidecar {path} must contain a pages list")
        pages: list[Page] = []
        for page_index, raw_page in enumerate(raw_pages):
            if not isinstance(raw_page, dict):
                raise DataValidationError(f"OCR sidecar page {page_index} must be an object")
            lines = [
                TextLine.from_dict(raw_line, page_index=page_index)
                for raw_line in raw_page.get("lines", raw_page.get("text_lines", []))
            ]
            for line_index, line in enumerate(lines):
                if line.line_id is None:
                    line.line_id = f"p{page_index}-l{line_index}"
            pages.append(
                Page(
                    page_index=page_index,
                    width=raw_page.get("width"),
                    height=raw_page.get("height"),
                    text_lines=lines,
                    metadata=dict(raw_page.get("metadata") or {}),
                )
            )
        tables = [Table.from_dict(item) for item in data.get("tables", [])]
        figures = [Figure.from_dict(item) for item in data.get("figures", [])]
        metadata = dict(data.get("metadata") or {})
        metadata.update({"engine": self.name, "sidecar_path": str(path)})
        return EngineOutput(pages=pages, tables=tables, figures=figures, metadata=metadata)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, *, language: str = "nep+eng", psm: int = 6) -> None:
        self.language = language
        self.psm = psm
        self.binary = shutil.which("tesseract")
        if self.binary is None:
            raise EngineUnavailableError("tesseract binary is not installed or not on PATH")

    def recognize(self, input_path: Path) -> EngineOutput:
        if input_path.is_dir():
            outputs: list[EngineOutput] = []
            for page_index, page_path in enumerate(_bundle_page_paths(input_path)):
                lines = self._recognize_page(page_path, page_index=page_index)
                outputs.append(
                    EngineOutput(
                        pages=[Page(page_index=page_index, text_lines=lines, metadata={"engine": self.name, "language": self.language})],
                        metadata={"engine": self.name, "language": self.language},
                    )
                )
            return _merge_engine_outputs(outputs, engine_name=self.name)
        lines = self._recognize_page(input_path, page_index=0)
        page = Page(page_index=0, text_lines=lines, metadata={"engine": self.name, "language": self.language})
        return EngineOutput(pages=[page], metadata={"engine": self.name, "language": self.language})

    def _recognize_page(self, input_path: Path, *, page_index: int) -> list[TextLine]:
        command = [
            self.binary or "tesseract",
            str(input_path),
            "stdout",
            "-l",
            self.language,
            "--psm",
            str(self.psm),
            "tsv",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ParseError(f"tesseract failed for {input_path}: {completed.stderr.strip()}")
        return self._parse_tsv(completed.stdout, page_offset=page_index)

    def _parse_tsv(self, text: str, *, page_offset: int = 0) -> list[TextLine]:
        reader = csv.DictReader(text.splitlines(), delimiter="\t")
        grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        for row in reader:
            if row.get("level") != "5":
                continue
            word = row.get("text", "")
            if not word.strip():
                continue
            key = (row.get("page_num", "1"), row.get("block_num", "0"), row.get("par_num", "0"), row.get("line_num", "0"))
            grouped.setdefault(key, []).append(row)
        lines: list[TextLine] = []
        for line_index, (key, rows) in enumerate(sorted(grouped.items(), key=lambda item: tuple(int(part) for part in item[0]))):
            boxes = []
            confidences = []
            words = []
            for row in rows:
                x = int(float(row.get("left", "0") or 0))
                y = int(float(row.get("top", "0") or 0))
                w = int(float(row.get("width", "0") or 0))
                h = int(float(row.get("height", "0") or 0))
                boxes.append(BBox(x, y, w, h))
                words.append(row.get("text", ""))
                try:
                    conf = float(row.get("conf", "-1"))
                except ValueError:
                    conf = -1
                if conf >= 0:
                    confidences.append(conf / 100)
            from .schemas import union_bbox

            page_index = page_offset + max(0, int(key[0]) - 1)
            confidence = sum(confidences) / len(confidences) if confidences else None
            lines.append(
                TextLine(
                    text=normalize_ocr_text(" ".join(words)),
                    bbox=union_bbox(boxes),
                    confidence=confidence,
                    page_index=page_index,
                    line_id=f"p{page_index}-l{line_index}",
                )
            )
        return lines


class PaddleOcrEngine:
    name = "paddleocr"

    def __init__(
        self,
        *,
        recognition_mode: str = "full_page",
        line_mode_max_height: int = 256,
        line_mode_min_aspect_ratio: float = 3.0,
        **kwargs: object,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise EngineUnavailableError("paddleocr is not installed. Install ocr-tech[engines].") from exc
        try:
            from paddleocr._models.text_recognition import TextRecognition
        except ImportError:
            TextRecognition = None
        normalized_mode = recognition_mode.strip().lower()
        if normalized_mode == "recognition_only":
            normalized_mode = "line"
        if normalized_mode not in {"full_page", "line", "auto"}:
            raise EngineUnavailableError(f"unsupported PaddleOCR recognition_mode: {recognition_mode!r}")
        self._paddle_cls = PaddleOCR
        self._text_recognition_cls = TextRecognition
        self._paddle_kwargs = dict(kwargs) if kwargs else {"lang": "ne"}
        self._recognition_kwargs = _paddle_text_recognition_kwargs(self._paddle_kwargs)
        self.recognition_mode = normalized_mode
        self.line_mode_max_height = int(line_mode_max_height)
        self.line_mode_min_aspect_ratio = float(line_mode_min_aspect_ratio)
        self.ocr = None
        self.text_recognizer = None

    def recognize(self, input_path: Path) -> EngineOutput:
        if input_path.is_dir():
            outputs: list[EngineOutput] = []
            for page_index, page_path in enumerate(_bundle_page_paths(input_path)):
                outputs.append(self._recognize_image(page_path, page_index=page_index))
            return _merge_engine_outputs(outputs, engine_name=self.name)
        return self._recognize_image(input_path, page_index=0)

    def _recognize_image(self, input_path: Path, *, page_index: int) -> EngineOutput:
        if self._select_recognition_mode(input_path) == "line":
            lines = self._recognize_line_image(input_path, page_index=page_index)
            page = Page(page_index=page_index, text_lines=lines, metadata={"engine": self.name, "paddle_recognition_mode": "line"})
            return EngineOutput(pages=[page], metadata={"engine": self.name, "paddle_recognition_mode": "line"})
        raw_result = self._run_ocr(input_path)
        lines = self._parse_result(raw_result, page_index=page_index)
        page = Page(page_index=page_index, text_lines=lines, metadata={"engine": self.name, "paddle_recognition_mode": "full_page"})
        return EngineOutput(pages=[page], metadata={"engine": self.name, "paddle_recognition_mode": "full_page"})

    def _select_recognition_mode(self, input_path: Path) -> str:
        if self.recognition_mode != "auto":
            return self.recognition_mode
        return "line" if _is_line_like_image(input_path, self.line_mode_max_height, self.line_mode_min_aspect_ratio) else "full_page"

    def _run_ocr(self, input_path: Path) -> object:
        ocr = self._full_page_ocr()
        if hasattr(ocr, "predict"):
            return ocr.predict(str(input_path))
        if hasattr(ocr, "ocr"):
            return ocr.ocr(str(input_path), cls=True)
        raise EngineUnavailableError("Installed paddleocr object exposes neither predict() nor ocr()")

    def _full_page_ocr(self) -> object:
        if self.ocr is None:
            try:
                self.ocr = self._paddle_cls(**self._paddle_kwargs)
            except RuntimeError as exc:
                raise EngineUnavailableError(f"PaddleOCR runtime is unavailable: {exc}") from exc
        return self.ocr

    def _line_recognizer(self) -> object:
        if self._text_recognition_cls is None:
            raise EngineUnavailableError("Installed paddleocr package does not expose TextRecognition")
        if self.text_recognizer is None:
            try:
                self.text_recognizer = self._text_recognition_cls(**self._recognition_kwargs)
            except RuntimeError as exc:
                raise EngineUnavailableError(f"PaddleOCR text-recognition runtime is unavailable: {exc}") from exc
        return self.text_recognizer

    def _recognize_line_image(self, input_path: Path, *, page_index: int) -> list[TextLine]:
        recognizer = self._line_recognizer()
        raw_result = recognizer.predict(str(input_path))
        width, height = _image_size(input_path)
        lines: list[TextLine] = []
        for item in raw_result if isinstance(raw_result, list) else [raw_result]:
            parsed = self._parse_recognition_result(item, width=width, height=height)
            if parsed is None:
                continue
            text, bbox, confidence = parsed
            if not text:
                continue
            lines.append(
                TextLine(
                    text=normalize_ocr_text(text),
                    bbox=bbox,
                    confidence=confidence,
                    page_index=page_index,
                    line_id=f"p{page_index}-l{len(lines)}",
                )
            )
        return lines

    def _parse_recognition_result(self, item: object, *, width: int, height: int) -> tuple[str, BBox, float | None] | None:
        payload = self._result_mapping(item)
        if payload is None:
            return None
        text = payload.get("rec_text") or payload.get("text") or payload.get("transcription")
        if text is None:
            return None
        score = payload.get("rec_score") or payload.get("score") or payload.get("confidence")
        confidence = float(score) if score is not None else None
        return str(text), BBox(0, 0, width, height), confidence

    def _parse_result(self, raw_result: object, *, page_index: int) -> list[TextLine]:
        lines: list[TextLine] = []
        candidates = raw_result
        if isinstance(raw_result, list) and len(raw_result) == 1 and isinstance(raw_result[0], list):
            candidates = raw_result[0]
        if not isinstance(candidates, list):
            raise ParseError(f"Unsupported PaddleOCR result type: {type(raw_result).__name__}")
        line_index = 0
        for item in candidates:
            parsed_lines = self._parse_page_result(item)
            if parsed_lines is None:
                parsed = self._parse_line(item)
                parsed_lines = [] if parsed is None else [parsed]
            for text, bbox, confidence in parsed_lines:
                lines.append(
                    TextLine(
                        text=normalize_ocr_text(text),
                        bbox=bbox,
                        confidence=confidence,
                        page_index=page_index,
                        line_id=f"p{page_index}-l{line_index}",
                    )
                )
                line_index += 1
        return lines

    def _parse_page_result(self, item: object) -> list[tuple[str, BBox, float | None]] | None:
        payload = self._result_mapping(item)
        if payload is None:
            return None
        texts = payload.get("rec_texts")
        boxes = payload.get("rec_polys") or payload.get("dt_polys")
        scores = payload.get("rec_scores")
        if not isinstance(texts, list) or not isinstance(boxes, list):
            return None
        parsed_lines: list[tuple[str, BBox, float | None]] = []
        for index, text in enumerate(texts):
            if text is None:
                continue
            if index >= len(boxes):
                continue
            confidence = None
            if isinstance(scores, list) and index < len(scores) and scores[index] is not None:
                confidence = float(scores[index])
            parsed_lines.append((str(text), _bbox_from_paddle_box(boxes[index]), confidence))
        return parsed_lines

    def _result_mapping(self, item: object) -> dict[str, object] | None:
        payload: object = item
        if not isinstance(payload, dict):
            payload = getattr(item, "json", None)
        if not isinstance(payload, dict):
            return None
        nested = payload.get("res")
        if isinstance(nested, dict):
            return nested
        return payload

    def _parse_line(self, item: object) -> tuple[str, BBox, float | None] | None:
        if isinstance(item, dict):
            text = item.get("text") or item.get("rec_text") or item.get("transcription")
            score = item.get("score") or item.get("confidence") or item.get("rec_score")
            bbox_value = item.get("bbox") or item.get("box") or item.get("dt_polys")
            if text is None:
                return None
            return str(text), _bbox_from_paddle_box(bbox_value), float(score) if score is not None else None
        if isinstance(item, list | tuple) and len(item) >= 2:
            bbox_value = item[0]
            rec = item[1]
            if isinstance(rec, list | tuple) and rec:
                text = str(rec[0])
                score = float(rec[1]) if len(rec) > 1 and rec[1] is not None else None
                return text, _bbox_from_paddle_box(bbox_value), score
        return None


class SuryaEngine:
    name = "surya"

    def __init__(
        self,
        *,
        mode: str = "auto",
        line_mode_max_height: int = 256,
        line_mode_min_aspect_ratio: float = 3.0,
        table_recognition_mode: str = "full",
    ) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise EngineUnavailableError("Pillow is not installed. Install ocr-tech[surya].") from exc
        try:
            from surya.inference import SuryaInferenceManager
            from surya.layout.schema import LayoutBox, LayoutResult
            from surya.recognition import RecognitionPredictor
            from surya.table_rec import TableRecPredictor
        except ImportError as exc:
            raise EngineUnavailableError("surya-ocr is not installed. Install ocr-tech[surya].") from exc

        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"auto", "full_page", "block"}:
            raise EngineUnavailableError(f"unsupported surya mode: {mode!r}")
        normalized_table_mode = table_recognition_mode.strip().lower()
        if normalized_table_mode not in {"simple", "full"}:
            raise EngineUnavailableError(f"unsupported surya table recognition mode: {table_recognition_mode!r}")
        if shutil.which("llama-server") is None and not os.environ.get("LLAMA_CPP_BINARY"):
            raise EngineUnavailableError("surya requires llama-server on PATH or LLAMA_CPP_BINARY to be set")

        self.mode = normalized_mode
        self.line_mode_max_height = int(line_mode_max_height)
        self.line_mode_min_aspect_ratio = float(line_mode_min_aspect_ratio)
        self.table_recognition_mode = normalized_table_mode
        self._image_module = Image
        self._layout_box_cls = LayoutBox
        self._layout_result_cls = LayoutResult
        self._manager = SuryaInferenceManager()
        self._predictor = RecognitionPredictor(self._manager)
        self._table_predictor = TableRecPredictor(self._manager)

    def recognize(self, input_path: Path) -> EngineOutput:
        if input_path.is_dir():
            outputs: list[EngineOutput] = []
            for page_index, page_path in enumerate(_bundle_page_paths(input_path)):
                outputs.append(self._recognize_image(page_path, page_index=page_index))
            return _merge_engine_outputs(outputs, engine_name=self.name)
        return self._recognize_image(input_path, page_index=0)

    def _recognize_image(self, input_path: Path, *, page_index: int) -> EngineOutput:
        image = self._image_module.open(input_path).convert("RGB")
        page_width, page_height = image.size
        selected_mode = self._select_mode(page_width, page_height)
        if selected_mode == "block":
            page_result = self._predict_block(image)
            table_results: dict[int, object] = {}
        else:
            page_result = self._predictor([image], full_page=True)[0]
            table_results = self._predict_tables(image, page_result)
        return self._convert_page_result(
            page_result,
            page_width,
            page_height,
            selected_mode,
            page_index=page_index,
            table_results=table_results,
        )

    def _select_mode(self, width: int, height: int) -> str:
        if self.mode != "auto":
            return self.mode
        if height <= max(1, self.line_mode_max_height):
            return "block"
        aspect_ratio = float(width) / float(max(height, 1))
        if aspect_ratio >= self.line_mode_min_aspect_ratio:
            return "block"
        return "full_page"

    def _predict_block(self, image: object) -> object:
        width, height = image.size
        layout = self._layout_result_cls(
            bboxes=[
                self._layout_box_cls(
                    polygon=[[0, 0], [width, 0], [width, height], [0, height]],
                    label="Text",
                    raw_label="Text",
                    position=0,
                    count=50,
                )
            ],
            image_bbox=[0, 0, float(width), float(height)],
        )
        return self._predictor([image], [layout], full_page=False)[0]

    def _predict_tables(self, image: object, page_result: object) -> dict[int, object]:
        raw_blocks = getattr(page_result, "blocks", None)
        if not isinstance(raw_blocks, list):
            return {}
        crops: list[object] = []
        indices: list[int] = []
        for index, block in enumerate(raw_blocks):
            label = str(getattr(block, "label", "") or "")
            if label != "Table":
                continue
            bbox = _bbox_from_surya_polygon(getattr(block, "polygon", None))
            crop = _crop_image_to_bbox(image, bbox)
            if crop is None:
                continue
            crops.append(crop)
            indices.append(index)
        if not crops:
            return {}
        predictions = self._table_predictor(crops, mode=self.table_recognition_mode)
        return {index: prediction for index, prediction in zip(indices, predictions, strict=False)}

    def _convert_page_result(
        self,
        page_result: object,
        page_width: int,
        page_height: int,
        selected_mode: str,
        page_index: int,
        *,
        table_results: dict[int, object] | None = None,
    ) -> EngineOutput:
        raw_blocks = getattr(page_result, "blocks", None)
        if not isinstance(raw_blocks, list):
            raise ParseError("surya output is missing blocks")
        text_lines: list[TextLine] = []
        tables: list[Table] = []
        figures: list[Figure] = []
        table_results = table_results or {}
        ordered_blocks = sorted(enumerate(raw_blocks), key=lambda item: int(getattr(item[1], "reading_order", 0) or 0))
        for line_index, (block_index, block) in enumerate(ordered_blocks):
            label = str(getattr(block, "label", "") or "")
            polygon = getattr(block, "polygon", None)
            bbox = _bbox_from_surya_polygon(polygon)
            confidence = validate_confidence(getattr(block, "confidence", None), "surya confidence")
            plain_text = _surya_html_to_text(str(getattr(block, "html", "") or ""))
            metadata = {
                "surya_label": label,
                "surya_raw_label": str(getattr(block, "raw_label", "") or ""),
                "surya_reading_order": int(getattr(block, "reading_order", line_index) or line_index),
                "surya_html": str(getattr(block, "html", "") or ""),
                "surya_skipped": bool(getattr(block, "skipped", False)),
                "surya_error": bool(getattr(block, "error", False)),
            }
            if label == "Table":
                table = _table_from_surya_block(
                    block_index=block_index,
                    table_id=f"p{page_index}-t{len(tables)}",
                    bbox=bbox,
                    confidence=confidence,
                    page_index=page_index,
                    metadata=metadata,
                    block_html=str(getattr(block, "html", "") or ""),
                    table_result=table_results.get(block_index),
                )
                tables.append(table)
            elif label in {"Picture", "Image", "Figure", "Chart"}:
                figures.append(
                    Figure(
                        figure_id=f"p{page_index}-f{len(figures)}",
                        page_index=page_index,
                        bbox=bbox,
                        summary=plain_text or None,
                        confidence=confidence,
                        metadata=metadata,
                    )
                )
            if plain_text:
                text_lines.append(
                    TextLine(
                        text=plain_text,
                        bbox=bbox,
                        confidence=confidence,
                        page_index=page_index,
                        line_id=f"p{page_index}-l{line_index}",
                        metadata=metadata,
                    )
                )
        page = Page(
            page_index=page_index,
            width=float(page_width),
            height=float(page_height),
            text_lines=text_lines,
            metadata={"engine": self.name, "surya_mode": selected_mode},
        )
        return EngineOutput(
            pages=[page],
            tables=tables,
            figures=figures,
            metadata={"engine": self.name, "surya_mode": selected_mode},
        )


def _crop_image_to_bbox(image: object, bbox: BBox) -> object | None:
    left = max(0, int(bbox.x))
    top = max(0, int(bbox.y))
    right = max(left + 1, int(bbox.right))
    bottom = max(top + 1, int(bbox.bottom))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _table_from_surya_block(
    *,
    block_index: int,
    table_id: str,
    bbox: BBox,
    confidence: float | None,
    page_index: int,
    metadata: dict[str, object],
    block_html: str,
    table_result: object | None,
) -> Table:
    html_value = ""
    if table_result is not None and isinstance(getattr(table_result, "html", None), str):
        html_value = str(getattr(table_result, "html", "") or "")
    elif block_html:
        html_value = block_html
    cells = _table_cells_from_html(html_value)
    table_metadata = dict(metadata)
    if table_result is not None:
        table_metadata["surya_table_mode"] = str(getattr(table_result, "mode", "") or "")
        table_metadata["surya_table_error"] = bool(getattr(table_result, "error", False))
        if hasattr(table_result, "rows"):
            table_metadata["surya_table_row_count"] = len(getattr(table_result, "rows", []) or [])
        if hasattr(table_result, "cols"):
            table_metadata["surya_table_col_count"] = len(getattr(table_result, "cols", []) or [])
    table_metadata["surya_block_index"] = block_index
    return Table(
        table_id=table_id,
        page_index=page_index,
        bbox=bbox,
        cells=cells,
        html=html_value or None,
        confidence=confidence,
        metadata=table_metadata,
    )


def _bbox_from_surya_polygon(value: object) -> BBox:
    if isinstance(value, list | tuple):
        points: list[tuple[float, float]] = []
        for point in value:
            if isinstance(point, list | tuple) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if points:
            left = min(point[0] for point in points)
            top = min(point[1] for point in points)
            right = max(point[0] for point in points)
            bottom = max(point[1] for point in points)
            return BBox(left, top, right - left, bottom - top)
    return BBox(0, 0, 0, 0)


class _SuryaHtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "tr", "li", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        joined = "".join(self.parts)
        collapsed = re.sub(r"\n{3,}", "\n\n", joined)
        collapsed = re.sub(r"[ \t]+", " ", collapsed)
        return normalize_ocr_text(html.unescape(collapsed).strip())


def _surya_html_to_text(value: str) -> str:
    if not value:
        return ""
    parser = _SuryaHtmlTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.get_text()


class _SuryaTableHtmlExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: list[TableCell] = []
        self._in_row = False
        self._cell_text: list[str] | None = None
        self._current_row = 0
        self._current_col = 0
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._current_col = 0
            return
        if tag in {"td", "th"} and self._in_row:
            attrs_dict = {key: value for key, value in attrs}
            self._cell_text = []
            self._cell_rowspan = _safe_positive_int(attrs_dict.get("rowspan"), default=1)
            self._cell_colspan = _safe_positive_int(attrs_dict.get("colspan"), default=1)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell_text is not None:
            text = normalize_ocr_text(html.unescape("".join(self._cell_text)).strip())
            self.cells.append(
                TableCell(
                    row=self._current_row,
                    col=self._current_col,
                    text=text,
                    rowspan=self._cell_rowspan,
                    colspan=self._cell_colspan,
                    bbox=None,
                    confidence=None,
                )
            )
            self._current_col += self._cell_colspan
            self._cell_text = None
            self._cell_rowspan = 1
            self._cell_colspan = 1
            return
        if tag == "tr" and self._in_row:
            self._current_row += 1
            self._current_col = 0
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._cell_text is not None:
            self._cell_text.append(data)


def _safe_positive_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _table_cells_from_html(value: str) -> list[TableCell]:
    if not value:
        return []
    parser = _SuryaTableHtmlExtractor()
    parser.feed(value)
    parser.close()
    return parser.cells


class CandidateEngine:
    name = "candidate"

    def __init__(self, *, model_config: str | Path | None = None) -> None:
        if model_config is None:
            raise EngineUnavailableError("candidate engine requires a model_config path")
        audit = audit_model_card(model_config)
        if not audit.passed:
            raise EngineUnavailableError("candidate model audit failed: " + "; ".join(audit.issues))
        self.model_config = Path(model_config)
        self.card = ModelCard.from_path(self.model_config)
        if self.card.backend == "sidecar":
            self.engine: OcrEngine = SidecarEngine()
        elif self.card.backend == "tesseract":
            self.engine = TesseractEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "paddleocr":
            self.engine = PaddleOcrEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "surya":
            self.engine = SuryaEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "hf_vision_encoder_decoder":
            self.engine = HuggingFaceVisionTextEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "text_correction_composite":
            self.engine = TextCorrectionCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "quality_select_composite":
            self.engine = QualitySelectCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "quality_ranked_ensemble":
            self.engine = QualityRankedEnsembleEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "script_select_composite":
            self.engine = ScriptSelectCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "line_align_composite":
            self.engine = LineAlignCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "table_cell_refine_composite":
            self.engine = TableCellRefineCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        elif self.card.backend == "figure_caption_refine_composite":
            self.engine = FigureCaptionRefineCompositeEngine(**_resolve_model_kwargs(self.card, self.model_config))
        else:
            raise EngineUnavailableError(f"unsupported candidate backend: {self.card.backend}")

    def recognize(self, input_path: Path) -> EngineOutput:
        output = self.engine.recognize(input_path)
        output.metadata["engine"] = self.name
        output.metadata["candidate_model"] = {
            "model_id": self.card.model_id,
            "backend": self.card.backend,
            "model_config": str(self.model_config),
            "base_model": self.card.base_model,
        }
        return output


def _bbox_from_paddle_box(value: object) -> BBox:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[assignment, attr-defined]
    if isinstance(value, list | tuple):
        if len(value) == 4 and all(isinstance(item, int | float) for item in value):
            return BBox.from_any(value)
        points = []
        for point in value:
            if isinstance(point, list | tuple) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if points:
            left = min(point[0] for point in points)
            top = min(point[1] for point in points)
            right = max(point[0] for point in points)
            bottom = max(point[1] for point in points)
            return BBox(left, top, right - left, bottom - top)
    return BBox(0, 0, 0, 0)


def _image_size(input_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise EngineUnavailableError("Pillow is required to inspect image dimensions") from exc
    try:
        with Image.open(input_path) as image:
            return image.size
    except Exception as exc:
        raise ParseError(f"could not inspect image dimensions for {input_path}: {type(exc).__name__}: {exc}") from exc


def _is_line_like_image(input_path: Path, max_height: int, min_aspect_ratio: float) -> bool:
    width, height = _image_size(input_path)
    if height <= 0:
        return False
    return height <= max(1, int(max_height)) and (width / height) >= float(min_aspect_ratio)


def _paddle_text_recognition_kwargs(paddle_kwargs: dict[str, object]) -> dict[str, object]:
    recognition_kwargs: dict[str, object] = {}
    rename = {
        "text_recognition_model_name": "model_name",
        "text_recognition_model_dir": "model_dir",
        "text_recognition_input_shape": "input_shape",
    }
    passthrough = {
        "device",
        "enable_hpi",
        "use_tensorrt",
        "precision",
        "enable_mkldnn",
        "mkldnn_cache_capacity",
        "cpu_threads",
        "paddlex_config",
    }
    for key, value in paddle_kwargs.items():
        if key in rename:
            recognition_kwargs[rename[key]] = value
        elif key in passthrough:
            recognition_kwargs[key] = value
    return recognition_kwargs


def create_engine(name: str, **kwargs: object) -> OcrEngine:
    engine_name = name.lower().strip()
    if engine_name == "sidecar":
        return SidecarEngine()
    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(**kwargs)
    if engine_name in {"paddleocr", "paddle", "stock-paddle"}:
        return PaddleOcrEngine(**kwargs)
    if engine_name == "surya":
        return SuryaEngine(**kwargs)
    if engine_name in {"candidate", "ours"}:
        return CandidateEngine(**kwargs)
    if engine_name == "auto":
        return AutoEngine()
    raise EngineUnavailableError(f"Unknown OCR engine: {name}")


class AutoEngine:
    name = "auto"

    def recognize(self, input_path: Path) -> EngineOutput:
        if has_sidecar(input_path):
            output = SidecarEngine().recognize(input_path)
            output.metadata["auto_selected_engine"] = "sidecar"
            return output
        errors: list[str] = []
        for factory_name in ("paddleocr", "tesseract"):
            try:
                engine = create_engine(factory_name)
                output = engine.recognize(input_path)
                output.metadata["auto_selected_engine"] = factory_name
                return output
            except EngineUnavailableError as exc:
                errors.append(f"{factory_name}: {exc}")
        raise EngineUnavailableError("No OCR engine available. " + " | ".join(errors))


def _resolve_model_kwargs(card: ModelCard, config_path: Path) -> dict[str, object]:
    return {
        key: _resolve_model_kwarg_value(key, value, config_path.parent)
        for key, value in card.backend_kwargs.items()
    }


def _resolve_model_kwarg_value(key: str, value: object, base_dir: Path) -> object:
    if isinstance(value, str) and key.endswith(("_dir", "_path", "_file", "_config")):
        path = Path(value)
        return str(path if path.is_absolute() else base_dir / path)
    if isinstance(value, dict):
        return {
            str(nested_key): _resolve_model_kwarg_value(str(nested_key), nested_value, base_dir)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_model_kwarg_value(key, item, base_dir) for item in value]
    return value


class HuggingFaceVisionTextEngine:
    name = "hf_vision_encoder_decoder"

    def __init__(self, *, model_dir: str, processor_dir: str | None = None, max_new_tokens: int = 128, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise EngineUnavailableError("torch is not installed. Install ocr-tech[hf-recognizer].") from exc
        try:
            from PIL import Image
        except ImportError as exc:
            raise EngineUnavailableError("Pillow is not installed. Install ocr-tech[hf-recognizer].") from exc
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise EngineUnavailableError("transformers is not installed. Install ocr-tech[hf-recognizer].") from exc

        self._torch = torch
        self._image_module = Image
        self.max_new_tokens = int(max_new_tokens)
        self.model_dir = Path(model_dir)
        self.processor_dir = Path(processor_dir) if processor_dir is not None else self.model_dir
        if not self.model_dir.exists():
            raise EngineUnavailableError(f"hf model_dir does not exist: {self.model_dir}")
        if not self.processor_dir.exists():
            raise EngineUnavailableError(f"hf processor_dir does not exist: {self.processor_dir}")
        self.processor = _load_trocr_processor(TrOCRProcessor, self.processor_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir))
        self.device = device or _default_torch_device(torch)
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
            try:
                generation_config.max_length = None
            except AttributeError:
                pass
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, input_path: Path) -> EngineOutput:
        if input_path.is_dir():
            outputs: list[EngineOutput] = []
            for page_index, page_path in enumerate(_bundle_page_paths(input_path)):
                outputs.append(self._recognize_page(page_path, page_index=page_index))
            merged = _merge_engine_outputs(outputs, engine_name=self.name)
            merged.metadata.update(
                {
                    "hf_model_dir": str(self.model_dir),
                    "hf_processor_dir": str(self.processor_dir),
                    "hf_device": self.device,
                }
            )
            return merged
        return self._recognize_page(input_path, page_index=0)

    def _recognize_page(self, input_path: Path, *, page_index: int) -> EngineOutput:
        image = self._image_module.open(input_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        with self._torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=self.max_new_tokens)
        text = normalize_ocr_text(self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        line = TextLine(
            text=text,
            bbox=BBox(0, 0, 0, 0),
            confidence=None,
            page_index=page_index,
            line_id=f"p{page_index}-l0",
        )
        page = Page(page_index=page_index, text_lines=[line], metadata={"engine": self.name})
        return EngineOutput(
            pages=[page],
            metadata={
                "engine": self.name,
                "hf_model_dir": str(self.model_dir),
                "hf_processor_dir": str(self.processor_dir),
                "hf_device": self.device,
            },
        )


def _default_torch_device(torch: object) -> str:
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_trocr_processor(processor_cls: type[object], source: str | Path) -> object:
    attempts = ({"backend": "pil"}, {"use_fast": False})
    for kwargs in attempts:
        try:
            return processor_cls.from_pretrained(str(source), **kwargs)
        except TypeError as exc:
            if any(key in str(exc) for key in kwargs):
                continue
            raise
        except ImportError as exc:
            raise EngineUnavailableError("TrOCR processor loading requires sentencepiece and protobuf. Install ocr-tech[hf-recognizer].") from exc
        except ValueError as exc:
            message = str(exc)
            if "SentencePiece" in message or "sentencepiece" in message or "tiktoken" in message or "protobuf" in message:
                raise EngineUnavailableError("TrOCR processor loading requires sentencepiece and protobuf. Install ocr-tech[hf-recognizer].") from exc
            raise
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" in str(exc):
                return _load_trocr_processor_with_sanitized_tokenizer_config(processor_cls, source, kwargs)
            raise
    return processor_cls.from_pretrained(str(source))


def _load_trocr_processor_with_sanitized_tokenizer_config(
    processor_cls: type[object],
    source: str | Path,
    kwargs: dict[str, object],
) -> object:
    source_path = Path(source)
    tokenizer_config_path = source_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise EngineUnavailableError(f"HF processor tokenizer_config.json does not exist: {tokenizer_config_path}")
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid tokenizer_config.json in HF processor directory {source_path}: {exc}") from exc
    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        raise EngineUnavailableError(
            "HF processor loading failed with extra_special_tokens compatibility error, "
            "but tokenizer_config.json did not contain a list to sanitize"
        )
    sanitized_config = dict(tokenizer_config)
    sanitized_config["extra_special_tokens"] = {}
    with tempfile.TemporaryDirectory(prefix="ocrtech-trocr-processor-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copytree(source_path, tmp_path, dirs_exist_ok=True)
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps(sanitized_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return processor_cls.from_pretrained(str(tmp_path), **kwargs)


class TextCorrectionCompositeEngine:
    name = "text_correction_composite"

    def __init__(
        self,
        *,
        model_dir: str,
        tokenizer_dir: str | None = None,
        base_engine: str = "tesseract",
        base_engine_kwargs: dict[str, object] | None = None,
        max_new_tokens: int = 256,
        device: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise EngineUnavailableError("torch is not installed. Install ocr-tech[hf-recognizer].") from exc
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise EngineUnavailableError("transformers is not installed. Install ocr-tech[hf-recognizer].") from exc

        if base_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"text correction composite does not support base_engine={base_engine!r}")
        self.base_engine = create_engine(base_engine, **(base_engine_kwargs or {}))
        self._torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else self.model_dir
        if not self.model_dir.exists():
            raise EngineUnavailableError(f"text corrector model_dir does not exist: {self.model_dir}")
        if not self.tokenizer_dir.exists():
            raise EngineUnavailableError(f"text corrector tokenizer_dir does not exist: {self.tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        self.device = device or _default_torch_device(torch)
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
            try:
                generation_config.max_length = None
            except AttributeError:
                pass
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, input_path: Path) -> EngineOutput:
        base_output = self.base_engine.recognize(input_path)
        all_lines = [line for page in base_output.pages for line in page.text_lines]
        corrected = self._correct_lines([line.text for line in all_lines])
        index = 0
        corrected_pages: list[Page] = []
        for page in base_output.pages:
            text_lines: list[TextLine] = []
            for line in page.text_lines:
                text_lines.append(replace(line, text=corrected[index]))
                index += 1
            corrected_pages.append(replace(page, text_lines=text_lines))
        metadata = dict(base_output.metadata)
        metadata.update(
            {
                "engine": self.name,
                "correction_base_engine": getattr(self.base_engine, "name", type(self.base_engine).__name__),
                "correction_model_dir": str(self.model_dir),
                "correction_tokenizer_dir": str(self.tokenizer_dir),
                "correction_device": self.device,
            }
        )
        return EngineOutput(pages=corrected_pages, tables=base_output.tables, figures=base_output.figures, metadata=metadata)

    def _correct_lines(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in tokenized.items()}
        with self._torch.no_grad():
            generated_ids = self.model.generate(**batch, max_new_tokens=self.max_new_tokens)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [normalize_ocr_text(text) for text in decoded]


class QualitySelectCompositeEngine:
    name = "quality_select_composite"

    def __init__(
        self,
        *,
        primary_engine: str,
        secondary_engine: str,
        primary_engine_kwargs: dict[str, object] | None = None,
        secondary_engine_kwargs: dict[str, object] | None = None,
        secondary_quality_margin: float = 0.0,
    ) -> None:
        if primary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"quality_select_composite does not support primary_engine={primary_engine!r}")
        if secondary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"quality_select_composite does not support secondary_engine={secondary_engine!r}")
        self.primary_engine = create_engine(primary_engine, **(primary_engine_kwargs or {}))
        self.secondary_engine = create_engine(secondary_engine, **(secondary_engine_kwargs or {}))
        self.secondary_quality_margin = float(secondary_quality_margin)

    def recognize(self, input_path: Path) -> EngineOutput:
        primary_output, primary_error = self._try_recognize(self.primary_engine, input_path)
        secondary_output, secondary_error = self._try_recognize(self.secondary_engine, input_path)
        if primary_output is None and secondary_output is None:
            raise ParseError(
                "quality_select_composite failed: "
                + "; ".join(
                    [
                        f"primary={type(primary_error).__name__}: {primary_error}",
                        f"secondary={type(secondary_error).__name__}: {secondary_error}",
                    ]
                )
            )
        if primary_output is None:
            assert secondary_output is not None
            return self._finalize_output(
                secondary_output,
                chosen_engine=getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                primary_quality=None,
                secondary_quality=_score_engine_output_quality(input_path, secondary_output),
                primary_error=primary_error,
                secondary_error=secondary_error,
            )
        if secondary_output is None:
            return self._finalize_output(
                primary_output,
                chosen_engine=getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                primary_quality=_score_engine_output_quality(input_path, primary_output),
                secondary_quality=None,
                primary_error=primary_error,
                secondary_error=secondary_error,
            )
        primary_quality = _score_engine_output_quality(input_path, primary_output)
        secondary_quality = _score_engine_output_quality(input_path, secondary_output)
        quality_delta = secondary_quality.quality_score - primary_quality.quality_score
        if quality_delta > self.secondary_quality_margin:
            return self._finalize_output(
                secondary_output,
                chosen_engine=getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                primary_quality=primary_quality,
                secondary_quality=secondary_quality,
                primary_error=primary_error,
                secondary_error=secondary_error,
            )
        return self._finalize_output(
            primary_output,
            chosen_engine=getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
            primary_quality=primary_quality,
            secondary_quality=secondary_quality,
            primary_error=primary_error,
            secondary_error=secondary_error,
        )

    def _try_recognize(self, engine: OcrEngine, input_path: Path) -> tuple[EngineOutput | None, Exception | None]:
        try:
            return engine.recognize(input_path), None
        except Exception as exc:
            return None, exc

    def _finalize_output(
        self,
        output: EngineOutput,
        *,
        chosen_engine: str,
        primary_quality: object | None,
        secondary_quality: object | None,
        primary_error: Exception | None,
        secondary_error: Exception | None,
    ) -> EngineOutput:
        metadata = dict(output.metadata)
        metadata.update(
            {
                "engine": self.name,
                "quality_select_composite": {
                    "primary_engine": getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                    "secondary_engine": getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                    "chosen_engine": chosen_engine,
                    "secondary_quality_margin": self.secondary_quality_margin,
                    "primary_quality_score": getattr(primary_quality, "quality_score", None),
                    "secondary_quality_score": getattr(secondary_quality, "quality_score", None),
                    "primary_error": f"{type(primary_error).__name__}: {primary_error}" if primary_error is not None else None,
                    "secondary_error": f"{type(secondary_error).__name__}: {secondary_error}" if secondary_error is not None else None,
                },
            }
        )
        return EngineOutput(pages=output.pages, tables=output.tables, figures=output.figures, metadata=metadata)


def _score_engine_output_quality(input_path: Path, output: EngineOutput) -> object:
    from .quality import evaluate_document_quality
    from .structure import build_document

    document = build_document(str(input_path), output)
    return evaluate_document_quality(document)


class QualityRankedEnsembleEngine:
    name = "quality_ranked_ensemble"

    def __init__(
        self,
        *,
        primary_engine: str,
        primary_engine_kwargs: dict[str, object] | None = None,
        alternative_engines: list[dict[str, object]],
        selection_margin: float = 0.0,
    ) -> None:
        if primary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"quality_ranked_ensemble does not support primary_engine={primary_engine!r}")
        if not alternative_engines:
            raise EngineUnavailableError("quality_ranked_ensemble requires at least one alternative engine")
        self.primary = _RankedEngineSpec(
            label=str(primary_engine),
            engine=create_engine(primary_engine, **(primary_engine_kwargs or {})),
            quality_bias=0.0,
        )
        self.alternatives: list[_RankedEngineSpec] = []
        for index, item in enumerate(alternative_engines):
            if not isinstance(item, dict):
                raise EngineUnavailableError(f"quality_ranked_ensemble alternative {index} must be an object")
            engine_name = str(item.get("engine") or "")
            if not engine_name:
                raise EngineUnavailableError(f"quality_ranked_ensemble alternative {index} is missing engine")
            if engine_name in {"candidate", "ours", "auto"}:
                raise EngineUnavailableError(f"quality_ranked_ensemble does not support alternative engine={engine_name!r}")
            self.alternatives.append(
                _RankedEngineSpec(
                    label=str(item.get("label") or engine_name),
                    engine=create_engine(engine_name, **dict(item.get("engine_kwargs") or {})),
                    quality_bias=float(item.get("quality_bias", 0.0)),
                )
            )
        self.selection_margin = float(selection_margin)

    def recognize(self, input_path: Path) -> EngineOutput:
        scored_specs: list[tuple[_RankedEngineSpec, EngineOutput, object, float]] = []
        errors: list[str] = []
        for spec in [self.primary, *self.alternatives]:
            try:
                output = spec.engine.recognize(input_path)
                quality = _score_engine_output_quality(input_path, output)
                adjusted_quality = float(getattr(quality, "quality_score", 0.0)) + spec.quality_bias
                scored_specs.append((spec, output, quality, adjusted_quality))
            except Exception as exc:
                errors.append(f"{spec.label}: {type(exc).__name__}: {exc}")
        if not scored_specs:
            raise ParseError("quality_ranked_ensemble failed: " + "; ".join(errors))
        primary_scored = next(item for item in scored_specs if item[0] is self.primary)
        chosen = max(scored_specs, key=lambda item: item[3])
        if chosen is not primary_scored and chosen[3] <= primary_scored[3] + self.selection_margin:
            chosen = primary_scored
        spec, output, quality, adjusted_quality = chosen
        metadata = dict(output.metadata)
        metadata.update(
            {
                "engine": self.name,
                "quality_ranked_ensemble": {
                    "primary_engine": self.primary.label,
                    "chosen_engine": spec.label,
                    "selection_margin": self.selection_margin,
                    "adjusted_quality_score": adjusted_quality,
                    "quality_score": getattr(quality, "quality_score", None),
                    "alternatives": [
                        {
                            "label": item_spec.label,
                            "quality_score": getattr(item_quality, "quality_score", None),
                            "adjusted_quality_score": item_adjusted,
                            "quality_bias": item_spec.quality_bias,
                        }
                        for item_spec, _item_output, item_quality, item_adjusted in scored_specs
                    ],
                    "errors": errors,
                },
            }
        )
        return EngineOutput(pages=output.pages, tables=output.tables, figures=output.figures, metadata=metadata)


@dataclass(slots=True)
class _RankedEngineSpec:
    label: str
    engine: OcrEngine
    quality_bias: float


class ScriptSelectCompositeEngine:
    name = "script_select_composite"

    def __init__(
        self,
        *,
        primary_engine: str,
        secondary_engine: str,
        primary_engine_kwargs: dict[str, object] | None = None,
        secondary_engine_kwargs: dict[str, object] | None = None,
        script: str = "devanagari",
        primary_script_threshold: float = 0.2,
        routing_granularity: str = "document",
        secondary_structure_backfill: bool = False,
        routing_signal_engine: str = "primary",
        primary_guard_script: str | None = None,
        primary_guard_threshold: float = 0.5,
        primary_guard_min_confidence: float | None = None,
    ) -> None:
        if primary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"script_select_composite does not support primary_engine={primary_engine!r}")
        if secondary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"script_select_composite does not support secondary_engine={secondary_engine!r}")
        self.primary_engine = create_engine(primary_engine, **(primary_engine_kwargs or {}))
        self.secondary_engine = create_engine(secondary_engine, **(secondary_engine_kwargs or {}))
        self.script = script.strip().lower()
        self.primary_script_threshold = float(primary_script_threshold)
        self.routing_granularity = routing_granularity.strip().lower()
        self.secondary_structure_backfill = bool(secondary_structure_backfill)
        self.routing_signal_engine = routing_signal_engine.strip().lower()
        self.primary_guard_script = primary_guard_script.strip().lower() if isinstance(primary_guard_script, str) and primary_guard_script.strip() else None
        self.primary_guard_threshold = float(primary_guard_threshold)
        self.primary_guard_min_confidence = None if primary_guard_min_confidence is None else float(primary_guard_min_confidence)
        if self.primary_guard_min_confidence is not None and not 0.0 <= self.primary_guard_min_confidence <= 1.0:
            raise EngineUnavailableError(
                f"primary_guard_min_confidence must be between 0 and 1: {primary_guard_min_confidence!r}"
            )
        if self.routing_granularity not in {"document", "page", "line"}:
            raise EngineUnavailableError(f"unsupported script routing granularity: {routing_granularity!r}")
        if self.routing_signal_engine not in {"primary", "secondary", "both"}:
            raise EngineUnavailableError(f"unsupported script routing signal engine: {routing_signal_engine!r}")
        if self.primary_guard_script is not None:
            _script_ratio("x", self.primary_guard_script)

    def recognize(self, input_path: Path) -> EngineOutput:
        primary_output = self.primary_engine.recognize(input_path)
        primary_text = _engine_output_text(primary_output)
        primary_ratio = _script_ratio(primary_text, self.script)
        if self.routing_granularity == "page":
            secondary_output = self.secondary_engine.recognize(input_path)
            return self._route_pages(
                primary_output,
                secondary_output,
                primary_ratio=primary_ratio,
                primary_text=primary_text,
            )
        if self.routing_granularity == "line":
            secondary_output = self.secondary_engine.recognize(input_path)
            return self._route_lines(
                primary_output,
                secondary_output,
                primary_ratio=primary_ratio,
                primary_text=primary_text,
            )
        secondary_output = None
        secondary_text = ""
        secondary_ratio = 0.0
        if self.routing_signal_engine in {"secondary", "both"}:
            secondary_output = self.secondary_engine.recognize(input_path)
            secondary_text = _engine_output_text(secondary_output)
            secondary_ratio = _script_ratio(secondary_text, self.script)
        signal_ratio = _select_signal_ratio(primary_ratio, secondary_ratio, self.routing_signal_engine)
        signal_text = secondary_text if self.routing_signal_engine == "secondary" else primary_text
        primary_guard_blocks_secondary, primary_guard_ratio = self._primary_guard_blocks_secondary(primary_text)
        if signal_ratio > self.primary_script_threshold and not primary_guard_blocks_secondary:
            if secondary_output is None:
                secondary_output = self.secondary_engine.recognize(input_path)
            return self._finalize_output(
                secondary_output,
                chosen_engine=getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                primary_ratio=primary_ratio,
                primary_text=primary_text,
                secondary_ratio=secondary_ratio,
                signal_ratio=signal_ratio,
                signal_text=signal_text,
                primary_guard_ratio=primary_guard_ratio,
                primary_guard_applied=primary_guard_blocks_secondary,
            )
        if self.secondary_structure_backfill:
            if secondary_output is None:
                secondary_output = self.secondary_engine.recognize(input_path)
            backfilled_output, structure_metadata = _backfill_structure_from_secondary(
                primary_output,
                secondary_output,
                page_indexes={page.page_index for page in primary_output.pages},
            )
            backfilled_output.metadata.update(structure_metadata)
            return self._finalize_output(
                backfilled_output,
                chosen_engine=getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                primary_ratio=primary_ratio,
                primary_text=primary_text,
                secondary_ratio=secondary_ratio,
                signal_ratio=signal_ratio,
                signal_text=signal_text,
                primary_guard_ratio=primary_guard_ratio,
                primary_guard_applied=primary_guard_blocks_secondary,
            )
        return self._finalize_output(
            primary_output,
            chosen_engine=getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
            primary_ratio=primary_ratio,
            primary_text=primary_text,
            secondary_ratio=secondary_ratio,
            signal_ratio=signal_ratio,
            signal_text=signal_text,
            primary_guard_ratio=primary_guard_ratio,
            primary_guard_applied=primary_guard_blocks_secondary,
        )

    def _route_pages(
        self,
        primary_output: EngineOutput,
        secondary_output: EngineOutput,
        *,
        primary_ratio: float,
        primary_text: str,
    ) -> EngineOutput:
        primary_pages = {page.page_index: page for page in primary_output.pages}
        secondary_pages = {page.page_index: page for page in secondary_output.pages}
        page_indexes = sorted(set(primary_pages) | set(secondary_pages))
        selected_pages: list[Page] = []
        page_decisions: list[dict[str, object]] = []
        secondary_page_count = 0
        for page_index in page_indexes:
            primary_page = primary_pages.get(page_index)
            secondary_page = secondary_pages.get(page_index)
            page_text = _page_text(primary_page)
            page_ratio = _script_ratio(page_text, self.script)
            secondary_page_text = _page_text(secondary_page)
            secondary_page_ratio = _script_ratio(secondary_page_text, self.script)
            signal_page_ratio = _select_signal_ratio(page_ratio, secondary_page_ratio, self.routing_signal_engine)
            primary_guard_blocks_secondary, primary_guard_ratio = self._primary_guard_blocks_secondary(
                page_text,
                confidence=_page_mean_confidence(primary_page),
            )
            choose_secondary = (
                primary_page is not None
                and secondary_page is not None
                and signal_page_ratio > self.primary_script_threshold
                and not primary_guard_blocks_secondary
            )
            if choose_secondary:
                selected_pages.append(secondary_page)
                chosen_engine = getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__)
                secondary_page_count += 1
            elif primary_page is not None:
                selected_pages.append(primary_page)
                chosen_engine = getattr(self.primary_engine, "name", type(self.primary_engine).__name__)
            elif secondary_page is not None:
                selected_pages.append(secondary_page)
                chosen_engine = getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__)
                secondary_page_count += 1
            else:
                continue
            page_decisions.append(
                {
                    "page_index": page_index,
                    "chosen_engine": chosen_engine,
                    "primary_script_ratio": page_ratio,
                    "secondary_script_ratio": secondary_page_ratio,
                    "signal_script_ratio": signal_page_ratio,
                    "primary_guard_script": self.primary_guard_script,
                    "primary_guard_ratio": primary_guard_ratio,
                    "primary_guard_min_confidence": self.primary_guard_min_confidence,
                    "primary_guard_confidence": _page_mean_confidence(primary_page),
                    "primary_guard_applied": primary_guard_blocks_secondary,
                    "primary_text_preview": page_text[:120],
                    "secondary_text_preview": secondary_page_text[:120],
                }
            )
        selected_page_indexes = {page.page_index for page in selected_pages}
        secondary_engine_name = getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__)
        chosen_engine = (
            "mixed"
            if secondary_page_count and secondary_page_count != len(selected_pages)
            else secondary_engine_name
            if secondary_page_count
            else getattr(self.primary_engine, "name", type(self.primary_engine).__name__)
        )
        metadata = {
            "page_routing": page_decisions,
            "selected_page_indexes": sorted(selected_page_indexes),
        }
        primary_guard_applied = any(bool(item.get("primary_guard_applied")) for item in page_decisions)
        if self.secondary_structure_backfill:
            primary_page_indexes = {
                int(item["page_index"])
                for item in page_decisions
                if item.get("chosen_engine") == getattr(self.primary_engine, "name", type(self.primary_engine).__name__)
            }
            merged_output, structure_metadata = _backfill_structure_from_secondary(
                EngineOutput(
                    pages=selected_pages,
                    tables=_select_page_items(primary_output.tables, secondary_output.tables, page_decisions, secondary_engine_name),
                    figures=_select_page_items(primary_output.figures, secondary_output.figures, page_decisions, secondary_engine_name),
                    metadata=metadata,
                ),
                secondary_output,
                page_indexes=primary_page_indexes,
            )
            merged_output.metadata.update(structure_metadata)
            return self._finalize_output(
                merged_output,
                chosen_engine=chosen_engine,
                primary_ratio=primary_ratio,
                primary_text=primary_text,
                secondary_ratio=_script_ratio(_engine_output_text(secondary_output), self.script),
                signal_ratio=_select_signal_ratio(
                    primary_ratio,
                    _script_ratio(_engine_output_text(secondary_output), self.script),
                    self.routing_signal_engine,
                ),
                signal_text=_engine_output_text(secondary_output) if self.routing_signal_engine == "secondary" else primary_text,
                primary_guard_ratio=self._primary_guard_ratio(primary_text),
                primary_guard_applied=primary_guard_applied,
            )
        return self._finalize_output(
            EngineOutput(
                pages=selected_pages,
                tables=_select_page_items(primary_output.tables, secondary_output.tables, page_decisions, secondary_engine_name),
                figures=_select_page_items(primary_output.figures, secondary_output.figures, page_decisions, secondary_engine_name),
                metadata=metadata,
            ),
            chosen_engine=chosen_engine,
            primary_ratio=primary_ratio,
            primary_text=primary_text,
            secondary_ratio=_script_ratio(_engine_output_text(secondary_output), self.script),
            signal_ratio=_select_signal_ratio(
                primary_ratio,
                _script_ratio(_engine_output_text(secondary_output), self.script),
                self.routing_signal_engine,
            ),
            signal_text=_engine_output_text(secondary_output) if self.routing_signal_engine == "secondary" else primary_text,
            primary_guard_ratio=self._primary_guard_ratio(primary_text),
            primary_guard_applied=primary_guard_applied,
        )

    def _route_lines(
        self,
        primary_output: EngineOutput,
        secondary_output: EngineOutput,
        *,
        primary_ratio: float,
        primary_text: str,
    ) -> EngineOutput:
        primary_pages = {page.page_index: page for page in primary_output.pages}
        secondary_pages = {page.page_index: page for page in secondary_output.pages}
        page_indexes = sorted(set(primary_pages) | set(secondary_pages))
        selected_pages: list[Page] = []
        line_decisions: list[dict[str, object]] = []
        secondary_line_count = 0
        primary_engine_name = getattr(self.primary_engine, "name", type(self.primary_engine).__name__)
        secondary_engine_name = getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__)
        for page_index in page_indexes:
            primary_page = primary_pages.get(page_index)
            secondary_page = secondary_pages.get(page_index)
            if primary_page is None and secondary_page is not None:
                selected_pages.append(secondary_page)
                secondary_line_count += len(secondary_page.text_lines)
                for line in secondary_page.text_lines:
                    line_decisions.append(
                        {
                            "page_index": page_index,
                            "line_id": line.line_id,
                            "chosen_engine": secondary_engine_name,
                            "primary_script_ratio": None,
                            "secondary_script_ratio": _script_ratio(line.text, self.script),
                            "signal_script_ratio": _script_ratio(line.text, self.script),
                            "reason": "missing_primary_page",
                        }
                    )
                continue
            if primary_page is None:
                continue
            secondary_lines = list(secondary_page.text_lines) if secondary_page is not None else []
            unmatched_secondary = set(range(len(secondary_lines)))
            routed_lines: list[TextLine] = []
            for line in primary_page.text_lines:
                line_ratio = _script_ratio(line.text, self.script)
                matched_index, alignment_iou, alignment_distance = _best_line_match(line, secondary_lines, unmatched_secondary)
                secondary_line_ratio = (
                    _script_ratio(secondary_lines[matched_index].text, self.script)
                    if matched_index is not None
                    else 0.0
                )
                signal_line_ratio = _select_signal_ratio(line_ratio, secondary_line_ratio, self.routing_signal_engine)
                primary_guard_blocks_secondary, primary_guard_ratio = self._primary_guard_blocks_secondary(
                    line.text,
                    confidence=line.confidence,
                )
                choose_secondary = (
                    signal_line_ratio > self.primary_script_threshold
                    and matched_index is not None
                    and not primary_guard_blocks_secondary
                )
                if choose_secondary:
                    matched_line = secondary_lines[matched_index]
                    unmatched_secondary.discard(matched_index)
                    metadata = dict(line.metadata)
                    metadata.update(
                        {
                            "script_select_composite": {
                                "chosen_engine": secondary_engine_name,
                                "primary_text": line.text,
                                "secondary_line_id": matched_line.line_id,
                                "primary_script_ratio": line_ratio,
                                "secondary_script_ratio": secondary_line_ratio,
                                "signal_script_ratio": signal_line_ratio,
                                "primary_guard_script": self.primary_guard_script,
                                "primary_guard_ratio": primary_guard_ratio,
                                "primary_guard_min_confidence": self.primary_guard_min_confidence,
                                "primary_guard_confidence": line.confidence,
                                "primary_guard_applied": primary_guard_blocks_secondary,
                                "alignment_iou": alignment_iou,
                                "alignment_center_distance": alignment_distance,
                            }
                        }
                    )
                    routed_lines.append(
                        replace(
                            line,
                            text=matched_line.text,
                            confidence=matched_line.confidence,
                            metadata=metadata,
                        )
                    )
                    chosen_engine = secondary_engine_name
                    secondary_line_count += 1
                else:
                    routed_lines.append(line)
                    chosen_engine = primary_engine_name
                line_decisions.append(
                    {
                        "page_index": page_index,
                        "line_id": line.line_id,
                        "chosen_engine": chosen_engine,
                        "primary_script_ratio": line_ratio,
                        "secondary_script_ratio": secondary_line_ratio if matched_index is not None else None,
                        "signal_script_ratio": signal_line_ratio,
                        "primary_guard_script": self.primary_guard_script,
                        "primary_guard_ratio": primary_guard_ratio,
                        "primary_guard_min_confidence": self.primary_guard_min_confidence,
                        "primary_guard_confidence": line.confidence,
                        "primary_guard_applied": primary_guard_blocks_secondary,
                        "matched_secondary_line_id": secondary_lines[matched_index].line_id if matched_index is not None else None,
                        "alignment_iou": alignment_iou if matched_index is not None else None,
                        "alignment_center_distance": alignment_distance if matched_index is not None else None,
                    }
                )
            selected_pages.append(replace(primary_page, text_lines=routed_lines))
        selected_page_indexes = {page.page_index for page in selected_pages}
        chosen_engine = "mixed" if secondary_line_count else primary_engine_name
        primary_guard_applied = any(bool(item.get("primary_guard_applied")) for item in line_decisions)
        output = EngineOutput(
            pages=selected_pages,
            tables=list(primary_output.tables),
            figures=list(primary_output.figures),
            metadata={
                "line_routing": line_decisions,
                "selected_page_indexes": sorted(selected_page_indexes),
                "secondary_line_count": secondary_line_count,
            },
        )
        if self.secondary_structure_backfill:
            output, structure_metadata = _backfill_structure_from_secondary(
                output,
                secondary_output,
                page_indexes=selected_page_indexes,
            )
            output.metadata.update(structure_metadata)
        return self._finalize_output(
            output,
            chosen_engine=chosen_engine,
            primary_ratio=primary_ratio,
            primary_text=primary_text,
            secondary_ratio=_script_ratio(_engine_output_text(secondary_output), self.script),
            signal_ratio=_select_signal_ratio(
                primary_ratio,
                _script_ratio(_engine_output_text(secondary_output), self.script),
                self.routing_signal_engine,
            ),
            signal_text=_engine_output_text(secondary_output) if self.routing_signal_engine == "secondary" else primary_text,
            primary_guard_ratio=self._primary_guard_ratio(primary_text),
            primary_guard_applied=primary_guard_applied,
        )

    def _primary_guard_ratio(self, text: str) -> float | None:
        if self.primary_guard_script is None:
            return None
        return _script_ratio(text, self.primary_guard_script)

    def _primary_guard_blocks_secondary(self, text: str, *, confidence: float | None = None) -> tuple[bool, float | None]:
        ratio = self._primary_guard_ratio(text)
        if ratio is None:
            return False, None
        if self.primary_guard_min_confidence is not None:
            if confidence is None or confidence < self.primary_guard_min_confidence:
                return False, ratio
        return ratio >= self.primary_guard_threshold, ratio

    def _finalize_output(
        self,
        output: EngineOutput,
        *,
        chosen_engine: str,
        primary_ratio: float,
        primary_text: str,
        secondary_ratio: float | None = None,
        signal_ratio: float | None = None,
        signal_text: str | None = None,
        primary_guard_ratio: float | None = None,
        primary_guard_applied: bool = False,
    ) -> EngineOutput:
        metadata = dict(output.metadata)
        routing_metadata = {}
        for key in ("page_routing", "line_routing", "selected_page_indexes", "secondary_line_count"):
            if key in metadata:
                routing_metadata[key] = metadata.pop(key)
        metadata.update(
            {
                "engine": self.name,
                "script_select_composite": {
                    "primary_engine": getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                    "secondary_engine": getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                    "chosen_engine": chosen_engine,
                    "script": self.script,
                    "primary_script_threshold": self.primary_script_threshold,
                    "routing_granularity": self.routing_granularity,
                    "routing_signal_engine": self.routing_signal_engine,
                    "secondary_structure_backfill": self.secondary_structure_backfill,
                    "primary_guard_script": self.primary_guard_script,
                    "primary_guard_threshold": self.primary_guard_threshold,
                    "primary_guard_min_confidence": self.primary_guard_min_confidence,
                    "primary_guard_ratio": primary_guard_ratio,
                    "primary_guard_applied": primary_guard_applied,
                    "primary_script_ratio": primary_ratio,
                    "secondary_script_ratio": secondary_ratio,
                    "signal_script_ratio": primary_ratio if signal_ratio is None else signal_ratio,
                    "primary_text_preview": primary_text[:120],
                    "signal_text_preview": (primary_text if signal_text is None else signal_text)[:120],
                    **routing_metadata,
                },
            }
        )
        return EngineOutput(pages=output.pages, tables=output.tables, figures=output.figures, metadata=metadata)


ScriptCodepointRange = tuple[int, int]

_DEVANAGARI_RANGES: tuple[ScriptCodepointRange, ...] = (
    (0x0900, 0x097F),
    (0xA8E0, 0xA8FF),
)
_LIMBU_RANGES: tuple[ScriptCodepointRange, ...] = ((0x1900, 0x194F),)
_LATIN_RANGES: tuple[ScriptCodepointRange, ...] = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00AA, 0x00AA),
    (0x00B5, 0x00B5),
    (0x00BA, 0x00BA),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x00FF),
)
_SUNUWAR_RANGES: tuple[ScriptCodepointRange, ...] = ((0x11BC0, 0x11BFF),)
SCRIPT_CODEPOINT_RANGES: dict[str, tuple[ScriptCodepointRange, ...]] = {
    "devanagari": _DEVANAGARI_RANGES,
    "limbu": _LIMBU_RANGES,
    "sirijonga": _LIMBU_RANGES,
    "limbu/sirijonga": _LIMBU_RANGES,
    "limbu-sirijonga": _LIMBU_RANGES,
    "latin": _LATIN_RANGES,
    "english": _LATIN_RANGES,
    "sunuwar": _SUNUWAR_RANGES,
    "jenticha": _SUNUWAR_RANGES,
    "sunuwar/jenticha": _SUNUWAR_RANGES,
    "lepcha": ((0x1C00, 0x1C4F),),
    "kirat_rai": ((0x16D40, 0x16D7F),),
    **NEWARI_TRANSCRIPTION_ROUTING_RANGES,
    "tirhuta": ((0x11480, 0x114DF),),
    "gurung_khema": ((0x16100, 0x1613F),),
    "ol_chiki": ((0x1C50, 0x1C7F),),
    "tibetan": ((0x0F00, 0x0FFF),),
}


def _select_signal_ratio(primary_ratio: float, secondary_ratio: float, routing_signal_engine: str) -> float:
    if routing_signal_engine == "secondary":
        return secondary_ratio
    if routing_signal_engine == "both":
        return max(primary_ratio, secondary_ratio)
    return primary_ratio


def _engine_output_text(output: EngineOutput) -> str:
    return "\n".join(line.text for page in output.pages for line in page.text_lines if line.text.strip())


def _page_text(page: Page | None) -> str:
    if page is None:
        return ""
    return "\n".join(line.text for line in page.text_lines if line.text.strip())


def _page_mean_confidence(page: Page | None) -> float | None:
    if page is None:
        return None
    confidences = [line.confidence for line in page.text_lines if line.confidence is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _select_page_items(
    primary_items: list[Table] | list[Figure],
    secondary_items: list[Table] | list[Figure],
    page_decisions: list[dict[str, object]],
    secondary_engine_name: str,
) -> list[Table] | list[Figure]:
    primary_by_page: dict[int, list[Table] | list[Figure]] = {}
    secondary_by_page: dict[int, list[Table] | list[Figure]] = {}
    for item in primary_items:
        primary_by_page.setdefault(item.page_index, []).append(item)
    for item in secondary_items:
        secondary_by_page.setdefault(item.page_index, []).append(item)
    selected: list[Table] | list[Figure] = []
    for decision in page_decisions:
        page_index = decision.get("page_index")
        if not isinstance(page_index, int):
            continue
        chosen_engine = decision.get("chosen_engine")
        if chosen_engine == secondary_engine_name:
            selected.extend(secondary_by_page.get(page_index, []))
        else:
            selected.extend(primary_by_page.get(page_index, []))
    return selected


def _backfill_structure_from_secondary(
    primary_output: EngineOutput,
    secondary_output: EngineOutput,
    *,
    page_indexes: set[int],
) -> tuple[EngineOutput, dict[str, object]]:
    page_indexes = {int(page_index) for page_index in page_indexes}
    primary_tables_by_page: dict[int, list[Table]] = {}
    secondary_tables_by_page: dict[int, list[Table]] = {}
    primary_figures_by_page: dict[int, list[Figure]] = {}
    secondary_figures_by_page: dict[int, list[Figure]] = {}
    for table in primary_output.tables:
        primary_tables_by_page.setdefault(table.page_index, []).append(table)
    for table in secondary_output.tables:
        secondary_tables_by_page.setdefault(table.page_index, []).append(table)
    for figure in primary_output.figures:
        primary_figures_by_page.setdefault(figure.page_index, []).append(figure)
    for figure in secondary_output.figures:
        secondary_figures_by_page.setdefault(figure.page_index, []).append(figure)

    tables = list(primary_output.tables)
    figures = list(primary_output.figures)
    backfilled_table_pages: list[int] = []
    backfilled_figure_pages: list[int] = []
    for page_index in sorted(page_indexes):
        if not primary_tables_by_page.get(page_index) and secondary_tables_by_page.get(page_index):
            tables.extend(secondary_tables_by_page[page_index])
            backfilled_table_pages.append(page_index)
        if not primary_figures_by_page.get(page_index) and secondary_figures_by_page.get(page_index):
            figures.extend(secondary_figures_by_page[page_index])
            backfilled_figure_pages.append(page_index)
    metadata = {
        "structure_backfill": {
            "secondary_tables_pages": backfilled_table_pages,
            "secondary_figures_pages": backfilled_figure_pages,
        }
    }
    return EngineOutput(pages=primary_output.pages, tables=tables, figures=figures, metadata=dict(primary_output.metadata)), metadata


def _script_ratio(text: str, script: str) -> float:
    ranges = _script_ranges(script)
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    return _script_codepoint_count(non_space, ranges=ranges) / len(non_space)


def _script_ranges(script: str) -> tuple[ScriptCodepointRange, ...]:
    normalized = script.strip().lower()
    if normalized in SCRIPT_CODEPOINT_RANGES:
        return SCRIPT_CODEPOINT_RANGES[normalized]
    supported = ", ".join(sorted(SCRIPT_CODEPOINT_RANGES))
    raise EngineUnavailableError(f"unsupported script routing target: {script!r}; supported scripts: {supported}")


def _script_codepoint_count(
    chars: Iterable[str],
    *,
    ranges: tuple[ScriptCodepointRange, ...] | None = None,
    script: str | None = None,
) -> int:
    if ranges is None:
        if script is None:
            raise RuntimeError("script or ranges is required for script codepoint counting")
        ranges = _script_ranges(script)
    return sum(1 for char in chars if _codepoint_in_ranges(ord(char), ranges))


def _codepoint_in_ranges(codepoint: int, ranges: tuple[ScriptCodepointRange, ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


class LineAlignCompositeEngine:
    name = "line_align_composite"

    def __init__(
        self,
        *,
        primary_engine: str,
        secondary_engine: str,
        primary_engine_kwargs: dict[str, object] | None = None,
        secondary_engine_kwargs: dict[str, object] | None = None,
        append_unmatched_primary: bool = True,
        secondary_structure_backfill: bool = False,
    ) -> None:
        if primary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"line_align_composite does not support primary_engine={primary_engine!r}")
        if secondary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"line_align_composite does not support secondary_engine={secondary_engine!r}")
        self.primary_engine = create_engine(primary_engine, **(primary_engine_kwargs or {}))
        self.secondary_engine = create_engine(secondary_engine, **(secondary_engine_kwargs or {}))
        self.append_unmatched_primary = bool(append_unmatched_primary)
        self.secondary_structure_backfill = bool(secondary_structure_backfill)

    def recognize(self, input_path: Path) -> EngineOutput:
        primary_output = self.primary_engine.recognize(input_path)
        secondary_output = self.secondary_engine.recognize(input_path)
        primary_pages = {page.page_index: page for page in primary_output.pages}
        secondary_pages = {page.page_index: page for page in secondary_output.pages}
        page_indexes = sorted(set(primary_pages) | set(secondary_pages))
        aligned_pages: list[Page] = []
        page_alignment: list[dict[str, object]] = []
        total_matches = 0
        for page_index in page_indexes:
            primary_page = primary_pages.get(page_index)
            secondary_page = secondary_pages.get(page_index)
            aligned_page, alignment_meta = _align_page_lines(
                primary_page,
                secondary_page,
                append_unmatched_primary=self.append_unmatched_primary,
            )
            total_matches += int(alignment_meta.get("matched_line_count", 0))
            page_alignment.append({"page_index": page_index, **alignment_meta})
            aligned_pages.append(aligned_page)
        output = EngineOutput(
            pages=aligned_pages,
            tables=list(primary_output.tables),
            figures=list(primary_output.figures),
            metadata={
                "engine": self.name,
                "line_align_composite": {
                    "primary_engine": getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                    "secondary_engine": getattr(self.secondary_engine, "name", type(self.secondary_engine).__name__),
                    "append_unmatched_primary": self.append_unmatched_primary,
                    "secondary_structure_backfill": self.secondary_structure_backfill,
                    "matched_line_count": total_matches,
                    "page_alignment": page_alignment,
                },
            },
        )
        if self.secondary_structure_backfill:
            output, structure_metadata = _backfill_structure_from_secondary(
                output,
                secondary_output,
                page_indexes={page.page_index for page in aligned_pages},
            )
            output.metadata.update(structure_metadata)
        return output


def _align_page_lines(
    primary_page: Page | None,
    secondary_page: Page | None,
    *,
    append_unmatched_primary: bool,
) -> tuple[Page, dict[str, object]]:
    if primary_page is None and secondary_page is None:
        return Page(page_index=0, text_lines=[], metadata={}), {"matched_line_count": 0, "unmatched_primary_count": 0, "used_secondary_scaffold": False}
    if secondary_page is None and primary_page is not None:
        return primary_page, {"matched_line_count": len(primary_page.text_lines), "unmatched_primary_count": 0, "used_secondary_scaffold": False}
    if primary_page is None and secondary_page is not None:
        return secondary_page, {"matched_line_count": 0, "unmatched_primary_count": 0, "used_secondary_scaffold": True}

    assert primary_page is not None
    assert secondary_page is not None
    unmatched_primary = set(range(len(primary_page.text_lines)))
    aligned_lines: list[TextLine] = []
    matched_count = 0
    for secondary_line in secondary_page.text_lines:
        matched_index, alignment_iou, alignment_distance = _best_line_match(secondary_line, primary_page.text_lines, unmatched_primary)
        if matched_index is None:
            aligned_lines.append(secondary_line)
            continue
        matched_primary = primary_page.text_lines[matched_index]
        unmatched_primary.discard(matched_index)
        matched_count += 1
        metadata = dict(secondary_line.metadata)
        metadata.update(
            {
                "aligned_primary_line_id": matched_primary.line_id,
                "alignment_iou": alignment_iou,
                "alignment_center_distance": alignment_distance,
            }
        )
        aligned_lines.append(
            TextLine(
                text=matched_primary.text,
                bbox=secondary_line.bbox,
                confidence=matched_primary.confidence,
                page_index=secondary_line.page_index,
                line_id=secondary_line.line_id,
                metadata=metadata,
            )
        )
    if append_unmatched_primary:
        for primary_index in sorted(unmatched_primary):
            aligned_lines.append(primary_page.text_lines[primary_index])
    aligned_page = Page(
        page_index=secondary_page.page_index,
        width=secondary_page.width,
        height=secondary_page.height,
        text_lines=aligned_lines,
        metadata=dict(secondary_page.metadata),
    )
    return aligned_page, {
        "matched_line_count": matched_count,
        "unmatched_primary_count": len(unmatched_primary),
        "used_secondary_scaffold": True,
    }


def _best_line_match(
    target_line: TextLine,
    candidate_lines: list[TextLine],
    unmatched_indexes: set[int],
) -> tuple[int | None, float, float]:
    best_index: int | None = None
    best_iou = -1.0
    best_distance = float("inf")
    for index in unmatched_indexes:
        candidate = candidate_lines[index]
        overlap = bbox_iou(target_line.bbox, candidate.bbox)
        distance = _bbox_center_distance(target_line.bbox, candidate.bbox)
        if overlap > best_iou or (overlap == best_iou and distance < best_distance):
            best_index = index
            best_iou = overlap
            best_distance = distance
    if best_index is None:
        return None, 0.0, float("inf")
    return best_index, best_iou, best_distance


def _bbox_center_distance(left: BBox, right: BBox) -> float:
    left_center_x = left.x + left.w / 2.0
    left_center_y = left.y + left.h / 2.0
    right_center_x = right.x + right.w / 2.0
    right_center_y = right.y + right.h / 2.0
    return ((left_center_x - right_center_x) ** 2 + (left_center_y - right_center_y) ** 2) ** 0.5


class TableCellRefineCompositeEngine:
    name = "table_cell_refine_composite"

    def __init__(
        self,
        *,
        structure_engine: str = "surya",
        crop_engine: str = "tesseract",
        structure_engine_kwargs: dict[str, object] | None = None,
        crop_engine_kwargs: dict[str, object] | None = None,
        crop_padding: int = 4,
        min_replacement_chars: int = 1,
        replacement_policy: str = "empty",
        min_replacement_confidence_delta: float = 0.05,
        deduplicate_tables: bool = False,
        duplicate_iou_threshold: float = 0.9,
    ) -> None:
        if structure_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"table_cell_refine_composite does not support structure_engine={structure_engine!r}")
        if crop_engine in {"ours", "auto"}:
            raise EngineUnavailableError(f"table_cell_refine_composite does not support crop_engine={crop_engine!r}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise EngineUnavailableError("table cell refinement requires Pillow. Install ocr-tech[eval].") from exc
        self.structure_engine = create_engine(structure_engine, **(structure_engine_kwargs or {}))
        self.crop_engine = create_engine(crop_engine, **(crop_engine_kwargs or {}))
        self.crop_padding = int(crop_padding)
        self.min_replacement_chars = int(min_replacement_chars)
        self.replacement_policy = replacement_policy.strip().lower()
        self.min_replacement_confidence_delta = float(min_replacement_confidence_delta)
        self.deduplicate_tables = bool(deduplicate_tables)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        if self.crop_padding < 0:
            raise EngineUnavailableError("table_cell_refine_composite crop_padding must be non-negative")
        if self.min_replacement_chars < 1:
            raise EngineUnavailableError("table_cell_refine_composite min_replacement_chars must be at least 1")
        if self.replacement_policy not in {"empty", "always", "guarded"}:
            raise EngineUnavailableError("table_cell_refine_composite replacement_policy must be empty, guarded, or always")
        if self.min_replacement_confidence_delta < 0:
            raise EngineUnavailableError("table_cell_refine_composite min_replacement_confidence_delta must be non-negative")
        if not 0.0 <= self.duplicate_iou_threshold <= 1.0:
            raise EngineUnavailableError("table_cell_refine_composite duplicate_iou_threshold must be between 0 and 1")
        self._image_module = Image

    def recognize(self, input_path: Path) -> EngineOutput:
        structure_output = self.structure_engine.recognize(input_path)
        page_paths = _input_page_paths(input_path)
        refined_tables: list[Table] = []
        refined_count = 0
        attempted_count = 0
        error_count = 0
        dropped_duplicate_tables = 0
        with tempfile.TemporaryDirectory(prefix="ocrtech-cell-refine-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for table in structure_output.tables:
                page_path = page_paths.get(table.page_index)
                if page_path is None:
                    refined_tables.append(table)
                    continue
                try:
                    with self._image_module.open(page_path).convert("RGB") as page_image:
                        refined_table, table_meta = self._refine_table(table, page_image, tmp_path)
                except Exception as exc:
                    metadata = dict(table.metadata)
                    metadata["table_cell_refine_error"] = f"{type(exc).__name__}: {exc}"
                    refined_tables.append(replace(table, metadata=metadata))
                    error_count += 1
                    continue
                refined_tables.append(refined_table)
                refined_count += int(table_meta["replaced_cells"])
                attempted_count += int(table_meta["attempted_cells"])
                error_count += int(table_meta["error_cells"])
        if self.deduplicate_tables:
            refined_tables, dropped_duplicate_tables = _deduplicate_overlapping_tables(refined_tables, iou_threshold=self.duplicate_iou_threshold)
        metadata = dict(structure_output.metadata)
        metadata.update(
            {
                "engine": self.name,
                "table_cell_refine_composite": {
                    "structure_engine": getattr(self.structure_engine, "name", type(self.structure_engine).__name__),
                    "crop_engine": getattr(self.crop_engine, "name", type(self.crop_engine).__name__),
                    "crop_padding": self.crop_padding,
                    "min_replacement_chars": self.min_replacement_chars,
                    "replacement_policy": self.replacement_policy,
                    "min_replacement_confidence_delta": self.min_replacement_confidence_delta,
                    "deduplicate_tables": self.deduplicate_tables,
                    "duplicate_iou_threshold": self.duplicate_iou_threshold,
                    "dropped_duplicate_tables": dropped_duplicate_tables,
                    "attempted_cells": attempted_count,
                    "replaced_cells": refined_count,
                    "error_cells": error_count,
                },
            }
        )
        return EngineOutput(
            pages=structure_output.pages,
            tables=refined_tables,
            figures=structure_output.figures,
            metadata=metadata,
        )

    def _refine_table(self, table: Table, page_image: object, tmp_dir: Path) -> tuple[Table, dict[str, int]]:
        cells = list(table.cells)
        if not cells:
            return table, {"attempted_cells": 0, "replaced_cells": 0, "error_cells": 0}
        rows = max(cell.row + cell.rowspan for cell in cells)
        cols = max(cell.col + cell.colspan for cell in cells)
        refined_cells: list[TableCell] = []
        attempted = 0
        replaced = 0
        errors = 0
        skipped_non_empty = 0
        skipped_guarded = 0
        for cell_index, cell in enumerate(cells):
            if self.replacement_policy == "empty" and cell.text.strip():
                refined_cells.append(_replace_cell(cell, bbox=cell.bbox if cell.bbox is not None else _infer_cell_bbox(table.bbox, cell, rows=rows, cols=cols)))
                skipped_non_empty += 1
                continue
            bbox = cell.bbox if cell.bbox is not None and cell.bbox.w > 0 and cell.bbox.h > 0 else _infer_cell_bbox(table.bbox, cell, rows=rows, cols=cols)
            crop = _crop_pil_image(page_image, _expand_bbox(bbox, self.crop_padding), self._image_module)
            crop_path = tmp_dir / f"{table.table_id}-r{cell.row:03d}-c{cell.col:03d}-{cell_index:04d}.png"
            crop.save(crop_path)
            attempted += 1
            try:
                crop_output = self.crop_engine.recognize(crop_path)
                crop_text = _engine_output_text(crop_output).strip()
                crop_confidence = crop_output.average_confidence
            except Exception as exc:
                metadata = {"cell_refine_error": f"{type(exc).__name__}: {exc}"}
                refined_cells.append(_replace_cell(cell, bbox=bbox, metadata=metadata))
                errors += 1
                continue
            normalized = normalize_ocr_text(crop_text)
            if len(normalized) >= self.min_replacement_chars:
                should_replace, guard_reason = self._should_replace_cell(cell, normalized, crop_confidence)
                if not should_replace:
                    metadata = {"cell_refine_guard": guard_reason} if guard_reason else None
                    refined_cells.append(_replace_cell(cell, bbox=bbox, metadata=metadata))
                    skipped_guarded += 1
                    continue
                refined_cells.append(_replace_cell(cell, text=normalized, bbox=bbox, confidence=crop_confidence))
                if normalized != cell.text:
                    replaced += 1
            else:
                refined_cells.append(_replace_cell(cell, bbox=bbox))
        from .tables import cells_to_html

        metadata = dict(table.metadata)
        metadata["table_cell_refine"] = {
            "attempted_cells": attempted,
            "replaced_cells": replaced,
            "error_cells": errors,
            "skipped_non_empty_cells": skipped_non_empty,
            "skipped_guarded_cells": skipped_guarded,
            "crop_engine": getattr(self.crop_engine, "name", type(self.crop_engine).__name__),
        }
        refined_table = replace(table, cells=refined_cells, html=cells_to_html(refined_cells), metadata=metadata)
        return refined_table, {"attempted_cells": attempted, "replaced_cells": replaced, "error_cells": errors}

    def _should_replace_cell(self, cell: TableCell, replacement_text: str, replacement_confidence: float | None) -> tuple[bool, str | None]:
        if self.replacement_policy == "always":
            return True, None
        if not cell.text.strip():
            return True, None
        if self.replacement_policy == "empty":
            return False, "non_empty_cell"
        if replacement_text == cell.text:
            return False, "same_text"
        if replacement_confidence is None:
            return False, "missing_replacement_confidence"
        if cell.confidence is None:
            return False, "missing_existing_confidence"
        if replacement_confidence < cell.confidence + self.min_replacement_confidence_delta:
            return False, "replacement_confidence_not_higher"
        return True, None


def _deduplicate_overlapping_tables(tables: list[Table], *, iou_threshold: float) -> tuple[list[Table], int]:
    kept: list[Table] = []
    dropped = 0
    for table in sorted(tables, key=_table_dedup_sort_key):
        duplicate_index = None
        for index, existing in enumerate(kept):
            if table.page_index != existing.page_index:
                continue
            if bbox_iou(table.bbox, existing.bbox) >= iou_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(table)
            continue
        existing = kept[duplicate_index]
        if _table_quality_score(table) > _table_quality_score(existing):
            kept[duplicate_index] = table
        dropped += 1
    return sorted(kept, key=lambda item: (item.page_index, item.bbox.y, item.bbox.x)), dropped


def _table_dedup_sort_key(table: Table) -> tuple[float, float, float, int, int, float]:
    structure_score, non_empty, confidence = _table_quality_score(table)
    return (table.page_index, table.bbox.y, table.bbox.x, -structure_score, -non_empty, -confidence)


def _table_quality_score(table: Table) -> tuple[int, int, float]:
    if not table.cells:
        return (0, 0, float(table.confidence or 0.0))
    rows = max(cell.row + cell.rowspan for cell in table.cells)
    cols = max(cell.col + cell.colspan for cell in table.cells)
    non_empty = sum(1 for cell in table.cells if cell.text.strip())
    rectangular_cells = rows * cols
    structure_score = min(len(table.cells), rectangular_cells)
    return (structure_score, non_empty, float(table.confidence or 0.0))


def _input_page_paths(input_path: Path) -> dict[int, Path]:
    if input_path.is_dir():
        return {index: page_path for index, page_path in enumerate(_bundle_page_paths(input_path))}
    return {0: input_path}


def _infer_cell_bbox(table_bbox: BBox, cell: TableCell, *, rows: int, cols: int) -> BBox:
    col_width = table_bbox.w / max(cols, 1)
    row_height = table_bbox.h / max(rows, 1)
    return BBox(
        table_bbox.x + col_width * cell.col,
        table_bbox.y + row_height * cell.row,
        col_width * cell.colspan,
        row_height * cell.rowspan,
    )


def _expand_bbox(bbox: BBox, padding: int) -> BBox:
    return BBox(bbox.x - padding, bbox.y - padding, bbox.w + padding * 2, bbox.h + padding * 2)


def _crop_pil_image(image: object, bbox: BBox, image_module: object) -> object:
    width, height = image.size
    left = max(0, int(bbox.x))
    top = max(0, int(bbox.y))
    right = min(width, max(left + 1, int(bbox.right)))
    bottom = min(height, max(top + 1, int(bbox.bottom)))
    if right <= left or bottom <= top:
        return image_module.new("RGB", (1, 1), "white")
    return image.crop((left, top, right, bottom))


def _replace_cell(
    cell: TableCell,
    *,
    text: str | None = None,
    bbox: BBox | None = None,
    confidence: float | None = None,
    metadata: dict[str, object] | None = None,
) -> TableCell:
    cell_metadata = dict(cell.metadata)
    if metadata:
        cell_metadata.update(metadata)
    return TableCell(
        row=cell.row,
        col=cell.col,
        text=cell.text if text is None else text,
        rowspan=cell.rowspan,
        colspan=cell.colspan,
        bbox=bbox if bbox is not None else cell.bbox,
        confidence=confidence if confidence is not None else cell.confidence,
        metadata=cell_metadata,
    )


FIGURE_MARKER_LINE_RE = re.compile(r"^\s*(?:\[(?:figure|chart|image|चित्र)[^\]]*\]|(?:figure|fig\.?|chart|चित्र)\s*[:：]?)", re.IGNORECASE)
CAPTION_PREFIX_SPLIT_RE = re.compile(r"[\s\d०-९IVXivx.:-]+", re.UNICODE)


class FigureCaptionRefineCompositeEngine:
    name = "figure_caption_refine_composite"

    def __init__(
        self,
        *,
        primary_engine: str = "surya",
        caption_engine: str = "tesseract",
        primary_engine_kwargs: dict[str, object] | None = None,
        caption_engine_kwargs: dict[str, object] | None = None,
        min_caption_iou: float = 0.5,
        min_score_delta: int = 1,
        max_gap_multiplier: float = 1.5,
        correct_structural_labels: bool = False,
        table_name_lexicon_path: str | Path | None = None,
        table_name_lexicon: list[str] | None = None,
        table_name_max_distance: int = 2,
        infer_table_header_colspans: bool = True,
        normalize_table_grids: bool = True,
    ) -> None:
        if primary_engine in {"candidate", "ours", "auto"}:
            raise EngineUnavailableError(f"figure_caption_refine_composite does not support primary_engine={primary_engine!r}")
        if caption_engine in {"ours", "auto"}:
            raise EngineUnavailableError(f"figure_caption_refine_composite does not support caption_engine={caption_engine!r}")
        self.primary_engine = create_engine(primary_engine, **(primary_engine_kwargs or {}))
        self.caption_engine = create_engine(caption_engine, **(caption_engine_kwargs or {}))
        self.min_caption_iou = float(min_caption_iou)
        self.min_score_delta = int(min_score_delta)
        self.max_gap_multiplier = float(max_gap_multiplier)
        self.correct_structural_labels = bool(correct_structural_labels)
        self.table_name_lexicon = _load_table_name_lexicon(table_name_lexicon_path, table_name_lexicon)
        self.table_name_max_distance = int(table_name_max_distance)
        self.infer_table_header_colspans = bool(infer_table_header_colspans)
        self.normalize_table_grids = bool(normalize_table_grids)
        if not 0.0 <= self.min_caption_iou <= 1.0:
            raise EngineUnavailableError("figure_caption_refine_composite min_caption_iou must be between 0 and 1")
        if self.min_score_delta < 0:
            raise EngineUnavailableError("figure_caption_refine_composite min_score_delta must be non-negative")
        if self.max_gap_multiplier <= 0:
            raise EngineUnavailableError("figure_caption_refine_composite max_gap_multiplier must be positive")
        if self.table_name_max_distance < 0:
            raise EngineUnavailableError("figure_caption_refine_composite table_name_max_distance must be non-negative")

    def recognize(self, input_path: Path) -> EngineOutput:
        primary_output = self.primary_engine.recognize(input_path)
        caption_output = self.caption_engine.recognize(input_path)
        secondary_by_page: dict[int, list[TextLine]] = {}
        for page in caption_output.pages:
            secondary_by_page.setdefault(page.page_index, []).extend(page.text_lines)

        refined_pages: list[Page] = []
        attempted = 0
        replaced_count = 0
        for page in primary_output.pages:
            lines = list(page.text_lines)
            refined_lines = list(lines)
            index = 0
            while index + 1 < len(lines):
                marker = lines[index]
                caption = lines[index + 1]
                if not _line_looks_like_figure_marker(marker.text) or not _line_can_be_caption_after_marker(marker, caption, self.max_gap_multiplier):
                    index += 1
                    continue
                attempted += 1
                replacement = _best_caption_replacement(
                    caption,
                    secondary_by_page.get(caption.page_index, []),
                    min_iou=self.min_caption_iou,
                    min_score_delta=self.min_score_delta,
                )
                if replacement is not None:
                    metadata = dict(caption.metadata)
                    metadata["figure_caption_refine"] = {
                        "original_text": caption.text,
                        "replacement_engine": getattr(self.caption_engine, "name", type(self.caption_engine).__name__),
                        "replacement_text": replacement.text,
                    }
                    refined_lines[index + 1] = replace(
                        caption,
                        text=normalize_ocr_text(replacement.text),
                        confidence=replacement.confidence,
                        metadata=metadata,
                    )
                    replaced_count += 1
                index += 1
            if self.correct_structural_labels:
                refined_lines = [_correct_structural_label_line(line) for line in refined_lines]
            refined_pages.append(replace(page, text_lines=refined_lines))

        refined_tables, inferred_colspans = _infer_table_header_colspans(primary_output.tables, enabled=self.infer_table_header_colspans)
        refined_tables, normalized_table_columns, normalized_table_rows = _normalize_table_grids(
            refined_tables,
            enabled=self.normalize_table_grids,
        )
        refined_tables, table_name_corrections = _correct_table_name_cells(
            refined_tables,
            lexicon=self.table_name_lexicon,
            max_distance=self.table_name_max_distance,
        )
        metadata = dict(primary_output.metadata)
        metadata.update(
            {
                "engine": self.name,
                "figure_caption_refine_composite": {
                    "primary_engine": getattr(self.primary_engine, "name", type(self.primary_engine).__name__),
                    "caption_engine": getattr(self.caption_engine, "name", type(self.caption_engine).__name__),
                    "min_caption_iou": self.min_caption_iou,
                    "min_score_delta": self.min_score_delta,
                    "max_gap_multiplier": self.max_gap_multiplier,
                    "correct_structural_labels": self.correct_structural_labels,
                    "table_name_lexicon_size": len(self.table_name_lexicon),
                    "table_name_max_distance": self.table_name_max_distance,
                    "table_name_corrections": table_name_corrections,
                    "infer_table_header_colspans": self.infer_table_header_colspans,
                    "inferred_table_header_colspans": inferred_colspans,
                    "normalize_table_grids": self.normalize_table_grids,
                    "normalized_table_grid_dropped_columns": normalized_table_columns,
                    "normalized_table_grid_merged_rows": normalized_table_rows,
                    "attempted_captions": attempted,
                    "replaced_captions": replaced_count,
                },
            }
        )
        return EngineOutput(pages=refined_pages, tables=refined_tables, figures=primary_output.figures, metadata=metadata)


def _line_looks_like_figure_marker(text: str) -> bool:
    return bool(FIGURE_MARKER_LINE_RE.match(text.strip()))


def _line_can_be_caption_after_marker(marker: TextLine, candidate: TextLine, max_gap_multiplier: float) -> bool:
    if _line_looks_like_figure_marker(candidate.text):
        return False
    if not candidate.text.strip():
        return False
    if "\t" in candidate.text or "|" in candidate.text:
        return False
    vertical_gap = candidate.bbox.y - marker.bbox.bottom
    max_gap = max(marker.bbox.h, candidate.bbox.h, 24) * max_gap_multiplier
    return 0 <= vertical_gap <= max_gap


def _best_caption_replacement(
    caption: TextLine,
    candidates: list[TextLine],
    *,
    min_iou: float,
    min_score_delta: int,
) -> TextLine | None:
    original_score = _caption_prefix_score(caption.text)
    best: tuple[int, float, TextLine] | None = None
    for candidate in candidates:
        overlap = bbox_iou(caption.bbox, candidate.bbox)
        if overlap < min_iou:
            continue
        candidate_score = _caption_prefix_score(candidate.text)
        if original_score - candidate_score < min_score_delta:
            continue
        if best is None or candidate_score < best[0] or (candidate_score == best[0] and overlap > best[1]):
            best = (candidate_score, overlap, candidate)
    return best[2] if best is not None else None


def _caption_prefix_score(text: str) -> int:
    stripped = normalize_ocr_text(text).strip()
    if not stripped:
        return 99
    prefix = CAPTION_PREFIX_SPLIT_RE.split(stripped, maxsplit=1)[0]
    if not prefix:
        return 99
    lowered = prefix.lower()
    if lowered in {"caption", "figure", "fig", "fig."}:
        return 0
    return min(edit_distance(prefix, target) for target in ("चित्र", "तालिका"))


def _correct_structural_label_line(line: TextLine) -> TextLine:
    corrected = _correct_structural_label_text(line.text)
    if corrected == line.text:
        return line
    metadata = dict(line.metadata)
    metadata["structural_label_correction"] = {"original_text": line.text, "corrected_text": corrected}
    return replace(line, text=corrected, metadata=metadata)


def _correct_structural_label_text(text: str) -> str:
    normalized = normalize_ocr_text(text)
    corrected = _correct_annual_report_title(normalized)
    corrected = _correct_daily_ledger_title(corrected)
    corrected = _correct_prefixed_label(corrected, target="चित्र", max_distance=2, require_separator=True)
    corrected = _correct_prefixed_label(corrected, target="निष्कर्ष", max_distance=2, require_separator=True)
    return corrected


def _correct_annual_report_title(text: str) -> str:
    match = re.match(r"^\s*(?P<first>\S+)\s+(?P<second>\S+)(?P<rest>.*)$", text)
    if match is None:
        return text
    first = match.group("first").rstrip(":ः")
    second = match.group("second").rstrip(":ः")
    if edit_distance(first, "वार्षिक") <= 1 and edit_distance(second, "प्रतिवेदन") <= 3:
        return f"वार्षिक प्रतिवेदन{match.group('rest')}"
    return text


def _correct_daily_ledger_title(text: str) -> str:
    match = re.match(r"^\s*(?P<first>\S+)\s+(?P<second>\S+)(?P<rest>.*)$", text)
    if match is None:
        return text
    first = match.group("first").rstrip(":ः")
    second = match.group("second").rstrip(":ः")
    if edit_distance(first, "दैनिक") <= 2 and edit_distance(second, "हिसाब") <= 2:
        return f"दैनिक हिसाब{match.group('rest')}"
    return text


def _correct_prefixed_label(text: str, *, target: str, max_distance: int, require_separator: bool) -> str:
    match = re.match(r"^(?P<prefix>[^\s:ः]+)(?P<sep>[:ः]?)(?P<rest>.*)$", text)
    if match is None:
        return text
    prefix = match.group("prefix")
    sep = match.group("sep")
    rest = match.group("rest")
    if require_separator and not (sep or re.match(r"^\s*[०-९0-9IVXivx.:-]", rest)):
        return text
    if edit_distance(prefix, target) <= max_distance:
        separator = ":" if sep else ""
        return f"{target}{separator}{rest}"
    return text


def _load_table_name_lexicon(path: str | Path | None, inline_values: list[str] | None) -> list[str]:
    values: list[str] = []
    if inline_values:
        values.extend(str(item) for item in inline_values)
    if path is not None:
        lexicon_path = Path(path)
        if not lexicon_path.exists():
            raise EngineUnavailableError(f"figure_caption_refine_composite table_name_lexicon_path does not exist: {lexicon_path}")
        values.extend(lexicon_path.read_text(encoding="utf-8").splitlines())
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_ocr_text(value).strip()
        if not item or item.startswith("#") or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _infer_table_header_colspans(tables: list[Table], *, enabled: bool) -> tuple[list[Table], int]:
    if not enabled:
        return tables, 0
    from .tables import infer_repeated_header_colspans

    inferred_tables: list[Table] = []
    total_inferred = 0
    for table in tables:
        inferred_table, count = infer_repeated_header_colspans(table)
        inferred_tables.append(inferred_table)
        total_inferred += count
    return inferred_tables, total_inferred


def _normalize_table_grids(tables: list[Table], *, enabled: bool) -> tuple[list[Table], int, int]:
    if not enabled:
        return tables, 0, 0
    from .tables import normalize_table_grid_structure

    normalized_tables: list[Table] = []
    total_dropped_columns = 0
    total_merged_rows = 0
    for table in tables:
        normalized_table, _changes = normalize_table_grid_structure(table)
        normalized_tables.append(normalized_table)
        total_dropped_columns += int(normalized_table.metadata.get("normalized_table_grid_dropped_columns") or 0)
        total_merged_rows += int(normalized_table.metadata.get("normalized_table_grid_merged_rows") or 0)
    return normalized_tables, total_dropped_columns, total_merged_rows


def _correct_table_name_cells(tables: list[Table], *, lexicon: list[str], max_distance: int) -> tuple[list[Table], int]:
    if not lexicon:
        return tables, 0
    corrected_tables: list[Table] = []
    total_corrections = 0
    for table in tables:
        name_headers = _table_name_headers(table)
        if not name_headers:
            corrected_tables.append(table)
            continue
        corrected_cells: list[TableCell] = []
        table_corrections = 0
        for cell in table.cells:
            header_row = name_headers.get(cell.col)
            if header_row is None or cell.row <= header_row:
                corrected_cells.append(cell)
                continue
            replacement_text = _nearest_unique_lexicon_match(cell.text, lexicon=lexicon, max_distance=max_distance)
            if replacement_text is None:
                corrected_cells.append(cell)
                continue
            corrected_cells.append(_replace_cell(cell, text=replacement_text))
            table_corrections += 1
        if table_corrections:
            from .tables import cells_to_html

            metadata = dict(table.metadata)
            metadata["table_name_lexicon_corrections"] = table_corrections
            corrected_table = replace(
                table,
                cells=corrected_cells,
                html=cells_to_html(corrected_cells),
                metadata=metadata,
            )
            corrected_tables.append(corrected_table)
            total_corrections += table_corrections
        else:
            corrected_tables.append(table)
    return corrected_tables, total_corrections


def _table_name_headers(table: Table) -> dict[int, int]:
    headers: dict[int, int] = {}
    for cell in table.cells:
        header = normalize_ocr_text(cell.text).strip().rstrip(":ः")
        if edit_distance(header, "नाम") <= 1 and (cell.col not in headers or cell.row < headers[cell.col]):
            headers[cell.col] = cell.row
    return headers


def _nearest_unique_lexicon_match(text: str, *, lexicon: list[str], max_distance: int) -> str | None:
    normalized = normalize_ocr_text(text).strip()
    if not normalized:
        return None
    if normalized in lexicon:
        return None
    suffix = ""
    match = re.match(r"^(?P<base>[^-\s]+)(?P<suffix>-[A-Za-z0-9०-९]+)$", normalized)
    if match is not None:
        normalized = match.group("base")
        suffix = match.group("suffix")
        if normalized in lexicon:
            return None
    scores = sorted((edit_distance(normalized, item), item) for item in lexicon)
    if not scores or scores[0][0] > max_distance:
        return None
    if len(scores) > 1 and scores[1][0] == scores[0][0]:
        return None
    return f"{scores[0][1]}{suffix}"
