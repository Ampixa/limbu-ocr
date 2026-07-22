#!/usr/bin/env python3
"""Score a PaddleOCR text recognizer against a PaddleOCR labels.txt pack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_metrics import evaluate  # noqa: E402


EXPORTED_MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
DEFAULT_MODEL_NAME = "PP-OCRv5_mobile_rec"
DEFAULT_CACHE_ROOT = Path("/tmp/ocrtech-paddle-cache")


@dataclass(frozen=True)
class LabelRow:
    image: str
    image_path: Path
    gt: str


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float | None


@dataclass(frozen=True)
class PredictionRow:
    image: str
    gt: str
    pred: str
    confidence: float | None


class TextRecognitionRunner:
    def __init__(self, model_dir: Path, *, model_name: str, device: str) -> None:
        try:
            from paddleocr._models.text_recognition import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR TextRecognition is unavailable. Use the documented "
                "Python 3.11 PaddleOCR environment."
            ) from exc
        self._model = TextRecognition(
            model_name=model_name,
            model_dir=str(model_dir),
            device=device,
        )

    def recognize(self, image_path: Path) -> RecognitionResult:
        raw_result = self._model.predict(str(image_path))
        best_text = ""
        best_confidence: float | None = None
        for item in raw_result if isinstance(raw_result, list) else [raw_result]:
            payload = _result_mapping(item)
            if payload is None:
                continue
            text = str(payload.get("rec_text") or payload.get("text") or payload.get("transcription") or "")
            score = payload.get("rec_score") or payload.get("score") or payload.get("confidence")
            confidence = _float_or_none(score)
            if best_confidence is None or (confidence is not None and confidence > best_confidence):
                best_text = text
                best_confidence = confidence
        return RecognitionResult(text=best_text, confidence=best_confidence)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_mapping(item: object) -> dict[str, object] | None:
    payload: object = item
    if not isinstance(payload, dict):
        payload = getattr(item, "json", None)
    if not isinstance(payload, dict):
        return None
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _read_labels(path: Path) -> list[LabelRow]:
    if not path.exists():
        raise FileNotFoundError(f"label file does not exist: {path}")
    rows: list[LabelRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{line_no}: expected image path and label separated by a tab")
            image, gt = line.split("\t", 1)
            if not image:
                raise ValueError(f"{path}:{line_no}: empty image path")
            if image in seen:
                raise ValueError(f"{path}:{line_no}: duplicate image path in labels: {image}")
            seen.add(image)
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"{path}:{line_no}: label references missing image: {image_path}")
            rows.append(LabelRow(image=image, image_path=image_path, gt=gt))
    if not rows:
        raise ValueError(f"label file contains no rows: {path}")
    return rows


def _is_exported_model_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in EXPORTED_MODEL_FILES)


def _is_checkpoint_archive(path: Path) -> bool:
    return path.is_file() and path.name.endswith((".tgz", ".tar.gz", ".tar"))


def _safe_extract(tar_path: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            member_path = (dest / member.name).resolve()
            if dest_resolved != member_path and dest_resolved not in member_path.parents:
                raise ValueError(f"archive member escapes extraction directory: {member.name}")
        archive.extractall(dest)


def _single_file(root: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} under {root}, found {len(matches)}")
    return matches[0]


def _resolve_paddleocr_source(path: Path | None) -> Path:
    source = path or Path(os.environ.get("PADDLEOCR_SOURCE", ""))
    if not str(source):
        raise ValueError("--paddleocr-source or PADDLEOCR_SOURCE is required when --model is a checkpoint tgz")
    export_script = source / "tools" / "export_model.py"
    if not export_script.exists():
        raise FileNotFoundError(f"PaddleOCR export script not found: {export_script}")
    return source


def _ensure_paddle_cache_env() -> None:
    env = _paddle_env(os.environ)
    os.environ.update(env)


def _paddle_env(base: os._Environ[str] | dict[str, str]) -> dict[str, str]:
    env = dict(base)
    cache_root = Path(env.get("OCRTECH_PADDLE_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)))
    home = Path(env.get("HOME", ""))
    if not _directory_is_writable(home / ".cache"):
        fallback_home = cache_root / "home"
        fallback_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(fallback_home)
    for key, suffix in (("XDG_CACHE_HOME", "xdg"), ("PADDLE_HOME", "paddle-home")):
        if not env.get(key):
            target = cache_root / suffix
            target.mkdir(parents=True, exist_ok=True)
            env[key] = str(target)
    return env


def _directory_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ocrtech-write-probe"
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("")
        probe.unlink()
        return True
    except OSError:
        return False


def _export_checkpoint_archive(
    archive_path: Path,
    *,
    dictionary: Path,
    out_dir: Path,
    paddleocr_source: Path,
) -> Path:
    export_dir = out_dir / "exported_model"
    if _is_exported_model_dir(export_dir):
        return export_dir
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError(f"export directory exists but is incomplete: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ocrtech-recognizer-", dir=str(out_dir)) as tmp:
        extracted = Path(tmp) / "artifact"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        config = _single_file(extracted, "**/paddle_recognizer.yml", label="paddle_recognizer.yml")
        checkpoint = _single_file(extracted, "**/checkpoints/best_accuracy.pdparams", label="best_accuracy.pdparams")
        cmd = [
            sys.executable,
            "tools/export_model.py",
            "-c",
            str(config),
            "-o",
            "Global.use_gpu=False",
            f"Global.character_dict_path={dictionary}",
            f"Global.pretrained_model={checkpoint}",
            f"Global.save_inference_dir={export_dir}",
        ]
        subprocess.run(
            cmd,
            cwd=paddleocr_source,
            env=_paddle_env(os.environ),
            check=True,
        )
    if not _is_exported_model_dir(export_dir):
        raise RuntimeError(f"export did not produce expected inference files in {export_dir}")
    return export_dir


def _resolve_model_dir(args: argparse.Namespace) -> tuple[Path, str]:
    model = args.model
    if _is_exported_model_dir(model):
        return model, "exported_dir"
    if _is_checkpoint_archive(model):
        source = _resolve_paddleocr_source(args.paddleocr_source)
        return (
            _export_checkpoint_archive(
                model,
                dictionary=args.dictionary,
                out_dir=args.out,
                paddleocr_source=source,
            ),
            "exported_from_checkpoint_tgz",
        )
    raise ValueError(
        f"--model must be an exported PaddleOCR dir containing {EXPORTED_MODEL_FILES} "
        f"or a checkpoint archive: {model}"
    )


def _build_recognizer(model_dir: Path, *, model_name: str, device: str) -> TextRecognitionRunner:
    return TextRecognitionRunner(model_dir, model_name=model_name, device=device)


def _score_rows(rows: list[LabelRow], recognizer: Any, predictions_path: Path) -> tuple[list[PredictionRow], dict[str, Any]]:
    predictions: list[PredictionRow] = []
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = recognizer.recognize(row.image_path)
            prediction = PredictionRow(
                image=row.image,
                gt=row.gt,
                pred=result.text,
                confidence=result.confidence,
            )
            predictions.append(prediction)
            handle.write(json.dumps(prediction.__dict__, ensure_ascii=False) + "\n")
    metrics = evaluate([(row.pred, row.gt) for row in predictions])
    return predictions, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_paddle_cache_env()
    args.model = args.model.expanduser().resolve(strict=False)
    args.labels = args.labels.expanduser().resolve(strict=False)
    args.dictionary = args.dictionary.expanduser().resolve(strict=False)
    args.out = args.out.expanduser().resolve(strict=False)
    if args.paddleocr_source is not None:
        args.paddleocr_source = args.paddleocr_source.expanduser().resolve(strict=False)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.dictionary.exists():
        raise FileNotFoundError(f"dictionary file does not exist: {args.dictionary}")
    rows = _read_labels(args.labels)
    model_dir, route = _resolve_model_dir(args)
    recognizer = _build_recognizer(model_dir, model_name=args.model_name, device=args.device)
    predictions_path = args.out / "predictions.jsonl"
    _, metrics = _score_rows(rows, recognizer, predictions_path)
    summary = {
        "route": route,
        "model": str(args.model),
        "model_dir": str(model_dir),
        "labels": str(args.labels),
        "dictionary": str(args.dictionary),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint .tgz/.tar.gz or exported PaddleOCR model dir")
    parser.add_argument("--labels", type=Path, required=True, help="PaddleOCR labels.txt file")
    parser.add_argument("--dictionary", type=Path, required=True, help="Recognizer character dictionary used for checkpoint export")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for predictions.jsonl and metrics.json")
    parser.add_argument("--paddleocr-source", type=Path, help="PaddleOCR source checkout; required when --model is a checkpoint archive")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    metrics = summary["metrics"]
    print(
        "scored "
        f"{metrics['n_lines']} lines "
        f"codepoint_CER={metrics['cer_codepoint']:.6f} "
        f"grapheme_CER={metrics['cer_grapheme']:.6f} "
        f"exact_line_acc={metrics['exact_line_acc']:.6f}"
    )
    print(f"wrote {summary['predictions']}")
    print(f"wrote {Path(summary['predictions']).parent / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
