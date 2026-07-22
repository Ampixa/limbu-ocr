"""Training recipe generation for recognizer and corrector models."""

from __future__ import annotations

import json
import math
import random
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .admission import assert_recognizer_family_not_stopped
from .datasets import audit_recognizer_corpus
from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest, prepare_hf_correction_pairs, sha256_file, sha256_text, write_manifest
from .metrics import cer, exact_line_accuracy, wer
from .models import audit_model_card, write_model_card
from .normalization import dictionary_from_texts, normalize_ocr_text


PADDLE_DEVANAGARI_PPOCRV3_PRETRAINED = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams"
)
# Current-generation default. The Devanagari PP-OCRv3 mobile family above is a confirmed
# dead end (recognizer-failure-review -> stop_family) and is wrong for non-Devanagari
# scripts. PP-OCRv5 mobile rec (SVTR_LCNet / PPLCNetV3 + CTC/NRTR) is the documented
# fine-tune base; its SVTR backbone transfers across scripts while the head re-inits for a
# changed character dictionary.
PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "PP-OCRv5_mobile_rec_pretrained.pdparams"
)
PADDLE_RECOGNIZER_MAIN_INDICATORS = frozenset({"acc", "norm_edit_dis"})


def write_recognizer_recipe(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED,
    dictionary_path: str | Path | None = None,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: float | None = None,
    warmup_epoch: float = 5.0,
    use_gpu: bool = True,
    train_num_workers: int = 2,
    eval_num_workers: int = 2,
    train_drop_last: bool = True,
    main_indicator: str = "acc",
    eval_batch_step: int = 200,
    failure_review: str | Path | None = None,
    run: bool = False,
) -> Path:
    return write_paddle_recognizer_recipe(
        train_manifest,
        eval_manifest,
        output_dir,
        base_model=base_model,
        dictionary_path=dictionary_path,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_epoch=warmup_epoch,
        use_gpu=use_gpu,
        train_num_workers=train_num_workers,
        eval_num_workers=eval_num_workers,
        train_drop_last=train_drop_last,
        main_indicator=main_indicator,
        eval_batch_step=eval_batch_step,
        failure_review=failure_review,
        run=run,
    )


def write_paddle_recognizer_recipe(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED,
    dictionary_path: str | Path | None = None,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: float | None = None,
    warmup_epoch: float = 5.0,
    use_gpu: bool = True,
    train_num_workers: int = 2,
    eval_num_workers: int = 2,
    train_drop_last: bool = True,
    main_indicator: str = "acc",
    eval_batch_step: int = 200,
    failure_review: str | Path | None = None,
    run: bool = False,
) -> Path:
    if train_batch_size <= 0:
        raise DataValidationError("train_batch_size must be positive")
    if eval_batch_size <= 0:
        raise DataValidationError("eval_batch_size must be positive")
    if learning_rate <= 0:
        raise DataValidationError("learning_rate must be positive")
    if epochs is not None and epochs <= 0:
        raise DataValidationError("epochs must be positive when provided")
    if warmup_epoch < 0:
        raise DataValidationError("warmup_epoch must be non-negative")
    if train_num_workers < 0 or eval_num_workers < 0:
        raise DataValidationError("num_workers must be non-negative")
    if eval_batch_step <= 0:
        raise DataValidationError("eval_batch_step must be positive")
    if main_indicator not in PADDLE_RECOGNIZER_MAIN_INDICATORS:
        allowed = ", ".join(sorted(PADDLE_RECOGNIZER_MAIN_INDICATORS))
        raise DataValidationError(f"main_indicator must be one of: {allowed}")
    if failure_review is not None:
        assert_recognizer_family_not_stopped(failure_review, backend="paddleocr", base_model=base_model)
    train_entries = load_manifest(train_manifest)
    eval_entries = load_manifest(eval_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels_dir = out / "labels"
    labels_dir.mkdir(exist_ok=True)
    train_label = labels_dir / "train.txt"
    eval_label = labels_dir / "eval.txt"
    _write_paddle_label_file(train_entries, train_label)
    _write_paddle_label_file(eval_entries, eval_label)

    dict_path = Path(dictionary_path) if dictionary_path else out / "nep_eng_dict.txt"
    if dictionary_path is None:
        chars = dictionary_from_texts([entry.text for entry in [*train_entries, *eval_entries]])
        dict_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    elif not dict_path.exists():
        raise DataValidationError(f"dictionary_path does not exist: {dict_path}")
    dictionary_chars = [line for line in dict_path.read_text(encoding="utf-8").splitlines() if line]
    _validate_paddle_dictionary_coverage([*train_entries, *eval_entries], dictionary_chars, dict_path)

    config_path = out / "paddle_recognizer.yml"
    config_path.write_text(
        _recognizer_config_text(
            train_label=train_label,
            eval_label=eval_label,
            dictionary_path=dict_path,
            base_model=base_model,
            output_dir=out / "checkpoints",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            warmup_epoch=warmup_epoch,
            use_gpu=use_gpu,
            train_num_workers=train_num_workers,
            eval_num_workers=eval_num_workers,
            train_drop_last=train_drop_last,
            main_indicator=main_indicator,
            eval_batch_step=eval_batch_step,
        ),
        encoding="utf-8",
    )
    summary_path = out / "recognizer-training-summary.json"
    summary_path.write_text(
        json.dumps(
            _recognizer_training_summary(
                train_manifest=Path(train_manifest),
                eval_manifest=Path(eval_manifest),
                train_entries=train_entries,
                eval_entries=eval_entries,
                train_label=train_label,
                eval_label=eval_label,
                dictionary_path=dict_path,
                dictionary_chars=dictionary_chars,
                config_path=config_path,
                base_model=base_model,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                warmup_epoch=warmup_epoch,
                use_gpu=use_gpu,
                train_num_workers=train_num_workers,
                eval_num_workers=eval_num_workers,
                train_drop_last=train_drop_last,
                main_indicator=main_indicator,
                eval_batch_step=eval_batch_step,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Recognizer Training Recipe",
                "",
                f"Training summary: `{summary_path}`",
                f"Base model: `{base_model}`",
                f"Train labels: `{train_label}`",
                f"Eval labels: `{eval_label}`",
                f"Dictionary: `{dict_path}`",
                f"Train batch size: `{train_batch_size}`",
                f"Eval batch size: `{eval_batch_size}`",
                f"Learning rate: `{learning_rate}`",
                f"Epochs: `{epochs if epochs is not None else 'PaddleOCR default'}`",
                f"Warmup epochs: `{warmup_epoch}`",
                f"Use GPU: `{use_gpu}`",
                f"Train num workers: `{train_num_workers}`",
                f"Eval num workers: `{eval_num_workers}`",
                f"Train drop last: `{train_drop_last}`",
                f"Main indicator: `{main_indicator}`",
                f"Eval batch step: `{eval_batch_step}`",
                "",
                "Run with a checked-out PaddleOCR repo:",
                "",
                "```bash",
                f"python PaddleOCR/tools/train.py -c {config_path}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if run:
        completed = subprocess.run(["python", "PaddleOCR/tools/train.py", "-c", str(config_path)], check=False)
        if completed.returncode != 0:
            raise DataValidationError(f"PaddleOCR training failed with exit code {completed.returncode}")
    return config_path


def _recognizer_training_summary(
    *,
    train_manifest: Path,
    eval_manifest: Path,
    train_entries: list[ManifestEntry],
    eval_entries: list[ManifestEntry],
    train_label: Path,
    eval_label: Path,
    dictionary_path: Path,
    dictionary_chars: list[str],
    config_path: Path,
    base_model: str,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    epochs: float | None,
    warmup_epoch: float,
    use_gpu: bool,
    train_num_workers: int,
    eval_num_workers: int,
    train_drop_last: bool,
    main_indicator: str,
    eval_batch_step: int,
) -> dict[str, Any]:
    pack_content_sha256 = _training_pack_content_sha256(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        train_entries=train_entries,
        eval_entries=eval_entries,
    )
    return {
        "backend": "paddleocr",
        "base_model": base_model,
        "pack_content_sha256": pack_content_sha256,
        "training_pack": {
            "schema": "ocrtech.recognizer-training-pack.v1",
            "pack_content_sha256": pack_content_sha256,
        },
        "manifests": {
            "train": _manifest_summary(train_manifest, train_entries),
            "eval": _manifest_summary(eval_manifest, eval_entries),
        },
        "artifacts": {
            "config": _file_summary(config_path),
            "train_label": _file_summary(train_label),
            "eval_label": _file_summary(eval_label),
            "dictionary": {
                **_file_summary(dictionary_path),
                "character_count": len(dictionary_chars),
                "contains_space": " " in dictionary_chars,
                "contains_ascii": any(char.isascii() and char.isalnum() for char in dictionary_chars),
                "contains_devanagari": any("\u0900" <= char <= "\u097f" for char in dictionary_chars),
            },
        },
        "hyperparameters": {
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "warmup_epoch": warmup_epoch,
            "use_gpu": use_gpu,
            "train_num_workers": train_num_workers,
            "eval_num_workers": eval_num_workers,
            "train_drop_last": train_drop_last,
            "main_indicator": main_indicator,
            "eval_batch_step": eval_batch_step,
        },
    }


def _manifest_summary(path: Path, entries: list[ManifestEntry]) -> dict[str, Any]:
    texts = [entry.text for entry in entries]
    chars = dictionary_from_texts(texts)
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else None,
        "sample_count": len(entries),
        "dataset_counts": _count_values(entry.dataset for entry in entries),
        "slice_counts": _slice_counts(entries),
        "character_count": len(chars),
        "contains_ascii": any(char.isascii() and char.isalnum() for char in chars),
        "contains_devanagari": any("\u0900" <= char <= "\u097f" for char in chars),
    }


def _file_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def _slice_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for slice_name in _entry_slices(entry):
            counts[slice_name] = counts.get(slice_name, 0) + 1
    return dict(sorted(counts.items()))


def _entry_slices(entry: ManifestEntry) -> list[str]:
    metadata = entry.metadata or {}
    values: set[str] = set()
    for key in ("slice", "script", "document_type", "language"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    for key in ("slices", "scripts", "tags"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.update(str(item) for item in value if str(item))
        elif isinstance(value, str) and value:
            values.add(value)
    return sorted(values)


def _count_values(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_paddle_label_file(entries: list[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            if "\t" in entry.image_path or "\n" in entry.image_path or "\r" in entry.image_path:
                raise DataValidationError(f"image_path contains a tab or newline and cannot be serialized for PaddleOCR: {entry.sample_id}")
            handle.write(f"{entry.image_path}\t{_paddle_label_text(entry.text)}\n")


def _validate_paddle_dictionary_coverage(
    entries: list[ManifestEntry],
    dictionary_chars: list[str],
    dictionary_path: Path,
) -> None:
    allowed = set(dictionary_chars)
    allowed.add(" ")
    missing: dict[str, list[str]] = {}
    for entry in entries:
        label = _paddle_label_text(entry.text)
        missing_chars = sorted({char for char in label if char not in allowed})
        if missing_chars:
            missing[entry.sample_id] = missing_chars
    if missing:
        details = [
            f"{sample_id}: {''.join(chars)}"
            for sample_id, chars in list(missing.items())[:10]
        ]
        suffix = "; ".join(details)
        raise DataValidationError(
            f"dictionary_path is missing label characters for {len(missing)} samples: {dictionary_path}; {suffix}"
        )


def _paddle_label_text(text: str) -> str:
    label = normalize_ocr_text(text).replace("\t", " ")
    return " ".join(label.split())


def _yaml_float(value: float) -> str:
    if not math.isfinite(value):
        raise DataValidationError(f"PaddleOCR YAML float must be finite: {value!r}")
    if value == 0:
        return "0.0"
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _recognizer_config_text(
    *,
    train_label: Path,
    eval_label: Path,
    dictionary_path: Path,
    base_model: str,
    output_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    epochs: float | None,
    warmup_epoch: float,
    use_gpu: bool,
    train_num_workers: int,
    eval_num_workers: int,
    train_drop_last: bool,
    main_indicator: str,
    eval_batch_step: int,
) -> str:
    epoch_value: int | float | None
    if epochs is None:
        epoch_value = None
    elif float(epochs).is_integer():
        epoch_value = int(epochs)
    else:
        epoch_value = epochs
    epoch_line = f"  epoch_num: {epoch_value}\n" if epoch_value is not None else ""
    learning_rate_value = _yaml_float(float(learning_rate))
    warmup_epoch_value = _yaml_float(float(warmup_epoch))
    use_gpu_value = "true" if use_gpu else "false"
    train_drop_last_value = "true" if train_drop_last else "false"
    return f"""# Generated by ocrtech train-recognizer.
# PP-OCRv5 mobile recognizer recipe for PaddleOCR's tools/train.py.
# Keep in sync with PaddleOCR configs/rec/PP-OCRv5/multi_language/devanagari_PP-OCRv5_mobile_rec.yaml.
# Fine-tune from a PP-OCRv5 mobile rec checkpoint: the SVTR backbone transfers across
# scripts; the recognition head re-inits for a changed character dictionary (new script).
Global:
  model_name: PP-OCRv5_mobile_rec
  debug: false
  use_gpu: {use_gpu_value}
{epoch_line}  log_smooth_window: 20
  print_batch_step: 10
  pretrained_model: {base_model}
  character_dict_path: {dictionary_path}
  save_model_dir: {output_dir}
  save_epoch_step: 1
  eval_batch_step:
    - 0
    - {eval_batch_step}
  cal_metric_during_train: true
  use_space_char: true
  max_text_length: &max_text_length 128
  infer_mode: false
  distributed: false
  save_res_path: {output_dir.parent / "predicts.txt"}

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: {learning_rate_value}
    warmup_epoch: {warmup_epoch_value}
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size:
              - 1
              - 3
            use_guide: true
          Head:
            fc_decay: 1.0e-05
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: {main_indicator}
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: .
    ext_op_transform_idx: 1
    label_file_list:
      - {train_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape:
            - 48
            - 320
            - 3
          max_text_length: *max_text_length
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape:
            - 3
            - 48
            - 320
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: {train_batch_size}
    drop_last: {train_drop_last_value}
    num_workers: {train_num_workers}

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: .
    label_file_list:
      - {eval_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape:
            - 3
            - 48
            - 320
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: false
    batch_size_per_card: {eval_batch_size}
    drop_last: false
    num_workers: {eval_num_workers}
"""


@dataclass(slots=True)
class HfRecognizerTrainingConfig:
    train_manifest: str
    eval_manifest: str
    output_dir: str
    base_model: str
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    warmup_steps: int
    logging_steps: int
    save_total_limit: int
    predict_with_generate: bool
    generation_max_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "output_dir": self.output_dir,
            "base_model": self.base_model,
            "max_target_length": self.max_target_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "predict_with_generate": self.predict_with_generate,
            "generation_max_length": self.generation_max_length,
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "HfRecognizerTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise DataValidationError("hf recognizer config must be a JSON object")
        return cls(
            train_manifest=str(payload.get("train_manifest") or ""),
            eval_manifest=str(payload.get("eval_manifest") or ""),
            output_dir=str(payload.get("output_dir") or ""),
            base_model=str(payload.get("base_model") or ""),
            max_target_length=int(payload.get("max_target_length", 128)),
            per_device_train_batch_size=int(payload.get("per_device_train_batch_size", 8)),
            per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size", 8)),
            learning_rate=float(payload.get("learning_rate", 5e-5)),
            num_train_epochs=float(payload.get("num_train_epochs", 3.0)),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            logging_steps=int(payload.get("logging_steps", 10)),
            save_total_limit=int(payload.get("save_total_limit", 2)),
            predict_with_generate=bool(payload.get("predict_with_generate", True)),
            generation_max_length=int(payload.get("generation_max_length", payload.get("max_target_length", 128))),
        )


@dataclass(slots=True)
class HfTextCorrectorTrainingConfig:
    train_pairs: str
    eval_pairs: str
    output_dir: str
    base_model: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    warmup_steps: int
    logging_steps: int
    save_total_limit: int
    predict_with_generate: bool
    generation_max_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_pairs": self.train_pairs,
            "eval_pairs": self.eval_pairs,
            "output_dir": self.output_dir,
            "base_model": self.base_model,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "predict_with_generate": self.predict_with_generate,
            "generation_max_length": self.generation_max_length,
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "HfTextCorrectorTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise DataValidationError("hf text corrector config must be a JSON object")
        return cls(
            train_pairs=str(payload.get("train_pairs") or ""),
            eval_pairs=str(payload.get("eval_pairs") or ""),
            output_dir=str(payload.get("output_dir") or ""),
            base_model=str(payload.get("base_model") or ""),
            max_source_length=int(payload.get("max_source_length", 256)),
            max_target_length=int(payload.get("max_target_length", 256)),
            per_device_train_batch_size=int(payload.get("per_device_train_batch_size", 16)),
            per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size", 16)),
            learning_rate=float(payload.get("learning_rate", 5e-5)),
            num_train_epochs=float(payload.get("num_train_epochs", 3.0)),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            logging_steps=int(payload.get("logging_steps", 10)),
            save_total_limit=int(payload.get("save_total_limit", 2)),
            predict_with_generate=bool(payload.get("predict_with_generate", True)),
            generation_max_length=int(payload.get("generation_max_length", payload.get("max_target_length", 256))),
        )


@dataclass(slots=True)
class RecognizerTrainingBundle:
    archive_path: str
    manifest_path: str
    readme_path: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "manifest_path": self.manifest_path,
            "readme_path": self.readme_path,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(slots=True)
class HfRecognizerFinalization:
    run_dir: str
    package_dir: str
    model_card_path: str
    audit_path: str
    report_json_path: str
    report_md_path: str
    audit_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "package_dir": self.package_dir,
            "model_card_path": self.model_card_path,
            "audit_path": self.audit_path,
            "report_json_path": self.report_json_path,
            "report_md_path": self.report_md_path,
            "audit_passed": self.audit_passed,
        }


def write_hf_recognizer_recipe(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "microsoft/trocr-small-printed",
    max_target_length: int = 128,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 3.0,
    warmup_steps: int = 0,
    logging_steps: int = 10,
    save_total_limit: int = 2,
    run: bool = False,
) -> Path:
    train_entries = load_manifest(train_manifest)
    eval_entries = load_manifest(eval_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = HfRecognizerTrainingConfig(
        train_manifest=str(Path(train_manifest)),
        eval_manifest=str(Path(eval_manifest)),
        output_dir=str(out),
        base_model=base_model,
        max_target_length=max_target_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        predict_with_generate=True,
        generation_max_length=max_target_length,
    )
    config_path = out / "hf_recognizer_config.json"
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = out / "recognizer-training-summary.json"
    summary_path.write_text(
        json.dumps(
            _hf_recognizer_training_summary(
                train_manifest=Path(train_manifest),
                eval_manifest=Path(eval_manifest),
                train_entries=train_entries,
                eval_entries=eval_entries,
                config_path=config_path,
                base_model=base_model,
                max_target_length=max_target_length,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_total_limit=save_total_limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Hugging Face Recognizer Training Recipe",
                "",
                f"Training summary: `{summary_path}`",
                f"Base model: `{base_model}`",
                f"Train manifest: `{train_manifest}` ({len(train_entries)} samples)",
                f"Eval manifest: `{eval_manifest}` ({len(eval_entries)} samples)",
                "",
                "Install the backend dependencies first:",
                "",
                "```bash",
                "python -m pip install -e '.[hf-recognizer]'",
                "```",
                "",
                "Run in-process:",
                "",
                "```bash",
                _hf_recognizer_train_command(config, run=True),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if run:
        run_hf_recognizer_training(config_path)
    return config_path


def _hf_recognizer_train_command(config: HfRecognizerTrainingConfig, *, run: bool = False) -> str:
    parts = [
        "ocrtech",
        "train-recognizer",
        "--backend",
        "hf-vision-encoder-decoder",
        "--base-model",
        config.base_model,
        "--train-manifest",
        config.train_manifest,
        "--eval-manifest",
        config.eval_manifest,
        "--out",
        config.output_dir,
        "--max-target-length",
        str(config.max_target_length),
        "--train-batch-size",
        str(config.per_device_train_batch_size),
        "--eval-batch-size",
        str(config.per_device_eval_batch_size),
        "--learning-rate",
        _format_float(config.learning_rate),
        "--epochs",
        _format_float(config.num_train_epochs),
    ]
    if run:
        parts.append("--run")
    return " ".join(shlex.quote(part) for part in parts)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _hf_recognizer_training_summary(
    *,
    train_manifest: Path,
    eval_manifest: Path,
    train_entries: list[ManifestEntry],
    eval_entries: list[ManifestEntry],
    config_path: Path,
    base_model: str,
    max_target_length: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    num_train_epochs: float,
    warmup_steps: int,
    logging_steps: int,
    save_total_limit: int,
) -> dict[str, Any]:
    texts = [entry.text for entry in [*train_entries, *eval_entries]]
    chars = dictionary_from_texts(texts)
    pack_content_sha256 = _training_pack_content_sha256(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        train_entries=train_entries,
        eval_entries=eval_entries,
    )
    return {
        "backend": "hf_vision_encoder_decoder",
        "base_model": base_model,
        "pack_content_sha256": pack_content_sha256,
        "training_pack": {
            "schema": "ocrtech.recognizer-training-pack.v1",
            "pack_content_sha256": pack_content_sha256,
        },
        "manifests": {
            "train": _manifest_summary(train_manifest, train_entries),
            "eval": _manifest_summary(eval_manifest, eval_entries),
        },
        "artifacts": {
            "config": _file_summary(config_path),
            "dictionary": {
                "character_count": len(chars),
                "contains_space": " " in chars,
                "contains_ascii": any(char.isascii() and char.isalnum() for char in chars),
                "contains_devanagari": any("\u0900" <= char <= "\u097f" for char in chars),
            },
        },
        "hyperparameters": {
            "max_target_length": max_target_length,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_total_limit": save_total_limit,
            "generation_max_length": max_target_length,
        },
    }


def _training_pack_content_sha256(
    *,
    train_manifest: Path,
    eval_manifest: Path,
    train_entries: list[ManifestEntry],
    eval_entries: list[ManifestEntry],
) -> str:
    payload = {
        "schema": "ocrtech.recognizer-training-pack.v1",
        "manifests": {
            "train": {
                "path": str(train_manifest),
                "sha256": sha256_file(train_manifest) if train_manifest.exists() else None,
                "sample_count": len(train_entries),
            },
            "eval": {
                "path": str(eval_manifest),
                "sha256": sha256_file(eval_manifest) if eval_manifest.exists() else None,
                "sample_count": len(eval_entries),
            },
        },
        "rows": {
            "train": [_training_pack_row(index, entry) for index, entry in enumerate(train_entries)],
            "eval": [_training_pack_row(index, entry) for index, entry in enumerate(eval_entries)],
        },
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _training_pack_row(index: int, entry: ManifestEntry) -> dict[str, Any]:
    return {
        "index": index,
        "sample_id": entry.sample_id,
        "dataset": entry.dataset,
        "split": entry.split,
        "image_path": entry.image_path,
        "text": _paddle_label_text(entry.text),
        "sha256": entry.sha256,
        "metadata": entry.metadata,
    }


def run_hf_recognizer_training(config_path: str | Path) -> Path:
    config = HfRecognizerTrainingConfig.from_path(config_path)
    train_entries = load_manifest(config.train_manifest)
    eval_entries = load_manifest(config.eval_manifest)
    if not train_entries:
        raise DataValidationError("hf recognizer training requires non-empty train_manifest")
    if not eval_entries:
        raise DataValidationError("hf recognizer training requires non-empty eval_manifest")
    try:
        import torch
    except ImportError as exc:
        raise DataValidationError("hf recognizer training requires torch. Install ocr-tech[hf-recognizer].") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("hf recognizer training requires Pillow. Install ocr-tech[hf-recognizer].") from exc
    try:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise DataValidationError("hf recognizer training requires transformers. Install ocr-tech[hf-recognizer].") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = _load_trocr_processor(TrOCRProcessor, config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)
    tokenizer = processor.tokenizer
    decoder_start_token_id = _read_token_id(model.config, "decoder_start_token_id")
    pad_token_id = _read_token_id(model.config, "pad_token_id")
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if decoder_start_token_id is None or pad_token_id is None:
        raise DataValidationError("hf recognizer base model tokenizer must define decoder_start_token_id and pad_token_id")
    _write_token_ids(model, decoder_start_token_id=decoder_start_token_id, pad_token_id=pad_token_id)
    model.generation_config.max_length = config.generation_max_length

    train_dataset = _VisionTextDataset(train_entries, processor, Image, config.max_target_length)
    eval_dataset = _VisionTextDataset(eval_entries, processor, Image, config.max_target_length)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=config.predict_with_generate,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=_VisionTextCollator(processor, torch),
        compute_metrics=_hf_compute_metrics_factory(processor),
    )
    trainer.train()
    final_model_dir = output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    metrics = trainer.evaluate(max_length=config.generation_max_length)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_model_dir


def bundle_recognizer_training(
    recipe_dir: str | Path,
    output_dir: str | Path,
    *,
    train_manifest: str | Path,
    eval_manifest: str | Path,
    archive_name: str = "recognizer-training-bundle.tar.gz",
    base_dir: str | Path = ".",
    require_corpus_audit: bool = False,
    min_train_samples: int = 1000,
    min_eval_samples: int = 100,
    min_train_english: int = 200,
    min_train_devanagari: int = 200,
    min_train_mixed: int = 25,
    min_eval_english: int = 50,
    min_eval_devanagari: int = 50,
    min_eval_mixed: int = 10,
    min_train_latin_only: int = 50,
    min_train_devanagari_only: int = 50,
    min_eval_latin_only: int = 10,
    min_eval_devanagari_only: int = 10,
    min_train_real: int = 100,
    min_train_synthetic: int = 100,
    require_eval_real: bool = False,
    failure_review: str | Path | None = None,
) -> RecognizerTrainingBundle:
    base = Path(base_dir).resolve()
    recipe_path = _resolve_under_base(recipe_dir, base=base)
    train_path = _resolve_under_base(train_manifest, base=base)
    eval_path = _resolve_under_base(eval_manifest, base=base)
    if not recipe_path.is_dir():
        raise DataValidationError(f"recipe_dir does not exist or is not a directory: {recipe_path}")
    if not train_path.is_file():
        raise DataValidationError(f"train_manifest does not exist or is not a file: {train_path}")
    if not eval_path.is_file():
        raise DataValidationError(f"eval_manifest does not exist or is not a file: {eval_path}")
    training_summary_path = recipe_path / "recognizer-training-summary.json"
    if failure_review is not None:
        if not training_summary_path.is_file():
            raise DataValidationError(f"recognizer training summary does not exist for stopped-family guard: {training_summary_path}")
        summary_payload = _read_json_object(training_summary_path, "recognizer training summary")
        assert_recognizer_family_not_stopped(
            failure_review,
            backend=str(summary_payload.get("backend") or ""),
            base_model=str(summary_payload.get("base_model") or ""),
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_audit_payload: dict[str, Any] = {"required": require_corpus_audit, "present": False}
    if require_corpus_audit:
        corpus_audit = audit_recognizer_corpus(
            train_path,
            eval_path,
            out / "recognizer-corpus-audit",
            min_train_samples=min_train_samples,
            min_eval_samples=min_eval_samples,
            min_train_english=min_train_english,
            min_train_devanagari=min_train_devanagari,
            min_train_mixed=min_train_mixed,
            min_eval_english=min_eval_english,
            min_eval_devanagari=min_eval_devanagari,
            min_eval_mixed=min_eval_mixed,
            min_train_latin_only=min_train_latin_only,
            min_train_devanagari_only=min_train_devanagari_only,
            min_eval_latin_only=min_eval_latin_only,
            min_eval_devanagari_only=min_eval_devanagari_only,
            min_train_real=min_train_real,
            min_train_synthetic=min_train_synthetic,
            require_eval_real=require_eval_real,
        )
        corpus_audit_payload = {
            "required": True,
            "present": True,
            "passed": corpus_audit.passed,
            "path": "recognizer-corpus-audit/recognizer-corpus-audit.json",
        }
        if not corpus_audit.passed:
            raise DataValidationError(
                "recognizer corpus audit failed; refusing to bundle training job: "
                + "; ".join(corpus_audit.issues)
            )
    archive_path = out / archive_name
    manifest_path = out / "recognizer-training-bundle.json"
    readme_path = out / "recognizer-training-bundle.md"
    train_entries = load_manifest(train_path)
    eval_entries = load_manifest(eval_path)
    with tempfile.TemporaryDirectory(prefix="ocrtech-recognizer-bundle-") as tmp_dir_name:
        staging = Path(tmp_dir_name)
        recipe_arcname = _relative_to_base(recipe_path, base=base)
        train_arcname = _relative_to_base(train_path, base=base)
        eval_arcname = _relative_to_base(eval_path, base=base)
        shutil.copytree(recipe_path, staging / recipe_arcname)
        rewritten_train = _stage_bundle_manifest(train_entries, staging, split_name="train", base=base)
        rewritten_eval = _stage_bundle_manifest(eval_entries, staging, split_name="eval", base=base)
        write_manifest(rewritten_train, staging / train_arcname)
        write_manifest(rewritten_eval, staging / eval_arcname)
        _write_paddle_label_file(rewritten_train, staging / recipe_arcname / "labels" / "train.txt")
        _write_paddle_label_file(rewritten_eval, staging / recipe_arcname / "labels" / "eval.txt")
        _rewrite_staged_paddle_config_for_bundle(
            staging / recipe_arcname / "paddle_recognizer.yml",
            recipe_arcname=recipe_arcname,
            recipe_path=recipe_path,
            staging=staging,
            base=base,
        )
        staged_files = sorted(path for path in staging.rglob("*") if path.is_file())
        total_bytes = sum(path.stat().st_size for path in staged_files)
        staged_summary = staging / recipe_arcname / "recognizer-training-summary.json"
        summary_payload = _bundle_file_reference(staged_summary, staging)
        recipe_backend, run_command = _bundle_recipe_run_command(staging / recipe_arcname, recipe_arcname=recipe_arcname)
        with tarfile.open(archive_path, "w:gz") as archive:
            for source in staged_files:
                archive.add(source, arcname=str(source.relative_to(staging)))
    payload = {
        "archive_path": str(archive_path),
        "base_dir": str(base),
        "recipe_dir": str(_relative_to_base(recipe_path, base=base)),
        "train_manifest": str(_relative_to_base(train_path, base=base)),
        "eval_manifest": str(_relative_to_base(eval_path, base=base)),
        "portable_image_root": "bundle_assets/images",
        "recipe_backend": recipe_backend,
        "run_command": run_command,
        "training_summary": summary_payload,
        "recognizer_corpus_audit": corpus_audit_payload,
        "file_count": len(staged_files),
        "total_bytes": total_bytes,
        "files": [str(path.relative_to(staging)) for path in staged_files],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "# Recognizer Training Bundle\n\n"
        f"Archive: `{archive_path}`\n"
        f"Training summary: `{summary_payload.get('path') or 'not present'}`\n"
        f"Recognizer corpus audit: `{corpus_audit_payload.get('path') or 'not required'}`\n"
        f"Files: `{len(staged_files)}`\n"
        f"Bytes before compression: `{total_bytes}`\n\n"
        "Extract this archive at the root of an ocr-tech workspace or an empty training workspace. "
        "The recipe uses relative paths, so run training from the extraction root.\n\n"
        "```bash\n"
        f"tar -xzf {archive_path.name}\n"
        f"{run_command}\n"
        "```\n",
        encoding="utf-8",
    )
    return RecognizerTrainingBundle(
        archive_path=str(archive_path),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        file_count=len(staged_files),
        total_bytes=total_bytes,
)


def _bundle_recipe_run_command(recipe_dir: Path, *, recipe_arcname: Path) -> tuple[str, str]:
    paddle_config = recipe_dir / "paddle_recognizer.yml"
    hf_config = recipe_dir / "hf_recognizer_config.json"
    if paddle_config.is_file():
        return "paddleocr", f"python PaddleOCR/tools/train.py -c {recipe_arcname / 'paddle_recognizer.yml'}"
    if hf_config.is_file():
        config = HfRecognizerTrainingConfig.from_path(hf_config)
        return "hf_vision_encoder_decoder", _hf_recognizer_train_command(config, run=True)
    return "unknown", f"# no supported recognizer training config found under {recipe_dir}"


def _rewrite_staged_paddle_config_for_bundle(
    config_path: Path,
    *,
    recipe_arcname: Path,
    recipe_path: Path,
    staging: Path,
    base: Path,
) -> None:
    if not config_path.exists():
        return
    train_label = recipe_arcname / "labels" / "train.txt"
    eval_label = recipe_arcname / "labels" / "eval.txt"
    output_dir = recipe_arcname / "checkpoints"
    save_res_path = recipe_arcname / "predicts.txt"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    section: str | None = None
    in_label_file_list = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Train:":
            section = "train"
            in_label_file_list = False
        elif stripped == "Eval:":
            section = "eval"
            in_label_file_list = False
        elif line and not line.startswith(" "):
            section = None
            in_label_file_list = False

        if stripped == "label_file_list:":
            in_label_file_list = True
            rewritten.append(line)
            continue
        if in_label_file_list and stripped.startswith("- "):
            label_path = train_label if section == "train" else eval_label if section == "eval" else None
            if label_path is not None:
                rewritten.append(f"      - {label_path}")
                continue
        if in_label_file_list and stripped and not stripped.startswith("- "):
            in_label_file_list = False

        if line.startswith("  character_dict_path: "):
            original = line.split(": ", 1)[1]
            dictionary_rel = _stage_bundle_dictionary(original, recipe_arcname=recipe_arcname, recipe_path=recipe_path, staging=staging, base=base)
            rewritten.append(f"  character_dict_path: {dictionary_rel}")
            continue
        if line.startswith("  save_model_dir: "):
            rewritten.append(f"  save_model_dir: {output_dir}")
            continue
        if line.startswith("  save_res_path: "):
            rewritten.append(f"  save_res_path: {save_res_path}")
            continue
        rewritten.append(line)
    config_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _stage_bundle_dictionary(
    value: str,
    *,
    recipe_arcname: Path,
    recipe_path: Path,
    staging: Path,
    base: Path,
) -> Path:
    source = _resolve_bundle_config_path(value, base=base)
    if not source.is_file():
        raise DataValidationError(f"PaddleOCR character_dict_path does not exist for training bundle: {source}")
    try:
        return recipe_arcname / source.relative_to(recipe_path)
    except ValueError:
        target_rel = recipe_arcname / "dictionaries" / source.name
        target = staging / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target_rel


def _resolve_bundle_config_path(value: str, *, base: Path) -> Path:
    source = Path(value)
    if source.is_absolute():
        return source.resolve()
    cwd_relative = source.resolve()
    if cwd_relative.exists():
        return cwd_relative
    return (base / source).resolve()


def _bundle_file_reference(path: Path, staging_root: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"present": False, "path": str(path.relative_to(staging_root))}
    return {
        "present": True,
        "path": str(path.relative_to(staging_root)),
        "sha256": sha256_file(path),
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _stage_bundle_manifest(entries: list[ManifestEntry], staging: Path, *, split_name: str, base: Path) -> list[ManifestEntry]:
    rewritten: list[ManifestEntry] = []
    for entry in entries:
        image_path = Path(entry.image_path)
        if not image_path.is_absolute():
            image_path = base / image_path
        image_path = image_path.resolve()
        if not image_path.exists():
            raise DataValidationError(f"manifest image_path does not exist for {entry.sample_id}: {image_path}")
        if not image_path.is_file():
            raise DataValidationError(f"recognizer training bundle only supports file image_path entries: {entry.sample_id}: {image_path}")
        suffix = image_path.suffix or ".img"
        target_rel = Path("bundle_assets") / "images" / split_name / f"{_safe_sample_id(entry.sample_id)}{suffix}"
        target = staging / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target)
        rewritten.append(
            ManifestEntry(
                sample_id=entry.sample_id,
                dataset=entry.dataset,
                split=entry.split,
                image_path=str(target_rel),
                text=entry.text,
                sha256=entry.sha256,
                metadata=dict(entry.metadata),
            )
        )
    return rewritten


def _safe_sample_id(sample_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in sample_id)
    return safe or "sample"


def _resolve_under_base(path: str | Path, *, base: Path) -> Path:
    value = Path(path)
    resolved = value if value.is_absolute() else base / value
    resolved = resolved.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise DataValidationError(f"path must be inside base_dir {base}: {path}") from exc
    return resolved


def _relative_to_base(path: Path, *, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base)
    except ValueError as exc:
        raise DataValidationError(f"path must be inside base_dir {base}: {path}") from exc


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
            raise DataValidationError("TrOCR processor loading requires sentencepiece and protobuf. Install ocr-tech[hf-recognizer].") from exc
        except ValueError as exc:
            message = str(exc)
            if "SentencePiece" in message or "sentencepiece" in message or "tiktoken" in message or "protobuf" in message:
                raise DataValidationError("TrOCR processor loading requires sentencepiece and protobuf. Install ocr-tech[hf-recognizer].") from exc
            raise
    return processor_cls.from_pretrained(str(source))


def _read_token_id(config: object, name: str) -> int | None:
    value = getattr(config, name, None)
    if value is not None:
        return int(value)
    nested_decoder = getattr(config, "decoder", None)
    nested_value = getattr(nested_decoder, name, None)
    return int(nested_value) if nested_value is not None else None


def _set_attr_if_possible(target: object, name: str, value: int) -> None:
    try:
        setattr(target, name, value)
    except AttributeError:
        return


def _write_token_ids(model: object, *, decoder_start_token_id: int, pad_token_id: int) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        _set_attr_if_possible(config, "decoder_start_token_id", decoder_start_token_id)
        _set_attr_if_possible(config, "pad_token_id", pad_token_id)
        nested_decoder = getattr(config, "decoder", None)
        if nested_decoder is not None:
            _set_attr_if_possible(nested_decoder, "decoder_start_token_id", decoder_start_token_id)
            _set_attr_if_possible(nested_decoder, "pad_token_id", pad_token_id)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _set_attr_if_possible(generation_config, "decoder_start_token_id", decoder_start_token_id)
        _set_attr_if_possible(generation_config, "pad_token_id", pad_token_id)


def write_corrector_recipe(
    output_dir: str | Path,
    *,
    clean_manifest: str | Path | None = None,
    pairs_source: str | Path | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    limit: int | None = None,
    seed: int = 13,
) -> Path:
    sources = [clean_manifest is not None, pairs_source is not None, hf_dataset is not None]
    if sum(1 for item in sources if item) != 1:
        raise DataValidationError("train-corrector requires exactly one of --clean-manifest, --pairs-source, or --hf-dataset")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "correction_pairs.jsonl"
    if pairs_source is not None:
        _copy_pairs(pairs_source, pairs_path, limit=limit)
    elif hf_dataset is not None:
        prepare_hf_correction_pairs(hf_dataset, pairs_path, split=hf_split, limit=limit)
    else:
        entries = load_manifest(clean_manifest or "")
        pairs = _synthetic_pairs(entries, limit=limit, seed=seed)
        with pairs_path.open("w", encoding="utf-8") as handle:
            for noisy, clean in pairs:
                handle.write(json.dumps({"noisy_text": noisy, "clean_text": clean}, ensure_ascii=False) + "\n")
    config_path = out / "corrector_config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "ocr_text_correction",
                "pairs": str(pairs_path),
                "recommended_model_family": "compact seq2seq or encoder-decoder text corrector",
                "metrics": ["cer", "wer", "exact_line_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Corrector Training Recipe\n\n"
        f"Pairs: `{pairs_path}`\n\n"
        "Use these noisy-clean pairs with a compact text-to-text model. Keep Nepali and English validation sets separate.\n",
        encoding="utf-8",
    )
    return config_path


def generate_correction_pairs(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    engine_name: str = "tesseract",
    model_config: str | Path | None = None,
    limit: int | None = None,
    only_errors: bool = False,
) -> Path:
    from .engines import create_engine

    entries = load_manifest(manifest_path)
    if not entries:
        raise DataValidationError("generate-correction-pairs requires a non-empty manifest")
    engine_kwargs: dict[str, Any] = {}
    if engine_name in {"candidate", "ours"}:
        engine_kwargs["model_config"] = model_config
    engine = create_engine(engine_name, **engine_kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            if limit is not None and written >= limit:
                break
            recognized = engine.recognize(Path(entry.image_path))
            noisy = normalize_ocr_text("\n".join(line.text for page in recognized.pages for line in page.text_lines))
            clean = normalize_ocr_text(entry.text)
            if only_errors and noisy == clean:
                continue
            handle.write(
                json.dumps(
                    {
                        "sample_id": entry.sample_id,
                        "engine": engine_name,
                        "noisy_text": noisy,
                        "clean_text": clean,
                        "slices": list(entry.metadata.get("slices") or []) if entry.metadata else [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    if written == 0:
        raise DataValidationError("No correction pairs generated")
    return out


def write_hf_text_corrector_recipe(
    pairs_source: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "google/byt5-small",
    max_source_length: int = 256,
    max_target_length: int = 256,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 3.0,
    warmup_steps: int = 0,
    logging_steps: int = 10,
    save_total_limit: int = 2,
    eval_ratio: float = 0.1,
    seed: int = 13,
    run: bool = False,
) -> Path:
    pairs = _load_correction_pairs(Path(pairs_source))
    if len(pairs) < 2:
        raise DataValidationError("hf text corrector training requires at least two correction pairs")
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise DataValidationError("hf text corrector eval_ratio must be between 0 and 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    eval_count = min(max(1, round(len(shuffled) * eval_ratio)), len(shuffled) - 1)
    eval_pairs = shuffled[:eval_count]
    train_pairs = shuffled[eval_count:]
    train_pairs_path = out / "train_pairs.jsonl"
    eval_pairs_path = out / "eval_pairs.jsonl"
    _write_correction_pairs_jsonl(train_pairs, train_pairs_path)
    _write_correction_pairs_jsonl(eval_pairs, eval_pairs_path)
    config = HfTextCorrectorTrainingConfig(
        train_pairs=str(train_pairs_path),
        eval_pairs=str(eval_pairs_path),
        output_dir=str(out),
        base_model=base_model,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        predict_with_generate=True,
        generation_max_length=max_target_length,
    )
    config_path = out / "hf_text_corrector_config.json"
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Hugging Face Text Corrector Training Recipe",
                "",
                f"Base model: `{base_model}`",
                f"Pairs source: `{pairs_source}` ({len(pairs)} pairs)",
                f"Train pairs: `{train_pairs_path}` ({len(train_pairs)} pairs)",
                f"Eval pairs: `{eval_pairs_path}` ({len(eval_pairs)} pairs)",
                "",
                "Install the backend dependencies first:",
                "",
                "```bash",
                "python -m pip install -e '.[hf-recognizer]'",
                "```",
                "",
                "Run in-process:",
                "",
                "```bash",
                f"ocrtech train-corrector --backend hf-seq2seq --pairs-source {pairs_source} --out {out} --run",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if run:
        run_hf_text_corrector_training(config_path)
    return config_path


def run_hf_text_corrector_training(config_path: str | Path) -> Path:
    config = HfTextCorrectorTrainingConfig.from_path(config_path)
    train_pairs = _load_correction_pairs(Path(config.train_pairs))
    eval_pairs = _load_correction_pairs(Path(config.eval_pairs))
    if not train_pairs:
        raise DataValidationError("hf text corrector training requires non-empty train_pairs")
    if not eval_pairs:
        raise DataValidationError("hf text corrector training requires non-empty eval_pairs")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError as exc:
        raise DataValidationError("hf text corrector training requires transformers. Install ocr-tech[hf-recognizer].") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
    model.generation_config.max_length = config.generation_max_length
    train_dataset = _TextCorrectionDataset(train_pairs, tokenizer, config.max_source_length, config.max_target_length)
    eval_dataset = _TextCorrectionDataset(eval_pairs, tokenizer, config.max_source_length, config.max_target_length)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=_hf_text_corrector_metrics_factory(tokenizer),
    )
    trainer.train()
    final_model_dir = output_dir / "final-model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    metrics = trainer.evaluate(max_length=config.generation_max_length)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_model_dir


def write_recognizer_export_recipe(
    training_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    inference_dir: str | Path | None = None,
    paddleocr_dir: str | Path = "PaddleOCR",
    export_options: Sequence[str] | None = None,
    run: bool = False,
) -> Path:
    config_path = Path(training_config)
    checkpoint_path = _paddle_checkpoint_prefix(Path(checkpoint))
    if not config_path.exists():
        raise DataValidationError(f"training_config does not exist: {config_path}")
    if run and not _paddle_checkpoint_exists(checkpoint_path):
        raise DataValidationError(f"checkpoint does not exist: {checkpoint_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dir = Path(inference_dir) if inference_dir else out / "inference"
    options = list(export_options or [])
    command = [
        "python",
        str(Path(paddleocr_dir) / "tools" / "export_model.py"),
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={checkpoint_path}",
        f"Global.save_inference_dir={save_dir}",
        *options,
    ]
    export_spec = {
        "training_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "paddleocr_dir": str(paddleocr_dir),
        "save_inference_dir": str(save_dir),
        "export_options": options,
        "command": command,
    }
    export_path = out / "export-recognizer.json"
    export_path.write_text(json.dumps(export_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "EXPORT.md").write_text(
        "# Recognizer Export\n\n"
        "Run this after training has produced a PaddleOCR checkpoint.\n\n"
        "```bash\n"
        + " ".join(export_spec["command"])
        + "\n```\n",
        encoding="utf-8",
    )
    if run:
        completed = subprocess.run(export_spec["command"], check=False)
        if completed.returncode != 0:
            raise DataValidationError(f"PaddleOCR export failed with exit code {completed.returncode}")
    return export_path


def _paddle_checkpoint_prefix(checkpoint: Path) -> Path:
    if checkpoint.name.endswith(".pdparams"):
        return checkpoint.with_name(checkpoint.name[: -len(".pdparams")])
    return checkpoint


def _paddle_checkpoint_exists(checkpoint_prefix: Path) -> bool:
    return checkpoint_prefix.exists() or checkpoint_prefix.with_name(f"{checkpoint_prefix.name}.pdparams").is_file()


def package_paddleocr_model(
    inference_dir: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    dictionary_path: str | Path,
    base_model: str = PADDLE_PPOCRV5_MOBILE_REC_PRETRAINED,
    paddle_lang: str = "ne",
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec",
    train_manifest: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    training_summary: str | Path | None = None,
    metrics_report: str | Path | None = None,
    admission_validation_report: str | Path | None = None,
    recognition_mode: str = "full_page",
    line_mode_max_height: int = 256,
    line_mode_min_aspect_ratio: float = 3.0,
    source_archive: str | Path | None = None,
    source_checkpoint: str | Path | None = None,
    source_training_config: str | Path | None = None,
    export_recipe: str | Path | None = None,
) -> Path:
    source_dir = Path(inference_dir)
    dictionary = Path(dictionary_path)
    if not source_dir.exists() or not source_dir.is_dir():
        raise DataValidationError(f"inference_dir does not exist or is not a directory: {source_dir}")
    if not dictionary.exists() or not dictionary.is_file():
        raise DataValidationError(f"dictionary_path does not exist or is not a file: {dictionary}")
    normalized_recognition_mode = recognition_mode.strip().lower()
    if normalized_recognition_mode == "recognition_only":
        normalized_recognition_mode = "line"
    if normalized_recognition_mode not in {"full_page", "line", "auto"}:
        raise DataValidationError(f"unsupported PaddleOCR recognition_mode: {recognition_mode!r}")
    model_files = _find_paddle_inference_files(source_dir)
    out = Path(output_dir)
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file_path in model_files:
        target = artifacts_dir / file_path.name
        shutil.copy2(file_path, target)
        copied.append(target)
    dict_target = artifacts_dir / dictionary.name
    shutil.copy2(dictionary, dict_target)
    copied.append(dict_target)
    card_path = out / "model-card.json"
    backend_kwargs: dict[str, object] = {
        "lang": paddle_lang,
        "text_recognition_model_name": text_recognition_model_name,
        "text_recognition_model_dir": str(artifacts_dir.relative_to(out)),
    }
    if normalized_recognition_mode != "full_page":
        backend_kwargs.update(
            {
                "recognition_mode": normalized_recognition_mode,
                "line_mode_max_height": int(line_mode_max_height),
                "line_mode_min_aspect_ratio": float(line_mode_min_aspect_ratio),
            }
        )
    provenance = _optional_path_provenance(
        {
            "source_archive": source_archive,
            "source_checkpoint": source_checkpoint,
            "source_training_config": source_training_config,
            "export_recipe": export_recipe,
        }
    )
    write_model_card(
        card_path,
        model_id=model_id,
        backend="paddleocr",
        base_model=base_model,
        artifact_paths=[path.relative_to(out) for path in copied],
        backend_kwargs=backend_kwargs,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        training_summary=training_summary,
        metrics_report=metrics_report,
        admission_validation_report=admission_validation_report,
        artifact_base_path=out,
        provenance=provenance,
    )
    (out / "PACKAGE.md").write_text(
        "# Packaged PaddleOCR Candidate\n\n"
        f"Model card: `{card_path}`\n"
        f"Artifacts: `{artifacts_dir}`\n\n"
        "Audit before benchmarking:\n\n"
        "```bash\n"
        f"ocrtech audit-model {card_path}\n"
        "```\n",
        encoding="utf-8",
    )
    return card_path


def _optional_path_provenance(paths: dict[str, str | Path | None]) -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}
    for key, raw_path in paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        provenance[key] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
    return provenance


def package_hf_recognizer_model(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    base_model: str,
    train_manifest: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    training_summary: str | Path | None = None,
    metrics_report: str | Path | None = None,
    admission_validation_report: str | Path | None = None,
    max_new_tokens: int = 128,
) -> Path:
    source_dir = Path(model_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise DataValidationError(f"model_dir does not exist or is not a directory: {source_dir}")
    out = Path(output_dir)
    artifacts_dir = out / "artifacts" / "model"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, artifacts_dir)
    card_path = out / "model-card.json"
    write_model_card(
        card_path,
        model_id=model_id,
        backend="hf_vision_encoder_decoder",
        base_model=base_model,
        artifact_paths=[artifacts_dir.relative_to(out)],
        backend_kwargs={
            "model_dir": str(artifacts_dir.relative_to(out)),
            "processor_dir": str(artifacts_dir.relative_to(out)),
            "max_new_tokens": max_new_tokens,
        },
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        training_summary=training_summary,
        metrics_report=metrics_report,
        admission_validation_report=admission_validation_report,
        artifact_base_path=out,
    )
    (out / "PACKAGE.md").write_text(
        "# Packaged Hugging Face OCR Candidate\n\n"
        f"Model card: `{card_path}`\n"
        f"Artifacts: `{artifacts_dir}`\n\n"
        "Audit before benchmarking:\n\n"
        "```bash\n"
        f"ocrtech audit-model {card_path}\n"
        "```\n",
        encoding="utf-8",
    )
    return card_path


def finalize_hf_recognizer_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    model_id: str,
    base_model: str,
    train_manifest: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    training_summary: str | Path | None = None,
    admission_validation_report: str | Path | None = None,
    max_new_tokens: int = 128,
) -> HfRecognizerFinalization:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise DataValidationError(f"run_dir does not exist or is not a directory: {run_path}")
    final_model_dir = run_path / "final-model"
    metrics_path = run_path / "training_metrics.json"
    if not final_model_dir.exists() or not final_model_dir.is_dir():
        raise DataValidationError(f"HF recognizer run is not finished; missing final model directory: {final_model_dir}")
    if not metrics_path.exists() or not metrics_path.is_file():
        raise DataValidationError(f"HF recognizer run is not finished; missing training metrics report: {metrics_path}")
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid training metrics JSON {metrics_path}: {exc}") from exc
    if not isinstance(metrics_payload, dict):
        raise DataValidationError(f"training metrics report must be a JSON object: {metrics_path}")

    out = Path(output_dir) if output_dir is not None else run_path / "finalized"
    package_dir = out / "package"
    audit_dir = out / "audit"
    report_json = out / "hf-recognizer-finalization.json"
    report_md = out / "hf-recognizer-finalization.md"
    out.mkdir(parents=True, exist_ok=True)
    card_path = package_hf_recognizer_model(
        final_model_dir,
        package_dir,
        model_id=model_id,
        base_model=base_model,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        training_summary=training_summary,
        metrics_report=metrics_path,
        admission_validation_report=admission_validation_report,
        max_new_tokens=max_new_tokens,
    )
    audit = audit_model_card(card_path, audit_dir)
    result = HfRecognizerFinalization(
        run_dir=str(run_path),
        package_dir=str(package_dir),
        model_card_path=str(card_path),
        audit_path=str(audit_dir / "model-audit.json"),
        report_json_path=str(report_json),
        report_md_path=str(report_md),
        audit_passed=audit.passed,
    )
    report_json.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_render_hf_recognizer_finalization(result), encoding="utf-8")
    return result


def _render_hf_recognizer_finalization(result: HfRecognizerFinalization) -> str:
    return "\n".join(
        [
            "# HF Recognizer Finalization",
            "",
            f"Run directory: `{result.run_dir}`",
            f"Package directory: `{result.package_dir}`",
            f"Model card: `{result.model_card_path}`",
            f"Model audit: `{result.audit_path}`",
            f"Audit passed: `{'yes' if result.audit_passed else 'no'}`",
            "",
            "## Next Claim-Bearing Steps",
            "",
            "1. Run `ocrtech audit-model` on the packaged model card if the generated audit did not pass.",
            "2. Benchmark the packaged candidate against named baselines on the held-out manifest.",
            "3. Summarize paired metrics and run `ocrtech validate-claim` before writing any SOTA claim.",
            "",
            "Example experiment command:",
            "",
            "```bash",
            "ocrtech run-experiment \\",
            "  --eval-manifest data/splits-mixed-v1/eval-with-refs.jsonl \\",
            "  --out outputs/experiment-hf-recognizer-mixed-v2-full \\",
            "  --baselines candidate,tesseract,surya,stock-paddle,glm-ocr,paddleocr-vl \\",
            "  --candidate-model-config " + result.model_card_path + " \\",
            "  --train-manifest data/splits-mixed-v1/train-rebalanced-nep-eng.jsonl",
            "```",
            "",
        ]
    )


def package_hf_text_corrector_model(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    model_id: str,
    base_model: str,
    base_engine: str = "tesseract",
    base_engine_kwargs: dict[str, Any] | None = None,
    train_manifest: str | Path | None = None,
    eval_manifest: str | Path | None = None,
    metrics_report: str | Path | None = None,
    max_new_tokens: int = 256,
) -> Path:
    source_dir = Path(model_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise DataValidationError(f"model_dir does not exist or is not a directory: {source_dir}")
    out = Path(output_dir)
    artifacts_dir = out / "artifacts" / "model"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, artifacts_dir)
    card_path = out / "model-card.json"
    write_model_card(
        card_path,
        model_id=model_id,
        backend="text_correction_composite",
        base_model=base_model,
        artifact_paths=[artifacts_dir.relative_to(out)],
        backend_kwargs={
            "base_engine": base_engine,
            "base_engine_kwargs": base_engine_kwargs or {},
            "model_dir": str(artifacts_dir.relative_to(out)),
            "tokenizer_dir": str(artifacts_dir.relative_to(out)),
            "max_new_tokens": max_new_tokens,
        },
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        metrics_report=metrics_report,
    )
    (out / "PACKAGE.md").write_text(
        "# Packaged Text Correction Composite Candidate\n\n"
        f"Model card: `{card_path}`\n"
        f"Artifacts: `{artifacts_dir}`\n\n"
        "Audit before benchmarking:\n\n"
        "```bash\n"
        f"ocrtech audit-model {card_path}\n"
        "```\n",
        encoding="utf-8",
    )
    return card_path


def _copy_pairs(source: str | Path, target: Path, *, limit: int | None) -> None:
    source_path = Path(source)
    count = 0
    with source_path.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            data = json.loads(line)
            if "noisy_text" not in data or "clean_text" not in data:
                raise DataValidationError(f"Correction pair at {source_path}:{line_number} needs noisy_text and clean_text")
            dst.write(json.dumps({"noisy_text": data["noisy_text"], "clean_text": data["clean_text"]}, ensure_ascii=False) + "\n")
            count += 1


def _synthetic_pairs(entries: list[ManifestEntry], *, limit: int | None, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        clean = normalize_ocr_text(entry.text)
        noisy = _corrupt_text(clean, rng)
        if noisy != clean:
            pairs.append((noisy, clean))
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise DataValidationError("No correction pairs generated")
    return pairs


def _load_correction_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            noisy = data.get("noisy_text")
            clean = data.get("clean_text")
            if not isinstance(noisy, str) or not isinstance(clean, str):
                raise DataValidationError(f"Correction pair at {path}:{line_number} needs string noisy_text and clean_text")
            pairs.append({"noisy_text": normalize_ocr_text(noisy), "clean_text": normalize_ocr_text(clean)})
    if not pairs:
        raise DataValidationError(f"No correction pairs found in {path}")
    return pairs


def _write_correction_pairs_jsonl(pairs: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _corrupt_text(text: str, rng: random.Random) -> str:
    chars = list(text)
    if len(chars) < 2:
        return text
    operations = max(1, len(chars) // 24)
    for _ in range(operations):
        index = rng.randrange(len(chars))
        draw = rng.random()
        if draw < 0.4:
            chars.pop(index)
            if not chars:
                break
        elif draw < 0.7:
            chars[index] = " "
        else:
            chars.insert(index, chars[index])
    return "".join(chars)


def _find_paddle_inference_files(source_dir: Path) -> list[Path]:
    files = [path for path in source_dir.iterdir() if path.is_file()]
    if not files:
        raise DataValidationError(f"inference_dir contains no files: {source_dir}")
    parameter_files = sorted(path for path in files if path.suffix == ".pdiparams" or path.name.endswith(".pdiparams"))
    model_files = sorted(path for path in files if path.suffix in {".pdmodel", ".json"} or path.name in {"inference.pdmodel", "inference.json"})
    config_files = sorted(path for path in files if path.name in {"inference.yml", "inference.yaml"})
    info_files = sorted(path for path in files if path.name.endswith(".pdiparams.info"))
    if not parameter_files:
        raise DataValidationError(f"inference_dir is missing PaddleOCR parameter file (*.pdiparams): {source_dir}")
    if not model_files:
        raise DataValidationError(f"inference_dir is missing PaddleOCR model file (*.pdmodel or *.json): {source_dir}")
    return [*model_files, *parameter_files, *config_files, *info_files]


class _VisionTextDataset:
    def __init__(self, entries: list[ManifestEntry], processor: Any, image_module: Any, max_target_length: int) -> None:
        self.entries = entries
        self.processor = processor
        self.image_module = image_module
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image = self.image_module.open(Path(entry.image_path)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            entry.text,
            padding=False,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "labels": labels}


class _VisionTextCollator:
    def __init__(self, processor: Any, torch: Any) -> None:
        self.processor = processor
        self.torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = self.torch.stack([feature["pixel_values"] for feature in features])
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt").input_ids
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {"pixel_values": pixel_values, "labels": labels}


class _TextCorrectionDataset:
    def __init__(self, pairs: list[dict[str, str]], tokenizer: Any, max_source_length: int, max_target_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        model_inputs = self.tokenizer(
            pair["noisy_text"],
            padding=False,
            truncation=True,
            max_length=self.max_source_length,
        )
        labels = self.tokenizer(
            text_target=pair["clean_text"],
            padding=False,
            truncation=True,
            max_length=self.max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def _hf_compute_metrics_factory(processor: Any):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        prediction_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        predicted_text = processor.batch_decode(
            _sanitize_batch_token_ids(processor.tokenizer, prediction_ids),
            skip_special_tokens=True,
        )
        reference_text = processor.batch_decode(
            _sanitize_batch_token_ids(processor.tokenizer, label_ids),
            skip_special_tokens=True,
        )
        return {
            "cer": sum(cer(pred, ref) for pred, ref in zip(predicted_text, reference_text, strict=True)) / len(reference_text),
            "wer": sum(wer(pred, ref) for pred, ref in zip(predicted_text, reference_text, strict=True)) / len(reference_text),
            "exact_line_accuracy": exact_line_accuracy(predicted_text, reference_text),
        }

    return compute_metrics


def _hf_text_corrector_metrics_factory(tokenizer: Any):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        prediction_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        predicted_text = [
            normalize_ocr_text(text)
            for text in tokenizer.batch_decode(_sanitize_batch_token_ids(tokenizer, prediction_ids), skip_special_tokens=True)
        ]
        reference_text = [
            normalize_ocr_text(text)
            for text in tokenizer.batch_decode(_sanitize_batch_token_ids(tokenizer, label_ids), skip_special_tokens=True)
        ]
        return {
            "cer": sum(cer(pred, ref) for pred, ref in zip(predicted_text, reference_text, strict=True)) / len(reference_text),
            "wer": sum(wer(pred, ref) for pred, ref in zip(predicted_text, reference_text, strict=True)) / len(reference_text),
            "exact_line_accuracy": exact_line_accuracy(predicted_text, reference_text),
        }

    return compute_metrics


def _sanitize_batch_token_ids(tokenizer: Any, token_rows: Any) -> list[list[int]]:
    rows = token_rows.tolist() if hasattr(token_rows, "tolist") else token_rows
    pad_token_id = _fallback_token_id(tokenizer)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    sanitized_rows: list[list[int]] = []
    for row in rows:
        values = row.tolist() if hasattr(row, "tolist") else row
        sanitized_rows.append([_sanitize_token_id(token, pad_token_id=pad_token_id, vocab_size=vocab_size) for token in values])
    return sanitized_rows


def _sanitize_token_id(token: Any, *, pad_token_id: int, vocab_size: int | None) -> int:
    try:
        value = int(token)
    except (TypeError, ValueError, OverflowError):
        return pad_token_id
    if value == -100:
        return pad_token_id
    if value < 0:
        return pad_token_id
    if vocab_size is not None and value >= vocab_size:
        return pad_token_id
    return value


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return len(tokenizer)
    except TypeError:
        return None


def _fallback_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    return 0
