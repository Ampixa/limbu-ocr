"""Dataset verification and manifest conversion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import DataValidationError
from .normalization import normalize_ocr_text, unsupported_characters


IMAGE_FIELD_CANDIDATES = ("image", "image_path", "path", "file", "filename", "img")
TEXT_FIELD_CANDIDATES = ("text", "label", "transcription", "ground_truth", "gt", "clean_text")
SUPPORTED_OCR_IMAGE_SUFFIXES = frozenset({".bmp", ".dib", ".jpeg", ".jpg", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pnm", ".sr", ".ras", ".tiff", ".tif"})


@dataclass(slots=True)
class ManifestEntry:
    sample_id: str
    dataset: str
    split: str
    image_path: str
    text: str
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise DataValidationError("manifest entry sample_id is required")
        if not self.dataset:
            raise DataValidationError("manifest entry dataset is required")
        if not self.split:
            raise DataValidationError("manifest entry split is required")
        if not self.image_path:
            raise DataValidationError(f"manifest entry {self.sample_id} image_path is required")
        if not isinstance(self.text, str):
            raise DataValidationError(f"manifest entry {self.sample_id} text must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "image_path": self.image_path,
            "text": self.text,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestEntry":
        return cls(
            sample_id=str(data.get("sample_id") or data.get("id") or ""),
            dataset=str(data.get("dataset") or ""),
            split=str(data.get("split") or ""),
            image_path=str(data.get("image_path") or data.get("image") or ""),
            text=str(data.get("text") if data.get("text") is not None else ""),
            sha256=data.get("sha256"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class PrepareReject:
    row_index: int
    reason: str
    row: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "reason": self.reason,
            "row": self.row,
        }


@dataclass(slots=True)
class PrepareReport:
    manifest_path: str
    sample_count: int
    rejected_count: int
    rejects_path: str | None
    summary_json_path: str
    summary_md_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "rejected_count": self.rejected_count,
            "rejects_path": self.rejects_path,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


@dataclass(slots=True)
class HfDatasetInspection:
    dataset_id: str
    split: str
    subset: str | None
    inspection_mode: str
    fields: list[str]
    inferred_image_field: str | None
    inferred_text_field: str | None
    inspected_count: int
    samples: list[dict[str, Any]]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_json_path: str | None = None
    output_md_path: str | None = None

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "subset": self.subset,
            "inspection_mode": self.inspection_mode,
            "passed": self.passed,
            "fields": self.fields,
            "inferred_image_field": self.inferred_image_field,
            "inferred_text_field": self.inferred_text_field,
            "inspected_count": self.inspected_count,
            "samples": self.samples,
            "issues": self.issues,
            "warnings": self.warnings,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
        }


@dataclass(slots=True)
class ManifestImageNormalizationReport:
    source_manifest: str
    manifest_path: str
    image_dir: str
    sample_count: int
    copied_count: int
    converted_count: int
    unsupported_suffixes: dict[str, int]
    summary_json_path: str
    summary_md_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_manifest": self.source_manifest,
            "manifest_path": self.manifest_path,
            "image_dir": self.image_dir,
            "sample_count": self.sample_count,
            "copied_count": self.copied_count,
            "converted_count": self.converted_count,
            "unsupported_suffixes": self.unsupported_suffixes,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise DataValidationError(f"cannot hash missing or unsupported path: {path}")
    digest = hashlib.sha256()
    file_paths = sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in file_paths:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataValidationError(f"JSONL row at {path}:{line_number} must be an object")
            yield value


def read_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DataValidationError(f"CSV has no header: {path}")
        for row in reader:
            yield dict(row)


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    manifest_path = Path(path)
    entries = [ManifestEntry.from_dict(row) for row in read_jsonl(manifest_path)]
    seen: set[str] = set()
    for entry in entries:
        if entry.sample_id in seen:
            raise DataValidationError(f"duplicate sample_id in manifest {manifest_path}: {entry.sample_id}")
        seen.add(entry.sample_id)
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def normalize_manifest_images(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    image_dir_name: str = "images",
    output_manifest_name: str = "manifest-images-normalized.jsonl",
) -> ManifestImageNormalizationReport:
    source_manifest = Path(manifest_path)
    entries = load_manifest(source_manifest)
    if not entries:
        raise DataValidationError("image normalization requires a non-empty manifest")
    out = Path(output_dir)
    image_dir = out / image_dir_name
    image_dir.mkdir(parents=True, exist_ok=True)
    rewritten: list[ManifestEntry] = []
    copied_count = 0
    converted_count = 0
    unsupported_suffixes: dict[str, int] = {}
    targets: set[Path] = set()
    for entry in entries:
        source_image = _resolve_manifest_image_path(entry.image_path, source_manifest.parent)
        if not source_image.exists():
            raise DataValidationError(f"manifest image_path does not exist for {entry.sample_id}: {source_image}")
        if not source_image.is_file():
            raise DataValidationError(f"image normalization only supports file image_path entries: {entry.sample_id}: {source_image}")
        source_suffix = source_image.suffix.lower()
        target_suffix = source_suffix if source_suffix in SUPPORTED_OCR_IMAGE_SUFFIXES else ".png"
        target = image_dir / f"{_safe_manifest_filename(entry.sample_id)}{target_suffix}"
        if target in targets:
            raise DataValidationError(f"duplicate normalized image target for sample_id {entry.sample_id}: {target}")
        targets.add(target)
        metadata = dict(entry.metadata or {})
        metadata["source_image_path"] = entry.image_path
        metadata["source_image_sha256"] = sha256_file(source_image)
        metadata["normalized_image_suffix"] = target_suffix
        if source_suffix in SUPPORTED_OCR_IMAGE_SUFFIXES:
            shutil.copy2(source_image, target)
            copied_count += 1
            metadata["image_normalization"] = "copied"
        else:
            unsupported_suffix = source_suffix or "<none>"
            unsupported_suffixes[unsupported_suffix] = unsupported_suffixes.get(unsupported_suffix, 0) + 1
            _convert_image_to_png(source_image, target, sample_id=entry.sample_id)
            converted_count += 1
            metadata["image_normalization"] = "converted_to_png"
            metadata["original_image_suffix"] = unsupported_suffix
        rewritten.append(
            ManifestEntry(
                sample_id=entry.sample_id,
                dataset=entry.dataset,
                split=entry.split,
                image_path=str(target),
                text=entry.text,
                sha256=sha256_file(target),
                metadata=metadata,
            )
        )
    manifest_out = out / output_manifest_name
    write_manifest(rewritten, manifest_out)
    summary_json = out / "image-normalization.json"
    summary_md = out / "image-normalization.md"
    report = ManifestImageNormalizationReport(
        source_manifest=str(source_manifest),
        manifest_path=str(manifest_out),
        image_dir=str(image_dir),
        sample_count=len(rewritten),
        copied_count=copied_count,
        converted_count=converted_count,
        unsupported_suffixes=dict(sorted(unsupported_suffixes.items())),
        summary_json_path=str(summary_json),
        summary_md_path=str(summary_md),
    )
    summary_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_image_normalization_markdown(summary_md, report)
    return report


def _resolve_manifest_image_path(path_value: str, manifest_dir: Path) -> Path:
    raw = Path(path_value)
    if raw.exists() or raw.is_absolute():
        return raw
    return manifest_dir / raw


def infer_field(row: dict[str, Any], explicit: str | None, candidates: tuple[str, ...], role: str) -> str:
    if explicit:
        if explicit not in row:
            raise DataValidationError(f"Requested {role} field {explicit!r} is missing. Available fields: {sorted(row)}")
        return explicit
    present = [field for field in candidates if field in row]
    matches = [field for field in present if _has_field_value(row[field])]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]
    if len(present) == 1:
        return present[0]
    if len(present) > 1:
        return present[0]
    raise DataValidationError(
        f"Could not infer {role} field. Available fields: {sorted(row)}. "
        f"Pass --{role.replace('_', '-')}-field explicitly."
    )


def _has_field_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _safe_manifest_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return safe or "sample"


def _convert_image_to_png(source: Path, target: Path, *, sample_id: str) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("Pillow is required to convert unsupported manifest image suffixes") from exc
    try:
        with Image.open(source) as image:
            image.save(target, format="PNG")
    except Exception as exc:
        raise DataValidationError(f"failed to convert image for {sample_id} to PNG: {source}: {exc}") from exc


def _write_image_normalization_markdown(path: Path, report: ManifestImageNormalizationReport) -> None:
    unsupported = ", ".join(f"`{suffix}`={count}" for suffix, count in report.unsupported_suffixes.items()) or "none"
    path.write_text(
        "\n".join(
            [
                "# Manifest Image Normalization",
                "",
                f"Source manifest: `{report.source_manifest}`",
                f"Rewritten manifest: `{report.manifest_path}`",
                f"Image directory: `{report.image_dir}`",
                f"Samples: `{report.sample_count}`",
                f"Copied images: `{report.copied_count}`",
                f"Converted images: `{report.converted_count}`",
                f"Unsupported suffixes converted: {unsupported}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def local_rows(source: Path) -> Iterator[dict[str, Any]]:
    if source.is_dir():
        for candidate in (source / "labels.jsonl", source / "manifest.jsonl", source / "labels.csv", source / "metadata.csv"):
            if candidate.exists():
                yield from local_rows(candidate)
                return
        raise DataValidationError(f"Directory source {source} must contain labels.jsonl, manifest.jsonl, labels.csv, or metadata.csv")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        yield from read_jsonl(source)
    elif suffix == ".csv":
        yield from read_csv_rows(source)
    else:
        raise DataValidationError(f"Unsupported local source format {source}. Use JSONL, CSV, or a labeled directory.")


def rows_to_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    base_dir: Path | None,
    image_field: str | None = None,
    text_field: str | None = None,
    limit: int | None = None,
    strict_chars: bool = False,
    ) -> list[ManifestEntry]:
    entries, rejects = convert_rows_to_manifest(
        rows,
        dataset=dataset,
        split=split,
        base_dir=base_dir,
        image_field=image_field,
        text_field=text_field,
        limit=limit,
        strict_chars=strict_chars,
        skip_invalid=False,
    )
    if rejects:
        raise DataValidationError(f"Unexpected rejected rows while preparing {dataset} {split}")
    if not entries:
        raise DataValidationError("No rows converted into manifest entries")
    return entries


def convert_rows_to_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    base_dir: Path | None,
    image_field: str | None = None,
    text_field: str | None = None,
    limit: int | None = None,
    strict_chars: bool = False,
    skip_invalid: bool = False,
) -> tuple[list[ManifestEntry], list[PrepareReject]]:
    entries: list[ManifestEntry] = []
    rejects: list[PrepareReject] = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        if not isinstance(row, dict):
            error = DataValidationError(f"Row {index} must be an object")
            if not skip_invalid:
                raise error
            rejects.append(PrepareReject(row_index=index, reason=str(error), row={"_raw": repr(row)}))
            continue
        try:
            entries.append(
                _manifest_entry_from_row(
                    row,
                    row_index=index,
                    dataset=dataset,
                    split=split,
                    base_dir=base_dir,
                    image_field=image_field,
                    text_field=text_field,
                    strict_chars=strict_chars,
                )
            )
        except DataValidationError as exc:
            if not skip_invalid:
                raise
            rejects.append(PrepareReject(row_index=index, reason=str(exc), row=dict(row)))
    return entries, rejects


def _manifest_entry_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset: str,
    split: str,
    base_dir: Path | None,
    image_field: str | None,
    text_field: str | None,
    strict_chars: bool,
) -> ManifestEntry:
    image_key = infer_field(row, image_field, IMAGE_FIELD_CANDIDATES, "image")
    text_key = infer_field(row, text_field, TEXT_FIELD_CANDIDATES, "text")
    image_value = row[image_key]
    if not _has_field_value(image_value):
        raise DataValidationError(f"Row {row_index} image field {image_key!r} is empty")
    raw_text = row[text_key]
    if not _has_field_value(raw_text):
        raise DataValidationError(f"Row {row_index} text field {text_key!r} is empty")
    text = normalize_ocr_text(str(raw_text))
    if not text:
        raise DataValidationError(f"Row {row_index} normalized text is empty")
    if strict_chars:
        unsupported = unsupported_characters(text)
        if unsupported:
            raise DataValidationError(f"Row {row_index} contains unsupported characters: {unsupported}")
    image_path = Path(str(image_value))
    if base_dir is not None and not image_path.is_absolute():
        image_path = base_dir / image_path
    if not image_path.exists():
        raise DataValidationError(f"Row {row_index} image does not exist: {image_path}")
    explicit_sample_id = row.get("sample_id")
    if _has_field_value(explicit_sample_id):
        sample_id = str(explicit_sample_id)
    elif _has_field_value(row.get("id")):
        sample_id = f"{dataset}-{split}-{row['id']}"
    else:
        sample_id = f"{dataset}-{split}-{row_index:08d}"
    image_sha256 = sha256_file(image_path)
    text_sha256 = sha256_text(text)
    return ManifestEntry(
        sample_id=sample_id,
        dataset=dataset,
        split=split,
        image_path=str(image_path),
        text=text,
        sha256=image_sha256,
        metadata={
            "source_row": row_index,
            "image_field": image_key,
            "text_field": text_key,
            "text_sha256": text_sha256,
            "sample_sha256": sha256_text(f"{image_sha256}\n{text_sha256}"),
        },
    )


def prepare_local_dataset(
    source: str | Path,
    output_dir: str | Path,
    *,
    dataset: str | None = None,
    split: str = "train",
    image_field: str | None = None,
    text_field: str | None = None,
    limit: int | None = None,
    strict_chars: bool = False,
    skip_invalid: bool = False,
    slices: list[str] | None = None,
) -> PrepareReport:
    source_path = Path(source)
    if not source_path.exists():
        raise DataValidationError(f"Local source does not exist: {source_path}")
    dataset_name = dataset or source_path.stem or source_path.name
    rows = local_rows(source_path)
    base_dir = source_path if source_path.is_dir() else source_path.parent
    entries, rejects = convert_rows_to_manifest(
        rows,
        dataset=dataset_name,
        split=split,
        base_dir=base_dir,
        image_field=image_field,
        text_field=text_field,
        limit=limit,
        strict_chars=strict_chars,
        skip_invalid=skip_invalid,
    )
    entries = _stamp_slices(entries, slices or [])
    return _finalize_prepare_report(output_dir, dataset_name, split, entries, rejects)


def prepare_hf_dataset(
    dataset_id: str,
    output_dir: str | Path,
    *,
    dataset: str | None = None,
    split: str = "train",
    image_field: str | None = None,
    text_field: str | None = None,
    hf_subset: str | None = None,
    limit: int | None = None,
    strict_chars: bool = False,
    skip_invalid: bool = False,
    slices: list[str] | None = None,
) -> PrepareReport:
    if dataset_id == "prashant0919/nepali-synthetic-ocr-lines":
        return _prepare_prashant_synthetic_dataset(
            dataset_id,
            output_dir,
            dataset=dataset,
            split=split,
            limit=limit,
            strict_chars=strict_chars,
            skip_invalid=skip_invalid,
            slices=slices,
        )
    if dataset_id == "nvidia/OCR-Synthetic-Multilingual-v1":
        return _prepare_nvidia_hdf5_dataset(
            dataset_id,
            output_dir,
            dataset=dataset,
            split=split,
            subset=hf_subset or "en",
            limit=limit,
            strict_chars=strict_chars,
            slices=slices,
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataValidationError("Hugging Face dataset support requires installing ocr-tech[datasets].") from exc
    try:
        hf_dataset = load_dataset(dataset_id, split=split, streaming=False)
    except Exception as exc:
        raise DataValidationError(f"Failed to load Hugging Face dataset {dataset_id!r} split {split!r}: {exc}") from exc

    output_root = Path(output_dir)
    image_dir = output_root / "images" / (dataset or dataset_id.replace("/", "__")) / split
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(hf_dataset):
        if limit is not None and index >= limit:
            break
        if not isinstance(row, dict):
            raise DataValidationError(f"HF row {index} is not an object")
        image_key = infer_field(row, image_field, IMAGE_FIELD_CANDIDATES, "image")
        text_key = infer_field(row, text_field, TEXT_FIELD_CANDIDATES, "text")
        image_value = row[image_key]
        image_path = image_dir / f"{index:08d}.png"
        if isinstance(image_value, str):
            candidate = Path(image_value)
            if not candidate.exists():
                raise DataValidationError(f"HF row {index} image path does not exist locally: {candidate}")
            image_path = candidate
        elif hasattr(image_value, "save"):
            image_value.save(image_path)
        elif isinstance(image_value, dict) and image_value.get("path"):
            candidate = Path(str(image_value["path"]))
            if not candidate.exists():
                raise DataValidationError(f"HF row {index} image path does not exist locally: {candidate}")
            image_path = candidate
        else:
            raise DataValidationError(f"HF row {index} has unsupported image field type: {type(image_value).__name__}")
        rows.append({"image_path": str(image_path), "text": row[text_key], "id": row.get("id", index)})
    dataset_name = dataset or dataset_id.replace("/", "__")
    entries, rejects = convert_rows_to_manifest(
        rows,
        dataset=dataset_name,
        split=split,
        base_dir=None,
        image_field="image_path",
        text_field="text",
        limit=None,
        strict_chars=strict_chars,
        skip_invalid=skip_invalid,
    )
    entries = _stamp_slices(entries, _default_hf_slices(dataset_id, subset=hf_subset) + (slices or []))
    return _finalize_prepare_report(output_dir, dataset_name, split, entries, rejects)


def inspect_hf_dataset(
    dataset_id: str,
    output_dir: str | Path,
    *,
    split: str = "train",
    hf_subset: str | None = None,
    image_field: str | None = None,
    text_field: str | None = None,
    limit: int = 3,
) -> HfDatasetInspection:
    if limit <= 0:
        raise DataValidationError("inspect-hf-dataset limit must be positive")
    if dataset_id == "prashant0919/nepali-synthetic-ocr-lines":
        report = _inspect_prashant_synthetic_dataset(dataset_id, split=split, limit=limit)
    elif dataset_id == "nvidia/OCR-Synthetic-Multilingual-v1":
        report = _inspect_nvidia_hdf5_dataset(dataset_id, split=split, subset=hf_subset or "en", limit=limit)
    else:
        report = _inspect_generic_hf_dataset(
            dataset_id,
            split=split,
            hf_subset=hf_subset,
            image_field=image_field,
            text_field=text_field,
            limit=limit,
        )
    return _write_hf_inspection(report, Path(output_dir))


def prepare_hf_correction_pairs(
    dataset_id: str,
    output_path: str | Path,
    *,
    split: str = "train",
    limit: int | None = None,
) -> Path:
    if dataset_id != "cfilt/RoundTripOCR-nepali":
        raise DataValidationError(f"unsupported correction dataset: {dataset_id}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataValidationError("Hugging Face correction dataset support requires installing ocr-tech[datasets].") from exc
    try:
        hf_dataset = load_dataset(dataset_id, split=split, streaming=False)
    except Exception as exc:
        raise DataValidationError(f"Failed to load Hugging Face dataset {dataset_id!r} split {split!r}: {exc}") from exc
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(hf_dataset):
            if limit is not None and index >= limit:
                break
            if not isinstance(row, dict):
                raise DataValidationError(f"HF correction row {index} is not an object")
            noisy = row.get("ocr")
            clean = row.get("correct")
            if not isinstance(noisy, str) or not isinstance(clean, str):
                raise DataValidationError(f"HF correction row {index} requires string ocr and correct fields")
            handle.write(
                json.dumps(
                    {
                        "noisy_text": normalize_ocr_text(noisy),
                        "clean_text": normalize_ocr_text(clean),
                        "font": row.get("font"),
                        "dataset": dataset_id,
                        "split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise DataValidationError(f"No correction pairs written for {dataset_id} split {split}")
    return out


def _inspect_generic_hf_dataset(
    dataset_id: str,
    *,
    split: str,
    hf_subset: str | None,
    image_field: str | None,
    text_field: str | None,
    limit: int,
) -> HfDatasetInspection:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataValidationError("Hugging Face dataset support requires installing ocr-tech[datasets].") from exc
    try:
        if hf_subset:
            hf_dataset = load_dataset(dataset_id, hf_subset, split=split, streaming=False)
        else:
            hf_dataset = load_dataset(dataset_id, split=split, streaming=False)
    except Exception as exc:
        raise DataValidationError(f"Failed to load Hugging Face dataset {dataset_id!r} split {split!r}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(hf_dataset):
        if index >= limit:
            break
        if not isinstance(row, dict):
            raise DataValidationError(f"HF row {index} is not an object")
        rows.append(row)
    if not rows:
        raise DataValidationError(f"No rows available for Hugging Face dataset {dataset_id!r} split {split!r}")

    fields = sorted(rows[0])
    issues: list[str] = []
    warnings: list[str] = []
    inferred_image = _try_infer_field(rows[0], image_field, IMAGE_FIELD_CANDIDATES, "image", issues)
    inferred_text = _try_infer_field(rows[0], text_field, TEXT_FIELD_CANDIDATES, "text", issues)
    samples = [
        _summarize_hf_row(row, row_index=index, image_field=inferred_image, text_field=inferred_text)
        for index, row in enumerate(rows)
    ]
    for sample in samples:
        if inferred_image and not sample.get("image_supported_by_prepare", False):
            warnings.append(f"row {sample['row_index']} image field {inferred_image!r} may need a dataset-specific converter")
        if inferred_text and sample.get("text_length", 0) == 0:
            issues.append(f"row {sample['row_index']} text field {inferred_text!r} is empty after string conversion")
    return HfDatasetInspection(
        dataset_id=dataset_id,
        split=split,
        subset=hf_subset,
        inspection_mode="datasets.load_dataset",
        fields=fields,
        inferred_image_field=inferred_image,
        inferred_text_field=inferred_text,
        inspected_count=len(samples),
        samples=samples,
        issues=issues,
        warnings=warnings,
    )


def _inspect_prashant_synthetic_dataset(dataset_id: str, *, split: str, limit: int) -> HfDatasetInspection:
    prefix = f"{split}/"
    files = set(_list_dataset_files(dataset_id, prefix=prefix, suffix=""))
    metadata_file = f"{prefix}metadata.jsonl"
    issues: list[str] = []
    if metadata_file not in files:
        issues.append(f"metadata.jsonl is missing: {metadata_file}")
        return HfDatasetInspection(dataset_id, split, None, "huggingface_hub files", [], None, None, 0, [], issues)
    metadata_path = _download_dataset_file(dataset_id, metadata_file)
    samples: list[dict[str, Any]] = []
    fields: set[str] = set()
    for index, row in enumerate(read_jsonl(metadata_path)):
        if index >= limit:
            break
        fields.update(row)
        file_name = row.get("file_name")
        dataset_file = file_name if isinstance(file_name, str) and file_name.startswith(prefix) else f"{prefix}{file_name}"
        samples.append(
            {
                "row_index": index,
                "fields": sorted(row),
                "image_type": "repo_file",
                "image_path": dataset_file,
                "image_exists_in_repo": dataset_file in files,
                "image_supported_by_prepare": dataset_file in files,
                "text_type": type(row.get("text")).__name__,
                "text_length": len(str(row.get("text") or "")),
                "text_preview": _preview_text(row.get("text")),
            }
        )
        if dataset_file not in files:
            issues.append(f"row {index} image file missing from dataset repo: {dataset_file}")
    return HfDatasetInspection(
        dataset_id=dataset_id,
        split=split,
        subset=None,
        inspection_mode="huggingface_hub metadata.jsonl",
        fields=sorted(fields),
        inferred_image_field="file_name",
        inferred_text_field="text",
        inspected_count=len(samples),
        samples=samples,
        issues=issues,
    )


def _inspect_nvidia_hdf5_dataset(dataset_id: str, *, split: str, subset: str, limit: int) -> HfDatasetInspection:
    try:
        import h5py
    except ImportError as exc:
        raise DataValidationError("NVIDIA HDF5 dataset inspection requires installing ocr-tech[datasets].") from exc
    shard_files = _list_dataset_files(dataset_id, prefix=f"{subset}/{split}/", suffix=".h5")
    if not shard_files:
        return HfDatasetInspection(
            dataset_id,
            split,
            subset,
            "hdf5 line_bboxes",
            [],
            None,
            None,
            0,
            [],
            [f"No HDF5 shards found for subset={subset!r} split={split!r}"],
        )
    samples: list[dict[str, Any]] = []
    shard_path = _download_dataset_file(dataset_id, shard_files[0])
    with h5py.File(shard_path, "r") as handle:
        fields = sorted(handle.keys())
        for required_key in ("images", "annotations"):
            if required_key not in handle:
                return HfDatasetInspection(
                    dataset_id,
                    split,
                    subset,
                    "hdf5 line_bboxes",
                    fields,
                    None,
                    None,
                    0,
                    [],
                    [f"HDF5 shard {shard_path} is missing dataset {required_key!r}"],
                )
        image_dataset = handle["images"]
        annotation_dataset = handle["annotations"]
        page_count = min(len(image_dataset), len(annotation_dataset))
        for page_index in range(page_count):
            if len(samples) >= limit:
                break
            annotation = _coerce_annotation_json(annotation_dataset[page_index], shard_path, page_index)
            line_boxes = annotation.get("line_bboxes")
            if not isinstance(line_boxes, list):
                samples.append(
                    {
                        "row_index": page_index,
                        "shard_file": shard_files[0],
                        "annotation_keys": sorted(annotation),
                        "line_bbox_count": None,
                        "image_supported_by_prepare": False,
                        "text_length": 0,
                        "text_preview": "",
                    }
                )
                continue
            for line_index, line_box in enumerate(line_boxes):
                if len(samples) >= limit:
                    break
                text = line_box.get("text") if isinstance(line_box, dict) else None
                samples.append(
                    {
                        "row_index": len(samples),
                        "shard_file": shard_files[0],
                        "page_index": page_index,
                        "line_index": line_index,
                        "annotation_keys": sorted(annotation),
                        "line_bbox_count": len(line_boxes),
                        "bbox": line_box.get("bbox") if isinstance(line_box, dict) else None,
                        "image_type": "hdf5_page_crop",
                        "image_supported_by_prepare": isinstance(line_box, dict) and isinstance(line_box.get("bbox"), list),
                        "text_type": type(text).__name__,
                        "text_length": len(str(text or "")),
                        "text_preview": _preview_text(text),
                    }
                )
    return HfDatasetInspection(
        dataset_id=dataset_id,
        split=split,
        subset=subset,
        inspection_mode="hdf5 line_bboxes",
        fields=["images", "annotations"],
        inferred_image_field="images + annotations.line_bboxes[].bbox",
        inferred_text_field="annotations.line_bboxes[].text",
        inspected_count=len(samples),
        samples=samples,
        issues=[] if samples else [f"No line boxes found in first shard: {shard_files[0]}"],
        warnings=[f"inspected first shard only: {shard_files[0]}"],
    )


def _try_infer_field(
    row: dict[str, Any],
    explicit: str | None,
    candidates: tuple[str, ...],
    role: str,
    issues: list[str],
) -> str | None:
    try:
        return infer_field(row, explicit, candidates, role)
    except DataValidationError as exc:
        issues.append(str(exc))
        return None


def _summarize_hf_row(
    row: dict[str, Any],
    *,
    row_index: int,
    image_field: str | None,
    text_field: str | None,
) -> dict[str, Any]:
    image_value = row.get(image_field) if image_field else None
    text_value = row.get(text_field) if text_field else None
    return {
        "row_index": row_index,
        "fields": sorted(row),
        "image_field": image_field,
        "image_type": type(image_value).__name__,
        "image_supported_by_prepare": _is_supported_hf_image_value(image_value),
        "image_summary": _summarize_value(image_value),
        "text_field": text_field,
        "text_type": type(text_value).__name__,
        "text_length": len(str(text_value or "")),
        "text_preview": _preview_text(text_value),
    }


def _is_supported_hf_image_value(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if hasattr(value, "save"):
        return True
    return isinstance(value, dict) and bool(value.get("path"))


def _summarize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
        return text[:160] + ("..." if len(text) > 160 else "")
    if isinstance(value, dict):
        return {str(key): _summarize_value(item) for key, item in list(value.items())[:10]}
    if hasattr(value, "size"):
        return {"type": type(value).__name__, "size": getattr(value, "size", None), "mode": getattr(value, "mode", None)}
    return {"type": type(value).__name__}


def _preview_text(value: Any, limit: int = 120) -> str:
    text = normalize_ocr_text(str(value or ""))
    return text[:limit] + ("..." if len(text) > limit else "")


def _write_hf_inspection(report: HfDatasetInspection, output_dir: Path) -> HfDatasetInspection:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report.dataset_id.replace("/", "__")
    if report.subset:
        stem = f"{stem}__{report.subset}"
    stem = f"{stem}-{report.split}-inspect"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    report.output_json_path = str(json_path)
    report.output_md_path = str(md_path)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Hugging Face Dataset Inspection",
        "",
        f"Dataset: `{report.dataset_id}`",
        f"Split: `{report.split}`",
        f"Subset: `{report.subset or ''}`",
        f"Mode: `{report.inspection_mode}`",
        f"Passed: `{report.passed}`",
        f"Fields: `{', '.join(report.fields)}`",
        f"Inferred image field: `{report.inferred_image_field or ''}`",
        f"Inferred text field: `{report.inferred_text_field or ''}`",
        f"Inspected samples: `{report.inspected_count}`",
        "",
        "## Issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in report.issues] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in report.warnings] or ["- none"])
    lines.extend(["", "## Samples", ""])
    for sample in report.samples:
        lines.append(f"- row `{sample.get('row_index')}`: text_len=`{sample.get('text_length')}` preview=`{sample.get('text_preview')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _prepare_prashant_synthetic_dataset(
    dataset_id: str,
    output_dir: str | Path,
    *,
    dataset: str | None,
    split: str,
    limit: int | None,
    strict_chars: bool,
    skip_invalid: bool,
    slices: list[str] | None,
) -> PrepareReport:
    prefix = f"{split}/"
    files = set(_list_dataset_files(dataset_id, prefix=prefix, suffix=""))
    metadata_file = f"{prefix}metadata.jsonl"
    if metadata_file not in files:
        raise DataValidationError(f"metadata.jsonl is missing for {dataset_id} split {split}: {metadata_file}")
    metadata_path = _download_dataset_file(dataset_id, metadata_file)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(metadata_path)):
        if limit is not None and index >= limit:
            break
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise DataValidationError(f"HF row {index} in {metadata_file} requires a non-empty file_name")
        dataset_file = file_name if file_name.startswith(prefix) else f"{prefix}{file_name}"
        if dataset_file not in files:
            raise DataValidationError(f"HF row {index} image file is missing from dataset repo: {dataset_file}")
        image_path = _download_dataset_file(dataset_id, dataset_file)
        rows.append(
            {
                "image_path": str(image_path),
                "text": row.get("text"),
                "id": row.get("id", index),
            }
        )
    dataset_name = dataset or dataset_id.replace("/", "__")
    entries, rejects = convert_rows_to_manifest(
        rows,
        dataset=dataset_name,
        split=split,
        base_dir=None,
        image_field="image_path",
        text_field="text",
        limit=limit,
        strict_chars=strict_chars,
        skip_invalid=skip_invalid,
    )
    entries = _stamp_slices(entries, _default_hf_slices(dataset_id, subset=None) + (slices or []))
    return _finalize_prepare_report(output_dir, dataset_name, split, entries, rejects)


def _prepare_nvidia_hdf5_dataset(
    dataset_id: str,
    output_dir: str | Path,
    *,
    dataset: str | None,
    split: str,
    subset: str,
    limit: int | None,
    strict_chars: bool,
    slices: list[str] | None,
) -> PrepareReport:
    try:
        import h5py
    except ImportError as exc:
        raise DataValidationError("NVIDIA HDF5 dataset support requires installing ocr-tech[datasets].") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataValidationError("NVIDIA HDF5 dataset support requires Pillow. Install ocr-tech[datasets].") from exc

    shard_files = _list_dataset_files(dataset_id, prefix=f"{subset}/{split}/", suffix=".h5")
    if not shard_files:
        raise DataValidationError(f"No HDF5 shards found for {dataset_id} subset {subset!r} split {split!r}")
    dataset_name = dataset or f"{dataset_id.replace('/', '__')}__{subset}"
    output_root = Path(output_dir)
    image_dir = output_root / "images" / dataset_name / split
    image_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []

    for shard_index, shard_file in enumerate(shard_files):
        shard_path = _download_dataset_file(dataset_id, shard_file)
        with h5py.File(shard_path, "r") as handle:
            for required_key in ("images", "annotations"):
                if required_key not in handle:
                    raise DataValidationError(f"HDF5 shard {shard_path} is missing dataset {required_key!r}")
            image_dataset = handle["images"]
            annotation_dataset = handle["annotations"]
            if len(image_dataset) != len(annotation_dataset):
                raise DataValidationError(f"HDF5 shard {shard_path} has mismatched images and annotations lengths")
            for page_index in range(len(image_dataset)):
                if limit is not None and len(entries) >= limit:
                    break
                image_bytes = _coerce_hdf5_bytes(image_dataset[page_index])
                annotation = _coerce_annotation_json(annotation_dataset[page_index], shard_path, page_index)
                line_boxes = annotation.get("line_bboxes")
                if not isinstance(line_boxes, list):
                    raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} annotation is missing line_bboxes")
                page_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                for line_index, line_box in enumerate(line_boxes):
                    if limit is not None and len(entries) >= limit:
                        break
                    if not isinstance(line_box, dict):
                        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} line {line_index} is not an object")
                    text = normalize_ocr_text(str(line_box.get("text") or ""))
                    if not text:
                        continue
                    if strict_chars:
                        unsupported = unsupported_characters(text)
                        if unsupported:
                            raise DataValidationError(
                                f"HDF5 shard {shard_path} page {page_index} line {line_index} contains unsupported characters: {unsupported}"
                            )
                    bbox = _coerce_bbox(line_box.get("bbox"), shard_path, page_index, line_index)
                    crop = page_image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                    if crop.width <= 0 or crop.height <= 0:
                        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} line {line_index} produced an empty crop")
                    file_name = f"{shard_path.stem}-p{page_index:06d}-l{line_index:04d}.png"
                    crop_path = image_dir / file_name
                    crop.save(crop_path)
                    image_sha256 = sha256_file(crop_path)
                    text_sha256 = sha256_text(text)
                    sample_id = f"{dataset_name}-{split}-{shard_index:03d}-{page_index:06d}-{line_index:04d}"
                    entries.append(
                        ManifestEntry(
                            sample_id=sample_id,
                            dataset=dataset_name,
                            split=split,
                            image_path=str(crop_path),
                            text=text,
                            sha256=image_sha256,
                            metadata={
                                "source_dataset": dataset_id,
                                "subset": subset,
                                "shard_path": str(shard_path),
                                "page_index": page_index,
                                "line_index": line_index,
                                "bbox": bbox,
                                "text_sha256": text_sha256,
                                "sample_sha256": sha256_text(f"{image_sha256}\n{text_sha256}"),
                                "slices": ["english", subset, "synthetic", "line_crop"],
                            },
                        )
                    )
        if limit is not None and len(entries) >= limit:
            break
    if not entries:
        raise DataValidationError(f"No line crops were prepared for {dataset_id} subset {subset!r} split {split!r}")
    entries = _stamp_slices(entries, slices or [])
    return _finalize_prepare_report(output_root, dataset_name, split, entries, [])


def _stamp_slices(entries: list[ManifestEntry], slices: list[str]) -> list[ManifestEntry]:
    normalized = [str(item).strip() for item in slices if str(item).strip()]
    if not normalized:
        return entries
    for entry in entries:
        metadata = dict(entry.metadata or {})
        existing = metadata.get("slices")
        combined: list[str] = []
        if isinstance(existing, list):
            combined.extend(str(item).strip() for item in existing if str(item).strip())
        elif isinstance(existing, str) and existing.strip():
            combined.append(existing.strip())
        combined.extend(normalized)
        metadata["slices"] = sorted(set(combined))
        entry.metadata = metadata
    return entries


def _default_hf_slices(dataset_id: str, *, subset: str | None) -> list[str]:
    if dataset_id == "gauravgiri/nepali-ocr-dataset":
        return ["nepali", "real"]
    if dataset_id == "prashant0919/nepali-synthetic-ocr-lines":
        return ["nepali", "synthetic"]
    if dataset_id == "nvidia/OCR-Synthetic-Multilingual-v1":
        values = ["english", "synthetic", "line_crop"]
        if subset:
            values.append(subset)
        return values
    return []


def _finalize_prepare_report(
    output_dir: str | Path,
    dataset_name: str,
    split: str,
    entries: list[ManifestEntry],
    rejects: list[PrepareReject],
) -> PrepareReport:
    if not entries:
        if rejects:
            raise DataValidationError(f"No rows converted into manifest entries; rejected rows={len(rejects)}")
        raise DataValidationError("No rows converted into manifest entries")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}-{split}"
    manifest_path = output_root / f"{stem}.jsonl"
    rejects_path = output_root / f"{stem}-rejects.jsonl"
    summary_json_path = output_root / f"{stem}-prepare.json"
    summary_md_path = output_root / f"{stem}-prepare.md"
    write_manifest(entries, manifest_path)
    if rejects:
        with rejects_path.open("w", encoding="utf-8") as handle:
            for reject in rejects:
                handle.write(json.dumps(reject.to_dict(), ensure_ascii=False) + "\n")
    elif rejects_path.exists():
        rejects_path.unlink()
    report = PrepareReport(
        manifest_path=str(manifest_path),
        sample_count=len(entries),
        rejected_count=len(rejects),
        rejects_path=str(rejects_path) if rejects else None,
        summary_json_path=str(summary_json_path),
        summary_md_path=str(summary_md_path),
    )
    _write_prepare_report(report, summary_json_path, summary_md_path)
    return report


def _write_prepare_report(report: PrepareReport, summary_json_path: Path, summary_md_path: Path) -> None:
    summary_json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Dataset Prepare",
        "",
        f"Manifest: `{report.manifest_path}`",
        f"Prepared samples: `{report.sample_count}`",
        f"Rejected rows: `{report.rejected_count}`",
        f"Reject log: `{report.rejects_path or ''}`",
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_dataset(dataset_id: str, *, allow_patterns: list[str]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DataValidationError("Snapshot download support requires installing ocr-tech[datasets].") from exc
    try:
        local_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset", allow_patterns=allow_patterns)
    except Exception as exc:
        raise DataValidationError(f"Failed to snapshot dataset {dataset_id!r}: {exc}") from exc
    return Path(local_dir)


def _list_dataset_files(dataset_id: str, *, prefix: str, suffix: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise DataValidationError("Dataset file listing requires installing ocr-tech[datasets].") from exc
    try:
        files = list_repo_files(repo_id=dataset_id, repo_type="dataset")
    except Exception as exc:
        raise DataValidationError(f"Failed to list dataset files for {dataset_id!r}: {exc}") from exc
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(suffix))


def _download_dataset_file(dataset_id: str, dataset_file: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DataValidationError("Dataset file download requires installing ocr-tech[datasets].") from exc
    try:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=dataset_file)
    except Exception as exc:
        raise DataValidationError(f"Failed to download dataset file {dataset_file!r} from {dataset_id!r}: {exc}") from exc
    return Path(local_path)


def _coerce_hdf5_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "tobytes"):
        return value.tobytes()
    raise DataValidationError(f"Unsupported HDF5 image payload type: {type(value).__name__}")


def _coerce_annotation_json(value: Any, shard_path: Path, page_index: int) -> dict[str, Any]:
    if isinstance(value, bytes):
        raw = value.decode("utf-8")
    elif isinstance(value, str):
        raw = value
    elif hasattr(value, "tobytes"):
        raw = value.tobytes().decode("utf-8")
    else:
        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} annotation has unsupported type: {type(value).__name__}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} annotation is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} annotation must be a JSON object")
    return payload


def _coerce_bbox(value: Any, shard_path: Path, page_index: int, line_index: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} line {line_index} bbox must be [x, y, w, h]")
    bbox = [int(item) for item in value]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise DataValidationError(f"HDF5 shard {shard_path} page {page_index} line {line_index} bbox width/height must be positive")
    return bbox
