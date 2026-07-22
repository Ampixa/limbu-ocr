"""Capture preparation helpers for operational OCR runs."""

from __future__ import annotations

import json
import math
import platform
import shutil
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import DataValidationError, EngineUnavailableError
from .manifest import sha256_file

_SUPPORTED_CAPTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PerspectiveQuad = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(slots=True)
class CapturePrepResult:
    source_path: str
    raw_copy_path: str
    prepared_image_path: str
    metadata_path: str
    source_sha256: str
    prepared_sha256: str
    operations: list[dict[str, Any]]
    image_quality: dict[str, Any] | None = None
    created_at_utc: str | None = None
    run_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "raw_copy_path": self.raw_copy_path,
            "prepared_image_path": self.prepared_image_path,
            "metadata_path": self.metadata_path,
            "source_sha256": self.source_sha256,
            "prepared_sha256": self.prepared_sha256,
            "operations": self.operations,
            "image_quality": self.image_quality or {},
            "created_at_utc": self.created_at_utc,
            "run_context": self.run_context or {},
        }


def prepare_limbu_capture(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    rotate_degrees: float = 0.0,
    crop_box: tuple[int, int, int, int] | None = None,
    perspective_quad: PerspectiveQuad | None = None,
    auto_detect_page: bool = False,
    auto_deskew: bool = False,
    max_auto_deskew_degrees: float = 8.0,
    autocontrast: bool = True,
    grayscale: bool = True,
    run_context: dict[str, Any] | None = None,
) -> CapturePrepResult:
    source = Path(input_path)
    if not source.is_file():
        raise DataValidationError(f"capture input does not exist or is not a file: {source}")
    if source.suffix.lower() not in _SUPPORTED_CAPTURE_SUFFIXES:
        raise DataValidationError(f"unsupported capture image suffix: {source.suffix or '<none>'}")
    if auto_detect_page and perspective_quad is not None:
        raise DataValidationError("auto_detect_page cannot be combined with an explicit perspective_quad")
    if max_auto_deskew_degrees <= 0 or max_auto_deskew_degrees > 45:
        raise DataValidationError(f"max_auto_deskew_degrees must be > 0 and <= 45, got {max_auto_deskew_degrees}")

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise EngineUnavailableError("Pillow is required for capture preparation") from exc

    out = Path(output_dir)
    raw_dir = out / "raw"
    prepared_dir = out / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_copy = raw_dir / source.name
    shutil.copy2(source, raw_copy)
    operations: list[dict[str, Any]] = [{"operation": "raw_copy", "path": str(raw_copy)}]

    try:
        with Image.open(source) as image:
            prepared = ImageOps.exif_transpose(image)
            operations.append({"operation": "exif_transpose", "mode": prepared.mode, "size": list(prepared.size)})
            if crop_box is not None:
                _validate_crop_box(crop_box, prepared.size)
                prepared = prepared.crop(crop_box)
                operations.append({"operation": "crop", "box": list(crop_box), "size": list(prepared.size)})
            if auto_detect_page:
                detected_quad, area_ratio, detection_method = _detect_page_quad(prepared)
                perspective_quad = detected_quad
                operations.append(
                    {
                        "operation": "auto_detect_page",
                        "method": detection_method,
                        "quad_order": "top_left,top_right,bottom_right,bottom_left",
                        "quad": [[x, y] for x, y in detected_quad],
                        "area_ratio": area_ratio,
                        "touches_image_border": _quad_touches_image_border(detected_quad, prepared.size),
                    }
                )
            if perspective_quad is not None:
                _validate_perspective_quad(perspective_quad, prepared.size)
                prepared, rectified_size = _rectify_perspective(prepared, perspective_quad)
                operations.append(
                    {
                        "operation": "perspective_rectify",
                        "quad_order": "top_left,top_right,bottom_right,bottom_left",
                        "quad": [[x, y] for x, y in perspective_quad],
                        "size": list(rectified_size),
                    }
                )
            if auto_deskew:
                detected_angle = _detect_skew_degrees(prepared, max_degrees=float(max_auto_deskew_degrees))
                if detected_angle is None:
                    operations.append(
                        {
                            "operation": "auto_deskew",
                            "status": "skipped",
                            "reason": "insufficient near-horizontal line evidence",
                            "max_degrees": float(max_auto_deskew_degrees),
                            "size": list(prepared.size),
                        }
                    )
                else:
                    prepared = prepared.rotate(detected_angle, expand=True, fillcolor="white")
                    operations.append(
                        {
                            "operation": "auto_deskew",
                            "status": "applied",
                            "detected_angle_degrees": detected_angle,
                            "applied_rotation_degrees": detected_angle,
                            "max_degrees": float(max_auto_deskew_degrees),
                            "size": list(prepared.size),
                        }
                    )
            if rotate_degrees:
                prepared = prepared.rotate(float(rotate_degrees), expand=True, fillcolor="white")
                operations.append({"operation": "rotate", "degrees": float(rotate_degrees), "size": list(prepared.size)})
            if grayscale:
                prepared = prepared.convert("L")
                operations.append({"operation": "grayscale", "mode": prepared.mode})
            if autocontrast:
                prepared = ImageOps.autocontrast(prepared)
                operations.append({"operation": "autocontrast"})
            prepared_path = prepared_dir / f"{source.stem}.prepared.png"
            prepared.save(prepared_path)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"failed to prepare capture image {source}: {type(exc).__name__}: {exc}") from exc

    result = CapturePrepResult(
        source_path=str(source),
        raw_copy_path=str(raw_copy),
        prepared_image_path=str(prepared_path),
        metadata_path=str(out / "limbu-capture-prep.json"),
        source_sha256=sha256_file(source),
        prepared_sha256=sha256_file(prepared_path),
        operations=operations,
        image_quality={
            "source": _capture_image_quality_metrics(source),
            "prepared": _capture_image_quality_metrics(prepared_path),
        },
        created_at_utc=datetime.now(UTC).isoformat(),
        run_context=_default_run_context(run_context),
    )
    Path(result.metadata_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def audit_limbu_capture(
    capture_metadata: str | Path,
    output_dir: str | Path | None = None,
    *,
    require_metadata_path_self: bool = False,
    min_prepared_width: int | None = None,
    min_prepared_height: int | None = None,
    min_prepared_entropy: float | None = None,
    min_prepared_luminance_stddev: float | None = None,
    min_prepared_edge_stddev: float | None = None,
    min_prepared_laplacian_variance: float | None = None,
    require_auto_detect_page: bool = False,
    min_auto_detect_page_area_ratio: float | None = None,
    fail_on_auto_detect_page_border_touch: bool = False,
    require_auto_deskew_applied: bool = False,
    max_abs_auto_deskew_degrees: float | None = None,
) -> dict[str, Any]:
    metadata_path = _resolve_capture_metadata_path(Path(capture_metadata))
    issues: list[str] = []
    warnings: list[str] = []
    quality_policy = {
        "require_metadata_path_self": require_metadata_path_self,
        "min_prepared_width": min_prepared_width,
        "min_prepared_height": min_prepared_height,
        "min_prepared_entropy": min_prepared_entropy,
        "min_prepared_luminance_stddev": min_prepared_luminance_stddev,
        "min_prepared_edge_stddev": min_prepared_edge_stddev,
        "min_prepared_laplacian_variance": min_prepared_laplacian_variance,
        "require_auto_detect_page": require_auto_detect_page,
        "min_auto_detect_page_area_ratio": min_auto_detect_page_area_ratio,
        "fail_on_auto_detect_page_border_touch": fail_on_auto_detect_page_border_touch,
        "require_auto_deskew_applied": require_auto_deskew_applied,
        "max_abs_auto_deskew_degrees": max_abs_auto_deskew_degrees,
    }
    _validate_capture_quality_policy(quality_policy)
    metadata = _read_capture_metadata(metadata_path, issues)
    artifacts: list[dict[str, Any]] = []
    if metadata is not None:
        _audit_capture_metadata_shape(
            metadata,
            metadata_path,
            issues,
            warnings,
            require_metadata_path_self=require_metadata_path_self,
        )
        artifacts.extend(_audit_capture_artifacts(metadata, metadata_path, issues, warnings))
        _audit_capture_image_quality(metadata, metadata_path, issues, warnings)
        _audit_capture_quality_policy(metadata, quality_policy, issues)
        _audit_capture_operations(metadata, quality_policy, issues, warnings)
    report = {
        "pipeline_id": "limbu-first-ocr-pipeline-v1",
        "audit_stage": "limbu_capture_prep_bundle",
        "metadata_path": str(metadata_path),
        "quality_policy": quality_policy,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "artifacts": artifacts,
    }
    target_dir = Path(output_dir) if output_dir is not None else metadata_path.parent
    _write_limbu_capture_audit(report, target_dir)
    return report


def _default_run_context(run_context: dict[str, Any] | None) -> dict[str, Any]:
    context = dict(run_context or {})
    context.setdefault("hostname", socket.gethostname())
    context.setdefault("platform", platform.platform())
    context.setdefault("python_version", platform.python_version())
    return context


def _resolve_capture_metadata_path(path: Path) -> Path:
    if path.is_dir():
        return path / "limbu-capture-prep.json"
    return path


def _read_capture_metadata(path: Path, issues: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(f"limbu-capture-prep.json is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"limbu-capture-prep.json is invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        issues.append("limbu-capture-prep.json must contain a JSON object")
        return None
    return payload


def _audit_capture_metadata_shape(
    metadata: dict[str, Any],
    metadata_path: Path,
    issues: list[str],
    warnings: list[str],
    *,
    require_metadata_path_self: bool = False,
) -> None:
    required_strings = [
        "source_path",
        "raw_copy_path",
        "prepared_image_path",
        "metadata_path",
        "source_sha256",
        "prepared_sha256",
        "created_at_utc",
    ]
    for key in required_strings:
        if not isinstance(metadata.get(key), str) or not str(metadata.get(key)).strip():
            issues.append(f"limbu-capture-prep.json missing non-empty string field: {key}")
    operations = metadata.get("operations")
    if not isinstance(operations, list) or not operations:
        issues.append("limbu-capture-prep.json operations must be a non-empty list")
    image_quality = metadata.get("image_quality")
    if image_quality is None:
        warnings.append("limbu-capture-prep.json missing image_quality metrics")
    elif not isinstance(image_quality, dict):
        issues.append("limbu-capture-prep.json image_quality must be an object")
    run_context = metadata.get("run_context")
    if not isinstance(run_context, dict):
        issues.append("limbu-capture-prep.json run_context must be an object")
    else:
        if not run_context.get("hostname"):
            warnings.append("capture run_context.hostname is missing")
        if not run_context.get("python_version"):
            warnings.append("capture run_context.python_version is missing")
    recorded_metadata_path = metadata.get("metadata_path")
    if isinstance(recorded_metadata_path, str) and recorded_metadata_path.strip():
        resolved = _resolve_capture_artifact_path(recorded_metadata_path, metadata_path.parent)
        try:
            if resolved.resolve() != metadata_path.resolve():
                message = f"metadata_path points to a different path: {recorded_metadata_path}"
                if require_metadata_path_self:
                    issues.append(message)
                else:
                    warnings.append(message)
        except OSError:
            message = f"metadata_path could not be resolved: {recorded_metadata_path}"
            if require_metadata_path_self:
                issues.append(message)
            else:
                warnings.append(message)


def _audit_capture_artifacts(
    metadata: dict[str, Any],
    metadata_path: Path,
    issues: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    artifacts.append(
        _audit_capture_artifact(
            metadata.get("source_path"),
            metadata.get("source_sha256"),
            "source_path",
            metadata_path.parent,
            issues,
            warnings,
            required=False,
        )
    )
    artifacts.append(
        _audit_capture_artifact(
            metadata.get("raw_copy_path"),
            metadata.get("source_sha256"),
            "raw_copy_path",
            metadata_path.parent,
            issues,
            warnings,
            required=True,
        )
    )
    artifacts.append(
        _audit_capture_artifact(
            metadata.get("prepared_image_path"),
            metadata.get("prepared_sha256"),
            "prepared_image_path",
            metadata_path.parent,
            issues,
            warnings,
            required=True,
        )
    )
    return artifacts


def _audit_capture_artifact(
    path_value: object,
    expected_sha: object,
    label: str,
    base_dir: Path,
    issues: list[str],
    warnings: list[str],
    *,
    required: bool,
) -> dict[str, Any]:
    path_text = str(path_value or "")
    path = _resolve_capture_artifact_path(path_text, base_dir) if path_text else Path("")
    exists = bool(path_text) and path.is_file()
    artifact = {
        "label": label,
        "path": path_text,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "expected_sha256": expected_sha if isinstance(expected_sha, str) else None,
        "size_bytes": path.stat().st_size if exists else None,
    }
    if not path_text:
        issues.append(f"{label} is missing from limbu-capture-prep.json")
        return artifact
    if not exists:
        message = f"{label} does not exist: {path_text}"
        if required:
            issues.append(message)
        else:
            warnings.append(message)
        return artifact
    if not isinstance(expected_sha, str) or not expected_sha:
        issues.append(f"{label} expected SHA-256 is missing")
        return artifact
    actual_sha = str(artifact["sha256"])
    if actual_sha != expected_sha:
        issues.append(f"{label} sha256 mismatch: metadata={expected_sha} actual={actual_sha}")
    _audit_capture_image_decodable(path, label, artifact, issues, warnings)
    return artifact


def _audit_capture_image_quality(
    metadata: dict[str, Any],
    metadata_path: Path,
    issues: list[str],
    warnings: list[str],
) -> None:
    image_quality = metadata.get("image_quality")
    if not isinstance(image_quality, dict):
        return
    for quality_label, path_field, required in (
        ("source", "source_path", False),
        ("prepared", "prepared_image_path", True),
    ):
        recorded = image_quality.get(quality_label)
        if recorded is None:
            warnings.append(f"limbu-capture-prep.json image_quality missing {quality_label} metrics")
            continue
        if not isinstance(recorded, dict):
            issues.append(f"limbu-capture-prep.json image_quality.{quality_label} must be an object")
            continue
        path_value = metadata.get(path_field)
        path_text = str(path_value or "")
        path = _resolve_capture_artifact_path(path_text, metadata_path.parent) if path_text else Path("")
        if not path_text or not path.is_file():
            message = f"image_quality.{quality_label} metrics cannot be replayed because {path_field} is unavailable: {path_text}"
            if required:
                issues.append(message)
            else:
                warnings.append(message)
            continue
        try:
            actual = _capture_image_quality_metrics(path)
        except Exception as exc:
            issues.append(f"image_quality.{quality_label} metrics replay failed: {type(exc).__name__}: {exc}")
            continue
        comparable_actual = {key: actual.get(key) for key in recorded}
        if recorded != comparable_actual:
            issues.append(f"image_quality.{quality_label} metrics mismatch: metadata={recorded!r} actual={actual!r}")
        missing_recorded_metrics = sorted(set(actual) - set(recorded))
        if missing_recorded_metrics:
            warnings.append(
                f"image_quality.{quality_label} metrics missing newer replay fields: {', '.join(missing_recorded_metrics)}"
            )


def _validate_capture_quality_policy(policy: dict[str, int | float | bool | None]) -> None:
    bool_keys = {
        "require_metadata_path_self",
        "require_auto_detect_page",
        "fail_on_auto_detect_page_border_touch",
        "require_auto_deskew_applied",
    }
    for key, value in policy.items():
        if value is None:
            continue
        if key in bool_keys:
            if not isinstance(value, bool):
                raise DataValidationError(f"{key} must be boolean, got {value!r}")
            continue
        if not isinstance(value, int | float):
            raise DataValidationError(f"{key} must be numeric, got {value!r}")
        if value < 0:
            raise DataValidationError(f"{key} must be non-negative, got {value!r}")


def _audit_capture_quality_policy(
    metadata: dict[str, Any],
    policy: dict[str, int | float | bool | None],
    issues: list[str],
) -> None:
    checks = {
        "min_prepared_width": "width",
        "min_prepared_height": "height",
        "min_prepared_entropy": "entropy",
        "min_prepared_luminance_stddev": "luminance_stddev",
        "min_prepared_edge_stddev": "edge_stddev",
        "min_prepared_laplacian_variance": "laplacian_variance",
    }
    if not any(policy.get(policy_key) is not None for policy_key in checks):
        return
    image_quality = metadata.get("image_quality")
    if not isinstance(image_quality, dict):
        issues.append("capture quality policy requires image_quality metrics")
        return
    prepared = image_quality.get("prepared")
    if not isinstance(prepared, dict):
        issues.append("capture quality policy requires image_quality.prepared metrics")
        return
    for policy_key, metric_key in checks.items():
        threshold = policy.get(policy_key)
        if threshold is None:
            continue
        actual = prepared.get(metric_key)
        if not isinstance(actual, int | float):
            issues.append(f"capture quality policy requires numeric image_quality.prepared.{metric_key}")
        elif float(actual) < float(threshold):
            issues.append(
                "prepared image quality below policy threshold: "
                f"{metric_key}={actual!r} minimum={threshold!r}"
            )


def _audit_capture_image_decodable(
    path: Path,
    label: str,
    artifact: dict[str, Any],
    issues: list[str],
    warnings: list[str],
) -> None:
    if path.suffix.lower() not in _SUPPORTED_CAPTURE_SUFFIXES:
        return
    try:
        from PIL import Image
    except ImportError:
        warnings.append(f"{label} image decode check skipped because Pillow is unavailable")
        return
    try:
        with Image.open(path) as image:
            artifact["image_mode"] = image.mode
            artifact["image_size"] = list(image.size)
            image.verify()
    except Exception as exc:
        issues.append(f"{label} is not a decodable image: {type(exc).__name__}: {exc}")


def _capture_image_quality_metrics(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageFilter, ImageOps, ImageStat
    except ImportError as exc:
        raise EngineUnavailableError("Pillow is required for capture quality metrics") from exc
    with Image.open(path) as image:
        image.load()
        grayscale = ImageOps.grayscale(image)
        luminance = ImageStat.Stat(grayscale)
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        mean = float(luminance.mean[0])
        stddev = float(luminance.stddev[0])
        edge_mean = float(edge_stat.mean[0])
        edge_stddev = float(edge_stat.stddev[0])
        return {
            "width": int(image.width),
            "height": int(image.height),
            "mode": image.mode,
            "luminance_mean": round(mean, 6),
            "luminance_stddev": round(stddev, 6),
            "edge_mean": round(edge_mean, 6),
            "edge_stddev": round(edge_stddev, 6),
            "laplacian_variance": _laplacian_variance(grayscale),
            "entropy": round(float(grayscale.entropy()), 6),
        }


def _laplacian_variance(grayscale: Any) -> float | None:
    try:
        import numpy as np
    except ImportError:
        return None
    array = np.asarray(grayscale, dtype=np.float32)
    if array.size == 0 or array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 3:
        return None
    center = array[1:-1, 1:-1] * -4.0
    laplacian = center + array[:-2, 1:-1] + array[2:, 1:-1] + array[1:-1, :-2] + array[1:-1, 2:]
    return round(float(laplacian.var()), 6)


def _resolve_capture_artifact_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _audit_capture_operations(
    metadata: dict[str, Any],
    policy: dict[str, int | float | bool | None],
    issues: list[str],
    warnings: list[str],
) -> None:
    operations = metadata.get("operations")
    if not isinstance(operations, list) or not operations:
        return
    operation_names: list[str] = []
    auto_detect_operations: list[dict[str, Any]] = []
    auto_deskew_operations: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            issues.append(f"capture operation {index} must be an object")
            continue
        name = operation.get("operation")
        if not isinstance(name, str) or not name:
            issues.append(f"capture operation {index} missing operation name")
            continue
        operation_names.append(name)
        if name == "auto_deskew":
            auto_deskew_operations.append(operation)
            _audit_auto_deskew_operation(operation, index, issues)
        elif name == "auto_detect_page":
            auto_detect_operations.append(operation)
            if "quad" not in operation or "area_ratio" not in operation:
                issues.append(f"capture operation {index} auto_detect_page missing quad or area_ratio")
            if operation.get("touches_image_border") is True:
                message = f"capture operation {index} auto_detect_page touches image border; source may be clipped or include background"
                if policy.get("fail_on_auto_detect_page_border_touch") is True:
                    issues.append(message)
                else:
                    warnings.append(message)
            threshold = policy.get("min_auto_detect_page_area_ratio")
            if threshold is not None:
                area_ratio = operation.get("area_ratio")
                if not isinstance(area_ratio, int | float):
                    issues.append(f"capture operation {index} auto_detect_page missing numeric area_ratio")
                elif float(area_ratio) < float(threshold):
                    issues.append(
                        "capture operation "
                        f"{index} auto_detect_page area_ratio below policy threshold: "
                        f"area_ratio={area_ratio!r} minimum={threshold!r}"
                    )
        elif name == "perspective_rectify":
            if "quad" not in operation or "size" not in operation:
                issues.append(f"capture operation {index} perspective_rectify missing quad or size")
    if operation_names and operation_names[0] != "raw_copy":
        issues.append(f"first capture operation must be raw_copy, got {operation_names[0]!r}")
    if "exif_transpose" not in operation_names:
        warnings.append("capture operations do not include exif_transpose")
    if "auto_detect_page" in operation_names and "perspective_rectify" not in operation_names:
        issues.append("auto_detect_page operation must be followed by perspective_rectify")
    if policy.get("require_auto_detect_page") is True and not auto_detect_operations:
        issues.append("capture operation policy requires auto_detect_page")
    if policy.get("min_auto_detect_page_area_ratio") is not None and not auto_detect_operations:
        issues.append("capture operation policy requires auto_detect_page for min_auto_detect_page_area_ratio")
    if policy.get("require_auto_deskew_applied") is True:
        if not auto_deskew_operations:
            issues.append("capture operation policy requires auto_deskew")
        elif not any(operation.get("status") == "applied" for operation in auto_deskew_operations):
            issues.append("capture operation policy requires applied auto_deskew")
    max_abs_angle = policy.get("max_abs_auto_deskew_degrees")
    if max_abs_angle is not None:
        if not auto_deskew_operations:
            issues.append("capture operation policy requires auto_deskew for max_abs_auto_deskew_degrees")
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict) or operation.get("operation") != "auto_deskew" or operation.get("status") != "applied":
                continue
            angle = operation.get("applied_rotation_degrees")
            if not isinstance(angle, int | float):
                issues.append(f"capture operation {index} auto_deskew missing numeric applied_rotation_degrees")
            elif abs(float(angle)) > float(max_abs_angle):
                issues.append(
                    "capture operation "
                    f"{index} auto_deskew angle above policy threshold: "
                    f"applied_rotation_degrees={angle!r} maximum_abs={max_abs_angle!r}"
                )


def _audit_auto_deskew_operation(operation: dict[str, Any], index: int, issues: list[str]) -> None:
    status = operation.get("status")
    if status not in {"applied", "skipped"}:
        issues.append(f"capture operation {index} auto_deskew status must be applied or skipped")
        return
    if status == "applied":
        angle = operation.get("applied_rotation_degrees")
        max_degrees = operation.get("max_degrees")
        if not isinstance(angle, int | float):
            issues.append(f"capture operation {index} auto_deskew missing numeric applied_rotation_degrees")
        if not isinstance(max_degrees, int | float):
            issues.append(f"capture operation {index} auto_deskew missing numeric max_degrees")
        elif isinstance(angle, int | float) and abs(float(angle)) > float(max_degrees):
            issues.append(f"capture operation {index} auto_deskew angle exceeds max_degrees")
    elif not operation.get("reason"):
        issues.append(f"capture operation {index} skipped auto_deskew requires reason")


def _write_limbu_capture_audit(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "limbu-capture-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Limbu Capture Audit",
        "",
        f"Metadata: `{report.get('metadata_path')}`",
        f"Passed: `{'yes' if report.get('passed') else 'no'}`",
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
    (output_dir / "limbu-capture-audit.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_crop_box(crop_box: tuple[int, int, int, int], image_size: tuple[int, int]) -> None:
    left, top, right, bottom = crop_box
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise DataValidationError(f"crop box {list(crop_box)} is outside image bounds {list(image_size)}")
    if right <= left or bottom <= top:
        raise DataValidationError(f"crop box must have positive width and height: {list(crop_box)}")


def _validate_perspective_quad(quad: PerspectiveQuad, image_size: tuple[int, int]) -> None:
    width, height = image_size
    unique_points = set(quad)
    if len(unique_points) != 4:
        raise DataValidationError(f"perspective quad must contain four unique points: {quad!r}")
    for x, y in quad:
        if x < 0 or y < 0 or x >= width or y >= height:
            raise DataValidationError(f"perspective quad point {[x, y]} is outside image bounds {list(image_size)}")
    output_width, output_height = _perspective_output_size(quad)
    if output_width < 8 or output_height < 8:
        raise DataValidationError(
            f"perspective quad output is too small for OCR: width={output_width} height={output_height}"
        )


def _rectify_perspective(image: Any, quad: PerspectiveQuad) -> tuple[Any, tuple[int, int]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise EngineUnavailableError("OpenCV and NumPy are required for perspective rectification") from exc

    output_width, output_height = _perspective_output_size(quad)
    rgb_image = image.convert("RGB")
    source_points = np.array(quad, dtype="float32")
    target_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    warped = cv2.warpPerspective(
        np.array(rgb_image),
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    from PIL import Image

    return Image.fromarray(warped), (output_width, output_height)


def _detect_page_quad(image: Any) -> tuple[PerspectiveQuad, float, str]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise EngineUnavailableError("OpenCV and NumPy are required for automatic page detection") from exc

    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = max(1.0, float(rgb_image.width * rgb_image.height))
    candidates: list[tuple[float, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area / image_area < 0.05:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidates.append((area, approx.reshape(4, 2)))
    if not candidates:
        return _detect_light_page_quad(rgb_image, cv2=cv2, np=np)
    area, points = max(candidates, key=lambda item: item[0])
    ordered = _order_quad_points(points, image_size=rgb_image.size)
    _validate_perspective_quad(ordered, rgb_image.size)
    return ordered, area / image_area, "edge_contour_quad"


def _detect_light_page_quad(image: Any, *, cv2: Any, np: Any) -> tuple[PerspectiveQuad, float, str]:
    """Fallback for phone captures where the paper edge is not a closed Canny contour.

    The target capture domain often has a white textbook/newspaper page on a
    laptop or table, with one or more sheet edges cut off by the photo boundary.
    In that case the stricter four-corner contour detector correctly refuses the
    image, but the runner still needs a bounded crop/rectification candidate and
    an explicit warning. This fallback finds the dominant bright, low-saturation
    region and uses its minimum-area rectangle as a conservative page candidate.
    """

    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    bright_threshold = max(110, int(np.percentile(gray, 55)))
    mask = np.where((gray >= bright_threshold) & (saturation <= 90), 255, 0).astype("uint8")
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = max(1.0, float(image.width * image.height))
    candidates: list[tuple[float, float, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < 0.08:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_area = max(1.0, float(rect_width * rect_height))
        rectangularity = min(1.0, area / rect_area)
        if rectangularity < 0.45:
            continue
        candidates.append((area_ratio, rectangularity, rect))
    if not candidates:
        raise DataValidationError("automatic page detection could not find a four-corner page contour")
    area_ratio, _rectangularity, rect = max(candidates, key=lambda item: (item[0], item[1]))
    points = cv2.boxPoints(rect)
    ordered = _order_quad_points(points, image_size=image.size)
    _validate_perspective_quad(ordered, image.size)
    return ordered, area_ratio, "light_region_min_area_rect"


def _quad_touches_image_border(quad: PerspectiveQuad, image_size: tuple[int, int], *, margin: int = 4) -> bool:
    width, height = image_size
    for x, y in quad:
        if x <= margin or y <= margin or x >= width - 1 - margin or y >= height - 1 - margin:
            return True
    return False


def _detect_skew_degrees(image: Any, *, max_degrees: float) -> float | None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise EngineUnavailableError("OpenCV and NumPy are required for automatic deskew") from exc

    gray = np.array(image.convert("L"))
    if gray.size == 0:
        return None
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    kernel_width = max(12, gray.shape[1] // 30)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, horizontal_kernel)
    edges = cv2.Canny(horizontal, 50, 150)
    min_line_length = max(20, gray.shape[1] // 8)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=max(8, gray.shape[1] // 40),
    )
    angles: list[float] = []
    if lines is not None:
        for raw_line in lines:
            x1, y1, x2, y2 = [int(value) for value in raw_line[0]]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_length:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            while angle <= -90:
                angle += 180
            while angle > 90:
                angle -= 180
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            if abs(angle) <= max_degrees:
                angles.append(angle)
    if len(angles) < 3:
        return _detect_projection_skew_degrees(thresholded, max_degrees=max_degrees, cv2=cv2, np=np)
    median = float(np.median(np.array(angles, dtype="float32")))
    if abs(median) < 0.05:
        return None
    return round(median, 3)


def _detect_projection_skew_degrees(binary_ink: Any, *, max_degrees: float, cv2: Any, np: Any) -> float | None:
    height, width = binary_ink.shape[:2]
    if height <= 0 or width <= 0:
        return None
    ys, xs = np.where(binary_ink > 0)
    ink_pixels = int(len(xs))
    if ink_pixels < max(50, int(width * height * 0.0005)):
        return None
    left = max(0, int(xs.min()) - 8)
    right = min(width, int(xs.max()) + 9)
    top = max(0, int(ys.min()) - 8)
    bottom = min(height, int(ys.max()) + 9)
    cropped = binary_ink[top:bottom, left:right]
    crop_height, crop_width = cropped.shape[:2]
    if crop_height < 20 or crop_width < 20:
        return None

    def score(angle: float) -> float:
        center = (crop_width / 2.0, crop_height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            cropped,
            matrix,
            (crop_width, crop_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_sums = np.count_nonzero(rotated > 0, axis=1).astype("float32")
        if row_sums.size == 0:
            return 0.0
        total = float(row_sums.sum())
        if total <= 0:
            return 0.0
        return float(np.sum(row_sums * row_sums) / total)

    coarse_step = 0.5
    coarse_angles = np.arange(-float(max_degrees), float(max_degrees) + coarse_step / 2.0, coarse_step)
    coarse_scores = [(float(angle), score(float(angle))) for angle in coarse_angles]
    if not coarse_scores:
        return None
    best_angle, best_score = max(coarse_scores, key=lambda item: item[1])
    zero_score = score(0.0)
    if best_score <= 0 or zero_score <= 0 or best_score < zero_score * 1.02:
        return None

    refine_start = max(-float(max_degrees), best_angle - coarse_step)
    refine_stop = min(float(max_degrees), best_angle + coarse_step)
    refine_step = 0.1
    refined_angles = np.arange(refine_start, refine_stop + refine_step / 2.0, refine_step)
    refined_scores = [(float(angle), score(float(angle))) for angle in refined_angles]
    best_angle, best_score = max(refined_scores, key=lambda item: item[1])
    if best_score < zero_score * 1.02 or abs(best_angle) < 0.05:
        return None
    return round(float(best_angle), 3)


def _order_quad_points(points: Any, *, image_size: tuple[int, int]) -> PerspectiveQuad:
    width, height = image_size
    pts = [(float(point[0]), float(point[1])) for point in points]
    by_sum = sorted(pts, key=lambda point: point[0] + point[1])
    top_left = by_sum[0]
    bottom_right = by_sum[-1]
    remaining = by_sum[1:3]
    top_right, bottom_left = sorted(remaining, key=lambda point: point[0] - point[1], reverse=True)

    def clamp(point: tuple[float, float]) -> tuple[int, int]:
        x = min(max(0, int(round(point[0]))), width - 1)
        y = min(max(0, int(round(point[1]))), height - 1)
        return x, y

    return (clamp(top_left), clamp(top_right), clamp(bottom_right), clamp(bottom_left))


def _perspective_output_size(quad: PerspectiveQuad) -> tuple[int, int]:
    top_left, top_right, bottom_right, bottom_left = quad
    width_top = _point_distance(top_left, top_right)
    width_bottom = _point_distance(bottom_left, bottom_right)
    height_left = _point_distance(top_left, bottom_left)
    height_right = _point_distance(top_right, bottom_right)
    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_left, height_right))))
    return output_width, output_height


def _point_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise DataValidationError("crop box must be four comma-separated integers: left,top,right,bottom")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise DataValidationError(f"crop box contains a non-integer value: {value}") from exc


def parse_perspective_quad(value: str | None) -> PerspectiveQuad | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 8:
        raise DataValidationError(
            "perspective quad must be eight comma-separated integers: "
            "top_left_x,top_left_y,top_right_x,top_right_y,bottom_right_x,bottom_right_y,bottom_left_x,bottom_left_y"
        )
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise DataValidationError(f"perspective quad contains a non-integer value: {value}") from exc
    return (
        (numbers[0], numbers[1]),
        (numbers[2], numbers[3]),
        (numbers[4], numbers[5]),
        (numbers[6], numbers[7]),
    )
