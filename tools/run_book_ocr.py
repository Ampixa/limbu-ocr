#!/usr/bin/env python3
"""Run config-driven book OCR for Nepal language-pack entries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from ocrtech.engines import PaddleOcrEngine, SCRIPT_CODEPOINT_RANGES, _script_ratio
from ocrtech.markdown import render_document_markdown
from ocrtech.manifest import sha256_file
from ocrtech.newari_scripts import resolve_newari_writing_system
from ocrtech.schemas import BBox, Block, Document, Page, TextLine


DEFAULT_CONFIG = Path("configs/nepal-language-pack-pipeline-v1.json")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
REQUIRED_PADDLE_ARTIFACT_FILES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
LOGGER_NAME = "run_book_ocr"
# Newar statuses are an execution allowlist. A new status must be reviewed and
# added here explicitly; artifact presence or relocation never promotes it.
NEWARI_EXECUTION_POLICY_BY_STATUS = {
    "operational_model_inference_admitted": "admitted",
    "operational_recovered_model_diagnostic_only": "diagnostic_opt_in_required",
    "operational_recovered_model_diagnostic_failed_fit_only": (
        "known_failed_fit_opt_in_required"
    ),
    "profile_registered_no_model_font_or_admitted_dataset": (
        "blocked_no_admitted_model_or_dataset"
    ),
    "image_profile_registered_unencoded_no_model_or_admitted_dataset": (
        "blocked_unencoded_no_admitted_model_or_dataset"
    ),
}


class BookOcrError(RuntimeError):
    """Base exception for this tool."""


class BookOcrPreflightError(BookOcrError):
    """Actionable preflight failure with all issues collected."""

    def __init__(self, message: str, issues: list[str]) -> None:
        self.issues = issues
        super().__init__(message + "\n" + "\n".join(f"- {issue}" for issue in issues))


@dataclass(slots=True)
class RecognizerSpec:
    ref: str
    script: str
    route_script: str
    writing_system_profile: str | None
    source_unicode_status: str | None
    output_policy: str | None
    artifact: Path | None
    config_artifact: str | None
    dictionary: Path | None
    status: str | None
    execution_policy: str | None
    artifact_overridden: bool


@dataclass(slots=True)
class LanguagePipelineConfig:
    config_path: Path
    config_id: str
    label: str | None
    registry_id: str | None
    primary_script: str | None
    secondary_script: str | None
    script_recognizer_refs: list[str]
    recognizer_ref: str | None
    converter: str | None
    text_layer_class: str | None
    recognizers: list[RecognizerSpec]
    declared_recognizers: list[RecognizerSpec]
    unavailable_recognizers: list[RecognizerSpec]
    inactive_recognizers: list[RecognizerSpec]
    blocked_recognizers: list[RecognizerSpec]
    recognizer_activation_mode: str
    requested_writing_system_profiles: list[str]
    newari_execution_opt_ins: list[str]
    raw_language: dict[str, Any]


@dataclass(slots=True)
class InputPage:
    page_number: int
    page_id: str
    image_path: Path
    source_path: Path
    source_page_number: int | None = None


@dataclass(slots=True)
class PageOcrResult:
    document: Document
    review_rows: list[dict[str, Any]]
    detection_line_count: int
    kept_line_count: int
    removed_line_count: int
    filter_report: dict[str, Any]


class PaddleBookOcrBackend:
    """Paddle detector plus per-script line recognizers."""

    def __init__(
        self, config: LanguagePipelineConfig, args: argparse.Namespace
    ) -> None:
        if not config.recognizers:
            raise BookOcrError("language config has no recognizers after resolution")
        seed = config.recognizers[0]
        if seed.artifact is None:
            raise BookOcrError(f"seed recognizer {seed.ref} has no artifact path")
        self.config = config
        self.args = args
        self.detector = PaddleOcrEngine(
            recognition_mode="full_page",
            text_recognition_model_name=args.paddle_recognition_model_name,
            text_recognition_model_dir=str(seed.artifact),
            text_det_limit_side_len=args.text_det_limit_side_len,
            text_det_thresh=args.text_det_thresh,
            text_det_box_thresh=args.text_det_box_thresh,
            text_det_unclip_ratio=args.text_det_unclip_ratio,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=args.device,
        )
        self.recognizers: dict[str, PaddleOcrEngine] = {}
        for spec in config.recognizers:
            if spec.artifact is None:
                raise BookOcrError(f"recognizer {spec.ref} has no artifact path")
            self.recognizers[spec.ref] = PaddleOcrEngine(
                recognition_mode="line",
                text_recognition_model_name=args.paddle_recognition_model_name,
                text_recognition_model_dir=str(spec.artifact),
                device=args.device,
            )

    def detect(self, image_path: Path) -> list[TextLine]:
        output = self.detector.recognize(image_path)
        if not output.pages:
            return []
        return list(output.pages[0].text_lines)

    def recognize_crop(
        self, spec: RecognizerSpec, crop_path: Path
    ) -> tuple[str, float | None]:
        recognizer = self.recognizers[spec.ref]
        output = recognizer.recognize(crop_path)
        best_text = ""
        best_confidence: float | None = None
        for page in output.pages:
            for line in page.text_lines:
                text = line.text.strip()
                confidence = line.confidence
                if not text:
                    continue
                if best_confidence is None or (
                    confidence is not None and confidence > best_confidence
                ):
                    best_text = text
                    best_confidence = confidence
        return best_text, best_confidence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sibling_module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BookOcrError(f"could not load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BookOcrError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BookOcrError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BookOcrError(f"{label} must contain a JSON object: {path}")
    return payload


def _safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip(".-")
    return cleaned or "page"


def _is_supported_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base_dir is not None and (base_dir / path).exists():
        return base_dir / path
    return _repo_root() / path


def _normalize_artifact_dir(path: Path) -> Path:
    if all((path / filename).is_file() for filename in REQUIRED_PADDLE_ARTIFACT_FILES):
        return path
    artifacts_dir = path / "artifacts"
    if all(
        (artifacts_dir / filename).is_file()
        for filename in REQUIRED_PADDLE_ARTIFACT_FILES
    ):
        return artifacts_dir
    return path


def _parse_recognizer_artifact_overrides(values: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise BookOcrError(
                f"--script-recognizer-artifact must be REF=PATH, got: {value!r}"
            )
        ref, path_value = value.split("=", 1)
        ref = ref.strip()
        if not ref:
            raise BookOcrError(
                f"--script-recognizer-artifact has an empty recognizer ref: {value!r}"
            )
        if not path_value.strip():
            raise BookOcrError(
                f"--script-recognizer-artifact has an empty path: {value!r}"
            )
        overrides[ref] = Path(path_value.strip())
    return overrides


def _find_language_entry(
    languages: list[Any],
    *,
    language: str | None,
    config_entry: str | None,
) -> dict[str, Any]:
    if language and config_entry:
        raise BookOcrError("use only one of --language or --config-entry")
    selector = language or config_entry
    if selector is None:
        raise BookOcrError("one of --language or --config-entry is required")
    if not isinstance(selector, str) or not selector.strip():
        raise BookOcrError("language/config-entry selector must be non-empty")
    selector = selector.strip()

    if config_entry is not None and selector.isdigit():
        index = int(selector)
        if index < 0 or index >= len(languages):
            raise BookOcrError(
                f"--config-entry index out of range: {index}; entries={len(languages)}"
            )
        entry = languages[index]
        if not isinstance(entry, dict):
            raise BookOcrError(f"config language entry {index} is not an object")
        return entry

    matches: list[dict[str, Any]] = []
    for entry in languages:
        if not isinstance(entry, dict):
            continue
        if language is not None:
            if entry.get("registry_id") == selector:
                matches.append(entry)
        else:
            if entry.get("registry_id") == selector or entry.get("label") == selector:
                matches.append(entry)
    if not matches:
        selector_kind = (
            "--language registry_id"
            if language is not None
            else "--config-entry registry_id/label/index"
        )
        raise BookOcrError(
            f"no language-pack config entry matched {selector_kind}: {selector!r}"
        )
    if len(matches) > 1:
        labels = ", ".join(
            str(item.get("label") or item.get("registry_id") or "<unnamed>")
            for item in matches
        )
        raise BookOcrError(
            f"selector {selector!r} matched multiple language entries: {labels}; use --config-entry index"
        )
    return matches[0]


def _recognizer_refs_for_language(
    language_entry: dict[str, Any], script_recognizers: dict[str, Any]
) -> list[str]:
    refs: list[str] = []

    def add(value: object) -> None:
        if (
            isinstance(value, str)
            and value
            and value in script_recognizers
            and value not in refs
        ):
            refs.append(value)

    add(language_entry.get("recognizer_ref"))
    add(language_entry.get("primary_script"))
    add(language_entry.get("secondary_script"))
    configured_refs = language_entry.get("script_recognizer_refs", [])
    if isinstance(configured_refs, list):
        for ref in configured_refs:
            add(ref)
    return refs


def _route_script_key(ref: str, language_entry: dict[str, Any]) -> str:
    if ref == language_entry.get("recognizer_ref") and isinstance(
        language_entry.get("primary_script"), str
    ):
        return str(language_entry["primary_script"])
    return ref


def _normalize_route_script(script: str) -> str:
    normalized = script.strip().lower()
    aliases = {
        "limbu_sirijonga": "limbu-sirijonga",
        "devanagari_limbu": "devanagari",
        "sunuwar_jenticha": "sunuwar",
        "newa_prachalit": "newa",
        "kiratrai": "kirat_rai",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in SCRIPT_CODEPOINT_RANGES:
        return normalized
    hyphenated = normalized.replace("_", "-")
    if hyphenated in SCRIPT_CODEPOINT_RANGES:
        return hyphenated
    return normalized


def _newari_execution_policy(ref: str, status: str | None) -> str:
    policy = NEWARI_EXECUTION_POLICY_BY_STATUS.get(status or "")
    if policy is None:
        supported = ", ".join(sorted(NEWARI_EXECUTION_POLICY_BY_STATUS))
        raise BookOcrError(
            f"Newar recognizer {ref} has unknown execution status {status!r}; "
            f"fail-closed statuses: {supported}"
        )
    return policy


def _newari_policy_error(spec: RecognizerSpec) -> BookOcrError:
    if spec.execution_policy == "diagnostic_opt_in_required":
        return BookOcrError(
            f"Newar writing-system profile {spec.writing_system_profile!r} recognizer "
            f"{spec.ref} is diagnostic-only; select it explicitly with --writing-system "
            "and pass --allow-newari-diagnostic"
        )
    if spec.execution_policy == "known_failed_fit_opt_in_required":
        return BookOcrError(
            f"Newar writing-system profile {spec.writing_system_profile!r} recognizer "
            f"{spec.ref} is a known failed fit; select it explicitly with "
            "--writing-system and pass --allow-newari-known-failed-fit"
        )
    return BookOcrError(
        f"Newar writing-system profile {spec.writing_system_profile!r} recognizer "
        f"{spec.ref} has non-runnable status {spec.status!r}; an artifact override may "
        "relocate an admitted model but cannot admit a model, dataset, or OCR claim"
    )


def _activate_recognizers(
    *,
    registry_id: object,
    declared: list[RecognizerSpec],
    writing_systems: list[str] | None,
    allow_newari_diagnostic: bool,
    allow_newari_known_failed_fit: bool,
) -> tuple[
    list[RecognizerSpec],
    list[RecognizerSpec],
    list[RecognizerSpec],
    list[RecognizerSpec],
    str,
    list[str],
    list[str],
]:
    selectors = list(writing_systems or [])
    if registry_id != "newar":
        if allow_newari_diagnostic or allow_newari_known_failed_fit:
            raise BookOcrError(
                "Newari diagnostic opt-ins are supported only for the Newar "
                "language entry (registry_id='newar')"
            )
        if selectors:
            raise BookOcrError(
                "--writing-system is supported only for the Newar language entry "
                "(registry_id='newar')"
            )
        unavailable = [spec for spec in declared if spec.artifact is None]
        return list(declared), unavailable, [], [], "all_declared", [], []

    by_profile: dict[str, RecognizerSpec] = {}
    for spec in declared:
        if not spec.writing_system_profile:
            raise BookOcrError(
                f"Newar recognizer {spec.ref} has no writing_system_profile"
            )
        try:
            profile = resolve_newari_writing_system(spec.writing_system_profile)
        except ValueError as exc:
            raise BookOcrError(
                f"Newar recognizer {spec.ref} has an invalid writing-system profile: {exc}"
            ) from exc
        if profile.profile_id != spec.writing_system_profile:
            raise BookOcrError(
                f"Newar recognizer {spec.ref} must use canonical writing_system_profile "
                f"{profile.profile_id!r}, got {spec.writing_system_profile!r}"
            )
        previous = by_profile.get(profile.profile_id)
        if previous is not None:
            raise BookOcrError(
                f"Newar writing-system profile {profile.profile_id!r} is declared by "
                f"multiple recognizers: {previous.ref}, {spec.ref}"
            )
        by_profile[profile.profile_id] = spec

    requested_profiles: list[str] = []
    for selector in selectors:
        try:
            profile = resolve_newari_writing_system(selector)
        except ValueError as exc:
            raise BookOcrError(str(exc)) from exc
        if profile.profile_id in requested_profiles:
            raise BookOcrError(
                f"--writing-system selects {profile.profile_id!r} more than once"
            )
        if profile.profile_id not in by_profile:
            raise BookOcrError(
                f"Newar language config does not declare writing-system profile "
                f"{profile.profile_id!r}"
            )
        requested_profiles.append(profile.profile_id)

    unavailable = [spec for spec in declared if spec.artifact is None]
    if (
        allow_newari_diagnostic or allow_newari_known_failed_fit
    ) and not requested_profiles:
        raise BookOcrError(
            "Newari diagnostic opt-ins require an explicit --writing-system selection"
        )
    opt_ins: list[str] = []
    if requested_profiles:
        active = []
        for profile_id in requested_profiles:
            spec = by_profile[profile_id]
            if spec.execution_policy == "admitted":
                active.append(spec)
            elif (
                spec.execution_policy == "diagnostic_opt_in_required"
                and allow_newari_diagnostic
            ):
                active.append(spec)
                if "diagnostic" not in opt_ins:
                    opt_ins.append("diagnostic")
            elif (
                spec.execution_policy == "known_failed_fit_opt_in_required"
                and allow_newari_known_failed_fit
            ):
                active.append(spec)
                if "known_failed_fit" not in opt_ins:
                    opt_ins.append("known_failed_fit")
            else:
                raise _newari_policy_error(spec)
        if allow_newari_diagnostic and "diagnostic" not in opt_ins:
            raise BookOcrError(
                "--allow-newari-diagnostic is unused by the selected writing systems"
            )
        if allow_newari_known_failed_fit and "known_failed_fit" not in opt_ins:
            raise BookOcrError(
                "--allow-newari-known-failed-fit is unused by the selected writing systems"
            )
        activation_mode = "explicit_policy_gated_writing_systems"
    else:
        active = [
            spec
            for spec in declared
            if spec.artifact is not None and spec.execution_policy == "admitted"
        ]
        activation_mode = "auto_inference_admitted"
    active_refs = {spec.ref for spec in active}
    inactive = [spec for spec in declared if spec.ref not in active_refs]
    blocked = [
        spec
        for spec in inactive
        if spec.execution_policy is not None and spec.execution_policy != "admitted"
    ]
    return (
        active,
        unavailable,
        inactive,
        blocked,
        activation_mode,
        requested_profiles,
        opt_ins,
    )


def resolve_language_config(
    config_path: Path,
    *,
    language: str | None = None,
    config_entry: str | None = None,
    recognizer_artifact_override: Path | None = None,
    script_recognizer_artifact_overrides: dict[str, Path] | None = None,
    writing_systems: list[str] | None = None,
    allow_newari_diagnostic: bool = False,
    allow_newari_known_failed_fit: bool = False,
) -> LanguagePipelineConfig:
    config_path = _resolve_path(config_path)
    payload = _read_json_object(config_path, label="language-pack pipeline config")
    languages = payload.get("languages")
    script_recognizers = payload.get("script_recognizers")
    if not isinstance(languages, list):
        raise BookOcrError(f"config languages must be a list: {config_path}")
    if not isinstance(script_recognizers, dict):
        raise BookOcrError(
            f"config script_recognizers must be an object: {config_path}"
        )

    language_entry = _find_language_entry(
        languages, language=language, config_entry=config_entry
    )
    configured_refs = language_entry.get("script_recognizer_refs", [])
    if not isinstance(configured_refs, list) or any(
        not isinstance(ref, str) or not ref for ref in configured_refs
    ):
        raise BookOcrError(
            "language config script_recognizer_refs must be a list of non-empty strings"
        )
    missing_configured_refs = [
        ref for ref in configured_refs if ref not in script_recognizers
    ]
    if missing_configured_refs:
        raise BookOcrError(
            "language config references missing script recognizers: "
            + ", ".join(missing_configured_refs)
        )
    refs = _recognizer_refs_for_language(language_entry, script_recognizers)
    if not refs:
        raise BookOcrError(
            "language config has no local recognizer refs: "
            f"registry_id={language_entry.get('registry_id')!r} label={language_entry.get('label')!r}"
        )

    overrides = dict(script_recognizer_artifact_overrides or {})
    recognizer_ref = language_entry.get("recognizer_ref")
    if (
        recognizer_artifact_override is not None
        and isinstance(recognizer_ref, str)
        and recognizer_ref
    ):
        overrides[recognizer_ref] = recognizer_artifact_override

    declared_recognizers: list[RecognizerSpec] = []
    for ref in refs:
        raw = script_recognizers.get(ref)
        if not isinstance(raw, dict):
            raise BookOcrError(f"script_recognizers.{ref} must be an object")
        config_artifact = (
            raw.get("artifact") if isinstance(raw.get("artifact"), str) else None
        )
        artifact: Path | None = None
        if ref in overrides:
            artifact = _resolve_path(overrides[ref], base_dir=config_path.parent)
        elif config_artifact:
            artifact = _resolve_path(config_artifact, base_dir=config_path.parent)
        if artifact is not None:
            artifact = _normalize_artifact_dir(artifact)
        dictionary = raw.get("dictionary")
        dictionary_path = (
            _resolve_path(dictionary, base_dir=config_path.parent)
            if isinstance(dictionary, str) and dictionary
            else None
        )
        script = _route_script_key(ref, language_entry)
        configured_route_script = raw.get("route_script")
        if configured_route_script is not None and not isinstance(
            configured_route_script, str
        ):
            raise BookOcrError(
                f"script_recognizers.{ref}.route_script must be a string when present"
            )
        route_script = _normalize_route_script(configured_route_script or script)
        status = str(raw.get("status")) if raw.get("status") is not None else None
        execution_policy = (
            _newari_execution_policy(ref, status)
            if language_entry.get("registry_id") == "newar"
            else None
        )
        declared_recognizers.append(
            RecognizerSpec(
                ref=ref,
                script=script,
                route_script=route_script,
                writing_system_profile=str(raw.get("writing_system_profile"))
                if raw.get("writing_system_profile") is not None
                else None,
                source_unicode_status=str(raw.get("source_unicode_status"))
                if raw.get("source_unicode_status") is not None
                else None,
                output_policy=str(raw.get("output_policy"))
                if raw.get("output_policy") is not None
                else None,
                artifact=artifact,
                config_artifact=config_artifact,
                dictionary=dictionary_path,
                status=status,
                execution_policy=execution_policy,
                artifact_overridden=ref in overrides,
            )
        )

    (
        recognizers,
        unavailable_recognizers,
        inactive_recognizers,
        blocked_recognizers,
        recognizer_activation_mode,
        requested_writing_system_profiles,
        newari_execution_opt_ins,
    ) = _activate_recognizers(
        registry_id=language_entry.get("registry_id"),
        declared=declared_recognizers,
        writing_systems=writing_systems,
        allow_newari_diagnostic=allow_newari_diagnostic,
        allow_newari_known_failed_fit=allow_newari_known_failed_fit,
    )

    return LanguagePipelineConfig(
        config_path=config_path,
        config_id=str(payload.get("config_id") or ""),
        label=str(language_entry.get("label"))
        if language_entry.get("label") is not None
        else None,
        registry_id=str(language_entry.get("registry_id"))
        if language_entry.get("registry_id") is not None
        else None,
        primary_script=str(language_entry.get("primary_script"))
        if language_entry.get("primary_script") is not None
        else None,
        secondary_script=str(language_entry.get("secondary_script"))
        if language_entry.get("secondary_script") is not None
        else None,
        script_recognizer_refs=list(configured_refs),
        recognizer_ref=str(recognizer_ref) if recognizer_ref is not None else None,
        converter=str(language_entry.get("converter"))
        if language_entry.get("converter") is not None
        else None,
        text_layer_class=str(language_entry.get("text_layer_class"))
        if language_entry.get("text_layer_class") is not None
        else None,
        recognizers=recognizers,
        declared_recognizers=declared_recognizers,
        unavailable_recognizers=unavailable_recognizers,
        inactive_recognizers=inactive_recognizers,
        blocked_recognizers=blocked_recognizers,
        recognizer_activation_mode=recognizer_activation_mode,
        requested_writing_system_profiles=requested_writing_system_profiles,
        newari_execution_opt_ins=newari_execution_opt_ins,
        raw_language=dict(language_entry),
    )


def validate_preflight(
    input_path: Path, config: LanguagePipelineConfig, args: argparse.Namespace
) -> None:
    issues: list[str] = []
    if not input_path.exists():
        issues.append(f"input path does not exist: {input_path}")
    elif input_path.is_dir():
        image_count = sum(
            1 for path in input_path.iterdir() if _is_supported_image_file(path)
        )
        if image_count == 0:
            issues.append(
                f"input image directory contains no supported image files: {input_path}"
            )
    elif input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            issues.append(
                f"input file must be a PDF or an image directory, got: {input_path}"
            )
    else:
        issues.append(f"input path is neither a file nor a directory: {input_path}")

    for name, value in (
        ("--min-confidence", args.min_confidence),
        ("--review-threshold", args.review_threshold),
        ("--text-det-thresh", args.text_det_thresh),
        ("--text-det-box-thresh", args.text_det_box_thresh),
    ):
        if value < 0.0 or value > 1.0:
            issues.append(f"{name} must be between 0 and 1, got {value}")
    if (
        args.min_y_ratio is not None
        and args.max_y_ratio is not None
        and args.min_y_ratio > args.max_y_ratio
    ):
        issues.append(
            f"--min-y-ratio must be <= --max-y-ratio, got {args.min_y_ratio} > {args.max_y_ratio}"
        )
    if args.max_pages is not None and args.max_pages < 1:
        issues.append(
            f"--max-pages must be at least 1 when provided, got {args.max_pages}"
        )

    if not config.recognizers:
        issues.append("resolved language config has no recognizers")
    for spec in config.recognizers:
        if spec.route_script not in SCRIPT_CODEPOINT_RANGES:
            supported = ", ".join(sorted(SCRIPT_CODEPOINT_RANGES))
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} uses unsupported script routing key "
                f"{spec.script!r} -> {spec.route_script!r}; supported: {supported}"
            )
        if spec.artifact is None:
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} has no local artifact configured; "
                "train/export the recognizer or pass --script-recognizer-artifact "
                f"{spec.ref}=PATH"
            )
            continue
        if not spec.artifact.exists():
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} artifact path is missing: {spec.artifact} "
                f"(config artifact={spec.config_artifact!r})"
            )
            continue
        if not spec.artifact.is_dir():
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} artifact path is not a directory: {spec.artifact}"
            )
            continue
        missing_files = [
            filename
            for filename in REQUIRED_PADDLE_ARTIFACT_FILES
            if not (spec.artifact / filename).is_file()
        ]
        if missing_files:
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} artifact directory is incomplete: "
                f"{spec.artifact}; missing {', '.join(missing_files)}"
            )
        if spec.dictionary is not None and not spec.dictionary.is_file():
            issues.append(
                f"{config_identity(config)} recognizer {spec.ref} dictionary is missing: {spec.dictionary}"
            )
    if issues:
        raise BookOcrPreflightError("run_book_ocr preflight failed", issues)


def config_identity(config: LanguagePipelineConfig) -> str:
    registry = config.registry_id or "<no-registry-id>"
    label = config.label or "<no-label>"
    return f"language registry_id={registry!r} label={label!r}"


def _recognizer_record(spec: RecognizerSpec) -> dict[str, Any]:
    return {
        "ref": spec.ref,
        "script": spec.script,
        "route_script": spec.route_script,
        "writing_system_profile": spec.writing_system_profile,
        "source_unicode_status": spec.source_unicode_status,
        "output_policy": spec.output_policy,
        "artifact": str(spec.artifact) if spec.artifact is not None else None,
        "config_artifact": spec.config_artifact,
        "dictionary": str(spec.dictionary) if spec.dictionary is not None else None,
        "status": spec.status,
        "execution_policy": spec.execution_policy,
        "artifact_overridden": spec.artifact_overridden,
        "quality_claim_ready": False,
        "claim_evidence_eligible": False,
    }


def _recognizer_activation_record(config: LanguagePipelineConfig) -> dict[str, Any]:
    return {
        "mode": config.recognizer_activation_mode,
        "requested_writing_system_profiles": config.requested_writing_system_profiles,
        "declared_refs": [spec.ref for spec in config.declared_recognizers],
        "active_refs": [spec.ref for spec in config.recognizers],
        "inactive_refs": [spec.ref for spec in config.inactive_recognizers],
        "unavailable_refs": [spec.ref for spec in config.unavailable_recognizers],
        "blocked_refs": [spec.ref for spec in config.blocked_recognizers],
        "newari_execution_opt_ins": config.newari_execution_opt_ins,
        "quality_claim_ready": False,
        "claim_evidence_eligible": False,
    }


def find_missing_config_artifact_paths(config_path: Path) -> list[dict[str, str]]:
    config_path = _resolve_path(config_path)
    payload = _read_json_object(config_path, label="language-pack pipeline config")
    languages = payload.get("languages")
    script_recognizers = payload.get("script_recognizers")
    if not isinstance(languages, list) or not isinstance(script_recognizers, dict):
        raise BookOcrError(f"invalid language-pack config shape: {config_path}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for language_entry in languages:
        if not isinstance(language_entry, dict):
            continue
        refs = _recognizer_refs_for_language(language_entry, script_recognizers)
        for ref in refs:
            recognizer = script_recognizers.get(ref)
            if not isinstance(recognizer, dict):
                continue
            artifact = recognizer.get("artifact")
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            resolved = _normalize_artifact_dir(
                _resolve_path(artifact, base_dir=config_path.parent)
            )
            if resolved.exists():
                missing_files = [
                    filename
                    for filename in REQUIRED_PADDLE_ARTIFACT_FILES
                    if not (resolved / filename).is_file()
                ]
                if not missing_files:
                    continue
            registry_id = str(language_entry.get("registry_id") or "")
            label = str(language_entry.get("label") or "")
            key = (registry_id, ref, artifact)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "registry_id": registry_id,
                    "label": label,
                    "recognizer_ref": ref,
                    "artifact": artifact,
                    "resolved_path": str(resolved),
                }
            )
    return rows


def prepare_input_pages(
    input_path: Path, out: Path, args: argparse.Namespace, logger: logging.Logger
) -> list[InputPage]:
    if input_path.is_dir():
        pages = _image_dir_pages(input_path, max_pages=args.max_pages)
        logger.info("input prep: using %d image files from %s", len(pages), input_path)
        return pages
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pages = _render_pdf_pages(input_path, out / "page-images", args, logger)
        logger.info("input prep: rendered %d PDF pages from %s", len(pages), input_path)
        return pages
    raise BookOcrError(f"unsupported input path after preflight: {input_path}")


def _image_dir_pages(input_dir: Path, *, max_pages: int | None) -> list[InputPage]:
    image_paths = [
        path for path in sorted(input_dir.iterdir()) if _is_supported_image_file(path)
    ]
    if max_pages is not None:
        image_paths = image_paths[:max_pages]
    pages: list[InputPage] = []
    for index, image_path in enumerate(image_paths, start=1):
        pages.append(
            InputPage(
                page_number=index,
                page_id=_safe_name(image_path.stem),
                image_path=image_path,
                source_path=image_path,
                source_page_number=None,
            )
        )
    if not pages:
        raise BookOcrError(
            f"no supported image files found in input directory: {input_dir}"
        )
    return pages


def _render_pdf_pages(
    pdf_path: Path, rendered_dir: Path, args: argparse.Namespace, logger: logging.Logger
) -> list[InputPage]:
    try:
        import fitz
    except ImportError as exc:
        raise BookOcrError(
            "PDF input requires PyMuPDF; install/run with pymupdf available"
        ) from exc
    rendered_dir.mkdir(parents=True, exist_ok=True)
    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise BookOcrError(
            f"could not open PDF {pdf_path}: {type(exc).__name__}: {exc}"
        ) from exc
    pages: list[InputPage] = []
    try:
        page_count = len(document)
        start = max(1, int(args.pdf_start_page))
        end = int(args.pdf_end_page) if args.pdf_end_page is not None else page_count
        end = min(end, page_count)
        if start > end:
            raise BookOcrError(
                f"empty PDF page range: start={start} end={end} page_count={page_count}"
            )
        matrix = fitz.Matrix(2, 2)
        for pdf_page_number in range(start, end + 1):
            if args.max_pages is not None and len(pages) >= args.max_pages:
                break
            page = document[pdf_page_number - 1]
            target = rendered_dir / f"page-{pdf_page_number:04d}.png"
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(str(target))
            except Exception as exc:
                raise BookOcrError(
                    f"could not render PDF page {pdf_page_number} from {pdf_path}: {type(exc).__name__}: {exc}"
                ) from exc
            logger.info("rendered PDF page %d -> %s", pdf_page_number, target)
            pages.append(
                InputPage(
                    page_number=len(pages) + 1,
                    page_id=f"{_safe_name(pdf_path.stem)}-p{pdf_page_number:04d}",
                    image_path=target,
                    source_path=pdf_path,
                    source_page_number=pdf_page_number,
                )
            )
    finally:
        document.close()
    if not pages:
        raise BookOcrError(f"PDF input produced no pages: {pdf_path}")
    return pages


def _build_backend(
    config: LanguagePipelineConfig, args: argparse.Namespace
) -> PaddleBookOcrBackend:
    return PaddleBookOcrBackend(config, args)


def _run_page_ocr(
    page: InputPage,
    page_dir: Path,
    config: LanguagePipelineConfig,
    args: argparse.Namespace,
    backend: PaddleBookOcrBackend,
    logger: logging.Logger,
) -> PageOcrResult:
    try:
        from PIL import Image
    except ImportError as exc:
        raise BookOcrError("Pillow is required for page crop handling") from exc

    logger.info("OCR page %d %s: %s", page.page_number, page.page_id, page.image_path)
    try:
        with Image.open(page.image_path) as source:
            image = source.convert("RGB")
    except OSError as exc:
        raise BookOcrError(
            f"could not open page image {page.image_path}: {type(exc).__name__}: {exc}"
        ) from exc
    width, height = image.size
    detections = backend.detect(page.image_path)
    crops_dir = page_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    text_lines: list[TextLine] = []
    with tempfile.TemporaryDirectory(prefix="ocrtech-book-page-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, detection in enumerate(detections, start=1):
            bbox = _clamped_bbox(detection.bbox, image_width=width, image_height=height)
            if bbox.w <= 0 or bbox.h <= 0:
                logger.warning(
                    "dropping invalid detection bbox on page %s line %d: %s",
                    page.page_id,
                    index,
                    detection.bbox.to_list(),
                )
                continue
            crop = _crop_image_to_bbox(image, bbox, padding=args.crop_padding)
            tmp_crop = tmp_dir / f"line-{index:04d}.png"
            crop.save(tmp_crop)
            crop_path = crops_dir / f"line-{index:04d}.png"
            crop.save(crop_path)
            selected = _recognize_and_route_line(
                backend=backend,
                specs=config.recognizers,
                crop_path=tmp_crop,
                seed_text=detection.text,
                seed_confidence=detection.confidence,
            )
            line_id = f"p0000-l{len(text_lines) + 1:04d}"
            metadata = {
                "book_ocr": {
                    "line_index": len(text_lines) + 1,
                    "crop_path": str(crop_path),
                    "crop_sha256": sha256_file(crop_path),
                    "detector_seed_text": detection.text,
                    "detector_seed_confidence": detection.confidence,
                    "selected_recognizer": selected["selected_recognizer"],
                    "selected_script": selected["selected_script"],
                    "selected_route_script": selected["selected_route_script"],
                    "selected_writing_system_profile": selected[
                        "selected_writing_system_profile"
                    ],
                    "selected_source_unicode_status": selected[
                        "selected_source_unicode_status"
                    ],
                    "selected_output_policy": selected["selected_output_policy"],
                    "selected_recognizer_status": selected[
                        "selected_recognizer_status"
                    ],
                    "selected_execution_policy": selected["selected_execution_policy"],
                    "quality_claim_ready": False,
                    "claim_evidence_eligible": False,
                    "selected_score": selected["selected_score"],
                    "candidates": selected["candidates"],
                }
            }
            text_lines.append(
                TextLine(
                    text=str(selected["text"]),
                    bbox=bbox,
                    confidence=selected["confidence"],
                    page_index=0,
                    line_id=line_id,
                    metadata=metadata,
                )
            )

    document = _document_from_lines(
        page=page,
        width=width,
        height=height,
        lines=text_lines,
        config=config,
        args=args,
    )
    ordered_document, reading_order_reports = _apply_reading_order(
        document, args.reading_order
    )
    ordered_document.metadata["reading_order_reports"] = reading_order_reports
    review_rows = _review_rows_for_page(
        ordered_document, page=page, threshold=args.review_threshold
    )
    filtered_document, filter_report = _filter_document(ordered_document, args)
    return PageOcrResult(
        document=filtered_document,
        review_rows=review_rows,
        detection_line_count=len(text_lines),
        kept_line_count=sum(
            len(doc_page.text_lines) for doc_page in filtered_document.pages
        ),
        removed_line_count=int(filter_report.get("removed_line_count") or 0),
        filter_report=filter_report,
    )


def _clamped_bbox(bbox: BBox, *, image_width: int, image_height: int) -> BBox:
    left = max(0.0, min(float(image_width), bbox.x))
    top = max(0.0, min(float(image_height), bbox.y))
    right = max(left, min(float(image_width), bbox.right))
    bottom = max(top, min(float(image_height), bbox.bottom))
    return BBox(left, top, right - left, bottom - top)


def _crop_image_to_bbox(image: Any, bbox: BBox, *, padding: int) -> Any:
    left = max(0, int(bbox.x) - padding)
    top = max(0, int(bbox.y) - padding)
    right = min(image.width, int(bbox.right) + padding)
    bottom = min(image.height, int(bbox.bottom) + padding)
    if right <= left or bottom <= top:
        return image.crop((0, 0, 1, 1))
    return image.crop((left, top, right, bottom))


def _recognize_and_route_line(
    *,
    backend: PaddleBookOcrBackend,
    specs: list[RecognizerSpec],
    crop_path: Path,
    seed_text: str,
    seed_confidence: float | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        text, confidence = backend.recognize_crop(spec, crop_path)
        ratio = _script_ratio(text, spec.route_script) if text else 0.0
        score = float(confidence or 0.0) + ratio * 0.25
        candidates.append(
            {
                "recognizer_ref": spec.ref,
                "script": spec.script,
                "route_script": spec.route_script,
                "writing_system_profile": spec.writing_system_profile,
                "source_unicode_status": spec.source_unicode_status,
                "output_policy": spec.output_policy,
                "recognizer_status": spec.status,
                "execution_policy": spec.execution_policy,
                "artifact_overridden": spec.artifact_overridden,
                "quality_claim_ready": False,
                "claim_evidence_eligible": False,
                "text": text,
                "confidence": confidence,
                "script_ratio": ratio,
                "score": score,
            }
        )
    if not candidates:
        return {
            "text": seed_text,
            "confidence": seed_confidence,
            "selected_recognizer": "paddle_seed",
            "selected_script": None,
            "selected_route_script": None,
            "selected_writing_system_profile": None,
            "selected_source_unicode_status": None,
            "selected_output_policy": None,
            "selected_recognizer_status": None,
            "selected_execution_policy": None,
            "selected_score": float(seed_confidence or 0.0),
            "candidates": [],
        }
    selected = max(
        candidates,
        key=lambda item: (float(item["score"]), float(item.get("confidence") or 0.0)),
    )
    return {
        "text": selected["text"],
        "confidence": selected["confidence"],
        "selected_recognizer": selected["recognizer_ref"],
        "selected_script": selected["script"],
        "selected_route_script": selected["route_script"],
        "selected_writing_system_profile": selected["writing_system_profile"],
        "selected_source_unicode_status": selected["source_unicode_status"],
        "selected_output_policy": selected["output_policy"],
        "selected_recognizer_status": selected["recognizer_status"],
        "selected_execution_policy": selected["execution_policy"],
        "selected_score": selected["score"],
        "candidates": candidates,
    }


def _document_from_lines(
    *,
    page: InputPage,
    width: int,
    height: int,
    lines: list[TextLine],
    config: LanguagePipelineConfig,
    args: argparse.Namespace,
) -> Document:
    blocks = [
        Block(
            block_id=f"{line.line_id or f'p0000-l{index:04d}'}-block",
            block_type="text",
            page_index=0,
            bbox=line.bbox,
            order=index,
            text=line.text,
            confidence=line.confidence,
            line_ids=[line.line_id] if line.line_id else [],
            metadata={"source": "run-book-ocr", "block_granularity": "line"},
        )
        for index, line in enumerate(lines, start=1)
    ]
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/run_book_ocr.py",
        "language": {
            "registry_id": config.registry_id,
            "label": config.label,
            "primary_script": config.primary_script,
            "secondary_script": config.secondary_script,
            "script_recognizer_refs": config.script_recognizer_refs,
            "recognizer_ref": config.recognizer_ref,
            "recognizer_activation": _recognizer_activation_record(config),
            "text_layer_class": config.text_layer_class,
            "converter": config.converter,
        },
        "paddle": {
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_thresh": args.text_det_thresh,
            "text_det_box_thresh": args.text_det_box_thresh,
            "text_det_unclip_ratio": args.text_det_unclip_ratio,
            "recognition_model_name": args.paddle_recognition_model_name,
        },
        "filter_defaults": _filter_parameters(args),
        "source_page": {
            "page_number": page.page_number,
            "page_id": page.page_id,
            "image_path": str(page.image_path),
            "image_sha256": sha256_file(page.image_path),
            "source_path": str(page.source_path),
            "source_page_number": page.source_page_number,
        },
    }
    return Document(
        source_path=str(page.image_path),
        pages=[
            Page(
                page_index=0,
                width=width,
                height=height,
                text_lines=lines,
                blocks=blocks,
                metadata={"source_page_id": page.page_id},
            )
        ],
        metadata=metadata,
    )


def _apply_reading_order(
    document: Document, mode: str
) -> tuple[Document, list[dict[str, Any]]]:
    module = _load_sibling_module("apply_page_reading_order_strategy")
    if not hasattr(module, "_apply_to_document"):
        raise BookOcrError(
            "apply_page_reading_order_strategy.py does not expose _apply_to_document"
        )
    ordered, reports = module._apply_to_document(document, mode)
    return ordered, reports


def _filter_document(
    document: Document, args: argparse.Namespace
) -> tuple[Document, dict[str, Any]]:
    module = _load_sibling_module("filter_page_ocr_documents")
    if not hasattr(module, "_filter_page"):
        raise BookOcrError("filter_page_ocr_documents.py does not expose _filter_page")
    filter_args = argparse.Namespace(
        drop_empty=args.drop_empty,
        min_confidence=args.min_confidence,
        require_script="any",
        script_ratio_threshold=0.20,
        min_height=None,
        max_height=None,
        min_width=None,
        max_width=None,
        min_width_ratio=None,
        max_width_ratio=None,
        min_height_ratio=None,
        max_height_ratio=None,
        min_y_ratio=args.min_y_ratio,
        max_y_ratio=args.max_y_ratio,
        infer_margin_page_markers=False,
        drop_structural_role=[],
        keep_structural_role=[],
    )
    pages: list[Page] = []
    page_reports: list[dict[str, Any]] = []
    source_line_count = 0
    kept_line_count = 0
    removed_line_count = 0
    removal_counts: dict[str, int] = {}
    for page in document.pages:
        filtered_page, page_report, _line_reports = module._filter_page(
            page, filter_args
        )
        pages.append(filtered_page)
        page_reports.append(page_report)
        source_line_count += int(page_report.get("source_line_count") or 0)
        kept_line_count += int(page_report.get("kept_line_count") or 0)
        removed_line_count += int(page_report.get("removed_line_count") or 0)
        for reason, count in (page_report.get("removal_counts") or {}).items():
            removal_counts[str(reason)] = removal_counts.get(str(reason), 0) + int(
                count
            )
    metadata = dict(document.metadata)
    metadata["ocr_line_filter"] = _filter_parameters(args)
    metadata["line_count"] = kept_line_count
    metadata["filtered_line_count"] = removed_line_count
    return (
        Document(
            source_path=document.source_path,
            pages=pages,
            tables=document.tables,
            figures=document.figures,
            metadata=metadata,
        ),
        {
            "source_line_count": source_line_count,
            "kept_line_count": kept_line_count,
            "removed_line_count": removed_line_count,
            "removal_counts": removal_counts,
            "pages": page_reports,
        },
    )


def _filter_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "drop_empty": bool(args.drop_empty),
        "min_confidence": args.min_confidence,
        "min_y_ratio": args.min_y_ratio,
        "max_y_ratio": args.max_y_ratio,
    }


def _review_rows_for_page(
    document: Document, *, page: InputPage, threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc_page in document.pages:
        for line_index, line in enumerate(doc_page.text_lines, start=1):
            confidence = line.confidence
            if confidence is not None and confidence >= threshold:
                continue
            metadata = line.metadata if isinstance(line.metadata, dict) else {}
            book_ocr = (
                metadata.get("book_ocr")
                if isinstance(metadata.get("book_ocr"), dict)
                else {}
            )
            reason = (
                "missing_confidence" if confidence is None else "below_review_threshold"
            )
            rows.append(
                {
                    "review_id": f"{page.page_id}:{line.line_id or line_index}",
                    "page_number": page.page_number,
                    "page_id": page.page_id,
                    "source_image_path": str(page.image_path),
                    "line_index": line_index,
                    "line_id": line.line_id,
                    "text": line.text,
                    "confidence": confidence,
                    "review_threshold": threshold,
                    "review_reason": reason,
                    "bbox": line.bbox.to_list(),
                    "crop_path": book_ocr.get("crop_path"),
                    "crop_sha256": book_ocr.get("crop_sha256"),
                    "selected_recognizer": book_ocr.get("selected_recognizer"),
                    "selected_script": book_ocr.get("selected_script"),
                    "selected_recognizer_status": book_ocr.get(
                        "selected_recognizer_status"
                    ),
                    "selected_execution_policy": book_ocr.get(
                        "selected_execution_policy"
                    ),
                    "quality_claim_ready": False,
                    "claim_evidence_eligible": False,
                    "candidates": book_ocr.get("candidates", []),
                }
            )
    return rows


def _write_document_artifacts(
    page_dir: Path, document: Document, review_rows: list[dict[str, Any]]
) -> None:
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "document.json").write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (page_dir / "document.md").write_text(
        render_document_markdown(document), encoding="utf-8"
    )
    (page_dir / "document.body.md").write_text(
        render_document_markdown(document, include_structural_roles=False),
        encoding="utf-8",
    )
    (page_dir / "review-queue.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )


def _read_existing_page(page_dir: Path) -> tuple[Document, list[dict[str, Any]]]:
    document_path = page_dir / "document.json"
    try:
        document = Document.from_dict(
            json.loads(document_path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise BookOcrError(f"resume document is missing: {document_path}") from exc
    except json.JSONDecodeError as exc:
        raise BookOcrError(
            f"resume document is invalid JSON: {document_path}: {exc}"
        ) from exc
    queue_path = page_dir / "review-queue.jsonl"
    rows: list[dict[str, Any]] = []
    if queue_path.is_file():
        for line_number, line in enumerate(
            queue_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BookOcrError(
                    f"resume review queue has invalid JSON at {queue_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise BookOcrError(
                    f"resume review queue row must be an object at {queue_path}:{line_number}"
                )
            rows.append(payload)
    return document, rows


def _write_book_outputs(
    out: Path, page_documents: list[Document], review_rows: list[dict[str, Any]]
) -> None:
    page_texts = [
        document.text.strip() for document in page_documents if document.text.strip()
    ]
    (out / "book.txt").write_text(
        "\n\n".join(page_texts) + ("\n" if page_texts else ""), encoding="utf-8"
    )
    md_lines: list[str] = ["# Book OCR", ""]
    for page_number, document in enumerate(page_documents, start=1):
        text = render_document_markdown(document).strip()
        md_lines.extend([f"## Page {page_number}", ""])
        if text:
            md_lines.extend([text, ""])
    (out / "book.md").write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    (out / "review-queue.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )


def run_book_ocr(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger(LOGGER_NAME)
    input_path = Path(args.input)
    config = resolve_language_config(
        Path(args.config),
        language=args.language,
        config_entry=args.config_entry,
        recognizer_artifact_override=args.recognizer_artifact_override,
        script_recognizer_artifact_overrides=_parse_recognizer_artifact_overrides(
            args.script_recognizer_artifact
        ),
        writing_systems=getattr(args, "writing_system", None),
        allow_newari_diagnostic=getattr(args, "allow_newari_diagnostic", False),
        allow_newari_known_failed_fit=getattr(
            args, "allow_newari_known_failed_fit", False
        ),
    )
    validate_preflight(input_path, config, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = prepare_input_pages(input_path, out, args, logger)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    backend: PaddleBookOcrBackend | None = None
    page_documents: list[Document] = []
    all_review_rows: list[dict[str, Any]] = []
    page_summaries: list[dict[str, Any]] = []

    for page in pages:
        page_dir = pages_dir / f"{page.page_number:04d}-{_safe_name(page.page_id)}"
        document_path = page_dir / "document.json"
        if args.resume and document_path.is_file():
            logger.info(
                "resume: skipping page %d %s because %s exists",
                page.page_number,
                page.page_id,
                document_path,
            )
            document, review_rows = _read_existing_page(page_dir)
            skipped = True
            detection_line_count = sum(
                len(doc_page.text_lines) for doc_page in document.pages
            )
            kept_line_count = detection_line_count
            removed_line_count = 0
            filter_report: dict[str, Any] = {}
        else:
            if backend is None:
                backend = _build_backend(config, args)
            result = _run_page_ocr(page, page_dir, config, args, backend, logger)
            _write_document_artifacts(page_dir, result.document, result.review_rows)
            document = result.document
            review_rows = result.review_rows
            skipped = False
            detection_line_count = result.detection_line_count
            kept_line_count = result.kept_line_count
            removed_line_count = result.removed_line_count
            filter_report = result.filter_report
        page_documents.append(document)
        all_review_rows.extend(review_rows)
        page_summaries.append(
            {
                "page_number": page.page_number,
                "page_id": page.page_id,
                "image_path": str(page.image_path),
                "image_sha256": sha256_file(page.image_path),
                "document_json": str(document_path),
                "document_json_sha256": sha256_file(document_path)
                if document_path.is_file()
                else None,
                "resumed": skipped,
                "detection_line_count": detection_line_count,
                "kept_line_count": kept_line_count,
                "removed_line_count": removed_line_count,
                "review_row_count": len(review_rows),
                "filter_report": filter_report,
            }
        )

    _write_book_outputs(out, page_documents, all_review_rows)
    run_summary = {
        "tool": "tools/run_book_ocr.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "machine": platform.node(),
        "config": str(config.config_path),
        "config_id": config.config_id,
        "language": {
            "registry_id": config.registry_id,
            "label": config.label,
            "primary_script": config.primary_script,
            "secondary_script": config.secondary_script,
            "script_recognizer_refs": config.script_recognizer_refs,
            "recognizer_ref": config.recognizer_ref,
            "recognizer_activation": _recognizer_activation_record(config),
        },
        "recognizers": [_recognizer_record(spec) for spec in config.recognizers],
        "declared_recognizers": [
            _recognizer_record(spec) for spec in config.declared_recognizers
        ],
        "input": str(input_path),
        "out": str(out),
        "page_count": len(page_summaries),
        "line_count": sum(int(page["kept_line_count"]) for page in page_summaries),
        "detection_line_count": sum(
            int(page["detection_line_count"]) for page in page_summaries
        ),
        "removed_line_count": sum(
            int(page["removed_line_count"]) for page in page_summaries
        ),
        "review_row_count": len(all_review_rows),
        "book_txt": str(out / "book.txt"),
        "book_md": str(out / "book.md"),
        "review_queue_jsonl": str(out / "review-queue.jsonl"),
        "pages": page_summaries,
    }
    summary_path = out / "book-ocr-run.json"
    run_summary["run_summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote run summary: %s", summary_path)
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--language", help="language registry_id from the pipeline config, e.g. limbu"
    )
    selector.add_argument(
        "--config-entry",
        help="language config entry by registry_id, label, or zero-based index",
    )
    parser.add_argument(
        "input", type=Path, help="input PDF or directory of page images"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--recognizer-artifact-override",
        type=Path,
        help="override the artifact path for the selected language's recognizer_ref",
    )
    parser.add_argument(
        "--script-recognizer-artifact",
        action="append",
        default=[],
        metavar="REF=PATH",
        help="override a specific script recognizer artifact path; may be repeated",
    )
    parser.add_argument(
        "--writing-system",
        action="append",
        default=[],
        metavar="PROFILE",
        help=(
            "activate an exact Newari writing-system profile (aliases such as "
            "Prachalit, Bhujimol, Ranjana, or Deva are accepted); may be repeated"
        ),
    )
    parser.add_argument(
        "--allow-newari-diagnostic",
        action="store_true",
        help=(
            "allow an explicitly selected Newari diagnostic-only recognizer; "
            "does not admit accuracy or claim evidence"
        ),
    )
    parser.add_argument(
        "--allow-newari-known-failed-fit",
        action="store_true",
        help=(
            "allow an explicitly selected Newari recognizer whose frozen evidence "
            "marks it as a known failed fit; does not admit accuracy or claim evidence"
        ),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--pdf-start-page", type=int, default=1)
    parser.add_argument("--pdf-end-page", type=int)
    parser.add_argument(
        "--reading-order",
        choices=[
            "Auto",
            "Rows LTR",
            "Rows RTL",
            "Rows LTR Top",
            "Rows RTL Top",
            "Columns LTR",
            "Columns RTL",
            "Column Major",
        ],
        default="Column Major",
    )
    parser.add_argument(
        "--drop-empty", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--min-y-ratio", type=float, default=0.12)
    parser.add_argument("--max-y-ratio", type=float, default=0.96)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--review-threshold", type=float, default=0.70)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument(
        "--paddle-recognition-model-name", default="PP-OCRv5_mobile_rec"
    )
    parser.add_argument("--text-det-limit-side-len", type=int, default=1280)
    parser.add_argument("--text-det-thresh", type=float, default=0.30)
    parser.add_argument("--text-det-box-thresh", type=float, default=0.60)
    parser.add_argument("--text-det-unclip-ratio", type=float, default=1.5)
    parser.add_argument(
        "--device",
        default=os.environ.get("OCRTECH_PADDLE_DEVICE")
        or os.environ.get("LIMBU_OCR_PADDLE_DEVICE")
        or "cpu",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    try:
        summary = run_book_ocr(args)
    except BookOcrPreflightError as exc:
        logging.getLogger(LOGGER_NAME).error("%s", exc)
        return 2
    except BookOcrError as exc:
        logging.getLogger(LOGGER_NAME).error("%s", exc)
        return 1
    except Exception as exc:
        logging.getLogger(LOGGER_NAME).exception(
            "unexpected run_book_ocr failure: %s", exc
        )
        return 1
    print(summary["run_summary"])
    print(
        f"pages={summary['page_count']} lines={summary['line_count']} review_rows={summary['review_row_count']}"
    )
    print(f"book_txt={summary['book_txt']}")
    print(f"review_queue={summary['review_queue_jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
