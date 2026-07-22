"""Preflight checks for claim-grade benchmark runs."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError, PreflightError
from .evalpack import audit_eval_pack
from .manifest import ManifestEntry, SUPPORTED_OCR_IMAGE_SUFFIXES, load_manifest, sha256_file, sha256_text
from .models import ModelCard, audit_model_card, component_baseline_names, inspect_routed_calibration_collapse, model_admission_status, trained_recognizer_provenance_status
from .references import load_reference
from .validation import ValidationConfig, check_leakage


_CANDIDATE_BASELINES = {"ours", "ocrtech", "candidate"}
_TESSERACT_BASELINES = {"tesseract"}
_PADDLE_BASELINES = {"stock-paddle", "paddleocr", "paddle"}
_SURYA_BASELINES = {"surya"}
_GLM_BASELINES = {"glm-ocr"}
_PADDLE_VL_BASELINES = {"paddleocr-vl"}
_KNOWN_BASELINES = _CANDIDATE_BASELINES | _TESSERACT_BASELINES | _PADDLE_BASELINES | _SURYA_BASELINES | _GLM_BASELINES | _PADDLE_VL_BASELINES


@dataclass(slots=True)
class PreflightCheck:
    name: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(slots=True)
class PreflightReport:
    passed: bool
    eval_manifest: str
    output_dir: str
    baselines: list[str]
    sample_count: int
    checks: list[PreflightCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "eval_manifest": self.eval_manifest,
            "output_dir": self.output_dir,
            "baselines": self.baselines,
            "sample_count": self.sample_count,
            "checks": [check.to_dict() for check in self.checks],
        }


def run_claim_preflight(
    eval_manifest: str | Path,
    output_dir: str | Path,
    *,
    baselines: list[str],
    references_dir: str | Path | None = None,
    candidate_model_config: str | Path | None = None,
    fallback_model_config: str | Path | None = None,
    train_manifests: list[str | Path] | None = None,
    validation_config: str | Path | None = None,
    gorkhapatra_review_audits: list[str | Path] | None = None,
    require_claim_ready_eval_pack: bool = True,
    require_trained_recognizer: bool = False,
    require_model_admission: bool = False,
) -> PreflightReport:
    normalized = [_normalize_baseline(baseline) for baseline in baselines]
    if not normalized:
        raise PreflightError("preflight requires at least one baseline")
    unknown = sorted({baseline for baseline in normalized if baseline not in _KNOWN_BASELINES})
    if unknown:
        raise PreflightError(f"unknown baselines for preflight: {', '.join(unknown)}")

    eval_path = Path(eval_manifest)
    manifest_dir = eval_path.parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[PreflightCheck] = []

    entries: list[ManifestEntry] | None = None
    references_root = Path(references_dir) if references_dir else None

    manifest_check, entries = _manifest_check(eval_path)
    checks.append(manifest_check)
    checks.append(_ocr_input_suffix_check(entries, eval_path.parent))

    checks.append(_reference_check(entries, manifest_dir, references_root))
    checks.append(_gorkhapatra_language_page_review_check(entries, gorkhapatra_review_audits or []))
    checks.append(_eval_pack_check(eval_path, entries, out / "eval-pack-audit", require_claim_ready_eval_pack))
    checks.extend(_baseline_checks(normalized, candidate_model_config))
    checks.append(_fallback_model_check(fallback_model_config))
    checks.append(_candidate_fallback_provenance_check(candidate_model_config, fallback_model_config))
    checks.append(_trained_recognizer_check(candidate_model_config, require_trained_recognizer))
    checks.append(_model_admission_check(candidate_model_config, require_model_admission))
    checks.append(_candidate_calibration_check(eval_path, candidate_model_config))
    checks.append(_candidate_component_baseline_check(normalized, candidate_model_config))
    checks.append(_validation_config_check(validation_config, normalized, entries))
    checks.append(_leakage_check(train_manifests or [], eval_path if entries is not None else None))

    report = PreflightReport(
        passed=all(check.status != "fail" for check in checks),
        eval_manifest=str(eval_path),
        output_dir=str(out),
        baselines=normalized,
        sample_count=len(entries or []),
        checks=checks,
    )
    _write_preflight_report(report, out)
    return report


def _fallback_model_check(fallback_model_config: str | Path | None) -> PreflightCheck:
    if fallback_model_config is None:
        return PreflightCheck("fallback-model", "pass", "fallback model config is not configured for this preflight")
    model_path = Path(fallback_model_config)
    if not model_path.exists():
        return PreflightCheck("fallback-model", "fail", f"fallback model config does not exist: {model_path}")
    if not model_path.is_file():
        return PreflightCheck("fallback-model", "fail", f"fallback model config is not a file: {model_path}")
    try:
        audit = audit_model_card(model_path)
    except Exception as exc:
        return PreflightCheck("fallback-model", "fail", f"fallback model card audit failed: {exc}")
    if not audit.passed:
        return PreflightCheck("fallback-model", "fail", "fallback model card failed audit", details=[*audit.issues, *audit.warnings])
    return PreflightCheck("fallback-model", "pass", f"fallback model card is ready: {audit.model_id}", details=audit.warnings)


def _candidate_fallback_provenance_check(
    candidate_model_config: str | Path | None,
    fallback_model_config: str | Path | None,
) -> PreflightCheck:
    if fallback_model_config is None:
        return PreflightCheck("candidate-fallback-provenance", "pass", "no fallback model config supplied")
    if candidate_model_config is None:
        return PreflightCheck(
            "candidate-fallback-provenance",
            "warn",
            "fallback model config supplied without candidate model config; candidate fallback provenance skipped",
        )
    candidate_path = Path(candidate_model_config)
    fallback_path = Path(fallback_model_config)
    if not candidate_path.exists():
        return PreflightCheck("candidate-fallback-provenance", "fail", f"candidate model config does not exist: {candidate_path}")
    if not fallback_path.exists() or not fallback_path.is_file():
        return PreflightCheck("candidate-fallback-provenance", "fail", f"fallback model config is not an existing file: {fallback_path}")
    try:
        candidate = ModelCard.from_path(candidate_path)
    except Exception as exc:
        return PreflightCheck("candidate-fallback-provenance", "fail", f"candidate model card could not be loaded: {exc}")
    payload = candidate.provenance.get("fallback_model_card")
    if not isinstance(payload, dict):
        return PreflightCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate model card missing provenance.fallback_model_card for configured fallback",
        )
    path_value = payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return PreflightCheck("candidate-fallback-provenance", "fail", "candidate fallback_model_card.path is missing")
    declared_path = Path(path_value)
    if not declared_path.is_absolute():
        declared_path = candidate_path.parent / declared_path
    if not declared_path.exists() or not declared_path.is_file():
        return PreflightCheck("candidate-fallback-provenance", "fail", f"candidate fallback model card does not exist: {declared_path}")
    actual_declared_sha = sha256_file(declared_path)
    declared_sha = payload.get("sha256")
    if isinstance(declared_sha, str) and declared_sha and declared_sha != actual_declared_sha:
        return PreflightCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate fallback model card sha256 mismatch",
            details=[f"expected={declared_sha}", f"actual={actual_declared_sha}", f"path={declared_path}"],
        )
    configured_sha = sha256_file(fallback_path)
    if actual_declared_sha != configured_sha:
        return PreflightCheck(
            "candidate-fallback-provenance",
            "fail",
            "candidate fallback model card does not match configured fallback model",
            details=[
                f"candidate_fallback={declared_path}",
                f"candidate_fallback_sha256={actual_declared_sha}",
                f"configured_fallback={fallback_path}",
                f"configured_fallback_sha256={configured_sha}",
            ],
        )
    return PreflightCheck(
        "candidate-fallback-provenance",
        "pass",
        "candidate fallback provenance matches configured fallback model",
        details=[f"fallback_model_card={declared_path}", f"sha256={actual_declared_sha}"],
    )


def _manifest_check(eval_path: Path) -> tuple[PreflightCheck, list[ManifestEntry] | None]:
    if not eval_path.exists():
        return PreflightCheck("eval-manifest", "fail", f"eval manifest does not exist: {eval_path}"), None
    try:
        entries = load_manifest(eval_path)
    except Exception as exc:
        return PreflightCheck("eval-manifest", "fail", f"failed to load eval manifest: {exc}"), None
    if not entries:
        return PreflightCheck("eval-manifest", "fail", "eval manifest contains no samples"), entries

    missing_inputs: list[str] = []
    stale_hashes: list[str] = []
    for entry in entries:
        resolved = _resolve_existing_path(entry.image_path, eval_path.parent)
        if resolved is None:
            missing_inputs.append(f"{entry.sample_id}: {entry.image_path}")
            continue
        if entry.sha256 and resolved.is_file():
            actual_sha = sha256_file(resolved)
            if actual_sha != entry.sha256:
                stale_hashes.append(f"{entry.sample_id}: image sha256 mismatch: expected {entry.sha256}, actual {actual_sha}")
        text_sha = entry.metadata.get("text_sha256") if entry.metadata else None
        if isinstance(text_sha, str) and text_sha and text_sha != sha256_text(entry.text):
            stale_hashes.append(f"{entry.sample_id}: text_sha256 mismatch")
    if missing_inputs:
        return PreflightCheck(
            "eval-manifest",
            "fail",
            f"{len(missing_inputs)} eval inputs are missing",
            details=missing_inputs[:25],
        ), entries
    if stale_hashes:
        return PreflightCheck(
            "eval-manifest",
            "fail",
            f"{len(stale_hashes)} eval manifest hashes are stale",
            details=stale_hashes[:25],
        ), entries
    return PreflightCheck("eval-manifest", "pass", f"{len(entries)} eval samples loaded"), entries


def _ocr_input_suffix_check(entries: list[ManifestEntry] | None, manifest_dir: Path) -> PreflightCheck:
    if entries is None:
        return PreflightCheck("eval-input-suffixes", "fail", "cannot verify input suffixes because eval manifest did not load")
    unsupported: list[str] = []
    allowed = set(SUPPORTED_OCR_IMAGE_SUFFIXES) | {".pdf"}
    for entry in entries:
        if str(entry.metadata.get("input_format") or "").lower() == "text":
            continue
        resolved = _resolve_existing_path(entry.image_path, manifest_dir)
        if resolved is None:
            continue
        if resolved.is_dir():
            continue
        suffix = resolved.suffix.lower()
        if suffix not in allowed:
            unsupported.append(f"{entry.sample_id}: {entry.image_path} ({suffix or '<none>'})")
    if unsupported:
        return PreflightCheck(
            "eval-input-suffixes",
            "fail",
            f"{len(unsupported)} eval inputs use unsupported OCR suffixes; run normalize-manifest-images first",
            details=unsupported[:25],
        )
    return PreflightCheck("eval-input-suffixes", "pass", f"all {len(entries)} eval inputs have OCR-supported suffixes or text input_format")


def _reference_check(
    entries: list[ManifestEntry] | None,
    manifest_dir: Path,
    references_root: Path | None,
) -> PreflightCheck:
    if entries is None:
        return PreflightCheck("references", "fail", "cannot verify references because eval manifest did not load")
    if references_root is not None and not references_root.exists():
        return PreflightCheck("references", "fail", f"references directory does not exist: {references_root}")

    missing: list[str] = []
    invalid: list[str] = []
    for entry in entries:
        input_path = _resolve_existing_path(entry.image_path, manifest_dir)
        if input_path is None:
            missing.append(f"{entry.sample_id}: input missing, reference lookup skipped")
            continue
        candidate = _resolve_reference_path(entry, input_path, manifest_dir, references_root)
        if candidate is None:
            missing.append(f"{entry.sample_id}: no reference found")
            continue
        try:
            reference = load_reference(input_path, explicit_path=candidate)
            if reference is None:
                raise DataValidationError("reference did not load")
        except Exception as exc:
            invalid.append(f"{entry.sample_id}: {candidate} ({exc})")
    if missing or invalid:
        details = [*missing[:20], *invalid[:20]]
        summary_parts: list[str] = []
        if missing:
            summary_parts.append(f"{len(missing)} missing")
        if invalid:
            summary_parts.append(f"{len(invalid)} invalid")
        return PreflightCheck("references", "fail", f"reference coverage failed: {', '.join(summary_parts)}", details=details)
    return PreflightCheck("references", "pass", f"references resolved for {len(entries)} samples")


def _gorkhapatra_language_page_review_check(
    entries: list[ManifestEntry] | None,
    review_audit_paths: list[str | Path],
) -> PreflightCheck:
    if entries is None:
        return PreflightCheck("gorkhapatra-language-page-review", "fail", "cannot verify Gorkhapatra review audits because eval manifest did not load")
    language_page_entries = [entry for entry in entries if _is_gorkhapatra_language_page_entry(entry)]
    pending_entries = [entry for entry in language_page_entries if (entry.metadata or {}).get("source_kind") != "gorkhapatra_language_page_verified"]
    if pending_entries:
        return PreflightCheck(
            "gorkhapatra-language-page-review",
            "fail",
            "Gorkhapatra language-page candidates are present but not finalized as verified eval rows",
            details=[f"{entry.sample_id}: source_kind={(entry.metadata or {}).get('source_kind')}" for entry in pending_entries[:25]],
        )
    if not language_page_entries and not review_audit_paths:
        return PreflightCheck("gorkhapatra-language-page-review", "pass", "eval manifest has no Gorkhapatra language-page samples")
    if language_page_entries and not review_audit_paths:
        return PreflightCheck(
            "gorkhapatra-language-page-review",
            "fail",
            "verified Gorkhapatra language-page samples require --gorkhapatra-review-audit evidence",
            details=[entry.sample_id for entry in language_page_entries[:25]],
        )

    issues: list[str] = []
    warnings: list[str] = []
    accepted_total = 0
    verified_total = 0
    for path_value in review_audit_paths:
        path = Path(path_value)
        if not path.is_file():
            issues.append(f"review audit does not exist: {path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"review audit is not valid JSON: {path} ({exc})")
            continue
        if not isinstance(payload, dict):
            issues.append(f"review audit must be a JSON object: {path}")
            continue
        if payload.get("passed") is not True:
            issues.append(f"review audit did not pass: {path}")
        if payload.get("require_verified_references") is not True:
            issues.append(f"review audit was not run with --require-verified-references: {path}")
        accepted = payload.get("accepted_count")
        verified = payload.get("verified_reference_count")
        if not isinstance(accepted, int) or accepted < 1:
            issues.append(f"review audit has no accepted rows: {path}")
            accepted = 0
        if not isinstance(verified, int):
            issues.append(f"review audit missing verified_reference_count: {path}")
            verified = 0
        if isinstance(accepted, int) and isinstance(verified, int) and verified < accepted:
            issues.append(f"review audit verified_reference_count is below accepted_count: {path}")
        accepted_total += int(accepted)
        verified_total += int(verified)
        for warning in payload.get("warnings") or []:
            warnings.append(f"{path}: {warning}")
    if language_page_entries and accepted_total < len(language_page_entries):
        issues.append(
            f"review audit accepted rows ({accepted_total}) fewer than verified Gorkhapatra eval samples ({len(language_page_entries)})"
        )
    if issues:
        return PreflightCheck("gorkhapatra-language-page-review", "fail", "Gorkhapatra language-page review audit evidence is incomplete", details=issues[:30])
    details = [f"accepted={accepted_total}", f"verified_references={verified_total}", *warnings[:20]]
    return PreflightCheck(
        "gorkhapatra-language-page-review",
        "pass",
        f"Gorkhapatra language-page review audit evidence covers {len(language_page_entries)} eval sample(s)",
        details=details,
    )


def _is_gorkhapatra_language_page_entry(entry: ManifestEntry) -> bool:
    metadata = entry.metadata or {}
    source_kind = metadata.get("source_kind")
    if source_kind in {"language_page_candidate", "gorkhapatra_language_page_verified"}:
        return True
    slices = set(_entry_slices(entry))
    return "gorkhapatra" in slices and "language_page" in slices


def _eval_pack_check(
    eval_path: Path,
    entries: list[ManifestEntry] | None,
    audit_output_dir: Path,
    require_claim_ready_eval_pack: bool,
) -> PreflightCheck:
    if entries is None:
        return PreflightCheck("eval-pack", "fail", "cannot audit eval pack because eval manifest did not load")
    if not _looks_like_eval_pack(eval_path, entries):
        return PreflightCheck("eval-pack", "warn", "manifest is not recognized as an ocrtech eval pack; claim-ready slice audit skipped")
    try:
        audit = audit_eval_pack(eval_path, audit_output_dir)
    except Exception as exc:
        return PreflightCheck("eval-pack", "fail", f"eval-pack audit failed: {exc}")
    details = [*audit.issues, *audit.warnings]
    if not audit.passed:
        return PreflightCheck("eval-pack", "fail", "eval-pack integrity audit failed", details=details)
    if require_claim_ready_eval_pack and not audit.claim_ready:
        return PreflightCheck("eval-pack", "fail", "eval pack is not claim-ready for OCR evidence", details=details)
    if not audit.claim_ready:
        return PreflightCheck("eval-pack", "warn", "eval pack is only smoke-ready; OCR claim evidence is disabled", details=details)
    return PreflightCheck("eval-pack", "pass", "eval pack is claim-ready", details=details)


def _baseline_checks(baselines: list[str], candidate_model_config: str | Path | None) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if any(baseline in _CANDIDATE_BASELINES for baseline in baselines):
        if candidate_model_config is None:
            checks.append(PreflightCheck("baseline:candidate", "fail", "candidate baselines require --candidate-model-config"))
        else:
            try:
                audit = audit_model_card(candidate_model_config)
            except Exception as exc:
                checks.append(PreflightCheck("baseline:candidate", "fail", f"candidate model card audit failed: {exc}"))
            else:
                if audit.passed:
                    checks.append(PreflightCheck("baseline:candidate", "pass", f"candidate model card is ready: {audit.model_id}", details=audit.warnings))
                else:
                    checks.append(PreflightCheck("baseline:candidate", "fail", "candidate model card failed audit", details=[*audit.issues, *audit.warnings]))
    elif candidate_model_config is not None:
        try:
            audit = audit_model_card(candidate_model_config)
        except Exception as exc:
            checks.append(PreflightCheck("candidate-model", "warn", f"candidate model card could not be audited: {exc}"))
        else:
            status = "pass" if audit.passed else "warn"
            summary = f"candidate model card {'is ready' if audit.passed else 'has audit issues'}: {audit.model_id}"
            checks.append(PreflightCheck("candidate-model", status, summary, details=[*audit.issues, *audit.warnings]))

    if any(baseline in _TESSERACT_BASELINES for baseline in baselines):
        checks.append(
            PreflightCheck(
                "baseline:tesseract",
                "pass" if shutil.which("tesseract") else "fail",
                "tesseract binary is available" if shutil.which("tesseract") else "tesseract binary is not available on PATH",
            )
        )
    if any(baseline in _PADDLE_BASELINES for baseline in baselines):
        paddleocr_available = importlib.util.find_spec("paddleocr") is not None
        paddle_runtime_available = importlib.util.find_spec("paddle") is not None
        paddle_available = paddleocr_available and paddle_runtime_available
        details: list[str] = []
        if not paddleocr_available:
            details.append("python module missing: paddleocr")
        if not paddle_runtime_available:
            details.append("python module missing: paddle")
        checks.append(
            PreflightCheck(
                "baseline:stock-paddle",
                "pass" if paddle_available else "fail",
                "paddleocr and paddle runtime are importable" if paddle_available else "paddle baseline dependencies are incomplete",
                details=details,
            )
        )
    if any(baseline in _SURYA_BASELINES for baseline in baselines):
        surya_available = importlib.util.find_spec("surya") is not None
        llama_available = shutil.which("llama-server") is not None or bool(os.environ.get("LLAMA_CPP_BINARY"))
        details: list[str] = []
        if not surya_available:
            details.append("python module missing: surya")
        if not llama_available:
            details.append("llama-server is not on PATH and LLAMA_CPP_BINARY is unset")
        checks.append(
            PreflightCheck(
                "baseline:surya",
                "pass" if surya_available and llama_available else "fail",
                "surya runtime is available" if surya_available and llama_available else "surya baseline dependencies are incomplete",
                details=details,
            )
        )
    if any(baseline in _GLM_BASELINES for baseline in baselines):
        checks.append(_external_command_check("baseline:glm-ocr", "OCRTECH_GLM_OCR_CMD"))
    if any(baseline in _PADDLE_VL_BASELINES for baseline in baselines):
        checks.append(_external_command_check("baseline:paddleocr-vl", "OCRTECH_PADDLEOCR_VL_CMD"))
    return checks


def _trained_recognizer_check(candidate_model_config: str | Path | None, required: bool) -> PreflightCheck:
    if not required:
        return PreflightCheck("trained-recognizer", "pass", "trained recognizer provenance is not required for this preflight")
    if candidate_model_config is None:
        return PreflightCheck("trained-recognizer", "fail", "trained recognizer provenance requires --candidate-model-config")
    try:
        card = ModelCard.from_path(candidate_model_config)
        status = trained_recognizer_provenance_status(card, candidate_model_config)
    except Exception as exc:
        return PreflightCheck("trained-recognizer", "fail", f"trained recognizer provenance check failed: {exc}")
    if not status.passed:
        return PreflightCheck("trained-recognizer", "fail", "candidate model lacks trained local recognizer provenance", details=status.details)
    return PreflightCheck("trained-recognizer", "pass", "candidate includes trained local recognizer provenance", details=status.details)


def _model_admission_check(candidate_model_config: str | Path | None, required: bool) -> PreflightCheck:
    if not required:
        return PreflightCheck("model-admission", "pass", "model admission validation is not required for this preflight")
    if candidate_model_config is None:
        return PreflightCheck("model-admission", "fail", "model admission validation requires --candidate-model-config")
    try:
        card = ModelCard.from_path(candidate_model_config)
        status = model_admission_status(card, candidate_model_config)
    except Exception as exc:
        return PreflightCheck("model-admission", "fail", f"model admission validation check failed: {exc}")
    if not status.passed:
        return PreflightCheck("model-admission", "fail", "candidate model has not passed admission validation", details=status.details)
    return PreflightCheck("model-admission", "pass", "candidate model passed admission validation", details=status.details)


def _external_command_check(check_name: str, env_key: str) -> PreflightCheck:
    command = os.environ.get(env_key)
    if not command:
        return PreflightCheck(check_name, "fail", f"{env_key} is not configured")
    details: list[str] = []
    if "{input}" not in command or "{out}" not in command:
        details.append("command template must include both {input} and {out} placeholders")
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return PreflightCheck(check_name, "fail", f"{env_key} is not shell-parseable: {exc}")
    if not parts:
        return PreflightCheck(check_name, "fail", f"{env_key} is empty after shell parsing")
    executable = parts[0]
    if shutil.which(executable) is None:
        details.append(f"executable not found on PATH: {executable}")
    script_path = _command_script_path(parts)
    if script_path is not None and not script_path.exists():
        details.append(f"script path does not exist: {script_path}")
    if details:
        return PreflightCheck(check_name, "fail", f"{env_key} is configured but not runnable", details=details)
    return PreflightCheck(check_name, "pass", f"{env_key} is configured and looks runnable")


def _command_script_path(parts: list[str]) -> Path | None:
    if len(parts) < 2:
        return None
    executable_name = Path(parts[0]).name
    if executable_name not in {"python", "python3", "uv"}:
        return None
    for index, part in enumerate(parts[1:], start=1):
        if executable_name == "uv" and part in {"run", "python"}:
            continue
        if part.startswith("-"):
            continue
        if executable_name == "uv" and part == "--":
            continue
        candidate = Path(part)
        if candidate.suffix == ".py" or "/" in part:
            return candidate
        return None
    return None


def _candidate_calibration_check(eval_manifest: Path, candidate_model_config: str | Path | None) -> PreflightCheck:
    if candidate_model_config is None:
        return PreflightCheck("candidate-calibration", "warn", "no candidate model config supplied; calibration provenance check skipped")
    try:
        card = ModelCard.from_path(candidate_model_config)
    except Exception as exc:
        return PreflightCheck("candidate-calibration", "warn", f"candidate model card could not be loaded for calibration audit: {exc}")
    if card.backend not in {"quality_select_composite", "quality_ranked_ensemble", "script_select_composite"}:
        return PreflightCheck("candidate-calibration", "pass", "candidate backend does not require routed-threshold calibration provenance")
    calibration = card.provenance.get("calibration")
    if not isinstance(calibration, dict):
        return PreflightCheck("candidate-calibration", "fail", f"{card.backend} candidate is missing provenance.calibration")
    eval_payload = calibration.get("eval_manifest")
    if not isinstance(eval_payload, dict):
        return PreflightCheck("candidate-calibration", "fail", f"{card.backend} calibration is missing eval_manifest provenance")
    calibration_sha = eval_payload.get("sha256")
    if not isinstance(calibration_sha, str) or not calibration_sha:
        return PreflightCheck("candidate-calibration", "fail", f"{card.backend} calibration eval_manifest is missing sha256")
    current_sha = sha256_file(eval_manifest) if eval_manifest.exists() else None
    if current_sha == calibration_sha:
        return PreflightCheck(
            "candidate-calibration",
            "fail",
            f"{card.backend} calibration eval manifest matches the claim eval manifest",
            details=[str(eval_payload.get("path") or "")],
        )
    threshold = calibration.get("selected_threshold")
    details = []
    if isinstance(threshold, int | float):
        details.append(f"selected_threshold={float(threshold):.6f}")
    metric = calibration.get("metric")
    if isinstance(metric, str) and metric:
        details.append(f"metric={metric}")
    collapse_reason, collapse_details = inspect_routed_calibration_collapse(card, candidate_model_config)
    if collapse_reason is not None:
        return PreflightCheck("candidate-calibration", "fail", collapse_reason, details=[*details, *collapse_details])
    return PreflightCheck("candidate-calibration", "pass", f"{card.backend} calibration provenance is separate from claim eval", details=details)


def _candidate_component_baseline_check(baselines: list[str], candidate_model_config: str | Path | None) -> PreflightCheck:
    if candidate_model_config is None:
        return PreflightCheck("candidate-components", "warn", "no candidate model config supplied; component-baseline coverage check skipped")
    try:
        card = ModelCard.from_path(candidate_model_config)
    except Exception as exc:
        return PreflightCheck("candidate-components", "warn", f"candidate model card could not be loaded for component-baseline audit: {exc}")
    components = component_baseline_names(card)
    if not components:
        return PreflightCheck("candidate-components", "pass", "candidate backend does not require component-baseline coverage")
    missing = [name for name in components if name not in baselines]
    if missing:
        return PreflightCheck(
            "candidate-components",
            "fail",
            "claim-grade benchmark must include the candidate's component baselines",
            details=[f"missing: {', '.join(missing)}", f"required: {', '.join(components)}", f"configured: {', '.join(baselines)}"],
        )
    return PreflightCheck(
        "candidate-components",
        "pass",
        "claim-grade benchmark includes the candidate's component baselines",
        details=components,
    )


def _validation_config_check(
    validation_config: str | Path | None,
    baselines: list[str],
    entries: list[ManifestEntry] | None,
) -> PreflightCheck:
    if validation_config is None:
        return PreflightCheck("validation-config", "warn", "validation config was not supplied; claim gates will not be preflighted")
    try:
        config = ValidationConfig.from_path(validation_config)
    except Exception as exc:
        return PreflightCheck("validation-config", "fail", f"validation config failed to load: {exc}")
    if not config.gates:
        return PreflightCheck("validation-config", "fail", "validation config has no gates; claim validation cannot pass")
    issues: list[str] = []
    baseline_set = set(baselines)
    if not _baseline_is_configured(config.candidate, baseline_set):
        issues.append(f"candidate baseline {config.candidate!r} is not included in preflight baselines")
    missing_gate_baselines = sorted({gate.baseline for gate in config.gates if not _baseline_is_configured(gate.baseline, baseline_set)})
    if missing_gate_baselines:
        issues.append(f"gate baselines not included in preflight baselines: {', '.join(missing_gate_baselines)}")
    if entries is not None:
        sample_count = len(entries)
        if config.min_samples > sample_count:
            issues.append(f"validation min_samples {config.min_samples} exceeds eval manifest sample count {sample_count}")
        slice_counts = _entry_slice_counts(entries)
        for slice_name, required_count in config.required_slices.items():
            actual = slice_counts.get(slice_name, 0)
            if required_count > actual:
                issues.append(f"required slice {slice_name!r} count {required_count} exceeds eval manifest count {actual}")
        for gate in config.gates:
            gate_slices = gate.slices or ([gate.slice] if gate.slice else [])
            if gate_slices:
                available_pairs = _entry_intersection_count(entries, [str(item) for item in gate_slices])
            else:
                available_pairs = sample_count
            if gate.min_pairs > available_pairs:
                suffix = f" on slices {','.join(gate_slices)}" if gate_slices else ""
                issues.append(
                    f"gate {gate.baseline}:{gate.metric}{suffix} min_pairs {gate.min_pairs} exceeds eval manifest capacity {available_pairs}"
                )
    if issues:
        return PreflightCheck("validation-config", "fail", "validation config cannot be satisfied by this preflight setup", details=issues)
    return PreflightCheck(
        "validation-config",
        "pass",
        f"validation config loaded with {len(config.gates)} gate(s)",
        details=[f"candidate={config.candidate}", f"min_samples={config.min_samples}"],
    )


def _baseline_is_configured(name: str, configured: set[str]) -> bool:
    normalized = _normalize_baseline(name)
    if normalized in _CANDIDATE_BASELINES:
        return bool(configured & _CANDIDATE_BASELINES)
    return normalized in configured


def _entry_slice_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for slice_name in _entry_slices(entry):
            counts[slice_name] = counts.get(slice_name, 0) + 1
    return counts


def _entry_intersection_count(entries: list[ManifestEntry], required_slices: list[str]) -> int:
    required = {str(item) for item in required_slices if str(item)}
    if not required:
        return len(entries)
    return sum(1 for entry in entries if required.issubset(set(_entry_slices(entry))))


def _entry_slices(entry: ManifestEntry) -> list[str]:
    metadata = entry.metadata or {}
    values: set[str] = set()
    for key in ("slice", "script", "document_type", "degradation", "language"):
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


def _leakage_check(train_manifests: list[str | Path], eval_manifest: Path | None) -> PreflightCheck:
    if not train_manifests:
        return PreflightCheck("leakage", "warn", "train manifests were not provided; leakage audit skipped")
    if eval_manifest is None:
        return PreflightCheck("leakage", "fail", "eval manifest is unavailable; leakage audit cannot run")
    try:
        report = check_leakage(train_manifests, eval_manifest)
    except Exception as exc:
        return PreflightCheck("leakage", "fail", f"leakage audit failed: {exc}")
    details: list[str] = []
    for key, values in report.overlaps.items():
        if values:
            details.append(f"{key}: {', '.join(values[:10])}")
    for key, values in report.warnings.items():
        if values:
            details.append(f"{key} warning: {', '.join(values[:10])}")
    if not report.passed:
        return PreflightCheck("leakage", "fail", "train/eval leakage detected", details=details)
    if any(report.warnings.values()):
        return PreflightCheck("leakage", "warn", "no fatal leakage detected; warning overlaps are present", details=details)
    return PreflightCheck("leakage", "pass", "no train/eval leakage detected")


def _looks_like_eval_pack(eval_path: Path, entries: list[ManifestEntry]) -> bool:
    if "eval-pack" in eval_path.name:
        return True
    datasets = {entry.dataset for entry in entries}
    if datasets == {"ocrtech-eval-pack"}:
        return True
    return any(isinstance(entry.metadata.get("reference_path"), str) and "references" in str(entry.metadata.get("reference_path")) for entry in entries)


def _resolve_existing_path(path_value: str, manifest_dir: Path) -> Path | None:
    raw = Path(path_value)
    if raw.exists():
        return raw
    if not raw.is_absolute():
        candidate = manifest_dir / raw
        if candidate.exists():
            return candidate
    return None


def _resolve_reference_path(
    entry: ManifestEntry,
    input_path: Path,
    manifest_dir: Path,
    references_root: Path | None,
) -> Path | None:
    explicit = entry.metadata.get("reference_path") if entry.metadata else None
    if isinstance(explicit, str) and explicit:
        resolved = _resolve_existing_path(explicit, manifest_dir)
        if resolved is not None:
            return resolved
        return None
    candidates: list[Path] = []
    if references_root:
        candidates.extend(
            [
                references_root / f"{input_path.stem}.json",
                references_root / input_path.name / "reference.json",
                references_root / input_path.stem / "reference.json",
                references_root / f"{input_path.stem}.txt",
                references_root / input_path.name / "document.md",
                references_root / input_path.stem / "document.md",
            ]
        )
    candidates.extend([input_path.with_suffix(input_path.suffix + ".ref.json"), input_path.with_suffix(".ref.json")])
    candidates.extend([input_path.with_suffix(input_path.suffix + ".ref.txt"), input_path.with_suffix(".ref.txt")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_baseline(baseline: str) -> str:
    normalized = baseline.strip().lower()
    if not normalized:
        raise PreflightError("baseline name cannot be empty")
    return normalized


def _write_preflight_report(report: PreflightReport, output_dir: Path) -> None:
    (output_dir / "preflight.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR Claim Preflight",
        "",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Eval manifest: `{report.eval_manifest}`",
        f"Samples: `{report.sample_count}`",
        f"Baselines: `{', '.join(report.baselines)}`",
        "",
        "| check | status | summary |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        summary = check.summary.replace("|", "\\|")
        lines.append(f"| {check.name} | {check.status} | {summary} |")
        for detail in check.details:
            escaped_detail = detail.replace("|", "\\|")
            lines.append(f"|  |  | {escaped_detail} |")
    (output_dir / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
