"""Runtime preflight for the Limbu-first operational OCR pipeline."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engines import SCRIPT_CODEPOINT_RANGES
from .errors import PreflightError
from .models import ModelCard, audit_model_card


@dataclass(slots=True)
class LimbuRuntimeCheck:
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
class LimbuRuntimePreflightReport:
    model_config: str
    output_dir: str
    passed: bool
    policy: dict[str, Any]
    checks: list[LimbuRuntimeCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "output_dir": self.output_dir,
            "passed": self.passed,
            "policy": self.policy,
            "checks": [check.to_dict() for check in self.checks],
        }


def preflight_limbu_pipeline_runtime(
    model_config: str | Path,
    output_dir: str | Path,
    *,
    required_script: str = "limbu",
    required_tesseract_languages: tuple[str, ...] = ("nep", "eng"),
    required_tesseract_langs: tuple[str, ...] | None = None,
    require_validated_component_admissions: bool = False,
    required_component_roles: tuple[str, ...] = (),
) -> LimbuRuntimePreflightReport:
    normalized_required_script = _normalize_required_script(required_script)
    normalized_required_tesseract_languages = _normalize_required_tesseract_languages(
        required_tesseract_languages=required_tesseract_languages,
        required_tesseract_langs=required_tesseract_langs,
    )
    config_path = Path(model_config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks: list[LimbuRuntimeCheck] = []
    checks.append(_check_python_runtime())
    checks.append(_check_router_model(config_path, required_script=normalized_required_script))
    checks.extend(_check_deployment_advisories(config_path, required_script=normalized_required_script))
    checks.extend(
        _check_component_admission(
            config_path,
            required_script=normalized_required_script,
            require_validated=bool(require_validated_component_admissions),
            required_roles=tuple(required_component_roles),
        )
    )
    checks.extend(_check_component_model_cards(config_path))
    checks.extend(_check_python_packages())
    checks.extend(_check_tesseract(normalized_required_tesseract_languages))

    passed = not any(check.status == "fail" for check in checks)
    report = LimbuRuntimePreflightReport(
        model_config=str(config_path),
        output_dir=str(out),
        passed=passed,
        policy={
            "required_script": normalized_required_script,
            "required_tesseract_languages": list(normalized_required_tesseract_languages),
            "require_validated_component_admissions": bool(require_validated_component_admissions),
            "required_component_roles": list(required_component_roles),
        },
        checks=checks,
    )
    (out / "limbu-runtime-preflight.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "limbu-runtime-preflight.md").write_text(_render_runtime_preflight_markdown(report), encoding="utf-8")
    return report


def _normalize_required_script(required_script: str) -> str:
    supported = ", ".join(sorted(SCRIPT_CODEPOINT_RANGES))
    if not isinstance(required_script, str):
        raise PreflightError(
            f"required_script must be a string, got {type(required_script).__name__}; supported scripts: {supported}"
        )
    normalized = required_script.strip().lower()
    if not normalized:
        raise PreflightError(f"required_script cannot be empty; supported scripts: {supported}")
    if normalized not in SCRIPT_CODEPOINT_RANGES:
        raise PreflightError(f"unsupported required_script {required_script!r}; supported scripts: {supported}")
    return normalized


def _normalize_required_tesseract_languages(
    *,
    required_tesseract_languages: tuple[str, ...],
    required_tesseract_langs: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if required_tesseract_langs is None:
        return tuple(required_tesseract_languages)
    if tuple(required_tesseract_languages) != ("nep", "eng") and tuple(required_tesseract_languages) != tuple(required_tesseract_langs):
        raise PreflightError(
            "required_tesseract_languages and required_tesseract_langs were both provided with different values"
        )
    return tuple(required_tesseract_langs)


def _check_router_model(config_path: Path, *, required_script: str) -> LimbuRuntimeCheck:
    try:
        audit = audit_model_card(config_path)
    except Exception as exc:
        return LimbuRuntimeCheck(
            name="router-model-card",
            status="fail",
            summary=f"router model card could not be audited: {type(exc).__name__}: {exc}",
        )
    details = [*audit.issues, *audit.warnings]
    if not audit.passed:
        return LimbuRuntimeCheck(
            name="router-model-card",
            status="fail",
            summary=f"router model card failed audit: {audit.model_id}",
            details=details,
        )
    try:
        card = ModelCard.from_path(config_path)
    except Exception as exc:
        return LimbuRuntimeCheck(
            name="router-model-card",
            status="fail",
            summary=f"router model card shape could not be loaded: {type(exc).__name__}: {exc}",
            details=details,
        )
    if card.backend != "script_select_composite":
        return LimbuRuntimeCheck(
            name="router-model-card",
            status="fail",
            summary=f"{required_script} pipeline requires script_select_composite backend, got {card.backend}",
            details=details,
        )
    kwargs = card.backend_kwargs
    backend_issues: list[str] = []
    if kwargs.get("script") != required_script:
        backend_issues.append(f"backend_kwargs.script must be {required_script}, got {kwargs.get('script')}")
    if kwargs.get("routing_granularity") != "line":
        backend_issues.append("backend_kwargs.routing_granularity should be line for mixed-script pages")
    if kwargs.get("routing_signal_engine") not in {"secondary", "both"}:
        backend_issues.append("backend_kwargs.routing_signal_engine should be secondary or both for script routing")
    if backend_issues:
        return LimbuRuntimeCheck(
            name="router-model-card",
            status="fail",
            summary=f"router model card is not configured for {required_script} routing",
            details=[*details, *backend_issues],
        )
    return LimbuRuntimeCheck(
        name="router-model-card",
        status="pass",
        summary=f"router model card is structurally ready: {audit.model_id}",
        details=details,
    )


def _check_component_model_cards(config_path: Path) -> list[LimbuRuntimeCheck]:
    try:
        card = ModelCard.from_path(config_path)
    except Exception:
        return []
    checks: list[LimbuRuntimeCheck] = []
    for artifact in card.artifacts:
        if not artifact.path.endswith("model-card.json"):
            continue
        nested_path = _resolve_model_path(artifact.path, config_path)
        try:
            audit = audit_model_card(nested_path)
        except Exception as exc:
            checks.append(
                LimbuRuntimeCheck(
                    name="component-model-card",
                    status="fail",
                    summary=f"component model card could not be audited: {nested_path}",
                    details=[f"{type(exc).__name__}: {exc}"],
                )
            )
            continue
        checks.append(
            LimbuRuntimeCheck(
                name="component-model-card",
                status="pass" if audit.passed else "fail",
                summary=f"component model card {'passed' if audit.passed else 'failed'} audit: {audit.model_id}",
                details=[str(nested_path), *audit.issues, *audit.warnings],
            )
        )
    if not checks:
        checks.append(
            LimbuRuntimeCheck(
                name="component-model-card",
                status="warn",
                summary="router model card does not list a nested component model-card artifact",
            )
        )
    return checks


def _check_component_admission(
    config_path: Path,
    *,
    required_script: str,
    require_validated: bool = False,
    required_roles: tuple[str, ...] = (),
) -> list[LimbuRuntimeCheck]:
    try:
        card = ModelCard.from_path(config_path)
    except Exception:
        return []
    raw_admissions = card.provenance.get("component_admission")
    if raw_admissions is None:
        if require_validated or required_roles:
            return [
                LimbuRuntimeCheck(
                    name="component-admission",
                    status="fail",
                    summary=f"provenance.component_admission is required for strict {required_script} pipeline preflight",
                    details=[f"required_roles={','.join(required_roles)}"] if required_roles else [],
                )
            ]
        return []
    malformed_entries = 0
    if isinstance(raw_admissions, dict):
        admissions = [raw_admissions]
    elif isinstance(raw_admissions, list):
        admissions = []
        for item in raw_admissions:
            if isinstance(item, dict):
                admissions.append(item)
            else:
                malformed_entries += 1
    else:
        return [
            LimbuRuntimeCheck(
                name="component-admission",
                status="fail",
                summary="provenance.component_admission must be an object or list of objects",
            )
        ]
    checks: list[LimbuRuntimeCheck] = []
    if malformed_entries:
        checks.append(
            LimbuRuntimeCheck(
                name="component-admission",
                status="fail",
                summary=f"provenance.component_admission contains {malformed_entries} non-object entries",
            )
        )
    if require_validated and not admissions:
        checks.append(
            LimbuRuntimeCheck(
                name="component-admission",
                status="fail",
                summary=f"strict {required_script} pipeline preflight requires at least one component admission object",
            )
        )
    observed_roles: set[str] = set()
    for index, admission in enumerate(admissions, start=1):
        role = str(admission.get("role") or admission.get("component") or f"component-{index}")
        observed_roles.add(role)
        status_value = str(admission.get("status") or "").strip().lower()
        details = _component_admission_details(admission)
        if status_value in {"fail", "failed", "rejected", "blocked"}:
            checks.append(
                LimbuRuntimeCheck(
                    name=f"component-admission:{role}",
                    status="fail",
                    summary=f"{role} is explicitly not admitted for the {required_script} pipeline",
                    details=details,
                )
            )
            continue
        if status_value in {"warn", "warning", "experimental", "pending"}:
            checks.append(
                LimbuRuntimeCheck(
                    name=f"component-admission:{role}",
                    status="fail" if require_validated else "warn",
                    summary=(
                        f"{role} admission is {status_value}; strict preflight requires validated/admitted status"
                        if require_validated
                        else f"{role} admission is {status_value}"
                    ),
                    details=details,
                )
            )
            continue
        if status_value in {"pass", "passed", "admitted", "validated"}:
            checks.append(
                LimbuRuntimeCheck(
                    name=f"component-admission:{role}",
                    status="pass",
                    summary=f"{role} admission is {status_value}",
                    details=details,
                )
            )
            continue
        checks.append(
            LimbuRuntimeCheck(
                name=f"component-admission:{role}",
                status="fail" if require_validated else "warn",
                summary=(
                    f"{role} admission status is not recognized: {status_value or 'missing'}; "
                    "strict preflight requires validated/admitted status"
                    if require_validated
                    else f"{role} admission status is not recognized: {status_value or 'missing'}"
                ),
                details=details,
            )
        )
    for role in required_roles:
        if role not in observed_roles:
            checks.append(
                LimbuRuntimeCheck(
                    name=f"component-admission:{role}",
                    status="fail",
                    summary=f"required component admission role is missing: {role}",
                    details=[f"observed_roles={','.join(sorted(observed_roles)) or '<none>'}"],
                )
            )
    return checks


def _component_admission_details(admission: dict[str, Any]) -> list[str]:
    details: list[str] = []
    for key in (
        "role",
        "component",
        "status",
        "engine",
        "model_name",
        "eval_pack",
        "score_output",
        "metrics_file",
        "audit_file",
        "reason",
        "required_action",
    ):
        value = admission.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            details.append(f"{key}={value}")
    metrics = admission.get("metrics")
    if isinstance(metrics, dict):
        for key, value in sorted(metrics.items()):
            if isinstance(value, str | int | float | bool) and value not in ("", None):
                details.append(f"metrics.{key}={value}")
    return details


def _check_deployment_advisories(config_path: Path, *, required_script: str) -> list[LimbuRuntimeCheck]:
    try:
        card = ModelCard.from_path(config_path)
    except Exception:
        return []
    advisories: list[LimbuRuntimeCheck] = []
    stronger_checkpoint = card.provenance.get("stronger_unexported_checkpoint")
    if isinstance(stronger_checkpoint, dict):
        details = _deployment_advisory_details(stronger_checkpoint)
        archive_path = stronger_checkpoint.get("archive_path")
        archive_status = ""
        if isinstance(archive_path, str) and archive_path:
            resolved = _resolve_model_path(archive_path, config_path)
            archive_status = "present" if resolved.is_file() else "missing"
            details.append(f"archive_status={archive_status}:{resolved}")
        status = "warn"
        summary = f"stronger {required_script} checkpoint is available but is not exported into the deployed router"
        if archive_status == "missing":
            summary = f"stronger {required_script} checkpoint is recorded but its archive is missing locally"
        advisories.append(
            LimbuRuntimeCheck(
                name="deployment-advisory:stronger-unexported-checkpoint",
                status=status,
                summary=summary,
                details=details,
            )
        )
    return advisories


def _deployment_advisory_details(advisory: dict[str, Any]) -> list[str]:
    details: list[str] = []
    for key in (
        "model_id",
        "archive_path",
        "archive_sha256",
        "checkpoint_member",
        "training_config_member",
        "deployment_status",
        "required_action",
    ):
        value = advisory.get(key)
        if isinstance(value, str) and value:
            details.append(f"{key}={value}")
    metrics = advisory.get("reported_metrics")
    if isinstance(metrics, dict):
        for key, value in sorted(metrics.items()):
            details.append(f"reported_metrics.{key}={value}")
    return details


def _check_python_runtime() -> LimbuRuntimeCheck:
    version = sys.version_info
    status = "pass" if (version.major, version.minor) <= (3, 13) else "warn"
    return LimbuRuntimeCheck(
        name="python-runtime",
        status=status,
        summary=f"Python {version.major}.{version.minor}.{version.micro} at {sys.executable}",
        details=[
            f"version={sys.version.split()[0]}",
            f"executable={sys.executable}",
            f"platform={sys.platform}",
        ],
    )


def _check_python_packages() -> list[LimbuRuntimeCheck]:
    checks = [
        _check_python_package("PIL", "Pillow image preparation"),
        _check_python_package("cv2", "OpenCV perspective rectification"),
        _check_python_package("numpy", "OpenCV array conversion"),
        _check_python_package("paddle", "Paddle runtime for the secondary recognizer"),
        _check_python_package("paddleocr", "PaddleOCR secondary recognizer runtime"),
    ]
    return checks


def _check_python_package(module_name: str, purpose: str) -> LimbuRuntimeCheck:
    found = importlib.util.find_spec(module_name) is not None
    return LimbuRuntimeCheck(
        name=f"python-package:{module_name}",
        status="pass" if found else "fail",
        summary=f"{module_name} {'is importable' if found else 'is not importable'} for {purpose}",
    )


def _check_tesseract(required_languages: tuple[str, ...]) -> list[LimbuRuntimeCheck]:
    binary = shutil.which("tesseract")
    if binary is None:
        return [
            LimbuRuntimeCheck(
                name="tesseract-binary",
                status="fail",
                summary="tesseract binary is not on PATH",
            ),
            LimbuRuntimeCheck(
                name="tesseract-languages",
                status="fail",
                summary="cannot check Tesseract language data because tesseract is unavailable",
                details=list(required_languages),
            ),
        ]
    binary_check = LimbuRuntimeCheck(
        name="tesseract-binary",
        status="pass",
        summary=f"tesseract binary found: {binary}",
    )
    try:
        completed = subprocess.run(
            [binary, "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return [
            binary_check,
            LimbuRuntimeCheck(
                name="tesseract-languages",
                status="fail",
                summary=f"failed to list Tesseract languages: {type(exc).__name__}: {exc}",
            ),
        ]
    output = "\n".join([completed.stdout or "", completed.stderr or ""])
    available = {
        line.strip()
        for line in output.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    }
    missing = [language for language in required_languages if language not in available]
    status = "pass" if completed.returncode == 0 and not missing else "fail"
    details = [f"available={','.join(sorted(available)) or '<none>'}"]
    if completed.returncode != 0:
        details.append(f"returncode={completed.returncode}")
    if missing:
        details.append(f"missing={','.join(missing)}")
    return [
        binary_check,
        LimbuRuntimeCheck(
            name="tesseract-languages",
            status=status,
            summary="required Tesseract languages are installed" if status == "pass" else "required Tesseract languages are missing",
            details=details,
        ),
    ]


def _resolve_model_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return config_path.parent / candidate


def _render_runtime_preflight_markdown(report: LimbuRuntimePreflightReport) -> str:
    lines = [
        "# Limbu Runtime Preflight",
        "",
        f"Model config: `{report.model_config}`",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Required script: `{report.policy.get('required_script') or 'none'}`",
        f"Require validated components: `{'yes' if report.policy.get('require_validated_component_admissions') else 'no'}`",
        f"Required component roles: `{', '.join(report.policy.get('required_component_roles') or []) or 'none'}`",
        f"Required Tesseract languages: `{', '.join(report.policy.get('required_tesseract_languages') or []) or 'none'}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.checks:
        lines.append(f"- `{check.status}` `{check.name}`: {check.summary}")
        for detail in check.details:
            lines.append(f"  - {detail}")
    lines.append("")
    return "\n".join(lines)
