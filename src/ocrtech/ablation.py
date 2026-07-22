"""Ablation-plan audits for research-grade OCR training branches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import ManifestEntry, load_manifest


@dataclass(slots=True)
class AugmentationAblationArmAudit:
    name: str
    profiles: list[str]
    train_manifest: str
    eval_manifest: str
    audit_path: str
    train_count: int | None
    eval_count: int | None
    train_manifest_count: int | None = None
    eval_manifest_count: int | None = None
    train_claim_eligible_count: int = 0
    eval_claim_eligible_count: int = 0
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profiles": self.profiles,
            "train_manifest": self.train_manifest,
            "eval_manifest": self.eval_manifest,
            "audit_path": self.audit_path,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "train_manifest_count": self.train_manifest_count,
            "eval_manifest_count": self.eval_manifest_count,
            "train_claim_eligible_count": self.train_claim_eligible_count,
            "eval_claim_eligible_count": self.eval_claim_eligible_count,
            "passed": self.passed,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class AugmentationAblationAudit:
    config_path: str
    output_json_path: str
    output_md_path: str
    ablation_id: str
    passed: bool
    promotion_ready: bool
    arm_count: int
    real_review_audits: list[str]
    real_evidence_count: int
    verified_reference_count: int
    arms: list[AugmentationAblationArmAudit]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "output_json_path": self.output_json_path,
            "output_md_path": self.output_md_path,
            "ablation_id": self.ablation_id,
            "passed": self.passed,
            "promotion_ready": self.promotion_ready,
            "arm_count": self.arm_count,
            "real_review_audits": self.real_review_audits,
            "real_evidence_count": self.real_evidence_count,
            "verified_reference_count": self.verified_reference_count,
            "arms": [arm.to_dict() for arm in self.arms],
            "issues": self.issues,
            "warnings": self.warnings,
        }


def audit_augmentation_ablation(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    real_review_audits: list[str | Path] | None = None,
) -> AugmentationAblationAudit:
    config_file = Path(config_path)
    config = _read_json_object(config_file, "augmentation ablation config")
    base_dir = config_file.parent
    issues: list[str] = []
    warnings: list[str] = []

    ablation_id = _clean_str(config.get("ablation_id")) or config_file.stem
    claim_policy = config.get("claim_policy")
    if not isinstance(claim_policy, dict):
        issues.append("claim_policy must be an object")
    elif claim_policy.get("claim_evidence_eligible") is not False:
        issues.append("claim_policy.claim_evidence_eligible must be false for synthetic ablation configs")

    source_bundle = config.get("source_bundle")
    if not isinstance(source_bundle, dict):
        issues.append("source_bundle must be an object")
    else:
        _audit_optional_passed_json(
            source_bundle.get("full_split_audit"),
            base_dir,
            "source_bundle.full_split_audit",
            issues,
            warnings,
            require_zero_issues=True,
        )
        font_readiness = source_bundle.get("font_readiness_report")
        if font_readiness:
            _audit_optional_passed_json(font_readiness, base_dir, "source_bundle.font_readiness_report", issues, warnings)

    arms_payload = config.get("arms")
    arms: list[AugmentationAblationArmAudit] = []
    if not isinstance(arms_payload, list) or not arms_payload:
        issues.append("arms must be a non-empty list")
    else:
        for index, payload in enumerate(arms_payload, start=1):
            if not isinstance(payload, dict):
                arm = AugmentationAblationArmAudit(
                    name=f"arm-{index}",
                    profiles=[],
                    train_manifest="",
                    eval_manifest="",
                    audit_path="",
                    train_count=None,
                    eval_count=None,
                    issues=[f"arms[{index}] must be an object"],
                )
                arms.append(arm)
                continue
            arms.append(_audit_ablation_arm(payload, base_dir, index))

    real_paths = [Path(path) for path in (real_review_audits or [])]
    real_issues, real_warnings, real_evidence_count, verified_reference_count = _audit_real_review_evidence(real_paths)
    issues.extend(real_issues)
    warnings.extend(real_warnings)

    for arm in arms:
        issues.extend(f"{arm.name}: {issue}" for issue in arm.issues)
        warnings.extend(f"{arm.name}: {warning}" for warning in arm.warnings)

    promotion_ready = not issues and real_evidence_count > 0 and verified_reference_count > 0 and all(arm.passed for arm in arms)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = AugmentationAblationAudit(
        config_path=str(config_file),
        output_json_path=str(out / "augmentation-ablation-audit.json"),
        output_md_path=str(out / "augmentation-ablation-audit.md"),
        ablation_id=ablation_id,
        passed=promotion_ready,
        promotion_ready=promotion_ready,
        arm_count=len(arms),
        real_review_audits=[str(path) for path in real_paths],
        real_evidence_count=real_evidence_count,
        verified_reference_count=verified_reference_count,
        arms=arms,
        issues=issues,
        warnings=warnings,
    )
    _write_augmentation_ablation_audit(report)
    return report


def _audit_ablation_arm(payload: dict[str, Any], base_dir: Path, index: int) -> AugmentationAblationArmAudit:
    issues: list[str] = []
    warnings: list[str] = []
    name = _clean_str(payload.get("name")) or f"arm-{index}"
    profiles = [str(item) for item in payload.get("profiles") or [] if str(item)]
    if not profiles:
        issues.append("profiles must be a non-empty list")
    train_manifest = _clean_str(payload.get("train_manifest"))
    eval_manifest = _clean_str(payload.get("eval_manifest"))
    audit_path = _clean_str(payload.get("audit"))
    train_count = _optional_int(payload.get("train_count"))
    eval_count = _optional_int(payload.get("eval_count"))

    train_manifest_count, train_claims = _audit_arm_manifest(train_manifest, base_dir, "train_manifest", train_count, issues)
    eval_manifest_count, eval_claims = _audit_arm_manifest(eval_manifest, base_dir, "eval_manifest", eval_count, issues)
    _audit_optional_passed_json(audit_path, base_dir, "audit", issues, warnings, require_zero_issues=True)

    if train_claims:
        issues.append(f"train_manifest has {train_claims} claim_evidence_eligible synthetic row(s)")
    if eval_claims:
        issues.append(f"eval_manifest has {eval_claims} claim_evidence_eligible synthetic row(s)")
    return AugmentationAblationArmAudit(
        name=name,
        profiles=profiles,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        audit_path=audit_path,
        train_count=train_count,
        eval_count=eval_count,
        train_manifest_count=train_manifest_count,
        eval_manifest_count=eval_manifest_count,
        train_claim_eligible_count=train_claims,
        eval_claim_eligible_count=eval_claims,
        passed=not issues,
        issues=issues,
        warnings=warnings,
    )


def _audit_arm_manifest(
    path_value: str,
    base_dir: Path,
    field_name: str,
    expected_count: int | None,
    issues: list[str],
) -> tuple[int | None, int]:
    if not path_value:
        issues.append(f"{field_name} is required")
        return None, 0
    path = _resolve_config_path(path_value, base_dir)
    if not path.is_file():
        issues.append(f"{field_name} does not exist: {path}")
        return None, 0
    try:
        entries = load_manifest(path)
    except Exception as exc:
        issues.append(f"{field_name} failed to load: {path}: {exc}")
        return None, 0
    if expected_count is not None and len(entries) != expected_count:
        issues.append(f"{field_name} count mismatch: expected {expected_count}, found {len(entries)}")
    claim_count = sum(1 for entry in entries if _claim_evidence_eligible(entry))
    return len(entries), claim_count


def _audit_real_review_evidence(paths: list[Path]) -> tuple[list[str], list[str], int, int]:
    issues: list[str] = []
    warnings: list[str] = []
    accepted_total = 0
    verified_total = 0
    if not paths:
        issues.append("at least one real Gorkhapatra review audit is required before ablation promotion")
        return issues, warnings, accepted_total, verified_total
    for path in paths:
        try:
            payload = _read_json_object(path, "Gorkhapatra review audit")
        except DataValidationError as exc:
            issues.append(str(exc))
            continue
        if payload.get("passed") is not True:
            issues.append(f"real review audit did not pass: {path}")
        if payload.get("require_verified_references") is not True:
            issues.append(f"real review audit was not run with --require-verified-references: {path}")
        accepted = _optional_int(payload.get("accepted_count")) or 0
        verified = _optional_int(payload.get("verified_reference_count")) or 0
        if accepted < 1:
            issues.append(f"real review audit has no accepted rows: {path}")
        if verified < accepted:
            issues.append(f"real review audit verified_reference_count is below accepted_count: {path}")
        accepted_total += accepted
        verified_total += verified
        for warning in payload.get("warnings") or []:
            warnings.append(f"{path}: {warning}")
    return issues, warnings, accepted_total, verified_total


def _audit_optional_passed_json(
    path_value: Any,
    base_dir: Path,
    field_name: str,
    issues: list[str],
    warnings: list[str],
    *,
    require_zero_issues: bool = False,
) -> None:
    path_string = _clean_str(path_value)
    if not path_string:
        issues.append(f"{field_name} is required")
        return
    path = _resolve_config_path(path_string, base_dir)
    try:
        payload = _read_json_object(path, field_name)
    except DataValidationError as exc:
        issues.append(str(exc))
        return
    if payload.get("passed") is not True:
        issues.append(f"{field_name} did not pass: {path}")
    if require_zero_issues and payload.get("issues"):
        issues.append(f"{field_name} has issues: {path}")
    for warning in payload.get("warnings") or []:
        warnings.append(f"{field_name}: {path}: {warning}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_config_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _claim_evidence_eligible(entry: ManifestEntry) -> bool:
    return (entry.metadata or {}).get("claim_evidence_eligible") is not False


def _write_augmentation_ablation_audit(report: AugmentationAblationAudit) -> None:
    json_path = Path(report.output_json_path)
    md_path = Path(report.output_md_path)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Augmentation Ablation Audit",
        "",
        f"- ablation: `{report.ablation_id}`",
        f"- passed: `{report.passed}`",
        f"- promotion ready: `{report.promotion_ready}`",
        f"- arms: `{report.arm_count}`",
        f"- real review audits: `{len(report.real_review_audits)}`",
        f"- real accepted rows: `{report.real_evidence_count}`",
        f"- verified references: `{report.verified_reference_count}`",
        "",
        "## Arms",
        "",
    ]
    for arm in report.arms:
        lines.extend(
            [
                f"### {arm.name}",
                "",
                f"- passed: `{arm.passed}`",
                f"- profiles: `{', '.join(arm.profiles)}`",
                f"- train rows: `{arm.train_manifest_count if arm.train_manifest_count is not None else 'unknown'}`",
                f"- eval rows: `{arm.eval_manifest_count if arm.eval_manifest_count is not None else 'unknown'}`",
                f"- train claim eligible rows: `{arm.train_claim_eligible_count}`",
                f"- eval claim eligible rows: `{arm.eval_claim_eligible_count}`",
                "",
            ]
        )
        if arm.issues:
            lines.extend(["Issues:", ""])
            lines.extend(f"- {issue}" for issue in arm.issues)
            lines.append("")
    if report.issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in report.issues)
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
