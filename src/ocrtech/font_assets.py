"""Provision audited font assets from a source manifest."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DataValidationError
from .manifest import sha256_file


@dataclass(slots=True)
class FontAssetRecord:
    asset_id: str
    url: str
    target_path: str
    sha256: str
    status: str
    bytes_written: int
    license: str | None = None
    source: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "url": self.url,
            "target_path": self.target_path,
            "sha256": self.sha256,
            "status": self.status,
            "bytes_written": self.bytes_written,
            "license": self.license,
            "source": self.source,
            "error": self.error,
        }


@dataclass(slots=True)
class FontAssetPreparationSummary:
    manifest_path: str
    manifest_id: str
    asset_root: str
    output_dir: str | None
    asset_count: int
    downloaded_count: int
    reused_count: int
    failed_count: int
    records: list[FontAssetRecord]
    errors: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    @property
    def passed(self) -> bool:
        return self.failed_count == 0 and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_id": self.manifest_id,
            "asset_root": self.asset_root,
            "output_dir": self.output_dir,
            "passed": self.passed,
            "asset_count": self.asset_count,
            "downloaded_count": self.downloaded_count,
            "reused_count": self.reused_count,
            "failed_count": self.failed_count,
            "errors": self.errors,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(slots=True)
class FontAssetReadinessReport:
    manifest_path: str
    preparation_report: str
    inventory_report: str
    renderability_report: str
    output_dir: str | None
    passed: bool
    asset_count: int
    checked_file_count: int
    issues: list[str] = field(default_factory=list)
    summary_json_path: str | None = None
    summary_md_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "preparation_report": self.preparation_report,
            "inventory_report": self.inventory_report,
            "renderability_report": self.renderability_report,
            "output_dir": self.output_dir,
            "passed": self.passed,
            "asset_count": self.asset_count,
            "checked_file_count": self.checked_file_count,
            "issues": self.issues,
            "summary_json_path": self.summary_json_path,
            "summary_md_path": self.summary_md_path,
        }


def prepare_font_assets(
    manifest_path: str | Path,
    *,
    asset_root: str | Path = ".",
    out_dir: str | Path | None = None,
    force: bool = False,
    timeout_seconds: float = 60,
) -> FontAssetPreparationSummary:
    manifest = Path(manifest_path)
    payload = _load_font_asset_manifest(manifest)
    manifest_id = _required_text(payload, "manifest_id", label=str(manifest))
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise DataValidationError(f"font asset manifest must contain a non-empty assets list: {manifest}")
    root = Path(asset_root)
    root.mkdir(parents=True, exist_ok=True)

    records: list[FontAssetRecord] = []
    errors: list[str] = []
    downloaded_count = 0
    reused_count = 0
    failed_count = 0

    for index, item in enumerate(assets):
        try:
            record = _prepare_font_asset(item, index=index, asset_root=root, force=force, timeout_seconds=timeout_seconds)
        except DataValidationError as exc:
            failed_count += 1
            message = str(exc)
            errors.append(message)
            records.append(
                FontAssetRecord(
                    asset_id=f"asset-{index}",
                    url="",
                    target_path="",
                    sha256="",
                    status="failed",
                    bytes_written=0,
                    error=message,
                )
            )
            continue
        records.append(record)
        if record.status == "downloaded":
            downloaded_count += 1
        elif record.status == "reused":
            reused_count += 1
        else:
            failed_count += 1
            if record.error:
                errors.append(record.error)

    summary = FontAssetPreparationSummary(
        manifest_path=str(manifest),
        manifest_id=manifest_id,
        asset_root=str(root),
        output_dir=str(out_dir) if out_dir is not None else None,
        asset_count=len(assets),
        downloaded_count=downloaded_count,
        reused_count=reused_count,
        failed_count=failed_count,
        records=records,
        errors=errors,
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "font-assets-preparation.json"
        md_path = out / "font-assets-preparation.md"
        json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_font_asset_preparation_markdown(md_path, summary)
        summary.summary_json_path = str(json_path)
        summary.summary_md_path = str(md_path)
    return summary


def audit_font_asset_readiness(
    manifest_path: str | Path,
    preparation_report_path: str | Path,
    inventory_report_path: str | Path,
    renderability_report_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    asset_root: str | Path = ".",
) -> FontAssetReadinessReport:
    manifest = Path(manifest_path)
    prep_path = Path(preparation_report_path)
    inventory_path = Path(inventory_report_path)
    renderability_path = Path(renderability_report_path)
    root = Path(asset_root)
    manifest_payload = _load_font_asset_manifest(manifest)
    prep_payload = _load_json_object(prep_path, "font asset preparation report")
    inventory_payload = _load_json_object(inventory_path, "font inventory report")
    renderability_payload = _load_json_object(renderability_path, "font renderability report")

    assets = manifest_payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise DataValidationError(f"font asset manifest must contain a non-empty assets list: {manifest}")

    issues: list[str] = []
    if prep_payload.get("passed") is not True:
        issues.append(f"font asset preparation did not pass: {prep_path}")
    if inventory_payload.get("passed") is not True:
        issues.append(f"font inventory did not pass: {inventory_path}")
    if renderability_payload.get("passed") is not True:
        issues.append(f"font renderability did not pass: {renderability_path}")

    inventory_items = inventory_payload.get("items")
    inventory_shas = {
        str(item.get("sha256"))
        for item in inventory_items
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    } if isinstance(inventory_items, list) else set()
    if not isinstance(inventory_items, list):
        issues.append(f"font inventory report missing items list: {inventory_path}")

    checked_file_count = 0
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            issues.append(f"manifest asset {index} is not an object")
            continue
        asset_id = str(item.get("id") or f"asset-{index}")
        target_value = item.get("target_path")
        expected_sha = item.get("sha256")
        if not isinstance(target_value, str) or not target_value:
            issues.append(f"{asset_id}: target_path missing")
            continue
        if not isinstance(expected_sha, str) or not expected_sha:
            issues.append(f"{asset_id}: sha256 missing")
            continue
        target_relative = Path(target_value)
        if target_relative.is_absolute() or ".." in target_relative.parts:
            issues.append(f"{asset_id}: target_path must be safe relative path: {target_value}")
            continue
        target = root / target_relative
        if not target.is_file():
            issues.append(f"{asset_id}: target file missing: {target}")
            continue
        checked_file_count += 1
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            issues.append(f"{asset_id}: target sha256 mismatch: expected={expected_sha} actual={actual_sha}")
        if _is_font_asset_path(target_relative) and expected_sha not in inventory_shas:
            issues.append(f"{asset_id}: expected sha256 not present in inventory report: {expected_sha}")

    report = FontAssetReadinessReport(
        manifest_path=str(manifest),
        preparation_report=str(prep_path),
        inventory_report=str(inventory_path),
        renderability_report=str(renderability_path),
        output_dir=str(out_dir) if out_dir is not None else None,
        passed=not issues,
        asset_count=len(assets),
        checked_file_count=checked_file_count,
        issues=issues,
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "font-asset-readiness.json"
        md_path = out / "font-asset-readiness.md"
        report.summary_json_path = str(json_path)
        report.summary_md_path = str(md_path)
        json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_font_asset_readiness_markdown(md_path, report)
    return report


def _prepare_font_asset(
    item: Any,
    *,
    index: int,
    asset_root: Path,
    force: bool,
    timeout_seconds: float,
) -> FontAssetRecord:
    if not isinstance(item, dict):
        raise DataValidationError(f"font asset row {index} must be an object")
    asset_id = _required_text(item, "id", label=f"font asset row {index}")
    url = _required_text(item, "url", label=f"font asset {asset_id}")
    target_value = _required_text(item, "target_path", label=f"font asset {asset_id}")
    expected_sha = _required_text(item, "sha256", label=f"font asset {asset_id}").lower()
    archive_member = _optional_text(item.get("archive_member"))
    if archive_member is not None:
        try:
            _validate_zip_archive_member(archive_member)
        except ValueError as exc:
            raise DataValidationError(f"font asset {asset_id} has invalid archive_member: {archive_member!r}") from exc
    archive_sha = _optional_text(item.get("archive_sha256"))
    if archive_sha is not None:
        archive_sha = archive_sha.lower()
        if len(archive_sha) != 64 or any(char not in "0123456789abcdef" for char in archive_sha):
            raise DataValidationError(f"font asset {asset_id} has invalid archive_sha256: {archive_sha!r}")
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise DataValidationError(f"font asset {asset_id} has invalid sha256: {expected_sha!r}")
    target_relative = Path(target_value)
    if target_relative.is_absolute() or ".." in target_relative.parts:
        raise DataValidationError(f"font asset {asset_id} target_path must be a safe relative path: {target_value}")
    target = asset_root / target_relative
    license_value = _optional_text(item.get("license"))
    source_value = _optional_text(item.get("source"))

    if target.exists():
        actual_sha = sha256_file(target)
        if actual_sha == expected_sha:
            return FontAssetRecord(
                asset_id=asset_id,
                url=url,
                target_path=str(target),
                sha256=actual_sha,
                status="reused",
                bytes_written=target.stat().st_size,
                license=license_value,
                source=source_value,
            )
        if not force:
            return FontAssetRecord(
                asset_id=asset_id,
                url=url,
                target_path=str(target),
                sha256=actual_sha,
                status="failed",
                bytes_written=target.stat().st_size,
                license=license_value,
                source=source_value,
                error=f"{asset_id}: existing file sha256 mismatch: expected={expected_sha} actual={actual_sha}",
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp_path = _download_asset_to_temp_path(
            url,
            target.parent,
            timeout_seconds=timeout_seconds,
            archive_member=archive_member,
            archive_sha256=archive_sha,
        )
    except (OSError, urllib.error.URLError, TimeoutError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        return FontAssetRecord(
            asset_id=asset_id,
            url=url,
            target_path=str(target),
            sha256="",
            status="failed",
            bytes_written=0,
            license=license_value,
            source=source_value,
            error=f"{asset_id}: download failed: {exc}",
        )
    actual_sha = sha256_file(temp_path)
    if actual_sha != expected_sha:
        bytes_written = temp_path.stat().st_size
        temp_path.unlink(missing_ok=True)
        return FontAssetRecord(
            asset_id=asset_id,
            url=url,
            target_path=str(target),
            sha256=actual_sha,
            status="failed",
            bytes_written=bytes_written,
            license=license_value,
            source=source_value,
            error=f"{asset_id}: downloaded sha256 mismatch: expected={expected_sha} actual={actual_sha}",
        )
    temp_path.replace(target)
    return FontAssetRecord(
        asset_id=asset_id,
        url=url,
        target_path=str(target),
        sha256=actual_sha,
        status="downloaded",
        bytes_written=target.stat().st_size,
        license=license_value,
        source=source_value,
    )


def _download_asset_to_temp_path(
    url: str,
    target_dir: Path,
    *,
    timeout_seconds: float,
    archive_member: str | None,
    archive_sha256: str | None,
) -> Path:
    downloaded = _download_url_to_temp_path(url, target_dir, timeout_seconds=timeout_seconds)
    if archive_member is None:
        if archive_sha256 is not None:
            actual_archive_sha = sha256_file(downloaded)
            if actual_archive_sha != archive_sha256:
                downloaded.unlink(missing_ok=True)
                raise ValueError(f"downloaded file sha256 mismatch: expected={archive_sha256} actual={actual_archive_sha}")
        return downloaded
    try:
        if archive_sha256 is not None:
            actual_archive_sha = sha256_file(downloaded)
            if actual_archive_sha != archive_sha256:
                raise ValueError(f"archive sha256 mismatch: expected={archive_sha256} actual={actual_archive_sha}")
        extracted = _extract_zip_member_to_temp_path(downloaded, archive_member, target_dir)
    finally:
        downloaded.unlink(missing_ok=True)
    return extracted


def _download_url_to_temp_path(url: str, target_dir: Path, *, timeout_seconds: float) -> Path:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target_dir)) as handle:
            temp_path = Path(handle.name)
            shutil.copyfileobj(response, handle)
    return temp_path


def _extract_zip_member_to_temp_path(archive_path: Path, archive_member: str, target_dir: Path) -> Path:
    _validate_zip_archive_member(archive_member)
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(archive_member)
        except KeyError as exc:
            raise KeyError(f"archive member not found: {archive_member}") from exc
        if info.is_dir():
            raise ValueError(f"archive_member is a directory: {archive_member}")
        with archive.open(info) as source:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(target_dir)) as handle:
                temp_path = Path(handle.name)
                shutil.copyfileobj(source, handle)
    return temp_path


def _validate_zip_archive_member(archive_member: str) -> None:
    if not archive_member or archive_member.startswith("/") or ".." in Path(archive_member).parts:
        raise ValueError(f"unsafe archive_member: {archive_member!r}")


def _load_font_asset_manifest(path: Path) -> dict[str, Any]:
    return _load_json_object(path, "font asset manifest")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _required_text(payload: dict[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} missing required string field {key!r}")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _is_font_asset_path(path: Path) -> bool:
    return path.suffix.lower() in {".ttf", ".otf", ".ttc", ".woff", ".woff2"}


def _write_font_asset_preparation_markdown(path: Path, summary: FontAssetPreparationSummary) -> None:
    lines = [
        "# Font Asset Preparation",
        "",
        f"- passed: `{summary.passed}`",
        f"- manifest: `{summary.manifest_path}`",
        f"- manifest id: `{summary.manifest_id}`",
        f"- asset root: `{summary.asset_root}`",
        f"- assets: `{summary.asset_count}`",
        f"- downloaded: `{summary.downloaded_count}`",
        f"- reused: `{summary.reused_count}`",
        f"- failed: `{summary.failed_count}`",
        "",
        "## Assets",
        "",
        "| Asset | Status | Target | SHA-256 | License | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary.records:
        lines.append(
            "| "
            f"`{record.asset_id}` | "
            f"`{record.status}` | "
            f"`{record.target_path}` | "
            f"`{record.sha256}` | "
            f"{record.license or ''} | "
            f"{record.error or ''} |"
        )
    if summary.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_font_asset_readiness_markdown(path: Path, report: FontAssetReadinessReport) -> None:
    lines = [
        "# Font Asset Readiness",
        "",
        f"- passed: `{report.passed}`",
        f"- manifest: `{report.manifest_path}`",
        f"- preparation report: `{report.preparation_report}`",
        f"- inventory report: `{report.inventory_report}`",
        f"- renderability report: `{report.renderability_report}`",
        f"- assets: `{report.asset_count}`",
        f"- checked files: `{report.checked_file_count}`",
        f"- issues: `{len(report.issues)}`",
    ]
    if report.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in report.issues)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
