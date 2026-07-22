"""Exact predelegation identity contracts for runnable Newari diagnostics.

These contracts admit relocation of previously audited runtime packages, not
new model bytes.  They bind the files used by the book-OCR recognizer plus its
configured dictionary to the historical r53 local snapshot.  They do not
establish provenance, licensing, quality, or which bytes a backend later opens.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


R53_MANIFEST_SHA256 = (
    "8dd83bab2b5c1c8bf03489b22b804b13e790caaaa9d5ac3cbb56e964f45eed4d"
)
R53_REPORT_SHA256 = (
    "9a1a9d04aba6a8f923b46c982fb97ed0e4cc9dab59b3321d9a99e578ffc1cfc7"
)
R53_TOOL_SHA256 = (
    "cbec433229269f5a004ebba33b6bd1ad1994df2947c01da0208877f5758668b5"
)
IDENTITY_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class FileIdentity:
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class NewariArtifactIdentityContract:
    ref: str
    script: str
    status: str
    execution_policy: str
    writing_system_profile: str
    route_script: str
    source_unicode_status: str
    output_policy: str
    config_artifact: str
    required_opt_in: str
    paddle_recognition_model_name: str
    artifact_files: tuple[tuple[str, FileIdentity], ...]
    dictionary: FileIdentity


NEWARI_ARTIFACT_IDENTITY_CONTRACTS: Mapping[
    str, NewariArtifactIdentityContract
] = MappingProxyType(
    {
        "devanagari": NewariArtifactIdentityContract(
            ref="devanagari",
            script="devanagari",
            status="operational_recovered_model_diagnostic_only",
            execution_policy="diagnostic_opt_in_required",
            writing_system_profile="devanagari",
            route_script="devanagari",
            source_unicode_status="encoded",
            output_policy="native_devanagari_unicode",
            config_artifact="spaces/limbu-ocr/models/deva-v2",
            required_opt_in="diagnostic",
            paddle_recognition_model_name="PP-OCRv5_mobile_rec",
            artifact_files=(
                (
                    "inference.json",
                    FileIdentity(
                        size_bytes=217712,
                        sha256=(
                            "772e801de4c1cc260058f7c426ab618ef52ecc7ea51442886c8c8fc1606cb88b"
                        ),
                    ),
                ),
                (
                    "inference.pdiparams",
                    FileIdentity(
                        size_bytes=7630503,
                        sha256=(
                            "32f5256f8a062830be4394606b8307c30dd9b9fb1be4e15af65606abffb479b4"
                        ),
                    ),
                ),
                (
                    "inference.yml",
                    FileIdentity(
                        size_bytes=1812,
                        sha256=(
                            "4f76c7bd21ea18ec118adef93b98017b55a449072b35f1c3ea2baa3df217c8dc"
                        ),
                    ),
                ),
            ),
            dictionary=FileIdentity(
                size_bytes=467,
                sha256=(
                    "96806153df021c99b0195de9757d8fe6443c52f7c8a1c74bf76a5895c37b8c0b"
                ),
            ),
        ),
        "newa_prachalit": NewariArtifactIdentityContract(
            ref="newa_prachalit",
            script="newa_prachalit",
            status="operational_recovered_model_diagnostic_failed_fit_only",
            execution_policy="known_failed_fit_opt_in_required",
            writing_system_profile="newa_prachalit",
            route_script="newa",
            source_unicode_status="encoded",
            output_policy="native_newa_unicode",
            config_artifact="spaces/limbu-ocr/models/newa-prachalit-v1",
            required_opt_in="known_failed_fit",
            paddle_recognition_model_name="PP-OCRv5_mobile_rec",
            artifact_files=(
                (
                    "inference.json",
                    FileIdentity(
                        size_bytes=217712,
                        sha256=(
                            "ad77021dc0bb6193a3230a653e8a1c3ac3b38af835febe2bf961e59cecfd571f"
                        ),
                    ),
                ),
                (
                    "inference.pdiparams",
                    FileIdentity(
                        size_bytes=7612593,
                        sha256=(
                            "e03e1be292801eba13b54300feb5df4475bbcfc40ccb87aa375d13f252364193"
                        ),
                    ),
                ),
                (
                    "inference.yml",
                    FileIdentity(
                        size_bytes=1683,
                        sha256=(
                            "8b6458685a1402c9b6160d18d7f9b65c68a4c0b9a884d16f53c3ef195b70ab29"
                        ),
                    ),
                ),
            ),
            dictionary=FileIdentity(
                size_bytes=515,
                sha256=(
                    "fa270ab0ed3685492554455dbb258e9a899ba4226f107895732ba52867a2155e"
                ),
            ),
        ),
    }
)


class IdentityVerificationError(RuntimeError):
    """A path could not be shown to contain one expected regular-file identity."""


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_node(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _safe_open_flags(*, directory: bool) -> int:
    """Return required no-follow flags or fail closed on an unsupported host."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise IdentityVerificationError(
            "this platform cannot enforce no-follow identity opens"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if directory:
        if not hasattr(os, "O_DIRECTORY"):
            raise IdentityVerificationError(
                "this platform cannot enforce directory-only identity opens"
            )
        flags |= os.O_DIRECTORY
    return flags


def _open_directory_chain_no_follow(path: Path) -> int:
    """Open an absolute directory while rejecting symlinks in every component."""

    if not path.is_absolute():
        raise IdentityVerificationError(f"identity path is not absolute: {path}")
    components = path.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise IdentityVerificationError(
            f"identity path has a non-canonical component: {path}"
        )
    flags = _safe_open_flags(directory=True)
    try:
        current_fd = os.open(path.anchor, flags)
    except OSError as exc:
        raise IdentityVerificationError(
            f"could not safely open identity path anchor {path.anchor!r}: {exc}"
        ) from exc
    try:
        for component in components:
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except OSError as exc:
                raise IdentityVerificationError(
                    f"could not inspect identity directory component {component!r} "
                    f"in {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise IdentityVerificationError(
                    "identity directory component is not a regular non-symlink "
                    f"directory: {component!r} in {path}"
                )
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise IdentityVerificationError(
                    f"could not safely open identity directory component "
                    f"{component!r} in {path}: {exc}"
                ) from exc
            try:
                opened = os.fstat(next_fd)
            except OSError as exc:
                os.close(next_fd)
                raise IdentityVerificationError(
                    f"could not inspect opened identity directory component "
                    f"{component!r} in {path}: {exc}"
                ) from exc
            if not stat.S_ISDIR(opened.st_mode) or not _same_node(before, opened):
                os.close(next_fd)
                raise IdentityVerificationError(
                    f"identity directory component changed while opening: {path}"
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def verify_regular_file_identity(
    path: Path, expected: FileIdentity, *, label: str
) -> None:
    """Stream-hash one exact regular file through a no-follow descriptor chain."""

    path = Path(path)
    parent_fd = _open_directory_chain_no_follow(path.parent)
    file_fd: int | None = None
    try:
        try:
            before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise IdentityVerificationError(
                f"{label} cannot be inspected at {path}: {exc}"
            ) from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise IdentityVerificationError(
                f"{label} is not a regular non-symlink file: {path}"
            )
        flags = _safe_open_flags(directory=False)
        try:
            file_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise IdentityVerificationError(
                f"{label} cannot be safely opened at {path}: {exc}"
            ) from exc
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_node(before, opened):
            raise IdentityVerificationError(
                f"{label} changed while it was being opened: {path}"
            )
        if opened.st_size != expected.size_bytes:
            raise IdentityVerificationError(
                f"{label} size mismatch at {path}: expected "
                f"{expected.size_bytes}, got {opened.st_size}"
            )

        digest = hashlib.sha256()
        remaining = expected.size_bytes + 1
        observed_size = 0
        while remaining:
            chunk = os.read(file_fd, min(IDENTITY_CHUNK_SIZE, remaining))
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_fd)
        if not _stable_file_metadata(opened, after):
            raise IdentityVerificationError(
                f"{label} changed while it was being hashed: {path}"
            )
        if observed_size != expected.size_bytes:
            raise IdentityVerificationError(
                f"{label} streamed size mismatch at {path}: expected "
                f"{expected.size_bytes}, got at least {observed_size}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected.sha256:
            raise IdentityVerificationError(
                f"{label} SHA-256 mismatch at {path}: expected "
                f"{expected.sha256}, got {actual_sha256}"
            )
    except OSError as exc:
        raise IdentityVerificationError(
            f"{label} could not be safely identity-checked at {path}: {exc}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def validate_active_newari_identity_contracts(
    resolved: Any,
    *,
    paddle_recognition_model_name: str | None,
    contracts: Mapping[str, NewariArtifactIdentityContract] = (
        NEWARI_ARTIFACT_IDENTITY_CONTRACTS
    ),
) -> list[str]:
    """Validate active Newari semantics without touching artifact paths."""

    if getattr(resolved, "registry_id", None) != "newar":
        return []
    issues: list[str] = []
    active = list(getattr(resolved, "recognizers", []))
    opt_ins = set(getattr(resolved, "newari_execution_opt_ins", []))
    field_names = (
        "ref",
        "script",
        "status",
        "execution_policy",
        "writing_system_profile",
        "route_script",
        "source_unicode_status",
        "output_policy",
        "config_artifact",
    )
    for spec in active:
        ref = getattr(spec, "ref", None)
        contract = contracts.get(ref)
        if contract is None:
            issues.append(
                f"active Newar recognizer {ref!r} has no exact identity contract"
            )
            continue
        for field_name in field_names:
            expected = getattr(contract, field_name)
            actual = getattr(spec, field_name, None)
            if actual != expected:
                issues.append(
                    f"recognizer {ref} {field_name} drift: expected "
                    f"{expected!r}, got {actual!r}"
                )
        if contract.required_opt_in not in opt_ins:
            issues.append(
                f"recognizer {ref} is missing exact execution opt-in "
                f"{contract.required_opt_in!r}"
            )
        if paddle_recognition_model_name != contract.paddle_recognition_model_name:
            issues.append(
                f"recognizer {ref} Paddle recognition model-name drift: expected "
                f"{contract.paddle_recognition_model_name!r}, got "
                f"{paddle_recognition_model_name!r}"
            )
        if getattr(spec, "artifact", None) is None:
            issues.append(f"recognizer {ref} has no artifact path to identity-check")
        if getattr(spec, "dictionary", None) is None:
            issues.append(f"recognizer {ref} has no dictionary path to identity-check")
    return issues


def verify_active_newari_artifact_identities(
    resolved: Any,
    *,
    contracts: Mapping[str, NewariArtifactIdentityContract] = (
        NEWARI_ARTIFACT_IDENTITY_CONTRACTS
    ),
) -> list[str]:
    """Verify exact active Newari bytes after semantic contracts have passed."""

    if getattr(resolved, "registry_id", None) != "newar":
        return []
    issues: list[str] = []
    for spec in getattr(resolved, "recognizers", []):
        ref = getattr(spec, "ref", None)
        contract = contracts.get(ref)
        if contract is None:
            issues.append(
                f"active Newar recognizer {ref!r} has no exact identity contract"
            )
            continue
        artifact = getattr(spec, "artifact", None)
        dictionary = getattr(spec, "dictionary", None)
        if artifact is None or dictionary is None:
            issues.append(f"recognizer {ref} has incomplete identity paths")
            continue
        checks = [
            (
                Path(artifact) / filename,
                expected,
                f"recognizer {ref} artifact {filename}",
            )
            for filename, expected in contract.artifact_files
        ]
        checks.append(
            (Path(dictionary), contract.dictionary, f"recognizer {ref} dictionary")
        )
        for path, expected, label in checks:
            try:
                verify_regular_file_identity(path, expected, label=label)
            except IdentityVerificationError as exc:
                issues.append(str(exc))
    return issues
