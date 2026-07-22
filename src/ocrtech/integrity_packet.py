"""Crash-fail-closed publication for small integrity packets.

The final directory is reserved without overwrite and remains visibly invalid
while ``.INCOMPLETE`` exists.  Every payload is written through an exclusive,
no-follow directory-relative descriptor and made durable before the marker is
removed as the commit point.  Strict readers reject the marker, sidecars,
special nodes, missing files, and every manifest mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


MARKER_NAME = ".INCOMPLETE"
MANIFEST_NAME = "SHA256SUMS"
MARKER_SCHEMA = "ocrtech_integrity_packet_incomplete_v1"
MAX_MEMBER_COUNT = 64
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_PACKET_BYTES = 32 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024


class PacketPublicationError(RuntimeError):
    """An integrity packet could not be safely published or recovered."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_node(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise PacketPublicationError(
            "host cannot enforce no-follow directory packet publication"
        )
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY


def _file_flags(*, exclusive: bool) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PacketPublicationError(
            "host cannot enforce no-follow packet publication"
        )
    flags = os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if exclusive:
        return flags | os.O_WRONLY | os.O_CREAT | os.O_EXCL
    return flags | os.O_RDONLY


def _open_directory_chain(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    components = absolute.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise PacketPublicationError(f"packet path is not canonical: {path}")
    flags = _directory_flags()
    try:
        current_fd = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise PacketPublicationError(
            f"could not open packet path anchor {absolute.anchor!r}: {exc}"
        ) from exc
    try:
        for component in components:
            before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PacketPublicationError(
                    f"packet path component is not a real directory: {component!r}"
                )
            next_fd = os.open(component, flags, dir_fd=current_fd)
            opened = os.fstat(next_fd)
            if not stat.S_ISDIR(opened.st_mode) or not _same_node(before, opened):
                os.close(next_fd)
                raise PacketPublicationError(
                    f"packet path component changed while opening: {absolute}"
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except (OSError, PacketPublicationError) as exc:
        os.close(current_fd)
        if isinstance(exc, PacketPublicationError):
            raise
        raise PacketPublicationError(
            f"could not safely open packet directory {absolute}: {exc}"
        ) from exc


def _validate_transaction_id(transaction_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", transaction_id) is None:
        raise PacketPublicationError("transaction_id is not a bounded safe token")


def _validate_members(members: Mapping[str, bytes]) -> dict[str, bytes]:
    if not members or len(members) > MAX_MEMBER_COUNT:
        raise PacketPublicationError("packet member count is outside the safe bound")
    normalized: dict[str, bytes] = {}
    total = 0
    for name, value in members.items():
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in {".", "..", MARKER_NAME, MANIFEST_NAME}
            or name.startswith("._")
            or Path(name).name != name
            or Path(name).as_posix() != name
        ):
            raise PacketPublicationError(f"unsafe flat packet member name: {name!r}")
        if not isinstance(value, bytes):
            raise PacketPublicationError(f"packet member is not bytes: {name}")
        if len(value) > MAX_MEMBER_BYTES:
            raise PacketPublicationError(f"packet member exceeds size bound: {name}")
        total += len(value)
        normalized[name] = value
    if total > MAX_PACKET_BYTES:
        raise PacketPublicationError("packet payload exceeds total size bound")
    return normalized


def _payloads(members: Mapping[str, bytes]) -> dict[str, bytes]:
    normalized = _validate_members(members)
    manifest = "".join(
        f"{_sha256_bytes(normalized[name])}  {name}\n" for name in sorted(normalized)
    ).encode("utf-8")
    return {**normalized, MANIFEST_NAME: manifest}


def _marker_bytes(transaction_id: str, payloads: Mapping[str, bytes]) -> bytes:
    _validate_transaction_id(transaction_id)
    payload = {
        "schema": MARKER_SCHEMA,
        "transaction_id": transaction_id,
        "commit_state": "incomplete",
        "payloads": {
            name: {"size_bytes": len(value), "sha256": _sha256_bytes(value)}
            for name, value in sorted(payloads.items())
        },
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise PacketPublicationError("packet descriptor made no write progress")
        offset += written


def _write_exclusive(
    directory_fd: int, name: str, value: bytes
) -> tuple[os.stat_result, tuple[str, os.stat_result]]:
    guard_name = f"._{name}"
    try:
        guard_fd = os.open(
            guard_name,
            _file_flags(exclusive=True),
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise PacketPublicationError(
            f"could not reserve packet sidecar guard {guard_name}: {exc}"
        ) from exc
    guard_stat = os.fstat(guard_fd)
    os.close(guard_fd)
    try:
        descriptor = os.open(
            name,
            _file_flags(exclusive=True),
            0o644,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise PacketPublicationError(
            f"could not exclusively create packet member {name}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PacketPublicationError(f"created packet member is not regular: {name}")
        _write_all(descriptor, value)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if not _stable_file(before, after) and not (
            _same_node(before, after)
            and stat.S_ISREG(after.st_mode)
            and after.st_size == len(value)
        ):
            raise PacketPublicationError(
                f"packet member changed unexpectedly while writing: {name}"
            )
        if after.st_size != len(value):
            raise PacketPublicationError(f"packet member size drifted: {name}")
    finally:
        os.close(descriptor)
    try:
        current_guard = os.stat(
            guard_name, dir_fd=directory_fd, follow_symlinks=False
        )
    except OSError:
        current_guard = None
    if current_guard is not None and _same_node(guard_stat, current_guard):
        os.unlink(guard_name, dir_fd=directory_fd)
    return after, (guard_name, guard_stat)


def _read_regular_at(
    directory_fd: int, name: str, *, max_bytes: int = MAX_MEMBER_BYTES
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PacketPublicationError(f"packet entry is not regular: {name}")
        descriptor = os.open(name, _file_flags(exclusive=False), dir_fd=directory_fd)
    except OSError as exc:
        raise PacketPublicationError(f"could not safely open packet entry {name}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not _same_node(before, opened) or not stat.S_ISREG(opened.st_mode):
            raise PacketPublicationError(
                f"packet entry changed while opening: {name}"
            )
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, READ_CHUNK_SIZE):
            total += len(chunk)
            if total > max_bytes:
                raise PacketPublicationError(f"packet entry exceeds read bound: {name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if not _stable_file(opened, after):
            raise PacketPublicationError(f"packet entry changed while reading: {name}")
        return b"".join(chunks), after
    except OSError as exc:
        raise PacketPublicationError(
            f"packet entry read failed for {name}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _manifest_rows(source: bytes) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeError as exc:
        return {}, [f"manifest is not UTF-8: {exc}"]
    rows: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"invalid manifest row {number}")
            continue
        digest, name = line[:64], line[66:]
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"invalid manifest digest at row {number}")
        elif (
            not name
            or name != name.strip()
            or name in {".", "..", MARKER_NAME, MANIFEST_NAME}
            or name.startswith("._")
            or Path(name).name != name
            or Path(name).as_posix() != name
        ):
            errors.append(f"unsafe manifest member at row {number}: {name!r}")
        elif name in rows:
            errors.append(f"duplicate manifest member: {name}")
        else:
            rows[name] = digest
    if not rows:
        errors.append("manifest has no payload rows")
    if len(rows) > MAX_MEMBER_COUNT:
        errors.append("manifest payload row count exceeds safe bound")
    return rows, errors


def _verify_open_packet(directory_fd: int, *, allow_incomplete: bool) -> list[str]:
    try:
        initial_entries = sorted(os.listdir(directory_fd))
    except OSError as exc:
        return [f"packet listing failed: {exc}"]
    errors: list[str] = []
    marker_present = MARKER_NAME in initial_entries
    if marker_present and not allow_incomplete:
        errors.append("packet publication is incomplete")
    if any(name.startswith("._") for name in initial_entries):
        errors.append("packet contains an AppleDouble sidecar")
    if MANIFEST_NAME not in initial_entries:
        errors.append("packet manifest is missing")
        return errors
    try:
        manifest_source, _ = _read_regular_at(directory_fd, MANIFEST_NAME)
    except PacketPublicationError as exc:
        errors.append(str(exc))
        return errors
    rows, row_errors = _manifest_rows(manifest_source)
    errors.extend(row_errors)
    expected_entries = sorted(
        [*rows, MANIFEST_NAME, *([MARKER_NAME] if marker_present else [])]
    )
    if initial_entries != expected_entries:
        errors.append("packet manifest file set mismatch")
    aggregate_size = 0
    for name, expected_digest in rows.items():
        try:
            source, _ = _read_regular_at(directory_fd, name)
        except PacketPublicationError as exc:
            errors.append(str(exc))
            continue
        aggregate_size += len(source)
        if aggregate_size > MAX_PACKET_BYTES:
            errors.append("packet payload exceeds aggregate read bound")
            break
        if _sha256_bytes(source) != expected_digest:
            errors.append(f"packet manifest hash mismatch: {name}")
    try:
        final_entries = sorted(os.listdir(directory_fd))
    except OSError as exc:
        errors.append(f"packet final listing failed: {exc}")
    else:
        if final_entries != initial_entries:
            errors.append("packet entries changed during verification")
    return errors


def verify_integrity_packet(root: Path) -> list[str]:
    """Return strict closure errors; an empty list means the packet is committed."""

    try:
        directory_fd = _open_directory_chain(root)
    except PacketPublicationError as exc:
        return [str(exc)]
    try:
        return _verify_open_packet(directory_fd, allow_incomplete=False)
    finally:
        os.close(directory_fd)


def publish_integrity_packet(
    out_dir: Path,
    members: Mapping[str, bytes],
    *,
    transaction_id: str,
    fault_hook: Callable[[str], None] | None = None,
) -> None:
    """Publish one no-overwrite packet, leaving any pre-commit failure invalid."""

    payloads = _payloads(members)
    marker_source = _marker_bytes(transaction_id, payloads)
    out_dir = Path(out_dir)
    parent_fd = _open_directory_chain(out_dir.parent)
    directory_fd: int | None = None
    try:
        try:
            os.mkdir(out_dir.name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing packet output: {out_dir}"
            ) from None
        os.fsync(parent_fd)
        if fault_hook is not None:
            fault_hook("after_reservation")
        before = os.stat(out_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        directory_fd = os.open(out_dir.name, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_node(before, opened):
            raise PacketPublicationError(
                "packet output directory changed during reservation"
            )
        _write_exclusive(directory_fd, MARKER_NAME, marker_source)
        os.fsync(directory_fd)
        if fault_hook is not None:
            fault_hook("after_marker")
        for name in sorted(payloads, key=lambda item: (item == MANIFEST_NAME, item)):
            _write_exclusive(directory_fd, name, payloads[name])
            if fault_hook is not None:
                fault_hook(f"after_member:{name}")
        issues = _verify_open_packet(directory_fd, allow_incomplete=True)
        if issues:
            raise PacketPublicationError(
                "invalid incomplete packet before commit: " + "; ".join(issues)
            )
        commit_stats: dict[str, os.stat_result] = {}
        for name, expected_source in {
            MARKER_NAME: marker_source,
            **payloads,
        }.items():
            observed_source, observed_stat = _read_regular_at(directory_fd, name)
            if observed_source != expected_source:
                raise PacketPublicationError(
                    f"packet entry bytes drifted before commit: {name}"
                )
            commit_stats[name] = observed_stat
        os.fsync(directory_fd)
        if fault_hook is not None:
            fault_hook("before_commit")
        for name, created_stat in commit_stats.items():
            current_stat = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not _stable_file(created_stat, current_stat):
                raise PacketPublicationError(
                    f"packet entry changed before commit: {name}"
                )
        current_directory = os.stat(
            out_dir.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not _same_node(opened, current_directory):
            raise PacketPublicationError(
                "packet output directory changed before commit"
            )
        os.unlink(MARKER_NAME, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if fault_hook is not None:
            fault_hook("after_commit")
    except FileExistsError:
        raise
    except OSError as exc:
        raise PacketPublicationError(f"packet publication failed: {exc}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def recover_incomplete_packet(
    out_dir: Path, members: Mapping[str, bytes], *, transaction_id: str
) -> None:
    """Remove only a token-bound incomplete packet with exact known entries."""

    payloads = _payloads(members)
    marker_source = _marker_bytes(transaction_id, payloads)
    out_dir = Path(out_dir)
    parent_fd = _open_directory_chain(out_dir.parent)
    directory_fd: int | None = None
    try:
        before = os.stat(out_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        directory_fd = os.open(out_dir.name, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_node(before, opened):
            raise PacketPublicationError(
                "incomplete packet directory changed while opening"
            )
        entries = set(os.listdir(directory_fd))
        allowed = {MARKER_NAME, *payloads}
        unknown = sorted(entries - allowed)
        if unknown:
            raise PacketPublicationError(
                "refusing recovery with foreign entries: " + ", ".join(unknown)
            )
        if MARKER_NAME not in entries:
            raise PacketPublicationError("incomplete packet marker is missing")
        actual_marker, marker_stat = _read_regular_at(directory_fd, MARKER_NAME)
        if actual_marker != marker_source:
            raise PacketPublicationError("incomplete packet marker/token drifted")
        captured: dict[str, os.stat_result] = {MARKER_NAME: marker_stat}
        for name in sorted(entries - {MARKER_NAME}):
            source, entry_stat = _read_regular_at(directory_fd, name)
            if source != payloads[name]:
                raise PacketPublicationError(
                    f"refusing recovery of drifted packet entry: {name}"
                )
            captured[name] = entry_stat
        for name in sorted(captured, key=lambda item: item == MARKER_NAME):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not _stable_file(captured[name], current):
                raise PacketPublicationError(
                    f"incomplete packet entry changed during recovery: {name}"
                )
            os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if os.listdir(directory_fd):
            raise PacketPublicationError(
                "foreign entries appeared during incomplete packet recovery"
            )
        current_dir = os.stat(out_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_node(opened, current_dir):
            raise PacketPublicationError(
                "incomplete packet directory changed during recovery"
            )
        os.close(directory_fd)
        directory_fd = None
        os.rmdir(out_dir.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileNotFoundError as exc:
        raise PacketPublicationError(f"incomplete packet is missing: {out_dir}") from exc
    except OSError as exc:
        raise PacketPublicationError(f"incomplete packet recovery failed: {exc}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def marker_payload(
    members: Mapping[str, bytes], *, transaction_id: str
) -> dict[str, Any]:
    """Expose the deterministic marker object for audit/reporting tests."""

    return json.loads(_marker_bytes(transaction_id, _payloads(members)))
