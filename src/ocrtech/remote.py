"""Remote host readiness checks for training and benchmark campaigns."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import RemoteAuditError


@dataclass(slots=True)
class RemoteCheck:
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
class RemoteAuditReport:
    passed: bool
    host: str
    user: str | None
    workdir: str | None
    checks: list[RemoteCheck]
    probe: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "host": self.host,
            "user": self.user,
            "workdir": self.workdir,
            "checks": [check.to_dict() for check in self.checks],
            "probe": self.probe,
        }


@dataclass(slots=True)
class RemoteBootstrapReport:
    passed: bool
    host: str
    user: str | None
    workdir: str
    python_command: str
    venv_dir: str
    actions: dict[str, Any]
    checks: list[RemoteCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "host": self.host,
            "user": self.user,
            "workdir": self.workdir,
            "python_command": self.python_command,
            "venv_dir": self.venv_dir,
            "actions": self.actions,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class RemoteSyncReport:
    passed: bool
    host: str
    user: str | None
    local_root: str
    remote_workdir: str
    archive_path: str
    checks: list[RemoteCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "host": self.host,
            "user": self.user,
            "local_root": self.local_root,
            "remote_workdir": self.remote_workdir,
            "archive_path": self.archive_path,
            "checks": [check.to_dict() for check in self.checks],
        }


DEFAULT_PYTHON_CANDIDATES = [
    "python3.11",
    "~/.local/bin/python3.11",
    "/opt/homebrew/bin/python3.11",
    "python3.12",
    "python3.13",
    "python3",
    "python",
]


def audit_remote_host(
    host: str,
    output_dir: str | Path,
    *,
    user: str | None = None,
    port: int = 22,
    password_env: str | None = None,
    min_python: tuple[int, int] = (3, 11),
    min_free_gb: float = 20.0,
    workdir: str | None = None,
    require_gpu: bool = False,
    require_paddle_training: bool = False,
    python_candidates: list[str] | None = None,
    require_commands: list[str] | None = None,
    optional_commands: list[str] | None = None,
    require_paths: list[str] | None = None,
) -> RemoteAuditReport:
    if not host.strip():
        raise RemoteAuditError("host is required")
    if port < 1 or port > 65535:
        raise RemoteAuditError(f"invalid port: {port}")
    if password_env:
        if password_env not in os.environ:
            raise RemoteAuditError(f"password env var is not set: {password_env}")
        if not shutil_which("expect"):
            raise RemoteAuditError("password-based remote audit requires local expect")
    required = require_commands or ["git"]
    optional = optional_commands or ["uv", "nvidia-smi"]
    paths = require_paths or []
    python_names = python_candidates or DEFAULT_PYTHON_CANDIDATES
    probe = _probe_remote(
        host,
        user=user,
        port=port,
        password_env=password_env,
        workdir=workdir,
        command_names=sorted({*required, *optional}),
        path_checks=paths,
        python_candidates=python_names,
    )
    checks = _evaluate_probe(
        probe,
        min_python=min_python,
        min_free_gb=min_free_gb,
        require_gpu=require_gpu,
        require_paddle_training=require_paddle_training,
        python_candidates=python_names,
        require_commands=required,
        optional_commands=optional,
        require_paths=paths,
        workdir=workdir,
    )
    report = RemoteAuditReport(
        passed=all(check.status != "fail" for check in checks),
        host=host,
        user=user,
        workdir=workdir,
        checks=checks,
        probe=probe,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_remote_audit(report, out)
    return report


def bootstrap_remote_host(
    host: str,
    output_dir: str | Path,
    *,
    user: str | None = None,
    port: int = 22,
    password_env: str | None = None,
    workdir: str,
    min_python: tuple[int, int] = (3, 11),
    python_candidates: list[str] | None = None,
    venv_name: str = ".venv",
    recreate_venv: bool = False,
) -> RemoteBootstrapReport:
    if not workdir.strip():
        raise RemoteAuditError("bootstrap requires a non-empty workdir")
    python_names = python_candidates or DEFAULT_PYTHON_CANDIDATES
    audit = audit_remote_host(
        host,
        Path(output_dir) / "preflight",
        user=user,
        port=port,
        password_env=password_env,
        min_python=min_python,
        min_free_gb=1.0,
        workdir=None,
        python_candidates=python_names,
        require_paths=[],
    )
    selected = _select_python_candidate(audit.probe, min_python)
    if selected is None:
        raise RemoteAuditError(f"no remote python candidate satisfies >= {min_python[0]}.{min_python[1]}")
    bootstrap_payload = _run_remote_json_script(
        host,
        user=user,
        port=port,
        password_env=password_env,
        script=_bootstrap_script(workdir, selected["command"], venv_name, recreate_venv=recreate_venv),
    )
    if not isinstance(bootstrap_payload, dict):
        raise RemoteAuditError("bootstrap payload must be an object")
    checks = [
        RemoteCheck("python", "pass", f"using remote interpreter {selected['command']} -> {selected['version']}"),
        RemoteCheck("workdir", "pass", f"remote workdir ready: {bootstrap_payload.get('workdir') or workdir}"),
        RemoteCheck("venv", "pass", f"remote virtualenv ready: {bootstrap_payload.get('venv_dir') or venv_name}"),
    ]
    report = RemoteBootstrapReport(
        passed=True,
        host=host,
        user=user,
        workdir=str(bootstrap_payload.get("workdir") or workdir),
        python_command=str(bootstrap_payload.get("python_command") or selected["command"]),
        venv_dir=str(bootstrap_payload.get("venv_dir") or venv_name),
        actions=dict(bootstrap_payload),
        checks=checks,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_remote_bootstrap(report, out)
    return report


def sync_remote_workspace(
    local_root: str | Path,
    output_dir: str | Path,
    *,
    host: str,
    user: str | None = None,
    port: int = 22,
    password_env: str | None = None,
    remote_workdir: str,
    exclude_patterns: list[str] | None = None,
) -> RemoteSyncReport:
    local_path = Path(local_root)
    if not local_path.exists() or not local_path.is_dir():
        raise RemoteAuditError(f"local_root does not exist or is not a directory: {local_path}")
    if not remote_workdir.strip():
        raise RemoteAuditError("remote_workdir is required")
    excludes = exclude_patterns or [".venv", "outputs", "runs", "__pycache__", ".pytest_cache"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    expanded_remote_workdir = _expand_remote_path(host, user=user, port=port, password_env=password_env, remote_path=remote_workdir)
    with tempfile.TemporaryDirectory(prefix="ocrtech-sync-") as tmp_dir:
        archive_path = Path(tmp_dir) / "workspace.tar.gz"
        _create_workspace_archive(local_path, archive_path, excludes)
        remote_archive = f"{expanded_remote_workdir.rstrip('/')}/.ocrtech-sync.tar.gz"
        _copy_file_to_remote(archive_path, remote_archive, host=host, user=user, port=port, password_env=password_env)
        _run_remote_json_script(
            host,
            user=user,
            port=port,
            password_env=password_env,
            script=_sync_extract_script(expanded_remote_workdir, remote_archive),
        )
        checks = [
            RemoteCheck("archive", "pass", f"workspace archive created from {local_path}"),
            RemoteCheck("transfer", "pass", f"workspace synced to {expanded_remote_workdir}"),
        ]
        report = RemoteSyncReport(
            passed=True,
            host=host,
            user=user,
            local_root=str(local_path),
            remote_workdir=expanded_remote_workdir,
            archive_path="temporary archive removed after sync",
            checks=checks,
        )
    _write_remote_sync(report, out)
    return report


def _probe_remote(
    host: str,
    *,
    user: str | None,
    port: int,
    password_env: str | None,
    workdir: str | None,
    command_names: list[str],
    path_checks: list[str],
    python_candidates: list[str],
) -> dict[str, Any]:
    return _run_remote_json_script(
        host,
        user=user,
        port=port,
        password_env=password_env,
        script=_probe_script(command_names, path_checks, workdir, python_candidates),
    )


def _run_remote_json_script(
    host: str,
    *,
    user: str | None,
    port: int,
    password_env: str | None,
    script: str,
) -> dict[str, Any]:
    encoded_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_python = f"import base64; exec(base64.b64decode({encoded_script!r}).decode('utf-8'))"
    command = _ssh_command(host, user=user, port=port, password_env=password_env, remote_python=remote_python)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RemoteAuditError(f"remote probe failed: {message}")
    combined_output = "\n".join(item for item in [completed.stdout.strip(), completed.stderr.strip()] if item)
    if not combined_output:
        raise RemoteAuditError("remote probe returned no output")
    for line in reversed([item.strip() for item in combined_output.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RemoteAuditError(f"remote probe did not return valid JSON: {combined_output[:200]}")


def _run_remote_shell_command(
    host: str,
    *,
    user: str | None,
    port: int,
    password_env: str | None,
    shell_command: str,
) -> None:
    command = _ssh_shell_command(host, user=user, port=port, password_env=password_env, shell_command=shell_command)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RemoteAuditError(f"remote command failed: {message}")


def _expand_remote_path(
    host: str,
    *,
    user: str | None,
    port: int,
    password_env: str | None,
    remote_path: str,
) -> str:
    payload = _run_remote_json_script(
        host,
        user=user,
        port=port,
        password_env=password_env,
        script="\n".join(
            [
                "import json",
                "import os",
                f"remote_path = os.path.expanduser({remote_path!r})",
                "print(json.dumps({'expanded_path': remote_path}, sort_keys=True))",
            ]
        ),
    )
    expanded = payload.get("expanded_path")
    if not isinstance(expanded, str) or not expanded.strip():
        raise RemoteAuditError(f"remote path expansion failed for: {remote_path}")
    return expanded


def _probe_script(command_names: list[str], path_checks: list[str], workdir: str | None, python_candidates: list[str]) -> str:
    command_list = ", ".join(repr(name) for name in command_names)
    path_list = ", ".join(repr(item) for item in path_checks)
    python_list = ", ".join(repr(item) for item in python_candidates)
    lines = [
        "import json",
        "import os",
        "import pathlib",
        "import platform",
        "import shutil",
        "import subprocess",
        "import sys",
        "",
        "def free_bytes(path):",
        "    stats = os.statvfs(path)",
        "    return int(stats.f_bavail * stats.f_frsize)",
        "",
        "def probe_python(candidate):",
        "    expanded = os.path.expanduser(candidate)",
        "    resolved = shutil.which(candidate) or (expanded if os.path.exists(expanded) else None)",
        "    if not resolved:",
        '        return {"candidate": candidate, "path": None, "version": None, "version_info": None}',
        "    completed = subprocess.run(",
        "        [resolved, '-c', 'import json, platform, sys; print(json.dumps({\"version\": platform.python_version(), \"version_info\": list(sys.version_info[:3])}))'],",
        "        check=False,",
        "        capture_output=True,",
        "        text=True,",
        "    )",
        "    if completed.returncode != 0:",
        '        return {"candidate": candidate, "path": resolved, "version": None, "version_info": None, "error": completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"}',
        "    try:",
        "        payload = json.loads(completed.stdout.strip())",
        "    except json.JSONDecodeError:",
        '        return {"candidate": candidate, "path": resolved, "version": None, "version_info": None, "error": completed.stdout.strip()}',
        '    return {"candidate": candidate, "path": resolved, "version": payload.get("version"), "version_info": payload.get("version_info")}',
        "",
        f"target_workdir = {workdir!r}",
        "if target_workdir:",
        "    os.chdir(os.path.expanduser(target_workdir))",
        "",
        f"command_names = [{command_list}]",
        f"path_checks = [{path_list}]",
        f"python_candidates = [{python_list}]",
        "payload = {",
        '    "python_version": platform.python_version(),',
        '    "python_version_info": list(sys.version_info[:3]),',
        '    "platform": platform.platform(),',
        '    "system": platform.system(),',
        '    "machine": platform.machine(),',
        '    "cwd": os.getcwd(),',
        '    "home": os.path.expanduser("~"),',
        '    "free_bytes_cwd": free_bytes(os.getcwd()),',
        '    "free_bytes_home": free_bytes(os.path.expanduser("~")),',
        '    "commands": {name: shutil.which(name) for name in command_names},',
        '    "python_candidates": {candidate: probe_python(candidate) for candidate in python_candidates},',
        '    "paths": {},',
        "}",
        "for item in path_checks:",
        "    expanded = os.path.expanduser(item)",
        '    payload["paths"][item] = {',
        '        "expanded": expanded,',
        '        "exists": os.path.exists(expanded),',
        '        "is_dir": os.path.isdir(expanded),',
        '        "is_file": os.path.isfile(expanded),',
        "    }",
        "print(json.dumps(payload, sort_keys=True))",
    ]
    return "\n".join(lines)


def _bootstrap_script(workdir: str, python_command: str, venv_name: str, *, recreate_venv: bool) -> str:
    lines = [
        "import json",
        "import os",
        "import pathlib",
        "import shutil",
        "import subprocess",
        "",
        f"workdir = os.path.expanduser({workdir!r})",
        f"python_command = os.path.expanduser({python_command!r})",
        f"venv_name = {venv_name!r}",
        f"recreate_venv = {recreate_venv!r}",
        "workdir_path = pathlib.Path(workdir)",
        "workdir_existed = workdir_path.exists()",
        "workdir_path.mkdir(parents=True, exist_ok=True)",
        "venv_path = workdir_path / venv_name",
        "venv_existed = venv_path.exists()",
        "venv_recreated = False",
        "if venv_existed and recreate_venv:",
        "    shutil.rmtree(venv_path)",
        "    venv_existed = False",
        "    venv_recreated = True",
        "if not venv_existed:",
        "    subprocess.run([python_command, '-m', 'venv', str(venv_path)], check=True)",
        "payload = {",
        '    "workdir": str(workdir_path),',
        '    "workdir_existed": workdir_existed,',
        '    "venv_dir": str(venv_path),',
        '    "venv_existed": venv_existed,',
        '    "venv_recreated": venv_recreated,',
        '    "python_command": python_command,',
        '    "python_bin": str(venv_path / "bin" / "python"),',
        '    "pip_bin": str(venv_path / "bin" / "pip"),',
        "}",
        "print(json.dumps(payload, sort_keys=True))",
    ]
    return "\n".join(lines)


def _sync_extract_script(remote_workdir: str, remote_archive: str) -> str:
    lines = [
        "import json",
        "import os",
        "import pathlib",
        "import shutil",
        "import subprocess",
        "",
        f"remote_workdir = os.path.expanduser({remote_workdir!r})",
        f"remote_archive = os.path.expanduser({remote_archive!r})",
        "workdir_path = pathlib.Path(remote_workdir)",
        "archive_path = pathlib.Path(remote_archive)",
        "workdir_path.mkdir(parents=True, exist_ok=True)",
        "subprocess.run(['tar', '-xzf', str(archive_path), '-C', str(workdir_path)], check=True)",
        "for cache_dir in workdir_path.rglob('__pycache__'):",
        "    shutil.rmtree(cache_dir, ignore_errors=True)",
        "pytest_cache = workdir_path / '.pytest_cache'",
        "if pytest_cache.exists():",
        "    shutil.rmtree(pytest_cache, ignore_errors=True)",
        "archive_path.unlink(missing_ok=True)",
        "print(json.dumps({'workdir': str(workdir_path), 'archive_path': str(archive_path)}, sort_keys=True))",
    ]
    return "\n".join(lines)


def _ssh_command(host: str, *, user: str | None, port: int, password_env: str | None, remote_python: str) -> list[str]:
    remote_command = f"python3 -c {_shell_quote(remote_python)}"
    remote_shell_command = f"sh -lc {_shell_quote(remote_command)}"
    return _ssh_shell_command(host, user=user, port=port, password_env=password_env, shell_command=remote_shell_command)


def _ssh_shell_command(host: str, *, user: str | None, port: int, password_env: str | None, shell_command: str) -> list[str]:
    target = f"{user}@{host}" if user else host
    ssh_args = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        target,
        shell_command,
    ]
    return _wrap_password_command(ssh_args, password_env=password_env)


def _copy_file_to_remote(
    local_path: Path,
    remote_path: str,
    *,
    host: str,
    user: str | None,
    port: int,
    password_env: str | None,
) -> None:
    target = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    scp_args = [
        "scp",
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(local_path),
        target,
    ]
    completed = subprocess.run(_wrap_password_command(scp_args, password_env=password_env), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RemoteAuditError(f"remote copy failed: {message}")


def _wrap_password_command(args: list[str], *, password_env: str | None) -> list[str]:
    if password_env is None:
        return args
    tcl_list = " ".join(_tcl_brace_quote(part) for part in args)
    expect_script = textwrap.dedent(
        f"""
        set timeout -1
        set cmd [list {tcl_list}]
        eval spawn -noecho $cmd
        expect {{
          -nocase -re {{password:}} {{
            send -- "$env({password_env})\\r"
            exp_continue
          }}
          timeout {{
            exit 124
          }}
          eof
        }}
        catch wait result
        set exit_code [lindex $result 3]
        exit $exit_code
        """
    ).strip()
    return ["expect", "-c", expect_script]


def _evaluate_probe(
    probe: dict[str, Any],
    *,
    min_python: tuple[int, int],
    min_free_gb: float,
    require_gpu: bool,
    require_paddle_training: bool,
    python_candidates: list[str],
    require_commands: list[str],
    optional_commands: list[str],
    require_paths: list[str],
    workdir: str | None,
) -> list[RemoteCheck]:
    checks: list[RemoteCheck] = []
    checks.append(_python_check(probe, min_python, python_candidates))
    checks.append(_platform_check(probe, require_paddle_training=require_paddle_training))
    checks.append(_disk_check(probe, min_free_gb, workdir))
    checks.append(_command_check(probe, require_commands, optional_commands))
    checks.append(_path_check(probe, require_paths))
    checks.append(_gpu_check(probe, require_gpu))
    return checks


def _python_check(probe: dict[str, Any], min_python: tuple[int, int], python_candidates: list[str]) -> RemoteCheck:
    selected = _select_python_candidate(probe, min_python, python_candidates=python_candidates)
    python_payload = probe.get("python_candidates")
    details: list[str] = []
    if isinstance(python_payload, dict):
        for candidate in python_candidates:
            item = python_payload.get(candidate)
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            version = item.get("version")
            error = item.get("error")
            if path and version:
                details.append(f"{candidate} -> {path} ({version})")
            elif path and error:
                details.append(f"{candidate} -> {path} error={error}")
            elif path:
                details.append(f"{candidate} -> {path}")
    if selected is None:
        return RemoteCheck("python", "fail", f"no remote python candidate satisfies >= {min_python[0]}.{min_python[1]}", details=details)
    return RemoteCheck("python", "pass", f"{selected['command']} -> {selected['version']} satisfies >= {min_python[0]}.{min_python[1]}", details=details)


def _disk_check(probe: dict[str, Any], min_free_gb: float, workdir: str | None) -> RemoteCheck:
    free_bytes = probe.get("free_bytes_cwd")
    if not isinstance(free_bytes, int | float):
        return RemoteCheck("disk", "fail", "remote probe did not report free_bytes_cwd")
    free_gb = float(free_bytes) / (1024**3)
    target = workdir or probe.get("cwd") or "."
    if free_gb < min_free_gb:
        return RemoteCheck("disk", "fail", f"{target} has {free_gb:.2f} GiB free < required {min_free_gb:.2f} GiB")
    return RemoteCheck("disk", "pass", f"{target} has {free_gb:.2f} GiB free")


def _platform_check(probe: dict[str, Any], *, require_paddle_training: bool) -> RemoteCheck:
    system = str(probe.get("system") or "")
    machine = str(probe.get("machine") or "")
    platform_name = str(probe.get("platform") or f"{system}-{machine}")
    if system == "Darwin" and machine == "arm64":
        summary = (
            "macOS arm64 host detected; PaddlePaddle official install guidance says macOS support requires x86_64, "
            "so PaddleOCR training is not a valid target here"
        )
        status = "fail" if require_paddle_training else "warn"
        return RemoteCheck("platform", status, summary, details=[platform_name])
    if not platform_name:
        return RemoteCheck("platform", "warn", "remote probe did not report platform details")
    return RemoteCheck("platform", "pass", f"remote platform reported as {platform_name}")


def _command_check(probe: dict[str, Any], require_commands: list[str], optional_commands: list[str]) -> RemoteCheck:
    commands = probe.get("commands")
    if not isinstance(commands, dict):
        return RemoteCheck("commands", "fail", "remote probe did not report command availability")
    missing_required = [name for name in require_commands if not commands.get(name)]
    missing_optional = [name for name in optional_commands if not commands.get(name)]
    details = []
    if missing_optional:
        details.append(f"optional missing: {', '.join(missing_optional)}")
    if missing_required:
        return RemoteCheck("commands", "fail", f"required commands missing: {', '.join(missing_required)}", details=details)
    return RemoteCheck("commands", "pass", "required remote commands are present", details=details)


def _path_check(probe: dict[str, Any], require_paths: list[str]) -> RemoteCheck:
    if not require_paths:
        return RemoteCheck("paths", "pass", "no required remote paths configured")
    paths = probe.get("paths")
    if not isinstance(paths, dict):
        return RemoteCheck("paths", "fail", "remote probe did not report path checks")
    missing = [path for path in require_paths if not isinstance(paths.get(path), dict) or not paths[path].get("exists")]
    if missing:
        return RemoteCheck("paths", "fail", f"required remote paths missing: {', '.join(missing)}")
    details = [f"{path} -> {paths[path].get('expanded')}" for path in require_paths if isinstance(paths.get(path), dict)]
    return RemoteCheck("paths", "pass", "required remote paths are present", details=details)


def _gpu_check(probe: dict[str, Any], require_gpu: bool) -> RemoteCheck:
    commands = probe.get("commands")
    if not isinstance(commands, dict):
        return RemoteCheck("gpu", "fail" if require_gpu else "warn", "remote probe did not report command availability for GPU check")
    has_gpu_tool = bool(commands.get("nvidia-smi"))
    if require_gpu and not has_gpu_tool:
        return RemoteCheck("gpu", "fail", "nvidia-smi is missing but GPU is required")
    if has_gpu_tool:
        return RemoteCheck("gpu", "pass", "nvidia-smi is available")
    return RemoteCheck("gpu", "warn", "nvidia-smi is not available")


def _write_remote_audit(report: RemoteAuditReport, output_dir: Path) -> None:
    (output_dir / "remote-audit.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Remote Host Audit",
        "",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Host: `{report.host}`",
        f"User: `{report.user or ''}`",
        f"Workdir: `{report.workdir or ''}`",
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
    (output_dir / "remote-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_remote_bootstrap(report: RemoteBootstrapReport, output_dir: Path) -> None:
    (output_dir / "remote-bootstrap.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Remote Host Bootstrap",
        "",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Host: `{report.host}`",
        f"User: `{report.user or ''}`",
        f"Workdir: `{report.workdir}`",
        f"Python: `{report.python_command}`",
        f"Virtualenv: `{report.venv_dir}`",
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
    (output_dir / "remote-bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_remote_sync(report: RemoteSyncReport, output_dir: Path) -> None:
    (output_dir / "remote-sync.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Remote Workspace Sync",
        "",
        f"Passed: `{'yes' if report.passed else 'no'}`",
        f"Host: `{report.host}`",
        f"User: `{report.user or ''}`",
        f"Local root: `{report.local_root}`",
        f"Remote workdir: `{report.remote_workdir}`",
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
    (output_dir / "remote-sync.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def shutil_which(command: str) -> str | None:
    return subprocess.run(
        ["sh", "-lc", f"command -v {_shell_quote(command)}"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or None


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _tcl_brace_quote(value: str) -> str:
    return "{" + value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}") + "}"


def _select_python_candidate(
    probe: dict[str, Any],
    min_python: tuple[int, int],
    *,
    python_candidates: list[str] | None = None,
) -> dict[str, str] | None:
    payload = probe.get("python_candidates")
    if not isinstance(payload, dict):
        return None
    candidates = python_candidates or list(payload)
    for candidate in candidates:
        item = payload.get(candidate)
        if not isinstance(item, dict):
            continue
        version_info = item.get("version_info")
        path = item.get("path")
        version = item.get("version")
        if not isinstance(version_info, list) or len(version_info) < 2 or not path or not version:
            continue
        key = (int(version_info[0]), int(version_info[1]))
        if key[:2] < min_python:
            continue
        return {"candidate": candidate, "command": str(path), "version": str(version)}
    return None


def _create_workspace_archive(local_root: Path, archive_path: Path, exclude_patterns: list[str]) -> None:
    command = ["tar", "-czf", str(archive_path)]
    for pattern in exclude_patterns:
        command.extend(["--exclude", pattern])
    command.extend(["-C", str(local_root), "."])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RemoteAuditError(f"failed to create workspace archive: {message}")
