from __future__ import annotations

import base64
import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import BackendInfo


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


class Backend(ABC):
    """One configured worker and its command transport."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.worker_id = str(spec["worker_id"])
        self.backend_id = str(spec["backend_id"])
        self.endpoint = str(spec["endpoint"])
        self.labels = {str(k): str(v) for k, v in spec.get("labels", {}).items()}
        self.command_timeout_seconds = int(spec.get("command_timeout_seconds", 1200))
        public_metadata = dict(spec.get("metadata", {}))
        public_metadata.update(
            {
                "transport": spec["transport"],
                "endpoint": self.endpoint,
                "labels": self.labels,
                "build_profiles": sorted(spec.get("build_profiles", {})),
            }
        )
        self._info = BackendInfo(
            backend_id=self.backend_id,
            kind=str(spec["kind"]),
            name=str(spec.get("name", self.backend_id)),
            vendor=str(spec.get("vendor", "unknown")),
            device=str(spec.get("device", "unknown")),
            peak_gflops=float(spec.get("peak_gflops", 1.0)),
            peak_gflops_by_dtype={
                str(key): float(value)
                for key, value in spec.get("peak_gflops_by_dtype", {}).items()
            },
            memory_bandwidth_gbs=float(spec.get("memory_bandwidth_gbs", 1.0)),
            supported_languages=list(spec.get("supported_languages", [])),
            metadata=public_metadata,
        )

    def info(self) -> BackendInfo:
        return self._info

    def healthcheck(self) -> tuple[bool, str]:
        healthcheck = self.spec.get("healthcheck")
        if not healthcheck:
            return True, "configured"
        try:
            completed = self.run(
                list(healthcheck["command"]),
                cwd=str(healthcheck.get("cwd", ".")),
                timeout=int(healthcheck.get("timeout_seconds", 20)),
            )
        except Exception as exc:
            return False, str(exc)
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            return False, output or f"healthcheck exited {completed.returncode}"
        return True, output or "healthy"

    @abstractmethod
    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        raise NotImplementedError

    def write_bytes(self, path: str, content: bytes) -> None:
        raise NotImplementedError(
            f"backend {self.backend_id!r} does not support binary artifact uploads"
        )

    @abstractmethod
    def path_exists(self, path: str, *, executable: bool = False) -> bool:
        raise NotImplementedError


class LocalBackend(Backend):
    def _environment(self, overrides: dict[str, str] | None) -> dict[str, str]:
        env = dict(os.environ)
        configured = self.spec.get("environment", {})
        path_prefix = [str(value) for value in configured.get("path_prepend", [])]
        library_prefix = [
            str(value) for value in configured.get("library_path_prepend", [])
        ]
        if path_prefix:
            env["PATH"] = os.pathsep.join(path_prefix + [env.get("PATH", "")])
        if library_prefix:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                library_prefix + [env.get("LD_LIBRARY_PATH", "")]
            )
        env.update({str(k): str(v) for k, v in configured.get("variables", {}).items()})
        env.update(overrides or {})
        return env

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env=self._environment(env),
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or self.command_timeout_seconds,
        )

    def write_text(self, path: str, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def write_bytes(self, path: str, content: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def path_exists(self, path: str, *, executable: bool = False) -> bool:
        target = Path(path)
        return target.is_file() and (not executable or os.access(target, os.X_OK))


class SshBackend(Backend):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        ssh = spec["ssh"]
        self.ssh_host = str(ssh["host"])
        self.ssh_user = ssh.get("user")
        self.ssh_port = ssh.get("port")
        self.identity_file = ssh.get("identity_file")

    def _ssh_base_command(self) -> list[str]:
        target = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        if self.identity_file:
            command.extend(["-i", str(self.identity_file)])
        if self.ssh_port:
            command.extend(["-p", str(self.ssh_port)])
        command.append(target)
        return command

    def _remote_command(
        self,
        command: list[str],
        cwd: str,
        env: dict[str, str] | None,
    ) -> str:
        configured = self.spec.get("environment", {})
        parts = [". /etc/profile >/dev/null 2>&1 || true"]
        for module in configured.get("modules", []):
            parts.append(f"module load {_shell_quote(str(module))}")
        parts.append(f"cd {_shell_quote(cwd)}")
        path_prefix = ":".join(
            str(value) for value in configured.get("path_prepend", [])
        )
        if path_prefix:
            parts.append(f"export PATH={_shell_quote(path_prefix)}:\"$PATH\"")
        library_prefix = ":".join(
            str(value) for value in configured.get("library_path_prepend", [])
        )
        if library_prefix:
            parts.append(
                f"export LD_LIBRARY_PATH={_shell_quote(library_prefix)}:\"${{LD_LIBRARY_PATH:-}}\""
            )
        variables = {
            **{str(k): str(v) for k, v in configured.get("variables", {}).items()},
            **(env or {}),
        }
        parts.extend(
            f"export {key}={_shell_quote(value)}" for key, value in variables.items()
        )
        command_parts = list(command)
        scheduler = self.spec.get("scheduler") or {}
        if scheduler.get("type") == "slurm":
            slurm = ["srun", "--job-name", str(scheduler.get("job_name", "kernelperf"))]
            for key in ("account", "partition", "gres", "cpus_per_task", "mem", "time"):
                value = scheduler.get(key)
                if value is None or value == "":
                    continue
                option = "--" + key.replace("_", "-")
                slurm.extend([option, str(value)])
            slurm.extend(["--kill-on-bad-exit=1", "--wait=0"])
            command_parts = slurm + command_parts
        parts.append("exec " + " ".join(_shell_quote(part) for part in command_parts))
        return " && ".join(parts)

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._ssh_base_command() + [self._remote_command(command, cwd, env)],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or self.command_timeout_seconds,
        )

    def write_text(self, path: str, content: str) -> None:
        target = Path(path)
        encoded = base64.b64encode(content.encode()).decode()
        command = (
            f"mkdir -p {_shell_quote(str(target.parent))} && "
            f"base64 --decode > {_shell_quote(str(target))}"
        )
        completed = subprocess.run(
            self._ssh_base_command() + [command],
            input=encoded,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or f"failed to write remote file {path}")

    def write_bytes(self, path: str, content: bytes) -> None:
        target = Path(path)
        encoded = base64.b64encode(content).decode()
        command = (
            f"mkdir -p {_shell_quote(str(target.parent))} && "
            f"base64 --decode > {_shell_quote(str(target))}"
        )
        completed = subprocess.run(
            self._ssh_base_command() + [command],
            input=encoded,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or f"failed to write remote file {path}")

    def path_exists(self, path: str, *, executable: bool = False) -> bool:
        flag = "-x" if executable else "-f"
        completed = self.run(["test", flag, path], cwd="/", timeout=20)
        return completed.returncode == 0


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._worker_ids: set[str] = set()

    def register(self, backend: Backend) -> None:
        if backend.backend_id in self._backends:
            raise ValueError(f"duplicate backend: {backend.backend_id}")
        if backend.worker_id in self._worker_ids:
            raise ValueError(f"duplicate worker: {backend.worker_id}")
        self._backends[backend.backend_id] = backend
        self._worker_ids.add(backend.worker_id)

    def get(self, backend_id: str) -> Backend:
        if backend_id not in self._backends:
            raise KeyError(f"Unknown backend: {backend_id}")
        return self._backends[backend_id]

    def backends(self) -> list[Backend]:
        return list(self._backends.values())

    def list(self) -> list[BackendInfo]:
        return [backend.info() for backend in self._backends.values()]

    def resolve(self, selectors: list[str]) -> list[str]:
        resolved: list[str] = []
        for selector in selectors:
            if selector in self._backends:
                matches = [selector]
            else:
                matches = [
                    backend.backend_id
                    for backend in self._backends.values()
                    if backend.info().kind == selector
                ]
            if not matches:
                raise KeyError(f"Unknown backend or backend kind: {selector}")
            for backend_id in matches:
                if backend_id not in resolved:
                    resolved.append(backend_id)
        return resolved


def backend_registry_from_config(path: str | Path) -> BackendRegistry:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"worker config not found: {config_path}")
    registry = BackendRegistry()
    transports = {"local": LocalBackend, "ssh": SshBackend}
    for spec in json.loads(config_path.read_text()):
        transport = str(spec["transport"])
        if transport not in transports:
            raise ValueError(f"unsupported worker transport: {transport}")
        registry.register(transports[transport](spec))
    return registry
