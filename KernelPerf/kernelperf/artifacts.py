from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Callable

from .models import JobRecord, KernelArtifact, SourceFile


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError("source paths must use non-empty POSIX relative paths")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe source path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe source path: {value!r}")
    if len(value) > 240:
        raise ValueError(f"source path is too long: {value!r}")
    return path


def source_tree(kernel: KernelArtifact) -> tuple[list[SourceFile], str, list[str]]:
    """Return a normalized source tree while retaining legacy single-file support."""
    if kernel.source_files:
        if kernel.source is not None:
            raise ValueError("source and source_files are mutually exclusive")
        files = kernel.source_files
        entry_source = kernel.entry_source
        if not entry_source:
            raise ValueError("multi-file submissions require entry_source")
    else:
        if kernel.source is None:
            raise ValueError("source submissions require source or source_files")
        entry_source = kernel.entry_source or "adapter.cu"
        files = [SourceFile(path=entry_source, content=kernel.source)]

    paths: set[str] = set()
    normalized: list[SourceFile] = []
    for item in files:
        path = safe_relative_path(item.path).as_posix()
        if path in paths:
            raise ValueError(f"duplicate source path: {path}")
        paths.add(path)
        normalized.append(SourceFile(path=path, content=item.content))

    entry_source = safe_relative_path(entry_source).as_posix()
    if entry_source not in paths:
        raise ValueError(f"entry_source is not present in source_files: {entry_source}")
    compile_units = [safe_relative_path(value).as_posix() for value in kernel.compile_units]
    if len(set(compile_units)) != len(compile_units):
        raise ValueError("compile_units must not contain duplicates")
    missing = sorted(set(compile_units) - paths)
    if missing:
        raise ValueError(f"compile_units are not present in source_files: {missing}")
    if entry_source in compile_units:
        raise ValueError("entry_source is already compiled and cannot also be a compile_unit")
    return normalized, entry_source, compile_units


def write_source_tree(root: Path, files: list[SourceFile]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        target = root.joinpath(*safe_relative_path(item.path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8", newline="")


def source_snapshot(kernel: KernelArtifact) -> dict[str, object] | None:
    if kernel.kind != "source":
        return None
    files, entry_source, compile_units = source_tree(kernel)
    return {
        "entry_source": entry_source,
        "compile_units": compile_units,
        "files": [item.model_dump() for item in files],
    }


class SourceArchive:
    """Persist each accepted source artifact outside the job database."""

    def __init__(
        self,
        root: str | Path,
        package_builder: Callable[[KernelArtifact], dict[str, object] | None] = source_snapshot,
    ) -> None:
        self.root = Path(root)
        self.package_builder = package_builder

    def archive_job(self, job: JobRecord) -> None:
        for kernel in job.kernels:
            snapshot = self.package_builder(kernel)
            if snapshot is None:
                continue
            operator_id = str(kernel.metadata.get("operator_id", "unbound"))
            configuration_id = str(kernel.metadata.get("configuration_id", "")).strip()
            target = self.root / _component(job.job_id) / _component(operator_id)
            if configuration_id:
                target = target / _component(configuration_id)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
            try:
                write_source_tree(
                    temporary / "files",
                    [SourceFile.model_validate(item) for item in snapshot["files"]],
                )
                plugin = {
                    "job_id": job.job_id,
                    "method_name": kernel.name,
                    "kind": kernel.kind,
                    "language": kernel.language,
                    "operator_id": operator_id,
                    "base_format": kernel.metadata.get("base_format"),
                    "build_profile": kernel.metadata.get("build_profile"),
                    "configuration_id": kernel.metadata.get("configuration_id"),
                    "candidate_group": kernel.metadata.get("candidate_group"),
                    "selection_role": kernel.metadata.get("selection_role", "candidate"),
                    **snapshot.get("plugin", {}),
                }
                plugin.setdefault("entry_source", snapshot["entry_source"])
                plugin.setdefault("compile_units", snapshot["compile_units"])
                plugin.setdefault("files", [item["path"] for item in snapshot["files"]])
                if "source_sha256" in snapshot:
                    plugin["source_sha256"] = snapshot["source_sha256"]
                (temporary / "plugin.json").write_text(
                    json.dumps(plugin, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)


def _component(value: str) -> str:
    normalized = _SAFE_COMPONENT.sub("-", value.strip()).strip("-.")
    return (normalized or "submission")[:128]
