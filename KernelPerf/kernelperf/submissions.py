from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .models import JobSubmitRequest, KernelArtifact, SourceFile


IGNORED = {".git", "__pycache__", "build", "dist", "node_modules"}
MAX_FILES = 4096
MAX_BYTES = 32 * 1024 * 1024


def _safe_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe submission path: {value!r}")
    return value


def _source_files(root: Path, include_prefixes: list[str] | None = None) -> list[SourceFile]:
    files: list[SourceFile] = []
    total = 0
    prefixes = [value.rstrip("/") for value in (include_prefixes or [])]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        relative_name = relative.as_posix()
        if prefixes and not any(
            relative_name == prefix or relative_name.startswith(prefix + "/")
            for prefix in prefixes
        ):
            continue
        if relative.name in {"submission.json", "CMakeLists.txt", "README-QIWU-PLUGIN.md"}:
            continue
        if relative.as_posix() in {
            "examples/standalone.cu",
            "include/qiwu/spmv_plugin.cuh",
            "include/qiwu/gpu_runtime.h",
        }:
            continue
        if any(part in IGNORED for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"submission cannot contain symlinks: {relative}")
        if not path.is_file():
            continue
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # A vendored upstream tree may contain legacy-encoded comments or
            # documentation. Preserve its source semantics while normalizing
            # the transport representation used by the source archive.
            try:
                content = raw.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise ValueError(f"submission files must be UTF-8 or GB18030 text: {relative}") from exc
        total += len(content.encode("utf-8"))
        files.append(SourceFile(path=_safe_relative(relative.as_posix()), content=content))
        if len(files) > MAX_FILES or total > MAX_BYTES:
            raise ValueError(f"submission exceeds {MAX_FILES} files or {MAX_BYTES} bytes")
    if not files:
        raise ValueError("submission contains no source files")
    return files


def load_submission_artifacts(
    root: str | Path,
    *,
    operator_id: str | None = None,
    configuration_ids: list[str] | None = None,
    cuda_version: str = "",
    language: str | None = None,
) -> list[KernelArtifact]:
    root = Path(root).resolve()
    manifest_path = root / "submission.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"submission must be a directory containing submission.json: {root}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("submission.json schema_version must be 1")
    declared_operator = str(manifest.get("operator_id", "")).strip()
    if not declared_operator:
        raise ValueError("submission.json requires operator_id")
    supported = [str(value) for value in manifest.get("supported_operators", [declared_operator])]
    if operator_id and operator_id not in supported:
        raise ValueError(f"submission supports {supported}, not {operator_id}")
    selected_operator = operator_id or declared_operator
    declared_languages = [
        str(value).strip().lower()
        for value in manifest.get("languages", [manifest.get("language", "cuda")])
    ]
    if not declared_languages or any(value not in {"cuda", "hip"} for value in declared_languages):
        raise ValueError("submission.json languages must contain cuda or hip")
    selected_language = (language or ("cuda" if "cuda" in declared_languages else declared_languages[0])).strip().lower()
    if selected_language not in declared_languages:
        raise ValueError(
            f"submission supports languages {declared_languages}, not {selected_language!r}"
        )
    language_names = manifest.get("method_name_by_language", {})
    if language_names and not isinstance(language_names, dict):
        raise ValueError("method_name_by_language must be an object")
    name = str(
        language_names.get(selected_language)
        or manifest.get("method_name")
        or root.name
    ).strip()
    base_metadata = {
        "operator_id": selected_operator,
        "base_format": str(manifest.get("base_format", "unmarked")),
        "candidate_group": str(manifest.get("candidate_group", "")),
    }
    build_profiles = manifest.get("build_profile_by_language", {})
    if build_profiles and not isinstance(build_profiles, dict):
        raise ValueError("build_profile_by_language must be an object")
    selected_profile = build_profiles.get(selected_language, manifest.get("build_profile"))
    if selected_profile is not None:
        base_metadata["build_profile"] = str(selected_profile)
    for key in ("configuration_id", "selection_role"):
        if manifest.get(key) is not None:
            base_metadata[key] = str(manifest[key])
    kind = str(manifest.get("kind", "source"))
    if kind == "source":
        include_by_language = manifest.get("source_include_by_language", {})
        include_prefixes = include_by_language.get(selected_language)
        if include_prefixes is None:
            include_prefixes = manifest.get("source_include")
        if include_prefixes is not None:
            if not isinstance(include_prefixes, list) or not all(isinstance(value, str) for value in include_prefixes):
                raise ValueError("source_include(_by_language) must be a list of paths")
            include_prefixes = [_safe_relative(value) for value in include_prefixes]
        files = _source_files(root, include_prefixes)
        paths = {item.path for item in files}
        entry_source = str(manifest.get("entry_source") or "adapter.cu")
        if entry_source not in paths:
            raise ValueError(f"entry_source is missing from submission: {entry_source}")
        compile_units = [_safe_relative(str(value)) for value in manifest.get("compile_units", [])]
        missing = sorted(set(compile_units) - paths)
        if missing:
            raise ValueError("compile_units are missing: " + ", ".join(missing))
        include_dirs_by_language = manifest.get("include_dirs_by_language", {})
        raw_include_dirs = include_dirs_by_language.get(selected_language, manifest.get("include_dirs", []))
        if not isinstance(raw_include_dirs, list):
            raise ValueError("include_dirs(_by_language) must be a list of paths")
        include_dirs = [_safe_relative(str(value)) for value in raw_include_dirs]
        if len(set(include_dirs)) != len(include_dirs):
            raise ValueError("include_dirs must not contain duplicates")
        if any(
            not any(path == directory or path.startswith(directory + "/") for path in paths)
            for directory in include_dirs
        ):
            raise ValueError("include_dirs must point inside the submitted source tree")
        configurations_file = manifest.get("configurations_file")
        if not configurations_file:
            return [KernelArtifact(
                name=_versioned_name(name, manifest, cuda_version, selected_language),
                kind="source",
                language=selected_language,
                source_files=files,
                entry_source=entry_source,
                compile_units=compile_units,
                include_dirs=include_dirs,
                metadata=base_metadata,
            )]
        configurations_path = root / _safe_relative(str(configurations_file))
        configurations = json.loads(configurations_path.read_text(encoding="utf-8"))
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("configurations_file must contain a non-empty JSON list")
        artifacts = []
        requested = set(configuration_ids or [])
        for configuration in configurations:
            configuration_id = str(configuration.get("configuration_id", "")).strip()
            if requested and configuration_id not in requested:
                continue
            configuration_entry = str(configuration.get("entry_source", "")).strip()
            if not configuration_id or configuration_entry not in paths:
                raise ValueError(f"invalid configuration: {configuration}")
            metadata = dict(base_metadata)
            metadata["configuration_id"] = configuration_id
            metadata["selection_role"] = "candidate"
            if configuration.get("base_format"):
                metadata["base_format"] = str(configuration["base_format"])
            elif metadata["base_format"] == "auto":
                metadata["base_format"] = configuration_id.split("-", 1)[0]
            configuration_names = configuration.get("method_name_by_language", {})
            if configuration_names and not isinstance(configuration_names, dict):
                raise ValueError("configuration method_name_by_language must be an object")
            configuration_name = str(
                configuration_names.get(selected_language)
                or configuration.get("method_name")
                or configuration_id
            )
            artifacts.append(KernelArtifact(
                name=_versioned_name(configuration_name, manifest, cuda_version, selected_language),
                kind="source",
                language=selected_language,
                source_files=files,
                entry_source=configuration_entry,
                compile_units=compile_units,
                include_dirs=include_dirs,
                metadata=metadata,
            ))
        if requested and not artifacts:
            raise ValueError("configuration_ids did not match any configuration")
        return artifacts
    if kind == "object":
        object_path = str(manifest.get("object_path", ""))
        object_file = (root / _safe_relative(object_path)).resolve()
        if root not in object_file.parents or not object_file.is_file():
            raise ValueError("object_path must point inside the submission directory")
        architecture_key = "hip_arch" if selected_language == "hip" else "cuda_arch"
        base_metadata[architecture_key] = str(manifest.get(architecture_key, ""))
        if not base_metadata[architecture_key]:
            raise ValueError(f"object submissions require {architecture_key}")
        return [KernelArtifact(
            name=name,
            kind="object",
            language=selected_language,
            object_base64=base64.b64encode(object_file.read_bytes()).decode(),
            metadata=base_metadata,
        )]
    raise ValueError("submission.json kind must be source or object")


def _versioned_name(
    name: str,
    manifest: dict[str, Any],
    cuda_version: str,
    language: str = "cuda",
) -> str:
    if language == "hip" and name.startswith("cuSPARSE"):
        name = name.replace("cuSPARSE", "hipSPARSE", 1)
    if language == "cuda" and manifest.get("cuda_version_in_method_name"):
        if cuda_version:
            return name.replace("cuSPARSE", f"cuSPARSE CUDA {cuda_version}", 1)
    return name


def request_for_submission(
    root: str | Path,
    *,
    backend_id: str,
    dataset_id: str,
    operator_id: str | None = None,
    configuration_ids: list[str] | None = None,
    matrix_ids: list[str] | None = None,
    cuda_version: str = "",
    language: str | None = None,
) -> JobSubmitRequest:
    kernels = load_submission_artifacts(
        root,
        operator_id=operator_id,
        configuration_ids=configuration_ids,
        cuda_version=cuda_version,
        language=language,
    )
    operator = str(kernels[0].metadata["operator_id"])
    suite = operator.split(".", 1)[0]
    return JobSubmitRequest(
        generator_id=str(kernels[0].metadata.get("candidate_group") or kernels[0].name),
        submitter="maintainer",
        backends=[backend_id],
        suites=[suite],
        dataset_id=dataset_id,
        operator_ids=[operator],
        matrix_ids=matrix_ids or None,
        kernels=kernels,
        tags={"submission_path": str(Path(root).as_posix())},
    )
