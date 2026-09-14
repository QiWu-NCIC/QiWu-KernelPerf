from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernelperf.artifacts import source_snapshot
from kernelperf.models import KernelArtifact


GENERATED_FILES = frozenset({
    "CMakeLists.txt",
    "plugin.json",
    "README-QIWU-PLUGIN.md",
    "include/qiwu/spmv_plugin.cuh",
    "include/qiwu/gpu_runtime.h",
    "examples/standalone.cu",
})


def make_source_package(kernel: KernelArtifact, contract_path: Path) -> dict[str, object] | None:
    snapshot = source_snapshot(kernel)
    if snapshot is None:
        return None
    submitted = {str(item["path"]) for item in snapshot["files"]}
    conflicts = sorted(submitted & GENERATED_FILES)
    if conflicts:
        raise ValueError(f"source submission uses reserved plugin paths: {conflicts}")

    metadata = kernel.metadata
    dependencies = ["CUDA Toolkit", "cuSPARSE"]
    if metadata.get("build_profile") == "ghost-cuda":
        dependencies.extend([
            "GHOST commit 22a004dbbfb604d7b04b0bde813c8100d403cfcc",
            "hwloc",
            "BLAS",
            "cuBLAS",
            "cuRAND",
        ])
    plugin = {
        "schema_version": 1,
        "kind": "qiwu-spmv-source-plugin",
        "plugin_id": kernel.name,
        "operator_id": metadata["operator_id"],
        "base_format": metadata["base_format"],
        "supported_dtypes": ["fp32", "fp64"],
        "entry_source": snapshot["entry_source"],
        "compile_units": snapshot["compile_units"],
        "include_dirs": snapshot["include_dirs"],
        "build_profile": metadata.get("build_profile", ""),
        "configuration_id": metadata.get("configuration_id", ""),
        "candidate_group": metadata.get("candidate_group", ""),
        "dependencies": dependencies,
        "snapshot_kind": "standalone-source-plugin",
    }
    configurations = _configurations(snapshot["files"])
    if configurations:
        plugin["configurations"] = configurations
    generated = [
        {
            "path": "include/qiwu/spmv_plugin.cuh",
            "content": contract_path.read_text(encoding="utf-8"),
        },
        {
            "path": "include/qiwu/gpu_runtime.h",
            "content": _runtime_header(contract_path).read_text(encoding="utf-8"),
        },
        {"path": "README-QIWU-PLUGIN.md", "content": _build_readme()},
        {
            "path": "CMakeLists.txt",
            "content": _cmake(
                snapshot["entry_source"],
                snapshot["compile_units"],
                snapshot["include_dirs"],
                str(metadata.get("build_profile", "")),
            ),
        },
        {"path": "examples/standalone.cu", "content": _standalone_example()},
    ]
    files = [*snapshot["files"], *generated]
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: str(value["path"])):
        digest.update(str(item["path"]).encode())
        digest.update(b"\0")
        digest.update(str(item["content"]).encode())
        digest.update(b"\0")
    source_sha256 = digest.hexdigest()
    plugin.update({
        "source_sha256": source_sha256,
        "files": [item["path"] for item in files],
    })
    return {
        **snapshot,
        "files": files,
        "plugin": plugin,
        "source_sha256": source_sha256,
    }


def _runtime_header(contract_path: Path) -> Path:
    sibling = contract_path.with_name("gpu_runtime.h")
    if sibling.is_file():
        return sibling
    # Tests and external callers may provide a temporary contract file only;
    # use the repository's stable public compatibility header in that case.
    return Path(__file__).resolve().parents[2] / "include" / "qiwu" / "gpu_runtime.h"


def _cmake(entry_source: object, compile_units: object, include_dirs: object, build_profile: str) -> str:
    unit_lines = ";".join(str(value) for value in compile_units)
    include_lines = ";".join(str(value) for value in include_dirs)
    profile_setup = ""
    if build_profile == "ghost-cuda":
        profile_setup = """
set(GHOST_ROOT "" CACHE PATH "Pinned GHOST installation prefix")
find_path(QIWU_GHOST_INCLUDE_DIR ghost.h HINTS "${GHOST_ROOT}/include" REQUIRED)
find_library(QIWU_GHOST_LIBRARY ghost
  HINTS "${GHOST_ROOT}/lib/ghost" "${GHOST_ROOT}/lib" REQUIRED)
find_library(QIWU_BLAS_LIBRARY NAMES openblas blas)
list(APPEND QIWU_PLUGIN_EXTRA_INCLUDE_DIRS "${QIWU_GHOST_INCLUDE_DIR}")
list(APPEND QIWU_PLUGIN_EXTRA_LIBRARIES
  "${QIWU_GHOST_LIBRARY}" hwloc CUDA::cublas CUDA::curand)
if(QIWU_BLAS_LIBRARY)
  list(APPEND QIWU_PLUGIN_EXTRA_LIBRARIES "${QIWU_BLAS_LIBRARY}")
endif()
list(APPEND QIWU_PLUGIN_EXTRA_LINK_OPTIONS "-Wl,--allow-shlib-undefined")
"""
    return f"""cmake_minimum_required(VERSION 3.20)
project(qiwu_spmv_plugin LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(QIWU_PLUGIN_ENTRY "" CACHE STRING "Plugin entry source")
if(NOT QIWU_PLUGIN_ENTRY)
  file(READ "${{CMAKE_CURRENT_SOURCE_DIR}}/plugin.json" QIWU_PLUGIN_MANIFEST)
  string(JSON QIWU_PLUGIN_ENTRY GET "${{QIWU_PLUGIN_MANIFEST}}" entry_source)
endif()
set(QIWU_PLUGIN_COMPILE_UNITS "{unit_lines}" CACHE STRING "Additional source files")
set(QIWU_PLUGIN_INCLUDE_DIRS "{include_lines}" CACHE STRING "Additional source include directories")
set(QIWU_PLUGIN_EXTRA_INCLUDE_DIRS "" CACHE STRING "Additional dependency include directories")
set(QIWU_PLUGIN_EXTRA_LIBRARIES "" CACHE STRING "Additional dependency libraries")
set(QIWU_PLUGIN_EXTRA_LINK_OPTIONS "" CACHE STRING "Additional linker options")
set(QIWU_PLUGIN_SOURCES "${{QIWU_PLUGIN_ENTRY}}" ${{QIWU_PLUGIN_COMPILE_UNITS}})
{profile_setup}

function(qiwu_add_spmv_target dtype)
  add_library(qiwu_spmv_${{dtype}} STATIC ${{QIWU_PLUGIN_SOURCES}})
  target_compile_features(qiwu_spmv_${{dtype}} PUBLIC cxx_std_17)
  target_include_directories(qiwu_spmv_${{dtype}}
    PUBLIC "${{CMAKE_CURRENT_SOURCE_DIR}}/include"
    PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}" ${{QIWU_PLUGIN_INCLUDE_DIRS}} ${{QIWU_PLUGIN_EXTRA_INCLUDE_DIRS}}
  )
  target_link_libraries(qiwu_spmv_${{dtype}}
    PUBLIC ${{QIWU_PLUGIN_EXTRA_LIBRARIES}} CUDA::cudart CUDA::cusparse
  )
  target_link_options(qiwu_spmv_${{dtype}} PUBLIC ${{QIWU_PLUGIN_EXTRA_LINK_OPTIONS}})
  set_target_properties(qiwu_spmv_${{dtype}} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_spmv_${{dtype}} PUBLIC QIWU_SPMV_FP64=1)
  endif()

  add_executable(qiwu_spmv_example_${{dtype}} examples/standalone.cu)
  target_link_libraries(qiwu_spmv_example_${{dtype}} PRIVATE qiwu_spmv_${{dtype}})
  set_target_properties(qiwu_spmv_example_${{dtype}} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

qiwu_add_spmv_target(fp32)
qiwu_add_spmv_target(fp64)
"""


def _configurations(files: object) -> list[dict[str, str]]:
    by_path = {str(item["path"]): str(item["content"]) for item in files}
    source_paths = set(by_path)
    for manifest_path in ("sweep_manifest.json", "sweep.json"):
        if manifest_path not in by_path:
            continue
        raw = json.loads(by_path[manifest_path])
        if not isinstance(raw, list):
            raise ValueError(f"{manifest_path} must contain a JSON list")
        configurations = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"{manifest_path} entries must be objects")
            configuration_id = str(item.get("configuration_id", "")).strip()
            entry_source = str(item.get("entry_source", "")).strip()
            if not configuration_id or entry_source not in source_paths:
                raise ValueError(f"invalid configuration in {manifest_path}: {item}")
            configuration = {
                "configuration_id": configuration_id,
                "entry_source": entry_source,
            }
            if item.get("method_name"):
                configuration["method_name"] = str(item["method_name"])
            configurations.append(configuration)
        return configurations
    return []


def _build_readme() -> str:
    return """# Qiwu SpMV source plugin

This directory is a standalone CUDA source plugin. It does not require KernelPerf.

```bash
cmake -S . -B build
cmake --build build -j
./build/qiwu_spmv_example_fp32
./build/qiwu_spmv_example_fp64
```

`plugin.json` is the package manifest. It records the payload files, content hash,
evaluated entry source, configurations, and dependencies. CMake uses its
`entry_source` by default. Select another packaged variant with
`-DQIWU_PLUGIN_ENTRY=variants/name.cu`. External libraries
can be supplied with `QIWU_PLUGIN_EXTRA_INCLUDE_DIRS` and
`QIWU_PLUGIN_EXTRA_LIBRARIES`. GHOST packages accept `-DGHOST_ROOT=/path/to/install`.

Applications include `include/qiwu/spmv_plugin.cuh`, link either
`qiwu_spmv_fp32` or `qiwu_spmv_fp64`, and call `qiwu_spmv_preprocess`,
`qiwu_spmv_solve`, and `qiwu_spmv_destroy`. `examples/standalone.cu` is the
smallest complete caller.
"""


def _standalone_example() -> str:
    return Path(__file__).with_name("standalone.cu").read_text(encoding="utf-8")
