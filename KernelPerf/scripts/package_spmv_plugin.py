from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.spmv.package import make_source_package
from kernelperf.artifacts import write_source_tree
from kernelperf.models import KernelArtifact, SourceFile


IGNORED_DIRECTORIES = {".git", "__pycache__", "build", "dist", "node_modules"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a standalone Qiwu SpMV source plugin package.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--entry-source", required=True)
    parser.add_argument("--compile-unit", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin-id", required=True)
    parser.add_argument("--base-format", required=True)
    parser.add_argument("--build-profile", default="")
    parser.add_argument("--configuration-id", default="")
    parser.add_argument("--candidate-group", default="")
    parser.add_argument("--source-repository", default="PlayGround")
    parser.add_argument("--source-path", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.source_dir.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        files.append(SourceFile(path=relative.as_posix(), content=path.read_text(encoding="utf-8")))

    metadata = {
        "operator_id": "spmv.csr.fp32",
        "base_format": args.base_format,
    }
    for key, value in (
        ("build_profile", args.build_profile),
        ("configuration_id", args.configuration_id),
        ("candidate_group", args.candidate_group),
    ):
        if value:
            metadata[key] = value
    kernel = KernelArtifact(
        name=args.plugin_id,
        kind="source",
        source_files=files,
        entry_source=args.entry_source,
        compile_units=args.compile_unit,
        metadata=metadata,
    )
    package = make_source_package(
        kernel,
        REPOSITORY_ROOT / "include" / "qiwu" / "spmv_plugin.cuh",
    )
    if package is None:
        raise SystemExit("source package was not generated")

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    write_source_tree(output / "files", [SourceFile.model_validate(item) for item in package["files"]])
    plugin = {
        **package["plugin"],
        "source_repository": args.source_repository,
        "source_path": args.source_path,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plugin.json").write_text(json.dumps(plugin, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_sha256": package["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
