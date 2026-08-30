from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path


BASE_FORMAT_CHOICES = (
    "csr",
    "coo",
    "ell",
    "sell",
    "hyb",
    "bsr",
    "dia",
    "auto",
    "unmarked",
)
ARCH_PATTERN = re.compile(r"^sm_[0-9]{2,3}$")
MAX_SOURCE_FILES = 512
MAX_SOURCE_BYTES = 8 * 1024 * 1024
IGNORED_DIRECTORIES = {".git", "__pycache__", "build", "dist", "node_modules"}


def add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-format",
        required=True,
        choices=BASE_FORMAT_CHOICES,
        help="Base sparse storage family; use auto for auto-tuned or unmarked when unspecified.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Leaderboard method name; defaults to the source/object file stem.",
    )
    parser.add_argument(
        "--build-profile",
        default=None,
        help="Allow-listed worker build profile, for example ghost-cuda.",
    )
    parser.add_argument(
        "--configuration-id",
        default=None,
        help="Stable identifier for one candidate configuration in a sweep.",
    )
    parser.add_argument(
        "--candidate-group",
        default=None,
        help="Group whose configurations are compared for per-matrix BEST.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--source",
        default=None,
        help="Lifecycle CUDA source path, or '-' to read from stdin.",
    )
    group.add_argument(
        "--source-dir",
        default=None,
        help="Source tree containing an adapter and unchanged upstream implementation files.",
    )
    group.add_argument(
        "--object",
        default=None,
        help="Linux ELF relocatable object built against the SpMV lifecycle interface.",
    )
    parser.add_argument(
        "--cuda-arch",
        default=None,
        help="Required for object submissions, for example sm_80.",
    )
    parser.add_argument(
        "--entry-source",
        default=None,
        help="Adapter .cu path relative to --source-dir; it implements the lifecycle interface.",
    )
    parser.add_argument(
        "--compile-unit",
        action="append",
        default=[],
        help="Additional .cu/.cpp path relative to --source-dir; may be repeated.",
    )


def artifact_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    default_source: str = "examples/spmv_template.cu",
    entry_source_override: str | None = None,
    method_name_override: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_path = args.object or args.source_dir or args.source or default_source
    default_name = "stdin-candidate" if selected_path == "-" else Path(selected_path).stem
    artifact: dict[str, object] = {
        "name": method_name_override or args.method_name or default_name,
        "kind": "object" if args.object else "source",
        "metadata": {
            "operator_id": args.operator,
            "base_format": args.base_format,
        },
    }
    if args.build_profile:
        artifact["metadata"]["build_profile"] = args.build_profile
    if args.configuration_id:
        artifact["metadata"]["configuration_id"] = args.configuration_id
    if args.candidate_group:
        artifact["metadata"]["candidate_group"] = args.candidate_group
    if metadata_overrides:
        artifact["metadata"].update(metadata_overrides)

    if args.object:
        if args.entry_source or args.compile_unit:
            parser.error("--entry-source and --compile-unit are only valid with --source-dir")
        if not args.cuda_arch or not ARCH_PATTERN.fullmatch(args.cuda_arch):
            parser.error("--object requires --cuda-arch such as sm_80")
        content = Path(args.object).read_bytes()
        if not is_elf_relocatable(content):
            parser.error("--object must be a Linux ELF relocatable object")
        artifact["object_base64"] = base64.b64encode(content).decode()
        artifact["metadata"]["cuda_arch"] = args.cuda_arch
    else:
        if args.cuda_arch:
            parser.error("--cuda-arch is only valid with --object")
        if args.source_dir:
            entry_argument = entry_source_override or args.entry_source
            if not entry_argument:
                parser.error("--source-dir requires --entry-source")
            files = read_source_tree(Path(args.source_dir), parser)
            paths = {item["path"] for item in files}
            entry_source = normalized_relative_path(entry_argument, parser)
            compile_units = [
                normalized_relative_path(value, parser) for value in args.compile_unit
            ]
            missing = sorted(({entry_source, *compile_units}) - paths)
            if missing:
                parser.error("source tree does not contain: " + ", ".join(missing))
            artifact["source_files"] = files
            artifact["entry_source"] = entry_source
            artifact["compile_units"] = compile_units
        else:
            if args.entry_source or args.compile_unit:
                parser.error("--entry-source and --compile-unit require --source-dir")
            artifact["source"] = (
                sys.stdin.read()
                if selected_path == "-"
                else Path(selected_path).read_text(encoding="utf-8")
            )
    return artifact


def normalized_relative_path(value: str, parser: argparse.ArgumentParser) -> str:
    path = Path(value)
    if path.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        parser.error(f"source paths must be safe POSIX relative paths: {value!r}")
    return value


def read_source_tree(
    root: Path,
    parser: argparse.ArgumentParser,
) -> list[dict[str, str]]:
    if not root.is_dir():
        parser.error(f"source directory does not exist: {root}")
    files: list[dict[str, str]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            parser.error(f"source trees cannot contain symlinks: {relative}")
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parser.error(f"source trees may contain only UTF-8 text files: {relative}")
        relative_path = relative.as_posix()
        total_bytes += len(content.encode())
        files.append({"path": relative_path, "content": content})
        if len(files) > MAX_SOURCE_FILES:
            parser.error(f"source tree exceeds {MAX_SOURCE_FILES} files")
        if total_bytes > MAX_SOURCE_BYTES:
            parser.error(f"source tree exceeds {MAX_SOURCE_BYTES} bytes")
    if not files:
        parser.error("source directory contains no UTF-8 text files")
    return files


def is_elf_relocatable(content: bytes) -> bool:
    if len(content) < 20 or not content.startswith(b"ELF"):
        return False
    byte_order = "little" if content[5] == 1 else "big" if content[5] == 2 else ""
    return bool(byte_order) and int.from_bytes(content[16:18], byte_order) == 1
