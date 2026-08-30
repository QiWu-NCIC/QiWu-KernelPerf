from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from spmv_artifact import ARCH_PATTERN, is_elf_relocatable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a KernelPerf SpMV Linux ELF relocatable object."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--arch", required=True, help="CUDA architecture such as sm_80.")
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    if sys.platform == "win32":
        parser.error("server-compatible ELF objects must be built on Linux")
    if not ARCH_PATTERN.fullmatch(args.arch):
        parser.error("--arch must look like sm_80")
    if args.output.suffix != ".o":
        parser.error("--output must end in .o")

    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "-dc",
        "-arch=" + args.arch,
        "-I",
        str(root / "include"),
    ]
    if args.dtype == "fp64":
        command.append("-DQIWU_SPMV_FP64=1")
    command.extend([str(args.source), "-o", str(args.output)])
    subprocess.run(command, check=True)

    content = args.output.read_bytes()
    if not is_elf_relocatable(content):
        raise SystemExit("nvcc output is not a Linux ELF relocatable object")
    print(args.output)


if __name__ == "__main__":
    main()
