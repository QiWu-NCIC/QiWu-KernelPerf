from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernelperf.submissions import load_submission_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reviewed QiWu submissions")
    parser.add_argument("root", type=Path, default=Path("submissions"), nargs="?")
    args = parser.parse_args()
    root = args.root
    directories = sorted(path.parent for path in root.rglob("submission.json"))
    if not directories:
        raise SystemExit(f"no submissions found under {root}")
    for directory in directories:
        manifest = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
        artifacts = load_submission_artifacts(directory)
        print(f"{directory}: {len(artifacts)} artifact(s) kind={manifest.get('kind', 'source')}")


if __name__ == "__main__":
    main()
