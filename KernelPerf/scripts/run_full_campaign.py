from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _replace(value, shard: int) -> object:
    if isinstance(value, str):
        return value.replace("__SHARD__", str(shard))
    if isinstance(value, list):
        return [_replace(item, shard) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, shard) for key, item in value.items()}
    return value


def _config_for_shard(template: Path, output: Path, shard: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(template.read_text(encoding="utf-8"))
    output.write_text(json.dumps(_replace(data, shard), indent=2) + "\n", encoding="utf-8")
    return output


def _run(command: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return completed.returncode


def _dataset_is_complete(config: Path, dataset_id: str, shard_index: int = 0,
                         shard_count: int = 1) -> bool:
    service = json.loads(config.read_text(encoding="utf-8"))
    datasets = json.loads(Path(service["datasets"]).read_text(encoding="utf-8"))
    spec = next(item for item in datasets if item["dataset_id"] == dataset_id)
    manifest = json.loads(Path(spec["manifest"]).read_text(encoding="utf-8-sig"))
    if shard_count > 1:
        manifest = manifest[shard_index::shard_count]
    template = str(spec.get("path_template", "{root}/{name}"))
    return all(Path(template.format(root=spec["root"], dataset_id=dataset_id,
                                    matrix_id=item["matrix_id"], name=item["name"],
                                    group=item.get("group", ""))).is_file()
               for item in manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a checkpointed full SpMV then SpMM campaign shard")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--config-template", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset-id", default="suitesparse_all")
    parser.add_argument("--campaign-prefix", required=True)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--barrier-timeout-seconds", type=int, default=0)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--adopt-fingerprint", action="store_true",
                        help="adopt the current source and dataset fingerprint for this named campaign")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within shard-count")

    root = args.root.resolve()
    default_selection = Path(__file__).resolve().parents[1] / "config/campaigns/suitesparse_all.json"
    selection_file = (args.selection_file or default_selection).resolve()
    if not selection_file.is_file():
        raise ValueError(f"missing full campaign selection file: {selection_file}")
    run_dir = (args.state_dir or (root / "campaign-run")).resolve()
    marker_dir = run_dir / "markers"
    config = _config_for_shard(
        args.config_template,
        run_dir / f"service-{args.shard_index}.json",
        args.shard_index,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    python = str(Path(args.python).expanduser())
    spmv_log = run_dir / "logs" / f"spmv-{args.shard_index}.log"
    spmm_log = run_dir / "logs" / f"spmm-{args.shard_index}.log"
    common = ["--config", str(config), "--backend", args.backend,
              "--dataset-id", args.dataset_id, "--shard-index", str(args.shard_index),
              "--shard-count", str(args.shard_count), "--selection-file", str(selection_file)]
    spmv_command = [python, str(root / "scripts/run_spmv_campaign.py"), *common,
                    "--campaign-id", f"{args.campaign_prefix}-spmv"]
    if args.retry_failures:
        spmv_command.append("--retry-failures")
    if args.adopt_fingerprint:
        spmv_command.append("--adopt-fingerprint")
    code = _run(spmv_command, spmv_log, env)
    if code != 0:
        return code
    dataset_complete = _dataset_is_complete(config, args.dataset_id, args.shard_index,
                                            args.shard_count)
    marker_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_complete:
        print(json.dumps({
            "status": "running_spmm_on_available_matrices",
            "shard_index": args.shard_index,
        }), flush=True)
    else:
        (marker_dir / f"spmv-ready-shard-{args.shard_index}.complete").touch()

    spmm_command = [python, str(root / "scripts/run_spmm_campaign.py"), *common,
                    "--campaign-id", f"{args.campaign_prefix}-spmm"]
    if args.retry_failures:
        spmm_command.append("--retry-failures")
    if args.adopt_fingerprint:
        spmm_command.append("--adopt-fingerprint")
    code = _run(spmm_command, spmm_log, env)
    if code != 0:
        return code
    if dataset_complete and _dataset_is_complete(config, args.dataset_id, args.shard_index,
                            args.shard_count):
        (marker_dir / f"spmm-shard-{args.shard_index}.complete").touch()
        return 0
    return 75


if __name__ == "__main__":
    raise SystemExit(main())
