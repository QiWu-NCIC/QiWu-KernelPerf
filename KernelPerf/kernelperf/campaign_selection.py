from __future__ import annotations

import json
from pathlib import Path


def load_campaign_selection(path: Path, backend_id: str, benchmark_id: str):
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported campaign selection schema: {path}")
    if data.get("dataset_id") != "suitesparse_all":
        raise ValueError(f"campaign selection must target suitesparse_all: {path}")
    backend = data.get("backends", {}).get(backend_id)
    if backend is None:
        raise ValueError(f"no campaign selection for backend {backend_id}: {path}")
    entries = backend.get(benchmark_id)
    if not entries:
        raise ValueError(f"no {benchmark_id} campaign selection for backend {backend_id}: {path}")
    base_dir = path.parents[2] if path.parent.name == "campaigns" else path.parent
    result = []
    seen = set()
    for entry in entries:
        submission = entry.get("submission")
        if not submission:
            raise ValueError(f"campaign selection entry has no submission: {path}")
        configuration_ids = tuple(entry.get("configuration_ids", []))
        key = (submission, configuration_ids)
        if key in seen:
            raise ValueError(f"duplicate campaign selection entry: {submission}")
        seen.add(key)
        submission_path = Path(submission)
        if not submission_path.is_absolute():
            submission_path = base_dir / submission_path
        result.append((submission_path, configuration_ids))
    return result
