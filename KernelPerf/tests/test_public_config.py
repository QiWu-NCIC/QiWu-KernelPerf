from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config"


def test_tracked_worker_and_dataset_profiles_are_portable():
    for path in (CONFIG / "service.json", CONFIG / "workers.json", CONFIG / "datasets.json"):
        text = path.read_text(encoding="utf-8")
        assert "identity_file" not in text
        assert "ssh://" not in text
        assert "/root/" not in text
        assert "/home/" not in text
        assert "/public/" not in text
        assert "10." not in text


def test_public_service_references_tracked_files():
    service = json.loads((CONFIG / "service.json").read_text(encoding="utf-8"))
    for key in ("workers", "benchmarks", "datasets"):
        assert (ROOT / service[key]).is_file()
