from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from kernelperf.datasets import dataset_registry_from_config


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "config/datasets/suitesparse_sample_100.json"
SELECTION = ROOT / "config/datasets/suitesparse_sample_100.selection.json"


def test_performance_manifest_is_real_stratified_and_unique():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    assert len(entries) == 100
    assert len({entry["matrix_id"] for entry in entries}) == 100
    assert len({entry["name"] for entry in entries}) == 100
    assert all(entry["real"] is True for entry in entries)
    assert all(100 <= entry["nnz"] <= 100_000_000 for entry in entries)
    assert all(math.isclose(entry["avg_row_length"], entry["nnz"] / entry["rows"]) for entry in entries)

    bins = Counter(min(7, max(2, int(math.log10(entry["nnz"])))) for entry in entries)
    assert bins == {2: 10, 3: 14, 4: 18, 5: 20, 6: 20, 7: 18}

    metadata = json.loads(SELECTION.read_text(encoding="utf-8-sig"))
    assert metadata["dataset_id"] == "suitesparse_sample_100"
    assert metadata["selection"]["real_only"] is True
    assert metadata["selection"]["nnz_min"] == 100
    assert metadata["selection"]["nnz_max"] == 100_000_000


def test_performance_manifest_resolves_download_layout():
    spec, cases = dataset_registry_from_config(ROOT / "config/datasets.json").get("suitesparse_sample_100")
    assert len(cases) == 100
    assert spec.path_template == "{root}/{name}"
    assert cases[0].local_path == f"{spec.root}/{cases[0].name}"
    assert cases[0].source_url.endswith(f"/{cases[0].matrix_id}.tar.gz")
