from __future__ import annotations

import json

from kernelperf.datasets import dataset_registry_from_config


def test_manifest_dataset_loading_is_name_agnostic(tmp_path):
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "matrix_id": "group/case",
                    "name": "case",
                    "rows": 2,
                    "cols": 3,
                    "nnz": 4,
                }
            ]
        )
    )
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "arbitrary-corpus",
                    "name": "Arbitrary corpus",
                    "kind": "matrix-manifest",
                    "root": "/datasets/root",
                    "manifest": str(manifest),
                    "path_template": "{root}/{matrix_id}",
                }
            ]
        )
    )

    spec, cases = dataset_registry_from_config(config).get("arbitrary-corpus")

    assert spec.kind == "matrix-manifest"
    assert cases[0].local_path == "/datasets/root/group/case"
    assert cases[0].dataset_id == "arbitrary-corpus"
