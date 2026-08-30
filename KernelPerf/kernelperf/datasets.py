from __future__ import annotations

import json
from pathlib import Path

from .models import DatasetSpec, MatrixCase


NO_DATASET_CASE = MatrixCase(
    matrix_id="benchmark-owned-case",
    name="benchmark-owned-case",
    rows=1,
    cols=1,
    nnz=1,
    source_url="",
    local_path=None,
    dataset_id="none",
)


class DatasetRegistry:
    def __init__(self) -> None:
        self._datasets: dict[str, tuple[DatasetSpec, list[MatrixCase]]] = {}

    def register(self, spec: DatasetSpec, matrices: list[MatrixCase]) -> None:
        self._datasets[spec.dataset_id] = (spec, matrices)

    def get(self, dataset_id: str) -> tuple[DatasetSpec, list[MatrixCase]]:
        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return self._datasets[dataset_id]

    def list(self) -> list[dict[str, object]]:
        return [
            {
                **spec.model_dump(),
                "matrix_count": len(matrices),
                "matrices": [m.model_dump() for m in matrices],
            }
            for spec, matrices in self._datasets.values()
        ]


def _load_manifest(path: Path, spec: DatasetSpec) -> list[MatrixCase]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    matrices: list[MatrixCase] = []
    for row in rows:
        item = dict(row)
        item.setdefault("dataset_id", spec.dataset_id)
        if not item.get("local_path"):
            item["local_path"] = spec.path_template.format(
                root=spec.root,
                dataset_id=spec.dataset_id,
                matrix_id=item["matrix_id"],
                name=item["name"],
            )
        matrices.append(MatrixCase.model_validate(item))
    return matrices


def dataset_registry_from_config(path: str | Path) -> DatasetRegistry:
    registry = DatasetRegistry()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"dataset config not found: {config_path}")
    specs = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base_dir = config_path.parent
    for raw_spec in specs:
        spec = DatasetSpec.model_validate(raw_spec)
        matrices: list[MatrixCase] = []
        if spec.manifest:
            manifest = Path(spec.manifest)
            if not manifest.is_absolute():
                manifest = base_dir.parent / manifest
            matrices = _load_manifest(manifest, spec)
        registry.register(spec, matrices)
    return registry
