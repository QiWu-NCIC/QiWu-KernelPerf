from __future__ import annotations

import json

import pytest

from scripts import run_full_campaign
from scripts.run_full_campaign import _dataset_is_complete


def test_dataset_completeness_is_scoped_to_shard(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    matrices = []
    for index in range(4):
        name = f"matrix-{index}.mtx"
        (data_root / name).write_text("matrix", encoding="utf-8")
        matrices.append({"matrix_id": str(index), "name": name})

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(matrices), encoding="utf-8")
    datasets = tmp_path / "datasets.json"
    datasets.write_text(json.dumps([{
        "dataset_id": "dataset",
        "manifest": str(manifest),
        "root": str(data_root),
        "path_template": "{root}/{name}",
    }]), encoding="utf-8")
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"datasets": str(datasets)}), encoding="utf-8")

    assert _dataset_is_complete(config, "dataset", 0, 2)
    assert _dataset_is_complete(config, "dataset", 1, 2)

    (data_root / "matrix-2.mtx").unlink()

    assert not _dataset_is_complete(config, "dataset", 0, 2)
    assert _dataset_is_complete(config, "dataset", 1, 2)


@pytest.mark.parametrize("dataset_complete", [False, True])
def test_available_spmv_precedes_spmm(tmp_path, monkeypatch, dataset_complete):
    template = tmp_path / "template.json"
    template.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(run_full_campaign.sys, "argv", [
        "run_full_campaign", "--root", str(tmp_path),
        "--config-template", str(template), "--backend", "test",
        "--campaign-prefix", "test", "--shard-index", "0", "--shard-count", "1",
    ])
    monkeypatch.setattr(run_full_campaign, "_dataset_is_complete",
                        lambda *args: dataset_complete)
    monkeypatch.setattr(run_full_campaign, "_run",
                        lambda command, *args: commands.append(command) or 0)

    assert run_full_campaign.main() == (0 if dataset_complete else 75)
    assert len(commands) == 2
    assert commands[0][1].endswith("run_spmv_campaign.py")
    assert commands[1][1].endswith("run_spmm_campaign.py")
    markers = list((tmp_path / "campaign-run" / "markers").glob("*.complete"))
    assert len(markers) == (2 if dataset_complete else 0)


@pytest.mark.parametrize("interrupted_stage", [0, 1])
def test_interrupted_campaign_does_not_mark_complete(tmp_path, monkeypatch, interrupted_stage):
    template = tmp_path / "template.json"
    template.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(run_full_campaign.sys, "argv", [
        "run_full_campaign", "--root", str(tmp_path),
        "--config-template", str(template), "--backend", "test",
        "--campaign-prefix", "test", "--shard-index", "0", "--shard-count", "1",
    ])
    monkeypatch.setattr(run_full_campaign, "_dataset_is_complete", lambda *args: True)

    def run(command, *args):
        commands.append(command)
        return 75 if len(commands) - 1 == interrupted_stage else 0

    monkeypatch.setattr(run_full_campaign, "_run", run)
    assert run_full_campaign.main() == 75
    assert len(commands) == interrupted_stage + 1
    assert not (tmp_path / "campaign-run" / "markers" / "spmm-shard-0.complete").exists()


def test_selection_file_is_passed_to_both_operator_campaigns(tmp_path, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text("{}", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(run_full_campaign.sys, "argv", [
        "run_full_campaign", "--root", str(tmp_path),
        "--config-template", str(template), "--selection-file", str(selection),
        "--backend", "test", "--campaign-prefix", "test",
        "--shard-index", "0", "--shard-count", "1",
    ])
    monkeypatch.setattr(run_full_campaign, "_dataset_is_complete", lambda *args: True)
    monkeypatch.setattr(run_full_campaign, "_run",
                        lambda command, *args: commands.append(command) or 0)

    assert run_full_campaign.main() == 0
    assert all(command[command.index("--selection-file") + 1] == str(selection.resolve())
               for command in commands)
