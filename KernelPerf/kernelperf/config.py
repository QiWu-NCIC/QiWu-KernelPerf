from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "service.json"


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    database: str
    contest_database: str = "data/contest.sqlite"
    workers: str
    benchmarks: str
    datasets: str
    web: str = "web"
    result_exports: str = "data/result_exports"
    source_submissions: str = "data/source"
    leaderboard: dict[str, str] = Field(default_factory=dict)

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_service_config(path: str | Path | None = None) -> ServiceConfig:
    selected = Path(
        path or os.environ.get("KERNELPERF_CONFIG", str(DEFAULT_CONFIG_PATH))
    ).expanduser()
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    if not selected.is_file():
        raise FileNotFoundError(f"service config not found: {selected}")
    return ServiceConfig.model_validate(json.loads(selected.read_text()))
