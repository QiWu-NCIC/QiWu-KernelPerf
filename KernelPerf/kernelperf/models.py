from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class WorkerStatus(str, Enum):
    online = "online"
    offline = "offline"
    busy = "busy"
    draining = "draining"


class CaseStatus(str, Enum):
    passed = "pass"
    error = "error"
    failed = "fail"


class MatrixCase(BaseModel):
    matrix_id: str
    name: str
    rows: int
    cols: int
    nnz: int
    source_url: str = ""
    local_path: str | None = None
    dataset_id: str | None = None


class DatasetSpec(BaseModel):
    dataset_id: str
    name: str
    kind: str = "manifest"
    root: str = "."
    manifest: str | None = None
    path_template: str = "{root}/{name}"
    description: str | None = None


class OperatorSpec(BaseModel):
    op_id: str
    name: str
    suite: str
    dtype: str = "fp32"
    problem_kind: str = "generic"
    flops_formula: str = "driver-defined"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceFile(BaseModel):
    """One UTF-8 text file belonging to a submitted source tree."""

    path: str
    content: str


class KernelArtifact(BaseModel):
    name: str
    kind: Literal["source", "object"] = "source"
    language: str = ""
    entrypoint: str = ""
    source: str | None = None
    source_files: list[SourceFile] = Field(default_factory=list)
    entry_source: str | None = None
    compile_units: list[str] = Field(default_factory=list)
    object_base64: str | None = None
    path: str | None = None
    compile_options: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobSubmitRequest(BaseModel):
    generator_id: str | None = None
    submitter: str | None = None
    backends: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Backend IDs or backend kinds.",
    )
    suites: list[str] = Field(
        default_factory=list,
        description="Benchmark IDs. An omitted value uses the configured default benchmark.",
    )
    dataset_id: str | None = Field(default=None, description="Optional dataset ID for matrix-backed benchmarks.")
    operator_ids: list[str] | None = Field(default=None, description="Optional operator filter.")
    matrix_ids: list[str] | None = Field(default=None, description="Optional dataset case filter.")
    kernels: list[KernelArtifact] = Field(..., min_length=1)
    priority: int = 100
    tags: dict[str, str] = Field(default_factory=dict)


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    generator_id: str
    submitter: str | None = None
    backends: list[str]
    suites: list[str]
    dataset_id: str | None = None
    operator_ids: list[str] | None = None
    matrix_ids: list[str] | None = None
    kernels: list[KernelArtifact]
    priority: int = 100
    tags: dict[str, str] = Field(default_factory=dict)
    status: JobStatus = JobStatus.queued
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    assigned_worker: str | None = None
    error: str | None = None


class WorkerNode(BaseModel):
    worker_id: str
    backend_id: str
    endpoint: str = "local"
    labels: dict[str, str] = Field(default_factory=dict)
    status: WorkerStatus = WorkerStatus.online
    last_heartbeat: str = Field(default_factory=utc_now_iso)
    current_job_id: str | None = None
    log_tail: list[str] = Field(default_factory=list)


class BackendInfo(BaseModel):
    backend_id: str
    kind: str
    name: str
    vendor: str
    device: str
    peak_gflops: float
    peak_gflops_by_dtype: dict[str, float] = Field(default_factory=dict)
    memory_bandwidth_gbs: float
    supported_languages: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    generator_id: str
    backend_id: str
    backend_kind: str
    suite: str
    operator_id: str
    operator_name: str
    matrix_id: str
    matrix_name: str
    rows: int
    cols: int
    nnz: int
    kernel_name: str
    status: CaseStatus = CaseStatus.passed
    preprocess_ms: float = 0.0
    runtime_ms: float
    gflops: float
    arithmetic_intensity: float
    timestamp: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
