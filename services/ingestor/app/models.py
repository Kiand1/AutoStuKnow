from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class SyncStatus(StrEnum):
    pending = "pending"
    not_configured = "not_configured"
    synced = "synced"
    failed = "failed"


class JobRequest(BaseModel):
    url: str = Field(min_length=10, max_length=2_048)
    language: str = Field(default="auto", max_length=32)
    workspace_slug: str | None = Field(default=None, max_length=128)
    force: bool = False


class BatchJobRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=50)
    language: str = Field(default="auto", max_length=32)
    workspace_slug: str | None = Field(default=None, max_length=128)
    force: bool = False


class WebLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class WebPasswordChangeRequest(BaseModel):
    current_password: str | None = Field(default=None, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


class JobRecord(BaseModel):
    id: str
    url: str
    canonical_url: str
    language: str = "auto"
    workspace_slug: str | None = None
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    title: str | None = None
    source_id: str | None = None
    uploader: str | None = None
    duration_seconds: int | None = None
    transcript_source: str | None = None
    document_path: str | None = None
    sync_status: SyncStatus = SyncStatus.pending
    anythingllm_document_location: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class JobSubmission(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    deduplicated: bool = False
    status_path: str


class Summary(BaseModel):
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    id: str
    title: str
    webpage_url: str
    uploader: str = "Unknown"
    channel_url: str | None = None
    upload_date: str | None = None
    duration: int | None = None
    description: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "whisper"
