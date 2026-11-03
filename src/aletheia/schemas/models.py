"""Pydantic schemas for model registration and versioning."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    task_type: str = Field(default="binary_classification", max_length=64)
    description: str | None = Field(default=None, max_length=1000)


class ModelVersionCreate(BaseModel):
    version: str = Field(..., min_length=1, max_length=64)
    artifact_uri: str = Field(..., min_length=1, max_length=512)
    expected_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    framework: str = Field(default="scikit-learn", max_length=64)
    status: str = Field(default="active", max_length=32)


class ModelVersionResponse(BaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    version: str
    status: str
    artifact_uri: str
    artifact_sha256: str
    adapter_name: str
    framework: str
    framework_version: str | None
    python_version: str | None
    threshold: float | None
    feature_names: list[str]
    source_columns: list[str]
    metadata: dict[str, Any]
    created_at: datetime


class ModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    task_type: str
    description: str | None
    created_at: datetime
    versions_count: int = 0


class ModelDetailResponse(BaseModel):
    id: uuid.UUID
    name: str
    task_type: str
    description: str | None
    created_at: datetime
    versions: list[ModelVersionResponse] = []

