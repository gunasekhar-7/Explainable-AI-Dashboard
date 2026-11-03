"""Pydantic schemas for single and batch model scoring."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw input feature key-value pairs")
    subject_id: str | None = Field(default=None, description="Optional subject identifier for auditing/hash")


class PredictionResponse(BaseModel):
    id: uuid.UUID
    model_version_id: uuid.UUID
    subject_ref_hash: str | None
    probability: float
    predicted_label: int
    threshold: float | None
    created_at: datetime


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int

