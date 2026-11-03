"""Pydantic schemas for ground truth labels and model performance metrics."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class LabelCreate(BaseModel):
    ground_truth: int = Field(..., ge=0, le=1)
    prediction_id: uuid.UUID | None = None
    subject_id: str | None = None
    label_source: str = Field(default="user", max_length=64)


class LabelResponse(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID | None
    subject_ref_hash: str | None
    ground_truth: int
    label_source: str
    observed_at: datetime
    created_at: datetime


class CalibrationMetricsResponse(BaseModel):
    model_version_id: uuid.UUID
    total_labeled: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    brier_score: float
    mean_predicted_prob: float
    actual_prevalence: float
    calibration_gap: float

