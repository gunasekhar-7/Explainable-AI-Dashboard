"""Pydantic schemas for explainability and additivity verification."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AdditivityReportSchema(BaseModel):
    max_abs_error: float
    tolerance: float
    rows_checked: int
    passed: bool


class ExplanationRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Observation features to explain")
    prediction_id: uuid.UUID | None = Field(default=None, description="Optional existing prediction to link")
    background_samples: list[dict[str, Any]] | None = Field(
        default=None, description="Optional custom background dataset for SHAP reference"
    )


class ExplanationResponse(BaseModel):
    id: uuid.UUID
    model_version_id: uuid.UUID
    prediction_id: uuid.UUID | None
    explainer_type: str
    output_space: str
    model_output: float
    base_value: float
    contributions: dict[str, float]
    grouped_contributions: dict[str, float]
    additivity: AdditivityReportSchema
    duration_ms: int
    caveats: list[str]
    created_at: datetime

