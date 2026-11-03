"""Scoring and inference API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from aletheia.core.errors import NotFoundError
from aletheia.db.connection import get_db
from aletheia.db.repositories import ModelRepository
from aletheia.schemas.predictions import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from aletheia.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/models", tags=["predictions"])


def _resolve_version_id(target_id: uuid.UUID, db: Session) -> uuid.UUID:
    repo = ModelRepository(db)
    # Check if target_id is directly a version_id
    version = repo.get_version_by_id(target_id)
    if version is not None:
        return version.id
    # Check if target_id is a model_id with an active version
    model = repo.get_by_id(target_id)
    if model is not None:
        active = repo.get_active_version(model.id)
        if active is not None:
            return active.id
        raise NotFoundError(f"model '{target_id}' has no active version")
    raise NotFoundError(f"model or version '{target_id}' not found")


@router.post(
    "/{model_id}/predict",
    response_model=PredictionResponse,
    summary="Score a single observation",
)
def predict_single(
    model_id: uuid.UUID,
    data: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    version_id = _resolve_version_id(model_id, db)
    service = PredictionService(db)
    return service.predict_single(version_id, data)


@router.post(
    "/{model_id}/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch score multiple observations",
)
def predict_batch(
    model_id: uuid.UUID,
    data: BatchPredictionRequest,
    db: Session = Depends(get_db),
) -> BatchPredictionResponse:
    version_id = _resolve_version_id(model_id, db)
    service = PredictionService(db)
    return service.predict_batch(version_id, data)

