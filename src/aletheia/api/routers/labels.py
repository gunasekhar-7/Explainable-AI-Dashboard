"""Late-arriving labels and performance calibration endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from aletheia.api.routers.predictions import _resolve_version_id
from aletheia.db.connection import get_db
from aletheia.schemas.labels import (
    CalibrationMetricsResponse,
    LabelCreate,
    LabelResponse,
)
from aletheia.services.audit_service import AuditService
from aletheia.services.label_service import LabelService

router = APIRouter(prefix="/api/v1/models", tags=["labels"])


@router.post(
    "/{model_id}/labels",
    response_model=LabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record late-arriving ground truth outcome",
)
def record_label(
    model_id: uuid.UUID,
    data: LabelCreate,
    db: Session = Depends(get_db),
) -> LabelResponse:
    version_id = _resolve_version_id(model_id, db)
    service = LabelService(db)
    response = service.record_label(version_id, data)
    AuditService(db).record_event(
        action="label.recorded",
        resource_type="prediction_label",
        resource_id=str(response.id),
        details={
            "model_version_id": str(version_id),
            "ground_truth": data.ground_truth,
            "prediction_id": str(data.prediction_id) if data.prediction_id else None,
        },
    )
    return response


@router.get(
    "/{model_id}/metrics",
    response_model=CalibrationMetricsResponse,
    summary="Retrieve outcome calibration and performance metrics",
)
def get_metrics(
    model_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> CalibrationMetricsResponse:
    version_id = _resolve_version_id(model_id, db)
    service = LabelService(db)
    return service.compute_calibration_metrics(version_id)

