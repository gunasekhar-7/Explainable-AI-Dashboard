"""Model registry API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from aletheia.db.connection import get_db
from aletheia.schemas.models import (
    ModelCreate,
    ModelDetailResponse,
    ModelResponse,
    ModelVersionCreate,
    ModelVersionResponse,
)
from aletheia.services.audit_service import AuditService
from aletheia.services.model_service import ModelService

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=list[ModelResponse], summary="List registered models")
def list_models(db: Session = Depends(get_db)) -> list[ModelResponse]:
    service = ModelService(db)
    return service.list_models()


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model entity",
)
def create_model(data: ModelCreate, db: Session = Depends(get_db)) -> ModelResponse:
    service = ModelService(db)
    response = service.create_model(data)
    AuditService(db).record_event(
        action="model.created",
        resource_type="model",
        resource_id=str(response.id),
        details={"name": data.name, "task_type": data.task_type},
    )
    return response


@router.get("/{model_id}", response_model=ModelDetailResponse, summary="Get model details and versions")
def get_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ModelDetailResponse:
    service = ModelService(db)
    return service.get_model(model_id)


@router.post(
    "/{model_id}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new version artifact for a model",
)
def register_version(
    model_id: uuid.UUID,
    data: ModelVersionCreate,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    service = ModelService(db)
    response = service.register_version(model_id, data)
    AuditService(db).record_event(
        action="model_version.registered",
        resource_type="model_version",
        resource_id=str(response.id),
        details={
            "model_id": str(model_id),
            "version": data.version,
            "artifact_sha256": response.artifact_sha256,
        },
    )
    return response

