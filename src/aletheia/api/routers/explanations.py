"""Local explainability API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from aletheia.api.routers.predictions import _resolve_version_id
from aletheia.db.connection import get_db
from aletheia.schemas.explanations import ExplanationRequest, ExplanationResponse
from aletheia.services.explanation_service import ExplanationService

router = APIRouter(prefix="/api/v1/models", tags=["explanations"])


@router.post(
    "/{model_id}/explain",
    response_model=ExplanationResponse,
    summary="Generate local SHAP explanation with additivity verification",
)
def explain_prediction(
    model_id: uuid.UUID,
    data: ExplanationRequest,
    db: Session = Depends(get_db),
) -> ExplanationResponse:
    version_id = _resolve_version_id(model_id, db)
    service = ExplanationService(db)
    return service.explain(version_id, data)

