"""Prediction and scoring service."""

from __future__ import annotations

import hashlib
import uuid

import pandas as pd
from sqlalchemy.orm import Session

from aletheia.core.errors import ValidationFailed
from aletheia.db.repositories import PredictionRepository
from aletheia.schemas.predictions import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from aletheia.services.model_service import ModelService


def _hash_subject(subject_id: str | None) -> str | None:
    if not subject_id:
        return None
    return hashlib.sha256(subject_id.strip().encode("utf-8")).hexdigest()


class PredictionService:
    """Service executing and recording model inference."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = PredictionRepository(session)
        self.model_service = ModelService(session)

    def predict_single(
        self, model_version_id: uuid.UUID, request: PredictionRequest
    ) -> PredictionResponse:
        version_rec, model = self.model_service.get_loaded_model(model_version_id)
        df = pd.DataFrame([request.features])

        try:
            probabilities = model.predict_proba(df)
            probability = float(probabilities[0])
            labels = model.predict_label(df)
            predicted_label = int(labels[0])
        except Exception as exc:
            raise ValidationFailed(f"model prediction failed: {exc}") from exc

        subject_hash = _hash_subject(request.subject_id)
        record = self.repo.create(
            model_version_id=version_rec.id,
            input_features=request.features,
            probability=probability,
            predicted_label=predicted_label,
            threshold=model.metadata.threshold,
            subject_ref_hash=subject_hash,
        )

        return PredictionResponse(
            id=record.id,
            model_version_id=record.model_version_id,
            subject_ref_hash=record.subject_ref_hash,
            probability=record.probability,
            predicted_label=record.predicted_label,
            threshold=record.threshold,
            created_at=record.created_at,
        )

    def predict_batch(
        self, model_version_id: uuid.UUID, request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        if not request.items:
            raise ValidationFailed("batch must contain at least one item")
        version_rec, model = self.model_service.get_loaded_model(model_version_id)

        df = pd.DataFrame([item.features for item in request.items])
        try:
            probabilities = model.predict_proba(df)
            labels = model.predict_label(df)
        except Exception as exc:
            raise ValidationFailed(f"batch scoring failed: {exc}") from exc

        responses: list[PredictionResponse] = []
        for i, item in enumerate(request.items):
            subject_hash = _hash_subject(item.subject_id)
            rec = self.repo.create(
                model_version_id=version_rec.id,
                input_features=item.features,
                probability=float(probabilities[i]),
                predicted_label=int(labels[i]),
                threshold=model.metadata.threshold,
                subject_ref_hash=subject_hash,
            )
            responses.append(
                PredictionResponse(
                    id=rec.id,
                    model_version_id=rec.model_version_id,
                    subject_ref_hash=rec.subject_ref_hash,
                    probability=rec.probability,
                    predicted_label=rec.predicted_label,
                    threshold=rec.threshold,
                    created_at=rec.created_at,
                )
            )

        return BatchPredictionResponse(predictions=responses, count=len(responses))
