"""Late-arriving labels and outcome calibration service."""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
from sqlalchemy.orm import Session

from aletheia.core.errors import NotFoundError
from aletheia.db.repositories import LabelRepository, ModelRepository
from aletheia.schemas.labels import (
    CalibrationMetricsResponse,
    LabelCreate,
    LabelResponse,
)


class LabelService:
    """Service ingesting ground truth outcomes and computing model calibration."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = LabelRepository(session)
        self.model_repo = ModelRepository(session)

    def record_label(self, model_version_id: uuid.UUID, data: LabelCreate) -> LabelResponse:
        version = self.model_repo.get_version_by_id(model_version_id)
        if version is None:
            raise NotFoundError(f"model version '{model_version_id}' not found")

        subject_hash = (
            hashlib.sha256(data.subject_id.strip().encode("utf-8")).hexdigest()
            if data.subject_id
            else None
        )

        record = self.repo.record_label(
            ground_truth=data.ground_truth,
            prediction_id=data.prediction_id,
            subject_ref_hash=subject_hash,
            label_source=data.label_source,
        )

        return LabelResponse(
            id=record.id,
            prediction_id=record.prediction_id,
            subject_ref_hash=record.subject_ref_hash,
            ground_truth=record.ground_truth,
            label_source=record.label_source,
            observed_at=record.observed_at,
            created_at=record.created_at,
        )

    def compute_calibration_metrics(
        self, model_version_id: uuid.UUID
    ) -> CalibrationMetricsResponse:
        version = self.model_repo.get_version_by_id(model_version_id)
        if version is None:
            raise NotFoundError(f"model version '{model_version_id}' not found")

        pairs = self.repo.list_labeled_predictions(model_version_id)
        if not pairs:
            return CalibrationMetricsResponse(
                model_version_id=model_version_id,
                total_labeled=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                brier_score=0.0,
                mean_predicted_prob=0.0,
                actual_prevalence=0.0,
                calibration_gap=0.0,
            )

        y_true = np.array([label.ground_truth for _, label in pairs], dtype=int)
        y_prob = np.array([pred.probability for pred, _ in pairs], dtype=float)
        y_pred = np.array([pred.predicted_label for pred, _ in pairs], dtype=int)

        total = len(pairs)
        accuracy = float(np.mean(y_true == y_pred))

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        brier = float(np.mean((y_prob - y_true) ** 2))
        mean_prob = float(np.mean(y_prob))
        prevalence = float(np.mean(y_true))
        gap = float(mean_prob - prevalence)

        return CalibrationMetricsResponse(
            model_version_id=model_version_id,
            total_labeled=total,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            brier_score=round(brier, 4),
            mean_predicted_prob=round(mean_prob, 4),
            actual_prevalence=round(prevalence, 4),
            calibration_gap=round(gap, 4),
        )

