"""Data access repositories for models, predictions, explanations, and audit events."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from aletheia.core.errors import NotFoundError
from aletheia.db.models import (
    AuditEventRecord,
    ExplanationRunRecord,
    ModelRecord,
    ModelVersionRecord,
    PredictionLabelRecord,
    PredictionRecord,
)


class ModelRepository:
    """Repository managing model registries and version records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, model_id: uuid.UUID) -> ModelRecord | None:
        stmt = (
            select(ModelRecord)
            .where(ModelRecord.id == model_id)
            .options(joinedload(ModelRecord.versions))
        )
        return self.session.scalars(stmt).unique().first()

    def get_by_name(self, name: str) -> ModelRecord | None:
        stmt = (
            select(ModelRecord)
            .where(ModelRecord.name == name)
            .options(joinedload(ModelRecord.versions))
        )
        return self.session.scalars(stmt).unique().first()

    def list_all(self, limit: int = 100) -> list[ModelRecord]:
        stmt = (
            select(ModelRecord)
            .options(joinedload(ModelRecord.versions))
            .order_by(ModelRecord.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).unique().all())

    def create(
        self,
        name: str,
        task_type: str = "binary_classification",
        description: str | None = None,
    ) -> ModelRecord:
        record = ModelRecord(name=name, task_type=task_type, description=description)
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_version_by_id(self, version_id: uuid.UUID) -> ModelVersionRecord | None:
        stmt = (
            select(ModelVersionRecord)
            .where(ModelVersionRecord.id == version_id)
            .options(joinedload(ModelVersionRecord.model))
        )
        return self.session.scalars(stmt).unique().first()

    def get_active_version(self, model_id: uuid.UUID) -> ModelVersionRecord | None:
        stmt = (
            select(ModelVersionRecord)
            .where(
                ModelVersionRecord.model_id == model_id,
                ModelVersionRecord.status == "active",
            )
            .order_by(ModelVersionRecord.created_at.desc())
        )
        return self.session.scalars(stmt).first()

    def create_version(
        self,
        model_id: uuid.UUID,
        version: str,
        artifact_uri: str,
        artifact_sha256: str,
        adapter_name: str,
        framework: str = "scikit-learn",
        framework_version: str | None = None,
        python_version: str | None = None,
        threshold: float | None = None,
        feature_names: list[str] | None = None,
        source_columns: list[str] | None = None,
        metadata_json: dict[str, Any] | None = None,
        status: str = "active",
    ) -> ModelVersionRecord:
        record = ModelVersionRecord(
            model_id=model_id,
            version=version,
            status=status,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            adapter_name=adapter_name,
            framework=framework,
            framework_version=framework_version,
            python_version=python_version,
            threshold=threshold,
            feature_names=feature_names or [],
            source_columns=source_columns or [],
            metadata_json=metadata_json or {},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record


class PredictionRepository:
    """Repository storing and querying scoring predictions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        model_version_id: uuid.UUID,
        input_features: dict[str, Any],
        probability: float,
        predicted_label: int,
        threshold: float | None = None,
        subject_ref_hash: str | None = None,
    ) -> PredictionRecord:
        record = PredictionRecord(
            model_version_id=model_version_id,
            input_features=input_features,
            probability=probability,
            predicted_label=predicted_label,
            threshold=threshold,
            subject_ref_hash=subject_ref_hash,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_by_id(self, prediction_id: uuid.UUID) -> PredictionRecord | None:
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.id == prediction_id)
            .options(
                joinedload(PredictionRecord.label),
                joinedload(PredictionRecord.explanations),
            )
        )
        return self.session.scalars(stmt).unique().first()

    def list_by_model_version(
        self, model_version_id: uuid.UUID, limit: int = 100
    ) -> list[PredictionRecord]:
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.model_version_id == model_version_id)
            .options(joinedload(PredictionRecord.label))
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).unique().all())


class ExplanationRepository:
    """Repository storing and retrieving explanation runs and cached attributions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_cache_key(self, cache_key: str) -> ExplanationRunRecord | None:
        stmt = select(ExplanationRunRecord).where(ExplanationRunRecord.cache_key == cache_key)
        return self.session.scalars(stmt).first()

    def create(
        self,
        model_version_id: uuid.UUID,
        prediction_id: uuid.UUID | None,
        explainer_type: str,
        model_output: float,
        base_value: float,
        output_space: str,
        contributions: dict[str, float],
        grouped_contributions: dict[str, float],
        fidelity_passed: bool,
        max_abs_error: float,
        tolerance: float,
        duration_ms: int,
        caveats: list[str],
        cache_key: str | None = None,
        scope: str = "local",
        method: str = "shap",
    ) -> ExplanationRunRecord:
        record = ExplanationRunRecord(
            model_version_id=model_version_id,
            prediction_id=prediction_id,
            scope=scope,
            method=method,
            explainer_type=explainer_type,
            cache_key=cache_key,
            model_output=model_output,
            base_value=base_value,
            output_space=output_space,
            contributions=contributions,
            grouped_contributions=grouped_contributions,
            fidelity_passed=fidelity_passed,
            max_abs_error=max_abs_error,
            tolerance=tolerance,
            duration_ms=duration_ms,
            caveats=caveats,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record


class LabelRepository:
    """Repository storing ground truth labels for predictions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record_label(
        self,
        ground_truth: int,
        prediction_id: uuid.UUID | None = None,
        subject_ref_hash: str | None = None,
        label_source: str = "user",
    ) -> PredictionLabelRecord:
        if prediction_id is None and subject_ref_hash is None:
            raise NotFoundError("either prediction_id or subject_ref_hash is required")

        record = PredictionLabelRecord(
            prediction_id=prediction_id,
            subject_ref_hash=subject_ref_hash,
            ground_truth=ground_truth,
            label_source=label_source,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def list_labeled_predictions(
        self, model_version_id: uuid.UUID
    ) -> list[tuple[PredictionRecord, PredictionLabelRecord]]:
        stmt = (
            select(PredictionRecord, PredictionLabelRecord)
            .join(
                PredictionLabelRecord,
                (PredictionRecord.id == PredictionLabelRecord.prediction_id)
                | (
                    (PredictionRecord.subject_ref_hash.is_not(None))
                    & (PredictionRecord.subject_ref_hash == PredictionLabelRecord.subject_ref_hash)
                ),
            )
            .where(PredictionRecord.model_version_id == model_version_id)
        )
        results = self.session.execute(stmt).all()
        return [(row[0], row[1]) for row in results]


class AuditRepository:
    """Repository for auditing sensitive events."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEventRecord:
        record = AuditEventRecord(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor=actor,
            details=details or {},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def list_events(self, limit: int = 100) -> list[AuditEventRecord]:
        stmt = select(AuditEventRecord).order_by(AuditEventRecord.created_at.desc()).limit(limit)
        return list(self.session.scalars(stmt).all())

