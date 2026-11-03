"""SQLAlchemy 2.0 declarative database models.

Follows the schema defined in docs/04-database-design.md with dual PostgreSQL / SQLite compatibility.
All foreign keys have explicit cascade behaviors, timestamps are UTC with timezone, and
primary keys use UUIDs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base declarative class for all Aletheia records."""


class ModelRecord(Base):
    """Top-level registered machine learning model definition."""

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), default="binary_classification", nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    versions: Mapped[list[ModelVersionRecord]] = relationship(
        "ModelVersionRecord",
        back_populates="model",
        cascade="all, delete-orphan",
        order_by="ModelVersionRecord.created_at.desc()",
    )


class ModelVersionRecord(Base):
    """An immutable versioned artifact iteration for a model."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    adapter_name: Mapped[str] = mapped_column(String(64), nullable=False)
    framework: Mapped[str] = mapped_column(String(64), default="scikit-learn", nullable=False)
    framework_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    python_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_names: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list, nullable=False
    )
    source_columns: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list, nullable=False
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    model: Mapped[ModelRecord] = relationship("ModelRecord", back_populates="versions")
    predictions: Mapped[list[PredictionRecord]] = relationship(
        "PredictionRecord",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )


class PredictionRecord(Base):
    """Single inference record scoring an input observation."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model_versions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    subject_ref_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    input_features: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    model_version: Mapped[ModelVersionRecord] = relationship(
        "ModelVersionRecord", back_populates="predictions"
    )
    label: Mapped[PredictionLabelRecord | None] = relationship(
        "PredictionLabelRecord",
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )
    explanations: Mapped[list[ExplanationRunRecord]] = relationship(
        "ExplanationRunRecord",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class PredictionLabelRecord(Base):
    """Ground truth outcome for a previously scored prediction or subject hash (late-arriving labels)."""

    __tablename__ = "prediction_labels"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=True,
        unique=True,
        index=True,
    )
    subject_ref_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    ground_truth: Mapped[int] = mapped_column(Integer, nullable=False)
    label_source: Mapped[str] = mapped_column(String(64), default="user", nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    prediction: Mapped[PredictionRecord | None] = relationship(
        "PredictionRecord", back_populates="label"
    )


class ExplanationRunRecord(Base):
    """Executed explainability run with additivity fidelity report."""

    __tablename__ = "explanation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model_versions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    scope: Mapped[str] = mapped_column(String(32), default="local", nullable=False)
    method: Mapped[str] = mapped_column(String(32), default="shap", nullable=False)
    explainer_type: Mapped[str] = mapped_column(String(64), nullable=False)
    cache_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True, index=True)
    model_output: Mapped[float] = mapped_column(Float, nullable=False)
    base_value: Mapped[float] = mapped_column(Float, nullable=False)
    output_space: Mapped[str] = mapped_column(String(32), default="log_odds", nullable=False)
    contributions: Mapped[dict[str, float]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    grouped_contributions: Mapped[dict[str, float]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    fidelity_passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    max_abs_error: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tolerance: Mapped[float] = mapped_column(Float, default=1e-6, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    caveats: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    prediction: Mapped[PredictionRecord | None] = relationship(
        "PredictionRecord", back_populates="explanations"
    )


class AuditEventRecord(Base):
    """Immutable audit trail of actions taken in the governance platform."""

    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    actor: Mapped[str] = mapped_column(String(128), default="system", nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
