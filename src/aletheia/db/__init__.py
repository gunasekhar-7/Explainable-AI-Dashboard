"""Persistence layer for Aletheia: models, repositories, and database connection."""

from __future__ import annotations

from aletheia.db.connection import get_db, get_engine, get_session_factory, init_db
from aletheia.db.models import (
    AuditEventRecord,
    Base,
    ExplanationRunRecord,
    ModelRecord,
    ModelVersionRecord,
    PredictionLabelRecord,
    PredictionRecord,
)

__all__ = [
    "AuditEventRecord",
    "Base",
    "ExplanationRunRecord",
    "ModelRecord",
    "ModelVersionRecord",
    "PredictionLabelRecord",
    "PredictionRecord",
    "get_db",
    "get_engine",
    "get_session_factory",
    "init_db",
]

