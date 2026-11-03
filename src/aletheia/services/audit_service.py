"""Platform governance audit logging service."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from aletheia.db.repositories import AuditRepository
from aletheia.schemas.audit import AuditEventResponse


class AuditService:
    """Service tracking governance actions and provenance."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = AuditRepository(session)

    def record_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEventResponse:
        record = self.repo.record(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor=actor,
            details=details,
        )
        return AuditEventResponse(
            id=record.id,
            action=record.action,
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            actor=record.actor,
            details=record.details,
            created_at=record.created_at,
        )

    def list_events(self, limit: int = 100) -> list[AuditEventResponse]:
        records = self.repo.list_events(limit=limit)
        return [
            AuditEventResponse(
                id=rec.id,
                action=rec.action,
                resource_type=rec.resource_type,
                resource_id=rec.resource_id,
                actor=rec.actor,
                details=rec.details,
                created_at=rec.created_at,
            )
            for rec in records
        ]

