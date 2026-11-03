"""Audit log query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from aletheia.db.connection import get_db
from aletheia.schemas.audit import AuditEventResponse
from aletheia.services.audit_service import AuditService

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("", response_model=list[AuditEventResponse], summary="List governance audit trail events")
def list_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[AuditEventResponse]:
    service = AuditService(db)
    return service.list_events(limit=limit)

