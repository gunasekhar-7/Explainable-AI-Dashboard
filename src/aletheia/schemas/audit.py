"""Pydantic schemas for audit logs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AuditEventResponse(BaseModel):
    id: uuid.UUID
    action: str
    resource_type: str
    resource_id: str
    actor: str
    details: dict[str, Any]
    created_at: datetime

