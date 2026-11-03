"""Error taxonomy and RFC 9457 problem documents.

Every application error carries a stable machine-readable ``code``. Clients branch on the code; the
``detail`` string is for humans and may change. Nothing here includes a stack trace, a credential, or a raw
artifact hash fragment in the message.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROBLEM_BASE = "https://aletheia.local/problems"


class AletheiaError(Exception):
    """Base class for errors that are safe to render to a client."""

    code: str = "INTERNAL_ERROR"
    status: int = 500
    title: str = "Internal error"

    def __init__(
        self,
        detail: str,
        *,
        code: str | None = None,
        status: int | None = None,
        title: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        if title is not None:
            self.title = title
        self.extra: dict[str, Any] = dict(extra or {})

    def to_problem(self, *, request_id: str | None = None, instance: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": f"{PROBLEM_BASE}/{self.code.lower()}",
            "title": self.title,
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
        }
        if instance is not None:
            body["instance"] = instance
        if request_id is not None:
            body["request_id"] = request_id
        body.update(self.extra)
        return body


class ValidationFailed(AletheiaError):
    code = "VALIDATION_FAILED"
    status = 422
    title = "Request validation failed"


class NotFoundError(AletheiaError):
    code = "RESOURCE_NOT_FOUND"
    status = 404
    title = "Resource not found"


class ArtifactInvalidError(AletheiaError):
    """The artifact is present but is not something this platform can govern."""

    code = "ARTIFACT_INVALID"
    status = 422
    title = "Artifact rejected"


class ArtifactUnavailableError(AletheiaError):
    """The artifact cannot be loaded right now. Never a reason to load it in-process instead."""

    code = "ARTIFACT_UNAVAILABLE"
    status = 503
    title = "Artifact unavailable"


class ExplainabilityUnsupportedError(AletheiaError):
    """No explainer in the allowlist applies to this estimator."""

    code = "EXPLAINER_UNSUPPORTED"
    status = 422
    title = "No supported explainer for this model"


class ExplanationFidelityError(AletheiaError):
    """The explanation failed its additivity check and must not be persisted or returned."""

    code = "EXPLANATION_FIDELITY_ERROR"
    status = 500
    title = "Explanation failed its fidelity check"
