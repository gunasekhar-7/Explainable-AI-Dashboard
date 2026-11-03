"""Error documents: stable codes, no leaked internals, request correlation."""

from __future__ import annotations

import pytest

from aletheia.core.errors import (
    AletheiaError,
    ArtifactInvalidError,
    ArtifactUnavailableError,
    ExplainabilityUnsupportedError,
    ExplanationFidelityError,
    NotFoundError,
    ValidationFailed,
)


def test_default_problem_document_shape() -> None:
    problem = AletheiaError("boom").to_problem()
    assert problem["status"] == 500
    assert problem["code"] == "INTERNAL_ERROR"
    assert problem["title"] == "Internal error"
    assert problem["type"] == "https://aletheia.local/problems/internal_error"
    assert "request_id" not in problem
    assert "instance" not in problem


def test_request_id_and_instance_are_attached_when_known() -> None:
    problem = ArtifactInvalidError("bad artifact").to_problem(
        request_id="req-123", instance="/api/v1/models"
    )
    assert problem["request_id"] == "req-123"
    assert problem["instance"] == "/api/v1/models"


def test_extra_payload_reaches_the_document() -> None:
    problem = ArtifactInvalidError(
        "duplicate features", extra={"duplicates": ["a", "b"]}
    ).to_problem()
    assert problem["duplicates"] == ["a", "b"]


def test_extra_mapping_is_copied() -> None:
    source = {"a": 1}
    error = AletheiaError("boom", extra=source)
    source["a"] = 2
    assert error.extra == {"a": 1}


def test_overrides_are_supported() -> None:
    error = AletheiaError("teapot", code="TEAPOT", status=418, title="I am a teapot")
    assert (error.code, error.status, error.title) == ("TEAPOT", 418, "I am a teapot")
    assert error.to_problem()["type"].endswith("/teapot")


@pytest.mark.parametrize(
    ("error_class", "status", "code"),
    [
        (ValidationFailed, 422, "VALIDATION_FAILED"),
        (NotFoundError, 404, "RESOURCE_NOT_FOUND"),
        (ArtifactInvalidError, 422, "ARTIFACT_INVALID"),
        (ArtifactUnavailableError, 503, "ARTIFACT_UNAVAILABLE"),
        (ExplainabilityUnsupportedError, 422, "EXPLAINER_UNSUPPORTED"),
        (ExplanationFidelityError, 500, "EXPLANATION_FIDELITY_ERROR"),
    ],
)
def test_subclasses_keep_stable_codes(error_class, status, code) -> None:
    error = error_class("detail text")
    assert error.status == status
    assert error.code == code
    assert error.detail == "detail text"
    assert error.to_problem()["status"] == status
