"""Liveness, readiness and capability discovery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from aletheia import __version__
from aletheia.core.config import Settings
from aletheia.core.errors import PROBLEM_BASE
from aletheia.core.health import check_database

router = APIRouter(tags=["system"])
api_router = APIRouter(tags=["system"])

PROBLEM_MEDIA_TYPE = "application/problem+json"


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


@router.get("/health", summary="Liveness")
def health() -> dict[str, Any]:
    """Process is up. Deliberately independent of the database: a liveness probe that fails on a
    dependency outage causes restart loops that make the outage worse."""
    return {"status": "ok", "version": __version__}


@router.get("/health/ready", summary="Readiness")
def readiness(request: Request) -> JSONResponse:
    settings = _settings(request)
    database_ok, database_detail = check_database(settings.database_url)
    checks = {"database": {"ok": database_ok, "detail": database_detail}}
    if database_ok:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "version": __version__,
                "environment": settings.environment,
                "checks": checks,
            },
        )
    return JSONResponse(
        status_code=503,
        content={
            "type": f"{PROBLEM_BASE}/not_ready",
            "title": "Service not ready",
            "status": 503,
            "code": "NOT_READY",
            "detail": "a required dependency is unavailable",
            "checks": checks,
        },
        media_type=PROBLEM_MEDIA_TYPE,
    )


@api_router.get("/meta", summary="Capabilities and limits")
def meta(request: Request) -> dict[str, Any]:
    """What this build can do, stated so a client can check it instead of guessing."""
    settings = _settings(request)
    return {
        "version": __version__,
        "environment": settings.environment,
        "capabilities": {
            "artifacts": ["sklearn-composite-v1"],
            "explainers": ["shap.LinearExplainer", "shap.TreeExplainer", "shap.KernelExplainer"],
            "output_spaces": ["log_odds", "margin", "probability"],
            "fairness": "reported, not adjudicated",
            "local_explanations": ["shap"],
        },
        "limits": {
            "worker_processes": settings.resolved_worker_processes(),
            "explainer_max_rows": settings.explainer_max_rows,
            "max_artifact_bytes": settings.max_artifact_bytes,
        },
        "caveats": [
            "Explanations describe model behaviour, not causal effects.",
            "SHAP values are in the model's output space, not probabilities.",
            "Fairness metrics are reported with sample sizes; no mitigation is applied in v1.",
        ],
    }
