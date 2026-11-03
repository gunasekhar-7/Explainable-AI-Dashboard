"""Explainability and attribution service."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from aletheia.db.repositories import ExplanationRepository
from aletheia.ml.explain import explain_local
from aletheia.ml.schema import group_contributions
from aletheia.schemas.explanations import (
    AdditivityReportSchema,
    ExplanationRequest,
    ExplanationResponse,
)
from aletheia.services.model_service import ModelService


def _generate_cache_key(artifact_sha256: str, features: dict[str, Any]) -> str:
    serialized = json.dumps(features, sort_keys=True, default=str)
    raw = f"{artifact_sha256}:{serialized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExplanationService:
    """Service computing and persisting local SHAP explanations with strict additivity."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = ExplanationRepository(session)
        self.model_service = ModelService(session)

    def explain(
        self, model_version_id: uuid.UUID, request: ExplanationRequest
    ) -> ExplanationResponse:
        version_rec, model = self.model_service.get_loaded_model(model_version_id)
        cache_key = _generate_cache_key(version_rec.artifact_sha256, request.features)

        # Check content-addressed cache
        cached = self.repo.get_by_cache_key(cache_key)
        if cached is not None and cached.fidelity_passed:
            return ExplanationResponse(
                id=cached.id,
                model_version_id=cached.model_version_id,
                prediction_id=cached.prediction_id,
                explainer_type=cached.explainer_type,
                output_space=cached.output_space,
                model_output=cached.model_output,
                base_value=cached.base_value,
                contributions=dict(cached.contributions),
                grouped_contributions=dict(cached.grouped_contributions),
                additivity=AdditivityReportSchema(
                    max_abs_error=cached.max_abs_error,
                    tolerance=cached.tolerance,
                    rows_checked=1,
                    passed=cached.fidelity_passed,
                ),
                duration_ms=cached.duration_ms,
                caveats=list(cached.caveats),
                created_at=cached.created_at,
            )

        row_df = pd.DataFrame([request.features])

        # Prepare background
        if request.background_samples:
            background_df = pd.DataFrame(request.background_samples)
        else:
            # Generate synthetic perturbation baseline around row for reference
            n_samples = 40
            rng = np.random.default_rng(42)
            bg_data: dict[str, Any] = {}
            for col in row_df.columns:
                val = row_df[col].iloc[0]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    scale = abs(float(val)) * 0.2 if val != 0 else 1.0
                    bg_data[col] = float(val) + rng.normal(0, scale, n_samples)
                else:
                    bg_data[col] = [val] * n_samples
            background_df = pd.DataFrame(bg_data)

        start_time = time.perf_counter()
        local_exp = explain_local(model, row_df, background_df)
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        grouped = group_contributions(local_exp.contributions, model.schema)

        run_record = self.repo.create(
            model_version_id=version_rec.id,
            prediction_id=request.prediction_id,
            explainer_type=local_exp.explainer_type,
            model_output=local_exp.model_output,
            base_value=local_exp.base_value,
            output_space=local_exp.output_space,
            contributions=dict(local_exp.contributions),
            grouped_contributions=dict(grouped),
            fidelity_passed=local_exp.additivity.passed,
            max_abs_error=local_exp.additivity.max_abs_error,
            tolerance=local_exp.additivity.tolerance,
            duration_ms=duration_ms,
            caveats=list(local_exp.caveats),
            cache_key=cache_key,
        )

        return ExplanationResponse(
            id=run_record.id,
            model_version_id=run_record.model_version_id,
            prediction_id=run_record.prediction_id,
            explainer_type=run_record.explainer_type,
            output_space=run_record.output_space,
            model_output=run_record.model_output,
            base_value=run_record.base_value,
            contributions=dict(run_record.contributions),
            grouped_contributions=dict(run_record.grouped_contributions),
            additivity=AdditivityReportSchema(
                max_abs_error=run_record.max_abs_error,
                tolerance=run_record.tolerance,
                rows_checked=1,
                passed=run_record.fidelity_passed,
            ),
            duration_ms=run_record.duration_ms,
            caveats=list(run_record.caveats),
            created_at=run_record.created_at,
        )
