"""The contract every adapter must satisfy.

The important detail is that ``model_output`` and ``predict_proba`` are different functions and are never
conflated: SHAP explains the model's output, which for a logistic regression is the log-odds, not the
probability. Reporting a SHAP value as "probability impact" is the most common way an explainability
dashboard misleads its users, so the output space travels with every value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from aletheia.core.errors import ArtifactInvalidError, ExplainabilityUnsupportedError
from aletheia.ml.schema import FeatureSchema

OutputSpace = Literal["log_odds", "probability", "margin"]


def _logit(probability: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    result: np.ndarray = np.log(clipped / (1.0 - clipped))
    return result


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_sha256: str
    adapter: str
    library_versions: Mapping[str, str] = field(default_factory=dict)
    threshold: float | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """A loaded artifact plus everything needed to explain it honestly."""

    artifact_sha256: str
    adapter: str
    estimator: Any
    feature_names: tuple[str, ...]
    schema: FeatureSchema
    metadata: ArtifactMetadata
    preprocessor: Any | None = None
    output_space: OutputSpace = "log_odds"

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the artifact's preprocessing, keeping column names attached."""
        if self.preprocessor is None:
            return features
        transformed = self.preprocessor.transform(features)
        if isinstance(transformed, pd.DataFrame):
            return transformed
        matrix = np.asarray(transformed)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ArtifactInvalidError(
                "preprocessor output shape does not match the declared feature names",
                extra={"expected_columns": len(self.feature_names)},
            )
        return pd.DataFrame(matrix, columns=list(self.feature_names), index=features.index)

    def estimator_input(self, matrix: pd.DataFrame) -> Any:
        """What the estimator should be handed.

        When a preprocessor exists it has already consumed the raw frame, so the estimator is given a plain
        array — which is what it was fitted on. When there is no preprocessor the estimator owns the raw
        columns and receives the frame with names intact.
        """
        return matrix if self.preprocessor is None else matrix.to_numpy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Positive-class probability. Binary and scalar outputs only, by design (v1)."""
        matrix = self.estimator_input(self.transform(features))
        predict_proba = getattr(self.estimator, "predict_proba", None)
        if predict_proba is None:
            raise ExplainabilityUnsupportedError(
                "model does not expose predict_proba; a probability cannot be reported for it"
            )
        proba = np.asarray(predict_proba(matrix), dtype=float)
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                raise ExplainabilityUnsupportedError(
                    "only binary classification is supported in v1",
                    extra={"classes": int(proba.shape[1])},
                )
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        raise ExplainabilityUnsupportedError("model returned an unsupported probability shape")

    def model_output(self, features: pd.DataFrame) -> np.ndarray:
        """The quantity SHAP explains: log-odds/margin, or probability for a probability-linked explainer."""
        if self.output_space == "probability":
            return self.predict_proba(features)
        matrix = self.estimator_input(self.transform(features))
        decision_function = getattr(self.estimator, "decision_function", None)
        if decision_function is not None:
            values: np.ndarray = np.asarray(decision_function(matrix), dtype=float).ravel()
            return values
        # No decision_function: reconstruct the log-odds from the probability and say so via output_space.
        return _logit(self.predict_proba(features))

    def predict_label(self, features: pd.DataFrame, *, threshold: float | None = None) -> np.ndarray:
        cut = threshold if threshold is not None else self.metadata.threshold
        if cut is None:
            raise ArtifactInvalidError(
                "no decision threshold supplied and the artifact does not declare one"
            )
        return (self.predict_proba(features) >= cut).astype(int)


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural contract for artifact adapters."""

    name: str

    def can_load(self, payload: object) -> bool: ...

    def load(self, payload: object, *, sha256: str) -> LoadedModel: ...
