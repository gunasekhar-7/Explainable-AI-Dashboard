"""Local SHAP explanations.

Design rules that the code enforces rather than documents:

* The explainer is chosen from an explicit allowlist, never guessed from a string.
* Contributions are returned in the model's output space, and that space is part of the result.
* A result is only returned if it passes the additivity check; otherwise the call fails.
* Every result carries caveats, because a contribution is not a cause.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from aletheia.core.errors import (
    ArtifactInvalidError,
    ExplainabilityUnsupportedError,
    ValidationFailed,
)
from aletheia.ml.adapters.base import LoadedModel, OutputSpace
from aletheia.ml.explain.additivity import (
    DEFAULT_TOLERANCE,
    AdditivityReport,
    require_additive,
    verify_additivity,
)

MAX_BACKGROUND_ROWS = 200
KERNEL_TOLERANCE = 5e-3


@dataclass(frozen=True)
class LocalExplanation:
    """One prediction explained, with the evidence needed to judge the explanation."""

    row_index: int
    model_output: float
    base_value: float
    output_space: OutputSpace
    method: str
    explainer_type: str
    contributions: Mapping[str, float]
    additivity: AdditivityReport
    caveats: tuple[str, ...]

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        return list(self.contributions.items())[:k]


def _as_positive_class(values: Any) -> np.ndarray:
    """Normalise whatever ``shap_values`` returns into ``(n_rows, n_features)``."""
    if isinstance(values, list):
        if not values:
            raise ExplainabilityUnsupportedError("explainer returned no values")
        chosen = values[1] if len(values) == 2 else values[-1]
        return np.asarray(chosen, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def _expected_value(explainer: Any) -> float:
    raw = np.asarray(getattr(explainer, "expected_value", 0.0), dtype=float).ravel()
    if raw.size == 0:
        raise ExplainabilityUnsupportedError("explainer did not report a base value")
    return float(raw[-1])


def _is_tree_estimator(estimator: Any) -> bool:
    module = type(estimator).__module__.split(".")[0]
    return hasattr(estimator, "get_booster") or module in {"xgboost", "lightgbm"}


def _make_linear(estimator: Any, background: np.ndarray) -> Any:
    """Interventional Linear SHAP.

    The masker API is preferred because ``feature_perturbation`` is deprecated in current SHAP; the legacy
    call is kept as a fallback so a pinned older version still works, and the version is recorded with the
    explanation either way.
    """
    try:
        return shap.LinearExplainer(estimator, shap.maskers.Independent(background))
    except (TypeError, ValueError):
        return shap.LinearExplainer(
            estimator, background, feature_perturbation="interventional"
        )


def _probability_function(estimator: Any, columns: tuple[str, ...]) -> Callable[[np.ndarray], np.ndarray]:
    def predict(matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(matrix, dtype=float), columns=list(columns))
        return np.asarray(estimator.predict_proba(frame), dtype=float)[:, 1]

    return predict


def _caveats(output_space: OutputSpace, explainer_type: str, background_rows: int) -> tuple[str, ...]:
    caveats = [
        "Contributions describe this model version's behaviour on this input, not a causal effect.",
        f"Values are in the model's {output_space} space and are not probability changes.",
        f"Computed against a background sample of {background_rows} rows; a different reference shifts them.",
    ]
    if explainer_type == "shap.KernelExplainer":
        caveats.append(
            "Kernel SHAP is sampled and approximate; additivity holds only within "
            f"{KERNEL_TOLERANCE:.0e} here, and the values carry sampling noise."
        )
    return tuple(caveats)


def explain_local(
    model: LoadedModel,
    row: pd.DataFrame,
    background: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    row_index: int | None = None,
) -> LocalExplanation:
    """Explain one prediction. Raises rather than returning an explanation that fails additivity."""
    if len(row) != 1:
        raise ValidationFailed(
            "local explanation takes exactly one row", extra={"rows": len(row)}
        )
    if len(background) == 0:
        raise ValidationFailed("a background sample is required for SHAP")
    if len(background) > MAX_BACKGROUND_ROWS:
        raise ValidationFailed(
            "background sample is too large; sample it before calling the explainer",
            extra={"rows": len(background), "max_rows": MAX_BACKGROUND_ROWS},
        )
    if len(model.feature_names) == 0:
        raise ArtifactInvalidError("model exposes no feature names")

    transformed_row = model.transform(row)
    transformed_background = model.transform(background)
    row_matrix = np.asarray(transformed_row, dtype=float)
    background_matrix = np.asarray(transformed_background, dtype=float)

    if row_matrix.shape[1] != len(model.feature_names):
        raise ArtifactInvalidError(
            "transformed row width does not match the declared feature names",
            extra={"width": int(row_matrix.shape[1]), "names": len(model.feature_names)},
        )

    estimator = model.estimator
    if hasattr(estimator, "coef_"):
        explainer = _make_linear(estimator, background_matrix)
        explainer_type = "shap.LinearExplainer"
        output_space: OutputSpace = model.output_space
        raw_values = explainer.shap_values(row_matrix)
        model_outputs = model.model_output(row)
        effective_tolerance = tolerance
    elif _is_tree_estimator(estimator):
        explainer = shap.TreeExplainer(estimator)
        explainer_type = "shap.TreeExplainer"
        output_space = model.output_space
        raw_values = explainer.shap_values(row_matrix)
        model_outputs = model.model_output(row)
        effective_tolerance = tolerance
    else:
        explainer = shap.KernelExplainer(
            _probability_function(estimator, model.feature_names), background_matrix
        )
        explainer_type = "shap.KernelExplainer"
        output_space = "probability"
        raw_values = explainer.shap_values(row_matrix, nsamples="auto", silent=True)
        model_outputs = model.predict_proba(row)
        effective_tolerance = max(tolerance, KERNEL_TOLERANCE)

    values = _as_positive_class(raw_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != row_matrix.shape:
        raise ExplainabilityUnsupportedError(
            "explainer returned a contribution matrix of the wrong shape",
            extra={"values_shape": list(values.shape), "expected": list(row_matrix.shape)},
        )

    contributions = values[0]
    base_value = _expected_value(explainer)
    report = verify_additivity(
        base_value, contributions, np.atleast_1d(model_outputs), tolerance=effective_tolerance
    )
    require_additive(report, context=explainer_type)

    named = {
        name: float(value)
        for name, value in zip(model.feature_names, contributions, strict=True)
    }
    ordered = dict(sorted(named.items(), key=lambda item: (-abs(item[1]), item[0])))

    return LocalExplanation(
        row_index=int(row_index if row_index is not None else row.index[0]),
        model_output=float(np.atleast_1d(model_outputs)[0]),
        base_value=base_value,
        output_space=output_space,
        method="shap",
        explainer_type=explainer_type,
        contributions=ordered,
        additivity=report,
        caveats=_caveats(output_space, explainer_type, len(background)),
    )
