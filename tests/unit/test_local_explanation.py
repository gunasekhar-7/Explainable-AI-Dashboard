"""End-to-end local explanation against a real fitted artifact and real SHAP."""

from __future__ import annotations

import numpy as np
import pytest

from aletheia.core.errors import ValidationFailed
from aletheia.ml.explain import explain_local


def test_explanation_reconstructs_the_model_output(loaded_model, feature_frame, background) -> None:
    row = feature_frame.head(1)
    explanation = explain_local(loaded_model, row, background)
    total = explanation.base_value + sum(explanation.contributions.values())
    assert total == pytest.approx(explanation.model_output, abs=1e-9)
    assert explanation.additivity.passed is True
    assert explanation.additivity.max_abs_error <= 1e-6


def test_explanation_uses_linear_shap_and_states_its_output_space(
    loaded_model, feature_frame, background
) -> None:
    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    assert explanation.explainer_type == "shap.LinearExplainer"
    assert explanation.method == "shap"
    assert explanation.output_space == "log_odds"
    assert any("not a causal effect" in caveat for caveat in explanation.caveats)
    assert any("log_odds" in caveat for caveat in explanation.caveats)


def test_contributions_are_ordered_by_magnitude(loaded_model, feature_frame, background) -> None:
    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    magnitudes = [abs(value) for value in explanation.contributions.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_every_contribution_names_a_known_feature(loaded_model, feature_frame, background) -> None:
    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    known = set(loaded_model.feature_names)
    assert set(explanation.contributions) == known


def test_different_rows_produce_different_contributions(
    loaded_model, feature_frame, background
) -> None:
    first = explain_local(loaded_model, feature_frame.iloc[[0]], background)
    second = explain_local(loaded_model, feature_frame.iloc[[1]], background)
    assert first.contributions != second.contributions


def test_model_output_is_not_a_probability(loaded_model, feature_frame, background) -> None:
    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    probability = loaded_model.predict_proba(feature_frame.head(1))[0]
    assert explanation.model_output != pytest.approx(probability)
    assert 1 / (1 + np.exp(-explanation.model_output)) == pytest.approx(probability, abs=1e-9)


def test_top_k_is_bounded(loaded_model, feature_frame, background) -> None:
    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    assert len(explanation.top(3)) == 3
    assert len(explanation.top(10_000)) == len(loaded_model.feature_names)


def test_multiple_rows_are_rejected(loaded_model, feature_frame, background) -> None:
    with pytest.raises(ValidationFailed, match="exactly one row"):
        explain_local(loaded_model, feature_frame.head(2), background)


def test_oversized_background_is_rejected(loaded_model, feature_frame) -> None:
    with pytest.raises(ValidationFailed, match="too large"):
        explain_local(loaded_model, feature_frame.head(1), feature_frame)


def test_empty_background_is_rejected(loaded_model, feature_frame) -> None:
    with pytest.raises(ValidationFailed, match="background sample is required"):
        explain_local(loaded_model, feature_frame.head(1), feature_frame.head(0))


def test_non_linear_model_uses_kernel_shap_with_a_stated_tolerance(
    churn_like_frame, fitted_preprocessor
) -> None:
    from sklearn.neighbors import KNeighborsClassifier

    from aletheia.ml.adapters import SklearnCompositeAdapter
    from tests.support import FEATURE_COLUMNS

    frame = churn_like_frame
    transformed = fitted_preprocessor.transform(frame[FEATURE_COLUMNS])
    estimator = KNeighborsClassifier(n_neighbors=5).fit(transformed, frame["Churn"])
    payload = {
        "model": estimator,
        "preprocessor": fitted_preprocessor,
        "feature_names": [
            str(name).split("__", 1)[-1]
            for name in fitted_preprocessor.get_feature_names_out()
        ],
        "metadata": {"threshold": 0.5},
    }
    model = SklearnCompositeAdapter().load(payload, sha256="0" * 64)

    explanation = explain_local(
        model, frame[FEATURE_COLUMNS].head(1), frame[FEATURE_COLUMNS].sample(30, random_state=5)
    )
    assert explanation.explainer_type == "shap.KernelExplainer"
    assert explanation.output_space == "probability"
    assert explanation.additivity.passed is True
    assert explanation.additivity.tolerance >= 5e-3
    assert any("approximate" in caveat for caveat in explanation.caveats)


def test_grouped_view_reconstructs_the_same_total(loaded_model, feature_frame, background) -> None:
    from aletheia.ml.schema import group_contributions

    explanation = explain_local(loaded_model, feature_frame.head(1), background)
    grouped = group_contributions(explanation.contributions, loaded_model.schema)
    assert sum(grouped.values()) == pytest.approx(
        sum(explanation.contributions.values()), abs=1e-12
    )
