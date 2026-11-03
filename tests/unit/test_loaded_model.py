"""LoadedModel behaviour at its edges: missing methods, wrong widths, multiclass, thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from aletheia.core.errors import ArtifactInvalidError, ExplainabilityUnsupportedError
from aletheia.ml.adapters.base import ArtifactMetadata, LoadedModel
from aletheia.ml.schema import FeatureSchema, FeatureSpec


def _schema(names: list[str]) -> FeatureSchema:
    return FeatureSchema(
        features=tuple(FeatureSpec(name=name, source_column=name) for name in names),
        source_columns=tuple(names),
    )


def _model(estimator, names: list[str], **overrides) -> LoadedModel:
    defaults = {
        "artifact_sha256": "0" * 64,
        "adapter": "test",
        "estimator": estimator,
        "feature_names": tuple(names),
        "schema": _schema(names),
        "metadata": ArtifactMetadata(
            artifact_sha256="0" * 64, adapter="test", threshold=overrides.pop("threshold", 0.5)
        ),
        "preprocessor": None,
        "output_space": "log_odds",
    }
    defaults.update(overrides)
    return LoadedModel(**defaults)


def test_frame_is_passed_through_when_there_is_no_preprocessor() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    estimator = LogisticRegression().fit(frame, [0, 1])
    model = _model(estimator, ["a", "b"])
    probabilities = model.predict_proba(frame.head(1))
    assert 0.0 <= float(probabilities[0]) <= 1.0


def test_transform_rejects_a_width_mismatch() -> None:
    class WrongWidth:
        def transform(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(frame), 3))

    model = _model(LogisticRegression(), ["a", "b"], preprocessor=WrongWidth())
    with pytest.raises(ArtifactInvalidError, match="shape does not match"):
        model.transform(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_probability_requires_predict_proba() -> None:
    model = _model(LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0]), ["a"])
    with pytest.raises(ExplainabilityUnsupportedError, match="predict_proba"):
        model.predict_proba(pd.DataFrame({"a": [0.5]}))


def test_multiclass_output_is_refused() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    estimator = LogisticRegression(max_iter=500).fit(frame, [0, 1, 2, 0, 1, 2])
    model = _model(estimator, ["a"])
    with pytest.raises(ExplainabilityUnsupportedError, match="binary"):
        model.predict_proba(frame.head(1))


def test_label_requires_a_threshold_somewhere() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    estimator = LogisticRegression().fit(frame, [0, 0, 1, 1])
    model = _model(estimator, ["a"], threshold=None)
    with pytest.raises(ArtifactInvalidError, match="threshold"):
        model.predict_label(frame.head(1))


def test_label_override_does_not_mutate_the_model() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    estimator = LogisticRegression().fit(frame, [0, 0, 1, 1])
    model = _model(estimator, ["a"], threshold=0.5)
    labels = model.predict_label(frame.head(1), threshold=0.01)
    assert int(labels[0]) == 1
    assert model.metadata.threshold == 0.5


def test_log_odds_are_reconstructed_when_the_estimator_has_no_decision_function() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})
    estimator = KNeighborsClassifier(n_neighbors=3).fit(frame, [0, 1, 0, 1, 0, 1])
    model = _model(estimator, ["a", "b"])
    row = frame.head(1)
    output = float(model.model_output(row)[0])
    probability = float(model.predict_proba(row)[0])
    assert np.isclose(1 / (1 + np.exp(-output)), probability, atol=1e-9)


def test_probability_output_space_returns_probabilities() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    estimator = LogisticRegression().fit(frame, [0, 0, 1, 1, 1, 1])
    model = _model(estimator, ["a"], output_space="probability")
    np.testing.assert_allclose(
        model.model_output(frame.head(2)), model.predict_proba(frame.head(2)), atol=1e-12
    )
