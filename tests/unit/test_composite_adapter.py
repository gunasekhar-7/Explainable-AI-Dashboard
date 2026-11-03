"""Adapter behaviour against the real composite artifact shape (names without prefixes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.adapters import SklearnCompositeAdapter, sha256_file
from aletheia.ml.feature_map import strip_encoder_prefix
from tests.support import FEATURE_COLUMNS


def test_loads_the_composite_shape(loaded_model) -> None:
    assert loaded_model.adapter == "sklearn-composite-v1"
    assert len(loaded_model.feature_names) == len(loaded_model.schema)
    assert loaded_model.schema.source_columns == tuple(FEATURE_COLUMNS)
    assert loaded_model.metadata.threshold == 0.45
    assert loaded_model.output_space == "log_odds"
    assert loaded_model.metadata.library_versions["scikit-learn"]


def test_hash_is_recorded_and_matches_the_file(loaded_model, artifact_path) -> None:
    assert loaded_model.artifact_sha256 == sha256_file(artifact_path)
    assert loaded_model.metadata.artifact_sha256 == loaded_model.artifact_sha256


def test_transform_preserves_names_and_index(loaded_model, feature_frame) -> None:
    sample = feature_frame.head(3)
    transformed = loaded_model.transform(sample)
    assert list(transformed.columns) == list(loaded_model.feature_names)
    assert list(transformed.index) == list(sample.index)


def test_predict_proba_matches_the_underlying_estimator(
    loaded_model, artifact_payload, feature_frame
) -> None:
    sample = feature_frame.head(5)
    probabilities = loaded_model.predict_proba(sample)
    preprocessor = artifact_payload["preprocessor"]
    estimator = artifact_payload["model"]
    expected = estimator.predict_proba(preprocessor.transform(sample))[:, 1]
    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    assert probabilities.shape == (5,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_prediction_is_deterministic(loaded_model, feature_frame) -> None:
    sample = feature_frame.tail(4)
    first = loaded_model.predict_proba(sample)
    second = loaded_model.predict_proba(sample)
    np.testing.assert_array_equal(first, second)


def test_label_uses_the_declared_threshold(loaded_model, feature_frame) -> None:
    sample = feature_frame.head(20)
    probabilities = loaded_model.predict_proba(sample)
    labels = loaded_model.predict_label(sample)
    np.testing.assert_array_equal(labels, (probabilities >= 0.45).astype(int))


def test_model_output_is_log_odds_not_probability(loaded_model, feature_frame) -> None:
    sample = feature_frame.head(3)
    output = loaded_model.model_output(sample)
    probabilities = loaded_model.predict_proba(sample)
    assert not np.allclose(output, probabilities)
    np.testing.assert_allclose(1 / (1 + np.exp(-output)), probabilities, atol=1e-9)


def test_rejects_a_payload_without_a_model() -> None:
    with pytest.raises(ArtifactInvalidError, match="no 'model'"):
        SklearnCompositeAdapter().load({"feature_names": ["a"]}, sha256="0" * 64)


def test_rejects_wrong_feature_name_count(artifact_payload) -> None:
    broken = dict(artifact_payload)
    broken["feature_names"] = list(artifact_payload["feature_names"])[:-1]
    with pytest.raises(ArtifactInvalidError, match="declares"):
        SklearnCompositeAdapter().load(broken, sha256="0" * 64)


def test_rejects_reordered_feature_names(artifact_payload) -> None:
    names = list(artifact_payload["feature_names"])
    broken = dict(artifact_payload)
    broken["feature_names"] = list(reversed(names))
    with pytest.raises(ArtifactInvalidError, match="position"):
        SklearnCompositeAdapter().load(broken, sha256="0" * 64)


def test_rejects_missing_feature_names(artifact_payload) -> None:
    broken = dict(artifact_payload)
    broken["feature_names"] = []
    with pytest.raises(ArtifactInvalidError, match="declares no feature_names"):
        SklearnCompositeAdapter().load(broken, sha256="0" * 64)


def test_rejects_out_of_range_threshold(artifact_payload) -> None:
    broken = dict(artifact_payload)
    broken["metadata"] = {"threshold": 1.5}
    with pytest.raises(ArtifactInvalidError, match="outside"):
        SklearnCompositeAdapter().load(broken, sha256="0" * 64)


def test_rejects_non_mapping_payload() -> None:
    with pytest.raises(ArtifactInvalidError, match="not a mapping"):
        SklearnCompositeAdapter().load(pd.DataFrame({"a": [1]}), sha256="0" * 64)


def test_preprocessor_without_names_is_rejected(artifact_payload) -> None:
    class NoNames:
        def transform(self, frame):  # pragma: no cover - never reached
            return frame

    broken = dict(artifact_payload)
    broken["preprocessor"] = NoNames()
    with pytest.raises(ArtifactInvalidError, match="get_feature_names_out"):
        SklearnCompositeAdapter().load(broken, sha256="0" * 64)


def test_declared_names_are_normalised_for_display(artifact_payload) -> None:
    padded = dict(artifact_payload)
    numeric = {"tenure", "MonthlyCharges", "TotalCharges"}
    padded["feature_names"] = [
        f"{'num' if name in numeric else 'cat'}__{name}"
        for name in artifact_payload["feature_names"]
    ]
    model = SklearnCompositeAdapter().load(padded, sha256="0" * 64)
    assert "tenure" in model.feature_names
    assert all(not name.startswith(("num__", "cat__")) for name in model.feature_names)
    assert strip_encoder_prefix(model.feature_names[0]) == model.feature_names[0]
