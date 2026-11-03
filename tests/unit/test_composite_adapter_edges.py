"""Adapter rejections and fallbacks: every malformed artifact gets a specific error, not a stack trace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.adapters import SklearnCompositeAdapter
from aletheia.ml.adapters.sklearn_composite import _raw_columns
from tests.support import CATEGORICAL, FEATURE_COLUMNS, NUMERIC

SHA = "0" * 64


def test_unknown_keys_are_recorded_not_rejected(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["trained_on"] = "elsewhere"
    model = SklearnCompositeAdapter().load(payload, sha256=SHA)
    assert model.metadata.notes["undeclared_keys"] == ["trained_on"]


def test_model_without_predict_is_rejected(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["model"] = object()
    with pytest.raises(ArtifactInvalidError, match="not an estimator"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_feature_names_must_be_a_sequence(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["feature_names"] = "tenure,MonthlyCharges"
    with pytest.raises(ArtifactInvalidError, match="sequence of strings"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_non_numeric_threshold_is_rejected(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["metadata"] = {"threshold": "high"}
    with pytest.raises(ArtifactInvalidError, match="not numeric"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_library_versions_must_be_a_mapping(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["metadata"] = {"library_versions": ["scikit-learn"]}
    with pytest.raises(ArtifactInvalidError, match="library_versions"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_absent_threshold_is_reported_as_absent(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["metadata"] = {}
    model = SklearnCompositeAdapter().load(payload, sha256=SHA)
    assert model.metadata.threshold is None
    with pytest.raises(ArtifactInvalidError, match="threshold"):
        model.predict_label(pd.DataFrame({column: [0] for column in FEATURE_COLUMNS}))


def test_preprocessor_without_declared_names_is_rejected(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["preprocessor"] = payload["preprocessor"]
    payload["feature_names"] = []
    with pytest.raises(ArtifactInvalidError, match="declares no feature_names"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_preprocessor_none_uses_declared_names(artifact_payload) -> None:
    """A bare estimator with declared names is governable, at the cost of no raw-column mapping."""
    payload = dict(artifact_payload)
    payload["preprocessor"] = None
    model = SklearnCompositeAdapter().load(payload, sha256=SHA)
    assert model.preprocessor is None
    assert model.feature_names == tuple(payload["feature_names"])
    assert all(spec.kind == "other" for spec in model.schema.features)


def test_preprocessor_without_names_and_without_declaration_is_rejected(artifact_payload) -> None:
    payload = dict(artifact_payload)
    payload["preprocessor"] = None
    payload["feature_names"] = []
    with pytest.raises(ArtifactInvalidError, match="feature identity"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_preprocessor_that_cannot_report_names_is_rejected(artifact_payload) -> None:
    class Exploding:
        def get_feature_names_out(self):
            raise RuntimeError("unfitted")

        def transform(self, frame):  # pragma: no cover - never reached
            return frame

    payload = dict(artifact_payload)
    payload["preprocessor"] = Exploding()
    with pytest.raises(ArtifactInvalidError, match="could not report"):
        SklearnCompositeAdapter().load(payload, sha256=SHA)


def test_raw_columns_come_from_the_transformers_when_absent() -> None:
    class Fitted:
        def __init__(self) -> None:
            self.transformers_ = [
                ("num", None, ["tenure", "MonthlyCharges"]),
                ("cat", None, ["Contract"]),
                ("passthrough", None, "InternetService"),
                ("drop", None, ["never"]),
                ("positional", None, [0, 1]),
            ]

    assert _raw_columns(Fitted()) == (
        "tenure",
        "MonthlyCharges",
        "Contract",
        "InternetService",
        "never",
    )


def test_feature_names_in_is_preferred_when_present(fitted_preprocessor) -> None:
    assert _raw_columns(fitted_preprocessor) == tuple(NUMERIC + CATEGORICAL)


def test_transform_output_shape_is_enforced(artifact_payload) -> None:
    class SilentTruncation:
        def get_feature_names_out(self):
            return np.array([str(name) for name in artifact_payload["feature_names"]])

        def transform(self, frame):
            return np.zeros((len(frame), 3))

    payload = dict(artifact_payload)
    payload["preprocessor"] = SilentTruncation()
    model = SklearnCompositeAdapter().load(payload, sha256=SHA)
    with pytest.raises(ArtifactInvalidError, match="shape"):
        model.transform(pd.DataFrame({column: [0] for column in FEATURE_COLUMNS}))
