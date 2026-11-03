"""Shared fixtures.

Every test runs against a synthetic composite artifact built here, not against a checked-in pickle: the
artifact contract is what is under test, and generating it keeps the repository free of binary blobs. The
frame mimics the upstream churn dataset's column types without copying its rows.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aletheia.ml.adapters import LoadedModel, load_artifact
from aletheia.ml.feature_map import strip_encoder_prefix
from tests.support import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, TARGET


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host environment variables out of configuration tests."""
    import os

    for key in list(os.environ):
        if key.startswith("ALETHEIA_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def churn_like_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_rows = 240
    tenure = rng.integers(0, 72, size=n_rows)
    monthly = np.round(rng.uniform(20, 120, size=n_rows), 2)
    total = np.round(tenure * monthly + rng.uniform(0, 50, size=n_rows), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fibre optic", "None"], n_rows, p=[0.40, 0.45, 0.15])
    payment = rng.choice(["Electronic check", "Credit card"], n_rows, p=[0.60, 0.40])

    logit = (
        -1.6
        + 0.030 * (72 - tenure)
        + 0.012 * (monthly - 60)
        + np.where(contract == "Month-to-month", 0.9, -0.4)
        + np.where(internet == "Fibre optic", 0.5, -0.1)
        + rng.normal(0, 0.35, n_rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            TARGET: churn,
        }
    )
    assert frame[TARGET].nunique() == 2, "fixture must contain both classes"
    return frame


@pytest.fixture(scope="session")
def fitted_preprocessor(churn_like_frame: pd.DataFrame) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    preprocessor.fit(churn_like_frame[FEATURE_COLUMNS])
    return preprocessor


@pytest.fixture(scope="session")
def artifact_payload(
    churn_like_frame: pd.DataFrame, fitted_preprocessor: ColumnTransformer
) -> dict[str, object]:
    features = churn_like_frame[FEATURE_COLUMNS]
    transformed = fitted_preprocessor.transform(features)
    estimator = LogisticRegression(max_iter=2000).fit(transformed, churn_like_frame[TARGET])

    # The upstream artifact declares names without the num__/cat__ prefixes; reproduce that quirk so the
    # adapter is tested against the real shape rather than a convenient one.
    declared = [strip_encoder_prefix(str(name)) for name in fitted_preprocessor.get_feature_names_out()]
    return {
        "model": estimator,
        "preprocessor": fitted_preprocessor,
        "feature_names": declared,
        "metadata": {
            "threshold": 0.45,
            "library_versions": {"scikit-learn": sklearn.__version__},
            "source": "synthetic fixture",
        },
    }


@pytest.fixture
def artifact_path(tmp_path: Path, artifact_payload: dict[str, object]) -> Path:
    path = tmp_path / "churn-composite.joblib"
    joblib.dump(artifact_payload, path)
    return path


@pytest.fixture
def loaded_model(artifact_path: Path) -> LoadedModel:
    return load_artifact(artifact_path, max_bytes=10 * 1024 * 1024)


@pytest.fixture
def feature_frame(churn_like_frame: pd.DataFrame) -> pd.DataFrame:
    return churn_like_frame[FEATURE_COLUMNS]


@pytest.fixture
def background(feature_frame: pd.DataFrame) -> pd.DataFrame:
    return feature_frame.sample(n=40, random_state=11)
