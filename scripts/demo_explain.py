"""Walking skeleton: artifact -> preprocessing -> probability -> SHAP -> additivity -> grouped view.

Run from the project root:

    PYTHONPATH=src python scripts/demo_explain.py

The model is trained here on synthetic data so the repository carries no binary artifact (ADR-015). What is
demonstrated is the path a real artifact takes: the same adapter, the same feature map and the same
additivity gate the API will use. The numbers printed are real outputs of this run on this synthetic data;
they are not churn results and are not quoted as such anywhere.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aletheia.ml.adapters import load_artifact, sha256_file
from aletheia.ml.explain import explain_local
from aletheia.ml.feature_map import strip_encoder_prefix
from aletheia.ml.schema import group_contributions

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"
ARTIFACT = Path("artifacts/demo-composite.joblib")
SEED = 7


def synthetic_frame(rows: int = 240, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=rows)
    monthly = np.round(rng.uniform(20, 120, size=rows), 2)
    total = np.round(tenure * monthly + rng.uniform(0, 50, size=rows), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fibre optic", "None"], rows, p=[0.40, 0.45, 0.15])
    payment = rng.choice(["Electronic check", "Credit card"], rows, p=[0.60, 0.40])
    logit = (
        -1.6
        + 0.030 * (72 - tenure)
        + 0.012 * (monthly - 60)
        + np.where(contract == "Month-to-month", 0.9, -0.4)
        + np.where(internet == "Fibre optic", 0.5, -0.1)
        + rng.normal(0, 0.35, rows)
    )
    churn = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return pd.DataFrame(
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


def build_artifact(path: Path) -> None:
    frame = synthetic_frame()
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    features = frame[FEATURES]
    preprocessor.fit(features)
    estimator = LogisticRegression(max_iter=2000).fit(
        preprocessor.transform(features), frame[TARGET]
    )
    declared = [
        strip_encoder_prefix(str(name))
        for name in preprocessor.get_feature_names_out()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": estimator,
            "preprocessor": preprocessor,
            "feature_names": declared,
            "metadata": {
                "threshold": 0.45,
                "library_versions": {"scikit-learn": sklearn.__version__},
                "source": "synthetic demo fixture",
            },
        },
        path,
    )


def main() -> int:
    if not ARTIFACT.exists():
        print(f"building demo artifact at {ARTIFACT} ...")
        build_artifact(ARTIFACT)

    model = load_artifact(ARTIFACT, max_bytes=64 * 1024 * 1024)
    frame = synthetic_frame()
    row = frame.head(1)
    background = frame.sample(n=60, random_state=3)

    probability = float(model.predict_proba(row)[0])
    label = int(model.predict_label(row)[0])
    explanation = explain_local(model, row, background)
    grouped = group_contributions(explanation.contributions, model.schema)

    print()
    print(f"artifact       {model.adapter}  sha256 {sha256_file(ARTIFACT)[:16]}...")
    print(f"features       {len(model.feature_names)} transformed from "
          f"{len(model.schema.source_columns)} raw columns")
    print(f"threshold      {model.metadata.threshold}")
    print(f"row 0          probability {probability:.4f} -> label {label}")
    print(f"model output   {model.output_space} {explanation.model_output:+.4f} "
          f"(base {explanation.base_value:+.4f})")
    print(
        f"additivity     |base + sum(phi) - output| = {explanation.additivity.max_abs_error:.2e} "
        f"-> {'PASS' if explanation.additivity.passed else 'FAIL'} "
        f"(tolerance {explanation.additivity.tolerance:.0e})"
    )
    print(f"explainer      {explanation.explainer_type}")

    print("\ntop contributions (model output space, not probability):")
    for name, value in explanation.top(5):
        print(f"  {value:+.4f}  {name}")

    print("\ngrouped by source field (aggregation of the same attributions):")
    for name, value in list(grouped.items())[:5]:
        print(f"  {value:+.4f}  {name}")

    print("\ncaveats carried with every explanation:")
    for caveat in explanation.caveats:
        print(f"  - {caveat}")

    return 0 if explanation.additivity.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
