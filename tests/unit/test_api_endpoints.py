"""Integration & contract tests for the Aletheia REST API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aletheia.api.app import create_app
from aletheia.core.config import Settings
from aletheia.db.connection import get_db
from aletheia.db.models import Base
from aletheia.ml.adapters import sha256_file


@pytest.fixture
def client_with_db(tmp_path: Path) -> tuple[TestClient, Session]:
    db_file = tmp_path / "test_api.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    settings = Settings(
        _env_file=None,
        environment="ci",
        database_url=db_url,
        artifact_root=tmp_path,
    )
    app = create_app(settings)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    return test_client, session


def test_complete_model_governance_api_lifecycle(
    client_with_db: tuple[TestClient, Session],
    artifact_path: Path,
    feature_frame,
) -> None:
    client, _ = client_with_db

    # 1. Register model
    resp = client.post(
        "/api/v1/models",
        json={
            "name": "churn-production-v1",
            "task_type": "binary_classification",
            "description": "Production telecom churn model",
        },
    )
    assert resp.status_code == 201, resp.text
    model_data = resp.json()
    model_id = model_data["id"]
    assert model_data["name"] == "churn-production-v1"

    # 2. Register version from artifact
    sha = sha256_file(artifact_path)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions",
        json={
            "version": "1.0.0",
            "artifact_uri": str(artifact_path),
            "expected_sha256": sha,
            "framework": "scikit-learn",
        },
    )
    assert resp.status_code == 201, resp.text
    version_data = resp.json()
    assert version_data["id"] is not None
    assert version_data["version"] == "1.0.0"
    assert version_data["artifact_sha256"] == sha
    assert len(version_data["feature_names"]) > 0

    # 3. Model details
    resp = client.get(f"/api/v1/models/{model_id}")
    assert resp.status_code == 200
    assert len(resp.json()["versions"]) == 1

    # 4. Predict single observation
    sample_features = feature_frame.iloc[0].to_dict()
    resp = client.post(
        f"/api/v1/models/{model_id}/predict",
        json={"features": sample_features, "subject_id": "customer-9941"},
    )
    assert resp.status_code == 200, resp.text
    pred_data = resp.json()
    prediction_id = pred_data["id"]
    assert 0.0 <= pred_data["probability"] <= 1.0
    assert pred_data["predicted_label"] in {0, 1}
    assert pred_data["subject_ref_hash"] is not None

    # 5. Batch prediction
    resp = client.post(
        f"/api/v1/models/{model_id}/predict/batch",
        json={
            "items": [
                {"features": feature_frame.iloc[i].to_dict(), "subject_id": f"cust-{i}"}
                for i in range(3)
            ]
        },
    )
    assert resp.status_code == 200
    batch_data = resp.json()
    assert batch_data["count"] == 3
    assert len(batch_data["predictions"]) == 3

    # 6. Explain prediction with strict additivity verification
    resp = client.post(
        f"/api/v1/models/{model_id}/explain",
        json={"features": sample_features, "prediction_id": prediction_id},
    )
    assert resp.status_code == 200, resp.text
    exp_data = resp.json()
    assert exp_data["explainer_type"] == "shap.LinearExplainer"
    assert exp_data["output_space"] == "log_odds"
    assert exp_data["additivity"]["passed"] is True
    assert exp_data["additivity"]["max_abs_error"] <= 1e-6
    assert len(exp_data["contributions"]) > 0
    assert len(exp_data["grouped_contributions"]) > 0
    assert any("not a causal effect" in c for c in exp_data["caveats"])

    # 7. Record late-arriving ground truth label
    resp = client.post(
        f"/api/v1/models/{model_id}/labels",
        json={
            "prediction_id": prediction_id,
            "ground_truth": 1,
            "label_source": "retention_crm",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["ground_truth"] == 1

    # 8. Check calibration and performance metrics
    resp = client.get(f"/api/v1/models/{model_id}/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert metrics["total_labeled"] == 1
    assert "accuracy" in metrics
    assert "brier_score" in metrics
    assert "calibration_gap" in metrics

    # 9. Check governance audit trail
    resp = client.get("/api/v1/audit")
    assert resp.status_code == 200
    audit_events = resp.json()
    assert len(audit_events) >= 3
    actions = [e["action"] for e in audit_events]
    assert "model.created" in actions
    assert "model_version.registered" in actions
    assert "label.recorded" in actions


def test_api_rejections_and_errors(
    client_with_db: tuple[TestClient, Session],
    artifact_path: Path,
) -> None:
    client, _ = client_with_db

    # Non-existent model 404
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/api/v1/models/{fake_uuid}")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "RESOURCE_NOT_FOUND"

    # Missing artifact 404
    resp = client.post(
        "/api/v1/models",
        json={"name": "test-err-model"},
    )
    m_id = resp.json()["id"]

    resp = client.post(
        f"/api/v1/models/{m_id}/versions",
        json={
            "version": "1.0.0",
            "artifact_uri": "non_existent_file.joblib",
        },
    )
    assert resp.status_code == 404

    # Duplicate model name rejected with 422
    resp = client.post(
        "/api/v1/models",
        json={"name": "test-err-model"},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "ARTIFACT_INVALID"
