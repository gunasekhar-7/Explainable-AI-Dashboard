"""Unit tests for services layer."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aletheia.core.errors import ArtifactInvalidError, NotFoundError, ValidationFailed
from aletheia.db.models import Base
from aletheia.ml.adapters import sha256_file
from aletheia.schemas.explanations import ExplanationRequest
from aletheia.schemas.labels import LabelCreate
from aletheia.schemas.models import ModelCreate, ModelVersionCreate
from aletheia.schemas.predictions import BatchPredictionRequest, PredictionRequest
from aletheia.services.audit_service import AuditService
from aletheia.services.explanation_service import ExplanationService
from aletheia.services.label_service import LabelService
from aletheia.services.model_service import ModelService
from aletheia.services.prediction_service import PredictionService


@pytest.fixture
def service_session() -> Session:
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_services_full_flow(
    service_session: Session,
    artifact_path: Path,
    feature_frame,
) -> None:
    m_svc = ModelService(service_session)
    p_svc = PredictionService(service_session)
    e_svc = ExplanationService(service_session)
    l_svc = LabelService(service_session)
    a_svc = AuditService(service_session)

    # 1. Model & Version registration
    model = m_svc.create_model(ModelCreate(name="telecom-churn", description="Churn v1"))
    assert model.name == "telecom-churn"

    sha = sha256_file(artifact_path)
    version = m_svc.register_version(
        model.id,
        ModelVersionCreate(
            version="1.0.0",
            artifact_uri=str(artifact_path),
            expected_sha256=sha,
        ),
    )
    assert version.version == "1.0.0"
    assert len(version.feature_names) > 0

    # 2. Prediction single
    row = feature_frame.iloc[0].to_dict()
    pred = p_svc.predict_single(
        version.id,
        PredictionRequest(features=row, subject_id="cust-101"),
    )
    assert pred.predicted_label in {0, 1}
    assert 0.0 <= pred.probability <= 1.0

    # 3. Explanation
    exp = e_svc.explain(
        version.id,
        ExplanationRequest(features=row, prediction_id=pred.id),
    )
    assert exp.additivity.passed is True
    assert exp.additivity.max_abs_error <= 1e-6
    assert exp.model_output is not None
    assert len(exp.contributions) == len(version.feature_names)
    assert len(exp.grouped_contributions) > 0

    # 4. Content-addressed explanation cache hit
    exp2 = e_svc.explain(
        version.id,
        ExplanationRequest(features=row, prediction_id=pred.id),
    )
    assert exp2.id == exp.id  # Same record retrieved from cache!

    # 5. Labeling and calibration
    label = l_svc.record_label(
        version.id,
        LabelCreate(ground_truth=1, prediction_id=pred.id),
    )
    assert label.ground_truth == 1

    metrics = l_svc.compute_calibration_metrics(version.id)
    assert metrics.total_labeled == 1
    assert metrics.accuracy in {0.0, 1.0}
    assert 0.0 <= metrics.brier_score <= 1.0

    # 6. Audit
    event = a_svc.record_event("test.action", "model", str(model.id), actor="tester")
    assert event.action == "test.action"
    events = a_svc.list_events()
    assert any(e.action == "test.action" for e in events)


def test_service_error_handling(
    service_session: Session,
    artifact_path: Path,
) -> None:
    m_svc = ModelService(service_session)
    p_svc = PredictionService(service_session)

    # Unknown model lookup
    with pytest.raises(NotFoundError):
        m_svc.get_model(uuid.uuid4())

    # Duplicate model creation
    m_svc.create_model(ModelCreate(name="dup-model"))
    with pytest.raises(ArtifactInvalidError, match="already exists"):
        m_svc.create_model(ModelCreate(name="dup-model"))

    # Empty batch prediction
    with pytest.raises(ValidationFailed, match="at least one item"):
        p_svc.predict_batch(uuid.uuid4(), BatchPredictionRequest.model_construct(items=[]))
