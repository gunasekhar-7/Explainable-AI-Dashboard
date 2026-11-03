"""Unit tests for SQLAlchemy 2.0 database models and repositories."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aletheia.db.models import (
    Base,
)
from aletheia.db.repositories import (
    AuditRepository,
    ExplanationRepository,
    LabelRepository,
    ModelRepository,
    PredictionRepository,
)


@pytest.fixture
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_model_repository_crud(db_session: Session) -> None:
    repo = ModelRepository(db_session)
    model = repo.create(name="churn-detector", description="Telecom churn predictor")
    assert model.name == "churn-detector"
    assert model.task_type == "binary_classification"

    fetched = repo.get_by_name("churn-detector")
    assert fetched is not None
    assert fetched.id == model.id

    version = repo.create_version(
        model_id=model.id,
        version="1.0.0",
        artifact_uri="/path/to/artifact.joblib",
        artifact_sha256="a" * 64,
        adapter_name="sklearn-composite-v1",
        feature_names=["tenure", "MonthlyCharges"],
        source_columns=["tenure", "MonthlyCharges"],
        threshold=0.5,
    )
    assert version.version == "1.0.0"
    assert version.model_id == model.id

    active = repo.get_active_version(model.id)
    assert active is not None
    assert active.version == "1.0.0"


def test_prediction_and_label_repository(db_session: Session) -> None:
    m_repo = ModelRepository(db_session)
    p_repo = PredictionRepository(db_session)
    l_repo = LabelRepository(db_session)

    model = m_repo.create(name="risk-model")
    version = m_repo.create_version(
        model_id=model.id,
        version="1.0.0",
        artifact_uri="dummy.joblib",
        artifact_sha256="b" * 64,
        adapter_name="sklearn-composite-v1",
    )

    pred = p_repo.create(
        model_version_id=version.id,
        input_features={"tenure": 12, "MonthlyCharges": 70.0},
        probability=0.75,
        predicted_label=1,
        threshold=0.5,
        subject_ref_hash="subj-123",
    )
    assert pred.probability == 0.75
    assert pred.predicted_label == 1

    label = l_repo.record_label(
        prediction_id=pred.id,
        ground_truth=1,
        label_source="billing",
    )
    assert label.ground_truth == 1

    pairs = l_repo.list_labeled_predictions(version.id)
    assert len(pairs) == 1
    assert pairs[0][0].id == pred.id
    assert pairs[0][1].ground_truth == 1


def test_explanation_repository_caching(db_session: Session) -> None:
    m_repo = ModelRepository(db_session)
    e_repo = ExplanationRepository(db_session)

    model = m_repo.create(name="exp-model")
    version = m_repo.create_version(
        model_id=model.id,
        version="1.0.0",
        artifact_uri="dummy.joblib",
        artifact_sha256="c" * 64,
        adapter_name="sklearn-composite-v1",
    )

    cache_key = "hash-key-abc"
    e_repo.create(
        model_version_id=version.id,
        prediction_id=None,
        explainer_type="shap.LinearExplainer",
        model_output=-1.25,
        base_value=0.1,
        output_space="log_odds",
        contributions={"tenure": -0.8, "MonthlyCharges": -0.55},
        grouped_contributions={"tenure": -0.8, "MonthlyCharges": -0.55},
        fidelity_passed=True,
        max_abs_error=1e-15,
        tolerance=1e-6,
        duration_ms=45,
        caveats=["Not causal"],
        cache_key=cache_key,
    )

    cached = e_repo.get_by_cache_key(cache_key)
    assert cached is not None
    assert cached.model_output == -1.25
    assert cached.fidelity_passed is True


def test_audit_repository(db_session: Session) -> None:
    repo = AuditRepository(db_session)
    event = repo.record(
        action="model.deployed",
        resource_type="model",
        resource_id="mod-1",
        actor="lead-engineer",
        details={"environment": "production"},
    )
    assert event.action == "model.deployed"

    events = repo.list_events()
    assert len(events) >= 1
    assert events[0].actor == "lead-engineer"
