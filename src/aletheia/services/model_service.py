"""Model registry and artifact lifecycle service."""

from __future__ import annotations

import platform
import threading
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from aletheia.core.config import get_settings
from aletheia.core.errors import ArtifactInvalidError, NotFoundError
from aletheia.db.models import ModelVersionRecord
from aletheia.db.repositories import ModelRepository
from aletheia.ml.adapters import LoadedModel, load_artifact, sha256_file
from aletheia.schemas.models import (
    ModelCreate,
    ModelDetailResponse,
    ModelResponse,
    ModelVersionCreate,
    ModelVersionResponse,
)

_MODEL_CACHE: dict[uuid.UUID, LoadedModel] = {}
_CACHE_LOCK = threading.Lock()


class ModelService:
    """Service governing registered models and artifact versions."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = ModelRepository(session)
        self.settings = get_settings()

    def create_model(self, data: ModelCreate) -> ModelResponse:
        existing = self.repo.get_by_name(data.name)
        if existing is not None:
            raise ArtifactInvalidError(f"model with name '{data.name}' already exists")
        record = self.repo.create(
            name=data.name,
            task_type=data.task_type,
            description=data.description,
        )
        return ModelResponse(
            id=record.id,
            name=record.name,
            task_type=record.task_type,
            description=record.description,
            created_at=record.created_at,
            versions_count=0,
        )

    def list_models(self) -> list[ModelResponse]:
        records = self.repo.list_all()
        return [
            ModelResponse(
                id=rec.id,
                name=rec.name,
                task_type=rec.task_type,
                description=rec.description,
                created_at=rec.created_at,
                versions_count=len(rec.versions),
            )
            for rec in records
        ]

    def get_model(self, model_id: uuid.UUID) -> ModelDetailResponse:
        record = self.repo.get_by_id(model_id)
        if record is None:
            raise NotFoundError(f"model '{model_id}' not found")
        versions = [
            ModelVersionResponse(
                id=v.id,
                model_id=v.model_id,
                version=v.version,
                status=v.status,
                artifact_uri=v.artifact_uri,
                artifact_sha256=v.artifact_sha256,
                adapter_name=v.adapter_name,
                framework=v.framework,
                framework_version=v.framework_version,
                python_version=v.python_version,
                threshold=v.threshold,
                feature_names=list(v.feature_names),
                source_columns=list(v.source_columns),
                metadata=dict(v.metadata_json),
                created_at=v.created_at,
            )
            for v in record.versions
        ]
        return ModelDetailResponse(
            id=record.id,
            name=record.name,
            task_type=record.task_type,
            description=record.description,
            created_at=record.created_at,
            versions=versions,
        )

    def register_version(
        self, model_id: uuid.UUID, data: ModelVersionCreate
    ) -> ModelVersionResponse:
        model = self.repo.get_by_id(model_id)
        if model is None:
            raise NotFoundError(f"model '{model_id}' not found")

        artifact_path = Path(data.artifact_uri)
        if not artifact_path.is_file():
            # Check relative to artifact root
            candidate = self.settings.artifact_root / data.artifact_uri
            if candidate.is_file():
                artifact_path = candidate
            else:
                raise NotFoundError(f"artifact file not found at {data.artifact_uri}")

        actual_sha = sha256_file(artifact_path, max_bytes=self.settings.max_artifact_bytes)
        if data.expected_sha256 and actual_sha != data.expected_sha256:
            raise ArtifactInvalidError(
                "artifact SHA-256 mismatch with expected value",
                extra={"expected": data.expected_sha256, "actual": actual_sha},
            )

        loaded = load_artifact(
            artifact_path,
            expected_sha256=actual_sha,
            max_bytes=self.settings.max_artifact_bytes,
        )

        metadata_dict = dict(loaded.metadata.notes)
        metadata_dict["library_versions"] = dict(loaded.metadata.library_versions)

        version_record = self.repo.create_version(
            model_id=model_id,
            version=data.version,
            artifact_uri=str(artifact_path),
            artifact_sha256=actual_sha,
            adapter_name=loaded.adapter,
            framework=data.framework,
            framework_version=loaded.metadata.library_versions.get("scikit-learn"),
            python_version=platform.python_version(),
            threshold=loaded.metadata.threshold,
            feature_names=list(loaded.feature_names),
            source_columns=list(loaded.schema.source_columns),
            metadata_json=metadata_dict,
            status=data.status,
        )

        with _CACHE_LOCK:
            _MODEL_CACHE[version_record.id] = loaded

        return ModelVersionResponse(
            id=version_record.id,
            model_id=version_record.model_id,
            version=version_record.version,
            status=version_record.status,
            artifact_uri=version_record.artifact_uri,
            artifact_sha256=version_record.artifact_sha256,
            adapter_name=version_record.adapter_name,
            framework=version_record.framework,
            framework_version=version_record.framework_version,
            python_version=version_record.python_version,
            threshold=version_record.threshold,
            feature_names=list(version_record.feature_names),
            source_columns=list(version_record.source_columns),
            metadata=dict(version_record.metadata_json),
            created_at=version_record.created_at,
        )

    def get_loaded_model(self, version_id: uuid.UUID) -> tuple[ModelVersionRecord, LoadedModel]:
        with _CACHE_LOCK:
            if version_id in _MODEL_CACHE:
                version_record = self.repo.get_version_by_id(version_id)
                if version_record is None:
                    raise NotFoundError(f"version '{version_id}' not found")
                return version_record, _MODEL_CACHE[version_id]

        version_record = self.repo.get_version_by_id(version_id)
        if version_record is None:
            raise NotFoundError(f"version '{version_id}' not found")

        path = Path(version_record.artifact_uri)
        loaded = load_artifact(
            path,
            expected_sha256=version_record.artifact_sha256,
            max_bytes=self.settings.max_artifact_bytes,
        )
        with _CACHE_LOCK:
            _MODEL_CACHE[version_id] = loaded
        return version_record, loaded
