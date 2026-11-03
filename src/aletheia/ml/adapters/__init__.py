"""Adapters translate an artifact into one governed object. One adapter per artifact family."""

from aletheia.ml.adapters.base import ArtifactMetadata, LoadedModel, ModelAdapter
from aletheia.ml.adapters.registry import (
    load_artifact,
    select_adapter,
    sha256_file,
    verify_artifact,
)
from aletheia.ml.adapters.sklearn_composite import SklearnCompositeAdapter

__all__ = [
    "ArtifactMetadata",
    "LoadedModel",
    "ModelAdapter",
    "SklearnCompositeAdapter",
    "load_artifact",
    "select_adapter",
    "sha256_file",
    "verify_artifact",
]
