"""Artifact loading entry point.

**Worker-side only.** ``joblib.load`` executes code embedded in the artifact, so ``load_artifact`` must be
called from the scrubbed explanation worker described in ``docs/03-system-architecture.md`` §5.1 — never
from the API process, and never once per request.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.adapters.base import LoadedModel, ModelAdapter
from aletheia.ml.adapters.sklearn_composite import SklearnCompositeAdapter

_ADAPTERS: tuple[ModelAdapter, ...] = (SklearnCompositeAdapter(),)
_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    """Stream the artifact through SHA-256, refusing anything over the size ceiling."""
    if not path.is_file():
        raise ArtifactInvalidError(f"artifact not found: {path.name}")
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ArtifactInvalidError(
            f"artifact is {size} bytes, over the {max_bytes}-byte limit",
            extra={"size_bytes": size, "limit_bytes": max_bytes},
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, expected_sha256: str, *, max_bytes: int | None = None) -> str:
    """Hash the artifact and require an exact match with the pinned value."""
    actual = sha256_file(path, max_bytes=max_bytes)
    if actual != expected_sha256:
        raise ArtifactInvalidError(
            "artifact SHA-256 does not match the pinned value",
            extra={"expected_prefix": expected_sha256[:12], "actual_prefix": actual[:12]},
        )
    return actual


def select_adapter(payload: object) -> ModelAdapter:
    for adapter in _ADAPTERS:
        if adapter.can_load(payload):
            return adapter
    raise ArtifactInvalidError(
        "no adapter recognises this artifact shape", extra={"adapters": [a.name for a in _ADAPTERS]}
    )


def load_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int | None = None,
) -> LoadedModel:
    digest = sha256_file(path, max_bytes=max_bytes)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArtifactInvalidError(
            "artifact SHA-256 does not match the pinned value",
            extra={"expected_prefix": expected_sha256[:12], "actual_prefix": digest[:12]},
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ArtifactInvalidError(f"artifact could not be read: {type(exc).__name__}") from exc
    adapter = select_adapter(payload)
    return adapter.load(payload, sha256=digest)
