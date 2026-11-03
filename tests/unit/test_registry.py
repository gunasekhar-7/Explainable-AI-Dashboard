"""Artifact integrity: hashing, pinning, and size limits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.adapters import load_artifact, sha256_file, verify_artifact


def test_hash_matches_a_direct_digest(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"aletheia" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactInvalidError, match="not found"):
        sha256_file(tmp_path / "absent.joblib")


def test_size_ceiling_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 2048)
    with pytest.raises(ArtifactInvalidError, match="over the"):
        sha256_file(path, max_bytes=1024)


def test_pinned_hash_must_match(artifact_path: Path) -> None:
    actual = sha256_file(artifact_path)
    assert verify_artifact(artifact_path, actual) == actual
    with pytest.raises(ArtifactInvalidError, match="does not match"):
        verify_artifact(artifact_path, "0" * 64)


def test_mismatch_reports_prefixes_not_the_full_hash(artifact_path: Path) -> None:
    with pytest.raises(ArtifactInvalidError) as excinfo:
        verify_artifact(artifact_path, "f" * 64)
    extra = excinfo.value.extra
    assert len(extra["expected_prefix"]) == 12
    assert len(extra["actual_prefix"]) == 12


def test_load_artifact_enforces_the_pin(artifact_path: Path) -> None:
    with pytest.raises(ArtifactInvalidError, match="does not match"):
        load_artifact(artifact_path, expected_sha256="0" * 64)
    digest = sha256_file(artifact_path)
    model = load_artifact(artifact_path, expected_sha256=digest)
    assert model.artifact_sha256 == digest


def test_unreadable_artifact_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "garbage.joblib"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(ArtifactInvalidError, match="could not be read"):
        load_artifact(path)
