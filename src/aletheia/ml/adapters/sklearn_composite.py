"""Adapter for the composite artifact shape ``{model, preprocessor, feature_names, metadata}``.

This is the shape the existing churn artifact uses, verified by probing it in Stage 0. Supporting it
directly is what lets a model trained elsewhere be governed without retraining it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.adapters.base import ArtifactMetadata, LoadedModel, OutputSpace
from aletheia.ml.feature_map import assert_names_match, derive_feature_map, strip_encoder_prefix
from aletheia.ml.schema import FeatureSchema, FeatureSpec

_REQUIRED_KEYS = ("model",)
_OPTIONAL_KEYS = ("preprocessor", "feature_names", "metadata")


def _raw_columns(preprocessor: Any) -> tuple[str, ...]:
    """Recover the raw input columns a fitted preprocessor expects."""
    feature_names_in = getattr(preprocessor, "feature_names_in_", None)
    if feature_names_in is not None:
        return tuple(str(name) for name in feature_names_in)

    columns: list[str] = []
    for _name, _transformer, raw in getattr(preprocessor, "transformers_", ()) or ():
        if raw is None:
            continue
        if isinstance(raw, str):
            columns.append(raw)
            continue
        for column in raw:
            if isinstance(column, int):
                continue
            columns.append(str(column))
    return tuple(dict.fromkeys(columns))


class SklearnCompositeAdapter:
    """Loads ``{model, preprocessor, feature_names, metadata}`` artifacts."""

    name = "sklearn-composite-v1"

    def can_load(self, payload: object) -> bool:
        return isinstance(payload, Mapping) and any(key in payload for key in _REQUIRED_KEYS)

    def load(self, payload: object, *, sha256: str) -> LoadedModel:
        if not isinstance(payload, Mapping):
            raise ArtifactInvalidError(
                "artifact is not a mapping; expected the composite shape", extra={"adapter": self.name}
            )
        unknown = sorted(set(payload) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
        estimator = payload.get("model")
        if estimator is None:
            raise ArtifactInvalidError("artifact has no 'model' entry")
        if not hasattr(estimator, "predict") and not hasattr(estimator, "predict_proba"):
            raise ArtifactInvalidError(
                "'model' is not an estimator: it exposes neither predict nor predict_proba"
            )

        preprocessor = payload.get("preprocessor")
        declared = payload.get("feature_names") or ()
        if isinstance(declared, str) or not isinstance(declared, Sequence):
            raise ArtifactInvalidError("'feature_names' must be a sequence of strings")
        declared_names = tuple(str(name) for name in declared)

        metadata_raw = payload.get("metadata") or {}
        if not isinstance(metadata_raw, Mapping):
            raise ArtifactInvalidError("'metadata' must be a mapping")

        threshold = self._threshold(metadata_raw.get("threshold"))
        names, schema = self._resolve_features(preprocessor, declared_names)
        library_versions = self._library_versions(metadata_raw.get("library_versions"))
        output_space: OutputSpace = (
            "log_odds" if hasattr(estimator, "decision_function") else "probability"
        )

        return LoadedModel(
            artifact_sha256=sha256,
            adapter=self.name,
            estimator=estimator,
            preprocessor=preprocessor,
            feature_names=names,
            schema=schema,
            metadata=ArtifactMetadata(
                artifact_sha256=sha256,
                adapter=self.name,
                library_versions=library_versions,
                threshold=threshold,
                notes={
                    "declared_metadata_keys": sorted(str(key) for key in metadata_raw),
                    "undeclared_keys": unknown,
                    "raw_columns": list(schema.source_columns),
                },
            ),
            output_space=output_space,
        )

    @staticmethod
    def _threshold(value: Any) -> float | None:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ArtifactInvalidError(f"threshold is not numeric: {value!r}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ArtifactInvalidError(f"threshold {threshold} is outside [0, 1]")
        return threshold

    @staticmethod
    def _library_versions(value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ArtifactInvalidError("'metadata.library_versions' must be a mapping")
        return {str(key): str(version) for key, version in value.items()}

    def _resolve_features(
        self, preprocessor: Any, declared_names: tuple[str, ...]
    ) -> tuple[tuple[str, ...], FeatureSchema]:
        if preprocessor is None:
            if not declared_names:
                raise ArtifactInvalidError(
                    "artifact has neither a preprocessor nor declared feature names; "
                    "feature identity cannot be established"
                )
            schema = FeatureSchema(
                features=tuple(
                    FeatureSpec(name=name, source_column=name, kind="other")
                    for name in declared_names
                ),
                source_columns=declared_names,
            )
            return declared_names, schema

        get_names = getattr(preprocessor, "get_feature_names_out", None)
        if get_names is None:
            raise ArtifactInvalidError(
                "preprocessor does not expose get_feature_names_out(); "
                "feature identity cannot be established"
            )
        try:
            transformed = tuple(str(name) for name in get_names())
        except Exception as exc:
            raise ArtifactInvalidError(
                "preprocessor could not report its output feature names"
            ) from exc

        if not declared_names:
            raise ArtifactInvalidError(
                "artifact declares no feature_names; refusing to guess them from the preprocessor"
            )
        assert_names_match(transformed, declared_names)

        display = tuple(strip_encoder_prefix(name) for name in declared_names)
        schema = derive_feature_map(
            transformed,
            raw_columns=_raw_columns(preprocessor),
            display_names=display,
        )
        return display, schema
