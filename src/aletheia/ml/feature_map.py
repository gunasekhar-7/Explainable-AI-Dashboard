"""Deriving the feature map from a fitted ``ColumnTransformer``.

The upstream churn artifact declares 46 feature names *without* the ``num__``/``cat__`` prefixes that
scikit-learn's ``get_feature_names_out`` emits. That quirk is real and was measured in Stage 0, so the
mapping is derived from the transformer (authoritative) and then checked against the declared names
(untrusted). A mismatch is a rejection with the offending position, not a silent guess.
"""

from __future__ import annotations

from collections.abc import Sequence

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.schema import FeatureSchema, FeatureSpec

_ENCODER_PREFIXES = (
    "num__",
    "cat__",
    "ordinal__",
    "bool__",
    "remainder__",
    "pipeline-",
)


def strip_encoder_prefix(name: str) -> str:
    """Remove a scikit-learn transformer prefix, if present."""
    for prefix in _ENCODER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def humanize(name: str) -> str:
    """Deterministic display label. No model calls, no locale dependence."""
    cleaned = name.replace("_", " ").strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def assert_names_match(expected: Sequence[str], provided: Sequence[str]) -> None:
    """Check declared names against what the preprocessor actually produces.

    Comparison is prefix-insensitive because the known-good artifact omits prefixes; order is not
    negotiable, because contributions are positional.
    """
    if len(expected) != len(provided):
        raise ArtifactInvalidError(
            f"artifact declares {len(provided)} feature names but the preprocessor produces {len(expected)}"
        )
    for position, (want, got) in enumerate(zip(expected, provided, strict=True)):
        if strip_encoder_prefix(want) != strip_encoder_prefix(got):
            raise ArtifactInvalidError(
                f"feature name mismatch at position {position}: "
                f"preprocessor produces {strip_encoder_prefix(want)!r}, artifact declares {got!r}"
            )


def derive_feature_map(
    transformed_names: Sequence[str],
    *,
    raw_columns: Sequence[str],
    display_names: Sequence[str] | None = None,
) -> FeatureSchema:
    """Build the transformed-name → source-column map.

    ``transformed_names`` must be the preprocessor's own output names (prefixed). ``display_names`` is what
    the artifact declares and what responses expose.
    """
    if display_names is not None and len(display_names) != len(transformed_names):
        raise ArtifactInvalidError(
            f"{len(display_names)} display names for {len(transformed_names)} transformed features"
        )

    raw = list(dict.fromkeys(raw_columns))
    # Longest first: a column named "Internet" must not claim "InternetService_DSL".
    ordered_raw = sorted(raw, key=len, reverse=True)

    specs: list[FeatureSpec] = []
    for position, transformed in enumerate(transformed_names):
        core = strip_encoder_prefix(transformed)
        display = (
            display_names[position] if display_names is not None else core
        )
        if core in raw:
            specs.append(
                FeatureSpec(
                    name=display,
                    source_column=core,
                    kind="numeric",
                    readable_label=humanize(core),
                )
            )
            continue
        source = next((column for column in ordered_raw if core.startswith(f"{column}_")), None)
        if source is None:
            specs.append(
                FeatureSpec(
                    name=display,
                    source_column=core,
                    kind="other",
                    readable_label=humanize(core),
                )
            )
            continue
        sources_value = core[len(source) + 1 :].replace("_", " ")
        specs.append(
            FeatureSpec(
                name=display,
                source_column=source,
                kind="one_hot_category",
                source_value=sources_value,
                readable_label=humanize(sources_value),
            )
        )

    return FeatureSchema(features=tuple(specs), source_columns=tuple(raw))
