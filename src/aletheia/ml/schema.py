"""Feature identity: what the model consumes, and what the user is shown.

The gap between these two is where explainability usually goes wrong. A one-hot encoded column named
``Contract_Month-to-month`` is not a feature a customer can act on; ``Contract = Month-to-month`` is. Every
explanation therefore carries a map from transformed name back to source column, and the grouped view is
labelled as an aggregation of attributions rather than a different model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from aletheia.core.errors import ArtifactInvalidError

FeatureKind = Literal["numeric", "one_hot_category", "other"]


@dataclass(frozen=True)
class FeatureSpec:
    """One transformed feature and the raw field it came from."""

    name: str
    source_column: str
    kind: FeatureKind = "other"
    source_value: str | None = None
    readable_label: str = ""


@dataclass(frozen=True)
class FeatureSchema:
    """Every transformed feature, in model column order."""

    features: tuple[FeatureSpec, ...]
    source_columns: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.features:
            raise ArtifactInvalidError("feature schema is empty")
        counts = Counter(spec.name for spec in self.features)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ArtifactInvalidError(
                f"duplicate feature names in schema: {duplicates[:5]}", extra={"duplicates": duplicates}
            )
        unnamed = [index for index, spec in enumerate(self.features) if not spec.name]
        if unnamed:
            raise ArtifactInvalidError(f"unnamed features at positions {unnamed[:5]}")

    def __len__(self) -> int:
        return len(self.features)

    def by_name(self) -> dict[str, FeatureSpec]:
        return {spec.name: spec for spec in self.features}

    def source_fields(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for spec in self.features:
            seen.setdefault(spec.source_column, None)
        return tuple(seen)

    def to_json(self) -> list[dict[str, str | None]]:
        return [
            {
                "name": spec.name,
                "source_column": spec.source_column,
                "kind": spec.kind,
                "source_value": spec.source_value,
                "readable_label": spec.readable_label,
            }
            for spec in self.features
        ]


def group_contributions(
    contributions: Mapping[str, float], schema: FeatureSchema
) -> dict[str, float]:
    """Sum transformed contributions back onto their source field.

    One-hot expansion otherwise shows a single decision split across several rows, which reads as several
    weak effects instead of one strong one. This is an aggregation of the same attributions, not a
    re-attribution, and the UI must say so.
    """
    grouped: dict[str, float] = {}
    for spec in schema.features:
        value = contributions.get(spec.name)
        if value is None:
            continue
        grouped[spec.source_column] = grouped.get(spec.source_column, 0.0) + float(value)
    return dict(sorted(grouped.items(), key=lambda item: (-abs(item[1]), item[0])))
