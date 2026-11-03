"""FeatureSchema invariants and the grouped view's contract."""

from __future__ import annotations

import pytest

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.schema import FeatureSchema, FeatureSpec, group_contributions


def _specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="tenure", source_column="tenure", kind="numeric", readable_label="Tenure"),
        FeatureSpec(
            name="Contract_One year",
            source_column="Contract",
            kind="one_hot_category",
            source_value="One year",
            readable_label="One year",
        ),
        FeatureSpec(
            name="Contract_Two year",
            source_column="Contract",
            kind="one_hot_category",
            source_value="Two year",
            readable_label="Two year",
        ),
    )


def test_empty_schema_is_rejected() -> None:
    with pytest.raises(ArtifactInvalidError, match="empty"):
        FeatureSchema(features=())


def test_unnamed_feature_is_rejected() -> None:
    with pytest.raises(ArtifactInvalidError, match="unnamed"):
        FeatureSchema(features=(FeatureSpec(name="", source_column="tenure"),))


def test_length_and_lookup() -> None:
    schema = FeatureSchema(features=_specs(), source_columns=("tenure", "Contract"))
    assert len(schema) == 3
    assert schema.by_name()["tenure"].kind == "numeric"


def test_source_fields_are_deduplicated_in_order() -> None:
    schema = FeatureSchema(features=_specs())
    assert schema.source_fields() == ("tenure", "Contract")


def test_json_export_is_complete() -> None:
    schema = FeatureSchema(features=_specs())
    payload = schema.to_json()
    assert len(payload) == 3
    assert payload[1]["source_column"] == "Contract"
    assert payload[1]["source_value"] == "One year"
    assert payload[0]["source_value"] is None


def test_grouping_sums_one_hot_levels_onto_the_field() -> None:
    schema = FeatureSchema(features=_specs())
    contributions = {"tenure": 0.5, "Contract_One year": -0.2, "Contract_Two year": -0.3}
    grouped = group_contributions(contributions, schema)
    assert grouped["Contract"] == pytest.approx(-0.5)
    assert list(grouped) == ["Contract", "tenure"]


def test_grouping_ignores_names_outside_the_schema() -> None:
    schema = FeatureSchema(features=_specs())
    grouped = group_contributions({"tenure": 1.0, "not_a_feature": 99.0}, schema)
    assert grouped == {"tenure": 1.0}
