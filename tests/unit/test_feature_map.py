"""The feature map is what makes an explanation readable; it must be derived, not guessed."""

from __future__ import annotations

import pytest

from aletheia.core.errors import ArtifactInvalidError
from aletheia.ml.feature_map import (
    assert_names_match,
    derive_feature_map,
    humanize,
    strip_encoder_prefix,
)
from aletheia.ml.schema import FeatureSchema, FeatureSpec, group_contributions
from tests.support import CATEGORICAL, NUMERIC


def _schema(fitted_preprocessor) -> FeatureSchema:
    transformed = [str(name) for name in fitted_preprocessor.get_feature_names_out()]
    display = [strip_encoder_prefix(name) for name in transformed]
    return derive_feature_map(
        transformed, raw_columns=NUMERIC + CATEGORICAL, display_names=display
    )


def test_prefix_stripping_is_explicit() -> None:
    assert strip_encoder_prefix("num__tenure") == "tenure"
    assert strip_encoder_prefix("cat__Contract_One year") == "Contract_One year"
    assert strip_encoder_prefix("tenure") == "tenure"


def test_numeric_features_map_to_themselves(fitted_preprocessor) -> None:
    schema = _schema(fitted_preprocessor)
    specs = schema.by_name()
    assert specs["tenure"].source_column == "tenure"
    assert specs["tenure"].kind == "numeric"
    assert specs["tenure"].source_value is None


def test_one_hot_features_map_back_to_their_source_value(fitted_preprocessor) -> None:
    schema = _schema(fitted_preprocessor)
    specs = schema.by_name()
    month_to_month = specs["Contract_Month-to-month"]
    assert month_to_month.source_column == "Contract"
    assert month_to_month.kind == "one_hot_category"
    assert month_to_month.source_value == "Month-to-month"
    assert month_to_month.readable_label == "Month-to-month"


def test_every_feature_maps_to_a_known_source_column(fitted_preprocessor) -> None:
    schema = _schema(fitted_preprocessor)
    known = set(NUMERIC + CATEGORICAL)
    unmapped = [spec.name for spec in schema.features if spec.source_column not in known]
    assert unmapped == []


def test_longest_column_wins_a_prefix_collision() -> None:
    schema = derive_feature_map(
        ["cat__InternetService_Fibre optic", "cat__Internet_1"],
        raw_columns=["Internet", "InternetService"],
    )
    specs = schema.by_name()
    assert specs["InternetService_Fibre optic"].source_column == "InternetService"
    assert specs["Internet_1"].source_column == "Internet"


def test_grouping_preserves_the_total_contribution(fitted_preprocessor) -> None:
    schema = _schema(fitted_preprocessor)
    contributions = {spec.name: float(index) - 4.0 for index, spec in enumerate(schema.features)}
    grouped = group_contributions(contributions, schema)
    assert set(grouped) <= set(NUMERIC + CATEGORICAL)
    assert sum(grouped.values()) == pytest.approx(sum(contributions.values()), abs=1e-12)


def test_grouping_is_ordered_by_magnitude(fitted_preprocessor) -> None:
    schema = _schema(fitted_preprocessor)
    contributions = {spec.name: 0.0 for spec in schema.features}
    contributions["tenure"] = -2.0
    contributions["MonthlyCharges"] = 3.0
    grouped = group_contributions(contributions, schema)
    assert list(grouped)[:2] == ["MonthlyCharges", "tenure"]


def test_duplicate_feature_names_are_rejected() -> None:
    spec = FeatureSpec(name="tenure", source_column="tenure", kind="numeric")
    with pytest.raises(ArtifactInvalidError):
        FeatureSchema(features=(spec, spec))


def test_declared_names_must_match_positionally() -> None:
    with pytest.raises(ArtifactInvalidError, match="declares 2 feature names"):
        assert_names_match(["a", "b", "c"], ["a", "b"])
    with pytest.raises(ArtifactInvalidError, match="position 1"):
        assert_names_match(["a", "b"], ["a", "c"])


def test_prefix_mismatch_is_tolerated_but_reordering_is_not() -> None:
    assert_names_match(["num__tenure", "cat__Contract_One year"], ["tenure", "Contract_One year"])


def test_unmapped_feature_is_marked_other() -> None:
    schema = derive_feature_map(["num__tenure", "mystery_signal"], raw_columns=["tenure"])
    specs = schema.by_name()
    assert specs["mystery_signal"].kind == "other"
    assert specs["mystery_signal"].source_column == "mystery_signal"
    assert specs["mystery_signal"].source_value is None


def test_display_name_count_must_match() -> None:
    with pytest.raises(ArtifactInvalidError, match="display names"):
        derive_feature_map(["num__a", "num__b"], raw_columns=["a", "b"], display_names=["a"])


def test_humanize_is_deterministic() -> None:
    assert humanize("MonthlyCharges") == "MonthlyCharges"
    assert humanize("month_to_month") == "Month to month"
    assert humanize("") == ""
