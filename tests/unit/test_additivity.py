"""Additivity is a gate, not a diagnostic: a failing explanation never leaves the function."""

from __future__ import annotations

import numpy as np
import pytest

from aletheia.core.errors import ExplanationFidelityError, ValidationFailed
from aletheia.ml.explain.additivity import require_additive, verify_additivity


def test_a_correct_decomposition_passes() -> None:
    base = -0.4
    phi = np.array([[0.2, -0.1, 0.3]])
    outputs = np.array([base + phi.sum()])
    report = verify_additivity(base, phi, outputs)
    assert report.passed is True
    assert report.rows_checked == 1
    assert report.max_abs_error < 1e-12


def test_a_small_drift_within_tolerance_passes() -> None:
    base = 0.0
    phi = np.array([[1.0, 1.0]])
    report = verify_additivity(base, phi, np.array([2.0000004]), tolerance=1e-6)
    assert report.passed is True


def test_a_real_drift_fails_and_blocks_persistence() -> None:
    base = 0.0
    phi = np.array([[1.0, 1.0]])
    report = verify_additivity(base, phi, np.array([2.5]), tolerance=1e-6)
    assert report.passed is False
    with pytest.raises(ExplanationFidelityError, match="additivity check failed"):
        require_additive(report, context="shap.LinearExplainer")


def test_row_count_mismatch_is_a_validation_error() -> None:
    with pytest.raises(ValidationFailed, match="do not match"):
        verify_additivity(0.0, np.zeros((2, 3)), np.zeros(1))


def test_non_finite_values_are_refused() -> None:
    phi = np.array([[0.1, np.nan]])
    with pytest.raises(ExplanationFidelityError, match="non-finite"):
        verify_additivity(0.0, phi, np.array([0.0]))


def test_one_dimensional_contributions_are_accepted() -> None:
    report = verify_additivity(1.0, np.array([0.5, -0.5]), np.array([1.0]))
    assert report.passed is True
    assert report.rows_checked == 1


def test_report_serialises_for_the_response() -> None:
    report = verify_additivity(0.0, np.zeros((1, 2)), np.zeros(1))
    payload = report.as_dict()
    assert payload["passed"] is True
    assert payload["tolerance"] == 1e-6
