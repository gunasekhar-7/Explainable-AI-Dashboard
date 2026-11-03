"""Additivity: the one property that separates a SHAP value from a plausible-looking number.

SHAP values are only meaningful if ``base_value + Σ φ`` reconstructs the model's own output. When that
identity fails, the explanation is not "slightly off" — it is describing something other than the model, so
it is refused rather than stored with a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aletheia.core.errors import ExplanationFidelityError, ValidationFailed

DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AdditivityReport:
    max_abs_error: float
    tolerance: float
    rows_checked: int
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "rows_checked": self.rows_checked,
            "passed": self.passed,
        }


def verify_additivity(
    base_value: float,
    contributions: np.ndarray,
    model_outputs: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AdditivityReport:
    """Check ``|base + sum(phi) - output| <= tolerance`` for each row.

    Returns a report instead of raising so callers can attach it to the response; ``require_additive``
    turns a failure into the error that blocks persistence.
    """
    phi = np.asarray(contributions, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    if phi.ndim != 2:
        raise ValidationFailed("contributions must be one- or two-dimensional")

    outputs = np.asarray(model_outputs, dtype=float).ravel()
    if phi.shape[0] != outputs.shape[0]:
        raise ValidationFailed(
            "contribution rows do not match model outputs",
            extra={"contribution_rows": int(phi.shape[0]), "output_rows": int(outputs.shape[0])},
        )
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(outputs)):
        raise ExplanationFidelityError(
            "non-finite values in the explanation; refusing to report it"
        )

    residual = np.abs((float(base_value) + phi.sum(axis=1)) - outputs)
    max_abs_error = float(residual.max()) if residual.size else 0.0
    return AdditivityReport(
        max_abs_error=max_abs_error,
        tolerance=float(tolerance),
        rows_checked=int(residual.size),
        passed=max_abs_error <= float(tolerance),
    )


def require_additive(report: AdditivityReport, *, context: str = "") -> None:
    """Raise when an explanation failed its own fidelity check."""
    if report.passed:
        return
    suffix = f" ({context})" if context else ""
    raise ExplanationFidelityError(
        "additivity check failed: max |base + sum(contributions) - model_output| = "
        f"{report.max_abs_error:.3e} exceeds tolerance {report.tolerance:.1e}{suffix}",
        extra=report.as_dict(),
    )
