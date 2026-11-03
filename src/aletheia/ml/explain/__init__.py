"""Explainers and the fidelity checks that gate their output."""

from aletheia.ml.explain.additivity import AdditivityReport, require_additive, verify_additivity
from aletheia.ml.explain.shap_local import LocalExplanation, explain_local

__all__ = [
    "AdditivityReport",
    "LocalExplanation",
    "explain_local",
    "require_additive",
    "verify_additivity",
]
