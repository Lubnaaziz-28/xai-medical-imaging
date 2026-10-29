"""Metrics package for MedCard."""

from medcard.metrics.classification import (
    compute_all_metrics,
    compute_metrics_with_ci,
    compute_auc,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_sensitivity,
    compute_specificity,
    compute_f1,
    compute_confusion_matrix,
)
from medcard.metrics.fairness import (
    compute_fairness_metrics,
    compute_demographic_parity,
    compute_equalized_odds,
    compute_disparate_impact,
    subgroup_analysis,
)

__all__ = [
    "compute_all_metrics",
    "compute_metrics_with_ci",
    "compute_auc",
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_sensitivity",
    "compute_specificity",
    "compute_f1",
    "compute_confusion_matrix",
    "compute_fairness_metrics",
    "compute_demographic_parity",
    "compute_equalized_odds",
    "compute_disparate_impact",
    "subgroup_analysis",
]