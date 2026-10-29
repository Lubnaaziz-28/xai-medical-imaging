"""Classification metrics for medical imaging models."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")


def compute_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multi_class: str = "ovr",
    average: str = "macro",
) -> float:
    """Compute AUC-ROC score."""
    return roc_auc_score(y_true, y_pred, multi_class=multi_class, average=average)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute accuracy score."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "binary",
    zero_division: int = 0,
) -> float:
    """Compute precision score."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    return precision_score(y_true, y_pred_binary, average=average, zero_division=zero_division)


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "binary",
    zero_division: int = 0,
) -> float:
    """Compute recall (sensitivity) score."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    return recall_score(y_true, y_pred_binary, average=average, zero_division=zero_division)


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute sensitivity (recall for positive class)."""
    return compute_recall(y_true, y_pred, threshold=threshold, average="binary")


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute specificity (true negative rate)."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "binary",
    zero_division: int = 0,
) -> float:
    """Compute F1 score."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average=average, zero_division=zero_division)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Compute confusion matrix."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred_binary)


def compute_roc_curve(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve points."""
    return roc_curve(y_true, y_pred)


def compute_pr_curve(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve points."""
    return precision_recall_curve(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute all classification metrics.

    Args:
        y_true: Ground truth labels (N,) or (N, C) for multi-class
        y_pred: Predicted probabilities (N,) or (N, C)
        threshold: Decision threshold for binary classification
        class_names: Optional class names for multi-class

    Returns:
        Dictionary with all metrics
    """
    # Handle multi-class
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        return _compute_multiclass_metrics(y_true, y_pred, class_names)

    # Binary classification
    y_pred_binary = (y_pred >= threshold).astype(int)

    metrics = {
        "auc": compute_auc(y_true, y_pred),
        "accuracy": compute_accuracy(y_true, y_pred, threshold),
        "precision": compute_precision(y_true, y_pred, threshold),
        "recall": compute_recall(y_true, y_pred, threshold),
        "sensitivity": compute_sensitivity(y_true, y_pred, threshold),
        "specificity": compute_specificity(y_true, y_pred, threshold),
        "f1": compute_f1(y_true, y_pred, threshold),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, threshold).tolist(),
        "threshold": threshold,
    }

    # Add per-class metrics if multi-label
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        metrics["per_class"] = {}
        for i in range(y_true.shape[1]):
            name = class_names[i] if class_names else f"class_{i}"
            metrics["per_class"][name] = compute_all_metrics(
                y_true[:, i], y_pred[:, i], threshold
            )

    return metrics


def _compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute metrics for multi-class classification."""
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

    metrics = {
        "auc": compute_auc(y_true, y_pred, multi_class="ovr", average="macro"),
        "accuracy": accuracy_score(y_true_classes, y_pred_classes),
        "precision_macro": precision_score(y_true_classes, y_pred_classes, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_classes, y_pred_classes, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_classes, y_pred_classes, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true_classes, y_pred_classes, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true_classes, y_pred_classes, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_classes, y_pred_classes, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_classes, y_pred_classes).tolist(),
        "classification_report": classification_report(
            y_true_classes, y_pred_classes, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    # Per-class AUC
    if class_names:
        metrics["per_class_auc"] = {}
        for i, name in enumerate(class_names):
            try:
                metrics["per_class_auc"][name] = roc_auc_score(
                    (y_true_classes == i).astype(int), y_pred[:, i]
                )
            except ValueError:
                metrics["per_class_auc"][name] = None

    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for a metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        metric_fn: Function to compute metric (y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)
        threshold: Decision threshold
        random_state: Random seed

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue

    if not bootstrap_scores:
        return (np.nan, np.nan)

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    return (float(lower), float(upper))


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.

    Returns:
        Dictionary with metric names mapping to {value, ci_lower, ci_upper}
    """
    metrics = {}

    # Define metric functions
    metric_funcs = {
        "auc": lambda yt, yp: compute_auc(yt, yp),
        "accuracy": lambda yt, yp: compute_accuracy(yt, yp, threshold),
        "precision": lambda yt, yp: compute_precision(yt, yp, threshold),
        "recall": lambda yt, yp: compute_recall(yt, yp, threshold),
        "sensitivity": lambda yt, yp: compute_sensitivity(yt, yp, threshold),
        "specificity": lambda yt, yp: compute_specificity(yt, yp, threshold),
        "f1": lambda yt, yp: compute_f1(yt, yp, threshold),
    }

    for name, fn in metric_funcs.items():
        value = fn(y_true, y_pred)
        ci_lower, ci_upper = bootstrap_confidence_intervals(
            y_true, y_pred, fn, n_bootstrap, confidence, threshold
        )
        metrics[name] = {
            "value": float(value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    return metrics