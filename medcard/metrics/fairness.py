"""Fairness and bias analysis metrics for medical imaging models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def compute_demographic_parity(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.5,
    privileged_value: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute demographic parity difference.

    Args:
        y_pred: Predicted probabilities
        sensitive_attr: Sensitive attribute values
        threshold: Decision threshold
        privileged_value: Value of privileged group (auto-detect if None)

    Returns:
        Dictionary with demographic parity metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    unique_groups = np.unique(sensitive_attr)

    if privileged_value is None:
        # Assume first group is privileged
        privileged_value = unique_groups[0]

    group_rates = {}
    for group in unique_groups:
        mask = sensitive_attr == group
        if mask.sum() > 0:
            group_rates[str(group)] = y_pred_binary[mask].mean()

    privileged_rate = group_rates.get(str(privileged_value), 0)
    unprivileged_rates = {k: v for k, v in group_rates.items() if k != str(privileged_value)}

    max_diff = 0
    for group, rate in unprivileged_rates.items():
        diff = abs(privileged_rate - rate)
        max_diff = max(max_diff, diff)

    return {
        "group_positive_rates": group_rates,
        "privileged_group": str(privileged_value),
        "privileged_rate": privileged_rate,
        "demographic_parity_difference": max_diff,
        "demographic_parity_ratio": (
            min(unprivileged_rates.values()) / privileged_rate if privileged_rate > 0 else 0
        ),
    }


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.5,
    privileged_value: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute equalized odds difference (TPR and FPR parity).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        sensitive_attr: Sensitive attribute values
        threshold: Decision threshold
        privileged_value: Value of privileged group

    Returns:
        Dictionary with equalized odds metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    unique_groups = np.unique(sensitive_attr)

    if privileged_value is None:
        privileged_value = unique_groups[0]

    group_metrics = {}
    for group in unique_groups:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred_binary[mask]

        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix_manual(yt, yp)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        group_metrics[str(group)] = {"tpr": tpr, "fpr": fpr, "support": int(mask.sum())}

    privileged_tpr = group_metrics.get(str(privileged_value), {}).get("tpr", 0)
    privileged_fpr = group_metrics.get(str(privileged_value), {}).get("fpr", 0)

    tpr_diffs = []
    fpr_diffs = []
    for group, metrics in group_metrics.items():
        if group != str(privileged_value):
            tpr_diffs.append(abs(privileged_tpr - metrics["tpr"]))
            fpr_diffs.append(abs(privileged_fpr - metrics["fpr"]))

    return {
        "group_metrics": group_metrics,
        "privileged_group": str(privileged_value),
        "equalized_odds_tpr_diff": max(tpr_diffs) if tpr_diffs else 0,
        "equalized_odds_fpr_diff": max(fpr_diffs) if fpr_diffs else 0,
        "max_equalized_odds_diff": max(max(tpr_diffs) if tpr_diffs else 0, max(fpr_diffs) if fpr_diffs else 0),
    }


def compute_disparate_impact(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold: float = 0.5,
    privileged_value: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute disparate impact ratio (4/5ths rule).

    Args:
        y_pred: Predicted probabilities
        sensitive_attr: Sensitive attribute values
        threshold: Decision threshold
        privileged_value: Value of privileged group

    Returns:
        Dictionary with disparate impact metrics
    """
    dp = compute_demographic_parity(y_pred, sensitive_attr, threshold, privileged_value)
    ratio = dp["demographic_parity_ratio"]

    return {
        "disparate_impact_ratio": ratio,
        "passes_four_fifths_rule": ratio >= 0.8,
        "privileged_group": dp["privileged_group"],
        "group_positive_rates": dp["group_positive_rates"],
    }


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Manual confusion matrix calculation."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp


def subgroup_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attrs: Dict[str, np.ndarray],
    threshold: float = 0.5,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Perform subgroup analysis across multiple sensitive attributes.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        sensitive_attrs: Dictionary of attribute_name -> attribute_values
        threshold: Decision threshold
        metrics: List of metrics to compute (default: all)

    Returns:
        Dictionary with per-subgroup metrics
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc", "sensitivity", "specificity"]

    y_pred_binary = (y_pred >= threshold).astype(int)
    results = {}

    for attr_name, attr_values in sensitive_attrs.items():
        results[attr_name] = {}
        unique_groups = np.unique(attr_values)

        for group in unique_groups:
            mask = attr_values == group
            if mask.sum() == 0:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            yp_bin = y_pred_binary[mask]

            group_results = {"support": int(mask.sum()), "prevalence": float(yt.mean())}

            if "accuracy" in metrics:
                group_results["accuracy"] = float(accuracy_score(yt, yp_bin))
            if "precision" in metrics:
                group_results["precision"] = float(
                    precision_score(yt, yp_bin, zero_division=0)
                )
            if "recall" in metrics:
                group_results["recall"] = float(recall_score(yt, yp_bin, zero_division=0))
            if "f1" in metrics:
                group_results["f1"] = float(f1_score(yt, yp_bin, zero_division=0))
            if "sensitivity" in metrics:
                group_results["sensitivity"] = float(recall_score(yt, yp_bin, zero_division=0))
            if "specificity" in metrics:
                tn, fp, fn, tp = confusion_matrix_manual(yt, yp_bin)
                group_results["specificity"] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
            if "auc" in metrics:
                try:
                    group_results["auc"] = float(roc_auc_score(yt, yp))
                except ValueError:
                    group_results["auc"] = None

            results[attr_name][str(group)] = group_results

    return results


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attrs: Dict[str, np.ndarray],
    threshold: float = 0.5,
    privileged_groups: Optional[Dict[str, Union[int, str]]] = None,
) -> Dict:
    """
    Compute comprehensive fairness metrics across all sensitive attributes.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        sensitive_attrs: Dictionary of attribute_name -> attribute_values
        threshold: Decision threshold
        privileged_groups: Dictionary of attribute_name -> privileged_value

    Returns:
        Comprehensive fairness report
    """
    if privileged_groups is None:
        privileged_groups = {}

    fairness_report = {
        "demographic_parity": {},
        "equalized_odds": {},
        "disparate_impact": {},
        "subgroup_analysis": subgroup_analysis(y_true, y_pred, sensitive_attrs, threshold),
        "overall_fairness_score": 0.0,
    }

    max_dp_diff = 0
    max_eo_diff = 0
    di_violations = 0
    total_attrs = len(sensitive_attrs)

    for attr_name, attr_values in sensitive_attrs.items():
        privileged = privileged_groups.get(attr_name)

        dp = compute_demographic_parity(y_pred, attr_values, threshold, privileged)
        fairness_report["demographic_parity"][attr_name] = dp
        max_dp_diff = max(max_dp_diff, dp["demographic_parity_difference"])

        eo = compute_equalized_odds(y_true, y_pred, attr_values, threshold, privileged)
        fairness_report["equalized_odds"][attr_name] = eo
        max_eo_diff = max(max_eo_diff, eo["max_equalized_odds_diff"])

        di = compute_disparate_impact(y_pred, attr_values, threshold, privileged)
        fairness_report["disparate_impact"][attr_name] = di
        if not di["passes_four_fifths_rule"]:
            di_violations += 1

    # Overall fairness score (0-1, higher is fairer)
    fairness_report["overall_fairness_score"] = 1.0 - (
        0.4 * max_dp_diff + 0.4 * max_eo_diff + 0.2 * (di_violations / total_attrs if total_attrs > 0 else 0)
    )
    fairness_report["overall_fairness_score"] = max(0.0, fairness_report["overall_fairness_score"])

    fairness_report["summary"] = {
        "max_demographic_parity_diff": max_dp_diff,
        "max_equalized_odds_diff": max_eo_diff,
        "disparate_impact_violations": di_violations,
        "total_attributes_analyzed": total_attrs,
    }

    return fairness_report