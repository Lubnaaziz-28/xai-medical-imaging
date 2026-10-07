import numpy as np


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    if np.std(a_flat) < 1e-8 and np.std(b_flat) < 1e-8:
        return 1.0
    if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
        return 0.0
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.8) -> float:
    a_bin = (a >= np.percentile(a, threshold * 100)).astype(np.uint8)
    b_bin = (b >= np.percentile(b, threshold * 100)).astype(np.uint8)
    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return _ssim_fallback(a, b)
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
    ssim_val, _ = structural_similarity(a_norm, b_norm, data_range=1.0, full=True)
    return float(np.clip(ssim_val, 0.0, 1.0))


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
    mu_a = np.mean(a_norm)
    mu_b = np.mean(b_norm)
    sigma_a = np.var(a_norm)
    sigma_b = np.var(b_norm)
    sigma_ab = np.mean((a_norm - mu_a) * (b_norm - mu_b))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.clip(num / (den + 1e-8), 0.0, 1.0))


def compute_metrics(gradcam: np.ndarray, shap: np.ndarray, ig: np.ndarray) -> dict[str, float]:
    corr_gc_shap = _pearson_correlation(gradcam, shap)
    corr_gc_ig = _pearson_correlation(gradcam, ig)
    corr_shap_ig = _pearson_correlation(shap, ig)
    iou_gc_shap = _iou(gradcam, shap)
    iou_gc_ig = _iou(gradcam, ig)
    iou_shap_ig = _iou(shap, ig)
    ssim_gc_shap = _ssim(gradcam, shap)
    ssim_gc_ig = _ssim(gradcam, ig)
    ssim_shap_ig = _ssim(shap, ig)
    return {
        "correlation_gc_shap": corr_gc_shap,
        "correlation_gc_ig": corr_gc_ig,
        "correlation_shap_ig": corr_shap_ig,
        "iou_gc_shap": iou_gc_shap,
        "iou_gc_ig": iou_gc_ig,
        "iou_shap_ig": iou_shap_ig,
        "ssim_gc_shap": ssim_gc_shap,
        "ssim_gc_ig": ssim_gc_ig,
        "ssim_shap_ig": ssim_shap_ig,
        "mean_correlation": float(np.mean([corr_gc_shap, corr_gc_ig, corr_shap_ig])),
        "mean_iou": float(np.mean([iou_gc_shap, iou_gc_ig, iou_shap_ig])),
        "mean_ssim": float(np.mean([ssim_gc_shap, ssim_gc_ig, ssim_shap_ig])),
    }


def consistency_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    if weights is None:
        weights = {
            "mean_correlation": 0.4,
            "mean_iou": 0.3,
            "mean_ssim": 0.3,
        }
    score = 0.0
    total_weight = 0.0
    for key, w in weights.items():
        if key in metrics:
            score += w * metrics[key]
            total_weight += w
    if total_weight == 0:
        return 0.0
    return float(np.clip(score / total_weight, 0.0, 1.0))
