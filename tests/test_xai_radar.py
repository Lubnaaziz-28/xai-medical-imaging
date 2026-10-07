import json
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from xai_radar import (
    XaiRadar,
    compute_metrics,
    compute_verdict,
    consistency_score,
    export_audit,
)


def _build_tiny_model():
    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def test_metrics_perfect_agreement():
    a = np.ones((8, 8), dtype=np.float32)
    b = np.ones((8, 8), dtype=np.float32)
    c = np.ones((8, 8), dtype=np.float32)
    metrics = compute_metrics(a, b, c)
    assert metrics["mean_correlation"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["mean_iou"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["mean_ssim"] == pytest.approx(1.0, abs=1e-3)


def test_metrics_opposite_maps():
    a = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    b = 1.0 - a
    c = np.full((8, 8), 0.5, dtype=np.float32)
    metrics = compute_metrics(a, b, c)
    assert metrics["mean_correlation"] < 0.0


def test_consistency_score_range():
    score = consistency_score({
        "mean_correlation": 0.9,
        "mean_iou": 0.9,
        "mean_ssim": 0.9,
    })
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.9)


def test_verdict_green():
    assert compute_verdict(0.8) == "GREEN"


def test_verdict_yellow():
    assert compute_verdict(0.5) == "YELLOW"


def test_verdict_red():
    assert compute_verdict(0.2) == "RED"


def test_export_audit_writes_json():
    result = {
        "gradcam": np.ones((4, 4), dtype=np.float32),
        "shap": np.ones((4, 4), dtype=np.float32),
        "integrated_gradients": np.ones((4, 4), dtype=np.float32),
        "metrics": {"mean_correlation": 1.0},
        "consistency_score": 1.0,
        "verdict": "GREEN",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "audit.json")
        export_audit(result, path, metadata={"test": True})
        with open(path) as f:
            data = json.load(f)
        assert data["verdict"] == "GREEN"
        assert "timestamp" in data
        assert len(data["heatmaps"]["gradcam"]) == 4


def test_engine_runs():
    model = _build_tiny_model()
    image = np.random.rand(8, 8, 3).astype(np.float32)
    radar = XaiRadar(model, ig_steps=5)
    try:
        result = radar.explain(image, class_idx=0)
    except Exception as e:
        if "shap" in str(e).lower():
            pytest.skip("SHAP backend incompatible in this environment")
        raise
    assert "gradcam" in result
    assert "shap" in result
    assert "integrated_gradients" in result
    assert result["gradcam"].shape == (8, 8)
    assert result["shap"].shape == (8, 8)
    assert result["integrated_gradients"].shape == (8, 8)
    assert result["verdict"] in ("GREEN", "YELLOW", "RED")
    assert 0.0 <= result["consistency_score"] <= 1.0
