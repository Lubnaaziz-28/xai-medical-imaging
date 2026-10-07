from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf

from .metrics import compute_metrics, consistency_score
from .verdict import compute_verdict


class XaiRadar:
    def __init__(
        self,
        model: tf.keras.Model,
        preprocess_fn: Callable | None = None,
        target_layer: str | None = None,
        ig_steps: int = 50,
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.target_layer = target_layer or self._find_last_conv(model)
        self.ig_steps = ig_steps
        self._conv_model = self._build_conv_model()

    def _find_last_conv(self, model: tf.keras.Model) -> str:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in model")

    def _build_conv_model(self):
        target_layer = self.model.get_layer(self.target_layer)
        return tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, self.model.output],
        )

    def _preprocess(self, image: np.ndarray) -> tf.Tensor:
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        return image

    def compute_gradcam(self, image: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        image_tensor = self._preprocess(image)
        if class_idx is None:
            class_idx = int(tf.argmax(self.model(image_tensor, training=False)[0]).numpy())

        @tf.function
        def compute_grads(img):
            with tf.GradientTape() as tape:
                tape.watch(img)
                conv_output, predictions = self._conv_model(img, training=False)
                class_score = predictions[:, class_idx]
            grads = tape.gradient(class_score, conv_output)
            return conv_output, grads

        conv_output, grads = compute_grads(image_tensor)
        conv_output = conv_output.numpy()[0]
        grads = grads.numpy()[0]

        weights = np.mean(grads, axis=(0, 1))
        cam = np.dot(conv_output, weights)
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., np.newaxis], image_tensor.shape[1:3]).numpy()[..., 0]
        return cam

    def compute_shap(self, image: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        import shap
        image_tensor = self._preprocess(image)

        def f(x):
            preds = self.model(x, training=False)
            if class_idx is None:
                return preds
            return preds[:, class_idx] if preds.shape[1] > 1 else preds[:, 0]

        explainer = shap.GradientExplainer(self.model, image_tensor)
        try:
            shap_values = explainer.shap_values(image_tensor, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(image_tensor)

        if isinstance(shap_values, list):
            if class_idx is None:
                class_idx = int(tf.argmax(self.model(image_tensor, training=False)[0]).numpy())
            shap_map = shap_values[class_idx][0].mean(axis=-1)
        else:
            shap_map = shap_values[0].mean(axis=-1)

        shap_map = np.maximum(shap_map, 0)
        shap_map = shap_map / (np.max(shap_map) + 1e-8)
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        shap_map = tf.image.resize(shap_map[..., np.newaxis], [h, w]).numpy()[..., 0]
        return shap_map

    def compute_integrated_gradients(
        self, image: np.ndarray, class_idx: int | None = None, baseline: np.ndarray | None = None
    ) -> np.ndarray:
        image_tensor = self._preprocess(image)
        if baseline is None:
            baseline = tf.zeros_like(image_tensor)
        else:
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        if class_idx is None:
            class_idx = int(tf.argmax(self.model(image_tensor, training=False)[0]).numpy())

        steps = self.ig_steps
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        accumulated_grads = tf.zeros_like(image_tensor)

        for alpha in alphas:
            with tf.GradientTape() as tape:
                tape.watch(image_tensor)
                interp = baseline + alpha * (image_tensor - baseline)
                preds = self.model(interp, training=False)
                class_score = preds[:, class_idx]
            grads = tape.gradient(class_score, image_tensor)
            accumulated_grads += grads

        avg_grads = accumulated_grads / (steps + 1)
        ig = (image_tensor - baseline) * avg_grads
        ig_map = tf.reduce_sum(ig, axis=-1).numpy()[0]
        ig_map = np.maximum(ig_map, 0)
        ig_map = ig_map / (np.max(ig_map) + 1e-8)
        return ig_map

    def explain(self, image: np.ndarray, class_idx: int | None = None) -> dict[str, Any]:
        gradcam = self.compute_gradcam(image, class_idx)
        shap = self.compute_shap(image, class_idx)
        ig = self.compute_integrated_gradients(image, class_idx)
        metrics = compute_metrics(gradcam, shap, ig)
        score = consistency_score(metrics)
        verdict = compute_verdict(score)
        return {
            "gradcam": gradcam,
            "shap": shap,
            "integrated_gradients": ig,
            "metrics": metrics,
            "consistency_score": float(score),
            "verdict": verdict,
        }
