"""Grad-CAM and Grad-CAM++ implementations for TensorFlow/Keras models."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import cv2
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """Grad-CAM implementation for visualizing CNN decisions."""

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
        class_idx: Optional[int] = None,
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained Keras model
            layer_name: Target convolutional layer name (auto-detect if None)
            class_idx: Target class index (use predicted class if None)
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.class_idx = class_idx
        self.grad_model = self._build_grad_model()

    def _find_target_layer(self) -> str:
        """Auto-detect the last convolutional layer."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name
        raise ValueError("No Conv2D layer found in model")

    def _build_grad_model(self) -> Model:
        """Build model that outputs target layer activations and predictions."""
        layer = self.model.get_layer(self.layer_name)
        return Model(
            inputs=self.model.inputs,
            outputs=[layer.output, self.model.output],
        )

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            image: Input image array (H, W, C) or (1, H, W, C)
            class_idx: Target class index
            eps: Small epsilon for numerical stability

        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        class_idx = class_idx if class_idx is not None else self.class_idx

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W, C) in [0, 1] or [0, 255]
            heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            alpha: Overlay transparency
            colormap: OpenCV colormap

        Returns:
            Overlaid image (H, W, C) in [0, 255] uint8
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        heatmap_resized = cv2.resize(
            (heatmap * 255).astype(np.uint8),
            (image.shape[1], image.shape[0]),
        )
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid

    def save_visualization(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        output_path: Union[str, Path],
        alpha: float = 0.4,
        dpi: int = 150,
    ) -> None:
        """Save Grad-CAM visualization as image file."""
        overlaid = self.overlay_heatmap(image, heatmap, alpha)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlaid)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation with improved localization."""

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute Grad-CAM++ heatmap with weighted gradients."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        class_idx = class_idx if class_idx is not None else self.class_idx

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        grads = grads[0]

        # Grad-CAM++ weights: alpha = grad^2 / (2*grad^2 + sum(conv*grad^3))
        grads_sq = grads ** 2
        grads_cb = grads ** 3
        conv_outputs = conv_outputs[0]

        sum_conv_grads = tf.reduce_sum(conv_outputs * grads_cb, axis=(0, 1))
        alpha_num = grads_sq
        alpha_denom = 2 * grads_sq + tf.reduce_sum(conv_outputs * grads_cb, axis=(0, 1), keepdims=True)
        alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom

        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy()