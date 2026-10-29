"""SHAP explainer for medical imaging models."""

import numpy as np
import tensorflow as tf
import shap
from typing import Optional, List, Tuple, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")


class SHAPExplainer:
    """SHAP explainer for TensorFlow/Keras models with support for image inputs."""

    def __init__(
        self,
        model: tf.keras.Model,
        background_data: Optional[np.ndarray] = None,
        background_samples: int = 100,
        output_names: Optional[List[str]] = None,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained Keras model
            background_data: Background dataset for expected values (auto-sample if None)
            background_samples: Number of background samples to use
            output_names: Names of output classes
        """
        self.model = model
        self.background_data = background_data
        self.background_samples = background_samples
        self.output_names = output_names or [f"Class_{i}" for i in range(model.output_shape[-1])]
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize the SHAP explainer with appropriate algorithm."""
        # For image models, use GradientExplainer (faster) or DeepExplainer
        try:
            if self.background_data is not None:
                background = shap.sample(self.background_data, self.background_samples)
            else:
                # Create dummy background based on input shape
                input_shape = self.model.input_shape[1:]
                background = np.zeros((self.background_samples, *input_shape))

            self.explainer = shap.GradientExplainer(self.model, background)
        except Exception:
            # Fallback to KernelExplainer for any model
            if self.background_data is not None:
                background = shap.sample(self.background_data, min(50, self.background_samples))
            else:
                input_shape = self.model.input_shape[1:]
                background = np.zeros((min(50, self.background_samples), *input_shape))

            def model_fn(x):
                return self.model(x, training=False).numpy()

            self.explainer = shap.KernelExplainer(model_fn, background)

    def explain(
        self,
        images: np.ndarray,
        max_evals: int = 500,
        batch_size: int = 10,
    ) -> np.ndarray:
        """
        Compute SHAP values for input images.

        Args:
            images: Input images (N, H, W, C)
            max_evals: Maximum evaluations for KernelExplainer
            batch_size: Batch size for processing

        Returns:
            SHAP values array (N, H, W, C, num_classes) or (N, H, W, C) for binary
        """
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        if hasattr(self.explainer, "shap_values"):
            # GradientExplainer / DeepExplainer
            shap_values = self.explainer.shap_values(images)
        else:
            # KernelExplainer
            shap_values = self.explainer.shap_values(images, max_evals=max_evals, batch_size=batch_size)

        return shap_values

    def explain_single(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        max_evals: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for a single image.

        Args:
            image: Input image (H, W, C) or (1, H, W, C)
            class_idx: Target class index (None for all classes)
            max_evals: Maximum evaluations

        Returns:
            SHAP values for the specified class or all classes
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        shap_values = self.explain(image, max_evals=max_evals)

        if isinstance(shap_values, list):
            # Multi-class output
            if class_idx is not None:
                return shap_values[class_idx][0]
            return [sv[0] for sv in shap_values]
        else:
            # Binary or single output
            if shap_values.ndim == 5:
                if class_idx is not None:
                    return shap_values[0, ..., class_idx]
                return shap_values[0]
            return shap_values[0]

    def visualize_shap(
        self,
        image: np.ndarray,
        shap_values: np.ndarray,
        class_idx: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Create SHAP visualization for an image.

        Args:
            image: Original image (H, W, C)
            shap_values: SHAP values from explain_single
            class_idx: Class index for labeling
            output_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if image.max() <= 1.0:
            image_display = image
        else:
            image_display = image / 255.0

        # Sum across channels for visualization
        if shap_values.ndim == 3:
            shap_sum = np.sum(shap_values, axis=-1)
        else:
            shap_sum = shap_values

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(image_display)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # SHAP values (positive = supports prediction, negative = contradicts)
        vmax = np.percentile(np.abs(shap_sum), 99)
        im = axes[1].imshow(shap_sum, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"SHAP Values{' - ' + self.output_names[class_idx] if class_idx is not None else ''}")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(image_display)
        axes[2].imshow(shap_sum, cmap="RdBu_r", alpha=0.5, vmin=-vmax, vmax=vmax)
        axes[2].set_title("SHAP Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

        return fig

    def save_shap_visualization(
        self,
        image: np.ndarray,
        shap_values: np.ndarray,
        output_path: Union[str, Path],
        class_idx: Optional[int] = None,
    ) -> None:
        """Save SHAP visualization to file."""
        self.visualize_shap(image, shap_values, class_idx, output_path)


def compute_shap_interaction_values(
    explainer: SHAPExplainer,
    image: np.ndarray,
    class_idx: int = 0,
) -> np.ndarray:
    """
    Compute SHAP interaction values for deeper analysis.

    Args:
        explainer: Initialized SHAPExplainer
        image: Single input image
        class_idx: Target class

    Returns:
        Interaction values matrix
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    if hasattr(explainer.explainer, "shap_interaction_values"):
        interactions = explainer.explainer.shap_interaction_values(image)
        if isinstance(interactions, list):
            return interactions[class_idx][0]
        return interactions[0]
    else:
        raise NotImplementedError("Interaction values only available for TreeExplainer")