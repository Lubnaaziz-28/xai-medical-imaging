"""ModelCardGenerator: Orchestrates model loading, metrics, XAI, fairness, and PDF generation."""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from medcard.config import Config, load_config
from medcard.metrics.classification import (
    compute_all_metrics,
    compute_metrics_with_ci,
)
from medcard.metrics.fairness import compute_fairness_metrics
from medcard.xai.gradcam import GradCAM, GradCAMPlusPlus
from medcard.xai.shap_explainer import SHAPExplainer
from medcard.pdf.generator import ModelCardPDFGenerator
from medcard.utils.logger import setup_logger, get_logger

warnings.filterwarnings("ignore")


class ModelCardGenerator:
    """Main orchestrator for generating comprehensive medical imaging model cards."""

    def __init__(self, config: Union[Config, str]):
        """
        Initialize the model card generator.

        Args:
            config: Config object or path to YAML config file
        """
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config

        self.logger = setup_logger(verbose=self.config.verbose)
        self.model = None
        self.data = None
        self.labels = None
        self.predictions = None
        self.sensitive_attributes = {}

        self._results = {
            "metrics": {},
            "xai_visualizations": [],
            "fairness": {},
            "model_info": {},
        }

    def load_model(self, model_path: Optional[str] = None) -> tf.keras.Model:
        """
        Load the trained Keras model.

        Args:
            model_path: Optional path override

        Returns:
            Loaded Keras model
        """
        path = model_path or self.config.model_path
        if not path:
            raise ValueError("No model path provided")

        self.logger.info(f"Loading model from {path}")
        self.model = load_model(path)
        self.logger.info(f"Model loaded: {self.model.name}")
        return self.model

    def load_data(
        self,
        data_path: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data and labels.

        Args:
            data_path: Path to data directory or file
            labels: Pre-loaded labels array (for numpy format)

        Returns:
            Tuple of (data, labels)
        """
        path = data_path or self.config.data_path
        if not path:
            raise ValueError("No data path provided")

        self.logger.info(f"Loading data from {path}")

        if self.config.data_format == "directory":
            self.data, self.labels = self._load_from_directory(path)
        elif self.config.data_format == "csv":
            self.data, self.labels = self._load_from_csv(path)
        elif self.config.data_format == "numpy":
            self.data = np.load(path)
            if labels is not None:
                self.labels = labels
            else:
                raise ValueError("Labels required for numpy format")
        else:
            raise ValueError(f"Unknown data format: {self.config.data_format}")

        self.logger.info(f"Loaded {len(self.data)} samples")
        return self.data, self.labels

    def _load_from_directory(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load images from directory structure (class subdirectories)."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_directory(
            path,
            target_size=self.model.input_shape[1:3] if self.model else (224, 224),
            batch_size=32,
            class_mode="categorical" if len(self.config.class_names) > 2 else "binary",
            shuffle=False,
            classes=self.config.class_names if self.config.class_names else None,
        )

        all_data = []
        all_labels = []
        for batch_data, batch_labels in generator:
            all_data.append(batch_data)
            all_labels.append(batch_labels)
            if generator.batch_index == 0:
                break

        return np.vstack(all_data), np.vstack(all_labels)

    def _load_from_csv(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV with image paths and labels."""
        import pandas as pd
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        df = pd.read_csv(path)
        image_paths = df["image_path"].values
        labels = df[self.config.class_names].values if len(self.config.class_names) > 1 else df["label"].values

        target_size = self.model.input_shape[1:3] if self.model else (224, 224)
        data = []
        for img_path in image_paths:
            img = load_img(img_path, target_size=target_size)
            data.append(img_to_array(img) / 255.0)

        return np.array(data), np.array(labels)

    def compute_predictions(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute model predictions on data.

        Args:
            data: Optional data override (uses loaded data if None)

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        eval_data = data if data is not None else self.data
        if eval_data is None:
            raise ValueError("No data available. Call load_data() first.")

        self.logger.info("Computing predictions...")
        self.predictions = self.model.predict(eval_data, verbose=0)
        self.logger.info(f"Predictions shape: {self.predictions.shape}")
        return self.predictions

    def compute_classification_metrics(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute classification metrics with confidence intervals.

        Args:
            y_true: Ground truth labels (uses loaded if None)
            y_pred: Predictions (uses computed if None)

        Returns:
            Dictionary of metrics with confidence intervals
        """
        yt = y_true if y_true is not None else self.labels
        yp = y_pred if y_pred is not None else self.predictions

        if yt is None or yp is None:
            raise ValueError("Labels and predictions required")

        self.logger.info("Computing classification metrics...")

        if self.config.compute_ci:
            metrics = compute_metrics_with_ci(
                yt,
                yp,
                threshold=self.config.threshold,
                n_bootstrap=self.config.ci_bootstrap_samples,
                confidence=self.config.ci_confidence,
            )
        else:
            metrics = compute_all_metrics(yt, yp, threshold=self.config.threshold, class_names=self.config.class_names)

        self._results["metrics"] = metrics
        self.logger.info(f"Metrics computed: {list(metrics.keys())}")
        return metrics

    def run_xai(
        self,
        num_samples: Optional[int] = None,
        methods: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run XAI analysis (Grad-CAM and/or SHAP) on samples.

        Args:
            num_samples: Number of samples to explain (default from config)
            methods: XAI methods to run (default from config)
            output_dir: Directory to save visualizations

        Returns:
            List of visualization metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        num_samples = num_samples or self.config.num_xai_samples
        methods = methods or self.config.xai_methods
        output_dir = Path(output_dir or self.config.output_path).parent / "xai_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running XAI with methods: {methods} on {num_samples} samples")

        indices = np.random.choice(len(self.data), min(num_samples, len(self.data)), replace=False)
        visualizations = []

        if "gradcam" in methods or "gradcam++" in methods:
            visualizations.extend(self._run_gradcam(indices, output_dir, methods))

        if "shap" in methods:
            visualizations.extend(self._run_shap(indices, output_dir))

        self._results["xai_visualizations"] = visualizations
        self.logger.info(f"Generated {len(visualizations)} XAI visualizations")
        return visualizations

    def _run_gradcam(
        self,
        indices: np.ndarray,
        output_dir: Path,
        methods: List[str],
    ) -> List[Dict[str, Any]]:
        """Run Grad-CAM/Grad-CAM++ on selected samples."""
        visualizations = []

        for idx in indices:
            image = self.data[idx]
            pred_class = np.argmax(self.predictions[idx])

            if "gradcam++" in methods:
                explainer = GradCAMPlusPlus(
                    self.model,
                    layer_name=self.config.gradcam_layer,
                    class_idx=self.config.gradcam_class_idx or pred_class,
                )
                method_name = "Grad-CAM++"
            else:
                explainer = GradCAM(
                    self.model,
                    layer_name=self.config.gradcam_layer,
                    class_idx=self.config.gradcam_class_idx or pred_class,
                )
                method_name = "Grad-CAM"

            heatmap = explainer.compute_heatmap(image, class_idx=pred_class)
            output_path = output_dir / f"gradcam_sample_{idx}_class_{pred_class}.png"
            explainer.save_visualization(image, heatmap, output_path)

            visualizations.append({
                "method": method_name,
                "sample_index": int(idx),
                "predicted_class": int(pred_class),
                "class_name": self.config.class_names[pred_class] if pred_class < len(self.config.class_names) else f"Class_{pred_class}",
                "image_path": str(output_path),
                "heatmap": heatmap.tolist(),
                "caption": f"{method_name} for sample {idx} (predicted: {self.config.class_names[pred_class] if pred_class < len(self.config.class_names) else f'Class {pred_class}'})",
            })

        return visualizations

    def _run_shap(
        self,
        indices: np.ndarray,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Run SHAP explainer on selected samples."""
        visualizations = []

        if len(self.config.class_names) == 0:
            class_names = [f"Class_{i}" for i in range(self.model.output_shape[-1])]
        else:
            class_names = self.config.class_names

        explainer = SHAPExplainer(
            self.model,
            background_data=self.data,
            background_samples=self.config.shap_background_samples,
            output_names=class_names,
        )

        for idx in indices:
            image = self.data[idx]
            pred_class = np.argmax(self.predictions[idx])

            shap_values = explainer.explain_single(
                image,
                class_idx=pred_class,
                max_evals=self.config.shap_max_evals,
            )

            output_path = output_dir / f"shap_sample_{idx}_class_{pred_class}.png"
            explainer.save_shap_visualization(image, shap_values, output_path, class_idx=pred_class)

            visualizations.append({
                "method": "SHAP",
                "sample_index": int(idx),
                "predicted_class": int(pred_class),
                "class_name": class_names[pred_class] if pred_class < len(class_names) else f"Class_{pred_class}",
                "image_path": str(output_path),
                "shap_values": shap_values.tolist() if hasattr(shap_values, "tolist") else shap_values,
                "caption": f"SHAP explanation for sample {idx} (predicted: {class_names[pred_class] if pred_class < len(class_names) else f'Class {pred_class}'})",
            })

        return visualizations

    def run_fairness_analysis(
        self,
        sensitive_attrs: Optional[Dict[str, np.ndarray]] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run fairness analysis across sensitive attributes.

        Args:
            sensitive_attrs: Dictionary of attribute_name -> values
            y_true: Ground truth labels
            y_pred: Predictions

        Returns:
            Fairness report dictionary
        """
        yt = y_true if y_true is not None else self.labels
        yp = y_pred if y_pred is not None else self.predictions

        if yt is None or yp is None:
            raise ValueError("Labels and predictions required for fairness analysis")

        attrs = sensitive_attrs or self.sensitive_attributes
        if not attrs:
            self.logger.warning("No sensitive attributes provided, skipping fairness analysis")
            return {}

        self.logger.info(f"Running fairness analysis on attributes: {list(attrs.keys())}")

        fairness_report = compute_fairness_metrics(
            yt,
            yp,
            attrs,
            threshold=self.config.threshold,
            privileged_groups=self.config.privileged_groups,
        )

        self._results["fairness"] = fairness_report
        self.logger.info(f"Fairness score: {fairness_report['overall_fairness_score']:.3f}")
        return fairness_report

    def set_sensitive_attributes(self, attrs: Dict[str, np.ndarray]) -> None:
        """Set sensitive attributes for fairness analysis."""
        self.sensitive_attributes = attrs

    def generate_pdf(
        self,
        output_path: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate the final PDF model card.

        Args:
            output_path: Output PDF path
            additional_context: Additional template variables

        Returns:
            Path to generated PDF
        """
        self.logger.info("Generating PDF model card...")

        pdf_generator = ModelCardPDFGenerator(output_dir=str(Path(output_path or self.config.output_path).parent))

        context = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "author": self.config.author,
            "author_email": self.config.author_email,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "intended_use": self.config.intended_use,
            "target_population": self.config.target_population,
            "clinical_context": self.config.clinical_context,
            "limitations": additional_context.get("limitations", "") if additional_context else "",
            "ethical_considerations": additional_context.get("ethical_considerations", "") if additional_context else "",
            "include_cover_page": self.config.include_cover_page,
            "include_appendix": self.config.include_appendix,
            "generate_xai": self.config.generate_xai and len(self._results["xai_visualizations"]) > 0,
            "generate_fairness": self.config.generate_fairness and bool(self._results["fairness"]),
            "metrics": self._format_metrics_for_template(),
            "xai_visualizations": self._results["xai_visualizations"],
            "fairness": self._results["fairness"],
        }

        if additional_context:
            context.update(additional_context)

        pdf_path = pdf_generator.generate(context, output_path)
        self.logger.info(f"PDF generated: {pdf_path}")
        return pdf_path

    def _format_metrics_for_template(self) -> List[Dict[str, Any]]:
        """Format metrics for template consumption."""
        formatted = []
        metrics = self._results["metrics"]

        if "value" in metrics and "ci_lower" in metrics and "ci_upper" in metrics:
            for name, data in metrics.items():
                if isinstance(data, dict) and "value" in data:
                    formatted.append({
                        "name": name.upper(),
                        "value": f"{data['value']:.4f}",
                        "ci_lower": f"{data['ci_lower']:.4f}",
                        "ci_upper": f"{data['ci_upper']:.4f}",
                    })
        else:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    formatted.append({
                        "name": name.upper(),
                        "value": f"{value:.4f}",
                        "ci_lower": "N/A",
                        "ci_upper": "N/A",
                    })

        return formatted

    def generate(
        self,
        config_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: load model → data → predictions → metrics → XAI → fairness → PDF.

        Args:
            config_path: Optional config file path
            output_path: Optional output PDF path

        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting full model card generation pipeline")

        if config_path:
            self.config = load_config(config_path)

        self.load_model()
        self.load_data()
        self.compute_predictions()
        self.compute_classification_metrics()

        if self.config.generate_xai:
            self.run_xai()

        if self.config.generate_fairness and self.sensitive_attributes:
            self.run_fairness_analysis()

        pdf_path = self.generate_pdf(output_path)

        self._results["model_info"] = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "model_path": self.config.model_path,
            "pdf_path": pdf_path,
        }

        self.logger.info("Model card generation complete")
        return self._results

    def save_results_json(self, output_path: str) -> None:
        """Save all results to JSON file."""
        serializable = self._make_serializable(self._results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj


def main():
    """CLI entry point for ModelCardGenerator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate medical imaging model card with XAI and fairness")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", help="Output PDF path")
    parser.add_argument("--json-output", help="Path to save JSON results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    config = load_config(args.config)
    config.verbose = args.verbose or config.verbose

    generator = ModelCardGenerator(config)
    results = generator.generate(output_path=args.output)

    if args.json_output:
        generator.save_results_json(args.json_output)

    print(f"Model card generated: {results['model_info']['pdf_path']}")


if __name__ == "__main__":
    main()