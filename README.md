<div align="center">

# XAI for Medical Imaging

### Explainable AI That Clinicians Actually Trust

[![CI](https://img.shields.io/github/actions/workflow/status/Lubnaaziz-28/xai-medical-imaging/ci.yml?logo=github&style=flat-square)]()
[![Paper](https://img.shields.io/badge/Paper-Scientific_Reports_2025-0076D6?logo=readthedocs&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

*Attention maps, concept-based explanations, and audit-ready outputs for medical diagnostics.*

</div>

---

## The Problem

Medical AI models make predictions, but clinicians won't trust what they can't understand. A heatmap alone doesn't earn clinical trust, you need **why** the model decided what it decided.

## The Solution

This toolkit provides **multi-layered explainability** for medical imaging models:

- **Grad-CAM / Grad-CAM++**: visual attention overlays on predictions
- **SHAP feature attribution**: for tabular + imaging fusion models
- **Concept-based explanations**: beyond pixel heatmaps, into clinical reasoning
- **Audit-ready reports**: exportable for regulatory and clinical review

```
Medical Image
    │
    ▼
┌─────────────────────────────────────────┐
│      Prediction Model (CNN/Transformer) │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   ┌─────────┐ ┌─────┐ ┌─────────┐
   │Grad-CAM │ │SHAP │ │Concept │
   │Overlay  │ │     │ │Based   │
   └────┬────┘ └──┬──┘ └────┬────┘
        │         │         │
        ▼         ▼         ▼
   ┌─────────────────────────────────────┐
   │    Audit Report (PDF/HTML/JSON)     │
   │    - Prediction confidence          │
   │    - Explanation evidence           │
   │    - Clinical reasoning trace       │
   └─────────────────────────────────────┘
```

## Projects

| Project | Domain | Metric | Paper |
|---|---|---|---|
| Breast Cancer Diagnosis | Transfer learning + XAI | **95.3% AUC** | Scientific Reports, 2025 |
| Drug-Response Prediction | XAI for pharmacogenomics | Clinical concordance | NIH Pakistan |
| ECG/CVD Analysis | Signal processing + interpretability | Beat-level accuracy | RAEng UK |

## Quickstart

```bash
pip install -r requirements.txt

# Breast cancer demo
python notebooks/breast_cancer_demo.ipynb

# Generate audit report
python explain.py --model trained_model.h5 --image patient_scan.dcm --output report.html
```

## Datasets

Datasets (ISIC, breast imaging) are downloaded separately. See `docs/DATASETS.md`. No patient data is included.

## XAI-Radar: Multi-Method Explanation Consistency Checker

XAI-Radar runs **Grad-CAM**, **SHAP**, and **Integrated Gradients** on the same prediction and measures pixel-wise agreement between them.

### How It Works

```
Medical Image
    │
    ▼
┌─────────────────────────────────────────┐
│      Prediction Model (CNN/Transformer) │
└─────────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
 Grad-CAM      SHAP      Integrated Gradients
    │              │              │
    ▼              ▼              ▼
   Pixel-Wise Disagreement Metrics
    (correlation, IoU, SSIM)
    │
    ▼
 Consistency Score (0-1)
    │
    ▼
 Trust Verdict: GREEN | YELLOW | RED
```

### Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit demo
streamlit run app.py

# Use as a library
python -c "
from xai_radar import XaiRadar
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('my_model.h5')
image = np.load('patient_scan.npy')

radar = XaiRadar(model, target_layer='conv2d_3')
result = radar.explain(image)
print('Verdict:', result['verdict'])
print('Score:', result['consistency_score'])
"
```

### Consistency Score

The score combines three pixel-wise metrics:

| Metric | Weight |
|--------|--------|
| Pearson Correlation | 40% |
| Intersection over Union (IoU) | 30% |
| Structural Similarity (SSIM) | 30% |

### Trust Verdicts

| Score | Verdict | Meaning |
|-------|---------|---------|
| >= 0.7 | **GREEN** | Explanations are consistent across methods |
| 0.4 - 0.7 | **YELLOW** | Moderate disagreement — review recommended |
| < 0.4 | **RED** | High disagreement — prediction should not be trusted |

### Audit Log

Every run exports a JSON audit log containing all heatmaps, scores, and metadata:

```bash
from xai_radar import export_audit
export_audit(result, "audit.json", metadata={"patient_id": "P123", "model_version": "v2"})
```

### Demo

![XAI-Radar Demo GIF](docs/demo/xai_radar_demo.gif)

## Citation

```bibtex
@article{aziz2025xai,
  title={Explainable AI for medical diagnostics},
  author={Aziz, Lubna and others},
  journal={Scientific Reports},
  year={2025}
}
```

## Contact

Dr. Lubna Aziz, engr.lubnaaziz@gmail.com, [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
