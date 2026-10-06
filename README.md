<div align="center">

# XAI for Medical Imaging

### Explainable AI That Clinicians Actually Trust

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
