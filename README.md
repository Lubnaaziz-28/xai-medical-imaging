# xai-medical-imaging

Explainability toolkit for medical imaging models. Because a heatmap alone doesn't earn clinician trust. This repo pairs predictions with concept-based explanations and audit-ready reports.

Built from published work: transfer learning for breast cancer diagnosis (Scientific Reports, 2025) and XAI for drug-response prediction (NIH Pakistan).

## Features
- Grad-CAM / Grad-CAM++ overlays per prediction
- SHAP feature attribution for tabular + imaging fusion models
- Concept-based explanations (beyond pixel heatmaps)
- Exportable audit reports for clinical review

## Quickstart
```bash
pip install -r requirements.txt
python notebooks/breast_cancer_demo.ipynb
```

## Note
Datasets (ISIC, breast imaging) are downloaded separately, see `docs/DATASETS.md`. No patient data is included.

## Contact
Dr. Lubna Aziz | engr.lubnaaziz@gmail.com | [Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
