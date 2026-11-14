# Diabetic Retinopathy Grading: Gemini, MedGemma, and RetiZero

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) [![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A research scaffold comparing three vision-language approaches for diabetic-retinopathy severity grading using the APTOS 2019 dataset.

> **Important:** Checked-in pilot outputs are generated from synthetic data and seeded mock predictions. They are not real model results, clinical evidence, or a validated comparison. Real model inference has not yet been run.

## Overview

The project compares a general-purpose multimodal model (Gemini), a medical vision-language model (MedGemma), and the retinal model RetiZero. RetiZero uses image-to-label embedding similarity, while the other systems generate predictions, so their methods are not identical.

## Highlights

- A reproducible notebook pipeline for data loading, prediction, and evaluation.
- Evaluation metrics include quadratic-weighted Cohen's kappa, accuracy with bootstrap confidence intervals, mean absolute grade error, per-class precision and recall, and confusion matrices.
- The reference grading scale is ICDR 0–4: no DR, mild NPDR, moderate NPDR, severe NPDR, and proliferative DR.
- Pilot mode runs without external credentials and produces deterministic synthetic outputs.

## Technology

Python, Google Colab, `google-genai`, Hugging Face Transformers, PyTorch, RetiZero, Kaggle CLI, pandas, and scikit-learn.

## Run locally

```bash
pip install -r requirements.txt
python build_notebook.py
```

For a real run, configure `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN`, set `PILOT_MODE=0`, and execute the generated notebook in a suitable GPU environment.

## Notes and limitations

- The default pilot uses 125 synthetic metadata rows and seeded predictions; its outputs must not be interpreted as model performance.
- Real mode has not been executed. Do not describe the project as a tested model comparison until real inference and validation are complete.
- RetiZero uses a different prediction approach from the generative models, and ground-truth grades can involve human disagreement.
- APTOS images are not redistributed; the repository includes an original schematic instead.

## Project files

- `build_notebook.py` — regenerates the notebook.
- `notebooks/DR_VLM_Comparison.ipynb` — pilot notebook by default.
- `reports/final_report.md` — methods, pilot results, and disclosures.
- `results/` — checked-in synthetic pilot artifacts.
- `figures/` — reference figures.
- `.github/workflows/main.yml` — CI for the pilot notebook.

## License

MIT — see `LICENSE`. The APTOS 2019 dataset remains subject to its Kaggle terms and is not included.
