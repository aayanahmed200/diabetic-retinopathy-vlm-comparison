# Diabetic Retinopathy Grading: Gemini, MedGemma, and RetiZero

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) [![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A research scaffold comparing three vision-language approaches for diabetic-retinopathy severity grading using the APTOS 2019 dataset.

> **Pilot disclosure:** Checked-in outputs use synthetic data and seeded mock predictions. They are not real model results, clinical evidence, or a validated comparison. Real model inference has not yet been run.

## Overview

The project compares Gemini, MedGemma, and RetiZero. RetiZero uses image-to-label embedding similarity, while the other systems generate predictions, so the methods are not identical.

## Highlights

- Reproducible notebook pipeline for data loading, prediction, and evaluation.
- Metrics include quadratic-weighted Cohen's kappa, accuracy with bootstrap confidence intervals, mean absolute grade error, per-class precision and recall, and confusion matrices.
- Uses the ICDR 0–4 severity scale.
- Pilot mode runs without external credentials and produces deterministic synthetic outputs.

## Technology

Python, Google Colab, `google-genai`, Hugging Face Transformers, PyTorch, RetiZero, Kaggle CLI, pandas, and scikit-learn.

## Run locally

```bash
pip install -r requirements.txt
python build_notebook.py
```

For a real run, configure `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN`, set `PILOT_MODE=0`, and execute the generated notebook in a suitable GPU environment.


## License

MIT — see `LICENSE`. The APTOS 2019 dataset remains subject to its Kaggle terms and is not included.
