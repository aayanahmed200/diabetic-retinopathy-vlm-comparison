# Diabetic Retinopathy: VLM Grading Pipeline

A notebook pipeline scaffold for evaluating Gemini, MedGemma and RetiZero on diabetic-retinopathy grading with the APTOS 2019 dataset.

## Overview

The repository currently ships with a **synthetic pilot**, not a real model comparison. By default, `build_notebook.py` generates a notebook with a balanced synthetic sample of 125 records and deterministic pseudo-random predictions. This mode exercises the sampling, scoring and reporting pipeline without downloading APTOS images or calling the models.

The script can also generate a real-run notebook when `PILOT_MODE=0`. That path is intended for a suitable GPU environment and requires Kaggle access plus the relevant model-service credentials. It has not been executed or validated against real model outputs in this repository.

## Repository contents

- `build_notebook.py` — generates the pilot notebook by default, or the real-run notebook when requested.
- `notebooks/` — generated comparison notebook.
- `reports/final_report.md` — explanation of the pilot methodology and results.
- `results/` — pilot metrics, predictions, summary tables and plots.
- `tests/` — checks for metadata sampling, prediction reproducibility and evaluation logic.
- `.env.example` — names of credentials used by the real-run workflow.

## Technology

Python, Jupyter/nbformat, PyTorch, Hugging Face Transformers, Google GenAI, Kaggle CLI, pandas, NumPy and scikit-learn. GPU support is needed for the intended real model run, not for generating the synthetic pilot.

## Run locally

Install the dependencies and generate the default pilot notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_notebook.py
```

Open `notebooks/DR_VLM_Comparison.ipynb` in Jupyter and run the cells to reproduce the pilot workflow.

For the unverified real-run path, set `PILOT_MODE=0` before generating the notebook, then run it in a suitable GPU environment with `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GEMINI_API_KEY` and `HF_TOKEN` configured. Do not use the checked-in pilot outputs as a substitute for that run.

Run the tests with:

```bash
pytest tests/
```

## Scope and limitations

The checked-in sample, ground-truth labels and model predictions are synthetic. The reported accuracy, kappa and other metrics measure the behaviour of seeded random predictions on synthetic data; they say nothing about the real grading performance of Gemini, MedGemma or RetiZero. No clinical conclusions can be drawn from this pilot, and the project is not a diagnostic tool. A real comparison remains to be run and independently reviewed.

The APTOS 2019 dataset is not included in this repository and remains subject to its Kaggle terms.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
