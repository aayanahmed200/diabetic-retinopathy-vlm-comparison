# Diabetic Retinopathy Grading: Gemini, MedGemma, and RetiZero

> A research scaffold for evaluating three vision-language systems on diabetic-retinopathy severity grading; the committed, verified run is a synthetic pilot, not real model inference.

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) [![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Overview

This repository scaffolds a comparison of Gemini, MedGemma, and RetiZero for grading diabetic-retinopathy severity on the ICDR 0–4 scale using the APTOS 2019 dataset. The intended real workflow samples images across grades, runs each system against the same images, and calculates classification metrics.

**Important status:** the notebook currently committed and verified in this repository runs in pilot mode. It generates synthetic metadata and deterministic mock predictions to exercise sampling, checkpointing, metric calculation, and reporting. It does not download real fundus images or call any of the three models.

## What the pipeline measures

- Accuracy with a bootstrap 95% confidence interval.
- Quadratic-weighted Cohen's kappa and mean absolute grade error.
- Per-class precision and recall and a confusion matrix for each model.
- Reproducible, per-image prediction checkpoints.
- Tests for sample construction, prediction determinism, and metric consistency.

## Implementation

`build_notebook.py` is the single source for both notebook modes. The default `PILOT_MODE=1` creates the synthetic pilot notebook and its fixtures. `PILOT_MODE=0` generates the intended real-run notebook, which uses the APTOS 2019 dataset and the three model integrations. The checked-in notebook and result files are pilot artifacts.

RetiZero is treated as an image/text-similarity classifier: it compares image embeddings with candidate label embeddings. Gemini and MedGemma use their respective model inference paths. The report distinguishes this methodological difference rather than presenting the systems as identical interfaces.

## Repository structure

- `build_notebook.py` — generates the pilot or real-run notebook.
- `notebooks/DR_VLM_Comparison.ipynb` — checked-in executed pilot notebook.
- `results/` — pilot predictions, metrics, summary tables, and figures.
- `reports/final_report.md` — written interpretation and status of the pilot.
- `reports/figures/` and `figures/` — report and project figures.
- `tests/` — metadata fixture and pilot-pipeline tests.
- `.env.example` — names of credentials needed for a real run; it contains no credentials.

## Run the verified pilot locally

Install the notebook and analysis dependencies:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install nbformat nbclient ipykernel pandas numpy scikit-learn matplotlib seaborn tabulate pytest
python build_notebook.py
```

Execute the generated notebook from the repository root:

```bash
python - <<'PY'
import nbformat
from nbclient import NotebookClient

path = "notebooks/DR_VLM_Comparison.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=300, resources={"metadata": {"path": "."}}).execute()
nbformat.write(notebook, path)
PY
```

Run the tests with `pytest tests/ -q`.

## Real-run status and limitations

The real comparison has not been executed end to end in this repository. Generating it requires `PILOT_MODE=0 python build_notebook.py`, a Colab GPU runtime, APTOS 2019 dataset access with the competition rules accepted, and the `GEMINI_API_KEY` and `HF_TOKEN` secrets. Real outputs must be generated and reviewed before making claims about model performance.

All committed metrics are based on synthetic data and seeded random predictions. They test the pipeline; they are not clinical evidence, model benchmarks, or conclusions about the systems' grading ability. The APTOS dataset is not included and remains subject to its Kaggle terms.

## License

MIT. See [LICENSE](LICENSE).
