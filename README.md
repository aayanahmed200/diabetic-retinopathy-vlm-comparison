# Diabetic Retinopathy Grading: Gemini, MedGemma, and RetiZero

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) [![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A research scaffold for comparing Gemini, MedGemma, and RetiZero on diabetic-retinopathy severity grading using the APTOS 2019 dataset.

## Overview

One notebook downloads the APTOS 2019 fundus photographs, samples them evenly across the ICDR grades 0-4, runs each system over the same images, and scores all three against ground truth. RetiZero is CLIP-style: it embeds the image and five candidate label strings and takes the argmax of their similarity, rather than generating a digit the way Gemini and MedGemma do. The report treats that as a difference in method, not a fair three-way tie.

## Status

The pipeline is verified in **pilot mode**: it builds a balanced synthetic sample, produces one seeded prediction per image per model, and scores them with the same code a real run uses. That run executes end to end with no errors and reproduces its committed outputs (`results/predictions.csv` and `results/metrics.txt` are byte-identical on re-run).

The real comparison has **not** been run. Every number checked into `results/` describes synthetic data and a uniform random predictor, so they measure whether the scoring code is correct, not how well any model grades. A prior partial real attempt (18 images, MedGemma only) is recorded in the appendix of `reports/final_report.md` and is kept separate from the pilot.

## Highlights

- Uses the ICDR 0-4 severity scale, with the grading rubric given to each model verbatim.
- Samples `N_PER_GRADE` images per grade so a rare severe grade is not crowded out.
- Calculates quadratic-weighted Cohen's kappa, accuracy with a bootstrap 95% confidence interval, mean absolute grade error, per-class precision and recall, and a confusion matrix per model.
- Checkpoints predictions per image, so a dropped runtime resumes instead of restarting.
- Runs in pilot mode without a GPU, Kaggle account or API keys.
- Ships tests that check the sample fixture, prediction determinism and the committed metrics against the committed predictions.

## Repository contents

- `build_notebook.py` - generates the notebook. `PILOT_MODE=1` (default) writes the synthetic pilot; `PILOT_MODE=0` writes the real-run version.
- `notebooks/DR_VLM_Comparison.ipynb` - the executed pilot notebook.
- `results/` - `predictions.csv`, `metrics.txt`, `metrics_summary.json`, `summary_table.csv` and the two pilot figures.
- `reports/final_report.md` - the write-up, including what the pilot does and does not show.
- `reports/figures/` - copies of the figures used in the report.
- `tests/` - the metadata fixture and `test_pilot_pipeline.py`.
- `.github/workflows/main.yml` - regenerates and executes the pilot notebook, then runs the tests.

## Run locally

Pilot mode needs only the data and plotting stack:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install nbformat nbclient ipykernel pandas numpy scikit-learn matplotlib seaborn tabulate
python build_notebook.py
```

That rewrites `notebooks/DR_VLM_Comparison.ipynb` and `tests/test_metadata_subset.csv`. Execute the notebook from the repository root, so its `results/` paths resolve where this README says they do:

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

Run the tests with:

```bash
pip install pytest
pytest tests/ -q
```

## Real run

`requirements.txt` is the real-run stack (Kaggle, `google-genai`, Transformers, PyTorch, `gdown`). Generate that notebook with `PILOT_MODE=0 python build_notebook.py`, then execute it in a Colab GPU runtime with the secrets named in `.env.example`: a Kaggle credential, `GEMINI_API_KEY` and `HF_TOKEN`, plus the APTOS 2019 competition rules accepted on Kaggle. Real inference has not been executed anywhere in this repository; treat that path as reviewed but unverified until someone runs it on real hardware and commits the outputs.

## Results (pilot run, synthetic)

| Model | N scored | Accuracy | Accuracy 95% CI | Quadratic-weighted kappa | MAE (grade-steps) |
|---|---|---|---|---|---|
| Gemini (synthetic) | 125 | 0.216 | [0.144, 0.288] | -0.0602 | 1.632 |
| MedGemma (synthetic) | 125 | 0.232 | [0.160, 0.304] | 0.1785 | 1.416 |
| RetiZero (synthetic) | 125 | 0.232 | [0.160, 0.312] | -0.0697 | 1.528 |

Chance-level accuracy for five balanced classes is 20%, which is where a uniform random predictor should land. These figures say nothing about the three systems' real grading ability.

## Notes and limitations

Checked-in pilot outputs use synthetic data and seeded mock predictions. They are not real model results, clinical evidence, or a validated comparison. The APTOS 2019 dataset is not included and remains subject to its Kaggle terms. Pilot figures are regenerated by the local matplotlib build, so their bytes differ between machines even though the numbers do not. With 125 images the confidence intervals stay wide, and the real dataset's own labels come from graders who disagree with each other on borderline cases.

## License

MIT - see `LICENSE`.
