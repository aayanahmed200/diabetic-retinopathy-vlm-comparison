"""Checks for the pilot pipeline: the fixture, determinism, and the committed metrics.

These run without a GPU, Kaggle account or API keys, because pilot mode needs none of
them. Run with:

    pytest tests/ -q
"""
import json
import pathlib
import re
import subprocess
import sys

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "build_notebook.py"
FIXTURE = ROOT / "tests" / "test_metadata_subset.csv"
PREDICTIONS = ROOT / "results" / "predictions.csv"
SUMMARY = ROOT / "results" / "metrics_summary.json"

# Mirrors the configuration cell in the generated notebook. If these change there, the
# fixture below changes with them, and this test is what notices.
N_PER_GRADE = 25
RANDOM_SEED = 42
GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
MODEL_COLUMNS = ["gemini_pred", "medgemma_pred", "retizero_pred"]


def _embedded_source(name):
    """Return one of the source strings build_notebook.py embeds in the notebook."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"^{name} = '''\\?\n(.*?)^'''$", text, re.S | re.M)
    if match is None:
        raise AssertionError(f"{name} not found in build_notebook.py")
    return match.group(1).replace("\\\n", "")


def _pilot_namespace():
    """Execute the notebook's own metadata and prediction source, as the notebook does."""
    namespace = {
        "random": __import__("random"),
        "hashlib": __import__("hashlib"),
        "N_PER_GRADE": N_PER_GRADE,
        "RANDOM_SEED": RANDOM_SEED,
        "GRADE_NAMES": GRADE_NAMES,
    }
    for name in ("PILOT_METADATA_SRC", "SEEDED_PREDICTION_SRC"):
        exec(_embedded_source(name), namespace)
    return namespace


def test_fixture_matches_the_notebooks_sample_generator():
    """The checked-in fixture is produced by the same code as the notebook's sample."""
    metadata = _pilot_namespace()["build_pilot_metadata"]()
    expected = pd.DataFrame(metadata)[["id_code", "image_path", "diagnosis"]]
    committed = pd.read_csv(FIXTURE)

    assert len(committed) == 5 * N_PER_GRADE
    assert list(committed.columns) == ["id_code", "image_path", "diagnosis"]
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert committed["diagnosis"].value_counts().sort_index().tolist() == [N_PER_GRADE] * 5


def test_committed_predictions_use_the_same_images_as_the_fixture():
    """The committed run was scored on the sample the fixture describes, not another one."""
    fixture = pd.read_csv(FIXTURE)
    predictions = pd.read_csv(PREDICTIONS)

    assert len(predictions) == len(fixture)
    merged = predictions.merge(
        fixture[["id_code", "diagnosis"]].rename(columns={"diagnosis": "fixture_grade"}),
        on="id_code",
        how="outer",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all(), "id_code sets differ between fixture and run"
    assert (merged["ground_truth"] == merged["fixture_grade"]).all()
    for column in MODEL_COLUMNS:
        assert predictions[column].notna().all(), f"{column} has missing predictions"
        assert predictions[column].between(0, 4).all(), f"{column} left the 0-4 range"


def test_seeded_prediction_is_stable_across_processes():
    """Same (image, model) pair gives the same grade regardless of PYTHONHASHSEED."""
    script = (
        "import random, hashlib\n"
        + _embedded_source("SEEDED_PREDICTION_SRC")
        + "\nprint(seeded_prediction('pilot_2_007', 'medgemma'))\n"
    )
    outputs = []
    for hash_seed in ("0", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
            check=True,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1]
    assert outputs[0].isdigit() and 0 <= int(outputs[0]) <= 4


def test_committed_metrics_recompute_from_committed_predictions():
    """metrics_summary.json describes results/predictions.csv, not some other run."""
    predictions = pd.read_csv(PREDICTIONS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert set(summary) == {"gemini", "medgemma", "retizero"}
    for model, column in zip(summary, MODEL_COLUMNS):
        scored = predictions[predictions[column].notna()]
        y_true = scored["ground_truth"].astype(int)
        y_pred = scored[column].astype(int)

        assert summary[model]["n_scored"] == len(scored)
        assert summary[model]["n_total"] == len(predictions)
        assert summary[model]["accuracy"] == pytest.approx(
            accuracy_score(y_true, y_pred), abs=5e-4
        )
        assert summary[model]["quadratic_weighted_kappa"] == pytest.approx(
            cohen_kappa_score(y_true, y_pred, weights="quadratic"), abs=5e-4
        )
        assert summary[model]["mean_absolute_error_grades"] == pytest.approx(
            mean_absolute_error(y_true, y_pred), abs=5e-4
        )

        low, high = summary[model]["accuracy_95ci"]
        assert 0.0 <= low <= summary[model]["accuracy"] <= high <= 1.0
