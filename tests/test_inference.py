from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from dns_exfiltration.data import DataValidationError
from dns_exfiltration.experiments import run_experiment
from dns_exfiltration.inference import predict_csv
from dns_exfiltration.models import make_model


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("inference-run")
    rng = np.random.default_rng(12)
    y = np.tile([0, 1], 120)
    rng.shuffle(y)
    frame = pd.DataFrame({
        "row_key": [f"input-{i}" for i in range(len(y))],
        "signal": y * 2 + rng.normal(0, 0.4, len(y)),
        "secondary": y * 0.5 + rng.normal(0, 1, len(y)),
        "noise": rng.normal(0, 1, len(y)), "label": y,
    })
    frame.loc[0, "noise"] = np.nan
    source = folder / "source.csv"
    frame.to_csv(source, index=False)
    config = {
        "mode": "within", "seed": 42, "top_k": 2,
        "model": {"name": "lr", "params": {}},
        "source": {
            "path": str(source), "protocol": "synthetic_toy", "label_column": "label",
            "label_mapping": {"0": 0, "1": 1},
            "feature_columns": ["signal", "secondary", "noise"], "id_column": "row_key",
            "provenance": {"kind": "synthetic_demo", "description": "Generated unit-test observations"},
            "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "adaptation": 0},
        },
    }
    run_dir = run_experiment(config, folder / "runs")
    expected = pd.read_csv(run_dir / "predictions.csv").query("variant == 'A_all'")
    test_frame = frame.set_index("row_key").loc[expected["row_id"]].reset_index()
    return run_dir, test_frame, expected


def input_csv(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path


def test_saved_model_inference_matches_original_test_predictions(saved_run, tmp_path):
    run_dir, frame, expected = saved_run
    path = input_csv(tmp_path, frame.drop(columns="label"))
    output = tmp_path / "predictions.csv"
    report = predict_csv(run_dir, path, output, id_column="row_key")
    actual = pd.read_csv(output)
    assert list(actual) == ["row_id", "score", "prediction", "threshold"]
    assert list(actual["row_id"]) == list(expected["row_id"])
    np.testing.assert_allclose(actual["score"], expected["score"], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(actual["prediction"], expected["prediction"])
    assert report["rows"] == len(frame) and report["threshold"] == 0.5
    assert report["score_kind"] == "probability" and report["synthetic_demo_only"]
    assert report["provenance"]["source"]["declaration"]["kind"] == "synthetic_demo"
    assert not report["allow_real_data"] and not report["confirm_provenance_audit"]


def test_inference_never_reads_label_columns_or_fits(saved_run, tmp_path, monkeypatch):
    import dns_exfiltration.inference as inference_module
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    run_dir, original, expected = saved_run
    frame = original.copy()
    frame["label"] = "unknown label text ignored by inference"
    path = input_csv(tmp_path, frame)
    model = joblib.load(run_dir / "model_A_all.joblib")
    features = list(model.feature_names_in_)
    before = {
        "median": model.named_steps["imputer"].statistics_.copy(),
        "mean": model.named_steps["scaler"].mean_.copy(),
        "scale": model.named_steps["scaler"].scale_.copy(),
        "coefficients": model.named_steps["classifier"].coef_.copy(),
    }
    original_read_csv = inference_module.pd.read_csv

    def guarded_read(*args, **kwargs):
        assert kwargs["usecols"] == features + ["row_key"]
        assert "label" not in kwargs["usecols"]
        return original_read_csv(*args, **kwargs)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Inference attempted to fit a model or preprocessing stage")

    monkeypatch.setattr(inference_module.pd, "read_csv", guarded_read)
    monkeypatch.setattr(inference_module.joblib, "load", lambda _: model)
    for estimator_class in (Pipeline, SimpleImputer, StandardScaler, LogisticRegression):
        monkeypatch.setattr(estimator_class, "fit", forbidden_fit)
    output = tmp_path / "predictions.csv"
    predict_csv(run_dir, path, output, id_column="row_key")
    np.testing.assert_array_equal(before["median"], model.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(before["mean"], model.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(before["scale"], model.named_steps["scaler"].scale_)
    np.testing.assert_array_equal(before["coefficients"], model.named_steps["classifier"].coef_)
    actual = original_read_csv(output)
    np.testing.assert_allclose(actual["score"], expected["score"], rtol=0, atol=1e-14)


@pytest.mark.parametrize("variant", ["A_all", "B_importance_top_k"])
def test_feature_order_is_saved_order_and_ids_can_be_generated(saved_run, tmp_path, variant):
    run_dir, frame, _ = saved_run
    frame = frame[list(reversed(frame.columns))].drop(columns=["row_key", "label"])
    output = tmp_path / "predictions.csv"
    report = predict_csv(run_dir, input_csv(tmp_path, frame), output, variant=variant)
    actual = pd.read_csv(output)
    assert actual["row_id"].tolist() == [f"row:{i + 1}" for i in range(len(frame))]
    model = joblib.load(run_dir / f"model_{variant}.joblib")
    features = list(model.feature_names_in_)
    np.testing.assert_allclose(actual["score"], model.predict_proba(frame[features])[:, 1], atol=1e-14)
    assert report["variant"] == variant


def test_infinity_and_missing_columns_use_saved_imputer(saved_run, tmp_path):
    run_dir, frame, _ = saved_run
    changed = frame.drop(columns="label").copy()
    changed["noise"] = np.inf
    changed.loc[0, "signal"] = np.nan
    output = tmp_path / "predictions.csv"
    predict_csv(run_dir, input_csv(tmp_path, changed), output, id_column="row_key")
    actual = pd.read_csv(output)
    assert np.isfinite(actual["score"]).all()
    model = joblib.load(run_dir / "model_A_all.joblib")
    features = list(model.feature_names_in_)
    clean = changed[features].replace([np.inf, -np.inf], np.nan)
    np.testing.assert_allclose(actual["score"], model.predict_proba(clean)[:, 1], atol=1e-14)


@pytest.mark.parametrize("mutation,match", [
    ("missing_feature", "missing required columns"),
    ("nonnumeric_feature", "nonnumeric"),
    ("duplicate_id", "unique"),
    ("blank_id", "empty"),
    ("duplicate_header", "duplicate column"),
    ("empty", "empty"),
])
def test_malformed_inference_input_fails_without_output(saved_run, tmp_path, mutation, match):
    run_dir, original, _ = saved_run
    frame = original.drop(columns="label").copy()
    if mutation == "missing_feature":
        frame = frame.drop(columns="signal")
    elif mutation == "nonnumeric_feature":
        frame["signal"] = frame["signal"].astype(object)
        frame.loc[0, "signal"] = "invalid text"
    elif mutation == "duplicate_id":
        frame.loc[1, "row_key"] = frame.loc[0, "row_key"]
    elif mutation == "blank_id":
        frame.loc[0, "row_key"] = ""
    path = input_csv(tmp_path, frame)
    if mutation == "duplicate_header":
        path.write_text("signal,signal,secondary,noise,row_key\n1,2,3,4,a\n")
    elif mutation == "empty":
        path.write_text("")
    output = tmp_path / "must-not-exist.csv"
    with pytest.raises(DataValidationError, match=match):
        predict_csv(run_dir, path, output, id_column="row_key")
    assert not output.exists()


def test_output_is_never_overwritten(saved_run, tmp_path):
    run_dir, frame, _ = saved_run
    path = input_csv(tmp_path, frame)
    output = tmp_path / "predictions.csv"
    output.write_text("existing important output\n")
    with pytest.raises(FileExistsError, match="already exists"):
        predict_csv(run_dir, path, output)
    assert output.read_text() == "existing important output\n"


def test_incomplete_run_and_unavailable_variant_are_rejected(saved_run, tmp_path):
    run_dir, frame, _ = saved_run
    path = input_csv(tmp_path, frame)
    with pytest.raises(DataValidationError, match="unavailable"):
        predict_csv(run_dir, path, tmp_path / "unavailable.csv", variant="C_satfs_top_k")
    incomplete = tmp_path / "incomplete-run"
    shutil.copytree(run_dir, incomplete)
    manifest = json.loads((incomplete / "run_manifest.json").read_text())
    manifest["status"] = "failed"
    (incomplete / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="completed"):
        predict_csv(incomplete, path, tmp_path / "incomplete.csv")


def test_svm_uses_fixed_decision_threshold_without_probability(tmp_path):
    run_dir = tmp_path / "svm-run"
    run_dir.mkdir()
    X = pd.DataFrame({"signal": [-2.0, -1.0, 1.0, 2.0]})
    model = make_model({"name": "svm"}, seed=42).fit(X, [0, 0, 1, 1])
    joblib.dump(model, run_dir / "model_A_all.joblib")
    (run_dir / "config.json").write_text(json.dumps({"model": {"name": "svm"}}))
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed", "synthetic_demo_only": True}))
    (run_dir / "selected_features.json").write_text(json.dumps({"A_all": ["signal"]}))
    output = tmp_path / "svm-predictions.csv"
    report = predict_csv(run_dir, input_csv(tmp_path, X), output)
    actual = pd.read_csv(output)
    assert report["threshold"] == 0 and report["score_kind"] == "decision_function"
    np.testing.assert_allclose(actual["score"], model.decision_function(X), atol=1e-14)
    assert actual["prediction"].tolist() == [0, 0, 1, 1]
