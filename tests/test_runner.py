from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dns_exfiltration.data import DataValidationError, load_numeric_csv
from dns_exfiltration.experiments import run_experiment
from dns_exfiltration.metrics import binary_metrics, model_scores
from dns_exfiltration.models import make_model
from dns_exfiltration.splits import make_partitions


SOURCE_SPLIT = {"train": 0.6, "validation": 0.2, "test": 0.2, "adaptation": 0.0}


def dataset_spec(tmp_path: Path, name="source", *, seed=1, n=240, groups=False):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    rng.shuffle(labels)
    frame = pd.DataFrame({
        "row_key": [f"{name}-{i}" for i in range(n)],
        "signal": labels * 2 + rng.normal(0, 0.5, n),
        "secondary": labels * 0.7 + rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
        "label": np.where(labels, "attack", "benign"),
    })
    if groups:
        frame["capture"] = [f"capture-{i // 8}" for i in range(n)]
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    spec = {
        "path": str(path), "label_column": "label",
        "protocol": "synthetic_toy",
        "label_mapping": {"benign": 0, "attack": 1},
        "feature_columns": ["signal", "secondary", "noise"], "id_column": "row_key",
        "provenance": {"kind": "synthetic_demo", "description": "Unit-test generated numeric data"},
        "split": SOURCE_SPLIT.copy(),
    }
    if groups:
        spec["group_column"] = "capture"
    return spec, frame


def write_frame(spec, frame):
    frame.to_csv(spec["path"], index=False)


def transfer_config(tmp_path, mode="adaptation_assisted"):
    source, _ = dataset_spec(tmp_path, "source", seed=1)
    target, _ = dataset_spec(tmp_path, "target", seed=2)
    target["split"] = ({"train": 0.4, "validation": 0.2, "test": 0.2, "adaptation": 0.2}
                       if mode == "adaptation_assisted" else SOURCE_SPLIT.copy())
    return {
        "experiment_name": "unit-test", "mode": mode, "seed": 42, "top_k": 2,
        "model": {"name": "lr", "params": {}}, "source": source, "target": target,
        "compatible_features": [
            {"name": column, "source_column": column, "target_column": column,
             "definition": f"Synthetic {column}", "unit": "arbitrary", "window": "one synthetic observation"}
            for column in source["feature_columns"]
        ],
    }


def test_deduplication_preserves_first_id_and_converts_infinity(tmp_path):
    spec, frame = dataset_spec(tmp_path)
    duplicate = frame.iloc[[0]].copy()
    duplicate["row_key"] = "duplicate-row"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame.loc[1, "noise"] = np.inf
    write_frame(spec, frame)
    data = load_numeric_csv(spec)
    assert data.removed_duplicates == 1
    assert len(data.X) == len(frame) - 1
    assert "source-0" in set(data.row_ids) and "duplicate-row" not in set(data.row_ids)
    assert np.isnan(data.X.iloc[1]["noise"])
    assert data.nonfinite_values == 1


@pytest.mark.parametrize("mutation,match", [
    ("unknown_label", "Unknown labels"),
    ("missing_label", "Labels may not be missing"),
    ("duplicate_id", "unique IDs"),
    ("empty_id", "missing values|empty"),
    ("nonnumeric", "nonnumeric"),
    ("label_copy", "copy or complement"),
    ("conflicting_duplicate", "conflicting labels"),
])
def test_invalid_dataset_rejected(tmp_path, mutation, match):
    spec, frame = dataset_spec(tmp_path)
    if mutation == "unknown_label":
        frame.loc[0, "label"] = "mystery"
    elif mutation == "missing_label":
        frame.loc[0, "label"] = None
    elif mutation == "duplicate_id":
        frame.loc[1, "row_key"] = frame.loc[0, "row_key"]
    elif mutation == "empty_id":
        frame.loc[0, "row_key"] = ""
    elif mutation == "nonnumeric":
        frame["noise"] = frame["noise"].astype(object)
        frame.loc[0, "noise"] = "not-a-number"
    elif mutation == "label_copy":
        frame["noise"] = frame["label"].map(spec["label_mapping"])
    else:
        frame.loc[1, spec["feature_columns"]] = frame.loc[0, spec["feature_columns"]]
        frame.loc[1, "label"] = "attack" if frame.loc[0, "label"] == "benign" else "benign"
    write_frame(spec, frame)
    with pytest.raises(DataValidationError, match=match):
        load_numeric_csv(spec)


def test_feature_list_and_label_mapping_are_explicit(tmp_path):
    spec, _ = dataset_spec(tmp_path)
    del spec["feature_columns"]
    with pytest.raises(DataValidationError, match="incomplete"):
        load_numeric_csv(spec)
    spec, frame = dataset_spec(tmp_path)
    frame["target_flag"] = 3.0
    spec["feature_columns"].append("target_flag")
    write_frame(spec, frame)
    with pytest.raises(DataValidationError, match="leakage"):
        load_numeric_csv(spec)
    spec["feature_columns"].remove("target_flag")
    spec["label_mapping"] = {"benign": False, "attack": True}
    with pytest.raises(DataValidationError, match="integer classes"):
        load_numeric_csv(spec)


def test_duplicate_header_is_not_silently_renamed(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("signal,signal,label\n1,2,benign\n3,4,attack\n")
    with pytest.raises(DataValidationError, match="duplicate column"):
        load_numeric_csv({"path": str(path), "feature_columns": ["signal"], "label_column": "label",
                          "label_mapping": {"benign": 0, "attack": 1}})


def test_malformed_extra_csv_fields_rejected(tmp_path):
    spec, _ = dataset_spec(tmp_path)
    Path(spec["path"]).write_text("row_key,signal,secondary,noise,label\na,1,2,3,benign,extra\nb,2,3,4,attack,extra\n")
    with pytest.raises(DataValidationError, match="Cannot parse"):
        load_numeric_csv(spec)


def test_label_mapping_preserves_literal_na_label(tmp_path):
    spec, frame = dataset_spec(tmp_path)
    frame["label"] = frame["label"].replace("benign", "NA")
    spec["label_mapping"] = {"NA": 0, "attack": 1}
    write_frame(spec, frame)
    assert set(load_numeric_csv(spec).y) == {0, 1}


def test_partial_label_copy_is_rejected(tmp_path):
    spec, frame = dataset_spec(tmp_path)
    frame["noise"] = frame["label"].map(spec["label_mapping"]).astype(float)
    frame.loc[0, "noise"] = np.nan
    write_frame(spec, frame)
    with pytest.raises(DataValidationError, match="copy or complement"):
        load_numeric_csv(spec)


def test_group_split_is_deterministic_and_disjoint(tmp_path):
    spec, _ = dataset_spec(tmp_path, groups=True)
    data = load_numeric_csv(spec)
    first = make_partitions(data, SOURCE_SPLIT, 42)
    second = make_partitions(data, SOURCE_SPLIT, 42)
    all_ids = set()
    all_groups = set()
    for name in first:
        assert list(first[name].row_ids) == list(second[name].row_ids)
        assert not all_ids.intersection(first[name].row_ids)
        assert not all_groups.intersection(first[name].groups)
        all_ids.update(first[name].row_ids)
        all_groups.update(first[name].groups)
    assert all_ids == set(data.row_ids)
    assert first["adaptation"].X.empty
    assert all(set(first[name].y) == {0, 1} for name in ("train", "validation", "test"))


def test_target_test_is_identical_with_and_without_adaptation(tmp_path):
    spec, _ = dataset_spec(tmp_path, groups=True)
    data = load_numeric_csv(spec)
    zero = make_partitions(data, SOURCE_SPLIT, 43, required_binary=("test",), label_blind=True)
    assisted = make_partitions(data, {"train": 0.4, "validation": 0.2, "test": 0.2, "adaptation": 0.2}, 43, required_binary=("test",), label_blind=True)
    assert list(zero["test"].row_ids) == list(assisted["test"].row_ids)
    assert set(assisted["test"].groups).isdisjoint(assisted["adaptation"].groups)


def test_target_partition_assignments_do_not_depend_on_labels(tmp_path):
    spec, _ = dataset_spec(tmp_path)
    data = load_numeric_csv(spec)
    fractions = {"train": 0.4, "validation": 0.2, "test": 0.2, "adaptation": 0.2}
    original = make_partitions(data, fractions, 43, required_binary=("test",), label_blind=True)
    changed = replace(data, y=data.y.sample(frac=1, random_state=7).reset_index(drop=True))
    rearranged = make_partitions(changed, fractions, 43, required_binary=("test",), label_blind=True)
    assert all(list(original[name].row_ids) == list(rearranged[name].row_ids) for name in original)


def test_dns_label_counts_and_query_classes_are_legal_predictors(tmp_path):
    spec, frame = dataset_spec(tmp_path)
    frame["num_labels"] = frame["noise"]
    frame["labels"] = frame["secondary"]
    frame["query_class"] = frame["signal"]
    spec["feature_columns"] = ["num_labels", "labels", "query_class"]
    write_frame(spec, frame)
    assert list(load_numeric_csv(spec).X) == spec["feature_columns"]


def test_split_rejects_invalid_fractions_and_infeasible_classes(tmp_path):
    spec, frame = dataset_spec(tmp_path)
    data = load_numeric_csv(spec)
    with pytest.raises(DataValidationError, match="sum to one"):
        make_partitions(data, {**SOURCE_SPLIT, "train": 0.7})
    frame["label"] = "benign"
    frame.loc[0, "label"] = "attack"
    write_frame(spec, frame)
    with pytest.raises(DataValidationError, match="infeasible"):
        make_partitions(load_numeric_csv(spec), SOURCE_SPLIT)


def test_preprocessing_learns_only_training_values():
    train = pd.DataFrame({"signal": [1.0, np.nan, 3.0, 5.0]})
    model = make_model({"name": "lr"}, seed=42)
    model.fit(train, [0, 0, 1, 1])
    assert model.named_steps["imputer"].statistics_[0] == 3
    assert model.named_steps["scaler"].mean_[0] == 3
    before = model.named_steps["imputer"].statistics_.copy()
    model_scores(model, pd.DataFrame({"signal": [100000.0, np.nan]}), "lr")
    np.testing.assert_array_equal(before, model.named_steps["imputer"].statistics_)


def test_ap_and_trapezoid_pr_auc_are_distinct():
    metrics, predictions = binary_metrics([0, 1, 0, 1], [0.4, 0.3, 0.2, 0.1], 0.25)
    assert metrics["average_precision"] != metrics["pr_auc_trapezoid"]
    assert predictions.tolist() == [1, 1, 0, 0]
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_real_data_requires_explicit_human_audit(tmp_path):
    config = transfer_config(tmp_path)
    config["source"]["provenance"]["kind"] = "genuine"
    with pytest.raises(DataValidationError, match="user attestations"):
        run_experiment(config, tmp_path / "runs")


def test_protocol_identity_is_required(tmp_path):
    config = transfer_config(tmp_path)
    del config["source"]["protocol"]
    with pytest.raises(DataValidationError, match="protocol"):
        run_experiment(config, tmp_path / "runs")


def test_transfer_requires_complete_semantic_mapping(tmp_path):
    config = transfer_config(tmp_path)
    del config["compatible_features"][0]["window"]
    with pytest.raises(DataValidationError, match="compatible_features"):
        run_experiment(config, tmp_path / "runs")


def test_same_dataset_cannot_be_transfer_target(tmp_path):
    config = transfer_config(tmp_path)
    config["target"]["path"] = config["source"]["path"]
    with pytest.raises(DataValidationError, match="same dataset"):
        run_experiment(config, tmp_path / "runs")


def test_adaptation_run_has_reproducible_selection_and_unique_artifacts(tmp_path):
    config = transfer_config(tmp_path)
    first = run_experiment(config, tmp_path / "runs")
    second = run_experiment(config, tmp_path / "runs")
    assert first != second and first.is_dir() and second.is_dir()
    selected = json.loads((first / "selected_features.json").read_text())
    assert selected == json.loads((second / "selected_features.json").read_text())
    assert len(selected["A_all"]) == 3
    assert len(selected["B_importance_top_k"]) == len(selected["C_satfs_top_k"]) == 2
    metrics = json.loads((first / "metrics.json").read_text())
    assert len(metrics) == 3 and all(row["threshold"] == 0.5 for row in metrics)
    assert all("average_precision" in row and "pr_auc_trapezoid" in row for row in metrics)
    ids = pd.read_csv(first / "split_ids.csv")
    test_ids = set(ids.query("dataset == 'target' and partition == 'test'")["row_id"])
    adaptation_ids = set(ids.query("dataset == 'target' and partition == 'adaptation'")["row_id"])
    assert test_ids.isdisjoint(adaptation_ids)
    predictions = pd.read_csv(first / "predictions.csv")
    assert set(predictions["row_id"]) == test_ids
    assert not set(predictions["row_id"]).intersection(adaptation_ids)
    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["synthetic_demo_only"]
    assert manifest["target_adaptation_label_usage"] == "unused"
    assert len(list(first.glob("model_*.joblib"))) == 3


@pytest.mark.parametrize("mode", ["within", "zero_shot"])
def test_nonadaptation_modes_leave_satfs_unavailable(tmp_path, mode):
    config = transfer_config(tmp_path, "zero_shot")
    config["mode"] = mode
    if mode == "within":
        del config["target"]
        del config["compatible_features"]
    run_dir = run_experiment(config, tmp_path / "runs")
    ranking = json.loads((run_dir / "feature_ranking.json").read_text())
    assert all(row["divergence"] is None and row["satfs"] is None for row in ranking["features"])
    assert ranking["method"]["js_log_base"] == 2
    selected = json.loads((run_dir / "selected_features.json").read_text())
    assert set(selected) == {"A_all", "B_importance_top_k"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert "C_satfs_top_k" in manifest["unavailable_variants"]


def test_all_fits_and_selection_precede_target_evaluation(tmp_path, monkeypatch):
    import dns_exfiltration.experiments as experiment_module
    import dns_exfiltration.satfs as satfs_module
    from sklearn.pipeline import Pipeline

    config = transfer_config(tmp_path)
    source = load_numeric_csv(config["source"])
    target = load_numeric_csv(config["target"])
    source_parts = make_partitions(source, config["source"]["split"], 42)
    target_parts = make_partitions(target, config["target"]["split"], 43, required_binary=("test",), label_blind=True)
    fit_events = []
    original_fit = Pipeline.fit
    original_rank = satfs_module.rank_features
    original_scores = experiment_module.model_scores

    def observed_fit(estimator, X, y, **kwargs):
        pd.testing.assert_frame_equal(X, source_parts["train"].X[list(X.columns)])
        pd.testing.assert_series_equal(y, source_parts["train"].y)
        fit_events.append("source_only_fit")
        return original_fit(estimator, X, y, **kwargs)

    def observed_rank(estimator, train_X, val_X, val_y, target_adaptation_X=None, seed=42):
        assert len(fit_events) == 1
        pd.testing.assert_frame_equal(target_adaptation_X, target_parts["adaptation"].X)
        pd.testing.assert_series_equal(val_y, source_parts["validation"].y)
        return original_rank(estimator, train_X, val_X, val_y, target_adaptation_X=target_adaptation_X, seed=seed)

    def observed_scores(estimator, X, name):
        assert len(fit_events) == 4  # Ranking estimator followed by independent A/B/C fits.
        pd.testing.assert_frame_equal(X, target_parts["test"].X[list(X.columns)])
        return original_scores(estimator, X, name)

    monkeypatch.setattr(Pipeline, "fit", observed_fit)
    monkeypatch.setattr(satfs_module, "rank_features", observed_rank)
    monkeypatch.setattr(experiment_module, "model_scores", observed_scores)
    run_experiment(config, tmp_path / "runs")
    assert len(fit_events) == 4
