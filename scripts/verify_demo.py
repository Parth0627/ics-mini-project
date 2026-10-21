"""Recalculate saved metrics and audit the complete synthetic initialization sweep."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dns_exfiltration.metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-list", type=Path)
    args = parser.parse_args()
    run_list = args.run_list or max(Path("data/demo").glob("*/run-paths.json"), key=lambda p: p.stat().st_mtime)
    directories = [Path(p) for p in json.loads(run_list.read_text())]
    matrix, target_sets, source_sets = Counter(), [], []
    rows, manifests = [], {}
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        state = json.loads((directory / "run_manifest.json").read_text())
        assert state["status"] == "completed" and state["synthetic_demo_only"], directory
        assert state["target_adaptation_label_usage"] == "unused"
        metrics = json.loads((directory / "metrics.json").read_text())
        predictions = pd.read_csv(directory / "predictions.csv")
        split_ids = pd.read_csv(directory / "split_ids.csv")
        selected = json.loads((directory / "selected_features.json").read_text())
        ranking = json.loads((directory / "feature_ranking.json").read_text())
        if config["mode"] == "adaptation_assisted":
            assert set(selected) == {"A_all", "B_importance_top_k", "C_satfs_top_k"}
            assert ranking["distribution_supports"]
        else:
            assert set(selected) == {"A_all", "B_importance_top_k"}
            assert all(v["satfs"] is None and v["divergence"] is None for v in ranking["features"])
        for metric in metrics:
            pred = predictions[predictions.variant == metric["variant"]]
            recalculated, classifications = binary_metrics(pred.y_true, pred.score, metric["threshold"])
            assert np.array_equal(classifications, pred.prediction), directory
            for key, value in recalculated.items():
                assert np.allclose(value, metric[key], rtol=1e-12, atol=1e-12), (directory, key)
            assert len(selected[metric["variant"]]) == metric["feature_count"]
            rows.append({"run_id": directory.name, "synthetic_only": True,
                         "k": config["top_k"], **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
        source_sets.append(set(split_ids[split_ids.dataset == "source"].apply(
            lambda r: (r.partition, r.row_id), axis=1)))
        if config["mode"] != "within":
            test = set(split_ids[(split_ids.dataset == "target") & (split_ids.partition == "test")].row_id)
            adaptation = set(split_ids[(split_ids.dataset == "target") & (split_ids.partition == "adaptation")].row_id)
            assert test.isdisjoint(adaptation)
            target_sets.append(test)
            assert set(predictions.row_id) == test
        matrix[(config["model"]["name"], config["top_k"], config["mode"])] += 1
        manifests[directory.name] = hashlib.sha256((directory / "split_ids.csv").read_bytes()).hexdigest()
    assert all(ids == source_sets[0] for ids in source_sets), "Source split changed across comparisons"
    assert all(ids == target_sets[0] for ids in target_sets), "Target test changed across transfer modes"
    expected = {(model, k, mode) for model in ["lr", "knn", "svm", "rf", "xgb", "mlp"]
                for k in [10, 15, 20] for mode in ["within", "zero_shot", "adaptation_assisted"]}
    assert set(matrix) == expected and set(matrix.values()) == {1}, "Incomplete or repeated synthetic sweep"
    report = {"status": "verified_synthetic_integration", "scientific_efficacy_claim": False,
              "genuine_dataset_runs": 0, "large_dataset_downloads": 0,
              "run_list": str(run_list.resolve()), "completed_runs": len(directories),
              "fresh_evaluation_model_fits": len(rows), "models": sorted({v[0] for v in matrix}),
              "top_k": [10, 15, 20], "modes": ["within", "zero_shot", "adaptation_assisted"],
              "metrics_recomputed_from_saved_predictions": True,
              "same_source_partitions_all_comparators": True, "same_target_test_all_transfer_modes": True,
              "target_adaptation_test_disjoint": True, "split_file_sha256": manifests}
    Path("results/initialization_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(rows).to_csv("results/synthetic_summary.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "split_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
