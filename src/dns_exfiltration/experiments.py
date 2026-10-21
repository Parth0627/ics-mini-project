"""Auditable A/B/C feature-selection experiments on user-supplied CSV files."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .data import DataValidationError, NumericDataset, load_numeric_csv
from .metrics import binary_metrics, model_scores
from .models import MODEL_NAMES, make_model
from .splits import make_partitions, partition_manifest


MODES = ("within", "zero_shot", "adaptation_assisted")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _check_provenance(spec: Mapping[str, Any], config: Mapping[str, Any], role: str) -> None:
    if spec.get("protocol") not in {"do53", "doh", "synthetic_toy"}:
        raise DataValidationError(f"{role}.protocol must explicitly be do53, doh, or synthetic_toy")
    provenance = spec.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in {"synthetic_demo", "genuine"}:
        raise DataValidationError(f"{role}.provenance.kind must explicitly be synthetic_demo or genuine")
    if not isinstance(provenance.get("description"), str) or not provenance["description"].strip():
        raise DataValidationError(f"{role}.provenance.description must identify how the data was obtained")
    if provenance["kind"] == "genuine" and (
        config.get("allow_real_data") is not True or config.get("confirm_provenance_audit") is not True
    ):
        raise DataValidationError(
            "Genuine datasets require explicit allow_real_data=true and confirm_provenance_audit=true. "
            "These are user attestations, not automated verification of dataset authenticity."
        )


def _compatible_mapping(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    compatible = config.get("compatible_features")
    if not isinstance(compatible, list) or not compatible:
        raise DataValidationError("Transfer requires explicit compatible_features with semantics, units, and windows")
    required = {"name", "source_column", "target_column", "definition", "unit", "window"}
    for entry in compatible:
        if not isinstance(entry, dict) or set(entry) != required or any(
            not isinstance(entry[key], str) or not entry[key].strip() for key in required
        ):
            raise DataValidationError(f"Every compatible_features entry must provide nonempty {sorted(required)}")
    for key in ("name", "source_column", "target_column"):
        if len({entry[key] for entry in compatible}) != len(compatible):
            raise DataValidationError(f"compatible_features contains duplicate {key} values")
    source_columns = [entry["source_column"] for entry in compatible]
    target_columns = [entry["target_column"] for entry in compatible]
    if config["source"].get("feature_columns") != source_columns or config["target"].get("feature_columns") != target_columns:
        raise DataValidationError("Dataset feature_columns must exactly match the ordered explicit compatibility mapping")
    return (
        {entry["source_column"]: entry["name"] for entry in compatible},
        {entry["target_column"]: entry["name"] for entry in compatible},
    )


def _check_config(config: Mapping[str, Any]) -> None:
    if config.get("mode") not in MODES:
        raise DataValidationError(f"mode must be one of {MODES}")
    if type(config.get("seed", 42)) is not int or config.get("seed", 42) < 0 or config.get("seed", 42) > 2**32 - 4:
        raise DataValidationError("seed must be a nonnegative 32-bit integer")
    if type(config.get("top_k")) is not int or config["top_k"] < 1:
        raise DataValidationError("top_k must be a positive integer")
    if not isinstance(config.get("source"), dict):
        raise DataValidationError("source must be a dataset specification")
    if not isinstance(config.get("model"), dict) or config["model"].get("name") not in MODEL_NAMES:
        raise DataValidationError(f"model.name must be one of {MODEL_NAMES}")
    if not isinstance(config["source"].get("split"), dict) or config["source"]["split"].get("adaptation") != 0:
        raise DataValidationError("Source split must explicitly have adaptation=0")
    _check_provenance(config["source"], config, "source")
    if config["mode"] == "within":
        if config.get("target") is not None:
            raise DataValidationError("within mode must not configure a target dataset")
    else:
        if not isinstance(config.get("target"), dict):
            raise DataValidationError("Transfer modes require a target dataset")
        _check_provenance(config["target"], config, "target")
        target_split = config["target"].get("split")
        if not isinstance(target_split, dict):
            raise DataValidationError("Target must provide an explicit split")
        if config["mode"] == "zero_shot" and target_split.get("adaptation") != 0:
            raise DataValidationError("zero_shot must explicitly have target adaptation=0")
        if config["mode"] == "adaptation_assisted" and (
            not isinstance(target_split.get("adaptation"), (int, float)) or target_split["adaptation"] <= 0
        ):
            raise DataValidationError("adaptation_assisted requires a positive target adaptation fraction")
        _compatible_mapping(config)


def _dataset_manifest(dataset: NumericDataset, spec: Mapping[str, Any], partitions: Mapping[str, Any]) -> dict:
    return {
        "path": str(dataset.path), "file_sha256": dataset.sha256,
        "original_rows": dataset.original_rows, "usable_rows": len(dataset.X),
        "removed_exact_duplicates": dataset.removed_duplicates,
        "nonfinite_values_converted_to_missing": dataset.nonfinite_values,
        "feature_columns": list(dataset.X.columns),
        "group_column": spec.get("group_column"),
        "protocol": spec["protocol"], "requested_split_fractions": spec["split"],
        "split_fraction_unit": "groups" if dataset.groups is not None else "rows",
        "provenance": spec["provenance"],
        "partition_sizes": {name: len(part.X) for name, part in partitions.items()},
    }


def _environment() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost", "shap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    package_dir = Path(__file__).resolve().parent
    code_hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                   for file in sorted(package_dir.glob("*.py"))}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpu_count_visible": os.cpu_count(),
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "source_file_sha256": code_hashes, "packages": versions}


def _validate_ranking(ranking: pd.DataFrame, feature_names: list[str], *, require_shift: bool) -> pd.DataFrame:
    required = ["feature", "importance_raw", "importance", "divergence", "satfs"]
    if not isinstance(ranking, pd.DataFrame) or not set(required).issubset(ranking.columns):
        raise ValueError(f"SATFS rank_features must return columns {required}")
    if ranking["feature"].duplicated().any() or set(ranking["feature"]) != set(feature_names):
        raise ValueError("SATFS ranking must contain every requested feature exactly once")
    if not np.isfinite(ranking[["importance_raw", "importance"]].to_numpy(dtype=float)).all():
        raise ValueError("Feature ranking contains nonfinite source importance scores")
    if require_shift:
        if not np.isfinite(ranking[["divergence", "satfs"]].to_numpy(dtype=float)).all():
            raise ValueError("Adaptation-assisted SATFS ranking contains nonfinite scores")
        if (ranking["divergence"] < 0).any() or (ranking["divergence"] > 1 + 1e-9).any():
            raise ValueError("Base-2 Jensen-Shannon divergence must lie in [0, 1]")
    elif not ranking[["divergence", "satfs"]].isna().all().all():
        raise ValueError("No-adaptation feature rankings must leave divergence and SATFS unavailable")
    return ranking.copy()


def run_experiment(config: dict, output_root: Path) -> Path:
    """Run independently fitted A/B(/C) models under one fixed split and threshold.

    Selection and fitting finish before evaluation features or labels are used.
    Target adaptation features, when enabled, are used only for marginal JSD;
    target labels never fit models or select features. Genuine provenance remains
    an explicit human attestation and is recorded verbatim in the artifacts.
    """
    # Round-trip to prevent mutable callers from changing the audited config.
    config = json.loads(json.dumps(config, allow_nan=False))
    _check_config(config)
    seed = config.get("seed", 42)
    source = load_numeric_csv(config["source"])
    target = None
    if config["mode"] != "within":
        target = load_numeric_csv(config["target"])
        source_map, target_map = _compatible_mapping(config)
        source = replace(source, X=source.X.rename(columns=source_map))
        target = replace(target, X=target.X.rename(columns=target_map))
        if source.sha256 == target.sha256:
            raise DataValidationError("Source and target cannot be the same dataset file")
        combined = pd.concat([source.X, target.X], ignore_index=True)
        if combined.duplicated().any():
            raise DataValidationError("Source and target share exact feature rows; remove cross-dataset contamination before transfer")
    feature_names = list(source.X.columns)
    if config["top_k"] > len(feature_names):
        raise DataValidationError("top_k cannot exceed the number of explicitly configured features")
    source_parts = make_partitions(source, config["source"]["split"], seed=seed)
    target_parts = None
    if target is not None:
        target_parts = make_partitions(
            target, config["target"]["split"], seed=seed + 1, required_binary=("test",), label_blind=True,
        )
        if config["mode"] == "adaptation_assisted" and target_parts["adaptation"].X.empty:
            raise DataValidationError("Target adaptation partition is empty")
    # Validate model parameters before allocating an artifact directory.
    ranking_estimator = make_model(config["model"], seed)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(config.get("experiment_name", "experiment"))).strip("-") or "experiment"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"{slug[:64]}-{timestamp}-seed{seed}-", dir=output_root))
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _environment())
    manifests = {"source": _dataset_manifest(source, config["source"], source_parts)}
    split_frames = [partition_manifest(source_parts, "source")]
    if target is not None and target_parts is not None:
        manifests["target"] = _dataset_manifest(target, config["target"], target_parts)
        split_frames.append(partition_manifest(target_parts, "target"))
    _write_json(run_dir / "datasets.json", manifests)
    pd.concat(split_frames, ignore_index=True).to_csv(run_dir / "split_ids.csv", index=False)
    state = {
        "status": "running", "mode": config["mode"], "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_verification": "user_attestation_only",
        "synthetic_demo_only": all(spec["provenance"]["kind"] == "synthetic_demo" for spec in (config["source"],) + ((config["target"],) if target is not None else ())),
        "threshold_policy": "fixed source-only defaults: probabilities >= 0.5; SVM decision scores >= 0",
        "target_train_validation_usage": "unused",
        "target_adaptation_label_usage": "unused",
        "compatibility_verification": "explicit user metadata; semantics not inferred",
        "target_split_label_usage": "label_blind_random_or_group_split; labels checked only for data integrity and final metric feasibility",
        "unavailable_variants": {} if config["mode"] == "adaptation_assisted" else {
            "C_satfs_top_k": "SATFS requires a separately reserved target adaptation sample; unavailable in this mode"
        },
    }
    _write_json(run_dir / "run_manifest.json", state)
    try:
        from .satfs import rank_features

        train, validation = source_parts["train"], source_parts["validation"]
        started = perf_counter()
        ranking_estimator.fit(train.X, train.y)
        ranking_fit_seconds = perf_counter() - started
        adaptation_X = target_parts["adaptation"].X if config["mode"] == "adaptation_assisted" else None
        started = perf_counter()
        ranking = rank_features(
            ranking_estimator, train.X, validation.X, validation.y,
            target_adaptation_X=adaptation_X, seed=seed,
        )
        ranking_seconds = perf_counter() - started
        ranking = _validate_ranking(ranking, feature_names, require_shift=config["mode"] == "adaptation_assisted")
        ranking.to_csv(run_dir / "feature_ranking.csv", index=False)
        _write_json(run_dir / "feature_ranking.json", {
            "method": ranking.attrs.get("method", {}),
            "distribution_supports": ranking.attrs.get("distribution_supports", {}),
            "features": ranking.astype(object).where(pd.notna(ranking), None).to_dict(orient="records"),
        })
        k = config["top_k"]
        selections = {
            "A_all": feature_names,
            "B_importance_top_k": ranking.sort_values(["importance", "feature"], ascending=[False, True], kind="stable")["feature"].head(k).tolist(),
        }
        if config["mode"] == "adaptation_assisted":
            selections["C_satfs_top_k"] = ranking.sort_values(["satfs", "feature"], ascending=[False, True], kind="stable")["feature"].head(k).tolist()
        _write_json(run_dir / "selected_features.json", selections)
        trained = {}
        fit_times = {}
        for variant, selected in selections.items():
            estimator = make_model(config["model"], seed)
            started = perf_counter()
            estimator.fit(train.X[selected], train.y)
            fit_times[variant] = perf_counter() - started
            trained[variant] = estimator
            joblib.dump(estimator, run_dir / f"model_{variant}.joblib")
        # All feature selection and model fits are now complete. First access
        # to evaluation features/labels in the experimental procedure follows.
        evaluation = source_parts["test"] if target_parts is None else target_parts["test"]
        results = []
        prediction_frames = []
        for variant, selected in selections.items():
            started = perf_counter()
            scores, threshold, score_kind = model_scores(trained[variant], evaluation.X[selected], config["model"]["name"])
            predict_seconds = perf_counter() - started
            metrics, predictions = binary_metrics(evaluation.y, scores, threshold)
            metrics.update({
                "variant": variant, "model": config["model"]["name"],
                "mode": config["mode"], "feature_count": len(selected), "score_kind": score_kind,
                "fit_seconds": fit_times[variant], "predict_seconds": predict_seconds,
                "ranking_fit_seconds": ranking_fit_seconds, "ranking_seconds": ranking_seconds,
            })
            results.append(metrics)
            prediction_frames.append(pd.DataFrame({
                "variant": variant, "dataset": "source" if target_parts is None else "target",
                "row_id": evaluation.row_ids.to_numpy(), "y_true": evaluation.y.to_numpy(),
                "score": scores, "prediction": predictions, "threshold": threshold,
            }))
        _write_json(run_dir / "metrics.json", results)
        pd.DataFrame([{key: value for key, value in result.items() if key != "confusion_matrix"} for result in results]).to_csv(run_dir / "metrics.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
        state["status"] = "completed"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        state["evaluation_dataset"] = "source" if target_parts is None else "target"
        state["ranking_fit_seconds"] = ranking_fit_seconds
        state["ranking_seconds"] = ranking_seconds
        _write_json(run_dir / "run_manifest.json", state)
    except Exception as error:
        state["status"] = "failed"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        _write_json(run_dir / "run_manifest.json", state)
        raise
    return run_dir
