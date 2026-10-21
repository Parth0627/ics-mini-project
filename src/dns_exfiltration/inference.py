"""Offline inference through an already fitted, local experiment pipeline.

This implementation loads selected numeric columns into memory. It never fits
preprocessing or a classifier, and ignored label columns are not materialized.
Run folders and joblib models are trusted local artifacts from this project;
joblib is a pickle-based serialization format, not an untrusted interchange.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from .data import DataValidationError
from .metrics import model_scores
from .models import MODEL_NAMES


_VARIANTS = {"A_all", "B_importance_top_k", "C_satfs_top_k"}
_MISSING_NUMERIC = {"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none"}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataValidationError(f"Expected an artifact JSON object: {path.name}")
    return value


def predict_csv(
    run_dir: Path,
    csv_path: Path,
    output_path: Path,
    variant: str = "A_all",
    id_column: str | None = None,
) -> dict:
    """Predict unlabeled numeric rows using fixed saved-model thresholds.

    Columns must have the canonical names recorded in selected_features.json.
    Other columns, including labels, are ignored. Infinity and standard numeric
    missing tokens are transformed by the saved training-fitted imputer. The
    output is created exclusively; an existing file is never overwritten.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    manifest = _json(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise DataValidationError("Inference requires a completed experiment run")
    if type(manifest.get("synthetic_demo_only")) is not bool:
        raise DataValidationError("Run manifest must record synthetic_demo_only provenance")
    config = _json(run_dir / "config.json")
    model_spec = config.get("model")
    model_name = model_spec.get("name") if isinstance(model_spec, dict) else None
    if model_name not in MODEL_NAMES:
        raise DataValidationError("Saved configuration has an unsupported model name")
    selections = _json(run_dir / "selected_features.json")
    if variant not in _VARIANTS or variant not in selections:
        raise DataValidationError(f"Saved feature variant {variant!r} is unavailable")
    features = selections[variant]
    if not isinstance(features, list) or not features or any(
        not isinstance(name, str) or not name.strip() for name in features
    ) or len(features) != len(set(features)):
        raise DataValidationError("Saved selected features must be unique, nonempty column names")
    if id_column is not None and (not isinstance(id_column, str) or not id_column.strip()):
        raise DataValidationError("id_column must be a nonempty column name or null")
    configured_labels = {
        spec.get("label_column") for spec in (config.get("source", {}), config.get("target", {}))
        if isinstance(spec, dict)
    } - {None}
    if set(features) & configured_labels:
        raise DataValidationError("Selected inference features cannot include configured label columns")
    if id_column in configured_labels:
        raise DataValidationError("A configured label column cannot be used as an inference row ID")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Inference CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            header = next(csv.reader(stream))
        except StopIteration as error:
            raise DataValidationError("Inference CSV is empty") from error
    if len(header) != len(set(header)):
        raise DataValidationError("Inference CSV contains duplicate column names")
    requested = features + ([id_column] if id_column is not None and id_column not in features else [])
    absent = set(requested) - set(header)
    if absent:
        raise DataValidationError(f"Inference CSV is missing required columns: {sorted(absent)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                csv_path, usecols=requested, dtype=str, encoding="utf-8-sig",
                keep_default_na=False, index_col=False,
            )
    except (pd.errors.ParserError, pd.errors.ParserWarning, UnicodeError, ValueError) as error:
        raise DataValidationError(f"Cannot parse inference CSV: {error}") from error
    if frame.empty:
        raise DataValidationError("Inference CSV contains no data rows")
    if id_column is None:
        row_ids = pd.Series([f"row:{i + 1}" for i in range(len(frame))])
    else:
        row_ids = frame[id_column].str.strip()
        if row_ids.isna().any() or row_ids.eq("").any() or row_ids.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
            raise DataValidationError("Inference row IDs contain missing, empty, or control-character values")
        if row_ids.duplicated().any():
            raise DataValidationError("Inference row IDs must be unique")
    columns = {}
    for feature in features:
        raw = frame[feature]
        missing = raw.str.strip().isin(_MISSING_NUMERIC)
        numeric = pd.to_numeric(raw.mask(missing, np.nan), errors="coerce").astype(float)
        invalid = raw.notna() & ~missing & numeric.isna()
        if invalid.any():
            examples = raw[invalid].drop_duplicates().head(3).tolist()
            raise DataValidationError(f"Inference feature {feature!r} has nonnumeric values: {examples}")
        columns[feature] = numeric.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(columns, columns=features)
    model = joblib.load(run_dir / f"model_{variant}.joblib")
    saved_names = getattr(model, "feature_names_in_", None)
    if saved_names is not None and list(saved_names) != features:
        raise DataValidationError("Saved model feature order does not match selected_features.json")
    scores, threshold, score_kind = model_scores(model, X, model_name)
    if scores.ndim != 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise DataValidationError("Saved model produced invalid inference scores")
    output = pd.DataFrame({
        "row_id": row_ids.to_numpy(), "score": scores,
        "prediction": (scores >= threshold).astype(int), "threshold": threshold,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive open also protects against concurrent writers after validation.
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        output.to_csv(stream, index=False)
    return {
        "rows": len(frame), "variant": variant, "model": model_name,
        "score_kind": score_kind, "threshold": threshold,
        "synthetic_demo_only": manifest["synthetic_demo_only"],
        "provenance_verification": manifest.get("provenance_verification", "user_attestation_only"),
        "provenance": {
            role: {"protocol": spec.get("protocol"), "declaration": spec.get("provenance")}
            for role in ("source", "target")
            if isinstance(spec := config.get(role), dict)
        },
        "allow_real_data": config.get("allow_real_data") is True,
        "confirm_provenance_audit": config.get("confirm_provenance_audit") is True,
        "output_path": str(output_path),
    }
