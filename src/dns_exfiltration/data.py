"""Strict, explicit ingestion of already extracted numeric DNS features."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """A dataset cannot safely be used under the requested protocol."""


@dataclass(frozen=True)
class NumericDataset:
    X: pd.DataFrame
    y: pd.Series
    row_ids: pd.Series
    groups: pd.Series | None
    path: Path
    sha256: str
    original_rows: int
    removed_duplicates: int
    nonfinite_values: int


_LEAKAGE_NAME = re.compile(
    r"(^|_)(classification|target|prediction|predicted|"
    r"split|partition|fold|id|row_id|sample_id|attack_type|is_attack|is_malicious|"
    r"is_benign)(_|$)", re.IGNORECASE
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(values: Any) -> list[str]:
    if not isinstance(values, list) or not values or any(
        not isinstance(name, str) or not name.strip() for name in values
    ):
        raise DataValidationError("feature_columns must be a nonempty explicit list of column names")
    if len(values) != len(set(values)):
        raise DataValidationError("feature_columns contains duplicate names")
    return values


def _identifier(frame: pd.DataFrame, name: str, *, unique: bool) -> pd.Series:
    if name not in frame:
        raise DataValidationError(f"Missing identifier column: {name}")
    raw = frame[name]
    if raw.isna().any():
        raise DataValidationError(f"Identifier column {name!r} contains missing values")
    result = raw.astype(str).str.strip()
    if result.eq("").any() or result.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
        raise DataValidationError(f"Identifier column {name!r} contains empty or control-character IDs")
    if unique and result.duplicated().any():
        raise DataValidationError(f"Identifier column {name!r} must contain unique IDs")
    return result


def load_numeric_csv(spec: Mapping[str, Any]) -> NumericDataset:
    """Load numeric features without guessing labels, IDs, or feature columns.

    Exact duplicate feature/label rows are removed before splitting. The first
    original row ID survives. Conflicting labels for identical features fail.
    Infinity is treated as missing and subsequently imputed from training data.
    """
    try:
        path = Path(spec["path"]).expanduser().resolve()
        label_column = spec["label_column"]
        features = _names(spec["feature_columns"])
        mapping = spec["label_mapping"]
    except (KeyError, TypeError) as error:
        raise DataValidationError(f"Dataset specification is incomplete: {error}") from error
    if not isinstance(label_column, str) or not label_column:
        raise DataValidationError("label_column must be a column name")
    if not isinstance(mapping, dict) or not mapping:
        raise DataValidationError("An explicit label_mapping dictionary is required")
    if any(not isinstance(k, str) or not k.strip() or k != k.strip() for k in mapping):
        raise DataValidationError("label_mapping keys must be nonempty, trimmed strings")
    if any(type(v) is not int or v not in (0, 1) for v in mapping.values()) or set(mapping.values()) != {0, 1}:
        raise DataValidationError("label_mapping must explicitly map labels to both integer classes 0 and 1")
    id_column, group_column = spec.get("id_column"), spec.get("group_column")
    for name in (id_column, group_column):
        if name is not None and (not isinstance(name, str) or not name):
            raise DataValidationError("id_column/group_column must be nonempty names or null")
    forbidden = {label_column, id_column, group_column} - {None}
    for name in features:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
        if name in forbidden or normalized.lower() in {"label", "class"} or _LEAKAGE_NAME.search(normalized):
            raise DataValidationError(f"Potential leakage column cannot be a feature: {name}")
    if not path.is_file():
        raise DataValidationError(f"Dataset file does not exist: {path}")
    # Read the CSV header separately so pandas cannot silently rename duplicate columns.
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            header = next(csv.reader(stream))
        except StopIteration as error:
            raise DataValidationError("Dataset CSV is empty") from error
    if len(header) != len(set(header)):
        raise DataValidationError("Dataset CSV contains duplicate column names")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, index_col=False)
    except (pd.errors.ParserError, pd.errors.ParserWarning, UnicodeError, ValueError) as error:
        raise DataValidationError(f"Cannot parse dataset CSV: {error}") from error
    missing = set(features + [label_column]) - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing requested columns: {sorted(missing)}")
    if frame.empty:
        raise DataValidationError("Dataset contains no data rows")
    raw_labels = frame[label_column]
    if raw_labels.isna().any() or raw_labels.str.strip().eq("").any():
        raise DataValidationError("Labels may not be missing")
    normalized_labels = raw_labels.str.strip()
    unknown = sorted(set(normalized_labels) - set(mapping))
    if unknown:
        raise DataValidationError(f"Unknown labels (provide an explicit mapping): {unknown}")
    y = normalized_labels.map(mapping).astype(np.int64).rename("label")
    if set(y.unique()) != {0, 1}:
        raise DataValidationError("Dataset must contain both mapped classes")
    ids = (_identifier(frame, id_column, unique=True) if id_column else
           pd.Series([f"row:{i + 1}" for i in range(len(frame))], index=frame.index))
    groups = _identifier(frame, group_column, unique=False) if group_column else None
    numeric = {}
    nonfinite_values = 0
    for column in features:
        raw = frame[column]
        missing_numeric = raw.str.strip().isin({"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none"})
        values = pd.to_numeric(raw.mask(missing_numeric, np.nan), errors="coerce").astype(float)
        bad = raw.notna() & ~missing_numeric & values.isna()
        if bad.any():
            examples = raw[bad].drop_duplicates().head(3).tolist()
            raise DataValidationError(f"Feature {column!r} has nonnumeric values: {examples}")
        nonfinite_values += int(np.isinf(values.to_numpy()).sum())
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.isna().all():
            raise DataValidationError(f"Feature {column!r} has no finite numeric values")
        # Catch accidental copies of binary truth even under an innocuous name.
        observed = values.notna()
        if set(y.loc[observed].unique()) == {0, 1} and (
            values.loc[observed].eq(y.loc[observed]).all() or values.loc[observed].eq(1 - y.loc[observed]).all()
        ):
            raise DataValidationError(f"Feature {column!r} is a copy or complement of the label")
        numeric[column] = values
    X = pd.DataFrame(numeric, index=frame.index)
    duplicate_X = X.duplicated(keep=False)
    if duplicate_X.any():
        temporary_label = "__validation_label__"
        while temporary_label in X.columns:
            temporary_label += "_"
        repeated = X.loc[duplicate_X].assign(**{temporary_label: y.loc[duplicate_X]})
        conflicts = repeated.groupby(features, dropna=False, sort=False)[temporary_label].nunique()
        if conflicts.gt(1).any():
            raise DataValidationError("Identical feature rows have conflicting labels")
    joint = X.copy()
    temporary_label = "__deduplication_label__"
    while temporary_label in joint.columns:
        temporary_label += "_"
    joint[temporary_label] = y
    keep = ~joint.duplicated(keep="first")
    removed = int((~keep).sum())
    return NumericDataset(
        X=X.loc[keep].reset_index(drop=True),
        y=y.loc[keep].reset_index(drop=True),
        row_ids=ids.loc[keep].reset_index(drop=True).rename("row_id"),
        groups=groups.loc[keep].reset_index(drop=True).rename("group") if groups is not None else None,
        path=path, sha256=_file_sha256(path), original_rows=len(frame),
        removed_duplicates=removed, nonfinite_values=nonfinite_values,
    )
