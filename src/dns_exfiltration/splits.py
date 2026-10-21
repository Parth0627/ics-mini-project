"""Deterministic disjoint partitions, with optional group isolation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import DataValidationError, NumericDataset


PARTITION_NAMES = ("train", "validation", "test", "adaptation")


@dataclass(frozen=True)
class Partition:
    X: pd.DataFrame
    y: pd.Series
    row_ids: pd.Series
    groups: pd.Series | None


def make_partitions(
    dataset: NumericDataset,
    fractions: Mapping[str, float],
    seed: int = 42,
    required_binary: Sequence[str] = ("train", "validation", "test"),
    label_blind: bool = False,
) -> dict[str, Partition]:
    if set(fractions) != set(PARTITION_NAMES):
        raise DataValidationError(f"split must specify exactly {list(PARTITION_NAMES)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 or v > 1 for v in fractions.values()):
        raise DataValidationError("Split fractions must be finite numbers between zero and one")
    if not np.isclose(sum(fractions.values()), 1.0, rtol=0, atol=1e-9):
        raise DataValidationError("Split fractions must sum to one")
    if any(name not in PARTITION_NAMES or fractions[name] <= 0 for name in required_binary):
        raise DataValidationError("Required binary partitions must have positive fractions")
    remaining = np.arange(len(dataset.X))
    remaining_mass = 1.0
    indices: dict[str, np.ndarray] = {}
    # Pick holdouts first; training is the final remainder. Each shuffle uses a
    # distinct deterministic seed, and group splits never divide a group.
    order = ("test", "adaptation", "validation", "train")
    try:
        for offset, name in enumerate(order):
            fraction = fractions[name]
            if fraction == 0:
                indices[name] = np.array([], dtype=int)
                continue
            future_positive = any(fractions[other] > 0 for other in order[offset + 1:])
            if not future_positive:
                indices[name] = remaining
                remaining = np.array([], dtype=int)
                remaining_mass -= fraction
                continue
            ratio = fraction / remaining_mass
            if dataset.groups is None:
                rest, selected = train_test_split(
                    remaining, test_size=ratio, random_state=seed + offset,
                    stratify=None if label_blind else dataset.y.iloc[remaining].to_numpy(),
                )
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=ratio, random_state=seed + offset)
                rest_local, selected_local = next(splitter.split(
                    remaining, groups=dataset.groups.iloc[remaining].to_numpy()
                ))
                rest, selected = remaining[rest_local], remaining[selected_local]
            indices[name] = np.sort(selected)
            remaining = rest
            remaining_mass -= fraction
    except ValueError as error:
        raise DataValidationError(f"Requested split is infeasible: {error}") from error
    if len(remaining) or len(np.concatenate(list(indices.values()))) != len(dataset.X):
        raise DataValidationError("Partition assignment did not cover the dataset")
    for name in required_binary:
        if set(dataset.y.iloc[indices[name]].unique()) != {0, 1}:
            raise DataValidationError(
                f"Partition {name!r} must contain both classes; adjust the fixed split seed/fractions or supply more independent groups"
            )
    partitions = {
        name: Partition(
            X=dataset.X.iloc[indices[name]].copy(),
            y=dataset.y.iloc[indices[name]].copy(),
            row_ids=dataset.row_ids.iloc[indices[name]].copy(),
            groups=dataset.groups.iloc[indices[name]].copy() if dataset.groups is not None else None,
        ) for name in PARTITION_NAMES
    }
    for index, name in enumerate(PARTITION_NAMES):
        for other in PARTITION_NAMES[index + 1:]:
            if set(partitions[name].row_ids) & set(partitions[other].row_ids):
                raise DataValidationError("Row IDs overlap across partitions")
            if dataset.groups is not None and set(partitions[name].groups) & set(partitions[other].groups):
                raise DataValidationError("Group IDs overlap across partitions")
    return partitions


def partition_manifest(partitions: Mapping[str, Partition], dataset_role: str) -> pd.DataFrame:
    frames = []
    for name in PARTITION_NAMES:
        partition = partitions[name]
        frames.append(pd.DataFrame({
            "dataset": dataset_role, "partition": name,
            "row_id": partition.row_ids.to_numpy(),
            "group": partition.groups.to_numpy() if partition.groups is not None else "",
        }))
    return pd.concat(frames, ignore_index=True)
