"""Bounded-memory CSV inventory. Does not download data or certify authenticity."""

from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path: Path, label_column: str, chunk_rows: int = 100_000) -> dict:
    path = Path(path).resolve(strict=True)
    if chunk_rows < 1:
        raise ValueError("Chunk size must be positive")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header):
        raise ValueError("Empty or duplicate CSV header columns")
    if label_column not in header:
        raise ValueError(f"Label column {label_column!r} does not exist")
    counts, missing, invalid_numeric, infinite = Counter(), Counter(), Counter(), Counter()
    rows = 0
    peak_chunk_bytes = 0
    for chunk in pd.read_csv(path, chunksize=chunk_rows, dtype=str):
        rows += len(chunk)
        peak_chunk_bytes = max(peak_chunk_bytes, int(chunk.memory_usage(deep=True).sum()))
        counts.update(chunk[label_column].fillna("<MISSING>").astype(str))
        for name in header:
            missing[name] += int(chunk[name].isna().sum())
            if name == label_column:
                continue
            numeric = pd.to_numeric(chunk[name], errors="coerce")
            invalid_numeric[name] += int((numeric.isna() & chunk[name].notna()).sum())
            infinite[name] += int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "rows": rows, "columns": header, "label_column": label_column,
            "raw_label_counts": dict(counts), "missing": dict(missing),
            "non_numeric_values": dict(invalid_numeric), "infinite_values": dict(infinite),
            "estimated_dense_numeric_matrix_bytes_float64": rows * (len(header) - 1) * 8,
            "peak_string_chunk_bytes": peak_chunk_bytes,
            "duplicate_audit": "not performed by streaming inventory; loader checks explicitly selected X+y",
            "provenance_status": "local inventory only; confirm source, license, labels and split independently"}

