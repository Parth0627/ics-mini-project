"""Synthetic integration demonstration; contains no genuine network observations."""

from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd


def create_demo(root: Path, seed: int = 42, rows: int = 800):
    if rows < 300:
        raise ValueError("Use at least 300 rows for a demonstration with validation and adaptation")
    folder = Path(root) / f"synthetic-{seed}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    names = [f"observable_{i:02d}" for i in range(24)]
    paths = []
    for domain, offset in (("source", 0), ("target", 1)):
        rng = np.random.default_rng(seed + offset)
        X = rng.normal(size=(rows, len(names)))
        y = (1.4 * X[:, 0] + X[:, 1] - 0.8 * X[:, 2] + rng.normal(0, 0.8, rows) > 1).astype(int)
        if offset:
            X[:, 0] += 1.5
            X[:, 4] *= 2
        X[rng.random(X.shape) < 0.01] = np.nan
        frame = pd.DataFrame(X, columns=names)
        frame["row_id"] = [f"{domain}-{i}" for i in range(rows)]
        frame["capture_group"] = [f"{domain}-group-{i // 10}" for i in range(rows)]
        frame["label"] = y
        path = folder / f"{domain}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    def dataset(path):
        return {"path": str(path.resolve()), "protocol": "synthetic_toy", "label_column": "label", "label_mapping": {"0": 0, "1": 1},
                "feature_columns": names, "id_column": "row_id", "group_column": "capture_group",
                "provenance": {"kind": "synthetic_demo", "description": "Generated numeric toy features; no network traffic"},
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "adaptation": 0}}
    config = {"experiment_name": "SYNTHETIC-PIPELINE-DEMONSTRATION", "seed": seed,
              "mode": "zero_shot", "top_k": 10, "model": {"name": "rf", "params": {"n_estimators": 60, "n_jobs": 1}},
              "source": dataset(paths[0]), "target": dataset(paths[1]),
              "compatible_features": [{"name": name, "source_column": name, "target_column": name,
                                       "definition": "Same synthetic normal-variate measurement in each toy domain",
                                       "unit": "arbitrary synthetic units", "window": "one synthetic observation"}
                                      for name in names]}
    (folder / "demo-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config, folder
