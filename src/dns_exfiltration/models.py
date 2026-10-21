"""Model factories whose preprocessing is fitted only inside training pipelines."""
from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = ("lr", "knn", "svm", "rf", "xgb", "mlp")


def make_model(spec: Mapping[str, Any], seed: int = 42) -> Pipeline:
    name = spec.get("name")
    params = spec.get("params", {})
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    if not isinstance(params, dict):
        raise ValueError("model.params must be a dictionary")
    params = dict(params)
    if "random_state" in params and params["random_state"] != seed:
        raise ValueError("model.params.random_state must equal the experiment seed")
    if name == "lr":
        defaults = {"max_iter": 2000, "random_state": seed}
        factory = LogisticRegression
    elif name == "knn":
        defaults = {"n_neighbors": 5}
        factory = KNeighborsClassifier
    elif name == "svm":
        if params.get("probability", False):
            raise ValueError("SVM uses decision scores and fixed threshold 0; probability must remain false")
        defaults = {"kernel": "rbf", "random_state": seed}
        factory = SVC
    elif name == "rf":
        defaults = {"n_estimators": 200, "random_state": seed, "n_jobs": 1}
        factory = RandomForestClassifier
    elif name == "mlp":
        defaults = {"hidden_layer_sizes": (64, 32), "max_iter": 500, "early_stopping": False, "random_state": seed}
        factory = MLPClassifier
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("xgb requires the optional xgboost dependency") from error
        defaults = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": seed, "n_jobs": 1}
        factory = XGBClassifier
    defaults.update(params)
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in {"lr", "knn", "svm", "mlp"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", factory(**defaults)))
    return Pipeline(steps)
