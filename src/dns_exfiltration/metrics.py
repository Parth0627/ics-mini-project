"""Binary evaluation with explicitly distinct AP and trapezoidal PR area."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def model_scores(estimator, X, model_name: str) -> tuple[np.ndarray, float, str]:
    if model_name == "svm":
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return scores, 0.0, "decision_function"
    probabilities = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Estimator does not contain exactly one positive class 1")
    return np.asarray(probabilities[:, positive[0]], dtype=float), 0.5, "probability"


def binary_metrics(y_true, scores, threshold: float) -> tuple[dict, np.ndarray]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and scores must be equally sized nonempty vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary evaluation requires both classes 0 and 1")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Evaluation scores and threshold must be finite")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "false_positive_rate": float(fp / (fp + tn)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "evaluation_rows": int(len(labels)),
    }
    return result, predictions
