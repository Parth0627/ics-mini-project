"""Experimental SATFS ranking; target labels are deliberately absent from this API.

Importance is measured on source validation after fitting the estimator on source
train. Distribution bins are fitted on raw source train only. See docs/satfs.md.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def js_divergence(p, q) -> float:
    """Base-2 Jensen-Shannon divergence, bounded by [0,1], not its square root."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or p.shape != q.shape or not p.size:
        raise ValueError("Distribution vectors must share a nonempty one-dimensional support")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError("Distribution mass must be finite and nonnegative")
    if p.max() <= 0 or q.max() <= 0:
        raise ValueError("Distribution vectors need positive total mass")
    # Rescale before summing to avoid overflow for large finite count vectors.
    p, q = p / p.max(), q / q.max()
    p, q = p / p.sum(), q / q.sum()
    mixture_mass = p + q
    def kl_bits(distribution):
        positive = distribution > 0
        # log differences avoid ratio underflow; 0 log(0) terms are omitted.
        return np.sum(distribution[positive] *
                      (np.log2(distribution[positive]) - (np.log2(mixture_mass[positive]) - 1)))
    divergence = float((kl_bits(p) + kl_bits(q)) / 2)
    if not np.isfinite(divergence) or divergence < -1e-12 or divergence > 1 + 1e-12:
        raise ValueError("Invalid numerical Jensen-Shannon divergence")
    # Tiny cancellation errors for nearly identical distributions may be negative.
    return min(1.0, max(0.0, divergence))


@dataclass(frozen=True)
class NumericSupport:
    minimum: float | None
    maximum: float | None
    interior: tuple[float, ...]

    @classmethod
    def fit(cls, values, bins: int = 10):
        if bins < 2:
            raise ValueError("Use at least two requested quantile bins")
        x = np.asarray(values, dtype=float)
        if x.ndim != 1 or not x.size:
            raise ValueError("Source training observations must be a nonempty vector")
        x = x[np.isfinite(x)]
        if not x.size:
            # No measured support: all finite target observations go to an unknown bucket.
            return cls(None, None, ())
        low, high = float(x.min()), float(x.max())
        qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        return cls(low, high, tuple(float(v) for v in qs if low < v < high))

    def histogram(self, values) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if x.ndim != 1 or not x.size:
            raise ValueError("Distribution observations must be a nonempty vector")
        missing = ~np.isfinite(x)
        if self.minimum is None:
            return np.asarray([(~missing).sum(), missing.sum()], dtype=float) / x.size
        # Underflow, in-support intervals, overflow, missing: retain tails explicitly.
        index = np.searchsorted(self.interior, np.where(missing, self.minimum, x), side="right") + 1
        index[x < self.minimum] = 0
        index[x > self.maximum] = len(self.interior) + 2
        index[missing] = len(self.interior) + 3
        return np.bincount(index, minlength=len(self.interior) + 4).astype(float) / x.size


def normalize_importance(raw) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Importance must be a finite nonempty vector")
    positive = np.maximum(x, 0)
    if positive.max() == 0:
        raise ValueError("No positive source-validation permutation importance; SATFS ranking is degenerate")
    return positive / positive.max()


def rank_features(estimator, source_train_X: pd.DataFrame,
                  source_validation_X: pd.DataFrame, source_validation_y,
                  target_adaptation_X: pd.DataFrame | None = None,
                  seed: int = 42, bins: int = 10, repeats: int = 5) -> pd.DataFrame:
    """Rank original numeric columns through a fitted source-only pipeline.

    Null divergence/SATFS means strict zero-shot. Adaptation input must contain
    features only, on a partition disjoint from the final target test set.
    Caller owns split integrity; this function cannot prove row independence.
    """
    names = list(source_train_X.columns)
    if not names or len(set(names)) != len(names) or list(source_validation_X.columns) != names:
        raise ValueError("Source train and validation must share unique ordered feature columns")
    if len(source_validation_X) != len(source_validation_y) or len(source_train_X) == 0:
        raise ValueError("Invalid source partitions")
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive")
    if target_adaptation_X is not None:
        if list(target_adaptation_X.columns) != names or len(target_adaptation_X) == 0:
            raise ValueError("Adaptation must share ordered source columns and cannot be empty")
    for frame in (source_train_X, source_validation_X, target_adaptation_X):
        if frame is not None and any(not pd.api.types.is_numeric_dtype(frame[c]) for c in names):
            raise ValueError("This implementation supports audited numeric features only")
    if set(np.unique(source_validation_y)) != {0, 1}:
        raise ValueError("Source validation must contain both binary classes")
    # Positive-class AP is source-only, threshold independent and suited to imbalance.
    result = permutation_importance(estimator, source_validation_X, source_validation_y,
                                    scoring="average_precision", n_repeats=repeats,
                                    random_state=seed, n_jobs=1)
    importance = normalize_importance(result.importances_mean)
    divergence = np.full(len(names), np.nan)
    supports = {}
    if target_adaptation_X is not None:
        for i, name in enumerate(names):
            support = NumericSupport.fit(source_train_X[name], bins=bins)
            p = support.histogram(source_train_X[name])
            q = support.histogram(target_adaptation_X[name])
            divergence[i] = js_divergence(p, q)
            supports[name] = {"minimum": support.minimum, "maximum": support.maximum,
                              "interior": list(support.interior), "source_mass": p.tolist(),
                              "adaptation_mass": q.tolist()}
    out = pd.DataFrame({"feature": names, "importance_raw": result.importances_mean,
                        "importance_std": result.importances_std, "importance": importance,
                        "divergence": divergence, "satfs": importance * (1 - divergence)})
    out.attrs["distribution_supports"] = supports
    out.attrs["method"] = {"importance_scoring": "source_validation_average_precision",
                           "normalization": "clip_negative_then_divide_by_max",
                           "js_log_base": 2, "bins_requested": bins, "permutation_repeats": repeats}
    return out
