import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dns_exfiltration.satfs import NumericSupport, js_divergence, normalize_importance, rank_features


def test_js_divergence_is_not_distance():
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)
    assert js_divergence([2, 4], [1, 2]) == pytest.approx(0)
    assert js_divergence([1, 0], [0.5, 0.5]) == pytest.approx(0.3112781244591328)


@pytest.mark.parametrize("p,q", [([0, 0], [1, 0]), ([-1, 2], [1, 0]), ([1], [1, 0]), ([np.nan], [1])])
def test_invalid_distributions(p, q):
    with pytest.raises(ValueError):
        js_divergence(p, q)


def test_constant_support_retains_tails_missing_and_boundary():
    bins = NumericSupport.fit([5, 5, 5])
    assert np.array_equal(bins.histogram([4, 5, 6, np.nan]), [0.25] * 4)
    assert js_divergence(bins.histogram([5, 5]), bins.histogram([6, 6])) == pytest.approx(1)


def test_all_missing_source_has_explicit_unseen_mass():
    bins = NumericSupport.fit([np.nan, np.inf])
    assert js_divergence(bins.histogram([np.nan]), bins.histogram([5])) == pytest.approx(1)


def test_source_support_is_immutable_and_quantile_tails_visible():
    bins = NumericSupport.fit(np.arange(100), bins=10)
    before = bins.histogram(np.arange(100))
    target = bins.histogram([1000, 2000, 3000])
    assert js_divergence(before, target) == pytest.approx(1)
    assert np.array_equal(before, bins.histogram(np.arange(100)))


def test_near_identical_and_large_finite_masses_are_numerically_safe():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.dirichlet(np.ones(20))
        q = np.maximum(p + rng.normal(0, 1e-15, 20), 0)
        assert 0 <= js_divergence(p, q) < 1e-12
    assert js_divergence([1e308, 1e308], [1, 1]) == pytest.approx(0)
    assert js_divergence([1e-320, 1e-320], [1, 1]) == pytest.approx(0)
    assert 0 <= js_divergence([np.nextafter(0.0, 1.0), 1], [0, 1]) <= 1e-12
    with pytest.raises(ValueError, match="vector"):
        NumericSupport.fit([[1, 2], [3, 4]])


def test_negative_importance_and_degenerate_case():
    assert np.array_equal(normalize_importance([-1, 0, 2, 4]), [0, 0, 0.5, 1])
    with pytest.raises(ValueError, match="degenerate"):
        normalize_importance([-1, 0])


def test_rank_uses_feature_only_adaptation_and_leaves_pipeline_fitted():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["signal", "noise", "stable"])
    y = (X.signal > 0).astype(int)
    pipeline = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
    pipeline.fit(X.iloc[:200], y.iloc[:200])
    coefficients = pipeline[-1].coef_.copy()
    zero = rank_features(pipeline, X.iloc[:200], X.iloc[200:], y.iloc[200:], repeats=2)
    assert zero.divergence.isna().all() and zero.satfs.isna().all()
    target = X.iloc[200:].copy()
    target["signal"] += 50
    shifted = rank_features(pipeline, X.iloc[:200], X.iloc[200:], y.iloc[200:], target, repeats=2)
    assert shifted.loc[shifted.feature == "signal", "divergence"].iloc[0] == pytest.approx(1)
    assert shifted.loc[shifted.feature == "signal", "satfs"].iloc[0] == pytest.approx(0)
    assert np.array_equal(coefficients, pipeline[-1].coef_)
    assert np.array_equal(zero.importance_raw, shifted.importance_raw)
    with pytest.raises(ValueError, match="ordered source columns"):
        rank_features(pipeline, X.iloc[:200], X.iloc[200:], y.iloc[200:], target.assign(label=y.iloc[200:]))
