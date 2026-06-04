"""
Tests for symbolearn.metrics.{regression, classification, transformer}
"""
import numpy as np
from symbolearn.metrics.regression import (
    mean_absolute_error, mean_square_error, root_mean_square_error,
)
from symbolearn.metrics.classification import (
    cross_entropy_loss, hinge_loss, focal_loss, nll_loss, accuracy,
)
from symbolearn.metrics.transformer import (
    weighted_pearson, weighted_spearman, silhouette_loss,
    davies_bouldin_loss, calinski_harabasz_loss, fisher_loss,
    compactness_loss, f_statistic_loss, wasserstein_loss,
    separability_loss,
)


def test_mae():
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    assert np.isclose(mean_absolute_error(y_reg, y_pred), np.mean(np.abs(y_reg - y_pred)))


def test_mse():
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    assert np.isclose(mean_square_error(y_reg, y_pred), np.mean((y_reg - y_pred) ** 2))


def test_rmse():
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    assert np.isclose(root_mean_square_error(y_reg, y_pred), np.sqrt(np.mean((y_reg - y_pred) ** 2)))


def test_mae_nan():
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(float)
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    y_nan = y_reg.copy()
    y_nan[2] = np.nan
    assert np.isnan(mean_absolute_error(y_nan, y_pred))


def test_mse_perfect():
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean_square_error(y_reg, y_reg) == 0.0


def test_cross_entropy_binary():
    y_bin = np.array([0, 1, 0, 1])
    p_bin = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    ce = cross_entropy_loss(y_bin, p_bin)
    assert isinstance(ce, float) and not np.isnan(ce)


def test_cross_entropy_multiclass():
    y_mc = np.array([0, 2, 1])
    p_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    ce_mc = cross_entropy_loss(y_mc, p_mc)
    assert isinstance(ce_mc, float)


def test_focal_loss():
    y_bin = np.array([0, 1, 0, 1])
    p_bin = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    fl = focal_loss(y_bin, p_bin)
    assert isinstance(fl, float)


def test_nll_loss():
    y_mc = np.array([0, 2, 1])
    p_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    nl = nll_loss(y_mc, p_mc)
    assert isinstance(nl, float)


def test_accuracy():
    y_mc = np.array([0, 2, 1])
    p_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    acc = accuracy(y_mc, p_mc)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_hinge_loss():
    y_mc = np.array([0, 2, 1])
    p_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    hl = hinge_loss(y_mc, p_mc)
    assert isinstance(hl, float)


def test_weighted_pearson_perfect():
    rng = np.random.RandomState(42)
    y = rng.randn(100, 1)
    assert np.isclose(weighted_pearson(y, y, None), 1.0)


def test_weighted_spearman_perfect():
    rng = np.random.RandomState(42)
    y = rng.randn(100, 1)
    assert np.isclose(weighted_spearman(y, y, None), 1.0)


def test_silhouette_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    sil = silhouette_loss(labels, y_pred, None)
    assert isinstance(sil, float) and not np.isnan(sil)
    assert -1 <= sil <= 1


def test_davies_bouldin_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    db = davies_bouldin_loss(labels, y_pred, None)
    assert isinstance(db, float) and not np.isnan(db)
    assert db >= 0


def test_calinski_harabasz_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    ch = calinski_harabasz_loss(labels, y_pred, None)
    assert isinstance(ch, float) and not np.isnan(ch)
    assert ch > 0


def test_fisher_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    fi = fisher_loss(labels, y_pred.ravel(), None)
    assert isinstance(fi, float)


def test_compactness_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    co = compactness_loss(labels, y_pred.ravel(), None)
    assert isinstance(co, float)


def test_f_statistic_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    fs = f_statistic_loss(labels, y_pred.ravel(), None)
    assert isinstance(fs, float)


def test_wasserstein_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    wa = wasserstein_loss(labels, y_pred.ravel(), None)
    assert isinstance(wa, float)


def test_separability_loss():
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    y_pred = np.zeros((n, 1))
    y_pred[: n // 2] = rng.randn(n // 2, 1) * 0.1
    y_pred[n // 2:] = rng.randn(n // 2, 1) * 0.1 + 5.0
    se = separability_loss(labels, y_pred.ravel(), None)
    assert isinstance(se, float)
