"""
Tests for symbolearn.utils — stratified_train_test_split,
extract_and_aggregate_spatial, poisson_sample, _idx_model_selection.
"""
import numpy as np
import pandas as pd
import pytest
from symbolearn.utils import (
    stratified_train_test_split,
    extract_and_aggregate_spatial,
    poisson_sample,
    _idx_model_selection,
)


def test_tabular_shapes():
    rng = np.random.RandomState(42)
    X_tab = rng.randn(100, 4)
    y_tab = np.array([0] * 50 + [1] * 50)
    rng.shuffle(y_tab)
    X_tr, X_te, y_tr, y_te = stratified_train_test_split(X_tab, y_tab, train_size=30, random_state=42)
    assert X_tr.shape[0] == len(y_tr) and X_te.shape[0] == len(y_te)
    assert 50 <= len(y_tr) <= 80
    assert set(y_tr) == {0, 1}
    assert set(y_te) == {0, 1}


def test_map_mode():
    y_map = np.zeros((20, 20), dtype=int)
    y_map[4:8, 4:8] = 1
    y_map[12:16, 12:16] = 2
    X_map = np.random.RandomState(42).randn(20, 20, 5).astype(np.float32)
    X_tr_m, X_te_m, y_tr_m, y_te_m = stratified_train_test_split(
        X_map, y_map, train_size=15, preserve_shape=True, random_state=42
    )
    assert X_tr_m.shape == (20, 20, 5) and y_tr_m.shape == (20, 20)
    assert len(np.unique(y_tr_m[~np.isnan(y_tr_m)])) >= 1

    X_tr_f, X_te_f, y_tr_f, y_te_f = stratified_train_test_split(
        X_map, y_map, train_size=15, preserve_shape=False, random_state=42
    )
    assert X_tr_f.ndim == 2
    assert y_tr_f.ndim == 1
    assert not np.any(np.isnan(X_tr_f))


def test_float_train_size():
    rng = np.random.RandomState(42)
    X_tab = rng.randn(100, 4)
    y_tab = np.array([0] * 50 + [1] * 50)
    rng.shuffle(y_tab)
    X_tr_f2, X_te_f2, y_tr_f2, y_te_f2 = stratified_train_test_split(
        X_tab, y_tab, train_size=0.8, per_class=False, balanced=True, random_state=42
    )
    assert len(y_tr_f2) > 0 and len(y_te_f2) > 0
    assert 60 <= len(y_tr_f2) <= 90


def test_poisson():
    val = poisson_sample(5.0, random_state=np.random.RandomState(42))
    assert isinstance(val, int)
    assert val >= 0


def test_poisson_reproducible():
    val = poisson_sample(5.0, random_state=np.random.RandomState(42))
    val2 = poisson_sample(5.0, random_state=np.random.RandomState(42))
    assert val == val2


def test_spatial_mean():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    X_spat = extract_and_aggregate_spatial(X_3d, window_size=3, method='mean')
    assert X_spat.shape == (6, 6, 5)
    assert np.all(np.isfinite(X_spat))


def test_spatial_std():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    X_std = extract_and_aggregate_spatial(X_3d, window_size=3, method='std')
    assert X_std.shape == (6, 6, 5)


def test_spatial_min():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    X_min = extract_and_aggregate_spatial(X_3d, window_size=3, method='min')
    assert X_min.shape == (6, 6, 5)


def test_spatial_max():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    X_max = extract_and_aggregate_spatial(X_3d, window_size=3, method='max')
    assert X_max.shape == (6, 6, 5)


def test_even_window_raises():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    with pytest.raises((ValueError, AssertionError)):
        extract_and_aggregate_spatial(X_3d, window_size=4, method='mean')


def test_invalid_method_raises():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    with pytest.raises((ValueError, KeyError)):
        extract_and_aggregate_spatial(X_3d, window_size=3, method='invalid')


def test_window_size_1():
    X_3d = np.arange(180, dtype=float).reshape(6, 6, 5)
    assert np.allclose(
        extract_and_aggregate_spatial(X_3d, window_size=1, method='mean'),
        X_3d
    )


def test_idx_model_selection_accuracy():
    hof_df = pd.DataFrame({
        'error': [0.5, 0.3, 0.8, 0.1],
        'score': [0.8, 0.9, 0.6, 0.95],
        'expression': ['expr0', 'expr1', 'expr2', 'expr3'],
    })
    idx = _idx_model_selection(hof_df, model_selection='accuracy', greater_is_better=False)
    assert hof_df.loc[idx, 'error'] == 0.1


def test_idx_model_selection_score():
    hof_df = pd.DataFrame({
        'error': [0.5, 0.3, 0.8, 0.1],
        'score': [0.8, 0.9, 0.6, 0.95],
        'expression': ['expr0', 'expr1', 'expr2', 'expr3'],
    })
    idx = _idx_model_selection(hof_df, model_selection='score', greater_is_better=False)
    assert hof_df.loc[idx, 'score'] == 0.95
