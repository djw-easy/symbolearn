"""
Tests for symbolearn.symbolic_estimators — SymbolicRegressor, SymbolicClassifier,
SymbolicTransformer (scikit-learn-compatible API).
"""
import warnings
import numpy as np
from symbolearn.symbolic_estimators import (
    SymbolicRegressor, SymbolicClassifier, SymbolicTransformer,
)


def test_regressor():
    rng = np.random.RandomState(42)
    X_reg = rng.randn(100, 3)
    y_reg = X_reg[:, 0] + 2.0 * X_reg[:, 1] - 0.5 * X_reg[:, 2] + rng.randn(100) * 0.1

    reg = SymbolicRegressor(
        maxsize=7, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=3, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X_reg, y_reg)

    assert hasattr(reg, 'hall_of_fame_')
    assert reg.n_features_in_ == 3
    assert reg.predict(X_reg).shape == (100,)
    assert isinstance(reg.score(X_reg, y_reg), float)
    assert reg.get_hof() is not None


def test_classifier():
    rng = np.random.RandomState(42)
    X_cls = rng.randn(100, 4)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

    clf = SymbolicClassifier(
        maxsize=7, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=3, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_cls, y_cls)

    assert hasattr(clf, 'hall_of_fame_')
    assert clf.classes_ is not None
    assert clf.predict(X_cls).shape == (100,)

    proba = clf.predict_proba(X_cls)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert isinstance(clf.score(X_cls, y_cls), float)
    assert clf.get_hof() is not None


def test_transformer():
    rng = np.random.RandomState(42)
    X_reg = rng.randn(100, 3)
    y_reg = X_reg[:, 0] + 2.0 * X_reg[:, 1] - 0.5 * X_reg[:, 2] + rng.randn(100) * 0.1

    trans = SymbolicTransformer(
        maxsize=5, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=2, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trans.fit(X_reg, y_reg)

    assert hasattr(trans, 'hall_of_fame_')
    transformed = trans.transform(X_reg)
    assert transformed.shape[0] == 100

    transformed_ft = trans.fit_transform(X_reg, y_reg)
    assert transformed_ft.shape == transformed.shape


def test_sklearn_compatibility():
    rng = np.random.RandomState(42)
    X_reg = rng.randn(100, 3)
    y_reg = X_reg[:, 0] + 2.0 * X_reg[:, 1] - 0.5 * X_reg[:, 2] + rng.randn(100) * 0.1

    reg = SymbolicRegressor(
        maxsize=7, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=3, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X_reg, y_reg)

    X_cls = rng.randn(100, 4)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

    clf = SymbolicClassifier(
        maxsize=7, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=3, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_cls, y_cls)

    trans = SymbolicTransformer(
        maxsize=5, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=2, n_jobs=1, random_state=42,
    )

    assert isinstance(reg.get_params(), dict)
    assert isinstance(clf.get_params(), dict)
    assert isinstance(trans.get_params(), dict)

    params = clf.get_params()
    assert 'metric' in params
    assert 'out_func' in params


def test_get_hof_include_dominated():
    rng = np.random.RandomState(42)
    X_cls = rng.randn(100, 4)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

    clf = SymbolicClassifier(
        maxsize=7, niterations=1, populations=1, population_size=15,
        ncycles_per_iteration=3, n_jobs=1, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_cls, y_cls)

    hof_all = clf.get_hof(include_dominated=True)
    assert hof_all is not None
