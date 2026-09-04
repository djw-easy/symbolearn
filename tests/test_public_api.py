"""Regression tests for the public API surface.

These tests guard against the bug class that motivated them:
``symbolearn/__init__.py`` had been nearly empty, so README claims like
``from symbolearn import SymbolicClassifier`` would raise ``ImportError``
even though the classes existed in ``symbolic_estimators.py``. Each
acceptance test here ensures a README-level import path actually works.
"""

from __future__ import annotations

import pandas as pd


# Canonical README Quick Start import path.
def test_top_level_public_classes_are_importable():
    from symbolearn import (  # noqa: F401 — existence-check, not usage
        Expression,
        ExpressionSet,
        Fitness,
        SymbolicClassifier,
        SymbolicRegressor,
        SymbolicTransformer,
    )


def test_top_level_version_is_a_non_empty_string():
    import symbolearn

    assert isinstance(symbolearn.__version__, str)
    assert symbolearn.__version__, symbolearn.__version__
    # Loose semver-ish: <digits>.<digits>.<digits>...
    parts = symbolearn.__version__.split(".")
    assert len(parts) >= 2
    for p in parts:
        # allow suffixes (e.g. "0a1", "1rc1"); at least numeric prefix.
        digits = "".join(c for c in p if c.isdigit())
        assert digits, symbolearn.__version__


def test_top_level_dunder_all_is_consistent():
    import symbolearn

    assert hasattr(symbolearn, "__all__")
    expected = {
        "Expression",
        "ExpressionSet",
        "Fitness",
        "SymbolicClassifier",
        "SymbolicRegressor",
        "SymbolicTransformer",
        "__version__",
    }
    assert set(symbolearn.__all__) >= expected, symbolearn.__all__


def test_estimators_can_be_constructed_with_minimal_kwargs():
    from symbolearn import (
        SymbolicClassifier,
        SymbolicRegressor,
        SymbolicTransformer,
    )

    common = dict(
        maxsize=5,
        niterations=1,
        populations=2,
        population_size=5,
        ncycles_per_iteration=2,
        n_jobs=2,
        random_state=0,
    )
    SymbolicClassifier(**common)
    SymbolicRegressor(**common)
    SymbolicTransformer(**common)


def test_fitness_class_is_constructable():
    from symbolearn import Fitness

    # Importing is enough; full fitness semantics are exercised in
    # tests/test_fitness.py. Here we only guard the top-level re-export.
    fn = Fitness(lambda y_true, y_pred, sw: 0.0, greater_is_better=True)
    assert fn is not None


def test_public_api_smoke_train_score_hof():
    """A minimal end-to-end smoke that mirrors the README Quick Start.

    Uses sklearn's synthetic tabular data so it stays self-contained
    (no hyperspectral files required) and clears the >1/n_classes bar
    with the same relaxed contract used by .github/workflows/tests.yml.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    from symbolearn import SymbolicClassifier

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=3,
        n_informative=8,
        class_sep=1.5,
        random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = SymbolicClassifier(
        maxsize=10,
        niterations=5,
        populations=5,
        population_size=11,
        ncycles_per_iteration=10,
        n_jobs=2,
        random_state=0,
    )
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    assert acc > 1.0 / 3.0, acc

    hof = clf.get_hof()
    assert isinstance(hof, pd.DataFrame)
    assert {"complexity", "error", "expression"}.issubset(set(hof.columns))

    best = clf.get_best()
    assert "expression" in best.index
    expr_field = best["expression"]
    if isinstance(expr_field, str):
        assert expr_field
    else:
        sub_exprs = list(getattr(expr_field, "expressions", expr_field))
        assert sub_exprs
