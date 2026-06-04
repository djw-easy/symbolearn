"""
Tests for symbolearn.fitness — Fitness with penalty/C regularization.
"""
import numpy as np
from symbolearn.fitness import Fitness
from symbolearn.expression import Expression
from symbolearn.tree import SymbolicNode
from symbolearn.node import add2, Constant, Variable
from symbolearn.metrics.regression import mean_square_error

tree = SymbolicNode(add2, children=[
    SymbolicNode(Variable(0, name="x0")),
    SymbolicNode(Constant(1.0)),
])
expr = Expression(tree, metric=Fitness(mean_square_error, greater_is_better=False))

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 3.0, 4.0, 5.0])


def test_fitness_returns_float():
    fn = Fitness(mean_square_error, greater_is_better=False)
    val = fn(expr, X, y)
    assert isinstance(val, float)


def test_fitness_stores_gib():
    fn = Fitness(mean_square_error, greater_is_better=False)
    assert fn.greater_is_better == False


def test_fitness_stores_loss_function():
    fn = Fitness(mean_square_error, greater_is_better=False)
    assert fn.loss_function is not None


def test_fitness_penalty_none():
    fn = Fitness(mean_square_error, greater_is_better=False)
    assert fn.penalty is None


def test_greater_is_better_returns_float():
    fn = Fitness(mean_square_error, greater_is_better=True)
    val = fn(expr, X, y)
    assert isinstance(val, float)


def test_greater_is_better_flag():
    fn = Fitness(mean_square_error, greater_is_better=True)
    assert fn.greater_is_better == True


def test_greater_is_better_non_negated():
    fn = Fitness(mean_square_error, greater_is_better=True)
    val = fn(expr, X, y)
    assert val >= 0


def test_l2_penalty_larger():
    fn_none = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    fn_l2 = Fitness(mean_square_error, greater_is_better=False, penalty='l2', C=0.1)
    assert fn_l2(expr, X, y) > fn_none(expr, X, y)


def test_l1_penalty_larger():
    fn_none = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    fn_l1 = Fitness(mean_square_error, greater_is_better=False, penalty='l1', C=0.1)
    assert fn_l1(expr, X, y) > fn_none(expr, X, y)


def test_elasticnet_penalty_larger():
    fn_none = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    fn_en = Fitness(mean_square_error, greater_is_better=False, penalty='elasticnet', C=0.1)
    assert fn_en(expr, X, y) > fn_none(expr, X, y)


def test_gib_plus_l2_returns_numeric():
    fn = Fitness(mean_square_error, greater_is_better=True, penalty='l2', C=0.1)
    val = fn(expr, X, y)
    assert isinstance(val, float)


def test_no_constants_no_penalty_diff():
    tree_no_const = SymbolicNode(Variable(0, name="x0"))
    expr_no_const = Expression(tree_no_const, metric=Fitness(mean_square_error, greater_is_better=False))
    fn_none = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    fn_l2 = Fitness(mean_square_error, greater_is_better=False, penalty='l2', C=0.1)
    v1 = fn_none(expr_no_const, X, y)
    v2 = fn_l2(expr_no_const, X, y)
    assert np.isclose(v1, v2)


def test_penalty_is_none():
    fn = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    assert fn.penalty is None


def test_default_c():
    fn = Fitness(mean_square_error, greater_is_better=False, penalty=None, C=1.0)
    assert fn.C == 1.0


def test_c_stored():
    fn = Fitness(mean_square_error, greater_is_better=False, penalty='l2', C=0.1)
    assert fn.C == 0.1
