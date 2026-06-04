"""
Tests for symbolearn.expression — Expression and ExpressionSet.
"""
import numpy as np
from symbolearn.expression import Expression, ExpressionSet
from symbolearn.tree import SymbolicNode, clone_tree
from symbolearn.node import add2, mul2, sub2, div2, sin1, cos1, Constant, Variable, softmax
from symbolearn.fitness import Fitness
from symbolearn.metrics.regression import mean_square_error

tree1 = SymbolicNode(add2, children=[
    SymbolicNode(Variable(0, name="x0")),
    SymbolicNode(Constant(1.0)),
])
tree2 = SymbolicNode(mul2, children=[
    SymbolicNode(Variable(0, name="x0")),
    SymbolicNode(Variable(1, name="x1")),
])
metric = Fitness(mean_square_error, greater_is_better=False)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([2.0, 4.0, 6.0])


def test_expression_metric_stored():
    expr1 = Expression(tree1, metric=metric)
    assert expr1.metric is metric


def test_expression_tree_stored():
    expr1 = Expression(tree1, metric=metric)
    assert expr1.tree is tree1


def test_expression_str():
    expr1 = Expression(tree1, metric=metric)
    assert str(expr1) != "" and isinstance(str(expr1), str)


def test_expression_repr():
    expr1 = Expression(tree1, metric=metric)
    assert repr(expr1) != ""


def test_execute_returns_ndarray():
    e1 = Expression(tree1, metric=metric)
    preds = e1.execute(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_execute_values():
    e1 = Expression(tree1, metric=metric)
    preds = e1.execute(X)
    assert np.allclose(preds, X[:, 0] + 1.0)


def test_execute_tree2():
    e2 = Expression(tree2, metric=metric)
    preds2 = e2.execute(X)
    assert isinstance(preds2, np.ndarray) and preds2.shape[0] == X.shape[0]


def test_fitness_evaluation():
    e1 = Expression(tree1, metric=metric)
    val = metric(e1, X, y)
    assert isinstance(val, float)
    assert np.isclose(val, 0.0, atol=1e-6)


def test_str_non_empty():
    expr1 = Expression(tree1, metric=metric)
    assert len(str(expr1)) > 0


def test_repr_non_empty():
    expr1 = Expression(tree1, metric=metric)
    assert len(repr(expr1)) > 0


def test_copy():
    e3 = Expression(tree1, metric=metric)
    e3_copy = e3.copy()
    assert e3_copy is not e3
    assert e3_copy.tree is not e3.tree
    assert np.allclose(e3_copy.execute(X), e3.execute(X))


def test_size():
    expr1 = Expression(tree1, metric=metric)
    assert expr1.size == tree1.size


def test_eq_same_trees():
    e4a = Expression(tree1, metric=metric)
    e4b = Expression(clone_tree(tree1), metric=metric)
    assert e4a == e4b


def test_eq_different_trees():
    e2 = Expression(tree2, metric=metric)
    e4a = Expression(tree1, metric=metric)
    assert e4a != e2


def test_eq_non_expression_no_crash():
    e4a = Expression(tree1, metric=metric)
    _ = (e4a == "string")


def test_eq_same_tree_diff_metric():
    e4a = Expression(tree1, metric=metric)
    e4c = Expression(tree1, metric=Fitness(mean_square_error, greater_is_better=False))
    assert e4a == e4c


def test_complexity_positive():
    expr1 = Expression(tree1, metric=metric)
    assert expr1.complexity > 0


def test_complexity_int():
    expr1 = Expression(tree1, metric=metric)
    assert isinstance(expr1.complexity, int)


def test_expr_set_len():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    assert len(expr_set) == 2


def test_expr_set_getitem():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    assert expr_set[0] == expr1
    assert expr_set[1] == e2


def test_expr_set_execute():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    evals = expr_set.execute(X)
    assert isinstance(evals, np.ndarray)
    assert evals.shape[1] == 2


def test_expr_set_properties():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    assert expr_set.size == sum(e.size for e in [expr1, e2])
    assert expr_set.complexity > 0
    assert expr_set.order == 2


def test_expr_set_none_placeholder():
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, None], metric=metric)
    assert len(expr_set) == 2
    assert expr_set.order == 1


def test_expr_set_copy():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    es_copy = expr_set.copy()
    assert es_copy is not expr_set
    assert len(es_copy) == len(expr_set)
    assert es_copy[0] is not expr_set[0]


def test_expr_set_str():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    assert len(str(expr_set)) > 0


def test_expr_set_repr():
    e2 = Expression(tree2, metric=metric)
    expr1 = Expression(tree1, metric=metric)
    expr_set = ExpressionSet([expr1, e2], metric=metric)
    assert len(repr(expr_set)) > 0
