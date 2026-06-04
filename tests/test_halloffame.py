"""
Tests for symbolearn.halloffame — Pareto front archive.
"""
import numpy as np
from symbolearn.halloffame import HallOfFame
from symbolearn.expression import Expression
from symbolearn.tree import SymbolicNode
from symbolearn.node import add2, Constant, Variable
from symbolearn.fitness import Fitness
from symbolearn.metrics.regression import mean_square_error

metric = Fitness(mean_square_error, greater_is_better=False)


def make_expr(val):
    t = SymbolicNode(add2, children=[
        SymbolicNode(Variable(0, name="x0")),
        SymbolicNode(Constant(float(val))),
    ])
    ex = Expression(t, metric=metric)
    ex.fitness = float(val)
    return ex


def test_empty():
    hof = HallOfFame(greater_is_better=False)
    assert len(hof) == 0


def test_add():
    hof = HallOfFame(greater_is_better=False)
    for i in range(5):
        e = make_expr(float(i))
        hof.add(e, float(i))
    assert len(hof) >= 1


def test_dominance():
    hof = HallOfFame(greater_is_better=False)
    better = make_expr(1.0)
    worse = make_expr(5.0)
    hof.add(better, 1.0)
    assert len(hof) == 1
    hof.add(worse, 5.0)
    assert len(hof) == 1


def test_non_dominated_front():
    hof = HallOfFame(greater_is_better=False)
    for v in [3.0, 1.0, 2.0, 4.0, 0.5]:
        e = make_expr(v)
        hof.add(e, v)
    assert len(hof) <= 5
    assert len(hof) >= 1


def test_get_pareto_front():
    hof = HallOfFame(greater_is_better=False)
    for v in [3.0, 1.0, 2.0, 4.0, 0.5]:
        e = make_expr(v)
        hof.add(e, v)
    pf = hof.get_pareto_front()
    assert hasattr(pf, 'to_csv')
    assert 'expression' in pf.columns


def test_empty_hof():
    empty = HallOfFame(greater_is_better=False)
    assert len(empty) == 0
    pf_empty = empty.get_pareto_front()
    assert pf_empty is not None and len(pf_empty) == 0
