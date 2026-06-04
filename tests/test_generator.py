"""
Tests for symbolearn.generator — ExprGenerator, ExprSetGenerator.
"""
import numpy as np
from symbolearn.generator import ExprGenerator, ExprSetGenerator
from symbolearn.fitness import Fitness
from symbolearn.metrics.regression import mean_square_error

metric = Fitness(mean_square_error, greater_is_better=False)


def test_gen_init():
    gen = ExprGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,),
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    assert gen is not None
    assert gen.maxsize == 7


def test_gen_generate():
    gen = ExprGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,),
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    for _ in range(10):
        expr = gen.generate_random_expr()
        assert expr is not None
        assert expr.size <= 7


def test_gen_all_generated():
    gen = ExprGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,),
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    exprs = [gen.generate_random_expr() for _ in range(20)]
    assert all(e is not None for e in exprs)
    assert all(e.size <= 7 for e in exprs)


def test_gen_evaluate():
    gen = ExprGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,),
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    exprs = [gen.generate_random_expr() for _ in range(20)]
    X = np.random.RandomState(0).randn(10, 5)
    y = np.random.RandomState(0).randn(10)
    vals = [metric(e, X, y) for e in exprs]
    assert all(isinstance(v, float) for v in vals)


def test_set_gen_init():
    set_gen = ExprSetGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,), order=3,
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    assert hasattr(set_gen, 'maxsize')
    assert set_gen.maxorder == 3


def test_set_gen_generate():
    set_gen = ExprSetGenerator(
        operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
        input_shape=(5,), order=3,
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=7,
    )
    for _ in range(10):
        expr_set = set_gen.generate_random_exprset()
        assert expr_set is not None
        assert len(expr_set) == 3


def test_reproducibility():
    kwargs = dict(
        operators=('add', 'mul'),
        input_shape=(3,),
        use_constants=True, use_variables=True,
        metric=metric, maxsize=7,
    )
    gen_a = ExprGenerator(random_state=42, **kwargs)
    gen_b = ExprGenerator(random_state=42, **kwargs)
    e_a = gen_a.generate_random_expr()
    e_b = gen_b.generate_random_expr()
    assert e_a is not None and e_b is not None
    assert e_a == e_b
