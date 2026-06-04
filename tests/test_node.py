"""
Tests for symbolearn.node — Variable, Constant, Operator, DynamicAggregation.
"""
import numpy as np
import pytest
from symbolearn.node import (
    Variable, Constant, Operator, DynamicAggregation,
    add2, sub2, mul2, div2, sin1, cos1, tanh1, identity,
    _operator_map, sigmoid, softplus, softmax,
)

X2d = np.array([[1.0, 2.0], [3.0, 4.0]])


def test_variable_name():
    v0 = Variable(0, name="x0")
    assert v0.name == "x0"


def test_variable_degree():
    assert Variable(0, name="x0").degree == 0


def test_variable_eq():
    assert Variable(0, name="x0") == Variable(0, name="x0")
    assert Variable(0, name="x0") != Variable(1, name="x1")


def test_variable_call():
    v0 = Variable(0, name="x0")
    assert np.allclose(v0(X2d), X2d[:, 0])


def test_constant_value():
    c42 = Constant(42.0)
    assert c42.value == 42.0


def test_constant_degree():
    assert Constant(42.0).degree == 0


def test_constant_call():
    assert np.allclose(Constant(42.0)(X2d), 42.0)


def test_constant_eq():
    assert Constant(42.0) == Constant(42.0)
    assert Constant(42.0) != Constant(43.0)


def test_operator_name():
    assert add2.name == "add"


def test_operator_degree():
    assert add2.degree == 2


def test_operator_call():
    assert np.allclose(add2(X2d[:, 0], X2d[:, 1]), X2d[:, 0] + X2d[:, 1])


def test_operator_identity():
    assert add2 == add2
    assert add2 != mul2


def test_numeric_add():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(add2(x, x), 2 * x)


def test_numeric_sub():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(sub2(x, x), 0 * x)


def test_numeric_mul():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(mul2(x, x), x ** 2)


def test_numeric_div():
    x = np.array([1.0, 2.0, 3.0])
    sdiv = div2(x, x)
    assert np.allclose(sdiv, np.ones_like(x)) and np.all(np.isfinite(sdiv))


def test_numeric_sin():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(sin1(x), np.sin(x))


def test_numeric_cos():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(cos1(x), np.cos(x))


def test_numeric_tanh():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(tanh1(x), np.tanh(x))


def test_numeric_identity():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(identity(x), x)


def test_protected_add_overflow():
    big = np.array([1e20, -1e20])
    assert np.all(np.isfinite(add2(big, big)))


def test_protected_div_by_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert np.all(np.isfinite(div2(x, np.zeros_like(x))))


def test_sigmoid_shape():
    x = np.array([1.0, 2.0, 3.0])
    s = sigmoid(x)
    assert s.shape == x.shape


def test_sigmoid_range():
    x = np.array([1.0, 2.0, 3.0])
    s = sigmoid(x)
    assert np.all((s > 0) & (s < 1))


def test_sigmoid_mid():
    assert np.allclose(sigmoid(np.array([0.0])), 0.5)


def test_softplus_positive():
    x = np.array([1.0, 2.0, 3.0])
    assert np.all(softplus(x) > 0)


def test_softmax_2d():
    sm = softmax(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert sm.shape == (2, 2)
    assert np.allclose(sm.sum(axis=1), 1.0)


def test_softmax_1d():
    sm1 = softmax(np.array([1.0, 2.0, 3.0]))
    assert sm1.shape == (3,)
    assert np.allclose(sm1.sum(), 1.0)


def test_softmax_scalar():
    assert np.isclose(softmax(np.array(5.0)), 1.0)


@pytest.mark.parametrize("name", ['add', 'mul', 'sin', 'sigmoid', 'softmax', 'identity', '+', '-', '*', '/'])
def test_operator_map(name):
    assert name in _operator_map


def test_spectral_degree():
    agg = DynamicAggregation(v_start=0, v_end=4, stat_name_spectral='mean', n_variables=5)
    assert agg.degree == 0


def test_spectral_output():
    agg = DynamicAggregation(v_start=0, v_end=4, stat_name_spectral='mean', n_variables=5)
    X_3d = np.arange(60, dtype=float).reshape(3, 4, 5)
    out = agg(X_3d, np.ones((3, 4), dtype=bool))
    assert out is not None and out.size > 0


def test_spectral_eq():
    a1 = DynamicAggregation(v_start=0, v_end=4, stat_name_spectral='mean', n_variables=5)
    a2 = DynamicAggregation(v_start=0, v_end=4, stat_name_spectral='mean', n_variables=5)
    assert a1 == a2


def test_spatial_output_2d():
    X_3d = np.arange(60, dtype=float).reshape(3, 4, 5)
    agg = DynamicAggregation(stat_name_spatial='mean', target_feature=1, window_size=3, n_variables=5)
    out = agg(X_3d)
    assert out.ndim == 2


def test_spatial_no_mask():
    X_3d = np.arange(60, dtype=float).reshape(3, 4, 5)
    agg = DynamicAggregation(stat_name_spatial='mean', target_feature=1, window_size=3, n_variables=5)
    out = agg(X_3d)
    assert out is not None and out.size > 0
