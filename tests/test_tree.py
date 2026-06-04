"""
Tests for symbolearn.tree — SymbolicNode, traversal iterators, clone_tree,
count_trees, generate_random_tree, RenderTree.
"""
import numpy as np
import pytest
from symbolearn.tree import (
    SymbolicNode, PreOrderIter, PostOrderIter,
    clone_tree, count_trees, generate_random_tree, RenderTree,
)
from symbolearn.node import add2, sub2, mul2, div2, sin1, cos1, Constant, Variable


def test_leaf_degree():
    leaf = SymbolicNode(Constant(5.0))
    assert leaf.degree == 0


def test_leaf_is_leaf():
    assert SymbolicNode(Constant(5.0)).is_leaf


def test_leaf_is_root():
    assert SymbolicNode(Constant(5.0)).is_root


def test_leaf_size():
    assert SymbolicNode(Constant(5.0)).size == 1


def test_leaf_name():
    leaf = SymbolicNode(Constant(5.0))
    assert "5.0" in leaf.name


def test_root_degree():
    left = SymbolicNode(Variable(0, name="x0"))
    right = SymbolicNode(Constant(1.0))
    root = SymbolicNode(add2, children=[left, right])
    assert root.degree == 2


def test_root_not_leaf():
    left = SymbolicNode(Variable(0, name="x0"))
    right = SymbolicNode(Constant(1.0))
    root = SymbolicNode(add2, children=[left, right])
    assert not root.is_leaf


def test_root_size():
    left = SymbolicNode(Variable(0, name="x0"))
    right = SymbolicNode(Constant(1.0))
    root = SymbolicNode(add2, children=[left, right])
    assert root.size == 3


def test_parent_links():
    left = SymbolicNode(Variable(0, name="x0"))
    right = SymbolicNode(Constant(1.0))
    root = SymbolicNode(add2, children=[left, right])
    assert left.parent is root
    assert right.parent is root


def test_unary_size():
    unary = SymbolicNode(sin1, children=[SymbolicNode(Variable(1, name="x1"))])
    assert unary.size == 2


def test_children_kwarg():
    c1 = SymbolicNode(Variable(0, name="x0"))
    c2 = SymbolicNode(Constant(7.0))
    ch_kw = SymbolicNode(add2, children=[c1, c2])
    assert ch_kw.size == 3
    assert c1.parent is ch_kw


def test_leaf_children_rejected():
    with pytest.raises((ValueError, AssertionError, TypeError)):
        SymbolicNode(Variable(0, name="x0"), children=[SymbolicNode(Variable(1, name="x1"))])


def test_wrong_child_count_rejected():
    with pytest.raises((ValueError, AssertionError, TypeError)):
        SymbolicNode(add2, children=[SymbolicNode(Variable(0, name="x0"))])


def test_preorder_iter():
    a = SymbolicNode(Variable(0, name="x0"))
    b = SymbolicNode(Variable(1, name="x1"))
    r = SymbolicNode(add2, children=[a, b])
    pre = list(PreOrderIter(r))
    assert len(pre) == 3 and pre[0] is r and pre[1] is a


def test_postorder_iter():
    a = SymbolicNode(Variable(0, name="x0"))
    b = SymbolicNode(Variable(1, name="x1"))
    r = SymbolicNode(add2, children=[a, b])
    post = list(PostOrderIter(r))
    assert len(post) == 3 and post[-1] is r


def test_properties():
    v0 = SymbolicNode(Variable(0, name="x0"))
    v1 = SymbolicNode(Variable(1, name="x1"))
    inner = SymbolicNode(add2, children=[v1, SymbolicNode(Constant(3.0))])
    root2 = SymbolicNode(add2, children=[v0, inner])
    assert root2.size == 5
    assert root2.depth == 0
    assert v0.depth == 1


def test_root_ref():
    v0 = SymbolicNode(Variable(0, name="x0"))
    v1 = SymbolicNode(Variable(1, name="x1"))
    inner = SymbolicNode(add2, children=[v1, SymbolicNode(Constant(3.0))])
    root2 = SymbolicNode(add2, children=[v0, inner])
    assert v0.root is root2


def test_leaves():
    v0 = SymbolicNode(Variable(0, name="x0"))
    v1 = SymbolicNode(Variable(1, name="x1"))
    inner = SymbolicNode(add2, children=[v1, SymbolicNode(Constant(3.0))])
    root2 = SymbolicNode(add2, children=[v0, inner])
    leaves = list(root2.leaves)
    assert len(leaves) == 3
    assert all(v.is_leaf for v in leaves)


def test_clone_tree():
    v0 = SymbolicNode(Variable(0, name="x0"))
    v1 = SymbolicNode(Variable(1, name="x1"))
    inner = SymbolicNode(add2, children=[v1, SymbolicNode(Constant(3.0))])
    root2 = SymbolicNode(add2, children=[v0, inner])
    cloned = clone_tree(root2)
    assert cloned.size == root2.size
    assert cloned is not root2


def test_render_tree():
    v0 = SymbolicNode(Variable(0, name="x0"))
    v1 = SymbolicNode(Variable(1, name="x1"))
    inner = SymbolicNode(add2, children=[v1, SymbolicNode(Constant(3.0))])
    root2 = SymbolicNode(add2, children=[v0, inner])
    rendered = list(RenderTree(root2))
    assert len(rendered) == root2.size


def test_count_trees_size_1():
    assert count_trees(1, [0]) >= 1


def test_count_trees_size_3():
    assert count_trees(3, [0, 2]) >= 1


def test_count_trees_size_5():
    assert count_trees(5, [0, 2]) >= 1


def test_generate_random_tree_not_none():
    t = generate_random_tree(7, [0, 2], random_state=42)
    assert t is not None
    assert t.size <= 7


def test_generate_random_tree_reproducible():
    t1 = generate_random_tree(7, [0, 2], random_state=42)
    t2 = generate_random_tree(7, [0, 2], random_state=42)
    assert t1 is not None and t2 is not None
    assert t1.size == t2.size


def test_generate_random_tree_diff_seed():
    t3 = generate_random_tree(7, [0, 2], random_state=99)
    assert t3 is not None


def test_generate_random_tree_infeasible():
    t4 = generate_random_tree(2, [0, 2], random_state=42)
    assert t4 is None
