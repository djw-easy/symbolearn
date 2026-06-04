"""
Tests for symbolearn.gpoperator — mutation and crossover.
"""
import numpy as np
from symbolearn.gpoperator import ExpressionGP, ExpressionSetGP
from symbolearn.generator import ExprGenerator, ExprSetGenerator
from symbolearn.fitness import Fitness
from symbolearn.metrics.regression import mean_square_error

metric = Fitness(mean_square_error, greater_is_better=False)

gen = ExprGenerator(
    operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
    input_shape=(5,),
    use_constants=True, use_variables=True,
    random_state=42, metric=metric, maxsize=21,
)
gen2 = ExprGenerator(
    operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
    input_shape=(5,),
    use_constants=True, use_variables=True,
    random_state=99, metric=metric, maxsize=21,
)

mut_weights = {
    'add_node': 2.0, 'mutate_constant': 0.5, 'mutate_operator': 0.5,
    'swap_operands': 0.3, 'delete_node': 0.3, 'simplify_tree': 0.1,
    'hoist_tree': 0.2, 'insert_node': 0.3, 'mutate_aggregation': 0.1,
    'mutate_variable': 0.3,
}

gp = ExpressionGP(generator=gen, mutation_weights=mut_weights, random_state=42)


def test_gp_init():
    assert gp is not None


def test_mutation():
    expr = gen.generate_random_expr()
    if expr is not None:
        gp.mutation(expr)


def test_crossover():
    parent1 = gen.generate_random_expr()
    parent2 = gen2.generate_random_expr()
    if parent1 is not None and parent2 is not None:
        result = gp.crossover(parent1, parent2)
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[2], bool)


def test_set_gp_init():
    set_gen = ExprSetGenerator(
        operators=('add', 'mul'),
        input_shape=(5,), order=3,
        use_constants=True, use_variables=True,
        random_state=42, metric=metric, maxsize=9,
    )
    set_mut_weights = {
        'mutate_expr': 2.0, 'randomize_expr': 1.0, 'do_nothing_set': 0.1,
        'swap_exprs': 1.0, 'add_expr': 1.0, 'delete_expr': 1.0,
        'mutate_constant': 0.5, 'simplify_set': 0.1, 'randomize_set': 0.2,
    }
    set_gp = ExpressionSetGP(
        generator=set_gen, gpoperator=gp,
        set_mutation_weights=set_mut_weights, random_state=42,
    )
    assert set_gp is not None
    set_expr = set_gen.generate_random_exprset()
    if set_expr is not None:
        set_gp.mutation(set_expr)
