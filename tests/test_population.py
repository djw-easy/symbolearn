"""
Tests for symbolearn.population — Population (single island) evolution.
"""
import numpy as np
from symbolearn.population import Population
from symbolearn.generator import ExprGenerator
from symbolearn.gpoperator import ExpressionGP
from symbolearn.fitness import Fitness
from symbolearn.metrics.regression import mean_square_error

metric = Fitness(mean_square_error, greater_is_better=False)

gen = ExprGenerator(
    operators=('add', 'mul', 'sub', 'div', 'sin', 'cos'),
    input_shape=(5,),
    use_constants=True, use_variables=True,
    random_state=42, metric=metric, maxsize=21,
)

mut_weights = {
    'add_node': 2.0, 'mutate_constant': 0.5, 'mutate_operator': 0.5,
    'swap_operands': 0.3, 'delete_node': 0.3, 'simplify_tree': 0.1,
    'hoist_tree': 0.2, 'insert_node': 0.3, 'mutate_aggregation': 0.1,
    'mutate_variable': 0.3,
}

gp = ExpressionGP(generator=gen, mutation_weights=mut_weights, random_state=42)

X = np.random.RandomState(0).randn(10, 5)
y = np.random.RandomState(0).randn(10)


def test_pop_init():
    pop = Population(population_size=20, generator=gen, gpoperator=gp)
    assert pop is not None
    assert pop.population_size == 20


def test_pop_after_init():
    pop = Population(population_size=20, generator=gen, gpoperator=gp)
    pop.init_population(X, y, seed=42)
    assert len(pop) == 20
    assert len(pop.hall_of_fame) > 0


def test_find_top_n():
    pop = Population(population_size=20, generator=gen, gpoperator=gp)
    pop.init_population(X, y, seed=42)
    top = pop.find_top_n(5)
    assert isinstance(top, np.ndarray)
    assert len(top) <= 5


def test_tournament_selection():
    pop = Population(population_size=20, generator=gen, gpoperator=gp)
    pop.init_population(X, y, seed=42)
    sel = pop.tournament_selection(np.random.RandomState(42), 3, 0.8)
    assert isinstance(sel, tuple)
    assert len(sel) == 2


def test_evolution():
    pop = Population(population_size=20, generator=gen, gpoperator=gp)
    pop.init_population(X, y, seed=42)
    pop.evolve(X, y, seed=42, ncycles=2, crossover_probability=0.5,
               tournament_selection_n=3, tournament_selection_p=0.8)
    assert len(pop) == 20
    assert len(pop.hall_of_fame) > 0
