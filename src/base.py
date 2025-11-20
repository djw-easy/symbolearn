import re
import math
import datetime
import warnings
import itertools
from time import time
from abc import ABCMeta
from collections import deque
from collections.abc import Callable
from typing import Union, Optional, List, Literal, Tuple


import joblib
import numpy as np
import pandas as pd
import jax.numpy as jnp
from joblib import cpu_count
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted



from src.halloffame import HallOfFame
from src.population import Population
from src.fitness import _fitness_map, Fitness
from src.expression import Expression, ExpressionSet
from src.gpoperator import ExpressionGP, ExpressionSetGP
from src.generator import ExprGenerator, ExprSetGenerator
from src.distributed_backend import DistributedBackend, JoblibBackend
from src.node import Operator, Variable, Constant, _operator_map, NodeContent, DynamicAggregation
from src.utils import (
    _get_n_jobs, _partition_estimators, check_random_state, _idx_model_selection, poisson_sample
)



MAX_INT = np.iinfo(np.int32).max
set_config(display='diagram')


def random_dynamic_aggregation(
    random_state: np.random.RandomState, n_variables: int, aggregation_operators: list
):
    v_start = random_state.randint(0, n_variables-2)
    v_end   = random_state.randint(v_start+1, n_variables)
    op_name = random_state.choice(aggregation_operators)
    return DynamicAggregation(v_start, v_end, op_name, n_variables, aggregation_operators)



def seconds_to_readable(seconds):
    # 转换为timedelta对象
    time_delta = datetime.timedelta(seconds=seconds)
    
    # 提取各个时间单位
    days = time_delta.days
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 构建可读字符串
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:  # 如果没有其他部分，至少显示秒数
        parts.append(f"{seconds}s")
    
    return ":".join(parts)




class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for symbolic estimators.

    This class provides the core functionality for the genetic programming
    framework, including the island model, evolutionary operators, and
    Pareto-based solution selection.

    Parameters
    ----------
    model_selection : {"accuracy", "best", "score"}, default="accuracy"
        The criterion for selecting the best model from the Hall of Fame.
        - "accuracy": Selects the model with the best fitness (lowest error).
        - "best": Selects the model with the best score, which balances
          fitness and complexity.
        - "score": An alias for "best".

    niterations : int, default=100
        The number of generations to evolve.

    maxsize : int, default=30
        The maximum size (number of nodes) of an individual expression tree.

    order : int or tuple of int, optional
        The number of expressions in an ExpressionSet for multi-output tasks.
        If an int, it specifies the exact number of expressions.
        If a tuple (min, max), it specifies the range for the number of
        expressions.

    populations : int, default=31
        The number of populations (islands) in the island model.

    population_size : int, default=27
        The number of individuals in each population.

    tournament_selection_n : int, default=15
        The number of individuals to select for a tournament.

    tournament_selection_p : float, default=0.982
        The probability of selecting the best individual in a tournament.

    stopping_criteria : float or callable, optional
        The stopping criterion for the evolution. If a float, the evolution
        stops when the best fitness reaches this value. If a callable, it
        should take the best individual as input and return True to stop.

    operators : list of str, default=('add', 'sub', 'mul', 'div')
        The set of operators to use in the expressions.

    use_constant : bool, default=True
        Whether to include a constant in the expressions.

    metric : str or Fitness, optional
        The fitness metric to use. Can be a string for a predefined metric
        or a custom Fitness object.

    out_func : str, Operator, or callable, optional
        The output function to apply to the result of the expression.

    migration : bool, default=True
        Whether to perform migration between populations.

    fraction_replaced : float, default=0.00036
        The fraction of individuals to replace during migration.

    hof_migration : bool, default=True
        Whether to perform migration from the Hall of Fame.

    fraction_replaced_hof : float, default=0.0614
        The fraction of individuals to replace during Hall of Fame migration.

    topn : int, default=12
        The number of top individuals to select for migration.

    ncycles_per_iteration : int, default=380
        The number of evolutionary cycles to perform in each generation.

    annealing : bool, default=False
        Whether to use simulated annealing for accepting offspring.

    use_frequency : bool, default=True
        Whether to use frequency-based selection pressure.

    alpha : float, default=3.17
        Initial temperature for simulated annealing (requires `annealing` to be `True`).

    crossover_probability : float, default=0.0259
        The probability of performing crossover.

    add_node : float, default=2.47
        The weight for the 'add_node' mutation operator.

    insert_node : float, default=0.0112
        The weight for the 'insert_node' mutation operator.

    delete_node : float, default=0.870
        The weight for the 'delete_node' mutation operator.

    do_nothing_tree : float, default=0.273
        The weight for the 'do_nothing_tree' mutation operator.

    mutate_constant : float, default=0.0346
        The weight for the 'mutate_constant' mutation operator.

    mutate_operator : float, default=0.293
        The weight for the 'mutate_operator' mutation operator.

    swap_operands : float, default=0.198
        The weight for the 'swap_operands' mutation operator.

    rotate_tree : float, default=4.26
        The weight for the 'rotate_tree' mutation operator.

    hoist_tree : float, default=0.411
        The weight for the 'hoist_tree' mutation operator.

    randomize_tree : float, default=0.502
        The weight for the 'randomize_tree' mutation operator.

    simplify_tree : float, default=0.00209
        The weight for the 'simplify_tree' mutation operator.

    set_crossover_method : {'single_point', 'two_point', 'multi_point'}, default='multi_point'
        The crossover method for ExpressionSet.

    set_crossover_probability : float, default=0.0259
        The probability of performing crossover on an ExpressionSet.

    add_expr : float, default=0.17
        The weight for the 'add_expr' mutation operator for ExpressionSet.

    delete_expr : float, default=0.07
        The weight for the 'delete_expr' mutation operator for ExpressionSet.

    randomize_expr : float, default=1.47
        The weight for the 'randomize_expr' mutation operator for ExpressionSet.

    do_nothing_set : float, default=0.273
        The weight for the 'do_nothing_set' mutation operator for ExpressionSet.

    swap_exprs : float, default=0.148
        The weight for the 'swap_exprs' mutation operator for ExpressionSet.

    randomize_set : float, default=0.000502
        The weight for the 'randomize_set' mutation operator for ExpressionSet.

    simplify_set : float, default=0.00209
        The weight for the 'simplify_set' mutation operator for ExpressionSet.

    mutate_expr : float, default=7.26
        The weight for the 'mutate_expr' mutation operator for ExpressionSet.

    should_simplify : bool, default=True
        Whether to simplify expressions after iteration.

    constants_tolerance : float, default=1e-5
        The tolerance for comparing constants.

    allow_duplicate_expressions : bool, default=True
        Whether to allow duplicate expressions in an ExpressionSet.

    should_optimize_constants : bool, default=True
        Whether to optimize constants in the expressions.

    should_optimize_aggregations : bool, default=True
        Whether to optimize aggregations in the expressions.

    optimizer_algorithm : str, default='L-BFGS-B'
        The algorithm for optimizing constants.

    optimizer_nrestarts : int, default=3
        The number of restarts for the constant optimizer.

    optimizer_probability : float, default=0.14
        The probability of optimizing constants for an individual.

    optimizer_iterations : int, default=8
        The number of iterations for the constant optimizer.

    perturbation_factor : float, default=0.129
        The perturbation factor for mutating constants.

    probability_negate_constant : float, default=0.00743
        The probability of negating a constant during mutation.

    n_jobs : int, default=1
        The number of jobs to run in parallel.

    verbose : int, default=0
        The verbosity level.
    
    warm_start : bool
        Tells fit to continue from where the last call to fit finished.
        If false, each call to fit will be fresh, overwriting previous results.
        Default is `False`.

    random_state : int, optional
        The seed for the random number generator.
    """
    equations_: pd.DataFrame | list[pd.DataFrame] | None
    n_features_in_: int
    feature_names_in_: list[str] | tuple[str]
    hall_of_fame_: HallOfFame
    is_multi_output_: bool
    
    def __init__(self,
                 model_selection: Literal["best", "accuracy", "score"] = "accuracy",
                 *,
                 niterations: int = 100,
                 maxsize: int = 30,
                 order: Optional[Union[int, Tuple[int, int]]]=None,
                 populations: int = 31,
                 population_size: int = 27,
                 tournament_selection_n: int = 15,
                 tournament_selection_p: float = 0.982,
                 stopping_criteria: Optional[Union[float, callable]]= None,
                 operators: List[str] = ('add', 'sub', 'mul', 'div'),
                 aggregation_operators: List[str] = ('mean', 'max', 'min', 'sum'),
                 use_constant: bool = True,
                 use_variable: bool = True,
                 use_aggregation: bool = False,
                 metric: Union[str, Fitness] = None,
                 out_func: Optional[Union[str, Operator, callable]] = None,
                 initial_constants: Optional[Union[int, List[float]]] = None,
                 constant_optimization_delay: int = 0,
                 migration: bool = True,
                 fraction_replaced: float = 0.00036,
                 hof_migration: bool = True,
                 fraction_replaced_hof: float = 0.0614,
                 topn: int = 12,
                 ncycles_per_iteration: int = 380,
                 annealing: bool = False,
                 use_frequency: bool = True,
                 alpha: float = 3.17,
                 crossover_probability: float = 0.0259,
                 add_node = 2.47,
                 insert_node = 0.0312,
                 delete_node = 0.870,
                 do_nothing_tree = 0.273,
                 mutate_constant = 0.346,
                 mutate_variable = 0.142,
                 mutate_operator = 0.036,
                 mutate_aggregation = 0.293,
                 swap_operands = 0.198,
                 rotate_tree = 4.26,
                 hoist_tree=0.411,
                 randomize_tree = 0.502,
                 simplify_tree = 0.00209,
                 set_crossover_method: Literal['single_point', 'two_point', 'multi_point'] = 'multi_point',
                 set_crossover_probability=0.0369,
                 add_expr = 0.07,
                 delete_expr = 0.07, 
                 randomize_expr = 0.147, 
                 do_nothing_set = 0.273, 
                 swap_exprs = 0.048, 
                 randomize_set = 0.0502, 
                 simplify_set = 0.00209,
                 mutate_expr = 7.26,
                 should_simplify: bool = True,
                 constants_tolerance: float = 1e-5,
                 allow_duplicate_expressions: bool = True,
                 should_optimize_constants: bool = True,
                 should_optimize_aggregations: bool = False,
                 optimizer_algorithm: Literal['Nelder-Mead', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 
                                              'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', ] = 'L-BFGS-B',
                 optimizer_nrestarts: int = 2,
                 optimizer_probability: float = 0.14,
                 optimizer_iterations: int = 10,
                 perturbation_factor: float = 0.129,
                 probability_negate_constant: float = 0.00743,
                 batching: bool = False,
                 batch_size: int = 256,
                 batching_strategy: str = 'fixed',
                 batching_params: dict = None,
                 n_jobs: float = 1,
                 verbose: int = 0,
                 warm_start: bool = False,
                 random_state: Optional[int] = None):
        self.model_selection = model_selection
        self.populations = populations
        self.population_size = population_size
        self.migration = migration
        self.fraction_replaced = fraction_replaced
        self.hof_migration = hof_migration
        self.fraction_replaced_hof = fraction_replaced_hof
        self.niterations = niterations
        self.maxsize = maxsize
        self.order = order
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        self.topn = topn
        self.stopping_criteria = stopping_criteria
        self.operators = operators
        self.aggregation_operators = aggregation_operators
        self.use_constant = use_constant
        self.use_variable = use_variable
        self.use_aggregation = use_aggregation
        self.metric = metric
        self.out_func = out_func
        self.initial_constants = initial_constants
        self.constant_optimization_delay = constant_optimization_delay
        self.ncycles_per_iteration = ncycles_per_iteration
        self.annealing = annealing
        self.use_frequency = use_frequency
        self.alpha = alpha
        self.crossover_probability = crossover_probability
        self.add_node = add_node
        self.insert_node = insert_node
        self.delete_node = delete_node
        self.do_nothing_tree = do_nothing_tree
        self.mutate_constant = mutate_constant
        self.mutate_variable = mutate_variable
        self.mutate_operator = mutate_operator
        self.mutate_aggregation = mutate_aggregation
        self.swap_operands = swap_operands
        self.rotate_tree = rotate_tree
        self.hoist_tree = hoist_tree
        self.randomize_tree = randomize_tree
        self.simplify_tree = simplify_tree
        self.set_crossover_method = set_crossover_method
        self.set_crossover_probability = set_crossover_probability
        self.add_expr = add_expr
        self.delete_expr = delete_expr
        self.randomize_expr = randomize_expr
        self.do_nothing_set = do_nothing_set
        self.swap_exprs = swap_exprs
        self.randomize_set = randomize_set
        self.simplify_set = simplify_set
        self.mutate_expr = mutate_expr
        self.should_simplify = should_simplify
        self.constants_tolerance = constants_tolerance
        self.allow_duplicate_expressions = allow_duplicate_expressions
        self.should_optimize_constants = should_optimize_constants
        self.should_optimize_aggregations = should_optimize_aggregations
        self.optimizer_algorithm = optimizer_algorithm
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimizer_probability = optimizer_probability
        self.optimizer_iterations = optimizer_iterations
        self.perturbation_factor = perturbation_factor
        self.probability_negate_constant = probability_negate_constant
        self.batching = batching
        self.batch_size = batch_size
        self.batching_strategy = batching_strategy
        self.batching_params = batching_params
        self.n_jobs = min(_get_n_jobs(n_jobs), self.populations)
        self.verbose = verbose
        self.warm_start = warm_start
        self.random_state = check_random_state(random_state)
        self.is_multi_output_ = self.order is not None
        
        self.expr_mutation_weights = {
            'add_node': add_node,
            'insert_node': insert_node,
            'delete_node': delete_node,
            'do_nothing_tree': do_nothing_tree,
            'mutate_constant': mutate_constant,
            'mutate_variable': mutate_variable,
            'mutate_operator': mutate_operator,
            'mutate_aggregation': mutate_aggregation,
            'swap_operands': swap_operands, 
            'rotate_tree': rotate_tree,
            'hoist_tree': hoist_tree,
            'randomize_tree': randomize_tree,
            'simplify_tree': simplify_tree
        }
        self.set_mutation_weights = {
            'add_expr': add_expr,
            'delete_expr': delete_expr,
            'randomize_expr': randomize_expr,
            'do_nothing_set': do_nothing_set,
            'swap_exprs': swap_exprs,
            'randomize_set': randomize_set,
            'mutate_constant': mutate_constant,
            'simplify_set': simplify_set,
            'mutate_expr': mutate_expr
        }
        
        if (not self.use_variable) and (not self.use_aggregation):
            raise ValueError('Either Variable or Aggregation must be used, but get use_variable=False and use_aggregation=False].')
        
        # 初始化分布式后端
        self._backend_instance = None
        self._metric = self._init_metric(metric)

    def _init_metric(self, metric: Union[str, Fitness]):
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _fitness_map:
                raise ValueError('Unsupported metric: %s' % metric)
            if metric not in self.typical_metrics:
                warnings.warn(f"Fitness function '{metric}' is not a typical function. "
                              f"Please use {', '.join(self.typical_metrics[:-1])}, or {self.typical_metrics[-1]}.")
            _metric = _fitness_map[metric]
        else:
            raise ValueError('Invalid type %s found in `metric`.' % type(metric))
        return _metric

    def _setup_backend(self, backend_instance: DistributedBackend = None):
        """设置分布式计算后端"""
        if isinstance(backend_instance, DistributedBackend):
            if not backend_instance.is_initialized():
                backend_instance.initialize()
            self._backend_instance = backend_instance
            return
        
        if isinstance(self._backend_instance, DistributedBackend):
            if not self._backend_instance.is_initialized():
                self._backend_instance.initialize()
            return 
        
        # 创建后端实例
        self._backend_instance = JoblibBackend(n_jobs=self.n_jobs, verbose=int(self.verbose > 1))
        # 初始化后端
        self._backend_instance.initialize()

    def _teardown_backend(self):
        """清理分布式计算后端"""
        if self._backend_instance is not None:
            self._backend_instance.shutdown()
            self._backend_instance = None

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.
        """
        if run_details is None:
            if self.is_multi_output_:
                print('     |{:^37}|{:^37}|{:^32}'.format('Population Average',
                                                   'Best Individual', 'Progress'))
                print('-' * 5 + '|' + '-' * 37 + '|' + '-' * 37 + '|' + '-' * 33)
                line_format = '{:>4} |{:>7} {:>12} {:>15} |{:>7} {:>12} {:>15} |{:>15}  {:>15}'
                print(line_format.format('Gen', 'Order', 'Complexity', 'Error', 'Order',
                                         'Complexity', 'Error', 'Time Left', 'Time Used'))
            else:
                print('     |{:^28}|{:^28}|{:^32}'.format('Population Average',
                                                   'Best Individual', 'Progress'))
                print('-' * 5 + '|' + '-' * 28 + '|' + '-' * 28 + '|' + '-' * 33)
                line_format = '{:>4} |{:>12} {:>14} |{:>12} {:>14} |{:>15}  {:>15}'
                print(line_format.format('Gen', 'Complexity', 'Error', 'Complexity',
                                         'Error', 'Time Left', 'Time Used'))
        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.niterations - gen) * generation_time
            remaining_time = seconds_to_readable(remaining_time)
            used_time = seconds_to_readable(run_details['total_time'][-1])

            if self.is_multi_output_:
                line_format = '{:4d} |{:7.2f} {:12.2f} {:15g} |{:7d} {:12d} {:15g} |{:>15}  {:>15}'
                print(line_format.format(run_details['generation'][-1],
                                         run_details['average_order'][-1],
                                         run_details['average_size'][-1],
                                         run_details['average_fitness'][-1],
                                         run_details['best_order'][-1],
                                         run_details['best_complexity'][-1],
                                         run_details['best_fitness'][-1],
                                         remaining_time, used_time))
            else:
                line_format = '{:4d} |{:12.2f} {:14g} |{:12d} {:14g} |{:>15}  {:>15}'
                print(line_format.format(run_details['generation'][-1],
                                         run_details['average_size'][-1],
                                         run_details['average_fitness'][-1],
                                         run_details['best_complexity'][-1],
                                         run_details['best_fitness'][-1],
                                         remaining_time, used_time))

    @staticmethod
    def _init_population(population: Population, X, y, seed):
        """并行进化一个Population对象的静态包装器。"""
        pop = population.init_population(X, y, seed)
        return pop

    @staticmethod
    def _evolve_population(population: Population, X, y, seed, ncycles_per_iteration,
                           crossover_probability, tournament_selection_n, tournament_selection_p):
        """并行进化一个Population对象的静态包装器。"""
        pop = population.evolve(X, y, seed, ncycles_per_iteration, crossover_probability, 
                                tournament_selection_n, tournament_selection_p)
        return pop
    
    def _perform_migration(self, gen, random_state):
        """
        执行迁移操作，结合了来自名人堂的精英和来自其他种群的个体。
    
        迁移策略:
        1. 名人堂迁移: 从全局名人堂中选择最优的个体，替换每个种群中的一部分。
           这有助于将全局最优解传播到各个种群，加速收敛。
        2. 种群间迁移: 在种群之间交换精英个体，以维持多样性并探索新的解空间。
    
        参数:
        - gen: 当前的进化代数。
        - random_state: 随机状态对象，用于可复现的随机操作。
        """
        def migration(pop: Population, candidates, candidates_fitness, frac: float):
            mean_number_replaced = int(len(pop) * frac)
            num_replace = poisson_sample(mean_number_replaced, random_state)
            num_replace = min(num_replace, len(candidates))
            num_replace = min(num_replace, len(pop))
            if num_replace > 0:
                # 从候选个体中随机选择个体
                immigrant_indices = random_state.choice(
                    len(candidates), size=num_replace, replace=True
                )
                immigrants = [candidates[i] for i in immigrant_indices]
                immigrants_fitness = [candidates_fitness[i] for i in immigrant_indices]
                
                # 在当前种群中随机选择被替换的位置
                replace_indices = random_state.choice(
                    len(pop), size=num_replace, replace=False
                )
                
                # 执行替换
                for i, idx in enumerate(replace_indices):
                    new_individual = immigrants[i].copy()
                    pop._replace_individual(idx, new_individual, immigrants_fitness[i])

        # 1. 种群间的迁移
        if self.migration:
            for i, pop in enumerate(self._populations):
                # 收集各种群的本地精英
                local_elites_by_pop = []
                local_fitnesses_by_pop = []
                for j, other_pop in enumerate(self._populations):
                    if i==j:
                        continue
                    top_n_indices = other_pop.find_top_n(self.topn, find_best=True)
                    local_elites_by_pop.append([other_pop[j] for j in top_n_indices])
                    local_fitnesses_by_pop.append([other_pop.fitnesses[j] for j in top_n_indices])
                
                migration(pop, local_elites_by_pop, local_fitnesses_by_pop, self.fraction_replaced)
        
        # 2. 从名人堂进行迁移
        if self.hof_migration:
            pareto_front = self.hall_of_fame_.get_pareto_front()
            pareto_front_individuals = list(pareto_front.expression)
            pareto_front_fitnesses = list(pareto_front.error)
            for pop in self._populations:
                migration(
                    pop, pareto_front_individuals, 
                    pareto_front_fitnesses, self.fraction_replaced_hof
                )

    def _init_generator_gpoperator(self, n_variables, variable_names, seed):
        random_state = check_random_state(seed)
        expr_generator = ExprGenerator(
            maxsize=self.maxsize,
            n_variables=n_variables,
            operators=self.operators,
            variable_names=variable_names,
            use_variables=self.use_variable,
            use_constants=self.use_constant,
            use_aggregations=self.use_aggregation,
            aggregation_operators=self.aggregation_operators,
            metric=self.metric, out_func=self.out_func, random_state=random_state
        )
        expr_gpoperator = ExpressionGP(
            generator=expr_generator,
            random_state=random_state,
            mutation_weights=self.expr_mutation_weights,
            constants_tolerance=self.constants_tolerance,
            perturbation_factor=self.perturbation_factor,
            probability_negate_constant=self.probability_negate_constant
        )
        if not self.is_multi_output_:
            return expr_generator, expr_gpoperator

        expr_set_generator = ExprSetGenerator(
            order=self.order,
            maxsize=self.maxsize,
            n_variables=n_variables,
            operators=self.operators,
            variable_names=variable_names,
            use_variables=self.use_variable,
            use_constants=self.use_constant,
            use_aggregations=self.use_aggregation,
            aggregation_operators=self.aggregation_operators,
            metric=self.metric, out_func=self.out_func, random_state=random_state
        )
        expr_set_gpoperator = ExpressionSetGP(
            generator=expr_set_generator,
            gpoperator=expr_gpoperator,
            random_state=random_state,
            set_mutation_weights=self.set_mutation_weights,
            set_crossover_method=self.set_crossover_probability,
        )
        return expr_set_generator, expr_set_gpoperator

    def _run(self, X: jnp.ndarray, y: jnp.ndarray, variable_names: Optional[List[str]] = None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # 设置并启动分布式后端
        self._setup_backend() if not isinstance(self._backend_instance, DistributedBackend) else None
        
        try:
            if self.warm_start:
                check_is_fitted(
                    self, 
                    attributes=['equations_', 'n_features_in_', 
                                'feature_names_in_', 'hall_of_fame_']
                )
                assert self.n_features_in_ == X.shape[1], \
                    'Number of features of the model must match the input. ' \
                    'Model n_features is %s and input n_features is %s.' \
                    % (self.n_features_in_, X.shape[1])
            
            # 预先分发数据以优化性能
            X_future = self._backend_instance.scatter(X)
            y_future = self._backend_instance.scatter(y)

            if self.use_aggregation and X.shape[1] < 2:
                warnings.warn('Dynamic aggregation is not supported for less than two variables.')
                self.use_aggregation = False
    
            self._run_details = {
                'generation': [], 'average_size': [], 'average_fitness': [],
                'best_complexity': [], 'best_fitness': [], 'best_oob_fitness': [],
                'generation_time': [], 'total_time': [], 'best_order': [], 'average_order': []
            }
    
            if self.verbose:
                # Print header fields
                self._verbose_reporter()
            
            # 1. 初始化种群
            ## 1.1 获取当前种群
            if self.warm_start:
                if hasattr(self, '_populations'):
                    for population in self._populations:
                        if not isinstance(population, Population):
                            raise ValueError(f'Invalid population type {type(population)} found in warm_start populations.')
                has_input_populations = hasattr(self, '_populations') and self.warm_start
            ## 1.2 并行初始化所有个体
            else:
                self._populations: List[Population] = []
                seeds = random_state.randint(MAX_INT, size=self.populations)
                for i in range(self.populations):
                    generator, gpoperator = self._init_generator_gpoperator(
                        X.shape[1], variable_names, seeds[i]
                    )
                    self._populations.append(
                        Population(
                            self.population_size,
                            generator=generator,
                            gpoperator=gpoperator,
                            annealing=self.annealing,
                            use_frequency=self.use_frequency,
                            alpha=self.alpha, topn=self.topn,
                            should_simplify=self.should_simplify,
                            optimizer_algorithm=self.optimizer_algorithm,
                            optimizer_nrestarts=self.optimizer_nrestarts,
                            optimizer_iterations=self.optimizer_iterations,
                            optimizer_probability=self.optimizer_probability,
                            batching=self.batching, batch_size=self.batch_size,
                            should_optimize_constants=self.should_optimize_constants,
                            should_optimize_aggregations=self.should_optimize_aggregations
                        )
                    )
                # Initialize populations
                evolve_tasks = [
                    (self._populations[i], X_future, y_future, seeds[i])
                    for i in range(self.populations)
                ]
                self._populations = self._backend_instance.map(BaseSymbolic._init_population, evolve_tasks)
            
            if self.warm_start and (not has_input_populations):
                pareto_front = self.hall_of_fame_.get_pareto_front()
                pop_indexes = random_state.choice(self.populations, size=len(self.hall_of_fame_))
                for i, pop_index in enumerate(pop_indexes):
                    population = self._populations[pop_index]
                    # 在当前种群中随机选择被替换的位置
                    replace_index = population.find_oldest_n(1)[0]
                    population._replace_individual(
                        replace_index, 
                        pareto_front.loc[i, 'expression'], 
                        pareto_front.loc[i, 'error']
                    )
    
            # 3. 使用初始种群更新名人堂
            self.hall_of_fame_ = HallOfFame(self._metric.greater_is_better)
            for population in self._populations:
                for individual, objectives in population.hall_of_fame.entries.values():
                    self.hall_of_fame_.add(individual, objectives[-1])
    
            # --- 主进化循环 ---
            evolve_start_time = time()
            for gen in range(1, self.niterations + 1):
                gen_start_time = time()
                
                seeds = random_state.randint(MAX_INT, size=self.populations)
    
                # 4. 并行进化每个种群
                # 首先，将每个Population对象scatter到工作节点
                pop_futures = [self._backend_instance.scatter(pop) for pop in self._populations]
                
                evolve_tasks = []
                for i in range(self.populations):
                    task_args = (
                        pop_futures[i], X_future, y_future, seeds[i],
                        self.ncycles_per_iteration, self.crossover_probability,
                        self.tournament_selection_n, self.tournament_selection_p
                    )
                    evolve_tasks.append(task_args)
                
                self._populations = self._backend_instance.map(BaseSymbolic._evolve_population, evolve_tasks)
                
                # 5. 使用进化后的种群更新名人堂
                for population in self._populations:
                    for individual, objectives in population.hall_of_fame.entries.values():
                        self.hall_of_fame_.add(individual, objectives[-1])
    
                # 6. 执行迁移
                if self.populations > 1 and gen != self.niterations:
                    self._perform_migration(gen, random_state)
    
                # 7. 记录和报告
                pareto_front = self.hall_of_fame_.get_pareto_front()
                idx = _idx_model_selection(
                    pareto_front, self.model_selection, self._metric.greater_is_better
                )
                best_individual = pareto_front.iloc[idx]
                if self.is_multi_output_:
                    best_order, best_complexity, best_fitness = best_individual.order, best_individual.complexity, best_individual.error
                    self._run_details['best_order'].append(best_order)
                else:
                    best_complexity, best_fitness = best_individual.complexity, best_individual.error
                    self._run_details['best_order'].append(0)
    
                current_individuals = [ind for pop in self._populations for ind in pop.individuals]
                all_fitness = np.ma.masked_invalid(
                    np.hstack([pop.fitnesses for pop in self._populations])
                )
                all_length = np.hstack([pop.sizes for pop in self._populations])
    
                self._run_details['generation'].append(gen)
                if self.is_multi_output_:
                    all_order = [ind.order for ind in current_individuals]
                    self._run_details['average_order'].append(np.mean(all_order))
                else:
                    # Append placeholder for non-multi-output cases
                    self._run_details['average_order'].append(0)
                self._run_details['average_size'].append(np.mean(all_length))
                self._run_details['average_fitness'].append(np.mean(all_fitness))
                self._run_details['best_complexity'].append(best_complexity)
                self._run_details['best_fitness'].append(best_fitness)
                generation_time = time() - gen_start_time
                self._run_details['generation_time'].append(generation_time)
                self._run_details['total_time'].append(time() - evolve_start_time)
    
                if self.verbose:
                    self._verbose_reporter(self._run_details)
    
                # Check for early stopping
                if self.stopping_criteria is not None:
                    if self._metric.greater_is_better:
                        if best_fitness >= self.stopping_criteria:
                            break
                    else:
                        if best_fitness <= self.stopping_criteria:
                            break
            
            pareto_front = self.hall_of_fame_.get_pareto_front()
            self.equations_ = pareto_front['expression']
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = variable_names
            return self
        finally:
            # 确保在退出时关闭后端
            self._teardown_backend()

    def __repr__(self) -> str:
        """
        Print all current equations fitted by the model.
        The string `>>>>` denotes which equation is selected.
        """
        # This part is good, it handles the unfitted case.
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        hof_df = self.get_hof()
        hof_df['expression'] = hof_df['expression'].apply(lambda x: str(x))
        
        # Determine the selected equation (your logic is fine)
        chosen_idx = _idx_model_selection(
            hof_df, self.model_selection, 
            self._metric.greater_is_better
        )

        # --- REVISED PART ---
        # Instead of creating a new DataFrame and calling its repr,
        # we format the string directly for full control.

        output_lines = []
        has_score = "score" in hof_df.columns
        
        # Create header
        symbolic_col_name = 'expression set' if self.is_multi_output_ else 'expression'
        header = f"{'':<4}{'pick':<8}{'score':<10}{symbolic_col_name:<70}{'loss':<12}{'complexity':<10}"
        if not has_score:
            header = f"{'':<4}{'pick':<8}{symbolic_col_name:<70}{'loss':<12}{'complexity':<10}"
        output_lines.append(header)

        # Create each data row
        for i, row in hof_df.iterrows():
            pick_marker = ">>>>" if i == chosen_idx else ""
            
            # Use f-string formatting for precise alignment
            score_str = f"{row.get('score', 0.0):<10.6f}" if has_score else ""
            
            if self.is_multi_output_:
                order_str = f"{int(row.get('order', 0)):<8d}"
                line = (
                    f"{i:<4}"
                    f"{pick_marker:<8}"
                    f"{order_str}"
                    f"{int(row['complexity']):<10}"
                    f"{row['expression']:<70}"
                    f"{row['error']:<12.6f}"
                    f"{score_str}"
                )
            else:
                line = (
                    f"{i:<4}"
                    f"{pick_marker:<8}"
                    f"{int(row['complexity']):<10}"
                    f"{row['expression']:<70}"
                    f"{row['error']:<12.6f}"
                    f"{score_str}"
                )
            output_lines.append(line)
        
        # Join everything together
        output = f"{self.__class__.__name__}.equations_ = [\n"
        output += "\n".join(output_lines)
        output += "\n]"
        
        return output

    def _html_repr(self):
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        try:
            from sklearn.utils._repr_html.estimator import estimator_html_repr
        except ImportError:  # pragma: no cover
            return super()._html_repr()
        
        html = estimator_html_repr(self)

        hof_df = self.get_hof()
        hof_df['expression'] = hof_df['expression'].apply(lambda x: str(x))
        chosen_idx = _idx_model_selection(
            hof_df, self.model_selection, self._metric.greater_is_better
        )
        if self.is_multi_output_:
            display_df = pd.DataFrame({
                "pick": [">>>>" if i == chosen_idx else "" for i in hof_df.index],
                "order": hof_df["order"],
                "complexity": hof_df["complexity"].astype(int),
                "expression set": hof_df["expression"],
                "error": hof_df["error"].round(6)
            })
        else:
            display_df = pd.DataFrame({
                "pick": [">>>>" if i == chosen_idx else "" for i in hof_df.index],
                "complexity": hof_df["complexity"].astype(int),
                "expression":  hof_df["expression"],
                "error": hof_df["error"].round(6)
            })
        if "score" in hof_df.columns:
            display_df.insert(len(display_df.columns), "score", hof_df["score"].round(6))

        # 官方样式：sk-table + sk-table-striped
        table_html = display_df.to_html(
            index=True, escape=True, border=0,
            classes="sk-table sk-table-striped"
        )

        # 官方可折叠模板（注意类名、id、for 要唯一）
        hof_block = f'''
        <div class="sk-item">
            <div class="sk-label-container">
                <div class="sk-label sk-toggleable">
                    <input class="sk-toggleable__control sk-hidden--visually"
                        id="hof-toggle" type="checkbox" checked>
                    <label class="sk-toggleable__label sk-toggleable__label-arrow"
                        for="hof-toggle">Hall of Fame</label>
                    <div class="sk-toggleable__content fitted">
                        {table_html}
                    </div>
                </div>
            </div>
        </div>
        '''

        # 用正则找到“第一层折叠的内容容器”的末尾
        # 官方结构：<div class="sk-estimator ...">...<div class="sk-toggleable__content"> ... </div>
        match = re.search(r'(<div[^>]*class="[^\"]*sk-toggleable__content[^\"]*"[^>]*>.*?</div>)',
                        html, flags=re.DOTALL)
        if match:
            div_end = match.end() - 6
            html = html[:div_end] + hof_block + html[div_end:]

        return html

    def get_hof(self):
        """
        Returns the Pareto front as a pandas DataFrame.

        For single-output problems, the columns are:
        ['complexity', 'error', 'expression']

        For multi-output problems (using ExpressionSet), the columns are:
        ['order', 'complexity', 'error', 'expression']

        Returns
        -------
        pandas.DataFrame
            A DataFrame representing the Pareto front.
            Returns None if pandas is not installed.
        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )
        
        return self.hall_of_fame_.get_pareto_front()
    
    def get_best(self, index: Optional[int] = None) -> pd.Series:
        """
        使用 `model_selection` 策略获取最佳方程。

        Parameters
        ----------
        index : int, optional
            如果希望从名人堂中选择一个特定的方程，请在此处提供行号。
            这会覆盖 `model_selection` 参数。

        Returns
        -------
        best_equation : pandas.Series
            代表找到的最佳表达式的Series。
        """
        check_is_fitted(
            self, 
            attributes=['equations_', 'n_features_in_', 
                        'feature_names_in_', 'hall_of_fame_']
        )

        hof_df = self.get_hof()

        if index is not None:
            if index >= hof_df.shape[0] or index < 0:
                raise IndexError(f'Index out of range [0, {hof_df.shape[0] - 1}].')
            best_individual = hof_df.iloc[index].copy()
            return best_individual

        idx = _idx_model_selection(
            hof_df, self.model_selection, 
            self._metric.greater_is_better
        )
        best_individual = hof_df.iloc[idx].copy()
        
        return best_individual



