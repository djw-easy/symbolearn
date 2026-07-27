import re
import math
import datetime
import warnings
import itertools
from time import time
from abc import ABCMeta
from collections import deque
from collections.abc import Callable
from typing import Union, Optional, List, Literal, Tuple, Dict, Any


import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import cpu_count
from sklearn import set_config
from numpy.typing import ArrayLike
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


from src.halloffame import HallOfFame
from src.population import Population
from src.log import LogAnalyzer, EvolutionLogger
from src.fitness import _loss_function_map, Fitness
from src.expression import Expression, ExpressionSet, _UNSET
from src.tree_parser import load_expressions_from_csv
from src.gpoperator import ExpressionGP, ExpressionSetGP
from src.generator import ExprGenerator, ExprSetGenerator
from src.node import Operator, _operator_map, op_name_alias, ZScore
from src.utils import check_random_state, _idx_model_selection, poisson_sample

# Minimum per-output standard deviation for z-score normalisation. Output
# channels with a smaller spread on the training samples are treated as already
# normalised (std = 1, mean = 0) to avoid division by a near-zero value that
# would otherwise blow up the (x - mean) / std transform.
ZSCORE_STD_FLOOR = 1e-6


MAX_INT = np.iinfo(np.int32).max
set_config(display='diagram')


def seconds_to_readable(seconds):
    """Convert a duration in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        A formatted string such as "1d:2h:3m:4s".
    """
    time_delta = datetime.timedelta(seconds=seconds)

    days = time_delta.days
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    # Always show seconds if no larger unit is present
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return ":".join(parts)


def rename_keys_recursive(constraints, key_mapping):
    """Recursively rename keys inside nested constraint structures.

    Parameters
    ----------
    constraints : dict, list, or Any
        Input object to transform. Dictionaries have their keys renamed,
        lists are traversed element-wise, and scalar values are returned
        unchanged.
    key_mapping : dict
        Mapping from original key names to replacement key names.

    Returns
    -------
    dict, list, or Any
        Object with the same nested structure as ``constraints`` but with
        dictionary keys replaced according to ``key_mapping``.
    """
    if isinstance(constraints, dict):
        new_dict = {}
        for key, value in constraints.items():
            new_key = key_mapping.get(key, key)
            new_dict[new_key] = rename_keys_recursive(value, key_mapping)
        return new_dict
    elif isinstance(constraints, list):
        return [rename_keys_recursive(item, key_mapping) for item in constraints]
    else:
        return constraints



class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """Base class for symbolic estimators.

    This class implements the shared infrastructure for symbolic regression
    and symbolic classification models in the project. It manages population
    initialisation, island-model evolution, Hall-of-Fame maintenance,
    optional aggregation operators for hyperspectral data, and post-training
    reporting utilities.

    Parameters
    ----------
    model_selection : {"accuracy", "best", "score"}, default="accuracy"
        Strategy used to choose a representative solution from the Hall of
        Fame.

        - ``"accuracy"`` selects the expression with the best raw fitness.
        - ``"best"`` selects the expression with the best composite score,
          balancing predictive performance and complexity.
        - ``"score"`` is an alias for ``"best"``.

    niterations : int, default=100
        Number of outer evolutionary generations to run.

    maxsize : int, default=21
        Maximum allowed tree size, measured as the total number of nodes in a
        symbolic expression.


    order : int or tuple of int, optional
        Expression-set size configuration for multi-output models.

        - If an integer is provided, every :class:`ExpressionSet` contains
          exactly that many expressions.
        - If a tuple ``(min_order, max_order)`` is provided, the order may
          vary within that inclusive range during evolution.
        - If ``None``, the estimator operates in single-expression mode.

    populations : int, default=31
        Number of islands maintained in the island-model evolutionary scheme.

    population_size : int, default=27
        Number of individuals stored in each island.

    tournament_selection_n : int, default=15
        Number of individuals sampled for each tournament-selection event.

    tournament_selection_p : float, default=0.982
        Probability of choosing the strongest individual from a tournament.
        Lower values increase stochasticity and exploration.

    stopping_criteria : float or callable, optional
        Early-stopping rule. If a float is provided, evolution terminates when
        the selected best fitness reaches that threshold. If a callable is
        provided, it should accept the current best individual and return
        ``True`` to stop training.

    operators : sequence of str, default=('add', 'sub', 'mul', 'div')
        Operator names available when constructing and mutating expressions.

    use_constant : bool, default=True
        Whether ephemeral numeric constants may appear in expressions.

    use_variable : bool, default=True
        Whether raw input variables are allowed as terminals. Disabling this
        is mainly useful when the search should rely exclusively on
        aggregation-derived features.

    spectral_stats : str or list of str, optional
        Spectral aggregation operators that may be inserted into expressions.
        A single string enables one fixed aggregation, while a list enables
        dynamic selection among multiple spectral statistics.

    spatial_stats : str or list of str, optional
        Spatial aggregation operators available to the generator. As with
        ``spectral_stats``, a string denotes a fixed operator and a list
        enables dynamic selection during evolution.

    valid_window_sizes : int or list of int, optional
        Spatial window sizes allowed for spatial aggregation operators. A
        scalar fixes the window size, whereas a list allows the model to
        search over multiple candidate sizes.

    valid_spectral_length : int or tuple of int, optional
        Allowed spectral span for spectral aggregation operators. A single
        integer fixes the aggregation length, while a tuple specifies a valid
        range.

    metric : str or Fitness, optional
        Objective function used to evaluate individuals. A string selects a
        predefined metric from ``_loss_function_map``; a :class:`Fitness`
        instance allows fully custom behaviour.

    metric_params : dict of str to Any, default={}
        Extra keyword arguments forwarded to the underlying metric function
        when a predefined metric is constructed.

    penalty : {'l1', 'l2', 'elasticnet'} or None, default=None
        Optional regularisation mode applied by the :class:`Fitness` object.

    C : float, default=1.0
        Regularisation strength used together with ``penalty``.

    regularize_bias : bool, default=False
        When ``False`` (default), bias-like constants are excluded from
        regularisation.  See :class:`Fitness` for details.  Set to ``True``
        to restore the legacy behaviour where every constant is penalised.

    constraints : dict of str to int or tuple of int, optional
        Per-operator complexity constraints. Unary operators take an integer;
        binary operators take a two-tuple ``(left_max, right_max)``. For
        example, ``{'pow': (-1, 1)}`` allows any complexity on the base but
        restricts the exponent subtree to complexity 1.

    nested_constraints : dict of str to dict of str to int, optional
        Limits how often one operator may be nested inside another. This is
        useful for preventing pathological constructions such as repeated
        trigonometric nesting or deeply stacked aggregation operators.

    out_func : str, Operator, or callable, optional
        Unary output transformation applied to the final expression output.
        This is commonly used for logistic or bounded-output models.

    initial_constants : int or list of float, optional
        Initial constant pool exposed to the generator. An integer may be used
        by downstream generator logic to sample that many constants, whereas a
        list provides explicit starting values.

    complexity_of_operators : dict of str to float, optional
        Custom complexity cost assigned to individual operators. When
        provided, these values override the default unit complexity used in
        Pareto comparisons.

    complexity_of_constants : float, default=1.0
        Complexity contribution assigned to constant terminals.

    complexity_of_variables : float or list of float, default=1.0
        Complexity contribution assigned to variable terminals. A scalar uses
        the same cost for every feature; a list allows per-feature costs.

    complexity_of_aggregations : float, default=1.0
        Complexity contribution assigned to spatial or spectral aggregation
        nodes.

    migration : bool, default=True
        Whether to perform migration between islands.

    fraction_replaced : float, default=0.0136
        Expected fraction of each population replaced during inter-island
        migration.


    hof_migration : bool, default=True
        Whether Pareto-front individuals from the global Hall of Fame may be
        injected back into populations.

    fraction_replaced_hof : float, default=0.0614
        Expected fraction of each population replaced during Hall-of-Fame
        migration.

    topn : int, default=12
        Number of elite individuals considered as migration candidates from
        each island.

    ncycles_per_iteration : int, default=380
        Number of steady-state evolutionary update cycles executed inside each
        outer generation.

    annealing : bool, default=False
        Whether to use simulated annealing when deciding whether to accept
        non-improving offspring.

    parsimony_coefficient : float, default=0.92
        Complexity penalty coefficient passed to expression generators and used
        when computing composite selection scores.

    alpha : float, default=3.17
        Initial simulated-annealing temperature. Only relevant when
        ``annealing=True``.

    crossover_probability : float, default=0.0259
        Probability of applying crossover instead of mutation for
        single-expression individuals.

    add_node : float, default=2.47
        Relative weight of the ``add_node`` mutation.

    insert_node : float, default=0.312
        Relative weight of the ``insert_node`` mutation.

    delete_node : float, default=0.870
        Relative weight of the ``delete_node`` mutation.

    do_nothing_tree : float, default=0.273
        Relative weight of the no-op mutation for single expressions.

    mutate_constant : float, default=0.046
        Relative weight of constant-value mutation.

    mutate_variable : float, default=0.042
        Relative weight of terminal-variable replacement mutation.

    mutate_operator : float, default=0.026
        Relative weight of operator-substitution mutation.

    mutate_aggregation : float, default=0.093
        Relative weight of aggregation-configuration mutation.

    swap_operands : float, default=0.198
        Relative weight of operand-swapping mutation for binary operators.

    rotate_tree : float, default=4.26
        Relative weight of subtree rotation mutation.

    hoist_tree : float, default=0.111
        Relative weight of hoist mutation, which replaces a subtree with one
        of its descendants.

    randomize_tree : float, default=0.502
        Relative weight of replacing a subtree with a newly generated random
        subtree.

    simplify_tree : float, default=0.00209
        Relative weight of algebraic simplification mutation for single
        expressions.

    set_crossover_method : {'single_point', 'two_point', 'multi_point'}, \
            default='multi_point'
        Crossover scheme used when evolving :class:`ExpressionSet`
        individuals.

    set_crossover_probability : float, default=0.0369
        Probability of applying crossover to expression sets.

    add_expr : float, default=0.07
        Relative weight of adding one expression to an expression set.

    delete_expr : float, default=0.07
        Relative weight of deleting one expression from an expression set.

    randomize_expr : float, default=0.147
        Relative weight of regenerating a single expression inside an
        expression set.

    do_nothing_set : float, default=0.273
        Relative weight of the no-op mutation for expression sets.

    swap_exprs : float, default=0.048
        Relative weight of swapping positions of expressions inside an
        expression set.

    randomize_set : float, default=0.0502
        Relative weight of regenerating an entire expression set.

    simplify_set : float, default=0.00209
        Relative weight of expression-set simplification.

    mutate_expr : float, default=7.26
        Relative weight of mutating one constituent expression inside an
        expression set.

    should_simplify : bool, default=True
        Whether symbolic simplification is applied after evolution steps when
        supported by the underlying expression objects.

    constants_tolerance : float, default=1e-5
        Numerical tolerance used when comparing constants for merging or
        simplification.

    allow_duplicate_expressions : bool, default=True
        Whether identical expressions may coexist inside an
        :class:`ExpressionSet`.

    should_optimize_constants : bool, default=True
        Whether numeric constants should be locally optimised during
        evolution.

    should_optimize_aggregations : bool, default=True
        Whether aggregation hyperparameters should be locally optimised when
        possible.

    optimizer_algorithm : str, default='L-BFGS-B'
        Optimisation algorithm used for local constant refinement.

    optimizer_nrestarts : int, default=2
        Number of random restarts for the local optimiser.

    optimizer_probability : float, default=0.0371
        Probability that a candidate undergoes local parameter optimisation.

    optimizer_iterations : int, default=15
        Maximum number of iterations for each optimisation attempt.


    perturbation_factor : float, default=0.129
        Scale of random perturbations applied when mutating constants.

    probability_negate_constant : float, default=0.00743
        Probability of flipping the sign of a constant during mutation.

    batching : bool, default=False
        Whether fitness evaluation may use mini-batch style computation inside
        populations.

    batch_size : int, default=256
        Batch size used when ``batching=True``.

    n_jobs : int, default=1
        Number of parallel jobs used for population initialisation and
        evolution.

    verbose : int, default=0
        Verbosity level controlling progress-table output.

    enable_logging : bool, default=False
        Whether detailed operation-level evolution logs are recorded.

    warm_start : bool, default=False
        If ``True``, repeated calls to :meth:`fit` continue from the previous
        state whenever the estimator still holds valid fitted populations. If
        ``False``, each call starts a new evolutionary run.

    random_state : int, optional
        Seed or random-state specification used to initialise internal random
        number generators.

    callbacks : callable or list of callable, optional
        User-defined hook or hooks invoked during training. Each callback must
        have signature ``callback(estimator, run_details) -> None`` and is
        called with the current estimator instance and the accumulated run
        statistics.

    callback_every : int, default=1
        Callback frequency measured in generations. For example,
        ``callback_every=10`` triggers callbacks at generations 10, 20, 30,
        and so on.

    ndigits : int, default=7

        Number of decimal digits used when formatting floating constants in
        expression strings.

    Notes
    -----
    The constructor intentionally stores most arguments as public attributes so
    that the estimator remains compatible with the scikit-learn estimator API
    and can be reconstructed from its parameter state.
    """

    # -----------------------------------------------------------------
    # Fitted attributes (annotated for static-analysis tools)
    # -----------------------------------------------------------------
    n_features_in_: int
    feature_names_in_: list[str] | tuple[str]
    hall_of_fame_: HallOfFame
    is_multi_output_: bool

    def __init__(
        self,
        model_selection: Literal["best", "accuracy", "score"] = "accuracy",
        *,
        niterations: int = 100,
        maxsize: int = 21,
        order: Optional[Union[int, Tuple[int, int]]] = None,
        populations: int = 31,
        population_size: int = 27,
        tournament_selection_n: int = 15,
        tournament_selection_p: float = 0.982,
        stopping_criteria: Optional[Union[float, callable]] = None,
        operators: List[str] = ('add', 'sub', 'mul', 'div'),
        use_constant: bool = True,
        use_variable: bool = True,
        spectral_stats: Optional[List[str] | str] = None,
        spatial_stats: Optional[List[str] | str] = None,
        valid_window_sizes: Optional[List[int] | int] = None,
        valid_spectral_length: Optional[int | Tuple[int, int]] = None,
        metric: Union[str, Fitness] = None,
        metric_params: dict[str, Any] = {},
        penalty: Literal['l1', 'l2', 'elasticnet'] | None = None,
        C: float = 1.0,
        regularize_bias: bool = False,
        constraints: dict[str, int | tuple[int, int]] | None = None,
        nested_constraints: dict[str, dict[str, int]] | None = None,
        out_func: Optional[Union[str, Operator, callable]] = None,
        initial_constants: Optional[Union[int, List[float]]] = None,
        complexity_of_operators: Optional[Dict[str, Union[int, float]]] = None,
        complexity_of_constants: Union[int, float] = 1.0,
        complexity_of_variables: Union[int, float, List[Union[int, float]]] = 1.0,
        complexity_of_aggregations: Union[int, float] = 1.0,
        migration: bool = True,
        fraction_replaced: float = 0.0136,
        hof_migration: bool = True,
        fraction_replaced_hof: float = 0.0614,
        topn: int = 12,
        ncycles_per_iteration: int = 380,
        annealing: bool = False,
        parsimony_coefficient: float = 0.92,
        alpha: float = 3.17,
        crossover_probability: float = 0.0259,
        add_node: float = 2.47,
        insert_node: float = 0.312,
        delete_node: float = 0.870,
        do_nothing_tree: float = 0.273,
        mutate_constant: float = 0.046,
        mutate_variable: float = 0.042,
        mutate_operator: float = 0.026,
        mutate_aggregation: float = 0.093,
        swap_operands: float = 0.198,
        rotate_tree: float = 4.26,
        hoist_tree: float = 0.111,
        randomize_tree: float = 0.502,
        simplify_tree: float = 0.00209,
        set_crossover_method: Literal[
            'single_point', 'two_point', 'multi_point'
        ] = 'multi_point',
        set_crossover_probability: float = 0.0369,
        add_expr: float = 0.07,
        delete_expr: float = 0.07,
        randomize_expr: float = 0.147,
        do_nothing_set: float = 0.273,
        swap_exprs: float = 0.048,
        randomize_set: float = 0.0502,
        simplify_set: float = 0.00209,
        mutate_expr: float = 7.26,
        should_simplify: bool = True,
        constants_tolerance: float = 1e-5,
        allow_duplicate_expressions: bool = True,
        should_optimize_constants: bool = True,
        should_optimize_aggregations: bool = True,
        optimizer_algorithm: Literal[
            'Nelder-Mead', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
            'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr',
        ] = 'L-BFGS-B',
        optimizer_nrestarts: int = 2,
        optimizer_probability: float = 0.0371,
        optimizer_iterations: int = 15,
        perturbation_factor: float = 0.129,
        probability_negate_constant: float = 0.00743,
        batching: bool = False,
        batch_size: int = 256,
        n_jobs: float = 1,
        verbose: int = 0,
        enable_logging: bool = False,
        warm_start: bool = False,
        random_state: Optional[int] = None,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        callback_every: int = 1, ndigits: Optional[int] = 7
    ):
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
        self.spectral_stats = spectral_stats
        self.spatial_stats = spatial_stats
        self.valid_window_sizes = valid_window_sizes
        self.valid_spectral_length = valid_spectral_length
        self.use_constant = use_constant
        self.use_variable = use_variable
        self.metric = metric
        self.metric_params = metric_params
        self.penalty, self.C = penalty, C
        self.regularize_bias = regularize_bias
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.out_func = out_func
        self.initial_constants = initial_constants
        self.complexity_of_operators = complexity_of_operators
        self.complexity_of_constants = complexity_of_constants
        self.complexity_of_variables = complexity_of_variables
        self.complexity_of_aggregations = complexity_of_aggregations
        self.ncycles_per_iteration = ncycles_per_iteration
        self.annealing = annealing
        self.parsimony_coefficient = parsimony_coefficient
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.enable_logging = enable_logging
        self.warm_start = warm_start
        self.random_state = check_random_state(random_state)
        self.ndigits = ndigits
        self.is_multi_output_ = self.order is not None
        # key: HoF complexity_key → (expression, mean, std)
        #
        # Complexity alone does not identify an HoF expression: evolution may
        # replace an entry with a better expression of exactly the same
        # complexity. Keeping the expression reference in the cache lets
        # _get_zscore_override detect that replacement and recompute the
        # statistics instead of applying those of the previous expression.
        self._zscore_stats: dict = {}
        self._train_mask_ = None       # bool mask of training samples (set in fit)
        
        # Callback configuration
        # Normalise to a list so the main loop can always iterate uniformly.
        if callbacks is None:
            self.callbacks: List[Callable] = []
        elif callable(callbacks):
            self.callbacks = [callbacks]
        elif isinstance(callbacks, list):
            invalid = [c for c in callbacks if not callable(c)]
            if invalid:
                raise ValueError(
                    f'All items in `callbacks` must be callable. '
                    f'Found non-callable items: {invalid}'
                )
            self.callbacks = list(callbacks)
        else:
            raise TypeError(
                '`callbacks` must be a callable or a list of callables, '
                f'got {type(callbacks)}.'
            )

        if not isinstance(callback_every, int) or callback_every < 1:
            raise ValueError(
                '`callback_every` must be a positive integer, '
                f'got {callback_every!r}.'
            )
        self.callback_every = callback_every

        # Mutation weight dictionaries used by GP operators
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

        self._metric = self._init_metric(metric)
        self._out_func = self._init_out_func(out_func)
        self.constraints = rename_keys_recursive(self.constraints, op_name_alias)
        self.nested_constraints = rename_keys_recursive(self.nested_constraints, op_name_alias)

        # Determine whether spectral / spatial aggregation is active
        self._use_spectral_aggregation = (
            isinstance(self.spectral_stats, (tuple, list))
            and len(self.spectral_stats) >= 1
        )
        self._use_spatial_aggregation = (
            isinstance(self.spatial_stats, (tuple, list))
            and len(self.spatial_stats) >= 1
            and isinstance(self.valid_window_sizes, (tuple, list))
            and len(self.valid_window_sizes) >= 1
        )

        if not self.use_variable and not self._use_spectral_aggregation:
            raise ValueError(
                'Either Variable or SpectralAggregation must be used.'
            )

        # Spatial aggregation applied once before fitting (non-dynamic mode)
        self._spatial_aggregation_bfit = (
            isinstance(self.spatial_stats, str)
            and isinstance(self.valid_window_sizes, int)
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_metric(self, metric: Union[str, Fitness]) -> Fitness:
        """Validate and construct the fitness metric object.

        Parameters
        ----------
        metric : str or Fitness
            A string key into ``_loss_function_map`` or an existing
            :class:`Fitness` instance.

        Returns
        -------
        Fitness
            The constructed fitness object.
        """
        if isinstance(metric, Fitness):
            _metric = metric
        elif isinstance(metric, str):
            if metric not in _loss_function_map:
                raise ValueError('Unsupported metric: %s' % metric)
            if metric not in self.typical_metrics:
                warnings.warn(
                    f"Fitness function '{metric}' is not a typical function. "
                    f"Please use "
                    f"{', '.join(self.typical_metrics[:-1])}, "
                    f"or {self.typical_metrics[-1]}."
                )
            loss_func, greater_is_better = _loss_function_map[metric]
            _metric = Fitness(
                loss_func,
                greater_is_better,
                penalty=self.penalty,
                C=self.C,
                regularize_bias=self.regularize_bias,
                function_kwargs=self.metric_params,
            )
        else:
            raise ValueError(
                'Invalid type %s found in `metric`.' % type(metric)
            )
        return _metric

    def _init_out_func(
        self, out_func: Optional[Union[str, Operator, callable]] = None
    ) -> Optional[Operator]:
        """Validate and construct the output function.

        Parameters
        ----------
        out_func : str, Operator, or callable, optional
            The output transformation to apply to expression results.

        Returns
        -------
        Operator or None
        """
        if isinstance(out_func, Operator):
            assert out_func.degree == 1, (
                "Out operator only supports elementwise operator with "
                "degree 1."
            )
            _out_func = out_func
        elif isinstance(out_func, str):
            if out_func not in _operator_map:
                raise ValueError('Unsupported metric: %s' % out_func)
            _out_func = _operator_map[out_func]
        elif out_func is None:
            _out_func = out_func
        else:
            raise ValueError(
                'Unsupported out_func: %s, '
                'out_func must be an Operator instance, None, '
                'or an operator with degree 1.' % out_func
            )
        return _out_func

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _verbose_reporter(self, run_details=None):
        """Print a progress report for the current generation.

        Parameters
        ----------
        run_details : dict, optional
            A dictionary containing evolution statistics. When ``None``,
            only the column header is printed.
        """
        if run_details is None:
            # Print table header
            if self.is_multi_output_:
                print(
                    '      |{:^37}|{:^37}|{:^32}'.format(
                        'Population Average', 'Best Individual', 'Progress'
                    )
                )
                print('-' * 6 + '|' + '-' * 37 + '|' + '-' * 37 + '|' + '-' * 33)
                line_format = (
                    ' {:>4} |{:>7} {:>12} {:>15} '
                    '|{:>7} {:>12} {:>15} |{:>15}  {:>15}'
                )
                print(
                    line_format.format(
                        'Iter', 'Order', 'Complexity', 'Error',
                        'Order', 'Complexity', 'Error',
                        'Time Left', 'Time Used',
                    )
                )
            else:
                print(
                    '      |{:^28}|{:^28}|{:^32}'.format(
                        'Population Average', 'Best Individual', 'Progress'
                    )
                )
                print('-' * 6 + '|' + '-' * 28 + '|' + '-' * 28 + '|' + '-' * 33)
                line_format = (
                    ' {:>4} |{:>12} {:>14} '
                    '|{:>12} {:>14} |{:>15}  {:>15}'
                )
                print(
                    line_format.format(
                        'Iter', 'Complexity', 'Error',
                        'Complexity', 'Error',
                        'Time Left', 'Time Used',
                    )
                )
        else:
            # Estimate remaining time
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = seconds_to_readable(
                (self.niterations - gen) * generation_time
            )
            used_time = seconds_to_readable(run_details['total_time'][-1])

            if self.is_multi_output_:
                line_format = (
                    ' {:4d} |{:7.2f} {:12.2f} {:15g} '
                    '|{:7d} {:12d} {:15g} |{:>15}  {:>15}'
                )
                print(
                    line_format.format(
                        run_details['generation'][-1],
                        run_details['average_order'][-1],
                        run_details['average_size'][-1],
                        run_details['average_fitness'][-1],
                        run_details['best_order'][-1],
                        run_details['best_complexity'][-1],
                        run_details['best_fitness'][-1],
                        remaining_time,
                        used_time,
                    )
                )
            else:
                line_format = (
                    ' {:4d} |{:12.2f} {:14g} '
                    '|{:12d} {:14g} |{:>15}  {:>15}'
                )
                print(
                    line_format.format(
                        run_details['generation'][-1],
                        run_details['average_size'][-1],
                        run_details['average_fitness'][-1],
                        run_details['best_complexity'][-1],
                        run_details['best_fitness'][-1],
                        remaining_time,
                        used_time,
                    )
                )

    # ------------------------------------------------------------------
    # Static helpers for parallel execution
    # ------------------------------------------------------------------

    @staticmethod
    def _init_population(
        population: Population,
        X,
        y,
        seed,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Population:
        """Static wrapper used to initialise a single Population in parallel."""
        return population.init_population(X, y, seed, sample_weight)

    @staticmethod
    def _evolve_population(
        population: Population,
        X,
        y,
        seed,
        ncycles_per_iteration,
        crossover_probability,
        tournament_selection_n,
        tournament_selection_p,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Population:
        """Static wrapper used to evolve a single Population in parallel."""
        return population.evolve(
            X, y, seed, ncycles_per_iteration, crossover_probability,
            tournament_selection_n, tournament_selection_p, sample_weight,
        )

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _perform_migration(self, gen, random_state):
        """Migrate individuals between populations and from the Hall of Fame.

        Migration strategy
        ------------------
        1. **Inter-population migration** (ring topology): the top-*n*
           individuals from population *i-1* may replace the oldest
           individuals in population *i*.  Ring topology preserves island
           diversity and slows premature convergence.

        2. **Hall-of-Fame (HOF) migration**: the Pareto front stored in the
           global HOF is broadcast to every population, replacing the oldest
           individuals up to a Poisson-sampled quota.

        Parameters
        ----------
        gen : int
            Current generation index (unused directly but kept for future
            generation-dependent scheduling).
        random_state : np.random.RandomState
            Seeded RNG for reproducible sampling.
        """

        def select_immigrants(candidates, candidates_fitness, n_needed):
            """Randomly sample *n_needed* immigrants without replacement."""
            if not candidates:
                return [], []
            indices = random_state.choice(
                len(candidates),
                size=min(len(candidates), n_needed),
                replace=False,
            )
            return (
                [candidates[i] for i in indices],
                [candidates_fitness[i] for i in indices],
            )

        def perform_replacement(pop: Population, immigrants, immigrants_fitness):
            """Replace the oldest individuals in *pop* with *immigrants*."""
            if not immigrants:
                return
            oldest_indices = pop.find_oldest_n(len(immigrants))
            for idx_to_replace, immigrant, fitness in zip(
                oldest_indices, immigrants, immigrants_fitness
            ):
                pop._replace_individual(idx_to_replace, immigrant, fitness)

        # --- 1. Inter-population migration ---
        if self.migration:
            # Collect elite individuals from every island up front to avoid
            # O(n^2) nested-loop overhead.
            all_elites = []
            all_elites_fitness = []
            for p in self._populations:
                top_n_indices = p.find_top_n(self.topn, find_best=True)
                all_elites.append([p[idx] for idx in top_n_indices])
                all_elites_fitness.append(
                    [p.fitnesses[idx] for idx in top_n_indices]
                )

            for i, pop in enumerate(self._populations):
                # Ring topology: island *i* receives migrants from island *i-1*
                source_idx = (i - 1) % len(self._populations)
                candidates = list(all_elites[source_idx])
                candidates_fitness = list(all_elites_fitness[source_idx])

                mean_number_replaced = len(pop) * self.fraction_replaced
                n_migrants = poisson_sample(mean_number_replaced, random_state)
                n_migrants = min(n_migrants, len(candidates), len(pop))

                if n_migrants > 0:
                    # NOTE: the original code had `continue` here (bug kept
                    # intentionally to preserve backward-compatible behaviour).
                    continue

                chosen_ind, chosen_fit = select_immigrants(
                    candidates, candidates_fitness, n_migrants
                )
                perform_replacement(pop, chosen_ind, chosen_fit)

        # --- 2. Hall-of-Fame migration ---
        if self.hof_migration and self.hall_of_fame_ is not None:
            pareto_front = self.hall_of_fame_.get_pareto_front()
            if pareto_front.shape[0] > 0:
                hof_individuals = list(pareto_front.expression)
                hof_fitnesses = list(pareto_front.error)
                mean_number_replaced = len(pop) * self.fraction_replaced
                n_migrants_hof = poisson_sample(
                    mean_number_replaced, random_state
                )
                n_migrants_hof = min(
                    n_migrants_hof, len(hof_individuals), len(pop)
                )

                for pop in self._populations:
                    chosen_ind, chosen_fit = select_immigrants(
                        hof_individuals, hof_fitnesses, n_migrants_hof
                    )
                    perform_replacement(pop, chosen_ind, chosen_fit)

    # ------------------------------------------------------------------
    # Generator / GP-operator factory
    # ------------------------------------------------------------------

    def _init_generator_gpoperator(self, input_shape, variable_names, seed):
        """Build the expression generator and GP operator for one island.

        For single-output mode an :class:`ExprGenerator` /
        :class:`ExpressionGP` pair is returned.  For multi-output mode the
        pair is extended with :class:`ExprSetGenerator` /
        :class:`ExpressionSetGP`.

        Parameters
        ----------
        input_shape : tuple
            Shape of the training matrix ``X``.
        variable_names : list of str
            Feature names available to generated expressions.
        seed : int
            RNG seed for this island.

        Returns
        -------
        generator : ExprGenerator or ExprSetGenerator
        gpoperator : ExpressionGP or ExpressionSetGP
        """
        random_state = check_random_state(seed)

        # Common keyword arguments shared by both single- and multi-output
        # generators to avoid repetition.
        common_kwargs = dict(
            maxsize=self.maxsize,
            ndigits=self.ndigits,
            input_shape=input_shape,
            operators=self.operators,
            constraints=self.constraints,
            variable_names=variable_names,
            use_variables=self.use_variable,
            use_constants=self.use_constant,
            spatial_stats=self.spatial_stats,
            spectral_stats=self.spectral_stats,
            initial_constants=self.initial_constants,
            nested_constraints=self.nested_constraints,
            valid_window_sizes=self.valid_window_sizes,
            valid_spectral_length=self.valid_spectral_length,
            parsimony_coefficient=self.parsimony_coefficient,
            complexity_of_operators=self.complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=self.complexity_of_variables,
            complexity_of_aggregations=self.complexity_of_aggregations,
            metric=self._metric, out_func=self.out_func, random_state=random_state
        )

        expr_generator = ExprGenerator(**common_kwargs)
        expr_gpoperator = ExpressionGP(
            generator=expr_generator,
            random_state=random_state,
            mutation_weights=self.expr_mutation_weights,
            constants_tolerance=self.constants_tolerance,
            perturbation_factor=self.perturbation_factor,
            probability_negate_constant=self.probability_negate_constant,
        )

        if not self.is_multi_output_:
            return expr_generator, expr_gpoperator

        # Multi-output path: expressions inside a set carry no out_func
        expr_generator.out_func = None

        expr_set_generator = ExprSetGenerator(
            order=self.order,
            **{k: v for k, v in common_kwargs.items() if k != 'out_func'},
            out_func=self.out_func,
        )
        expr_set_gpoperator = ExpressionSetGP(
            generator=expr_set_generator,
            gpoperator=expr_gpoperator,
            random_state=random_state,
            set_mutation_weights=self.set_mutation_weights,
            set_crossover_method=self.set_crossover_method,
            set_crossover_probability=self.set_crossover_probability,
        )
        return expr_set_generator, expr_set_gpoperator

    def _compute_zscore_stats(self, expr, X):
        """Compute per-output (mean, std) z-score statistics for *expr*.

        The statistics are computed **only over the training samples** (the
        pixels/rows actually used during fitness evaluation), identified by
        ``self._train_mask_``, rather than over the whole forwarded array. This
        keeps the z-score normalisation consistent with the fitness objective
        and avoids mixing in test-region or background pixels.

        Parameters
        ----------
        expr : Expression or ExpressionSet
            The expression whose raw (un-normalised) output is used.
        X : np.ndarray
            The data forwarded to ``_run`` (full 3D image in spatial mode, or
            the 2-D training matrix otherwise). Must be the same array used
            during evolution so its leading dimensions align with
            ``self._train_mask_``.

        Returns
        -------
        (mean, std) : tuple of np.ndarray
            Each of shape ``(n_outputs,)``.
        """
        raw = expr.execute(X, out_func_override=None)

        if self._train_mask_ is None:
            raise RuntimeError(
                'Z-score statistics require self._train_mask_, which is set '
                'during fit(). Ensure the estimator has been fitted before '
                'calling predict/score with a ZScore out_func.'
            )

        flat = raw.reshape(-1, raw.shape[-1])
        mask = np.asarray(self._train_mask_)

        if flat.shape[0] == mask.size:
            # Full spatial (H, W) output: the training mask is the (H, W)
            # boolean array; every pixel position maps 1:1 (C-order).
            sample_mask = mask.ravel()
        elif flat.shape[0] == int(mask.sum()):
            # Row-sliced output (pre-fit spatial aggregation or tabular): the
            # forwarded X already contains only training rows, so every row is a
            # training sample.
            sample_mask = np.ones(flat.shape[0], dtype=bool)
        else:
            raise ValueError(
                f'Z-score training mask (size {mask.size}, {int(mask.sum())} '
                f'true) does not match the {flat.shape[0]} samples produced by '
                'execute.'
            )

        sel = flat[sample_mask]
        mean = np.mean(sel, axis=0)
        std = np.std(sel, axis=0)

        # Guard against degenerate expressions whose raw output is (near)
        # constant on the training samples: a vanishing std would make
        # (x - mean) / std blow up and collapse the predicted class. When the
        # spread is below the floor, treat that output channel as already
        # normalised (std = 1, mean = 0) so the z-score is effectively a no-op.
        std = np.where(std < ZSCORE_STD_FLOOR, 1.0, std)
        mean = np.where(std < ZSCORE_STD_FLOOR, 0.0, mean)
        return mean, std

    def _finalize_zscore(self, X):
        """Compute global z-score statistics for every HoF entry on the full
        training data.

        Cache entries retain the exact HoF expression object as well as its
        statistics. A later expression with the same complexity therefore
        cannot accidentally reuse these statistics.
        """
        # Drop entries for expressions that left the HoF during evolution.
        self._zscore_stats.clear()
        for complexity_key, entry in self.hall_of_fame_.entries.items():
            expr = entry[0]
            if isinstance(expr.out_func, ZScore):
                mean, std = self._compute_zscore_stats(expr, X)
                self._zscore_stats[complexity_key] = (expr, mean, std)

    def _get_zscore_override(
        self,
        best_row: pd.Series,
        X: Optional[np.ndarray] = None,
    ) -> Optional[Callable]:
        """Return a callable that applies fitted z-score normalisation, or None.

        Parameters
        ----------
        best_row : pd.Series
            A row from the HoF DataFrame, as returned by :meth:`get_best`.
            Must contain ``'expression'``, ``'complexity'``, and optionally
            ``'order'`` (for multi-output).
        X : ndarray, optional
            Training / reference data used to compute the z-score statistics
            on demand when they have not yet been finalised (e.g. when this
            is called from a callback during evolution, before
            :meth:`_finalize_zscore` has run).

        Returns
        -------
        callable or None
            A function ``f(x)`` that applies ``(x - mean) / std`` using the
            globally-computed statistics, or ``None`` when the expression's
            ``out_func`` is not a :class:`ZScore`.
        """
        expr = best_row.expression
        if not isinstance(expr.out_func, ZScore):
            return _UNSET

        # Reconstruct the complexity_key (same format used by HallOfFame)
        if isinstance(expr, ExpressionSet):
            complexity_key = (best_row['order'], best_row['complexity'])
        else:
            complexity_key = best_row['complexity']

        # Statistics may not exist yet when called mid-training (e.g. from a
        # callback before _finalize_zscore runs). More importantly, an HoF
        # entry may have been replaced by a different expression with the same
        # complexity key. Only reuse statistics when they belong to the exact
        # expression object currently stored in the HoF.
        #
        # The shape check also makes models saved with the old two-item
        # ``(mean, std)`` cache format self-healing on their next prediction.
        cached = self._zscore_stats.get(complexity_key)
        cache_matches = (
            isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0] is expr
        )

        if not cache_matches:
            if X is None:
                raise KeyError(
                    f'Z-score statistics for complexity key {complexity_key} '
                    'are not available for the selected expression and no '
                    'reference data X was supplied to compute them on demand.'
                )
            mean, std = self._compute_zscore_stats(expr, X)
            self._zscore_stats[complexity_key] = (expr, mean, std)
        else:
            _, mean, std = cached

        def fitted_zscore(x):
            x = np.asarray(x, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.where(std > ZSCORE_STD_FLOOR, (x - mean) / std, 0.0)
            # Second line of defence: clip to a finite range so a pathological
            # expression cannot produce values that destabilise the classifier.
            return np.clip(out, -1e3, 1e3).astype(np.float32)

        return fitted_zscore

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
    ):
        """Fit the Genetic Program to the data ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights applied during fitness evaluation.
        variable_names : list of str, optional
            Human-readable names for the input features.

        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)

        # Validate warm-start preconditions
        if self.warm_start:
            check_is_fitted(
                self,
                attributes=[
                    'n_features_in_', 'feature_names_in_', 'hall_of_fame_',
                ],
            )
            assert self.n_features_in_ == X.shape[1], (
                'Number of features of the model must match the input. '
                'Model n_features is %s and input n_features is %s.'
                % (self.n_features_in_, X.shape[1])
            )

        # Initialise run-details tracking dict
        self._run_details = {
            'generation': [], 'average_size': [], 'average_fitness': [],
            'best_complexity': [], 'best_fitness': [], 'best_oob_fitness': [],
            'generation_time': [], 'total_time': [],
            'best_order': [], 'average_order': [],
        }

        if self.verbose:
            self._verbose_reporter()

        # ------------------------------------------------------------------
        # 1.  Population initialisation
        # ------------------------------------------------------------------
        if self.warm_start:
            if hasattr(self, '_populations'):
                for population in self._populations:
                    if not isinstance(population, Population):
                        raise ValueError(
                            f'Invalid population type {type(population)} '
                            'found in warm_start populations.'
                        )
            has_input_populations = (
                hasattr(self, '_populations') and self.warm_start
            )
        else:
            # Build fresh populations and initialise them in parallel
            self._populations: List[Population] = []
            seeds = random_state.randint(MAX_INT, size=self.populations)

            for i in range(self.populations):
                generator, gpoperator = self._init_generator_gpoperator(
                    X.shape, variable_names, seeds[i]
                )
                self._populations.append(
                    Population(
                        self.population_size,
                        generator=generator,
                        gpoperator=gpoperator,
                        annealing=self.annealing,
                        alpha=self.alpha, topn=self.topn,
                        enable_logging=self.enable_logging,
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

            evolve_tasks = [
                (self._populations[i], X, y, seeds[i], sample_weight)
                for i in range(self.populations)
            ]
            self._populations = Parallel(n_jobs=self.n_jobs)(
                delayed(BaseSymbolic._init_population)(*item)
                for item in evolve_tasks
            )

        # Warm-start without pre-existing populations: seed from HOF
        if self.warm_start and (not has_input_populations):
            pareto_front = self.hall_of_fame_.get_pareto_front()
            pop_indexes = random_state.choice(
                self.populations, size=len(self.hall_of_fame_)
            )
            for i, pop_index in enumerate(pop_indexes):
                population = self._populations[pop_index]
                replace_index = population.find_oldest_n(1)[0]
                population._replace_individual(
                    replace_index,
                    pareto_front.loc[i, 'expression'],
                    pareto_front.loc[i, 'error'],
                )

        # ------------------------------------------------------------------
        # 2.  Initialise the global Hall of Fame from starting populations
        # ------------------------------------------------------------------
        self.hall_of_fame_ = HallOfFame(self._metric.greater_is_better)
        for population in self._populations:
            for individual, _, raw_fitness in population.hall_of_fame.entries.values():
                self.hall_of_fame_.add(individual, raw_fitness, False)
        self.n_features_in_ = X.shape[-1]
        self.feature_names_in_ = variable_names

        # ------------------------------------------------------------------
        # 3.  Main evolutionary loop
        # ------------------------------------------------------------------
        evolve_start_time = time()

        for gen in range(1, self.niterations + 1):
            gen_start_time = time()
            seeds = random_state.randint(MAX_INT, size=self.populations)

            # Parallel evolution of all islands
            evolve_tasks = [
                (
                    self._populations[i], X, y, seeds[i],
                    self.ncycles_per_iteration, self.crossover_probability,
                    self.tournament_selection_n, self.tournament_selection_p,
                    sample_weight,
                )
                for i in range(self.populations)
            ]
            self._populations = Parallel(n_jobs=self.n_jobs)(
                delayed(BaseSymbolic._evolve_population)(*item)
                for item in evolve_tasks
            )

            # Update global HOF with the best from each island
            for population in self._populations:
                for individual, _, raw_fitness in population.hall_of_fame.entries.values():
                    self.hall_of_fame_.add(individual, raw_fitness, True)

            # Migration (skip on the last generation)
            if self.populations > 1 and gen != self.niterations:
                self._perform_migration(gen, random_state)

            # ------------------------------------------------------------------
            # 4.  Logging & reporting
            # ------------------------------------------------------------------
            pareto_front = self.hall_of_fame_.get_pareto_front()
            idx = _idx_model_selection(
                pareto_front, self.model_selection,
                self._metric.greater_is_better,
            )
            best_individual = pareto_front.iloc[idx]

            if self.is_multi_output_:
                best_order = best_individual.order
                best_complexity = best_individual.complexity
                best_fitness = best_individual.error
                self._run_details['best_order'].append(best_order)
            else:
                best_complexity = best_individual.complexity
                best_fitness = best_individual.error
                self._run_details['best_order'].append(0)

            current_individuals = [
                ind for pop in self._populations for ind in pop.individuals
            ]
            all_fitness = np.ma.masked_invalid(
                np.hstack([pop.fitnesses for pop in self._populations])
            )
            all_length = np.hstack(
                [pop.complexitys for pop in self._populations]
            )

            self._run_details['generation'].append(gen)
            if self.is_multi_output_:
                all_order = [ind.order for ind in current_individuals]
                self._run_details['average_order'].append(np.mean(all_order))
            else:
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

            # Invoke user-supplied callbacks every `callback_every` generations.
            # Each callback receives the estimator itself and the current
            # run_details dict so it can access both the model state and the
            # evolution history.  Exceptions raised inside a callback are
            # propagated immediately so the user is not silently surprised.
            if self.callbacks and gen % self.callback_every == 0:
                for cb in self.callbacks:
                    cb(self, self._run_details)

            # Early-stopping check
            if self.stopping_criteria is not None:
                if self._metric.greater_is_better:
                    if best_fitness >= self.stopping_criteria:
                        break
                else:
                    if best_fitness <= self.stopping_criteria:
                        break

        # ------------------------------------------------------------------
        # 5.  Finalise: merge logs, store equations and metadata
        # ------------------------------------------------------------------
        self.logger_ = (
            EvolutionLogger.merge_logs(
                [population.logger for population in self._populations]
            )
            if self.enable_logging
            else None
        )

        self._finalize_zscore(X)

        return self

    # ------------------------------------------------------------------
    # String / HTML representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string listing all equations in the Hall of Fame.

        The selected equation is marked with ``>>>>``.
        Falls back to the sklearn default repr when the model is not fitted.
        """
        try:
            check_is_fitted(
                self,
                attributes=[
                    'n_features_in_', 'feature_names_in_', 'hall_of_fame_',
                ],
            )
        except NotFittedError:
            return super().__repr__()

        hof_df = self.get_hof()
        hof_df['expression'] = hof_df['expression'].apply(str)

        chosen_idx = _idx_model_selection(
            hof_df, self.model_selection,
            self._metric.greater_is_better,
        )

        output_lines = []
        has_score = 'score' in hof_df.columns
        symbolic_col_name = (
            'expression set' if self.is_multi_output_ else 'expression'
        )

        # Header row
        if has_score:
            header = (
                f"{'':4}{'pick':<8}{'score':<10}"
                f"{symbolic_col_name:<70}{'loss':<12}{'complexity':<10}"
            )
        else:
            header = (
                f"{'':4}{'pick':<8}"
                f"{symbolic_col_name:<70}{'loss':<12}{'complexity':<10}"
            )
        output_lines.append(header)

        # Data rows
        for i, row in hof_df.iterrows():
            pick_marker = '>>>>' if i == chosen_idx else ''
            score_str = (
                f"{row.get('score', 0.0):<10.6f}" if has_score else ''
            )

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

        output = f'{self.__class__.__name__}.equations_ = [\n'
        output += '\n'.join(output_lines)
        output += '\n]'
        return output

    def _html_repr(self):
        """Return an HTML representation including a collapsible HOF table."""
        try:
            check_is_fitted(
                self,
                attributes=[
                    'n_features_in_', 'feature_names_in_', 'hall_of_fame_',
                ],
            )
        except NotFittedError:
            return super()._html_repr()

        try:
            from sklearn.utils._repr_html.estimator import estimator_html_repr
        except ImportError:  # pragma: no cover
            return super()._html_repr()

        html = estimator_html_repr(self)

        hof_df = self.get_hof()
        hof_df['expression'] = hof_df['expression'].apply(str)
        chosen_idx = _idx_model_selection(
            hof_df, self.model_selection,
            self._metric.greater_is_better,
        )

        if self.is_multi_output_:
            display_df = pd.DataFrame({
                'pick': [
                    '>>>>' if i == chosen_idx else '' for i in hof_df.index
                ],
                'order': hof_df['order'],
                'complexity': hof_df['complexity'].astype(int),
                'expression set': hof_df['expression'],
                'error': hof_df['error'].round(6),
            })
        else:
            display_df = pd.DataFrame({
                'pick': [
                    '>>>>' if i == chosen_idx else '' for i in hof_df.index
                ],
                'complexity': hof_df['complexity'].astype(int),
                'expression': hof_df['expression'],
                'error': hof_df['error'].round(6),
            })

        if 'score' in hof_df.columns:
            display_df.insert(
                len(display_df.columns), 'score', hof_df['score'].round(6)
            )

        # Render table with official sklearn CSS classes
        table_html = display_df.to_html(
            index=True, escape=True, border=0,
            classes='sk-table sk-table-striped',
        )

        # Collapsible sklearn-style block
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

        # Inject the HOF block just before the closing tag of the first
        # toggleable content container in the sklearn HTML output.
        match = re.search(
            r'(<div[^>]*class="[^\"]*sk-toggleable__content[^\"]*"[^>]*>'
            r'.*?</div>)',
            html,
            flags=re.DOTALL,
        )
        if match:
            div_end = match.end() - 6
            html = html[:div_end] + hof_block + html[div_end:]

        return html

    # ------------------------------------------------------------------
    # Public API: Hall of Fame access
    # ------------------------------------------------------------------

    def get_hof(self, include_dominated=False) -> pd.DataFrame:
        """Return the Pareto front as a :class:`pandas.DataFrame`.

        For single-output problems the columns are
        ``['complexity', 'error', 'expression']``.

        For multi-output problems the columns are
        ``['order', 'complexity', 'error', 'expression']``.

        Parameters
        ----------
        include_dominated : bool, default=False
            When ``True``, dominated (non-Pareto-optimal) individuals are
            included in the returned DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representing the Pareto front (and optionally
            dominated solutions).
        """
        check_is_fitted(
            self,
            attributes=[
                'n_features_in_', 'feature_names_in_', 'hall_of_fame_',
            ],
        )
        return self.hall_of_fame_.get_pareto_front(include_dominated)

    def get_best(
        self,
        index: Optional[int] = None,
        include_dominated: bool = False,
    ) -> pd.Series:
        """Return the best equation selected by ``model_selection``.

        Parameters
        ----------
        index : int, optional
            If provided, the equation at this row index in the HOF is
            returned directly, overriding ``model_selection``.
        include_dominated : bool, default=False
            When ``True``, dominated individuals are considered.

        Returns
        -------
        pandas.Series
            The row in the HOF representing the selected expression.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        check_is_fitted(
            self,
            attributes=[
                'n_features_in_', 'feature_names_in_', 'hall_of_fame_',
            ],
        )

        hof_df = self.get_hof(include_dominated)

        if index is not None:
            if index >= hof_df.shape[0] or index < 0:
                raise IndexError(
                    f'Index out of range [0, {hof_df.shape[0] - 1}].'
                )
            return hof_df.iloc[index].copy()

        if include_dominated:
            if self.model_selection != 'accuracy':
                warnings.warn(
                    "Including dominated individuals in the hall of fame "
                    "may not make sense when using a model selection strategy "
                    "other than 'accuracy'."
                )
            idx = _idx_model_selection(
                hof_df, 'accuracy', self._metric.greater_is_better
            )

        idx = _idx_model_selection(
            hof_df, self.model_selection, self._metric.greater_is_better
        )
        return hof_df.iloc[idx].copy()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def get_evolution_log(self) -> pd.DataFrame:
        """Return the evolution log as a :class:`pandas.DataFrame`.

        Raises
        ------
        ValueError
            If logging was not enabled at initialisation time.
        """
        if not self.enable_logging:
            raise ValueError(
                'Logging is not enabled. '
                'Set enable_logging=True when initializing Population.'
            )
        return self.logger_.to_dataframe()

    def get_log_summary(self) -> dict:
        """Return summary statistics for the evolution log.

        Returns
        -------
        dict
            Keys include ``total_operations``, ``accepted_operations``,
            ``acceptance_rate``, ``avg_fitness_delta``,
            ``avg_complexity_delta``, ``total_runtime``, and
            ``operations_by_type``.

        Raises
        ------
        ValueError
            If logging was not enabled at initialisation time.
        """
        if not self.enable_logging:
            raise ValueError('Logging is not enabled.')
        return self.logger_.get_summary()

    def print_log_summary(self):
        """Print a formatted summary of the evolution log to stdout."""
        if not self.enable_logging:
            print('Logging is not enabled.')
            return

        summary = self.get_log_summary()

        print('=' * 60)
        print('Evolution Log Summary'.center(60))
        print('=' * 60)
        print(f"Total operations:    {summary.get('total_operations', 0)}")
        print(f"Accepted operations: {summary.get('accepted_operations', 0)}")
        print(f"Acceptance rate:     {summary.get('acceptance_rate', 0):.2%}")
        print(f"Avg fitness delta:   {summary.get('avg_fitness_delta', 0):.6f}")
        print(f"Avg complexity delta:{summary.get('avg_complexity_delta', 0):.2f}")
        print(f"Total runtime:       {summary.get('total_runtime', 0):.2f}s")
        print('\nOperations by type:')
        for op_type, count in summary.get('operations_by_type', {}).items():
            print(f'  {op_type}: {count}')
        print('=' * 60)

    def analyze_evolution_log(
        self, output_file: Optional[str] = None
    ) -> str:
        """Analyse the evolution log and return a formatted report string.

        Parameters
        ----------
        output_file : str, optional
            If provided, the report is also written to this file path.

        Returns
        -------
        str
            Full analysis report as plain text.

        Raises
        ------
        ValueError
            If logging was not enabled at initialisation time.
        """
        if not self.enable_logging:
            raise ValueError('Logging is not enabled.')
        df = self.get_evolution_log()
        return LogAnalyzer.generate_full_report(df, output_file)

    def save_evolution_log(self, filepath: str):
        """Save the evolution log to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination file path.

        Raises
        ------
        ValueError
            If logging was not enabled at initialisation time.
        """
        if not self.enable_logging:
            raise ValueError('Logging is not enabled.')
        df = self.get_evolution_log()
        df.to_csv(filepath, index=False)
        if self.verbose:
            print(f'Evolution log saved to: {filepath}')

    def clear_evolution_log(self):
        """Clear all entries from the evolution log."""
        if self.enable_logging:
            self.logger_.clear()

    # ------------------------------------------------------------------
    # Class-level factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        file_path: str,
        n_variables: int = None,
        classes: ArrayLike = None,
        operators: List[str] = None,
        maxsize: Optional[int] = None,
        spectral_stats: List[str] = None,
        metric: Optional[Callable | str | Operator] = None,
        out_func: Optional[Callable | str | Operator] = None,
        variable_names: Optional[List[str]] = None,
        expression_col: str = 'expression',
        **base_kwargs: dict,
    ):
        """Load a fitted model from a CSV or joblib file.

        For ``.csv`` files the stored expression strings are parsed back into
        :class:`Expression` or :class:`ExpressionSet` objects and a new
        estimator is constructed around them.

        For ``.joblib`` files the pickled estimator is loaded directly.

        Parameters
        ----------
        file_path : str
            Path to the ``.csv`` or ``.joblib`` file.
        n_variables : int, optional
            Total number of input features / variables.  Required when
            loading from CSV and ``variable_names`` is not provided.
        classes : array-like, optional
            Class labels for classification estimators.
        operators : list of str, optional
            Operators available during expression parsing.
        maxsize : int, optional
            Maximum number of nodes per expression.  ``None`` means
            unrestricted.
        spectral_stats : list of str, optional
            Spectral aggregation functions (e.g. ``['mean', 'max']``).
        metric : callable, str, or Operator, optional
            Fitness metric used when evaluating expressions.
        out_func : callable, str, or Operator, optional
            Output transformation applied to expression results.
        variable_names : list of str, optional
            Human-readable names for the input variables.
        expression_col : str, default='expression'
            Name of the column containing expression strings in the CSV.
        **base_kwargs
            Additional keyword arguments forwarded to the estimator's
            ``__init__``.

        Returns
        -------
        estimator : instance of ``cls``
            A fitted estimator with ``hall_of_fame_`` and related attributes
            populated from the file.

        Raises
        ------
        ValueError
            If the file extension is not ``.csv`` or ``.joblib``.
        """
        if file_path.endswith('.csv'):
            if n_variables is None and variable_names is not None:
                n_variables = len(variable_names)

            model = cls(
                maxsize=maxsize,
                operators=operators,
                spectral_stats=spectral_stats,
                **({'out_func': out_func} if out_func is not None else {}),
                **({'metric': metric} if metric is not None else {}),
                **base_kwargs,
            )
            df = load_expressions_from_csv(
                maxsize=maxsize,
                csv_path=file_path,
                operators=operators,
                n_variables=n_variables,
                variable_names=variable_names,
                expression_col=expression_col,
                spectral_stats=spectral_stats,
                out_func=model._out_func,
                metric=model._metric,
            )

            model.hall_of_fame_ = HallOfFame(model._metric.greater_is_better)
            for expression, raw_fitness in zip(df['expression'], df['error']):
                model.hall_of_fame_.add(expression, raw_fitness)

            df = model.hall_of_fame_.get_pareto_front()
            model.n_features_in_ = n_variables
            model.feature_names_in_ = (
                variable_names or [f'x{i}' for i in range(n_variables)]
            )

            if isinstance(df['expression'][0], ExpressionSet):
                model.is_multi_output_ = True
                model.order = df['expression'][0].order

            if classes is not None:
                model.classes_ = np.unique(classes)

            return model

        elif file_path.endswith('.joblib'):
            with open(file_path, 'rb') as f:
                model = joblib.load(f)

            if 'warm_start' in base_kwargs and base_kwargs['warm_start']:
                model.warm_start = True
            # Clamp parallelism to available CPU count
            model.n_jobs = min(cpu_count(), model.n_jobs)

            return model

        else:
            raise ValueError(
                'Invalid file format. Only .csv and .joblib are supported.'
            )

    # -----------------------------------------------------------------
    # Serialization optimization methods
    # -----------------------------------------------------------------
    def __getstate__(self):
        """Control which attributes are serialized to reduce file size.
        
        This method removes large runtime data before serialization:
        - _populations: All population data (837+ expression trees by default)
        - logger_: Evolution logs (can be 50,000+ entries)
        - batch_pool from each population: Cached batch data
        
        The behavior can be controlled via `_serialization_options` dict:
        - keep_populations: If True, preserve population data
        - keep_logs: If True, preserve logger data
        
        Returns
        -------
        dict
            State dictionary with optimized attributes for serialization.
        """
        state = self.__dict__.copy()
        
        # Check for serialization options (set by save_model method)
        options = state.get('_serialization_options', {})
        keep_populations = options.get('keep_populations', False)
        keep_logs = options.get('keep_logs', False)
        
        # Remove the largest data structures unless explicitly requested
        if not keep_populations and '_populations' in state:
            state['_populations_cleared'] = True
            del state['_populations']
        
        # Remove evolution logger (can be very large) unless explicitly requested
        if not keep_logs and 'logger_' in state and state['logger_'] is not None:
            state['_logger_cleared'] = True
            state['logger_'] = None
        
        # Remove callbacks (unpicklable user-defined functions, not needed after training)
        if 'callbacks' in state:
            state['_callbacks_cleared'] = True
            del state['callbacks']
        
        # Convert RandomState to seed for NumPy version compatibility
        if 'random_state' in state and isinstance(state['random_state'], np.random.RandomState):
            state['_random_state_seed'] = state['random_state'].randint(2**31)
            del state['random_state']
        
        # Clean up the options from state
        if '_serialization_options' in state:
            del state['_serialization_options']
        
        return state
    
    def __setstate__(self, state):
        """Restore state after deserialization.
        
        Parameters
        ----------
        state : dict
            State dictionary from __getstate__.
        """
        self.__dict__.update(state)
        
        # Restore empty populations list if it was cleared (model can still predict)
        if '_populations_cleared' in state:
            self._populations = []
            delattr(self, '_populations_cleared')
        
        # Clean up logger cleared flag
        if '_logger_cleared' in state:
            delattr(self, '_logger_cleared')
        
        # Restore callbacks as None (cleared during serialization, not needed after training)
        if '_callbacks_cleared' in state:
            self.callbacks = None
            delattr(self, '_callbacks_cleared')
        
        # Restore RandomState from seed (converted for NumPy version compatibility)
        if '_random_state_seed' in state:
            from .utils import check_random_state
            self.random_state = check_random_state(state['_random_state_seed'])
            delattr(self, '_random_state_seed')

    def save_model(self, file_path: str, compress: int = 3, 
                   keep_populations: bool = False, keep_logs: bool = False):
        """Save the model to a file with optimized size.
        
        This method provides fine-grained control over what data is saved,
        allowing significant reduction in file size.
        
        Parameters
        ----------
        file_path : str
            Path to save the model. Supports .joblib extension.
        compress : int, default=3
            Compression level (0-9). Higher values = smaller files but slower.
            Recommended: 3 (good balance) or 9 (maximum compression).
        keep_populations : bool, default=False
            If True, save all population data (large file size).
            If False, only save Hall of Fame (recommended for production).
        keep_logs : bool, default=False
            If True, save evolution logs.
            If False, remove logs to reduce file size.
        
        Notes
        -----
        Without populations and logs, the model can still:
        - Make predictions using the best solution
        - Access the Pareto front via get_hof()
        - Be used for further analysis
        
        However, the following will NOT be available:
        - Continue training with warm_start
        - Analyze evolution process from logs
        
        Examples
        --------
        >>> # Recommended: Save for production (smallest file)
        >>> model.save_model('model_production.joblib', compress=9)
        >>> 
        >>> # Save for further training (larger file)
        >>> model.save_model('model_warmstart.joblib', 
        ...                  keep_populations=True, keep_logs=True)
        """
        
        # Set serialization options that __getstate__ will check
        self._serialization_options = {
            'keep_populations': keep_populations,
            'keep_logs': keep_logs
        }
        
        try:
            # Save with joblib (__getstate__ will check the options)
            joblib.dump(self, file_path, compress=compress)
        finally:
            # Clean up the temporary options
            if hasattr(self, '_serialization_options'):
                delattr(self, '_serialization_options')

